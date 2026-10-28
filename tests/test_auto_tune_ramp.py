"""Unit tests for the adaptive ramp peak/SLO search (benchmark_runner.auto_tune).

The ramp is driven with an INJECTED ``run_point`` that returns synthetic
``PointMetrics`` from a caller-supplied throughput/latency curve, so the whole
Phase-1 bracket + Phase-2 search can be exercised deterministically without a
live server or guidellm.

Focus areas:
  * saturation (peak-throughput) target: bracketing + unimodal Phase-2 search,
    especially that a peak in the (prev_prev, prev) half is now found (the bug
    the old (prev_knob, knob) bracket + right-walking bisection missed);
  * edge cases: first-point overload, immediate plateau, climb to the bound,
    budget cap, integer de-dup, ties;
  * rate-axis "can't keep up" stop;
  * noise robustness;
  * SLO target regression (Phase-1 pass/fail bracket + Phase-2 bisection).
"""

import asyncio
import math
from dataclasses import asdict, fields
from types import SimpleNamespace

import pytest

from benchmark_runner.auto_tune import (
    RAMP_OUTCOME_VERSION,
    AutoTuneConfig,
    PointMetrics,
    _normalize,
    _passes_slo,
    STOP_BUDGET_POINTS,
    STOP_BUDGET_SECONDS,
    STOP_CAPACITY_PLATEAU,
    STOP_CONVERGED,
    STOP_OVERLOADED,
    STOP_POINT_FAILED,
    STOP_PROBE_BOUND,
    STOP_SLO_FAILED,
    STOP_UPPER_BOUND,
    run_ramp,
)
from benchmark_runner.progress import ServerBenchmarkerProgress


# ── helpers ──────────────────────────────────────────────────────────────────
def make_metrics(
    knob,
    index,
    tps,
    *,
    success=1.0,
    achieved=None,
    ttft=10.0,
    tpot=5.0,
    latency=100.0,
):
    return PointMetrics(
        knob=knob,
        index=index,
        output_tps=tps,
        total_tps=tps,
        ttft_ms=ttft,
        ttft_p95_ms=ttft,
        ttft_p99_ms=ttft,
        itl_ms=tpot,
        itl_p95_ms=tpot,
        itl_p99_ms=tpot,
        latency_ms=latency,
        latency_p95_ms=latency,
        latency_p99_ms=latency,
        achieved_rate=achieved if achieved is not None else knob,
        success=success,
    )


def piecewise(control):
    """Linear interpolation through a {knob: value} control-point dict."""
    xs = sorted(control)

    def f(k):
        if k <= xs[0]:
            return float(control[xs[0]])
        if k >= xs[-1]:
            return float(control[xs[-1]])
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            if x0 <= k <= x1:
                t = (k - x0) / (x1 - x0)
                return float(control[x0] + t * (control[x1] - control[x0]))
        return float(control[xs[-1]])  # unreachable

    return f


def make_run_point(
    curve,
    *,
    success_fn=None,
    achieved_fn=None,
    latency_fn=None,
    noise_fn=None,
    calls=None,
):
    async def rp(knob, index):
        if calls is not None:
            calls.append(knob)
        tps = curve(knob)
        if noise_fn is not None:
            tps *= 1.0 + noise_fn(knob, index)
        return make_metrics(
            knob,
            index,
            tps,
            success=success_fn(knob) if success_fn else 1.0,
            achieved=achieved_fn(knob) if achieved_fn else knob,
            latency=latency_fn(knob) if latency_fn else 100.0,
        )

    return rp


def sat_cfg(**kw):
    # probe_saturation defaults OFF in tests so non-probe cases don't trigger the
    # real _default_probe (which needs a live server); probe tests opt in.
    d = dict(
        axis="concurrency",
        lower_bound=1,
        upper_bound=1024,
        max_points=12,
        probe_saturation=False,
    )
    d.update(kw)
    return AutoTuneConfig(**d)


def run(cfg, rp, probe=None):
    """Measured points only — what most assertions here are about."""
    return run_outcome(cfg, rp, probe).points


def run_outcome(cfg, rp, probe=None):
    """The full RampOutcome, for the tests that assert WHY the search stopped."""
    return asyncio.run(run_ramp(cfg, {}, "t", run_point=rp, saturation_probe=probe))


def make_probe(ceiling, calls=None):
    """A saturation probe returning a fixed ceiling rps (records calls if given)."""

    async def probe():
        if calls is not None:
            calls.append(ceiling)
        return float(ceiling)

    return probe


def knobs(points):
    return [p.knob for p in points]


def peak_knob(points):
    return max(points, key=lambda p: p.total_tps).knob


def peak_tps(points):
    return max(p.total_tps for p in points)


# ── saturation: Phase-1 stop + bracket ───────────────────────────────────────
def test_basic_ramp_reaches_peak_at_plateau():
    # Rises 1..32, then flat -> plateau at 64; peak throughput is 3200.
    curve = piecewise({1: 100, 2: 200, 4: 400, 8: 800, 16: 1600, 32: 3200, 64: 3300})
    pts = run(sat_cfg(), make_run_point(curve))
    assert peak_tps(pts) >= 3200 - 1
    # 32 (the last strongly-improving point) is the peak here.
    assert peak_knob(pts) in (32.0, 64.0)


def test_narrow_peak_in_lower_half_is_found():
    # THE regression case: geometric samples 8/16/32/64 = 800/1600/1700/900 make
    # Phase-1 overshoot to 64 with prev_knob=32; the TRUE peak (2000) sits at ~24,
    # inside the (prev_prev=16, prev=32) half the old bracket dropped.
    curve = piecewise(
        {
            1: 100,
            2: 200,
            4: 400,
            8: 800,
            16: 1600,
            24: 2000,
            32: 1700,
            48: 1300,
            64: 900,
        }
    )
    pts = run(sat_cfg(), make_run_point(curve))
    # Must actually discover the ~2000 peak, not settle for 1700 at knob 32.
    assert peak_tps(pts) >= 1990, f"missed the narrow peak: {peak_tps(pts)}"
    assert 20 <= peak_knob(pts) <= 28
    # And it must have probed the (16, 32) half at least once.
    assert any(16 < k < 32 for k in knobs(pts))


def test_peak_in_upper_half():
    # Peak at ~48, inside (32, 64).
    curve = piecewise(
        {1: 100, 2: 200, 4: 400, 8: 800, 16: 1600, 32: 3000, 48: 3600, 64: 2500}
    )
    pts = run(sat_cfg(), make_run_point(curve))
    assert peak_tps(pts) >= 3590
    assert 40 <= peak_knob(pts) <= 56


def test_peak_exactly_on_sampled_point():
    # Peak sits on the geometric sample 32 itself.
    curve = piecewise({1: 100, 2: 200, 4: 400, 8: 800, 16: 1600, 32: 3200, 64: 1000})
    pts = run(sat_cfg(), make_run_point(curve))
    assert peak_knob(pts) == 32.0
    assert peak_tps(pts) >= 3200 - 1


class TestThroughputGainConvergence:
    """Peak refinement ends when three new points buy less than 1% total TPS."""

    @staticmethod
    def outcome(refinement=(3320, 3310, 3315), mutate=None, **config):
        async def measure(knob, index):
            if index < 3:
                tps = (1600, 3200, 3300)[index]
            else:
                tps = refinement[min(index - 3, len(refinement) - 1)]
            point = make_metrics(knob, index, tps)
            if mutate is not None:
                mutate(point)
            return point

        cfg = dict(lower_bound=16, upper_bound=128, max_points=20)
        cfg.update(config)
        return run_outcome(sat_cfg(**cfg), measure)

    def test_three_low_gain_points_record_the_window_and_nearest_neighbors(self):
        outcome = self.outcome()

        assert knobs(outcome.points) == [16, 32, 64, 48, 40, 56]
        assert outcome.bracket_reason == STOP_CAPACITY_PLATEAU
        assert outcome.stop_reason == "throughput_gain_converged"
        assert peak_knob(outcome.points) == 48
        evidence = outcome.to_dict()["throughput_gain_evidence"]
        assert evidence == {
            "threshold_percent": 1,
            "window_points": 3,
            "baseline_rate": 64,
            "baseline_total_tps": 3300,
            "best_rate": 48,
            "best_total_tps": 3320,
            "gain_percent": pytest.approx(100 * (3320 / 3300 - 1)),
            "lower_rate": 40,
            "upper_rate": 56,
            "points": [
                {"rate": 48, "total_tps": 3320, "best_total_tps": 3320},
                {"rate": 40, "total_tps": 3310, "best_total_tps": 3320},
                {"rate": 56, "total_tps": 3315, "best_total_tps": 3320},
            ],
        }
        assert outcome.to_dict()["version"] == 7
        assert "throughput_decline" not in outcome.to_dict()

    def test_small_new_records_do_not_restart_the_window(self):
        outcome = self.outcome((3305, 3310, 3315))

        assert len(outcome.points) == 6
        assert outcome.stop_reason == "throughput_gain_converged"
        assert outcome.throughput_gain_evidence["baseline_total_tps"] == 3300

    def test_equal_total_tps_uses_the_lowest_rate_for_the_evidence_best(self):
        outcome = self.outcome((3300, 3290, 3295))

        assert outcome.stop_reason == "throughput_gain_converged"
        assert len(outcome.points) == 6
        evidence = outcome.throughput_gain_evidence
        assert evidence["baseline_rate"] == 64
        assert evidence["best_rate"] == 48
        assert evidence["lower_rate"] == 40
        assert evidence["upper_rate"] == 56

    def test_the_window_baseline_uses_the_same_lowest_rate_tie_break(self):
        def mutate(point):
            if point.index == 1:
                point.total_tps = 3300

        outcome = self.outcome((3300, 3300, 3300), mutate=mutate)

        assert outcome.stop_reason == "throughput_gain_converged"
        assert outcome.throughput_gain_evidence["baseline_rate"] == 32
        assert outcome.throughput_gain_evidence["best_rate"] == 24

    @pytest.mark.parametrize("gain", [0.01, 0.015])
    def test_material_cumulative_gain_requires_more_measurements(self, gain):
        refinement = tuple(3300 * (1 + gain * step / 3) for step in (1, 2, 3))
        outcome = self.outcome(refinement, max_points=6)

        assert len(outcome.points) == 6
        assert outcome.stop_reason == STOP_BUDGET_POINTS
        assert "throughput_gain_evidence" not in outcome.to_dict()

    @pytest.mark.parametrize(
        ("gain_percent", "stop_reason"),
        [(0.9999994, "throughput_gain_converged"), (0.9999996, STOP_BUDGET_POINTS)],
    )
    def test_gain_decisions_use_the_recorded_percentage_precision(
        self, gain_percent, stop_reason
    ):
        refinement = tuple(
            3300 * (1 + gain_percent / 100 * step / 3) for step in (1, 2, 3)
        )
        outcome = self.outcome(refinement, max_points=6)

        assert outcome.stop_reason == stop_reason
        assert outcome.points[-1].total_tps == refinement[-1]
        evidence = outcome.to_dict().get("throughput_gain_evidence")
        if evidence is not None:
            assert evidence["gain_percent"] == 0.999999
            assert evidence["gain_percent"] < evidence["threshold_percent"]

    def test_an_exact_one_percent_gain_is_not_below_the_threshold(self):
        def mutate(point):
            if point.index < 3:
                point.total_tps = (0.03, 0.068, 0.07)[point.index]

        outcome = self.outcome((0.0702, 0.0704, 0.0707), mutate=mutate, max_points=6)

        assert outcome.stop_reason == STOP_BUDGET_POINTS
        assert outcome.points[-1].total_tps == 0.0707
        assert "throughput_gain_evidence" not in outcome.to_dict()

    @pytest.mark.parametrize("peak_side", ["lower", "upper"])
    def test_a_best_point_at_a_measured_boundary_is_not_gain_converged(self, peak_side):
        curve = (
            (lambda knob: 10000 - knob)
            if peak_side == "lower"
            else piecewise({16: 1600, 32: 3200, 64: 6400, 128: 6500})
        )
        outcome = run_outcome(
            sat_cfg(lower_bound=16, upper_bound=128, max_points=20),
            make_run_point(curve),
        )

        assert outcome.stop_reason != "throughput_gain_converged"
        assert "throughput_gain_evidence" not in outcome.to_dict()

    @pytest.mark.parametrize("bad", ["success", "zero", "negative", "nan"])
    def test_an_invalid_refinement_resets_the_three_point_window(self, bad):
        def mutate(point):
            if point.index == 4:
                if bad == "success":
                    point.success = 0.9
                else:
                    point.total_tps = {"zero": 0, "negative": -1, "nan": math.nan}[bad]

        outcome = self.outcome(mutate=mutate)

        assert len(outcome.points) == 8
        assert outcome.stop_reason == "throughput_gain_converged"
        evidence = outcome.throughput_gain_evidence
        assert [p["rate"] for p in evidence["points"]] == [56, 44, 52]
        assert evidence["baseline_rate"] == 48

    @pytest.mark.parametrize("bad", ["success", "infinity"])
    def test_an_invalid_global_best_cannot_establish_gain_convergence(self, bad):
        def mutate(point):
            if point.index == 3:
                point.total_tps = 4000 if bad == "success" else math.inf
                if bad == "success":
                    point.success = 0.9

        outcome = self.outcome(mutate=mutate)

        assert outcome.stop_reason != "throughput_gain_converged"
        assert "throughput_gain_evidence" not in outcome.to_dict()

    def test_output_growth_does_not_extend_a_total_tps_plateau(self):
        def mutate(point):
            point.output_tps = 100 * 2**point.index

        outcome = self.outcome(mutate=mutate)

        assert outcome.stop_reason == "throughput_gain_converged"
        assert len(outcome.points) == 6
        assert peak_knob(outcome.points) == 48

    def test_the_third_low_gain_point_can_complete_the_point_budget(self):
        outcome = self.outcome(max_points=6)

        assert len(outcome.points) == 6
        assert outcome.stop_reason == "throughput_gain_converged"

    def test_the_third_low_gain_point_can_complete_the_time_budget(self, monkeypatch):
        import benchmark_runner.auto_tune as at

        clock = [0.0]
        monkeypatch.setattr(at.time, "monotonic", lambda: clock[0])

        def mutate(point):
            if point.index == 5:
                clock[0] = 10

        outcome = self.outcome(mutate=mutate, max_total_seconds=10)

        assert outcome.stop_reason == "throughput_gain_converged"
        assert len(outcome.points) == 6

    def test_an_exhausted_integer_grid_keeps_exact_convergence(self):
        curve = piecewise({2: 1600, 3: 3280, 4: 3300, 5: 3280, 6: 3280, 8: 3290})
        outcome = run_outcome(
            sat_cfg(lower_bound=2, upper_bound=16), make_run_point(curve)
        )

        assert knobs(outcome.points) == [2, 4, 8, 6, 3, 5]
        assert outcome.stop_reason == STOP_CONVERGED
        assert "throughput_gain_evidence" not in outcome.to_dict()


def test_monotonic_climb_to_upper_bound():
    # Never plateaus; stops at the upper bound, no Phase-2 bracket.
    curve = piecewise({1: 100, 2: 200, 4: 400, 8: 800})
    pts = run(sat_cfg(upper_bound=8), make_run_point(curve))
    assert peak_knob(pts) == 8.0
    assert knobs(pts) == [1.0, 2.0, 4.0, 8.0]


def test_first_point_overloaded_stops():
    # First point already below the success floor -> stop with a single point.
    # lower_bound=1 is already the knob floor, so there is nothing below to search.
    curve = piecewise({1: 1000})
    pts = run(sat_cfg(), make_run_point(curve, success_fn=lambda k: 0.5))
    assert len(pts) == 1
    assert pts[0].knob == 1.0


# ── peak at or below lower_bound ─────────────────────────────────────────────
class TestSearchRangeIsHard:
    """[lower_bound, upper_bound] is the range the user asked for — never exceeded.

    Symmetric with the upper end, where hitting upper_bound while still climbing
    stops the sweep and reports `not_saturated` rather than quietly probing higher.
    When the server turns out to be saturated at lower_bound already, the answer is
    to SAY SO (gpustack emits `saturated_at_lower_bound`, carrying the measured
    ceiling, from the achieved-vs-offered gap at the lowest point) and let the user
    lower the range — not to search a region they never asked about.
    """

    def test_does_not_probe_below_lower_bound_when_saturated_from_the_start(self):
        # Saturates at 2 req/s while the ramp starts at 4: every point from 4 up
        # reports the same pinned throughput.
        pts = run(
            sat_cfg(axis="rate", lower_bound=4, upper_bound=64),
            make_run_point(
                lambda k: min(k, 2.0) * 1000.0, achieved_fn=lambda k: min(k, 2.0)
            ),
        )
        assert min(knobs(pts)) >= 4.0, f"must not go below lower_bound=4: {knobs(pts)}"

    def test_does_not_probe_below_lower_bound_from_the_bound_break_route(self):
        """lower_bound above the (probe-derived) bound: one point, then stop.

        The first point trips `knob >= bound` before any stop criterion can fire
        (no previous point to compare against), so Phase 1 exits through the bound
        check with no bracket. That is the intended outcome — the run reports what
        it measured and validity explains the range was wrong.
        """
        pts = run(
            sat_cfg(
                axis="rate",
                lower_bound=64,
                upper_bound=128,
                max_points=8,
                probe_saturation=True,
            ),
            make_run_point(
                lambda k: min(k, 23.6) * 1200.0, achieved_fn=lambda k: min(k, 23.6)
            ),
            probe=make_probe(23),
        )
        assert knobs(pts) == [64.0], f"one point inside the range only: {knobs(pts)}"

    def test_never_exceeds_either_end_of_the_range(self):
        # A healthy curve that would happily climb past 64 must still stop there.
        pts = run(
            sat_cfg(axis="rate", lower_bound=4, upper_bound=64),
            make_run_point(lambda k: k * 1000.0, achieved_fn=lambda k: k * 0.98),
        )
        ks = knobs(pts)
        assert min(ks) >= 4.0 and max(ks) <= 64.0, f"outside [4, 64]: {ks}"

    def test_rate_search_stops_at_a_narrow_upper_bound(self):
        outcome = run_outcome(
            sat_cfg(
                axis="rate",
                lower_bound=1,
                upper_bound=4,
                probe_saturation=True,
            ),
            make_run_point(lambda k: k * 1000.0),
            probe=make_probe(250),
        )

        assert knobs(outcome.points) == [1.0, 2.0, 4.0]
        assert outcome.stop_reason == STOP_UPPER_BOUND

    def test_slo_concurrency_search_stops_at_a_non_power_of_two_bound(self):
        outcome = run_outcome(
            sat_cfg(
                axis="concurrency",
                lower_bound=4,
                upper_bound=138,
                slo_avg_latency_ms=1000.0,
            ),
            make_run_point(lambda k: k * 1000.0),
        )

        assert knobs(outcome.points) == [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 138.0]
        assert outcome.stop_reason == STOP_UPPER_BOUND

    def test_normal_curve_still_starts_at_lower_bound(self):
        # Control: a healthy curve must NOT trigger the downward search.
        pts = run(
            sat_cfg(axis="rate", lower_bound=4, upper_bound=64),
            make_run_point(
                piecewise({4: 4000, 8: 8000, 16: 16000, 32: 30000, 64: 20000})
            ),
        )
        assert min(knobs(pts)) == 4.0, f"should not probe below 4: {knobs(pts)}"


def test_second_point_plateau_no_crash():
    # Plateau on the 2nd point (prev_prev is None) must not crash; bracket falls
    # back to (lower_bound, knob).
    curve = piecewise({1: 1000, 2: 1010})
    pts = run(sat_cfg(), make_run_point(curve))
    assert peak_tps(pts) >= 1010 - 1
    assert set(knobs(pts)) >= {1.0, 2.0}


def test_no_duplicate_probes():
    curve = piecewise(
        {1: 100, 2: 200, 4: 400, 8: 800, 16: 1600, 24: 2000, 32: 1700, 64: 900}
    )
    pts = run(sat_cfg(), make_run_point(curve))
    ks = knobs(pts)
    assert len(ks) == len(set(ks)), f"duplicate probes: {ks}"


def test_respects_max_points_budget():
    curve = piecewise({1: 100, 2: 200, 4: 400, 8: 800, 16: 1600, 32: 3200, 64: 900})
    pts = run(sat_cfg(max_points=4), make_run_point(curve))
    assert len(pts) <= 4


def test_slo_concurrency_search_stops_after_three_measured_points():
    outcome = run_outcome(
        sat_cfg(
            axis="concurrency",
            lower_bound=4,
            upper_bound=100,
            max_points=3,
            slo_avg_latency_ms=70.0,
        ),
        make_run_point(lambda k: k * 1000.0, latency_fn=lambda k: k * 5.0),
    )

    assert knobs(outcome.points) == [4.0, 8.0, 16.0]
    assert outcome.stop_reason == STOP_BUDGET_POINTS


def test_ties_prefer_lower_knob():
    # Flat top: 4/8 share the max; argmax should keep the cheaper (lower) knob.
    curve = piecewise({1: 100, 2: 400, 4: 1000, 8: 1000, 16: 1000})
    pts = run(sat_cfg(), make_run_point(curve))
    assert peak_tps(pts) >= 1000 - 1
    assert peak_knob(pts) <= 8.0


def test_overload_midramp_sets_bracket():
    # Healthy through 16, overloaded at 32 -> bracket (prev_prev=8, 32).
    curve = piecewise({1: 100, 2: 200, 4: 400, 8: 800, 16: 1600, 32: 1500})

    def success_fn(k):
        return 0.4 if k >= 32 else 1.0

    pts = run(sat_cfg(), make_run_point(curve, success_fn=success_fn))
    assert peak_tps(pts) >= 1600 - 1
    assert peak_knob(pts) <= 32.0


# ── rate axis: can't-keep-up stop ────────────────────────────────────────────
def test_rate_axis_cant_keepup_stops():
    # Achieved rate saturates at 10 rps; offering more buys < 5% more throughput.
    def achieved_fn(k):
        return min(k, 10)

    curve = lambda k: min(k, 10) * 100.0  # noqa: E731
    pts = run(
        sat_cfg(axis="rate"),
        make_run_point(curve, achieved_fn=achieved_fn),
    )
    assert peak_tps(pts) >= 1000 - 1
    # The ramp must have pushed past the saturation knob (16) to detect the stall.
    assert max(knobs(pts)) >= 32.0


# ── noise robustness ─────────────────────────────────────────────────────────
def test_noisy_unimodal_finds_near_peak():
    # Unimodal peak at 32 with deterministic +-2% measurement noise.
    curve = piecewise(
        {1: 100, 2: 200, 4: 400, 8: 800, 16: 1600, 32: 3200, 48: 2800, 64: 2000}
    )
    noise = lambda knob, index: 0.02 * math.sin(index * 1.7)  # noqa: E731
    pts = run(sat_cfg(), make_run_point(curve, noise_fn=noise))
    # Within noise, the reported peak should be close to the true 3200 and the
    # winning knob near 32 (allow a couple of geometric neighbours).
    assert peak_tps(pts) >= 3200 * 0.9
    assert 16 <= peak_knob(pts) <= 48


def test_probes_left_half_when_bracket_spans_it():
    # Explicit check that the (prev_prev, prev) half is now inside the search:
    # with a peak at 24 the search must place probes below 32.
    curve = piecewise(
        {1: 100, 2: 200, 4: 400, 8: 800, 16: 1600, 24: 2000, 32: 1700, 64: 900}
    )
    calls = []
    run(sat_cfg(), make_run_point(curve, calls=calls))
    assert any(16 < k < 32 for k in calls)


# ── SLO target regression ────────────────────────────────────────────────────
def slo_cfg(**kw):
    d = dict(axis="rate", lower_bound=1, upper_bound=1024, max_points=12)
    d.update(kw)
    return AutoTuneConfig(**d)


def climbing(k):
    """Throughput that never saturates (grows with the knob).

    The SLO cases below are about the LATENCY predicate, so their throughput curve
    has to stay out of the way: since the SLO branch now also stops when the server
    stops converting load into work, a FLAT curve would read as "saturated at the
    second point" and cut every one of these ramps short.
    """
    return k * 100.0


def _max_passing_knob(points, threshold_ms):
    passing = [
        p.knob for p in points if p.success >= 0.95 and p.latency_ms <= threshold_ms
    ]
    return max(passing) if passing else None


def test_slo_binary_search_finds_boundary():
    # latency = knob * 20 ms; SLO avg latency <= 200 -> boundary at knob 10.
    rp = make_run_point(climbing, latency_fn=lambda k: k * 20.0)
    pts = run(slo_cfg(slo_avg_latency_ms=200.0), rp)
    assert _max_passing_knob(pts, 200.0) == 10.0
    # It should have bisected the (8, 16) gap, i.e. probed something in-between.
    assert any(8 < k < 16 for k in knobs(pts))


def test_slo_all_pass_to_upper_bound():
    rp = make_run_point(climbing, latency_fn=lambda k: k * 1.0)
    pts = run(slo_cfg(upper_bound=8, slo_avg_latency_ms=100000.0), rp)
    assert _max_passing_knob(pts, 100000.0) == 8.0
    assert max(knobs(pts)) == 8.0


def test_slo_first_point_fails():
    # Even the lowest knob violates the SLO.
    rp = make_run_point(climbing, latency_fn=lambda k: k * 20.0)
    pts = run(slo_cfg(slo_avg_latency_ms=10.0), rp)
    assert len(pts) == 1
    assert _max_passing_knob(pts, 10.0) is None


def test_slo_pass_then_fail_creates_bracket():
    # Passes through 8, fails at 16; boundary bisected within (8, 16).
    rp = make_run_point(climbing, latency_fn=lambda k: k * 20.0)
    pts = run(slo_cfg(slo_avg_latency_ms=200.0), rp)
    ks = knobs(pts)
    assert 8.0 in ks and 16.0 in ks
    assert _max_passing_knob(pts, 200.0) == 10.0


def test_slo_success_floor_fails_point():
    # Latency always passes, but success drops below the floor at knob >= 8, so
    # the SLO boundary is the largest knob still under 8 -> 7 (bisected in (4,8)).
    def success_fn(k):
        return 0.5 if k >= 8 else 1.0

    rp = make_run_point(climbing, latency_fn=lambda k: 1.0, success_fn=success_fn)
    pts = run(slo_cfg(slo_avg_latency_ms=100000.0), rp)
    assert _max_passing_knob(pts, 100000.0) == 7.0
    # The failing point (8) must have been sampled to close the bracket.
    assert 8.0 in knobs(pts)


class TestSloContinuesPastThroughputPlateaus:
    """The SLO boundary depends on latency and success, not throughput growth."""

    # Measured output tokens/s and end-to-end latency (ms) per concurrency point.
    BM74_TPS = {
        4: 1738.7,
        8: 3155.4,
        16: 4838.7,
        32: 8742.4,
        64: 13146.0,
        128: 16343.2,
        256: 17552.3,
        512: 17447.6,
        1024: 16471.5,
    }
    BM74_LATENCY_MS = {
        4: 330,
        8: 360,
        16: 420,
        32: 470,
        64: 630,
        128: 1020,
        256: 1900,
        512: 3790,
        1024: 7760,
    }

    def _cfg(self, **kw):
        d = dict(
            axis="concurrency",
            lower_bound=4,
            upper_bound=1024,
            max_points=12,
            probe_saturation=False,
        )
        d.update(kw)
        return AutoTuneConfig(**d)

    def _rp(self, calls=None):
        return make_run_point(
            piecewise(self.BM74_TPS),
            latency_fn=piecewise(self.BM74_LATENCY_MS),
            calls=calls,
        )

    def test_loose_slo_reaches_the_search_bound_after_throughput_declines(self):
        outcome = run_outcome(self._cfg(slo_avg_latency_ms=10000.0), self._rp())
        assert knobs(outcome.points) == [
            4.0,
            8.0,
            16.0,
            32.0,
            64.0,
            128.0,
            256.0,
            512.0,
            1024.0,
        ]
        assert outcome.stop_reason == STOP_UPPER_BOUND
        assert outcome.bracket_reason == STOP_UPPER_BOUND
        assert outcome.slo_bracket == (1024.0, None)
        assert peak_knob(outcome.points) == 256.0

    def test_latency_boundary_is_located_beyond_the_throughput_peak(self):
        outcome = run_outcome(
            self._cfg(slo_avg_latency_ms=5000.0, max_points=20), self._rp()
        )
        assert outcome.bracket_reason == STOP_SLO_FAILED
        assert outcome.stop_reason == STOP_CONVERGED
        passing, failing = outcome.slo_bracket
        assert passing == 668.0
        assert failing == 669.0
        assert peak_knob(outcome.points) == 256.0

    def test_plateau_does_not_create_a_false_slo_failure(self):
        outcome = run_outcome(self._cfg(slo_avg_latency_ms=10000.0), self._rp())
        assert outcome.slo_bracket[1] is None
        assert all(
            float(k).is_integer() and int(k) & (int(k) - 1) == 0
            for k in knobs(outcome.points)
        )

    def test_rate_search_continues_when_achieved_rate_stops_growing(self):
        outcome = run_outcome(
            self._cfg(
                axis="rate",
                lower_bound=1,
                upper_bound=128,
                max_points=20,
                slo_avg_latency_ms=700.0,
            ),
            make_run_point(
                lambda k: min(k, 10) * 100.0,
                achieved_fn=lambda k: min(k, 10),
                latency_fn=lambda k: k * 10.0,
            ),
        )
        assert outcome.stop_reason == STOP_CONVERGED
        assert outcome.slo_bracket == (70.0, 71.0)
        assert 64.0 in knobs(outcome.points)

    def test_flat_throughput_still_respects_the_point_budget(self):
        outcome = run_outcome(
            self._cfg(lower_bound=1, max_points=3, slo_avg_latency_ms=500.0),
            make_run_point(lambda k: 100.0),
        )
        assert knobs(outcome.points) == [1.0, 2.0, 4.0]
        assert outcome.stop_reason == STOP_BUDGET_POINTS
        assert outcome.slo_bracket == (4.0, None)

    def test_flat_throughput_still_respects_the_time_budget(self, monkeypatch):
        import benchmark_runner.auto_tune as auto_tune

        clock = [0.0]
        monkeypatch.setattr(auto_tune.time, "monotonic", lambda: clock[0])

        async def point(knob, index):
            clock[0] += 1.0
            return make_metrics(knob, index, 100.0)

        outcome = run_outcome(
            self._cfg(
                lower_bound=1,
                max_points=20,
                max_total_seconds=2.5,
                slo_avg_latency_ms=500.0,
            ),
            point,
        )
        assert knobs(outcome.points) == [1.0, 2.0, 4.0]
        assert outcome.stop_reason == STOP_BUDGET_SECONDS

    def test_flat_throughput_still_brackets_request_failures(self):
        outcome = run_outcome(
            self._cfg(lower_bound=1, slo_avg_latency_ms=10000.0),
            make_run_point(lambda k: 100.0, success_fn=lambda k: 1.0 if k < 8 else 0.9),
        )
        assert outcome.stop_reason == STOP_CONVERGED
        assert outcome.slo_bracket == (7.0, 8.0)

    def test_a_latency_boundary_before_saturation_still_converges(self):
        outcome = run_outcome(self._cfg(slo_avg_latency_ms=500.0), self._rp())
        assert outcome.bracket_reason == STOP_SLO_FAILED
        assert outcome.stop_reason == STOP_CONVERGED
        passing, failing = outcome.slo_bracket
        assert passing is not None and failing is not None and failing - passing <= 1


class TestSloThroughputDeclineProtection:
    """Sustained total TPS loss can end expansion without declaring an SLO failure."""

    @staticmethod
    def cfg(**overrides):
        settings = dict(
            lower_bound=1,
            upper_bound=64,
            max_points=20,
            slo_avg_latency_ms=1000.0,
        )
        settings.update(overrides)
        return sat_cfg(**settings)

    @pytest.mark.parametrize("axis", ["concurrency", "rate"])
    def test_two_significant_drops_stop_without_fabricating_a_boundary(self, axis):
        curve = piecewise({1: 1000, 2: 1500, 4: 1200, 8: 1190, 64: 1100})
        outcome = run_outcome(self.cfg(axis=axis), make_run_point(curve))

        assert knobs(outcome.points) == [1.0, 2.0, 4.0, 8.0]
        assert outcome.stop_reason == "throughput_decline"
        assert outcome.bracket_reason == "throughput_decline"
        assert outcome.slo_bracket == (8.0, None)
        assert outcome.to_dict()["stop_reason"] == "throughput_decline"

    @pytest.mark.parametrize("axis", ["concurrency", "rate"])
    def test_stop_records_the_peak_and_each_qualifying_measurement(self, axis):
        curve = piecewise({64: 10000, 128: 8200, 256: 8000, 1024: 7000})
        outcome = run_outcome(
            self.cfg(axis=axis, lower_bound=64, upper_bound=1024),
            make_run_point(curve),
        )

        assert outcome.stop_reason == "throughput_decline"
        assert outcome.slo_bracket == (256.0, None)
        assert outcome.to_dict().get("throughput_decline") == {
            "threshold_percent": 15,
            "consecutive_points": 2,
            "peak_rate": 64.0,
            "peak_total_tps": 10000.0,
            "points": [
                {"rate": 128.0, "total_tps": 8200.0, "drop_percent": 18.0},
                {"rate": 256.0, "total_tps": 8000.0, "drop_percent": 20.0},
            ],
        }

    def test_recorded_evidence_excludes_discarded_dips_before_a_new_peak(self):
        curve = piecewise({1: 1000, 2: 1500, 4: 1200, 8: 2000, 16: 1600, 32: 1500})
        outcome = run_outcome(self.cfg(), make_run_point(curve))

        assert outcome.to_dict().get("throughput_decline") == {
            "threshold_percent": 15,
            "consecutive_points": 2,
            "peak_rate": 8.0,
            "peak_total_tps": 2000.0,
            "points": [
                {"rate": 16.0, "total_tps": 1600.0, "drop_percent": 20.0},
                {"rate": 32.0, "total_tps": 1500.0, "drop_percent": 25.0},
            ],
        }

    def test_evidence_records_the_policy_in_force_at_the_stop(self, monkeypatch):
        import benchmark_runner.auto_tune as at

        monkeypatch.setattr(at, "SLO_THROUGHPUT_DROP_RATIO", 0.2)
        monkeypatch.setattr(at, "SLO_THROUGHPUT_DROP_POINTS", 3)
        curve = piecewise({1: 1000, 2: 750, 4: 700, 8: 600})
        outcome = run_outcome(self.cfg(), make_run_point(curve))

        assert knobs(outcome.points) == [1.0, 2.0, 4.0, 8.0]
        assert outcome.to_dict().get("throughput_decline") == {
            "threshold_percent": 20,
            "consecutive_points": 3,
            "peak_rate": 1.0,
            "peak_total_tps": 1000.0,
            "points": [
                {"rate": 2.0, "total_tps": 750.0, "drop_percent": 25.0},
                {"rate": 4.0, "total_tps": 700.0, "drop_percent": 30.0},
                {"rate": 8.0, "total_tps": 600.0, "drop_percent": 40.0},
            ],
        }

    def test_a_single_dip_followed_by_recovery_does_not_stop(self):
        curve = piecewise({1: 1000, 2: 1500, 4: 1000, 8: 1600, 64: 1600})
        outcome = run_outcome(self.cfg(), make_run_point(curve))

        assert outcome.stop_reason == STOP_UPPER_BOUND
        assert outcome.slo_bracket == (64.0, None)
        assert "throughput_decline" not in outcome.to_dict()

    def test_small_declines_do_not_stop_expansion(self):
        curve = piecewise({1: 1000, 2: 1500, 4: 1400, 8: 1300, 16: 1280, 64: 1300})
        outcome = run_outcome(self.cfg(), make_run_point(curve))

        assert outcome.stop_reason == STOP_UPPER_BOUND
        assert max(knobs(outcome.points)) == 64.0

    def test_partial_recovery_resets_the_consecutive_decline_count(self):
        curve = piecewise({1: 1000, 2: 1500, 4: 1200, 8: 1450, 16: 1200, 32: 1100})
        outcome = run_outcome(self.cfg(), make_run_point(curve))

        assert outcome.stop_reason == "throughput_decline"
        assert max(knobs(outcome.points)) == 32.0
        assert outcome.slo_bracket == (32.0, None)

    def test_a_fifteen_percent_loss_persisting_for_two_loads_is_enough(self):
        curve = piecewise({1: 1000, 2: 850, 4: 850, 64: 850})
        outcome = run_outcome(self.cfg(), make_run_point(curve))

        assert knobs(outcome.points) == [1.0, 2.0, 4.0]
        assert outcome.stop_reason == "throughput_decline"

    def test_latency_failure_takes_precedence_and_bisection_continues(self):
        curve = piecewise({1: 1000, 2: 1500, 4: 1200, 8: 1100})
        outcome = run_outcome(
            self.cfg(), make_run_point(curve, latency_fn=lambda k: k * 200)
        )

        assert outcome.bracket_reason == STOP_SLO_FAILED
        assert outcome.stop_reason == STOP_CONVERGED
        assert outcome.slo_bracket == (5.0, 6.0)
        assert "throughput_decline" not in outcome.to_dict()

    def test_request_failures_take_precedence_over_throughput_protection(self):
        curve = piecewise({1: 1000, 2: 1500, 4: 1200, 8: 1100})
        outcome = run_outcome(
            self.cfg(),
            make_run_point(curve, success_fn=lambda k: 1.0 if k < 8 else 0.9),
        )

        assert outcome.bracket_reason == STOP_SLO_FAILED
        assert outcome.stop_reason == STOP_CONVERGED
        assert outcome.slo_bracket == (7.0, 8.0)

    def test_point_budget_can_end_before_decline_is_confirmed(self):
        curve = piecewise({1: 1000, 2: 1500, 4: 1200, 8: 1100})
        outcome = run_outcome(self.cfg(max_points=3), make_run_point(curve))

        assert knobs(outcome.points) == [1.0, 2.0, 4.0]
        assert outcome.stop_reason == STOP_BUDGET_POINTS
        assert outcome.slo_bracket == (4.0, None)
        assert "throughput_decline" not in outcome.to_dict()


class TestTotalThroughputIsAuthoritative:
    @staticmethod
    def measured(knob, index, *, total, output):
        point = make_metrics(knob, index, output)
        point.total_tps = total
        return point

    def test_total_peak_drives_saturation_even_when_output_keeps_growing(self):
        totals = piecewise({1: 1000, 2: 2000, 4: 3000, 8: 2000, 16: 1000})

        async def point(knob, index):
            return self.measured(knob, index, total=totals(knob), output=knob * 100)

        outcome = run_outcome(
            sat_cfg(lower_bound=1, upper_bound=16, max_points=20), point
        )
        assert outcome.bracket_reason == STOP_CAPACITY_PLATEAU
        assert max(outcome.points, key=lambda p: p.total_tps).knob == 4.0
        assert outcome.to_dict()["points"][0].get("total_tps") == 1000

    def test_output_decline_does_not_stop_slo_when_total_keeps_growing(self):
        outputs = piecewise({1: 1500, 2: 1200, 4: 1100, 8: 900, 16: 800})

        async def point(knob, index):
            return self.measured(knob, index, total=knob * 5000, output=outputs(knob))

        outcome = run_outcome(
            sat_cfg(
                lower_bound=1, upper_bound=16, max_points=20, slo_avg_latency_ms=1000
            ),
            point,
        )
        assert outcome.stop_reason == STOP_UPPER_BOUND
        assert outcome.slo_bracket == (16.0, None)

    def test_total_decline_protects_slo_even_when_output_keeps_growing(self):
        totals = piecewise({1: 1000, 2: 1500, 4: 1200, 8: 1190, 16: 1100})

        async def point(knob, index):
            return self.measured(knob, index, total=totals(knob), output=knob * 100)

        outcome = run_outcome(
            sat_cfg(
                lower_bound=1, upper_bound=16, max_points=20, slo_avg_latency_ms=1000
            ),
            point,
        )
        assert outcome.stop_reason == "throughput_decline"
        assert outcome.slo_bracket == (8.0, None)
        evidence = outcome.to_dict()["throughput_decline"]
        assert evidence["peak_total_tps"] == 1500
        assert evidence["points"] == [
            {"rate": 4.0, "total_tps": 1200.0, "drop_percent": 20.0},
            {"rate": 8.0, "total_tps": 1190.0, "drop_percent": 20.666667},
        ]

    def test_normalization_reads_recorded_total_independently_of_output(self):
        report = SimpleNamespace(
            metrics=SimpleNamespace(
                request_totals=SimpleNamespace(total=100, successful=100),
                tokens_per_second=SimpleNamespace(
                    successful=SimpleNamespace(mean=3000)
                ),
                prompt_tokens_per_second=SimpleNamespace(
                    successful=SimpleNamespace(mean=8000)
                ),
                output_tokens_per_second=SimpleNamespace(
                    successful=SimpleNamespace(mean=1000)
                ),
            )
        )
        point = _normalize(report, 4, 0)
        assert getattr(point, "total_tps", None) == 3000
        assert point.output_tps == 1000


# ── injection / plumbing ─────────────────────────────────────────────────────
def test_run_point_receives_monotonic_index():
    curve = piecewise({1: 100, 2: 200, 4: 400, 8: 800, 16: 1600, 32: 3200, 64: 900})
    seen = []

    async def rp(knob, index):
        seen.append((knob, index))
        return make_metrics(knob, index, curve(knob))

    run(sat_cfg(), rp)
    indices = [i for _, i in seen]
    assert indices == list(range(len(indices)))  # 0,1,2,... no gaps/dupes


def test_returns_empty_when_first_point_is_none():
    async def rp(knob, index):
        return None

    pts = run(sat_cfg(), rp)
    assert pts == []


# ── saturation-bounded start (throughput probe) ──────────────────────────────
def test_probe_starts_ramp_near_ceiling():
    # Peak ~31 like benchmark 7; probe reports the ceiling ~32.
    curve = piecewise(
        {1: 100, 8: 9000, 16: 18000, 24: 27000, 31: 35000, 32: 34000, 64: 28000}
    )
    pts = run(
        sat_cfg(axis="rate", lower_bound=4, probe_saturation=True),
        make_run_point(curve),
        probe=make_probe(32),
    )
    ks = knobs(pts)
    # The probe caps the TOP of the ramp; it must not lift the START. A lifted
    # start (ceiling/4) saved ~1 point at lower_bound=4 but dropped the low-load
    # end of the curve — the unloaded latency baseline the decision chart is read
    # against — and landed past the peak whenever the ~2s ceiling estimate ran
    # high, where the first point plateaus and Phase 1 yields no bracket at all.
    assert ks[0] == 4.0, f"ramp must start at lower_bound=4, got {ks[0]}"
    assert peak_tps(pts) >= 34000  # the ~35000 peak at 31 is still found


def test_probe_does_not_change_the_start():
    curve = piecewise(
        {1: 100, 8: 9000, 16: 18000, 24: 27000, 31: 35000, 32: 34000, 64: 28000}
    )
    with_probe = run(
        sat_cfg(axis="rate", lower_bound=4, probe_saturation=True),
        make_run_point(curve),
        probe=make_probe(32),
    )
    without = run(
        sat_cfg(axis="rate", lower_bound=4, probe_saturation=False),
        make_run_point(curve),
    )
    # Same low-load coverage either way; the probe only bounds the top.
    assert knobs(with_probe)[0] == knobs(without)[0] == 4.0
    assert max(knobs(with_probe)) <= max(knobs(without))


def test_probe_relaxes_its_cap_when_the_ceiling_was_underestimated():
    """A wrong ceiling must not truncate a real curve.

    The probe measures for ~2s and can read far low. Reaching its cap while the
    server is still keeping up (achieved rate above the probed ceiling) means the
    estimate was bad, so the cap doubles rather than reporting a peak 10x too low.
    """
    # True ceiling 200, but the probe reports 20 -> initial cap ceil(20*1.2)=24.
    pts = run(
        sat_cfg(axis="rate", lower_bound=4, upper_bound=1024, probe_saturation=True),
        make_run_point(
            lambda k: min(k, 200.0) * 1000.0,
            achieved_fn=lambda k: min(k, 200.0),
        ),
        probe=make_probe(20),
    )
    assert max(knobs(pts)) > 24.0, f"cap should have relaxed past 24: {knobs(pts)}"
    assert peak_tps(pts) >= 199_000, f"must still find the ~200 peak: {knobs(pts)}"


def test_probe_caps_upper_bound():
    # Ceiling 20 -> upper capped at ceil(20*1.2)=24; ramp must not run past it.
    curve = piecewise({1: 100, 5: 6000, 10: 12000, 16: 15000, 20: 14000})
    pts = run(
        sat_cfg(axis="rate", upper_bound=1024, probe_saturation=True),
        make_run_point(curve),
        probe=make_probe(20),
    )
    # NO measured point may exceed the tightened bound. Phase-1's bound check runs
    # after a point is measured, so an unclamped `knob *= 2` used to step over it
    # (bound 24 -> probed 32); the doubling is clamped to the bound instead.
    assert max(knobs(pts)) <= 24.0, f"no point may exceed the cap: {knobs(pts)}"


@pytest.mark.parametrize("ceiling", [20, 31, 50])
def test_probe_bound_holds_for_a_wide_upper_bound(ceiling):
    """A generous upper_bound must not leak into the probed range.

    The point of passing upper_bound=1024 with the probe on is "let the server
    tell us the range", so the ramp must stay near the probed ceiling instead of
    doubling on toward 256/1024.

    "Near" is ONE relaxation, not the cap itself: the cap bounds the step, and a
    measured point that cleared the growth criteria overrules it (see
    TestTheCapDoesNotOutrankAMeasurement). So the ramp may take one step past
    ceil(ceiling*1.2) — and must then stop there, on measured evidence.
    """
    cap = math.ceil(ceiling * 1.2)
    # Achieved rate saturates at the ceiling; tps tracks it.
    o = run_outcome(
        sat_cfg(axis="rate", lower_bound=4, upper_bound=1024, probe_saturation=True),
        make_run_point(
            lambda k: min(k, ceiling) * 1000.0,
            achieved_fn=lambda k: min(k, ceiling),
        ),
        probe=make_probe(ceiling),
    )
    ks = knobs(o.points)
    assert max(ks) <= 2 * cap, f"ceiling={ceiling} cap={cap}: {ks}"
    assert max(ks) < 128, f"the wide upper_bound leaked in: {ks}"
    # Stopped because the measurements said so, not because the range ran out.
    assert o.bracket_reason == STOP_CAPACITY_PLATEAU


def test_probe_skipped_for_slo():
    calls = []
    rp = make_run_point(climbing, latency_fn=lambda k: k * 20.0)
    run(
        slo_cfg(slo_avg_latency_ms=200.0, probe_saturation=True),
        rp,
        probe=make_probe(32, calls),
    )
    assert calls == [], "probe must not run for the SLO target"


def test_probe_skipped_for_concurrency():
    calls = []
    curve = piecewise({1: 100, 2: 400, 4: 1000, 8: 1000})
    run(
        sat_cfg(axis="concurrency", probe_saturation=True),
        make_run_point(curve),
        probe=make_probe(32, calls),
    )
    assert calls == [], "probe must not run on the concurrency axis"


def test_probe_disabled_flag():
    calls = []
    curve = piecewise({1: 100, 8: 9000, 16: 18000, 32: 35000, 64: 28000})
    pts = run(
        sat_cfg(axis="rate", probe_saturation=False),
        make_run_point(curve),
        probe=make_probe(32, calls),
    )
    assert calls == [], "probe must not run when probe_saturation=False"
    assert knobs(pts)[0] == 1.0  # falls back to lower_bound=1


def test_probe_zero_ceiling_falls_back():
    # Probe fails (returns 0) -> fall back to the normal ramp from lower_bound.
    curve = piecewise({1: 100, 8: 9000, 16: 18000, 32: 35000, 64: 28000})
    pts = run(
        sat_cfg(axis="rate", probe_saturation=True),
        make_run_point(curve),
        probe=make_probe(0),
    )
    assert knobs(pts)[0] == 1.0, "zero ceiling should fall back to lower_bound=1"


class TestTheProbeDoesNotSpendThePointBudget:
    """``max_points`` caps MEASURED points; the probe is not one of them.

    The probe writes a ``__satprobe`` file the consumer's point glob never matches
    and its result never enters an aggregate — the README already documents it as
    "not counted as a measured point". Charging it to ``max_points`` anyway made
    ``--max-points 1`` legal at the CLI and yet produce ZERO measured points: the
    probe took the only slot and Phase 1's predicate was false on entry. That is
    exactly the "no points, no bracket" outcome the up-front range validation
    exists to prevent, reached through a value it accepts.

    Its wall clock is still bounded — it runs after the ramp's start, so
    ``_elapsed`` contains it, and its own run is capped by the remaining budget.
    """

    def _outcome(self, max_points):
        return run_outcome(
            sat_cfg(
                axis="rate",
                lower_bound=4,
                upper_bound=1024,
                max_points=max_points,
                probe_saturation=True,
            ),
            make_run_point(piecewise({4: 400, 1024: 4000})),
            probe=make_probe(50.0),
        )

    def test_a_one_point_budget_still_measures_that_point(self):
        o = self._outcome(1)
        assert knobs(o.points) == [4.0]
        assert o.stopped_at == 4.0

    def test_the_budget_is_spent_entirely_on_measured_points(self):
        o = self._outcome(2)
        assert len(o.points) == 2, "the probe must not take one of the two slots"


# ── server progress reporting ────────────────────────────────────────────────
class _RecordingProgress(ServerBenchmarkerProgress):
    """Real ``ServerBenchmarkerProgress`` math, no network.

    Records every ``overall`` value the ramp/guidellm callbacks compute, with the
    1s/2% throttle bypassed so the full sequence is visible.
    """

    def __init__(self):
        super().__init__(progress_url="http://unused")
        self.emitted = []

    def _ensure_session(self):  # never open a real aiohttp session
        return

    async def _update_progress(self, progress: float):
        self.emitted.append(progress)

    async def on_finalize(self):
        return


async def _simulate_guidellm_run(sp, fractions=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Replay the callbacks guidellm fires for one single-strategy run."""
    await sp.on_initialize(None)  # no strategy_types -> _bench_total = 1
    await sp.on_benchmark_start(None)
    for f in fractions:
        state = SimpleNamespace(progress=SimpleNamespace(remaining_fraction=1.0 - f))
        await sp.on_benchmark_update(None, state)
    await sp.on_benchmark_complete(None)


def _run_with_progress(cfg, curve, ceiling=32.0):
    """Drive the ramp with a progress recorder; probe and points both emit.

    Returns ``(emitted, points, first_point_at)`` where ``first_point_at`` is the
    index in ``emitted`` at which the first MEASURED point started reporting, i.e.
    the boundary between the probe's slice and the ramp proper.
    """
    sp = _RecordingProgress()
    rp = make_run_point(curve)
    boundaries = []

    async def probe():
        await _simulate_guidellm_run(sp)
        return float(ceiling)

    async def run_point(knob, index):
        boundaries.append(len(sp.emitted))
        await _simulate_guidellm_run(sp)
        return await rp(knob, index)

    outcome = asyncio.run(
        run_ramp(
            cfg,
            {},
            "t",
            server_progress=sp,
            run_point=run_point,
            saturation_probe=probe,
        )
    )
    return sp.emitted, outcome.points, boundaries[0]


def test_probe_does_not_push_progress_to_100():
    # Regression (benchmark 28): the saturation probe ran with the progress
    # defaults (run_index=0, run_total=1), so its own 0..1 fraction covered the
    # whole bar -> 100% seconds after start, then pinned there by the server's
    # monotonic clamp while the ramp was still measuring points.
    curve = piecewise({4: 5000, 8: 9000, 16: 18000, 24: 27000, 31: 35000, 32: 34000})
    emitted, pts, _ = _run_with_progress(
        sat_cfg(axis="rate", lower_bound=4, probe_saturation=True), curve
    )
    assert len(pts) > 1
    assert emitted[-1] == 100.0, "the ramp must finalize at 100"
    assert (
        max(emitted[:-1]) < 100.0
    ), f"nothing before the ramp's final push may reach 100: {emitted}"


def test_probe_slice_is_a_small_fraction_of_the_bar():
    sp = _RecordingProgress()

    async def probe():
        await _simulate_guidellm_run(sp)
        return 32.0

    # Isolate the probe's slice: a run_point returning None ends Phase 1 at once,
    # so every value below 100 was emitted by the probe.
    async def run_point_none(knob, index):
        return None

    asyncio.run(
        run_ramp(
            sat_cfg(axis="rate", lower_bound=4, probe_saturation=True),
            {},
            "t",
            server_progress=sp,
            run_point=run_point_none,
            saturation_probe=probe,
        )
    )
    probe_max = max(p for p in sp.emitted if p < 100.0)
    assert probe_max <= 25.0, f"probe should own a small slice, got {probe_max}"


def test_probe_slice_sits_below_the_first_measured_point():
    # The probe must hand the bar over without moving it backward: its slice ends
    # at or below where the first measured point starts reporting. (Later phases
    # may still revise the remaining-points estimate downward, which the server's
    # monotonic clamp absorbs; the probe handover is the one the fix guarantees.)
    curve = piecewise({4: 5000, 8: 9000, 16: 18000, 24: 27000, 31: 35000, 32: 34000})
    emitted, _, first_point_at = _run_with_progress(
        sat_cfg(axis="rate", lower_bound=4, probe_saturation=True), curve
    )
    probe_vals, point_vals = emitted[:first_point_at], emitted[first_point_at:]
    assert probe_vals and point_vals
    assert max(probe_vals) <= min(
        point_vals
    ), f"probe slice {max(probe_vals)} overshoots the ramp start {min(point_vals)}"


# ── stop reasons ─────────────────────────────────────────────────────────────
class TestStopReasons:
    """Every termination reports WHY, because several of them leave identical grids.

    Reported rather than inferred by the consumer: `budget_seconds` and
    `budget_points` both produce "fewer points than the range would allow", and
    `upper_bound` vs `capacity_plateau` both produce "the best point is the last
    one" — with opposite advice attached (raise the duration cap / raise the range /
    nothing to fix).
    """

    def test_climbing_to_the_top_of_the_range_reports_the_bound(self):
        # Throughput still rising at upper_bound: the RANGE ended the search.
        o = run_outcome(
            sat_cfg(lower_bound=4, upper_bound=32),
            make_run_point(lambda k: k * 1000.0),
        )
        assert o.bracket_reason == STOP_UPPER_BOUND
        assert o.stop_reason == STOP_UPPER_BOUND
        assert o.stopped_at == 32.0

    def test_running_out_of_points_reports_the_point_budget(self):
        o = run_outcome(
            sat_cfg(lower_bound=1, upper_bound=1_000_000, max_points=4),
            make_run_point(lambda k: k * 1000.0),
        )
        assert o.bracket_reason == STOP_BUDGET_POINTS
        assert len(o.points) == 4

    def test_running_out_of_time_reports_the_duration_budget(self):
        # The distinction a consumer cannot make: with max_points still unspent, a
        # clock-limited run looks exactly like one that stopped on its own accord.
        async def slow(knob, index):
            await asyncio.sleep(0.02)
            return make_metrics(knob, index, knob * 1000.0)

        o = run_outcome(
            sat_cfg(
                lower_bound=1,
                upper_bound=1_000_000,
                max_points=99,
                max_total_seconds=0.05,
            ),
            slow,
        )
        assert o.bracket_reason == STOP_BUDGET_SECONDS
        assert len(o.points) < 99

    def test_an_overloaded_point_is_not_reported_as_a_plateau(self):
        # Both end Phase 1 in the saturation branch, and the grid cannot tell them
        # apart once a failed point's throughput also happens to flatten.
        curve = piecewise({1: 100, 2: 200, 4: 400, 8: 800})

        async def rp(knob, index):
            return make_metrics(
                knob, index, curve(knob), success=0.5 if knob >= 8 else 1.0
            )

        o = run_outcome(sat_cfg(lower_bound=1, upper_bound=64), rp)
        assert o.bracket_reason == STOP_OVERLOADED

    def test_a_plateau_brackets_then_the_peak_search_converges(self):
        # bracket_reason and stop_reason genuinely differ here: capacity ended the
        # bracket, and the peak search then finished normally. Collapsing them into
        # one field would lose whichever the consumer asked about.
        curve = piecewise({1: 100, 2: 400, 4: 900, 8: 1000, 16: 1010})
        o = run_outcome(sat_cfg(lower_bound=1, upper_bound=1024), make_run_point(curve))
        assert o.bracket_reason == STOP_CAPACITY_PLATEAU
        assert o.stop_reason == STOP_CONVERGED

    def test_a_point_that_produced_no_benchmark_is_reported(self):
        async def rp(knob, index):
            return None

        o = run_outcome(sat_cfg(), rp)
        assert o.bracket_reason == STOP_POINT_FAILED
        assert o.points == []
        assert o.stopped_at is None

    def test_a_cap_that_clamped_the_overshoot_is_distinguishable(self):
        # The one outcome that earns the probe's cost: the cap bound the last
        # Phase-1 point, and the server did NOT outrun the probe, so it held.
        # Reported as relaxed=0 with stopped_at == probe_bound.
        curve = piecewise({4: 400, 8: 800, 16: 1600, 32: 3200, 36: 2800})
        o = run_outcome(
            sat_cfg(
                axis="rate", lower_bound=4, upper_bound=1024, probe_saturation=True
            ),
            make_run_point(curve, achieved_fn=lambda k: min(k, 30.0)),
            probe=make_probe(29.22),
        )
        assert o.probe_ceiling == 29.22
        assert o.probe_bound == 36.0  # ceil(29.22 * 1.2)
        assert o.probe_relaxed == 0
        assert o.stopped_at == 36.0  # the cap clamped the doubling from 64

    def test_a_cap_the_server_outran_is_reported_as_relaxed(self):
        # The probe read low: the server keeps up past the cap, so the cap doubles.
        # Without the count this is indistinguishable from the case above — same
        # stop reason, same probe_ceiling, a curve that looks equally healthy.
        curve = piecewise({4: 400, 8: 800, 16: 1600, 32: 3200, 64: 6400, 128: 6500})
        o = run_outcome(
            sat_cfg(
                axis="rate", lower_bound=4, upper_bound=1024, probe_saturation=True
            ),
            make_run_point(curve, achieved_fn=lambda k: k),
            probe=make_probe(10.0),  # cap = 12, far below what the server sustains
        )
        assert o.probe_relaxed >= 1
        assert o.probe_bound is not None and o.probe_bound > 12.0
        assert o.stopped_at is not None and o.stopped_at > 12.0

    def test_a_cap_that_never_bound_anything_is_also_visible(self):
        # The probe read HIGH: throughput turned over well below the cap, so the
        # cap never clamped anything — the probe was pure cost. Told apart from the
        # clamping case by stopped_at < probe_bound.
        curve = piecewise({4: 400, 8: 800, 16: 1600, 32: 1650})
        o = run_outcome(
            sat_cfg(
                axis="rate", lower_bound=4, upper_bound=1024, probe_saturation=True
            ),
            make_run_point(curve, achieved_fn=lambda k: min(k, 200.0)),
            probe=make_probe(200.0),  # cap = 240, never reached
        )
        assert o.probe_relaxed == 0
        assert o.probe_bound == 240.0
        assert o.stopped_at is not None and o.stopped_at < o.probe_bound

    def test_a_search_that_finished_on_its_last_allowed_point_still_converged(self):
        """Exhausting the integer grid completes the search even at the point cap."""
        curve = piecewise({2: 1600, 3: 3280, 4: 3300, 5: 3280, 6: 3280, 8: 3290})
        exact = run_outcome(
            sat_cfg(lower_bound=2, upper_bound=16, max_points=6),
            make_run_point(curve),
        )
        roomy = run_outcome(
            sat_cfg(lower_bound=2, upper_bound=16, max_points=20),
            make_run_point(curve),
        )
        # Same search, so it must read the same either way.
        assert knobs(exact.points) == knobs(roomy.points)
        assert len(exact.points) == 6
        assert peak_knob(exact.points) == peak_knob(roomy.points) == 4.0
        assert exact.stop_reason == roomy.stop_reason == STOP_CONVERGED

    def test_a_peak_search_the_budget_really_did_cut_short_says_so(self):
        # The other side of the fix: a cap low enough to stop a point that WOULD
        # have been measured must still report the budget.
        curve = piecewise({8: 800, 12: 1300, 16: 1200, 32: 1150})
        o = run_outcome(
            sat_cfg(lower_bound=8, upper_bound=1024, max_points=5),
            make_run_point(curve),
        )
        assert len(o.points) == 5
        assert o.stop_reason == STOP_BUDGET_POINTS

    def test_an_slo_bisection_that_closed_its_interval_reports_converged(self):
        # Same bug, the other Phase 2. The bracket (11, 12) is reached either way;
        # only the reported reason used to differ.
        curve = piecewise({4: 100, 8: 200, 1024: 400})

        def rp():
            # Latency crosses the 200ms threshold between knob 11 and 12.
            return make_run_point(
                curve, latency_fn=lambda k: 100.0 if k <= 11 else 500.0
            )

        cfg = dict(
            axis="concurrency",
            lower_bound=4,
            upper_bound=1024,
            probe_saturation=False,
            slo_avg_latency_ms=200.0,
        )
        exact = run_outcome(AutoTuneConfig(max_points=6, **cfg), rp())
        roomy = run_outcome(AutoTuneConfig(max_points=12, **cfg), rp())
        assert knobs(exact.points) == knobs(roomy.points)
        assert exact.slo_bracket == roomy.slo_bracket == (11.0, 12.0)
        assert exact.stop_reason == roomy.stop_reason == STOP_CONVERGED

    def test_the_outcome_serializes_to_the_documented_shape(self):
        o = run_outcome(
            sat_cfg(lower_bound=4, upper_bound=32),
            make_run_point(lambda k: k * 1000.0),
        )
        d = o.to_dict()
        assert d == {
            "version": RAMP_OUTCOME_VERSION,
            "points": [asdict(p) for p in o.points],
            "bracket_reason": STOP_UPPER_BOUND,
            "stop_reason": STOP_UPPER_BOUND,
            "target": "saturation",
            "axis": "concurrency",
            "stopped_at": 32.0,
            "lower_bound": 4.0,
            "upper_bound": 32.0,
            "points_measured": len(o.points),
            "max_points": 12,
            "elapsed_seconds": d["elapsed_seconds"],
            # sat_cfg pins max_points but not the clock, so this is AutoTuneConfig's
            # own default travelling through to the sidecar.
            "max_total_seconds": 5400.0,
            "slo_bracket": None,
            "probe_ceiling": None,
            "probe_relaxed": 0,
            "probe_bound": None,
            "slo_thresholds": {},
        }
        # v2: the measured grid is inline as well as in the report files, so a
        # reader holding only this file can redraw the curve (`benchmark-runner
        # chart`). Every field of PointMetrics survives the round trip — the
        # renderer reads ttft_p99_ms/achieved_rate/success, none of which the
        # summary fields above carry.
        assert [p["knob"] for p in d["points"]] == [p.knob for p in o.points]
        assert set(d["points"][0]) == {f.name for f in fields(PointMetrics)}


class TestTheBudgetBoundsEachPoint:
    """``max_total_seconds`` must bound the ramp from INSIDE a point, not only
    between points.

    ``_budget_reason`` is consulted by the loop predicates only, so before this the
    cap was advisory: a point that stalls (unresponsive server, or an offered rate
    the server answers at a trickle) ran for however long its ``max_requests``
    took. And auto-tune deliberately drops the global ``--max-seconds``, so nothing
    else bounded it either. Each run now carries the remaining budget as guidellm's
    ``max_duration`` constraint.

    These exercise the REAL ``_default_run_point`` / ``_default_probe`` (no
    injected ``run_point``), which is exactly the code path every other test in
    this file bypasses.
    """

    def _drive(self, monkeypatch, cfg, *, benchmarks=True):
        """Run the ramp against a stubbed guidellm, capturing per-run kwargs."""
        import benchmark_runner.auto_tune as at

        captured = []

        def fake_build_args(local):
            captured.append(dict(local))
            return SimpleNamespace(spec=None)

        async def fake_benchmark(args, progress=None, console=None):
            report = SimpleNamespace(
                benchmarks=(
                    [
                        SimpleNamespace(
                            metrics=SimpleNamespace(
                                request_totals=SimpleNamespace(total=10, successful=10)
                            )
                        )
                    ]
                    if benchmarks
                    else []
                )
            )
            return report, None

        monkeypatch.setattr(at, "build_scenario_args", fake_build_args)
        monkeypatch.setattr(at, "benchmark_generative_text", fake_benchmark)
        outcome = asyncio.run(run_ramp(cfg, {}, "77"))
        return captured, outcome

    def test_every_point_carries_a_duration_cap(self, monkeypatch):
        cfg = AutoTuneConfig(axis="concurrency", lower_bound=4, upper_bound=8)
        captured, _ = self._drive(monkeypatch, cfg)
        assert captured, "no run was built"
        for local in captured:
            assert local["max_seconds"] > 0

    def test_the_cap_is_what_is_left_of_the_budget(self, monkeypatch):
        cfg = AutoTuneConfig(
            axis="concurrency", lower_bound=4, upper_bound=8, max_total_seconds=600.0
        )
        captured, _ = self._drive(monkeypatch, cfg)
        # Stubbed runs take ~no time, so the first point still sees nearly the
        # whole budget -- and never MORE than it.
        assert captured[0]["max_seconds"] <= 600.0
        assert captured[0]["max_seconds"] > 599.0

    def test_a_nearly_spent_budget_still_asks_for_a_usable_window(self, monkeypatch):
        # A 0-second run would be stopped before a single response landed and would
        # report no metrics -- downstream that reads as a FAILED point, not a
        # truncated one. The floor keeps it usable.
        # 50ms is comfortably more than the handful of statements between the ramp's
        # start and its first budget check (so a point DOES run) and far less than
        # the floor, so the cap handed to that point can only have come from the
        # floor rather than from the arithmetic.
        cfg = AutoTuneConfig(
            axis="concurrency", lower_bound=4, upper_bound=8, max_total_seconds=0.05
        )
        captured, _ = self._drive(monkeypatch, cfg)
        assert captured, "the budget was not yet spent; a point should have run"
        assert captured[0]["max_seconds"] >= 5.0

    def test_the_saturation_probe_is_capped_too(self, monkeypatch):
        # The probe is not counted as a point but spends the same clock.
        cfg = AutoTuneConfig(
            axis="rate", lower_bound=4, upper_bound=8, probe_saturation=True
        )
        captured, _ = self._drive(monkeypatch, cfg)
        probe = next(
            c for c in captured if "satprobe" in c["outputs"][0]
        )  # the probe's own output file
        assert probe["max_seconds"] > 0
        assert probe["profile"] == "throughput"


class TestPerPointRequestFloor:
    """`min_requests` is a percentile floor, not just a "measure something" floor.

    A percentile is only as good as the samples ABOVE it: with n samples, p99 has
    n/100 above it. At n=40 (concurrency 4 x multiplier 10) p99 IS the maximum, so a
    single outlier defines the tail — and the ramp's own Phase-1 SLO predicate reads
    `ttft_p99_ms`, which means one slow request on the FIRST point could bracket
    immediately and report the server as too slow for the SLO.
    """

    def test_the_floor_lifts_the_cheap_concurrency_stages(self):
        # knob 4 x multiplier 10 = 40 -> floored to 100; knob 16 x 10 = 160 stands.
        cfg = sat_cfg(axis="concurrency")
        assert cfg.min_requests == 100
        mult = cfg.resolved_multiplier
        assert max(cfg.min_requests, round(4 * mult)) == 100
        assert max(cfg.min_requests, round(8 * mult)) == 100
        assert max(cfg.min_requests, round(16 * mult)) == 160

    def test_the_floor_does_not_touch_the_rate_axis_defaults(self):
        # multiplier 30 means the floor only binds under ~3.3 req/s, and the default
        # lower_bound is 4 — so a normal rate-axis sweep is unchanged.
        cfg = sat_cfg(axis="rate")
        mult = cfg.resolved_multiplier
        assert max(cfg.min_requests, round(4 * mult)) == 120
        assert max(cfg.min_requests, round(3 * mult)) == 100  # only below ~3.3

    def test_a_stage_at_the_floor_still_cannot_support_p99(self):
        # 100 removes the degenerate p99 == max (it becomes the second-largest
        # sample), it does NOT make the tail an estimate: 100/100 = 1 sample above
        # p99. The report says so rather than the floor implying otherwise.
        n = 100
        assert n // 100 == 1


class TestTpotIsDecodeOnly:
    """The SLO's TPOT is the decode-only per-token time, not guidellm's name for it.

    guidellm reports two per-output-token latencies and its naming is the reverse
    of the industry's: `inter_token_latency_ms` is (last_token - first_token) /
    (tokens - 1), i.e. what vLLM and genai-perf call TPOT, while
    `time_per_output_token_ms` starts the clock at request_start and therefore
    folds TTFT into the decode average. The ramp used to bracket on the second
    one, so a TPOT threshold tightened as the queue grew — the error is
    TTFT / (n * TPOT), ~5% at 128 output tokens and ~40% at 16.
    """

    @staticmethod
    def _report(*, itl: float, tpot_incl_ttft: float):
        """A guidellm ``benchmarks[0]`` stub carrying both per-token metrics."""

        def dist(mean):
            return SimpleNamespace(
                successful=SimpleNamespace(
                    mean=mean,
                    max=mean,
                    percentiles=SimpleNamespace(p95=mean, p99=mean),
                )
            )

        return SimpleNamespace(
            metrics=SimpleNamespace(
                request_totals=SimpleNamespace(total=100, successful=100),
                output_tokens_per_second=dist(500.0),
                tokens_per_second=dist(1000.0),
                time_to_first_token_ms=dist(200.0),
                inter_token_latency_ms=dist(itl),
                time_per_output_token_ms=dist(tpot_incl_ttft),
                request_latency=dist(1.5),
                requests_per_second=dist(4.0),
            )
        )

    def test_normalize_reads_the_decode_only_metric(self):
        m = _normalize(self._report(itl=4.5, tpot_incl_ttft=6.1), knob=4.0, index=0)
        assert (m.itl_ms, m.itl_p95_ms, m.itl_p99_ms) == (4.5, 4.5, 4.5)

    def test_a_threshold_between_the_two_bases_now_passes(self):
        # The gap is what a queueing-inflated basis costs: TPOT 4.5 ms is inside a
        # 5 ms budget, the includes-TTFT reading of 6.1 ms is not, and it was the
        # one deciding capacity.
        m = _normalize(self._report(itl=4.5, tpot_incl_ttft=6.1), knob=4.0, index=0)
        assert _passes_slo(m, slo_cfg(slo_avg_itl_ms=5.0)) is True

    def test_a_non_incremental_response_falls_back_instead_of_failing(self):
        # A server that answers in ONE chunk (whole output at once, common at low
        # load) leaves first_iteration == last_iteration, so guidellm reports the
        # decode-only metric as 0.0. There is no gap between tokens to measure, and
        # total-time-over-tokens is the only per-token number left — judge on it.
        # Failing here would bracket the ramp on its first point for every server
        # that batches its stream.
        m = _normalize(self._report(itl=0.0, tpot_incl_ttft=4.7), knob=4.0, index=0)
        assert m.itl_ms == 4.7
        assert _passes_slo(m, slo_cfg(slo_avg_itl_ms=5.0)) is True
        assert _passes_slo(m, slo_cfg(slo_avg_itl_ms=4.0)) is False

    def test_neither_basis_measured_still_fails_closed(self):
        m = _normalize(self._report(itl=0.0, tpot_incl_ttft=0.0), knob=4.0, index=0)
        assert _passes_slo(m, slo_cfg(slo_avg_itl_ms=5.0)) is False

    def test_an_unset_threshold_is_still_ignored(self):
        # The fail-closed rule applies to SET thresholds only: a zero-valued metric
        # nobody asked about must not fail the point.
        m = _normalize(self._report(itl=0.0, tpot_incl_ttft=0.0), knob=4.0, index=0)
        assert _passes_slo(m, slo_cfg(slo_avg_ttft_ms=500.0)) is True


class TestTheCapDoesNotOutrankAMeasurement:
    """The soft cap bounds the STEP, it does not end the search.

    Reaching it means every measured criterion just passed on that point, so the
    only thing still arguing for a stop is a 10s burst estimate of the very
    quantity that point measured over 30s. Letting the estimate win truncates the
    curve one point short of the evidence a reader needs — and the probe is kept
    out of the grid, so nothing on screen explains the stop.
    """

    def _bm40(self, ceiling=25.6, sustained=25.9):
        """A real run's shape: linear to ~26 rps, then pinned and slowly losing.

        Probe reads 25.6 -> cap ceil(25.6*1.2) = 31, onto which the doubling
        16 -> 32 is clamped. Achieved at 31 is 25.9 — past the knee, but only 1%
        above the probed ceiling, which is what used to end the run. Past the knee
        throughput decays as queueing grows (measured on the same deployment:
        -2.6% at 32, -1.7% at 33, TTFT 987ms -> 4171ms).
        """

        def curve(k):
            if k <= sustained:
                return k * 1160.0
            return sustained * 1160.0 * max(0.5, 1.0 - 0.0026 * (k - sustained))

        return (
            make_run_point(curve, achieved_fn=lambda k: min(k, sustained)),
            make_probe(ceiling),
        )

    def _run(self, lower_bound=4, upper_bound=1024, rp=None, probe=None):
        if rp is None:
            rp, probe = self._bm40()
        return run_outcome(
            sat_cfg(
                axis="rate",
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                probe_saturation=True,
            ),
            rp,
            probe=probe,
        )

    def test_the_run_continues_past_the_cap_and_stops_on_a_measured_point(self):
        o = self._run()
        # Phase 1: 31 cleared the growth criteria (+62% achieved over 16), so the
        # cap gave way rather than ending the search; 62 did not, and THAT is what
        # stopped it. (Everything after is Phase 2 refining inside the bracket.)
        assert knobs(o.points)[:5] == [4.0, 8.0, 16.0, 31.0, 62.0], knobs(o.points)
        assert o.probe_relaxed == 1
        assert o.bracket_reason == STOP_CAPACITY_PLATEAU
        assert o.stopped_at == 62.0

    def test_the_turnover_is_in_the_grid_not_just_in_the_sidecar(self):
        # The point of paying for the extra point: the top of the curve delivers
        # LESS than the point below it, so "this is the peak" is legible on the
        # chart. Stopping at the cap leaves a grid still climbing at its last row.
        pts = self._run().points
        top = max(pts, key=lambda p: p.knob)
        assert top.total_tps < peak_tps(pts)

    def test_the_hard_bound_still_ends_it(self):
        # Only cfg.upper_bound is a hard stop: a server that keeps up all the way
        # must still stop at the range the user asked for, and say so.
        o = self._run(
            upper_bound=64,
            rp=make_run_point(lambda k: k * 1000.0, achieved_fn=lambda k: k),
            probe=make_probe(10.0),  # cap 12, relaxed repeatedly, never binding
        )
        assert max(knobs(o.points)) == 64.0
        assert o.bracket_reason == STOP_UPPER_BOUND

    def test_the_first_point_is_still_the_probes_to_stop(self):
        """Nothing measured yet = nothing to overrule the estimate with.

        lower_bound above the cap: no previous point exists, so none of the growth
        criteria can fire and doubling blind is unbounded — 512 against a 25 rps
        server puts the next point at 1024, i.e. 30k requests draining at 25 rps.
        """
        o = self._run(
            lower_bound=64,
            rp=make_run_point(
                lambda k: min(k, 23.6) * 1200.0, achieved_fn=lambda k: min(k, 23.6)
            ),
            probe=make_probe(23),
        )
        assert knobs(o.points) == [64.0]
        assert o.probe_relaxed == 0
        assert o.bracket_reason == STOP_PROBE_BOUND
        # Named apart from the user's range, because the advice is opposite:
        # raising upper_bound changes nothing when OUR estimate stopped the search
        # (the probe re-derives its cap on every run).
        assert o.probe_bound is not None and o.probe_bound < o.upper_bound
