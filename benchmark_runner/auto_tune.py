"""
Adaptive ramp auto-tune engine.

A benchmark task = automatically probe ONE deterministic answer (peak throughput
OR the SLA-capacity boundary) instead of asking the user to guess a load value.
The engine orchestrates repeated single-strategy guidellm runs (each one knob
point) and reads the metrics back to decide the next point:

    Phase 1  geometric bracket   (knob *= 2 until a stop criterion trips)
    Phase 2  binary search       (SLA target: bisect the pass/fail bracket for the
                                  max knob within SLA; saturation target: bisect the
                                  last doubling gap for the throughput argmax)

Each ramp point is ONE ``benchmark_generative_text`` invocation with a single
``concurrent``/``constant`` strategy, a ``max_requests`` constraint, and a
DISTINCT random seed (defeats prefix/KV cache reuse across points). Each
point writes its own dual_json pair ``{base}__p{index}.json`` /
``{base}__p{index}.full.json`` which the gpustack manager globs.

This module is invoked from ``benchmark_runner.main`` when ``--auto-tune`` is set;
non-auto-tune modes (``--profile`` / ``--stages`` / ``--rate``) are untouched.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from guidellm.benchmark import BenchmarkScenario
from guidellm.benchmark.entrypoints import benchmark_generative_text

from benchmark_runner.scenario_builder import build_scenario_args

# 95% success floor shared by both stop targets. Peak probing uses
# a loose floor (not 100%) so a stray timeout doesn't prematurely cut the peak.
SUCCESS_FLOOR = 0.95
# Throughput plateau threshold: <5% gain over previous point = saturated.
PLATEAU_GAIN = 0.05
# Max concurrency cap for the rate axis so an over-offered rate can't pile up
# unbounded in-flight requests and drag the server down.
DEFAULT_RATE_MAX_CONCURRENCY = 512


@dataclass
class PointMetrics:
    """Normalized metrics for one measured knob point.

    All latency-family fields are stored in MILLISECONDS so they compare directly
    to the ms-denominated SLA thresholds. TTFT/TPOT are already ms in guidellm;
    request latency is SECONDS in guidellm and is converted to ms here (x1000).
    """

    knob: float
    index: int
    output_tps: float  # metrics.output_tokens_per_second.successful.mean
    ttft_ms: float  # metrics.time_to_first_token_ms.successful.mean
    ttft_p99_ms: float  # ...time_to_first_token_ms.successful.percentiles.p99
    tpot_ms: float  # metrics.time_per_output_token_ms.successful.mean
    tpot_p99_ms: float  # ...time_per_output_token_ms.successful.percentiles.p99
    latency_ms: float  # metrics.request_latency.successful.mean * 1000 (s -> ms)
    latency_p99_ms: float  # ...request_latency.successful.percentiles.p99 * 1000
    achieved_rate: float  # metrics.requests_per_second.successful.mean
    success: float  # successful / total


@dataclass
class AutoTuneConfig:
    """Ramp engine inputs (all supplied via CLI).

    SLA is a set of up to 6 OPTIONAL "<=" latency thresholds (all in ms): avg + p99
    of TTFT, TPOT, and end-to-end latency. Any subset may be set; the target becomes
    "sla" iff at least one is set, and a point passes iff every SET threshold holds.
    """

    axis: str  # "rate" | "concurrency"
    lower_bound: float = 1.0
    upper_bound: float = 1024.0
    multiplier: Optional[float] = None  # default resolved by axis (10 conc / 30 rate)
    min_requests: int = 30
    max_points: int = 12
    max_total_seconds: float = 1800.0
    # SLA thresholds (all optional, all in ms, all "<=" comparisons).
    sla_avg_ttft_ms: Optional[float] = None
    sla_p99_ttft_ms: Optional[float] = None
    sla_avg_tpot_ms: Optional[float] = None
    sla_p99_tpot_ms: Optional[float] = None
    sla_avg_latency_ms: Optional[float] = None
    sla_p99_latency_ms: Optional[float] = None
    random_seed_base: int = 42
    # True: each point's seed = base + index (points differ, spreading prefix/KV
    # cache reuse). False: every point uses the base seed.
    seed_increment: bool = True

    @property
    def resolved_multiplier(self) -> float:
        if self.multiplier is not None:
            return self.multiplier
        # concurrency: requests-per-slot; rate: seconds of test time.
        return 10.0 if self.axis == "concurrency" else 30.0

    def sla_pairs(self, m: PointMetrics) -> list[tuple[Optional[float], float]]:
        """(threshold, metric_value) pairs for the 6 SLA dimensions.

        Threshold is None when unset (ignored by the pass predicate). All values
        are in ms, aligned with the thresholds.
        """
        return [
            (self.sla_avg_ttft_ms, m.ttft_ms),
            (self.sla_p99_ttft_ms, m.ttft_p99_ms),
            (self.sla_avg_tpot_ms, m.tpot_ms),
            (self.sla_p99_tpot_ms, m.tpot_p99_ms),
            (self.sla_avg_latency_ms, m.latency_ms),
            (self.sla_p99_latency_ms, m.latency_p99_ms),
        ]

    @property
    def target(self) -> str:
        """ "sla" if ANY of the 6 SLA thresholds is set, else "saturation"."""
        any_set = any(
            t is not None
            for t in (
                self.sla_avg_ttft_ms,
                self.sla_p99_ttft_ms,
                self.sla_avg_tpot_ms,
                self.sla_p99_tpot_ms,
                self.sla_avg_latency_ms,
                self.sla_p99_latency_ms,
            )
        )
        return "sla" if any_set else "saturation"


def _mean(node: Any, *path: str) -> float:
    """Safely walk ``node.a.b.c`` returning 0.0 on any missing attribute/None."""
    cur = node
    for attr in path:
        cur = getattr(cur, attr, None)
        if cur is None:
            return 0.0
    return float(cur or 0.0)


def _normalize(benchmark: Any, knob: float, index: int) -> PointMetrics:
    """Map a guidellm ``benchmarks[0]`` result to our flat PointMetrics.

    Note: ``request_latency`` is in SECONDS in guidellm; the SLA thresholds are in
    ms, so its mean/p99 are multiplied by 1000 here. TTFT/TPOT are already ms.
    """
    m = benchmark.metrics
    totals = m.request_totals
    total = getattr(totals, "total", 0) or 0
    successful = getattr(totals, "successful", 0) or 0
    success = (successful / total) if total else 0.0
    return PointMetrics(
        knob=knob,
        index=index,
        output_tps=_mean(m, "output_tokens_per_second", "successful", "mean"),
        ttft_ms=_mean(m, "time_to_first_token_ms", "successful", "mean"),
        ttft_p99_ms=_mean(
            m, "time_to_first_token_ms", "successful", "percentiles", "p99"
        ),
        tpot_ms=_mean(m, "time_per_output_token_ms", "successful", "mean"),
        tpot_p99_ms=_mean(
            m, "time_per_output_token_ms", "successful", "percentiles", "p99"
        ),
        # request_latency is seconds -> convert to ms to match the SLA thresholds.
        latency_ms=_mean(m, "request_latency", "successful", "mean") * 1000.0,
        latency_p99_ms=_mean(m, "request_latency", "successful", "percentiles", "p99")
        * 1000.0,
        achieved_rate=_mean(m, "requests_per_second", "successful", "mean"),
        success=success,
    )


def _passes_sla(m: PointMetrics, cfg: AutoTuneConfig) -> bool:
    """SLA-pass = success>=95% AND every SET threshold holds (<=).

    AND is taken over SET thresholds only; unset (None) thresholds are ignored.
    Up to 6 dimensions: avg+p99 of TTFT, TPOT, and end-to-end latency (all ms).
    """
    if m.success < SUCCESS_FLOOR:
        return False
    for threshold, value in cfg.sla_pairs(m):
        if threshold is not None and value > threshold:
            return False
    return True


# A run_point callable executes ONE guidellm benchmark for (knob, index, seed)
# and returns its single benchmark result object (report.benchmarks[0]).
RunPointFn = Callable[[float, int, int], Awaitable[Any]]


async def run_ramp(  # noqa: C901
    cfg: AutoTuneConfig,
    base_kwargs: dict[str, Any],
    output_base: str,
    server_progress: Any = None,
    progress: Any = None,
    console: Any = None,
) -> list[PointMetrics]:
    """
    Execute the adaptive ramp and return the measured points.

    :param cfg: Ramp configuration (axis, bounds, budget, SLA).
    :param base_kwargs: Common guidellm kwargs shared by every point (target, data,
        backend, processor, ...). Per-point profile/rate/seed/outputs/max_requests
        are layered on top for each run.
    :param output_base: Dual_json base id (from --outputs, e.g. "123"); each point
        writes ``{output_base}__p{index}.dual_json`` -> ``__p{index}.json`` + full.
    """
    start = time.monotonic()

    def _elapsed() -> float:
        return time.monotonic() - start

    def _budget_ok(points_done: int) -> bool:
        return points_done < cfg.max_points and _elapsed() < cfg.max_total_seconds

    def _prep_progress(index: int, remaining_est: int) -> None:
        # Smooth (within-point) server progress: guidellm's per-point
        # on_benchmark_update callbacks fire continuously during a run, computing
        # overall = (run_index + point_fraction) / run_total. Setting run_index to
        # the count of completed points maps this point's live fraction onto its
        # slice of the overall bar, so the bar creeps instead of jumping once per
        # point. run_total is kept >= index + 2 so on_benchmark_complete can never
        # reach 100 (the ramp pushes the final 100 itself); remaining_est (points
        # left, incl. this one) shrinks as the sweep nears the end, keeping the bar
        # roughly proportional. The count is adaptive, so this is an estimate.
        if server_progress is None:
            return
        server_progress.run_index = index
        server_progress.run_total = index + max(2, remaining_est)

    def _phase1_remaining(knob: float) -> int:
        # This point + remaining doublings to the upper bound + a few Phase-2 steps.
        doublings = max(
            0, math.floor(math.log2(max(cfg.upper_bound / max(knob, 1e-9), 1.0)))
        )
        return 1 + doublings + 3

    def _phase2_remaining(lo: float, hi: float) -> int:
        # Bisection steps left to close the (lo, hi) bracket.
        return max(1, math.ceil(math.log2(max(2.0, hi - lo))))

    async def _run_point(knob: float, index: int) -> Optional[PointMetrics]:
        # number = max(min_requests, round(knob * multiplier)) -> max_requests
        number = max(cfg.min_requests, round(knob * cfg.resolved_multiplier))
        # Per-point seed: increment by index so points differ, unless the
        # user pinned a fixed seed (seed_increment=False → same seed each point).
        seed = (
            cfg.random_seed_base + index if cfg.seed_increment else cfg.random_seed_base
        )

        local = dict(base_kwargs)
        local["random_seed"] = seed
        local["max_requests"] = number
        local["outputs"] = [f"{output_base}__p{index}.dual_json"]
        if cfg.axis == "concurrency":
            local["profile"] = "concurrent"
            local["rate"] = [float(int(knob))]  # streams = int(knob)
        else:
            local["profile"] = "constant"
            local["rate"] = [float(knob)]
            # Cap in-flight requests on the (open-loop) rate axis. guidellm
            # 0.7.1's flat args expose this via backend_kwargs' max concurrency is
            # not surfaced; we approximate by not over-offering beyond upper_bound.

        args = _build_args(local)
        # max_requests is carried in the scenario spec.constraints (built from
        # local["max_requests"]); no need to also pass it as a **constraints kwarg.
        report, _ = await benchmark_generative_text(
            args=args, progress=progress, console=console
        )
        if not report.benchmarks:
            return None
        return _normalize(report.benchmarks[0], knob, index)

    points: list[PointMetrics] = []
    knob = float(cfg.lower_bound)
    prev_tps: Optional[float] = None
    prev_achieved: Optional[float] = None
    prev_knob: Optional[float] = None  # last knob whose tps still improved
    last_pass: Optional[float] = None
    first_fail: Optional[float] = None
    # Saturation bracket for the Phase-2 peak search: (lo, hi) where lo is the
    # last still-improving knob (holds the running-best tps) and hi is the knob
    # that stopped improving / overloaded. Left None when the sweep hit the upper
    # bound while still climbing (peak is at the bound; nothing above to search).
    sat_bracket: Optional[tuple[float, float]] = None

    # ── Phase 1: geometric bracket ──────────────────────────────────────────
    while _budget_ok(len(points)):
        _prep_progress(len(points), _phase1_remaining(knob))
        m = await _run_point(knob, len(points))
        if m is None:
            break
        points.append(m)

        if cfg.target == "sla":
            if _passes_sla(m, cfg):
                last_pass = knob
                if knob >= cfg.upper_bound:
                    break
                knob *= 2
            else:
                first_fail = knob  # bracket = (last_pass, first_fail)
                break
        else:  # saturation (peak throughput)
            overloaded = m.success < SUCCESS_FLOOR
            # On the rate axis, saturation = raising the offered rate no longer
            # buys proportionally more ACHIEVED throughput. We must compare growth
            # between consecutive points, NOT a single-point achieved-vs-offered
            # ratio: with a finite max_requests the drain tail biases achieved_rate
            # a ~constant fraction below offered (achieved ≈ offered/(1 + latency/
            # window)) even on an idle server, so an absolute floor false-trips on
            # the very first, unsaturated point.
            cant_keepup = (
                cfg.axis == "rate"
                and prev_achieved is not None
                and prev_achieved > 0
                and (m.achieved_rate / prev_achieved - 1.0) < PLATEAU_GAIN
            )
            plateau = (
                prev_tps is not None
                and prev_tps > 0
                and (m.output_tps / prev_tps - 1.0) < PLATEAU_GAIN
            )
            if overloaded or cant_keepup or plateau:
                # Throughput stopped climbing (or the server overloaded): the peak
                # sits in the last doubling gap, between the last still-improving
                # knob and this one. Best is at prev_knob (this point grew < 5% or
                # dropped), so the Phase-2 search keeps best pinned to the left end.
                if prev_knob is not None:
                    sat_bracket = (prev_knob, knob)
                break
            prev_tps = m.output_tps
            prev_achieved = m.achieved_rate
            prev_knob = knob
            if knob >= cfg.upper_bound:
                break
            knob *= 2

    # ── Phase 2: binary search (SLA target only) ────────────────────────────
    if cfg.target == "sla" and last_pass is not None and first_fail is not None:
        lo, hi = last_pass, first_fail
        while hi - lo > 1 and _budget_ok(len(points)):
            mid = math.floor((lo + hi) / 2)
            if mid <= lo:
                break
            _prep_progress(len(points), _phase2_remaining(lo, hi))
            m = await _run_point(float(mid), len(points))
            if m is None:
                break
            points.append(m)
            if _passes_sla(m, cfg):
                lo = float(mid)
            else:
                hi = float(mid)

    # ── Phase 2: peak-seeking binary search (saturation target only) ─────────
    # The goal is the single knob that MAXIMISES throughput, not a dense curve.
    # Phase 1 doubles, so the peak sits inside the last 2x gap [lo, hi]; bisect it
    # and walk toward whichever side improves — the same argmax bisection evalscope
    # uses (SLAAutoTuner._tune_optimization). This assumes throughput is unimodal
    # in the knob (rises, then plateaus/drops past saturation); ``lo`` starts as
    # the running best so ties keep the CHEAPER (lower) knob at equal throughput.
    if cfg.target == "saturation" and sat_bracket is not None:
        lo, hi = sat_bracket
        best_tps = prev_tps if prev_tps is not None else -1.0
        measured = {p.knob for p in points}
        while hi - lo > 1 and _budget_ok(len(points)):
            mid = math.floor((lo + hi) / 2)
            if mid <= lo or float(mid) in measured:
                break
            _prep_progress(len(points), _phase2_remaining(lo, hi))
            m = await _run_point(float(mid), len(points))
            if m is None:
                break
            points.append(m)
            measured.add(float(mid))
            if m.output_tps > best_tps:
                best_tps = m.output_tps  # still climbing -> peak is at/above mid
                lo = float(mid)
            else:
                hi = float(mid)  # mid no better -> peak is below mid

    # Converged: push progress to 100 once (server clamps monotonically), then
    # close the session — the ramp owns server_progress' lifecycle here (it is
    # kept out of the per-point guidellm runs, so guidellm never finalizes it).
    if server_progress is not None:
        try:
            await server_progress._update_progress(100.0)
        except Exception:
            pass
        try:
            await server_progress.on_finalize()
        except Exception:
            pass

    return points


def _build_args(local_kwargs: dict[str, Any]) -> BenchmarkScenario:
    """Build a guidellm ``BenchmarkScenario`` for one ramp point.

    Mirrors main.py's ``_run_once``: delegates to the shared spec builder, which
    maps the flat CLI kwargs onto ``spec`` (backend/profile/data/seed/outputs/
    constraints) and normalizes data sources (e.g. ShareGPT -> guidellm jsonl).
    The per-point profile ({kind: concurrent, streams:[N]} for the concurrency
    axis, {kind: constant, rate:[R]} for the rate axis), the per-point seed, and
    the ``max_requests`` constraint are already set on ``local_kwargs`` by the
    caller.
    """
    return build_scenario_args(local_kwargs)
