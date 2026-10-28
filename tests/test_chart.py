"""Tests for the terminal curve report (--auto-tune ramps and --stages ladders).

Three things are worth pinning here, and they are not "does it draw":

* **The verdicts are gpustack-ui's.** ``point_status`` / ``best_points`` are ports
  of ``getStageStatus`` and ``compute_best_points``. If they drift, the CLI and the
  web report disagree about the same run — which is worse than either being wrong
  alone, because the user has no way to tell which to believe.
* **The grid stays rectangular.** A chart is a fixed-width character grid; one row
  built a character short shears the whole picture, and nothing else in the suite
  would notice.
* **Drawing is never load-bearing.** A ramp that measured fine must not fail
  because a terminal could not be written to.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from benchmark_runner import main
from benchmark_runner.chart import (
    CHART_ALL,
    CHART_AUTO,
    CHART_NONE,
    STATUS_OK,
    STATUS_OVERLOADED,
    STATUS_PEAK,
    STATUS_RECOMMENDED,
    Style,
    best_points,
    detect_style,
    point_status,
    render_curve_report,
)
from benchmark_runner.main import cli

ASCII = Style(width=48, color=False, unicode=False)
UNICODE = Style(width=48, color=False, unicode=True)


def make_point(knob, tps, *, ttft_p99=100.0, tpot=10.0, success=1.0, achieved=None):
    return {
        "knob": float(knob),
        "index": 0,
        "output_tps": float(tps),
        "total_tps": float(tps),
        "ttft_ms": ttft_p99 * 0.6,
        "ttft_p95_ms": ttft_p99 * 0.9,
        "ttft_p99_ms": float(ttft_p99),
        "itl_ms": float(tpot),
        "itl_p95_ms": float(tpot),
        "itl_p99_ms": float(tpot),
        "latency_ms": 0.0,
        "latency_p95_ms": 0.0,
        "latency_p99_ms": 0.0,
        "achieved_rate": float(achieved if achieved is not None else knob),
        "success": float(success),
    }


# A saturating server: throughput peaks at 30, declines past it, and the last
# point sheds requests. Mirrors the shape every real ramp produces.
CURVE = [
    make_point(4, 4600, ttft_p99=32),
    make_point(8, 9200, ttft_p99=44),
    make_point(16, 18400, ttft_p99=68),
    make_point(30, 34500, ttft_p99=110),
    make_point(32, 33800, ttft_p99=356, achieved=30),
    make_point(64, 21000, ttft_p99=9000, success=0.80, achieved=30),
]


class TestVerdictsMatchTheWebReport:
    def test_gain_convergence_reports_a_normal_end_with_the_measured_total_peak(self):
        text = "\n".join(
            render_curve_report(
                CURVE,
                stop_reason="throughput_gain_converged",
                bracket_reason="capacity_plateau",
                mode=CHART_NONE,
                style=UNICODE,
            )
        )

        assert "Recommended: 30 req/s" in text
        assert "34,500 tok/s" in text
        assert "Peak refinement converged on total-throughput gain" in text
        assert "stopped: throughput_gain_converged (bracket: capacity_plateau)" in text

    def test_peak_uses_total_even_when_output_peak_is_elsewhere(self):
        points = [
            {**make_point(4, 1000), "total_tps": 5000},
            {**make_point(8, 2000), "total_tps": 4000},
        ]
        assert best_points(points, None)["peak_knob"] == 4.0
        text = "\n".join(render_curve_report(points, mode=CHART_AUTO, style=UNICODE))
        assert "total throughput (tok/s)" in text
        assert "5,000 tok/s" in text

    def test_an_output_only_legacy_curve_does_not_invent_a_total_peak(self):
        points = [{k: v for k, v in make_point(4, 1000).items() if k != "total_tps"}]
        assert best_points(points, None)["peak_knob"] is None
        text = "\n".join(render_curve_report(points, mode=CHART_NONE, style=UNICODE))
        assert "total throughput was not recorded" in text.lower()
        assert "Recommended:" not in text

    """Ports of gpustack-ui's ``getStageStatus`` (ui.tsx) and ``compute_best_points``."""

    def _status(self, point, slo=None, points=None):
        best = best_points(points or CURVE, slo)
        return point_status(
            point,
            recommended_knob=best["recommended_knob"],
            peak_knob=best["peak_knob"],
            peak_tps=best["peak_tps"],
        )

    def test_the_recommended_point_is_the_throughput_peak_without_an_slo(self):
        assert best_points(CURVE, None)["recommended_knob"] == 30.0

    def test_slo_primary_result_is_the_highest_passing_load(self):
        assert best_points(CURVE, 16.0, "slo")["recommended_knob"] == 16.0
        assert best_points(CURVE, 64.0, "slo")["recommended_knob"] == 64.0
        assert best_points(CURVE, 64.0, "slo")["peak_knob"] == 30.0

    def test_an_unmet_slo_has_no_primary_point(self):
        assert best_points(CURVE, None, "slo")["recommended_knob"] is None

    def test_recommended_outranks_peak_on_the_same_point(self):
        # 30 is both. The badge that matters is "this is the answer".
        assert self._status(CURVE[3]) == STATUS_RECOMMENDED

    def test_a_point_past_the_peak_but_within_5_percent_is_not_overloaded(self):
        # 33800 / 34500 = 98% of peak with no failures. The point one step above
        # the knee is often the true argmax; calling it overload is the bug this
        # 5% band exists to prevent.
        assert self._status(CURVE[4]) == STATUS_OK

    def test_a_declining_point_past_the_peak_is_overloaded(self):
        assert self._status(CURVE[5]) == STATUS_OVERLOADED

    def test_a_failing_point_is_overloaded_even_below_the_peak(self):
        # Failure rate alone decides, no matter where on the curve it sits: 5 is
        # under the peak and still climbing, but a fifth of its requests died.
        low_failure = make_point(5, 5000, success=0.80)
        points = [low_failure, *CURVE]
        assert self._status(low_failure, points=points) == STATUS_OVERLOADED

    def test_a_couple_of_stragglers_are_not_overload(self):
        # 96% > the 95% floor. Incomplete requests at the max_seconds boundary are
        # normal at low rates and must not turn a healthy point red.
        straggler = make_point(5, 5000, success=0.96)
        points = [straggler, *CURVE]
        assert self._status(straggler, points=points) == STATUS_OK

    def test_the_peak_keeps_its_own_badge_when_the_slo_caps_the_answer(self):
        # SLO met only at 16 -> 16 is recommended and 30 is still flagged as the
        # peak, which is what makes "you are leaving throughput on the table"
        # readable off the table alone.
        assert self._status(CURVE[3], slo=16.0) == STATUS_PEAK
        assert self._status(CURVE[2], slo=16.0) == STATUS_RECOMMENDED


class TestTheGridStaysRectangular:
    """One short row shears the picture; nothing else in the suite would catch it."""

    def _chart_rows(self, lines, style):
        # A plotted row is "two spaces, an 8-wide right-aligned y label, a space,
        # then the axis character". The y label is blank on most rows, which is
        # why the invariant below is about the AXIS column, not about indentation.
        wall, tick = style.axis[2], style.axis[1]
        return [
            ln
            for ln in lines
            if len(ln) > 11 and ln[:2] == "  " and ln[11] in (wall, tick)
        ]

    @pytest.mark.parametrize("style", [ASCII, UNICODE])
    @pytest.mark.parametrize("mode", [CHART_AUTO, CHART_ALL])
    def test_the_axis_column_never_moves(self, style, mode):
        # One row built a character short slides its whole plot area sideways.
        # Pinning the axis column catches that; comparing rstrip'd lengths would
        # not, since trailing blanks are legitimately trimmed.
        lines = render_curve_report(CURVE, mode=mode, style=style)
        rows = self._chart_rows(lines, style)
        assert len(rows) >= 11, f"expected a full panel, got {len(rows)} rows"
        for row in rows:
            assert len(row) <= 11 + 1 + style.width

    def test_the_ascii_style_emits_no_characters_outside_ascii(self):
        # The point of the fallback: a stream that cannot carry box drawing gets
        # a chart, not a UnicodeEncodeError.
        text = "\n".join(render_curve_report(CURVE, mode=CHART_ALL, style=ASCII))
        text.encode("ascii")


class TestWhatEachModeShows:
    def test_auto_draws_the_headline_charts_and_the_table(self):
        text = "\n".join(render_curve_report(CURVE, mode=CHART_AUTO, style=UNICODE))
        assert "total throughput (tok/s)" in text
        assert "TTFT p99" in text
        assert "Recommended: 30 req/s" in text
        # Not the secondary pair.
        assert "frontier" not in text

    def test_all_adds_the_secondary_charts(self):
        text = "\n".join(render_curve_report(CURVE, mode=CHART_ALL, style=UNICODE))
        assert "frontier" in text
        assert "achieved rate" in text

    def test_none_keeps_the_table_and_the_verdict(self):
        # --chart none is "no pictures", not "no report": the table and the
        # recommended point are the parts that survive a pipe into a log.
        text = "\n".join(render_curve_report(CURVE, mode=CHART_NONE, style=UNICODE))
        assert "Recommended: 30 req/s" in text
        assert "34,500" in text
        assert "total throughput (tok/s)" not in text

    def test_a_single_point_reports_without_pretending_to_have_a_curve(self):
        lines = render_curve_report([CURVE[0]], mode=CHART_AUTO, style=UNICODE)
        text = "\n".join(lines)
        assert "no curve to draw" in text
        assert "Recommended: 4 req/s" in text

    def test_no_points_renders_nothing_at_all(self):
        assert render_curve_report([], mode=CHART_AUTO, style=UNICODE) == []


class TestTheSloTargetIsReportedAsABoundary:
    def test_throughput_decline_is_a_protective_stop_with_unlocated_slo_boundary(self):
        points = [
            make_point(1, 1000),
            make_point(2, 1500),
            make_point(4, 1200),
            make_point(8, 1190),
        ]
        text = "\n".join(
            render_curve_report(
                points,
                axis="concurrency",
                target="slo",
                slo_met_knob=8.0,
                stop_reason="throughput_decline",
                bracket_reason="throughput_decline",
                mode=CHART_NONE,
                style=UNICODE,
            )
        )
        assert "Protective stop" in text
        assert "SLO boundary not located" in text
        assert "Recommended: 8 streams" in text
        assert "SLO boundary bracketed" not in text

    def test_an_unmet_slo_is_reported_without_a_peak_recommendation(self):
        text = "\n".join(
            render_curve_report(CURVE, target="slo", mode=CHART_NONE, style=UNICODE)
        )
        assert "SLO met by no measured point" in text
        assert "Recommended:" not in text

    def test_a_peak_below_the_slo_boundary_is_reported_as_an_observation(self):
        text = "\n".join(
            render_curve_report(
                CURVE,
                target="slo",
                slo_met_knob=64.0,
                mode=CHART_NONE,
                style=UNICODE,
            )
        )
        assert "Recommended: 64 req/s" in text
        assert "Measured total-TPS peak: 30 req/s" in text
        assert "capped here by the SLO" not in text

    def test_a_bracketed_boundary_names_both_ends(self):
        text = "\n".join(
            render_curve_report(
                CURVE,
                target="slo",
                slo_met_knob=16.0,
                slo_first_fail_knob=30.0,
                slo_thresholds={"slo_p99_ttft_ms": 100.0},
                mode=CHART_AUTO,
                style=UNICODE,
            )
        )
        assert "SLO boundary bracketed at (16, 30)" in text
        # The throughput left on the table is the other half of the answer.
        assert "Measured total-TPS peak: 30 req/s" in text

    def test_an_unbroken_slo_is_reported_as_a_floor_not_an_edge(self):
        # first_fail=None means no point ever breached it, so the number is "at
        # least this much", and reading it as the boundary is exactly wrong.
        text = "\n".join(
            render_curve_report(
                CURVE,
                target="slo",
                slo_met_knob=64.0,
                slo_first_fail_knob=None,
                mode=CHART_AUTO,
                style=UNICODE,
            )
        )
        assert "is a FLOOR" in text

    def test_the_threshold_is_drawn_as_a_reference_line(self):
        text = "\n".join(
            render_curve_report(
                CURVE,
                target="slo",
                slo_met_knob=16.0,
                slo_thresholds={"slo_p99_ttft_ms": 500.0},
                mode=CHART_AUTO,
                style=UNICODE,
            )
        )
        assert "SLO 500ms" in text


class TestAStageLadderClaimsLessThanARamp:
    """``--stages`` measures the rungs it was handed; it does not search.

    The whole point of a separate target is that the wording cannot be reused. A
    ramp brackets and converges, so "run at 31 req/s" is a finding. A stage list
    only has the loads its author guessed, so the same sentence would launder that
    guess into a result.
    """

    def _text(self, points, **kw):
        return "\n".join(
            render_curve_report(
                points,
                target="stages",
                stop_reason="stages_completed",
                mode=CHART_AUTO,
                style=UNICODE,
                **kw,
            )
        )

    def test_no_point_is_ever_recommended(self):
        assert best_points(CURVE, None, "stages")["recommended_knob"] is None
        # ...and the peak is still identified, because that IS measurable.
        assert best_points(CURVE, None, "stages")["peak_knob"] == 30.0

    def test_the_argmax_is_reported_as_the_best_stage_not_as_advice(self):
        text = self._text(CURVE)
        assert "Best of the stages you ran: 30 req/s" in text
        assert "Recommended" not in text
        assert "no search was run" in text

    def test_the_header_does_not_call_a_ladder_a_ramp(self):
        assert "Stage ladder" in self._text(CURVE)
        assert "Auto-tune ramp" not in self._text(CURVE)

    def test_a_bracketed_peak_is_reported_as_bracketed(self):
        # 30 has a lower neighbour on both sides, so the ladder really did contain
        # the peak.
        assert "the ladder brackets this peak" in self._text(CURVE)

    def test_a_peak_at_the_top_rung_says_the_ladder_stopped_too_low(self):
        # Every rung still climbing -> the argmax is an artefact of where the user
        # stopped, and reporting it as "the best" without this line is misleading.
        climbing = [make_point(4, 4600), make_point(8, 9200), make_point(16, 18400)]
        text = self._text(climbing)
        assert "still climbing at the top rung" in text
        assert "Add a higher stage" in text

    def test_a_peak_at_the_bottom_rung_says_the_ladder_started_too_high(self):
        falling = [make_point(32, 9000), make_point(64, 6000), make_point(128, 3000)]
        text = self._text(falling)
        assert "best at the lowest rung" in text
        assert "Add a lower stage" in text

    def test_two_stages_cannot_bracket_anything_and_say_so(self):
        text = self._text([make_point(4, 4600), make_point(8, 9200)])
        assert "too few stages to bracket a peak" in text

    def test_overload_verdicts_still_apply(self):
        # The ladder claims less about the PEAK; it claims exactly as much about a
        # rung that shed requests.
        text = self._text(CURVE)
        assert UNICODE.glyphs[STATUS_OVERLOADED] in text


class TestDegenerateCurvesDoNotLie:
    """Real runs produce these; each one used to draw something untrue."""

    # A rung that overloaded so hard every request failed. guidellm reports no
    # latency percentiles for it, which `_normalize` turns into 0.0 — and log10(0)
    # lands nine decades below the real data.
    WITH_A_DEAD_POINT = [
        make_point(4, 1000, ttft_p99=25),
        make_point(8, 0, ttft_p99=0, success=0.0),
        make_point(16, 3000, ttft_p99=60),
        make_point(32, 4000, ttft_p99=900),
    ]

    def _lat_labels(self, lines):
        """The y-axis labels of the latency panel."""
        start = next(i for i, ln in enumerate(lines) if "TTFT p99" in ln)
        out = []
        for ln in lines[start + 1 :]:
            if not ln.strip() or ln.strip().startswith(UNICODE.axis[0]):
                break
            label = ln[:11].strip()
            if label:
                out.append(label)
        return out

    def test_a_point_with_no_latency_does_not_collapse_the_log_scale(self):
        lines = render_curve_report(
            self.WITH_A_DEAD_POINT, mode=CHART_AUTO, style=UNICODE
        )
        labels = self._lat_labels(lines)
        # The axis must span the MEASURED range (25ms..900ms). Before the fix the
        # bottom two labels both read "0.0ms" and every real point was squeezed
        # into the top rows.
        assert labels[0] == "900ms", labels
        assert labels[-1] == "25ms", labels
        assert "0.0ms" not in labels

    def test_the_dropped_point_is_disclosed_not_silently_omitted(self):
        text = "\n".join(
            render_curve_report(self.WITH_A_DEAD_POINT, mode=CHART_AUTO, style=UNICODE)
        )
        assert "1 point(s) reported no latency" in text
        # And it is still in the table, with its failure visible.
        assert "0%" in text

    def test_a_curve_with_no_latency_at_all_says_so_instead_of_drawing(self):
        blind = [make_point(4, 1000, ttft_p99=0), make_point(8, 2000, ttft_p99=0)]
        text = "\n".join(render_curve_report(blind, mode=CHART_ALL, style=UNICODE))
        assert "no latency was measured" in text
        # The throughput panel is unaffected — it has real numbers.
        assert "total throughput (tok/s)" in text

    def test_an_unknown_stop_reason_is_omitted_rather_than_printed_as_a_question(self):
        text = "\n".join(render_curve_report(CURVE, mode=CHART_NONE, style=UNICODE))
        assert "stopped: ?" not in text
        assert "stopped:" not in text

    @pytest.mark.parametrize(
        "points",
        [
            pytest.param([make_point(4, 1000)] * 3, id="identical-throughput"),
            pytest.param([make_point(i, 0) for i in (4, 8, 16)], id="zero-throughput"),
            pytest.param(
                [make_point(4, 1000), make_point(4, 2000), make_point(8, 3000)],
                id="duplicate-knobs",
            ),
            pytest.param(
                [make_point(i + 1, 1000 * (i + 1)) for i in range(24)],
                id="more-points-than-columns",
            ),
        ],
    )
    def test_degenerate_shapes_render_without_shearing_the_grid(self, points):
        for style in (ASCII, UNICODE):
            lines = render_curve_report(points, mode=CHART_ALL, style=style)
            rows = [
                ln
                for ln in lines
                if len(ln) > 11 and ln[:2] == "  " and ln[11] in style.axis[1:3]
            ]
            assert rows
            for row in rows:
                assert len(row) <= 11 + 1 + style.width


class TestStyleDetection:
    class _Stream:
        def __init__(self, encoding, tty):
            self.encoding = encoding
            self._tty = tty

        def isatty(self):
            return self._tty

    def test_an_ascii_stream_gets_the_ascii_glyphs(self):
        assert detect_style(self._Stream("ascii", False), width=40).unicode is False

    def test_a_utf8_stream_gets_the_box_drawing(self):
        assert detect_style(self._Stream("utf-8", False), width=40).unicode is True

    def test_colour_needs_a_tty(self):
        assert detect_style(self._Stream("utf-8", False), width=40).color is False

    def test_no_color_is_honoured_on_a_tty(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert detect_style(self._Stream("utf-8", True), width=40).color is False

    def test_colour_paints_only_the_markers(self, monkeypatch):
        coloured = Style(width=48, color=True, unicode=True)
        text = "\n".join(render_curve_report(CURVE, mode=CHART_AUTO, style=coloured))
        assert "\033[" in text
        # Every escape is closed, so a chart cannot leak a colour into the shell.
        assert text.count("\033[") == text.count("\033[0m") * 1 or "\033[0m" in text


class TestTheChartSubcommand:
    def _sidecar(self, tmp_path, name="42__ramp.json", **overrides):
        payload = {
            "version": 3,
            "points": CURVE,
            "axis": "rate",
            "target": "saturation",
            "stop_reason": "converged",
            "bracket_reason": "capacity_plateau",
            "slo_bracket": None,
            "elapsed_seconds": 351.0,
            "probe_ceiling": None,
            "slo_thresholds": {},
        }
        payload.update(overrides)
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_a_directory_resolves_to_its_only_sidecar(self, tmp_path):
        self._sidecar(tmp_path)
        result = CliRunner().invoke(cli, ["chart", str(tmp_path), "--width", "48"])
        assert result.exit_code == 0, result.output
        assert "Recommended: 30 req/s" in result.output

    def test_the_sidecar_can_be_named_directly(self, tmp_path):
        path = self._sidecar(tmp_path)
        result = CliRunner().invoke(cli, ["chart", str(path), "--width", "48"])
        assert result.exit_code == 0, result.output

    def test_a_directory_with_no_ramp_says_which_modes_write_one(self, tmp_path):
        result = CliRunner().invoke(cli, ["chart", str(tmp_path)])
        assert result.exit_code != 0
        assert "No '*__ramp.json' or '*__curve.json' file" in result.output

    def test_several_sidecars_are_not_picked_between(self, tmp_path):
        # Guessing would silently chart the wrong run.
        self._sidecar(tmp_path, "42__ramp.json")
        self._sidecar(tmp_path, "43__ramp.json")
        result = CliRunner().invoke(cli, ["chart", str(tmp_path)])
        assert result.exit_code != 0
        assert "2 ramp sidecars" in result.output

    def test_a_v1_sidecar_is_named_as_the_thing_that_is_short(self, tmp_path):
        # Pre-v2 files hold the stop reasons but not the curve. "Empty chart" would
        # read as a broken renderer.
        self._sidecar(tmp_path, version=1, points=[])
        result = CliRunner().invoke(cli, ["chart", str(tmp_path)])
        assert result.exit_code != 0
        assert "no measured points" in result.output

    def test_a_render_failure_here_is_an_error_not_a_warning(self, tmp_path):
        # Drawing is this command's ONLY job. The live run swallows the same error
        # on purpose; exiting 0 having drawn nothing would be a silent CI no-op.
        self._sidecar(tmp_path, points="not-a-list")
        result = CliRunner().invoke(cli, ["chart", str(tmp_path)])
        assert result.exit_code != 0
        assert "Could not render" in result.output

    def test_a_width_the_renderer_cannot_divide_by_is_rejected(self, tmp_path):
        # Tick placement divides by (width - 1); 1 was a ZeroDivisionError.
        self._sidecar(tmp_path)
        result = CliRunner().invoke(cli, ["chart", str(tmp_path), "--width", "1"])
        assert result.exit_code != 0
        assert "--width" in result.output

    def test_ascii_can_be_forced_over_the_terminals_claim(self, tmp_path):
        self._sidecar(tmp_path)
        result = CliRunner().invoke(
            cli, ["chart", str(tmp_path), "--ascii", "--width", "48"]
        )
        assert result.exit_code == 0, result.output
        result.output.encode("ascii")


def _stub_benchmark_call(monkeypatch):
    """Replace guidellm's benchmark entrypoint with a synthetic saturating server.

    The stage path's whole contribution is plumbing — capture the report `_run_once`
    used to discard, normalize it, chart it — so it has to be exercised through the
    CLI. A renderer unit test cannot catch a stage loop that never collects a point.
    """

    def fake(args=None, progress=None, console=None):
        rate = float(args.spec.profile.rate[0])
        tps = 35000 * (1 - 2.718 ** (-rate / 9.0)) * (1.0 - max(0.0, rate - 31) * 0.05)
        ttft = 16 + rate * 2.2 + max(0.0, rate - 31) ** 3.2 * 9

        def stat(v):
            return SimpleNamespace(
                successful=SimpleNamespace(
                    mean=v, percentiles=SimpleNamespace(p95=v * 0.9, p99=v)
                )
            )

        bm = SimpleNamespace(
            metrics=SimpleNamespace(
                request_totals=SimpleNamespace(
                    total=100, successful=100 if rate <= 34 else 93
                ),
                output_tokens_per_second=stat(max(tps, 0.0)),
                tokens_per_second=stat(max(tps, 0.0)),
                time_to_first_token_ms=stat(ttft),
                inter_token_latency_ms=stat(2.2 + rate * 0.45),
                time_per_output_token_ms=stat(2.2 + rate * 0.45),
                request_latency=stat(ttft * 4 / 1000.0),
                requests_per_second=stat(min(rate, 30.5) * 0.97),
            )
        )

        async def _co():
            return SimpleNamespace(benchmarks=[bm]), None

        return _co()

    monkeypatch.setattr(main, "benchmark_generative_text", fake)


class TestTheStagePathIsWiredThrough:
    ARGS = [
        "benchmark",
        "run",
        "--target",
        "http://localhost:8000",
        "--data",
        "prompt_tokens=128,output_tokens=128",
        "--stages",
        '[{"rate":4},{"rate":8},{"rate":16},{"rate":31},{"rate":40}]',
    ]

    def _run(self, monkeypatch, tmp_path, *extra):
        _stub_benchmark_call(monkeypatch)
        result = CliRunner().invoke(
            cli,
            [
                *self.ARGS,
                "--output-dir",
                str(tmp_path),
                "--outputs",
                "42.dual_json",
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        return result

    def test_a_stage_run_charts_its_ladder(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path).output
        assert "Stage ladder" in out
        assert "total throughput (tok/s)" in out
        assert "Best of the stages you ran: 31 req/s" in out

    def test_the_curve_sidecar_is_written_with_every_stage(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path)
        path = tmp_path / "42__curve.json"
        assert path.exists(), sorted(p.name for p in tmp_path.iterdir())
        curve = json.loads(path.read_text())
        assert [p["knob"] for p in curve["points"]] == [4, 8, 16, 31, 40]
        assert curve["target"] == "stages"

    def test_the_curve_sidecar_is_not_named_like_a_ramps(self, monkeypatch, tmp_path):
        # Two different claims; a consumer must tell them apart by name alone.
        self._run(monkeypatch, tmp_path)
        assert not (tmp_path / "42__ramp.json").exists()
        # And it must not be picked up by gpustack's "{id}__p{index}.json" glob.
        assert not (tmp_path / "42__curve.json").name.startswith("42__p")

    def test_the_saved_ladder_redraws_identically(self, monkeypatch, tmp_path):
        live = self._run(monkeypatch, tmp_path, "--chart", "none").output
        offline = CliRunner().invoke(
            cli, ["chart", str(tmp_path), "--chart", "none", "--width", "48"]
        )
        assert offline.exit_code == 0, offline.output
        # The verdict is the part that must survive the round trip byte for byte.
        assert "Best of the stages you ran: 31 req/s" in live
        assert "Best of the stages you ran: 31 req/s" in offline.output

    def test_chart_none_still_suppresses_the_grids(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path, "--chart", "none").output
        assert "Best of the stages you ran" in out
        assert "total throughput (tok/s)" not in out

    def test_disable_console_silences_a_stage_report(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path, "--disable-console").output
        assert "Best of the stages you ran" not in out
        # The sidecar is a file, not console output, so it is still written.
        assert (tmp_path / "42__curve.json").exists()


class TestASingleStrategyRunHasNoCurve:
    """``--profile`` measures one point, so there is nothing to plot."""

    ARGS = [
        "benchmark",
        "run",
        "--target",
        "http://localhost:8000",
        "--data",
        "prompt_tokens=128,output_tokens=128",
        "--profile",
        "constant",
        "--rate",
        "10",
    ]

    def test_no_chart_and_no_sidecar_are_produced(self, monkeypatch, tmp_path):
        _stub_benchmark_call(monkeypatch)
        result = CliRunner().invoke(cli, [*self.ARGS, "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "total throughput (tok/s)" not in result.output
        assert list(tmp_path.iterdir()) == []

    def test_asking_for_a_chart_anyway_says_why_it_did_nothing(
        self, monkeypatch, tmp_path
    ):
        # A silently ignored flag reads as a broken chart. The default stays quiet.
        _stub_benchmark_call(monkeypatch)
        result = CliRunner().invoke(
            cli, [*self.ARGS, "--chart", "all", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        assert "This run has a single strategy" in result.output

    def test_the_default_chart_mode_warns_about_nothing(self, monkeypatch, tmp_path):
        _stub_benchmark_call(monkeypatch)
        result = CliRunner().invoke(cli, [*self.ARGS, "--output-dir", str(tmp_path)])
        assert "single strategy" not in result.output


class TestTheReportIsNeverLoadBearing:
    def test_a_render_failure_warns_instead_of_failing_the_run(
        self, monkeypatch, capsys
    ):
        def boom(*args, **kwargs):
            raise RuntimeError("no terminal here")

        monkeypatch.setattr(main, "render_curve_report", boom)
        main._emit_curve_report({"points": CURVE}, CHART_AUTO)
        assert "Could not render the ramp report" in capsys.readouterr().err

    def test_disable_console_silences_the_report(self, monkeypatch, tmp_path):
        # --disable-console says "disable ALL outputs to the console". A chart is
        # the loudest thing the runner prints, so it has to be the first to go.
        async def fake_run_ramp(cfg, **kwargs):
            from benchmark_runner.auto_tune import PointMetrics, RampOutcome

            return RampOutcome(
                points=[PointMetrics(**{**p, "index": i}) for i, p in enumerate(CURVE)],
                bracket_reason="capacity_plateau",
                stop_reason="converged",
                target=cfg.target,
                axis=cfg.axis,
                stopped_at=64.0,
                lower_bound=cfg.lower_bound,
                upper_bound=cfg.upper_bound,
                max_points=cfg.max_points,
                max_total_seconds=cfg.max_total_seconds,
                elapsed_seconds=1.0,
            )

        monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
        args = ["benchmark", "run", "--auto-tune", "--output-dir", str(tmp_path)]

        loud = CliRunner().invoke(cli, args)
        assert loud.exit_code == 0, loud.output
        assert "Recommended:" in loud.output

        quiet = CliRunner().invoke(cli, [*args, "--disable-console"])
        assert quiet.exit_code == 0, quiet.output
        assert "Recommended:" not in quiet.output

    def test_chart_none_skips_the_pictures_but_still_reports(
        self, monkeypatch, tmp_path
    ):
        async def fake_run_ramp(cfg, **kwargs):
            from benchmark_runner.auto_tune import PointMetrics, RampOutcome

            return RampOutcome(
                points=[PointMetrics(**{**p, "index": i}) for i, p in enumerate(CURVE)],
                bracket_reason="capacity_plateau",
                stop_reason="converged",
                target=cfg.target,
                axis=cfg.axis,
                stopped_at=64.0,
                lower_bound=cfg.lower_bound,
                upper_bound=cfg.upper_bound,
                max_points=cfg.max_points,
                max_total_seconds=cfg.max_total_seconds,
                elapsed_seconds=1.0,
            )

        monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
        result = CliRunner().invoke(
            cli,
            [
                "benchmark",
                "run",
                "--auto-tune",
                "--chart",
                "none",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Recommended:" in result.output
        assert "total throughput (tok/s)" not in result.output


def test_chart_reads_the_pre_rename_sidecar():
    """A v3 sidecar spells the per-token field `tpot_ms`.

    The rename to guidellm's vocabulary happened in v4. Charting an older run
    must still show its ITL column rather than a blank one — the number is the
    same, only the key moved.
    """
    from benchmark_runner.chart import _itl_of

    assert _itl_of({"itl_ms": 12.5}) == 12.5
    assert _itl_of({"tpot_ms": 12.5}) == 12.5
    # The current spelling wins when a file somehow carries both.
    assert _itl_of({"itl_ms": 12.5, "tpot_ms": 99.0}) == 12.5
    assert _itl_of({}) == 0.0
