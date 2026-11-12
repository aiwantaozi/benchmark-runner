"""Additional capacity conclusions must retain measured and fitted semantics."""

import asyncio
from types import SimpleNamespace

import pytest

from benchmark_runner.auto_tune import AutoTuneConfig, _normalize, _passes_slo, run_ramp
from guidellm.schemas.benchmark import GoodputSLO
from tests.test_auto_tune_ramp import make_metrics, make_run_point, piecewise


@pytest.mark.parametrize("confirmations", [0, -1, 1.5, True, "2", None])
def test_invalid_confirmation_counts_fail_at_configuration(confirmations):
    with pytest.raises(
        ValueError, match="plateau_confirmations must be a positive integer"
    ):
        AutoTuneConfig(axis="concurrency", plateau_confirmations=confirmations)


def test_joint_attainment_is_an_additional_slo_requirement():
    cfg = AutoTuneConfig(axis="concurrency", slo_attainment=0.95, slo_p95_ttft_ms=20)
    point = make_metrics(4, 0, 100)
    point.slo_attainment = 0.90
    point.slo_determined_requests = 100
    assert cfg.target == "slo"
    assert not _passes_slo(point, cfg)
    point.slo_attainment = 0.99
    assert _passes_slo(point, cfg)
    point.ttft_p95_ms = 30
    assert not _passes_slo(point, cfg)


def test_disjoint_latency_violations_fail_joint_attainment():
    slo = GoodputSLO(ttft_ms=20, tpot_ms=20)
    requests = [(30 if i < 4 else 10, 30 if i >= 96 else 5) for i in range(100)]
    assert sum(ttft <= 20 for ttft, _ in requests) == 96
    assert sum(tpot <= 20 for _, tpot in requests) == 96
    point = make_metrics(4, 0, 100)
    point.slo_attainment = sum(
        slo.is_conforming(ttft, tpot, None) for ttft, tpot in requests
    ) / len(requests)
    point.slo_determined_requests = len(requests)
    cfg = AutoTuneConfig(
        axis="concurrency",
        slo_p95_ttft_ms=20,
        slo_p95_itl_ms=20,
        slo_attainment=0.95,
    )
    assert point.slo_attainment == 0.92
    assert not _passes_slo(point, cfg)


@pytest.mark.parametrize(
    "attainment,population", [(None, 0), (0.99, 0), (float("nan"), 10)]
)
def test_unknown_attainment_cannot_pass(attainment, population):
    cfg = AutoTuneConfig(axis="concurrency", slo_attainment=0.95)
    point = make_metrics(4, 0, 100)
    point.slo_attainment = attainment
    point.slo_determined_requests = population
    assert not _passes_slo(point, cfg)


def test_normalization_keeps_goodput_and_uncertainty():
    metrics = SimpleNamespace(
        request_totals=SimpleNamespace(total=100, successful=100),
        slo_attainment=0.9,
        slo_determined_requests=100,
        request_goodput=SimpleNamespace(successful=SimpleNamespace(mean=9.0)),
    )
    point = _normalize(SimpleNamespace(metrics=metrics), 4, 0)
    assert point.slo_attainment == 0.9
    assert point.request_goodput == 9.0
    assert point.slo_determined_requests == 100
    assert point.slo_attainment_lower < 0.9 < point.slo_attainment_upper
    assert point.slo_attainment_confidence == 0.95


def test_completed_sample_boundary_cancellations_are_not_server_errors():
    benchmark = SimpleNamespace(
        metrics=SimpleNamespace(
            request_totals=SimpleNamespace(
                total=120, successful=100, errored=0, incomplete=20
            )
        ),
        scheduler_state=SimpleNamespace(
            end_processing_constraints={
                "min_requests": SimpleNamespace(metadata={"processed_exceeded": True})
            }
        ),
    )
    assert _normalize(benchmark, 16, 0).success == 1.0
    benchmark.metrics.request_totals.successful = 95
    benchmark.metrics.request_totals.errored = 5
    assert _normalize(benchmark, 16, 0).success == 0.95
    benchmark.scheduler_state.end_processing_constraints = {}
    assert _normalize(benchmark, 16, 0).success == pytest.approx(95 / 120)


def test_one_local_plateau_does_not_hide_later_growth():
    curve = piecewise({4: 40, 8: 80, 16: 82, 32: 200, 64: 350, 128: 360, 256: 361})
    cfg = AutoTuneConfig(
        axis="concurrency",
        lower_bound=4,
        upper_bound=256,
        max_points=24,
        probe_saturation=False,
    )
    result = asyncio.run(run_ramp(cfg, {}, "unused", run_point=make_run_point(curve)))
    assert max(p.total_tps for p in result.points) >= 350
    assert all(4 <= p.knob <= 256 for p in result.points)


@pytest.mark.parametrize("confirmations", [1, 2])
def test_confirmation_policy_is_recorded_in_the_sidecar(confirmations):
    cfg = AutoTuneConfig(
        axis="concurrency",
        max_points=1,
        probe_saturation=False,
        plateau_confirmations=confirmations,
    )
    result = asyncio.run(
        run_ramp(cfg, {}, "unused", run_point=make_run_point(lambda k: 100))
    )
    assert result.to_dict()["plateau_confirmations"] == confirmations


@pytest.mark.parametrize("max_points,upper", [(1, 256), (3, 256), (18, 16), (18, 256)])
def test_default_confirmation_respects_budgets_and_range(max_points, upper):
    cfg = AutoTuneConfig(
        axis="concurrency",
        lower_bound=4,
        upper_bound=upper,
        max_points=max_points,
        probe_saturation=False,
    )
    curve = piecewise({4: 40, 8: 80, 16: 82, 32: 200, 64: 350, 128: 360, 256: 361})
    result = asyncio.run(run_ramp(cfg, {}, "unused", run_point=make_run_point(curve)))
    assert len(result.points) <= max_points
    assert all(4 <= p.knob <= upper for p in result.points)
    if max_points <= 3:
        assert result.stop_reason == "budget_points"
    elif upper == 16:
        assert result.stop_reason == "upper_bound"


def test_knee_and_peak_are_distinct_conclusions():
    from benchmark_runner.curve_analysis import analyze_curve

    xs = [4, 8, 16, 32, 64, 128, 256]
    ys = [40, 80, 160, 320, 350, 365, 366]
    points = [make_metrics(x, i, y) for i, (x, y) in enumerate(zip(xs, ys))]
    analysis = analyze_curve(points, axis="concurrency", slo_thresholds={})
    assert analysis["throughput_peak"]["knob"] == 256
    assert analysis["throughput_peak"]["metric"] == "total_tokens_per_second"
    assert analysis["throughput_knee"]["metric"] == "output_tokens_per_second"
    assert analysis["throughput_knee"]["knee"] == pytest.approx(33.576, abs=0.01)
    assert analysis["throughput_knee"]["saturation_concurrency"] == 64
    assert analysis["slo_capacity"] is None


@pytest.mark.parametrize(
    "axis,n,status",
    [
        ("rate", 7, "unavailable"),
        ("concurrency", 4, "no_knee"),
        ("concurrency", 7, "no_knee"),
    ],
)
def test_knee_does_not_invent_an_answer(axis, n, status):
    from benchmark_runner.curve_analysis import analyze_curve

    points = [make_metrics(2**i, i, 100 * 2**i) for i in range(n)]
    result = analyze_curve(points, axis=axis, slo_thresholds={})
    assert result["throughput_knee"]["status"] == status


def test_reported_slo_capacity_is_a_measured_pass():
    from benchmark_runner.curve_analysis import analyze_curve

    points = [
        make_metrics(x, i, 100, ttft=x * 10) for i, x in enumerate([1, 2, 4, 8, 16])
    ]
    result = analyze_curve(
        points, axis="concurrency", slo_thresholds={"slo_avg_ttft_ms": 50}
    )
    assert result["slo_capacity"]["knob"] == 4
    assert result["slo_capacity"]["measured"] is True


def test_intervals_crossing_the_target_are_reported_as_unresolved():
    from benchmark_runner.curve_analysis import analyze_curve

    point = make_metrics(4, 0, 100)
    point.slo_attainment = 0.96
    point.slo_determined_requests = 100
    point.slo_attainment_lower = 0.90
    point.slo_attainment_upper = 0.99
    result = analyze_curve(
        [point], axis="concurrency", slo_thresholds={"slo_attainment": 0.95}
    )
    assert result["slo_capacity"]["knob"] == 4
    assert result["unresolved_slo_knobs"] == [4]
