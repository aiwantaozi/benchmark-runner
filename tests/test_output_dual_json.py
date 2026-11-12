"""Tests for the dual-JSON output's ITL aggregation.

The samples themselves are produced by ``openai_http_error_detail_backend`` (see
that module's tests); here the concern is what the output layer does with them:
flatten across requests, shape the result like guidellm's own latency metrics,
and keep the raw per-request lists out of the files.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    StatusBreakdown,
)

from benchmark_runner.openai_http_error_detail_backend import ITL_TIMINGS_FIELD
from benchmark_runner.output_dual_json import (
    ITL_METRIC_KEY,
    KEEP_PER_REQUEST_ITL_ENV,
    GenerativeBenchmarkerDualJson,
    keep_per_request_itl,
)


def _stats(gaps: list[float] | None) -> SimpleNamespace:
    """One request's stats, carrying its ITL samples where the backend puts them.

    A real ``RequestInfo`` rather than a stub, because the samples live in
    pydantic's ``extra`` storage and the point is that they are reachable as a
    plain attribute after the model has been through the scheduler.
    """
    info = RequestInfo(request_id="r", status="completed")
    if gaps is not None:
        setattr(info.timings, ITL_TIMINGS_FIELD, gaps)
    return SimpleNamespace(info=info)


def test_itl_filters_each_gap_by_its_token_event_time():
    stat = _stats([100.0, 900.0, 200.0])
    stat.request_start_time = 8.0
    stat.request_end_time = 22.0
    stat.info.timings.inter_token_latency_timestamps = [9.0, 11.0, 21.0]
    benchmark = _report(successful=[stat]).benchmarks[0]
    benchmark.start_time = 10.0
    benchmark.end_time = 20.0
    metric = _output()._itl_distribution(benchmark)
    assert metric.successful.mean == pytest.approx(900.0)
    assert metric.successful.min == pytest.approx(900.0)
    assert metric.successful.max == pytest.approx(900.0)


def _report(successful, incomplete=(), errored=()) -> SimpleNamespace:
    return SimpleNamespace(
        benchmarks=[
            SimpleNamespace(
                requests=SimpleNamespace(
                    successful=list(successful),
                    incomplete=list(incomplete),
                    errored=list(errored),
                )
            )
        ]
    )


def _dicts(with_requests: list[dict] | None = None) -> tuple[dict, dict]:
    """A summary dict (no ``requests``) and a full dict, as ``finalize`` builds them."""
    summary = {"benchmarks": [{"metrics": {"time_to_first_token_ms": {}}}]}
    full = {
        "benchmarks": [
            {
                "metrics": {"time_to_first_token_ms": {}},
                "requests": {"successful": with_requests or []},
            }
        ]
    }
    return summary, full


def _output() -> GenerativeBenchmarkerDualJson:
    return GenerativeBenchmarkerDualJson(output_path=Path("/tmp"))


def test_itl_samples_are_flattened_across_requests():
    """Every gap is its own sample, so a lone stall lands in the tail.

    Two requests, one of which stalled once for 800 ms. Averaged per request
    that stall would come out as ~275 ms and the tail would show nothing; the
    metric exists precisely to keep the 800 ms visible.
    """
    output = _output()
    report = _report([_stats([10.0, 800.0, 10.0]), _stats([10.0, 10.0])])
    summary, full = _dicts()

    output._attach_itl_metrics(report, summary, full)

    metric = summary["benchmarks"][0]["metrics"][ITL_METRIC_KEY]
    assert metric["successful"]["count"] == 5
    assert metric["successful"]["max"] == pytest.approx(800.0)
    assert metric["successful"]["percentiles"]["p99"] == pytest.approx(800.0)
    assert metric["successful"]["mean"] == pytest.approx(168.0)


def test_itl_metric_shape_matches_guidellm_latency_metrics():
    """Readers must reach it by the path they already use for TTFT."""
    output = _output()
    summary, full = _dicts()

    output._attach_itl_metrics(_report([_stats([5.0, 7.0])]), summary, full)

    metric = summary["benchmarks"][0]["metrics"][ITL_METRIC_KEY]
    assert set(metric) >= {"successful", "incomplete", "errored", "total"}
    assert {"mean", "median", "max", "count", "percentiles"} <= set(
        metric["successful"]
    )
    # The existing metrics are left alone.
    assert "time_to_first_token_ms" in summary["benchmarks"][0]["metrics"]


def test_itl_metric_written_to_both_summary_and_full():
    output = _output()
    summary, full = _dicts()

    output._attach_itl_metrics(_report([_stats([5.0, 7.0])]), summary, full)

    for target in (summary, full):
        assert ITL_METRIC_KEY in target["benchmarks"][0]["metrics"]


def test_no_samples_leaves_metrics_untouched():
    """ "Not measured" must not be written as a distribution of zeros.

    A non-streaming run, or one recorded before this metric existed, carries no
    samples. An all-zero distribution there would read as "measured, and every
    gap was 0 ms" — a perfect score for a run that measured nothing.
    """
    output = _output()
    summary, full = _dicts()

    # Requests present, but none carrying samples (attribute absent entirely,
    # and present-but-empty).
    output._attach_itl_metrics(_report([_stats(None), _stats([])]), summary, full)

    assert ITL_METRIC_KEY not in summary["benchmarks"][0]["metrics"]
    assert ITL_METRIC_KEY not in full["benchmarks"][0]["metrics"]


def test_incomplete_and_errored_samples_are_kept_separate():
    """Same status breakdown as every other metric, so the tails don't mix.

    A cancelled request's partial gaps must not inflate the successful
    distribution.
    """
    output = _output()
    report = _report(
        successful=[_stats([10.0, 10.0])],
        incomplete=[_stats([900.0])],
    )
    summary, full = _dicts()

    output._attach_itl_metrics(report, summary, full)

    metric = summary["benchmarks"][0]["metrics"][ITL_METRIC_KEY]
    assert metric["successful"]["max"] == pytest.approx(10.0)
    assert metric["incomplete"]["max"] == pytest.approx(900.0)


def test_raw_per_request_samples_are_stripped_from_full_report():
    """The aggregate is kept; the raw lists are not.

    Under ``--sample-requests 0`` the requests carry no prompt/output text, so
    these lists would become the bulk of the file — ~1.4 MB per stage at 128
    output tokens, and every stage of every run is kept on disk.
    """
    output = _output()
    serialized = [
        {"info": {"timings": {ITL_TIMINGS_FIELD: [10.0, 800.0], "token_iterations": 3}}}
    ]
    summary, full = _dicts(with_requests=serialized)

    output._attach_itl_metrics(_report([_stats([10.0, 800.0])]), summary, full)

    timings = full["benchmarks"][0]["requests"]["successful"][0]["info"]["timings"]
    assert ITL_TIMINGS_FIELD not in timings
    # Only that key goes; the rest of the timings record stays.
    assert timings["token_iterations"] == 3
    # And the aggregate survived the strip.
    assert full["benchmarks"][0]["metrics"][ITL_METRIC_KEY]["successful"][
        "max"
    ] == pytest.approx(800.0)


def _real_request_stats(gaps: list[float]):
    """A genuine ``GenerativeRequestStats``, built the way the accumulator builds it."""
    info = RequestInfo(request_id="r", status="completed")
    timings = info.timings
    timings.request_start = 0.0
    timings.first_token_iteration = 1.0
    timings.last_token_iteration = 2.0
    timings.token_iterations = 3
    timings.request_end = timings.resolve_end = 2.5
    setattr(timings, ITL_TIMINGS_FIELD, gaps)

    response = GenerationResponse(request_id="r", request_args=None)
    return response.compile_stats(
        request=GenerationRequest(request_id="r"),
        info=info,
        prefer_response=False,
    )


def test_strip_targets_the_real_serialized_layout():
    """Pins the ``info.timings`` path the strip walks.

    The other strip test hand-builds that shape, so it would keep passing if
    guidellm moved ``timings`` elsewhere and the strip silently became a no-op
    (leaving the raw lists in every file). This one dumps a real stats object,
    so a layout change fails here instead.
    """
    output = _output()
    stats = _real_request_stats([10.0, 800.0])
    dumped = stats.model_dump(mode="json")
    assert dumped["info"]["timings"][ITL_TIMINGS_FIELD] == [10.0, 800.0]

    summary, full = _dicts(with_requests=[dumped])
    output._attach_itl_metrics(_report([stats]), summary, full)

    assert (
        ITL_TIMINGS_FIELD
        not in full["benchmarks"][0]["requests"]["successful"][0]["info"]["timings"]
    )
    assert summary["benchmarks"][0]["metrics"][ITL_METRIC_KEY]["successful"][
        "max"
    ] == pytest.approx(800.0)


def test_reads_the_real_status_breakdown_layout():
    """Pins the ``requests.successful`` / ``incomplete`` / ``errored`` names.

    A real benchmark's ``requests`` is a ``StatusBreakdown``, walked here by
    attribute. With a stub it would keep passing after an upstream rename while
    quietly producing an empty distribution for every run.
    """
    output = _output()
    breakdown = StatusBreakdown(
        successful=[_real_request_stats([10.0, 800.0])],
        incomplete=[],
        errored=[],
        total=None,
    )
    report = SimpleNamespace(benchmarks=[SimpleNamespace(requests=breakdown)])
    summary, full = _dicts()

    output._attach_itl_metrics(report, summary, full)

    assert summary["benchmarks"][0]["metrics"][ITL_METRIC_KEY]["successful"][
        "max"
    ] == pytest.approx(800.0)


def test_itl_tail_is_invisible_to_guidellms_own_per_token_metric():
    """Why this metric exists, stated as a test.

    Same request, two readings: guidellm's ``inter_token_latency_ms`` (which is
    the industry's TPOT — one value per request) averages the stall away, while
    the per-chunk samples keep it. If these two ever agree, the new metric has
    stopped measuring anything the old one did not.
    """
    output = _output()
    stats = _real_request_stats([10.0, 800.0])
    summary, full = _dicts()

    output._attach_itl_metrics(_report([stats]), summary, full)

    # (last_token - first_token) / (tokens - 1) = 1.0s / 2 = 500ms: the 800ms
    # stall is diluted by the healthy 10ms gap.
    assert stats.inter_token_latency_ms == pytest.approx(500.0)
    # The per-chunk tail keeps it.
    metric = summary["benchmarks"][0]["metrics"][ITL_METRIC_KEY]["successful"]
    assert metric["max"] == pytest.approx(800.0)


def test_raw_samples_are_kept_when_the_env_var_is_set(monkeypatch):
    """The escape hatch for investigating one request's stall pattern.

    An env var rather than a source edit: the runner ships in a container, and
    the moment you want the raw samples is while chasing a stall on a live
    deployment — precisely when rebuilding an image is not an option.
    """
    monkeypatch.setenv(KEEP_PER_REQUEST_ITL_ENV, "1")
    output = _output()
    serialized = [{"info": {"timings": {ITL_TIMINGS_FIELD: [10.0, 800.0]}}}]
    summary, full = _dicts(with_requests=serialized)

    output._attach_itl_metrics(_report([_stats([10.0, 800.0])]), summary, full)

    timings = full["benchmarks"][0]["requests"]["successful"][0]["info"]["timings"]
    assert timings[ITL_TIMINGS_FIELD] == [10.0, 800.0]
    # The aggregate is written either way — the flag only decides whether the
    # raw lists ride along with it.
    assert full["benchmarks"][0]["metrics"][ITL_METRIC_KEY]["successful"][
        "max"
    ] == pytest.approx(800.0)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_truthy_spellings_all_enable_it(monkeypatch, value):
    monkeypatch.setenv(KEEP_PER_REQUEST_ITL_ENV, value)
    assert keep_per_request_itl() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_everything_else_leaves_it_off(monkeypatch, value):
    """Unset and anything unrecognized both mean off — the safe direction here
    is the one that does not silently multiply the report size."""
    monkeypatch.setenv(KEEP_PER_REQUEST_ITL_ENV, value)
    assert keep_per_request_itl() is False


def _consistent_stats(request_id: str, gaps_ms: list[float], ttft_s: float = 0.2):
    """A real `GenerativeRequestStats` whose timings agree with `gaps_ms`.

    The timings are derived FROM the gaps (`last = first + sum(gaps)`,
    `token_iterations = len(gaps) + 1`), which is exactly the relationship the
    backend produces when it records them. That makes guidellm's own
    `inter_token_latency_ms` an independent check on our aggregate.
    """
    info = RequestInfo(request_id=request_id, status="completed")
    timings = info.timings
    timings.request_start = 0.0
    timings.first_token_iteration = ttft_s
    timings.last_token_iteration = ttft_s + sum(gaps_ms) / 1000.0
    timings.token_iterations = len(gaps_ms) + 1
    timings.request_end = timings.resolve_end = timings.last_token_iteration + 0.01
    setattr(timings, ITL_TIMINGS_FIELD, gaps_ms)

    response = GenerationResponse(request_id=request_id, request_args=None)
    return response.compile_stats(
        request=GenerationRequest(request_id=request_id),
        info=info,
        prefer_response=False,
    )


def test_aggregate_mean_matches_guidellms_token_weighted_mean():
    """The pooled mean must equal guidellm's own numbers, token-weighted.

    Pooling every gap and averaging is algebraically the same as taking each
    request's `inter_token_latency_ms` weighted by its gap count — guidellm
    aggregates that way on purpose. Checking it here pins the flattening step:
    a request dropped from the pool, or counted once instead of per gap, breaks
    the identity even though each individual value stays correct.

    Output lengths deliberately differ (2, 5 and 500 gaps) — with equal lengths
    a request-equal-weighted bug would pass unnoticed.
    """
    requests = [
        _consistent_stats("a", [10.0, 10.0]),
        _consistent_stats("b", [100.0] * 5),
        _consistent_stats("c", [10.0] * 499 + [800.0]),
    ]

    distribution = GenerativeBenchmarkerDualJson._itl_distribution(
        _report(requests).benchmarks[0]
    )

    weighted_sum = sum(
        s.inter_token_latency_ms * (s.output_tokens - 1) for s in requests
    )
    total_weight = sum(s.output_tokens - 1 for s in requests)

    assert distribution.successful.mean == pytest.approx(weighted_sum / total_weight)
    # And the pool really is per-gap, not per-request.
    assert distribution.successful.count == total_weight == 2 + 5 + 500


def test_aggregate_keeps_the_stall_that_the_per_request_value_hides():
    """The whole point of the metric, asserted end to end.

    The stalling request's own `inter_token_latency_ms` is ~11.6 ms — it reads
    as healthy. The 800 ms gap has to survive into the pooled distribution.
    """
    stalling = _consistent_stats("c", [10.0] * 499 + [800.0])
    assert stalling.inter_token_latency_ms == pytest.approx(11.58, abs=0.05)

    distribution = GenerativeBenchmarkerDualJson._itl_distribution(
        _report([stalling]).benchmarks[0]
    )

    assert distribution.successful.max == pytest.approx(800.0)


def _windowed_report(successful, start: float, end: float):
    """A benchmark whose measurement window excludes part of its requests."""
    report = _report(successful)
    report.benchmarks[0].start_time = start
    report.benchmarks[0].end_time = end
    return report


def _timed_stats(request_id: str, gaps_ms: list[float], started: float, ended: float):
    """A real stats object placed at a specific point on the run's timeline."""
    stats = _consistent_stats(request_id, gaps_ms)
    stats.info.timings.request_start = started
    stats.info.timings.first_token_iteration = started + 0.01
    stats.info.timings.last_token_iteration = ended - 0.01
    stats.info.timings.request_end = stats.info.timings.resolve_end = ended
    return stats


def test_warmup_requests_are_excluded_from_the_distribution():
    """The population must match every other metric's.

    guidellm compiles its metrics from the measurement window only, so a run
    with `warmup.percent = 0.1` leaves ~10% of the requests out. Those requests
    are still present in `benchmark.requests`, and counting them here produced
    an ITL distribution drawn from a different set of requests than the TTFT
    and TPOT printed beside it.
    """
    warmup = _timed_stats("warmup", [5.0, 5.0], started=0.0, ended=9.0)
    measured = _timed_stats("measured", [50.0, 50.0], started=11.0, ended=20.0)

    distribution = GenerativeBenchmarkerDualJson._itl_distribution(
        _windowed_report([warmup, measured], start=10.0, end=30.0).benchmarks[0]
    )

    # Only the measured request's gaps survive.
    assert distribution.successful.count == 2
    assert distribution.successful.mean == pytest.approx(50.0)


def test_a_warmup_stall_cannot_reach_max():
    """The specific failure the window guards against.

    `max` is what this metric is read for. A stall while the engine is still
    warming up is not a property of the steady state being measured.
    """
    warmup = _timed_stats("warmup", [10.0, 900.0], started=0.0, ended=9.0)
    measured = _timed_stats("measured", [50.0, 50.0], started=11.0, ended=20.0)

    distribution = GenerativeBenchmarkerDualJson._itl_distribution(
        _windowed_report([warmup, measured], start=10.0, end=30.0).benchmarks[0]
    )

    assert distribution.successful.max == pytest.approx(50.0)


def test_a_request_overlapping_the_window_is_kept():
    """Overlap, not containment — same rule guidellm applies.

    A request that starts inside the window but finishes after it still
    counted toward every other metric, so it has to count here too.
    """
    straddling = _timed_stats("straddling", [50.0, 50.0], started=29.0, ended=35.0)

    distribution = GenerativeBenchmarkerDualJson._itl_distribution(
        _windowed_report([straddling], start=10.0, end=30.0).benchmarks[0]
    )

    assert distribution.successful.count == 2


def test_no_window_means_no_filtering():
    """A benchmark without the timing fields must not lose every sample."""
    stats = _timed_stats("x", [50.0, 50.0], started=0.0, ended=9.0)

    distribution = GenerativeBenchmarkerDualJson._itl_distribution(
        _report([stats]).benchmarks[0]
    )

    assert distribution.successful.count == 2
