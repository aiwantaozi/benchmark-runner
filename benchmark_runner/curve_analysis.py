"""Measured capacity and fitted throughput-knee conclusions.

The runner's search objective is total tokens/s. GuideLLM's knee estimator uses
output tokens/s and only applies to concurrency curves; keep both names and
quantities explicit instead of turning a fitted point into a recommendation.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from guidellm.benchmark.analysis import find_throughput_knee
from guidellm.utils.statistics import wilson_interval

from benchmark_runner.constants import SUCCESS_FLOOR


def passes_slo(point: Any, thresholds: dict[str, float]) -> bool:
    """Require success, each aggregate objective and optional joint attainment."""
    if not math.isfinite(point.success) or point.success < SUCCESS_FLOOR:
        return False
    for name, threshold in thresholds.items():
        if name == "slo_attainment":
            value = point.slo_attainment
            if (
                value is None
                or not math.isfinite(value)
                or not 0 <= value <= 1
                or point.slo_determined_requests <= 0
                or value < threshold
            ):
                return False
            continue
        aggregation, metric = name.removeprefix("slo_").split("_", 1)
        field = (
            metric
            if aggregation == "avg"
            else metric.replace("_ms", f"_{aggregation}_ms")
        )
        value = getattr(point, field)
        if not math.isfinite(value) or value <= 0 or value > threshold:
            return False
    return True


def goodput_fields(benchmark: Any) -> dict[str, Any]:
    """Keep missing goodput unknown and report uncertainty over determined requests."""
    metrics = benchmark.metrics
    attainment = getattr(metrics, "slo_attainment", None)
    population = getattr(metrics, "slo_determined_requests", 0) or 0
    distribution = getattr(metrics, "request_goodput", None)
    confidence = getattr(getattr(benchmark, "config", None), "confidence", 0.95)
    lower = upper = None
    if (
        attainment is not None
        and math.isfinite(attainment)
        and 0 <= attainment <= 1
        and population > 0
        and confidence is not None
    ):
        lower, upper = wilson_interval(
            successes=round(attainment * population),
            trials=population,
            confidence=confidence,
        )
    return {
        "slo_attainment": attainment,
        "slo_determined_requests": population,
        "request_goodput": (
            distribution.successful.mean if distribution is not None else None
        ),
        "slo_attainment_lower": lower,
        "slo_attainment_upper": upper,
        "slo_attainment_confidence": confidence if lower is not None else None,
    }


def analyze_curve(
    points: Sequence[Any],
    *,
    axis: str,
    slo_thresholds: dict[str, float],
) -> dict[str, Any]:
    """Separate observed total throughput, output-throughput knee and measured SLO load."""
    valid = [
        p
        for p in points
        if p.success >= SUCCESS_FLOOR and math.isfinite(p.total_tps) and p.total_tps > 0
    ]
    peak = max(valid, key=lambda p: (p.total_tps, -p.knob, -p.index), default=None)
    capacity = max(
        (p for p in valid if slo_thresholds and passes_slo(p, slo_thresholds)),
        key=lambda p: p.knob,
        default=None,
    )
    if axis == "concurrency":
        knee_points = [
            p for p in valid if math.isfinite(p.output_tps) and p.output_tps >= 0
        ]
        knee = find_throughput_knee(
            [p.knob for p in knee_points], [p.output_tps for p in knee_points]
        ).model_dump(mode="json")
    else:
        knee = {
            "status": "unavailable",
            "reason": "Knee fitting requires a concurrency axis",
        }
    knee["metric"] = "output_tokens_per_second"
    knee["measured"] = False
    target = slo_thresholds.get("slo_attainment")
    unresolved = [
        p.knob
        for p in points
        if target is not None
        and p.slo_attainment_lower is not None
        and p.slo_attainment_upper is not None
        and p.slo_attainment_lower < target <= p.slo_attainment_upper
    ]
    return {
        "unresolved_slo_knobs": unresolved,
        "throughput_peak": (
            {
                "knob": peak.knob,
                "value": peak.total_tps,
                "metric": "total_tokens_per_second",
                "measured": True,
            }
            if peak is not None
            else None
        ),
        "throughput_knee": knee,
        "slo_capacity": (
            {"knob": capacity.knob, "measured": True, "status": "highest_measured_pass"}
            if capacity is not None
            else None
        ),
    }
