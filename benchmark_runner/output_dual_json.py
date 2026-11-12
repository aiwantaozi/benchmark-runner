"""
Output handler for serializing generative benchmark reports to JSON (both summary and full).

This module implements a dual-output JSON handler that saves both:
1. Summary JSON - Excludes large fields like individual requests and detailed metrics
2. Full JSON - Contains complete benchmark data including all requests

Both files are saved to the same directory with clear naming conventions.

This module also aggregates the per-chunk ITL samples that
``openai_http_error_detail_backend`` records, into a distribution shaped exactly
like guidellm's own latency metrics -- see ``ITL_METRIC_KEY`` below.

guidellm 0.7.x note:
- Output formatters are resolved by ``GenerativeBenchmarkerOutput.resolve(args)``
  which looks up the implementation by ``args.kind`` and calls its ``from_args``
  factory. So a custom output needs BOTH a ``BenchmarkOutputArgs`` subclass (the
  spec, registered by kind) AND the ``GenerativeBenchmarkerOutput`` implementation
  (registered by the same kind) implementing ``from_args``.
- Response handlers still live in ``guidellm.backends.openai.request_handlers``
  with ``OpenAIRequestHandlerFactory`` (used by the JSON encoder fallback to map a
  handler class back to its registered name).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import GenerativeBenchmarksReport
from guidellm.schemas.benchmark import BenchmarkOutputArgs
from guidellm.schemas import StatusDistributionSummary

from benchmark_runner.openai_http_error_detail_backend import (
    ITL_TIMINGS_FIELD,
    ITL_TIMESTAMPS_FIELD,
)

__all__ = [
    "GenerativeBenchmarkerDualJson",
    "DualJsonBenchmarkOutputArgs",
    "AutoMarshalJSONEncoder",
    "ITL_METRIC_KEY",
    "KEEP_PER_REQUEST_ITL_ENV",
    "keep_per_request_itl",
]

# Where the aggregated ITL distribution lands, inside each benchmark's
# ``metrics`` object.
#
# Deliberately NOT named ``inter_token_latency_ms``: guidellm already owns that
# name for a per-REQUEST average -- ``(last_token - first_token) / (n - 1)``,
# which is the industry's TPOT, not an inter-token latency. Reusing the name
# would silently redefine an existing field. ``_per_chunk`` says what the samples
# actually are: one per gap between consecutive streamed outputs, which is
# vLLM's ITL.
ITL_METRIC_KEY = "inter_token_latency_per_chunk_ms"

KEEP_PER_REQUEST_ITL_ENV = "BENCHMARK_RUNNER_KEEP_PER_REQUEST_ITL"
"""Set to 1/true/yes/on to keep each request's raw ITL sample list in the full report."""


def keep_per_request_itl() -> bool:
    """Whether each request keeps its raw sample list in the full report.

    Off by default: the aggregated distribution answers the question, while the
    raw lists are the dominant term in the file size once requests carry no
    prompt/output text (the case under ``--sample-requests 0``). At 128 output
    tokens that is ~1.4 MB per stage, at 1024 tokens ~11 MB, multiplied by every
    stage of every run kept on disk.

    Turned on by env var rather than a source edit, because the runner ships in
    a container: the one time you want the raw samples is while investigating a
    stall on a live deployment, which is exactly when rebuilding an image to
    flip a constant is not an option. Read per call so a test can set it without
    reimporting the module.
    """
    return os.environ.get(KEEP_PER_REQUEST_ITL_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _within_window(stat: Any, start: float | None, end: float | None) -> bool:
    """Whether a request counts toward the measurement window.

    Mirrors guidellm's ``RequestsAccumulator.get_within_range``, which is what
    every other metric is compiled from. Overlap, not containment: a request
    that started inside the window still counts when it finishes after it.

    Left unfiltered when the window is unavailable — dropping every sample
    would be a worse failure than including a few extra.
    """
    if start is None or end is None:
        return True

    stat_end = getattr(stat, "request_end_time", None)
    if stat_end is None or stat_end < start:
        return False

    stat_start = getattr(stat, "request_start_time", None)
    if stat_start is not None:
        return stat_start <= end
    return stat_end <= end


class AutoMarshalJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder with auto-marshal support (similar to Golang's MarshalJSON).

    This encoder automatically checks if objects have __class_json__() or __json__()
    methods and calls them for serialization, providing a Golang-like interface for
    custom JSON marshaling in Python.
    """

    def default(self, o):
        """
        Override default serialization for non-serializable objects.

        Args:
            o: Object to serialize.

        Returns:
            Serializable representation of the object.
        """
        # Check if the object has a __class_json__ method (for class objects)
        if isinstance(o, type) and hasattr(o, "__class_json__"):
            return o.__class_json__()

        # Check if the object has a __json__ method (for instances)
        if hasattr(o, "__json__") and callable(o.__json__):
            return o.__json__()

        # Handle class/type objects (like request handler classes)
        if isinstance(o, type):
            # Try to find the registered name for this handler class.
            # guidellm 0.7.x: request handlers live in
            # guidellm.backends.openai.request_handlers with OpenAIRequestHandlerFactory.
            from guidellm.backends.openai.request_handlers import (
                OpenAIRequestHandlerFactory,
            )

            registry = OpenAIRequestHandlerFactory.registry or {}
            class_to_name = {v: k for k, v in registry.items()}

            handler_name = class_to_name.get(o)
            if handler_name:
                return handler_name
            else:
                # Fallback: use the full class name
                return f"{o.__module__}.{o.__name__}"

        # Let the base class handle other types or raise TypeError
        return super().default(o)


@BenchmarkOutputArgs.register("dual_json")
class DualJsonBenchmarkOutputArgs(BenchmarkOutputArgs):
    """Spec model for the dual-JSON output (summary + full)."""

    kind: Literal["dual_json"] = Field(
        default="dual_json",
        description="The kind of output.",
        examples=["dual_json"],
    )
    path: Path = Field(
        default=Path("./benchmarks.json"),
        description=(
            "Directory or summary file path. The full report is written alongside "
            "with a '.full' suffix inserted before the extension."
        ),
        examples=["./benchmarks.json"],
    )
    error_limit: int | None = Field(
        default=20,
        description="Maximum number of errored requests to include in the summary.",
    )
    incomplete_limit: int | None = Field(
        default=20,
        description="Maximum number of incomplete requests to include in the summary.",
    )


@GenerativeBenchmarkerOutput.register("dual_json")
class GenerativeBenchmarkerDualJson(GenerativeBenchmarkerOutput):
    """
    Output handler for serializing benchmark reports to both summary and full JSON files.

    This class saves two JSON files:
    1. Summary JSON - Excludes large fields (requests, detailed metrics) for quick overview
    2. Full JSON - Contains complete benchmark data including all requests and metrics

    If a directory is provided, default filenames are used. If a file path is provided,
    the summary uses that path and the full version adds a suffix.

    Example:
        # Using directory
        output = GenerativeBenchmarkerDualJson(output_path="/path/to/dir")
        # Creates: /path/to/dir/benchmarks.json (summary)
        #          /path/to/dir/benchmarks.full.json (full)

        # Using file path
        output = GenerativeBenchmarkerDualJson(output_path="/path/to/results.json")
        # Creates: /path/to/results.json (summary)
        #          /path/to/results.full.json (full)
    """

    DEFAULT_FILE: ClassVar[str] = "benchmarks.json"
    EXCLUDE_FIELDS: ClassVar[dict[str, dict[str, Any]]] = {
        "benchmarks": {
            "__all__": {
                "requests": ...,
                "metrics": {"audio", "image", "video"},
            }
        }
    }

    output_path: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Directory or file path for saving the serialized report.",
    )
    error_limit: int | None = Field(
        default=20,
        description="Maximum number of errored requests to include.",
    )
    incomplete_limit: int | None = Field(
        default=20,
        description="Maximum number of incomplete requests to include.",
    )

    @classmethod
    def from_args(cls, args: BenchmarkOutputArgs) -> GenerativeBenchmarkerDualJson:
        """
        Create a dual-JSON output formatter from output arguments.

        :param args: Output configuration with path/limits and kind ``dual_json``
        :return: Configured dual-JSON output formatter
        """
        if not isinstance(args, DualJsonBenchmarkOutputArgs):
            raise ValueError(f"Invalid args type: {type(args)}.")

        output_path = args.path
        # A ".dual_json" suffix (from a bare id like "123.dual_json") is normalized
        # back to ".json" so the summary file uses the standard extension.
        if output_path.suffix.lower() == ".dual_json":
            output_path = output_path.with_suffix(".json")

        return cls(
            output_path=output_path,
            error_limit=args.error_limit,
            incomplete_limit=args.incomplete_limit,
        )

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Serialize and save the benchmark report to both summary and full JSON files.

        Args:
            report: The generative benchmarks report to serialize.
        Returns:
            Path to the saved summary report file.
        """
        # Determine output paths
        summary_path = self.output_path
        if summary_path.is_dir():
            summary_path = summary_path / self.DEFAULT_FILE

        # Create full path by inserting ".full" before the extension
        full_path = (
            summary_path.parent / f"{summary_path.stem}.full{summary_path.suffix}"
        )

        # Ensure parent directory exists
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data. Use mode="json" so pydantic serializes non-JSON-native
        # leaves (e.g. Path in the embedded scenario config, enums, datetimes) to
        # JSON-compatible values; guidellm 0.7.x embeds the BenchmarkScenario
        # (with Path output targets) in report.config.
        full_dict = report.model_dump(mode="json")
        summary_dict = report.model_dump(mode="json", exclude=self.EXCLUDE_FIELDS)
        self._attach_error_samples(summary_dict, full_dict)
        # Read off the report OBJECT, not the dicts: the summary excludes
        # ``requests`` entirely, so the samples are only reachable here.
        self._attach_itl_metrics(report, summary_dict, full_dict)

        # Use custom encoder to handle request handler classes
        encoder_cls = AutoMarshalJSONEncoder

        # Save summary JSON
        summary_str = json.dumps(summary_dict, indent=4, cls=encoder_cls)
        with summary_path.open("w", encoding="utf-8") as file:
            file.write(summary_str)

        # Save full JSON
        full_str = json.dumps(full_dict, indent=4, cls=encoder_cls)
        with full_path.open("w", encoding="utf-8") as file:
            file.write(full_str)

        return summary_path

    def _attach_itl_metrics(
        self,
        report: GenerativeBenchmarksReport,
        summary_dict: dict[str, Any],
        full_dict: dict[str, Any],
    ) -> None:
        """Aggregate each benchmark's per-chunk ITL samples into its metrics.

        The result is a ``StatusDistributionSummary``, the same type guidellm
        uses for TTFT and the rest, so every reader downstream (the report UI's
        ``metrics.<field>.successful.percentiles.p99`` lookups, gpustack's
        collection pass) reaches it by the path it already uses -- no special
        casing for this one metric.

        Benchmarks whose requests carry no samples are left untouched rather
        than given an all-zero distribution: "not measured" (an older runner, a
        non-streaming run) must not read as "measured, and the gaps were 0 ms".
        """
        for index, benchmark in enumerate(report.benchmarks or []):
            distribution = self._itl_distribution(benchmark)
            if distribution is None:
                continue

            dumped = distribution.model_dump(mode="json")
            for target in (summary_dict, full_dict):
                benchmarks = target.get("benchmarks") or []
                if index >= len(benchmarks):
                    continue
                metrics = benchmarks[index].get("metrics")
                if isinstance(metrics, dict):
                    metrics[ITL_METRIC_KEY] = dumped

            if not keep_per_request_itl():
                self._strip_per_request_itl(full_dict, index)

    @staticmethod
    def _itl_distribution(benchmark: Any) -> StatusDistributionSummary | None:
        """This benchmark's ITL samples, flattened across requests by status.

        Flattened, not averaged per request: the whole point of the metric is
        that every gap is its own sample, so one 800 ms stall inside an
        otherwise healthy request survives into the tail instead of being
        divided away by that request's other gaps.

        Restricted to the measurement window, because the population here has
        to match every other metric's. guidellm compiles those from
        ``get_within_range(measure_start, measure_end)``, while
        ``benchmark.requests`` spans the WHOLE run — warmup and cooldown
        included. Taking it as-is silently mixed warmup traffic into this one
        distribution: measured on a real 9-stage run with
        ``warmup.percent = 0.1``, ~10% extra samples per stage, which pulled
        the mean 0.6–1.6% below guidellm's own per-token figure (warmup
        requests are FASTER — the system is not yet saturated) and left a
        warmup stall free to land in ``max``, the one reading this metric
        exists to provide.
        """
        requests = getattr(benchmark, "requests", None)
        if requests is None:
            return None

        # These two ARE the measurement window: `GenerativeBenchmark.start_time`
        # is defined as `scheduler_metrics.measure_start_time`, and `end_time`
        # as `measure_end_time`.
        start = getattr(benchmark, "start_time", None)
        end = getattr(benchmark, "end_time", None)

        def samples(stats: Any) -> list[float]:
            flattened: list[float] = []
            for stat in stats or []:
                if not _within_window(stat, start, end):
                    continue
                timings = getattr(stat, "info", None)
                timings = getattr(timings, "timings", None)
                gaps = getattr(timings, ITL_TIMINGS_FIELD, None)
                if gaps:
                    timestamps = getattr(timings, ITL_TIMESTAMPS_FIELD, None)
                    if timestamps is not None and len(timestamps) == len(gaps):
                        flattened.extend(
                            float(gap)
                            for gap, timestamp in zip(gaps, timestamps)
                            if (start is None or timestamp >= start)
                            and (end is None or timestamp <= end)
                        )
                    else:
                        # Older reports contain gaps without event timestamps.
                        flattened.extend(float(gap) for gap in gaps)
            return flattened

        successful = samples(getattr(requests, "successful", None))
        incomplete = samples(getattr(requests, "incomplete", None))
        errored = samples(getattr(requests, "errored", None))

        if not successful and not incomplete and not errored:
            return None

        return StatusDistributionSummary.from_values(
            successful=successful,
            incomplete=incomplete,
            errored=errored,
        )

    @staticmethod
    def _strip_per_request_itl(full_dict: dict[str, Any], index: int) -> None:
        """Drop the raw sample lists from one benchmark's serialized requests."""
        benchmarks = full_dict.get("benchmarks") or []
        if index >= len(benchmarks):
            return
        requests = benchmarks[index].get("requests")
        if not isinstance(requests, dict):
            return

        for status_requests in requests.values():
            if not isinstance(status_requests, list):
                continue
            for request in status_requests:
                timings = (request or {}).get("info", {}).get("timings")
                if isinstance(timings, dict):
                    timings.pop(ITL_TIMINGS_FIELD, None)
                    timings.pop(ITL_TIMESTAMPS_FIELD, None)

    def _attach_error_samples(
        self, summary_dict: dict[str, Any], full_dict: dict[str, Any]
    ) -> None:
        summary_benchmarks = summary_dict.get("benchmarks") or []
        full_benchmarks = full_dict.get("benchmarks") or []

        for idx, benchmark in enumerate(summary_benchmarks):
            full_benchmark = full_benchmarks[idx] if idx < len(full_benchmarks) else {}
            requests = (full_benchmark or {}).get("requests") or {}

            errored = self._limit_items(requests.get("errored") or [], self.error_limit)
            incomplete = self._limit_items(
                requests.get("incomplete") or [], self.incomplete_limit
            )

            if errored or incomplete:
                benchmark["requests_truncated"] = {}
                if errored:
                    benchmark["requests_truncated"]["errored"] = errored
                if incomplete:
                    benchmark["requests_truncated"]["incomplete"] = incomplete

    @staticmethod
    def _limit_items(items: list[Any], limit: int | None) -> list[Any]:
        if limit is None:
            return list(items)
        return list(items)[: max(limit, 0)]
