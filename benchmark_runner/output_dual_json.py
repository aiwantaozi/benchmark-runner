"""
Output handler for serializing generative benchmark reports to JSON (both summary and full).

This module implements a dual-output JSON handler that saves both:
1. Summary JSON - Excludes large fields like individual requests and detailed metrics
2. Full JSON - Contains complete benchmark data including all requests

Both files are saved to the same directory with clear naming conventions.

guidellm 0.7.1 note:
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
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import BenchmarkOutputArgs, GenerativeBenchmarksReport

__all__ = [
    "GenerativeBenchmarkerDualJson",
    "DualJsonBenchmarkOutputArgs",
    "AutoMarshalJSONEncoder",
]


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
        if hasattr(o, "__json__") and callable(getattr(o, "__json__")):
            return o.__json__()

        # Handle class/type objects (like request handler classes)
        if isinstance(o, type):
            # Try to find the registered name for this handler class.
            # guidellm 0.7.1: request handlers live in
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
        # JSON-compatible values; guidellm 0.7.1 embeds the BenchmarkScenario
        # (with Path output targets) in report.config.
        full_dict = report.model_dump(mode="json")
        summary_dict = report.model_dump(mode="json", exclude=self.EXCLUDE_FIELDS)
        self._attach_error_samples(summary_dict, full_dict)

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
