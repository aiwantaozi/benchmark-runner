"""
Build a guidellm 0.7.1 ``BenchmarkScenario`` from benchmark-runner's flat CLI
options.

guidellm 0.7.1 replaced the flat ``BenchmarkGenerativeTextArgs`` with a nested
``BenchmarkScenario`` whose ``spec`` is a ``BenchmarkArgs`` dict. This module maps
the benchmark-runner CLI options (which mirror the old flat args, and are also
reused by the stages / auto-tune loops) onto that ``spec`` and calls
``BenchmarkScenario.create(spec=..., scenario=...)`` exactly like guidellm's own
``cli/run.py`` template.

Reference: ``git show v0.7.1:src/guidellm/cli/run.py`` and
``src/guidellm/benchmark/schemas/entrypoints.py`` (BenchmarkArgs field set).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guidellm.benchmark import BenchmarkScenario

from benchmark_runner.sharegpt_adapter import prepare_datasets

__all__ = ["build_scenario_args"]

# Output kinds we know how to emit. "dual_json" is our custom summary+full writer.
_OUTPUT_ALIASES = {"json", "yaml", "csv", "html", "dual_json"}
_OUTPUT_DEFAULT_FILENAME = {
    "json": "benchmarks.json",
    "yaml": "benchmarks.yaml",
    "csv": "benchmarks.csv",
    "html": "benchmarks.html",
    "dual_json": "benchmarks.json",
}
# File-source extensions -> guidellm FileDataArgs kinds.
_FILE_EXT_KIND = {
    ".json": "json_file",
    ".jsonl": "json_file",
    ".csv": "csv_file",
    ".txt": "text_file",
    ".parquet": "parquet_file",
    ".arrow": "arrow_file",
}


def _coerce_scalar(value: str) -> Any:
    """Coerce a ``key=value`` string value into int/float/bool/str."""
    text = value.strip()
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text


def _looks_like_path(source: str) -> bool:
    """Heuristic: does the string look like a filesystem path / HF id, not kv?"""
    if "/" in source or "\\" in source:
        return True
    suffix = Path(source).suffix.lower()
    return suffix in _FILE_EXT_KIND


def _parse_data_source(raw: Any) -> dict[str, Any] | str:
    """Parse a single --data value into a DataArgs dict or a raw source string.

    - JSON object -> dict (used as-is).
    - ``key=value,key2=value2`` -> dict (synthetic-style config).
    - Anything else (path / HF id) -> returned as a string.
    """
    if isinstance(raw, dict):
        return raw

    text = str(raw).strip()

    # JSON object form.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (ValueError, TypeError):
        pass

    # key=value form (synthetic config), but not a path.
    if "=" in text and not _looks_like_path(text):
        parsed: dict[str, Any] = {}
        for pair in text.split(","):
            if "=" not in pair:
                continue
            key, val = pair.split("=", 1)
            parsed[key.strip()] = _coerce_scalar(val)
        if parsed:
            return parsed

    return text


def _wrap_file_source(source: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Wrap a file/HF source string into a DataArgs dict."""
    suffix = Path(source).suffix.lower()
    kind = _FILE_EXT_KIND.get(suffix)
    if kind is not None:
        data_arg: dict[str, Any] = {"kind": kind, "path": source}
    else:
        # Local dataset directory / HuggingFace hub id.
        data_arg = {"kind": "huggingface", "source": source}
    if extra:
        data_arg.update(extra)
    if kind is not None:
        # guidellm's file deserializers call ``load_dataset(<fmt>, data_files=path)``
        # without a split, which returns a ``DatasetDict{"train"}``. The column
        # mapper then does ``dataset.info`` and crashes ('DatasetDict' object has
        # no attribute 'info'). Pin ``split="train"`` so a single file resolves to
        # a flat ``Dataset``. (User-provided load_kwargs.split still wins.)
        load_kwargs = dict(data_arg.get("load_kwargs") or {})
        load_kwargs.setdefault("split", "train")
        data_arg["load_kwargs"] = load_kwargs
    return data_arg


def _build_data(
    values: list[Any],
    data_args_list: list[Any],
    tokenizer_model: str | None,
    max_items: int | None,
) -> list[dict[str, Any]]:
    """Build ``spec.data`` (a list of DataArgs dicts) from --data / --data-args."""
    extra: dict[str, Any] = {}
    for entry in data_args_list:
        if isinstance(entry, dict):
            extra.update(entry)

    result: list[dict[str, Any]] = []
    for raw in values:
        parsed = _parse_data_source(raw)
        if isinstance(parsed, dict):
            data_arg = dict(parsed)
            data_arg.setdefault("kind", "synthetic_text")
            # --data-args provide shared extras; explicit per-source keys win.
            merged = {**extra, **data_arg}
            result.append(merged)
        else:
            # File / HF source: run through the ShareGPT adapter (converts a
            # ShareGPT json/jsonl to a guidellm jsonl) then wrap.
            prepared = prepare_datasets(
                [parsed], tokenizer=tokenizer_model or "", max_items=max_items
            )
            for src in prepared:
                result.append(_wrap_file_source(src, extra))
    return result


def _flatten_rate(rate: Any) -> list[float]:
    """Flatten --rate (tuple of parsed lists, or a plain list) into a flat list."""
    if not rate:
        return []
    flat: list[float] = []
    for item in rate:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _output_arg(kind: str, path: Path) -> dict[str, Any]:
    """Build a single BenchmarkOutputArgs dict for the given kind/path."""
    return {"kind": kind, "path": str(path)}


def _build_outputs(outputs: list[Any], output_dir: Any) -> list[dict[str, Any]]:
    """Build ``spec.outputs`` from --outputs filenames/aliases and --output-dir.

    Extension/alias -> output kind:
      - alias "json"/"csv"/"html"/"yaml"/"dual_json" -> default filename
      - "<name>.<ext>" -> kind from ext (unknown ext defaults to dual_json)
    A ``.dual_json`` summary path is normalized to ``.json`` (the full report is
    written alongside with a ``.full`` infix by the dual_json writer).
    """
    base = Path(output_dir) if output_dir else None
    items = [str(o) for o in outputs] if outputs else ["benchmarks.dual_json"]

    result: list[dict[str, Any]] = []
    for name in items:
        if name in _OUTPUT_ALIASES:
            kind = name
            filename = _OUTPUT_DEFAULT_FILENAME[name]
        else:
            ext = name.rpartition(".")[2].lower() if "." in name else ""
            kind = ext if ext in _OUTPUT_ALIASES else "dual_json"
            filename = name

        path = (base / filename) if base else Path(filename)
        if kind == "dual_json" and path.suffix.lower() == ".dual_json":
            path = path.with_suffix(".json")
        result.append(_output_arg(kind, path))
    return result


def build_scenario_args(kwargs: dict[str, Any]) -> BenchmarkScenario:  # noqa: C901
    """Build a ``BenchmarkScenario`` from benchmark-runner's flat CLI kwargs.

    Only keys that are present are mapped so that any values coming from a
    ``--scenario`` file survive unless explicitly overridden.
    """
    kwargs = dict(kwargs)
    scenario = kwargs.pop("scenario", None)

    spec: dict[str, Any] = {}

    # ── Backend ────────────────────────────────────────────────────────────
    backend_kind = kwargs.pop("backend", None) or "openai_http"
    backend_kwargs = dict(kwargs.pop("backend_kwargs", None) or {})
    backend: dict[str, Any] = {"kind": backend_kind}
    backend.update(backend_kwargs)  # target, model, request_handlers, timeout, ...
    spec["backend"] = backend

    # ── Profile (rate/streams meaning depends on kind) ──────────────────────
    profile_kind = kwargs.pop("profile", None) or "sweep"
    rate = _flatten_rate(kwargs.pop("rate", None))
    profile: dict[str, Any] = {"kind": profile_kind}
    if profile_kind == "concurrent":
        if rate:
            profile["streams"] = [int(r) for r in rate]
    elif profile_kind in ("async", "constant", "poisson"):
        if rate:
            profile["rate"] = [float(r) for r in rate]
    elif profile_kind == "sweep":
        if rate:
            profile["sweep_size"] = int(rate[0])
    # synchronous / throughput take no rate

    warmup = kwargs.pop("warmup", None)
    if warmup is not None:
        profile["warmup"] = warmup
    cooldown = kwargs.pop("cooldown", None)
    if cooldown is not None:
        profile["cooldown"] = cooldown
    rampup = kwargs.pop("rampup", None)
    if rampup is not None:
        profile["rampup_duration"] = float(rampup)
    spec["profile"] = profile

    # ── Tokenizer / processor ───────────────────────────────────────────────
    processor = kwargs.pop("processor", None)
    kwargs.pop("processor_args", None)  # no dedicated spec field in 0.7.1
    tokenizer: dict[str, Any] = {"kind": "huggingface_auto"}
    if processor:
        tokenizer["model"] = processor
    spec["tokenizer"] = tokenizer

    # ── Seed (drives synthetic data + profile RNG; distinct per auto-tune pt) ─
    seed = kwargs.pop("random_seed", None)
    if seed is not None:
        spec["seed"] = {"kind": "static", "value": int(seed)}

    # ── Data ──────────────────────────────────────────────────────────────
    data_values = list(kwargs.pop("data", None) or ())
    data_args_list = list(kwargs.pop("data_args", None) or ())
    data_samples = kwargs.pop("data_samples", None)
    # For the ShareGPT adapter's conversion cap, prefer data_samples then
    # max_requests (still present in kwargs at this point).
    if isinstance(data_samples, int) and data_samples > 0:
        max_items: int | None = data_samples
    else:
        mr = kwargs.get("max_requests")
        max_items = int(mr) if isinstance(mr, int) and mr > 0 else None
    data = _build_data(data_values, data_args_list, tokenizer.get("model"), max_items)
    if data:
        spec["data"] = data

    # ── Data loader (sample cap) ────────────────────────────────────────────
    if isinstance(data_samples, int) and data_samples > 0:
        spec["data_loader"] = {"kind": "pytorch", "samples": data_samples}
    # These 0.6.0 knobs have no direct 0.7.1 spec field; accept + ignore.
    kwargs.pop("data_num_workers", None)
    kwargs.pop("dataloader_kwargs", None)
    kwargs.pop("data_sampler", None)

    # ── Data column mapper ──────────────────────────────────────────────────
    dcm = kwargs.pop("data_column_mapper", None)
    if isinstance(dcm, dict) and dcm:
        spec["data_column_mapper"] = {"kind": "generative_column_mapper", **dcm}

    # ── Metrics (output request sampling) ───────────────────────────────────
    sample_requests = kwargs.pop("sample_requests", None)
    if sample_requests is not None:
        spec["metrics"] = {"kind": "generative", "sample_size": int(sample_requests)}

    # ── Constraints ─────────────────────────────────────────────────────────
    constraints: list[dict[str, Any]] = []
    max_requests = kwargs.pop("max_requests", None)
    if max_requests is not None:
        constraints.append({"kind": "max_requests", "count": int(max_requests)})
    max_seconds = kwargs.pop("max_seconds", None)
    if max_seconds is not None:
        constraints.append({"kind": "max_duration", "seconds": float(max_seconds)})
    max_errors = kwargs.pop("max_errors", None)
    if max_errors is not None:
        constraints.append({"kind": "max_errors", "count": int(max_errors)})
    max_error_rate = kwargs.pop("max_error_rate", None)
    if max_error_rate is not None:
        constraints.append({"kind": "max_error_rate", "rate": float(max_error_rate)})
    max_global_error_rate = kwargs.pop("max_global_error_rate", None)
    if max_global_error_rate is not None:
        constraints.append(
            {"kind": "max_global_error_rate", "rate": float(max_global_error_rate)}
        )
    if constraints:
        spec["constraints"] = constraints

    # over-saturation detection has no stable 0.7.1 spec mapping yet; ignore.
    kwargs.pop("over_saturation", None)

    # ── Outputs ─────────────────────────────────────────────────────────────
    output_dir = kwargs.pop("output_dir", None)
    outputs = list(kwargs.pop("outputs", None) or [])
    spec["outputs"] = _build_outputs(outputs, output_dir)

    return BenchmarkScenario.create(scenario=scenario, spec=spec)
