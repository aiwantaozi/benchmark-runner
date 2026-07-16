# The functions in this file are adapted from:
# https://github.com/vllm-project/guidellm/blob/v0.7.1/src/guidellm/cli/run.py
# Modifications have been made to fit project requirements.

"""
Benchmark Runner command-line interface entry point.

This is the main CLI for Benchmark Runner, customized for this project.
Key customizations:
- Uses custom progress and output modules (see benchmark_runner.progress, benchmark_runner.chained_progress).
- Adds the adaptive ramp auto-tune engine (--auto-tune, see benchmark_runner.auto_tune).
- Removes unnecessary subcommands, focusing on core benchmark and config functionality.

guidellm 0.7.1 notes:
- The benchmark is driven by ``BenchmarkScenario.create(spec=..., ...)`` +
  ``benchmark_generative_text(args=scenario, ...)``; the old flat
  ``BenchmarkGenerativeTextArgs`` was removed. The scenario spec is built from the
  CLI options by ``benchmark_runner.scenario_builder.build_scenario_args``.
- Request handlers still live in ``guidellm.backends.openai.request_handlers`` on
  ``OpenAIRequestHandlerFactory`` (registered by API PATH). The custom
  reasoning-aware handler is selected through the ``openai_http_error_detail``
  backend's ``request_handlers`` field (path -> registered handler name).
"""

from __future__ import annotations

import asyncio
import os
import platform
from pathlib import Path

import click
from pydantic import ValidationError
from benchmark_runner.chained_progress import ChainedBenchmarkerProgress
from benchmark_runner.openai_http_error_detail_backend import (
    ERROR_DETAIL_BACKEND_TYPE,
)
from benchmark_runner.scenario_builder import build_scenario_args
from guidellm.benchmark.entrypoints import benchmark_generative_text
from benchmark_runner.progress import ServerBenchmarkerProgress
from benchmark_runner.auto_tune import AutoTuneConfig, run_ramp

try:
    import uvloop
except ImportError:
    uvloop = None  # type: ignore[assignment] # Optional dependency

from guidellm.benchmark import (
    GenerativeConsoleBenchmarkerProgress,
    get_builtin_scenarios,
)
from guidellm.settings import print_config, settings as guidellm_settings
from guidellm.utils.console import Console
from guidellm.utils.default_group import DefaultGroupHandler
from guidellm.utils import cli as cli_tools

# guidellm 0.7.1 removed the ProfileType/StrategyType/BackendType Literal aliases
# that older builds imported. The valid profile/strategy names are the registered
# ProfileArgs kinds; we list them here for the --profile choice.
STRATEGY_PROFILE_CHOICES: list[str] = [
    "synchronous",
    "concurrent",
    "throughput",
    "sweep",
    "async",
    "constant",
    "poisson",
]

"""Available backend type choices for benchmark execution."""
BACKEND_CHOICES: list[str] = [
    "openai_http",
    ERROR_DETAIL_BACKEND_TYPE,
]

# Literal defaults for the CLI options. Replaces 0.6.0's
# ``BenchmarkGenerativeTextArgs.get_default(...)`` (that class was removed in
# guidellm 0.7.1). ``set_if_not_default`` drops any option left at its default so
# only user-provided values are threaded into the scenario spec.
_OPTION_DEFAULTS: dict = {
    "profile": "sweep",
    "rate": None,
    "backend": "openai_http",
    "backend_kwargs": None,
    "processor": None,
    "processor_args": None,
    "data_args": None,
    "data_samples": -1,
    "data_column_mapper": None,
    "data_sampler": None,
    "data_num_workers": None,
    "dataloader_kwargs": None,
    "random_seed": 42,
    "seed_increment": True,
    "output_dir": None,
    "outputs": None,
    "warmup": None,
    "cooldown": None,
    "rampup": None,
    "max_seconds": None,
    "max_requests": None,
    "max_errors": None,
    "max_error_rate": None,
    "max_global_error_rate": None,
}


def _opt_default(name: str):
    """Return the literal click default for an option (see ``_OPTION_DEFAULTS``)."""
    return _OPTION_DEFAULTS.get(name)


# guidellm 0.7.1's ``guidellm.utils.cli`` no longer ships ``parse_list_floats`` or
# ``parse_json`` (only parse_list / parse_kv_str / parse_overrides / Union /
# set_if_not_default remain). We provide local click callbacks with the same
# behavior the benchmark-runner options relied on.
def _split_floats(value: str) -> list[float]:
    return [float(part) for part in str(value).split(",") if part.strip() != ""]


def _cb_list_floats(ctx, param, value):
    """Parse comma-separated floats. For ``multiple`` options returns a tuple of
    lists (one per occurrence); otherwise a single list. ``None``/empty pass through.
    """
    if value is None or value == ():
        return None
    if isinstance(value, tuple):
        return tuple(_split_floats(item) for item in value)
    return _split_floats(value)


def _parse_json_scalar(value):
    import json as _json

    if value is None or isinstance(value, (dict, list, int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    try:
        return _json.loads(text)
    except (ValueError, TypeError):
        # Fall back to a bare string; downstream models coerce (e.g. warmup) or
        # the value is used verbatim.
        return value


def _cb_parse_json(ctx, param, value):
    """Parse a JSON string (or key=value-ish scalar). Handles ``multiple`` options
    by parsing each occurrence into a tuple."""
    if value is None or value == ():
        return None
    if isinstance(value, tuple):
        return tuple(_parse_json_scalar(item) for item in value)
    return _parse_json_scalar(value)


DISABLE_MACOS_WORKAROUNDS_ENV = "BENCHMARK_RUNNER_DISABLE_MACOS_WORKAROUNDS"
"""Set to 1/true/yes to disable runtime macOS defaults for process/data workers."""


def apply_macos_runtime_workarounds(kwargs: dict) -> None:
    if platform.system() != "Darwin":
        return

    if os.environ.get(DISABLE_MACOS_WORKAROUNDS_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return

    # Why this exists:
    # - GuideLLM defaults to multiprocessing "fork", which can hang on macOS in
    #   mixed runtime stacks (tokenizers/torch/http clients).
    # - See Python multiprocessing docs and macOS fork notes:
    #   https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    #   https://bugs.python.org/issue33725
    if (
        "GUIDELLM__MP_CONTEXT_TYPE" not in os.environ
        and guidellm_settings.mp_context_type == "fork"
    ):
        guidellm_settings.mp_context_type = "spawn"

    # Keep DataLoader single-process by default on macOS unless user overrides it.
    if "data_num_workers" not in kwargs:
        kwargs["data_num_workers"] = 0


@click.group()
@click.version_option(
    package_name="benchmark-runner", message="benchmark-runner version: %(version)s"
)
def cli():
    """Benchmark Runner CLI for benchmarking, preprocessing, and testing language models."""


@cli.group(
    help="Run a benchmark or load a previously saved benchmark report.",
    cls=DefaultGroupHandler,
    default="run",
)
def benchmark():
    """Benchmark commands for performance testing generative models."""


@benchmark.command(
    "run",
    help=(
        "Run a benchmark against a generative model. "
        "Supports multiple backends, data sources, strategies, and output formats. "
        "Configuration can be loaded from a scenario file or specified via options."
    ),
    context_settings={"auto_envvar_prefix": "BENCHMARK_RUNNER"},
)
@click.option(
    "--scenario",
    "-c",
    type=cli_tools.Union(
        click.Path(
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            path_type=Path,
        ),
        click.Choice(tuple(get_builtin_scenarios().keys())),
    ),
    default=None,
    help=(
        "Builtin scenario name or path to config file. "
        "CLI options override scenario settings."
    ),
)
@click.option(
    "--target",
    type=str,
    help="Target backend URL (e.g., http://localhost:8000).",
)
@click.option(
    "--data",
    type=str,
    multiple=True,
    help=(
        "HuggingFace dataset ID, path to dataset, path to data file "
        "(csv/json/jsonl/txt), or synthetic data config (json/key=value)."
    ),
)
@click.option(
    "--stages",
    "stages",
    callback=_cb_parse_json,
    default=None,
    help=(
        "JSON list of per-stage configs (v2.1 stages model). Each item: "
        '{"rate": float, "max_requests"?: int, "max_seconds"?: float}. When set, '
        "runs one single-rate concurrent benchmark per stage with that stage's own "
        "constraints, writing separate output files ({base}__stage{i}.{ext})."
    ),
)
# ── Adaptive ramp auto-tune ───────────────────────────────────────────────────
@click.option(
    "--auto-tune",
    "auto_tune",
    is_flag=True,
    default=False,
    help=(
        "Enable the adaptive ramp auto-tune engine (geometric bracket + binary "
        "search). Probes ONE answer: peak throughput (no SLA) or the SLA-capacity "
        "boundary (any --sla-* set). Mutually exclusive with --profile/--stages."
    ),
)
@click.option(
    "--axis",
    "axis",
    type=click.Choice(["rate", "concurrency"]),
    default="rate",
    help="Auto-tune load axis: 'rate' (constant, open-loop) or 'concurrency' "
    "(concurrent, closed-loop).",
)
@click.option(
    "--lower-bound",
    "lower_bound",
    type=float,
    default=1.0,
    help="Auto-tune knob lower bound (starting point).",
)
@click.option(
    "--upper-bound",
    "upper_bound",
    type=float,
    default=1024.0,
    help="Auto-tune knob upper bound (ceiling to prevent runaway doubling).",
)
@click.option(
    "--multiplier",
    "multiplier",
    type=float,
    default=None,
    help="Auto-tune per-point request multiplier (number = max(min_requests, "
    "round(knob*multiplier))). Default: 10 (concurrency) / 30 (rate).",
)
@click.option(
    "--min-requests",
    "min_requests",
    type=int,
    default=30,
    help="Auto-tune per-point minimum request count (measurement-window floor).",
)
@click.option(
    "--max-points",
    "max_points",
    type=int,
    default=12,
    help="Auto-tune max number of measured points (guards a too-flat curve).",
)
@click.option(
    "--max-total-seconds",
    "max_total_seconds",
    type=float,
    default=1800.0,
    help="Auto-tune total wall-clock budget across all points (seconds).",
)
@click.option(
    "--sla-avg-ttft-ms",
    "sla_avg_ttft_ms",
    type=float,
    default=None,
    help="Auto-tune SLA target: max acceptable avg TTFT in ms (setting any --sla-* "
    "switches the target to the SLA-capacity boundary).",
)
@click.option(
    "--sla-avg-tpot-ms",
    "sla_avg_tpot_ms",
    type=float,
    default=None,
    help="Auto-tune SLA target: max acceptable avg TPOT in ms.",
)
@click.option(
    "--sla-p99-ttft-ms",
    "sla_p99_ttft_ms",
    type=float,
    default=None,
    help="Auto-tune SLA target: max acceptable p99 TTFT in ms.",
)
@click.option(
    "--sla-p99-tpot-ms",
    "sla_p99_tpot_ms",
    type=float,
    default=None,
    help="Auto-tune SLA target: max acceptable p99 TPOT in ms.",
)
@click.option(
    "--sla-avg-latency-ms",
    "sla_avg_latency_ms",
    type=float,
    default=None,
    help="Auto-tune SLA target: max acceptable avg end-to-end request latency "
    "in ms (guidellm reports latency in seconds; converted internally).",
)
@click.option(
    "--sla-p99-latency-ms",
    "sla_p99_latency_ms",
    type=float,
    default=None,
    help="Auto-tune SLA target: max acceptable p99 end-to-end request latency "
    "in ms (guidellm reports latency in seconds; converted internally).",
)
@click.option(
    "--profile",
    "--rate-type",  # legacy alias
    "profile",
    default=_opt_default("profile"),
    type=click.Choice(STRATEGY_PROFILE_CHOICES),
    help=f"Benchmark profile type. Options: {', '.join(STRATEGY_PROFILE_CHOICES)}.",
)
@click.option(
    "--rate",
    callback=_cb_list_floats,
    multiple=True,
    default=_opt_default("rate"),
    help=(
        "Benchmark rate(s) to test. Meaning depends on profile: "
        "sweep=number of benchmarks, concurrent=concurrent requests, "
        "async/constant/poisson=requests per second."
    ),
)
# Backend configuration
@click.option(
    "--backend",
    "--backend-type",  # legacy alias
    "backend",
    type=click.Choice(BACKEND_CHOICES),
    default=_opt_default("backend"),
    help=f"Backend type. Options: {', '.join(BACKEND_CHOICES)}.",
)
@click.option(
    "--backend-kwargs",
    "--backend-args",  # legacy alias
    "backend_kwargs",
    callback=_cb_parse_json,
    default=_opt_default("backend_kwargs"),
    help="JSON string of arguments to pass to the backend.",
)
@click.option(
    "--model",
    default=None,
    type=str,
    help="Model ID to benchmark. If not provided, uses first available model.",
)
# Data configuration
@click.option(
    "--processor",
    default=_opt_default("processor"),
    type=str,
    help=(
        "Processor or tokenizer for token count calculations. "
        "If not provided, loads from model."
    ),
)
@click.option(
    "--processor-args",
    default=_opt_default("processor_args"),
    callback=_cb_parse_json,
    help="JSON string of arguments to pass to the processor constructor.",
)
@click.option(
    "--data-args",
    multiple=True,
    default=_opt_default("data_args"),
    callback=_cb_parse_json,
    help="JSON string of arguments to pass to dataset creation.",
)
@click.option(
    "--data-samples",
    default=_opt_default("data_samples"),
    type=int,
    help=(
        "Number of samples from dataset. -1 (default) uses all samples "
        "and dynamically generates more."
    ),
)
@click.option(
    "--data-column-mapper",
    default=_opt_default("data_column_mapper"),
    callback=_cb_parse_json,
    help="JSON string of column mappings to apply to the dataset.",
)
@click.option(
    "--data-sampler",
    default=_opt_default("data_sampler"),
    type=click.Choice(["shuffle"]),
    help="Data sampler type.",
)
@click.option(
    "--data-num-workers",
    default=_opt_default("data_num_workers"),
    type=int,
    help="Number of worker processes for data loading.",
)
@click.option(
    "--dataloader-kwargs",
    default=_opt_default("dataloader_kwargs"),
    callback=_cb_parse_json,
    help="JSON string of arguments to pass to the dataloader constructor.",
)
@click.option(
    "--random-seed",
    default=_opt_default("random_seed"),
    type=int,
    help="Random seed for reproducibility. In --auto-tune this is the seed base "
    "(each point uses random_seed + point_index).",
)
@click.option(
    "--seed-increment/--no-seed-increment",
    "seed_increment",
    default=_opt_default("seed_increment"),
    help="In --auto-tune, increment the seed per point (base + point_index) so "
    "points differ (default). --no-seed-increment pins the same seed for every "
    "point. Only affects the Random synthetic dataset.",
)
# Output configuration
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=_opt_default("output_dir"),
    help="The directory path to save file output types in",
)
@click.option(
    "--outputs",
    callback=cli_tools.parse_list,
    multiple=True,
    default=_opt_default("outputs"),
    help=(
        "The filename.ext for each of the outputs to create or the "
        "alises (json, csv, html) for the output files to create with "
        "their default file names (benchmark.[EXT])"
    ),
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=None,
    help=(
        "Legacy parameter for the output path to save the output result to. "
        "Resolves to fill in output-dir and outputs based on input path."
    ),
)
@click.option(
    "--disable-console",
    "--disable-console-outputs",  # legacy alias
    "disable_console",
    is_flag=True,
    help=(
        "Disable all outputs to the console (updates, interactive progress, results)."
    ),
)
@click.option(
    "--disable-console-interactive",
    "--disable-progress",  # legacy alias
    "disable_console_interactive",
    is_flag=True,
    help="Disable interactive console progress updates.",
)
# Aggregators configuration
@click.option(
    "--warmup",
    "--warmup-percent",  # legacy alias
    "warmup",
    default=_opt_default("warmup"),
    callback=_cb_parse_json,
    help=(
        "Warmup specification: int, float, or dict as string (json or key=value). "
        "Controls time or requests before measurement starts."
    ),
)
@click.option(
    "--cooldown",
    "--cooldown-percent",  # legacy alias
    "cooldown",
    default=_opt_default("cooldown"),
    callback=_cb_parse_json,
    help=(
        "Cooldown specification: int, float, or dict as string (json or key=value). "
        "Controls time or requests after measurement ends."
    ),
)
@click.option(
    "--rampup",
    type=float,
    default=_opt_default("rampup"),
    help=(
        "The time, in seconds, to ramp up the request rate over. "
        "Only applicable for Throughput/Concurrent strategies"
    ),
)
@click.option(
    "--sample-requests",
    "--output-sampling",  # legacy alias
    "sample_requests",
    type=int,
    help=(
        "Number of sample requests per status to save. "
        "None (default) saves all, recommended: 20."
    ),
)
# Constraints configuration
@click.option(
    "--max-seconds",
    type=float,
    default=_opt_default("max_seconds"),
    help=(
        "Maximum seconds per benchmark. "
        "If None, runs until max_requests or data exhaustion."
    ),
)
@click.option(
    "--max-requests",
    type=int,
    default=_opt_default("max_requests"),
    help=(
        "Maximum requests per benchmark. "
        "If None, runs until max_seconds or data exhaustion."
    ),
)
@click.option(
    "--max-errors",
    type=int,
    default=_opt_default("max_errors"),
    help="Maximum errors before stopping the benchmark.",
)
@click.option(
    "--max-error-rate",
    type=float,
    default=_opt_default("max_error_rate"),
    help="Maximum error rate before stopping the benchmark.",
)
@click.option(
    "--max-global-error-rate",
    type=float,
    default=_opt_default("max_global_error_rate"),
    help="Maximum global error rate across all benchmarks.",
)
@click.option(
    "--over-saturation",
    "over_saturation",
    callback=_cb_parse_json,
    default=None,
    help=(
        "Enable over-saturation detection. "
        "Pass a JSON dict with configuration "
        '(e.g., \'{"enabled": true, "min_seconds": 30}\'). '
        "Defaults to None (disabled)."
    ),
)
@click.option(
    "--detect-saturation",
    "--default-over-saturation",
    "over_saturation",
    callback=_cb_parse_json,
    flag_value='{"enabled": true}',
    help="Enable over-saturation detection with default settings.",
)
@click.option(
    "--progress-url",
    type=str,
    default=None,
    help="URL to send benchmark progress updates to server.",
)
@click.option(
    "--progress-auth",
    type=str,
    default=None,
    help="Authentication token or credential for progress update requests.",
)
def run(**kwargs):  # noqa: C901
    # Only set CLI args that differ from click defaults
    kwargs = cli_tools.set_if_not_default(click.get_current_context(), **kwargs)
    apply_macos_runtime_workarounds(kwargs)

    # guidellm 0.7.1: target/model are backend concerns and are folded into the
    # scenario's ``spec.backend`` (see scenario_builder.build_scenario_args). We
    # gather any extra backend options from --backend-kwargs here, plus target and
    # model, and normalize the custom-handler override to the backend's
    # ``request_handlers`` field (keyed by API path -> registered handler NAME).
    #
    # NOTE (0.7.1): the ``OpenAIRequestHandlerFactory`` registers handlers by API
    # PATH, and our ``openai_http_error_detail`` backend exposes a
    # ``request_handlers`` field (path -> handler name string) that it resolves to
    # classes at runtime. We therefore pass handler names as STRINGS and keep the
    # key path-based. A legacy ``response_handlers`` (request_type keyed) dict is
    # translated to the path-keyed ``request_handlers`` form for compatibility.
    backend_kwargs = dict(kwargs.pop("backend_kwargs", None) or {})
    for alias in ("target", "model"):
        value = kwargs.pop(alias, None)
        if value is not None:
            backend_kwargs[alias] = value

    _normalize_request_handlers(backend_kwargs)
    if backend_kwargs:
        kwargs["backend_kwargs"] = backend_kwargs

    # Handle output path remapping
    if (output_path := kwargs.pop("output_path", None)) is not None:
        if kwargs.get("output_dir", None) is not None:
            raise click.BadParameter("Cannot use --output-path with --output-dir.")
        path = Path(output_path)
        if path.is_dir():
            kwargs["output_dir"] = path
        else:
            kwargs["output_dir"] = path.parent
            kwargs["outputs"] = (path.name,)

    # Handle console options
    disable_console = kwargs.pop("disable_console", False)
    disable_console_interactive = (
        kwargs.pop("disable_console_interactive", False) or disable_console
    )
    console = Console() if not disable_console else None

    progress_url = kwargs.pop("progress_url", None)
    progress_auth = kwargs.pop("progress_auth", None)
    # Keep a reference so the multi-run loops (ramp / stages / input matrix) can
    # tell it which run (run_index / run_total) is executing, for a smooth overall
    # progress that doesn't reset between runs (see progress-design.md).
    server_progress = (
        ServerBenchmarkerProgress(
            progress_url=progress_url, progress_auth=progress_auth
        )
        if progress_url
        else None
    )
    progress_chain = [
        *([server_progress] if server_progress else []),
        *(
            [GenerativeConsoleBenchmarkerProgress()]
            if not disable_console_interactive
            else []
        ),
    ]
    progress = ChainedBenchmarkerProgress(progress_chain) if progress_chain else None

    # Pop project-specific args that are not guidellm args before create().
    stages = kwargs.pop("stages", None)
    # Auto-tune knobs (consumed by the ramp engine, never passed to guidellm).
    auto_tune = kwargs.pop("auto_tune", False)
    axis = kwargs.pop("axis", "rate")
    lower_bound = kwargs.pop("lower_bound", 1.0)
    upper_bound = kwargs.pop("upper_bound", 1024.0)
    multiplier = kwargs.pop("multiplier", None)
    min_requests = kwargs.pop("min_requests", 30)
    max_points = kwargs.pop("max_points", 12)
    max_total_seconds = kwargs.pop("max_total_seconds", 1800.0)
    # SLA thresholds: up to 6 optional "<=" latency targets (avg + p99 of TTFT,
    # TPOT, end-to-end latency), all in ms. Any subset may be set.
    sla_avg_ttft_ms = kwargs.pop("sla_avg_ttft_ms", None)
    sla_avg_tpot_ms = kwargs.pop("sla_avg_tpot_ms", None)
    sla_p99_ttft_ms = kwargs.pop("sla_p99_ttft_ms", None)
    sla_p99_tpot_ms = kwargs.pop("sla_p99_tpot_ms", None)
    sla_avg_latency_ms = kwargs.pop("sla_avg_latency_ms", None)
    sla_p99_latency_ms = kwargs.pop("sla_p99_latency_ms", None)

    def _run_once(local_kwargs):
        try:
            args = build_scenario_args(local_kwargs)
        except ValidationError as err:
            # Translate pydantic validation error to click argument error
            errs = err.errors(
                include_url=False, include_context=True, include_input=True
            )
            param_name = "--" + str(errs[0]["loc"][0]).replace("_", "-")
            raise click.BadParameter(
                errs[0]["msg"],
                ctx=click.get_current_context(),
                param_hint=param_name,
            ) from err

        asyncio.run(
            benchmark_generative_text(
                args=args,
                progress=progress,
                console=console,
            )
        )

    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    def _suffix_output(name: str, tag: str) -> str:
        if "." in name:
            stem, _, ext = name.rpartition(".")
            return f"{stem}__{tag}.{ext}"
        return f"{name}__{tag}"

    def _output_base(name: str) -> str:
        # "123.dual_json" -> "123"; the ramp writes {base}__p{index}.dual_json.
        return name.rpartition(".")[0] if "." in name else name

    if auto_tune:
        # Adaptive ramp: one single-strategy guidellm run per probed knob point.
        # The target (SLA boundary vs throughput saturation) is derived from
        # whether any --sla-* is set (see AutoTuneConfig.target).
        cfg = AutoTuneConfig(
            axis=axis,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            multiplier=multiplier,
            min_requests=min_requests,
            max_points=max_points,
            max_total_seconds=max_total_seconds,
            sla_avg_ttft_ms=sla_avg_ttft_ms,
            sla_avg_tpot_ms=sla_avg_tpot_ms,
            sla_p99_ttft_ms=sla_p99_ttft_ms,
            sla_p99_tpot_ms=sla_p99_tpot_ms,
            sla_avg_latency_ms=sla_avg_latency_ms,
            sla_p99_latency_ms=sla_p99_latency_ms,
            random_seed_base=kwargs.get("random_seed", 42) or 42,
            seed_increment=kwargs.get("seed_increment", True),
        )
        base_outputs = list(kwargs.get("outputs") or ["benchmarks.dual_json"])
        output_base = _output_base(base_outputs[0])
        # Per-point kwargs share everything except profile/rate/seed/outputs/
        # max_requests, which the ramp engine sets for each run.
        base_kwargs = dict(kwargs)
        for k in ("profile", "rate", "outputs", "max_requests", "max_seconds"):
            base_kwargs.pop(k, None)
        print(
            f"[DEBUG] Auto-tune ramp: axis={axis} target={cfg.target} "
            f"bounds=[{lower_bound},{upper_bound}] multiplier={cfg.resolved_multiplier} "
            f"min_requests={min_requests} max_points={max_points} "
            f"max_total_seconds={max_total_seconds} base={output_base}"
        )
        # The ramp keeps server_progress in the per-point runs so guidellm's live
        # on_benchmark_update callbacks drive a smooth (within-point) server bar; the
        # ramp sets run_index/run_total per point (see _prep_progress) so the bar
        # stays proportional and never hits 100 before it explicitly finalizes.
        points = asyncio.run(
            run_ramp(
                cfg=cfg,
                base_kwargs=base_kwargs,
                output_base=output_base,
                server_progress=server_progress,
                progress=progress,
                console=console,
            )
        )
        print(f"[DEBUG] Auto-tune ramp finished: {len(points)} point(s) measured")
    elif stages:
        # v2.1 stages: one single-rate `concurrent` run per stage, each carrying
        # its own max_requests / max_seconds. Each writes a separate output file
        # {base}__stage{i}.{ext}.
        base_outputs = list(kwargs.get("outputs") or [])
        for i, stage in enumerate(stages):
            if server_progress is not None:
                server_progress.run_index = i
                server_progress.run_total = len(stages)
            local = dict(kwargs)
            local["profile"] = "concurrent"
            local["rate"] = [float(stage["rate"])]
            # Per-stage seed mirrors the ramp: increment by stage index unless the
            # user pinned a fixed seed (only affects the Random synthetic dataset).
            if (
                kwargs.get("seed_increment", True)
                and kwargs.get("random_seed") is not None
            ):
                local["random_seed"] = int(kwargs["random_seed"]) + i
            if stage.get("max_requests") is not None:
                local["max_requests"] = int(stage["max_requests"])
            if stage.get("max_seconds") is not None:
                local["max_seconds"] = float(stage["max_seconds"])
            if base_outputs:
                local["outputs"] = tuple(
                    _suffix_output(o, f"stage{i}") for o in base_outputs
                )
            print(
                f"[DEBUG] Stage run {i}: rate={stage['rate']} "
                f"max_requests={local.get('max_requests')} "
                f"max_seconds={local.get('max_seconds')}"
            )
            _run_once(local)
    else:
        _run_once(dict(kwargs))


def _normalize_request_handlers(backend_kwargs: dict) -> None:
    """Normalize a custom request-handler override to the backend's expected shape.

    guidellm 0.7.1 registers request handlers on ``OpenAIRequestHandlerFactory`` by
    API PATH. Our ``openai_http_error_detail`` backend exposes a ``request_handlers``
    field keyed by API path -> registered handler NAME (a string) and resolves the
    names to classes itself. We therefore keep handler names as STRINGS here.

    For backward compatibility we also accept a legacy ``response_handlers`` dict
    keyed by request_type name (e.g. "chat_completions") and translate it into the
    path-keyed ``request_handlers`` form.
    """
    _REQUEST_TYPE_TO_PATH = {
        "chat_completions": "/v1/chat/completions",
        "text_completions": "/v1/completions",
        "audio_transcriptions": "/v1/audio/transcriptions",
        "audio_translations": "/v1/audio/translations",
    }

    legacy = backend_kwargs.pop("response_handlers", None)
    if isinstance(legacy, dict):
        request_handlers = dict(backend_kwargs.get("request_handlers") or {})
        for key, value in legacy.items():
            path = _REQUEST_TYPE_TO_PATH.get(key, key)
            # Explicit request_handlers entries win on clash.
            request_handlers.setdefault(path, value)
        backend_kwargs["request_handlers"] = request_handlers


@cli.command(
    short_help="Show configuration settings.",
    help="Display environment variables for configuring GuideLLM behavior.",
)
def config():
    print_config()


if __name__ == "__main__":
    cli()
