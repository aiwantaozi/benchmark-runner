# The functions in this file are adapted from:
# https://github.com/vllm-project/guidellm/blob/v0.7.3/src/guidellm/cli/run.py
# Modifications have been made to fit project requirements.

"""
Benchmark Runner command-line interface entry point.

This is the main CLI for Benchmark Runner, customized for this project.
Key customizations:
- Uses custom progress and output modules (see benchmark_runner.progress, benchmark_runner.chained_progress).
- Adds the adaptive ramp auto-tune engine (--auto-tune, see benchmark_runner.auto_tune).
- Removes unnecessary subcommands, focusing on core benchmark and config functionality.

guidellm 0.7.x notes:
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
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import click
from pydantic import ValidationError
from benchmark_runner.chained_progress import ChainedBenchmarkerProgress
from benchmark_runner.openai_http_error_detail_backend import (
    ERROR_DETAIL_BACKEND_TYPE,
)
from benchmark_runner.scenario_builder import build_scenario_args
from benchmark_runner.sharegpt_adapter import prepare_datasets
from guidellm.benchmark.entrypoints import benchmark_generative_text
from benchmark_runner.progress import ServerBenchmarkerProgress
from benchmark_runner.auto_tune import (
    RAMP_OUTCOME_POINTS_VERSION,
    AutoTuneConfig,
    RampOutcome,
    run_ramp,
    # The ramp's guidellm-benchmark -> PointMetrics mapping, reused verbatim for
    # manual stages. Private to auto_tune only in the underscore sense: it is the
    # shared definition of "one measured point", and a second copy here is exactly
    # how the stage table and the ramp table would drift apart (which metric TPOT
    # reads from, whether latency is seconds or ms). Imported rather than renamed
    # because five tests call it by name.
    _normalize,
)
from benchmark_runner.chart import (
    CHART_AUTO,
    CHART_MODES,
    detect_style,
    render_curve_report,
)

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

# Sidecar name for the ramp outcome, deliberately NOT matching the manager's
# "{base}__p{index}.json" point glob.
RAMP_OUTCOME_SUFFIX = "__ramp.json"

# Sidecar name for a --stages ladder's curve. Deliberately NOT the ramp's suffix:
# the two carry different claims (a search that converged vs. a list of rungs the
# user chose), and a reader that cannot tell them apart would report one as the
# other. Also not matched by the manager's "{base}__p{index}.json" point glob.
CURVE_OUTCOME_SUFFIX = "__curve.json"
# Bumped independently of RAMP_OUTCOME_VERSION — these are two schemas that happen
# to share a renderer, not one schema with two writers.
CURVE_OUTCOME_VERSION = 2


def _finish_stages(
    points: list,
    *,
    axis: str,
    output_dir: Path | None,
    output_base: str,
    elapsed: float,
    chart_mode: str,
    console: Any,
    log: Callable[[str], None],
    warn: Callable[[str], None],
) -> None:
    """Sidecar + terminal report for a finished ``--stages`` ladder.

    A stage list is a manual sweep: the user picks the load points instead of the
    ramp searching for them, but what comes back is the same curve, so it gets the
    same chart and the same overload verdicts.

    What it does NOT get is a recommendation. The ramp earns "run at 31 req/s" by
    bracketing and converging on it; a stage list has measured only the rungs it
    was given, and reusing that wording would claim a search that never happened.
    ``target="stages"`` is what tells the renderer to report the ladder's peak and
    whether the ladder actually contained one — see ``chart._verdict``.
    """
    if not points:
        return
    outcome = {
        "version": CURVE_OUTCOME_VERSION,
        "points": [asdict(p) for p in points],
        "axis": axis,
        "target": "stages",
        # A ladder ends because it ran out of rungs. There is no search to stop, so
        # the field carries the one fact that IS true rather than a borrowed reason.
        "stop_reason": "stages_completed",
        "bracket_reason": "",
        "points_measured": len(points),
        "slo_bracket": None,
        "slo_thresholds": {},
        "elapsed_seconds": round(elapsed, 2),
        "probe_ceiling": None,
    }
    _write_curve_sidecar(output_dir, output_base, outcome, log, warn)
    if console is not None:
        _emit_curve_report(
            outcome, chart_mode, stream=getattr(console, "file", sys.stdout)
        )


def _write_curve_sidecar(
    output_dir: Path | None,
    output_base: str,
    outcome: dict,
    log: Callable[[str], None],
    warn: Callable[[str], None],
) -> None:
    """Write ``{base}__curve.json`` beside the stage reports.

    A separate suffix from the ramp's, because the two say different things and a
    consumer must be able to tell them apart by name: ``__ramp.json`` carries the
    reasons an adaptive SEARCH ended, ``__curve.json`` carries a ladder the user
    wrote out. Neither name is matched by gpustack's point glob (which requires a
    ``{id}__p`` prefix), so this adds a file to the benchmark directory without
    entering any existing collection.

    Best-effort for the same reason the ramp's is: the stage reports are the
    measurement, and a diagnostic file must not fail a run that succeeded.
    """
    try:
        directory = Path(output_dir) if output_dir is not None else Path.cwd()
        path = directory / f"{output_base}{CURVE_OUTCOME_SUFFIX}"
        path.write_text(json.dumps(outcome, indent=2, sort_keys=True), encoding="utf-8")
        log(f"Wrote stage curve: {path}")
    except Exception as e:  # noqa: BLE001 - diagnostics must not break the run
        warn(f"Could not write stage curve sidecar: {e}")


def _emit_curve_report(
    outcome: dict,
    mode: str,
    stream: Any = None,
    style: Any = None,
    strict: bool = False,
) -> None:
    """Print the terminal report for a finished ramp or stage ladder.

    Takes the sidecar DICT, not the RampOutcome, so the live run and the ``chart``
    subcommand render from the same input: whatever the live path shows is exactly
    what re-rendering the saved file reproduces. Best-effort for the same reason
    the sidecar write is — the measurement is in the files, and a run must not
    fail because a terminal could not be drawn on.
    """
    stream = stream if stream is not None else sys.stdout
    try:
        bracket = outcome.get("slo_bracket") or [None, None]
        lines = render_curve_report(
            outcome.get("points") or [],
            axis=outcome.get("axis") or "rate",
            target=outcome.get("target") or "saturation",
            stop_reason=outcome.get("stop_reason") or "",
            bracket_reason=outcome.get("bracket_reason") or "",
            slo_met_knob=bracket[0] if len(bracket) > 0 else None,
            slo_first_fail_knob=bracket[1] if len(bracket) > 1 else None,
            slo_thresholds=outcome.get("slo_thresholds") or {},
            elapsed_seconds=outcome.get("elapsed_seconds"),
            probe_ceiling=outcome.get("probe_ceiling"),
            mode=mode,
            style=style if style is not None else detect_style(stream),
        )
        for line in lines:
            print(line, file=stream)
        stream.flush()
    except Exception as e:  # noqa: BLE001 - a chart must not break a benchmark
        message = f"Could not render {_sidecar_hint(outcome)}: {e}"
        # strict = the `chart` subcommand, whose only job is to draw. The live run
        # warns instead: a benchmark that measured fine must not die because a
        # terminal could not be written to.
        if strict:
            raise click.ClickException(message)
        print(f"[WARN] {message}", file=sys.stderr)


def _sidecar_hint(outcome: dict) -> str:
    """Name the thing that failed to draw, for an error a user has to act on."""
    return "the stage curve" if outcome.get("target") == "stages" else "the ramp report"


def _write_ramp_outcome(
    output_dir: Path | None,
    output_base: str,
    outcome: RampOutcome,
    log: Callable[[str], None],
    warn: Callable[[str], None],
) -> None:
    """Write ``{base}__ramp.json`` beside the point files.

    Best-effort: the points ARE the measurement and a run must not fail because a
    diagnostic file could not be written. A missing sidecar reads as "no facts
    available", which every consumer has to handle anyway (runs from before this
    existed, and non-ramp modes, have none).

    ``log``/``warn`` are injected rather than printed directly so this respects the
    caller's console policy (see ``--disable-console``).
    """
    try:
        directory = Path(output_dir) if output_dir is not None else Path.cwd()
        path = directory / f"{output_base}{RAMP_OUTCOME_SUFFIX}"
        path.write_text(
            json.dumps(outcome.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        log(f"Wrote ramp outcome: {path}")
    except Exception as e:  # noqa: BLE001 - diagnostics must not break the run
        warn(f"Could not write ramp outcome sidecar: {e}")


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
    "max_concurrency": None,
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
    "probe_saturation": True,
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


# guidellm 0.7.x's ``guidellm.utils.cli`` no longer ships ``parse_list_floats`` or
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
        "runs one single-strategy benchmark per stage with that stage's own "
        "constraints, writing separate output files ({base}__stage{i}.{ext}). "
        "--axis selects what each stage's `rate` means (rate -> constant req/s, "
        "concurrency -> concurrent streams)."
    ),
)
# ── Adaptive ramp auto-tune ───────────────────────────────────────────────────
@click.option(
    "--auto-tune",
    "auto_tune",
    is_flag=True,
    default=False,
    help=(
        "Enable the adaptive ramp auto-tune engine: a geometric bracket, then a "
        "unimodal search for the throughput argmax (no SLO) or a bisection for the "
        "SLO-capacity boundary (any --slo-* set). Probes ONE answer. Mutually "
        "exclusive with --profile/--stages."
    ),
)
@click.option(
    "--axis",
    "axis",
    type=click.Choice(["rate", "concurrency"]),
    default="rate",
    help="Load axis: 'rate' (constant, open-loop) or 'concurrency' (concurrent, "
    "closed-loop). Applies to both --auto-tune ramp points and --stages runs.",
)
@click.option(
    "--lower-bound",
    "lower_bound",
    type=float,
    default=4.0,
    help="Auto-tune knob lower bound: where the ramp starts AND the floor it "
    "never measures below (the range is hard at both ends). Default 4 skips the "
    "low-information 1/2 points. Raise with care: a floor at or above the SLO "
    "boundary fails the first point, leaving no bracket to bisect.",
)
@click.option(
    "--upper-bound",
    "upper_bound",
    type=float,
    default=8192.0,
    help="Auto-tune knob upper bound: hard ceiling on offered load. Throughput "
    "searches can stop on saturation; SLO searches continue while latency targets "
    "and request success pass, even when throughput plateaus, with a protective "
    "stop for sustained significant total-throughput decline. Set a practical "
    "bound: a loose SLO can reach it or exhaust the budget without locating a "
    "boundary. The saturation probe's soft cap never overrides this ceiling.",
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
    default=100,
    help="Auto-tune per-point minimum request count (measurement-window floor). "
    "100 keeps p99 off the maximum: with n samples only n/100 sit above p99.",
)
@click.option(
    "--max-points",
    "max_points",
    type=int,
    default=18,
    help="Auto-tune max number of measured points (guards a too-flat curve). A "
    "ceiling, not a quota: a converged search stops early, so this is sized for "
    "the expensive case (an SLO bisection on a large deployment) and costs a small "
    "one nothing. Below ~18 the largest deployments report a boundary the "
    "bisection never closed on.",
)
@click.option(
    "--max-total-seconds",
    "max_total_seconds",
    type=float,
    default=5400.0,
    help="Auto-tune total wall-clock budget across all points (seconds). Also "
    "handed to each point as max_seconds, so a budget the run cannot fit shortens "
    "the last measurements rather than dropping them.",
)
@click.option(
    "--slo-avg-ttft-ms",
    "slo_avg_ttft_ms",
    type=float,
    default=None,
    help="Auto-tune SLO target: max acceptable avg TTFT in ms (setting any --slo-* "
    "switches the target to the SLO-capacity boundary).",
)
@click.option(
    "--slo-avg-itl-ms",
    # Accepted under its former name too, so a gpustack that predates the rename
    # keeps working against a newer runner image (and vice versa): the two ship
    # separately, and a flag mismatch would fail the run outright.
    "--slo-avg-tpot-ms",
    "slo_avg_itl_ms",
    type=float,
    default=None,
    help="Auto-tune SLO target: max acceptable avg inter-token latency in ms "
    "(guidellm's `inter_token_latency_ms`; gpustack calls it TPOT).",
)
@click.option(
    "--slo-p95-ttft-ms",
    "slo_p95_ttft_ms",
    type=float,
    default=None,
    help="Auto-tune SLO target: max acceptable p95 TTFT in ms.",
)
@click.option(
    "--slo-p95-itl-ms",
    # Accepted under its former name too, so a gpustack that predates the rename
    # keeps working against a newer runner image (and vice versa): the two ship
    # separately, and a flag mismatch would fail the run outright.
    "--slo-p95-tpot-ms",
    "slo_p95_itl_ms",
    type=float,
    default=None,
    help="Auto-tune SLO target: max acceptable p95 inter-token latency in ms "
    "(guidellm's `inter_token_latency_ms`; gpustack calls it TPOT).",
)
@click.option(
    "--slo-p99-ttft-ms",
    "slo_p99_ttft_ms",
    type=float,
    default=None,
    help="Auto-tune SLO target: max acceptable p99 TTFT in ms.",
)
@click.option(
    "--slo-p99-itl-ms",
    # Accepted under its former name too, so a gpustack that predates the rename
    # keeps working against a newer runner image (and vice versa): the two ship
    # separately, and a flag mismatch would fail the run outright.
    "--slo-p99-tpot-ms",
    "slo_p99_itl_ms",
    type=float,
    default=None,
    help="Auto-tune SLO target: max acceptable p99 inter-token latency in ms "
    "(guidellm's `inter_token_latency_ms`; gpustack calls it TPOT).",
)
@click.option(
    "--slo-avg-latency-ms",
    "slo_avg_latency_ms",
    type=float,
    default=None,
    help="Auto-tune SLO target: max acceptable avg end-to-end request latency "
    "in ms (guidellm reports latency in seconds; converted internally).",
)
@click.option(
    "--slo-p95-latency-ms",
    "slo_p95_latency_ms",
    type=float,
    default=None,
    help="Auto-tune SLO target: max acceptable p95 end-to-end request latency "
    "in ms (guidellm reports latency in seconds; converted internally).",
)
@click.option(
    "--slo-p99-latency-ms",
    "slo_p99_latency_ms",
    type=float,
    default=None,
    help="Auto-tune SLO target: max acceptable p99 end-to-end request latency "
    "in ms (guidellm reports latency in seconds; converted internally).",
)
@click.option(
    "--max-concurrency",
    "max_concurrency",
    type=int,
    default=_opt_default("max_concurrency"),
    help="Cap on in-flight requests. Required by the `throughput` profile (and "
    "`sweep`'s throughput pass, including the --auto-tune saturation probe), "
    "where it defaults to 512 to match guidellm's own sweep default. Optional "
    "for the open-loop rate profiles (constant/poisson/async), where it bounds "
    "pile-up when the server can't keep up. Ignored by concurrent/synchronous.",
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
    "--sharegpt-min-input-tokens",
    "sharegpt_input_min",
    type=click.IntRange(min=1),
    default=None,
)
@click.option(
    "--sharegpt-max-input-tokens",
    "sharegpt_input_max",
    type=click.IntRange(min=1),
    default=None,
)
@click.option("--sharegpt-max-output-tokens", type=click.IntRange(min=1), default=None)
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
    "(each point uses random_seed + point_index) and applies whether or not you "
    "pass it. With --stages it is what OPTS IN to per-stage variation: pass any "
    "value (42 included) to get random_seed + stage_index per stage; leave it out "
    "and every stage shares one seed, i.e. identical prompts, so only the load "
    "differs between stages. Affects the Random synthetic dataset only.",
)
@click.option(
    "--seed-increment/--no-seed-increment",
    "seed_increment",
    default=_opt_default("seed_increment"),
    help="Increment the seed per run (base + index) so successive runs use "
    "different prompts (default). In --auto-tune this is always in force. With "
    "--stages it applies only when --random-seed was passed explicitly. "
    "--no-seed-increment pins the same seed for every point/stage. Only affects "
    "the Random synthetic dataset.",
)
@click.option(
    "--probe-saturation/--no-probe-saturation",
    "probe_saturation",
    default=_opt_default("probe_saturation"),
    help="In --auto-tune (rate axis, throughput target), run one throughput "
    "probe up front to measure the server's ceiling rps and use it as a SOFT cap "
    "(~ceiling x 1.2) on the ramp, so the sweep doesn't spend its budget doubling "
    "far past saturation (default on). The ramp still starts at --lower-bound, "
    "and the cap never overrides --upper-bound. --no-probe-saturation disables it.",
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
@click.option(
    "--chart",
    "chart_mode",
    type=click.Choice(list(CHART_MODES)),
    default=CHART_AUTO,
    show_default=True,
    help=(
        "Terminal report printed after an --auto-tune ramp: 'auto' = load/throughput "
        "chart, latency chart, stage table and the recommended operating point; "
        "'all' additionally plots offered-vs-achieved rate and the "
        "latency-throughput frontier; 'none' = table and verdict only. Ignored by "
        "non-ramp modes and silenced by --disable-console."
    ),
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
    # guidellm's MaxErrorRateConstraintArgs.rate is gt=0, lt=1 — an OPEN interval,
    # so 0 and 1 are both rejected. Left unchecked here, `--max-error-rate 1.0`
    # only fails inside BenchmarkScenario.create, i.e. after the process is up but
    # before a single request goes out. Range-check it at the CLI boundary so the
    # error names the option the user actually typed. Either endpoint means "no
    # constraint" anyway (1 = tolerate everything, 0 = tolerate nothing, which this
    # constraint cannot express — that is --max-errors' job): omit the option.
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    default=_opt_default("max_error_rate"),
    help="Maximum error rate (fraction, strictly between 0 and 1) before stopping "
    "the benchmark. Omit for no error-rate constraint.",
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
@click.option(
    "--progress-ca-cert",
    "progress_ca_cert",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="CA bundle used to verify an HTTPS --progress-url, for a server behind a "
    "CA this image's trust store does not carry. Scoped to the progress channel: "
    "unlike SSL_CERT_FILE it does not replace the trust store for the rest of the "
    "process, so a bundle holding only a private CA still leaves other TLS calls "
    "(a Hugging Face processor fetch, say) able to verify public roots.",
)
@click.option(
    "--progress-insecure-skip-tls-verify",
    "progress_insecure_skip_tls_verify",
    is_flag=True,
    default=False,
    help="Skip TLS verification when reporting progress to an HTTPS --progress-url. "
    "Last resort for when no bundle can verify the server at all; prefer "
    "--progress-ca-cert. Takes precedence over it when both are given. Affects the "
    "progress channel only -- never the benchmarked --target.",
)
def run(**kwargs):  # noqa: C901
    # Only set CLI args that differ from click defaults
    kwargs = cli_tools.set_if_not_default(click.get_current_context(), **kwargs)
    input_min = kwargs.get("sharegpt_input_min")
    input_max = kwargs.get("sharegpt_input_max")
    if input_min is not None and input_max is not None and input_min > input_max:
        raise click.BadParameter(
            "must not exceed --sharegpt-max-input-tokens",
            param_hint="--sharegpt-min-input-tokens",
        )
    apply_macos_runtime_workarounds(kwargs)

    # guidellm 0.7.x: target/model are backend concerns and are folded into the
    # scenario's ``spec.backend`` (see scenario_builder.build_scenario_args). We
    # gather any extra backend options from --backend-kwargs here, plus target and
    # model, and normalize the custom-handler override to the backend's
    # ``request_handlers`` field (keyed by API path -> registered handler NAME).
    #
    # NOTE (0.7.x): the ``OpenAIRequestHandlerFactory`` registers handlers by API
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
    chart_mode = kwargs.pop("chart_mode", CHART_AUTO)
    disable_console = kwargs.pop("disable_console", False)
    disable_console_interactive = (
        kwargs.pop("disable_console_interactive", False) or disable_console
    )
    console = Console() if not disable_console else None

    def _log(message: str) -> None:
        """Runner-level diagnostics (which mode ran, what it wrote, why).

        Routed through the console so ``--disable-console`` ("Disable all outputs to
        the console") actually holds for these too. They used to be bare ``print``s,
        which the flag could not reach — a run asked to stay quiet still emitted
        ``[DEBUG] ...`` lines onto stdout, interleaved with whatever was consuming it.
        """
        if console is not None:
            console.print_update(message, status="debug")

    def _warn(message: str) -> None:
        """A problem the operator has to see even with the console silenced.

        stderr, not the console: the runner is driven headless by gpustack, which
        only has the process' streams, and a warning that a diagnostic file could not
        be written must not be swallowed by a display flag. stderr also keeps it out
        of whatever is parsing stdout.
        """
        print(f"[WARN] {message}", file=sys.stderr)

    progress_url = kwargs.pop("progress_url", None)
    progress_auth = kwargs.pop("progress_auth", None)
    progress_ca_cert = kwargs.pop("progress_ca_cert", None)
    # `set_if_not_default` drops the flag while it sits at its False default, so
    # read it with a default rather than assuming the key is present.
    progress_insecure_skip_tls_verify = kwargs.pop(
        "progress_insecure_skip_tls_verify", False
    )
    if progress_url and progress_insecure_skip_tls_verify:
        _warn(
            "--progress-insecure-skip-tls-verify is enabled: TLS verification is "
            "disabled for progress updates to the server."
            + (" --progress-ca-cert is ignored." if progress_ca_cert else "")
        )
    # Keep a reference so the multi-run loops (ramp / stages / input matrix) can
    # tell it which run (run_index / run_total) is executing, for a smooth overall
    # progress that doesn't reset between runs (see progress-design.md).
    server_progress = (
        ServerBenchmarkerProgress(
            progress_url=progress_url,
            progress_auth=progress_auth,
            ca_cert=progress_ca_cert,
            insecure_skip_tls_verify=progress_insecure_skip_tls_verify,
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
    lower_bound = kwargs.pop("lower_bound", 4.0)
    upper_bound = kwargs.pop("upper_bound", 1024.0)
    multiplier = kwargs.pop("multiplier", None)
    min_requests = kwargs.pop("min_requests", 100)
    max_points = kwargs.pop("max_points", 12)
    max_total_seconds = kwargs.pop("max_total_seconds", 3600.0)
    # SLO thresholds: up to 9 optional "<=" latency targets (avg + p95 + p99 of
    # TTFT, ITL, end-to-end latency), all in ms. Any subset may be set.
    slo_avg_ttft_ms = kwargs.pop("slo_avg_ttft_ms", None)
    slo_avg_itl_ms = kwargs.pop("slo_avg_itl_ms", None)
    slo_p95_ttft_ms = kwargs.pop("slo_p95_ttft_ms", None)
    slo_p95_itl_ms = kwargs.pop("slo_p95_itl_ms", None)
    slo_p99_ttft_ms = kwargs.pop("slo_p99_ttft_ms", None)
    slo_p99_itl_ms = kwargs.pop("slo_p99_itl_ms", None)
    slo_avg_latency_ms = kwargs.pop("slo_avg_latency_ms", None)
    slo_p95_latency_ms = kwargs.pop("slo_p95_latency_ms", None)
    slo_p99_latency_ms = kwargs.pop("slo_p99_latency_ms", None)

    # --auto-tune and --stages each drive the load axis themselves, so whichever
    # loses the dispatch below would be SILENTLY IGNORED. All three modes are
    # therefore mutually exclusive, as the README states:
    #
    #   * --auto-tune + --profile / --stages — the ramp branch wins.
    #   * --stages + --profile — the stages branch overwrites profile with
    #     constant/concurrent per --axis, so `--stages ... --profile throughput`
    #     ran a constant-rate sweep and exited 0 while reporting nothing about it.
    #
    # set_if_not_default has already dropped --profile when left at its "sweep"
    # default, so a present "profile" key means the user set it explicitly.
    explicit_profile = kwargs.get("profile") is not None
    selected = [
        flag
        for flag, chosen in (
            ("--auto-tune", auto_tune),
            ("--stages", bool(stages)),
            ("--profile", explicit_profile),
        )
        if chosen
    ]
    if len(selected) > 1:
        raise click.UsageError(
            f"{selected[0]} is mutually exclusive with {', '.join(selected[1:])}."
        )

    if auto_tune:
        # Range/budget sanity, at the CLI boundary. Unchecked, these do not fail —
        # they burn a whole benchmark producing nothing usable, and the reason is
        # only visible by reading the ramp's arithmetic:
        #   * lower_bound 0 -> knob stays 0 forever (min(0*2, bound) == 0), so every
        #     one of max_points points measures the same invalid zero load.
        #   * lower_bound > upper_bound -> the first point is measured OUTSIDE the
        #     range the ramp documents as hard at both ends.
        #   * max_points < 1 / max_total_seconds <= 0 -> the Phase-1 predicate is
        #     false on entry, so the ramp returns zero points and no bracket.
        # Neither the gpustack API (both bounds are unconstrained Optional[float])
        # nor click's own types reject any of these, so this is the only gate.
        for flag, value in (
            ("--lower-bound", lower_bound),
            ("--upper-bound", upper_bound),
            ("--min-requests", min_requests),
            ("--max-points", max_points),
            ("--max-total-seconds", max_total_seconds),
        ):
            if value is not None and value <= 0:
                raise click.BadParameter(
                    f"must be greater than 0 (got {value}).",
                    ctx=click.get_current_context(),
                    param_hint=flag,
                )
        if multiplier is not None and multiplier <= 0:
            raise click.BadParameter(
                f"must be greater than 0 (got {multiplier}).",
                ctx=click.get_current_context(),
                param_hint="--multiplier",
            )
        if lower_bound > upper_bound:
            raise click.BadParameter(
                f"--lower-bound ({lower_bound}) must not exceed --upper-bound "
                f"({upper_bound}); the ramp measures nothing outside that range.",
                ctx=click.get_current_context(),
                param_hint="--lower-bound",
            )

    def _run_once(local_kwargs):
        try:
            args = build_scenario_args(local_kwargs)
        except ValidationError as err:
            # Translate pydantic validation error to click argument error.
            #
            # The loc path is nested (e.g. ("spec", "constraints", 2,
            # "max_error_rate", "rate", "<validator chain>")), so naming the
            # option after loc[0] alone reported every one of them as "--spec"
            # and threw away the only part that identifies the offending option.
            # "Invalid value for --spec: Input should be less than 1" gave the
            # reader nothing to act on.
            errs = err.errors(
                include_url=False, include_context=True, include_input=True
            )
            # Report the whole scenario path. Naming a single option would mean
            # guessing: the path holds several segments that are also real CLI
            # options (…max_error_rate.rate — both `--max-error-rate` and `--rate`
            # exist) and picking the wrong one sends the reader to the wrong
            # knob. The trailing segment is pydantic's validator chain, which
            # carries no information for a user.
            loc = [str(part) for part in errs[0]["loc"]]
            if loc and loc[-1].startswith(("function-after", "function-before")):
                loc = loc[:-1]
            raise click.BadParameter(
                errs[0]["msg"],
                ctx=click.get_current_context(),
                param_hint=".".join(loc) or "spec",
            ) from err

        # The report is returned, not discarded: a stage list is a manual load
        # ladder, so its per-stage metrics are the same curve the ramp charts and
        # the only place they exist is this return value. Single-run callers
        # ignore it.
        report, _ = asyncio.run(
            benchmark_generative_text(
                args=args,
                progress=progress,
                console=console,
            )
        )
        return report

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
        # The target (SLO boundary vs throughput saturation) is derived from
        # whether any --slo-* is set (see AutoTuneConfig.target).
        cfg = AutoTuneConfig(
            axis=axis,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            multiplier=multiplier,
            min_requests=min_requests,
            max_points=max_points,
            max_total_seconds=max_total_seconds,
            slo_avg_ttft_ms=slo_avg_ttft_ms,
            slo_avg_itl_ms=slo_avg_itl_ms,
            slo_p95_ttft_ms=slo_p95_ttft_ms,
            slo_p95_itl_ms=slo_p95_itl_ms,
            slo_p99_ttft_ms=slo_p99_ttft_ms,
            slo_p99_itl_ms=slo_p99_itl_ms,
            slo_avg_latency_ms=slo_avg_latency_ms,
            slo_p95_latency_ms=slo_p95_latency_ms,
            slo_p99_latency_ms=slo_p99_latency_ms,
            # `... or 42` would turn an explicit --random-seed 0 (a valid seed)
            # into 42; test None explicitly instead so 0 survives.
            random_seed_base=(
                42 if kwargs.get("random_seed") is None else int(kwargs["random_seed"])
            ),
            seed_increment=kwargs.get("seed_increment", True),
            probe_saturation=kwargs.get("probe_saturation", True),
        )
        base_outputs = list(kwargs.get("outputs") or ["benchmarks.dual_json"])
        # The ramp derives ONE dual_json pair per point from the first output's base
        # id ({base}__p{index}), so any further --outputs entry cannot be honored.
        # Say so: silently emitting only dual_json for a run that asked for
        # `--outputs json,html` looks like the html writer is broken.
        if len(base_outputs) > 1:
            _warn(
                f"--auto-tune writes one dual_json pair per measured point, "
                f"derived from '{base_outputs[0]}'; ignoring the remaining "
                f"--outputs entries: {', '.join(base_outputs[1:])}."
            )
        output_base = _output_base(base_outputs[0])
        # Per-point kwargs share everything except profile/rate/seed/outputs/
        # max_requests, which the ramp engine sets for each run.
        base_kwargs = dict(kwargs)
        for k in ("profile", "rate", "outputs", "max_requests", "max_seconds"):
            base_kwargs.pop(k, None)
        # Warm the ShareGPT conversion cache ONCE, sized for the LARGEST ramp
        # point. The adapter caches its converted jsonl by output-file existence,
        # so without this the first (smallest) point's max_requests would pin the
        # converted sample count and every later, larger point would be starved to
        # that first cap. Mirror scenario_builder's max_items rule: --data-samples
        # wins when set, else the biggest point's request count. Non-ShareGPT
        # sources pass through untouched.
        data_sources = [s for s in (kwargs.get("data") or []) if isinstance(s, str)]
        if data_sources:
            ds = kwargs.get("data_samples")
            if isinstance(ds, int) and ds > 0:
                warm_items = ds
            else:
                warm_items = max(
                    min_requests, round(upper_bound * cfg.resolved_multiplier)
                )
            prepare_datasets(
                data_sources,
                tokenizer=str(kwargs.get("processor") or ""),
                max_items=warm_items,
                input_min=input_min,
                input_max=input_max,
                output_tokens=kwargs.get("sharegpt_max_output_tokens"),
            )
        _log(
            f"Auto-tune ramp: axis={axis} target={cfg.target} "
            f"bounds=[{lower_bound},{upper_bound}] multiplier={cfg.resolved_multiplier} "
            f"min_requests={min_requests} max_points={max_points} "
            f"max_total_seconds={max_total_seconds} base={output_base}"
        )
        # The ramp keeps server_progress in the per-point runs so guidellm's live
        # on_benchmark_update callbacks drive a smooth (within-point) server bar; the
        # ramp sets run_index/run_total per point (see _prep_progress) so the bar
        # stays proportional and never hits 100 before it explicitly finalizes.
        outcome = asyncio.run(
            run_ramp(
                cfg=cfg,
                base_kwargs=base_kwargs,
                output_base=output_base,
                server_progress=server_progress,
                progress=progress,
                console=console,
            )
        )
        points = outcome.points
        # Why the search stopped, as a fact next to the point files. A sidecar
        # rather than a field inside the last point's report for three reasons:
        # "the last point" depends on file ordering, a consumer polling mid-run
        # reads a set that has no terminal point yet (the file's ABSENCE is then
        # the unambiguous "still running / no ramp"), and the report files are
        # re-serialized through a schema on the consumer side, which silently drops
        # fields it does not declare.
        _write_ramp_outcome(kwargs.get("output_dir"), output_base, outcome, _log, _warn)
        _log(
            f"Auto-tune ramp finished: {len(points)} point(s) measured, "
            f"bracket={outcome.bracket_reason} stop={outcome.stop_reason} "
            f"stopped_at={outcome.stopped_at}"
        )
        # The answer, on screen. Written to the console's underlying stream rather
        # than through Console.print: the report carries its own ANSI and its own
        # fixed-width grid, and rich would re-wrap and re-style both. `console is
        # None` is exactly --disable-console, so that flag silences this too.
        if console is not None:
            _emit_curve_report(
                outcome.to_dict(),
                chart_mode,
                stream=getattr(console, "file", sys.stdout),
            )
    elif stages:
        # v2.1 stages: one single-strategy run per stage, each carrying its own
        # max_requests / max_seconds. Each writes a separate output file
        # {base}__stage{i}.{ext}.
        #
        # --axis picks the load axis for every stage, exactly as it does for the
        # ramp: "rate" -> `constant` (open-loop, stage["rate"] req/s) and
        # "concurrency" -> `concurrent` (closed-loop, stage["rate"] streams).
        # This used to be hardcoded to `concurrent`, which silently turned a
        # fixed-rate stage list into a concurrency sweep — asking for 2 req/s ran
        # 2 concurrent streams and actually offered whatever the server could take.
        stage_profile = "concurrent" if axis == "concurrency" else "constant"
        base_outputs = list(kwargs.get("outputs") or [])
        stage_points: list = []
        stages_started = time.monotonic()
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict) or "rate" not in stage:
                raise click.BadParameter(
                    f"stage {i} is missing the required 'rate' key.",
                    ctx=click.get_current_context(),
                    param_hint="--stages",
                )
            if server_progress is not None:
                server_progress.run_index = i
                server_progress.run_total = len(stages)
            local = dict(kwargs)
            local["profile"] = stage_profile
            local["rate"] = [float(stage["rate"])]
            # Per-stage seed. DELIBERATELY not the ramp's rule, and the difference
            # is the point:
            #
            # * The ramp MUST vary its seed. Its points sit a doubling apart and are
            #   measured back to back, so reusing prompts lets the server answer
            #   later points out of its prefix/KV cache — which inflates the very
            #   throughput curve the peak is read off. There it is a correctness
            #   requirement, so the ramp defaults the base to 42 and always
            #   increments.
            # * Stages are load points the USER picked. The normal reason to write a
            #   stage list is to vary one thing — offered load — and read the
            #   response, which argues for holding the data fixed across stages.
            #
            # So a seed is only threaded through when the user asked for one:
            # `set_if_not_default` drops --random-seed while it sits at its 42
            # default, so "no random_seed key" means "not requested" and every stage
            # then shares guidellm's own default seed (identical prompts, comparable
            # stages). Passing --random-seed <anything>, 42 included, opts into
            # per-stage variation; --no-seed-increment pins it back. Only the Random
            # synthetic dataset is affected — file datasets are read in file order
            # regardless of seed.
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
            _log(
                f"Stage run {i}: rate={stage['rate']} "
                f"max_requests={local.get('max_requests')} "
                f"max_seconds={local.get('max_seconds')}"
            )
            report = _run_once(local)
            # Same normalization the ramp applies to its points, so the two modes
            # produce one comparable curve rather than two dialects of one. A stage
            # that produced no benchmark is skipped rather than charted as a hole:
            # the ladder is the user's, and a missing rung is visible as a gap in
            # the table's rate column.
            if report is not None and getattr(report, "benchmarks", None):
                stage_points.append(
                    _normalize(report.benchmarks[0], float(stage["rate"]), i)
                )
        _finish_stages(
            stage_points,
            axis=axis,
            output_dir=kwargs.get("output_dir"),
            output_base=_output_base((base_outputs or ["benchmarks.dual_json"])[0]),
            elapsed=time.monotonic() - stages_started,
            chart_mode=chart_mode,
            console=console,
            log=_log,
            warn=_warn,
        )
    else:
        # A single-strategy run measures ONE point, so there is no curve to draw
        # and no sidecar to write; guidellm prints its own result table for it.
        # Say so when --chart was asked for anyway, rather than accepting the flag
        # and doing nothing — a silently ignored option reads as a broken chart.
        if chart_mode != CHART_AUTO:
            _warn(
                f"--chart {chart_mode} applies to --auto-tune and --stages, which "
                "measure a curve. This run has a single strategy, so there is "
                "nothing to plot; the flag is ignored."
            )
        _run_once(dict(kwargs))


def _normalize_request_handlers(backend_kwargs: dict) -> None:
    """Normalize a custom request-handler override to the backend's expected shape.

    guidellm 0.7.x registers request handlers on ``OpenAIRequestHandlerFactory`` by
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
    "chart",
    short_help="Re-draw the report for a finished ramp or stage ladder.",
    help=(
        "Render the terminal report — load/throughput chart, latency chart, stage "
        "table and the best measured point — from a run's curve sidecar: "
        f"'{RAMP_OUTCOME_SUFFIX.lstrip('_')}' for --auto-tune, "
        f"'{CURVE_OUTCOME_SUFFIX.lstrip('_')}' for --stages. PATH is that file, or "
        "a directory containing one. Both modes print this themselves when they "
        "finish; this command is for reading it back afterwards, when the run "
        "scrolled away or the results came from CI."
    ),
)
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--chart",
    "chart_mode",
    type=click.Choice(list(CHART_MODES)),
    default=CHART_AUTO,
    show_default=True,
    help="Which charts to draw. See `benchmark run --help`.",
)
@click.option(
    "--width",
    # Bounded, not merely advisory: the renderer divides by (width - 1) to place
    # axis ticks, so a width of 1 is a ZeroDivisionError, and anything under ~20
    # has no room for a y-label gutter plus a readable curve.
    type=click.IntRange(20, 300),
    default=None,
    help=(
        "Plot width in columns (20-300). Defaults to fitting the terminal. Pin it "
        "to get byte-identical output across machines (docs, README, golden files)."
    ),
)
@click.option(
    "--ascii",
    "force_ascii",
    is_flag=True,
    help="Draw with ASCII only, ignoring what the terminal claims to support.",
)
def chart(path: Path, chart_mode: str, width: int | None, force_ascii: bool):
    sidecar = _resolve_ramp_sidecar(path)
    try:
        outcome = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise click.ClickException(f"Could not read {sidecar}: {e}") from e
    # A pre-v2 ramp sidecar predates inline points, so the curve is on disk but not
    # in THIS file. Say which file is short rather than printing an empty report —
    # and only blame the schema version for the file that actually has that history
    # (a stage curve has carried its points since v1).
    if not outcome.get("points"):
        detail = ""
        if sidecar.name.endswith(RAMP_OUTCOME_SUFFIX):
            detail = (
                f" (schema version {outcome.get('version', '?')}; inline points "
                f"need version {RAMP_OUTCOME_POINTS_VERSION} or later) — re-run the "
                "ramp to get a chartable sidecar"
            )
        raise click.ClickException(f"{sidecar} carries no measured points{detail}.")
    style = detect_style(sys.stdout, width=width)
    if force_ascii:
        style.unicode = False
    # strict=True: drawing IS this command's only job, so a render failure is a
    # failure. The live run swallows the same error on purpose (a benchmark that
    # measured fine must not die because a terminal could not be drawn on), but
    # here exiting 0 having printed nothing is a silent no-op in CI.
    _emit_curve_report(outcome, chart_mode, stream=sys.stdout, style=style, strict=True)


def _resolve_ramp_sidecar(path: Path) -> Path:
    """Accept either a sidecar itself or a directory holding exactly one.

    A benchmark directory is what a user has a path to (it is what --output-dir
    took), while the sidecar's name is derived from an output base id they never
    chose. Globbing for it is the difference between `chart ./benchmarks` and
    making them go find `123__ramp.json` first.

    Both curve-bearing modes are searched, because a user asking to see the chart
    for a directory does not think in terms of which writer produced it.
    """
    if path.is_file():
        return path
    found = sorted(
        p
        for suffix in (RAMP_OUTCOME_SUFFIX, CURVE_OUTCOME_SUFFIX)
        for p in path.glob(f"*{suffix}")
    )
    if not found:
        raise click.ClickException(
            f"No '*{RAMP_OUTCOME_SUFFIX}' or '*{CURVE_OUTCOME_SUFFIX}' file in "
            f"{path}. An --auto-tune ramp writes the first and --stages the "
            "second; a single-strategy --profile run measures no curve and writes "
            "neither."
        )
    if len(found) > 1:
        listing = ", ".join(p.name for p in found)
        raise click.ClickException(
            f"{len(found)} ramp sidecars in {path} ({listing}). Pass the one to draw."
        )
    return found[0]


@cli.command(
    short_help="Show configuration settings.",
    help="Display environment variables for configuring GuideLLM behavior.",
)
def config():
    print_config()


if __name__ == "__main__":
    cli()
