# The functions in this file are adapted from:
# https://github.com/vllm-project/guidellm/blob/0d730d28d32b0f1e75232b2129ecf85c82c141eb/src/guidellm/__main__.py
# Modifications have been made to fit project requirements.

"""
Benchmark Runner command-line interface entry point.

This is the main CLI for Benchmark Runner, customized for this project.
Key customizations:
- Uses custom progress and output modules (see benchmark_runner.progress, benchmark_runner.chained_progress).
- Removes unnecessary subcommands, focusing on core benchmark and config functionality.

Provides:
- Benchmark execution for generative models.
- Configuration display.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from pydantic import ValidationError
from benchmark_runner.chained_progress import ChainedBenchmarkerProgress
from guidellm.benchmark.entrypoints import benchmark_generative_text
from benchmark_runner.progress import ServerBenchmarkerProgress
from benchmark_runner.sharegpt_adapter import prepare_datasets

try:
    import uvloop
except ImportError:
    uvloop = None  # type: ignore[assignment] # Optional dependency

from guidellm.backends import BackendType
from guidellm.benchmark import (
    BenchmarkGenerativeTextArgs,
    GenerativeConsoleBenchmarkerProgress,
    ProfileType,
    get_builtin_scenarios,
)
from guidellm.scheduler import StrategyType
from guidellm.schemas import GenerativeRequestType
from guidellm.settings import print_config
from guidellm.utils import Console, DefaultGroupHandler, get_literal_vals
from guidellm.utils import cli as cli_tools

STRATEGY_PROFILE_CHOICES: list[str] = list(get_literal_vals(ProfileType | StrategyType))
"""Available strategy and profile type choices for benchmark execution."""


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
    "--profile",
    "--rate-type",  # legacy alias
    "profile",
    default=BenchmarkGenerativeTextArgs.get_default("profile"),
    type=click.Choice(STRATEGY_PROFILE_CHOICES),
    help=f"Benchmark profile type. Options: {', '.join(STRATEGY_PROFILE_CHOICES)}.",
)
@click.option(
    "--rate",
    callback=cli_tools.parse_list_floats,
    multiple=True,
    default=BenchmarkGenerativeTextArgs.get_default("rate"),
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
    type=click.Choice(list(get_literal_vals(BackendType))),
    default=BenchmarkGenerativeTextArgs.get_default("backend"),
    help=f"Backend type. Options: {', '.join(get_literal_vals(BackendType))}.",
)
@click.option(
    "--backend-kwargs",
    "--backend-args",  # legacy alias
    "backend_kwargs",
    callback=cli_tools.parse_json,
    default=BenchmarkGenerativeTextArgs.get_default("backend_kwargs"),
    help="JSON string of arguments to pass to the backend.",
)
@click.option(
    "--model",
    default=BenchmarkGenerativeTextArgs.get_default("model"),
    type=str,
    help="Model ID to benchmark. If not provided, uses first available model.",
)
# Data configuration
@click.option(
    "--request-type",
    default=BenchmarkGenerativeTextArgs.get_default("data_request_formatter"),
    type=click.Choice(list(get_literal_vals(GenerativeRequestType))),
    help=(
        f"Request type to create for each data sample. "
        f"Options: {', '.join(get_literal_vals(GenerativeRequestType))}."
    ),
)
@click.option(
    "--request-formatter-kwargs",
    default=None,
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to the request formatter.",
)
@click.option(
    "--processor",
    default=BenchmarkGenerativeTextArgs.get_default("processor"),
    type=str,
    help=(
        "Processor or tokenizer for token count calculations. "
        "If not provided, loads from model."
    ),
)
@click.option(
    "--processor-args",
    default=BenchmarkGenerativeTextArgs.get_default("processor_args"),
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to the processor constructor.",
)
@click.option(
    "--data-args",
    multiple=True,
    default=BenchmarkGenerativeTextArgs.get_default("data_args"),
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to dataset creation.",
)
@click.option(
    "--data-samples",
    default=BenchmarkGenerativeTextArgs.get_default("data_samples"),
    type=int,
    help=(
        "Number of samples from dataset. -1 (default) uses all samples "
        "and dynamically generates more."
    ),
)
@click.option(
    "--data-column-mapper",
    default=BenchmarkGenerativeTextArgs.get_default("data_column_mapper"),
    callback=cli_tools.parse_json,
    help="JSON string of column mappings to apply to the dataset.",
)
@click.option(
    "--data-sampler",
    default=BenchmarkGenerativeTextArgs.get_default("data_sampler"),
    type=click.Choice(["shuffle"]),
    help="Data sampler type.",
)
@click.option(
    "--data-num-workers",
    default=BenchmarkGenerativeTextArgs.get_default("data_num_workers"),
    type=int,
    help="Number of worker processes for data loading.",
)
@click.option(
    "--dataloader-kwargs",
    default=BenchmarkGenerativeTextArgs.get_default("dataloader_kwargs"),
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to the dataloader constructor.",
)
@click.option(
    "--random-seed",
    default=BenchmarkGenerativeTextArgs.get_default("random_seed"),
    type=int,
    help="Random seed for reproducibility.",
)
# Output configuration
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=BenchmarkGenerativeTextArgs.get_default("output_dir"),
    help="The directory path to save file output types in",
)
@click.option(
    "--outputs",
    callback=cli_tools.parse_list,
    multiple=True,
    default=BenchmarkGenerativeTextArgs.get_default("outputs"),
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
    default=BenchmarkGenerativeTextArgs.get_default("warmup"),
    callback=cli_tools.parse_json,
    help=(
        "Warmup specification: int, float, or dict as string "
        "(json or key=value). "
        "Controls time or requests before measurement starts. "
        "Numeric in (0, 1): percent of duration or request count. "
        "Numeric >=1: duration in seconds or request count. "
        "Advanced config: see TransientPhaseConfig schema."
    ),
)
@click.option(
    "--cooldown",
    "--cooldown-percent",  # legacy alias
    "cooldown",
    default=BenchmarkGenerativeTextArgs.get_default("cooldown"),
    callback=cli_tools.parse_json,
    help=(
        "Cooldown specification: int, float, or dict as string "
        "(json or key=value). "
        "Controls time or requests after measurement ends. "
        "Numeric in (0, 1): percent of duration or request count. "
        "Numeric >=1: duration in seconds or request count. "
        "Advanced config: see TransientPhaseConfig schema."
    ),
)
@click.option(
    "--rampup",
    type=float,
    default=BenchmarkGenerativeTextArgs.get_default("rampup"),
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
    default=BenchmarkGenerativeTextArgs.get_default("max_seconds"),
    help=(
        "Maximum seconds per benchmark. "
        "If None, runs until max_requests or data exhaustion."
    ),
)
@click.option(
    "--max-requests",
    type=int,
    default=BenchmarkGenerativeTextArgs.get_default("max_requests"),
    help=(
        "Maximum requests per benchmark. "
        "If None, runs until max_seconds or data exhaustion."
    ),
)
@click.option(
    "--max-errors",
    type=int,
    default=BenchmarkGenerativeTextArgs.get_default("max_errors"),
    help="Maximum errors before stopping the benchmark.",
)
@click.option(
    "--max-error-rate",
    type=float,
    default=BenchmarkGenerativeTextArgs.get_default("max_error_rate"),
    help="Maximum error rate before stopping the benchmark.",
)
@click.option(
    "--max-global-error-rate",
    type=float,
    default=BenchmarkGenerativeTextArgs.get_default("max_global_error_rate"),
    help="Maximum global error rate across all benchmarks.",
)
@click.option(
    "--over-saturation",
    "over_saturation",
    callback=cli_tools.parse_json,
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
    callback=cli_tools.parse_json,
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

    # Handle remapping for request params
    request_type = kwargs.pop("request_type", None)
    request_formatter_kwargs = kwargs.pop("request_formatter_kwargs", None)
    if request_type is not None:
        kwargs["data_request_formatter"] = (
            request_type
            if not request_formatter_kwargs
            else {"request_type": request_type, **request_formatter_kwargs}
        )
    elif request_formatter_kwargs is not None:
        kwargs["data_request_formatter"] = request_formatter_kwargs

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
    envs = cli_tools.list_set_env()
    if console and envs:
        console.print_update(
            title=(
                "Note: the following environment variables "
                "are set and **may** affect configuration"
            ),
            details=", ".join(envs),
            status="warning",
        )

    progress_url = kwargs.pop("progress_url", None)
    progress_auth = kwargs.pop("progress_auth", None)
    progress_chain = [
        *(
            [
                ServerBenchmarkerProgress(
                    progress_url=progress_url, progress_auth=progress_auth
                )
            ]
            if progress_url
            else []
        ),
        *(
            [GenerativeConsoleBenchmarkerProgress()]
            if not disable_console_interactive
            else []
        ),
    ]
    progress = ChainedBenchmarkerProgress(progress_chain) if progress_chain else None

    try:
        args = BenchmarkGenerativeTextArgs.create(
            scenario=kwargs.pop("scenario", None), **kwargs
        )

        args.data, args.max_requests = prepare_datasets(
            data=args.data,
            tokenizer=args.processor,
            max_items=args.max_requests,
        )
        print(f"[DEBUG] Prepared data sources: {args.data}")
    except ValidationError as err:
        # Translate pydantic valdation error to click argument error
        errs = err.errors(include_url=False, include_context=True, include_input=True)
        param_name = "--" + str(errs[0]["loc"][0]).replace("_", "-")
        raise click.BadParameter(
            errs[0]["msg"], ctx=click.get_current_context(), param_hint=param_name
        ) from err

    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(
        benchmark_generative_text(
            args=args,
            progress=progress,
            console=console,
        )
    )


@cli.command(
    short_help="Show configuration settings.",
    help="Display environment variables for configuring GuideLLM behavior.",
)
def config():
    print_config()


if __name__ == "__main__":
    cli()
