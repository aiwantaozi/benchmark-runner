"""GuideLLM 0.8.1 execution with runner-owned progress callbacks.

Adapted from GuideLLM v0.8.1's benchmark/entrypoints.py (Apache-2.0).
Its public entrypoint accepts only a boolean progress toggle. Use the same
component resolvers and Benchmarker lifecycle to retain our server tracker,
including on headless runs, without modifying upstream process-global state.
"""

from guidellm.benchmark import entrypoints as upstream
from guidellm.benchmark.progress import (
    CompositeBenchmarkerProgress,
    GenerativeConsoleBenchmarkerProgress,
    GenerativeLoggingBenchmarkerProgress,
)
from guidellm.schemas.benchmark import GenerativeMetricsArgs


async def benchmark_generative_text(args, progress=None, console=None, **constraints):
    """Run a validated scenario and preserve custom trackers and conclusions."""
    spec = upstream.resolve_to_single_benchmark(args.get_benchmarks())
    metrics = spec.metrics
    if not isinstance(metrics, GenerativeMetricsArgs):
        raise TypeError("The runner requires generative metrics")

    trackers = [GenerativeLoggingBenchmarkerProgress()]
    if progress is True:
        trackers.append(GenerativeConsoleBenchmarkerProgress())
    elif progress is not None and progress is not False:
        trackers.append(progress)

    backend, model = await upstream.resolve_backend(
        backend_args=spec.backend, console=console
    )
    await upstream.resolve_tokenizer(args=spec, model=model, console=console)
    loader = await upstream.create_data_loader(
        loader_config=spec.data_loader,
        data_config=spec.data,
        tokenizer_config=spec.tokenizer,
        column_mapper_config=spec.data_column_mapper,
        preprocessors_config=spec.data_preprocessors,
        finalizer_config=spec.data_finalizer,
        random_seed=spec.seed.value,
        console=console,
    )
    profile = await upstream.resolve_profile(
        profile=spec.profile,
        constraints=upstream.resolve_constraints(spec, **constraints),
        console=console,
        random_seed=spec.seed.value,
    )
    outputs = await upstream.resolve_output_formats(spec.outputs, console=console)
    report = upstream.GenerativeBenchmarksReport(config=args)
    benchmarker = upstream.Benchmarker()
    async for benchmark in benchmarker.run(
        accumulator_class=upstream.GenerativeBenchmarkAccumulator,
        benchmark_class=upstream.GenerativeBenchmark,
        requests=loader,
        backend=backend,
        profile=profile,
        environment=upstream.NonDistributedEnvironment(),
        progress=CompositeBenchmarkerProgress(trackers),
        sample_size=metrics.sample_size,
        warmup=spec.profile.warmup,
        cooldown=spec.profile.cooldown,
        prefer_response_metrics=metrics.prefer_response_metrics,
        slo=metrics.slo,
        confidence=metrics.confidence,
        warnings=metrics.warnings,
    ):
        if benchmark is not None:
            report.benchmarks.append(benchmark)
    if profile.conclusion is not None:
        report.conclusions.append(profile.conclusion)
    results = []
    for output_arg, output in zip(spec.outputs, outputs, strict=True):
        results.append((output_arg.kind, await output.finalize(report)))
    if console is not None:
        await upstream.GenerativeBenchmarkerConsole(console=console).finalize(report)
    return report, results
