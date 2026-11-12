"""The runner must retain custom progress when using GuideLLM 0.8.1."""

import asyncio
import json
from types import SimpleNamespace

from guidellm.benchmark import entrypoints
from guidellm.benchmark.progress import GenerativeLoggingBenchmarkerProgress

from benchmark_runner.scenario_builder import build_scenario_args


def test_custom_progress_metrics_and_conclusions_survive_execution(
    monkeypatch, tmp_path
):
    from benchmark_runner.execution import benchmark_generative_text

    class Progress(GenerativeLoggingBenchmarkerProgress):
        finalized = False

        async def on_finalize(self):
            self.finalized = True

    tracker = Progress()
    captured = {}

    async def backend(**kwargs):
        return object(), "test-model"

    async def tokenizer(**kwargs):
        return None

    async def loader(**kwargs):
        return []

    async def profile(**kwargs):
        return SimpleNamespace(conclusion={"kind": "test", "stop_reason": "done"})

    async def run(self, **kwargs):
        captured.update(kwargs)
        yield None  # A scheduler update without a completed benchmark.
        await kwargs["progress"].on_finalize()

    monkeypatch.setattr(entrypoints, "resolve_backend", backend)
    monkeypatch.setattr(entrypoints, "resolve_tokenizer", tokenizer)
    monkeypatch.setattr(entrypoints, "create_data_loader", loader)
    monkeypatch.setattr(entrypoints, "resolve_profile", profile)
    monkeypatch.setattr(entrypoints.Benchmarker, "run", run)
    scenario = build_scenario_args(
        {
            "backend_kwargs": {"target": "http://localhost:8000"},
            "profile": "concurrent",
            "rate": [1],
            "data": ["prompt_tokens=8,output_tokens=8"],
            "outputs": [str(tmp_path / "result.dual_json")],
        }
    )
    report, outputs = asyncio.run(
        benchmark_generative_text(args=scenario, progress=tracker)
    )

    assert tracker.finalized
    assert report.benchmarks == []
    assert captured["confidence"] == 0.95
    assert "warnings" in captured
    assert report.conclusions == [{"kind": "test", "stop_reason": "done"}]
    assert len(outputs) == 1
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["conclusions"] == report.conclusions
