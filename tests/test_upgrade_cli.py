import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from benchmark_runner import main
from benchmark_runner.main import cli
from tests.test_cli_dispatch import _fake_outcome


def test_joint_target_and_objectives_reach_auto_tune(monkeypatch, tmp_path):
    seen = {}

    async def run(cfg, **kwargs):
        seen["cfg"] = cfg
        seen.update(kwargs)
        return _fake_outcome(cfg)

    monkeypatch.setattr(main, "run_ramp", run)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--auto-tune",
            "--axis",
            "concurrency",
            "--slo-attainment",
            "0.99",
            "--metrics",
            '{"slo":{"ttft_ms":200}}',
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["cfg"].slo_attainment == 0.99
    assert seen["cfg"].target == "slo"
    assert seen["cfg"].plateau_confirmations == 2
    assert seen["base_kwargs"]["metrics"]["slo"] == {"ttft_ms": 200}


@pytest.mark.parametrize(
    "profile,args,metrics",
    [
        ("knee", {"initial_streams": [1, 2, 4, 8, 16]}, {}),
        ("goodput", {"max_streams": 16}, {"slo": {"ttft_ms": 200}}),
    ],
)
def test_native_profiles_and_metrics_reach_execution(
    monkeypatch, tmp_path, profile, args, metrics
):
    seen = []

    async def run(*, args, **kwargs):
        seen.append(args.spec)
        return SimpleNamespace(benchmarks=[]), []

    monkeypatch.setattr(main, "benchmark_generative_text", run)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--profile",
            profile,
            "--target",
            "http://localhost:8000",
            "--profile-args",
            json.dumps(args),
            "--metrics",
            json.dumps(metrics),
            "--data",
            "prompt_tokens=8,output_tokens=8",
            "--min-completed-requests",
            "100",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen[0].profile.kind == profile
    assert seen[0].constraints[0].kind == "min_requests"
    if metrics:
        assert seen[0].metrics.slo.ttft_ms == 200


@pytest.mark.parametrize("option", ["--metrics", "--profile-args"])
def test_configuration_options_require_objects(option):
    result = CliRunner().invoke(cli, ["benchmark", "run", option, "[]"])
    assert result.exit_code != 0
    assert "JSON object" in result.output


@pytest.mark.parametrize("count", ["0", "-1"])
def test_cli_rejects_nonpositive_confirmation_counts(count):
    result = CliRunner().invoke(
        cli, ["benchmark", "run", "--auto-tune", "--plateau-confirmations", count]
    )
    assert result.exit_code == 2
    assert "--plateau-confirmations" in result.output
    assert count in result.output


def test_cli_rejects_unsupported_metrics_kind_before_execution(monkeypatch):
    async def unexpected_run(**kwargs):
        pytest.fail("invalid metrics must fail before benchmark execution")

    monkeypatch.setattr(main, "benchmark_generative_text", unexpected_run)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--target",
            "http://localhost:8000",
            "--data",
            "prompt_tokens=8,output_tokens=8",
            "--metrics",
            '{"kind":"audio"}',
        ],
    )
    assert result.exit_code == 2
    assert "metrics.kind must be 'generative'" in result.output


@pytest.mark.parametrize(
    "args,message",
    [
        (
            ["--max-requests", "10", "--min-completed-requests", "10"],
            "mutually exclusive",
        ),
        (["--auto-tune", "--slo-attainment", "0.95"], "metrics.slo"),
        (["--auto-tune", "--profile-args", '{"adaptive":true}'], "--profile-args"),
        (["--profile", "knee", "--rate", "4"], "--rate"),
        (["--slo-attainment", "0.95"], "--auto-tune"),
    ],
)
def test_incompatible_options_fail_before_execution(args, message):
    result = CliRunner().invoke(cli, ["benchmark", "run", *args])
    assert result.exit_code != 0
    assert message in result.output
    assert "Traceback" not in result.output
