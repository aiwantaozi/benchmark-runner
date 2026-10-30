"""Scenario-spec construction from benchmark-runner's flat CLI kwargs."""

from unittest.mock import Mock

import pytest
from guidellm.data.tokenizers.huggingface import HuggingFaceTokenizer

from benchmark_runner.scenario_builder import (
    DEFAULT_MAX_CONCURRENCY,
    build_scenario_args,
)


def test_sharegpt_options_reach_dataset_preparation(monkeypatch):
    seen = {}

    def prepare(data, **kwargs):
        seen.update(kwargs)
        return ["converted_sharegpt.jsonl"]

    monkeypatch.setattr("benchmark_runner.scenario_builder.prepare_datasets", prepare)
    build_scenario_args(
        dict(
            backend_kwargs={"target": "http://127.0.0.1:8000"},
            data=("sharegpt.json",),
            outputs=("bench.dual_json",),
            profile="constant",
            rate=[2.0],
            sharegpt_input_min=256,
            sharegpt_input_max=2048,
            sharegpt_max_output_tokens=128,
        )
    )
    assert seen["input_min"] == 256
    assert seen["input_max"] == 2048
    assert seen["output_tokens"] == 128


def base_kwargs(**extra):
    """The minimum a real run supplies, plus whatever the case overrides."""
    return dict(
        backend_kwargs={"target": "http://127.0.0.1:8000"},
        data=("prompt_tokens=8,output_tokens=8",),
        outputs=("bench.dual_json",),
        max_requests=10,
        **extra,
    )


@pytest.mark.parametrize(
    "processor_args",
    [
        None,
        {},
        {"trust_remote_code": True, "revision": "custom", "use_fast": False},
        {"trust_remote_code": False},
    ],
)
def test_processor_args_reach_tokenizer_loading(monkeypatch, processor_args):
    scenario = build_scenario_args(
        base_kwargs(processor="custom-tokenizer", processor_args=processor_args)
    )
    load = Mock(return_value=object())
    monkeypatch.setattr(
        "guidellm.data.tokenizers.huggingface.AutoTokenizer.from_pretrained", load
    )

    HuggingFaceTokenizer(scenario.spec.tokenizer)()

    load.assert_called_once_with("custom-tokenizer", **(processor_args or {}))


# Every profile the runner can emit must produce a spec guidellm accepts.
# BenchmarkScenario.create validates eagerly, so a missing REQUIRED profile field
# raises here rather than after the container is up.
@pytest.mark.parametrize(
    "profile,extra",
    [
        ("synchronous", {}),
        ("throughput", {}),
        ("concurrent", {"rate": [4.0]}),
        ("constant", {"rate": [2.0]}),
        ("poisson", {"rate": [2.0]}),
        ("sweep", {}),
    ],
)
def test_every_profile_kind_builds(profile, extra):
    scenario = build_scenario_args(base_kwargs(profile=profile, **extra))
    assert scenario.spec.profile.kind == profile


def test_constant_profile_preserves_fractional_request_rate():
    scenario = build_scenario_args(base_kwargs(profile="constant", rate=[0.1]))
    assert scenario.spec.profile.rate == [0.1]


class TestThroughputMaxConcurrency:
    """Regression: the saturation probe used to emit a bare {"kind": "throughput"}.

    ThroughputProfileArgs.max_concurrency is `PositiveInt | None` with NO default,
    so pydantic treats it as required and the whole Max Throughput run died with
    "max_concurrency Field required" before issuing a single request.
    """

    def test_defaults_to_the_sweep_cap(self):
        scenario = build_scenario_args(base_kwargs(profile="throughput"))
        assert scenario.spec.profile.max_concurrency == DEFAULT_MAX_CONCURRENCY

    def test_explicit_cap_is_honored(self):
        scenario = build_scenario_args(
            base_kwargs(profile="throughput", max_concurrency=64)
        )
        assert scenario.spec.profile.max_concurrency == 64

    def test_sweep_throughput_pass_is_capped_too(self):
        scenario = build_scenario_args(base_kwargs(profile="sweep", max_concurrency=64))
        assert scenario.spec.profile.max_concurrency == 64


class TestOverSaturationConstraint:
    """`--detect-saturation` must reach the scenario, not just the command line.

    gpustack passes it for ANY mode (it is a general runtime cap, not auto-tune
    only), but the runner does not log resolved constraints — an E2E test can see
    the flag on the command line and still not know whether guidellm received it.
    That gap is what these assertions close.
    """

    def _kinds(self, scenario):
        return [c.kind for c in scenario.spec.constraints or []]

    def test_flag_adds_the_constraint(self):
        # The CLI's flag_value is '{"enabled": true}', parsed to a dict before it
        # reaches the builder.
        scenario = build_scenario_args(
            base_kwargs(
                profile="constant", rate=[2.0], over_saturation={"enabled": True}
            )
        )
        assert "over_saturation" in self._kinds(scenario)

    def test_enforce_mode_so_it_actually_stops_the_run(self):
        scenario = build_scenario_args(
            base_kwargs(
                profile="constant", rate=[2.0], over_saturation={"enabled": True}
            )
        )
        constraint = next(
            c for c in scenario.spec.constraints if c.kind == "over_saturation"
        )
        # "enforce" (not "observe"): the point of the flag is to stop once
        # throughput saturates.
        assert constraint.mode == "enforce"

    def test_absent_by_default(self):
        scenario = build_scenario_args(base_kwargs(profile="constant", rate=[2.0]))
        assert "over_saturation" not in self._kinds(scenario)

    def test_disabled_explicitly_is_actually_off(self):
        # Regression: the option value is a dict, and the builder tested it for
        # truthiness — so {"enabled": false}, a non-empty dict, ASKED FOR IT OFF AND
        # GOT IT ON.
        scenario = build_scenario_args(
            base_kwargs(
                profile="constant", rate=[2.0], over_saturation={"enabled": False}
            )
        )
        assert "over_saturation" not in self._kinds(scenario)

    def test_explicit_settings_reach_the_constraint(self):
        # `--over-saturation '{"enabled": true, "min_seconds": 90}'` is documented as
        # "the same detector with explicit settings"; the builder used to emit a
        # hardcoded {kind, mode} and drop every setting on the floor.
        scenario = build_scenario_args(
            base_kwargs(
                profile="constant",
                rate=[2.0],
                over_saturation={
                    "enabled": True,
                    "min_seconds": 90,
                    "minimum_window_size": 9,
                },
            )
        )
        constraint = next(
            c for c in scenario.spec.constraints if c.kind == "over_saturation"
        )
        assert constraint.min_seconds == 90
        assert constraint.minimum_window_size == 9
        assert constraint.mode == "enforce"  # still the default

    def test_monitor_mode_can_be_asked_for(self):
        scenario = build_scenario_args(
            base_kwargs(
                profile="constant",
                rate=[2.0],
                over_saturation={"mode": "monitor"},
            )
        )
        constraint = next(
            c for c in scenario.spec.constraints if c.kind == "over_saturation"
        )
        assert constraint.mode == "monitor"

    def test_settings_without_enabled_still_enable_it(self):
        # Passing the option at all is the request; only enabled:false turns it off.
        scenario = build_scenario_args(
            base_kwargs(
                profile="constant", rate=[2.0], over_saturation={"min_seconds": 45}
            )
        )
        assert "over_saturation" in self._kinds(scenario)

    def test_applies_to_manual_stages_too(self):
        # A stage run is a plain single-strategy run, so the same constraint path
        # is used — this is the Custom-mode case E2E could not observe.
        scenario = build_scenario_args(
            base_kwargs(
                profile="constant", rate=[4.0], over_saturation={"enabled": True}
            )
        )
        assert "over_saturation" in self._kinds(scenario)


class TestRateAxisMaxConcurrency:
    """Optional on the open-loop rate profiles: unset means unbounded."""

    def test_unset_leaves_it_unbounded(self):
        scenario = build_scenario_args(base_kwargs(profile="constant", rate=[2.0]))
        assert scenario.spec.profile.max_concurrency is None

    def test_explicit_cap_is_applied(self):
        scenario = build_scenario_args(
            base_kwargs(profile="constant", rate=[2.0], max_concurrency=128)
        )
        assert scenario.spec.profile.max_concurrency == 128
