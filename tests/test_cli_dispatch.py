"""CLI dispatch guards for the `benchmark run` command.

These cover the validation/wiring that happens in ``main.run`` *before* any
benchmark is executed, so they need no live server: the mutual-exclusion guard,
the per-stage ``rate`` requirement, that an explicit ``--random-seed 0`` survives
into the ramp config (a ``... or 42`` would silently drop it), and that the ramp
outcome lands on disk next to the point files.
"""

import json
from types import SimpleNamespace

import pytest

from click.testing import CliRunner

import benchmark_runner.main as main
from benchmark_runner import auto_tune
from benchmark_runner.auto_tune import RampOutcome
from benchmark_runner.main import cli
from benchmark_runner.scenario_builder import build_scenario_args


@pytest.mark.parametrize(
    "mode,expected_runs",
    [
        ([], 1),
        (["--stages", '[{"rate": 2}, {"rate": 4}]'], 2),
        (["--auto-tune", "--no-probe-saturation", "--max-points", "1"], 1),
    ],
    ids=["single", "stages", "auto-tune"],
)
def test_processor_args_reach_each_run(monkeypatch, tmp_path, mode, expected_runs):
    tokenizers = []

    async def capture_run(*, args, **kwargs):
        tokenizers.append(args.spec.tokenizer)
        return SimpleNamespace(benchmarks=[]), None

    monkeypatch.setattr(main, "benchmark_generative_text", capture_run)
    monkeypatch.setattr(auto_tune, "benchmark_generative_text", capture_run)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--target",
            "http://127.0.0.1:8000",
            "--processor",
            "custom-tokenizer",
            "--processor-args",
            '{"trust_remote_code": true}',
            "--data",
            "prompt_tokens=8,output_tokens=8",
            "--output-dir",
            str(tmp_path),
            *mode,
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(tokenizers) == expected_runs
    for tokenizer in tokenizers:
        assert tokenizer.model == "custom-tokenizer"
        assert tokenizer.load_kwargs == {"trust_remote_code": True}


def test_auto_tune_rejects_explicit_profile():
    result = CliRunner().invoke(
        cli, ["benchmark", "run", "--auto-tune", "--profile", "constant"]
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_sharegpt_rejects_inverted_input_range():
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--sharegpt-min-input-tokens",
            "2048",
            "--sharegpt-max-input-tokens",
            "256",
        ],
    )
    assert result.exit_code != 0
    assert "must not exceed --sharegpt-max-input-tokens" in result.output


def test_sharegpt_max_output_flag_is_available():
    params = {p.name: p for p in cli.commands["benchmark"].commands["run"].params}
    assert "--sharegpt-min-input-tokens" in params["sharegpt_input_min"].opts
    assert "--sharegpt-max-input-tokens" in params["sharegpt_input_max"].opts
    assert "--sharegpt-max-output-tokens" in params["sharegpt_max_output_tokens"].opts


def test_auto_tune_rejects_stages():
    result = CliRunner().invoke(
        cli, ["benchmark", "run", "--auto-tune", "--stages", '[{"rate": 2}]']
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_stages_rejects_explicit_profile():
    # The third pair, which used to slip through: the stages branch overwrites
    # profile with constant/concurrent per --axis, so `--stages ... --profile
    # throughput` exited 0 while running a constant-rate sweep and saying nothing
    # about the profile it discarded. README documents all three as exclusive.
    result = CliRunner().invoke(
        cli,
        ["benchmark", "run", "--stages", '[{"rate": 2}]', "--profile", "throughput"],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    assert "--profile" in result.output


def test_all_three_modes_at_once_is_rejected():
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--auto-tune",
            "--stages",
            '[{"rate": 2}]',
            "--profile",
            "throughput",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


class TestAutoTuneRangeIsValidated:
    """A bad range/budget must fail at the CLI, not silently waste a benchmark.

    None of these are rejected anywhere else: click's ``float``/``int`` types accept
    them and the gpustack API stores both bounds as unconstrained
    ``Optional[float]``, so a bad value travels all the way into the ramp — where it
    produces a full run's worth of useless points and a reason visible only by
    reading the ramp's arithmetic.
    """

    def _invoke(self, *args):
        return CliRunner().invoke(cli, ["benchmark", "run", "--auto-tune", *args])

    def test_zero_lower_bound_is_rejected(self):
        # knob = min(0 * 2, bound) == 0 forever: every one of max_points points
        # would measure the same invalid zero load.
        result = self._invoke("--lower-bound", "0")
        assert result.exit_code != 0
        assert "--lower-bound" in result.output
        assert "greater than 0" in result.output
        assert "Traceback" not in result.output

    def test_negative_lower_bound_is_rejected(self):
        result = self._invoke("--lower-bound", "-4")
        assert result.exit_code != 0
        assert "greater than 0" in result.output

    def test_inverted_range_is_rejected(self):
        # The ramp documents [lower, upper] as hard at BOTH ends, then measures its
        # first point at lower_bound -- outside the range when the range is
        # inverted.
        result = self._invoke("--lower-bound", "100", "--upper-bound", "50")
        assert result.exit_code != 0
        assert "must not exceed" in result.output
        assert "Traceback" not in result.output

    def test_zero_max_points_is_rejected(self):
        # Phase 1's predicate is false on entry -> zero points, no bracket, no
        # answer.
        result = self._invoke("--max-points", "0")
        assert result.exit_code != 0
        assert "greater than 0" in result.output

    def test_zero_budget_is_rejected(self):
        result = self._invoke("--max-total-seconds", "0")
        assert result.exit_code != 0
        assert "greater than 0" in result.output

    def test_zero_multiplier_is_rejected(self):
        result = self._invoke("--multiplier", "0")
        assert result.exit_code != 0
        assert "--multiplier" in result.output

    def test_equal_bounds_are_allowed(self, monkeypatch, tmp_path):
        # A single-point probe at one exact load is a legitimate request.
        captured = {}

        async def fake_run_ramp(cfg, **kwargs):
            captured["cfg"] = cfg
            return _fake_outcome(cfg)

        monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
        result = CliRunner().invoke(
            cli,
            [
                "benchmark",
                "run",
                "--auto-tune",
                "--lower-bound",
                "8",
                "--upper-bound",
                "8",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["cfg"].lower_bound == 8.0

    def test_the_range_is_not_checked_without_auto_tune(self):
        # These options are auto-tune's; a non-ramp run that happens to carry a
        # leftover value must not be blocked by a check that cannot apply to it.
        result = CliRunner().invoke(
            cli,
            [
                "benchmark",
                "run",
                "--stages",
                '[{"max_requests": 10}]',
                "--lower-bound",
                "0",
            ],
        )
        # Still fails, but on the stage's missing 'rate' -- not on the bound.
        assert "greater than 0" not in result.output


def test_stage_missing_rate_is_a_friendly_error():
    result = CliRunner().invoke(
        cli, ["benchmark", "run", "--stages", '[{"max_requests": 10}]']
    )
    assert result.exit_code != 0
    # BadParameter names the option and the missing key, not a raw KeyError.
    assert "rate" in result.output
    assert "Traceback" not in result.output


class TestStageSeedPolicy:
    """Stages hold the data FIXED by default; --random-seed opts into variation.

    This is a deliberate difference from the ramp, not an oversight, so it is
    pinned here — the asymmetry looks like a bug on a quick read and would
    otherwise get "fixed" into ramp behavior, silently changing what every
    existing stage list measures.

    Why they differ: the ramp's points sit a doubling apart and run back to back,
    so replaying prompts lets the server answer later points from its prefix/KV
    cache and inflates the curve the peak is read off — varying the seed there is
    a correctness requirement. A stage list is user-picked load points, usually
    written to vary ONE thing (offered load), which argues for identical prompts.

    Mechanism worth knowing: ``set_if_not_default`` drops --random-seed while it
    sits at its 42 default, so "absent" means "not requested". Passing the option
    is the opt-in, even at the same value 42.
    """

    def _stage_seeds(self, monkeypatch, *extra):
        """Seeds the three stages actually hand to guidellm."""
        seeds = []

        def spy(local_kwargs):
            seeds.append(local_kwargs.get("random_seed"))
            return object()  # stands in for the BenchmarkScenario

        async def noop(*a, **kw):
            return None, None

        monkeypatch.setattr(main, "build_scenario_args", spy)
        monkeypatch.setattr(main, "benchmark_generative_text", noop)
        result = CliRunner().invoke(
            cli,
            [
                "benchmark",
                "run",
                "--stages",
                '[{"rate": 2}, {"rate": 4}, {"rate": 8}]',
                "--data",
                "prompt_tokens=8,output_tokens=8",
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(seeds) == 3, f"expected 3 stages, saw {seeds}"
        return seeds

    def test_no_seed_by_default_so_stages_are_comparable(self, monkeypatch):
        # Every stage replays the same synthetic prompts; only the load differs.
        assert self._stage_seeds(monkeypatch) == [None, None, None]

    def test_passing_the_option_opts_into_per_stage_variation(self, monkeypatch):
        assert self._stage_seeds(monkeypatch, "--random-seed", "7") == [7, 8, 9]

    def test_the_default_value_passed_explicitly_still_opts_in(self, monkeypatch):
        # 42 IS the default, but typing it is a request -- this is the one that
        # reads as surprising, hence the test.
        assert self._stage_seeds(monkeypatch, "--random-seed", "42") == [42, 43, 44]

    def test_no_seed_increment_pins_one_seed(self, monkeypatch):
        seeds = self._stage_seeds(
            monkeypatch, "--random-seed", "7", "--no-seed-increment"
        )
        assert seeds == [7, 7, 7]


def test_manual_rate_stages_preserve_fractional_rates(monkeypatch, tmp_path):
    rates = []

    def spy(local_kwargs):
        scenario = build_scenario_args(local_kwargs)
        rates.extend(scenario.spec.profile.rate)
        return scenario

    async def noop(*args, **kwargs):
        return None, None

    monkeypatch.setattr(main, "build_scenario_args", spy)
    monkeypatch.setattr(main, "benchmark_generative_text", noop)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--axis",
            "rate",
            "--target",
            "http://127.0.0.1:8000",
            "--stages",
            '[{"rate": 0.1}, {"rate": 0.3}]',
            "--data",
            "prompt_tokens=8,output_tokens=8",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert rates == [0.1, 0.3]


def test_auto_tune_preserves_explicit_seed_zero(monkeypatch, tmp_path):
    # --output-dir is passed only to keep the ramp sidecar out of the repo root:
    # with no output dir the writer falls back to cwd (gpustack always passes one).
    captured = {}

    async def fake_run_ramp(cfg, **kwargs):
        captured["cfg"] = cfg
        return RampOutcome(
            points=[],
            bracket_reason="point_failed",
            stop_reason="point_failed",
            target=cfg.target,
            axis=cfg.axis,
            stopped_at=None,
            lower_bound=cfg.lower_bound,
            upper_bound=cfg.upper_bound,
            max_points=cfg.max_points,
            max_total_seconds=cfg.max_total_seconds,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--auto-tune",
            "--random-seed",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["cfg"].random_seed_base == 0


def _fake_outcome(cfg, **facts):
    """A RampOutcome with no points — enough to exercise the sidecar writer."""
    base = dict(
        points=[],
        bracket_reason="capacity_plateau",
        stop_reason="converged",
        target=cfg.target,
        axis=cfg.axis,
        stopped_at=256.0,
        lower_bound=cfg.lower_bound,
        upper_bound=cfg.upper_bound,
        max_points=cfg.max_points,
        max_total_seconds=cfg.max_total_seconds,
        elapsed_seconds=54.321,
        slo_bracket=(256.0, None),
    )
    base.update(facts)
    return RampOutcome(**base)


def test_ramp_outcome_is_written_beside_the_point_files(monkeypatch, tmp_path):
    # The consumer globs "{base}__p{index}.json" for points; the outcome goes to
    # "{base}__ramp.json", which that glob must not pick up.
    async def fake_run_ramp(cfg, **kwargs):
        return _fake_outcome(cfg)

    monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--auto-tune",
            "--output-dir",
            str(tmp_path),
            "--outputs",
            "42.dual_json",
        ],
    )
    assert result.exit_code == 0, result.output

    path = tmp_path / "42__ramp.json"
    assert path.exists(), sorted(p.name for p in tmp_path.iterdir())
    facts = json.loads(path.read_text())
    assert facts["bracket_reason"] == "capacity_plateau"
    assert facts["stop_reason"] == "converged"
    assert facts["stopped_at"] == 256.0
    assert facts["slo_bracket"] == [256.0, None]
    assert facts["version"] == auto_tune.RAMP_OUTCOME_VERSION
    # v2 carries the measured grid inline, which is what `benchmark-runner chart`
    # redraws from. Without it the sidecar says only why the search stopped, and
    # rebuilding the curve means globbing and re-parsing the point reports.
    assert len(facts["points"]) == facts["points_measured"]
    # Not matched by the point glob.
    assert "__p" not in path.name.replace("__ramp", "")


class TestConsoleOutputRespectsTheFlag:
    """``--disable-console`` says "Disable all outputs to the console".

    The runner's own diagnostics used to be bare ``print``s, which no flag could
    reach: a run asked to stay quiet still wrote ``[DEBUG] ...`` onto stdout,
    interleaved with whatever was consuming it. They now go through the console.

    Warnings are deliberately NOT suppressed — they go to stderr instead, because
    gpustack drives this headless and only has the process' streams, so "the ramp
    sidecar could not be written" must survive a display flag.
    """

    def _run_a_stage(self, monkeypatch, *extra):
        def spy(local_kwargs):
            return object()

        async def noop(*a, **kw):
            return None, None

        monkeypatch.setattr(main, "build_scenario_args", spy)
        monkeypatch.setattr(main, "benchmark_generative_text", noop)
        result = CliRunner().invoke(
            cli,
            [
                "benchmark",
                "run",
                "--stages",
                '[{"rate": 2}]',
                "--data",
                "prompt_tokens=8,output_tokens=8",
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        return result.output

    def test_the_stage_diagnostic_is_visible_by_default(self, monkeypatch):
        assert "Stage run 0" in self._run_a_stage(monkeypatch)

    def test_disable_console_silences_it(self, monkeypatch):
        assert self._run_a_stage(monkeypatch, "--disable-console").strip() == ""


def test_a_sidecar_that_cannot_be_written_does_not_fail_the_run(monkeypatch, tmp_path):
    # The points ARE the measurement; a diagnostic file must never take a run down.
    async def fake_run_ramp(cfg, **kwargs):
        return _fake_outcome(cfg)

    def boom(*a, **kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
    monkeypatch.setattr(main.Path, "write_text", boom)
    result = CliRunner().invoke(
        cli, ["benchmark", "run", "--auto-tune", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "Could not write ramp outcome" in result.output


def test_dropped_outputs_are_reported_in_auto_tune(monkeypatch, tmp_path):
    # The ramp derives one dual_json pair per point from the FIRST output's base id,
    # so further --outputs entries cannot be honored. Silently emitting only
    # dual_json for `--outputs 42.dual_json,report.html` reads as a broken html
    # writer; the run says what it ignored instead.
    async def fake_run_ramp(cfg, **kwargs):
        return _fake_outcome(cfg)

    monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--auto-tune",
            "--output-dir",
            str(tmp_path),
            "--outputs",
            "42.dual_json,report.html",
            # The warning must survive a silenced console: it is a real problem, and
            # it is routed to stderr rather than through the console for that reason.
            "--disable-console",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ignoring the remaining --outputs" in result.output
    assert "report.html" in result.output
    # The point files still come from the first entry.
    assert (tmp_path / "42__ramp.json").exists()


def test_a_single_output_does_not_warn(monkeypatch, tmp_path):
    async def fake_run_ramp(cfg, **kwargs):
        return _fake_outcome(cfg)

    monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--auto-tune",
            "--output-dir",
            str(tmp_path),
            "--outputs",
            "42.dual_json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ignoring the remaining --outputs" not in result.output


def test_both_itl_flag_spellings_reach_the_same_option():
    """The old `--slo-*-tpot-ms` spelling stays accepted.

    gpustack and this runner ship separately, so an upgrade lands on one side
    first. A flag rename that dropped the old name would fail those runs
    outright — the runner would reject an argument it no longer knows.
    """
    from click.testing import CliRunner

    from benchmark_runner.main import cli

    params = {p.name: p for p in cli.commands["benchmark"].commands["run"].params}
    for agg in ("avg", "p95", "p99"):
        option = params[f"slo_{agg}_itl_ms"]
        assert f"--slo-{agg}-itl-ms" in option.opts
        assert f"--slo-{agg}-tpot-ms" in option.opts

    assert CliRunner() is not None
