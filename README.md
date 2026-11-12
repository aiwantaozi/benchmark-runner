# Benchmark Runner

Find peak throughput or the highest load meeting latency SLOs with **Auto-tune**.
Built on GuideLLM, with terminal reports, goodput metrics, manual stages,
ShareGPT conversion and server progress updates.

## Install

Python 3.10–3.12 is required.

```bash
pip install -e .
```

## Auto-tune

Set a target server, tokenizer and workload. Auto-tune chooses the load points
and reports the best measured result.

**Find peak total throughput:**

```bash
benchmark-runner benchmark run \
  --target http://localhost:8000 --processor MODEL_PATH \
  --data "prompt_tokens=1024,output_tokens=128" \
  --auto-tune --axis rate \
  --max-points 18 --max-total-seconds 1800 \
  --output-dir ./benchmarks
```

**Find the highest concurrency meeting latency targets:**

```bash
benchmark-runner benchmark run \
  --target http://localhost:8000 --processor MODEL_PATH \
  --data "prompt_tokens=1024,output_tokens=128" \
  --auto-tune --axis concurrency \
  --slo-avg-ttft-ms 500 --slo-p95-itl-ms 50 \
  --max-points 18 --max-total-seconds 1800 \
  --output-dir ./benchmarks
```

| Option | Purpose |
|---|---|
| `--axis rate` | Search offered requests/second; supports fractional rates. |
| `--axis concurrency` | Search whole-number concurrent streams. |
| `--lower-bound`, `--upper-bound` | Keep measurements inside the requested range. |
| `--max-points`, `--max-total-seconds` | Bound the search budget. |
| `--min-requests` | Set the per-point sample floor. |
| `--slo-{avg,p95,p99}-{ttft,itl,latency}-ms` | Require every configured latency target to pass. |
| `--plateau-confirmations` | Confirm low throughput growth; defaults to two consecutive points. |

With no SLO, Auto-tune searches for peak throughput. With SLOs, it searches for
the highest passing load, requiring at least 95% request success. It expands the
load geometrically, then refines the peak or bisects the SLO boundary. Each point
uses a distinct seed by default.

Check the reported stop reason: reaching a bound or exhausting a budget can
leave the true peak or SLO boundary unlocated. Auto-tune derives per-point limits;
use its total budget instead of `--max-requests` or `--max-seconds`.

[Search details and stop reasons](docs/reference.md#auto-tune-adaptive-ramp).

## Goodput and joint SLO attainment

Goodput counts requests that succeed **and meet every configured per-request
latency objective**. Adding SLO metrics preserves the original latency
statistics, including successful requests that exceed those objectives.

Add these options to an Auto-tune command to require 95% joint attainment:

```bash
--metrics '{"slo":{"ttft_ms":500,"tpot_ms":50,"e2el_ms":10000}}' \
--slo-attainment 0.95
```

`--metrics` alone enables measurement. `--slo-attainment` enables the Auto-tune
criterion and requires explicit `--metrics.slo`. Errors count as non-conforming;
unmeasurable and incomplete requests are excluded from attainment.

Reports include attainment, effective requests/second, sample counts and
confidence intervals. Intervals crossing the target are marked unresolved.
`--metrics` also accepts `confidence`, `warnings` and `sample_size`.

[Metric definitions and uncertainty](docs/reference.md#metrics-and-goodput).

## Other benchmark modes

Use these flags with the same target, processor and data options shown above:

| Mode | Example flags |
|---|---|
| Fixed request rate | `--profile constant --rate 10 --max-seconds 60` |
| Manual stages | `--axis rate --stages '[{"rate":2,"max_seconds":30},{"rate":4,"max_seconds":30}]'` |
| Throughput knee | `--profile knee --profile-args '{"initial_streams":[1,4,8,16,32,64],"adaptive":true}' --max-seconds 60` |
| Joint SLO search | `--profile goodput --profile-args '{"target_attainment":0.95,"max_streams":512}' --metrics '{"slo":{"ttft_ms":500}}' --max-seconds 60` |

`--auto-tune`, `--stages` and `--profile` are mutually exclusive.
`--profile-args` configures profiles only. The `knee` and `goodput` profiles use
concurrency and per-point duration limits; choose Auto-tune for a total search
budget and a hard load range.

[Profiles](docs/reference.md#adaptive-profiles) ·
[Manual stages](docs/reference.md#manual-stages).

## Sampling and limits

| Option | Stop behavior |
|---|---|
| `--max-requests N` | Stop creating requests at N, then drain. |
| `--min-completed-requests N` | Maintain load until N requests finish processing. |
| `--max-seconds N` | Limit the duration of a benchmark point. |
| `--max-concurrency N` | Cap in-flight requests for open-loop rate profiles. |

The two request-count options are mutually exclusive. Completed-count sampling
can cancel remaining in-flight requests at the boundary; those planned
cancellations are excluded from the runner's success-rate denominator. Auto-tune
uses completed-count sampling automatically, with `--min-requests` as its floor.

[Sampling](docs/reference.md#steady-load-sampling) ·
[Constraints and saturation detection](docs/reference.md#constraints).

## Reports and outputs

Auto-tune and manual stages print throughput/latency charts and a result table.
Use `--chart all` for extra panels or `--chart none` for the table and verdict.
Reopen saved results with:

```bash
benchmark-runner chart ./benchmarks --chart all
```

The default output is a summary/full JSON pair. Auto-tune writes
`{base}__p{index}.json`, `{base}__p{index}.full.json` and `{base}__ramp.json`.
The ramp report records:

- **Throughput peak:** the measured total tokens/s peak.
- **Throughput knee:** the fitted output-throughput knee on concurrency curves.
- **SLO capacity:** the highest measured passing load.
- Stop reasons, budgets and unresolved attainment measurements.

A fitted knee requires measurement before use as a deployment setting. Manual
stages report the best stage measured, without an automatic recommendation.
Single-point runs also support `json`, `yaml`, `csv` and `html` outputs.

[Terminal report](docs/reference.md#terminal-report) ·
[Capacity analysis](docs/reference.md#capacity-analysis) ·
[Output configuration](docs/reference.md#outputs).

## Datasets and integrations

- **Synthetic workloads:** `--data "prompt_tokens=1024,output_tokens=128"`.
- **ShareGPT:** automatic single-turn conversion for files containing `sharegpt`
  in the name, with input-token filters and an output-token cap.
  [Dataset options](docs/reference.md#sharegpt-dataset-support).
- **Server progress:** `--progress-url` and optional `--progress-auth`.
  [Configuration](docs/reference.md#progress-reporting).
- **HTTP error details:** `--backend openai_http_error_detail` preserves server
  error messages. [Example](docs/reference.md#http-error-details-for-failed-requests).
- **Reasoning output:** an optional handler includes thinking text in generated
  output. [Configuration](docs/reference.md#reasoning-tokens-support).

## Development

```bash
uv sync --group dev
uv run pytest
uv run pre-commit run --all-files
```

Build a runtime image with `docker build -t benchmark-runner .`.
Run `benchmark-runner benchmark run --help` for all CLI options.

[Automated code review setup](docs/code-review.md) · [License](LICENSE).
