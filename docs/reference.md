# Benchmark reference

Search behavior, report formats and advanced configuration. [Quick start](../README.md).

Auto-tune (adaptive ramp)
-------------------------
`--auto-tune` replaces "guess a load value and read a curve" with probing a single
deterministic answer. It runs one single-strategy GuideLLM benchmark per measured
point and reads the metrics back to choose the next one:

- **Phase 1** — geometric bracket: a throughput search doubles the knob until
  total throughput stops climbing at two consecutive points, or request failures
  exceed the success floor. Growth recovery resets confirmation; set
  `--plateau-confirmations 1` for single-point confirmation. Confirmation
  respects the same bounds and budgets, so a truncated search remains a bound
  rather than a confirmed plateau.
  An SLO search keeps increasing load while its latency targets and success floor
  hold, even beyond a throughput plateau. Two consecutive increasing-load points
  at least 15% below the highest observed total throughput trigger a protective
  `throughput_decline` stop. A single dip or a plateau does not trigger it; recovery
  resets the count. All counted points still meet SLO, so this stop leaves the
  SLO boundary unlocated. Both searches respect the search range and budget.
- **Phase 2** — for an SLO target, bisect the pass/fail bracket for the maximum knob
  still within SLO; for a throughput target, run a unimodal (ternary/golden hybrid)
  search inside the bracket for the throughput argmax. Throughput refinement also
  converges when the running best total TPS improves by less than 1% across the
  latest three valid new refinement points, provided valid measured loads lie on
  both sides of the best point. Small new records do not restart this rolling
  window. Invalid measurements reset it. This gain criterion does not apply to
  SLO searches or geometric expansion.

The **target is derived**, not configured: set any `--slo-*` threshold and the
answer becomes "the maximum load meeting that SLO"; set none and it becomes "peak
throughput".

`--axis` picks the load axis, which decides what the knob means:

| `--axis` | GuideLLM profile | Knob | Loop |
|---|---|---|---|
| `rate` (default) | `constant` | requests/second offered | open |
| `concurrency` | `concurrent` | in-flight streams | closed |

Request-rate bounds and measured points support positive fractions, such as
`--lower-bound 0.1 --upper-bound 0.8`. Phase 2 uses a request-rate interval
tolerance of 1% of its lower endpoint, clamped between 0.000001 and 1
request/second. Concurrency uses whole streams and exhausts its search grid when
no interior integer remains. Throughput-gain convergence and point/time budgets
can end refinement before it reaches the interval tolerance; no measurement
accuracy is implied by the interval width.

```bash
# Peak throughput on the request-rate axis.
benchmark-runner benchmark run \
  --target http://localhost:8000 \
  --auto-tune --axis rate \
  --lower-bound 4 --upper-bound 8192 \
  --max-points 15 --max-total-seconds 3600 \
  --data "prompt_tokens=1024,output_tokens=128" \
  --processor PROCESSOR_PATH

# Maximum concurrency that keeps avg TTFT <= 500ms and p95 TPOT <= 50ms.
benchmark-runner benchmark run \
  --target http://localhost:8000 \
  --auto-tune --axis concurrency \
  --slo-avg-ttft-ms 500 --slo-p95-tpot-ms 50 \
  --data "prompt_tokens=128,output_tokens=128" \
  --processor PROCESSOR_PATH
```

**SLO thresholds** — 3 metrics x 3 aggregations = 9 optional `<=` targets, all in
milliseconds: `--slo-{avg,p95,p99}-{ttft,tpot,latency}-ms`. Any subset may be set; a
point passes when **every** set threshold holds (AND) and its success rate is
>= 95%. (GuideLLM reports end-to-end latency in seconds; it is converted
internally so every threshold is in ms.)

**Search range is hard** — `[--lower-bound, --upper-bound]` is the range you asked
for, and the ramp measures **nothing outside either end**. Hitting the upper bound
while throughput is still climbing stops the sweep (rather than quietly probing
higher); symmetrically, a server that is *already* saturated at the lower bound is
**reported** rather than searched below — the ramp stops, and gpustack raises a
`saturated_at_lower_bound` warning carrying the measured sustainable rate so you can
lower the range and re-run. Defaults are 4 / 8192. The ceiling is deliberately far
above many deployments' operating ranges, but still limits the maximum pressure
that can be offered. A throughput search may end on saturation; an SLO search
continues until a latency or success boundary, the range ceiling or a budget limit.
If every measured point passes at the upper bound, the highest passing load is
observed but a higher SLO boundary has not been located.

**Budget** — `--max-points` (default 18) and `--max-total-seconds` (default 5400) cap
the whole run. Both are ceilings, not quotas: a converged search stops early, so a
search can finish before spending either budget. They are sized for the expensive
case — an SLO bisection on a large deployment, which needs Phase 1's doublings plus
log2 of the bracket width. A saturation run can be given less: its answer is an argmax
that Phase 1 has already found, and the bisection-style narrowing buys it nothing —
which is why gpustack's Max Throughput preset ships a smaller budget than its latency
one. Per-point request counts are derived as
`max(--min-requests, knob * multiplier)` with the multiplier defaulting by axis
(10 concurrency / 30 rate), so every point below saturation gets a comparable ~30s
measurement window.

The duration cap binds from *inside* a point as well as between points: each run
(the saturation probe included) is given whatever is left of the budget as its own
duration limit, so a stalled point — an unresponsive server, or an offered rate the
server answers at a trickle — cannot run past the cap while working through its
request count. A point cut short this way still reports the metrics it gathered, and
the ramp ends with `stop_reason: budget_seconds`. The auto-tune mode takes neither
`--max-requests` nor `--max-seconds`: it derives both per point.

The search range and budget are validated up front — `--lower-bound` must be
positive and not exceed `--upper-bound`, and `--min-requests` / `--max-points` /
`--max-total-seconds` / `--multiplier` must be positive. A zero lower bound would
otherwise leave the knob pinned at 0 for every point.

**Saturation probe** — on the rate axis with a throughput target, one up-front
`throughput` run measures the server's ceiling rps and caps the ramp at
~`ceil(ceiling * 1.2)`: past the ceiling every point reports the same throughput
with worse latency, while costing geometrically more wall clock. That cap is *soft* —
reaching it while the server is still keeping up means the estimate read low, so it
doubles rather than truncating a real curve — and it never overrides `--upper-bound`.
The probe writes a `__satprobe` output file and is not counted as a measured point.
Disable with `--no-probe-saturation`.

Each measured point writes its own output pair, `{base}__p{index}.{ext}`, and uses a
distinct random seed (`--random-seed` as the base, `+1` per point) so cached prompts
and KV/prefix reuse cannot inflate later points. This needs no flag and happens
whether or not you pass `--random-seed` (the base defaults to 42): the ramp's points
are a doubling apart and measured back to back, so cache reuse would inflate the
throughput curve the peak is read off. `--no-seed-increment` pins one seed for every
point instead. (Manual `--stages` deliberately does the opposite by default — see
"Manual stages" below.)

The point files are dual-JSON pairs derived from the base id of the **first**
`--outputs` entry, so auto-tune cannot honor additional output kinds; passing more
than one logs which entries it ignored rather than quietly emitting only the first.

**Why the search stopped** — the ramp also writes `{base}__ramp.json`:

```json
{
  "version": 8,
  "bracket_reason": "capacity_plateau",
  "stop_reason": "converged",
  "stopped_at": 256.0,
  "points_measured": 7, "max_points": 18,
  "elapsed_seconds": 54.32, "max_total_seconds": 5400.0,
  "slo_bracket": [256.0, null],
  "probe_ceiling": null
}
```

`bracket_reason` is why the geometric bracket ended — which of the SLO, the server's
capacity, your search range, or the budget bounded the answer. `stop_reason` is why
the ramp as a whole ended, Phase 2 included; the two differ whenever a bracket found
its answer and the following search then converged normally. One of
`slo_failed` · `capacity_plateau` · `overloaded` · `upper_bound` · `budget_points` ·
`budget_seconds` · `converged` · `throughput_gain_converged` · `point_failed` ·
`throughput_decline`.

This is reported rather than left to be inferred from the measured curve because
several terminations leave an **identical** curve behind: a run ended by
`--max-total-seconds` can have the same point count as one that reached a search
bound. SLO searches report their pass/fail bracket separately; a throughput
plateau is a stop condition only for throughput searches. The file appears when the
ramp returns, so its absence means "still running, or not a ramp at all"; a write
failure is logged and does not fail the run.

`throughput_decline` is a protective end to SLO expansion, not an SLO failure or a
completed latency boundary. The highest measured passing load remains the primary
result and the failed end of `slo_bracket` stays null. The 15% / two-point policy is
internal and does not add CLI parameters. Ramp schema v6 optionally records the
actual trigger in `throughput_decline`:

```json
{
  "threshold_percent": 15,
  "consecutive_points": 2,
  "peak_rate": 64,
  "peak_total_tps": 10000,
  "points": [
    {"rate": 128, "total_tps": 8200, "drop_percent": 18},
    {"rate": 256, "total_tps": 8000, "drop_percent": 20}
  ]
}
```

`rate` uses the configured load axis (concurrency or req/s), and both drops are
relative to the recorded peak, not the preceding point. The field is omitted for
other stop reasons. Consumers of older sidecars without this evidence can show
the stop reason, but cannot assume the policy or trigger measurements.

`throughput_gain_converged` is a normal end to peak refinement. Ramp schema v7
records its actual rolling window in `throughput_gain_evidence`:

```json
{
  "threshold_percent": 1,
  "window_points": 3,
  "baseline_rate": 64,
  "baseline_total_tps": 3300,
  "best_rate": 48,
  "best_total_tps": 3320,
  "gain_percent": 0.606061,
  "lower_rate": 40,
  "upper_rate": 56,
  "points": [
    {"rate": 48, "total_tps": 3320, "best_total_tps": 3320},
    {"rate": 40, "total_tps": 3310, "best_total_tps": 3320},
    {"rate": 56, "total_tps": 3315, "best_total_tps": 3320}
  ]
}
```

The baseline is the measured total-TPS best before the window; the gain compares
the current best with that baseline, rather than requiring each sample to improve
by 1%. Measurements must have at least 95% success and positive finite total TPS.
Gain comparisons use the same six-decimal percentage as the evidence: a gain
that rounds to 1% requires more measurements. Measured TPS values retain their
original precision.
The baseline and best must also be valid; an invalid measurement holding the
global maximum prevents gain convergence. Equal total-TPS reference points use
the lowest load, then earliest measurement. The bounds are the nearest valid
measured loads on either side of the best, including expansion points. This
evidence is omitted for all other stop reasons. Exhausting the integer grid still
reports `converged`; an already satisfied gain criterion takes precedence over a
budget that would prevent another point. The reported peak remains the highest
measured average total TPS.

The ramp sidecar records `points[].total_tps` from
`metrics.tokens_per_second.successful.mean` independently of `output_tps`.
Peak selection, throughput search, decline protection and TPS charts use this
recorded total; input/output rates are auxiliary metrics, not substitutes or a
recomputed sum. Ramp schema v5 and manual curve schema v2 include `total_tps`.
Older output-only sidecars cannot reconstruct a total-TPS peak: offline charts
report that total throughput was not recorded instead of treating output as total.

Metrics and goodput
-------------------

`--metrics` accepts a JSON object configuring generative metrics, including `slo`,
`confidence`, `warnings` and `sample_size`. `--sample-requests` overrides only
`sample_size`. Reports include goodput, scheduler diagnostics and warnings.
Dispatch delay and schedule-relative latency describe open-loop workloads;
generation delay helps identify a client that cannot supply the intended load.

```bash
# Measure joint per-request SLO attainment and conforming requests/second.
benchmark-runner benchmark run \
  --target http://localhost:8000 --profile constant --rate 10 \
  --data "prompt_tokens=1024,output_tokens=128" --processor MODEL_PATH \
  --metrics '{"slo":{"ttft_ms":500,"tpot_ms":50},"confidence":0.95}' \
  --min-completed-requests 1000 --max-seconds 300

# Search with the runner's bounds/budgets and a joint attainment target.
benchmark-runner benchmark run \
  --target http://localhost:8000 --auto-tune --axis concurrency \
  --data "prompt_tokens=1024,output_tokens=128" --processor MODEL_PATH \
  --metrics '{"slo":{"ttft_ms":500,"tpot_ms":50}}' \
  --slo-attainment 0.95 --max-points 18 --max-total-seconds 1800
```

`--slo-attainment` requires `--auto-tune` and explicit `--metrics` containing
`slo`. It is an additional condition alongside any configured average/p95/p99 SLO
thresholds and the 95% success floor. Each request must satisfy **all** configured
per-request objectives; separate passing p95 metrics do not imply 95% joint
attainment. Errors count against attainment, while unmeasurable requests and
requests cancelled at the measurement boundary are excluded.
`tpot_ms` uses decode-only inter-token latency; a one-token response cannot
establish it. Missing attainment cannot pass the target.

Point records include `slo_attainment`, `slo_determined_requests`,
`request_goodput`, Wilson interval bounds and their confidence level. Decisions
use the observed attainment. Intervals crossing the target are recorded under
`analysis.unresolved_slo_knobs` and called out in the terminal report; increase
samples or duration to resolve them. `"confidence": null` disables intervals.

Steady-load sampling
--------------------

`--min-completed-requests` keeps supplying requests until the completed count
reaches the target. It is mutually exclusive with `--max-requests`, which stops
supplying at the created count and
then drains. It is a sample target, not an exact request quota, so additional
requests can be generated or cancelled at the boundary. Keep a `--max-seconds`
safety limit. When that completed-count target stops a point, its planned
in-flight cancellations are excluded from the runner's success-rate denominator;
server errors still count. Auto-tune uses completed counts automatically and
uses `--min-requests` as the per-point sample floor; those two flags have different
meanings. The short saturation probe uses a request-count backstop.

Adaptive profiles
------------------

Use `knee` to estimate where throughput growth flattens, or `goodput` to search
for the highest concurrency meeting a joint SLO attainment target:

```bash
benchmark-runner benchmark run \
  --target http://localhost:8000 --profile knee \
  --profile-args '{"initial_streams":[1,4,8,16,32,64],"adaptive":true}' \
  --data "prompt_tokens=1024,output_tokens=128" --processor MODEL_PATH \
  --max-seconds 60

benchmark-runner benchmark run \
  --target http://localhost:8000 --profile goodput \
  --profile-args '{"target_attainment":0.95,"max_streams":512,"max_probes":15}' \
  --metrics '{"slo":{"ttft_ms":500,"tpot_ms":50}}' \
  --data "prompt_tokens=1024,output_tokens=128" --processor MODEL_PATH \
  --max-seconds 60
```

`--profile-args` is a JSON object without `kind`. It cannot be combined with
`--auto-tune` or `--stages`; `knee` and `goodput` take their load settings there
instead of `--rate`. Duration caps apply **per point**. Knee needs at least five
distinct concurrency points, and adaptive knee can extend outside the initial
range. Goodput searches only concurrency. Their final conclusions are stored in
the report. Choose
`--auto-tune` when a hard search range and a total time/point budget are required.

Capacity analysis
------------------

Ramp reports store separate `analysis.throughput_peak` (measured total
tokens/s), `analysis.throughput_knee` (fitted output tokens/s versus concurrency)
and `analysis.slo_capacity` (highest measured passing load). A fitted knee can be
fractional and requires measurement before being used as a deployment setting.
`saturation_concurrency` is the first measured point at or above it, not its
integer ceiling. Linear or insufficient curves report `no_knee`; a rate-axis
curve reports `unavailable`. Reports also record why the search stopped, its
budget and the selected `plateau_confirmations` policy.

Terminal report
---------------
When a ramp or a stage ladder finishes it prints the answer, not just a directory
of JSON. The same report is available afterwards with `benchmark-runner chart`, so
a run that scrolled off screen — or one whose results came out of CI — can be read
back without re-measuring anything.

An `--auto-tune` ramp looks like this:

```

  Auto-tune ramp · axis=rate · target=saturation · 12 point(s) · 5m51s

  total throughput (tok/s)
       34k ┤                    ╭─●────●─────●────●─────●────◆─────●
           │          ╭●────●───╯                                  │
           │        ╭─╯                                            ╰╮
           │      ╭─╯                                               │
           │   ╭─●╯                                                 ╰╮
       17k ┤ ╭─╯                                                     ╰╮
           │●╯                                                        │
           │                                                          ╰╮
           │                                                           ╰✕
           │
         0 ┤

  TTFT p99 (log)
     78.0s ┤                                                           ╭✕
           │                                                          ╭╯
           │                                                          │
           │                                                         ╭╯
      1.4s ┤                                                        ╭╯
           │                                                       ╭╯
           │                                                       │
           │       ╭───●────●─────●────●─────●────●─────●────◆─────●
      25ms ┤●────●─╯
           └┬────┬─────┬────┬─────┬────┬─────┬────┬─────┬────┬─────┬────┬
            4    8    16   20    24   26    28   29    30   31    32   48
                                 offered rate (req/s)

  ◆ recommended  ● measured  ✕ overloaded

     offered  achieved   TTFT p99      TPOT      tok/s                        ok
  ────────────────────────────────────────────────────────────────────────────
           4       3.9       25ms     4.0ms     12,558  ███████             100%
           8       7.8       34ms     5.8ms     20,610  ███████████         100%
          16      15.5       51ms     9.4ms     29,083  ███████████████     100%
          20      19.4       60ms    11.2ms     31,206  █████████████████   100%
          24      23.3       69ms    13.0ms     32,567  █████████████████   100%
          26      25.2       73ms    13.9ms     33,052  ██████████████████  100%
          28      27.2       78ms    14.8ms     33,440  ██████████████████  100%
          29      28.1       80ms    15.2ms     33,604  ██████████████████  100%
          30      29.1       82ms    15.7ms     33,751  ██████████████████  100%
   ◆      31      29.6       84ms    16.2ms     33,882  ██████████████████  100%
          32      29.6       95ms    16.6ms     32,300  █████████████████   100%
   ✕      48      29.6      78.0s    23.8ms      5,225  ███                  93%

  ◆ Recommended: 31 req/s  →  33,882 tok/s
    TTFT p99 84ms · TPOT 16.2ms · achieved 29.6 req/s · 100% ok
    stopped: converged (bracket: overloaded) · probe ceiling 28.2 req/s
```

The two panels share one x axis: throughput over offered load on top, the tail
latency that load costs underneath. `◆` is the recommended operating point, `✕`
a point that shed requests or fell off the far side of the peak. The verdict line
carries the number you came for, and the stop reason says whether the search
converged or ran out of budget.

`--chart` selects how much is drawn:

| `--chart` | Shows |
|---|---|
| `auto` (default) | throughput + latency panels, stage table, recommended point |
| `all` | ... plus offered-vs-achieved rate, and the latency-throughput frontier |
| `none` | stage table and verdict only — no character grids |

`--disable-console` silences the report along with everything else, so a headless
run (gpustack drives the runner this way) is unaffected.

**The primary result follows the search objective.** `◆ recommended` identifies
the highest measured load meeting the SLO, or the measured total-TPS peak
when no SLO is set. An unmet SLO has no primary point. A different total-TPS peak is
reported separately. `✕ overloaded` means a failure rate over 5% *or* a point past
the peak delivering under 95% of it. A point one step past the knee at ~peak
throughput is deliberately NOT overload; it is
often the true argmax.

**It degrades rather than breaks.** Colour is added only on a TTY (and never
carries information a glyph does not), the box drawing falls back to ASCII on a
stream that cannot encode it, and a rendering failure warns on stderr instead of
failing a benchmark that already measured fine.

Re-drawing a saved ramp:

```bash
# The sidecar itself, or the directory holding it.
benchmark-runner chart ./benchmarks/42__ramp.json
benchmark-runner chart ./benchmarks --chart all

# Pin the geometry for docs and golden files.
benchmark-runner chart ./benchmarks --width 61 --ascii
```

This reads the run's curve sidecar — `{base}__ramp.json` for `--auto-tune`,
`{base}__curve.json` for `--stages`. The ramp's carries the measured grid inline
from schema version 2 on; a sidecar written by an older runner has the stop reasons
but not the curve, and `chart` says so rather than drawing an empty report.

### Stage ladders report less, on purpose

`--stages` gets the same charts, the same table and the same overload verdicts —
it is the same curve, just with the load points chosen by you instead of found by
a search. What it does **not** get is a recommendation:

```
  ▲ peak  ● measured  ✕ overloaded

     offered  achieved   TTFT p99      TPOT      tok/s                        ok
  ────────────────────────────────────────────────────────────────────────────
           4       3.9       25ms     4.0ms     12,558  ███████             100%
           8       7.8       34ms     5.8ms     20,610  ███████████         100%
          16      15.5       51ms     9.4ms     29,083  ███████████████     100%
          24      23.3       69ms    13.0ms     32,567  █████████████████   100%
   ▲      31      29.6       84ms    16.2ms     33,882  ██████████████████  100%
   ✕      40      29.6      10.3s    20.2ms     19,024  ██████████           93%

  ▲ Best of the stages you ran: 31 req/s  →  33,882 tok/s
    TTFT p99 84ms · TPOT 16.2ms · achieved 29.6 req/s · 100% ok
    the ladder brackets this peak — both neighbours of 31 req/s are lower
    6 stage(s), at the loads you gave — no search was run
    Use --auto-tune to have the peak located.
```

The ramp earns "run at 31 req/s" by bracketing the peak and converging on it. A
stage list has measured only the rungs it was handed, so its argmax is *the best
of what you ran*, and the line that matters is whether the ladder contained a peak
at all — an argmax at either end means the curve was still moving when the rungs
ran out, and the real peak is outside what was measured.

Single-strategy runs (`--profile constant --rate 10`, `--profile throughput`, ...)
measure one point, so there is no curve to draw and no sidecar is written; guidellm
prints its own result table for those.

Manual stages
-------------
`--stages` runs one single-strategy benchmark per stage, each carrying its own
constraints, writing `{base}__stage{i}.{ext}`. `--axis` selects what each stage's
`rate` means, exactly as it does for the ramp:

```bash
benchmark-runner benchmark run \
  --target http://localhost:8000 \
  --axis rate \
  --stages '[{"rate": 2, "max_requests": 60},
             {"rate": 4, "max_requests": 120},
             {"rate": 8, "max_seconds": 30}]' \
  --data "prompt_tokens=128,output_tokens=128" \
  --processor PROCESSOR_PATH
```

**Stage data is held fixed by default** — every stage uses the same seed, so the
same synthetic prompts are replayed at each load and the only thing that changes
between stages is the offered load. That is usually what a stage list is for.

To give **each stage its own prompts**, pass `--random-seed` explicitly — any value,
`42` included — and each stage gets `random_seed + stage_index`:

```bash
# stages get seeds 7, 8, 9
benchmark-runner benchmark run \
  --stages '[{"rate": 2}, {"rate": 4}, {"rate": 8}]' \
  --random-seed 7 \
  --data "prompt_tokens=128,output_tokens=128" ...
```

Passing the option is what opts in: left out, the seed stays at its default and no
per-stage variation happens. `--no-seed-increment` pins one seed for every stage even
when `--random-seed` is given. This only affects the Random synthetic dataset —
file datasets (including ShareGPT) are read in file order regardless of seed.

Vary the seed when you want each stage measured on cold prompts, since replaying
identical prompts lets the server answer later stages out of its prefix/KV cache and
flatters them. Keep it fixed when you want the stages to be directly comparable.
Note this differs from `--auto-tune`, which **always** varies its seed and needs no
flag to do so: its points are a doubling apart and measured back to back, so cache
reuse would inflate the throughput curve the peak is read off — there it is a
correctness requirement, not a preference.

`--auto-tune`, `--stages` and `--profile` are mutually exclusive.

Concurrency cap
---------------
`--max-concurrency` bounds in-flight requests. It is **required** by GuideLLM's
`throughput` profile (and `sweep`'s throughput pass, including the auto-tune
saturation probe), where it defaults to 512 to match GuideLLM's own sweep default.
It is optional for the open-loop rate profiles (`constant`/`poisson`/`async`), where
it bounds pile-up when the server cannot keep up with the offered rate, and ignored
by `concurrent`/`synchronous`.

Constraints
-----------
Per-benchmark stop conditions, all optional and passed straight through to
GuideLLM's constraint list: `--max-seconds`, `--max-requests`, `--max-errors`,
`--max-error-rate`, `--max-global-error-rate`.

`--max-error-rate` is a **fraction strictly between 0 and 1** (GuideLLM rejects both
endpoints). Omit it for no error-rate constraint — `1` would mean "tolerate every
failure" and `0` is not expressible as a rate (use `--max-errors` for a count).

`--detect-saturation` (alias `--default-over-saturation`) enables GuideLLM's native
`OverSaturationConstraint` in `enforce` mode: the run stops once throughput stops
keeping up with the offered load. `--over-saturation '{"enabled": true, ...}'` takes
the same detector with explicit settings — any `OverSaturationConstraint` field is
forwarded (`min_seconds`, `max_window_seconds`, `minimum_window_size`,
`moe_threshold`, `minimum_ttft`, `maximum_window_ratio`, `confidence`, and `mode`,
which can be set to `monitor` to report without stopping). Passing the option at all
enables the detector; `'{"enabled": false}'` is the way to ask for it off. It is a **generic runtime constraint that
applies to every mode**, not just auto-tune; on the ramp's short per-point runs it
simply won't fire (GuideLLM guards with `min_seconds=30` / `minimum_window_size=5`).

Progress reporting
------------------
You can send progress updates to a server endpoint during a benchmark:

```bash
benchmark-runner benchmark \
  --target http://localhost:8000 \
  --profile constant \
  --rate 10 \
  --max-seconds 20 \
  --data "prompt_tokens=128,output_tokens=256" \
  --processor PROCESSOR_PATH \
  --progress-url https://example.com/api/progress/123 \
  --progress-auth YOUR_TOKEN
```

HTTP Error Details for Failed Requests
--------------------------------------
GuideLLM's default `openai_http` backend does not always preserve response-body
error payloads in request-level benchmark errors. Benchmark Runner provides an
opt-in backend type that enriches failed request errors using OpenAI-style error
fields (`error.message`, `error.type`, `error.code`):

```bash
benchmark-runner benchmark run \
  --target http://localhost:8000/v1 \
  --backend openai_http_error_detail \
  --profile constant \
  --rate 10 \
  --max-requests 100 \
  --sample-requests 20 \
  --data "prompt_tokens=128,output_tokens=256" \
  --processor PROCESSOR_PATH
```

When a request fails, `requests.errored[*].info.error` in benchmark outputs will
contain text similar to:
`HTTP 400: ... (type=BadRequestError, code=400)`.

Note: if `--sample-requests 0` is used, request-level samples are omitted by design,
including failed request details.

ShareGPT dataset support
------------------------
If a dataset filename contains "sharegpt" and ends with `.json` or `.jsonl`,
Benchmark Runner will convert it to a GuideLLM-compatible JSONL file before running
the benchmark.

Use `--sharegpt-min-input-tokens`, `--sharegpt-max-input-tokens`, or both to keep
first-turn prompts whose token counts meet the supplied inclusive limits.
`--sharegpt-max-output-tokens` optionally sets the maximum output tokens requested
for every retained sample. If omitted, each sample uses its original answer's
token length. Token counts use the processor supplied to the run. A
range with no matching samples fails before the benchmark starts.

The conversion is **single-turn**: `sharegpt_to_guidellm.extract_first_turn` keeps
only the first human->gpt pair of each conversation. Multi-turn is therefore
synthetic-only — `--data "...,turns=N"` on GuideLLM's `synthetic_text` source. Passing
`turns` alongside a ShareGPT file has no effect.

Example:

```bash
benchmark-runner benchmark \
  --target http://localhost:8000 \
  --profile constant \
  --rate 10 \
  --max-seconds 20 \
  --processor PROCESSOR_PATH \
  --data ./ShareGPT_V3_unfiltered_cleaned_split.json
```

Outputs
-------
Benchmark Runner supports GuideLLM outputs plus a JSON summary output.
To save summary JSON:

```bash
benchmark-runner benchmark \
  --target http://localhost:8000 \
  --profile constant \
  --rate 10 \
  --max-seconds 20 \
  --data "prompt_tokens=128,output_tokens=256" \
  --processor PROCESSOR_PATH \
  --outputs dual_json \
  --output-dir ./benchmarks
```

Reasoning Tokens Support
-------------------------
Streaming timings include the thinking phase —
its chat handler treats a `reasoning_content` delta as a token arrival, and TTFOT
(`time_to_first_output_token_ms`) reports the first *content* token separately. You
do not need a custom handler for the timings.

What the custom handler adds is text accounting: it also counts reasoning as
generated output, so `text` and the word/character metrics derived from it
(`metrics.text.words` / `metrics.text.characters`) include the thinking. That
matters because a reasoning model given a small `output_tokens` never reaches
content at all — GuideLLM pins `max_completion_tokens=output_tokens` and
`ignore_eos=true`, so the whole budget goes to thinking and the response is cut off
mid-thought. Without the handler those runs report an empty output.

```bash
benchmark-runner benchmark run \
  --target http://localhost:8000/v1 \
  --backend openai_http_error_detail \
  --backend-kwargs '{"request_handlers": {"/v1/chat/completions": "chat_completions_with_reasoning"}}' \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --data your-dataset \
  --max-requests 100
```

Request handlers are registered by **API path**, so `request_handlers` is
keyed by path and its value is a registered handler *name* (a string the backend
resolves to a class). A legacy `response_handlers` dict keyed by request type
(`{"chat_completions": ...}`) is still accepted and translated, but the path-keyed
form above is the current one.
