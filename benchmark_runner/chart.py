"""Terminal rendering of a measured load curve.

Two modes produce one: ``--auto-tune`` searches for the load a server should run
at, and ``--stages`` measures a ladder the user wrote out. Both write the evidence
to JSON; this module puts it on screen, so a run ends with a chart and a sentence
instead of a directory of files.

The two are rendered by the same code and deliberately do NOT say the same thing.
A ramp brackets and converges, so it names an operating point. A ladder measured
only the rungs it was handed, so it reports its best rung and whether the ladder
contained a peak at all — reusing "recommended" there would launder the user's own
guess into a finding. See ``best_points`` and ``_verdict``.

Design constraints that shaped it, all of them terminal-specific:

* **Status verdicts are the UI's, not new ones.** ``point_status`` reimplements
  gpustack-ui's ``getStageStatus`` (recommended / peak / overloaded / ok) and the
  glyphs mirror its badges. A CLI that called 32 req/s "healthy" while the web
  report called the same point "overloaded" would be two products.
* **Colour is decoration, never information.** Every verdict is carried by a
  glyph; ANSI is added on top when the stream is a TTY. Piped into a log file or
  a CI job, the chart still says which point is which.
* **The x axis is categorical, not logarithmic.** The ramp doubles in Phase 1 and
  then refines inside the winning bracket, so log-x crushes exactly the points
  that carry the answer (4, 8, 16, 32 spread out; 28, 29, 30, 31 on top of each
  other). Even spacing per measured point is what the web report does too.
* **Latency is log-y.** A run whose top point blew up to 9s would otherwise flatten
  every sub-second point into one row, hiding the knee the chart exists to show.
* **Throughput is zero-based and allowed to look flat.** A plateau IS the finding;
  the marker says which point won, the curve does not have to exaggerate to prove it.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

# A point's success rate below this = overloaded. Same floor the ramp brackets on
# (auto_tune.SUCCESS_FLOOR) and the same 5% failure rate gpustack-ui gates its
# "overloaded" badge on. Duplicated rather than imported to keep this module a
# leaf: it is also driven by the `chart` subcommand, which has no ramp in memory.
SUCCESS_FLOOR = 0.95
# Past the peak, a point counts as declining once it delivers less than this share
# of the peak throughput. Being merely one step past the recommended knee at ~peak
# throughput is NOT overload — that point is often the true argmax.
DECLINE_RATIO = 0.95

# Verdicts, in the order they are checked. Mirrors gpustack-ui's StageStatusKind.
STATUS_RECOMMENDED = "recommended"
STATUS_PEAK = "peak"
STATUS_OVERLOADED = "overloaded"
STATUS_OK = "ok"

# Rendering modes for --chart.
CHART_AUTO = "auto"  # headline chart + table + verdict
CHART_ALL = "all"  # ... plus the two secondary charts
CHART_NONE = "none"
CHART_MODES = (CHART_AUTO, CHART_ALL, CHART_NONE)

# Glyph sets. The Unicode one deliberately avoids East-Asian WIDE code points
# (U+2605 BLACK STAR is one) — those render double-width in a CJK-configured
# terminal and shear every row of a fixed-width grid. What is left is the
# "ambiguous width" class, which is the same class the box-drawing characters
# below already belong to, so the chart either lines up or it doesn't; markers
# cannot break alignment on their own.
_GLYPHS_UNICODE = {
    STATUS_RECOMMENDED: "◆",
    STATUS_PEAK: "▲",
    STATUS_OVERLOADED: "✕",
    STATUS_OK: "●",
}
_GLYPHS_ASCII = {
    STATUS_RECOMMENDED: "*",
    STATUS_PEAK: "^",
    STATUS_OVERLOADED: "x",
    STATUS_OK: "o",
}
# (flat, up-corner, down-corner, up-elbow, down-elbow, vertical)
_LINES_UNICODE = ("─", "╭", "╰", "╯", "╮", "│")
_LINES_ASCII = ("-", ".", "'", "'", ".", "|")
_AXIS_UNICODE = ("└", "┤", "│", "┬")
_AXIS_ASCII = ("+", "+", "|", "+")

# Typography OUTSIDE the grid — separators, arrows, the table's bar column. Gated
# on the same flag as the grid because the fallback's promise is per-stream, not
# per-element: one stray "·" in a heading is enough to raise UnicodeEncodeError on
# the ascii-encoded stream the fallback exists for, and it would do it AFTER the
# chart had already been written.
_TEXT_UNICODE = {
    "sep": "·",
    "dash": "—",
    "arrow": "→",
    "dot": "┈",
    "thin": "┊",
    "rule": "─",
    "bar": "█",
}
_TEXT_ASCII = {
    "sep": "-",
    "dash": "--",
    "arrow": "->",
    "dot": ".",
    "thin": ":",
    "rule": "-",
    "bar": "#",
}

_ANSI = {
    STATUS_RECOMMENDED: "\033[1;34m",  # blue   — the answer
    STATUS_PEAK: "\033[32m",  # green  — throughput argmax
    STATUS_OVERLOADED: "\033[31m",  # red    — degraded
    STATUS_OK: "",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


@dataclass
class Style:
    """How to draw, decided once from the output stream.

    Kept as data rather than read from globals inside the renderer so tests can
    pin a deterministic style (ASCII, no colour, fixed width) without touching
    the environment.
    """

    width: int = 61  # plot area, excluding the y-label gutter
    color: bool = False
    unicode: bool = True

    @property
    def glyphs(self) -> dict[str, str]:
        return _GLYPHS_UNICODE if self.unicode else _GLYPHS_ASCII

    @property
    def lines(self) -> tuple[str, ...]:
        return _LINES_UNICODE if self.unicode else _LINES_ASCII

    @property
    def axis(self) -> tuple[str, ...]:
        return _AXIS_UNICODE if self.unicode else _AXIS_ASCII

    @property
    def text(self) -> dict[str, str]:
        return _TEXT_UNICODE if self.unicode else _TEXT_ASCII

    def paint(self, text: str, key: str) -> str:
        if not self.color:
            return text
        code = _ANSI.get(key, "")
        return f"{code}{text}{_ANSI['reset']}" if code else text


def detect_style(stream: Any = None, width: Optional[int] = None) -> Style:
    """Pick a Style for ``stream`` (default stdout).

    Unicode support is tested by ENCODING, not guessed from the locale name: the
    question is literally "can this stream carry these characters", and a stream
    that cannot would otherwise raise UnicodeEncodeError mid-chart. Colour follows
    the usual TTY + NO_COLOR contract.
    """
    stream = stream if stream is not None else sys.stdout
    try:
        tty = bool(stream.isatty())
    except Exception:  # noqa: BLE001 - a stream that can't answer is not a TTY
        tty = False
    color = tty and not os.environ.get("NO_COLOR")

    probe = "".join(_GLYPHS_UNICODE.values()) + "".join(_LINES_UNICODE)
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        probe.encode(encoding)
        unicode_ok = True
    except (UnicodeEncodeError, LookupError):
        unicode_ok = False

    if width is None:
        # Leave room for the y-label gutter and the two leading spaces; clamp so a
        # 300-column terminal doesn't stretch 10 points across the whole screen
        # (the chart is meant to be screenshot-able) and a narrow one still fits.
        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        width = max(40, min(90, cols - 12))
    return Style(width=width, color=color, unicode=unicode_ok)


# ── Point access ──────────────────────────────────────────────────────────────
# Points arrive as plain dicts from both callers: the live run passes
# dataclasses.asdict(PointMetrics) and the `chart` subcommand reads the same
# shape back out of the __ramp.json sidecar. One representation, one code path —
# so the offline render cannot drift from the one printed at the end of a run.


def _num(point: dict, key: str, default: float = 0.0) -> float:
    value = point.get(key)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def point_status(
    point: dict,
    *,
    recommended_knob: Optional[float],
    peak_knob: Optional[float],
    peak_tps: float,
) -> str:
    """One point's verdict — a port of gpustack-ui's ``getStageStatus``.

    Order matters: the recommended point wins its badge even when it is also the
    peak, because "this is the answer" outranks "this is the argmax".
    """
    knob = _num(point, "knob")
    if recommended_knob is not None and knob == recommended_knob:
        return STATUS_RECOMMENDED
    fail_rate = 1.0 - _num(point, "success")
    past_peak = peak_knob is not None and knob > peak_knob
    declining = (
        past_peak
        and peak_tps > 0
        and _num(point, "total_tps") < peak_tps * DECLINE_RATIO
    )
    if fail_rate > (1.0 - SUCCESS_FLOOR) or declining:
        return STATUS_OVERLOADED
    if peak_knob is not None and knob == peak_knob:
        return STATUS_PEAK
    return STATUS_OK


def best_points(
    points: Sequence[dict],
    slo_met_knob: Optional[float],
    target: str = "saturation",
) -> dict:
    """Measured total-TPS peak and the primary result's load.

    The SLO answer is the highest measured passing load, regardless of where
    throughput peaks. Without an SLO, the primary result is the total-TPS peak.

    ``target="stages"`` yields NO recommendation. A ladder is a list of rungs the
    user chose, so its argmax is "the best of what you measured", not "the load to
    run at" — those coincide only when the ladder happened to bracket the peak,
    which is precisely what a stage list does not verify. Reporting the peak as a
    recommendation would launder the user's own guess into a finding.
    """
    recorded = [point for point in points if point.get("total_tps") is not None]
    peak = max(
        recorded,
        key=lambda p: (_num(p, "total_tps"), -_num(p, "knob")),
        default=None,
    )
    peak_knob = _num(peak, "knob") if peak is not None else None
    peak_tps = _num(peak, "total_tps") if peak is not None else 0.0
    if target == "stages":
        recommended = None
    elif slo_met_knob is not None:
        recommended = float(slo_met_knob)
    elif target == "slo":
        recommended = None
    else:
        recommended = peak_knob
    return {
        "peak_knob": peak_knob,
        "peak_tps": peak_tps,
        "recommended_knob": recommended,
    }


# ── Plotting primitives ───────────────────────────────────────────────────────


def _columns(n: int, width: int) -> list[int]:
    """Even (categorical) column for each of ``n`` measured points."""
    if n <= 1:
        return [0]
    return [round(i * (width - 1) / (n - 1)) for i in range(n)]


def _resample(values: Sequence[float], cols: Sequence[int], width: int) -> list[float]:
    """Linear interpolation of sparse measured points onto every column.

    The line between two measured points is drawn, not implied, because a chart of
    8 isolated marks reads as noise; the interpolation is presentational and the
    markers stay visible on top of it so the measured points are never in doubt.
    """
    dense: list[float] = []
    for c in range(width):
        for i in range(len(cols) - 1):
            if cols[i] <= c <= cols[i + 1]:
                span = (cols[i + 1] - cols[i]) or 1
                t = (c - cols[i]) / span
                dense.append(values[i] + t * (values[i + 1] - values[i]))
                break
        else:
            dense.append(values[-1])
    return dense


def _draw_line(
    grid: list[list[str]],
    dense: Sequence[float],
    to_row: Callable[[float], int],
    style: Style,
    reference: bool = False,
) -> None:
    """asciichart-style stroke: corners where the slope turns, verticals to bridge."""
    if reference:
        dot, thin = style.text["dot"], style.text["thin"]
        flat = up = down = up_elbow = down_elbow = dot
        vert = thin
    else:
        flat, up, down, up_elbow, down_elbow, vert = style.lines
    height = len(grid)
    put = lambda r, c, ch: grid[height - 1 - r].__setitem__(c, ch)  # noqa: E731
    for c in range(len(dense) - 1):
        y0, y1 = to_row(dense[c]), to_row(dense[c + 1])
        if y0 == y1:
            put(y0, c, flat)
            continue
        put(y1, c, down if y0 > y1 else up)
        put(y0, c, down_elbow if y0 > y1 else up_elbow)
        for y in range(min(y0, y1) + 1, max(y0, y1)):
            put(y, c, vert)


@dataclass
class Series:
    """One line on a panel."""

    values: Sequence[float]
    marks: Sequence[str]  # per-point glyph key (a status, or "" for no marker)
    draw_line: bool = True
    # Draw with a dotted stroke instead of the box-drawing one. Two series in one
    # character grid are otherwise indistinguishable — same glyphs, same colour
    # policy — so the one that is context rather than measurement gets a visibly
    # lighter line. Used for "offered rate", which is the knob we asked for, not
    # a thing the server did.
    reference: bool = False
    # Column per point. Defaults to categorical; the frontier chart overrides it
    # so x can be a measured quantity (throughput) rather than a rank.
    cols: Optional[Sequence[int]] = None


def _measured_only(
    values: Sequence[float], marks: Sequence[str], cols: Sequence[int]
) -> tuple[list[float], list[str], list[int], int]:
    """Drop points whose value is non-positive, keeping the survivors' columns.

    On a LOG panel a non-positive value is *missing*, not *small*: ``_normalize``
    returns 0.0 for any metric absent from a report, and a point whose requests all
    failed has no latency percentiles at all. Feeding that 0 to log10 lands it nine
    decades below the real data, which does two wrong things at once — it compresses
    every measured point into the top rows, and it draws a dive to the floor and
    back that describes a latency the server never had.

    Excluding them keeps the scale honest. The line then interpolates across the
    gap, which is the same thing it already does between any two measured points,
    and the table still shows the point with its 0% success rate. The caller says
    how many were dropped so the omission is visible rather than silent.
    """
    keep = [i for i, v in enumerate(values) if v > 0]
    return (
        [values[i] for i in keep],
        [marks[i] if i < len(marks) else "" for i in keep],
        [cols[i] for i in keep],
        len(values) - len(keep),
    )


def _panel(  # noqa: C901 - one grid, drawn once; splitting it only moves state
    title: str,
    series: Sequence[Series],
    *,
    height: int,
    style: Style,
    fmt: Callable[[float], str],
    logy: bool = False,
    zero_base: bool = True,
    rules: Sequence[tuple[float, str]] = (),
) -> list[str]:
    """Render one chart panel: title, y-labelled grid, no x axis (the caller adds it).

    ``rules`` are horizontal reference lines (value, label) — used for SLO
    thresholds, which are the one thing on the latency chart that is a target
    rather than a measurement.
    """
    fwd = (lambda v: math.log10(max(v, 1e-9))) if logy else (lambda v: float(v))
    all_values = [fwd(v) for s in series for v in s.values]
    all_values += [fwd(v) for v, _ in rules]
    if not all_values:
        return []
    hi = max(all_values)
    lo = min(all_values) if (logy or not zero_base) else 0.0
    if hi <= lo:
        hi = lo + 1.0

    def to_row(v: float) -> int:
        return max(0, min(height - 1, round((v - lo) / (hi - lo) * (height - 1))))

    grid = [[" "] * style.width for _ in range(height)]
    put = lambda r, c, ch: grid[height - 1 - r].__setitem__(c, ch)  # noqa: E731

    # Reference lines first: they are background, and any data drawn later wins
    # the cell. A dotted stroke distinguishes "target" from "measured" without
    # relying on colour.
    dotted = style.text["dot"]
    for value, _ in rules:
        row = to_row(fwd(value))
        for c in range(style.width):
            put(row, c, dotted)

    marker_cells: list[tuple[int, int, str]] = []
    for s in series:
        cols = (
            list(s.cols) if s.cols is not None else _columns(len(s.values), style.width)
        )
        if s.draw_line and len(s.values) > 1:
            dense = _resample([fwd(v) for v in s.values], cols, style.width)
            _draw_line(grid, dense, to_row, style, reference=s.reference)
        for i, value in enumerate(s.values):
            key = s.marks[i] if i < len(s.marks) else ""
            if not key:
                continue
            r, c = to_row(fwd(value)), max(0, min(style.width - 1, cols[i]))
            put(r, c, style.glyphs.get(key, style.glyphs[STATUS_OK]))
            marker_cells.append((height - 1 - r, c, key))

    corner, tick, wall, _ = style.axis
    label_rows = {0: hi, height // 2: lo + (hi - lo) / 2, height - 1: lo}
    out = [f"  {style.paint(title, 'dim')}"]
    for i, row in enumerate(grid):
        text = "".join(row)
        if style.color:
            text = _paint_row(text, i, marker_cells, style)
        if i in label_rows:
            value = label_rows[i]
            out.append(f"  {fmt(10 ** value if logy else value):>8} {tick}{text}")
        else:
            out.append(f"  {'':>8} {wall}{text}")
    return out


def _paint_row(
    text: str, row: int, cells: Sequence[tuple[int, int, str]], style: Style
) -> str:
    """Colour just the marker cells of one row, leaving the stroke default."""
    todo = sorted((c, key) for r, c, key in cells if r == row)
    if not todo:
        return text
    out, cursor = [], 0
    for col, key in todo:
        if col < cursor:
            continue
        out.append(text[cursor:col])
        out.append(style.paint(text[col], key))
        cursor = col + 1
    out.append(text[cursor:])
    return "".join(out)


def _x_axis(
    cols: Sequence[int], labels: Sequence[str], style: Style, caption: str
) -> list[str]:
    corner, _, _, tick = style.axis
    flat = style.lines[0]
    marks = set(cols)
    rule = "".join(tick if c in marks else flat for c in range(style.width))
    row = [" "] * style.width
    for col, text in zip(cols, labels):
        start = min(max(col - len(text) // 2, 0), style.width - len(text))
        # Skip a label that would collide with one already placed. Dropping a tick
        # label is better than printing two knobs fused into one unreadable number.
        if all(ch == " " for ch in row[max(0, start - 1) : start + len(text) + 1]):
            row[start : start + len(text)] = list(text)
    return [
        f"  {'':>8} {corner}{rule}",
        f"  {'':>8}  {''.join(row).rstrip()}",
        f"  {'':>8}  {style.paint(caption.center(style.width).rstrip(), 'dim')}",
    ]


# ── Formatting ────────────────────────────────────────────────────────────────


def _fmt_count(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1000:
        return f"{v / 1000:.0f}k"
    return f"{v:.0f}"


def _fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.1f}s"
    if v >= 10:
        return f"{v:.0f}ms"
    return f"{v:.1f}ms"


def _fmt_ms_precise(v: float) -> str:
    """Table formatting for ITL, which lives in the single-digit-ms range.

    ``_fmt_ms`` drops the decimal above 10ms, which is right for an axis label and
    wrong for an ITL column: 16.2 vs 12.1 ms per token is the difference between
    two operating points, and rounding both to whole ms hides it.
    """
    if v >= 1000:
        return f"{v / 1000:.2f}s"
    return f"{v:.1f}ms"


def _fmt_knob(v: float) -> str:
    return f"{v:g}"


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _axis_names(axis: str) -> tuple[str, str, str]:
    """(x-axis caption, knob column header, knob unit) for the load axis in force.

    The header and the unit differ on the rate axis on purpose: the table puts the
    knob next to the ACHIEVED rate, where both are req/s and only "offered" vs
    "achieved" tells them apart, while the verdict sentence has no such neighbour
    and needs the unit spelled out.
    """
    if axis == "concurrency":
        return "concurrent streams", "streams", "streams"
    return "offered rate (req/s)", "offered", "req/s"


# ── Report ────────────────────────────────────────────────────────────────────


def render_curve_report(
    points: Sequence[dict],
    *,
    axis: str = "rate",
    target: str = "saturation",
    stop_reason: str = "",
    bracket_reason: str = "",
    slo_met_knob: Optional[float] = None,
    slo_first_fail_knob: Optional[float] = None,
    slo_thresholds: Optional[dict] = None,
    elapsed_seconds: Optional[float] = None,
    probe_ceiling: Optional[float] = None,
    mode: str = CHART_AUTO,
    style: Optional[Style] = None,
) -> list[str]:
    """The whole terminal report: charts, the stage table, and the verdict.

    Returns lines rather than printing, so the caller decides where they go (the
    live run routes them through guidellm's Console, which ``--disable-console``
    silences) and tests can assert on them.
    """
    if not points:
        return []
    style = style or detect_style()
    ordered = sorted(points, key=lambda p: _num(p, "knob"))
    best = best_points(ordered, slo_met_knob, target)
    statuses = [
        point_status(
            p,
            recommended_knob=best["recommended_knob"],
            peak_knob=best["peak_knob"],
            peak_tps=best["peak_tps"],
        )
        for p in ordered
    ]

    sep = style.text["sep"]
    # Name what actually ran. "Auto-tune ramp" over a hand-written stage list is
    # the same overclaim the verdict is careful to avoid one screen further down.
    if target == "stages":
        title = f"Stage ladder {sep} axis={axis} {sep} {len(ordered)} stage(s)"
    else:
        title = (
            f"Auto-tune ramp {sep} axis={axis} {sep} target={target} {sep} "
            f"{len(ordered)} point(s)"
        )
    out: list[str] = [""]
    out.append(
        "  " + style.paint(f"{title} {sep} {_fmt_duration(elapsed_seconds)}", "bold")
    )
    out.append("")

    total_available = all(point.get("total_tps") is not None for point in ordered)
    charts = mode != CHART_NONE and len(ordered) > 1 and total_available
    if not total_available:
        out.append(
            "  Total throughput was not recorded for every point; total-TPS charts unavailable."
        )
        out.append("")
    if charts:
        out += _headline_charts(ordered, statuses, axis, slo_thresholds, style)
        if mode == CHART_ALL:
            out += _secondary_charts(ordered, statuses, axis, style)
        out.append("")
        out.append("  " + _legend(statuses, style))
        out.append("")
    elif mode != CHART_NONE and len(ordered) == 1:
        out.append(
            "  "
            + style.paint(
                f"(only one point measured {style.text['dash']} no curve to draw)",
                "dim",
            )
        )
        out.append("")

    out += _table(ordered, statuses, axis, style)
    out.append("")
    out += _verdict(
        ordered,
        statuses,
        best,
        axis=axis,
        target=target,
        stop_reason=stop_reason,
        bracket_reason=bracket_reason,
        slo_met_knob=slo_met_knob,
        slo_first_fail_knob=slo_first_fail_knob,
        slo_thresholds=slo_thresholds,
        probe_ceiling=probe_ceiling,
        style=style,
    )
    out.append("")
    return out


def _itl_of(point: dict) -> float:
    """The point's inter-token latency, reading sidecars from before the rename.

    The field was called ``tpot_ms`` until this project adopted guidellm's
    vocabulary (gpustack still calls the same quantity TPOT — see
    ``PointMetrics``). Same number either way, so an older sidecar charts
    normally instead of showing a blank column.
    """
    return _num(point, "itl_ms") or _num(point, "tpot_ms")


def _headline_charts(
    points: Sequence[dict],
    statuses: Sequence[str],
    axis: str,
    slo_thresholds: Optional[dict],
    style: Style,
) -> list[str]:
    """Throughput over load, with the latency that buys it directly underneath.

    Two stacked panels sharing one x axis rather than a dual-y single panel: a
    second y scale in a character grid has nowhere to put its labels, and the
    reading the chart is FOR — "throughput stopped climbing here, and look what
    latency did at the same load" — is a vertical comparison anyway.
    """
    cols = _columns(len(points), style.width)
    tps = Series([_num(p, "total_tps") for p in points], statuses)
    out = _panel(
        "total throughput (tok/s)",
        [tps],
        height=11,
        style=style,
        fmt=_fmt_count,
        zero_base=True,
    )
    # No x axis under the throughput panel: the two panels SHARE one, drawn once
    # at the bottom. A blank separator keeps them legible as two charts without
    # spending a row on a duplicate axis.
    out.append("")

    # TTFT p99 is the tail the operator actually promises, and the one the web
    # report plots. Fall back to the mean when a sidecar predates the percentile.
    ttft = [_num(p, "ttft_p99_ms") or _num(p, "ttft_ms") for p in points]
    rules: list[tuple[float, str]] = []
    for key, label in (
        ("slo_p99_ttft_ms", "p99"),
        ("slo_p95_ttft_ms", "p95"),
        ("slo_avg_ttft_ms", "avg"),
    ):
        value = (slo_thresholds or {}).get(key)
        if value:
            rules.append((float(value), label))
            break
    dot = style.text["dot"]
    title = "TTFT p99 (log)" + (f"  {dot} SLO {_fmt_ms(rules[0][0])}" if rules else "")
    values, marks, at, dropped = _measured_only(ttft, statuses, cols)
    if dropped:
        title += (
            f"  {style.text['dash']} {dropped} point(s) reported no latency"
            " (see the table)"
        )
    if values:
        out += _panel(
            title,
            [Series(values, marks, cols=at)],
            height=9,
            style=style,
            fmt=_fmt_ms,
            logy=True,
            rules=rules,
        )
    else:
        out.append(f"  {style.paint(title, 'dim')}")
        out.append(f"  {'':>8} {style.paint('(no latency was measured)', 'dim')}")
    caption, _, _ = _axis_names(axis)
    out += _x_axis(cols, [_fmt_knob(_num(p, "knob")) for p in points], style, caption)
    return out


def _secondary_charts(
    points: Sequence[dict], statuses: Sequence[str], axis: str, style: Style
) -> list[str]:
    """The two extra views --chart all adds.

    Both answer a question the headline chart cannot:

    * offered vs achieved — WHERE the server stopped keeping up. The two lines
      track each other until the server saturates and then diverge; that fork is
      the real ceiling, independent of any latency threshold. Rate axis only: on
      the concurrency axis the knob is in-flight streams, so there is no offered
      rate to compare against.
    * latency-throughput frontier — what a millisecond of tail latency BUYS. Drawn
      as a scatter, not a line: past the peak the trajectory doubles back on itself
      (less throughput at higher latency) and a column grid cannot hold two y
      values for one x, so a connected stroke would draw a shape that never happened.
    """
    out: list[str] = []
    if axis == "rate":
        out.append("")
        offered = [_num(p, "knob") for p in points]
        achieved = [_num(p, "achieved_rate") for p in points]
        dot, line = style.text["dot"], style.text["rule"]
        out += _panel(
            f"achieved rate (req/s)  {line}{line} achieved {style.text['sep']} "
            f"{dot}{dot} offered  {style.text['dash']} the fork is the real ceiling",
            [
                Series(offered, [""] * len(points), reference=True),
                Series(achieved, statuses),
            ],
            height=9,
            style=style,
            fmt=lambda v: f"{v:.0f}",
            zero_base=True,
        )
        caption, _, _ = _axis_names(axis)
        out += _x_axis(
            _columns(len(points), style.width),
            [_fmt_knob(_num(p, "knob")) for p in points],
            style,
            caption,
        )

    out.append("")
    tps = [_num(p, "total_tps") for p in points]
    ttft = [_num(p, "ttft_p99_ms") or _num(p, "ttft_ms") for p in points]
    lo, hi = min(tps), max(tps)
    span = (hi - lo) or 1.0
    scatter_cols = [round((v - lo) / span * (style.width - 1)) for v in tps]
    # Same log-axis rule as the headline latency panel: a point with no measured
    # latency has no place on the frontier, since the frontier IS a latency claim.
    values, marks, at, _ = _measured_only(ttft, statuses, scatter_cols)
    if not values:
        return out
    out += _panel(
        f"latency-throughput frontier {style.text['dash']} "
        "what a ms of tail latency buys",
        [Series(values, marks, draw_line=False, cols=at)],
        height=9,
        style=style,
        fmt=_fmt_ms,
        logy=True,
    )
    ticks = sorted({0, (style.width - 1) // 2, style.width - 1})
    out += _x_axis(
        ticks,
        [_fmt_count(lo + span * c / max(style.width - 1, 1)) for c in ticks],
        style,
        "total throughput (tok/s)",
    )
    return out


def _legend(statuses: Sequence[str], style: Style) -> str:
    names = {
        STATUS_RECOMMENDED: "recommended",
        STATUS_PEAK: "peak",
        STATUS_OVERLOADED: "overloaded",
        STATUS_OK: "measured",
    }
    present = [
        k
        for k in (STATUS_RECOMMENDED, STATUS_PEAK, STATUS_OK, STATUS_OVERLOADED)
        if k in statuses
    ]
    return "  ".join(f"{style.paint(style.glyphs[k], k)} {names[k]}" for k in present)


def _table(
    points: Sequence[dict], statuses: Sequence[str], axis: str, style: Style
) -> list[str]:
    """One row per measured point.

    The bar column is not decoration: it is the only place the throughput plateau
    and the point that broke it are visible side by side with the numbers that
    caused it, which is what someone pastes into a ticket.
    """
    _, knob_header, _ = _axis_names(axis)
    peak_tps = max((_num(p, "total_tps") for p in points), default=0.0) or 1.0
    header = (
        f"   {knob_header:>7} {'achieved':>9} {'TTFT p99':>10} {'ITL':>9} "
        f"{'tok/s':>10}  {'':<18} {'ok':>5}"
    )
    ruler = style.text["rule"] * (len(header) - 2)
    out = [f"  {style.paint(header, 'dim')}", "  " + style.paint(ruler, "dim")]
    for point, status in zip(points, statuses):
        glyph = style.glyphs[status] if status != STATUS_OK else " "
        tps = _num(point, "total_tps")
        recorded = point.get("total_tps") is not None
        bar = style.text["bar"] * round(tps / peak_tps * 18) if recorded else ""
        tps_label = f"{tps:,.0f}" if recorded else "-"
        row = (
            f" {style.paint(glyph, status)} {_fmt_knob(_num(point, 'knob')):>7} "
            f"{_num(point, 'achieved_rate'):>9.1f} "
            f"{_fmt_ms(_num(point, 'ttft_p99_ms') or _num(point, 'ttft_ms')):>10} "
            f"{_fmt_ms_precise(_itl_of(point)):>9} "
            f"{tps_label:>10}  {bar:<18} {_num(point, 'success') * 100:>4.0f}%"
        )
        out.append(f"  {row}")
    return out


def _verdict(  # noqa: C901 - one sentence per mode; a dispatch table hides the wording
    points: Sequence[dict],
    statuses: Sequence[str],
    best: dict,
    *,
    axis: str,
    target: str,
    stop_reason: str,
    bracket_reason: str,
    slo_met_knob: Optional[float],
    slo_first_fail_knob: Optional[float],
    slo_thresholds: Optional[dict],
    probe_ceiling: Optional[float],
    style: Style,
) -> list[str]:
    """The sentence the whole run exists to produce, plus why the search stopped.

    Two claims, and the mode decides which one is honest:

    * a ramp SEARCHED, so it names an operating point ("Recommended: 31 req/s");
    * a stage ladder measured the rungs it was handed, so it can only report which
      of those was best and whether the ladder contained a peak at all.
    """
    _, _, unit = _axis_names(axis)
    sep, dash = style.text["sep"], style.text["dash"]
    ladder = target == "stages"
    lead_status = STATUS_PEAK if ladder else STATUS_RECOMMENDED
    chosen = None
    for point, status in zip(points, statuses):
        if status == lead_status:
            chosen = point
            break
    if chosen is None:
        if target == "slo":
            out = [f"  SLO met by no measured point {dash} no primary result"]
            if stop_reason:
                out.append(f"    stopped: {stop_reason}")
            return out
        if best["peak_knob"] is None:
            return ["  Total throughput was not recorded; total-TPS peak unavailable."]
        return []

    glyph = style.paint(style.glyphs[lead_status], lead_status)
    knob = _fmt_knob(_num(chosen, "knob"))
    label = "Best of the stages you ran" if ladder else "Recommended"
    throughput = (
        f"{_num(chosen, 'total_tps'):,.0f}"
        if chosen.get("total_tps") is not None
        else "-"
    )
    headline = (
        f"{glyph} {label}: {style.paint(knob + ' ' + unit, 'bold')}"
        f"  {style.text['arrow']}  {throughput} tok/s"
    )
    out = [f"  {headline}"]
    out.append(
        f"    TTFT p99 {_fmt_ms(_num(chosen, 'ttft_p99_ms') or _num(chosen, 'ttft_ms'))}"
        f" {sep} ITL {_fmt_ms_precise(_itl_of(chosen))}"
        f" {sep} achieved {_num(chosen, 'achieved_rate'):.1f} req/s"
        f" {sep} {_num(chosen, 'success') * 100:.0f}% ok"
    )

    if target == "slo":
        if slo_met_knob is None:
            note = (
                f"SLO met by no measured point {dash} every point breached a threshold"
            )
        elif stop_reason == "throughput_decline" and slo_first_fail_knob is None:
            note = (
                f"Protective stop: sustained total-throughput decline {dash} "
                "SLO boundary not located; this is the highest measured passing load"
            )
        elif slo_first_fail_knob is None:
            note = (
                f"SLO still met at the top of the sweep {dash} {knob} {unit} is a "
                "FLOOR, "
                "not the boundary"
            )
        else:
            note = (
                f"SLO boundary bracketed at ({_fmt_knob(float(slo_met_knob))}, "
                f"{_fmt_knob(float(slo_first_fail_knob))}) {unit}"
            )
        out.append(f"    {style.paint(note, 'dim')}")
        if best["peak_knob"] is not None and best["peak_knob"] != _num(chosen, "knob"):
            out.append(
                "    "
                + style.paint(
                    f"Measured total-TPS peak: {_fmt_knob(best['peak_knob'])} {unit} "
                    f"({best['peak_tps']:,.0f} tok/s)",
                    "dim",
                )
            )
    elif stop_reason == "throughput_gain_converged":
        out.append(
            "    "
            + style.paint("Peak refinement converged on total-throughput gain.", "dim")
        )

    if ladder:
        coverage = _ladder_coverage(points, chosen, unit, dash)
        out.append(f"    {style.paint(coverage, 'dim')}")
        out.append(
            "    "
            + style.paint(
                f"{len(points)} stage(s), at the loads you gave {dash} no search "
                "was run",
                "dim",
            )
        )
        out.append(
            f"    {style.paint('Use --auto-tune to have the peak located.', 'dim')}"
        )
        return out

    # A ramp always records why it stopped; an absent reason means the caller had
    # none to give (a hand-built sidecar, a truncated file). "stopped: ?" states a
    # non-fact, so the line is dropped instead.
    if stop_reason:
        tail = f"stopped: {stop_reason}"
        if bracket_reason and bracket_reason != stop_reason:
            tail += f" (bracket: {bracket_reason})"
        if probe_ceiling:
            tail += f" {sep} probe ceiling {probe_ceiling:.1f} req/s"
        out.append(f"    {style.paint(tail, 'dim')}")
    return out


def _ladder_coverage(points: Sequence[dict], chosen: dict, unit: str, dash: str) -> str:
    """Whether the stage list actually contains a peak, or just ends at one.

    This is the only thing a manual ladder can say that a table of rows cannot,
    and it is the difference between an answer and a coincidence: an argmax at
    either END of the ladder means the curve was still moving when the rungs ran
    out, so the real peak is outside what was measured. Only an argmax with a
    lower neighbour on BOTH sides is bracketed by the data.
    """
    knobs = [_num(p, "knob") for p in points]
    knob = _num(chosen, "knob")
    top = f"{_fmt_knob(knob)} {unit}"
    if len(points) < 3:
        return (
            f"too few stages to bracket a peak {dash} {top} is simply the best of them"
        )
    if knob >= max(knobs):
        return (
            f"still climbing at the top rung {dash} the peak may lie above {top}. "
            "Add a higher stage."
        )
    if knob <= min(knobs):
        return (
            f"best at the lowest rung {dash} the peak may lie below {top}. "
            "Add a lower stage."
        )
    return f"the ladder brackets this peak {dash} both neighbours of {top} are lower"
