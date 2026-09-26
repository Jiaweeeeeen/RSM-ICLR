"""Shared table and figure helpers for the saved-record reports.

One colour-blind-safe mapping across every figure: hue and marker name the
attention family, line style names the memory regime. :func:`_write_table`
writes one CSV (every column of every row) with its Markdown twin;
:func:`_plot_series` draws one condition's series with its interval. The tier
report in :mod:`reasoned_icrl.analysis.tier` is the consumer.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasoned_icrl.analysis.runs import markdown_table
from reasoned_icrl.experiments.contracts import ALL_CONDITIONS

# The categorical hues of the validated reference palette, in its fixed order
# (slots 1-5), assigned to the attention families; regimes take line styles.
ATTENTION_HUES = {
    "ordinary": "#2a78d6",
    "dual": "#eb6834",
    "gru": "#1baf7a",
    "dual_content": "#eda100",
    "write_off": "#e87ba4",
    "memo": "#CC79A7",  # the figure plan's purple for both Memo recipes
}
ATTENTION_MARKERS = {
    "ordinary": "o",
    "dual": "^",
    "gru": "D",
    "dual_content": "s",
    "write_off": "v",
    "memo": "p",
}
LineStyle = str | tuple[int, tuple[int, ...]]
"""A Matplotlib line style: a named style or an ``(offset, dashes)`` pattern."""

REGIME_LINESTYLES: dict[str, LineStyle] = {
    "full": "-",
    "segment": ":",
    "summary": "--",
    "window": "-.",
}
MEMO_LINESTYLES: dict[str, LineStyle] = {"jittered": "-", "fixed": "-."}
"""The figure plan's Memo rule: solid for the jittered recipe, dash-dot for
fixed segments; the family hue is purple for both."""


@dataclass(frozen=True, slots=True)
class Style:
    """How one condition is drawn in every figure."""

    color: str
    marker: str
    linestyle: LineStyle
    family: str
    regime: str


def condition_style(condition: str) -> Style:
    """Hue and marker by attention family, line style by memory regime."""
    spec = ALL_CONDITIONS.get(condition)
    if spec is None:
        return Style(ATTENTION_HUES["ordinary"], "o", "-", "ordinary", "full")
    if spec.memory == "accumulated":
        return Style(
            ATTENTION_HUES["memo"],
            ATTENTION_MARKERS["memo"],
            MEMO_LINESTYLES[spec.segmentation],
            "memo",
            str(spec.memory),
        )
    if spec.trajectory_encoder == "gru":
        family = "gru"
    elif spec.writer == "relational_off":
        family = "write_off"
    elif spec.dat_mode == "dual_content":
        family = "dual_content"
    elif spec.dat_mode is not None:
        family = "dual"
    else:
        family = "ordinary"
    regime = str(spec.memory)
    return Style(
        ATTENTION_HUES[family],
        ATTENTION_MARKERS[family],
        REGIME_LINESTYLES.get(regime, "-"),
        family,
        regime,
    )


def _write_table(
    root: Path,
    name: str,
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
    note: str = "",
) -> Path:
    """One CSV (every column of every row) and its Markdown twin."""
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with (root / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or ["status"])
        writer.writeheader()
        writer.writerows(rows if rows else [{"status": "pending"}])
    body = (
        markdown_table(
            [{k: ("" if v is None else v) for k, v in row.items()} for row in rows]
        )
        if rows
        else "_pending: no complete records_"
    )
    text = f"### {title}\n\n"
    if note:
        text += note + "\n\n"
    path = root / f"{name}.md"
    path.write_text(text + body + "\n", encoding="utf-8")
    return path


def _plot_series(
    ax: Any,
    condition: str,
    xs: Sequence[float],
    ys: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    *,
    label: str | None = None,
    markers: bool = True,
) -> None:
    style = condition_style(condition)
    ax.plot(
        xs,
        ys,
        color=style.color,
        linestyle=style.linestyle,
        linewidth=1.5,
        marker=style.marker if markers else None,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=condition if label is None else label,
    )
    ax.fill_between(xs, lows, highs, color=style.color, alpha=0.15, linewidth=0)


__all__ = [
    "ATTENTION_HUES",
    "ATTENTION_MARKERS",
    "MEMO_LINESTYLES",
    "REGIME_LINESTYLES",
    "Style",
    "condition_style",
]
