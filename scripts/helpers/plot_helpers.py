#!/usr/bin/env python3
# scripts/helpers/plot_helpers.py

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from scripts.helpers.plot_config import (
    PLOT_TITLE_FONTSIZE,
    PLOT_AXIS_LABEL_FONTSIZE,
    PLOT_TICK_FONTSIZE,
    PLOT_FIGURE_DPI,
    PLOT_FORMATS,
)

# Unified color palette shared across all experiment scripts
PLOT_COLORS = list(plt.get_cmap("tab20c").colors)


def configure_matplotlib() -> None:
    """Apply shared matplotlib rcParams for all experiment plots."""
    mpl.rcParams.update(
        {
            "axes.titlesize": PLOT_TITLE_FONTSIZE,
            "axes.labelsize": PLOT_AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": PLOT_TICK_FONTSIZE,
            "ytick.labelsize": PLOT_TICK_FONTSIZE,
        }
    )


def save_figure(
    fig: mpl.figure.Figure,
    out_path: Path,
    *,
    formats: List[str] = PLOT_FORMATS,
    dpi: int = PLOT_FIGURE_DPI,
    bbox_inches: Optional[str] = "tight",
) -> None:
    """Save figure in multiple formats, then close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fname = out_path.with_suffix(f".{ext}")
        fig.savefig(fname, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def compute_step(span: float, target_ticks: int) -> float:
    if span <= 0:
        return 1.0
    raw = span / max(1, int(target_ticks))
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10 ** exp
    cands = [1 * base, 2 * base, 5 * base, 10 * base]
    return min(cands, key=lambda s: abs(s - raw))


def set_linear_ticks(ax: plt.Axes, ylim: Tuple[float, float], min_ticks: int = 5) -> None:
    y0, y1 = float(ylim[0]), float(ylim[1])
    lo = int(math.ceil(y0))
    hi = int(math.floor(y1))
    ticks = list(range(lo, hi + 1))
    if len(ticks) < min_ticks:
        ticks = [round(float(x), 2) for x in np.linspace(y0, y1, min_ticks)]
    ax.set_yticks(ticks)


def set_count_ticks(ax: plt.Axes, ylim: Tuple[float, float], max_ticks: int = 7) -> None:
    lo, hi = max(0.0, float(ylim[0])), float(ylim[1])
    if hi <= 0:
        ax.set_yticks([0])
        return
    step = max(1.0, compute_step(hi - lo, max_ticks - 1))
    ticks = [0.0]
    t = 0.0
    for _ in range(100):
        t += step
        if t > hi + 1e-9:
            break
        ticks.append(t)
    if all(abs(x - round(x)) < 1e-9 for x in ticks):
        ticks = [int(round(x)) for x in ticks]
    ax.set_yticks(ticks)


def set_symmetric_count_ticks(ax: plt.Axes, ylim: Tuple[float, float], max_ticks_total: int = 7) -> None:
    lo, hi = float(ylim[0]), float(ylim[1])
    m = max(abs(lo), abs(hi))
    if m <= 0:
        ax.set_yticks([0])
        return
    per_side = max(1, (max_ticks_total - 1) // 2)
    step = max(1e-12, compute_step(m, per_side))
    ticks = [0.0]
    for i in range(1, per_side + 1):
        ticks.extend([-(i * step), i * step])
    ticks = sorted([t for t in ticks if lo - 1e-9 <= t <= hi + 1e-9])
    if all(abs(x - round(x)) < 1e-9 for x in ticks):
        ticks = [int(round(x)) for x in ticks]
    ax.set_yticks(ticks)


def lighter_color(color: Any, factor: float = 0.45) -> Tuple[float, float, float]:
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (
        r + (1.0 - r) * factor,
        g + (1.0 - g) * factor,
        b + (1.0 - b) * factor,
    )


def center_two_legends(
    fig: plt.Figure,
    *,
    left_handles: Sequence[Any],
    left_labels: Sequence[str],
    right_handles: Sequence[Any],
    right_labels: Sequence[str],
    left_title: str,
    right_title: str,
    y_top: float,
    x_center: float,
    gap: float,
    legend_kwargs: Dict[str, Any],
    title_fontsize: float,
    x_offset: float = 0.0,
    left_ncol: Optional[int] = None,
    right_ncol: Optional[int] = None,
) -> Tuple[Any, Any]:
    left_kwargs = dict(legend_kwargs)
    right_kwargs = dict(legend_kwargs)

    if left_ncol is not None:
        left_kwargs["ncol"] = left_ncol
    if right_ncol is not None:
        right_kwargs["ncol"] = right_ncol

    leg_l = fig.legend(
        left_handles, left_labels,
        title=left_title,
        title_fontproperties={"size": title_fontsize, "weight": "bold"},
        loc="upper left", bbox_to_anchor=(0, y_top), **left_kwargs
    )
    leg_r = fig.legend(
        right_handles, right_labels,
        title=right_title,
        title_fontproperties={"size": title_fontsize, "weight": "bold"},
        loc="upper left", bbox_to_anchor=(0, y_top), **right_kwargs
    )

    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    bb_l = leg_l.get_window_extent(ren).transformed(fig.transFigure.inverted())
    bb_r = leg_r.get_window_extent(ren).transformed(fig.transFigure.inverted())

    total_w = bb_l.width + gap + bb_r.width
    x0 = x_center - total_w / 2 + x_offset

    leg_l.set_bbox_to_anchor((x0, y_top), transform=fig.transFigure)
    leg_l._loc = leg_l.codes["upper left"]  # noqa: SLF001
    leg_r.set_bbox_to_anchor((x0 + bb_l.width + gap, y_top), transform=fig.transFigure)
    leg_r._loc = leg_r.codes["upper left"]  # noqa: SLF001
    return leg_l, leg_r


def shape_legend_handles(
    *,
    nodes_order: Sequence[int],
    specs: Sequence[Tuple[str, str, float, str]],
    edge_width: float,
) -> Tuple[List[Line2D], List[str]]:
    handles: List[Line2D] = []
    labels: List[str] = []
    for i, (marker, label_tmpl, msize, face) in enumerate(specs):
        handles.append(
            Line2D(
                [0], [0], marker=marker, color="none",
                markerfacecolor=face,
                markeredgecolor="black",
                markeredgewidth=edge_width,
                markersize=msize, linestyle="None"
            )
        )
        if "{nodes}" in label_tmpl and i < len(nodes_order):
            labels.append(label_tmpl.format(nodes=nodes_order[i]))
        else:
            labels.append(label_tmpl)
    return handles, labels


def color_patch_handles(colors: Sequence[Any], *, edge_width: float = 0.6) -> List[mpatches.Rectangle]:
    return [mpatches.Rectangle((0, 0), 1, 1, fc=c, ec="black", linewidth=edge_width) for c in colors]