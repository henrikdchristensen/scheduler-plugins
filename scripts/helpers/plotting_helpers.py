#!/usr/bin/env python3
# scripts/helpers/plotting_helpers.py
"""
Shared plotting helper functions for kwok_workload_once and kwok_trace_replayer.
"""

from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt

from scripts.config.plot_config import (
    PLOT_FIGURE_DPI,
    PLOT_TITLE_FONTSIZE,
    PLOT_AXIS_LABEL_FONTSIZE,
    PLOT_TICK_FONTSIZE,
)

# Default figure formats for saving
DEFAULT_FIGURE_FORMATS: List[str] = ["pdf", "png"]


def configure_matplotlib() -> None:
    """
    Configure matplotlib with standard font sizes for publication-quality plots.
    Uses settings from scripts.config.plot_config.
    """
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
    formats: List[str] = DEFAULT_FIGURE_FORMATS,
    dpi: int = PLOT_FIGURE_DPI,
    close: bool = True,
    verbose: bool = True,
) -> None:
    """
    Save a matplotlib figure to multiple file formats.

    Args:
        fig: The matplotlib figure to save.
        out_path: Base output path (without extension).
        formats: List of formats to save (default: ["pdf", "png"]).
        dpi: Resolution for raster formats (default from plot_config).
        close: Whether to close the figure after saving.
        verbose: Whether to print confirmation message.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fname = out_path.with_suffix(f".{ext}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    if verbose:
        print(f"[ok] saved figure: {out_path} ({', '.join(formats)})")
