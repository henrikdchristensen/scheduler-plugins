#!/usr/bin/env python3
# scripts/helpers/plot_helpers.py

from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

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

