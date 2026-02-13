#!/usr/bin/env python3
# scripts/helpers/table_helpers.py

from pathlib import Path
from typing import List, Optional, Tuple

from scripts.helpers.data_helpers import is_finite

# ---------------------------------------------------------------------------
# LaTeX table formatting defaults
# ---------------------------------------------------------------------------

DEFAULT_TABLE_FONT_SIZE = r"\tiny"
DEFAULT_TABLE_TABCOLSEP = "1.2pt"
DEFAULT_TABLE_ARRAYSTRETCH = "1.12"

def nan_str() -> str:
    """
    Return LaTeX representation for NaN/missing values.
    """
    return r"\text{--}"

def fmt_pct(x: object, decimals: int = 1) -> str:
    """
    Format a value as a percentage string for LaTeX tables.
    """
    if not is_finite(x):
        return nan_str()
    v = float(x)
    if abs(v) < 5e-13:
        v = 0.0
    return f"{v:.{decimals}f}"

def fmt_signed(x: object, decimals: int) -> str:
    """
    Format a signed float with explicit + or - prefix.
    """
    if not is_finite(x):
        return nan_str()
    v = round(float(x), int(decimals))
    return f"{(0.0 if v == 0.0 else v):+.{decimals}f}"

def fmt_unsigned_int(x: object) -> str:
    """
    Format an unsigned integer value.
    """
    if not is_finite(x):
        return nan_str()
    return f"{int(round(float(x))):d}"

def fmt_mean_std(
    mean_v: object,
    std_v: object,
    *,
    mean_signed: bool,
    mean_dec: int,
    std_dec: int,
) -> str:
    """
    Format a mean ± std value for LaTeX.
    """
    if not is_finite(mean_v):
        return nan_str()
    m = f"{float(mean_v):+.{mean_dec}f}" if mean_signed else f"{float(mean_v):.{mean_dec}f}"
    if not is_finite(std_v):
        return m
    return rf"${m}\,\pm\,{abs(float(std_v)):.{std_dec}f}$"

def fmt_mean_std_split(
    mean_v: object,
    std_v: object,
    *,
    mean_signed: bool,
    mean_dec: int,
    std_dec: int,
) -> Tuple[str, str]:
    """
    Format a mean ± std value for LaTeX as separate columns.
    Returns (mean_str, std_str) for aligned ± tables.
    """
    if not is_finite(mean_v):
        return (nan_str(), "")
    m = f"{float(mean_v):+.{mean_dec}f}" if mean_signed else f"{float(mean_v):.{mean_dec}f}"
    if not is_finite(std_v):
        s = ""
    else:
        s = f"{abs(float(std_v)):.{std_dec}f}"
    return (m, s)

def metric_header_tex(label: str) -> str:
    """
    Create a LaTeX makecell header from a metric label.
    """
    s = str(label).strip()
    def split_core(core: str) -> Optional[Tuple[str, str]]:
        core = core.strip()
        if r"\;(" in core:
            left, right = core.split(r"\;(", 1)
            return left.strip(), "(" + right.strip()
        if " (" in core and core.endswith(")"):
            left, right = core.rsplit(" (", 1)
            return left.strip(), "(" + right.strip()
        return None
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        core = s[1:-1].strip()
        parts = split_core(core)
        if parts is None:
            return rf"\makecell{{${core}$}}"
        left, right = parts
        return rf"\makecell{{${left}$\\${right}$}}"
    parts = split_core(s)
    if parts is None:
        return rf"\makecell{{{s}}}"
    left, right = parts
    return rf"\makecell{{{left}\\{right}}}"


# ---------------------------------------------------------------------------
# Shared LaTeX table structure helpers
# ---------------------------------------------------------------------------

def latex_cmidrules(n_groups: int, cols_per_group: int, *, start_col: int = 2) -> str:
    """Generate ``\\cmidrule(lr)`` separators for *n_groups* column groups."""
    parts: List[str] = []
    col = start_col
    for _ in range(n_groups):
        end = col + cols_per_group - 1
        parts.append(rf"\cmidrule(lr){{{col}-{end}}}")
        col = end + 1
    return "".join(parts)


def write_latex_table(
    out_path: Path,
    tabular_lines: List[str],
    *,
    caption: str = "",
    label: str = "",
    font_size: str = DEFAULT_TABLE_FONT_SIZE,
    tabcolsep: str = DEFAULT_TABLE_TABCOLSEP,
    arraystretch: str = DEFAULT_TABLE_ARRAYSTRETCH,
) -> None:
    """Wrap *tabular_lines* in a ``table*`` float and write to *out_path*."""
    wrapped: List[str] = [
        r"\begin{table}[H]",
        font_size,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        *tabular_lines,
        r"\end{table}",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(wrapped), encoding="utf-8")
