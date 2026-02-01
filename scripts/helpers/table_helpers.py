#!/usr/bin/env python3
# scripts/helpers/table_helpers.py
"""
Shared LaTeX table formatting helper functions for kwok_workload_once and kwok_trace_replayer.
"""

from typing import Optional, Tuple

from scripts.helpers.data_helpers import is_finite

def nan_str() -> str:
    """Return LaTeX representation for NaN/missing values."""
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

def fmt_pm(
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
        return rf"\ensuremath{{{m}}}"
    return rf"\ensuremath{{{m}\,\pm\,{abs(float(std_v)):.{std_dec}f}}}"

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
