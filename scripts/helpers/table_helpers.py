#!/usr/bin/env python3
# scripts/helpers/table_helpers.py

from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Callable, Dict, Any

import numpy as np
import pandas as pd

from scripts.helpers.data_helpers import is_finite

# ---------------------------------------------------------------------------
# LaTeX table formatting defaults
# ---------------------------------------------------------------------------

DEFAULT_TABLE_PLACEMENT = "H"  # e.g. "H", "t", "ht", "htbp"
DEFAULT_TABLE_ENVIRONMENT = "table"  # "table" or "table*"

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
    return rf"${m}\pm{abs(float(std_v)):.{std_dec}f}$"

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
    caption_short: Optional[str] = None,
    label: str = "",
    resizebox: bool = True,
    max_height: Optional[str] = None,
    placement: str = DEFAULT_TABLE_PLACEMENT,
    environment: str = DEFAULT_TABLE_ENVIRONMENT,
) -> None:
    """Wrap *tabular_lines* in a ``table`` / ``table*`` float and write to *out_path*.

    When *resizebox* is True, the tabular is wrapped so it fits the available
    width.  If *max_height* is also given (e.g. ``"0.9\\textheight"``), an
    ``\\adjustbox`` with both ``max width`` and ``max totalheight`` is used
    instead of ``\\resizebox`` so the table never overflows the page.
    """
    # Build caption line while preserving current behavior
    if caption_short and caption:
        caption_line = rf"\caption[{caption_short}]{{{caption}}}"
    else:
        caption_line = rf"\caption{{{caption}}}"

    wrapped: List[str] = [
        rf"\begin{{{environment}}}[{placement}]",
        r"\centering",
        caption_line,
        rf"\label{{{label}}}",
    ]
    width = r"\textwidth" if environment.endswith("*") else r"\linewidth"
    if resizebox and max_height:
        wrapped.append(rf"\adjustbox{{max width={width}, max totalheight={max_height}}}{{%")
    elif resizebox:
        wrapped.append(rf"\adjustbox{{max width={width}}}{{%")
    wrapped.extend(tabular_lines)
    if resizebox:
        # Append % to last tabular line to avoid spurious whitespace before closing brace
        wrapped[-1] = wrapped[-1] + "%"
        wrapped.append(r"}")
    wrapped.extend([
        rf"\end{{{environment}}}",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(wrapped), encoding="utf-8")


# ---------------------------------------------------------------------------
# Data / lookup helpers
# ---------------------------------------------------------------------------

def ensure_int(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
            out = out.dropna(subset=[c])
            out[c] = out[c].astype(int)
    return out


def as_lookup(df: pd.DataFrame, index_cols: Sequence[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=list(index_cols), keep="last").set_index(list(index_cols)).sort_index()


def safe_at(lookup: pd.DataFrame, key: Tuple[Any, ...], col: str, default: float = float("nan")) -> float:
    try:
        return float(lookup.at[key, col])
    except Exception:
        return default


def to_float_list(v: Any) -> List[float]:
    if v is None:
        return []
    try:
        if np.isscalar(v):
            return [float(v)]
    except Exception:
        pass
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    return [float(v)]


def aggregate_mean_std(df: pd.DataFrame, group_cols: Sequence[str], seed_col: str = "seed") -> pd.DataFrame:
    numeric_cols = [
        c for c in df.columns
        if c not in set(group_cols).union({seed_col}) and pd.api.types.is_numeric_dtype(df[c])
    ]
    grp = df.groupby(list(group_cols), dropna=False)
    mean_df = grp[numeric_cols].mean(numeric_only=True).reset_index()
    std_df = grp[numeric_cols].std(numeric_only=True).reset_index()

    out = mean_df.copy()
    for c in numeric_cols:
        out[f"{c}_std"] = std_df[c].to_numpy()
    return out


def paired_diff_mean(
    df: pd.DataFrame,
    *,
    left_filter: Dict[str, Any],
    right_filter: Dict[str, Any],
    merge_cols: Sequence[str],
    metric_cols: Sequence[str],
    suffix_left: str = "_L",
    suffix_right: str = "_R",
) -> pd.DataFrame:
    left = df.copy()
    right = df.copy()
    for k, v in left_filter.items():
        left = left[left[k] == v]
    for k, v in right_filter.items():
        right = right[right[k] == v]

    left = left[list(merge_cols) + list(metric_cols)].rename(columns={c: f"{c}{suffix_left}" for c in metric_cols})
    right = right[list(merge_cols) + list(metric_cols)].rename(columns={c: f"{c}{suffix_right}" for c in metric_cols})

    merged = right.merge(left, on=list(merge_cols), how="inner")
    if merged.empty:
        return merged

    out = merged[list(merge_cols)].copy()
    for c in metric_cols:
        out[c] = merged[f"{c}{suffix_right}"] - merged[f"{c}{suffix_left}"]
    return out


# ---------------------------------------------------------------------------
# LaTeX helpers (grouped headers / formatting)
# ---------------------------------------------------------------------------

def header_values_with_first_label(values: Sequence[Any], *, first_prefix_tex: str, fmt: Callable[[Any], str]) -> List[str]:
    vals = list(values)
    out: List[str] = []
    for i, v in enumerate(vals):
        s = fmt(v)
        if i == 0:
            out.append(rf"\llap{{{first_prefix_tex}}}{s}")
        else:
            out.append(s)
    return out


def repeat_header_values_with_first_label(
    groups: int,
    values: Sequence[Any],
    *,
    first_prefix_tex: str,
    fmt: Callable[[Any], str],
) -> List[str]:
    out: List[str] = []
    first = True
    for _ in range(groups):
        for v in values:
            s = fmt(v)
            if first:
                out.append(rf"\llap{{{first_prefix_tex}}}{s}")
                first = False
            else:
                out.append(s)
    return out

