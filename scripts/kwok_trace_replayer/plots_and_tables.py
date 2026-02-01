#!/usr/bin/env python3
# scripts/kwok_trace_replayer/plots_and_tables.py
"""
python -m scripts.kwok_trace_replayer.plots_and_tables
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.config.plot_config import (
    PLOT_AXIS_LABEL_FONTSIZE,
    PLOT_FIGURE_DPI,
    PLOT_LEGEND_COLUMN_SPACING,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_HANDLE_LENGTH,
    PLOT_LEGEND_HANDLE_TEXT_PAD,
    PLOT_TICK_FONTSIZE,
    PLOT_TITLE_FONTSIZE,
)

# =============================================================================
# CONFIG (edit here)
# =============================================================================

IN_RESULTS_SEEDS = Path("analysis/kwok_trace_replayer/results_seeds.csv")

OUT_DIR = Path("analysis/kwok_trace_replayer")
OUT_TABLES_DIR = OUT_DIR / "tables"
OUT_FIGURES_DIR = OUT_DIR / "figures"

SEED_COL = "seed"
SEED_JITTER_FRAC = 0.12  # jitter as fraction of mode_spacing

PRIORITIES_TO_SHOW = [1, 4]
INTER_ARRIVALS_TO_SHOW = [2.0, 4.0, 8.0, 16.0]

# =============================================================================
# Expected schema from seal_results.py (no backward compatibility)
# =============================================================================

SEED_OUT_COLS = [
    "job_name",
    "plugin_config",
    "seed",
    "T_end_s_mean",
    "delta_U_pct_cpu_mean","delta_U_pct_mem_mean","delta_U_pct_eff_mean",
    "delta_R_num_p1_mean","delta_R_num_p2_mean","delta_R_num_p3_mean","delta_R_num_p4_mean","delta_R_num_total_mean",
    "delta_D_num_p1_mean","delta_D_num_p2_mean","delta_D_num_p3_mean","delta_D_num_p4_mean","delta_D_num_total_mean",
    "delta_L_ms_p1_mean","delta_L_ms_p2_mean","delta_L_ms_p3_mean","delta_L_ms_p4_mean","delta_L_ms_total_mean",
    "solver_attempts_mean","solver_optimal_mean","solver_feasible_mean","solver_failed_mean",
    "plan_not_applicable_mean","plan_activated_mean",
]

# =============================================================================
# Parsing: job_name + plugin_config
# =============================================================================

JOB_RE = re.compile(r"nodes=(\d+)_prio=(\d+)_arrival=([0-9.]+)s")


def parse_job(job_name: str) -> Tuple[int, int, float]:
    m = JOB_RE.fullmatch(str(job_name).strip())
    if not m:
        raise ValueError(f"Invalid job_name: {job_name}")
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def parse_plugin_config(plugin_config: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for tok in str(plugin_config).split("_"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return kv


def canonical_mode(mode: str) -> str:
    s = str(mode).strip().lower()
    s_compact = s.replace("-", "").replace("_", "")
    if s_compact in {"schedulingfailure", "schedfailure"}:
        return "schedulingfailure"

    m = re.fullmatch(r"periodic-?([0-9.]+)s", s)
    if m:
        return f"periodic{m.group(1)}s"

    m = re.fullmatch(r"(?:stable-queue|stablequeue)-?([0-9.]+)s", s)
    if m:
        return f"stable-queue-{m.group(1)}s"

    return s


@dataclass(frozen=True)
class RowKey:
    mode: str
    blocking: int
    defpreempt: int

    @staticmethod
    def from_plugin_config(plugin_config: str) -> "RowKey":
        kv = parse_plugin_config(plugin_config)
        mode = canonical_mode(kv.get("mode", "unknown"))
        blocking = 1 if str(kv.get("blocking", "0")).lower() in {"1", "true", "yes"} else 0
        defpreempt = int(str(kv.get("defpreempt", "0"))) if "defpreempt" in kv else 0
        return RowKey(mode=mode, blocking=blocking, defpreempt=defpreempt)


# =============================================================================
# Modes (rows in main tables, styling in plots)
# =============================================================================

@dataclass(frozen=True)
class ModeSpec:
    mode: str
    blocking: int
    abbr: str
    label: str
    rank: int
    palette: str  # "set2" or "set3"
    color_idx: int


MODE_SPECS: List[ModeSpec] = [
    ModeSpec("schedulingfailure", 1, "SF-B", "Scheduling-failure (blocking)", 0, "set2", 0),
    ModeSpec("schedulingfailure", 0, "SF-NB", "Scheduling-failure (non-blocking)", 1, "set2", 1),
    ModeSpec("periodic8s", 1, "PR-8-B", "Periodic (blocking), 8s interval", 2, "set2", 2),
    ModeSpec("periodic8s", 0, "PR-8-NB", "Periodic (non-blocking), 8s interval", 3, "set2", 3),
    ModeSpec("periodic32s", 1, "PR-32-B", "Periodic (blocking), 32s interval", 4, "set2", 4),
    ModeSpec("periodic32s", 0, "PR-32-NB", "Periodic (non-blocking), 32s interval", 5, "set2", 5),
    ModeSpec("stable-queue-2s", 1, "SQ-2-B", "Stable-queue (blocking), 2s delay", 6, "set2", 6),
    ModeSpec("stable-queue-2s", 0, "SQ-2-NB", "Stable-queue (non-blocking), 2s delay", 7, "set2", 7),
    ModeSpec("stable-queue-8s", 1, "SQ-8-B", "Stable-queue (blocking), 8s delay", 8, "set3", 3),
    ModeSpec("stable-queue-8s", 0, "SQ-8-NB", "Stable-queue (non-blocking), 8s delay", 9, "set3", 4),
]

_SPEC_BY_MODE_BLOCK: Dict[Tuple[str, int], ModeSpec] = {(s.mode, int(s.blocking)): s for s in MODE_SPECS}


def rk_rank(rk: RowKey) -> int:
    spec = _SPEC_BY_MODE_BLOCK.get((rk.mode, int(rk.blocking)))
    return int(spec.rank) if spec else 10_000


def rk_label(rk: RowKey) -> str:
    spec = _SPEC_BY_MODE_BLOCK.get((rk.mode, int(rk.blocking)))
    return spec.label if spec else f"{rk.mode}:{rk.blocking}"


def rk_color(rk: RowKey):
    set2 = plt.get_cmap("Set2").colors
    set3 = plt.get_cmap("Set3").colors
    spec = _SPEC_BY_MODE_BLOCK.get((rk.mode, int(rk.blocking)))
    if not spec:
        return set3[0]
    palette = set2 if spec.palette == "set2" else set3
    return palette[int(spec.color_idx) % len(palette)]


def sort_rks(rks: Iterable[RowKey]) -> List[RowKey]:
    return sorted(set(rks), key=lambda r: (rk_rank(r), r.mode, int(r.blocking), int(r.defpreempt)))


# =============================================================================
# Plot selection (fixed)
# =============================================================================

def select_arrivals_order(all_arrivals: List[float]) -> List[float]:
    all_sorted = sorted(set(float(a) for a in all_arrivals))
    if INTER_ARRIVALS_TO_SHOW is None:
        return all_sorted

    wanted = [float(a) for a in INTER_ARRIVALS_TO_SHOW]
    wanted_set = set(wanted)

    missing = [a for a in wanted if a not in set(all_sorted)]
    if missing:
        raise SystemExit(f"INTER_ARRIVALS_TO_SHOW_S contains values not in data: {missing}. Available: {all_sorted}")

    # Preserve the order provided in INTER_ARRIVALS_TO_SHOW_S (don’t re-sort unless you want to)
    return [a for a in wanted if a in wanted_set]


# Main grids show only a subset (but tables include ALL MODE_SPECS)
MAIN_PLOT_MODES: List[Tuple[str, int]] = [
    ("schedulingfailure", 1),
    ("schedulingfailure", 0),
    ("periodic8s", 1),
    ("periodic8s", 0),
    ("stable-queue-2s", 1),
    ("stable-queue-2s", 0),
]

# Periodic vs stable deltas (fixed)
DELTA_SERIES: List[Tuple[str, str, str, int]] = [
    ("PR 8→32 (B)", "periodic8s", "periodic32s", 1),
    ("SQ 2→8 (B)", "stable-queue-2s", "stable-queue-8s", 1),
]

DELTA_NAMES = [d[0] for d in DELTA_SERIES]

# Metric columns used in delta computations
DELTA_METRIC_COLS = [
    "delta_U_pct_eff_mean",
    "delta_L_ms_total_mean",
    "delta_D_num_total_mean",
    "solver_attempts_mean",
    "plan_activated_mean",
]

# =============================================================================
# Plot styling / layout
# =============================================================================

PLOT_TICK_PAD = 2.0
PLOT_MARKER_SIZE = 2.5
PLOT_MARKER_LINEWIDTH = 0.4
PLOT_MIN_LINEAR_YTICKS = 5

PLOT_HEIGHT = 8.3

GRID_FIGSIZE_MAIN = (5.4, PLOT_HEIGHT)
GRID_FIGSIZE_DELTAS = (2.15, PLOT_HEIGHT)

GRID_LEGEND_NCOL_MAIN = 3
GRID_LEGEND_NCOL_DELTAS = 1

GRID_LEGEND_PAD = 0.033
GRID_LEFT_MAIN = 0.08
GRID_LEFT_DELTAS = 0.2
GRID_RIGHT = 0.99
GRID_BOTTOM = 0.03
GRID_TOP = 0.925
GRID_WSPACE = 0.10
GRID_HSPACE = 0.10
GRID_YLABEL_PAD_PT = 22.0

PLOT_ARRIVAL_X_SPACING = 0.35
PLOT_MODE_X_SPACING_MAIN = 0.05
PLOT_MODE_X_SPACING_DELTAS = 0.11

# =============================================================================
# Y-axis config: SAME for defpreempt=0/1.
# Separate config for main vs deltas.
# =============================================================================

SYMLOG_BASE = 10.0
SYMLOG_LINSCALE = 1.0
SYMLOG_LINTHRESH_LAT_MS = 1.0
SYMLOG_LINTHRESH_DELETIONS = 1.0


@dataclass(frozen=True)
class YAxisCfg:
    scale: str  # "linear" | "symlog"
    ylim: Tuple[float, float]
    symlog_linthresh: float = 1.0


Y_MAIN: Dict[str, YAxisCfg] = {
    "util": YAxisCfg("linear", (-5.0, 5.0)),
    "latency": YAxisCfg("symlog", (-1e5 - 1.0, 1e5 + 1.0), SYMLOG_LINTHRESH_LAT_MS),
    "deletions": YAxisCfg("symlog", (-1e4 - 1.0, 1e4 + 1.0), SYMLOG_LINTHRESH_DELETIONS),
}

Y_DELTAS: Dict[str, YAxisCfg] = {
    "util": YAxisCfg("linear", (-1.0, 1.0)),
    "latency": YAxisCfg("symlog", (-1e4 - 1.0, 1e4 + 1.0), SYMLOG_LINTHRESH_LAT_MS),
    "deletions": YAxisCfg("symlog", (-1e3 - 1.0, 1e3 + 1.0), SYMLOG_LINTHRESH_DELETIONS),
}


# =============================================================================
# Grid row configuration (for unified grid function)
# =============================================================================

@dataclass(frozen=True)
class GridRowSpec:
    """Configuration for one row in a grid plot."""
    col_name: str
    ycfg_key: str  # key in Y_MAIN or Y_DELTAS
    y_tick_strategy: Optional[str] = None  # "count_sparse", "count_sparse_symmetric", or None
    ylabel: str = ""
    is_bottom: bool = False  # whether this is the bottom row (shows x labels)


# Row configurations - reused for both main and delta grids
GRID_ROW_SPECS = [
    GridRowSpec("delta_U_pct_eff_mean", "util", ylabel=r"$\mathrm{diff.}\ \mathrm{usage}\;(\%)$"),
    GridRowSpec("delta_L_ms_total_mean", "latency", ylabel=r"$\mathrm{diff.}\ \mathrm{latency}\;(\mathrm{ms})$"),
    GridRowSpec("delta_D_num_total_mean", "deletions", ylabel=r"$\mathrm{diff.}\ \mathrm{deletions}$"),
    GridRowSpec("solver_attempts_mean", "solver", y_tick_strategy="count_sparse", ylabel=r"$\mathrm{solver\ runs}$"),
    GridRowSpec("plan_activated_mean", "plans", y_tick_strategy="count_sparse", ylabel=r"$\mathrm{plan\ activations}$", is_bottom=True),
]

# =============================================================================
# Formatting helpers (tables)
# =============================================================================

TABLE_DECIMALS = 1


def is_finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def nan_str() -> str:
    return r"\text{--}"


def fmt_arrival_value(a: float) -> Union[int, float]:
    """Convert arrival to int if close to integer, otherwise keep as float."""
    return int(a) if abs(a - round(a)) < 1e-9 else a


def fmt_signed(x: object, decimals: int) -> str:
    if not is_finite(x): return nan_str()
    v = round(float(x), int(decimals))
    return f"{(0.0 if v == 0.0 else v):+.{decimals}f}"


def fmt_unsigned_int(x: object) -> str:
    return f"{int(round(float(x))):d}" if is_finite(x) else nan_str()


def fmt_pm(mean_v: object, std_v: object, *, mean_signed: bool, mean_dec: int, std_dec: int) -> str:
    if not is_finite(mean_v): return nan_str()
    m = f"{float(mean_v):+.{mean_dec}f}" if mean_signed else f"{float(mean_v):.{mean_dec}f}"
    if not is_finite(std_v): return rf"\ensuremath{{{m}}}"
    return rf"\ensuremath{{{m}\,\pm\,{abs(float(std_v)):.{std_dec}f}}}"


def metric_header_tex(label: str) -> str:
    """
    Splits "... (unit)" or "...\\;(unit)" into two-line makecell header.
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


def rk_label_tex(rk: RowKey) -> str:
    return rf"\makecell[l]{{{rk_label(rk)}}}"


# =============================================================================
# Data loading + aggregation
# =============================================================================

KEY_COLS_MAIN = ["nodes", "priorities", "arrival_s", "mode", "blocking", "defpreempt"]


def load_results_seeds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in SEED_OUT_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing required columns: {', '.join(missing)}")

    # Parse job_name
    parsed = df["job_name"].map(parse_job)
    df[["nodes", "priorities", "arrival_s"]] = pd.DataFrame(parsed.tolist(), index=df.index, columns=["nodes", "priorities", "arrival_s"])

    # Parse plugin_config into mode/blocking/defpreempt
    rks = df["plugin_config"].map(RowKey.from_plugin_config)
    df[["mode", "blocking", "defpreempt"]] = pd.DataFrame([(r.mode, r.blocking, r.defpreempt) for r in rks], index=df.index, columns=["mode", "blocking", "defpreempt"])

    return df


def _generic_aggregate_mean_std(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Generic aggregation: compute mean+std across seeds for numeric columns.
    Adds <col>_std columns for each numeric column.
    """
    numeric_cols = [c for c in df.columns if c not in set(group_cols + [SEED_COL]) and pd.api.types.is_numeric_dtype(df[c])]

    grp = df.groupby(group_cols, dropna=False)
    mean_df = grp[numeric_cols].mean(numeric_only=True)
    std_df = grp[numeric_cols].std(numeric_only=True)

    out = mean_df.reset_index()
    std_reset = std_df.reset_index()

    for c in numeric_cols:
        out[f"{c}_std"] = std_reset[c].to_numpy()

    return out


def aggregate_mean_std(df_seeds: pd.DataFrame) -> pd.DataFrame:
    """
    One row per configuration. Produces:
      - *_mean columns: mean across seeds (for numeric cols)
      - *_std columns: std across seeds
    """
    group_cols = ["job_name", "plugin_config"] + KEY_COLS_MAIN
    return _generic_aggregate_mean_std(df_seeds, group_cols)


def _build_lookup_from_df(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """Build a lookup DataFrame indexed by key columns."""
    return df.drop_duplicates(subset=key_cols, keep="first").set_index(key_cols).sort_index()


def build_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return _build_lookup_from_df(df, KEY_COLS_MAIN)


def _safe_lookup(lookup: pd.DataFrame, key: Tuple, col: str) -> float:
    """Safely lookup a value from a DataFrame index, returning NaN if not found."""
    try:
        return float(lookup.at[key, col])
    except KeyError:
        return float("nan")


def lookup_val(lookup: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, rk: RowKey, col: str) -> float:
    key = (int(nodes), int(priorities), float(arrival_s), str(rk.mode), int(rk.blocking), int(rk.defpreempt))
    return _safe_lookup(lookup, key, col)


def lookup_mean_std(lookup: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, rk: RowKey, col_mean: str) -> Tuple[float, float]:
    key = (int(nodes), int(priorities), float(arrival_s), str(rk.mode), int(rk.blocking), int(rk.defpreempt))
    return _safe_lookup(lookup, key, col_mean), _safe_lookup(lookup, key, f"{col_mean}_std")


# =============================================================================
# Delta data (periodic vs stable) - computed from seed rows
# =============================================================================

DELTA_KEY_COLS = ["nodes", "priorities", "arrival_s", "defpreempt", "delta_name"]


def build_delta_seeds(df_seeds: pd.DataFrame) -> pd.DataFrame:
    """
    For each delta series (left->right) and each defpreempt value, compute per-seed deltas.
    Output rows:
      nodes, priorities, arrival_s, defpreempt, seed, delta_name, <metric cols>
    """
    out_rows: List[pd.DataFrame] = []

    base_cols = ["nodes", "priorities", "arrival_s", "defpreempt", SEED_COL]

    for (delta_name, left_mode, right_mode, blocking) in DELTA_SERIES:
        left = df_seeds[(df_seeds["mode"] == canonical_mode(left_mode)) & (df_seeds["blocking"] == int(blocking))].copy()
        right = df_seeds[(df_seeds["mode"] == canonical_mode(right_mode)) & (df_seeds["blocking"] == int(blocking))].copy()

        left = left[base_cols + DELTA_METRIC_COLS].rename(columns={c: f"{c}_L" for c in DELTA_METRIC_COLS})
        right = right[base_cols + DELTA_METRIC_COLS].rename(columns={c: f"{c}_R" for c in DELTA_METRIC_COLS})

        merged = right.merge(left, on=base_cols, how="inner")
        if merged.empty:
            continue

        d = merged[base_cols].copy()
        d["delta_name"] = delta_name
        for c in DELTA_METRIC_COLS:
            d[c] = merged[f"{c}_R"] - merged[f"{c}_L"]

        out_rows.append(d)

    if not out_rows:
        return pd.DataFrame(columns=DELTA_KEY_COLS + [SEED_COL])

    return pd.concat(out_rows, ignore_index=True)


def aggregate_delta_mean_std(df_delta_seeds: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate delta-by-seed to mean+std per configuration and delta_name.
    Adds <col>_std columns.
    """
    return _generic_aggregate_mean_std(df_delta_seeds, DELTA_KEY_COLS)


def build_lookup_deltas(df_delta_meanstd: pd.DataFrame) -> pd.DataFrame:
    return _build_lookup_from_df(df_delta_meanstd, DELTA_KEY_COLS)


def lookup_delta_mean_std(lookup: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, defpreempt: int, delta_name: str, col_mean: str) -> Tuple[float, float]:
    key = (int(nodes), int(priorities), float(arrival_s), int(defpreempt), str(delta_name))
    return _safe_lookup(lookup, key, col_mean), _safe_lookup(lookup, key, f"{col_mean}_std")


# =============================================================================
# Plot helpers
# =============================================================================

YVal = Union[float, Sequence[float]]
YOfFn = Callable[[RowKey, int, float, int], YVal]  # scalar or list (seeds)


def _as_finite_list(v: object) -> List[float]:
    if v is None:
        return []
    try:
        if np.isscalar(v):
            return [float(v)] if is_finite(v) else []
    except Exception:
        pass
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v if is_finite(x)]
    return [float(v)] if is_finite(v) else []


def nice_step(span: float, target_ticks: int) -> float:
    if span <= 0:
        return 1.0
    raw = span / max(1, int(target_ticks))
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10**exp
    candidates = [1 * base, 2 * base, 5 * base, 10 * base]
    return min(candidates, key=lambda s: abs(s - raw))


def set_linear_yticks(ax: plt.Axes, ylim: Tuple[float, float], *, min_ticks: int = PLOT_MIN_LINEAR_YTICKS) -> None:
    ylo, yhi = float(ylim[0]), float(ylim[1])
    y0 = int(math.ceil(ylo))
    y1 = int(math.floor(yhi))
    ticks = list(range(y0, y1 + 1))
    if len(ticks) < int(min_ticks):
        ticks = [round(float(t), 2) for t in np.linspace(ylo, yhi, int(min_ticks))]
    ax.set_yticks(ticks)


def set_count_yticks(ax: plt.Axes, ylim: Tuple[float, float], *, max_ticks: int = 5) -> None:
    ylo, yhi = float(ylim[0]), float(ylim[1])
    ylo = max(0.0, ylo)
    if yhi <= 0:
        ax.set_yticks([0])
        return
    step = max(1.0, nice_step(yhi - ylo, max_ticks - 1))
    ticks: List[float] = []
    t = 0.0
    for _ in range(50):
        ticks.append(t)
        t += step
        if t > yhi + 1e-9:
            break
    if abs(step - round(step)) < 1e-9:
        ticks = [int(round(x)) for x in ticks]
    ax.set_yticks(ticks)


def set_symmetric_count_yticks(ax: plt.Axes, ylim: Tuple[float, float], *, max_ticks_total: int = 7) -> None:
    ylo, yhi = float(ylim[0]), float(ylim[1])
    hi = max(abs(ylo), abs(yhi))
    if hi <= 0:
        ax.set_yticks([0])
        return
    max_ticks_total = max(3, int(max_ticks_total))
    per_side = max(1, (max_ticks_total - 1) // 2)
    step = max(1e-12, float(nice_step(hi, per_side)))
    ticks: List[float] = [0.0]
    for i in range(1, per_side + 1):
        t = i * step
        ticks.extend([-t, t])
    ticks = sorted([t for t in ticks if ylo - 1e-9 <= t <= yhi + 1e-9])
    if all(abs(t - round(t)) < 1e-9 for t in ticks):
        ticks = [int(round(t)) for t in ticks]
    ax.set_yticks(ticks)


def arrival_tick_label(a: float) -> str:
    a_i = fmt_arrival_value(a)
    return f"{a_i}s"


def arrival_tick_label_with_axis(a: float, xi: int, n_arr: int) -> str:
    base = arrival_tick_label(a)
    return base + ("\ninter-arrival (s)" if xi == n_arr // 2 else "")


def x_from_left_with_pad_points(fig: plt.Figure, left: float, pad_pt: float) -> float:
    pad_frac = float(pad_pt) / (72.0 * float(fig.get_figwidth()))
    return max(0.0, float(left) - pad_frac)


def draw_points_on_ax(
    *,
    ax: plt.Axes,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities: int,
    series: List[RowKey],
    y_of: YOfFn,
    ycfg: YAxisCfg,
    show_xticklabels: bool,
    show_yticklabels: bool,
    arrival_x_spacing: float,
    mode_x_spacing: float,
    y_tick_strategy: str = "auto",
    color_of: Optional[Callable[[RowKey], Any]] = None,
    seed_jitter_frac: float = SEED_JITTER_FRAC,
) -> None:
    n_arr = len(arrivals_order)
    step = float(arrival_x_spacing)

    boundaries = [i * step for i in range(n_arr + 1)]
    x_base = [(i + 0.5) * step for i in range(n_arr)]

    m = max(1, len(series))
    max_allowed = 0.45 * step
    mode_spacing = 0.0 if m <= 1 else min(float(mode_x_spacing), (2.0 * max_allowed) / float(m - 1))

    _color_of = color_of if color_of is not None else rk_color

    for i, rk in enumerate(series):
        color = _color_of(rk)
        mode_offset = (i - (m - 1) / 2.0) * mode_spacing

        for xi, a in enumerate(arrivals_order):
            x_center = x_base[xi] + mode_offset
            y0_list = _as_finite_list(y_of(rk, nodes_order[0], a, priorities))
            y1_list = _as_finite_list(y_of(rk, nodes_order[1], a, priorities))

            def plot_many(xs_center: float, ys: List[float], marker: str) -> None:
                if not ys:
                    return
                if len(ys) == 1:
                    xs = [xs_center]
                else:
                    j = max(0.001, float(seed_jitter_frac) * max(0.001, mode_spacing))
                    offsets = np.linspace(-j, j, len(ys))
                    xs = [xs_center + float(o) for o in offsets]

                for x, y in zip(xs, ys):
                    ax.plot(
                        [x],
                        [y],
                        marker=marker,
                        linestyle="None",
                        markersize=PLOT_MARKER_SIZE,
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markeredgewidth=PLOT_MARKER_LINEWIDTH,
                    )

            plot_many(x_center, y0_list, marker="o")
            plot_many(x_center, y1_list, marker="s")

    if ycfg.scale == "symlog":
        ax.set_yscale("symlog", base=SYMLOG_BASE, linthresh=ycfg.symlog_linthresh, linscale=SYMLOG_LINSCALE)
    else:
        ax.set_yscale("linear")

    ax.axhline(0.0, linewidth=0.8, color="black", linestyle="-", alpha=0.7)
    ax.set_ylim(float(ycfg.ylim[0]), float(ycfg.ylim[1]))

    if ycfg.scale == "linear":
        if y_tick_strategy == "count_sparse":
            set_count_yticks(ax, ycfg.ylim, max_ticks=5)
        elif y_tick_strategy == "count_sparse_symmetric":
            set_symmetric_count_yticks(ax, ycfg.ylim, max_ticks_total=7)
        else:
            set_linear_yticks(ax, ycfg.ylim)

    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.margins(x=0)

    for bx in boundaries:
        ax.axvline(bx, linewidth=0.8, color="black", linestyle="--", alpha=0.7, zorder=0)

    ax.set_xticks(boundaries)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE, pad=PLOT_TICK_PAD)

    if show_xticklabels:
        ax.set_xticklabels([""] * len(boundaries))
        for xi, a in enumerate(arrivals_order):
            ax.text(
                x_base[xi],
                -0.03,
                arrival_tick_label_with_axis(a, xi, len(arrivals_order)),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=PLOT_TICK_FONTSIZE,
                clip_on=False,
                linespacing=1.35 if xi == len(arrivals_order) // 2 else 1.0,
            )
    else:
        ax.tick_params(labelbottom=False)

    if not show_yticklabels:
        ax.tick_params(labelleft=False)

    for y in ax.get_yticks():
        if abs(y) < 1e-8:
            continue
        ax.axhline(y, linewidth=0.8, color="black", linestyle="--", alpha=0.15, zorder=0)


def values_from_df_seeds(
    df_seeds: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, rk: RowKey, col: str
) -> List[float]:
    sub = df_seeds[
        (df_seeds["nodes"] == int(nodes))
        & (df_seeds["priorities"] == int(priorities))
        & (df_seeds["arrival_s"] == float(arrival_s))
        & (df_seeds["mode"] == rk.mode)
        & (df_seeds["blocking"] == int(rk.blocking))
        & (df_seeds["defpreempt"] == int(rk.defpreempt))
    ][[SEED_COL, col]].sort_values(SEED_COL, kind="mergesort")

    vals = [float(v) for v in sub[col].tolist() if is_finite(v)]
    return vals


def _compute_scaled_ylim(max_val: float, symmetric: bool = False, scale_factor: float = 1.08) -> Tuple[float, float]:
    """Compute y-axis limits with scaling."""
    hi = max_val * scale_factor if max_val > 0 else 1.0
    return (-float(hi), float(hi)) if symmetric else (0.0, float(hi))


def compute_nonnegative_ylim_main(lookup_main: pd.DataFrame, *, series_all: List[RowKey], nodes_order: List[int], arrivals_order: List[float], priorities_cols: List[int], col: str) -> Tuple[float, float]:
    vals = []
    for rk in series_all:
        for n in nodes_order:
            for a in arrivals_order:
                for k in priorities_cols:
                    v = lookup_val(lookup_main, nodes=n, priorities=k, arrival_s=a, rk=rk, col=col)
                    if is_finite(v):
                        vals.append(v)
    return _compute_scaled_ylim(max(vals) if vals else 0.0, symmetric=False)


def compute_symmetric_ylim_deltas(df_delta_meanstd: pd.DataFrame, *, priorities_cols: List[int], col: str) -> Tuple[float, float]:
    sub = df_delta_meanstd[df_delta_meanstd["priorities"].isin(priorities_cols)]
    vals = [abs(float(v)) for v in sub[col].tolist() if is_finite(v)]
    return _compute_scaled_ylim(max(vals) if vals else 0.0, symmetric=True)


# =============================================================================
# Plots (main + deltas)
# =============================================================================

def _make_grid_unified(
    *,
    y_function_factory: Callable[[str], YOfFn],
    series: List[RowKey],
    color_override: Optional[Callable[[RowKey], Any]],
    legend_labels: List[str],
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
    ylim_solver: Tuple[float, float],
    ylim_plans: Tuple[float, float],
    out_stem: str,
    y_config: Dict[str, YAxisCfg],
    figsize: Tuple[float, float],
    grid_left: float,
    mode_x_spacing: float,
    legend_ncol: int,
    y_tick_symmetric: bool = False,
) -> None:
    """Unified grid plotting function used by both main and delta grids."""
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=figsize, sharex=True)
    axes[0, 0].set_title(f"#priorities = {priorities_cols[0]}", fontsize=PLOT_TITLE_FONTSIZE)
    axes[0, 1].set_title(f"#priorities = {priorities_cols[1]}", fontsize=PLOT_TITLE_FONTSIZE)

    # Build y-axis configs for solver and plans rows
    y_config_with_limits = dict(y_config)
    y_config_with_limits["solver"] = YAxisCfg("linear", ylim_solver, 1.0)
    y_config_with_limits["plans"] = YAxisCfg("linear", ylim_plans, 1.0)

    # Draw all rows using configuration
    for row_idx, row_spec in enumerate(GRID_ROW_SPECS):
        y_of = y_function_factory(row_spec.col_name)
        ycfg = y_config_with_limits[row_spec.ycfg_key]
        
        # Determine y_tick_strategy
        y_tick_strategy = row_spec.y_tick_strategy
        if y_tick_symmetric and y_tick_strategy == "count_sparse":
            y_tick_strategy = "count_sparse_symmetric"

        for col_i, k in enumerate(priorities_cols):
            draw_points_on_ax(
                ax=axes[row_idx, col_i],
                nodes_order=nodes_order,
                arrivals_order=arrivals_order,
                priorities=k,
                series=series,
                y_of=y_of,
                ycfg=ycfg,
                show_xticklabels=row_spec.is_bottom,
                show_yticklabels=(col_i == 0),
                arrival_x_spacing=PLOT_ARRIVAL_X_SPACING,
                mode_x_spacing=mode_x_spacing,
                y_tick_strategy=y_tick_strategy,
                color_of=color_override,
            )

    fig.subplots_adjust(
        left=grid_left,
        right=GRID_RIGHT,
        bottom=GRID_BOTTOM,
        top=GRID_TOP,
        wspace=GRID_WSPACE,
        hspace=GRID_HSPACE,
    )

    # Legend
    if color_override:
        # Custom colors for delta plots
        legend_handles = [Line2D([0], [0], color=color_override(rk), linewidth=1.8) for rk in series]
    else:
        # Standard colors from RowKey
        legend_handles = [Line2D([0], [0], color=rk_color(rk), linewidth=1.8) for rk in series]

    bbox_l = axes[0, 0].get_position()
    bbox_r = axes[0, 1].get_position()
    x_center_grid = 0.5 * (bbox_l.x0 + bbox_r.x1)
    y_top_grid = max(bbox_l.y1, bbox_r.y1)
    legend_y = min(0.98, y_top_grid + float(GRID_LEGEND_PAD))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(x_center_grid, legend_y),
        ncol=min(legend_ncol, len(legend_labels)),
        fontsize=PLOT_LEGEND_FONTSIZE,
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
    )

    # Row labels (left)
    x_text = x_from_left_with_pad_points(fig, grid_left, GRID_YLABEL_PAD_PT)
    for r, row_spec in enumerate(GRID_ROW_SPECS):
        bbox = axes[r, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(x_text, y_center, row_spec.ylabel, rotation=90, va="center", ha="right", fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    fig.savefig(OUT_FIGURES_DIR / f"{out_stem}.png", dpi=PLOT_FIGURE_DPI)
    fig.savefig(OUT_FIGURES_DIR / f"{out_stem}.pdf")
    plt.close(fig)


def make_grid_main(
    *,
    df_seeds: pd.DataFrame,
    lookup_main: pd.DataFrame,
    plot_seeds: bool,
    defpreempt: int,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
    ylim_solver: Tuple[float, float],
    ylim_plans: Tuple[float, float],
    out_stem: str,
) -> None:
    series = sort_rks([RowKey(mode=m, blocking=b, defpreempt=defpreempt) for (m, b) in MAIN_PLOT_MODES])

    def y_function_factory(col: str) -> YOfFn:
        if plot_seeds:
            return lambda rk, n, a, k: values_from_df_seeds(df_seeds, nodes=n, priorities=k, arrival_s=a, rk=rk, col=col)
        return lambda rk, n, a, k: lookup_val(lookup_main, nodes=n, priorities=k, arrival_s=a, rk=rk, col=col)

    legend_labels = [rk_label(rk) for rk in series]

    _make_grid_unified(
        y_function_factory=y_function_factory,
        series=series,
        color_override=None,
        legend_labels=legend_labels,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_cols=priorities_cols,
        ylim_solver=ylim_solver,
        ylim_plans=ylim_plans,
        out_stem=out_stem,
        y_config=Y_MAIN,
        figsize=GRID_FIGSIZE_MAIN,
        grid_left=GRID_LEFT_MAIN,
        mode_x_spacing=PLOT_MODE_X_SPACING_MAIN,
        legend_ncol=GRID_LEGEND_NCOL_MAIN,
        y_tick_symmetric=False,
    )


def make_grid_periodic_vs_stable(
    *,
    df_delta_seeds: pd.DataFrame,
    lookup_deltas: pd.DataFrame,
    plot_seeds: bool,
    defpreempt: int,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
    ylim_solver: Tuple[float, float],
    ylim_plans: Tuple[float, float],
    out_stem: str,
) -> None:
    cmap = plt.get_cmap("Set2").colors

    # Create fake series for coloring
    fake_series = [RowKey(mode=f"custom{i}", blocking=0, defpreempt=defpreempt) for i in range(len(DELTA_NAMES))]

    def _extract_custom_index(rk: RowKey) -> int:
        """Extract index from custom RowKey mode."""
        return int(rk.mode.replace("custom", "")) if rk.mode.startswith("custom") else 0

    def color_override(rk: RowKey) -> Any:
        return cmap[_extract_custom_index(rk) % len(cmap)]

    def y_function_factory(col: str) -> YOfFn:
        """Creates y-value function for deltas that dispatches based on RowKey."""
        # Build individual delta functions with cleaner closure handling
        def make_delta_fn(delta_name: str):
            if plot_seeds:
                def _y(_rk: RowKey, nodes: int, a: float, priorities: int) -> List[float]:
                    sub = df_delta_seeds[
                        (df_delta_seeds["defpreempt"] == defpreempt) &
                        (df_delta_seeds["nodes"] == nodes) &
                        (df_delta_seeds["priorities"] == priorities) &
                        (df_delta_seeds["arrival_s"] == a) &
                        (df_delta_seeds["delta_name"] == delta_name)
                    ][[SEED_COL, col]].sort_values(SEED_COL, kind="mergesort")
                    return [float(v) for v in sub[col].tolist() if is_finite(v)]
            else:
                def _y(_rk: RowKey, nodes: int, a: float, priorities: int) -> float:
                    m, _s = lookup_delta_mean_std(lookup_deltas, nodes=nodes, priorities=priorities, arrival_s=a, defpreempt=defpreempt, delta_name=delta_name, col_mean=col)
                    return float(m)
            return _y
        
        delta_fns = {i: make_delta_fn(dn) for i, dn in enumerate(DELTA_NAMES)}
        return lambda rk, nodes, a, priorities: delta_fns[_extract_custom_index(rk)](rk, nodes, a, priorities)

    _make_grid_unified(
        y_function_factory=y_function_factory,
        series=fake_series,
        color_override=color_override,
        legend_labels=DELTA_NAMES,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_cols=priorities_cols,
        ylim_solver=ylim_solver,
        ylim_plans=ylim_plans,
        out_stem=out_stem,
        y_config=Y_DELTAS,
        figsize=GRID_FIGSIZE_DELTAS,
        grid_left=GRID_LEFT_DELTAS,
        mode_x_spacing=PLOT_MODE_X_SPACING_DELTAS,
        legend_ncol=GRID_LEGEND_NCOL_DELTAS,
        y_tick_symmetric=True,
    )


# =============================================================================
# Tables (LaTeX)
# =============================================================================

@dataclass(frozen=True)
class MetricSpec:
    col_mean: str
    latex_label: str
    mean_signed: bool
    mean_dec: int
    std_dec: int
    kind: str  # "signed" | "unsigned_int"

    def format_cell(self, mean_val: object, std_val: object, include_std: bool) -> str:
        """Format a table cell for this metric."""
        if include_std:
            if self.kind == "unsigned_int":
                return fmt_pm(mean_val, std_val, mean_signed=False, mean_dec=self.mean_dec, std_dec=self.std_dec)
            else:
                return fmt_pm(mean_val, std_val, mean_signed=self.mean_signed, mean_dec=self.mean_dec, std_dec=self.std_dec)
        else:
            if self.kind == "unsigned_int":
                return fmt_unsigned_int(mean_val)
            else:
                return fmt_signed(mean_val, self.mean_dec)


# Metric specs used for periodic vs stable delta tables
METRICS_DELTAS: List[MetricSpec] = [
    MetricSpec("delta_U_pct_eff_mean", r"$\Delta\ \mathrm{usage}\;(\%)$", True, 2, 2, "signed"),
    MetricSpec("delta_L_ms_total_mean", r"$\Delta\ \mathrm{latency}_{\mathrm{total}}\;(\mathrm{ms})$", True, 0, 0, "signed"),
    MetricSpec("delta_D_num_total_mean", r"$\Delta\ \mathrm{deletions}_{\mathrm{total}}$", True, 1, 1, "signed"),
    MetricSpec("solver_attempts_mean", r"$\Delta\ \mathrm{solver\ runs}$", True, 1, 1, "signed"),
    MetricSpec("plan_activated_mean", r"$\Delta\ \mathrm{plan\ activations}$", True, 1, 1, "signed"),
]


def metrics_main(priorities: int) -> List[MetricSpec]:
    out: List[MetricSpec] = [
        MetricSpec("delta_U_pct_eff_mean", r"$\Delta\mathrm{usage}\;(\%)$", True, 2, 2, "signed"),
        MetricSpec("delta_L_ms_total_mean", r"$\Delta\mathrm{latency}_{\mathrm{total}}\;(\mathrm{ms})$", True, 0, 0, "signed"),
    ]

    if priorities != 1:
        for p in (1, 2, 3, 4):
            out.append(MetricSpec(f"delta_L_ms_p{p}_mean", rf"$\Delta\mathrm{{latency}}_{{p{p}}}\;(\mathrm{{ms}})$", True, 0, 0, "signed"))

    out.append(MetricSpec("delta_D_num_total_mean", r"$\Delta\mathrm{deletions}_{\mathrm{total}}$", True, 1, 1, "signed"))

    if priorities != 1:
        for p in (1, 2, 3, 4):
            out.append(MetricSpec(f"delta_D_num_p{p}_mean", rf"$\Delta\mathrm{{deletions}}_{{p{p}}}$", True, 1, 1, "signed"))

    out.append(MetricSpec("solver_attempts_mean", r"\#solver\\runs", False, 0, 1, "unsigned_int"))
    out.append(MetricSpec("plan_activated_mean", r"\#plan\\activations", False, 0, 1, "unsigned_int"))
    return out


def latex_table_main(
    *,
    out_path: Path,
    lookup_main: pd.DataFrame,
    defpreempt: int,
    priorities: int,
    std: int,
    nodes_order: List[int],
    arrivals_order: List[float],
) -> None:
    modes = sort_rks([RowKey(mode=s.mode, blocking=int(s.blocking), defpreempt=int(defpreempt)) for s in MODE_SPECS])
    specs = metrics_main(priorities)

    lines: List[str] = []
    colspec = "l " + " ".join(["c"] * len(specs))
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join([metric_header_tex(s.latex_label) for s in specs]) + r" \\")
    lines.append(r"\midrule")

    total_cols = 1 + len(specs)

    first_block = True
    for n in nodes_order:
        for a in arrivals_order:
            if not first_block:
                lines.append(r"\midrule")
            first_block = False

            a_str = fmt_arrival_value(a)
            lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\#nodes = {int(n)}, inter-arrival = {a_str}\,s}} \\")
            lines.append(r"\midrule")

            for rk in modes:
                cells = [s.format_cell(*(lookup_mean_std(lookup_main, nodes=n, priorities=priorities, arrival_s=a, rk=rk, col_mean=s.col_mean) if std else (lookup_val(lookup_main, nodes=n, priorities=priorities, arrival_s=a, rk=rk, col=s.col_mean), None)), include_std=bool(std)) for s in specs]
                lines.append(f"{rk_label_tex(rk)} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def latex_table_periodic_vs_stable(
    *,
    out_path: Path,
    lookup_deltas: pd.DataFrame,
    priorities: int,
    std: int,
    nodes_order: List[int],
    arrivals_order: List[float],
) -> None:
    # Two delta columns per arrival
    n_modes = len(DELTA_NAMES)
    n_arr = len(arrivals_order)
    total_cols = 1 + n_modes * n_arr
    tab_spec = "l" + " c" * (total_cols - 1)

    cmid = []
    for i in range(n_arr):
        start = 2 + i * n_modes
        end = start + n_modes - 1
        cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")

    tex_lines: List[str] = []

    for defpreempt in (1, 0):
        tex_lines.append(r"% ------------------------------------------------------------")
        tex_lines.append(rf"% periodic vs stable, defpreempt={defpreempt}")
        tex_lines.append(r"% ------------------------------------------------------------")
        tex_lines.append("")
        tex_lines.append(rf"\begin{{tabular}}{{{tab_spec}}}")
        tex_lines.append(r"\toprule")
        tex_lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{Mode $\Delta$ (PR 8$\rightarrow$32, SQ 2$\rightarrow$8), \#priorities = {priorities}, defaultpreemption = {defpreempt}}} \\")
        tex_lines.append(r"\addlinespace[0.2em]")
        tex_lines.append(" & " + " & ".join([rf"\multicolumn{{{n_modes}}}{{c}}{{inter-arrival = {fmt_arrival_value(a)}\,s}}" for a in arrivals_order]) + r" \\")
        tex_lines.append("".join(cmid))
        tex_lines.append(" & " + " & ".join([c for _a in arrivals_order for c in DELTA_NAMES]) + r" \\")
        tex_lines.append(r"\midrule")

        for ni, n in enumerate(nodes_order):
            if ni > 0:
                tex_lines.append(r"\midrule")
            tex_lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\#nodes = {n}}} \\")
            tex_lines.append(r"\midrule")

            for ms in METRICS_DELTAS:
                cells = [ms.format_cell(*lookup_delta_mean_std(lookup_deltas, nodes=n, priorities=priorities, arrival_s=float(a), defpreempt=int(defpreempt), delta_name=str(name), col_mean=ms.col_mean), include_std=(std == 1)) for a in arrivals_order for name in DELTA_NAMES]
                tex_lines.append(f"{ms.latex_label} & " + " & ".join(cells) + r" \\")

        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append("")
        tex_lines.append("")

    out_path.write_text("\n".join(tex_lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_seeds = load_results_seeds(IN_RESULTS_SEEDS)
    if df_seeds.empty:
        raise SystemExit("results_seeds.csv is empty")

    df_meanstd = aggregate_mean_std(df_seeds)
    lookup_main = build_lookup(df_meanstd)

    nodes_order = sorted(df_meanstd["nodes"].unique().tolist())
    arrivals_order_all = df_meanstd["arrival_s"].unique().tolist()
    arrivals_order = select_arrivals_order(arrivals_order_all)
    priorities_order = sorted(df_meanstd["priorities"].unique().tolist())

    if len(nodes_order) != 2:
        raise SystemExit(f"Expected exactly 2 node values for plots; found {nodes_order}")
    if any(k not in priorities_order for k in PRIORITIES_TO_SHOW):
        raise SystemExit(f"Expected priorities {PRIORITIES_TO_SHOW}; found {priorities_order}")

    # Build delta datasets (seed-level + mean/std)
    df_delta_seeds = build_delta_seeds(df_seeds)
    df_delta_meanstd = aggregate_delta_mean_std(df_delta_seeds) if not df_delta_seeds.empty else pd.DataFrame(columns=DELTA_KEY_COLS)
    lookup_deltas = build_lookup_deltas(df_delta_meanstd) if not df_delta_meanstd.empty else pd.DataFrame().set_index(DELTA_KEY_COLS)

    # Shared y-lims (main counters) across defpreempt=0/1
    series_all_for_limits = sort_rks([RowKey(mode=m, blocking=b, defpreempt=d) for d in (0, 1) for (m, b) in MAIN_PLOT_MODES])
    ylim_solver_main = compute_nonnegative_ylim_main(lookup_main, series_all=series_all_for_limits, nodes_order=nodes_order, arrivals_order=arrivals_order, priorities_cols=PRIORITIES_TO_SHOW, col="solver_attempts_mean")
    ylim_plans_main = compute_nonnegative_ylim_main(lookup_main, series_all=series_all_for_limits, nodes_order=nodes_order, arrivals_order=arrivals_order, priorities_cols=PRIORITIES_TO_SHOW, col="plan_activated_mean")

    # Shared y-lims (delta counters) across defpreempt=0/1 (symmetric)
    ylim_solver_deltas = compute_symmetric_ylim_deltas(df_delta_meanstd, priorities_cols=PRIORITIES_TO_SHOW, col="solver_attempts_mean") if not df_delta_meanstd.empty else (-1.0, 1.0)
    ylim_plans_deltas = compute_symmetric_ylim_deltas(df_delta_meanstd, priorities_cols=PRIORITIES_TO_SHOW, col="plan_activated_mean") if not df_delta_meanstd.empty else (-1.0, 1.0)

    produced_tables: List[Path] = []
    produced_figs: List[Path] = []

    # Generate all tables
    for defpreempt in (1, 0):
        for k in PRIORITIES_TO_SHOW:
            for std in (0, 1):
                out_tex = OUT_TABLES_DIR / f"table_main_defaultpreemption={defpreempt}_priorities={k}_std={std}.tex"
                latex_table_main(out_path=out_tex, lookup_main=lookup_main, defpreempt=defpreempt, priorities=k, std=std, nodes_order=nodes_order, arrivals_order=arrivals_order)
                produced_tables.append(out_tex)
    
    for k in PRIORITIES_TO_SHOW:
        for std in (0, 1):
            out_tex = OUT_TABLES_DIR / f"table_periodic_vs_stable_priorities={k}_std={std}.tex"
            latex_table_periodic_vs_stable(out_path=out_tex, lookup_deltas=lookup_deltas, priorities=k, std=std, nodes_order=nodes_order, arrivals_order=arrivals_order)
            produced_tables.append(out_tex)

    # Generate all figures
    for defpreempt in (1, 0):
        for plot_seeds in (False, True):
            seeds_flag = 1 if plot_seeds else 0
            out_stem = f"grid_main_defaultpreemption={defpreempt}_seeds={seeds_flag}"
            make_grid_main(df_seeds=df_seeds, lookup_main=lookup_main, plot_seeds=plot_seeds, defpreempt=defpreempt, nodes_order=nodes_order, arrivals_order=arrivals_order, priorities_cols=PRIORITIES_TO_SHOW, ylim_solver=ylim_solver_main, ylim_plans=ylim_plans_main, out_stem=out_stem)
            produced_figs.extend([OUT_FIGURES_DIR / f"{out_stem}.png", OUT_FIGURES_DIR / f"{out_stem}.pdf"])
    
    for defpreempt in (1, 0):
        for plot_seeds in (False, True):
            seeds_flag = 1 if plot_seeds else 0
            out_stem = f"grid_periodic_vs_stable_defaultpreemption={defpreempt}_seeds={seeds_flag}"
            make_grid_periodic_vs_stable(df_delta_seeds=df_delta_seeds, lookup_deltas=lookup_deltas, plot_seeds=plot_seeds, defpreempt=defpreempt, nodes_order=nodes_order, arrivals_order=arrivals_order, priorities_cols=PRIORITIES_TO_SHOW, ylim_solver=ylim_solver_deltas, ylim_plans=ylim_plans_deltas, out_stem=out_stem)
            produced_figs.extend([OUT_FIGURES_DIR / f"{out_stem}.png", OUT_FIGURES_DIR / f"{out_stem}.pdf"])

    # Summary
    for label, paths in [("Tables", produced_tables), ("Figures", produced_figs)]:
        print(f"{label}:\n" + "\n".join(f"  - {p}" for p in paths))

if __name__ == "__main__":
    main()
