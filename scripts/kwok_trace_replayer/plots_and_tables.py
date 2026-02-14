#!/usr/bin/env python3
# plots_and_tables.py
"""
python -m scripts.kwok_trace_replayer.plots_and_tables
"""

import math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.helpers.plot_config import (
    PLOT_AXIS_LABEL_FONTSIZE,
    PLOT_FIGURE_DPI,
    PLOT_LEGEND_COLUMN_SPACING,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_HANDLE_LENGTH,
    PLOT_LEGEND_HANDLE_TEXT_PAD,
    PLOT_TICK_FONTSIZE,
    PLOT_TITLE_FONTSIZE,
    PLOT_FORMATS,
)
from scripts.helpers.data_helpers import is_finite
from scripts.helpers.table_helpers import (
    fmt_mean_std,
    latex_cmidrules,
    write_latex_table,
)
from scripts.helpers.plot_helpers import PLOT_COLORS, configure_matplotlib, save_figure

# =============================================================================
# CONFIG
# =============================================================================

IN_RESULTS_SEEDS = Path("analysis/kwok_trace_replayer/results_seeds.csv")

OUT_DIR = Path("analysis/kwok_trace_replayer")
OUT_TABLES_DIR = OUT_DIR / "tables"
OUT_FIGURES_DIR = OUT_DIR / "figures"

MAX_PRIORITIES = 4

SEED_COL = "seed"

# Base modes — each is auto-expanded into blocking=1 and blocking=0 variants.
# (mode, base_label, detail, rank_base, color_idx)
BASE_MODES: List[Tuple[str, str, str, int, int]] = [
    ("schedulingfailure",  "Scheduling-failure", "",             0,  1),
    ("periodic4s",         "Periodic",           "4s interval",  2,  5),
    ("periodic8s",         "Periodic",           "8s interval",  4,  6),
    ("periodic16s",        "Periodic",           "16s interval", 6,  7),
    ("stable-queue-4s",    "Stable-queue",       "4s delay",     8,  9),
    ("stable-queue-8s",    "Stable-queue",       "8s delay",     10, 10),
    ("stable-queue-16s",   "Stable-queue",       "16s delay",    12, 11),
]

PRIORITIES_TO_SHOW = [1, 4]
INTER_ARRIVALS_TO_SHOW = [2.0, 4.0, 8.0, 16.0] # TODO: for thesis add 1.0 and 32.0

# Fixed y-axis limits for solver runs and plan activations (None = auto-compute)
YLIM_SOLVER_MAIN: Optional[Tuple[float, float]] = (0.0, 400.0)
YLIM_PLANS_MAIN: Optional[Tuple[float, float]] = (0.0, 300.0)
YLIM_SOLVER_DELTAS: Optional[Tuple[float, float]] = (-150.0, 150.0)
YLIM_PLANS_DELTAS: Optional[Tuple[float, float]] = (-60.0, 60.0)

# Y-axis configurations for main and delta grids
Y_MAIN: Dict[str, "YAxisCfg"] = {}   # populated after YAxisCfg is defined
Y_DELTAS: Dict[str, "YAxisCfg"] = {} # populated after YAxisCfg is defined

TABLE_DECIMALS = 1

SEED_COLS_NEEDED = [
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

KEY_COLS_MAIN = ["nodes", "priorities", "arrival_s", "mode", "blocking", "defpreempt"]

# Main grids show only a subset of modes (tables include ALL modes)
MAIN_PLOT_MODE_NAMES: List[str] = ["schedulingfailure", "periodic8s", "stable-queue-8s"]

# Delta comparisons (baseline → compared) with their plot color index
@dataclass(frozen=True)
class DeltaSpec:
    label: str
    baseline: str
    compared: str
    color_idx: int

DELTA_SPECS: List[DeltaSpec] = [
    DeltaSpec("Periodic, 8→4s interval",       "periodic8s",       "periodic4s",       4),
    DeltaSpec("Periodic, 8→16s interval",      "periodic8s",       "periodic16s",      7),
    DeltaSpec("Stable-queue, 8→4s delay",      "stable-queue-8s",  "stable-queue-4s",  8),
    DeltaSpec("Stable-queue, 8→16s delay",     "stable-queue-8s",  "stable-queue-16s", 11),
]

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
PLOT_MARKER_SIZE = 2.0
PLOT_MARKER_SEED_ALPHA = 0.5
PLOT_MEAN_MARKER_SIZE = 3.5
PLOT_MARKER_LINEWIDTH = 0.4

# Shape legend: (marker, label_template, marker_size, facecolor)
#   label_template may contain "{nodes}" which will be replaced with the actual node count.
#   Set facecolor to "none" for outline-only markers, or any color string to fill.
SHAPE_LEGEND_SPECS: List[Tuple[str, str, float, str]] = [
    ("o", "run with {nodes} nodes", 3.5, "none"),  # nodes_order[0]
    ("s", "run with {nodes} nodes", 3.4, "none"),  # nodes_order[1]
    ("D", "avg. for all runs", 3.6, "none"),       # mean
]
SHAPE_LEGEND_EDGE_WIDTH = 0.6
PLOT_MIN_LINEAR_YTICKS = 5
PLOT_COUNT_MAX_TICKS = 7
PLOT_COUNT_MAX_TICKS_SYMMETRIC = 8

PLOT_HEIGHT = 7.0

GRID_FIGSIZE_MAIN = (3.2, PLOT_HEIGHT)
GRID_FIGSIZE_DELTAS = (3.6, PLOT_HEIGHT)

GRID_LEGEND_NCOL_COLORS = 1
GRID_LEGEND_NCOL_SHAPES = 1

GRID_LEGEND_GAP = 0.01       # horizontal gap between the two legend boxes (figure fraction)
GRID_LEGEND_X_OFFSET_MAIN: Dict[int, float] = {
    0: -0.03,  # non-blocking
    1: -0.02,  # blocking
}
GRID_LEGEND_X_OFFSET_DELTAS: Dict[int, float] = {
    0: -0.012,   # non-blocking
    1: -0.012,   # blocking
}
GRID_LEGEND_PAD_MAIN = 0.113
GRID_LEGEND_PAD_DELTAS = 0.13
GRID_LEFT_MAIN = 0.14
GRID_LEFT_DELTAS = 0.125
GRID_RIGHT = 0.99
GRID_BOTTOM = 0.03
GRID_TOP_MAIN = 0.89
GRID_TOP_DELTAS = 0.873
GRID_WSPACE = 0.10
GRID_HSPACE = 0.10
GRID_YLABEL_PAD_PT = 23.0

PLOT_ARRIVAL_X_SPACING = 0.35
PLOT_MODE_X_SPACING_MAIN = 0.09
PLOT_MODE_X_SPACING_DELTAS = 0.08



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
    for token in str(plugin_config).split("_"):
        if "=" in token:
            key, value = token.split("=", 1)
            kv[key.strip().lower()] = value.strip()
    return kv

def canonical_mode(mode_match: str) -> str:
    s = str(mode_match).strip().lower()
    mode_match = re.fullmatch(r"periodic-?([0-9.]+)s", s)
    if mode_match:
        return f"periodic{mode_match.group(1)}s"
    mode_match = re.fullmatch(r"(?:stable-queue|stablequeue)-?([0-9.]+)s", s)
    if mode_match:
        return f"stable-queue-{mode_match.group(1)}s"
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
# Modes
# =============================================================================

@dataclass(frozen=True)
class ModeSpec:
    mode: str
    blocking: int
    label: str
    base_label: str
    detail: str
    rank: int
    color_idx: int  # Index into PLOT_COLORS

def _expand_mode_specs() -> List[ModeSpec]:
    specs: List[ModeSpec] = []
    for mode, base_label, detail, rank_base, color_idx in BASE_MODES:
        for blocking in (1, 0):
            b_str = "blocking" if blocking else "non-blocking"
            label = f"{base_label} ({b_str})"
            if detail:
                label += f", {detail}"
            rank = rank_base + (0 if blocking else 1)
            specs.append(ModeSpec(mode=mode, blocking=blocking, label=label, base_label=base_label, detail=detail, rank=rank, color_idx=color_idx))
    return specs

MODE_SPECS: List[ModeSpec] = _expand_mode_specs()

_SPEC_BY_MODE_BLOCK: Dict[Tuple[str, int], ModeSpec] = {(s.mode, int(s.blocking)): s for s in MODE_SPECS}

def row_key_rank(row_key: RowKey) -> int:
    mode_spec = _SPEC_BY_MODE_BLOCK.get((row_key.mode, int(row_key.blocking)))
    return int(mode_spec.rank) if mode_spec else 10_000

def row_key_label(row_key: RowKey) -> str:
    mode_spec = _SPEC_BY_MODE_BLOCK.get((row_key.mode, int(row_key.blocking)))
    return mode_spec.label if mode_spec else f"{row_key.mode}:{row_key.blocking}"

def row_key_table_label(row_key: RowKey, *, multiline: bool = True) -> str:
    """LaTeX label for tables. Multi-line when multiline=True, single-line otherwise."""
    mode_spec = _SPEC_BY_MODE_BLOCK.get((row_key.mode, int(row_key.blocking)))
    if not mode_spec:
        return f"{row_key.mode}:{row_key.blocking}"
    if not multiline:
        return mode_spec.label
    b_str = "blocking" if mode_spec.blocking else "non-blocking"
    parts = [mode_spec.base_label, f"({b_str})"]
    if mode_spec.detail:
        parts.append(mode_spec.detail)
    inner = r"\\".join(parts)
    return rf"\makebox[6.5em][c]{{\begin{{tabular}}[t]{{@{{}}c@{{}}}}{inner}\end{{tabular}}}}"

def row_key_color(row_key: RowKey):
    mode_spec = _SPEC_BY_MODE_BLOCK.get((row_key.mode, int(row_key.blocking)))
    if not mode_spec:
        return PLOT_COLORS[0]
    return PLOT_COLORS[int(mode_spec.color_idx) % len(PLOT_COLORS)]

def sort_row_keys(row_keys: Iterable[RowKey]) -> List[RowKey]:
    return sorted(set(row_keys), key=lambda r: (row_key_rank(r), r.mode, int(r.blocking), int(r.defpreempt)))

# =============================================================================
# Plot selection
# =============================================================================

def select_arrivals_order(all_arrivals: List[float]) -> List[float]:
    all_sorted = sorted(set(float(a) for a in all_arrivals))
    if INTER_ARRIVALS_TO_SHOW is None:
        return all_sorted
    wanted = [float(a) for a in INTER_ARRIVALS_TO_SHOW]
    wanted_set = set(wanted)
    missing = [a for a in wanted if a not in set(all_sorted)]
    if missing:
        raise SystemExit(f"INTER_ARRIVALS_TO_SHOW contains values not in data: {missing}. Available: {all_sorted}")
    # Preserve the order provided in INTER_ARRIVALS_TO_SHOW
    return [a for a in wanted if a in wanted_set]

@dataclass(frozen=True)
class YAxisCfg:
    """
    Configuration for a Y axis.
    """
    scale: str  # "linear" | "symlog"
    y_lim: Tuple[float, float]
    symlog_linthresh: float = 1.0

# Populate Y_MAIN and Y_DELTAS now that YAxisCfg is defined
Y_MAIN.update({
    "util": YAxisCfg("linear", (-2.0, 5.0)),
    "latency": YAxisCfg("symlog", (-1e5 - 1.0, 1e5 + 1.0), 1.0),
    "deletions": YAxisCfg("symlog", (-1e4 - 1.0, 1e4 + 1.0), 1.0),
})

Y_DELTAS.update({
    "util": YAxisCfg("linear", (-0.5, 0.5)),
    "latency": YAxisCfg("symlog", (-1e4 - 1.0, 1e4 + 1.0), 1.0),
    "deletions": YAxisCfg("symlog", (-1e3 - 1.0, 1e3 + 1.0), 1.0),
})

# =============================================================================
# Grid row configuration
# =============================================================================

@dataclass(frozen=True)
class GridRowSpec:
    """
    Configuration for one row in a grid plot.
    """
    col_name: str
    ycfg_key: str # key in Y_MAIN or Y_DELTAS
    y_tick_strategy: Optional[str] = None
    y_label: str = ""
    is_bottom: bool = False # whether this is the bottom row

# Row configurations - reused for both main and delta grids
GRID_ROW_SPECS = [
    GridRowSpec("delta_U_pct_eff_mean", "util", y_label="diff. usage (%)"),
    GridRowSpec("delta_L_ms_total_mean", "latency", y_label="diff. latency (ms)"),
    GridRowSpec("delta_D_num_total_mean", "deletions", y_label="diff. deletions"),
    GridRowSpec("solver_attempts_mean", "solver", y_tick_strategy="count_sparse", y_label="optimizer runs"),
    GridRowSpec("plan_activated_mean", "plans", y_tick_strategy="count_sparse", y_label="plan activations", is_bottom=True),
]

# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_arrival_value(a: float) -> Union[int, float]:
    """
    Convert arrival to int if close to integer, otherwise keep as float.
    """
    return int(a) if abs(a - round(a)) < 1e-9 else a

# =============================================================================
# Data loading + aggregation
# =============================================================================

def load_results_seeds(path: Path) -> pd.DataFrame:
    """
    Load per-seed results CSV into a DataFrame, parsing job_name and plugin_config.
    """
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in SEED_COLS_NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing required columns: {', '.join(missing)}")

    # Parse job_name
    parsed = df["job_name"].map(parse_job)
    df[["nodes", "priorities", "arrival_s"]] = pd.DataFrame(parsed.tolist(), index=df.index, columns=["nodes", "priorities", "arrival_s"])

    # Parse plugin_config into mode/blocking/defpreempt
    row_keys = df["plugin_config"].map(RowKey.from_plugin_config)
    df[["mode", "blocking", "defpreempt"]] = pd.DataFrame([(r.mode, r.blocking, r.defpreempt) for r in row_keys], index=df.index, columns=["mode", "blocking", "defpreempt"])

    return df

def _aggregate_mean_std(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Compute mean+std across seeds for numeric columns.
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
    Aggregate per-seed DataFrame to mean+std per configuration.
    """
    group_cols = ["job_name", "plugin_config"] + KEY_COLS_MAIN
    return _aggregate_mean_std(df_seeds, group_cols)

def _build_lookup_from_df(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    Build a lookup DataFrame indexed by key columns for fast access.
    """
    return df.drop_duplicates(subset=key_cols, keep="first").set_index(key_cols).sort_index()

def build_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lookup DataFrame from the main aggregated DataFrame.
    """
    return _build_lookup_from_df(df, KEY_COLS_MAIN)

def _safe_lookup(lookup: pd.DataFrame, key: Tuple, col: str) -> float:
    """
    Safely lookup a value from a DataFrame index, returning NaN if not found.
    """
    try:
        return float(lookup.at[key, col])
    except KeyError:
        return float("nan")

def lookup_val(lookup: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, row_key: RowKey, col: str) -> float:
    """
    Lookup a value from the lookup DataFrame.
    """
    key = (int(nodes), int(priorities), float(arrival_s), str(row_key.mode), int(row_key.blocking), int(row_key.defpreempt))
    return _safe_lookup(lookup, key, col)

# =============================================================================
# Delta data (periodic vs stable) - computed from seed rows
# =============================================================================

_DELTA_KEY_COLS = ["nodes", "priorities", "arrival_s", "defpreempt", "blocking", "delta_name"]

def build_delta_seeds(df_seeds: pd.DataFrame) -> pd.DataFrame:
    """
    For each delta series (left->right) and each defpreempt value, compute per-seed deltas.
    """
    out_rows: List[pd.DataFrame] = []

    base_cols = ["nodes", "priorities", "arrival_s", "defpreempt", SEED_COL]

    for blocking in (0, 1):
        for ds in DELTA_SPECS:
            left = df_seeds[(df_seeds["mode"] == canonical_mode(ds.baseline)) & (df_seeds["blocking"] == int(blocking))].copy()
            right = df_seeds[(df_seeds["mode"] == canonical_mode(ds.compared)) & (df_seeds["blocking"] == int(blocking))].copy()

            left = left[base_cols + DELTA_METRIC_COLS].rename(columns={c: f"{c}_L" for c in DELTA_METRIC_COLS})
            right = right[base_cols + DELTA_METRIC_COLS].rename(columns={c: f"{c}_R" for c in DELTA_METRIC_COLS})

            merged = right.merge(left, on=base_cols, how="inner")
            if merged.empty:
                continue

            d = merged[base_cols].copy()
            d["blocking"] = int(blocking)
            d["delta_name"] = ds.label
            for c in DELTA_METRIC_COLS:
                d[c] = merged[f"{c}_R"] - merged[f"{c}_L"]

            out_rows.append(d)

    if not out_rows:
        return pd.DataFrame(columns=_DELTA_KEY_COLS + [SEED_COL])

    return pd.concat(out_rows, ignore_index=True)

def aggregate_delta_mean_std(df_delta_seeds: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate delta-by-seed to mean+std per configuration and delta_name.
    Adds <col>_std columns.
    """
    return _aggregate_mean_std(df_delta_seeds, _DELTA_KEY_COLS)

def build_lookup_deltas(df_delta_meanstd: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lookup DataFrame from the delta aggregated DataFrame.
    """
    return _build_lookup_from_df(df_delta_meanstd, _DELTA_KEY_COLS)

def lookup_delta_mean_std(lookup: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, defpreempt: int, blocking: int, delta_name: str, col_mean: str) -> Tuple[float, float]:
    """
    Lookup delta mean and std from delta lookup DataFrame.
    """
    key = (int(nodes), int(priorities), float(arrival_s), int(defpreempt), int(blocking), str(delta_name))
    return _safe_lookup(lookup, key, col_mean), _safe_lookup(lookup, key, f"{col_mean}_std")

# =============================================================================
# Plot helpers
# =============================================================================

YVal = Union[float, Sequence[float]] # scalar or list (seeds)
YOfFn = Callable[[RowKey, int, float, int], YVal]  # scalar or list (seeds)

def to_finite_list(v: object) -> List[float]:
    """
    Convert v to a list of floats (including non-finite values).
    """
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

def compute_step(span: float, target_ticks: int) -> float:
    """
    Compute a step size for ticks, given the span and target number of ticks
    """
    if span <= 0:
        return 1.0
    raw = span / max(1, int(target_ticks))
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10**exp
    candidates = [1 * base, 2 * base, 5 * base, 10 * base]
    return min(candidates, key=lambda s: abs(s - raw))

def set_linear_yticks(ax: plt.Axes, ylim: Tuple[float, float], *, min_ticks: int = PLOT_MIN_LINEAR_YTICKS) -> None:
    """
    Set y-ticks for linear scale, with a minimum number of ticks.
    """
    y_lo, y_hi = float(ylim[0]), float(ylim[1])
    y0 = int(math.ceil(y_lo))
    y1 = int(math.floor(y_hi))
    ticks = list(range(y0, y1 + 1))
    if len(ticks) < int(min_ticks):
        ticks = [round(float(t), 2) for t in np.linspace(y_lo, y_hi, int(min_ticks))]
    ax.set_yticks(ticks)

def set_count_yticks(ax: plt.Axes, ylim: Tuple[float, float], *, max_ticks: int = 5) -> None:
    """
    Set y-ticks for count data (non-negative), with a maximum number of ticks.
    0 is always included.
    """
    y_lo, y_hi = float(ylim[0]), float(ylim[1])
    y_lo = max(0.0, y_lo)
    if y_hi <= 0:
        ax.set_yticks([0])
        return
    step = max(1.0, compute_step(y_hi - y_lo, max_ticks - 1))
    ticks: List[float] = []
    t = 0.0
    for _ in range(50):
        ticks.append(t)
        t += step
        if t > y_hi + 1e-9:
            break
    if abs(step - round(step)) < 1e-9:
        ticks = [int(round(x)) for x in ticks]
    ax.set_yticks(ticks)

def set_symmetric_count_yticks(ax: plt.Axes, ylim: Tuple[float, float], *, max_ticks_total: int = 7) -> None:
    """
    Set y-ticks symmetrically around zero, with a maximum total number of ticks.
    0 is always included.
    """
    y_lo, y_hi = float(ylim[0]), float(ylim[1])
    hi = max(abs(y_lo), abs(y_hi))
    if hi <= 0:
        ax.set_yticks([0])
        return
    max_ticks_total = max(3, int(max_ticks_total))
    per_side = max(1, (max_ticks_total - 1) // 2)
    step = max(1e-12, float(compute_step(hi, per_side)))
    ticks: List[float] = [0.0]
    for i in range(1, per_side + 1):
        t = i * step
        ticks.extend([-t, t])
    ticks = sorted([t for t in ticks if y_lo - 1e-9 <= t <= y_hi + 1e-9])
    if all(abs(t - round(t)) < 1e-9 for t in ticks):
        ticks = [int(round(t)) for t in ticks]
    ax.set_yticks(ticks)

def arrival_tick_label(arrival: float) -> str:
    """
    Arrival tick label.
    """
    a_i = fmt_arrival_value(arrival)
    return f"{a_i}s"

def x_from_left_with_pad_points(fig: plt.Figure, left: float, pad_pt: float) -> float:
    """
    Compute x coordinate from left with padding in points.
    1 point = 1/72 inch.
    """
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
) -> None:
    """
    Draw points on the given Axes.
    """
    n_arrivals = len(arrivals_order)
    step = float(arrival_x_spacing)

    boundaries = [i * step for i in range(n_arrivals + 1)]
    x_base = [(i + 0.5) * step for i in range(n_arrivals)]

    n_modes = max(1, len(series))
    max_allowed = 0.45 * step
    mode_spacing = 0.0 if n_modes <= 1 else min(float(mode_x_spacing), (2.0 * max_allowed) / float(n_modes - 1))

    _color_of = color_of if color_of is not None else row_key_color

    for i, row_key in enumerate(series):
        color = _color_of(row_key)
        mode_offset = (i - (n_modes - 1) / 2.0) * mode_spacing

        for xi, arrival in enumerate(arrivals_order):
            x_center = x_base[xi] + mode_offset
            y0_list = to_finite_list(y_of(row_key, nodes_order[0], arrival, priorities))
            y1_list = to_finite_list(y_of(row_key, nodes_order[1], arrival, priorities))

            def plot_many(xs_center: float, ys: List[float], marker: str) -> None:
                """
                Plot all points at the same x_center.
                """
                for y in ys:
                    ax.plot(
                        [xs_center],
                        [y],
                        marker=marker,
                        linestyle="None",
                        markersize=PLOT_MARKER_SIZE,
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markeredgewidth=PLOT_MARKER_LINEWIDTH,
                        alpha=PLOT_MARKER_SEED_ALPHA,
                    )

            plot_many(x_center, y0_list, marker="o")
            plot_many(x_center, y1_list, marker="s")
            
            # Plot mean across both node configurations (only when seeds are being plotted)
            if y0_list or y1_list:
                all_values = y0_list + y1_list
                if all_values:
                    mean_val = np.mean(all_values)
                    ax.plot(
                        [x_center],
                        [mean_val],
                        marker="D",
                        linestyle="None",
                        markersize=PLOT_MEAN_MARKER_SIZE,
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markeredgewidth=PLOT_MARKER_LINEWIDTH * 1.5,
                        zorder=10,
                    )

    if ycfg.scale == "symlog":
        ax.set_yscale("symlog", base=10, linthresh=ycfg.symlog_linthresh, linscale=1.0)
    else:
        ax.set_yscale("linear")

    ax.axhline(0.0, linewidth=0.8, color="black", linestyle="-", alpha=0.7)
    ax.set_ylim(float(ycfg.y_lim[0]), float(ycfg.y_lim[1]))

    if ycfg.scale == "linear":
        if y_tick_strategy == "count_sparse":
            set_count_yticks(ax, ycfg.y_lim, max_ticks=PLOT_COUNT_MAX_TICKS)
        elif y_tick_strategy == "count_sparse_symmetric":
            set_symmetric_count_yticks(ax, ycfg.y_lim, max_ticks_total=PLOT_COUNT_MAX_TICKS_SYMMETRIC)
        else:
            set_linear_yticks(ax, ycfg.y_lim)

    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.margins(x=0)

    for boundary_x in boundaries:
        ax.axvline(boundary_x, linewidth=0.8, color="black", linestyle="--", alpha=0.7, zorder=0)

    ax.set_xticks(boundaries)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE, pad=PLOT_TICK_PAD)

    if show_xticklabels:
        ax.set_xticklabels([""] * len(boundaries))
        for xi, arrival in enumerate(arrivals_order):
            ax.text(
                x_base[xi],
                -0.03,
                arrival_tick_label(arrival),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=PLOT_TICK_FONTSIZE,
                clip_on=False,
            )
        # Centered inter-arrival axis label below tick labels
        x_mid = 0.5 * (boundaries[0] + boundaries[-1])
        ax.text(
            x_mid,
            -0.12,
            "inter-arrival (s)",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=PLOT_TICK_FONTSIZE,
            clip_on=False,
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
    df_seeds: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, row_key: RowKey, col: str
) -> List[float]:
    """
    Extract per-seed values from df_seeds for given configuration.
    """
    sub = df_seeds[
        (df_seeds["nodes"] == int(nodes))
        & (df_seeds["priorities"] == int(priorities))
        & (df_seeds["arrival_s"] == float(arrival_s))
        & (df_seeds["mode"] == row_key.mode)
        & (df_seeds["blocking"] == int(row_key.blocking))
        & (df_seeds["defpreempt"] == int(row_key.defpreempt))
    ][[SEED_COL, col]].sort_values(SEED_COL, kind="mergesort")
    vals = [float(v) for v in sub[col].tolist() if is_finite(v)]
    return vals

def _compute_scaled_ylim(max_val: float, symmetric: bool = False, scale_factor: float = 1.08) -> Tuple[float, float]:
    """
    Compute y-axis limits with scaling.
    """
    hi = max_val * scale_factor if max_val > 0 else 1.0
    return (-float(hi), float(hi)) if symmetric else (0.0, float(hi))

def compute_nonnegative_ylim_main(lookup_main: pd.DataFrame, *, series_all: List[RowKey], nodes_order: List[int], arrivals_order: List[float], priorities_cols: List[int], col: str) -> Tuple[float, float]:
    """
    Compute non-negative y-axis limits for main plots.
    """
    vals = []
    for row_key in series_all:
        for nodes in nodes_order:
            for arrivals in arrivals_order:
                for priorities in priorities_cols:
                    value = lookup_val(lookup_main, nodes=nodes, priorities=priorities, arrival_s=arrivals, row_key=row_key, col=col)
                    if is_finite(value):
                        vals.append(value)
    return _compute_scaled_ylim(max(vals) if vals else 0.0, symmetric=False)

def compute_symmetric_ylim_deltas(df_delta_meanstd: pd.DataFrame, *, priorities_cols: List[int], col: str) -> Tuple[float, float]:
    """
    Compute symmetric y-axis limits for delta plots.
    """
    sub = df_delta_meanstd[df_delta_meanstd["priorities"].isin(priorities_cols)]
    vals = [abs(float(v)) for v in sub[col].tolist() if is_finite(v)]
    return _compute_scaled_ylim(max(vals) if vals else 0.0, symmetric=True)

# =============================================================================
# Plots
# =============================================================================

def make_grid(
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
    grid_top: float,
    legend_pad: float,
    mode_x_spacing: float,
    legend_x_offset: float = 0.0,
    y_tick_symmetric: bool = False,
) -> None:
    """
    Grid plotting used by both main and delta grids.
    """
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=figsize, sharex=True)
    axes[0, 0].set_title(f"#priorities={priorities_cols[0]}", fontsize=PLOT_TITLE_FONTSIZE)
    axes[0, 1].set_title(f"#priorities={priorities_cols[1]}", fontsize=PLOT_TITLE_FONTSIZE)

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
        top=grid_top,
        wspace=GRID_WSPACE,
        hspace=GRID_HSPACE,
    )

    # Color legend handles (square patches to match workload_once style)
    if color_override:
        color_handles = [mpatches.Rectangle((0, 0), 1, 1, fc=color_override(row_key), ec="black", linewidth=0.6) for row_key in series]
    else:
        color_handles = [mpatches.Rectangle((0, 0), 1, 1, fc=row_key_color(row_key), ec="black", linewidth=0.6) for row_key in series]

    # Shape legend handles
    shape_handles = []
    shape_labels = []
    for idx, (marker, label_tmpl, msize, mfc) in enumerate(SHAPE_LEGEND_SPECS):
        shape_handles.append(
            Line2D([0], [0], marker=marker, color="none",
                   markerfacecolor=mfc,
                   markeredgecolor="black",
                   markeredgewidth=SHAPE_LEGEND_EDGE_WIDTH,
                   markersize=msize, linestyle="None")
        )
        if "{nodes}" in label_tmpl and idx < len(nodes_order):
            shape_labels.append(label_tmpl.format(nodes=nodes_order[idx]))
        else:
            shape_labels.append(label_tmpl)

    bbox_l = axes[0, 0].get_position()
    bbox_r = axes[0, 1].get_position()
    x_center_grid = 0.5 * (bbox_l.x0 + bbox_r.x1)
    y_top_grid = max(bbox_l.y1, bbox_r.y1)
    legend_y = y_top_grid + float(legend_pad)

    legend_kwargs = dict(
        fontsize=PLOT_LEGEND_FONTSIZE,
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
    )

    # Place color legend first (off-screen) to measure its width
    leg_colors = fig.legend(
        color_handles,
        legend_labels,
        title="Colors",
        title_fontproperties={"size": PLOT_LEGEND_FONTSIZE, "weight": "bold"},
        loc="upper left",
        bbox_to_anchor=(0, legend_y),
        ncol=GRID_LEGEND_NCOL_COLORS,
        **legend_kwargs,
    )

    # Place shape legend (off-screen) to measure its width
    leg_shapes = fig.legend(
        shape_handles,
        shape_labels,
        title="Shapes",
        title_fontproperties={"size": PLOT_LEGEND_FONTSIZE, "weight": "bold"},
        loc="upper left",
        bbox_to_anchor=(0, legend_y),
        ncol=GRID_LEGEND_NCOL_SHAPES,
        **legend_kwargs,
    )

    # Measure widths in figure-fraction coordinates and reposition centered
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w_colors = leg_colors.get_window_extent(renderer).transformed(fig.transFigure.inverted()).width
    w_shapes = leg_shapes.get_window_extent(renderer).transformed(fig.transFigure.inverted()).width
    total_w = w_colors + GRID_LEGEND_GAP + w_shapes
    x_start = x_center_grid - total_w / 2 + legend_x_offset

    leg_colors.set_bbox_to_anchor((x_start, legend_y), transform=fig.transFigure)
    leg_colors._loc = leg_colors.codes["upper left"]
    leg_shapes.set_bbox_to_anchor((x_start + w_colors + GRID_LEGEND_GAP, legend_y), transform=fig.transFigure)
    leg_shapes._loc = leg_shapes.codes["upper left"]
    
    x_text = x_from_left_with_pad_points(fig, grid_left, GRID_YLABEL_PAD_PT)
    for r, row_spec in enumerate(GRID_ROW_SPECS):
        bbox = axes[r, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(x_text, y_center, row_spec.y_label, rotation=90, va="center", ha="right", fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    for fmt in PLOT_FORMATS:
        fig.savefig(OUT_FIGURES_DIR / f"{out_stem}.{fmt}", dpi=PLOT_FIGURE_DPI)
    plt.close(fig)

def make_grid_main(
    *,
    df_seeds: pd.DataFrame,
    lookup_main: pd.DataFrame,
    plot_seeds: bool,
    defpreempt: int,
    plot_modes: List[Tuple[str, int]],
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
    ylim_solver: Tuple[float, float],
    ylim_plans: Tuple[float, float],
    out_stem: str,
    legend_x_offset: float = 0.0,
) -> None:
    """
    Make main grid plot.
    Each mode/blocking combination is a separate series.
    """
    series = sort_row_keys([RowKey(mode=m, blocking=b, defpreempt=defpreempt) for (m, b) in plot_modes])

    def y_function_factory(col: str) -> YOfFn:
        if plot_seeds:
            return lambda row_key, n, a, k: values_from_df_seeds(df_seeds, nodes=n, priorities=k, arrival_s=a, row_key=row_key, col=col)
        return lambda row_key, n, a, k: lookup_val(lookup_main, nodes=n, priorities=k, arrival_s=a, row_key=row_key, col=col)

    legend_labels = [row_key_label(row_key) for row_key in series]

    make_grid(
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
        grid_top=GRID_TOP_MAIN,
        legend_pad=GRID_LEGEND_PAD_MAIN,
        mode_x_spacing=PLOT_MODE_X_SPACING_MAIN,
        legend_x_offset=legend_x_offset,
        y_tick_symmetric=False,
    )

def make_grid_periodic_vs_stable(
    *,
    df_delta_seeds: pd.DataFrame,
    lookup_deltas: pd.DataFrame,
    plot_seeds: bool,
    defpreempt: int,
    blocking: int,
    delta_series_names: List[str],
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
    ylim_solver: Tuple[float, float],
    ylim_plans: Tuple[float, float],
    out_stem: str,
    legend_x_offset: float = 0.0,
) -> None:
    """
    Make grid plot comparing periodic vs stable scheduling deltas.
    Each delta metric is a separate series, with custom coloring.
    """
    # Create fake series for coloring
    fake_series = [RowKey(mode=f"custom{i}", blocking=0, defpreempt=defpreempt) for i in range(len(delta_series_names))]

    def _extract_custom_index(row_key: RowKey) -> int:
        """
        Extract index from custom RowKey mode.
        """
        return int(row_key.mode.replace("custom", "")) if row_key.mode.startswith("custom") else 0

    def color_override(row_key: RowKey) -> Any:
        """
        Color override using unified color palette from PLOT_COLORS.
        """
        idx = _extract_custom_index(row_key)
        color_idx = DELTA_SPECS[idx].color_idx
        return PLOT_COLORS[int(color_idx) % len(PLOT_COLORS)]

    def y_function_factory(col: str) -> YOfFn:
        """
        Creates y-value function for deltas that dispatches based on RowKey.
        """
        # Build individual delta functions with cleaner closure handling
        def make_delta_func(delta_name: str):
            """
            Create y-value function for a specific delta_name.
            """
            if plot_seeds:
                def _y(_row_key: RowKey, nodes: int, a: float, priorities: int) -> List[float]:
                    sub = df_delta_seeds[
                        (df_delta_seeds["defpreempt"] == defpreempt) &
                        (df_delta_seeds["blocking"] == blocking) &
                        (df_delta_seeds["nodes"] == nodes) &
                        (df_delta_seeds["priorities"] == priorities) &
                        (df_delta_seeds["arrival_s"] == a) &
                        (df_delta_seeds["delta_name"] == delta_name)
                    ][[SEED_COL, col]].sort_values(SEED_COL, kind="mergesort")
                    return [float(v) for v in sub[col].tolist() if is_finite(v)]
            else:
                def _y(_row_key: RowKey, nodes: int, a: float, priorities: int) -> float:
                    m, _ = lookup_delta_mean_std(lookup_deltas, nodes=nodes, priorities=priorities, arrival_s=a, defpreempt=defpreempt, blocking=blocking, delta_name=delta_name, col_mean=col)
                    return float(m)
            return _y
        
        delta_fns = {i: make_delta_func(dn) for i, dn in enumerate(delta_series_names)}
        return lambda row_key, nodes, a, priorities: delta_fns[_extract_custom_index(row_key)](row_key, nodes, a, priorities)

    # Build legend labels with blocking type annotation
    blocking_suffix = "(blocking)" if blocking == 1 else "(non-blocking)"
    legend_labels = [f"{name} {blocking_suffix}" for name in delta_series_names]

    make_grid(
        y_function_factory=y_function_factory,
        series=fake_series,
        color_override=color_override,
        legend_labels=legend_labels,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_cols=priorities_cols,
        ylim_solver=ylim_solver,
        ylim_plans=ylim_plans,
        out_stem=out_stem,
        y_config=Y_DELTAS,
        figsize=GRID_FIGSIZE_DELTAS,
        grid_left=GRID_LEFT_DELTAS,
        grid_top=GRID_TOP_DELTAS,
        legend_pad=GRID_LEGEND_PAD_DELTAS,
        mode_x_spacing=PLOT_MODE_X_SPACING_DELTAS,
        legend_x_offset=legend_x_offset,
        y_tick_symmetric=True,
    )

# =============================================================================
# Tables (LaTeX)
# =============================================================================

@dataclass(frozen=True)
class MetricSpec:
    name: str
    col_total: str
    col_prio_pattern: Optional[str]
    mean_signed: bool
    mean_dec: int
    std_dec: int

    def format_val(self, mean_val: object, std_val: object) -> str:
        """Format a single value (mean ± std)."""
        return fmt_mean_std(mean_val, std_val, mean_signed=self.mean_signed, mean_dec=self.mean_dec, std_dec=self.std_dec)

METRIC_SPECS_ALL: List[MetricSpec] = [
    MetricSpec("usage",            "delta_U_pct_eff_mean",  None,                     True,  1, 1),
    MetricSpec("latency",          "delta_L_ms_total_mean", "delta_L_ms_p{p}_mean",   True,  0, 0),
    MetricSpec("deletions",        "delta_D_num_total_mean","delta_D_num_p{p}_mean",   True,  1, 1),
    MetricSpec("optimizer_runs",   "solver_attempts_mean",  None,                     False, 0, 1),
    MetricSpec("plan_activations", "plan_activated_mean",   None,                     False, 0, 1),
]

def latex_table_metric(
    *,
    out_path: Path,
    lookup_main: pd.DataFrame,
    spec: MetricSpec,
    defpreempt: int,
    priorities: int,
    nodes_order: List[int],
    arrivals_order: List[float],
) -> None:
    """
    Generate a LaTeX table for one metric.
    """
    modes = sort_row_keys([RowKey(mode=s.mode, blocking=int(s.blocking), defpreempt=int(defpreempt)) for s in MODE_SPECS])
    n_arrivals = len(arrivals_order)
    n_nodes = len(nodes_order)

    lines: List[str] = []

    # Column specification: label column + data columns with vertical separator between node groups
    col_groups = []
    for i, _ in enumerate(nodes_order):
        col_groups.append(" @{\\hspace{0.5em}} ".join(["c"] * n_arrivals))
    if priorities > 1:
        colspec = "l @{\\hspace{1em}} " + " @{\\hspace{1.5em}} ".join(col_groups)
    else:
        colspec = "l " + " @{\\hspace{1.5em}} ".join(col_groups)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    # First header row: node counts (each spans n_arrivals columns)
    node_headers = [
        rf"\multicolumn{{{n_arrivals}}}{{c}}{{\textbf{{\# Nodes =\,{n}}}}}"
        for n in nodes_order
    ]
    lines.append(r"\multirow{2}{*}{\textbf{Trigger Mode}} & " + " & ".join(node_headers) + r" \\")

    # Add cmidrule under each node group for distinction
    lines.append(latex_cmidrules(n_nodes, n_arrivals))

    arrival_headers = []
    is_first_overall = True
    for _ in nodes_order:
        for i, a in enumerate(arrivals_order):
            if i == 0 and is_first_overall:
                arrival_headers.append(rf"\textbf{{\llap{{Inter-arrival =\,}}{fmt_arrival_value(a)}s}}")
                is_first_overall = False
            else:
                arrival_headers.append(rf"\textbf{{{fmt_arrival_value(a)}s}}")
    lines.append(" & " + " & ".join(arrival_headers) + r" \\")
    lines.append(r"\midrule")

    # Data rows: one per mode
    for row_key in modes:
        cells: List[str] = []
        for nodes in nodes_order:
            for arrival in arrivals_order:
                # Always get the total value first
                col_total = spec.col_total
                col_total_std = f"{col_total}_std"
                mean_total = lookup_val(lookup_main, nodes=nodes, priorities=priorities, arrival_s=arrival, row_key=row_key, col=col_total)
                std_total = lookup_val(lookup_main, nodes=nodes, priorities=priorities, arrival_s=arrival, row_key=row_key, col=col_total_std)
                total_str = spec.format_val(mean_total, std_total)

                # Check if we need per-priority values
                if spec.col_prio_pattern is not None and priorities > 1:
                    # Use tabular for colon alignment
                    cell_parts: List[str] = [rf"total & {total_str}"]
                    for p in range(1, MAX_PRIORITIES + 1):
                        col = spec.col_prio_pattern.format(p=p)
                        col_std = f"{col}_std"
                        mean_v = lookup_val(lookup_main, nodes=nodes, priorities=priorities, arrival_s=arrival, row_key=row_key, col=col)
                        std_v = lookup_val(lookup_main, nodes=nodes, priorities=priorities, arrival_s=arrival, row_key=row_key, col=col_std)
                        cell_parts.append(rf"p{p} & {spec.format_val(mean_v, std_v)}")
                    inner_rows = r"\\".join(cell_parts)
                    cell = rf"\begin{{tabular}}[t]{{@{{}}r@{{:\ }}l@{{}}}}{inner_rows}\end{{tabular}}"
                else:
                    cell = total_str
                cells.append(cell)
        mode_label = row_key_table_label(row_key, multiline=(priorities > 1))
        lines.append(f"{mode_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Wrap in resizebox to fit text width
    lines.insert(0, r"\resizebox{\textwidth}{!}{%")
    lines.append("}")

    # Generate caption based on parameters
    prio_desc = "one priority" if priorities == 1 else f"{priorities} priorities"
    preempt_desc = "with DefaultPreemption enabled" if defpreempt else "with DefaultPreemption disabled"
    metric_captions = {
        "usage": "effective resource usage (\\%)",
        "latency": "scheduling latency (ms)",
        "deletions": "number of pod deletions",
        "optimizer_runs": "number of optimizer runs",
        "plan_activations": "number of plan activations",
    }
    metric_desc = metric_captions.get(spec.name, spec.name)
    caption = (
        f"Mean paired differences in {metric_desc} between the plugin {preempt_desc} and the default scheduler for runs with {prio_desc} (mean ± std)."
    )
    label = f"tab:{spec.name}-defpreempt{defpreempt}-prio{priorities}"

    write_latex_table(out_path, lines, caption=caption, label=label)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Generating tables and figures...")

    configure_matplotlib()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_seeds = load_results_seeds(IN_RESULTS_SEEDS)

    df_mean_std = aggregate_mean_std(df_seeds)
    lookup_main = build_lookup(df_mean_std)

    nodes_order = sorted(df_mean_std["nodes"].unique().tolist())
    arrivals_order_all = df_mean_std["arrival_s"].unique().tolist()
    arrivals_order = select_arrivals_order(arrivals_order_all)

    # Build delta datasets (seed-level + mean/std)
    df_delta_seeds = build_delta_seeds(df_seeds)
    df_delta_mean_std = aggregate_delta_mean_std(df_delta_seeds) if not df_delta_seeds.empty else pd.DataFrame(columns=_DELTA_KEY_COLS)
    lookup_deltas = build_lookup_deltas(df_delta_mean_std) if not df_delta_mean_std.empty else pd.DataFrame().set_index(_DELTA_KEY_COLS)

    # Shared y-lims (main counters) across defpreempt=0/1 and all blocking types
    series_all_for_limits = sort_row_keys([
        RowKey(mode=m, blocking=b, defpreempt=d)
        for d in (0, 1) for b in (0, 1) for m in MAIN_PLOT_MODE_NAMES
    ])
    ylim_solver_main = YLIM_SOLVER_MAIN or compute_nonnegative_ylim_main(lookup_main, series_all=series_all_for_limits, nodes_order=nodes_order, arrivals_order=arrivals_order, priorities_cols=PRIORITIES_TO_SHOW, col="solver_attempts_mean")
    ylim_plans_main = YLIM_PLANS_MAIN or compute_nonnegative_ylim_main(lookup_main, series_all=series_all_for_limits, nodes_order=nodes_order, arrivals_order=arrivals_order, priorities_cols=PRIORITIES_TO_SHOW, col="plan_activated_mean")

    # Shared y-lims (delta counters) across defpreempt=0/1 and all blocking types (symmetric)
    if df_delta_mean_std.empty:
        ylim_solver_deltas = YLIM_SOLVER_DELTAS or (-1.0, 1.0)
        ylim_plans_deltas = YLIM_PLANS_DELTAS or (-1.0, 1.0)
    else:
        ylim_solver_deltas = YLIM_SOLVER_DELTAS or compute_symmetric_ylim_deltas(df_delta_mean_std, priorities_cols=PRIORITIES_TO_SHOW, col="solver_attempts_mean")
        ylim_plans_deltas = YLIM_PLANS_DELTAS or compute_symmetric_ylim_deltas(df_delta_mean_std, priorities_cols=PRIORITIES_TO_SHOW, col="plan_activated_mean")

    produced_tables: List[Path] = []
    produced_figs: List[Path] = []

    # ---------- Tables ----------
    delta_names = [d.label for d in DELTA_SPECS]

    for spec in METRIC_SPECS_ALL:
        for defpreempt in (0, 1):
            for k in PRIORITIES_TO_SHOW:
                out = OUT_TABLES_DIR / f"table_defpreempt={defpreempt}_priorities={k}_{spec.name}.tex"
                latex_table_metric(out_path=out, lookup_main=lookup_main, spec=spec, defpreempt=defpreempt, priorities=k, nodes_order=nodes_order, arrivals_order=arrivals_order)
                produced_tables.append(out)

    # ---------- Figures ----------
    for blocking in (0, 1):
        plot_modes = [(m, blocking) for m in MAIN_PLOT_MODE_NAMES]
        for defpreempt in (1, 0):
            # Main grid
            stem = f"main_defaultpreempt={defpreempt}_blocking={blocking}"
            make_grid_main(df_seeds=df_seeds, lookup_main=lookup_main, plot_seeds=True, defpreempt=defpreempt, plot_modes=plot_modes, nodes_order=nodes_order, arrivals_order=arrivals_order, priorities_cols=PRIORITIES_TO_SHOW, ylim_solver=ylim_solver_main, ylim_plans=ylim_plans_main, out_stem=stem, legend_x_offset=GRID_LEGEND_X_OFFSET_MAIN.get(blocking, 0.0))
            produced_figs.extend([OUT_FIGURES_DIR / f"{stem}.{fmt}" for fmt in PLOT_FORMATS])

            # Delta grid (periodic vs stable)
            stem = f"periodic_vs_stable_defaultpreempt={defpreempt}_blocking={blocking}"
            make_grid_periodic_vs_stable(df_delta_seeds=df_delta_seeds, lookup_deltas=lookup_deltas, plot_seeds=True, defpreempt=defpreempt, blocking=blocking, delta_series_names=delta_names, nodes_order=nodes_order, arrivals_order=arrivals_order, priorities_cols=PRIORITIES_TO_SHOW, ylim_solver=ylim_solver_deltas, ylim_plans=ylim_plans_deltas, out_stem=stem, legend_x_offset=GRID_LEGEND_X_OFFSET_DELTAS.get(blocking, 0.0))
            produced_figs.extend([OUT_FIGURES_DIR / f"{stem}.{fmt}" for fmt in PLOT_FORMATS])

    # Summary
    for label, paths in [("Tables", produced_tables), ("Figures", produced_figs)]:
        print(f"{label}:\n" + "\n".join(f"  - {p}" for p in paths))

if __name__ == "__main__":
    main()