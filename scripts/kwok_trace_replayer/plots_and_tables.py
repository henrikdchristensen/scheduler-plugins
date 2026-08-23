#!/usr/bin/env python3
# scripts/kwok_trace_replayer/plots_and_tables.py
"""
python -m scripts.kwok_trace_replayer.plots_and_tables
"""

import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Any, Dict, Iterable, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.helpers.data_helpers import (
    is_finite
)
from scripts.helpers.table_helpers import (
    fmt_mean_std,
    fmt_signed,
    nan_str,
    latex_cmidrules,
    write_latex_table,
    aggregate_mean_std,
    as_lookup,
    safe_at,
    to_float_list,
    repeat_header_values_with_first_label,
)
from scripts.helpers.plot_helpers import (
    PLOT_COLORS,
    configure_matplotlib,
    save_figure,
    set_linear_ticks,
    set_count_ticks,
    set_symmetric_count_ticks,
    lighter_color,
    center_two_legends,
    shape_legend_handles,
    color_patch_handles,
)
from scripts.helpers.plot_config import (
    PLOT_TITLE_FONTSIZE,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_HANDLE_LENGTH,
    PLOT_LEGEND_COLUMN_SPACING,
    PLOT_LEGEND_HANDLE_TEXT_PAD,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IN_RESULTS_SEEDS = Path("analysis/kwok_trace_replayer/results_seeds.csv")
TRACES_DIR = Path("data/traces")

OUT_DIR = Path("analysis/kwok_trace_replayer")
OUT_TABLES_DIR = OUT_DIR / "tables"
OUT_FIGS_DIR = OUT_DIR / "figures"

SEED_COL = "seed"
SOLVER_DISPLAY_NAME = "solver"
AXIS_LABEL_FONTSIZE = 6.0
TICK_LABEL_FONTSIZE = 5.5
MAX_PRIORITIES = 4
TABLE_METRIC_ENVIRONMENT = "table*"
TABLE_PLACEMENT = "htbp"
TABLE_METRIC_MAX_HEIGHT = r"0.9\textheight"

PRIORITIES_TO_SHOW = [1, 4]
INTER_ARRIVALS_TO_SHOW = [4.0, 8.0, 16.0]

TABLE_COL_WIDTH = "2.5em"
TABLE_COL_WIDTH_METRIC = "5.0em"
TABLE_COL_WIDTH_METRIC_P4 = "9.0em"

YLIM_SOLVER_MAIN = (0.0, 300.0)
YLIM_PLANS_MAIN = (0.0, 300.0)
YLIM_SOLVER_DELTAS = (-150.0, 150.0)
YLIM_PLANS_DELTAS = (-60.0, 60.0)
YLIM_CERTIFIED_MAIN = (0.0, 105.0)
YLIM_CERTIFIED_DELTAS = (-40.0, 40.0)

BASE_MODES: List[Tuple[str, str, str, int, int]] = [
    ("schedulingfailure",  "Scheduling-failure", "",             0,  1),
    ("periodic4s",         "Periodic",           "4s interval",  2,  5),
    ("periodic8s",         "Periodic",           "8s interval",  4,  6),
    ("periodic16s",        "Periodic",           "16s interval", 6,  7),
    ("stable-queue-4s",    "Stable-queue",       "4s delay",     8,  9),
    ("stable-queue-8s",    "Stable-queue",       "8s delay",     10, 10),
    ("stable-queue-16s",   "Stable-queue",       "16s delay",    12, 11),
]

MAIN_PLOT_MODE_NAMES = ["schedulingfailure", "periodic8s", "stable-queue-8s"]
TABLE_METRIC_MODE_NAMES = [
    "schedulingfailure",
    "periodic4s", "periodic8s", "periodic16s",
    "stable-queue-4s", "stable-queue-8s", "stable-queue-16s",
]
BLOCKING_DIFF_MODE_NAMES = ["schedulingfailure", "periodic8s", "stable-queue-8s"]

TIMING_DELTA_SPECS = [
    ("periodic8s", "periodic4s", "Periodic", "8s $\\to$ 4s"),
    ("periodic8s", "periodic16s", "Periodic", "8s $\\to$ 16s"),
    ("stable-queue-8s", "stable-queue-4s", "Stable-queue", "8s $\\to$ 4s"),
    ("stable-queue-8s", "stable-queue-16s", "Stable-queue", "8s $\\to$ 16s"),
]

SEED_COLS_NEEDED = [
    "job_name", "plugin_config", "seed", "T_end_s_mean",
    "delta_U_pct_cpu_mean","delta_U_pct_mem_mean","delta_U_pct_eff_mean",
    "delta_R_num_p1_mean","delta_R_num_p2_mean","delta_R_num_p3_mean","delta_R_num_p4_mean","delta_R_num_total_mean",
    "delta_D_num_p1_mean","delta_D_num_p2_mean","delta_D_num_p3_mean","delta_D_num_p4_mean","delta_D_num_total_mean",
    "delta_L_ms_p1_mean","delta_L_ms_p2_mean","delta_L_ms_p3_mean","delta_L_ms_p4_mean","delta_L_ms_total_mean",
    "solver_attempts_mean","solver_optimal_mean","solver_feasible_mean","solver_failed_mean",
    "plan_not_applicable_mean","plan_activated_mean",
]

KEY_COLS_MAIN = ["nodes", "priorities", "arrival_s", "mode", "blocking", "defpreempt"]

# Grid / plot styling
GRID = {
    "plot_height": 7.0,
    "tick_pad": 2.0,
    "seed_marker_size": 2.0,
    "seed_alpha": 0.5,
    "mean_marker_size": 3.5,
    "marker_edge": 0.4,
    "shape_edge": 0.6,
    "arrival_x_spacing": 0.35,
    "mode_spacing_main": 0.05,   # combined blocking/non-blocking
    "mode_spacing_delta": 0.08,
    "main_figsize": (6.5, 7.0),
    "delta_figsize": (3.6, 7.0),
    "main_left": 0.14,
    "delta_left": 0.125,
    "right": 0.99,
    "bottom": 0.04,
    "main_top": 0.89,
    "delta_top": 0.873,
    "wspace": 0.10,
    "hspace": 0.10,
    "legend_pad_main": 0.113,
    "legend_pad_delta": 0.13,
    "legend_gap_main": 0.001,
    "legend_gap_delta": 0.01,
    "legend_xoff_main": -0.04,
    "legend_xoff_delta": -0.012,
    "ylabel_pad_pt": 30.0,
    "legend_ncol_colors_main": 3,
    "legend_ncol_colors_delta": 1,
}

SHAPE_LEGEND_SPECS = [
    ("o", "run with {nodes} nodes", 3.6, "none"),
    ("s", "run with {nodes} nodes", 3.5, "none"),
    ("D", "avg. for all runs", 3.6, "none"),
]

# ---------------------------------------------------------------------------
# Specs / models
# ---------------------------------------------------------------------------

JOB_RE = re.compile(r"nodes=(\d+)_prio=(\d+)_arrival=([0-9.]+)s")


@dataclass(frozen=True)
class RowKey:
    mode: str
    blocking: int
    defpreempt: int


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    blocking: int
    base_label: str
    detail: str
    rank: int
    color_idx: int

    @property
    def label(self) -> str:
        b = "blocking" if self.blocking else "non-blocking"
        s = f"{self.base_label} ({b})"
        return f"{s}, {self.detail}" if self.detail else s


@dataclass(frozen=True)
class AxisCfg:
    scale: str  # linear | symlog
    ylim: Tuple[float, float]
    linthresh: float = 1.0


@dataclass(frozen=True)
class GridRowSpec:
    col: str
    axis_key: str
    ylabel: str
    delta_ylabel: Optional[str] = None
    tick_strategy: str = "auto"   # auto | count | count_symmetric | percent | percent_symmetric
    bottom: bool = False


@dataclass(frozen=True)
class MetricSpec:
    name: str
    col_total: str
    col_prio_pattern: Optional[str]
    mean_signed: bool
    mean_dec: int
    std_dec: int

    def fmt(self, mean_v: object, std_v: object) -> str:
        return fmt_mean_std(
            mean_v, std_v,
            mean_signed=self.mean_signed,
            mean_dec=self.mean_dec,
            std_dec=self.std_dec,
        )


MODE_SPECS: List[ModeSpec] = []
for mode, base, detail, rank_base, cidx in BASE_MODES:
    for blocking in (1, 0):
        MODE_SPECS.append(
            ModeSpec(
                mode=mode,
                blocking=blocking,
                base_label=base,
                detail=detail,
                rank=rank_base + (0 if blocking else 1),
                color_idx=cidx,
            )
        )

MODE_BY_KEY = {(m.mode, m.blocking): m for m in MODE_SPECS}

Y_MAIN = {
    "util": AxisCfg("linear", (-2.0, 5.0)),
    "latency": AxisCfg("symlog", (-1e5 - 1.0, 1e5 + 1.0)),
    "deletions": AxisCfg("symlog", (-1e4 - 1.0, 1e4 + 1.0)),
    "solver": AxisCfg("linear", YLIM_SOLVER_MAIN),
    "plans": AxisCfg("linear", YLIM_PLANS_MAIN),
    "certified": AxisCfg("linear", YLIM_CERTIFIED_MAIN),
}
Y_DELTAS = {
    "util": AxisCfg("linear", (-0.5, 0.5)),
    "latency": AxisCfg("symlog", (-1e4 - 1.0, 1e4 + 1.0)),
    "deletions": AxisCfg("symlog", (-1e3 - 1.0, 1e3 + 1.0)),
    "solver": AxisCfg("linear", YLIM_SOLVER_DELTAS),
    "plans": AxisCfg("linear", YLIM_PLANS_DELTAS),
    "certified": AxisCfg("linear", YLIM_CERTIFIED_DELTAS),
}

GRID_ROWS = [
    GridRowSpec("delta_U_pct_eff_mean", "util", "diff.\nusage (%)"),
    GridRowSpec("delta_L_ms_total_mean", "latency", "diff.\nlatency (ms)"),
    GridRowSpec("delta_D_num_total_mean", "deletions", "diff. pod\ndeletions"),
    GridRowSpec("solver_attempts_mean", "solver", f"{SOLVER_DISPLAY_NAME} runs", tick_strategy="count"),
    GridRowSpec("plan_activated_mean", "plans", "activated\nimproving plans", tick_strategy="count"),
    GridRowSpec(
        "optimal_solver_run_pct_mean",
        "certified",
        "runs certified\noptimal (%)",
        delta_ylabel="diff. runs certified\noptimal (%)",
        tick_strategy="percent",
    ),
    GridRowSpec(
        "proven_optimal_plan_pct_mean",
        "certified",
        "optimal among\nimproving plans (%)",
        delta_ylabel="diff. optimal among\nimproving plans (%)",
        tick_strategy="percent",
        bottom=True,
    ),
]

METRICS_ALL = [
    MetricSpec("usage", "delta_U_pct_eff_mean", None, True, 1, 1),
    MetricSpec("latency", "delta_L_ms_total_mean", "delta_L_ms_p{p}_mean", True, 0, 0),
    MetricSpec("deletions", "delta_D_num_total_mean", "delta_D_num_p{p}_mean", True, 1, 1),
    MetricSpec("optimizer_runs", "solver_attempts_mean", None, False, 0, 1),
    MetricSpec("plan_activations", "plan_activated_mean", None, False, 0, 1),
    MetricSpec("optimal_solver_runs", "optimal_solver_run_pct_mean", None, False, 1, 1),
    MetricSpec("proven_optimal_plans", "proven_optimal_plan_pct_mean", None, False, 1, 1),
]

DIFF_METRICS = METRICS_ALL

# Delta (timing) plot specs
@dataclass(frozen=True)
class DeltaPlotSpec:
    label: str
    baseline: str
    compared: str
    color_idx: int

DELTA_PLOT_SPECS = [
    DeltaPlotSpec("Periodic, 8→4s interval", "periodic8s", "periodic4s", 4),
    DeltaPlotSpec("Periodic, 8→16s interval", "periodic8s", "periodic16s", 7),
    DeltaPlotSpec("Stable-queue, 8→4s delay", "stable-queue-8s", "stable-queue-4s", 8),
    DeltaPlotSpec("Stable-queue, 8→16s delay", "stable-queue-8s", "stable-queue-16s", 11),
]
DELTA_METRIC_COLS = [
    "delta_U_pct_eff_mean", "delta_L_ms_total_mean", "delta_D_num_total_mean",
    "solver_attempts_mean", "plan_activated_mean", "optimal_solver_run_pct_mean",
    "proven_optimal_plan_pct_mean",
]

# ---------------------------------------------------------------------------
# Formatting / captions
# ---------------------------------------------------------------------------

def fmt_arrival(a: float) -> str:
    return f"{int(a)}s" if abs(a - round(a)) < 1e-9 else f"{a}s"

def _prio_desc(prio: int) -> str:
    return "1 priority level" if int(prio) == 1 else f"{int(prio)} priority levels"

def _prio_desc_title(prio: Optional[int]) -> str:
    if prio is None:
        return "All Priorities"
    return "1 Priority" if int(prio) == 1 else f"{int(prio)} Priorities"

def _preempt_title(defpreempt: int) -> str:
    return "DefaultPreemption enabled" if int(defpreempt) else "DefaultPreemption disabled"

def _blocking_title(blocking: int) -> str:
    return "Blocking" if int(blocking) else "Non-blocking"

def rowkey_label(rk: RowKey) -> str:
    spec = MODE_BY_KEY.get((rk.mode, rk.blocking))
    return spec.label if spec else f"{rk.mode}:{rk.blocking}"

def rowkey_table_label(rk: RowKey, *, multiline: bool) -> str:
    spec = MODE_BY_KEY.get((rk.mode, rk.blocking))
    if spec is None:
        return f"{rk.mode}:{rk.blocking}"
    if not multiline:
        return spec.label
    b = "blocking" if spec.blocking else "non-blocking"
    parts = [spec.base_label, f"({b})"]
    if spec.detail:
        parts.append(spec.detail)
    inner = r"\\".join(parts)
    return rf"\begin{{tabular}}[t]{{@{{}}l@{{}}}}{inner}\end{{tabular}}"

def rowkey_color(rk: RowKey):
    spec = MODE_BY_KEY.get((rk.mode, rk.blocking))
    return PLOT_COLORS[(spec.color_idx if spec else 0) % len(PLOT_COLORS)]

# ---------------------------------------------------------------------------
# Parse / load
# ---------------------------------------------------------------------------

def parse_job_name(job_name: str) -> Tuple[int, int, float]:
    m = JOB_RE.fullmatch(str(job_name).strip())
    if not m:
        raise ValueError(f"Invalid job_name: {job_name}")
    return int(m.group(1)), int(m.group(2)), float(m.group(3))

def canonical_mode(mode: str) -> str:
    s = str(mode).strip().lower()
    m = re.fullmatch(r"periodic-?([0-9.]+)s", s)
    if m:
        return f"periodic{m.group(1)}s"
    m = re.fullmatch(r"(?:stable-queue|stablequeue)-?([0-9.]+)s", s)
    if m:
        return f"stable-queue-{m.group(1)}s"
    return s

def parse_plugin_config(cfg: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in str(cfg).split("_"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out

def rowkey_from_plugin(cfg: str) -> RowKey:
    kv = parse_plugin_config(cfg)
    mode = canonical_mode(kv.get("mode", "unknown"))
    blocking = 1 if str(kv.get("blocking", "0")).lower() in {"1", "true", "yes"} else 0
    defpreempt = int(kv.get("defpreempt", "0"))
    return RowKey(mode=mode, blocking=blocking, defpreempt=defpreempt)

def load_results_seeds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in SEED_COLS_NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {path}: {', '.join(missing)}")

    parsed_jobs = df["job_name"].map(parse_job_name)
    df[["nodes", "priorities", "arrival_s"]] = pd.DataFrame(parsed_jobs.tolist(), index=df.index)

    parsed_cfg = df["plugin_config"].map(rowkey_from_plugin)
    df[["mode", "blocking", "defpreempt"]] = pd.DataFrame(
        [(r.mode, r.blocking, r.defpreempt) for r in parsed_cfg], index=df.index
    )
    df["optimal_solver_run_pct_mean"] = (
        100.0 * df["solver_optimal_mean"] / df["solver_attempts_mean"]
    ).where(df["solver_attempts_mean"] > 0)
    improving_plans = df["solver_optimal_mean"] + df["solver_feasible_mean"]
    df["proven_optimal_plan_pct_mean"] = (
        100.0 * df["solver_optimal_mean"] / improving_plans
    ).where(improving_plans > 0)
    return df

def select_arrivals(all_arrivals: Iterable[float]) -> List[float]:
    available = sorted(set(float(x) for x in all_arrivals))
    wanted = [float(x) for x in INTER_ARRIVALS_TO_SHOW]
    missing = [x for x in wanted if x not in set(available)]
    if missing:
        raise SystemExit(f"INTER_ARRIVALS_TO_SHOW missing in data: {missing}; available={available}")
    return wanted

# ---------------------------------------------------------------------------
# Aggregation / lookups
# ---------------------------------------------------------------------------

def build_main_mean_std(df_seeds: pd.DataFrame) -> pd.DataFrame:
    return aggregate_mean_std(df_seeds, ["job_name", "plugin_config"] + KEY_COLS_MAIN, seed_col=SEED_COL)

def build_main_lookup(df_mean_std: pd.DataFrame) -> pd.DataFrame:
    return as_lookup(df_mean_std, KEY_COLS_MAIN)

def lookup_main(lookup: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, rk: RowKey, col: str) -> float:
    key = (int(nodes), int(priorities), float(arrival_s), rk.mode, int(rk.blocking), int(rk.defpreempt))
    return safe_at(lookup, key, col)

# ---------------------------------------------------------------------------
# Variant-diff seed datasets (to match the *diff* tables)
# ---------------------------------------------------------------------------

MODE_META: Dict[str, Tuple[str, str, int]] = {
    mode: (base_label, detail, cidx) for (mode, base_label, detail, _rank, cidx) in BASE_MODES
}

def _mode_short_label(mode: str) -> str:
    base_label, detail, _cidx = MODE_META.get(mode, (mode, "", 0))
    return f"{base_label}, {detail}" if detail else base_label

def _mode_base_color(mode: str):
    _base_label, _detail, cidx = MODE_META.get(mode, (mode, "", 0))
    return PLOT_COLORS[cidx % len(PLOT_COLORS)]

BLOCKING_DIFF_SEED_COLS = ["nodes", "priorities", "arrival_s", "defpreempt", "mode", SEED_COL]
DEFPREEMPT_DIFF_SEED_COLS = ["nodes", "priorities", "arrival_s", "blocking", "mode", SEED_COL]

def build_blocking_diff_seeds(df_seeds: pd.DataFrame) -> pd.DataFrame:
    """
    Per-seed differences between non-blocking and blocking variants:
      diff = (non-blocking metric) - (blocking metric)
    Matches table_blocking_diff_defpreempt=*.tex.
    """
    metric_cols = [m.col_total for m in DIFF_METRICS]
    base_cols = ["nodes", "priorities", "arrival_s", "defpreempt", "mode", SEED_COL]

    left = df_seeds[df_seeds["blocking"] == 1][base_cols + metric_cols]   # blocking
    right = df_seeds[df_seeds["blocking"] == 0][base_cols + metric_cols]  # non-blocking

    merged = right.merge(left, on=base_cols, how="inner", suffixes=("_NB", "_B"))
    if merged.empty:
        return pd.DataFrame(columns=BLOCKING_DIFF_SEED_COLS + metric_cols)

    out = merged[base_cols].copy()
    for c in metric_cols:
        out[c] = merged[f"{c}_NB"] - merged[f"{c}_B"]
    return out

def build_defpreempt_diff_seeds(df_seeds: pd.DataFrame) -> pd.DataFrame:
    """
    Per-seed differences between DefaultPreemption enabled and disabled:
      diff = (enabled metric) - (disabled metric)
    Computed within fixed blocking variant.
    Matches table_defpreempt_diff_blocking=*.tex.
    """
    metric_cols = [m.col_total for m in DIFF_METRICS]
    base_cols = ["nodes", "priorities", "arrival_s", "blocking", "mode", SEED_COL]

    left = df_seeds[df_seeds["defpreempt"] == 0][base_cols + metric_cols]   # disabled
    right = df_seeds[df_seeds["defpreempt"] == 1][base_cols + metric_cols]  # enabled

    merged = right.merge(left, on=base_cols, how="inner", suffixes=("_EN", "_DIS"))
    if merged.empty:
        return pd.DataFrame(columns=DEFPREEMPT_DIFF_SEED_COLS + metric_cols)

    out = merged[base_cols].copy()
    for c in metric_cols:
        out[c] = merged[f"{c}_EN"] - merged[f"{c}_DIS"]
    return out

# ---------------------------------------------------------------------------
# Variant-diff plots (main-style grids)
# ---------------------------------------------------------------------------

def make_blocking_diff_grid(
    *,
    df_blockdiff_seeds: pd.DataFrame,
    defpreempt: int,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
) -> None:
    """
    Main-style grid for (non-blocking - blocking) deltas.
    One color per mode (no blocking split, since it's already differenced).
    """
    if df_blockdiff_seeds.empty:
        return

    df = df_blockdiff_seeds[
        (df_blockdiff_seeds["defpreempt"] == int(defpreempt))
        & (df_blockdiff_seeds["mode"].isin(BLOCKING_DIFF_MODE_NAMES))
    ].copy()
    if df.empty:
        return

    series = list(BLOCKING_DIFF_MODE_NAMES)

    def color_of(mode: str):
        return _mode_base_color(mode)

    def y_factory(col: str) -> YOfFn:
        def _f(mode: str, n: int, a: float, p: int):
            sub = df[
                (df["mode"] == mode)
                & (df["nodes"] == int(n))
                & (df["priorities"] == int(p))
                & (df["arrival_s"] == float(a))
            ][[SEED_COL, col]].sort_values(SEED_COL, kind="mergesort")
            return [float(v) for v in sub[col].tolist() if is_finite(v)]
        return _f

    make_grid_figure(
        y_factory=y_factory,
        series=series,
        legend_labels=[f"{_mode_short_label(m)} (non-blocking$-$blocking)" for m in series],
        color_of=color_of,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_cols=priorities_cols,
        axes_cfg=Y_DELTAS,
        out_stem=f"blocking_diffs_defpreempt={defpreempt}",
        mode_spacing=GRID["mode_spacing_delta"],
        is_delta_grid=True,
        symmetric_count_rows=True,
    )

def make_defpreempt_diff_grid(
    *,
    df_defpreemptdiff_seeds: pd.DataFrame,
    blocking: int,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
) -> None:
    """
    Main-style grid for (DefaultPreemption enabled - disabled) deltas.
    One color per mode (no defpreempt split, since it's already differenced).
    """
    if df_defpreemptdiff_seeds.empty:
        return

    df = df_defpreemptdiff_seeds[
        (df_defpreemptdiff_seeds["blocking"] == int(blocking))
       & (df_defpreemptdiff_seeds["mode"].isin(BLOCKING_DIFF_MODE_NAMES))
    ].copy()
    if df.empty:
        return

    series = list(BLOCKING_DIFF_MODE_NAMES)

    def color_of(mode: str):
        return _mode_base_color(mode)

    def y_factory(col: str) -> YOfFn:
        def _f(mode: str, n: int, a: float, p: int):
            sub = df[
                (df["mode"] == mode)
                & (df["nodes"] == int(n))
                & (df["priorities"] == int(p))
                & (df["arrival_s"] == float(a))
            ][[SEED_COL, col]].sort_values(SEED_COL, kind="mergesort")
            return [float(v) for v in sub[col].tolist() if is_finite(v)]
        return _f

    btxt = "blocking" if int(blocking) else "non-blocking"
    make_grid_figure(
        y_factory=y_factory,
        series=series,
        legend_labels=[f"{_mode_short_label(m)} (enabled$-$disabled, {btxt})" for m in series],
        color_of=color_of,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_cols=priorities_cols,
        axes_cfg=Y_DELTAS,
        out_stem=f"defpreempt_diffs_blocking={blocking}",
        mode_spacing=GRID["mode_spacing_delta"],
        is_delta_grid=True,
        symmetric_count_rows=True,
    )


def values_from_seeds(df_seeds: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, rk: RowKey, col: str) -> List[float]:
    sub = df_seeds[
        (df_seeds["nodes"] == int(nodes))
        & (df_seeds["priorities"] == int(priorities))
        & (df_seeds["arrival_s"] == float(arrival_s))
        & (df_seeds["mode"] == rk.mode)
        & (df_seeds["blocking"] == int(rk.blocking))
        & (df_seeds["defpreempt"] == int(rk.defpreempt))
    ][[SEED_COL, col]].sort_values(SEED_COL, kind="mergesort")
    return [float(v) for v in sub[col].tolist() if is_finite(v)]

def sort_rowkeys(keys: Iterable[RowKey]) -> List[RowKey]:
    uniq = list(set(keys))
    return sorted(
        uniq,
        key=lambda r: (MODE_BY_KEY.get((r.mode, r.blocking), ModeSpec(r.mode, r.blocking, r.mode, "", 999, 0)).rank, r.mode, r.blocking, r.defpreempt)
    )

# Delta seed dataset for periodic/stable timing comparisons
DELTA_KEY_COLS = ["nodes", "priorities", "arrival_s", "defpreempt", "blocking", "delta_name"]

def build_delta_seeds(df_seeds: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    base_cols = ["nodes", "priorities", "arrival_s", "defpreempt", SEED_COL]

    for blocking in (0, 1):
        for spec in DELTA_PLOT_SPECS:
            left = df_seeds[(df_seeds["mode"] == canonical_mode(spec.baseline)) & (df_seeds["blocking"] == blocking)]
            right = df_seeds[(df_seeds["mode"] == canonical_mode(spec.compared)) & (df_seeds["blocking"] == blocking)]

            if left.empty or right.empty:
                continue

            l = left[base_cols + DELTA_METRIC_COLS].rename(columns={c: f"{c}_L" for c in DELTA_METRIC_COLS})
            r = right[base_cols + DELTA_METRIC_COLS].rename(columns={c: f"{c}_R" for c in DELTA_METRIC_COLS})
            m = r.merge(l, on=base_cols, how="inner")
            if m.empty:
                continue

            d = m[base_cols].copy()
            d["blocking"] = blocking
            d["delta_name"] = spec.label
            for c in DELTA_METRIC_COLS:
                d[c] = m[f"{c}_R"] - m[f"{c}_L"]
            rows.append(d)

    if not rows:
        return pd.DataFrame(columns=DELTA_KEY_COLS + [SEED_COL] + DELTA_METRIC_COLS)
    return pd.concat(rows, ignore_index=True)

def build_delta_mean_std(df_delta_seeds: pd.DataFrame) -> pd.DataFrame:
    if df_delta_seeds.empty:
        return pd.DataFrame(columns=DELTA_KEY_COLS)
    return aggregate_mean_std(df_delta_seeds, DELTA_KEY_COLS, seed_col=SEED_COL)

def build_delta_lookup(df_delta_mean_std: pd.DataFrame) -> pd.DataFrame:
    if df_delta_mean_std.empty:
        return pd.DataFrame().set_index(DELTA_KEY_COLS)
    return as_lookup(df_delta_mean_std, DELTA_KEY_COLS)

def lookup_delta(lookup: pd.DataFrame, *, nodes: int, priorities: int, arrival_s: float, defpreempt: int, blocking: int, delta_name: str, col: str) -> float:
    key = (int(nodes), int(priorities), float(arrival_s), int(defpreempt), int(blocking), str(delta_name))
    return safe_at(lookup, key, col)

# ---------------------------------------------------------------------------
# Grid plotting (shared main + delta)
# ---------------------------------------------------------------------------

YVal = Union[float, Sequence[float]]
YOfFn = Callable[[Any, int, float, int], YVal]

def _draw_points_axis(
    *,
    ax: plt.Axes,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities: int,
    series: List[Any],
    y_of: YOfFn,
    axis_cfg: AxisCfg,
    show_xlabels: bool,
    show_ylabels: bool,
    mode_spacing: float,
    color_of: Callable[[Any], Any],
    tick_strategy: str = "auto",
) -> None:
    step = GRID["arrival_x_spacing"]
    boundaries = [i * step for i in range(len(arrivals_order) + 1)]
    x_base = [(i + 0.5) * step for i in range(len(arrivals_order))]

    n_modes = max(1, len(series))
    max_allowed = 0.45 * step
    spacing = 0.0 if n_modes <= 1 else min(float(mode_spacing), (2 * max_allowed) / (n_modes - 1))

    for i, s in enumerate(series):
        c = color_of(s)
        off = (i - (n_modes - 1) / 2.0) * spacing
        for xi, arr in enumerate(arrivals_order):
            x = x_base[xi] + off

            ys0 = to_float_list(y_of(s, nodes_order[0], arr, priorities))
            ys1 = to_float_list(y_of(s, nodes_order[1], arr, priorities))

            for y in ys0:
                ax.plot([x], [y], "o", ms=GRID["seed_marker_size"], mfc=c, mec="black", mew=GRID["marker_edge"], alpha=GRID["seed_alpha"], ls="None")
            for y in ys1:
                ax.plot([x], [y], "s", ms=GRID["seed_marker_size"], mfc=c, mec="black", mew=GRID["marker_edge"], alpha=GRID["seed_alpha"], ls="None")

            both = [v for v in (ys0 + ys1) if is_finite(v)]
            if both:
                ax.plot([x], [float(np.mean(both))], "D", ms=GRID["mean_marker_size"], mfc=c, mec="black", mew=GRID["marker_edge"] * 1.5, ls="None", zorder=10)

    if axis_cfg.scale == "symlog":
        ax.set_yscale("symlog", base=10, linthresh=axis_cfg.linthresh, linscale=1.0)
    else:
        ax.set_yscale("linear")

    ax.set_ylim(*axis_cfg.ylim)
    ax.axhline(0.0, lw=0.8, color="black", alpha=0.7)

    if axis_cfg.scale == "linear":
        if tick_strategy == "count":
            set_count_ticks(ax, axis_cfg.ylim, max_ticks=7)
        elif tick_strategy == "count_symmetric":
            set_symmetric_count_ticks(ax, axis_cfg.ylim, max_ticks_total=8)
        elif tick_strategy == "percent":
            ax.set_yticks([0, 25, 50, 75, 100])
        elif tick_strategy == "percent_symmetric":
            ax.set_yticks([-40, -30, -20, -10, 0, 10, 20, 30, 40])
        else:
            set_linear_ticks(ax, axis_cfg.ylim, min_ticks=5)

    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.margins(x=0)
    for bx in boundaries:
        ax.axvline(bx, lw=0.8, ls="--", color="black", alpha=0.7, zorder=0)

    ax.set_xticks(boundaries)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE, pad=GRID["tick_pad"])

    if show_xlabels:
        ax.set_xticklabels([""] * len(boundaries))
        for xi, arr in enumerate(arrivals_order):
            ax.text(x_base[xi], -0.03, fmt_arrival(arr), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE)
        x_mid = 0.5 * (boundaries[0] + boundaries[-1])
        ax.text(x_mid, -0.12, "inter-arrival (s)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.tick_params(labelbottom=False)

    if not show_ylabels:
        ax.tick_params(labelleft=False)

    for y in ax.get_yticks():
        if abs(y) < 1e-8:
            continue
        ax.axhline(y, lw=0.8, ls="--", color="black", alpha=0.15, zorder=0)

def _left_x_with_points(fig: plt.Figure, left: float, pad_pt: float) -> float:
    return max(0.0, left - pad_pt / (72.0 * fig.get_figwidth()))

def make_grid_figure(
    *,
    y_factory: Callable[[str], YOfFn],
    series: List[Any],
    legend_labels: List[str],
    color_of: Callable[[Any], Any],
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
    axes_cfg: Dict[str, AxisCfg],
    out_stem: str,
    mode_spacing: float,
    is_delta_grid: bool,
    values_are_differences: Optional[bool] = None,
    symmetric_count_rows: bool = False,
) -> None:
    if values_are_differences is None:
        values_are_differences = is_delta_grid
    figsize = GRID["delta_figsize"] if is_delta_grid else GRID["main_figsize"]
    left = GRID["delta_left"] if is_delta_grid else GRID["main_left"]
    top = GRID["delta_top"] if is_delta_grid else GRID["main_top"]
    legend_pad = GRID["legend_pad_delta"] if is_delta_grid else GRID["legend_pad_main"]
    legend_gap = GRID["legend_gap_delta"] if is_delta_grid else GRID["legend_gap_main"]
    legend_xoff = GRID["legend_xoff_delta"] if is_delta_grid else GRID["legend_xoff_main"]
    legend_ncol_colors = GRID["legend_ncol_colors_delta"] if is_delta_grid else GRID["legend_ncol_colors_main"]

    fig, axes = plt.subplots(nrows=len(GRID_ROWS), ncols=2, figsize=figsize, sharex=True)
    axes[0, 0].set_title(f"#priorities={priorities_cols[0]}", fontsize=PLOT_TITLE_FONTSIZE)
    axes[0, 1].set_title(f"#priorities={priorities_cols[1]}", fontsize=PLOT_TITLE_FONTSIZE)

    for r, row in enumerate(GRID_ROWS):
        y_of = y_factory(row.col)
        for c, prio in enumerate(priorities_cols):
            tick_strategy = row.tick_strategy
            if symmetric_count_rows and tick_strategy == "count":
                tick_strategy = "count_symmetric"
            elif values_are_differences and tick_strategy == "percent":
                tick_strategy = "percent_symmetric"

            _draw_points_axis(
                ax=axes[r, c],
                nodes_order=nodes_order,
                arrivals_order=arrivals_order,
                priorities=prio,
                series=series,
                y_of=y_of,
                axis_cfg=axes_cfg[row.axis_key],
                show_xlabels=row.bottom,
                show_ylabels=(c == 0),
                mode_spacing=mode_spacing,
                color_of=color_of,
                tick_strategy=tick_strategy,
            )

    fig.subplots_adjust(
        left=left, right=GRID["right"], bottom=GRID["bottom"], top=top,
        wspace=GRID["wspace"], hspace=GRID["hspace"]
    )

    # legends
    color_handles = color_patch_handles([color_of(s) for s in series], edge_width=0.6)
    shape_handles, shape_labels = shape_legend_handles(
        nodes_order=nodes_order,
        specs=SHAPE_LEGEND_SPECS,
        edge_width=GRID["shape_edge"],
    )

    bbox_l = axes[0, 0].get_position()
    bbox_r = axes[0, 1].get_position()
    x_center = 0.5 * (bbox_l.x0 + bbox_r.x1)
    y_top_grid = max(bbox_l.y1, bbox_r.y1)
    legend_y = y_top_grid + legend_pad

    legend_kwargs = dict(
        fontsize=PLOT_LEGEND_FONTSIZE,
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
        ncol=legend_ncol_colors,
    )
    center_two_legends(
        fig,
        left_handles=color_handles, left_labels=legend_labels,
        right_handles=shape_handles, right_labels=shape_labels,
        left_title="Colors", right_title="Shapes",
        y_top=legend_y, x_center=x_center, gap=legend_gap,
        legend_kwargs=legend_kwargs,
        title_fontsize=PLOT_LEGEND_FONTSIZE,
        x_offset=legend_xoff,
        left_ncol=legend_ncol_colors,  # keep colors as before
        right_ncol=1,                  # force shapes into one column
    )

    # shared y labels
    x_text = _left_x_with_points(fig, left, GRID["ylabel_pad_pt"])
    for r, row in enumerate(GRID_ROWS):
        bb = axes[r, 0].get_position()
        y = 0.5 * (bb.y0 + bb.y1)
        ylabel = row.delta_ylabel if values_are_differences and row.delta_ylabel else row.ylabel
        fig.text(
            x_text,
            y,
            ylabel,
            rotation=90,
            rotation_mode="anchor",
            multialignment="center",
            va="center",
            ha="center",
            fontsize=AXIS_LABEL_FONTSIZE,
        )

    save_figure(fig, OUT_FIGS_DIR / out_stem)

def make_main_grid_combined(
    *,
    df_seeds: pd.DataFrame,
    defpreempt: int,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
) -> None:
    # series = blocking then non-blocking for each mode
    series = [RowKey(mode=m, blocking=b, defpreempt=defpreempt) for m in MAIN_PLOT_MODE_NAMES for b in (1, 0)]

    # lighter color for non-blocking
    cmap: Dict[Tuple[str, int], Any] = {}
    for m in MAIN_PLOT_MODE_NAMES:
        base = rowkey_color(RowKey(mode=m, blocking=1, defpreempt=defpreempt))
        cmap[(m, 1)] = base
        cmap[(m, 0)] = lighter_color(base)

    def color_of(rk: RowKey):
        return cmap[(rk.mode, rk.blocking)]

    def y_factory(col: str) -> YOfFn:
        return lambda rk, n, a, p: values_from_seeds(df_seeds, nodes=n, priorities=p, arrival_s=a, rk=rk, col=col)

    make_grid_figure(
        y_factory=y_factory,
        series=series,
        legend_labels=[rowkey_label(rk) for rk in series],
        color_of=color_of,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_cols=priorities_cols,
        axes_cfg=Y_MAIN,
        out_stem=f"main_defaultpreempt={defpreempt}",
        mode_spacing=GRID["mode_spacing_main"],
        is_delta_grid=False,
        symmetric_count_rows=False,
    )

def make_main_grid_single_blocking(
    *,
    df_seeds: pd.DataFrame,
    defpreempt: int,
    blocking: int,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
) -> None:
    """
    Main-style grid but only for *one* blocking variant (blocking OR non-blocking),
    keeping the original combined plots as-is.
    """
    b = int(blocking)
    series = [RowKey(mode=m, blocking=b, defpreempt=defpreempt) for m in MAIN_PLOT_MODE_NAMES]

    # Use the same color indices for blocking and non-blocking (no lightening).
    cmap: Dict[str, Any] = {}
    for m in MAIN_PLOT_MODE_NAMES:
        cmap[m] = rowkey_color(RowKey(mode=m, blocking=1, defpreempt=defpreempt))

    def color_of(rk: RowKey):
        return cmap[rk.mode]

    def y_factory(col: str) -> YOfFn:
        return lambda rk, n, a, p: values_from_seeds(
            df_seeds, nodes=n, priorities=p, arrival_s=a, rk=rk, col=col
        )

    make_grid_figure(
        y_factory=y_factory,
        series=series,
        legend_labels=[rowkey_label(rk) for rk in series],
        color_of=color_of,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_cols=priorities_cols,
        axes_cfg=Y_MAIN,
        out_stem=f"main_defaultpreempt={defpreempt}_blocking={b}",
        mode_spacing=GRID["mode_spacing_delta"],
        is_delta_grid=True,
        values_are_differences=False,
        symmetric_count_rows=False,
    )


# Optional delta plots (kept available if you want them back)
def make_timing_delta_grid(
    *,
    df_delta_seeds: pd.DataFrame,
    defpreempt: int,
    blocking: int,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
) -> None:
    fake_series = list(range(len(DELTA_PLOT_SPECS)))

    def color_of(i: int):
        return PLOT_COLORS[DELTA_PLOT_SPECS[i].color_idx % len(PLOT_COLORS)]

    def y_factory(col: str) -> YOfFn:
        def _f(idx: int, n: int, a: float, p: int):
            name = DELTA_PLOT_SPECS[idx].label
            sub = df_delta_seeds[
                (df_delta_seeds["defpreempt"] == defpreempt)
                & (df_delta_seeds["blocking"] == blocking)
                & (df_delta_seeds["nodes"] == n)
                & (df_delta_seeds["priorities"] == p)
                & (df_delta_seeds["arrival_s"] == a)
                & (df_delta_seeds["delta_name"] == name)
            ][[SEED_COL, col]].sort_values(SEED_COL, kind="mergesort")
            return [float(v) for v in sub[col].tolist() if is_finite(v)]
        return _f

    suffix = "(blocking)" if blocking else "(non-blocking)"
    make_grid_figure(
        y_factory=y_factory,
        series=fake_series,
        legend_labels=[f"{d.label} {suffix}" for d in DELTA_PLOT_SPECS],
        color_of=color_of,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_cols=priorities_cols,
        axes_cfg=Y_DELTAS,
        out_stem=f"timing_deltas_defpreempt={defpreempt}_blocking={blocking}",
        mode_spacing=GRID["mode_spacing_delta"],
        is_delta_grid=True,
        symmetric_count_rows=True,
    )

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def latex_metric_table(
    *,
    out_path: Path,
    lookup: pd.DataFrame,
    metric: MetricSpec,
    defpreempt: int,
    priorities: int,
    nodes_order: List[int],
    arrivals_order: List[float],
) -> None:
    allowed = set(TABLE_METRIC_MODE_NAMES) if TABLE_METRIC_MODE_NAMES is not None else None
    modes = [
        RowKey(m.mode, m.blocking, defpreempt)
        for m in MODE_SPECS
        if allowed is None or m.mode in allowed
    ]
    modes = sort_rowkeys(modes)

    n_arrivals = len(arrivals_order)
    col_w = TABLE_COL_WIDTH_METRIC_P4 if (priorities == 4 and metric.name in {"latency", "deletions"}) else TABLE_COL_WIDTH_METRIC
    colspec = "l " + " ".join([" ".join([f"w{{c}}{{{col_w}}}"] * n_arrivals) for _ in nodes_order])

    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]

    node_hdrs = [rf"\multicolumn{{{n_arrivals}}}{{c}}{{\# Nodes =\,{n}}}" for n in nodes_order]
    lines.append(r"\multirow{2}{*}{\textbf{Trigger Mode}} & " + " & ".join(node_hdrs) + r" \\")
    lines.append(latex_cmidrules(len(nodes_order), n_arrivals))

    arrival_cells = repeat_header_values_with_first_label(
        len(nodes_order),
        arrivals_order,
        first_prefix_tex=r"Inter-arrival =\,",
        fmt=lambda a: fmt_arrival(a),
    )
    lines.append(" & " + " & ".join(arrival_cells) + r" \\")
    lines.append(r"\midrule")

    for rk in modes:
        cells: List[str] = []
        for n in nodes_order:
            for a in arrivals_order:
                m = lookup_main(lookup, nodes=n, priorities=priorities, arrival_s=a, rk=rk, col=metric.col_total)
                s = lookup_main(lookup, nodes=n, priorities=priorities, arrival_s=a, rk=rk, col=f"{metric.col_total}_std")
                total = metric.fmt(m, s)

                if metric.col_prio_pattern and priorities > 1:
                    parts = [rf"total & {total}"]
                    for p in range(1, MAX_PRIORITIES + 1):
                        c = metric.col_prio_pattern.format(p=p)
                        mv = lookup_main(lookup, nodes=n, priorities=priorities, arrival_s=a, rk=rk, col=c)
                        sv = lookup_main(lookup, nodes=n, priorities=priorities, arrival_s=a, rk=rk, col=f"{c}_std")
                        parts.append(rf"p{p} & {metric.fmt(mv, sv)}")
                    inner = r"\\".join(parts)
                    cell = rf"\begin{{tabular}}[t]{{@{{}}r@{{:\ }}l@{{}}}}{inner}\end{{tabular}}"
                else:
                    cell = total
                cells.append(cell)

        multiline = bool(metric.col_prio_pattern and priorities > 1)
        lines.append(f"{rowkey_table_label(rk, multiline=multiline)} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    metric_long = {
        "usage": r"effective resource usage (\%)",
        "latency": r"scheduling latency (ms)",
        "deletions": r"number of pod deletions",
        "optimizer_runs": f"number of {SOLVER_DISPLAY_NAME} runs",
        "plan_activations": r"number of activated improving plans",
        "optimal_solver_runs": r"solver runs certified as optimal (\%)",
        "proven_optimal_plans": r"improving plans certified as optimal (\%)",
    }[metric.name]
    priority_desc = "one priority" if priorities == 1 else f"{priorities} priorities"
    prio_note = ""
    if priorities > 1 and metric.name in {"latency", "deletions"}:
        prio_note = " p1 is the lowest priority."

    if metric.name == "optimal_solver_runs":
        caption = (
            f"Solver runs certified optimal (\\% of all solver runs), with "
            f"{_preempt_title(defpreempt)} and {priority_desc} (mean $\\pm$ std)."
        )
    elif metric.name == "proven_optimal_plans":
        caption = (
            f"Optimal among improving plans (\\%), with "
            f"{_preempt_title(defpreempt)} and {priority_desc} (mean $\\pm$ std)."
        )
    else:
        caption = (
            f"Mean paired differences in {metric_long} between the plugin with "
            f"{_preempt_title(defpreempt)} and the default scheduler for runs with "
            f"{priority_desc} (mean $\\pm$ std)."
            f"{prio_note}"
        )
    label = f"tab:{metric.name}-defpreempt{defpreempt}-prio{priorities}"

    write_latex_table(
        out_path, lines,
        caption=caption, label=label,
        placement=TABLE_PLACEMENT,
        environment=TABLE_METRIC_ENVIRONMENT,
        max_height=TABLE_METRIC_MAX_HEIGHT,
    )

def _fmt_mean_only(mean_v: object, *, signed: bool = False, decimals: int = 1) -> str:
    if not is_finite(mean_v):
        return nan_str()
    v = float(mean_v)
    return f"${v:+.{decimals}f}$" if signed else f"${v:.{decimals}f}$"

def latex_overview_table(
    *,
    out_path: Path,
    df_seeds: pd.DataFrame,
    defpreempt: int,
    arrivals_order: List[float],
) -> None:
    # subset modes shown
    summary_modes = [m for m in BASE_MODES if m[0] in set(BLOCKING_DIFF_MODE_NAMES)]
    df = df_seeds[df_seeds["defpreempt"] == int(defpreempt)].copy()
    df["acceptance_rate_pct"] = np.where(df["solver_attempts_mean"] > 0, 100.0 * df["plan_activated_mean"] / df["solver_attempts_mean"], np.nan)

    mode_labels: List[str] = []
    data: Dict[str, Dict[Tuple[int, float], Dict[str, str]]] = {}

    for mode, base_label, detail, _rank, _cidx in summary_modes:
        label = f"{base_label}, {detail}" if detail else base_label
        mode_labels.append(label)
        data[label] = {}
        for blocking in (1, 0):
            sub = df[(df["mode"] == mode) & (df["blocking"] == blocking)]
            for arr in arrivals_order:
                sub_arr = sub[sub["arrival_s"] == arr]
                vals: Dict[str, str] = {}
                for spec in DIFF_METRICS:
                    v = float(sub_arr[spec.col_total].mean()) if not sub_arr.empty else float("nan")
                    vals[spec.name] = _fmt_mean_only(v, signed=spec.mean_signed, decimals=spec.mean_dec)
                ar = float(sub_arr["acceptance_rate_pct"].mean()) if not sub_arr.empty else float("nan")
                vals["acceptance_rate"] = _fmt_mean_only(ar, decimals=1)
                data[label][(blocking, arr)] = vals

    n_arr = len(arrivals_order)
    cols_per_mode = 2 * n_arr
    colspec = "l " + " ".join([" ".join([f"w{{c}}{{{TABLE_COL_WIDTH_METRIC}}}"] * cols_per_mode) for _ in mode_labels])

    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]

    h1 = [r"\multirow{3}{*}{\textbf{Metric}}"]
    for label in mode_labels:
        h1.append(rf"\multicolumn{{{cols_per_mode}}}{{c}}{{\textbf{{{label}}}}}")
    lines.append(" & ".join(h1) + r" \\")
    lines.append(latex_cmidrules(len(mode_labels), cols_per_mode, start_col=2))

    h2 = [""]
    for _ in mode_labels:
        h2.append(rf"\multicolumn{{{n_arr}}}{{c}}{{Blocking}}")
        h2.append(rf"\multicolumn{{{n_arr}}}{{c}}{{Non-blocking}}")
    lines.append(" & ".join(h2) + r" \\")
    lines.append(latex_cmidrules(len(mode_labels) * 2, n_arr, start_col=2))

    h3 = [""] + repeat_header_values_with_first_label(
        groups=len(mode_labels) * 2,
        values=arrivals_order,
        first_prefix_tex=r"Inter-arrival =\,",
        fmt=lambda a: fmt_arrival(a),
    )
    lines.append(" & ".join(h3) + r" \\")
    lines.append(r"\midrule")

    rows = [
        ("usage", r"Diff. usage (\%)"),
        ("latency", r"Diff. latency (ms)"),
        ("deletions", r"Diff. pod deletions"),
        ("optimizer_runs", rf"Diff. {SOLVER_DISPLAY_NAME} runs"),
        ("plan_activations", r"Diff. activated improving plans"),
        ("optimal_solver_runs", r"Runs certified optimal (\%)"),
        ("proven_optimal_plans", r"Optimal among improving plans (\%)"),
        ("acceptance_rate", r"Acceptance rate (\%)"),
    ]
    for key, label in rows:
        cells = [label]
        for mlabel in mode_labels:
            for blocking in (1, 0):
                for a in arrivals_order:
                    cells.append(data[mlabel][(blocking, a)][key])
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    cap = (
        f"Overview of mean paired differences and plan acceptance rate per inter-arrival time, "
        f"aggregated over node counts and all priority configurations, for blocking and "
        f"non-blocking runs with {_preempt_title(defpreempt)}."
    )
    write_latex_table(
        out_path, lines, caption=cap, label=f"tab:overview-defpreempt{defpreempt}",
        placement=TABLE_PLACEMENT, environment=TABLE_METRIC_ENVIRONMENT,
    )

def _build_generic_diff_table(
    *,
    out_path: Path,
    df_seeds: pd.DataFrame,
    base_filter: Dict[str, Any],
    left_selector: Dict[str, Any],   # subtracted
    right_selector: Dict[str, Any],  # minuend
    title_suffix: str,
    caption: str,
    caption_short: str,
    label: str,
    mode_header_line: Callable[[str], str],
    mode_set: Sequence[str],
    arrivals_order: List[float],
) -> None:
    # rows = right - left
    df = df_seeds.copy()
    for k, v in base_filter.items():
        df = df[df[k] == v]

    merge_cols = ["nodes", "priorities", "arrival_s", SEED_COL]
    metric_cols = [m.col_total for m in DIFF_METRICS]

    mode_labels: List[str] = []
    mode_data: Dict[str, Dict[float, Dict[str, str]]] = {}

    for mode, base_label, detail, _rank, _cidx in BASE_MODES:
        if mode not in set(mode_set):
            continue

        lsel = dict(left_selector, mode=mode)
        rsel = dict(right_selector, mode=mode)

        d_left = df.copy()
        d_right = df.copy()
        for k, v in lsel.items():
            d_left = d_left[d_left[k] == v]
        for k, v in rsel.items():
            d_right = d_right[d_right[k] == v]

        merged = d_right[merge_cols + metric_cols].merge(
            d_left[merge_cols + metric_cols],
            on=merge_cols, how="inner", suffixes=("_R", "_L")
        )

        label_text = f"{base_label}, {detail}" if detail else base_label
        mode_labels.append(label_text)
        mode_data[label_text] = {}

        for a in arrivals_order:
            sub = merged[merged["arrival_s"] == a]
            vals: Dict[str, str] = {}
            for ms in DIFF_METRICS:
                col = ms.col_total
                diff = sub[f"{col}_R"] - sub[f"{col}_L"]
                vals[ms.name] = fmt_signed(float(diff.mean()) if not diff.empty else float("nan"), ms.mean_dec)
            mode_data[label_text][a] = vals

    n_arr = len(arrivals_order)
    colspec = "l " + " ".join([" ".join([f"w{{c}}{{{TABLE_COL_WIDTH}}}"] * n_arr) for _ in mode_labels])

    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]

    h1 = [r"\multirow{2}{*}{\textbf{Metric}}"]
    for label_text in mode_labels:
        h1.append(rf"\multicolumn{{{n_arr}}}{{c}}{{{mode_header_line(label_text)}}}")
    lines.append(" & ".join(h1) + r" \\")
    lines.append(latex_cmidrules(len(mode_labels), n_arr, start_col=2))

    h2 = [""] + repeat_header_values_with_first_label(
        groups=len(mode_labels),
        values=arrivals_order,
        first_prefix_tex=r"Inter-arrival =\,",
        fmt=lambda a: fmt_arrival(a),
    )
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    row_labels = {
        "usage": r"Diff. usage (\%)",
        "latency": r"Diff. latency (ms)",
        "deletions": r"Diff. pod deletions",
        "optimizer_runs": rf"Diff. {SOLVER_DISPLAY_NAME} runs",
        "plan_activations": r"Diff. activated improving plans",
        "optimal_solver_runs": r"Diff. runs certified optimal (\%)",
        "proven_optimal_plans": r"Diff. optimal among improving plans (\%)",
    }

    for ms in DIFF_METRICS:
        row = [row_labels[ms.name]]
        for label_text in mode_labels:
            for a in arrivals_order:
                row.append(mode_data[label_text][a][ms.name])
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    write_latex_table(
        out_path, lines, caption=caption, label=label,
        placement=TABLE_PLACEMENT, environment=TABLE_METRIC_ENVIRONMENT,
    )

def latex_blocking_diff_table(
    *,
    out_path: Path,
    df_seeds: pd.DataFrame,
    defpreempt: int,
    arrivals_order: List[float],
    priorities: Optional[int] = None,
) -> None:
    df = df_seeds[df_seeds["defpreempt"] == int(defpreempt)]
    if priorities is not None:
        df = df[df["priorities"] == int(priorities)]

    caption = (
        f"Mean difference (non-blocking $-$ blocking) per inter-arrival time, "
        f"aggregated over node counts and all priority configurations, with "
        f"{_preempt_title(defpreempt)}."
    )
    cap_short = f"Plugin Results: Non-blocking vs. Blocking with {_preempt_title(defpreempt)}."

    _build_generic_diff_table(
        out_path=out_path,
        df_seeds=df,
        base_filter={},
        left_selector={"blocking": 1},   # subtract blocking
        right_selector={"blocking": 0},  # non-blocking - blocking
        title_suffix="(non-blocking-blocking)",
        caption=caption,
        caption_short=cap_short,
        label=f"tab:blocking-diff-defpreempt{defpreempt}-prio{priorities if priorities is not None else 'all'}",
        mode_header_line=lambda lbl: rf"\makecell{{\textbf{{{lbl}}}\\(non-blocking\,$-$\,blocking)}}",
        mode_set=BLOCKING_DIFF_MODE_NAMES,
        arrivals_order=arrivals_order,
    )

def latex_defpreempt_diff_table(
    *,
    out_path: Path,
    df_seeds: pd.DataFrame,
    blocking: int,
    arrivals_order: List[float],
    priorities: Optional[int] = None,
) -> None:
    df = df_seeds[df_seeds["blocking"] == int(blocking)]
    if priorities is not None:
        df = df[df["priorities"] == int(priorities)]

    caption = (
        f"Mean difference (DefaultPreemption enabled $-$ disabled) per inter-arrival time, "
        f"aggregated over node counts and all priority configurations, for "
        f"{_blocking_title(blocking).lower()} runs."
    )
    cap_short = f"Plugin Results: DefaultPreemption Enabled vs. Disabled with {_blocking_title(blocking)}."

    _build_generic_diff_table(
        out_path=out_path,
        df_seeds=df,
        base_filter={},
        left_selector={"defpreempt": 0},
        right_selector={"defpreempt": 1},   # enabled - disabled
        title_suffix="(enabled-disabled)",
        caption=caption,
        caption_short=cap_short,
        label=f"tab:defpreempt-diff-blocking{blocking}",
        mode_header_line=lambda lbl: rf"\makecell{{\textbf{{{lbl}}}\\(enabled\,$-$\,disabled)}}",
        mode_set=BLOCKING_DIFF_MODE_NAMES,
        arrivals_order=arrivals_order,
    )

def latex_timing_diff_table(
    *,
    out_path: Path,
    df_seeds: pd.DataFrame,
    defpreempt: int,
    blocking: int,
    arrivals_order: List[float],
) -> None:
    df = df_seeds[(df_seeds["defpreempt"] == int(defpreempt)) & (df_seeds["blocking"] == int(blocking))].copy()

    merge_cols = ["nodes", "priorities", "arrival_s", SEED_COL]
    metric_cols = [m.col_total for m in DIFF_METRICS]

    comp_keys: List[Tuple[str, str]] = []
    comp_data: Dict[Tuple[str, str], Dict[float, Dict[str, str]]] = {}

    for baseline, variant, family, direction in TIMING_DELTA_SPECS:
        left = df[df["mode"] == baseline][merge_cols + metric_cols]
        right = df[df["mode"] == variant][merge_cols + metric_cols]
        merged = right.merge(left, on=merge_cols, suffixes=("_V", "_B"), how="inner")

        key = (family, direction)
        comp_keys.append(key)
        comp_data[key] = {}

        for a in arrivals_order:
            sub = merged[merged["arrival_s"] == a]
            vals: Dict[str, str] = {}
            for ms in DIFF_METRICS:
                diff = sub[f"{ms.col_total}_V"] - sub[f"{ms.col_total}_B"]
                vals[ms.name] = fmt_signed(float(diff.mean()) if not diff.empty else float("nan"), ms.mean_dec)
            comp_data[key][a] = vals

    families: List[Tuple[str, List[str]]] = []
    fam_idx: Dict[str, int] = {}
    for fam, dirn in comp_keys:
        if fam not in fam_idx:
            fam_idx[fam] = len(families)
            families.append((fam, []))
        families[fam_idx[fam]][1].append(dirn)

    n_arr = len(arrivals_order)
    colspec = "l " + " ".join([" ".join([f"w{{c}}{{{TABLE_COL_WIDTH}}}"] * n_arr) for _ in comp_keys])
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]

    # row 1: families
    h1 = [r"\multirow{3}{*}{\textbf{Metric}}"]
    cm1: List[str] = []
    col_cursor = 2
    for fam, dirs in families:
        span = len(dirs) * n_arr
        h1.append(rf"\multicolumn{{{span}}}{{c}}{{\textbf{{{fam} ({'blocking' if blocking else 'non-blocking'})}}}}")
        cm1.append(rf"\cmidrule(lr){{{col_cursor}-{col_cursor+span-1}}}")
        col_cursor += span
    lines.append(" & ".join(h1) + r" \\")
    lines.append("".join(cm1))

    # row 2: directions
    prefix_map = {"Periodic": "Interval", "Stable-queue": "Delay"}
    h2 = [""]
    for fam, dirs in families:
        pref = prefix_map.get(fam, "")
        for d in dirs:
            title = f"{pref}: {d}" if pref else d
            h2.append(rf"\multicolumn{{{n_arr}}}{{c}}{{{title}}}")
    lines.append(" & ".join(h2) + r" \\")
    lines.append(latex_cmidrules(len(comp_keys), n_arr, start_col=2))

    # row 3: inter-arrival
    h3 = [""] + repeat_header_values_with_first_label(
        groups=len(comp_keys),
        values=arrivals_order,
        first_prefix_tex=r"Inter-arrival =\,",
        fmt=lambda a: fmt_arrival(a),
    )
    lines.append(" & ".join(h3) + r" \\")
    lines.append(r"\midrule")

    row_labels = {
        "usage": r"Diff. usage (\%)",
        "latency": r"Diff. latency (ms)",
        "deletions": r"Diff. pod deletions",
        "optimizer_runs": rf"Diff. {SOLVER_DISPLAY_NAME} runs",
        "plan_activations": r"Diff. activated improving plans",
        "optimal_solver_runs": r"Diff. runs certified optimal (\%)",
        "proven_optimal_plans": r"Diff. optimal among improving plans (\%)",
    }
    for ms in DIFF_METRICS:
        row = [row_labels[ms.name]]
        for key in comp_keys:
            for a in arrivals_order:
                row.append(comp_data[key][a][ms.name])
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    caption = (
        f"Mean difference between 8s baseline and other timing variants per inter-arrival time, "
        f"aggregated over node counts and priority configurations, for "
        f"{_blocking_title(blocking).lower()} runs with {_preempt_title(defpreempt)}."
    )

    write_latex_table(
        out_path, lines,
        caption=caption,
        label=f"tab:timing-diff-defpreempt{defpreempt}-blocking{blocking}",
        placement=TABLE_PLACEMENT, environment=TABLE_METRIC_ENVIRONMENT,
    )

# Mean lifetime table (trace generator)
def load_mean_lifetime_data(traces_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for y in sorted(traces_dir.rglob("info_generate.yaml")):
        with open(y, "r", encoding="utf-8") as fh:
            info = yaml.safe_load(fh)
        gen = info.get("inputs", {}).get("generated", {})
        args = info.get("inputs", {}).get("args", {})
        rows.append({
            "nodes": int(args.get("num_nodes", 0)),
            "priorities": int(args.get("priority_max", 1)),
            "arrival_s": float(args.get("mean_arrival", 0)),
            "seed": str(args.get("seed", y.parent.name)),
            "mean_lifetime_s": float(gen.get("derived_mean_lifetime_s", float("nan"))),
        })
    return pd.DataFrame(rows)

def latex_mean_lifetime_table(
    *,
    out_path: Path,
    df_life: pd.DataFrame,
    priorities: int,
    nodes_order: List[int],
    arrivals_order: List[float],
    seeds_order: List[str],
) -> None:
    dff = df_life[df_life["priorities"] == int(priorities)].copy()
    dff = dff.set_index(["nodes", "arrival_s", "seed"]).sort_index()

    n_arr = len(arrivals_order)
    colspec = "p{7.2em} " + " ".join([" ".join([f"w{{c}}{{{TABLE_COL_WIDTH}}}"] * n_arr) for _ in nodes_order])

    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines.append(
        r"\multirow{2}{*}{\centering\textbf{Trace Seed}} & "
        + " & ".join([rf"\multicolumn{{{n_arr}}}{{c}}{{\# Nodes =\,{n}}}" for n in nodes_order])
        + r" \\"
    )
    lines.append(latex_cmidrules(len(nodes_order), n_arr))
    h2 = [""] + repeat_header_values_with_first_label(
        groups=len(nodes_order),
        values=arrivals_order,
        first_prefix_tex=r"Inter-arrival =\,",
        fmt=lambda a: fmt_arrival(a),
    )
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    for i, seed in enumerate(seeds_order, start=1):
        cells = []
        for n in nodes_order:
            for a in arrivals_order:
                try:
                    v = float(dff.at[(n, a, seed), "mean_lifetime_s"])
                except Exception:
                    v = float("nan")
                cells.append(f"{v:.1f}" if is_finite(v) else r"\text{--}")
        lines.append(rf"\centering {i} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    mean_cells: List[str] = []
    for n in nodes_order:
        for a in arrivals_order:
            vals = []
            for s in seeds_order:
                try:
                    v = float(dff.at[(n, a, s), "mean_lifetime_s"])
                    if is_finite(v):
                        vals.append(v)
                except Exception:
                    pass
            mean_cells.append(f"{float(np.mean(vals)):.1f}" if vals else r"\text{--}")
    lines.append(r"\centering\textbf{Mean} & " + " & ".join(mean_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    prio_short = "1 priority" if priorities == 1 else f"{priorities} priorities"
    caption = (
        f"Calibrated mean workload lifetime (s) used by the trace generator for runs with {prio_short}. "
        f"Rows correspond to trace seeds, and columns are grouped by number of nodes and inter-arrival time. "
        f"The final row reports the mean across seeds."
    )
    cap_short = f"Plugin Results: Calibrated Mean Lifetime for {prio_short}."

    write_latex_table(
        out_path, lines,
        caption=caption, caption_short=cap_short, label=f"tab:mean-lifetime-prio{priorities}",
        resizebox=False,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating trace-replayer tables/figures...")
    configure_matplotlib()

    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

    df_seeds = load_results_seeds(IN_RESULTS_SEEDS)
    df_mean_std = build_main_mean_std(df_seeds)
    lookup = build_main_lookup(df_mean_std)

    nodes_order = sorted(df_mean_std["nodes"].unique().tolist())
    arrivals_order = select_arrivals(df_mean_std["arrival_s"].unique().tolist())

    # delta datasets (kept for timing-diff tables and optional delta plots)
    df_delta_seeds = build_delta_seeds(df_seeds)
    df_delta_mean_std = build_delta_mean_std(df_delta_seeds)

    # variant-diff datasets (to match the diff tables)
    df_blockdiff_seeds = build_blocking_diff_seeds(df_seeds)
    df_defpreemptdiff_seeds = build_defpreempt_diff_seeds(df_seeds)

    produced_tables: List[Path] = []
    produced_figs: List[Path] = []

    # Metric tables
    for ms in METRICS_ALL:
        for dp in (0, 1):
            for prio in PRIORITIES_TO_SHOW:
                out = OUT_TABLES_DIR / f"table_defpreempt={dp}_priorities={prio}_{ms.name}.tex"
                latex_metric_table(
                    out_path=out, lookup=lookup, metric=ms, defpreempt=dp, priorities=prio,
                    nodes_order=nodes_order, arrivals_order=arrivals_order
                )
                produced_tables.append(out)

    # Blocking diff tables
    for dp in (0, 1):
        out = OUT_TABLES_DIR / f"table_blocking_diff_defpreempt={dp}.tex"
        latex_blocking_diff_table(out_path=out, df_seeds=df_seeds, defpreempt=dp, arrivals_order=arrivals_order, priorities=None)
        produced_tables.append(out)

    # DefaultPreemption diff tables
    for blocking in (0, 1):
        out = OUT_TABLES_DIR / f"table_defpreempt_diff_blocking={blocking}.tex"
        latex_defpreempt_diff_table(out_path=out, df_seeds=df_seeds, blocking=blocking, arrivals_order=arrivals_order, priorities=None)
        produced_tables.append(out)

    # Timing diff tables
    for dp in (0, 1):
        for blocking in (0, 1):
            out = OUT_TABLES_DIR / f"table_timing_diff_defpreempt={dp}_blocking={blocking}.tex"
            latex_timing_diff_table(out_path=out, df_seeds=df_seeds, defpreempt=dp, blocking=blocking, arrivals_order=arrivals_order)
            produced_tables.append(out)

    # Overview tables
    for dp in (0, 1):
        out = OUT_TABLES_DIR / f"table_overview_defaultpreempt={dp}.tex"
        latex_overview_table(out_path=out, df_seeds=df_seeds, defpreempt=dp, arrivals_order=arrivals_order)
        produced_tables.append(out)

    # Mean lifetime tables
    df_life = load_mean_lifetime_data(TRACES_DIR)
    if not df_life.empty:
        seeds_order = sorted(df_life["seed"].unique().tolist())
        for prio in PRIORITIES_TO_SHOW:
            out = OUT_TABLES_DIR / f"table_mean_lifetime_priorities={prio}.tex"
            latex_mean_lifetime_table(
                out_path=out, df_life=df_life, priorities=prio,
                nodes_order=nodes_order, arrivals_order=arrivals_order,
                seeds_order=seeds_order,
            )
            produced_tables.append(out)

    # Main combined grids (same output stems as your current script)
    for dp in (1, 0):
        make_main_grid_combined(
            df_seeds=df_seeds, defpreempt=dp,
            nodes_order=nodes_order, arrivals_order=arrivals_order,
            priorities_cols=PRIORITIES_TO_SHOW,
        )
        produced_figs.append(OUT_FIGS_DIR / f"main_defaultpreempt={dp}.pdf")

    # NEW: split main plots by blocking variant (keep combined plots too)
    for dp in (1, 0):
        for b in (1, 0):
            make_main_grid_single_blocking(
                df_seeds=df_seeds,
                defpreempt=dp,
                blocking=b,
                nodes_order=nodes_order,
                arrivals_order=arrivals_order,
                priorities_cols=PRIORITIES_TO_SHOW,
            )
            produced_figs.append(OUT_FIGS_DIR / f"main_defaultpreempt={dp}_blocking={b}.pdf")

    # ---------------------------------------------------------------------
    # main plots for the diff results
    # ---------------------------------------------------------------------

    # (A) Non-blocking vs. blocking (non-blocking - blocking), per defpreempt
    for dp in (0, 1):
        make_blocking_diff_grid(
            df_blockdiff_seeds=df_blockdiff_seeds,
            defpreempt=dp,
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities_cols=PRIORITIES_TO_SHOW,
        )
        produced_figs.append(OUT_FIGS_DIR / f"blocking_diffs_defpreempt={dp}.pdf")

    # (B) DefaultPreemption enabled vs disabled (enabled - disabled), per blocking
    for b in (0, 1):
        make_defpreempt_diff_grid(
            df_defpreemptdiff_seeds=df_defpreemptdiff_seeds,
            blocking=b,
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities_cols=PRIORITIES_TO_SHOW,
        )
        produced_figs.append(OUT_FIGS_DIR / f"defpreempt_diffs_blocking={b}.pdf")

    # (C) Timing diffs (variant - 8s baseline), per defpreempt and blocking
    for dp in (0, 1):
        for b in (0, 1):
            make_timing_delta_grid(
                df_delta_seeds=df_delta_seeds,
                defpreempt=dp,
                blocking=b,
                nodes_order=nodes_order,
                arrivals_order=arrivals_order,
                priorities_cols=PRIORITIES_TO_SHOW,
            )
            produced_figs.append(OUT_FIGS_DIR / f"timing_deltas_defpreempt={dp}_blocking={b}.pdf")


    print("Tables:")
    for p in produced_tables:
        print(f"  - {p}")
    print("Figures:")
    for p in produced_figs:
        print(f"  - {p}")


if __name__ == "__main__":
    main()