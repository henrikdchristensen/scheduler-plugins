#!/usr/bin/env python3
# scripts/kwok_workload_once/plots_and_tables.py
"""
python -m scripts.kwok_workload_once.plots_and_tables
python -m scripts.kwok_workload_once.plots_and_tables --solver gurobi
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from scripts.helpers.data_helpers import is_finite, safe_div
from scripts.helpers.plot_config import (
    PLOT_AXIS_LABEL_FONTSIZE,
    PLOT_FORMATS,
    PLOT_LEGEND_COLUMN_SPACING,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_HANDLE_LENGTH,
    PLOT_LEGEND_HANDLE_TEXT_PAD,
    PLOT_TICK_FONTSIZE,
    PLOT_TITLE_FONTSIZE,
)
from scripts.helpers.plot_helpers import (
    center_two_legends,
    color_patch_handles,
    configure_matplotlib,
    save_figure,
)
from scripts.helpers.table_helpers import (
    fmt_pct,
    latex_cmidrules,
    nan_str,
    write_latex_table,
)

#################################################################
# CONFIG (constants)
#################################################################

DEFAULT_RESULTS_ROOT = Path("analysis/kwok_workload_once")
DEFAULT_SOLVER = "cp_sat"
ALL_SOLVERS = ["cp_sat", "gurobi"]

REQUIRED_DISRUPTION_COLUMNS = {
    "n_default_optimal",
    "n_solver_optimal",
    "n_solver_feasible",
    "moves_pct_sum_default_optimal",
    "evictions_pct_sum_default_optimal",
    "moves_pct_sum_solver_optimal",
    "evictions_pct_sum_solver_optimal",
    "moves_pct_sum_solver_feasible",
    "evictions_pct_sum_solver_feasible",
}
OPTIMAL_PURPLE = "#9467bd"

# Fixed plot/table dimensions
PLOT_PPNS = [4, 8]
PLOT_PRIORITIES = [1, 2, 4]
PLOT_TIMEOUTS = [1, 10, 20]
PLOT_NODES = [4, 8, 16, 32]
PLOT_UTILS = [90, 95, 100, 105]

EPS = 1e-9
ANNOT_FS = 5.0

# Labels
TARGET_UTIL_LABEL = "target usage (%)"
TARGET_UTIL_TABLE_LABEL = r"Target usage (\%)"
NODES_PLOT_LABEL = "#nodes"
NODES_TABLE_LABEL = r"\# Nodes"
INSTANCES_LABEL = "% of instances"
PODS_PER_NODE_PLOT_LABEL = "pods/node"
PODS_PER_NODE_TABLE_LABEL = "Pods/Node"
PRIORITIES_PLOT_LABEL = "#priorities"
PRIORITIES_TABLE_LABEL = r"\# Priorities"
SOLVER_TIMEOUT_LABEL = "Optimizer timeout"

# Disruption plot style
GRID_2D_CELL_FIGSIZE_DISRUPT = (1.5, 1.0)
GRID_2D_LEFT_DISRUPT = 0.15
GRID_2D_YLABEL_XPOS_DISRUPT = 0.01
GRID_2D_TOP_DISRUPT = 0.80
GRID_2D_LEGEND_PAD_DISRUPT = 0.33
GRID_2D_LEGEND_GAP_DISRUPT = 0.012
GRID_2D_ROWLABEL_PAD_DISRUPT = 8.0
HATCH_MOVES = "xxxxxxxxxx"
HATCH_MOVES_LEGEND = "xxxxxx"
HATCH_LINEWIDTH = 0.25
DISRUPT_YLIM = (0.0, 40.0)
DISRUPT_YTICKS = [0, 10, 20, 30, 40]
GRID_2D_WSPACE_DISRUPT = 0.10   # horizontal gap between panels
GRID_2D_HSPACE_DISRUPT = 0.15   # vertical gap between panels


# 2D grid sizes
GRID_2D_CELL_FIGSIZE = (2.6, 1.6)
GRID_2D_BAR_WIDTH = 0.88
GRID_2D_WSPACE = 0.06
GRID_2D_HSPACE = 0.10
GRID_2D_LEFT = 0.1
GRID_2D_RIGHT = 0.995
GRID_2D_BOTTOM = 0.0
GRID_2D_TOP = 0.88
GRID_2D_YLABEL_XPOS = 0.017

# 3D sizes
FIGSIZE_3D = (8, 4)
BAR_WIDTH_3D = 0.13
ELEV_3D, AZIM_3D = 20.0, -54.0

# Dot-grid sizes
DOT_CELL_FIGSIZE = (1.5, 1.0)
DOT_MARKER_SIZE = 4.0
DOT_MARKER_LINEWIDTH = 0.5
DOT_WSPACE = 0.12
DOT_HSPACE = 0.16
DOT_LEFT = 0.14
DOT_RIGHT = 0.99
DOT_BOTTOM = 0.07
DOT_TOP = 0.82
DOT_YLABEL_XPOS = 0.02

DOT_YLIM_DIFF_USAGE: Optional[Tuple[float, float]] = (-5.0, 15.0)
DOT_YLIM_SOLVER_DUR: Optional[Tuple[float, float]] = (0.0, 25.0)
DOT_NTICKS_DIFF_USAGE: Optional[int] = 5
DOT_NTICKS_SOLVER_DUR: Optional[int] = 6
DOT_YLABEL_X_DIFF: float = 0.026
DOT_YLABEL_PAD_DIFF: float = 7.0

# Colors
set2 = plt.get_cmap("Set2").colors
set3 = plt.get_cmap("Set3").colors
TIMEOUT_COLORS = {t: set2[i % len(set2)] for i, t in enumerate(PLOT_TIMEOUTS)}

CATEGORIES = [
    {"key": "other", "label": "Other", "col": "other_rate", "color": set2[3]},
    {"key": "solver_optimal", "label": "Better&Optimal", "col": "solver_optimal_rate", "color": set2[0]},
    {"key": "solver_feasible", "label": "Better", "col": "solver_feasible_rate", "color": set2[1]},
    {"key": "default_optimal", "label": "KWOK Optimal", "col": "default_optimal_rate", "color": set2[2]},
    {"key": "default_all", "label": "No Calls", "col": "default_all_running_rate", "color": set3[11]},
    {"key": "solver_failed", "label": "Failures", "col": "solver_failed_rate", "color": set2[7]},
]

OUTCOME_BY_USAGE_ROWS: List[Tuple[str, str]] = [
    (r"Failures (\%)", "solver_failed_rate"),
    (r"No Calls (\%)", "default_all_running_rate"),
    (r"KWOK Optimal (\%)", "default_optimal_rate"),
    (r"Better (\%)", "solver_feasible_rate"),
    (r"Better\&Optimal (\%)", "solver_optimal_rate"),
]

METRIC_ROWS: List[Tuple[str, str, float, int, bool]] = [
    (r"Optimizer\,duration\,(s)", "solver_duration_ms_mean", 1.0 / 1000.0, 1, False),
    (r"Diff.\,eff.\,usage\,(\%)", "eff_delta_mean", 100.0, 1, True),
]

OUTCOME_ROWS: List[Tuple[str, str]] = [
    (r"Failures (\%)", "solver_failed_rate"),
    (r"No Calls (\%)", "default_all_running_rate"),
    (r"KWOK Optimal (\%)", "default_optimal_rate"),
    (r"Better (\%)", "solver_feasible_rate"),
    (r"Better\&Optimal (\%)", "solver_optimal_rate"),
]

DISRUPTION_ROWS: List[Tuple[str, str, float, int]] = [
    (r"Better: moves (\%)", "moves_pct_mean_per_better", 1.0, 1),
    (r"Optimal: moves (\%)", "moves_pct_mean_per_optimal", 1.0, 1),
    (r"Better: evictions (\%)", "evictions_pct_mean_per_better", 1.0, 1),
    (r"Optimal: evictions (\%)", "evictions_pct_mean_per_optimal", 1.0, 1),
]

#################################################################
# Small shared helpers
#################################################################


@dataclass(frozen=True)
class SolverPaths:
    df_path: Path
    figures_dir: Path
    tables_dir: Path


def get_paths_for_solver(solver: str, results_root: Path = DEFAULT_RESULTS_ROOT) -> SolverPaths:
    return SolverPaths(
        df_path=results_root / f"results_per_combo_{solver}.csv",
        figures_dir=results_root / "figures" / solver,
        tables_dir=results_root / "tables" / solver,
    )


def _legend_kwargs() -> dict:
    return dict(
        fontsize=PLOT_LEGEND_FONTSIZE,
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
    )


def _require_columns(df: pd.DataFrame, required: Iterable[str], *, context: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns in {context}: {', '.join(sorted(missing))}"
        )


def _set_row_ylabel(ax: plt.Axes, ppn: int, *, labelpad: float) -> None:
    lbl = ax.set_ylabel(
        f"{PODS_PER_NODE_PLOT_LABEL}={ppn}",
        fontsize=PLOT_AXIS_LABEL_FONTSIZE,
        labelpad=labelpad,
    )
    lbl.set_va("center")
    lbl.set_ha("center")
    lbl.set_linespacing(1.8)


def _make_category_legend_handles_labels(seen_keys: set[str], *, linewidth: float):
    legends = [s for s in CATEGORIES if s["key"] in seen_keys][::-1]
    handles = color_patch_handles([s["color"] for s in legends], edge_width=linewidth)
    labels = [s["label"] for s in legends]
    return handles, labels


def _select_present_order(values: Sequence[int], present: Iterable[int]) -> List[int]:
    present_set = {int(v) for v in present}
    return [int(v) for v in values if int(v) in present_set]


def _append_prio_ppn_nodes_header(
    lines: List[str],
    *,
    lead_header: str,
    lead_cols: int,
    prio_order: Sequence[int],
    ppn_order: Sequence[int],
    nodes_order: Sequence[int],
) -> None:
    n_nodes = len(nodes_order)
    n_ppn = len(ppn_order)
    cols_per_prio = n_ppn * n_nodes

    # row 1: priorities
    prio_cells = [
        rf"\multicolumn{{{cols_per_prio}}}{{c}}{{{PRIORITIES_TABLE_LABEL} =\,{prio}}}"
        for prio in prio_order
    ]
    lines.append(lead_header + " & ".join(prio_cells) + r" \\")
    lines.append(latex_cmidrules(len(prio_order), cols_per_prio, start_col=lead_cols + 1))

    # row 2: pods per node
    lead_prefix = " & " * lead_cols
    ppn_cells: List[str] = []
    for _ in prio_order:
        for ppn in ppn_order:
            ppn_cells.append(rf"\multicolumn{{{n_nodes}}}{{c}}{{{PODS_PER_NODE_TABLE_LABEL} =\,{ppn}}}")
    lines.append(lead_prefix + " & ".join(ppn_cells) + r" \\")
    lines.append(latex_cmidrules(len(prio_order) * len(ppn_order), n_nodes, start_col=lead_cols + 1))

    # row 3: nodes
    node_cells: List[str] = []
    first = True
    for _ in prio_order:
        for _ in ppn_order:
            for n in nodes_order:
                if first:
                    node_cells.append(rf"\llap{{{NODES_TABLE_LABEL} =\,}}{n}")
                    first = False
                else:
                    node_cells.append(f"{n}")
    lines.append(lead_prefix + " & ".join(node_cells) + r" \\")
    lines.append(r"\midrule")


def _prepare_timeout_indexed_table(
    *,
    df_table: pd.DataFrame,
    timeout: int,
    priorities_list: Sequence[int],
    value_cols: Sequence[str],
    breaker_col: Optional[str],
    ppns: Sequence[int],
    nodes: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, List[int], List[int], List[int]]:
    dff = df_table[df_table["timeout_s"].astype(int) == int(timeout)].copy()
    dff = dff[dff["priorities"].astype(int).isin([int(p) for p in priorities_list])]

    nodes_present = dff["nodes"].dropna().astype(int).unique().tolist() if "nodes" in dff.columns else []
    ppn_present = dff["pods_per_node"].dropna().astype(int).unique().tolist() if "pods_per_node" in dff.columns else []
    prio_present = dff["priorities"].dropna().astype(int).unique().tolist() if "priorities" in dff.columns else []

    nodes_order = _select_present_order(list(nodes) if nodes is not None else sorted(nodes_present), nodes_present)
    ppn_order = _select_present_order(ppns, ppn_present)
    prio_order = _select_present_order(priorities_list, prio_present)

    if not nodes_order or not ppn_order or not prio_order:
        return pd.DataFrame(), [], [], []

    idx_cols: List[str] = ["priorities", "timeout_s"]
    if breaker_col and breaker_col in dff.columns:
        idx_cols.append(breaker_col)
    idx_cols.extend(["pods_per_node", "nodes"])

    for c in ["timeout_s", "nodes", "pods_per_node", "priorities"]:
        if c in dff.columns:
            dff[c] = pd.to_numeric(dff[c], errors="coerce").round().astype("Int64")
    if breaker_col and breaker_col in dff.columns:
        dff[breaker_col] = pd.to_numeric(dff[breaker_col], errors="coerce").round().astype("Int64")

    dff = dff.dropna(subset=idx_cols).copy()
    for c in idx_cols:
        dff[c] = dff[c].astype(int)

    keep_cols = idx_cols + [c for c in value_cols if c in dff.columns]
    dff = dff[keep_cols].drop_duplicates(subset=idx_cols, keep="last").set_index(idx_cols).sort_index()
    return dff, nodes_order, ppn_order, prio_order


def _build_breaker_sections(
    *,
    dff_indexed: pd.DataFrame,
    timeout: int,
    breaker_col: Optional[str],
):
    def make_lookup(breaker_val: Optional[int]):
        def get_value(prio: int, ppn: int, node: int, col: str) -> float:
            try:
                if breaker_val is None:
                    key = (int(prio), int(timeout), int(ppn), int(node))
                else:
                    key = (int(prio), int(timeout), int(breaker_val), int(ppn), int(node))
                return float(dff_indexed.at[key, col])
            except Exception:
                return float("nan")
        return get_value

    if breaker_col and breaker_col in dff_indexed.index.names:
        vals = sorted(int(v) for v in dff_indexed.index.get_level_values(breaker_col).unique())
        return [(f"{v}\\%", make_lookup(v)) for v in vals]
    return [(None, make_lookup(None))]


def _render_timeout_breakdown_table(
    *,
    out_path: Path,
    sections,
    prio_order: Sequence[int],
    ppn_order: Sequence[int],
    nodes_order: Sequence[int],
    row_specs: Sequence[Tuple[str, str]],
    cell_formatter: Callable[[str, float], str],
    lead_left_label: str,
    lead_row_label: str,
    caption: str,
    caption_short: Optional[str],
    label: str,
) -> None:
    has_breaker = any(section_title is not None for section_title, _ in sections)
    n_nodes = len(nodes_order)
    n_ppn = len(ppn_order)
    n_prio = len(prio_order)
    cols_per_prio = n_ppn * n_nodes
    data_cols = n_prio * cols_per_prio

    if has_breaker:
        lead_cols = 2
        colspec = "c l" + " c" * data_cols
        lead_header = rf"\multirow{{3}}{{*}}{{\textbf{{{lead_left_label}}}}} & \multirow{{3}}{{*}}{{\textbf{{{lead_row_label}}}}} & "
    else:
        lead_cols = 1
        colspec = "l" + " c" * data_cols
        lead_header = rf"\multirow{{3}}{{*}}{{\textbf{{{lead_row_label}}}}} & "

    lines: List[str] = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    _append_prio_ppn_nodes_header(
        lines,
        lead_header=lead_header,
        lead_cols=lead_cols,
        prio_order=prio_order,
        ppn_order=ppn_order,
        nodes_order=nodes_order,
    )

    first_section = True
    for section_title, get_value in sections:
        if not first_section:
            lines.append(r"\midrule")
        first_section = False

        for row_idx, (row_label, col) in enumerate(row_specs):
            cells: List[str] = []
            for prio in prio_order:
                for ppn in ppn_order:
                    for n in nodes_order:
                        cells.append(cell_formatter(col, get_value(prio, ppn, n, col)))

            if has_breaker:
                if row_idx == 0:
                    lead = rf"\multirow{{{len(row_specs)}}}{{*}}{{{section_title}}}" + " & " + row_label
                else:
                    lead = " & " + row_label
            else:
                lead = row_label

            lines.append(f"{lead} & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex_table(out_path, lines, caption=caption, caption_short=caption_short, label=label)


#################################################################
# Aggregation helpers
#################################################################


def _aggregate_counts_to_rates(per_combo_df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """
    Aggregate by `keys` (sum counts/sums) and compute rate columns.
    """
    df = per_combo_df.copy()
    _require_columns(df, REQUIRED_DISRUPTION_COLUMNS, context="sealed results CSV")

    df["eff_delta_sum"] = np.maximum(df["cpu_delta_sum"], df["mem_delta_sum"])

    g = (
        df.groupby(keys, as_index=False).agg(
            {
                "n_seeds": "sum",
                "n_seeds_not_all_running": "sum",
                "n_default_all_running": "sum",
                "n_solver_called": "sum",
                "n_solver_solution": "sum",
                "n_solver_failed": "sum",
                "n_default_optimal": "sum",
                "n_solver_optimal": "sum",
                "n_solver_feasible": "sum",
                "n_solver_improve": "sum",
                "n_other": "sum",
                "solver_duration_ms_sum": "sum",
                "cpu_delta_sum": "sum",
                "mem_delta_sum": "sum",
                "eff_delta_sum": "sum",
                "moves_pct_sum_default_optimal": "sum",
                "evictions_pct_sum_default_optimal": "sum",
                "moves_pct_sum_solver_optimal": "sum",
                "evictions_pct_sum_solver_optimal": "sum",
                "moves_pct_sum_solver_feasible": "sum",
                "evictions_pct_sum_solver_feasible": "sum",
            }
        )
    )

    g["default_all_running_rate"] = safe_div(g["n_default_all_running"], g["n_seeds"])
    g["solver_called_rate"] = safe_div(g["n_solver_called"], g["n_seeds"])
    g["solver_failed_rate"] = safe_div(g["n_solver_failed"], g["n_seeds"])
    g["default_optimal_rate"] = safe_div(g["n_default_optimal"], g["n_seeds"])
    g["solver_optimal_rate"] = safe_div(g["n_solver_optimal"], g["n_seeds"])
    g["solver_feasible_rate"] = safe_div(g["n_solver_feasible"], g["n_seeds"])
    g["solver_improve_rate"] = safe_div(g["n_solver_improve"], g["n_seeds"])
    g["other_rate"] = safe_div(g["n_other"], g["n_seeds"])

    g["solver_duration_ms_mean"] = safe_div(g["solver_duration_ms_sum"], g["n_solver_called"])
    g["cpu_delta_mean"] = safe_div(g["cpu_delta_sum"], g["n_seeds"])
    g["mem_delta_mean"] = safe_div(g["mem_delta_sum"], g["n_seeds"])
    g["eff_delta_mean"] = safe_div(g["eff_delta_sum"], g["n_seeds"])

    g["moves_pct_mean_per_better"] = safe_div(
        g["moves_pct_sum_solver_feasible"], g["n_solver_feasible"]
    )
    g["evictions_pct_mean_per_better"] = safe_div(
        g["evictions_pct_sum_solver_feasible"], g["n_solver_feasible"]
    )

    # Combined Optimal = KWOK Optimal + Better&Optimal
    g["n_optimal_combined"] = g["n_default_optimal"] + g["n_solver_optimal"]
    g["moves_pct_sum_optimal_combined"] = (
        g["moves_pct_sum_default_optimal"] + g["moves_pct_sum_solver_optimal"]
    )
    g["evictions_pct_sum_optimal_combined"] = (
        g["evictions_pct_sum_default_optimal"] + g["evictions_pct_sum_solver_optimal"]
    )

    g["moves_pct_mean_per_optimal"] = safe_div(
        g["moves_pct_sum_optimal_combined"], g["n_optimal_combined"]
    )
    g["evictions_pct_mean_per_optimal"] = safe_div(
        g["evictions_pct_sum_optimal_combined"], g["n_optimal_combined"]
    )
    return g.copy()


def aggregate_over_util_and_timeout(per_combo_df: pd.DataFrame) -> pd.DataFrame:
    return _aggregate_counts_to_rates(per_combo_df, ["pods_per_node", "priorities", "nodes"])


def aggregate_over_util(per_combo_df: pd.DataFrame) -> pd.DataFrame:
    return _aggregate_counts_to_rates(per_combo_df, ["pods_per_node", "priorities", "timeout_s", "nodes"])


def aggregate_keep_util(per_combo_df: pd.DataFrame) -> pd.DataFrame:
    return _aggregate_counts_to_rates(per_combo_df, ["util", "pods_per_node", "priorities", "timeout_s", "nodes"])


def aggregate_over_all_except_util(per_combo_df: pd.DataFrame) -> pd.DataFrame:
    return _aggregate_counts_to_rates(per_combo_df, ["util"])


#################################################################
# Tables
#################################################################


def write_metric_table_tex(
    *,
    df_table: pd.DataFrame,
    out_path: Path,
    ppns: List[int],
    nodes: List[int],
    timeout: int,
    priorities_list: List[int],
    breaker_col: Optional[str] = None,
    caption: str = "",
    caption_short: Optional[str] = None,
    label: str = "",
) -> None:
    metric_cols = [col for _, col, _, _, _ in METRIC_ROWS]
    dff, nodes_order, ppn_order, prio_order = _prepare_timeout_indexed_table(
        df_table=df_table,
        timeout=timeout,
        priorities_list=priorities_list,
        value_cols=metric_cols,
        breaker_col=breaker_col,
        ppns=ppns,
        nodes=nodes,
    )
    if dff.empty:
        out_path.write_text("% empty: no data after filters\n", encoding="utf-8")
        print(f"[warn] metric table empty -> {out_path}")
        return

    sections = _build_breaker_sections(dff_indexed=dff, timeout=timeout, breaker_col=breaker_col)

    row_specs = [(row_label, col) for row_label, col, _, _, _ in METRIC_ROWS]

    metric_meta = {
        col: (scale, decimals, signed)
        for _, col, scale, decimals, signed in METRIC_ROWS
    }

    def fmt_metric(col: str, raw: float) -> str:
        scale, decimals, signed = metric_meta[col]
        if not is_finite(raw):
            return nan_str()
        v = float(raw) * float(scale)
        if signed:
            if abs(v) < 5e-13:
                v = 0.0
            return f"{v:+.{decimals}f}"
        return f"{v:.{decimals}f}"

    _render_timeout_breakdown_table(
        out_path=out_path,
        sections=sections,
        prio_order=prio_order,
        ppn_order=ppn_order,
        nodes_order=nodes_order,
        row_specs=row_specs,
        cell_formatter=fmt_metric,
        lead_left_label="Usage",
        lead_row_label="Metric",
        caption=caption,
        caption_short=caption_short,
        label=label,
    )


def write_outcome_table_tex(
    *,
    df_table: pd.DataFrame,
    out_path: Path,
    ppns: List[int],
    timeout: int,
    priorities_list: List[int],
    breaker_col: Optional[str] = None,
    decimals: int = 1,
    caption: str = "",
    caption_short: Optional[str] = None,
    label: str = "",
) -> None:
    outcome_cols = [c for _, c in OUTCOME_ROWS]
    dff, nodes_order, ppn_order, prio_order = _prepare_timeout_indexed_table(
        df_table=df_table,
        timeout=timeout,
        priorities_list=priorities_list,
        value_cols=outcome_cols,
        breaker_col=breaker_col,
        ppns=ppns,
        nodes=None,
    )
    if dff.empty:
        out_path.write_text("% empty: no data after filters\n", encoding="utf-8")
        print(f"[warn] table empty -> {out_path}")
        return

    sections = _build_breaker_sections(dff_indexed=dff, timeout=timeout, breaker_col=breaker_col)

    def fmt_outcome(col: str, raw: float) -> str:
        if is_finite(raw):
            return fmt_pct(100.0 * float(raw), decimals=decimals)
        # preserve original fallback behavior
        if col == "default_all_running_rate":
            return fmt_pct(100.0, decimals=decimals)
        return fmt_pct(0.0, decimals=decimals)

    _render_timeout_breakdown_table(
        out_path=out_path,
        sections=sections,
        prio_order=prio_order,
        ppn_order=ppn_order,
        nodes_order=nodes_order,
        row_specs=OUTCOME_ROWS,
        cell_formatter=fmt_outcome,
        lead_left_label="Usage",
        lead_row_label="Outcome",
        caption=caption,
        caption_short=caption_short,
        label=label,
    )


def write_outcome_table_by_usage_tex(
    *,
    df_table: pd.DataFrame,
    out_path: Path,
    utils: List[int] = PLOT_UTILS,
    decimals: int = 1,
    caption: str = "",
    caption_short: Optional[str] = None,
    label: str = "",
) -> None:
    dff = df_table.copy()
    if "util" not in dff.columns:
        out_path.write_text("% empty: missing util column\n", encoding="utf-8")
        print(f"[warn] usage table empty -> {out_path}")
        return

    dff["util"] = pd.to_numeric(dff["util"], errors="coerce").round().astype("Int64")
    dff = dff.dropna(subset=["util"]).copy()
    dff["util"] = dff["util"].astype(int)

    keep_cols = ["util"] + [c for _, c in OUTCOME_BY_USAGE_ROWS]
    dff = dff[[c for c in keep_cols if c in dff.columns]].drop_duplicates(subset=["util"], keep="last")
    dff = dff.set_index("util").sort_index()

    util_order = [u for u in utils]

    def get_rate(util_val: int, col: str) -> float:
        try:
            return float(dff.at[int(util_val), col])
        except Exception:
            return float("nan")

    n_utils = len(util_order)
    lines: List[str] = [
        r"\begin{tabular}{" + "l" + " c" * n_utils + "}",
        r"\toprule",
    ]

    lines.append(
        rf"\multirow{{2}}{{*}}{{\textbf{{Outcome}}}} & "
        rf"\multicolumn{{{n_utils}}}{{c}}{{\textbf{{{TARGET_UTIL_TABLE_LABEL}}}}} \\"
    )
    lines.append(latex_cmidrules(1, n_utils, start_col=2))
    lines.append(" & " + " & ".join([rf"{u}\%" for u in util_order]) + r" \\")
    lines.append(r"\midrule")

    for row_label, col in OUTCOME_BY_USAGE_ROWS:
        cells = []
        for u in util_order:
            r_ = get_rate(u, col)
            cells.append(fmt_pct(100.0 * r_, decimals=decimals) if is_finite(r_) else nan_str())
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex_table(out_path, lines, caption=caption, caption_short=caption_short, label=label)


def write_disruption_table_tex(
    *,
    df_table: pd.DataFrame,
    out_path: Path,
    ppns: List[int],
    nodes: List[int],
    priorities_list: List[int],
    caption: str = "",
    caption_short: Optional[str] = None,
    label: str = "",
) -> None:
    dff = df_table.copy()
    dff = dff[dff["priorities"].astype(int).isin(priorities_list)]

    nodes_order = _select_present_order(nodes, dff["nodes"].dropna().astype(int).unique().tolist())
    ppn_order = _select_present_order(ppns, dff["pods_per_node"].dropna().astype(int).unique().tolist())
    prio_order = _select_present_order(priorities_list, dff["priorities"].dropna().astype(int).unique().tolist())

    if not nodes_order or not ppn_order or not prio_order:
        out_path.write_text("% empty: no data after filters\n", encoding="utf-8")
        print(f"[warn] disruption table empty -> {out_path}")
        return

    idx_cols = ["priorities", "pods_per_node", "nodes"]
    for c in idx_cols:
        dff[c] = pd.to_numeric(dff[c], errors="coerce").round().astype("Int64")
    dff = dff.dropna(subset=idx_cols).copy()
    for c in idx_cols:
        dff[c] = dff[c].astype(int)

    keep_cols = idx_cols + [c for _, c, _, _ in DISRUPTION_ROWS]
    dff = dff[keep_cols].drop_duplicates(subset=idx_cols, keep="last").set_index(idx_cols).sort_index()

    def get_val(prio: int, ppn: int, node: int, col: str) -> float:
        try:
            return float(dff.at[(int(prio), int(ppn), int(node)), col])
        except Exception:
            return float("nan")

    n_nodes = len(nodes_order)
    n_ppn = len(ppn_order)
    cols_per_prio = n_nodes * n_ppn
    data_cols = len(prio_order) * cols_per_prio

    lines: List[str] = [r"\begin{tabular}{" + "l" + " c" * data_cols + "}", r"\toprule"]
    _append_prio_ppn_nodes_header(
        lines,
        lead_header=rf"\multirow{{3}}{{*}}{{\textbf{{Disruption metric}}}} & ",
        lead_cols=1,
        prio_order=prio_order,
        ppn_order=ppn_order,
        nodes_order=nodes_order,
    )

    for row_label, col, scale, decimals in DISRUPTION_ROWS:
        cells = []
        for prio in prio_order:
            for ppn in ppn_order:
                for n in nodes_order:
                    raw = get_val(prio, ppn, n, col)
                    cells.append(f"{raw * scale:.{decimals}f}" if is_finite(raw) else nan_str())
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex_table(out_path, lines, caption=caption, caption_short=caption_short, label=label)


#################################################################
# Plots
#################################################################


def _draw_centered_dual_legend(
    *,
    fig: plt.Figure,
    axes_top_row: Sequence[plt.Axes],
    left_handles,
    left_labels,
    right_handles,
    right_labels,
    left_title: str,
    right_title: str,
    pad_above_axes: float,
    gap: float,
    left_ncol: int = 1,
    right_ncol: int = 1,
    extra_legend_kwargs: Optional[dict] = None,
) -> None:
    bbox_l = axes_top_row[0].get_position()
    bbox_r = axes_top_row[-1].get_position()
    x_center = 0.5 * (bbox_l.x0 + bbox_r.x1)
    y_top_grid = max(ax.get_position().y1 for ax in axes_top_row)

    legend_kwargs = _legend_kwargs()
    if extra_legend_kwargs:
        legend_kwargs.update(extra_legend_kwargs)

    center_two_legends(
        fig,
        left_handles=left_handles,
        left_labels=left_labels,
        right_handles=right_handles,
        right_labels=right_labels,
        left_title=left_title,
        right_title=right_title,
        y_top=y_top_grid + pad_above_axes,
        x_center=x_center,
        gap=gap,
        legend_kwargs=legend_kwargs,
        title_fontsize=PLOT_LEGEND_FONTSIZE,
        left_ncol=left_ncol,
        right_ncol=right_ncol,
    )


def plot_2d_grid_moves_evictions_better_vs_optimal(
    df_agg: pd.DataFrame,
    ppns: List[int],
    priorities: List[int],
    out_path: Path,
    cell_figsize: Tuple[float, float],
    *,
    fixed_nodes: List[int] = PLOT_NODES,
) -> None:
    nrows, ncols = len(ppns), len(priorities)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * cell_figsize[0], nrows * cell_figsize[1]),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig.supylabel(
        "disruption (% of total pods)",
        fontsize=PLOT_AXIS_LABEL_FONTSIZE,
        x=GRID_2D_YLABEL_XPOS_DISRUPT,
        y=(GRID_2D_BOTTOM + GRID_2D_TOP_DISRUPT) / 2,
    )

    cat_color = {c["key"]: c["color"] for c in CATEGORIES}
    series = [
        {
            "label": "Better",
            "key": "solver_feasible",
            "color": cat_color["solver_feasible"],
            "moves_col": "moves_pct_mean_per_better",
            "evictions_col": "evictions_pct_mean_per_better",
        },
        {
            "label": "Optimal",
            "key": "optimal_combined",
            "color": OPTIMAL_PURPLE,
            "moves_col": "moves_pct_mean_per_optimal",
            "evictions_col": "evictions_pct_mean_per_optimal",
        },
    ]

    with mpl.rc_context({"hatch.linewidth": HATCH_LINEWIDTH}):
        for r, ppn in enumerate(ppns):
            for c, prio in enumerate(priorities):
                ax = axes[r][c]
                panel = df_agg[
                    (df_agg["pods_per_node"] == ppn)
                    & (df_agg["priorities"] == prio)
                ].copy()

                x = np.arange(len(fixed_nodes))
                bars_per_group = 2
                group_width = 0.60
                width = group_width / bars_per_group
                offsets = [(j - (bars_per_group - 1) / 2.0) * width for j in range(bars_per_group)]

                if not panel.empty:
                    panel = panel.set_index("nodes")

                for j, s in enumerate(series):
                    xj = x + offsets[j]

                    if not panel.empty:
                        moves = pd.to_numeric(panel.get(s["moves_col"], pd.Series(dtype=float)), errors="coerce")
                        moves = moves.reindex(fixed_nodes).fillna(0.0).values
                        evictions = pd.to_numeric(panel.get(s["evictions_col"], pd.Series(dtype=float)), errors="coerce")
                        evictions = evictions.reindex(fixed_nodes).fillna(0.0).values
                    else:
                        moves = np.zeros(len(fixed_nodes))
                        evictions = np.zeros(len(fixed_nodes))

                    total = moves + evictions

                    ax.bar(xj, total, width=width, color=s["color"], edgecolor="black", linewidth=0.35, zorder=2)
                    ax.bar(
                        xj,
                        moves,
                        width=width,
                        bottom=0.0,
                        color=s["color"],
                        edgecolor="white",
                        linewidth=0.0,
                        hatch=HATCH_MOVES,
                        zorder=3,
                    )

                    for xi, m, e in zip(xj, moves, evictions):
                        if (m > EPS) and (e > EPS):
                            ax.plot(
                                [xi - width / 2.0, xi + width / 2.0],
                                [m, m],
                                color="white",
                                linewidth=HATCH_LINEWIDTH,
                                solid_capstyle="butt",
                                zorder=3.5,
                            )

                    ax.bar(xj, total, width=width, facecolor="none", edgecolor="black", linewidth=0.35, zorder=4)

                ax.set_xticks(x)
                ax.set_xticklabels([str(n) for n in fixed_nodes], fontsize=PLOT_TICK_FONTSIZE)
                ax.set_ylim(*DISRUPT_YLIM)
                ax.set_yticks(DISRUPT_YTICKS)
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=0))
                ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)
                ax.grid(axis="y", linewidth=0.4, alpha=0.4)

                if r == 0:
                    ax.set_title(rf"{PRIORITIES_PLOT_LABEL}={prio}", fontsize=PLOT_TITLE_FONTSIZE)
                if c == 0:
                    _set_row_ylabel(ax, ppn, labelpad=GRID_2D_ROWLABEL_PAD_DISRUPT)
                if r == nrows - 1:
                    ax.set_xlabel(NODES_PLOT_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    fig.subplots_adjust(
        left=GRID_2D_LEFT_DISRUPT,
        right=GRID_2D_RIGHT,
        bottom=GRID_2D_BOTTOM,
        top=GRID_2D_TOP_DISRUPT,
        wspace=GRID_2D_WSPACE_DISRUPT,
        hspace=GRID_2D_HSPACE_DISRUPT,
    )

    LEGEND_BORDER_LW = 0.4
    color_handles = color_patch_handles([s["color"] for s in series], edge_width=LEGEND_BORDER_LW)
    color_labels = [s["label"] for s in series]

    hatched_fill = mpatches.Rectangle((0, 0), 1, 1, fc="0.75", ec="none")
    hatched_hatch = mpatches.Rectangle((0, 0), 1, 1, fc="none", ec="white", linewidth=0.0, hatch=HATCH_MOVES_LEGEND)
    hatched_outline = mpatches.Rectangle((0, 0), 1, 1, fc="none", ec="black", linewidth=LEGEND_BORDER_LW)

    fill_handles = [
        (hatched_fill, hatched_hatch, hatched_outline),
        mpatches.Rectangle((0, 0), 1, 1, fc="0.75", ec="black", linewidth=LEGEND_BORDER_LW),
    ]
    fill_labels = ["Hatched = moves", "Solid = evictions"]

    _draw_centered_dual_legend(
        fig=fig,
        axes_top_row=list(axes[0, :]),
        left_handles=color_handles,
        left_labels=color_labels,
        right_handles=fill_handles,
        right_labels=fill_labels,
        left_title="Colors",
        right_title="Fill",
        pad_above_axes=GRID_2D_LEGEND_PAD_DISRUPT,
        gap=GRID_2D_LEGEND_GAP_DISRUPT,
        left_ncol=1,
        right_ncol=1,
        extra_legend_kwargs={"handler_map": {tuple: HandlerTuple(ndivide=1, pad=0.0)}},
    )

    save_figure(fig, out_path)


def plot_2d_grid_ppn_prio_with_aggregated_util(
    df_util_agg: pd.DataFrame,
    ppns: List[int],
    priorities: List[int],
    out_path: Path,
    cell_figsize: Tuple[float, float],
    *,
    fixed_nodes: List[int] = PLOT_NODES,
    fixed_timeouts: List[int] = PLOT_TIMEOUTS,
) -> None:
    nrows, ncols = len(ppns), len(priorities)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * cell_figsize[0], nrows * cell_figsize[1]),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig.supylabel(
        INSTANCES_LABEL,
        fontsize=PLOT_AXIS_LABEL_FONTSIZE,
        x=GRID_2D_YLABEL_XPOS,
        y=(GRID_2D_BOTTOM + GRID_2D_TOP) / 2,
    )

    seen_keys: set[str] = set()

    for r, ppn in enumerate(ppns):
        for c, prio in enumerate(priorities):
            ax = axes[r][c]
            panel = df_util_agg[
                (df_util_agg["pods_per_node"] == ppn) & (df_util_agg["priorities"] == prio)
            ].copy()

            nodes_vals = fixed_nodes
            timeouts = fixed_timeouts

            if not panel.empty:
                panel = panel.groupby(["nodes", "timeout_s"], as_index=False)[
                    [s["col"] for s in CATEGORIES if s["col"].endswith("_rate")]
                ].mean()

            bars_per_group = max(1, len(timeouts))
            width = GRID_2D_BAR_WIDTH / bars_per_group
            x = np.arange(len(nodes_vals))

            for j, timeout in enumerate(timeouts):
                sub = panel[panel["timeout_s"] == timeout].set_index("nodes") if not panel.empty else pd.DataFrame()
                xj = x + (j - (bars_per_group - 1) / 2.0) * width
                bar_height = np.zeros(len(nodes_vals), dtype=float)

                for category in CATEGORIES:
                    if not sub.empty:
                        vals = (
                            sub.get(category["col"], pd.Series(0.0, index=sub.index))
                            .reindex(nodes_vals)
                            .fillna(0.0)
                            .values
                        )
                    else:
                        vals = np.zeros(len(nodes_vals))
                    h = vals * 100.0
                    if np.any(h > EPS):
                        ax.bar(
                            xj,
                            h,
                            width=width,
                            bottom=bar_height,
                            color=category["color"],
                            edgecolor="black",
                            linewidth=0.35,
                        )
                        bar_height += h
                        seen_keys.add(category["key"])

                for xi, top in zip(xj, bar_height):
                    if top > EPS:
                        ax.text(float(xi), float(top) + 1.0, f"{int(timeout)}s", ha="center", va="bottom", fontsize=ANNOT_FS)

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in nodes_vals], fontsize=PLOT_TICK_FONTSIZE)
            ax.set_ylim(0, 110)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(100.0))
            ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)

            if r == 0:
                ax.set_title(rf"{PRIORITIES_PLOT_LABEL}={prio}", fontsize=PLOT_TITLE_FONTSIZE)
            if c == 0:
                _set_row_ylabel(ax, ppn, labelpad=10.0)
            if r == nrows - 1:
                ax.set_xlabel(NODES_PLOT_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    legend_handles, legend_labels = _make_category_legend_handles_labels(seen_keys, linewidth=0.4)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.02),
        ncol=len(legend_labels),
        **_legend_kwargs(),
    )

    fig.subplots_adjust(
        left=GRID_2D_LEFT,
        right=GRID_2D_RIGHT,
        bottom=GRID_2D_BOTTOM,
        top=GRID_2D_TOP,
        wspace=GRID_2D_WSPACE,
        hspace=GRID_2D_HSPACE,
    )
    save_figure(fig, out_path)


def plot_3d_ppn_prio_timeout(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    fixed_utils: List[int] = PLOT_UTILS,
    fixed_nodes: List[int] = PLOT_NODES,
) -> None:
    utils = fixed_utils
    nodes = fixed_nodes

    def get_rate(df_: pd.DataFrame, util_val, nodes_val, col):
        df_idx = df_.set_index(["util", "nodes"])
        try:
            return float(df_idx.loc[(util_val, nodes_val), col])
        except KeyError:
            return 0.0

    x_index = {r: i for i, r in enumerate(utils)}
    y_index = {n: j for j, n in enumerate(nodes)}

    fig, ax = plt.subplots(figsize=FIGSIZE_3D, subplot_kw={"projection": "3d"})
    ax.view_init(elev=ELEV_3D, azim=AZIM_3D)
    ax.set_proj_type("ortho")

    ax.set_xlim(0, len(utils))
    ax.set_ylim(0, len(nodes))
    ax.set_zlim(0, 100)

    ax.set_xticks([x_index[r] + 0.5 for r in utils])
    ax.set_yticks([y_index[n] + 0.5 for n in nodes])
    ax.set_zticks([0, 20, 40, 60, 80, 100])
    ax.zaxis.set_major_formatter(mtick.PercentFormatter(100.0))

    ax.set_xticklabels([f"{int(round(r))}%" for r in utils], fontsize=PLOT_TICK_FONTSIZE)
    ax.set_yticklabels([str(n) for n in nodes], fontsize=PLOT_TICK_FONTSIZE)

    ax.tick_params(axis="x", labelsize=PLOT_TICK_FONTSIZE, pad=-2)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE, pad=-2)
    ax.tick_params(axis="z", labelsize=PLOT_TICK_FONTSIZE, pad=-1)

    ax.set_xlabel(TARGET_UTIL_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=-4.0)
    ax.set_ylabel(NODES_PLOT_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=-5.5)
    ax.set_zlabel(INSTANCES_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=-5.0)
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE, y=1.01, pad=0)

    dx = max(0.05, min(1.0, BAR_WIDTH_3D))
    dy = max(0.05, min(1.0, BAR_WIDTH_3D))

    seen_keys: set[str] = set()
    for util in utils:
        for n in nodes:
            x0 = x_index[util] + (1 - dx) / 2
            y0 = y_index[n] + (1 - dy) / 2
            z = 0.0

            rate_solver_opt = get_rate(df, util, n, "solver_optimal_rate")
            rate_solver_feas = get_rate(df, util, n, "solver_feasible_rate")
            rate_solver_fail = get_rate(df, util, n, "solver_failed_rate")
            rate_default_opt = get_rate(df, util, n, "default_optimal_rate")
            rate_default_all = get_rate(df, util, n, "default_all_running_rate")
            rate_other = get_rate(df, util, n, "other_rate")

            for key, rate_ in [
                ("other", rate_other),
                ("solver_optimal", rate_solver_opt),
                ("solver_feasible", rate_solver_feas),
                ("default_optimal", rate_default_opt),
                ("default_all", rate_default_all),
                ("solver_failed", rate_solver_fail),
            ]:
                h = rate_ * 100.0
                if h > EPS:
                    color = next(s["color"] for s in CATEGORIES if s["key"] == key)
                    ax.bar3d(x0, y0, z, dx, dy, h, color=color, edgecolor="black", linewidth=0.35, shade=False)
                    z += h
                    seen_keys.add(key)

    legend_handles, legend_labels = _make_category_legend_handles_labels(seen_keys, linewidth=0.6)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.545, 0.92),
        ncol=len(legend_labels),
        **_legend_kwargs(),
    )

    save_figure(fig, out_path)


GRID_2D_SINGLE_FIGSIZE = (2.8, 2.0)
GRID_2D_SINGLE_BAR_WIDTH = 0.55


def plot_2d_single_aggregated_by_util(
    df_agg: pd.DataFrame,
    out_path: Path,
    *,
    fixed_utils: List[int] = PLOT_UTILS,
    figsize: Tuple[float, float] = GRID_2D_SINGLE_FIGSIZE,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(fixed_utils))
    width = GRID_2D_SINGLE_BAR_WIDTH
    bar_height = np.zeros(len(fixed_utils), dtype=float)
    seen_keys: set[str] = set()

    df_idx = df_agg.set_index("util") if not df_agg.empty else pd.DataFrame()

    for category in CATEGORIES:
        if not df_idx.empty and category["col"] in df_idx.columns:
            vals = df_idx[category["col"]].reindex(fixed_utils).fillna(0.0).values
        else:
            vals = np.zeros(len(fixed_utils))
        h = vals * 100.0
        if np.any(h > EPS):
            ax.bar(
                x,
                h,
                width=width,
                bottom=bar_height,
                color=category["color"],
                edgecolor="black",
                linewidth=0.35,
                label=category["label"],
            )
            bar_height += h
            seen_keys.add(category["key"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{u}%" for u in fixed_utils], fontsize=PLOT_TICK_FONTSIZE)
    ax.set_xlabel(TARGET_UTIL_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100.0))
    ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)
    ax.set_ylabel(INSTANCES_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    legend_handles, legend_labels = _make_category_legend_handles_labels(seen_keys, linewidth=0.4)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.48, 0.96),
        ncol=min(len(legend_labels), 5),
        **_legend_kwargs(),
    )

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.15, top=0.82)
    save_figure(fig, out_path)


def plot_grid(
    df_util_agg: pd.DataFrame,
    *,
    metric_col: str,
    y_label: str,
    ppns: List[int],
    priorities: List[int],
    out_path: Path,
    cell_figsize: Tuple[float, float] = DOT_CELL_FIGSIZE,
    fixed_nodes: List[int] = PLOT_NODES,
    fixed_timeouts: List[int] = PLOT_TIMEOUTS,
    y_scale: float = 1.0,
    y_lim: Optional[Tuple[float, float]] = None,
    y_nticks: Optional[int] = None,
    y_label_x: float = DOT_YLABEL_XPOS,
    row_label_pad: float = 10.0,
    show_zero_line: bool = True,
) -> None:
    nrows, ncols = len(ppns), len(priorities)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * cell_figsize[0], nrows * cell_figsize[1]),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x = np.arange(len(fixed_nodes))
    n_timeouts = len(fixed_timeouts)
    jitter_width = 0.6
    offsets = np.linspace(-jitter_width / 2, jitter_width / 2, n_timeouts) if n_timeouts > 1 else np.array([0.0])

    for r, ppn in enumerate(ppns):
        for c, prio in enumerate(priorities):
            ax = axes[r][c]
            panel = df_util_agg[
                (df_util_agg["pods_per_node"] == ppn)
                & (df_util_agg["priorities"] == prio)
            ].copy()

            for j, timeout in enumerate(fixed_timeouts):
                sub = panel[panel["timeout_s"] == timeout].set_index("nodes") if not panel.empty else pd.DataFrame()
                vals = (
                    sub[metric_col].reindex(fixed_nodes).values
                    if (not sub.empty and metric_col in sub.columns)
                    else np.full(len(fixed_nodes), np.nan)
                )

                ax.plot(
                    x + offsets[j],
                    vals * y_scale,
                    marker="D",
                    linestyle="None",
                    markersize=DOT_MARKER_SIZE,
                    markerfacecolor=TIMEOUT_COLORS.get(timeout, "grey"),
                    markeredgecolor="black",
                    markeredgewidth=DOT_MARKER_LINEWIDTH,
                    zorder=5,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in fixed_nodes], fontsize=PLOT_TICK_FONTSIZE)
            ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)
            ax.grid(axis="y", linewidth=0.4, alpha=0.4)

            if show_zero_line:
                ax.axhline(0.0, linewidth=0.8, color="black", linestyle="-", alpha=0.7)

            if y_lim is not None:
                ax.set_ylim(y_lim)

            if y_nticks is not None and y_nticks >= 2:
                lo, hi = ax.get_ylim()
                ticks = np.linspace(lo, hi, y_nticks)
                if all(abs(t - round(t)) < 1e-9 for t in ticks):
                    ticks = [int(round(t)) for t in ticks]
                ax.set_yticks(ticks)

            for bx in [xi - 0.5 for xi in range(len(fixed_nodes) + 1)]:
                ax.axvline(bx, linewidth=0.8, color="black", linestyle="--", alpha=0.7, zorder=0)
            ax.set_xlim(-0.5, len(fixed_nodes) - 0.5)

            if r == 0:
                ax.set_title(rf"{PRIORITIES_PLOT_LABEL}={prio}", fontsize=PLOT_TITLE_FONTSIZE)
            if c == 0:
                _set_row_ylabel(ax, ppn, labelpad=row_label_pad)
            if r == nrows - 1:
                ax.set_xlabel(NODES_PLOT_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    fig.supylabel(y_label, fontsize=PLOT_AXIS_LABEL_FONTSIZE, x=y_label_x)

    legend_handles = [
        Line2D(
            [0], [0],
            marker="D",
            linestyle="None",
            markersize=DOT_MARKER_SIZE,
            markerfacecolor=TIMEOUT_COLORS.get(t, "grey"),
            markeredgecolor="black",
            markeredgewidth=DOT_MARKER_LINEWIDTH,
        )
        for t in fixed_timeouts
    ]
    legend_labels = [f"{SOLVER_TIMEOUT_LABEL}={t}s" for t in fixed_timeouts]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.02),
        ncol=len(legend_labels),
        **_legend_kwargs(),
    )

    fig.subplots_adjust(
        left=DOT_LEFT,
        right=DOT_RIGHT,
        bottom=DOT_BOTTOM,
        top=DOT_TOP,
        wspace=DOT_WSPACE,
        hspace=DOT_HSPACE,
    )

    save_figure(fig, out_path)


#################################################################
# Main
#################################################################


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate plots and tables from solver results")
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        choices=ALL_SOLVERS,
        help="Solver name (determines input/output paths). If omitted, runs all solvers.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory for results",
    )
    args = parser.parse_args(argv)

    solvers = [args.solver] if args.solver else ALL_SOLVERS
    configure_matplotlib()

    for solver in solvers:
        _run_solver(solver, args.results_root)


def _run_solver(solver: str, results_root: Path) -> None:
    paths = get_paths_for_solver(solver, results_root)

    print(f"Generating tables and figures for solver={solver}...")
    print(f"  Input: {paths.df_path}")
    print(f"  Figures: {paths.figures_dir}")
    print(f"  Tables: {paths.tables_dir}")

    if not paths.df_path.exists():
        print(f"[error] Input file not found: {paths.df_path}")
        print(f"  Run: python -m scripts.kwok_workload_once.seal_results --solver-dir plugin-{solver}")
        return

    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    df_per_combo = pd.read_csv(paths.df_path)
    _require_columns(df_per_combo, REQUIRED_DISRUPTION_COLUMNS, context=str(paths.df_path))

    produced_tables: List[Path] = []
    produced_figs: List[Path] = []

    # --- TABLES: outcomes (per timeout, priorities side-by-side)
    df_table = aggregate_keep_util(df_per_combo)
    present_prios = set(df_table["priorities"].astype(int).unique().tolist())
    prio_list = [p for p in PLOT_PRIORITIES if p in present_prios]

    for t in PLOT_TIMEOUTS:
        out_tex = paths.tables_dir / f"table_outcomes_timeout={int(t)}.tex"
        write_outcome_table_tex(
            df_table=df_table,
            out_path=out_tex,
            ppns=PLOT_PPNS,
            timeout=int(t),
            priorities_list=prio_list,
            breaker_col="util",
            decimals=1,
            caption_short=f"Optimizer Results: Outcome Breakdown at {t}\\,s Optimizer Timeout",
            caption=(
                f"Outcome breakdown at {t}\\,s optimizer timeout. "
                f"Values are reported as percentages of the 100 paired instances "
                f"(optimizer vs.\\ default scheduler) for each parameter configuration. "
                f"Rows are grouped by target usage levels, with outcome categories listed "
                f"within each usage block. Columns are grouped by number of "
                f"priority levels, pods per node, and number of nodes."
            ),
            label=f"tab:outcomes-timeout{t}",
        )
        produced_tables.append(out_tex)

    # --- TABLES: metrics (per timeout)
    df_metric_keep_util = aggregate_keep_util(df_per_combo)
    for t in PLOT_TIMEOUTS:
        out_tex_metric = paths.tables_dir / f"table_metrics_timeout={int(t)}.tex"
        write_metric_table_tex(
            df_table=df_metric_keep_util,
            out_path=out_tex_metric,
            ppns=PLOT_PPNS,
            nodes=PLOT_NODES,
            timeout=int(t),
            priorities_list=prio_list,
            breaker_col="util",
            caption_short=f"Optimizer Results: Performance Metrics at {t}\\,s Optimizer Timeout",
            caption=(
                f"Performance metrics at {t}\\,s optimizer timeout. "
                f"Values are reported for the 100 paired instances (optimizer vs.\\ default scheduler) "
                f"for each parameter configuration. diff. eff. usage is reported as the mean paired difference "
                f"(optimizer minus default scheduler), while optimizer duration is the mean optimizer runtime "
                f"over instances where the optimizer is called (not a paired difference). "
                f"Rows are grouped by target usage levels, with metrics listed within each usage block. "
                f"Columns are grouped by number of priority levels, pods per node, and number of nodes. "
                f"Optimizer duration can slightly exceed the timeout because the timeout applies to solving only; "
                f"the reported time also includes solution extraction and I/O."
            ),
            label=f"tab:metrics-timeout{t}",
        )
        produced_tables.append(out_tex_metric)

    # --- TABLE: disruptions (aggregated over util + timeout)
    df_disrupt_table = aggregate_over_util_and_timeout(df_per_combo)
    out_tex_disrupt = paths.tables_dir / "table_disruptions_agg_util_timeout.tex"
    write_disruption_table_tex(
        df_table=df_disrupt_table,
        out_path=out_tex_disrupt,
        ppns=PLOT_PPNS,
        nodes=PLOT_NODES,
        priorities_list=prio_list,
        caption_short="Optimizer Results: Disruption Breakdown",
        caption=(
            "Disruption breakdown aggregated over target usage levels and optimizer timeouts. "
            "Values are average disruption percentages (\\% of total pods) conditioned on outcome category. "
            "The Better rows are averaged over Better instances, while the Optimal rows are averaged over "
            "combined Optimal instances (KWOK Optimal + Better\\&Optimal). These values are not paired "
            "differences relative to the default scheduler. Rows list disruption metrics (moves and evictions) "
            "for the two categories. Columns are grouped by number of priority levels, pods per node, and number of nodes."
        ),
        label="tab:disruptions-agg-util-timeout",
    )
    produced_tables.append(out_tex_disrupt)

    # --- Aggregated datasets for plots
    df_util_agg = aggregate_over_util(df_per_combo)
    df_util_only = aggregate_over_all_except_util(df_per_combo)

    # --- PLOT: outcomes by usage (single panel)
    out_2d_util = paths.figures_dir / "2d_by_usage"
    plot_2d_single_aggregated_by_util(df_agg=df_util_only, out_path=out_2d_util)
    produced_figs.extend([out_2d_util.with_suffix(f".{ext}") for ext in PLOT_FORMATS])

    # --- PLOT: main 2D grid
    out_2d = paths.figures_dir / "2d_main"
    plot_2d_grid_ppn_prio_with_aggregated_util(
        df_util_agg=df_util_agg,
        ppns=PLOT_PPNS,
        priorities=PLOT_PRIORITIES,
        out_path=out_2d,
        cell_figsize=GRID_2D_CELL_FIGSIZE,
    )
    produced_figs.extend([out_2d.with_suffix(f".{ext}") for ext in PLOT_FORMATS])

    # --- TABLE: outcomes by usage
    out_tex_usage = paths.tables_dir / "table_outcomes_by_usage.tex"
    write_outcome_table_by_usage_tex(
        df_table=df_util_only,
        out_path=out_tex_usage,
        utils=PLOT_UTILS,
        decimals=1,
        caption_short="Optimizer Results: Outcome Breakdown by Usage Levels",
        caption=(
            "Outcome breakdown by target usage levels, aggregated over number of nodes, pods per node, "
            "priority levels, and optimizer timeouts. Values are reported as percentages of instances "
            "in each outcome category after aggregation across those dimensions. "
            "Rows list the outcome categories, and columns correspond to target usage levels."
        ),
        label="tab:outcomes-by-usage",
    )
    produced_tables.append(out_tex_usage)

    # --- PLOT: diff. usage (dot grid)
    out_dot_usage = paths.figures_dir / "diff_usage"
    plot_grid(
        df_util_agg,
        metric_col="eff_delta_mean",
        y_label="diff. usage (%)",
        ppns=PLOT_PPNS,
        priorities=PLOT_PRIORITIES,
        out_path=out_dot_usage,
        y_scale=100.0,
        y_lim=DOT_YLIM_DIFF_USAGE,
        y_nticks=DOT_NTICKS_DIFF_USAGE,
        y_label_x=DOT_YLABEL_X_DIFF,
        row_label_pad=DOT_YLABEL_PAD_DIFF,
    )
    produced_figs.extend([out_dot_usage.with_suffix(f".{ext}") for ext in PLOT_FORMATS])

    # --- PLOT: disruption composition (moves+evictions)
    df_disrupt_agg = aggregate_over_util_and_timeout(df_per_combo)
    out_disrupt = paths.figures_dir / "moves_evictions_better_vs_optimal"
    plot_2d_grid_moves_evictions_better_vs_optimal(
        df_agg=df_disrupt_agg,
        ppns=PLOT_PPNS,
        priorities=PLOT_PRIORITIES,
        out_path=out_disrupt,
        cell_figsize=GRID_2D_CELL_FIGSIZE_DISRUPT,
    )
    produced_figs.extend([out_disrupt.with_suffix(f".{ext}") for ext in PLOT_FORMATS])

    # --- PLOT: optimizer duration (dot grid)
    out_dot_solver = paths.figures_dir / "optimizer_duration"
    plot_grid(
        df_util_agg,
        metric_col="solver_duration_ms_mean",
        y_label=r"optimizer duration (s)",
        ppns=PLOT_PPNS,
        priorities=PLOT_PRIORITIES,
        out_path=out_dot_solver,
        y_scale=1.0 / 1000.0,
        y_lim=DOT_YLIM_SOLVER_DUR,
        y_nticks=DOT_NTICKS_SOLVER_DUR,
        y_label_x=DOT_YLABEL_X_DIFF,
        row_label_pad=DOT_YLABEL_PAD_DIFF,
        show_zero_line=False,
    )
    produced_figs.extend([out_dot_solver.with_suffix(f".{ext}") for ext in PLOT_FORMATS])

    # --- PLOT: per-(ppn,prio,timeout) 3D plots
    for ppn in PLOT_PPNS:
        for prio in PLOT_PRIORITIES:
            for t in PLOT_TIMEOUTS:
                sub = df_per_combo[
                    (df_per_combo["pods_per_node"] == ppn)
                    & (df_per_combo["priorities"] == prio)
                    & (df_per_combo["timeout_s"] == t)
                ]
                if sub.empty:
                    print(f"[skip] no per-combo rows for ppn={ppn}, prio={prio}, t={t}")
                    continue

                title = rf"{PODS_PER_NODE_PLOT_LABEL}={ppn}, {PRIORITIES_PLOT_LABEL}={prio}, {SOLVER_TIMEOUT_LABEL}={t}s"
                out_file = paths.figures_dir / f"3d_ppn={ppn}_priorities={prio}_timeout={t:02d}"
                plot_3d_ppn_prio_timeout(sub, title, out_file)
                produced_figs.extend([out_file.with_suffix(f".{ext}") for ext in PLOT_FORMATS])

    for label_name, paths_list in [("Tables", produced_tables), ("Figures", produced_figs)]:
        print(f"{label_name}:")
        for p in paths_list:
            print(f"  - {p}")


if __name__ == "__main__":
    main()