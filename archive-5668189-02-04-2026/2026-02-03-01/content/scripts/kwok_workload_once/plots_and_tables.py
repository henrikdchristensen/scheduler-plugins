#!/usr/bin/env python3
# scripts/kwok_workload_once/plots_and_tables.py
"""
python -m scripts.kwok_workload_once.plots_and_tables
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

from scripts.helpers.plot_config import (
    PLOT_TITLE_FONTSIZE,
    PLOT_AXIS_LABEL_FONTSIZE,
    PLOT_TICK_FONTSIZE,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_HANDLE_LENGTH,
    PLOT_LEGEND_COLUMN_SPACING,
    PLOT_LEGEND_HANDLE_TEXT_PAD,
    PLOT_FIGURE_DPI,
    PLOT_FORMATS,
)

from scripts.helpers.data_helpers import is_finite, safe_div
from scripts.helpers.table_helpers import fmt_pct

#################################################################
# CONFIG (constants)
#################################################################

DF_PER_COMBO_PATH = Path("analysis/kwok_workload_once/results_per_combo.csv")

OUT_DIR = Path("analysis/kwok_workload_once")
OUT_FIGURES_DIR = OUT_DIR / "figures"
OUT_TABLES_DIR = OUT_DIR / "tables"

# filters
PLOT_PPNS = [4, 8]
PLOT_PRIORITIES = [1, 2, 4]
PLOT_TIMEOUTS = [1, 10, 20, 60]

# precision
EPS = 1e-9

# fonts
ANNOT_FS = 3.5

# labels
TARGET_UTIL_LABEL = "target util (%)"
NODES_LABEL = "# of nodes"
INSTANCES_LABEL = "% of instances"
PODS_PER_NODE_LABEL = "pods/node"

# figure saving


# 2d sizes
GRID_2D_CELL_FIGSIZE = (2.6, 1.6)
GRID_2D_BAR_WIDTH = 0.88
GRID_2D_WSPACE = 0.08
GRID_2D_HSPACE = 0.12
GRID_2D_LEFT = 0.1
GRID_2D_RIGHT = 0.995
GRID_2D_BOTTOM = 0.0
GRID_2D_TOP = 0.88
GRID_2D_YLABEL_XPOS = 0.015

# 3d sizes
FIGSIZE_3D = (8, 4)
BAR_WIDTH_3D = 0.13
ELEV_3D, AZIM_3D = 20.0, -54.0

# colors / series (stack order = bottom -> top)
set2, set3 = plt.get_cmap("Set2").colors, plt.get_cmap("Set3").colors
CATEGORIES = [
    {"key": "other", "label": "Other", "col": "other_rate", "color": set2[3]},
    {"key": "solver_optimal", "label": "Better&Optimal", "col": "solver_optimal_rate", "color": set2[0]},
    {"key": "solver_feasible", "label": "Better", "col": "solver_feasible_rate", "color": set2[1]},
    {"key": "default_optimal", "label": "KWOK Optimal", "col": "default_optimal_rate", "color": set2[2]},
    {"key": "default_all", "label": "No Calls", "col": "default_all_running_rate", "color": set3[11]},
    {"key": "solver_failed", "label": "Failures", "col": "solver_failed_rate", "color": set2[7]},
]

#################################################################
# Aggregation helpers
#################################################################

def _aggregate_counts_to_rates(per_combo_df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """
    Aggregate by `keys` (sum counts/sums) and compute rate columns.

    NOTE:
    - If keys exclude 'util' -> results are aggregated over util (used for plots).
    - If keys include 'util'  -> results preserve util (used for tables with util breaker).
    """
    g = (
        per_combo_df.groupby(keys, as_index=False).agg(
            {
                "n_seeds": "sum",
                "n_seeds_not_all_running": "sum",
                "n_default_all_running": "sum",
                "n_solver_called": "sum",
                "n_solver_failed": "sum",
                "n_default_optimal": "sum",
                "n_solver_optimal": "sum",
                "n_solver_feasible": "sum",
                "n_solver_improve": "sum",
                "n_other": "sum",
                "solver_duration_ms_sum": "sum",
                "cpu_delta_sum": "sum",
                "mem_delta_sum": "sum",
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
    return g.copy()

def aggregate_over_util(per_combo_df: pd.DataFrame) -> pd.DataFrame:
    # Used by plots: util is aggregated away (unchanged behavior)
    keys = ["pods_per_node", "priorities", "timeout_s", "nodes"]
    return _aggregate_counts_to_rates(per_combo_df, keys)

def aggregate_keep_util(per_combo_df: pd.DataFrame) -> pd.DataFrame:
    # Used by tables: keep util so breaker can show (timeout, util)
    keys = ["util", "pods_per_node", "priorities", "timeout_s", "nodes"]
    return _aggregate_counts_to_rates(per_combo_df, keys)

#################################################################
# Tables
#################################################################

OUTCOME_ROWS: List[Tuple[str, str]] = [
    ("Failures", "solver_failed_rate"),
    ("No Calls", "default_all_running_rate"),
    ("KWOK Optimal", "default_optimal_rate"),
    ("Better", "solver_feasible_rate"),
    ("Better\\&Optimal", "solver_optimal_rate"),
    ("Other", "other_rate"),
]

def _infer_orders_for_table(
    df_table: pd.DataFrame,
    *,
    priorities: int,
    ppns: List[int],
    timeouts: List[int],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    dff = df_table[df_table["priorities"].astype(int) == int(priorities)].copy()

    # orders
    nodes_order = sorted(int(x) for x in dff["nodes"].dropna().unique().tolist())

    ppn_present = set(int(x) for x in dff["pods_per_node"].dropna().unique().tolist())
    ppn_order = [p for p in ppns if p in ppn_present]

    timeout_present = set(int(x) for x in dff["timeout_s"].dropna().unique().tolist())
    timeout_order = [t for t in timeouts if t in timeout_present]

    # util order (rounded to integer %)
    util_vals = pd.to_numeric(dff["util"], errors="coerce").dropna().round().astype(int).unique().tolist()
    util_order = sorted(int(u) for u in util_vals)

    return nodes_order, ppn_order, timeout_order, util_order

def write_outcome_breakdown_table_tex(
    *,
    df_table: pd.DataFrame,
    out_path: Path,
    priorities: int,
    ppns: List[int],
    timeouts: List[int],
    decimals: int = 1,
) -> None:
    nodes_order, ppn_order, timeout_order, util_order = _infer_orders_for_table(
        df_table,
        priorities=priorities,
        ppns=ppns,
        timeouts=timeouts,
    )

    if not nodes_order or not ppn_order or not timeout_order or not util_order:
        out_path.write_text("% empty: no nodes/ppn/timeout/util after filters\n", encoding="utf-8")
        print(f"[warn] table empty after filters -> {out_path}")
        return

    idx_cols = ["priorities", "timeout_s", "util", "nodes", "pods_per_node"]
    keep_cols = idx_cols + [c for _lbl, c in OUTCOME_ROWS]
    missing = [c for c in keep_cols if c not in df_table.columns]
    if missing:
        raise SystemExit(f"[table] missing columns in aggregated df: {missing}")

    dff = df_table[df_table["priorities"].astype(int) == int(priorities)].copy()
    dff["timeout_s"] = dff["timeout_s"].astype(int)
    dff["nodes"] = dff["nodes"].astype(int)
    dff["pods_per_node"] = dff["pods_per_node"].astype(int)
    dff["priorities"] = dff["priorities"].astype(int)
    dff["util"] = pd.to_numeric(dff["util"], errors="coerce").round().astype(int)

    dff = dff[keep_cols].drop_duplicates(subset=idx_cols, keep="last")
    dff = dff.set_index(idx_cols).sort_index()

    def get_rate(timeout_s: int, util: int, nodes: int, ppn: int, col: str) -> float:
        key = (int(priorities), int(timeout_s), int(util), int(nodes), int(ppn))
        try:
            return float(dff.at[key, col])
        except Exception:
            return float("nan")

    n_nodes = len(nodes_order)
    n_ppn = len(ppn_order)
    data_cols = n_nodes * n_ppn
    total_cols = 1 + data_cols

    colspec = "l" + (" " + "c" * data_cols if data_cols > 0 else "")
    lines: List[str] = []
    lines.append("% Values are percent of instances (%).")
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")

    lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textbf{{\#priorities = {int(priorities)}}}}} \\")
    lines.append(r"\addlinespace[0.2em]")

    header_nodes = " & " + " & ".join([rf"\multicolumn{{{n_ppn}}}{{c}}{{{n}}}" for n in nodes_order]) + r" \\"
    lines.append(rf"\makecell[l]{{Outcome\\(\%)}}{header_nodes}")

    cmid = []
    start = 2
    for _ in nodes_order:
        end = start + n_ppn - 1
        cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cmid))

    ppn_hdr = " & " + " & ".join([str(ppn) for _n in nodes_order for ppn in ppn_order]) + r" \\"
    lines.append(rf"\makecell[l]{{}}{ppn_hdr}")
    lines.append(r"\midrule")

    # Breaker: timeout, util
    first_block = True
    for t in timeout_order:
        for u in util_order:
            # skip blocks that do not exist in data
            try:
                _ = dff.loc[(int(priorities), int(t), int(u))]
            except Exception:
                continue

            if not first_block:
                lines.append(r"\midrule")
            first_block = False

            lines.append(
                rf"\multicolumn{{{total_cols}}}{{l}}{{\textbf{{timeout = {int(t)}\,s, util = {int(u)}\%}}}} \\"
            )
            lines.append(r"\midrule")

            for label, col in OUTCOME_ROWS:
                cells: List[str] = []
                for n in nodes_order:
                    for ppn in ppn_order:
                        r_ = get_rate(timeout_s=t, util=u, nodes=n, ppn=ppn, col=col)
                        pct = 100.0 * r_ if is_finite(r_) else float("nan")
                        cells.append(fmt_pct(pct, decimals=decimals))
                lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

#################################################################
# Plots
#################################################################

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
    formats: List[str] = PLOT_FORMATS,
    dpi: int = PLOT_FIGURE_DPI,
) -> None:
    """
    Save a matplotlib figure to multiple file formats.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fname = out_path.with_suffix(f".{ext}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_2d_grid_ppn_prio_with_aggregated_util(
    df_util_agg: pd.DataFrame,
    ppns: list[int],
    priorities: list[int],
    out_path: Path,
    cell_figsize: tuple[float, float],
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

    fig.supylabel("% of instances", fontsize=PLOT_AXIS_LABEL_FONTSIZE, x=GRID_2D_YLABEL_XPOS)

    seen_keys = set()

    for r, ppn in enumerate(ppns):
        for c, prio in enumerate(priorities):
            ax = axes[r][c]
            panel = df_util_agg[(df_util_agg["pods_per_node"] == ppn) & (df_util_agg["priorities"] == prio)].copy()

            if panel.empty:
                ax.axis("off")
                continue

            panel = panel.groupby(["nodes", "timeout_s"], as_index=False)[
                [s["col"] for s in CATEGORIES if s["col"].endswith("_rate")]
            ].mean()

            nodes_vals = sorted(panel["nodes"].unique().tolist())
            ts = sorted(panel["timeout_s"].unique().tolist())
            bars_per_group = max(1, len(ts))
            width = GRID_2D_BAR_WIDTH / bars_per_group
            x = np.arange(len(nodes_vals))

            for j, timeout in enumerate(ts):
                sub = panel[panel["timeout_s"] == timeout].set_index("nodes")
                xj = x + (j - (bars_per_group - 1) / 2.0) * width
                bar_height = np.zeros(len(nodes_vals), dtype=float)

                for category in CATEGORIES:
                    vals = (
                        sub.get(category["col"], pd.Series(0.0, index=sub.index))
                        .reindex(nodes_vals)
                        .fillna(0.0)
                        .values
                    )
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
                    ax.text(
                        float(xi),
                        float(top) + 1.0,
                        f"{int(timeout)}s",
                        ha="center",
                        va="bottom",
                        fontsize=ANNOT_FS,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in nodes_vals], fontsize=PLOT_TICK_FONTSIZE)
            ax.set_ylim(0, 110)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(100.0))
            ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)

            if r == 0:
                ax.set_title(rf"#priorities={prio}", fontsize=PLOT_TITLE_FONTSIZE)

            if c == 0:
                if nrows % 2 == 1 and r == nrows // 2:
                    axis_text = f"{INSTANCES_LABEL}\n{PODS_PER_NODE_LABEL} = {ppn}"
                else:
                    axis_text = f"\n{PODS_PER_NODE_LABEL} = {ppn}"
                lbl = ax.set_ylabel(axis_text, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=10)
                lbl.set_va("center")
                lbl.set_ha("center")
                lbl.set_linespacing(1.8)

            if r == nrows - 1:
                ax.set_xlabel(NODES_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    legends = [s for s in CATEGORIES if s["key"] in seen_keys][::-1]
    legend_handles = [mpatches.Rectangle((0, 0), 1, 1, fc=s["color"]) for s in legends]
    legend_labels = [s["label"] for s in legends]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.02),
        fontsize=PLOT_LEGEND_FONTSIZE,
        ncol=len(legend_labels),
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
    )
    
    # Make the grid tighter (reduce space between panels)
    fig.subplots_adjust(
        left=GRID_2D_LEFT,
        right=GRID_2D_RIGHT,
        bottom=GRID_2D_BOTTOM,
        top=GRID_2D_TOP,
        wspace=GRID_2D_WSPACE,
        hspace=GRID_2D_HSPACE,
    )

    save_figure(fig, out_path)

def plot_3d_ppn_prio_timeout(df: pd.DataFrame, title: str, out_path: Path) -> None:
    utils = sorted(df["util"].unique().tolist())
    nodes = sorted(int(n) for n in df["nodes"].unique().tolist())

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
    ax.set_ylabel(NODES_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=-5.5)
    ax.set_zlabel(INSTANCES_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=-5.0)

    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE, y=1.01, pad=0)

    dx = max(0.05, min(1.0, BAR_WIDTH_3D))
    dy = max(0.05, min(1.0, BAR_WIDTH_3D))

    seen_keys = set()
    for u in utils:
        for n in nodes:
            x0 = x_index[u] + (1 - dx) / 2
            y0 = y_index[n] + (1 - dy) / 2
            z = 0.0

            rate_solver_opt = get_rate(df, u, n, "solver_optimal_rate")
            rate_solver_feas = get_rate(df, u, n, "solver_feasible_rate")
            rate_solver_fail = get_rate(df, u, n, "solver_failed_rate")
            rate_default_opt = get_rate(df, u, n, "default_optimal_rate")
            rate_default_all = get_rate(df, u, n, "default_all_running_rate")
            rate_other = get_rate(df, u, n, "other_rate")

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
                    ax.bar3d(
                        x0,
                        y0,
                        z,
                        dx,
                        dy,
                        h,
                        color=color,
                        edgecolor="black",
                        linewidth=0.35,
                        shade=False,
                    )
                    z += h
                    seen_keys.add(key)

    legends = [s for s in CATEGORIES if s["key"] in seen_keys][::-1]
    legend_handles = [mpatches.Rectangle((0, 0), 1, 1, fc=s["color"]) for s in legends]
    legend_labels = [s["label"] for s in legends]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.545, 0.92),
        fontsize=PLOT_LEGEND_FONTSIZE,
        ncol=len(legend_labels),
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
    )

    save_figure(fig, out_path)

#################################################################
# main
#################################################################

def main() -> None:
    print("Generating tables and figures...")
    
    configure_matplotlib()

    OUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df_per_combo = pd.read_csv(DF_PER_COMBO_PATH)

    produced_tables: List[Path] = []
    produced_figs: List[Path] = []

    # --- TABLES: keep util (breaker = timeout + util)
    df_table = aggregate_keep_util(df_per_combo)
    present_prios = set(df_table["priorities"].astype(int).unique().tolist())
    for prio in [p for p in PLOT_PRIORITIES if p in present_prios]:
        out_tex = OUT_TABLES_DIR / f"table_outcomes_priorities={int(prio)}.tex"
        write_outcome_breakdown_table_tex(
            df_table=df_table,
            out_path=out_tex,
            priorities=int(prio),
            ppns=PLOT_PPNS,
            timeouts=PLOT_TIMEOUTS,
            decimals=1,
        )
        produced_tables.append(out_tex)

    # --- PLOTS (aggregate away util for 2D, and 3D uses per-combo)
    df_util_agg = aggregate_over_util(df_per_combo)

    out_path_2d = OUT_FIGURES_DIR / "2d_grid_ppn_prio"
    plot_2d_grid_ppn_prio_with_aggregated_util(
        df_util_agg=df_util_agg,
        ppns=PLOT_PPNS,
        priorities=PLOT_PRIORITIES,
        out_path=out_path_2d,
        cell_figsize=GRID_2D_CELL_FIGSIZE,
    )
    produced_figs.extend([out_path_2d.with_suffix(f".{ext}") for ext in PLOT_FORMATS])

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
                title = rf"{PODS_PER_NODE_LABEL}={ppn}, #priorities={prio}, timeout={t}s"
                out_file = OUT_FIGURES_DIR / f"3d_ppn{ppn}_prio{prio}_timeout{t:02d}"
                plot_3d_ppn_prio_timeout(sub, title, out_file)
                produced_figs.extend([out_file.with_suffix(f".{ext}") for ext in PLOT_FORMATS])

    # Summary
    for label, paths in [("Tables", produced_tables), ("Figures", produced_figs)]:
        print(f"{label}:\n" + "\n".join(f"  - {p}" for p in paths))

if __name__ == "__main__":
    main()
