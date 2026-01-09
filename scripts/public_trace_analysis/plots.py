#!/usr/bin/env python3
# plots.py

# ----------------------------------------------------------------------
# Examples:
#   python -m scripts.public_trace_analysis.plots --layout datasets-cols
#   python -m scripts.public_trace_analysis.plots --layout datasets-rows
# ----------------------------------------------------------------------

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from scripts.kwok_trace_replayer.plot_helpers import (
    plot_histogram_with_pareto,
    plot_bar_with_geometric,
)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DATASETS: List[Dict[str, Any]] = [
    {
        "name": "Google",
        "csv_path": "data/public_trace_data/google-cluster-data/ClusterData2019/data.csv",
        "plots": {
            "inter_arrival_us": {
                "x_label": "Δt (milliseconds)",
                "scale": 1e-3,  # µs → ms
                "fit_pareto": True,
                "pareto_alpha": 0.21,
                "pareto_xmin": 0.001,
            },
            "life_time_us": {
                "fit_pareto": True,
                "pareto_alpha": 0.72,
                "pareto_xmin": 1.0,
            },
            "cpu_request": {
                "x_label": "Requested CPU (fraction of node capacity)",
                "x_max": 0.2,
                "fit_pareto": True,
                "pareto_alpha": 2.5,
                "pareto_xmin": 0.005,
            },
            "mem_request": {
                "x_label": "Requested memory (fraction of node capacity)",
                "x_max": 0.2,
                "fit_pareto": True,
                "pareto_alpha": 2.5,
                "pareto_xmin": 0.005,
            },
            "priority": {
                "log_y": True,
                "fit_geometric": True,
                "geom_ratio": 0.98,
            },
        },
    },
    {
        "name": "Alibaba",
        "csv_path": "data/public_trace_data/alibaba-clusterdata/cluster-trace-gpu-v2025/data.csv",
        "plots": {
            "inter_arrival_us": {
                "fit_pareto": True,
                "pareto_alpha": 0.46,
                "pareto_xmin": 10.0,
            },
            "life_time_us": {
                "fit_pareto": True,
                "pareto_alpha": 0.43,
                "pareto_xmin": 1.0,
            },
            "cpu_request": {
                "x_label": "Requested CPU (cores)",
            },
            "mem_request": {
                "x_label": "Requested memory (GB)",
            },
        },
    },
]

# ----------------------------------------------------------------------
# BASE PLOT CONFIG
# ----------------------------------------------------------------------

BASE_PLOT_SPECS = [
    {
        "column": "inter_arrival_us",
        "title": "Inter-arrival time",
        "x_label": "Δt (seconds)",
        "y_label": "Probability density",
        "bins": 100,
        "log_y": True,
        "y_min": 1e-6,
        "y_max": 1e-1,
        "x_max": 4_000.0,       # after scaling
        "scale": 1e-6,          # µs → seconds
        "fit_pareto": False,
    },
    {
        "column": "life_time_us",
        "title": "Instance lifetime",
        "x_label": "Lifetime (hours)",
        "y_label": "Probability density",
        "bins": 100,
        "log_y": True,
        "y_min": 1e-5,
        "y_max": 1e-1,
        "x_max": 550.0,
        "scale": 1.0 / (1_000_000.0 * 3600.0),  # µs → hours
        "fit_pareto": False,
    },
    {
        "column": "cpu_request",
        "title": "CPU request",
        "x_label": "Requested CPU",
        "y_label": "Probability density",
        "bins": 100,
        "log_y": True,
        "y_min": 1e-4,
        "y_max": 1e3,
        "x_max": None,
        "scale": 1.0,
        "fit_pareto": False,
    },
    {
        "column": "mem_request",
        "title": "Memory request",
        "x_label": "Requested memory",
        "y_label": "Probability density",
        "bins": 100,
        "log_y": True,
        "y_min": 1e-4,
        "y_max": 1e3,
        "x_max": None,
        "scale": 1.0,
        "fit_pareto": False,
    },
    {
        "column": "priority",
        "title": "Priority",
        "x_label": "Priority",
        "y_label": "Probability mass",
        "bins": 10,
        "log_y": False,
        "y_min": None,
        "y_max": None,
        "x_max": None,
        "scale": 1.0,
        "fit_geometric": False,
        "geom_ratio": 1.0,
    },
]


def build_plot_specs_for_dataset(per_column_cfg: Dict[str, Dict[str, Any]] | None) -> Dict[str, Dict[str, Any]]:
    """
    Take BASE_PLOT_SPECS and apply per-column configs.
    """
    per_column_cfg = per_column_cfg or {}
    specs_by_col: Dict[str, Dict[str, Any]] = {}
    for base in BASE_PLOT_SPECS:
        col = base["column"]
        spec = dict(base) # copy
        if col in per_column_cfg:
            spec.update({k: v for k, v in per_column_cfg[col].items()})
        specs_by_col[col] = spec
    return specs_by_col


def _draw_cell(ax, col_name: str, data, spec: Dict[str, Any], y_label: str) -> None:
    """
    Draw one subplot cell (either histogram+pareto or bar+geometric).
    """
    if col_name == "priority":
        plot_bar_with_geometric(
            ax,
            data,
            title="",
            x_label=spec["x_label"],
            y_label=y_label,
            geom_fit=spec.get("fit_geometric", False),
            geom_ratio=spec.get("geom_ratio"),
            x_min=None,
            x_max=spec.get("x_max"),
            y_min=spec.get("y_min"),
            y_max=spec.get("y_max"),
            log_y=spec.get("log_y", False),
        )
    else:
        plot_histogram_with_pareto(
            ax,
            data,
            title="",
            x_label=spec["x_label"],
            y_label=y_label,
            bins=spec["bins"],
            x_max=spec.get("x_max"),
            y_min=spec.get("y_min"),
            y_max=spec.get("y_max"),
            log_y=spec["log_y"],
            scale=spec.get("scale", 1.0),
            pareto_fit=spec.get("fit_pareto", False),
            pareto_alpha=spec.get("pareto_alpha"),
            pareto_xmin=spec.get("pareto_xmin"),
        )


def make_combined_grid(layout: str, out_dir: str) -> None:
    """
    layout:
      - "datasets-cols": columns=datasets (Google|Alibaba), rows=metrics
      - "datasets-rows": rows=datasets (Google over Alibaba), cols=metrics
    """
    loaded: List[Dict[str, Any]] = []
    for ds in DATASETS:
        name = ds["name"]
        csv_path = ds["csv_path"]
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(df)} rows from {csv_path} for dataset {name}")
        specs_by_col = build_plot_specs_for_dataset(ds.get("plots"))
        loaded.append({"name": name, "df": df, "specs_by_col": specs_by_col})

    metric_columns = [spec["column"] for spec in BASE_PLOT_SPECS]
    base_by_col = {s["column"]: s for s in BASE_PLOT_SPECS}

    if layout == "datasets-cols":
        # Rows=metrics, Cols=datasets
        n_rows = len(metric_columns)
        n_cols = len(loaded)
        fig_w = 3.2 * n_cols
        fig_h = 1.9 * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

        for row_idx, col_name in enumerate(metric_columns):
            base_spec = base_by_col[col_name]
            for col_idx, ds_info in enumerate(loaded):
                ax = axes[row_idx, col_idx]
                df = ds_info["df"]
                spec = ds_info["specs_by_col"][col_name]

                if col_name not in df.columns:
                    ax.axis("off")
                    continue

                data = df[col_name].to_numpy()

                # Left-side label per metric row, only on first dataset column.
                y_label = spec["y_label"] if col_idx == 0 else ""

                _draw_cell(ax, col_name, data, spec, y_label)

                # Bold metric label (once per row, left side)
                if col_idx == 0:
                    ax.text(
                        -0.20, 0.5, base_spec["title"],
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=7,
                        fontweight="bold",
                    )

        # ONE dataset label per dataset
        for col_idx, ds_info in enumerate(loaded):
            axes[0, col_idx].set_title(ds_info["name"], fontsize=11, pad=8, fontweight="bold")

    else:
        # Rows=datasets, Cols=metrics
        n_rows = len(loaded)
        n_cols = len(metric_columns)
        fig_w = 2.1 * n_cols
        fig_h = 1.6 * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

        for row_idx, ds_info in enumerate(loaded):
            ds_name = ds_info["name"]
            df = ds_info["df"]
            specs_by_col = ds_info["specs_by_col"]

            for col_idx, col_name in enumerate(metric_columns):
                ax = axes[row_idx, col_idx]

                if col_name not in df.columns:
                    ax.axis("off")
                    continue

                spec = specs_by_col[col_name]
                data = df[col_name].to_numpy()

                # y-label only once per dataset row
                y_label = spec["y_label"] if col_idx == 0 else ""

                _draw_cell(ax, col_name, data, spec, y_label)

                # Dataset label once per row, on the left side (first column only)
                if col_idx == 0:
                    ax.text(
                        -0.32, 0.5, ds_name,
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontweight="bold",
                    )

        # Metric titles (top row)
        for col_idx, col_name in enumerate(metric_columns):
            axes[0, col_idx].set_title(base_by_col[col_name]["title"], fontsize=7, pad=8, fontweight="bold")

    fig.tight_layout(
        rect=(0.01, 0.01, 0.99, 0.99),
        pad=0.3,
        w_pad=0.4,
        h_pad=0.4,
    )

    out_dir = out_dir.rstrip("/") + "/"
    fig.savefig(out_dir + "public_trace_histograms.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir + "public_trace_histograms.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved combined grid to {out_dir} (layout={layout})")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Public trace histogram grid plotter.")
    p.add_argument(
        "--layout",
        type=str,
        default="datasets-cols",
        choices=["datasets-cols", "datasets-rows"],
        help="Layout mode: datasets as columns or datasets as rows.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="analysis/kwok_trace_replayer/figures/",
        help="Output directory for the combined plots.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    make_combined_grid(layout=args.layout, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
