#!/usr/bin/env python3
"""scripts/kwok_trace_replayer/plot_sealed_results.py

Plot + table generation from *sealed* outputs produced by seal_results.py.

Inputs (under --in-dir):
  - results_long.csv     (optional here; not needed for plots/tables below)
  - results_agg.csv      (optional here; not needed for plots/tables below)
  - results_paired.csv   (USED for LaTeX tables; aggregated across seeds)
  - series/*.csv         (USED for plots; mean time-series across seeds)

Series files naming:
  series/<scheduler>__<job_name>.csv

Each series CSV contains:
  scheduler, job_name, n_seed, time_s, <metric>_mean columns...

Outputs (under --out-dir):
  - plots/<job_name>/<metric>.png
  - plots/<job_name>/<metric>.pdf
  - tables/<metric>.txt     (LaTeX table for each metric)

Default behavior:
  - Plot ALL schedulers found for each job (including "default").
  - Write LaTeX tables for paired deltas (plugin - default) using mean only.

Filtering:
  - --only-schedulers "default,mode=periodic8s_blocking=0_defpreempt=0"
  - --include-scheduler-regex / --exclude-scheduler-regex

Metrics:
  - By default, plots ALL numeric series columns (except metadata columns).
  - Use --only-metrics or --include-metric-regex / --exclude-metric-regex
    if you want to reduce output volume.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# -----------------------------
# Constants / formatting
# -----------------------------
META_COLS = {"scheduler", "job_name", "n_seed", "time_s"}
DEFAULT_TABLE_DECIMALS = 1

_JOB_RE = re.compile(r"nodes=(\d+)_prio=(\d+)_arrival=([0-9.]+)s$")

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape_text(s: str) -> str:
    """Escape LaTeX special chars for TEXT context (caption, plain text)."""
    out = []
    for ch in str(s):
        out.append(_LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot + LaTeX tables from sealed_out results.")
    p.add_argument(
        "--in-dir",
        required=True,
        help="Directory produced by seal_results.py (contains results_paired.csv and series/).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <in-dir>/report_out).",
    )

    # Scheduler filtering (plots only)
    p.add_argument(
        "--only-schedulers",
        default="",
        help="Comma-separated scheduler keys to include in plots. Empty = all.",
    )
    p.add_argument(
        "--include-scheduler-regex",
        default="",
        help="Regex: keep schedulers matching this (applied after --only-schedulers).",
    )
    p.add_argument(
        "--exclude-scheduler-regex",
        default="",
        help="Regex: drop schedulers matching this (applied last).",
    )

    # Metric filtering (plots only)
    p.add_argument(
        "--only-metrics",
        default="",
        help="Comma-separated metric columns to plot from series files (e.g., cpu_run_util_mean). Empty = all numeric.",
    )
    p.add_argument(
        "--include-metric-regex",
        default="",
        help="Regex: keep metrics matching this (applied after --only-metrics).",
    )
    p.add_argument(
        "--exclude-metric-regex",
        default="",
        help="Regex: drop metrics matching this (applied last).",
    )

    # Plot style
    p.add_argument("--fig-w", type=float, default=8.0, help="Figure width in inches (default: 8).")
    p.add_argument("--fig-h", type=float, default=4.5, help="Figure height in inches (default: 4.5).")
    p.add_argument("--dpi", type=int, default=200, help="PNG DPI (default: 200).")

    # Tables
    p.add_argument(
        "--table-decimals",
        type=int,
        default=DEFAULT_TABLE_DECIMALS,
        help="Decimals for LaTeX table cell formatting (default: 1).",
    )
    p.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip LaTeX table generation.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return p.parse_args()


# -----------------------------
# Helpers: parsing / formatting
# -----------------------------
@dataclass(frozen=True)
class JobKey:
    job_name: str
    n_nodes: int
    k_max: int
    arrival_s: float


def parse_job_name(job_name: str) -> JobKey:
    m = _JOB_RE.match(job_name.strip())
    if not m:
        raise ValueError(f"Unrecognized job_name format: {job_name}")
    n = int(m.group(1))
    k = int(m.group(2))
    a = float(m.group(3))
    return JobKey(job_name=job_name, n_nodes=n, k_max=k, arrival_s=a)


def _fmt_arrival(a: float) -> str:
    if abs(a - round(a)) < 1e-9:
        return str(int(round(a)))
    return f"{a:g}"


def _fmt_signed(x: object, *, decimals: int) -> str:
    """Signed numeric for LaTeX cells. NaN -> \\text{--}."""
    try:
        v = float(x)
    except Exception:
        return r"\text{--}"
    if not math.isfinite(v):
        return r"\text{--}"
    fmt = f"{{:+.{int(decimals)}f}}"
    s = fmt.format(v)
    # avoid "-0.0"
    if s.startswith("-0") and abs(v) < 0.5 * (10 ** (-decimals)):
        s = s.replace("-", "+", 1)
    return s


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)


def _scheduler_row_label(scheduler_key: str) -> str:
    """Make a compact row label from scheduler key (plugin_config)."""
    if scheduler_key == "default":
        return "Default"

    # Expected: mode=<mode>_blocking=<0/1>_defpreempt=<0/1>
    kv = {}
    for part in scheduler_key.split("_"):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip().lower()] = v.strip()

    mode_raw = kv.get("mode", scheduler_key)
    blocking = kv.get("blocking", "0") in {"1", "true", "yes"}
    defp = kv.get("defpreempt", "0") in {"1", "true", "yes"}

    # Mode name + optional param
    mode_title = "Unknown"
    mode_param = ""
    mr = mode_raw.lower()

    if mr.startswith("periodic"):
        mode_title = "Periodic"
        tail = mr[len("periodic") :]
        if tail:
            mode_param = tail
    elif mr.startswith("stable-queue") or mr.startswith("stablequeue"):
        mode_title = "Stable-queue"
        tail = mr.replace("stable-queue", "").replace("stablequeue", "")
        if tail:
            mode_param = tail
    elif mr.startswith("scheduling-failure") or mr.startswith("schedulingfailure"):
        mode_title = "Scheduling-failure"
    else:
        mode_title = mode_raw

    enf = "blk" if blocking else "non-blk"

    if mode_param:
        base = f"{mode_title}-{mode_param} ({enf})"
    else:
        base = f"{mode_title} ({enf})"

    if defp:
        return base.replace(")", ", defpreempt)")
    return base


# -----------------------------
# Series scanning + filtering
# -----------------------------
def iter_series_files(series_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (scheduler, job_name, path) from series/<scheduler>__<job_name>.csv"""
    if not series_dir.exists():
        return
    for p in sorted(series_dir.glob("*.csv")):
        stem = p.stem  # "<scheduler>__<job_name>"
        if "__" not in stem:
            continue
        scheduler, job_name = stem.split("__", 1)
        yield scheduler, job_name, p


def filter_values(
    values: Sequence[str],
    *,
    only_csv: str,
    include_regex: str,
    exclude_regex: str,
) -> List[str]:
    out = list(values)

    if only_csv.strip():
        allowed = {x.strip() for x in only_csv.split(",") if x.strip()}
        out = [x for x in out if x in allowed]

    if include_regex.strip():
        rx = re.compile(include_regex)
        out = [x for x in out if rx.search(x)]

    if exclude_regex.strip():
        rx = re.compile(exclude_regex)
        out = [x for x in out if not rx.search(x)]

    return out


def discover_jobs_and_schedulers(series_dir: Path) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Path]]:
    """Return (jobs, schedulers, (scheduler,job)->path)."""
    paths: Dict[Tuple[str, str], Path] = {}
    jobs = set()
    scheds = set()
    for scheduler, job_name, p in iter_series_files(series_dir):
        paths[(scheduler, job_name)] = p
        jobs.add(job_name)
        scheds.add(scheduler)
    return sorted(jobs), sorted(scheds), paths


# -----------------------------
# Plotting
# -----------------------------
def load_series_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time_s" not in df.columns:
        raise ValueError(f"{path} missing time_s column")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df = df.sort_values("time_s")
    return df


def plot_job(
    *,
    job_name: str,
    scheduler_paths: Dict[str, Path],
    out_dir: Path,
    metric_cols: List[str],
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    job_dir = out_dir / "plots" / _sanitize_filename(job_name)
    job_dir.mkdir(parents=True, exist_ok=True)

    series: Dict[str, pd.DataFrame] = {}
    for sched, p in scheduler_paths.items():
        try:
            series[sched] = load_series_csv(p)
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")

    if not series:
        return

    if not metric_cols:
        cols = set()
        for df in series.values():
            for c in df.columns:
                if c in META_COLS:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]) or c.endswith("_mean"):
                    cols.add(c)
        metric_cols = sorted(cols)

    for metric in metric_cols:
        plt.figure(figsize=(fig_w, fig_h))

        for sched, df in series.items():
            if metric not in df.columns:
                continue
            x = df["time_s"].to_numpy(dtype=float)
            y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
            plt.plot(x, y, label=_scheduler_row_label(sched))

        plt.xlabel("time (s)")
        plt.ylabel(metric)
        plt.title(f"{job_name}: {metric}")
        plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        plt.legend(loc="best", fontsize=8)

        png_path = job_dir / f"{_sanitize_filename(metric)}.png"
        pdf_path = job_dir / f"{_sanitize_filename(metric)}.pdf"
        plt.tight_layout()
        plt.savefig(png_path, dpi=dpi)
        plt.savefig(pdf_path)
        plt.close()


# -----------------------------
# LaTeX tables from results_paired.csv
# -----------------------------
def prio_cell_from_row(
    row: pd.Series,
    *,
    k_max: int,
    prefix: str,
    decimals: int,
) -> str:
    """
    k_max=1 -> "+0.9"
    k_max=4 -> "{\\scriptsize$\\langle+0.9,-0.2,+0.0,-1.1\\rangle$}"
    """
    if k_max <= 0:
        return r"\text{--}"

    if k_max == 1:
        return _fmt_signed(row.get(f"{prefix}1_mean", np.nan), decimals=decimals)

    vals = [
        _fmt_signed(row.get(f"{prefix}{p}_mean", np.nan), decimals=decimals)
        for p in range(1, k_max + 1)
    ]

    if all(v == r"\text{--}" for v in vals):
        return r"\text{--}"

    inner = ",".join(vals)
    return r"{\scriptsize$\langle" + inner + r"\rangle$}"


def angle_pair_cell_from_row(
    row: pd.Series,
    *,
    col_a: str,
    col_b: str,
    decimals: int,
) -> str:
    r"""
    Emit a 2-tuple in angle brackets, e.g.:
      {\scriptsize$\langle+0.02,-0.01\rangle$}
    """
    a = _fmt_signed(row.get(col_a, np.nan), decimals=decimals)
    b = _fmt_signed(row.get(col_b, np.nan), decimals=decimals)
    if a == r"\text{--}" and b == r"\text{--}":
        return r"\text{--}"
    return r"{\scriptsize$\langle" + f"{a},{b}" + r"\rangle$}"


def latex_full_table_for_metric(
    *,
    df_metric_allk: pd.DataFrame,
    title_line: str,
    metric_name: str,
    decimals: int,
    caption_tex: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    Emit ONE full LaTeX table (table+adjustbox+tabular) for this metric,
    stacking k_max sections vertically inside the same tabular.
    """
    nodes = sorted({int(x) for x in df_metric_allk["n_nodes"].unique()})
    arrivals = sorted({float(x) for x in df_metric_allk["mean_arrival_s"].unique()})
    k_values = sorted({int(x) for x in pd.to_numeric(df_metric_allk["k_max"], errors="coerce").dropna().unique()})

    ncols = 1 + len(nodes) * len(arrivals)

    def section_header(k: int) -> str:
        if k == 1:
            return rf"$k_{{\max}}={k}$ (no priorities)"
        return rf"$k_{{\max}}={k}$ (priorities enabled)"

    # Build cell maps per k
    cell_maps: Dict[int, Dict[Tuple[str, int, float], str]] = {}
    row_keys_by_k: Dict[int, List[str]] = {}

    for k in k_values:
        dfk = df_metric_allk[df_metric_allk["k_max"].astype(int) == int(k)].copy()
        if dfk.empty:
            continue

        row_keys = sorted(
            dfk["plugin_config"].astype(str).unique(),
            key=lambda s: _scheduler_row_label(str(s)),
        )
        row_keys_by_k[int(k)] = row_keys

        cm: Dict[Tuple[str, int, float], str] = {}
        for _, r in dfk.iterrows():
            rk = str(r["plugin_config"])
            n = int(r["n_nodes"])
            a = float(r["mean_arrival_s"])
            cm[(rk, n, a)] = dfk.attrs["value_fn"](r)
        cell_maps[int(k)] = cm

    # ---- LaTeX assembly ----
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append("")
    lines.append(r"\begin{adjustbox}{max width=\linewidth}")
    lines.append("")
    lines.append(r"\begin{tabular}{" + "l " + " ".join(["c"] * (ncols - 1)) + "}")
    lines.append(r"\toprule")

    lines.append(r"\multicolumn{" + str(ncols) + r"}{l}{" + title_line + r"} \\")
    lines.append(r"\addlinespace[0.2em]")

    # N header row
    if len(nodes) > 1:
        parts = ["& "]
        for n in nodes:
            parts.append(r"\multicolumn{" + str(len(arrivals)) + r"}{c}{$N=" + str(n) + r"$}")
            parts.append(" & ")
        lines.append("".join(parts).rstrip(" & ") + r" \\")

        start = 2
        cmr = []
        for _i in range(len(nodes)):
            end = start + len(arrivals) - 1
            cmr.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")
            start = end + 1
        lines.append("".join(cmr))
    else:
        lines.append(r"& \multicolumn{" + str(len(arrivals)) + r"}{c}{$N=" + str(nodes[0]) + r"$} \\")
        lines.append(r"\cmidrule(lr){2-" + str(1 + len(arrivals)) + "}")

    # Arrival header row
    arr_hdr = ["& "]
    for _n in nodes:
        for a in arrivals:
            arr_hdr.append(r"$\mu_A{=}" + _fmt_arrival(a) + r"$s")
            arr_hdr.append(" & ")
    lines.append("".join(arr_hdr).rstrip(" & ") + r" \\")
    lines.append(r"\midrule")

    first_section = True
    for k in k_values:
        if k not in cell_maps:
            continue

        if not first_section:
            lines.append(r"\midrule")
        first_section = False

        lines.append(r"\multicolumn{" + str(ncols) + r"}{l}{" + section_header(int(k)) + r"} \\")
        lines.append(r"\midrule")

        cm = cell_maps[int(k)]
        row_keys = row_keys_by_k.get(int(k), [])

        for rk in row_keys:
            label_txt = _scheduler_row_label(rk)
            row = [label_txt]
            for n in nodes:
                for a in arrivals:
                    row.append(cm.get((rk, n, a), r"\text{--}"))
            lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\end{adjustbox}")

    if caption_tex is not None and str(caption_tex).strip():
        # caption_tex is assumed to be valid LaTeX (may include math)
        lines.append(r"\caption{" + str(caption_tex) + r"}")
    else:
        caption_txt = latex_escape_text(f"{metric_name} (mean paired difference vs.\\ baseline).")
        lines.append(r"\caption{" + caption_txt + r"}")

    if label is not None and str(label).strip():
        lines.append(r"\label{" + str(label) + r"}")
    else:
        safe_label = _sanitize_filename(metric_name).replace("_", "-")
        lines.append(r"\label{tab:" + safe_label + r"}")

    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_latex_tables(
    *,
    in_dir: Path,
    out_dir: Path,
    decimals: int,
) -> None:
    paired_path = in_dir / "results_paired.csv"
    if not paired_path.exists():
        print(f"[warn] not found: {paired_path} (skipping tables)")
        return

    df = pd.read_csv(paired_path)

    if not {"n_nodes", "k_max", "mean_arrival_s"}.issubset(df.columns):
        rows = []
        for jn in df["job_name"].astype(str).tolist():
            jk = parse_job_name(jn)
            rows.append((jk.n_nodes, jk.k_max, jk.arrival_s))
        df[["n_nodes", "k_max", "mean_arrival_s"]] = pd.DataFrame(rows, index=df.index)

    if "plugin_config" not in df.columns:
        raise SystemExit("results_paired.csv missing 'plugin_config' column")

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ---- table specs (mean only) ----
    def scalar_value_fn(col: str):
        def _fn(r: pd.Series) -> str:
            return _fmt_signed(r.get(col, np.nan), decimals=decimals)
        return _fn

    metric_defs = [
        {
            "name": "delta_R_p",
            "title": r"$\mathbf{\Delta \textit{R}_{p}\textit{(T)}}$ (pod-seconds), mean paired difference vs.\ baseline",
            "kind": "prio_vector",
            "prefix": "delta_R_p",
            "caption_tex": (
                r"Mean paired difference in cumulative running pod-seconds "
                r"$\Delta R_p(T)$ between plugin and baseline. For $k_{\max}>1$, cells report "
                r"${\scriptsize$\langle\Delta R_1,\ldots,\Delta R_{k_{\max}}\rangle$}$."
            ),
            "label": "tab:delta-running-pod-seconds",
        },
        {
            "name": "delta_D_p",
            "title": r"$\mathbf{\Delta \textit{D}_{p}\textit{(T)}}$ (deletions), mean paired difference vs.\ baseline",
            "kind": "prio_vector",
            "prefix": "delta_D_p",
            "caption_tex": (
                r"Mean paired difference in cumulative re-queue events (deletions) "
                r"$\Delta D_p(T)$ between plugin and baseline. For $k_{\max}>1$, cells report "
                r"${\scriptsize$\langle\Delta D_1,\ldots,\Delta D_{k_{\max}}\rangle$}$."
            ),
            "label": "tab:delta-deletions",
        },
        # ---- NEW: CPU+MEM running util as ONE table with angle brackets ----
        {
            "name": "delta_run_util_mean",
            "title": r"$\mathbf{\Delta u_{\mathrm{run}}}$ (running utilisation), mean paired difference vs.\ baseline",
            "kind": "angle_pair",
            "col_a": "delta_cpu_run_util_mean_mean",
            "col_b": "delta_mem_run_util_mean_mean",
            "caption_tex": (
                r"Mean paired difference in running utilisation between plugin and baseline. "
                r"Each cell reports ${\scriptsize$\langle\Delta u_{\mathrm{cpu}},\Delta u_{\mathrm{mem}}\rangle$}$."
            ),
            "label": "tab:delta-run-util",
        },
        {
            "name": "delta_latency_mean_s",
            "title": r"$\mathbf{\Delta \mathrm{latency}}$ (s), mean paired difference vs.\ baseline",
            "kind": "scalar",
            "col": "delta_latency_mean_s_mean",
            "caption_tex": (
                r"Mean paired difference in time-to-first-admit (seconds) between plugin and baseline."
            ),
            "label": "tab:delta-latency",
        },
        {
            "name": "delta_R_total",
            "title": r"$\mathbf{\Delta \textit{R}_{\mathrm{total}}\textit{(T)}}$ (pod-seconds), mean paired difference vs.\ baseline",
            "kind": "scalar",
            "col": "delta_R_total_mean",
            "caption_tex": (
                r"Mean paired difference in total cumulative running pod-seconds "
                r"$\Delta R_{\mathrm{total}}(T)$ between plugin and baseline."
            ),
            "label": "tab:delta-R-total",
        },
        {
            "name": "delta_D_total",
            "title": r"$\mathbf{\Delta \textit{D}_{\mathrm{total}}\textit{(T)}}$ (deletions), mean paired difference vs.\ baseline",
            "kind": "scalar",
            "col": "delta_D_total_mean",
            "caption_tex": (
                r"Mean paired difference in total cumulative re-queue events "
                r"$\Delta D_{\mathrm{total}}(T)$ between plugin and baseline."
            ),
            "label": "tab:delta-D-total",
        },
    ]

    for md in metric_defs:
        out_txt = tables_dir / f"{md['name']}.txt"
        df_metric = df.copy()

        if md["kind"] == "prio_vector":
            prefix = str(md["prefix"])

            def value_fn_factory(k: int):
                def _fn(r: pd.Series) -> str:
                    return prio_cell_from_row(r, k_max=int(k), prefix=prefix, decimals=decimals)
                return _fn

            def _dispatch(r: pd.Series) -> str:
                k = int(r["k_max"])
                return value_fn_factory(k)(r)

            df_metric.attrs["value_fn"] = _dispatch

        elif md["kind"] == "angle_pair":
            col_a = str(md["col_a"])
            col_b = str(md["col_b"])
            if col_a not in df_metric.columns or col_b not in df_metric.columns:
                continue

            def _pair(r: pd.Series) -> str:
                return angle_pair_cell_from_row(r, col_a=col_a, col_b=col_b, decimals=decimals)

            df_metric.attrs["value_fn"] = _pair

        else:
            col = str(md["col"])
            if col not in df_metric.columns:
                continue
            df_metric.attrs["value_fn"] = scalar_value_fn(col)

        table_tex = latex_full_table_for_metric(
            df_metric_allk=df_metric,
            title_line=md["title"],
            metric_name=md["name"],
            decimals=decimals,
            caption_tex=md.get("caption_tex"),
            label=md.get("label"),
        )
        out_txt.write_text(table_tex + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    series_dir = in_dir / "series"
    if not series_dir.exists():
        raise SystemExit(f"Missing series dir: {series_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "report_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, schedulers_all, paths = discover_jobs_and_schedulers(series_dir)

    schedulers_plot = filter_values(
        schedulers_all,
        only_csv=args.only_schedulers,
        include_regex=args.include_scheduler_regex,
        exclude_regex=args.exclude_scheduler_regex,
    )

    only_metrics = [x.strip() for x in args.only_metrics.split(",") if x.strip()] if args.only_metrics.strip() else []

    if not args.no_plots:
        for job_name in jobs:
            sched_paths: Dict[str, Path] = {}
            for sched in schedulers_plot:
                p = paths.get((sched, job_name))
                if p is not None:
                    sched_paths[sched] = p
            if not sched_paths:
                continue

            metric_cols = only_metrics.copy()
            if metric_cols:
                metric_cols = filter_values(
                    metric_cols,
                    only_csv=",".join(metric_cols),
                    include_regex=args.include_metric_regex,
                    exclude_regex=args.exclude_metric_regex,
                )

            plot_job(
                job_name=job_name,
                scheduler_paths=sched_paths,
                out_dir=out_dir,
                metric_cols=metric_cols,
                fig_w=float(args.fig_w),
                fig_h=float(args.fig_h),
                dpi=int(args.dpi),
            )

    if not args.no_tables:
        write_latex_tables(in_dir=in_dir, out_dir=out_dir, decimals=int(args.table_decimals))

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
