#!/usr/bin/env python3
"""scripts/kwok_trace_replayer/plots_and_tables.py

python -m scripts.kwok_trace_replayer.plots_and_tables --in-dir analysis/kwok_trace_replayer/sealed --out-dir analysis/kwok_trace_replayer/plots_and_tables

Plot + LaTeX table generation from sealed outputs produced by seal_results.py.

Inputs (under --in-dir):
  - results_paired.csv   (USED for LaTeX tables; aggregated across seeds)
  - series/default/<job_name>.csv
  - series/plugin/<plugin_config>__<job_name>.csv

Each series CSV contains:
  scheduler, job_name, n_seed, time_s, <metric>_mean columns...

Outputs (under --out-dir):
  - plots/<job_name>/<metric>.png
  - plots/<job_name>/<metric>.pdf
  - tables/<metric>.txt         (LaTeX table for each metric)
  - tables/big_deltas.txt       (big table: util + latency + running pod-seconds + solver attempts)
  - tables/latency_and_solver.txt (single table: latency(total) + solver attempts)

Behavior changes:
  - Plots are created only for:
      * cpu_run_util_mean, mem_run_util_mean, util_eff_run_mean
      * latency_s_total_mean
      * solver_attempts_total_mean
    (and they are filtered further by CLI if provided).
  - Plots are saved WITHOUT std/uncertainty bands (lines only).
  - Only plugin configs with default preemption enabled (defpreempt=1) are plotted,
    but row/legend labels do NOT mention defpreempt.
  - Plots are NOT saved for:
      * unsched_p1_mean
      * cpu_req_util_mean
      * mem_req_util_mean
    and we do not save per-priority latency plots (we use total only).
"""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Constants / formatting
# -----------------------------
META_COLS = {"scheduler", "job_name", "n_seed", "time_s"}
DEFAULT_TABLE_DECIMALS = 1

JOB_RE = re.compile(r"nodes=(\d+)_prio=(\d+)_arrival=([0-9.]+)s$")

LATEX_SPECIALS = {
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

# Only these series metrics are saved by default.
# (CLI metric filters can still reduce further, but we won't add more.)
DEFAULT_PLOT_METRICS = [
    "cpu_run_util_mean",
    "mem_run_util_mean",
    "util_eff_run_mean",
    "running_total_mean",
    "deletions_cum_total_mean",
    "latency_s_total_mean",
]

# Explicitly never save these, even if present / requested.
NEVER_PLOT_METRICS = {
    "unsched_p1_mean",
    "cpu_req_util_mean",
    "mem_req_util_mean",
}

# Only show configs with default preemption enabled in plots (defpreempt=1).
DEFPREEMPT_ONLY_REGEX = re.compile(r"(?:^|_)defpreempt=1(?:_|$)")

# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot + LaTeX tables from sealed results.")
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

    # Scheduler filtering
    p.add_argument(
        "--only-schedulers",
        default="",
        help="Comma-separated scheduler keys to include in plots. Empty = all discovered.",
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

    # Metric filtering
    p.add_argument(
        "--only-metrics",
        default="",
        help="Comma-separated metric columns to plot from series files. Empty = defaults used by this script.",
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
    # Smoothing
    p.add_argument(
        "--smooth-s",
        type=float,
        default=100.0,
        help="Rolling mean window in seconds (0 = no smoothing). Applied per-series after seed-aggregation.",
    )
    
    p.add_argument(
        "--lat-hist-bin-s",
        type=float,
        default=1.0,
        help="Latency histogram bin width in seconds (default: 1.0).",
    )
    p.add_argument(
        "--lat-hist-max-s",
        type=float,
        default=0.0,
        help="Max latency shown in histogram (0 = auto).",
    )
    p.add_argument(
        "--lat-hist-density",
        action="store_true",
        help="Normalize latency histogram to probability density instead of counts.",
    )

    # Tables
    p.add_argument(
        "--table-decimals",
        type=int,
        default=DEFAULT_TABLE_DECIMALS,
        help="Decimals for LaTeX table cell formatting (default: 1).",
    )
    p.add_argument("--no-tables", action="store_true", help="Skip LaTeX table generation.")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
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


def latex_escape_text(s: str) -> str:
    """Escape LaTeX special chars for TEXT context (caption, plain text)."""
    out = []
    for ch in str(s):
        out.append(LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def parse_job_name(job_name: str) -> JobKey:
    m = JOB_RE.match(job_name.strip())
    if not m:
        raise ValueError(f"Unrecognized job_name format: {job_name}")
    n = int(m.group(1))
    k = int(m.group(2))
    a = float(m.group(3))
    return JobKey(job_name=job_name, n_nodes=n, k_max=k, arrival_s=a)


def fmt_arrival(a: float) -> str:
    if abs(a - round(a)) < 1e-9:
        return str(int(round(a)))
    return f"{a:g}"


def fmt_signed(x: object, *, decimals: int) -> str:
    """Signed numeric for LaTeX cells. NaN -> \\text{--}."""
    try:
        v = float(x)
    except Exception:
        return r"\text{--}"
    if not math.isfinite(v):
        return r"\text{--}"
    fmt = f"{{:+.{int(decimals)}f}}"
    s = fmt.format(v)
    if s.startswith("-0") and abs(v) < 0.5 * (10 ** (-decimals)):
        s = s.replace("-", "+", 1)
    return s


def fmt_signed_scaled(x: object, *, decimals: int, scale: float) -> str:
    """Signed numeric for LaTeX cells, after scaling. NaN -> \\text{--}."""
    try:
        v = float(x) * float(scale)
    except Exception:
        return r"\text{--}"
    if not math.isfinite(v):
        return r"\text{--}"
    fmt = f"{{:+.{int(decimals)}f}}"
    s = fmt.format(v)
    if s.startswith("-0") and abs(v) < 0.5 * (10 ** (-decimals)):
        s = s.replace("-", "+", 1)
    return s


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)


def scheduler_row_label_hide_defpreempt(scheduler_key: str) -> str:
    """Compact legend/row label; NEVER mentions defpreempt."""
    if scheduler_key == "default":
        return "Default"

    kv: Dict[str, str] = {}
    for part in str(scheduler_key).split("_"):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip().lower()] = v.strip()

    mode_raw = kv.get("mode", str(scheduler_key))
    blocking = kv.get("blocking", "0") in {"1", "true", "yes"}

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
        return f"{mode_title}-{mode_param} ({enf})"
    return f"{mode_title} ({enf})"


def parse_scheduler_kv(scheduler_key: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for part in str(scheduler_key).split("_"):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return kv


def is_target_plugin_mode(scheduler_key: str) -> bool:
    if scheduler_key == "default":
        return False

    kv = parse_scheduler_kv(scheduler_key)
    mode = kv.get("mode", "").lower()
    mode_norm = mode.replace("_", "-")

    is_periodic = mode_norm.startswith("periodic")
    is_stableq = ("stable" in mode_norm) and ("queue" in mode_norm)
    if not (is_periodic or is_stableq):
        return False

    # Periodic: only keep 8s
    if is_periodic:
        return "8s" in mode_norm

    # Stable-queue: keep regardless of interval (e.g. stablequeue2s)
    return True


# -----------------------------
# Series scanning + filtering
# -----------------------------
def iter_series_files(series_root: Path) -> Iterable[Tuple[str, str, Path]]:
    """
    Yield (scheduler, job_name, path) from:
      - series/default/<job_name>.csv              (scheduler="default")
      - series/plugin/<scheduler>__<job_name>.csv  (scheduler parsed from filename)
    """
    if not series_root.exists():
        return

    ddir = series_root / "default"
    if ddir.exists():
        for p in sorted(ddir.glob("*.csv")):
            job_name = p.stem
            yield "default", job_name, p

    pdir = series_root / "plugin"
    if pdir.exists():
        for p in sorted(pdir.glob("*.csv")):
            stem = p.stem
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


def discover_jobs_and_schedulers(series_root: Path) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Path]]:
    """Return (jobs, schedulers, (scheduler,job)->path)."""
    paths: Dict[Tuple[str, str], Path] = {}
    jobs = set()
    scheds = set()
    for scheduler, job_name, p in iter_series_files(series_root):
        paths[(scheduler, job_name)] = p
        jobs.add(job_name)
        scheds.add(scheduler)
    return sorted(jobs), sorted(scheds), paths


def _keep_scheduler_for_plots(s: str) -> bool:
    """Plot only: default + plugin configs with defpreempt=1."""
    if s == "default":
        return True
    return bool(DEFPREEMPT_ONLY_REGEX.search(str(s)))


# -----------------------------
# Plotting
# -----------------------------
def load_series_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time_s" not in df.columns:
        raise ValueError(f"{path} missing time_s column")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df = df.dropna(subset=["time_s"])

    # numeric coercion for metric cols
    value_cols = [c for c in df.columns if c not in META_COLS]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # collapse multiple seeds -> mean series per time_s
    if "n_seed" in df.columns:
        df = df.groupby("time_s", as_index=False)[value_cols].mean(numeric_only=True)

    df = df.sort_values("time_s")
    return df

def _is_cumulative_metric(metric: str) -> bool:
    m = metric.lower()
    return ("_cum_" in m) or m.startswith("cum_") or m.endswith("_cum") or m.startswith("deletions_cum")

def smooth_series(df: pd.DataFrame, metric: str, window_s: float) -> np.ndarray:
    """Time-based rolling mean. For cumulative metrics, smooth increments then re-cumsum."""
    if window_s <= 0:
        return pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)

    t = pd.to_timedelta(df["time_s"].to_numpy(dtype=float), unit="s")
    s = pd.Series(pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float), index=t)

    win = f"{float(window_s)}s"
    if _is_cumulative_metric(metric):
        inc = s.diff().fillna(0.0)
        inc_sm = inc.rolling(win, min_periods=1).mean()
        return inc_sm.cumsum().to_numpy(dtype=float)
    else:
        return s.rolling(win, min_periods=1).mean().to_numpy(dtype=float)

def interp_on_grid(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Interpolate y_src(t_src) onto t_grid; outside range -> NaN."""
    m = np.isfinite(t_src) & np.isfinite(y_src)
    t_src = t_src[m]
    y_src = y_src[m]
    if t_src.size < 2:
        return np.full_like(t_grid, np.nan, dtype=float)

    y_i = np.interp(t_grid, t_src, y_src)
    in_range = (t_grid >= t_src.min()) & (t_grid <= t_src.max())
    return np.where(in_range, y_i, np.nan)


def delta_series_vs_default(default_df: pd.DataFrame, plugin_df: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t, plugin(t)-default(t)) on default time grid."""
    t0 = default_df["time_s"].to_numpy(dtype=float)
    y0 = pd.to_numeric(default_df[metric], errors="coerce").to_numpy(dtype=float)

    t1 = plugin_df["time_s"].to_numpy(dtype=float)
    y1 = pd.to_numeric(plugin_df[metric], errors="coerce").to_numpy(dtype=float)

    y1i = interp_on_grid(t1, y1, t0)
    return t0, (y1i - y0)


def cumtrapz_from_delta(t: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Cumulative integral of delta over time using trapezoidal rule."""
    out = np.zeros_like(delta, dtype=float)
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        if not (math.isfinite(dt) and dt >= 0):
            dt = 0.0
        a = delta[i - 1]
        b = delta[i]
        if not (math.isfinite(a) and math.isfinite(b)):
            out[i] = out[i - 1]
        else:
            out[i] = out[i - 1] + 0.5 * (a + b) * dt
    return out


def _select_plot_metrics(
    *,
    series_frames: Dict[str, pd.DataFrame],
    only_metrics_csv: str,
    include_metric_regex: str,
    exclude_metric_regex: str,
) -> List[str]:
    # Base set: defaults
    metrics = list(DEFAULT_PLOT_METRICS)

    # Optional override via CLI
    if only_metrics_csv.strip():
        metrics = [x.strip() for x in only_metrics_csv.split(",") if x.strip()]

    # Apply include/exclude
    metrics = filter_values(metrics, only_csv="", include_regex=include_metric_regex, exclude_regex=exclude_metric_regex)

    # Enforce never-plot list and "total only" rules:
    out = []
    for m in metrics:
        if m in NEVER_PLOT_METRICS:
            continue
        # never plot per-priority latency; enforce total only
        if m.startswith("latency_s_p") and not m.endswith("_total_mean") and m != "latency_s_total_mean":
            continue
        out.append(m)

    # Also only keep metrics that exist in at least one DF
    existing = set()
    for df in series_frames.values():
        existing.update(df.columns.tolist())
    out = [m for m in out if m in existing]

    return out


def plot_job(
    *,
    job_name: str,
    scheduler_paths: Dict[str, Path],
    out_dir: Path,
    fig_w: float,
    fig_h: float,
    dpi: int,
    only_metrics_csv: str,
    include_metric_regex: str,
    exclude_metric_regex: str,
    smooth_s: float,
    lat_hist_bin_s: float,
    lat_hist_max_s: float,
    lat_hist_density: bool,
) -> None:
    job_dir = out_dir / "plots" / sanitize_filename(job_name)
    job_dir.mkdir(parents=True, exist_ok=True)

    series: Dict[str, pd.DataFrame] = {}
    for sched, p in scheduler_paths.items():
        try:
            series[sched] = load_series_csv(p)
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")

    if not series:
        return

    if "default" not in series:
        print(f"[warn] no default series for {job_name}; skipping")
        return

    default_df = series["default"]

    # only your plugin modes (periodic8s + stable-queue*), still already defpreempt-filtered in main()
    plugin_series = {s: df for s, df in series.items() if is_target_plugin_mode(s)}
    if not plugin_series:
        return

    metric_cols = _select_plot_metrics(
        series_frames=series,
        only_metrics_csv=only_metrics_csv,
        include_metric_regex=include_metric_regex,
        exclude_metric_regex=exclude_metric_regex,
    )
    if not metric_cols:
        return

    for metric in metric_cols:
        plt.figure(figsize=(fig_w, fig_h))

        # Reference line at 0 since we plot deltas
        plt.axhline(0.0, linewidth=0.8)

        for sched, df in plugin_series.items():
            if metric not in df.columns or metric not in default_df.columns:
                continue

            # --- util deltas (line) ---
            if metric in {"cpu_run_util_mean", "mem_run_util_mean", "util_eff_run_mean"}:
                x, d = delta_series_vs_default(default_df, df, metric)
                # smooth delta directly
                if smooth_s > 0:
                    t = pd.to_timedelta(x, unit="s")
                    s = pd.Series(d, index=t)
                    d = s.rolling(f"{float(smooth_s)}s", min_periods=1).mean().to_numpy(dtype=float)
                plt.plot(x, d, label=scheduler_row_label_hide_defpreempt(sched), linewidth=0.8)

            # --- deletions: delta of cumulative curves (still cumulative trend) ---
            elif metric == "deletions_cum_total_mean":
                x, d_cum = delta_series_vs_default(default_df, df, metric)
                # (optional) smoothing: smooth increments then re-cumsum (reuse your idea)
                if smooth_s > 0:
                    t = pd.to_timedelta(x, unit="s")
                    s = pd.Series(d_cum, index=t)
                    inc = s.diff().fillna(0.0)
                    inc_sm = inc.rolling(f"{float(smooth_s)}s", min_periods=1).mean()
                    d_cum = inc_sm.cumsum().to_numpy(dtype=float)
                plt.plot(x, d_cum, label=scheduler_row_label_hide_defpreempt(sched), linewidth=0.8)

            # --- running pods: cumulative integral of delta running pods (pod-seconds) ---
            elif metric == "running_total_mean":
                x, d = delta_series_vs_default(default_df, df, metric)  # Δ running pods over time

                if smooth_s > 0:
                    t = pd.to_timedelta(x, unit="s")
                    s = pd.Series(d, index=t)
                    d = s.rolling(f"{float(smooth_s)}s", min_periods=1).mean().to_numpy(dtype=float)

                # ∫ Δn(t) dt  (pod-seconds)
                d_pod_seconds = cumtrapz_from_delta(x, d)

                # Convert to pods by dividing by elapsed time (running average)
                elapsed = x - x[0]
                elapsed = np.where(elapsed > 0, elapsed, np.nan)  # avoid divide-by-zero at t0
                d_avg_pods = d_pod_seconds / elapsed
                d_avg_pods[0] = 0.0  # define at start

                plt.plot(x, d_avg_pods, label=scheduler_row_label_hide_defpreempt(sched), linewidth=0.8)

            # --- latency: keep as delta vs default for time-series (optional, but consistent) ---
            elif metric == "latency_s_total_mean":
                x, d = delta_series_vs_default(default_df, df, metric)
                if smooth_s > 0:
                    t = pd.to_timedelta(x, unit="s")
                    s = pd.Series(d, index=t)
                    d = s.rolling(f"{float(smooth_s)}s", min_periods=1).mean().to_numpy(dtype=float)
                plt.plot(x, d, label=scheduler_row_label_hide_defpreempt(sched), linewidth=0.8)

            # fallback: plot delta
            else:
                x, d = delta_series_vs_default(default_df, df, metric)
                if smooth_s > 0:
                    t = pd.to_timedelta(x, unit="s")
                    s = pd.Series(d, index=t)
                    d = s.rolling(f"{float(smooth_s)}s", min_periods=1).mean().to_numpy(dtype=float)
                plt.plot(x, d, label=scheduler_row_label_hide_defpreempt(sched), linewidth=0.8)

        plt.xlabel("time (s)")

        if metric == "running_total_mean":
            plt.ylabel("Δ mean running pods vs Default")
            title = f"{job_name}: Δ mean running pods"
        elif metric == "deletions_cum_total_mean":
            plt.ylabel("Δ deletions (cum) vs Default")
            title = f"{job_name}: Δ deletions (cum)"
        else:
            plt.ylabel(f"Δ {metric} vs Default")
            title = f"{job_name}: Δ {metric}"

        plt.title(title)
        plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        plt.legend(loc="best", fontsize=8)

        png_path = job_dir / f"delta__{sanitize_filename(metric)}.png"
        pdf_path = job_dir / f"delta__{sanitize_filename(metric)}.pdf"
        plt.tight_layout()
        plt.savefig(png_path, dpi=dpi)
        plt.savefig(pdf_path)
        plt.close()
        
    plot_latency_histogram(
        job_dir=job_dir,
        job_name=job_name,
        default_df=default_df,
        plugin_series=plugin_series,
        dpi=dpi,
        bin_s=lat_hist_bin_s,
        max_s=lat_hist_max_s,
        density=lat_hist_density,
    )

def plot_latency_histogram(
    *,
    job_dir: Path,
    job_name: str,
    default_df: pd.DataFrame,
    plugin_series: Dict[str, pd.DataFrame],
    dpi: int,
    bin_s: float,
    max_s: float,
    density: bool,
) -> None:
    col = "latency_s_total_mean"
    if col not in default_df.columns:
        return

    # Collect series for histogram (Default + plugins)
    series_map: List[Tuple[str, np.ndarray]] = []
    y0 = pd.to_numeric(default_df[col], errors="coerce").to_numpy(dtype=float)
    y0 = y0[np.isfinite(y0)]
    series_map.append(("Default", y0))

    for sched, df in plugin_series.items():
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        series_map.append((scheduler_row_label_hide_defpreempt(sched), y))

    if not series_map:
        return

    # Determine histogram range
    all_vals = np.concatenate([v for _, v in series_map if v.size > 0])
    if all_vals.size == 0:
        return

    vmax = float(max_s) if max_s and max_s > 0 else float(np.quantile(all_vals, 0.995))
    vmax = max(vmax, float(bin_s))
    edges = np.arange(0.0, vmax + float(bin_s), float(bin_s))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # compute hist
    hists = []
    for name, vals in series_map:
        h, _ = np.histogram(vals, bins=edges, density=density)
        hists.append((name, h))

    # plot grouped bars
    import numpy as _np
    n = len(hists)
    width = (edges[1] - edges[0]) * 0.9 / max(1, n)

    plt.figure(figsize=(10.0, 4.5))
    for i, (name, h) in enumerate(hists):
        x = centers - 0.45 * (edges[1] - edges[0]) + (i + 0.5) * width
        plt.bar(x, h, width=width, label=name, alpha=0.8)

    plt.xlabel("latency_s_total_mean (s)")
    plt.ylabel("density" if density else "count")
    plt.title(f"{job_name}: Latency histogram (mean over time samples)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.legend(loc="best", fontsize=8)

    png_path = job_dir / "latency_hist.png"
    pdf_path = job_dir / "latency_hist.pdf"
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
        return fmt_signed(row.get(f"{prefix}1_mean", np.nan), decimals=decimals)

    vals = [fmt_signed(row.get(f"{prefix}{p}_mean", np.nan), decimals=decimals) for p in range(1, k_max + 1)]
    if all(v == r"\text{--}" for v in vals):
        return r"\text{--}"
    return r"{\scriptsize$\langle" + ",".join(vals) + r"\rangle$}"


def angle_pair_cell_from_row(
    row: pd.Series,
    *,
    col_a: str,
    col_b: str,
    decimals: int,
    as_percentage_points: bool = False,
) -> str:
    r"""Emit a 2-tuple in angle brackets."""
    if as_percentage_points:
        a = fmt_signed_scaled(row.get(col_a, np.nan), decimals=decimals, scale=100.0)
        b = fmt_signed_scaled(row.get(col_b, np.nan), decimals=decimals, scale=100.0)
    else:
        a = fmt_signed(row.get(col_a, np.nan), decimals=decimals)
        b = fmt_signed(row.get(col_b, np.nan), decimals=decimals)

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

    cell_maps: Dict[int, Dict[Tuple[str, int, float], str]] = {}
    row_keys_by_k: Dict[int, List[str]] = {}

    for k in k_values:
        dfk = df_metric_allk[df_metric_allk["k_max"].astype(int) == int(k)].copy()
        if dfk.empty:
            continue

        row_keys = sorted(
            dfk["plugin_config"].astype(str).unique(),
            key=lambda s: scheduler_row_label_hide_defpreempt(str(s)),
        )
        row_keys_by_k[int(k)] = row_keys

        cm: Dict[Tuple[str, int, float], str] = {}
        for _, r in dfk.iterrows():
            rk = str(r["plugin_config"])
            n = int(r["n_nodes"])
            a = float(r["mean_arrival_s"])
            cm[(rk, n, a)] = dfk.attrs["value_fn"](r)
        cell_maps[int(k)] = cm

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

    if len(nodes) > 1:
        parts = ["& "]
        for n in nodes:
            parts.append(r"\multicolumn{" + str(len(arrivals)) + r"}{c}{$N=" + str(n) + r"$}")
            parts.append(" & ")
        lines.append("".join(parts).rstrip(" & ") + r" \\")

        start = 2
        cmr = []
        for _ in nodes:
            end = start + len(arrivals) - 1
            cmr.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")
            start = end + 1
        lines.append("".join(cmr))
    else:
        lines.append(r"& \multicolumn{" + str(len(arrivals)) + r"}{c}{$N=" + str(nodes[0]) + r"$} \\")
        lines.append(r"\cmidrule(lr){2-" + str(1 + len(arrivals)) + "}")

    arr_hdr = ["& "]
    for _n in nodes:
        for a in arrivals:
            arr_hdr.append(r"$\mu_A{=}" + fmt_arrival(a) + r"$s")
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
            label_txt = scheduler_row_label_hide_defpreempt(rk)
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
        lines.append(r"\caption{" + str(caption_tex) + r"}")
    else:
        lines.append(r"\caption{" + latex_escape_text(f"{metric_name} (mean paired difference vs. baseline).") + r"}")

    if label is not None and str(label).strip():
        lines.append(r"\label{" + str(label) + r"}")
    else:
        safe_label = sanitize_filename(metric_name).replace("_", "-")
        lines.append(r"\label{tab:" + safe_label + r"}")

    lines.append(r"\end{table}")
    return "\n".join(lines)


def _latex_big_table(*, df_all: pd.DataFrame, decimals: int) -> str:
    """
    One big table. For each (N, mu_A) we create 4 columns in this order:
      1) Δutilisation = <Δcpu, Δmem> in percentage points (pp)
      2) Δlatency     = per-priority vector (seconds)
      3) ΔR_p(T)      = per-priority vector (pod-seconds)
      4) solver runs  = solver_attempts_total_mean (count)
    Rows: scheduler configs (ONLY defpreempt=1, but not stated in row label).
    """
    nodes = sorted({int(x) for x in df_all["n_nodes"].unique()})
    arrivals = sorted({float(x) for x in df_all["mean_arrival_s"].unique()})
    k_values = sorted({int(x) for x in pd.to_numeric(df_all["k_max"], errors="coerce").dropna().unique()})

    ncols = 1 + len(nodes) * len(arrivals) * 4

    def section_header(k: int) -> str:
        if k == 1:
            return rf"$k_{{\max}}={k}$ (no priorities)"
        return rf"$k_{{\max}}={k}$ (priorities enabled)"

    def util_cell(r: pd.Series) -> str:
        return angle_pair_cell_from_row(
            r,
            col_a="delta_cpu_run_util_mean",
            col_b="delta_mem_run_util_mean",
            decimals=decimals,
            as_percentage_points=True,
        )

    def latency_cell(r: pd.Series, k: int) -> str:
        kk = max(1, min(int(k), 4))
        if kk == 1:
            return fmt_signed(r.get("delta_latency_s_p1_mean", np.nan), decimals=decimals)
        vals = [fmt_signed(r.get(f"delta_latency_s_p{p}_mean", np.nan), decimals=decimals) for p in range(1, kk + 1)]
        if all(v == r"\text{--}" for v in vals):
            return r"\text{--}"
        return r"{\scriptsize$\langle" + ",".join(vals) + r"\rangle$}"

    def running_cell(r: pd.Series, k: int) -> str:
        kk = max(1, min(int(k), 4))
        if kk == 1:
            return fmt_signed(r.get("delta_R_p1_mean", np.nan), decimals=decimals)
        vals = [fmt_signed(r.get(f"delta_R_p{p}_mean", np.nan), decimals=decimals) for p in range(1, kk + 1)]
        if all(v == r"\text{--}" for v in vals):
            return r"\text{--}"
        return r"{\scriptsize$\langle" + ",".join(vals) + r"\rangle$}"

    def solver_cell(r: pd.Series) -> str:
        return fmt_signed(r.get("solver_attempts_total_mean", np.nan), decimals=0)

    cell_maps: Dict[int, Dict[Tuple[str, int, float], Tuple[str, str, str, str]]] = {}
    row_keys_by_k: Dict[int, List[str]] = {}

    for k in k_values:
        dfk = df_all[df_all["k_max"].astype(int) == int(k)].copy()
        if dfk.empty:
            continue

        row_keys = sorted(
            dfk["plugin_config"].astype(str).unique(),
            key=lambda s: scheduler_row_label_hide_defpreempt(str(s)),
        )
        row_keys_by_k[int(k)] = row_keys

        cm: Dict[Tuple[str, int, float], Tuple[str, str, str, str]] = {}
        for _, r in dfk.iterrows():
            rk = str(r["plugin_config"])
            n = int(r["n_nodes"])
            a = float(r["mean_arrival_s"])
            cm[(rk, n, a)] = (util_cell(r), latency_cell(r, int(k)), running_cell(r, int(k)), solver_cell(r))
        cell_maps[int(k)] = cm

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{2.2pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.90}")
    lines.append("")
    lines.append(r"\begin{adjustbox}{max width=\linewidth}")
    lines.append("")
    lines.append(r"\begin{tabular}{" + "l " + " ".join(["c"] * (ncols - 1)) + "}")
    lines.append(r"\toprule")

    title_line = (
        r"Mean paired deltas vs.\ baseline: "
        r"{\scriptsize$\langle\Delta u_{\mathrm{cpu}},\Delta u_{\mathrm{mem}}\rangle$} (pp), "
        r"$\Delta \mathrm{latency}$ (s), $\Delta R_p(T)$ (pod-seconds), solver runs"
    )
    lines.append(r"\multicolumn{" + str(ncols) + r"}{l}{" + title_line + r"} \\")
    lines.append(r"\addlinespace[0.2em]")

    # N header row (each N has len(arrivals)*4 cols)
    if len(nodes) > 1:
        parts = ["& "]
        for n in nodes:
            parts.append(r"\multicolumn{" + str(len(arrivals) * 4) + r"}{c}{$N=" + str(n) + r"$}")
            parts.append(" & ")
        lines.append("".join(parts).rstrip(" & ") + r" \\")
        start = 2
        cmr = []
        for _ in nodes:
            end = start + len(arrivals) * 4 - 1
            cmr.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")
            start = end + 1
        lines.append("".join(cmr))
    else:
        lines.append(r"& \multicolumn{" + str(len(arrivals) * 4) + r"}{c}{$N=" + str(nodes[0]) + r"$} \\")
        lines.append(r"\cmidrule(lr){2-" + str(1 + len(arrivals) * 4) + "}")

    # Arrival header row (each arrival has 4 cols)
    arr_hdr = ["& "]
    for _n in nodes:
        for a in arrivals:
            arr_hdr.append(r"\multicolumn{4}{c}{$\mu_A{=}" + fmt_arrival(a) + r"$s}")
            arr_hdr.append(" & ")
    lines.append("".join(arr_hdr).rstrip(" & ") + r" \\")
    start = 2
    cmr2 = []
    for _n in nodes:
        for _a in arrivals:
            end = start + 3
            cmr2.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")
            start = end + 1
    lines.append("".join(cmr2))

    # Subheader row: the 4 metric columns
    sub = ["& "]
    for _n in nodes:
        for _a in arrivals:
            sub += [r"$\Delta u$", " & ", r"$\Delta \mathrm{lat}$", " & ", r"$\Delta R_p$", " & ", r"solver", " & "]
    lines.append("".join(sub).rstrip(" & ") + r" \\")
    lines.append(r"\midrule")

    first_section = True
    for k in k_values:
        if int(k) not in cell_maps:
            continue
        if not first_section:
            lines.append(r"\midrule")
        first_section = False

        lines.append(r"\multicolumn{" + str(ncols) + r"}{l}{" + section_header(int(k)) + r"} \\")
        lines.append(r"\midrule")

        cm = cell_maps[int(k)]
        row_keys = row_keys_by_k.get(int(k), [])

        for rk in row_keys:
            label_txt = scheduler_row_label_hide_defpreempt(rk)
            row_cells: List[str] = [label_txt]
            for n in nodes:
                for a in arrivals:
                    util, lat, run, sol = cm.get((rk, n, a), (r"\text{--}", r"\text{--}", r"\text{--}", r"\text{--}"))
                    row_cells.extend([util, lat, run, sol])
            lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\caption{" + latex_escape_text("Big table of mean paired deltas vs baseline.") + r"}")
    lines.append(r"\label{tab:big-deltas}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _latex_latency_and_solver_table(*, df_all: pd.DataFrame, decimals: int) -> str:
    """
    Single table: latency(total) + solver runs for each (N, mu_A).
    Rows: scheduler configs (ONLY defpreempt=1, but not stated).
    Sections: k_max as usual.
    """
    nodes = sorted({int(x) for x in df_all["n_nodes"].unique()})
    arrivals = sorted({float(x) for x in df_all["mean_arrival_s"].unique()})
    k_values = sorted({int(x) for x in pd.to_numeric(df_all["k_max"], errors="coerce").dropna().unique()})

    # 2 columns per (n, a): latency_total + solver_attempts
    ncols = 1 + len(nodes) * len(arrivals) * 2

    def section_header(k: int) -> str:
        if k == 1:
            return rf"$k_{{\max}}={k}$ (no priorities)"
        return rf"$k_{{\max}}={k}$ (priorities enabled)"

    def lat_total_cell(r: pd.Series) -> str:
        return fmt_signed(r.get("delta_latency_s_total_mean", np.nan), decimals=decimals)

    def solver_cell(r: pd.Series) -> str:
        return fmt_signed(r.get("solver_attempts_total_mean", np.nan), decimals=0)

    cell_maps: Dict[int, Dict[Tuple[str, int, float], Tuple[str, str]]] = {}
    row_keys_by_k: Dict[int, List[str]] = {}

    for k in k_values:
        dfk = df_all[df_all["k_max"].astype(int) == int(k)].copy()
        if dfk.empty:
            continue

        row_keys = sorted(
            dfk["plugin_config"].astype(str).unique(),
            key=lambda s: scheduler_row_label_hide_defpreempt(str(s)),
        )
        row_keys_by_k[int(k)] = row_keys

        cm: Dict[Tuple[str, int, float], Tuple[str, str]] = {}
        for _, r in dfk.iterrows():
            rk = str(r["plugin_config"])
            n = int(r["n_nodes"])
            a = float(r["mean_arrival_s"])
            cm[(rk, n, a)] = (lat_total_cell(r), solver_cell(r))
        cell_maps[int(k)] = cm

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.92}")
    lines.append("")
    lines.append(r"\begin{adjustbox}{max width=\linewidth}")
    lines.append("")
    lines.append(r"\begin{tabular}{" + "l " + " ".join(["c"] * (ncols - 1)) + "}")
    lines.append(r"\toprule")

    title_line = r"Mean paired deltas vs.\ baseline: $\Delta \mathrm{latency}_{\mathrm{total}}$ (s) and solver runs"
    lines.append(r"\multicolumn{" + str(ncols) + r"}{l}{" + title_line + r"} \\")
    lines.append(r"\addlinespace[0.2em]")

    if len(nodes) > 1:
        parts = ["& "]
        for n in nodes:
            parts.append(r"\multicolumn{" + str(len(arrivals) * 2) + r"}{c}{$N=" + str(n) + r"$}")
            parts.append(" & ")
        lines.append("".join(parts).rstrip(" & ") + r" \\")
        start = 2
        cmr = []
        for _ in nodes:
            end = start + len(arrivals) * 2 - 1
            cmr.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")
            start = end + 1
        lines.append("".join(cmr))
    else:
        lines.append(r"& \multicolumn{" + str(len(arrivals) * 2) + r"}{c}{$N=" + str(nodes[0]) + r"$} \\")
        lines.append(r"\cmidrule(lr){2-" + str(1 + len(arrivals) * 2) + "}")

    arr_hdr = ["& "]
    for _n in nodes:
        for a in arrivals:
            arr_hdr.append(r"\multicolumn{2}{c}{$\mu_A{=}" + fmt_arrival(a) + r"$s}")
            arr_hdr.append(" & ")
    lines.append("".join(arr_hdr).rstrip(" & ") + r" \\")
    start = 2
    cmr2 = []
    for _n in nodes:
        for _a in arrivals:
            end = start + 1
            cmr2.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")
            start = end + 1
    lines.append("".join(cmr2))

    sub = ["& "]
    for _n in nodes:
        for _a in arrivals:
            sub += [r"$\Delta \mathrm{lat}_{\mathrm{tot}}$", " & ", r"solver", " & "]
    lines.append("".join(sub).rstrip(" & ") + r" \\")
    lines.append(r"\midrule")

    first_section = True
    for k in k_values:
        if int(k) not in cell_maps:
            continue
        if not first_section:
            lines.append(r"\midrule")
        first_section = False

        lines.append(r"\multicolumn{" + str(ncols) + r"}{l}{" + section_header(int(k)) + r"} \\")
        lines.append(r"\midrule")

        cm = cell_maps[int(k)]
        row_keys = row_keys_by_k.get(int(k), [])

        for rk in row_keys:
            label_txt = scheduler_row_label_hide_defpreempt(rk)
            row_cells: List[str] = [label_txt]
            for n in nodes:
                for a in arrivals:
                    lat, sol = cm.get((rk, n, a), (r"\text{--}", r"\text{--}"))
                    row_cells.extend([lat, sol])
            lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\caption{" + latex_escape_text("Latency(total) and solver runs (mean across seeds), reported as paired deltas vs baseline.") + r"}")
    lines.append(r"\label{tab:latency-solver}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def write_latex_tables(*, in_dir: Path, out_dir: Path, decimals: int) -> None:
    paired_path = in_dir / "results_paired.csv"
    if not paired_path.exists():
        print(f"[warn] not found: {paired_path} (skipping tables)")
        return

    df = pd.read_csv(paired_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    if "plugin_config" not in df.columns:
        raise SystemExit("results_paired.csv missing 'plugin_config' column")

    # derive n_nodes, k_max, mean_arrival_s if not present
    if not {"n_nodes", "k_max", "mean_arrival_s"}.issubset(df.columns):
        rows = []
        for jn in df["job_name"].astype(str).tolist():
            jk = parse_job_name(jn)
            rows.append((jk.n_nodes, jk.k_max, jk.arrival_s))
        df[["n_nodes", "k_max", "mean_arrival_s"]] = pd.DataFrame(rows, index=df.index)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    def scalar_value_fn(col: str):
        def _fn(r: pd.Series) -> str:
            return fmt_signed(r.get(col, np.nan), decimals=decimals)
        return _fn

    metric_defs = [
        {
            "name": "delta_R_p",
            "title": r"$\mathbf{\Delta \bar{n}_{p}}$ (mean running pods), mean paired difference vs.\ baseline",
            "kind": "prio_vector",
            "prefix": "delta_R_p",
            "caption_tex": (
                r"Mean paired difference in mean running pods per priority tier "
                r"$\Delta \bar{n}_p$ between plugin and baseline. For $k_{\max}>1$, cells report "
                r"${\scriptsize$\langle\Delta \bar{n}_1,\ldots,\Delta \bar{n}_{k_{\max}}\rangle$}$."
            ),
            "label": "tab:delta-mean-running-pods",
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
        {
            "name": "delta_run_util_mean",
            "title": r"$\mathbf{\Delta u_{\mathrm{run}}}$ (running utilisation, pp), mean paired difference vs.\ baseline",
            "kind": "angle_pair",
            "col_a": "delta_cpu_run_util_mean",
            "col_b": "delta_mem_run_util_mean",
            "caption_tex": (
                r"Mean paired difference in running utilisation between plugin and baseline. "
                r"Each cell reports {\scriptsize$\langle\Delta u_{\mathrm{cpu}},\Delta u_{\mathrm{mem}}\rangle$} "
                r"in percentage points (pp)."
            ),
            "label": "tab:delta-run-util",
        },
        {
            "name": "delta_R_total",
            "title": r"$\mathbf{\Delta \bar{n}_{\mathrm{total}}}$ (mean running pods), mean paired difference vs.\ baseline",
            "kind": "scalar",
            "col": "delta_R_total_mean",
            "caption_tex": (
                r"Mean paired difference in total mean running pods "
                r"$\Delta \bar{n}_{\mathrm{total}}$ between plugin and baseline."
            ),
            "label": "tab:delta-n-total",
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
        {
            "name": "delta_latency_total",
            "title": r"$\mathbf{\Delta \mathrm{latency}_{\mathrm{total}}}$ (s), mean paired difference vs.\ baseline",
            "kind": "scalar",
            "col": "delta_latency_s_total_mean",
            "caption_tex": (
                r"Mean paired difference in total latency $\Delta \mathrm{latency}_{\mathrm{total}}$ (seconds) "
                r"between plugin and baseline."
            ),
            "label": "tab:delta-latency-total",
        },
        {
            "name": "solver_attempts_total",
            "title": r"$\mathbf{\mathrm{solver\_attempts}}$ (count), mean across seeds",
            "kind": "scalar_int",
            "col": "solver_attempts_total_mean",
            "caption_tex": (
                r"Mean number of solver runs (solver attempts) across seeds for the plugin configuration. "
                r"(Baseline has no solver attempts.)"
            ),
            "label": "tab:solver-attempts",
        },
    ]

    for md in metric_defs:
        out_txt = tables_dir / f"{md['name']}.txt"
        df_metric = df.copy()

        if md["kind"] == "prio_vector":
            prefix = str(md["prefix"])

            def _dispatch(r: pd.Series) -> str:
                k = int(r["k_max"])
                return prio_cell_from_row(r, k_max=int(k), prefix=prefix, decimals=decimals)

            df_metric.attrs["value_fn"] = _dispatch

        elif md["kind"] == "angle_pair":
            col_a = str(md["col_a"])
            col_b = str(md["col_b"])
            if col_a not in df_metric.columns or col_b not in df_metric.columns:
                continue

            def _pair(r: pd.Series) -> str:
                return angle_pair_cell_from_row(
                    r,
                    col_a=col_a,
                    col_b=col_b,
                    decimals=decimals,
                    as_percentage_points=True,
                )

            df_metric.attrs["value_fn"] = _pair

        elif md["kind"] == "scalar_int":
            col = str(md["col"])
            if col not in df_metric.columns:
                continue

            def _int_cell(r: pd.Series) -> str:
                return fmt_signed(r.get(col, np.nan), decimals=0)

            df_metric.attrs["value_fn"] = _int_cell

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

    # ---- Big table + latency/solver table: ONLY defpreempt=1 ----
    df_big = df.copy()
    df_big["plugin_config"] = df_big["plugin_config"].astype(str)
    df_big = df_big[df_big["plugin_config"].str.contains(DEFPREEMPT_ONLY_REGEX.pattern, regex=True)].copy()

    required_big_cols = [
        "delta_cpu_run_util_mean",
        "delta_mem_run_util_mean",
        "delta_R_p1_mean",
        "delta_latency_s_p1_mean",
        "solver_attempts_total_mean",
    ]

    if not df_big.empty and all(c in df_big.columns for c in required_big_cols):
        (tables_dir / "big_deltas.txt").write_text(_latex_big_table(df_all=df_big, decimals=decimals) + "\n", encoding="utf-8")
        (tables_dir / "latency_and_solver.txt").write_text(
            _latex_latency_and_solver_table(df_all=df_big, decimals=decimals) + "\n",
            encoding="utf-8",
        )
    else:
        if df_big.empty:
            print("[warn] big tables: no rows after filtering defpreempt=1")
        else:
            missing = [c for c in required_big_cols if c not in df_big.columns]
            print(f"[warn] big tables: missing columns in results_paired.csv: {missing}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    series_root = in_dir / "series"
    if not series_root.exists():
        raise SystemExit(f"Missing series dir: {series_root}")

    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "report_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("This may take a while...")

    jobs, schedulers_all, paths = discover_jobs_and_schedulers(series_root)

    # Apply CLI scheduler filters first...
    schedulers_plot = filter_values(
        schedulers_all,
        only_csv=args.only_schedulers,
        include_regex=args.include_scheduler_regex,
        exclude_regex=args.exclude_scheduler_regex,
    )
    # ...then enforce "defpreempt only" for plots (but keep default)
    schedulers_plot = [s for s in schedulers_plot if _keep_scheduler_for_plots(s)]

    if not args.no_plots:
        for job_name in jobs:
            sched_paths: Dict[str, Path] = {}
            for sched in schedulers_plot:
                p = paths.get((sched, job_name))
                if p is not None:
                    sched_paths[sched] = p
            if not sched_paths:
                continue

            plot_job(
                job_name=job_name,
                scheduler_paths=sched_paths,
                out_dir=out_dir,
                fig_w=float(args.fig_w),
                fig_h=float(args.fig_h),
                dpi=int(args.dpi),
                only_metrics_csv=args.only_metrics,
                include_metric_regex=args.include_metric_regex,
                exclude_metric_regex=args.exclude_metric_regex,
                smooth_s=float(args.smooth_s),
                lat_hist_bin_s=float(args.lat_hist_bin_s),
                lat_hist_max_s=float(args.lat_hist_max_s),
                lat_hist_density=bool(args.lat_hist_density),
            )

    if not args.no_tables:
        write_latex_tables(in_dir=in_dir, out_dir=out_dir, decimals=int(args.table_decimals))

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
