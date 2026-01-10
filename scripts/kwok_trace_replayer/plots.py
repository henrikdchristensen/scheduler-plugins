# scripts/kwok_trace_replayer/compare_plots.py
# compare_plots.py

#######################################################################
"""
python -m scripts.kwok_trace_replayer.compare_plots \
    --default-dir <dir-with-default-csvs> \
    --plugin-dir  <dir-with-plugin-csvs> \
    [--plot both|default|plugin] \
  [--latency-bins 40] \
  [--latency-xmax <seconds>] \
  [--out-dir <dir>] \
  [--no-show]
"""
#######################################################################

import argparse
import math
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Input filenames (config)
# -----------------------------
GENERAL_STATS_FILENAME = "general_stats.csv"
POD_STATS_FILENAME = "pod_stats.csv"


# -----------------------------
# CSV schema (new)
# -----------------------------
TIME_COL = "time_s"
CPU_COL = "cpu_run_util"
MEM_COL = "mem_run_util"

# general_stats: running_pK, deletions_cum_pK
RUNNING_PREFIX = "running_p"
DELETIONS_CUM_PREFIX = "deletions_cum_p"

# pod_stats: timestamp,event,pod_name,pod_uid,priority,time_s
POD_EVENT_COL = "event"
POD_NAME_COL = "pod_name"
POD_UID_COL = "pod_uid"
POD_PRIO_COL = "priority"
POD_TIME_COL = "time_s"

_RS_PREFIX_RE = re.compile(r"^(rs-\d{6})(?:-.*)?$")


# -----------------------------
# Argparse
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare two scheduler runs using the new CSV outputs.\n\n"
            "Produces:\n"
            "  1) Effective utilization (max(cpu_run_util, mem_run_util)) with one line per scheduler.\n"
            "  2) Cumulative running pod-seconds difference (plugin - default), one line per priority (single plot).\n"
            "  3) Cumulative deletions difference (plugin - default), one line per priority (single plot).\n"
            "  4) ONE total scheduling-latency histogram (first-admit only): side-by-side bars per bin for default vs plugin,\n"
            "     and dashed vertical mean lines for both.\n"
        )
    )

    p.add_argument(
        "--plot",
        choices=["both", "default", "plugin"],
        default="both",
        help=(
            "Which run(s) to plot. "
            "'both' compares default vs plugin (diff plots + side-by-side histogram). "
            "'default' or 'plugin' plots only that run (no diffs)."
        ),
    )

    default_src = p.add_mutually_exclusive_group(required=False)
    default_src.add_argument(
        "--default-dir",
        default=None,
        help=(
            "Directory containing the default scheduler CSVs "
            f"({GENERAL_STATS_FILENAME} and {POD_STATS_FILENAME})."
        ),
    )
    default_src.add_argument(
        "--default-general",
        default=None,
        help=(
            f"[DEPRECATED] Path to default scheduler {GENERAL_STATS_FILENAME}. "
            "Prefer --default-dir."
        ),
    )

    p.add_argument(
        "--plugin-dir",
        default=None,
        help=(
            "Directory containing the plugin scheduler CSVs "
            f"({GENERAL_STATS_FILENAME} and {POD_STATS_FILENAME})."
        ),
    )

    p.add_argument(
        "--latency-bins",
        type=int,
        default=60,
        help="Number of latency histogram bins (default: 60).",
    )
    p.add_argument(
        "--latency-xmax",
        type=float,
        default=30.0,
        help="Optional max latency (seconds) to clip the histogram x-axis to.",
    )

    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write PNGs (default: plugin directory if provided, else default directory).",
    )

    p.add_argument("--no-show", action="store_true", help="Do not open an interactive window; just save figures.")
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def _choose_time_unit(max_time_s: float) -> tuple[float, str]:
    """
    Match trace_generator.py behavior:
      <= 7h   -> minutes
      <= 7d   -> hours
      else    -> days
    Returns (scale, label) where x_plot = x_seconds * scale.
    """
    if max_time_s <= 7 * 3600:
        return 1.0 / 60.0, "time (minutes)"
    if max_time_s <= 7 * 24 * 3600:
        return 1.0 / 3600.0, "time (hours)"
    return 1.0 / (24.0 * 3600.0), "time (days)"


def _time_scaled(
    t_def_s: np.ndarray,
    t_our_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    max_s = 0.0
    if t_def_s.size:
        max_s = max(max_s, float(np.nanmax(t_def_s)))
    if t_our_s.size:
        max_s = max(max_s, float(np.nanmax(t_our_s)))
    scale, label = _choose_time_unit(max_s)
    return t_def_s * scale, t_our_s * scale, label


def _rs_prefix_from_pod_name(pod_name: str) -> str:
    m = _RS_PREFIX_RE.match(pod_name or "")
    if m:
        return m.group(1)
    if "-" in (pod_name or ""):
        return (pod_name or "").split("-", 1)[0]
    return pod_name or ""


def _require_cols(df: pd.DataFrame, path: Path, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")


def _detect_priorities_from_general(df: pd.DataFrame) -> List[int]:
    prios: List[int] = []
    for c in df.columns:
        if c.startswith(RUNNING_PREFIX):
            try:
                prios.append(int(c[len(RUNNING_PREFIX) :]))
            except ValueError:
                pass
    prios = sorted(set(prios))
    if not prios:
        raise SystemExit("No priorities detected in general_stats.csv (no running_pK columns found).")
    return prios


def _detect_priorities_union(*dfs: pd.DataFrame) -> List[int]:
    prios: set[int] = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        for c in df.columns:
            if c.startswith(RUNNING_PREFIX):
                try:
                    prios.add(int(c[len(RUNNING_PREFIX) :]))
                except ValueError:
                    pass
    out = sorted(prios)
    if not out:
        raise SystemExit("No priorities detected in general_stats.csv (no running_pK columns found).")
    return out


def build_stepwise_diff(
    t_def: np.ndarray,
    y_def: np.ndarray,
    t_our: np.ndarray,
    y_our: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two stepwise-constant series on the union of timestamps and return (t, y_our - y_def).
    """
    if t_def.size == 0 and t_our.size == 0:
        return np.array([]), np.array([])

    od = np.argsort(t_def)
    t_def = t_def[od]
    y_def = y_def[od]

    oo = np.argsort(t_our)
    t_our = t_our[oo]
    y_our = y_our[oo]

    all_t = np.unique(np.concatenate([t_def, t_our]))

    i_def = -1
    i_our = -1
    last_def = 0.0
    last_our = 0.0

    out_t: List[float] = []
    out_d: List[float] = []

    for t in all_t:
        while i_def + 1 < t_def.size and t_def[i_def + 1] <= t:
            i_def += 1
            last_def = float(y_def[i_def])
        while i_our + 1 < t_our.size and t_our[i_our + 1] <= t:
            i_our += 1
            last_our = float(y_our[i_our])

        out_t.append(float(t))
        out_d.append(float(last_our - last_def))

    return np.asarray(out_t, dtype=float), np.asarray(out_d, dtype=float)


def _integrate_pod_seconds(t: np.ndarray, diff: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return np.array([], dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0.0, None)
    return np.cumsum(diff * dt)


def _read_pod_stats(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    _require_cols(
        df,
        p,
        [POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL],
    )
    df[POD_TIME_COL] = df[POD_TIME_COL].astype(float)
    df[POD_PRIO_COL] = df[POD_PRIO_COL].astype(int)
    df[POD_EVENT_COL] = df[POD_EVENT_COL].astype(str)
    df[POD_NAME_COL] = df[POD_NAME_COL].astype(str)
    df[POD_UID_COL] = df[POD_UID_COL].astype(str)
    return df


def _first_admit_events(
    pod_df: pd.DataFrame,
    *,
    eps_s: float = 1.0,
) -> pd.DataFrame:
    """
    Return first-admit events with:
      priority, admit_time_s, latency_s

    "First-admit only" heuristic for workload pods:
      - Group by rs prefix (e.g., rs-000022).
      - Take initial batch = pods whose apply-time is within eps_s of the first apply-time.
      - For those instances, latency = first running-time - apply-time.
    """
    apply_df = pod_df[pod_df[POD_EVENT_COL] == "apply-time"].copy()
    run_df = pod_df[pod_df[POD_EVENT_COL] == "running-time"].copy()

    apply_df = apply_df.sort_values(POD_TIME_COL).drop_duplicates(subset=[POD_UID_COL], keep="first")
    run_df = run_df.sort_values(POD_TIME_COL).drop_duplicates(subset=[POD_UID_COL], keep="first")

    apply_df["rs_prefix"] = apply_df[POD_NAME_COL].map(_rs_prefix_from_pod_name)

    joined = apply_df.merge(
        run_df[[POD_UID_COL, POD_TIME_COL]].rename(columns={POD_TIME_COL: "running_time_s"}),
        on=POD_UID_COL,
        how="left",
    ).rename(columns={POD_TIME_COL: "apply_time_s"})

    out_rows: List[Tuple[int, float, float]] = []

    for _, g in joined.groupby("rs_prefix", sort=False):
        g = g.sort_values("apply_time_s")
        if g.empty:
            continue

        t0 = float(g["apply_time_s"].iloc[0])
        batch_mask = (g["apply_time_s"] - t0) <= float(eps_s)
        batch_n = int(batch_mask.sum())
        if batch_n <= 0:
            continue

        first_batch = g.iloc[:batch_n]

        for _, row in first_batch.iterrows():
            rt = row.get("running_time_s")
            at = row.get("apply_time_s")
            pr = row.get(POD_PRIO_COL)
            if pd.isna(rt) or pd.isna(at) or pd.isna(pr):
                continue
            lat = float(rt) - float(at)
            if math.isfinite(lat) and lat >= 0.0:
                out_rows.append((int(pr), float(rt), float(lat)))

    out = pd.DataFrame(out_rows, columns=["priority", "admit_time_s", "latency_s"])
    if not out.empty:
        out = out.sort_values("admit_time_s")
    return out


# -----------------------------
# Plotting
# -----------------------------
def plot_effective_utilization(
    df_def: pd.DataFrame,
    df_our: pd.DataFrame,
    *,
    out_path: Path,
) -> None:
    t_def_s = df_def[TIME_COL].astype(float).to_numpy()
    t_our_s = df_our[TIME_COL].astype(float).to_numpy()
    t_def, t_our, x_label = _time_scaled(t_def_s, t_our_s)

    eff_def = np.maximum(df_def[CPU_COL].astype(float).to_numpy(), df_def[MEM_COL].astype(float).to_numpy())
    eff_our = np.maximum(df_our[CPU_COL].astype(float).to_numpy(), df_our[MEM_COL].astype(float).to_numpy())

    plt.figure(figsize=(8.5, 3.0))
    plt.plot(t_def, eff_def, linewidth=1.5, label="default")
    plt.plot(t_our, eff_our, linewidth=1.5, label="plugin")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel("effective utilization\nmax(CPU, MEM)")
    plt.title("Effective utilization")

    ymax = float(np.nanmax([np.nanmax(eff_def) if eff_def.size else 0.0, np.nanmax(eff_our) if eff_our.size else 0.0]))
    plt.ylim(0.0, max(1.0, 1.05 * ymax))

    plt.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_effective_utilization_single(
    df: pd.DataFrame,
    *,
    label: str,
    out_path: Path,
) -> None:
    t_s = df[TIME_COL].astype(float).to_numpy()
    scale, x_label = _choose_time_unit(float(np.nanmax(t_s)) if t_s.size else 0.0)
    t = t_s * scale

    eff = np.maximum(df[CPU_COL].astype(float).to_numpy(), df[MEM_COL].astype(float).to_numpy())

    plt.figure(figsize=(8.5, 3.0))
    plt.plot(t, eff, linewidth=1.5, label=label)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel("effective utilization\nmax(CPU, MEM)")
    plt.title("Effective utilization")

    ymax = float(np.nanmax(eff) if eff.size else 0.0)
    plt.ylim(0.0, max(1.0, 1.05 * ymax))

    plt.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_cumulative_running_pod_seconds_diff(
    df_def: pd.DataFrame,
    df_our: pd.DataFrame,
    prios: List[int],
    *,
    out_path: Path,
) -> None:
    t_def = df_def[TIME_COL].astype(float).to_numpy()
    t_our = df_our[TIME_COL].astype(float).to_numpy()

    plt.figure(figsize=(8.5, 3.3))

    all_cums: List[np.ndarray] = []

    for p in prios:
        col = f"{RUNNING_PREFIX}{p}"
        if col not in df_def.columns or col not in df_our.columns:
            continue

        y_def = df_def[col].fillna(0).astype(float).to_numpy()
        y_our = df_our[col].fillna(0).astype(float).to_numpy()

        x_s, diff = build_stepwise_diff(t_def, y_def, t_our, y_our)  # seconds
        cum = _integrate_pod_seconds(x_s, diff)                      # integrates in seconds (keep!)
        # choose unit based on *this plot's* horizon
        scale, x_label = _choose_time_unit(float(np.nanmax(x_s)) if x_s.size else 0.0)
        x = x_s * scale
        
        all_cums.append(cum)

        plt.plot(x, cum, linewidth=1.5, label=f"p{p}")

    plt.axhline(0.0, linewidth=0.8, linestyle="--")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel("cumulative running pod-seconds\n(plugin - default)")
    plt.title("Cumulative running pod-seconds difference (positive = plugin better)")
    plt.legend(frameon=False, ncol=min(6, max(1, len(prios))))

    if all_cums:
        ymin = min(float(np.nanmin(c)) for c in all_cums if c.size)
        ymax = max(float(np.nanmax(c)) for c in all_cums if c.size)
        if math.isfinite(ymin) and math.isfinite(ymax) and ymin != ymax:
            pad = 0.05 * (ymax - ymin)
            plt.ylim(ymin - pad, ymax + pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_cumulative_running_pod_seconds_single(
    df: pd.DataFrame,
    prios: List[int],
    *,
    label: str,
    out_path: Path,
) -> None:
    t_s = df[TIME_COL].astype(float).to_numpy()

    plt.figure(figsize=(8.5, 3.3))

    all_cums: List[np.ndarray] = []

    for p in prios:
        col = f"{RUNNING_PREFIX}{p}"
        if col not in df.columns:
            continue

        y = df[col].fillna(0).astype(float).to_numpy()
        od = np.argsort(t_s)
        t_sorted = t_s[od]
        y_sorted = y[od]

        # Integrate running pods over time => pod-seconds
        dt = np.diff(t_sorted, prepend=t_sorted[0])
        dt = np.clip(dt, 0.0, None)
        cum = np.cumsum(y_sorted * dt)

        scale, x_label = _choose_time_unit(float(np.nanmax(t_sorted)) if t_sorted.size else 0.0)
        x = t_sorted * scale

        all_cums.append(cum)
        plt.plot(x, cum, linewidth=1.5, label=f"p{p}")

    plt.axhline(0.0, linewidth=0.8, linestyle="--")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel("cumulative running pod-seconds")
    plt.title(f"Cumulative running pod-seconds ({label})")
    plt.legend(frameon=False, ncol=min(6, max(1, len(prios))))

    if all_cums:
        ymin = min(float(np.nanmin(c)) for c in all_cums if c.size)
        ymax = max(float(np.nanmax(c)) for c in all_cums if c.size)
        if math.isfinite(ymin) and math.isfinite(ymax) and ymin != ymax:
            pad = 0.05 * (ymax - ymin)
            plt.ylim(ymin - pad, ymax + pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_cumulative_deletions_diff(
    df_def: pd.DataFrame,
    df_our: pd.DataFrame,
    prios: List[int],
    *,
    out_path: Path,
) -> None:
    """
    One plot total: cumulative deletions diff (ours - default),
    one line per priority (same style as running pod-seconds plot).
    """
    t_def = df_def[TIME_COL].astype(float).to_numpy()
    t_our = df_our[TIME_COL].astype(float).to_numpy()

    plt.figure(figsize=(8.5, 3.3))

    all_series: List[np.ndarray] = []

    for p in prios:
        col = f"{DELETIONS_CUM_PREFIX}{p}"
        if col not in df_def.columns or col not in df_our.columns:
            continue

        y_def = df_def[col].fillna(0).astype(float).to_numpy()
        y_our = df_our[col].fillna(0).astype(float).to_numpy()

        x_s, diff = build_stepwise_diff(t_def, y_def, t_our, y_our)
        scale, x_label = _choose_time_unit(float(np.nanmax(x_s)) if x_s.size else 0.0)
        x = x_s * scale
        
        all_series.append(diff)

        plt.plot(x, diff, linewidth=1.5, label=f"p{p}")

    plt.axhline(0.0, linewidth=0.8, linestyle="--")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel("cumulative deletions diff\n(plugin - default)")
    plt.title("Cumulative deletions difference (positive = more deletions in plugin)")
    plt.legend(frameon=False, ncol=min(6, max(1, len(prios))))

    # Optional global y padding
    if all_series:
        ymin = min(float(np.nanmin(s)) for s in all_series if s.size)
        ymax = max(float(np.nanmax(s)) for s in all_series if s.size)
        if math.isfinite(ymin) and math.isfinite(ymax) and ymin != ymax:
            pad = 0.05 * (ymax - ymin)
            plt.ylim(ymin - pad, ymax + pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_cumulative_deletions_single(
    df: pd.DataFrame,
    prios: List[int],
    *,
    label: str,
    out_path: Path,
) -> None:
    t_s = df[TIME_COL].astype(float).to_numpy()

    plt.figure(figsize=(8.5, 3.3))

    all_series: List[np.ndarray] = []

    for p in prios:
        col = f"{DELETIONS_CUM_PREFIX}{p}"
        if col not in df.columns:
            continue

        y = df[col].fillna(0).astype(float).to_numpy()
        od = np.argsort(t_s)
        t_sorted = t_s[od]
        y_sorted = y[od]

        scale, x_label = _choose_time_unit(float(np.nanmax(t_sorted)) if t_sorted.size else 0.0)
        x = t_sorted * scale

        all_series.append(y_sorted)
        plt.plot(x, y_sorted, linewidth=1.5, label=f"p{p}")

    plt.axhline(0.0, linewidth=0.8, linestyle="--")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel("cumulative deletions")
    plt.title(f"Cumulative deletions ({label})")
    plt.legend(frameon=False, ncol=min(6, max(1, len(prios))))

    if all_series:
        ymin = min(float(np.nanmin(s)) for s in all_series if s.size)
        ymax = max(float(np.nanmax(s)) for s in all_series if s.size)
        if math.isfinite(ymin) and math.isfinite(ymax) and ymin != ymax:
            pad = 0.05 * (ymax - ymin)
            plt.ylim(ymin - pad, ymax + pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)



def plot_first_admit_latency_histogram_total(
    pod_def: pd.DataFrame,
    pod_our: pd.DataFrame,
    *,
    bins: int,
    xmax: float | None,
    out_path: Path,
) -> None:
    ev_def = _first_admit_events(pod_def, eps_s=1.0)
    ev_our = _first_admit_events(pod_our, eps_s=1.0)

    lat_def = ev_def["latency_s"].astype(float).to_numpy() if not ev_def.empty else np.array([], dtype=float)
    lat_our = ev_our["latency_s"].astype(float).to_numpy() if not ev_our.empty else np.array([], dtype=float)

    # Optional clipping for plotting range (keeps means computed on unclipped data)
    lat_def_plot = lat_def
    lat_our_plot = lat_our
    if xmax is not None and math.isfinite(float(xmax)):
        xmx = float(xmax)
        lat_def_plot = lat_def_plot[lat_def_plot <= xmx]
        lat_our_plot = lat_our_plot[lat_our_plot <= xmx]

    combined = np.concatenate([lat_def_plot, lat_our_plot]) if (lat_def_plot.size or lat_our_plot.size) else np.array([0.0])
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, max(1.0, hi if math.isfinite(hi) else 1.0)

    edges = np.linspace(lo, hi, num=max(2, int(bins) + 1))

    c_def, _ = np.histogram(lat_def_plot, bins=edges)
    c_our, _ = np.histogram(lat_our_plot, bins=edges)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    frac = 0.45
    w = widths * frac
    offset = widths * 0.5 * frac

    plt.figure(figsize=(8.5, 3.2))

    # Draw bars and capture their actual colors
    bars_def = plt.bar(centers - offset, c_def, width=w, align="center", label="default")
    bars_our = plt.bar(centers + offset, c_our, width=w, align="center", label="plugin")

    color_def = bars_def.patches[0].get_facecolor() if len(bars_def.patches) else None
    color_our = bars_our.patches[0].get_facecolor() if len(bars_our.patches) else None

    # Mean lines in matching colors
    if lat_def.size:
        plt.axvline(
            float(np.mean(lat_def)),
            linestyle="--",
            linewidth=1.4,
            color=color_def,
            label="default mean",
        )
    if lat_our.size:
        plt.axvline(
            float(np.mean(lat_our)),
            linestyle="--",
            linewidth=1.4,
            color=color_our,
            label="plugin mean",
        )

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel("scheduling latency (s)")
    plt.yscale("log")
    plt.ylabel("count")
    plt.title("Scheduling latency")
    plt.legend(frameon=False, ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_first_admit_latency_histogram_single(
    pod_df: pd.DataFrame,
    *,
    bins: int,
    xmax: float | None,
    label: str,
    out_path: Path,
) -> None:
    ev = _first_admit_events(pod_df, eps_s=1.0)
    lat = ev["latency_s"].astype(float).to_numpy() if not ev.empty else np.array([], dtype=float)

    lat_plot = lat
    if xmax is not None and math.isfinite(float(xmax)):
        xmx = float(xmax)
        lat_plot = lat_plot[lat_plot <= xmx]

    combined = lat_plot if lat_plot.size else np.array([0.0])
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, max(1.0, hi if math.isfinite(hi) else 1.0)

    edges = np.linspace(lo, hi, num=max(2, int(bins) + 1))
    counts, _ = np.histogram(lat_plot, bins=edges)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    plt.figure(figsize=(8.5, 3.2))
    bars = plt.bar(centers, counts, width=widths * 0.9, align="center", label=label)

    color = bars.patches[0].get_facecolor() if len(bars.patches) else None
    if lat.size:
        plt.axvline(
            float(np.mean(lat)),
            linestyle="--",
            linewidth=1.4,
            color=color,
            label=f"{label} mean",
        )

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel("scheduling latency (s)")
    plt.ylabel("count")
    plt.title(f"Scheduling latency ({label})")
    plt.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)



# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    # Resolve input paths
    def _resolve_general(which: str) -> Path | None:
        if which == "default":
            if args.default_dir:
                return Path(args.default_dir) / GENERAL_STATS_FILENAME
            if args.default_general:
                return Path(args.default_general)
            return None
        if which == "plugin":
            if args.plugin_dir:
                return Path(args.plugin_dir) / GENERAL_STATS_FILENAME
            return None
        raise ValueError(which)

    p_def_gen = _resolve_general("default")
    p_plugin_gen = _resolve_general("plugin")

    if args.plot in ("both", "default") and p_def_gen is None:
        raise SystemExit("Missing default input. Provide --default-dir (preferred) or --default-general.")
    if args.plot in ("both", "plugin") and p_plugin_gen is None:
        raise SystemExit("Missing plugin input. Provide --plugin-dir (preferred) or --plugin-general.")

    # Validate existence
    if p_def_gen is not None and not p_def_gen.exists():
        raise SystemExit(f"Not found: {p_def_gen}")
    if p_plugin_gen is not None and not p_plugin_gen.exists():
        raise SystemExit(f"Not found: {p_plugin_gen}")

    p_def_pod = (p_def_gen.parent / POD_STATS_FILENAME) if p_def_gen is not None else None
    p_plugin_pod = (p_plugin_gen.parent / POD_STATS_FILENAME) if p_plugin_gen is not None else None

    if p_def_pod is not None and not p_def_pod.exists():
        raise SystemExit(f"Not found: {p_def_pod}")
    if p_plugin_pod is not None and not p_plugin_pod.exists():
        raise SystemExit(f"Not found: {p_plugin_pod}")

    # Default output directory: plugin dir if present, else default dir.
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (p_plugin_gen.parent if p_plugin_gen is not None else p_def_gen.parent)  # type: ignore[union-attr]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load general_stats
    df_def = pd.read_csv(p_def_gen) if p_def_gen is not None else None
    df_plugin = pd.read_csv(p_plugin_gen) if p_plugin_gen is not None else None

    if df_def is not None:
        _require_cols(df_def, p_def_gen, [TIME_COL, CPU_COL, MEM_COL])  # type: ignore[arg-type]
    if df_plugin is not None:
        _require_cols(df_plugin, p_plugin_gen, [TIME_COL, CPU_COL, MEM_COL])  # type: ignore[arg-type]

    if args.plot == "both":
        prios = _detect_priorities_union(df_def, df_plugin)  # type: ignore[arg-type]
    elif args.plot == "default":
        prios = _detect_priorities_from_general(df_def)  # type: ignore[arg-type]
    else:
        prios = _detect_priorities_from_general(df_plugin)  # type: ignore[arg-type]

    # 1) Effective utilization
    if args.plot == "both":
        plot_effective_utilization(df_def, df_plugin, out_path=out_dir / "effective_utilization.png")  # type: ignore[arg-type]
    elif args.plot == "default":
        plot_effective_utilization_single(
            df_def, label="default", out_path=out_dir / "effective_utilization.png"  # type: ignore[arg-type]
        )
    else:
        plot_effective_utilization_single(
            df_plugin, label="plugin", out_path=out_dir / "effective_utilization.png"  # type: ignore[arg-type]
        )

    # 2) Running pod-seconds
    if args.plot == "both":
        plot_cumulative_running_pod_seconds_diff(
            df_def, df_plugin, prios, out_path=out_dir / "cumulative_running_pod_seconds_diff.png"  # type: ignore[arg-type]
        )
    elif args.plot == "default":
        plot_cumulative_running_pod_seconds_single(
            df_def, prios, label="default", out_path=out_dir / "cumulative_running_pod_seconds.png"  # type: ignore[arg-type]
        )
    else:
        plot_cumulative_running_pod_seconds_single(
            df_plugin, prios, label="plugin", out_path=out_dir / "cumulative_running_pod_seconds.png"  # type: ignore[arg-type]
        )

    # 3) Deletions
    if args.plot == "both":
        plot_cumulative_deletions_diff(
            df_def, df_plugin, prios, out_path=out_dir / "cumulative_deletions_diff.png"  # type: ignore[arg-type]
        )
    elif args.plot == "default":
        plot_cumulative_deletions_single(
            df_def, prios, label="default", out_path=out_dir / "cumulative_deletions.png"  # type: ignore[arg-type]
        )
    else:
        plot_cumulative_deletions_single(
            df_plugin, prios, label="plugin", out_path=out_dir / "cumulative_deletions.png"  # type: ignore[arg-type]
        )

    # 4) Scheduling latency histogram
    if args.plot == "both":
        pod_def = _read_pod_stats(p_def_pod)  # type: ignore[arg-type]
        pod_plugin = _read_pod_stats(p_plugin_pod)  # type: ignore[arg-type]
        plot_first_admit_latency_histogram_total(
            pod_def,
            pod_plugin,
            bins=max(5, int(args.latency_bins)),
            xmax=args.latency_xmax,
            out_path=out_dir / "scheduling_latency.png",
        )
    elif args.plot == "default":
        pod_def = _read_pod_stats(p_def_pod)  # type: ignore[arg-type]
        plot_first_admit_latency_histogram_single(
            pod_def,
            bins=max(5, int(args.latency_bins)),
            xmax=args.latency_xmax,
            label="default",
            out_path=out_dir / "scheduling_latency.png",
        )
    else:
        pod_plugin = _read_pod_stats(p_plugin_pod)  # type: ignore[arg-type]
        plot_first_admit_latency_histogram_single(
            pod_plugin,
            bins=max(5, int(args.latency_bins)),
            xmax=args.latency_xmax,
            label="plugin",
            out_path=out_dir / "scheduling_latency.png",
        )

    print(f"Saved plots to: {out_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
