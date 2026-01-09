# scripts/kwok_trace_replayer/compare_plots.py
# compare_plots.py

#######################################################################
"""
python -m scripts.kwok_trace_replayer.compare_plots \
  --default-general <path-to-default/general_stats.csv> \
  --python-general  <path-to-our/general_stats.csv> \
  [--default-pod <path-to-default/pod_stats.csv>] \
  [--python-pod  <path-to-our/pod_stats.csv>] \
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
            "  2) Cumulative running pod-seconds difference (ours - default), one line per priority (single plot).\n"
            "  3) Cumulative deletions difference (ours - default), one line per priority (single plot).\n"
            "  4) ONE total scheduling-latency histogram (first-admit only): side-by-side bars per bin for default vs ours,\n"
            "     and dashed vertical mean lines for both.\n"
        )
    )

    p.add_argument("--default-general", required=True, help="Path to default scheduler general_stats.csv")
    p.add_argument("--python-general", required=True, help="Path to our scheduler general_stats.csv")

    p.add_argument(
        "--default-pod",
        default=None,
        help="Path to default scheduler pod_stats.csv (default: alongside --default-general)",
    )
    p.add_argument(
        "--python-pod",
        default=None,
        help="Path to our scheduler pod_stats.csv (default: alongside --python-general)",
    )

    p.add_argument(
        "--latency-bins",
        type=int,
        default=40,
        help="Number of latency histogram bins (default: 40).",
    )
    p.add_argument(
        "--latency-xmax",
        type=float,
        default=None,
        help="Optional max latency (seconds) to clip the histogram x-axis to.",
    )

    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write PNGs (default: directory of --python-general).",
    )

    p.add_argument("--no-show", action="store_true", help="Do not open an interactive window; just save figures.")
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
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
    t_def = df_def[TIME_COL].astype(float).to_numpy()
    t_our = df_our[TIME_COL].astype(float).to_numpy()

    eff_def = np.maximum(df_def[CPU_COL].astype(float).to_numpy(), df_def[MEM_COL].astype(float).to_numpy())
    eff_our = np.maximum(df_our[CPU_COL].astype(float).to_numpy(), df_our[MEM_COL].astype(float).to_numpy())

    plt.figure(figsize=(8.5, 3.0))
    plt.plot(t_def, eff_def, linewidth=1.5, label="default")
    plt.plot(t_our, eff_our, linewidth=1.5, label="ours")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Effective utilization\nmax(CPU, MEM)")
    plt.title("Effective utilization")

    ymax = float(np.nanmax([np.nanmax(eff_def) if eff_def.size else 0.0, np.nanmax(eff_our) if eff_our.size else 0.0]))
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

        x, diff = build_stepwise_diff(t_def, y_def, t_our, y_our)  # ours - default
        cum = _integrate_pod_seconds(x, diff)
        all_cums.append(cum)

        plt.plot(x, cum, linewidth=1.5, label=f"p{p}")

    plt.axhline(0.0, linewidth=0.8, linestyle="--")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative running pod-seconds\n(ours - default)")
    plt.title("Cumulative running pod-seconds difference (positive = ours better)")
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

        x, diff = build_stepwise_diff(t_def, y_def, t_our, y_our)  # ours - default
        all_series.append(diff)

        plt.plot(x, diff, linewidth=1.5, label=f"p{p}")

    plt.axhline(0.0, linewidth=0.8, linestyle="--")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative deletions diff\n(ours - default)")
    plt.title("Cumulative deletions difference (positive = more deletions in ours)")
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
    bars_our = plt.bar(centers + offset, c_our, width=w, align="center", label="ours")

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
            label="ours mean",
        )

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel("Scheduling latency (s)")
    plt.ylabel("Count")
    plt.title("Scheduling latency (total)")
    plt.legend(frameon=False, ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)



# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    p_def_gen = Path(args.default_general)
    p_our_gen = Path(args.python_general)
    if not p_def_gen.exists():
        raise SystemExit(f"Not found: {p_def_gen}")
    if not p_our_gen.exists():
        raise SystemExit(f"Not found: {p_our_gen}")

    p_def_pod = Path(args.default_pod) if args.default_pod else (p_def_gen.parent / "pod_stats.csv")
    p_our_pod = Path(args.python_pod) if args.python_pod else (p_our_gen.parent / "pod_stats.csv")

    if not p_def_pod.exists():
        raise SystemExit(f"Not found: {p_def_pod}")
    if not p_our_pod.exists():
        raise SystemExit(f"Not found: {p_our_pod}")

    out_dir = Path(args.out_dir) if args.out_dir else p_our_gen.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load general_stats
    df_def = pd.read_csv(p_def_gen)
    df_our = pd.read_csv(p_our_gen)

    _require_cols(df_def, p_def_gen, [TIME_COL, CPU_COL, MEM_COL])
    _require_cols(df_our, p_our_gen, [TIME_COL, CPU_COL, MEM_COL])

    prios = _detect_priorities_from_general(df_def)

    # 1) Effective utilization (single plot)
    plot_effective_utilization(df_def, df_our, out_path=out_dir / "effective_utilization.png")

    # 2) Cumulative running pod-seconds diff (single plot, one line per priority)
    plot_cumulative_running_pod_seconds_diff(
        df_def, df_our, prios, out_path=out_dir / "cumulative_running_pod_seconds_diff.png"
    )

    # 3) Cumulative deletions diff: one plot per priority
    plot_cumulative_deletions_diff(
        df_def, df_our, prios, out_path=out_dir / "cumulative_deletions_diff.png"
    )

    # 4) Scheduling latency histogram (TOTAL): default vs ours, two bars per bin + mean lines
    pod_def = _read_pod_stats(p_def_pod)
    pod_our = _read_pod_stats(p_our_pod)
    plot_first_admit_latency_histogram_total(
        pod_def,
        pod_our,
        bins=max(5, int(args.latency_bins)),
        xmax=args.latency_xmax,
        out_path=out_dir / "scheduling_latency.png",
    )

    print(f"Saved plots to: {out_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
