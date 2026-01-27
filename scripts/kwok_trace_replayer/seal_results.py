#!/usr/bin/env python3
# scripts/kwok_trace_replayer/seal_results.py
"""
python -m scripts.kwok_trace_replayer.seal_results --root analysis/kwok_trace_replayer --out-dir analysis/kwok_trace_replayer
"""

import argparse, json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.kwok_trace_replayer.trace_helpers import rs_prefix_from_pod_name

# -----------------------------
# Constants
# -----------------------------

FLOAT_DECIMALS = 4

GENERAL_STATS_FILENAME = "general_stats.csv"
POD_STATS_FILENAME = "pod_stats.csv"
OPT_STATS_FILENAME = "optimization_stats.json"

TIME_COL = "time_s"

CPU_RUN_COL = "cpu_run_util"
MEM_RUN_COL = "mem_run_util"
RUNNING_PREFIX = "running_p"
DELETIONS_CUM_PREFIX = "deletions_cum_p"

POD_EVENT_COL = "event"
POD_NAME_COL = "pod_name"
POD_UID_COL = "pod_uid"
POD_PRIO_COL = "priority"
POD_TIME_COL = "time_s"

MAX_K_OUT = 4

# latency in seconds conversion
LATENCY_S_SCALE = 1000.0  # to milliseconds

# Util scaling:
# - internal util values are assumed in [0,1]
# - we STORE delta_U_* in percent (%) => multiply by 100
UTIL_TO_PERCENT = 100.0

PAIRED_COLS = [
    "job_name",
    "plugin_config",
    "n_seed",
    "T_end_s_mean",
    # STORED IN percent (%)
    "delta_U_pct_cpu_mean", "delta_U_pct_mem_mean", "delta_U_pct_eff_mean",
    "delta_R_num_p1_mean", "delta_R_num_p2_mean", "delta_R_num_p3_mean", "delta_R_num_p4_mean", "delta_R_num_total_mean",
    "delta_D_num_p1_mean", "delta_D_num_p2_mean", "delta_D_num_p3_mean", "delta_D_num_p4_mean", "delta_D_num_total_mean",
    "delta_L_ms_p1_mean", "delta_L_ms_p2_mean", "delta_L_ms_p3_mean", "delta_L_ms_p4_mean", "delta_L_ms_total_mean",
    "solver_attempts_mean", "solver_optimal_mean", "solver_feasible_mean", "solver_failed_mean",
    "plan_not_applicable_mean", "plan_activated_mean",
]

OPT_TOTAL_KEYS = {
    "solver_attempts_total": "solver_attempts",
    "best_solver_optimal_total": "solver_optimal",
    "best_solver_feasible_total": "solver_feasible",
    "best_solver_failed_total": "solver_failed",
    "plan_not_applicable_total": "plan_not_applicable",
    "plan_activated_total": "plan_activated",
}

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seal KWOK trace replayer outputs into results_paired.csv (fast path).")
    p.add_argument("--root", required=True, help="Root dir containing 'default/' and 'plugin/' subfolders.")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <root>/sealed).")
    p.add_argument("--eps-s", type=float, default=1.0, help="First-batch epsilon window in seconds (default: 1.0).")
    return p.parse_args()

# -----------------------------
# Metrics
# -----------------------------

@dataclass(frozen=True)
class GeneralData:
    t: np.ndarray  # normalized time (starts at 0)
    T_end: float
    cpu_util: np.ndarray
    mem_util: np.ndarray
    eff_util: np.ndarray
    pods_running: np.ndarray
    pods_deleted: np.ndarray
    cum_cpu_util: np.ndarray
    cum_mem_util: np.ndarray
    cum_eff_util: np.ndarray
    cum_pods_running: np.ndarray

def cummulative_time_integral(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute cumulative time integral of stepwise y over t.
    cumulative[i] = ∫_0^{t[i]} y(s) ds, with y stepwise using y[j] on [t[j], t[j+1]).
    Used for time-weighted means.
    Supports:
      - y shape (n,) -> returns shape (n,)
      - y shape (n, m) -> returns shape (n, m) with column-wise integrals
    """
    n = t.size
    y_arr = np.asarray(y)
    if y_arr.ndim not in (1, 2):
        raise ValueError(f"_cummulative_time_integral expects 1D or 2D y, got shape {y_arr.shape}")

    if n == 0:
        if y_arr.ndim == 1:
            return np.array([], dtype=float)
        return np.zeros((0, y_arr.shape[1]), dtype=float)
    if n == 1:
        if y_arr.ndim == 1:
            return np.zeros(1, dtype=float)
        return np.zeros((1, y_arr.shape[1]), dtype=float)

    dt = np.diff(t)

    if y_arr.ndim == 1:
        cum1 = np.zeros(n, dtype=float)
        cum1[1:] = np.cumsum(y_arr[:-1] * dt)
        return cum1

    dt_col = dt[:, None]
    cum2 = np.zeros((n, y_arr.shape[1]), dtype=float)
    cum2[1:, :] = np.cumsum(y_arr[:-1, :] * dt_col, axis=0)
    return cum2

def mean_over_horizon(t: np.ndarray, y: np.ndarray, cum: np.ndarray, H: float) -> float:
    """
    Time-weighted mean over [0, H] for stepwise y using cumulative integral.
    """
    if not (math.isfinite(H) and H > 0.0) or t.size == 0:
        return float("nan")
    Hc = min(H, float(t[-1]))
    idx = int(np.searchsorted(t, Hc, side="right") - 1)
    if idx < 0:
        idx = 0
    base = float(cum[idx])
    tail = float(y[idx]) * float(Hc - t[idx])
    return float((base + tail) / Hc) if Hc > 0 else float("nan")

def value_at_horizon(t: np.ndarray, y: np.ndarray, H: float) -> float:
    """
    Step value at time H (last observation carried forward).
    """
    if t.size == 0 or not math.isfinite(H):
        return float("nan")
    Hc = min(max(H, 0.0), float(t[-1]))
    idx = int(np.searchsorted(t, Hc, side="right") - 1)
    if idx < 0:
        idx = 0
    v = float(y[idx])
    return v if math.isfinite(v) else float("nan")

def compute_horizon_metrics(g: GeneralData, H: float) -> Dict[str, float]:
    """
    Compute horizon metrics over [0, H] from general data.
    Returns dict of metrics.
    """
    t = g.t
    out: Dict[str, float] = {
        "U_cpu_mean": mean_over_horizon(t, g.cpu_util, g.cum_cpu_util, H),
        "U_mem_mean": mean_over_horizon(t, g.mem_util, g.cum_mem_util, H),
        "U_eff_mean": mean_over_horizon(t, g.eff_util, g.cum_eff_util, H),
    }
    for i, p in enumerate(range(1, MAX_K_OUT + 1)):
        out[f"R_p{p}_mean"] = mean_over_horizon(t, g.pods_running[:, i], g.cum_pods_running[:, i], H)
        out[f"D_p{p}"] = value_at_horizon(t, g.pods_deleted[:, i], H)

    out["R_total_mean"] = float(np.nansum([out[f"R_p{p}_mean"] for p in range(1, MAX_K_OUT + 1)]))
    out["D_total"] = float(np.nansum([out[f"D_p{p}"] for p in range(1, MAX_K_OUT + 1)]))
    return out

def latency_means_first_batch_ms(pod_csv: Path, eps_s: float) -> Dict[str, float]:
    """
    Mean latency (running - apply) for first-batch pods, overall and per priority p1..p4.
    Returned in MILLISECONDS (ms).

    First-batch: for each rs_prefix, take t0=min(apply), include apply times within eps_s of t0.
    """
    nan_out = {**{f"L_ms_p{p}": float("nan") for p in range(1, MAX_K_OUT + 1)}, "L_ms_total": float("nan")}
    try:
        pod_df = read_pod(pod_csv)
    except Exception:
        return nan_out

    pod_df = pod_df.dropna(subset=[POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL]).copy()
    if pod_df.empty:
        return nan_out

    apply_time_df = pod_df[pod_df[POD_EVENT_COL] == "apply-time"].copy()
    running_time_df = pod_df[pod_df[POD_EVENT_COL] == "running-time"].copy()
    if apply_time_df.empty:
        return nan_out

    apply_time_df = apply_time_df.sort_values(POD_TIME_COL, kind="mergesort").drop_duplicates(subset=[POD_UID_COL], keep="first")
    running_time_df = running_time_df.sort_values(POD_TIME_COL, kind="mergesort").drop_duplicates(subset=[POD_UID_COL], keep="first")

    apply_time_df["rs_prefix"] = apply_time_df[POD_NAME_COL].map(rs_prefix_from_pod_name)

    joined = apply_time_df.merge(
        running_time_df[[POD_UID_COL, POD_TIME_COL]].rename(columns={POD_TIME_COL: "running_time_s"}),
        on=POD_UID_COL,
        how="left",
    ).rename(columns={POD_TIME_COL: "apply_time_s"})

    joined["apply_time_s"] = pd.to_numeric(joined["apply_time_s"], errors="coerce")
    joined["running_time_s"] = pd.to_numeric(joined["running_time_s"], errors="coerce")
    joined[POD_PRIO_COL] = pd.to_numeric(joined[POD_PRIO_COL], errors="coerce")
    joined = joined.dropna(subset=["rs_prefix", "apply_time_s", POD_PRIO_COL]).copy()
    if joined.empty:
        return nan_out

    joined = joined.sort_values(["rs_prefix", "apply_time_s"], kind="mergesort")
    t0 = joined.groupby("rs_prefix", sort=False)["apply_time_s"].transform("min")
    in_batch = (joined["apply_time_s"] - t0) <= float(eps_s)
    batch = joined.loc[in_batch].copy()
    if batch.empty:
        return nan_out

    # Compute latency in SECONDS, then convert to MS
    latency_s = batch["running_time_s"] - batch["apply_time_s"]
    latency_s = latency_s.where(np.isfinite(latency_s) & (latency_s >= 0.0), np.nan)
    batch["latency_ms"] = latency_s * LATENCY_S_SCALE

    out: Dict[str, float] = {}
    out["L_ms_total"] = float(np.nanmean(batch["latency_ms"].to_numpy(dtype=float))) if batch.shape[0] else float("nan")

    pod_prio = batch[POD_PRIO_COL].to_numpy(dtype=float)
    latency_value = batch["latency_ms"].to_numpy(dtype=float)

    for p in range(1, MAX_K_OUT + 1):
        mask = pod_prio == float(p)
        out[f"L_ms_p{p}"] = float(np.nanmean(latency_value[mask])) if np.any(mask) else float("nan")

    return out

# -----------------------------
# General Helpers
# -----------------------------

def round_numeric_df(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = exclude or []
    num_cols = [c for c in df.columns if c not in set(exclude) and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].round(FLOAT_DECIMALS)
    return df

def canonicalize_mode(mode: str) -> str:
    """
    Canonicalize mode so scheduling-failure uses the new convention:
      - schedulingfailure (no hyphen)
    """
    s = str(mode).strip().lower()
    s_compact = s.replace("-", "").replace("_", "")
    if s_compact in {"schedulingfailure", "schedfailure", "scheduingfailure"}:
        return "schedulingfailure"
    return s

def parse_job_dir_name(name: str) -> Optional[str]:
    """
    Parse job dir name.
    Example: nodes=16_prio=2_arrival=10s
    Returns canonical job_name: nodes=<N>_prio=<K>_arrival=<A>s
    """
    s = str(name)
    m_nodes = re.search(r"nodes=?(\d+)", s)
    m_prio = re.search(r"prio=?(\d+)", s)
    m_arrival = re.search(r"arrival=?([0-9.]+)s?", s)
    if not (m_nodes and m_prio and m_arrival):
        return None
    n = int(m_nodes.group(1))
    kmax = int(m_prio.group(1))
    arrival = float(m_arrival.group(1))
    arrival_str = str(int(arrival)) if abs(arrival - round(arrival)) < 1e-9 else f"{arrival:g}"
    return f"nodes={n}_prio={kmax}_arrival={arrival_str}s"

def parse_plugin_run_dir(name: str) -> Optional[Tuple[str, str]]:
    """
    Parse plugin run dir name.
    Example: mode=schedulingfailure_blocking=0_defpreempt=0_nodes=32_prio=1_arrival=4s

    Returns: (job_name, plugin_config)
        plugin_config: mode=<mode>_blocking=<0/1>_defpreempt=<0/1>
    """
    tokens = str(name).split("_")
    kv: Dict[str, str] = {}
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    try:
        n = int(kv["nodes"])
        kmax = int(kv["prio"])
        arrival_raw = kv["arrival"]
        arrival = float(arrival_raw[:-1] if arrival_raw.endswith("s") else arrival_raw)
    except Exception:
        return None
    arrival_str = str(int(arrival)) if abs(arrival - round(arrival)) < 1e-9 else f"{arrival:g}"
    job_name = f"nodes={n}_prio={kmax}_arrival={arrival_str}s"

    mode = canonicalize_mode(kv.get("mode", "unknown"))

    blocking = 1 if kv.get("blocking", "0").strip().lower() in {"1", "true", "yes"} else 0
    try:
        defpreempt = int(str(kv.get("defpreempt", "0")).strip())
    except Exception:
        defpreempt = 0

    plugin_config = f"mode={mode}_blocking={blocking}_defpreempt={defpreempt}"
    return job_name, plugin_config

def iter_seed_dirs(parent: Path) -> Iterable[Tuple[str, Path]]:
    if not parent.exists() or not parent.is_dir():
        return
    for d in sorted(parent.iterdir()):
        if not d.is_dir():
            continue
        if (d / GENERAL_STATS_FILENAME).exists() and (d / POD_STATS_FILENAME).exists():
            yield d.name, d

def read_optimization_stats(opt_json: Path) -> Dict[str, float]:
    out = {v: float("nan") for v in OPT_TOTAL_KEYS.values()}
    if not opt_json.exists():
        return out
    try:
        data = json.loads(opt_json.read_text(encoding="utf-8"))
        for src, dst in OPT_TOTAL_KEYS.items():
            if src in data:
                try:
                    out[dst] = float(data[src])
                except Exception:
                    out[dst] = float("nan")
    except Exception:
        return out
    return out

def read_pod(pod_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(
        pod_csv,
        usecols=[POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL],
        dtype={
            POD_EVENT_COL: "category",
            POD_NAME_COL: "string",
            POD_UID_COL: "string",
        },
        low_memory=False,
    )
    df[POD_TIME_COL] = pd.to_numeric(df[POD_TIME_COL], errors="coerce")
    df[POD_PRIO_COL] = pd.to_numeric(df[POD_PRIO_COL], errors="coerce")
    return df

def read_general_data(general_csv: Path) -> GeneralData:
    cols_needed = {TIME_COL, CPU_RUN_COL, MEM_RUN_COL}
    for p in range(1, MAX_K_OUT + 1):
        cols_needed.add(f"{RUNNING_PREFIX}{p}")
        cols_needed.add(f"{DELETIONS_CUM_PREFIX}{p}")

    df = pd.read_csv(
        general_csv,
        usecols=lambda c: c in cols_needed,
    )

    time_raw = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.loc[time_raw.notna()].copy()
    if df.empty:
        return GeneralData(
            t=np.array([], dtype=float),
            T_end=float("nan"),
            cpu_util=np.array([], dtype=float),
            mem_util=np.array([], dtype=float),
            eff_util=np.array([], dtype=float),
            pods_running=np.zeros((0, MAX_K_OUT), dtype=float),
            pods_deleted=np.zeros((0, MAX_K_OUT), dtype=float),
            cum_cpu_util=np.array([], dtype=float),
            cum_mem_util=np.array([], dtype=float),
            cum_eff_util=np.array([], dtype=float),
            cum_pods_running=np.zeros((0, MAX_K_OUT), dtype=float),
        )

    df[TIME_COL] = time_raw.loc[df.index].astype(float)
    df = df.sort_values(TIME_COL, kind="mergesort")
    t0 = float(df[TIME_COL].iloc[0])
    t = (df[TIME_COL].to_numpy(dtype=float) - t0).astype(float)

    valid = np.isfinite(t) & (t >= 0.0)
    if not np.any(valid):
        t = np.array([], dtype=float)
    else:
        df = df.loc[valid].copy()
        t = t[valid]

    if t.size == 0:
        return GeneralData(
            t=t,
            T_end=float("nan"),
            cpu_util=np.array([], dtype=float),
            mem_util=np.array([], dtype=float),
            eff_util=np.array([], dtype=float),
            pods_running=np.zeros((0, MAX_K_OUT), dtype=float),
            pods_deleted=np.zeros((0, MAX_K_OUT), dtype=float),
            cum_cpu_util=np.array([], dtype=float),
            cum_mem_util=np.array([], dtype=float),
            cum_eff_util=np.array([], dtype=float),
            cum_pods_running=np.zeros((0, MAX_K_OUT), dtype=float),
        )

    T_end = float(t[-1])

    util_cpu_run = (
        pd.to_numeric(df[CPU_RUN_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if CPU_RUN_COL in df.columns
        else np.zeros_like(t, dtype=float)
    )
    mem_util_run = (
        pd.to_numeric(df[MEM_RUN_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if MEM_RUN_COL in df.columns
        else np.zeros_like(t, dtype=float)
    )
    eff_util_run = np.maximum(util_cpu_run, mem_util_run)

    pods_running = np.zeros((t.size, MAX_K_OUT), dtype=float)
    pods_deleted = np.full((t.size, MAX_K_OUT), np.nan, dtype=float)

    for i, p in enumerate(range(1, MAX_K_OUT + 1)):
        rc = f"{RUNNING_PREFIX}{p}"
        dc = f"{DELETIONS_CUM_PREFIX}{p}"
        if rc in df.columns:
            pods_running[:, i] = pd.to_numeric(df[rc], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if dc in df.columns:
            pods_deleted[:, i] = pd.to_numeric(df[dc], errors="coerce").to_numpy(dtype=float)

    cum_cpu_util_run = cummulative_time_integral(t, util_cpu_run)
    cum_mem_util_run = cummulative_time_integral(t, mem_util_run)
    cum_eff_util_run = cummulative_time_integral(t, eff_util_run)
    cum_pods_running = cummulative_time_integral(t, pods_running)

    return GeneralData(
        t=t,
        T_end=T_end,
        cpu_util=util_cpu_run,
        mem_util=mem_util_run,
        eff_util=eff_util_run,
        pods_running=pods_running,
        pods_deleted=pods_deleted,
        cum_cpu_util=cum_cpu_util_run,
        cum_mem_util=cum_mem_util_run,
        cum_eff_util=cum_eff_util_run,
        cum_pods_running=cum_pods_running,
    )

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    root = Path(args.root)
    default_root = root / "default"
    plugin_root = root / "plugin"

    if not default_root.exists():
        raise SystemExit(f"Not found: {default_root}")
    if not plugin_root.exists():
        raise SystemExit(f"Not found: {plugin_root}")

    out_dir = Path(args.out_dir) if args.out_dir else (root / "sealed")
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_s = float(args.eps_s)

    print("Sealing results...")

    default_idx: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
    default_seeds_by_job: Dict[str, set[str]] = {}

    for job_dir in sorted(default_root.iterdir()):
        if not job_dir.is_dir():
            continue
        job_name = parse_job_dir_name(job_dir.name)
        if job_name is None:
            continue
        seeds = set()
        for seed, seed_dir in iter_seed_dirs(job_dir):
            seeds.add(seed)
            default_idx[(job_name, seed)] = (seed_dir / GENERAL_STATS_FILENAME, seed_dir / POD_STATS_FILENAME)
        if seeds:
            default_seeds_by_job[job_name] = seeds

    seed_rows: List[Dict[str, object]] = []

    for run_dir in sorted(plugin_root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_plugin_run_dir(run_dir.name)
        if parsed is None:
            continue
        job_name, plugin_config = parsed

        default_seeds = default_seeds_by_job.get(job_name, set())
        if not default_seeds:
            print(f"[warn] no default seeds for job={job_name}")
            continue

        plugin_seeds = {seed for seed, _ in iter_seed_dirs(run_dir)}
        common_seeds = sorted(plugin_seeds & default_seeds)
        if not common_seeds:
            print(f"[warn] no common seeds for job={job_name} plugin={plugin_config}")
            continue

        for seed in common_seeds:
            default_general_stats_path, default_pod_stats = default_idx[(job_name, seed)]
            plugin_seed_dir = run_dir / seed
            plugin_general_stats_path = plugin_seed_dir / GENERAL_STATS_FILENAME
            plugin_pod_stats = plugin_seed_dir / POD_STATS_FILENAME
            plugin_opt_stats = plugin_seed_dir / OPT_STATS_FILENAME

            try:
                default_general_stats = read_general_data(default_general_stats_path)
                plugin_general_stats = read_general_data(plugin_general_stats_path)
            except Exception:
                continue

            if not (math.isfinite(default_general_stats.T_end) and math.isfinite(plugin_general_stats.T_end)):
                continue
            if default_general_stats.T_end <= 0.0 or plugin_general_stats.T_end <= 0.0:
                continue
            H = float(min(default_general_stats.T_end, plugin_general_stats.T_end))
            if H <= 0.0:
                continue

            default_metrics = compute_horizon_metrics(default_general_stats, H)
            plugin_metrics = compute_horizon_metrics(plugin_general_stats, H)

            # Util deltas (fraction) -> STORE IN percent (%)
            dU_cpu_pct = (float(plugin_metrics["U_cpu_mean"]) - float(default_metrics["U_cpu_mean"])) * UTIL_TO_PERCENT
            dU_mem_pct = (float(plugin_metrics["U_mem_mean"]) - float(default_metrics["U_mem_mean"])) * UTIL_TO_PERCENT
            dU_eff_pct = (float(plugin_metrics["U_eff_mean"]) - float(default_metrics["U_eff_mean"])) * UTIL_TO_PERCENT

            dR = {p: float(plugin_metrics[f"R_p{p}_mean"]) - float(default_metrics[f"R_p{p}_mean"]) for p in range(1, MAX_K_OUT + 1)}
            dD = {p: float(plugin_metrics[f"D_p{p}"]) - float(default_metrics[f"D_p{p}"]) for p in range(1, MAX_K_OUT + 1)}
            dR_total = float(np.nansum(list(dR.values())))
            dD_total = float(np.nansum(list(dD.values())))

            # Latency (MS)
            default_L = latency_means_first_batch_ms(default_pod_stats, eps_s=eps_s)
            plugin_L = latency_means_first_batch_ms(plugin_pod_stats, eps_s=eps_s)
            dL_total = float(plugin_L["L_ms_total"]) - float(default_L["L_ms_total"])
            dL_prio = {p: float(plugin_L[f"L_ms_p{p}"]) - float(default_L[f"L_ms_p{p}"]) for p in range(1, MAX_K_OUT + 1)}

            opt_totals = read_optimization_stats(plugin_opt_stats)

            seed_rows.append(
                {
                    "job_name": job_name,
                    "plugin_config": plugin_config,
                    "seed": seed,
                    "T_end_s": H,

                    # STORED IN percent (%)
                    "delta_U_pct_cpu": dU_cpu_pct,
                    "delta_U_pct_mem": dU_mem_pct,
                    "delta_U_pct_eff": dU_eff_pct,

                    "delta_R_num_p1": dR[1],
                    "delta_R_num_p2": dR[2],
                    "delta_R_num_p3": dR[3],
                    "delta_R_num_p4": dR[4],
                    "delta_R_num_total": dR_total,

                    "delta_D_num_p1": dD[1],
                    "delta_D_num_p2": dD[2],
                    "delta_D_num_p3": dD[3],
                    "delta_D_num_p4": dD[4],
                    "delta_D_num_total": dD_total,

                    "delta_L_ms_p1": dL_prio[1],
                    "delta_L_ms_p2": dL_prio[2],
                    "delta_L_ms_p3": dL_prio[3],
                    "delta_L_ms_p4": dL_prio[4],
                    "delta_L_ms_total": dL_total,

                    "solver_attempts": float(opt_totals["solver_attempts"]),
                    "solver_optimal": float(opt_totals["solver_optimal"]),
                    "solver_feasible": float(opt_totals["solver_feasible"]),
                    "solver_failed": float(opt_totals["solver_failed"]),
                    "plan_not_applicable": float(opt_totals["plan_not_applicable"]),
                    "plan_activated": float(opt_totals["plan_activated"]),
                }
            )

    df_seed = pd.DataFrame(seed_rows)
    if df_seed.empty:
        pd.DataFrame(columns=PAIRED_COLS).to_csv(out_dir / "results_paired.csv", index=False)
        print(f"Wrote outputs to: {out_dir}")
        return

    grp = df_seed.groupby(["job_name", "plugin_config"], dropna=False)
    agg = grp.mean(numeric_only=True).reset_index()
    agg.insert(2, "n_seed", grp.size().to_numpy())

    rename = {
        "T_end_s": "T_end_s_mean",
        "delta_U_pct_cpu": "delta_U_pct_cpu_mean",
        "delta_U_pct_mem": "delta_U_pct_mem_mean",
        "delta_U_pct_eff": "delta_U_pct_eff_mean",
        "delta_R_num_p1": "delta_R_num_p1_mean",
        "delta_R_num_p2": "delta_R_num_p2_mean",
        "delta_R_num_p3": "delta_R_num_p3_mean",
        "delta_R_num_p4": "delta_R_num_p4_mean",
        "delta_R_num_total": "delta_R_num_total_mean",
        "delta_D_num_p1": "delta_D_num_p1_mean",
        "delta_D_num_p2": "delta_D_num_p2_mean",
        "delta_D_num_p3": "delta_D_num_p3_mean",
        "delta_D_num_p4": "delta_D_num_p4_mean",
        "delta_D_num_total": "delta_D_num_total_mean",
        "delta_L_ms_p1": "delta_L_ms_p1_mean",
        "delta_L_ms_p2": "delta_L_ms_p2_mean",
        "delta_L_ms_p3": "delta_L_ms_p3_mean",
        "delta_L_ms_p4": "delta_L_ms_p4_mean",
        "delta_L_ms_total": "delta_L_ms_total_mean",
        "solver_attempts": "solver_attempts_mean",
        "solver_optimal": "solver_optimal_mean",
        "solver_feasible": "solver_feasible_mean",
        "solver_failed": "solver_failed_mean",
        "plan_not_applicable": "plan_not_applicable_mean",
        "plan_activated": "plan_activated_mean",
    }
    agg = agg.rename(columns=rename)

    for c in PAIRED_COLS:
        if c not in agg.columns:
            agg[c] = np.nan
    agg = agg[PAIRED_COLS]

    round_numeric_df(agg, exclude=["job_name", "plugin_config", "n_seed"])
    agg.to_csv(out_dir / "results_paired.csv", index=False)

    print(f"Wrote outputs to: {out_dir}")

if __name__ == "__main__":
    main()
