#!/usr/bin/env python3
# seal_results.py
"""
python -m scripts.kwok_trace_replayer.seal_results
"""

import json, math, re, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.helpers.data_helpers import round_numeric_df
from scripts.kwok_trace_replayer.trace_helpers import rs_prefix_from_pod_name

# =============================================================================
# CONFIG
# =============================================================================

ROOT_DIR = Path("analysis/kwok_trace_replayer")
DEFAULT_ROOT = ROOT_DIR / "default"
PLUGIN_ROOT = ROOT_DIR / "plugin"

OUT_DIR = ROOT_DIR
OUT_FILENAME = "results_seeds.csv"

# First-batch epsilon window (seconds)
# eps_s=1.0 means: include pods with apply_time <= t0 + 1.0s,
# where t0 = min(apply_time) for the rs_prefix.
EPS_S = 1.0

# Number of decimals for float rounding in output CSV
FLOAT_DECIMALS = 4

# =============================================================================
# Filenames / columns
# =============================================================================

JOB_RE = re.compile(r"nodes=?(\d+).*prio=?(\d+).*arrival=?([0-9.]+)s?", re.IGNORECASE)

GENERAL_STATS_FILENAME = "general_stats.csv"
POD_STATS_FILENAME = "pod_stats.csv"
OPT_STATS_FILENAME = "optimization_stats.json"

# Columns in replayer CSVs
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

MAX_PRIORITIES = 4

# Output columns
OUT_COLS = [
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

# Optimization stats keys mapping
OPT_TOTAL_KEYS = {
    "solver_attempts_total": "solver_attempts",
    "best_solver_optimal_total": "solver_optimal",
    "best_solver_feasible_total": "solver_feasible",
    "best_solver_failed_total": "solver_failed",
    "plan_not_applicable_total": "plan_not_applicable",
    "plan_activated_total": "plan_activated",
}

# =============================================================================
# Types
# =============================================================================

@dataclass(frozen=True)
class Data:
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

# =============================================================================
# Parsing helpers
# =============================================================================

def arrival_str(arrival: float) -> str:
    """
    Format arrival time as string, removing trailing .0 if integer.
    """
    return str(int(arrival)) if abs(arrival - round(arrival)) < 1e-9 else f"{arrival:g}"

def parse_default_run_dir(name: str) -> Optional[str]:
    """
    Parse default run dir into job_name.
    Example: nodes=16_prio=2_arrival=10s  -> nodes=16_prio=2_arrival=10s
    """
    m = JOB_RE.search(str(name))
    if not m:
        return None
    nodes = int(m.group(1))
    priorities = int(m.group(2))
    arrival = float(m.group(3))
    return f"nodes={nodes}_prio={priorities}_arrival={arrival_str(arrival)}s"

def parse_plugin_run_dir(name: str) -> Optional[Tuple[str, str]]:
    """
    Parse plugin run dir name into (job_name, plugin_config).
    Example: mode=schedulingfailure_blocking=0_defpreempt=0_nodes=32_prio=1_arrival=4s
    """
    kv: Dict[str, str] = {}
    for token in str(name).split("_"):
        if "=" in token:
            key, value = token.split("=", 1)
            kv[key.strip().lower()] = value.strip()
    try:
        nodes = int(kv["nodes"])
        priorities = int(kv["prio"])
        arrival_raw = kv["arrival"]
        arrival = float(arrival_raw[:-1] if arrival_raw.endswith("s") else arrival_raw)
    except Exception:
        return None

    job_name = f"nodes={nodes}_prio={priorities}_arrival={arrival_str(arrival)}s"
    mode = kv.get("mode", "unknown").strip().lower()
    blocking = 1 if kv.get("blocking", "0").strip().lower() in {"1", "true", "yes"} else 0
    defpreempt = int(str(kv.get("defpreempt", "0")).strip()) if "defpreempt" in kv else 0
    plugin_config = f"mode={mode}_blocking={blocking}_defpreempt={defpreempt}"
    return job_name, plugin_config

def iter_seed_dirs(parent: Path) -> Iterable[Tuple[str, Path]]:
    """
    Yields (seed, seed_dir) for dirs that contain the expected files.
    """
    if not parent.exists() or not parent.is_dir():
        return
    for d in sorted(parent.iterdir()):
        if not d.is_dir():
            continue
        if (d / GENERAL_STATS_FILENAME).exists() and (d / POD_STATS_FILENAME).exists():
            yield d.name, d

# =============================================================================
# I/O helpers
# =============================================================================

def read_optimization_stats(opt_json: Path) -> Dict[str, float]:
    """
    Read optimization stats from JSON file.
    ️If file does not exist, returns NaNs.
    """
    out = {v: float("nan") for v in OPT_TOTAL_KEYS.values()}
    if not opt_json.exists():
        return out
    data = json.loads(opt_json.read_text(encoding="utf-8"))
    for src, dst in OPT_TOTAL_KEYS.items():
        if src in data:
            try:
                out[dst] = float(data[src])
            except Exception:
                out[dst] = float("nan")
    return out

def read_pod(pod_csv: Path) -> pd.DataFrame:
    """
    Read pod CSV with needed columns and types.
    """
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

# =============================================================================
# Math helpers
# =============================================================================

def cumulative_integral_step(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    cumulative[i] = ∫_0^{t[i]} y(s) ds, with y stepwise using y[j] on [t[j], t[j+1]).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = t.size
    if n == 0:
        return np.array([], dtype=float) if y.ndim == 1 else np.zeros((0, y.shape[1]), dtype=float)
    if n == 1:
        return np.zeros((1,), dtype=float) if y.ndim == 1 else np.zeros((1, y.shape[1]), dtype=float)
    dt = np.diff(t)
    if y.ndim == 1:
        out = np.zeros(n, dtype=float)
        out[1:] = np.cumsum(y[:-1] * dt)
        return out
    out2 = np.zeros((n, y.shape[1]), dtype=float)
    out2[1:, :] = np.cumsum(y[:-1, :] * dt[:, None], axis=0)
    return out2

def mean_over_horizon(t: np.ndarray, y: np.ndarray, cum: np.ndarray, T_end: float) -> float:
    """
    Mean of y over horizon [0, T_end]:
    Hc means the effective horizon, min(T_end, t[-1]). That is, if T_end exceeds
    the last timestamp, we consider the function y to be constant after t[-1].
    """
    Hc = min(T_end, float(t[-1]))
    idx = int(np.searchsorted(t, Hc, side="right") - 1)
    idx = max(idx, 0)
    base = float(cum[idx])
    tail = float(y[idx]) * float(Hc - t[idx])
    return float((base + tail) / Hc) if Hc > 0 else float("nan")

def value_at_horizon(t: np.ndarray, y: np.ndarray, T_end: float) -> float:
    """
    Value of y at horizon T_end.
    """
    Hc = min(max(T_end, 0.0), float(t[-1]))
    idx = int(np.searchsorted(t, Hc, side="right") - 1)
    idx = max(idx, 0)
    v = float(y[idx])
    return v if math.isfinite(v) else float("nan")

# =============================================================================
# General stats reading + horizon metrics
# =============================================================================

def read_general_data(general_csv: Path) -> Data:
    """
    Read general stats CSV into Data.
    """
    cols_needed = {TIME_COL, CPU_RUN_COL, MEM_RUN_COL}
    for p in range(1, MAX_PRIORITIES + 1):
        cols_needed.add(f"{RUNNING_PREFIX}{p}")
        cols_needed.add(f"{DELETIONS_CUM_PREFIX}{p}")

    df = pd.read_csv(general_csv, usecols=lambda c: c in cols_needed)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.loc[df[TIME_COL].notna()].copy()
    if df.empty:
        return Data(
            t=np.array([], dtype=float),
            T_end=float("nan"),
            cpu_util=np.array([], dtype=float),
            mem_util=np.array([], dtype=float),
            eff_util=np.array([], dtype=float),
            pods_running=np.zeros((0, MAX_PRIORITIES), dtype=float),
            pods_deleted=np.zeros((0, MAX_PRIORITIES), dtype=float),
            cum_cpu_util=np.array([], dtype=float),
            cum_mem_util=np.array([], dtype=float),
            cum_eff_util=np.array([], dtype=float),
            cum_pods_running=np.zeros((0, MAX_PRIORITIES), dtype=float),
        )

    df = df.sort_values(TIME_COL, kind="mergesort")
    t0 = float(df[TIME_COL].iloc[0])
    t = (df[TIME_COL].to_numpy(dtype=float) - t0).astype(float)
    valid = np.isfinite(t) & (t >= 0.0)
    df = df.loc[valid].copy()
    t = t[valid]
    if t.size == 0:
        return Data(
            t=t,
            T_end=float("nan"),
            cpu_util=np.array([], dtype=float),
            mem_util=np.array([], dtype=float),
            eff_util=np.array([], dtype=float),
            pods_running=np.zeros((0, MAX_PRIORITIES), dtype=float),
            pods_deleted=np.zeros((0, MAX_PRIORITIES), dtype=float),
            cum_cpu_util=np.array([], dtype=float),
            cum_mem_util=np.array([], dtype=float),
            cum_eff_util=np.array([], dtype=float),
            cum_pods_running=np.zeros((0, MAX_PRIORITIES), dtype=float),
        )

    cpu = pd.to_numeric(df.get(CPU_RUN_COL, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mem = pd.to_numeric(df.get(MEM_RUN_COL, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    eff = np.maximum(cpu, mem)

    pods_running = np.zeros((t.size, MAX_PRIORITIES), dtype=float)
    pods_deleted = np.full((t.size, MAX_PRIORITIES), np.nan, dtype=float)

    for i, p in enumerate(range(1, MAX_PRIORITIES + 1)):
        rc = f"{RUNNING_PREFIX}{p}"
        dc = f"{DELETIONS_CUM_PREFIX}{p}"
        if rc in df.columns:
            pods_running[:, i] = pd.to_numeric(df[rc], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if dc in df.columns:
            pods_deleted[:, i] = pd.to_numeric(df[dc], errors="coerce").to_numpy(dtype=float)

    return Data(
        t=t,
        T_end=float(t[-1]),
        cpu_util=cpu,
        mem_util=mem,
        eff_util=eff,
        pods_running=pods_running,
        pods_deleted=pods_deleted,
        cum_cpu_util=cumulative_integral_step(t, cpu),
        cum_mem_util=cumulative_integral_step(t, mem),
        cum_eff_util=cumulative_integral_step(t, eff),
        cum_pods_running=cumulative_integral_step(t, pods_running),
    )

def compute_horizon_metrics(data: Data, T_end: float) -> Dict[str, float]:
    """
    Compute horizon metrics up to T_end from Data.
    """
    t = data.t
    out: Dict[str, float] = {
        "U_cpu_mean": mean_over_horizon(t, data.cpu_util, data.cum_cpu_util, T_end),
        "U_mem_mean": mean_over_horizon(t, data.mem_util, data.cum_mem_util, T_end),
        "U_eff_mean": mean_over_horizon(t, data.eff_util, data.cum_eff_util, T_end),
    }
    for i, p in enumerate(range(1, MAX_PRIORITIES + 1)):
        out[f"R_p{p}_mean"] = mean_over_horizon(t, data.pods_running[:, i], data.cum_pods_running[:, i], T_end)
        out[f"D_p{p}"] = value_at_horizon(t, data.pods_deleted[:, i], T_end)
    out["R_total_mean"] = float(np.nansum([out[f"R_p{p}_mean"] for p in range(1, MAX_PRIORITIES + 1)]))
    out["D_total"] = float(np.nansum([out[f"D_p{p}"] for p in range(1, MAX_PRIORITIES + 1)]))
    return out

# =============================================================================
# Latency computation
# =============================================================================

def latency_means_first_batch_ms(pod_csv: Path, eps_s: float) -> Dict[str, float]:
    """
    Mean latency (running - apply) for first-batch pods, overall and per priority p1..p4.
    Returned in MILLISECONDS (ms).
    First-batch: for each rs_prefix, take t0=min(apply), include apply times within eps_s of t0.
    """
    nan_out = {**{f"L_ms_p{p}": float("nan") for p in range(1, MAX_PRIORITIES + 1)}, "L_ms_total": float("nan")}

    pod_df = read_pod(pod_csv).dropna(subset=[POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL])
    if pod_df.empty:
        return nan_out

    apply_time_df = pod_df[pod_df[POD_EVENT_COL] == "apply-time"].copy()
    running_time_df = pod_df[pod_df[POD_EVENT_COL] == "running-time"].copy()
    if apply_time_df.empty:
        return nan_out

    apply_time_df = apply_time_df.sort_values(POD_TIME_COL, kind="mergesort").drop_duplicates(subset=[POD_UID_COL], keep="first")
    running_time_df = running_time_df.sort_values(POD_TIME_COL, kind="mergesort").drop_duplicates(subset=[POD_UID_COL], keep="first")

    apply_time_df["rs_prefix"] = apply_time_df[POD_NAME_COL].map(rs_prefix_from_pod_name)

    # Join apply times with running times
    joined = apply_time_df.merge(
        running_time_df[[POD_UID_COL, POD_TIME_COL]].rename(columns={POD_TIME_COL: "running_time_s"}),
        on=POD_UID_COL,
        how="left",
    ).rename(columns={POD_TIME_COL: "apply_time_s"})
    joined["apply_time_s"] = pd.to_numeric(joined["apply_time_s"], errors="coerce")
    joined["running_time_s"] = pd.to_numeric(joined["running_time_s"], errors="coerce")
    joined[POD_PRIO_COL] = pd.to_numeric(joined[POD_PRIO_COL], errors="coerce")
    joined = joined.dropna(subset=["rs_prefix", "apply_time_s", POD_PRIO_COL])
    if joined.empty:
        return nan_out

    # First batch selection, by taking pods with apply_time <= t0 + eps_s per rs_prefix
    joined = joined.sort_values(["rs_prefix", "apply_time_s"], kind="mergesort")
    t0 = joined.groupby("rs_prefix", sort=False)["apply_time_s"].transform("min")
    batch = joined.loc[(joined["apply_time_s"] - t0) <= float(eps_s)].copy()
    if batch.empty:
        return nan_out

    latency_s = batch["running_time_s"] - batch["apply_time_s"]
    latency_s = latency_s.where(np.isfinite(latency_s) & (latency_s >= 0.0), np.nan)
    batch["latency_ms"] = latency_s * 1000.0 # convert to milliseconds

    out: Dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["L_ms_total"] = float(np.nanmean(batch["latency_ms"].to_numpy(dtype=float))) if batch.shape[0] else float("nan")

    priorities = batch[POD_PRIO_COL].to_numpy(dtype=float)
    latencies = batch["latency_ms"].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for prio in range(1, MAX_PRIORITIES + 1):
            mask = priorities == float(prio)
            out[f"L_ms_p{prio}"] = float(np.nanmean(latencies[mask])) if np.any(mask) else float("nan")

    return out

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Sealing results...")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_FILENAME

    # Index default runs by (job_name, seed) -> (general_csv, pod_csv)
    default_idx: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
    default_seeds_by_job: Dict[str, set[str]] = {}

    # Process default runs
    for run_dir in sorted(DEFAULT_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_default_run_dir(run_dir.name)
        if parsed is None:
            continue
        seeds: set[str] = set()
        for seed, seed_dir in iter_seed_dirs(run_dir):
            seeds.add(seed)
            default_idx[(parsed, seed)] = (seed_dir / GENERAL_STATS_FILENAME, seed_dir / POD_STATS_FILENAME)
        if seeds:
            default_seeds_by_job[parsed] = seeds

    rows: List[Dict[str, object]] = []

    # Process plugin runs
    for run_dir in sorted(PLUGIN_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_plugin_run_dir(run_dir.name)
        if parsed is None:
            continue
        parsed, plugin_config = parsed
        
        # Find default seeds for this job
        default_seeds = default_seeds_by_job.get(parsed, set())
        if not default_seeds:
            print(f"[warn] no default seeds for job={parsed}")
            continue

        plugin_seeds = {seed for seed, _ in iter_seed_dirs(run_dir)}
        common_seeds = sorted(plugin_seeds & default_seeds)
        if not common_seeds:
            print(f"[warn] no common seeds for job={parsed} plugin={plugin_config}")
            continue

        # Process common seeds
        for seed in common_seeds:
            default_general_csv, default_pod_csv = default_idx[(parsed, seed)]
            plugin_seed_dir = run_dir / seed
            plugin_general_csv = plugin_seed_dir / GENERAL_STATS_FILENAME
            plugin_pod_csv = plugin_seed_dir / POD_STATS_FILENAME
            plugin_opt_json = plugin_seed_dir / OPT_STATS_FILENAME

            default_g = read_general_data(default_general_csv)
            plugin_g = read_general_data(plugin_general_csv)

            # Determine common horizon T_end
            T_end = float(min(default_g.T_end, plugin_g.T_end))
            default_metrics = compute_horizon_metrics(default_g, T_end)
            plugin_metrics = compute_horizon_metrics(plugin_g, T_end)

            # Deltas (plugin - default) & totals
            dU_cpu_pct = (float(plugin_metrics["U_cpu_mean"]) - float(default_metrics["U_cpu_mean"])) * 100.0
            dU_mem_pct = (float(plugin_metrics["U_mem_mean"]) - float(default_metrics["U_mem_mean"])) * 100.0
            dU_eff_pct = (float(plugin_metrics["U_eff_mean"]) - float(default_metrics["U_eff_mean"])) * 100.0
            dR = {p: float(plugin_metrics[f"R_p{p}_mean"]) - float(default_metrics[f"R_p{p}_mean"]) for p in range(1, MAX_PRIORITIES + 1)}
            dD = {p: float(plugin_metrics[f"D_p{p}"]) - float(default_metrics[f"D_p{p}"]) for p in range(1, MAX_PRIORITIES + 1)}
            dR_total = float(np.nansum(list(dR.values())))
            dD_total = float(np.nansum(list(dD.values())))

            default_L = latency_means_first_batch_ms(default_pod_csv, eps_s=EPS_S)
            plugin_L = latency_means_first_batch_ms(plugin_pod_csv, eps_s=EPS_S)
            dL_total = float(plugin_L["L_ms_total"]) - float(default_L["L_ms_total"])
            dL_prio = {p: float(plugin_L[f"L_ms_p{p}"]) - float(default_L[f"L_ms_p{p}"]) for p in range(1, MAX_PRIORITIES + 1)}

            # Optimization stats
            opt_stats = read_optimization_stats(plugin_opt_json)

            rows.append(
                {
                    "job_name": parsed,
                    "plugin_config": plugin_config,
                    "seed": seed,
                    "T_end_s_mean": T_end,
                    "delta_U_pct_cpu_mean": dU_cpu_pct,
                    "delta_U_pct_mem_mean": dU_mem_pct,
                    "delta_U_pct_eff_mean": dU_eff_pct,
                    "delta_R_num_p1_mean": dR[1],
                    "delta_R_num_p2_mean": dR[2],
                    "delta_R_num_p3_mean": dR[3],
                    "delta_R_num_p4_mean": dR[4],
                    "delta_R_num_total_mean": dR_total,
                    "delta_D_num_p1_mean": dD[1],
                    "delta_D_num_p2_mean": dD[2],
                    "delta_D_num_p3_mean": dD[3],
                    "delta_D_num_p4_mean": dD[4],
                    "delta_D_num_total_mean": dD_total,
                    "delta_L_ms_p1_mean": dL_prio[1],
                    "delta_L_ms_p2_mean": dL_prio[2],
                    "delta_L_ms_p3_mean": dL_prio[3],
                    "delta_L_ms_p4_mean": dL_prio[4],
                    "delta_L_ms_total_mean": dL_total,
                    "solver_attempts_mean": float(opt_stats["solver_attempts"]),
                    "solver_optimal_mean": float(opt_stats["solver_optimal"]),
                    "solver_feasible_mean": float(opt_stats["solver_feasible"]),
                    "solver_failed_mean": float(opt_stats["solver_failed"]),
                    "plan_not_applicable_mean": float(opt_stats["plan_not_applicable"]),
                    "plan_activated_mean": float(opt_stats["plan_activated"]),
                }
            )

    df = pd.DataFrame(rows)

    # Ensure schema even when empty
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[OUT_COLS]

    round_numeric_df(df, decimals=FLOAT_DECIMALS, exclude=["job_name", "plugin_config", "seed"])
    df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()