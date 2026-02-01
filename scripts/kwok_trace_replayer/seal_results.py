#!/usr/bin/env python3
# scripts/kwok_trace_replayer/seal_results.py
"""
python -m scripts.kwok_trace_replayer.seal_results
"""

import json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.kwok_trace_replayer.trace_helpers import rs_prefix_from_pod_name

# =============================================================================
# CONFIG
# =============================================================================

ROOT_DIR = Path("analysis/kwok_trace_replayer")
OUT_DIR = ROOT_DIR  # results_seeds.csv will be written here

# First-batch epsilon window (seconds)
EPS_S = 1.0

# Rounding for numeric output
FLOAT_DECIMALS = 4

# =============================================================================
# Filenames / columns
# =============================================================================

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

LATENCY_S_TO_MS = 1000.0
UTIL_TO_PERCENT = 100.0

OUT_SEEDS_FILENAME = "results_seeds.csv"

SEED_OUT_COLS = [
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
class GeneralData:
    t: np.ndarray  # normalized time (starts at 0)
    T_end: float
    cpu_util: np.ndarray
    mem_util: np.ndarray
    eff_util: np.ndarray
    pods_running: np.ndarray  # (n, MAX_K_OUT)
    pods_deleted: np.ndarray  # (n, MAX_K_OUT)
    cum_cpu_util: np.ndarray
    cum_mem_util: np.ndarray
    cum_eff_util: np.ndarray
    cum_pods_running: np.ndarray  # (n, MAX_K_OUT)


# =============================================================================
# Parsing helpers (directory names)
# =============================================================================

_JOB_RE = re.compile(r"nodes=?(\d+).*prio=?(\d+).*arrival=?([0-9.]+)s?", re.IGNORECASE)


def _arrival_str(arrival: float) -> str:
    return str(int(arrival)) if abs(arrival - round(arrival)) < 1e-9 else f"{arrival:g}"


def parse_job_dir_name(name: str) -> Optional[str]:
    """
    Example: nodes=16_prio=2_arrival=10s  -> nodes=16_prio=2_arrival=10s
    """
    m = _JOB_RE.search(str(name))
    if not m:
        return None
    n = int(m.group(1))
    pr = int(m.group(2))
    a = float(m.group(3))
    return f"nodes={n}_prio={pr}_arrival={_arrival_str(a)}s"


def canonicalize_mode(mode: str) -> str:
    """
    Only canonicalize scheduling-failure to: schedulingfailure
    """
    s = str(mode).strip().lower()
    s_compact = s.replace("-", "").replace("_", "")
    if s_compact in {"schedulingfailure", "schedfailure", "scheduingfailure"}:
        return "schedulingfailure"
    return s


def parse_plugin_run_dir(name: str) -> Optional[Tuple[str, str]]:
    """
    Example:
      mode=schedulingfailure_blocking=0_defpreempt=0_nodes=32_prio=1_arrival=4s

    Returns: (job_name, plugin_config)
      plugin_config: mode=<mode>_blocking=<0/1>_defpreempt=<0/1>
    """
    kv: Dict[str, str] = {}
    for tok in str(name).split("_"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip().lower()] = v.strip()

    try:
        n = int(kv["nodes"])
        pr = int(kv["prio"])
        arrival_raw = kv["arrival"]
        a = float(arrival_raw[:-1] if arrival_raw.endswith("s") else arrival_raw)
    except Exception:
        return None

    job_name = f"nodes={n}_prio={pr}_arrival={_arrival_str(a)}s"

    mode = canonicalize_mode(kv.get("mode", "unknown"))
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
    Supports y shape (n,) and (n,m).
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


def mean_over_horizon(t: np.ndarray, y: np.ndarray, cum: np.ndarray, H: float) -> float:
    if not (math.isfinite(H) and H > 0.0) or t.size == 0:
        return float("nan")
    Hc = min(H, float(t[-1]))
    idx = int(np.searchsorted(t, Hc, side="right") - 1)
    idx = max(idx, 0)
    base = float(cum[idx])
    tail = float(y[idx]) * float(Hc - t[idx])
    return float((base + tail) / Hc) if Hc > 0 else float("nan")


def value_at_horizon(t: np.ndarray, y: np.ndarray, H: float) -> float:
    if t.size == 0 or not math.isfinite(H):
        return float("nan")
    Hc = min(max(H, 0.0), float(t[-1]))
    idx = int(np.searchsorted(t, Hc, side="right") - 1)
    idx = max(idx, 0)
    v = float(y[idx])
    return v if math.isfinite(v) else float("nan")


# =============================================================================
# General stats reading + horizon metrics
# =============================================================================


def read_general_data(general_csv: Path) -> GeneralData:
    cols_needed = {TIME_COL, CPU_RUN_COL, MEM_RUN_COL}
    for p in range(1, MAX_K_OUT + 1):
        cols_needed.add(f"{RUNNING_PREFIX}{p}")
        cols_needed.add(f"{DELETIONS_CUM_PREFIX}{p}")

    df = pd.read_csv(general_csv, usecols=lambda c: c in cols_needed)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.loc[df[TIME_COL].notna()].copy()
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

    df = df.sort_values(TIME_COL, kind="mergesort")
    t0 = float(df[TIME_COL].iloc[0])
    t = (df[TIME_COL].to_numpy(dtype=float) - t0).astype(float)
    valid = np.isfinite(t) & (t >= 0.0)
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

    cpu = pd.to_numeric(df.get(CPU_RUN_COL, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mem = pd.to_numeric(df.get(MEM_RUN_COL, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    eff = np.maximum(cpu, mem)

    pods_running = np.zeros((t.size, MAX_K_OUT), dtype=float)
    pods_deleted = np.full((t.size, MAX_K_OUT), np.nan, dtype=float)

    for i, p in enumerate(range(1, MAX_K_OUT + 1)):
        rc = f"{RUNNING_PREFIX}{p}"
        dc = f"{DELETIONS_CUM_PREFIX}{p}"
        if rc in df.columns:
            pods_running[:, i] = pd.to_numeric(df[rc], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if dc in df.columns:
            pods_deleted[:, i] = pd.to_numeric(df[dc], errors="coerce").to_numpy(dtype=float)

    return GeneralData(
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


def compute_horizon_metrics(g: GeneralData, H: float) -> Dict[str, float]:
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


# =============================================================================
# Latency (first-batch) in ms
# =============================================================================


def latency_means_first_batch_ms(pod_csv: Path, eps_s: float) -> Dict[str, float]:
    """
    Mean latency (running - apply) for first-batch pods, overall and per priority p1..p4.
    Returned in MILLISECONDS (ms).

    First-batch: for each rs_prefix, take t0=min(apply), include apply times within eps_s of t0.
    """
    nan_out = {**{f"L_ms_p{p}": float("nan") for p in range(1, MAX_K_OUT + 1)}, "L_ms_total": float("nan")}

    pod_df = read_pod(pod_csv).dropna(subset=[POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL])
    if pod_df.empty:
        return nan_out

    apply_df = pod_df[pod_df[POD_EVENT_COL] == "apply-time"].copy()
    run_df = pod_df[pod_df[POD_EVENT_COL] == "running-time"].copy()
    if apply_df.empty:
        return nan_out

    apply_df = apply_df.sort_values(POD_TIME_COL, kind="mergesort").drop_duplicates(subset=[POD_UID_COL], keep="first")
    run_df = run_df.sort_values(POD_TIME_COL, kind="mergesort").drop_duplicates(subset=[POD_UID_COL], keep="first")

    apply_df["rs_prefix"] = apply_df[POD_NAME_COL].map(rs_prefix_from_pod_name)

    joined = apply_df.merge(
        run_df[[POD_UID_COL, POD_TIME_COL]].rename(columns={POD_TIME_COL: "running_time_s"}),
        on=POD_UID_COL,
        how="left",
    ).rename(columns={POD_TIME_COL: "apply_time_s"})

    joined["apply_time_s"] = pd.to_numeric(joined["apply_time_s"], errors="coerce")
    joined["running_time_s"] = pd.to_numeric(joined["running_time_s"], errors="coerce")
    joined[POD_PRIO_COL] = pd.to_numeric(joined[POD_PRIO_COL], errors="coerce")
    joined = joined.dropna(subset=["rs_prefix", "apply_time_s", POD_PRIO_COL])
    if joined.empty:
        return nan_out

    joined = joined.sort_values(["rs_prefix", "apply_time_s"], kind="mergesort")
    t0 = joined.groupby("rs_prefix", sort=False)["apply_time_s"].transform("min")
    batch = joined.loc[(joined["apply_time_s"] - t0) <= float(eps_s)].copy()
    if batch.empty:
        return nan_out

    latency_s = batch["running_time_s"] - batch["apply_time_s"]
    latency_s = latency_s.where(np.isfinite(latency_s) & (latency_s >= 0.0), np.nan)
    batch["latency_ms"] = latency_s * LATENCY_S_TO_MS

    out: Dict[str, float] = {}
    out["L_ms_total"] = float(np.nanmean(batch["latency_ms"].to_numpy(dtype=float))) if batch.shape[0] else float("nan")

    prios = batch[POD_PRIO_COL].to_numpy(dtype=float)
    lat = batch["latency_ms"].to_numpy(dtype=float)
    for p in range(1, MAX_K_OUT + 1):
        mask = prios == float(p)
        out[f"L_ms_p{p}"] = float(np.nanmean(lat[mask])) if np.any(mask) else float("nan")

    return out


# =============================================================================
# Output helpers
# =============================================================================


def round_numeric_df(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = exclude or []
    num_cols = [c for c in df.columns if c not in set(exclude) and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].round(FLOAT_DECIMALS)
    return df


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    default_root = ROOT_DIR / "default"
    plugin_root = ROOT_DIR / "plugin"

    if not default_root.exists():
        raise SystemExit(f"Not found: {default_root}")
    if not plugin_root.exists():
        raise SystemExit(f"Not found: {plugin_root}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_SEEDS_FILENAME

    # Index default runs by (job_name, seed) -> (general_csv, pod_csv)
    default_idx: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
    default_seeds_by_job: Dict[str, set[str]] = {}

    for job_dir in sorted(default_root.iterdir()):
        if not job_dir.is_dir():
            continue
        job_name = parse_job_dir_name(job_dir.name)
        if job_name is None:
            continue
        seeds: set[str] = set()
        for seed, seed_dir in iter_seed_dirs(job_dir):
            seeds.add(seed)
            default_idx[(job_name, seed)] = (seed_dir / GENERAL_STATS_FILENAME, seed_dir / POD_STATS_FILENAME)
        if seeds:
            default_seeds_by_job[job_name] = seeds

    rows: List[Dict[str, object]] = []

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
            default_general_csv, default_pod_csv = default_idx[(job_name, seed)]
            plugin_seed_dir = run_dir / seed
            plugin_general_csv = plugin_seed_dir / GENERAL_STATS_FILENAME
            plugin_pod_csv = plugin_seed_dir / POD_STATS_FILENAME
            plugin_opt_json = plugin_seed_dir / OPT_STATS_FILENAME

            default_g = read_general_data(default_general_csv)
            plugin_g = read_general_data(plugin_general_csv)

            if not (math.isfinite(default_g.T_end) and math.isfinite(plugin_g.T_end)):
                continue
            if default_g.T_end <= 0.0 or plugin_g.T_end <= 0.0:
                continue

            H = float(min(default_g.T_end, plugin_g.T_end))
            if H <= 0.0:
                continue

            default_m = compute_horizon_metrics(default_g, H)
            plugin_m = compute_horizon_metrics(plugin_g, H)

            dU_cpu_pct = (float(plugin_m["U_cpu_mean"]) - float(default_m["U_cpu_mean"])) * UTIL_TO_PERCENT
            dU_mem_pct = (float(plugin_m["U_mem_mean"]) - float(default_m["U_mem_mean"])) * UTIL_TO_PERCENT
            dU_eff_pct = (float(plugin_m["U_eff_mean"]) - float(default_m["U_eff_mean"])) * UTIL_TO_PERCENT

            dR = {p: float(plugin_m[f"R_p{p}_mean"]) - float(default_m[f"R_p{p}_mean"]) for p in range(1, MAX_K_OUT + 1)}
            dD = {p: float(plugin_m[f"D_p{p}"]) - float(default_m[f"D_p{p}"]) for p in range(1, MAX_K_OUT + 1)}
            dR_total = float(np.nansum(list(dR.values())))
            dD_total = float(np.nansum(list(dD.values())))

            default_L = latency_means_first_batch_ms(default_pod_csv, eps_s=EPS_S)
            plugin_L = latency_means_first_batch_ms(plugin_pod_csv, eps_s=EPS_S)
            dL_total = float(plugin_L["L_ms_total"]) - float(default_L["L_ms_total"])
            dL_prio = {p: float(plugin_L[f"L_ms_p{p}"]) - float(default_L[f"L_ms_p{p}"]) for p in range(1, MAX_K_OUT + 1)}

            opt = read_optimization_stats(plugin_opt_json)

            rows.append(
                {
                    "job_name": job_name,
                    "plugin_config": plugin_config,
                    "seed": seed,
                    "T_end_s_mean": H,
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
                    "solver_attempts_mean": float(opt["solver_attempts"]),
                    "solver_optimal_mean": float(opt["solver_optimal"]),
                    "solver_feasible_mean": float(opt["solver_feasible"]),
                    "solver_failed_mean": float(opt["solver_failed"]),
                    "plan_not_applicable_mean": float(opt["plan_not_applicable"]),
                    "plan_activated_mean": float(opt["plan_activated"]),
                }
            )

    df = pd.DataFrame(rows)

    # Ensure schema even when empty
    for c in SEED_OUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[SEED_OUT_COLS]

    round_numeric_df(df, exclude=["job_name", "plugin_config", "seed"])
    df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
