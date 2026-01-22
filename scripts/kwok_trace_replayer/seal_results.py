#!/usr/bin/env python3
# scripts/kwok_trace_replayer/seal_results.py
"""
python -m scripts.kwok_trace_replayer.seal_results --root analysis/kwok_trace_replayer --out-dir analysis/kwok_trace_replayer/sealed

We compute per-seed metrics over the common horizon H = min(T_end_default, T_end_plugin), then mean over seeds.
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
RUNNING_PREFIX = "running_p"              # running_p1..pK (instantaneous count)
DELETIONS_CUM_PREFIX = "deletions_cum_p"  # deletions_cum_p1..pK (cumulative counter)

POD_EVENT_COL = "event"
POD_NAME_COL = "pod_name"
POD_UID_COL = "pod_uid"
POD_PRIO_COL = "priority"
POD_TIME_COL = "time_s"

MAX_K_OUT = 4

PAIRED_COLS = [
    "job_name",
    "plugin_config",
    "n_seed",
    "T_end_s_mean",
    "delta_cpu_run_util_mean",
    "delta_mem_run_util_mean",
    "delta_util_eff_run_mean",
    "delta_R_p1_mean",
    "delta_R_p2_mean",
    "delta_R_p3_mean",
    "delta_R_p4_mean",
    "delta_R_total_mean",
    "delta_D_p1_mean",
    "delta_D_p2_mean",
    "delta_D_p3_mean",
    "delta_D_p4_mean",
    "delta_D_total_mean",
    "delta_latency_s_p1_mean",
    "delta_latency_s_p2_mean",
    "delta_latency_s_p3_mean",
    "delta_latency_s_p4_mean",
    "delta_latency_s_total_mean",
    "solver_attempts_mean",
    "solver_optimal_mean",
    "solver_feasible_mean",
    "solver_failed_mean",
    "plan_not_applicable_mean",
    "plan_activated_mean",
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
# Helpers
# -----------------------------

def round_numeric_df(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = exclude or []
    num_cols = [c for c in df.columns if c not in set(exclude) and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].round(FLOAT_DECIMALS)
    return df

def parse_job_dir_name(name: str) -> Optional[str]:
    """
    Accepts:
      - nodes=16_prio=4_arrival=8s
      - nodes16_prio4_arrival8
      - nodes=16_prio=4_arrival=8
    Returns canonical job_name: nodes=<N>_prio=<K>_arrival=<A>s
    """
    s = str(name)
    m_nodes = re.search(r"nodes=?(\d+)", s)
    m_prio = re.search(r"prio=?(\d+)", s)
    m_arr = re.search(r"arrival=?([0-9.]+)s?", s)
    if not (m_nodes and m_prio and m_arr):
        return None
    n = int(m_nodes.group(1))
    k = int(m_prio.group(1))
    a = float(m_arr.group(1))
    a_str = str(int(a)) if abs(a - round(a)) < 1e-9 else f"{a:g}"
    return f"nodes={n}_prio={k}_arrival={a_str}s"

def parse_plugin_run_dir(name: str) -> Optional[Tuple[str, str]]:
    """
    Example:
      mode=periodic8s_blocking=0_defpreempt=1_nodes=16_prio=1_arrival=8s
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
        arr_raw = kv["arrival"]
        arr = float(arr_raw[:-1] if arr_raw.endswith("s") else arr_raw)
    except Exception:
        return None
    a_str = str(int(arr)) if abs(arr - round(arr)) < 1e-9 else f"{arr:g}"
    job_name = f"nodes={n}_prio={kmax}_arrival={a_str}s"
    mode = kv.get("mode", "unknown")
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

def read_opt_totals(opt_json: Path) -> Dict[str, float]:
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

# -----------------------------
# Horizon metrics
# -----------------------------

@dataclass(frozen=True)
class GeneralData:
    t: np.ndarray                     # normalized time (starts at 0)
    T_end: float
    cpu_util: np.ndarray
    mem_util: np.ndarray
    eff_util: np.ndarray
    pods_running: np.ndarray                   # shape (n, MAX_K_OUT) for p=1..MAX_K_OUT
    pods_deleted: np.ndarray                  # shape (n, MAX_K_OUT), NaN where missing
    pref_cpu_util: np.ndarray              # prefix integral arrays (length n)
    pref_mem_util: np.ndarray
    pref_eff_util: np.ndarray
    pref_pods_running: np.ndarray              # shape (n, MAX_K_OUT)

def _prefix_integral(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """prefix[i] = ∫_0^{t[i]} y(s) ds, with y stepwise using y[j] on [t[j], t[j+1])."""
    n = t.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)
    dt = np.diff(t)
    pref = np.zeros(n, dtype=float)
    pref[1:] = np.cumsum(y[:-1] * dt)
    return pref

def _prefix_integral_mat(t: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Same as _prefix_integral, but for matrix columns."""
    n = t.size
    if n == 0:
        return np.zeros((0, Y.shape[1]), dtype=float)
    if n == 1:
        return np.zeros((1, Y.shape[1]), dtype=float)
    dt = np.diff(t)[:, None]
    pref = np.zeros((n, Y.shape[1]), dtype=float)
    pref[1:, :] = np.cumsum(Y[:-1, :] * dt, axis=0)
    return pref

def read_general_data(general_csv: Path) -> GeneralData:
    wanted = {TIME_COL, CPU_RUN_COL, MEM_RUN_COL}
    for p in range(1, MAX_K_OUT + 1):
        wanted.add(f"{RUNNING_PREFIX}{p}")
        wanted.add(f"{DELETIONS_CUM_PREFIX}{p}")

    # Single read; usecols callable ignores missing columns without failing.
    df = pd.read_csv(
        general_csv,
        usecols=lambda c: c in wanted,
        low_memory=False,
    )

    if TIME_COL not in df.columns:
        raise SystemExit(f"{general_csv} missing {TIME_COL}")

    t_raw = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.loc[t_raw.notna()].copy()
    if df.empty:
        return GeneralData(
            t=np.array([], dtype=float),
            T_end=float("nan"),
            cpu_util=np.array([], dtype=float),
            mem_util=np.array([], dtype=float),
            eff_util=np.array([], dtype=float),
            pods_running=np.zeros((0, MAX_K_OUT), dtype=float),
            pods_deleted=np.zeros((0, MAX_K_OUT), dtype=float),
            pref_cpu_util=np.array([], dtype=float),
            pref_mem_util=np.array([], dtype=float),
            pref_eff_util=np.array([], dtype=float),
            pref_pods_running=np.zeros((0, MAX_K_OUT), dtype=float),
        )

    df[TIME_COL] = t_raw.loc[df.index].astype(float)
    df = df.sort_values(TIME_COL, kind="mergesort")

    t0 = float(df[TIME_COL].iloc[0])
    t = (df[TIME_COL].to_numpy(dtype=float) - t0).astype(float)
    # Drop any negative / NaN issues after normalization
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
            pref_cpu_util=np.array([], dtype=float),
            pref_mem_util=np.array([], dtype=float),
            pref_eff_util=np.array([], dtype=float),
            pref_pods_running=np.zeros((0, MAX_K_OUT), dtype=float),
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

    pref_cpu_util_run = _prefix_integral(t, util_cpu_run)
    pref_mem_util_run = _prefix_integral(t, mem_util_run)
    pref_eff_util_run = _prefix_integral(t, eff_util_run)
    pref_pods_running = _prefix_integral_mat(t, pods_running)

    return GeneralData(
        t=t,
        T_end=T_end,
        cpu_util=util_cpu_run,
        mem_util=mem_util_run,
        eff_util=eff_util_run,
        pods_running=pods_running,
        pods_deleted=pods_deleted,
        pref_cpu_util=pref_cpu_util_run,
        pref_mem_util=pref_mem_util_run,
        pref_eff_util=pref_eff_util_run,
        pref_pods_running=pref_pods_running,
    )

def _mean_over_horizon(t: np.ndarray, y: np.ndarray, pref: np.ndarray, H: float) -> float:
    """Time-weighted mean over [0, H] for stepwise y using prefix integral pref."""
    if not (math.isfinite(H) and H > 0.0) or t.size == 0:
        return float("nan")
    if H <= 0.0:
        return float("nan")
    # If H beyond end, clamp to end (matches previous behavior via min horizon selection anyway)
    Hc = min(H, float(t[-1]))
    idx = int(np.searchsorted(t, Hc, side="right") - 1)
    if idx < 0:
        idx = 0
    base = float(pref[idx])
    tail = float(y[idx]) * float(Hc - t[idx])
    return float((base + tail) / Hc) if Hc > 0 else float("nan")

def _value_at_horizon(t: np.ndarray, y: np.ndarray, H: float) -> float:
    """Step value at time H (last observation carried forward)."""
    if t.size == 0 or not math.isfinite(H):
        return float("nan")
    Hc = min(max(H, 0.0), float(t[-1]))
    idx = int(np.searchsorted(t, Hc, side="right") - 1)
    if idx < 0:
        idx = 0
    v = float(y[idx])
    return v if math.isfinite(v) else float("nan")

def compute_horizon_metrics(g: GeneralData, H: float) -> Dict[str, float]:
    t = g.t
    out: Dict[str, float] = {
        "cpu_run_util_mean": _mean_over_horizon(t, g.cpu_util, g.pref_cpu_util, H),
        "mem_run_util_mean": _mean_over_horizon(t, g.mem_util, g.pref_mem_util, H),
        "util_eff_run_mean": _mean_over_horizon(t, g.eff_util, g.pref_eff_util, H),
    }

    for i, p in enumerate(range(1, MAX_K_OUT + 1)):
        out[f"R_p{p}_mean"] = _mean_over_horizon(t, g.pods_running[:, i], g.pref_pods_running[:, i], H)
        out[f"D_p{p}"] = _value_at_horizon(t, g.pods_deleted[:, i], H)

    out["R_total_mean"] = float(np.nansum([out[f"R_p{p}_mean"] for p in range(1, MAX_K_OUT + 1)]))
    out["D_total"] = float(np.nansum([out[f"D_p{p}"] for p in range(1, MAX_K_OUT + 1)]))
    return out


# -----------------------------
# Latency means (vectorized first-batch)
# -----------------------------

def read_pod_minimal(pod_csv: Path) -> pd.DataFrame:
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


def latency_means_first_batch(pod_csv: Path, eps_s: float) -> Dict[str, float]:
    """
    Mean latency (running - apply) for first-batch pods, overall and per priority p1..p4.
    First-batch: for each rs_prefix, take t0=min(apply), include apply times within eps_s of t0.
    """
    try:
        pod_df = read_pod_minimal(pod_csv)
    except Exception:
        return {**{f"latency_s_p{p}": float("nan") for p in range(1, MAX_K_OUT + 1)}, "latency_s_total": float("nan")}

    pod_df = pod_df.dropna(subset=[POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL]).copy()
    if pod_df.empty:
        return {**{f"latency_s_p{p}": float("nan") for p in range(1, MAX_K_OUT + 1)}, "latency_s_total": float("nan")}

    apply_df = pod_df[pod_df[POD_EVENT_COL] == "apply-time"].copy()
    run_df = pod_df[pod_df[POD_EVENT_COL] == "running-time"].copy()
    if apply_df.empty:
        return {**{f"latency_s_p{p}": float("nan") for p in range(1, MAX_K_OUT + 1)}, "latency_s_total": float("nan")}

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

    joined = joined.dropna(subset=["rs_prefix", "apply_time_s", POD_PRIO_COL]).copy()
    if joined.empty:
        return {**{f"latency_s_p{p}": float("nan") for p in range(1, MAX_K_OUT + 1)}, "latency_s_total": float("nan")}

    joined = joined.sort_values(["rs_prefix", "apply_time_s"], kind="mergesort")
    t0 = joined.groupby("rs_prefix", sort=False)["apply_time_s"].transform("min")
    in_batch = (joined["apply_time_s"] - t0) <= float(eps_s)
    batch = joined.loc[in_batch].copy()
    if batch.empty:
        return {**{f"latency_s_p{p}": float("nan") for p in range(1, MAX_K_OUT + 1)}, "latency_s_total": float("nan")}

    lat = batch["running_time_s"] - batch["apply_time_s"]
    # Only keep finite, non-negative latencies (same semantics as before)
    lat = lat.where(np.isfinite(lat) & (lat >= 0.0), np.nan)
    batch["latency_s"] = lat

    out: Dict[str, float] = {}
    out["latency_s_total"] = float(np.nanmean(batch["latency_s"].to_numpy(dtype=float))) if batch.shape[0] else float("nan")

    pr = batch[POD_PRIO_COL].to_numpy(dtype=float)
    latv = batch["latency_s"].to_numpy(dtype=float)

    for p in range(1, MAX_K_OUT + 1):
        mask = pr == float(p)
        out[f"latency_s_p{p}"] = float(np.nanmean(latv[mask])) if np.any(mask) else float("nan")

    return out


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

    print("Sealing ...")

    # Index defaults and seeds per job
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
            def_gen, def_pod = default_idx[(job_name, seed)]
            plu_seed_dir = run_dir / seed
            plu_gen = plu_seed_dir / GENERAL_STATS_FILENAME
            plu_pod = plu_seed_dir / POD_STATS_FILENAME
            plu_opt = plu_seed_dir / OPT_STATS_FILENAME

            # ---- General data (single read per file) ----
            try:
                def_g = read_general_data(def_gen)
            except Exception:
                continue

            try:
                plu_g = read_general_data(plu_gen)
            except Exception:
                continue

            if not (math.isfinite(def_g.T_end) and math.isfinite(plu_g.T_end)):
                continue
            if def_g.T_end <= 0.0 or plu_g.T_end <= 0.0:
                continue

            H = float(min(def_g.T_end, plu_g.T_end))
            if H <= 0.0:
                continue

            def_m = compute_horizon_metrics(def_g, H)
            plu_m = compute_horizon_metrics(plu_g, H)

            d_cpu = float(plu_m["cpu_run_util_mean"]) - float(def_m["cpu_run_util_mean"])
            d_mem = float(plu_m["mem_run_util_mean"]) - float(def_m["mem_run_util_mean"])
            d_eff = float(plu_m["util_eff_run_mean"]) - float(def_m["util_eff_run_mean"])

            dR = {p: float(plu_m[f"R_p{p}_mean"]) - float(def_m[f"R_p{p}_mean"]) for p in range(1, MAX_K_OUT + 1)}
            dD = {p: float(plu_m[f"D_p{p}"]) - float(def_m[f"D_p{p}"]) for p in range(1, MAX_K_OUT + 1)}
            dR_total = float(np.nansum(list(dR.values())))
            dD_total = float(np.nansum(list(dD.values())))

            # ---- Latency means ----
            def_lat = latency_means_first_batch(def_pod, eps_s=eps_s)

            plu_lat = latency_means_first_batch(plu_pod, eps_s=eps_s)

            delta_lat_total = float(plu_lat["latency_s_total"]) - float(def_lat["latency_s_total"])
            delta_lat_p = {
                p: float(plu_lat[f"latency_s_p{p}"]) - float(def_lat[f"latency_s_p{p}"])
                for p in range(1, MAX_K_OUT + 1)
            }

            # ---- Optimization totals (plugin only) ----
            opt = read_opt_totals(plu_opt)

            seed_rows.append(
                {
                    "job_name": job_name,
                    "plugin_config": plugin_config,
                    "seed": seed,
                    "T_end_s": H,
                    "delta_cpu_run_util": d_cpu,
                    "delta_mem_run_util": d_mem,
                    "delta_util_eff_run": d_eff,
                    "delta_R_p1": dR[1],
                    "delta_R_p2": dR[2],
                    "delta_R_p3": dR[3],
                    "delta_R_p4": dR[4],
                    "delta_R_total": dR_total,
                    "delta_D_p1": dD[1],
                    "delta_D_p2": dD[2],
                    "delta_D_p3": dD[3],
                    "delta_D_p4": dD[4],
                    "delta_D_total": dD_total,
                    "delta_latency_s_p1": delta_lat_p[1],
                    "delta_latency_s_p2": delta_lat_p[2],
                    "delta_latency_s_p3": delta_lat_p[3],
                    "delta_latency_s_p4": delta_lat_p[4],
                    "delta_latency_s_total": delta_lat_total,
                    "solver_attempts": float(opt["solver_attempts"]),
                    "solver_optimal": float(opt["solver_optimal"]),
                    "solver_feasible": float(opt["solver_feasible"]),
                    "solver_failed": float(opt["solver_failed"]),
                    "plan_not_applicable": float(opt["plan_not_applicable"]),
                    "plan_activated": float(opt["plan_activated"]),
                }
            )

    # results_paired.csv
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
        "delta_cpu_run_util": "delta_cpu_run_util_mean",
        "delta_mem_run_util": "delta_mem_run_util_mean",
        "delta_util_eff_run": "delta_util_eff_run_mean",
        "delta_R_p1": "delta_R_p1_mean",
        "delta_R_p2": "delta_R_p2_mean",
        "delta_R_p3": "delta_R_p3_mean",
        "delta_R_p4": "delta_R_p4_mean",
        "delta_R_total": "delta_R_total_mean",
        "delta_D_p1": "delta_D_p1_mean",
        "delta_D_p2": "delta_D_p2_mean",
        "delta_D_p3": "delta_D_p3_mean",
        "delta_D_p4": "delta_D_p4_mean",
        "delta_D_total": "delta_D_total_mean",
        "delta_latency_s_p1": "delta_latency_s_p1_mean",
        "delta_latency_s_p2": "delta_latency_s_p2_mean",
        "delta_latency_s_p3": "delta_latency_s_p3_mean",
        "delta_latency_s_p4": "delta_latency_s_p4_mean",
        "delta_latency_s_total": "delta_latency_s_total_mean",
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
