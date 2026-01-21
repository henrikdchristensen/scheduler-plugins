#!/usr/bin/env python3
"""
scripts/kwok_trace_replayer/seal_results.py

python -m scripts.kwok_trace_replayer.seal_results --root analysis/kwok_trace_replayer --out-dir analysis/kwok_trace_replayer/sealed

Outputs (under --out-dir):
  1) results_paired.csv  (mean across seeds; columns/order match PAIRED_COLS)
  2) series/ mean time-series CSVs for plotting:
       - series/default/<job_name>.csv
       - series/plugin/<plugin_config>__<job_name>.csv    (flat, no nested subdirs)
     (columns/order match SERIES_COLS)
  3) per_pod_stats/ per-seed matched-pod latency comparisons:
       - per_pod_stats/<plugin_config>__<job_name>.csv
     (columns/order match PER_POD_COLS)

Notes:
  - Series: for each time_s we average values across seeds (time_s is normalized per seed).
    R_cum_* are cumulative pod-seconds inside each seed, then averaged across seeds.
    D_cum_* are cumulative deletion counters inside each seed, then averaged across seeds.
    Util columns are per-sample means across seeds.
    Latency columns are per-seed scalars (first-batch mean latency) replicated over time, then averaged.
  - results_paired: we compute per-seed metrics over the common horizon (min T_end), then mean over seeds.
  - per_pod_stats: per-seed rows only (no averaging), only for pods where both schedulers have a latency value.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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

# columns in general_stats.csv (baseline naming)
CPU_RUN_COL = "cpu_run_util"
MEM_RUN_COL = "mem_run_util"

RUNNING_PREFIX = "running_p"              # running_p1..pK (instantaneous count)
DELETIONS_CUM_PREFIX = "deletions_cum_p"  # deletions_cum_p1..pK (cumulative counter)

# columns in pod_stats.csv
POD_EVENT_COL = "event"
POD_NAME_COL = "pod_name"
POD_UID_COL = "pod_uid"
POD_PRIO_COL = "priority"
POD_TIME_COL = "time_s"

MAX_K_OUT = 4  # always output p1..p4 columns (even if job has fewer)

# results_paired.csv exact columns/order
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

# series CSV exact columns/order (solver/plan columns REMOVED)
SERIES_COLS = [
    "scheduler",
    "job_name",
    "n_seed",
    "time_s",
    "cpu_run_util_mean",
    "mem_run_util_mean",
    "eff_run_util_mean",
    "R_cum_p1_mean",
    "R_cum_p2_mean",
    "R_cum_p3_mean",
    "R_cum_p4_mean",
    "R_cum_total_mean",
    "D_cum_p1_mean",
    "D_cum_p2_mean",
    "D_cum_p3_mean",
    "D_cum_p4_mean",
    "D_cum_total_mean",
    "latency_s_p1_mean",
    "latency_s_p2_mean",
    "latency_s_p3_mean",
    "latency_s_p4_mean",
    "latency_s_total_mean",
]

# per_pod_stats exact columns/order
PER_POD_COLS = [
    "job_name",
    "plugin_config",
    "seed",
    "priority",
    "rs_prefix",
    "replica_index",
    "latency_default_s",
    "latency_plugin_s",
    "delta_latency_s",
]

# optimization_stats.json (totals) -> results_paired (means across seeds)
OPT_TOTAL_KEYS = {
    "solver_attempts_total": "solver_attempts",
    "best_solver_optimal_total": "solver_optimal",
    "best_solver_feasible_total": "solver_feasible",
    "best_solver_failed_total": "solver_failed",
    "plan_not_applicable_total": "plan_not_applicable",
    "plan_activated_total": "plan_activated",
}

# latency key for matching
LatencyKey = Tuple[int, str, int]  # (priority, rs_prefix, replica_index_in_first_batch)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seal KWOK trace replayer outputs into paired + series CSVs.")
    p.add_argument("--root", required=True, help="Root dir containing 'default/' and 'plugin/' subfolders.")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <root>/sealed).")
    p.add_argument("--eps-s", type=float, default=1.0, help="First-batch epsilon window in seconds (default: 1.0).")
    return p.parse_args()


# -----------------------------
# Small helpers
# -----------------------------

def round_numeric_df(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = exclude or []
    num_cols = [c for c in df.columns if c not in set(exclude) and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].round(FLOAT_DECIMALS)
    return df


def safe_stem(label: str) -> str:
    """
    Windows-safe filename stem:
      - only [A-Za-z0-9._-]
      - no trailing dots/spaces
      - NO hash suffix
    """
    raw = str(label)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip(" .")
    return cleaned or "x"


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
    Expected tokens separated by underscores, e.g.:
      mode=periodic8s_blocking=0_defpreempt=1_nodes=16_prio=1_arrival=8s

    Returns: (job_name, plugin_config_key)
      plugin_config_key format:
        mode=<mode>_blocking=<0/1>_defpreempt=<0/1>
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


def value_at_step(t: np.ndarray, y: np.ndarray, x: float, default: float = 0.0) -> float:
    """Last observation carried forward at time x (stepwise)."""
    if t.size == 0 or y.size == 0:
        return float(default)
    idx = np.searchsorted(t, x, side="right") - 1
    if idx < 0:
        return float(default)
    idx = min(idx, y.size - 1)
    v = y[idx]
    return float(v) if math.isfinite(float(v)) else float(default)


def time_weighted_mean_step(t: np.ndarray, y: np.ndarray, H: float) -> float:
    """Time-weighted mean over [0, H] with stepwise y(t)."""
    if H <= 0 or t.size == 0 or y.size == 0:
        return float("nan")

    t_clip = t[t <= H]
    if t_clip.size == 0:
        t_clip = np.array([H], dtype=float)
    elif t_clip[-1] < H:
        t_clip = np.concatenate([t_clip, np.array([H], dtype=float)])

    dt = np.diff(t_clip, prepend=t_clip[0])
    dt = np.clip(dt, 0.0, None)

    y_clip = np.array([value_at_step(t, y, float(tt), default=0.0) for tt in t_clip], dtype=float)
    return float(np.sum(y_clip * dt) / H)


# -----------------------------
# Latency helpers
# -----------------------------

def read_pod_df(pod_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pod_csv)
    need = [POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{pod_csv} missing columns: {', '.join(missing)}")

    df[POD_TIME_COL] = pd.to_numeric(df[POD_TIME_COL], errors="coerce")
    df[POD_PRIO_COL] = pd.to_numeric(df[POD_PRIO_COL], errors="coerce").astype("Int64")
    df[POD_EVENT_COL] = df[POD_EVENT_COL].astype(str)
    df[POD_NAME_COL] = df[POD_NAME_COL].astype(str)
    df[POD_UID_COL] = df[POD_UID_COL].astype(str)
    return df


def latency_map_first_batch(pod_df: pd.DataFrame, eps_s: float) -> Dict[LatencyKey, float]:
    """
    For each ReplicaSet prefix, take the first apply-time as t0, then consider the first-batch
    as pods with apply_time within eps_s of t0. Return latency (running_time - apply_time)
    keyed by (priority, rs_prefix, replica_index_in_first_batch).
    """
    apply_df = pod_df[pod_df[POD_EVENT_COL] == "apply-time"].copy()
    run_df = pod_df[pod_df[POD_EVENT_COL] == "running-time"].copy()

    apply_df = apply_df.sort_values(POD_TIME_COL).drop_duplicates(subset=[POD_UID_COL], keep="first")
    run_df = run_df.sort_values(POD_TIME_COL).drop_duplicates(subset=[POD_UID_COL], keep="first")

    apply_df["rs_prefix"] = apply_df[POD_NAME_COL].map(rs_prefix_from_pod_name)

    joined = apply_df.merge(
        run_df[[POD_UID_COL, POD_TIME_COL]].rename(columns={POD_TIME_COL: "running_time_s"}),
        on=POD_UID_COL,
        how="left",
    ).rename(columns={POD_TIME_COL: "apply_time_s"})

    out: Dict[LatencyKey, float] = {}

    for rs_prefix, g in joined.groupby("rs_prefix", sort=False):
        g = g.sort_values("apply_time_s")
        if g.empty:
            continue
        t0 = float(g["apply_time_s"].iloc[0])
        batch_mask = (g["apply_time_s"] - t0) <= float(eps_s)
        batch_n = int(batch_mask.sum())
        if batch_n <= 0:
            continue

        first_batch = g.iloc[:batch_n]
        for i, (_, row) in enumerate(first_batch.iterrows()):
            pr = row.get(POD_PRIO_COL)
            at = row.get("apply_time_s")
            rt = row.get("running_time_s")
            if pd.isna(pr) or pd.isna(at) or pd.isna(rt):
                continue
            lat = float(rt) - float(at)
            if math.isfinite(lat) and lat >= 0.0:
                out[(int(pr), str(rs_prefix), int(i))] = float(lat)

    return out


def latency_means_from_map(lat_map: Dict[LatencyKey, float]) -> Dict[str, float]:
    """Return latency_s_p{1..4} and latency_s_total from a map (means)."""
    out: Dict[str, float] = {}
    if not lat_map:
        for p in range(1, MAX_K_OUT + 1):
            out[f"latency_s_p{p}"] = float("nan")
        out["latency_s_total"] = float("nan")
        return out

    vals_all = np.array(list(lat_map.values()), dtype=float)
    out["latency_s_total"] = float(np.nanmean(vals_all)) if vals_all.size else float("nan")

    for p in range(1, MAX_K_OUT + 1):
        vals_p = np.array([v for (prio, _rs, _i), v in lat_map.items() if int(prio) == p], dtype=float)
        out[f"latency_s_p{p}"] = float(np.nanmean(vals_p)) if vals_p.size else float("nan")

    return out


def latency_deltas_from_maps(def_map: Dict[LatencyKey, float], plu_map: Dict[LatencyKey, float]) -> Dict[str, float]:
    """Return delta_latency_s_p{1..4} and delta_latency_s_total (plugin - default) over intersection keys."""
    keys = sorted(set(def_map.keys()) & set(plu_map.keys()))
    out: Dict[str, float] = {}
    if not keys:
        out["delta_latency_s_total"] = float("nan")
        for p in range(1, MAX_K_OUT + 1):
            out[f"delta_latency_s_p{p}"] = float("nan")
        return out

    diffs_all = np.array([float(plu_map[k]) - float(def_map[k]) for k in keys], dtype=float)
    out["delta_latency_s_total"] = float(np.mean(diffs_all)) if diffs_all.size else float("nan")

    for p in range(1, MAX_K_OUT + 1):
        kp = [k for k in keys if int(k[0]) == p]
        if not kp:
            out[f"delta_latency_s_p{p}"] = float("nan")
            continue
        diffs_p = np.array([float(plu_map[k]) - float(def_map[k]) for k in kp], dtype=float)
        out[f"delta_latency_s_p{p}"] = float(np.mean(diffs_p)) if diffs_p.size else float("nan")

    return out


def per_pod_rows_from_maps(
    *,
    job_name: str,
    plugin_config: str,
    seed: str,
    def_map: Dict[LatencyKey, float],
    plu_map: Dict[LatencyKey, float],
) -> List[Dict[str, object]]:
    """Per-seed per-pod stats rows (intersection keys only)."""
    keys = sorted(set(def_map.keys()) & set(plu_map.keys()))
    rows: List[Dict[str, object]] = []
    for (prio, rs_prefix, replica_idx) in keys:
        ld = float(def_map[(prio, rs_prefix, replica_idx)])
        lp = float(plu_map[(prio, rs_prefix, replica_idx)])
        rows.append(
            {
                "job_name": job_name,
                "plugin_config": plugin_config,
                "seed": seed,
                "priority": int(prio),
                "rs_prefix": str(rs_prefix),
                "replica_index": int(replica_idx),
                "latency_default_s": ld,
                "latency_plugin_s": lp,
                "delta_latency_s": lp - ld,
            }
        )
    return rows


# -----------------------------
# Series building
# -----------------------------

def read_general_series(general_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(general_csv)
    if TIME_COL not in df.columns:
        raise SystemExit(f"{general_csv} missing column: {TIME_COL}")

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).copy()
    if df.empty:
        return df

    df = df.sort_values(TIME_COL).copy()
    t0 = float(df[TIME_COL].iloc[0])
    df[TIME_COL] = (df[TIME_COL] - t0).astype(float)
    df[TIME_COL] = df[TIME_COL].round(6)  # stabilize grouping across seeds
    return df


def build_seed_series(*, general_csv: Path, pod_csv: Path, eps_s: float) -> pd.DataFrame:
    """
    Build a per-seed time series with BASE column names (no _mean suffix).
    Later we average across seeds at each time_s and add _mean.
    """
    df = read_general_series(general_csv)
    if df.empty:
        return df

    # Util columns
    for c in [CPU_RUN_COL, MEM_RUN_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

    df["eff_run_util"] = df[[CPU_RUN_COL, MEM_RUN_COL]].max(axis=1)

    # dt (for cumulative running pod-seconds)
    t = df[TIME_COL].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0.0, None)

    # R_cum_p*: integrate running_p* over time
    for p in range(1, MAX_K_OUT + 1):
        run_col = f"{RUNNING_PREFIX}{p}"
        if run_col in df.columns:
            y = pd.to_numeric(df[run_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            y = np.zeros_like(t, dtype=float)
        df[f"R_cum_p{p}"] = np.cumsum(y * dt)

    df["R_cum_total"] = df[[f"R_cum_p{p}" for p in range(1, MAX_K_OUT + 1)]].sum(axis=1)

    # D_cum_p*: take deletions cumulative directly (already cumulative counter)
    for p in range(1, MAX_K_OUT + 1):
        del_col = f"{DELETIONS_CUM_PREFIX}{p}"
        df[f"D_cum_p{p}"] = pd.to_numeric(df[del_col], errors="coerce") if del_col in df.columns else np.nan

    df["D_cum_total"] = df[[f"D_cum_p{p}" for p in range(1, MAX_K_OUT + 1)]].sum(axis=1, min_count=1)

    # Latency (scalar per seed) replicated over time
    pod_df = read_pod_df(pod_csv)
    lat_map = latency_map_first_batch(pod_df, eps_s=float(eps_s))
    lat_means = latency_means_from_map(lat_map)

    for p in range(1, MAX_K_OUT + 1):
        df[f"latency_s_p{p}"] = float(lat_means[f"latency_s_p{p}"])
    df["latency_s_total"] = float(lat_means["latency_s_total"])

    keep = (
        [TIME_COL, CPU_RUN_COL, MEM_RUN_COL, "eff_run_util"]
        + [f"R_cum_p{p}" for p in range(1, MAX_K_OUT + 1)]
        + ["R_cum_total"]
        + [f"D_cum_p{p}" for p in range(1, MAX_K_OUT + 1)]
        + ["D_cum_total"]
        + [f"latency_s_p{p}" for p in range(1, MAX_K_OUT + 1)]
        + ["latency_s_total"]
    )
    return df[keep].copy()


def mean_series_across_seeds(seed_series: List[pd.DataFrame]) -> pd.DataFrame:
    """Average numeric columns across seeds grouped by time_s; add _mean suffix."""
    if not seed_series:
        return pd.DataFrame(columns=[TIME_COL])

    df_all = pd.concat(seed_series, axis=0, ignore_index=True)
    if df_all.empty:
        return pd.DataFrame(columns=[TIME_COL])

    metric_cols = [c for c in df_all.columns if c != TIME_COL and pd.api.types.is_numeric_dtype(df_all[c])]
    g = df_all.groupby(TIME_COL, dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    g = g.sort_values(TIME_COL)

    g = g.rename(columns={c: f"{c}_mean" for c in metric_cols})
    g[TIME_COL] = g[TIME_COL].astype(float)
    return g


def write_series_files(*, default_root: Path, plugin_root: Path, out_dir: Path, eps_s: float) -> None:
    series_root = out_dir / "series"
    default_out = series_root / "default"
    plugin_out = series_root / "plugin"
    default_out.mkdir(parents=True, exist_ok=True)
    plugin_out.mkdir(parents=True, exist_ok=True)

    def _finalize_series(df_mean: pd.DataFrame, *, scheduler: str, job_name: str, n_seed: int) -> pd.DataFrame:
        df_mean.insert(0, "n_seed", int(n_seed))
        df_mean.insert(0, "job_name", job_name)
        df_mean.insert(0, "scheduler", scheduler)

        df_mean = df_mean.rename(
            columns={
                f"{CPU_RUN_COL}_mean": "cpu_run_util_mean",
                f"{MEM_RUN_COL}_mean": "mem_run_util_mean",
                "eff_run_util_mean": "eff_run_util_mean",
                TIME_COL: "time_s",
            }
        )

        for c in SERIES_COLS:
            if c not in df_mean.columns:
                df_mean[c] = np.nan
        df_mean = df_mean[SERIES_COLS]

        round_numeric_df(df_mean, exclude=["scheduler", "job_name"])
        return df_mean

    # ---- default: per job across all seeds ----
    for job_dir in sorted(default_root.iterdir()):
        if not job_dir.is_dir():
            continue
        job_name = parse_job_dir_name(job_dir.name)
        if job_name is None:
            continue

        seed_series: List[pd.DataFrame] = []
        seed_count = 0
        for _seed, seed_dir in iter_seed_dirs(job_dir):
            seed_count += 1
            gen = seed_dir / GENERAL_STATS_FILENAME
            pod = seed_dir / POD_STATS_FILENAME
            try:
                s = build_seed_series(general_csv=gen, pod_csv=pod, eps_s=eps_s)
                if not s.empty:
                    seed_series.append(s)
            except Exception:
                continue

        df_mean = mean_series_across_seeds(seed_series)
        if df_mean.empty:
            continue

        df_out = _finalize_series(df_mean, scheduler="default", job_name=job_name, n_seed=seed_count)
        out_path = default_out / f"{safe_stem(job_name)}.csv"
        df_out.to_csv(out_path, index=False)

    # ---- plugin: per run-dir across all seeds ----
    for run_dir in sorted(plugin_root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_plugin_run_dir(run_dir.name)
        if parsed is None:
            continue
        job_name, plugin_config = parsed

        seed_series: List[pd.DataFrame] = []
        seed_count = 0
        for _seed, seed_dir in iter_seed_dirs(run_dir):
            seed_count += 1
            gen = seed_dir / GENERAL_STATS_FILENAME
            pod = seed_dir / POD_STATS_FILENAME
            try:
                s = build_seed_series(general_csv=gen, pod_csv=pod, eps_s=eps_s)
                if not s.empty:
                    seed_series.append(s)
            except Exception:
                continue

        df_mean = mean_series_across_seeds(seed_series)
        if df_mean.empty:
            continue

        df_out = _finalize_series(df_mean, scheduler=plugin_config, job_name=job_name, n_seed=seed_count)
        out_name = f"{safe_stem(plugin_config)}__{safe_stem(job_name)}.csv"
        out_path = plugin_out / out_name
        df_out.to_csv(out_path, index=False)


# -----------------------------
# results_paired + per_pod_stats
# -----------------------------

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


def load_time_vector(general_csv: Path) -> np.ndarray:
    df = pd.read_csv(general_csv, usecols=[TIME_COL])
    t = pd.to_numeric(df[TIME_COL], errors="coerce").dropna().to_numpy(dtype=float)
    return t


def read_general_for_horizon(general_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(general_csv)
    if TIME_COL not in df.columns:
        raise SystemExit(f"{general_csv} missing {TIME_COL}")
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).copy()
    df = df.sort_values(TIME_COL).copy()
    t0 = float(df[TIME_COL].iloc[0])
    df[TIME_COL] = (df[TIME_COL] - t0).astype(float)
    return df


def compute_seed_horizon_metrics(general_csv: Path, H: float) -> Dict[str, float]:
    """
    Compute time-weighted mean util and mean running pods (per priority) over [0,H],
    and deletions at time H (stepwise).
    """
    df = read_general_for_horizon(general_csv)
    t = df[TIME_COL].to_numpy(dtype=float)

    cpu = (
        pd.to_numeric(df[CPU_RUN_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if CPU_RUN_COL in df.columns
        else np.zeros_like(t)
    )
    mem = (
        pd.to_numeric(df[MEM_RUN_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if MEM_RUN_COL in df.columns
        else np.zeros_like(t)
    )
    eff = np.maximum(cpu, mem)

    out: Dict[str, float] = {
        "cpu_run_util_mean": time_weighted_mean_step(t, cpu, H),
        "mem_run_util_mean": time_weighted_mean_step(t, mem, H),
        "util_eff_run_mean": time_weighted_mean_step(t, eff, H),
    }

    for p in range(1, MAX_K_OUT + 1):
        run_col = f"{RUNNING_PREFIX}{p}"
        y = (
            pd.to_numeric(df[run_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if run_col in df.columns
            else np.zeros_like(t)
        )
        out[f"R_p{p}_mean"] = time_weighted_mean_step(t, y, H)

        del_col = f"{DELETIONS_CUM_PREFIX}{p}"
        if del_col in df.columns:
            d = pd.to_numeric(df[del_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            out[f"D_p{p}"] = value_at_step(t, d, H, default=0.0)
        else:
            out[f"D_p{p}"] = float("nan")

    out["R_total_mean"] = float(sum(out[f"R_p{p}_mean"] for p in range(1, MAX_K_OUT + 1)))
    out["D_total"] = float(np.nansum([out[f"D_p{p}"] for p in range(1, MAX_K_OUT + 1)]))
    return out


def write_per_pod_stats(per_pod_rows: List[Dict[str, object]], out_dir: Path) -> None:
    out_root = out_dir / "per_pod_stats"
    out_root.mkdir(parents=True, exist_ok=True)
    if not per_pod_rows:
        return

    df = pd.DataFrame(per_pod_rows)
    for c in PER_POD_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[PER_POD_COLS].sort_values(
        ["job_name", "plugin_config", "seed", "priority", "rs_prefix", "replica_index"],
        kind="mergesort",
    )

    for (job_name, plugin_config), g in df.groupby(["job_name", "plugin_config"], dropna=False):
        out_name = f"{safe_stem(plugin_config)}__{safe_stem(job_name)}.csv"
        path = out_root / out_name
        round_numeric_df(g, exclude=["job_name", "plugin_config", "seed", "rs_prefix"])
        g.to_csv(path, index=False)


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

    print("This may take a while...")

    # 1) series files
    write_series_files(default_root=default_root, plugin_root=plugin_root, out_dir=out_dir, eps_s=eps_s)

    # Build index for default: (job_name, seed) -> (general_csv, pod_csv)
    default_idx: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
    for job_dir in sorted(default_root.iterdir()):
        if not job_dir.is_dir():
            continue
        job_name = parse_job_dir_name(job_dir.name)
        if job_name is None:
            continue
        for seed, seed_dir in iter_seed_dirs(job_dir):
            default_idx[(job_name, seed)] = (seed_dir / GENERAL_STATS_FILENAME, seed_dir / POD_STATS_FILENAME)

    # Iterate plugin runs
    seed_rows: List[Dict[str, object]] = []
    per_pod_rows: List[Dict[str, object]] = []

    for run_dir in sorted(plugin_root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_plugin_run_dir(run_dir.name)
        if parsed is None:
            continue
        job_name, plugin_config = parsed

        plugin_seeds = {seed for seed, _ in iter_seed_dirs(run_dir)}
        default_seeds = {seed for (jn, seed) in default_idx.keys() if jn == job_name}
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

            # Common horizon H = min(T_end_default, T_end_plugin)
            t_def_raw = load_time_vector(def_gen)
            t_plu_raw = load_time_vector(plu_gen)
            if t_def_raw.size == 0 or t_plu_raw.size == 0:
                continue

            T_def = float(np.nanmax(t_def_raw) - np.nanmin(t_def_raw))
            T_plu = float(np.nanmax(t_plu_raw) - np.nanmin(t_plu_raw))
            if not (math.isfinite(T_def) and math.isfinite(T_plu)) or T_def <= 0.0 or T_plu <= 0.0:
                continue

            H = float(min(T_def, T_plu))
            if H <= 0.0:
                continue

            # Horizon metrics (per scheduler)
            def_m = compute_seed_horizon_metrics(def_gen, H)
            plu_m = compute_seed_horizon_metrics(plu_gen, H)

            d_cpu = float(plu_m["cpu_run_util_mean"]) - float(def_m["cpu_run_util_mean"])
            d_mem = float(plu_m["mem_run_util_mean"]) - float(def_m["mem_run_util_mean"])
            d_eff = float(plu_m["util_eff_run_mean"]) - float(def_m["util_eff_run_mean"])

            dR = {p: float(plu_m[f"R_p{p}_mean"]) - float(def_m[f"R_p{p}_mean"]) for p in range(1, MAX_K_OUT + 1)}
            dD = {p: float(plu_m[f"D_p{p}"]) - float(def_m[f"D_p{p}"]) for p in range(1, MAX_K_OUT + 1)}
            dR_total = float(sum(dR.values()))
            dD_total = float(sum(dD.values()))

            # Latency maps + deltas + per_pod rows
            try:
                def_pod_df = read_pod_df(def_pod)
                plu_pod_df = read_pod_df(plu_pod)
                def_map = latency_map_first_batch(def_pod_df, eps_s=eps_s)
                plu_map = latency_map_first_batch(plu_pod_df, eps_s=eps_s)
            except Exception:
                def_map, plu_map = {}, {}

            lat_delta = latency_deltas_from_maps(def_map, plu_map)
            per_pod_rows.extend(
                per_pod_rows_from_maps(
                    job_name=job_name,
                    plugin_config=plugin_config,
                    seed=seed,
                    def_map=def_map,
                    plu_map=plu_map,
                )
            )

            # Optimization totals (plugin only)
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
                    "delta_latency_s_p1": float(lat_delta["delta_latency_s_p1"]),
                    "delta_latency_s_p2": float(lat_delta["delta_latency_s_p2"]),
                    "delta_latency_s_p3": float(lat_delta["delta_latency_s_p3"]),
                    "delta_latency_s_p4": float(lat_delta["delta_latency_s_p4"]),
                    "delta_latency_s_total": float(lat_delta["delta_latency_s_total"]),
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
        write_per_pod_stats(per_pod_rows, out_dir)
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

    # per-pod stats
    write_per_pod_stats(per_pod_rows, out_dir)

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
