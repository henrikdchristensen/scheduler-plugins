#!/usr/bin/env python3
"""scripts/kwok_trace_replayer/seal_results.py

Aggregate KWOK trace replayer outputs across an arbitrary number of seeds.

This script emits:
  1) results_long.csv
     One row per (scheduler variant, job, seed) with raw end-of-run metrics.
     This is meant for plotting across seeds (e.g., box plots).

  2) results_agg.csv
     One row per (scheduler variant, job) aggregated across seeds.
     It has the *same* columns as results_long.csv, but replaces "seed" with
     "n_seeds".

  3) results_paired.csv
     One row per (job, plugin_config) with mean/std of per-seed deltas
     (plugin - default) across the seeds that exist in both runs.

Additionally, the script writes time-series mean CSVs under <out-dir>/series/:
  <scheduler>__<job_name>.csv
  Each file contains mean time series across seeds for a given scheduler+job.

Input layout (under --root):
  default/<job_name>/<seed>/{general_stats.csv,pod_stats.csv}
  plugin/<run_dir>/<seed>/{general_stats.csv,pod_stats.csv}

New trace replayer naming (examples):
  default job dir:
    nodes=16_prio=1_arrival=8s
  plugin run dir:
    mode=periodic8s_blocking=0_defpreempt=0_nodes=16_prio=1_arrival=8s

Matching rule:
  - We parse (nodes, prio, arrival) from both default/<job_name> and plugin/<run_dir>.
  - We canonicalize job_name as: nodes=<N>_prio=<K>_arrival=<A>s
  - Plugin runs are matched to default by this canonical job_name.

Latency (time-to-first-admit):
  - For each ReplicaSet prefix (rs-XXXXXX), we consider the "first batch" of
    applies within eps_s of the first apply timestamp for that prefix.
  - We compute per-run latency percentiles over these latencies.
  - For paired deltas, we compute latency deltas over the intersection of
    comparable pods/replicas keyed by (priority, rs_prefix, replica_index).

Notes:
  - The script makes no assumptions about which plugin modes/configs exist.
  - It supports an arbitrary number of seeds.

"""

import argparse, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.kwok_trace_replayer.trace_helpers import rs_prefix_from_pod_name

# -----------------------------
# Numeric formatting
# -----------------------------

# Round floating outputs to this many decimals when writing CSVs.
FLOAT_DECIMALS = 4

# -----------------------------
# Filenames / schema
# -----------------------------

GENERAL_STATS_FILENAME = "general_stats.csv"
POD_STATS_FILENAME = "pod_stats.csv"

TIME_COL = "time_s"
CPU_RUN_COL = "cpu_run_util"
MEM_RUN_COL = "mem_run_util"
CPU_REQ_COL = "cpu_req_util"
MEM_REQ_COL = "mem_req_util"

RUNNING_PREFIX = "running_p"
UNSCHED_PREFIX = "unsched_p"
DELETIONS_CUM_PREFIX = "deletions_cum_p"

POD_EVENT_COL = "event"
POD_NAME_COL = "pod_name"
POD_UID_COL = "pod_uid"
POD_PRIO_COL = "priority"
POD_TIME_COL = "time_s"


# -----------------------------
# Types
# -----------------------------

@dataclass(frozen=True)
class TestCombo:
    job_name: str  # nodes=<N>_prio=<K>_arrival=<A>s
    n_nodes: int
    k_max: int
    mean_arrival_s: float

@dataclass(frozen=True)
class PluginConfig:
    mode_raw: str
    blocking: bool
    defpreempt: bool

    def key(self) -> str:
        # Stable order, excludes the job/regime tokens.
        b = "1" if self.blocking else "0"
        d = "1" if self.defpreempt else "0"
        return f"mode={self.mode_raw}_blocking={b}_defpreempt={d}"

# latency key that is stable across runs
LatencyKey = Tuple[int, str, int]  # (priority, rs_prefix, replica_index_in_first_batch)

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate KWOK trace replayer outputs across seeds and write CSV summaries."
    )
    p.add_argument("--root", required=True,
        help="Root directory containing 'default/' and 'plugin/' subfolders.",
    )
    p.add_argument("--out-dir", default=None,
        help="Where to write outputs (default: <root>/sealed_out).",
    )
    p.add_argument("--eps-s", type=float, default=1.0,
        help="Epsilon window (seconds) for the first-batch latency heuristic (default: 1.0).",
    )
    p.add_argument("--time-round", type=int, default=3,
        help="Decimals to round time_s when building the cross-seed time grid (default: 3).",
    )
    return p.parse_args()


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_int_token(tok: str, prefix: str) -> Optional[int]:
    if not tok.startswith(prefix):
        return None
    v = tok.split("=", 1)[1] if "=" in tok else tok[len(prefix) :]
    v = v.strip()
    return int(v) if v else None


def parse_float_seconds_token(tok: str, prefix: str) -> Optional[float]:
    if not tok.startswith(prefix):
        return None
    v = tok.split("=", 1)[1] if "=" in tok else tok[len(prefix) :]
    v = v.strip()
    if v.endswith("s"):
        v = v[:-1]
    return float(v) if v else None


def fmt_arrival(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:g}"


def canon_job_name(n_nodes: int, k_max: int, mean_arrival_s: float) -> str:
    return f"nodes={n_nodes}_prio={k_max}_arrival={fmt_arrival(mean_arrival_s)}s"


def parse_test_combo(dirname: str) -> TestCombo:
    """Parse default job dir names.

    Accepts:
      - nodes=16_prio=4_arrival=8s
      - nodes16_prio4_arrival8
      - nodes=16_prio=4_arrival=8
    """
    tokens = str(dirname).split("_")
    n: Optional[int] = None
    k: Optional[int] = None
    arr: Optional[float] = None

    for tok in tokens:
        if n is None and tok.startswith("nodes"):
            n = parse_int_token(tok, "nodes")
        if k is None and tok.startswith("prio"):
            k = parse_int_token(tok, "prio")
        if arr is None and tok.startswith("arrival"):
            arr = parse_float_seconds_token(tok, "arrival")

    if n is None or k is None or arr is None:
        raise SystemExit(
            f"Could not parse job from directory name: '{dirname}'. "
            "Expected tokens like nodes=16, prio=4, arrival=8s (or without '=' / 's')."
        )

    job_name = canon_job_name(n, k, arr)
    return TestCombo(job_name=job_name, n_nodes=n, k_max=k, mean_arrival_s=arr)


def parse_plugin_dirname(dirname: str) -> Tuple[TestCombo, PluginConfig]:
    """Parse plugin run dir names.

    Expected key-value tokens separated by underscores, e.g.:
      mode=periodic8s_blocking=0_defpreempt=1_nodes=16_prio=1_arrival=8s

    Extra tokens are ignored.
    """
    tokens = str(dirname).split("_")
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
    except Exception as e:
        raise SystemExit(f"Could not parse regime fields from plugin dir '{dirname}': {e}")

    test_combo = TestCombo(job_name=canon_job_name(n, kmax, arr), n_nodes=n, k_max=kmax, mean_arrival_s=arr)

    mode_raw = kv.get("mode", "unknown")

    blocking_raw = kv.get("blocking", "0")
    blocking = str(blocking_raw).strip().lower() in {"1", "true", "yes"}

    defpreempt_raw = kv.get("defpreempt", "0")
    try:
        defpreempt = bool(int(str(defpreempt_raw).strip()))
    except Exception:
        defpreempt = False

    cfg = PluginConfig(mode_raw=mode_raw, blocking=blocking, defpreempt=defpreempt)
    return test_combo, cfg

def require_cols(df: pd.DataFrame, path: Path, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")

# -----------------------------
# Metric computation
# -----------------------------

def sorted_time_and_dt(df: pd.DataFrame, time_col: str = TIME_COL) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (t, dt, order) where t is sorted and normalized to start at 0.
    """
    t_raw = df[time_col].astype(float).to_numpy()
    if t_raw.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int)
    order = np.argsort(t_raw)
    t = t_raw[order]
    t0 = float(t[0])
    t = t - t0
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0.0, None)
    return t, dt, order

def step_values_at(t: np.ndarray, y: np.ndarray, grid: np.ndarray, *, default: float = 0.0) -> np.ndarray:
    """
    Stepwise-constant y(t) at each time in grid, using last observation carried forward.
    """
    if t.size == 0 or y.size == 0:
        return np.full_like(grid, fill_value=float(default), dtype=float)
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.full(grid.shape, float(default), dtype=float)
    ok = idx >= 0
    idx2 = np.clip(idx, 0, y.size - 1)
    out[ok] = y[idx2[ok]]
    return out

def nearest_values_at(t: np.ndarray, y: np.ndarray, grid: np.ndarray, *, default: float = 0.0) -> np.ndarray:
    """
    Nearest-neighbor sampling of y(t) at each time in grid.
    For each grid point g, pick the observation y(t_i) where |t_i - g| is minimal.
    Uses default when inputs are empty.
    """
    if t.size == 0 or y.size == 0:
        return np.full_like(grid, fill_value=float(default), dtype=float)
    # insertion positions
    idx = np.searchsorted(t, grid, side="left")
    idx_r = np.clip(idx, 0, t.size - 1)
    idx_l = np.clip(idx - 1, 0, t.size - 1)
    t_r = t[idx_r]
    t_l = t[idx_l]
    # choose left when it's closer or equal (stable towards earlier sample)
    choose_l = np.abs(t_l - grid) <= np.abs(t_r - grid)
    idx_best = np.where(choose_l, idx_l, idx_r)
    return y[idx_best]

def latency_map_first_batch(pod_df: pd.DataFrame, *, eps_s: float) -> Dict[LatencyKey, float]:
    """
    Latency map keyed by (priority, rs_prefix, replica_index_in_first_batch).
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
            rt = row.get("running_time_s")
            at = row.get("apply_time_s")
            pr = row.get(POD_PRIO_COL)
            if pd.isna(rt) or pd.isna(at) or pd.isna(pr):
                continue
            lat = float(rt) - float(at)
            if math.isfinite(lat) and lat >= 0.0:
                out[(int(pr), str(rs_prefix), int(i))] = float(lat)

    return out

def percentiles(vals: np.ndarray, qs: List[float]) -> Dict[float, float]:
    if vals.size == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.quantile(vals, q)) for q in qs}

def compute_run_metrics(
    general_path: Path,
    pod_path: Path,
    *,
    eps_s: float,
) -> Tuple[Dict[str, float], Dict[LatencyKey, float]]:
    """Compute per-run scalar metrics + per-run latency map."""
    df_g = pd.read_csv(general_path)
    require_cols(df_g, general_path, [TIME_COL, CPU_RUN_COL, MEM_RUN_COL, CPU_REQ_COL, MEM_REQ_COL])

    t, dt, order = sorted_time_and_dt(df_g, TIME_COL)
    T_end = float(t[-1]) if t.size else 0.0

    cpu_run = df_g[CPU_RUN_COL].astype(float).to_numpy()[order] if t.size else np.array([], dtype=float)
    mem_run = df_g[MEM_RUN_COL].astype(float).to_numpy()[order] if t.size else np.array([], dtype=float)
    cpu_req = df_g[CPU_REQ_COL].astype(float).to_numpy()[order] if t.size else np.array([], dtype=float)
    mem_req = df_g[MEM_REQ_COL].astype(float).to_numpy()[order] if t.size else np.array([], dtype=float)

    eff_run = np.maximum(cpu_run, mem_run) if t.size else np.array([], dtype=float)
    eff_req = np.maximum(cpu_req, mem_req) if t.size else np.array([], dtype=float)

    def _time_mean(y: np.ndarray) -> float:
        return float(np.sum(y * dt) / T_end) if (t.size and T_end > 0) else 0.0

    cpu_run_mean = _time_mean(cpu_run)
    mem_run_mean = _time_mean(mem_run)
    cpu_req_mean = _time_mean(cpu_req)
    mem_req_mean = _time_mean(mem_req)
    util_eff_run_mean = _time_mean(eff_run)
    util_eff_req_mean = _time_mean(eff_req)

    # detect priorities from running_p columns
    prios: List[int] = []
    for c in df_g.columns:
        if c.startswith(RUNNING_PREFIX):
            try:
                prios.append(int(c[len(RUNNING_PREFIX) :]))
            except ValueError:
                pass
    prios = sorted(set(prios))

    metrics: Dict[str, float] = {
        "T_end_s": T_end,

        # per-resource means
        "cpu_run_util_mean": cpu_run_mean,
        "mem_run_util_mean": mem_run_mean,
        "cpu_req_util_mean": cpu_req_mean,
        "mem_req_util_mean": mem_req_mean,

        # effective means (max(cpu,mem))
        "util_eff_run_mean": util_eff_run_mean,
        "util_eff_req_mean": util_eff_req_mean,
    }

    # R_p(T)
    for p in prios:
        col = f"{RUNNING_PREFIX}{p}"
        y = df_g[col].fillna(0).astype(float).to_numpy()[order]
        metrics[f"R_p{p}"] = float(np.sum(y * dt))

    # D_p(T)
    for p in prios:
        col = f"{DELETIONS_CUM_PREFIX}{p}"
        if col not in df_g.columns:
            continue
        y = df_g[col].fillna(0).astype(float).to_numpy()[order]
        metrics[f"D_p{p}"] = float(y[-1]) if y.size else 0.0

    df_p = pd.read_csv(pod_path)
    require_cols(df_p, pod_path, [POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL])

    df_p[POD_TIME_COL] = df_p[POD_TIME_COL].astype(float)
    df_p[POD_PRIO_COL] = df_p[POD_PRIO_COL].astype(int)
    df_p[POD_EVENT_COL] = df_p[POD_EVENT_COL].astype(str)
    df_p[POD_NAME_COL] = df_p[POD_NAME_COL].astype(str)
    df_p[POD_UID_COL] = df_p[POD_UID_COL].astype(str)

    lat_map = latency_map_first_batch(df_p, eps_s=float(eps_s))
    lat_vals = np.asarray(list(lat_map.values()), dtype=float)

    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    pct = percentiles(lat_vals, qs)

    metrics["latency_first_admit_p05_s"] = pct[0.05]
    metrics["latency_first_admit_p25_s"] = pct[0.25]
    metrics["latency_first_admit_median_s"] = pct[0.5]
    metrics["latency_first_admit_p75_s"] = pct[0.75]
    metrics["latency_first_admit_p95_s"] = pct[0.95]
    metrics["latency_first_admit_mean_s"] = float(np.mean(lat_vals)) if lat_vals.size else float("nan")

    return metrics, lat_map

def series_metrics_at_horizon(general_csv: Path, horizon_s: float) -> Dict[str, object]:
    """
    Horizon-aware metrics using stepwise carry-forward.
    """
    df = pd.read_csv(general_csv)
    require_cols(df, general_csv, [TIME_COL, CPU_RUN_COL, MEM_RUN_COL, CPU_REQ_COL, MEM_REQ_COL])

    t_raw = df[TIME_COL].astype(float).to_numpy()
    if t_raw.size == 0:
        return {"prios": [], "R": {}, "D": {}, "util_eff_mean": 0.0}

    order = np.argsort(t_raw)
    t = t_raw[order]
    t0 = float(t[0])
    t = t - t0

    # build t_clip including horizon endpoint
    mask = t <= horizon_s
    t_clip = t[mask]
    if t_clip.size == 0:
        t_clip = np.array([horizon_s], dtype=float)
    elif t_clip[-1] < horizon_s:
        t_clip = np.concatenate([t_clip, np.array([horizon_s], dtype=float)])

    dt = np.diff(t_clip, prepend=t_clip[0])
    dt = np.clip(dt, 0.0, None)

    cpu_run = df[CPU_RUN_COL].astype(float).to_numpy()[order]
    mem_run = df[MEM_RUN_COL].astype(float).to_numpy()[order]
    cpu_req = df[CPU_REQ_COL].astype(float).to_numpy()[order]
    mem_req = df[MEM_REQ_COL].astype(float).to_numpy()[order]

    eff_run = np.maximum(cpu_run, mem_run)
    eff_req = np.maximum(cpu_req, mem_req)

    cpu_run_clip = step_values_at(t, cpu_run, t_clip, default=0.0)
    mem_run_clip = step_values_at(t, mem_run, t_clip, default=0.0)
    cpu_req_clip = step_values_at(t, cpu_req, t_clip, default=0.0)
    mem_req_clip = step_values_at(t, mem_req, t_clip, default=0.0)
    eff_run_clip = step_values_at(t, eff_run, t_clip, default=0.0)
    eff_req_clip = step_values_at(t, eff_req, t_clip, default=0.0)

    def _time_mean_clip(y_clip: np.ndarray) -> float:
        return float(np.sum(y_clip * dt) / float(horizon_s)) if horizon_s > 0 else 0.0

    cpu_run_mean = _time_mean_clip(cpu_run_clip)
    mem_run_mean = _time_mean_clip(mem_run_clip)
    cpu_req_mean = _time_mean_clip(cpu_req_clip)
    mem_req_mean = _time_mean_clip(mem_req_clip)
    util_eff_run_mean = _time_mean_clip(eff_run_clip)
    util_eff_req_mean = _time_mean_clip(eff_req_clip)

    prios: List[int] = []
    for c in df.columns:
        if c.startswith(RUNNING_PREFIX):
            try:
                prios.append(int(c[len(RUNNING_PREFIX) :]))
            except ValueError:
                pass
    prios = sorted(set(prios))

    R: Dict[int, float] = {}
    D: Dict[int, float] = {}

    for p in prios:
        run_col = f"{RUNNING_PREFIX}{p}"
        y_run = df[run_col].fillna(0).astype(float).to_numpy()[order]
        y_vals = step_values_at(t, y_run, t_clip, default=0.0)
        R[p] = float(np.sum(y_vals * dt))

        del_col = f"{DELETIONS_CUM_PREFIX}{p}"
        if del_col in df.columns:
            y_del = df[del_col].fillna(0).astype(float).to_numpy()[order]
            D[p] = float(step_values_at(t, y_del, np.array([horizon_s], dtype=float), default=0.0)[0])

    return {
        "prios": prios,
        "R": R,
        "D": D,

        "cpu_run_util_mean": cpu_run_mean,
        "mem_run_util_mean": mem_run_mean,
        "cpu_req_util_mean": cpu_req_mean,
        "mem_req_util_mean": mem_req_mean,
        "util_eff_run_mean": util_eff_run_mean,
        "util_eff_req_mean": util_eff_req_mean,
    }

# -----------------------------
# Filesystem scanning
# -----------------------------

def iter_seed_dirs(parent: Path) -> Iterable[Tuple[str, Path]]:
    if not parent.exists() or not parent.is_dir():
        return
    for d in sorted(parent.iterdir()):
        if not d.is_dir():
            continue
        if (d / GENERAL_STATS_FILENAME).exists() and (d / POD_STATS_FILENAME).exists():
            yield d.name, d

def scan_default(default_root: Path) -> Dict[Tuple[str, str], Tuple[Path, Path]]:
    """
    (job_name, seed) -> (general_csv, pod_csv).
    """
    idx: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
    for job_dir in sorted(default_root.iterdir()):
        if not job_dir.is_dir():
            continue
        try:
            r = parse_test_combo(job_dir.name)
        except SystemExit:
            continue
        for seed, seed_dir in iter_seed_dirs(job_dir):
            idx[(r.job_name, seed)] = (seed_dir / GENERAL_STATS_FILENAME, seed_dir / POD_STATS_FILENAME)
    return idx

def iter_plugin_runs(plugin_root: Path) -> Iterable[Tuple[str, str, Path]]:
    """
    Yield (job_name, plugin_config_key, run_dir_path).
    """
    if not plugin_root.exists() or not plugin_root.is_dir():
        return
    for run_dir in sorted(plugin_root.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            regime, cfg = parse_plugin_dirname(run_dir.name)
        except SystemExit:
            continue
        yield regime.job_name, cfg.key(), run_dir

# -----------------------------
# Output schemas
# -----------------------------

def round_numeric_df(df: pd.DataFrame, *, exclude: List[str] | None = None) -> pd.DataFrame:
    """
    Round numeric columns to FLOAT_DECIMALS, excluding some columns.
    """
    if exclude is None:
        exclude = []
    num_cols = [c for c in df.columns if c not in set(exclude) and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].round(FLOAT_DECIMALS)
    return df

def metric_cols_for_k(max_k: int, *, t_end_col: str = "T_end_s") -> List[str]:
    cols: List[str] = [
        t_end_col,
        "cpu_run_util_mean",
        "mem_run_util_mean",
        "cpu_req_util_mean",
        "mem_req_util_mean",
        "util_eff_run_mean",
        "util_eff_req_mean",
    ]
    for p in range(1, max_k + 1):
        cols.append(f"R_p{p}")
    for p in range(1, max_k + 1):
        cols.append(f"D_p{p}")
    cols += [
        "latency_first_admit_p05_s",
        "latency_first_admit_p25_s",
        "latency_first_admit_median_s",
        "latency_first_admit_mean_s",
        "latency_first_admit_p75_s",
        "latency_first_admit_p95_s",
    ]
    return cols

def results_long_columns(max_k: int) -> List[str]:
    return ["scheduler", "job_name", "seed", *metric_cols_for_k(max_k, t_end_col="T_end_s")]


def results_agg_columns(max_k: int) -> List[str]:
    return ["scheduler", "job_name", "n_seeds", *metric_cols_for_k(max_k, t_end_col="T_end_s_mean")]

# -----------------------------
# Time-series aggregation
# -----------------------------

def load_series(general_csv: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    df = pd.read_csv(general_csv)
    require_cols(df, general_csv, [TIME_COL])

    t_raw = df[TIME_COL].astype(float).to_numpy()
    if t_raw.size == 0:
        return np.array([], dtype=float), {}

    order = np.argsort(t_raw)
    t = t_raw[order]
    t0 = float(t[0])
    t = t - t0

    cols: Dict[str, np.ndarray] = {}
    for c in df.columns:
        if c == TIME_COL:
            continue
        # numeric coercion; non-numeric -> NaN
        cols[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()[order]

    return t, cols


def sanitize_filename(s: str) -> str:
    # keep it filesystem-friendly
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)

def write_series_means(
    *,
    out_dir: Path,
    scheduler: str,
    job_name: str,
    seed_general_paths: List[Path],
    k_max: int,
    time_round: int,
) -> None:
    """
    Write mean time-series across seeds for a single scheduler+job.

    Output grid: one row per *integer* second (0,1,2,...) up to the minimum
    observed T_end across seeds (so each time point is comparable across seeds).

    Sampling: for each seed and each integer second, take the value from the
    *closest* recorded sample time (nearest neighbor).

    Note: time_round is kept for CLI compatibility but is not used.
    """

    series_list: List[Tuple[np.ndarray, Dict[str, np.ndarray]]] = []
    t_ends: List[float] = []

    for p in seed_general_paths:
        t, cols = load_series(p)
        if t.size == 0:
            continue
        series_list.append((t.astype(float), cols))
        t_ends.append(float(t[-1]))

    if not series_list:
        return

    # Common horizon: keep only the time range where all seeds have coverage.
    t_end_common = float(min(t_ends)) if t_ends else 0.0
    t_end_common = max(0.0, t_end_common)

    # Integer-second grid (inclusive)
    grid = np.arange(0, int(math.floor(t_end_common)) + 1, 1, dtype=float)

    # columns we want to output
    base_cols_in = [CPU_RUN_COL, MEM_RUN_COL, CPU_REQ_COL, MEM_REQ_COL]
    base_cols_out = [
        "cpu_run_util_mean",
        "mem_run_util_mean",
        "cpu_req_util_mean",
        "mem_req_util_mean",
    ]

    prio_cols_in: List[str] = []
    prio_cols_out: List[str] = []
    for p in range(1, k_max + 1):
        prio_cols_in += [
            f"{RUNNING_PREFIX}{p}",
            f"{UNSCHED_PREFIX}{p}",
            f"{DELETIONS_CUM_PREFIX}{p}",
        ]
        prio_cols_out += [
            f"running_p{p}_mean",
            f"unsched_p{p}_mean",
            f"deletions_cum_p{p}_mean",
        ]

    want_in = base_cols_in + prio_cols_in
    want_out = base_cols_out + prio_cols_out

    out_cols: Dict[str, np.ndarray] = {}

    for cin, cout in zip(want_in, want_out):
        vals_per_seed: List[np.ndarray] = []
        default_val = 0.0

        for t, cols in series_list:
            y = cols.get(cin)
            if y is None:
                vals_per_seed.append(np.full_like(grid, fill_value=default_val, dtype=float))
                continue

            y_clean = np.nan_to_num(y.astype(float), nan=default_val)
            vals_per_seed.append(nearest_values_at(t, y_clean, grid, default=default_val))

        mat = np.vstack(vals_per_seed)
        out_cols[cout] = np.mean(mat, axis=0)

    df_out = pd.DataFrame({"time_s": grid.astype(int), **out_cols})
    df_out.insert(0, "n_seed", int(len(series_list)))
    df_out.insert(0, "job_name", job_name)
    df_out.insert(0, "scheduler", scheduler)

    # Round numeric outputs (keep time_s as integer)
    round_numeric_df(df_out, exclude=["scheduler", "job_name", "n_seed", "time_s"])

    series_dir = out_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{sanitize_filename(scheduler)}__{sanitize_filename(job_name)}.csv"
    df_out.to_csv(series_dir / fname, index=False)

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

    out_dir = Path(args.out_dir) if args.out_dir else (root / "sealed_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    default_idx = scan_default(default_root)

    # Determine max_k from observed jobs (default + plugin)
    k_candidates: List[int] = []
    for (job_name, _seed) in default_idx.keys():
        try:
            k_candidates.append(parse_test_combo(job_name).k_max)
        except SystemExit:
            # parse_regime expects dir-ish names; job_name is canonical so should parse
            k_candidates.append(int(re.search(r"prio=(\d+)", job_name).group(1)))

    for job_name, _, _ in iter_plugin_runs(plugin_root):
        try:
            k_candidates.append(parse_test_combo(job_name).k_max)
        except SystemExit:
            m = re.search(r"prio=(\d+)", job_name)
            if m:
                k_candidates.append(int(m.group(1)))

    max_k_global = max(k_candidates) if k_candidates else 0

    # Cache: (scheduler, job_name, seed) -> metrics / latency_map
    metrics_cache: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    latency_cache: Dict[Tuple[str, str, str], Dict[LatencyKey, float]] = {}

    # -------------------------
    # results_long.csv
    # -------------------------
    rows_long: List[Dict[str, object]] = []

    # default
    for (job_name, seed), (gen_p, pod_p) in sorted(default_idx.items()):
        metrics, lat_map = compute_run_metrics(gen_p, pod_p, eps_s=float(args.eps_s))
        scheduler = "default"
        metrics_cache[(scheduler, job_name, seed)] = metrics
        latency_cache[(scheduler, job_name, seed)] = lat_map
        rows_long.append({"scheduler": scheduler, "job_name": job_name, "seed": seed, **metrics})

    # plugin
    for job_name, plugin_cfg_key, run_dir in iter_plugin_runs(plugin_root):
        for seed, seed_dir in iter_seed_dirs(run_dir):
            gen_p = seed_dir / GENERAL_STATS_FILENAME
            pod_p = seed_dir / POD_STATS_FILENAME
            metrics, lat_map = compute_run_metrics(gen_p, pod_p, eps_s=float(args.eps_s))
            scheduler = plugin_cfg_key
            metrics_cache[(scheduler, job_name, seed)] = metrics
            latency_cache[(scheduler, job_name, seed)] = lat_map
            rows_long.append({"scheduler": scheduler, "job_name": job_name, "seed": seed, **metrics})

    df_long = pd.DataFrame(rows_long)

    # Ensure stable column set
    for c in results_long_columns(max_k_global):
        if c not in df_long.columns:
            df_long[c] = np.nan

    # Fill missing R/D with 0 (meaning: no such tier in that job)
    for p in range(1, max_k_global + 1):
        for pref in ("R_p", "D_p"):
            col = f"{pref}{p}"
            if col in df_long.columns:
                df_long[col] = pd.to_numeric(df_long[col], errors="coerce").fillna(0.0)

    # numeric coercion
    for c in metric_cols_for_k(max_k_global):
        if c in df_long.columns:
            df_long[c] = pd.to_numeric(df_long[c], errors="coerce")

    df_long = df_long[results_long_columns(max_k_global)]
    round_numeric_df(df_long, exclude=["scheduler", "job_name", "seed"])
    df_long.to_csv(out_dir / "results_long.csv", index=False)

    # -------------------------
    # results_agg.csv (across seeds)
    # -------------------------
    if df_long.empty:
        print("No runs found.")
        return

    numeric_cols = metric_cols_for_k(max_k_global, t_end_col="T_end_s")

    grp = df_long.groupby(["scheduler", "job_name"], dropna=False)

    df_agg = grp[numeric_cols].mean(numeric_only=True).reset_index()
    df_agg.insert(2, "n_seeds", grp.size().to_numpy())

    # Rename T_end column to make clear it is an average across seeds
    if "T_end_s" in df_agg.columns:
        df_agg = df_agg.rename(columns={"T_end_s": "T_end_s_mean"})

    round_numeric_df(df_agg, exclude=["scheduler", "job_name", "n_seeds"])

    # ensure column order and presence
    for c in results_agg_columns(max_k_global):
        if c not in df_agg.columns:
            df_agg[c] = np.nan

    df_agg = df_agg[results_agg_columns(max_k_global)]
    df_agg.to_csv(out_dir / "results_agg.csv", index=False)

    # -------------------------
    # results_paired.csv (plugin - default) aggregated across seeds
    # -------------------------
    paired_seed_rows: List[Dict[str, object]] = []

    # Build a map from (job, plugin_cfg_key) -> run_dir
    plugin_runs: Dict[Tuple[str, str], Path] = {}
    for job_name, plugin_cfg_key, run_dir in iter_plugin_runs(plugin_root):
        plugin_runs[(job_name, plugin_cfg_key)] = run_dir

    # For each plugin run, compute per-seed deltas vs default
    for (job_name, plugin_cfg_key), run_dir in sorted(plugin_runs.items()):
        # find seeds common to default and this plugin run
        plugin_seeds = {seed for seed, _sd in iter_seed_dirs(run_dir)}
        default_seeds = {seed for (jn, seed) in default_idx.keys() if jn == job_name}
        common_seeds = sorted(plugin_seeds & default_seeds)
        if not common_seeds:
            print(f"[warn] no common seeds for job={job_name} plugin={plugin_cfg_key}")
            continue

        for seed in common_seeds:
            def_gen, def_pod = default_idx[(job_name, seed)]
            plug_gen = run_dir / seed / GENERAL_STATS_FILENAME
            plug_pod = run_dir / seed / POD_STATS_FILENAME

            # Use common horizon for R/D/util comparisons
            def_metrics = metrics_cache.get(("default", job_name, seed))
            plu_metrics = metrics_cache.get((plugin_cfg_key, job_name, seed))
            if def_metrics is None or plu_metrics is None:
                # fall back: compute if missing
                def_metrics, def_lat = compute_run_metrics(def_gen, def_pod, eps_s=float(args.eps_s))
                plu_metrics, plu_lat = compute_run_metrics(plug_gen, plug_pod, eps_s=float(args.eps_s))
                latency_cache[("default", job_name, seed)] = def_lat
                latency_cache[(plugin_cfg_key, job_name, seed)] = plu_lat
                metrics_cache[("default", job_name, seed)] = def_metrics
                metrics_cache[(plugin_cfg_key, job_name, seed)] = plu_metrics

            T_common = min(float(def_metrics.get("T_end_s", 0.0)), float(plu_metrics.get("T_end_s", 0.0)))

            def_series = series_metrics_at_horizon(def_gen, T_common)
            plu_series = series_metrics_at_horizon(plug_gen, T_common)

            row: Dict[str, object] = {
                "job_name": job_name,
                "plugin_config": plugin_cfg_key,
                "seed": seed,
                "T_end_s": T_common,
            }

            # util
            row["delta_cpu_run_util_mean"] = float(plu_series.get("cpu_run_util_mean", 0.0)) - float(def_series.get("cpu_run_util_mean", 0.0))
            row["delta_mem_run_util_mean"] = float(plu_series.get("mem_run_util_mean", 0.0)) - float(def_series.get("mem_run_util_mean", 0.0))
            row["delta_cpu_req_util_mean"] = float(plu_series.get("cpu_req_util_mean", 0.0)) - float(def_series.get("cpu_req_util_mean", 0.0))
            row["delta_mem_req_util_mean"] = float(plu_series.get("mem_req_util_mean", 0.0)) - float(def_series.get("mem_req_util_mean", 0.0))

            row["delta_util_eff_run_mean"] = float(plu_series.get("util_eff_run_mean", 0.0)) - float(def_series.get("util_eff_run_mean", 0.0))
            row["delta_util_eff_req_mean"] = float(plu_series.get("util_eff_req_mean", 0.0)) - float(def_series.get("util_eff_req_mean", 0.0))

            # R/D per priority (only up to max_k_global for stable columns)
            for p in range(1, max_k_global + 1):
                R_def = float(def_series["R"].get(p, 0.0))
                R_plu = float(plu_series["R"].get(p, 0.0))
                D_def = float(def_series["D"].get(p, 0.0))
                D_plu = float(plu_series["D"].get(p, 0.0))
                row[f"delta_R_p{p}"] = R_plu - R_def
                row[f"delta_D_p{p}"] = D_plu - D_def

            row["delta_R_total"] = sum(float(row.get(f"delta_R_p{p}", 0.0)) for p in range(1, max_k_global + 1))
            row["delta_D_total"] = sum(float(row.get(f"delta_D_p{p}", 0.0)) for p in range(1, max_k_global + 1))

            # latency deltas over intersection of comparable replicas
            def_lat_map = latency_cache.get(("default", job_name, seed), {})
            plu_lat_map = latency_cache.get((plugin_cfg_key, job_name, seed), {})

            keys_all = sorted(set(def_lat_map.keys()) & set(plu_lat_map.keys()))
            if keys_all:
                def_vals = np.asarray([def_lat_map[k] for k in keys_all], dtype=float)
                plu_vals = np.asarray([plu_lat_map[k] for k in keys_all], dtype=float)
                row["delta_latency_mean_s"] = float(np.mean(plu_vals) - np.mean(def_vals))
            else:
                row["delta_latency_mean_s"] = float("nan")

            # per-priority latency deltas (mean)
            for p in range(1, max_k_global + 1):
                kset = [k for k in keys_all if k[0] == p]
                if not kset:
                    row[f"delta_latency_mean_p{p}_s"] = float("nan")
                    continue
                def_vals_p = np.asarray([def_lat_map[k] for k in kset], dtype=float)
                plu_vals_p = np.asarray([plu_lat_map[k] for k in kset], dtype=float)
                row[f"delta_latency_mean_p{p}_s"] = float(np.mean(plu_vals_p) - np.mean(def_vals_p))

            paired_seed_rows.append(row)

    df_pair_seed = pd.DataFrame(paired_seed_rows)

    if df_pair_seed.empty:
        # still write empty file with header
        df_pair_seed.to_csv(out_dir / "results_paired.csv", index=False)
    else:
        # aggregate across seeds
        delta_cols = [c for c in df_pair_seed.columns if c.startswith("delta_")]

        gcols = ["job_name", "plugin_config"]
        grp2 = df_pair_seed.groupby(gcols, dropna=False)

        df_pair_agg = grp2[delta_cols].agg(["mean", "std"]).reset_index()

        # flatten columns
        df_pair_agg.columns = [
            ("_".join([x for x in col if x]) if isinstance(col, tuple) else col)
            for col in df_pair_agg.columns
        ]

        # n_seed and T_end
        df_pair_agg.insert(2, "n_seed", grp2.size().to_numpy())
        df_pair_agg.insert(3, "T_end_s_mean", grp2["T_end_s"].mean().to_numpy())

        # build requested column order
        out_cols: List[str] = ["job_name", "plugin_config", "n_seed", "T_end_s_mean"]

        # util
        out_cols += [
            "delta_cpu_run_util_mean_mean", "delta_cpu_run_util_mean_std",
            "delta_mem_run_util_mean_mean", "delta_mem_run_util_mean_std",
            "delta_cpu_req_util_mean_mean", "delta_cpu_req_util_mean_std",
            "delta_mem_req_util_mean_mean", "delta_mem_req_util_mean_std",
            "delta_util_eff_run_mean_mean", "delta_util_eff_run_mean_std",
            "delta_util_eff_req_mean_mean", "delta_util_eff_req_mean_std",
        ]


        # R_p
        for p in range(1, max_k_global + 1):
            out_cols += [f"delta_R_p{p}_mean", f"delta_R_p{p}_std"]

        out_cols += ["delta_R_total_mean", "delta_R_total_std"]

        # D_p
        for p in range(1, max_k_global + 1):
            out_cols += [f"delta_D_p{p}_mean", f"delta_D_p{p}_std"]

        out_cols += ["delta_D_total_mean", "delta_D_total_std"]

        # latency overall
        out_cols += ["delta_latency_mean_s_mean", "delta_latency_mean_s_std"]

        # latency per priority (optional but requested)
        for p in range(1, max_k_global + 1):
            out_cols += [f"delta_latency_mean_p{p}_s_mean", f"delta_latency_mean_p{p}_s_std"]

        # ensure all exist
        for c in out_cols:
            if c not in df_pair_agg.columns:
                df_pair_agg[c] = np.nan

        df_pair_agg = df_pair_agg[out_cols]
        round_numeric_df(df_pair_agg, exclude=["job_name", "plugin_config", "n_seed"])
        df_pair_agg.to_csv(out_dir / "results_paired.csv", index=False)

    # -------------------------
    # Time-series mean files per (scheduler, job)
    # -------------------------

    # Map (scheduler, job_name) -> list of general_stats paths
    series_groups: Dict[Tuple[str, str], List[Path]] = {}

    # default series
    for (job_name, seed), (gen_p, _pod_p) in default_idx.items():
        series_groups.setdefault(("default", job_name), []).append(gen_p)

    # plugin series
    for job_name, plugin_cfg_key, run_dir in iter_plugin_runs(plugin_root):
        for seed, seed_dir in iter_seed_dirs(run_dir):
            series_groups.setdefault((plugin_cfg_key, job_name), []).append(seed_dir / GENERAL_STATS_FILENAME)

    for (scheduler, job_name), gen_paths in sorted(series_groups.items()):
        try:
            k_max = parse_test_combo(job_name).k_max
        except SystemExit:
            m = re.search(r"prio=(\d+)", job_name)
            k_max = int(m.group(1)) if m else max_k_global

        write_series_means(
            out_dir=out_dir,
            scheduler=scheduler,
            job_name=job_name,
            seed_general_paths=gen_paths,
            k_max=k_max,
            time_round=int(args.time_round),
        )

    print(f"Wrote outputs to: {out_dir}")

if __name__ == "__main__":
    main()
