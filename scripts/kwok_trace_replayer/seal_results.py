#!/usr/bin/env python3
"""scripts/kwok_trace_replayer/seal_results.py

python -m scripts.kwok_trace_replayer.seal_results --root analysis/kwok_trace_replayer --out-dir analysis/kwok_trace_replayer/sealed

Compute:
  1) results_paired.csv  (as before)
  2) series/ mean time-series CSVs for plotting:
       - series/default/<job_name>.csv
       - series/plugin/<plugin_config>__<job_name>.csv    (NO nested subdirs)

Series CSV schema:
  scheduler, job_name, n_seed, time_s, <metric>_mean...

Notes:
  - Default series are aggregated across all seeds found for that job.
  - Plugin series are aggregated across all seeds found for that plugin run dir.
  - We do not create per-config subdirectories under series/plugin/.
"""

import argparse, json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.kwok_trace_replayer.trace_helpers import rs_prefix_from_pod_name

# -----------------------------
# Numeric formatting
# -----------------------------

FLOAT_DECIMALS = 4

# -----------------------------
# Filenames / schema
# -----------------------------

GENERAL_STATS_FILENAME = "general_stats.csv"
POD_STATS_FILENAME = "pod_stats.csv"
OPT_STATS_FILENAME = "optimization_stats.json"

TIME_COL = "time_s"
CPU_RUN_COL = "cpu_run_util"
MEM_RUN_COL = "mem_run_util"
CPU_REQ_COL = "cpu_req_util"
MEM_REQ_COL = "mem_req_util"

RUNNING_PREFIX = "running_p"
DELETIONS_CUM_PREFIX = "deletions_cum_p"

POD_EVENT_COL = "event"
POD_NAME_COL = "pod_name"
POD_UID_COL = "pod_uid"
POD_PRIO_COL = "priority"
POD_TIME_COL = "time_s"

# We always write p1..p4 columns (even if a job has fewer priority tiers)
MAX_K_OUT = 4

OPT_STATS_KEYS_TOTAL: List[str] = [
    "solver_attempts_total",
    "best_solver_optimal_total",
    "best_solver_feasible_total",
    "best_solver_failed_total",
    "plan_not_applicable_total",
    "plan_activated_total",
]

# latency key stable across runs
LatencyKey = Tuple[int, str, int]  # (priority, rs_prefix, replica_index_in_first_batch)

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
        b = "1" if self.blocking else "0"
        d = "1" if self.defpreempt else "0"
        return f"mode={self.mode_raw}_blocking={b}_defpreempt={d}"


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write results_paired.csv + series/ time-series from KWOK replayer outputs.")
    p.add_argument("--root", required=True, help="Root directory containing 'default/' and 'plugin/' subfolders.")
    p.add_argument("--out-dir", default=None, help="Where to write outputs (default: <root>/sealed).")
    p.add_argument(
        "--eps-s",
        type=float,
        default=1.0,
        help="Epsilon window (seconds) for the first-batch latency heuristic (default: 1.0).",
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

    test_combo = TestCombo(
        job_name=canon_job_name(n, kmax, arr),
        n_nodes=n,
        k_max=kmax,
        mean_arrival_s=arr,
    )

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


def sanitize_filename(s: str) -> str:
    # keep = and _ and . and -; replace everything else
    return re.sub(r"[^A-Za-z0-9._=\-]+", "_", str(s))


# -----------------------------
# IO helpers
# -----------------------------


def read_opt_stats_plugin(opt_path: Path) -> Dict[str, float]:
    """Read optimization_stats.json; missing file/keys -> NaN."""
    out: Dict[str, float] = {k: float("nan") for k in OPT_STATS_KEYS_TOTAL}
    if not opt_path.exists():
        return out
    try:
        with opt_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for k in OPT_STATS_KEYS_TOTAL:
            if k in data:
                try:
                    out[k] = float(data[k])
                except Exception:
                    out[k] = float("nan")
    except Exception:
        return out
    return out


def round_numeric_df(df: pd.DataFrame, *, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    if exclude is None:
        exclude = []
    num_cols = [c for c in df.columns if c not in set(exclude) and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].round(FLOAT_DECIMALS)
    return df


# -----------------------------
# Metric computation (horizon-aware)
# -----------------------------


def step_values_at(t: np.ndarray, y: np.ndarray, grid: np.ndarray, *, default: float = 0.0) -> np.ndarray:
    """Stepwise-constant y(t) at each time in grid, using last observation carried forward."""
    if t.size == 0 or y.size == 0:
        return np.full_like(grid, fill_value=float(default), dtype=float)
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.full(grid.shape, float(default), dtype=float)
    ok = idx >= 0
    idx2 = np.clip(idx, 0, y.size - 1)
    out[ok] = y[idx2[ok]]
    return out


def series_metrics_at_horizon(general_csv: Path, horizon_s: float) -> Dict[str, object]:
    """
    Horizon-aware metrics using stepwise carry-forward.

    Returns:
      cpu_run_util_mean, mem_run_util_mean, util_eff_run_mean,
      cpu_req_util_mean, mem_req_util_mean, util_eff_req_mean,
      R (dict): mean running pods per priority over the horizon
      D (dict): cumulative deletions at horizon
    """
    df = pd.read_csv(general_csv)
    require_cols(df, general_csv, [TIME_COL, CPU_RUN_COL, MEM_RUN_COL, CPU_REQ_COL, MEM_REQ_COL])

    t_raw = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    if t_raw.size == 0 or not math.isfinite(float(horizon_s)) or float(horizon_s) <= 0.0:
        return {
            "R": {},
            "D": {},
            "cpu_run_util_mean": float("nan"),
            "mem_run_util_mean": float("nan"),
            "util_eff_run_mean": float("nan"),
            "cpu_req_util_mean": float("nan"),
            "mem_req_util_mean": float("nan"),
            "util_eff_req_mean": float("nan"),
        }

    order = np.argsort(t_raw)
    t = t_raw[order]
    t0 = float(t[0])
    t = t - t0

    mask = t <= float(horizon_s)
    t_clip = t[mask]
    if t_clip.size == 0:
        t_clip = np.array([float(horizon_s)], dtype=float)
    elif t_clip[-1] < float(horizon_s):
        t_clip = np.concatenate([t_clip, np.array([float(horizon_s)], dtype=float)])

    dt = np.diff(t_clip, prepend=t_clip[0])
    dt = np.clip(dt, 0.0, None)

    cpu_run = pd.to_numeric(df[CPU_RUN_COL], errors="coerce").to_numpy(dtype=float)[order]
    mem_run = pd.to_numeric(df[MEM_RUN_COL], errors="coerce").to_numpy(dtype=float)[order]
    cpu_req = pd.to_numeric(df[CPU_REQ_COL], errors="coerce").to_numpy(dtype=float)[order]
    mem_req = pd.to_numeric(df[MEM_REQ_COL], errors="coerce").to_numpy(dtype=float)[order]

    eff_run = np.maximum(cpu_run, mem_run)
    eff_req = np.maximum(cpu_req, mem_req)

    cpu_run_clip = step_values_at(t, np.nan_to_num(cpu_run, nan=0.0), t_clip, default=0.0)
    mem_run_clip = step_values_at(t, np.nan_to_num(mem_run, nan=0.0), t_clip, default=0.0)
    cpu_req_clip = step_values_at(t, np.nan_to_num(cpu_req, nan=0.0), t_clip, default=0.0)
    mem_req_clip = step_values_at(t, np.nan_to_num(mem_req, nan=0.0), t_clip, default=0.0)
    eff_run_clip = step_values_at(t, np.nan_to_num(eff_run, nan=0.0), t_clip, default=0.0)
    eff_req_clip = step_values_at(t, np.nan_to_num(eff_req, nan=0.0), t_clip, default=0.0)

    H = float(horizon_s)

    def _time_mean_clip(y_clip: np.ndarray) -> float:
        return float(np.sum(y_clip * dt) / H) if H > 0 else float("nan")

    cpu_run_mean = _time_mean_clip(cpu_run_clip)
    mem_run_mean = _time_mean_clip(mem_run_clip)
    cpu_req_mean = _time_mean_clip(cpu_req_clip)
    mem_req_mean = _time_mean_clip(mem_req_clip)
    util_eff_run_mean = _time_mean_clip(eff_run_clip)
    util_eff_req_mean = _time_mean_clip(eff_req_clip)

    R: Dict[int, float] = {}  # now: MEAN running pods over horizon, not pod-seconds
    D: Dict[int, float] = {}

    for p in range(1, MAX_K_OUT + 1):
        run_col = f"{RUNNING_PREFIX}{p}"
        if run_col in df.columns:
            y_run = pd.to_numeric(df[run_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)[order]
            y_vals = step_values_at(t, y_run, t_clip, default=0.0)

            # mean number of running pods over [0, H]
            R[p] = float(np.sum(y_vals * dt) / H) if H > 0 else float("nan")
        else:
            R[p] = 0.0

        del_col = f"{DELETIONS_CUM_PREFIX}{p}"
        if del_col in df.columns:
            y_del = pd.to_numeric(df[del_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)[order]
            D[p] = float(step_values_at(t, y_del, np.array([H], dtype=float), default=0.0)[0])
        else:
            D[p] = 0.0

    return {
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
# Latency: first-batch map and deltas
# -----------------------------


def latency_map_first_batch(pod_df: pd.DataFrame, *, eps_s: float) -> Dict[LatencyKey, float]:
    """Latency map keyed by (priority, rs_prefix, replica_index_in_first_batch)."""
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


def latency_deltas_for_seed(def_pod_csv: Path, plu_pod_csv: Path, *, eps_s: float) -> Dict[str, float]:
    """Return dict with delta_latency_s_p{1..4} and delta_latency_s_total (plugin-default)."""
    df_def = pd.read_csv(def_pod_csv)
    df_plu = pd.read_csv(plu_pod_csv)
    require_cols(df_def, def_pod_csv, [POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL])
    require_cols(df_plu, plu_pod_csv, [POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL])

    for df in (df_def, df_plu):
        df[POD_TIME_COL] = pd.to_numeric(df[POD_TIME_COL], errors="coerce")
        df[POD_PRIO_COL] = pd.to_numeric(df[POD_PRIO_COL], errors="coerce").astype("Int64")
        df[POD_EVENT_COL] = df[POD_EVENT_COL].astype(str)
        df[POD_NAME_COL] = df[POD_NAME_COL].astype(str)
        df[POD_UID_COL] = df[POD_UID_COL].astype(str)

    def_map = latency_map_first_batch(df_def, eps_s=float(eps_s))
    plu_map = latency_map_first_batch(df_plu, eps_s=float(eps_s))

    keys_all = list(set(def_map.keys()) & set(plu_map.keys()))

    out: Dict[str, float] = {}
    if not keys_all:
        out["delta_latency_s_total"] = float("nan")
        for p in range(1, MAX_K_OUT + 1):
            out[f"delta_latency_s_p{p}"] = float("nan")
        return out

    diffs_all = np.asarray([float(plu_map[k]) - float(def_map[k]) for k in keys_all], dtype=float)
    out["delta_latency_s_total"] = float(np.mean(diffs_all)) if diffs_all.size else float("nan")

    for p in range(1, MAX_K_OUT + 1):
        kset = [k for k in keys_all if int(k[0]) == int(p)]
        if not kset:
            out[f"delta_latency_s_p{p}"] = float("nan")
            continue
        diffs_p = np.asarray([float(plu_map[k]) - float(def_map[k]) for k in kset], dtype=float)
        out[f"delta_latency_s_p{p}"] = float(np.mean(diffs_p)) if diffs_p.size else float("nan")

    return out


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
    """(job_name, seed) -> (general_csv, pod_csv)."""
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
    """Yield (job_name, plugin_config_key, run_dir_path)."""
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
# Output schema helpers
# -----------------------------


def out_columns_exact() -> List[str]:
    return [
        "job_name",
        "plugin_config",
        "n_seed",
        "T_end_s_mean",
        "delta_cpu_run_util_mean",
        "delta_mem_run_util_mean",
        "delta_util_eff_run_mean",
        "cpu_req_util_mean",
        "mem_req_util_mean",
        "util_eff_req_mean",
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
        "solver_attempts_total_mean",
        "best_solver_optimal_total_mean",
        "best_solver_feasible_total_mean",
        "best_solver_failed_total_mean",
        "plan_not_applicable_total_mean",
        "plan_activated_total_mean",
    ]


# -----------------------------
# Series writer (NEW layout)
# -----------------------------


def _read_and_normalize_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_cols(df, path, [TIME_COL])
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).copy()
    if df.empty:
        return df
    df = df.sort_values(TIME_COL).copy()
    t0 = float(df[TIME_COL].iloc[0])
    df[TIME_COL] = df[TIME_COL] - t0
    # reduce float noise so we can group across seeds robustly
    df[TIME_COL] = df[TIME_COL].round(6)
    return df


def _mean_series_across_seeds(general_paths: List[Path]) -> pd.DataFrame:
    """Return DF with time_s and <metric>_mean columns (mean across seeds)."""
    frames: List[pd.DataFrame] = []
    for p in general_paths:
        try:
            dfi = _read_and_normalize_series(p)
        except Exception:
            continue
        if dfi.empty:
            continue
        frames.append(dfi)

    if not frames:
        return pd.DataFrame(columns=[TIME_COL])

    df_all = pd.concat(frames, axis=0, ignore_index=True)

    # Keep only numeric metric columns + time
    metric_cols: List[str] = []
    for c in df_all.columns:
        if c == TIME_COL:
            continue
        if pd.api.types.is_numeric_dtype(df_all[c]):
            metric_cols.append(c)

    if not metric_cols:
        return df_all[[TIME_COL]].drop_duplicates().sort_values(TIME_COL)

    # group by time and compute mean for each metric
    g = df_all.groupby(TIME_COL, dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    g = g.sort_values(TIME_COL)

    # rename metric columns -> <metric>_mean
    rename = {c: f"{c}_mean" for c in metric_cols}
    g = g.rename(columns=rename)

    # effective utilisation (dominant resource)
    if "cpu_run_util_mean" in g.columns and "mem_run_util_mean" in g.columns:
        g["util_eff_run_mean"] = g[["cpu_run_util_mean", "mem_run_util_mean"]].max(axis=1)

    if "cpu_req_util_mean" in g.columns and "mem_req_util_mean" in g.columns:
        g["util_eff_req_mean"] = g[["cpu_req_util_mean", "mem_req_util_mean"]].max(axis=1)

    # total running pods (mean across seeds; stepwise values)
    run_cols = [f"running_p{p}_mean" for p in range(1, MAX_K_OUT + 1) if f"running_p{p}_mean" in g.columns]
    if run_cols:
        g["running_total_mean"] = g[run_cols].sum(axis=1)

    # total cumulative deletions (still stepwise, just summed)
    del_cols = [f"deletions_cum_p{p}_mean" for p in range(1, MAX_K_OUT + 1) if f"deletions_cum_p{p}_mean" in g.columns]
    if del_cols:
        g["deletions_cum_total_mean"] = g[del_cols].sum(axis=1)

    return g


def write_series_files(*, default_root: Path, plugin_root: Path, out_dir: Path) -> None:
    """Write series/default/*.csv and series/plugin/*.csv with NEW layout."""
    series_root = out_dir / "series"
    default_out = series_root / "default"
    plugin_out = series_root / "plugin"
    default_out.mkdir(parents=True, exist_ok=True)
    plugin_out.mkdir(parents=True, exist_ok=True)

    # ---- default: per job_name across all seeds ----
    for job_dir in sorted(default_root.iterdir()):
        if not job_dir.is_dir():
            continue
        try:
            r = parse_test_combo(job_dir.name)
        except SystemExit:
            continue

        seed_general_paths: List[Path] = []
        for _seed, seed_dir in iter_seed_dirs(job_dir):
            seed_general_paths.append(seed_dir / GENERAL_STATS_FILENAME)

        df_mean = _mean_series_across_seeds(seed_general_paths)
        if df_mean.empty:
            continue

        # time_s already exists
        df_mean["time_s"] = df_mean["time_s"].astype(float)

        # add metadata columns
        df_mean.insert(0, "n_seed", int(len(seed_general_paths)))
        df_mean.insert(0, "job_name", r.job_name)
        df_mean.insert(0, "scheduler", "default")

        out_path = default_out / f"{sanitize_filename(r.job_name)}.csv"
        round_numeric_df(df_mean, exclude=["scheduler", "job_name"])
        df_mean.to_csv(out_path, index=False)

    # ---- plugin: per (plugin_config, job_name) across all seeds in run dir ----
    for run_dir in sorted(plugin_root.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            regime, cfg = parse_plugin_dirname(run_dir.name)
        except SystemExit:
            continue

        seed_general_paths: List[Path] = []
        for _seed, seed_dir in iter_seed_dirs(run_dir):
            seed_general_paths.append(seed_dir / GENERAL_STATS_FILENAME)

        df_mean = _mean_series_across_seeds(seed_general_paths)
        if df_mean.empty:
            continue

        sched_key = cfg.key()

        # time_s already exists
        df_mean["time_s"] = df_mean["time_s"].astype(float)

        # add metadata columns
        df_mean.insert(0, "n_seed", int(len(seed_general_paths)))
        df_mean.insert(0, "job_name", regime.job_name)
        df_mean.insert(0, "scheduler", sched_key)

        # encode scheduler in filename
        out_name = f"{sanitize_filename(sched_key)}__{sanitize_filename(regime.job_name)}.csv"
        out_path = plugin_out / out_name
        round_numeric_df(df_mean, exclude=["scheduler", "job_name"])
        df_mean.to_csv(out_path, index=False)


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
    
    print("This may take a while...")

    # write series files with flat plugin layout
    write_series_files(default_root=default_root, plugin_root=plugin_root, out_dir=out_dir)

    default_idx = scan_default(default_root)

    # Map (job_name, plugin_cfg_key) -> run_dir
    plugin_runs: Dict[Tuple[str, str], Path] = {}
    for job_name, plugin_cfg_key, run_dir in iter_plugin_runs(plugin_root):
        plugin_runs[(job_name, plugin_cfg_key)] = run_dir

    seed_rows: List[Dict[str, object]] = []

    for (job_name, plugin_cfg_key), run_dir in sorted(plugin_runs.items()):
        plugin_seeds = {seed for seed, _sd in iter_seed_dirs(run_dir)}
        default_seeds = {seed for (jn, seed) in default_idx.keys() if jn == job_name}
        common_seeds = sorted(plugin_seeds & default_seeds)
        if not common_seeds:
            print(f"[warn] no common seeds for job={job_name} plugin={plugin_cfg_key}")
            continue

        for seed in common_seeds:
            def_gen, def_pod = default_idx[(job_name, seed)]
            plu_seed_dir = run_dir / seed
            plu_gen = plu_seed_dir / GENERAL_STATS_FILENAME
            plu_pod = plu_seed_dir / POD_STATS_FILENAME
            plu_opt = plu_seed_dir / OPT_STATS_FILENAME

            # Common horizon = min(T_end_default, T_end_plugin) based on time_s span
            df_def_t = pd.read_csv(def_gen, usecols=[TIME_COL])
            df_plu_t = pd.read_csv(plu_gen, usecols=[TIME_COL])
            require_cols(df_def_t, def_gen, [TIME_COL])
            require_cols(df_plu_t, plu_gen, [TIME_COL])

            t_def_raw = pd.to_numeric(df_def_t[TIME_COL], errors="coerce").to_numpy(dtype=float)
            t_plu_raw = pd.to_numeric(df_plu_t[TIME_COL], errors="coerce").to_numpy(dtype=float)
            if t_def_raw.size == 0 or t_plu_raw.size == 0:
                continue

            T_def = float(np.nanmax(t_def_raw) - np.nanmin(t_def_raw))
            T_plu = float(np.nanmax(t_plu_raw) - np.nanmin(t_plu_raw))
            if not math.isfinite(T_def) or not math.isfinite(T_plu) or T_def <= 0.0 or T_plu <= 0.0:
                continue

            T_common = float(min(T_def, T_plu))
            if T_common <= 0.0:
                continue

            def_series = series_metrics_at_horizon(def_gen, T_common)
            plu_series = series_metrics_at_horizon(plu_gen, T_common)

            # Run-util deltas (plugin - default)
            d_cpu_run = float(plu_series["cpu_run_util_mean"]) - float(def_series["cpu_run_util_mean"])
            d_mem_run = float(plu_series["mem_run_util_mean"]) - float(def_series["mem_run_util_mean"])
            d_eff_run = float(plu_series["util_eff_run_mean"]) - float(def_series["util_eff_run_mean"])

            # Req-util means (DEFAULT)
            cpu_req_mean = float(def_series["cpu_req_util_mean"])
            mem_req_mean = float(def_series["mem_req_util_mean"])
            eff_req_mean = float(def_series["util_eff_req_mean"])

            # R/D deltas and totals (p1..p4)
            dR: Dict[int, float] = {}
            dD: Dict[int, float] = {}
            for p in range(1, MAX_K_OUT + 1):
                dR[p] = float(plu_series["R"].get(p, 0.0)) - float(def_series["R"].get(p, 0.0))
                dD[p] = float(plu_series["D"].get(p, 0.0)) - float(def_series["D"].get(p, 0.0))
            dR_total = float(sum(dR.values()))
            dD_total = float(sum(dD.values()))

            # Latency deltas
            lat = latency_deltas_for_seed(def_pod, plu_pod, eps_s=float(args.eps_s))

            # Optimization stats (PLUGIN as-is)
            opt = read_opt_stats_plugin(plu_opt)

            seed_rows.append(
                {
                    "job_name": job_name,
                    "plugin_config": plugin_cfg_key,
                    "seed": seed,
                    "T_end_s": T_common,
                    "delta_cpu_run_util_mean": d_cpu_run,
                    "delta_mem_run_util_mean": d_mem_run,
                    "delta_util_eff_run_mean": d_eff_run,
                    "cpu_req_util_mean": cpu_req_mean,
                    "mem_req_util_mean": mem_req_mean,
                    "util_eff_req_mean": eff_req_mean,
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
                    "delta_latency_s_p1": float(lat["delta_latency_s_p1"]),
                    "delta_latency_s_p2": float(lat["delta_latency_s_p2"]),
                    "delta_latency_s_p3": float(lat["delta_latency_s_p3"]),
                    "delta_latency_s_p4": float(lat["delta_latency_s_p4"]),
                    "delta_latency_s_total": float(lat["delta_latency_s_total"]),
                    "solver_attempts_total": float(opt["solver_attempts_total"]),
                    "best_solver_optimal_total": float(opt["best_solver_optimal_total"]),
                    "best_solver_feasible_total": float(opt["best_solver_feasible_total"]),
                    "best_solver_failed_total": float(opt["best_solver_failed_total"]),
                    "plan_not_applicable_total": float(opt["plan_not_applicable_total"]),
                    "plan_activated_total": float(opt["plan_activated_total"]),
                }
            )

    df_seed = pd.DataFrame(seed_rows)
    if df_seed.empty:
        pd.DataFrame(columns=out_columns_exact()).to_csv(out_dir / "results_paired.csv", index=False)
        print(f"Wrote outputs to: {out_dir}")
        return

    # Aggregate across seeds (mean-only)
    grp = df_seed.groupby(["job_name", "plugin_config"], dropna=False)

    agg = grp.mean(numeric_only=True).reset_index()
    agg.insert(2, "n_seed", grp.size().to_numpy())

    # Rename into requested output names
    rename_map = {
        "T_end_s": "T_end_s_mean",
        "delta_cpu_run_util_mean": "delta_cpu_run_util_mean",
        "delta_mem_run_util_mean": "delta_mem_run_util_mean",
        "delta_util_eff_run_mean": "delta_util_eff_run_mean",
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
        "solver_attempts_total": "solver_attempts_total_mean",
        "best_solver_optimal_total": "best_solver_optimal_total_mean",
        "best_solver_feasible_total": "best_solver_feasible_total_mean",
        "best_solver_failed_total": "best_solver_failed_total_mean",
        "plan_not_applicable_total": "plan_not_applicable_total_mean",
        "plan_activated_total": "plan_activated_total_mean",
    }
    agg = agg.rename(columns=rename_map)

    # Ensure all requested columns exist
    for c in out_columns_exact():
        if c not in agg.columns:
            agg[c] = np.nan

    agg = agg[out_columns_exact()]
    round_numeric_df(agg, exclude=["job_name", "plugin_config", "n_seed"])
    agg.to_csv(out_dir / "results_paired.csv", index=False)

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
