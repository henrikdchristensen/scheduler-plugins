#!/usr/bin/env python3
"""scripts/kwok_trace_replayer/aggregate_tables.py

Aggregate KWOK trace replayer CSV outputs across seeds and emit:
  1) results_long.csv   : one row per (run, seed) with raw metrics
  2) results_paired.csv : one row per (plugin run, seed) with paired deltas vs default
  3) results_agg.csv    : mean/std/count across seeds per (regime, plugin-config)
  4) tables/*.csv       : table-ready wide CSVs

Input layout (under --root):
  default/<regime>/<seed>/{general_stats.csv,pod_stats.csv}
  plugin/<regime_mode...>/<seed>/{general_stats.csv,pod_stats.csv}

Matching rule:
  plugin dir name is parsed as:
    <base_regime>_mode<mode+params>_<blocking|nonblocking>[_withdefaultpreemption]
  and is matched to baseline directory <base_regime> under default/.

Metrics (raw per run):
  - util_eff_mean: time-weighted mean of max(cpu_run_util, mem_run_util)
  - R_p(T): cumulative running pod-seconds per priority p (integral of running_p{p})
  - D_p(T): cumulative deletions per priority p at T (deletions_cum_p{p})
  - latency_first_admit_mean_s / p95: time-to-first-admit using the rs-prefix heuristic

Paired comparison:
  For each (plugin run, seed), compute deltas vs default at a common horizon
    T_common = min(T_end_default, T_end_plugin)
  using stepwise-carry-forward to avoid bias when end timestamps differ.

Exports:
  - results_paired.csv includes per-priority deltas:
      delta_R_p{p}, delta_D_p{p}
    plus "improvement" versions where positive = better:
      impr_R_p{p} = delta_R_p{p}
      impr_D_p{p} = -(delta_D_p{p})

  - tables include both numeric per-priority columns and a LaTeX-friendly
    "cell macro" column that packs per-priority values into one cell:
      k_max=1 : "+0.9" (single value)
      k_max=4 : "\\prioDeltaFour{+3.8}{-0.2}{+0.0}{-1.1}"

"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Filenames / schema
# -----------------------------
GENERAL_STATS_FILENAME = "general_stats.csv"
POD_STATS_FILENAME = "pod_stats.csv"

TIME_COL = "time_s"
CPU_COL = "cpu_run_util"
MEM_COL = "mem_run_util"

RUNNING_PREFIX = "running_p"
DELETIONS_CUM_PREFIX = "deletions_cum_p"

POD_EVENT_COL = "event"
POD_NAME_COL = "pod_name"
POD_UID_COL = "pod_uid"
POD_PRIO_COL = "priority"
POD_TIME_COL = "time_s"

_RS_PREFIX_RE = re.compile(r"^(rs-\d{6})(?:-.*)?$")

# Regime directory name parsing: nodes16_prio4_arrival8 (arrival may be float)
_REGIME_RE = re.compile(r"nodes(?P<n>\d+)_prio(?P<kmax>\d+)_arrival(?P<arr>\d+(?:\.\d+)?)")


# -----------------------------
# Types
# -----------------------------
@dataclass(frozen=True)
class Regime:
    regime_key: str
    n_nodes: int
    k_max: int
    mean_arrival_s: float


@dataclass(frozen=True)
class PluginConfig:
    mode: str  # scheduling-failure | periodic | stable-queue | unknown
    param_s: Optional[float]
    enforcement: str  # blocking | nonblocking | unknown
    with_default_preemption: bool

    def row_label(self) -> str:
        mode_title = {
            "scheduling-failure": "Scheduling-failure",
            "periodic": "Periodic",
            "stable-queue": "Stable-queue",
            "unknown": "Unknown",
        }.get(self.mode, self.mode)

        enf = {
            "blocking": "blk",
            "nonblocking": "non-blk",
            "unknown": "?",
        }.get(self.enforcement, self.enforcement)

        if self.with_default_preemption:
            return f"{mode_title} ({enf}, defpreempt)"
        return f"{mode_title} ({enf})"


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate KWOK replayer outputs across seeds and export table-ready CSVs."
    )
    p.add_argument(
        "--root",
        required=True,
        help="Root directory containing 'default/' and 'plugin/' subfolders.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write CSV outputs (default: <root>/tables_out).",
    )
    p.add_argument(
        "--eps-s",
        type=float,
        default=1.0,
        help="Epsilon window (seconds) for the first-admit latency heuristic (default: 1.0).",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Decimals for LaTeX cell formatting (default: 1).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce warnings about missing matches.",
    )
    return p.parse_args()


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_regime(regime_key: str) -> Regime:
    m = _REGIME_RE.search(regime_key)
    if not m:
        raise SystemExit(
            f"Could not parse regime from directory name: '{regime_key}'. "
            "Expected something like 'nodes16_prio4_arrival8'."
        )
    n = int(m.group("n"))
    k = int(m.group("kmax"))
    arr = float(m.group("arr"))
    return Regime(regime_key=regime_key, n_nodes=n, k_max=k, mean_arrival_s=arr)


def parse_plugin_dirname(plugin_dirname: str) -> Tuple[str, PluginConfig]:
    """Return (base_regime_key, PluginConfig) from a plugin run directory name."""

    with_defpreempt = plugin_dirname.endswith("_withdefaultpreemption")
    name = plugin_dirname[:-len("_withdefaultpreemption")] if with_defpreempt else plugin_dirname

    if "_mode" not in name:
        return name, PluginConfig(
            mode="unknown",
            param_s=None,
            enforcement="unknown",
            with_default_preemption=with_defpreempt,
        )

    base, mode_part = name.split("_mode", 1)

    enforcement = "unknown"
    if mode_part.endswith("_blocking"):
        enforcement = "blocking"
        core = mode_part[: -len("_blocking")]
    elif mode_part.endswith("_nonblocking"):
        enforcement = "nonblocking"
        core = mode_part[: -len("_nonblocking")]
    else:
        core = mode_part

    mode = "unknown"
    param: Optional[float] = None

    if core.startswith("periodic"):
        mode = "periodic"
        tail = core[len("periodic") :]
        if tail:
            try:
                param = float(tail)
            except ValueError:
                param = None
    elif core.startswith("stablequeue"):
        mode = "stable-queue"
        tail = core[len("stablequeue") :]
        if tail:
            try:
                param = float(tail)
            except ValueError:
                param = None
    elif core.startswith("schedulingfailure"):
        mode = "scheduling-failure"
        param = None

    return base, PluginConfig(
        mode=mode,
        param_s=param,
        enforcement=enforcement,
        with_default_preemption=with_defpreempt,
    )


def rs_prefix_from_pod_name(pod_name: str) -> str:
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


def _regime_fields(regime_key: str) -> Dict[str, object]:
    r = parse_regime(regime_key)
    return {"n_nodes": r.n_nodes, "k_max": r.k_max, "mean_arrival_s": r.mean_arrival_s}


# -----------------------------
# Metric computation
# -----------------------------

def _sorted_time_and_dt(df: pd.DataFrame, time_col: str = TIME_COL) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, dt, order) where t is sorted and normalized to start at 0."""
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


def _step_value_at(t: np.ndarray, y: np.ndarray, query_t: float) -> float:
    """Return stepwise-constant y(t) at query_t using last observation carried forward."""
    if t.size == 0 or y.size == 0:
        return 0.0
    idx = np.searchsorted(t, query_t, side="right") - 1
    if idx < 0:
        return 0.0
    if idx >= y.size:
        idx = y.size - 1
    return float(y[idx])


def _first_admit_latencies(pod_df: pd.DataFrame, *, eps_s: float = 1.0) -> np.ndarray:
    """Vector of first-admit latencies (seconds) using the rs-prefix heuristic."""

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

    out: List[float] = []

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
            if pd.isna(rt) or pd.isna(at):
                continue
            lat = float(rt) - float(at)
            if math.isfinite(lat) and lat >= 0.0:
                out.append(lat)

    return np.asarray(out, dtype=float)


def compute_metrics_for_run(general_path: Path, pod_path: Path, *, eps_s: float) -> Dict[str, float]:
    """Compute raw per-run metrics from CSV files."""

    df_g = pd.read_csv(general_path)
    _require_cols(df_g, general_path, [TIME_COL, CPU_COL, MEM_COL])

    t, dt, order = _sorted_time_and_dt(df_g, TIME_COL)
    T_end = float(t[-1]) if t.size else 0.0

    cpu = df_g[CPU_COL].astype(float).to_numpy()[order] if t.size else np.array([], dtype=float)
    mem = df_g[MEM_COL].astype(float).to_numpy()[order] if t.size else np.array([], dtype=float)
    eff = np.maximum(cpu, mem) if t.size else np.array([], dtype=float)

    util_eff_mean = float(np.sum(eff * dt) / T_end) if (t.size and T_end > 0) else 0.0

    prios: List[int] = []
    for c in df_g.columns:
        if c.startswith(RUNNING_PREFIX):
            try:
                prios.append(int(c[len(RUNNING_PREFIX) :]))
            except ValueError:
                pass
    prios = sorted(set(prios))
    k_max = max(prios) if prios else 0

    metrics: Dict[str, float] = {}
    metrics["T_end_s"] = T_end
    metrics["util_eff_mean"] = util_eff_mean
    metrics["k_max_detected"] = float(k_max)

    for p in prios:
        y = df_g[f"{RUNNING_PREFIX}{p}"].fillna(0).astype(float).to_numpy()[order]
        metrics[f"R_p{p}"] = float(np.sum(y * dt))

    for p in prios:
        col = f"{DELETIONS_CUM_PREFIX}{p}"
        if col not in df_g.columns:
            continue
        y = df_g[col].fillna(0).astype(float).to_numpy()[order]
        metrics[f"D_p{p}"] = float(y[-1]) if y.size else 0.0

    df_p = pd.read_csv(pod_path)
    _require_cols(df_p, pod_path, [POD_EVENT_COL, POD_NAME_COL, POD_UID_COL, POD_PRIO_COL, POD_TIME_COL])

    df_p[POD_TIME_COL] = df_p[POD_TIME_COL].astype(float)
    df_p[POD_PRIO_COL] = df_p[POD_PRIO_COL].astype(int)
    df_p[POD_EVENT_COL] = df_p[POD_EVENT_COL].astype(str)
    df_p[POD_NAME_COL] = df_p[POD_NAME_COL].astype(str)
    df_p[POD_UID_COL] = df_p[POD_UID_COL].astype(str)

    lat = _first_admit_latencies(df_p, eps_s=eps_s)
    if lat.size:
        metrics["latency_first_admit_mean_s"] = float(np.mean(lat))
        metrics["latency_first_admit_p95_s"] = float(np.quantile(lat, 0.95))
        metrics["latency_first_admit_n"] = float(lat.size)
    else:
        metrics["latency_first_admit_mean_s"] = float("nan")
        metrics["latency_first_admit_p95_s"] = float("nan")
        metrics["latency_first_admit_n"] = 0.0

    return metrics


# -----------------------------
# Horizon-aware series metrics
# -----------------------------

def _series_metrics_at_horizon(general_csv: Path, horizon_s: float) -> Dict[str, object]:
    """Compute horizon-aware metrics using stepwise carry-forward."""

    df = pd.read_csv(general_csv)
    _require_cols(df, general_csv, [TIME_COL, CPU_COL, MEM_COL])

    t_raw = df[TIME_COL].astype(float).to_numpy()
    if t_raw.size == 0:
        return {"prios": [], "R": {}, "D": {}, "util_eff_mean": 0.0}

    order = np.argsort(t_raw)
    t = t_raw[order]
    t0 = float(t[0])
    t = t - t0

    mask = t <= horizon_s
    t_clip = t[mask]
    if t_clip.size == 0:
        t_clip = np.array([horizon_s], dtype=float)
    elif t_clip[-1] < horizon_s:
        t_clip = np.concatenate([t_clip, np.array([horizon_s], dtype=float)])

    dt = np.diff(t_clip, prepend=t_clip[0])
    dt = np.clip(dt, 0.0, None)

    cpu = df[CPU_COL].astype(float).to_numpy()[order]
    mem = df[MEM_COL].astype(float).to_numpy()[order]
    eff = np.maximum(cpu, mem)
    eff_clip = np.array([_step_value_at(t, eff, float(tt)) for tt in t_clip], dtype=float)
    util_eff_mean = float(np.sum(eff_clip * dt) / float(horizon_s)) if horizon_s > 0 else 0.0

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
        y_vals = np.array([_step_value_at(t, y_run, float(tt)) for tt in t_clip], dtype=float)
        R[p] = float(np.sum(y_vals * dt))

        del_col = f"{DELETIONS_CUM_PREFIX}{p}"
        if del_col in df.columns:
            y_del = df[del_col].fillna(0).astype(float).to_numpy()[order]
            D[p] = float(_step_value_at(t, y_del, float(horizon_s)))

    return {"prios": prios, "R": R, "D": D, "util_eff_mean": util_eff_mean}


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


def build_default_index(default_root: Path) -> Dict[Tuple[str, str], Tuple[Path, Path]]:
    idx: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
    for regime_dir in sorted(default_root.iterdir()):
        if not regime_dir.is_dir():
            continue
        regime_key = regime_dir.name
        try:
            parse_regime(regime_key)
        except SystemExit:
            continue
        for seed, seed_dir in iter_seed_dirs(regime_dir):
            idx[(regime_key, seed)] = (seed_dir / GENERAL_STATS_FILENAME, seed_dir / POD_STATS_FILENAME)
    return idx


def iter_plugin_runs(plugin_root: Path) -> Iterable[Tuple[str, PluginConfig, str, Path]]:
    if not plugin_root.exists() or not plugin_root.is_dir():
        return
    for run_dir in sorted(plugin_root.iterdir()):
        if not run_dir.is_dir():
            continue
        base, cfg = parse_plugin_dirname(run_dir.name)
        try:
            parse_regime(base)
        except SystemExit:
            continue
        yield base, cfg, run_dir.name, run_dir


# -----------------------------
# Table helpers
# -----------------------------

def _fmt_arrival(x: object) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def _col_sort_key(c: str) -> Tuple[int, float]:
    m = re.match(r"N(\d+)_muA(.+)$", c)
    if not m:
        return (10**9, 10**9)
    return (int(m.group(1)), float(m.group(2)))


def _row_sort_key(mode: str, enforcement: str, with_defpreempt: bool) -> Tuple[int, int, int]:
    mode_rank = {"scheduling-failure": 0, "periodic": 1, "stable-queue": 2}.get(mode, 99)
    enf_rank = {"blocking": 0, "nonblocking": 1}.get(enforcement, 9)
    defp_rank = 1 if with_defpreempt else 0
    return (mode_rank, enf_rank, defp_rank)


def _make_wide_table(agg: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    keep_cols = [
        "k_max",
        "n_nodes",
        "mean_arrival_s",
        "row_label",
        "plugin_mode",
        "plugin_enforcement",
        "with_default_preemption",
        value_col,
    ]
    keep = agg[keep_cols].copy()

    keep["col"] = keep.apply(
        lambda r: f"N{int(r['n_nodes'])}_muA{_fmt_arrival(r['mean_arrival_s'])}", axis=1
    )

    piv = keep.pivot_table(
        index=["k_max", "row_label", "plugin_mode", "plugin_enforcement", "with_default_preemption"],
        columns="col",
        values=value_col,
        aggfunc="first",
    ).reset_index()

    # column order
    cols = [c for c in piv.columns if c not in ("k_max", "row_label", "plugin_mode", "plugin_enforcement", "with_default_preemption")]
    cols_sorted = sorted(cols, key=_col_sort_key)

    piv = piv[["k_max", "row_label", "plugin_mode", "plugin_enforcement", "with_default_preemption", *cols_sorted]]

    # row order
    piv["_rowkey"] = piv.apply(
        lambda r: _row_sort_key(str(r["plugin_mode"]), str(r["plugin_enforcement"]), bool(r["with_default_preemption"])),
        axis=1,
    )
    piv = piv.sort_values(["k_max", "_rowkey", "row_label"]).drop(columns=["_rowkey"])

    return piv


def _fmt_signed(x: float, *, decimals: int) -> str:
    if x is None or not math.isfinite(float(x)):
        return r"\text{--}"
    v = float(x)
    fmt = f"{{:+.{int(decimals)}f}}"
    s = fmt.format(v)
    # avoid "-0.0"
    if s.startswith("-0") and abs(v) < 0.5 * (10 ** (-decimals)):
        s = s.replace("-", "+", 1)
    return s


def _prio_macro_name(k_max: int) -> Optional[str]:
    if k_max == 2:
        return r"\prioDeltaTwo"
    if k_max == 3:
        return r"\prioDeltaThree"
    if k_max == 4:
        return r"\prioDeltaFour"
    return None


def _make_prio_cell_from_row(row: pd.Series, *, metric_prefix: str, decimals: int) -> str:
    """Create a LaTeX-friendly cell string for per-priority values.

    Expects aggregated columns like f"{metric_prefix}{p}_mean" for p=1..k_max.
    """
    try:
        k_max = int(row["k_max"])
    except Exception:
        k_max = 0

    if k_max <= 0:
        return r"\text{--}"

    # scalar for k_max=1
    if k_max == 1:
        col = f"{metric_prefix}1_mean"
        return _fmt_signed(float(row.get(col, float("nan"))), decimals=decimals)

    macro = _prio_macro_name(k_max)
    if macro is None:
        # fallback: join as comma-separated
        vals = []
        for p in range(k_max, 0, -1):
            col = f"{metric_prefix}{p}_mean"
            vals.append(_fmt_signed(float(row.get(col, float("nan"))), decimals=decimals))
        return ",".join(vals)

    # pack highest -> lowest
    args: List[str] = []
    for p in range(k_max, 0, -1):
        col = f"{metric_prefix}{p}_mean"
        args.append(_fmt_signed(float(row.get(col, float("nan"))), decimals=decimals))

    return f"{macro}" + "{" + "}{".join(args) + "}"


# -----------------------------
# Main aggregation
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

    out_dir = Path(args.out_dir) if args.out_dir else (root / "tables_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    default_idx = build_default_index(default_root)

    # Cache metrics per (kind, run_name, regime, seed)
    metrics_cache: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}

    # --- Raw metrics: default + plugin ---
    rows_long: List[Dict[str, object]] = []

    for (regime_key, seed), (gen_p, pod_p) in sorted(default_idx.items()):
        m = compute_metrics_for_run(gen_p, pod_p, eps_s=float(args.eps_s))
        metrics_cache[("default", "default", regime_key, seed)] = m
        rows_long.append(
            {
                "kind": "default",
                "run_name": "default",
                "regime_key": regime_key,
                "seed": seed,
                **_regime_fields(regime_key),
                **m,
            }
        )

    for base_regime, cfg, plugin_dirname, run_dir in iter_plugin_runs(plugin_root):
        for seed, seed_dir in iter_seed_dirs(run_dir):
            gen_p = seed_dir / GENERAL_STATS_FILENAME
            pod_p = seed_dir / POD_STATS_FILENAME
            m = compute_metrics_for_run(gen_p, pod_p, eps_s=float(args.eps_s))
            metrics_cache[("plugin", plugin_dirname, base_regime, seed)] = m
            rows_long.append(
                {
                    "kind": "plugin",
                    "run_name": plugin_dirname,
                    "regime_key": base_regime,
                    "seed": seed,
                    "plugin_mode": cfg.mode,
                    "plugin_param_s": cfg.param_s,
                    "plugin_enforcement": cfg.enforcement,
                    "with_default_preemption": cfg.with_default_preemption,
                    "row_label": cfg.row_label(),
                    **_regime_fields(base_regime),
                    **m,
                }
            )

    df_long = pd.DataFrame(rows_long)
    df_long.to_csv(out_dir / "results_long.csv", index=False)

    # --- Paired deltas: plugin - default per seed ---
    paired_rows: List[Dict[str, object]] = []

    for base_regime, cfg, plugin_dirname, run_dir in iter_plugin_runs(plugin_root):
        for seed, _ in iter_seed_dirs(run_dir):
            d_key = (base_regime, seed)
            if d_key not in default_idx:
                if not args.quiet:
                    print(f"[warn] missing default match for regime={base_regime} seed={seed}")
                continue

            default_m = metrics_cache.get(("default", "default", base_regime, seed))
            plugin_m = metrics_cache.get(("plugin", plugin_dirname, base_regime, seed))
            if default_m is None or plugin_m is None:
                continue

            T_common = min(float(default_m.get("T_end_s", 0.0)), float(plugin_m.get("T_end_s", 0.0)))

            def_gen, _def_pod = default_idx[d_key]
            plug_gen = run_dir / seed / GENERAL_STATS_FILENAME

            def_series = _series_metrics_at_horizon(def_gen, T_common)
            plug_series = _series_metrics_at_horizon(plug_gen, T_common)

            prios = sorted(set(def_series["prios"]) | set(plug_series["prios"]))
            p_hi = max(prios) if prios else 0

            delta: Dict[str, float] = {}

            # Util (higher better)
            delta["delta_util_eff_mean"] = float(plug_series.get("util_eff_mean", 0.0)) - float(
                def_series.get("util_eff_mean", 0.0)
            )
            delta["impr_util_eff_mean"] = delta["delta_util_eff_mean"]

            # Per-priority deltas
            for p in prios:
                R_def = float(def_series["R"].get(p, 0.0))
                R_plu = float(plug_series["R"].get(p, 0.0))
                D_def = float(def_series["D"].get(p, 0.0))
                D_plu = float(plug_series["D"].get(p, 0.0))

                delta[f"delta_R_p{p}"] = R_plu - R_def
                delta[f"impr_R_p{p}"] = delta[f"delta_R_p{p}"]

                delta[f"delta_D_p{p}"] = D_plu - D_def
                delta[f"impr_D_p{p}"] = -delta[f"delta_D_p{p}"]

            # Convenience: hi and total
            if p_hi > 0:
                delta["delta_R_hi"] = delta.get(f"delta_R_p{p_hi}", 0.0)
                delta["impr_R_hi"] = delta["delta_R_hi"]

                delta["delta_D_hi"] = delta.get(f"delta_D_p{p_hi}", 0.0)
                delta["impr_D_hi"] = -delta["delta_D_hi"]

            R_def_tot = sum(float(def_series["R"].get(p, 0.0)) for p in prios)
            R_plu_tot = sum(float(plug_series["R"].get(p, 0.0)) for p in prios)
            D_def_tot = sum(float(def_series["D"].get(p, 0.0)) for p in prios)
            D_plu_tot = sum(float(plug_series["D"].get(p, 0.0)) for p in prios)

            delta["delta_R_total"] = R_plu_tot - R_def_tot
            delta["impr_R_total"] = delta["delta_R_total"]

            delta["delta_D_total"] = D_plu_tot - D_def_tot
            delta["impr_D_total"] = -delta["delta_D_total"]

            # Latency mean / p95 (lower better)
            lat_def = default_m.get("latency_first_admit_mean_s")
            lat_plu = plugin_m.get("latency_first_admit_mean_s")
            if lat_def is not None and lat_plu is not None and math.isfinite(float(lat_def)) and math.isfinite(float(lat_plu)):
                delta["delta_latency_mean_s"] = float(lat_plu) - float(lat_def)
                delta["impr_latency_mean_s"] = -delta["delta_latency_mean_s"]
            else:
                delta["delta_latency_mean_s"] = float("nan")
                delta["impr_latency_mean_s"] = float("nan")

            p95_def = default_m.get("latency_first_admit_p95_s")
            p95_plu = plugin_m.get("latency_first_admit_p95_s")
            if p95_def is not None and p95_plu is not None and math.isfinite(float(p95_def)) and math.isfinite(float(p95_plu)):
                delta["delta_latency_p95_s"] = float(p95_plu) - float(p95_def)
                delta["impr_latency_p95_s"] = -delta["delta_latency_p95_s"]
            else:
                delta["delta_latency_p95_s"] = float("nan")
                delta["impr_latency_p95_s"] = float("nan")

            paired_rows.append(
                {
                    "regime_key": base_regime,
                    "seed": seed,
                    "run_name": plugin_dirname,
                    "row_label": cfg.row_label(),
                    "plugin_mode": cfg.mode,
                    "plugin_param_s": cfg.param_s,
                    "plugin_enforcement": cfg.enforcement,
                    "with_default_preemption": cfg.with_default_preemption,
                    "T_common_s": T_common,
                    "p_hi": p_hi,
                    **_regime_fields(base_regime),
                    **delta,
                }
            )

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(out_dir / "results_paired.csv", index=False)

    if df_paired.empty:
        print("No paired rows produced; check directory structure and matching.")
        return

    # --- Aggregate across seeds ---
    group_cols = [
        "regime_key",
        "n_nodes",
        "k_max",
        "mean_arrival_s",
        "run_name",
        "row_label",
        "plugin_mode",
        "plugin_param_s",
        "plugin_enforcement",
        "with_default_preemption",
        "p_hi",
    ]

    metric_cols = [c for c in df_paired.columns if c.startswith("delta_") or c.startswith("impr_")]

    agg = (
        df_paired.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # flatten multiindex columns
    agg.columns = [
        ("_".join([c for c in col if c]) if isinstance(col, tuple) else col)
        for col in agg.columns
    ]

    agg.to_csv(out_dir / "results_agg.csv", index=False)

    # --- Tables ---
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Previous key metrics (mean paired improvement)
    for mcol in [
        "impr_R_hi_mean",
        "impr_D_hi_mean",
        "impr_latency_mean_s_mean",
        "impr_util_eff_mean_mean",
        "impr_R_total_mean",
        "impr_D_total_mean",
    ]:
        if mcol in agg.columns:
            df_tbl = _make_wide_table(agg, value_col=mcol)
            df_tbl.to_csv(tables_dir / f"table_{mcol}.csv", index=False)

    # Per-priority numeric tables (one CSV per p, for R and D improvements)
    max_k = int(pd.to_numeric(agg["k_max"], errors="coerce").max()) if ("k_max" in agg.columns) else 0
    max_k = max(0, max_k)

    for p in range(1, max_k + 1):
        col_r = f"delta_R_p{p}_mean"  # R: delta == improvement
        if col_r in agg.columns:
            _make_wide_table(agg, value_col=col_r).to_csv(tables_dir / f"table_{col_r}.csv", index=False)

        col_d_impr = f"impr_D_p{p}_mean"
        if col_d_impr in agg.columns:
            _make_wide_table(agg, value_col=col_d_impr).to_csv(tables_dir / f"table_{col_d_impr}.csv", index=False)

    # Per-priority packed LaTeX cell tables (matches your \prioDeltaFour style)
    # Build a derived column on agg (string) and pivot it.
    agg_r_cell = agg.copy()
    agg_r_cell["delta_R_p_cell_mean"] = agg_r_cell.apply(
        lambda r: _make_prio_cell_from_row(r, metric_prefix="delta_R_p", decimals=int(args.decimals)), axis=1
    )
    _make_wide_table(agg_r_cell, value_col="delta_R_p_cell_mean").to_csv(
        tables_dir / "table_delta_R_p_cell_mean.csv", index=False
    )

    agg_d_cell = agg.copy()
    agg_d_cell["impr_D_p_cell_mean"] = agg_d_cell.apply(
        lambda r: _make_prio_cell_from_row(r, metric_prefix="impr_D_p", decimals=int(args.decimals)), axis=1
    )
    _make_wide_table(agg_d_cell, value_col="impr_D_p_cell_mean").to_csv(
        tables_dir / "table_impr_D_p_cell_mean.csv", index=False
    )

    print(f"Wrote CSV outputs to: {out_dir}")


if __name__ == "__main__":
    main()
