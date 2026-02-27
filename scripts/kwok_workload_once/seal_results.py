#!/usr/bin/env python3
# scripts/kwok_workload_once/seal_results.py
"""
CP-SAT:
python -m scripts.kwok_workload_once.seal_results --solver-dir plugin-cp_sat

Gurobi:
python -m scripts.kwok_workload_once.seal_results --solver-dir plugin-gurobi
"""

import argparse, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from scripts.helpers.data_helpers import format_num, rate
from scripts.helpers.general_helpers import (
    cmp_placed_by_prio_row,
    parse_json_cell,
)

# ============================================================
# CONFIG
# ============================================================

RESULTS_ROOT: Path = Path("analysis/kwok_workload_once")
SOLVER_DIRNAME: str = "plugin-cp_sat"
DEFAULT_DIRNAME: str = "default"
RESULTS_CSV_NAME: str = "results.csv"
OUT_DIR: Path = Path("analysis/kwok_workload_once")

# If None: keep raw floats, else format with decimals
DECIMALS: Optional[int] = 4  # set to None to disable formatting

# Matching directory names:
# nodes16_pods128_prio4_util090_timeout10
DIR_RE_SOLVER = re.compile(
    r"^nodes(?P<nodes>\d+)_pods(?P<pods>\d+)_prio(?P<prio>\d+)_util(?P<util>\d{3})_timeout(?P<timeout>\d{2})$"
)

# ============================================================
# Helpers
# ============================================================

def parse_solver_dirname(name: str) -> Optional[Dict[str, Any]]:
    """
    Parse solver directory name into metadata.
    """
    m = DIR_RE_SOLVER.match(name)
    if not m:
        return None
    d = m.groupdict()
    nodes = int(d["nodes"])
    pods = int(d["pods"])
    priorities = int(d["prio"])
    util = float(d["util"])
    timeout = int(d["timeout"])
    return {
        "nodes": nodes,
        "pods": pods,
        "priorities": priorities,
        "util": util,
        "timeout": timeout,
        "pods_per_node": int(pods / nodes),
        "default_dirname": f"nodes{nodes}_pods{pods}_prio{priorities}_util{d['util']}",
    }

def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load results CSV and normalize columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    cols = {c.lower(): c for c in df.columns}

    rename_map = {
        "seed": "seed",
        "util_run_cpu_now": "util_run_cpu",
        "util_run_mem_now": "util_run_mem",
        "running_placed_by_prio_now": "placed_by_prio_running",

        "unscheduled_count_before": "unscheduled_cnt",
        "unscheduled_count_now": "unscheduled_cnt_now",

        "error": "error",
        "best_solver_status": "solver_status",
        "best_solver_name": "solver_name",
        "best_solver_duration_ms": "solver_duration_ms",
        "best_solver_score": "solver_score",
    }

    df = df.rename(columns={cols[k]: v for k, v in rename_map.items() if k in cols and cols[k] != v})

    # Ensure optional solver columns exist
    for c in [
        "solver_status", "solver_name", "solver_duration_ms", "solver_score",
        "unscheduled_cnt", "unscheduled_cnt_now",
    ]:
        if c not in df.columns:
            df[c] = ""

    # minimal conversions
    df["seed"] = df["seed"].astype(str).str.strip()
    df["util_run_cpu"] = pd.to_numeric(df["util_run_cpu"].astype(str), errors="coerce")
    df["util_run_mem"] = pd.to_numeric(df["util_run_mem"].astype(str), errors="coerce")
    df["solver_duration_ms"] = pd.to_numeric(df["solver_duration_ms"].astype(str), errors="coerce")
    df["solver_status"] = df["solver_status"].astype(str).str.strip().str.upper()

    # placed_by_prio: use what is actually running (as you had)
    df["placed_by_prio"] = df.apply(
        lambda r: json.dumps(parse_json_cell(r.get("placed_by_prio_running", "")), separators=(",", ":")),
        axis=1,
    )
    
    # rows where solver actually produced a solution
    df["solver_has_solution"] = df["solver_status"].isin(["OPTIMAL", "FEASIBLE"]).astype(int)

    # Extract moves + evictions directly from best_solver_score
    disrupt_df = df["solver_score"].apply(
        lambda x: pd.Series(
            _extract_solver_disruption_parts(x),
            index=["moves_raw", "evictions_raw"],
        )
    )
    df = pd.concat([df, disrupt_df], axis=1)

    # Final normalized disruption columns (only meaningful for rows with a solution)
    def _finalize_disruption(row: pd.Series) -> Tuple[int, int]:
        if int(row.get("solver_has_solution", 0)) != 1:
            return 0, 0

        moves = _safe_int_from_any(row.get("moves_raw"))
        evictions = _safe_int_from_any(row.get("evictions_raw"))

        # If field is missing, default to 0 (common in single-priority cases: no evictions)
        return max(moves or 0, 0), max(evictions or 0, 0)

    moves_ev_df = df.apply(
        lambda r: pd.Series(_finalize_disruption(r), index=["moves", "evictions"]),
        axis=1,
    )
    df = pd.concat([df, moves_ev_df], axis=1)
    
    return df

def _sum_placed_by_prio(cell: Any) -> int:
    """
    Sum total running pods from a placed_by_prio JSON cell.
    Accepts dict values as int/float/str.
    """
    d = parse_json_cell(cell)
    if not isinstance(d, dict):
        return 0
    total = 0
    for v in d.values():
        try:
            if v is None or v == "":
                continue
            total += int(float(v))
        except Exception:
            continue
    return int(total)

def default_vs_solver_per_seed(solver_csv: Path, default_csv: Path) -> pd.DataFrame:
    """
    Compare solver vs default per-seed results.
    """
    df_s = load_csv(solver_csv)
    df_d = load_csv(default_csv)

    joined = df_s[
        [
            "seed",
            "util_run_cpu",
            "util_run_mem",
            "placed_by_prio",
            "unscheduled_cnt",
            "unscheduled_cnt_now",
            "error",
            "solver_status",
            "solver_name",
            "solver_duration_ms",
            "solver_has_solution",
            "moves",
            "evictions",
        ]
    ].rename(
        columns={
            "util_run_cpu": "util_cpu_solver",
            "util_run_mem": "util_mem_solver",
            "placed_by_prio": "placed_by_prio_solver",
            "unscheduled_cnt": "unscheduled_cnt_solver",
            "unscheduled_cnt_now": "unscheduled_cnt_now_solver",
        }
    ).merge(
        df_d[
            ["seed", "util_run_cpu", "util_run_mem", "placed_by_prio", "unscheduled_cnt", "unscheduled_cnt_now"]
        ].rename(
            columns={
                "util_run_cpu": "util_cpu_default",
                "util_run_mem": "util_mem_default",
                "placed_by_prio": "placed_by_prio_default",
                "unscheduled_cnt": "unscheduled_cnt_default",
                "unscheduled_cnt_now": "unscheduled_cnt_now_default",
            }
        ),
        on="seed",
        how="inner",
        validate="one_to_one",
    )

    no_pending_default = pd.to_numeric(joined["unscheduled_cnt_default"], errors="coerce").eq(0)
    no_pending_solver = pd.to_numeric(joined["unscheduled_cnt_solver"], errors="coerce").eq(0)
    joined["default_all_running"] = no_pending_default & no_pending_solver

    joined["solver_all_running_after_opt"] = pd.to_numeric(
        joined["unscheduled_cnt_now_solver"], errors="coerce"
    ).eq(0)

    # solver_called: any of status/name/duration present
    joined["solver_called"] = (
        joined["solver_status"].astype(str).str.strip().ne("")
        | joined["solver_name"].astype(str).str.strip().ne("")
        | pd.to_numeric(joined["solver_duration_ms"], errors="coerce").notna()
    ).astype(int)

    # compare placements
    joined["placed_cmp"] = joined.apply(cmp_placed_by_prio_row, axis=1)

    # total running pods (sum across priorities) and delta vs default
    joined["running_pods_solver"] = joined["placed_by_prio_solver"].apply(_sum_placed_by_prio)
    joined["running_pods_default"] = joined["placed_by_prio_default"].apply(_sum_placed_by_prio)
    joined["running_pods_delta"] = joined["running_pods_solver"] - joined["running_pods_default"]

    # deltas for resource utilization
    joined["cpu_delta"] = joined["util_cpu_solver"] - joined["util_cpu_default"]
    joined["mem_delta"] = joined["util_mem_solver"] - joined["util_mem_default"]

    return joined

def _safe_int_from_any(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def _extract_solver_disruption_parts(score_cell: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract disruption counts from best_solver_score JSON.

    Supports common key variants:
      moved / moves
      evicted / evictions
    """
    score = parse_json_cell(score_cell)
    if not isinstance(score, dict):
        return None, None

    def pick_first_int(d: dict, keys: List[str]) -> Optional[int]:
        for k in keys:
            if k in d:
                v = _safe_int_from_any(d.get(k))
                if v is not None:
                    return v
        return None

    moves = pick_first_int(score, ["moved", "moves"])
    evictions = pick_first_int(score, ["evicted", "evictions"])

    return moves, evictions

@dataclass(frozen=True)
class CombineResultsArgs:
    results_root: Path
    solver_dir: str
    default_dir: str
    results_csv: str
    out_dir: Path
    decimals: Optional[int]

@dataclass(frozen=True)
class CategoryCounts:
    n_seeds: int
    n_seeds_not_all_running: int
    n_default_all_running: int
    n_solver_called: int
    n_default_optimal: int
    n_solver_optimal: int
    n_solver_feasible: int
    n_solver_failed: int
    n_solver_improve: int
    n_other: int
    default_all_running_rate: float
    solver_called_rate: float
    default_optimal_rate: float
    solver_optimal_rate: float
    solver_feasible_rate: float
    solver_failed_rate: float
    solver_improve_rate: float
    other_rate: float

class CombineResultsAnalyzer:
    def __init__(self, args: CombineResultsArgs) -> None:
        self.args = args

    @staticmethod
    def _split_default_all_running(per_seed_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Split per-seed DataFrame into default_all_running mask and not_all_running DataFrame.
        """
        mask_default_all = per_seed_df["default_all_running"]
        not_all_running = per_seed_df[~mask_default_all].copy()
        return mask_default_all, not_all_running

    @staticmethod
    def _status_flags(not_all_running: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Status flags: is_optimal, is_feasible, is_ok.
        """
        status = not_all_running["solver_status"]
        is_optimal = status.eq("OPTIMAL")
        is_feasible = status.eq("FEASIBLE")
        is_ok = is_optimal | is_feasible
        return is_optimal, is_feasible, is_ok

    @staticmethod
    def _placement_flags(not_all_running: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Placement flags: placed_equal, placed_better, placed_worse.
        """
        placed_equal = not_all_running["placed_cmp"].eq(0)
        placed_better = not_all_running["placed_cmp"].gt(0)
        placed_worse = not_all_running["placed_cmp"].lt(0)
        return placed_equal, placed_better, placed_worse

    @staticmethod
    def _solver_called_count(not_all_running: pd.DataFrame) -> int:
        """
        Count of solver_called in not_all_running DataFrame.
        """
        return int(not_all_running["solver_called"].sum())

    @staticmethod
    def _compute_category_counts(per_seed_df: pd.DataFrame) -> CategoryCounts:
        """
        Compute category counts from per-seed DataFrame.
        """
        mask_default_all, not_all_running = CombineResultsAnalyzer._split_default_all_running(per_seed_df)
        is_optimal, is_feasible, is_ok = CombineResultsAnalyzer._status_flags(not_all_running)
        placed_equal, placed_better, placed_worse = CombineResultsAnalyzer._placement_flags(not_all_running)

        n_solver_called = CombineResultsAnalyzer._solver_called_count(not_all_running)
        n_default_all = int(mask_default_all.sum())
        n_default_optimal = int((is_optimal & placed_equal).sum())
        n_solver_optimal = int((is_optimal & placed_better).sum())
        n_solver_feasible = int((is_feasible & placed_better).sum())

        n_solver_failed = int(((~is_ok) | (is_feasible & ~placed_better) | (is_optimal & placed_worse)).sum())
        n_solver_better = n_solver_optimal + n_solver_feasible

        n_other = len(not_all_running) - (n_default_optimal + n_solver_optimal + n_solver_feasible + n_solver_failed)

        n_seeds = max(1, len(per_seed_df))
        default_all_rate = rate(n_default_all, n_seeds)
        solver_called_rate = rate(n_solver_called, n_seeds)
        solver_failed_rate = rate(n_solver_failed, n_seeds)
        default_optimal_rate = rate(n_default_optimal, n_seeds)
        solver_optimal_rate = rate(n_solver_optimal, n_seeds)
        solver_feasible_rate = rate(n_solver_feasible, n_seeds)
        solver_improve_rate = rate(n_solver_better, n_seeds)

        # Keep existing behavior (even though you also compute n_other above):
        other_rate = 1.0 - (
            default_all_rate
            + default_optimal_rate
            + solver_optimal_rate
            + solver_feasible_rate
            + solver_failed_rate
        )

        return CategoryCounts(
            n_seeds=n_seeds,
            n_seeds_not_all_running=int(len(not_all_running)),
            n_default_all_running=n_default_all,
            n_solver_called=n_solver_called,
            n_default_optimal=n_default_optimal,
            n_solver_optimal=n_solver_optimal,
            n_solver_feasible=n_solver_feasible,
            n_solver_failed=n_solver_failed,
            n_solver_improve=n_solver_better,
            n_other=int(n_other),
            default_all_running_rate=float(default_all_rate),
            solver_called_rate=float(solver_called_rate),
            default_optimal_rate=float(default_optimal_rate),
            solver_optimal_rate=float(solver_optimal_rate),
            solver_feasible_rate=float(solver_feasible_rate),
            solver_failed_rate=float(solver_failed_rate),
            solver_improve_rate=float(solver_improve_rate),
            other_rate=float(other_rate),
        )

    def analyze_combo(self, *, solver_dir: Path, default_dir: Path, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze a solver/default combo and return summary row.
        """
        solver_csv = solver_dir / self.args.results_csv
        default_csv = default_dir / self.args.results_csv

        if not default_dir.exists():
            print(f"[skip] default folder missing: {default_dir.name}")
            return None
        if not solver_csv.exists():
            print(f"[skip] solver results.csv missing: {solver_csv}")
            return None
        if not default_csv.exists():
            print(f"[skip] default results.csv missing: {default_csv}")
            return None

        per_seed_df = default_vs_solver_per_seed(solver_csv, default_csv)
        _, not_all_running = self._split_default_all_running(per_seed_df)
        counts = self._compute_category_counts(per_seed_df)

        # Sanity warnings (keep behavior)
        rate_sum = (
            counts.default_all_running_rate
            + counts.default_optimal_rate
            + counts.solver_optimal_rate
            + counts.solver_feasible_rate
            + counts.solver_failed_rate
            + counts.other_rate
        )
        if rate_sum > 1.00 + 1e-6:
            print(f"[warn] {solver_dir.name}: rates sum > 1.00 (sum={rate_sum:.4f})")
        if rate_sum < 0.00 - 1e-6:
            print(f"[warn] {solver_dir.name}: rates sum < 0.00 (sum={rate_sum:.4f})")

        # Durations & deltas (keep behavior)
        t_sum = float(not_all_running["solver_duration_ms"].sum())
        t_mean = not_all_running["solver_duration_ms"].mean()
        cpu_delta_mean = not_all_running["cpu_delta"].mean()
        mem_delta_mean = not_all_running["mem_delta"].mean()
        cpu_delta_sum = float(not_all_running["cpu_delta"].sum())
        mem_delta_sum = float(not_all_running["mem_delta"].sum())

        count_sum = (
            counts.n_default_all_running
            + counts.n_default_optimal
            + counts.n_solver_optimal
            + counts.n_solver_feasible
            + counts.n_solver_failed
            + counts.n_other
        )
        if count_sum != len(per_seed_df):
            print(f"[warn] {solver_dir.name}: category counts sum={count_sum} != joined={len(per_seed_df)}")

        # Disruption statistics (only rows where solver produced a solution)
        solver_sol_mask = (
            pd.to_numeric(not_all_running["solver_has_solution"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
        )
        n_solver_solution = int(solver_sol_mask.sum())

        def _sum_num(mask: pd.Series, col: str) -> float:
            return float(pd.to_numeric(not_all_running.loc[mask, col], errors="coerce").fillna(0).sum())

        # Split by outcome category (for disruption accounting)
        # - default_optimal  := solver found OPTIMAL and placement is equal to default (KWOK Optimal)
        # - solver_optimal   := solver found OPTIMAL and placement is better than default (Better&Optimal)
        # - solver_feasible  := solver found FEASIBLE and placement is better than default (Better)
        is_optimal_better = (
            not_all_running["solver_status"].eq("OPTIMAL")
            & not_all_running["placed_cmp"].gt(0)
        )
        is_feasible_better = (
            not_all_running["solver_status"].eq("FEASIBLE")
            & not_all_running["placed_cmp"].gt(0)
        )
        
        # "All pods scheduled" means solver leaves no pods unscheduled after optimization
        solver_all_running_mask = not_all_running["solver_all_running_after_opt"].astype(bool)

        # Counts within outcome categories
        n_solver_optimal_all_running = int((is_optimal_better & solver_all_running_mask).sum())   # Better&Optimal
        n_solver_feasible_all_running = int((is_feasible_better & solver_all_running_mask).sum()) # Better

        def _safe_frac(num: int, den: int) -> float:
            return float(num) / float(den) if den > 0 else float("nan")

        solver_optimal_all_running_within_rate = _safe_frac(
            n_solver_optimal_all_running, counts.n_solver_optimal
        )
        solver_feasible_all_running_within_rate = _safe_frac(
            n_solver_feasible_all_running, counts.n_solver_feasible
        )

        # Better&Optimal
        moves_sum_solver_optimal = _sum_num(is_optimal_better, "moves")
        evictions_sum_solver_optimal = _sum_num(is_optimal_better, "evictions")

        # Better
        moves_sum_solver_feasible = _sum_num(is_feasible_better, "moves")
        evictions_sum_solver_feasible = _sum_num(is_feasible_better, "evictions")

        # Normalize disruptions to % of total pods (constant per configuration)
        total_pods = int(meta["nodes"] * meta["pods_per_node"])  # same as meta["pods"]

        def _pct_sum_of_total_pods(x: float) -> float:
            if total_pods <= 0:
                return float("nan")
            return 100.0 * float(x) / float(total_pods)

        # --- Additional pods admitted (solver - default), split by category
        pods_delta_sum_solver_optimal = float(
            pd.to_numeric(not_all_running.loc[is_optimal_better, "running_pods_delta"], errors="coerce").fillna(0).sum()
        )
        pods_delta_sum_solver_feasible = float(
            pd.to_numeric(not_all_running.loc[is_feasible_better, "running_pods_delta"], errors="coerce").fillna(0).sum()
        )

        pods_delta_pct_sum_solver_optimal = _pct_sum_of_total_pods(pods_delta_sum_solver_optimal)
        pods_delta_pct_sum_solver_feasible = _pct_sum_of_total_pods(pods_delta_sum_solver_feasible)

        decimals = self.args.decimals
        return {
            "util": meta["util"],
            "nodes": meta["nodes"],
            "pods": meta["pods"],
            "pods_per_node": meta["pods_per_node"],
            "priorities": meta["priorities"],
            "timeout_s": meta["timeout"],
            "config_dir": solver_dir.name,
            "n_seeds": int(len(per_seed_df)),
            "n_seeds_not_all_running": int(len(not_all_running)),
            "n_default_all_running": counts.n_default_all_running,
            "default_all_running_rate": format_num(counts.default_all_running_rate, decimals),
            "n_solver_called": counts.n_solver_called,
            "solver_called_rate": format_num(counts.solver_called_rate, decimals),
            "n_solver_failed": counts.n_solver_failed,
            "solver_failed_rate": format_num(counts.solver_failed_rate, decimals),
            "n_default_optimal": counts.n_default_optimal,
            "default_optimal_rate": format_num(counts.default_optimal_rate, decimals),
            "n_solver_optimal": counts.n_solver_optimal,
            "solver_optimal_rate": format_num(counts.solver_optimal_rate, decimals),
            "n_solver_feasible": counts.n_solver_feasible,
            "solver_feasible_rate": format_num(counts.solver_feasible_rate, decimals),
            "n_solver_improve": counts.n_solver_improve,
            "solver_improve_rate": format_num(counts.solver_improve_rate, decimals),
            "n_other": counts.n_other,
            "other_rate": format_num(counts.other_rate, decimals),
            "solver_duration_ms_sum": format_num(t_sum, decimals),
            "solver_duration_ms_mean": format_num(float(t_mean) if t_mean == t_mean else float("nan"), decimals),
            "cpu_delta_sum": format_num(cpu_delta_sum, decimals),
            "mem_delta_sum": format_num(mem_delta_sum, decimals),
            "cpu_delta_mean": format_num(
                float(cpu_delta_mean) if cpu_delta_mean == cpu_delta_mean else float("nan"), decimals
            ),
            "mem_delta_mean": format_num(
                float(mem_delta_mean) if mem_delta_mean == mem_delta_mean else float("nan"), decimals
            ),
            "total_pods": total_pods,
            "n_solver_solution": n_solver_solution,

            # Normalized disruption sums (sum over seeds of % of total pods)
            "moves_pct_sum_solver_optimal": format_num(_pct_sum_of_total_pods(moves_sum_solver_optimal), decimals),
            "evictions_pct_sum_solver_optimal": format_num(_pct_sum_of_total_pods(evictions_sum_solver_optimal), decimals),
            "moves_pct_sum_solver_feasible": format_num(_pct_sum_of_total_pods(moves_sum_solver_feasible), decimals),
            "evictions_pct_sum_solver_feasible": format_num(_pct_sum_of_total_pods(evictions_sum_solver_feasible), decimals),
            
            # All-pods-scheduled within optimal categories (conditional on category)
            "n_solver_feasible_all_running": n_solver_feasible_all_running,
            "solver_feasible_all_running_within_rate": format_num(solver_feasible_all_running_within_rate, decimals),

            "n_solver_optimal_all_running": n_solver_optimal_all_running,
            "solver_optimal_all_running_within_rate": format_num(solver_optimal_all_running_within_rate, decimals),
            
            # Additional pods admitted (% of total pods), summed over seeds within category
            "admitted_pods_pct_sum_solver_optimal": format_num(pods_delta_pct_sum_solver_optimal, decimals),  # Better&Optimal
            "admitted_pods_pct_sum_solver_feasible": format_num(pods_delta_pct_sum_solver_feasible, decimals),  # Better
        }

    def run(self) -> None:
        """
        Main runner.
        """
        solver_root = (self.args.results_root / self.args.solver_dir).resolve()
        default_root = (self.args.results_root / self.args.default_dir).resolve()
        out_dir = self.args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        per_combo_rows: List[Dict[str, Any]] = []
        solver_combos = sorted([p for p in solver_root.iterdir() if p.is_dir()])

        for solver_dir in solver_combos:
            meta = parse_solver_dirname(solver_dir.name)
            if not meta:
                print(f"[skip] bad folder name: {solver_dir.name}")
                continue
            default_dir = default_root / meta["default_dirname"]
            row = self.analyze_combo(solver_dir=solver_dir, default_dir=default_dir, meta=meta)
            if row is not None:
                per_combo_rows.append(row)

        per_combo_df = pd.DataFrame(per_combo_rows)

        numeric_cols = [
            "n_seeds",
            "n_default_all_running",
            "n_solver_called",
            "n_solver_solution",
            "n_solver_failed",
            "n_default_optimal",
            "n_solver_optimal",
            "n_solver_feasible",
            "n_solver_improve",
            "n_other",
            "total_pods",
            "cpu_delta_mean",
            "mem_delta_mean",
            "cpu_delta_sum",
            "mem_delta_sum",
            "solver_duration_ms_sum",
            "solver_duration_ms_mean",
            "moves_pct_sum_solver_optimal",
            "evictions_pct_sum_solver_optimal",
            "moves_pct_sum_solver_feasible",
            "evictions_pct_sum_solver_feasible",            
            "n_solver_feasible_all_running",
            "solver_feasible_all_running_within_rate",
            "n_solver_optimal_all_running",
            "solver_optimal_all_running_within_rate",
            "admitted_pods_pct_sum_solver_optimal",
            "admitted_pods_pct_sum_solver_feasible",
        ]
        for c in numeric_cols:
            if c in per_combo_df.columns:
                per_combo_df[c] = pd.to_numeric(per_combo_df[c], errors="coerce")

        # Derive output filename from solver_dir (e.g., plugin -> cp_sat, plugin-gurobi -> gurobi)
        solver_suffix = self.args.solver_dir.replace("plugin-", "").replace("plugin", "cp_sat")
        if not solver_suffix:
            solver_suffix = "cp_sat"
        out_filename = f"results_per_combo_{solver_suffix}.csv"
        
        out_per_combo = out_dir / out_filename
        per_combo_df.sort_values(
            ["util", "nodes", "pods_per_node", "priorities", "timeout_s", "config_dir"]
        ).to_csv(out_per_combo, index=False)
        print(f"[ok] wrote {out_per_combo} (rows={len(per_combo_df)})")

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Combine scheduler results")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT, help="Root directory for results")
    parser.add_argument("--solver-dir", type=str, default=SOLVER_DIRNAME, help="Solver directory name")
    parser.add_argument("--default-dir", type=str, default=DEFAULT_DIRNAME, help="Default directory name")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument("--results-csv", type=str, default=RESULTS_CSV_NAME, help="Results CSV filename")
    parser.add_argument("--decimals", type=int, default=DECIMALS, help="Number of decimal places to format (-1 to disable)")
    
    parsed_args = parser.parse_args(argv)
    
    # Convert -1 to None for decimals
    decimals = parsed_args.decimals if parsed_args.decimals >= 0 else None
    
    args = CombineResultsArgs(
        results_root=parsed_args.results_root,
        solver_dir=parsed_args.solver_dir,
        default_dir=parsed_args.default_dir,
        results_csv=parsed_args.results_csv,
        out_dir=parsed_args.out_dir,
        decimals=decimals,
    )
    CombineResultsAnalyzer(args).run()

if __name__ == "__main__":
    main()
