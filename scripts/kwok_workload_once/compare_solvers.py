#!/usr/bin/env python3
# scripts/kwok_workload_once/compare_solvers.py
"""
Compare solver results (CP-SAT, CBC, Gurobi) and report which is better for each configuration.

Supported solvers: cp_sat, cbc, gurobi

Usage:
    # Default: CP-SAT vs CBC
    python -m scripts.kwok_workload_once.compare_solvers

    # Full custom comparison
    python -m scripts.kwok_workload_once.compare_solvers \
        --results-root analysis/kwok_workload_once \
        --solver-a-dir plugin --solver-a-name cp_sat \
        --solver-b-dir plugin-cbc --solver-b-name cbc \
        --out-dir analysis/kwok_workload_once/comparison
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from scripts.helpers.data_helpers import format_num, rate
from scripts.helpers.general_helpers import parse_json_cell

# ============================================================
# CONFIG
# ============================================================

RESULTS_ROOT = Path("analysis/kwok_workload_once")

# Solver directory and name mappings
# Available solvers: cp_sat, cbc, gurobi
SOLVER_A_DIR = "plugin"          # CP-SAT (default)
SOLVER_A_NAME = "cp_sat"
SOLVER_B_DIR = "plugin-cbc"      # CBC (default comparison)
SOLVER_B_NAME = "cbc"

DEFAULT_DIR = "default"
RESULTS_CSV_NAME = "results.csv"
OUT_DIR = Path("analysis/kwok_workload_once/comparison")

DECIMALS = 4

# Valid solver types for validation
VALID_SOLVER_NAMES = {"cp_sat", "cbc", "gurobi"}

# ============================================================
# Helpers
# ============================================================

def load_results_csv(csv_path: Path) -> pd.DataFrame:
    """Load and normalize a results CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    cols = {c.lower(): c for c in df.columns}

    rename_map = {
        "seed": "seed",
        "util_run_cpu_now": "util_cpu",
        "util_run_mem_now": "util_mem",
        "running_placed_by_prio_now": "placed_by_prio",
        "unscheduled_count_before": "unscheduled_cnt",
        "unscheduled_count_now": "unscheduled_cnt_after",
        "error": "error",
        "best_solver_status": "solver_status",
        "best_solver_name": "solver_name",
        "best_solver_duration_ms": "solver_duration_ms",
        "best_solver_score": "solver_score",
    }

    df = df.rename(columns={cols[k]: v for k, v in rename_map.items() if k in cols and cols[k] != v})

    df["seed"] = df["seed"].astype(str).str.strip()
    df["util_cpu"] = pd.to_numeric(df.get("util_cpu", "").astype(str), errors="coerce")
    df["util_mem"] = pd.to_numeric(df.get("util_mem", "").astype(str), errors="coerce")
    df["solver_duration_ms"] = pd.to_numeric(df.get("solver_duration_ms", "").astype(str), errors="coerce")
    df["solver_status"] = df.get("solver_status", "").astype(str).str.strip().str.upper()
    df["unscheduled_cnt"] = pd.to_numeric(df.get("unscheduled_cnt", "").astype(str), errors="coerce")
    df["unscheduled_cnt_after"] = pd.to_numeric(df.get("unscheduled_cnt_after", "").astype(str), errors="coerce")

    return df


def parse_placed_by_prio(val: str) -> Dict[str, int]:
    """Parse placed_by_prio JSON cell into dict."""
    return parse_json_cell(val) if val else {}


def count_total_placed(placed_dict: Dict[str, int]) -> int:
    """Sum all placed pods across priorities."""
    return sum(int(v) for v in placed_dict.values())


def compare_placements(placed_a: Dict[str, int], placed_b: Dict[str, int]) -> int:
    """
    Compare two placement dicts by priority order (higher priority = lower number).
    Returns: >0 if A better, <0 if B better, 0 if equal.
    Priority keys can be "p1", "p2", "1", "2", etc.
    """
    def parse_prio(x: str) -> int:
        """Parse priority key like 'p1' or '1' to int."""
        s = str(x).lower().lstrip("p")
        try:
            return int(s)
        except ValueError:
            return 999  # Unknown priority goes last
    
    all_prios = sorted(set(placed_a.keys()) | set(placed_b.keys()), key=parse_prio)
    
    for prio in all_prios:
        a_count = placed_a.get(prio, 0)
        b_count = placed_b.get(prio, 0)
        if a_count != b_count:
            return a_count - b_count  # More placed = better
    return 0


@dataclass
class SeedComparison:
    seed: str
    # Solver A
    a_status: str
    a_duration_ms: float
    a_placed_total: int
    a_unscheduled_after: int
    a_util_cpu: float
    # Solver B
    b_status: str
    b_duration_ms: float
    b_placed_total: int
    b_unscheduled_after: int
    b_util_cpu: float
    # Comparison
    winner: str  # 'A', 'B', 'tie', 'both_failed'
    reason: str


def compare_seed(row_a: pd.Series, row_b: pd.Series, solver_a_name: str, solver_b_name: str) -> SeedComparison:
    """Compare two solver results for the same seed."""
    
    a_status = str(row_a.get("solver_status", "")).upper()
    b_status = str(row_b.get("solver_status", "")).upper()
    
    a_duration = float(row_a.get("solver_duration_ms", 0) or 0)
    b_duration = float(row_b.get("solver_duration_ms", 0) or 0)
    
    a_placed = parse_placed_by_prio(str(row_a.get("placed_by_prio", "")))
    b_placed = parse_placed_by_prio(str(row_b.get("placed_by_prio", "")))
    
    a_placed_total = count_total_placed(a_placed)
    b_placed_total = count_total_placed(b_placed)
    
    a_unscheduled = int(row_a.get("unscheduled_cnt_after", 0) or 0)
    b_unscheduled = int(row_b.get("unscheduled_cnt_after", 0) or 0)
    
    a_util = float(row_a.get("util_cpu", 0) or 0)
    b_util = float(row_b.get("util_cpu", 0) or 0)
    
    # Determine winner
    a_ok = a_status in ("OPTIMAL", "FEASIBLE")
    b_ok = b_status in ("OPTIMAL", "FEASIBLE")
    
    if not a_ok and not b_ok:
        winner = "both_failed"
        reason = f"Both failed: {solver_a_name}={a_status}, {solver_b_name}={b_status}"
    elif a_ok and not b_ok:
        winner = "A"
        reason = f"{solver_a_name} succeeded ({a_status}), {solver_b_name} failed ({b_status})"
    elif b_ok and not a_ok:
        winner = "B"
        reason = f"{solver_b_name} succeeded ({b_status}), {solver_a_name} failed ({a_status})"
    else:
        # Both succeeded - compare by placement quality
        placement_cmp = compare_placements(a_placed, b_placed)
        
        if placement_cmp > 0:
            winner = "A"
            reason = f"{solver_a_name} placed more/better pods"
        elif placement_cmp < 0:
            winner = "B"
            reason = f"{solver_b_name} placed more/better pods"
        else:
            # Same placement - compare by optimality
            a_optimal = a_status == "OPTIMAL"
            b_optimal = b_status == "OPTIMAL"
            
            if a_optimal and not b_optimal:
                winner = "A"
                reason = f"{solver_a_name} reached OPTIMAL, {solver_b_name} only FEASIBLE"
            elif b_optimal and not a_optimal:
                winner = "B"
                reason = f"{solver_b_name} reached OPTIMAL, {solver_a_name} only FEASIBLE"
            else:
                # Same optimality - compare by speed
                if a_duration < b_duration * 0.9:  # 10% tolerance
                    winner = "A"
                    reason = f"{solver_a_name} faster ({a_duration:.0f}ms vs {b_duration:.0f}ms)"
                elif b_duration < a_duration * 0.9:
                    winner = "B"
                    reason = f"{solver_b_name} faster ({b_duration:.0f}ms vs {a_duration:.0f}ms)"
                else:
                    winner = "tie"
                    reason = "Equal placement and similar speed"
    
    return SeedComparison(
        seed=str(row_a["seed"]),
        a_status=a_status,
        a_duration_ms=a_duration,
        a_placed_total=a_placed_total,
        a_unscheduled_after=a_unscheduled,
        a_util_cpu=a_util,
        b_status=b_status,
        b_duration_ms=b_duration,
        b_placed_total=b_placed_total,
        b_unscheduled_after=b_unscheduled,
        b_util_cpu=b_util,
        winner=winner,
        reason=reason,
    )


@dataclass
class ConfigComparison:
    config_name: str
    nodes: int
    pods: int
    priorities: int
    util: int
    timeout: int
    n_seeds: int
    n_a_wins: int
    n_b_wins: int
    n_ties: int
    n_both_failed: int
    a_win_rate: float
    b_win_rate: float
    avg_a_duration_ms: float
    avg_b_duration_ms: float
    speed_ratio: float  # A/B (>1 means B faster)
    winner: str


def compare_config(
    solver_a_csv: Path,
    solver_b_csv: Path,
    config_name: str,
    meta: Dict[str, Any],
    solver_a_name: str,
    solver_b_name: str,
) -> Tuple[Optional[ConfigComparison], List[SeedComparison]]:
    """Compare two solver results for a configuration."""
    
    if not solver_a_csv.exists() or not solver_b_csv.exists():
        return None, []
    
    df_a = load_results_csv(solver_a_csv)
    df_b = load_results_csv(solver_b_csv)
    
    # Join on seed
    merged = df_a.merge(df_b, on="seed", suffixes=("_a", "_b"), how="inner")
    
    if merged.empty:
        return None, []
    
    seed_comparisons: List[SeedComparison] = []
    
    for _, row in merged.iterrows():
        row_a = {k.replace("_a", ""): v for k, v in row.items() if k.endswith("_a") or k == "seed"}
        row_b = {k.replace("_b", ""): v for k, v in row.items() if k.endswith("_b") or k == "seed"}
        row_a["seed"] = row["seed"]
        row_b["seed"] = row["seed"]
        
        cmp = compare_seed(pd.Series(row_a), pd.Series(row_b), solver_a_name, solver_b_name)
        seed_comparisons.append(cmp)
    
    n_seeds = len(seed_comparisons)
    n_a_wins = sum(1 for c in seed_comparisons if c.winner == "A")
    n_b_wins = sum(1 for c in seed_comparisons if c.winner == "B")
    n_ties = sum(1 for c in seed_comparisons if c.winner == "tie")
    n_both_failed = sum(1 for c in seed_comparisons if c.winner == "both_failed")
    
    a_win_rate = n_a_wins / n_seeds if n_seeds > 0 else 0
    b_win_rate = n_b_wins / n_seeds if n_seeds > 0 else 0
    
    avg_a_duration = sum(c.a_duration_ms for c in seed_comparisons) / n_seeds if n_seeds > 0 else 0
    avg_b_duration = sum(c.b_duration_ms for c in seed_comparisons) / n_seeds if n_seeds > 0 else 0
    
    speed_ratio = avg_a_duration / avg_b_duration if avg_b_duration > 0 else float("inf")
    
    if n_a_wins > n_b_wins:
        winner = solver_a_name
    elif n_b_wins > n_a_wins:
        winner = solver_b_name
    else:
        winner = "tie"
    
    config_cmp = ConfigComparison(
        config_name=config_name,
        nodes=meta["nodes"],
        pods=meta["pods"],
        priorities=meta["priorities"],
        util=meta["util"],
        timeout=meta["timeout"],
        n_seeds=n_seeds,
        n_a_wins=n_a_wins,
        n_b_wins=n_b_wins,
        n_ties=n_ties,
        n_both_failed=n_both_failed,
        a_win_rate=a_win_rate,
        b_win_rate=b_win_rate,
        avg_a_duration_ms=avg_a_duration,
        avg_b_duration_ms=avg_b_duration,
        speed_ratio=speed_ratio,
        winner=winner,
    )
    
    return config_cmp, seed_comparisons


def parse_config_dirname(name: str) -> Optional[Dict[str, Any]]:
    """Parse config directory name into metadata."""
    import re
    m = re.match(
        r"^nodes(?P<nodes>\d+)_pods(?P<pods>\d+)_prio(?P<prio>\d+)_util(?P<util>\d{3})_timeout(?P<timeout>\d{2})$",
        name,
    )
    if not m:
        return None
    d = m.groupdict()
    return {
        "nodes": int(d["nodes"]),
        "pods": int(d["pods"]),
        "priorities": int(d["prio"]),
        "util": int(d["util"]),
        "timeout": int(d["timeout"]),
    }


@dataclass
class CompareSolversArgs:
    results_root: Path
    solver_a_dir: str
    solver_a_name: str
    solver_b_dir: str
    solver_b_name: str
    results_csv: str
    out_dir: Path
    decimals: Optional[int]


class SolverComparator:
    def __init__(self, args: CompareSolversArgs) -> None:
        self.args = args

    def run(self) -> None:
        solver_a_root = (self.args.results_root / self.args.solver_a_dir).resolve()
        solver_b_root = (self.args.results_root / self.args.solver_b_dir).resolve()
        out_dir = self.args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not solver_a_root.exists():
            print(f"[error] Solver A directory not found: {solver_a_root}")
            return
        if not solver_b_root.exists():
            print(f"[error] Solver B directory not found: {solver_b_root}")
            return

        # Find common configurations
        a_configs = {p.name for p in solver_a_root.iterdir() if p.is_dir()}
        b_configs = {p.name for p in solver_b_root.iterdir() if p.is_dir()}
        common_configs = sorted(a_configs & b_configs)

        print(f"Found {len(common_configs)} common configurations between {self.args.solver_a_name} and {self.args.solver_b_name}")

        config_comparisons: List[ConfigComparison] = []
        all_seed_comparisons: List[Dict[str, Any]] = []

        for config_name in common_configs:
            meta = parse_config_dirname(config_name)
            if not meta:
                print(f"[skip] Cannot parse config name: {config_name}")
                continue

            solver_a_csv = solver_a_root / config_name / self.args.results_csv
            solver_b_csv = solver_b_root / config_name / self.args.results_csv

            config_cmp, seed_cmps = compare_config(
                solver_a_csv, solver_b_csv, config_name, meta,
                self.args.solver_a_name, self.args.solver_b_name,
            )

            if config_cmp:
                config_comparisons.append(config_cmp)
                for sc in seed_cmps:
                    all_seed_comparisons.append({
                        "config": config_name,
                        **meta,
                        "seed": sc.seed,
                        f"{self.args.solver_a_name}_status": sc.a_status,
                        f"{self.args.solver_a_name}_duration_ms": sc.a_duration_ms,
                        f"{self.args.solver_a_name}_placed": sc.a_placed_total,
                        f"{self.args.solver_b_name}_status": sc.b_status,
                        f"{self.args.solver_b_name}_duration_ms": sc.b_duration_ms,
                        f"{self.args.solver_b_name}_placed": sc.b_placed_total,
                        "winner": sc.winner.replace("A", self.args.solver_a_name).replace("B", self.args.solver_b_name),
                        "reason": sc.reason,
                    })

        # Summary statistics
        total_seeds = sum(c.n_seeds for c in config_comparisons)
        total_a_wins = sum(c.n_a_wins for c in config_comparisons)
        total_b_wins = sum(c.n_b_wins for c in config_comparisons)
        total_ties = sum(c.n_ties for c in config_comparisons)
        total_both_failed = sum(c.n_both_failed for c in config_comparisons)

        print("\n" + "=" * 60)
        print("SOLVER COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Solver A: {self.args.solver_a_name} ({self.args.solver_a_dir})")
        print(f"Solver B: {self.args.solver_b_name} ({self.args.solver_b_dir})")
        print(f"Configurations compared: {len(config_comparisons)}")
        print(f"Total seeds compared: {total_seeds}")
        print("-" * 60)
        print(f"{self.args.solver_a_name} wins: {total_a_wins} ({100*total_a_wins/total_seeds:.1f}%)" if total_seeds > 0 else "")
        print(f"{self.args.solver_b_name} wins: {total_b_wins} ({100*total_b_wins/total_seeds:.1f}%)" if total_seeds > 0 else "")
        print(f"Ties: {total_ties} ({100*total_ties/total_seeds:.1f}%)" if total_seeds > 0 else "")
        print(f"Both failed: {total_both_failed} ({100*total_both_failed/total_seeds:.1f}%)" if total_seeds > 0 else "")
        print("=" * 60)

        # Per-config summary
        print("\nPER-CONFIGURATION RESULTS:")
        print("-" * 60)
        for c in sorted(config_comparisons, key=lambda x: (x.util, x.timeout, x.nodes)):
            winner_symbol = "←" if c.winner == self.args.solver_a_name else ("→" if c.winner == self.args.solver_b_name else "=")
            print(
                f"  {c.config_name}: "
                f"{self.args.solver_a_name}={c.n_a_wins} {winner_symbol} {self.args.solver_b_name}={c.n_b_wins} "
                f"(ties={c.n_ties}, failed={c.n_both_failed}) "
                f"| avg time: {c.avg_a_duration_ms:.0f}ms vs {c.avg_b_duration_ms:.0f}ms"
            )

        # Write CSVs
        config_df = pd.DataFrame([
            {
                "config": c.config_name,
                "nodes": c.nodes,
                "pods": c.pods,
                "priorities": c.priorities,
                "util": c.util,
                "timeout_s": c.timeout,
                "n_seeds": c.n_seeds,
                f"n_{self.args.solver_a_name}_wins": c.n_a_wins,
                f"n_{self.args.solver_b_name}_wins": c.n_b_wins,
                "n_ties": c.n_ties,
                "n_both_failed": c.n_both_failed,
                f"{self.args.solver_a_name}_win_rate": format_num(c.a_win_rate, self.args.decimals),
                f"{self.args.solver_b_name}_win_rate": format_num(c.b_win_rate, self.args.decimals),
                f"avg_{self.args.solver_a_name}_duration_ms": format_num(c.avg_a_duration_ms, 1),
                f"avg_{self.args.solver_b_name}_duration_ms": format_num(c.avg_b_duration_ms, 1),
                "speed_ratio": format_num(c.speed_ratio, 2),
                "winner": c.winner,
            }
            for c in config_comparisons
        ])

        config_csv_path = out_dir / "comparison_per_config.csv"
        config_df.to_csv(config_csv_path, index=False)
        print(f"\n[ok] Wrote per-config comparison: {config_csv_path}")

        seed_df = pd.DataFrame(all_seed_comparisons)
        seed_csv_path = out_dir / "comparison_per_seed.csv"
        seed_df.to_csv(seed_csv_path, index=False)
        print(f"[ok] Wrote per-seed comparison: {seed_csv_path}")

        # Write summary
        summary = {
            "solver_a_name": self.args.solver_a_name,
            "solver_a_dir": self.args.solver_a_dir,
            "solver_b_name": self.args.solver_b_name,
            "solver_b_dir": self.args.solver_b_dir,
            "n_configs": len(config_comparisons),
            "n_seeds": total_seeds,
            f"n_{self.args.solver_a_name}_wins": total_a_wins,
            f"n_{self.args.solver_b_name}_wins": total_b_wins,
            "n_ties": total_ties,
            "n_both_failed": total_both_failed,
            f"{self.args.solver_a_name}_win_rate": total_a_wins / total_seeds if total_seeds > 0 else 0,
            f"{self.args.solver_b_name}_win_rate": total_b_wins / total_seeds if total_seeds > 0 else 0,
            "overall_winner": self.args.solver_a_name if total_a_wins > total_b_wins else (
                self.args.solver_b_name if total_b_wins > total_a_wins else "tie"
            ),
        }
        summary_path = out_dir / "comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[ok] Wrote summary: {summary_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two solvers (e.g., CP-SAT vs CBC)")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT,
                        help="Root directory for results")
    parser.add_argument("--solver-a-dir", type=str, default=SOLVER_A_DIR,
                        help="Solver A directory name (default: plugin)")
    parser.add_argument("--solver-a-name", type=str, default=SOLVER_A_NAME,
                        help="Solver A display name. Options: cp_sat, cbc (default: cp_sat)")
    parser.add_argument("--solver-b-dir", type=str, default=SOLVER_B_DIR,
                        help="Solver B directory name (default: plugin-cbc)")
    parser.add_argument("--solver-b-name", type=str, default=SOLVER_B_NAME,
                        help="Solver B display name. Options: cp_sat, cbc (default: cbc)")
    parser.add_argument("--results-csv", type=str, default=RESULTS_CSV_NAME,
                        help="Results CSV filename")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Output directory")
    parser.add_argument("--decimals", type=int, default=DECIMALS,
                        help="Decimal places for formatting")

    parsed = parser.parse_args(argv)

    args = CompareSolversArgs(
        results_root=parsed.results_root,
        solver_a_dir=parsed.solver_a_dir,
        solver_a_name=parsed.solver_a_name,
        solver_b_dir=parsed.solver_b_dir,
        solver_b_name=parsed.solver_b_name,
        results_csv=parsed.results_csv,
        out_dir=parsed.out_dir,
        decimals=parsed.decimals,
    )

    SolverComparator(args).run()


if __name__ == "__main__":
    main()
