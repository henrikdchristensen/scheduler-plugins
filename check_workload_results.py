#!/usr/bin/env python3
"""
python check_workload_results.py analysis/kwok_workload_once/plugin-gurobi/
"""

import csv
import sys
from pathlib import Path


EXPECTED_UNIQUE_SEEDS = 100


def check_results_csv(path: Path) -> tuple[bool, int, int, str]:
    """
    Returns (ok, unique_seeds, total_rows, message).
    """
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if "seed" not in (reader.fieldnames or []):
                return False, 0, 0, "missing 'seed' column"
            seeds: list[str] = []
            for row in reader:
                seeds.append(row.get("seed", "").strip())
            total_rows = len(seeds)
            unique_seeds = len(set(seeds))
            if unique_seeds == EXPECTED_UNIQUE_SEEDS:
                return True, unique_seeds, total_rows, "OK"
            else:
                return False, unique_seeds, total_rows, f"expected {EXPECTED_UNIQUE_SEEDS} unique seeds, got {unique_seeds}"
    except Exception as e:
        return False, 0, 0, f"error reading file: {e}"


def main():
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "analysis"

    if not root_dir.is_dir():
        print(f"ERROR: {root_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    results_files = sorted(root_dir.rglob("results.csv"))

    # Find subdirectories that have no results.csv (empty/incomplete runs)
    all_subdirs = sorted(
        d for d in root_dir.rglob("*")
        if d.is_dir() and not any(d.iterdir())  # completely empty
        or (d.is_dir() and not (d / "results.csv").exists() and not any(c.is_dir() for c in d.iterdir()))
    )
    # Filter out dirs that are parents of other dirs (only leaf dirs matter)
    missing_dirs = [
        d for d in all_subdirs
        if d.is_dir() and not (d / "results.csv").exists()
    ]

    if not results_files and not missing_dirs:
        print(f"No results.csv files found under {root_dir}")
        sys.exit(0)

    failures = []
    passes = 0

    for d in missing_dirs:
        rel = d.relative_to(root_dir)
        failures.append((rel, 0, 0, "missing results.csv"))
        print(f"  FAIL  {rel}  (missing results.csv)")

    for path in results_files:
        rel = path.relative_to(root_dir)
        ok, unique, total, msg = check_results_csv(path)
        if ok:
            passes += 1
            print(f"  PASS  {rel}  ({unique} unique seeds, {total} rows)")
        else:
            failures.append((rel, unique, total, msg))
            print(f"  FAIL  {rel}  ({msg}, {total} rows)")

    total_checked = len(results_files) + len(missing_dirs)
    print()
    print(f"Checked {total_checked} location(s): {passes} passed, {len(failures)} failed")

    if failures:
        print()
        print("Failures:")
        for rel, unique, total, msg in failures:
            print(f"  {rel}: {msg} (total_rows={total}, unique_seeds={unique})")
        sys.exit(1)


if __name__ == "__main__":
    main()
