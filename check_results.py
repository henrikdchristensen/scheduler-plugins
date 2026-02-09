#!/usr/bin/env python3
"""
Recursively find all results.csv files under a root directory and verify
that each one contains exactly 100 unique seeds.  Reports failures.

Usage:
    python scripts/check_results.py [ROOT_DIR]

If ROOT_DIR is omitted it defaults to the `analysis/` folder next to this script's
repo root.
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

    if not results_files:
        print(f"No results.csv files found under {root_dir}")
        sys.exit(0)

    failures = []
    passes = 0

    for path in results_files:
        rel = path.relative_to(root_dir)
        ok, unique, total, msg = check_results_csv(path)
        if ok:
            passes += 1
            print(f"  PASS  {rel}  ({unique} unique seeds, {total} rows)")
        else:
            failures.append((rel, unique, total, msg))
            print(f"  FAIL  {rel}  ({msg}, {total} rows)")

    print()
    print(f"Checked {len(results_files)} file(s): {passes} passed, {len(failures)} failed")

    if failures:
        print()
        print("Failures:")
        for rel, unique, total, msg in failures:
            print(f"  {rel}: {msg} (total_rows={total}, unique_seeds={unique})")
        sys.exit(1)


if __name__ == "__main__":
    main()
