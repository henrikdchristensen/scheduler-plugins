#!/usr/bin/env python3
"""
check_results_rows.py

Recursively scan a root directory for files named 'results.csv' and verify each
has exactly N data rows (default: 101). Header is NOT counted.

Examples:
  python check_results_rows.py /path/to/root
  python check_results_rows.py /path/to/root --rows 101 --fail-only
  python check_results_rows.py /path/to/root --rows 101 --write-report report.csv
  python check_results_rows.py /path/to/root --include-gz   # also check results.csv.gz

Exit codes:
  0 = all matched files have the expected row count (or no matches found)
  1 = at least one file failed (wrong row count, unreadable, etc.)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class CheckResult:
    path: str
    ok: bool
    rows: Optional[int]
    reason: str


def open_maybe_gzip(path: Path) -> io.TextIOBase:
    if path.suffix.lower() == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def count_csv_data_rows(path: Path) -> int:
    """
    Count data rows in a CSV (header excluded).
    Uses csv.reader for correctness (handles quoted newlines).
    """
    with open_maybe_gzip(path) as f:
        reader = csv.reader(f)
        # Read header (must exist)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Root directory to scan recursively")
    ap.add_argument("--rows", type=int, default=101, help="Expected number of data rows (default: 101)")
    ap.add_argument("--fail-only", action="store_true", help="Only print failing files")
    ap.add_argument("--write-report", type=Path, default=None, help="Write a CSV report to this path")
    ap.add_argument("--include-gz", action="store_true", help="Also match results.csv.gz")
    args = ap.parse_args(argv)

    root = args.root
    if not root.exists() or not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        return 1

    target_names = {"results.csv"}
    if args.include_gz:
        target_names.add("results.csv.gz")

    results: List[CheckResult] = []
    matched = 0

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name not in target_names:
                continue
            matched += 1
            path = Path(dirpath) / name
            try:
                n = count_csv_data_rows(path)
                ok = (n == args.rows)
                reason = "" if ok else f"expected {args.rows}, got {n}"
                results.append(CheckResult(str(path), ok, n, reason))
            except Exception as e:
                results.append(CheckResult(str(path), False, None, f"error: {type(e).__name__}: {e}"))

    if matched == 0:
        print(f"No files named {', '.join(sorted(target_names))} found under: {root}")
        return 0

    failed = 0
    for r in results:
        if args.fail_only and r.ok:
            continue
        status = "OK" if r.ok else "FAIL"
        row_str = "" if r.rows is None else str(r.rows)
        msg = f"{status}\t{r.path}\trows={row_str}"
        if not r.ok and r.reason:
            msg += f"\t({r.reason})"
        print(msg)
        if not r.ok:
            failed += 1

    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        with args.write_report.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "ok", "rows", "reason"])
            for r in results:
                w.writerow([r.path, int(r.ok), "" if r.rows is None else r.rows, r.reason])
        print(f"Wrote report: {args.write_report}")

    if failed:
        print(f"\nFAILED: {failed}/{len(results)} files had the wrong row count")
        return 1

    print(f"\nPASSED: {len(results)}/{len(results)} files have exactly {args.rows} rows")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
