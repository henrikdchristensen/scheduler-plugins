#!/usr/bin/env python3
"""
check_general_stats.py

Validate a directory structure like:

ROOT/
  experiment_A/
    seed_1/
      general_stats.csv
    seed_2/
      general_stats.csv
    seed_3/
      general_stats.csv
  experiment_B/
    ...

Rules per experiment directory (immediate child of ROOT):
  - Must contain at least --min-seeds seed folders (default 3).
  - Each seed folder must contain general_stats.csv (or general_stats.csv.gz if --allow-gz).
  - Each general_stats file must contain a numeric 'time_s' column with
    --threshold-min <= max(time_s) <= --threshold-max.

Usage:
  python check_general_stats.py /path/to/root
  python check_general_stats.py /path/to/root --threshold-min 7190 --threshold-max 7210 --min-seeds 3 --fail-only
  python check_general_stats.py /path/to/root --allow-gz --write-report report.csv

Exit codes:
  0 = all experiment dirs pass (or none found)
  1 = at least one experiment dir fails
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class SeedCheck:
    seed_dir: str
    ok: bool
    stats_path: Optional[str]
    max_time_s: Optional[float]
    reason: str


@dataclass
class ExperimentCheck:
    experiment_dir: str
    ok: bool
    seed_dirs_found: int
    seeds_checked: int
    seeds_ok: int
    reason: str
    seed_results: List[SeedCheck]


def open_maybe_gzip(path: Path) -> io.TextIOBase:
    if path.suffix.lower() == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def check_time_s_csv(path: Path, threshold_min: float, threshold_max: float, time_col: str = "time_s") -> tuple[bool, Optional[float], str]:
    """
    Stream-read a CSV and return:
      ok, max_time, reason_if_fail
    """
    try:
        with open_maybe_gzip(path) as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False, None, "empty or missing header"

            if time_col not in reader.fieldnames:
                return False, None, f"missing column '{time_col}' (found: {', '.join(reader.fieldnames)})"

            max_time: Optional[float] = None
            for row in reader:
                raw = row.get(time_col, "")
                if raw is None:
                    continue
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if (max_time is None) or (v > max_time):
                    max_time = v

            if max_time is None:
                return False, None, f"no numeric '{time_col}' values"
            if max_time < threshold_min:
                return False, max_time, f"max {time_col} ({max_time}) < {threshold_min}"
            if max_time > threshold_max:
                return False, max_time, f"max {time_col} ({max_time}) > {threshold_max}"
            return True, max_time, ""
    except Exception as e:
        return False, None, f"error: {type(e).__name__}: {e}"


def find_general_stats(seed_dir: Path, allow_gz: bool) -> Optional[Path]:
    p = seed_dir / "general_stats.csv"
    if p.is_file():
        return p
    if allow_gz:
        pgz = seed_dir / "general_stats.csv.gz"
        if pgz.is_file():
            return pgz
    return None


def check_experiment_dir(exp_dir: Path, min_seeds: int, threshold_min: float, threshold_max: float, allow_gz: bool, time_col: str) -> ExperimentCheck:
    seed_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir()])
    seed_results: List[SeedCheck] = []

    # Check all seeds first, even if we don't have enough
    seeds_ok = 0
    for sd in seed_dirs:
        stats = find_general_stats(sd, allow_gz=allow_gz)
        if stats is None:
            seed_results.append(
                SeedCheck(
                    seed_dir=str(sd),
                    ok=False,
                    stats_path=None,
                    max_time_s=None,
                    reason="missing general_stats.csv" + ("/.gz" if allow_gz else ""),
                )
            )
            continue

        ok, max_time, reason = check_time_s_csv(stats, threshold_min=threshold_min, threshold_max=threshold_max, time_col=time_col)
        if ok:
            seeds_ok += 1
        seed_results.append(
            SeedCheck(
                seed_dir=str(sd),
                ok=ok,
                stats_path=str(stats),
                max_time_s=max_time,
                reason=reason,
            )
        )

    exp_ok = (len(seed_dirs) >= min_seeds) and (seeds_ok >= min_seeds) and all(r.ok for r in seed_results)
    # Note: if there are >min_seeds dirs, we still require *all* seed dirs to pass.
    # If you instead want "any 3 seeds pass", I can adjust.
    if not exp_ok:
        if len(seed_dirs) < min_seeds:
            reason = f"found {len(seed_dirs)} seed dirs, expected at least {min_seeds}"
        else:
            bad = [r for r in seed_results if not r.ok]
            reason = f"{len(bad)}/{len(seed_results)} seed(s) failed"
    else:
        reason = ""

    return ExperimentCheck(
        experiment_dir=str(exp_dir),
        ok=exp_ok,
        seed_dirs_found=len(seed_dirs),
        seeds_checked=len(seed_results),
        seeds_ok=seeds_ok,
        reason=reason,
        seed_results=seed_results,
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Root directory containing experiment subdirectories")
    ap.add_argument("--min-seeds", type=int, default=5, help="Minimum number of seed folders required per experiment")
    ap.add_argument("--threshold-min", type=float, default=7190.0, help="Require max(time_s) >= threshold-min")
    ap.add_argument("--threshold-max", type=float, default=7220.0, help="Require max(time_s) <= threshold-max (default: inf)")
    ap.add_argument("--time-col", default="time_s", help="Time column name (default: time_s)")
    ap.add_argument("--allow-gz", action="store_true", help="Also accept general_stats.csv.gz")
    ap.add_argument("--fail-only", action="store_true", help="Only print failing experiments / seeds")
    ap.add_argument("--write-report", type=Path, default=None, help="Write a CSV report to this path")
    args = ap.parse_args(argv)

    root = args.root
    if not root.exists() or not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 1

    exp_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not exp_dirs:
        print(f"No experiment subdirectories found under: {root}")
        return 0

    checks: list[ExperimentCheck] = []
    any_fail = False

    for exp in exp_dirs:
        chk = check_experiment_dir(
            exp_dir=exp,
            min_seeds=args.min_seeds,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            allow_gz=args.allow_gz,
            time_col=args.time_col,
        )
        checks.append(chk)
        if not chk.ok:
            any_fail = True

        if args.fail_only and chk.ok:
            continue

        status = "OK" if chk.ok else "FAIL"
        print(f"{status}\t{exp}\tseeds={chk.seed_dirs_found}\t(seeds_ok={chk.seeds_ok})" + (f"\t{chk.reason}" if chk.reason else ""))

        # Print seed details if failing (or if not fail-only)
        if (not chk.ok) or (not args.fail_only):
            for sr in chk.seed_results:
                if args.fail_only and sr.ok:
                    continue
                sstatus = "OK" if sr.ok else "FAIL"
                max_str = "" if sr.max_time_s is None else f"{sr.max_time_s:.6f}"
                spath = sr.stats_path or "-"
                msg = f"  {sstatus}\t{Path(sr.seed_dir).name}\t{spath}\tmax_{args.time_col}={max_str}"
                if not sr.ok and sr.reason:
                    msg += f"\t({sr.reason})"
                print(msg)

    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        with args.write_report.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["experiment_dir", "experiment_ok", "seed_dirs_found", "seeds_checked", "seeds_ok", "experiment_reason",
                        "seed_dir", "seed_ok", "stats_path", "max_time_s", "seed_reason"])
            for chk in checks:
                if not chk.seed_results:
                    w.writerow([chk.experiment_dir, int(chk.ok), chk.seed_dirs_found, chk.seeds_checked, chk.seeds_ok, chk.reason,
                                "", "", "", "", ""])
                    continue
                for sr in chk.seed_results:
                    w.writerow([
                        chk.experiment_dir,
                        int(chk.ok),
                        chk.seed_dirs_found,
                        chk.seeds_checked,
                        chk.seeds_ok,
                        chk.reason,
                        sr.seed_dir,
                        int(sr.ok),
                        sr.stats_path or "",
                        "" if sr.max_time_s is None else sr.max_time_s,
                        sr.reason,
                    ])
        print(f"Wrote report: {args.write_report}")

    if any_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
