#!/usr/bin/env python3
"""
rename_timeouts.py

Rename immediate subdirectories (no recursion) by replacing a trailing
'_timeout<NUMBER>' with '_timeout<TARGET>'.

Example:
  nodes4_pods16_prio1_util095_timeout20  ->  nodes4_pods16_prio1_util095_timeout60

Usage:
  python rename_timeouts.py /path/to/dir --to 60
  python rename_timeouts.py /path/to/dir --to 60 --dry-run
  python rename_timeouts.py /path/to/dir --to 60 --from 20   # only rename timeout20 -> timeout60

Notes:
  - Only renames folders directly under the given directory.
  - Skips if no trailing timeout pattern is found.
  - Skips (and reports) if the target name already exists.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional


TIMEOUT_RE = re.compile(r"^(?P<prefix>.*)_timeout(?P<timeout>\d+)$")


def compute_new_name(name: str, target: int, only_from: Optional[int]) -> Optional[str]:
    m = TIMEOUT_RE.match(name)
    if not m:
        return None
    old = int(m.group("timeout"))
    if only_from is not None and old != only_from:
        return None
    if old == target:
        return None
    return f"{m.group('prefix')}_timeout{target}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path, help="Directory whose immediate subfolders will be renamed")
    ap.add_argument("--to", type=int, required=True, help="Target timeout value (e.g., 60)")
    ap.add_argument("--from", dest="only_from", type=int, default=None, help="Only rename folders with this timeout value")
    ap.add_argument("--dry-run", action="store_true", help="Print what would change, but do not rename")
    args = ap.parse_args(argv)

    root = args.dir
    if not root.exists() or not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        return 1

    renamed = 0
    skipped = 0
    collisions = 0

    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue

        new_name = compute_new_name(p.name, target=args.to, only_from=args.only_from)
        if new_name is None:
            skipped += 1
            continue

        dst = p.with_name(new_name)
        if dst.exists():
            print(f"SKIP (exists)\t{p.name} -> {dst.name}")
            collisions += 1
            continue

        if args.dry_run:
            print(f"DRY\t\t{p.name} -> {dst.name}")
        else:
            p.rename(dst)
            print(f"RENAMED\t\t{p.name} -> {dst.name}")
        renamed += 1

    print(f"\nSummary: renamed={renamed}, skipped={skipped}, collisions={collisions}")
    return 0 if collisions == 0 else 1

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))