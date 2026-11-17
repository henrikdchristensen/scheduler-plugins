#!/usr/bin/env python3
# scripts/kwok_workload_once/check_results.py
"""
python -m scripts.kwok_workload_once.check_results results/plugin
python -m scripts.kwok_workload_once.check_results results/plugin \
    --job-list data/jobs/kwok_workload_once/lists/cp_sat/*.txt --commit <hash>
"""

import argparse, csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

csv.field_size_limit(sys.maxsize)

EXPECTED_ERRORS = {"", "no pending pods", "no improving solution from any solver"}
EXPECTED_STATUSES = {"", "OPTIMAL", "FEASIBLE", "UNKNOWN"}
USABLE_STATUSES = {"OPTIMAL", "FEASIBLE"}
DURATION_SLACK_MS = 2000


def placed_by_priority(cell: str) -> dict[int, int]:
    """Parse {"p1": 3}, {"1": 3} or a solver score into {priority: count}, dropping zeros."""
    try:
        doc = json.loads(cell) if cell else {}
    except json.JSONDecodeError:
        return {}
    if not isinstance(doc, dict):
        return {}
    if "placed_by_priority" in doc or "evicted" in doc or "moved" in doc:
        doc = doc.get("placed_by_priority") or {}
    return {int(str(k).lstrip("p")): int(v) for k, v in doc.items() if int(v)}


def check_folder(folder: Path, optimal_top: dict) -> tuple[dict, list[str]]:
    """Check one result folder. Returns (stats, problems) and records optimal runs in optimal_top."""
    problems: list[str] = []
    stats = {"seeds": 0, "expected": None, "status": Counter(), "usable": 0, "mismatch": 0, "commit": "", "solver": ""}

    info_file, results_file = folder / "info.yaml", folder / "results.csv"
    info = yaml.safe_load(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    stats["commit"] = ((info.get("meta") or {}).get("git") or {}).get("commit", "")
    inputs = info.get("inputs") or {}
    solver = re.search(r"--solver-type\s+(\S+)", str(inputs.get("cli-cmd", "")))
    stats["solver"] = solver.group(1) if solver else ""
    if not stats["commit"]:
        problems.append("no git commit recorded in info.yaml")
    elif stats["commit"].endswith("-dirty"):
        problems.append(f"built from uncommitted changes ({stats['commit']})")

    if not results_file.exists():
        return stats, problems + ["results.csv missing"]
    with open(results_file, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Seeds: exactly those of the job's seed file, each once.
    seeds = [r["seed"] for r in rows]
    stats["seeds"] = len(set(seeds))
    if len(seeds) != len(set(seeds)):
        problems.append(f"{len(seeds) - len(set(seeds))} duplicate seed rows")
    seed_file = Path(str((inputs.get("job") or {}).get("seed-file", "")))
    if seed_file.is_file():
        expected = set(seed_file.read_text(encoding="utf-8").split())
        stats["expected"] = len(expected)
        if expected - set(seeds):
            problems.append(f"{len(expected - set(seeds))} seeds missing")
        if set(seeds) - expected:
            problems.append(f"{len(set(seeds) - expected)} seeds not in {seed_file.name}")
    else:
        problems.append("seed file not found, seeds not verified")

    config, _, timeout = folder.name.partition("_timeout")
    timeout_ms = int(timeout) * 1000 if timeout.isdigit() else None
    errors, slow, fewer = Counter(), 0, 0
    for r in rows:
        status, error, score = r["best_solver_status"], r["error"], r["best_solver_score"]
        stats["status"][status] += 1
        errors[error] += 1
        if timeout_ms and r["best_solver_duration_ms"] and float(r["best_solver_duration_ms"]) > timeout_ms + DURATION_SLACK_MS:
            slow += 1
        promised = placed_by_priority(score) if score else None
        if status in USABLE_STATUSES and not error and promised is not None:
            # The cluster must end up where the plan said it would.
            now = placed_by_priority(r["running_placed_by_prio_now"])
            stats["usable"] += 1
            if now != promised:
                stats["mismatch"] += 1
                fewer += sum(now.values()) < sum(promised.values())
        if status == "OPTIMAL" and promised is not None:
            workload = placed_by_priority(r["running_placed_by_prio_before"]) | placed_by_priority(r["unschedulable_by_prio_before"])
            if workload:
                optimal_top[(config, r["seed"])].append((folder.name, promised.get(max(workload), 0)))

    for text, n in errors.items():
        if text not in EXPECTED_ERRORS:
            problems.append(f"{n} rows with error '{text[:60]}'")
    for status, n in stats["status"].items():
        if status not in EXPECTED_STATUSES:
            problems.append(f"{n} rows with status {status}")
    if stats["mismatch"]:
        problems.append(f"{stats['mismatch']} of {stats['usable']} applied plans: final state differs from the plan ({fewer} with fewer pods running)")
    if slow:
        problems.append(f"{slow} solver runs exceeded the timeout by more than {DURATION_SLACK_MS // 1000} s")
    return stats, problems


def expected_folders(job_lists: list[Path]) -> set[str]:
    names: set[str] = set()
    for job_list in job_lists:
        for line in job_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(Path(line).stem)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check kwok_workload_once results for completeness and consistency")
    parser.add_argument("results_dirs", nargs="+", type=Path, help="Folders holding one sub-folder per job, e.g. results/plugin")
    parser.add_argument("--job-list", nargs="*", type=Path, default=[], help="Job lists whose jobs must all be present")
    parser.add_argument("--commit", default=None, help="Git commit all results must have been produced with")
    args = parser.parse_args(argv)

    optimal_top: dict = defaultdict(list)
    problems: dict[str, list[str]] = {}
    found: set[str] = set()
    commits, solvers, totals = Counter(), Counter(), Counter()

    print(f"{'folder':46} {'seeds':>9} {'optimal':>8} {'feasible':>8} {'unknown':>8} {'plan!=final':>12}")
    for results_dir in args.results_dirs:
        for folder in sorted(p for p in results_dir.iterdir() if p.is_dir()):
            stats, folder_problems = check_folder(folder, optimal_top)
            found.add(folder.name)
            commits[stats["commit"]] += 1
            solvers[stats["solver"]] += 1
            totals.update({"usable": stats["usable"], "mismatch": stats["mismatch"]})
            if folder_problems:
                problems[str(folder)] = folder_problems
            seeds = f"{stats['seeds']}/{stats['expected'] if stats['expected'] is not None else '?'}"
            status = stats["status"]
            print(f"{folder.name:46} {seeds:>9} {status['OPTIMAL']:>8} {status['FEASIBLE']:>8} {status['UNKNOWN']:>8} "
                  f"{str(stats['mismatch']) + '/' + str(stats['usable']):>12}")

    general: list[str] = []
    if args.job_list:
        missing = sorted(expected_folders(args.job_list) - found)
        if missing:
            general.append(f"{len(missing)} expected job folders missing, e.g. {', '.join(missing[:3])}")
    if len(commits) > 1:
        general.append(f"results come from {len(commits)} different commits: {dict(commits)}")
    if args.commit and set(commits) != {args.commit}:
        general.append(f"commit differs from {args.commit}: {dict(commits)}")
    # Runs proven optimal on the same seed must agree on the top-priority count: it depends on the workload only.
    for (config, seed), runs in optimal_top.items():
        if len({count for _, count in runs}) > 1:
            general.append(f"optimal runs disagree for {config} seed {seed}: {runs}")

    print(f"\nfolders: {len(found)}   commits: {dict(commits)}   solvers: {dict(solvers)}")
    print(f"applied plans whose final state differs from the plan: {totals['mismatch']} of {totals['usable']}")
    print(f"seeds with at least two optimal runs compared: {sum(1 for runs in optimal_top.values() if len(runs) > 1)}")
    if not problems and not general:
        print("no problems found")
        return 0
    print("\nproblems:")
    for line in general:
        print(f"  - {line}")
    for folder, lines in problems.items():
        for line in lines:
            print(f"  - {folder}: {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
