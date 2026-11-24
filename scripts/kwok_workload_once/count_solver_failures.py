"""Audit recorded solver attempts against their per-seed scheduler logs.

python -m scripts.kwok_workload_once.count_solver_failures results/plugin/cp-sat results/plugin/cplex
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


JOB_BUDGET = re.compile(r"_timeout([0-9]+)$")
LOG_FAILURE = re.compile(
    r'solver failed" solver="(?P<solver>[^"\s]+)" status="FAILED" '
    r'err="(?P<error>(?:\\.|[^"\\])*)" durationMs=(?P<duration>[0-9]+)\b'
)
COLUMNS = ("calls", "UNKNOWN", "FAILED", "deadline", "stdout", "other", "unresolved")
JOBS_DIR = Path("data", "jobs", "kwok_workload_once")
RERUN_JOBS = JOBS_DIR / "rerun-stdout"


def failure_category(error: str) -> str:
    if error.endswith("solver context: context deadline exceeded"):
        return "deadline"
    if "read solver stdout:" in error:
        return "stdout"
    return "other"


def audit_job(folder: Path, counts: Counter[str], issues: list[str]) -> set[str]:
    rerun_seeds: set[str] = set()
    with (folder / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"seed", "solver_attempts"}.issubset(rows.fieldnames or []):
            raise ValueError("results.csv must contain seed and solver_attempts")

        seeds = set()
        for row in rows:
            seed = row["seed"]
            if not seed or not seed.isascii() or not seed.isdecimal() or seed in seeds:
                raise ValueError(f"invalid or duplicate seed at CSV line {rows.line_num}: {seed!r}")
            seeds.add(seed)
            cell = row["solver_attempts"]
            if cell is None:
                raise ValueError(f"missing solver_attempts cell for seed {seed}")
            attempts = json.loads(cell) if cell else []
            if attempts is None:
                attempts = []
            if not isinstance(attempts, list):
                raise ValueError(f"solver_attempts must be a list for seed {seed}")

            failed = Counter()
            for attempt in attempts:
                if not isinstance(attempt, dict) or not isinstance(attempt.get("status"), str):
                    raise ValueError(f"invalid solver attempt for seed {seed}")
                status = attempt["status"]
                counts["calls"] += 1
                if status in {"UNKNOWN", "FAILED"}:
                    counts[status] += 1
                if status == "FAILED":
                    name, duration = attempt.get("name"), attempt.get("duration_ms")
                    if not isinstance(name, str) or type(duration) is not int:
                        raise ValueError(f"failed attempt lacks a solver name or duration for seed {seed}")
                    failed[name, duration] += 1

            if not failed:
                continue
            logged = defaultdict(list)
            for log in (folder / "scheduler-logs").glob(f"sched_logs_seed-{seed}_run-*.log"):
                for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                    match = LOG_FAILURE.search(line)
                    if match:
                        key = match["solver"], int(match["duration"])
                        logged[key].append(match["error"])
            for (name, duration), attempt_count in failed.items():
                errors = logged[name, duration]
                if attempt_count == len(errors) == 1:
                    category = failure_category(errors[0])
                    counts[category] += 1
                    if category == "stdout" and errors[0].endswith("file already closed"):
                        rerun_seeds.add(seed)
                else:
                    counts["unresolved"] += attempt_count
                    issues.append(
                        f"{folder}: seed {seed}, {name}, {duration} ms: "
                        f"{attempt_count} failed attempt(s), {len(errors)} matching log entries"
                    )

        if not seeds:
            raise ValueError("results.csv has no seed rows")
    return rerun_seeds


def write_rerun_jobs(selected: dict[tuple[str, str], set[str]]) -> None:
    import yaml

    if not selected:
        raise ValueError("no matched closed-stdout failures to rerun")
    if RERUN_JOBS.exists():
        raise ValueError(f"{RERUN_JOBS} already exists; move it aside before regenerating")

    files: dict[Path, str] = {}
    job_lists: dict[str, list[str]] = defaultdict(list)
    seed_counts: Counter[str] = Counter()
    for (solver, config), seeds in sorted(selected.items()):
        template = JOBS_DIR / "plugin" / f"{config}.yaml"
        try:
            job = yaml.safe_load(template.read_text(encoding="utf-8-sig"))
        except yaml.YAMLError as exc:
            raise ValueError(f"{template}: invalid YAML: {exc}") from exc
        if not isinstance(job, dict) or not isinstance(job.get("seed-file"), str):
            raise ValueError(f"{template}: expected a job with a seed-file")
        if any(key in job for key in ("seed", "count", "repeats", "seeds-not-all-running")):
            raise ValueError(f"{template}: expected one run per entry in the seed file")
        original_seeds = Path(job["seed-file"]).read_text(encoding="utf-8-sig").split()
        subset = [seed for seed in original_seeds if seed in seeds]
        if len(subset) != len(seeds) or set(subset) != seeds:
            raise ValueError(f"{template}: rerun seeds do not match the original seed file")

        job_dir = RERUN_JOBS / solver
        seed_file = job_dir / f"{config}.txt"
        files[seed_file] = "\n".join(subset) + "\n"
        job["seed-file"] = seed_file.as_posix()
        job["output-dir"] = (Path("results", "rerun-stdout", "{solver}") / config).as_posix()
        job_file = job_dir / f"{config}.yaml"
        files[job_file] = yaml.safe_dump(job, sort_keys=False)
        job_lists[solver].append(job_file.name)
        seed_counts[solver] += len(seeds)

    RERUN_JOBS.mkdir(parents=True, exist_ok=False)
    for solver, jobs in job_lists.items():
        job_list = RERUN_JOBS / solver / "jobs.txt"
        job_list.parent.mkdir()
        files[job_list] = "\n".join(jobs) + "\n"
    for path, text in files.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    for solver, jobs in job_lists.items():
        print(f"{RERUN_JOBS / solver / 'jobs.txt'}: {len(jobs)} jobs, {seed_counts[solver]} seeds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("solver_dirs", metavar="SOLVER_DIR", type=Path, nargs="+")
    parser.add_argument("--write-stdout-reruns", action="store_true",
                        help=f"write bootstrap job lists for closed-stdout failures to {RERUN_JOBS}")
    args = parser.parse_args()
    root_names = [root.name for root in args.solver_dirs]
    if args.write_stdout_reruns and (
        not set(root_names) <= {"cp-sat", "cplex"} or len(set(root_names)) != len(root_names)
    ):
        parser.error("--write-stdout-reruns requires distinct solver directories named cp-sat or cplex")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    issues: list[str] = []
    selected: dict[tuple[str, str], set[str]] = {}
    print("\t".join(("solver", "budget_s", *COLUMNS)))
    for root in args.solver_dirs:
        by_budget: dict[int, Counter[str]] = defaultdict(Counter)
        for folder in sorted(root.iterdir()) if root.is_dir() else []:
            budget = JOB_BUDGET.search(folder.name)
            if not folder.is_dir() or not budget:
                continue
            counts: Counter[str] = Counter()
            try:
                seeds = audit_job(folder, counts, issues)
            except (OSError, ValueError, csv.Error) as exc:
                issues.append(f"{folder}: {exc}; job excluded from totals")
                continue
            if args.write_stdout_reruns and seeds:
                selected[root.name, folder.name] = seeds
            by_budget[int(budget[1])].update(counts)
        if not by_budget:
            issues.append(f"{root}: no readable job results found")
            continue
        total: Counter[str] = Counter()
        for budget, counts in sorted(by_budget.items()):
            print("\t".join((root.name, str(budget), *(str(counts[key]) for key in COLUMNS))))
            total.update(counts)
        print("\t".join((root.name, "all", *(str(total[key]) for key in COLUMNS))))

    for issue in issues:
        print(issue, file=sys.stderr)
    if args.write_stdout_reruns and not issues:
        try:
            write_rerun_jobs(selected)
        except (OSError, ValueError, ImportError) as exc:
            print(f"cannot write rerun jobs: {exc}", file=sys.stderr)
            return 1
    return int(bool(issues))


if __name__ == "__main__":
    sys.exit(main())