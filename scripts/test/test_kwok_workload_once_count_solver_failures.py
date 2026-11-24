import csv
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts.kwok_workload_once import count_solver_failures as audit


def test_rerun_jobs_select_only_matched_closed_stdout_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    solver, config = "cp-sat", "nodes16_pods128_prio1_util090_timeout01"
    folder = Path("results", "plugin", solver, config)
    logs = folder / "scheduler-logs"
    logs.mkdir(parents=True)
    errors = {
        "11": "read solver stdout: read |0: file already closed",
        "22": "solver context: context deadline exceeded",
        "33": "read solver stdout: other error",
        "44": "read solver stdout: read |0: file already closed",
        "55": "read solver stdout: read |0: file already closed",
    }
    with (folder / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "solver_attempts"])
        for seed, error in errors.items():
            status = "UNKNOWN" if seed == "44" else "FAILED"
            writer.writerow([seed, json.dumps([{"status": status, "name": "python", "duration_ms": 20}])])
            duration = 30 if seed == "44" else 20
            (logs / f"sched_logs_seed-{seed}_run-1.log").write_text(
                f'solver failed" solver="python" status="FAILED" err="{error}" durationMs={duration}\n',
                encoding="utf-8",
            )

    template = audit.JOBS_DIR / "plugin" / f"{config}.yaml"
    template.parent.mkdir(parents=True)
    seed_file = Path("data", "original-seeds.txt")
    seed_file.write_text("55\n44\n33\n22\n11\n", encoding="utf-8")
    original = {
        "seed-file": seed_file.as_posix(), "output-dir": "results/plugin/{solver}/original",
        "override-kwokctl-envs": [{"name": "SOLVER_PYTHON_TIMEOUT", "value": "1s"}],
    }
    template.write_text(yaml.safe_dump(original), encoding="utf-8")

    counts: Counter[str] = Counter()
    issues: list[str] = []
    seeds = audit.audit_job(folder, counts, issues)
    assert not issues
    assert seeds == {"11", "55"}
    assert counts["deadline"] == 1 and counts["stdout"] == 3
    audit.write_rerun_jobs({(solver, config): seeds})

    rerun = audit.RERUN_JOBS / solver
    assert (rerun / "jobs.txt").read_text(encoding="utf-8").splitlines() == [f"{config}.yaml"]
    job = yaml.safe_load((rerun / f"{config}.yaml").read_text(encoding="utf-8"))
    assert job == {**original, "seed-file": (rerun / f"{config}.txt").as_posix(),
                   "output-dir": f"results/rerun-stdout/{{solver}}/{config}"}
    assert Path(job["seed-file"]).read_text(encoding="utf-8").splitlines() == ["55", "11"]
    with pytest.raises(ValueError, match="already exists"):
        audit.write_rerun_jobs({(solver, config): seeds})
