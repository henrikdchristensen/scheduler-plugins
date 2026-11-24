import csv
import json
from itertools import product
from pathlib import Path

import pytest

from scripts.kwok_workload_once import cpaior_tables as tables


TARGET = tables.Configuration(4, 4, 1, 90, 1)


@pytest.fixture
def small_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep every cluster size and budget, with one workload per combination.
    densities, priorities, loads = (4,), (1,), (90,)
    monkeypatch.setattr(tables, "PODS_PER_NODE", densities)
    monkeypatch.setattr(tables, "PRIORITIES", priorities)
    monkeypatch.setattr(tables, "LOADS", loads)
    monkeypatch.setattr(tables, "EXPECTED", {
        tables.Configuration(*values)
        for values in product(tables.NODES, densities, priorities, loads, tables.BUDGETS)
    })


def write_results(path: Path, solver: str, *, missing: bool = False, duplicate: bool = False) -> None:
    rows = []
    for config in tables.EXPECTED:
        if missing and config == TARGET:
            continue
        included = improved = 0
        if config == TARGET:
            included, improved = {"cp_sat": (40, 10), "cplex": (80, 15), "gurobi": (50, 25)}[solver]
        row = {
            "nodes": config.nodes,
            "pods_per_node": config.pods_per_node,
            "priorities": config.priorities,
            "util": config.load,
            "timeout_s": config.budget,
            "total_pods": config.nodes * config.pods_per_node,
            "n_seeds": 100,
            "n_seeds_not_all_running": included,
            "n_default_all_running": 100 - included,
            "n_solver_optimal": improved,
            "n_solver_feasible": 0,
            "n_default_optimal": 0,
            "n_solver_failed": included - improved,
            "n_solver_solution": improved,
            "solver_duration_ms_sum": 500 * improved,
            **{f"{metric}_pct_sum_solver_{status}": 0
               for metric in tables.METRICS for status in ("optimal", "feasible")},
        }
        rows.append(row)
        if duplicate and config == TARGET:
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_raw_results(root: Path, *, seed_mismatch: bool = False, gurobi: bool = False) -> None:
    def write_raw(folder: Path, best: str, status: str, duration: int, running: int, offset: int = 0) -> None:
        folder.mkdir(parents=True)
        with (folder / "results.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("seed", "running_placed_by_prio_now", "unscheduled_count_before",
                             "best_solver_status", "solver_attempts"))
            attempts = [{"status": status, "duration_ms": duration}] if status else []
            for seed in range(offset, offset + 100):
                writer.writerow((seed, json.dumps({"1": running}), 1, best, json.dumps(attempts)))

    for config in tables.EXPECTED:
        name = (f"nodes{config.nodes}_pods{config.nodes * config.pods_per_node}_"
                f"prio{config.priorities}_util{int(config.load):03d}")
        job = f"{name}_timeout{config.budget:02d}"
        if config.budget == tables.BUDGETS[0]:
            write_raw(root / "default" / name, "", "", 0, 14)
        write_raw(root / "plugin" / "cp-sat" / job, "", "UNKNOWN", 20, 14)
        write_raw(root / "plugin" / "cplex" / job, "OPTIMAL", "OPTIMAL", 10, 15, int(seed_mismatch))
        if gurobi:
            write_raw(root / "plugin" / "gurobi" / job, "FEASIBLE", "FEASIBLE", 9, 16)


@pytest.mark.usefixtures("small_benchmark")
def test_generates_all_paper_tables_with_no_call_excluded(tmp_path: Path) -> None:
    for solver in tables.SOLVERS:
        write_results(tmp_path / f"results_per_combo_{solver}.csv", solver)
    write_raw_results(tmp_path)
    out = tmp_path / "paper"
    tables.main(["--results-root", str(tmp_path), "--out-dir", str(out)])
    assert {file.name for file in out.iterdir()} == {
        "tab_outcomes.tex", "tab_nodes.tex", "tab_load_ppn.tex", "tab_no_gain.tex", "tab_paired.tex",
    }
    outcomes = (out / "tab_outcomes.tex").read_text(encoding="utf-8")
    assert "MIP (Gurobi)" not in outcomes
    assert r"Improved (total) & 25.0 & -- & -- & 18.8 & -- & -- \\" in outcomes
    nodes = (out / "tab_nodes.tex").read_text(encoding="utf-8")
    assert r"\multirow{3}{*}{4} & 1 & \textbf{25.0} & 18.8 & \textbf{25.0} & 18.8 & 0.50 & 0.50 & \textbf{1} & 0 \\" in nodes
    assert r"\texttt{UNKNOWN} & 400 & 400 & 400 & 0 & 0 & 0 \\" in (out / "tab_no_gain.tex").read_text(encoding="utf-8")
    assert r"1 & All & 0 & 400 & 0 & $9.47\times10^{-29}$ & 0 & 400 \\" in (out / "tab_paired.tex").read_text(encoding="utf-8")


@pytest.mark.usefixtures("small_benchmark")
def test_rejects_unmatched_appendix_seeds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for solver in tables.SOLVERS:
        write_results(tmp_path / f"results_per_combo_{solver}.csv", solver)
    write_raw_results(tmp_path, seed_mismatch=True)
    out = tmp_path / "paper"
    with pytest.raises(SystemExit) as exc:
        tables.main(["--results-root", str(tmp_path), "--out-dir", str(out)])
    assert exc.value.code == 1
    assert "seed sets differ" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("issue", ["missing", "duplicate"])
def test_rejects_incomplete_grid_without_writing_tables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], issue: str
) -> None:
    write_results(tmp_path / "results_per_combo_cp_sat.csv", "cp_sat", **{issue: True})
    out = tmp_path / "paper"
    with pytest.raises(SystemExit) as exc:
        tables.main(["--results-root", str(tmp_path), "--out-dir", str(out)])
    assert exc.value.code == 1
    assert f"{issue} configuration" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.usefixtures("small_benchmark")
@pytest.mark.timeout(15)  # Cold Matplotlib import and PDF rendering can exceed the unit-test limit.
def test_automatically_generates_tables_and_figures_with_gurobi(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    for solver in (*tables.SOLVERS, "gurobi"):
        write_results(tmp_path / f"results_per_combo_{solver}.csv", solver)
    write_raw_results(tmp_path)
    figures = tmp_path / "figures"
    tables.main(["--results-root", str(tmp_path), "--figures", "--figures-dir", str(figures)])

    out = tmp_path / "tables" / "cpaior"
    outcomes = (out / "tab_outcomes.tex").read_text(encoding="utf-8")
    assert "MIP (Gurobi)" in outcomes
    assert r"Improved (total) & 25.0 & -- & -- & 18.8 & -- & -- & 50.0 & -- & -- \\" in outcomes
    nodes = (out / "tab_nodes.tex").read_text(encoding="utf-8")
    assert r"\textbf{50.0}" in nodes and r" & 0 & 0 & \textbf{1} \\" in nodes
    assert "MIP (Gurobi)" in (out / "tab_load_ppn.tex").read_text(encoding="utf-8")
    assert {p.name for p in figures.iterdir()} == {"outcomes_by_nodes.pdf", "runtime_proofs.pdf"}
    assert all(p.read_bytes().startswith(b"%PDF-") for p in figures.iterdir())


def test_rejects_incomplete_gurobi_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for solver in tables.SOLVERS:
        write_results(tmp_path / f"results_per_combo_{solver}.csv", solver)
    write_results(tmp_path / "results_per_combo_gurobi.csv", "gurobi", missing=True)
    with pytest.raises(SystemExit) as exc:
        tables.main(["--results-root", str(tmp_path)])
    assert exc.value.code == 1
    assert "results_per_combo_gurobi.csv: 1 missing configurations" in capsys.readouterr().err
    assert not (tmp_path / "tables").exists()


@pytest.mark.usefixtures("small_benchmark")
def test_gurobi_primary_keeps_cplex_in_appendix(tmp_path: Path) -> None:
    for solver in (*tables.SOLVERS, "gurobi"):
        write_results(tmp_path / f"results_per_combo_{solver}.csv", solver)
    write_raw_results(tmp_path, gurobi=True)
    out = tmp_path / "paper"
    tables.main(["--results-root", str(tmp_path), "--out-dir", str(out), "--mip", "gurobi"])
    for name in ("tab_outcomes", "tab_no_gain"):
        text = (out / f"{name}.tex").read_text(encoding="utf-8")
        assert "Gurobi" in text and "CPLEX" not in text
    paired = (out / "tab_paired.tex").read_text(encoding="utf-8")
    assert r"1 & All & 0 & 400 & 0" in paired
    appendix = (out / "tab_mip_comparison.tex").read_text(encoding="utf-8")
    assert r"1 & 50.0 & 50.0 & 0.50 & 18.8 & 18.8 & 0.50 & 400 & 0 & 0 \\" in appendix


@pytest.mark.timeout(15)
def test_figures_use_the_correct_denominators_and_runtime() -> None:
    pytest.importorskip("matplotlib")
    from scripts.kwok_workload_once import cpaior_figures

    row = tables.Outcome(100, 50, 25, 0, 0, 25, 25, 12500, 0, 0, 0)
    groups = {(node, budget): row for node in tables.NODES for budget in tables.BUDGETS}
    summaries = {"cp_sat": groups, "cplex": groups, "gurobi": groups}

    *_, gurobi = cpaior_figures.plot_outcomes(summaries).axes
    assert gurobi.get_title() == "MIP (Gurobi)"
    assert gurobi.containers[0][0].get_height() == 25  # Of all 100 runs.
    proof, runtime = cpaior_figures.plot_runtime(summaries).axes
    assert proof.containers[6][0].get_height() == 50  # Of the 50 runs excluding No call.
    assert runtime.containers[6][0].get_height() == .5
