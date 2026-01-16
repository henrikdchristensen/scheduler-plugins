#!/usr/bin/env python3
# test_seal_results.py

import pytest

import csv, math
from pathlib import Path
import pandas as pd

from scripts.kwok_workload_once import seal_results as cr

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "seed",
    "util_run_cpu_now",
    "util_run_mem_now",
    "running_placed_by_prio_now",
    "unscheduled_count_before",
    "error",
    "best_solver_status",
    "best_solver_name",
    "best_solver_duration_ms",
    "best_solver_score",
]


def _write_results_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num,den,expect_nan,expect_value",
    [
        (2, 4, False, 0.5),
        (0, 4, False, 0.0),
        (1, 0, True, None),
        (1, None, True, None),
    ],
)
def test_rate(num, den, expect_nan: bool, expect_value: float | None):
    got = cr.rate(num, den)
    if expect_nan:
        assert math.isnan(got)
    else:
        assert got == expect_value


# ---------------------------------------------------------------------------
# parse_solver_dirname
# ---------------------------------------------------------------------------

def test_parse_solver_dirname_invalid_returns_none():
    assert cr.parse_solver_dirname("not-a-match") is None
    assert cr.parse_solver_dirname("") is None


def test_parse_solver_dirname_valid_extracts_meta():
    meta = cr.parse_solver_dirname("nodes2_pods10_prio3_util050_timeout10")
    assert meta is not None
    assert meta["nodes"] == 2
    assert meta["pods"] == 10
    assert meta["priorities"] == 3
    assert meta["util"] == 50.0
    assert meta["timeout"] == 10
    assert meta["pods_per_node"] == 5
    assert meta["default_dirname"] == "nodes2_pods10_prio3_util050"


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

def test_load_csv_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        cr.load_csv(tmp_path / "does-not-exist.csv")


def test_load_csv_renames_and_converts(tmp_path: Path):
    p = tmp_path / "results.csv"
    _write_results_csv(
        p,
        [
            {
                "seed": " s1 ",
                "util_run_cpu_now": "10",
                "util_run_mem_now": "100",
                "running_placed_by_prio_now": '{"p2": 1}',
                "unscheduled_count_before": "0",
                "error": "",
                "best_solver_status": "optimal",
                "best_solver_name": "solver",
                "best_solver_duration_ms": "5",
                "best_solver_score": "",
            }
        ],
    )

    df = cr.load_csv(p)

    # renamed columns exist
    for c in ["seed", "util_run_cpu", "util_run_mem", "solver_status", "solver_duration_ms", "placed_by_prio"]:
        assert c in df.columns

    # conversions
    assert df.loc[0, "seed"] == "s1"
    assert float(df.loc[0, "util_run_cpu"]) == 10.0
    assert float(df.loc[0, "util_run_mem"]) == 100.0
    assert df.loc[0, "solver_status"] == "OPTIMAL"
    assert float(df.loc[0, "solver_duration_ms"]) == 5.0

    # placed_by_prio is compact JSON (exact value depends on parse_json_cell, but should be a JSON string)
    assert isinstance(df.loc[0, "placed_by_prio"], str)
    assert df.loc[0, "placed_by_prio"].startswith("{") and df.loc[0, "placed_by_prio"].endswith("}")


# ---------------------------------------------------------------------------
# CombineResultsAnalyzer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _format_num
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,decimals,expected",
    [
        (1.23456, None, 1.23456),
        (1.23456, 2, "1.23"),
        (0.0, 3, "0.000"),
    ],
)
def test_format_num(value: float, decimals: int | None, expected):
    got = cr.CombineResultsAnalyzer._format_num(value, decimals)
    assert got == expected


# ---------------------------------------------------------------------------
# _split_default_all_running
# ---------------------------------------------------------------------------

def test_split_default_all_running_splits_rows():
    df = pd.DataFrame(
        {
            "seed": ["a", "b", "c"],
            "default_all_running": [True, False, True],
        }
    )
    mask, not_all = cr.CombineResultsAnalyzer._split_default_all_running(df)
    assert mask.tolist() == [True, False, True]
    assert not_all["seed"].tolist() == ["b"]


# ---------------------------------------------------------------------------
# _status_flags
# ---------------------------------------------------------------------------

def test_status_flags_optimal_feasible_ok():
    df = pd.DataFrame({"solver_status": ["OPTIMAL", "FEASIBLE", "FAILED", ""]})
    is_opt, is_feas, is_ok = cr.CombineResultsAnalyzer._status_flags(df)
    assert is_opt.tolist() == [True, False, False, False]
    assert is_feas.tolist() == [False, True, False, False]
    assert is_ok.tolist() == [True, True, False, False]


# ---------------------------------------------------------------------------
# _placement_flags
# ---------------------------------------------------------------------------

def test_placement_flags_equal_better_worse():
    df = pd.DataFrame({"placed_cmp": [0, 1, -1, 2, -2]})
    eq, better, worse = cr.CombineResultsAnalyzer._placement_flags(df)
    assert eq.tolist() == [True, False, False, False, False]
    assert better.tolist() == [False, True, False, True, False]
    assert worse.tolist() == [False, False, True, False, True]


# ---------------------------------------------------------------------------
# _solver_called_count
# ---------------------------------------------------------------------------

def test_solver_called_count_sums_int_flags():
    df = pd.DataFrame({"solver_called": [0, 1, 1, 0]})
    assert cr.CombineResultsAnalyzer._solver_called_count(df) == 2


# ---------------------------------------------------------------------------
# _compute_category_counts
# ---------------------------------------------------------------------------

def test_compute_category_counts_empty_df_uses_denominator_one():
    # Provide the required columns with no rows.
    df = pd.DataFrame(
        {
            "default_all_running": pd.Series(dtype=bool),
            "solver_status": pd.Series(dtype=str),
            "placed_cmp": pd.Series(dtype=float),
            "solver_called": pd.Series(dtype=int),
            "solver_duration_ms": pd.Series(dtype=float),
            "cpu_delta": pd.Series(dtype=float),
            "mem_delta": pd.Series(dtype=float),
        }
    )
    counts = cr.CombineResultsAnalyzer._compute_category_counts(df)
    assert counts.n_seeds == 1
    assert counts.n_default_all_running == 0
    assert counts.default_all_running_rate == 0.0
    assert counts.other_rate == 1.0


# ---------------------------------------------------------------------------
# default_vs_solver_per_seed
# ---------------------------------------------------------------------------

def test_default_vs_solver_per_seed_and_category_counts(tmp_path: Path):
    solver_csv = tmp_path / "solver" / "results.csv"
    default_csv = tmp_path / "default" / "results.csv"

    # Seeds crafted to cover the categories.
    solver_rows = [
        # all running
        {
            "seed": "s_all",
            "util_run_cpu_now": "10",
            "util_run_mem_now": "100",
            "running_placed_by_prio_now": '{"p2": 1}',
            "unscheduled_count_before": "0",
            "best_solver_status": "OPTIMAL",
            "best_solver_name": "solver",
            "best_solver_duration_ms": "5",
        },
        # default optimal (equal placement)
        {
            "seed": "s_def_opt",
            "util_run_cpu_now": "11",
            "util_run_mem_now": "110",
            "running_placed_by_prio_now": '{"p2": 1}',
            "unscheduled_count_before": "1",
            "best_solver_status": "OPTIMAL",
            "best_solver_name": "solver",
            "best_solver_duration_ms": "10",
        },
        # solver optimal (better placement)
        {
            "seed": "s_sol_opt",
            "util_run_cpu_now": "12",
            "util_run_mem_now": "120",
            "running_placed_by_prio_now": '{"p2": 2}',
            "unscheduled_count_before": "1",
            "best_solver_status": "OPTIMAL",
            "best_solver_name": "solver",
            "best_solver_duration_ms": "11",
        },
        # solver feasible (better placement)
        {
            "seed": "s_sol_feas",
            "util_run_cpu_now": "13",
            "util_run_mem_now": "130",
            "running_placed_by_prio_now": '{"p2": 2}',
            "unscheduled_count_before": "1",
            "best_solver_status": "FEASIBLE",
            "best_solver_name": "solver",
            "best_solver_duration_ms": "12",
        },
        # solver failed
        {
            "seed": "s_fail",
            "util_run_cpu_now": "14",
            "util_run_mem_now": "140",
            "running_placed_by_prio_now": '{"p2": 0}',
            "unscheduled_count_before": "1",
            "best_solver_status": "FAILED",
            "best_solver_name": "",
            "best_solver_duration_ms": "",
        },
    ]

    default_rows = [
        {
            "seed": "s_all",
            "util_run_cpu_now": "9",
            "util_run_mem_now": "90",
            "running_placed_by_prio_now": '{"p2": 1}',
            "unscheduled_count_before": "0",
            "best_solver_status": "",
            "best_solver_name": "",
            "best_solver_duration_ms": "",
        },
        {
            "seed": "s_def_opt",
            "util_run_cpu_now": "10",
            "util_run_mem_now": "100",
            "running_placed_by_prio_now": '{"p2": 1}',
            "unscheduled_count_before": "1",
            "best_solver_status": "",
            "best_solver_name": "",
            "best_solver_duration_ms": "",
        },
        {
            "seed": "s_sol_opt",
            "util_run_cpu_now": "10",
            "util_run_mem_now": "100",
            "running_placed_by_prio_now": '{"p2": 1}',
            "unscheduled_count_before": "1",
            "best_solver_status": "",
            "best_solver_name": "",
            "best_solver_duration_ms": "",
        },
        {
            "seed": "s_sol_feas",
            "util_run_cpu_now": "10",
            "util_run_mem_now": "100",
            "running_placed_by_prio_now": '{"p2": 1}',
            "unscheduled_count_before": "1",
            "best_solver_status": "",
            "best_solver_name": "",
            "best_solver_duration_ms": "",
        },
        {
            "seed": "s_fail",
            "util_run_cpu_now": "10",
            "util_run_mem_now": "100",
            "running_placed_by_prio_now": '{"p2": 1}',
            "unscheduled_count_before": "1",
            "best_solver_status": "",
            "best_solver_name": "",
            "best_solver_duration_ms": "",
        },
    ]

    _write_results_csv(solver_csv, solver_rows)
    _write_results_csv(default_csv, default_rows)

    joined = cr.default_vs_solver_per_seed(solver_csv, default_csv, cfg_name="cfg")
    assert len(joined) == 5

    # default_all_running path
    assert joined.loc[joined["seed"] == "s_all", "default_all_running"].item() is True

    # placement comparisons (rely on cmp_placed_by_prio_row behavior)
    assert joined.loc[joined["seed"] == "s_def_opt", "placed_cmp"].item() == 0
    assert joined.loc[joined["seed"] == "s_sol_opt", "placed_cmp"].item() == 1

    counts = cr.CombineResultsAnalyzer._compute_category_counts(joined)
    assert counts.n_seeds == 5
    assert counts.n_default_all_running == 1
    assert counts.n_default_optimal == 1
    assert counts.n_solver_optimal == 1
    assert counts.n_solver_feasible == 1
    assert counts.n_solver_failed == 1
    assert counts.n_solver_improve == 2
    assert counts.n_other == 0


# ---------------------------------------------------------------------------
# CombineResultsAnalyzer.analyze_combo / run
# ---------------------------------------------------------------------------

def test_analyze_combo_skips_missing_dirs_and_files(tmp_path: Path, capsys):
    args = cr.CombineResultsArgs(
        results_root=tmp_path,
        solver_dir="solver",
        default_dir="default",
        results_csv="results.csv",
        out_dir=tmp_path / "analysis",
        decimals=2,
    )
    analyzer = cr.CombineResultsAnalyzer(args)

    meta = cr.parse_solver_dirname("nodes1_pods1_prio1_util050_timeout10")
    assert meta is not None

    solver_dir = tmp_path / "solver" / "nodes1_pods1_prio1_util050_timeout10"
    default_dir = tmp_path / "default" / meta["default_dirname"]

    # missing default dir
    assert analyzer.analyze_combo(solver_dir=solver_dir, default_dir=default_dir, meta=meta) is None
    out = capsys.readouterr().out
    assert "default folder missing" in out

    # create default dir but missing solver results.csv
    default_dir.mkdir(parents=True)
    assert analyzer.analyze_combo(solver_dir=solver_dir, default_dir=default_dir, meta=meta) is None
    out = capsys.readouterr().out
    assert "solver results.csv missing" in out

    # create solver csv but missing default results.csv
    solver_dir.mkdir(parents=True)
    _write_results_csv(solver_dir / "results.csv", [])
    assert analyzer.analyze_combo(solver_dir=solver_dir, default_dir=default_dir, meta=meta) is None
    out = capsys.readouterr().out
    assert "default results.csv missing" in out


def test_analyze_combo_emits_warn_branches_when_patched(tmp_path: Path, capsys, monkeypatch):
    # Minimal valid directory structure
    solver_dir = tmp_path / "solver" / "nodes1_pods1_prio1_util050_timeout10"
    default_dir = tmp_path / "default" / "nodes1_pods1_prio1_util050"
    solver_csv = solver_dir / "results.csv"
    default_csv = default_dir / "results.csv"

    _write_results_csv(
        solver_csv,
        [
            {
                "seed": "s1",
                "util_run_cpu_now": "10",
                "util_run_mem_now": "100",
                "running_placed_by_prio_now": '{"p1": 1}',
                "unscheduled_count_before": "1",
                "best_solver_status": "OPTIMAL",
                "best_solver_name": "solver",
                "best_solver_duration_ms": "1",
            }
        ],
    )
    _write_results_csv(
        default_csv,
        [
            {
                "seed": "s1",
                "util_run_cpu_now": "9",
                "util_run_mem_now": "90",
                "running_placed_by_prio_now": '{"p1": 1}',
                "unscheduled_count_before": "1",
                "best_solver_status": "",
                "best_solver_name": "",
                "best_solver_duration_ms": "",
            }
        ],
    )

    args = cr.CombineResultsArgs(
        results_root=tmp_path,
        solver_dir="solver",
        default_dir="default",
        results_csv="results.csv",
        out_dir=tmp_path / "analysis",
        decimals=2,
    )
    analyzer = cr.CombineResultsAnalyzer(args)
    meta = cr.parse_solver_dirname(solver_dir.name)
    assert meta is not None

    # Force warning branches in analyze_combo
    def fake_counts(_df):
        return cr.CategoryCounts(
            n_seeds=1,
            n_seeds_not_all_running=1,
            n_default_all_running=0,
            n_solver_called=0,
            n_default_optimal=0,
            n_solver_optimal=0,
            n_solver_feasible=0,
            n_solver_failed=0,
            n_solver_improve=0,
            n_other=0,
            default_all_running_rate=2.0,
            solver_called_rate=2.0,
            default_optimal_rate=2.0,
            solver_optimal_rate=2.0,
            solver_feasible_rate=2.0,
            solver_failed_rate=2.0,
            solver_improve_rate=2.0,
            other_rate=2.0,
        )

    monkeypatch.setattr(cr.CombineResultsAnalyzer, "_compute_category_counts", staticmethod(fake_counts))
    row = analyzer.analyze_combo(solver_dir=solver_dir, default_dir=default_dir, meta=meta)
    assert row is not None

    out = capsys.readouterr().out
    assert "rates sum > 1.00" in out
    assert "category counts sum" in out


def test_analyze_combo_emits_warn_rate_sum_lt_zero_when_patched(tmp_path: Path, capsys, monkeypatch):
    # Minimal valid directory structure
    solver_dir = tmp_path / "solver" / "nodes1_pods1_prio1_util050_timeout10"
    default_dir = tmp_path / "default" / "nodes1_pods1_prio1_util050"
    solver_csv = solver_dir / "results.csv"
    default_csv = default_dir / "results.csv"

    _write_results_csv(
        solver_csv,
        [
            {
                "seed": "s1",
                "util_run_cpu_now": "10",
                "util_run_mem_now": "100",
                "running_placed_by_prio_now": '{"p1": 1}',
                "unscheduled_count_before": "1",
                "best_solver_status": "OPTIMAL",
                "best_solver_name": "solver",
                "best_solver_duration_ms": "1",
            }
        ],
    )
    _write_results_csv(
        default_csv,
        [
            {
                "seed": "s1",
                "util_run_cpu_now": "9",
                "util_run_mem_now": "90",
                "running_placed_by_prio_now": '{"p1": 1}',
                "unscheduled_count_before": "1",
                "best_solver_status": "",
                "best_solver_name": "",
                "best_solver_duration_ms": "",
            }
        ],
    )

    args = cr.CombineResultsArgs(
        results_root=tmp_path,
        solver_dir="solver",
        default_dir="default",
        results_csv="results.csv",
        out_dir=tmp_path / "analysis",
        decimals=2,
    )
    analyzer = cr.CombineResultsAnalyzer(args)
    meta = cr.parse_solver_dirname(solver_dir.name)
    assert meta is not None

    # Force the rate_sum < 0.00 branch
    def fake_counts(_df):
        return cr.CategoryCounts(
            n_seeds=1,
            n_seeds_not_all_running=1,
            n_default_all_running=0,
            n_solver_called=0,
            n_default_optimal=0,
            n_solver_optimal=0,
            n_solver_feasible=0,
            n_solver_failed=0,
            n_solver_improve=0,
            n_other=0,
            default_all_running_rate=-1.0,
            solver_called_rate=-1.0,
            default_optimal_rate=-1.0,
            solver_optimal_rate=-1.0,
            solver_feasible_rate=-1.0,
            solver_failed_rate=-1.0,
            solver_improve_rate=-1.0,
            other_rate=-1.0,
        )

    monkeypatch.setattr(cr.CombineResultsAnalyzer, "_compute_category_counts", staticmethod(fake_counts))
    row = analyzer.analyze_combo(solver_dir=solver_dir, default_dir=default_dir, meta=meta)
    assert row is not None

    out = capsys.readouterr().out
    assert "rates sum < 0.00" in out
    assert "category counts sum" in out


def test_run_writes_per_combo_csv_and_skips_bad_folder(tmp_path: Path, capsys):
    results_root = tmp_path / "results"
    solver_root = results_root / "solver"
    default_root = results_root / "default"

    good_solver_dir = solver_root / "nodes1_pods1_prio1_util050_timeout10"
    bad_solver_dir = solver_root / "bad-name"

    good_default_dir = default_root / "nodes1_pods1_prio1_util050"

    _write_results_csv(
        good_solver_dir / "results.csv",
        [
            {
                "seed": "s1",
                "util_run_cpu_now": "10",
                "util_run_mem_now": "100",
                "running_placed_by_prio_now": '{"p1": 1}',
                "unscheduled_count_before": "0",
                "best_solver_status": "OPTIMAL",
                "best_solver_name": "solver",
                "best_solver_duration_ms": "1",
            }
        ],
    )
    _write_results_csv(
        good_default_dir / "results.csv",
        [
            {
                "seed": "s1",
                "util_run_cpu_now": "9",
                "util_run_mem_now": "90",
                "running_placed_by_prio_now": '{"p1": 1}',
                "unscheduled_count_before": "0",
                "best_solver_status": "",
                "best_solver_name": "",
                "best_solver_duration_ms": "",
            }
        ],
    )

    bad_solver_dir.mkdir(parents=True, exist_ok=True)

    args = cr.CombineResultsArgs(
        results_root=results_root,
        solver_dir="solver",
        default_dir="default",
        results_csv="results.csv",
        out_dir=tmp_path / "analysis",
        decimals=None,
    )
    cr.CombineResultsAnalyzer(args).run()

    out = capsys.readouterr().out
    assert "bad folder name" in out
    assert "wrote" in out

    out_csv = tmp_path / "analysis" / "per_combo_results.csv"
    assert out_csv.exists()
    text = out_csv.read_text(encoding="utf-8")
    assert "config_dir" in text
    assert "nodes1_pods1_prio1_util050_timeout10" in text


# ---------------------------------------------------------------------------
# build_argparser / main
# ---------------------------------------------------------------------------

def test_main_parses_args_and_decimals_disable(tmp_path: Path, monkeypatch):
    seen: dict = {}

    def fake_run(self):
        seen["args"] = self.args

    monkeypatch.setattr(cr.CombineResultsAnalyzer, "run", fake_run)

    cr.main(
        [
            "--results-root",
            str(tmp_path),
            "--solver-dir",
            "solver",
            "--default-dir",
            "default",
            "--out-dir",
            str(tmp_path / "analysis"),
            "--decimals",
            "-1",
        ]
    )

    assert "args" in seen
    assert seen["args"].results_root == tmp_path
    assert seen["args"].decimals is None
