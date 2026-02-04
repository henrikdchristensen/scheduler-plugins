#!/usr/bin/env python3
# test_python_solver_glop.py
"""
Tests for the Glop LP Solver for Kubernetes Pod Scheduling Optimization.

These tests mirror the structure of test_python_solver_main.py but are adapted
for the Glop solver's specific behavior and status codes.
"""

# Suppress SWIG deprecation warnings from ortools
import warnings
warnings.filterwarnings("ignore", message="builtin type Swig", category=DeprecationWarning)

import pytest
import io
import json

from ortools.linear_solver import pywraplp

from scripts.python_solver.solver_glop import (
    GlopSolver,
    NO_NODES,
    NO_PODS,
    SolverOptions,
    ROUNDING_THRESHOLD,
    main as solver_main,
)


# ---------------------------------------------------------------------------
# Shared Test Data Builders
# ---------------------------------------------------------------------------

def go_payload(*, solver_input: dict, solver_options: dict | None = None) -> dict:
    """Match the exact JSON shape Go sends to the Python solver."""
    return {
        "solver_input": solver_input or {},
        "solver_options": solver_options or {},
    }


def node(name="n1", cpu=1000, mem=1_000_000_000):
    return {"name": name, "cap_cpu_m": cpu, "cap_mem_bytes": mem}


def pod(
    uid: str,
    *,
    cpu: int = 100,
    mem: int = 100_000_000,
    priority: int = 0,
    node: str = "",
    namespace: str = "default",
    protected: bool = False,
):
    return {
        "uid": uid,
        "namespace": namespace,
        "name": f"pod-{uid}",
        "req_cpu_m": cpu,
        "req_mem_bytes": mem,
        "priority": priority,
        "protected": protected,
        "node": node,  # "" = pending
    }


# A small timeout to allow tests to run quickly
DEFAULT_TIMEOUT_MS = 2000


# ---------------------------------------------------------------------------
# Schema Assertions
# ---------------------------------------------------------------------------

def assert_solver_output_schema(out: dict, *, expect_full: bool) -> None:
    assert isinstance(out, dict)
    assert "status" in out
    assert isinstance(out["status"], str)
    if not expect_full:
        return
    assert "placements" in out
    assert "evictions" in out
    assert "phases" in out
    assert "duration_ms" in out
    assert isinstance(out["placements"], list)
    assert isinstance(out["evictions"], list)
    assert isinstance(out["phases"], list)
    assert isinstance(out["duration_ms"], int)


def assert_placement_entry_schema(pl: dict) -> None:
    assert isinstance(pl, dict)
    assert set(["uid", "name", "namespace", "old_node", "node"]).issubset(pl.keys())
    assert isinstance(pl["uid"], str)
    assert isinstance(pl["name"], str)
    assert isinstance(pl["namespace"], str)
    assert isinstance(pl["old_node"], str)
    assert isinstance(pl["node"], str)


def assert_eviction_entry_schema(ev: dict) -> None:
    assert isinstance(ev, dict)
    assert set(["uid", "name", "namespace", "node"]).issubset(ev.keys())
    assert isinstance(ev["uid"], str)
    assert isinstance(ev["name"], str)
    assert isinstance(ev["namespace"], str)
    assert isinstance(ev["node"], str)


# ---------------------------------------------------------------------------
# GlopSolver._unwrap_go_payload
# ---------------------------------------------------------------------------

def test_unwrap_go_payload_defaults_to_empty_dicts():
    s = GlopSolver()
    solver_input, solver_options = s._unwrap_go_payload(None)
    assert solver_input == {}
    assert solver_options == {}


def test_unwrap_go_payload_reads_expected_keys_and_ignores_others():
    s = GlopSolver()
    inst = {
        "solver_input": {"nodes": [node("n1")], "pods": [pod("p1")]},
        "solver_options": {"log_progress": True},
        "some_future_top_level": 123,
    }
    solver_input, solver_options = s._unwrap_go_payload(inst)
    assert solver_input["nodes"][0]["name"] == "n1"
    assert solver_input["pods"][0]["uid"] == "p1"
    assert solver_options["log_progress"] is True


def test_unwrap_go_payload_treats_none_values_as_empty_dicts():
    s = GlopSolver()
    inst = {"solver_input": None, "solver_options": None}
    solver_input, solver_options = s._unwrap_go_payload(inst)
    assert solver_input == {}
    assert solver_options == {}


# ---------------------------------------------------------------------------
# GlopSolver._read_input
# ---------------------------------------------------------------------------

def test_read_input_defaults_to_empty_lists_and_none_preemptor():
    s = GlopSolver()
    nodes, pods, preemptor = s._read_input({})
    assert nodes == []
    assert pods == []
    assert preemptor is None


def test_read_input_reads_nodes_pods_and_preemptor():
    s = GlopSolver()
    solver_input = {
        "nodes": [node("n1")],
        "pods": [pod("p1")],
        "preemptor": {"uid": "pre"},
    }
    nodes, pods, preemptor = s._read_input(solver_input)
    assert nodes[0]["name"] == "n1"
    assert pods[0]["uid"] == "p1"
    assert preemptor["uid"] == "pre"


# ---------------------------------------------------------------------------
# GlopSolver._status_str
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (pywraplp.Solver.OPTIMAL, "OPTIMAL"),
        (pywraplp.Solver.FEASIBLE, "FEASIBLE"),
        (pywraplp.Solver.INFEASIBLE, "INFEASIBLE"),
        (pywraplp.Solver.UNBOUNDED, "INFEASIBLE"),  # Mapped for compatibility
        (pywraplp.Solver.ABNORMAL, "UNKNOWN"),      # Mapped for compatibility
        (pywraplp.Solver.NOT_SOLVED, "UNKNOWN"),    # Mapped for compatibility
        (pywraplp.Solver.MODEL_INVALID, "MODEL_INVALID"),
        (123456, "UNKNOWN"),  # unknown status int
    ],
)
def test_status_str_accepts_ints(value, expected):
    assert GlopSolver._status_str(value) == expected


@pytest.mark.parametrize("value", ["FOO", None, 1.23, object(), [pywraplp.Solver.OPTIMAL]])
def test_status_str_rejects_non_int(value):
    with pytest.raises(TypeError):
        GlopSolver._status_str(value)


# ---------------------------------------------------------------------------
# GlopSolver._parse_options
# ---------------------------------------------------------------------------

def test_parse_options_applies_defaults_and_timeout_safety_pad():
    s = GlopSolver()
    opts = s._parse_options(solver_input={}, solver_options={})
    assert isinstance(opts, SolverOptions)
    # default 3000 - 200
    assert opts.timeout_ms == 2800
    assert opts.ignore_affinity is True
    assert opts.log_progress is False
    assert opts.guaranteed_tier_fraction == 0.6
    assert opts.move_fraction_of_tier == 0.5
    assert opts.gap_limit == 0.0


def test_parse_options_reads_fields_from_correct_sources():
    s = GlopSolver()
    solver_input = {"timeout_ms": 2500, "ignore_affinity": False}
    solver_options = {
        "log_progress": True,
        "guaranteed_tier_fraction": 0.4,
        "move_fraction_of_tier": 0.3,
        "gap_limit": 0.05,
    }
    opts = s._parse_options(solver_input=solver_input, solver_options=solver_options)
    # timeout still gets safety pad
    assert opts.timeout_ms == 2300
    assert opts.ignore_affinity is False
    assert opts.log_progress is True
    assert opts.guaranteed_tier_fraction == 0.4
    assert opts.move_fraction_of_tier == 0.3
    assert opts.gap_limit == 0.05


# ---------------------------------------------------------------------------
# GlopSolver._create_solver
# ---------------------------------------------------------------------------

def test_create_solver_returns_glop_solver():
    s = GlopSolver()
    opts = SolverOptions(
        timeout_ms=DEFAULT_TIMEOUT_MS,
        ignore_affinity=True,
        log_progress=False,
        guaranteed_tier_fraction=0.6,
        move_fraction_of_tier=0.5,
        gap_limit=0.0,
    )
    solver = s._create_solver(opts)
    assert solver is not None
    # Glop should be available
    assert solver.SolverVersion() != ""


# ---------------------------------------------------------------------------
# GlopSolver._dedupe_pods_by_uid
# ---------------------------------------------------------------------------

def test_dedupe_pods_by_uid_skips_missing_uid_records():
    s = GlopSolver()
    pods = [
        {"namespace": "default", "name": "no-uid", "node": "n1"},
        pod("p1", node="n1"),
    ]
    out = s._dedupe_pods_by_uid(pods)
    assert list(out.keys()) == ["p1"]


def test_dedupe_pods_by_uid_prefers_record_with_node_assignment():
    s = GlopSolver()
    pods = [
        pod("p1", node=""),   # pending
        pod("p1", node="n1"),  # running should win
    ]
    out = s._dedupe_pods_by_uid(pods)
    assert list(out.keys()) == ["p1"]
    assert out["p1"]["node"] == "n1"


def test_dedupe_pods_by_uid_does_not_downgrade_running_to_pending():
    s = GlopSolver()
    pods = [
        pod("p1", node="n1"),  # running first
        pod("p1", node=""),    # pending later must not override
    ]
    out = s._dedupe_pods_by_uid(pods)
    assert out["p1"]["node"] == "n1"


# ---------------------------------------------------------------------------
# GlopSolver._apply_preemptor_to_pods
# ---------------------------------------------------------------------------

def test_apply_preemptor_to_pods_noop_when_preemptor_missing_or_invalid():
    s = GlopSolver()
    pod_by_uid, single_mode, pre_uid = s._apply_preemptor_to_pods({}, None)
    assert pod_by_uid == {}
    assert single_mode is False
    assert pre_uid is None
    pod_by_uid, single_mode, pre_uid = s._apply_preemptor_to_pods({}, {"name": "no-uid"})
    assert pod_by_uid == {}
    assert single_mode is False
    assert pre_uid is None


def test_apply_preemptor_to_pods_adds_pending_preemptor_when_missing():
    s = GlopSolver()
    preemptor = {
        "uid": "pre",
        "namespace": "default",
        "name": "pre",
        "req_cpu_m": 300,
        "req_mem_bytes": 100_000_000,
        "priority": 10,
        "protected": False,
    }
    pod_by_uid, single_mode, pre_uid = s._apply_preemptor_to_pods({}, preemptor)
    assert single_mode is True
    assert pre_uid == "pre"
    assert "pre" in pod_by_uid
    assert pod_by_uid["pre"]["node"] == ""  # pending


def test_apply_preemptor_to_pods_does_not_override_existing_uid():
    s = GlopSolver()
    existing = {"pre": pod("pre", node="n1", cpu=111)}
    preemptor = {"uid": "pre", "req_cpu_m": 999, "req_mem_bytes": 999, "priority": 99, "protected": False}
    pod_by_uid, single_mode, pre_uid = s._apply_preemptor_to_pods(existing, preemptor)
    assert single_mode is True
    assert pre_uid == "pre"
    assert pod_by_uid["pre"]["req_cpu_m"] == 111
    assert pod_by_uid["pre"]["node"] == "n1"


# ---------------------------------------------------------------------------
# GlopSolver._freeze_problem
# ---------------------------------------------------------------------------

def test_freeze_problem_returns_quick_exit_on_no_nodes():
    s = GlopSolver()
    out = s._freeze_problem(nodes=[], pods=[pod("p1")], single_preemptor_mode=False, preemptor_uid=None)
    assert out == {"status": NO_NODES}


def test_freeze_problem_returns_quick_exit_on_no_pods():
    s = GlopSolver()
    out = s._freeze_problem(nodes=[node("n1")], pods=[], single_preemptor_mode=False, preemptor_uid=None)
    assert out == {"status": NO_PODS}


def test_freeze_problem_builds_indices_running_pending_and_eligibility():
    s = GlopSolver()
    nodes = [
        node("n1", cpu=500, mem=1_000_000_000),
        node("n2", cpu=1000, mem=1_000_000_000),
    ]
    pods = [
        pod("p1", cpu=600, mem=100_000_000, node=""),   # pending; only fits n2
        pod("p2", cpu=200, mem=100_000_000, node="n1"),  # running; fits both
    ]
    problem = s._freeze_problem(nodes=nodes, pods=pods, single_preemptor_mode=False, preemptor_uid=None)
    assert isinstance(problem, dict) is False
    assert set(problem.pending_idxs) == {0}
    assert set(problem.running_idxs) == {1}
    assert problem.eligible_nodes[0] == [1]
    assert problem.eligible_nodes[1] == [0, 1]


def test_freeze_problem_preemptor_uid_not_found_disables_single_preemptor_mode():
    s = GlopSolver()
    nodes = [node("n1")]
    pods = [pod("p1", node="")]
    problem = s._freeze_problem(nodes=nodes, pods=pods, single_preemptor_mode=True, preemptor_uid="missing")
    assert isinstance(problem, dict) is False
    assert problem.single_preemptor_mode is False
    assert problem.preemptor_idx is None


# ---------------------------------------------------------------------------
# GlopSolver._build_decision_vars
# ---------------------------------------------------------------------------

def test_build_decision_vars_creates_correct_shapes():
    s = GlopSolver()
    nodes = [node("n1"), node("n2")]
    pods = [
        pod("p1", cpu=100, mem=100_000_000, node=""),  # fits both
        pod("p2", cpu=9999, mem=100_000_000, node=""),  # fits none
    ]
    problem = s._freeze_problem(nodes=nodes, pods=pods, single_preemptor_mode=False, preemptor_uid=None)
    assert isinstance(problem, dict) is False

    opts = SolverOptions(
        timeout_ms=DEFAULT_TIMEOUT_MS,
        ignore_affinity=True,
        log_progress=False,
        guaranteed_tier_fraction=0.6,
        move_fraction_of_tier=0.5,
        gap_limit=0.0,
    )
    solver = s._create_solver(opts)
    dv = s._build_decision_vars(solver, problem)

    assert len(dv.placed) == problem.num_pods
    assert len(dv.assign) == problem.num_pods
    assert len(dv.assign[0]) == 2  # eligible nodes for p1
    assert len(dv.assign[1]) == 0  # no eligible nodes for p2


# ---------------------------------------------------------------------------
# GlopSolver constraint helpers
# ---------------------------------------------------------------------------

def _make_small_model_for_constraints(nodes, pods, *, single_preemptor_mode=False, preemptor_uid=None):
    """Tiny helper for tests that want to call constraint helpers directly."""
    s = GlopSolver()
    problem = s._freeze_problem(
        nodes=nodes,
        pods=pods,
        single_preemptor_mode=single_preemptor_mode,
        preemptor_uid=preemptor_uid,
    )
    assert isinstance(problem, dict) is False
    opts = SolverOptions(
        timeout_ms=DEFAULT_TIMEOUT_MS,
        ignore_affinity=True,
        log_progress=False,
        guaranteed_tier_fraction=0.6,
        move_fraction_of_tier=0.5,
        gap_limit=0.0,
    )
    solver = s._create_solver(opts)
    dv = s._build_decision_vars(solver, problem)
    return s, problem, solver, dv


def test_add_assign_constraints_sets_placed_zero_when_no_eligible_nodes():
    # pod p1 fits no nodes => placed_0 forced to 0
    s, problem, solver, dv = _make_small_model_for_constraints(
        nodes=[node("n1", cpu=50, mem=100_000_000)],
        pods=[pod("p1", cpu=200, mem=200_000_000, node="")],
    )
    s._add_assign_constraints(solver, problem, dv)

    # With no objective, solve should succeed
    st = solver.Solve()
    assert st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    assert dv.placed[0].solution_value() < ROUNDING_THRESHOLD


def test_add_protected_stay_constraints_returns_model_invalid_when_cannot_stay():
    s = GlopSolver()
    nodes = [node("n1", cpu=50, mem=100_000_000)]
    pods = [
        pod("p1", cpu=200, mem=100_000_000, node="n1", protected=True),  # running but ineligible => cannot stay
    ]
    problem = s._freeze_problem(nodes=nodes, pods=pods, single_preemptor_mode=False, preemptor_uid=None)
    assert isinstance(problem, dict) is False

    opts = SolverOptions(
        timeout_ms=DEFAULT_TIMEOUT_MS,
        ignore_affinity=True,
        log_progress=False,
        guaranteed_tier_fraction=0.6,
        move_fraction_of_tier=0.5,
        gap_limit=0.0,
    )
    solver = s._create_solver(opts)
    dv = s._build_decision_vars(solver, problem)

    # constraint helper should early-return dict
    err = s._add_protected_stay_constraints(solver, problem, dv)
    assert err == {"status": "MODEL_INVALID"}


# ---------------------------------------------------------------------------
# GlopSolver._compute_overall_status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phases,st,expected",
    [
        ([{"status": "MODEL_INVALID"}], pywraplp.Solver.OPTIMAL, "MODEL_INVALID"),
        ([{"status": "UNKNOWN"}], pywraplp.Solver.OPTIMAL, "UNKNOWN"),
        ([{"status": "INFEASIBLE"}], pywraplp.Solver.OPTIMAL, "INFEASIBLE"),
        ([{"status": "FEASIBLE"}], pywraplp.Solver.OPTIMAL, "FEASIBLE"),
        ([{"status": "OPTIMAL"}], pywraplp.Solver.FEASIBLE, "OPTIMAL"),
        ([], pywraplp.Solver.FEASIBLE, "FEASIBLE"),  # fallback to _status_str(st)
    ],
)
def test_compute_overall_status_priority(phases, st, expected):
    s = GlopSolver()
    assert s._compute_overall_status(phases, st) == expected


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_solve_quick_exits_no_nodes_or_no_pods():
    s = GlopSolver()
    out = s.solve(go_payload(solver_input={"nodes": [], "pods": [pod("p1")], "timeout_ms": DEFAULT_TIMEOUT_MS}))
    assert_solver_output_schema(out, expect_full=False)
    assert out["status"] == NO_NODES
    out = s.solve(go_payload(solver_input={"nodes": [node("n1")], "pods": [], "timeout_ms": DEFAULT_TIMEOUT_MS}))
    assert_solver_output_schema(out, expect_full=False)
    assert out["status"] == NO_PODS


def test_solve_single_pending_pod_is_placed_on_single_node():
    s = GlopSolver()
    out = s.solve(
        go_payload(
            solver_input={
                "nodes": [node("n1", cpu=1000, mem=1_000_000_000)],
                "pods": [pod("p1", cpu=200, mem=100_000_000, node="")],
                "timeout_ms": DEFAULT_TIMEOUT_MS,
            }
        )
    )
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] in ("FEASIBLE", "OPTIMAL")
    assert len(out["placements"]) == 1
    pl = out["placements"][0]
    assert_placement_entry_schema(pl)
    assert pl["uid"] == "p1"
    assert pl["old_node"] == ""
    assert pl["node"] == "n1"
    assert out["evictions"] == []


def test_solve_pending_pod_too_big_is_infeasible_in_background_mode():
    s = GlopSolver()
    out = s.solve(
        go_payload(
            solver_input={
                "nodes": [node("n1", cpu=50, mem=100_000_000)],
                "pods": [pod("p1", cpu=200, mem=100_000_000, node="")],
                "timeout_ms": DEFAULT_TIMEOUT_MS,
            }
        )
    )
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] == "INFEASIBLE"
    assert out["placements"] == []
    assert out["evictions"] == []


def test_solve_duplicate_uid_prefers_running_copy_and_avoids_infeasible_capacity():
    s = GlopSolver()
    out = s.solve(
        go_payload(
            solver_input={
                "nodes": [node("n1", cpu=300, mem=100_000_000)],
                "pods": [
                    pod("p1", cpu=300, mem=100_000_000, node="n1"),  # running
                    pod("p1", cpu=300, mem=100_000_000, node=""),    # duplicate pending
                ],
                "timeout_ms": DEFAULT_TIMEOUT_MS,
            }
        )
    )
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] in ("FEASIBLE", "OPTIMAL")
    assert out["placements"] == []
    assert out["evictions"] == []


def test_solve_single_preemptor_mode_places_preemptor_when_feasible():
    s = GlopSolver()
    preemptor = {
        "uid": "pre",
        "namespace": "default",
        "name": "pre",
        "req_cpu_m": 300,
        "req_mem_bytes": 100_000_000,
        "priority": 10,
        "protected": False,
    }
    out = s.solve(
        go_payload(
            solver_input={
                "nodes": [node("n1", cpu=1000, mem=1_000_000_000)],
                "pods": [],
                "preemptor": preemptor,
                "timeout_ms": DEFAULT_TIMEOUT_MS,
            }
        )
    )
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] in ("FEASIBLE", "OPTIMAL")
    assert len(out["placements"]) == 1
    pl = out["placements"][0]
    assert_placement_entry_schema(pl)
    assert pl["uid"] == "pre"
    assert pl["old_node"] == ""
    assert pl["node"] == "n1"
    assert out["evictions"] == []


def test_solve_running_pods_over_capacity_leads_to_one_eviction():
    s = GlopSolver()
    out = s.solve(
        go_payload(
            solver_input={
                "nodes": [node("n1", cpu=600, mem=1_000_000_000)],
                "pods": [
                    pod("p1", cpu=600, mem=100_000_000, node="n1"),
                    pod("p2", cpu=600, mem=100_000_000, node="n1"),
                ],
                "timeout_ms": DEFAULT_TIMEOUT_MS,
            }
        )
    )
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] in ("FEASIBLE", "OPTIMAL")
    assert out["placements"] == []
    assert len(out["evictions"]) == 1
    assert_eviction_entry_schema(out["evictions"][0])


def test_solve_multiple_pods_multiple_nodes():
    s = GlopSolver()
    out = s.solve(
        go_payload(
            solver_input={
                "nodes": [
                    node("n1", cpu=500, mem=1_000_000_000),
                    node("n2", cpu=500, mem=1_000_000_000),
                ],
                "pods": [
                    pod("p1", cpu=400, mem=100_000_000, node=""),
                    pod("p2", cpu=400, mem=100_000_000, node=""),
                ],
                "timeout_ms": DEFAULT_TIMEOUT_MS,
            }
        )
    )
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] in ("FEASIBLE", "OPTIMAL")
    # Both pods should be placed
    assert len(out["placements"]) == 2
    # They should be on different nodes
    nodes_used = {pl["node"] for pl in out["placements"]}
    assert len(nodes_used) == 2


def test_solve_priority_ordering_high_priority_placed_first():
    s = GlopSolver()
    out = s.solve(
        go_payload(
            solver_input={
                "nodes": [node("n1", cpu=100, mem=1_000_000_000)],
                "pods": [
                    pod("p_low", cpu=100, mem=100_000_000, node="", priority=1),
                    pod("p_high", cpu=100, mem=100_000_000, node="", priority=10),
                ],
                "timeout_ms": DEFAULT_TIMEOUT_MS,
            }
        )
    )
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] in ("FEASIBLE", "OPTIMAL")
    # Only one pod can fit; should be the high priority one
    assert len(out["placements"]) == 1
    assert out["placements"][0]["uid"] == "p_high"


# ---------------------------------------------------------------------------
# main(): Go<=>Python contract tests
# ---------------------------------------------------------------------------

def run_main_with_stdin(monkeypatch, capsys, payload_dict):
    stdin_data = json.dumps(payload_dict)
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_data))
    solver_main()
    captured = capsys.readouterr()
    out_str = captured.out.strip()
    assert out_str, "Expected main() to print something on stdout"
    return json.loads(out_str)


def test_main_valid_instance_roundtrip(monkeypatch, capsys):
    payload = go_payload(
        solver_input={
            "nodes": [node("n1")],
            "pods": [pod("p1")],
            "timeout_ms": DEFAULT_TIMEOUT_MS,
        },
        solver_options={},
    )
    out = run_main_with_stdin(monkeypatch, capsys, payload)
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] in ("FEASIBLE", "OPTIMAL")


@pytest.mark.parametrize(
    "payload,expected_status",
    [
        (go_payload(solver_input={"nodes": [], "pods": [pod("p1")], "timeout_ms": DEFAULT_TIMEOUT_MS}), NO_NODES),
        (go_payload(solver_input={"nodes": [node("n1")], "pods": [], "timeout_ms": DEFAULT_TIMEOUT_MS}), NO_PODS),
    ],
)
def test_main_quick_exits_only_require_status(monkeypatch, capsys, payload, expected_status):
    out = run_main_with_stdin(monkeypatch, capsys, payload)
    assert_solver_output_schema(out, expect_full=False)
    assert out["status"] == expected_status


def test_main_invalid_json_returns_python_exception(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{not-json}"))
    solver_main()
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert_solver_output_schema(data, expect_full=False)
    assert data["status"] == "PYTHON_EXCEPTION"


def test_main_last_resort_json_print_when_json_dumps_fails(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{not-json}"))

    import scripts.python_solver.solver_glop as solver_mod

    def dumps_boom(*_args, **_kwargs):
        raise RuntimeError("json is broken")

    monkeypatch.setattr(solver_mod.json, "dumps", dumps_boom)
    solver_mod.main()
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"status":"PYTHON_EXCEPTION","error":"unserializable error"}'


# ---------------------------------------------------------------------------
# Additional LP-specific tests
# ---------------------------------------------------------------------------

def test_rounding_threshold_is_sensible():
    """Verify the rounding threshold is within expected bounds."""
    assert 0.0 < ROUNDING_THRESHOLD < 1.0
    assert ROUNDING_THRESHOLD == 0.5  # commonly used threshold


def test_solve_with_protected_pod_stays_on_original_node():
    s = GlopSolver()
    out = s.solve(
        go_payload(
            solver_input={
                "nodes": [
                    node("n1", cpu=500, mem=1_000_000_000),
                    node("n2", cpu=500, mem=1_000_000_000),
                ],
                "pods": [
                    pod("protected", cpu=400, mem=100_000_000, node="n1", protected=True),
                    pod("pending", cpu=400, mem=100_000_000, node=""),
                ],
                "timeout_ms": DEFAULT_TIMEOUT_MS,
            }
        )
    )
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] in ("FEASIBLE", "OPTIMAL")
    # Protected pod should not appear in placements (it stays)
    placement_uids = {pl["uid"] for pl in out["placements"]}
    assert "protected" not in placement_uids
    # Pending pod should be placed on n2 (n1 doesn't have room)
    assert "pending" in placement_uids
    pending_pl = [pl for pl in out["placements"] if pl["uid"] == "pending"][0]
    assert pending_pl["node"] == "n2"
