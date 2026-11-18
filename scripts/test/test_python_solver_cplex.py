#!/usr/bin/env python3
# test_python_solver_cplex.py
"""
Tests for the CPLEX Python solver. Skipped when the cplex package is not installed.
"""

import random

import pytest

cplex = pytest.importorskip("cplex")

from scripts.python_solver.solver_cp_sat import CPSATSolver
from scripts.python_solver.solver_cplex import (
    CplexSolver,
    INFEASIBLE_CODES,
    NO_NODES,
    NO_PODS,
    OPTIMAL_CODES,
    main as cplex_solver_main,
)
from scripts.test.test_python_solver import (
    DEFAULT_TIMEOUT_MS,
    assert_eviction_entry_schema,
    assert_placement_entry_schema,
    assert_solver_output_schema,
    go_payload,
    node,
    pod,
    run_main_with_stdin,
)


def solve(nodes, pods, *, preemptor=None, timeout_ms=DEFAULT_TIMEOUT_MS, solver=CplexSolver):
    solver_input = {"nodes": nodes, "pods": pods, "timeout_ms": timeout_ms}
    if preemptor is not None:
        solver_input["preemptor"] = preemptor
    return solver().solve(go_payload(solver_input=solver_input))


def stage_values(pods, out):
    """Per priority tier, highest first: (pods placed with priority >= tier, disruption of running pods with priority >= tier)."""
    after = {p["uid"]: p["node"] or None for p in pods}
    for ev in out["evictions"]:
        after[ev["uid"]] = None
    for pl in out["placements"]:
        after[pl["uid"]] = pl["node"]
    values = []
    for tier in sorted({p["priority"] for p in pods}, reverse=True):
        ge = [p for p in pods if p["priority"] >= tier]
        placed = sum(after[p["uid"]] is not None for p in ge)
        disruption = sum(0 if after[p["uid"]] == p["node"] else (3 if after[p["uid"]] is None else 2) for p in ge if p["node"])
        values.append((placed, disruption))
    return values


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

def test_status_codes_match_the_installed_cplex():
    status = cplex.Cplex().solution.status
    assert OPTIMAL_CODES == {status.MIP_optimal, status.optimal_tolerance}
    assert INFEASIBLE_CODES == {status.MIP_infeasible, status.MIP_unbounded, status.MIP_infeasible_or_unbounded}


@pytest.mark.parametrize(
    "code,has_solution,expected",
    [
        (101, True, "OPTIMAL"),
        (102, True, "OPTIMAL"),
        (103, False, "INFEASIBLE"),
        (119, False, "INFEASIBLE"),
        (107, True, "FEASIBLE"),   # time limit with an incumbent
        (108, False, "UNKNOWN"),   # time limit without an incumbent
    ],
)
def test_status_str(code, has_solution, expected):
    assert CplexSolver._status_str(code, has_solution) == expected


# ---------------------------------------------------------------------------
# solve()
# ---------------------------------------------------------------------------

def test_solve_quick_exits_no_nodes_or_no_pods():
    assert solve([], [pod("p1")])["status"] == NO_NODES
    assert solve([node("n1")], [])["status"] == NO_PODS


def test_solve_single_pending_pod_is_placed():
    out = solve([node("n1")], [pod("p1", cpu=200)])
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] == "OPTIMAL"
    assert len(out["placements"]) == 1
    assert_placement_entry_schema(out["placements"][0])
    assert (out["placements"][0]["uid"], out["placements"][0]["old_node"], out["placements"][0]["node"]) == ("p1", "", "n1")
    assert out["evictions"] == []


def test_solve_pending_pod_too_big_returns_empty_plan():
    out = solve([node("n1", cpu=50)], [pod("p1", cpu=200)])
    assert out["status"] == "OPTIMAL"
    assert out["placements"] == [] and out["evictions"] == []


def test_solve_nothing_to_do_returns_empty_plan():
    out = solve([node("n1"), node("n2")], [pod("p1", cpu=400, node="n1"), pod("p2", cpu=400, node="n2")])
    assert out["status"] == "OPTIMAL"
    assert out["placements"] == [] and out["evictions"] == []


def test_solve_running_pods_over_capacity_leads_to_one_eviction():
    out = solve([node("n1", cpu=600)], [pod("p1", cpu=600, node="n1"), pod("p2", cpu=600, node="n1")])
    assert out["status"] == "OPTIMAL"
    assert out["placements"] == []
    assert len(out["evictions"]) == 1
    assert_eviction_entry_schema(out["evictions"][0])


def test_solve_higher_priority_pending_pod_displaces_lower_priority_pod():
    out = solve(
        [node("n1", cpu=1000)],
        [pod("low", cpu=800, priority=1, node="n1"), pod("high", cpu=800, priority=2)],
    )
    assert out["status"] == "OPTIMAL"
    assert [(pl["uid"], pl["node"]) for pl in out["placements"]] == [("high", "n1")]
    assert [ev["uid"] for ev in out["evictions"]] == ["low"]


def test_solve_moves_a_pod_rather_than_evicting_it():
    # "big" only fits on n1 once "small" leaves; "small" also fits on n2
    out = solve(
        [node("n1", cpu=1000), node("n2", cpu=300)],
        [pod("small", cpu=300, priority=1, node="n1"), pod("big", cpu=900, priority=1)],
    )
    assert out["status"] == "OPTIMAL"
    assert sorted((pl["uid"], pl["old_node"], pl["node"]) for pl in out["placements"]) == [("big", "", "n1"), ("small", "n1", "n2")]
    assert out["evictions"] == []


def test_solve_single_preemptor_mode_places_preemptor():
    preemptor = {"uid": "pre", "namespace": "default", "name": "pre", "req_cpu_m": 300, "req_mem_bytes": 100_000_000, "priority": 10, "protected": False}
    out = solve([node("n1")], [], preemptor=preemptor)
    assert out["status"] == "OPTIMAL"
    assert [(pl["uid"], pl["node"]) for pl in out["placements"]] == [("pre", "n1")]


def test_solve_protected_pod_that_cannot_stay_is_model_invalid():
    out = solve([node("n1", cpu=50)], [pod("p1", cpu=200, node="n1", protected=True)])
    assert out["status"] == "MODEL_INVALID"
    assert out["placements"] == [] and out["evictions"] == []


def test_solve_protected_pod_is_not_displaced():
    out = solve(
        [node("n1", cpu=1000)],
        [pod("low", cpu=800, priority=1, node="n1", protected=True), pod("high", cpu=800, priority=2)],
    )
    assert out["status"] == "OPTIMAL"
    assert out["placements"] == [] and out["evictions"] == []


def test_solve_without_time_returns_unknown_and_empty_plan():
    out = solve([node("n1")], [pod("p1")], timeout_ms=200)  # the solver keeps a 200 ms safety pad
    assert out["status"] == "UNKNOWN"
    assert out["placements"] == [] and out["evictions"] == []


@pytest.mark.parametrize("seed", range(25))
def test_solve_agrees_with_cp_sat_on_random_instances(seed):
    rng = random.Random(seed)
    nodes = [node(f"n{j}", cpu=4000, mem=4_000_000_000) for j in range(rng.randint(2, 4))]
    free = {n["name"]: [n["cap_cpu_m"], n["cap_mem_bytes"]] for n in nodes}
    pods = []
    for k in range(rng.randint(6, 14)):
        p = pod(f"p{k}", cpu=rng.randint(300, 2200), mem=rng.randint(300, 2200) * 1_000_000, priority=rng.randint(1, 3))
        fits = [name for name, (cpu, mem) in free.items() if cpu >= p["req_cpu_m"] and mem >= p["req_mem_bytes"]]
        if fits and rng.random() < 0.8:
            p["node"] = rng.choice(fits)
            free[p["node"]][0] -= p["req_cpu_m"]
            free[p["node"]][1] -= p["req_mem_bytes"]
        pods.append(p)

    cplex_out = solve(nodes, pods, timeout_ms=10_000)
    cp_sat_out = solve(nodes, pods, timeout_ms=10_000, solver=CPSATSolver)

    assert cplex_out["status"] == cp_sat_out["status"] == "OPTIMAL"
    assert stage_values(pods, cplex_out) == stage_values(pods, cp_sat_out)


# ---------------------------------------------------------------------------
# main(): Go<=>Python contract
# ---------------------------------------------------------------------------

def test_main_valid_instance_roundtrip(monkeypatch, capsys):
    payload = go_payload(solver_input={"nodes": [node("n1")], "pods": [pod("p1")], "timeout_ms": DEFAULT_TIMEOUT_MS})
    out = run_main_with_stdin(monkeypatch, capsys, payload, cplex_solver_main)
    assert_solver_output_schema(out, expect_full=True)
    assert out["status"] == "OPTIMAL"
