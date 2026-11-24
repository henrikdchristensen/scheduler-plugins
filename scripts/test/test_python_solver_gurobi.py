import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.python_solver import solver_gurobi


def reject_non_json_number(token: str) -> None:
    pytest.fail(f"Non-JSON number on solver stdout: {token}")


@pytest.mark.parametrize(
    ("bound", "expected"),
    [(float("inf"), None), (float("-inf"), None), (float("nan"), None), (42.0, 42.0)],
)
def test_stdout_reports_only_finite_objective_bounds(monkeypatch, capsys, bound, expected):
    monkeypatch.setattr(solver_gurobi, "GUROBI_AVAILABLE", True)
    monkeypatch.setattr(solver_gurobi, "GRB", SimpleNamespace(LOADED=1, MAXIMIZE=-1))
    monkeypatch.setattr(solver_gurobi, "gp", SimpleNamespace(quicksum=sum))
    monkeypatch.setattr(solver_gurobi, "STATUS_MAP", {9: "FEASIBLE"})

    model = Mock(Status=9, SolCount=0, ObjBound=bound)
    problem = SimpleNamespace(num_pods=1, pod_priority=[1], pod_node_j=[None])
    variables = solver_gurobi.DecisionVars(placed=[1], assign=[[]])

    def solve(self, _):
        options = self._parse_options({"timeout_ms": 1000}, {})
        status, phases, _ = self._solve_lexicographically(model, problem, variables, options)
        return {"status": self._compute_overall_status(phases, status), "phases": phases}

    monkeypatch.setattr(solver_gurobi.GurobiSolver, "solve", solve)
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))

    solver_gurobi.main()

    output = json.loads(capsys.readouterr().out, parse_constant=reject_non_json_number)
    assert output["status"] == "UNKNOWN"
    assert output["phases"][0]["bound"] == expected
    assert output["phases"][0]["relative_gap"] == ""
