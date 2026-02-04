#!/usr/bin/env python3
# solver_glop.py
"""
Glop LP Solver for Kubernetes Pod Scheduling Optimization.

This solver uses OR-Tools' Glop (linear programming) to solve the pod placement
problem. Unlike CP-SAT which uses constraint programming with discrete variables,
Glop uses continuous relaxation and linear constraints.

Key differences from CP-SAT:
1. Variables are continuous (0.0-1.0) instead of binary
2. Uses simplex algorithm for optimization
3. Solutions may need rounding to get integer assignments
4. Generally faster but may produce fractional solutions

This solver uses the SAME Go contract as the CP-SAT solver (solver_cp_sat.py).
It can be used by setting SOLVER_TYPE=glop or SOLVER_PATH to point to this script.
"""

# Suppress SWIG deprecation warnings from ortools
import warnings
warnings.filterwarnings("ignore", message="builtin type Swig", category=DeprecationWarning)

import time
import sys
import json
from dataclasses import dataclass
from typing import Optional, Final
from ortools.linear_solver import pywraplp

#################################################
# --- Constants ---------------------------------
#################################################
NO_NODES: Final[str] = "NO_NODES"
NO_PODS: Final[str] = "NO_PODS"

# Glop solver status codes mapped to the same strings as CP-SAT
STATUS_MAP: Final[dict[int, str]] = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "INFEASIBLE",  # Map to INFEASIBLE for compatibility
    pywraplp.Solver.ABNORMAL: "UNKNOWN",      # Map to UNKNOWN for compatibility
    pywraplp.Solver.NOT_SOLVED: "UNKNOWN",    # Map to UNKNOWN for compatibility
    pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
}

# Threshold for rounding LP solutions to binary
ROUNDING_THRESHOLD: Final[float] = 0.5

#################################################
# --- Solver Options Class ----------------------
#################################################

@dataclass(frozen=True)
class SolverOptions:
    """Parsed solver options - same as CP-SAT solver for compatibility."""

    timeout_ms: int
    ignore_affinity: bool
    log_progress: bool
    guaranteed_tier_fraction: float
    move_fraction_of_tier: float
    gap_limit: float


#################################################
# --- Problem Class -----------------------------
#################################################

@dataclass(frozen=True)
class Problem:
    """A frozen, index-based view of the input.

    Identical to the CP-SAT solver's Problem class for compatibility.
    """

    # Raw payload
    nodes: list[dict]
    pods: list[dict]

    # Node lookup
    node_idx: dict[str, int]

    # Sizes
    num_nodes: int
    num_pods: int

    # Frozen nodes
    node_names: list[str]
    node_cap_cpu_m: list[int]
    node_cap_mem_bytes: list[int]

    # Frozen pods
    pod_uid: list[str]
    pod_namespace: list[str]
    pod_name: list[str]
    pod_req_cpu_m: list[int]
    pod_req_mem_bytes: list[int]
    pod_priority: list[int]
    pod_protected: list[bool]
    pod_node_j: list[Optional[int]]  # None => pending

    # Convenience index sets
    running_idxs: list[int]
    pending_idxs: list[int]

    # Eligible nodes per pod
    eligible_nodes: list[list[int]]
    eligible_pos: list[dict[int, int]]

    # Preemptor mode
    single_preemptor_mode: bool
    preemptor_uid: Optional[str]
    preemptor_idx: Optional[int]


#################################################
# --- Decision Vars Class -----------------------
#################################################

@dataclass(frozen=True)
class DecisionVars:
    """Decision variables for the Glop LP model.

    Uses continuous variables in [0,1] range which are interpreted
    as binary after rounding.
    """

    # placed[i] = 1 if pod i is placed somewhere, 0 otherwise
    placed: list[pywraplp.Variable]
    # assign[i][local] = 1 if pod i is assigned to eligible_nodes[i][local]
    assign: list[list[pywraplp.Variable]]


#################################################
# --- Glop LP Solver Class ----------------------
#################################################

class GlopSolver:
    """Linear Programming solver using OR-Tools Glop for pod scheduling."""

    #################################################
    # --- Helpers -----------------------------------
    #################################################

    @classmethod
    def _status_str(cls, st: int) -> str:
        """Convert Glop solver status to string."""
        if not isinstance(st, int):
            raise TypeError(f"status must be int (Glop solver status), got {type(st).__name__}")
        return STATUS_MAP.get(st, "UNKNOWN")

    def solve(self, instance: dict) -> dict:
        """
        Solve the given instance and return the plan.

        The instance is a dict with keys:
            - nodes: list of nodes (dicts with name, cap_cpu_m, cap_mem_bytes)
            - pods: list of pods (dicts with uid, namespace, name, req_cpu_m, req_mem_bytes, priority, protected, node)
            - preemptor: optional dict with uid, namespace, name, req_cpu_m, req_mem_bytes, priority, protected
            - timeout_ms: int, total timeout in milliseconds (default 3000)
            - ignore_affinity: bool, whether to ignore affinity constraints (default True)
            - log_progress: bool, whether to log progress (default False)
            - guaranteed_tier_fraction: float in [0.0, 1.0], fraction of time guaranteed for all tiers (default 0.6)
            - move_fraction_of_tier: float in [0.0, 1.0], fraction of tier time for moves (default 0.5)
            - gap_limit: float in [0.0, 1.0], relative gap limit (default 0.00)

        Returns a dict with keys:
            - placements: list of placements (dicts with pod {uid, namespace, name}, from_node, to_node)
            - evictions: list of evictions (dicts with pod {uid, namespace, name}, node)
            - phases: list of phases (dicts with tier, stage ("place" or "moves"), status, duration_ms, relative_gap)
            - duration_ms: total duration in milliseconds
            - status: overall status string
        """
        started_at = time.monotonic()

        #################################################
        # --- Unwrap Go payload -------------------------
        #################################################
        solver_input, solver_options = self._unwrap_go_payload(instance)

        #################################################
        # --- Read Input --------------------------------
        #################################################
        nodes, pods, preemptor = self._read_input(solver_input)

        #################################################
        # --- Options/Parameters ------------------------
        #################################################
        options = self._parse_options(solver_input, solver_options)

        #################################################
        # --- Ensure Pods from Input --------------------
        #################################################
        pod_by_uid = self._dedupe_pods_by_uid(pods)

        #################################################
        # --- Preemptor (per-pod) setup -----------------
        #################################################
        pod_by_uid, single_preemptor_mode, preemptor_uid = self._apply_preemptor_to_pods(
            pod_by_uid, preemptor
        )

        #################################################
        # --- Freeze nodes and pods and quick checks ----
        #################################################
        pods = list(pod_by_uid.values())
        frozen_or_err = self._freeze_problem(
            nodes=nodes,
            pods=pods,
            single_preemptor_mode=single_preemptor_mode,
            preemptor_uid=preemptor_uid,
        )
        if isinstance(frozen_or_err, dict):
            return frozen_or_err
        problem: Problem = frozen_or_err

        #################################################
        # --- Create Glop Solver ------------------------
        #################################################
        solver = self._create_solver(options)
        if solver is None:
            return {"status": "SOLVER_CREATION_FAILED"}

        #################################################
        # --- Decision Variables ------------------------
        #################################################
        vars = self._build_decision_vars(solver, problem)

        #################################################
        # --- Common Constraints ------------------------
        #################################################
        # Node capacity constraints
        self._add_capacity_constraints(solver, problem, vars)

        # Assignment constraints
        self._add_assign_constraints(solver, problem, vars)

        # Protected pod stay constraints
        protected_err = self._add_protected_stay_constraints(solver, problem, vars)
        if protected_err is not None:
            return protected_err

        #################################################
        # --- Mode Specific Constraints -----------------
        #################################################
        self._add_mode_specific_constraints(solver, problem, vars)

        #################################################
        # --- Solve with lexicographic optimization -----
        #################################################
        st, phases = self._solve_lexicographically(solver, problem, vars, options)

        #################################################
        # --- Extract and return plan -------------------
        #################################################
        placements, evictions = self._extract_plan(problem, vars, solver)
        total_time = max(0.0, time.monotonic() - started_at)

        overall_status = self._compute_overall_status(phases, st)

        return {
            "placements": placements,
            "evictions": evictions,
            "phases": phases,
            "duration_ms": round(total_time * 1000),
            "status": overall_status,
        }

    #################################################
    # --- Input parsing / freezing ------------------
    #################################################
    def _unwrap_go_payload(self, instance: dict) -> tuple[dict, dict]:
        """Unwrap Go payload wrapper."""
        solver_input = (instance or {}).get("solver_input") or {}
        solver_options = (instance or {}).get("solver_options") or {}
        return solver_input, solver_options

    def _read_input(self, solver_input: dict) -> tuple[list[dict], list[dict], Optional[dict]]:
        """Read the main solver input payload."""
        nodes = solver_input.get("nodes") or []
        pods = solver_input.get("pods") or []
        preemptor = solver_input.get("preemptor") or None
        return nodes, pods, preemptor

    def _parse_options(self, solver_input: dict, solver_options: dict) -> SolverOptions:
        """Parse the option fields."""
        timeout_ms = int(solver_input.get("timeout_ms", 3000)) - 200
        ignore_affinity = bool(solver_input.get("ignore_affinity", True))
        log_progress = bool(solver_options.get("log_progress", False))
        guaranteed_tier_fraction = float(solver_options.get("guaranteed_tier_fraction", 0.6))
        move_fraction_of_tier = float(solver_options.get("move_fraction_of_tier", 0.5))
        gap_limit = float(solver_options.get("gap_limit", 0.00))
        return SolverOptions(
            timeout_ms=timeout_ms,
            ignore_affinity=ignore_affinity,
            log_progress=log_progress,
            guaranteed_tier_fraction=guaranteed_tier_fraction,
            move_fraction_of_tier=move_fraction_of_tier,
            gap_limit=gap_limit,
        )

    def _create_solver(self, options: SolverOptions) -> Optional[pywraplp.Solver]:
        """Create and configure the Glop solver."""
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            return None
        # Set time limit in milliseconds
        solver.SetTimeLimit(max(0, options.timeout_ms))
        return solver

    def _dedupe_pods_by_uid(self, pods: list[dict]) -> dict[str, dict]:
        """Deduplicate pod records by UID."""
        pod_by_uid: dict[str, dict] = {}
        for p in pods or []:
            uid = p.get("uid")
            if not uid:
                continue
            old = pod_by_uid.get(uid)
            if old is None:
                pod_by_uid[uid] = p
                continue
            old_has_node = bool((old.get("node") or "").strip())
            new_has_node = bool((p.get("node") or "").strip())
            if new_has_node and not old_has_node:
                pod_by_uid[uid] = p
        return pod_by_uid

    def _apply_preemptor_to_pods(
        self,
        pod_by_uid: dict[str, dict],
        preemptor: Optional[dict],
    ) -> tuple[dict[str, dict], bool, Optional[str]]:
        """Preemptor (per-pod) mode setup."""
        single_preemptor_mode = False
        preemptor_uid: Optional[str] = None
        if isinstance(preemptor, dict) and preemptor.get("uid"):
            preemptor_uid = preemptor["uid"]
            single_preemptor_mode = True
            if preemptor_uid not in pod_by_uid:
                pod_by_uid[preemptor_uid] = {
                    "uid": preemptor["uid"],
                    "namespace": preemptor.get("namespace", "default"),
                    "name": preemptor.get("name", "preemptor"),
                    "req_cpu_m": int(preemptor.get("req_cpu_m", 0)),
                    "req_mem_bytes": int(preemptor.get("req_mem_bytes", 0)),
                    "priority": int(preemptor.get("priority", 0)),
                    "protected": bool(preemptor.get("protected", False)),
                    "node": "",
                }
        return pod_by_uid, single_preemptor_mode, preemptor_uid

    def _freeze_problem(
        self,
        *,
        nodes: list[dict],
        pods: list[dict],
        single_preemptor_mode: bool,
        preemptor_uid: Optional[str],
    ) -> Problem | dict:
        """Freeze nodes/pods into a compact indexed representation."""
        num_nodes = len(nodes)
        num_pods = len(pods)
        if num_nodes == 0:
            return {"status": NO_NODES}
        if num_pods == 0:
            return {"status": NO_PODS}

        node_idx = {n["name"]: j for j, n in enumerate(nodes)}

        node_names = [n.get("name", "") for n in nodes]
        node_cap_cpu_m = [int(n.get("cap_cpu_m", 0)) for n in nodes]
        node_cap_mem_bytes = [int(n.get("cap_mem_bytes", 0)) for n in nodes]

        pod_uid = [p.get("uid", "") for p in pods]
        pod_namespace = [p.get("namespace", "default") for p in pods]
        pod_name = [p.get("name", "") for p in pods]
        pod_req_cpu_m = [int(p.get("req_cpu_m", 0)) for p in pods]
        pod_req_mem_bytes = [int(p.get("req_mem_bytes", 0)) for p in pods]
        pod_priority = [int(p.get("priority", 0)) for p in pods]
        pod_protected = [bool(p.get("protected", False)) for p in pods]
        pod_node_j: list[Optional[int]] = []
        for p in pods:
            w = (p.get("node") or "").strip()
            pod_node_j.append(node_idx.get(w) if w else None)

        running_idxs = [i for i in range(num_pods) if pod_node_j[i] is not None]
        pending_idxs = [i for i in range(num_pods) if pod_node_j[i] is None]

        preemptor_idx: Optional[int] = None
        if single_preemptor_mode and preemptor_uid is not None:
            for i in range(num_pods):
                if pod_uid[i] == preemptor_uid:
                    preemptor_idx = i
                    break
            if preemptor_idx is None:
                single_preemptor_mode = False

        eligible_nodes: list[list[int]] = []
        for i in range(num_pods):
            cpu_i, mem_i = pod_req_cpu_m[i], pod_req_mem_bytes[i]
            lst: list[int] = []
            for j in range(num_nodes):
                if node_cap_cpu_m[j] >= cpu_i and node_cap_mem_bytes[j] >= mem_i:
                    lst.append(j)
            eligible_nodes.append(lst)

        eligible_pos: list[dict[int, int]] = [
            {j: pos for pos, j in enumerate(eligible_nodes[i])} for i in range(num_pods)
        ]

        return Problem(
            nodes=nodes,
            pods=pods,
            node_idx=node_idx,
            num_nodes=num_nodes,
            num_pods=num_pods,
            node_names=node_names,
            node_cap_cpu_m=node_cap_cpu_m,
            node_cap_mem_bytes=node_cap_mem_bytes,
            pod_uid=pod_uid,
            pod_namespace=pod_namespace,
            pod_name=pod_name,
            pod_req_cpu_m=pod_req_cpu_m,
            pod_req_mem_bytes=pod_req_mem_bytes,
            pod_priority=pod_priority,
            pod_protected=pod_protected,
            pod_node_j=pod_node_j,
            running_idxs=running_idxs,
            pending_idxs=pending_idxs,
            eligible_nodes=eligible_nodes,
            eligible_pos=eligible_pos,
            single_preemptor_mode=single_preemptor_mode,
            preemptor_uid=preemptor_uid,
            preemptor_idx=preemptor_idx,
        )

    #################################################
    # --- Model building ----------------------------
    #################################################
    def _build_decision_vars(self, solver: pywraplp.Solver, problem: Problem) -> DecisionVars:
        """Build decision variables for the LP model.

        In LP relaxation, we use continuous variables in [0, 1] instead of binary.
        The final solution is rounded to get integer assignments.
        """
        placed = [
            solver.NumVar(0.0, 1.0, f"placed_{i}")
            for i in range(problem.num_pods)
        ]
        assign = [
            [
                solver.NumVar(0.0, 1.0, f"assign_{i}_{j}")
                for j in problem.eligible_nodes[i]
            ]
            for i in range(problem.num_pods)
        ]
        return DecisionVars(placed=placed, assign=assign)

    def _add_capacity_constraints(
        self, solver: pywraplp.Solver, problem: Problem, vars: DecisionVars
    ) -> None:
        """Add node capacity constraints."""
        for j in range(problem.num_nodes):
            cpu_expr = solver.Sum([])
            mem_expr = solver.Sum([])
            for i in range(problem.num_pods):
                pos = problem.eligible_pos[i].get(j)
                if pos is not None:
                    cpu_expr += vars.assign[i][pos] * problem.pod_req_cpu_m[i]
                    mem_expr += vars.assign[i][pos] * problem.pod_req_mem_bytes[i]
            solver.Add(cpu_expr <= problem.node_cap_cpu_m[j])
            solver.Add(mem_expr <= problem.node_cap_mem_bytes[j])

    def _add_assign_constraints(
        self, solver: pywraplp.Solver, problem: Problem, vars: DecisionVars
    ) -> None:
        """Add assignment constraints: sum of assignments equals placed."""
        for i in range(problem.num_pods):
            if problem.eligible_nodes[i]:
                solver.Add(solver.Sum(vars.assign[i]) == vars.placed[i])
            else:
                solver.Add(vars.placed[i] == 0)

    def _add_protected_stay_constraints(
        self,
        solver: pywraplp.Solver,
        problem: Problem,
        vars: DecisionVars,
    ) -> Optional[dict]:
        """Add constraints for protected pods to stay on their current node."""
        for i in problem.running_idxs:
            if problem.pod_protected[i]:
                orig = problem.pod_node_j[i]
                pos = problem.eligible_pos[i].get(orig)
                if pos is not None:
                    solver.Add(vars.assign[i][pos] == 1)
                else:
                    return {"status": "MODEL_INVALID"}
        return None

    def _add_mode_specific_constraints(
        self, solver: pywraplp.Solver, problem: Problem, vars: DecisionVars
    ) -> None:
        """Add mode-specific constraints."""
        if problem.single_preemptor_mode and problem.preemptor_idx is not None:
            solver.Add(solver.Sum(vars.assign[problem.preemptor_idx]) == 1)
        else:
            if problem.pending_idxs:
                solver.Add(
                    solver.Sum([vars.placed[i] for i in problem.pending_idxs]) >= 1
                )

    #################################################
    # --- Optimization ------------------------------
    #################################################
    def _solve_lexicographically(
        self,
        solver: pywraplp.Solver,
        problem: Problem,
        vars: DecisionVars,
        options: SolverOptions,
    ) -> tuple[int, list[dict]]:
        """Run optimization using weighted objectives.

        For LP, we use a weighted objective approach instead of true
        lexicographic solving since Glop doesn't support incremental solving
        well. We combine:
        1. Maximize placements (weighted by priority)
        2. Minimize disruption (evictions + moves)

        The weights are chosen to ensure priority ordering is respected.
        """
        total_sec = max(0.0, options.timeout_ms / 1000.0)
        phases: list[dict] = []

        # Build weighted objective
        # Higher priority pods get higher placement weight
        priorities = sorted(
            {problem.pod_priority[i] for i in range(problem.num_pods)}, reverse=True
        )
        
        # Create priority weights - each priority level gets exponentially higher weight
        # to ensure lexicographic-like behavior
        max_prio = max(priorities) if priorities else 0
        min_prio = min(priorities) if priorities else 0
        prio_range = max(1, max_prio - min_prio + 1)
        
        def get_priority_weight(prio: int) -> float:
            """Get weight for a priority level. Higher priority = higher weight."""
            # Use exponential weighting to ensure strict priority ordering
            level = prio - min_prio  # 0 to prio_range-1
            return float(10 ** level)

        def orig_node_var(i: int):
            """Get the assignment variable for pod i on its original node."""
            orig = problem.pod_node_j[i]
            pos = problem.eligible_pos[i].get(orig)
            return vars.assign[i][pos] if pos is not None else None

        # Build combined objective:
        # Maximize: sum(priority_weight * placed[i]) - small_penalty * disruption
        # Disruption penalty is smaller than placement benefit to prioritize placement
        objective = solver.Objective()
        
        # Placement terms (positive = maximize placement)
        for i in range(problem.num_pods):
            weight = get_priority_weight(problem.pod_priority[i])
            objective.SetCoefficient(vars.placed[i], weight)
        
        # Disruption terms for running pods (negative = minimize disruption)
        # Use a smaller weight to not override placement decisions
        disruption_penalty = 0.1  # Small penalty for disruption
        for i in problem.running_idxs:
            orig_var = orig_node_var(i)
            if orig_var is not None:
                # Reward staying on original node
                objective.SetCoefficient(orig_var, disruption_penalty)
        
        objective.SetMaximization()
        solver.SetTimeLimit(int(total_sec * 1000))

        t0 = time.monotonic()
        st = solver.Solve()
        time_spent = min(total_sec, max(0.0, time.monotonic() - t0))

        phases.append({
            "tier": 0,
            "stage": "combined",
            "status": self._status_str(st),
            "duration_ms": round(time_spent * 1000),
            "relative_gap": "",
        })

        return st, phases

    #################################################
    # --- Output extraction -------------------------
    #################################################
    def _extract_plan(
        self,
        problem: Problem,
        vars: DecisionVars,
        solver: pywraplp.Solver,
    ) -> tuple[list[dict], list[dict]]:
        """Extract placements/evictions from the LP solution.

        Uses rounding to convert continuous solutions to binary decisions.
        """

        def get_var_value(var: pywraplp.Variable) -> float:
            """Get variable value, handling potential solver issues."""
            try:
                return var.solution_value()
            except Exception:
                return 0.0

        def is_pod_placed(i: int) -> bool:
            """Return whether pod i is placed (after rounding)."""
            return get_var_value(vars.placed[i]) >= ROUNDING_THRESHOLD

        def chosen_node_for_pod(i: int) -> Optional[int]:
            """Return the chosen node index for pod i based on highest assignment value."""
            best_j = None
            best_val = -1.0
            for local, j in enumerate(problem.eligible_nodes[i]):
                val = get_var_value(vars.assign[i][local])
                if val > best_val:
                    best_val = val
                    best_j = j
            # Only return if the best value is above threshold
            if best_val >= ROUNDING_THRESHOLD:
                return best_j
            return None

        def stayed_on_same_node(i: int) -> bool:
            """Check if pod stayed on same node."""
            orig = problem.pod_node_j[i]
            pos = problem.eligible_pos[i].get(orig)
            if pos is not None:
                return get_var_value(vars.assign[i][pos]) >= ROUNDING_THRESHOLD
            return False

        placements: list[dict] = []
        evictions: list[dict] = []

        for i in range(problem.num_pods):
            was_running = problem.pod_node_j[i] is not None
            is_placed = is_pod_placed(i)

            if was_running and not is_placed:
                evictions.append({
                    "uid": problem.pod_uid[i],
                    "name": problem.pod_name[i],
                    "namespace": problem.pod_namespace[i],
                    "node": problem.node_names[problem.pod_node_j[i]],
                })
                continue

            if is_placed:
                if was_running and stayed_on_same_node(i):
                    continue  # stayed - no placement entry

                chosen_j = chosen_node_for_pod(i)
                if chosen_j is None:
                    continue

                orig_j = problem.pod_node_j[i]
                moved = was_running and not stayed_on_same_node(i)
                if moved or orig_j is None:
                    placements.append({
                        "uid": problem.pod_uid[i],
                        "name": problem.pod_name[i],
                        "namespace": problem.pod_namespace[i],
                        "old_node": problem.node_names[orig_j] if orig_j is not None else "",
                        "node": problem.node_names[chosen_j],
                    })

        return placements, evictions

    def _compute_overall_status(self, phases: list[dict], st: int) -> str:
        """Compute the overall status string.

        Uses the same priority as CP-SAT solver for Go contract compatibility:
        MODEL_INVALID > UNKNOWN > INFEASIBLE > FEASIBLE > OPTIMAL
        """
        seen = {s.get("status") for s in phases or []}
        if "MODEL_INVALID" in seen:
            return "MODEL_INVALID"
        if "UNKNOWN" in seen:
            return "UNKNOWN"
        if "INFEASIBLE" in seen:
            return "INFEASIBLE"
        if "FEASIBLE" in seen:
            return "FEASIBLE"
        if seen:  # all OPTIMAL
            return "OPTIMAL"
        return self._status_str(st)


#################################################
# --- Main --------------------------------------
#################################################
def main():
    try:
        raw = sys.stdin.read()
        inst = json.loads(raw or "{}")
        solver = GlopSolver()
        out = solver.solve(inst if isinstance(inst, dict) else {})
        print(json.dumps(out))
    except Exception as e:
        err = {"status": "PYTHON_EXCEPTION", "error": str(e)}
        try:
            print(json.dumps(err))
        except Exception:
            print('{"status":"PYTHON_EXCEPTION","error":"unserializable error"}')


if __name__ == "__main__":
    main()
