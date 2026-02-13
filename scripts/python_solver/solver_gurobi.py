#!/usr/bin/env python3
# solver_gurobi.py
"""
This solver uses Gurobi's Python API (gurobipy) for mixed-integer programming. It uses the same optimization approach as the CP-SAT solver.
It can be used by setting SOLVER_TYPE=gurobi or SOLVER_PATH to point to this script.

Requirements:
    pip install gurobipy
    A valid Gurobi license (academic WLS, trial, or commercial)
    We noted that the using the WLS license a maximum of 5 jobs can run concurrently, properly due to token and session limits.
"""

import os
import time
import sys
import json
from dataclasses import dataclass
from typing import Optional, Final

# Import Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp = None
    GRB = None

#################################################
# --- Constants ---------------------------------
#################################################
NO_NODES: Final[str] = "NO_NODES"
NO_PODS: Final[str] = "NO_PODS"

# Gurobi status codes mapped to the same strings as CP-SAT
STATUS_MAP: Final[dict] = {}
if GUROBI_AVAILABLE:
    STATUS_MAP.update({
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "FEASIBLE",       # MIP with gap
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INFEASIBLE",    # Map to INFEASIBLE
        GRB.UNBOUNDED: "INFEASIBLE",      # Map to INFEASIBLE
        GRB.CUTOFF: "INFEASIBLE",         # Map to INFEASIBLE
        GRB.ITERATION_LIMIT: "FEASIBLE",  # Hit limit with solution
        GRB.NODE_LIMIT: "FEASIBLE",       # Hit limit with solution
        GRB.TIME_LIMIT: "FEASIBLE",       # Hit limit with solution
        GRB.SOLUTION_LIMIT: "FEASIBLE",   # Got requested solutions
        GRB.INTERRUPTED: "FEASIBLE",      # User interrupt with solution
        GRB.NUMERIC: "UNKNOWN",           # Numerical issues
        GRB.LOADED: "UNKNOWN",            # Not optimized yet
    })

# Binary variable threshold
BINARY_THRESHOLD: Final[float] = 0.5

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
    """Decision variables for the Gurobi MIP model.

    Uses true binary variables (0 or 1) for exact solutions.
    """

    # placed[i] = 1 if pod i is placed somewhere, 0 otherwise
    placed: list  # list of gp.Var
    # assign[i][local] = 1 if pod i is assigned to eligible_nodes[i][local]
    assign: list  # list of list of gp.Var


#################################################
# --- Gurobi MIP Solver Class -------------------
#################################################

class GurobiSolver:
    """Mixed Integer Programming solver using Gurobi for pod scheduling.
    
    Uses true binary variables for exact 0/1 solutions.
    Supports warm starts between optimization stages via MIP starts.
    """

    #################################################
    # --- Helpers -----------------------------------
    #################################################

    @classmethod
    def _status_str(cls, st: int) -> str:
        """Convert Gurobi solver status to string."""
        if not GUROBI_AVAILABLE:
            return "SOLVER_UNAVAILABLE"
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

        if not GUROBI_AVAILABLE:
            return {
                "status": "SOLVER_UNAVAILABLE",
                "duration_ms": int((time.monotonic() - started_at) * 1000),
                "placements": [],
                "evictions": [],
                "phases": [],
                "error": "gurobipy not installed. Install with: pip install gurobipy",
            }

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
        # --- Dedupe pods -------------------------------
        #################################################
        pod_by_uid = self._dedupe_pods_by_uid(pods)
        pod_by_uid, single_preemptor_mode, preemptor_uid = self._apply_preemptor_to_pods(
            pod_by_uid, preemptor
        )
        pods = list(pod_by_uid.values())

        #################################################
        # --- Freeze problem ----------------------------
        #################################################
        problem = self._freeze_problem(
            nodes=nodes,
            pods=pods,
            single_preemptor_mode=single_preemptor_mode,
            preemptor_uid=preemptor_uid,
        )
        if isinstance(problem, dict):
            elapsed = int((time.monotonic() - started_at) * 1000)
            return {**problem, "duration_ms": elapsed, "placements": [], "evictions": [], "phases": []}

        #################################################
        # --- Create model and variables ----------------
        #################################################
        model, model_err = self._create_model(options)
        if model is None:
            return {
                "status": "SOLVER_UNAVAILABLE",
                "duration_ms": int((time.monotonic() - started_at) * 1000),
                "placements": [],
                "evictions": [],
                "phases": [],
                "error": model_err or "Gurobi license not available or invalid",
            }

        dvars = self._build_decision_vars(model, problem)

        #################################################
        # --- Add constraints ---------------------------
        #################################################
        self._add_capacity_constraints(model, problem, dvars)
        self._add_assign_constraints(model, problem, dvars)
        err = self._add_protected_stay_constraints(model, problem, dvars)
        if err is not None:
            elapsed = int((time.monotonic() - started_at) * 1000)
            return {**err, "duration_ms": elapsed, "placements": [], "evictions": [], "phases": []}

        self._add_mode_specific_constraints(model, problem, dvars)

        #################################################
        # --- Solve (tiered lexicographic) --------------
        #################################################
        st, phases, solution = self._solve_lexicographically(model, problem, dvars, options)

        #################################################
        # --- Extract plan ------------------------------
        #################################################
        placements, evictions = self._extract_plan(problem, dvars, solution)
        overall_status = self._compute_overall_status(phases, st)

        elapsed = int((time.monotonic() - started_at) * 1000)
        return {
            "status": overall_status,
            "duration_ms": elapsed,
            "placements": placements,
            "evictions": evictions,
            "phases": phases,
        }

    #################################################
    # --- Payload/options helpers -------------------
    #################################################
    def _unwrap_go_payload(self, instance: dict) -> tuple[dict, dict]:
        """Unwrap Go scheduler's payload layout."""
        solver_input = instance.get("solver_input") or instance
        solver_options = instance.get("solver_options") or {}
        if not isinstance(solver_input, dict):
            solver_input = {}
        if not isinstance(solver_options, dict):
            solver_options = {}
        return solver_input, solver_options

    def _read_input(self, solver_input: dict) -> tuple[list, list, Optional[dict]]:
        """Read nodes, pods, preemptor from payload."""
        nodes = solver_input.get("nodes") or []
        pods = solver_input.get("pods") or []
        preemptor = solver_input.get("preemptor")
        return nodes, pods, preemptor

    def _parse_options(self, solver_input: dict, solver_options: dict) -> SolverOptions:
        """Parse solver options from both input and options dict."""
        timeout_ms = int(solver_input.get("timeout_ms", 3000)) - 200
        ignore_affinity = bool(solver_options.get("ignore_affinity", True))
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

    def _create_model(self, options: SolverOptions):
        """Create and configure the Gurobi model."""
        if not GUROBI_AVAILABLE:
            return None, "gurobipy not installed"
        try:
            # Create model with suppressed console output
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)  # Suppress console output

            # WLS license credentials (academic Web License Service)
            # Override via env vars GRB_WLSACCESSID, GRB_WLSSECRET, GRB_LICENSEID if needed. Parameters can be found in the license file.
            wls_access_id = os.environ.get('GRB_WLSACCESSID', '<your_access_id_here>')
            wls_secret = os.environ.get('GRB_WLSSECRET', '<your_secret_here>')
            wls_license_id = os.environ.get('GRB_LICENSEID', '<your_license_id_here>')
            if wls_access_id and wls_secret and wls_license_id:
                env.setParam('WLSACCESSID', wls_access_id)
                env.setParam('WLSSECRET', wls_secret)
                env.setParam('LICENSEID', int(wls_license_id))

            env.start()
            model = gp.Model("pod_scheduler", env=env)
            # Set time limit in seconds
            model.Params.TimeLimit = max(0.0, options.timeout_ms / 1000.0)
            # Set MIP gap limit
            model.Params.MIPGap = options.gap_limit
            # Enable warm starts
            model.Params.StartNodeLimit = -1  # Use all MIP starts
            return model, None
        except gp.GurobiError as e:
            # License or other Gurobi error - capture the message
            return None, f"Gurobi error: {e}"

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
                    "namespace": preemptor.get("namespace", ""),
                    "name": preemptor.get("name", ""),
                    "req_cpu_m": int(preemptor.get("req_cpu_m", 0)),
                    "req_mem_bytes": int(preemptor.get("req_mem_bytes", 0)),
                    "priority": int(preemptor.get("priority", 0)),
                    "protected": bool(preemptor.get("protected", False)),
                    "node": None,
                }
        return pod_by_uid, single_preemptor_mode, preemptor_uid

    def _freeze_problem(
        self,
        nodes: list[dict],
        pods: list[dict],
        single_preemptor_mode: bool,
        preemptor_uid: Optional[str],
    ):
        """Freeze and validate the problem, returning a Problem or error dict."""
        if not nodes:
            return {"status": NO_NODES}
        if not pods:
            return {"status": NO_PODS}

        node_idx = {n["name"]: j for j, n in enumerate(nodes)}
        num_nodes = len(nodes)
        num_pods = len(pods)

        node_names = [n["name"] for n in nodes]
        node_cap_cpu_m = [int(n.get("cap_cpu_m", 0)) for n in nodes]
        node_cap_mem_bytes = [int(n.get("cap_mem_bytes", 0)) for n in nodes]

        pod_uid = [p["uid"] for p in pods]
        pod_namespace = [p.get("namespace", "") for p in pods]
        pod_name = [p.get("name", "") for p in pods]
        pod_req_cpu_m = [int(p.get("req_cpu_m", 0)) for p in pods]
        pod_req_mem_bytes = [int(p.get("req_mem_bytes", 0)) for p in pods]
        pod_priority = [int(p.get("priority", 0)) for p in pods]
        pod_protected = [bool(p.get("protected", False)) for p in pods]

        pod_node_j: list[Optional[int]] = []
        for p in pods:
            nname = (p.get("node") or "").strip()
            if nname and nname in node_idx:
                pod_node_j.append(node_idx[nname])
            else:
                pod_node_j.append(None)

        running_idxs = [i for i in range(num_pods) if pod_node_j[i] is not None]
        pending_idxs = [i for i in range(num_pods) if pod_node_j[i] is None]

        # Eligible nodes (all for now - affinity TODO)
        eligible_nodes = [list(range(num_nodes)) for _ in range(num_pods)]
        eligible_pos = [{j: local for local, j in enumerate(enl)} for enl in eligible_nodes]

        preemptor_idx: Optional[int] = None
        if single_preemptor_mode and preemptor_uid:
            for i, uid in enumerate(pod_uid):
                if uid == preemptor_uid:
                    preemptor_idx = i
                    break

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

    def _build_decision_vars(self, model, problem: Problem) -> DecisionVars:
        """Build the binary decision variables for placement."""
        placed = []
        assign = []

        for i in range(problem.num_pods):
            # placed[i] = 1 if pod i is placed
            placed.append(model.addVar(vtype=GRB.BINARY, name=f"placed_{i}"))
            
            # assign[i][local] = 1 if pod i is on eligible_nodes[i][local]
            row = []
            for local, j in enumerate(problem.eligible_nodes[i]):
                row.append(model.addVar(vtype=GRB.BINARY, name=f"assign_{i}_{j}"))
            assign.append(row)

        model.update()  # Apply variable additions
        return DecisionVars(placed=placed, assign=assign)

    def _add_capacity_constraints(self, model, problem: Problem, dvars: DecisionVars):
        """Add node capacity constraints for CPU and memory."""
        for j in range(problem.num_nodes):
            # CPU capacity
            cpu_terms = []
            for i in range(problem.num_pods):
                pos = problem.eligible_pos[i].get(j)
                if pos is not None:
                    cpu_terms.append(problem.pod_req_cpu_m[i] * dvars.assign[i][pos])
            if cpu_terms:
                model.addConstr(
                    gp.quicksum(cpu_terms) <= problem.node_cap_cpu_m[j],
                    name=f"cpu_cap_{j}"
                )

            # Memory capacity
            mem_terms = []
            for i in range(problem.num_pods):
                pos = problem.eligible_pos[i].get(j)
                if pos is not None:
                    mem_terms.append(problem.pod_req_mem_bytes[i] * dvars.assign[i][pos])
            if mem_terms:
                model.addConstr(
                    gp.quicksum(mem_terms) <= problem.node_cap_mem_bytes[j],
                    name=f"mem_cap_{j}"
                )

    def _add_assign_constraints(self, model, problem: Problem, dvars: DecisionVars):
        """Ensure each pod is placed on at most one node, linked to placed var."""
        for i in range(problem.num_pods):
            if dvars.assign[i]:
                # Sum of assignments == placed[i]
                model.addConstr(
                    gp.quicksum(dvars.assign[i]) == dvars.placed[i],
                    name=f"assign_link_{i}"
                )

    def _add_protected_stay_constraints(
        self,
        model,
        problem: Problem,
        dvars: DecisionVars,
    ) -> Optional[dict]:
        """Force protected running pods to stay on their current node."""
        for i in problem.running_idxs:
            if not problem.pod_protected[i]:
                continue
            orig_j = problem.pod_node_j[i]
            if orig_j is None:
                continue
            pos = problem.eligible_pos[i].get(orig_j)
            if pos is None:
                return {
                    "status": "INFEASIBLE",
                    "error": f"protected pod {problem.pod_uid[i]} cannot stay on node {orig_j}",
                }
            # Must stay
            model.addConstr(dvars.assign[i][pos] == 1, name=f"protected_stay_{i}")
        return None

    def _add_mode_specific_constraints(self, model, problem: Problem, dvars: DecisionVars):
        """Add preemptor-mode specific constraints if applicable."""
        if not problem.single_preemptor_mode or problem.preemptor_idx is None:
            return
        pi = problem.preemptor_idx
        # Preemptor must be placed
        model.addConstr(dvars.placed[pi] == 1, name="preemptor_must_place")

    def _solve_lexicographically(
        self,
        model,
        problem: Problem,
        dvars: DecisionVars,
        options: SolverOptions,
    ) -> tuple[int, list[dict], dict]:
        """Run the tiered lexicographic optimization with warm starts.

        Returns:
            tuple of (status, phases, solution_dict)
        """

        #########################
        # Solution storage
        #########################
        solution: dict = {"placed": {}, "assign": {}}

        def capture_solution():
            """Capture current solution values before modifying the model."""
            for i in range(problem.num_pods):
                try:
                    solution["placed"][i] = dvars.placed[i].X
                except Exception:
                    solution["placed"][i] = 0.0
                solution["assign"][i] = {}
                for local in range(len(problem.eligible_nodes[i])):
                    try:
                        solution["assign"][i][local] = dvars.assign[i][local].X
                    except Exception:
                        solution["assign"][i][local] = 0.0

        def apply_warm_start():
            """Apply current solution as MIP start for next solve.
            
            Gurobi supports MIP starts via the Start attribute, which can
            significantly speed up subsequent solves.
            """
            for i in range(problem.num_pods):
                placed_val = solution["placed"].get(i, 0.0)
                dvars.placed[i].Start = placed_val
                
                for local in range(len(problem.eligible_nodes[i])):
                    assign_val = solution["assign"].get(i, {}).get(local, 0.0)
                    dvars.assign[i][local].Start = assign_val

        def apply_current_placement_as_hint():
            """Seed MIP starts from the current cluster placement.

            For each pod on an eligible node, we hint:
            - placed[i].Start = 1
            - assign[i][pos].Start = 1
            """
            for i in range(problem.num_pods):
                orig_j = problem.pod_node_j[i]
                if orig_j is None:
                    continue
                pos = problem.eligible_pos[i].get(orig_j)
                if pos is None:
                    continue
                dvars.placed[i].Start = 1.0
                dvars.assign[i][pos].Start = 1.0

        #########################
        # Solver helpers
        #########################
        total_sec = max(0.0, options.timeout_ms / 1000.0)
        deadline = time.monotonic() + total_sec

        def remaining_wall() -> float:
            return max(0.0, deadline - time.monotonic())

        def relative_gap(sense: str, obj: float, bound: float) -> Optional[float]:
            if obj is None or bound is None:
                return None
            denom = max(1.0, abs(obj))
            if sense == "max":
                return max(0.0, (bound - obj) / denom)
            else:
                return max(0.0, (obj - bound) / denom)

        def orig_node(i: int):
            """Get the assignment variable for pod i on its original node, or None."""
            orig = problem.pod_node_j[i]
            pos = problem.eligible_pos[i].get(orig)
            return dvars.assign[i][pos] if pos is not None else None

        def run_stage(obj_vars: list, coeffs: list, sense: str, cap_sec: float) -> dict:
            """Run a single optimization stage."""
            started = time.monotonic()
            wall_sec = min(cap_sec, remaining_wall())
            
            if wall_sec <= 0:
                return {
                    "status": "UNKNOWN",
                    "duration_ms": 0,
                    "objective": None,
                    "bound": None,
                    "relative_gap": None,
                }

            # Set time limit
            model.Params.TimeLimit = wall_sec
            
            # Build and set objective
            if obj_vars and coeffs:
                obj_expr = gp.quicksum(c * v for c, v in zip(coeffs, obj_vars))
                if sense == "max":
                    model.setObjective(obj_expr, GRB.MAXIMIZE)
                else:
                    model.setObjective(obj_expr, GRB.MINIMIZE)
            else:
                # Feasibility only
                model.setObjective(0, GRB.MINIMIZE)

            model.update()
            model.optimize()

            st = model.Status
            elapsed_ms = int((time.monotonic() - started) * 1000)

            # Extract objective/bound
            try:
                obj_val = model.ObjVal if model.SolCount > 0 else None
            except Exception:
                obj_val = None
            try:
                bound_val = model.ObjBound
            except Exception:
                bound_val = None

            gap = relative_gap(sense, obj_val, bound_val)

            # Convert gap to string to match expected JSON schema
            def gap_str(g: Optional[float]) -> str:
                return f"{g:.4f}" if isinstance(g, (int, float)) else ""

            return {
                "status": self._status_str(st),
                "duration_ms": elapsed_ms,
                "objective": obj_val,
                "bound": bound_val,
                "relative_gap": gap_str(gap),
            }

        #########################
        # Tiered optimization
        #########################
        phases: list[dict] = []
        final_status = GRB.LOADED if GUROBI_AVAILABLE else 0

        # Get priority tiers
        tiers = sorted(set(problem.pod_priority), reverse=True)
        if not tiers:
            tiers = [0]

        num_tiers = len(tiers)
        guaranteed_budget = options.guaranteed_tier_fraction * total_sec
        per_tier_guaranteed = guaranteed_budget / num_tiers if num_tiers else 0

        # Apply initial warm start from current placements
        apply_current_placement_as_hint()

        for tier_idx, tier in enumerate(tiers):
            # Pods at or above this tier
            tier_pod_idxs = [i for i in range(problem.num_pods) if problem.pod_priority[i] >= tier]
            tier_running = [i for i in tier_pod_idxs if problem.pod_node_j[i] is not None]

            # Time budget for this tier
            remaining = remaining_wall()
            tiers_left = num_tiers - tier_idx
            fair_share = remaining / tiers_left if tiers_left else remaining
            tier_budget = max(per_tier_guaranteed, fair_share)
            place_budget = tier_budget * (1.0 - options.move_fraction_of_tier)
            move_budget = tier_budget * options.move_fraction_of_tier

            ########################################
            # Stage 1: Maximize placements for tier
            ########################################
            place_vars = [dvars.placed[i] for i in tier_pod_idxs]
            place_coeffs = [1.0] * len(place_vars)

            place_result = run_stage(place_vars, place_coeffs, "max", place_budget)
            phases.append({
                "tier": tier,
                "stage": "place",
                **place_result,
            })
            final_status = model.Status

            # Capture solution and apply as warm start for next stage
            if model.SolCount > 0:
                capture_solution()
                
                # Lock in placements: sum(placed[i] for i in tier) >= achieved
                achieved_place = sum(
                    1 for i in tier_pod_idxs
                    if solution["placed"].get(i, 0) >= BINARY_THRESHOLD
                )
                if achieved_place > 0:
                    model.addConstr(
                        gp.quicksum(dvars.placed[i] for i in tier_pod_idxs) >= achieved_place,
                        name=f"lock_place_tier_{tier}"
                    )
                    model.update()

                apply_warm_start()

            ########################################
            # Stage 2: Minimize moves for running pods
            ########################################
            if tier_running:
                # Minimize number of pods that move (not staying on original node)
                move_vars = []
                move_coeffs = []
                for i in tier_running:
                    stay_var = orig_node(i)
                    if stay_var is not None:
                        # Minimize (1 - stay): effectively minimize moves
                        move_vars.append(stay_var)
                        move_coeffs.append(1.0)  # Maximize staying = minimize moving

                if move_vars:
                    move_result = run_stage(move_vars, move_coeffs, "max", move_budget)
                    phases.append({
                        "tier": tier,
                        "stage": "moves",
                        **move_result,
                    })
                    final_status = model.Status

                    if model.SolCount > 0:
                        capture_solution()
                        
                        # Lock in stay count
                        achieved_stay = sum(
                            1 for i in tier_running
                            if orig_node(i) is not None and solution["assign"].get(i, {}).get(
                                problem.eligible_pos[i].get(problem.pod_node_j[i]), 0
                            ) >= BINARY_THRESHOLD
                        )
                        if achieved_stay > 0 and move_vars:
                            model.addConstr(
                                gp.quicksum(move_vars) >= achieved_stay,
                                name=f"lock_stay_tier_{tier}"
                            )
                            model.update()

                        apply_warm_start()

        return final_status, phases, solution

    def _extract_plan(
        self,
        problem: Problem,
        dvars: DecisionVars,
        solution: dict,
    ) -> tuple[list[dict], list[dict]]:
        """Extract placements and evictions from the solution."""

        def is_pod_placed(i: int) -> bool:
            return solution.get("placed", {}).get(i, 0) >= BINARY_THRESHOLD

        def chosen_node_for_pod(i: int) -> Optional[int]:
            for local, j in enumerate(problem.eligible_nodes[i]):
                if solution.get("assign", {}).get(i, {}).get(local, 0) >= BINARY_THRESHOLD:
                    return j
            return None

        def stayed_on_same_node(i: int) -> bool:
            orig_j = problem.pod_node_j[i]
            if orig_j is None:
                return False
            pos = problem.eligible_pos[i].get(orig_j)
            if pos is None:
                return False
            return solution.get("assign", {}).get(i, {}).get(pos, 0) >= BINARY_THRESHOLD

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
                    continue

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
        if seen:
            return "OPTIMAL"
        return self._status_str(st)


#################################################
# --- Main --------------------------------------
#################################################
def main():
    try:
        raw = sys.stdin.read()
        # Log input size to stderr for debugging
        sys.stderr.write(f"[solver_gurobi] input size: {len(raw)} bytes\n")
        sys.stderr.flush()
        inst = json.loads(raw or "{}")
        solver = GurobiSolver()
        out = solver.solve(inst if isinstance(inst, dict) else {})
        # Log output status to stderr for debugging
        sys.stderr.write(f"[solver_gurobi] output status: {out.get('status', 'UNKNOWN')}\n")
        if out.get('error'):
            sys.stderr.write(f"[solver_gurobi] error: {out.get('error')}\n")
        sys.stderr.flush()
        print(json.dumps(out))
    except Exception as e:
        import traceback
        sys.stderr.write(f"[solver_gurobi] EXCEPTION: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        err = {"status": "PYTHON_EXCEPTION", "error": str(e)}
        try:
            print(json.dumps(err))
        except Exception:
            print('{"status":"PYTHON_EXCEPTION","error":"unserializable error"}')


if __name__ == "__main__":
    main()
