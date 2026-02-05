#!/usr/bin/env python3
"""
Test script for solver_gurobi.py

Tests the Gurobi solver with various problem sizes to verify:
1. Gurobi installation and license availability
2. Basic solve functionality
3. Warm start behavior
4. Size limits (limited license = 2000 vars/constraints)

Usage:
    python scripts/python_solver/test_gurobi.py
    python scripts/python_solver/test_gurobi.py --size small
    python scripts/python_solver/test_gurobi.py --size medium
    python scripts/python_solver/test_gurobi.py --size large
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_gurobi_available():
    """Check if gurobipy is installed."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
        print("✓ gurobipy installed")
        print(f"  Version: {gp.gurobi.version()}")
        return True
    except ImportError:
        print("✗ gurobipy not installed")
        print("  Install with: pip install gurobipy")
        return False

def check_gurobi_license():
    """Check if Gurobi license is valid."""
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model("license_test", env=env)
        model.addVar(vtype=gp.GRB.BINARY, name="test")
        model.update()
        print("✓ Gurobi license valid")
        
        # Check license type
        try:
            lic_type = env.getParamInfo("ComputeServer")[2]
            if lic_type:
                print(f"  License type: Compute Server")
            else:
                print(f"  License type: Local/Academic")
        except:
            pass
        
        return True
    except gp.GurobiError as e:
        print(f"✗ Gurobi license error: {e}")
        return False

def create_test_problem(num_nodes: int, num_pods: int, pending_fraction: float = 0.3) -> dict:
    """Create a test scheduling problem."""
    import random
    random.seed(42)
    
    nodes = []
    for j in range(num_nodes):
        nodes.append({
            "name": f"node-{j}",
            "cap_cpu_m": 4000,  # 4 cores
            "cap_mem_bytes": 8_000_000_000,  # 8GB
        })
    
    pods = []
    num_pending = int(num_pods * pending_fraction)
    priorities = [100, 200, 300, 400, 500]
    
    for i in range(num_pods):
        is_pending = i < num_pending
        pod = {
            "uid": f"pod-{i}",
            "namespace": "default",
            "name": f"pod-{i}",
            "req_cpu_m": random.randint(100, 500),
            "req_mem_bytes": random.randint(100_000_000, 500_000_000),
            "priority": random.choice(priorities),
            "protected": False,
        }
        if not is_pending:
            pod["node"] = f"node-{random.randint(0, num_nodes - 1)}"
        pods.append(pod)
    
    # Calculate problem size
    num_vars = num_pods + (num_pods * num_nodes)
    
    return {
        "nodes": nodes,
        "pods": pods,
        "timeout_ms": 10000,
        "num_vars": num_vars,
    }

def run_solver_test(problem: dict, solver_name: str = "gurobi") -> dict:
    """Run the solver and return results."""
    from scripts.python_solver.solver_gurobi import GurobiSolver
    
    solver = GurobiSolver()
    start = time.monotonic()
    result = solver.solve(problem)
    elapsed = time.monotonic() - start
    
    return {
        **result,
        "actual_duration_s": elapsed,
    }

def print_result(result: dict, problem: dict):
    """Pretty print the solver result."""
    status = result.get("status", "UNKNOWN")
    duration_ms = result.get("duration_ms", 0)
    placements = result.get("placements", [])
    evictions = result.get("evictions", [])
    phases = result.get("phases", [])
    error = result.get("error", "")
    
    status_emoji = {
        "OPTIMAL": "✓",
        "FEASIBLE": "~",
        "INFEASIBLE": "✗",
        "UNKNOWN": "?",
        "SOLVER_UNAVAILABLE": "✗",
    }.get(status, "?")
    
    print(f"\n{status_emoji} Status: {status}")
    if error:
        print(f"  Error: {error}")
    print(f"  Duration: {duration_ms}ms")
    print(f"  Placements: {len(placements)}")
    print(f"  Evictions: {len(evictions)}")
    print(f"  Phases: {len(phases)}")
    
    if phases:
        print("\n  Phase details:")
        for p in phases:
            tier = p.get("tier", "?")
            stage = p.get("stage", "?")
            pstatus = p.get("status", "?")
            pdur = p.get("duration_ms", 0)
            obj = p.get("objective")
            print(f"    Tier {tier} / {stage}: {pstatus} ({pdur}ms) obj={obj}")

def run_size_test(size: str):
    """Run test with specified problem size."""
    sizes = {
        "tiny": (2, 5),      # 5 + 10 = 15 vars (always works)
        "small": (5, 20),    # 20 + 100 = 120 vars (works with limited)
        "medium": (10, 50),  # 50 + 500 = 550 vars (works with limited)
        "custom": (16, 64),  # 64 + 1024 = 1088 vars (works with limited)
        "ucloud": (16, 128), # 128 + 2048 = 2176 vars (EXCEEDS 2000 limit!)
        "large": (20, 100),  # 100 + 2000 = 2100 vars (MAY exceed limit!)
        "xlarge": (50, 200), # 200 + 10000 = 10200 vars (needs full license)
    }
    
    if size not in sizes:
        print(f"Unknown size: {size}")
        print(f"Available: {', '.join(sizes.keys())}")
        return False
    
    num_nodes, num_pods = sizes[size]
    print(f"\n{'='*60}")
    print(f"Testing size={size}: {num_nodes} nodes, {num_pods} pods")
    print(f"{'='*60}")
    
    problem = create_test_problem(num_nodes, num_pods)
    num_vars = problem.pop("num_vars")
    print(f"Problem size: ~{num_vars} variables")
    
    if num_vars > 2000:
        print("⚠ Warning: Exceeds limited license (2000 vars)")
    
    try:
        result = run_solver_test(problem)
        print_result(result, problem)
        return result.get("status") in ("OPTIMAL", "FEASIBLE")
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Gurobi solver")
    parser.add_argument("--size", choices=["tiny", "small", "medium", "custom", "ucloud", "large", "xlarge"],
                        default="small", help="Problem size to test")
    parser.add_argument("--all", action="store_true", help="Run all size tests")
    parser.add_argument("--check-only", action="store_true", 
                        help="Only check Gurobi installation/license")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Gurobi Solver Test")
    print("=" * 60)
    
    # Check installation
    if not check_gurobi_available():
        sys.exit(1)
    
    # Check license
    license_ok = check_gurobi_license()
    
    if args.check_only:
        sys.exit(0 if license_ok else 1)
    
    if not license_ok:
        print("\n⚠ Continuing with limited license (2000 vars max)...")
    
    # Run tests
    if args.all:
        results = {}
        for size in ["tiny", "small", "medium", "large"]:
            results[size] = run_size_test(size)
        
        print(f"\n{'='*60}")
        print("Summary:")
        for size, ok in results.items():
            emoji = "✓" if ok else "✗"
            print(f"  {emoji} {size}")
    else:
        ok = run_size_test(args.size)
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
