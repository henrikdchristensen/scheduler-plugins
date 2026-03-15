#!/usr/bin/env bash
set -euo pipefail
# Can be run as: ./run_tests.sh [all|unit_py|unit_go|unit_all|int_kwok|integration] [--solver cp_sat|gurobi|all]

# Load environment variables
ENV_FILE="opt-prio.env"
echo "=== Load environment variables from ${ENV_FILE} ==="
# shellcheck source=/dev/null
set -a
source "${ENV_FILE}"
set +a
echo "Environment variables loaded."

# Default: run all tests
MODE="${1:-all}"

# Parse optional --solver argument (for integration tests)
INT_SOLVER_FILTER="cp_sat"  # default: run cp_sat only (use --solver all to include gurobi)
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --solver)
      INT_SOLVER_FILTER="${2:-all}"
      shift 2 || true
      ;;
    *)
      shift
      ;;
  esac
done

RUN_UNIT_PY=false
RUN_UNIT_GO=false
RUN_INT_KWOK=false

case "$MODE" in
  all|"")
    RUN_UNIT_PY=true
    RUN_UNIT_GO=true
    RUN_INT_KWOK=true
    ;;
  unit_all|unit)
    RUN_UNIT_PY=true
    RUN_UNIT_GO=true
    ;;
  unit_py|unit_python)
    RUN_UNIT_PY=true
    ;;
  unit_go)
    RUN_UNIT_GO=true
    ;;
  int_all|int|int_kwok|integration)
    RUN_INT_KWOK=true
    ;;
  *)
    echo "Usage: $0 [all|unit_py|unit_go|unit_all|int_kwok|integration] [--solver cp_sat|gurobi|all]" >&2
    echo "  all         - run unit tests + integration tests" >&2
    echo "  unit_all    - run Python and Go unit tests (default)" >&2
    echo "  unit        - alias for unit_all" >&2
    echo "  unit_py     - run only Python unit tests (pytest)" >&2
    echo "  unit_python - alias for unit_py" >&2
    echo "  unit_go     - run only Go unit tests (mypriorityoptimizer package)" >&2
    echo "  int_all     - run only integration tests with KWOK" >&2
    echo "  int         - alias for int_all" >&2
    echo "  int_kwok    - run only integration tests with KWOK" >&2
    echo "  integration - alias for int_kwok" >&2
    echo "" >&2
    echo "Options:" >&2
    echo "  --solver    - solver for integration tests: cp_sat, gurobi, or all (default: all)" >&2
    exit 1
    ;;
esac

# -------------------------------------------------------------------
# Shared settings for solver env (only used for integration tests)
# -------------------------------------------------------------------
ensure_python_solver_env() {
  echo "=== Ensuring Python solver environment ==="

  # Strip possible Windows \r from env file values
  PYTHON_SOLVER_OUT_SCRIPT_DIR="${PYTHON_SOLVER_OUT_SCRIPT_DIR%$'\r'}"
  PYTHON_SOLVER_SCRIPT_PATH="${PYTHON_SOLVER_SCRIPT_PATH%$'\r'}"

  # For local development/testing, use local directories instead of system paths
  # This avoids requiring sudo for running tests
  if [[ -z "${PYTHON_SOLVER_OUT_SCRIPT_DIR:-}" ]] || [[ "${PYTHON_SOLVER_OUT_SCRIPT_DIR}" == "/opt/"* ]]; then
    PYTHON_SOLVER_OUT_SCRIPT_DIR="${PWD}/.solver"
  fi
  # Venv lives inside the solver directory
  PYTHON_SOLVER_OUT_VENV_DIR="${PYTHON_SOLVER_OUT_SCRIPT_DIR}/venv"
  if [[ -z "${PYTHON_SOLVER_SCRIPT_PATH:-}" ]]; then
    PYTHON_SOLVER_SCRIPT_PATH="scripts/python_solver/solver_cp_sat.py"
  fi

  # Note: Integration tests run both solvers via pytest parametrization.
  # We copy all solver scripts so SOLVER_PATH can be dynamically selected per-test.
  echo "Copying solver scripts: solver_cp_sat.py, solver_gurobi.py"

  # Try to create directories
  mkdir -p "${PYTHON_SOLVER_OUT_SCRIPT_DIR}" "${PYTHON_SOLVER_OUT_VENV_DIR}"

  # Copy all solver scripts so integration tests can select dynamically
  cp "scripts/python_solver/solver_cp_sat.py" "${PYTHON_SOLVER_OUT_SCRIPT_DIR}/solver_cp_sat.py"
  cp "scripts/python_solver/solver_gurobi.py" "${PYTHON_SOLVER_OUT_SCRIPT_DIR}/solver_gurobi.py"

  # Create venv if missing
  if [[ ! -x "${PYTHON_SOLVER_OUT_VENV_DIR}/bin/python" ]]; then
    python -m venv "${PYTHON_SOLVER_OUT_VENV_DIR}"
  fi

  # Install solver requirements into that venv
  "${PYTHON_SOLVER_OUT_VENV_DIR}/bin/pip" install -r scripts/python_solver/requirements.txt
}

# -------------------------------------------------------------------
# Coverage folders
# -------------------------------------------------------------------
mkdir -p coverage/python/html
mkdir -p coverage/go

if "$RUN_UNIT_PY"; then
  echo "=== Running Python unit tests (pytest) ==="
  python -m pytest \
    scripts/test \
    --cov=scripts \
    --cov-config=pytest-cfg.coveragerc \
    --cov-report=term \
    --cov-report=html:coverage/python \
    --cov-report=term-missing \
    --cov-fail-under="${PYTHON_COVERAGE_FAIL_UNDER}" \
    --timeout=2
  echo "Python tests completed. Coverage HTML: coverage/python/index.html"
fi

if "$RUN_UNIT_GO"; then
  echo "=== Running Go unit tests (pkg/mypriorityoptimizer) ==="
  go test ./pkg/mypriorityoptimizer -timeout 3s -coverprofile=coverage/go/go_coverage.out
  go tool cover -func=coverage/go/go_coverage.out
  go tool cover -html=coverage/go/go_coverage.out -o coverage/go/coverage.html

  # Enforce minimum total coverage threshold
  THRESHOLD="${GO_COVERAGE_FAIL_UNDER}"
  TOTAL=$(go tool cover -func=coverage/go/go_coverage.out | awk '/total:/ {print $3}' | sed 's/%//')

  GREEN='\033[0;32m'
  RED='\033[0;31m'
  NC='\033[0m'

  # Print colored status + fail if below threshold
  awk -v t="$THRESHOLD" -v c="$TOTAL" -v g="$GREEN" -v r="$RED" -v n="$NC" '
    BEGIN {
      if (c+0 < t+0) {
        printf "%sGo total coverage: %s%% (threshold: %s%%)%s\n", r, c, t, n
        print "Go coverage below threshold"
        exit 1
      } else {
        printf "%sGo total coverage: %s%% (threshold: %s%%)%s\n", g, c, t, n
      }
    }'

  echo "Go coverage reports generated in coverage/go/"
fi

if "$RUN_INT_KWOK"; then
  echo "=== Running Integration tests with KWOK ==="

  echo "Building kube-scheduler with mypriorityoptimizer plugin..."
  HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
  case "$(uname -m)" in
    x86_64)  HOST_ARCH="amd64" ;;
    aarch64|arm64) HOST_ARCH="arm64" ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
  esac
  make build-scheduler GO_BUILD_ENV="CGO_ENABLED=0 GOOS=${HOST_OS} GOARCH=${HOST_ARCH}" VERSION=${SCHEDULER_VERSION}

  ensure_python_solver_env

  # Export solver environment variables so Go scheduler can find the solver
  export SOLVER_PATH="${PYTHON_SOLVER_OUT_SCRIPT_DIR}/solver.py"
  export SOLVER_PYTHON_BIN="${PYTHON_SOLVER_OUT_VENV_DIR}/bin/python"
  echo "SOLVER_PATH=${SOLVER_PATH}"
  echo "SOLVER_PYTHON_BIN=${SOLVER_PYTHON_BIN}"

  python -m pip install --upgrade pip
  if [ -f scripts/kwok_integration_tests/requirements.txt ]; then
    python -m pip install -r scripts/kwok_integration_tests/requirements.txt
  fi

  # Build pytest filter based on solver selection
  PYTEST_SOLVER_FILTER=""
  case "${INT_SOLVER_FILTER}" in
    cp_sat)
      PYTEST_SOLVER_FILTER="-k cp_sat"
      echo "Running integration tests with CP-SAT solver only"
      ;;
    gurobi)
      PYTEST_SOLVER_FILTER="-k gurobi"
      echo "Running integration tests with Gurobi solver only"
      ;;
    all|*)
      echo "Running integration tests with all solvers (cp_sat, gurobi)"
      ;;
  esac

  # shellcheck disable=SC2086
  python -m pytest -s scripts/kwok_integration_tests/test_modes.py ${PYTEST_SOLVER_FILTER}

  echo "Integration tests with KWOK completed."
fi

# Summary
if "$RUN_UNIT_PY" && "$RUN_UNIT_GO"; then
  echo "Unit coverage:"
  echo "  Python: coverage/python/index.html"
  echo "  Go:     coverage/go/coverage.html"
elif "$RUN_UNIT_PY"; then
  echo "Python unit coverage: coverage/python/index.html"
elif "$RUN_UNIT_GO"; then
  echo "Go unit coverage: coverage/go/coverage.html"
fi
