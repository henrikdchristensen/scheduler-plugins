// args.go
package mypriorityoptimizer

// ======= Optimality where/when settings =======

// OptimizeMode is the frequency at which optimization is performed. Choices:
// "scheduling_failure", "periodic", "stable_queue", "manual", "manual_blocking"
var OptimizeMode = parseOptimizeMode(getEnv("OPTIMIZE_MODE", "periodic"))

// OptimizeBlockingSolving controls whether solver runs blocks normal scheduling
// during execution. Determines when to take the Active lock.
var OptimizeBlockingSolving = parseBool(getEnv("OPTIMIZE_BLOCKING_SOLVING", "false"))

// OptimizePeriodicInterval is the duration between consecutive optimization
// runs in periodic mode. If a plan is currently active, the loop is skipped.
var OptimizePeriodicInterval = parseTime(getEnv("OPTIMIZE_PERIODIC_INTERVAL", "8s"))

// OptimizeStableQueueDelay is the duration of idle time (no changes in the
// pending set) before triggering stable queue optimization.
var OptimizeStableQueueDelay = parseTime(getEnv("OPTIMIZE_STABLE_QUEUE_DELAY", "2s"))

// OptimizeStableQueueCheckInterval is the interval at which we poll for stable queue
// "free time" conditions.
var OptimizeStableQueueCheckInterval = parseTime(getEnv("OPTIMIZE_STABLE_QUEUE_CHECK_INTERVAL", "250ms"))

// ===============================

// ======= Solver settings =======

// Save failed attempts to config map (for debugging)
var SolverSaveAllAttempts = parseBool(getEnv("SOLVER_SAVE_FAILED_ATTEMPTS", "true"))

// SolverPythonEnabled indicates whether the Python solver is enabled.
var SolverPythonEnabled = parseBool(getEnv("SOLVER_PYTHON_ENABLED", "false"))

// SolverPythonTimeout is the timeout for the python solver to complete.
var SolverPythonTimeout = parseTime(getEnv("SOLVER_PYTHON_TIMEOUT", "16s"))

// SolverPythonScriptPath is the path to the solver executable.
// Default is /opt/solver/solver.py which is populated by bootstrap.sh based on SOLVER_TYPE.
// Can be overridden to use a specific solver script directly.
var SolverPythonScriptPath = getEnv("SOLVER_PATH", "/opt/solver/solver.py")

// Path to the Python binary to use for running the solver.
var SolverPythonBin = getEnv("SOLVER_PYTHON_BIN", "/opt/venv/bin/python")

// SolverPythonGapLimit is the gap to optimality for the python solver (0.00 =
// optimal).
var SolverPythonGapLimit = parseFloat(getEnv("SOLVER_PYTHON_GAP_LIMIT", "0.00"), 0.00, 1.00)

// SolverPythonGuaranteedTierFraction is the guaranteed fraction of time for all
// tiers (0.00-1.00).
var SolverPythonGuaranteedTierFraction = parseFloat(getEnv("SOLVER_PYTHON_GUARANTEED_TIER_FRACTION", "0.40"), 0.00, 1.00)

// SolverPythonMoveFractionOfTier is the fraction of a tier's budget for moves
// (0.00-1.00).
var SolverPythonMoveFractionOfTier = parseFloat(getEnv("SOLVER_PYTHON_MOVE_FRACTION_OF_TIER", "0.30"), 0.00, 1.00)

// SolverPythonGraceMs is the grace period for the python solver (ms).
var SolverPythonGraceMs = parseInt(getEnv("SOLVER_PYTHON_GRACE_MS", "1000"))

// ===============================
