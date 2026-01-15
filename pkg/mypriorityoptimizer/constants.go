// constants.go
package mypriorityoptimizer

import "time"

const (
	// ================ General settings =======================

	// Name is the name of the component.
	Name = "MyPriorityOptimizer"
	// PluginVersion is the current version of the plugin.
	PluginVersion = "v0.0.1"
	// MyV is the klog verbosity level; set to 0 for extra verbose logging.
	// Cannot be added to args.go as it needs to be a constant for build tags.
	MyV = 2
	// SystemNamespace is the namespace in which the plugin operates.
	// Used to prevent deletion of configmaps when cleaning up pods for a new run.
	// For ease of use, it should match the kube-scheduler namespace.
	SystemNamespace = "kube-system"
	// CacheWarmupSettleDelay is the duration to wait before proceeding after cache has warmed up.
	CacheWarmupSettleDelay = 2 * time.Second
	// PluginReadinessUsableNodeInterval is the interval at which the plugin checks for readiness.
	PluginReadinessUsableNodeInterval = 250 * time.Millisecond
	// =========================================================

	// ================ Plugin config snapshot =================

	// PluginCfgConfigMapName is the name of the ConfigMap storing plugin configuration.
	// NB: Kubernetes resource names must be DNS-1123 compliant, so we use "plugin-config"
	// instead of "plugin_config" (underscores are not allowed in resource names).
	PluginCfgConfigMapName = "plugin-config"
	// PluginCfgConfigMapLabelKey is the label key used for plugin configuration ConfigMaps.
	PluginCfgConfigMapLabelKey = "plugin-config"
	// =========================================================

	// ================ Solver settings ========================

	// SolverLogProgress is a flag that enables/disables logging of solver progress.
	SolverLogProgress = false
	// SolverStatsConfigMapName is the name of the exported stats config map.
	SolverStatsConfigMapName = "solver-stats"
	// SolverStatsConfigMapLabelKey is the label key used for solver configuration config maps.
	SolverStatsConfigMapLabelKey = "runs"
	// Solver optimal status string.
	SolverStatusOptimal = "OPTIMAL"
	// Solver feasible status string.
	SolverStatusFeasible = "FEASIBLE"
	// Address the HTTP server should listen on (used for manual optimization and
	// debugging in all modes). Only works on a KWOK cluster if running with binary
	// runtime. Examples: ":18080", "0.0.0.0:18080"
	HTTPAddr = ":18080"
	// =========================================================

	// ================ Plan settings ==========================

	// Prefix for plan ConfigMaps.
	PlanConfigMapNamePrefix = "plan-"
	// PlanConfigMapLabelKey is the name of the ConfigMap used for plan configuration.
	PlanConfigMapLabelKey = "plan"
	// PlanCompletionCheckInterval is how often we check whether an active plan has reached its desired state.
	PlanCompletionCheckInterval = 250 * time.Millisecond
	// PlanPendingBindInterval is the interval at which pending binds are retried.
	PlanPendingBindInterval = 250 * time.Millisecond
	// PlansToRetain is the number of ConfigMaps plans to retain before the oldest are deleted.
	PlansToRetain = 32
	// The overall timeout for plan activation operations (like evictions and recreations).
	PlanActivationTimeout = 4 * time.Second
	// Interval for waiting for pods to be gone after eviction.
	WaitPodsGoneInterval = 250 * time.Millisecond
	// Degree of parallelism for eviction and recreate operations.
	EvictRecreateParallelism = 8
	// PlanRealizationTimeout is the maximum duration a plan may run before
	// being terminated.
	PlanRealizationTimeout = 4 * time.Second
	// =========================================================

	// ================ Loop config defaults ===================
	// OptimizePeriodicCancelOnChange indicates whether periodic optimization
	// runs should be cancelled if the pending set changes.
	OptimizePeriodicCancelOnChange = false // do NOT cancel if new pods arrive (can be made configurable later)
	// OptimizeStableQueueCancelOnChange indicates whether stable queue
	// optimization runs should be cancelled if the pending set changes.
	OptimizeStableQueueCancelOnChange = true
	// =========================================================
)
