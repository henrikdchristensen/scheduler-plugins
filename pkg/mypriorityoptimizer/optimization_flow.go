// optimization_flow.go
package mypriorityoptimizer

import (
	"context"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

// -------------------------
// runOptimizationFlow
// -------------------------

// runOptimizationFlow runs the optimisation flow for the given phase. For
// Single phase, the preemptor must be provided. Returns the target node name
// for the preemptor pod (if any) and error (if any).
func (pl *SharedState) runOptimizationFlow(ctx context.Context, preemptor *v1.Pod) (*Plan, *SolverScore, string, *SolverResult, []SolverResult, error) {
	strategy := getModeCombinedAsString()

	// Cumulative persistent stats (one write per optimization-flow call).
	delta := &OptimizationStatsDelta{OptimizationFlowCalls: 1}
	defer func() {
		pl.persistOptimizationStatsDelta(context.Background(), *delta)
	}()

	// Sync modes: take PlanActive now.
	// Async modes: will take PlanActive later, after plan computation.
	holdingActivePlan := false
	if !isNonBlockingSolvingFn() {
		if !pl.tryEnterActivePlan() {
			klog.InfoS(msg(strategy, InfoActivePlanInProgress))
			return nil, nil, "", nil, nil, ErrActiveInProgress
		}
		holdingActivePlan = true
	}

	// leaveActivePlanIfHeld releases the active-plan lock only when we
	// actually acquired it, preventing a non-blocking flow from
	// accidentally releasing another goroutine's lock.
	leaveActivePlanIfHeld := func() {
		if holdingActivePlan {
			pl.tryLeaveActivePlan()
			holdingActivePlan = false
		}
	}

	// Ensure only one optimization flow at a time.
	if !pl.tryEnterOptimizationFlow() {
		klog.InfoS(msg(strategy, InfoOptimizationInProgress))
		leaveActivePlanIfHeld()
		return nil, nil, "", nil, nil, ErrOptimizationInProgress
	}
	delta.OptimizationFlowEntered = 1
	defer pl.tryLeaveOptimizationFlow()

	start := time.Now()

	// Plan context: snapshot, solver input, baseline, pending count.
	nodes, pods, inp, err := planContextFn(pl, preemptor)
	if err != nil {
		klog.Error(msg(strategy, InfoPlanContextFailed), "err", err)
		leaveActivePlanIfHeld()
		return nil, nil, "", nil, nil, err
	}
	baselineScore := inp.BaselineScore

	// Only proceed if there are pending pods to schedule.
	pendingPrePlan := countPendingPods(pods)
	if pendingPrePlan == 0 {
		klog.InfoS(msg(strategy, InfoNoPendingPods))
		leaveActivePlanIfHeld()
		return nil, &baselineScore, "", nil, nil, ErrNoPendingPods
	}

	// Plan computation
	bestName, hadImp, bestAttempt, bestOut, attempts := planComputationFn(pl, ctx, inp)

	// Count solver calls (attempts = one entry per enabled solver attempt run).
	delta.SolverAttempts = int64(len(attempts))

	// Check if any solver solution was improving, if not, exit early.
	if !hadImp {
		klog.Error(msg(strategy, InfoNoImprovingSolutionFromAnySolver))
		leaveActivePlanIfHeld()
		if len(attempts) > 0 {
			delta.BestSolverFailed = 1
		}
		exportSolverStatsFn(pl, strategy, baselineScore, bestName, attempts, ErrNoImprovingSolutionFromAnySolver.Error())
		return nil, &baselineScore, bestName, bestAttempt, attempts, ErrNoImprovingSolutionFromAnySolver
	}

	// Best solver status counters (when we have an improving solution).
	if bestAttempt != nil {
		if strings.EqualFold(bestAttempt.Status, SolverStatusOptimal) {
			delta.BestSolverOptimal = 1
		} else if strings.EqualFold(bestAttempt.Status, SolverStatusFeasible) {
			delta.BestSolverFeasible = 1
		}
	}

	// Verify that plan (still) can be applied
	// Mainly for async modes, where the cluster state may have changed since plan computation.
	ok, why := isSolutionApplicableFn(pl, bestOut, nodes, pods)
	if !ok {
		klog.Error(msg(strategy, InfoPlanNotApplicable), "solver", bestName, "status", bestOut.Status, "reason", why)
		leaveActivePlanIfHeld()
		delta.PlanNotApplicable = 1
		exportSolverStatsFn(pl, strategy, baselineScore, bestName, attempts, ErrPlanNotApplicable.Error())
		return nil, &baselineScore, bestName, bestAttempt, attempts, ErrPlanNotApplicable
	}

	// Async modes: take PlanActive now that we know it is worth applying the plan.
	if isNonBlockingSolvingFn() {
		if !pl.tryEnterActivePlan() {
			klog.InfoS(msg(strategy, InfoActivePlanInProgress))
			exportSolverStatsFn(pl, strategy, baselineScore, bestName, attempts, ErrActiveInProgress.Error())
			return nil, nil, "", nil, nil, ErrActiveInProgress
		}
		holdingActivePlan = true
	}

	// How much is actually schedulable?
	pendingScheduled, totalPrePlan, totalPostPlan := computePlanPodCountsFn(bestOut, pods)
	if pendingScheduled == 0 {
		klog.InfoS(msg(strategy, InfoNoPendingPodsScheduled))
		leaveActivePlanIfHeld()
		exportSolverStatsFn(pl, strategy, baselineScore, bestName, attempts, ErrNoPendingPodsScheduled.Error())
		return nil, &baselineScore, bestName, bestAttempt, attempts, ErrNoPendingPodsScheduled
	}

	// NOTE: If any error occurs from here on, we must call onPlanCompleted instead of just leaveActivePlan.

	// Plan registration
	plan, ap, err := planRegistrationFn(pl, ctx, *bestAttempt, bestOut, preemptor, pods)
	if err != nil {
		klog.Error(msg(strategy, InfoPlanRegistrationFailed))
		pl.onPlanCompleted(PlanStatusFailed)
		exportSolverStatsFn(pl, strategy, baselineScore, bestName, attempts, ErrPlanRegistration.Error())
		return nil, &baselineScore, bestName, bestAttempt, attempts, ErrPlanRegistration
	}

	// Plan eviction and recreate standalone pods
	if err := planActivationFn(pl, plan, pods); err != nil {
		klog.Error(msg(strategy, InfoPlanActivationFailed))
		pl.onPlanCompleted(PlanStatusFailed)
		exportSolverStatsFn(pl, strategy, baselineScore, bestName, attempts, ErrPlanActivationFailed.Error())
		return nil, &baselineScore, bestName, bestAttempt, attempts, ErrPlanActivationFailed
	}

	delta.PlanActivated = 1

	// Start a periodically plan completion watcher. The watcher stops itself.
	startPlanCompletionWatchFn(pl, ap)

	// Export stats (success)
	exportSolverStatsFn(pl, strategy, baselineScore, bestName, attempts, "")

	// Log summary
	klog.InfoS(
		msg(strategy, InfoPlanExecutionFinished),
		"planID", ap.ID,
		"bestAttempt", bestAttempt,
		"pendingPrePlan", pendingPrePlan,
		"pendingScheduled", pendingScheduled,
		"totalPrePlan", totalPrePlan,
		"totalPostPlan", totalPostPlan,
		"totalDuration", time.Since(start),
	)

	return plan, &baselineScore, bestName, bestAttempt, attempts, nil
}

// -------------------------
// Test Hooks
// -------------------------

var (
	isNonBlockingSolvingFn = isNonBlockingSolving

	planContextFn = func(pl *SharedState, preemptor *v1.Pod) ([]*v1.Node, []*v1.Pod, SolverInput, error) {
		return pl.planContext(preemptor)
	}
	planComputationFn = func(pl *SharedState, ctx context.Context, in SolverInput) (string, bool, *SolverResult, *SolverOutput, []SolverResult) {
		return pl.planComputation(ctx, in)
	}
	isSolutionApplicableFn = func(pl *SharedState, out *SolverOutput, nodes []*v1.Node, pods []*v1.Pod) (bool, string) {
		return pl.isSolutionApplicable(out, nodes, pods)
	}
	computePlanPodCountsFn = func(out *SolverOutput, pods []*v1.Pod) (int, int, int) {
		return computePlanPodCounts(out, pods)
	}
	planRegistrationFn = func(pl *SharedState, ctx context.Context, res SolverResult, out *SolverOutput, preemptor *v1.Pod, pods []*v1.Pod) (*Plan, *ActivePlan, error) {
		return pl.planRegistration(ctx, res, out, preemptor, pods)
	}
	planActivationFn = func(pl *SharedState, plan *Plan, pods []*v1.Pod) error {
		return pl.planActivation(plan, pods)
	}
	startPlanCompletionWatchFn = func(pl *SharedState, ap *ActivePlan) {
		pl.startPlanCompletionWatch(ap)
	}
	exportSolverStatsFn = func(pl *SharedState, strategy string, baseline SolverScore, bestName string, attempts []SolverResult, errMsg string) {
		pl.exportSolverStatsToConfigMap(context.Background(), strategy, baseline, bestName, attempts, errMsg)
	}
)
