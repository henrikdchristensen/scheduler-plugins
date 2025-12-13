// pkg/mypriorityoptimizer/optimization_flow_test.go
// optimization_flow_test.go
package mypriorityoptimizer

import (
	"context"
	"errors"
	"testing"

	v1 "k8s.io/api/core/v1"
)

// -------------------------
// Helpers
// -------------------------

// dummySolverInput returns a minimal SolverInput with a specific baseline.
func dummySolverInput(evicted int) SolverInput {
	return SolverInput{
		BaselineScore: SolverScore{Evicted: evicted},
	}
}

type flowCaptures struct {
	exportCalled bool
	export       struct {
		strategy string
		baseline SolverScore
		bestName string
		attempts []SolverResult
		errMsg   string
	}
	watchCalled bool
	watchAP     *ActivePlan

	completedCalled bool
	completedStatus PlanStatus
	completedAP     *ActivePlan
}

type flowHarness struct {
	t  *testing.T
	pl *SharedState

	async bool

	// planContextFn
	nodes         []*v1.Node
	pods          []*v1.Pod
	baselineEvict int
	planCtxErr    error

	// planComputationFn
	bestName       string
	hadImprovement bool
	bestAttempt    *SolverResult
	bestOut        *SolverOutput
	attempts       []SolverResult

	// isSolutionApplicableFn
	applicable    bool
	applicableWhy string

	// computePlanPodCountsFn
	pendingScheduled int
	totalPrePlan     int
	totalPostPlan    int

	// planRegistrationFn / planActivationFn
	regErr error
	actErr error
	plan   *Plan
	ap     *ActivePlan
}

func (h *flowHarness) install(t *testing.T) *flowCaptures {
	t.Helper()

	// Save originals and restore on cleanup.
	origAsync := isAsyncSolvingFn
	origPlanCtx := planContextFn
	origPlanComp := planComputationFn
	origApplicable := isSolutionApplicableFn
	origCounts := computePlanPodCountsFn
	origReg := planRegistrationFn
	origAct := planActivationFn
	origWatch := startPlanCompletionWatchFn
	origExport := exportSolverStatsFn
	origOnCompleted := onPlanCompletedHook

	t.Cleanup(func() {
		isAsyncSolvingFn = origAsync
		planContextFn = origPlanCtx
		planComputationFn = origPlanComp
		isSolutionApplicableFn = origApplicable
		computePlanPodCountsFn = origCounts
		planRegistrationFn = origReg
		planActivationFn = origAct
		startPlanCompletionWatchFn = origWatch
		exportSolverStatsFn = origExport
		onPlanCompletedHook = origOnCompleted
	})

	caps := &flowCaptures{}

	// Hooks
	isAsyncSolvingFn = func() bool { return h.async }

	planContextFn = func(_ *SharedState, _ *v1.Pod) ([]*v1.Node, []*v1.Pod, SolverInput, error) {
		if h.planCtxErr != nil {
			return nil, nil, SolverInput{}, h.planCtxErr
		}
		return h.nodes, h.pods, dummySolverInput(h.baselineEvict), nil
	}

	planComputationFn = func(_ *SharedState, _ context.Context, _ SolverInput) (string, bool, *SolverResult, *SolverOutput, []SolverResult) {
		return h.bestName, h.hadImprovement, h.bestAttempt, h.bestOut, h.attempts
	}

	isSolutionApplicableFn = func(_ *SharedState, _ *SolverOutput, _ []*v1.Node, _ []*v1.Pod) (bool, string) {
		return h.applicable, h.applicableWhy
	}

	computePlanPodCountsFn = func(_ *SolverOutput, _ []*v1.Pod) (int, int, int) {
		return h.pendingScheduled, h.totalPrePlan, h.totalPostPlan
	}

	planRegistrationFn = func(pl *SharedState, _ context.Context, _ SolverResult, _ *SolverOutput, _ *v1.Pod, _ []*v1.Pod) (*Plan, *ActivePlan, error) {
		// If registration fails, some implementations still want to mark an already-created
		// ActivePlan as failed. Your onPlanCompletedHook may only fire when ActivePlan != nil,
		// so allow tests to seed one via h.ap.
		if h.regErr != nil {
			if h.ap != nil {
				pl.ActivePlan.Store(h.ap)
			}
			return nil, nil, h.regErr
		}

		if h.plan == nil {
			h.plan = &Plan{}
		}
		if h.ap == nil {
			h.ap = &ActivePlan{ID: "ap-1"}
		}

		// Mimic production: registration makes ActivePlan visible.
		pl.ActivePlan.Store(h.ap)
		return h.plan, h.ap, nil
	}

	planActivationFn = func(_ *SharedState, _ *Plan, _ []*v1.Pod) error {
		return h.actErr
	}

	startPlanCompletionWatchFn = func(_ *SharedState, ap *ActivePlan) {
		caps.watchCalled = true
		caps.watchAP = ap
	}

	exportSolverStatsFn = func(_ *SharedState, strategy string, baseline SolverScore, bestName string, attempts []SolverResult, errMsg string) {
		caps.exportCalled = true
		caps.export.strategy = strategy
		caps.export.baseline = baseline
		caps.export.bestName = bestName
		caps.export.attempts = attempts
		caps.export.errMsg = errMsg
	}

	onPlanCompletedHook = func(_ *SharedState, status PlanStatus, ap *ActivePlan) {
		caps.completedCalled = true
		caps.completedStatus = status
		caps.completedAP = ap
	}

	return caps
}

func assertZeroReturns(t *testing.T, plan *Plan, baseline *SolverScore, bestName string, bestAttempt *SolverResult, attempts []SolverResult) {
	t.Helper()
	if plan != nil || baseline != nil || bestName != "" || bestAttempt != nil || attempts != nil {
		t.Fatalf("expected all return values to be zero (plan=%v baseline=%v bestName=%q bestAttempt=%v attempts=%v)",
			plan, baseline, bestName, bestAttempt, attempts)
	}
}

func assertPlanActiveReleased(t *testing.T, pl *SharedState) {
	t.Helper()
	if !pl.tryEnterActivePlan() {
		t.Fatalf("expected ActivePlan gate to be released")
	}
}

// -------------------------
// 1) Optimization gate: ErrOptimizationInProgress
// -------------------------

func TestRunOptimizationFlow_OptimizationInProgress(t *testing.T) {
	pl := &SharedState{}
	h := &flowHarness{
		t:  t,
		pl: pl,

		async: false,

		// Should never be used:
		planCtxErr: errors.New("must-not-be-called"),
	}
	caps := h.install(t)

	// First caller "owns" optimization.
	if !pl.tryEnterOptimizationFlow() {
		t.Fatalf("precondition: tryEnterOptimizationFlow() should succeed on fresh SharedState")
	}

	plan, baseline, bestName, bestAttempt, attempts, err :=
		pl.runOptimizationFlow(context.Background(), nil)

	if !errors.Is(err, ErrOptimizationInProgress) {
		t.Fatalf("err=%v, want ErrOptimizationInProgress", err)
	}
	assertZeroReturns(t, plan, baseline, bestName, bestAttempt, attempts)

	if caps.exportCalled {
		t.Fatalf("exportSolverStatsFn must not be called on optimization-in-progress early return")
	}
}

// -------------------------
// Main non-async scenarios (table-driven)
// -------------------------

func TestRunOptimizationFlow_NonAsync_Scenarios(t *testing.T) {
	type want struct {
		err error

		// Returned values
		planNonNil     bool
		baselineEvict  *int // nil means baseline ptr expected to be nil
		bestName       string
		bestAttemptPtr *SolverResult
		attemptsLen    int

		// Side-effects
		exportCalled        bool
		exportErrMsg        string
		watchCalled         bool
		completedCalled     bool
		completedStatus     PlanStatus
		checkActiveReleased bool
	}

	mkPendingPods := func() []*v1.Pod {
		return []*v1.Pod{pod("default", "p-pending")}
	}

	tests := []struct {
		name string
		h    flowHarness
		want want
	}{
		{
			name: "planContext_error",
			h: flowHarness{
				async:      false,
				planCtxErr: errors.New("boom-plancontext"),
			},
			want: want{
				err:                 errors.New("boom-plancontext"), // compared via errors.Is below with a separate handle
				exportCalled:        false,
				watchCalled:         false,
				completedCalled:     false,
				checkActiveReleased: true,
			},
		},
		{
			name: "no_pending_pods",
			h: flowHarness{
				async:         false,
				nodes:         []*v1.Node{},
				pods:          []*v1.Pod{}, // <-- this makes pendingPrePlan == 0
				baselineEvict: 99,
			},
			want: want{
				err:             ErrNoPendingPods,
				planNonNil:      false,
				baselineEvict:   ptrInt(99), // function returns &baselineScore on this path
				bestName:        "",
				attemptsLen:     0,
				exportCalled:    false,
				watchCalled:     false,
				completedCalled: false,

				// See note below
				checkActiveReleased: true,
			},
		},
		{
			name: "no_improving_solution",
			h: flowHarness{
				async:          false,
				nodes:          nil,
				pods:           mkPendingPods(),
				baselineEvict:  42,
				bestName:       "solverB",
				hadImprovement: false,
				bestAttempt:    &SolverResult{Name: "attempt-1"},
				bestOut:        nil,
				attempts: []SolverResult{
					{Name: "solverA", Status: "FEASIBLE"},
					{Name: "solverB", Status: "OPTIMAL"},
				},
			},
			want: want{
				err:                 ErrNoImprovingSolutionFromAnySolver,
				planNonNil:          false,
				baselineEvict:       ptrInt(42),
				bestName:            "solverB",
				bestAttemptPtr:      &SolverResult{Name: "attempt-1"}, // pointer equality checked separately below
				attemptsLen:         2,
				exportCalled:        true,
				exportErrMsg:        ErrNoImprovingSolutionFromAnySolver.Error(),
				watchCalled:         false,
				completedCalled:     false,
				checkActiveReleased: true,
			},
		},
		{
			name: "plan_not_applicable",
			h: flowHarness{
				async:          false,
				nodes:          []*v1.Node{},
				pods:           mkPendingPods(),
				baselineEvict:  7,
				bestName:       "solverX",
				hadImprovement: true,
				bestAttempt:    &SolverResult{Name: "attempt-2"},
				bestOut:        &SolverOutput{Status: "OPTIMAL"},
				attempts:       []SolverResult{{Name: "solverX"}},
				applicable:     false,
				applicableWhy:  "stale-cluster",
			},
			want: want{
				err:                 ErrPlanNotApplicable,
				planNonNil:          false,
				baselineEvict:       ptrInt(7),
				bestName:            "solverX",
				attemptsLen:         1,
				exportCalled:        true,
				exportErrMsg:        ErrPlanNotApplicable.Error(),
				watchCalled:         false,
				completedCalled:     false,
				checkActiveReleased: true,
			},
		},
		{
			name: "no_pending_scheduled",
			h: flowHarness{
				async:            false,
				nodes:            []*v1.Node{},
				pods:             mkPendingPods(),
				baselineEvict:    10,
				bestName:         "solverY",
				hadImprovement:   true,
				bestAttempt:      &SolverResult{Name: "attempt-3"},
				bestOut:          &SolverOutput{Status: "OPTIMAL"},
				attempts:         []SolverResult{{Name: "solverY"}},
				applicable:       true,
				pendingScheduled: 0,
				totalPrePlan:     5,
				totalPostPlan:    5,
			},
			want: want{
				err:                 ErrNoPendingPodsScheduled,
				planNonNil:          false,
				baselineEvict:       ptrInt(10),
				bestName:            "solverY",
				attemptsLen:         1,
				exportCalled:        true,
				exportErrMsg:        ErrNoPendingPodsScheduled.Error(),
				watchCalled:         false,
				completedCalled:     false,
				checkActiveReleased: true,
			},
		},
		{
			name: "plan_registration_error_calls_onPlanCompleted",
			h: flowHarness{
				async:            false,
				nodes:            []*v1.Node{},
				pods:             mkPendingPods(),
				baselineEvict:    5,
				bestName:         "solverReg",
				hadImprovement:   true,
				bestAttempt:      &SolverResult{Name: "attempt-regerr"},
				bestOut:          &SolverOutput{Status: "OPTIMAL"},
				attempts:         []SolverResult{{Name: "solverReg"}},
				applicable:       true,
				pendingScheduled: 2,
				totalPrePlan:     4,
				totalPostPlan:    6,
				regErr:           errors.New("boom-planreg"),

				// Seed an active plan so onPlanCompletedHook is expected to fire
				ap: &ActivePlan{ID: "ap-regerr"},
			},
			want: want{
				err:                 ErrPlanRegistration,
				planNonNil:          false,
				baselineEvict:       ptrInt(5),
				bestName:            "solverReg",
				attemptsLen:         1,
				exportCalled:        true,
				exportErrMsg:        ErrPlanRegistration.Error(),
				watchCalled:         false,
				completedCalled:     true,
				completedStatus:     PlanStatusFailed,
				checkActiveReleased: true,
			},
		},
		{
			name: "plan_activation_error_calls_onPlanCompleted",
			h: flowHarness{
				async:            false,
				nodes:            []*v1.Node{},
				pods:             mkPendingPods(),
				baselineEvict:    2,
				bestName:         "solverAct",
				hadImprovement:   true,
				bestAttempt:      &SolverResult{Name: "attempt-acterr"},
				bestOut:          &SolverOutput{Status: "OPTIMAL"},
				attempts:         []SolverResult{{Name: "solverAct"}},
				applicable:       true,
				pendingScheduled: 1,
				totalPrePlan:     3,
				totalPostPlan:    4,
				actErr:           errors.New("boom-activation"),
				ap:               &ActivePlan{ID: "plan-activation"},
			},
			want: want{
				err:                 ErrPlanActivationFailed,
				planNonNil:          false,
				baselineEvict:       ptrInt(2),
				bestName:            "solverAct",
				attemptsLen:         1,
				exportCalled:        true,
				exportErrMsg:        ErrPlanActivationFailed.Error(),
				watchCalled:         false,
				completedCalled:     true,
				completedStatus:     PlanStatusFailed,
				checkActiveReleased: true, // onPlanCompleted should release
			},
		},
		{
			name: "success",
			h: flowHarness{
				async:            false,
				nodes:            []*v1.Node{},
				pods:             mkPendingPods(),
				baselineEvict:    0,
				bestName:         "solverZ",
				hadImprovement:   true,
				bestAttempt:      &SolverResult{Name: "attempt-best"},
				bestOut:          &SolverOutput{Status: "OPTIMAL"},
				attempts:         []SolverResult{{Name: "solverZ"}},
				applicable:       true,
				pendingScheduled: 2,
				totalPrePlan:     5,
				totalPostPlan:    7,
				plan:             &Plan{},
				ap:               &ActivePlan{ID: "plan-123"},
			},
			want: want{
				err:                 nil,
				planNonNil:          true,
				baselineEvict:       ptrInt(0),
				bestName:            "solverZ",
				attemptsLen:         1,
				exportCalled:        true,
				exportErrMsg:        "",
				watchCalled:         true,
				completedCalled:     false,
				checkActiveReleased: false, // success keeps plan active
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			pl := &SharedState{}
			tc.h.t = t
			tc.h.pl = pl

			caps := tc.h.install(t)

			plan, baselinePtr, bestName, bestAttemptOut, attemptsOut, err :=
				pl.runOptimizationFlow(context.Background(), nil)

			// Error checking
			if tc.want.err == nil {
				if err != nil {
					t.Fatalf("err=%v, want nil", err)
				}
			} else {
				// Special-case the planContext error test: we need the exact error instance.
				if tc.name == "planContext_error" {
					if tc.h.planCtxErr == nil {
						t.Fatalf("test bug: missing planCtxErr")
					}
					if !errors.Is(err, tc.h.planCtxErr) {
						t.Fatalf("err=%v, want %v", err, tc.h.planCtxErr)
					}
				} else if !errors.Is(err, tc.want.err) {
					t.Fatalf("err=%v, want %v", err, tc.want.err)
				}
			}

			// Return values
			if tc.want.planNonNil {
				if plan == nil {
					t.Fatalf("plan=nil, want non-nil")
				}
			} else if plan != nil {
				t.Fatalf("plan=%v, want nil", plan)
			}

			if tc.want.baselineEvict == nil {
				if baselinePtr != nil {
					t.Fatalf("baseline=%v, want nil", baselinePtr)
				}
			} else {
				if baselinePtr == nil || baselinePtr.Evicted != *tc.want.baselineEvict {
					t.Fatalf("baseline=%v, want Evicted=%d", baselinePtr, *tc.want.baselineEvict)
				}
			}

			if bestName != tc.want.bestName {
				t.Fatalf("bestName=%q, want %q", bestName, tc.want.bestName)
			}

			if tc.h.bestAttempt != nil && tc.want.bestAttemptPtr != nil {
				// Pointer identity: must match the exact object returned by planComputationFn.
				if bestAttemptOut != tc.h.bestAttempt {
					t.Fatalf("bestAttempt pointer mismatch: got=%p want=%p", bestAttemptOut, tc.h.bestAttempt)
				}
			} else {
				// Otherwise just ensure nilness is sensible.
				if tc.h.bestAttempt == nil && bestAttemptOut != nil {
					t.Fatalf("bestAttempt=%v, want nil", bestAttemptOut)
				}
			}

			if tc.want.attemptsLen == 0 {
				if attemptsOut != nil && len(attemptsOut) != 0 {
					t.Fatalf("attempts len=%d, want 0 (or nil)", len(attemptsOut))
				}
			} else {
				if len(attemptsOut) != tc.want.attemptsLen {
					t.Fatalf("attempts len=%d, want %d", len(attemptsOut), tc.want.attemptsLen)
				}
			}

			// Side-effects
			if caps.exportCalled != tc.want.exportCalled {
				t.Fatalf("exportCalled=%v, want %v", caps.exportCalled, tc.want.exportCalled)
			}
			if tc.want.exportCalled && caps.export.errMsg != tc.want.exportErrMsg {
				t.Fatalf("export errMsg=%q, want %q", caps.export.errMsg, tc.want.exportErrMsg)
			}

			if caps.watchCalled != tc.want.watchCalled {
				t.Fatalf("watchCalled=%v, want %v", caps.watchCalled, tc.want.watchCalled)
			}
			if tc.want.watchCalled && tc.h.ap != nil && caps.watchAP != tc.h.ap {
				t.Fatalf("watch ap=%#v, want %#v", caps.watchAP, tc.h.ap)
			}

			if caps.completedCalled != tc.want.completedCalled {
				t.Fatalf("completedCalled=%v, want %v", caps.completedCalled, tc.want.completedCalled)
			}
			if tc.want.completedCalled && caps.completedStatus != tc.want.completedStatus {
				t.Fatalf("completedStatus=%v, want %v", caps.completedStatus, tc.want.completedStatus)
			}

			if tc.want.checkActiveReleased {
				assertPlanActiveReleased(t, pl)
			}
		})
	}
}

// -------------------------
// 7) Non-async: Active plan already in progress -> ErrActiveInProgress
// -------------------------

func TestRunOptimizationFlow_ActivePlanAlreadyInProgress_NonAsync(t *testing.T) {
	pl := &SharedState{}
	pl.ActivePlanInProgress.Store(true)

	h := &flowHarness{
		t:     t,
		pl:    pl,
		async: false,

		// If planContext is called, the test should fail (must return early).
		planCtxErr: errors.New("must-not-be-called"),
	}
	caps := h.install(t)

	// Hard fail if any of the heavy hooks are reached.
	planContextFn = func(_ *SharedState, _ *v1.Pod) ([]*v1.Node, []*v1.Pod, SolverInput, error) {
		t.Fatalf("planContextFn must not be called when ActivePlan is already in progress")
		return nil, nil, SolverInput{}, nil
	}
	exportSolverStatsFn = func(_ *SharedState, _ string, _ SolverScore, _ string, _ []SolverResult, _ string) {
		t.Fatalf("exportSolverStatsFn must not be called on early ActivePlan conflict")
	}

	plan, baseline, bestName, bestAttempt, attempts, err :=
		pl.runOptimizationFlow(context.Background(), nil)

	if !errors.Is(err, ErrActiveInProgress) {
		t.Fatalf("err=%v, want ErrActiveInProgress", err)
	}
	assertZeroReturns(t, plan, baseline, bestName, bestAttempt, attempts)

	if caps.exportCalled || caps.watchCalled || caps.completedCalled {
		t.Fatalf("unexpected side-effects on early ActivePlan conflict: export=%v watch=%v completed=%v",
			caps.exportCalled, caps.watchCalled, caps.completedCalled)
	}
}

// -------------------------
// 8) Async: Active plan in progress at apply-time -> ErrActiveInProgress (and stats exported)
// -------------------------

func TestRunOptimizationFlow_Async_ActivePlanInProgressAtApply(t *testing.T) {
	pl := &SharedState{}

	h := &flowHarness{
		t:     t,
		pl:    pl,
		async: true,

		nodes:         []*v1.Node{},
		pods:          []*v1.Pod{pod("default", "p-pending")},
		baselineEvict: 11,

		bestName:       "solverAsync",
		hadImprovement: true,
		bestAttempt:    &SolverResult{Name: "attempt-async"},
		bestOut:        &SolverOutput{Status: "OPTIMAL"},
		attempts:       []SolverResult{{Name: "solverAsync"}},
		applicable:     true,

		pendingScheduled: 1,
		totalPrePlan:     4,
		totalPostPlan:    5,
	}
	caps := h.install(t)

	// Simulate that an Active plan is already in progress at apply-time.
	pl.ActivePlanInProgress.Store(true)

	// These must not be reached if async tryEnterActivePlan fails.
	planRegistrationFn = func(_ *SharedState, _ context.Context, _ SolverResult, _ *SolverOutput, _ *v1.Pod, _ []*v1.Pod) (*Plan, *ActivePlan, error) {
		t.Fatalf("planRegistrationFn must not be called when async tryEnterActivePlan fails")
		return nil, nil, nil
	}
	planActivationFn = func(_ *SharedState, _ *Plan, _ []*v1.Pod) error {
		t.Fatalf("planActivationFn must not be called when async tryEnterActivePlan fails")
		return nil
	}
	startPlanCompletionWatchFn = func(_ *SharedState, _ *ActivePlan) {
		t.Fatalf("startPlanCompletionWatchFn must not be called when async tryEnterActivePlan fails")
	}

	plan, baselinePtr, bestName, bestAttemptOut, attemptsOut, err :=
		pl.runOptimizationFlow(context.Background(), nil)

	if !errors.Is(err, ErrActiveInProgress) {
		t.Fatalf("err=%v, want ErrActiveInProgress", err)
	}
	assertZeroReturns(t, plan, baselinePtr, bestName, bestAttemptOut, attemptsOut)

	if !caps.exportCalled {
		t.Fatalf("exportSolverStatsFn must be called on async ActiveInProgress path")
	}
	if caps.export.baseline.Evicted != 11 {
		t.Fatalf("exported baseline Evicted=%d, want %d", caps.export.baseline.Evicted, 11)
	}
	if caps.export.bestName != "solverAsync" {
		t.Fatalf("exported bestName=%q, want %q", caps.export.bestName, "solverAsync")
	}
	if caps.export.errMsg != ErrActiveInProgress.Error() {
		t.Fatalf("exported errMsg=%q, want %q", caps.export.errMsg, ErrActiveInProgress.Error())
	}
}

// -------------------------
// small helper to avoid &int literals in table
// -------------------------

func ptrInt(v int) *int { return &v }
