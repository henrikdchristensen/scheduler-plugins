// plan_computation_test.go
package mypriorityoptimizer

import (
	"context"
	"errors"
	"testing"
	"time"
)

// -------------------------
// planComputation
// -------------------------

func TestPlanComputation_NoEnabledSolvers(t *testing.T) {
	pl := &SharedState{}

	withPythonAttemptConfig(t, false, 10*time.Millisecond, 0)
	withVar(t, &runPythonSolverHook, (func(*SharedState, context.Context, SolverInput, PythonSolverOptions) (*SolverOutput, error))(nil))
	withVar(t, &runPythonSolverFn, func(_ *SharedState, _ context.Context, _ SolverInput, _ PythonSolverOptions) (*SolverOutput, error) {
		t.Fatalf("runPythonSolverFn should not be called when solver disabled")
		return nil, nil
	})

	bestName, hadUsable, bestAttempt, bestOutput, attempts := pl.planComputation(context.Background(), SolverInput{})

	must(t, !hadUsable, "hadUsable want false")
	mustEq(t, bestName, "", "bestName")
	mustEq(t, bestAttempt, (*SolverResult)(nil), "bestAttempt")
	mustEq(t, bestOutput, (*SolverOutput)(nil), "bestOutput")
	mustEq(t, len(attempts), 0, "attempts len")
}

func TestPlanComputation_NilOutputNoError(t *testing.T) {
	pl := &SharedState{}

	withPythonAttemptConfig(t, true, 10*time.Millisecond, 7)

	var gotTimeoutMs int64
	withVar(t, &runPythonSolverHook, func(_ *SharedState, _ context.Context, in SolverInput, _ PythonSolverOptions) (*SolverOutput, error) {
		gotTimeoutMs = in.TimeoutMs
		return nil, nil // nil output, nil error
	})

	bestName, hadUsable, bestAttempt, bestOutput, attempts := pl.planComputation(context.Background(), SolverInput{})

	must(t, !hadUsable, "hadUsable want false")
	mustEq(t, bestName, "", "bestName")
	mustEq(t, bestAttempt, (*SolverResult)(nil), "bestAttempt")
	mustEq(t, bestOutput, (*SolverOutput)(nil), "bestOutput")

	mustEq(t, len(attempts), 1, "attempts len")
	mustEq(t, attempts[0].Name, "python", "attempt name")
	mustEq(t, attempts[0].Status, "FAILED", "attempt status")
	mustEq(t, attempts[0].Score, (SolverScore{}), "attempt score should be zero")

	// TimeoutMs should include grace (10ms + 7ms = 17ms).
	mustEq(t, gotTimeoutMs, int64(17), "TimeoutMs should include grace")
}

func TestPlanComputation_ErrorWithNonNilOutput(t *testing.T) {
	pl := &SharedState{}

	withPythonAttemptConfig(t, true, 10*time.Millisecond, 0)

	pre := &SolverPod{
		UID:       "u-pre",
		Namespace: "ns",
		Name:      "pre",
		Priority:  5,
		Node:      "", // pending
	}

	withVar(t, &runPythonSolverHook, func(_ *SharedState, _ context.Context, _ SolverInput, _ PythonSolverOptions) (*SolverOutput, error) {
		return &SolverOutput{
			Status: "OPTIMAL",
			Placements: []SolverPod{
				{UID: pre.UID, Namespace: pre.Namespace, Name: pre.Name, Priority: pre.Priority, Node: "n1"},
			},
		}, errors.New("boom")
	})

	in := SolverInput{
		Preemptor:     pre,
		Pods:          []SolverPod{*pre},
		BaselineScore: SolverScore{},
	}

	bestName, hadUsable, bestAttempt, bestOutput, attempts := pl.planComputation(context.Background(), in)

	// Error path never selects a best result.
	must(t, !hadUsable, "hadUsable want false on error path")
	mustEq(t, bestName, "", "bestName")
	mustEq(t, bestAttempt, (*SolverResult)(nil), "bestAttempt")
	mustEq(t, bestOutput, (*SolverOutput)(nil), "bestOutput")

	mustEq(t, len(attempts), 1, "attempts len")
	mustEq(t, attempts[0].Status, "OPTIMAL", "status should come from out.Status even on error")
	mustEq(t, attempts[0].Score.PlacedByPriority["5"], 1, "score should be computed when out != nil")
}

func TestPlanComputation_NotUsableStatus(t *testing.T) {
	pl := &SharedState{}

	withPythonAttemptConfig(t, true, 10*time.Millisecond, 0)

	withVar(t, &runPythonSolverHook, func(_ *SharedState, _ context.Context, _ SolverInput, _ PythonSolverOptions) (*SolverOutput, error) {
		return &SolverOutput{Status: "INFEASIBLE"}, nil
	})

	bestName, hadUsable, bestAttempt, bestOutput, attempts := pl.planComputation(context.Background(), SolverInput{})

	must(t, !hadUsable, "hadUsable want false when not usable")
	mustEq(t, bestName, "", "bestName")
	mustEq(t, bestAttempt, (*SolverResult)(nil), "bestAttempt")
	mustEq(t, bestOutput, (*SolverOutput)(nil), "bestOutput")

	mustEq(t, len(attempts), 1, "attempts len")
	mustEq(t, attempts[0].Status, "INFEASIBLE", "attempt status")
}

func TestPlanComputation_UsableButNotImproving(t *testing.T) {
	pl := &SharedState{}

	withPythonAttemptConfig(t, true, 10*time.Millisecond, 0)

	withVar(t, &runPythonSolverHook, func(_ *SharedState, _ context.Context, _ SolverInput, _ PythonSolverOptions) (*SolverOutput, error) {
		return &SolverOutput{Status: "OPTIMAL"}, nil // no placements/evicts => zero score
	})

	bestName, hadUsable, bestAttempt, bestOutput, attempts := pl.planComputation(context.Background(), SolverInput{
		BaselineScore: SolverScore{}, // equal to produced score
	})

	must(t, !hadUsable, "hadUsable want false when not improving")
	mustEq(t, bestName, "", "bestName")
	mustEq(t, bestAttempt, (*SolverResult)(nil), "bestAttempt")
	mustEq(t, bestOutput, (*SolverOutput)(nil), "bestOutput")
	mustEq(t, len(attempts), 1, "attempts len")
	mustEq(t, attempts[0].Status, "OPTIMAL", "attempt status")
}

func TestPlanComputation_UsesRunPythonSolver(t *testing.T) {
	pl := &SharedState{}

	withPythonAttemptConfig(t, true, 10*time.Millisecond, 0)
	withVar(t, &runPythonSolverHook, (func(*SharedState, context.Context, SolverInput, PythonSolverOptions) (*SolverOutput, error))(nil))

	pre := &SolverPod{
		UID:       "u-pre",
		Namespace: "ns",
		Name:      "pre",
		Priority:  5,
		Node:      "",
	}

	withVar(t, &runPythonSolverFn, func(_ *SharedState, _ context.Context, _ SolverInput, _ PythonSolverOptions) (*SolverOutput, error) {
		return &SolverOutput{
			Status: "OPTIMAL",
			Placements: []SolverPod{
				// Include Priority to make scoring robust.
				{UID: pre.UID, Namespace: pre.Namespace, Name: pre.Name, Priority: pre.Priority, Node: "n1"},
			},
		}, nil
	})

	in := SolverInput{
		Preemptor:     pre,
		Pods:          []SolverPod{*pre},
		BaselineScore: SolverScore{},
	}

	bestName, hadUsable, bestAttempt, bestOutput, attempts := pl.planComputation(context.Background(), in)

	must(t, hadUsable, "hadUsable want true")
	mustEq(t, bestName, "python", "bestName")
	must(t, bestAttempt != nil, "bestAttempt want non-nil")
	must(t, bestOutput != nil, "bestOutput want non-nil")
	mustEq(t, len(attempts), 1, "attempts len")
	mustEq(t, attempts[0].Score.PlacedByPriority["5"], 1, "placedByPriority")
}

// -------------------------
// Test Helpers
// -------------------------

func withPythonAttemptConfig(t *testing.T, enabled bool, timeout time.Duration, graceMs int) {
	t.Helper()
	withVar(t, &SolverPythonEnabled, enabled)
	withVar(t, &SolverPythonTimeout, timeout)
	withVar(t, &SolverPythonGraceMs, graceMs)
}
