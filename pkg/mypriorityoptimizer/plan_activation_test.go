// plan_activation_test.go
package mypriorityoptimizer

import (
	"context"
	"errors"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// -------------------------
// planActivation
// -------------------------

func TestPlanActivation_NilPlan(t *testing.T) {
	pl := &SharedState{}
	err := pl.planActivation(nil, nil)
	must(t, errors.Is(err, ErrNoPlanProvided), "err=%v want %v", err, ErrNoPlanProvided)
}

func TestPlanActivation_NoMovesOrEvicts(t *testing.T) {
	pl := &SharedState{}
	plan := &Plan{}
	pods := []*v1.Pod{pod("ns", "p1")}

	withVar(t, &evictTargetsHook, func(_ *SharedState, _ context.Context, _ []*v1.Pod) error {
		t.Fatalf("evictTargetsHook must not be called when no moves/evicts")
		return nil
	})
	withVar(t, &waitPodsGoneHook, func(_ *SharedState, _ context.Context, _ []*v1.Pod) error {
		t.Fatalf("waitPodsGoneHook must not be called when no moves/evicts")
		return nil
	})
	withVar(t, &getPodForPlanActivation, func(_ *SharedState, _ types.UID, _, _ string) *v1.Pod {
		t.Fatalf("getPodForPlanActivation must not be called when no moves/evicts")
		return nil
	})

	called := false
	withVar(t, &activatePlannedPodsFn, func(hpl *SharedState, p *Plan, ps []*v1.Pod) {
		called = true
		must(t, hpl == pl, "pl mismatch")
		must(t, p == plan, "plan mismatch")
		mustEq(t, ps, pods, "pods mismatch")
	})

	mustNoErr(t, pl.planActivation(plan, pods), "planActivation err")
	must(t, called, "activatePlannedPodsFn not called")
}

func TestPlanActivation_MovesAndEvicts_EvictsWaitsActivates(t *testing.T) {
	pl := &SharedState{}

	// Cancel the active-plan ctx; planActivation must use WithoutCancel(ap.Ctx)
	// so the eviction ctx is NOT immediately canceled.
	apCtx, cancel := context.WithCancel(context.Background())
	cancel()
	pl.ActivePlan.Store(&ActivePlan{Ctx: apCtx})
	t.Cleanup(func() { pl.ActivePlan.Store(nil) })

	p1 := pod("ns", "p1", withUID("u1"))
	p2 := pod("ns", "p2", withUID("u2"))
	pods := []*v1.Pod{p1, p2}

	plan := &Plan{
		Moves: []SolverPod{
			{UID: p1.UID, Namespace: p1.Namespace, Name: p1.Name, OldNode: "n1", Node: "n2"},
			{UID: p1.UID, Namespace: p1.Namespace, Name: p1.Name, OldNode: "n1", Node: "n2"},
			{UID: types.UID("u-missing"), Namespace: "ns", Name: "missing", OldNode: "n1", Node: "n2"},
		},
		Evicts: []SolverPod{
			{UID: p2.UID, Namespace: p2.Namespace, Name: p2.Name, Node: "n1"},
		},
	}

	lookups := map[types.UID]int{}
	withVar(t, &getPodForPlanActivation, func(_ *SharedState, uid types.UID, _, _ string) *v1.Pod {
		lookups[uid]++
		switch uid {
		case p1.UID:
			return p1
		case p2.UID:
			return p2
		default:
			return nil
		}
	})

	var evicted, waited []*v1.Pod
	withVar(t, &evictTargetsHook, func(_ *SharedState, ctx context.Context, targets []*v1.Pod) error {
		must(t, ctx.Err() == nil, "ctx must not be canceled (WithoutCancel path), got %v", ctx.Err())
		_, ok := ctx.Deadline()
		must(t, ok, "evict ctx must have deadline")
		evicted = append([]*v1.Pod(nil), targets...)
		return nil
	})
	withVar(t, &waitPodsGoneHook, func(_ *SharedState, ctx context.Context, targets []*v1.Pod) error {
		must(t, ctx.Err() == nil, "wait ctx must not be canceled, got %v", ctx.Err())
		_, ok := ctx.Deadline()
		must(t, ok, "wait ctx must have deadline")
		waited = append([]*v1.Pod(nil), targets...)
		return nil
	})

	activated := false
	withVar(t, &activatePlannedPodsFn, func(_ *SharedState, p *Plan, ps []*v1.Pod) {
		activated = true
		must(t, p == plan, "plan mismatch")
		mustEq(t, ps, pods, "pods mismatch")
	})

	mustNoErr(t, pl.planActivation(plan, pods), "planActivation err")

	// Dedup by UID => exactly p1+p2 in both calls.
	mustPodSet(t, evicted, "ns/p1", "ns/p2")
	mustPodSet(t, waited, "ns/p1", "ns/p2")
	must(t, activated, "activatePlannedPodsFn not called")

	// Ensure seen[] dedup prevented extra lookup for the duplicate move.
	mustEq(t, lookups[p1.UID], 1, "p1 lookup count wrong (dup should not re-resolve)")
	mustEq(t, lookups[p2.UID], 1, "p2 lookup count wrong")
}

func TestPlanActivation_MovesAndEvicts_NoEvictOrWait(t *testing.T) {
	pl := &SharedState{}
	plan := &Plan{
		Moves:  []SolverPod{{UID: types.UID("u-missing"), Namespace: "ns", Name: "missing", OldNode: "a", Node: "b"}},
		Evicts: []SolverPod{{UID: types.UID("u-missing2"), Namespace: "ns", Name: "missing2", Node: "n1"}},
	}
	pods := []*v1.Pod{pod("ns", "p")}

	withVar(t, &getPodForPlanActivation, func(_ *SharedState, _ types.UID, _, _ string) *v1.Pod {
		return nil // resolves nothing => len(targets)==0 branch
	})
	withVar(t, &evictTargetsHook, func(_ *SharedState, _ context.Context, _ []*v1.Pod) error {
		t.Fatalf("evictTargetsHook must not be called when no targets resolve")
		return nil
	})
	withVar(t, &waitPodsGoneHook, func(_ *SharedState, _ context.Context, _ []*v1.Pod) error {
		t.Fatalf("waitPodsGoneHook must not be called when no targets resolve")
		return nil
	})

	activated := false
	withVar(t, &activatePlannedPodsFn, func(_ *SharedState, p *Plan, ps []*v1.Pod) {
		activated = true
		must(t, p == plan, "plan mismatch")
		mustEq(t, ps, pods, "pods mismatch")
	})

	mustNoErr(t, pl.planActivation(plan, pods), "planActivation err")
	must(t, activated, "activatePlannedPodsFn not called")
}

func TestPlanActivation_EvictOrWaitErrors(t *testing.T) {
	pl := &SharedState{}
	p1 := pod("ns", "p1", withUID("u1"))
	pods := []*v1.Pod{p1}
	plan := &Plan{
		Moves: []SolverPod{{UID: p1.UID, Namespace: p1.Namespace, Name: p1.Name, OldNode: "n1", Node: "n2"}},
	}

	withVar(t, &getPodForPlanActivation, func(_ *SharedState, uid types.UID, _, _ string) *v1.Pod {
		if uid == p1.UID {
			return p1
		}
		return nil
	})

	t.Run("evict error stops before wait/activate", func(t *testing.T) {
		evictErr := errors.New("evict failed")
		waitCalled := false
		activateCalled := false

		withVar(t, &evictTargetsHook, func(_ *SharedState, _ context.Context, _ []*v1.Pod) error {
			return evictErr
		})
		withVar(t, &waitPodsGoneHook, func(_ *SharedState, _ context.Context, _ []*v1.Pod) error {
			waitCalled = true
			return nil
		})
		withVar(t, &activatePlannedPodsFn, func(_ *SharedState, _ *Plan, _ []*v1.Pod) {
			activateCalled = true
		})

		err := pl.planActivation(plan, pods)
		must(t, errors.Is(err, evictErr), "err=%v want %v", err, evictErr)
		must(t, !waitCalled, "waitPodsGoneHook must not be called after evict error")
		must(t, !activateCalled, "activatePlannedPodsFn must not be called after evict error")
	})

	t.Run("wait error wraps and stops before activate", func(t *testing.T) {
		waitErr := errors.New("pods not gone")
		activateCalled := false

		withVar(t, &evictTargetsHook, func(_ *SharedState, _ context.Context, _ []*v1.Pod) error {
			return nil
		})
		withVar(t, &waitPodsGoneHook, func(_ *SharedState, _ context.Context, _ []*v1.Pod) error {
			return waitErr
		})
		withVar(t, &activatePlannedPodsFn, func(_ *SharedState, _ *Plan, _ []*v1.Pod) {
			activateCalled = true
		})

		err := pl.planActivation(plan, pods)
		must(t, err != nil, "expected non-nil err")
		must(t, errors.Is(err, waitErr), "err=%v want wrap %v", err, waitErr)
		mustContains(t, err.Error(), "wait for targeted pods gone")
		must(t, !activateCalled, "activatePlannedPodsFn must not be called after wait error")
	})
}
