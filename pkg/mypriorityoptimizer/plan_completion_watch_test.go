// plan_completion_watch_test.go
// with the help of AI tools to cover more branches/cases
//TODO
package mypriorityoptimizer

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

// -------------------------
// startPlanCompletionWatch
// -------------------------

func TestStartPlanCompletionWatch_NilActivePlan(t *testing.T) {
	pl := &SharedState{}

	called := atomic.Bool{}
	withVar(t, &planCompletionWatchFn, func(_ *SharedState, _ *ActivePlan) {
		called.Store(true)
	})

	// ap == nil -> should return immediately and not call the watcher.
	pl.startPlanCompletionWatch(nil)

	if called.Load() {
		t.Fatalf("planCompletionWatchFn was called, want not called for nil ap")
	}
}

func TestStartPlanCompletionWatch_SpawnsWatcher(t *testing.T) {
	pl := &SharedState{}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ap := &ActivePlan{
		ID:     "plan-1",
		Ctx:    ctx,
		Cancel: func() {},
	}

	done := make(chan struct{})
	var gotPL *SharedState
	var gotAP *ActivePlan

	withVar(t, &planCompletionWatchFn, func(hpl *SharedState, hap *ActivePlan) {
		gotPL = hpl
		gotAP = hap
		close(done)
	})

	pl.startPlanCompletionWatch(ap)

	select {
	case <-done:
		// ok
	case <-time.After(250 * time.Millisecond):
		t.Fatalf("planCompletionWatchFn was not called from startPlanCompletionWatch")
	}

	if gotPL != pl {
		t.Fatalf("planCompletionWatchFn got pl=%p, want %p", gotPL, pl)
	}
	if gotAP != ap {
		t.Fatalf("planCompletionWatchFn got ap=%p, want %p", gotAP, ap)
	}
}

// -------------------------
// planCompletionWatch – active plan cleared
// -------------------------

func TestPlanCompletionWatch_ActivePlanClearedStopsWatcher(t *testing.T) {
	pl := &SharedState{}

	// Context that never finishes (so only ticker drives the loop).
	ctx := context.Background()
	ap := &ActivePlan{
		ID:     "plan-cleared",
		Ctx:    ctx,
		Cancel: func() {},
	}

	// Make ticker fast for tests.
	withVar(t, &getPlanCompletionCheckInterval, func() time.Duration { return 1 * time.Millisecond })

	// First tick: return nil active plan so watcher exits.
	withVar(t, &getActivePlanForWatch, func(_ *SharedState) *ActivePlan { return nil })

	// Should never be called if we exit due to cleared plan.
	withVar(t, &isPlanCompletedFn, func(_ *SharedState, _ *ActivePlan) (bool, error) {
		t.Fatalf("isPlanCompletedFn should not be called when active plan is nil")
		return false, nil
	})

	onCalled := atomic.Bool{}
	withVar(t, &onPlanCompletedFn, func(_ *SharedState, _ PlanStatus) {
		onCalled.Store(true)
	})

	start := time.Now()
	pl.planCompletionWatch(ap)
	elapsed := time.Since(start)

	if onCalled.Load() {
		t.Fatalf("onPlanCompletedFn was called, want not called when active plan is cleared")
	}
	// Sanity check that we didn't block forever.
	if elapsed > 250*time.Millisecond {
		t.Fatalf("planCompletionWatch took too long, watcher may not have stopped (elapsed=%v)", elapsed)
	}
}

func TestPlanCompletionWatch_ActivePlanIDMismatchStopsWatcher(t *testing.T) {
	pl := &SharedState{}

	ap := &ActivePlan{ID: "plan-a", Ctx: context.Background(), Cancel: func() {}}
	other := &ActivePlan{ID: "plan-b", Ctx: context.Background(), Cancel: func() {}}

	withVar(t, &getPlanCompletionCheckInterval, func() time.Duration { return 1 * time.Millisecond })
	withVar(t, &getActivePlanForWatch, func(_ *SharedState) *ActivePlan { return other })

	withVar(t, &isPlanCompletedFn, func(_ *SharedState, _ *ActivePlan) (bool, error) {
		t.Fatalf("isPlanCompletedFn should not be called when plan ID mismatches")
		return false, nil
	})

	called := atomic.Bool{}
	withVar(t, &onPlanCompletedFn, func(_ *SharedState, _ PlanStatus) { called.Store(true) })

	pl.planCompletionWatch(ap)

	if called.Load() {
		t.Fatalf("onPlanCompletedFn was called, want not called when plan ID mismatches")
	}
}

// -------------------------
// planCompletionWatch – successful completion path
// -------------------------

func TestPlanCompletionWatch_PlanCompletesSuccessfully(t *testing.T) {
	pl := &SharedState{}

	ctx := context.Background()
	ap := &ActivePlan{
		ID:     "plan-success",
		Ctx:    ctx,
		Cancel: func() {},
	}

	withVar(t, &getPlanCompletionCheckInterval, func() time.Duration { return 1 * time.Millisecond })
	withVar(t, &getActivePlanForWatch, func(_ *SharedState) *ActivePlan { return ap })

	var calls int32
	withVar(t, &isPlanCompletedFn, func(_ *SharedState, _ *ActivePlan) (bool, error) {
		// First call: not done, second call: done
		if atomic.AddInt32(&calls, 1) >= 2 {
			return true, nil
		}
		return false, nil
	})

	done := make(chan struct{})
	var gotStatus PlanStatus
	withVar(t, &onPlanCompletedFn, func(_ *SharedState, st PlanStatus) {
		gotStatus = st
		close(done)
	})

	go pl.planCompletionWatch(ap)

	select {
	case <-done:
		// ok
	case <-time.After(250 * time.Millisecond):
		t.Fatalf("planCompletionWatch did not mark plan as completed in time")
	}

	if gotStatus != PlanStatusCompleted {
		t.Fatalf("onPlanCompletedFn status = %v, want %v", gotStatus, PlanStatusCompleted)
	}
	if atomic.LoadInt32(&calls) < 2 {
		t.Fatalf("isPlanCompletedFn call count = %d, want >= 2", calls)
	}
}

func TestPlanCompletionWatch_IsPlanCompletedError_RetriesThenCompletes(t *testing.T) {
	pl := &SharedState{}
	ap := &ActivePlan{ID: "plan-retry", Ctx: context.Background(), Cancel: func() {}}

	withVar(t, &getPlanCompletionCheckInterval, func() time.Duration { return 1 * time.Millisecond })
	withVar(t, &getActivePlanForWatch, func(_ *SharedState) *ActivePlan { return ap })

	var n int32
	withVar(t, &isPlanCompletedFn, func(_ *SharedState, _ *ActivePlan) (bool, error) {
		if atomic.AddInt32(&n, 1) == 1 {
			// Any non-nil error triggers the "will retry" branch.
			return false, context.Canceled
		}
		return true, nil
	})

	done := make(chan PlanStatus, 1)
	withVar(t, &onPlanCompletedFn, func(_ *SharedState, st PlanStatus) { done <- st })

	go pl.planCompletionWatch(ap)

	select {
	case st := <-done:
		mustEq(t, st, PlanStatusCompleted, "status mismatch")
	case <-time.After(250 * time.Millisecond):
		t.Fatalf("expected completion after retry, but watcher did not settle in time")
	}

	must(t, atomic.LoadInt32(&n) >= 2, "expected >=2 isPlanCompleted calls, got=%d", n)
}

// -------------------------
// planCompletionWatch – timeout (DeadlineExceeded) -> Failed
// -------------------------

func TestPlanCompletionWatch_TimeoutMarksPlanFailed(t *testing.T) {
	pl := &SharedState{}

	// Already-expired deadline -> Done triggers immediately with DeadlineExceeded.
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	ap := &ActivePlan{
		ID:     "plan-timeout",
		Ctx:    ctx,
		Cancel: func() {},
	}

	withVar(t, &getActivePlanForWatch, func(_ *SharedState) *ActivePlan { return ap })

	statusCh := make(chan PlanStatus, 1)
	withVar(t, &onPlanCompletedFn, func(_ *SharedState, st PlanStatus) {
		statusCh <- st
	})

	pl.planCompletionWatch(ap)

	select {
	case st := <-statusCh:
		if st != PlanStatusFailed {
			t.Fatalf("onPlanCompletedFn status = %v, want %v", st, PlanStatusFailed)
		}
	default:
		t.Fatalf("onPlanCompletedFn was not called on timeout")
	}
}

func TestPlanCompletionWatch_TimeoutButPlanReplaced_DoesNotSettleFailed(t *testing.T) {
	pl := &SharedState{}

	// Already-expired deadline -> Done triggers immediately with DeadlineExceeded.
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	ap := &ActivePlan{ID: "plan-timeout-old", Ctx: ctx, Cancel: func() {}}
	other := &ActivePlan{ID: "plan-timeout-new", Ctx: context.Background(), Cancel: func() {}}

	// When timing out, watcher checks current active plan; since it's not the same ID,
	// it must NOT settle the old plan as failed.
	withVar(t, &getActivePlanForWatch, func(_ *SharedState) *ActivePlan { return other })

	called := atomic.Bool{}
	withVar(t, &onPlanCompletedFn, func(_ *SharedState, _ PlanStatus) { called.Store(true) })

	pl.planCompletionWatch(ap)

	if called.Load() {
		t.Fatalf("onPlanCompletedFn was called, want not called when plan was replaced before timeout handling")
	}
}

// -------------------------
// planCompletionWatch – cancelled context (not DeadlineExceeded) -> no status
// -------------------------

func TestPlanCompletionWatch_CancelledContextDoesNotSettlePlan(t *testing.T) {
	pl := &SharedState{}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	ap := &ActivePlan{
		ID:     "plan-cancelled",
		Ctx:    ctx,
		Cancel: func() {},
	}

	called := atomic.Bool{}
	withVar(t, &onPlanCompletedFn, func(_ *SharedState, _ PlanStatus) {
		called.Store(true)
	})

	pl.planCompletionWatch(ap)

	if called.Load() {
		t.Fatalf("onPlanCompletedFn was called, want not called for context cancellation")
	}
}

func TestPlanCompletionWatch_IntervalNonPositive_DefaultsAndStillExits(t *testing.T) {
	pl := &SharedState{}

	// Force interval <= 0 branch.
	withVar(t, &getPlanCompletionCheckInterval, func() time.Duration { return 0 })

	// Exit immediately via cancellation (so we don't wait for the 500ms ticker).
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	ap := &ActivePlan{ID: "plan-interval-default", Ctx: ctx, Cancel: func() {}}

	called := atomic.Bool{}
	withVar(t, &onPlanCompletedFn, func(_ *SharedState, _ PlanStatus) { called.Store(true) })

	start := time.Now()
	pl.planCompletionWatch(ap)

	if time.Since(start) > 250*time.Millisecond {
		t.Fatalf("expected quick exit; interval<=0 default path should still exit on ctx cancel")
	}
	if called.Load() {
		t.Fatalf("onPlanCompletedFn was called, want not called on ctx cancel")
	}
}
