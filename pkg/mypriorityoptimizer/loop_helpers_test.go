// loop_helpers_test.go
package mypriorityoptimizer

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// -------------------------
// Helpers
// -------------------------

func uidSet(uids ...string) map[types.UID]struct{} {
	m := make(map[types.UID]struct{}, len(uids))
	for _, u := range uids {
		m[types.UID(u)] = struct{}{}
	}
	return m
}

func withBackgroundHooks(
	t *testing.T,
	snapFn func(pl *SharedState) (*PendingSnapshot, error),
	startFn func(pl *SharedState, cfg OptimizeLoopConfig, ctxRun context.Context, runDone chan<- bool),
	body func(),
) {
	t.Helper()

	origSnap := buildPendingSnapshotHook
	origStart := startBackgroundOptimization
	buildPendingSnapshotHook = snapFn
	startBackgroundOptimization = startFn
	t.Cleanup(func() {
		buildPendingSnapshotHook = origSnap
		startBackgroundOptimization = origStart
	})

	body()
}

// small “eventually” helper to avoid flaky sleeps
func eventually(t *testing.T, timeout time.Duration, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(1 * time.Millisecond)
	}
	t.Fatalf("timeout after %v: %s", timeout, msg)
}

// -------------------------
// startLoops
// -------------------------

func TestStartLoops_DoesNothingWhenNotReady(t *testing.T) {
	origMode := OptimizeMode
	defer func() { OptimizeMode = origMode }()

	OptimizeMode = ModePeriodic

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pl := &SharedState{} // PluginReady default is false

	called := false

	withOptimizeLoopFunc(t,
		func(pl *SharedState, ctx context.Context, cfg OptimizeLoopConfig) {
			called = true
		},
		func() {
			pl.startLoops(ctx)
			// If it accidentally started a goroutine, this gives it a chance to run.
			time.Sleep(20 * time.Millisecond)
		},
	)

	if called {
		t.Fatalf("startLoops called optimizeBackgroundLoopFunc even though PluginReady was false")
	}
}

func TestStartLoops_LaunchesPeriodicLoopWhenModePeriodic(t *testing.T) {
	origMode := OptimizeMode
	defer func() { OptimizeMode = origMode }()

	OptimizeMode = ModePeriodic

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pl := &SharedState{}
	pl.PluginReady.Store(true)

	cfgCh := make(chan OptimizeLoopConfig, 1)

	withOptimizeLoopFunc(t,
		func(_ *SharedState, _ context.Context, cfg OptimizeLoopConfig) {
			cfgCh <- cfg
		},
		func() {
			pl.startLoops(ctx)

			select {
			case cfg := <-cfgCh:
				if cfg.Label != "PeriodicLoop" {
					t.Fatalf("expected Label=PeriodicLoop, got %q", cfg.Label)
				}
			case <-time.After(200 * time.Millisecond):
				t.Fatalf("optimizeBackgroundLoopFunc was not called for ModePeriodic")
			}
		},
	)
}

func TestStartLoops_LaunchesInterludeLoopWhenModeInterlude(t *testing.T) {
	origMode := OptimizeMode
	defer func() { OptimizeMode = origMode }()

	OptimizeMode = ModeInterlude

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pl := &SharedState{}
	pl.PluginReady.Store(true)

	cfgCh := make(chan OptimizeLoopConfig, 1)

	withOptimizeLoopFunc(t,
		func(_ *SharedState, _ context.Context, cfg OptimizeLoopConfig) {
			cfgCh <- cfg
		},
		func() {
			pl.startLoops(ctx)

			select {
			case cfg := <-cfgCh:
				if cfg.Label != "InterludeLoop" {
					t.Fatalf("expected Label=InterludeLoop, got %q", cfg.Label)
				}
			case <-time.After(200 * time.Millisecond):
				t.Fatalf("optimizeBackgroundLoopFunc was not called for ModeInterlude")
			}
		},
	)
}

// -------------------------
// optimizeBackgroundLoop
// -------------------------

func TestOptimizeBackgroundLoop_DefaultInterval_ImmediateCancel(t *testing.T) {
	pl := &SharedState{}
	pl.PluginReady.Store(true)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before entering => hit ctx.Done path immediately

	cfg := OptimizeLoopConfig{
		Label:          "TestLoop",
		Interval:       0, // <-- covers interval<=0 => default 1s path
		InterludeDelay: 0,
		CancelOnChange: false,
	}

	// No hooks needed; it should exit immediately on ctx.Done.
	pl.optimizeBackgroundLoop(ctx, cfg)
}

func TestOptimizeBackgroundLoop_BranchScript(t *testing.T) {
	pl := &SharedState{}

	// Fast loop, but still allows interlude gating to fire.
	cfg := OptimizeLoopConfig{
		Label:          "TestLoop",
		Interval:       5 * time.Millisecond,
		InterludeDelay: 15 * time.Millisecond, // cover idle window logic
		CancelOnChange: true,                  // cover cancel-on-change branch
	}

	// Snapshots we will “serve” via the hook.
	snapErr := errors.New("snap boom")
	snapEmpty := &PendingSnapshot{PendingUIDs: uidSet(), PendingCount: 0, Fingerprint: "fp0"}
	snapU1 := &PendingSnapshot{PendingUIDs: uidSet("u1"), PendingCount: 1, Fingerprint: "fp1"}
	snapU2 := &PendingSnapshot{PendingUIDs: uidSet("u2"), PendingCount: 1, Fingerprint: "fp2"}
	snapU3 := &PendingSnapshot{PendingUIDs: uidSet("u3"), PendingCount: 1, Fingerprint: "fp3"}

	// Control knobs for the snapshot hook.
	var servedErrOnce atomic.Bool
	var serveEmpty atomic.Bool
	var serveU3 atomic.Bool

	// Run counters + ctx capture from started runs.
	var runCount atomic.Int32
	run1CtxCh := make(chan context.Context, 1)
	run3CtxCh := make(chan context.Context, 1)

	ctx, cancel := context.WithCancel(context.Background())

	// Make sure we also cover:
	// - PluginReady=false warm-up branch
	// - Active plan branch
	pl.PluginReady.Store(false)

	// NOTE: getActivePlan() may depend on ActivePlan and/or ActivePlanInProgress in your impl,
	// so we set both during the “active plan” window.
	go func() {
		time.Sleep(10 * time.Millisecond)
		pl.PluginReady.Store(true)

		// Active plan for a brief window.
		pl.ActivePlanInProgress.Store(true)
		pl.ActivePlan.Store(&ActivePlan{ID: "ap-1"})
		time.Sleep(10 * time.Millisecond)

		// Clear it (typed-nil works for both atomic.Pointer and atomic.Value setups).
		pl.ActivePlanInProgress.Store(false)
		pl.ActivePlan.Store((*ActivePlan)(nil))
	}()

	withBackgroundHooks(t,
		// buildPendingSnapshotHook
		func(_ *SharedState) (*PendingSnapshot, error) {
			// 1) Force snapshot error once.
			if !servedErrOnce.Load() {
				servedErrOnce.Store(true)
				return nil, snapErr
			}
			// 2) After we decide to serve empty/u3, do that.
			if serveU3.Load() {
				return snapU3, nil
			}
			if serveEmpty.Load() {
				return snapEmpty, nil
			}
			// 3) Before first run starts, serve u1. After first run starts, switch to u2
			//    so the in-flight run sees a changed set and cancels.
			if runCount.Load() >= 1 {
				return snapU2, nil
			}
			return snapU1, nil
		},

		// startBackgroundOptimization hook
		func(_ *SharedState, _ OptimizeLoopConfig, ctxRun context.Context, runDone chan<- bool) {
			n := runCount.Add(1)

			switch n {
			case 1:
				// Run 1: stays in-flight until cancelled -> returns solved=false.
				run1CtxCh <- ctxRun
				go func() {
					<-ctxRun.Done()
					runDone <- false
				}()
			case 2:
				// Run 2: immediate solved=true => should set lastSolvedSet+fingerprint and enable skip.
				runDone <- true
			case 3:
				// Run 3: used to cover ctx.Done branch (outer ctx cancelled while run in-flight).
				run3CtxCh <- ctxRun
				go func() {
					<-ctxRun.Done()
					runDone <- true
				}()
			default:
				// Should not happen; keep the loop unblocked if it does.
				runDone <- false
			}
		},

		func() {
			done := make(chan struct{})
			go func() {
				pl.optimizeBackgroundLoop(ctx, cfg)
				close(done)
			}()

			// ---- Run 1: must start, then be cancelled due to pending set change.
			var run1 context.Context
			select {
			case run1 = <-run1CtxCh:
			case <-time.After(500 * time.Millisecond):
				t.Fatalf("run #1 did not start")
			}
			select {
			case <-run1.Done():
				// ok: cancel-on-change fired
			case <-time.After(500 * time.Millisecond):
				t.Fatalf("run #1 was not cancelled (expected cancel-on-change)")
			}

			// ---- Run 2: must start, and then we should not start any extra runs
			// because lastSolvedSet+fingerprint match => skip optimization.
			eventually(t, 700*time.Millisecond, func() bool {
				return runCount.Load() >= 2
			}, "run #2 did not start")

			// Allow time for “run finished” to be observed and “skip” to kick in.
			time.Sleep(40 * time.Millisecond)

			// If skip is working, we should still be at exactly 2 runs.
			if got := runCount.Load(); got != 2 {
				t.Fatalf("expected skip to prevent extra runs; runCount=%d, want 2", got)
			}

			// ---- pendingCount==0 path (also resets state when previous state exists)
			serveEmpty.Store(true)
			time.Sleep(20 * time.Millisecond)
			serveEmpty.Store(false)

			// ---- Run 3: switch to u3, let it start, then cancel outer ctx to hit ctx.Done cleanup.
			serveU3.Store(true)

			var run3 context.Context
			select {
			case run3 = <-run3CtxCh:
			case <-time.After(700 * time.Millisecond):
				t.Fatalf("run #3 did not start")
			}

			// Cancel outer loop while run is in-flight: ctx.Done branch must cancel run + drain runDone.
			cancel()

			select {
			case <-run3.Done():
				// ok: ctx.Done path cancelled the in-flight run
			case <-time.After(500 * time.Millisecond):
				t.Fatalf("run #3 was not cancelled by outer ctx.Done")
			}

			select {
			case <-done:
				// ok: loop exited
			case <-time.After(700 * time.Millisecond):
				t.Fatalf("optimizeBackgroundLoop did not exit after ctx cancel")
			}
		},
	)
}

// -------------------------
// isSameUIDSet
// -------------------------

func TestIsSameUIDSet_NilVsNil(t *testing.T) {
	if !isSameUIDSet(nil, nil) {
		t.Fatalf("isSameUIDSet(nil, nil) = false, want true")
	}
}

func TestIsSameUIDSet_NilVsNonNil(t *testing.T) {
	a := uidSet("u1")
	if isSameUIDSet(nil, a) || isSameUIDSet(a, nil) {
		t.Fatalf("isSameUIDSet(nil, non-nil) or reverse = true, want false")
	}
}

func TestIsSameUIDSet_DifferentLengths(t *testing.T) {
	a := uidSet("u1")
	b := uidSet("u1", "u2")
	if isSameUIDSet(a, b) {
		t.Fatalf("isSameUIDSet() with different lengths = true, want false")
	}
}

func TestIsSameUIDSet_SameElements(t *testing.T) {
	a := uidSet("u1", "u2")
	b := uidSet("u2", "u1")
	if !isSameUIDSet(a, b) {
		t.Fatalf("isSameUIDSet() with same elements = false, want true")
	}
}

func TestIsSameUIDSet_DifferentElements(t *testing.T) {
	a := uidSet("u1", "u2")
	b := uidSet("u1", "u3")
	if isSameUIDSet(a, b) {
		t.Fatalf("isSameUIDSet() with different elements = true, want false")
	}
}

// -------------------------
// cloneUIDSet
// -------------------------

func TestCloneUIDSet_Nil(t *testing.T) {
	if got := cloneUIDSet(nil); got != nil {
		t.Fatalf("cloneUIDSet(nil) = %#v, want nil", got)
	}
}

func TestCloneUIDSet_Independence(t *testing.T) {
	src := uidSet("u1", "u2")
	cloned := cloneUIDSet(src)
	if !isSameUIDSet(src, cloned) {
		t.Fatalf("cloneUIDSet() produced different contents: src=%v cloned=%v", src, cloned)
	}
	// Mutate clone and ensure src is unaffected
	delete(cloned, types.UID("u1"))
	if _, ok := src[types.UID("u1")]; !ok {
		t.Fatalf("mutating clone mutated source: src=%v cloned=%v", src, cloned)
	}
	// Mutate src and ensure clone is unaffected
	src[types.UID("u3")] = struct{}{}
	if _, ok := cloned[types.UID("u3")]; ok {
		t.Fatalf("mutating source mutated clone: src=%v cloned=%v", src, cloned)
	}
}

// -------------------------
// isAlreadyComputedForPendingSet
// -------------------------

func TestIsAlreadyComputedForPendingSet_BestAttemptNil(t *testing.T) {
	if got := isAlreadyComputedForPendingSet(nil, nil); got {
		t.Fatalf("isAlreadyComputedForPendingSet(nil, nil) = true, want false")
	}
}

func TestIsAlreadyComputedForPendingSet_NonOptimalStatus(t *testing.T) {
	best := &SolverResult{Status: "FEASIBLE"}
	if got := isAlreadyComputedForPendingSet(ErrNoImprovingSolutionFromAnySolver, best); got {
		t.Fatalf("isAlreadyComputedForPendingSet(non-OPTIMAL) = true, want false")
	}
}

func TestIsAlreadyComputedForPendingSet_OptimalWithNoImprovementErr(t *testing.T) {
	best := &SolverResult{Status: "OPTIMAL"}
	if got := isAlreadyComputedForPendingSet(ErrNoImprovingSolutionFromAnySolver, best); !got {
		t.Fatalf("isAlreadyComputedForPendingSet(OPTIMAL, ErrNoImprovingSolutionFromAnySolver) = false, want true")
	}
}

func TestIsAlreadyComputedForPendingSet_OptimalWithNoPendingPodsErr(t *testing.T) {
	best := &SolverResult{Status: "OPTIMAL"}
	if got := isAlreadyComputedForPendingSet(ErrNoPendingPodsScheduled, best); !got {
		t.Fatalf("isAlreadyComputedForPendingSet(OPTIMAL, ErrNoPendingPodsScheduled) = false, want true")
	}
}

func TestIsAlreadyComputedForPendingSet_OptimalWithOtherError(t *testing.T) {
	best := &SolverResult{Status: "OPTIMAL"}
	if got := isAlreadyComputedForPendingSet(context.DeadlineExceeded, best); got {
		t.Fatalf("isAlreadyComputedForPendingSet(OPTIMAL, other error) = true, want false")
	}
}

// -------------------------
// buildPendingSnapshot
// -------------------------

func TestBuildPendingSnapshot(t *testing.T) {
	pl := &SharedState{}

	// One usable node
	n := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "n1"},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1000m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
		},
	}

	// Pending pod (counts)
	pPending := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "p-pending", Namespace: "ns", UID: types.UID("pu1")},
		Status:     v1.PodStatus{Phase: v1.PodPending},
	}

	// Running pod (ignored)
	pRunning := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "p-running", Namespace: "ns", UID: types.UID("pu2")},
		Status:     v1.PodStatus{Phase: v1.PodRunning},
		Spec: v1.PodSpec{
			NodeName: "n1",
			Containers: []v1.Container{{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				},
			}},
		},
	}

	// Pending but deleting (must be ignored)
	now := metav1.Now()
	pDeletingPending := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "p-deleting",
			Namespace:         "ns",
			UID:               types.UID("pu3"),
			DeletionTimestamp: &now,
		},
		Status: v1.PodStatus{Phase: v1.PodPending},
	}

	// Store includes a nil pod pointer to hit (p == nil) branch.
	store := map[string]map[string]*v1.Pod{
		"ns": {
			"p-pending":  pPending,
			"p-running":  pRunning,
			"p-deleting": pDeletingPending,
			"p-nil":      nil,
		},
	}

	withNodeLister(&fakeNodeLister{nodes: []*v1.Node{n}}, func() {
		withPodLister(&fakePodLister{store: store}, func() {
			snap, err := pl.buildPendingSnapshot()
			if err != nil {
				t.Fatalf("buildPendingSnapshot() unexpected error: %v", err)
			}

			// Only pPending should count.
			if snap.PendingCount != 1 {
				t.Fatalf("PendingCount = %d, want 1", snap.PendingCount)
			}
			if _, ok := snap.PendingUIDs[pPending.UID]; !ok {
				t.Fatalf("pending UID set missing %q", pPending.UID)
			}

			// Deleting pending must be excluded.
			if _, ok := snap.PendingUIDs[pDeletingPending.UID]; ok {
				t.Fatalf("deleting pending pod must be excluded from pending set")
			}

			if snap.Fingerprint == "" {
				t.Fatalf("Fingerprint should not be empty")
			}
			if len(snap.Pods) != 4 || len(snap.Nodes) != 1 {
				t.Fatalf("snap sizes wrong: pods=%d nodes=%d", len(snap.Pods), len(snap.Nodes))
			}
		})
	})
}

func TestBuildPendingSnapshot_NodesErrorPropagated(t *testing.T) {
	pl := &SharedState{}
	sentinel := errors.New("nodes boom")

	withNodeLister(&fakeNodeLister{err: sentinel}, func() {
		// Pod lister should not matter if node listing already fails,
		// but we provide a no-op lister for completeness.
		withPodLister(&fakePodLister{}, func() {
			snap, err := pl.buildPendingSnapshot()
			if snap != nil {
				t.Fatalf("expected nil snapshot on error, got %#v", snap)
			}
			if !errors.Is(err, sentinel) {
				t.Fatalf("buildPendingSnapshot() err = %v, want wrapped %v", err, sentinel)
			}
		})
	})
}

func TestBuildPendingSnapshot_PodsErrorPropagated(t *testing.T) {
	pl := &SharedState{}
	sentinel := errors.New("pods boom")

	// One usable node so we get past node listing.
	n := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "n1"},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1000m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
		},
	}

	withNodeLister(&fakeNodeLister{nodes: []*v1.Node{n}}, func() {
		withPodLister(&fakePodLister{err: sentinel}, func() {
			snap, err := pl.buildPendingSnapshot()
			if snap != nil {
				t.Fatalf("expected nil snapshot on error, got %#v", snap)
			}
			if !errors.Is(err, sentinel) {
				t.Fatalf("buildPendingSnapshot() err = %v, want wrapped %v", err, sentinel)
			}
		})
	})
}
