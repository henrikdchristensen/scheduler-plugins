// loop_helpers_test.go
// with the help of AI tools to cover more branches/cases
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
// Test Helpers
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

// helper that retries until timeout for a condition to become true
func eventually(t *testing.T, timeout time.Duration, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timeout after %v: %s", timeout, msg)
}

func withOptimizeLoopFunc(t *testing.T,
	fn func(pl *SharedState, ctx context.Context, cfg OptimizeLoopConfig),
	body func(),
) {
	t.Helper()
	orig := optimizeBackgroundLoopFunc
	optimizeBackgroundLoopFunc = fn
	t.Cleanup(func() { optimizeBackgroundLoopFunc = orig })
	body()
}

// -------------------------
// startLoops
// -------------------------

func TestStartLoops_DoesNothingWhenNotReady(t *testing.T) {
	withVar(t, &OptimizeMode, ModePeriodic)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	pl := &SharedState{} // PluginReady default is false
	calledCh := make(chan OptimizeLoopConfig, 1)

	withOptimizeLoopFunc(t,
		func(_ *SharedState, _ context.Context, cfg OptimizeLoopConfig) { calledCh <- cfg },
		func() {
			pl.startLoops(ctx)
			select {
			case cfg := <-calledCh:
				t.Fatalf("unexpected loop start: cfg=%+v (PluginReady=false)", cfg)
			case <-time.After(50 * time.Millisecond):
				// ok
			}
		},
	)
}

func TestStartLoops_StartsPeriodicLoopWhenModePeriodic(t *testing.T) {
	withVar(t, &OptimizeMode, ModePeriodic)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	pl := &SharedState{}
	pl.PluginReady.Store(true)

	cfgCh := make(chan OptimizeLoopConfig, 1)

	withOptimizeLoopFunc(t,
		func(_ *SharedState, _ context.Context, cfg OptimizeLoopConfig) { cfgCh <- cfg },
		func() {
			pl.startLoops(ctx)
			select {
			case cfg := <-cfgCh:
				if cfg.Label != "PeriodicLoop" {
					t.Fatalf("expected Label=PeriodicLoop, got %q", cfg.Label)
				}
			case <-time.After(500 * time.Millisecond):
				t.Fatalf("optimizeBackgroundLoopFunc was not called for ModePeriodic")
			}
		},
	)
}

func TestStartLoops_StartsInterludeLoopWhenModeInterlude(t *testing.T) {
	withVar(t, &OptimizeMode, ModeInterlude)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	pl := &SharedState{}
	pl.PluginReady.Store(true)

	cfgCh := make(chan OptimizeLoopConfig, 1)

	withOptimizeLoopFunc(t,
		func(_ *SharedState, _ context.Context, cfg OptimizeLoopConfig) { cfgCh <- cfg },
		func() {
			pl.startLoops(ctx)
			select {
			case cfg := <-cfgCh:
				if cfg.Label != "InterludeLoop" {
					t.Fatalf("expected Label=InterludeLoop, got %q", cfg.Label)
				}
			case <-time.After(500 * time.Millisecond):
				t.Fatalf("optimizeBackgroundLoopFunc was not called for ModeInterlude")
			}
		},
	)
}

// -------------------------
// optimizeBackgroundLoop
// -------------------------

func TestOptimizeBackgroundLoop_ImmediateCancel(t *testing.T) {
	pl := &SharedState{}
	pl.PluginReady.Store(true)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before entering => hit ctx.Done path immediately

	cfg := OptimizeLoopConfig{
		Label:          "TestLoop",
		Interval:       0, // covers interval<=0 => default 1s path
		InterludeDelay: 0,
		CancelOnChange: false,
	}

	// No hooks needed; it should exit immediately on ctx.Done.
	pl.optimizeBackgroundLoop(ctx, cfg)
}

func TestOptimizeBackgroundLoop_BranchScript(t *testing.T) {
	pl := &SharedState{}

	cfg := OptimizeLoopConfig{
		Label:          "TestLoop",
		Interval:       5 * time.Millisecond,
		InterludeDelay: 15 * time.Millisecond,
		CancelOnChange: true,
	}

	// Snapshots we will “serve” via the hook.
	snapErr := errors.New("snap boom")
	snapEmpty := &PendingSnapshot{PendingUIDs: uidSet(), PendingCount: 0, Fingerprint: "fp0"}
	snapU1 := &PendingSnapshot{PendingUIDs: uidSet("u1"), PendingCount: 1, Fingerprint: "fp1"}
	snapU2 := &PendingSnapshot{PendingUIDs: uidSet("u2"), PendingCount: 1, Fingerprint: "fp2"}
	snapU3 := &PendingSnapshot{PendingUIDs: uidSet("u3"), PendingCount: 1, Fingerprint: "fp3"}

	var servedErrOnce atomic.Bool
	var serveEmpty atomic.Bool
	var serveU3 atomic.Bool

	var runCount atomic.Int32
	run1CtxCh := make(chan context.Context, 1)
	run3CtxCh := make(chan context.Context, 1)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	// Cover PluginReady=false warm-up branch + active plan skip branch.
	pl.PluginReady.Store(false)
	time.AfterFunc(10*time.Millisecond, func() {
		pl.PluginReady.Store(true)
		pl.ActivePlanInProgress.Store(true)
		pl.ActivePlan.Store(&ActivePlan{ID: "ap-1"})
		time.AfterFunc(10*time.Millisecond, func() {
			pl.ActivePlanInProgress.Store(false)
			pl.ActivePlan.Store((*ActivePlan)(nil))
		})
	})

	withBackgroundHooks(t,
		// buildPendingSnapshotHook
		func(_ *SharedState) (*PendingSnapshot, error) {
			if !servedErrOnce.Load() {
				servedErrOnce.Store(true)
				return nil, snapErr
			}
			if serveU3.Load() {
				return snapU3, nil
			}
			if serveEmpty.Load() {
				return snapEmpty, nil
			}
			// Before first run starts -> u1. While run1 is in-flight -> u2 (forces cancel-on-change).
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
				run1CtxCh <- ctxRun
				go func() {
					<-ctxRun.Done()
					runDone <- false
				}()
			case 2:
				// solved=true should record lastSolvedSet+fingerprint => skip further runs for same set+fp.
				runDone <- true
			case 3:
				run3CtxCh <- ctxRun
				go func() {
					<-ctxRun.Done()
					runDone <- true
				}()
			default:
				runDone <- false
			}
		},

		func() {
			done := make(chan struct{})
			go func() {
				pl.optimizeBackgroundLoop(ctx, cfg)
				close(done)
			}()

			// Run 1: must start, then be cancelled due to pending set change.
			var run1 context.Context
			select {
			case run1 = <-run1CtxCh:
			case <-time.After(2 * time.Second):
				t.Fatalf("run #1 did not start")
			}
			select {
			case <-run1.Done():
			case <-time.After(2 * time.Second):
				t.Fatalf("run #1 was not cancelled (expected cancel-on-change)")
			}

			// Run 2: must start.
			eventually(t, 2*time.Second, func() bool {
				return runCount.Load() >= 2
			}, "run #2 did not start")

			// Let loop observe completion and establish skip state.
			time.Sleep(30 * time.Millisecond)

			// If skip works (same set+fingerprint), it should not start extra runs yet.
			if got := runCount.Load(); got != 2 {
				t.Fatalf("expected skip to prevent extra runs; runCount=%d, want 2", got)
			}

			// pendingCount==0 path resets internal state.
			serveEmpty.Store(true)
			time.Sleep(20 * time.Millisecond)
			serveEmpty.Store(false)

			// Run 3: serve u3, let it start, then cancel outer ctx to hit ctx.Done cleanup.
			serveU3.Store(true)

			var run3 context.Context
			select {
			case run3 = <-run3CtxCh:
			case <-time.After(2 * time.Second):
				t.Fatalf("run #3 did not start")
			}

			cancel()

			select {
			case <-run3.Done():
			case <-time.After(2 * time.Second):
				t.Fatalf("run #3 was not cancelled by outer ctx.Done")
			}

			select {
			case <-done:
			case <-time.After(2 * time.Second):
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
	delete(cloned, types.UID("u1"))
	if _, ok := src[types.UID("u1")]; !ok {
		t.Fatalf("mutating clone mutated source: src=%v cloned=%v", src, cloned)
	}
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

func TestIsAlreadyComputedForPendingSet_OptimalWithNoImprovementError(t *testing.T) {
	best := &SolverResult{Status: "OPTIMAL"}
	if got := isAlreadyComputedForPendingSet(ErrNoImprovingSolutionFromAnySolver, best); !got {
		t.Fatalf("isAlreadyComputedForPendingSet(OPTIMAL, ErrNoImprovingSolutionFromAnySolver) = false, want true")
	}
}

func TestIsAlreadyComputedForPendingSet_OptimalWithNoPendingPodsError(t *testing.T) {
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

	pPending := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "p-pending", Namespace: "ns", UID: types.UID("pu1")},
		Status:     v1.PodStatus{Phase: v1.PodPending},
	}

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

	store := map[string]map[string]*v1.Pod{
		"ns": {
			"p-pending":  pPending,
			"p-running":  pRunning,
			"p-deleting": pDeletingPending,
			"p-nil":      nil,
		},
	}

	withNodeLister(&FakeNodeLister{Nodes: []*v1.Node{n}}, func() {
		withPodLister(&FakePodLister{Store: store}, func() {
			snap, err := pl.buildPendingSnapshot()
			if err != nil {
				t.Fatalf("buildPendingSnapshot() unexpected error: %v", err)
			}

			if snap.PendingCount != 1 {
				t.Fatalf("PendingCount = %d, want 1", snap.PendingCount)
			}
			if _, ok := snap.PendingUIDs[pPending.UID]; !ok {
				t.Fatalf("pending UID set missing %q", pPending.UID)
			}
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

	withNodeLister(&FakeNodeLister{Error: sentinel}, func() {
		withPodLister(&FakePodLister{}, func() {
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

	withNodeLister(&FakeNodeLister{Nodes: []*v1.Node{n}}, func() {
		withPodLister(&FakePodLister{Error: sentinel}, func() {
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
