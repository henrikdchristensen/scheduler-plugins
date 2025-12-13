package mypriorityoptimizer

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
)

// -------------------------
// Helpers
// -------------------------

type readinessEnv struct {
	interval time.Duration
	warmup   time.Duration

	getNodes func(*SharedState) ([]*v1.Node, error)
	isUsable func(*v1.Node) bool

	persist    func(*SharedState, context.Context) error
	activate   func(*SharedState)
	startLoops func(*SharedState, context.Context)
}

func withReadinessEnv(t *testing.T, env readinessEnv, fn func()) {
	t.Helper()

	// defaults
	if env.interval == 0 {
		env.interval = 1 * time.Millisecond
	}
	if env.getNodes == nil {
		env.getNodes = func(*SharedState) ([]*v1.Node, error) { return nil, nil }
	}
	if env.isUsable == nil {
		env.isUsable = func(*v1.Node) bool { return false }
	}
	if env.persist == nil {
		env.persist = func(*SharedState, context.Context) error { return nil }
	}
	if env.activate == nil {
		env.activate = func(*SharedState) {}
	}
	if env.startLoops == nil {
		env.startLoops = func(*SharedState, context.Context) {}
	}

	// save
	oldInterval := readinessUsableNodeInterval
	oldWarmup := cacheWarmupDelay
	oldGet := getNodesForReadiness
	oldUsable := isNodeUsableForReadiness
	oldPersist := persistPluginConfigForReadiness
	oldActivate := activateBlockedPodsForReadiness
	oldStart := startLoopsForReadiness

	// set
	readinessUsableNodeInterval = env.interval
	cacheWarmupDelay = env.warmup
	getNodesForReadiness = env.getNodes
	isNodeUsableForReadiness = env.isUsable
	persistPluginConfigForReadiness = env.persist
	activateBlockedPodsForReadiness = env.activate
	startLoopsForReadiness = env.startLoops

	t.Cleanup(func() {
		readinessUsableNodeInterval = oldInterval
		cacheWarmupDelay = oldWarmup
		getNodesForReadiness = oldGet
		isNodeUsableForReadiness = oldUsable
		persistPluginConfigForReadiness = oldPersist
		activateBlockedPodsForReadiness = oldActivate
		startLoopsForReadiness = oldStart
	})

	fn()
}

func newTestPodInformer() cache.SharedIndexInformer {
	lw := &cache.ListWatch{
		ListFunc: func(_ metav1.ListOptions) (runtime.Object, error) {
			return &v1.PodList{}, nil
		},
		WatchFunc: func(_ metav1.ListOptions) (watch.Interface, error) {
			return watch.NewFake(), nil
		},
	}
	return cache.NewSharedIndexInformer(lw, &v1.Pod{}, 0, cache.Indexers{})
}

// -------------------------
// isCacheReady
// -------------------------

func TestIsCacheReady_NoInformers_ReturnsTrue(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if got := isCacheReady(ctx); !got {
		t.Fatalf("isCacheReady() = %v, want true", got)
	}
}

func TestIsCacheReady_AllNilInformers_ReturnsTrue(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if got := isCacheReady(ctx, nil, nil); !got {
		t.Fatalf("isCacheReady(nil informers) = %v, want true", got)
	}
}

func TestIsCacheReady_ContextCanceled_ReturnsFalse(t *testing.T) {
	inf := newTestPodInformer()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if got := isCacheReady(ctx, inf); got {
		t.Fatalf("isCacheReady(canceled ctx) = %v, want false", got)
	}
}

// -------------------------
// waitForUsableNode
// -------------------------

func TestWaitForUsableNode_ContextCanceled_ReturnsFalse(t *testing.T) {
	pl := &SharedState{}
	withReadinessEnv(t, readinessEnv{
		getNodes: func(*SharedState) ([]*v1.Node, error) { return []*v1.Node{node("nodeA")}, nil },
		isUsable: func(*v1.Node) bool { return true },
	}, func() {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		if got := pl.waitForUsableNode(ctx); got {
			t.Fatalf("waitForUsableNode(canceled ctx) = %v, want false", got)
		}
	})
}

func TestWaitForUsableNode_EventuallyFindsUsableNode(t *testing.T) {
	pl := &SharedState{}
	var calls atomic.Int32

	withReadinessEnv(t, readinessEnv{
		getNodes: func(*SharedState) ([]*v1.Node, error) {
			switch calls.Add(1) {
			case 1:
				return []*v1.Node{node("nodeA")}, nil
			default:
				return []*v1.Node{node("nodeB")}, nil
			}
		},
		isUsable: func(n *v1.Node) bool { return n != nil && n.Name == "nodeB" },
	}, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
		defer cancel()

		if got := pl.waitForUsableNode(ctx); !got {
			t.Fatalf("waitForUsableNode() = %v, want true", got)
		}
		if calls.Load() < 2 {
			t.Fatalf("expected getNodesForReadiness to be called at least twice, got %d", calls.Load())
		}
	})
}

func TestWaitForUsableNode_GetNodesErrorThenSucceeds(t *testing.T) {
	pl := &SharedState{}
	var calls atomic.Int32

	withReadinessEnv(t, readinessEnv{
		getNodes: func(*SharedState) ([]*v1.Node, error) {
			switch calls.Add(1) {
			case 1:
				return nil, context.DeadlineExceeded
			default:
				return []*v1.Node{node("nodeB")}, nil
			}
		},
		isUsable: func(n *v1.Node) bool { return n != nil && n.Name == "nodeB" },
	}, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
		defer cancel()

		if got := pl.waitForUsableNode(ctx); !got {
			t.Fatalf("waitForUsableNode() = %v, want true", got)
		}
	})
}

// -------------------------
// pluginReadiness
// -------------------------

func TestPluginReadiness_InformerSyncCanceled_DoesNotMarkReady_AndNoSideEffects(t *testing.T) {
	pl := &SharedState{}
	pl.BlockedWhileActive = newPodSet("blocked")
	pl.PluginReady.Store(false)

	var persistCalls, activateCalls, startCalls atomic.Int32

	inf := newTestPodInformer()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	withReadinessEnv(t, readinessEnv{
		persist:    func(*SharedState, context.Context) error { persistCalls.Add(1); return nil },
		activate:   func(*SharedState) { activateCalls.Add(1) },
		startLoops: func(*SharedState, context.Context) { startCalls.Add(1) },
	}, func() {
		pl.pluginReadiness(ctx, inf)

		if pl.PluginReady.Load() {
			t.Fatalf("PluginReady = true, want false when informers never synced")
		}
		if persistCalls.Load() != 0 || activateCalls.Load() != 0 || startCalls.Load() != 0 {
			t.Fatalf("side effects should not run (persist=%d activate=%d start=%d)",
				persistCalls.Load(), activateCalls.Load(), startCalls.Load(),
			)
		}
	})
}

func TestPluginReadiness_WarmupCanceled_DoesNotMarkReady_AndNoSideEffects(t *testing.T) {
	pl := &SharedState{}
	pl.BlockedWhileActive = newPodSet("blocked")
	pl.PluginReady.Store(false)

	var persistCalls, activateCalls, startCalls atomic.Int32

	withReadinessEnv(t, readinessEnv{
		warmup:   250 * time.Millisecond,
		getNodes: func(*SharedState) ([]*v1.Node, error) { return []*v1.Node{node("n1")}, nil },
		isUsable: func(*v1.Node) bool { return true },
		persist:  func(*SharedState, context.Context) error { persistCalls.Add(1); return nil },
		activate: func(*SharedState) { activateCalls.Add(1) },
		startLoops: func(*SharedState, context.Context) {
			startCalls.Add(1)
		},
	}, func() {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		pl.pluginReadiness(ctx)

		if pl.PluginReady.Load() {
			t.Fatalf("PluginReady = true, want false when warmup is canceled")
		}
		if persistCalls.Load() != 0 || activateCalls.Load() != 0 || startCalls.Load() != 0 {
			t.Fatalf("side effects should not run (persist=%d activate=%d start=%d)",
				persistCalls.Load(), activateCalls.Load(), startCalls.Load(),
			)
		}
	})
}

func TestPluginReadiness_UsableNodeNeverFound_DoesNotMarkReady_AndNoSideEffects(t *testing.T) {
	pl := &SharedState{}
	pl.BlockedWhileActive = newPodSet("blocked")
	pl.PluginReady.Store(false)

	var persistCalls, activateCalls, startCalls atomic.Int32

	withReadinessEnv(t, readinessEnv{
		warmup:   0,
		getNodes: func(*SharedState) ([]*v1.Node, error) { return []*v1.Node{node("n1")}, nil },
		isUsable: func(*v1.Node) bool { return false },
		persist:  func(*SharedState, context.Context) error { persistCalls.Add(1); return nil },
		activate: func(*SharedState) { activateCalls.Add(1) },
		startLoops: func(*SharedState, context.Context) {
			startCalls.Add(1)
		},
	}, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
		defer cancel()

		pl.pluginReadiness(ctx)

		if pl.PluginReady.Load() {
			t.Fatalf("PluginReady = true, want false when no usable node is found")
		}
		if persistCalls.Load() != 0 || activateCalls.Load() != 0 || startCalls.Load() != 0 {
			t.Fatalf("side effects should not run (persist=%d activate=%d start=%d)",
				persistCalls.Load(), activateCalls.Load(), startCalls.Load(),
			)
		}
	})
}

func TestPluginReadiness_Success_MarksReady_CallsSideEffects_AndHonorsWarmup(t *testing.T) {
	pl := &SharedState{}
	pl.BlockedWhileActive = newPodSet("blocked")
	pl.PluginReady.Store(false)

	var persistCalls, activateCalls, startCalls atomic.Int32

	withReadinessEnv(t, readinessEnv{
		warmup:   20 * time.Millisecond,
		getNodes: func(*SharedState) ([]*v1.Node, error) { return []*v1.Node{node("n1")}, nil },
		isUsable: func(*v1.Node) bool { return true },
		persist:  func(*SharedState, context.Context) error { persistCalls.Add(1); return nil },
		activate: func(*SharedState) { activateCalls.Add(1) },
		startLoops: func(*SharedState, context.Context) {
			startCalls.Add(1)
		},
	}, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
		defer cancel()

		start := time.Now()
		pl.pluginReadiness(ctx)
		elapsed := time.Since(start)

		if !pl.PluginReady.Load() {
			t.Fatalf("PluginReady = false, want true on success")
		}
		if persistCalls.Load() != 1 || activateCalls.Load() != 1 || startCalls.Load() != 1 {
			t.Fatalf("expected side effects once (persist=%d activate=%d start=%d)",
				persistCalls.Load(), activateCalls.Load(), startCalls.Load(),
			)
		}
		if elapsed < 15*time.Millisecond {
			t.Fatalf("expected warmup delay to be honored, elapsed=%v", elapsed)
		}
	})
}
