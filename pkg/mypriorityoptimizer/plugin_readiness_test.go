// plugin_readiness_test.go
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
// isCacheReady
// -------------------------

func TestIsCacheReadyn(t *testing.T) {
	tests := []struct {
		name      string
		ctxFn     func() (context.Context, context.CancelFunc)
		informers []cache.SharedIndexInformer
		startInf  bool
		want      bool
	}{
		{
			name:  "no informers",
			ctxFn: func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) },
			want:  true,
		},
		{
			name:      "all nil informers",
			ctxFn:     func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) },
			informers: []cache.SharedIndexInformer{nil, nil},
			want:      true,
		},
		{
			name: "context canceled returns false",
			ctxFn: func() (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx, func() {}
			},
			informers: []cache.SharedIndexInformer{newTestPodInformer()},
			want:      false,
		},
		{
			name: "running informer returns true",
			ctxFn: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), 500*time.Millisecond)
			},
			informers: []cache.SharedIndexInformer{newTestPodInformer()},
			startInf:  true,
			want:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := tt.ctxFn()
			defer cancel()
			infs := tt.informers

			if tt.startInf && len(infs) > 0 && infs[0] != nil {
				startInformer(t, infs[0])
			}

			got := isCacheReady(ctx, infs...)
			if got != tt.want {
				t.Fatalf("isCacheReady() = %v, want %v", got, tt.want)
			}
		})
	}
}

// -------------------------
// waitForUsableNode
// -------------------------

func TestWaitForUsableNode_ContextCanceled(t *testing.T) {
	pl := &SharedState{}
	withReadinessEnv(t, ReadinessEnvs{
		GetNodes: func(*SharedState) ([]*v1.Node, error) { return []*v1.Node{node("nodeA")}, nil },
		IsUsable: func(*v1.Node) bool { return true },
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

	withReadinessEnv(t, ReadinessEnvs{
		GetNodes: func(*SharedState) ([]*v1.Node, error) {
			switch calls.Add(1) {
			case 1:
				return []*v1.Node{node("nodeA")}, nil
			default:
				return []*v1.Node{node("nodeB")}, nil
			}
		},
		IsUsable: func(n *v1.Node) bool { return n != nil && n.Name == "nodeB" },
	}, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
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

	withReadinessEnv(t, ReadinessEnvs{
		GetNodes: func(*SharedState) ([]*v1.Node, error) {
			switch calls.Add(1) {
			case 1:
				return nil, context.DeadlineExceeded
			default:
				return []*v1.Node{node("nodeB")}, nil
			}
		},
		IsUsable: func(n *v1.Node) bool { return n != nil && n.Name == "nodeB" },
	}, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
		defer cancel()

		if got := pl.waitForUsableNode(ctx); !got {
			t.Fatalf("waitForUsableNode() = %v, want true", got)
		}
		if calls.Load() < 2 {
			t.Fatalf("expected getNodesForReadiness >=2, got %d", calls.Load())
		}
	})
}

// -------------------------
// pluginReadiness
// -------------------------

func TestPluginReadiness_InformerSyncCanceled(t *testing.T) {
	pl := &SharedState{}
	pl.BlockedWhileActive = newPodSet("blocked")
	pl.PluginReady.Store(false)

	var persistCalls, activateCalls, startCalls atomic.Int32

	inf := newTestPodInformer()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	withReadinessEnv(t, ReadinessEnvs{
		Persist:    func(*SharedState, context.Context) error { persistCalls.Add(1); return nil },
		Activate:   func(*SharedState) { activateCalls.Add(1) },
		StartLoops: func(*SharedState, context.Context) { startCalls.Add(1) },
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

func TestPluginReadiness_WarmupCanceled(t *testing.T) {
	pl := &SharedState{}
	pl.BlockedWhileActive = newPodSet("blocked")
	pl.PluginReady.Store(false)

	var persistCalls, activateCalls, startCalls atomic.Int32

	withReadinessEnv(t, ReadinessEnvs{
		Warmup:   250 * time.Millisecond, // should not actually sleep because ctx is already canceled
		GetNodes: func(*SharedState) ([]*v1.Node, error) { return []*v1.Node{node("n1")}, nil },
		IsUsable: func(*v1.Node) bool { return true },
		Persist:  func(*SharedState, context.Context) error { persistCalls.Add(1); return nil },
		Activate: func(*SharedState) { activateCalls.Add(1) },
		StartLoops: func(*SharedState, context.Context) {
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

func TestPluginReadiness_WithWarmup(t *testing.T) {
	pl := &SharedState{}
	pl.BlockedWhileActive = newPodSet("blocked")
	pl.PluginReady.Store(false)

	var persistCalls, activateCalls, startCalls atomic.Int32

	withReadinessEnv(t, ReadinessEnvs{
		Warmup:   0,
		GetNodes: func(*SharedState) ([]*v1.Node, error) { return []*v1.Node{node("n1")}, nil },
		IsUsable: func(*v1.Node) bool { return false },
		Persist:  func(*SharedState, context.Context) error { persistCalls.Add(1); return nil },
		Activate: func(*SharedState) { activateCalls.Add(1) },
		StartLoops: func(*SharedState, context.Context) {
			startCalls.Add(1)
		},
	}, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
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

func TestPluginReadiness_CompletesWithInformer(t *testing.T) {
	pl := &SharedState{}
	pl.BlockedWhileActive = newPodSet("blocked")
	pl.PluginReady.Store(false)

	var persistCalls, activateCalls, startCalls atomic.Int32
	var getNodesCalls atomic.Int32

	inf := newTestPodInformer()
	startInformer(t, inf)

	withReadinessEnv(t, ReadinessEnvs{
		Warmup: 0,
		GetNodes: func(*SharedState) ([]*v1.Node, error) {
			getNodesCalls.Add(1)
			return []*v1.Node{node("n1")}, nil
		},
		IsUsable: func(*v1.Node) bool { return true },
		Persist:  func(*SharedState, context.Context) error { persistCalls.Add(1); return nil },
		Activate: func(*SharedState) { activateCalls.Add(1) },
		StartLoops: func(*SharedState, context.Context) {
			startCalls.Add(1)
		},
	}, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()

		pl.pluginReadiness(ctx, inf)

		if !pl.PluginReady.Load() {
			t.Fatalf("PluginReady = false, want true on success")
		}
		if persistCalls.Load() != 1 || activateCalls.Load() != 1 || startCalls.Load() != 1 {
			t.Fatalf("expected side effects once (persist=%d activate=%d start=%d)",
				persistCalls.Load(), activateCalls.Load(), startCalls.Load(),
			)
		}
		if getNodesCalls.Load() == 0 {
			t.Fatalf("expected getNodesForReadiness to be called at least once")
		}
	})
}

func TestPluginReadiness_CompletesWithWarmup(t *testing.T) {
	pl := &SharedState{}
	pl.BlockedWhileActive = newPodSet("blocked")
	pl.PluginReady.Store(false)

	var persistCalls, activateCalls, startCalls atomic.Int32

	// Use a warmup long enough to make the timing assertion robust.
	warmup := 40 * time.Millisecond

	withReadinessEnv(t, ReadinessEnvs{
		Warmup:   warmup,
		GetNodes: func(*SharedState) ([]*v1.Node, error) { return []*v1.Node{node("n1")}, nil },
		IsUsable: func(*v1.Node) bool { return true },
		Persist:  func(*SharedState, context.Context) error { persistCalls.Add(1); return nil },
		Activate: func(*SharedState) { activateCalls.Add(1) },
		StartLoops: func(*SharedState, context.Context) {
			startCalls.Add(1)
		},
	}, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
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

		// Timers should not fire early; allow a tiny scheduling margin.
		if elapsed < warmup-5*time.Millisecond {
			t.Fatalf("expected warmup delay to be honored, warmup=%v elapsed=%v", warmup, elapsed)
		}
	})
}

// -------------------------
// Test Helpers
// -------------------------

type ReadinessEnvs struct {
	Interval   time.Duration
	Warmup     time.Duration
	GetNodes   func(*SharedState) ([]*v1.Node, error)
	IsUsable   func(*v1.Node) bool
	Persist    func(*SharedState, context.Context) error
	Activate   func(*SharedState)
	StartLoops func(*SharedState, context.Context)
}

func withReadinessEnv(t *testing.T, env ReadinessEnvs, fn func()) {
	t.Helper()

	if env.Interval <= 0 {
		env.Interval = 2 * time.Millisecond
	}
	if env.GetNodes == nil {
		env.GetNodes = func(*SharedState) ([]*v1.Node, error) { return nil, nil }
	}
	if env.IsUsable == nil {
		env.IsUsable = func(*v1.Node) bool { return false }
	}
	if env.Persist == nil {
		env.Persist = func(*SharedState, context.Context) error { return nil }
	}
	if env.Activate == nil {
		env.Activate = func(*SharedState) {}
	}
	if env.StartLoops == nil {
		env.StartLoops = func(*SharedState, context.Context) {}
	}

	withVar(t, &readinessUsableNodeInterval, env.Interval)
	withVar(t, &cacheWarmupDelay, env.Warmup)
	withVar(t, &getNodesForReadiness, env.GetNodes)
	withVar(t, &isNodeUsableForReadiness, env.IsUsable)
	withVar(t, &persistPluginConfigForReadiness, env.Persist)
	withVar(t, &activateBlockedPodsForReadiness, env.Activate)
	withVar(t, &startLoopsForReadiness, env.StartLoops)

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

func startInformer(t *testing.T, inf cache.SharedIndexInformer) chan struct{} {
	t.Helper()
	stopCh := make(chan struct{})
	go inf.Run(stopCh)
	t.Cleanup(func() { close(stopCh) })
	return stopCh
}
