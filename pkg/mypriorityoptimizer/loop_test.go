// loop_test.go
package mypriorityoptimizer

import (
	"context"
	"testing"
	"time"
)

// -------------------------
// Tests
// -------------------------

func TestLoopConfigs(t *testing.T) {
	ctx := context.Background()
	pl := &SharedState{}

	tests := []struct {
		name      string
		setup     func(t *testing.T)
		call      func()
		wantCfg   OptimizeLoopConfig
		wantAfter func(t *testing.T)
	}{
		{
			name: "stable queue defaults",
			setup: func(t *testing.T) {
				withVar(t, &OptimizeStableQueueDelay, time.Duration(0))
				withVar(t, &OptimizeStableQueueCheckInterval, time.Duration(0))
			},
			call: func() { pl.loopStableQueue(ctx) },
			wantCfg: OptimizeLoopConfig{
				Label:            "StableQueueLoop",
				Interval:         250 * time.Millisecond,
				StableQueueDelay: 2 * time.Second,
				CancelOnChange:   true,
			},
		},
		{
			name: "stable queue configured",
			setup: func(t *testing.T) {
				withVar(t, &OptimizeStableQueueDelay, 5*time.Second)
				withVar(t, &OptimizeStableQueueCheckInterval, 123*time.Millisecond)
			},
			call: func() { pl.loopStableQueue(ctx) },
			wantCfg: OptimizeLoopConfig{
				Label:            "StableQueueLoop",
				Interval:         123 * time.Millisecond,
				StableQueueDelay: 5 * time.Second,
				CancelOnChange:   true,
			},
		},
		{
			name: "periodic default interval when too small",
			setup: func(t *testing.T) {
				withVar(t, &OptimizePeriodicInterval, time.Duration(0))
			},
			call: func() { pl.loopPeriodic(ctx) },
			wantCfg: OptimizeLoopConfig{
				Label:            "PeriodicLoop",
				Interval:         2 * time.Second,
				StableQueueDelay: 0,
				CancelOnChange:   false,
			},
			wantAfter: func(t *testing.T) {
				if OptimizePeriodicInterval != 2*time.Second {
					t.Fatalf("OptimizePeriodicInterval=%v, want %v", OptimizePeriodicInterval, 2*time.Second)
				}
			},
		},
		{
			name: "periodic configured",
			setup: func(t *testing.T) {
				withVar(t, &OptimizePeriodicInterval, 5*time.Second)
			},
			call: func() { pl.loopPeriodic(ctx) },
			wantCfg: OptimizeLoopConfig{
				Label:            "PeriodicLoop",
				Interval:         5 * time.Second,
				StableQueueDelay: 0,
				CancelOnChange:   false,
			},
			wantAfter: func(t *testing.T) {
				if OptimizePeriodicInterval != 5*time.Second {
					t.Fatalf("OptimizePeriodicInterval=%v, want %v", OptimizePeriodicInterval, 5*time.Second)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.setup != nil {
				tc.setup(t)
			}
			got := captureLoopCfg(t, tc.call)
			assertCfg(t, got, tc.wantCfg)
			if tc.wantAfter != nil {
				tc.wantAfter(t)
			}
		})
	}
}

// -------------------------
// Test Helpers
// -------------------------

func captureLoopCfg(t *testing.T, call func()) OptimizeLoopConfig {
	t.Helper()

	var got OptimizeLoopConfig
	called := false

	withVar(t, &optimizeBackgroundLoopFunc, func(_ *SharedState, _ context.Context, cfg OptimizeLoopConfig) {
		called = true
		got = cfg
	})

	call()

	if !called {
		t.Fatalf("expected optimizeBackgroundLoopFunc to be called")
	}
	return got
}

func assertCfg(t *testing.T, got, want OptimizeLoopConfig) {
	t.Helper()
	if got.Label != want.Label {
		t.Fatalf("Label=%q, want %q", got.Label, want.Label)
	}
	if got.Interval != want.Interval {
		t.Fatalf("Interval=%v, want %v", got.Interval, want.Interval)
	}
	if got.StableQueueDelay != want.StableQueueDelay {
		t.Fatalf("StableQueueDelay=%v, want %v", got.StableQueueDelay, want.StableQueueDelay)
	}
	if got.CancelOnChange != want.CancelOnChange {
		t.Fatalf("CancelOnChange=%v, want %v", got.CancelOnChange, want.CancelOnChange)
	}
}
