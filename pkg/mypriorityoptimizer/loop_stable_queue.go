// loop_stable_queue.go
package mypriorityoptimizer

import (
	"context"
	"time"
)

// -------------------------
// loopStableQueue
// -------------------------

// loopStableQueue runs optimization in stable queue mode when the pending set is stable.
func (pl *SharedState) loopStableQueue(ctx context.Context) {
	delay := OptimizeStableQueueDelay
	if delay <= 0 {
		delay = 2 * time.Second
	}
	checkInterval := OptimizeStableQueueCheckInterval
	if checkInterval <= 0 {
		checkInterval = 250 * time.Millisecond
	}
	cfg := OptimizeLoopConfig{
		Label:            "StableQueueLoop",
		Interval:         checkInterval,                     // check this often
		StableQueueDelay: delay,                             // require stability for this long
		CancelOnChange:   OptimizeStableQueueCancelOnChange, // cancel if pending set changes
	}
	optimizeBackgroundLoopFunc(pl, ctx, cfg)
}
