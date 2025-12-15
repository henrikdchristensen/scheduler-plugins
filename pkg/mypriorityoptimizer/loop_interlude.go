// loop_interlude.go
package mypriorityoptimizer

import (
	"context"
	"time"
)

// -------------------------
// loopInterlude
// -------------------------

// loopInterlude runs optimization in interludes when the pending set is stable.
func (pl *SharedState) loopInterlude(ctx context.Context) {
	delay := OptimizeInterludeDelay
	if delay <= 0 {
		delay = 2 * time.Second
	}
	checkInterval := OptimizeInterludeCheckInterval
	if checkInterval <= 0 {
		checkInterval = 250 * time.Millisecond
	}

	cfg := OptimizeLoopConfig{
		Label:          "InterludeLoop",
		Interval:       checkInterval, // check this often
		InterludeDelay: delay,         // require stability for this long
		CancelOnChange: true,          // cancel if pending set changes
	}
	// delegated through hook
	optimizeBackgroundLoopFunc(pl, ctx, cfg)
}
