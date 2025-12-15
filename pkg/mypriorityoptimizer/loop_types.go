// loop_types.go
package mypriorityoptimizer

import (
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// OptimizeLoopConfig holds configuration for an optimization loop.
type OptimizeLoopConfig struct {
	Label          string        // log label
	Interval       time.Duration // base tick interval
	InterludeDelay time.Duration // 0 => no "idle window"; >0 => require this long of stability
	CancelOnChange bool          // cancel in-flight run if pending set changes
}

// PendingSnapshot represents a snapshot of the current pending pods
// and nodes in the cluster.
type PendingSnapshot struct {
	PendingUIDs  map[types.UID]struct{}
	PendingCount int
	Fingerprint  string     // clusterFingerprint(nodes, pods)
	Pods         []*v1.Pod  // live snapshot (for solver input)
	Nodes        []*v1.Node // live snapshot (for solver input)
}
