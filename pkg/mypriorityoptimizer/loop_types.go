// loop_types.go
package mypriorityoptimizer

import (
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// OptimizeLoopConfig holds configuration for an optimization loop.
type OptimizeLoopConfig struct {
	// Label used for logging.
	Label string
	// Tick interval for the loop.
	Interval time.Duration
	// StableQueueDelay is the duration of the idle window required for stability.
	// 0 means no idle window; >0 means require this long of stability.
	StableQueueDelay time.Duration
	// CancelOnChange indicates whether to cancel in-flight run if pending set changes.
	CancelOnChange bool
}

// PendingSnapshot represents a snapshot of the current pending pods and nodes
// in the cluster.
type PendingSnapshot struct {
	// UIDs of pending pods
	PendingUIDs map[types.UID]struct{}
	// Number of pending pods
	PendingCount int
	// Fingerprint of the cluster state. Meaning a combination of:
	// - cluster state (nodes, pods)
	// - pending set (uids)
	// Used to detect changes in the cluster state.
	Fingerprint string
	// Current live pods
	Pods []*v1.Pod
	// Current live nodes
	Nodes []*v1.Node
}
