// pod_set_types.go
package mypriorityoptimizer

import (
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// PodSet is a thread-safe set of pods.
type PodSet struct {
	// Name of the set
	Name string
	// Mutex to protect the map
	mu sync.RWMutex
	// Map of pod UID to SolverPod
	m map[types.UID]SolverPod
}

// PodSetItem represents an item in the PodSet.
type PodSetItem struct {
	// Pod pointer
	p *v1.Pod
	// Key for identifying the pod
	key SolverPod
}
