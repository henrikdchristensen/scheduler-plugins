// objects_types.go
// CHECKED
package mypriorityoptimizer

// WorkloadKind represents the kind of workload.
type WorkloadKind int

const (
	// Workload is a ReplicaSet
	wkReplicaSet WorkloadKind = iota
	// Workload is a StatefulSet
	wkStatefulSet
	// Workload is a DaemonSet
	wkDaemonSet
	// Workload is a Job
	wkJob
)

// WorkloadKey is a key to identify a workload.
type WorkloadKey struct {
	// What kind of workload
	Kind WorkloadKind
	// Namespace of the workload
	Namespace string
	// Name of the workload
	Name string
}

// WorkloadStatus represents the status of a workload.
type wkStatus struct {
	// HasLive means at least one live pod (not terminating) for this workload
	HasLive bool
	// HasPending means at least one live pending pod for this workload
	HasPending bool
}
