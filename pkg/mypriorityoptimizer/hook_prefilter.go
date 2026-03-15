// hook_prefilter.go
package mypriorityoptimizer

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// -------------------------
// PreFilter
// -------------------------

// PreFilter is called at the beginning of scheduling cycle. It is used, here,
// to filter the node(s) that the pod can be (tried) scheduled on. If a pod part
// of a plan was scheduled on a wrong node due to workload quotas, it is
// determined in Reserve plugin and will be retried again.
func (pl *SharedState) PreFilter(ctx context.Context, st fwk.CycleState, pending *v1.Pod, nodes []fwk.NodeInfo) (*framework.PreFilterResult, *fwk.Status) {

	stage := "PreFilter"

	ap := pl.getActivePlan()

	// Always allow protected pods and pods if there is no active plan; don't filter any nodes.
	if isPodProtected(pending) || ap == nil {
		return nil, fwk.NewStatus(fwk.Success)
	}

	// Get filtered nodes for the pending pod from the active plan.
	filteredNodes, filterMsg, ok := pl.filterNodes(pending)

	// Convert filteredNodes to slice for logging
	var nodeNames []string
	if filteredNodes != nil {
		nodeNames = filteredNodes.UnsortedList()
	}
	klog.V(MyV).InfoS(msg(stage, "filter decision"),
		"activePlan", ap != nil,
		"pod", mergeNsName(pending.Namespace, pending.Name),
		"nodes", nodeNames,
		"reason", filterMsg,
	)

	switch {
	case ok && filteredNodes == nil: // pending pod allowed on all nodes
		klog.V(MyV).InfoS(msg(stage, InfoAllowPod),
			"pod", klog.KObj(pending),
			"reason", filterMsg,
		)
		return nil, fwk.NewStatus(fwk.Success)

	case ok && filteredNodes.Len() > 0: // pending pod allowed on specific nodes
		klog.V(MyV).InfoS(msg(stage, InfoPinPod),
			"pod", klog.KObj(pending),
			"nodes", nodeNames,
			"reason", filterMsg,
		)
		return &framework.PreFilterResult{NodeNames: filteredNodes}, fwk.NewStatus(fwk.Success)

	default: // not allowed on any node; block the pending pod
		klog.V(MyV).InfoS(msg(stage, InfoBlockPod),
			"pod", klog.KObj(pending),
			"reason", filterMsg,
		)
		pl.BlockedWhileActive.AddPod(pending)
		return nil, fwk.NewStatus(fwk.Unschedulable, msg(stage, filterMsg))
	}
}

// -------------------------
// PreFilter Extensions
// -------------------------

// PreFilterExtensions returns nil (not used).
func (pl *SharedState) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

var _ framework.PreFilterPlugin = &SharedState{}
