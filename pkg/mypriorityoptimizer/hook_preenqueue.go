// hook_preenqueue.go
// CHECKED
package mypriorityoptimizer

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
)

// -------------------------
// PreEnqueue
// -------------------------

// PreEnqueue is called before a pod is enqueued for scheduling.
func (pl *SharedState) PreEnqueue(ctx context.Context, pending *v1.Pod) *fwk.Status {
	const stage = "PreEnqueue"

	// Always allow protected pods (e.g., kube-system).
	if isPodProtected(pending) {
		return fwk.NewStatus(fwk.Success)
	}

	// If plugin is not ready, block the pod. It will be re-queued when ready.
	if !pl.PluginReady.Load() {
		pl.BlockedWhileActive.AddPod(pending)
		klog.V(MyV).Info(msg(stage, "plugin not ready yet; waiting"))
		return fwk.NewStatus(fwk.Pending, msg(stage, "plugin not ready yet; waiting"))
	}

	// If there is an active plan, enforce it.
	if ap := pl.getActivePlan(); ap != nil {
		if !pl.isPodAllowedByPlan(pending) {
			// Plan exists and pod is NOT allowed by the plan -> block.
			pl.BlockedWhileActive.AddPod(pending)
			klog.V(MyV).InfoS(
				msg(stage, InfoActivePlanInProgress+"; "+InfoBlockPod),
				"pod", klog.KObj(pending),
			)
			return fwk.NewStatus(fwk.Pending, msg(stage, InfoActivePlanInProgress+"; "+InfoBlockPod))
		}

		// Plan exists and pod is allowed by the plan -> let it through.
		klog.V(MyV).InfoS(
			msg(stage, "allowed by active plan; pass-through"),
			"pod", klog.KObj(pending),
		)
		return fwk.NewStatus(fwk.Success)
	}

	// No active plan:
	//	- In ManualBlocking mode, block the pod to accumulate work for the solver.
	if isManualBlockingMode() {
		klog.V(MyV).InfoS(msg(stage, InfoPendingPod), "pod", klog.KObj(pending))
		return fwk.NewStatus(fwk.Pending, msg(stage, InfoPendingPod))
	}
	//	- Other modes, just let the pod through.
	klog.V(MyV).InfoS(msg(stage, "pass-through"), "pod", klog.KObj(pending))
	return fwk.NewStatus(fwk.Success)
}
