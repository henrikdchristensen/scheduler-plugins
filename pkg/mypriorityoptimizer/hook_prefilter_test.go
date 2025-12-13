// hook_prefilter_test.go
package mypriorityoptimizer

import (
	"context"
	"testing"

	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// kube-system pods should always be allowed and not constrained by any active plan.
func TestPreFilter_KubeSystemAlwaysAllowed(t *testing.T) {
	pl := &SharedState{}
	pod := newSystemPod("sys-pod")

	res, st := pl.PreFilter(context.Background(), framework.NewCycleState(), pod)
	if st == nil {
		t.Fatalf("PreFilter() returned nil status")
	}
	if st.Code() != framework.Success {
		t.Fatalf("PreFilter() code = %v, want %v", st.Code(), framework.Success)
	}
	if res != nil {
		t.Fatalf("PreFilter() result = %#v, want nil for kube-system pod", res)
	}
}

// When there is no active plan, regular pods should pass through PreFilter unmodified.
func TestPreFilter_NoActivePlan_AllowsPod(t *testing.T) {
	pl := &SharedState{}
	pod := newWorkloadPod("work-pod")

	res, st := pl.PreFilter(context.Background(), framework.NewCycleState(), pod)
	if st == nil {
		t.Fatalf("PreFilter() returned nil status")
	}
	if st.Code() != framework.Success {
		t.Fatalf("PreFilter() code = %v, want %v", st.Code(), framework.Success)
	}
	if res != nil {
		t.Fatalf("PreFilter() result = %#v, want nil when no active plan", res)
	}
}

// PreFilterExtensions should be nil (no additional callbacks).
func TestPreFilterExtensions_IsNil(t *testing.T) {
	pl := &SharedState{}
	if ext := pl.PreFilterExtensions(); ext != nil {
		t.Fatalf("PreFilterExtensions() = %#v, want nil", ext)
	}
}
