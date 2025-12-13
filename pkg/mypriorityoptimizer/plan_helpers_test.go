// plan_helpers_test.go
package mypriorityoptimizer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
)

// -------------------------
// Test Helpers
// -------------------------

func must(t *testing.T, cond bool, msg string, args ...any) {
	t.Helper()
	if !cond {
		t.Fatalf(msg, args...)
	}
}

func mustEq[T any](t *testing.T, got, want T, msg string, args ...any) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf(msg+" (got=%#v want=%#v)", append(args, got, want)...)
	}
}

func mustContains(t *testing.T, s, sub string) {
	t.Helper()
	if !strings.Contains(s, sub) {
		t.Fatalf("want %q to contain %q", s, sub)
	}
}

func keysOf(m map[string]*v1.Pod) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// -------------------------
// Active plan / optimization flow flags
// -------------------------

func TestActivePlanFlags(t *testing.T) {
	pl := &SharedState{}

	must(t, pl.tryEnterActivePlan(), "first tryEnterActivePlan() want true")
	must(t, pl.ActivePlanInProgress.Load(), "ActivePlanInProgress want true after enter")
	must(t, !pl.tryEnterActivePlan(), "second tryEnterActivePlan() want false")
	pl.tryLeaveActivePlan()
	must(t, !pl.ActivePlanInProgress.Load(), "ActivePlanInProgress want false after leave")
}

func TestOptimizationFlowFlags(t *testing.T) {
	pl := &SharedState{}

	must(t, pl.tryEnterOptimizationFlow(), "first tryEnterOptimizationFlow() want true")
	must(t, pl.OptimizationInProgress.Load(), "OptimizationInProgress want true after enter")
	must(t, !pl.tryEnterOptimizationFlow(), "second tryEnterOptimizationFlow() want false")
	pl.tryLeaveOptimizationFlow()
	must(t, !pl.OptimizationInProgress.Load(), "OptimizationInProgress want false after leave")
}

func TestGetAndClearActivePlan(t *testing.T) {
	pl := &SharedState{}
	ap1 := &ActivePlan{ID: "p1"}
	ap2 := &ActivePlan{ID: "p2"}

	must(t, pl.getActivePlan() == nil, "expected nil plan initially")

	pl.ActivePlan.Store(ap1)
	must(t, pl.getActivePlan() == ap1, "expected getActivePlan() to return stored plan")

	must(t, !pl.tryClearActivePlan(ap2), "tryClearActivePlan(wrong ptr) want false")
	must(t, pl.getActivePlan() == ap1, "plan changed after wrong clear")

	must(t, pl.tryClearActivePlan(ap1), "tryClearActivePlan(correct ptr) want true")
	must(t, pl.getActivePlan() == nil, "plan not cleared")

	must(t, !pl.tryClearActivePlan(nil), "tryClearActivePlan(nil) want false")
}

// -------------------------
// tiny plan helper funcs
// -------------------------

func TestPlanPodHelpers(t *testing.T) {
	p := pod("ns", "p1", withUID("u1"), onNode("n1"))
	mustEq(t, toPlanPod(p), SolverPod{UID: p.UID, Namespace: "ns", Name: "p1"}, "toPlanPod mismatch")

	mustEq(t, makePlacement(p, "nx"), SolverPod{UID: p.UID, Namespace: "ns", Name: "p1", Node: "nx"}, "makePlacement mismatch")
	mustEq(t, makeNewPlacement(p, "a", "b"), SolverPod{UID: p.UID, Namespace: "ns", Name: "p1", OldNode: "a", Node: "b"}, "makeNewPlacement mismatch")

	must(t, isPlanPodUnscheduled(""), "isPlanPodUnscheduled(\"\") want true")
	must(t, !isPlanPodUnscheduled("n"), "isPlanPodUnscheduled(\"n\") want false")

	must(t, isPlanPodMove("n1", "n2"), "isPlanPodMove(n1,n2) want true")
	must(t, !isPlanPodMove("", "n2"), "isPlanPodMove(\"\",n2) want false")
	must(t, !isPlanPodMove("n1", "n1"), "isPlanPodMove(n1,n1) want false")

	must(t, isPlanPodPlacementChanged("", "n2"), "isPlanPodPlacementChanged(\"\",n2) want true")
	must(t, isPlanPodPlacementChanged("n1", "n2"), "isPlanPodPlacementChanged(n1,n2) want true")
	must(t, !isPlanPodPlacementChanged("n1", "n1"), "isPlanPodPlacementChanged(n1,n1) want false")

	must(t, isPlanPodNewlyScheduled("", "n1"), "isPlanPodNewlyScheduled(\"\",n1) want true")
	must(t, !isPlanPodNewlyScheduled("n0", "n1"), "isPlanPodNewlyScheduled(n0,n1) want false")
	must(t, !isPlanPodNewlyScheduled("", ""), "isPlanPodNewlyScheduled(\"\",\"\") want false")
}

// -------------------------
// increaseWorkloadQuota + sorting
// -------------------------

func TestIncreaseWorkloadQuota(t *testing.T) {
	wq := WorkloadQuotas{}
	wk := WorkloadKey{}
	increaseWorkloadQuota(wq, wk, "n1")
	increaseWorkloadQuota(wq, wk, "n1")
	increaseWorkloadQuota(wq, wk, "n2")

	key := wk.String()
	must(t, wq[key] != nil, "inner map nil")
	mustEq(t, wq[key]["n1"], int32(2), "n1 count wrong")
	mustEq(t, wq[key]["n2"], int32(1), "n2 count wrong")
}

func TestSortPlacementsByPod(t *testing.T) {
	in := []SolverPod{
		{Namespace: "ns-b", Name: "x"},
		{Namespace: "ns-a", Name: "z"},
		{Namespace: "ns-a", Name: "a"},
	}
	sortPlacementsByPod(in)
	got := []string{in[0].Namespace + "/" + in[0].Name, in[1].Namespace + "/" + in[1].Name, in[2].Namespace + "/" + in[2].Name}
	want := []string{"ns-a/a", "ns-a/z", "ns-b/x"}
	mustEq(t, got, want, "sortPlacementsByPod order wrong")
}

func TestSortPodSetItemsByPriorityAndCreation(t *testing.T) {
	prioLow := int32(1)
	prioHigh := int32(10)

	now := metav1.Now()
	older := metav1.NewTime(now.Add(-time.Minute))

	pLow := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "low", CreationTimestamp: older}, Spec: v1.PodSpec{Priority: &prioLow}}
	pHigh := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "high", CreationTimestamp: now}, Spec: v1.PodSpec{Priority: &prioHigh}}

	items := []PodSetItem{{p: pLow}, {p: pHigh}}
	sortPodSetItemsByPriorityAndCreation(items)
	mustEq(t, items[0].p.Name, "high", "priority sort wrong")

	// same prio, timestamp => older first
	prio := int32(5)
	pOld := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "old", CreationTimestamp: older}, Spec: v1.PodSpec{Priority: &prio}}
	pNew := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new", CreationTimestamp: now}, Spec: v1.PodSpec{Priority: &prio}}
	items = []PodSetItem{{p: pNew}, {p: pOld}}
	sortPodSetItemsByPriorityAndCreation(items)
	mustEq(t, items[0].p.Name, "old", "timestamp sort wrong")

	// zero timestamps => name fallback
	zeroTS := metav1.Time{}
	pA := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "a", CreationTimestamp: zeroTS}, Spec: v1.PodSpec{Priority: &prio}}
	pC := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "c", CreationTimestamp: zeroTS}, Spec: v1.PodSpec{Priority: &prio}}
	items = []PodSetItem{{p: pC}, {p: pA}}
	sortPodSetItemsByPriorityAndCreation(items)
	mustEq(t, items[0].p.Name, "a", "name fallback sort wrong")
}

// -------------------------
// buildPlan
// -------------------------

func TestBuildPlan_NilOutputReturnsEmptyPlan(t *testing.T) {
	pl := &SharedState{}
	plan, err := pl.buildPlan(nil, nil, nil)
	must(t, err == nil, "err=%v", err)
	must(t, plan != nil, "plan nil")
	must(t, len(plan.Evicts) == 0 && len(plan.Moves) == 0 && len(plan.NewPlacements) == 0, "expected empty plan, got=%#v", plan)
}

func TestBuildPlan_CoversBranches(t *testing.T) {
	pl := &SharedState{}

	// pRunEvict: running, evicted
	pRunEvict := pod("ns", "evict", withUID("u-evict"), onNode("n1"))

	// pRunMove: running, move n1->n2
	pRunMove := pod("ns", "move", withUID("u-move"), onNode("n1"))

	// pPendingStandalone: pending -> scheduled => goes to PlacementByName
	pPendingStandalone := pod("ns", "pend", withUID("u-pend"))

	// pOwnedPending: controller-owned pending -> scheduled => goes to WorkloadQuotas
	pOwnedPending := pod("ns", "owned", withUID("u-owned"), withOwner("ReplicaSet", "rs-1"))

	// pUnchanged: running already on n1, solver says n1 => not “changed”, should NOT count quota or placementByName
	pUnchanged := pod("ns", "unchanged", withUID("u-unch"), onNode("n1"), withOwner("ReplicaSet", "rs-1"))

	// pDeleted: should be ignored
	pDeleted := pod("ns", "deleted", withUID("u-del"))
	now := metav1.Now()
	pDeleted.DeletionTimestamp = &now

	out := &SolverOutput{
		Evictions: []SolverPod{
			{UID: pRunEvict.UID},
		},
		Placements: []SolverPod{
			{UID: pRunMove.UID, Node: "n2"},
			{UID: pPendingStandalone.UID, Node: "n1"},
			{UID: pOwnedPending.UID, Node: "n2"},
			{UID: pUnchanged.UID, Node: "n1"},       // unchanged placement
			{UID: pDeleted.UID, Node: "n1"},         // deleted -> ignored
			{UID: "u-missing", Node: "n1"},          // missing from index -> ignored safely
			{UID: pPendingStandalone.UID, Node: ""}, // unscheduled entry -> ignored
		},
	}

	plan, err := pl.buildPlan(out, nil, []*v1.Pod{
		pRunEvict, pRunMove, pPendingStandalone, pOwnedPending, pUnchanged, pDeleted,
	})
	must(t, err == nil, "err=%v", err)
	must(t, plan != nil, "plan nil")

	// Evicts contains evict@n1
	mustEq(t, len(plan.Evicts), 1, "evicts len wrong")
	mustEq(t, plan.Evicts[0].Name, "evict", "evict name wrong")
	mustEq(t, plan.Evicts[0].Node, "n1", "evict node wrong")

	// Moves contains move old=n1 -> n2
	mustEq(t, len(plan.Moves), 1, "moves len wrong")
	mustEq(t, plan.Moves[0].Name, "move", "move name wrong")
	mustEq(t, plan.Moves[0].OldNode, "n1", "move old wrong")
	mustEq(t, plan.Moves[0].Node, "n2", "move new wrong")

	// NewPlacements should include move, pend, owned, unchanged (but not deleted, not unscheduled)
	mustEq(t, len(plan.NewPlacements), 4, "new placements len wrong")

	// PlacementByName: pending standalone should be present, but owned should NOT (it uses workload quotas)
	mustEq(t, plan.PlacementByName[mergeNsName("ns", "pend")], "n1", "standalone pending placementByName wrong")
	_, ownedPresent := plan.PlacementByName[mergeNsName("ns", "owned")]
	must(t, !ownedPresent, "owned pod must not be in PlacementByName")

	// Workload quota: should count only the owned pending that is newly scheduled/changed
	wk, ok := getTopWorkload(pOwnedPending)
	must(t, ok, "expected owned pod to have workload")
	wkStr := wk.String()
	must(t, plan.WorkloadQuotas[wkStr] != nil, "missing workload quota entry")
	mustEq(t, plan.WorkloadQuotas[wkStr]["n2"], int32(1), "workload quota count wrong")

	// Unchanged placement must not contribute additional quota
	if plan.WorkloadQuotas[wkStr]["n1"] != 0 {
		t.Fatalf("unchanged placement should not increase quota; got n1=%d", plan.WorkloadQuotas[wkStr]["n1"])
	}
}

func TestBuildPlan_WithPreemptorNomination(t *testing.T) {
	pl := &SharedState{}
	pre := pod("ns", "pre", withUID("u-pre"))

	out := &SolverOutput{
		Placements: []SolverPod{{UID: pre.UID, Node: "n-pre"}},
	}
	plan, err := pl.buildPlan(out, pre, []*v1.Pod{pre})
	must(t, err == nil, "err=%v", err)
	mustEq(t, plan.NominatedNode, "n-pre", "nominated node wrong")
	mustEq(t, plan.PlacementByName[mergeNsName("ns", "pre")], "n-pre", "preemptor placementByName wrong")
	mustEq(t, len(plan.Moves), 0, "preemptor-only plan should not report moves")
}

// -------------------------
// setActivePlan + buildWorkloadQuotas
// -------------------------

func TestSetActivePlan_NilPlan_NoActivePlanStored(t *testing.T) {
	pl := &SharedState{}
	pl.setActivePlan(nil, "id", nil)
	must(t, pl.getActivePlan() == nil, "expected no plan stored")
}

func TestSetActivePlan_ReplacesOldAndInitializesQuotas(t *testing.T) {
	pl := &SharedState{}

	oldCanceled := false
	pl.ActivePlan.Store(&ActivePlan{ID: "old", Cancel: func() { oldCanceled = true }})

	plan := &Plan{
		PlacementByName: map[string]string{mergeNsName("ns", "p1"): "n1"},
		WorkloadQuotas:  WorkloadQuotas{"wk1": {"n1": 2, "n2": 0}},
	}
	pl.setActivePlan(plan, "new", nil)
	must(t, oldCanceled, "old cancel not called")

	ap := pl.getActivePlan()
	must(t, ap != nil, "active plan nil")
	mustEq(t, ap.ID, "new", "plan id wrong")
	must(t, ap.Ctx != nil && ap.Cancel != nil, "ctx/cancel missing")
	mustEq(t, ap.PlacementByName[mergeNsName("ns", "p1")], "n1", "PlacementByName wrong")
	mustEq(t, ap.WorkloadQuotas["wk1"]["n1"].Load(), int32(2), "quota atomics n1 wrong")
	mustEq(t, ap.WorkloadQuotas["wk1"]["n2"].Load(), int32(0), "quota atomics n2 wrong")
}

func TestBuildWorkloadQuotas_NilAndCounts(t *testing.T) {
	must(t, buildWorkloadQuotas(nil) == nil, "expected nil output for nil input")

	wk := WorkloadQuotas{"wk1": {"n1": 3, "n2": 0, "n3": -1}}
	got := buildWorkloadQuotas(wk)
	must(t, got != nil, "expected non-nil")
	mustEq(t, got["wk1"]["n1"].Load(), int32(3), "n1 wrong")
	mustEq(t, got["wk1"]["n2"].Load(), int32(0), "n2 wrong")
	mustEq(t, got["wk1"]["n3"].Load(), int32(0), "n3 wrong")
}

// -------------------------
// evictTargets
// -------------------------

func TestEvictTargets_UsesHook(t *testing.T) {
	pl := &SharedState{}
	ctx := context.Background()
	targets := []*v1.Pod{pod("ns", "p1"), pod("ns", "p2")}

	var called bool
	withVar(t, &evictTargetsHook, func(pl *SharedState, ctx context.Context, targets []*v1.Pod) error {
		called = true
		must(t, pl == pl, "pl mismatch")
		must(t, ctx == ctx, "ctx mismatch")
		mustEq(t, targets, targets, "targets mismatch")
		return nil
	})

	must(t, pl.evictTargets(ctx, targets) == nil, "expected nil error")
	must(t, called, "hook not called")
}

func TestEvictTargets_NotFoundIgnored_NonNotFoundPropagates(t *testing.T) {
	pl := &SharedState{}
	ctx := context.Background()

	p1 := pod("ns", "p1")
	p2 := pod("ns", "p2")

	withEvictHook(func(_ *SharedState, _ context.Context, pod *v1.Pod, _ *policyv1.Eviction) error {
		if pod.Name == "p2" {
			return apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "pods"}, pod.Name)
		}
		return nil
	}, func() {
		must(t, pl.evictTargets(ctx, []*v1.Pod{p1, p2}) == nil, "expected nil error")
	})

	wantErr := errors.New("boom")
	withEvictHook(func(_ *SharedState, _ context.Context, _ *v1.Pod, _ *policyv1.Eviction) error {
		return wantErr
	}, func() {
		err := pl.evictTargets(ctx, []*v1.Pod{p1})
		must(t, err != nil, "expected error")
		mustContains(t, err.Error(), "boom")
	})
}

// -------------------------
// waitPodsGone
// -------------------------

func TestWaitPodsGone_UsesHookWhenNonEmpty(t *testing.T) {
	pl := &SharedState{}
	ctx := context.Background()
	p := pod("ns", "p")

	var called bool
	withVar(t, &waitPodsGoneHook, func(pl *SharedState, ctx context.Context, pods []*v1.Pod) error {
		called = true
		must(t, pl == pl, "pl mismatch")
		must(t, ctx == ctx, "ctx mismatch")
		mustEq(t, pods, []*v1.Pod{p}, "pods mismatch")
		return nil
	})

	must(t, pl.waitPodsGone(ctx, []*v1.Pod{p}) == nil, "expected nil error")
	must(t, called, "hook not called")
}

func TestWaitPodsGone_EmptyOrNilPodFastPath(t *testing.T) {
	pl := &SharedState{}
	ctx := context.Background()
	must(t, pl.waitPodsGone(ctx, nil) == nil, "expected nil")
	must(t, pl.waitPodsGone(ctx, []*v1.Pod{nil}) == nil, "expected nil")
}

func TestWaitPodsGone_NotFound_UIDChange_Terminating(t *testing.T) {
	pl := &SharedState{}
	ctx := context.Background()

	orig := pod("ns", "p", withUID("u1"))
	changed := pod("ns", "p", withUID("u2")) // same name, diff UID

	now := metav1.Now()
	term := pod("ns", "t", withUID("u3"))
	term.DeletionTimestamp = &now

	// Case 1: NotFound via empty store
	withPodLister(&fakePodLister{store: map[string]map[string]*v1.Pod{}}, func() {
		must(t, pl.waitPodsGone(ctx, []*v1.Pod{orig}) == nil, "expected nil (NotFound)")
	})

	// Case 2/3: UID changed + terminating treated as gone
	withPodLister(&fakePodLister{store: storeFromPods(changed, term)}, func() {
		must(t, pl.waitPodsGone(ctx, []*v1.Pod{orig}) == nil, "expected nil (uid changed)")
		must(t, pl.waitPodsGone(ctx, []*v1.Pod{term}) == nil, "expected nil (terminating)")
	})

	// Case 4: transient lister error should keep polling; we make ctx cancel quickly and ensure it returns ctx error.
	ctx2, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	withPodLister(&fakePodLister{
		store: storeFromPods(orig),
		err:   errors.New("transient"),
	}, func() {
		err := pl.waitPodsGone(ctx2, []*v1.Pod{orig})
		must(t, err != nil, "expected ctx error due to timeout")
	})
}

// -------------------------
// activatePods (blocked set)
// -------------------------

func TestActivatePods_NilOrEmptySetDoesNothing(t *testing.T) {
	pl := &SharedState{}

	var called bool
	withVar(t, &activatePlannedPodsHook, func(_ *SharedState, _ map[string]*v1.Pod) { called = true })

	tried := pl.activatePods(nil, true, -1)
	mustEq(t, tried, []types.UID(nil), "tried mismatch for nil set")
	must(t, !called, "activatePods should not be called for nil set")

	set := newPodSet("blocked")
	tried = pl.activatePods(set, false, -1)
	mustEq(t, tried, []types.UID(nil), "tried mismatch for empty set")
	must(t, !called, "activatePods should not be called for empty set")
}

func TestActivatePods_SortsLimitsAndRemovesActivated(t *testing.T) {
	pl := &SharedState{}
	set := newPodSet("blocked")

	now := metav1.Now()
	older := metav1.NewTime(now.Add(-time.Minute))

	prioLow := int32(1)
	prioHigh := int32(10)

	pHigh := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "p-high", UID: types.UID("uid-high"), CreationTimestamp: now}, Spec: v1.PodSpec{Priority: &prioHigh}}
	pOld := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "p-old", UID: types.UID("uid-old"), CreationTimestamp: older}, Spec: v1.PodSpec{Priority: &prioLow}}
	pKeep := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "p-keep", UID: types.UID("uid-keep"), CreationTimestamp: now}, Spec: v1.PodSpec{Priority: &prioLow}}

	set.AddPod(pOld)
	set.AddPod(pHigh)
	set.AddPod(pKeep)

	var got map[string]*v1.Pod
	withVar(t, &activatePods, func(_ *SharedState, toAct map[string]*v1.Pod) { got = toAct })

	withPodLister(&fakePodLister{store: storeFromPods(pOld, pHigh, pKeep)}, func() {
		tried := pl.activatePods(set, true, 2)
		mustEq(t, tried, []types.UID{pHigh.UID, pOld.UID}, "tried order mismatch")
		must(t, got != nil, "expected activation set")
		mustEq(t, keysOf(got), []string{"default/p-high", "default/p-old"}, "activation keys mismatch")
		mustEq(t, set.Size(), 1, "set size after removeActivated mismatch")
	})
}

func TestActivatePods_PruneNotFound_ConservativeOnListerErr(t *testing.T) {
	pl := &SharedState{}
	set := newPodSet("blocked")

	prio := int32(1)
	pOK := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "p-ok", UID: types.UID("uid-ok")}, Spec: v1.PodSpec{Priority: &prio}}
	pGone := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "p-gone", UID: types.UID("uid-gone")}, Spec: v1.PodSpec{Priority: &prio}}

	set.AddPod(pOK)
	set.AddPod(pGone)

	var got map[string]*v1.Pod
	withVar(t, &activatePods, func(_ *SharedState, toAct map[string]*v1.Pod) { got = toAct })

	// NotFound pruning for p-gone
	withPodLister(&fakePodLister{store: storeFromPods(pOK)}, func() {
		tried := pl.activatePods(set, false, -1)
		mustEq(t, tried, []types.UID{pOK.UID}, "tried mismatch")
		mustEq(t, keysOf(got), []string{"default/p-ok"}, "activation keys mismatch")
		mustEq(t, set.Size(), 1, "set should still contain p-ok only")
	})

	// lister error => no activation, no tried, entry kept
	set2 := newPodSet("blocked")
	set2.AddPod(pOK)
	withPodLister(&fakePodLister{store: storeFromPods(pOK), err: errors.New("boom")}, func() {
		got = nil
		tried := pl.activatePods(set2, false, -1)
		mustEq(t, tried, []types.UID(nil), "tried should be empty on lister err")
		must(t, got == nil, "activation should not run on lister err")
		mustEq(t, set2.Size(), 1, "conservative keep on lister err")
	})
}

// -------------------------
// activatePlannedPods
// -------------------------

func TestActivatePlannedPods_HookAndGlobalPaths(t *testing.T) {
	pl := &SharedState{}

	uidAllowed := types.UID("u-allowed")
	uidMove := types.UID("u-move")

	plan := &Plan{
		NewPlacements: []SolverPod{
			{UID: uidAllowed, OldNode: "", Node: "n1"}, // pending->node => activate
			{UID: uidMove, OldNode: "n0", Node: "n2"},  // move => not activate
		},
	}

	pending := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "p-allowed", UID: uidAllowed}}
	running := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "p-running", UID: uidMove}, Spec: v1.PodSpec{NodeName: "n0"}}
	deleted := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "p-del", UID: uidAllowed, DeletionTimestamp: &metav1.Time{Time: time.Now()}}}

	// Hook path
	var hookGot map[string]*v1.Pod
	withVar(t, &activatePods, func(_ *SharedState, toAct map[string]*v1.Pod) {
		hookGot = toAct
	})
	pl.activatePlannedPods(plan, []*v1.Pod{pending, running, deleted})
	mustEq(t, keysOf(hookGot), []string{"ns/p-allowed"}, "hook activation mismatch")

	// Global path
	activatePlannedPodsHook = nil // explicitly drop hook for this subcase
	var globalGot map[string]*v1.Pod
	withVar(t, &activatePlannedPodsHook, func(_ *SharedState, toAct map[string]*v1.Pod) { globalGot = toAct })
	pl.activatePlannedPods(plan, []*v1.Pod{pending, running})
	mustEq(t, keysOf(globalGot), []string{"ns/p-allowed"}, "global activation mismatch")
}

func TestActivatePlannedPods_Guards(t *testing.T) {
	pl := &SharedState{}
	pl.activatePlannedPods(nil, nil)
	pl.activatePlannedPods(&Plan{}, nil)
	pl.activatePlannedPods(&Plan{NewPlacements: nil}, []*v1.Pod{})
}

func TestActivatePlannedPods_NoNewlyScheduledPlacements_NoActivation(t *testing.T) {
	pl := &SharedState{}

	// Only a move => allow-set becomes empty => early return branch.
	plan := &Plan{
		NewPlacements: []SolverPod{
			{UID: types.UID("u1"), OldNode: "n1", Node: "n2"},
		},
	}

	called := false
	withVar(t, &activatePlannedPodsHook, func(_ *SharedState, _ map[string]*v1.Pod) { called = true })

	pl.activatePlannedPods(plan, []*v1.Pod{pod("ns", "p", withUID("u1"))})
	must(t, !called, "expected no activation when no newly-scheduled placements exist")
}

func TestActivatePlannedPods_AllowButNoMatchingPending_NoActivation(t *testing.T) {
	pl := &SharedState{}

	uid := types.UID("u-allow")
	plan := &Plan{
		NewPlacements: []SolverPod{
			{UID: uid, OldNode: "", Node: "n1"}, // newly scheduled
		},
	}

	// Present but not eligible: one assigned and one deleted.
	pAssigned := pod("ns", "p", withUID(string(uid)), onNode("n0"))
	now := metav1.Now()
	pDeleted := pod("ns", "p2", withUID(string(uid)))
	pDeleted.DeletionTimestamp = &now

	called := false
	withVar(t, &activatePlannedPodsHook, func(_ *SharedState, _ map[string]*v1.Pod) { called = true })

	pl.activatePlannedPods(plan, []*v1.Pod{pAssigned, pDeleted})
	must(t, !called, "expected no activation when no matching *pending* pods exist")
}

// -------------------------
// isPlanCompleted
// -------------------------

func TestIsPlanCompleted_UsesHook(t *testing.T) {
	pl := &SharedState{}
	ap := &ActivePlan{ID: PlanConfigMapNamePrefix + "1"}

	var called bool
	withVar(t, &isPlanCompletedHook, func(hpl *SharedState, hap *ActivePlan) (bool, error) {
		called = true
		must(t, hpl == pl, "pl mismatch")
		must(t, hap == ap, "ap mismatch")
		return true, nil
	})

	ok, err := pl.isPlanCompleted(ap)
	must(t, err == nil, "err=%v", err)
	must(t, ok, "expected true from hook")
	must(t, called, "hook not called")
}

func TestIsPlanCompleted_NilPlan(t *testing.T) {
	pl := &SharedState{}
	ok, err := pl.isPlanCompleted(nil)
	must(t, err == nil, "err=%v", err)
	must(t, !ok, "expected false for nil plan")
}

func TestIsPlanCompleted_ErrorPaths(t *testing.T) {
	pl := &SharedState{}

	// invalid ns/name key => splitNsName error
	ap := &ActivePlan{
		ID:              "x",
		PlacementByName: map[string]string{"not-a-nsname": "n1"},
	}
	withPodLister(&fakePodLister{store: map[string]map[string]*v1.Pod{}}, func() {
		ok, err := pl.isPlanCompleted(ap)
		must(t, err != nil, "expected error")
		must(t, !ok, "expected not completed on error")
	})

	// pinned Get() returns real error => propagate
	ap = &ActivePlan{
		ID:              "x",
		PlacementByName: map[string]string{mergeNsName("ns", "p"): "n1"},
	}
	withPodLister(&fakePodLister{store: map[string]map[string]*v1.Pod{"ns": {"p": pod("ns", "p")}}, errPerKey: map[string]error{"ns/p": errors.New("boom")}}, func() {
		ok, err := pl.isPlanCompleted(ap)
		must(t, err != nil, "expected error")
		must(t, !ok, "expected not completed")
	})
}

func TestIsPlanCompleted_CoreCases(t *testing.T) {
	pl := &SharedState{}

	t.Run("pinned pod gone satisfied when no quotas", func(t *testing.T) {
		ap := &ActivePlan{
			ID:              "x",
			PlacementByName: map[string]string{mergeNsName("ns", "p-gone"): "n1"},
			WorkloadQuotas:  WorkloadQuotasAtomics{},
		}
		withPodLister(&fakePodLister{store: map[string]map[string]*v1.Pod{}}, func() {
			ok, err := pl.isPlanCompleted(ap)
			must(t, err == nil, "err=%v", err)
			must(t, ok, "expected completed")
		})
	})

	t.Run("pinned pod wrong node => not completed", func(t *testing.T) {
		ap := &ActivePlan{
			ID:              "x",
			PlacementByName: map[string]string{mergeNsName("ns", "p"): "n-expected"},
		}
		p := pod("ns", "p", onNode("n-other"))
		withPodLister(&fakePodLister{store: storeFromPods(p)}, func() {
			ok, err := pl.isPlanCompleted(ap)
			must(t, err == nil, "err=%v", err)
			must(t, !ok, "expected not completed")
		})
	})

	t.Run("pinned pod terminating treated as satisfied", func(t *testing.T) {
		ap := &ActivePlan{
			ID:              "x",
			PlacementByName: map[string]string{mergeNsName("ns", "p"): "n1"},
			WorkloadQuotas:  WorkloadQuotasAtomics{},
		}
		p := pod("ns", "p", onNode("n1"))
		now := metav1.Now()
		p.DeletionTimestamp = &now
		withPodLister(&fakePodLister{store: storeFromPods(p)}, func() {
			ok, err := pl.isPlanCompleted(ap)
			must(t, err == nil, "err=%v", err)
			must(t, ok, "expected completed")
		})
	})

	t.Run("workload deleted/scale-to-zero ignores remaining", func(t *testing.T) {
		d := pod("ns", "dummy", withOwner("ReplicaSet", "rs-1"))
		wk, owned := getTopWorkload(d)
		must(t, owned, "expected owned")
		wkStr := wk.String()

		ap := &ActivePlan{
			ID:             "x",
			WorkloadQuotas: buildWorkloadQuotas(WorkloadQuotas{wkStr: {"n1": 3}}),
		}
		withPodLister(&fakePodLister{store: map[string]map[string]*v1.Pod{}}, func() {
			ok, err := pl.isPlanCompleted(ap)
			must(t, err == nil, "err=%v", err)
			must(t, ok, "expected completed")
		})
	})

	t.Run("workload has pending + remaining => not completed", func(t *testing.T) {
		pRun := pod("ns", "run", onNode("n1"), withOwner("ReplicaSet", "rs-1"))
		pPend := pod("ns", "pend", withOwner("ReplicaSet", "rs-1"))
		wk, _ := getTopWorkload(pRun)
		wkStr := wk.String()

		ap := &ActivePlan{
			ID:             "x",
			WorkloadQuotas: buildWorkloadQuotas(WorkloadQuotas{wkStr: {"n1": 1}}),
		}
		withPodLister(&fakePodLister{store: storeFromPods(pRun, pPend)}, func() {
			ok, err := pl.isPlanCompleted(ap)
			must(t, err == nil, "err=%v", err)
			must(t, !ok, "expected not completed")
		})
	})

	t.Run("workload live but no pending => remaining treated satisfied", func(t *testing.T) {
		pRun := pod("ns", "run", onNode("n1"), withOwner("ReplicaSet", "rs-1"))
		wk, _ := getTopWorkload(pRun)
		wkStr := wk.String()

		ap := &ActivePlan{
			ID:             "x",
			WorkloadQuotas: buildWorkloadQuotas(WorkloadQuotas{wkStr: {"n1": 2}}),
		}
		withPodLister(&fakePodLister{store: storeFromPods(pRun)}, func() {
			ok, err := pl.isPlanCompleted(ap)
			must(t, err == nil, "err=%v", err)
			must(t, ok, "expected completed")
		})
	})
}

func TestIsPlanCompleted_PinnedCorrectNode_Succeeds(t *testing.T) {
	pl := &SharedState{}

	ap := &ActivePlan{
		ID:              "x",
		PlacementByName: map[string]string{mergeNsName("ns", "p"): "n1"},
		WorkloadQuotas:  WorkloadQuotasAtomics{}, // none
	}
	p := pod("ns", "p", onNode("n1"))

	withPodLister(&fakePodLister{store: storeFromPods(p)}, func() {
		ok, err := pl.isPlanCompleted(ap)
		must(t, err == nil, "err=%v", err)
		must(t, ok, "expected completed")
	})
}

func TestIsPlanCompleted_QuotaZero_IsSatisfiedBranch(t *testing.T) {
	pl := &SharedState{}

	ap := &ActivePlan{
		ID:             "x",
		WorkloadQuotas: buildWorkloadQuotas(WorkloadQuotas{"wk": {"n1": 0}}), // totalRemaining <= 0 => satisfied
	}

	withPodLister(&fakePodLister{store: map[string]map[string]*v1.Pod{}}, func() {
		ok, err := pl.isPlanCompleted(ap)
		must(t, err == nil, "err=%v", err)
		must(t, ok, "expected completed")
	})
}

func TestIsPlanCompleted_GetPodsErrorPropagates(t *testing.T) {
	pl := &SharedState{}
	ap := &ActivePlan{ID: "x"}

	withPodLister(&fakePodLister{err: errors.New("boom")}, func() {
		ok, err := pl.isPlanCompleted(ap)
		must(t, !ok, "expected not completed on error")
		must(t, err != nil, "expected error")
	})
}

// -------------------------
// onPlanCompleted
// -------------------------

func TestOnPlanCompleted_HookAndDefaultPaths(t *testing.T) {
	pl := &SharedState{}
	pl.ActivePlanInProgress.Store(true)

	cancelled := false
	ap := &ActivePlan{
		ID: PlanConfigMapNamePrefix + "cm-123",
		Cancel: func() {
			cancelled = true
		},
	}
	pl.ActivePlan.Store(ap)

	t.Run("hook path", func(t *testing.T) {
		pl.ActivePlanInProgress.Store(true)
		pl.ActivePlan.Store(ap)
		cancelled = false

		var called bool
		withVar(t, &onPlanCompletedHook, func(hpl *SharedState, status PlanStatus, hap *ActivePlan) {
			called = true
			must(t, hpl == pl, "pl mismatch")
			must(t, status == PlanStatusCompleted, "status mismatch")
			must(t, hap == ap, "ap mismatch")
			must(t, !hpl.ActivePlanInProgress.Load(), "active flag should be false inside hook")
		})

		ok := pl.onPlanCompleted(PlanStatusCompleted)
		must(t, ok, "expected ok")
		must(t, called, "hook not called")
		must(t, cancelled, "cancel not called")
		must(t, pl.getActivePlan() == nil, "plan not cleared")
	})

	t.Run("default path calls markPlanStatus hook", func(t *testing.T) {
		// ensure no onPlanCompletedHook
		onPlanCompletedHook = nil

		pl.ActivePlanInProgress.Store(true)
		pl.ActivePlan.Store(ap)
		cancelled = false

		var statusCalled bool
		withVar(t, &markPlanStatusToConfigMapHook, func(hpl *SharedState, _ context.Context, planCM string, status PlanStatus) bool {
			statusCalled = true
			must(t, hpl == pl, "pl mismatch")
			mustEq(t, planCM, ap.ID, "planCM mismatch")
			mustEq(t, status, PlanStatusCompleted, "status mismatch")
			return true
		})

		ok := pl.onPlanCompleted(PlanStatusCompleted)
		must(t, ok, "expected ok")
		must(t, statusCalled, "markPlanStatus hook not called")
		must(t, cancelled, "cancel not called")
		must(t, pl.getActivePlan() == nil, "plan not cleared")
		must(t, !pl.ActivePlanInProgress.Load(), "active flag not cleared")
	})

	t.Run("no active plan => false", func(t *testing.T) {
		pl2 := &SharedState{}
		must(t, !pl2.onPlanCompleted(PlanStatusCompleted), "expected false")
	})
}

func TestOnPlanCompleted_CASFail_ReturnsFalseAndKeepsState(t *testing.T) {
	pl := &SharedState{}
	pl.ActivePlanInProgress.Store(true)

	ap := &ActivePlan{ID: "cm-1"}
	pl.ActivePlan.Store(ap)

	withVar(t, &activePlanCompareAndSwap, func(_ *SharedState, _, _ *ActivePlan) bool {
		return false // simulate someone else winning the race
	})

	ok := pl.onPlanCompleted(PlanStatusCompleted)
	must(t, !ok, "want ok=false when CAS fails")
	must(t, pl.getActivePlan() == ap, "active plan should remain when CAS fails")
	must(t, pl.ActivePlanInProgress.Load(), "active flag should remain true when CAS fails")
}

// -------------------------
// isPodAllowedByPlan + filterNodes
// -------------------------

func TestIsPodAllowedByPlan_AndFilterNodes(t *testing.T) {
	pl := &SharedState{}

	// no active plan
	must(t, !pl.isPodAllowedByPlan(pod("ns", "p")), "expected false (no plan)")
	nodes, reason, ok := pl.filterNodes(pod("ns", "p"))
	must(t, ok, "expected ok=true when no plan")
	must(t, nodes == nil, "expected nil nodes when no plan")
	mustEq(t, reason, InfoNoActivePlan, "reason mismatch")

	// pinned standalone
	ap := &ActivePlan{
		ID:              "plan",
		PlacementByName: map[string]string{mergeNsName("ns", "p1"): "n1", mergeNsName("ns", "pAny"): ""},
	}
	pl.ActivePlan.Store(ap)

	must(t, pl.isPodAllowedByPlan(pod("ns", "p1")), "expected allowed (pinned)")
	s, _, ok := pl.filterNodes(pod("ns", "p1"))
	must(t, ok, "expected ok")
	must(t, s.Len() == 1 && s.Has("n1"), "expected only n1")

	s, _, ok = pl.filterNodes(pod("ns", "pAny"))
	must(t, ok, "expected ok")
	must(t, s == nil, "expected nil set for empty pinned node")

	// workload quotas
	pOwned := pod("ns", "owned", withOwner("ReplicaSet", "rs-1"))
	wk, _ := getTopWorkload(pOwned)
	wkStr := wk.String()
	ap.WorkloadQuotas = buildWorkloadQuotas(WorkloadQuotas{wkStr: {"n1": 1, "n2": 0}})

	must(t, pl.isPodAllowedByPlan(pOwned), "expected allowed (any node has quota)")

	allowed, _, ok := pl.filterNodes(pOwned)
	must(t, ok, "expected ok")
	must(t, allowed.Has("n1") && !allowed.Has("n2"), "expected only n1 allowed")

	// workload missing from plan => block
	pOther := pod("ns", "other", withOwner("ReplicaSet", "rs-other"))
	_, _, ok = pl.filterNodes(pOther)
	must(t, !ok, "expected block for workload not in plan")

	// node-selected path (NodeName set)
	pOwnedOnN1 := pod("ns", "owned-n1", withOwner("ReplicaSet", "rs-1"), onNode("n1"))
	must(t, pl.isPodAllowedByPlan(pOwnedOnN1), "expected allowed when bound to node with remaining quota")

	pOwnedOnN2 := pod("ns", "owned-n2", withOwner("ReplicaSet", "rs-1"), onNode("n2"))
	must(t, !pl.isPodAllowedByPlan(pOwnedOnN2), "expected blocked when bound to node with NO remaining quota")

	// quotas exhausted => block
	ap.WorkloadQuotas = buildWorkloadQuotas(WorkloadQuotas{wkStr: {"n1": 0, "n2": 0}})
	_, reason, ok = pl.filterNodes(pOwned)
	must(t, !ok, "expected block when quotas exhausted")
	mustContains(t, reason, "quotas exhausted")

}

// -------------------------
// computePlanPodCounts
// -------------------------

func TestComputePlanPodCounts(t *testing.T) {
	// nil output
	a, b, c := computePlanPodCounts(nil, nil)
	mustEq(t, []int{a, b, c}, []int{0, 0, 0}, "nil out mismatch")

	now := metav1.Now()
	run1 := pod("ns", "run1", onNode("n1"))
	run2 := pod("ns", "run2", onNode("n2"))
	pend := pod("ns", "pend")
	del := pod("ns", "del", onNode("n1"))
	del.DeletionTimestamp = &now

	out := &SolverOutput{
		Placements: []SolverPod{
			{UID: pend.UID, Node: "n1"},
			{UID: "u-unknown", Node: "n1"},
			{UID: pend.UID, Node: ""}, // ignored
		},
		Evictions: []SolverPod{
			{UID: run1.UID},
			{UID: "u-missing"},
		},
	}
	pendingScheduled, pre, post := computePlanPodCounts(out, []*v1.Pod{run1, run2, pend, del})
	mustEq(t, pendingScheduled, 1, "pendingScheduled wrong")
	mustEq(t, pre, 2, "pre wrong")
	mustEq(t, post, 2, "post wrong")

	// clamp non-negative
	out2 := &SolverOutput{Evictions: []SolverPod{{UID: "u-missing"}}}
	_, pre2, post2 := computePlanPodCounts(out2, []*v1.Pod{pod("ns", "only-pend")})
	mustEq(t, pre2, 0, "pre2 wrong")
	mustEq(t, post2, 0, "post2 wrong")
}

func TestComputePlanPodCounts_IgnoresPendingEvictions_AndRunningPlacements(t *testing.T) {
	run := pod("ns", "run", withUID("u-run"), onNode("n1"))
	pend := pod("ns", "pend", withUID("u-pend")) // pending (no node)

	out := &SolverOutput{
		// Evicting a pending pod must NOT reduce runningBefore.
		Evictions: []SolverPod{{UID: pend.UID}},
		// A placement for an already-running pod must NOT count as pendingScheduled.
		Placements: []SolverPod{
			{UID: run.UID, Node: "n2"},
			{UID: pend.UID, Node: "n1"},
		},
	}

	pendingScheduled, pre, post := computePlanPodCounts(out, []*v1.Pod{run, pend})
	mustEq(t, pendingScheduled, 1, "pendingScheduled wrong")
	mustEq(t, pre, 1, "runningBefore wrong")
	mustEq(t, post, 2, "runningAfter wrong")
}

func TestComputePlanPodCounts_ClampsNegative(t *testing.T) {
	run := pod("ns", "run", withUID("u-run"), onNode("n1"))

	out := &SolverOutput{
		// Duplicate eviction entries => evictedRunning > runningBefore
		Evictions: []SolverPod{{UID: run.UID}, {UID: run.UID}},
	}
	_, pre, post := computePlanPodCounts(out, []*v1.Pod{run})

	mustEq(t, pre, 1, "runningBefore wrong")
	mustEq(t, post, 0, "runningAfter should clamp to 0")
}

// -------------------------
// exportPlanToConfigMap + setPlanStatusInConfigMap
// (kept mostly like your original, but with cleaner hook usage)
// -------------------------

func TestExportPlanToConfigMap_UsesHook(t *testing.T) {
	pl := &SharedState{}
	ctx := context.Background()
	sp := &StoredPlan{}
	name := PlanConfigMapNamePrefix + "cm"

	var called bool
	withVar(t, &exportPlanToConfigMapHook, func(hpl *SharedState, hctx context.Context, hname string, hsp *StoredPlan) error {
		called = true
		must(t, hpl == pl, "pl mismatch")
		must(t, hctx == ctx, "ctx mismatch")
		mustEq(t, hname, name, "name mismatch")
		must(t, hsp == sp, "sp mismatch")
		return nil
	})

	must(t, pl.exportPlanToConfigMap(ctx, name, sp) == nil, "expected nil")
	must(t, called, "hook not called")
}

func TestExportPlanToConfigMap_DefaultPath_CreatesConfigMap(t *testing.T) {
	// Ensure hook is disabled to exercise real code.
	orig := exportPlanToConfigMapHook
	exportPlanToConfigMapHook = nil
	t.Cleanup(func() { exportPlanToConfigMapHook = orig })

	ctx := context.Background()
	pl, cleanup := newSharedStateWithConfigMapInformer(t /* no objects */)
	defer cleanup()

	name := fmt.Sprintf("%s%s", PlanConfigMapNamePrefix, "unit-default")
	sp := &StoredPlan{
		PluginVersion: "test",
		GeneratedAt:   time.Unix(1, 0).UTC(),
		PlanStatus:    PlanStatusActive,
		Plan:          &Plan{},
	}

	err := pl.exportPlanToConfigMap(ctx, name, sp)
	must(t, err == nil, "export err=%v", err)

	gotCM, err := pl.Client.CoreV1().ConfigMaps(SystemNamespace).Get(ctx, name, metav1.GetOptions{})
	must(t, err == nil, "get err=%v", err)
	must(t, gotCM.Labels[PlanConfigMapLabelKey] == "true", "expected plan label")

	raw := gotCM.Data[PlanConfigMapLabelKey+".json"]
	must(t, raw != "", "expected json data")
	var got StoredPlan
	must(t, json.Unmarshal([]byte(raw), &got) == nil, "unmarshal failed")
	mustEq(t, got.PluginVersion, "test", "plugin version mismatch")
}

func newSharedStateWithConfigMapInformer(t *testing.T, objects ...runtime.Object) (*SharedState, func()) {
	t.Helper()

	client := fake.NewSimpleClientset(objects...)
	factory := informers.NewSharedInformerFactory(client, 0)
	cmInformer := factory.Core().V1().ConfigMaps().Informer()

	stopCh := make(chan struct{})
	factory.Start(stopCh)
	if ok := cache.WaitForCacheSync(stopCh, cmInformer.HasSynced); !ok {
		close(stopCh)
		t.Fatalf("ConfigMap informer failed to sync")
	}

	h := &fakeHandle{client: client, factory: factory}
	pl := &SharedState{Client: client, Handle: h}

	return pl, func() { close(stopCh) }
}

func TestSetPlanStatusInConfigMap_UsesHook(t *testing.T) {
	pl := &SharedState{}
	ctx := context.Background()

	var called bool
	withVar(t, &markPlanStatusToConfigMapHook, func(hpl *SharedState, hctx context.Context, planCM string, status PlanStatus) bool {
		called = true
		must(t, hpl == pl, "pl mismatch")
		must(t, hctx == ctx, "ctx mismatch")
		mustEq(t, planCM, "cm-name", "cm mismatch")
		mustEq(t, status, PlanStatusFailed, "status mismatch")
		return true
	})

	pl.setPlanStatusInConfigMap(ctx, "cm-name", PlanStatusFailed)
	must(t, called, "hook not called")
}

func TestSetPlanStatusInConfigMap_Default_SetsAndStickyAndBadJson(t *testing.T) {
	// disable hook
	orig := markPlanStatusToConfigMapHook
	markPlanStatusToConfigMapHook = nil
	t.Cleanup(func() { markPlanStatusToConfigMapHook = orig })

	ctx := context.Background()

	t.Run("sets status + CompletedAt", func(t *testing.T) {
		name := "plan-status-cm"
		sp := &StoredPlan{PluginVersion: "test", GeneratedAt: time.Unix(1, 0).UTC(), PlanStatus: PlanStatusActive, Plan: &Plan{}}
		b, _ := json.MarshalIndent(sp, "", "  ")
		cm := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Namespace: SystemNamespace, Name: name, Labels: map[string]string{PlanConfigMapLabelKey: "true"}},
			Data:       map[string]string{PlanConfigMapLabelKey + ".json": string(b)},
		}

		pl, cleanup := newSharedStateWithConfigMapInformer(t, cm)
		defer cleanup()

		pl.setPlanStatusInConfigMap(ctx, name, PlanStatusCompleted)

		gotCM, err := pl.Client.CoreV1().ConfigMaps(SystemNamespace).Get(ctx, name, metav1.GetOptions{})
		must(t, err == nil, "get err=%v", err)

		var got StoredPlan
		must(t, json.Unmarshal([]byte(gotCM.Data[PlanConfigMapLabelKey+".json"]), &got) == nil, "unmarshal failed")
		mustEq(t, got.PlanStatus, PlanStatusCompleted, "status wrong")
		must(t, !got.CompletedAt.IsZero(), "CompletedAt should be non-zero")
	})

	t.Run("final is sticky", func(t *testing.T) {
		name := "plan-status-sticky"
		completedAt := time.Unix(10, 0).UTC()
		sp := &StoredPlan{PluginVersion: "test", GeneratedAt: time.Unix(1, 0).UTC(), CompletedAt: completedAt, PlanStatus: PlanStatusFailed, Plan: &Plan{}}
		b, _ := json.MarshalIndent(sp, "", "  ")
		cm := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Namespace: SystemNamespace, Name: name, Labels: map[string]string{PlanConfigMapLabelKey: "true"}},
			Data:       map[string]string{PlanConfigMapLabelKey + ".json": string(b)},
		}

		pl, cleanup := newSharedStateWithConfigMapInformer(t, cm)
		defer cleanup()

		pl.setPlanStatusInConfigMap(ctx, name, PlanStatusCompleted)

		gotCM, err := pl.Client.CoreV1().ConfigMaps(SystemNamespace).Get(ctx, name, metav1.GetOptions{})
		must(t, err == nil, "get err=%v", err)

		var got StoredPlan
		must(t, json.Unmarshal([]byte(gotCM.Data[PlanConfigMapLabelKey+".json"]), &got) == nil, "unmarshal failed")
		mustEq(t, got.PlanStatus, PlanStatusFailed, "status should stay failed")
		must(t, got.CompletedAt.Equal(completedAt), "CompletedAt should remain unchanged")
	})

	t.Run("invalid json is best-effort noop", func(t *testing.T) {
		name := "plan-status-invalid-json"
		bad := "{"
		cm := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Namespace: SystemNamespace, Name: name, Labels: map[string]string{PlanConfigMapLabelKey: "true"}},
			Data:       map[string]string{PlanConfigMapLabelKey + ".json": bad},
		}

		pl, cleanup := newSharedStateWithConfigMapInformer(t, cm)
		defer cleanup()

		pl.setPlanStatusInConfigMap(ctx, name, PlanStatusCompleted)

		gotCM, err := pl.Client.CoreV1().ConfigMaps(SystemNamespace).Get(ctx, name, metav1.GetOptions{})
		must(t, err == nil, "get err=%v", err)
		mustEq(t, gotCM.Data[PlanConfigMapLabelKey+".json"], bad, "expected unchanged bad json")
	})
}

// -------------------------
// clusterFingerprint
// -------------------------

func TestClusterFingerprint_Deterministic_ExcludesPending_AndUnusableNodes(t *testing.T) {
	// Nodes: one usable, one unschedulable, one not-ready
	n1 := node("n1", withAllocatable("2000m", "2Gi"))
	nBad := node("n-bad", unschedulable())
	nNR := node("n-nr", notReady())

	// Pods:
	// - running on usable node => included
	pRun := pod("ns", "run", withUID("u-run"), onNode("n1"), withReqs("100m", "64Mi"), withPrio(10))
	// - pending => excluded
	pPend := pod("ns", "pend", withUID("u-pend"), withReqs("100m", "64Mi"), withPrio(10))
	// - running on unusable node => excluded
	pOnBad := pod("ns", "bad", withUID("u-bad"), onNode("n-bad"), withReqs("100m", "64Mi"), withPrio(10))

	// Same data but shuffled order should yield identical fingerprint
	fp1 := clusterFingerprint([]*v1.Node{n1, nBad, nNR}, []*v1.Pod{pRun, pPend, pOnBad})
	fp2 := clusterFingerprint([]*v1.Node{nNR, nBad, n1}, []*v1.Pod{pOnBad, pPend, pRun})
	mustEq(t, fp1, fp2, "fingerprint should be deterministic")

	// Changing only pending pods must not affect fingerprint
	pPend2 := pod("ns", "pend2", withUID("u-pend2"), withReqs("500m", "1Gi"), withPrio(999))
	fp3 := clusterFingerprint([]*v1.Node{n1}, []*v1.Pod{pRun, pPend, pPend2})
	mustEq(t, fp1, fp3, "pending pods should not affect fingerprint")

	// Changing a running pod request should affect fingerprint
	pRunMut := pod("ns", "run", withUID("u-run"), onNode("n1"), withReqs("200m", "64Mi"), withPrio(10))
	fp4 := clusterFingerprint([]*v1.Node{n1}, []*v1.Pod{pRunMut})
	must(t, fp4 != clusterFingerprint([]*v1.Node{n1}, []*v1.Pod{pRun}), "expected fingerprint to change when running pod changes")
}

func TestClusterFingerprint_IgnoresDeletedPods_AndNilNodes(t *testing.T) {
	n1 := node("n1", withAllocatable("2000m", "2Gi"))

	pRun := pod("ns", "run", withUID("u-run"), onNode("n1"), withReqs("100m", "64Mi"), withPrio(10))
	pDel := pod("ns", "del", withUID("u-del"), onNode("n1"), withReqs("100m", "64Mi"), withPrio(10))
	now := metav1.Now()
	pDel.DeletionTimestamp = &now

	fp1 := clusterFingerprint([]*v1.Node{nil, n1}, []*v1.Pod{pRun, pDel})
	fp2 := clusterFingerprint([]*v1.Node{n1}, []*v1.Pod{pRun})
	mustEq(t, fp1, fp2, "deleted pods and nil nodes must not affect fingerprint")
}

func TestCollectEvictions_NilOrEmpty(t *testing.T) {
	byUID := map[types.UID]*v1.Pod{
		types.UID("u1"): pod("ns", "p1", withUID("u1"), onNode("n1")),
	}

	mustEq(t, collectEvictions(nil, byUID), []SolverPod(nil), "nil out should return nil")
	mustEq(t, collectEvictions(&SolverOutput{Evictions: nil}, byUID), []SolverPod(nil), "empty evictions should return nil")
	mustEq(t, collectEvictions(&SolverOutput{Evictions: []SolverPod{}}, byUID), []SolverPod(nil), "empty evictions slice should return nil")
}

func TestCollectEvictions_SkipsMissingPendingAndDeleting(t *testing.T) {
	pRun := pod("ns", "run", withUID("u-run"), onNode("n1"))
	pPend := pod("ns", "pend", withUID("u-pend")) // pending (no node)
	pDel := pod("ns", "del", withUID("u-del"), onNode("n2"))
	now := metav1.Now()
	pDel.DeletionTimestamp = &now // terminating => not alive

	byUID := map[types.UID]*v1.Pod{
		pRun.UID:  pRun,
		pPend.UID: pPend,
		pDel.UID:  pDel,
		// note: no entry for "u-missing"
	}

	out := &SolverOutput{
		Evictions: []SolverPod{
			{UID: pRun.UID},               // included
			{UID: pPend.UID},              // skipped (not assigned)
			{UID: pDel.UID},               // skipped (terminating)
			{UID: types.UID("u-missing")}, // skipped (unknown UID)
		},
	}

	ev := collectEvictions(out, byUID)
	mustEq(t, len(ev), 1, "only running+alive should be evicted")
	mustEq(t, ev[0].UID, pRun.UID, "wrong evicted UID")
	mustEq(t, ev[0].Node, "n1", "wrong node for evicted placement")
}

func TestIsPodAllowedByPlan_MissingWorkload_QuotaZero_NodeNotInQuota(t *testing.T) {
	pl := &SharedState{}

	// Set an active plan with quotas only for rs-1
	pOwned := pod("ns", "owned", withOwner("ReplicaSet", "rs-1"))
	wk, _ := getTopWorkload(pOwned)
	wkStr := wk.String()

	ap := &ActivePlan{
		ID:              "plan",
		PlacementByName: map[string]string{},
		WorkloadQuotas:  buildWorkloadQuotas(WorkloadQuotas{wkStr: {"n1": 1}}),
	}
	pl.ActivePlan.Store(ap)

	// (A) workload not present in plan => false
	pOther := pod("ns", "other", withOwner("ReplicaSet", "rs-other"))
	must(t, !pl.isPodAllowedByPlan(pOther), "expected false for workload not in plan")

	// (B) node is set, but NOT present in quota map => false (hits `exists == false`)
	pOnNX := pod("ns", "owned-nx", withOwner("ReplicaSet", "rs-1"), onNode("nx"))
	must(t, !pl.isPodAllowedByPlan(pOnNX), "expected false when node not present in perNode quotas")

	// (C) no node selected, but all quotas are 0 => false (hits the final loop-return false)
	ap.WorkloadQuotas = buildWorkloadQuotas(WorkloadQuotas{wkStr: {"n1": 0}})
	pPending := pod("ns", "owned-pend", withOwner("ReplicaSet", "rs-1")) // pending
	must(t, !pl.isPodAllowedByPlan(pPending), "expected false when all quotas exhausted and pod has no node")
}

func TestExportPlanToConfigMap_DefaultPath_CreateErrorPropagates(t *testing.T) {
	orig := exportPlanToConfigMapHook
	exportPlanToConfigMapHook = nil
	t.Cleanup(func() { exportPlanToConfigMapHook = orig })

	ctx := context.Background()
	pl, cleanup := newSharedStateWithConfigMapInformer(t)
	defer cleanup()

	// Make ConfigMap create fail inside ensureJson()
	client := pl.Client.(*fake.Clientset)
	client.PrependReactor("create", "configmaps", func(action clienttesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("create boom")
	})

	name := fmt.Sprintf("%s%s", PlanConfigMapNamePrefix, "unit-create-fail")
	sp := &StoredPlan{PluginVersion: "test", GeneratedAt: time.Unix(1, 0).UTC(), PlanStatus: PlanStatusActive, Plan: &Plan{}}

	err := pl.exportPlanToConfigMap(ctx, name, sp)
	must(t, err != nil, "expected error")
	mustContains(t, err.Error(), "create boom")
}

func TestExportPlanToConfigMap_DefaultPath_PruneDeleteErrorPropagates(t *testing.T) {
	orig := exportPlanToConfigMapHook
	exportPlanToConfigMapHook = nil
	t.Cleanup(func() { exportPlanToConfigMapHook = orig })

	// Create enough existing plan CMs so pruneConfigMaps tries to delete at least one.
	var objs []runtime.Object
	for i := 0; i < PlansToRetain+2; i++ {
		cm := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace:         SystemNamespace,
				Name:              fmt.Sprintf("%sold-%02d", PlanConfigMapNamePrefix, i),
				Labels:            map[string]string{PlanConfigMapLabelKey: "true"},
				CreationTimestamp: metav1.NewTime(time.Unix(int64(i+1), 0).UTC()),
			},
			Data: map[string]string{PlanConfigMapLabelKey + ".json": `{}`},
		}
		objs = append(objs, cm)
	}

	ctx := context.Background()
	pl, cleanup := newSharedStateWithConfigMapInformer(t, objs...)
	defer cleanup()

	// Make delete fail during pruning.
	client := pl.Client.(*fake.Clientset)
	client.PrependReactor("delete", "configmaps", func(action clienttesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("delete boom")
	})
	client.PrependReactor("delete-collection", "configmaps", func(action clienttesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("delete boom")
	})

	name := fmt.Sprintf("%s%s", PlanConfigMapNamePrefix, "unit-prune-fail")
	sp := &StoredPlan{PluginVersion: "test", GeneratedAt: time.Unix(1, 0).UTC(), PlanStatus: PlanStatusActive, Plan: &Plan{}}

	err := pl.exportPlanToConfigMap(ctx, name, sp)
	must(t, err != nil, "expected error")
	mustContains(t, err.Error(), "delete boom")
}
