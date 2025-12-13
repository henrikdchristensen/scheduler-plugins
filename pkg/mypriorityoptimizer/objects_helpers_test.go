// objects_helpers_test.go
package mypriorityoptimizer

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

// -------------------------
// Test Helpers
// -------------------------

func storeFromPods(pods ...*v1.Pod) map[string]map[string]*v1.Pod {
	out := map[string]map[string]*v1.Pod{}
	for _, p := range pods {
		if p == nil {
			continue
		}
		if out[p.Namespace] == nil {
			out[p.Namespace] = map[string]*v1.Pod{}
		}
		out[p.Namespace][p.Name] = p
	}
	return out
}

func mustPodSet(t *testing.T, got []*v1.Pod, wantNsNames ...string) {
	t.Helper()
	gotSet := map[string]struct{}{}
	for _, p := range got {
		gotSet[mergeNsName(p.Namespace, p.Name)] = struct{}{}
	}
	wantSet := map[string]struct{}{}
	for _, k := range wantNsNames {
		wantSet[k] = struct{}{}
	}
	if !reflect.DeepEqual(gotSet, wantSet) {
		t.Fatalf("pods = %#v, want set %#v", gotSet, wantSet)
	}
}

// -------------------------
// listers + getNodes/getPods
// -------------------------

func TestListersAndGetters(t *testing.T) {
	pl := &SharedState{}

	t.Run("nodesLister/podsLister injection", func(t *testing.T) {
		nl := &fakeNodeLister{}
		plst := &fakePodLister{}
		withNodeLister(nl, func() {
			if got := pl.nodesLister(); got != nl {
				t.Fatalf("nodesLister() != injected")
			}
		})
		withPodLister(plst, func() {
			if got := pl.podsLister(); got != plst {
				t.Fatalf("podsLister() != injected")
			}
		})
	})

	t.Run("getNodes success + error", func(t *testing.T) {
		withNodeLister(&fakeNodeLister{nodes: []*v1.Node{{ObjectMeta: metav1.ObjectMeta{Name: "n1"}}}}, func() {
			got, err := pl.getNodes()
			if err != nil || len(got) != 1 || got[0].Name != "n1" {
				t.Fatalf("getNodes() = %#v err=%v", got, err)
			}
		})

		sentinel := errors.New("boom")
		withNodeLister(&fakeNodeLister{err: sentinel}, func() {
			_, err := pl.getNodes()
			if !errors.Is(err, sentinel) {
				t.Fatalf("getNodes() err=%v want %v", err, sentinel)
			}
		})
	})

	t.Run("getPods: lister error bubbles", func(t *testing.T) {
		sentinel := errors.New("boom")
		withPodLister(&fakePodLister{err: sentinel}, func() {
			_, err := pl.getPods()
			if !errors.Is(err, sentinel) {
				t.Fatalf("getPods() err=%v want %v", err, sentinel)
			}
		})
	})

	t.Run("getPods: hasAssigned -> returns lister view (no fallback)", func(t *testing.T) {
		pAssigned := pod("ns", "p1", onNode("n1"))
		lister := &fakePodLister{store: storeFromPods(pAssigned)}
		pl.Client = fake.NewSimpleClientset() // non-nil; still should early-return due to hasAssigned
		t.Cleanup(func() { pl.Client = nil })

		withPodLister(lister, func() {
			got, err := pl.getPods()
			if err != nil {
				t.Fatalf("getPods() err=%v", err)
			}
			mustPodSet(t, got, "ns/p1")
		})
	})

	t.Run("getPods: Client=nil and only unassigned -> returns lister view", func(t *testing.T) {
		pUnassigned := pod("ns", "p1")
		pl.Client = nil
		withPodLister(&fakePodLister{store: storeFromPods(pUnassigned)}, func() {
			got, err := pl.getPods()
			if err != nil {
				t.Fatalf("getPods() err=%v", err)
			}
			mustPodSet(t, got, "ns/p1")
		})
	})

	t.Run("getPods: fallback API list success returns API pods", func(t *testing.T) {
		// Lister only sees unassigned -> triggers fallback.
		pListerOnly := pod("ns", "p1")
		withPodLister(&fakePodLister{store: storeFromPods(pListerOnly)}, func() {
			// API sees more (including assigned).
			pAPIExtra := pod("ns", "p2", onNode("n1"))
			pl.Client = fake.NewSimpleClientset(pListerOnly, pAPIExtra)
			t.Cleanup(func() { pl.Client = nil })

			got, err := pl.getPods()
			if err != nil {
				t.Fatalf("getPods() err=%v", err)
			}
			mustPodSet(t, got, "ns/p1", "ns/p2")
		})
	})

	t.Run("getPods: fallback API list error -> best-effort returns lister view", func(t *testing.T) {
		pListerOnly := pod("ns", "p1")
		withPodLister(&fakePodLister{store: storeFromPods(pListerOnly)}, func() {
			cs := fake.NewSimpleClientset()
			sentinel := errors.New("list boom")
			cs.PrependReactor("list", "pods", func(k8stesting.Action) (bool, runtime.Object, error) {
				return true, nil, sentinel
			})
			pl.Client = cs
			t.Cleanup(func() { pl.Client = nil })

			got, err := pl.getPods()
			if err != nil {
				t.Fatalf("getPods() err=%v", err)
			}
			// Should fall back to lister output.
			mustPodSet(t, got, "ns/p1")
		})
	})
}

// -------------------------
// podRef / merge / split
// -------------------------

func TestNamespaceNameHelpers(t *testing.T) {
	p := pod("ns", "p")
	if got := podRef(p); got != "ns/p" {
		t.Fatalf("podRef()=%q want %q", got, "ns/p")
	}

	if got := mergeNsName("ns", "name"); got != "ns/name" {
		t.Fatalf("mergeNsName()=%q want %q", got, "ns/name")
	}

	ns, name, err := splitNsName("ns/name")
	if err != nil || ns != "ns" || name != "name" {
		t.Fatalf("splitNsName()=(%q,%q,%v) want (ns,name,nil)", ns, name, err)
	}
	if _, _, err := splitNsName("invalid"); err == nil {
		t.Fatalf("splitNsName(invalid) expected error")
	}
}

// -------------------------
// countPendingPods
// -------------------------

func TestCountPendingPods(t *testing.T) {
	if got := countPendingPods(nil); got != 0 {
		t.Fatalf("countPendingPods(nil)=%d want 0", got)
	}

	now := metav1.NewTime(time.Now())
	pods := []*v1.Pod{
		nil,
		pod("ns", "running", onNode("n1")),
		pod("ns", "pending"),
		{ObjectMeta: metav1.ObjectMeta{Name: "terminating", Namespace: "ns", DeletionTimestamp: &now}},
	}
	if got := countPendingPods(pods); got != 1 {
		t.Fatalf("countPendingPods()=%d want 1", got)
	}
}

// -------------------------
// evictPod
// -------------------------

func TestEvictPod(t *testing.T) {
	pl := &SharedState{}
	p := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p", Namespace: "ns", UID: types.UID("uid-1")}}

	t.Run("success captures eviction body", func(t *testing.T) {
		var gotEv *policyv1.Eviction
		withEvictHook(func(_ *SharedState, _ context.Context, pod *v1.Pod, ev *policyv1.Eviction) error {
			if pod != p {
				t.Fatalf("unexpected pod")
			}
			gotEv = ev
			return nil
		}, func() {
			if err := pl.evictPod(context.Background(), p); err != nil {
				t.Fatalf("evictPod err=%v", err)
			}
		})

		if gotEv == nil || gotEv.DeleteOptions == nil || gotEv.DeleteOptions.GracePeriodSeconds == nil {
			t.Fatalf("eviction missing delete options")
		}
		if *gotEv.DeleteOptions.GracePeriodSeconds != 0 {
			t.Fatalf("grace=%d want 0", *gotEv.DeleteOptions.GracePeriodSeconds)
		}
		if gotEv.DeleteOptions.Preconditions == nil || gotEv.DeleteOptions.Preconditions.UID == nil || *gotEv.DeleteOptions.Preconditions.UID != p.UID {
			t.Fatalf("preconditions UID mismatch")
		}
	})

	t.Run("error propagates", func(t *testing.T) {
		sentinel := errors.New("boom")
		withEvictHook(func(_ *SharedState, _ context.Context, _ *v1.Pod, _ *policyv1.Eviction) error {
			return sentinel
		}, func() {
			err := pl.evictPod(context.Background(), p)
			if !errors.Is(err, sentinel) {
				t.Fatalf("evictPod err=%v want %v", err, sentinel)
			}
		})
	})
}

// -------------------------
// node helpers
// -------------------------

func TestNodeHelpers(t *testing.T) {
	t.Run("allocatable getters", func(t *testing.T) {
		n := node("n1", withAllocatable("1500m", "2Gi"))
		if got := getNodeCPUAllocatable(n); got != 1500 {
			t.Fatalf("cpu=%d want 1500", got)
		}
		qMem := resource.MustParse("2Gi")
		if got := getNodeMemoryAllocatable(n); got != qMem.Value() {
			t.Fatalf("mem=%d want %d", got, qMem.Value())
		}
	})

	t.Run("isNodeControlPlane variants", func(t *testing.T) {
		tests := []struct {
			name string
			n    *v1.Node
			want bool
		}{
			{"worker", &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "worker", Labels: map[string]string{}}}, false},
			{"label control-plane", &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "n1", Labels: map[string]string{"node-role.kubernetes.io/control-plane": "true"}}}, true},
			{"label master", &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "n2", Labels: map[string]string{"node-role.kubernetes.io/master": "true"}}}, true},
			{"name control-plane", &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "control-plane"}}, true},
			{"name kind-control-plane", &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "kind-control-plane"}}, true},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := isNodeControlPlane(tt.n); got != tt.want {
					t.Fatalf("got=%v want=%v", got, tt.want)
				}
			})
		}
	})

	t.Run("isNodeReady: NodeReady not-first condition", func(t *testing.T) {
		n := &v1.Node{Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{Type: v1.NodeDiskPressure, Status: v1.ConditionFalse},
				{Type: v1.NodeReady, Status: v1.ConditionTrue},
			},
		}}
		if !isNodeReady(n) {
			t.Fatalf("expected ready")
		}
	})

	t.Run("isNodeNoScheduleConditionTainted covers empty Effect too", func(t *testing.T) {
		tests := []struct {
			name string
			n    *v1.Node
			want bool
		}{
			{"none", &v1.Node{}, false},
			{"not-ready NoSchedule", &v1.Node{Spec: v1.NodeSpec{Taints: []v1.Taint{{Key: "node.kubernetes.io/not-ready", Effect: v1.TaintEffectNoSchedule}}}}, true},
			{"unreachable NoSchedule", &v1.Node{Spec: v1.NodeSpec{Taints: []v1.Taint{{Key: "node.kubernetes.io/unreachable", Effect: v1.TaintEffectNoSchedule}}}}, true},
			{"not-ready empty effect counts", &v1.Node{Spec: v1.NodeSpec{Taints: []v1.Taint{{Key: "node.kubernetes.io/not-ready"}}}}, true},
			{"PreferNoSchedule ignored", &v1.Node{Spec: v1.NodeSpec{Taints: []v1.Taint{{Key: "node.kubernetes.io/not-ready", Effect: v1.TaintEffectPreferNoSchedule}}}}, false},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := isNodeNoScheduleConditionTainted(tt.n); got != tt.want {
					t.Fatalf("got=%v want=%v", got, tt.want)
				}
			})
		}
	})

	t.Run("isNodeUsable table (covers every sub-check)", func(t *testing.T) {
		base := node("n", withAllocatable("1000m", "1Gi"))

		tests := []struct {
			name string
			n    *v1.Node
			want bool
		}{
			{"nil", nil, false},
			{"control-plane", func() *v1.Node {
				n := base.DeepCopy()
				n.Labels = map[string]string{"node-role.kubernetes.io/control-plane": "true"}
				return n
			}(), false},
			{"unschedulable", func() *v1.Node {
				n := base.DeepCopy()
				n.Spec.Unschedulable = true
				return n
			}(), false},
			{"not ready", func() *v1.Node {
				n := base.DeepCopy()
				n.Status.Conditions = []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}}
				return n
			}(), false},
			{"tainted", func() *v1.Node {
				n := base.DeepCopy()
				n.Spec.Taints = []v1.Taint{{Key: "node.kubernetes.io/not-ready", Effect: v1.TaintEffectNoSchedule}}
				return n
			}(), false},
			{"zero allocatable", func() *v1.Node {
				n := base.DeepCopy()
				n.Status.Allocatable[v1.ResourceCPU] = resource.MustParse("0m")
				n.Status.Allocatable[v1.ResourceMemory] = resource.MustParse("0")
				return n
			}(), false},
			{"usable", base, true},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := isNodeUsable(tt.n); got != tt.want {
					t.Fatalf("got=%v want=%v", got, tt.want)
				}
			})
		}
	})
}

func TestIsNodeReady(t *testing.T) {
	t.Run("no conditions", func(t *testing.T) {
		n := &v1.Node{}
		if isNodeReady(n) {
			t.Fatalf("node with no conditions should not be ready")
		}
	})

	t.Run("no NodeReady condition present", func(t *testing.T) {
		n := &v1.Node{
			Status: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{Type: v1.NodeMemoryPressure, Status: v1.ConditionFalse},
					{Type: v1.NodeDiskPressure, Status: v1.ConditionFalse},
				},
			},
		}
		if isNodeReady(n) {
			t.Fatalf("node with no NodeReady condition should not be ready")
		}
	})

	t.Run("NodeReady appears later", func(t *testing.T) {
		n := &v1.Node{
			Status: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{Type: v1.NodeDiskPressure, Status: v1.ConditionFalse},
					{Type: v1.NodeReady, Status: v1.ConditionTrue},
				},
			},
		}
		if !isNodeReady(n) {
			t.Fatalf("expected node to be ready when NodeReady=True exists later in list")
		}
	})

	t.Run("NodeReady false", func(t *testing.T) {
		n := &v1.Node{
			Status: v1.NodeStatus{
				Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			},
		}
		if isNodeReady(n) {
			t.Fatalf("node with NodeReady=False should not be ready")
		}
	})
}

// -------------------------
// pod lookup (getPodByName/UID/getPod)
// -------------------------

func TestPodLookup(t *testing.T) {
	pl := &SharedState{}

	pName := pod("ns", "p", withUID("uid-1"))
	pOther := pod("other-ns", "other", withUID("uid-target"))

	t.Run("getPodByName success + per-key error", func(t *testing.T) {
		withPodLister(&fakePodLister{store: storeFromPods(pName)}, func() {
			got, err := pl.getPodByName("ns", "p")
			if err != nil || got != pName {
				t.Fatalf("getPodByName got=%#v err=%v", got, err)
			}
		})

		sentinel := errors.New("boom")
		withPodLister(&fakePodLister{errPerKey: map[string]error{"ns/p": sentinel}}, func() {
			got, err := pl.getPodByName("ns", "p")
			if !errors.Is(err, sentinel) || got != nil {
				t.Fatalf("got=%#v err=%v", got, err)
			}
		})
	})

	t.Run("getPodByUID success / notfound / list error", func(t *testing.T) {
		withPodLister(&fakePodLister{store: storeFromPods(pName, pOther)}, func() {
			got, err := pl.getPodByUID(types.UID("uid-target"))
			if err != nil || got != pOther {
				t.Fatalf("getPodByUID got=%#v err=%v", got, err)
			}
		})

		withPodLister(&fakePodLister{store: storeFromPods(pName)}, func() {
			got, err := pl.getPodByUID(types.UID("nope"))
			if err == nil || got != nil {
				t.Fatalf("expected notfound err, got=%#v err=%v", got, err)
			}
		})

		sentinel := errors.New("list boom")
		withPodLister(&fakePodLister{err: sentinel}, func() {
			got, err := pl.getPodByUID(types.UID("uid-1"))
			if !errors.Is(err, sentinel) || got != nil {
				t.Fatalf("got=%#v err=%v", got, err)
			}
		})
	})

	t.Run("getPod: fast path by name, then fallback by UID", func(t *testing.T) {
		withPodLister(&fakePodLister{store: storeFromPods(pName)}, func() {
			got := pl.getPod(types.UID("uid-1"), "ns", "p")
			if got != pName {
				t.Fatalf("fast path got=%#v", got)
			}
		})

		// Name exists but UID mismatch -> fallback to UID.
		pWrong := pod("ns", "p")
		withPodLister(&fakePodLister{store: storeFromPods(pWrong, pOther)}, func() {
			got := pl.getPod(types.UID("uid-target"), "ns", "p")
			if got != pOther {
				t.Fatalf("fallback got=%#v want other", got)
			}
		})

		// Get(ns/p) errors -> fallback to UID scan.
		sentinel := errors.New("boom")
		withPodLister(&fakePodLister{
			store:     storeFromPods(pName),
			errPerKey: map[string]error{"ns/p": sentinel},
		}, func() {
			got := pl.getPod(types.UID("uid-1"), "ns", "p")
			if got != pName {
				t.Fatalf("fallback-on-error got=%#v", got)
			}
		})

		withPodLister(&fakePodLister{store: storeFromPods()}, func() {
			got := pl.getPod(types.UID("missing"), "ns", "p")
			if got != nil {
				t.Fatalf("expected nil, got=%#v", got)
			}
		})
	})
}

// -------------------------
// resource + pod predicate helpers
// -------------------------

func TestPodResourceAndPredicateHelpers(t *testing.T) {
	t.Run("CPU/mem request sums", func(t *testing.T) {
		p := &v1.Pod{Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m"), v1.ResourceMemory: resource.MustParse("64Mi")}}},
				{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("250m"), v1.ResourceMemory: resource.MustParse("128Mi")}}},
			},
		}}
		if got := getPodCPURequest(p); got != 350 {
			t.Fatalf("cpu=%d want 350", got)
		}
		q64 := resource.MustParse("64Mi")
		q128 := resource.MustParse("128Mi")
		wantMem := q64.Value() + q128.Value()
		if got := getPodMemoryRequest(p); got != wantMem {
			t.Fatalf("mem=%d want %d", got, wantMem)
		}
	})

	t.Run("priority + assigned node", func(t *testing.T) {
		pr := int32(10)
		p := &v1.Pod{Spec: v1.PodSpec{Priority: &pr, NodeName: "n1"}}
		if getPodPriority(p) != 10 {
			t.Fatalf("priority wrong")
		}
		if getPodAssignedNodeName(p) != "n1" {
			t.Fatalf("node name wrong")
		}
		if getPodPriority(&v1.Pod{}) != 0 {
			t.Fatalf("nil priority should be 0")
		}
	})

	t.Run("isSamePodUID branches", func(t *testing.T) {
		if !isSamePodUID("u1", "u1") {
			t.Fatalf("expected same")
		}
		if isSamePodUID("u1", "u2") || isSamePodUID("", "u2") || isSamePodUID("u1", "") || isSamePodUID("", "") {
			t.Fatalf("unexpected true")
		}
	})

	t.Run("isPodDeleted / assigned / assigned+alive / protected", func(t *testing.T) {
		now := metav1.NewTime(time.Now())

		if !isPodDeleted(nil) {
			t.Fatalf("nil should be deleted")
		}
		p := &v1.Pod{}
		if isPodDeleted(p) {
			t.Fatalf("should not be deleted")
		}
		p.DeletionTimestamp = &now
		if !isPodDeleted(p) {
			t.Fatalf("should be deleted")
		}

		if isPodAssigned(nil) || isPodAssigned(&v1.Pod{}) {
			t.Fatalf("should not be assigned")
		}
		if !isPodAssigned(&v1.Pod{Spec: v1.PodSpec{NodeName: "n"}}) {
			t.Fatalf("should be assigned")
		}

		if isPodAssignedAndAlive(&v1.Pod{}) {
			t.Fatalf("unassigned should not be alive+assigned")
		}
		if !isPodAssignedAndAlive(&v1.Pod{Spec: v1.PodSpec{NodeName: "n"}}) {
			t.Fatalf("assigned should be alive")
		}
		if isPodAssignedAndAlive(&v1.Pod{ObjectMeta: metav1.ObjectMeta{DeletionTimestamp: &now}, Spec: v1.PodSpec{NodeName: "n"}}) {
			t.Fatalf("terminating should not count")
		}

		if isPodProtected(nil) {
			t.Fatalf("nil not protected")
		}
		if !isPodProtected(pod(SystemNamespace, "sys")) {
			t.Fatalf("system pod protected")
		}
		if isPodProtected(pod("default", "p")) {
			t.Fatalf("default pod not protected")
		}
	})

	t.Run("podsByUID skips deleted/nil and overwrites by UID", func(t *testing.T) {
		now := metav1.NewTime(time.Now())
		p1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "ns", UID: types.UID("u1")}}
		p2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p2", Namespace: "ns", UID: types.UID("u2"), DeletionTimestamp: &now}}
		p3 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p3", Namespace: "ns", UID: types.UID("u1")}}
		m := podsByUID([]*v1.Pod{p1, p2, nil, p3})
		if len(m) != 1 || m[types.UID("u1")].Name != "p3" {
			t.Fatalf("map=%#v", m)
		}
	})
}

// -------------------------
// clusterFingerprint (kept compact, still hits branches)
// -------------------------

func TestClusterFingerprint_CoreProperties(t *testing.T) {
	n1 := node("n1", withAllocatable("1000m", "1Gi"))
	n2 := node("n2", withAllocatable("2000m", "2Gi"))

	running := pod("ns", "p1", onNode("n1"), withReqs("100m", "128Mi"))
	pending := pod("ns", "p2")

	fp1 := clusterFingerprint([]*v1.Node{n2, n1}, []*v1.Pod{pending, running})
	fp2 := clusterFingerprint([]*v1.Node{n1, n2}, []*v1.Pod{running, pending})
	if fp1 != fp2 {
		t.Fatalf("not deterministic: %q vs %q", fp1, fp2)
	}

	// running CPU change changes fingerprint
	r2 := running.DeepCopy()
	r2.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("200m")
	if fp3 := clusterFingerprint([]*v1.Node{n1, n2}, []*v1.Pod{r2}); fp3 == fp1 {
		t.Fatalf("expected fingerprint to change on running CPU change")
	}

	// pending-only changes should not matter
	pending2 := pod("ns", "p3")
	if fp4 := clusterFingerprint([]*v1.Node{n1, n2}, []*v1.Pod{running, pending, pending2}); fp4 != fp1 {
		t.Fatalf("pending pods should not affect fingerprint")
	}

	// ignore nil/unusable nodes + pods scheduled on unusable nodes
	nBad := &v1.Node{ObjectMeta: metav1.ObjectMeta{
		Name:   "bad",
		Labels: map[string]string{"node-role.kubernetes.io/control-plane": "true"},
	}}
	onBad := pod("ns", "pbad", onNode("bad"), withReqs("50m", "64Mi"))

	if fp5 := clusterFingerprint([]*v1.Node{nil, nBad, n1, n2}, []*v1.Pod{running, onBad, pending}); fp5 != fp1 {
		t.Fatalf("should ignore unusable nodes/pods-on-them: fp5=%q base=%q", fp5, fp1)
	}
}

// specifically hits the pod key sorting comparator paths (same node + diff node)
func TestClusterFingerprint_SortingBranches(t *testing.T) {
	n1 := node("n1", withAllocatable("1000m", "1Gi"))
	n2 := node("n2", withAllocatable("1000m", "1Gi"))

	p1 := pod("ns", "p1", onNode("n1"), withReqs("100m", "128Mi"))
	p2 := pod("ns", "p2", onNode("n1"), withReqs("100m", "128Mi"))
	p3 := pod("ns", "p3", onNode("n2"), withReqs("100m", "128Mi"))
	fpA := clusterFingerprint([]*v1.Node{n2, n1}, []*v1.Pod{p3, p2, p1})
	fpB := clusterFingerprint([]*v1.Node{n1, n2}, []*v1.Pod{p1, p3, p2})
	if fpA != fpB {
		t.Fatalf("not deterministic: fpA=%q fpB=%q", fpA, fpB)
	}
}

// -------------------------
// WorkloadKey.String + getTopWorkload
// -------------------------

func TestWorkloadKeyString(t *testing.T) {
	tests := []struct {
		name string
		wk   WorkloadKey
		want string
	}{
		{"rs", WorkloadKey{Kind: wkReplicaSet, Namespace: "ns", Name: "foo"}, "rs:ns/foo"},
		{"ss", WorkloadKey{Kind: wkStatefulSet, Namespace: "ns", Name: "foo"}, "ss:ns/foo"},
		{"ds", WorkloadKey{Kind: wkDaemonSet, Namespace: "ns", Name: "foo"}, "ds:ns/foo"},
		{"job", WorkloadKey{Kind: wkJob, Namespace: "ns", Name: "foo"}, "job:ns/foo"},
		{"unknown", WorkloadKey{Kind: WorkloadKind(999), Namespace: "ns", Name: "foo"}, "ns/foo"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.wk.String(); got != tt.want {
				t.Fatalf("got=%q want=%q", got, tt.want)
			}
		})
	}
}

func TestGetTopWorkload(t *testing.T) {
	ctrlTrue := true
	ctrlFalse := false

	type owner struct {
		kind string
		name string
		ctrl *bool
	}

	tests := []struct {
		name   string
		owners []owner
		wantOK bool
		wantWK WorkloadKey
	}{
		{
			name:   "ReplicaSet controller true",
			owners: []owner{{kind: "ReplicaSet", name: "rs1", ctrl: &ctrlTrue}},
			wantOK: true,
			wantWK: WorkloadKey{Kind: wkReplicaSet, Namespace: "ns", Name: "rs1"},
		},
		{
			name:   "StatefulSet controller true",
			owners: []owner{{kind: "StatefulSet", name: "ss1", ctrl: &ctrlTrue}},
			wantOK: true,
			wantWK: WorkloadKey{Kind: wkStatefulSet, Namespace: "ns", Name: "ss1"},
		},
		{
			name:   "DaemonSet controller true",
			owners: []owner{{kind: "DaemonSet", name: "ds1", ctrl: &ctrlTrue}},
			wantOK: true,
			wantWK: WorkloadKey{Kind: wkDaemonSet, Namespace: "ns", Name: "ds1"},
		},
		{
			name:   "Job controller true",
			owners: []owner{{kind: "Job", name: "job1", ctrl: &ctrlTrue}},
			wantOK: true,
			wantWK: WorkloadKey{Kind: wkJob, Namespace: "ns", Name: "job1"},
		},
		{
			name:   "controller nil ignored",
			owners: []owner{{kind: "ReplicaSet", name: "rs1", ctrl: nil}},
			wantOK: false,
		},
		{
			name:   "controller false ignored",
			owners: []owner{{kind: "ReplicaSet", name: "rs1", ctrl: &ctrlFalse}},
			wantOK: false,
		},
		{
			name:   "unknown kind ignored",
			owners: []owner{{kind: "Deployment", name: "d1", ctrl: &ctrlTrue}},
			wantOK: false,
		},
		{
			name:   "skip non-controller then accept later controller",
			owners: []owner{{kind: "ReplicaSet", name: "rs-old", ctrl: &ctrlFalse}, {kind: "ReplicaSet", name: "rs-new", ctrl: &ctrlTrue}},
			wantOK: true,
			wantWK: WorkloadKey{Kind: wkReplicaSet, Namespace: "ns", Name: "rs-new"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p", Namespace: "ns"}}
			for _, o := range tt.owners {
				p.OwnerReferences = append(p.OwnerReferences, metav1.OwnerReference{
					Kind:       o.kind,
					Name:       o.name,
					Controller: o.ctrl,
				})
			}

			got, ok := getTopWorkload(p)
			if ok != tt.wantOK {
				t.Fatalf("ok=%v want %v (wk=%#v)", ok, tt.wantOK, got)
			}
			if tt.wantOK && got != tt.wantWK {
				t.Fatalf("wk=%#v want %#v", got, tt.wantWK)
			}
		})
	}
}

func TestBuildPendingSnapshot_NoUsableNodes(t *testing.T) {
	pl := &SharedState{}

	// control-plane node => not usable
	nBad := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "cp",
			Labels: map[string]string{"node-role.kubernetes.io/control-plane": "true"},
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1000m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
		},
	}

	// Pending/unassigned pod (should count as pending, but NOT affect fingerprint)
	pPending := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "p1",
			Namespace: "ns",
			UID:       types.UID("u1"),
		},
		Status: v1.PodStatus{Phase: v1.PodPending},
		Spec:   v1.PodSpec{NodeName: ""},
	}

	withNodeLister(&fakeNodeLister{nodes: []*v1.Node{nBad}}, func() {
		withPodLister(&fakePodLister{store: storeFromPods(pPending)}, func() {
			snap, err := pl.buildPendingSnapshot()
			if err != nil {
				t.Fatalf("buildPendingSnapshot() unexpected error: %v", err)
			}

			// Snapshot still returns nodes/pods as observed by listers.
			if len(snap.Nodes) != 1 || snap.Nodes[0].Name != "cp" {
				t.Fatalf("Nodes=%#v, want [cp]", snap.Nodes)
			}
			if len(snap.Pods) != 1 || snap.Pods[0].Name != "p1" {
				t.Fatalf("Pods=%#v, want [ns/p1]", snap.Pods)
			}

			// Pending pod should be counted.
			if snap.PendingCount != 1 {
				t.Fatalf("PendingCount=%d, want 1", snap.PendingCount)
			}
			if _, ok := snap.PendingUIDs[pPending.UID]; !ok {
				t.Fatalf("PendingUIDs missing %q", pPending.UID)
			}

			// Fingerprint should match the same helper used elsewhere (and will be the "empty" basis here).
			wantFP := clusterFingerprint([]*v1.Node{nBad}, []*v1.Pod{pPending})
			if snap.Fingerprint != wantFP {
				t.Fatalf("Fingerprint=%q, want %q", snap.Fingerprint, wantFP)
			}
		})
	})
}
