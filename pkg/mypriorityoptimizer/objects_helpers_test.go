// objects_helpers_test.go
// with the help of AI tools to cover more branches/cases
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

func mustPodSet(t *testing.T, got []*v1.Pod, want ...string) {
	t.Helper()
	gotSet := map[string]struct{}{}
	for _, p := range got {
		gotSet[mergeNsName(p.Namespace, p.Name)] = struct{}{}
	}
	wantSet := map[string]struct{}{}
	for _, k := range want {
		wantSet[k] = struct{}{}
	}
	if !reflect.DeepEqual(gotSet, wantSet) {
		t.Fatalf("pods=%v want=%v", gotSet, wantSet)
	}
}

func toRuntimeObjs(pods ...*v1.Pod) []runtime.Object {
	out := make([]runtime.Object, 0, len(pods))
	for _, p := range pods {
		if p != nil {
			out = append(out, p)
		}
	}
	return out
}

// -------------------------
// Listers + getNodes/getPods
// -------------------------

func TestListersAndGetters(t *testing.T) {
	pl := &SharedState{}

	t.Run("nodesLister/podsLister can be injected", func(t *testing.T) {
		nl := &FakeNodeLister{}
		plst := &FakePodLister{}
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
		withNodeLister(&FakeNodeLister{Nodes: []*v1.Node{node("n1")}}, func() {
			got, err := pl.getNodes()
			if err != nil {
				t.Fatalf("getNodes() err=%v", err)
			}
			if len(got) != 1 || got[0].Name != "n1" {
				t.Fatalf("getNodes()=%#v want [n1]", got)
			}
		})

		sentinel := errors.New("boom")
		withNodeLister(&FakeNodeLister{Error: sentinel}, func() {
			_, err := pl.getNodes()
			if !errors.Is(err, sentinel) {
				t.Fatalf("getNodes() err=%v want %v", err, sentinel)
			}
		})
	})

	t.Run("getPods paths", func(t *testing.T) {
		type tc struct {
			name          string
			listerPods    []*v1.Pod
			clientPods    []*v1.Pod
			clientListErr error
			want          []string
		}

		tests := []tc{
			{
				name:       "lister error bubbles",
				listerPods: nil,
				want:       nil,
			},
			{
				name:       "hasAssigned -> returns lister view (no fallback)",
				listerPods: []*v1.Pod{pod("ns", "p1", onNode("n1"))},
				clientPods: []*v1.Pod{pod("ns", "ignored")},
				want:       []string{"ns/p1"},
			},
			{
				name:       "Client=nil and only unassigned -> returns lister view",
				listerPods: []*v1.Pod{pod("ns", "p1")},
				clientPods: nil,
				want:       []string{"ns/p1"},
			},
			{
				name:       "fallback API list success returns API pods",
				listerPods: []*v1.Pod{pod("ns", "p1")}, // only unassigned -> triggers fallback
				clientPods: []*v1.Pod{pod("ns", "p1"), pod("ns", "p2", onNode("n1"))},
				want:       []string{"ns/p1", "ns/p2"},
			},
			{
				name:          "fallback API list error -> best-effort returns lister view",
				listerPods:    []*v1.Pod{pod("ns", "p1")},
				clientPods:    []*v1.Pod{},
				clientListErr: errors.New("list boom"),
				want:          []string{"ns/p1"},
			},
		}

		// Case where for lister error
		t.Run("lister error", func(t *testing.T) {
			sentinel := errors.New("boom")
			withPodLister(&FakePodLister{Error: sentinel}, func() {
				_, err := pl.getPods()
				if !errors.Is(err, sentinel) {
					t.Fatalf("getPods err=%v want %v", err, sentinel)
				}
			})
		})

		for _, tt := range tests[1:] {
			t.Run(tt.name, func(t *testing.T) {
				lister := &FakePodLister{Store: storeFromPods(tt.listerPods...)}
				withPodLister(lister, func() {
					if tt.clientPods != nil {
						cs := fake.NewSimpleClientset(toRuntimeObjs(tt.clientPods...)...)
						if tt.clientListErr != nil {
							cs.Fake.PrependReactor("list", "pods", func(k8stesting.Action) (bool, runtime.Object, error) {
								return true, nil, tt.clientListErr
							})
						}
						pl.Client = cs
						t.Cleanup(func() { pl.Client = nil })
					} else {
						pl.Client = nil
					}

					got, err := pl.getPods()
					if err != nil {
						t.Fatalf("getPods() err=%v", err)
					}
					mustPodSet(t, got, tt.want...)
				})
			})
		}
	})
}

// -------------------------
// podRef / merge / split
// -------------------------

func TestNamespaceNameHelpers(t *testing.T) {
	p := pod("ns", "p")
	if got := podRef(p); got != "ns/p" {
		t.Fatalf("podRef=%q want %q", got, "ns/p")
	}

	if got := mergeNsName("ns", "name"); got != "ns/name" {
		t.Fatalf("mergeNsName=%q want %q", got, "ns/name")
	}

	tests := []struct {
		in       string
		wantNS   string
		wantName string
		wantErr  bool
	}{
		{"ns/name", "ns", "name", false},
		{"ns/name/extra", "ns", "name/extra", false},
		{"/name", "", "name", false},
		{"invalid", "", "", true},
	}

	for _, tt := range tests {
		ns, name, err := splitNsName(tt.in)
		if tt.wantErr {
			if err == nil {
				t.Fatalf("splitNsName(%q) expected error", tt.in)
			}
			continue
		}
		if err != nil || ns != tt.wantNS || name != tt.wantName {
			t.Fatalf("splitNsName(%q)=(%q,%q,%v) want (%q,%q,nil)",
				tt.in, ns, name, err, tt.wantNS, tt.wantName)
		}
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
		pod("ns", "terminating", withDeletionTimestamp(now)),
	}
	if got := countPendingPods(pods); got != 1 {
		t.Fatalf("countPendingPods=%d want 1", got)
	}
}

// -------------------------
// evictPod
// -------------------------

func TestEvictPod(t *testing.T) {
	pl := &SharedState{}
	p := pod("ns", "p", withUID("uid-1"))

	t.Run("success captures eviction body", func(t *testing.T) {
		var gotEv *policyv1.Eviction
		withEvictHook(func(_ *SharedState, _ context.Context, podIn *v1.Pod, ev *policyv1.Eviction) error {
			if podIn != p {
				t.Fatalf("unexpected pod pointer")
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
		if gotEv.ObjectMeta.Namespace != "ns" || gotEv.ObjectMeta.Name != "p" {
			t.Fatalf("ObjectMeta mismatch: %#v", gotEv.ObjectMeta)
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
// Node helpers
// -------------------------

func TestNodeHelpers(t *testing.T) {
	t.Run("allocatable getters", func(t *testing.T) {
		n := node("n1", withAllocatable("1500m", "2Gi"))
		if got := getNodeCPUAllocatable(n); got != 1500 {
			t.Fatalf("cpu=%d want 1500", got)
		}
		qMem := resource.MustParse("2Gi")
		wantMem := qMem.Value()
		if got := getNodeMemoryAllocatable(n); got != wantMem {
			t.Fatalf("mem=%d want %d", got, wantMem)
		}
	})

	t.Run("isNodeControlPlane variants", func(t *testing.T) {
		tests := []struct {
			name string
			n    *v1.Node
			want bool
		}{
			{"worker", node("worker"), false},
			{"label control-plane", node("n1", withNodeLabels(map[string]string{"node-role.kubernetes.io/control-plane": "true"})), true},
			{"label master", node("n2", withNodeLabels(map[string]string{"node-role.kubernetes.io/master": "true"})), true},
			{"name control-plane", node("control-plane"), true},
			{"name kind-control-plane", node("kind-control-plane"), true},
		}
		for _, tt := range tests {
			if got := isNodeControlPlane(tt.n); got != tt.want {
				t.Fatalf("%s: got=%v want=%v", tt.name, got, tt.want)
			}
		}
	})

	t.Run("isNodeReady branches", func(t *testing.T) {
		if isNodeReady(&v1.Node{}) {
			t.Fatalf("no conditions => not ready")
		}
		nNoReady := &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{
			{Type: v1.NodeDiskPressure, Status: v1.ConditionFalse},
		}}}
		if isNodeReady(nNoReady) {
			t.Fatalf("no NodeReady condition => not ready")
		}
		nReadyLater := &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{
			{Type: v1.NodeDiskPressure, Status: v1.ConditionFalse},
			{Type: v1.NodeReady, Status: v1.ConditionTrue},
		}}}
		if !isNodeReady(nReadyLater) {
			t.Fatalf("NodeReady later => ready")
		}
		nReadyFalse := &v1.Node{Status: v1.NodeStatus{Conditions: []v1.NodeCondition{
			{Type: v1.NodeReady, Status: v1.ConditionFalse},
		}}}
		if isNodeReady(nReadyFalse) {
			t.Fatalf("NodeReady false => not ready")
		}
	})

	t.Run("isNodeNoScheduleConditionTainted covers empty Effect too", func(t *testing.T) {
		tests := []struct {
			name string
			n    *v1.Node
			want bool
		}{
			{"none", &v1.Node{}, false},
			{"not-ready NoSchedule", node("n", withNodeTaints(v1.Taint{Key: "node.kubernetes.io/not-ready", Effect: v1.TaintEffectNoSchedule})), true},
			{"unreachable NoSchedule", node("n", withNodeTaints(v1.Taint{Key: "node.kubernetes.io/unreachable", Effect: v1.TaintEffectNoSchedule})), true},
			{"not-ready empty effect counts", node("n", withNodeTaints(v1.Taint{Key: "node.kubernetes.io/not-ready"})), true},
			{"PreferNoSchedule ignored", node("n", withNodeTaints(v1.Taint{Key: "node.kubernetes.io/not-ready", Effect: v1.TaintEffectPreferNoSchedule})), false},
		}
		for _, tt := range tests {
			if got := isNodeNoScheduleConditionTainted(tt.n); got != tt.want {
				t.Fatalf("%s: got=%v want=%v", tt.name, got, tt.want)
			}
		}
	})

	t.Run("isNodeUsable table (covers all sub-checks)", func(t *testing.T) {
		base := node("n", withAllocatable("1000m", "1Gi"))

		tests := []struct {
			name string
			n    *v1.Node
			want bool
		}{
			{"nil", nil, false},
			{"control-plane", node("n", withAllocatable("1000m", "1Gi"), withNodeLabels(map[string]string{"node-role.kubernetes.io/control-plane": "true"})), false},
			{"unschedulable", func() *v1.Node { n := base.DeepCopy(); n.Spec.Unschedulable = true; return n }(), false},
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
			if got := isNodeUsable(tt.n); got != tt.want {
				t.Fatalf("%s: got=%v want=%v", tt.name, got, tt.want)
			}
		}
	})
}

// -------------------------
// Pod lookup (getPodByName/UID/getPod)
// -------------------------

func TestPodLookup(t *testing.T) {
	pl := &SharedState{}

	pName := pod("ns", "p", withUID("uid-1"))
	pOther := pod("other-ns", "other", withUID("uid-target"))

	t.Run("getPodByName success + per-key error", func(t *testing.T) {
		withPodLister(&FakePodLister{Store: storeFromPods(pName)}, func() {
			got, err := pl.getPodByName("ns", "p")
			if err != nil || got != pName {
				t.Fatalf("getPodByName got=%#v err=%v", got, err)
			}
		})

		sentinel := errors.New("boom")
		withPodLister(&FakePodLister{ErrorPerKey: map[string]error{"ns/p": sentinel}}, func() {
			got, err := pl.getPodByName("ns", "p")
			if !errors.Is(err, sentinel) || got != nil {
				t.Fatalf("got=%#v err=%v", got, err)
			}
		})
	})

	t.Run("getPodByUID success / notfound / list error", func(t *testing.T) {
		withPodLister(&FakePodLister{Store: storeFromPods(pName, pOther)}, func() {
			got, err := pl.getPodByUID(types.UID("uid-target"))
			if err != nil || got != pOther {
				t.Fatalf("getPodByUID got=%#v err=%v", got, err)
			}
		})

		withPodLister(&FakePodLister{Store: storeFromPods(pName)}, func() {
			got, err := pl.getPodByUID(types.UID("nope"))
			if err == nil || got != nil {
				t.Fatalf("expected notfound err, got=%#v err=%v", got, err)
			}
		})

		sentinel := errors.New("list boom")
		withPodLister(&FakePodLister{Error: sentinel}, func() {
			got, err := pl.getPodByUID(types.UID("uid-1"))
			if !errors.Is(err, sentinel) || got != nil {
				t.Fatalf("got=%#v err=%v", got, err)
			}
		})
	})

	t.Run("getPod: fast path by name, then fallback by UID", func(t *testing.T) {
		withPodLister(&FakePodLister{Store: storeFromPods(pName)}, func() {
			got := pl.getPod(types.UID("uid-1"), "ns", "p")
			if got != pName {
				t.Fatalf("fast path got=%#v", got)
			}
		})

		// Name exists but UID mismatch -> fallback to UID scan.
		pWrong := pod("ns", "p") // empty UID
		withPodLister(&FakePodLister{Store: storeFromPods(pWrong, pOther)}, func() {
			got := pl.getPod(types.UID("uid-target"), "ns", "p")
			if got != pOther {
				t.Fatalf("fallback got=%#v want other", got)
			}
		})

		// Get(ns/p) errors -> fallback to UID scan.
		sentinel := errors.New("boom")
		withPodLister(&FakePodLister{
			Store:       storeFromPods(pName),
			ErrorPerKey: map[string]error{"ns/p": sentinel},
		}, func() {
			got := pl.getPod(types.UID("uid-1"), "ns", "p")
			if got != pName {
				t.Fatalf("fallback-on-error got=%#v", got)
			}
		})

		withPodLister(&FakePodLister{Store: storeFromPods()}, func() {
			got := pl.getPod(types.UID("missing"), "ns", "p")
			if got != nil {
				t.Fatalf("expected nil, got=%#v", got)
			}
		})
	})
}

// -------------------------
// Pod resource and predicate helpers
// -------------------------

func TestPodResourceAndPredicateHelpers(t *testing.T) {
	t.Run("CPU/mem request sums", func(t *testing.T) {
		p := &v1.Pod{Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("100m"), v1.ResourceMemory: resource.MustParse("64Mi"),
				}}},
				{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("250m"), v1.ResourceMemory: resource.MustParse("128Mi"),
				}}},
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
		p := pod("ns", "p", withPrio(pr), onNode("n1"))
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

	t.Run("isSamePodUID requires non-empty and equal", func(t *testing.T) {
		if !isSamePodUID("u1", "u1") {
			t.Fatalf("expected true")
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
		if isPodAssignedAndAlive(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{DeletionTimestamp: &now},
			Spec:       v1.PodSpec{NodeName: "n"},
		}) {
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
		p1 := pod("ns", "p1", withUID("u1"))
		p2 := pod("ns", "p2", withUID("u2"), withDeletionTimestamp(now))
		p3 := pod("ns", "p3", withUID("u1")) // overwrites u1
		m := podsByUID([]*v1.Pod{p1, p2, nil, p3})
		if len(m) != 1 || m[types.UID("u1")].Name != "p3" {
			t.Fatalf("map=%#v", m)
		}
	})
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
		if got := tt.wk.String(); got != tt.want {
			t.Fatalf("%s: got=%q want=%q", tt.name, got, tt.want)
		}
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
		{"ReplicaSet controller true", []owner{{"ReplicaSet", "rs1", &ctrlTrue}}, true, WorkloadKey{Kind: wkReplicaSet, Namespace: "ns", Name: "rs1"}},
		{"StatefulSet controller true", []owner{{"StatefulSet", "ss1", &ctrlTrue}}, true, WorkloadKey{Kind: wkStatefulSet, Namespace: "ns", Name: "ss1"}},
		{"DaemonSet controller true", []owner{{"DaemonSet", "ds1", &ctrlTrue}}, true, WorkloadKey{Kind: wkDaemonSet, Namespace: "ns", Name: "ds1"}},
		{"Job controller true", []owner{{"Job", "job1", &ctrlTrue}}, true, WorkloadKey{Kind: wkJob, Namespace: "ns", Name: "job1"}},
		{"controller nil ignored", []owner{{"ReplicaSet", "rs1", nil}}, false, WorkloadKey{}},
		{"controller false ignored", []owner{{"ReplicaSet", "rs1", &ctrlFalse}}, false, WorkloadKey{}},
		{"unknown kind ignored", []owner{{"Deployment", "d1", &ctrlTrue}}, false, WorkloadKey{}},
		{"skip non-controller then accept later controller", []owner{{"ReplicaSet", "rs-old", &ctrlFalse}, {"ReplicaSet", "rs-new", &ctrlTrue}}, true, WorkloadKey{Kind: wkReplicaSet, Namespace: "ns", Name: "rs-new"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := pod("ns", "p")
			for _, o := range tt.owners {
				p.OwnerReferences = append(p.OwnerReferences, metav1.OwnerReference{
					Kind:       o.kind,
					Name:       o.name,
					Controller: o.ctrl,
				})
			}

			got, ok := getTopWorkload(p)
			if ok != tt.wantOK {
				t.Fatalf("ok=%v want=%v (wk=%#v)", ok, tt.wantOK, got)
			}
			if tt.wantOK && got != tt.wantWK {
				t.Fatalf("wk=%#v want=%#v", got, tt.wantWK)
			}
		})
	}
}
