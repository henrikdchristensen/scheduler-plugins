// test_helpers_test.go
package mypriorityoptimizer

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/rest"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

type fakePodLister struct {
	store     map[string]map[string]*v1.Pod
	err       error
	errPerKey map[string]error
}

func (f *fakePodLister) List(_ labels.Selector) ([]*v1.Pod, error) {
	if f.err != nil {
		return nil, f.err
	}
	var out []*v1.Pod
	for _, nsMap := range f.store {
		for _, p := range nsMap {
			out = append(out, p)
		}
	}
	return out, nil
}

type fakePodNamespaceLister struct {
	ns        string
	store     map[string]map[string]*v1.Pod
	err       error
	errPerKey map[string]error
}

func (f *fakePodLister) Pods(namespace string) corev1listers.PodNamespaceLister {
	return &fakePodNamespaceLister{
		ns:        namespace,
		store:     f.store,
		err:       f.err,
		errPerKey: f.errPerKey,
	}
}

func (f *fakePodNamespaceLister) List(_ labels.Selector) ([]*v1.Pod, error) {
	if f.err != nil {
		return nil, f.err
	}
	var out []*v1.Pod
	if nsMap, ok := f.store[f.ns]; ok {
		for _, p := range nsMap {
			out = append(out, p)
		}
	}
	return out, nil
}

func (f *fakePodNamespaceLister) Get(name string) (*v1.Pod, error) {
	key := f.ns + "/" + name

	// Per-key error overrides everything else.
	if err, ok := f.errPerKey[key]; ok {
		return nil, err
	}
	if f.err != nil {
		return nil, f.err
	}

	nsMap := f.store[f.ns]
	if nsMap == nil {
		return nil, apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "pods"}, name)
	}
	p, ok := nsMap[name]
	if !ok {
		return nil, apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "pods"}, name)
	}
	return p, nil
}

type fakeHandle struct {
	cfg *rest.Config
	framework.Handle
	client  kubernetes.Interface
	factory informers.SharedInformerFactory
}

func (f *fakeHandle) KubeConfig() *rest.Config {
	return f.cfg
}

func (f *fakeHandle) ClientSet() kubernetes.Interface {
	return f.client
}

func (f *fakeHandle) SharedInformerFactory() informers.SharedInformerFactory {
	return f.factory
}

type fakeNodeLister struct {
	nodes []*v1.Node
	err   error
}

func (f *fakeNodeLister) List(selector labels.Selector) ([]*v1.Node, error) {
	return f.nodes, f.err
}

func (f *fakeNodeLister) Get(name string) (*v1.Node, error) {
	for _, n := range f.nodes {
		if n.Name == name {
			return n, nil
		}
	}
	return nil, fmt.Errorf("not found")
}

func withNodeLister(nl corev1listers.NodeLister, fn func()) {
	orig := nodesListerFor
	nodesListerFor = func(pl *SharedState) corev1listers.NodeLister { return nl }
	defer func() { nodesListerFor = orig }()
	fn()
}

func withPodLister(plister corev1listers.PodLister, fn func()) {
	orig := podsListerFor
	podsListerFor = func(pl *SharedState) corev1listers.PodLister { return plister }
	defer func() { podsListerFor = orig }()
	fn()
}

func withEvictHook(hook func(pl *SharedState, ctx context.Context, pod *v1.Pod, ev *policyv1.Eviction) error, fn func()) {
	orig := evictPodFor
	evictPodFor = hook
	defer func() { evictPodFor = orig }()
	fn()
}

// -------------------------
// withMode
// -------------------------

// withMode is a small helper to temporarily set the mode during a test and
// restore to the original values.
func withMode(mode ModeType, synch bool, fn func()) {
	oldMode := OptimizeMode
	oldSynch := OptimizeSolveSynch

	OptimizeMode = mode
	OptimizeSolveSynch = synch
	defer func() {
		OptimizeMode = oldMode
		OptimizeSolveSynch = oldSynch
	}()
	fn()
}

// -------------------------
// pod
// -------------------------

type PodOpt func(*v1.Pod)

func pod(ns, name string, opts ...PodOpt) *v1.Pod {
	// sensible defaults for tests
	prio := int32(0)
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
		Spec: v1.PodSpec{
			Priority: &prio,
		},
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func withUID(uid string) PodOpt {
	return func(p *v1.Pod) { p.UID = types.UID(uid) }
}

func onNode(node string) PodOpt {
	return func(p *v1.Pod) { p.Spec.NodeName = node }
}

func withPrio(prio int32) PodOpt {
	return func(p *v1.Pod) { p.Spec.Priority = &prio }
}

func withOwner(kind, name string) PodOpt {
	return func(p *v1.Pod) {
		controller := true
		p.OwnerReferences = append(p.OwnerReferences, metav1.OwnerReference{
			APIVersion: "apps/v1",
			Kind:       kind,
			Name:       name,
			Controller: &controller,
		})
	}
}

func withReqs(cpuReq, memReq string) PodOpt {
	return func(p *v1.Pod) {
		p.Spec.Containers = []v1.Container{{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(cpuReq),
					v1.ResourceMemory: resource.MustParse(memReq),
				},
			},
		}}
	}
}

func withPhase(ph v1.PodPhase) PodOpt {
	return func(p *v1.Pod) { p.Status.Phase = ph }
}

func withCreationTimestamp(ts metav1.Time) PodOpt {
	return func(p *v1.Pod) { p.CreationTimestamp = ts }
}

func withDeletionTimestamp(ts metav1.Time) PodOpt {
	return func(p *v1.Pod) {
		t := ts // ensure a unique address per pod
		p.DeletionTimestamp = &t
	}
}

// -------------------------
// node
// -------------------------

type NodeOpt func(*v1.Node)

func node(name string, opts ...NodeOpt) *v1.Node {
	n := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("1000m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
		},
	}
	for _, o := range opts {
		o(n)
	}
	return n
}

func withAllocatable(cpu, mem string) NodeOpt {
	return func(n *v1.Node) {
		if n.Status.Allocatable == nil {
			n.Status.Allocatable = v1.ResourceList{}
		}
		n.Status.Allocatable[v1.ResourceCPU] = resource.MustParse(cpu)
		n.Status.Allocatable[v1.ResourceMemory] = resource.MustParse(mem)
	}
}

func unschedulable() NodeOpt {
	return func(n *v1.Node) { n.Spec.Unschedulable = true }
}

func notReady() NodeOpt {
	return func(n *v1.Node) {
		n.Status.Conditions = []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}}
	}
}

// -------------------------
// mustHookStatus
// -------------------------

// mustHookStatus asserts the framework status code and (optionally) that the message contains a substring.
func mustHookStatus(t *testing.T, stage string, st *fwk.Status, want fwk.Code, contains string) {
	t.Helper()
	if st == nil {
		t.Fatalf("%s() returned nil status", stage)
	}
	if st.Code() != want {
		t.Fatalf("%s() code = %v, want %v (msg=%q)", stage, st.Code(), want, st.Message())
	}
	if contains != "" && !strings.Contains(st.Message(), contains) {
		t.Fatalf("%s() message = %q, want to contain %q", stage, st.Message(), contains)
	}
}

// -------------------------
// storeFromPods
// -------------------------

// storeFromPods creates a nested map from a list of pods for easy lookup by namespace and name.
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

// withVar temporarily sets a package-level var and restores it.
func withVar[T any](t *testing.T, ptr *T, v T) {
	t.Helper()
	old := *ptr
	*ptr = v
	t.Cleanup(func() { *ptr = old })
}

// -------------------------
// writeFakeSolverScript
// -------------------------

// writeFakeSolverScript writes a fake solver script to the specified directory
// with the specified body, and returns the full path to the script.
func writeFakeSolverScript(t *testing.T, dir, body string) string {
	t.Helper()
	path := filepath.Join(dir, "fake_solver.sh")
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatalf("failed to write fake solver script: %v", err)
	}
	return path
}

// requireBash skips the test if bash is not available.
func requireBash(t *testing.T) {
	t.Helper()
	requireNonWindows(t)
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not found on PATH")
	}
}

// testCtx returns a context that will be cancelled before the test's deadline
// (if one exists), otherwise uses a short timeout.
func testCtx(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()

	if dl, ok := t.Deadline(); ok {
		return context.WithDeadline(context.Background(), dl.Add(-200*time.Millisecond))
	}
	return context.WithTimeout(context.Background(), 1*time.Second)
}
