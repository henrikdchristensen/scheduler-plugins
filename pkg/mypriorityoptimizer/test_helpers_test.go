// test_helpers_test.go
package mypriorityoptimizer

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
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
	"k8s.io/client-go/kubernetes/fake"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/rest"
	fwk "k8s.io/kube-scheduler/framework"
	Framework "k8s.io/kubernetes/pkg/scheduler/framework"
)

// -------------------------
// Fake listers and handles
// -------------------------

// FakePodLister is a fake implementation of PodLister for testing.
type FakePodLister struct {
	Store       map[string]map[string]*v1.Pod
	Error       error
	ErrorPerKey map[string]error
}

// List lists all pods in the indexer.
func (f *FakePodLister) List(_ labels.Selector) ([]*v1.Pod, error) {
	if f.Error != nil {
		return nil, f.Error
	}
	var out []*v1.Pod
	for _, nsMap := range f.Store {
		for _, p := range nsMap {
			out = append(out, p)
		}
	}
	return out, nil
}

// FakePodNamespaceLister is a fake implementation of PodNamespaceLister for testing.
type FakePodNamespaceLister struct {
	Namespace   string
	Store       map[string]map[string]*v1.Pod
	Error       error
	ErrorPerKey map[string]error
}

// Pods returns an object that can list and get pods in the given namespace.
func (f *FakePodLister) Pods(namespace string) corev1listers.PodNamespaceLister {
	return &FakePodNamespaceLister{
		Namespace:   namespace,
		Store:       f.Store,
		Error:       f.Error,
		ErrorPerKey: f.ErrorPerKey,
	}
}

// List lists all pods in the indexer for a given namespace.
func (f *FakePodNamespaceLister) List(_ labels.Selector) ([]*v1.Pod, error) {
	if f.Error != nil {
		return nil, f.Error
	}
	var out []*v1.Pod
	if nsMap, ok := f.Store[f.Namespace]; ok {
		for _, p := range nsMap {
			out = append(out, p)
		}
	}
	return out, nil
}

// Get retrieves a pod by name.
func (f *FakePodNamespaceLister) Get(name string) (*v1.Pod, error) {
	key := f.Namespace + "/" + name

	// Per-key error overrides everything else.
	if err, ok := f.ErrorPerKey[key]; ok {
		return nil, err
	}
	if f.Error != nil {
		return nil, f.Error
	}

	nsMap := f.Store[f.Namespace]
	if nsMap == nil {
		return nil, apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "pods"}, name)
	}
	p, ok := nsMap[name]
	if !ok {
		return nil, apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "pods"}, name)
	}
	return p, nil
}

// FakeHandle is a fake implementation of framework.Handle for testing.
type FakeHandle struct {
	Cfg *rest.Config
	Framework.Handle
	Client  kubernetes.Interface
	Factory informers.SharedInformerFactory
}

// makeHandle creates a FakeHandle with the given host string.
func makeHandle(host string) *FakeHandle {
	cfg := &rest.Config{Host: host}
	return &FakeHandle{
		Cfg:     cfg,
		Factory: informers.NewSharedInformerFactory(fake.NewSimpleClientset(), 0),
	}
}

// KubeConfig returns the kubeconfig.
func (f *FakeHandle) KubeConfig() *rest.Config {
	return f.Cfg
}

// ClientSet returns the clientset.
func (f *FakeHandle) ClientSet() kubernetes.Interface {
	return f.Client
}

// SharedInformerFactory returns the shared informer factory.
func (f *FakeHandle) SharedInformerFactory() informers.SharedInformerFactory {
	return f.Factory
}

// Fake node lister
type FakeNodeLister struct {
	Nodes []*v1.Node
	Error error
}

// List lists all nodes.
func (f *FakeNodeLister) List(selector labels.Selector) ([]*v1.Node, error) {
	return f.Nodes, f.Error
}

// Get retrieves a node by name.
func (f *FakeNodeLister) Get(name string) (*v1.Node, error) {
	for _, n := range f.Nodes {
		if n.Name == name {
			return n, nil
		}
	}
	return nil, fmt.Errorf("not found")
}

// withNodeLister temporarily replaces nodesListerFor with the given lister
func withNodeLister(nl corev1listers.NodeLister, fn func()) {
	orig := nodesListerFor
	nodesListerFor = func(pl *SharedState) corev1listers.NodeLister { return nl }
	defer func() { nodesListerFor = orig }()
	fn()
}

// withPodLister temporarily replaces podsListerFor with the given lister
func withPodLister(plister corev1listers.PodLister, fn func()) {
	orig := podsListerFor
	podsListerFor = func(pl *SharedState) corev1listers.PodLister { return plister }
	defer func() { podsListerFor = orig }()
	fn()
}

// withEvictHook temporarily replaces evictPodFor with the given hook function
func withEvictHook(hook func(pl *SharedState, ctx context.Context, pod *v1.Pod, ev *policyv1.Eviction) error, fn func()) {
	orig := evictPodFor
	evictPodFor = hook
	defer func() { evictPodFor = orig }()
	fn()
}

// -------------------------
// node test helpers
// -------------------------

// NodeOpt is a functional option for node test helpers.
type NodeOpt func(*v1.Node)

// node creates a node with the given name and options.
func node(name string, opts ...NodeOpt) *v1.Node {
	n := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
	}
	for _, o := range opts {
		o(n)
	}
	return n
}

// withAllocatable sets the node's allocatable CPU and memory.
func withAllocatable(cpu, mem string) NodeOpt {
	return func(n *v1.Node) {
		if n.Status.Allocatable == nil {
			n.Status.Allocatable = v1.ResourceList{}
		}
		n.Status.Allocatable[v1.ResourceCPU] = resource.MustParse(cpu)
		n.Status.Allocatable[v1.ResourceMemory] = resource.MustParse(mem)
	}
}

// withNodeLabels sets the node's labels.
func withNodeLabels(labels map[string]string) NodeOpt {
	return func(n *v1.Node) { n.Labels = labels }
}

// withNodeTaints sets the node's taints.
func withNodeTaints(taints ...v1.Taint) NodeOpt {
	return func(n *v1.Node) {
		n.Spec.Taints = append([]v1.Taint(nil), taints...)
	}
}

// unschedulable marks the node as unschedulable.
func unschedulable() NodeOpt {
	return func(n *v1.Node) { n.Spec.Unschedulable = true }
}

// notReady marks the node as not ready.
func notReady() NodeOpt {
	return func(n *v1.Node) {
		n.Status.Conditions = []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}}
	}
}

// -------------------------
// pod test helpers
// -------------------------

// PodOpt is a functional option for pod test helpers.
type PodOpt func(*v1.Pod)

// pod creates a pod with the given namespace, name, and options.
func pod(ns, name string, opts ...PodOpt) *v1.Pod {
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

// withUID sets the pod UID.
func withUID(uid string) PodOpt {
	return func(p *v1.Pod) { p.UID = types.UID(uid) }
}

// onNode sets the pod's NodeName.
func onNode(node string) PodOpt {
	return func(p *v1.Pod) { p.Spec.NodeName = node }
}

// withPrio sets the pod's priority.
func withPrio(prio int32) PodOpt {
	return func(p *v1.Pod) { p.Spec.Priority = &prio }
}

// withOwner adds an owner reference to the pod.
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

// withReqs sets the pod's CPU and memory requests.
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

// withPhase sets the pod's phase.
func withPhase(ph v1.PodPhase) PodOpt {
	return func(p *v1.Pod) { p.Status.Phase = ph }
}

// withCreationTimestamp sets the pod's creation timestamp.
func withCreationTimestamp(ts metav1.Time) PodOpt {
	return func(p *v1.Pod) { p.CreationTimestamp = ts }
}

// withDeletionTimestamp sets the pod's deletion timestamp.
func withDeletionTimestamp(ts metav1.Time) PodOpt {
	return func(p *v1.Pod) {
		t := ts // ensure a unique address per pod
		p.DeletionTimestamp = &t
	}
}

// -------------------------
// Other helper functions
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

// requireNonWindows skips the test if running on Windows.
func requireNonWindows(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("skipping on windows (shell scripts / /dev/zero assumptions)")
	}
}

// runBashSolver writes a temporary bash script with the given content and runs
func runBashSolver(t *testing.T, script string, payload []byte) ([]byte, error) {
	t.Helper()
	tmpDir := t.TempDir()
	scriptPath := writeFakeSolverScript(t, tmpDir, script)

	pl := &SharedState{}
	ctx, cancel := testCtx(t)
	defer cancel()

	return pl.runSolverExternal(ctx, payload, "bash", scriptPath)
}

// withExecCommandContext temporarily replaces execCommandContext for the
// duration of the test.
func withExecCommandContext(t *testing.T, f func(ctx context.Context, name string, args ...string) *exec.Cmd) {
	t.Helper()
	orig := execCommandContext
	execCommandContext = f
	t.Cleanup(func() { execCommandContext = orig })
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

// ptr is a small helper to get a pointer to a value.
func ptr[T any](v T) *T { return &v }

// atomicInt is a small helper to create an *atomic.Int32 with the given initial value.
func atomicInt(v int32) *atomic.Int32 {
	a := new(atomic.Int32)
	a.Store(v)
	return a
}

// withMode is a small helper to temporarily set the mode during a test and
// restore to the original values.
func withMode(mode ModeType, synch bool, fn func()) {
	oldMode := OptimizeMode
	oldSynch := OptimizeBlockingSolving

	OptimizeMode = mode
	OptimizeBlockingSolving = synch
	defer func() {
		OptimizeMode = oldMode
		OptimizeBlockingSolving = oldSynch
	}()
	fn()
}
