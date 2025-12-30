// plugin.go
// TODO: MISSING CHECK FOR THIS FILE; simplifications, readability, etc.
package mypriorityoptimizer

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// -------------------------
// Name
// -------------------------

// Name returns name of the plugin.
func (pl *SharedState) Name() string { return Name }

// -------------------------
// newFromHandle
// -------------------------

// newFromHandle contains the real logic of New, parameterized by the client
// constructor and the full framework.Handle (to allow tests to pass a fake).
func newFromHandle(
	ctx context.Context,
	obj runtime.Object,
	clientFn func(*rest.Config) (kubernetes.Interface, error),
	h HandleDeps,
	fullHandle framework.Handle,
) (framework.Plugin, error) {
	// obj is not used by the current implementation, but the factory signature
	// requires it; assign to _ to keep the compiler happy.
	_ = obj

	// Build client from kubeconfig
	client, err := clientFn(h.KubeConfig())
	if err != nil {
		return nil, err
	}

	// Initialize shared state
	pl := &SharedState{
		Handle:             fullHandle,
		Client:             client,
		BlockedWhileActive: newPodSet("BlockedWhileActive"),
	}

	// Ensure at least one solver is enabled
	if !solverEnabled(pl) {
		klog.Error(ErrNoSolverEnabled)
		return nil, ErrNoSolverEnabled
	}

	// Warm informers via the factory
	f := h.SharedInformerFactory()
	podsInf := f.Core().V1().Pods().Informer()
	if err := podsInf.AddIndexers(cache.Indexers{
		cache.NamespaceIndex: cache.MetaNamespaceIndexFunc,
	}); err != nil {
		klog.ErrorS(err, "failed adding namespace indexer to Pod informer")
	}
	nodesInf := f.Core().V1().Nodes().Informer()
	cmsInf := f.Core().V1().ConfigMaps().Informer()
	rsInf := f.Apps().V1().ReplicaSets().Informer()
	ssInf := f.Apps().V1().StatefulSets().Informer()
	dsInf := f.Apps().V1().DaemonSets().Informer()
	jobInf := f.Batch().V1().Jobs().Informer()

	// Let the readiness hook decide how to run (goroutine in prod, inline in tests)
	pluginReadinessStarter(pl, ctx, podsInf, nodesInf, cmsInf, rsInf, ssInf, dsInf, jobInf)

	// Plugin configuration logging
	klog.InfoS("Plugin initialized", "name", Name, "version", PluginVersion, "mode", getModeCombinedAsString())
	klog.InfoS("Plan configuration", "executionTimeout", PlanExecutionTimeout.String())
	klog.InfoS("Solver configuration", solverConfigArgs()...)

	// Start HTTP server (hooked for tests)
	httpServerStarter(pl, ctx, HTTPAddr)

	return pl, nil
}

// -------------------------
// New
// -------------------------

// New is the scheduler's plugin factory. It delegates to newFromHandle with the
// real kubernetes.NewForConfig and the full framework.Handle.
func New(ctx context.Context, obj runtime.Object, h framework.Handle) (framework.Plugin, error) {
	clientFn := func(c *rest.Config) (kubernetes.Interface, error) {
		return kubernetes.NewForConfig(c)
	}
	return newFromHandle(ctx, obj, clientFn, h, h)
}

// -------------------------
// Test Hooks
// -------------------------

var (
	pluginReadinessStarter = func(pl *SharedState, ctx context.Context, infs ...cache.SharedIndexInformer) {
		go pl.pluginReadiness(ctx, infs...)
	}
	httpServerStarter = func(pl *SharedState, ctx context.Context, addr string) {
		go pl.startHttpServer(ctx, addr)
	}
	solverEnabled = func(pl *SharedState) bool {
		return pl.isAnySolverEnabled()
	}
)
