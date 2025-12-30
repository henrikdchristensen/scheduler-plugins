// plugin_test.go
// TODO: MISSING CHECK FOR THIS FILE; simplifications, readability, etc.
package mypriorityoptimizer

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
)

// -------------------------
// newFromHandle
// -------------------------

func TestNewFromHandle(t *testing.T) {
	wantClientErr := errors.New("boom-client")

	tests := []struct {
		name      string
		clientErr error
		solverOn  bool

		wantErrIs         error
		wantPluginNil     bool
		wantReadinessHook bool
		wantHTTPHook      bool
		wantHTTPAddr      string
	}{
		{
			name:              "client_error_propagated_no_hooks",
			clientErr:         wantClientErr,
			solverOn:          true,
			wantErrIs:         wantClientErr,
			wantPluginNil:     true,
			wantReadinessHook: false,
			wantHTTPHook:      false,
		},
		{
			name:              "no_solver_enabled_no_hooks",
			solverOn:          false,
			wantErrIs:         ErrNoSolverEnabled,
			wantPluginNil:     true,
			wantReadinessHook: false,
			wantHTTPHook:      false,
		},
		{
			name:              "success_calls_hooks_with_expected_informers",
			solverOn:          true,
			wantErrIs:         nil,
			wantPluginNil:     false,
			wantReadinessHook: true,
			wantHTTPHook:      true,
			wantHTTPAddr:      HTTPAddr,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			h := makeHandle("https://localhost")

			clientFn := func(*rest.Config) (kubernetes.Interface, error) {
				if tt.clientErr != nil {
					return nil, tt.clientErr
				}
				return fake.NewSimpleClientset(), nil
			}

			var (
				readinessCalled bool
				readinessInfs   []cache.SharedIndexInformer
				httpCalled      bool
				gotHTTPAddr     string
			)

			withVar(t, &solverEnabled, func(*SharedState) bool { return tt.solverOn })
			withVar(t, &pluginReadinessStarter, func(_ *SharedState, _ context.Context, infs ...cache.SharedIndexInformer) {
				readinessCalled = true
				readinessInfs = append([]cache.SharedIndexInformer(nil), infs...)
			})
			withVar(t, &httpServerStarter, func(_ *SharedState, _ context.Context, addr string) {
				httpCalled = true
				gotHTTPAddr = addr
			})

			p, err := newFromHandle(ctx, runtime.Object(nil), clientFn, h, nil)

			if tt.wantErrIs == nil {
				if err != nil {
					t.Fatalf("err=%v, want nil", err)
				}
			} else if !errors.Is(err, tt.wantErrIs) {
				t.Fatalf("err=%v, want %v", err, tt.wantErrIs)
			}

			if tt.wantPluginNil {
				if p != nil {
					t.Fatalf("expected nil plugin, got %T", p)
				}
			} else {
				pl, ok := p.(*SharedState)
				if !ok {
					t.Fatalf("expected *SharedState, got %T", p)
				}
				if pl.Client == nil {
					t.Fatalf("Client must be initialized")
				}
				if pl.BlockedWhileActive == nil {
					t.Fatalf("BlockedWhileActive must be initialized")
				}
			}

			if readinessCalled != tt.wantReadinessHook {
				t.Fatalf("readinessCalled=%v, want %v", readinessCalled, tt.wantReadinessHook)
			}
			if httpCalled != tt.wantHTTPHook {
				t.Fatalf("httpCalled=%v, want %v", httpCalled, tt.wantHTTPHook)
			}
			if tt.wantHTTPHook && gotHTTPAddr != tt.wantHTTPAddr {
				t.Fatalf("http addr=%q, want %q", gotHTTPAddr, tt.wantHTTPAddr)
			}

			if tt.wantErrIs == nil {
				wantInfs := expectedInformers(h.Factory)
				if !reflect.DeepEqual(readinessInfs, wantInfs) {
					t.Fatalf("readiness informers mismatch:\n got:  %#v\n want: %#v", readinessInfs, wantInfs)
				}

				podsInf := h.Factory.Core().V1().Pods().Informer()
				if podsInf.GetIndexer().GetIndexers()[cache.NamespaceIndex] == nil {
					t.Fatalf("pod informer missing %q indexer", cache.NamespaceIndex)
				}
			}
		})
	}
}

// -------------------------
// Name
// -------------------------

func TestName(t *testing.T) {
	pl := &SharedState{}
	if got := pl.Name(); got != Name {
		t.Fatalf("Name() = %q, want %q", got, Name)
	}
}

// -------------------------
// New
// -------------------------

func TestNew_InvalidKubeConfig_NoHooks(t *testing.T) {
	ctx := context.Background()

	// Must be a framework.Handle; mkHandle provides that (via embedding).
	h := makeHandle("://bad")

	withVar(t, &solverEnabled, func(*SharedState) bool { return true })
	withVar(t, &pluginReadinessStarter, func(*SharedState, context.Context, ...cache.SharedIndexInformer) {
		t.Fatalf("pluginReadinessStarter must not be called on client config error")
	})
	withVar(t, &httpServerStarter, func(*SharedState, context.Context, string) {
		t.Fatalf("httpServerStarter must not be called on client config error")
	})

	p, err := New(ctx, nil, h)
	if err == nil {
		t.Fatalf("expected error, got nil (plugin=%T)", p)
	}
	if p != nil {
		t.Fatalf("expected nil plugin, got %T", p)
	}
}

func TestNew_Success_CallsHooks_AndStoresHandle(t *testing.T) {
	ctx := context.Background()
	h := makeHandle("https://localhost")

	var (
		readinessCalled bool
		readinessInfs   []cache.SharedIndexInformer
		httpCalled      bool
		gotHTTPAddr     string
	)

	withVar(t, &solverEnabled, func(*SharedState) bool { return true })
	withVar(t, &pluginReadinessStarter, func(_ *SharedState, _ context.Context, infs ...cache.SharedIndexInformer) {
		readinessCalled = true
		readinessInfs = append([]cache.SharedIndexInformer(nil), infs...)
	})
	withVar(t, &httpServerStarter, func(_ *SharedState, _ context.Context, addr string) {
		httpCalled = true
		gotHTTPAddr = addr
	})

	p, err := New(ctx, runtime.Object(nil), h)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	pl, ok := p.(*SharedState)
	if !ok {
		t.Fatalf("expected *SharedState, got %T", p)
	}

	// This will now type-check because h is a framework.Handle.
	if pl.Handle != h {
		t.Fatalf("pl.Handle was not set to the passed framework.Handle")
	}

	if pl.Client == nil {
		t.Fatalf("pl.Client must be initialized")
	}
	if pl.BlockedWhileActive == nil {
		t.Fatalf("BlockedWhileActive must be initialized")
	}

	if !readinessCalled {
		t.Fatalf("expected readiness hook to be called")
	}
	if !httpCalled {
		t.Fatalf("expected http hook to be called")
	}
	if gotHTTPAddr != HTTPAddr {
		t.Fatalf("http addr=%q, want %q", gotHTTPAddr, HTTPAddr)
	}

	wantInfs := expectedInformers(h.Factory)
	if !reflect.DeepEqual(readinessInfs, wantInfs) {
		t.Fatalf("readiness informers mismatch:\n got:  %#v\n want: %#v", readinessInfs, wantInfs)
	}

	podsInf := h.Factory.Core().V1().Pods().Informer()
	if podsInf.GetIndexer().GetIndexers()[cache.NamespaceIndex] == nil {
		t.Fatalf("pod informer missing %q indexer", cache.NamespaceIndex)
	}
}

// -------------------------
// Test Helpers
// -------------------------

func expectedInformers(f informers.SharedInformerFactory) []cache.SharedIndexInformer {
	return []cache.SharedIndexInformer{
		f.Core().V1().Pods().Informer(),
		f.Core().V1().Nodes().Informer(),
		f.Core().V1().ConfigMaps().Informer(),
		f.Apps().V1().ReplicaSets().Informer(),
		f.Apps().V1().StatefulSets().Informer(),
		f.Apps().V1().DaemonSets().Informer(),
		f.Batch().V1().Jobs().Informer(),
	}
}
