// pkg/mypriorityoptimizer/plugin_test.go
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
// Test Helpers
// -------------------------

type fakeHandleDeps struct {
	cfg     *rest.Config
	factory informers.SharedInformerFactory
}

func (f *fakeHandleDeps) KubeConfig() *rest.Config { return f.cfg }
func (f *fakeHandleDeps) SharedInformerFactory() informers.SharedInformerFactory {
	return f.factory
}

func mkHandleDeps(host string) *fakeHandleDeps {
	cfg := &rest.Config{Host: host}
	return &fakeHandleDeps{
		cfg:     cfg,
		factory: informers.NewSharedInformerFactory(fake.NewSimpleClientset(), 0),
	}
}

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

// -------------------------
// newFromHandle
// -------------------------

func TestNewFromHandle(t *testing.T) {
	type want struct {
		errIs     error
		pluginNil bool

		readinessCalled bool
		httpCalled      bool
		httpAddr        string

		wantInformers []cache.SharedIndexInformer
	}

	wantClientErr := errors.New("boom-client")

	tests := []struct {
		name string

		// inputs
		clientErr error
		solverOn  bool

		// expected
		want want
	}{
		{
			name:      "client_error_propagated_no_hooks",
			clientErr: wantClientErr,
			solverOn:  true, // irrelevant: should fail before solver check
			want: want{
				errIs:           wantClientErr,
				pluginNil:       true,
				readinessCalled: false,
				httpCalled:      false,
			},
		},
		{
			name:     "no_solver_enabled_no_hooks",
			solverOn: false,
			want: want{
				errIs:           ErrNoSolverEnabled,
				pluginNil:       true,
				readinessCalled: false,
				httpCalled:      false,
			},
		},
		{
			name:     "success_calls_hooks_with_expected_informers",
			solverOn: true,
			want: want{
				errIs:           nil,
				pluginNil:       false,
				readinessCalled: true,
				httpCalled:      true,
				httpAddr:        HTTPAddr,
				// wantInformers filled per-test from handle factory
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			h := mkHandleDeps("https://localhost")

			// client builder
			clientFn := func(*rest.Config) (kubernetes.Interface, error) {
				if tt.clientErr != nil {
					return nil, tt.clientErr
				}
				return fake.NewSimpleClientset(), nil
			}

			// capture hook calls
			var (
				readinessCalled bool
				readinessInfs   []cache.SharedIndexInformer
				httpCalled      bool
				gotHTTPAddr     string
			)

			withVar(t, &solverEnabled, func(_ *SharedState) bool { return tt.solverOn })
			withVar(t, &pluginReadinessStarter, func(_ *SharedState, _ context.Context, infs ...cache.SharedIndexInformer) {
				readinessCalled = true
				readinessInfs = append([]cache.SharedIndexInformer(nil), infs...)
			})
			withVar(t, &httpServerStarter, func(_ *SharedState, _ context.Context, addr string) {
				httpCalled = true
				gotHTTPAddr = addr
			})

			// run
			p, err := newFromHandle(ctx, runtime.Object(nil), clientFn, h, nil)

			// error expectation
			if tt.want.errIs == nil {
				if err != nil {
					t.Fatalf("err=%v, want nil", err)
				}
			} else if !errors.Is(err, tt.want.errIs) {
				t.Fatalf("err=%v, want %v", err, tt.want.errIs)
			}

			// plugin expectation
			if tt.want.pluginNil {
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

			// hooks
			if readinessCalled != tt.want.readinessCalled {
				t.Fatalf("readinessCalled=%v, want %v", readinessCalled, tt.want.readinessCalled)
			}
			if httpCalled != tt.want.httpCalled {
				t.Fatalf("httpCalled=%v, want %v", httpCalled, tt.want.httpCalled)
			}
			if tt.want.httpCalled && gotHTTPAddr != tt.want.httpAddr {
				t.Fatalf("http addr=%q, want %q", gotHTTPAddr, tt.want.httpAddr)
			}

			// informer contract (only on success)
			if tt.want.errIs == nil {
				wantInfs := expectedInformers(h.factory)
				if !reflect.DeepEqual(readinessInfs, wantInfs) {
					t.Fatalf("readiness informers mismatch:\n got:  %#v\n want: %#v", readinessInfs, wantInfs)
				}

				// Also assert we added the namespace indexer to the pod informer.
				podsInf := h.factory.Core().V1().Pods().Informer()
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
