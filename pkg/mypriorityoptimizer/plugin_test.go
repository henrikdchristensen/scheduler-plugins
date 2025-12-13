// plugin_test.go
package mypriorityoptimizer

import (
	"context"
	"errors"
	"testing"

	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
)

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

func mkHandle(host string) *fakeHandle {
	cfg := &rest.Config{}
	if host != "" {
		cfg.Host = host
	}
	return &fakeHandle{
		cfg:     cfg,
		factory: informers.NewSharedInformerFactory(fake.NewSimpleClientset(), 0),
	}
}

func TestNewFromHandle_ClientError_NoHooks(t *testing.T) {
	ctx := context.Background()
	h := mkHandle("https://localhost")

	wantErr := errors.New("boom-client")
	clientFn := func(*rest.Config) (kubernetes.Interface, error) {
		return nil, wantErr
	}

	// Hooks must NOT run on client creation error.
	withVar(t, &pluginReadinessStarter, func(_ *SharedState, _ context.Context, _ ...cache.SharedIndexInformer) {
		t.Fatalf("pluginReadinessStarter must not be called on client error")
	})
	withVar(t, &httpServerStarter, func(_ *SharedState, _ context.Context, _ string) {
		t.Fatalf("httpServerStarter must not be called on client error")
	})

	p, err := newFromHandle(ctx, nil, clientFn, h, nil)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}
	if p != nil {
		t.Fatalf("expected nil plugin, got %#v", p)
	}
}

func TestNew_Scenarios(t *testing.T) {
	type want struct {
		errIs     error
		pluginNil bool
		readiness bool
		http      bool
		informers int
		httpAddr  string
	}

	cases := []struct {
		name     string
		solverOn bool
		want     want
	}{
		{
			name:     "no_solver_enabled",
			solverOn: false,
			want: want{
				errIs:     ErrNoSolverEnabled,
				pluginNil: true,
				readiness: false,
				http:      false,
			},
		},
		{
			name:     "success_calls_hooks",
			solverOn: true,
			want: want{
				errIs:     nil,
				pluginNil: false,
				readiness: true,
				http:      true,
				informers: 7,
				httpAddr:  HTTPAddr,
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			h := mkHandle("https://localhost")

			readinessCalled := false
			readinessCount := 0
			httpCalled := false
			httpAddr := ""

			withVar(t, &solverEnabled, func(_ *SharedState) bool { return tc.solverOn })
			withVar(t, &pluginReadinessStarter, func(_ *SharedState, _ context.Context, infs ...cache.SharedIndexInformer) {
				readinessCalled = true
				readinessCount = len(infs)
			})
			withVar(t, &httpServerStarter, func(_ *SharedState, _ context.Context, addr string) {
				httpCalled = true
				httpAddr = addr
			})

			p, err := New(ctx, nil, h)

			if tc.want.errIs == nil {
				if err != nil {
					t.Fatalf("err=%v, want nil", err)
				}
			} else if !errors.Is(err, tc.want.errIs) {
				t.Fatalf("err=%v, want %v", err, tc.want.errIs)
			}

			if tc.want.pluginNil {
				if p != nil {
					t.Fatalf("expected nil plugin, got %#v", p)
				}
			} else {
				if p == nil {
					t.Fatalf("expected plugin, got nil")
				}
				pl, ok := p.(*SharedState)
				if !ok {
					t.Fatalf("expected *SharedState, got %T", p)
				}
				if pl.BlockedWhileActive == nil {
					t.Fatalf("BlockedWhileActive must be initialized")
				}
			}

			if readinessCalled != tc.want.readiness {
				t.Fatalf("readinessCalled=%v, want %v", readinessCalled, tc.want.readiness)
			}
			if httpCalled != tc.want.http {
				t.Fatalf("httpCalled=%v, want %v", httpCalled, tc.want.http)
			}
			if tc.want.readiness && readinessCount != tc.want.informers {
				t.Fatalf("readiness informers=%d, want %d", readinessCount, tc.want.informers)
			}
			if tc.want.http && httpAddr != tc.want.httpAddr {
				t.Fatalf("http addr=%q, want %q", httpAddr, tc.want.httpAddr)
			}
		})
	}
}
