// http_server_test.go
package mypriorityoptimizer

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
)

// -------------------------
// Test Helpers
// -------------------------

func httpCall(t *testing.T, h http.HandlerFunc, method, path string) *httptest.ResponseRecorder {
	t.Helper()
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(method, path, nil)
	h(rr, req)
	return rr
}

func mustStatus(t *testing.T, rr *httptest.ResponseRecorder, want int) {
	t.Helper()
	if rr.Code != want {
		t.Fatalf("status=%d want=%d body=%q", rr.Code, want, rr.Body.String())
	}
}

func mustJSON[T any](t *testing.T, rr *httptest.ResponseRecorder, wantStatus int) T {
	t.Helper()
	mustStatus(t, rr, wantStatus)

	ct := rr.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Fatalf("Content-Type=%q want=%q body=%q", ct, "application/json", rr.Body.String())
	}

	var out T
	if err := json.Unmarshal(rr.Body.Bytes(), &out); err != nil {
		t.Fatalf("invalid JSON: %v body=%q", err, rr.Body.String())
	}
	return out
}

func withRunOptFlow(t *testing.T, fn func(*SharedState, context.Context) (*Plan, *SolverScore, string, *SolverResult, []SolverResult, error), body func()) {
	t.Helper()
	old := runOptFlow
	runOptFlow = fn
	t.Cleanup(func() { runOptFlow = old })
	body()
}

func reserveAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen :0: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	return addr
}

func waitHTTP(t *testing.T, url string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)

	client := &http.Client{Timeout: 200 * time.Millisecond}
	var lastErr error

	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			return
		}
		lastErr = err
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("server never became reachable at %s (last err: %v)", url, lastErr)
}

// -------------------------
// /healthz
// -------------------------

func TestHTTP_Healthz(t *testing.T) {
	tests := []struct {
		name     string
		ready    bool
		wantCode int
		wantBody string
	}{
		{"warming", false, http.StatusServiceUnavailable, ""},
		{"ready", true, http.StatusOK, "ok"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &SharedState{}
			pl.PluginReady.Store(tt.ready)

			rr := httpCall(t, pl.httpHealthzHandler, http.MethodGet, "/healthz")
			mustStatus(t, rr, tt.wantCode)
			if tt.wantBody != "" && rr.Body.String() != tt.wantBody {
				t.Fatalf("body=%q want=%q", rr.Body.String(), tt.wantBody)
			}
		})
	}
}

// -------------------------
// /active
// -------------------------

func TestHTTP_Active(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		active     bool
		wantCode   int
		wantActive *bool // nil => no JSON expected
	}{
		{"method not allowed", http.MethodPost, true, http.StatusMethodNotAllowed, nil},
		{"ok true", http.MethodGet, true, http.StatusOK, ptr(true)},
		{"ok false", http.MethodGet, false, http.StatusOK, ptr(false)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &SharedState{}
			pl.ActivePlanInProgress.Store(tt.active)

			rr := httpCall(t, pl.httpActiveHandler, tt.method, "/active")
			if tt.wantActive == nil {
				mustStatus(t, rr, tt.wantCode)
				return
			}

			resp := mustJSON[HttpResponse](t, rr, tt.wantCode)
			if resp.Active != *tt.wantActive {
				t.Fatalf("Active=%v want=%v", resp.Active, *tt.wantActive)
			}
		})
	}
}

// -------------------------
// /solve
// -------------------------

func TestHTTP_Solve(t *testing.T) {
	p1 := pod("ns", "p1")
	p2 := pod("ns", "p2")
	p3 := pod("ns", "p3", onNode("n1"))
	fpl := &fakePodLister{store: map[string]map[string]*v1.Pod{
		"ns": {"p1": p1, "p2": p2, "p3": p3},
	}}

	attempts := []SolverResult{
		{Name: "solverA", Status: "FEASIBLE"},
		{Name: "solverB", Status: "OPTIMAL"},
	}

	type tc struct {
		name       string
		method     string
		ready      bool
		active     bool
		runErr     error
		wantCode   int
		wantStatus string
		wantErrSet bool
	}

	tests := []tc{
		{
			name:     "method not allowed",
			method:   http.MethodGet,
			ready:    true,
			active:   true,
			wantCode: http.StatusMethodNotAllowed,
		},
		{
			name:       "not ready",
			method:     http.MethodPost,
			ready:      false,
			active:     true, // should be reflected back
			wantCode:   http.StatusPreconditionFailed,
			wantStatus: "not-ready",
		},
		{
			name:       "ok",
			method:     http.MethodPost,
			ready:      true,
			active:     true,
			runErr:     nil,
			wantCode:   http.StatusOK,
			wantStatus: "ok",
			wantErrSet: false,
		},
		{
			name:       "busy",
			method:     http.MethodPost,
			ready:      true,
			active:     true,
			runErr:     ErrActiveInProgress,
			wantCode:   http.StatusOK,
			wantStatus: "busy",
			wantErrSet: true,
		},
		{
			name:       "noop",
			method:     http.MethodPost,
			ready:      true,
			active:     true,
			runErr:     ErrNoPendingPods,
			wantCode:   http.StatusOK,
			wantStatus: "noop",
			wantErrSet: true,
		},
		{
			name:       "error",
			method:     http.MethodPost,
			ready:      true,
			active:     true,
			runErr:     fmt.Errorf("boom"),
			wantCode:   http.StatusOK,
			wantStatus: "error",
			wantErrSet: true,
		},
	}

	withPodLister(fpl, func() {
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				pl := &SharedState{}
				pl.PluginReady.Store(tt.ready)
				pl.ActivePlanInProgress.Store(tt.active)

				// Ensure runOptFlow is never called on early exits.
				if tt.method != http.MethodPost || !tt.ready {
					withRunOptFlow(t, func(*SharedState, context.Context) (*Plan, *SolverScore, string, *SolverResult, []SolverResult, error) {
						t.Fatalf("runOptFlow must not be called for method=%s ready=%v", tt.method, tt.ready)
						return nil, nil, "", nil, nil, nil
					}, func() {
						rr := httpCall(t, pl.httpSolveHandler, tt.method, "/solve")
						mustStatus(t, rr, tt.wantCode)
						if tt.wantStatus == "not-ready" {
							resp := mustJSON[HttpResponse](t, rr, tt.wantCode)
							if resp.Status != "not-ready" {
								t.Fatalf("Status=%q want=%q", resp.Status, "not-ready")
							}
							if resp.Active != tt.active {
								t.Fatalf("Active=%v want=%v", resp.Active, tt.active)
							}
							if resp.DurationMs < 0 {
								t.Fatalf("DurationMs=%d want>=0", resp.DurationMs)
							}
						}
					})
					return
				}

				// Normal flow with runOptFlow mocked.
				withRunOptFlow(t, func(*SharedState, context.Context) (*Plan, *SolverScore, string, *SolverResult, []SolverResult, error) {
					return nil, &SolverScore{Evicted: 1}, "solverB", nil, attempts, tt.runErr
				}, func() {
					rr := httpCall(t, pl.httpSolveHandler, tt.method, "/solve")
					resp := mustJSON[HttpResponse](t, rr, tt.wantCode)

					if resp.Status != tt.wantStatus {
						t.Fatalf("Status=%q want=%q", resp.Status, tt.wantStatus)
					}
					if resp.Active != tt.active {
						t.Fatalf("Active=%v want=%v", resp.Active, tt.active)
					}
					if resp.PendingBefore != 2 {
						t.Fatalf("PendingBefore=%d want=2", resp.PendingBefore)
					}
					if tt.wantErrSet && resp.Error == "" {
						t.Fatalf("expected Error to be populated (err=%v)", tt.runErr)
					}
					if !tt.wantErrSet && resp.Error != "" {
						t.Fatalf("expected Error empty, got %q", resp.Error)
					}
				})
			})
		}
	})
}

// -------------------------
// startHttpServer
// -------------------------

func TestStartHttpServer_ShutsDownOnContextCancel(t *testing.T) {
	pl := &SharedState{}
	pl.PluginReady.Store(true) // makes /healthz return 200 once reachable

	addr := reserveAddr(t)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})

	go func() {
		pl.startHttpServer(ctx, addr)
		close(done)
	}()

	// Wait until the server is reachable (no sleeps/guesses).
	waitHTTP(t, "http://"+addr+"/healthz", 2*time.Second)

	cancel()

	select {
	case <-done:
		// ok
	case <-time.After(2 * time.Second):
		t.Fatalf("startHttpServer did not shut down after context cancel")
	}
}

func TestStartHttpServer_ListenAndServeError(t *testing.T) {
	pl := &SharedState{}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // avoid leaking shutdown goroutine

	// Invalid port -> ListenAndServe returns immediately with an error
	pl.startHttpServer(ctx, "127.0.0.1:-1")
}

// -------------------------
// writeHttpJson
// -------------------------

func TestWriteHttpJson(t *testing.T) {
	rr := httptest.NewRecorder()

	type payload struct {
		A string `json:"a"`
		N int    `json:"n"`
	}
	want := payload{A: "x", N: 7}

	writeHttpJson(rr, http.StatusTeapot, want)

	mustStatus(t, rr, http.StatusTeapot)
	if ct := rr.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("Content-Type=%q want=%q", ct, "application/json")
	}

	var got payload
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("invalid JSON: %v body=%q", err, rr.Body.String())
	}
	if got != want {
		t.Fatalf("decoded=%#v want=%#v", got, want)
	}

	// ensure trailing newline
	if !strings.HasSuffix(rr.Body.String(), "\n") {
		t.Fatalf("expected trailing newline from Encoder, got %q", rr.Body.String())
	}
}
