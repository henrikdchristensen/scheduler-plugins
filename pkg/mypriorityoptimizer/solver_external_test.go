// solver_external_test.gos
package mypriorityoptimizer

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// -------------------------
// runSolverExternal
// -------------------------

func TestRunSolverExternal_Success(t *testing.T) {
	requireBash(t)

	// Script that writes to stderr and stdout.
	script := `#!/usr/bin/env bash
cat >/dev/null
echo "log line 1" 1>&2
echo "log line 2" 1>&2
printf '{"status":"OPTIMAL"}'
`
	out, err := runBashSolver(t, script, []byte(`{"dummy":"input"}`))
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got := strings.TrimSpace(string(out)); got != `{"status":"OPTIMAL"}` {
		t.Fatalf("out=%q, want %q", got, `{"status":"OPTIMAL"}`)
	}
}

const fastSolverOutput = `{"status":"OPTIMAL"}`

func TestFastSolverHelper(t *testing.T) {
	switch os.Getenv("GO_TEST_FAST_SOLVER") {
	case "stderr":
		fmt.Fprint(os.Stderr, strings.Repeat("x", 2*1024*1024), "\n")
	case "timeout":
		time.Sleep(5 * time.Second)
	case "stdout":
	default:
		return
	}
	fmt.Fprint(os.Stdout, fastSolverOutput)
	os.Exit(0)
}

func useFastSolverHelper(t *testing.T, mode string) {
	t.Helper()
	withExecCommandContext(t, func(ctx context.Context, _ string, _ ...string) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestFastSolverHelper$")
		cmd.Env = append(os.Environ(), "GO_TEST_FAST_SOLVER="+mode, "GORACE=atexit_sleep_ms=0")
		return cmd
	})
}

func TestRunSolverExternal_CapturesOutputFromFastProcess(t *testing.T) {
	useFastSolverHelper(t, "stdout")
	pl := &SharedState{}
	for i := 0; i < 8; i++ {
		ctx, cancel := testCtx(t)
		out, err := pl.runSolverExternal(ctx, nil, "", "")
		cancel()
		if err != nil || string(out) != fastSolverOutput {
			t.Fatalf("run %d: output=%q, err=%v; want %q", i, out, err, fastSolverOutput)
		}
	}
}

func TestRunSolverExternal_DrainsStderrAfterScannerError(t *testing.T) {
	useFastSolverHelper(t, "stderr")
	ctx, cancel := testCtx(t)
	defer cancel()

	out, err := (&SharedState{}).runSolverExternal(ctx, nil, "", "")
	if err != nil || string(out) != fastSolverOutput {
		t.Fatalf("output=%q, err=%v; want %q", out, err, fastSolverOutput)
	}
}

func TestRunSolverExternal_WaitError(t *testing.T) {
	requireBash(t)

	script := `#!/usr/bin/env bash
cat >/dev/null
printf '{"status":"OPTIMAL"}'
exit 3
`
	out, err := runBashSolver(t, script, []byte(`{}`))
	if err == nil || !strings.Contains(err.Error(), "solver run") {
		t.Fatalf("err=%v, want contains %q", err, "solver run")
	}
	if out != nil {
		t.Fatalf("out=%q, want nil on error", string(out))
	}
}

func TestRunSolverExternal_ForwardsStdin(t *testing.T) {
	requireBash(t)

	script := `#!/usr/bin/env bash
# Echo stdin back to stdout
cat
`
	payload := []byte("hello solver\n")
	out, err := runBashSolver(t, script, payload)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if string(out) != string(payload) {
		t.Fatalf("out=%q want %q", string(out), string(payload))
	}
}

func TestRunSolverExternal_StartError(t *testing.T) {
	requireNonWindows(t)

	pl := &SharedState{}
	ctx, cancel := testCtx(t)
	defer cancel()

	// Non-existent binary triggers cmd.Start() error branch.
	out, err := pl.runSolverExternal(ctx, []byte(`{}`), "not-a-real-executable-xyz", "unused")
	if err == nil || !strings.Contains(err.Error(), "solver start") {
		t.Fatalf("err=%v, want contains %q", err, "solver start")
	}
	if out != nil {
		t.Fatalf("out=%q, want nil on error", string(out))
	}
}

func TestRunSolverExternal_ContextDeadlineExceeded(t *testing.T) {
	useFastSolverHelper(t, "timeout")
	pl := &SharedState{}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	out, err := pl.runSolverExternal(ctx, nil, "", "")
	if err == nil {
		t.Fatalf("expected error, got nil (out=%q)", string(out))
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err=%v, want deadline exceeded (wrapped)", err)
	}
	if out != nil {
		t.Fatalf("out=%q, want nil on error", string(out))
	}
}

// -------------------------
// streamSolverStderr
// -------------------------

func TestStreamSolverStderr_ScansLines(t *testing.T) {
	err := streamSolverStderr(strings.NewReader("line1\nline2\n"))
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
}

func TestStreamSolverStderr_TokenTooLong(t *testing.T) {
	tooBig := bytes.Repeat([]byte("a"), 1024*1024+1) // > 1MB token
	err := streamSolverStderr(bytes.NewReader(tooBig))
	if err == nil {
		t.Fatalf("expected scanner error, got nil")
	}
	if !errors.Is(err, bufio.ErrTooLong) {
		t.Fatalf("err=%v, want bufio.ErrTooLong", err)
	}
}
