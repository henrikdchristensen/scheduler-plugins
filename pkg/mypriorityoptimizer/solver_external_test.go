// solver_external_test.go
package mypriorityoptimizer

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// -------------------------
// Test Helpers
// -------------------------

// withReadAllStdout temporarily replaces readAllStdout for the duration of the test.
func withReadAllStdout(t *testing.T, f func(r io.Reader) ([]byte, error)) {
	t.Helper()
	orig := readAllStdout
	readAllStdout = f
	t.Cleanup(func() { readAllStdout = orig })
}

// withStreamSolverStderr temporarily replaces streamSolverStderrFn for the duration of the test.
func withStreamSolverStderr(t *testing.T, f func(r io.Reader) error) {
	t.Helper()
	withVar(t, &streamSolverStderrFn, f)
}

// -------------------------
// runSolverExternal
// -------------------------

func TestRunSolverExternal_Success_AndScansStderr(t *testing.T) {
	requireBash(t)

	// Emit two stderr lines so the goroutine's Scan() loop body is exercised.
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

func TestRunSolverExternal_WaitError_NonZeroExit(t *testing.T) {
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

func TestRunSolverExternal_ReadStdoutError(t *testing.T) {
	requireBash(t)

	withReadAllStdout(t, func(r io.Reader) ([]byte, error) {
		return nil, fmt.Errorf("forced read error")
	})

	// Must be a script that exits quickly so the internal `_ = cmd.Wait()` does not hang.
	script := `#!/usr/bin/env bash
cat >/dev/null
printf '{"status":"OPTIMAL"}'
`
	out, err := runBashSolver(t, script, []byte(`{}`))
	if err == nil || !strings.Contains(err.Error(), "read solver stdout") {
		t.Fatalf("err=%v, want contains %q", err, "read solver stdout")
	}
	if out != nil {
		t.Fatalf("out=%q, want nil on error", string(out))
	}
}

func TestRunSolverExternal_StdoutPipeError(t *testing.T) {
	requireNonWindows(t)

	// Force StdoutPipe() to fail by pre-setting cmd.Stdout.
	withExecCommandContext(t, func(ctx context.Context, name string, args ...string) *exec.Cmd {
		cmd := exec.CommandContext(ctx, name, args...)
		cmd.Stdout = &bytes.Buffer{}
		return cmd
	})

	pl := &SharedState{}
	ctx, cancel := testCtx(t)
	defer cancel()

	out, err := pl.runSolverExternal(ctx, []byte(`{}`), "bash", "/does/not/matter.sh")
	if err == nil || !strings.Contains(err.Error(), "stdout pipe") {
		t.Fatalf("err=%v, want contains %q", err, "stdout pipe")
	}
	if out != nil {
		t.Fatalf("out=%q, want nil on error", string(out))
	}
}

func TestRunSolverExternal_StderrPipeError(t *testing.T) {
	requireNonWindows(t)

	// Force StderrPipe() to fail by pre-setting cmd.Stderr.
	withExecCommandContext(t, func(ctx context.Context, name string, args ...string) *exec.Cmd {
		cmd := exec.CommandContext(ctx, name, args...)
		cmd.Stderr = &bytes.Buffer{}
		return cmd
	})

	pl := &SharedState{}
	ctx, cancel := testCtx(t)
	defer cancel()

	out, err := pl.runSolverExternal(ctx, []byte(`{}`), "bash", "/does/not/matter.sh")
	if err == nil || !strings.Contains(err.Error(), "stderr pipe") {
		t.Fatalf("err=%v, want contains %q", err, "stderr pipe")
	}
	if out != nil {
		t.Fatalf("out=%q, want nil on error", string(out))
	}
}

func TestRunSolverExternal_StartError(t *testing.T) {
	requireNonWindows(t)

	pl := &SharedState{}
	ctx, cancel := testCtx(t)
	defer cancel()

	// Non-existent binary triggers cmd.Start() error branch.
	out, err := pl.runSolverExternal(ctx, []byte(`{}`), "definitely-not-a-real-executable-xyz", "unused")
	if err == nil || !strings.Contains(err.Error(), "solver start") {
		t.Fatalf("err=%v, want contains %q", err, "solver start")
	}
	if out != nil {
		t.Fatalf("out=%q, want nil on error", string(out))
	}
}

func TestRunSolverExternal_ContextDeadlineExceeded(t *testing.T) {
	requireBash(t)

	script := `#!/usr/bin/env bash
cat >/dev/null
sleep 5
echo "never"
`
	pl := &SharedState{}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	out, err := pl.runSolverExternal(ctx, []byte(`{}`), "bash", writeFakeSolverScript(t, t.TempDir(), script))
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
