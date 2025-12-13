// solver_external_test.go
package mypriorityoptimizer

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"
)

// -------------------------
// Test Helpers
// -------------------------

// testCtx returns a context that will be cancelled before the test's deadline
// (if one exists), otherwise uses a short timeout.
func testCtx(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()

	if dl, ok := t.Deadline(); ok {
		return context.WithDeadline(context.Background(), dl.Add(-200*time.Millisecond))
	}
	return context.WithTimeout(context.Background(), 1*time.Second)
}

// requireNonWindows skips the test if running on Windows.
func requireNonWindows(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("skipping on windows (shell scripts / /dev/zero assumptions)")
	}
}

// requireBash skips the test if bash is not available.
func requireBash(t *testing.T) {
	t.Helper()
	requireNonWindows(t)
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not found on PATH")
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

// withExecCommandContext temporarily replaces execCommandContext for the duration of the test.
func withExecCommandContext(t *testing.T, f func(ctx context.Context, name string, args ...string) *exec.Cmd) {
	t.Helper()
	orig := execCommandContext
	execCommandContext = f
	t.Cleanup(func() { execCommandContext = orig })
}

// withReadAllStdout temporarily replaces readAllStdout for the duration of the test.
func withReadAllStdout(t *testing.T, f func(r io.Reader) ([]byte, error)) {
	t.Helper()
	orig := readAllStdout
	readAllStdout = f
	t.Cleanup(func() { readAllStdout = orig })
}

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

func TestStreamSolverStderr_ScansLines(t *testing.T) {
	// Covers: for s.Scan() loop body
	err := streamSolverStderr(strings.NewReader("line1\nline2\n"))
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
}

func TestStreamSolverStderr_TokenTooLong(t *testing.T) {
	// Covers: s.Err() branch with bufio.Scanner ErrTooLong
	tooBig := bytes.Repeat([]byte("a"), 1024*1024+1) // 1MB + 1
	err := streamSolverStderr(bytes.NewReader(tooBig))
	if err == nil {
		t.Fatalf("expected scanner error, got nil")
	}
	if !strings.Contains(err.Error(), "token too long") {
		t.Fatalf("err=%v, want contains %q", err, "token too long")
	}
}
