// solver_external.go
package mypriorityoptimizer

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"

	"k8s.io/klog/v2"
)

// -------------------------
// Test Hooks
// -------------------------

var (
	execCommandContext   = exec.CommandContext
	readAllStdout        = io.ReadAll
	streamSolverStderrFn = streamSolverStderr
)

// -------------------------
// runSolverExternal
// -------------------------

// runSolverExternal is the generic external solver runner.
// CHECKED
func (pl *SharedState) runSolverExternal(
	ctx context.Context,
	payload []byte,
	binary string,
	scriptPath string,
) ([]byte, error) {
	cmd := execCommandContext(ctx, binary, scriptPath)
	cmd.Stdin = bytes.NewReader(payload)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("stderr pipe: %w", err)
	}

	// Stream solver logs from stderr (best-effort).
	go func() { _ = streamSolverStderrFn(stderr) }()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("solver start: %w", err)
	}

	type readRes struct {
		b   []byte
		err error
	}
	stdoutCh := make(chan readRes, 1)
	go func() {
		b, e := readAllStdout(stdout)
		stdoutCh <- readRes{b: b, err: e}
	}()

	waitCh := make(chan error, 1)
	go func() { waitCh <- cmd.Wait() }()

	select {
	case <-ctx.Done():
		// Ensure the process is gone quickly.
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		// Wait for goroutines to finish.
		<-waitCh
		<-stdoutCh
		return nil, fmt.Errorf("solver context: %w", ctx.Err())

	case werr := <-waitCh:
		r := <-stdoutCh
		if r.err != nil {
			return nil, fmt.Errorf("read solver stdout: %w", r.err)
		}
		if werr != nil {
			return nil, fmt.Errorf("solver run: %w", werr)
		}
		return r.b, nil
	}
}

// -------------------------
// streamSolverStderr
// -------------------------

// streamSolverStderr scans stderr and logs it. Returns scanner error (if any).
// CHECKED
func streamSolverStderr(r io.Reader) error {
	s := bufio.NewScanner(r)
	buf := make([]byte, 0, 256*1024) // 256KB initial buffer
	s.Buffer(buf, 1024*1024)         // 1MB max token size
	for s.Scan() {
		klog.V(MyV).Info("solver: " + s.Text())
	}
	if err := s.Err(); err != nil {
		klog.Info("solver scan failed: " + err.Error())
		return err
	}
	return nil
}
