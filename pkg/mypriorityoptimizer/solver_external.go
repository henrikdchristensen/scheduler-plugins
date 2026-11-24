// solver_external.go
package mypriorityoptimizer

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"time"

	"k8s.io/klog/v2"
)

// -------------------------
// runSolverExternal
// -------------------------

// runSolverExternal is the generic external solver runner.
func (pl *SharedState) runSolverExternal(
	ctx context.Context,
	payload []byte,
	binary string,
	scriptPath string,
) ([]byte, error) {
	cmd := execCommandContext(ctx, binary, scriptPath)
	cmd.Stdin = bytes.NewReader(payload)

	// Let os/exec finish reading stdout and stderr before closing the pipes.
	var stdout bytes.Buffer
	stderrReader, stderrWriter := io.Pipe()
	defer stderrReader.Close()
	defer stderrWriter.Close()
	cmd.Stdout = &stdout
	cmd.Stderr = stderrWriter
	// Bound cleanup if a descendant keeps an output descriptor open.
	cmd.WaitDelay = 250 * time.Millisecond

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("solver start: %w", err)
	}

	stderrDone := make(chan struct{})
	go func() {
		defer close(stderrDone)
		if err := streamSolverStderr(stderrReader); err != nil {
			// A scanner error must not stop draining the child's stderr.
			_, _ = io.Copy(io.Discard, stderrReader)
		}
	}()

	waitErr := cmd.Wait()
	_ = stderrWriter.Close()
	<-stderrDone
	if waitErr != nil {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("solver context: %w", err)
		}
		return nil, fmt.Errorf("solver run: %w", waitErr)
	}
	return stdout.Bytes(), nil
}

// -------------------------
// streamSolverStderr
// -------------------------

// streamSolverStderr scans stderr and logs it. Returns scanner error (if any).
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

// -------------------------
// Test Hooks
// -------------------------

var execCommandContext = exec.CommandContext
