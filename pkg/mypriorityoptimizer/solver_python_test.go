// solver_python_test.go
package mypriorityoptimizer

import (
	"math"
	"strings"
	"testing"
)

// -------------------------
// runPythonSolver
// -------------------------

func TestRunPythonSolver(t *testing.T) {
	requireBash(t)

	cases := []struct {
		name       string
		script     string
		opts       PythonSolverOptions
		wantStatus string
		wantErrSub []string
	}{
		{
			name: "success",
			script: `#!/usr/bin/env bash
cat >/dev/null
printf '{"status":"OPTIMAL","placements":[],"evictions":[]}'
`,
			wantStatus: "OPTIMAL",
		},
		{
			name: "success_with_phases", // covers SolvePhases loop
			script: `#!/usr/bin/env bash
cat >/dev/null
printf '{"status":"OPTIMAL","durationMs":7,"placements":[],"evictions":[],"phases":[{"tier":1,"stage":"presolve","status":"ok","durationMs":1,"relativeGap":0.10}]}'
`,
			wantStatus: "OPTIMAL",
		},
		{
			name: "invalid_json",
			script: `#!/usr/bin/env bash
cat >/dev/null
echo 'not-json'
`,
			wantErrSub: []string{"decode", "output"},
		},
		{
			name: "external_error", // covers runSolverExternal error wrap
			script: `#!/usr/bin/env bash
cat >/dev/null
echo 'boom' >&2
exit 42
`,
			wantErrSub: []string{"python solver external"},
		},
		{
			name:       "marshal_error", // covers json.Marshal error
			opts:       PythonSolverOptions{GapLimit: math.NaN()},
			wantErrSub: []string{"marshal", "payload"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmpDir := t.TempDir()

			// Only write/override script when we actually need to run the process.
			if tc.script != "" {
				scriptPath := writeFakeSolverScript(t, tmpDir, tc.script)
				withVar(t, &solverBinary, "bash")
				withVar(t, &solverScriptPath, scriptPath)
			}

			pl := &SharedState{}
			ctx, cancel := testCtx(t)
			defer cancel()

			out, err := pl.runPythonSolver(ctx, SolverInput{}, tc.opts)

			if len(tc.wantErrSub) > 0 {
				if err == nil {
					t.Fatalf("expected error, got nil (out=%#v)", out)
				}
				for _, sub := range tc.wantErrSub {
					if !strings.Contains(err.Error(), sub) {
						t.Fatalf("err=%v, want contains %q", err, sub)
					}
				}
				if out != nil {
					t.Fatalf("expected nil out on error, got %#v", out)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if out == nil {
				t.Fatalf("expected non-nil out")
			}
			if out.Status != tc.wantStatus {
				t.Fatalf("Status=%q, want %q", out.Status, tc.wantStatus)
			}
		})
	}
}
