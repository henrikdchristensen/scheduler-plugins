// mode_helpers_test.go
package mypriorityoptimizer

import (
	"fmt"
	"testing"
)

// -------------------------
// Mode Predicates
// -------------------------

func TestModePredicates(t *testing.T) {
	modes := []ModeType{
		ModePerPod,
		ModePeriodic,
		ModeInterlude,
		ModeManual,
		ModeManualBlocking,
		ModeType(999),
	}
	syncFlags := []bool{true, false}

	for _, mode := range modes {
		for _, synch := range syncFlags {
			name := fmt.Sprintf("mode=%s synch=%v", mode.String(), synch)
			t.Run(name, func(t *testing.T) {
				withMode(mode, synch, func() {
					// Expected behavior derived directly from implementation contracts.
					wantPerPod := mode == ModePerPod
					wantManualBlocking := mode == ModeManualBlocking
					wantAsync := (mode != ModePerPod) && !synch

					wantSyncStr := "Synch"
					if wantAsync {
						wantSyncStr = "Asynch"
					}
					wantCombined := mode.String() + "/" + wantSyncStr

					if got := isPerPodMode(); got != wantPerPod {
						t.Fatalf("isPerPodMode()=%v want %v", got, wantPerPod)
					}
					if got := isManualBlockingMode(); got != wantManualBlocking {
						t.Fatalf("isManualBlockingMode()=%v want %v", got, wantManualBlocking)
					}
					if got := isAsyncSolving(); got != wantAsync {
						t.Fatalf("isAsyncSolving()=%v want %v", got, wantAsync)
					}
					if got := getSyncAsString(); got != wantSyncStr {
						t.Fatalf("getSyncAsString()=%q want %q", got, wantSyncStr)
					}
					if got := getModeCombinedAsString(); got != wantCombined {
						t.Fatalf("getModeCombinedAsString()=%q want %q", got, wantCombined)
					}
				})
			})
		}
	}
}

// -------------------------
// ModeType String()
// -------------------------

func TestModeType_String(t *testing.T) {
	tests := []struct {
		in   ModeType
		want string
	}{
		{ModePerPod, "PerPod"},
		{ModePeriodic, "Periodic"},
		{ModeInterlude, "Interlude"},
		{ModeManual, "Manual"},
		{ModeManualBlocking, "ManualBlocking"},
		{ModeType(999), "Periodic"}, // default branch
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.in.String(); got != tt.want {
				t.Fatalf("ModeType(%d).String()=%q want %q", tt.in, got, tt.want)
			}
		})
	}
}
