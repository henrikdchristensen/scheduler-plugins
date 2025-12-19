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
	blockingFlags := []bool{true, false}

	for _, mode := range modes {
		for _, blocking := range blockingFlags {
			name := fmt.Sprintf("mode=%s blocking=%v", mode.String(), blocking)
			t.Run(name, func(t *testing.T) {
				withMode(mode, blocking, func() {
					// Expected behavior derived directly from implementation contracts.
					wantPerPod := mode == ModePerPod
					wantManualBlocking := mode == ModeManualBlocking
					wantNonBlocking := (mode != ModePerPod) && !blocking

					wantBlockingStr := "Blocking"
					if wantNonBlocking {
						wantBlockingStr = "Non-Blocking"
					}
					wantCombined := mode.String() + "/" + wantBlockingStr

					if got := isPerPodMode(); got != wantPerPod {
						t.Fatalf("isPerPodMode()=%v want %v", got, wantPerPod)
					}
					if got := isManualBlockingMode(); got != wantManualBlocking {
						t.Fatalf("isManualBlockingMode()=%v want %v", got, wantManualBlocking)
					}
					if got := isNonBlockingSolving(); got != wantNonBlocking {
						t.Fatalf("isNonBlockingSolving()=%v want %v", got, wantNonBlocking)
					}
					if got := getBlockingAsString(); got != wantBlockingStr {
						t.Fatalf("getBlockingAsString()=%q want %q", got, wantBlockingStr)
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
