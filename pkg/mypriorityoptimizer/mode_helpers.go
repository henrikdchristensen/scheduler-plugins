// mode_helpers.go
package mypriorityoptimizer

// -------------------------
// isPerPodMode
// -------------------------

// isPerPodMode is the optimizer cadence that optimizes for every new pod.
// CHECKED
func isPerPodMode() bool { return OptimizeMode == ModePerPod }

// -------------------------
// isManualBlockingMode
// -------------------------

// isManualBlockingMode is true if the mode is ManualBlocking.
// CHECKED
func isManualBlockingMode() bool { return OptimizeMode == ModeManualBlocking }

// -------------------------
// isNonBlockingSolving
// -------------------------

// isNonBlockingSolving is true for modes where we:
// - collect pods at PostFilter, and
// - take Active only after we know a plan is worthwhile.
// PerPod is always treated as blocking.
// CHECKED
func isNonBlockingSolving() bool {
	return OptimizeMode != ModePerPod && !OptimizeBlockingSolving
}

// -------------------------
// getBlockingAsString
// -------------------------

// getBlockingAsString returns "Blocking" or "Non-Blocking".
// CHECKED
func getBlockingAsString() string {
	if isNonBlockingSolving() {
		return "Non-Blocking"
	}
	return "Blocking"
}

// -------------------------
// getModeCombinedAsString
// -------------------------

// getModeCombinedAsString returns "<Mode>/<Blocking|Non-Blocking>".
// CHECKED
func getModeCombinedAsString() string {
	return OptimizeMode.String() + "/" + getBlockingAsString()
}

// -------------------------
// ModeType String()
// -------------------------

// String returns the string representation of the ModeType.
// CHECKED
func (m ModeType) String() string {
	switch m {
	case ModePerPod:
		return "PerPod"
	case ModePeriodic:
		return "Periodic"
	case ModeInterlude:
		return "Interlude"
	case ModeManual:
		return "Manual"
	case ModeManualBlocking:
		return "ManualBlocking"
	default:
		return "Periodic"
	}
}
