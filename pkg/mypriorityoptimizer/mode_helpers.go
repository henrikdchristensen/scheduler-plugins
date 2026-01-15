// mode_helpers.go
package mypriorityoptimizer

// -------------------------
// isSchedulingFailureMode
// -------------------------

// isSchedulingFailureMode is the optimizer cadence that optimizes for every new pod.
func isSchedulingFailureMode() bool { return OptimizeMode == ModeSchedulingFailure }

// -------------------------
// isManualBlockingMode
// -------------------------

// isManualBlockingMode is true if the mode is ManualBlocking.
func isManualBlockingMode() bool { return OptimizeMode == ModeManualBlocking }

// -------------------------
// isNonBlockingSolving
// -------------------------

// isNonBlockingSolving is true for modes where we:
// - collect pods at PostFilter, and
// - take Active only after we know a plan is worthwhile.
// SchedulingFailure is always treated as blocking.
func isNonBlockingSolving() bool {
	return OptimizeMode != ModeSchedulingFailure && !OptimizeBlockingSolving
}

// -------------------------
// getBlockingAsString
// -------------------------

// getBlockingAsString returns "Blocking" or "Non-Blocking".
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
func getModeCombinedAsString() string {
	return OptimizeMode.String() + "/" + getBlockingAsString()
}

// -------------------------
// ModeType String()
// -------------------------

// String returns the string representation of the ModeType.
func (m ModeType) String() string {
	switch m {
	case ModeSchedulingFailure:
		return "SchedulingFailure"
	case ModePeriodic:
		return "Periodic"
	case ModeStableQueue:
		return "StableQueue"
	case ModeManual:
		return "Manual"
	case ModeManualBlocking:
		return "ManualBlocking"
	default:
		return "Periodic"
	}
}
