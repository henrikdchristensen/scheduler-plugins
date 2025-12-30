// http_types.go
// CHECKED
package mypriorityoptimizer

// HttpResponse represents the JSON response structure for optimization requests.
type HttpResponse struct {
	// Overall status of the optimization request
	Status string `json:"status"`
	// Duration of the optimization in milliseconds
	DurationMs int64 `json:"duration_ms"`
	// Error message, if any
	Error string `json:"error,omitempty"`
	// Whether there is an active plan in progress
	Active bool `json:"active"`
	// Baseline solver score before optimization
	Baseline *SolverScore `json:"baseline,omitempty"`
	// Name of the best solver used
	BestName string `json:"best_name,omitempty"`
	// Solver attempts made during optimization
	Attempts []SolverResult `json:"attempts,omitempty"`
	// Number of pods that were pending before optimization
	PendingBefore int `json:"pending_before"`
}
