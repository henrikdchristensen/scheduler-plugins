package mypriorityoptimizer

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

// OptimizationStats is a persistent cumulative counter doc stored in a ConfigMap.
type OptimizationStats struct {
	UpdatedAt time.Time `json:"updated_at"`

	OptimizationFlowCallsTotal   int64 `json:"optimization_flow_calls_total"`
	OptimizationFlowEnteredTotal int64 `json:"optimization_flow_entered_total"`

	SolverAttemptsTotal int64 `json:"solver_attempts_total"`

	BestSolverOptimalTotal  int64 `json:"best_solver_optimal_total"`
	BestSolverFeasibleTotal int64 `json:"best_solver_feasible_total"`
	BestSolverFailedTotal   int64 `json:"best_solver_failed_total"`

	PlanNotApplicableTotal     int64 `json:"plan_not_applicable_total"`
	NoImprovingSolutionTotal   int64 `json:"no_improving_solution_total"`
}

// OptimizationStatsDelta is applied to OptimizationStats.
type OptimizationStatsDelta struct {
	OptimizationFlowCalls   int64
	OptimizationFlowEntered int64

	SolverAttempts int64

	BestSolverOptimal  int64
	BestSolverFeasible int64
	BestSolverFailed   int64

	PlanNotApplicable   int64
	NoImprovingSolution int64
}

func (d OptimizationStatsDelta) isZero() bool {
	return d.OptimizationFlowCalls == 0 &&
		d.OptimizationFlowEntered == 0 &&
		d.SolverAttempts == 0 &&
		d.BestSolverOptimal == 0 &&
		d.BestSolverFeasible == 0 &&
		d.BestSolverFailed == 0 &&
		d.PlanNotApplicable == 0 &&
		d.NoImprovingSolution == 0
}

func (s *OptimizationStats) apply(d OptimizationStatsDelta) {
	s.OptimizationFlowCallsTotal += d.OptimizationFlowCalls
	s.OptimizationFlowEnteredTotal += d.OptimizationFlowEntered
	s.SolverAttemptsTotal += d.SolverAttempts
	s.BestSolverOptimalTotal += d.BestSolverOptimal
	s.BestSolverFeasibleTotal += d.BestSolverFeasible
	s.BestSolverFailedTotal += d.BestSolverFailed
	s.PlanNotApplicableTotal += d.PlanNotApplicable
	s.NoImprovingSolutionTotal += d.NoImprovingSolution
}

func (pl *SharedState) persistOptimizationStatsDelta(ctx context.Context, d OptimizationStatsDelta) {
	if d.isZero() {
		return
	}
	if persistOptimizationStatsHook != nil {
		_ = persistOptimizationStatsHook(pl, ctx, d)
		return
	}
	if err := pl.addOptimizationStatsDelta(ctx, d); err != nil {
		klog.V(MyV).ErrorS(err, "failed to persist optimization-stats")
	}
}

func (pl *SharedState) addOptimizationStatsDelta(ctx context.Context, d OptimizationStatsDelta) error {
	if pl == nil || pl.Client == nil {
		// In tests we often have nil Client; treat as no-op.
		return nil
	}

	doc := ConfigMapDoc{
		Namespace: SystemNamespace,
		Name:      OptimizationStatsConfigMapName,
		LabelKey:  OptimizationStatsConfigMapLabelKey,
		DataKey:   OptimizationStatsConfigMapLabelKey + ".json",
	}

	cms := pl.Client.CoreV1().ConfigMaps(SystemNamespace)

	// Conflict-safe read-modify-update loop
	const maxRetries = 6
	for i := 0; i < maxRetries; i++ {
		cm, err := cms.Get(ctx, doc.Name, metav1.GetOptions{})
		switch {
		case apierrors.IsNotFound(err):
			// Create fresh doc with delta applied.
			st := OptimizationStats{UpdatedAt: getTimestampNowUtc()}
			st.apply(d)
			return doc.ensureJson(ctx, cms, st)

		case err != nil:
			return err

		default:
			var st OptimizationStats
			raw := ""
			if cm.Data != nil {
				raw = cm.Data[doc.DataKey]
			}
			if raw != "" {
				_ = json.Unmarshal([]byte(raw), &st) // best-effort
			}
			st.apply(d)
			st.UpdatedAt = getTimestampNowUtc()

			b, err := json.MarshalIndent(&st, "", "  ")
			if err != nil {
				return err
			}
			if cm.Data == nil {
				cm.Data = map[string]string{}
			}
			cm.Data[doc.DataKey] = string(b)

			_, err = cms.Update(ctx, cm, metav1.UpdateOptions{})
			if apierrors.IsConflict(err) {
				continue
			}
			return err
		}
	}

	return fmt.Errorf("optimization-stats update: too many conflicts")
}

// Test hook (optional)
var (
	persistOptimizationStatsHook func(pl *SharedState, ctx context.Context, d OptimizationStatsDelta) error
)
