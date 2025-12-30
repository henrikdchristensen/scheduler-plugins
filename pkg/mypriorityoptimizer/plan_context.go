// plan_context.go
package mypriorityoptimizer

import (
	v1 "k8s.io/api/core/v1"
)

// -------------------------
// planContext
// -------------------------

// planContext builds the context (nodes, pods, solver input) for optimization.
func (pl *SharedState) planContext(preemptor *v1.Pod) (
	nodes []*v1.Node,
	pods []*v1.Pod,
	inp SolverInput,
	err error,
) {
	nodes, err = getNodesForPlanContext(pl)
	if err != nil {
		return nil, nil, SolverInput{}, ErrFailedToListNodes
	}

	pods, err = getPodsForPlanContext(pl)
	if err != nil {
		return nodes, nil, SolverInput{}, ErrFailedToListPods
	}

	inp, err = buildInputForPlanCtx(pl, nodes, pods, preemptor)
	if err != nil {
		return nodes, pods, SolverInput{}, ErrFailedToBuildSolverInput
	}

	return nodes, pods, inp, nil
}

// -------------------------
// Test Hooks
// -------------------------

var (
	getNodesForPlanContext = func(pl *SharedState) ([]*v1.Node, error) { return pl.getNodes() }

	getPodsForPlanContext = func(pl *SharedState) ([]*v1.Pod, error) { return pl.getPods() }

	buildInputForPlanCtx = func(pl *SharedState, nodes []*v1.Node, pods []*v1.Pod, preemptor *v1.Pod) (SolverInput, error) {
		return pl.buildSolverInput(nodes, pods, preemptor)
	}
)
