// plan_context_test.go
package mypriorityoptimizer

import (
	"errors"
	"testing"

	v1 "k8s.io/api/core/v1"
)

// -------------------------
// planContext
// -------------------------

func TestPlanContext(t *testing.T) {
	pl := &SharedState{}

	n1 := node("n1")
	p1 := pod("ns", "p1", onNode("n1"), withPhase(v1.PodRunning))
	wantInp := SolverInput{}

	t.Run("node list error", func(t *testing.T) {
		withVar(t, &getNodesForPlanContext, func(*SharedState) ([]*v1.Node, error) {
			return nil, errors.New("nodes failed")
		})

		nodes, pods, _, err := pl.planContext(nil)
		mustEq(t, err, ErrFailedToListNodes, "err mismatch")
		must(t, nodes == nil, "nodes should be nil")
		must(t, pods == nil, "pods should be nil")
	})

	t.Run("pod list error", func(t *testing.T) {
		withVar(t, &getNodesForPlanContext, func(*SharedState) ([]*v1.Node, error) {
			return []*v1.Node{n1}, nil
		})
		withVar(t, &getPodsForPlanContext, func(*SharedState) ([]*v1.Pod, error) {
			return nil, errors.New("pods failed")
		})

		nodes, pods, _, err := pl.planContext(nil)
		mustEq(t, err, ErrFailedToListPods, "err mismatch")
		mustEq(t, len(nodes), 1, "nodes len")
		must(t, pods == nil, "pods should be nil")
	})

	t.Run("build input error", func(t *testing.T) {
		withVar(t, &getNodesForPlanContext, func(*SharedState) ([]*v1.Node, error) {
			return []*v1.Node{n1}, nil
		})
		withVar(t, &getPodsForPlanContext, func(*SharedState) ([]*v1.Pod, error) {
			return []*v1.Pod{p1}, nil
		})
		withVar(t, &buildInputForPlanCtx, func(*SharedState, []*v1.Node, []*v1.Pod, *v1.Pod) (SolverInput, error) {
			return SolverInput{}, errors.New("boom")
		})

		nodes, pods, _, err := pl.planContext(nil)
		mustEq(t, err, ErrFailedToBuildSolverInput, "err mismatch")
		mustEq(t, len(nodes), 1, "nodes len")
		mustEq(t, len(pods), 1, "pods len")
	})

	t.Run("success", func(t *testing.T) {
		withVar(t, &getNodesForPlanContext, func(*SharedState) ([]*v1.Node, error) {
			return []*v1.Node{n1}, nil
		})
		withVar(t, &getPodsForPlanContext, func(*SharedState) ([]*v1.Pod, error) {
			return []*v1.Pod{p1}, nil
		})
		withVar(t, &buildInputForPlanCtx, func(*SharedState, []*v1.Node, []*v1.Pod, *v1.Pod) (SolverInput, error) {
			return wantInp, nil
		})

		nodes, pods, inp, err := pl.planContext(nil)
		mustNoErr(t, err, "unexpected err")
		mustEq(t, len(nodes), 1, "nodes len")
		mustEq(t, len(pods), 1, "pods len")
		mustEq(t, inp, wantInp, "inp mismatch")
	})
}
