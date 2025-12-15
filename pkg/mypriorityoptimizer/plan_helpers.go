// plan_helpers.go
package mypriorityoptimizer

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"sort"
	"sync/atomic"

	"golang.org/x/sync/errgroup"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

// -------------------------
// Test Hooks
// -------------------------

var (
	evictTargetsHook              func(pl *SharedState, ctx context.Context, targets []*v1.Pod) error
	waitPodsGoneHook              func(pl *SharedState, ctx context.Context, pods []*v1.Pod) error
	activatePlannedPodsHook       func(pl *SharedState, toActivate map[string]*v1.Pod)
	isPlanCompletedHook           func(pl *SharedState, ap *ActivePlan) (bool, error)
	onPlanCompletedHook           func(pl *SharedState, status PlanStatus, ap *ActivePlan)
	exportPlanToConfigMapHook     func(pl *SharedState, ctx context.Context, name string, sp *StoredPlan) error
	markPlanStatusToConfigMapHook func(pl *SharedState, ctx context.Context, planCM string, status PlanStatus) bool
)

// -------------------------
// Active plan / optimization flags
// -------------------------

// tryEnterActivePlan attempts to enter the active plan state. Use
// CompareAndSwap to ensure only one goroutine can enter the active state by
// checking that the previous value is false before setting it to true.
func (pl *SharedState) tryEnterActivePlan() bool {
	return pl.ActivePlanInProgress.CompareAndSwap(false, true)
}

// tryLeaveActivePlan exits the active plan state.
func (pl *SharedState) tryLeaveActivePlan() {
	pl.ActivePlanInProgress.Store(false)
}

// getActivePlan returns the currently active plan, if any.
func (pl *SharedState) getActivePlan() *ActivePlan {
	return pl.ActivePlan.Load()
}

// override in unit tests to simulate CAS failure deterministically.
var activePlanCompareAndSwap = func(pl *SharedState, old, new *ActivePlan) bool {
	return pl.ActivePlan.CompareAndSwap(old, new)
}

// tryClearActivePlan clears the currently active plan, if any (CAS-based).
func (pl *SharedState) tryClearActivePlan(ap *ActivePlan) bool {
	if ap == nil {
		return false
	}
	return activePlanCompareAndSwap(pl, ap, nil)
}

// tryEnterOptimizationFlow attempts to enter the optimization flow state.
func (pl *SharedState) tryEnterOptimizationFlow() bool {
	return pl.OptimizationInProgress.CompareAndSwap(false, true)
}

// tryLeaveOptimizationFlow exits the optimization flow state.
func (pl *SharedState) tryLeaveOptimizationFlow() {
	pl.OptimizationInProgress.Store(false)
}

// podKey returns the unique key for a pod in "namespace/name" format.
func podKey(p *v1.Pod) string {
	return mergeNsName(p.Namespace, p.Name)
}

// toPlanPod converts a core/v1 Pod into a SolverPod (UID, Namespace, Name).
func toPlanPod(p *v1.Pod) SolverPod {
	return SolverPod{
		UID:       p.UID,
		Namespace: p.Namespace,
		Name:      p.Name,
	}
}

// makePlacement builds a Placement for a pod on a given node.
func makePlacement(p *v1.Pod, node string) SolverPod {
	return SolverPod{
		UID:       p.UID,
		Namespace: p.Namespace,
		Name:      p.Name,
		Node:      node,
	}
}

// makeNewPlacement builds a NewPlacement for a pod moving from src -> dst.
func makeNewPlacement(p *v1.Pod, oldNode, toNode string) SolverPod {
	return SolverPod{
		UID:       p.UID,
		Namespace: p.Namespace,
		Name:      p.Name,
		OldNode:   oldNode,
		Node:      toNode,
	}
}

// increaseWorkloadQuota increments the quota count for a workload/node pair.
func increaseWorkloadQuota(wq WorkloadQuotas, wk WorkloadKey, node string) {
	wkKey := wk.String()
	if wq[wkKey] == nil {
		wq[wkKey] = map[string]int32{}
	}
	wq[wkKey][node]++
}

// sortPlacementsByPod sorts placements by (namespace, name) for stable output.
func sortPlacementsByPod(pls []SolverPod) {
	sort.Slice(pls, func(i, j int) bool {
		pi, pj := pls[i], pls[j]
		if pi.Namespace != pj.Namespace {
			return pi.Namespace < pj.Namespace
		}
		return pi.Name < pj.Name
	})
}

// sortPodSetItemsByPriorityAndCreation sorts PodSetItems by:
//  1. priority (higher first)
//  2. creation timestamp (older first)
//  3. name (for zero/identical timestamps)
func sortPodSetItemsByPriorityAndCreation(items []PodSetItem) {
	sort.Slice(items, func(i, j int) bool {
		pi := getPodPriority(items[i].p)
		pj := getPodPriority(items[j].p)
		if pi != pj {
			return pi > pj
		}
		ti := items[i].p.GetCreationTimestamp().Time
		tj := items[j].p.GetCreationTimestamp().Time
		if ti.IsZero() || tj.IsZero() {
			// fallback: deterministic order by name if timestamps are missing
			return items[i].p.GetName() < items[j].p.GetName()
		}
		return ti.Before(tj)
	})
}

// isPlanPodUnscheduled returns true if the solver does not want the pod placed.
func isPlanPodUnscheduled(toNode string) bool {
	return toNode == ""
}

// isPlanPodMove returns true if the pod is currently bound to a node
// and the solver wants it on a different node.
func isPlanPodMove(fromNode, toNode string) bool {
	return fromNode != "" && fromNode != toNode
}

// isPlanPodPlacementChanged returns true if the pod's placement is changing.
func isPlanPodPlacementChanged(fromNode, toNode string) bool {
	return fromNode == "" || fromNode != toNode
}

// isPlanPodNewlyScheduled returns true if pod was pending (no node) and is now placed.
func isPlanPodNewlyScheduled(fromNode, toNode string) bool {
	return fromNode == "" && toNode != ""
}

// -------------------------
// Pod indexing / collection helpers
// -------------------------

// indexPodsForPlan builds a UID -> *Pod map for all live pods plus (optionally) the preemptor.
func indexPodsForPlan(pods []*v1.Pod, preemptor *v1.Pod) map[types.UID]*v1.Pod {
	byUID := podsByUID(pods)
	if preemptor != nil && !isPodDeleted(preemptor) {
		byUID[preemptor.UID] = preemptor
	}
	return byUID
}

// collectOldPlacements returns placements for all currently assigned & alive pods.
func collectOldPlacements(byUID map[types.UID]*v1.Pod) []SolverPod {
	oldPlacements := make([]SolverPod, 0, len(byUID))
	for _, p := range byUID {
		if isPodAssignedAndAlive(p) {
			oldPlacements = append(oldPlacements, makePlacement(p, getPodAssignedNodeName(p)))
		}
	}
	sortPlacementsByPod(oldPlacements)
	return oldPlacements
}

// collectEvictions builds evict placements from the solver output and pod index.
func collectEvictions(out *SolverOutput, byUID map[types.UID]*v1.Pod) []SolverPod {
	if out == nil || len(out.Evictions) == 0 {
		return nil
	}
	evicts := make([]SolverPod, 0, len(out.Evictions))
	for _, e := range out.Evictions {
		p := byUID[e.UID]
		if p == nil {
			continue
		}
		if isPodAssignedAndAlive(p) {
			evicts = append(evicts, makePlacement(p, getPodAssignedNodeName(p)))
		}
	}
	// Keep stable output (helps tests & debugging).
	sortPlacementsByPod(evicts)
	return evicts
}

// -------------------------
// buildPlan
// -------------------------

// buildPlan builds the evictions, movements, old placements, new placements,
// placementByName, workloadQuotas and the nominatedNode (if preemptor exists)
// from the output of the solver.
func (pl *SharedState) buildPlan(out *SolverOutput, preemptor *v1.Pod, pods []*v1.Pod) (*Plan, error) {
	if out == nil {
		return &Plan{}, nil
	}

	byUID := indexPodsForPlan(pods, preemptor)

	oldPlacements := collectOldPlacements(byUID)
	evicts := collectEvictions(out, byUID)

	var (
		moves         []SolverPod
		newPlacements []SolverPod
		nominatedNode string
	)

	placementByName := make(map[string]string)
	workloadQuotas := make(WorkloadQuotas)

	preUID := types.UID("")
	if preemptor != nil && !isPodDeleted(preemptor) {
		preUID = preemptor.UID
	}

	for _, plm := range out.Placements {
		if isPlanPodUnscheduled(plm.Node) {
			continue
		}

		p := byUID[plm.UID]
		if p == nil || isPodDeleted(p) {
			// stale solver output or pod already terminating -> ignore
			continue
		}

		fromNode := getPodAssignedNodeName(p)
		np := makeNewPlacement(p, fromNode, plm.Node)
		newPlacements = append(newPlacements, np)

		// Preemptor: always nominate and pin by exact name; never counted as a "move".
		if preUID != "" && isSamePodUID(plm.UID, preUID) {
			placementByName[podKey(p)] = plm.Node
			nominatedNode = plm.Node
			continue
		}

		// Moves (assigned->different assigned).
		if isPlanPodMove(fromNode, plm.Node) {
			moves = append(moves, np)
		}

		// If placement isn't changing (already where it should be), don't consume quota / pins.
		if !isPlanPodPlacementChanged(fromNode, plm.Node) {
			continue
		}

		// Controller-owned pods: track via per-workload per-node quotas.
		if wk, owned := getTopWorkload(p); owned {
			increaseWorkloadQuota(workloadQuotas, wk, plm.Node)
			continue
		}

		// Standalone pods: pin directly by name.
		placementByName[podKey(p)] = plm.Node
	}

	sortPlacementsByPod(newPlacements)
	sortPlacementsByPod(moves)

	return &Plan{
		Evicts:          evicts,
		Moves:           moves,
		OldPlacements:   oldPlacements,
		NewPlacements:   newPlacements,
		PlacementByName: placementByName,
		WorkloadQuotas:  workloadQuotas,
		NominatedNode:   nominatedNode,
	}, nil
}

// -------------------------
// setActivePlan
// -------------------------

// setActivePlan sets the given stored plan as the active plan and initializes
// its counters, deriving both WorkloadPerNodeCnts and PlacementByName solely
// from NewPlacements. For controller-owned pods, quotas are keyed by the
// controller (e.g., ReplicaSet) name.
func (pl *SharedState) setActivePlan(plan *Plan, id string, _ []*v1.Pod) {
	if plan == nil {
		klog.V(MyV).ErrorS(ErrNoPlanProvided, InfoNoPlanProvided+" in setActivePlan", nil)
		return
	}

	// Cancel any previous plan's timeout watcher.
	if old := pl.getActivePlan(); old != nil && old.Cancel != nil {
		old.Cancel()
	}

	ctxPlan, cancel := context.WithTimeout(context.Background(), PlanExecutionTimeout)
	ap := &ActivePlan{
		ID:              id,
		WorkloadQuotas:  buildWorkloadQuotas(plan.WorkloadQuotas),
		PlacementByName: plan.PlacementByName,
		Ctx:             ctxPlan,
		Cancel:          cancel,
	}

	pl.ActivePlan.Store(ap)
}

// -------------------------
// buildWorkloadQuotas
// -------------------------

// buildWorkloadQuotas converts WorkloadQuotas (int32) to WorkloadPerNodeCnts
// (atomic.Int32) for faster concurrent access during plan execution.
func buildWorkloadQuotas(wkQuotas WorkloadQuotas) WorkloadQuotasAtomics {
	if wkQuotas == nil {
		return nil
	}

	remaining := make(WorkloadQuotasAtomics) // workload -> node -> *atomic.Int32
	for wk, perNode := range wkQuotas {
		if remaining[wk] == nil {
			remaining[wk] = map[string]*atomic.Int32{}
		}
		for node, cnt := range perNode {
			ctr := remaining[wk][node]
			if ctr == nil {
				ctr = new(atomic.Int32)
				remaining[wk][node] = ctr
			}
			if cnt > 0 {
				ctr.Store(cnt)
			} else {
				ctr.Store(0)
			}
		}
	}
	return remaining
}

// -------------------------
// evictTargets
// -------------------------

// evictTargets evicts all target pods with bounded parallelism and per-op timeouts.
func (pl *SharedState) evictTargets(ctx context.Context, targets []*v1.Pod) error {
	if evictTargetsHook != nil {
		return evictTargetsHook(pl, ctx, targets)
	}
	if len(targets) == 0 {
		return nil
	}

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(EvictParallelism)

	for _, pod := range targets {
		p := pod // explicit capture (even though Go 1.22+ fixes range capture)
		if p == nil {
			continue
		}
		g.Go(func() error {
			opCtx, cancel := context.WithTimeout(gctx, EvictTimeout)
			defer cancel()
			if err := pl.evictPod(opCtx, p); err != nil && !apierrors.IsNotFound(err) {
				return fmt.Errorf("evict %s: %w", podRef(p), err)
			}
			return nil
		})
	}

	return g.Wait()
}

// -------------------------
// waitPodsGone
// -------------------------

// waitPodsGone waits until the evicted pods disappear from cache.
func (pl *SharedState) waitPodsGone(ctx context.Context, pods []*v1.Pod) error {
	if waitPodsGoneHook != nil {
		return waitPodsGoneHook(pl, ctx, pods)
	}
	if len(pods) == 0 {
		return nil
	}

	remaining := make(map[SolverPod]struct{}, len(pods))
	for _, p := range pods {
		if p == nil {
			continue
		}
		remaining[toPlanPod(p)] = struct{}{}
	}
	if len(remaining) == 0 {
		return nil
	}

	return wait.PollUntilContextCancel(ctx, WaitPodsGoneInterval, true, func(ctx context.Context) (bool, error) {
		if len(remaining) == 0 {
			return true, nil
		}
		for key := range remaining {
			p, err := pl.getPodByName(key.Namespace, key.Name)
			switch {
			case apierrors.IsNotFound(err):
				delete(remaining, key)
			case err != nil:
				// transient lister error; keep polling
				return false, nil
			default:
				// Consider it "gone" if UID changed or it started deleting.
				if !isSamePodUID(p.UID, key.UID) || isPodDeleted(p) {
					delete(remaining, key)
				}
			}
		}
		return len(remaining) == 0, nil
	})
}

// -------------------------
// activatePods
// -------------------------

// activatePods performs the actual framework.Handle.Activate call.
// In tests we override this to capture which pods would be activated.
var activatePods = func(pl *SharedState, toAct map[string]*v1.Pod) {
	pl.Handle.Activate(klog.Background(), toAct)
}

// activatePods activates up to 'max' pods from the blocked set; clear only the
// ones activated. It returns the UIDs of the pods that were attempted to be
// activated (in priority/time order). If max <= 0, all pods are activated.
func (pl *SharedState) activatePods(podSet *PodSet, removeActivated bool, max int) (tried []types.UID) {
	// Prune stale entries first
	_ = pl.prunePodSet(podSet)

	if !doesPodSetExist(podSet) {
		return
	}

	blockedPods := podSet.Snapshot()
	items := make([]PodSetItem, 0, len(blockedPods))

	// Resolve current Pod objects so we don't activate stale/deleted ones.
	for _, k := range blockedPods {
		if p, err := pl.getPodByName(k.Namespace, k.Name); err == nil && p != nil {
			items = append(items, PodSetItem{p: p, key: k})
		}
	}
	if len(items) == 0 {
		return
	}

	sortPodSetItemsByPriorityAndCreation(items)

	limit := len(items)
	if max > 0 && max < limit {
		limit = max
	}

	toAct := make(map[string]*v1.Pod, limit)
	for _, it := range items[:limit] {
		toAct[podKey(it.p)] = it.p
		tried = append(tried, it.key.UID)
	}

	if len(toAct) == 0 {
		return tried
	}

	activatePods(pl, toAct)
	klog.InfoS("activated pods", "set", podSet.Name, "count", len(toAct))

	if removeActivated {
		for _, it := range items[:limit] {
			podSet.RemovePod(it.key.UID)
		}
	}

	return tried
}

// -------------------------
// activatePlannedPods
// -------------------------

// activatePlannedPods activates all live pending pods that the plan intends to
// place (i.e., NewPlacement with FromNode == "" and ToNode != "").
func (pl *SharedState) activatePlannedPods(plan *Plan, pods []*v1.Pod) {
	if plan == nil || len(plan.NewPlacements) == 0 || len(pods) == 0 {
		klog.V(MyV).InfoS("activatePlannedPods: no plan or no new placements or no pods",
			"planNil", plan == nil,
			"newPlacementsLen", func() int {
				if plan == nil {
					return 0
				}
				return len(plan.NewPlacements)
			}(),
			"podsLen", len(pods),
		)
		return
	}

	allow := make(map[types.UID]struct{}, len(plan.NewPlacements))
	for _, np := range plan.NewPlacements {
		if isPlanPodNewlyScheduled(np.OldNode, np.Node) {
			allow[np.UID] = struct{}{}
		}
	}
	if len(allow) == 0 {
		klog.V(MyV).InfoS("activatePlannedPods: no new placements with FromNode == \"\" and ToNode != \"\"")
		return
	}

	toAct := make(map[string]*v1.Pod, len(allow))
	for _, p := range pods {
		if isPodDeleted(p) || isPodAssigned(p) {
			continue
		}
		if _, ok := allow[p.UID]; !ok {
			continue
		}
		toAct[podKey(p)] = p
	}

	if len(toAct) == 0 {
		klog.V(MyV).InfoS("activatePlannedPods: no matching pending pods found")
		return
	}

	klog.InfoS(InfoActivatingPlannedPendingPods, "count", len(toAct))

	// Test hook: let unit tests observe the activation set without a real Handle.
	if activatePlannedPodsHook != nil {
		activatePlannedPodsHook(pl, toAct)
		return
	}

	activatePods(pl, toAct)
}

// -------------------------
// isPlanCompleted
// -------------------------

// computeWorkloadStatus computes the status of all workloads in the cluster,
// returning a map from workload key to wkStatus (HasLive, HasPending).
func (pl *SharedState) computeWorkloadStatus(allPods []*v1.Pod) map[string]wkStatus {
	workloadStatus := make(map[string]wkStatus)

	for _, p := range allPods {
		if isPodDeleted(p) {
			continue
		}
		if wk, owned := getTopWorkload(p); owned {
			key := wk.String()
			st := workloadStatus[key]
			st.HasLive = true
			if !isPodAssigned(p) {
				st.HasPending = true
			}
			workloadStatus[key] = st
		}
	}

	return workloadStatus
}

// checkPinnedPodsSatisfied verifies pinned pods are on expected nodes.
// Deleted/terminating pinned pods are treated as satisfied.
func (pl *SharedState) checkPinnedPodsSatisfied(ap *ActivePlan) (bool, error) {
	for nsname, wantNode := range ap.PlacementByName {
		ns, name, err := splitNsName(nsname)
		if err != nil {
			return false, err
		}
		po, err := pl.getPodByName(ns, name)
		if apierrors.IsNotFound(err) {
			klog.V(MyV).InfoS("plan completion: pinned pod gone; treating as satisfied",
				"pod", nsname,
				"expectedNode", wantNode,
			)
			continue
		}
		if err != nil {
			return false, err
		}
		if isPodDeleted(po) {
			klog.V(MyV).InfoS("plan completion: pinned pod terminating; treating as satisfied",
				"pod", nsname,
				"expectedNode", wantNode,
			)
			continue
		}
		haveNode := getPodAssignedNodeName(po)
		if isPlanPodPlacementChanged(haveNode, wantNode) {
			klog.V(MyV).InfoS("plan incomplete: pinned pod mismatch",
				"pod", nsname,
				"expectedNode", wantNode,
				"haveNode", haveNode,
			)
			return false, nil
		}
	}
	return true, nil
}

// checkQuotasSatisfied applies the quota rules described in isPlanCompleted().
func (pl *SharedState) checkQuotasSatisfied(ap *ActivePlan, workloadStatus map[string]wkStatus) (bool, error) {
	for wk, perNode := range ap.WorkloadQuotas {
		var totalRemaining int32
		for _, ctr := range perNode {
			totalRemaining += ctr.Load()
		}

		if totalRemaining <= 0 {
			continue
		}

		st := workloadStatus[wk]

		// Workload gone => ignore remaining quota.
		if !st.HasLive {
			klog.V(MyV).InfoS("plan completion: workload scaled down or deleted; ignoring remaining quota",
				"workload", wk,
				"remaining", totalRemaining,
			)
			continue
		}

		// Still has pending => not done yet.
		if st.HasPending {
			klog.V(MyV).InfoS("plan incomplete: workload still has pending pods and remaining quota",
				"workload", wk,
				"remaining", totalRemaining,
			)
			return false, nil
		}

		// Live but no pending => treat remaining as satisfied.
		klog.V(MyV).InfoS("plan completion: workload has no pending pods; treating remaining quota as satisfied",
			"workload", wk,
			"remaining", totalRemaining,
		)
	}
	return true, nil
}

// isPlanCompleted checks if the plan is completed by verifying the state of the
// cluster. It is based on the current active plan snapshot (ap):
//
//	A) all pinned pods (PlacementByName) that still exist must run on the planned node;
//	   if a pinned pod was deleted or is terminating, we treat it as "no longer required".
//	B) all per-workload per-node quotas must be consumed, except for workloads that
//	   have been scaled down / deleted (no live pods) or have no pending pods left.
func (pl *SharedState) isPlanCompleted(ap *ActivePlan) (bool, error) {
	if isPlanCompletedHook != nil {
		return isPlanCompletedHook(pl, ap)
	}
	if ap == nil {
		// Plan got torn down concurrently; treat as "not completed yet" (retry later).
		klog.V(MyV).InfoS("plan completion check skipped: no active plan doc")
		return false, nil
	}

	allPods, err := pl.getPods()
	if err != nil {
		return false, err
	}
	workloadStatus := pl.computeWorkloadStatus(allPods)

	ok, err := pl.checkPinnedPodsSatisfied(ap)
	if err != nil || !ok {
		return ok, err
	}

	ok, err = pl.checkQuotasSatisfied(ap, workloadStatus)
	if err != nil || !ok {
		return ok, err
	}

	return true, nil
}

// -------------------------
// onPlanCompleted
// -------------------------

// onPlanCompleted is called when a plan is settled (i.e., all its actions are completed).
func (pl *SharedState) onPlanCompleted(status PlanStatus) bool {
	ap := pl.getActivePlan()

	// Win-or-lose: swap the ActivePlan pointer from 'ap' to nil.
	if !pl.tryClearActivePlan(ap) {
		return false
	}

	// Winner zone: one-time teardown.
	pl.tryLeaveActivePlan()
	if ap != nil && ap.Cancel != nil {
		ap.Cancel()
	}

	// Allow tests to intercept teardown without hitting external deps.
	if onPlanCompletedHook != nil {
		onPlanCompletedHook(pl, status, ap)
		return true
	}

	// Activate blocked pods
	pl.activatePods(pl.BlockedWhileActive, false, -1)

	if ap != nil {
		klog.InfoS(InfoDeactivatingActivePlan, "planID", ap.ID)
		// Mark the plan statuses in ConfigMaps
		pl.setPlanStatusInConfigMap(context.Background(), ap.ID, status)
	}

	return true
}

// -------------------------
// isPodAllowedByPlan
// -------------------------

// isPodAllowedByPlan returns true if the pod is allowed by the active plan.
// Standalone/preemptor pods are allowed by exact name match. For
// controller-owned pods, we allow only if the plan still has remaining per-node
// quota for that workload. If the pod already targets a specific node (NodeName
// set), we check that node's remaining quota; otherwise we allow if ANY node
// for that workload has remaining > 0.
func (pl *SharedState) isPodAllowedByPlan(pod *v1.Pod) bool {
	ap := pl.getActivePlan()
	if ap == nil || pod == nil {
		return false
	}

	// Standalone/preemptor pins addressed by name.
	if _, ok := ap.PlacementByName[podKey(pod)]; ok {
		return true
	}

	// Workload quotas (pods created by a controller).
	if wk, ok := getTopWorkload(pod); ok {
		perNode, ok := ap.WorkloadQuotas[wk.String()]
		if !ok || len(perNode) == 0 {
			return false
		}

		// If a node is already selected, require quota on that specific node.
		if node := getPodAssignedNodeName(pod); node != "" {
			if ctr, exists := perNode[node]; exists && ctr.Load() > 0 {
				return true
			}
			return false
		}

		// Otherwise, allow if ANY node still has remaining quota.
		for _, ctr := range perNode {
			if ctr.Load() > 0 {
				return true
			}
		}
		return false
	}

	return false
}

// -------------------------
// filterNodes
// -------------------------

// filterNodes returns the set of nodes the pod is allowed to run on according to the active plan.
func (pl *SharedState) filterNodes(pod *v1.Pod) (sets.Set[string], string, bool) {
	ap := pl.getActivePlan()
	if ap == nil {
		return nil, InfoNoActivePlan, true
	}
	if pod == nil {
		return nil, "nil pod; block", false
	}

	// Standalone/preemptor addressed by name.
	if node, present := ap.PlacementByName[podKey(pod)]; present {
		if node != "" {
			return sets.New(node), "standalone; pin to planned node", true
		}
		return nil, "standalone; allowed by plan", true
	}

	// Controller-owned: enforce per-workload per-node quotas.
	if wk, owned := getTopWorkload(pod); owned {
		perNode := ap.WorkloadQuotas[wk.String()]
		if len(perNode) == 0 {
			return nil, "workload not in active plan; block", false
		}
		allowed := sets.New[string]()
		for node, ctr := range perNode {
			if ctr.Load() > 0 {
				allowed.Insert(node)
			}
		}
		if allowed.Len() == 0 {
			return nil, "workload quotas exhausted; block", false
		}
		return allowed, "workload nodes allowed", true
	}

	return nil, "pod not in active plan; block", false
}

// -------------------------
// computePlanPodCounts
// -------------------------

// computePlanPodCounts summarizes the effect of a plan on pod counts:
//
//	pendingScheduled = number of currently-pending pods that get a placement
//	runningBefore    = number of pods currently assigned and alive
//	runningAfter     = runningBefore - evictedRunning + pendingScheduled
func computePlanPodCounts(out *SolverOutput, pods []*v1.Pod) (
	pendingScheduled, runningBefore, runningAfter int,
) {
	if out == nil {
		return 0, 0, 0
	}

	runningUIDs := make(map[types.UID]struct{}, len(pods))
	pendingUIDs := make(map[types.UID]struct{}, len(pods))

	for _, p := range pods {
		if isPodDeleted(p) {
			continue
		}
		if isPodAssigned(p) {
			if isPodAssignedAndAlive(p) {
				runningBefore++
				runningUIDs[p.UID] = struct{}{}
			}
			continue
		}
		pendingUIDs[p.UID] = struct{}{}
	}

	evictedRunning := 0
	for _, e := range out.Evictions {
		if _, ok := runningUIDs[e.UID]; ok {
			evictedRunning++
		}
	}

	for _, plm := range out.Placements {
		if isPlanPodUnscheduled(plm.Node) {
			continue
		}
		if _, ok := pendingUIDs[plm.UID]; ok {
			pendingScheduled++
		}
	}

	runningAfter = runningBefore - evictedRunning + pendingScheduled
	if runningAfter < 0 {
		runningAfter = 0
	}

	return pendingScheduled, runningBefore, runningAfter
}

// -------------------------
// exportPlanToConfigMap
// -------------------------

// exportPlanToConfigMap exports the given plan to a ConfigMap.
func (pl *SharedState) exportPlanToConfigMap(ctx context.Context, name string, sp *StoredPlan) error {
	if exportPlanToConfigMapHook != nil {
		return exportPlanToConfigMapHook(pl, ctx, name, sp)
	}

	doc := ConfigMapDoc{
		Namespace: SystemNamespace,
		Name:      name,
		LabelKey:  PlanConfigMapLabelKey,
		DataKey:   PlanConfigMapLabelKey + ".json",
	}

	cms := pl.Client.CoreV1().ConfigMaps(SystemNamespace)

	if err := doc.ensureJson(ctx, cms, sp); err != nil {
		return err
	}

	nsLister := pl.Handle.SharedInformerFactory().
		Core().V1().ConfigMaps().
		Lister().
		ConfigMaps(SystemNamespace)

	return pruneConfigMaps(ctx, cms, nsLister, PlanConfigMapLabelKey, PlansToRetain)
}

// -------------------------
// setPlanStatusInConfigMap
// -------------------------

// setPlanStatusInConfigMap updates the plan's ConfigMap status.
// - The plan CM is put into the requested status (unless already final).
// - Final is sticky (never overwrite Failed/Completed).
func (pl *SharedState) setPlanStatusInConfigMap(ctx context.Context, planCM string, status PlanStatus) {
	if markPlanStatusToConfigMapHook != nil && markPlanStatusToConfigMapHook(pl, ctx, planCM, status) {
		return
	}

	nsLister := pl.Handle.SharedInformerFactory().
		Core().V1().ConfigMaps().
		Lister().
		ConfigMaps(SystemNamespace)

	planDoc := ConfigMapDoc{
		Namespace: SystemNamespace,
		Name:      planCM,
		LabelKey:  PlanConfigMapLabelKey,
		DataKey:   PlanConfigMapLabelKey + ".json",
	}

	cms := pl.Client.CoreV1().ConfigMaps(SystemNamespace)

	_ = planDoc.mutateRaw(ctx, cms, nsLister, func(raw []byte) ([]byte, error) {
		var sp StoredPlan
		if err := json.Unmarshal(raw, &sp); err != nil {
			return nil, nil // best-effort
		}
		if sp.PlanStatus == PlanStatusCompleted || sp.PlanStatus == PlanStatusFailed {
			return nil, nil // final is sticky
		}
		sp.PlanStatus = status
		if status == PlanStatusCompleted || status == PlanStatusFailed {
			sp.CompletedAt = getTimestampNowUtc()
		}
		b, _ := json.MarshalIndent(&sp, "", "  ")
		return b, nil
	})
}

// -------------------------
// clusterFingerprint
// -------------------------

// clusterFingerprint returns a deterministic fingerprint of the "relevant"
// cluster state for scheduling/plan-cancellation purposes.
func clusterFingerprint(nodes []*v1.Node, pods []*v1.Pod) string {
	// 1) Keep only usable nodes; dedupe by name; sort for determinism.
	usable := make(map[string]*v1.Node, len(nodes))
	nodeNames := make([]string, 0, len(nodes))

	for _, n := range nodes {
		if n == nil {
			continue
		}
		if !isNodeUsable(n) {
			continue
		}
		if _, exists := usable[n.Name]; exists {
			continue
		}
		usable[n.Name] = n
		nodeNames = append(nodeNames, n.Name)
	}
	sort.Strings(nodeNames)

	// 2) Keep only assigned+alive pods on usable nodes; sort for determinism.
	type podEntry struct {
		node string
		ns   string
		name string
		uid  string
		cpu  int64
		mem  int64
		prio int32
	}
	entries := make([]podEntry, 0, len(pods))

	for _, p := range pods {
		if p == nil {
			continue
		}
		if !isPodAssignedAndAlive(p) {
			continue
		}
		node := getPodAssignedNodeName(p)
		if _, ok := usable[node]; !ok {
			continue
		}
		entries = append(entries, podEntry{
			node: node,
			ns:   p.Namespace,
			name: p.Name,
			uid:  string(p.UID),
			cpu:  getPodCPURequest(p),
			mem:  getPodMemoryRequest(p),
			prio: getPodPriority(p),
		})
	}

	sort.Slice(entries, func(i, j int) bool {
		a, b := entries[i], entries[j]
		if a.node != b.node {
			return a.node < b.node
		}
		if a.ns != b.ns {
			return a.ns < b.ns
		}
		if a.name != b.name {
			return a.name < b.name
		}
		return a.uid < b.uid
	})

	// 3) Hash a stable textual representation.
	h := fnv.New64a()

	for _, name := range nodeNames {
		n := usable[name]
		fmt.Fprintf(h, "N:%s:%d:%d|", name, getNodeCPUAllocatable(n), getNodeMemoryAllocatable(n))
	}
	for _, e := range entries {
		fmt.Fprintf(h, "P:%s:%s/%s:%s:%d:%d:%d|", e.node, e.ns, e.name, e.uid, e.cpu, e.mem, e.prio)
	}

	return fmt.Sprintf("%x", h.Sum64())
}
