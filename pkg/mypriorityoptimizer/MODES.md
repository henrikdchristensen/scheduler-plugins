# MyPriorityOptimizer Modes Documentation

This document provides a detailed verification and description of the optimization modes and concurrency mechanisms in the MyPriorityOptimizer scheduler plugin.

## Table of Contents

1. [Optimization Modes](#optimization-modes)
2. [Sync vs Async Solving](#sync-vs-async-solving)
3. [Concurrency Control](#concurrency-control)
4. [Implementation Correctness](#implementation-correctness)

---

## Optimization Modes

The plugin supports five distinct optimization modes that determine **when** the solver is triggered to compute optimal pod placements.

### 1. Manual Mode

**File**: `mode_types.go:18-20`  
**Constant**: `ModeManual`

**Description**:
Manual mode collects pending pods like Periodic mode but **only optimizes when manually triggered via HTTP**. Normal pod scheduling continues to run while waiting for the manual trigger.

**How it works**:
- Pods enter the scheduler queue normally and the default scheduler attempts to place them
- If pods fail to schedule, they remain pending but **do not trigger automatic optimization**
- Background loops (periodic/interlude) do **not** run in Manual mode (see `loop_helpers.go:33-38`)
- Optimization only occurs when the `/solve` endpoint is called via HTTP POST
- Normal scheduling flow is **not blocked** - pods can be scheduled by the default scheduler while waiting for manual trigger

**Use case**: Testing and evaluation where you want full control over when optimization occurs.

**Implementation verification**:
```go
// mode_types.go:18-20
// ModeManual collects like ModePeriodic but only optimizes when the HTTP
// /solve endpoint is called.
ModeManual

// mode_helpers.go:11-12
func isManualMode() bool { return OptimizeMode == ModeManual }

// loop_helpers.go:33-38 - No background loop started for Manual mode
switch OptimizeMode {
case ModePeriodic:
    go pl.loopPeriodic(ctx)
case ModeInterlude:
    go pl.loopInterlude(ctx)
}
// Note: Manual mode does NOT start a background loop

// http_server.go:96-140 - Manual trigger via HTTP
func (pl *SharedState) solveHandler(w http.ResponseWriter, r *http.Request) {
    // ... validation ...
    _, baseline, bestName, _, attempts, err := runFlowForHTTP(pl, context.Background())
    // ... response handling ...
}
```

---

### 2. Manual Blocking Mode

**File**: `mode_types.go:22-23`  
**Constant**: `ModeManualBlocking`

**Description**:
Manual Blocking mode **prevents any pod from entering the cluster** until the solver is manually triggered via HTTP and completes its optimization.

**How it works**:
- All pods are blocked at the PreEnqueue stage (see `hook_preenqueue.go:51-57`)
- Pods remain in Pending status and cannot be scheduled
- The scheduler accumulates a backlog of pending pods
- When `/solve` is called, the solver optimizes all pending pods
- After the plan is applied, blocked pods are released and can be scheduled according to the plan

**Key difference from Manual**:
- **Manual**: Normal scheduling continues, solver only runs on manual trigger
- **ManualBlocking**: **No scheduling** occurs until manual trigger

**Use case**: Testing and evaluation where you want to batch all pods and solve them together in one optimization run.

**Implementation verification**:
```go
// mode_types.go:22-23
// ModeManualBlocking blocks the normal scheduling flow until /solve is called.
ModeManualBlocking

// mode_helpers.go:14-15
func isManualBlockingMode() bool { return OptimizeMode == ModeManualBlocking }

// hook_preenqueue.go:51-57 - Blocks pods in ManualBlocking mode
if isManualBlockingMode() {
    klog.V(MyV).InfoS(msg(stage, InfoPendingPod), "pod", klog.KObj(pending))
    // Pods are blocked from entering the scheduling queue
    return framework.NewStatus(framework.Pending, msg(stage, InfoPendingPod))
}
```

---

### 3. Per Pod Mode

**File**: `mode_types.go:8-9`  
**Constant**: `ModePerPod`

**Description**:
Per Pod mode triggers optimization **for every pod that hits PostFilter**, meaning every pod that the default scheduler fails to place.

**How it works**:
- When a pod enters the scheduler, it first attempts normal scheduling
- If the default scheduler cannot place the pod (reaches PostFilter stage), the plugin immediately triggers optimization
- The solver runs for that specific pod plus any other pending pods
- The optimization is **always synchronous** in PerPod mode (see `mode_helpers.go:21-26`)
- New pods are blocked while optimization is running

**Key characteristics**:
- **Immediate response**: Optimization happens as soon as a pod fails to schedule
- **Always synchronous**: Blocks scheduling while solver runs
- **Most aggressive**: Highest optimization frequency, best for small clusters
- **Overhead**: Can be expensive for large clusters with many pods

**Use case**: Small clusters where immediate optimal placement is desired for every pod.

**Implementation verification**:
```go
// mode_types.go:8-9
// ModePerPod optimizes for every new pod.
ModePerPod

// mode_helpers.go:8-9
func isPerPodMode() bool { return OptimizeMode == ModePerPod }

// hook_postfilter.go:14-49 - Triggers optimization for every failed pod
func (pl *SharedState) PostFilter(ctx context.Context, _ *framework.CycleState, pending *v1.Pod, _ framework.NodeToStatusMap) (*framework.PostFilterResult, *framework.Status) {
    // Only proceed if PerPod is enabled
    if !isPerPodMode() {
        return nil, framework.NewStatus(framework.Unschedulable, ...)
    }
    
    // Run optimization flow for the pod
    plan, _, _, _, _, err := pl.runOptimizationFlow(ctx, pending)
    // ...
}

// mode_helpers.go:21-26 - PerPod is always synchronous
func isAsyncSolving() bool {
    if OptimizeMode == ModePerPod {
        return false  // Always synchronous
    }
    return !OptimizeSolveSynch
}
```

---

### 4. Periodic Mode

**File**: `mode_types.go:11-12`  
**Constant**: `ModePeriodic`

**Description**:
Periodic mode triggers optimization **at fixed time intervals**, regardless of cluster activity.

**How it works**:
- A background loop runs continuously (see `loop_periodic.go`)
- Every `OPTIMIZE_PERIODIC_INTERVAL` (e.g., 30s), the loop checks for pending pods
- If pending pods exist, optimization is triggered for the entire pending set
- Normal scheduling continues between optimization cycles
- The mode **does not** cancel ongoing optimization when new pods arrive (configurable)

**Key characteristics**:
- **Predictable**: Runs on a fixed schedule
- **Batches work**: Optimizes all pending pods together
- **Moderate overhead**: Less frequent than PerPod, more frequent than Interlude
- **Clock-driven**: Independent of cluster state changes

**Use case**: Production clusters with predictable workload patterns where periodic batch optimization is desired.

**Implementation verification**:
```go
// mode_types.go:11-12
// ModePeriodic runs periodic optimization over the accumulated pending set.
ModePeriodic

// loop_periodic.go:9-22
func (pl *SharedState) loopPeriodic(ctx context.Context) {
    if OptimizePeriodicInterval <= 1 {
        OptimizePeriodicInterval = 2 * time.Second
    }

    cfg := OptimizeLoopConfig{
        Label:          "PeriodicLoop",
        Interval:       OptimizePeriodicInterval,  // Fixed interval
        InterludeDelay: 0,                         // No idle window requirement
        CancelOnChange: false,                     // Continue even if pods arrive
    }
    optimizeBackgroundLoopFunc(pl, ctx, cfg)
}

// loop_helpers.go:33-38 - Started when mode is Periodic
switch OptimizeMode {
case ModePeriodic:
    go pl.loopPeriodic(ctx)
case ModeInterlude:
    go pl.loopInterlude(ctx)
}
```

---

### 5. Interlude Mode

**File**: `mode_types.go:14-16`  
**Constant**: `ModeInterlude`

**Description**:
Interlude mode triggers optimization **only during "quiet" periods** when the pending pod set has been stable for a configured amount of time.

**How it works**:
- A background loop checks cluster state at regular intervals (e.g., every 250ms)
- Tracks changes to the pending pod set
- Only triggers optimization when the pending set has been **stable** (unchanged) for at least `OPTIMIZE_INTERLUDE_DELAY` (e.g., 2s)
- If new pods arrive during solver execution, the run is **cancelled** and reset (see `loop_helpers.go:131-140`)
- Waits for another stable period before retrying

**Key characteristics**:
- **Event-driven**: Responds to cluster stability, not time
- **Least intrusive**: Only runs when cluster is "quiet"
- **Adaptive**: Automatically adjusts to workload patterns
- **Smart batching**: Waits for workload bursts to settle before optimizing
- **Cancels on change**: Aborts if cluster state changes during solving

**Use case**: Production clusters with bursty workloads where you want to optimize during lulls without interfering with active scheduling.

**Implementation verification**:
```go
// mode_types.go:14-16
// ModeInterlude runs optimization only during "quiet" periods where the
// pending set has been stable for some time.
ModeInterlude

// loop_interlude.go:9-27
func (pl *SharedState) loopInterlude(ctx context.Context) {
    delay := OptimizeInterludeDelay           // Stability requirement
    if delay <= 0 {
        delay = 2 * time.Second
    }
    checkInterval := OptimizeInterludeCheckInterval
    if checkInterval <= 0 {
        checkInterval = 250 * time.Millisecond
    }

    cfg := OptimizeLoopConfig{
        Label:          "InterludeLoop",
        Interval:       checkInterval,  // Check frequently
        InterludeDelay: delay,          // Require stability for this long
        CancelOnChange: true,           // Cancel if pending set changes
    }
    optimizeBackgroundLoopFunc(pl, ctx, cfg)
}

// loop_helpers.go:169-178 - Tracks pending set changes and resets timer
if !sameUIDSet(currentSet, lastPendingSet) {
    lastPendingSet = cloneUIDSet(currentSet)
    lastChange = time.Now()  // Reset the stability timer
    klog.V(MyV).InfoS(
        msg(cfg.Label, "pending set changed; reset idle timer"),
        "pending", pendingCount,
    )
    timer.Reset(interval)
    continue
}

// loop_helpers.go:180-187 - Checks if stable for long enough
if cfg.InterludeDelay > 0 {
    idleFor := time.Since(lastChange)
    if idleFor < cfg.InterludeDelay {
        timer.Reset(cfg.InterludeDelay - idleFor)  // Not stable long enough
        continue
    }
}

// loop_helpers.go:131-140 - Cancels if pending set changes during solving
if cfg.CancelOnChange && !sameUIDSet(currentSet, baselineSet) {
    klog.V(MyV).InfoS(
        msg(cfg.Label, "pending set changed; cancelling run"),
        "pending", pendingCount,
    )
    runCancel()  // Cancel the ongoing solver
}
```

---

## Sync vs Async Solving

The plugin supports two solving strategies that control **when** the `Active` lock is acquired during optimization:

### Synchronous Solving

**Configuration**: `OptimizeSolveSynch = true` or `OPTIMIZE_SOLVE_SYNCH=true`

**Behavior**:
- The `Active` lock is acquired **immediately** at the start of `runOptimizationFlow()`
- All new pods are **blocked** from scheduling while:
  - The solver is computing the plan
  - The plan is being applied (evictions + recreations)
- Ensures no cluster state changes occur during solving and plan application
- Provides the strongest consistency guarantees

**When to use**:
- When you need guaranteed cluster state consistency
- When solver runtime is short (< 1-2 seconds)
- In testing/evaluation scenarios
- PerPod mode (always uses sync)

**Implementation**:
```go
// optimization_flow.go:19-26
// Periodic-sync/Per-pod: take Active early.
if !isAsyncSolving() {
    if !pl.tryEnterActive() {
        klog.InfoS(msg(strategy, InfoActivePlanInProgress))
        return nil, nil, "", nil, nil, ErrActiveInProgress
    }
}
// ... snapshot cluster state, run solver ...
```

**Timeline**:
```
Time ──────────────────────────────────────────────────────────►
      │◄─────── Active lock held ──────────►│
      │                                      │
      ├─ Snapshot cluster                   │
      ├─ Run solver (blocked)               │
      ├─ Validate plan                      │
      ├─ Apply plan (evict/recreate)        │
      └─ Release lock                       │
      
New pods: BLOCKED ──────────────────────────┼─ ALLOWED
```

---

### Asynchronous Solving

**Configuration**: `OptimizeSolveSynch = false` or `OPTIMIZE_SOLVE_SYNCH=false`

**Behavior**:
- The `Active` lock is acquired **only after** the solver completes successfully
- New pods can continue scheduling while the solver is running
- Before applying the plan, the cluster state is **re-validated** to ensure the plan is still valid
- If the cluster changed significantly, the plan is discarded and pods continue with normal scheduling
- Only blocks new pods during plan application

**When to use**:
- When solver runtime is long (> 2-5 seconds)
- In production with high pod arrival rates
- When you want to minimize scheduling disruption
- Periodic and Interlude modes benefit most from async

**Implementation**:
```go
// optimization_flow.go:19-26 - Does NOT take Active early in async mode
if !isAsyncSolving() {
    // ... only sync mode takes lock here ...
}

// optimization_flow.go:63-70 - Takes Active only after solver succeeds
// Async modes: take Active now that we know it is worth applying the plan.
if isAsyncSolving() {
    if !pl.tryEnterActive() {
        klog.InfoS(msg(strategy, InfoActivePlanInProgress))
        pl.exportSolverStatsToConfigMap(...)
        return nil, nil, "", nil, nil, ErrActiveInProgress
    }
}
```

**Timeline**:
```
Time ──────────────────────────────────────────────────────────►
      │                           │◄─ Active lock held ─►│
      │                           │                       │
      ├─ Snapshot cluster         │                       │
      ├─ Run solver (background)  │                       │
      │   ▲                       │                       │
      │   └─ pods can schedule ───┤                       │
      │                           │                       │
      │                           ├─ Re-validate plan     │
      │                           ├─ Apply plan           │
      │                           └─ Release lock         │
      
New pods: ALLOWED ────────────────┼─ BLOCKED ────────────┼─ ALLOWED
```

**Trade-offs**:

| Aspect | Synchronous | Asynchronous |
|--------|------------|--------------|
| **Consistency** | Guaranteed - cluster frozen | Best effort - may discard plan |
| **Blocking time** | Entire solver + apply time | Only plan application time |
| **Solver overhead** | All pods blocked | Background, non-blocking |
| **Wasted work** | None | May compute plans that get discarded |
| **Pod latency** | Higher for new pods | Lower for new pods |
| **Best for** | Fast solvers, small clusters | Slow solvers, large clusters |

---

## Concurrency Control

The plugin implements a sophisticated concurrency control mechanism to ensure correctness and prevent race conditions.

### The Active Lock

**Type**: `atomic.Bool` (see `plugin_types.go:19`)  
**Purpose**: Ensures only one optimization can run at a time

**Key operations**:
```go
// plan_helpers.go:40-42
func (pl *SharedState) tryEnterActive() bool {
    return pl.Active.CompareAndSwap(false, true)  // Atomic CAS
}

// plan_helpers.go:45-47
func (pl *SharedState) leaveActive() {
    pl.Active.Store(false)  // Atomic store
}
```

**Why it works**:
- Uses **atomic Compare-And-Swap (CAS)** operation
- CAS is a CPU-level atomic operation that:
  1. Checks if current value is `false`
  2. If yes, sets it to `true` and returns `true`
  3. If no (already `true`), returns `false`
  4. Steps 1-2 happen **atomically** - no race condition possible
- Only one goroutine can successfully CAS from `false` to `true`
- All others get `false` and back off

### The Active Plan Pointer

**Type**: `atomic.Pointer[ActivePlan]` (see `plugin_types.go:21`)  
**Purpose**: Stores the currently executing plan

**Key operations**:
```go
// plan_helpers.go:50-52
func (pl *SharedState) getActivePlan() *ActivePlan {
    return pl.ActivePlan.Load()  // Atomic load
}

// plan_helpers.go:55-60
func (pl *SharedState) tryClearActivePlan(ap *ActivePlan) bool {
    if ap == nil {
        return false
    }
    return pl.ActivePlan.CompareAndSwap(ap, nil)  // Atomic CAS
}
```

**Why it works**:
- All reads/writes use atomic operations
- `Load()` gives consistent view of the pointer
- `CompareAndSwap(ap, nil)` ensures only the owner can clear it
- Prevents ABA problem (plan replaced while we're working on it)

### Blocked Pod Set

**Type**: `*PodSet` with internal mutex (see `plugin_types.go:23`)  
**Purpose**: Tracks pods blocked while a plan is active

**Thread safety**:
- All operations on `BlockedWhileActive` are mutex-protected internally
- Safe for concurrent access from multiple scheduler workers

### Workload Quotas

**Type**: `WorkloadQuotasAtomics = map[string]map[string]*atomic.Int32`  
**Purpose**: Track remaining placement slots per workload per node

**Key operations**:
```go
// Each counter is an atomic.Int32
// Multiple workers can decrement simultaneously:
if quota.Add(-1) >= 0 {
    // Pod can be placed
} else {
    quota.Add(1)  // Restore and fail
}
```

**Why it works**:
- `Add(-1)` is atomic - no race condition
- Returns the *new* value after decrement
- If new value >= 0, this goroutine "won" a slot
- If new value < 0, too many goroutines tried - restore and fail

---

## Implementation Correctness

### Why the concurrency model is correct

#### 1. **Mutual Exclusion via Active Lock**

**Property**: At most one optimization can hold the Active lock
**Mechanism**: Atomic CAS on `pl.Active`
**Proof**:
- Initial state: `Active = false`
- When goroutine G1 calls `tryEnterActive()`:
  - CAS(false, true) succeeds → G1 holds lock
  - `Active = true`
- When goroutine G2 calls `tryEnterActive()`:
  - CAS(false, true) fails because `Active = true`
  - G2 returns `false` and backs off
- Only when G1 calls `leaveActive()` (Store(false)) can another goroutine enter

#### 2. **No Concurrent Plan Modifications**

**Property**: A plan is modified only by its creator
**Mechanism**: Atomic pointer operations
**Proof**:
- Plan is stored via `ActivePlan.Store(ap)` by optimizer
- Plan is read via `ActivePlan.Load()` by workers
- Plan is cleared via `ActivePlan.CompareAndSwap(ap, nil)` by plan completion watch
- CAS ensures only the exact plan pointer can be cleared
- No goroutine can modify a plan - they can only read it or atomically swap it

#### 3. **Safe Pod Blocking/Unblocking**

**Property**: Pods are safely blocked and released
**Mechanism**: Synchronized operations
**Flow**:
```go
// Blocking (PreEnqueue/PostFilter)
if ap := pl.getActivePlan(); ap != nil {
    pl.BlockedWhileActive.AddPod(pending)  // Mutex protected
    return Pending
}

// Unblocking (onPlanCompleted)
pl.activatePods(pl.BlockedWhileActive, false, -1)  // Mutex protected
```

**Correctness**:
- `getActivePlan()` atomically reads the plan pointer
- If plan exists, pod is added to blocked set (mutex protected)
- When plan completes, all blocked pods are released atomically
- No pod can be "lost" between blocking and unblocking

#### 4. **Safe Quota Decrements**

**Property**: No over-subscription of node resources
**Mechanism**: Atomic counters
**Proof**:
- Quota counter initialized to N (number of pods to place on node)
- Each worker does: `newVal := counter.Add(-1)`
- If `newVal >= 0`: Worker successfully claimed a slot
- If `newVal < 0`: Too many workers tried, this one restores: `counter.Add(1)`
- Mathematical invariant: Sum of successful claims ≤ N

#### 5. **Async Mode Re-validation**

**Property**: Plan is applied only if still valid
**Mechanism**: Deferred lock acquisition
**Flow**:
```go
// Async mode
// 1. Snapshot cluster (no lock)
nodes, pods, inp, baseline, pending, err := pl.planContext(preemptor)

// 2. Run solver (no lock, other pods can schedule)
bestName, hadImproving, bestAttempt, attempts := pl.planComputation(...)

// 3. Only now, try to take lock
if isAsyncSolving() {
    if !pl.tryEnterActive() {
        return ErrActiveInProgress  // Another optimizer got here first
    }
}

// 4. Inside planActivation(), cluster state is implicitly re-checked:
//    - Pods that were pending might now be running
//    - Nodes might have different allocations
//    - These are handled by the eviction/recreation logic
```

**Correctness**:
- Lock is taken only after solver completes
- If another optimizer took the lock first, we abort
- Plan application automatically handles cluster changes:
  - Pods that are now running won't be in the pending set anymore
  - Node allocations are re-checked before placement
  - Invalid placements fail gracefully

---

## Summary Table

| Mode | Trigger | Sync/Async | Blocks Normal Scheduling | Background Loop | Use Case |
|------|---------|------------|-------------------------|----------------|----------|
| **Manual** | HTTP `/solve` | Both supported | No | No | Testing/Evaluation |
| **ManualBlocking** | HTTP `/solve` | Both supported | Yes (until triggered) | No | Testing/Evaluation |
| **PerPod** | Every PostFilter | Always Sync | Yes (during solve) | No | Small clusters |
| **Periodic** | Fixed intervals | Both supported | Depends on sync/async | Yes (periodic) | Predictable workloads |
| **Interlude** | Stable periods | Both supported | Depends on sync/async | Yes (adaptive) | Bursty workloads |

---

## Configuration Examples

### Manual Mode (Sync)
```yaml
extraEnvs:
  - name: OPTIMIZE_MODE
    value: "manual"
  - name: OPTIMIZE_SOLVE_SYNCH
    value: "true"
```

### Manual Blocking Mode (Sync)
```yaml
extraEnvs:
  - name: OPTIMIZE_MODE
    value: "manual_blocking"
  - name: OPTIMIZE_SOLVE_SYNCH
    value: "true"
```

### Per Pod Mode (Always Sync)
```yaml
extraEnvs:
  - name: OPTIMIZE_MODE
    value: "per_pod"
  # OPTIMIZE_SOLVE_SYNCH is ignored - always sync
```

### Periodic Mode (Async, 30s interval)
```yaml
extraEnvs:
  - name: OPTIMIZE_MODE
    value: "periodic"
  - name: OPTIMIZE_SOLVE_SYNCH
    value: "false"
  - name: OPTIMIZE_PERIODIC_INTERVAL
    value: "30s"
```

### Interlude Mode (Async, 2s stability required)
```yaml
extraEnvs:
  - name: OPTIMIZE_MODE
    value: "interlude"
  - name: OPTIMIZE_SOLVE_SYNCH
    value: "false"
  - name: OPTIMIZE_INTERLUDE_DELAY
    value: "2s"
  - name: OPTIMIZE_INTERLUDE_CHECK_INTERVAL
    value: "250ms"
```

---

## References

- Mode types: `pkg/mypriorityoptimizer/mode_types.go`
- Mode helpers: `pkg/mypriorityoptimizer/mode_helpers.go`
- Optimization flow: `pkg/mypriorityoptimizer/optimization_flow.go`
- Periodic loop: `pkg/mypriorityoptimizer/loop_periodic.go`
- Interlude loop: `pkg/mypriorityoptimizer/loop_interlude.go`
- PreEnqueue hook: `pkg/mypriorityoptimizer/hook_preenqueue.go`
- PostFilter hook: `pkg/mypriorityoptimizer/hook_postfilter.go`
- HTTP server: `pkg/mypriorityoptimizer/http_server.go`
- Plan helpers: `pkg/mypriorityoptimizer/plan_helpers.go`
- Plugin types: `pkg/mypriorityoptimizer/plugin_types.go`
