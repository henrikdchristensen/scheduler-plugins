# Supporting QoS Classes, Quotas, LimitRanges, and Affinities in MyPriorityOptimizer

## Overview

This document outlines the requirements and implementation strategy for supporting Kubernetes QoS classes, Resource Quotas, LimitRanges, and Pod Affinity/Anti-affinity constraints in the **MyPriorityOptimizer** plugin and its Python solver.

Currently, the plugin makes several simplifying assumptions:
- **QoS Classes**: All pods are treated as `Guaranteed` pods (i.e., resource requests equal limits)
- **Affinities**: Pod affinity and anti-affinity rules are ignored (see `IgnoreAffinity` flag)
- **Quotas**: Resource quotas are not considered during optimization
- **LimitRanges**: LimitRange constraints are not enforced

Supporting these features will make the plugin production-ready and ensure compliance with Kubernetes resource management best practices.

## Table of Contents

1. [Background](#background)
   - [QoS Classes](#qos-classes)
   - [Affinity and Anti-Affinity](#affinity-and-anti-affinity)
   - [Resource Quotas](#resource-quotas)
   - [LimitRanges](#limitranges)
2. [Current Implementation Analysis](#current-implementation-analysis)
3. [Implementation Requirements](#implementation-requirements)
   - [QoS Classes Support](#qos-classes-support)
   - [Affinity Support](#affinity-support)
   - [Resource Quotas Support](#resource-quotas-support)
   - [LimitRanges Support](#limitranges-support)
4. [Required Changes](#required-changes)
   - [Go Plugin Changes](#go-plugin-changes)
   - [Python Solver Changes](#python-solver-changes)
5. [Testing Strategy](#testing-strategy)
6. [Impact on Test Results Without These Features](#impact-on-test-results-without-these-features)
7. [Performance Considerations](#performance-considerations)
8. [References](#references)

---

## Background

### QoS Classes

Kubernetes assigns QoS classes to pods based on their resource requests and limits. Understanding these is critical for proper scheduling and preemption decisions.

#### The Three QoS Classes

1. **Guaranteed** (Highest Priority)
   - Every container in the pod must have:
     - CPU request = CPU limit
     - Memory request = Memory limit
   - Gets highest priority during eviction
   - Example:
     ```yaml
     resources:
       requests:
         cpu: "500m"
         memory: "256Mi"
       limits:
         cpu: "500m"
         memory: "256Mi"
     ```

2. **Burstable** (Medium Priority)
   - At least one container has a memory or CPU request
   - Does not meet criteria for Guaranteed
   - Can use more resources than requested (up to limits)
   - Example:
     ```yaml
     resources:
       requests:
         cpu: "250m"
         memory: "128Mi"
       limits:
         cpu: "500m"
         memory: "256Mi"
     ```

3. **BestEffort** (Lowest Priority)
   - No containers have CPU or memory requests/limits
   - First to be evicted under resource pressure
   - Example:
     ```yaml
     resources: {}
     ```

#### QoS Impact on Scheduling

- **Eviction Order**: Under resource pressure, Kubernetes evicts in order: BestEffort → Burstable → Guaranteed
- **Resource Accounting**: Different QoS classes affect how resources are accounted:
  - Guaranteed: Uses requests (= limits) for scheduling
  - Burstable: Uses requests for scheduling, but can burst to limits
  - BestEffort: Uses zero for scheduling but consumes actual resources

**Reference**: [Kubernetes QoS Classes Guide](https://medium.com/@muppedaanvesh/a-hands-on-guide-to-kubernetes-qos-classes-571b5f8f7e58)

### Affinity and Anti-Affinity

Pod affinity and anti-affinity allow you to constrain which nodes your pod can be scheduled on based on labels and existing pods.

#### Types of Affinity

1. **Node Affinity**
   - Similar to `nodeSelector` but more expressive
   - Required vs Preferred rules
   - Example:
     ```yaml
     affinity:
       nodeAffinity:
         requiredDuringSchedulingIgnoredDuringExecution:
           nodeSelectorTerms:
           - matchExpressions:
             - key: kubernetes.io/zone
               operator: In
               values:
               - us-west-1a
     ```

2. **Pod Affinity**
   - Schedule pod on same node/zone as other pods matching selector
   - Example:
     ```yaml
     affinity:
       podAffinity:
         requiredDuringSchedulingIgnoredDuringExecution:
         - labelSelector:
             matchLabels:
               app: database
           topologyKey: kubernetes.io/hostname
     ```

3. **Pod Anti-Affinity**
   - Prevent scheduling on same node/zone as pods matching selector
   - Common for high availability (spread replicas across zones)
   - Example:
     ```yaml
     affinity:
       podAntiAffinity:
         preferredDuringSchedulingIgnoredDuringExecution:
         - weight: 100
           podAffinityTerm:
             labelSelector:
               matchLabels:
                 app: web
             topologyKey: kubernetes.io/hostname
     ```

#### Required vs Preferred

- **Required** (`requiredDuringSchedulingIgnoredDuringExecution`): Hard constraint - must be satisfied
- **Preferred** (`preferredDuringSchedulingIgnoredDuringExecution`): Soft constraint - should be satisfied if possible

### Resource Quotas

ResourceQuota objects set aggregate resource consumption limits per namespace.

#### Quota Types

1. **Compute Resource Quotas**
   ```yaml
   apiVersion: v1
   kind: ResourceQuota
   metadata:
     name: compute-quota
     namespace: production
   spec:
     hard:
       requests.cpu: "10"
       requests.memory: "20Gi"
       limits.cpu: "20"
       limits.memory: "40Gi"
   ```

2. **Object Count Quotas**
   ```yaml
   spec:
     hard:
       pods: "100"
       services: "10"
       persistentvolumeclaims: "20"
   ```

3. **QoS-Specific Quotas**
   ```yaml
   spec:
     hard:
       requests.cpu: "10"
       limits.cpu: "20"
     scopeSelector:
       matchExpressions:
       - operator: In
         scopeName: PriorityClass
         values: ["high-priority"]
   ```

#### Quota Enforcement

- Quotas are enforced at **namespace level**
- Pod creation/updates are rejected if they would exceed quota
- Scheduler must respect quota limits when placing pods

**Reference**: [Debug Preempted Pods with Quotas](https://medium.com/codex/what-you-need-to-know-to-debug-a-preempted-pod-on-kubernetes-1c956eec3f35)

### LimitRanges

LimitRange objects set default resource requests/limits and enforce min/max constraints per pod/container.

#### LimitRange Capabilities

1. **Set Default Requests/Limits**
   ```yaml
   apiVersion: v1
   kind: LimitRange
   metadata:
     name: limit-range
     namespace: production
   spec:
     limits:
     - default:
         cpu: "500m"
         memory: "512Mi"
       defaultRequest:
         cpu: "250m"
         memory: "256Mi"
       type: Container
   ```

2. **Enforce Min/Max**
   ```yaml
   spec:
     limits:
     - max:
         cpu: "2"
         memory: "4Gi"
       min:
         cpu: "100m"
         memory: "128Mi"
       type: Container
   ```

3. **Limit Ratios**
   ```yaml
   spec:
     limits:
     - maxLimitRequestRatio:
         cpu: "4"
         memory: "2"
       type: Container
   ```

#### LimitRange Impact

- Applied when pods are **admitted** to the cluster
- If pod doesn't specify requests/limits, defaults are injected
- Pod creation is rejected if constraints are violated

---

## Current Implementation Analysis

### What the Plugin Currently Does

The MyPriorityOptimizer plugin:
1. Collects all running and pending pods
2. Converts them to `SolverPod` objects with:
   - Resource requests (CPU, memory)
   - Priority
   - Protection status
   - Current node assignment
3. Passes this data to the Python CP-SAT solver
4. Solver computes optimal placement maximizing high-priority pod placement
5. Plugin applies the solution via evictions and placements

### Current Limitations

#### 1. QoS Classes (Ignored)

**Current Code** (`pkg/mypriorityoptimizer/solver_types.go`):
```go
type SolverPod struct {
    UID         types.UID `json:"uid"`
    ReqCPUm     int64     `json:"req_cpu_m"`
    ReqMemBytes int64     `json:"req_mem_bytes"`
    Priority    int32     `json:"priority"`
    // ...
}
```

**Problem**:
- Only `requests` are captured, not `limits`
- No QoS class field
- Solver treats all pods as if they're Guaranteed
- BestEffort pods (with zero requests) may not be handled correctly
- Eviction priority doesn't account for QoS class

#### 2. Affinities (Explicitly Ignored)

**Current Code** (`scripts/python_solver/main.py`):
```python
ignore_affinity = bool(instance.get("ignore_affinity", True))  # TODO: consider to use this
```

**Problem**:
- All affinity rules are ignored
- Solutions may violate required affinity constraints
- Pod anti-affinity for HA is not enforced
- No consideration of topology spread

#### 3. Quotas (Not Considered)

**Problem**:
- No namespace-level resource tracking
- Solver may propose solutions that exceed namespace quotas
- Plugin may attempt to schedule pods that will be rejected by quota admission

#### 4. LimitRanges (Not Considered)

**Problem**:
- Pod resource values are taken as-is from the pod spec
- No validation against namespace LimitRanges
- Default requests/limits are not applied

---

## Implementation Requirements

### QoS Classes Support

#### Requirements

1. **Data Collection** (Go Plugin)
   - Extract QoS class for each pod
   - Capture both `requests` and `limits` for all resources
   - Pass QoS class to solver

2. **Eviction Priority** (Python Solver)
   - When choosing pods to evict, prefer lower QoS classes:
     - BestEffort evicted first
     - Then Burstable
     - Then Guaranteed (last resort)
   - Within same QoS class, use priority as tiebreaker

3. **Resource Accounting** (Python Solver)
   - Guaranteed: Use requests for bin-packing (requests == limits)
   - Burstable: Use requests for bin-packing, track limits for overcommit scenarios
   - BestEffort: Use zero for requests (but acknowledge they consume real resources)

4. **Preemption Logic** (Go Plugin)
   - Follow Kubernetes eviction semantics
   - Lower QoS class → evicted first under resource pressure
   - Update `Protected` flag based on QoS + Priority combination

#### Data Structure Changes

**Go Plugin** (`solver_types.go`):
```go
type SolverPod struct {
    UID         types.UID `json:"uid"`
    Namespace   string    `json:"namespace"`
    Name        string    `json:"name"`
    ReqCPUm     int64     `json:"req_cpu_m"`
    ReqMemBytes int64     `json:"req_mem_bytes"`
    LimCPUm     int64     `json:"lim_cpu_m"`      // NEW
    LimMemBytes int64     `json:"lim_mem_bytes"`  // NEW
    QoSClass    string    `json:"qos_class"`      // NEW: "Guaranteed", "Burstable", "BestEffort"
    Priority    int32     `json:"priority"`
    Protected   bool      `json:"protected,omitempty"`
    Node        string    `json:"node"`
}
```

**Python Solver**:
```python
def p_qos_class(i):  
    """Return QoS class: 'Guaranteed', 'Burstable', or 'BestEffort'"""
    return pods[i].get("qos_class", "Guaranteed")

# QoS precedence for eviction (lower = evict first)
QOS_PRIORITY = {
    "BestEffort": 1,
    "Burstable": 2,
    "Guaranteed": 3,
}
```

#### Implementation Steps

1. **Go Plugin**:
   - Use `k8s.io/kubernetes/pkg/apis/core/v1/helper/qos` package to determine QoS class
   - Extract limits from pod spec in addition to requests
   - Add QoS class to `SolverPod` when building solver input
   - Example:
     ```go
     import v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
     
     qosClass := v1qos.GetPodQOS(pod)
     solverPod.QoSClass = string(qosClass) // "Guaranteed", "Burstable", or "BestEffort"
     ```

2. **Python Solver**:
   - Add QoS class as a factor in disruption objective
   - When optimizing for minimal disruption, weight evictions by QoS priority
   - Modified disruption term:
     ```python
     # Original: disr_expr = sum(placed[i] - 2 * orig_node(i) for i in running_ge)
     # New: Weight by QoS class
     qos_weight = {i: QOS_PRIORITY[p_qos_class(i)] for i in running_ge}
     disr_expr = sum(
         qos_weight[i] * (placed[i] - 2 * orig_node(i)) 
         for i in running_ge
     )
     ```

3. **Testing**:
   - Create workloads with mixed QoS classes
   - Verify BestEffort pods are evicted before Burstable
   - Verify Burstable pods are evicted before Guaranteed
   - Test edge cases: all BestEffort, all Guaranteed, etc.

### Affinity Support

#### Requirements

1. **Data Collection** (Go Plugin)
   - Extract node affinity rules from pod spec
   - Extract pod affinity rules
   - Extract pod anti-affinity rules
   - Collect node labels
   - Track pod labels and namespaces

2. **Constraint Encoding** (Python Solver)
   - **Required** affinity: Hard constraints (must satisfy)
   - **Preferred** affinity: Soft constraints (maximize satisfaction)
   - Support topology keys (hostname, zone, region, etc.)

3. **Node Affinity**
   - Encode `requiredDuringSchedulingIgnoredDuringExecution` as binary constraints
   - Encode `preferredDuringSchedulingIgnoredDuringExecution` in objective

4. **Pod Affinity**
   - For each affinity rule, identify matching pods
   - Add constraints based on topology key (same node, same zone, etc.)

5. **Pod Anti-Affinity**
   - For each anti-affinity rule, identify matching pods
   - Add exclusion constraints based on topology key

#### Data Structure Changes

**Go Plugin** (`solver_types.go`):
```go
type SolverNode struct {
    Name        string            `json:"name"`
    CapCPUm     int64             `json:"cap_cpu_m"`
    CapMemBytes int64             `json:"cap_mem_bytes"`
    Labels      map[string]string `json:"labels,omitempty"` // Already exists - contains topology info
}

type SolverPod struct {
    // ... existing fields ...
    Labels       map[string]string   `json:"labels,omitempty"`        // NEW
    Affinity     *SolverAffinity     `json:"affinity,omitempty"`      // NEW
}

// NEW: Affinity structures
type SolverAffinity struct {
    NodeAffinity    *SolverNodeAffinity    `json:"node_affinity,omitempty"`
    PodAffinity     *SolverPodAffinity     `json:"pod_affinity,omitempty"`
    PodAntiAffinity *SolverPodAntiAffinity `json:"pod_anti_affinity,omitempty"`
}

type SolverNodeAffinity struct {
    Required  []SolverNodeSelector `json:"required,omitempty"`
    Preferred []SolverNodeSelector `json:"preferred,omitempty"`
}

type SolverNodeSelector struct {
    MatchExpressions []SolverMatchExpression `json:"match_expressions,omitempty"`
    Weight           int32                   `json:"weight,omitempty"` // For preferred
}

type SolverMatchExpression struct {
    Key      string   `json:"key"`
    Operator string   `json:"operator"` // "In", "NotIn", "Exists", "DoesNotExist", "Gt", "Lt"
    Values   []string `json:"values,omitempty"`
}

type SolverPodAffinity struct {
    Required  []SolverPodAffinityTerm `json:"required,omitempty"`
    Preferred []SolverPodAffinityTerm `json:"preferred,omitempty"`
}

type SolverPodAffinityTerm struct {
    LabelSelector SolverLabelSelector `json:"label_selector"`
    TopologyKey   string              `json:"topology_key"`
    Namespaces    []string            `json:"namespaces,omitempty"`
    Weight        int32               `json:"weight,omitempty"` // For preferred
}

type SolverLabelSelector struct {
    MatchLabels      map[string]string       `json:"match_labels,omitempty"`
    MatchExpressions []SolverMatchExpression `json:"match_expressions,omitempty"`
}

type SolverPodAntiAffinity struct {
    Required  []SolverPodAffinityTerm `json:"required,omitempty"`
    Preferred []SolverPodAffinityTerm `json:"preferred,omitempty"`
}
```

**Python Solver**:
```python
# Input structure
def p_affinity(i):
    return pods[i].get("affinity") or {}

def n_labels(j):
    return nodes[j].get("labels") or {}

# Helper: Check if node matches node selector
def node_matches_selector(node_idx, selector):
    """Check if node matches the selector expressions"""
    node_labels = n_labels(node_idx)
    for expr in selector.get("match_expressions", []):
        key = expr["key"]
        operator = expr["operator"]
        values = expr.get("values", [])
        
        if operator == "In":
            if node_labels.get(key) not in values:
                return False
        elif operator == "NotIn":
            if node_labels.get(key) in values:
                return False
        elif operator == "Exists":
            if key not in node_labels:
                return False
        elif operator == "DoesNotExist":
            if key in node_labels:
                return False
        elif operator == "Gt":
            # Greater than - for numeric comparisons
            if key not in node_labels or len(values) == 0:
                return False
            try:
                node_value = float(node_labels[key])
                threshold = float(values[0])
                if node_value <= threshold:
                    return False
            except (ValueError, TypeError):
                return False
        elif operator == "Lt":
            # Less than - for numeric comparisons
            if key not in node_labels or len(values) == 0:
                return False
            try:
                node_value = float(node_labels[key])
                threshold = float(values[0])
                if node_value >= threshold:
                    return False
            except (ValueError, TypeError):
                return False
    return True

# Helper: Get pods matching label selector
def pods_matching_selector(selector, namespace_filter=None):
    """Return indices of pods matching the label selector"""
    matching = []
    for i in range(num_pods):
        if namespace_filter and p_namespace(i) not in namespace_filter:
            continue
        pod_labels = pods[i].get("labels") or {}
        # Check matchLabels
        match_labels = selector.get("match_labels") or {}
        if not all(pod_labels.get(k) == v for k, v in match_labels.items()):
            continue
        # Check matchExpressions
        # ... (similar to node_matches_selector)
        matching.append(i)
    return matching

# Helper: Get nodes in same topology domain
def nodes_in_topology(node_idx, topology_key):
    """Return indices of nodes with same value for topology_key"""
    node_labels = n_labels(node_idx)
    if topology_key not in node_labels:
        return []
    topology_value = node_labels[topology_key]
    return [j for j in range(num_nodes) 
            if n_labels(j).get(topology_key) == topology_value]
```

#### Implementation Steps

1. **Go Plugin**:
   - Extract affinity spec from `pod.Spec.Affinity`
   - Convert to simplified `SolverAffinity` structure
   - Collect and pass node labels and topology information
   - Example:
     ```go
     if pod.Spec.Affinity != nil {
         solverPod.Affinity = convertAffinity(pod.Spec.Affinity)
     }
     solverPod.Labels = pod.Labels
     ```

2. **Python Solver** - Node Affinity Constraints:
   ```python
   # Required node affinity
   for i in range(num_pods):
       aff = p_affinity(i)
       node_aff = aff.get("node_affinity")
       if not node_aff:
           continue
       
       # Required terms (hard constraint)
       required = node_aff.get("required") or []
       for selector in required:
           # Pod i can only be assigned to nodes matching this selector
           eligible_for_selector = [
               j for j in eligible_nodes[i]
               if node_matches_selector(j, selector)
           ]
           # Update eligible_nodes[i] to intersection
           eligible_nodes[i] = [j for j in eligible_nodes[i] 
                               if j in eligible_for_selector]
       
       # Preferred terms (soft constraint - add to objective)
       preferred = node_aff.get("preferred") or []
       # Add preference terms to objective function
   ```

3. **Python Solver** - Pod Affinity Constraints:
   ```python
   # Required pod affinity
   for i in range(num_pods):
       aff = p_affinity(i)
       pod_aff = aff.get("pod_affinity")
       if not pod_aff:
           continue
       
       required = pod_aff.get("required") or []
       for term in required:
           # Find pods matching the label selector
           target_pods = pods_matching_selector(
               term["label_selector"],
               term.get("namespaces")
           )
           if not target_pods:
               # No matching pods - constraint cannot be satisfied
               # Mark pod as unschedulable or relax constraint
               continue
           
           topology_key = term["topology_key"]
           
           # For each possible node j for pod i:
           # Pod i can only be on node j if at least one target pod
           # is on a node with the same topology_key value
           for local, j in enumerate(eligible_nodes[i]):
               topo_nodes = nodes_in_topology(j, topology_key)
               # Check if any target pod is assigned to topo_nodes
               can_assign = model.NewBoolVar(f"can_assign_{i}_{j}_affinity")
               
               # Build list of assignment variables for target pods in same topology
               clauses = []
               for target_i in target_pods:
                   for target_j in topo_nodes:
                       if target_j in eligible_nodes[target_i]:
                           pos = eligible_pos[target_i][target_j]
                           clauses.append(assign[target_i][pos])
               
               # can_assign == 1 iff at least one target pod is on same topology
               # Use AddBoolOr to create logical OR constraint
               if clauses:
                   model.AddBoolOr(clauses).OnlyEnforceIf(can_assign)
                   model.AddBoolAnd([c.Not() for c in clauses]).OnlyEnforceIf(can_assign.Not())
               else:
                   model.Add(can_assign == 0)
               
               # If assign[i][local] == 1, then can_assign must be 1
               model.Add(assign[i][local] <= can_assign)
   ```

4. **Python Solver** - Pod Anti-Affinity Constraints:
   ```python
   # Required pod anti-affinity
   for i in range(num_pods):
       aff = p_affinity(i)
       pod_anti_aff = aff.get("pod_anti_affinity")
       if not pod_anti_aff:
           continue
       
       required = pod_anti_aff.get("required") or []
       for term in required:
           target_pods = pods_matching_selector(
               term["label_selector"],
               term.get("namespaces")
           )
           topology_key = term["topology_key"]
           
           # For each possible node j for pod i:
           # Pod i cannot be on node j if any target pod is on a node
           # with the same topology_key value
           for local, j in enumerate(eligible_nodes[i]):
               topo_nodes = nodes_in_topology(j, topology_key)
               
               # If any target pod is assigned to topo_nodes, pod i cannot be on j
               for target_i in target_pods:
                   if target_i == i:  # Don't anti-affinity with self
                       continue
                   for target_j in topo_nodes:
                       if target_j in eligible_nodes[target_i]:
                           pos_target = eligible_pos[target_i][target_j]
                           # Cannot both be assigned
                           model.Add(assign[i][local] + assign[target_i][pos_target] <= 1)
   ```

5. **Testing**:
   - Test required node affinity (pod must schedule on specific nodes)
   - Test preferred node affinity (pod prefers specific nodes)
   - Test pod affinity (co-location of related pods)
   - Test pod anti-affinity (spread pods across topology domains)
   - Test combinations of affinity types
   - Verify infeasible scenarios are detected

### Resource Quotas Support

#### Requirements

1. **Data Collection** (Go Plugin)
   - Query ResourceQuota objects for each namespace
   - Track current resource usage per namespace
   - Calculate remaining quota capacity

2. **Quota Validation** (Go Plugin)
   - Before passing to solver, verify quotas won't be exceeded
   - Filter out pods that would violate quotas
   - Or pass quota constraints to solver

3. **Quota-Aware Optimization** (Python Solver)
   - Add namespace-level resource constraints
   - Ensure total pod requests per namespace ≤ quota

4. **QoS-Scoped Quotas**
   - Handle quotas that apply only to specific QoS classes
   - Handle quotas scoped by PriorityClass

#### Data Structure Changes

**Go Plugin** (`solver_types.go`):
```go
type SolverInput struct {
    // ... existing fields ...
    Quotas []NamespaceQuota `json:"quotas,omitempty"` // NEW
}

// NEW: Namespace quota information
type NamespaceQuota struct {
    Namespace     string            `json:"namespace"`
    HardLimits    ResourceLimits    `json:"hard_limits"`
    CurrentUsage  ResourceLimits    `json:"current_usage"`
    Scopes        []string          `json:"scopes,omitempty"`      // e.g., ["BestEffort"], ["NotTerminating"]
    ScopeSelector *ScopeSelector    `json:"scope_selector,omitempty"`
}

type ResourceLimits struct {
    RequestsCPUm     int64 `json:"requests_cpu_m,omitempty"`
    RequestsMemBytes int64 `json:"requests_mem_bytes,omitempty"`
    LimitsCPUm       int64 `json:"limits_cpu_m,omitempty"`
    LimitsMemBytes   int64 `json:"limits_mem_bytes,omitempty"`
    PodsCount        int64 `json:"pods_count,omitempty"`
}

type ScopeSelector struct {
    MatchExpressions []ScopeSelectorRequirement `json:"match_expressions,omitempty"`
}

type ScopeSelectorRequirement struct {
    ScopeName string   `json:"scope_name"` // e.g., "PriorityClass"
    Operator  string   `json:"operator"`   // "In", "NotIn", "Exists", "DoesNotExist"
    Values    []string `json:"values,omitempty"`
}
```

**Python Solver**:
```python
# Group pods by namespace
pods_by_namespace = {}
for i in range(num_pods):
    ns = p_namespace(i)
    if ns not in pods_by_namespace:
        pods_by_namespace[ns] = []
    pods_by_namespace[ns].append(i)

# Add quota constraints
quotas = instance.get("quotas") or []
for quota in quotas:
    ns = quota["namespace"]
    hard = quota["hard_limits"]
    current = quota["current_usage"]
    
    # Get pods in this namespace
    ns_pods = pods_by_namespace.get(ns, [])
    
    # Filter by scope if applicable
    if quota.get("scopes") or quota.get("scope_selector"):
        ns_pods = filter_by_quota_scope(ns_pods, quota)
    
    # Calculate remaining capacity
    remaining_cpu = hard.get("requests_cpu_m", 0) - current.get("requests_cpu_m", 0)
    remaining_mem = hard.get("requests_mem_bytes", 0) - current.get("requests_mem_bytes", 0)
    remaining_pods = hard.get("pods_count", 0) - current.get("pods_count", 0)
    
    # Add constraint: sum of placed pods' requests <= remaining
    if ns_pods:
        cpu_terms = [placed[i] * p_req_cpu_m(i) for i in ns_pods]
        mem_terms = [placed[i] * p_req_mem_bytes(i) for i in ns_pods]
        pod_terms = [placed[i] for i in ns_pods]
        
        if remaining_cpu >= 0:
            model.Add(sum(cpu_terms) <= remaining_cpu)
        if remaining_mem >= 0:
            model.Add(sum(mem_terms) <= remaining_mem)
        if remaining_pods >= 0:
            model.Add(sum(pod_terms) <= remaining_pods)
```

#### Implementation Steps

1. **Go Plugin**:
   - Use client to list ResourceQuota objects
   - Calculate current usage for each namespace (from running pods)
   - Convert to `NamespaceQuota` structures
   - Example:
     ```go
     quotas, err := plugin.handle.ClientSet().CoreV1().ResourceQuotas("").List(ctx, metav1.ListOptions{})
     for _, quota := range quotas.Items {
         nsQuota := convertQuota(&quota, currentUsage)
         solverInput.Quotas = append(solverInput.Quotas, nsQuota)
     }
     ```

2. **Python Solver**:
   - Add quota constraints before solving
   - Handle quota scopes (filter pods by QoS class, PriorityClass, etc.)
   - Ensure placed pods don't exceed quota

3. **Testing**:
   - Test with namespace CPU quota
   - Test with namespace memory quota
   - Test with pod count quota
   - Test QoS-scoped quotas (e.g., only Guaranteed pods)
   - Verify solver doesn't violate quotas
   - Test quota exceeded scenarios

### LimitRanges Support

#### Requirements

1. **Data Collection** (Go Plugin)
   - Query LimitRange objects for each namespace
   - Apply defaults to pods missing requests/limits
   - Validate pod specs against min/max constraints

2. **Pre-Processing** (Go Plugin)
   - Before passing pods to solver:
     - Apply default requests/limits from LimitRange
     - Validate pods against LimitRange constraints
     - Reject/skip pods that violate constraints

3. **Validation** (Go Plugin)
   - After solver computes plan, validate all pods still satisfy LimitRanges
   - Handle ratio constraints (max limit/request ratio)

#### Data Structure Changes

**Go Plugin** (`solver_types.go`):
```go
type SolverInput struct {
    // ... existing fields ...
    LimitRanges []NamespaceLimitRange `json:"limit_ranges,omitempty"` // NEW (optional, for validation)
}

// NEW: Namespace limit range information
type NamespaceLimitRange struct {
    Namespace string            `json:"namespace"`
    Limits    []LimitRangeItem  `json:"limits"`
}

type LimitRangeItem struct {
    Type                 string            `json:"type"` // "Pod", "Container", "PersistentVolumeClaim"
    Max                  ResourceLimits    `json:"max,omitempty"`
    Min                  ResourceLimits    `json:"min,omitempty"`
    Default              ResourceLimits    `json:"default,omitempty"`
    DefaultRequest       ResourceLimits    `json:"default_request,omitempty"`
    MaxLimitRequestRatio ResourceRatios    `json:"max_limit_request_ratio,omitempty"`
}

type ResourceRatios struct {
    CPU    float64 `json:"cpu,omitempty"`
    Memory float64 `json:"memory,omitempty"`
}
```

#### Implementation Steps

1. **Go Plugin - Pre-Processing**:
   - Query LimitRange objects before building solver input
   - For each pod:
     - If missing requests, apply `defaultRequest` from LimitRange
     - If missing limits, apply `default` from LimitRange
     - Validate pod against `min`, `max`, and `maxLimitRequestRatio`
   - Example:
     ```go
     limitRanges := getLimitRangesForNamespace(pod.Namespace)
     pod = applyLimitRangeDefaults(pod, limitRanges)
     if err := validateAgainstLimitRange(pod, limitRanges); err != nil {
         // Skip pod or log error
     }
     ```

2. **Go Plugin - Post-Processing** (Optional):
   - After solver returns plan, double-check all pods satisfy LimitRanges
   - Filter out any placements that would violate constraints

3. **Python Solver**:
   - No changes required (LimitRanges are enforced in Go pre-processing)
   - Alternatively, add validation constraints in solver for completeness

4. **Testing**:
   - Test pods without requests/limits get defaults applied
   - Test pods violating min are rejected
   - Test pods violating max are rejected
   - Test limit/request ratio constraints
   - Test per-pod and per-container LimitRanges

---

## Required Changes

### Go Plugin Changes

#### File: `pkg/mypriorityoptimizer/solver_types.go`

**Changes**:
1. Add QoS fields to `SolverPod`:
   - `LimCPUm int64`
   - `LimMemBytes int64`
   - `QoSClass string`
2. Add affinity structures:
   - `SolverAffinity`, `SolverNodeAffinity`, `SolverPodAffinity`, etc.
3. Add quota structures:
   - `NamespaceQuota`, `ResourceLimits`, `ScopeSelector`
4. Add LimitRange structures:
   - `NamespaceLimitRange`, `LimitRangeItem`
5. Add to `SolverInput`:
   - `Quotas []NamespaceQuota`
   - `LimitRanges []NamespaceLimitRange` (optional)
6. Add labels to `SolverPod`:
   - `Labels map[string]string`

#### File: `pkg/mypriorityoptimizer/solver_helpers.go`

**Changes**:
1. Update `prepareNodesPodsForSolver()` to:
   - Extract QoS class using `v1qos.GetPodQOS(pod)`
   - Extract limits from pod containers
   - Extract affinity from `pod.Spec.Affinity`
   - Extract pod labels
   - Convert affinity to solver structures

2. Add new helper functions:
   - `extractQoSClass(pod *v1.Pod) string`
   - `extractLimits(pod *v1.Pod) (cpuM, memBytes int64)`
   - `convertAffinity(affinity *v1.Affinity) *SolverAffinity`
   - `getQuotasForSolver(ctx, client) []NamespaceQuota`
   - `getLimitRangesForSolver(ctx, client) []NamespaceLimitRange`
   - `applyLimitRangeDefaults(pod *v1.Pod, limitRanges []LimitRangeItem) *v1.Pod`
   - `validateAgainstLimitRange(pod *v1.Pod, limitRanges []LimitRangeItem) error`

3. Update `IgnoreAffinity` to default to `false` once affinity is implemented

#### File: `pkg/mypriorityoptimizer/objects_helpers.go`

**Changes**:
1. Add quota and LimitRange client calls:
   - List ResourceQuota objects
   - List LimitRange objects
   - Calculate current namespace resource usage

2. Add caching for quota/LimitRange objects to avoid repeated API calls

#### File: `pkg/mypriorityoptimizer/plan_computation.go`

**Changes**:
1. Before calling solver:
   - Query quotas and LimitRanges
   - Apply LimitRange defaults to pods
   - Validate pods against LimitRanges
   - Pass quotas to solver input

2. After solver returns:
   - Validate plan doesn't violate quotas
   - Validate plan respects LimitRanges

### Python Solver Changes

#### File: `scripts/python_solver/main.py`

**Changes**:

1. **Input parsing** (lines 20-75):
   ```python
   # Add new input fields
   quotas = instance.get("quotas") or []
   limit_ranges = instance.get("limit_ranges") or []
   ```

2. **Field accessors** (lines 152-165):
   ```python
   def p_lim_cpu_m(i):     return int(pods[i].get("lim_cpu_m", pods[i].get("req_cpu_m", 0)))
   def p_lim_mem_bytes(i): return int(pods[i].get("lim_mem_bytes", pods[i].get("req_mem_bytes", 0)))
   def p_qos_class(i):     return pods[i].get("qos_class", "Guaranteed")
   def p_labels(i):        return pods[i].get("labels") or {}
   def p_affinity(i):      return pods[i].get("affinity") or {}
   def n_labels(j):        return nodes[j].get("labels") or {}
   ```

3. **QoS priority mapping** (after line 165):
   ```python
   # QoS precedence for eviction (lower = evict first)
   QOS_PRIORITY = {
       "BestEffort": 1,
       "Burstable": 2,
       "Guaranteed": 3,
   }
   ```

4. **Affinity helpers** (new section around line 200):
   ```python
   def node_matches_selector(node_idx, selector):
       """Check if node matches the selector expressions"""
       # Implementation as shown in Affinity Support section
   
   def pods_matching_selector(selector, namespace_filter=None):
       """Return indices of pods matching the label selector"""
       # Implementation as shown in Affinity Support section
   
   def nodes_in_topology(node_idx, topology_key):
       """Return indices of nodes with same value for topology_key"""
       # Implementation as shown in Affinity Support section
   
   def filter_by_quota_scope(pod_indices, quota):
       """Filter pods by quota scope"""
       # Implementation for scope filtering
   ```

5. **Affinity constraints** (around line 250, before "Common Constraints"):
   ```python
   # --- Affinity constraints ---
   # Apply node affinity, pod affinity, and pod anti-affinity
   # Implementation as shown in Affinity Support section
   
   # Note: This may modify eligible_nodes[i] for each pod i
   if not ignore_affinity:
       # Apply required node affinity
       for i in range(num_pods):
           # ... (see Affinity Support section)
       
       # Apply required pod affinity
       for i in range(num_pods):
           # ... (see Affinity Support section)
       
       # Apply required pod anti-affinity
       for i in range(num_pods):
           # ... (see Affinity Support section)
   ```

6. **Quota constraints** (around line 280, after "Node constraints"):
   ```python
   # --- Quota constraints (namespace-level) ---
   pods_by_namespace = {}
   for i in range(num_pods):
       ns = p_namespace(i)
       if ns not in pods_by_namespace:
           pods_by_namespace[ns] = []
       pods_by_namespace[ns].append(i)
   
   for quota in quotas:
       # ... (see Resource Quotas Support section)
   ```

7. **QoS-aware disruption** (around line 490, modify disruption objective):
   ```python
   # Modified disruption term to account for QoS
   running_ge = [i for i in running_idxs if p_priority(i) >= p]
   if running_ge:
       # Weight evictions by QoS class (prefer evicting lower QoS)
       qos_weight = {i: QOS_PRIORITY[p_qos_class(i)] for i in running_ge}
       disr_expr = sum(
           qos_weight[i] * (placed[i] - 2 * orig_node(i)) 
           for i in running_ge
       )
       # ... rest of disruption stage
   ```

8. **Update comments and docstrings**:
   - Update the module docstring to document new input fields
   - Update the `solve()` docstring to mention QoS, affinity, quotas

#### Testing the Python Solver

Create test cases in `scripts/python_solver/test_main.py` (new file):
```python
import json
from main import CPSATSolver

def test_qos_eviction_priority():
    """Test that BestEffort pods are evicted before Guaranteed"""
    instance = {
        "nodes": [
            {"name": "node1", "cap_cpu_m": 1000, "cap_mem_bytes": 1000000000}
        ],
        "pods": [
            {
                "uid": "pod1",
                "namespace": "default",
                "name": "guaranteed-pod",
                "req_cpu_m": 800,
                "req_mem_bytes": 800000000,
                "lim_cpu_m": 800,
                "lim_mem_bytes": 800000000,
                "qos_class": "Guaranteed",
                "priority": 0,
                "node": "node1"
            },
            {
                "uid": "pod2",
                "namespace": "default",
                "name": "besteffort-pod",
                "req_cpu_m": 0,
                "req_mem_bytes": 0,
                "qos_class": "BestEffort",
                "priority": 0,
                "node": "node1"
            },
            {
                "uid": "pod3",
                "namespace": "default",
                "name": "new-guaranteed-pod",
                "req_cpu_m": 500,
                "req_mem_bytes": 500000000,
                "lim_cpu_m": 500,
                "lim_mem_bytes": 500000000,
                "qos_class": "Guaranteed",
                "priority": 1,
                "node": ""
            }
        ],
        "timeout_ms": 5000
    }
    
    solver = CPSATSolver()
    result = solver.solve(instance)
    
    # Expect: BestEffort pod evicted, Guaranteed pods stay/placed
    assert len(result["evictions"]) == 1
    assert result["evictions"][0]["pod"]["name"] == "besteffort-pod"
    assert len(result["placements"]) >= 1

def test_affinity_node():
    """Test required node affinity"""
    # ... implementation

def test_affinity_pod():
    """Test required pod affinity"""
    # ... implementation

def test_anti_affinity():
    """Test required pod anti-affinity"""
    # ... implementation

def test_quota_constraint():
    """Test namespace quota is not exceeded"""
    # ... implementation

if __name__ == "__main__":
    test_qos_eviction_priority()
    test_affinity_node()
    test_affinity_pod()
    test_anti_affinity()
    test_quota_constraint()
    print("All tests passed!")
```

---

## Testing Strategy

### Unit Tests

1. **Go Plugin Tests** (`pkg/mypriorityoptimizer/`)
   - Test QoS class extraction
   - Test limit extraction
   - Test affinity conversion
   - Test quota querying
   - Test LimitRange application
   - Test validation logic

2. **Python Solver Tests** (`scripts/python_solver/test_main.py`)
   - Test QoS eviction priority
   - Test node affinity constraints
   - Test pod affinity constraints
   - Test pod anti-affinity constraints
   - Test quota constraints
   - Test combined scenarios

### Integration Tests

1. **KWOK Cluster Tests** (`scripts/kwok_integration_tests/`)
   - Create test scenarios with:
     - Mixed QoS pods
     - Affinity rules
     - Namespace quotas
     - LimitRanges
   - Verify solver respects all constraints
   - Verify pods are scheduled correctly

2. **Test Scenarios**:
   - **QoS Test**: Deploy Guaranteed, Burstable, BestEffort pods; verify eviction order
   - **Affinity Test**: Deploy pods with node affinity; verify placement
   - **Pod Affinity Test**: Deploy pods with pod affinity; verify co-location
   - **Anti-Affinity Test**: Deploy pods with anti-affinity; verify spreading
   - **Quota Test**: Deploy pods exceeding quota; verify rejection
   - **LimitRange Test**: Deploy pods without limits; verify defaults applied
   - **Combined Test**: All features together

### Manual Testing

1. Create test cluster with KWOK
2. Deploy sample workloads (see `manifests/mypriorityoptimizer/test-*.yaml`)
3. Monitor scheduler logs
4. Verify solver stats
5. Check pod placements match expectations

---

## Impact on Test Results Without These Features

This section describes how the **absence** of QoS, Affinity, Quota, and LimitRange support affects the plugin's evaluation and what test scenarios would be **invalid or misleading** without proper implementation.

### Testing Gaps Without QoS Support

#### What You're Missing

Without QoS class support, your plugin evaluation will have these critical gaps:

1. **Incorrect Eviction Decisions**
   - **Problem**: Plugin treats all pods equally during preemption, ignoring QoS class
   - **Impact**: May evict a Guaranteed pod before a BestEffort pod, violating Kubernetes semantics
   - **Test Limitation**: Cannot validate that eviction order respects QoS hierarchy
   - **Real-world Consequence**: Production workloads (Guaranteed) could be evicted in favor of batch jobs (BestEffort)

2. **Misleading Resource Accounting**
   - **Problem**: BestEffort pods have zero requests but consume real resources
   - **Impact**: Node capacity calculations are incorrect for BestEffort pods
   - **Test Limitation**: Cannot measure true cluster utilization with mixed QoS workloads
   - **Real-world Consequence**: Over-provisioning or under-utilization in clusters with BestEffort workloads

3. **Incomplete Optimization Metrics**
   - **Problem**: Cannot differentiate between evicting a critical service vs a batch job
   - **Impact**: Optimization may produce "better" solutions by evicting important pods
   - **Test Limitation**: Solver score doesn't reflect true business value of placements
   - **Real-world Consequence**: SLA violations when Guaranteed pods are preempted unnecessarily

#### Test Results That Would Be Invalid

- **Eviction order tests**: If you test with mixed QoS pods, results showing "fewer evictions" could be misleading if the plugin evicts Guaranteed pods while keeping BestEffort pods
- **Resource efficiency tests**: Utilization metrics will be inaccurate for workloads mixing QoS classes
- **Comparison with default scheduler**: Default scheduler respects QoS; your plugin won't, making A/B testing unfair

#### Example Test Scenario That Fails

```
Scenario: Node at capacity with 1 Guaranteed + 1 BestEffort pod
         New high-priority Guaranteed pod needs to be placed

Current Behavior (WITHOUT QoS support):
- Plugin may evict the existing Guaranteed pod (if it has lower priority)
- Keeps the BestEffort pod running
- Result: Violates Kubernetes QoS semantics

Expected Behavior (WITH QoS support):
- Plugin evicts the BestEffort pod first
- Keeps both Guaranteed pods
- Result: Respects QoS hierarchy
```

**Metric Impact**: Without QoS support, your "pods placed" metric could be high, but you're placing them by violating QoS guarantees.

---

### Testing Gaps Without Affinity Support

#### What You're Missing

1. **Compliance Violations**
   - **Problem**: Plugin ignores required affinity rules
   - **Impact**: Pods may be scheduled on nodes they should never run on
   - **Test Limitation**: Cannot validate compliance with security/regulatory requirements
   - **Real-world Consequence**: PCI-DSS workloads scheduled on non-compliant nodes; data residency violations

2. **High Availability Failures**
   - **Problem**: Pod anti-affinity is not enforced
   - **Impact**: All replicas of a service may be co-located on the same node/zone
   - **Test Limitation**: Cannot test fault tolerance of your scheduling decisions
   - **Real-world Consequence**: Single point of failure; entire service down when one node fails

3. **Co-location Optimization Ignored**
   - **Problem**: Pod affinity for data locality is not considered
   - **Impact**: Database pods not co-located with their application pods
   - **Test Limitation**: Cannot measure latency improvements from affinity
   - **Real-world Consequence**: Increased cross-node network traffic; higher latencies

4. **Invalid Solutions**
   - **Problem**: Solver may produce "optimal" solutions that violate required affinity
   - **Impact**: Pods fail to start due to admission webhook rejecting placement
   - **Test Limitation**: Plugin appears to work but pods never actually run
   - **Real-world Consequence**: Plan execution fails; cluster in inconsistent state

#### Test Results That Would Be Invalid

- **Placement optimization tests**: High placement counts are meaningless if pods are placed on non-compliant nodes and get rejected
- **HA tests**: Cannot validate that your plugin maintains fault tolerance
- **Performance tests**: Cannot measure impact of data locality on application performance

#### Example Test Scenario That Fails

```
Scenario: Web app with required anti-affinity (HA)
         3 replicas must be spread across zones

Current Behavior (WITHOUT affinity support):
- Plugin places all 3 replicas on same zone (e.g., all in us-west-1a)
- Optimization metric: "3/3 pods placed" ✓
- Reality: Single zone failure takes down entire service ✗

Expected Behavior (WITH affinity support):
- Plugin places 1 replica per zone (us-west-1a, us-west-1b, us-west-1c)
- Optimization metric: "3/3 pods placed" ✓
- Reality: Service survives zone failures ✓
```

**Metric Impact**: Your "pods placed" and "minimized disruption" metrics may look good, but you've created a single point of failure.

---

### Testing Gaps Without Quota Support

#### What You're Missing

1. **Multi-tenancy Violations**
   - **Problem**: Plugin doesn't respect namespace resource quotas
   - **Impact**: One namespace can consume all cluster resources
   - **Test Limitation**: Cannot validate fair resource allocation across namespaces
   - **Real-world Consequence**: Resource starvation for other tenants; quota violations cause pod rejections

2. **Plan Execution Failures**
   - **Problem**: Solver proposes placements that exceed namespace quotas
   - **Impact**: Kubernetes API rejects pod creation/updates
   - **Test Limitation**: Plugin reports "success" but pods never actually run
   - **Real-world Consequence**: Optimization plan is partially or completely unexecutable

3. **Cost Control Failures**
   - **Problem**: Cannot enforce resource budgets per team/project
   - **Impact**: Cloud costs exceed budget due to unconstrained resource usage
   - **Test Limitation**: Cannot test cost optimization scenarios
   - **Real-world Consequence**: Budget overruns; unexpected cloud bills

#### Test Results That Would Be Invalid

- **Resource utilization tests**: May show "100% utilization" but violate quota limits
- **Multi-tenant tests**: Cannot validate fair sharing between namespaces
- **Plan success rate**: Solver claims success but placements get rejected by quota admission

#### Example Test Scenario That Fails

```
Scenario: Namespace "team-a" has quota: 10 CPU cores
         Currently using: 8 cores (running pods)
         New pods need: 5 cores (would total 13 cores)

Current Behavior (WITHOUT quota support):
- Plugin places all new pods in team-a namespace
- Solver reports: "5/5 pods placed" ✓
- Reality: API rejects pod creation (quota exceeded) ✗
- Actual result: 0/5 new pods running

Expected Behavior (WITH quota support):
- Plugin recognizes quota limit (2 cores remaining)
- Solver places pods that fit within quota or evicts lower-priority pods
- Reality: Placements succeed ✓
```

**Metric Impact**: Your solver might report "optimal" solutions, but if they violate quotas, execution fails completely.

---

### Testing Gaps Without LimitRange Support

#### What You're Missing

1. **Inconsistent Pod Specs**
   - **Problem**: Pods without requests/limits don't get defaults applied
   - **Impact**: Pod behavior differs between plugin and actual cluster
   - **Test Limitation**: Test environment behaves differently than production
   - **Real-world Consequence**: Pods scheduled in testing but rejected in production

2. **Invalid Pod Configurations**
   - **Problem**: Pods violating LimitRange min/max are not detected
   - **Impact**: Solver optimizes for pods that will be rejected by admission
   - **Test Limitation**: Cannot test with realistic pod specifications
   - **Real-world Consequence**: Plan execution fails due to admission rejection

3. **Resource Request Inaccuracy**
   - **Problem**: Solver uses incomplete resource values
   - **Impact**: Bin-packing calculations are incorrect
   - **Test Limitation**: Resource utilization metrics are unreliable
   - **Real-world Consequence**: Over-subscription or under-utilization

#### Test Results That Would Be Invalid

- **Scheduling tests with pods lacking resources**: Results don't reflect real cluster behavior
- **Resource packing tests**: Incorrect if pods would have different requests/limits in production
- **Plan validation tests**: Cannot detect pods that would fail admission

#### Example Test Scenario That Fails

```
Scenario: Namespace has LimitRange with default: 500m CPU
         Pod submitted without CPU request

Current Behavior (WITHOUT LimitRange support):
- Plugin treats pod as having 0 CPU request (BestEffort)
- Solver places pod easily (no resource constraint)
- Test result: "Pod placed successfully" ✓

Expected Behavior (WITH LimitRange support):
- Plugin applies default: 500m CPU request
- Solver must find node with 500m available
- Test result: May require eviction or different node
```

**Metric Impact**: Your placement success rate may be artificially high because you're not accounting for real resource requirements.

---

### Combined Impact on Plugin Evaluation

#### Overall Test Quality Issues

Without proper support for these features, your plugin evaluation suffers from:

1. **False Positives**: Tests pass but wouldn't work in production
2. **Misleading Metrics**: High placement rates that violate constraints
3. **Incomplete Comparison**: Cannot fairly compare with default scheduler
4. **Limited Applicability**: Cannot test with realistic production workloads

#### Key Metrics That Become Unreliable

| Metric | Impact Without Feature Support |
|--------|--------------------------------|
| **Pods Placed** | May include pods that violate constraints or would be rejected (e.g., pods violating required affinity, exceeding quotas) |
| **Evictions** | Count is meaningless if wrong pods are evicted (e.g., Guaranteed before BestEffort) |
| **Disruption** | Metric doesn't account for HA violations (e.g., co-locating all replicas on same node/zone) |
| **Resource Utilization** | Inaccurate with mixed QoS (BestEffort shows 0 requests) or missing LimitRange defaults |
| **Plan Success Rate** | Artificially high if plans violate quotas/affinity and fail during execution |
| **Solver Performance** | Cannot test on realistic constraint complexity (affinity rules, quota checks) |

#### Production Readiness Assessment

Without these features, you **cannot claim** that your plugin:
- ✗ Respects Kubernetes scheduling semantics (QoS, affinity)
- ✗ Works in multi-tenant environments (quotas)
- ✗ Maintains high availability (anti-affinity)
- ✗ Complies with security/regulatory requirements (affinity)
- ✗ Produces executable plans (quota/affinity validation)

#### Recommended Testing Approach

Until these features are implemented:

1. **Clearly Document Limitations** in test reports:
   ```
   ⚠️ Note: Tests use homogeneous Guaranteed pods only.
   ⚠️ Real-world results may differ with mixed QoS classes.
   ```

2. **Constrain Test Scenarios** to avoid unsupported features:
   - Only test with Guaranteed pods (uniform QoS)
   - Avoid pods with affinity/anti-affinity rules
   - Single namespace tests only (no quota conflicts)
   - Always specify explicit resource requests/limits

3. **Separate Evaluation Metrics**:
   - Mark tests as "Simplified Environment" vs "Production-Ready"
   - Report solver performance separately from constraint compliance

4. **Comparison Fairness**:
   - When comparing with default scheduler, disable its affinity/quota enforcement for fair comparison
   - Or acknowledge that comparison is not apples-to-apples

#### Example: How to Present Limited Test Results

```markdown
## Plugin Evaluation Results

### Test Environment
- **Workload Type**: Uniform QoS (all Guaranteed pods)
- **Affinity**: Not tested (not implemented)
- **Quotas**: Not tested (not implemented)
- **Constraints**: CPU/Memory capacity only

### Results (with caveats)
- Pods Placed: 95% (in simplified environment)
- Evictions: 15 pods (note: QoS priority not considered)
- Performance: 2.3s solver time for 100 pods

### Known Limitations
⚠️ These results do NOT validate:
- Correct eviction order for mixed QoS workloads
- High availability (anti-affinity) compliance
- Multi-tenant quota enforcement
- Production constraint complexity

### Next Steps
To make plugin production-ready, implement:
1. QoS class support (est. 1-2 weeks)
2. Quota enforcement (est. 1-2 weeks)
3. Affinity constraints (est. 3-4 weeks)
```

This transparency helps stakeholders understand that strong performance in simplified tests doesn't guarantee production readiness.

---

## Performance Considerations

### Complexity Analysis

1. **QoS Support**:
   - Minimal overhead (constant time per pod)
   - No significant solver complexity increase

2. **Affinity Support**:
   - **Node Affinity**: O(pods × nodes × selectors) for filtering
   - **Pod Affinity/Anti-Affinity**: O(pods² × topology_domains)
   - Can significantly increase solver complexity
   - May need timeout adjustments

3. **Quota Support**:
   - O(namespaces × pods) for constraints
   - Minimal overhead (linear in pods)

4. **LimitRange Support**:
   - Pre-processing only (not in solver)
   - Minimal overhead

### Optimization Strategies

1. **Affinity Pre-Processing**:
   - Pre-compute eligible nodes per pod before solver
   - Cache label matching results
   - Use early filtering to reduce solver search space

2. **Solver Timeout**:
   - May need to increase `SOLVER_PYTHON_TIMEOUT` when affinity is enabled
   - Recommended timeouts based on cluster size:
     - Small clusters (< 50 pods): 5-10s
     - Medium clusters (50-200 pods): 10-30s
     - Large clusters (> 200 pods): 30-60s
   - Add 50-100% more time when complex affinity rules are present
   - Consider adaptive timeout: `base_timeout + (num_pods × 0.05s) + (num_affinity_rules × 1s)`

3. **Incremental Updates**:
   - Cache affinity rules and quota information
   - Avoid re-querying on every solver run

4. **Relaxation**:
   - For preferred affinity, use weighted objectives instead of hard constraints
   - Allow solver to trade off affinity satisfaction for other goals

5. **Parallelization**:
   - CP-SAT solver supports parallel search using multiple threads
   - The `num_search_workers` parameter controls parallelization:
     - `0` = use all available CPU cores (recommended for production)
     - `1` = single-threaded (useful for debugging)
     - `n` = use exactly n worker threads
   - Current implementation uses `num_search_workers = 0` for maximum performance

---

## References

### Kubernetes Documentation

- [QoS Classes](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod/)
- [Assign Pods to Nodes](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/)
- [Pod Affinity and Anti-Affinity](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity)
- [Resource Quotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
- [Limit Ranges](https://kubernetes.io/docs/concepts/policy/limit-range/)
- [Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)

### External Articles

- [A Hands-On Guide to Kubernetes QoS Classes](https://medium.com/@muppedaanvesh/a-hands-on-guide-to-kubernetes-qos-classes-571b5f8f7e58)
- [Debug a Preempted Pod on Kubernetes](https://medium.com/codex/what-you-need-to-know-to-debug-a-preempted-pod-on-kubernetes-1c956eec3f35)

### Code References

- `k8s.io/kubernetes/pkg/apis/core/v1/helper/qos` - QoS class helpers
- `k8s.io/api/core/v1` - Pod, Node, Affinity types
- `k8s.io/api/core/v1` - ResourceQuota, LimitRange types
- Google OR-Tools CP-SAT: https://developers.google.com/optimization/cp/cp_solver

---

## Conclusion

Supporting QoS classes, affinities, quotas, and LimitRanges will make the MyPriorityOptimizer plugin production-ready and fully compatible with Kubernetes resource management policies.

**Implementation Priority**:
1. **QoS Classes** - Relatively simple, high impact on eviction correctness
2. **Quotas** - Important for multi-tenant clusters, moderate complexity
3. **LimitRanges** - Pre-processing only, low complexity
4. **Affinity** - Most complex, highest solver impact, but critical for HA and compliance

**Estimated Effort**:
- QoS Support: 1-2 weeks (Go + Python + tests)
- Quota Support: 1-2 weeks (Go + Python + tests)
- LimitRange Support: 3-5 days (Go + tests)
- Affinity Support: 3-4 weeks (Go + Python + tests + optimization)
- **Total**: 6-9 weeks for complete implementation

This document should serve as a roadmap for implementing these features incrementally, with testing and validation at each step.
