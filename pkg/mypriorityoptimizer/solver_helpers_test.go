// solver_helpers_test.go
package mypriorityoptimizer

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

// -------------------------
// isAnySolverEnabled
// -------------------------

func TestIsAnySolverEnabled(t *testing.T) {
	pl := &SharedState{}

	withVar(t, &SolverPythonEnabled, false)
	if pl.isAnySolverEnabled() {
		t.Fatalf("want false when all disabled")
	}

	withVar(t, &SolverPythonEnabled, true)
	if !pl.isAnySolverEnabled() {
		t.Fatalf("want true when python enabled")
	}
}

// -------------------------
// buildSolverInput
// -------------------------

func TestBuildSolverInput(t *testing.T) {
	pl := &SharedState{}

	t.Run("no_usable_nodes", func(t *testing.T) {
		in, err := pl.buildSolverInput(nil, nil, nil)
		if !errors.Is(err, ErrNoUsableNodes) {
			t.Fatalf("err=%v, want ErrNoUsableNodes", err)
		}
		if len(in.Nodes) != 0 || len(in.Pods) != 0 {
			t.Fatalf("want empty input on error, got %+v", in)
		}
	})

	t.Run("filters_nodes_dedups_pods_sets_preemptor_and_protected", func(t *testing.T) {
		n1 := node("n1")
		n2 := node("n2", unschedulable()) // ignored

		pPending := pod("ns", "p-pending", withUID("u-pending"), withReqs("100m", "64Mi"))
		pRunUsable := pod("ns", "p-run", withUID("u-run"), onNode("n1"), withReqs("200m", "128Mi"))
		pRunBad := pod("ns", "p-run-bad", withUID("u-bad"), onNode("n2"), withReqs("300m", "256Mi"))
		pSys := pod(SystemNamespace, "p-sys", withUID("u-sys"), withReqs("50m", "32Mi"))

		pre := pod("ns", "p-pre", withUID("u-pre"), withReqs("100m", "64Mi"))

		pods := []*v1.Pod{
			pPending, pRunUsable, pRunBad, pre, pSys,
			pPending, // dup uid
			nil,
		}

		in, err := pl.buildSolverInput([]*v1.Node{n1, n2}, pods, pre)
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}

		if len(in.Nodes) != 1 || in.Nodes[0].Name != "n1" {
			t.Fatalf("Nodes=%+v, want [n1]", in.Nodes)
		}
		if in.Preemptor == nil || string(in.Preemptor.UID) != "u-pre" {
			t.Fatalf("Preemptor=%#v, want uid u-pre", in.Preemptor)
		}

		// expect: pending + running-on-n1 + system-pending (protected) = 3
		if len(in.Pods) != 3 {
			t.Fatalf("Pods len=%d, want 3", len(in.Pods))
		}

		got := map[string]SolverPod{}
		for _, sp := range in.Pods {
			got[string(sp.UID)] = sp
		}
		if got["u-pending"].Node != "" {
			t.Fatalf("pending Node=%q, want empty", got["u-pending"].Node)
		}
		if got["u-run"].Node != "n1" {
			t.Fatalf("running Node=%q, want n1", got["u-run"].Node)
		}
		if !got["u-sys"].Protected {
			t.Fatalf("system pod must be Protected=true")
		}
	})
}

// -------------------------
// buildBaselineScore
// -------------------------

func TestBuildBaselineScore(t *testing.T) {
	p1 := pod("ns", "p1", withUID("p1"), onNode("n1"), withPrio(1), withPhase(v1.PodRunning))
	p2 := pod("ns", "p2", withUID("p2"), onNode("n2"), withPrio(2), withPhase(v1.PodRunning))
	p3 := pod("ns", "p3", withUID("p3"), withPrio(2), withPhase(v1.PodPending)) // not assigned

	score := buildBaselineScore([]*v1.Pod{p1, p2, p3})

	if score.Evicted != 0 || score.Moved != 0 {
		t.Fatalf("Evicted/Moved=%d/%d, want 0/0", score.Evicted, score.Moved)
	}
	if score.PlacedByPriority["1"] != 1 || score.PlacedByPriority["2"] != 1 || len(score.PlacedByPriority) != 2 {
		t.Fatalf("PlacedByPriority=%v, want {1:1,2:1}", score.PlacedByPriority)
	}
}

// -------------------------
// solverConfigArgs
// -------------------------

func kvHas(args []any, key string) bool {
	for i := 0; i+1 < len(args); i += 2 {
		if k, ok := args[i].(string); ok && k == key {
			return true
		}
	}
	return false
}

func TestSolverConfigArgs(t *testing.T) {
	withVar(t, &SolverPythonEnabled, false)
	withVar(t, &SolverSaveAllAttempts, false)

	args := solverConfigArgs()
	if kvHas(args, "pythonSolver") {
		t.Fatalf("unexpected pythonSolver when disabled: %v", args)
	}
	if !kvHas(args, "saveFailedAttempts") {
		t.Fatalf("expected saveFailedAttempts always present: %v", args)
	}

	withVar(t, &SolverPythonEnabled, true)
	args = solverConfigArgs()
	if !kvHas(args, "pythonSolver") {
		t.Fatalf("missing pythonSolver when enabled: %v", args)
	}
}

// -------------------------
// isSolutionBetter / isSolutionUsable
// -------------------------

func TestIsSolutionBetter(t *testing.T) {
	base := SolverScore{PlacedByPriority: map[string]int{"1": 1, "0": 1}, Evicted: 2, Moved: 3}

	cases := []struct {
		name string
		new  SolverScore
		want int
	}{
		{"better_placed", SolverScore{PlacedByPriority: map[string]int{"1": 2, "0": 0}, Evicted: 2, Moved: 3}, 1},
		{"fewer_evictions", SolverScore{PlacedByPriority: map[string]int{"1": 1, "0": 1}, Evicted: 1, Moved: 3}, 1},
		{"more_moves_worse", SolverScore{PlacedByPriority: map[string]int{"1": 1, "0": 1}, Evicted: 2, Moved: 4}, -1},
		{"equal", SolverScore{PlacedByPriority: map[string]int{"1": 1, "0": 1}, Evicted: 2, Moved: 3}, 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isSolutionBetter(&base, &tc.new); got != tc.want {
				t.Fatalf("got=%d, want=%d", got, tc.want)
			}
		})
	}
}

func TestIsSolutionUsable(t *testing.T) {
	cases := map[string]bool{
		"":           false,
		"OPTIMAL":    true,
		"FEASIBLE":   true,
		"INFEASIBLE": false,
		"something":  false,
	}
	for st, want := range cases {
		if got := isSolutionUsable(st); got != want {
			t.Fatalf("isSolutionUsable(%q)=%v, want %v", st, got, want)
		}
	}
}

// -------------------------
// isSolutionApplicable
// -------------------------

func TestIsSolutionApplicable(t *testing.T) {
	pl := &SharedState{}

	n1 := node("n1", withAllocatable("1000m", "1Gi"))
	n2Bad := node("n2", unschedulable())

	pOnN1 := pod("ns", "p1", withUID("u1"), onNode("n1"), withReqs("100m", "128Mi"))
	pOnN2 := pod("ns", "p2", withUID("u2"), onNode("n2"), withReqs("100m", "128Mi"))

	t.Run("nil_plan", func(t *testing.T) {
		ok, reason := pl.isSolutionApplicable(nil, nil, nil)
		if ok || reason != "nil plan" {
			t.Fatalf("ok=%v reason=%q, want false/'nil plan'", ok, reason)
		}
	})

	t.Run("pod_vanished", func(t *testing.T) {
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "missing", Namespace: "ns", Name: "missing", OldNode: "", Node: "n1"}},
		}
		ok, reason := pl.isSolutionApplicable(out, []*v1.Node{n1}, []*v1.Pod{pOnN1})
		if ok || !strings.Contains(reason, "pod vanished") {
			t.Fatalf("ok=%v reason=%q, want pod vanished", ok, reason)
		}
	})

	t.Run("dest_node_now_unusable", func(t *testing.T) {
		// place pending to n2, but n2 is unusable
		pPending := pod("ns", "p3", withUID("u3"), withReqs("100m", "128Mi"))
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u3", Namespace: "ns", Name: "p3", OldNode: "", Node: "n2"}},
		}
		ok, reason := pl.isSolutionApplicable(out, []*v1.Node{n1, n2Bad}, []*v1.Pod{pPending})
		if ok || !strings.Contains(reason, "dest node now unusable") {
			t.Fatalf("ok=%v reason=%q, want dest node now unusable", ok, reason)
		}
	})

	t.Run("move_precondition_changed", func(t *testing.T) {
		// plan expects u2 on n1, but actually on n2
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u2", Namespace: "ns", Name: "p2", OldNode: "n1", Node: "n1"}},
		}
		ok, reason := pl.isSolutionApplicable(out, []*v1.Node{n1, node("n2")}, []*v1.Pod{pOnN2})
		if ok || !strings.Contains(reason, "move precondition changed") {
			t.Fatalf("ok=%v reason=%q, want move precondition changed", ok, reason)
		}
	})

	t.Run("success_move_branch", func(t *testing.T) {
		// u1 moves from n1 -> n1 (no-op move) but still exercises OldNode branch
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u1", Namespace: "ns", Name: "p1", OldNode: "n1", Node: "n1"}},
		}
		ok, reason := pl.isSolutionApplicable(out, []*v1.Node{n1}, []*v1.Pod{pOnN1})
		if !ok {
			t.Fatalf("want ok=true, reason=%q", reason)
		}
	})
	t.Run("evict_node_now_unusable", func(t *testing.T) {
		n2Bad := node("n2", unschedulable())
		p := pod("ns", "p", withUID("u-ev"), onNode("n2"), withReqs("100m", "128Mi"), withPhase(v1.PodRunning))

		out := &SolverOutput{Evictions: []SolverPod{{UID: "u-ev"}}}
		ok, reason := pl.isSolutionApplicable(out, []*v1.Node{n1, n2Bad}, []*v1.Pod{p})
		if ok || !strings.Contains(reason, "evict node now unusable") {
			t.Fatalf("ok=%v reason=%q, want evict node now unusable", ok, reason)
		}
	})

	t.Run("pending_precondition_changed", func(t *testing.T) {
		// OldNode == "" means plan expects pending, but pod is already bound.
		p := pod("ns", "p", withUID("u-bind"), onNode("n1"), withReqs("100m", "128Mi"), withPhase(v1.PodRunning))
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u-bind", Namespace: "ns", Name: "p", OldNode: "", Node: "n1"}},
		}
		ok, reason := pl.isSolutionApplicable(out, []*v1.Node{n1}, []*v1.Pod{p})
		if ok || !strings.Contains(reason, "pending precondition changed") {
			t.Fatalf("ok=%v reason=%q, want pending precondition changed", ok, reason)
		}
	})

	t.Run("capacity_exceeded", func(t *testing.T) {
		// Tight node, existing usage + planned placement exceeds capacity.
		nSmall := node("n1", withAllocatable("100m", "64Mi"))
		pRun := pod("ns", "run", withUID("u-run"), onNode("n1"), withReqs("90m", "60Mi"), withPhase(v1.PodRunning))
		pPend := pod("ns", "pend", withUID("u-pend"), withReqs("20m", "16Mi"), withPhase(v1.PodPending))

		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u-pend", Namespace: "ns", Name: "pend", OldNode: "", Node: "n1"}},
		}
		ok, reason := pl.isSolutionApplicable(out, []*v1.Node{nSmall}, []*v1.Pod{pRun, pPend})
		if ok || !strings.Contains(reason, "capacity exceeded") {
			t.Fatalf("ok=%v reason=%q, want capacity exceeded", ok, reason)
		}
	})

	t.Run("eviction_pending_is_ignored", func(t *testing.T) {
		// Eviction entry for a pod that is already pending should just be skipped.
		pPending := pod("ns", "p", withUID("u-pend"), withReqs("10m", "8Mi"), withPhase(v1.PodPending))
		out := &SolverOutput{
			Evictions: []SolverPod{{UID: "u-pend"}},
		}
		ok, reason := pl.isSolutionApplicable(out, []*v1.Node{n1}, []*v1.Pod{pPending})
		if !ok {
			t.Fatalf("want ok=true, reason=%q", reason)
		}
	})

	t.Run("eviction_frees_capacity_for_placement", func(t *testing.T) {
		// Small capacity node
		nSmall := node("n1", withAllocatable("100m", "64Mi"))

		// Running pod almost fills node
		pRun := pod("ns", "run",
			withUID("u-run"),
			onNode("n1"),
			withReqs("90m", "60Mi"),
			withPhase(v1.PodRunning),
		)

		// Pending pod we want to place onto n1
		pPend := pod("ns", "pend",
			withUID("u-pend"),
			withReqs("20m", "16Mi"),
			withPhase(v1.PodPending),
		)

		out := &SolverOutput{
			// Evict the big running pod (must execute the negative addUse line)
			Evictions: []SolverPod{{UID: "u-run"}},
			// Then place the pending pod
			Placements: []SolverPod{{UID: "u-pend", Namespace: "ns", Name: "pend", OldNode: "", Node: "n1"}},
		}

		ok, reason := pl.isSolutionApplicable(out, []*v1.Node{nSmall}, []*v1.Pod{pRun, pPend})
		if !ok {
			t.Fatalf("want ok=true, reason=%q", reason)
		}
	})
}

// -------------------------
// logLeaderboard
// -------------------------

func TestLogLeaderboard_CoversNilBestAndTiePath(t *testing.T) {
	baseline := SolverScore{PlacedByPriority: map[string]int{"1": 1}, Evicted: 0, Moved: 0}

	// best == nil branch
	logLeaderboard("label", nil, baseline, nil)

	// tie-tagging path: adjacent tied scores
	attempts := []SolverResult{
		{Name: "a", Status: "OPTIMAL", DurationMs: 1, Score: baseline},
		{Name: "b", Status: "FEASIBLE", DurationMs: 2, Score: baseline}, // tie with a
	}
	best := attempts[0]
	logLeaderboard("label", attempts, baseline, &best)
}

func TestLogLeaderboard_AttemptsEmptyButBestNonNil(t *testing.T) {
	baseline := SolverScore{PlacedByPriority: map[string]int{"1": 1}, Evicted: 0, Moved: 0}
	best := SolverResult{Name: "baseline", Status: "BASELINE", DurationMs: 0, Score: baseline}
	logLeaderboard("label", []SolverResult{}, baseline, &best)
}

func TestLogLeaderboard_BetterEqualWorseAndBestNotBaseline(t *testing.T) {
	baseline := SolverScore{PlacedByPriority: map[string]int{"1": 1}, Evicted: 0, Moved: 0}
	better := SolverResult{Name: "better", Status: "OPTIMAL", DurationMs: 1, Score: SolverScore{PlacedByPriority: map[string]int{"1": 2}, Evicted: 0, Moved: 0}}
	equal := SolverResult{Name: "equal", Status: "FEASIBLE", DurationMs: 2, Score: baseline}
	worse := SolverResult{Name: "worse", Status: "FEASIBLE", DurationMs: 3, Score: SolverScore{PlacedByPriority: map[string]int{"1": 1}, Evicted: 0, Moved: 1}}

	best := better
	logLeaderboard("label", []SolverResult{worse, equal, better}, baseline, &best)
}

// -------------------------
// scoreSolution / toSolverPod
// -------------------------

func TestScoreSolution(t *testing.T) {
	in := SolverInput{
		Pods: []SolverPod{
			{UID: "u1", Priority: 1, Node: "n1"},
			{UID: "u2", Priority: 2, Node: ""},
			{UID: "u3", Priority: 1, Node: "n1"},
		},
		Preemptor: &SolverPod{UID: "u-pre", Priority: 5},
	}

	t.Run("nil_output", func(t *testing.T) {
		got := scoreSolution(in, nil)
		if got.Evicted != 0 || got.Moved != 0 || len(got.PlacedByPriority) != 0 {
			t.Fatalf("got=%v, want zero score", got)
		}
	})

	t.Run("basic", func(t *testing.T) {
		out := &SolverOutput{
			Placements: []SolverPod{
				{UID: "u2", Node: "n1"},    // place pending
				{UID: "u3", Node: "n2"},    // move
				{UID: "uX", Node: "n1"},    // unknown ignored
				{UID: "u-pre", Node: "n1"}, // place preemptor
			},
			Evictions: []SolverPod{{UID: "u1"}},
		}
		got := scoreSolution(in, out)
		if got.PlacedByPriority["1"] != 1 || got.PlacedByPriority["2"] != 1 || got.PlacedByPriority["5"] != 1 {
			t.Fatalf("PlacedByPriority=%v, want prio1=1 prio2=1 prio5=1", got.PlacedByPriority)
		}
		if got.Evicted != 1 || got.Moved != 1 {
			t.Fatalf("Evicted/Moved=%d/%d, want 1/1", got.Evicted, got.Moved)
		}
	})
}

func TestScoreSolution_PreemptorAlreadyIncludedAndEmptyPlacementNode(t *testing.T) {
	in := SolverInput{
		Pods: []SolverPod{
			{UID: "u1", Priority: 1, Node: "n1"},
			{UID: "u-pre", Priority: 5, Node: ""}, // preemptor already included
		},
		Preemptor: &SolverPod{UID: "u-pre", Priority: 5},
	}

	out := &SolverOutput{
		Placements: []SolverPod{
			{UID: "u-pre", Node: ""}, // covers plm.Node=="" continue
		},
		Evictions: nil,
	}

	got := scoreSolution(in, out)
	if got.PlacedByPriority["1"] != 1 || len(got.PlacedByPriority) != 1 {
		t.Fatalf("PlacedByPriority=%v, want only prio1=1", got.PlacedByPriority)
	}
	if got.Evicted != 0 || got.Moved != 0 {
		t.Fatalf("Evicted/Moved=%d/%d, want 0/0", got.Evicted, got.Moved)
	}
}

func TestToSolverPod(t *testing.T) {
	p := pod("ns", "mypod", withUID("uid-1"))
	sp := toSolverPod(p, "nodeX")

	if sp.UID != p.UID || sp.Namespace != p.Namespace || sp.Name != p.Name || sp.Node != "nodeX" {
		t.Fatalf("unexpected mapping: %+v", sp)
	}
}

// -------------------------
// exportSolverStatsToConfigMap + appendSolverStatsCM
// -------------------------

func withAppendStatsHook(t *testing.T, hook func(pl *SharedState, ctx context.Context, entry ExportedSolverStats)) {
	t.Helper()
	orig := appendSolverStatsCMHook
	appendSolverStatsCMHook = hook
	t.Cleanup(func() { appendSolverStatsCMHook = orig })
}

func TestExportSolverStatsToConfigMap_UsesHook(t *testing.T) {
	pl := &SharedState{}
	var got ExportedSolverStats

	withAppendStatsHook(t, func(_ *SharedState, _ context.Context, e ExportedSolverStats) { got = e })

	baseline := SolverScore{PlacedByPriority: map[string]int{"1": 1}}
	attempts := []SolverResult{{Name: "python", Status: "OPTIMAL", DurationMs: 42, Score: SolverScore{PlacedByPriority: map[string]int{"1": 2}}}}

	pl.exportSolverStatsToConfigMap(context.Background(), "strategy", baseline, "python", attempts, "boom")

	if got.BestName != "python" || got.Error != "boom" || got.TimestampNs == 0 {
		t.Fatalf("got=%+v", got)
	}
	if !reflect.DeepEqual(got.Baseline, baseline) {
		t.Fatalf("baseline mismatch: got=%v want=%v", got.Baseline, baseline)
	}
	if len(got.Attempts) != 1 || got.Attempts[0].Name != "python" {
		t.Fatalf("attempts=%v", got.Attempts)
	}
}

func TestAppendSolverStatsCM_HookShortCircuit(t *testing.T) {
	pl := &SharedState{}
	called := false
	withAppendStatsHook(t, func(_ *SharedState, _ context.Context, _ ExportedSolverStats) { called = true })

	if err := pl.appendSolverStatsCM(context.Background(), ExportedSolverStats{BestName: "x"}); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if !called {
		t.Fatalf("expected hook to be called")
	}
}

func TestAppendSolverStatsCM_NoClientset(t *testing.T) {
	pl := &SharedState{Handle: &fakeHandle{client: nil, factory: nil}}
	withAppendStatsHook(t, nil) // ensure hook disabled (explicitly)

	err := pl.appendSolverStatsCM(context.Background(), ExportedSolverStats{BestName: "x"})
	if !errors.Is(err, ErrNoClientset) {
		t.Fatalf("err=%v, want ErrNoClientset", err)
	}
}

func TestAppendSolverStatsCM_CreateOnMissing(t *testing.T) {
	ctx := context.Background()
	client := fake.NewSimpleClientset()
	factory := informers.NewSharedInformerFactory(client, 0)
	pl := &SharedState{Handle: &fakeHandle{client: client, factory: factory}}

	// make sure hook is off
	withAppendStatsHook(t, nil)

	stopCh := make(chan struct{})
	t.Cleanup(func() { close(stopCh) })
	_ = factory.Core().V1().ConfigMaps().Informer()
	factory.Start(stopCh)
	factory.WaitForCacheSync(stopCh)

	e1 := ExportedSolverStats{BestName: "first"}
	if err := pl.appendSolverStatsCM(ctx, e1); err != nil {
		t.Fatalf("append err=%v", err)
	}

	cm, err := client.CoreV1().ConfigMaps(SystemNamespace).
		Get(ctx, SolverStatsConfigMapName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get cm err=%v", err)
	}

	key := SolverStatsConfigMapLabelKey + ".json"
	raw := cm.Data[key]
	if raw == "" {
		t.Fatalf("expected json payload at %q", key)
	}

	var arr []ExportedSolverStats
	if err := json.Unmarshal([]byte(raw), &arr); err != nil {
		t.Fatalf("unmarshal err=%v (raw=%q)", err, raw)
	}
	if len(arr) != 1 || arr[0].BestName != "first" {
		t.Fatalf("arr=%+v, want [{BestName:first}]", arr)
	}
}

func TestAppendSolverStatsCM_AppendWhenFound(t *testing.T) {
	ctx := context.Background()
	client := fake.NewSimpleClientset()
	factory := informers.NewSharedInformerFactory(client, 0)
	pl := &SharedState{Handle: &fakeHandle{client: client, factory: factory}}

	// make sure hook is off
	withAppendStatsHook(t, nil)

	// Pre-create CM BEFORE informer starts, so lister sees it on initial LIST.
	doc := ConfigMapDoc{
		Namespace: SystemNamespace,
		Name:      SolverStatsConfigMapName,
		LabelKey:  SolverStatsConfigMapLabelKey,
		DataKey:   SolverStatsConfigMapLabelKey + ".json",
	}
	cms := client.CoreV1().ConfigMaps(SystemNamespace)

	if err := doc.ensureJson(ctx, cms, []ExportedSolverStats{{BestName: "first"}}); err != nil {
		t.Fatalf("ensureJson err=%v", err)
	}

	stopCh := make(chan struct{})
	t.Cleanup(func() { close(stopCh) })
	_ = factory.Core().V1().ConfigMaps().Informer()
	factory.Start(stopCh)
	factory.WaitForCacheSync(stopCh)

	// Now the lister should reliably report "found", so we hit mutateJson append path.
	if err := pl.appendSolverStatsCM(ctx, ExportedSolverStats{BestName: "second"}); err != nil {
		t.Fatalf("append err=%v", err)
	}

	cm, err := client.CoreV1().ConfigMaps(SystemNamespace).
		Get(ctx, SolverStatsConfigMapName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get cm err=%v", err)
	}

	key := SolverStatsConfigMapLabelKey + ".json"
	raw := cm.Data[key]

	var arr []ExportedSolverStats
	if err := json.Unmarshal([]byte(raw), &arr); err != nil {
		t.Fatalf("unmarshal err=%v (raw=%q)", err, raw)
	}
	if len(arr) != 2 {
		t.Fatalf("want 2 entries, got %d (arr=%+v)", len(arr), arr)
	}
	if arr[0].BestName != "first" || arr[1].BestName != "second" {
		t.Fatalf("arr=%+v, want first then second", arr)
	}
}
