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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes/fake"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
)

func statsCMWithRaw(t *testing.T, raw string) *v1.ConfigMap {
	t.Helper()
	key := SolverStatsConfigMapLabelKey + ".json"
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      SolverStatsConfigMapName,
			Namespace: SystemNamespace,
			Labels:    map[string]string{SolverStatsConfigMapLabelKey: "true"},
		},
		Data: map[string]string{key: raw},
	}
}

func statsCMWithEntries(t *testing.T, entries []ExportedSolverStats) *v1.ConfigMap {
	t.Helper()
	b, err := json.MarshalIndent(entries, "", "  ")
	mustNoErr(t, err, "marshal entries")
	return statsCMWithRaw(t, string(b))
}

func readStatsArr(t *testing.T, ctx context.Context, pl *SharedState) []ExportedSolverStats {
	t.Helper()
	key := SolverStatsConfigMapLabelKey + ".json"
	cm, err := pl.Handle.ClientSet().CoreV1().ConfigMaps(SystemNamespace).
		Get(ctx, SolverStatsConfigMapName, metav1.GetOptions{})
	mustNoErr(t, err, "get stats CM")

	raw := cm.Data[key]
	must(t, raw != "", "expected non-empty json payload at %q", key)

	var arr []ExportedSolverStats
	mustNoErr(t, json.Unmarshal([]byte(raw), &arr), "unmarshal stats json (raw=%q)", raw)
	return arr
}

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

	// Local helper: keeps each subtest to ~1 line of intent.
	mustApplicable := func(t *testing.T, out *SolverOutput, nodes []*v1.Node, pods []*v1.Pod, wantOK bool, wantSubstr string) {
		t.Helper()
		ok, reason := pl.isSolutionApplicable(out, nodes, pods)

		if ok != wantOK {
			t.Fatalf("ok=%v, want %v (reason=%q)", ok, wantOK, reason)
		}
		if wantOK {
			if reason != "" {
				t.Fatalf("reason=%q, want empty on success", reason)
			}
			return
		}
		if wantSubstr != "" && !strings.Contains(reason, wantSubstr) {
			t.Fatalf("reason=%q, want contains %q", reason, wantSubstr)
		}
	}

	// Shared fixtures
	n1 := node("n1", withAllocatable("1000m", "1Gi"))
	n2Bad := node("n2", unschedulable())

	pOnN1 := pod("ns", "p1", withUID("u1"), onNode("n1"), withReqs("100m", "128Mi"))
	pOnN2 := pod("ns", "p2", withUID("u2"), onNode("n2"), withReqs("100m", "128Mi"))

	t.Run("nil_plan", func(t *testing.T) {
		mustApplicable(t, nil, nil, nil, false, "nil plan")
	})

	t.Run("pod_vanished", func(t *testing.T) {
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "missing", Namespace: "ns", Name: "missing", OldNode: "", Node: "n1"}},
		}
		mustApplicable(t, out, []*v1.Node{n1}, []*v1.Pod{pOnN1}, false, "pod vanished")
	})

	t.Run("dest_node_now_unusable", func(t *testing.T) {
		// place pending to n2, but n2 is unusable
		pPending := pod("ns", "p3", withUID("u3"), withReqs("100m", "128Mi"))
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u3", Namespace: "ns", Name: "p3", OldNode: "", Node: "n2"}},
		}
		mustApplicable(t, out, []*v1.Node{n1, n2Bad}, []*v1.Pod{pPending}, false, "dest node now unusable")
	})

	t.Run("move_precondition_changed", func(t *testing.T) {
		// plan expects u2 on n1, but actually on n2
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u2", Namespace: "ns", Name: "p2", OldNode: "n1", Node: "n1"}},
		}
		mustApplicable(t, out, []*v1.Node{n1, node("n2")}, []*v1.Pod{pOnN2}, false, "move precondition changed")
	})

	t.Run("success_move_branch", func(t *testing.T) {
		// u1 moves from n1 -> n1 (no-op move) but still exercises OldNode branch
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u1", Namespace: "ns", Name: "p1", OldNode: "n1", Node: "n1"}},
		}
		mustApplicable(t, out, []*v1.Node{n1}, []*v1.Pod{pOnN1}, true, "")
	})

	t.Run("evict_node_now_unusable", func(t *testing.T) {
		p := pod("ns", "p", withUID("u-ev"), onNode("n2"), withReqs("100m", "128Mi"), withPhase(v1.PodRunning))
		out := &SolverOutput{Evictions: []SolverPod{{UID: "u-ev"}}}
		mustApplicable(t, out, []*v1.Node{n1, n2Bad}, []*v1.Pod{p}, false, "evict node now unusable")
	})

	t.Run("pending_precondition_changed", func(t *testing.T) {
		// OldNode == "" means plan expects pending, but pod is already bound.
		p := pod("ns", "p", withUID("u-bind"), onNode("n1"), withReqs("100m", "128Mi"), withPhase(v1.PodRunning))
		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u-bind", Namespace: "ns", Name: "p", OldNode: "", Node: "n1"}},
		}
		mustApplicable(t, out, []*v1.Node{n1}, []*v1.Pod{p}, false, "pending precondition changed")
	})

	t.Run("capacity_exceeded", func(t *testing.T) {
		// Tight node, existing usage + planned placement exceeds capacity.
		nSmall := node("n1", withAllocatable("100m", "64Mi"))
		pRun := pod("ns", "run", withUID("u-run"), onNode("n1"), withReqs("90m", "60Mi"), withPhase(v1.PodRunning))
		pPend := pod("ns", "pend", withUID("u-pend"), withReqs("20m", "16Mi"), withPhase(v1.PodPending))

		out := &SolverOutput{
			Placements: []SolverPod{{UID: "u-pend", Namespace: "ns", Name: "pend", OldNode: "", Node: "n1"}},
		}
		mustApplicable(t, out, []*v1.Node{nSmall}, []*v1.Pod{pRun, pPend}, false, "capacity exceeded")
	})

	t.Run("eviction_pending_is_ignored", func(t *testing.T) {
		// Eviction entry for a pod that is already pending should just be skipped.
		pPending := pod("ns", "p", withUID("u-pend"), withReqs("10m", "8Mi"), withPhase(v1.PodPending))
		out := &SolverOutput{Evictions: []SolverPod{{UID: "u-pend"}}}
		mustApplicable(t, out, []*v1.Node{n1}, []*v1.Pod{pPending}, true, "")
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

		mustApplicable(t, out, []*v1.Node{nSmall}, []*v1.Pod{pRun, pPend}, true, "")
	})
}

// -------------------------
// logLeaderboard
// -------------------------

func TestLogLeaderboard(t *testing.T) {
	baseline := SolverScore{PlacedByPriority: map[string]int{"1": 1}, Evicted: 0, Moved: 0}

	// best == nil branch
	logLeaderboard("label", nil, baseline, nil)

	// attempts empty but best non-nil branch
	best := SolverResult{Name: "baseline", Status: "BASELINE", DurationMs: 0, Score: baseline}
	logLeaderboard("label", []SolverResult{}, baseline, &best)

	// tie tagging + better/equal/worse + best != baseline placed-map selection
	better := SolverResult{Name: "better", Status: "OPTIMAL", DurationMs: 1, Score: SolverScore{PlacedByPriority: map[string]int{"1": 2}}}
	equal := SolverResult{Name: "equal", Status: "FEASIBLE", DurationMs: 2, Score: baseline}
	worse := SolverResult{Name: "worse", Status: "FEASIBLE", DurationMs: 3, Score: SolverScore{PlacedByPriority: map[string]int{"1": 1}, Moved: 1}}

	// tie: adjacent equal scores
	attempts := []SolverResult{
		{Name: "a", Status: "OPTIMAL", DurationMs: 1, Score: baseline},
		{Name: "b", Status: "FEASIBLE", DurationMs: 2, Score: baseline},
	}
	best2 := attempts[0]
	logLeaderboard("label", attempts, baseline, &best2)

	best3 := better
	logLeaderboard("label", []SolverResult{worse, equal, better}, baseline, &best3)
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

func TestAppendSolverStatsCM_UpsertAndAppend(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		name     string
		initial  *v1.ConfigMap // nil => missing
		entry    ExportedSolverStats
		wantBest []string
	}{
		{
			name:     "create_on_missing",
			initial:  nil,
			entry:    ExportedSolverStats{BestName: "first"},
			wantBest: []string{"first"},
		},
		{
			name:     "append_when_found",
			initial:  statsCMWithEntries(t, []ExportedSolverStats{{BestName: "first"}}),
			entry:    ExportedSolverStats{BestName: "second"},
			wantBest: []string{"first", "second"},
		},
		{
			name:     "append_when_found_empty_payload",
			initial:  statsCMWithRaw(t, ""), // key exists but empty
			entry:    ExportedSolverStats{BestName: "only"},
			wantBest: []string{"only"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			withAppendStatsHook(t, nil) // ensure hook is off

			var pl *SharedState
			var cleanup func()
			if tc.initial == nil {
				pl, cleanup = newSharedStateWithConfigMapInformer(t /* no objects */)
			} else {
				pl, cleanup = newSharedStateWithConfigMapInformer(t, tc.initial)
			}
			defer cleanup()

			mustNoErr(t, pl.appendSolverStatsCM(ctx, tc.entry), "append err")

			arr := readStatsArr(t, ctx, pl)
			got := make([]string, 0, len(arr))
			for _, e := range arr {
				got = append(got, e.BestName)
			}
			mustEq(t, got, tc.wantBest, "bestName sequence mismatch")
		})
	}
}

type errConfigMapNamespaceLister struct{ err error }

func (e errConfigMapNamespaceLister) List(_ labels.Selector) ([]*v1.ConfigMap, error) {
	return nil, e.err
}

func (e errConfigMapNamespaceLister) Get(_ string) (*v1.ConfigMap, error) {
	return nil, e.err
}

func TestAppendSolverStatsCM_ReadJsonError(t *testing.T) {
	ctx := context.Background()
	client := fake.NewSimpleClientset()
	pl := &SharedState{Handle: &fakeHandle{client: client, factory: nil}}

	withAppendStatsHook(t, nil)

	origCmsFor := solverStatsConfigMapsFor
	origNsListerFor := solverStatsConfigMapNsListerFor
	t.Cleanup(func() {
		solverStatsConfigMapsFor = origCmsFor
		solverStatsConfigMapNsListerFor = origNsListerFor
	})

	solverStatsConfigMapsFor = func(_ *SharedState) corev1client.ConfigMapInterface {
		return client.CoreV1().ConfigMaps(SystemNamespace)
	}
	solverStatsConfigMapNsListerFor = func(_ *SharedState) corev1listers.ConfigMapNamespaceLister {
		return errConfigMapNamespaceLister{err: errors.New("boom")}
	}

	err := pl.appendSolverStatsCM(ctx, ExportedSolverStats{BestName: "x"})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("err=%v, want contains 'boom'", err)
	}
}
