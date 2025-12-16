// pod_set_helpers_test.go
// TODO
package mypriorityoptimizer

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// -------------------------
// Test Helpers
// -------------------------

func mustContainsUID(t *testing.T, snap map[types.UID]SolverPod, uid types.UID) SolverPod {
	t.Helper()
	got, ok := snap[uid]
	if !ok {
		t.Fatalf("snapshot missing uid=%q; snap=%v", uid, snap)
	}
	return got
}

func mustNotContainsUID(t *testing.T, snap map[types.UID]SolverPod, uid types.UID) {
	t.Helper()
	if _, ok := snap[uid]; ok {
		t.Fatalf("snapshot unexpectedly contains uid=%q; snap=%v", uid, snap)
	}
}

func mustSize(t *testing.T, ps *PodSet, want int) {
	t.Helper()
	if got := ps.Size(); got != want {
		t.Fatalf("Size()=%d want %d", got, want)
	}
}

func mustMapLen(t *testing.T, m map[types.UID]SolverPod, want int) {
	t.Helper()
	if got := len(m); got != want {
		t.Fatalf("len(snapshot)=%d want %d", got, want)
	}
}

// -------------------------
// newPodSet + doesPodSetExist
// -------------------------

func TestPodSet_NewAndExistence(t *testing.T) {
	ps := newPodSet("blocked")
	if ps == nil {
		t.Fatal("newPodSet returned nil")
	}
	if ps.Name != "blocked" {
		t.Fatalf("Name=%q want %q", ps.Name, "blocked")
	}
	mustSize(t, ps, 0)
	mustMapLen(t, ps.Snapshot(), 0)

	tests := []struct {
		name string
		in   *PodSet
		want bool
	}{
		{"nil", nil, false},
		{"empty", newPodSet("empty"), false},
		{"non-empty", func() *PodSet {
			s := newPodSet("x")
			s.AddPod(pod("ns", "p1", withUID("u1")))
			return s
		}(), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := doesPodSetExist(tt.in); got != tt.want {
				t.Fatalf("doesPodSetExist()=%v want %v", got, tt.want)
			}
		})
	}
}

// -------------------------
// AddPod / RemovePod / Snapshot contracts
// -------------------------

func TestPodSet_BasicOperations(t *testing.T) {
	t.Run("AddPod(nil) is no-op", func(t *testing.T) {
		ps := newPodSet("s")
		ps.AddPod(nil)
		mustSize(t, ps, 0)
	})

	t.Run("Add stores SolverPod fields", func(t *testing.T) {
		ps := newPodSet("s")
		p := pod("ns", "p1", withUID("u1"))
		ps.AddPod(p)

		mustSize(t, ps, 1)
		snap := ps.Snapshot()
		mustMapLen(t, snap, 1)

		got := mustContainsUID(t, snap, p.UID)
		if got.UID != p.UID || got.Namespace != p.Namespace || got.Name != p.Name {
			t.Fatalf("stored key mismatch: got=%#v want uid=%q ns=%q name=%q", got, p.UID, p.Namespace, p.Name)
		}
	})

	t.Run("Snapshot returns a copy", func(t *testing.T) {
		ps := newPodSet("s")
		p := pod("ns", "p1", withUID("u1"))
		ps.AddPod(p)

		snap := ps.Snapshot()
		delete(snap, p.UID) // mutate returned map

		// internal state should be unchanged
		mustSize(t, ps, 1)
		mustContainsUID(t, ps.Snapshot(), p.UID)
	})

	t.Run("RemovePod(missing) is no-op", func(t *testing.T) {
		ps := newPodSet("s")
		ps.RemovePod(types.UID("nope"))
		mustSize(t, ps, 0)
	})

	t.Run("RemovePod removes existing", func(t *testing.T) {
		ps := newPodSet("s")
		p := pod("ns", "p1", withUID("u1"))
		ps.AddPod(p)
		ps.RemovePod(p.UID)

		mustSize(t, ps, 0)
		mustMapLen(t, ps.Snapshot(), 0)
	})

	t.Run("Add overwrites by UID (last write wins)", func(t *testing.T) {
		ps := newPodSet("s")
		p1 := pod("ns1", "a", withUID("same"))
		p2 := pod("ns2", "b", withUID("same"))

		ps.AddPod(p1)
		ps.AddPod(p2)

		mustSize(t, ps, 1)
		got := mustContainsUID(t, ps.Snapshot(), p1.UID)
		if got.Namespace != "ns2" || got.Name != "b" {
			t.Fatalf("overwrite failed: got=%#v want ns2/b", got)
		}
	})
}

// -------------------------
// prunePodSet
// -------------------------

func TestPrunePodSet_NilOrEmpty(t *testing.T) {
	pl := &SharedState{}

	if got := pl.prunePodSet(nil); got != 0 {
		t.Fatalf("removed=%d want 0", got)
	}
	empty := newPodSet("empty")
	if got := pl.prunePodSet(empty); got != 0 {
		t.Fatalf("removed=%d want 0", got)
	}
}

func TestPrunePodSet_Branches(t *testing.T) {
	pl := &SharedState{}
	ps := newPodSet("blocked")

	// 1) NotFound => pruned
	pGone := pod("ns", "gone", withUID("u-gone"))

	// 2) Recreated (same ns/name, different UID) => pruned
	pOld := pod("ns", "recreated", withUID("u-old"))
	pNew := pod("ns", "recreated", withUID("u-new"))

	// 3) Terminating => pruned
	pTerm := pod("ns", "term", withUID("u-term"))
	now := metav1.Now()
	pTerm.DeletionTimestamp = &now

	// 4) Bound => pruned
	pBound := pod("ns", "bound", withUID("u-bound"), onNode("node1"))

	// 5) Lister error => kept
	pErr := pod("ns", "err", withUID("u-err"))

	// 6) Valid pending => kept
	pKeep := pod("ns", "keep", withUID("u-keep"))

	for _, p := range []*v1.Pod{pGone, pOld, pTerm, pBound, pErr, pKeep} {
		ps.AddPod(p)
	}

	store := storeFromPods(
		pNew,   // makes old UID look "recreated"
		pTerm,  // exists but terminating
		pBound, // exists but already assigned
		pKeep,  // exists and pending
	)
	errPerKey := map[string]error{"ns/err": fmt.Errorf("some lister error")}

	withPodLister(&fakePodLister{store: store, errPerKey: errPerKey}, func() {
		removed := pl.prunePodSet(ps)
		if removed != 4 {
			t.Fatalf("removed=%d want 4", removed)
		}

		snap := ps.Snapshot()
		mustMapLen(t, snap, 2)
		mustNotContainsUID(t, snap, pGone.UID)
		mustNotContainsUID(t, snap, pOld.UID)
		mustNotContainsUID(t, snap, pTerm.UID)
		mustNotContainsUID(t, snap, pBound.UID)

		mustContainsUID(t, snap, pErr.UID)
		mustContainsUID(t, snap, pKeep.UID)
	})
}

func TestPrunePodSet_ConservativeOnListerError(t *testing.T) {
	pl := &SharedState{}
	ps := newPodSet("blocked")

	p1 := pod("ns", "p1", withUID("u1"))
	p2 := pod("ns", "p2", withUID("u2"))
	ps.AddPod(p1)
	ps.AddPod(p2)

	withPodLister(&fakePodLister{
		store: storeFromPods(p1, p2),
		err:   errors.New("lister down"),
	}, func() {
		if removed := pl.prunePodSet(ps); removed != 0 {
			t.Fatalf("removed=%d want 0", removed)
		}
		snap := ps.Snapshot()
		mustContainsUID(t, snap, p1.UID)
		mustContainsUID(t, snap, p2.UID)
	})
}

// -------------------------
// Concurrency
// -------------------------

func TestPodSet_ConcurrentAccess_NoPanic(t *testing.T) {
	ps := newPodSet("blocked")

	const workers = 8
	const iters = 200

	var wg sync.WaitGroup
	wg.Add(workers)

	for i := 0; i < workers; i++ {
		i := i
		go func() {
			defer wg.Done()
			uid := types.UID(fmt.Sprintf("u-%d", i))
			p := pod("ns", fmt.Sprintf("p-%d", i), withUID(string(uid)))

			for j := 0; j < iters; j++ {
				ps.AddPod(p)
				_ = ps.Snapshot()
				ps.RemovePod(uid)
			}
		}()
	}

	wg.Wait()
	// After all removes, should be empty (best-effort; deterministic here).
	mustSize(t, ps, 0)
}
