// pod_set_helpers_test.go
package mypriorityoptimizer

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// -------------------------
// Assertions
// -------------------------

func mustHasUID(t *testing.T, snap map[types.UID]SolverPod, uid types.UID) {
	t.Helper()
	if _, ok := snap[uid]; !ok {
		t.Fatalf("snapshot missing uid=%q; snap=%v", uid, snap)
	}
}

func mustNotHaveUID(t *testing.T, snap map[types.UID]SolverPod, uid types.UID) {
	t.Helper()
	if _, ok := snap[uid]; ok {
		t.Fatalf("snapshot unexpectedly contains uid=%q; snap=%v", uid, snap)
	}
}

// -------------------------
// newSafePodSet + doesSafePodSetExist
// -------------------------

func TestSafePodSet_NewAndExistence(t *testing.T) {
	t.Run("newSafePodSet returns empty set with name", func(t *testing.T) {
		ps := newPodSet("blocked")
		if ps == nil {
			t.Fatalf("newSafePodSet returned nil")
		}
		if ps.Name != "blocked" {
			t.Fatalf("Name=%q want %q", ps.Name, "blocked")
		}
		if got := ps.Size(); got != 0 {
			t.Fatalf("Size=%d want 0", got)
		}
		if snap := ps.Snapshot(); len(snap) != 0 {
			t.Fatalf("Snapshot len=%d want 0", len(snap))
		}
	})

	t.Run("doesSafePodSetExist", func(t *testing.T) {
		if doesPodSetExist(nil) {
			t.Fatalf("nil should not exist")
		}
		empty := newPodSet("empty")
		if doesPodSetExist(empty) {
			t.Fatalf("empty should not exist")
		}

		ps := newPodSet("blocked")
		ps.AddPod(pod("ns", "p1"))
		if !doesPodSetExist(ps) {
			t.Fatalf("non-empty should exist")
		}
	})
}

// -------------------------
// SafePodSet operations
// -------------------------

func TestSafePodSet_AddRemoveSnapshot(t *testing.T) {
	ps := newPodSet("blocked")

	t.Run("AddPodSafely(nil) is no-op", func(t *testing.T) {
		ps.AddPod(nil)
		if ps.Size() != 0 {
			t.Fatalf("size=%d want 0", ps.Size())
		}
	})

	t.Run("Add -> Snapshot contains SolverPod fields", func(t *testing.T) {
		p := pod("ns", "p1", withUID("u1"))
		ps.AddPod(p)

		if ps.Size() != 1 {
			t.Fatalf("size=%d want 1", ps.Size())
		}

		snap := ps.Snapshot()
		mustHasUID(t, snap, p.UID)

		got := snap[p.UID]
		if got.UID != p.UID || got.Namespace != p.Namespace || got.Name != p.Name {
			t.Fatalf("got=%#v want uid=%q ns=%q name=%q", got, p.UID, p.Namespace, p.Name)
		}
	})

	t.Run("SnapshotSafely returns a copy", func(t *testing.T) {
		p := pod("ns", "copy", withUID("u-copy"))
		ps2 := newPodSet("blocked")
		ps2.AddPod(p)

		snap := ps2.Snapshot()
		delete(snap, p.UID) // mutate returned map

		if ps2.Size() != 1 {
			t.Fatalf("internal map affected by snapshot mutation; size=%d want 1", ps2.Size())
		}
	})

	t.Run("RemovePodSafely missing uid is no-op", func(t *testing.T) {
		ps2 := newPodSet("blocked")
		ps2.RemovePod(types.UID("nope"))
		if ps2.Size() != 0 {
			t.Fatalf("size=%d want 0", ps2.Size())
		}
	})

	t.Run("Add overwrites by UID (last write wins)", func(t *testing.T) {
		ps2 := newPodSet("blocked")

		p1 := pod("ns1", "a", withUID("same"))
		p2 := pod("ns2", "b", withUID("same")) // same UID, different ns/name
		ps2.AddPod(p1)
		ps2.AddPod(p2)

		snap := ps2.Snapshot()
		if ps2.Size() != 1 {
			t.Fatalf("size=%d want 1", ps2.Size())
		}
		got := snap[p1.UID]
		if got.Namespace != "ns2" || got.Name != "b" {
			t.Fatalf("overwrite failed: got=%#v", got)
		}
	})

	t.Run("RemovePodSafely removes existing", func(t *testing.T) {
		ps2 := newPodSet("blocked")
		p := pod("ns", "rm", withUID("u-rm"))
		ps2.AddPod(p)
		ps2.RemovePod(p.UID)

		if ps2.Size() != 0 {
			t.Fatalf("size=%d want 0", ps2.Size())
		}
		if snap := ps2.Snapshot(); len(snap) != 0 {
			t.Fatalf("snap len=%d want 0", len(snap))
		}
	})
}

// -------------------------
// pruneSafePodSet
// -------------------------

func TestPruneSafePodSet_NilOrEmpty(t *testing.T) {
	pl := &SharedState{}

	if got := pl.prunePodSet(nil); got != 0 {
		t.Fatalf("removed=%d want 0", got)
	}
	empty := newPodSet("empty")
	if got := pl.prunePodSet(empty); got != 0 {
		t.Fatalf("removed=%d want 0", got)
	}
}

func TestPruneSafePodSet_BranchCoverage(t *testing.T) {
	pl := &SharedState{}
	ps := newPodSet("blocked")

	// 1) NotFound (absent from store) => pruned
	pGone := pod("ns", "gone", withUID("u-gone"))

	// 2) Recreated (same ns/name but different UID) => pruned
	pRecreatedOld := pod("ns", "recreated", withUID("u-old"))
	pRecreatedNew := pod("ns", "recreated", withUID("u-new"))

	// 3) Terminating => pruned
	pTerminating := pod("ns", "term", withUID("u-term"))
	now := metav1.Now()
	pTerminating.DeletionTimestamp = &now

	// 4) Bound => pruned
	pBound := pod("ns", "bound", withUID("u-bound"), onNode("node1"))

	// 5) Lister error (non-NotFound) => kept
	pErr := pod("ns", "err", withUID("u-err"))

	// 6) Valid pending => kept
	pKeep := pod("ns", "keep", withUID("u-keep"))

	ps.AddPod(pGone)
	ps.AddPod(pRecreatedOld)
	ps.AddPod(pTerminating)
	ps.AddPod(pBound)
	ps.AddPod(pErr)
	ps.AddPod(pKeep)

	store := storeFromPods(
		pRecreatedNew, // replaces old UID case via Get
		pTerminating,
		pBound,
		pKeep,
	)
	errPerKey := map[string]error{
		"ns/err": fmt.Errorf("some lister error"),
	}

	withPodLister(&fakePodLister{
		store:     store,
		errPerKey: errPerKey,
	}, func() {
		removed := pl.prunePodSet(ps)
		if removed != 4 {
			t.Fatalf("removed=%d want 4", removed)
		}

		snap := ps.Snapshot()

		mustNotHaveUID(t, snap, pGone.UID)
		mustNotHaveUID(t, snap, pRecreatedOld.UID)
		mustNotHaveUID(t, snap, pTerminating.UID)
		mustNotHaveUID(t, snap, pBound.UID)

		// kept:
		mustHasUID(t, snap, pErr.UID)
		mustHasUID(t, snap, pKeep.UID)
	})
}

func TestPruneSafePodSet_ConservativeOnListerError(t *testing.T) {
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
		removed := pl.prunePodSet(ps)
		if removed != 0 {
			t.Fatalf("removed=%d want 0", removed)
		}
		// should keep everything on lister error
		snap := ps.Snapshot()
		mustHasUID(t, snap, p1.UID)
		mustHasUID(t, snap, p2.UID)
	})
}

// -------------------------
// concurrency
// -------------------------

func TestSafePodSet_ConcurrentAccess_NoPanic(t *testing.T) {
	ps := newPodSet("blocked")

	const workers = 8
	const iters = 200

	var wg sync.WaitGroup
	wg.Add(workers)

	uids := make([]types.UID, 0, workers)
	for i := 0; i < workers; i++ {
		uids = append(uids, types.UID(fmt.Sprintf("u-%d", i)))
	}

	for i := 0; i < workers; i++ {
		i := i
		go func() {
			defer wg.Done()
			for j := 0; j < iters; j++ {
				ps.AddPod(pod("ns", fmt.Sprintf("p-%d", i), withUID(string(uids[i]))))
				_ = ps.Snapshot()
				ps.RemovePod(uids[i])
			}
		}()
	}
	wg.Wait()

	// Best-effort cleanup
	for _, uid := range uids {
		ps.RemovePod(uid)
	}
	if ps.Size() != 0 {
		t.Fatalf("size=%d want 0", ps.Size())
	}
}
