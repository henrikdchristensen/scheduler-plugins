// config_map_helpers_test.go
package mypriorityoptimizer

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
)

// -------------------------
// Test Helpers
// -------------------------

type cmNSLister struct {
	listFn func() ([]*v1.ConfigMap, error)
	getFn  func(name string) (*v1.ConfigMap, error)
}

func (c cmNSLister) List(_ labels.Selector) ([]*v1.ConfigMap, error) { return c.listFn() }
func (c cmNSLister) Get(name string) (*v1.ConfigMap, error)          { return c.getFn(name) }

func nsLister(ns string, cms ...*v1.ConfigMap) corev1listers.ConfigMapNamespaceLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{
		cache.NamespaceIndex: cache.MetaNamespaceIndexFunc,
	})
	for _, cm := range cms {
		if cm != nil {
			_ = indexer.Add(cm)
		}
	}
	return corev1listers.NewConfigMapLister(indexer).ConfigMaps(ns)
}

func cm(ns, name string, lbls map[string]string, data map[string]string, ts time.Time) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         ns,
			Name:              name,
			Labels:            lbls,
			CreationTimestamp: metav1.NewTime(ts),
		},
		Data: data,
	}
}

func cmDoc(ns, name, labelKey, dataKey string) ConfigMapDoc {
	return ConfigMapDoc{Namespace: ns, Name: name, LabelKey: labelKey, DataKey: dataKey}
}

func setupCm(
	ns string,
	cm *v1.ConfigMap,
) (*fake.Clientset, corev1client.ConfigMapInterface, corev1listers.ConfigMapNamespaceLister) {
	var objs []runtime.Object
	if cm != nil {
		objs = append(objs, cm)
	}
	cli := fake.NewSimpleClientset(objs...)
	lister := nsLister(ns) // empty
	if cm != nil {
		lister = nsLister(ns, cm)
	}

	return cli, cli.CoreV1().ConfigMaps(ns), lister
}

func deleteActions(cli *fake.Clientset) []string {
	var out []string
	for _, a := range cli.Actions() {
		if a.GetVerb() != "delete" || a.GetResource().Resource != "configmaps" {
			continue
		}
		da, ok := a.(k8stesting.DeleteAction)
		if ok {
			out = append(out, da.GetName())
		}
	}
	return out
}

func mustReadJSON[T any](t *testing.T, cm *v1.ConfigMap, key string) T {
	t.Helper()
	var out T
	if err := json.Unmarshal([]byte(cm.Data[key]), &out); err != nil {
		t.Fatalf("json.Unmarshal(Data[%q]) err = %v", key, err)
	}
	return out
}

func assertReadJSON(t *testing.T, raw []byte, found bool, err error, wantFound bool, wantRaw *string, wantErrSubstr string) {
	t.Helper()
	if wantErrSubstr != "" {
		if err == nil || !strings.Contains(err.Error(), wantErrSubstr) {
			t.Fatalf("err=%v, want substring %q", err, wantErrSubstr)
		}
		return
	}
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if found != wantFound {
		t.Fatalf("found=%v, want %v", found, wantFound)
	}
	if wantRaw == nil {
		if raw != nil {
			t.Fatalf("raw=%q, want nil", string(raw))
		}
		return
	}
	if raw == nil {
		t.Fatalf("raw=nil, want %q", *wantRaw)
	}
	if string(raw) != *wantRaw {
		t.Fatalf("raw=%q, want %q", string(raw), *wantRaw)
	}
}

// -------------------------
// listConfigMaps
// -------------------------

func TestListConfigMaps(t *testing.T) {
	ns := "ns"
	labelKey := "myx/keep"

	base := time.Unix(1_700_000_000, 0)

	// newest to oldest cms
	cmOld := cm(ns, "old", map[string]string{labelKey: "true"}, nil, base.Add(-2*time.Hour))
	cmMid := cm(ns, "mid", map[string]string{labelKey: "true"}, nil, base.Add(-1*time.Hour))
	cmNew := cm(ns, "new", map[string]string{labelKey: "true"}, nil, base)
	cmNoLabel := cm(ns, "nolabel", nil, nil, base.Add(-30*time.Minute))

	t.Run("sorts newest first and filters by label", func(t *testing.T) {
		items, err := listConfigMaps(nsLister(ns, cmOld, cmMid, cmNew, cmNoLabel), labelKey)
		if err != nil {
			t.Fatalf("listConfigMaps error: %v", err)
		}
		got := []string{}
		for _, it := range items {
			got = append(got, it.Name)
		}
		want := []string{"new", "mid", "old"}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("order=%v, want %v", got, want)
		}
	})

	t.Run("propagates list error", func(t *testing.T) {
		wantErr := fmt.Errorf("boom")
		l := cmNSLister{
			listFn: func() ([]*v1.ConfigMap, error) { return nil, wantErr },
			getFn:  func(string) (*v1.ConfigMap, error) { t.Fatal("unexpected Get"); return nil, nil },
		}
		_, err := listConfigMaps(l, labelKey)
		if err == nil || !strings.Contains(err.Error(), "boom") {
			t.Fatalf("expected list error, got %v", err)
		}
	})
}

// -------------------------
// pruneConfigMaps
// -------------------------

func TestPruneConfigMaps(t *testing.T) {
	ctx := context.Background()
	base := time.Unix(700_000_000, 0)

	type tc struct {
		name          string
		keep          int
		listerCMs     []*v1.ConfigMap
		clientObjs    []runtime.Object
		listErr       error
		deleteErr     map[string]error
		wantErrSubstr string
		wantDeletes   []string
	}

	ns := "ns"
	labelKey := "myx/prune"

	// newest to oldest cms
	cm1 := cm(ns, "cm1", map[string]string{labelKey: "true"}, nil, base.Add(-3*time.Hour))
	cm2 := cm(ns, "cm2", map[string]string{labelKey: "true"}, nil, base.Add(-2*time.Hour))
	cm3 := cm(ns, "cm3", map[string]string{labelKey: "true"}, nil, base.Add(-1*time.Hour))
	cm4 := cm(ns, "cm4", map[string]string{labelKey: "true"}, nil, base)

	tests := []tc{
		{
			name:          "keep<=0 returns immediately",
			keep:          0,
			listerCMs:     []*v1.ConfigMap{cm1, cm2},
			clientObjs:    []runtime.Object{cm1, cm2},
			wantDeletes:   nil,
			wantErrSubstr: "",
		},
		{
			name:        "len(items)<=keep is a no-op",
			keep:        10,
			listerCMs:   []*v1.ConfigMap{cm1, cm2, cm3},
			clientObjs:  []runtime.Object{cm1, cm2, cm3},
			wantDeletes: nil,
		},
		{
			name:        "deletes older beyond keep",
			keep:        2,
			listerCMs:   []*v1.ConfigMap{cm1, cm2, cm3, cm4},
			clientObjs:  []runtime.Object{cm1, cm2, cm3, cm4},
			wantDeletes: []string{"cm2", "cm1"}, // keep cm4,cm3; delete cm2,cm1
		},
		{
			name:          "list error propagates",
			keep:          1,
			listErr:       fmt.Errorf("boom"),
			wantErrSubstr: "boom",
		},
		{
			name: "delete NotFound is ignored",
			keep: 1,
			// lister sees 3 labeled, but client only has newest
			listerCMs:   []*v1.ConfigMap{cm2, cm3, cm4},
			clientObjs:  []runtime.Object{cm4},
			wantDeletes: []string{"cm3", "cm2"},
			// fake client will return NotFound for deletes of cm3/cm2
		},
		{
			name:          "delete other error is returned",
			keep:          1,
			listerCMs:     []*v1.ConfigMap{cm2, cm3, cm4},
			clientObjs:    []runtime.Object{cm2, cm3, cm4},
			deleteErr:     map[string]error{"cm3": fmt.Errorf("delete-fail")},
			wantErrSubstr: "delete-fail",
			// delete attempts: starts at cm3 then cm2, but should stop on cm3 error
			wantDeletes: []string{"cm3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cli := fake.NewSimpleClientset(tt.clientObjs...)
			cms := cli.CoreV1().ConfigMaps(ns)

			var l corev1listers.ConfigMapNamespaceLister
			if tt.listErr != nil {
				l = cmNSLister{
					listFn: func() ([]*v1.ConfigMap, error) { return nil, tt.listErr },
					getFn:  func(string) (*v1.ConfigMap, error) { t.Fatal("unexpected Get"); return nil, nil },
				}
			} else {
				l = nsLister(ns, tt.listerCMs...)
			}

			// Per-name delete errors (and explicit NotFound behavior when object not present)
			if len(tt.deleteErr) > 0 {
				cli.Fake.PrependReactor("delete", "configmaps", func(a k8stesting.Action) (bool, runtime.Object, error) {
					da := a.(k8stesting.DeleteAction)
					if err, ok := tt.deleteErr[da.GetName()]; ok {
						return true, nil, err
					}
					return false, nil, nil // fall through to default fake behavior
				})
			}

			err := pruneConfigMaps(ctx, cms, l, labelKey, tt.keep)

			if tt.wantErrSubstr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErrSubstr) {
					t.Fatalf("err=%v, want substring %q", err, tt.wantErrSubstr)
				}
			} else if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			gotDeletes := deleteActions(cli)
			if !reflect.DeepEqual(gotDeletes, tt.wantDeletes) {
				t.Fatalf("delete actions=%v, want %v", gotDeletes, tt.wantDeletes)
			}
		})
	}

	// Prove fake client NotFound path behaves like apiserver
	t.Run("fake delete of missing returns NotFound", func(t *testing.T) {
		cli := fake.NewSimpleClientset()
		err := cli.CoreV1().ConfigMaps(ns).Delete(ctx, "missing", metav1.DeleteOptions{})
		if err == nil || !apierrors.IsNotFound(err) {
			t.Fatalf("expected NotFound, got %v", err)
		}
	})
}

// -------------------------
// marshalJsonIndented
// -------------------------

func TestMarshalJsonIndented(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		type payload struct {
			Foo string `json:"foo"`
			Bar int    `json:"bar"`
		}

		want := payload{Foo: "x", Bar: 42}

		b, err := marshalJsonIndented(want)
		if err != nil {
			t.Fatalf("marshalJsonIndented() err = %v", err)
		}

		// Round-trip into the same type
		var got payload
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("json.Unmarshal() err = %v; json=%q", err, string(b))
		}
		if got != want {
			t.Fatalf("round-trip = %#v, want %#v", got, want)
		}

		// Indentation check
		if !strings.Contains(string(b), "\n  \"foo\":") {
			t.Fatalf("expected indented JSON, got %q", string(b))
		}
	})

	t.Run("error", func(t *testing.T) {
		if _, err := marshalJsonIndented(make(chan int)); err == nil {
			t.Fatalf("marshalJsonIndented() expected error, got nil")
		}
	})
}

// -------------------------
// marshalToJsonString
// -------------------------

func TestMarshalToJsonString(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		type payload struct {
			Answer int `json:"answer"`
		}
		want := payload{Answer: 1234}

		s, err := marshalToJsonString(want)
		if err != nil {
			t.Fatalf("marshalToJsonString() err = %v", err)
		}

		// Round-trip into the same type
		var got payload
		if err := json.Unmarshal([]byte(s), &got); err != nil {
			t.Fatalf("json.Unmarshal() err = %v; json=%q", err, s)
		}
		if got != want {
			t.Fatalf("round-trip = %#v, want %#v", got, want)
		}

		// Indentation check
		if !strings.Contains(s, "\n  \"answer\":") {
			t.Fatalf("expected indented JSON, got %q", s)
		}
	})

	t.Run("error", func(t *testing.T) {
		if _, err := marshalToJsonString(make(chan int)); err == nil {
			t.Fatalf("marshalToJsonString() expected error, got nil")
		}
	})
}

// -------------------------
// patchDataString
// -------------------------

func TestPatchDataString(t *testing.T) {
	ctx := context.Background()
	ns := "ns-patch"
	name := "cm-patch"
	dataKey := "myx/plan.json"

	initial := cm(ns, name, nil, map[string]string{
		dataKey: "old-value",
		"other": "keep-me",
	}, time.Now())

	_, cms, _ := setupCm(ns, initial)
	doc := cmDoc(ns, name, "", dataKey)

	raw := `{"foo":"bar"}`
	if err := doc.patchDataString(ctx, cms, raw); err != nil {
		t.Fatalf("patchDataString() err = %v", err)
	}

	gotCM, err := cms.Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Get() after patchDataString err = %v", err)
	}

	want := map[string]string{
		dataKey: raw,
		"other": "keep-me",
	}
	for k, v := range want {
		if got := gotCM.Data[k]; got != v {
			t.Fatalf("cm.Data[%q] = %q, want %q", k, got, v)
		}
	}
}

// -------------------------
// ensureJson
// -------------------------

func TestEnsureJson_CreateAndUpdate(t *testing.T) {
	ctx := context.Background()
	ns, name := "ns1", "cm1"
	labelKey, dataKey := "myx/plan", "myx/plan.json"

	_, cms, _ := setupCm(ns, nil)
	doc := cmDoc(ns, name, labelKey, dataKey)

	type payload struct {
		Value string `json:"value"`
	}

	cases := []string{"first", "second"}
	for _, v := range cases {
		if err := doc.ensureJson(ctx, cms, payload{Value: v}); err != nil {
			t.Fatalf("ensureJson(%q) err = %v", v, err)
		}

		gotCM, err := cms.Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Get(%q) err = %v", name, err)
		}

		if got := gotCM.Labels[labelKey]; got != "true" {
			t.Fatalf("label %q = %q, want %q", labelKey, got, "true")
		}

		got := mustReadJSON[payload](t, gotCM, dataKey)
		if got.Value != v {
			t.Fatalf("Value = %q, want %q", got.Value, v)
		}
	}
}

func TestEnsureJson_UpdateOnNilData(t *testing.T) {
	ctx := context.Background()
	ns, name := "ns1-nildata", "cm-nildata"
	labelKey, dataKey := "myx/plan", "myx/plan.json"

	existing := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels:    map[string]string{labelKey: "true"},
		},
		Data: nil,
	}

	_, cms, _ := setupCm(ns, existing)
	doc := cmDoc(ns, name, labelKey, dataKey)

	type payload struct {
		Value string `json:"value"`
	}

	if err := doc.ensureJson(ctx, cms, payload{Value: "from-nil"}); err != nil {
		t.Fatalf("ensureJson err = %v", err)
	}

	gotCM, err := cms.Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Get(%q) err = %v", name, err)
	}
	if gotCM.Data == nil {
		t.Fatalf("Data is nil, want initialized map")
	}

	got := mustReadJSON[payload](t, gotCM, dataKey)
	if got.Value != "from-nil" {
		t.Fatalf("Value = %q, want %q", got.Value, "from-nil")
	}
}

func TestEnsureJson_PropagatesClientErrors(t *testing.T) {
	ctx := context.Background()
	const ns, name, lk, dk = "ns", "cm", "lk", "dk"
	doc := cmDoc(ns, name, lk, dk)

	seedForUpdate := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels:    map[string]string{lk: "true"},
		},
		Data: map[string]string{dk: `{"old":true}`},
	}

	tests := []struct {
		name, verb, wantSubstr string
		seed                   *v1.ConfigMap
	}{
		{"get fails", "get", "get-fail", nil},
		{"create fails", "create", "create-fail", nil},
		{"update fails", "update", "update-fail", seedForUpdate},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cli *fake.Clientset
			if tt.seed != nil {
				cli = fake.NewSimpleClientset(tt.seed)
			} else {
				cli = fake.NewSimpleClientset()
			}

			cli.Fake.PrependReactor(tt.verb, "configmaps", func(_ k8stesting.Action) (bool, runtime.Object, error) {
				return true, nil, fmt.Errorf("%s", tt.wantSubstr)
			})

			err := doc.ensureJson(ctx, cli.CoreV1().ConfigMaps(ns), map[string]any{"x": 1})
			if err == nil || !strings.Contains(err.Error(), tt.wantSubstr) {
				t.Fatalf("err=%v, want substring %q", err, tt.wantSubstr)
			}
		})
	}
}

func TestEnsureJson_MarshalError_NoClientActions(t *testing.T) {
	ctx := context.Background()
	cli, cms, _ := setupCm("ns", nil)
	doc := cmDoc("ns", "cm", "lk", "dk")

	if err := doc.ensureJson(ctx, cms, make(chan int)); err == nil {
		t.Fatalf("expected marshal error, got nil")
	}
	if len(cli.Actions()) != 0 {
		t.Fatalf("expected no client actions, got %#v", cli.Actions())
	}
}

// -------------------------
// readJson
// -------------------------

func TestReadJson(t *testing.T) {
	ns, name, dk := "ns", "cm", "dk"
	doc := cmDoc(ns, name, "", dk)
	now := time.Now()
	presentVal := `{"hello":"world"}`
	presentCM := cm(ns, name, map[string]string{"": ""}, map[string]string{dk: presentVal}, now)
	emptyKeyCM := cm(ns, name, map[string]string{"": ""}, map[string]string{}, now)

	tests := []struct {
		name         string
		lister       corev1listers.ConfigMapNamespaceLister
		customLister corev1listers.ConfigMapNamespaceLister
		wantFound    bool
		wantRaw      *string
		wantErrSub   string
	}{
		{
			name: "nil configmap treated as missing",
			customLister: cmNSLister{
				getFn:  func(string) (*v1.ConfigMap, error) { return nil, nil },
				listFn: func() ([]*v1.ConfigMap, error) { t.Fatal("unexpected List"); return nil, nil },
			},
			wantFound: false,
			wantRaw:   nil,
		},
		{
			name:      "not found is missing",
			lister:    nsLister(ns),
			wantFound: false,
			wantRaw:   nil,
		},
		{
			name:      "present returns bytes",
			lister:    nsLister(ns, presentCM),
			wantFound: true,
			wantRaw:   &presentVal,
		},
		{
			name: "get error propagates",
			customLister: cmNSLister{
				getFn:  func(string) (*v1.ConfigMap, error) { return nil, fmt.Errorf("boom") },
				listFn: func() ([]*v1.ConfigMap, error) { t.Fatal("unexpected List"); return nil, nil },
			},
			wantErrSub: "boom",
		},
		{
			name:      "key missing returns empty bytes but found=true",
			lister:    nsLister(ns, emptyKeyCM),
			wantFound: true,
			wantRaw:   ptr(""),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := tt.lister
			if tt.customLister != nil {
				l = tt.customLister
			}
			raw, found, err := doc.readJson(l)
			assertReadJSON(t, raw, found, err, tt.wantFound, tt.wantRaw, tt.wantErrSub)
		})
	}
}

// -------------------------
// patchJson
// -------------------------

func TestPatchJson(t *testing.T) {
	ctx := context.Background()

	type payload struct {
		Value string `json:"value"`
	}

	tests := []struct {
		name      string
		ns        string
		cmName    string
		labelKey  string
		dataKey   string
		in        any
		wantErr   bool
		wantValue string // only used on success
		wantRaw   string // expected raw stored in cm after call
	}{
		{
			name:      "success patches json",
			ns:        "ns2",
			cmName:    "cm2",
			labelKey:  "myx/plan",
			dataKey:   "myx/plan.json",
			in:        payload{Value: "patched"},
			wantErr:   false,
			wantValue: "patched",
			wantRaw:   "", // computed after marshal/unmarshal check
		},
		{
			name:     "marshal failure returns error and does not change data",
			ns:       "ns2-err",
			cmName:   "cm2-err",
			labelKey: "myx/plan",
			dataKey:  "myx/plan.json",
			in:       make(chan int),
			wantErr:  true,
			wantRaw:  `{"value":"old"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			old := `{"value":"old"}`
			initial := cm(tt.ns, tt.cmName, map[string]string{tt.labelKey: "true"}, map[string]string{
				tt.dataKey: old,
			}, time.Now())

			_, cms, _ := setupCm(tt.ns, initial)
			doc := cmDoc(tt.ns, tt.cmName, tt.labelKey, tt.dataKey)
			err := doc.patchJson(ctx, cms, tt.in)

			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				after, gerr := cms.Get(ctx, tt.cmName, metav1.GetOptions{})
				if gerr != nil {
					t.Fatalf("Get after patchJson failed: %v", gerr)
				}
				if got := after.Data[tt.dataKey]; got != old {
					t.Fatalf("expected dataKey unchanged; got %q, want %q", got, old)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			after, gerr := cms.Get(ctx, tt.cmName, metav1.GetOptions{})
			if gerr != nil {
				t.Fatalf("Get after patch failed: %v", gerr)
			}

			var got payload
			if uerr := json.Unmarshal([]byte(after.Data[tt.dataKey]), &got); uerr != nil {
				t.Fatalf("unmarshal patched data failed: %v", uerr)
			}
			if got.Value != tt.wantValue {
				t.Fatalf("Value=%q, want %q", got.Value, tt.wantValue)
			}
		})
	}
}

// -------------------------
// mutateJson
// -------------------------

func TestMutateJson(t *testing.T) {
	ctx := context.Background()

	type item struct {
		ID int `json:"id"`
	}

	mustGet := func(t *testing.T, cms interface {
		Get(context.Context, string, metav1.GetOptions) (*v1.ConfigMap, error)
	}, name string) *v1.ConfigMap {
		t.Helper()
		got, err := cms.Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Get(%s) failed: %v", name, err)
		}
		return got
	}

	tests := []struct {
		name     string
		ns       string
		cmName   string
		dataKey  string
		raw      string
		lister   corev1listers.ConfigMapNamespaceLister
		patchErr string
		run      func(t *testing.T, cms corev1client.ConfigMapInterface) error
		assert   func(t *testing.T, cms corev1client.ConfigMapInterface)
		wantSub  string
	}{
		{
			name:    "appends",
			ns:      "ns4",
			cmName:  "cm4",
			dataKey: "myx/arr.json",
			raw:     `[{"id":1},{"id":2}]`,
			run: func(t *testing.T, cms corev1client.ConfigMapInterface) error {
				cm0 := cm("ns4", "cm4", nil, map[string]string{"myx/arr.json": `[{"id":1},{"id":2}]`}, time.Now())
				doc := cmDoc("ns4", "cm4", "", "myx/arr.json")
				l := nsLister("ns4", cm0)
				return mutateJson(ctx, cms, l, doc, func(existing []item) ([]item, error) {
					return append(existing, item{ID: 3}), nil
				})
			},
			assert: func(t *testing.T, cms corev1client.ConfigMapInterface) {
				updated := mustGet(t, cms, "cm4")
				var arr []item
				if err := json.Unmarshal([]byte(updated.Data["myx/arr.json"]), &arr); err != nil {
					t.Fatalf("unmarshal failed: %v", err)
				}
				if len(arr) != 3 || arr[2].ID != 3 {
					t.Fatalf("expected appended ID=3, got %#v", arr)
				}
			},
		},
		{
			name:    "mutate error leaves data unchanged",
			ns:      "ns4-err",
			cmName:  "cm4-err",
			dataKey: "myx/arr.json",
			raw:     `[{"id":1}]`,
			run: func(t *testing.T, cms corev1client.ConfigMapInterface) error {
				cm0 := cm("ns4-err", "cm4-err", nil, map[string]string{"myx/arr.json": `[{"id":1}]`}, time.Now())
				doc := cmDoc("ns4-err", "cm4-err", "", "myx/arr.json")
				l := nsLister("ns4-err", cm0)
				return mutateJson(ctx, cms, l, doc, func(existing []item) ([]item, error) {
					if len(existing) != 1 || existing[0].ID != 1 {
						t.Fatalf("unexpected existing: %#v", existing)
					}
					return nil, fmt.Errorf("boom")
				})
			},
			wantSub: "boom",
			assert: func(t *testing.T, cms corev1client.ConfigMapInterface) {
				updated := mustGet(t, cms, "cm4-err")
				if got := updated.Data["myx/arr.json"]; got != `[{"id":1}]` {
					t.Fatalf("expected unchanged raw, got %q", got)
				}
			},
		},
		{
			name: "read error propagates and mutate not called",
			ns:   "ns",
			run: func(t *testing.T, cms corev1client.ConfigMapInterface) error {
				doc := cmDoc("ns", "cm", "", "dk")
				called := false
				l := cmNSLister{
					getFn:  func(string) (*v1.ConfigMap, error) { return nil, fmt.Errorf("read-fail") },
					listFn: func() ([]*v1.ConfigMap, error) { return nil, nil },
				}
				err := mutateJson(ctx, cms, l, doc, func(_ []int) ([]int, error) {
					called = true
					return nil, nil
				})
				if called {
					t.Fatalf("mutate must not run on read error")
				}
				return err
			},
			wantSub: "read-fail",
		},
		{
			name:     "patch failure propagates",
			ns:       "ns",
			cmName:   "cm",
			dataKey:  "dk",
			raw:      `[1,2]`,
			patchErr: "patch-fail",
			run: func(t *testing.T, cms corev1client.ConfigMapInterface) error {
				cm0 := cm("ns", "cm", nil, map[string]string{"dk": `[1,2]`}, time.Now())
				doc := cmDoc("ns", "cm", "", "dk")
				l := nsLister("ns", cm0)
				return mutateJson(ctx, cms, l, doc, func(existing []int) ([]int, error) {
					return append(existing, 3), nil
				})
			},
			wantSub: "patch-fail",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var seed *v1.ConfigMap
			if tt.cmName != "" && tt.dataKey != "" {
				seed = cm(tt.ns, tt.cmName, nil, map[string]string{tt.dataKey: tt.raw}, time.Now())
			}

			var cli *fake.Clientset
			if seed != nil {
				cli = fake.NewSimpleClientset(seed)
			} else {
				cli = fake.NewSimpleClientset()
			}
			cms := cli.CoreV1().ConfigMaps(tt.ns)

			if tt.patchErr != "" {
				cli.Fake.PrependReactor("patch", "configmaps", func(_ k8stesting.Action) (bool, runtime.Object, error) {
					return true, nil, fmt.Errorf("%s", tt.patchErr)
				})
			}

			err := tt.run(t, cms)

			if tt.wantSub != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantSub) {
					t.Fatalf("err=%v, want substring %q", err, tt.wantSub)
				}
			} else if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			if tt.assert != nil {
				tt.assert(t, cms)
			}
		})
	}
}

// -------------------------
// mutateRaw
// -------------------------

func TestMutateRaw(t *testing.T) {
	ctx := context.Background()

	type tc struct {
		name      string
		ns, cm    string
		dataKey   string
		seed      *v1.ConfigMap // nil => missing
		mutate    func(t *testing.T) func([]byte) ([]byte, error)
		wantErr   string
		wantCalls int
		wantVal   *string // nil => CM must not exist
	}

	tests := []tc{
		{
			name:    "uppercases",
			ns:      "ns5",
			cm:      "cm5",
			dataKey: "myx/raw.json",
			seed: cm("ns5", "cm5", nil, map[string]string{
				"myx/raw.json": `{"msg":"hello"}`,
			}, time.Now()),
			mutate: func(t *testing.T) func([]byte) ([]byte, error) {
				return func(_ []byte) ([]byte, error) {
					return []byte(`{"msg":"HELLO"}`), nil
				}
			},
			wantCalls: 1,
			wantVal:   ptr(`{"msg":"HELLO"}`),
		},
		{
			name:    "missing configmap => no-op, mutate not called, nothing created",
			ns:      "ns5-miss",
			cm:      "cm5-miss",
			dataKey: "myx/raw.json",
			seed:    nil,
			mutate: func(t *testing.T) func([]byte) ([]byte, error) {
				return func(raw []byte) ([]byte, error) {
					t.Fatalf("mutate must not be called on missing CM (raw=%q)", string(raw))
					return raw, nil
				}
			},
			wantCalls: 0,
			wantVal:   nil, // CM must not exist
		},
		{
			name:    "nil newRaw => no-op (keeps original)",
			ns:      "ns5-nilraw",
			cm:      "cm5-nilraw",
			dataKey: "myx/raw.json",
			seed: cm("ns5-nilraw", "cm5-nilraw", nil, map[string]string{
				"myx/raw.json": `{"msg":"hello"}`,
			}, time.Now()),
			mutate: func(t *testing.T) func([]byte) ([]byte, error) {
				original := `{"msg":"hello"}`
				return func(raw []byte) ([]byte, error) {
					if string(raw) != original {
						t.Fatalf("unexpected raw: got %q want %q", string(raw), original)
					}
					return nil, nil // "no change"
				}
			},
			wantCalls: 1,
			wantVal:   ptr(`{"msg":"hello"}`),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, cms, _ := setupCm(tt.ns, tt.seed)
			doc := cmDoc(tt.ns, tt.cm, "", tt.dataKey)

			var calls int
			err := doc.mutateRaw(ctx, cms, nsLister(tt.ns, tt.seed), func(raw []byte) ([]byte, error) {
				calls++
				return tt.mutate(t)(raw)
			})

			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err=%v, want substring %q", err, tt.wantErr)
				}
			} else if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			if calls != tt.wantCalls {
				t.Fatalf("calls=%d, want %d", calls, tt.wantCalls)
			}

			got, getErr := cms.Get(ctx, tt.cm, metav1.GetOptions{})
			if tt.wantVal == nil {
				if getErr == nil || !apierrors.IsNotFound(getErr) {
					t.Fatalf("expected CM to be missing (NotFound), got err=%v cm=%#v", getErr, got)
				}
				return
			}

			if getErr != nil {
				t.Fatalf("Get failed: %v", getErr)
			}
			if got.Data[tt.dataKey] != *tt.wantVal {
				t.Fatalf("data[%q]=%q, want %q", tt.dataKey, got.Data[tt.dataKey], *tt.wantVal)
			}
		})
	}
}
