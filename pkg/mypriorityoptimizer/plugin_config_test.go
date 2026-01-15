// plugin_config_test.go
package mypriorityoptimizer

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

// -------------------------
// buildPluginConfigSnapshot
// -------------------------

func TestBuildPluginConfigSnapshot(t *testing.T) {
	snap := buildPluginConfigSnapshot()

	if snap.Name != Name {
		t.Fatalf("Name=%q want %q", snap.Name, Name)
	}
	if snap.Version != PluginVersion {
		t.Fatalf("Version=%q want %q", snap.Version, PluginVersion)
	}
	if snap.SystemNamespace != SystemNamespace {
		t.Fatalf("SystemNamespace=%q want %q", snap.SystemNamespace, SystemNamespace)
	}
	if snap.OptimizeMode != getModeCombinedAsString() {
		t.Fatalf("OptimizeMode=%q want %q", snap.OptimizeMode, getModeCombinedAsString())
	}
	if snap.Timestamp.IsZero() {
		t.Fatalf("Timestamp is zero; want non-zero")
	}
	// Timestamp sanity: it should be "around now" (avoid being too strict).
	if dt := time.Since(snap.Timestamp); dt < 0 || dt > 5*time.Minute {
		t.Fatalf("Timestamp looks wrong: now-snap=%v (snap=%v)", dt, snap.Timestamp)
	}
}

// -------------------------
// persistPluginConfig
// -------------------------

func TestPersistPluginConfig_NoOpOnNil(t *testing.T) {
	ctx, cancel := testCtx(t)
	defer cancel()

	var pl *SharedState
	if err := pl.persistPluginConfig(ctx); err != nil {
		t.Fatalf("nil receiver: want nil error, got %v", err)
	}

	pl = &SharedState{Client: nil}
	if err := pl.persistPluginConfig(ctx); err != nil {
		t.Fatalf("nil client: want nil error, got %v", err)
	}
}

func TestPersistPluginConfig_CreateThenUpdate(t *testing.T) {
	ctx, cancel := testCtx(t)
	defer cancel()

	client := fake.NewSimpleClientset()
	pl := &SharedState{
		Client:             client,
		BlockedWhileActive: newPodSet("x"),
	}
	// First create.
	if err := pl.persistPluginConfig(ctx); err != nil {
		t.Fatalf("persistPluginConfig(create) error: %v", err)
	}

	cm1 := mustGetPluginCfgCM(t, ctx, client)
	mustLabelTrue(t, cm1, PluginCfgConfigMapLabelKey)
	snap1 := mustDecodePluginCfgSnap(t, cm1)
	if snap1.HTTPAddr != HTTPAddr {
		t.Fatalf("snap1.HTTPAddr=%q want %q", snap1.HTTPAddr, HTTPAddr)
	}
	key := PluginCfgConfigMapLabelKey + ".json"
	raw1 := cm1.Data[key]
	if raw1 == "" {
		t.Fatalf("expected snapshot data %q to be non-empty", key)
	}

	// Ensure the next snapshot is very likely to differ (timestamp is part of the snapshot).
	time.Sleep(2 * time.Millisecond)

	// ...then persist again and verify it updated.
	if err := pl.persistPluginConfig(ctx); err != nil {
		t.Fatalf("persistPluginConfig(update) error: %v", err)
	}

	cm2 := mustGetPluginCfgCM(t, ctx, client)
	mustLabelTrue(t, cm2, PluginCfgConfigMapLabelKey)
	snap2 := mustDecodePluginCfgSnap(t, cm2)
	if snap2.HTTPAddr != HTTPAddr {
		t.Fatalf("snap2.HTTPAddr=%q want %q", snap2.HTTPAddr, HTTPAddr)
	}
	raw2 := cm2.Data[key]
	if raw2 == "" {
		t.Fatalf("expected snapshot data %q to be non-empty on update", key)
	}
	if raw1 == raw2 {
		t.Fatalf("expected snapshot JSON to change between create and update")
	}
}

func TestPersistPluginConfig_PropagatesErrors(t *testing.T) {
	ctx, cancel := testCtx(t)
	defer cancel()

	type tc struct {
		name       string
		verb       string
		seed       *v1.ConfigMap
		wantSubstr string
	}

	tests := []tc{
		{
			name:       "get fails",
			verb:       "get",
			seed:       nil,
			wantSubstr: "get-fail",
		},
		{
			name:       "create fails",
			verb:       "create",
			seed:       nil,
			wantSubstr: "create-fail",
		},
		{
			name: "update fails",
			verb: "update",
			seed: cm(SystemNamespace, PluginCfgConfigMapName,
				map[string]string{PluginCfgConfigMapLabelKey: "true"},
				map[string]string{PluginCfgConfigMapLabelKey + ".json": `{}`},
				time.Now(),
			),
			wantSubstr: "update-fail",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var client *fake.Clientset
			if tt.seed != nil {
				client = fake.NewSimpleClientset(tt.seed)
			} else {
				client = fake.NewSimpleClientset()
			}

			client.Fake.PrependReactor(tt.verb, "configmaps", func(_ k8stesting.Action) (bool, runtime.Object, error) {
				return true, nil, fmt.Errorf("%s", tt.wantSubstr)
			})

			pl := &SharedState{Client: client, BlockedWhileActive: newPodSet("x")}
			err := pl.persistPluginConfig(ctx)
			if err == nil || !strings.Contains(err.Error(), tt.wantSubstr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantSubstr, err)
			}

			// Verify CM existence
			_, getErr := client.CoreV1().ConfigMaps(SystemNamespace).Get(ctx, PluginCfgConfigMapName, metav1.GetOptions{})
			if tt.verb == "create" {
				if getErr == nil {
					t.Fatalf("expected CM to not exist after create failure")
				}
				if !apierrors.IsNotFound(getErr) {
					t.Fatalf("expected NotFound after create failure, got %v", getErr)
				}
			}
			if tt.verb == "update" {
				if getErr != nil {
					t.Fatalf("expected seeded CM to still exist after update failure, got %v", getErr)
				}
			}
		})
	}
}

// -------------------------
// Test Helpers
// -------------------------

func mustGetPluginCfgCM(t *testing.T, ctx context.Context, client *fake.Clientset) *v1.ConfigMap {
	t.Helper()
	cm, err := client.CoreV1().ConfigMaps(SystemNamespace).Get(ctx, PluginCfgConfigMapName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Get(%s/%s) failed: %v", SystemNamespace, PluginCfgConfigMapName, err)
	}
	return cm
}

func mustDecodePluginCfgSnap(t *testing.T, cm *v1.ConfigMap) PluginConfigSnapshot {
	t.Helper()

	if cm == nil {
		t.Fatal("nil ConfigMap")
	}
	key := PluginCfgConfigMapLabelKey + ".json"
	if cm.Data == nil {
		t.Fatalf("expected cm.Data to be non-nil (missing key %q)", key)
	}
	raw, ok := cm.Data[key]
	if !ok || raw == "" {
		t.Fatalf("expected data key %q to be present and non-empty; data=%v", key, cm.Data)
	}

	var snap PluginConfigSnapshot
	if err := json.Unmarshal([]byte(raw), &snap); err != nil {
		t.Fatalf("unmarshal snapshot failed: %v (raw=%q)", err, raw)
	}
	return snap
}

func mustLabelTrue(t *testing.T, cm *v1.ConfigMap, labelKey string) {
	t.Helper()
	if cm.Labels == nil || cm.Labels[labelKey] != "true" {
		t.Fatalf("expected label %q=true, got labels=%v", labelKey, cm.Labels)
	}
}
