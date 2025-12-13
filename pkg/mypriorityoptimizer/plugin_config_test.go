// pkg/mypriorityoptimizer/plugin_config_test.go
package mypriorityoptimizer

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

func getPluginCfgCM(t *testing.T, ctx context.Context, client *fake.Clientset) *v1.ConfigMap {
	t.Helper()
	cm, err := client.CoreV1().ConfigMaps(SystemNamespace).Get(ctx, PluginCfgConfigMapName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Get(%s/%s) failed: %v", SystemNamespace, PluginCfgConfigMapName, err)
	}
	return cm
}

func decodePluginCfgSnap(t *testing.T, cm *v1.ConfigMap) PluginConfigSnapshot {
	t.Helper()

	key := PluginCfgConfigMapLabelKey + ".json"
	raw := cm.Data[key]
	if raw == "" {
		t.Fatalf("expected data key %q to be present and non-empty", key)
	}

	var snap PluginConfigSnapshot
	if err := json.Unmarshal([]byte(raw), &snap); err != nil {
		t.Fatalf("unmarshal snapshot failed: %v", err)
	}
	return snap
}

func TestBuildPluginConfigSnapshot_BasicFields(t *testing.T) {
	snap := buildPluginConfigSnapshot()
	if snap.Name != Name {
		t.Fatalf("expected Name=%q, got %q", Name, snap.Name)
	}
	if snap.Version != PluginVersion {
		t.Fatalf("expected Version=%q, got %q", PluginVersion, snap.Version)
	}
	if snap.SystemNamespace != SystemNamespace {
		t.Fatalf("expected SystemNamespace=%q, got %q", SystemNamespace, snap.SystemNamespace)
	}
	if snap.OptimizeMode == "" {
		t.Fatalf("expected non-empty OptimizeMode")
	}
}

func TestPersistPluginConfig(t *testing.T) {
	ctx, cancel := testCtx(t)
	defer cancel()

	t.Run("no-op when nil receiver or nil client", func(t *testing.T) {
		var pl *SharedState
		if err := pl.persistPluginConfig(ctx); err != nil {
			t.Fatalf("nil receiver: expected nil error, got %v", err)
		}
		pl = &SharedState{Client: nil}
		if err := pl.persistPluginConfig(ctx); err != nil {
			t.Fatalf("nil client: expected nil error, got %v", err)
		}
	})

	t.Run("create then update succeeds", func(t *testing.T) {
		client := fake.NewSimpleClientset()
		pl := &SharedState{Client: client, BlockedWhileActive: newPodSet("x")}

		// Create
		if err := pl.persistPluginConfig(ctx); err != nil {
			t.Fatalf("persistPluginConfig(create) error: %v", err)
		}
		cm := getPluginCfgCM(t, ctx, client)
		if cm.Labels[PluginCfgConfigMapLabelKey] != "true" {
			t.Fatalf("expected label %q=true, got %v", PluginCfgConfigMapLabelKey, cm.Labels)
		}
		snap := decodePluginCfgSnap(t, cm)
		if snap.Name != Name {
			t.Fatalf("expected snap.Name=%q, got %q", Name, snap.Name)
		}

		// Update
		if err := pl.persistPluginConfig(ctx); err != nil {
			t.Fatalf("persistPluginConfig(update) error: %v", err)
		}
		list, err := client.CoreV1().ConfigMaps(SystemNamespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			t.Fatalf("List failed: %v", err)
		}
		found := 0
		for _, c := range list.Items {
			if c.Name == PluginCfgConfigMapName {
				found++
			}
		}
		if found != 1 {
			t.Fatalf("expected exactly 1 configmap named %q, found %d", PluginCfgConfigMapName, found)
		}
	})

	t.Run("ensureJson get fails -> propagates error", func(t *testing.T) {
		client := fake.NewSimpleClientset()
		client.Fake.PrependReactor("get", "configmaps", func(_ k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, fmt.Errorf("get-fail")
		})

		pl := &SharedState{Client: client, BlockedWhileActive: newPodSet("x")}
		err := pl.persistPluginConfig(ctx)
		if err == nil || !strings.Contains(err.Error(), "get-fail") {
			t.Fatalf("expected get-fail error, got %v", err)
		}
	})

	t.Run("create fails -> propagates error", func(t *testing.T) {
		client := fake.NewSimpleClientset()
		client.Fake.PrependReactor("create", "configmaps", func(_ k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, fmt.Errorf("create-fail")
		})

		pl := &SharedState{Client: client, BlockedWhileActive: newPodSet("x")}
		err := pl.persistPluginConfig(ctx)
		if err == nil || !strings.Contains(err.Error(), "create-fail") {
			t.Fatalf("expected create-fail error, got %v", err)
		}
	})

	t.Run("update fails -> propagates error", func(t *testing.T) {
		seed := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      PluginCfgConfigMapName,
				Namespace: SystemNamespace,
				Labels:    map[string]string{PluginCfgConfigMapLabelKey: "true"},
			},
			Data: map[string]string{PluginCfgConfigMapLabelKey + ".json": `{}`},
		}
		client := fake.NewSimpleClientset(seed)
		client.Fake.PrependReactor("update", "configmaps", func(_ k8stesting.Action) (bool, runtime.Object, error) {
			return true, nil, fmt.Errorf("update-fail")
		})

		pl := &SharedState{Client: client, BlockedWhileActive: newPodSet("x")}
		err := pl.persistPluginConfig(ctx)
		if err == nil || !strings.Contains(err.Error(), "update-fail") {
			t.Fatalf("expected update-fail error, got %v", err)
		}
	})
}
