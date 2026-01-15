// plugin_config.go
package mypriorityoptimizer

import (
	"context"

	"k8s.io/klog/v2"
)

// buildPluginConfigSnapshot collects the current plugin configuration into a
// PluginConfigSnapshot structure.
func buildPluginConfigSnapshot() PluginConfigSnapshot {
	return PluginConfigSnapshot{
		Timestamp: getTimestampNowUtc(),

		Name:    Name,
		Version: PluginVersion,

		SystemNamespace:                   SystemNamespace,
		CacheWarmupSettleDelay:            CacheWarmupSettleDelay.String(),
		PluginReadinessUsableNodeInterval: PluginReadinessUsableNodeInterval.String(),

		OptimizeMode:                     getModeCombinedAsString(),
		OptimizePeriodicInterval:         OptimizePeriodicInterval.String(),
		OptimizeStableQueueDelay:         OptimizeStableQueueDelay.String(),
		OptimizeStableQueueCheckInterval: OptimizeStableQueueCheckInterval.String(),

		HTTPAddr: HTTPAddr,

		SolverSaveAllAttempts:              SolverSaveAllAttempts,
		SolverPythonEnabled:                SolverPythonEnabled,
		SolverPythonTimeout:                SolverPythonTimeout.String(),
		SolverPythonScriptPath:             SolverPythonScriptPath,
		SolverPythonBin:                    SolverPythonBin,
		SolverPythonGapLimit:               SolverPythonGapLimit,
		SolverPythonGuaranteedTierFraction: SolverPythonGuaranteedTierFraction,
		SolverPythonMoveFractionOfTier:     SolverPythonMoveFractionOfTier,
		SolverPythonGraceMs:                SolverPythonGraceMs,

		SolverLogProgress:            SolverLogProgress,
		SolverStatsConfigMapName:     SolverStatsConfigMapName,
		SolverStatsConfigMapLabelKey: SolverStatsConfigMapLabelKey,

		PlanRealizationTimeout:      PlanRealizationTimeout.String(),
		PlanConfigMapLabelKey:       PlanConfigMapLabelKey,
		PlanConfigMapNamePrefix:     PlanConfigMapNamePrefix,
		PlansToRetain:               PlansToRetain,
		PlanCompletionCheckInterval: PlanCompletionCheckInterval.String(),
		PlanActivationTimeout:       PlanActivationTimeout.String(),
		WaitPodsGoneInterval:        WaitPodsGoneInterval.String(),
		EvictRecreateParallelism:    EvictRecreateParallelism,
	}
}

// persistPluginConfig writes the PluginConfigSnapshot to a ConfigMap in
// SystemNamespace using ConfigMapDoc.ensureJson.
func (pl *SharedState) persistPluginConfig(ctx context.Context) error {
	if pl == nil || pl.Client == nil {
		// In tests we often have a nil Client; treat as no-op.
		return nil
	}

	doc := ConfigMapDoc{
		Namespace: SystemNamespace,
		Name:      PluginCfgConfigMapName,
		LabelKey:  PluginCfgConfigMapLabelKey,
		DataKey:   PluginCfgConfigMapLabelKey + ".json",
	}

	snap := buildPluginConfigSnapshot()

	cms := pl.Client.CoreV1().ConfigMaps(SystemNamespace)

	if err := doc.ensureJson(ctx, cms, snap); err != nil {
		klog.ErrorS(err, "persistPluginConfig: failed to create/update plugin configuration ConfigMap",
			"namespace", doc.Namespace, "name", doc.Name)
		return err
	}

	klog.InfoS("persistPluginConfig: updated plugin configuration ConfigMap",
		"namespace", doc.Namespace, "name", doc.Name)
	return nil
}
