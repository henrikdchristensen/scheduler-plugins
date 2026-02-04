#!/usr/bin/env python3
# test_kwok_trace_replayer_seal_results.py

import pytest

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from scripts.kwok_trace_replayer import seal_results as sr

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, data: Dict[str, list]) -> None:
    """Write a simple CSV from dict of columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def _write_json(path: Path, data: dict) -> None:
    """Write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# arrival_str
# ---------------------------------------------------------------------------

class TestArrivalStr:
    def test_integer_value(self):
        assert sr.arrival_str(10.0) == "10"
        assert sr.arrival_str(1.0) == "1"
        assert sr.arrival_str(0.0) == "0"

    def test_float_value(self):
        assert sr.arrival_str(10.5) == "10.5"
        assert sr.arrival_str(0.25) == "0.25"

    def test_near_integer(self):
        # Very close to integer should round
        assert sr.arrival_str(10.0000000001) == "10"


# ---------------------------------------------------------------------------
# parse_default_run_dir
# ---------------------------------------------------------------------------

class TestParseDefaultRunDir:
    def test_valid_format(self):
        result = sr.parse_default_run_dir("nodes=16_prio=2_arrival=10s")
        assert result == "nodes=16_prio=2_arrival=10s"

    def test_valid_with_float_arrival(self):
        result = sr.parse_default_run_dir("nodes=32_prio=3_arrival=4.5s")
        assert result == "nodes=32_prio=3_arrival=4.5s"

    def test_invalid_format(self):
        assert sr.parse_default_run_dir("invalid") is None
        assert sr.parse_default_run_dir("") is None
        assert sr.parse_default_run_dir("nodes_prio_arrival") is None


# ---------------------------------------------------------------------------
# parse_plugin_run_dir
# ---------------------------------------------------------------------------

class TestParsePluginRunDir:
    def test_valid_format(self):
        result = sr.parse_plugin_run_dir("mode=schedulingfailure_blocking=0_defpreempt=0_nodes=32_prio=1_arrival=4s")
        assert result is not None
        job_name, plugin_config = result
        assert job_name == "nodes=32_prio=1_arrival=4s"
        assert plugin_config == "mode=schedulingfailure_blocking=0_defpreempt=0"

    def test_blocking_true(self):
        result = sr.parse_plugin_run_dir("mode=test_blocking=1_defpreempt=2_nodes=16_prio=2_arrival=10s")
        assert result is not None
        job_name, plugin_config = result
        assert job_name == "nodes=16_prio=2_arrival=10s"
        assert plugin_config == "mode=test_blocking=1_defpreempt=2"

    def test_invalid_format(self):
        assert sr.parse_plugin_run_dir("invalid") is None
        assert sr.parse_plugin_run_dir("") is None
        assert sr.parse_plugin_run_dir("mode=test_blocking=1") is None  # Missing nodes/prio/arrival


# ---------------------------------------------------------------------------
# iter_seed_dirs
# ---------------------------------------------------------------------------

class TestIterSeedDirs:
    def test_nonexistent_dir(self, tmp_path: Path):
        result = list(sr.iter_seed_dirs(tmp_path / "nonexistent"))
        assert result == []

    def test_empty_dir(self, tmp_path: Path):
        result = list(sr.iter_seed_dirs(tmp_path))
        assert result == []

    def test_valid_seed_dirs(self, tmp_path: Path):
        # Create seed directories with required files
        seed1 = tmp_path / "seed1"
        seed1.mkdir()
        (seed1 / sr.GENERAL_STATS_FILENAME).write_text("dummy")
        (seed1 / sr.POD_STATS_FILENAME).write_text("dummy")

        seed2 = tmp_path / "seed2"
        seed2.mkdir()
        (seed2 / sr.GENERAL_STATS_FILENAME).write_text("dummy")
        (seed2 / sr.POD_STATS_FILENAME).write_text("dummy")

        # Create dir without required files (should be skipped)
        seed3 = tmp_path / "seed3"
        seed3.mkdir()

        result = list(sr.iter_seed_dirs(tmp_path))
        assert len(result) == 2
        seeds = [s for s, _ in result]
        assert "seed1" in seeds
        assert "seed2" in seeds


# ---------------------------------------------------------------------------
# read_optimization_stats
# ---------------------------------------------------------------------------

class TestReadOptimizationStats:
    def test_missing_file(self, tmp_path: Path):
        result = sr.read_optimization_stats(tmp_path / "nonexistent.json")
        for key in sr.OPT_TOTAL_KEYS.values():
            assert math.isnan(result[key])

    def test_valid_file(self, tmp_path: Path):
        opt_json = tmp_path / "optimization_stats.json"
        _write_json(opt_json, {
            "solver_attempts_total": 10,
            "best_solver_optimal_total": 8,
            "best_solver_feasible_total": 1,
            "best_solver_failed_total": 1,
            "plan_not_applicable_total": 2,
            "plan_activated_total": 5,
        })

        result = sr.read_optimization_stats(opt_json)
        assert result["solver_attempts"] == 10.0
        assert result["solver_optimal"] == 8.0
        assert result["solver_feasible"] == 1.0
        assert result["solver_failed"] == 1.0
        assert result["plan_not_applicable"] == 2.0
        assert result["plan_activated"] == 5.0

    def test_partial_file(self, tmp_path: Path):
        opt_json = tmp_path / "optimization_stats.json"
        _write_json(opt_json, {
            "solver_attempts_total": 5,
        })

        result = sr.read_optimization_stats(opt_json)
        assert result["solver_attempts"] == 5.0
        assert math.isnan(result["solver_optimal"])


# ---------------------------------------------------------------------------
# read_pod
# ---------------------------------------------------------------------------

class TestReadPod:
    def test_valid_pod_csv(self, tmp_path: Path):
        pod_csv = tmp_path / "pod_stats.csv"
        _write_csv(pod_csv, {
            sr.POD_EVENT_COL: ["apply-time", "running-time"],
            sr.POD_NAME_COL: ["rs-000001-abc", "rs-000001-abc"],
            sr.POD_UID_COL: ["uid1", "uid1"],
            sr.POD_PRIO_COL: [1, 1],
            sr.POD_TIME_COL: [0.0, 0.5],
        })

        df = sr.read_pod(pod_csv)
        assert len(df) == 2
        assert sr.POD_EVENT_COL in df.columns
        assert sr.POD_NAME_COL in df.columns


# ---------------------------------------------------------------------------
# cumulative_integral_step
# ---------------------------------------------------------------------------

class TestCumulativeIntegralStep:
    def test_empty_array(self):
        t = np.array([])
        y = np.array([])
        result = sr.cumulative_integral_step(t, y)
        assert len(result) == 0

    def test_single_element(self):
        t = np.array([0.0])
        y = np.array([1.0])
        result = sr.cumulative_integral_step(t, y)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_simple_integral(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = sr.cumulative_integral_step(t, y)
        # cumulative[0] = 0
        # cumulative[1] = y[0] * (t[1] - t[0]) = 1 * 1 = 1
        # cumulative[2] = cumulative[1] + y[1] * (t[2] - t[1]) = 1 + 2 * 1 = 3
        # cumulative[3] = cumulative[2] + y[2] * (t[3] - t[2]) = 3 + 3 * 1 = 6
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 3.0, 6.0])

    def test_2d_array(self):
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = sr.cumulative_integral_step(t, y)
        assert result.shape == (3, 2)
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(result[1], [1.0, 2.0])
        np.testing.assert_array_almost_equal(result[2], [4.0, 6.0])


# ---------------------------------------------------------------------------
# mean_over_horizon
# ---------------------------------------------------------------------------

class TestMeanOverHorizon:
    def test_simple_mean(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        cum = sr.cumulative_integral_step(t, y)
        
        # Mean over [0, 3]: integral = 6, mean = 6/3 = 2
        result = sr.mean_over_horizon(t, y, cum, 3.0)
        assert abs(result - 2.0) < 1e-9

    def test_partial_horizon(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        cum = sr.cumulative_integral_step(t, y)
        
        # Mean over [0, 1.5]: integral at t=1 is 1, plus tail 2.0 * 0.5 = 1
        # total = 2, mean = 2 / 1.5 ≈ 1.333
        result = sr.mean_over_horizon(t, y, cum, 1.5)
        assert abs(result - 4/3) < 1e-9

    def test_horizon_beyond_data(self):
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        cum = sr.cumulative_integral_step(t, y)
        
        # T_end = 5.0, but t[-1] = 2.0, so Hc = 2.0
        result = sr.mean_over_horizon(t, y, cum, 5.0)
        # Mean over [0, 2]: integral = 0 + 1*1 + 2*1 = 3, mean = 3/2 = 1.5
        assert abs(result - 1.5) < 1e-9


# ---------------------------------------------------------------------------
# value_at_horizon
# ---------------------------------------------------------------------------

class TestValueAtHorizon:
    def test_exact_point(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        
        # At T_end = 1.0, we should get y[0] since searchsorted returns 1, idx = 0
        # Actually, searchsorted(t, 1.0, side='right') = 2, idx = 1
        result = sr.value_at_horizon(t, y, 1.0)
        assert result == 20.0

    def test_between_points(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        
        # At T_end = 1.5, searchsorted = 2, idx = 1
        result = sr.value_at_horizon(t, y, 1.5)
        assert result == 20.0

    def test_horizon_beyond_data(self):
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        
        # T_end = 5.0, Hc = min(5.0, 2.0) = 2.0
        result = sr.value_at_horizon(t, y, 5.0)
        assert result == 30.0


# ---------------------------------------------------------------------------
# read_general_data
# ---------------------------------------------------------------------------

class TestReadGeneralData:
    def test_valid_data(self, tmp_path: Path):
        csv_path = tmp_path / "general_stats.csv"
        _write_csv(csv_path, {
            sr.TIME_COL: [0.0, 1.0, 2.0],
            sr.CPU_RUN_COL: [0.1, 0.2, 0.3],
            sr.MEM_RUN_COL: [0.15, 0.25, 0.35],
            "running_p1": [5, 10, 15],
            "running_p2": [2, 4, 6],
            "running_p3": [0, 0, 0],
            "running_p4": [0, 0, 0],
            "deletions_cum_p1": [0, 1, 2],
            "deletions_cum_p2": [0, 0, 1],
            "deletions_cum_p3": [0, 0, 0],
            "deletions_cum_p4": [0, 0, 0],
        })

        data = sr.read_general_data(csv_path)
        assert data.T_end == 2.0
        assert len(data.t) == 3
        np.testing.assert_array_almost_equal(data.cpu_util, [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(data.mem_util, [0.15, 0.25, 0.35])
        np.testing.assert_array_almost_equal(data.eff_util, [0.15, 0.25, 0.35])  # max(cpu, mem)

    def test_empty_csv(self, tmp_path: Path):
        csv_path = tmp_path / "general_stats.csv"
        csv_path.write_text(f"{sr.TIME_COL},{sr.CPU_RUN_COL},{sr.MEM_RUN_COL}\n")

        data = sr.read_general_data(csv_path)
        assert len(data.t) == 0
        assert math.isnan(data.T_end)


# ---------------------------------------------------------------------------
# compute_horizon_metrics
# ---------------------------------------------------------------------------

class TestComputeHorizonMetrics:
    def test_computes_metrics(self, tmp_path: Path):
        csv_path = tmp_path / "general_stats.csv"
        _write_csv(csv_path, {
            sr.TIME_COL: [0.0, 1.0, 2.0],
            sr.CPU_RUN_COL: [0.1, 0.2, 0.3],
            sr.MEM_RUN_COL: [0.15, 0.25, 0.35],
            "running_p1": [10, 20, 30],
            "running_p2": [5, 10, 15],
            "running_p3": [0, 0, 0],
            "running_p4": [0, 0, 0],
            "deletions_cum_p1": [0, 1, 2],
            "deletions_cum_p2": [0, 0, 1],
            "deletions_cum_p3": [0, 0, 0],
            "deletions_cum_p4": [0, 0, 0],
        })

        data = sr.read_general_data(csv_path)
        metrics = sr.compute_horizon_metrics(data, T_end=2.0)

        assert "U_cpu_mean" in metrics
        assert "U_mem_mean" in metrics
        assert "U_eff_mean" in metrics
        assert "R_p1_mean" in metrics
        assert "D_p1" in metrics
        assert "R_total_mean" in metrics
        assert "D_total" in metrics

        # Check deletions at horizon
        assert metrics["D_p1"] == 2.0
        assert metrics["D_p2"] == 1.0
        assert metrics["D_total"] == 3.0


# ---------------------------------------------------------------------------
# latency_means_first_batch_ms
# ---------------------------------------------------------------------------

class TestLatencyMeansFirstBatchMs:
    def test_valid_latencies(self, tmp_path: Path):
        pod_csv = tmp_path / "pod_stats.csv"
        _write_csv(pod_csv, {
            sr.POD_EVENT_COL: ["apply-time", "running-time", "apply-time", "running-time"],
            sr.POD_NAME_COL: ["rs-000001-abc", "rs-000001-abc", "rs-000001-def", "rs-000001-def"],
            sr.POD_UID_COL: ["uid1", "uid1", "uid2", "uid2"],
            sr.POD_PRIO_COL: [1, 1, 2, 2],
            sr.POD_TIME_COL: [0.0, 0.1, 0.0, 0.2],
        })

        result = sr.latency_means_first_batch_ms(pod_csv, eps_s=1.0)
        
        # uid1: latency = 0.1 - 0.0 = 0.1s = 100ms
        # uid2: latency = 0.2 - 0.0 = 0.2s = 200ms
        assert result["L_ms_p1"] == 100.0
        assert result["L_ms_p2"] == 200.0
        assert result["L_ms_total"] == 150.0  # (100 + 200) / 2

    def test_empty_csv(self, tmp_path: Path):
        pod_csv = tmp_path / "pod_stats.csv"
        _write_csv(pod_csv, {
            sr.POD_EVENT_COL: [],
            sr.POD_NAME_COL: [],
            sr.POD_UID_COL: [],
            sr.POD_PRIO_COL: [],
            sr.POD_TIME_COL: [],
        })

        result = sr.latency_means_first_batch_ms(pod_csv, eps_s=1.0)
        assert math.isnan(result["L_ms_total"])
        for p in range(1, sr.MAX_PRIORITIES + 1):
            assert math.isnan(result[f"L_ms_p{p}"])

    def test_no_running_time(self, tmp_path: Path):
        pod_csv = tmp_path / "pod_stats.csv"
        _write_csv(pod_csv, {
            sr.POD_EVENT_COL: ["apply-time"],
            sr.POD_NAME_COL: ["rs-000001-abc"],
            sr.POD_UID_COL: ["uid1"],
            sr.POD_PRIO_COL: [1],
            sr.POD_TIME_COL: [0.0],
        })

        result = sr.latency_means_first_batch_ms(pod_csv, eps_s=1.0)
        # No running time, so latency should be NaN
        assert math.isnan(result["L_ms_total"])


# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

class TestConstants:
    def test_out_cols_defined(self):
        assert len(sr.OUT_COLS) > 0
        assert "job_name" in sr.OUT_COLS
        assert "plugin_config" in sr.OUT_COLS
        assert "seed" in sr.OUT_COLS

    def test_opt_total_keys_mapping(self):
        assert "solver_attempts_total" in sr.OPT_TOTAL_KEYS
        assert sr.OPT_TOTAL_KEYS["solver_attempts_total"] == "solver_attempts"

    def test_max_priorities(self):
        assert sr.MAX_PRIORITIES == 4

    def test_eps_s_default(self):
        assert sr.EPS_S == 1.0


# ---------------------------------------------------------------------------
# iter_seed_dirs - additional edge cases
# ---------------------------------------------------------------------------

class TestIterSeedDirsEdgeCases:
    def test_skips_files(self, tmp_path: Path):
        """Files in the directory should be skipped (line 160)."""
        # Create a file instead of directory
        (tmp_path / "not_a_dir.txt").write_text("dummy")
        
        # Create valid seed dir
        seed1 = tmp_path / "seed1"
        seed1.mkdir()
        (seed1 / sr.GENERAL_STATS_FILENAME).write_text("dummy")
        (seed1 / sr.POD_STATS_FILENAME).write_text("dummy")

        result = list(sr.iter_seed_dirs(tmp_path))
        assert len(result) == 1
        assert result[0][0] == "seed1"


# ---------------------------------------------------------------------------
# read_optimization_stats - edge cases
# ---------------------------------------------------------------------------

class TestReadOptimizationStatsEdgeCases:
    def test_invalid_value_type(self, tmp_path: Path):
        """Test that invalid value types result in NaN (line 181-182)."""
        opt_json = tmp_path / "optimization_stats.json"
        _write_json(opt_json, {
            "solver_attempts_total": "not_a_number",
            "best_solver_optimal_total": {"nested": "object"},
        })

        result = sr.read_optimization_stats(opt_json)
        assert math.isnan(result["solver_attempts"])
        assert math.isnan(result["solver_optimal"])


# ---------------------------------------------------------------------------
# read_general_data - edge cases
# ---------------------------------------------------------------------------

class TestReadGeneralDataEdgeCases:
    def test_invalid_time_values_filtered(self, tmp_path: Path):
        """Test that invalid time values are filtered out, leading to empty result (line 288)."""
        csv_path = tmp_path / "general_stats.csv"
        # All time values are NaN or negative after normalization
        _write_csv(csv_path, {
            sr.TIME_COL: [float("nan"), float("nan")],
            sr.CPU_RUN_COL: [0.1, 0.2],
            sr.MEM_RUN_COL: [0.15, 0.25],
        })

        data = sr.read_general_data(csv_path)
        assert len(data.t) == 0
        assert math.isnan(data.T_end)


# ---------------------------------------------------------------------------
# latency_means_first_batch_ms - edge cases
# ---------------------------------------------------------------------------

class TestLatencyMeansFirstBatchMsEdgeCases:
    def test_no_apply_time_events(self, tmp_path: Path):
        """Test when there are no apply-time events (line 367)."""
        pod_csv = tmp_path / "pod_stats.csv"
        _write_csv(pod_csv, {
            sr.POD_EVENT_COL: ["running-time", "running-time"],
            sr.POD_NAME_COL: ["rs-000001-abc", "rs-000001-def"],
            sr.POD_UID_COL: ["uid1", "uid2"],
            sr.POD_PRIO_COL: [1, 2],
            sr.POD_TIME_COL: [0.1, 0.2],
        })

        result = sr.latency_means_first_batch_ms(pod_csv, eps_s=1.0)
        assert math.isnan(result["L_ms_total"])

    def test_joined_empty_after_dropna(self, tmp_path: Path):
        """Test when joined dataframe is empty after dropna (line 385)."""
        pod_csv = tmp_path / "pod_stats.csv"
        # rs_prefix will be empty string for pod with no name pattern
        _write_csv(pod_csv, {
            sr.POD_EVENT_COL: ["apply-time", "running-time"],
            sr.POD_NAME_COL: [None, None],  # Will result in empty rs_prefix
            sr.POD_UID_COL: ["uid1", "uid1"],
            sr.POD_PRIO_COL: [float("nan"), float("nan")],  # NaN priority
            sr.POD_TIME_COL: [0.0, 0.1],
        })

        result = sr.latency_means_first_batch_ms(pod_csv, eps_s=1.0)
        assert math.isnan(result["L_ms_total"])

    def test_batch_empty_with_eps(self, tmp_path: Path):
        """Test when batch is empty due to eps_s filtering (line 392)."""
        pod_csv = tmp_path / "pod_stats.csv"
        # Two pods from same rs_prefix but second one arrives much later
        _write_csv(pod_csv, {
            sr.POD_EVENT_COL: ["apply-time", "running-time", "apply-time", "running-time"],
            sr.POD_NAME_COL: ["rs-000001-abc", "rs-000001-abc", "rs-000001-def", "rs-000001-def"],
            sr.POD_UID_COL: ["uid1", "uid1", "uid2", "uid2"],
            sr.POD_PRIO_COL: [1, 1, 1, 1],
            sr.POD_TIME_COL: [0.0, 0.1, 100.0, 100.1],  # Second pod at t=100
        })

        # With eps_s=0.0, only pods at exactly t0 should be included
        # But both pods are in the first batch since they each have their own rs_prefix... wait
        # Actually rs-000001-abc and rs-000001-def have same prefix "rs-000001"
        # So t0 for "rs-000001" = 0.0, and pod at 100.0 is outside eps_s=0.5
        result = sr.latency_means_first_batch_ms(pod_csv, eps_s=0.0)
        # Only uid1 should be in batch (at t=0.0)
        assert result["L_ms_p1"] == 100.0  # 0.1 - 0.0 = 0.1s = 100ms

    def test_multiple_priorities_with_gaps(self, tmp_path: Path):
        """Test latencies with gaps in priorities (some priorities have no pods)."""
        pod_csv = tmp_path / "pod_stats.csv"
        _write_csv(pod_csv, {
            sr.POD_EVENT_COL: ["apply-time", "running-time", "apply-time", "running-time"],
            sr.POD_NAME_COL: ["rs-000001-a", "rs-000001-a", "rs-000002-b", "rs-000002-b"],
            sr.POD_UID_COL: ["uid1", "uid1", "uid2", "uid2"],
            sr.POD_PRIO_COL: [1, 1, 4, 4],  # Only prio 1 and 4, skip 2 and 3
            sr.POD_TIME_COL: [0.0, 0.1, 0.0, 0.3],
        })

        result = sr.latency_means_first_batch_ms(pod_csv, eps_s=1.0)
        assert result["L_ms_p1"] == 100.0
        assert math.isnan(result["L_ms_p2"])
        assert math.isnan(result["L_ms_p3"])
        assert result["L_ms_p4"] == 300.0
        assert result["L_ms_total"] == 200.0  # (100 + 300) / 2


# ---------------------------------------------------------------------------
# Main function integration test (lines 414-537)
# ---------------------------------------------------------------------------

class TestMainIntegration:
    def test_main_creates_output(self, tmp_path: Path, monkeypatch):
        """Integration test for main() function."""
        # Setup directory structure
        default_root = tmp_path / "default"
        plugin_root = tmp_path / "plugin"
        out_dir = tmp_path

        # Create default run
        default_job = default_root / "nodes=4_prio=2_arrival=1s"
        default_seed = default_job / "seed1"
        default_seed.mkdir(parents=True)
        
        _write_csv(default_seed / sr.GENERAL_STATS_FILENAME, {
            sr.TIME_COL: [0.0, 1.0, 2.0],
            sr.CPU_RUN_COL: [0.1, 0.2, 0.3],
            sr.MEM_RUN_COL: [0.1, 0.2, 0.3],
            "running_p1": [5, 10, 15],
            "running_p2": [2, 4, 6],
            "running_p3": [0, 0, 0],
            "running_p4": [0, 0, 0],
            "deletions_cum_p1": [0, 1, 2],
            "deletions_cum_p2": [0, 0, 1],
            "deletions_cum_p3": [0, 0, 0],
            "deletions_cum_p4": [0, 0, 0],
        })
        _write_csv(default_seed / sr.POD_STATS_FILENAME, {
            sr.POD_EVENT_COL: ["apply-time", "running-time"],
            sr.POD_NAME_COL: ["rs-000001-abc", "rs-000001-abc"],
            sr.POD_UID_COL: ["uid1", "uid1"],
            sr.POD_PRIO_COL: [1, 1],
            sr.POD_TIME_COL: [0.0, 0.1],
        })

        # Create plugin run
        plugin_job = plugin_root / "mode=test_blocking=0_defpreempt=0_nodes=4_prio=2_arrival=1s"
        plugin_seed = plugin_job / "seed1"
        plugin_seed.mkdir(parents=True)
        
        _write_csv(plugin_seed / sr.GENERAL_STATS_FILENAME, {
            sr.TIME_COL: [0.0, 1.0, 2.0],
            sr.CPU_RUN_COL: [0.15, 0.25, 0.35],
            sr.MEM_RUN_COL: [0.15, 0.25, 0.35],
            "running_p1": [6, 12, 18],
            "running_p2": [3, 6, 9],
            "running_p3": [0, 0, 0],
            "running_p4": [0, 0, 0],
            "deletions_cum_p1": [0, 0, 1],
            "deletions_cum_p2": [0, 0, 0],
            "deletions_cum_p3": [0, 0, 0],
            "deletions_cum_p4": [0, 0, 0],
        })
        _write_csv(plugin_seed / sr.POD_STATS_FILENAME, {
            sr.POD_EVENT_COL: ["apply-time", "running-time"],
            sr.POD_NAME_COL: ["rs-000001-abc", "rs-000001-abc"],
            sr.POD_UID_COL: ["uid1", "uid1"],
            sr.POD_PRIO_COL: [1, 1],
            sr.POD_TIME_COL: [0.0, 0.05],
        })
        _write_json(plugin_seed / sr.OPT_STATS_FILENAME, {
            "solver_attempts_total": 5,
            "best_solver_optimal_total": 4,
            "best_solver_feasible_total": 1,
            "best_solver_failed_total": 0,
            "plan_not_applicable_total": 1,
            "plan_activated_total": 3,
        })

        # Monkeypatch the module constants
        monkeypatch.setattr(sr, "DEFAULT_ROOT", default_root)
        monkeypatch.setattr(sr, "PLUGIN_ROOT", plugin_root)
        monkeypatch.setattr(sr, "OUT_DIR", out_dir)

        # Run main
        sr.main()

        # Check output file was created
        out_path = out_dir / sr.OUT_FILENAME
        assert out_path.exists()

        # Read and validate output
        df = pd.read_csv(out_path)
        assert len(df) == 1
        assert df.loc[0, "job_name"] == "nodes=4_prio=2_arrival=1s"
        assert df.loc[0, "plugin_config"] == "mode=test_blocking=0_defpreempt=0"
        assert df.loc[0, "seed"] == "seed1"
        assert df.loc[0, "solver_attempts_mean"] == 5.0

    def test_main_no_common_seeds(self, tmp_path: Path, monkeypatch, capsys):
        """Test main() when there are no common seeds between default and plugin."""
        default_root = tmp_path / "default"
        plugin_root = tmp_path / "plugin"
        out_dir = tmp_path

        # Create default run with seed1
        default_job = default_root / "nodes=4_prio=2_arrival=1s"
        default_seed = default_job / "seed1"
        default_seed.mkdir(parents=True)
        _write_csv(default_seed / sr.GENERAL_STATS_FILENAME, {sr.TIME_COL: [0.0]})
        _write_csv(default_seed / sr.POD_STATS_FILENAME, {sr.POD_EVENT_COL: []})

        # Create plugin run with seed2 (different seed)
        plugin_job = plugin_root / "mode=test_blocking=0_defpreempt=0_nodes=4_prio=2_arrival=1s"
        plugin_seed = plugin_job / "seed2"
        plugin_seed.mkdir(parents=True)
        _write_csv(plugin_seed / sr.GENERAL_STATS_FILENAME, {sr.TIME_COL: [0.0]})
        _write_csv(plugin_seed / sr.POD_STATS_FILENAME, {sr.POD_EVENT_COL: []})

        monkeypatch.setattr(sr, "DEFAULT_ROOT", default_root)
        monkeypatch.setattr(sr, "PLUGIN_ROOT", plugin_root)
        monkeypatch.setattr(sr, "OUT_DIR", out_dir)

        sr.main()

        captured = capsys.readouterr()
        assert "no common seeds" in captured.out

    def test_main_no_default_seeds(self, tmp_path: Path, monkeypatch, capsys):
        """Test main() when there are no default seeds for a job."""
        default_root = tmp_path / "default"
        plugin_root = tmp_path / "plugin"
        out_dir = tmp_path

        default_root.mkdir(parents=True)

        # Create plugin run without matching default
        plugin_job = plugin_root / "mode=test_blocking=0_defpreempt=0_nodes=4_prio=2_arrival=1s"
        plugin_seed = plugin_job / "seed1"
        plugin_seed.mkdir(parents=True)
        _write_csv(plugin_seed / sr.GENERAL_STATS_FILENAME, {sr.TIME_COL: [0.0]})
        _write_csv(plugin_seed / sr.POD_STATS_FILENAME, {sr.POD_EVENT_COL: []})

        monkeypatch.setattr(sr, "DEFAULT_ROOT", default_root)
        monkeypatch.setattr(sr, "PLUGIN_ROOT", plugin_root)
        monkeypatch.setattr(sr, "OUT_DIR", out_dir)

        sr.main()

        captured = capsys.readouterr()
        assert "no default seeds" in captured.out

    def test_main_empty_output(self, tmp_path: Path, monkeypatch):
        """Test main() with empty directories produces empty output with correct schema."""
        default_root = tmp_path / "default"
        plugin_root = tmp_path / "plugin"
        out_dir = tmp_path

        default_root.mkdir(parents=True)
        plugin_root.mkdir(parents=True)

        monkeypatch.setattr(sr, "DEFAULT_ROOT", default_root)
        monkeypatch.setattr(sr, "PLUGIN_ROOT", plugin_root)
        monkeypatch.setattr(sr, "OUT_DIR", out_dir)

        sr.main()

        out_path = out_dir / sr.OUT_FILENAME
        assert out_path.exists()
        
        df = pd.read_csv(out_path)
        assert len(df) == 0
        # Check all columns exist
        for col in sr.OUT_COLS:
            assert col in df.columns

