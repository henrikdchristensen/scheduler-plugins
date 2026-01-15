#!/usr/bin/env python3
# test_kwok_workload_once.py

import pytest

import argparse, builtins, csv, json, random
from pathlib import Path

from scripts.test.test_utils import TimeController
from scripts.kwok_workload_once import test_runner as tr


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def touch_configs(tmp_path: Path) -> tuple[Path, Path]:
    wl = tmp_path / "wl.yaml"
    kw = tmp_path / "kwokctl.yaml"
    wl.write_text("namespace: ns\n", encoding="utf-8")
    kw.write_text("componentsPatches: []\n", encoding="utf-8")
    return wl, kw


def applied_cfg(
    *,
    namespace: str = "ns",
    num_nodes: int = 1,
    num_pods: int = 1,
    num_priorities: int = 1,
    node_cpu_m: int = 100,
    node_mem_b: int = 100,
    wait_pod_mode=None,
    wait_pod_timeout_s: int = 0,
    settle_timeout_min_s: int = 0,
    settle_timeout_max_s: int = 0,
) -> tr.TestConfigApplied:
    return tr.TestConfigApplied(
        namespace=namespace,
        num_nodes=num_nodes,
        num_pods=num_pods,
        num_priorities=num_priorities,
        num_replicaset=1,
        num_replicas_per_rs=(1, 1),
        node_cpu_m=node_cpu_m,
        node_mem_b=node_mem_b,
        cpu_per_pod_m=(10, 10),
        mem_per_pod_b=(10, 10),
        util=0.5,
        wait_pod_mode=wait_pod_mode,
        wait_pod_timeout_s=wait_pod_timeout_s,
        settle_timeout_min_s=settle_timeout_min_s,
        settle_timeout_max_s=settle_timeout_max_s,
    )


# ---------------------------------------------------------------------------
# build_argparser
# ---------------------------------------------------------------------------

def test_build_argparser_parses_minimal_seed_run():
    ap = tr.build_argparser()
    args = ap.parse_args(
        [
            "--workload-config-file",
            "wl.yaml",
            "--kwokctl-config-file",
            "kwokctl.yaml",
            "--seed",
            "1",
        ]
    )
    assert args.workload_config_file == "wl.yaml"
    assert args.kwokctl_config_file == "kwokctl.yaml"
    assert args.seed == 1


# ---------------------------------------------------------------------------
# TestRunner._initialize
# ---------------------------------------------------------------------------

def test_initialize_loads_job_file_and_configs_and_sets_state(monkeypatch, tmp_path):
    wl = tmp_path / "wl.yaml"
    wl.write_text(
        """\
apiVersion: v1
kind: WorkloadConfiguration
namespace: ns1
num_nodes: 1
num_pods: 1
util: 0.5
num_priorities: [1, 1]
num_replicas_per_rs: [1, 1]
cpu_per_pod: [100m, 100m]
mem_per_pod: [128Mi, 128Mi]
wait_pod_mode: none
""",
        encoding="utf-8",
    )

    kw = tmp_path / "kwokctl.yaml"
    kw.write_text(
        """\
apiVersion: config.kwok.x-k8s.io/v1alpha1
kind: KwokctlConfiguration
componentsPatches:
  - name: kube-scheduler
    extraEnvs: []
""",
        encoding="utf-8",
    )

    job = tmp_path / "job.yaml"
    job.write_text(
        """\
override-workload-config:
  namespace: ns2
override-kwokctl-envs:
  - name: X
    value: "1"
""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"

    monkeypatch.setattr(tr, "setup_logging", lambda **_k: None)
    monkeypatch.setattr(tr, "build_cli_cmd", lambda: "CMD")

    # Avoid depending on kwokctl env merge behavior
    def fake_merge_kwokctl_envs(doc, override_envs):
        doc = dict(doc)
        cps = list(doc.get("componentsPatches") or [])
        if not cps:
            cps = [{"name": "kube-scheduler", "extraEnvs": []}]
        cps0 = dict(cps[0])
        cps0["extraEnvs"] = list(override_envs or [])
        cps[0] = cps0
        doc["componentsPatches"] = cps
        return doc

    monkeypatch.setattr(tr, "merge_kwokctl_envs", fake_merge_kwokctl_envs)

    captured = {"info": None, "envs": None}
    monkeypatch.setattr(
        tr,
        "write_info_file",
        lambda out_path, meta_extra, inputs, logger: captured.__setitem__(
            "info",
            {"out_path": out_path, "meta_extra": meta_extra, "inputs": inputs},
        ),
    )
    monkeypatch.setattr(
        tr.TestRunner,
        "log_kwokctl_envs",
        staticmethod(lambda envs: captured.__setitem__("envs", envs)),
    )

    ap = tr.build_argparser()
    args = ap.parse_args(
        [
            "--job-file",
            str(job),
            "--workload-config-file",
            str(wl),
            "--kwokctl-config-file",
            str(kw),
            "--seed",
            "1",
            "--output-dir",
            str(out_dir),
            "--clean-start",
        ]
    )

    runner = tr.TestRunner(args)
    assert runner.workload_config is not None
    assert runner.workload_config.namespace == "ns2"
    assert runner.ctx == f"kwok-{runner.args.cluster_name}"

    assert captured["info"] is not None
    assert captured["info"]["inputs"]["cli-cmd"] == "CMD"
    assert captured["envs"] == {"X": "1"}


def test_initialize_missing_job_file_raises_system_exit(tmp_path):
    ap = tr.build_argparser()
    args = ap.parse_args(
        [
            "--job-file",
            str(tmp_path / "missing.yaml"),
            "--seed",
            "1",
            "--workload-config-file",
            str(tmp_path / "wl.yaml"),
            "--kwokctl-config-file",
            str(tmp_path / "kw.yaml"),
        ]
    )
    with pytest.raises(SystemExit):
        tr.TestRunner(args)


def test_initialize_job_file_must_be_mapping(tmp_path, monkeypatch):
    job = tmp_path / "job.yaml"
    job.write_text("- 1\n- 2\n", encoding="utf-8")

    monkeypatch.setattr(tr, "setup_logging", lambda **_k: None)

    ap = tr.build_argparser()
    args = ap.parse_args(
        [
            "--job-file",
            str(job),
            "--seed",
            "1",
            "--workload-config-file",
            str(tmp_path / "wl.yaml"),
            "--kwokctl-config-file",
            str(tmp_path / "kw.yaml"),
        ]
    )
    with pytest.raises(SystemExit):
        tr.TestRunner(args)


# ---------------------------------------------------------------------------
# TestRunner.ensure_default_args
# ---------------------------------------------------------------------------

def test_ensure_default_args_sets_defaults_and_validates_paths(tmp_path):
    wl, kw = touch_configs(tmp_path)

    args = ns(
        workload_config_file=str(wl),
        kwokctl_config_file=str(kw),
        seed=1,
        seed_file=None,
        count=None,
        seeds_not_all_running=0,
        # leave the rest unset on purpose
    )
    out = tr.TestRunner.ensure_default_args(args)

    assert out.cluster_name == "kwok1"
    assert out.kwok_runtime == "binary"
    assert out.output_dir == "./output"
    assert out.clean_start is False
    assert out.re_run_seeds is False
    assert out.pause is False
    assert out.log_level == "INFO"
    assert out.default_scheduler is False
    assert out.repeats == 1
    assert out.solver_timeout_ms == 10000


@pytest.mark.parametrize(
    "args_kwargs",
    [
        # seed + count is invalid
        dict(seed=1, seed_file=None, count=5, seeds_not_all_running=0),
        # seed + seed_file is invalid
        dict(seed=1, seed_file="SEEDS", count=None, seeds_not_all_running=0),
        # seed_file + count is invalid
        dict(seed=None, seed_file="SEEDS", count=1, seeds_not_all_running=0),
    ],
)
def test_ensure_default_args_rejects_incompatible_seed_args(tmp_path, args_kwargs):
    wl, kw = touch_configs(tmp_path)
    if args_kwargs.get("seed_file") == "SEEDS":
        seed_file = tmp_path / "seeds.txt"
        seed_file.write_text("1\n", encoding="utf-8")
        args_kwargs["seed_file"] = str(seed_file)

    args = ns(workload_config_file=str(wl), kwokctl_config_file=str(kw), **args_kwargs)
    with pytest.raises(SystemExit):
        tr.TestRunner.ensure_default_args(args)


def test_ensure_default_args_defaults_count_from_seeds_not_all_running(tmp_path):
    wl, kw = touch_configs(tmp_path)
    args = ns(
        workload_config_file=str(wl),
        kwokctl_config_file=str(kw),
        seed=None,
        seed_file=None,
        count=None,
        seeds_not_all_running=3,
    )
    out = tr.TestRunner.ensure_default_args(args)
    assert out.count == 3


def test_ensure_default_args_requires_a_run_mode(tmp_path):
    wl, kw = touch_configs(tmp_path)
    args = ns(
        workload_config_file=str(wl),
        kwokctl_config_file=str(kw),
        seed=None,
        seed_file=None,
        count=None,
        seeds_not_all_running=0,
    )
    with pytest.raises(SystemExit):
        tr.TestRunner.ensure_default_args(args)


def test_ensure_default_args_rejects_invalid_seed_and_count_bounds(tmp_path):
    wl, kw = touch_configs(tmp_path)

    bad_seed = ns(
        workload_config_file=str(wl),
        kwokctl_config_file=str(kw),
        seed=0,
        seed_file=None,
        count=None,
        seeds_not_all_running=0,
    )
    with pytest.raises(SystemExit):
        tr.TestRunner.ensure_default_args(bad_seed)

    bad_count = ns(
        workload_config_file=str(wl),
        kwokctl_config_file=str(kw),
        seed=None,
        seed_file=None,
        count=-2,
        seeds_not_all_running=0,
    )
    with pytest.raises(SystemExit):
        tr.TestRunner.ensure_default_args(bad_count)


def test_ensure_default_args_validates_seed_file_and_config_paths(tmp_path):
    wl, kw = touch_configs(tmp_path)

    args_missing_seedfile = ns(
        workload_config_file=str(wl),
        kwokctl_config_file=str(kw),
        seed=None,
        seed_file=str(tmp_path / "missing-seeds.txt"),
        count=None,
        seeds_not_all_running=0,
    )
    with pytest.raises(SystemExit):
        tr.TestRunner.ensure_default_args(args_missing_seedfile)

    args_missing_wl = ns(
        workload_config_file=str(tmp_path / "missing-wl.yaml"),
        kwokctl_config_file=str(kw),
        seed=1,
        seed_file=None,
        count=None,
        seeds_not_all_running=0,
    )
    with pytest.raises(SystemExit):
        tr.TestRunner.ensure_default_args(args_missing_wl)

    args_missing_kw = ns(
        workload_config_file=str(wl),
        kwokctl_config_file=str(tmp_path / "missing-kw.yaml"),
        seed=1,
        seed_file=None,
        count=None,
        seeds_not_all_running=0,
    )
    with pytest.raises(SystemExit):
        tr.TestRunner.ensure_default_args(args_missing_kw)


def test_ensure_default_args_default_scheduler_rejects_incompatible_flags(tmp_path):
    wl, kw = touch_configs(tmp_path)
    args = ns(
        workload_config_file=str(wl),
        kwokctl_config_file=str(kw),
        seed=1,
        seed_file=None,
        count=None,
        seeds_not_all_running=0,
        default_scheduler=True,
        solver_trigger=True,  # incompatible
    )
    with pytest.raises(SystemExit):
        tr.TestRunner.ensure_default_args(args)


def test_ensure_default_args_solver_trigger_requires_binary_runtime(tmp_path):
    wl, kw = touch_configs(tmp_path)
    args = ns(
        workload_config_file=str(wl),
        kwokctl_config_file=str(kw),
        seed=1,
        seed_file=None,
        count=None,
        seeds_not_all_running=0,
        default_scheduler=False,
        solver_trigger=True,
        kwok_runtime="docker",
    )
    with pytest.raises(SystemExit):
        tr.TestRunner.ensure_default_args(args)


# ---------------------------------------------------------------------------
# TestRunner._read_seeds_file
# ---------------------------------------------------------------------------

def test_read_seeds_file(tmp_path):
    p = tmp_path / "seeds.txt"
    p.write_text(
        """
# comment
1
2
2
nope
0
-5
3
""",
        encoding="utf-8",
    )
    seeds = tr.TestRunner._read_seeds_file(p)
    assert seeds == [1, 2, 3]


def test_read_seeds_file_missing_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        tr.TestRunner._read_seeds_file(tmp_path / "missing.txt")


# ---------------------------------------------------------------------------
# TestRunner._parse_config_doc
# ---------------------------------------------------------------------------

def test_parse_config_doc():
    runner = tr.TestRunner(ns(), initialize=False)
    base = {
        "namespace": "ns1",
        "num_nodes": 2,
        "num_pods": 5,
        "util": 0.5,
        "wait_pod_mode": "none",
        "num_priorities": [1, 3],
        "cpu_per_pod": ["100m", "200m"],
        "mem_per_pod": ["128Mi", "256Mi"],
        "num_replicas_per_rs": [1, 3],
    }
    override = {"namespace": "ns2", "num_nodes": 3, "wait_pod_mode": "ready"}
    cfg = runner._parse_config_doc(base, override=override)
    assert cfg.namespace == "ns2"
    assert cfg.num_nodes == 3
    assert cfg.wait_pod_mode == "ready"


# ---------------------------------------------------------------------------
# TestRunner._validate_workload_config
# ---------------------------------------------------------------------------

def test_validate_workload_config_reports_multiple_errors():
    bad = tr.TestConfigRaw(
        namespace="",
        num_nodes=0,
        num_pods=0,
        num_priorities=None,
        num_replicas_per_rs=None,
        cpu_per_pod=None,
        mem_per_pod=None,
        util=0.0,
    )
    ok, msg = tr.TestRunner._validate_workload_config(bad)
    assert ok is False
    assert "namespace" in msg
    assert "num_nodes" in msg
    assert "num_pods" in msg
    assert "util" in msg


def test_validate_workload_config_ok():
    good = tr.TestConfigRaw(
        namespace="ns",
        num_nodes=2,
        num_pods=5,
        num_priorities=(1, 2),
        num_replicas_per_rs=(1, 3),
        cpu_per_pod=("100m", "200m"),
        mem_per_pod=("128Mi", "256Mi"),
        util=0.5,
    )
    ok, msg = tr.TestRunner._validate_workload_config(good)
    assert ok is True
    assert msg == ""


# ---------------------------------------------------------------------------
# TestRunner._resolve_config_for_seed
# ---------------------------------------------------------------------------

def test_resolve_config_for_seed():
    runner = tr.TestRunner(ns(), initialize=False)
    runner.workload_config = tr.TestConfigRaw(
        namespace="ns",
        num_nodes=1,
        num_pods=1,
        num_priorities=(1, 1),
        num_replicas_per_rs=None,
        cpu_per_pod=("100m", "100m"),
        mem_per_pod=("128Mi", "128Mi"),
        util=0.5,
        wait_pod_mode="none",
    )
    with pytest.raises(SystemExit):
        runner._resolve_config_for_seed(1)


# ---------------------------------------------------------------------------
# TestRunner._write_info_file
# ---------------------------------------------------------------------------

def test_write_info_file_calls_write_info_file(monkeypatch, tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.output_dir_resolved = tmp_path
    runner.args = ns(
        job_file="job.yaml",
        workload_config_file="wl.yaml",
        kwokctl_config_file="kwokctl.yaml",
        seed_file=None,
    )
    runner.job_doc = {"a": 1}
    runner.workload_config_doc = {"b": 2}
    runner.kwokctl_config_doc = {"c": 3}

    seen = {}

    def fake_write_info_file(out_path, meta_extra, inputs, logger):
        seen["out_path"] = out_path
        seen["meta_extra"] = meta_extra
        seen["inputs"] = inputs
        seen["logger"] = logger

    monkeypatch.setattr(tr, "write_info_file", fake_write_info_file)
    monkeypatch.setattr(tr, "build_cli_cmd", lambda: "CMD")

    runner._write_info_file()
    assert Path(seen["out_path"]).name == "info.yaml"
    assert seen["meta_extra"]["job_file"] == "job.yaml"
    assert seen["inputs"]["cli-cmd"] == "CMD"
    assert seen["inputs"]["job"] == {"a": 1}


def test_write_info_file_logs_warning_on_exception(monkeypatch, tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.output_dir_resolved = tmp_path
    runner.args = ns(job_file=None, workload_config_file="wl.yaml", kwokctl_config_file="kwokctl.yaml", seed_file=None)
    runner.job_doc = {}
    runner.workload_config_doc = {}
    runner.kwokctl_config_doc = {}

    monkeypatch.setattr(tr, "build_cli_cmd", lambda: "CMD")
    monkeypatch.setattr(tr, "write_info_file", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))

    seen = {"warn": 0}
    monkeypatch.setattr(tr.LOG, "warning", lambda *_a, **_k: seen.__setitem__("warn", seen["warn"] + 1))

    runner._write_info_file()
    assert seen["warn"] == 1


# ---------------------------------------------------------------------------
# TestRunner._combined_job_configs_seed_str
# ---------------------------------------------------------------------------

def test_combined_job_configs_seed_str():
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(
        job_file="job.yaml",
        workload_config_file="wl.yaml",
        kwokctl_config_file="kwok.yaml",
        seed_file="seeds.txt",
    )
    s = runner._combined_job_configs_seed_str()
    assert "job_file=job.yaml" in s
    assert "workload_config_file=wl.yaml" in s
    assert "kwokctl_config_file=kwok.yaml" in s
    assert "seed_file=seeds.txt" in s


# ---------------------------------------------------------------------------
# TestRunner._record_failure
# ---------------------------------------------------------------------------

def test_record_failure(tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.suppress_fail_log = True
    runner.failure = None
    runner.failed_f = tmp_path / "failed.tsv"

    runner._record_failure("cat", 7, "phase", "msg", details="d")
    assert runner.failure is not None
    assert runner.failure.category == "cat"
    assert runner.failure.seed == 7
    assert runner.failure.phase == "phase"
    assert runner.failure.message == "msg"
    assert runner.failure.details == "d"
    assert not runner.failed_f.exists()


# ---------------------------------------------------------------------------
# TestRunner._eta_record_seed_duration
# ---------------------------------------------------------------------------

def test_eta_record_seed_duration(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.seed_durations = []
    monkeypatch.setattr(tr.time, "time", lambda: 100.0)
    runner._eta_record_seed_duration(started_at=90.0)
    assert runner.seed_durations == [10.0]


# ---------------------------------------------------------------------------
# TestRunner._eta_estimation
# ---------------------------------------------------------------------------

def test_eta_estimation_returns_none_when_unknown_or_no_samples():
    runner = tr.TestRunner(ns(), initialize=False)
    runner.seed_durations = []
    assert runner._eta_estimation(1, 10) is None

    runner.seed_durations = [1.0]
    assert runner._eta_estimation(1, -1) is None
    assert runner._eta_estimation(1, 0) is None


def test_eta_estimation_returns_epoch(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.seed_durations = [2.0, 4.0]  # avg = 3s
    monkeypatch.setattr(tr.time, "time", lambda: 100.0)
    # seed_idx=3 => done=2 => left=8 (for total=10) => eta=100+8*3
    assert runner._eta_estimation(3, 10) == 124.0


# ---------------------------------------------------------------------------
# TestRunner._eta_write_file
# ---------------------------------------------------------------------------

def test_eta_write_file(tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    out = tmp_path / "out"
    out.mkdir()

    # stale markers
    (out / "eta_old_1").write_text("x", encoding="utf-8")
    (out / "eta_old_2").write_text("y", encoding="utf-8")

    runner.output_dir_resolved = out
    runner.args = ns(seeds_not_all_running=5)
    runner.saved_not_all_running = 2  # left = 3

    runner._eta_write_file(eta_epoch=1234567890.0, seed_idx=3, seeds_total=10)

    # old markers removed, exactly one new marker should exist
    markers = sorted(out.glob("eta_*"))
    assert len(markers) == 1
    p = markers[0]
    assert "seeds-3-of-10" in p.name
    assert "_snar-3" in p.name

    payload = json.loads(p.read_text(encoding="utf-8").strip())
    assert payload["seeds_at"] == 3
    assert payload["seeds_total"] == 10
    assert payload["snar_total"] == 5
    assert payload["snar_left"] == 3

    # also cover eta_epoch=None + unknown totals
    runner._eta_write_file(eta_epoch=None, seed_idx=7, seeds_total=-1)
    markers2 = sorted(out.glob("eta_*"))
    assert len(markers2) == 1
    assert "eta-unknown" in markers2[0].name
    payload2 = json.loads(markers2[0].read_text(encoding="utf-8").strip())
    assert payload2["eta_epoch"] is None
    assert payload2["seeds_total"] == -1


# ---------------------------------------------------------------------------
# TestRunner._eta_summary
# ---------------------------------------------------------------------------

def test_eta_summary(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(seeds_not_all_running=0)
    runner.seed_durations = []

    # unknown/infinite
    runner._eta_summary(next_seed_idx=1, seeds_total=-1)

    # valid totals but no durations
    runner._eta_summary(next_seed_idx=1, seeds_total=10)

    # force "eta_epoch is None" branch
    runner.seed_durations = [1.0]
    runner.args = ns(seeds_not_all_running=2)
    runner.saved_not_all_running = 1
    monkeypatch.setattr(runner, "_eta_estimation", lambda *_a, **_k: None)
    runner._eta_summary(next_seed_idx=2, seeds_total=10)


# ---------------------------------------------------------------------------
# TestRunner._eta_update_marker
# ---------------------------------------------------------------------------

def test_eta_update_marker(monkeypatch, tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.output_dir_resolved = tmp_path
    runner.args = ns(seeds_not_all_running=0)
    called = {"n": 0, "got": None}

    monkeypatch.setattr(runner, "_eta_estimation", lambda *_a, **_k: 111.0)

    def fake_write(eta_epoch, seed_idx, seeds_total):
        called["n"] += 1
        called["got"] = (eta_epoch, seed_idx, seeds_total)

    monkeypatch.setattr(runner, "_eta_write_file", fake_write)
    runner._eta_update_marker(seed_idx=2, seeds_total=5)

    assert called["n"] == 1
    assert called["got"] == (111.0, 2, 5)


# ---------------------------------------------------------------------------
# TestRunner._parse_waits
# ---------------------------------------------------------------------------

def test_parse_waits():
    runner = tr.TestRunner(ns(), initialize=False)
    raw = tr.TestConfigRaw(
        wait_pod_mode="none",
        wait_pod_timeout=None,
        settle_timeout_min=None,
        settle_timeout_max=None,
    )
    mode, pod_to, settle_min, settle_max = runner._parse_waits(raw)
    assert mode is None
    assert pod_to == 3
    assert settle_min == 2
    assert settle_max == 0


# ---------------------------------------------------------------------------
# TestRunner._get_wait_pod_mode_from_dict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        ("none", None),
        ("exist", "exist"),
        ("ready", "ready"),
        ("running", "running"),
    ],
)
def test_get_wait_pod_mode_from_dict(raw, expected):
    doc = {"wait_pod_mode": raw}
    if expected is None:
        assert tr.TestRunner._get_wait_pod_mode_from_dict(doc, "wait_pod_mode", None) is None
    else:
        assert tr.TestRunner._get_wait_pod_mode_from_dict(doc, "wait_pod_mode", None) == expected


def test_get_wait_pod_mode_from_dict_invalid_raises():
    with pytest.raises(ValueError):
        tr.TestRunner._get_wait_pod_mode_from_dict({"wait_pod_mode": "bogus"}, "wait_pod_mode", None)


# ---------------------------------------------------------------------------
# TestRunner.merge_job_fields_into_args
# ---------------------------------------------------------------------------

def test_merge_job_fields_into_args():
    args = ns(
        cluster_name="cli",
        kwok_runtime=None,
        workload_config_file=None,
        kwokctl_config_file=None,
        output_dir=None,
        clean_start=None,
        re_run_seeds=None,
        log_level=None,
        default_scheduler=None,
        seed=None,
        seed_file=None,
        count=None,
        repeats=None,
        seeds_not_all_running=None,
        save_solver_stats=None,
        save_scheduler_logs=None,
        solver_trigger=None,
    )
    job = {
        "cluster-name": "job",
        "kwok-runtime": "docker",
        "workload-config-file": "w.yaml",
        "kwokctl-config-file": "k.yaml",
        "output-dir": "./out",
        "clean-start": True,
        "re-run-seeds": True,
        "log-level": "DEBUG",
        "default-scheduler": False,
        "seed": 7,
        "count": 9,
        "repeats": 2,
        "seeds-not-all-running": 1,
        "save-solver-stats": True,
        "save-scheduler-logs": True,
        "solver-trigger": True,
        "override-workload-config": {"namespace": "ns"},
        "override-kwokctl-envs": [{"name": "X", "value": "1"}],
    }
    merged, overrides = tr.TestRunner.merge_job_fields_into_args(args, job)
    assert merged.cluster_name == "cli"  # CLI already set -> preserved
    assert merged.kwok_runtime == "docker"
    assert merged.workload_config_file == "w.yaml"
    assert merged.kwokctl_config_file == "k.yaml"
    assert merged.output_dir == "./out"
    assert merged.seed == 7
    assert merged.count == 9
    assert overrides["workload_config"] == {"namespace": "ns"}
    assert overrides["kwokctl_envs"] == [{"name": "X", "value": "1"}]


# ---------------------------------------------------------------------------
# TestRunner._get_kwokctl_envs
# ---------------------------------------------------------------------------

def test_get_kwokctl_envs():
    doc = {
        "componentsPatches": [
            {
                "name": "kube-scheduler",
                "extraEnvs": [
                    {"name": "B", "value": "2"},
                    {"name": "A", "value": "1"},
                    {"name": "EMPTY", "value": ""},
                    {"name": "NOVAL"},
                ],
            }
        ]
    }
    envs = tr.TestRunner._get_kwokctl_envs(doc, component="kube-scheduler")
    assert envs == {"A": "1", "B": "2", "EMPTY": None, "NOVAL": None}


# ---------------------------------------------------------------------------
# TestRunner._record_seed_outcome
# ---------------------------------------------------------------------------

def test_record_seed_outcome(tmp_path, monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.seeds_all_running_f = tmp_path / "all.txt"
    runner.seeds_not_all_running_f = tmp_path / "notall.txt"

    runner._record_seed_outcome(seed=7, all_running=True)
    assert runner.seeds_all_running_f.read_text(encoding="utf-8") == "7\n"

    warned = {"n": 0}
    monkeypatch.setattr(tr.LOG, "warning", lambda *_a, **_k: warned.__setitem__("n", warned["n"] + 1))

    real_open = builtins.open

    def bad_open(path, mode="r", *a, **k):
        if str(path).endswith("notall.txt") and mode.startswith("a"):
            raise OSError("boom")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", bad_open)

    runner._record_seed_outcome(seed=8, all_running=False)
    assert warned["n"] >= 1


# ---------------------------------------------------------------------------
# TestRunner._prepare_output_dir
# ---------------------------------------------------------------------------

def test_prepare_output_dir(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "old.txt").write_text("x", encoding="utf-8")

    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(output_dir=str(out_dir), clean_start=True)
    resolved = runner._prepare_output_dir()

    assert resolved.exists()
    assert resolved.is_dir()
    assert not (resolved / "old.txt").exists()


# ---------------------------------------------------------------------------
# TestRunner._append_result_csv
# ---------------------------------------------------------------------------

def test_append_result_csv_rerun_seeds_removes_old_rows(tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(clean_start=False, re_run_seeds=True)
    runner.results_f = tmp_path / "results.csv"

    with open(runner.results_f, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=tr.RESULTS_HEADER)
        w.writeheader()
        w.writerow({"timestamp": "t1", "seed": "5"})
        w.writerow({"timestamp": "t2", "seed": "6"})

    runner._append_result_csv({"timestamp": "t3", "seed": "5"})
    with open(runner.results_f, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    seeds = [r.get("seed") for r in rows]
    assert seeds.count("5") == 1
    assert "6" in seeds


def test_append_result_csv_prune_failure_logs_warning(tmp_path, monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(clean_start=False, re_run_seeds=True)
    runner.results_f = tmp_path / "results.csv"

    # Make a valid header so _purge_mismatched_results_csv doesn't try to delete.
    runner.results_f.write_text(",".join(tr.RESULTS_HEADER) + "\n", encoding="utf-8")

    # Avoid touching filesystem
    monkeypatch.setattr(tr, "csv_read_header", lambda *_a, **_k: [c.strip() for c in tr.RESULTS_HEADER])

    appended = {"n": 0}
    monkeypatch.setattr(tr, "csv_append_row", lambda *_a, **_k: appended.__setitem__("n", appended["n"] + 1))

    warned = {"n": 0}
    monkeypatch.setattr(tr.LOG, "warning", lambda *_a, **_k: warned.__setitem__("n", warned["n"] + 1))

    # Force the prune block to throw
    real_open = builtins.open

    def bad_open(path, mode="r", *a, **k):
        if str(path) == str(runner.results_f) and mode.startswith("r"):
            raise OSError("boom")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", bad_open)

    runner._append_result_csv({"timestamp": "t", "seed": "1"})

    assert warned["n"] >= 1
    assert appended["n"] == 1


# ---------------------------------------------------------------------------
# TestRunner._purge_mismatched_results_csv
# ---------------------------------------------------------------------------

def test_purge_mismatched_results_csv_deletes_when_clean_start(tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(clean_start=True)

    p = tmp_path / "results.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")

    deleted = runner._purge_mismatched_results_csv(p, expected_header=["x", "y"])
    assert deleted is True
    assert not p.exists()


def test_purge_mismatched_results_csv_does_not_delete_without_clean_start(tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(clean_start=False)

    p = tmp_path / "results.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")

    deleted = runner._purge_mismatched_results_csv(p, expected_header=["x", "y"])
    assert deleted is False
    assert p.exists()


# ---------------------------------------------------------------------------
# TestRunner._load_seen_results_csv
# ---------------------------------------------------------------------------

def test_load_seen_results_csv(tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.results_f = tmp_path / "results.csv"
    runner.results_f.write_text("seed\nnotint\n", encoding="utf-8")
    seen = runner._load_seen_results_csv()
    assert seen == set()


# ---------------------------------------------------------------------------
# TestRunner._extract_best_attempt_fields
# ---------------------------------------------------------------------------

def test_extract_best_attempt_fields_extracts_values():
    attempts = [
        {"name": "a", "score": {"x": 1}, "duration_ms": "12", "status": "OK"},
        {"name": "b", "score": {"y": 2}, "duration_ms": 7, "status": "FAIL"},
    ]
    score, dur, status = tr.TestRunner._extract_best_attempt_fields("b", attempts)
    assert score == json.dumps({"y": 2}, separators=(",", ":"), sort_keys=True)
    assert dur == 7
    assert status == "FAIL"


def test_extract_best_attempt_fields_handles_missing_or_invalid():
    # invalid attempts
    score, dur, status = tr.TestRunner._extract_best_attempt_fields("b", attempts=[])
    assert (score, dur, status) == (None, None, "")

    # best not found
    score, dur, status = tr.TestRunner._extract_best_attempt_fields("nope", attempts=[{"name": "a"}])
    assert (score, dur, status) == (None, None, "")

    # invalid duration_ms
    attempts = [{"name": "b", "score": object(), "duration_ms": "nope", "status": None}]
    score, dur, status = tr.TestRunner._extract_best_attempt_fields("b", attempts)
    assert dur is None
    assert status == ""


# ---------------------------------------------------------------------------
# TestRunner._get_solver_attempts
# ---------------------------------------------------------------------------

def test_get_solver_attempts_prefers_last_solver_result():
    runner = tr.TestRunner(ns(), initialize=False)
    runner.last_solver_result = {
        "error": "",
        "best_name": "b",
        "attempts": [{"name": "b", "score": 1}],
        "baseline": {"k": "v"},
    }
    baseline, best_name, attempts, error = runner._get_solver_attempts()
    assert baseline == {"k": "v"}
    assert best_name == "b"
    assert attempts == [{"name": "b", "score": 1}]
    assert error == ""


def test_get_solver_attempts_fallback_to_configmap_parses_latest(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.ctx = "ctx"
    runner.last_solver_result = None

    cm = {
        "data": {
            "runs.json": json.dumps(
                [
                    {"best_name": "a", "attempts": [{"name": "a", "score": 0}], "baseline": {"b": 0}},
                    {"best_name": "b", "attempts": [{"name": "b", "score": 1}], "baseline": {"b": 1}},
                ]
            )
        }
    }
    monkeypatch.setattr(runner, "_get_latest_configmap", lambda *_a, **_k: cm)

    baseline, best_name, attempts, error = runner._get_solver_attempts()
    assert baseline == {"b": 1}
    assert best_name == "b"
    assert attempts == [{"name": "b", "score": 1}]
    assert error == ""


def test_get_solver_attempts_fallback_handles_cm_missing_or_bad_json(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.ctx = "ctx"
    runner.last_solver_result = None

    # no cm
    monkeypatch.setattr(runner, "_get_latest_configmap", lambda *_a, **_k: None)
    baseline, best_name, attempts, error = runner._get_solver_attempts()
    assert (baseline, best_name, attempts, error) == ({}, "", [], "")

    # bad json
    monkeypatch.setattr(runner, "_get_latest_configmap", lambda *_a, **_k: {"data": {"runs.json": "not-json"}})
    baseline, best_name, attempts, error = runner._get_solver_attempts()
    assert (baseline, best_name, attempts, error) == ({}, "", [], "")

    # empty list
    monkeypatch.setattr(runner, "_get_latest_configmap", lambda *_a, **_k: {"data": {"runs.json": "[]"}})
    baseline, best_name, attempts, error = runner._get_solver_attempts()
    assert (baseline, best_name, attempts, error) == ({}, "", [], "")


# ---------------------------------------------------------------------------
# TestRunner._write_solver_stats_json
# ---------------------------------------------------------------------------

def test_write_solver_stats_json_writes_runs_raw(tmp_path, monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.ctx = "ctx"
    runner.solver_stats_dir = tmp_path / "solver-stats"

    runs_raw = '[{"best_name":"x"}]'
    cm = {"data": {"runs.json": runs_raw}}
    monkeypatch.setattr(runner, "_get_latest_configmap", lambda *a, **k: cm)

    runner._write_solver_stats_json(seed=7, run_idx=2)
    out = runner.solver_stats_dir / "solver_stats_seed-7_run-2.json"
    assert out.exists()
    assert out.read_text(encoding="utf-8") == runs_raw


def test_write_solver_stats_json_skips_when_no_cm_or_missing_runs(tmp_path, monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.ctx = "ctx"
    runner.solver_stats_dir = tmp_path / "solver-stats"

    warned = {"n": 0}
    monkeypatch.setattr(tr.LOG, "warning", lambda *_a, **_k: warned.__setitem__("n", warned["n"] + 1))

    # no cm
    monkeypatch.setattr(runner, "_get_latest_configmap", lambda *_a, **_k: None)
    runner._write_solver_stats_json(seed=1, run_idx=1)
    assert warned["n"] >= 1

    # missing runs.json
    monkeypatch.setattr(runner, "_get_latest_configmap", lambda *_a, **_k: {"data": {}})
    runner._write_solver_stats_json(seed=1, run_idx=1)
    assert warned["n"] >= 2

    # write error
    cm = {"data": {"runs.json": "[]"}}
    monkeypatch.setattr(runner, "_get_latest_configmap", lambda *_a, **_k: cm)
    real_open = builtins.open

    def bad_open(path, mode="r", *a, **k):
        if str(path).endswith(".json") and mode.startswith("w"):
            raise OSError("nope")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", bad_open)
    runner._write_solver_stats_json(seed=2, run_idx=3)
    assert warned["n"] >= 3


# ---------------------------------------------------------------------------
# TestRunner._get_latest_configmap
# ---------------------------------------------------------------------------

def test_get_latest_configmap_retries_then_returns_latest():
    # First call fails, second returns list with two matching items; should pick newest.
    calls = {"n": 0}

    class R:
        def __init__(self, rc, out=b"{}"):
            self.returncode = rc
            self.stdout = out
            self.stderr = b""

    items = {
        "items": [
            {"metadata": {"name": "solver-stats-aaa", "creationTimestamp": "2020-01-01T00:00:00Z", "resourceVersion": "1"}},
            {"metadata": {"name": "solver-stats-bbb", "creationTimestamp": "2021-01-01T00:00:00Z", "resourceVersion": "2"}},
            {"metadata": {"name": "other", "creationTimestamp": "2022-01-01T00:00:00Z", "resourceVersion": "9"}},
        ]
    }

    def fake_run(args, stdout, stderr, check):
        calls["n"] += 1
        if calls["n"] == 1:
            return R(1)
        return R(0, json.dumps(items).encode("utf-8"))

    class Clock:
        def __init__(self):
            self.slept = 0

        def sleep(self, _s):
            self.slept += 1

    clk = Clock()
    cm = tr.TestRunner._get_latest_configmap(
        "ctx",
        "ns",
        "solver-stats",
        retries=1,
        sleep_seconds=0.0,
        runner=fake_run,
        clock=clk,
    )
    assert cm is not None
    assert cm["metadata"]["name"] == "solver-stats-bbb"
    assert clk.slept == 1


def test_get_latest_configmap_label_selector_accepts_any(monkeypatch):
    class R:
        def __init__(self, rc, out=b"{}"):
            self.returncode = rc
            self.stdout = out
            self.stderr = b""

    items = {
        "items": [
            {"metadata": {"name": "x", "creationTimestamp": "2020-01-01T00:00:00Z", "resourceVersion": "1"}},
            {"metadata": {"name": "y", "creationTimestamp": "2021-01-01T00:00:00Z", "resourceVersion": "2"}},
        ]
    }

    def run_ok(*_a, **_k):
        return R(0, json.dumps(items).encode("utf-8"))

    cm = tr.TestRunner._get_latest_configmap(
        "ctx",
        "ns",
        "base",
        label_selector="app=solver",
        accept_prefix=True,
        retries=0,
        sleep_seconds=0.0,
        runner=run_ok,
        clock=TimeController(),
    )
    assert cm is not None
    assert cm["metadata"]["name"] == "y"

    cm2 = tr.TestRunner._get_latest_configmap(
        "ctx",
        "ns",
        "base",
        label_selector=None,
        accept_prefix=False,
        retries=0,
        sleep_seconds=0.0,
        runner=run_ok,
        clock=TimeController(),
    )
    assert cm2 is None

    # bad json => None
    def run_bad_json(*_a, **_k):
        return R(0, b"{not-json")

    cm3 = tr.TestRunner._get_latest_configmap(
        "ctx",
        "ns",
        "base",
        retries=0,
        sleep_seconds=0.0,
        runner=run_bad_json,
        clock=TimeController(),
    )
    assert cm3 is None


# ---------------------------------------------------------------------------
# TestRunner._save_scheduler_logs
# ---------------------------------------------------------------------------

def test_save_scheduler_logs(tmp_path, monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.scheduler_logs_dir = tmp_path / "scheduler-logs"
    runner.args = ns(cluster_name="kwok1")

    seen = {}

    def fake_save(cluster_name, out_path, runner, logger):
        seen["cluster_name"] = cluster_name
        seen["out_path"] = out_path

    monkeypatch.setattr(tr, "save_kwok_scheduler_logs", fake_save)

    runner._save_scheduler_logs(seed=3, run_idx=1)
    assert seen["cluster_name"] == "kwok1"
    assert Path(seen["out_path"]).name == "sched_logs_seed-3_run-1.log"


# ---------------------------------------------------------------------------
# TestRunner._build_pod_list
# ---------------------------------------------------------------------------

def test_build_pod_list():
    running_by_name = {"rs-01-p2-abc": "n1"}
    unsched = ["rs-01-p2-def", "other"]
    rs_specs = [{"name": "rs-01-p2", "req_cpu_m": 100, "req_mem_bytes": 200, "priority": 2}]

    items = tr.TestRunner._build_pod_list(running_by_name, unsched, rs_specs)
    by_name = {p["name"]: p for p in items}

    assert by_name["rs-01-p2-abc"]["node"] == "n1"
    assert by_name["rs-01-p2-abc"]["cpu_m"] == 100
    assert by_name["rs-01-p2-def"]["cpu_m"] == 100
    assert by_name["other"]["cpu_m"] == 0


# ---------------------------------------------------------------------------
# TestRunner._make_replicaset_specs_only
# ---------------------------------------------------------------------------

def test_make_replicaset_specs_only_builds_specs_and_rejects_invalid():
    runner = tr.TestRunner(ns(), initialize=False)

    class TA:
        pass

    ta = TA()
    ta.rs_sets = [2, 3]
    ta.rs_parts_cpu_m = [100, 200]
    ta.rs_parts_mem_b = [1000, 2000]
    ta.num_priorities = 5

    specs = runner._make_replicaset_specs_only(random.Random(0), ta)
    assert len(specs) == 2
    assert specs[0]["replicas"] == 2
    assert specs[0]["req_cpu_m"] == 100
    assert specs[0]["req_mem_bytes"] == 1000
    assert specs[0]["name"].startswith("rs-01-p")

    ta.rs_sets = []
    with pytest.raises(ValueError):
        runner._make_replicaset_specs_only(random.Random(0), ta)

    ta.rs_sets = [0]
    ta.rs_parts_cpu_m = [1]
    ta.rs_parts_mem_b = [1]
    with pytest.raises(ValueError):
        runner._make_replicaset_specs_only(random.Random(0), ta)


def test_make_replicaset_specs_only_clamps_priority_when_num_priorities_zero():
    runner = tr.TestRunner(ns(), initialize=False)

    class TA:
        pass

    ta = TA()
    ta.rs_sets = [1]
    ta.rs_parts_cpu_m = [10]
    ta.rs_parts_mem_b = [20]
    ta.num_priorities = 0  # should behave as if 1

    specs = runner._make_replicaset_specs_only(random.Random(0), ta)
    assert specs[0]["priority"] == 1
    assert "p1" in specs[0]["name"]


# ---------------------------------------------------------------------------
# TestRunner._apply_replicasets
# ---------------------------------------------------------------------------

def test_apply_replicasets_calls_kubectl_and_wait(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.ctx = "ctx"

    calls = {"apply": [], "wait": []}
    monkeypatch.setattr(tr, "yaml_kwok_rs", lambda *a, **k: "YAML")
    monkeypatch.setattr(tr, "kubectl_apply_yaml", lambda log, ctx, y: calls["apply"].append((ctx, y)))
    monkeypatch.setattr(tr, "wait_rs_pods", lambda log, ctx, name, ns, timeout_s, mode: calls["wait"].append((name, mode)))

    ta = applied_cfg(wait_pod_mode="ready", wait_pod_timeout_s=3, num_pods=2)
    specs = [{"name": "rs-01-p1", "priority": 1, "req_cpu_m": 100, "req_mem_bytes": 200, "replicas": 2}]

    runner._apply_replicasets(ta, specs)
    assert calls["apply"] == [("ctx", "YAML")]
    assert calls["wait"] == [("rs-01-p1", "ready")]


def test_apply_replicasets_skips_wait_when_mode_none(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.ctx = "ctx"

    calls = {"apply": 0, "wait": 0}
    monkeypatch.setattr(tr, "yaml_kwok_rs", lambda *a, **k: "YAML")
    monkeypatch.setattr(tr, "kubectl_apply_yaml", lambda *_a, **_k: calls.__setitem__("apply", calls["apply"] + 1))
    monkeypatch.setattr(tr, "wait_rs_pods", lambda *_a, **_k: calls.__setitem__("wait", calls["wait"] + 1))

    ta = applied_cfg(wait_pod_mode=None, wait_pod_timeout_s=3, num_pods=2)
    specs = [{"name": "rs-01-p1", "priority": 1, "req_cpu_m": 100, "req_mem_bytes": 200, "replicas": 2}]

    runner._apply_replicasets(ta, specs)
    assert calls["apply"] == 1
    assert calls["wait"] == 0


# ---------------------------------------------------------------------------
# TestRunner._gen_rs_sizes
# ---------------------------------------------------------------------------

def test_gen_rs_sizes():
    rng = random.Random(0)
    sizes = tr.TestRunner._gen_rs_sizes(rng, num_pods=20, replicas_per_set=(6, 8))
    assert sum(sizes) == 20
    assert all(1 <= s <= 8 for s in sizes)
    assert sizes[0] >= 6


# ---------------------------------------------------------------------------
# TestRunner._solver_directly
# ---------------------------------------------------------------------------

def test_solver_directly_exports_and_preplace(tmp_path, monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(
        solver_timeout_ms=123,
        solver_input_export=tmp_path / "in.json",
        solver_output_export=tmp_path / "out.json",
        solver_directly_running_target_util=0.5,
    )

    class TA:
        namespace = "ns"
        num_nodes = 2
        node_cpu_m = 1000
        node_mem_b = 2000

    rs_specs = [{"name": "rs-01-p1", "priority": 1, "req_cpu_m": 100, "req_mem_bytes": 200, "replicas": 6}]

    monkeypatch.setattr(tr.importlib_metadata, "version", lambda _d: "9.14.6206")

    class R:
        stdout = json.dumps({"status": "OK", "placements": [1]}).encode("utf-8")
        stderr = b""

    monkeypatch.setattr(tr.subprocess, "run", lambda *a, **k: R())

    resp, meta = runner._solver_directly(TA(), seed=9, rs_specs=rs_specs)
    assert resp["status"] == "OK"
    assert meta["total_pods"] == 6
    assert meta["total_node_cpu"] == 2000
    assert meta["total_node_mem"] == 4000
    assert meta["initial_running_uids"]

    inst = json.loads((tmp_path / "in.json").read_text(encoding="utf-8"))
    assert any((p.get("node") or "") for p in inst.get("pods", []))
    assert (tmp_path / "out.json").exists()


def test_solver_directly_stdout_parse_error_writes_output(tmp_path, monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(
        solver_timeout_ms=1,
        solver_input_export=None,
        solver_output_export=tmp_path / "out.json",
        solver_directly_running_target_util=0.0,
    )

    class TA:
        namespace = "ns"
        num_nodes = 1
        node_cpu_m = 100
        node_mem_b = 100

    rs_specs = [{"name": "rs-01-p1", "priority": 1, "req_cpu_m": 1, "req_mem_bytes": 1, "replicas": 1}]

    monkeypatch.setattr(tr.importlib_metadata, "version", lambda _d: "9.14.6206")

    class R:
        stdout = b"not-json"
        stderr = b""

    monkeypatch.setattr(tr.subprocess, "run", lambda *a, **k: R())

    resp, _ = runner._solver_directly(TA(), seed=1, rs_specs=rs_specs)
    assert resp["status"] == "PY_SOLVER_STDOUT_PARSE_ERROR"
    assert (tmp_path / "out.json").exists()


def test_solver_directly_missing_or_wrong_ortools_version_raises(monkeypatch, tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(
        solver_timeout_ms=1,
        solver_input_export=None,
        solver_output_export=tmp_path / "out.json",
        solver_directly_running_target_util=0.0,
    )

    class TA:
        namespace = "ns"
        num_nodes = 1
        node_cpu_m = 100
        node_mem_b = 100

    rs_specs = [{"name": "rs", "priority": 1, "req_cpu_m": 1, "req_mem_bytes": 1, "replicas": 1}]

    # missing
    monkeypatch.setattr(tr.importlib_metadata, "version", lambda _d: (_ for _ in ()).throw(tr.importlib_metadata.PackageNotFoundError()))
    with pytest.raises(SystemExit):
        runner._solver_directly(TA(), seed=1, rs_specs=rs_specs)

    # mismatch
    monkeypatch.setattr(tr.importlib_metadata, "version", lambda _d: "0.0.0")
    with pytest.raises(SystemExit):
        runner._solver_directly(TA(), seed=1, rs_specs=rs_specs)


def test_solver_directly_python_exception_logs_error(tmp_path, monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(
        solver_timeout_ms=1,
        solver_input_export=None,
        solver_output_export=tmp_path / "out.json",
        solver_directly_running_target_util=0.0,
    )

    class TA:
        namespace = "ns"
        num_nodes = 1
        node_cpu_m = 100
        node_mem_b = 100

    rs_specs = [{"name": "rs", "priority": 1, "req_cpu_m": 1, "req_mem_bytes": 1, "replicas": 1}]

    monkeypatch.setattr(tr.importlib_metadata, "version", lambda _d: "9.14.6206")

    class R:
        stdout = json.dumps({"status": "PYTHON_EXCEPTION", "error": "boom"}).encode("utf-8")
        stderr = b""

    seen = {"err": 0}
    monkeypatch.setattr(tr.LOG, "error", lambda *_a, **_k: seen.__setitem__("err", seen["err"] + 1))
    monkeypatch.setattr(tr.subprocess, "run", lambda *a, **k: R())

    resp, _ = runner._solver_directly(TA(), seed=1, rs_specs=rs_specs)
    assert resp["status"] == "PYTHON_EXCEPTION"
    assert seen["err"] >= 1
    assert (tmp_path / "out.json").exists()


# ---------------------------------------------------------------------------
# TestRunner._wait_solver_inactive_http
# ---------------------------------------------------------------------------

def test_wait_solver_inactive_http_invalid_json(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    tc = TimeController(now=0.0)

    # First: invalid JSON -> keep waiting; then becomes inactive
    responses = [(200, "{not-json"), (200, '{"Active": false}')]

    monkeypatch.setattr(tr.time, "time", tc.time)
    monkeypatch.setattr(tr.time, "sleep", tc.sleep)
    monkeypatch.setattr(tr, "get_solver_active_status_http", lambda _url: responses.pop(0))

    ok = runner._wait_solver_inactive_http("http://x/active", timeout_s=10, poll_initial_s=0.0, poll_interval_s=0.0)
    assert ok is True


def test_wait_solver_inactive_http_timeout(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    tc = TimeController(now=0.0)

    monkeypatch.setattr(tr.time, "time", tc.time)
    monkeypatch.setattr(tr.time, "sleep", tc.sleep)
    monkeypatch.setattr(tr, "get_solver_active_status_http", lambda _url: (200, '{"Active": true}'))

    ok = runner._wait_solver_inactive_http("http://x/active", timeout_s=0, poll_initial_s=0.0, poll_interval_s=0.0)
    assert ok is False


def test_wait_solver_inactive_http_logs_and_non_str_body(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    tc = TimeController(now=3.0)

    def fake_status(_url):
        tc.now += 0.6
        return (500, b"bytes")

    monkeypatch.setattr(tr.time, "time", tc.time)
    monkeypatch.setattr(tr.time, "sleep", tc.sleep)
    monkeypatch.setattr(tr, "get_solver_active_status_http", fake_status)

    ok = runner._wait_solver_inactive_http("http://x/active", timeout_s=1, poll_initial_s=0.0, poll_interval_s=0.2)
    assert ok is False


# ---------------------------------------------------------------------------
# TestRunner._pause
# ---------------------------------------------------------------------------

def test_pause_skips_when_disabled(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(pause=False)

    monkeypatch.setattr(tr.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not prompt")),
    )

    runner._pause(next_exists=True)


def test_pause_skips_when_not_tty(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(pause=True)

    monkeypatch.setattr(tr.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not prompt")),
    )

    runner._pause(next_exists=True)


def test_pause_keyboard_interrupt(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(pause=True)

    monkeypatch.setattr(tr.sys.stdin, "isatty", lambda: True)

    def boom(*_a, **_k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(builtins, "input", boom)
    with pytest.raises(SystemExit):
        runner._pause(next_exists=True)


# ---------------------------------------------------------------------------
# TestRunner.run_mode_single_seed
# ---------------------------------------------------------------------------

def test_run_mode_single_seed(monkeypatch, tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.seen_results = {123}
    runner.results_f = tmp_path / "results.csv"
    runner.args = ns(seed=123, re_run_seeds=False, repeats=1)

    monkeypatch.setattr(runner, "_eta_update_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_eta_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_log_seed_run", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_resolve_config_for_seed", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not resolve")))

    runner.run_mode_single_seed()


# ---------------------------------------------------------------------------
# TestRunner.run_mode_count
# ---------------------------------------------------------------------------

def test_run_mode_count(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.seen_results = {1}  # force a skip
    runner.quota_reached = False
    runner.args = ns(count=2, re_run_seeds=False, repeats=1, pause=False, seed=None)

    seq = {"vals": [1, 1, 2, 3]}

    class RNG:
        def getrandbits(self, _n):
            return seq["vals"].pop(0)

    monkeypatch.setattr(tr, "seeded_random", lambda *_a, **_k: RNG())
    monkeypatch.setattr(tr.time, "time_ns", lambda: 42)
    monkeypatch.setattr(runner, "_log_seed_run", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_eta_update_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_eta_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_pause", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_resolve_config_for_seed", lambda *_a, **_k: object())

    ran = {"seeds": []}

    def fake_run_single(seed, _cfg, **_k):
        ran["seeds"].append(seed)
        return True

    monkeypatch.setattr(runner, "_run_single_seed", fake_run_single)

    runner.run_mode_count()
    assert ran["seeds"] == [2, 3]


# ---------------------------------------------------------------------------
# TestRunner.run_mode_seed_file
# ---------------------------------------------------------------------------

def test_run_mode_seed_file(monkeypatch, tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.seen_results = {2}
    runner.quota_reached = False
    runner.results_f = tmp_path / "results.csv"
    runner.args = ns(seed_file=str(tmp_path / "seeds.txt"), re_run_seeds=False, repeats=1, pause=False, seed=None)

    # Use real file reading to cover that path too
    (tmp_path / "seeds.txt").write_text("1\n2\n3\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_log_seed_run", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_eta_update_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_eta_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_pause", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_resolve_config_for_seed", lambda *_a, **_k: object())

    ran = {"seeds": []}

    def fake_run_single(seed, _cfg, *_a, **_k):
        ran["seeds"].append(seed)
        return True

    monkeypatch.setattr(runner, "_run_single_seed", fake_run_single)

    runner.run_mode_seed_file()
    assert ran["seeds"] == [1, 3]


# ---------------------------------------------------------------------------
# TestRunner._run_single_seed
# ---------------------------------------------------------------------------

def test_run_single_seed(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(solver_directly=True)
    runner.quota_reached = False

    monkeypatch.setattr(tr, "RETRIES_ON_FAIL", 1)

    calls = {"n": 0}

    def fake_exec(_seed, _ta):
        calls["n"] += 1
        return calls["n"] == 2

    monkeypatch.setattr(runner, "_execute_seed_direct", fake_exec)

    ok = runner._run_single_seed(seed=123, ta=object(), run_idx=1)
    assert ok is True
    assert calls["n"] == 2
    assert runner.suppress_fail_log is False
    assert runner.failure is None


# ---------------------------------------------------------------------------
# TestRunner._execute_seed_direct
# ---------------------------------------------------------------------------

def test_execute_seed_direct(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(solver_directly=True)

    monkeypatch.setattr(
        runner,
        "_make_replicaset_specs_only",
        lambda _rng, _ta: [{"name": "rs", "priority": 1, "req_cpu_m": 10, "req_mem_bytes": 20, "replicas": 2}],
    )

    resp = {
        "status": "OK",
        "placements": [
            {"pod": {"uid": "u1"}, "from_node": ""},
            {"pod": {"uid": "u2"}, "from_node": "node-0"},
        ],
        "evictions": [{"pod": {"uid": "u2"}}],
    }
    meta = {
        "initial_running_uids": ["u2"],
        "uid_to_priority": {"u1": 5, "u2": 1},
        "uid_to_cpu": {"u1": 10, "u2": 10},
        "uid_to_mem": {"u1": 20, "u2": 20},
        "total_pods": 2,
        "total_node_cpu": 100,
        "total_node_mem": 200,
        "elapsed_ms": 7,
    }
    monkeypatch.setattr(runner, "_solver_directly", lambda _ta, _seed, _rs_specs: (resp, meta))

    seen = {"seed": None, "note": None}
    monkeypatch.setattr(runner, "_log_seed_summary", lambda seed, note="": seen.update({"seed": seed, "note": note}))

    ta = applied_cfg(num_pods=2, node_cpu_m=100, node_mem_b=200)
    ok = runner._execute_seed_direct(seed=5, ta=ta)

    assert ok is True
    assert seen["seed"] == 5
    assert "direct-solving status=OK" in (seen["note"] or "")
    assert "running=" in (seen["note"] or "")
    assert "unscheduled=" in (seen["note"] or "")


# ---------------------------------------------------------------------------
# TestRunner._execute_seed_on_cluster
# ---------------------------------------------------------------------------

def test_execute_seed_on_cluster_ensure_cluster_failure(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(cluster_name="kwok1", kwok_runtime="binary", default_scheduler=True, solver_trigger=False)
    runner.kwokctl_config_doc = {}
    runner.ctx = "ctx"

    monkeypatch.setattr(tr, "ensure_kwok_cluster", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    seen = {}
    monkeypatch.setattr(
        runner,
        "_record_failure",
        lambda cat, seed, phase, msg, details="": seen.update({"cat": cat, "seed": seed, "phase": phase, "msg": msg}),
    )
    monkeypatch.setattr(runner, "_log_seed_summary", lambda *_a, **_k: None)

    ok = runner._execute_seed_on_cluster(seed=1, ta=applied_cfg(), run_idx=1)
    assert ok is False
    assert seen["cat"] == "seed"
    assert seen["phase"] == "ensure_cluster"


def test_execute_seed_on_cluster_nodes_exception_records_failure(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.ctx = "ctx"
    runner.kwokctl_config_doc = {}
    runner.args = ns(
        cluster_name="kwok1",
        kwok_runtime="binary",
        default_scheduler=True,
        solver_trigger=False,
        save_solver_stats=False,
        save_scheduler_logs=False,
        seeds_not_all_running=0,
    )

    monkeypatch.setattr(tr, "ensure_kwok_cluster", lambda *a, **k: None)
    monkeypatch.setattr(tr, "qty_to_mcpu_str", lambda *_a, **_k: "100m")
    monkeypatch.setattr(tr, "qty_to_bytes_str", lambda *_a, **_k: "1Mi")
    monkeypatch.setattr(tr, "kwok_pods_cap", lambda *_a, **_k: 10)
    monkeypatch.setattr(tr, "create_kwok_nodes", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))

    seen = {}
    monkeypatch.setattr(
        runner,
        "_record_failure",
        lambda cat, seed, phase, msg, details="": seen.update({"cat": cat, "seed": seed, "phase": phase, "msg": msg}),
    )
    monkeypatch.setattr(runner, "_log_seed_summary", lambda *_a, **_k: None)

    ok = runner._execute_seed_on_cluster(seed=99, ta=applied_cfg(node_mem_b=200), run_idx=1)
    assert ok is False
    assert seen["phase"] == "nodes"
    assert "boom" in seen["msg"]


def test_execute_seed_on_cluster_snapshot_validation_mismatch_fails(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.args = ns(cluster_name="kwok1", kwok_runtime="binary", default_scheduler=True, solver_trigger=False)
    runner.kwokctl_config_doc = {}
    runner.ctx = "ctx"

    monkeypatch.setattr(tr, "ensure_kwok_cluster", lambda *a, **k: None)
    monkeypatch.setattr(tr, "create_kwok_nodes", lambda *a, **k: None)
    monkeypatch.setattr(tr, "ensure_namespace", lambda *a, **k: None)
    monkeypatch.setattr(tr, "ensure_service_account", lambda *a, **k: None)
    monkeypatch.setattr(tr, "ensure_priority_classes", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_apply_replicasets", lambda *a, **k: None)
    monkeypatch.setattr(tr.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(tr, "qty_to_mcpu_str", lambda *_a, **_k: "100m")
    monkeypatch.setattr(tr, "qty_to_bytes_str", lambda *_a, **_k: "1Mi")
    monkeypatch.setattr(tr, "kwok_pods_cap", lambda *_a, **_k: 10)
    monkeypatch.setattr(
        runner,
        "_make_replicaset_specs_only",
        lambda *_a, **_k: [{"name": "rs", "priority": 1, "req_cpu_m": 1, "req_mem_bytes": 1, "replicas": 1}],
    )

    # running+unsched != expected -> fail early
    class Snap:
        pods_running = ["p1"]
        pods_unscheduled = []
        cpu_run_util = 0.0
        mem_run_util = 0.0
        cpu_req_by_node = {}
        mem_req_by_node = {}
        pods_run_by_node = {}
        running_placed_by_prio = {}
        unschedulable_by_prio = {}

    monkeypatch.setattr(tr, "stat_snapshot", lambda *_a, **_k: Snap())

    seen = {}
    monkeypatch.setattr(
        runner,
        "_record_failure",
        lambda cat, seed, phase, msg, details="": seen.update({"cat": cat, "seed": seed, "phase": phase, "msg": msg}),
    )
    monkeypatch.setattr(runner, "_log_seed_summary", lambda *_a, **_k: None)

    ok = runner._execute_seed_on_cluster(seed=2, ta=applied_cfg(num_pods=2), run_idx=1)
    assert ok is False
    assert seen["phase"] == "snapshot_validation_before"
    assert "pod count mismatch" in seen["msg"]


def test_execute_seed_on_cluster_happy_path(monkeypatch, tmp_path):
    runner = tr.TestRunner(ns(), initialize=False)
    runner.ctx = "ctx"
    runner.kwokctl_config_doc = {}
    runner.results_f = tmp_path / "results.csv"
    runner.quota_reached = False
    runner.saved_not_all_running = 0
    runner.seeds_all_running_f = tmp_path / "seeds-all-running.txt"
    runner.seeds_not_all_running_f = tmp_path / "seeds-not-all-running.txt"
    runner.solver_stats_dir = tmp_path / "solver-stats"
    runner.scheduler_logs_dir = tmp_path / "scheduler-logs"

    runner.args = ns(
        cluster_name="kwok1",
        kwok_runtime="binary",
        default_scheduler=False,
        solver_trigger=False,
        save_solver_stats=True,
        save_scheduler_logs=True,
        seeds_not_all_running=1,
    )

    monkeypatch.setattr(tr, "ensure_kwok_cluster", lambda *a, **k: None)
    monkeypatch.setattr(tr, "create_kwok_nodes", lambda *a, **k: None)
    monkeypatch.setattr(tr, "ensure_namespace", lambda *a, **k: None)
    monkeypatch.setattr(tr, "ensure_service_account", lambda *a, **k: None)
    monkeypatch.setattr(tr, "ensure_priority_classes", lambda *a, **k: None)
    monkeypatch.setattr(tr, "qty_to_mcpu_str", lambda *_a, **_k: "100m")
    monkeypatch.setattr(tr, "qty_to_bytes_str", lambda *_a, **_k: "1Mi")
    monkeypatch.setattr(tr, "kwok_pods_cap", lambda *_a, **_k: 10)
    monkeypatch.setattr(tr.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(tr, "get_timestamp", lambda: "TS")

    rs_specs = [{"name": "rs-01-p1", "priority": 1, "req_cpu_m": 10, "req_mem_bytes": 20, "replicas": 2}]
    monkeypatch.setattr(runner, "_make_replicaset_specs_only", lambda *_a, **_k: rs_specs)
    monkeypatch.setattr(runner, "_apply_replicasets", lambda *_a, **_k: None)

    class Snap:
        def __init__(self, running, unsched, cpu_util, mem_util):
            self.pods_running = running
            self.pods_unscheduled = unsched
            self.cpu_run_util = cpu_util
            self.mem_run_util = mem_util
            self.cpu_req_by_node = {"n1": 10}
            self.mem_req_by_node = {"n1": 20}
            self.pods_run_by_node = {"n1": [p[0] for p in running]}
            self.running_placed_by_prio = {"1": len(running)}
            self.unschedulable_by_prio = {"1": len(unsched)}

    # Before + after (both must have total pods == expected)
    snaps = [
        Snap(running=[("rs-01-p1-0", "n1")], unsched=["rs-01-p1-1"], cpu_util=0.1, mem_util=0.2),
        Snap(running=[("rs-01-p1-0", "n1")], unsched=["rs-01-p1-1"], cpu_util=0.1, mem_util=0.2),
    ]
    monkeypatch.setattr(tr, "stat_snapshot", lambda *_a, **_k: snaps.pop(0))

    monkeypatch.setattr(
        runner,
        "_get_solver_attempts",
        lambda: (
            {"x": 1},
            "best",
            [{"name": "best", "score": {"s": 1}, "duration_ms": 7, "status": "OK"}],
            "",
        ),
    )

    seen = {"row": None, "solver_stats": 0, "sched_logs": 0, "outcome": None}
    monkeypatch.setattr(runner, "_append_result_csv", lambda row: seen.__setitem__("row", row))
    monkeypatch.setattr(runner, "_write_solver_stats_json", lambda *_a, **_k: seen.__setitem__("solver_stats", seen["solver_stats"] + 1))
    monkeypatch.setattr(runner, "_save_scheduler_logs", lambda *_a, **_k: seen.__setitem__("sched_logs", seen["sched_logs"] + 1))
    monkeypatch.setattr(runner, "_record_seed_outcome", lambda *, seed, all_running: seen.__setitem__("outcome", (seed, all_running)))
    monkeypatch.setattr(runner, "_log_seed_summary", lambda *_a, **_k: None)

    ta = applied_cfg(num_nodes=1, num_pods=2, num_priorities=1, node_cpu_m=100, node_mem_b=200)

    ok = runner._execute_seed_on_cluster(seed=7, ta=ta, run_idx=1)
    assert ok is True

    assert seen["row"] is not None
    assert seen["row"]["timestamp"] == "TS"
    assert seen["row"]["seed"] == "7"
    assert seen["solver_stats"] == 1
    assert seen["sched_logs"] == 1

    # unscheduled != 0 and seeds_not_all_running=1 => quota reached
    assert runner.quota_reached is True
    assert runner.saved_not_all_running == 1

    # outcome recorded: not all running
    assert seen["outcome"] == (7, False)


# ---------------------------------------------------------------------------
# Main runner dispatch
# ---------------------------------------------------------------------------

def test_run_dispatches_to_correct_mode(monkeypatch):
    runner = tr.TestRunner(ns(), initialize=False)

    called = {"single": 0, "count": 0, "file": 0}
    monkeypatch.setattr(runner, "run_mode_single_seed", lambda: called.__setitem__("single", called["single"] + 1))
    monkeypatch.setattr(runner, "run_mode_count", lambda: called.__setitem__("count", called["count"] + 1))
    monkeypatch.setattr(runner, "run_mode_seed_file", lambda: called.__setitem__("file", called["file"] + 1))

    runner.args = ns(seed=1, count=None, seed_file=None)
    runner.run()
    assert called == {"single": 1, "count": 0, "file": 0}

    runner.args = ns(seed=None, count=5, seed_file=None)
    runner.run()
    assert called == {"single": 1, "count": 1, "file": 0}

    runner.args = ns(seed=None, count=None, seed_file="seeds.txt")
    runner.run()
    assert called == {"single": 1, "count": 1, "file": 1}
