#!/usr/bin/env python3
# test_kwok_trace_replayer.py

import argparse, json, time
from pathlib import Path

import pytest

from scripts.kwok_trace_replayer import trace_replayer as tr

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def mk_paths(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("kind: KwokctlConfiguration\n", encoding="utf-8")
    result_dir = tmp_path / "results"
    return trace_dir, cfg, result_dir


def mk_args(tmp_path: Path, **overrides):
    trace_dir, cfg, result_dir = mk_paths(tmp_path)
    base = dict(
        trace_dir=str(trace_dir),
        kwokctl_config_file=str(cfg),
        result_dir=str(result_dir),
        cluster_name="kwok1",
        kwok_runtime="binary",
        namespace="trace",
        node_cpu="1000m",
        node_mem="1Gi",
        monitor_interval=1.0,
        start_delay=0.0,
        log_level="INFO",
        job_file=None,
        save_scheduler_logs=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base), trace_dir, cfg, result_dir


def mk_replayer(tmp_path: Path, monkeypatch, **overrides):
    args, trace_dir, cfg, result_dir = mk_args(tmp_path, **overrides)

    # Avoid noisy info writes/logging unless explicitly tested
    monkeypatch.setattr(tr.TraceReplayer, "write_info_file", lambda self: None)
    monkeypatch.setattr(tr.TraceReplayer, "log_args", lambda self: None)

    rp = tr.TraceReplayer(args)
    return rp, args, trace_dir, cfg, result_dir


class FakeClock:
    def __init__(self, t0: float = 0.0):
        self._t = float(t0)
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        seconds = float(seconds)
        self.sleeps.append(seconds)
        if seconds > 0:
            self._t += seconds


class OkFuture:
    def result(self):
        return None


class BadFuture:
    def result(self):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# build_argparser()
# ---------------------------------------------------------------------------

def test_build_argparser_parses_boolean_optional_action():
    p = tr.build_argparser()

    args = p.parse_args(["--save-scheduler-logs"])
    assert args.save_scheduler_logs is True

    args = p.parse_args(["--no-save-scheduler-logs"])
    assert args.save_scheduler_logs is False


# ---------------------------------------------------------------------------
# merge_job_fields_into_args()
# ---------------------------------------------------------------------------

def test_merge_job_fields_into_args_cli_wins_and_parses_monitor_interval_and_start_delay():
    args = argparse.Namespace(
        trace_dir="/cli/trace",
        cluster_name=None,
        kwok_runtime=None,
        kwokctl_config_file=None,
        namespace=None,
        node_cpu=None,
        node_mem=None,
        monitor_interval=None,
        start_delay=None,
        log_level=None,
        result_dir=None,
        save_scheduler_logs=None,
        job_file=None,
    )

    job = {
        "trace-dir": "/job/trace",
        "cluster-name": "kwok2",
        "kwok-runtime": "docker",
        "kwokctl-config-file": "/job/cfg.yaml",
        "namespace": "ns",
        "node-cpu": "500m",
        "node-mem": "512Mi",
        "monitor-interval": "2.5",
        "start-delay": "1.25",
        "log-level": "DEBUG",
        "result-dir": "/job/results",
        "save-scheduler-logs": True,
        "override-kwokctl-envs": [{"name": "X", "value": "Y"}],
    }

    merged, override_envs = tr.merge_job_fields_into_args(args, job)

    assert merged.trace_dir == "/cli/trace"
    assert merged.cluster_name == "kwok2"
    assert merged.kwok_runtime == "docker"
    assert merged.monitor_interval == 2.5
    assert merged.start_delay == 1.25
    assert merged.save_scheduler_logs is True
    assert override_envs == [{"name": "X", "value": "Y"}]


def test_merge_job_fields_into_args_ignores_invalid_types_and_strict_bool():
    args = argparse.Namespace(
        trace_dir=None,
        cluster_name=None,
        kwok_runtime=None,
        kwokctl_config_file=None,
        namespace=None,
        node_cpu=None,
        node_mem=None,
        monitor_interval=None,
        start_delay=None,
        log_level=None,
        result_dir=None,
        save_scheduler_logs=None,
        job_file=None,
    )

    job = {
        "trace-dir": "/job/trace",
        "monitor-interval": "not-a-float",
        "start-delay": "not-a-float",
        "save-scheduler-logs": "yes",  # strict bool only => ignored
        "override-kwokctl-envs": [],
    }

    merged, override_envs = tr.merge_job_fields_into_args(args, job)
    assert merged.trace_dir == "/job/trace"
    assert merged.monitor_interval is None
    assert merged.start_delay is None
    assert merged.save_scheduler_logs is None
    assert override_envs == []


# ---------------------------------------------------------------------------
# ensure_default_args()
# ---------------------------------------------------------------------------

def test_ensure_default_args_requires_trace_cfg_results(tmp_path: Path):
    args = argparse.Namespace(
        trace_dir=None,
        kwokctl_config_file=None,
        result_dir=None,
        cluster_name=None,
        kwok_runtime=None,
        namespace=None,
        node_cpu=None,
        node_mem=None,
        monitor_interval=None,
        start_delay=None,
        log_level=None,
        job_file=None,
        save_scheduler_logs=None,
    )
    with pytest.raises(SystemExit):
        tr.ensure_default_args(args)


def test_ensure_default_args_sets_defaults_and_validates_paths(tmp_path: Path):
    trace_dir, cfg, result_dir = mk_paths(tmp_path)

    args = argparse.Namespace(
        trace_dir=str(trace_dir),
        kwokctl_config_file=str(cfg),
        result_dir=str(result_dir),
        cluster_name=None,
        kwok_runtime=None,
        namespace=None,
        node_cpu=None,
        node_mem=None,
        monitor_interval=None,
        start_delay=None,
        log_level=None,
        job_file=None,
        save_scheduler_logs=None,
    )

    out = tr.ensure_default_args(args)
    assert out.cluster_name == "kwok1"
    assert out.kwok_runtime == "binary"
    assert out.namespace == "trace"
    assert out.node_cpu == "1000m"
    assert out.node_mem == "1Gi"
    assert out.monitor_interval == 1.0
    assert out.start_delay == 0.0
    assert out.log_level == "INFO"
    assert out.save_scheduler_logs is False
    assert Path(out.result_dir).is_absolute()


# ---------------------------------------------------------------------------
# discover_trace_run_dirs()
# ---------------------------------------------------------------------------

def test_discover_trace_run_dirs_trace_dir_with_trace_json_returns_self(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "trace.json").write_text("[]\n", encoding="utf-8")

    runs = tr.discover_trace_run_dirs(trace_dir)
    assert runs == [trace_dir.resolve()]


def test_discover_trace_run_dirs_scans_and_sorts_seed_dirs(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()

    (trace_dir / "2").mkdir()
    (trace_dir / "2" / "trace.json").write_text("[]\n", encoding="utf-8")

    (trace_dir / "1").mkdir()
    (trace_dir / "1" / "trace.json").write_text("[]\n", encoding="utf-8")

    runs = tr.discover_trace_run_dirs(trace_dir)
    assert [p.name for p in runs] == ["1", "2"]


def test_discover_trace_run_dirs_no_seed_traces_falls_back_to_trace_dir(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()

    runs = tr.discover_trace_run_dirs(trace_dir)
    assert runs == [trace_dir.resolve()]


# ---------------------------------------------------------------------------
# parse_optional_bool_strict()
# ---------------------------------------------------------------------------

def test_parse_optional_bool_strict():
    assert tr.parse_optional_bool_strict(True) is True
    assert tr.parse_optional_bool_strict(False) is False
    assert tr.parse_optional_bool_strict("yes") is None
    assert tr.parse_optional_bool_strict(1) is None


# ---------------------------------------------------------------------------
# rs_prefix_from_pod_name()
# ---------------------------------------------------------------------------

def test_rs_prefix_from_pod_name():
    assert tr.rs_prefix_from_pod_name("rs-000001-abc-0") == "rs-000001"
    assert tr.rs_prefix_from_pod_name("custom-foo-0") == "custom"
    assert tr.rs_prefix_from_pod_name("plain") == "plain"


# ---------------------------------------------------------------------------
# parse_rfc3339_to_epoch()
# ---------------------------------------------------------------------------

def test_parse_rfc3339_to_epoch_parses_z_and_fractional():
    t1 = tr.parse_rfc3339_to_epoch("2026-01-09T12:34:56Z")
    t2 = tr.parse_rfc3339_to_epoch("2026-01-09T12:34:56.123Z")
    assert isinstance(t1, float) and t1 > 0
    assert isinstance(t2, float) and t2 > 0
    assert t2 >= t1


def test_parse_rfc3339_to_epoch_invalid_returns_none():
    assert tr.parse_rfc3339_to_epoch("") is None
    assert tr.parse_rfc3339_to_epoch("nope") is None
    assert tr.parse_rfc3339_to_epoch(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TraceReplayer.__init__()
# ---------------------------------------------------------------------------

def test_trace_replayer_init_sets_paths_and_state(tmp_path: Path, monkeypatch):
    rp, args, trace_dir, _cfg, result_dir = mk_replayer(tmp_path, monkeypatch)

    assert rp.base_dir == trace_dir.resolve()
    assert rp.trace_path == trace_dir.resolve() / "trace.json"
    assert rp.initial_path == trace_dir.resolve() / "initial.json"
    assert rp.results_dir == result_dir.resolve()

    assert rp.trace_pods == []
    assert rp.initial_pods == []
    assert rp.events == []
    assert rp.ctx == f"kwok-{args.cluster_name}"


# ---------------------------------------------------------------------------
# TraceReplayer.log_args()
# ---------------------------------------------------------------------------

def test_log_args_calls_log_args_block(tmp_path: Path, monkeypatch):
    args, *_ = mk_args(tmp_path)
    monkeypatch.setattr(tr.TraceReplayer, "write_info_file", lambda self: None)

    called = {"n": 0, "include": None, "title": None}

    def fake_log_args_block(logger, ns, title, include):
        called["n"] += 1
        called["include"] = include
        called["title"] = title

    monkeypatch.setattr(tr, "log_args_block", fake_log_args_block)
    _ = tr.TraceReplayer(args)

    assert called["n"] == 1
    assert called["title"] == "ARGS"
    assert "trace_dir" in called["include"]
    assert "start_delay" in called["include"]
    assert "save_scheduler_logs" in called["include"]


# ---------------------------------------------------------------------------
# TraceReplayer.write_info_file()
# ---------------------------------------------------------------------------

def test_write_info_file_success(tmp_path: Path, monkeypatch):
    args, *_ = mk_args(tmp_path)

    monkeypatch.setattr(tr.TraceReplayer, "log_args", lambda self: None)
    monkeypatch.setattr(tr, "build_cli_cmd", lambda: "CLI")
    seen = {}

    def fake_write_info_file(out_path, meta_extra, inputs, logger):
        seen["out_path"] = out_path
        seen["meta_extra"] = meta_extra
        seen["inputs"] = inputs

    monkeypatch.setattr(tr, "write_info_file", fake_write_info_file)

    rp = tr.TraceReplayer(args, job_doc={"k": 1})
    assert seen["out_path"] == rp.results_dir / "info_replayer.yaml"
    assert seen["meta_extra"]["kind"] == "trace_replayer"
    assert seen["inputs"]["cli-cmd"] == "CLI"
    assert seen["inputs"]["job"] == {"k": 1}


def test_write_info_file_failure_is_non_fatal(tmp_path: Path, monkeypatch):
    args, *_ = mk_args(tmp_path)

    monkeypatch.setattr(tr.TraceReplayer, "log_args", lambda self: None)
    monkeypatch.setattr(tr, "write_info_file", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))

    _ = tr.TraceReplayer(args)


# ---------------------------------------------------------------------------
# TraceReplayer._save_scheduler_logs()
# ---------------------------------------------------------------------------

def test_save_scheduler_logs(tmp_path: Path, monkeypatch):
    rp, args, *_ = mk_replayer(tmp_path, monkeypatch)

    seen = {}

    def fake_save(cluster_name, out_path, runner, logger):
        seen["cluster_name"] = cluster_name
        seen["out_path"] = out_path
        seen["runner"] = runner

    monkeypatch.setattr(tr, "save_kwok_scheduler_logs", fake_save)

    rp.save_scheduler_logs()
    assert seen["cluster_name"] == args.cluster_name
    assert seen["out_path"] == rp.results_dir / "scheduler-logs" / "sched_logs.log"
    assert seen["runner"] == rp.runner


# ---------------------------------------------------------------------------
# TraceReplayer._rs_name_for_record()
# ---------------------------------------------------------------------------

def test_rs_name_for_record():
    assert tr.TraceReplayer.rs_name_for_record(1) == "rs-000001"
    assert tr.TraceReplayer.rs_name_for_record(123456) == "rs-123456"


# ---------------------------------------------------------------------------
# TraceReplayer._load_json_pods()
# ---------------------------------------------------------------------------

def test_load_json_pods_missing_file_raises(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        rp.load_json_pods(tmp_path / "nope.json")


def test_load_json_pods_requires_record_keys(tmp_path: Path, monkeypatch):
    rp, _, trace_dir, *_ = mk_replayer(tmp_path, monkeypatch)

    (trace_dir / "trace.json").write_text(json.dumps({"pods": [{"id": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        rp.load_json_pods(trace_dir / "trace.json")


def test_load_json_pods_parses_and_sorts(tmp_path: Path, monkeypatch):
    rp, _, trace_dir, *_ = mk_replayer(tmp_path, monkeypatch)

    raw = {
        "pods": [
            {"id": 2, "start_time": 5.0, "end_time": 6.0, "cpu": 0.2, "mem": 0.2, "priority": 2, "replicas": 1},
            {"id": 1, "start_time": 1.0, "end_time": 4.0, "cpu": 0.1, "mem": 0.1, "priority": 3, "replicas": 2},
        ],
    }
    (trace_dir / "trace.json").write_text(json.dumps(raw), encoding="utf-8")

    pods = rp.load_json_pods(trace_dir / "trace.json")
    assert [p.id for p in pods] == [1, 2]


# ---------------------------------------------------------------------------
# TraceReplayer.load_initial_and_trace()
# ---------------------------------------------------------------------------

def test_load_initial_and_trace_infers_when_no_info_yaml(tmp_path: Path, monkeypatch, caplog):
    rp, _, trace_dir, *_ = mk_replayer(tmp_path, monkeypatch)

    # trace + initial
    (trace_dir / "trace.json").write_text(
        json.dumps(
            {
                "pods": [
                    {"id": 1, "start_time": 1.0, "end_time": 4.0, "cpu": 0.1, "mem": 0.1, "priority": 2, "replicas": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "initial.json").write_text(
        json.dumps(
            {
                "pods": [
                    {"id": 9, "start_time": 0.0, "end_time": 10.0, "cpu": 0.2, "mem": 0.2, "priority": 3, "replicas": 2},
                ]
            }
        ),
        encoding="utf-8",
    )

    caplog.set_level("WARNING", logger=tr.LOGGER_NAME)
    rp.load_initial_and_trace()

    # num_nodes fallback to 8 (as coded)
    assert rp.num_nodes == 8
    # trace_time inferred from max end_time across initial+trace
    assert rp.trace_time_s == 10.0
    assert rp.max_prio == 3
    assert "rs-000009" in rp.initial_rs_names
    assert rp.prio_by_rs["rs-000001"] == 2


def test_load_initial_and_trace_prefers_info_yaml(tmp_path: Path, monkeypatch):
    rp, _, trace_dir, *_ = mk_replayer(tmp_path, monkeypatch)

    (trace_dir / "trace.json").write_text(
        json.dumps({"pods": [{"id": 1, "start_time": 0.0, "end_time": 1.0, "cpu": 0.1, "mem": 0.1, "priority": 1, "replicas": 1}]}),
        encoding="utf-8",
    )
    (trace_dir / "initial.json").write_text(json.dumps({"pods": []}), encoding="utf-8")

    # minimal info_generate.yaml shape accepted by _deep_get()
    (trace_dir / "info_generate.yaml").write_text(
        "\n".join(
            [
                "generated:",
                "  num_nodes: 3",
                "  trace_time_s: 7.0",
                "  max_priority_seen: 5",
            ]
        ) + "\n",
        encoding="utf-8",
    )

    rp.load_initial_and_trace()
    assert rp.num_nodes == 3
    assert rp.trace_time_s == 7.0
    assert rp.max_prio == 5


# ---------------------------------------------------------------------------
# TraceReplayer._build_trace_events()
# ---------------------------------------------------------------------------

def test_build_trace_events_creates_sorted_create_delete_pairs(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.node_cpu_m = 1000
    rp.node_mem_b = 1000
    rp.trace_pods = [
        tr.TraceRecord(id=1, start_time=1.0, end_time=2.0, cpu=0.0, mem=0.0, priority=2, replicas=0),
    ]

    monkeypatch.setattr(tr, "qty_to_mcpu_str", lambda m: f"{m}m")
    monkeypatch.setattr(tr, "qty_to_bytes_str", lambda b: f"{b}")

    rp.build_trace_events()
    assert len(rp.events) == 2
    assert rp.events[0].kind == "create"
    assert rp.events[1].kind == "delete"
    assert rp.events[0].cpu_str == "1m"
    assert rp.events[0].mem_str == "1"
    assert rp.events[0].replicas == 1


def test_build_trace_events_applies_start_delay_only_to_trace_events(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    rp.node_cpu_m = 1000
    rp.node_mem_b = 1000
    rp.args.start_delay = 5.0

    # Initial pod delete should NOT be shifted.
    rp.initial_pods = [tr.TraceRecord(id=10, start_time=0.0, end_time=3.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
    # Trace pod events SHOULD be shifted by start_delay.
    rp.trace_pods = [tr.TraceRecord(id=1, start_time=1.0, end_time=2.0, cpu=0.0, mem=0.0, priority=2, replicas=0)]

    monkeypatch.setattr(tr, "qty_to_mcpu_str", lambda m: f"{m}m")
    monkeypatch.setattr(tr, "qty_to_bytes_str", lambda b: f"{b}")

    rp.build_trace_events()

    kinds_times = [(e.kind, e.sim_time_s, e.record_id) for e in rp.events]
    # initial delete at t=3.0
    assert ("delete", pytest.approx(3.0), 10) in kinds_times
    # trace create/delete shifted by +5.0
    assert ("create", pytest.approx(6.0), 1) in kinds_times
    assert ("delete", pytest.approx(7.0), 1) in kinds_times


def test_build_trace_events_sorts_deletes_before_creates_at_same_time(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    rp.node_cpu_m = 1000
    rp.node_mem_b = 1000
    rp.args.start_delay = 0.0

    # Make an initial delete and a trace create land at the same sim time.
    rp.initial_pods = [tr.TraceRecord(id=9, start_time=0.0, end_time=1.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
    rp.trace_pods = [tr.TraceRecord(id=1, start_time=1.0, end_time=2.0, cpu=0.0, mem=0.0, priority=2, replicas=0)]

    monkeypatch.setattr(tr, "qty_to_mcpu_str", lambda m: f"{m}m")
    monkeypatch.setattr(tr, "qty_to_bytes_str", lambda b: f"{b}")

    rp.build_trace_events()
    assert rp.events[0].sim_time_s == pytest.approx(1.0)
    assert rp.events[1].sim_time_s == pytest.approx(1.0)
    assert rp.events[0].kind == "delete"
    assert rp.events[1].kind == "create"


# ---------------------------------------------------------------------------
# TraceReplayer._apply_initial_workload()
# ---------------------------------------------------------------------------

def test_apply_initial_workload_no_initial_is_noop(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    rp.initial_pods = []
    rp.apply_initial_workload(namespace="trace")  # should not raise


def test_apply_initial_workload_submits_kubectl_tasks(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    rp.ctx = "ctx"
    rp.node_cpu_m = 1000
    rp.node_mem_b = 1000
    rp.initial_pods = [tr.TraceRecord(id=1, start_time=0.0, end_time=10.0, cpu=0.1, mem=0.1, priority=1, replicas=2)]

    monkeypatch.setattr(tr, "qty_to_mcpu_str", lambda m: f"{m}m")
    monkeypatch.setattr(tr, "qty_to_bytes_str", lambda b: f"{b}")
    monkeypatch.setattr(tr, "yaml_kwok_rs", lambda **_k: "YAML")

    submitted = []

    class CapturingExec:
        def __init__(self, *a, **k):
            pass

        def submit(self, fn, *a, **k):
            submitted.append((fn, a, k))
            return OkFuture()

        def shutdown(self, wait=True):
            return None

    rp.executor_factory = lambda **_k: CapturingExec()

    rp.apply_initial_workload(namespace="trace")
    assert len(submitted) == 1
    assert submitted[0][0] == tr.kubectl_apply_yaml


# ---------------------------------------------------------------------------
# TraceReplayer._replay_trace_events()
# ---------------------------------------------------------------------------

def test_replay_trace_events_no_events_sleeps_until_end(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    rp.clock = FakeClock(0.0)
    rp.events = []
    rp.trace_time_s = 10.0

    rp.replay_trace_events(namespace="trace", trace_start_wall=0.0)
    assert rp.clock.sleeps == [pytest.approx(10.0)]


def test_replay_trace_events_batch_sleeps_to_batch_and_aligns_end(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    rp.clock = FakeClock(0.0)
    rp.trace_time_s = 2.0
    rp.ctx = "ctx"

    rp.events = [
        tr.Event(sim_time_s=1.0, kind="create", record_id=1, cpu_str="1m", mem_str="1", pc_name="p1", replicas=2),
        tr.Event(sim_time_s=1.0, kind="delete", record_id=1),
    ]

    monkeypatch.setattr(tr, "yaml_kwok_rs", lambda **_k: "YAML")

    class CapturingExec:
        def __init__(self, *a, **k):
            self.submitted = []

        def submit(self, fn, *a, **k):
            self.submitted.append((fn, a, k))
            return OkFuture()

        def shutdown(self, wait=True):
            return None

    ex = CapturingExec()
    rp.executor_factory = lambda **_k: ex

    rp.replay_trace_events(namespace="trace", trace_start_wall=0.0)

    # sleep to batch at t=1, then align to end t=2
    assert rp.clock.sleeps == [pytest.approx(1.0), pytest.approx(1.0)]
    assert len(ex.submitted) == 2


def test_replay_trace_events_future_exception_is_caught(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    rp.clock = FakeClock(0.0)
    rp.trace_time_s = 0.0
    rp.events = [tr.Event(sim_time_s=0.0, kind="create", record_id=1, cpu_str="1m", mem_str="1", pc_name="p1")]

    monkeypatch.setattr(tr, "yaml_kwok_rs", lambda **_k: "YAML")

    class ExecBad:
        def submit(self, *_a, **_k):
            return BadFuture()

        def shutdown(self, wait=True):
            return None

    rp.executor_factory = lambda **_k: ExecBad()

    rp.replay_trace_events(namespace="trace", trace_start_wall=0.0)  # should not raise


# ---------------------------------------------------------------------------
# TraceReplayer._snapshot_from_pods()
# ---------------------------------------------------------------------------

def test_snapshot_from_pods_computes_utilization_and_counts(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.num_nodes = 2
    rp.node_cpu_m = 1000
    rp.node_mem_b = 1000
    rp.max_prio = 2
    rp.prio_by_rs = {"rs-000001": 2, "custom": 1}

    pods_json = {
        "items": [
            {
                "metadata": {"name": "rs-000001-aaa-0", "uid": "u1"},
                "spec": {"nodeName": "n1", "containers": [{"resources": {"requests": {"cpu": "100m", "memory": "10"}}}]},
                "status": {"phase": "Running"},
            },
            {
                "metadata": {"name": "custom-foo-0", "uid": "u2"},
                "spec": {"containers": [{"resources": {"requests": {"cpu": "200m", "memory": "20"}}}]},
                "status": {"phase": "Pending"},
            },
        ]
    }

    monkeypatch.setattr(tr, "get_json_ctx", lambda *_a, **_k: pods_json)
    monkeypatch.setattr(tr, "qty_to_mcpu_int", lambda q: 0 if q is None else int(str(q).rstrip("m")))
    monkeypatch.setattr(tr, "qty_to_bytes_int", lambda q: 0 if q is None else int(str(q)))

    cpu_u, mem_u, cpu_req_u, mem_req_u, running_by_prio, pending_by_prio, items = rp.snapshot_from_pods("trace")
    assert items == pods_json["items"]
    assert cpu_u == pytest.approx(100 / (2 * 1000))
    assert mem_u == pytest.approx(10 / (2 * 1000))
    assert cpu_req_u == pytest.approx((100 + 200) / (2 * 1000))
    assert mem_req_u == pytest.approx((10 + 20) / (2 * 1000))
    assert running_by_prio == {1: 0, 2: 1}
    assert pending_by_prio == {1: 1, 2: 0}


# ---------------------------------------------------------------------------
# TraceReplayer._pod_start_epoch() / _pod_creation_epoch()
# ---------------------------------------------------------------------------

def test_pod_epoch_extractors():
    rp = tr.TraceReplayer.__new__(tr.TraceReplayer)  # bypass init
    pod = {
        "metadata": {"creationTimestamp": "2026-01-09T12:00:00Z"},
        "status": {"startTime": "2026-01-09T12:00:01Z"},
    }
    assert rp.pod_creation_time(pod) is not None
    assert rp.pod_start_time(pod) is not None


# ---------------------------------------------------------------------------
# TraceReplayer._monitor_loop()
# ---------------------------------------------------------------------------

def test_monitor_loop_writes_csv_and_skips_initial_rs_in_pod_stats(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.clock = FakeClock(0.0)
    rp.max_prio = 2
    rp.run_start_monotonic = 0.0
    rp.run_start_wall = 0.0

    rp.prio_by_rs = {"rs-000001": 2, "rs-000002": 1}
    rp.initial_rs_names = {"rs-000001"}  # should be skipped in pod_stats

    # One snapshot, then stop.
    items = [
        # initial RS pod (skip in pod_stats)
        {"metadata": {"name": "rs-000001-aaa-0", "uid": "u1", "creationTimestamp": "2026-01-09T12:00:00Z"}, "status": {"phase": "Running", "startTime": "2026-01-09T12:00:01Z"}},
        # trace RS pod (include)
        {"metadata": {"name": "rs-000002-bbb-0", "uid": "u2", "creationTimestamp": "2026-01-09T12:00:00Z"}, "status": {"phase": "Running", "startTime": "2026-01-09T12:00:02Z"}},
    ]

    def snap(_ns):
        return (0.25, 0.50, 0.75, 0.80, {1: 0, 2: 2}, {1: 0, 2: 0}, items)

    monkeypatch.setattr(rp, "snapshot_from_pods", snap)
    monkeypatch.setattr(tr, "get_timestamp", lambda: "T")

    stop_event = tr.threading.Event()

    def sleep_and_stop(_s: float):
        stop_event.set()

    rp.clock.sleep = sleep_and_stop

    general_csv = tmp_path / "general.csv"
    pod_csv = tmp_path / "pod.csv"

    rp.monitor_loop(
        namespace="trace",
        interval_s=10.0,
        general_csv=general_csv,
        pod_stats_csv=pod_csv,
        stop_event=stop_event,
    )

    # general has header + 1 row
    glines = general_csv.read_text(encoding="utf-8").splitlines()
    assert len(glines) == 2
    assert glines[0].startswith("timestamp,time_s,cpu_run_util,mem_run_util,cpu_req_util,mem_req_util,running_p1")

    # pod_stats should only include rs-000002... entries (not rs-000001...)
    plines = pod_csv.read_text(encoding="utf-8").splitlines()
    assert plines[0].startswith("timestamp,event,pod_name,pod_uid,priority,time_s")
    assert any("rs-000002-bbb-0" in ln for ln in plines[1:])
    assert not any("rs-000001-aaa-0" in ln for ln in plines[1:])


def test_monitor_loop_deletions_count_running_to_pending_and_disappearance(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.clock = FakeClock(0.0)
    rp.max_prio = 1
    rp.run_start_monotonic = 0.0
    rp.run_start_wall = 0.0

    rp.prio_by_rs = {"rs-000002": 1}
    rp.initial_rs_names = set()

    # Sequence:
    #  1) u2 Running           -> eligible
    #  2) u2 Pending           -> count +1 (Running->Pending)
    #  3) u2 Running           -> eligible again
    #  4) u2 Pending           -> count +1 (Running->Pending)
    snapshots = [
        [
            {"metadata": {"name": "rs-000002-bbb-0", "uid": "u2"}, "status": {"phase": "Running"}},
        ],
        [
            {"metadata": {"name": "rs-000002-bbb-0", "uid": "u2"}, "status": {"phase": "Pending"}},
        ],
        [
            {"metadata": {"name": "rs-000002-bbb-0", "uid": "u2"}, "status": {"phase": "Running"}},
        ],
        [
            {"metadata": {"name": "rs-000002-bbb-0", "uid": "u2"}, "status": {"phase": "Pending"}},
        ],
    ]

    call_i = {"i": 0}
    stop_event = tr.threading.Event()

    def snap(_ns):
        i = call_i["i"]
        if i >= len(snapshots):
            stop_event.set()
            return (0.0, 0.0, 0.0, 0.0, {1: 0}, {1: 0}, [])
        items = snapshots[i]
        call_i["i"] += 1
        if call_i["i"] >= len(snapshots):
            stop_event.set()
        # utilization + counts are irrelevant for deletion metric assertion
        return (0.0, 0.0, 0.0, 0.0, {1: 0}, {1: 0}, items)

    monkeypatch.setattr(rp, "snapshot_from_pods", snap)
    monkeypatch.setattr(tr, "get_timestamp", lambda: "T")

    general_csv = tmp_path / "general.csv"
    pod_csv = tmp_path / "pod.csv"

    rp.monitor_loop(
        namespace="trace",
        interval_s=0.0,
        general_csv=general_csv,
        pod_stats_csv=pod_csv,
        stop_event=stop_event,
    )

    glines = general_csv.read_text(encoding="utf-8").splitlines()
    assert len(glines) >= 2  # header + at least one row

    header = glines[0].split(",")
    idx_del = header.index("deletions_cum_p1")
    last_row = glines[-1].split(",")
    assert last_row[idx_del] == "2"


# ---------------------------------------------------------------------------
# TraceReplayer.run_seed()
# ---------------------------------------------------------------------------

def test_run_orchestrates_cluster_setup_and_saves_logs(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch, save_scheduler_logs=True)
    rp.clock = FakeClock(0.0)

    # Make load_initial_and_trace cheap + deterministic
    def fake_load():
        rp.trace_pods = [tr.TraceRecord(id=1, start_time=0.0, end_time=0.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
        rp.initial_pods = []
        rp.num_nodes = 1
        rp.trace_time_s = 0.0
        rp.max_prio = 1

    monkeypatch.setattr(rp, "load_initial_and_trace", fake_load)

    monkeypatch.setattr(tr, "qty_to_mcpu_int", lambda _q: 1000)
    monkeypatch.setattr(tr, "qty_to_bytes_int", lambda _q: 1024)

    called = {k: 0 for k in ["build_events", "ensure_cluster", "create_nodes", "ensure_ns", "ensure_pcs", "apply_initial", "replay", "save_logs"]}

    monkeypatch.setattr(rp, "build_trace_events", lambda: called.__setitem__("build_events", called["build_events"] + 1))
    monkeypatch.setattr(rp, "apply_initial_workload", lambda *_a, **_k: called.__setitem__("apply_initial", called["apply_initial"] + 1))
    monkeypatch.setattr(rp, "replay_trace_events", lambda **_k: called.__setitem__("replay", called["replay"] + 1))
    monkeypatch.setattr(rp, "save_scheduler_logs", lambda: called.__setitem__("save_logs", called["save_logs"] + 1))

    monkeypatch.setattr(tr.yaml, "safe_load", lambda *_a, **_k: {"x": 1})
    monkeypatch.setattr(tr, "merge_kwokctl_envs", lambda doc, envs: doc)

    monkeypatch.setattr(tr, "ensure_kwok_cluster", lambda **_k: called.__setitem__("ensure_cluster", called["ensure_cluster"] + 1))
    monkeypatch.setattr(tr, "create_kwok_nodes", lambda **_k: called.__setitem__("create_nodes", called["create_nodes"] + 1))
    monkeypatch.setattr(tr, "ensure_namespace", lambda *_a, **_k: called.__setitem__("ensure_ns", called["ensure_ns"] + 1))
    monkeypatch.setattr(tr, "ensure_priority_classes", lambda *_a, **_k: called.__setitem__("ensure_pcs", called["ensure_pcs"] + 1))
    monkeypatch.setattr(tr, "kwok_pods_cap", lambda: 10)

    # Avoid real monitor thread
    class DummyThread:
        def __init__(self, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.joined = False

        def start(self):
            return None

        def join(self, timeout=None):
            self.joined = True

    monkeypatch.setattr(tr.threading, "Thread", DummyThread)

    # Stabilize run_start_wall source
    monkeypatch.setattr(time, "time", lambda: 0.0)

    rp.run_seed()

    assert called["build_events"] == 1
    assert called["ensure_cluster"] == 1
    assert called["create_nodes"] == 1
    assert called["ensure_ns"] == 1
    assert called["ensure_pcs"] == 1
    assert called["apply_initial"] == 1
    assert called["replay"] == 1
    assert called["save_logs"] == 1


def test_run_finally_saves_logs_even_if_replay_raises(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch, save_scheduler_logs=True)
    rp.clock = FakeClock(0.0)

    def fake_load():
        rp.trace_pods = []
        rp.initial_pods = []
        rp.num_nodes = 1
        rp.trace_time_s = 0.0
        rp.max_prio = 1

    monkeypatch.setattr(rp, "load_initial_and_trace", fake_load)
    monkeypatch.setattr(tr, "qty_to_mcpu_int", lambda _q: 1000)
    monkeypatch.setattr(tr, "qty_to_bytes_int", lambda _q: 1024)

    monkeypatch.setattr(rp, "build_trace_events", lambda: None)
    monkeypatch.setattr(tr.yaml, "safe_load", lambda *_a, **_k: {})
    monkeypatch.setattr(tr, "ensure_kwok_cluster", lambda **_k: None)
    monkeypatch.setattr(tr, "create_kwok_nodes", lambda **_k: None)
    monkeypatch.setattr(tr, "ensure_namespace", lambda *_a, **_k: None)
    monkeypatch.setattr(tr, "ensure_priority_classes", lambda *_a, **_k: None)
    monkeypatch.setattr(tr, "kwok_pods_cap", lambda: 10)

    class DummyThread:
        def __init__(self, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            return None

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(tr.threading, "Thread", DummyThread)
    monkeypatch.setattr(rp, "apply_initial_workload", lambda *_a, **_k: None)

    monkeypatch.setattr(rp, "replay_trace_events", lambda **_k: (_ for _ in ()).throw(RuntimeError("replay failed")))

    saved = {"n": 0}
    monkeypatch.setattr(rp, "save_scheduler_logs", lambda: saved.__setitem__("n", saved["n"] + 1))

    with pytest.raises(RuntimeError):
        rp.run_seed()
    assert saved["n"] == 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_job_file_missing_raises(tmp_path: Path, monkeypatch):
    class DummyParser:
        def parse_args(self):
            return argparse.Namespace(job_file=str(tmp_path / "nope.yaml"))

    monkeypatch.setattr(tr, "build_argparser", lambda: DummyParser())
    with pytest.raises(SystemExit):
        tr.main()


def test_main_job_file_must_be_mapping(tmp_path: Path, monkeypatch):
    job = tmp_path / "job.yaml"
    job.write_text("- 1\n- 2\n", encoding="utf-8")  # YAML list

    class DummyParser:
        def parse_args(self):
            return argparse.Namespace(job_file=str(job))

    monkeypatch.setattr(tr, "build_argparser", lambda: DummyParser())
    with pytest.raises(SystemExit):
        tr.main()


def test_main_happy_path_constructs_replayer_and_runs(tmp_path: Path, monkeypatch):
    trace_dir, cfg, result_dir = mk_paths(tmp_path)

    job = tmp_path / "job.yaml"
    job.write_text(
        "\n".join(
            [
                "trace-dir: " + str(trace_dir),
                "kwokctl-config-file: " + str(cfg),
                "result-dir: " + str(result_dir),
                "override-kwokctl-envs:",
                "  - name: X",
                "    value: Y",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class DummyParser:
        def parse_args(self):
            return argparse.Namespace(
                job_file=str(job),
                trace_dir=None,
                kwokctl_config_file=None,
                result_dir=None,
                cluster_name=None,
                kwok_runtime=None,
                namespace=None,
                node_cpu=None,
                node_mem=None,
                monitor_interval=None,
                start_delay=None,
                log_level=None,
                save_scheduler_logs=None,
            )

    monkeypatch.setattr(tr, "build_argparser", lambda: DummyParser())
    monkeypatch.setattr(tr, "setup_logging", lambda **_k: None)

    ran = {"n": 0}

    def fake_run(self):
        # ensure defaults applied
        assert self.args is not None
        assert self.args.cluster_name == "kwok1"
        assert self.args.kwok_runtime == "binary"
        assert self.args.namespace == "trace"
        assert self.args.start_delay == 0.0
        assert self.args.save_scheduler_logs is False
        assert self.override_kwokctl_envs == [{"name": "X", "value": "Y"}]
        assert isinstance(self.job_doc, dict)
        ran["n"] += 1

    monkeypatch.setattr(tr.TraceReplayer, "run", fake_run)
    tr.main()
    assert ran["n"] == 1
