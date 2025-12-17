#!/usr/bin/env python3
# test_kwok_trace_replayer.py

import argparse, csv, json
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
        log_level="INFO",
        job_file=None,
        save_scheduler_logs=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base), trace_dir, cfg, result_dir


def mk_replayer(tmp_path: Path, monkeypatch, **overrides):
    args, trace_dir, cfg, result_dir = mk_args(tmp_path, **overrides)

    # Avoid noisy real logging + filesystem metadata writes unless explicitly tested.
    monkeypatch.setattr(tr.TraceReplayer, "_write_info_file", lambda self: None)
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


# =============================================================================
# build_argparser()
# =============================================================================

def test_build_argparser_parses_boolean_optional_action():
    p = tr.build_argparser()

    args = p.parse_args(["--save-scheduler-logs"])
    assert args.save_scheduler_logs is True

    args = p.parse_args(["--no-save-scheduler-logs"])
    assert args.save_scheduler_logs is False


# =============================================================================
# merge_job_fields_into_args()
# =============================================================================

def test_merge_job_fields_into_args_cli_wins_and_parses_monitor_interval():
    args = argparse.Namespace(
        trace_dir="/cli/trace",
        cluster_name=None,
        kwok_runtime=None,
        kwokctl_config_file=None,
        namespace=None,
        node_cpu=None,
        node_mem=None,
        monitor_interval=None,
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
        log_level=None,
        result_dir=None,
        save_scheduler_logs=None,
        job_file=None,
    )

    job = {
        "trace-dir": "/job/trace",
        "monitor-interval": "not-a-float",
        "save-scheduler-logs": "yes",  # strict bool only => ignored
        "override-kwokctl-envs": [],
    }

    merged, override_envs = tr.merge_job_fields_into_args(args, job)
    assert merged.trace_dir == "/job/trace"
    assert merged.monitor_interval is None
    assert merged.save_scheduler_logs is None
    assert override_envs == []


# =============================================================================
# ensure_default_args()
# =============================================================================

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
    assert out.log_level == "INFO"
    assert out.save_scheduler_logs is False
    assert Path(out.result_dir).is_absolute()


def test_ensure_default_args_missing_trace_dir_or_cfg_raises(tmp_path: Path):
    _, cfg, result_dir = mk_paths(tmp_path)

    args = argparse.Namespace(
        trace_dir=str(tmp_path / "nope"),
        kwokctl_config_file=str(cfg),
        result_dir=str(result_dir),
        cluster_name=None, kwok_runtime=None, namespace=None,
        node_cpu=None, node_mem=None, monitor_interval=None,
        log_level=None, job_file=None, save_scheduler_logs=None,
    )
    with pytest.raises(SystemExit):
        tr.ensure_default_args(args)

    trace_dir = tmp_path / "trace2"
    trace_dir.mkdir()
    args = argparse.Namespace(
        trace_dir=str(trace_dir),
        kwokctl_config_file=str(tmp_path / "missing_cfg.yaml"),
        result_dir=str(result_dir),
        cluster_name=None, kwok_runtime=None, namespace=None,
        node_cpu=None, node_mem=None, monitor_interval=None,
        log_level=None, job_file=None, save_scheduler_logs=None,
    )
    with pytest.raises(SystemExit):
        tr.ensure_default_args(args)


# =============================================================================
# TraceReplayer.__init__()
# =============================================================================

def test_trace_replayer_init_sets_paths_and_ctx(tmp_path: Path, monkeypatch):
    rp, args, trace_dir, _cfg, result_dir = mk_replayer(tmp_path, monkeypatch)

    assert rp.base_dir == trace_dir.resolve()
    assert rp.trace_path == trace_dir.resolve() / "trace.json"
    assert rp.results_dir == result_dir.resolve()
    assert rp.monitor_path == result_dir.resolve() / "results.csv"
    assert rp.ctx == f"kwok-{args.cluster_name}"
    assert rp.pods == []
    assert rp.events == []


# =============================================================================
# TraceReplayer.log_args()
# =============================================================================

def test_log_args(tmp_path: Path, monkeypatch):
    args, *_ = mk_args(tmp_path)
    monkeypatch.setattr(tr.TraceReplayer, "_write_info_file", lambda self: None)

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
    assert "save_scheduler_logs" in called["include"]


# =============================================================================
# TraceReplayer._write_info_file()
# =============================================================================

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

    # should not raise
    _ = tr.TraceReplayer(args)


# =============================================================================
# TraceReplayer._save_scheduler_logs()
# =============================================================================

def test_save_scheduler_logs(tmp_path: Path, monkeypatch):
    rp, args, *_ = mk_replayer(tmp_path, monkeypatch)

    seen = {}

    def fake_save(cluster_name, out_path, runner, logger):
        seen["cluster_name"] = cluster_name
        seen["out_path"] = out_path
        seen["runner"] = runner

    monkeypatch.setattr(tr, "save_kwok_scheduler_logs", fake_save)

    rp._save_scheduler_logs()
    assert seen["cluster_name"] == args.cluster_name
    assert seen["out_path"] == rp.results_dir / "scheduler-logs" / "sched_logs.log"
    assert seen["runner"] == rp.runner


# =============================================================================
# TraceReplayer._rs_name_for_record()
# =============================================================================

def test_rs_name_for_record():
    assert tr.TraceReplayer._rs_name_for_record(1) == "rs-000001"
    assert tr.TraceReplayer._rs_name_for_record(123456) == "rs-123456"


# =============================================================================
# TraceReplayer.load_trace()
# =============================================================================

def test_load_trace_missing_file_raises(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        rp.load_trace()


def test_load_trace_parses_and_sorts_and_meta_trace_time_fallback(tmp_path: Path, monkeypatch):
    rp, _, trace_dir, *_ = mk_replayer(tmp_path, monkeypatch)

    raw = {
        "meta": {"num_nodes": 2, "trace_time_s": "not-a-float"},
        "pods": [
            {"id": 2, "start_time": 5.0, "end_time": 6.0, "cpu": 0.2, "mem": 0.3, "priority": 2, "replicas": 1},
            {"id": 1, "start_time": 1.0, "end_time": 4.0, "cpu": 0.1, "mem": 0.2, "priority": 3, "replicas": 2},
        ],
    }
    (trace_dir / "trace.json").write_text(json.dumps(raw), encoding="utf-8")

    rp.load_trace()
    assert rp.num_nodes == 2
    assert [p.id for p in rp.pods] == [1, 2]  # sorted by start_time
    assert rp.max_prio == 3
    assert rp.t_min == 1.0
    # fallback to max(end_time)=6.0 since meta trace_time_s is invalid
    assert rp.trace_time == 6.0


def test_load_trace_requires_meta_num_nodes(tmp_path: Path, monkeypatch):
    rp, _, trace_dir, *_ = mk_replayer(tmp_path, monkeypatch)

    raw = {"meta": {}, "pods": []}
    (trace_dir / "trace.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError):
        rp.load_trace()


def test_load_trace_requires_record_keys(tmp_path: Path, monkeypatch):
    rp, _, trace_dir, *_ = mk_replayer(tmp_path, monkeypatch)

    raw = {"meta": {"num_nodes": 1}, "pods": [{"id": 1}]}  # missing keys
    (trace_dir / "trace.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError):
        rp.load_trace()


# =============================================================================
# TraceReplayer._build_events()
# =============================================================================

def test_build_events(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    # one pod with start=end -> create+delete at same sim time
    rp.pods = [
        tr.TraceRecord(id=1, start_time=1.0, end_time=1.0, cpu=0.0, mem=0.0, priority=2, replicas=0),
    ]
    rp.node_cpu_m = 1000
    rp.node_mem_b = 1000

    monkeypatch.setattr(tr, "qty_to_mcpu_str", lambda m: f"{m}m")
    monkeypatch.setattr(tr, "qty_to_bytes_str", lambda b: f"{b}")

    rp._build_events()
    assert len(rp.events) == 2
    assert rp.events[0].kind == "create"
    assert rp.events[1].kind == "delete"
    # min 1 behavior
    assert rp.events[0].cpu_str == "1m"
    assert rp.events[0].mem_str == "1"
    assert rp.events[0].replicas == 1


# =============================================================================
# TraceReplayer._replay_events()
# =============================================================================

def test_replay_events_no_events_sleeps_until_end(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.clock = FakeClock(0.0)
    rp.events = []
    rp.trace_time = 10.0

    rp._replay_events(namespace="trace", start_wall_time=0.0, sim_t0=0.0)
    assert rp.clock.sleeps == [pytest.approx(10.0)]


def test_replay_events_degenerate_end_before_start_does_not_sleep(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.clock = FakeClock(0.0)
    rp.events = []
    rp.trace_time = 0.0

    rp._replay_events(namespace="trace", start_wall_time=0.0, sim_t0=1.0)
    assert rp.clock.sleeps == []


def test_replay_events_next_batch_beyond_end_aligns_and_exits(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.clock = FakeClock(0.0)
    rp.trace_time = 2.0
    rp.events = [tr.Event(sim_time_s=5.0, kind="create", record_id=1, cpu_str="1m", mem_str="1", pc_name="p1")]

    class CapturingExec:
        def __init__(self, *a, **k):
            self.submitted = []
            self.shutdown_called = False

        def submit(self, fn, *a, **k):
            self.submitted.append((fn, a, k))
            return OkFuture()

        def shutdown(self, wait=True):
            self.shutdown_called = True

    ex = CapturingExec()
    rp.executor_factory = lambda **_k: ex

    rp._replay_events(namespace="trace", start_wall_time=0.0, sim_t0=0.0)
    assert rp.clock.sleeps == [pytest.approx(2.0)]
    assert ex.submitted == []


def test_replay_events_processes_batch_sleeps_to_batch_and_aligns_end(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.clock = FakeClock(0.0)
    rp.trace_time = 2.0
    rp.ctx = "ctx"

    rp.events = [
        tr.Event(sim_time_s=1.0, kind="create", record_id=1, cpu_str="1m", mem_str="1", pc_name="p1", replicas=2),
        tr.Event(sim_time_s=1.0, kind="delete", record_id=1),
    ]

    monkeypatch.setattr(tr, "yaml_kwok_rs", lambda **_k: "YAML")
    calls = {"apply": [], "delete": []}
    monkeypatch.setattr(tr, "kubectl_apply_yaml", lambda *a, **k: calls["apply"].append((a, k)))
    monkeypatch.setattr(tr, "delete_rs", lambda *a, **k: calls["delete"].append((a, k)))

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

    rp._replay_events(namespace="trace", start_wall_time=0.0, sim_t0=0.0)

    # sleep to batch at t=1, then align to end t=2
    assert rp.clock.sleeps == [pytest.approx(1.0), pytest.approx(1.0)]

    assert len(ex.submitted) == 2
    assert len(calls["apply"]) == 0  # executed by executor in real life; here we only capture submit
    assert len(calls["delete"]) == 0


def test_replay_events_future_exception_is_caught(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.clock = FakeClock(0.0)
    rp.trace_time = 0.0
    rp.events = [tr.Event(sim_time_s=0.0, kind="create", record_id=1, cpu_str="1m", mem_str="1", pc_name="p1")]

    monkeypatch.setattr(tr, "yaml_kwok_rs", lambda **_k: "YAML")

    class ExecBad:
        def submit(self, *_a, **_k):
            return BadFuture()

        def shutdown(self, wait=True):
            return None

    rp.executor_factory = lambda **_k: ExecBad()

    # should not raise
    rp._replay_events(namespace="trace", start_wall_time=0.0, sim_t0=0.0)


# =============================================================================
# TraceReplayer._snapshot_from_pods()
# =============================================================================

def test_snapshot_from_pods_computes_utilization_and_pending(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.num_nodes = 2
    rp.node_cpu_m = 1000
    rp.node_mem_b = 1000

    monkeypatch.setattr(tr, "qty_to_mcpu_int", lambda q: 0 if q is None else int(str(q).rstrip("m")))
    monkeypatch.setattr(tr, "qty_to_bytes_int", lambda q: 0 if q is None else int(str(q)))

    pods_json = {
        "items": [
            {
                "metadata": {"name": "a"},
                "spec": {
                    "nodeName": "n1",
                    "containers": [
                        {"resources": {"requests": {"cpu": "100m", "memory": "10"}}},
                        {"resources": {"requests": {"cpu": "200m", "memory": "20"}}},
                    ],
                },
                "status": {"phase": "Running"},
            },
            {
                "metadata": {"name": "b"},
                "spec": {"containers": [{"resources": {"requests": {"cpu": "50m", "memory": "5"}}}]},
                "status": {"phase": "Pending"},
            },
            {
                "metadata": {"name": "c"},
                "spec": {"containers": [{"resources": {"requests": {}}}]},
                "status": {"phase": "Succeeded"},
            },
        ]
    }
    monkeypatch.setattr(tr, "get_json_ctx", lambda *_a, **_k: pods_json)

    cpu_u, mem_u, running, pending = rp._snapshot_from_pods("trace")
    assert running == [("a", "n1")]
    assert pending == ["b"]
    assert cpu_u == pytest.approx(0.15)   # 300 / (2*1000)
    assert mem_u == pytest.approx(0.015)  # 30 / (2*1000)


def test_snapshot_from_pods_zero_capacity_returns_zero(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    rp.num_nodes = 0
    rp.node_cpu_m = 0
    rp.node_mem_b = 0
    monkeypatch.setattr(tr, "get_json_ctx", lambda *_a, **_k: {"items": []})

    cpu_u, mem_u, running, pending = rp._snapshot_from_pods("trace")
    assert cpu_u == 0.0
    assert mem_u == 0.0
    assert running == []
    assert pending == []


# =============================================================================
# TraceReplayer._monitor_loop()
# =============================================================================

def test_monitor_loop_writes_csv_and_counts_by_priority_and_identity_fallback(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)

    rp.clock = FakeClock(0.0)

    # include RS-style + fallback identity stripping ("custom-foo-0" -> "custom-foo")
    def snap(_ns):
        return (
            0.25,
            0.50,
            [("rs-000001-0", "n1"), ("custom-foo-0", "n2")],
            ["rs-000001-1", "unknown-zzz"],
        )

    monkeypatch.setattr(rp, "_snapshot_from_pods", snap)
    monkeypatch.setattr(tr, "get_timestamp", lambda: "T")

    stop_event = tr.threading.Event()

    def sleep_and_stop(_s: float):
        stop_event.set()

    rp.clock.sleep = sleep_and_stop  # stop after 1 tick

    out_csv = tmp_path / "m.csv"
    rp._monitor_loop(
        namespace="trace",
        interval_s=10.0,
        max_prio=2,
        prio_by_identity={"rs-000001": 2, "custom-foo": 1},
        out_csv=out_csv,
        stop_event=stop_event,
        start_wall_time=0.0,
        sim_t0=0.0,
    )

    lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp,wall_time_s,sim_time_s,cpu_run_util,mem_run_util")
    row = next(csv.reader([lines[1]]))
    assert row[0] == "T"
    assert row[5] == "2"  # running_count
    assert row[6] == "2"  # unsched_count

    running_by_prio = json.loads(row[7])
    unsched_by_prio = json.loads(row[8])
    assert running_by_prio == {"p1": 1, "p2": 1}
    assert unsched_by_prio == {"p1": 0, "p2": 1}


def test_monitor_loop_snapshot_exception_path_sleeps_and_writes_only_header(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch)
    rp.clock = FakeClock(0.0)

    stop_event = tr.threading.Event()
    calls = {"n": 0}

    def snap(_ns):
        calls["n"] += 1
        raise RuntimeError("snap failed")

    monkeypatch.setattr(rp, "_snapshot_from_pods", snap)

    def sleep_then_stop(_s: float):
        stop_event.set()

    rp.clock.sleep = sleep_then_stop

    out_csv = tmp_path / "m.csv"
    rp._monitor_loop(
        namespace="trace",
        interval_s=1.0,
        max_prio=1,
        prio_by_identity={},
        out_csv=out_csv,
        stop_event=stop_event,
        start_wall_time=0.0,
        sim_t0=0.0,
    )

    lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # header only
    assert calls["n"] == 1


# =============================================================================
# TraceReplayer.run()
# =============================================================================

def test_run_orchestrates_cluster_setup_and_saves_logs(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch, save_scheduler_logs=True)

    rp.clock = FakeClock(0.0)

    # Avoid file IO inside load_trace; set required fields
    def fake_load_trace():
        rp.pods = [tr.TraceRecord(id=1, start_time=0.0, end_time=0.0, cpu=0.1, mem=0.1, priority=1, replicas=1)]
        rp.num_nodes = 1
        rp.trace_time = 0.0
        rp.max_prio = 1
        rp.t_min = 0.0

    monkeypatch.setattr(rp, "load_trace", fake_load_trace)
    monkeypatch.setattr(tr, "qty_to_mcpu_int", lambda _q: 1000)
    monkeypatch.setattr(tr, "qty_to_bytes_int", lambda _q: 1024)

    called = {k: 0 for k in [
        "merge_envs", "ensure_cluster", "create_nodes", "ensure_ns", "ensure_pcs",
        "build_events", "replay", "save_logs",
    ]}

    monkeypatch.setattr(rp, "_build_events", lambda: called.__setitem__("build_events", called["build_events"] + 1))
    monkeypatch.setattr(rp, "_replay_events", lambda **_k: called.__setitem__("replay", called["replay"] + 1))
    monkeypatch.setattr(rp, "_save_scheduler_logs", lambda: called.__setitem__("save_logs", called["save_logs"] + 1))

    monkeypatch.setattr(tr, "merge_kwokctl_envs", lambda doc, envs: (called.__setitem__("merge_envs", called["merge_envs"] + 1) or doc))
    monkeypatch.setattr(tr, "ensure_kwok_cluster", lambda **_k: called.__setitem__("ensure_cluster", called["ensure_cluster"] + 1))
    monkeypatch.setattr(tr, "create_kwok_nodes", lambda **_k: called.__setitem__("create_nodes", called["create_nodes"] + 1))
    monkeypatch.setattr(tr, "ensure_namespace", lambda *_a, **_k: called.__setitem__("ensure_ns", called["ensure_ns"] + 1))
    monkeypatch.setattr(tr, "ensure_priority_classes", lambda *_a, **_k: called.__setitem__("ensure_pcs", called["ensure_pcs"] + 1))
    monkeypatch.setattr(tr, "kwok_pods_cap", lambda: 10)

    monkeypatch.setattr(tr.yaml, "safe_load", lambda *_a, **_k: {"x": 1})

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

    # force env override path
    rp.override_kwokctl_envs = [{"name": "A", "value": "B"}]

    rp.run()

    assert called["merge_envs"] == 1
    assert called["ensure_cluster"] == 1
    assert called["create_nodes"] == 1
    assert called["ensure_ns"] == 1
    assert called["ensure_pcs"] == 1
    assert called["build_events"] == 1
    assert called["replay"] == 1
    assert called["save_logs"] == 1


def test_run_finally_saves_logs_even_if_replay_raises(tmp_path: Path, monkeypatch):
    rp, *_ = mk_replayer(tmp_path, monkeypatch, save_scheduler_logs=True)
    rp.clock = FakeClock(0.0)

    def fake_load_trace():
        rp.pods = []
        rp.num_nodes = 1
        rp.trace_time = 0.0
        rp.max_prio = 1
        rp.t_min = 0.0

    monkeypatch.setattr(rp, "load_trace", fake_load_trace)
    monkeypatch.setattr(tr, "qty_to_mcpu_int", lambda _q: 1000)
    monkeypatch.setattr(tr, "qty_to_bytes_int", lambda _q: 1024)
    monkeypatch.setattr(rp, "_build_events", lambda: None)
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

    monkeypatch.setattr(rp, "_replay_events", lambda **_k: (_ for _ in ()).throw(RuntimeError("replay failed")))

    saved = {"n": 0}
    monkeypatch.setattr(rp, "_save_scheduler_logs", lambda: saved.__setitem__("n", saved["n"] + 1))

    with pytest.raises(RuntimeError):
        rp.run()
    assert saved["n"] == 1


# =============================================================================
# main()
# =============================================================================

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
        "\n".join([
            "trace-dir: " + str(trace_dir),
            "kwokctl-config-file: " + str(cfg),
            "result-dir: " + str(result_dir),
            "override-kwokctl-envs:",
            "  - name: X",
            "    value: Y",
        ]) + "\n",
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
                log_level=None,
                save_scheduler_logs=None,
            )

    monkeypatch.setattr(tr, "build_argparser", lambda: DummyParser())
    monkeypatch.setattr(tr, "setup_logging", lambda **_k: None)

    ran = {"n": 0}

    class DummyReplayer:
        def __init__(self, args, job_doc=None, override_kwokctl_envs=None):
            # ensure defaults have been applied
            assert args.cluster_name == "kwok1"
            assert args.kwok_runtime == "binary"
            assert args.namespace == "trace"
            assert args.save_scheduler_logs is False
            assert override_kwokctl_envs == [{"name": "X", "value": "Y"}]
            assert isinstance(job_doc, dict)

        def run(self):
            ran["n"] += 1

    monkeypatch.setattr(tr, "TraceReplayer", DummyReplayer)
    tr.main()
    assert ran["n"] == 1
