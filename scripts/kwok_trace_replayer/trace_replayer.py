#!/usr/bin/env python3
# trace_replayer.py

#######################################################################
"""
python -m scripts.kwok_trace_replayer.trace_replayer \
--job-file <job-file.yaml> \
--result-dir <result-dir> \
--trace-dir <trace-dir> \
--kwokctl-config-file <kwokctl-config.yaml> \
--cluster-name <kwok-cluster-name> \
--namespace <k8s-namespace> \
--node-cpu <k8s-cpu-quantity> \
--node-mem <k8s-mem-quantity> \
--monitor-interval <seconds> \
--start-delay <seconds> \
--log-level <log-level> \
--save-scheduler-logs <True|False>
"""
#######################################################################

import argparse, csv, json, logging, threading, time, yaml, subprocess, re
from argparse import BooleanOptionalAction
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple, Set

from scripts.helpers.job_helpers import (
    JobField,
    merge_job_fields_into_args as _merge_job_fields_into_args,
    parse_optional_str,
    parse_optional_float,
)
from scripts.helpers.general_helpers import (
    setup_logging,
    make_header_footer,
    get_timestamp,
    qty_to_mcpu_int,
    qty_to_mcpu_str,
    qty_to_bytes_int,
    qty_to_bytes_str,
    build_cli_cmd,
    write_info_file,
    SystemClock,
    Runner,
    Clock,
    log_args_block,
)
from scripts.helpers.kubectl_helpers import (
    kubectl_apply_yaml,
    ensure_namespace,
    ensure_priority_classes,
    delete_rs,
    get_json_ctx,
)
from scripts.helpers.kwokctl_helpers import (
    yaml_kwok_rs,
    create_kwok_nodes,
    ensure_kwok_cluster,
    kwok_pods_cap,
    merge_kwokctl_envs,
    save_kwok_scheduler_logs,
)
from scripts.kwok_trace_replayer.trace_helpers import TraceRecord

#######################################################################
# Constants
#######################################################################
MAX_REPLAY_WORKERS = 5
_RS_PREFIX_RE = re.compile(r"^(rs-\d{6})(?:-.*)?$")

LOGGER_NAME = "trace-replayer"
LOG = logging.getLogger(LOGGER_NAME)


def _parse_optional_bool_strict(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _rs_prefix_from_pod_name(pod_name: str) -> str:
    """
    Extract "rs-000001" from a pod name like "rs-000001-<hash>-<suffix>".
    """
    m = _RS_PREFIX_RE.match(pod_name)
    if m:
        return m.group(1)
    if "-" in pod_name:
        return pod_name.split("-", 1)[0]
    return pod_name


def _parse_rfc3339_to_epoch(ts: str) -> Optional[float]:
    """
    Kubernetes timestamps are RFC3339, often like:
      2026-01-09T12:34:56Z
      2026-01-09T12:34:56.123Z
    """
    if not ts or not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


class _TimeClock:
    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass
class Event:
    sim_time_s: float
    kind: str  # "create" or "delete"
    record_id: int
    cpu_str: str | None = None
    mem_str: str | None = None
    pc_name: str | None = None
    replicas: int = 1


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Replay a JSON pod trace on a KWOK cluster and monitor utilization. "
            "Expects <trace-dir>/trace.json and <trace-dir>/initial.json as produced by trace_generator.py."
        )
    )

    p.add_argument("--result-dir", dest="result_dir", required=False, default=None)
    p.add_argument("--job-file", dest="job_file", default=None)
    p.add_argument("--trace-dir", dest="trace_dir", required=False, default=None)

    p.add_argument("--cluster-name", dest="cluster_name", default=None)
    p.add_argument("--kwok-runtime", dest="kwok_runtime", choices=["binary", "docker"], default=None)
    p.add_argument("--kwokctl-config-file", dest="kwokctl_config_file", required=False, default=None)
    p.add_argument("--namespace", dest="namespace", default=None)
    p.add_argument("--node-cpu", dest="node_cpu", default=None)
    p.add_argument("--node-mem", dest="node_mem", default=None)

    p.add_argument("--monitor-interval", dest="monitor_interval", type=float, default=None)
    p.add_argument("--start-delay", dest="start_delay", type=float, default=None)

    p.add_argument("--log-level", dest="log_level", default=None)

    p.add_argument(
        "--save-scheduler-logs",
        dest="save_scheduler_logs",
        action=BooleanOptionalAction,
        default=None,
    )

    return p


def merge_job_fields_into_args(
    args: argparse.Namespace,
    job: Dict[str, Any],
) -> tuple[argparse.Namespace, List[Dict[str, Any]]]:
    fields = [
        JobField("trace-dir", "trace_dir", parse=parse_optional_str),
        JobField("cluster-name", "cluster_name", parse=parse_optional_str),
        JobField("kwok-runtime", "kwok_runtime", parse=parse_optional_str),
        JobField("kwokctl-config-file", "kwokctl_config_file", parse=parse_optional_str),
        JobField("namespace", "namespace", parse=parse_optional_str),
        JobField("node-cpu", "node_cpu", parse=parse_optional_str),
        JobField("node-mem", "node_mem", parse=parse_optional_str),
        JobField("monitor-interval", "monitor_interval", parse=parse_optional_float),
        JobField("start-delay", "start_delay", parse=parse_optional_float),
        JobField("log-level", "log_level", parse=parse_optional_str),
        JobField("result-dir", "result_dir", parse=parse_optional_str),
        JobField(
            "save-scheduler-logs",
            "save_scheduler_logs",
            parse=_parse_optional_bool_strict,
            accept=lambda v: isinstance(v, bool),
        ),
    ]
    args = _merge_job_fields_into_args(args, job or {}, fields)
    override_envs = (job or {}).get("override-kwokctl-envs") or []
    return args, override_envs


def ensure_default_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "cluster_name", None) is None:
        args.cluster_name = "kwok1"
    if getattr(args, "kwok_runtime", None) is None:
        args.kwok_runtime = "binary"
    if getattr(args, "namespace", None) is None:
        args.namespace = "trace"
    if getattr(args, "node_cpu", None) is None:
        args.node_cpu = "1000m"
    if getattr(args, "node_mem", None) is None:
        args.node_mem = "1Gi"
    if getattr(args, "monitor_interval", None) is None:
        args.monitor_interval = 1.0
    if getattr(args, "start_delay", None) is None:
        args.start_delay = 0.0
    if getattr(args, "log_level", None) is None:
        args.log_level = "INFO"
    if getattr(args, "job_file", None) is None:
        args.job_file = None
    if getattr(args, "save_scheduler_logs", None) is None:
        args.save_scheduler_logs = False

    if not getattr(args, "trace_dir", None):
        raise SystemExit("--trace-dir (or trace-dir in job-file) is required")
    if not getattr(args, "kwokctl_config_file", None):
        raise SystemExit("--kwokctl-config-file (or kwokctl-config-file in job-file) is required")
    if not getattr(args, "result_dir", None):
        raise SystemExit("--result-dir (or result-dir in job-file) is required")

    trace_dir = Path(args.trace_dir).resolve()
    if not trace_dir.exists():
        raise SystemExit(f"--trace-dir not found: {trace_dir}")

    kwok_cfg = Path(args.kwokctl_config_file).resolve()
    if not kwok_cfg.exists():
        raise SystemExit(f"--kwokctl-config-file not found: {kwok_cfg}")

    args.result_dir = str(Path(args.result_dir).resolve())
    return args


class TraceReplayer:
    """
    Replay a trace on a KWOK cluster and monitor utilization.
    """
    def __init__(
        self,
        args: argparse.Namespace,
        job_doc: Dict[str, Any] | None = None,
        override_kwokctl_envs: List[Dict[str, Any]] | None = None,
        runner: Runner = subprocess.run,
        clock: Clock | None = None,
        executor_factory: Callable[..., Any] = ThreadPoolExecutor,
    ) -> None:
        self.args = args
        self.job_doc: Dict[str, Any] = job_doc or {}
        self.override_kwokctl_envs: List[Dict[str, Any]] = list(override_kwokctl_envs or [])

        self.base_dir: Path = Path(args.trace_dir).resolve()
        self.trace_path: Path = self.base_dir / "trace.json"
        self.initial_path: Path = self.base_dir / "initial.json"
        self.info_generate_path: Path = self.base_dir / "info_generate.yaml"

        self.results_dir: Path = Path(self.args.result_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.general_stats_path = self.results_dir / "general_stats.csv"
        self.pod_stats_path = self.results_dir / "pod_stats.csv"

        self.trace_pods: List[TraceRecord] = []
        self.initial_pods: List[TraceRecord] = []

        self.num_nodes: int = 0
        self.max_prio: int = 0
        self.trace_time_s: float = 0.0

        self.node_cpu_m: int = 0
        self.node_mem_b: int = 0

        self.events: List[Event] = []

        # rs -> priority (include BOTH initial + trace)
        self.prio_by_rs: Dict[str, int] = {}

        # rs names that belong to initial workload (skip these in pod_stats.csv)
        self.initial_rs_names: Set[str] = set()

        # Runner + clock + executor factory
        self.ctx: str = f"kwok-{args.cluster_name}"
        self.runner = runner
        self.clock = clock or SystemClock()
        if not (hasattr(self.clock, "time") and hasattr(self.clock, "sleep")):
            self.clock = _TimeClock()
        self.executor_factory = executor_factory

        # Run start epoch/time base (set in run())
        self.run_start_wall: float = 0.0       # epoch seconds
        self.run_start_monotonic: float = 0.0  # from clock.time()

        LOG.info("logging arguments and git info to results_dir...")
        self._write_info_file()
        self.log_args()

    def log_args(self) -> None:
        include = [
            "trace_dir",
            "cluster_name",
            "kwok_runtime",
            "kwokctl_config_file",
            "namespace",
            "node_cpu",
            "node_mem",
            "monitor_interval",
            "start_delay",
            "result_dir",
            "log_level",
            "save_scheduler_logs",
            "job_file",
        ]
        log_args_block(LOG, self.args, title="ARGS", include=include)

    def _write_info_file(self) -> None:
        try:
            out_path = self.results_dir / "info_replayer.yaml"
            meta_extra = {
                "kind": "trace_replayer",
                "job_file": getattr(self.args, "job_file", None),
                "kwokctl_config_file": self.args.kwokctl_config_file,
            }
            inputs = {
                "cli-cmd": build_cli_cmd(),
                "args": {k: v for k, v in vars(self.args).items()},
                "job": self.job_doc or {},
            }
            write_info_file(out_path, meta_extra=meta_extra, inputs=inputs, logger=LOG)
        except Exception as e:
            LOG.warning("failed to write info_replayer.yaml: %s", e)

    def _save_scheduler_logs(self) -> None:
        sched_dir = self.results_dir / "scheduler-logs"
        out_path = sched_dir / "sched_logs.log"
        save_kwok_scheduler_logs(self.args.cluster_name, out_path, runner=self.runner, logger=LOG)

    @staticmethod
    def _rs_name_for_record(record_id: int) -> str:
        return f"rs-{record_id:06d}"

    def _load_generate_info(self) -> Dict[str, Any]:
        if not self.info_generate_path.exists():
            LOG.warning("info_generate.yaml not found at %s (will infer from JSON)", self.info_generate_path)
            return {}
        try:
            with open(self.info_generate_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
            return doc if isinstance(doc, dict) else {}
        except Exception as e:
            LOG.warning("failed to read info_generate.yaml: %s", e)
            return {}

    def _load_json_pods(self, path: Path) -> List[TraceRecord]:
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw.get("pods", []) or []

        def req(rec: dict, key: str) -> Any:
            if key not in rec:
                raise ValueError(f"trace record missing required key '{key}': {rec}")
            return rec[key]

        pods: List[TraceRecord] = []
        for rec in records:
            pods.append(
                TraceRecord(
                    id=int(req(rec, "id")),
                    start_time=float(req(rec, "start_time")),
                    end_time=float(req(rec, "end_time")),
                    cpu=float(req(rec, "cpu")),
                    mem=float(req(rec, "mem")),
                    priority=int(req(rec, "priority")),
                    replicas=int(req(rec, "replicas")),
                )
            )
        pods.sort(key=lambda p: p.start_time)
        return pods

    def load_initial_and_trace(self) -> None:
        gen_info = self._load_generate_info()

        self.trace_pods = self._load_json_pods(self.trace_path)
        self.initial_pods = self._load_json_pods(self.initial_path) if self.initial_path.exists() else []

        # Prefer info_generate.yaml for these if present; otherwise infer
        num_nodes = None
        trace_time_s = None
        max_prio = None

        def _deep_get(d: Dict[str, Any], path: List[str]) -> Any:
            cur: Any = d
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
            return cur

        for candidate in [
            _deep_get(gen_info, ["inputs", "generated", "num_nodes"]),
            _deep_get(gen_info, ["generated", "num_nodes"]),
        ]:
            if isinstance(candidate, int):
                num_nodes = candidate

        for candidate in [
            _deep_get(gen_info, ["inputs", "generated", "trace_time_s"]),
            _deep_get(gen_info, ["generated", "trace_time_s"]),
        ]:
            if isinstance(candidate, (int, float)):
                trace_time_s = float(candidate)

        for candidate in [
            _deep_get(gen_info, ["inputs", "generated", "max_priority_seen"]),
            _deep_get(gen_info, ["generated", "max_priority_seen"]),
        ]:
            if isinstance(candidate, int):
                max_prio = candidate

        if num_nodes is None:
            LOG.warning("num_nodes not found in info_generate.yaml; falling back to --num-nodes assumption (8)")
            num_nodes = 8

        if trace_time_s is None:
            tmax = 0.0
            for p in (self.initial_pods + self.trace_pods):
                tmax = max(tmax, float(p.end_time))
            trace_time_s = float(tmax)

        if max_prio is None:
            m = 0
            for p in (self.initial_pods + self.trace_pods):
                m = max(m, int(p.priority))
            max_prio = int(m)

        self.num_nodes = int(num_nodes)
        self.trace_time_s = float(trace_time_s)
        self.max_prio = int(max_prio)

        LOG.info(
            "loaded trace=%d records (%s), initial=%d records (%s), trace_time_s=%.3f, max_prio=%d, num_nodes=%d",
            len(self.trace_pods),
            self.trace_path,
            len(self.initial_pods),
            self.initial_path,
            self.trace_time_s,
            self.max_prio,
            self.num_nodes,
        )

        # Build rs->priority mapping for both initial + trace
        for p in self.initial_pods:
            rs = self._rs_name_for_record(p.id)
            self.prio_by_rs[rs] = int(p.priority)
            self.initial_rs_names.add(rs)

        for p in self.trace_pods:
            rs = self._rs_name_for_record(p.id)
            self.prio_by_rs[rs] = int(p.priority)

    def _build_trace_events(self) -> None:
        events: List[Event] = []
        for p in self.trace_pods:
            cpu_m = max(1, int(round(p.cpu * self.node_cpu_m)))
            mem_b = max(1, int(round(p.mem * self.node_mem_b)))
            cpu_str = qty_to_mcpu_str(cpu_m)
            mem_str = qty_to_bytes_str(mem_b)
            pc_name = f"p{int(p.priority)}"
            replicas = max(1, int(getattr(p, "replicas", 1)))

            events.append(
                Event(
                    sim_time_s=float(p.start_time),
                    kind="create",
                    record_id=p.id,
                    cpu_str=cpu_str,
                    mem_str=mem_str,
                    pc_name=pc_name,
                    replicas=replicas,
                )
            )
            events.append(Event(sim_time_s=float(p.end_time), kind="delete", record_id=p.id))

        events.sort(key=lambda e: (e.sim_time_s, 0 if e.kind == "create" else 1))
        self.events = events
        LOG.info("built %d trace events from %d trace pods", len(events), len(self.trace_pods))

    def _apply_initial_workload(self, namespace: str) -> None:
        header, footer = make_header_footer("APPLY INITIAL WORKLOAD")
        LOG.info("\n%s\nstart_wall=%s initial_records=%d\n%s", header, get_timestamp(), len(self.initial_pods), footer)

        if not self.initial_pods:
            LOG.info("no initial pods; skipping initial apply")
            return

        executor = self.executor_factory(max_workers=MAX_REPLAY_WORKERS)
        futures: List[Future] = []

        try:
            for p in self.initial_pods:
                cpu_m = max(1, int(round(p.cpu * self.node_cpu_m)))
                mem_b = max(1, int(round(p.mem * self.node_mem_b)))
                cpu_str = qty_to_mcpu_str(cpu_m)
                mem_str = qty_to_bytes_str(mem_b)
                pc_name = f"p{int(p.priority)}"
                replicas = max(1, int(getattr(p, "replicas", 1)))

                rs_name = self._rs_name_for_record(p.id)
                yaml_text = yaml_kwok_rs(
                    ns=namespace,
                    rs_name=rs_name,
                    replicas=replicas,
                    cpu=cpu_str,
                    mem=mem_str,
                    pc=pc_name,
                )
                LOG.info(
                    "INITIAL CREATE: rs=%s (id=%d) replicas=%d cpu=%s mem=%s pc=%s",
                    rs_name,
                    p.id,
                    replicas,
                    cpu_str,
                    mem_str,
                    pc_name,
                )
                futures.append(executor.submit(kubectl_apply_yaml, LOG, self.ctx, yaml_text))

        finally:
            for fut in futures:
                try:
                    fut.result()
                except Exception as e:
                    LOG.error("initial kubectl task failed: %s", e)
            try:
                executor.shutdown(wait=True)
            except Exception:
                pass

        LOG.info("initial workload applied (kubectl tasks completed)")

    def _replay_trace_events(self, namespace: str, trace_start_wall: float) -> None:
        header, footer = make_header_footer("TRACE REPLAY")
        LOG.info(
            "\n%s\ntrace_start_wall=%s trace_end_s=%.3f\n%s",
            header,
            get_timestamp(),
            self.trace_time_s,
            footer,
        )

        events = self.events
        num_events = len(events)
        trace_end_s = float(self.trace_time_s)

        if num_events == 0:
            LOG.info("no events in trace; sleeping to trace_end_s=%.3f", trace_end_s)
            target_wall_end = trace_start_wall + trace_end_s
            now = float(self.clock.time())
            sleep_s = max(0.0, target_wall_end - now)
            if sleep_s > 0:
                self.clock.sleep(sleep_s)
            return

        idx = 0
        executor = self.executor_factory(max_workers=MAX_REPLAY_WORKERS)
        futures: List[Future] = []

        try:
            while idx < num_events:
                current_t = events[idx].sim_time_s
                if current_t > trace_end_s:
                    break

                batch_events: List[Event] = []
                while idx < num_events and events[idx].sim_time_s == current_t:
                    batch_events.append(events[idx])
                    idx += 1

                target_wall = trace_start_wall + max(0.0, current_t)
                now_before = float(self.clock.time())
                sleep_s = max(0.0, target_wall - now_before)
                if sleep_s > 0:
                    self.clock.sleep(sleep_s)

                creates = [ev for ev in batch_events if ev.kind == "create"]
                deletes = [ev for ev in batch_events if ev.kind == "delete"]

                for ev in creates:
                    assert ev.cpu_str is not None and ev.mem_str is not None and ev.pc_name is not None
                    rs_name = self._rs_name_for_record(ev.record_id)
                    yaml_text = yaml_kwok_rs(
                        ns=namespace,
                        rs_name=rs_name,
                        replicas=ev.replicas,
                        cpu=ev.cpu_str,
                        mem=ev.mem_str,
                        pc=ev.pc_name,
                    )
                    LOG.info(
                        "CREATE @ sim_t=%.3f: rs=%s (id=%d) replicas=%d cpu=%s mem=%s pc=%s",
                        ev.sim_time_s,
                        rs_name,
                        ev.record_id,
                        ev.replicas,
                        ev.cpu_str,
                        ev.mem_str,
                        ev.pc_name,
                    )
                    futures.append(executor.submit(kubectl_apply_yaml, LOG, self.ctx, yaml_text))

                for ev in deletes:
                    rs_name = self._rs_name_for_record(ev.record_id)
                    LOG.info("DELETE @ sim_t=%.3f: rs=%s (id=%d)", ev.sim_time_s, rs_name, ev.record_id)
                    futures.append(executor.submit(delete_rs, LOG, self.ctx, namespace, rs_name))

            # Align to end
            target_wall_end = trace_start_wall + trace_end_s
            now = float(self.clock.time())
            sleep_s = max(0.0, target_wall_end - now)
            if sleep_s > 0:
                self.clock.sleep(sleep_s)

        finally:
            for fut in futures:
                try:
                    fut.result()
                except Exception as e:
                    LOG.error("kubectl task failed: %s", e)
            try:
                executor.shutdown(wait=True)
            except Exception:
                pass
            LOG.info("all kubectl tasks completed")

    # ------------------------------
    # Monitoring
    # ------------------------------
    def _time_s(self) -> float:
        return float(self.clock.time()) - float(self.run_start_monotonic)

    def _snapshot_from_pods(self, ns: str) -> Tuple[float, float, Dict[int, int], Dict[int, int], List[Dict[str, Any]]]:
        """
        Returns:
            cpu_run_util
            mem_run_util
            running_count_by_prio
            unsched_count_by_prio (Pending)
            pod_items (raw items)
        """
        pods_json = get_json_ctx(self.ctx, ["-n", ns, "get", "pods", "-o", "json"])
        items = pods_json.get("items", []) or []

        total_cpu_m = 0
        total_mem_b = 0

        running_by_prio: Dict[int, int] = {p: 0 for p in range(1, self.max_prio + 1)}
        pending_by_prio: Dict[int, int] = {p: 0 for p in range(1, self.max_prio + 1)}

        for pod in items:
            meta = pod.get("metadata", {}) or {}
            spec = pod.get("spec", {}) or {}
            status = pod.get("status", {}) or {}
            pod_name = meta.get("name", "")
            phase = status.get("phase", "")

            rs_name = _rs_prefix_from_pod_name(pod_name)
            prio = int(self.prio_by_rs.get(rs_name, 0))

            if phase == "Running":
                if prio in running_by_prio:
                    running_by_prio[prio] += 1

                containers = spec.get("containers", []) or []
                for c in containers:
                    res = (c.get("resources") or {}).get("requests", {}) or {}
                    cpu_q = res.get("cpu")
                    mem_q = res.get("memory")
                    total_cpu_m += qty_to_mcpu_int(cpu_q)
                    total_mem_b += qty_to_bytes_int(mem_q)

            elif phase == "Pending":
                if prio in pending_by_prio:
                    pending_by_prio[prio] += 1

        cpu_capacity_m = self.num_nodes * self.node_cpu_m
        mem_capacity_b = self.num_nodes * self.node_mem_b
        cpu_run_util = (total_cpu_m / cpu_capacity_m) if cpu_capacity_m > 0 else 0.0
        mem_run_util = (total_mem_b / mem_capacity_b) if mem_capacity_b > 0 else 0.0
        return cpu_run_util, mem_run_util, running_by_prio, pending_by_prio, items

    def _pod_start_epoch(self, pod: Dict[str, Any]) -> Optional[float]:
        status = pod.get("status", {}) or {}
        st = status.get("startTime")
        return _parse_rfc3339_to_epoch(st) if isinstance(st, str) else None

    def _pod_creation_epoch(self, pod: Dict[str, Any]) -> Optional[float]:
        meta = pod.get("metadata", {}) or {}
        ts = meta.get("creationTimestamp")
        return _parse_rfc3339_to_epoch(ts) if isinstance(ts, str) else None

    def _monitor_loop(
        self,
        namespace: str,
        interval_s: float,
        general_csv: Path,
        pod_stats_csv: Path,
        stop_event: threading.Event,
    ) -> None:
        general_csv.parent.mkdir(parents=True, exist_ok=True)
        pod_stats_csv.parent.mkdir(parents=True, exist_ok=True)

        LOG.info("monitor: writing %s (interval=%.3fs)", general_csv, interval_s)
        LOG.info("monitor: writing %s (apply-time + running-time; trace pods only)", pod_stats_csv)

        # pod_stats: only for trace pods (skip initial rs)
        seen_apply: set[str] = set()     # pod_name
        seen_running: set[str] = set()   # pod_name

        # deletion proxy: UID disappears between snapshots
        prev_live_uids: set[str] = set()
        uid_to_prio: Dict[str, int] = {}  # last known prio for that uid (best-effort)

        deletions_cum_by_prio: Dict[int, int] = {p: 0 for p in range(1, self.max_prio + 1)}

        with open(general_csv, "w", encoding="utf-8", newline="") as f_ts, open(
            pod_stats_csv, "w", encoding="utf-8", newline=""
        ) as f_ps:
            tsw = csv.writer(f_ts)
            psw = csv.writer(f_ps)

            # general_stats.csv header
            header = ["timestamp", "time_s", "cpu_run_util", "mem_run_util"]
            for p in range(1, self.max_prio + 1):
                header.append(f"running_p{p}")
            for p in range(1, self.max_prio + 1):
                header.append(f"unsched_p{p}")

            # only cumulative deletions per priority (as requested)
            for p in range(1, self.max_prio + 1):
                header.append(f"deletions_cum_p{p}")

            tsw.writerow(header)

            # pod_stats.csv header (only apply-time and running-time)
            psw.writerow(["timestamp", "event", "pod_name", "pod_uid", "priority", "time_s"])

            while not stop_event.is_set():
                loop_start = float(self.clock.time())
                now_ts = get_timestamp()
                t_s = self._time_s()

                try:
                    cpu_run_util, mem_run_util, running_by_prio, pending_by_prio, pod_items = self._snapshot_from_pods(
                        namespace
                    )
                except Exception as e:
                    LOG.warning("monitor: snapshot failed: %s", e)
                    self.clock.sleep(interval_s)
                    continue

                # ---- deletion proxy (UID disappearance) ----
                cur_live_uids: set[str] = set()
                for pod in pod_items:
                    meta = pod.get("metadata", {}) or {}
                    pod_uid = meta.get("uid", "")
                    pod_name = meta.get("name", "")
                    if not pod_uid or not pod_name:
                        continue
                    cur_live_uids.add(pod_uid)

                    rs_name = _rs_prefix_from_pod_name(pod_name)
                    prio = int(self.prio_by_rs.get(rs_name, 0))
                    if prio > 0:
                        uid_to_prio[pod_uid] = prio

                gone = prev_live_uids - cur_live_uids
                if gone:
                    delta_by_prio: Dict[int, int] = {p: 0 for p in range(1, self.max_prio + 1)}
                    for uid in gone:
                        p = int(uid_to_prio.get(uid, 0))
                        if p in delta_by_prio:
                            delta_by_prio[p] += 1
                    for p in range(1, self.max_prio + 1):
                        deletions_cum_by_prio[p] += delta_by_prio[p]

                prev_live_uids = cur_live_uids

                # ---- write general_stats row ----
                row = [now_ts, f"{t_s:.6f}", f"{cpu_run_util:.6f}", f"{mem_run_util:.6f}"]
                for p in range(1, self.max_prio + 1):
                    row.append(str(int(running_by_prio.get(p, 0))))
                for p in range(1, self.max_prio + 1):
                    row.append(str(int(pending_by_prio.get(p, 0))))
                for p in range(1, self.max_prio + 1):
                    row.append(str(int(deletions_cum_by_prio.get(p, 0))))

                tsw.writerow(row)
                f_ts.flush()

                # ---- pod_stats: apply-time + running-time (trace pods only) ----
                for pod in pod_items:
                    meta = pod.get("metadata", {}) or {}
                    status = pod.get("status", {}) or {}
                    pod_uid = meta.get("uid", "")
                    pod_name = meta.get("name", "")
                    if not pod_name:
                        continue

                    rs_name = _rs_prefix_from_pod_name(pod_name)
                    if rs_name in self.initial_rs_names:
                        # skip initial pods entirely in pod_stats.csv
                        continue

                    prio = int(self.prio_by_rs.get(rs_name, 0))

                    # APPLY: first time we observe the pod at all
                    if pod_name not in seen_apply:
                        seen_apply.add(pod_name)
                        ce = self._pod_creation_epoch(pod)
                        if ce is not None and self.run_start_wall > 0:
                            apply_time_s = max(0.0, ce - self.run_start_wall)
                        else:
                            apply_time_s = float(t_s)
                        psw.writerow([now_ts, "apply-time", pod_name, pod_uid, prio, f"{apply_time_s:.6f}"])
                        f_ps.flush()

                    # RUNNING: first time we see it Running
                    phase = status.get("phase", "")
                    if phase == "Running" and pod_name not in seen_running:
                        seen_running.add(pod_name)
                        st = self._pod_start_epoch(pod)  # status.startTime only
                        if st is not None and self.run_start_wall > 0:
                            run_time_s = max(0.0, st - self.run_start_wall)
                        else:
                            run_time_s = float(t_s)
                        psw.writerow([now_ts, "running-time", pod_name, pod_uid, prio, f"{run_time_s:.6f}"])
                        f_ps.flush()

                elapsed = float(self.clock.time()) - loop_start
                sleep_s = max(0.0, interval_s - elapsed)
                if sleep_s > 0:
                    self.clock.sleep(sleep_s)

        LOG.info("monitor: stop signal received; exiting")

    def run(self) -> None:
        # Load trace inputs
        self.load_initial_and_trace()

        # Node capacities
        self.node_cpu_m = qty_to_mcpu_int(self.args.node_cpu)
        self.node_mem_b = qty_to_bytes_int(self.args.node_mem)
        LOG.info(
            "per-node capacity: cpu_m=%d mem_bytes=%d (num_nodes=%d)",
            self.node_cpu_m,
            self.node_mem_b,
            self.num_nodes,
        )

        # Build trace events (needs node_* set)
        self._build_trace_events()

        # Create KWOK cluster
        kwok_cfg_path = Path(self.args.kwokctl_config_file).resolve()
        with open(kwok_cfg_path, "r", encoding="utf-8") as f:
            config_doc = yaml.safe_load(f) or {}

        if self.override_kwokctl_envs:
            config_doc = merge_kwokctl_envs(config_doc, self.override_kwokctl_envs)

        ensure_kwok_cluster(
            logger=LOG,
            cluster_name=self.args.cluster_name,
            kwok_runtime=self.args.kwok_runtime,
            config_doc=config_doc,
            recreate=True,
        )

        # Create KWOK nodes
        create_kwok_nodes(
            logger=LOG,
            ctx=self.ctx,
            num_nodes=self.num_nodes,
            node_cpu=self.args.node_cpu,
            node_mem=self.args.node_mem,
            pods_cap=kwok_pods_cap(),
        )

        # Namespace + PriorityClasses
        ensure_namespace(LOG, self.ctx, self.args.namespace)
        ensure_priority_classes(LOG, self.ctx, self.max_prio)

        # Start time base (time_s reference) BEFORE initial apply
        self.run_start_monotonic = float(self.clock.time())
        self.run_start_wall = time.time()
        LOG.info("run start: %s", get_timestamp())

        stop_event = threading.Event()
        monitor_thread: Optional[threading.Thread] = None

        try:
            # 1) Apply initial workload now
            self._apply_initial_workload(self.args.namespace)

            # 2) Start monitor ONLY AFTER initial load has been applied
            monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(
                    self.args.namespace,
                    float(self.args.monitor_interval),
                    self.general_stats_path,
                    self.pod_stats_path,
                    stop_event,
                ),
                daemon=True,
            )
            monitor_thread.start()

            # 3) Start-delay (wall time)
            start_delay = float(self.args.start_delay or 0.0)
            if start_delay > 0:
                LOG.info("start-delay: sleeping %.3fs before trace replay", start_delay)
                self.clock.sleep(start_delay)

            # Trace starts after delay
            trace_start_wall = float(self.clock.time())

            # 4) Replay trace aligned so sim_time 0 happens at trace_start_wall
            self._replay_trace_events(namespace=self.args.namespace, trace_start_wall=trace_start_wall)

        finally:
            stop_event.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=10.0)
            LOG.info("monitor thread joined; done.")

            if self.args.save_scheduler_logs:
                LOG.info("saving scheduler logs via kwokctl...")
                self._save_scheduler_logs()

        LOG.info("Done.")


def main() -> None:
    args = build_argparser().parse_args()

    job_doc: Dict[str, Any] | None = None
    override_kwokctl_envs: List[Dict[str, Any]] = []

    if getattr(args, "job_file", None):
        job_path = Path(args.job_file)
        if not job_path.exists():
            raise SystemExit(f"--job-file not found: {job_path}")
        try:
            with open(job_path, "r", encoding="utf-8") as f:
                job_doc = yaml.safe_load(f) or {}
            if not isinstance(job_doc, dict):
                raise SystemExit(f"--job-file must be a YAML mapping/object, got {type(job_doc).__name__}")
        except Exception as e:
            raise SystemExit(f"--job-file parse error for {job_path}: {e}")
        args, override_kwokctl_envs = merge_job_fields_into_args(args, job_doc)

    args = ensure_default_args(args)
    setup_logging(name="trace-replayer", prefix="[trace-replayer] ", level=args.log_level)

    replayer = TraceReplayer(args, job_doc=job_doc, override_kwokctl_envs=override_kwokctl_envs)
    replayer.run()


if __name__ == "__main__":
    main()
