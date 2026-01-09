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
--log-level <log-level> \
--save-scheduler-logs <True|False>
"""
#######################################################################

import argparse, csv, json, logging, threading, time, yaml, subprocess, re
from argparse import BooleanOptionalAction
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Callable

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
    SystemClock, Runner, Clock,
    log_args_block
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
from scripts.kwok_trace_replayer.trace_helpers import (
    TraceRecord,
)

#######################################################################
# Constants
#######################################################################
MAX_REPLAY_WORKERS = 5  # number of threads for replaying events. If more than 1, tasks run "async"

# Base ReplicaSet name produced by _rs_name_for_record (e.g. rs-000001)
_RS_PREFIX_RE = re.compile(r"^(rs-\d{6})(?:-.*)?$")

#######################################################################
# Logging setup
#######################################################################
LOGGER_NAME = "trace-replayer"
LOG = logging.getLogger(LOGGER_NAME)

#######################################################################
# Small helpers (pure)
#######################################################################
def _parse_optional_bool_strict(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _identity_from_pod_name(pod_name: str) -> str:
    """
    Map a pod name to the "identity" used by prio_by_identity.

    - For RS pods, we expect names like: rs-000001-<hash>-<index> or rs-000001-...
      and we map that to identity "rs-000001".
    - Otherwise, we try stripping a last "-suffix".
    """
    m = _RS_PREFIX_RE.match(pod_name)
    if m:
        return m.group(1)

    # Generic fallback: strip the last "-something"
    if "-" in pod_name:
        identity, _suffix = pod_name.rsplit("-", 1)
        return identity
    return pod_name


def _prio_for_pod_name(
    pod_name: str,
    prio_by_identity: Dict[str, int],
    max_prio: int,
) -> int | None:
    identity = _identity_from_pod_name(pod_name)
    prio = prio_by_identity.get(identity)
    if prio is None:
        return None
    if prio < 1 or prio > max_prio:
        return None
    return prio


class _TimeClock:
    """Fallback clock if an injected Clock doesn't provide time/sleep."""
    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


#######################################################################
# Event model
#######################################################################
@dataclass
class Event:
    sim_time_s: float   # seconds in trace's time
    kind: str           # "create" or "delete"
    record_id: int
    cpu_str: str | None = None
    mem_str: str | None = None
    pc_name: str | None = None
    replicas: int = 1


#####################################################################
# Argument parser
#####################################################################
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(
        "Replay a JSON pod trace on a KWOK cluster and monitor utilization. "
        "Expects <trace-dir>/trace.json as produced by trace_generator.py."
    ))

    # Result directory
    p.add_argument(
        "--result-dir", dest="result_dir", required=False, default=None,
        help="Directory where replay results (e.g., trace-monitor.csv) are written.",
    )

    # Job file (optional)
    p.add_argument(
        "--job-file", dest="job_file", default=None,
        help="Path to a YAML job file describing the trace replay job (trace-dir, kwokctl-config-file, overrides, ...).",
    )

    # Trace directory (can come from CLI or job-file)
    p.add_argument(
        "--trace-dir", dest="trace_dir", required=False, default=None,
        help="Directory containing trace.json from trace_generator.py",
    )

    # Cluster / KWOK options
    p.add_argument(
        "--cluster-name", dest="cluster_name", default=None,
        help="KWOK cluster name (kwokctl --name) (default: kwok1).",
    )
    p.add_argument(
        "--kwok-runtime", dest="kwok_runtime", choices=["binary", "docker"], default=None,
        help="KWOK runtime (default: binary).",
    )
    p.add_argument(
        "--kwokctl-config-file", dest="kwokctl_config_file", required=False, default=None,
        help="KwokctlConfiguration YAML used to create the KWOK cluster.",
    )
    p.add_argument(
        "--namespace", dest="namespace", default=None,
        help="Kubernetes namespace in which to create pods (default: trace).",
    )
    p.add_argument(
        "--node-cpu", dest="node_cpu", default=None,
        help=(
            "Per-node CPU capacity as a Kubernetes quantity. "
            "The trace stores CPU as a fraction of one node; this flag defines what "
            "'1.0' means when converting to pod requests "
            "(e.g., 0.25 → 250m if --node-cpu=1000m). "
            "Default: 1000m (≈1 core)."
        ),
    )
    p.add_argument(
        "--node-mem", dest="node_mem", default=None,
        help=(
            "Per-node memory capacity as a Kubernetes quantity. "
            "The trace stores memory as a fraction of one node; this flag defines what "
            "'1.0' means when converting to pod requests "
            "(e.g., 0.5 → 512Mi if --node-mem=1Gi). "
            "Default: 1Gi."
        ),
    )

    # Monitoring
    p.add_argument(
        "--monitor-interval", dest="monitor_interval", type=float, default=None,
        help="Monitor sampling interval in seconds (default: 1.0).",
    )

    # Logging
    p.add_argument(
        "--log-level", dest="log_level", default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO).",
    )

    # Scheduler logs
    p.add_argument(
        "--save-scheduler-logs",
        dest="save_scheduler_logs",
        action=BooleanOptionalAction,
        default=None,
        help="Save 'kwokctl logs kube-scheduler' under <result-dir>/scheduler-logs before exiting.",
    )

    return p


def merge_job_fields_into_args(
    args: argparse.Namespace,
    job: Dict[str, Any],
) -> tuple[argparse.Namespace, List[Dict[str, Any]]]:
    """
    Merge job-file fields into args. CLI has priority.
    Returns (args, override_kwokctl_envs).
    """
    fields = [
        JobField("trace-dir", "trace_dir", parse=parse_optional_str),
        JobField("cluster-name", "cluster_name", parse=parse_optional_str),
        JobField("kwok-runtime", "kwok_runtime", parse=parse_optional_str),
        JobField("kwokctl-config-file", "kwokctl_config_file", parse=parse_optional_str),
        JobField("namespace", "namespace", parse=parse_optional_str),
        JobField("node-cpu", "node_cpu", parse=parse_optional_str),
        JobField("node-mem", "node_mem", parse=parse_optional_str),
        JobField("monitor-interval", "monitor_interval", parse=parse_optional_float),
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
    """
    Final defaults + sanity checks after we've merged job-file and CLI.
    """
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
    if getattr(args, "log_level", None) is None:
        args.log_level = "INFO"
    if getattr(args, "job_file", None) is None:
        args.job_file = None
    if getattr(args, "save_scheduler_logs", None) is None:
        args.save_scheduler_logs = False

    # Required: trace_dir + kwokctl_config_file + result_dir
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


#######################################################################
# TraceReplayer class
#######################################################################
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

        # Base directory containing trace.json
        self.base_dir: Path = Path(args.trace_dir).resolve()
        self.trace_path: Path = self.base_dir / "trace.json"

        # Result directory (required; normalized in ensure_default_args)
        self.results_dir: Path = Path(self.args.result_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_path = self.results_dir / "results.csv"

        # Filled by load_trace
        self.pods: List[TraceRecord] = []
        self.max_prio: int = 0
        self.t_min: float = 0.0
        self.trace_time: float = 0.0
        self.meta: dict = {}

        # Derived from meta / args
        self.num_nodes: int = 0
        self.node_cpu_m: int = 0
        self.node_mem_b: int = 0

        # Monitoring fields
        self.ctx: str = f"kwok-{args.cluster_name}"
        self.events: List[Event] = []
        self.prio_by_identity: Dict[str, int] = {}

        # Runner + clock + executor factory
        self.runner = runner
        self.clock = clock or SystemClock()
        if not (hasattr(self.clock, "time") and hasattr(self.clock, "sleep")):
            self.clock = _TimeClock()
        self.executor_factory = executor_factory

        # Write metadata bundle (non-fatal on error)
        LOG.info("logging arguments and git info to trace_dir...")
        self._write_info_file()

        # Log args
        self.log_args()

    ##############################################
    # ------------ Info/logging helpers ----------
    ##############################################
    def log_args(self) -> None:
        include = [
            "trace_dir", "cluster_name", "kwok_runtime", "kwokctl_config_file",
            "namespace", "node_cpu", "node_mem", "monitor_interval",
            "result_dir", "log_level", "save_scheduler_logs", "job_file",
        ]
        log_args_block(LOG, self.args, title="ARGS", include=include)

    def _write_info_file(self) -> None:
        """
        Write info_replayer.yaml in results_dir with git + CLI + args.
        """
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
            write_info_file(
                out_path,
                meta_extra=meta_extra,
                inputs=inputs,
                logger=LOG,
            )
        except Exception as e:
            LOG.warning("failed to write info_replayer.yaml: %s", e)

    def _save_scheduler_logs(self) -> None:
        sched_dir = self.results_dir / "scheduler-logs"
        out_path = sched_dir / "sched_logs.log"
        save_kwok_scheduler_logs(
            self.args.cluster_name,
            out_path,
            runner=self.runner,
            logger=LOG,
        )

    ##############################################
    # ------------ Replay helpers ----------------
    ##############################################
    @staticmethod
    def _rs_name_for_record(record_id: int) -> str:
        """
        Stable ReplicaSet name derived from the trace record id.
        Example: record_id=1 -> "rs-000001"
        """
        return f"rs-{record_id:06d}"

    def load_trace(self) -> None:
        """Load trace JSON and populate pods, max_prio, t_min, trace_time, meta."""
        if not self.trace_path.exists():
            raise FileNotFoundError(
                f"Trace file not found: {self.trace_path} "
                f"(expected trace.json inside --trace-dir={self.base_dir})"
            )

        with open(self.trace_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        meta = raw.get("meta", {}) or {}
        records = raw.get("pods", []) or []

        def req(rec: dict, key: str) -> Any:
            if key not in rec:
                raise ValueError(f"trace record missing required key '{key}': {rec}")
            return rec[key]

        pods: List[TraceRecord] = []
        max_prio = 0
        t_min = float("inf")
        trace_time = 0.0

        for rec in records:
            id_val   = int(req(rec, "id"))
            start    = float(req(rec, "start_time"))
            end      = float(req(rec, "end_time"))
            cpu      = float(req(rec, "cpu"))
            mem      = float(req(rec, "mem"))
            prio     = int(req(rec, "priority"))
            replicas = int(req(rec, "replicas"))

            max_prio   = max(max_prio, prio)
            t_min      = min(t_min, start)
            trace_time = max(trace_time, end)

            pods.append(
                TraceRecord(
                    id=id_val,
                    start_time=start,
                    end_time=end,
                    cpu=cpu,
                    mem=mem,
                    priority=prio,
                    replicas=replicas,
                )
            )

        # Prefer the trace horizon from meta if provided: "trace_time_s"
        meta_trace_time = meta.get("trace_time_s")
        if meta_trace_time is not None:
            try:
                trace_time = float(meta_trace_time)
            except (TypeError, ValueError):
                pass

        LOG.info(
            "loaded %d pods from %s (t_min=%.3f, trace_time=%.3f, max_priority=%d)",
            len(pods),
            self.trace_path,
            t_min if t_min != float("inf") else 0.0,
            trace_time,
            max_prio,
        )

        pods.sort(key=lambda p: p.start_time)

        self.pods = pods
        self.max_prio = max_prio
        self.t_min = 0.0 if t_min == float("inf") else t_min
        self.trace_time = trace_time
        self.meta = meta

        if "num_nodes" not in meta:
            raise ValueError("trace meta missing required key 'num_nodes'")
        self.num_nodes = int(meta["num_nodes"])

    def _build_events(self) -> None:
        """
        Turn trace pods into a sorted list of events with concrete K8s quantities.
        """
        events: List[Event] = []
        for p in self.pods:
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
            events.append(
                Event(
                    sim_time_s=float(p.end_time),
                    kind="delete",
                    record_id=p.id,
                )
            )

        # sort by sim_time, then create before delete
        events.sort(key=lambda e: (e.sim_time_s, 0 if e.kind == "create" else 1))
        LOG.info("built %d events from %d pods", len(events), len(self.pods))
        self.events = events

    ##############################################
    # ------------ Replayer ----------------------
    ##############################################
    def _replay_events(self, namespace: str, start_wall_time: float, sim_t0: float) -> None:
        """
        Replay events against the cluster.

        We respect a global trace horizon self.trace_time (usually meta['trace_time_s']):
        - All events with sim_time_s <= trace_time are executed.
        - If the next batch would be after trace_time, we instead sleep until the
          corresponding wall time for trace_time and then exit.
        - If there are no more events but trace_time is still in the future,
          we also sleep until trace_time before finishing.
        """
        header, footer = make_header_footer("TRACE REPLAY")
        LOG.info(
            "\n%s\nstart_wall=%s sim_t0=%.3f trace_end_s=%.3f\n%s",
            header,
            get_timestamp(),
            sim_t0,
            self.trace_time,
            footer,
        )

        events = self.events
        num_events = len(events)
        trace_end_s = float(self.trace_time)
        trace_total_wall = max(0.0, trace_end_s - sim_t0)

        def sleep_until(reason: str) -> None:
            if trace_end_s <= sim_t0:
                LOG.info(
                    "trace_end_s=%.3f <= sim_t0=%.3f; no extra sleep (%s)",
                    trace_end_s,
                    sim_t0,
                    reason,
                )
                return

            target_wall_end = start_wall_time + max(0.0, trace_end_s - sim_t0)
            now = float(self.clock.time())
            sleep_s = max(0.0, target_wall_end - now)
            if sleep_s > 0:
                LOG.info(
                    "sleeping %.3fs to reach trace_end_s=%.3f (reason=%s)",
                    sleep_s,
                    trace_end_s,
                    reason,
                )
                self.clock.sleep(sleep_s)
            else:
                LOG.info(
                    "trace_end_s=%.3f already reached in wall time (reason=%s); no sleep",
                    trace_end_s,
                    reason,
                )

        if num_events == 0:
            LOG.info("no events in trace; only aligning to trace_end_s=%.3f", trace_end_s)
            sleep_until("no-events")
            return

        idx = 0
        executor = self.executor_factory(max_workers=MAX_REPLAY_WORKERS)
        futures: List[Future] = []
        reached_end_sleep = False

        try:
            while idx < num_events:
                current_t = events[idx].sim_time_s

                if current_t > trace_end_s:
                    LOG.info(
                        "next batch sim_t=%.3f is beyond trace_end_s=%.3f; no more events will be replayed",
                        current_t,
                        trace_end_s,
                    )
                    sleep_until("next-batch-after-end")
                    reached_end_sleep = True
                    break

                batch_events: List[Event] = []
                while idx < num_events and events[idx].sim_time_s == current_t:
                    batch_events.append(events[idx])
                    idx += 1

                target_wall = start_wall_time + max(0.0, current_t - sim_t0)

                now_before = float(self.clock.time())
                sleep_s = max(0.0, target_wall - now_before)
                if sleep_s > 0:
                    self.clock.sleep(sleep_s)

                creates = [ev for ev in batch_events if ev.kind == "create"]
                deletes = [ev for ev in batch_events if ev.kind == "delete"]

                now_after = float(self.clock.time())
                batch_drift = now_after - target_wall

                sim_remaining_s = max(0.0, trace_end_s - current_t)
                wall_elapsed_s = now_after - start_wall_time
                wall_remaining_s = max(0.0, trace_total_wall - wall_elapsed_s)

                LOG.info(
                    "TIME DRIFT batch @ sim_t=%.3f: target_wall=%.3f actual_wall=%.3f "
                    "drift=%.6fs (creates=%d deletes=%d)",
                    current_t,
                    target_wall - start_wall_time,
                    now_after - start_wall_time,
                    batch_drift,
                    len(creates),
                    len(deletes),
                )

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
                    fut = executor.submit(kubectl_apply_yaml, LOG, self.ctx, yaml_text)
                    futures.append(fut)

                for ev in deletes:
                    rs_name = self._rs_name_for_record(ev.record_id)
                    LOG.info(
                        "DELETE @ sim_t=%.3f: rs=%s (id=%d)",
                        ev.sim_time_s,
                        rs_name,
                        ev.record_id,
                    )
                    fut = executor.submit(delete_rs, LOG, self.ctx, namespace, rs_name)
                    futures.append(fut)

                LOG.info(
                    "batch done @ sim_t=%.3f submitted: creates=%d deletes=%d "
                    "(sim_remaining=%.3fs wall_remaining=%.3fs)",
                    current_t,
                    len(creates),
                    len(deletes),
                    sim_remaining_s,
                    wall_remaining_s,
                )

            if not reached_end_sleep:
                LOG.info(
                    "all events <= trace_end_s=%.3f processed; aligning to trace end if needed",
                    trace_end_s,
                )
                sleep_until("after-last-batch")

            LOG.info("all events processed; waiting for kubectl tasks to finish...")

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

    ##############################################
    # ------------ Monitor helpers ---------------
    ##############################################
    def _snapshot_from_pods(self, ns: str) -> tuple[float, float, List[tuple[str, str]], List[str]]:
        """
        Build a snapshot directly from pods.

        Returns:
            cpu_run_util:  fraction of total cluster CPU capacity requested by running pods
            mem_run_util:  fraction of total cluster memory capacity requested by running pods
            pods_running:  list of (pod_name, node_name) for running pods
            pending_pods:  list of pod names that are Pending
        """
        pods_json = get_json_ctx(self.ctx, ["-n", ns, "get", "pods", "-o", "json"])
        items = pods_json.get("items", []) or []
        total_cpu_m = 0
        total_mem_b = 0
        pods_running: List[tuple[str, str]] = []
        pending_pods: List[str] = []

        for pod in items:
            meta = pod.get("metadata", {}) or {}
            spec = pod.get("spec", {}) or {}
            status = pod.get("status", {}) or {}
            pod_name = meta.get("name", "")
            node_name = spec.get("nodeName", "") or ""
            phase = status.get("phase", "")

            if phase == "Running":
                pods_running.append((pod_name, node_name))
                containers = spec.get("containers", []) or []
                for c in containers:
                    res = (c.get("resources") or {}).get("requests", {}) or {}
                    cpu_q = res.get("cpu")
                    mem_q = res.get("memory")
                    total_cpu_m += qty_to_mcpu_int(cpu_q)
                    total_mem_b += qty_to_bytes_int(mem_q)
            elif phase == "Pending":
                pending_pods.append(pod_name)

        cpu_capacity_m = self.num_nodes * self.node_cpu_m
        mem_capacity_b = self.num_nodes * self.node_mem_b
        cpu_run_util = (total_cpu_m / cpu_capacity_m) if cpu_capacity_m > 0 else 0.0
        mem_run_util = (total_mem_b / mem_capacity_b) if mem_capacity_b > 0 else 0.0

        return cpu_run_util, mem_run_util, pods_running, pending_pods

    def _monitor_loop(
        self,
        namespace: str,
        interval_s: float,
        max_prio: int,
        prio_by_identity: Dict[str, int],
        out_csv: Path,
        stop_event: threading.Event,
        start_wall_time: float,
        sim_t0: float,
    ) -> None:
        """
        Periodically sample cluster state and write CSV.

        We record:
          - total CPU/mem utilization
          - total running / unsched counts
          - running_by_prio:  { "p1": <count>, ... }
          - unsched_by_prio:  { "p1": <count>, ... }
        """
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        LOG.info("monitor: writing time series to %s (interval=%.3fs)", out_csv, interval_s)

        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = [
                "timestamp",
                "wall_time_s",
                "sim_time_s",
                "cpu_run_util",
                "mem_run_util",
                "running_count",
                "unsched_count",
                "running_by_prio",
                "unsched_by_prio",
            ]
            writer.writerow(header)

            while not stop_event.is_set():
                loop_start = float(self.clock.time())
                try:
                    cpu_run_util, mem_run_util, pods_running, pending_pods = self._snapshot_from_pods(namespace)
                except Exception as e:
                    LOG.warning("monitor: snapshot_from_pods failed: %s", e)
                    self.clock.sleep(interval_s)
                    continue

                now_abs = float(self.clock.time())
                real_ts = get_timestamp()
                wall_time_s = now_abs - start_wall_time
                sim_time_s = sim_t0 + wall_time_s
                running_cnt = len(pods_running)
                unsched_cnt = len(pending_pods)

                running_by_prio: Dict[str, int] = {f"p{p}": 0 for p in range(1, max_prio + 1)}
                unsched_by_prio: Dict[str, int] = {f"p{p}": 0 for p in range(1, max_prio + 1)}

                for pod_name, _node_name in pods_running:
                    prio = _prio_for_pod_name(pod_name, prio_by_identity, max_prio)
                    if prio is not None:
                        running_by_prio[f"p{prio}"] += 1

                for pod_name in pending_pods:
                    prio = _prio_for_pod_name(pod_name, prio_by_identity, max_prio)
                    if prio is not None:
                        unsched_by_prio[f"p{prio}"] += 1

                running_dict_str = json.dumps(running_by_prio, separators=(",", ":"), sort_keys=True)
                unsched_dict_str = json.dumps(unsched_by_prio, separators=(",", ":"), sort_keys=True)

                row: list[Any] = [
                    real_ts,
                    f"{wall_time_s:.3f}",
                    f"{sim_time_s:.3f}",
                    f"{cpu_run_util:.6f}",
                    f"{mem_run_util:.6f}",
                    running_cnt,
                    unsched_cnt,
                    running_dict_str,
                    unsched_dict_str,
                ]
                writer.writerow(row)
                f.flush()

                elapsed = float(self.clock.time()) - loop_start
                sleep_s = max(0.0, interval_s - elapsed)
                if sleep_s > 0:
                    self.clock.sleep(sleep_s)

        LOG.info("monitor: stop signal received; exiting")

    ##############################################
    # ------------ Runner ------------------------
    ##############################################
    def run(self) -> None:
        # Load trace
        self.load_trace()

        # self.prio_by_identity holds all pods from the trace and their priorities
        self.prio_by_identity = {self._rs_name_for_record(p.id): p.priority for p in self.pods}

        # Convert node capacities to ints (mCPU / bytes)
        self.node_cpu_m = qty_to_mcpu_int(self.args.node_cpu)
        self.node_mem_b = qty_to_bytes_int(self.args.node_mem)
        LOG.info(
            "per-node capacity: cpu_m=%d mem_bytes=%d (num_nodes=%d from trace meta)",
            self.node_cpu_m,
            self.node_mem_b,
            self.num_nodes,
        )

        # Build events
        self._build_events()

        # Create KWOK cluster
        kwok_cfg_path = Path(self.args.kwokctl_config_file).resolve()
        with open(kwok_cfg_path, "r", encoding="utf-8") as f:
            config_doc = yaml.safe_load(f) or {}

        # Apply override-kwokctl-envs from job-file, if any
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

        # Start monitor thread
        start_wall_time = float(self.clock.time())
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(
                self.args.namespace,
                float(self.args.monitor_interval),
                self.max_prio,
                self.prio_by_identity,
                self.monitor_path,
                stop_event,
                start_wall_time,
                self.t_min,
            ),
            daemon=True,
        )
        monitor_thread.start()

        try:
            self._replay_events(
                namespace=self.args.namespace,
                start_wall_time=start_wall_time,
                sim_t0=self.t_min,
            )
        finally:
            stop_event.set()
            monitor_thread.join(timeout=10.0)
            LOG.info("monitor thread joined; done.")

            if self.args.save_scheduler_logs:
                LOG.info("saving scheduler logs via kwokctl...")
                self._save_scheduler_logs()

        LOG.info("Done.")


###############################################
# ------------ Main entry point ---------------
###############################################
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
