#!/usr/bin/env python3
# trace_replayer.py
"""
python -m scripts.kwok_trace_replayer.trace_replayer --job-file <job-file.yaml>
"""

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
    parse_optional_str, parse_optional_float,
)
from scripts.helpers.general_helpers import (
    setup_logging, make_header_footer, get_timestamp,
    qty_to_mcpu_int, qty_to_mcpu_str, qty_to_bytes_int, qty_to_bytes_str,
    build_cli_cmd, write_info_file, SystemClock, Runner, Clock, log_args_block,
)
from scripts.helpers.kubectl_helpers import (
    kubectl_apply_yaml, ensure_namespace, ensure_priority_classes, delete_rs, get_json_ctx,
)
from scripts.helpers.kwokctl_helpers import (
    yaml_kwok_rs, create_kwok_nodes, ensure_kwok_cluster, kwok_pods_cap, merge_kwokctl_envs, save_kwok_scheduler_logs,
)
from scripts.kwok_trace_replayer.trace_helpers import TraceRecord

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MAX_REPLAY_WORKERS = 5 # max concurrent kubectl apply/delete calls during replay
LOGGER_NAME = "trace-replayer" # logger name
LOG = logging.getLogger(LOGGER_NAME) # module logger
from scripts.kwok_trace_replayer.trace_helpers import TraceRecord, rs_prefix_from_pod_name

# Plugin-exported optimization stats
OPT_STATS_NS = "kube-system"
OPT_STATS_CM = "optimization-stats"
OPT_STATS_KEY = "optimization-stats.json"
OPT_STATS_DUMP_INTERVAL_S = 30.0

# Seeds to skip during replay for selective runs
SKIP_SEEDS = {1420052706459400740, 2219457405427907235, 3848061858430934892}

# ---------------------------------------------------------------------
# CLI + Job File
# ---------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.
    """
    p = argparse.ArgumentParser(description=(
            "Replay a JSON pod trace on a KWOK cluster and monitor utilization. "
            "Expects <trace-dir>/trace.json and <trace-dir>/initial.json as produced by trace_generator.py."
        )
    )

    # General config
    p.add_argument("--job-file", dest="job_file", default=None)
    p.add_argument("--result-dir", dest="result_dir", required=False, default=None)
    p.add_argument("--trace-dir", dest="trace_dir", required=False, default=None)

    # KWOK cluster config
    p.add_argument("--cluster-name", dest="cluster_name", default=None)
    p.add_argument("--kwok-runtime", dest="kwok_runtime", choices=["binary", "docker"], default=None)
    p.add_argument("--kwokctl-config-file", dest="kwokctl_config_file", required=False, default=None)
    p.add_argument("--namespace", dest="namespace", default=None)
    p.add_argument("--node-cpu", dest="node_cpu", default=None)
    p.add_argument("--node-mem", dest="node_mem", default=None)
    p.add_argument("--save-scheduler-logs", dest="save_scheduler_logs", action=BooleanOptionalAction, default=None)
    
    # Monitoring config
    p.add_argument("--monitor-interval", dest="monitor_interval", type=float, default=None)
    p.add_argument("--start-delay", dest="start_delay", type=float, default=None)

    # Logging config
    p.add_argument("--log-level", dest="log_level", default=None)

    return p

def merge_job_fields_into_args(
    args: argparse.Namespace,
    job: Dict[str, Any],
) -> tuple[argparse.Namespace, List[Dict[str, Any]]]:
    """
    Merge supported job-file fields into args and return override kwokctl envs.
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
        JobField("start-delay", "start_delay", parse=parse_optional_float),
        JobField("log-level", "log_level", parse=parse_optional_str),
        JobField("result-dir", "result_dir", parse=parse_optional_str),
        JobField("save-scheduler-logs", "save_scheduler_logs", parse=parse_optional_bool_strict,
            accept=lambda v: isinstance(v, bool),
        ),
    ]
    args = _merge_job_fields_into_args(args, job or {}, fields)
    override_envs = (job or {}).get("override-kwokctl-envs") or []
    return args, override_envs

def ensure_default_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply defaults and validate required args for the trace replayer.
    """
    # Set defaults
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

    # Validate required args
    if not getattr(args, "trace_dir", None):
        raise SystemExit("--trace-dir (or trace-dir in job-file) is required")
    if not getattr(args, "kwokctl_config_file", None):
        raise SystemExit("--kwokctl-config-file (or kwokctl-config-file in job-file) is required")
    if not getattr(args, "result_dir", None):
        raise SystemExit("--result-dir (or result-dir in job-file) is required")

    # Resolve paths
    trace_dir = Path(args.trace_dir).resolve()
    if not trace_dir.exists():
        raise SystemExit(f"--trace-dir not found: {trace_dir}")
    
    kwok_cfg = Path(args.kwokctl_config_file).resolve()
    if not kwok_cfg.exists():
        raise SystemExit(f"--kwokctl-config-file not found: {kwok_cfg}")
    args.result_dir = str(Path(args.result_dir).resolve())
    
    return args

# ---------------------------------------------------------------------
# Trace Discovery Helpers
# ---------------------------------------------------------------------

def discover_trace_run_dirs(trace_dir: Path) -> List[Path]:
    """
    Discover run directories containing a trace.json under the given trace-dir.
    """
    trace_dir = trace_dir.resolve()
    if (trace_dir / "trace.json").exists():
        return [trace_dir]
    if not trace_dir.is_dir():
        raise SystemExit(f"--trace-dir must be a directory: {trace_dir}")
    run_dirs: List[Path] = []
    try:
        for child in sorted(trace_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            if (child / "trace.json").exists():
                run_dirs.append(child)
    except Exception as e:
        raise SystemExit(f"failed to scan --trace-dir for seeds: {trace_dir} ({e})")
    if run_dirs:
        return run_dirs
    return [trace_dir]

# ---------------------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------------------

def parse_optional_bool_strict(v: Any) -> bool | None:
    """
    Parse a job-file value that must be a real boolean (or None).
    """
    return v if isinstance(v, bool) else None

def parse_rfc3339_to_epoch(ts: str) -> Optional[float]:
    """
    Parse a Kubernetes RFC3339 timestamp string into epoch seconds.
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

# ---------------------------------------------------------------------
# Small Models
# ---------------------------------------------------------------------

class TimeClock:
    def time(self) -> float:
        """
        Return wall-clock time in seconds (time.time()).
        """
        return time.time()

    def sleep(self, seconds: float) -> None:
        """
        Sleep for the given number of seconds.
        """
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

# ---------------------------------------------------------------------
# Trace Replayer
# ---------------------------------------------------------------------

class TraceReplayer:
    """
    Replay a trace on a KWOK cluster and monitor state.
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
        """
        Initialize replayer state, resolve paths, and record metadata.
        """
        self.init_runtime(runner=runner, clock=clock, executor_factory=executor_factory)
        self.init_context(args=args, job_doc=job_doc, override_kwokctl_envs=override_kwokctl_envs)
        self.initialize()
        self.configure_from_args()
        self.write_info_file()

    def init_runtime(self, *, runner: Runner, clock: Clock | None, executor_factory: Callable[..., Any]) -> None:
        """
        Initialize runtime components like runner, clock, and executor factory.
        """
        self.runner = runner
        self.clock = self.resolve_clock(clock)
        self.executor_factory = executor_factory

    def init_context(
        self,
        *,
        args: argparse.Namespace,
        job_doc: Dict[str, Any] | None,
        override_kwokctl_envs: List[Dict[str, Any]] | None,
    ) -> None:
        """
        Initialize context from args, job doc, and override envs.
        """
        self.args = args
        self.job_doc = job_doc or {}
        self.override_kwokctl_envs = list(override_kwokctl_envs or [])

    @staticmethod
    def resolve_clock(clock: Clock | None) -> Clock:
        """
        Choose a clock implementation.
        """
        c = clock or SystemClock()
        return c if (hasattr(c, "time") and hasattr(c, "sleep")) else TimeClock()

    def initialize(self) -> None:
        """
        Resolve args (job-file + defaults), setup logging, and log args once.
        """
        args = self.args

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
            self.job_doc = job_doc
            self.override_kwokctl_envs = override_kwokctl_envs

        args = ensure_default_args(args)
        setup_logging(name=LOGGER_NAME, prefix=f"[{LOGGER_NAME}] ", level=args.log_level)

        self.args = args
        self.log_args()

    def configure_from_args(self) -> None:
        """
        Configure derived paths/state from (resolved) args.
        """
        self.base_dir: Path = Path(self.args.trace_dir).resolve()
        self.trace_path: Path = self.base_dir / "trace.json"
        self.initial_path: Path = self.base_dir / "initial.json"
        self.info_generate_path: Path = self.base_dir / "info_generate.yaml"

        self.results_dir: Path = Path(self.args.result_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.general_stats_path = self.results_dir / "general_stats.csv"
        self.pod_stats_path = self.results_dir / "pod_stats.csv"
        self.optimization_stats_path = self.results_dir / "optimization_stats.json"

        self.trace_pods: List[TraceRecord] = []
        self.initial_pods: List[TraceRecord] = []

        self.num_nodes: int = 0
        self.max_prio: int = 0
        self.trace_time_s: float = 0.0
        self.replay_end_s: float = 0.0

        self.node_cpu_m: int = 0
        self.node_mem_b: int = 0

        self.events: List[Event] = [] # sorted create/delete events
        self.prio_by_rs: Dict[str, int] = {} # rs -> priority (include both initial + trace pods)
        self.initial_rs_names: Set[str] = set() # rs names that belong to initial pods (skip these in pod_stats.csv)

        self.ctx: str = f"kwok-{self.args.cluster_name}"

        # Run start epoch/time base
        self.run_start_wall: float = 0.0      # epoch seconds
        self.run_start_monotonic: float = 0.0 # monotonic seconds

    # ------------------------------
    # Logging / Metadata
    # ------------------------------
    
    def log_args(self) -> None:
        """
        Log key arguments in a stable order.
        """
        include = [
            "job_file",
            "trace_dir",
            "result_dir",
            "cluster_name",
            "kwok_runtime",
            "kwokctl_config_file",
            "namespace",
            "node_cpu",
            "node_mem",
            "save_scheduler_logs",
            "monitor_interval",
            "start_delay",
            "log_level",
        ]
        log_args_block(LOG, self.args, title="ARGS", include=include)

    def write_info_file(self) -> None:
        """
        Write info_replayer.yaml metadata into the results directory.
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
            write_info_file(out_path, meta_extra=meta_extra, inputs=inputs, logger=LOG)
        except Exception as e:
            LOG.warning("failed to write info_replayer.yaml: %s", e)

    def save_scheduler_logs(self) -> None:
        """
        Save scheduler logs into results_dir/scheduler-logs/.
        """
        sched_dir = self.results_dir / "scheduler-logs"
        out_path = sched_dir / "sched_logs.log"
        save_kwok_scheduler_logs(self.args.cluster_name, out_path, runner=self.runner, logger=LOG)

    # ------------------------------
    # Trace Input Loading
    # ------------------------------

    @staticmethod
    def rs_name_for_record(record_id: int) -> str:
        """
        Format a record id into an rs name like rs-000001.
        """
        return f"rs-{record_id:06d}"

    def load_generate_info(self) -> Dict[str, Any]:
        """
        Load info_generate.yaml from the trace directory if present.
        """
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

    def load_json_pods(self, path: Path) -> List[TraceRecord]:
        """
        Load pods from a trace JSON file into sorted TraceRecord objects.
        """
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw.get("pods", []) or []

        def require_key(rec: dict, key: str) -> Any:
            """
            Require a key in a trace record dict and return its value.
            """
            if key not in rec:
                raise ValueError(f"trace record missing required key '{key}': {rec}")
            return rec[key]

        pods: List[TraceRecord] = []
        for rec in records:
            pods.append(
                TraceRecord(
                    id=int(require_key(rec, "id")),
                    start_time=float(require_key(rec, "start_time")),
                    end_time=float(require_key(rec, "end_time")),
                    cpu=float(require_key(rec, "cpu")),
                    mem=float(require_key(rec, "mem")),
                    priority=int(require_key(rec, "priority")),
                    replicas=int(require_key(rec, "replicas")),
                )
            )
        pods.sort(key=lambda p: p.start_time)
        return pods

    def load_initial_and_trace(self) -> None:
        """
        Load initial/trace pods and derive num_nodes, trace_time_s, and max_prio.
        """
        gen_info = self.load_generate_info()

        self.trace_pods = self.load_json_pods(self.trace_path)
        self.initial_pods = self.load_json_pods(self.initial_path) if self.initial_path.exists() else []

        # Prefer info_generate.yaml for these if present; otherwise infer
        num_nodes = None
        trace_time_s = None
        max_prio = None

        def deep_get(d: Dict[str, Any], path: List[str]) -> Any:
            """
            Get a nested value from a dict, returning None if missing.
            """
            cur: Any = d
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
            return cur

        # Try multiple possible paths for each value
        for candidate in [
            deep_get(gen_info, ["inputs", "generated", "num_nodes"]),
            deep_get(gen_info, ["generated", "num_nodes"]),
        ]:
            if isinstance(candidate, int):
                num_nodes = candidate

        for candidate in [
            deep_get(gen_info, ["inputs", "generated", "trace_time_s"]),
            deep_get(gen_info, ["generated", "trace_time_s"]),
        ]:
            if isinstance(candidate, (int, float)):
                trace_time_s = float(candidate)

        for candidate in [
            deep_get(gen_info, ["inputs", "generated", "max_priority_seen"]),
            deep_get(gen_info, ["generated", "max_priority_seen"]),
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
            rs = self.rs_name_for_record(p.id)
            self.prio_by_rs[rs] = int(p.priority)
            self.initial_rs_names.add(rs)

        for p in self.trace_pods:
            rs = self.rs_name_for_record(p.id)
            self.prio_by_rs[rs] = int(p.priority)

    # ------------------------------
    # Event Construction
    # ------------------------------
    
    def build_trace_events(self) -> None:
        """
        Build sorted create/delete events from trace pod records.
        """
        events: List[Event] = []

        start_delay = float(getattr(self.args, "start_delay", 0.0) or 0.0)
        effective_trace_time_s = float(getattr(self, "trace_time_s", 0.0) or 0.0)
        # If no trace_time_s, derive from max end_time of trace pods.
        if effective_trace_time_s <= 0.0 and self.trace_pods:
            t_max = 0.0 # max end_time from trace pods
            for p in self.trace_pods:
                t_max = max(t_max, float(p.end_time))
            effective_trace_time_s = float(t_max)
        # Replay end time is start_delay + effective_trace_time_s
        replay_end_s = start_delay + effective_trace_time_s
        self.replay_end_s = float(replay_end_s)

        # Initial pods are created up-front by apply_initial_pods().
        # Schedule their deletions here so they don't persist forever.
        for p in self.initial_pods:
            end_t = float(p.end_time)
            # Only schedule deletes that happen within the replay horizon.
            if end_t <= replay_end_s:
                events.append(Event(sim_time_s=end_t, kind="delete", record_id=p.id))
        # Trace pods: schedule creates and deletes within horizon.
        for p in self.trace_pods:
            cpu_m = max(1, int(round(p.cpu * self.node_cpu_m)))
            mem_b = max(1, int(round(p.mem * self.node_mem_b)))
            cpu_str = qty_to_mcpu_str(cpu_m)
            mem_str = qty_to_bytes_str(mem_b)
            pc_name = f"p{int(p.priority)}"
            replicas = max(1, int(getattr(p, "replicas", 1)))

            # Only create if within horizon
            create_t = start_delay + float(p.start_time)
            if create_t <= replay_end_s:
                events.append(
                    Event(
                        sim_time_s=create_t,
                        kind="create",
                        record_id=p.id,
                        cpu_str=cpu_str,
                        mem_str=mem_str,
                        pc_name=pc_name,
                        replicas=replicas,
                    )
                )

            # Only delete if within horizon; if a pod would naturally end after
            # the trace horizon, we just stop the replay while it is still alive.
            delete_t = start_delay + float(p.end_time)
            if delete_t <= replay_end_s:
                events.append(Event(sim_time_s=delete_t, kind="delete", record_id=p.id))

        # At identical timestamps, process deletes first to free capacity and
        # for determinism.
        events.sort(key=lambda e: (e.sim_time_s, 0 if e.kind == "delete" else 1))
        self.events = events
        
        LOG.info(
            "built %d events (initial_deletes_in_horizon=%d, trace_pods=%d, replay_end_s=%.3f)",
            len(events),
            sum(1 for e in events if e.kind == "delete" and self.rs_name_for_record(e.record_id) in self.initial_rs_names),
            len(self.trace_pods),
            self.replay_end_s,
        )

    # ------------------------------
    # KWOK Apply / Replay
    # ------------------------------
    
    def apply_initial_pods(self, namespace: str) -> None:
        """
        Apply the initial pods as KWOK ReplicaSets in the given namespace.
        """
        header, footer = make_header_footer("APPLY INITIAL PODS")
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

                rs_name = self.rs_name_for_record(p.id)
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

    def replay_trace_events(self, namespace: str, trace_start_wall: float) -> None:
        """
        Replay trace events aligned to a wall-clock start time.
        """
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
        trace_end_s = float(getattr(self, "replay_end_s", 0.0) or float(self.trace_time_s))

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
            while idx < num_events: # while there are events left
                current_t = events[idx].sim_time_s
                if current_t > trace_end_s: # bail if past trace end
                    break

                # Gather all events at current_t
                batch_events: List[Event] = []
                while idx < num_events and events[idx].sim_time_s == current_t:
                    batch_events.append(events[idx])
                    idx += 1
                
                # Sleep until current_t
                target_wall = trace_start_wall + max(0.0, current_t)
                now_before = float(self.clock.time())
                sleep_s = max(0.0, target_wall - now_before)
                if sleep_s > 0:
                    self.clock.sleep(sleep_s)

                # Apply batch events (deletes first)
                deletes = [ev for ev in batch_events if ev.kind == "delete"]
                creates = [ev for ev in batch_events if ev.kind == "create"]
                for ev in deletes:
                    rs_name = self.rs_name_for_record(ev.record_id)
                    LOG.info("DELETE @ sim_t=%.3f: rs=%s (id=%d)", ev.sim_time_s, rs_name, ev.record_id)
                    futures.append(executor.submit(delete_rs, LOG, self.ctx, namespace, rs_name))
                for ev in creates:
                    assert ev.cpu_str is not None and ev.mem_str is not None and ev.pc_name is not None
                    rs_name = self.rs_name_for_record(ev.record_id)
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

            # Align replayer to trace end time even if no events at the end
            target_wall_end = trace_start_wall + trace_end_s
            now = float(self.clock.time())
            sleep_s = max(0.0, target_wall_end - now)
            if sleep_s > 0:
                self.clock.sleep(sleep_s)

        # Ensure all kubectl tasks complete
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
    
    def time_s(self) -> float:
        """
        Return elapsed time in seconds since run start (monotonic clock).
        """
        return float(self.clock.time()) - float(self.run_start_monotonic)

    def dump_optimization_stats(self) -> tuple[bool, str | None]:
        """
        Dump optimization stats ConfigMap to a local JSON file.
        """
        # IMPORTANT: Don't call get_json_ctx() here as kubectl can block
        # indefinitely. Use a small timeout for this best-effort dump.
        cmd = ["kubectl"]
        if self.ctx:
            cmd += ["--context", str(self.ctx)]
        cmd += ["-n", OPT_STATS_NS, "get", "configmap", OPT_STATS_CM, "-o", "json"]

        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=1.0)
            cm = json.loads(out)
        except subprocess.TimeoutExpired as e:
            LOG.debug("opt-stats: kubectl timed out: %s", e)
            return False, None
        except subprocess.CalledProcessError as e:
            raw = getattr(e, "output", None) or getattr(e, "stdout", None) or b""
            tail = raw.decode("utf-8", "replace")[-1200:]
            LOG.debug("opt-stats: kubectl failed: rc=%s output_tail=%r", getattr(e, "returncode", None), tail)
            return False, None
        except Exception as e:
            LOG.debug("opt-stats: CM not readable: %s", e)
            return False, None

        data = (cm.get("data") or {}) if isinstance(cm, dict) else {}
        raw = data.get(OPT_STATS_KEY)
        if not isinstance(raw, str) or not raw.strip():
            LOG.debug("opt-stats: missing key %s in CM %s/%s", OPT_STATS_KEY, OPT_STATS_NS, OPT_STATS_CM)
            return False, None

        updated_at = None
        try:
            doc = json.loads(raw)
            updated_at = doc.get("updated_at")
        except Exception:
            pass

        try:
            self.optimization_stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.optimization_stats_path.write_text(raw.strip() + "\n", encoding="utf-8")
            return True, updated_at
        except Exception as e:
            LOG.debug("opt-stats: write failed: %s", e)
            return False, None

    def snapshot_from_pods(self, ns: str) -> Tuple[float, float, float, float, Dict[int, int], Dict[int, int], List[Dict[str, Any]]]:
        """
        Snapshot req/run utilization, priority counts, and raw pod items from current pods.
        """
        pods_json = get_json_ctx(self.ctx, ["-n", ns, "get", "pods", "-o", "json"])
        pods = pods_json.get("items", []) or []

        total_cpu_run_m = 0
        total_mem_run_b = 0
        total_cpu_req_m = 0
        total_mem_req_b = 0

        running_by_prio: Dict[int, int] = {p: 0 for p in range(1, self.max_prio + 1)}
        pending_by_prio: Dict[int, int] = {p: 0 for p in range(1, self.max_prio + 1)}

        for pod in pods:
            meta = pod.get("metadata", {}) or {}
            spec = pod.get("spec", {}) or {}
            status = pod.get("status", {}) or {}
            pod_name = meta.get("name", "")
            phase = status.get("phase", "")
            rs_name = rs_prefix_from_pod_name(pod_name)
            prio = int(self.prio_by_rs.get(rs_name, 0))

            # Count running / pending by priority
            if phase == "Running":
                if prio in running_by_prio:
                    running_by_prio[prio] += 1
            elif phase == "Pending":
                if prio in pending_by_prio:
                    pending_by_prio[prio] += 1

            # Requested utilization is based on resource requests of pods that
            # are either Running or Pending
            if phase in ("Running", "Pending"):
                containers = spec.get("containers", []) or []
                for c in containers:
                    res = (c.get("resources") or {}).get("requests", {}) or {}
                    cpu_q = res.get("cpu")
                    mem_q = res.get("memory")
                    cpu_m = qty_to_mcpu_int(cpu_q)
                    mem_b = qty_to_bytes_int(mem_q)
                    total_cpu_req_m += cpu_m
                    total_mem_req_b += mem_b
                    if phase == "Running":
                        total_cpu_run_m += cpu_m
                        total_mem_run_b += mem_b

        cpu_capacity_m = self.num_nodes * self.node_cpu_m
        mem_capacity_b = self.num_nodes * self.node_mem_b
        cpu_run_util = (total_cpu_run_m / cpu_capacity_m) if cpu_capacity_m > 0 else 0.0
        mem_run_util = (total_mem_run_b / mem_capacity_b) if mem_capacity_b > 0 else 0.0
        cpu_req_util = (total_cpu_req_m / cpu_capacity_m) if cpu_capacity_m > 0 else 0.0
        mem_req_util = (total_mem_req_b / mem_capacity_b) if mem_capacity_b > 0 else 0.0
        return cpu_run_util, mem_run_util, cpu_req_util, mem_req_util, running_by_prio, pending_by_prio, pods

    def pod_start_time(self, pod: Dict[str, Any]) -> Optional[float]:
        """
        Return pod status.startTime as epoch seconds (or None).
        """
        status = pod.get("status", {}) or {}
        st = status.get("startTime")
        return parse_rfc3339_to_epoch(st) if isinstance(st, str) else None

    def pod_creation_time(self, pod: Dict[str, Any]) -> Optional[float]:
        """
        Return pod metadata.creationTimestamp as epoch seconds (or None).
        """
        meta = pod.get("metadata", {}) or {}
        ts = meta.get("creationTimestamp")
        return parse_rfc3339_to_epoch(ts) if isinstance(ts, str) else None

    def monitor_loop(
        self,
        namespace: str,
        interval_s: float,
        general_csv: Path,
        pod_stats_csv: Path,
        stop_event: threading.Event,
    ) -> None:
        """
        Continuously sample cluster state and write CSV stats until stopped.
        """
        general_csv.parent.mkdir(parents=True, exist_ok=True)
        pod_stats_csv.parent.mkdir(parents=True, exist_ok=True)

        LOG.info("monitor: writing %s (interval=%.3fs)", general_csv, interval_s)
        LOG.info("monitor: writing %s (apply-time + running-time; trace pods only)", pod_stats_csv)

        # pod_stats: only for trace pods (skip initial rs)
        seen_apply: set[str] = set()     # pod_name
        seen_running: set[str] = set()   # pod_name

        # deletion semantics (for deletions_cum_pK): Count a "deletion" when a
        # pod UID has been observed Running since the last count and then either
        # disappears from the live pod list, or transitions Running -> Pending.
        # The same UID can be counted multiple times if it becomes Running again
        # later.
        prev_live_uids: set[str] = set()
        prev_phase_by_uid: Dict[str, str] = {}
        eligible_uids: set[str] = set()  # UIDs eligible to be counted deleted (must have been Running)
        uid_to_prio: Dict[str, int] = {}  # last known prio for that uid (best-effort)
        deletions_cum_by_prio: Dict[int, int] = {p: 0 for p in range(1, self.max_prio + 1)}
        with open(general_csv, "w", encoding="utf-8", newline="") as f_ts, open(
            pod_stats_csv, "w", encoding="utf-8", newline=""
        ) as f_ps:
            timestamp_writer = csv.writer(f_ts)
            pod_stats_writer = csv.writer(f_ps)

            # general_stats.csv header
            header = ["timestamp", "time_s", "cpu_run_util", "mem_run_util", "cpu_req_util", "mem_req_util"]
            for p in range(1, self.max_prio + 1):
                header.append(f"running_p{p}")
            for p in range(1, self.max_prio + 1):
                header.append(f"unsched_p{p}")

            # cumulative deletions per priority
            for p in range(1, self.max_prio + 1):
                header.append(f"deletions_cum_p{p}")

            timestamp_writer.writerow(header)

            # pod_stats.csv header
            pod_stats_writer.writerow(["timestamp", "event", "pod_name", "pod_uid", "priority", "time_s"])

            # Don't force an immediate opt-stats dump on startup.
            last_opt_dump_s = float(self.time_s())

            # monitoring loop
            while not stop_event.is_set():
                loop_start = float(self.clock.time())
                now_ts = get_timestamp()
                t_s = self.time_s()

                # Dump optimization-stats every 30s (observable)
                if (t_s - last_opt_dump_s) >= OPT_STATS_DUMP_INTERVAL_S:
                    did_write, updated_at = self.dump_optimization_stats()
                    if did_write:
                        LOG.info("opt-stats: dumped (t=%.1fs, updated_at=%s) -> %s",
                                t_s, updated_at, self.optimization_stats_path)
                    last_opt_dump_s = t_s

                try:
                    cpu_run_util, mem_run_util, cpu_req_util, mem_req_util, running_by_prio, pending_by_prio, pod_items = \
                        self.snapshot_from_pods(namespace)
                except Exception as e:
                    LOG.warning("monitor: snapshot failed: %s", e)
                    self.clock.sleep(interval_s)
                    continue
                
                #### deletions_cum_pK logic ###
                cur_live_uids: set[str] = set()
                cur_phase_by_uid: Dict[str, str] = {}

                for pod in pod_items:
                    meta = pod.get("metadata", {}) or {}
                    status = pod.get("status", {}) or {}
                    pod_uid = meta.get("uid", "")
                    pod_name = meta.get("name", "")
                    if not pod_uid or not pod_name:
                        continue

                    phase = str(status.get("phase", "") or "")
                    cur_live_uids.add(pod_uid)
                    cur_phase_by_uid[pod_uid] = phase

                    rs_name = rs_prefix_from_pod_name(pod_name)
                    prio = int(self.prio_by_rs.get(rs_name, 0))
                    if prio > 0:
                        uid_to_prio[pod_uid] = prio

                    # Once we've seen a UID Running, it becomes eligible to be counted deleted.
                    if phase == "Running":
                        eligible_uids.add(pod_uid)

                delta_by_prio: Dict[int, int] = {p: 0 for p in range(1, self.max_prio + 1)}

                # Running -> Pending transitions
                for uid in (prev_live_uids & cur_live_uids):
                    if uid not in eligible_uids:
                        continue
                    if prev_phase_by_uid.get(uid) == "Running" and cur_phase_by_uid.get(uid) == "Pending":
                        p = int(uid_to_prio.get(uid, 0))
                        if p in delta_by_prio:
                            delta_by_prio[p] += 1
                        eligible_uids.discard(uid)

                # Count + cleanup: if a previously-running UID disappears from the pod list,
                # treat it as a deletion event.
                gone = prev_live_uids - cur_live_uids
                if gone:
                    for uid in gone:
                        if uid in eligible_uids and prev_phase_by_uid.get(uid) == "Running":
                            p = int(uid_to_prio.get(uid, 0))
                            if p in delta_by_prio:
                                delta_by_prio[p] += 1
                        eligible_uids.discard(uid)
                        prev_phase_by_uid.pop(uid, None)
                        uid_to_prio.pop(uid, None)

                for p in range(1, self.max_prio + 1):
                    deletions_cum_by_prio[p] += delta_by_prio[p]

                prev_live_uids = cur_live_uids
                prev_phase_by_uid = cur_phase_by_uid
                ###############################

                #### Write general_stats row ##
                row = [
                    now_ts,
                    f"{t_s:.6f}",
                    f"{cpu_run_util:.6f}",
                    f"{mem_run_util:.6f}",
                    f"{cpu_req_util:.6f}",
                    f"{mem_req_util:.6f}",
                ]
                for p in range(1, self.max_prio + 1):
                    row.append(str(int(running_by_prio.get(p, 0))))
                for p in range(1, self.max_prio + 1):
                    row.append(str(int(pending_by_prio.get(p, 0))))
                for p in range(1, self.max_prio + 1):
                    row.append(str(int(deletions_cum_by_prio.get(p, 0))))

                timestamp_writer.writerow(row)
                f_ts.flush()

                #### pod_stats: apply-time + running-time (only pods from trace.json) ####
                for pod in pod_items:
                    meta = pod.get("metadata", {}) or {}
                    status = pod.get("status", {}) or {}
                    pod_uid = meta.get("uid", "")
                    pod_name = meta.get("name", "")
                    if not pod_name:
                        continue

                    rs_name = rs_prefix_from_pod_name(pod_name)
                    if rs_name in self.initial_rs_names:
                        # skip initial pods in pod_stats.csv
                        continue

                    prio = int(self.prio_by_rs.get(rs_name, 0))

                    # APPLY: first time we observe the pod
                    if pod_name not in seen_apply:
                        seen_apply.add(pod_name)
                        creation_time = self.pod_creation_time(pod)
                        if creation_time is not None and self.run_start_wall > 0:
                            apply_time_s = max(0.0, creation_time - self.run_start_wall)
                        else:
                            apply_time_s = float(t_s)
                        pod_stats_writer.writerow([now_ts, "apply-time", pod_name, pod_uid, prio, f"{apply_time_s:.6f}"])
                        f_ps.flush()

                    # RUNNING: first time we see it Running
                    phase = status.get("phase", "")
                    if phase == "Running" and pod_name not in seen_running:
                        seen_running.add(pod_name)
                        start_time = self.pod_start_time(pod)
                        if start_time is not None and self.run_start_wall > 0:
                            run_time_s = max(0.0, start_time - self.run_start_wall)
                        else:
                            run_time_s = float(t_s)
                        pod_stats_writer.writerow([now_ts, "running-time", pod_name, pod_uid, prio, f"{run_time_s:.6f}"])
                        f_ps.flush()

                elapsed = float(self.clock.time()) - loop_start
                sleep_s = max(0.0, interval_s - elapsed)
                if sleep_s > 0:
                    self.clock.sleep(sleep_s)

        LOG.info("monitor: stop signal received; exiting")

    # ------------------------------
    # Runner
    # ------------------------------

    def run_seed(self) -> None:
        """
        Run a single trace replay for the current trace_dir/result_dir.
        """
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
        self.build_trace_events()

        # KWOK config
        kwok_cfg_path = Path(self.args.kwokctl_config_file).resolve()
        with open(kwok_cfg_path, "r", encoding="utf-8") as f:
            config_doc = yaml.safe_load(f) or {}
        if self.override_kwokctl_envs:
            config_doc = merge_kwokctl_envs(config_doc, self.override_kwokctl_envs)

        # Create KWOK cluster
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

        # Monitoring + Replay
        stop_event = threading.Event()
        monitor_thread: Optional[threading.Thread] = None

        try:
            # 1) Apply initial pods now
            self.apply_initial_pods(self.args.namespace)

            # Anchor sim_time_s=0 at the moment the initial pods have been
            # applied. start_delay is modeled as an offset applied only to trace
            # pod events.
            trace_start_wall = float(self.clock.time())

            # 2) Start monitor AFTER initial pods have been applied
            monitor_thread = threading.Thread(
                target=self.monitor_loop,
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

            # 3) Replay trace aligned so sim_time 0 happens at trace_start_wall
            self.replay_trace_events(namespace=self.args.namespace, trace_start_wall=trace_start_wall)

        # 4) Cleanup
        finally:
            stop_event.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=10.0)
            LOG.info("monitor thread joined; done.")

            # Final best-effort snapshot
            self.dump_optimization_stats()

            if self.args.save_scheduler_logs:
                LOG.info("saving scheduler logs via kwokctl...")
                self.save_scheduler_logs()

        LOG.info("Done.")

    def run(self) -> None:
        """
        Run the trace replayer for one or more seed directories.
        """
        trace_dir = Path(self.args.trace_dir).resolve()
        run_dirs = discover_trace_run_dirs(trace_dir)
        base_results_dir = Path(self.args.result_dir).resolve()

        # Determine if multi-seed mode
        multi_seed = len(run_dirs) > 1 or run_dirs[0] != trace_dir
        if multi_seed:
            LOG.info("discovered %d seed trace directories under %s", len(run_dirs), trace_dir)

        # Run each seed
        for run_dir in run_dirs:
            if multi_seed:
                seed_name = run_dir.name
                
                # Skip seeds in SKIP_SEEDS
                try:
                    seed_num = int(seed_name.replace("seed-", ""))
                    if seed_num in SKIP_SEEDS:
                        LOG.info("Skipping seed %d (in SKIP_SEEDS)", seed_num)
                        continue
                except ValueError:
                    pass  # Not a numeric seed, proceed normally
                
                run_result_dir = base_results_dir / seed_name
                header, footer = make_header_footer(f"SEED RUN {seed_name}")
                LOG.info(
                    "\n%s\nseed=%s trace_dir=%s result_dir=%s\n%s",
                    header,
                    seed_name,
                    run_dir,
                    run_result_dir,
                    footer,
                )
            else:
                run_result_dir = base_results_dir
                header, footer = make_header_footer("TRACE REPLAY")
                LOG.info("\n%s\ntrace_dir=%s result_dir=%s\n%s", header, run_dir, run_result_dir, footer)

            run_args = argparse.Namespace(**vars(self.args))
            run_args.trace_dir = str(run_dir)
            run_args.result_dir = str(run_result_dir)

            # Per-seed instance from already-resolved args; skips _initialize().
            replayer = TraceReplayer.__new__(TraceReplayer)
            replayer.init_runtime(runner=self.runner, clock=self.clock, executor_factory=self.executor_factory)
            replayer.init_context(
                args=run_args,
                job_doc=self.job_doc,
                override_kwokctl_envs=self.override_kwokctl_envs,
            )
            replayer.configure_from_args()
            replayer.write_info_file()
            replayer.run_seed()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point for trace replayer.
    """
    args = build_argparser().parse_args()
    trace_replayer = TraceReplayer(args)
    trace_replayer.run()

if __name__ == "__main__":
    main()