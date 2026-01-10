#!/usr/bin/env python3
# trace_generator.py

"""
python -m scripts.kwok_trace_replayer.trace_generator \
--job-file <job-file.yaml>
"""

import argparse, copy, heapq, json, logging, os, yaml
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.helpers.general_helpers import (
    parse_duration_to_seconds, setup_logging, build_cli_cmd,
    write_info_file, log_args_block, derive_seed, read_seeds_file,
)
from scripts.helpers.job_helpers import (
    JobField,
    merge_job_fields_into_args,
    parse_optional_bool, parse_optional_float, parse_optional_int, parse_optional_str,
)
from scripts.kwok_trace_replayer.trace_helpers import TraceRecord
from scripts.kwok_trace_replayer.plot_helpers import (
    plot_generator_histograms, plot_utilization_time_series,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SOLVE_ALPHA_MAX_ITERATIONS = 10_000
SOLVE_ALPHA_TOLERANCE = 1e-3
SOLVE_ALPHA_SAMPLES = 50_000
SOLVE_ALPHA_LOWER_BOUND = 0.01
SOLVE_ALPHA_UPPER_BOUND = 20.0

UTIL_TOL = 0.01
CALIB_MEAN_LIFE_MAX_ITER = 200
INITIAL_FILL_TOL = 0.002
INITIAL_MAX_PODS = 200_000

MAX_DECIMALS = 6

LOGGER_NAME = "trace-generator"
LOG = logging.getLogger(LOGGER_NAME)

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SHOW_PLOTS = False

# ---------------------------------------------------------------------
# Small state models
# ---------------------------------------------------------------------

@dataclass
class ClusterState:
    live_req: float = 0.0
    live_pods: int = 0

@dataclass(order=True)
class EndHeapEntry:
    end_time: float
    req: float = field(compare=False)
    replicas: int = field(compare=False)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.
    """
    p = argparse.ArgumentParser(description="Generate initial + trace workload JSON files.")

    # General
    p.add_argument("--job-file", dest="job_file", default=None,
                   help="Path to a YAML job file describing arguments. CLI overrides job file.")
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="Run exactly this seed.")
    p.add_argument("--seed-file", dest="seed_file", default=None,
                   help="Path to a seed list file (one integer per line).")
    p.add_argument("--log-level", dest="log_level", default=None)

    p.add_argument(
        "--show-plots",
        dest="show_plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show plots after generation (plots are always saved).",
    )

    # Cluster / horizon
    p.add_argument("--num-nodes", type=int, default=None)
    p.add_argument("--trace-time", type=str, default=None)
    p.add_argument("--target-util", type=float, default=None)

    # Inter-arrival (Pareto)
    p.add_argument("--xmin-arrival", type=float, default=None)
    p.add_argument("--xmax-arrival", type=float, default=None)
    p.add_argument("--mean-arrival", type=float, default=None)

    # Lifetime bounds (mean is inferred from util)
    p.add_argument("--xmin-life", type=float, default=None)
    p.add_argument("--xmax-life", type=float, default=None)
    p.add_argument("--mean-life", type=float, default=None)

    # Requests (Pareto)
    p.add_argument("--xmin-req", type=float, default=None)
    p.add_argument("--xmax-req", type=float, default=None)
    p.add_argument("--mean-req", type=float, default=None)

    # Priority + replicas (geometric/uniform)
    p.add_argument("--priority-min", type=int, default=None)
    p.add_argument("--priority-max", type=int, default=None)
    p.add_argument("--priority-ratio", type=float, default=None)

    p.add_argument("--replicas-min", type=int, default=None)
    p.add_argument("--replicas-max", type=int, default=None)
    p.add_argument("--replicas-ratio", type=float, default=None)

    return p

def _load_job_doc(path: str | Path) -> dict:
    """
    Load and validate a YAML job file as a mapping.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise SystemExit(f"--job-file not found: {p}")
    try:
        with open(p, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        raise SystemExit(f"failed reading --job-file {p}: {e}")
    if not isinstance(doc, dict):
        raise SystemExit(f"--job-file must be a YAML mapping/object: {p}")
    return doc

def _merge_job_fields(args: argparse.Namespace, job: dict) -> argparse.Namespace:
    """
    Merge supported job-file fields into an argparse namespace.
    """
    fields = [
        JobField("output-dir", "output_dir", parse=parse_optional_str),
        JobField("seed", "seed", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
        JobField("seed-file", "seed_file", parse=parse_optional_str),
        JobField("log-level", "log_level", parse=parse_optional_str),
        JobField("show-plots", "show_plots", parse=parse_optional_bool, accept=lambda v: v is not None),

        JobField("num-nodes", "num_nodes", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
        JobField("trace-time", "trace_time", parse=parse_optional_str),
        JobField("target-util", "target_util", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),

        JobField("xmin-arrival", "xmin_arrival", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
        JobField("xmax-arrival", "xmax_arrival", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
        JobField("mean-arrival", "mean_arrival", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),

        JobField("xmin-life", "xmin_life", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
        JobField("xmax-life", "xmax_life", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
        JobField("mean-life", "mean_life", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),

        JobField("xmin-req", "xmin_req", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
        JobField("xmax-req", "xmax_req", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
        JobField("mean-req", "mean_req", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),

        JobField("priority-min", "priority_min", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
        JobField("priority-max", "priority_max", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
        JobField("priority-ratio", "priority_ratio", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),

        JobField("replicas-min", "replicas_min", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
        JobField("replicas-max", "replicas_max", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
        JobField("replicas-ratio", "replicas_ratio", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
    ]
    return merge_job_fields_into_args(args, job or {}, fields)

def _apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply default values for optional arguments.
    """
    if getattr(args, "log_level", None) is None:
        args.log_level = DEFAULT_LOG_LEVEL
    if getattr(args, "show_plots", None) is None:
        args.show_plots = bool(DEFAULT_SHOW_PLOTS)
    return args

def _validate_args(args: argparse.Namespace) -> None:
    """
    Validate that required arguments are present and consistent.
    """
    missing: list[str] = []

    # Output location is always required.
    if getattr(args, "output_dir", None) is None:
        missing.append("output_dir")

    # Seed selection: require exactly one mode.
    seed = getattr(args, "seed", None)
    seed_file = getattr(args, "seed_file", None)
    if seed is not None and seed_file:
        raise SystemExit("--seed and --seed-file cannot be used together")
    if seed is None and not seed_file:
        missing.append("seed or seed_file")

    # Everything else must be explicitly provided.
    for k in (
        "num_nodes",
        "trace_time",
        "target_util",
        "xmin_arrival",
        "mean_arrival",
        "xmin_life",
        "xmin_req",
        "xmax_req",
        "mean_req",
        "priority_min",
        "priority_max",
        "priority_ratio",
        "replicas_min",
        "replicas_max",
        "replicas_ratio",
    ):
        if getattr(args, k, None) is None:
            missing.append(k)

    if missing:
        raise SystemExit(f"missing required arguments (via CLI or job-file): {', '.join(missing)}")

    if getattr(args, "job_file", None):
        p = Path(args.job_file).resolve()
        if not p.exists():
            raise SystemExit(f"--job-file not found: {p}")

    if getattr(args, "seed_file", None):
        p = Path(args.seed_file).resolve()
        if not p.exists():
            raise SystemExit(f"--seed-file not found: {p}")

def resolve_effective_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    """
    Resolve effective arguments with CLI > job-file > defaults precedence.
    """
    args = cli_args
    job_doc = _load_job_doc(args.job_file) if getattr(args, "job_file", None) else None
    if job_doc is not None:
        args = _merge_job_fields(args, job_doc)
    args = _apply_defaults(args)
    _validate_args(args)
    return args

def expand_seed_runs(args: argparse.Namespace) -> list[argparse.Namespace]:
    """
    Expand a resolved args namespace into one namespace per seed run.
    """
    if getattr(args, "seed_file", None):
        seeds = read_seeds_file(Path(args.seed_file).resolve(), logger=LOG)
    else:
        seeds = [int(args.seed)]
    base_out = Path(args.output_dir).resolve()
    runs: list[argparse.Namespace] = []
    for s in seeds:
        a = copy.copy(args)
        a.seed = int(s)
        if getattr(args, "seed_file", None):
            a.output_dir = str(base_out / str(int(s)))
        runs.append(a)
    return runs

def round_float_args(args: argparse.Namespace, ndigits: int) -> None:
    """
    Round all float fields on an argparse namespace in-place.
    """
    for k, v in vars(args).items():
        if isinstance(v, float):
            setattr(args, k, round(v, ndigits))

# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------
class TraceGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize generator state and output paths from resolved args.
        """
        self.args = args
        self.base_seed = int(args.seed)

        self.trace_time_s = float(parse_duration_to_seconds(self.args.trace_time))

        self.output_dir = Path(self.args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Outputs
        self.initial_path = self.output_dir / "initial.json"
        self.trace_path = self.output_dir / "trace.json"
        self.info_path = self.output_dir / "info_generate.yaml"
        self.util_plot_path = self.figures_dir / "utilization.png"
        self.hist_plot_path = self.figures_dir / "histograms.png"

        # Pareto alphas
        self.alpha_req: Optional[float] = None
        self.alpha_arrival: Optional[float] = None
        self.alpha_life: Optional[float] = None

        # For plotting / logging
        self.times: List[float] = []
        self.u_req_hist: List[float] = []
        self.pods_hist: List[int] = []
        self.initial_pods_count: int = 0

    # -------------------------
    # Logging / metadata
    # -------------------------
    
    def log_args(self) -> None:
        """
        Log key arguments in a stable order for reproducibility.
        """
        include = [
            "job_file",
            "output_dir",
            "seed",
            "seed_file",
            "log_level",
            "show_plots",
            "num_nodes",
            "trace_time",
            "xmin_arrival",
            "xmax_arrival",
            "mean_arrival",
            "xmin_life",
            "xmax_life",
            "target_util",
            "xmin_req",
            "xmax_req",
            "mean_req",
            "priority_min",
            "priority_max",
            "priority_ratio",
            "replicas_min",
            "replicas_max",
            "replicas_ratio",
        ]
        log_args_block(LOG, self.args, title="ARGS", include=include)

    def _write_info_file(self, extra: Dict[str, object]) -> None:
        """
        Write generation metadata to info_generate.yaml.
        """
        try:
            inputs = {
                "cli-cmd": build_cli_cmd(),
                "args": {k: v for k, v in vars(self.args).items()},
                "generated": extra,
            }
            write_info_file(self.info_path, inputs=inputs, logger=LOG)
            LOG.info("wrote %s", self.info_path)
        except Exception as e:
            LOG.warning("failed to write info_generate.yaml: %s", e)

    # -------------------------
    # Distribution helpers
    # -------------------------
    
    @staticmethod
    def _sample_pareto(
        rng: np.random.Generator,
        alpha: float,
        x_min: float,
        x_max: Optional[float],
        size: int = 1,
    ) -> np.ndarray:
        """
        Sample Pareto-distributed values with an optional upper bound.
        """
        if alpha <= 0 or x_min <= 0:
            raise ValueError("Pareto alpha and x_min must be > 0.")
        if x_max is not None and x_max < x_min:
            raise ValueError("Pareto x_max must be >= x_min.")

        u = rng.random(size)
        if x_max is not None:
            u_min_tail = (x_min / x_max) ** alpha
            u = np.clip(u, max(1e-12, float(u_min_tail)), 1.0)
        else:
            u = np.clip(u, 1e-12, 1.0)

        return x_min / (u ** (1.0 / alpha))

    @classmethod
    def _solve_alpha_for_mean(
        cls,
        rng: np.random.Generator,
        x_min: float,
        x_max: Optional[float],
        target_mean: float,
    ) -> float:
        """
        Solve for Pareto alpha (via bisection + MC) that matches a target mean.
        """
        if x_min <= 0:
            raise ValueError("x_min must be > 0.")
        if x_max is not None and not (x_min < target_mean < x_max):
            raise ValueError(f"target_mean={target_mean} must be between x_min={x_min} and x_max={x_max}.")
        if x_max is None and target_mean <= x_min:
            raise ValueError(f"target_mean={target_mean} must be > x_min={x_min} for unbounded Pareto.")

        a_lo = SOLVE_ALPHA_LOWER_BOUND
        a_hi = SOLVE_ALPHA_UPPER_BOUND
        if x_max is None:
            a_lo = max(a_lo, 1.0001)

        m_lo = float(cls._sample_pareto(rng, a_lo, x_min, x_max, size=SOLVE_ALPHA_SAMPLES).mean())
        m_hi = float(cls._sample_pareto(rng, a_hi, x_min, x_max, size=SOLVE_ALPHA_SAMPLES).mean())
        if not (m_hi <= target_mean <= m_lo):
            raise ValueError(
                f"Target mean {target_mean} not bracketed by MC means [{m_hi:.4f}, {m_lo:.4f}] "
                f"for alpha in [{a_lo}, {a_hi}]. Adjust bounds."
            )

        for _ in range(SOLVE_ALPHA_MAX_ITERATIONS):
            a_mid = 0.5 * (a_lo + a_hi)
            m_mid = float(cls._sample_pareto(rng, a_mid, x_min, x_max, size=SOLVE_ALPHA_SAMPLES).mean())
            if m_mid >= target_mean:
                a_lo, m_lo = a_mid, m_mid
            else:
                a_hi, m_hi = a_mid, m_mid
            if abs(m_mid - target_mean) <= SOLVE_ALPHA_TOLERANCE * target_mean:
                return float(a_mid)

        raise RuntimeError("Failed to converge solving alpha for mean.")

    @staticmethod
    def _build_geometric_support(min_val: int, max_val: int, ratio: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build discrete support and probabilities for a truncated geometric distribution.
        """
        if ratio <= 0:
            raise ValueError("ratio must be > 0.")
        lo = int(min_val)
        hi = max(int(max_val), lo)
        vals = np.arange(lo, hi + 1, dtype=int)
        if vals.size == 1 or np.isclose(ratio, 1.0):
            return vals, None
        exps = np.arange(vals.size, dtype=float)
        w = ratio ** exps
        p = w / w.sum()
        return vals, p

    @staticmethod
    def _expected_value_for_support(vals: np.ndarray, probs: Optional[np.ndarray]) -> float:
        """
        Compute an expected value for integer support with optional probabilities.
        """
        return float(np.mean(vals)) if probs is None else float(np.sum(vals.astype(float) * probs.astype(float)))

    def infer_mean_life_from_target_util(self) -> float:
        """
        Infer mean pod lifetime to hit target utilization given arrival/req distributions.
        """
        mean_arrival = float(self.args.mean_arrival)
        if mean_arrival <= 0:
            raise ValueError("mean-arrival must be > 0.")
        lam = 1.0 / mean_arrival

        r_vals, r_probs = self._build_geometric_support(
            max(1, int(self.args.replicas_min)),
            max(1, int(self.args.replicas_max)),
            float(self.args.replicas_ratio),
        )
        e_rep = self._expected_value_for_support(r_vals, r_probs)

        e_req = float(self.args.mean_req)
        if e_req <= 0:
            raise ValueError("mean-req must be > 0.")

        U = float(self.args.target_util)
        if not (0.0 < U <= 1.0):
            raise ValueError("target-util must be in (0,1].")

        N = float(self.args.num_nodes)
        if N <= 0:
            raise ValueError("num-nodes must be > 0.")

        mean_life = (U * N) / (lam * e_rep * e_req)

        xmin = float(self.args.xmin_life)
        xmax = self.args.xmax_life
        if mean_life <= xmin:
            raise ValueError(f"Inferred mean_life={mean_life:.3f}s <= xmin-life={xmin}.")
        if xmax is not None and mean_life >= float(xmax):
            raise ValueError(f"Inferred mean_life={mean_life:.3f}s >= xmax-life={xmax}.")

        return float(mean_life)

    def _fit_alphas(self, rng: np.random.Generator) -> None:
        """
        Fit Pareto alphas for req/arrival/life to match configured means.
        """
        a_req = self._solve_alpha_for_mean(
            rng,
            float(self.args.xmin_req),
            float(self.args.xmax_req),
            float(self.args.mean_req),
        )
        a_arr = self._solve_alpha_for_mean(
            rng,
            float(self.args.xmin_arrival),
            self.args.xmax_arrival,
            float(self.args.mean_arrival),
        )
        a_life = self._solve_alpha_for_mean(
            rng,
            float(self.args.xmin_life),
            self.args.xmax_life,
            float(self.args.mean_life),
        )

        self.alpha_req = round(float(a_req), MAX_DECIMALS)
        self.alpha_arrival = round(float(a_arr), MAX_DECIMALS)
        self.alpha_life = round(float(a_life), MAX_DECIMALS)

        # store fitted params on args for plotting helpers
        self.args.alpha_req = float(a_req)
        self.args.alpha_arrival = float(a_arr)
        self.args.alpha_life = float(a_life)

        LOG.info(
            "[pareto-fit] req:     mean=%.4f xmin=%.4f xmax=%.4f alpha≈%.4f",
            float(self.args.mean_req),
            float(self.args.xmin_req),
            float(self.args.xmax_req),
            float(a_req),
        )
        LOG.info(
            "[pareto-fit] arrival: mean=%.4f xmin=%.4f xmax=%s alpha≈%.4f",
            float(self.args.mean_arrival),
            float(self.args.xmin_arrival),
            str(self.args.xmax_arrival),
            float(a_arr),
        )
        LOG.info(
            "[pareto-fit] life:    mean=%.4f xmin=%.4f xmax=%s alpha≈%.4f",
            float(self.args.mean_life),
            float(self.args.xmin_life),
            str(self.args.xmax_life),
            float(a_life),
        )

    # -------------------------
    # Util metric
    # -------------------------
    
    def _time_avg_req_util(self, pods: List[TraceRecord]) -> float:
        """
        Compute time-average requested utilization over the trace horizon.
        """
        T = float(self.trace_time_s)
        if T <= 0:
            return 0.0
        N = float(self.args.num_nodes)

        area = 0.0
        for p in pods:
            s = float(p.start_time)
            e = float(p.end_time)
            if e <= 0.0 or s >= T:
                continue
            active = max(0.0, min(e, T) - max(s, 0.0))
            area += float(p.replicas) * float(p.cpu) * active
        return area / (N * T)

    # -------------------------
    # Initial load generation (load at t=0)
    # -------------------------
    
    def _build_initial_load(
        self,
        rng: np.random.Generator,
        prio_vals: np.ndarray,
        prio_probs: Optional[np.ndarray],
        rep_vals: np.ndarray,
        rep_probs: Optional[np.ndarray],
        next_id: int,
    ) -> Tuple[List[TraceRecord], int]:
        """
        Generate a steady-state initial pod set approximating target utilization.
        """
        assert self.alpha_req is not None and self.alpha_life is not None

        target_req_total = float(self.args.target_util) * float(self.args.num_nodes)
        tol = float(INITIAL_FILL_TOL)
        max_pods = int(INITIAL_MAX_PODS)

        cur_total = 0.0
        pods: List[TraceRecord] = []

        while cur_total < target_req_total * (1.0 - tol):
            if len(pods) >= max_pods:
                LOG.warning(
                    "initial snapshot hit initial-max-pods=%d; stopping fill at util=%.3f",
                    max_pods,
                    cur_total / float(self.args.num_nodes),
                )
                break

            req = float(
                self._sample_pareto(
                    rng,
                    float(self.args.alpha_req),
                    float(self.args.xmin_req),
                    float(self.args.xmax_req),
                    1,
                )[0]
            )
            L = float(
                self._sample_pareto(
                    rng,
                    float(self.args.alpha_life),
                    float(self.args.xmin_life),
                    self.args.xmax_life,
                    1,
                )[0]
            )

            age = float(rng.random()) * L
            remaining = max(0.0, L - age)
            if remaining <= 1e-9:
                continue

            prio = int(rng.choice(prio_vals, p=prio_probs))
            reps = int(rng.choice(rep_vals, p=rep_probs))

            add = reps * req
            cur_total += add

            next_id += 1
            pods.append(
                TraceRecord(
                    id=next_id,
                    start_time=0.0,
                    end_time=round(remaining, MAX_DECIMALS),
                    cpu=round(req, MAX_DECIMALS),
                    mem=round(req, MAX_DECIMALS),
                    priority=prio,
                    replicas=reps,
                )
            )

        while pods and cur_total > target_req_total * (1.0 + tol):
            last = pods.pop()
            cur_total -= float(last.replicas) * float(last.cpu)
            next_id -= 1 

        LOG.info(
            "initial snapshot: pods=%d req_total=%.3f util=%.3f (target=%.3f tol=±%.3f)",
            len(pods),
            cur_total,
            cur_total / float(self.args.num_nodes),
            float(self.args.target_util),
            tol,
        )

        return pods, next_id

    # -------------------------
    # Trace generation (events from t>=0)
    # -------------------------
    
    def _generate_trace_events(
        self,
        rng: np.random.Generator,
        prio_vals: np.ndarray,
        prio_probs: Optional[np.ndarray],
        rep_vals: np.ndarray,
        rep_probs: Optional[np.ndarray],
        next_id: int,
        initial_pods: List[TraceRecord],
    ) -> Tuple[List[TraceRecord], int, List[float], List[float], List[int]]:
        """
        Generate trace pods (t>=0) and utilization series seeded by initial pods.
        """
        
        assert self.alpha_req is not None and self.alpha_arrival is not None and self.alpha_life is not None

        state = ClusterState()
        end_heap: List[EndHeapEntry] = []

        # Seed baseline from initial pods
        for p in initial_pods:
            req = float(p.cpu)
            reps = int(p.replicas)
            state.live_req += reps * req
            state.live_pods += reps
            heapq.heappush(end_heap, EndHeapEntry(end_time=float(p.end_time), req=req, replicas=reps))

        pods: List[TraceRecord] = []

        # Series starts at baseline at t=0
        times: List[float] = [0.0]
        u_hist: List[float] = [state.live_req / float(self.args.num_nodes)]
        pods_hist: List[int] = [state.live_pods]

        t = 0.0

        while True:
            if t >= self.trace_time_s:
                break

            dt = float(
                self._sample_pareto(
                    rng,
                    float(self.args.alpha_arrival),
                    float(self.args.xmin_arrival),
                    self.args.xmax_arrival,
                    1,
                )[0]
            )
            start = t + dt
            if start >= self.trace_time_s:
                break

            while end_heap and end_heap[0].end_time <= start:
                entry = heapq.heappop(end_heap)
                state.live_req = max(0.0, state.live_req - entry.req * entry.replicas)
                state.live_pods = max(0, state.live_pods - entry.replicas)

            life = float(
                self._sample_pareto(
                    rng,
                    float(self.args.alpha_life),
                    float(self.args.xmin_life),
                    self.args.xmax_life,
                    1,
                )[0]
            )
            end = start + life

            req = float(
                self._sample_pareto(
                    rng,
                    float(self.args.alpha_req),
                    float(self.args.xmin_req),
                    float(self.args.xmax_req),
                    1,
                )[0]
            )

            prio = int(rng.choice(prio_vals, p=prio_probs))
            reps = int(rng.choice(rep_vals, p=rep_probs))

            state.live_req += reps * req
            state.live_pods += reps
            heapq.heappush(end_heap, EndHeapEntry(end_time=end, req=req, replicas=reps))

            next_id += 1
            pods.append(
                TraceRecord(
                    id=next_id,
                    start_time=round(start, MAX_DECIMALS),
                    end_time=round(end, MAX_DECIMALS),
                    cpu=round(req, MAX_DECIMALS),
                    mem=round(req, MAX_DECIMALS),
                    priority=prio,
                    replicas=reps,
                )
            )

            times.append(start)
            u_hist.append(state.live_req / float(self.args.num_nodes))
            pods_hist.append(state.live_pods)

            t = start

        return pods, next_id, times, u_hist, pods_hist

    # -------------------------
    # Calibration loop
    # -------------------------
    
    def _generate_tracedata_once(self, iter_seed: int) -> Tuple[List[TraceRecord], List[TraceRecord], Dict[str, float], Dict[str, object]]:
        """
        Generate one candidate initial+trace dataset and associated metadata.
        """
        iter_seed_int = int(iter_seed)
        rng_fit = np.random.default_rng(derive_seed(iter_seed_int, "fit-alphas"))
        rng_initial = np.random.default_rng(derive_seed(iter_seed_int, "initial-snapshot"))
        rng_trace = np.random.default_rng(derive_seed(iter_seed_int, "trace-events"))

        self._fit_alphas(rng_fit)

        prio_vals, prio_probs = self._build_geometric_support(
            int(self.args.priority_min),
            int(self.args.priority_max),
            float(self.args.priority_ratio),
        )
        rep_vals, rep_probs = self._build_geometric_support(
            int(self.args.replicas_min),
            int(self.args.replicas_max),
            float(self.args.replicas_ratio),
        )

        next_id = 0

        initial_pods, next_id = self._build_initial_load(
            rng_initial,
            prio_vals,
            prio_probs,
            rep_vals,
            rep_probs,
            next_id=next_id,
        )
        trace_pods, next_id, times, u_hist, pods_hist = self._generate_trace_events(
            rng_trace,
            prio_vals,
            prio_probs,
            rep_vals,
            rep_probs,
            next_id=next_id,
            initial_pods=initial_pods,
        )

        self.times = times
        self.u_req_hist = u_hist
        self.pods_hist = pods_hist
        self.initial_pods_count = int(sum(int(p.replicas) for p in initial_pods))

        all_pods = initial_pods + trace_pods
        util = self._time_avg_req_util(all_pods)

        stats = {
            "util_time_avg": float(util),
            "initial_count": float(len(initial_pods)),
            "trace_count": float(len(trace_pods)),
        }

        # Extra info that we want in info_generate.yaml
        max_prio = 0
        for p in all_pods:
            max_prio = max(max_prio, int(p.priority))

        extra_info: Dict[str, object] = {
            "num_nodes": int(self.args.num_nodes),
            "trace_time_s": round(float(self.trace_time_s), 6),
            "seed": int(self.args.seed),
            "rng": {
                "iter_seed": int(iter_seed_int),
                "derived": {
                    "fit_alphas": int(derive_seed(iter_seed_int, "fit-alphas")),
                    "initial_snapshot": int(derive_seed(iter_seed_int, "initial-snapshot")),
                    "trace_events": int(derive_seed(iter_seed_int, "trace-events")),
                },
            },
            "measured_util_time_avg": float(util),
            "target_util_time_avg": float(self.args.target_util),
            "calibration": {
                "util_tol": float(UTIL_TOL),
                "calib_max_iter": int(CALIB_MEAN_LIFE_MAX_ITER),
                "initial_fill_tol": float(INITIAL_FILL_TOL),
                "initial_max_pods": int(INITIAL_MAX_PODS),
            },
            "derived_mean_life_s": round(float(self.args.mean_life), 6),
            "pareto": {
                "req": {
                    "alpha": float(self.alpha_req) if self.alpha_req is not None else None,
                    "xmin": float(self.args.xmin_req),
                    "xmax": float(self.args.xmax_req),
                    "mean": float(self.args.mean_req),
                },
                "arrival": {
                    "alpha": float(self.alpha_arrival) if self.alpha_arrival is not None else None,
                    "xmin": float(self.args.xmin_arrival),
                    "xmax": self.args.xmax_arrival,
                    "mean": float(self.args.mean_arrival),
                },
                "life": {
                    "alpha": float(self.alpha_life) if self.alpha_life is not None else None,
                    "xmin": float(self.args.xmin_life),
                    "xmax": self.args.xmax_life,
                    "mean": float(self.args.mean_life),
                },
            },
            "priority": {
                "min": int(self.args.priority_min),
                "max": int(self.args.priority_max),
                "ratio": float(self.args.priority_ratio),
            },
            "replicas": {
                "min": int(self.args.replicas_min),
                "max": int(self.args.replicas_max),
                "ratio": float(self.args.replicas_ratio),
            },
            "counts": {
                "initial_records": int(len(initial_pods)),
                "trace_records": int(len(trace_pods)),
                "initial_replicas_total": int(sum(int(p.replicas) for p in initial_pods)),
                "trace_replicas_total": int(sum(int(p.replicas) for p in trace_pods)),
            },
            "max_priority_seen": int(max_prio),
            "id_ranges": {
                "initial_min_id": int(min((p.id for p in initial_pods), default=0)),
                "initial_max_id": int(max((p.id for p in initial_pods), default=0)),
                "trace_min_id": int(min((p.id for p in trace_pods), default=0)),
                "trace_max_id": int(max((p.id for p in trace_pods), default=0)),
                "global_max_id": int(max((p.id for p in all_pods), default=0)),
            },
            "files": {
                "initial_json": str(self.initial_path),
                "trace_json": str(self.trace_path),
                "utilization_plot": str(self.util_plot_path),
                "histograms_plot": str(self.hist_plot_path),
            },
        }

        return initial_pods, trace_pods, stats, extra_info

    def calibrate_mean_life(self) -> Tuple[List[TraceRecord], List[TraceRecord], Dict[str, object]]:
        """
        Iteratively calibrate mean-life to match target utilization within tolerance.
        """
        target = float(self.args.target_util)
        tol = float(UTIL_TOL)
        max_iter = int(CALIB_MEAN_LIFE_MAX_ITER)

        best_initial: List[TraceRecord] = []
        best_trace: List[TraceRecord] = []
        best_err = float("inf")
        best_extra: Dict[str, object] = {}

        for it in range(1, max_iter + 1):
            iter_seed = int(derive_seed(self.base_seed, "calibrate-mean-life", it))
            initial_pods, trace_pods, stats, extra_info = self._generate_tracedata_once(iter_seed)

            measured = float(stats["util_time_avg"])
            err = abs(measured - target) / max(1e-12, target)
            best_err = min(best_err, err)

            LOG.info(
                "[calibrate-mean-life] iter=%d measured=%.3f target=%.3f rel_err=%.2f%% (initial=%d trace=%d)",
                it,
                measured,
                target,
                100.0 * err,
                len(initial_pods),
                len(trace_pods),
            )

            best_initial, best_trace, best_extra = initial_pods, trace_pods, extra_info
            if err <= tol:
                return best_initial, best_trace, best_extra

            if measured <= 1e-12:
                raise RuntimeError("Measured utilization is ~0; cannot calibrate mean-life.")

            new_mean_life = float(self.args.mean_life) * (target / measured)

            xmin = float(self.args.xmin_life)
            xmax = self.args.xmax_life
            if new_mean_life <= xmin:
                new_mean_life = xmin * 1.001
            if xmax is not None and new_mean_life >= float(xmax):
                new_mean_life = float(xmax) * 0.999

            self.args.mean_life = float(new_mean_life)
            LOG.info("[calibrate-mean-life] updated mean-life=%.3fs", float(self.args.mean_life))

        LOG.warning("[calibrate-mean-life] calibration did not reach util-tol; best rel_err=%.2f%%", 100.0 * best_err)
        return best_initial, best_trace, best_extra

    # -------------------------
    # Output writers
    # -------------------------
    
    def _write_json(self, path: Path, pods: List[TraceRecord]) -> None:
        """
        Write a list of TraceRecords to a JSON file.
        """
        obj = {"pods": [asdict(p) for p in pods]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        LOG.info("wrote %s (%d records)", path, len(pods))

    def write_outputs(self, initial_pods: List[TraceRecord], trace_pods: List[TraceRecord], extra_info: Dict[str, object]) -> None:
        """
        Write generated JSON outputs and metadata to disk.
        """
        all_pods = initial_pods + trace_pods
        util_time = self._time_avg_req_util(all_pods)

        self._write_json(self.initial_path, initial_pods)
        self._write_json(self.trace_path, trace_pods)

        LOG.info(
            "[utilization] time-avg over whole horizon: %.4f (target=%.4f)",
            float(util_time),
            float(self.args.target_util),
        )

        # Persist all metadata into info_generate.yaml
        self._write_info_file(extra=extra_info)

    # -------------------------
    # Runner
    # -------------------------
    
    def run(self) -> None:
        """
        Run the full generation workflow for a single seed.
        """
        if self.args.mean_life is None:
            self.args.mean_life = self.infer_mean_life_from_target_util()
            LOG.info(
                "inferred mean_life=%.3fs from target-util=%.3f",
                float(self.args.mean_life),
                float(self.args.target_util),
            )
        self.log_args()

        initial_pods, trace_pods, extra_info = self.calibrate_mean_life()

        all_pods = initial_pods + trace_pods
        self.write_outputs(initial_pods, trace_pods, extra_info)

        plot_utilization_time_series(
            times=self.times,
            u_req_hist=self.u_req_hist,
            pods_hist=self.pods_hist,
            all_pods=all_pods,
            initial_pods_count=int(self.initial_pods_count),
            out_path=str(self.util_plot_path),
            show_plots=bool(getattr(self.args, "show_plots", False)),
            logger=LOG,
        )
        plot_generator_histograms(
            all_pods=all_pods,
            out_path=str(self.hist_plot_path),
            alpha_arrival=float(self.args.alpha_arrival),
            xmin_arrival=float(self.args.xmin_arrival),
            xmax_arrival=self.args.xmax_arrival,
            alpha_life=float(self.args.alpha_life),
            xmin_life=float(self.args.xmin_life),
            xmax_life=self.args.xmax_life,
            alpha_req=float(self.args.alpha_req),
            xmin_req=float(self.args.xmin_req),
            xmax_req=float(self.args.xmax_req),
            priority_ratio=float(self.args.priority_ratio),
            priority_min=int(self.args.priority_min),
            priority_max=int(self.args.priority_max),
            replicas_ratio=float(self.args.replicas_ratio),
            replicas_min=int(self.args.replicas_min),
            replicas_max=int(self.args.replicas_max),
            show_plots=bool(getattr(self.args, "show_plots", False)),
            logger=LOG,
        )

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if os.getenv("TRACE_GENERATOR_NOOP") == "1":
        return
    cli_args = build_arg_parser().parse_args()
    args = resolve_effective_args(cli_args)
    round_float_args(args, MAX_DECIMALS)
    setup_logging(name=LOGGER_NAME, prefix=f"[{LOGGER_NAME}] ", level=args.log_level)

    runs = expand_seed_runs(args)
    total = len(runs)
    # Loop over all runs (different seeds)
    for i, a in enumerate(runs, start=1):
        if total > 1:
            LOG.info("run %d/%d seed=%d output_dir=%s", i, total, int(a.seed), a.output_dir)
        gen = TraceGenerator(a)
        gen.run()
    
    # All runs (seeds) done
    LOG.info("done.")

if __name__ == "__main__":
    main()
