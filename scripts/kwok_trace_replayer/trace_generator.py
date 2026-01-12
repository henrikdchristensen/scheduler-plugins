#!/usr/bin/env python3
# trace_generator.py

"""
python -m scripts.kwok_trace_replayer.trace_generator --job-file <job-file.yaml>

High-level flow per seed:
  1) resolve args (CLI > job-file), validate (always bounded params)
  2) if mean_life unset: infer it from target util (Little's law style)
  3) calibrate mean_life: generate -> measure util -> proportional update
  4) write initial.json + trace.json + info_generate.yaml + plots

NOTE (lifetime alignment):
    - Initial pods use residual-life sampling (stationary snapshot): we sample a length-biased
        total lifetime and then take a uniform residual remaining lifetime.
    - Trace pods still use fresh lifetime draws from the bounded Pareto.
    - We enforce xmin-life >= 2s (hard requirement for the lifetime distribution bounds).
"""

import os, argparse, math, copy, heapq, json, logging, yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from scripts.helpers.general_helpers import (
    build_cli_cmd, derive_seed, log_args_block, make_header_footer,
    parse_duration_to_seconds, read_seeds_file, setup_logging, write_info_file,
)
from scripts.helpers.job_helpers import (
    JobField, merge_job_fields_into_args,
    parse_optional_bool, parse_optional_duration_seconds, parse_optional_float,
    parse_optional_int, parse_optional_str,
)
from scripts.kwok_trace_replayer.plot_helpers import (
    plot_generator_histograms, plot_utilization_and_num_pods,
)
from scripts.kwok_trace_replayer.trace_helpers import TraceRecord

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MAX_DECIMALS = 6

MIN_LIFETIME_S = 2.0

MEAN_LIFE_CALIBRATION_UTIL_TOLERANCE = 0.01
MEAN_LIFE_CALIBRATION_MAX_ITER = 20

INITIAL_PODS_UTIL_TOLERANCE = 0.002
INITIAL_MAX_PODS = 200_000

ALPHA_SOLVE_MAX_ITER = 200
ALPHA_SOLVE_TOLERANCE = 1e-10

LOGGER_NAME = "trace-generator"
LOG = logging.getLogger(LOGGER_NAME)

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SHOW_PLOTS = False


# -----------------------------------------------------------------------------
# Small state models
# -----------------------------------------------------------------------------
@dataclass
class ClusterState:
    live_req: float = 0.0
    live_pods: int = 0


@dataclass(order=True)
class EndHeapEntry:
    end_time: float
    req: float = field(compare=False)
    replicas: int = field(compare=False)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate initial + trace workload JSON files.")

    # General
    p.add_argument("--job-file", dest="job_file", default=None,
                   help="Path to a YAML job file describing arguments. CLI overrides job file.")
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.add_argument("--seed", type=int, default=None, help="Run exactly this seed.")
    p.add_argument("--seed-file", dest="seed_file", default=None,
                   help="Path to a seed list file (one integer per line).")
    p.add_argument("--log-level", dest="log_level", default=None)
    p.add_argument("--show-plots", dest="show_plots", action=argparse.BooleanOptionalAction, default=None,
        help="Show plots after generation (plots are always saved).",
    )

    # Cluster / horizon
    p.add_argument("--num-nodes", type=int, default=None)
    p.add_argument("--trace-time", type=str, default=None)
    p.add_argument("--target-util", type=float, default=None)

    # Inter-arrival (bounded Pareto)
    p.add_argument("--xmin-arrival", type=lambda s: float(parse_duration_to_seconds(s)), default=None)
    p.add_argument("--xmax-arrival", type=lambda s: float(parse_duration_to_seconds(s)), default=None)
    p.add_argument("--mean-arrival", type=lambda s: float(parse_duration_to_seconds(s)), default=None)

    # Lifetime (bounded Pareto; mean inferred/calibrated)
    p.add_argument("--xmin-life", type=lambda s: float(parse_duration_to_seconds(s)), default=None)
    p.add_argument("--xmax-life", type=lambda s: float(parse_duration_to_seconds(s)), default=None)
    p.add_argument("--mean-life", type=lambda s: float(parse_duration_to_seconds(s)), default=None)

    # Requests (bounded Pareto)
    p.add_argument("--xmin-req", type=float, default=None)
    p.add_argument("--xmax-req", type=float, default=None)
    p.add_argument("--mean-req", type=float, default=None)

    # Priority + replicas (geometric / uniform)
    p.add_argument("--priority-min", type=int, default=None)
    p.add_argument("--priority-max", type=int, default=None)
    p.add_argument("--priority-ratio", type=float, default=None)

    p.add_argument("--replicas-min", type=int, default=None)
    p.add_argument("--replicas-max", type=int, default=None)
    p.add_argument("--replicas-ratio", type=float, default=None)

    return p

# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------

class TraceGenerator:
    def __init__(self, cli_args: argparse.Namespace) -> None:
        args = TraceGenerator.resolve_args(cli_args)
        setup_logging(name=LOGGER_NAME, prefix=f"[{LOGGER_NAME}] ", level=args.log_level)
        self.init_from_args(args, create_figures_dir=not bool(getattr(args, "seed_file", None)), log_args=True)

    def init_from_args(self, args: argparse.Namespace, *, create_figures_dir: bool, log_args: bool) -> None:
        """
        Shared initializer for both __init__ (CLI/job-file resolved) and per-seed resolved runs.
        """
        self.args = args
        self.base_seed = int(getattr(args, "seed", 0) or 0)

        self.trace_time_s = float(parse_duration_to_seconds(self.args.trace_time))

        # Output dirs / files
        self.output_dir = Path(self.args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        if create_figures_dir:
            self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.initial_path = self.output_dir / "initial.json"
        self.trace_path = self.output_dir / "trace.json"
        self.info_path = self.output_dir / "info_generate.yaml"
        self.util_plot_path = self.figures_dir / "utilization.png"
        self.hist_plot_path = self.figures_dir / "histograms.png"

        # Fitted Pareto alphas
        self.alpha_req = None
        self.alpha_arrival = None
        self.alpha_life = None

        # Plot series
        self.times = []
        self.u_req_hist = []
        self.pods_hist = []
        self.initial_pods_count = 0

        if log_args:
            self.log_args()

    # -------------------------------------------------------------------------
    # Argument resolution + validation
    # -------------------------------------------------------------------------

    @staticmethod
    def resolve_args(cli_args: argparse.Namespace) -> argparse.Namespace:
        """
        Resolve effective args from CLI + job file.
        """
        args = cli_args
        job_doc = TraceGenerator.load_job_doc(args.job_file) if getattr(args, "job_file", None) else None
        if job_doc is not None:
            args = TraceGenerator.merge_job_fields(args, job_doc)
        args = TraceGenerator.apply_defaults(args)
        TraceGenerator.validate_args(args)
        TraceGenerator.round_float_args(args, MAX_DECIMALS)
        return args

    @staticmethod
    def load_job_doc(path: str | Path) -> dict:
        """
        Load and parse a job YAML file.
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

    @staticmethod
    def merge_job_fields(args: argparse.Namespace, job: dict) -> argparse.Namespace:
        """
        Merge job-file fields into args.
        """
        fields = [
            # General
            JobField("output-dir", "output_dir", parse=parse_optional_str),
            JobField("seed", "seed", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
            JobField("seed-file", "seed_file", parse=parse_optional_str),
            JobField("log-level", "log_level", parse=parse_optional_str),
            JobField("show-plots", "show_plots", parse=parse_optional_bool, accept=lambda v: v is not None),

            # Cluster / horizon
            JobField("num-nodes", "num_nodes", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
            JobField("trace-time", "trace_time", parse=parse_optional_str),
            JobField("target-util", "target_util", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),

            # Inter-arrival
            JobField("xmin-arrival", "xmin_arrival", parse=parse_optional_duration_seconds, accept=lambda v: isinstance(v, float)),
            JobField("xmax-arrival", "xmax_arrival", parse=parse_optional_duration_seconds, accept=lambda v: isinstance(v, float)),
            JobField("mean-arrival", "mean_arrival", parse=parse_optional_duration_seconds, accept=lambda v: isinstance(v, float)),

            # Lifetime
            JobField("xmin-life", "xmin_life", parse=parse_optional_duration_seconds, accept=lambda v: isinstance(v, float)),
            JobField("xmax-life", "xmax_life", parse=parse_optional_duration_seconds, accept=lambda v: isinstance(v, float)),
            JobField("mean-life", "mean_life", parse=parse_optional_duration_seconds, accept=lambda v: isinstance(v, float)),

            # Requests
            JobField("xmin-req", "xmin_req", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
            JobField("xmax-req", "xmax_req", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
            JobField("mean-req", "mean_req", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),

            # Priority
            JobField("priority-min", "priority_min", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
            JobField("priority-max", "priority_max", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
            JobField("priority-ratio", "priority_ratio", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),

            # Replicas
            JobField("replicas-min", "replicas_min", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
            JobField("replicas-max", "replicas_max", parse=parse_optional_int, accept=lambda v: isinstance(v, int)),
            JobField("replicas-ratio", "replicas_ratio", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
        ]
        
        return merge_job_fields_into_args(args, job or {}, fields)

    @staticmethod
    def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
        """
        Apply default values to args when unset.
        """
        if getattr(args, "log_level", None) is None:
            args.log_level = DEFAULT_LOG_LEVEL
        if getattr(args, "show_plots", None) is None:
            args.show_plots = bool(DEFAULT_SHOW_PLOTS)
        return args

    @staticmethod
    def validate_args(args: argparse.Namespace) -> None:
        """
        Validate required args and value ranges.
        """
        missing: list[str] = []

        if getattr(args, "output_dir", None) is None:
            missing.append("output_dir")

        # Seed selection: require exactly one mode.
        seed = getattr(args, "seed", None)
        seed_file = getattr(args, "seed_file", None)
        if seed is not None and seed_file:
            raise SystemExit("--seed and --seed-file cannot be used together")
        if seed is None and not seed_file:
            missing.append("seed or seed_file")

        for k in (
            "num_nodes",
            "trace_time",
            "target_util",
            "xmin_arrival",
            "xmax_arrival",
            "mean_arrival",
            "xmin_life",
            "xmax_life",
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

        def _pos(name: str) -> float:
            """
            Ensure arg is > 0 and return its float value.
            """
            v = float(getattr(args, name))
            if v <= 0:
                raise SystemExit(f"{name} must be > 0 (got {v})")
            return v

        _pos("num_nodes")
        _pos("xmin_arrival"); _pos("xmax_arrival"); _pos("mean_arrival")
        _pos("xmin_life"); _pos("xmax_life")
        _pos("xmin_req"); _pos("xmax_req"); _pos("mean_req")

        if float(args.xmin_life) < MIN_LIFETIME_S:
            raise SystemExit(f"xmin-life must be >= {MIN_LIFETIME_S:.1f}s (got {float(args.xmin_life):.6f})")
        if float(args.xmax_arrival) <= float(args.xmin_arrival):
            raise SystemExit("require xmax-arrival > xmin-arrival")
        if float(args.xmax_life) <= float(args.xmin_life):
            raise SystemExit("require xmax-life > xmin-life")
        if float(args.xmax_req) <= float(args.xmin_req):
            raise SystemExit("require xmax-req > xmin-req")

        target_util = float(args.target_util)
        if not (0.0 < target_util <= 1.0):
            raise SystemExit("target-util must be in (0,1]")

        if getattr(args, "job_file", None):
            p = Path(args.job_file).resolve()
            if not p.exists():
                raise SystemExit(f"--job-file not found: {p}")

        if getattr(args, "seed_file", None):
            p = Path(args.seed_file).resolve()
            if not p.exists():
                raise SystemExit(f"--seed-file not found: {p}")

    @staticmethod
    def round_float_args(args: argparse.Namespace, ndigits: int) -> None:
        """
        Round all float args to ndigits decimal places.
        """
        for k, v in vars(args).items():
            if isinstance(v, float):
                setattr(args, k, round(v, ndigits))

    # -------------------------------------------------------------------------
    # Seed expansion for multi-run
    # -------------------------------------------------------------------------
    @staticmethod
    def expand_seed_runs(args: argparse.Namespace) -> list[argparse.Namespace]:
        """
        Expand args into multiple runs based on seed or seed-file.
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

    # -------------------------------------------------------------------------
    # Logging + info file
    # -------------------------------------------------------------------------
    
    def log_args(self) -> None:
        """
        Log effective args.
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
            "mean_life",
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

    def write_info_file(self, extra: Dict[str, object]) -> None:
        """
        Write info_generate.yaml with inputs + generated params.
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

    # -------------------------------------------------------------------------
    # Bounded Pareto: sampling + alpha solve + mean 
    # -------------------------------------------------------------------------
    @staticmethod
    def sample_bounded_pareto(
        rng: np.random.Generator,
        *,
        alpha: float,
        x_min: float,
        x_max: float,
        size: int = 1,
    ) -> np.ndarray:
        """
        Bounded Pareto via inverse CDF.
        Formula: X = x_min * (1 - U * (1 - (x_min/x_max)^alpha))^(-1/alpha)
        where U ~ Uniform(0, 1)
        See: https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution
        """
        if not (alpha > 0 and x_min > 0 and x_max > x_min):
            raise ValueError("Require alpha>0, x_min>0, x_max>x_min")

        u = rng.random(size) # U ~ Uniform(0, 1)
        u = np.clip(u, 1e-12, 1.0 - 1e-12) # clamp U to avoid edge cases

        # Adjust U to account for bounded tail
        u_min_tail = (x_min / x_max) ** alpha # F(x_min) lower CDF tail
        u = u_min_tail + (1.0 - u_min_tail) * u  # uniform on [u_min_tail, 1)

        return x_min * np.exp((-np.log(u)) / alpha)

    @staticmethod
    def solve_alpha_for_bounded_mean(
        *,
        x_min: float,
        x_max: float,
        target_mean: float,
        max_iter: int = ALPHA_SOLVE_MAX_ITER,
        rel_tol: float = ALPHA_SOLVE_TOLERANCE,
    ) -> float:
        """
        Solve the bounded-Pareto shape parameter (alpha) from a target mean.

        We want alpha > 0 such that:
            bounded_pareto_mean(alpha, x_min, x_max) == target_mean

        Key properties (fixed x_min, x_max):
        - The mean is a decreasing function of alpha.
        - As alpha -> 0+, the mean approaches the maximum achievable value:
              mean_max = (x_max - x_min) / ln(x_max / x_min)
        - As alpha -> +infinity, the distribution concentrates near x_min and the mean
          approaches x_min.

        Because of monotonicity, we can solve for alpha with bisection:
        1) Validate that target_mean is achievable (x_min < target_mean < mean_max).
        2) Bracket the root by growing an upper bound hi until mean(hi) <= target_mean.
        3) Bisect until the mean matches target_mean within a relative tolerance.
        """
        if not (x_min > 0 and x_max > x_min):
            raise ValueError("Require x_min>0 and x_max>x_min")
        if not (x_min < target_mean < x_max):
            raise ValueError(f"target_mean must be in (x_min, x_max); got {target_mean}")

        mean_max = TraceGenerator.bounded_pareto_max_mean(x_min, x_max)
        if target_mean >= mean_max:
            raise ValueError(
                f"target_mean={target_mean} is not achievable for bounded Pareto on [{x_min}, {x_max}]. "
                f"Maximum achievable mean is about {mean_max:.6f}. Increase x_max or reduce target_mean."
            )

        lo = 1e-12
        hi = 1.0
        while TraceGenerator.bounded_pareto_mean(hi, x_min, x_max) > target_mean:
            hi *= 2.0
            if hi > 1e12:
                raise RuntimeError("Failed to bracket alpha; hi grew too large.")

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            mm = TraceGenerator.bounded_pareto_mean(mid, x_min, x_max)
            if abs(mm - target_mean) <= rel_tol * target_mean:
                return float(mid)
            if mm >= target_mean:
                lo = mid
            else:
                hi = mid

        return float(0.5 * (lo + hi))

    @staticmethod
    def bounded_pareto_max_mean(x_min: float, x_max: float) -> float:
        """
        Maximum achievable mean for a bounded Pareto on [x_min, x_max].

        This value comes from taking the bounded-Pareto mean formula and letting
        alpha -> 0+ (the heaviest possible tail while still being a proper
        bounded distribution). In that limit, the mean approaches:

            mean_max = (x_max - x_min) / ln(x_max / x_min)

        Intuition: for fixed bounds [x_min, x_max], the bounded-Pareto mean
        decreases as alpha increases (lighter tail puts less mass near x_max).
        So the largest mean happens at the smallest alpha.

        You can derive the limit from the Wikipedia mean expression by setting r
        = x_min/x_max and using the standard limit:

            (1 - r^k) / k -> -ln(r) as k -> 0

        See:
        https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution
        """
        if not (x_min > 0 and x_max > x_min):
            raise ValueError("Require x_min>0, x_max>x_min")
        return (x_max - x_min) / math.log(x_max / x_min)

    @staticmethod
    def bounded_pareto_mean(alpha: float, x_min: float, x_max: float) -> float:
        """
        Mean E[X] for the bounded Pareto on [x_min, x_max].

        See: https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution
                Wikipedia notation (mapped to this function):
                - alpha  -> shape parameter ("a" on Wikipedia)
                - x_min  -> minimum ("L" on Wikipedia)
                - x_max  -> maximum ("H" on Wikipedia)

        For alpha != 1, the Wikipedia page writes the mean as:
            E[X] = (x_min^alpha / (1 - (x_min/x_max)^alpha))
                            * (alpha / (alpha - 1))
                            * (1/(x_min^(alpha-1)) - 1/(x_max^(alpha-1)))

            If you expand the last factor:
                L^alpha * (1/L^(alpha-1) - 1/H^(alpha-1))
                = x_min - x_min^alpha / x_max^(alpha-1)
                = x_min * (1 - (x_min/x_max)^(alpha-1))
        
            So the whole expression becomes:
                E[X] = (alpha * x_min / (alpha - 1)) * (1 - (x_min/x_max)^(alpha - 1)) / (1 - (x_min/x_max)^alpha)

        For alpha == 1, Wikipedia gives:
            E[X] = (x_min * x_max / (x_max - x_min)) * ln(x_max / x_min)

        Correspondence to the code below:
            - Define r = x_min / x_max. Because x_min < x_max, we have 0 < r < 1.
            - The code sets log_r = log(r), so log_r is negative.
            - Python's math.expm1(x) returns exp(x) - 1. Therefore:
                -math.expm1(x) == 1 - exp(x)
            - Using that identity:
                den = -expm1(alpha * log_r)         = 1 - exp(alpha * log_r)        = 1 - r^alpha
                num = -expm1((alpha - 1) * log_r)   = 1 - exp((alpha - 1) * log_r)  = 1 - r^(alpha - 1)
            - We use expm1(...) because when r^k is close to 1 (i.e., k*log(r) is close to 0),
                computing 1 - r^k directly can lose precision due to cancellation.
        """
        if not (x_min > 0 and x_max > x_min and alpha > 0):
            raise ValueError("Require x_min>0, x_max>x_min, alpha>0")

        #################
        # alpha == 1 case
        #################
        if abs(alpha - 1.0) < 1e-10:
            return (x_max * x_min / (x_max - x_min)) * math.log(x_max / x_min)
        
        #################
        # alpha != 1 case
        #################
        # Let r = x_min/x_max in (0, 1). Wikipedia uses powers of r:
        #   1 - r^alpha     and     1 - r^(alpha-1)
        # Compute these as -expm1(k*log(r)) for better precision when r^k ~ 1.
        log_r = math.log(x_min / x_max)             # log(r) < 0
        den = -math.expm1(alpha * log_r)            # 1 - r^alpha
        num = -math.expm1((alpha - 1.0) * log_r)    # 1 - r^(alpha-1)
        return (alpha * x_min / (alpha - 1.0)) * (num / den)

    # -------------------------------------------------------------------------
    # Mean-life inference and alpha fitting
    # -------------------------------------------------------------------------
    def infer_mean_life_from_target_util(self) -> float:
        """
        Infer an initial guess for mean pod lifetime from a target steady-state
        utilization.

        This uses a simple steady-state flow-balance / Little's-law style
        relation. See: https://en.wikipedia.org/wiki/Little%27s_law In
        expectation, the long-run average requested CPU across the cluster is:

            expected_total_req = arrival_rate * E[replicas] * E[req_per_replica] * E[lifetime]

        We set expected_total_req equal to the target total requested CPU, which is:
            target_total_req = target_util * num_nodes

        Solving for E[lifetime] gives:
            mean_life = (target_util * num_nodes) / (arrival_rate * E[replicas] * E[req_per_replica])

        Where this method gets each term:
            - arrival_rate is 1/mean_arrival.
            - E[replicas] comes from the (possibly truncated) geometric / uniform replica model.
            - E[req_per_replica] is mean_req.

        This is only a first guess: calibration may adjust mean_life to better
        match the measured time-averaged utilization of a generated trace.
        """
        mean_arrival = float(self.args.mean_arrival)
        lam = 1.0 / mean_arrival

        rep_vals, rep_probs = self.build_trunc_geometric_support(
            int(self.args.replicas_min),
            int(self.args.replicas_max),
            float(self.args.replicas_ratio),
        )
        e_rep = self.expected_value(rep_vals, rep_probs)

        e_req = float(self.args.mean_req)
        U = float(self.args.target_util)
        N = float(self.args.num_nodes)

        mean_life = (U * N) / (lam * e_rep * e_req)

        xmin = float(self.args.xmin_life)
        xmax = float(self.args.xmax_life)
        if mean_life <= xmin:
            raise ValueError(f"Inferred mean_life={mean_life:.6f}s <= xmin-life={xmin}.")
        if mean_life >= xmax:
            raise ValueError(f"Inferred mean_life={mean_life:.6f}s >= xmax-life={xmax}.")

        return float(mean_life)

#TODO: HERTIL
    def fit_pareto_alphas(self) -> None:
        """
        Deterministically solve Pareto alphas from (xmin, xmax, mean).
        Cache req/arrival; recompute life because mean_life changes during calibration.
        """
        if self.alpha_req is None:
            self.alpha_req = round(
                self.solve_alpha_for_bounded_mean(
                    x_min=float(self.args.xmin_req),
                    x_max=float(self.args.xmax_req),
                    target_mean=float(self.args.mean_req),
                ),
                MAX_DECIMALS,
            )
            self.args.alpha_req = float(self.alpha_req)

        if self.alpha_arrival is None:
            self.alpha_arrival = round(
                self.solve_alpha_for_bounded_mean(
                    x_min=float(self.args.xmin_arrival),
                    x_max=float(self.args.xmax_arrival),
                    target_mean=float(self.args.mean_arrival),
                ),
                MAX_DECIMALS,
            )
            self.args.alpha_arrival = float(self.alpha_arrival)

        self.alpha_life = round(
            self.solve_alpha_for_bounded_mean(
                x_min=float(self.args.xmin_life),
                x_max=float(self.args.xmax_life),
                target_mean=float(self.args.mean_life),
            ),
            MAX_DECIMALS,
        )
        self.args.alpha_life = float(self.alpha_life)

        LOG.info("[pareto-fit] req:     mean=%.6f xmin=%.6f xmax=%.6f alpha=%.6f",
                 float(self.args.mean_req), float(self.args.xmin_req), float(self.args.xmax_req), float(self.alpha_req))
        LOG.info("[pareto-fit] arrival: mean=%.6f xmin=%.6f xmax=%.6f alpha=%.6f",
                 float(self.args.mean_arrival), float(self.args.xmin_arrival), float(self.args.xmax_arrival), float(self.alpha_arrival))
        LOG.info("[pareto-fit] life:    mean=%.6f xmin=%.6f xmax=%.6f alpha=%.6f",
                 float(self.args.mean_life), float(self.args.xmin_life), float(self.args.xmax_life), float(self.alpha_life))

    # -------------------------------------------------------------------------
    # Discrete helpers (priority/replicas)
    # -------------------------------------------------------------------------
    @staticmethod
    def build_trunc_geometric_support(min_val: int, max_val: int, ratio: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build discrete support + probs for truncated geometric on [min_val, max_val] with given ratio.
        1) If ratio == 1.0, uniform discrete on [min_val, max_val].
        2) If ratio != 1.0, pmf p(k) ∝ ratio^{k - min_val} for k in [min_val, max_val].
        3) If only one value in support, return uniform with single value.
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
    def expected_value(vals: np.ndarray, probs: Optional[np.ndarray]) -> float:
        """
        E[X] for discrete RV with support vals and optional probs.
        """
        return float(np.mean(vals)) if probs is None else float(np.sum(vals.astype(float) * probs.astype(float)))

    # -------------------------------------------------------------------------
    # Util metric
    # -------------------------------------------------------------------------
    def time_avg_req_util(self, pods: List[TraceRecord]) -> float:
        """
        Time-averaged requested CPU utilization over [0, T]:
          U = (1 / (N*T)) * ∫[0 to T] Σ_{p in pods active at t} replicas_p * cpu_p dt
        0 <= U <= 1
        """
        T = float(self.trace_time_s)
        if T <= 0.0:
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

    # -------------------------------------------------------------------------
    # Initial snapshot: fill to target util (stationary residual-life sampling)
    # -------------------------------------------------------------------------
    def build_initial_snapshot(
        self,
        rng: np.random.Generator,
        *,
        prio_vals: np.ndarray,
        prio_probs: Optional[np.ndarray],
        rep_vals: np.ndarray,
        rep_probs: Optional[np.ndarray],
        next_id: int,
    ) -> Tuple[List[TraceRecord], int]:
        """
        Sample pods until total requested ~= target_util * num_nodes (within tol).
        """
        assert self.alpha_req is not None and self.alpha_life is not None

        target_total = float(self.args.target_util) * float(self.args.num_nodes)
        tol = float(INITIAL_PODS_UTIL_TOLERANCE)

        cur_total = 0.0
        pods: List[TraceRecord] = []

        while cur_total < target_total * (1.0 - tol):
            if len(pods) >= int(INITIAL_MAX_PODS):
                LOG.warning("[snapshot] hit initial-max-pods=%d; stopping at util=%.3f",
                            int(INITIAL_MAX_PODS), cur_total / float(self.args.num_nodes))
                break

            req = float(self.sample_bounded_pareto(
                rng,
                alpha=float(self.alpha_req),
                x_min=float(self.args.xmin_req),
                x_max=float(self.args.xmax_req),
                size=1,
            )[0])

            # sample steady-state residual life (equilibrium remaining lifetime)
            life = float(self.sample_steady_state_residual_life(
                rng,
                alpha=float(self.alpha_life),
                x_min=float(self.args.xmin_life),
                x_max=float(self.args.xmax_life),
                size=1,
            )[0])

            if life <= 1e-9:
                continue

            reps = int(rng.choice(rep_vals, p=rep_probs))
            prio = int(rng.choice(prio_vals, p=prio_probs))

            cur_total += reps * req
            next_id += 1
            pods.append(TraceRecord(
                id=next_id,
                start_time=0.0,
                end_time=round(life, MAX_DECIMALS),
                cpu=round(req, MAX_DECIMALS),
                mem=round(req, MAX_DECIMALS),
                priority=prio,
                replicas=reps,
            ))

        # small overshoot cleanup
        while pods and cur_total > target_total * (1.0 + tol):
            last = pods.pop()
            cur_total -= float(last.replicas) * float(last.cpu)
            next_id -= 1

        LOG.info("[snapshot] pods=%d util=%.3f (target=%.3f tol=±%.3f)",
                 len(pods), cur_total / float(self.args.num_nodes),
                 float(self.args.target_util), tol)

        return pods, next_id

    @staticmethod
    def sample_steady_state_residual_life(
        rng: np.random.Generator,
        *,
        alpha: float,
        x_min: float,
        x_max: float,
        size: int = 1,
    ) -> np.ndarray:
        """
        Sample an equilibrium (steady-state) remaining lifetime.

        If pod lifetimes are i.i.d. draws X supported on [x_min, x_max], then a
        snapshot of a running system does not see fresh lifetimes X. It sees the
        residual life R of a renewal process. A standard way to generate R is:

        1) Draw a length-biased lifetime X* with density
            f_{X*}(x) ∝ x f_X(x).
        2) Given X* = x, draw R ~ Uniform(0, x).

        For the bounded Pareto used here, f_X(x) ∝ x^(-(\alpha+1)) on [x_min,
        x_max], so the length-biased density simplifies to f_{X*}(x) ∝
        x^(-\alpha). This function samples X* via an inverse-CDF construction
        (with the \alpha=1 logarithmic special case) and then samples R
        uniformly in [0, X*]. Using R instead of fresh X avoids a startup
        transient where many pods would otherwise end quickly right after t=0.
        With heavy tails (Pareto-like), the length bias intentionally
        over-represents long lives, which matches what a real "live cluster"
        snapshot tends to contain.
        """
        if not (alpha > 0 and x_min > 0 and x_max > x_min):
            raise ValueError("Require alpha>0, x_min>0, x_max>x_min")

        u = rng.random(size)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)

        # Sample X* with pdf ∝ x^{-alpha} on [x_min, x_max]
        # CDF:
        #  if alpha != 1: F(x) = (x^(1-alpha) - x_min^(1-alpha)) / (x_max^(1-alpha) - x_min^(1-alpha))
        #  if alpha == 1: F(x) = ln(x/x_min) / ln(x_max/x_min)
        if abs(alpha - 1.0) < 1e-12:
            x_star = x_min * np.exp(u * np.log(x_max / x_min))
        else:
            p = 1.0 - alpha
            x_min_p = x_min ** p
            x_max_p = x_max ** p
            x_star = (x_min_p + u * (x_max_p - x_min_p)) ** (1.0 / p)

        v = rng.random(size)
        v = np.clip(v, 1e-12, 1.0 - 1e-12)
        # Residual is Uniform(0, X*). Using (1-v)*X* is equivalent.
        return x_star * (1.0 - v)

    # -------------------------------------------------------------------------
    # Trace events
    # -------------------------------------------------------------------------
    def generate_trace_events(
        self,
        rng: np.random.Generator,
        *,
        prio_vals: np.ndarray,
        prio_probs: Optional[np.ndarray],
        rep_vals: np.ndarray,
        rep_probs: Optional[np.ndarray],
        next_id: int,
        initial_pods: List[TraceRecord],
    ) -> Tuple[List[TraceRecord], int, List[float], List[float], List[int]]:
        """
        Generate trace pod events after initial snapshot.
        """
        assert self.alpha_req is not None and self.alpha_arrival is not None and self.alpha_life is not None

        state = ClusterState()
        end_heap: List[EndHeapEntry] = []

        for p in initial_pods:
            req = float(p.cpu)
            reps = int(p.replicas)
            state.live_req += reps * req
            state.live_pods += reps
            heapq.heappush(end_heap, EndHeapEntry(end_time=float(p.end_time), req=req, replicas=reps))

        pods: List[TraceRecord] = []

        times: List[float] = [0.0]
        u_hist: List[float] = [state.live_req / float(self.args.num_nodes)]
        pods_hist: List[int] = [state.live_pods]

        t = 0.0
        while t < self.trace_time_s:
            dt = float(self.sample_bounded_pareto(
                rng,
                alpha=float(self.alpha_arrival),
                x_min=float(self.args.xmin_arrival),
                x_max=float(self.args.xmax_arrival),
                size=1,
            )[0])

            start = t + dt
            if start >= self.trace_time_s:
                break

            while end_heap and end_heap[0].end_time <= start:
                entry = heapq.heappop(end_heap)
                state.live_req = max(0.0, state.live_req - entry.req * entry.replicas)
                state.live_pods = max(0, state.live_pods - entry.replicas)

            life = float(self.sample_bounded_pareto(
                rng,
                alpha=float(self.alpha_life),
                x_min=float(self.args.xmin_life),
                x_max=float(self.args.xmax_life),
                size=1,
            )[0])
            end = start + life

            req = float(self.sample_bounded_pareto(
                rng,
                alpha=float(self.alpha_req),
                x_min=float(self.args.xmin_req),
                x_max=float(self.args.xmax_req),
                size=1,
            )[0])

            reps = int(rng.choice(rep_vals, p=rep_probs))
            prio = int(rng.choice(prio_vals, p=prio_probs))

            state.live_req += reps * req
            state.live_pods += reps
            heapq.heappush(end_heap, EndHeapEntry(end_time=end, req=req, replicas=reps))

            next_id += 1
            pods.append(TraceRecord(
                id=next_id,
                start_time=round(start, MAX_DECIMALS),
                end_time=round(end, MAX_DECIMALS),
                cpu=round(req, MAX_DECIMALS),
                mem=round(req, MAX_DECIMALS),
                priority=prio,
                replicas=reps,
            ))

            times.append(start)
            u_hist.append(state.live_req / float(self.args.num_nodes))
            pods_hist.append(state.live_pods)

            t = start

        return pods, next_id, times, u_hist, pods_hist

    # -------------------------------------------------------------------------
    # One generation pass
    # -------------------------------------------------------------------------
    def generate_once(self, iter_seed: int) -> Tuple[List[TraceRecord], List[TraceRecord], float, Dict[str, object]]:
        """
        Generate one initial + trace pod set with given iteration seed.
        """
        rng_initial = np.random.default_rng(derive_seed(iter_seed, "initial-snapshot"))
        rng_trace = np.random.default_rng(derive_seed(iter_seed, "trace-events"))

        self.fit_pareto_alphas()

        prio_vals, prio_probs = self.build_trunc_geometric_support(
            int(self.args.priority_min),
            int(self.args.priority_max),
            float(self.args.priority_ratio),
        )
        rep_vals, rep_probs = self.build_trunc_geometric_support(
            int(self.args.replicas_min),
            int(self.args.replicas_max),
            float(self.args.replicas_ratio),
        )

        next_id = 0
        initial_pods, next_id = self.build_initial_snapshot(
            rng_initial,
            prio_vals=prio_vals, prio_probs=prio_probs,
            rep_vals=rep_vals, rep_probs=rep_probs,
            next_id=next_id,
        )

        trace_pods, next_id, times, u_hist, pods_hist = self.generate_trace_events(
            rng_trace,
            prio_vals=prio_vals, prio_probs=prio_probs,
            rep_vals=rep_vals, rep_probs=rep_probs,
            next_id=next_id,
            initial_pods=initial_pods,
        )

        self.times = times
        self.u_req_hist = u_hist
        self.pods_hist = pods_hist
        self.initial_pods_count = int(sum(int(p.replicas) for p in initial_pods))

        all_pods = initial_pods + trace_pods
        util = float(self.time_avg_req_util(all_pods))

        max_prio = max((int(p.priority) for p in all_pods), default=0)

        extra_info: Dict[str, object] = {
            "num_nodes": int(self.args.num_nodes),
            "trace_time_s": round(float(self.trace_time_s), 6),
            "seed": int(self.args.seed),
            "rng": {
                "iter_seed": int(iter_seed),
                "derived": {
                    "initial_snapshot": int(derive_seed(iter_seed, "initial-snapshot")),
                    "trace_events": int(derive_seed(iter_seed, "trace-events")),
                },
            },
            "measured_util_time_avg": float(util),
            "target_util_time_avg": float(self.args.target_util),
            "calibration": {
                "util_tol": float(MEAN_LIFE_CALIBRATION_UTIL_TOLERANCE),
                "calib_max_iter": int(MEAN_LIFE_CALIBRATION_MAX_ITER),
                "initial_fill_tol": float(INITIAL_PODS_UTIL_TOLERANCE),
                "initial_max_pods": int(INITIAL_MAX_PODS),
            },
            "derived_mean_life_s": round(float(self.args.mean_life), 6),
            "pareto": {
                "req": {"alpha": float(self.alpha_req), "xmin": float(self.args.xmin_req), "xmax": float(self.args.xmax_req), "mean": float(self.args.mean_req)},
                "arrival": {"alpha": float(self.alpha_arrival), "xmin": float(self.args.xmin_arrival), "xmax": float(self.args.xmax_arrival), "mean": float(self.args.mean_arrival)},
                "life": {"alpha": float(self.alpha_life), "xmin": float(self.args.xmin_life), "xmax": float(self.args.xmax_life), "mean": float(self.args.mean_life)},
            },
            "priority": {"min": int(self.args.priority_min), "max": int(self.args.priority_max), "ratio": float(self.args.priority_ratio)},
            "replicas": {"min": int(self.args.replicas_min), "max": int(self.args.replicas_max), "ratio": float(self.args.replicas_ratio)},
            "counts": {
                "initial_records": int(len(initial_pods)),
                "trace_records": int(len(trace_pods)),
                "initial_replicas_total": int(sum(int(p.replicas) for p in initial_pods)),
                "trace_replicas_total": int(sum(int(p.replicas) for p in trace_pods)),
            },
            "max_priority_seen": int(max_prio),
            "files": {
                "initial_json": str(self.initial_path),
                "trace_json": str(self.trace_path),
                "utilization_plot": str(self.util_plot_path),
                "histograms_plot": str(self.hist_plot_path),
            },
        }

        return initial_pods, trace_pods, util, extra_info

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------
    def calibrate_mean_life(self) -> Tuple[List[TraceRecord], List[TraceRecord], Dict[str, object]]:
        """
        Calibrate mean-life to hit target utilization within tolerance.
        """
        target = float(self.args.target_util)
        tol = float(MEAN_LIFE_CALIBRATION_UTIL_TOLERANCE)

        # fixed seed across iterations (CRN => stable updates)
        iter_seed = int(derive_seed(self.base_seed, "calibrate-mean-life-crn"))

        best_err = float("inf")
        best_initial: List[TraceRecord] = []
        best_trace: List[TraceRecord] = []
        best_extra: Dict[str, object] = {}

        for it in range(1, int(MEAN_LIFE_CALIBRATION_MAX_ITER) + 1):
            initial_pods, trace_pods, measured, extra = self.generate_once(iter_seed)

            err = abs(measured - target) / max(1e-12, target)
            if err < best_err:
                best_err = err
                best_initial, best_trace, best_extra = initial_pods, trace_pods, extra

            LOG.info(
                "[calibrate-mean-life] iter=%d measured=%.4f target=%.4f rel_err=%.2f%% mean_life=%.3fs",
                it, measured, target, 100.0 * err, float(self.args.mean_life),
            )

            if err <= tol:
                return best_initial, best_trace, best_extra

            if measured <= 1e-12:
                raise RuntimeError("Measured utilization is ~0; cannot calibrate mean-life.")

            # proportional update
            new_mean = float(self.args.mean_life) * (target / measured)

            xmin = float(self.args.xmin_life)
            xmax = float(self.args.xmax_life)
            new_mean = max(new_mean, xmin * 1.001)
            new_mean = min(new_mean, xmax * 0.999)

            self.args.mean_life = float(new_mean)

        LOG.warning("[calibrate-mean-life] did not reach util-tol; best rel_err=%.2f%%", 100.0 * best_err)
        return best_initial, best_trace, best_extra

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    @staticmethod
    def write_json(path: Path, pods: List[TraceRecord]) -> None:
        """
        Write pods to a JSON file at the given path.
        """
        obj = {"pods": [asdict(p) for p in pods]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        LOG.info("wrote %s (%d records)", path, len(pods))

    def write_outputs(self, initial_pods: List[TraceRecord], trace_pods: List[TraceRecord], extra_info: Dict[str, object]) -> None:
        """
        Write output files: initial JSON, trace JSON, info YAML.
        """
        all_pods = initial_pods + trace_pods
        util_time = self.time_avg_req_util(all_pods)

        self.write_json(self.initial_path, initial_pods)
        self.write_json(self.trace_path, trace_pods)

        LOG.info("[utilization] util-time-avg over whole trace: %.4f (target=%.4f)",
                 float(util_time), float(self.args.target_util))

        self.write_info_file(extra=extra_info)

    # -------------------------------------------------------------------------
    # Runner
    # -------------------------------------------------------------------------
    
    def run_seed(self) -> None:
        """
        Run a single seed instance.
        """
        if self.args.mean_life is None:
            self.args.mean_life = self.infer_mean_life_from_target_util()
            LOG.info(
                "[inferred-mean-life] mean_life=%.3fs from target-util=%.3f",
                float(self.args.mean_life),
                float(self.args.target_util),
            )

            initial_pods, trace_pods, extra_info = self.calibrate_mean_life()
        else:
            iter_seed = int(derive_seed(self.base_seed, "provided-mean-life"))
            initial_pods, trace_pods, measured, extra_info = self.generate_once(iter_seed)
            extra_info.setdefault("calibration", {})
            extra_info["calibration"].update({"mode": "skipped", "reason": "mean_life_provided", "iterations": 1})
            LOG.info(
                "[provided-mean-life] skipping calibration; measured=%.4f target=%.4f mean_life=%.3fs",
                float(measured),
                float(self.args.target_util),
                float(self.args.mean_life),
            )

        all_pods = initial_pods + trace_pods

        self.write_outputs(initial_pods, trace_pods, extra_info)

        plot_utilization_and_num_pods(
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
            xmax_arrival=float(self.args.xmax_arrival),
            alpha_life=float(self.args.alpha_life),
            xmin_life=float(self.args.xmin_life),
            xmax_life=float(self.args.xmax_life),
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

    def run(self) -> None:
        """
        Run the trace generator, possibly expanding multiple seed runs.
        """
        runs = TraceGenerator.expand_seed_runs(self.args)
        total = len(runs)

        for i, a in enumerate(runs, start=1):
            header, footer = make_header_footer(
                f"SEED RUN {int(a.seed)} ({i}/{total})" if total > 1 else f"SEED RUN {int(a.seed)}"
            )
            LOG.info("\n%s\nseed=%d output_dir=%s\n%s", header, int(a.seed), str(a.output_dir), footer)

            # Per-seed run instance: resolved args, no re-logging of full ARGS block.
            gen = TraceGenerator.__new__(TraceGenerator)
            setup_logging(name=LOGGER_NAME, prefix=f"[{LOGGER_NAME}] ", level=a.log_level)
            gen.init_from_args(a, create_figures_dir=True, log_args=False)
            gen.run_seed()

        LOG.info("done.")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    if os.getenv("TRACE_GENERATOR_NOOP") == "1":
        return
    args = build_arg_parser().parse_args()
    trace_generator = TraceGenerator(args)
    trace_generator.run()


if __name__ == "__main__":
    main()
