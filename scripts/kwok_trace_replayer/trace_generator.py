#!/usr/bin/env python3
# trace_generator.py
"""
python -m scripts.kwok_trace_replayer.trace_generator --job-dir data/jobs/kwok_trace_generator/
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

MEAN_LIFETIME_CALIBRATION_UTIL_TOLERANCE = 0.01
MEAN_LIFETIME_CALIBRATION_MAX_ITERATIONS = 20

# Maximum initial pods to prevent runaway.
MAX_INITIAL_PODS = 200_000

# Monte Carlo sample count for estimating E[max(cpu, mem)] per replica in the
# Little's-law-style mean-life inference.
EFFECTIVE_MAX_REQ_MC_SAMPLES = 50_000

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
    live_cpu_req: float = 0.0
    live_mem_req: float = 0.0
    live_pods: int = 0

@dataclass(order=True)
class EndHeapEntry:
    end_time: float
    cpu_req: float = field(compare=False)
    mem_req: float = field(compare=False)
    replicas: int = field(compare=False)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate initial + trace workload JSON files.")

    # General
    p.add_argument("--job-file", dest="job_file", default=None,
                   help="Path to a YAML job file describing arguments. CLI overrides job file.")
    p.add_argument("--job-dir", dest="job_dir", default=None,
        help="Directory containing YAML job files; when set, all *.yaml/*.yml files are generated.",
    )
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

    # CPU requests (bounded Pareto)
    p.add_argument("--xmin-cpu", type=float, default=None)
    p.add_argument("--xmax-cpu", type=float, default=None)
    p.add_argument("--mean-cpu", type=float, default=None)

    # Memory requests (bounded Pareto)
    p.add_argument("--xmin-mem", type=float, default=None)
    p.add_argument("--xmax-mem", type=float, default=None)
    p.add_argument("--mean-mem", type=float, default=None)

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
    def __init__(
        self,
        cli_args: argparse.Namespace,
        *,
        resolved: bool = False,
        create_figures_dir: Optional[bool] = None,
        log_args: bool = True,
        setup_logger: bool = True,
    ) -> None:
        """
        Create a TraceGenerator.

        By default, this resolves and validates args (including optional job-file
        merging) and configures logging.

        For per-seed runs where args are already resolved, pass resolved=True and
        provide create_figures_dir/log_args as desired.
        """
        args = cli_args if resolved else TraceGenerator.resolve_args(cli_args)
        if setup_logger:
            setup_logging(name=LOGGER_NAME, prefix=f"[{LOGGER_NAME}] ", level=args.log_level)

        if create_figures_dir is None:
            create_figures_dir = not bool(getattr(args, "seed_file", None))

        self.init_from_args(args, create_figures_dir=bool(create_figures_dir), log_args=bool(log_args))

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
        self.alpha_arrival = None
        self.alpha_life = None
        self.alpha_cpu = None
        self.alpha_mem = None

        # Plot series
        self.times = []
        self.u_eff_hist = []
        self.u_cpu_hist = []
        self.u_mem_hist = []
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
    def expand_job_dir_runs(cli_args: argparse.Namespace) -> list[argparse.Namespace]:
        """Expand a --job-dir into one resolved args object per job file.
        Each expanded run will:
        - set job_file to the discovered YAML file
        - resolve/merge/validate args against that job file
        - place outputs under <output-dir>/<job-stem>/ to avoid collisions
        """
        if getattr(cli_args, "job_file", None) and getattr(cli_args, "job_dir", None):
            raise SystemExit("--job-file and --job-dir cannot be used together")

        job_dir = getattr(cli_args, "job_dir", None)
        if not job_dir:
            # Not in job-dir mode. Resolve single run normally.
            return [TraceGenerator.resolve_args(cli_args)]

        p = Path(job_dir).resolve()
        if not p.exists() or not p.is_dir():
            raise SystemExit(f"--job-dir must be an existing directory: {p}")

        job_files = sorted(list(p.glob("*.yaml")) + list(p.glob("*.yml")))
        if not job_files:
            raise SystemExit(f"--job-dir contains no *.yaml/*.yml job files: {p}")

        runs: list[argparse.Namespace] = []
        for jf in job_files:
            a = copy.copy(cli_args)
            a.job_file = str(jf)
            a.job_dir = None

            resolved = TraceGenerator.resolve_args(a)
            base_out = Path(resolved.output_dir)
            resolved.output_dir = str(base_out)
            runs.append(resolved)

        return runs

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

            # CPU requests
            JobField("xmin-cpu", "xmin_cpu", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
            JobField("xmax-cpu", "xmax_cpu", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
            JobField("mean-cpu", "mean_cpu", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
        
            # Memory requests
            JobField("xmin-mem", "xmin_mem", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
            JobField("xmax-mem", "xmax_mem", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),
            JobField("mean-mem", "mean_mem", parse=parse_optional_float, accept=lambda v: isinstance(v, float)),

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

        We do basic sanity checks and validate that configured bounded-Pareto
        means are achievable given their bounds:
        - mean must satisfy xmin < mean < xmax
        - mean must be < max achievable mean for bounded Pareto:
                mean_max = (xmax - xmin) / ln(xmax/xmin)
        """
        missing: list[str] = []

        if getattr(args, "job_file", None) and getattr(args, "job_dir", None):
            raise SystemExit("--job-file and --job-dir cannot be used together")
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
            "xmin_cpu",
            "xmax_cpu",
            "mean_cpu",
            "xmin_mem",
            "xmax_mem",
            "mean_mem",
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

        def positive_val(name: str) -> float:
            """Ensure arg is > 0 and return its float value."""
            v = float(getattr(args, name))
            if v <= 0:
                raise SystemExit(f"{name} must be > 0 (got {v})")
            return v

        positive_val("num_nodes")

        positive_val("xmin_arrival")
        positive_val("xmax_arrival")
        positive_val("mean_arrival")
        
        positive_val("xmin_life")
        positive_val("xmax_life")
        
        positive_val("xmin_cpu")
        positive_val("xmax_cpu")
        positive_val("mean_cpu")
        
        positive_val("xmin_mem")
        positive_val("xmax_mem")
        positive_val("mean_mem")

        positive_val("priority_ratio")
        positive_val("replicas_ratio")

        # Bounds ordering
        if float(args.xmin_life) < MIN_LIFETIME_S:
            raise SystemExit(f"xmin-life must be >= {MIN_LIFETIME_S:.1f}s (got {float(args.xmin_life):.6f})")
        if float(args.xmax_arrival) <= float(args.xmin_arrival):
            raise SystemExit("require xmax-arrival > xmin-arrival")
        if float(args.xmax_life) <= float(args.xmin_life):
            raise SystemExit("require xmax-life > xmin-life")
        if float(args.xmax_cpu) <= float(args.xmin_cpu):
            raise SystemExit("require xmax-cpu > xmin-cpu")
        if float(args.xmax_mem) <= float(args.xmin_mem):
            raise SystemExit("require xmax-mem > xmin-mem")

        # Discrete supports must be ordered
        if int(args.priority_max) < int(args.priority_min):
            raise SystemExit("require priority-max >= priority-min")
        if int(args.replicas_max) < int(args.replicas_min):
            raise SystemExit("require replicas-max >= replicas-min")

        # Target util
        target_util = float(args.target_util)
        if not (0.0 < target_util <= 1.0):
            raise SystemExit("target-util must be in (0,1]")

        def validate_bounded_pareto_mean(mean_name: str, xmin_name: str, xmax_name: str) -> None:
            mu = float(getattr(args, mean_name))
            xmin = float(getattr(args, xmin_name))
            xmax = float(getattr(args, xmax_name))

            if not (xmin < mu < xmax):
                raise SystemExit(f"require {mean_name} in ({xmin_name}, {xmax_name}); got {mu} with bounds [{xmin}, {xmax}]")

            mu_max = TraceGenerator.bounded_pareto_max_mean(xmin, xmax)
            if mu >= mu_max:
                raise SystemExit(
                    f"{mean_name}={mu} is not achievable for bounded Pareto on [{xmin}, {xmax}]. "
                    f"Maximum achievable mean is about {mu_max:.6f}. Increase {xmax_name} or reduce {mean_name}."
                )

        # Validate bounded Pareto means
        validate_bounded_pareto_mean("mean_arrival", "xmin_arrival", "xmax_arrival")
        validate_bounded_pareto_mean("mean_cpu", "xmin_cpu", "xmax_cpu")
        validate_bounded_pareto_mean("mean_mem", "xmin_mem", "xmax_mem")

        # mean-life is optional (can be inferred) - only validate if provided.
        if getattr(args, "mean_life", None) is not None:
            positive_val("mean_life")
            validate_bounded_pareto_mean("mean_life", "xmin_life", "xmax_life")

        # File existence checks
        if getattr(args, "job_file", None):
            p = Path(args.job_file).resolve()
            if not p.exists():
                raise SystemExit(f"--job-file not found: {p}")
        if getattr(args, "job_dir", None):
            p = Path(args.job_dir).resolve()
            if not p.exists() or not p.is_dir():
                raise SystemExit(f"--job-dir must be an existing directory: {p}")
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
            "job_dir",
            "output_dir",
            "seed",
            "seed_file",
            "log_level",
            "show_plots",
            "num_nodes",
            "trace_time",
            "target_util",
            "xmin_arrival",
            "xmax_arrival",
            "mean_arrival",
            "xmin_life",
            "xmax_life",
            "mean_life",
            "xmin_cpu",
            "xmax_cpu",
            "mean_cpu",
            "xmin_mem",
            "xmax_mem",
            "mean_mem",
            "priority_min",
            "priority_max",
            "priority_ratio",
            "replicas_min",
            "replicas_max",
            "replicas_ratio",
        ]
        log_args_block(LOG, self.args, title="ARGS", include=include)

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
        Sample from the upper-truncated Pareto (Type I) distribution on [x_min, x_max]
        via inverse-CDF sampling.

        Paper alignment:
          - Truncated Pareto CDF: Zaninetti & Ferraro (2008), Eq. (4), p. 2.
          - Random variate generation: Zaninetti & Ferraro (2008), Eq. (13), p. 3.

        Using Eq. (13) with our notation (a=x_min, b=x_max, c=alpha):
            X = x_min * ( 1 - R * (1 - (x_min/x_max)^alpha) )^(-1/alpha),
        where R ~ Uniform(0,1).

        Args:
            rng: NumPy Generator
            alpha: shape parameter (>0)
            x_min: lower bound (>0)
            x_max: upper bound (>x_min)
            size: number of samples

        Returns:
            np.ndarray of shape (size,)
        """
        if not (alpha > 0 and x_min > 0 and x_max > x_min):
            raise ValueError("Require alpha>0, x_min>0, x_max>x_min")
        if not (isinstance(size, int) and size >= 1):
            raise ValueError("Require size to be a positive int")

        r = rng.random(size)
        r = np.clip(r, 1e-12, 1.0 - 1e-12)

        # Eq. (13): base = 1 - R*(1 - (a/b)^c)  in ( (a/b)^c, 1 ]
        tail_factor = (x_min / x_max) ** alpha
        base = 1.0 - r * (1.0 - tail_factor)
        base = np.clip(base, 1e-300, 1.0)

        return (x_min * np.power(base, -1.0 / alpha)).astype(float)

    @staticmethod
    def bounded_pareto_mean(alpha: float, x_min: float, x_max: float) -> float:
        """
        Mean E[X] of the upper-truncated Pareto (Type I) on [x_min, x_max].

        Paper alignment:
          - Zaninetti & Ferraro (2008), Eq. (5), p. 2 (mean for c != 1 and c = 1).

        With a=x_min, b=x_max, c=alpha:

        If alpha != 1:
            E[X] = (alpha * x_min / (alpha - 1))
                   * (1 - (x_min/x_max)^(alpha-1)) / (1 - (x_min/x_max)^alpha)

        If alpha == 1:
            E[X] = (alpha * x_min^alpha / (1 - (x_min/x_max)^alpha)) * ln(x_max/x_min)
                 = (x_min / (1 - x_min/x_max)) * ln(x_max/x_min)
        """
        if not (x_min > 0 and x_max > x_min and alpha > 0):
            raise ValueError("Require x_min>0, x_max>x_min, alpha>0")

        r = x_min / x_max  # in (0,1)
        log_r = math.log(r)  # < 0

        # alpha == 1 case (Eq. 5, c=1)
        if abs(alpha - 1.0) < 1e-10:
            # Use stable forms for (1 - r) and log(x_max/x_min)
            one_minus_r = -math.expm1(log_r)          # 1 - r
            return (x_min / one_minus_r) * math.log(x_max / x_min)

        # alpha != 1 case (Eq. 5, c != 1), computed stably with expm1
        den = -math.expm1(alpha * log_r)              # 1 - r^alpha
        num = -math.expm1((alpha - 1.0) * log_r)      # 1 - r^(alpha-1)
        return (alpha * x_min / (alpha - 1.0)) * (num / den)

    @staticmethod
    def bounded_pareto_max_mean(x_min: float, x_max: float) -> float:
        """
        Maximum achievable mean for a truncated Pareto on [x_min, x_max].

        This value is obtained by taking the limit alpha -> 0+ in the mean
        expression of Zaninetti & Ferraro (2008), Eq. (5), p. 2, yielding:
            mean_max = (x_max - x_min) / ln(x_max / x_min)

        Used to validate that a requested target mean is achievable for the given
        bounds.
        """
        if not (x_min > 0 and x_max > x_min):
            raise ValueError("Require x_min>0, x_max>x_min")
        return (x_max - x_min) / math.log(x_max / x_min)

    @staticmethod
    def solve_alpha_for_bounded_mean(
        *,
        x_min: float,
        x_max: float,
        target_mean: float,
        max_iterations: int = ALPHA_SOLVE_MAX_ITER,
        relative_tolerance: float = ALPHA_SOLVE_TOLERANCE,
    ) -> float:
        """
        Solve the bounded Pareto shape parameter alpha from a target mean.

        We solve for alpha > 0 such that:
            bounded_pareto_mean(alpha, x_min, x_max) == target_mean

        Paper alignment:
          - Mean formula is from Zaninetti & Ferraro (2008), Eq. (5), p. 2.
          - Sampling (used elsewhere) is from Eq. (13), p. 3.

        We rely on the fact that for fixed bounds [x_min, x_max], the mean is a
        monotone decreasing function of alpha, so bisection applies.
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

        for _ in range(max_iterations):
            mid = 0.5 * (lo + hi)
            mm = TraceGenerator.bounded_pareto_mean(mid, x_min, x_max)
            if abs(mm - target_mean) <= relative_tolerance * target_mean:
                return float(mid)
            if mm >= target_mean:
                lo = mid
            else:
                hi = mid

        return float(0.5 * (lo + hi))

    @staticmethod
    def sample_steady_state_residual_lifetime(
        rng: np.random.Generator,
        *,
        alpha: float,
        x_min: float,
        x_max: float,
        size: int = 1,
    ) -> np.ndarray:
        """Sample the steady-state residual lifetime for a bounded Pareto.

        In a renewal process observed at a random time, the residual lifetime R
        (time remaining until completion) is not distributed as the original
        lifetime X. Instead, X is length-biased and then R is uniform on [0, X].

        For bounded Pareto with lifetime pdf f(x) ∝ x^{-(alpha+1)} on [x_min,
        x_max], the length-biased lifetime has pdf g(x) ∝ x f(x) ∝ x^{-alpha} on
        the same bounds.

        Returns:
            An array of shape (size,) with values in [0, x_max].
        """
        if not (alpha > 0 and x_min > 0 and x_max > x_min):
            raise ValueError("Require alpha>0, x_min>0, x_max>x_min")
        if not (isinstance(size, int) and size >= 1):
            raise ValueError("Require size to be a positive int")

        u = rng.random(size)

        # Sample length-biased lifetime X with pdf ∝ x^{-alpha} on [x_min, x_max].
        # CDF inversion:
        #   alpha != 1: F(x) = (x^(1-alpha) - x_min^(1-alpha)) / (x_max^(1-alpha) - x_min^(1-alpha))
        #   alpha == 1: F(x) = (ln x - ln x_min) / (ln x_max - ln x_min)
        if abs(alpha - 1.0) < 1e-12:
            log_min = math.log(x_min)
            log_max = math.log(x_max)
            x = np.exp(log_min + (log_max - log_min) * u)
        else:
            p = 1.0 - alpha
            a = x_min ** p
            b = x_max ** p
            x = np.power(a + (b - a) * u, 1.0 / p)

        # Given X, residual lifetime R is uniform on [0, X].
        r = x * (1.0 - rng.random(size))
        return r.astype(float)

    # -------------------------------------------------------------------------
    # Mean-life inference and alpha fitting
    # -------------------------------------------------------------------------

    def infer_mean_lifetime_from_target_util(self) -> float:
        """
        Infer an initial guess for mean pod lifetime from the target steady-state utilization.

        We approximate steady-state flow-balance (Little's-law style) under the generator's
        effective-load objective:
            eff(t) = max(total_cpu_req(t), total_mem_req(t))

        In expectation, the long-run average effective requested load across the cluster is:
            E[eff_total] ≈ arrival_rate * E[replicas] * E[max(cpu_req, mem_req)] * E[lifetime]

        We set E[eff_total] equal to the target effective requested capacity:
            target_eff_total = target_util * num_nodes

        Solving for E[lifetime]:
            mean_life = (target_util * num_nodes) /
                        (arrival_rate * E[replicas] * E[max(cpu_req, mem_req)])

        Notes:
        - arrival_rate = 1/mean_arrival
        - E[replicas] comes from the truncated geometric/uniform replica model
        - E[max(cpu_req, mem_req)] is estimated via deterministic Monte Carlo

        This is an initial guess; calibration adjusts mean_life to match the measured
        time-mean effective utilization of a generated trace.
        """
        mean_arrival = float(self.args.mean_arrival)
        lam = 1.0 / mean_arrival

        rep_vals, rep_probs = self.build_trunc_geometric_support(
            int(self.args.replicas_min),
            int(self.args.replicas_max),
            float(self.args.replicas_ratio),
        )
        e_rep = self.expected_value(rep_vals, rep_probs)

        e_req_eff = self.mc_estimate_eff_mean_max_req()

        U = float(self.args.target_util)
        N = float(self.args.num_nodes)

        mean_life = (U * N) / (lam * e_rep * e_req_eff)

        xmin = float(self.args.xmin_life)
        xmax = float(self.args.xmax_life)
        if mean_life <= xmin:
            raise ValueError(f"Inferred mean_life={mean_life:.6f}s <= xmin-life={xmin}.")
        if mean_life >= xmax:
            raise ValueError(f"Inferred mean_life={mean_life:.6f}s >= xmax-life={xmax}.")

        LOG.info(
            "[infer-mean-life] using E[max(cpu,mem)]=%.6f (per replica), E[replicas]=%.3f, lambda=%.6f",
            float(e_req_eff),
            float(e_rep),
            float(lam),
        )

        return float(mean_life)

    def mc_estimate_eff_mean_max_req(self) -> float:
        """
        Estimate E[max(cpu_req, mem_req)] per replica under the configured bounded-Pareto
        request distribution.

        This improves the Little's-law-style mean-life inference because the generator
        targets effective utilization:
            eff(t) = max(total_cpu_req(t), total_mem_req(t))

        We estimate E[max(cpu, mem)] using deterministic Monte Carlo with a derived seed.
        """
        # Solve alpha for requests deterministically from bounds + mean.
        alpha_cpu = self.solve_alpha_for_bounded_mean(
            x_min=float(self.args.xmin_cpu),
            x_max=float(self.args.xmax_cpu),
            target_mean=float(self.args.mean_cpu),
        )
        
        alpha_mem = self.solve_alpha_for_bounded_mean(
            x_min=float(self.args.xmin_mem),
            x_max=float(self.args.xmax_mem),
            target_mean=float(self.args.mean_mem),
        )

        n = EFFECTIVE_MAX_REQ_MC_SAMPLES
        rng = np.random.default_rng(derive_seed(self.base_seed, "infer-mean-life-e-max-req"))

        cpu = self.sample_bounded_pareto(
            rng,
            alpha=float(alpha_cpu),
            x_min=float(self.args.xmin_cpu),
            x_max=float(self.args.xmax_cpu),
            size=n,
        ).astype(float)

        mem = self.sample_bounded_pareto(
            rng,
            alpha=float(alpha_mem),
            x_min=float(self.args.xmin_mem),
            x_max=float(self.args.xmax_mem),
            size=n,
        ).astype(float)

        e_max = float(np.mean(np.maximum(cpu, mem)))
        if not (e_max > 0.0):
            raise ValueError(f"Invalid E[max(cpu,mem)] estimate: {e_max}")
        return e_max

    def fit_pareto_alphas(self) -> None:
        """
        Fit bounded-Pareto shape parameters (alphas) from configured bounds and
        means.

        For each bounded Pareto distribution (request size, inter-arrival time,
        and pod lifetime), we solve the shape parameter alpha such that the
        distribution on [x_min, x_max] has the desired mean. The solve is
        deterministic and uses solve_alpha_for_bounded_mean.

        Caching behavior:
        - alpha_cpu, alpha_mem, and alpha_arrival are computed once and cached because
            their target means are fixed for a run.
        - alpha_life is recomputed on every call because args.mean_life may
            be updated during utilization calibration.
        """
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

        if self.alpha_cpu is None:
            self.alpha_cpu = round(
                self.solve_alpha_for_bounded_mean(
                    x_min=float(self.args.xmin_cpu),
                    x_max=float(self.args.xmax_cpu),
                    target_mean=float(self.args.mean_cpu),
                ),
                MAX_DECIMALS,
            )
            self.args.alpha_cpu = float(self.alpha_cpu)
        if self.alpha_mem is None:
            self.alpha_mem = round(
                self.solve_alpha_for_bounded_mean(
                    x_min=float(self.args.xmin_mem),
                    x_max=float(self.args.xmax_mem),
                    target_mean=float(self.args.mean_mem),
                ),
                MAX_DECIMALS,
            )
            self.args.alpha_mem = float(self.alpha_mem)
        
        LOG.info("[pareto-fit] arrival:  mean=%.6f xmin=%.6f xmax=%.6f alpha=%.6f",
                 float(self.args.mean_arrival), float(self.args.xmin_arrival), float(self.args.xmax_arrival), float(self.alpha_arrival))
        LOG.info("[pareto-fit] lifetime: mean=%.6f xmin=%.6f xmax=%.6f alpha=%.6f",
                 float(self.args.mean_life), float(self.args.xmin_life), float(self.args.xmax_life), float(self.alpha_life))
        LOG.info("[pareto-fit] cpu request:  mean=%.6f xmin=%.6f xmax=%.6f alpha=%.6f",
                 float(self.args.mean_cpu), float(self.args.xmin_cpu), float(self.args.xmax_cpu), float(self.alpha_cpu))
        LOG.info("[pareto-fit] mem request:  mean=%.6f xmin=%.6f xmax=%.6f alpha=%.6f",
                 float(self.args.mean_mem), float(self.args.xmin_mem), float(self.args.xmax_mem), float(self.alpha_mem))

    # -------------------------------------------------------------------------
    # Geometric support + expected value
    # -------------------------------------------------------------------------

    @staticmethod
    def build_trunc_geometric_support(min_val: int, max_val: int, ratio: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build the discrete integer support and PMF for a truncated geometric family
        on the inclusive range k ∈ [min_val, max_val].

        Paper alignment (geometric + truncation):
          - Geometric PMF: Chattopadhyay et al. (2014), Eq. (7), p. 31:
                f(x) = p (1-p)^(x-1) = p q^(x-1),  x = 1,2,...
          - Truncation/renormalization to a finite range: Chattopadhyay et al. (2014),
            p. 26 (definition of truncation + table of truncated PMFs).

        Mapping to this function:
          - We expose a single "ratio" parameter which corresponds to q = (1-p).
          - Over a finite support, the multiplicative constant p cancels during
            renormalization, so it is sufficient to use weights proportional to q^(x-1).

        Concretely, for k in [min_val, max_val], we assign unnormalized weights
            w(k) = ratio^(k - min_val)
        and normalize over the finite interval to obtain probabilities.

        Interpretation of ratio:
          - ratio < 1 biases probability toward smaller values (fast decay).
          - ratio > 1 biases probability toward larger values (increasing weights).
          - ratio ≈ 1 yields a uniform distribution over the support.

        Returns:
            (vals, probs) where vals is an int array of support values.
            If ratio≈1 (or support size is 1), probs is None and callers may treat
            the distribution as uniform.
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
        Compute expectation for a discrete random variable on a finite support.
        """
        return float(np.mean(vals)) if probs is None else float(np.sum(vals.astype(float) * probs.astype(float)))

    # -------------------------------------------------------------------------
    # Util metric
    # -------------------------------------------------------------------------

    def time_mean_utils_from_hist(
        self,
        times: List[float],
        u_cpu_hist: List[float],
        u_mem_hist: List[float],
    ) -> Tuple[float, float, float]:
        """
        Compute time-mean (cpu_util, mem_util, eff_util) from already-built utilization
        histories.

        The histories are assumed to represent a right-continuous step function:
        value[i] applies on [times[i], times[i+1]) and value[-1] applies on [times[-1], T].
        Values are already normalized by num_nodes (i.e., utilization units).
        """
        T = float(self.trace_time_s)
        if T <= 0.0:
            return 0.0, 0.0, 0.0
        if not times or len(times) != len(u_cpu_hist) or len(times) != len(u_mem_hist):
            return 0.0, 0.0, 0.0

        area_cpu = 0.0
        area_mem = 0.0
        area_eff = 0.0

        # Integrate over consecutive segments.
        for i in range(len(times) - 1):
            t0 = float(times[i])
            t1 = float(times[i + 1])
            dt = t1 - t0
            if dt <= 0.0:
                continue
            cpu_u = float(u_cpu_hist[i])
            mem_u = float(u_mem_hist[i])
            area_cpu += cpu_u * dt
            area_mem += mem_u * dt
            area_eff += max(cpu_u, mem_u) * dt

        # Tail to T.
        last_t = float(times[-1])
        if last_t < T:
            dt = T - last_t
            cpu_u = float(u_cpu_hist[-1])
            mem_u = float(u_mem_hist[-1])
            area_cpu += cpu_u * dt
            area_mem += mem_u * dt
            area_eff += max(cpu_u, mem_u) * dt

        return area_cpu / T, area_mem / T, area_eff / T

    # -------------------------------------------------------------------------
    # Initial pods via warm-up simulation
    # -------------------------------------------------------------------------

    def generate_initial_pods(
        self,
        rng: np.random.Generator,
        *,
        priority_vals: np.ndarray,
        priority_probs: Optional[np.ndarray],
        replicas_vals: np.ndarray,
        replicas_probs: Optional[np.ndarray],
        next_id: int,
        max_pods: int = MAX_INITIAL_PODS,
    ) -> Tuple[List[TraceRecord], int]:
        """
        Generate the initial snapshot at t=0.

        We generate a list of pods that are alive at the snapshot and aim for an
        effective requested load close to the target utilization, where:
            effective = max(total_cpu_req, total_mem_req)

        Implementation notes:
        - All returned pods have start_time = 0.0.
        - end_time is a *residual* lifetime sampled from the steady-state residual
        distribution (not the raw lifetime distribution).
        - We add pods until we reach the target request, and if the last pod causes an
        overshoot we pop it (and roll back next_id).
        """
        assert self.alpha_cpu is not None and self.alpha_mem is not None and self.alpha_life is not None and self.alpha_arrival is not None

        pods: List[TraceRecord] = []
        total_cpu_req = 0.0
        total_mem_req = 0.0
        target_total_req = float(self.args.target_util) * float(self.args.num_nodes)

        while max(total_cpu_req, total_mem_req) < target_total_req and len(pods) < int(max_pods):
            cpu_req = float(self.sample_bounded_pareto(
                rng,
                alpha=float(self.alpha_cpu),
                x_min=float(self.args.xmin_cpu),
                x_max=float(self.args.xmax_cpu),
                size=1,
            )[0])

            mem_req = float(self.sample_bounded_pareto(
                rng,
                alpha=float(self.alpha_mem),
                x_min=float(self.args.xmin_mem),
                x_max=float(self.args.xmax_mem),
                size=1,
            )[0])

            remaining = float(self.sample_steady_state_residual_lifetime(
                rng,
                alpha=float(self.alpha_life),
                x_min=float(self.args.xmin_life),
                x_max=float(self.args.xmax_life),
                size=1,
            )[0])
            if remaining <= 0.0:
                continue

            replicas = int(rng.choice(replicas_vals, p=replicas_probs))
            priority = int(rng.choice(priority_vals, p=priority_probs))

            next_id += 1
            rec = TraceRecord(
                id=next_id,
                start_time=0.0,
                end_time=round(remaining, MAX_DECIMALS),
                cpu=round(cpu_req, MAX_DECIMALS),
                mem=round(mem_req, MAX_DECIMALS),
                priority=priority,
                replicas=replicas,
            )
            pods.append(rec)
            total_cpu_req += float(rec.replicas) * float(rec.cpu)
            total_mem_req += float(rec.replicas) * float(rec.mem)

            # If we overshoot, pop the last pod and roll back the id counter.
            if max(total_cpu_req, total_mem_req) > target_total_req + 1e-12:
                last = pods.pop()
                total_cpu_req -= float(last.replicas) * float(last.cpu)
                total_mem_req -= float(last.replicas) * float(last.mem)
                next_id -= 1
                break

        N = float(self.args.num_nodes)
        cpu_util = (total_cpu_req / N) if N > 0 else 0.0
        mem_util = (total_mem_req / N) if N > 0 else 0.0
        eff_util = max(cpu_util, mem_util)

        LOG.info(
            "[initial-pods] records=%d cpu≈%.3f mem≈%.3f eff≈%.3f target=%.3f",
            len(pods),
            cpu_util,
            mem_util,
            eff_util,
            float(self.args.target_util),
        )

        return pods, next_id

    # -------------------------------------------------------------------------
    # Generate trace pod events
    # -------------------------------------------------------------------------

    def generate_trace_pod_events(
        self,
        rng: np.random.Generator,
        *,
        prio_vals: np.ndarray,
        prio_probs: Optional[np.ndarray],
        rep_vals: np.ndarray,
        rep_probs: Optional[np.ndarray],
        next_id: int,
        initial_pods: List[TraceRecord],
    ) -> Tuple[List[TraceRecord], int, List[float], List[float], List[float], List[float], List[int]]:
        """
        Generate trace pod events over the trace horizon.
        """
        assert self.alpha_cpu is not None and self.alpha_mem is not None and self.alpha_arrival is not None and self.alpha_life is not None

        state = ClusterState()
        end_heap: List[EndHeapEntry] = []

        for p in initial_pods:
            cpu_req = float(p.cpu)
            mem_req = float(p.mem)
            replicas = int(p.replicas)
            state.live_cpu_req += replicas * cpu_req
            state.live_mem_req += replicas * mem_req
            state.live_pods += replicas
            heapq.heappush(
                end_heap,
                EndHeapEntry(end_time=float(p.end_time), cpu_req=cpu_req, mem_req=mem_req, replicas=replicas),
            )

        pods: List[TraceRecord] = []

        N = float(self.args.num_nodes)

        times: List[float] = [0.0]
        u_cpu_hist: List[float] = [state.live_cpu_req / N]
        u_mem_hist: List[float] = [state.live_mem_req / N]
        u_eff_hist: List[float] = [max(u_cpu_hist[-1], u_mem_hist[-1])]
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
                end_t = float(end_heap[0].end_time)
                while end_heap and float(end_heap[0].end_time) == end_t:
                    entry = heapq.heappop(end_heap)
                    state.live_cpu_req = max(0.0, state.live_cpu_req - entry.cpu_req * entry.replicas)
                    state.live_mem_req = max(0.0, state.live_mem_req - entry.mem_req * entry.replicas)
                    state.live_pods = max(0, state.live_pods - entry.replicas)

                if end_t <= self.trace_time_s and end_t >= times[-1]:
                    times.append(end_t)
                    u_cpu_hist.append(state.live_cpu_req / N)
                    u_mem_hist.append(state.live_mem_req / N)
                    u_eff_hist.append(max(u_cpu_hist[-1], u_mem_hist[-1]))
                    pods_hist.append(state.live_pods)

            lifetime = float(self.sample_bounded_pareto(
                rng,
                alpha=float(self.alpha_life),
                x_min=float(self.args.xmin_life),
                x_max=float(self.args.xmax_life),
                size=1,
            )[0])

            end = round(start + lifetime, MAX_DECIMALS)

            cpu_req = float(self.sample_bounded_pareto(
                rng,
                alpha=float(self.alpha_cpu),
                x_min=float(self.args.xmin_cpu),
                x_max=float(self.args.xmax_cpu),
                size=1,
            )[0])

            mem_req = float(self.sample_bounded_pareto(
                rng,
                alpha=float(self.alpha_mem),
                x_min=float(self.args.xmin_mem),
                x_max=float(self.args.xmax_mem),
                size=1,
            )[0])

            replicas = int(rng.choice(rep_vals, p=rep_probs))
            priority = int(rng.choice(prio_vals, p=prio_probs))

            state.live_cpu_req += replicas * cpu_req
            state.live_mem_req += replicas * mem_req
            state.live_pods += replicas
            heapq.heappush(
                end_heap,
                EndHeapEntry(end_time=float(end), cpu_req=cpu_req, mem_req=mem_req, replicas=replicas),
            )

            next_id += 1
            pods.append(TraceRecord(
                id=next_id,
                start_time=round(start, MAX_DECIMALS),
                end_time=float(end),
                cpu=round(cpu_req, MAX_DECIMALS),
                mem=round(mem_req, MAX_DECIMALS),
                priority=priority,
                replicas=replicas,
            ))
            times.append(start)
            u_cpu_hist.append(state.live_cpu_req / N)
            u_mem_hist.append(state.live_mem_req / N)
            u_eff_hist.append(max(u_cpu_hist[-1], u_mem_hist[-1]))
            pods_hist.append(state.live_pods)

            t = start

        while end_heap and float(end_heap[0].end_time) <= float(self.trace_time_s):
            end_t = float(end_heap[0].end_time)
            while end_heap and float(end_heap[0].end_time) == end_t:
                entry = heapq.heappop(end_heap)
                state.live_cpu_req = max(0.0, state.live_cpu_req - entry.cpu_req * entry.replicas)
                state.live_mem_req = max(0.0, state.live_mem_req - entry.mem_req * entry.replicas)
                state.live_pods = max(0, state.live_pods - entry.replicas)

            if end_t >= times[-1]:
                times.append(end_t)
                u_cpu_hist.append(state.live_cpu_req / N)
                u_mem_hist.append(state.live_mem_req / N)
                u_eff_hist.append(max(u_cpu_hist[-1], u_mem_hist[-1]))
                pods_hist.append(state.live_pods)

        return pods, next_id, times, u_eff_hist, u_cpu_hist, u_mem_hist, pods_hist

    # -------------------------------------------------------------------------
    # Make trace: one generation pass
    # -------------------------------------------------------------------------

    def make_trace(self, iter_seed: int) -> Tuple[List[TraceRecord], List[TraceRecord], float, Dict[str, object]]:
        """
        Make one trace generation pass: initial pods + trace pods.

        Measured utilization is the time-mean effective utilization:
            eff(t) = max(cpu_util(t), mem_util(t))
        computed from the already-built utilization histories.
        """
        rng_initial = np.random.default_rng(derive_seed(iter_seed, "initial-pods"))
        rng_trace = np.random.default_rng(derive_seed(iter_seed, "trace-pods"))

        self.fit_pareto_alphas()

        priority_vals, priority_probs = self.build_trunc_geometric_support(
            int(self.args.priority_min),
            int(self.args.priority_max),
            float(self.args.priority_ratio),
        )
        replicas_vals, replicas_probs = self.build_trunc_geometric_support(
            int(self.args.replicas_min),
            int(self.args.replicas_max),
            float(self.args.replicas_ratio),
        )

        next_id = 0
        initial_pods, next_id = self.generate_initial_pods(
            rng_initial,
            priority_vals=priority_vals, priority_probs=priority_probs,
            replicas_vals=replicas_vals, replicas_probs=replicas_probs,
            next_id=next_id,
        )

        trace_pods, next_id, times, u_eff_hist, u_cpu_hist, u_mem_hist, pods_hist = self.generate_trace_pod_events(
            rng_trace,
            prio_vals=priority_vals, prio_probs=priority_probs,
            rep_vals=replicas_vals, rep_probs=replicas_probs,
            next_id=next_id,
            initial_pods=initial_pods,
        )

        # Store plot series (plot expects u_req_hist but it is effective util in our case).
        self.times = times
        self.u_eff_hist = u_eff_hist
        self.u_cpu_hist = u_cpu_hist
        self.u_mem_hist = u_mem_hist
        self.pods_hist = pods_hist
        self.initial_pods_count = int(sum(int(p.replicas) for p in initial_pods))

        # Measure util from histories
        util_cpu, util_mem, util_eff = self.time_mean_utils_from_hist(times, u_cpu_hist, u_mem_hist)
        util = float(util_eff)

        all_pods = initial_pods + trace_pods
        max_priority = max((int(p.priority) for p in all_pods), default=0)

        extra_info: Dict[str, object] = {
            "num_nodes": int(self.args.num_nodes),
            "trace_time_s": round(float(self.trace_time_s), 6),
            "seed": int(self.args.seed),
            "rng": {
                "iter_seed": int(iter_seed),
                "derived": {
                    "initial_pods": int(derive_seed(iter_seed, "initial-pods")),
                    "trace_pods": int(derive_seed(iter_seed, "trace-pods")),
                },
            },
            "measured_cpu_util_time_mean": float(util_cpu),
            "measured_mem_util_time_mean": float(util_mem),
            "measured_util_time_mean": float(util_eff),  # effective
            "target_util_time_mean": float(self.args.target_util),
            "calibration": {
                "util_tol": float(MEAN_LIFETIME_CALIBRATION_UTIL_TOLERANCE),
                "calib_max_iter": int(MEAN_LIFETIME_CALIBRATION_MAX_ITERATIONS),
                "initial_max_pods": int(MAX_INITIAL_PODS),
            },
            "derived_mean_lifetime_s": round(float(self.args.mean_life), 6),
            "pareto": {
                "cpu": {"alpha": float(self.alpha_cpu), "xmin": float(self.args.xmin_cpu), "xmax": float(self.args.xmax_cpu), "mean": float(self.args.mean_cpu)},
                "mem": {"alpha": float(self.alpha_mem), "xmin": float(self.args.xmin_mem), "xmax": float(self.args.xmax_mem), "mean": float(self.args.mean_mem)},
                "arrival": {"alpha": float(self.alpha_arrival), "xmin": float(self.args.xmin_arrival), "xmax": float(self.args.xmax_arrival), "mean": float(self.args.mean_arrival)},
                "life": {"alpha": float(self.alpha_life), "xmin": float(self.args.xmin_life), "xmax": float(self.args.xmax_life), "mean": float(self.args.mean_life)},
            },
            "priority": {"min": int(self.args.priority_min), "max": int(self.args.priority_max), "ratio": float(self.args.priority_ratio)},
            "replicas": {"min": int(self.args.replicas_min), "max": int(self.args.replicas_max), "ratio": float(self.args.replicas_ratio)},
            "counts": {
                "initial_pods": int(len(initial_pods)),
                "trace_pods": int(len(trace_pods)),
                "initial_replicas_total": int(sum(int(p.replicas) for p in initial_pods)),
                "trace_replicas_total": int(sum(int(p.replicas) for p in trace_pods)),
            },
            "max_priority_seen": int(max_priority),
            "files": {
                "initial_json": str(self.initial_path),
                "trace_json": str(self.trace_path),
                "utilization_plot": str(self.util_plot_path),
                "histograms_plot": str(self.hist_plot_path),
            },
        }

        return initial_pods, trace_pods, util, extra_info

    # -------------------------------------------------------------------------
    # Calibrate mean-life to hit target utilization
    # -------------------------------------------------------------------------

    def calibrate_mean_lifetime(
        self,
        tolerance=float(MEAN_LIFETIME_CALIBRATION_UTIL_TOLERANCE),
        max_iterations=int(MEAN_LIFETIME_CALIBRATION_MAX_ITERATIONS),
    ) -> Tuple[List[TraceRecord], List[TraceRecord], Dict[str, object]]:
        """
        Calibrate args.mean_life to match the requested target utilization.

        Objective:
        Adjust the bounded-Pareto mean lifetime parameter (args.mean_life) so that
        the generated trace's time-mean effective utilization matches
        args.target_util within tolerance.

        Effective utilization is defined as:
            eff(t) = max(cpu_util(t), mem_util(t))

        How it works:
        1) Use a fixed iteration seed (CRN: common random numbers) across calibration
            iterations. This reduces RNG noise so utilization changes mostly reflect
            changes in mean_life.
        2) For each iteration, generate a full trace realization via make_trace and
            measure time-mean effective utilization.
        3) Compute relative error err = |measured - target| / target.
        4) If within tolerance, stop and return the best trace seen.
        5) Otherwise update mean_life proportionally:
                mean_life <- mean_life * (target / measured)
            Intuition: if utilization is low, increase lifetimes; if high, decrease.
        6) Clamp mean_life to stay strictly within (xmin_life, xmax_life) so the
            bounded-Pareto alpha solver remains valid.

        Returns:
        (best_initial_pods, best_trace_pods, best_extra) for the closest iteration
        encountered (including one that meets tolerance).
        """
        target = float(self.args.target_util)
        iteration_seed = int(derive_seed(self.base_seed, "calibrate-mean-life-crn"))

        best_error = float("inf")
        best_initial: List[TraceRecord] = []
        best_trace: List[TraceRecord] = []
        best_extra: Dict[str, object] = {}

        for it in range(1, max_iterations + 1):
            initial_pods, trace_pods, measured, extra = self.make_trace(iteration_seed)

            err = abs(measured - target) / max(1e-12, target)
            if err < best_error:
                best_error = err
                best_initial, best_trace, best_extra = initial_pods, trace_pods, extra

            LOG.info(
                "[calibrate-mean-life] iteration=%d measured=%.4f target=%.4f rel_err=%.2f%% mean_life=%.3fs",
                it, measured, target, 100.0 * err, float(self.args.mean_life),
            )

            if err <= tolerance:
                return best_initial, best_trace, best_extra
            if measured <= 1e-12:
                raise RuntimeError("Measured utilization is ~0; cannot calibrate mean-life.")

            new_mean = float(self.args.mean_life) * (target / measured)

            xmin = float(self.args.xmin_life)
            xmax = float(self.args.xmax_life)
            new_mean = max(new_mean, xmin * 1.001)
            new_mean = min(new_mean, xmax * 0.999)

            self.args.mean_life = float(new_mean)

        LOG.warning("[calibrate-mean-life] did not reach util-tolerance; best error=%.2f%%", 100.0 * best_error)
        return best_initial, best_trace, best_extra

    # -------------------------------------------------------------------------
    # Write outputs: JSON and info YAML
    # -------------------------------------------------------------------------

    def write_outputs(self, initial_pods: List[TraceRecord], trace_pods: List[TraceRecord], extra_info: Dict[str, object]) -> None:
        """
        Write output files: initial JSON, trace JSON, info YAML.
        """
        _, _, util_eff = self.time_mean_utils_from_hist(self.times, self.u_cpu_hist, self.u_mem_hist)
        self.pods_to_json(self.initial_path, initial_pods)
        self.pods_to_json(self.trace_path, trace_pods)

        LOG.info("[utilization] util-time-mean over whole trace: %.4f (target=%.4f)",
                float(util_eff), float(self.args.target_util))

        self.write_info_file(extra=extra_info)

    @staticmethod
    def pods_to_json(path: Path, pods: List[TraceRecord]) -> None:
        """
        Write pods to a JSON file at the given path.
        """
        obj = {"pods": [asdict(p) for p in pods]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        LOG.info("wrote %s (%d records)", path, len(pods))

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
    # Runner
    # -------------------------------------------------------------------------

    def run_seed(self) -> None:
        """
        Run one seeded experiment and generate outputs + plots.
        """
        if self.args.mean_life is None:
            self.args.mean_life = self.infer_mean_lifetime_from_target_util()
            LOG.info(
                "[inferred-mean-life] mean_life=%.3fs from target-util=%.3f",
                float(self.args.mean_life),
                float(self.args.target_util),
            )
            initial_pods, trace_pods, extra_info = self.calibrate_mean_lifetime()
        else:
            iteration_seed = int(derive_seed(self.base_seed, "provided-mean-life"))
            initial_pods, trace_pods, measured, extra_info = self.make_trace(iteration_seed)
            extra_info.setdefault("calibration", {})
            extra_info["calibration"].update({"mode": "skipped", "reason": "mean_lifetime_provided", "iterations": 1})
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
            u_req_hist=self.u_eff_hist,
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
            alpha_cpu=float(self.args.alpha_cpu),
            xmin_cpu=float(self.args.xmin_cpu),
            xmax_cpu=float(self.args.xmax_cpu),
            alpha_mem=float(self.args.alpha_mem),
            xmin_mem=float(self.args.xmin_mem),
            xmax_mem=float(self.args.xmax_mem),
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

            setup_logging(name=LOGGER_NAME, prefix=f"[{LOGGER_NAME}] ", level=a.log_level)
            gen = TraceGenerator(
                a,
                resolved=True,
                create_figures_dir=True,
                log_args=False,
                setup_logger=False,
            )
            gen.run_seed()

        header, footer = make_header_footer("DONE")
        LOG.info("\n%s\n%s\n%s", header, "All runs complete.", footer)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    if os.getenv("TRACE_GENERATOR_NOOP") == "1":
        return
    args = build_arg_parser().parse_args()

    if getattr(args, "job_dir", None):
        job_runs = TraceGenerator.expand_job_dir_runs(args)
        total = len(job_runs)
        for i, a in enumerate(job_runs, start=1):
            header, footer = make_header_footer(
                f"JOB {Path(a.job_file).name} ({i}/{total})" if total > 1 else f"JOB {Path(a.job_file).name}"
            )
            setup_logging(name=LOGGER_NAME, prefix=f"[{LOGGER_NAME}] ", level=a.log_level)
            LOG.info("\n%s\njob_file=%s\noutput_dir=%s\n%s", header, str(a.job_file), str(a.output_dir), footer)

            gen = TraceGenerator(
                a,
                resolved=True,
                create_figures_dir=None,
                log_args=True,
                setup_logger=False,
            )
            gen.run()
        return

    trace_generator = TraceGenerator(args)
    trace_generator.run()


if __name__ == "__main__":
    main()
