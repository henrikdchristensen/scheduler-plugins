#!/usr/bin/env python3
# trace_generator.py

import argparse, heapq, json, logging, os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from scripts.helpers.general_helpers import (
    parse_duration_to_seconds,
    setup_logging,
    build_cli_cmd,
    write_info_file,
    log_args_block,
)
from scripts.kwok_trace_replayer.trace_helpers import (
    TraceRecord,
)
from scripts.kwok_trace_replayer.plot_helpers import (
    plot_histogram_with_pareto,
    plot_bar_with_geometric
)

#####################################################################
# Constants
#####################################################################
SOLVE_ALPHA_MAX_ITERATIONS  = 10_000 # number of bisection iterations
SOLVE_ALPHA_TOLERANCE       = 1e-3   # tolerance on mean error for alpha solving
SOLVE_ALPHA_SAMPLES         = 50_000 # Monte Carlo samples per mean evaluation
SOLVE_ALPHA_LOWER_BOUND     = 0.1    # lower bound on alpha search
SOLVE_ALPHA_UPPER_BOUND     = 10.0   # upper bound on alpha search

MAX_DECIMALS = 6  # 6 is enough for microsecond precision.

#####################################################################
# Logging setup
#####################################################################
LOGGER_NAME = "trace-generator"
LOG = logging.getLogger(LOGGER_NAME)

#####################################################################
# Cluster state dataclass
#####################################################################
@dataclass
class ClusterState:
    num_nodes: int
    live_cpu: float = 0.0
    live_mem: float = 0.0
    live_pods: int = 0

#####################################################################
# Heap entries for completed pods
#####################################################################
@dataclass(order=True)
class EndHeapEntry:
    end_time: float
    cpu: float = field(compare=False)
    mem: float = field(compare=False)
    replicas: int = field(compare=False)

#####################################################################
# Argument parser
#####################################################################
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=("Generate pod traces."))

    # General
    p.add_argument("--output-dir", dest="output_dir", default="./output-traces",
        help="Directory to store generated JSON and plots (default: ./output-traces).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--log-level", dest="log_level", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    # Cluster / utilization
    p.add_argument("--num-nodes", type=int, default=8,
        help="Number of nodes (each has capacity 1.0 CPU, 1.0 MEM).",
    )
    p.add_argument("--trace-time", type=str, default="3600s",
        help="Trace time; seconds or suffixed like '1h', '30m', ...",
    )

    # Inter-arrival times (seconds)
    p.add_argument("--xmin-arrival", type=float, default=0.01,
        help="Pareto I x_min for inter-arrival times (seconds).",
    )
    p.add_argument("--xmax-arrival", type=float, default=None,
        help="Maximum inter-arrival time (seconds, upper clamp).",
    )
    p.add_argument("--mean-arrival", type=float, required=True,
        help="Target mean inter-arrival time (seconds).",
    )

    # Lifetimes (seconds)
    p.add_argument("--xmin-life", type=float, default=10.0,
        help="Pareto I x_min for lifetimes (seconds).",
    )
    p.add_argument("--xmax-life", type=float, default=None,
        help="Maximum lifetime (seconds, upper clamp).",
    )
    p.add_argument("--mean-life", type=float, required=True,
        help="Target mean lifetime (seconds).",
    )

    # CPU distribution params
    p.add_argument("--xmin-cpu", type=float, default=0.01,
        help="Pareto I x_min (lower bound) for CPU requests.",
    )
    p.add_argument("--xmax-cpu", type=float, default=1.0,
        help="Maximum CPU request (upper clamp).",
    )
    p.add_argument("--mean-cpu", type=float, required=True,
        help="Target mean CPU request (fraction of node capacity).",
    )

    # MEM distribution params
    p.add_argument("--xmin-mem", type=float, default=0.01,
        help="Pareto I x_min (lower bound) for MEM requests.",
    )
    p.add_argument("--xmax-mem", type=float, default=1.0,
        help="Maximum MEM request (upper clamp).",
    )
    p.add_argument("--mean-mem", type=float, required=True,
        help="Target mean MEM request (fraction of node capacity).",
    )

    # Priority (min/max like replicas)
    p.add_argument("--priority-min", type=int, default=1,
        help="Minimum priority value (inclusive).",
    )
    p.add_argument("--priority-max", type=int, default=3,
        help="Maximum priority value (inclusive).",
    )
    p.add_argument("--priority-ratio", type=float, default=1.0,
        help=("Geometric ratio factor in (0,1]. Use ratio=1.0 for uniform."),
    )

    # Replica counts
    p.add_argument("--replicas-min", type=int, default=1,
        help="Minimum replicas per trace pod (default: 1).",
    )
    p.add_argument("--replicas-max", type=int, default=1,
        help="Maximum replicas per trace pod (inclusive, default: 1).",
    )
    p.add_argument("--replicas-ratio", type=float, default=1.0,
        help=("Geometric ratio factor in (0,1]. Use ratio=1.0 for uniform."),
    )

    return p

def round_float_args(args, ndigits: int) -> None:
    """
    Round all float fields on the argparse Namespace to a fixed number of decimals.
    """
    for name, value in vars(args).items():
        if isinstance(value, float):
            setattr(args, name, round(value, ndigits))

#####################################################################
# Trace generator class
#####################################################################
class TraceGenerator:
    """
    Encapsulates trace generation, statistics, and plotting.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = np.random.default_rng(args.seed)

        # Resolve and create output directory
        self.output_dir: Path = Path(self.args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir: Path = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Build paths under output_dir
        self.trace_path = self.output_dir / "trace.json"
        self.util_plot_path = self.figures_dir / "utilization.png"
        self.hist_plot_path = self.figures_dir / "histograms.png"

        # Filled by _fit_alphas()
        self.alpha_cpu: float | None = None
        self.alpha_mem: float | None = None
        self.alpha_arrival: float | None = None
        self.alpha_life: float | None = None

        # Results of _generate_trace()
        self.pods: List[TraceRecord] = []
        self.times: List[float] = []
        self.u_cpu_hist: List[float] = []
        self.u_mem_hist: List[float] = []
        self.max_runnable_pods_hist: List[int] = []

        # Parse trace_time duration string into seconds
        self.trace_time_s = parse_duration_to_seconds(self.args.trace_time)

        # Side-effects are delayed until first generation/run
        self._prepared = False

    def _ensure_prepared(self) -> None:
        if self._prepared:
            return
        LOG.info("logging arguments and git info to output_dir...")
        self._write_info_file()
        self.log_args()

        LOG.info("fitting Pareto alphas parameters...")
        self._fit_alphas()

        self._prepared = True

    ##############################################
    # ------------ Info/logging helpers ----------
    ##############################################
    def log_args(self) -> None:
        include = [
            "output_dir","seed","log_level","num_nodes","trace_time",
            "priority_min","priority_max","priority_ratio",
            "replicas_min","replicas_max","replicas_ratio",
            "xmin_cpu","xmax_cpu","mean_cpu",
            "xmin_mem","xmax_mem","mean_mem",
            "xmin_arrival","xmax_arrival","mean_arrival",
            "xmin_life","xmax_life","mean_life",
        ]
        log_args_block(LOG, self.args, title="ARGS", include=include)

    def _write_info_file(self) -> None:
        """
        Write info_generate.yaml in output_dir with git + CLI + args.
        """
        try:
            out_path = self.output_dir / "info_generate.yaml"
            inputs = {
                "cli-cmd": build_cli_cmd(),
                "args": {k: v for k, v in vars(self.args).items()},
            }
            write_info_file(out_path, inputs=inputs, logger=LOG)
        except Exception as e:
            LOG.warning("failed to write info_generate.yaml: %s", e)

    ##############################################
    # ------------ Pareto helpers ----------------
    ##############################################
    @staticmethod
    def _sample_pareto(
        rng: np.random.Generator,
        alpha: float,
        x_min: float,
        x_max: float | None = None,
        size: int = 1,
    ) -> np.ndarray:
        """
        Sample from a Pareto Type I distribution with parameters (alpha, x_min):
            X = x_min / U^(1/alpha),  U ~ Uniform(0, 1).
        With optional hard upper truncation at x_max.
        """
        if alpha <= 0.0 or x_min <= 0.0:
            raise ValueError("Pareto alpha and x_min must be > 0.")
        if x_max is not None and x_max < x_min:
            raise ValueError("Pareto x_max must be >= x_min when provided.")

        u = rng.random(size)
        if x_max is not None:
            u_min_tail = (x_min / x_max) ** alpha  # in (0, 1]
            u_min = max(1e-12, float(u_min_tail))
            u = np.clip(u, u_min, 1.0)
        else:
            u = np.clip(u, 1e-12, 1.0)

        x = x_min / (u ** (1.0 / alpha))
        return x

    @classmethod
    def _solve_alpha_of_pareto_for_mean(
        cls,
        rng: np.random.Generator,
        x_min: float,
        x_max: float | None,
        target_mean: float,
    ) -> float:
        """
        Find alpha such that E[X | alpha, x_min, x_max] ≈ target_mean using
        Monte Carlo + bisection.
        """
        if x_min <= 0.0:
            raise ValueError("x_min must be > 0.")
        if x_max is not None and not (x_min < target_mean < x_max):
            raise ValueError(
                f"target_mean={target_mean} must lie between x_min={x_min} and x_max={x_max}"
            )
        if x_max is None and target_mean <= x_min:
            raise ValueError(
                f"target_mean={target_mean} must be > x_min={x_min} for unbounded Pareto."
            )

        alpha_low = SOLVE_ALPHA_LOWER_BOUND
        alpha_high = SOLVE_ALPHA_UPPER_BOUND

        if x_max is None:
            alpha_low = max(alpha_low, 1.0001)

        m_low = float(cls._sample_pareto(rng, alpha_low, x_min, x_max, size=SOLVE_ALPHA_SAMPLES).mean())
        m_high = float(cls._sample_pareto(rng, alpha_high, x_min, x_max, size=SOLVE_ALPHA_SAMPLES).mean())

        if not (m_high <= target_mean <= m_low):
            raise ValueError(
                f"Target mean {target_mean} not bracketed by MC means "
                f"[{m_high:.4f}, {m_low:.4f}] for alpha in [{alpha_low}, {alpha_high}]. "
                "Adjust xmin/xmax or alpha bounds."
            )

        for _ in range(SOLVE_ALPHA_MAX_ITERATIONS):
            alpha_mid = 0.5 * (alpha_low + alpha_high)
            m_mid = float(
                cls._sample_pareto(rng, alpha_mid, x_min, x_max, size=SOLVE_ALPHA_SAMPLES).mean()
            )

            if m_mid >= target_mean:
                alpha_low, m_low = alpha_mid, m_mid
            else:
                alpha_high, m_high = alpha_mid, m_mid

            if abs(m_mid - target_mean) <= SOLVE_ALPHA_TOLERANCE * target_mean:
                return alpha_mid

        raise RuntimeError("Failed to converge to target mean within max_iter")

    def _fit_alphas(self) -> None:
        args = self.args

        alpha_cpu = self._solve_alpha_of_pareto_for_mean(
            self.rng, x_min=args.xmin_cpu, x_max=args.xmax_cpu, target_mean=args.mean_cpu
        )
        alpha_mem = self._solve_alpha_of_pareto_for_mean(
            self.rng, x_min=args.xmin_mem, x_max=args.xmax_mem, target_mean=args.mean_mem
        )
        alpha_arrival = self._solve_alpha_of_pareto_for_mean(
            self.rng, x_min=args.xmin_arrival, x_max=args.xmax_arrival, target_mean=args.mean_arrival
        )
        alpha_life = self._solve_alpha_of_pareto_for_mean(
            self.rng, x_min=args.xmin_life, x_max=args.xmax_life, target_mean=args.mean_life
        )

        self.alpha_cpu = round(alpha_cpu, MAX_DECIMALS)
        self.alpha_mem = round(alpha_mem, MAX_DECIMALS)
        self.alpha_arrival = round(alpha_arrival, MAX_DECIMALS)
        self.alpha_life = round(alpha_life, MAX_DECIMALS)

        args.alpha_cpu = alpha_cpu
        args.alpha_mem = alpha_mem
        args.alpha_arrival = alpha_arrival
        args.alpha_life = alpha_life

        LOG.info(
            "[pareto-fit] CPU:      mean=%.4f  xmin=%.4f  xmax=%.4f  alpha≈%.4f",
            args.mean_cpu, args.xmin_cpu, args.xmax_cpu, alpha_cpu,
        )
        LOG.info(
            "[pareto-fit] MEM:      mean=%.4f  xmin=%.4f  xmax=%.4f  alpha≈%.4f",
            args.mean_mem, args.xmin_mem, args.xmax_mem, alpha_mem,
        )
        LOG.info(
            "[pareto-fit] arrival:  mean=%.4f  xmin=%.4f  xmax=%s  alpha≈%.4f",
            args.mean_arrival, args.xmin_arrival, str(args.xmax_arrival), alpha_arrival,
        )
        LOG.info(
            "[pareto-fit] lifetime: mean=%.4f  xmin=%.4f  xmax=%s  alpha≈%.4f",
            args.mean_life, args.xmin_life, str(args.xmax_life), alpha_life,
        )

    ##############################################
    # ------------ Geometric helpers ------------
    ##############################################
    @staticmethod
    def _build_geometric_support(
        min_val: int,
        max_val: int,
        ratio: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if ratio <= 0.0:
            raise ValueError("Geometric ratio factor must be > 0.")

        lo = int(min_val)
        hi = int(max_val)
        if hi < lo:
            hi = lo

        values = np.arange(lo, hi + 1, dtype=int)
        n = values.size

        if n == 1 or np.isclose(ratio, 1.0):
            return values, None

        exponents = np.arange(n, dtype=float)
        weights = ratio ** exponents
        probs = weights / weights.sum()
        return values, probs

    ##############################################
    # ------------ Utilization helpers -----------
    ##############################################
    @staticmethod
    def _delete_completed_pods(
        state: ClusterState,
        end_heap: List[EndHeapEntry],
        t: float,
    ) -> None:
        while end_heap and end_heap[0].end_time <= t:
            entry = heapq.heappop(end_heap)
            total_cpu = entry.cpu * entry.replicas
            total_mem = entry.mem * entry.replicas
            state.live_cpu = max(0.0, state.live_cpu - total_cpu)
            state.live_mem = max(0.0, state.live_mem - total_mem)
            state.live_pods = max(0, state.live_pods - entry.replicas)

    def _summarize_utilization(self) -> None:
        if not self.times:
            LOG.info("[utilization] no utilization snapshots recorded.")
            return

        u_cpu_arr = np.asarray(self.u_cpu_hist, dtype=float)
        u_mem_arr = np.asarray(self.u_mem_hist, dtype=float)
        u_eff_arr = np.maximum(u_cpu_arr, u_mem_arr)

        def _stats(arr: np.ndarray) -> tuple[float, float, float]:
            return float(arr.mean()), float(arr.min()), float(arr.max())

        mean_cpu, min_cpu, max_cpu = _stats(u_cpu_arr)
        mean_mem, min_mem, max_mem = _stats(u_mem_arr)
        mean_eff, min_eff, max_eff = _stats(u_eff_arr)

        LOG.info("[utilization] CPU: mean=%.3f, min=%.3f, max=%.3f", mean_cpu, min_cpu, max_cpu)
        LOG.info("[utilization] MEM: mean=%.3f, min=%.3f, max=%.3f", mean_mem, min_mem, max_mem)
        LOG.info(
            "[utilization] EFF(max(cpu,mem)): mean=%.3f, min=%.3f, max=%.3f",
            mean_eff, min_eff, max_eff,
        )

    ##############################################
    # ------------ Plotting helpers --------------
    ##############################################
    def _plot_trace_utilization(self) -> None:
        max_time = max(self.times)

        if max_time <= 7 * 3600:
            x_scale = 1.0 / 60.0
            x_label = "Time (minutes)"
        elif max_time <= 7 * 24 * 3600:
            x_scale = 1.0 / 3600.0
            x_label = "Time (hours)"
        else:
            x_scale = 1.0 / (24.0 * 3600.0)
            x_label = "Time (days)"

        fig, ax1 = plt.subplots(figsize=(9, 4))

        ax1.plot(self.times, self.u_cpu_hist, label="CPU utilization",
                 linestyle="--", color="tab:blue", alpha=0.7, linewidth=0.5)
        ax1.plot(self.times, self.u_mem_hist, label="Memory utilization",
                 linestyle="--", color="tab:orange", alpha=0.7, linewidth=0.5)

        ax1.set_xlabel(x_label, labelpad=20)
        ax1.set_ylabel("Utilization (fraction of total capacity)")

        ax2 = ax1.twinx()
        ax2.plot(self.times, self.max_runnable_pods_hist, linestyle="-",
                 linewidth=1.0, color="tab:green", alpha=0.7, label="Max runnable pods")
        ax2.set_ylabel("Max runnable pods")

        ax1.grid(True, linestyle="--", alpha=0.4)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        events: List[tuple[float, str]] = []
        for p in self.pods:
            events.append((p.start_time, "C"))
            events.append((p.end_time, "D"))
        events.sort(key=lambda e: e[0])

        xticks = ax1.get_xticks()
        xticks = [t for t in xticks if 0.0 <= t <= max_time]

        if len(events) > 0 and len(xticks) > 0:
            ax1.set_xticks(xticks)
            base_labels = [f"{t * x_scale:.0f}" for t in xticks]
            ax1.set_xticklabels(base_labels)

            idx = 0
            n_events = len(events)
            for i, tick in enumerate(xticks):
                if tick < 0:
                    continue
                left = 0.0 if i == 0 else xticks[i - 1]
                right = tick
                c_count = 0
                d_count = 0
                while idx < n_events and events[idx][0] <= right:
                    t_ev, kind = events[idx]
                    idx += 1
                    if t_ev > left:
                        if kind == "C":
                            c_count += 1
                        else:
                            d_count += 1
                if c_count or d_count:
                    ax1.text(
                        tick,
                        -0.1,
                        f"+{c_count} -{d_count}",
                        transform=ax1.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=8,
                    )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
        plt.savefig(self.util_plot_path)
        plt.close(fig)
        LOG.info("saved utilization plot to %s", self.util_plot_path)

    def _plot_generated_histograms(self) -> None:
        pods_sorted = sorted(self.pods, key=lambda p: p.start_time)
        start_times = np.array([p.start_time for p in pods_sorted], dtype=float)
        cpu_vals = np.array([p.cpu for p in pods_sorted], dtype=float)
        mem_vals = np.array([p.mem for p in pods_sorted], dtype=float)
        lifetimes = np.array([p.end_time - p.start_time for p in pods_sorted], dtype=float)
        prio_vals = np.array([p.priority for p in pods_sorted], dtype=int)
        replicas_vals = np.array([p.replicas for p in pods_sorted], dtype=int)

        inter_arrivals = np.empty_like(start_times)
        if len(start_times) > 0:
            inter_arrivals[0] = start_times[0]
        if len(start_times) > 1:
            inter_arrivals[1:] = np.diff(start_times)

        fig, axes = plt.subplots(3, 2, figsize=(5, 4.5))
        axes = axes.flatten()

        plot_histogram_with_pareto(
            axes[0], inter_arrivals,
            title="Generated inter-arrival times",
            x_label="Δt (seconds)", y_label="Probability density",
            bins=80, log_y=True, x_max=self.args.xmax_arrival, scale=1.0,
            pareto_fit=True, pareto_alpha=self.args.alpha_arrival, pareto_xmin=self.args.xmin_arrival,
        )
        plot_histogram_with_pareto(
            axes[1], lifetimes,
            title="Generated lifetimes",
            x_label="Lifetime (seconds)", y_label="Probability density",
            bins=80, log_y=True, x_max=self.args.xmax_life, scale=1.0,
            pareto_fit=True, pareto_alpha=self.args.alpha_life, pareto_xmin=self.args.xmin_life,
        )
        plot_histogram_with_pareto(
            axes[2], cpu_vals,
            title="Generated CPU requests",
            x_label="Requested CPU (fraction of node capacity)", y_label="Probability density",
            bins=80, log_y=True, x_max=self.args.xmax_cpu, scale=1.0,
            pareto_fit=True, pareto_alpha=self.args.alpha_cpu, pareto_xmin=self.args.xmin_cpu,
        )
        plot_histogram_with_pareto(
            axes[3], mem_vals,
            title="Generated memory requests",
            x_label="Requested memory (fraction of node capacity)", y_label="Probability density",
            bins=80, log_y=True, x_max=self.args.xmax_mem, scale=1.0,
            pareto_fit=True, pareto_alpha=self.args.alpha_mem, pareto_xmin=self.args.xmin_mem,
        )

        plot_bar_with_geometric(
            axes[4], prio_vals,
            title="Generated priorities",
            x_label="Priority", y_label="Probability mass",
            geom_fit=True, geom_ratio=self.args.priority_ratio,
            x_min=self.args.priority_min, x_max=self.args.priority_max,
        )
        plot_bar_with_geometric(
            axes[5], replicas_vals,
            title="Generated replicas",
            x_label="Replicas", y_label="Probability mass",
            geom_fit=True, geom_ratio=self.args.replicas_ratio,
            x_min=self.args.replicas_min, x_max=self.args.replicas_max,
        )

        fig.tight_layout()
        fig.savefig(self.hist_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOG.info("saved generated histograms to %s", self.hist_plot_path)

    ##############################################
    # ------------ Trace generator ---------------
    ##############################################
    def _generate_trace(self) -> None:
        self._ensure_prepared()
        rng = self.rng

        if self.args.xmin_arrival <= 0.0:
            raise ValueError("xmin_arrival must be > 0 for Pareto I.")
        if self.args.xmin_life <= 0.0:
            raise ValueError("xmin_life must be > 0 for Pareto I.")

        if getattr(self.args, "alpha_arrival", None) is None:
            raise RuntimeError("alpha_arrival missing (did _fit_alphas run?)")
        if getattr(self.args, "alpha_life", None) is None:
            raise RuntimeError("alpha_life missing (did _fit_alphas run?)")
        if getattr(self.args, "alpha_cpu", None) is None:
            raise RuntimeError("alpha_cpu missing (did _fit_alphas run?)")
        if getattr(self.args, "alpha_mem", None) is None:
            raise RuntimeError("alpha_mem missing (did _fit_alphas run?)")

        p_min = max(1, int(self.args.priority_min))
        p_max = max(p_min, int(self.args.priority_max))
        prio_values, prio_probs = self._build_geometric_support(
            min_val=p_min,
            max_val=p_max,
            ratio=self.args.priority_ratio,
        )

        r_min = max(1, int(self.args.replicas_min))
        r_max = max(r_min, int(self.args.replicas_max))
        replica_values, replica_probs = self._build_geometric_support(
            min_val=r_min,
            max_val=r_max,
            ratio=self.args.replicas_ratio,
        )

        state = ClusterState(num_nodes=self.args.num_nodes)
        end_heap: List[EndHeapEntry] = []
        pods: List[TraceRecord] = []

        times: List[float] = [0.0]
        u_cpu_hist: List[float] = [0.0]
        u_mem_hist: List[float] = [0.0]
        max_runnable_pods_hist: List[int] = [0]
        current_time = 0.0
        pod_id = 0

        LOG.info("starting trace generation (trace_time=%.1f seconds)", self.trace_time_s)

        while True:
            if current_time >= self.trace_time_s:
                break

            delta_t = float(self._sample_pareto(
                rng, self.args.alpha_arrival, self.args.xmin_arrival, self.args.xmax_arrival, size=1
            )[0])
            start_time = current_time + delta_t

            if start_time >= self.trace_time_s:
                break

            self._delete_completed_pods(state, end_heap, start_time)

            lifetime = float(self._sample_pareto(
                rng, self.args.alpha_life, self.args.xmin_life, self.args.xmax_life, size=1
            )[0])
            end_time = start_time + lifetime

            cpu = float(self._sample_pareto(
                rng, self.args.alpha_cpu, self.args.xmin_cpu, self.args.xmax_cpu, size=1
            )[0])
            mem = float(self._sample_pareto(
                rng, self.args.alpha_mem, self.args.xmin_mem, self.args.xmax_mem, size=1
            )[0])

            priority = int(rng.choice(prio_values, p=prio_probs))
            replicas = int(rng.choice(replica_values, p=replica_probs))

            new_live_cpu = state.live_cpu + replicas * cpu
            new_live_mem = state.live_mem + replicas * mem
            u_cpu = new_live_cpu / float(self.args.num_nodes)
            u_mem = new_live_mem / float(self.args.num_nodes)

            pod_id += 1
            pods.append(
                TraceRecord(
                    id=pod_id,
                    start_time=round(start_time, MAX_DECIMALS),
                    end_time=round(end_time, MAX_DECIMALS),
                    cpu=round(cpu, MAX_DECIMALS),
                    mem=round(mem, MAX_DECIMALS),
                    priority=priority,
                    replicas=replicas,
                )
            )

            state.live_cpu = new_live_cpu
            state.live_mem = new_live_mem
            state.live_pods += replicas
            heapq.heappush(end_heap, EndHeapEntry(end_time=end_time, cpu=cpu, mem=mem, replicas=replicas))
            current_time = start_time

            times.append(start_time)
            u_cpu_hist.append(u_cpu)
            u_mem_hist.append(u_mem)
            max_runnable_pods_hist.append(state.live_pods)

        self.pods = pods
        self.times = times
        self.u_cpu_hist = u_cpu_hist
        self.u_mem_hist = u_mem_hist
        self.max_runnable_pods_hist = max_runnable_pods_hist

        LOG.info("total pods %d generated before stopping at t=%.1f seconds.", len(self.pods), self.trace_time_s)

    def _write_trace(self) -> None:
        obj = {
            "meta": {
                "num_nodes": self.args.num_nodes,
                "trace_time_s": round(self.trace_time_s, 2),
                "seed": self.args.seed,
                "cpu_params": {
                    "pareto_alpha": self.alpha_cpu,
                    "pareto_xmin":  self.args.xmin_cpu,
                    "pareto_xmax":  self.args.xmax_cpu,
                    "pareto_mean":  self.args.mean_cpu,
                },
                "mem_params": {
                    "pareto_alpha": self.alpha_mem,
                    "pareto_xmin":  self.args.xmin_mem,
                    "pareto_xmax":  self.args.xmax_mem,
                    "pareto_mean":  self.args.mean_mem,
                },
                "arrival_params": {
                    "pareto_alpha": self.alpha_arrival,
                    "pareto_xmin":  self.args.xmin_arrival,
                    "pareto_xmax":  self.args.xmax_arrival,
                    "pareto_mean":  self.args.mean_arrival,
                },
                "lifetime_params": {
                    "pareto_alpha": self.alpha_life,
                    "pareto_xmin":  self.args.xmin_life,
                    "pareto_xmax":  self.args.xmax_life,
                    "pareto_mean":  self.args.mean_life,
                },
                "priority_params": {
                    "priority_min": self.args.priority_min,
                    "priority_max": self.args.priority_max,
                    "ratio":        self.args.priority_ratio,
                },
                "replica_params": {
                    "replicas_min": self.args.replicas_min,
                    "replicas_max": self.args.replicas_max,
                    "ratio":        self.args.replicas_ratio,
                },
            },
            "pods": [asdict(p) for p in self.pods],
        }
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        LOG.info("wrote trace JSON to %s", self.trace_path)

    ##############################################
    # ------------ Runner ------------------------
    ##############################################
    def run(self) -> None:
        self._generate_trace()
        self._write_trace()
        self._summarize_utilization()
        self._plot_trace_utilization()
        self._plot_generated_histograms()

###############################################
# ------------ Main entry point ---------------
###############################################
def main() -> None:
    if os.getenv("TRACE_GENERATOR_NOOP") == "1":
        return

    args = build_arg_parser().parse_args()
    round_float_args(args, ndigits=MAX_DECIMALS)
    setup_logging(name=LOGGER_NAME, prefix="[trace-generator] ", level=args.log_level)
    generator = TraceGenerator(args)
    generator.run()
    LOG.info("done.")

if __name__ == "__main__":
    main()
