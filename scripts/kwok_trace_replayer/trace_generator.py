#!/usr/bin/env python3
# trace_generator.py

"""
python -m scripts.kwok_trace_replayer.trace_generator \
--output-dir ./output-traces \
--seed 12345 \
--log-level INFO \
--num-nodes 8 \
--trace-time 1h \
--target-util 0.95 \
--xmin-arrival 0.01 \
--xmax-arrival 30.0 \
--mean-arrival 10.0 \
--xmin-life 10.0 \
--xmax-life 3600.0 \
--xmin-req 0.01 \
--xmax-req 0.5 \
--mean-req 0.1 \
--priority-min 1 \
--priority-max 3 \
--priority-ratio 0.9 \
--replicas-min 1 \
--replicas-max 3 \
--replicas-ratio 0.8 \
--util-tol 0.01 \
--calib-max-iter 100 \
--initial-fill-tol 0.002
"""

import argparse, heapq, json, logging, os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from scripts.helpers.general_helpers import (
    parse_duration_to_seconds,
    setup_logging,
    build_cli_cmd,
    write_info_file,
    log_args_block,
)
from scripts.kwok_trace_replayer.trace_helpers import TraceRecord
from scripts.kwok_trace_replayer.plot_helpers import (
    plot_histogram_with_pareto,
    plot_bar_with_geometric,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SOLVE_ALPHA_MAX_ITERATIONS = 10_000
SOLVE_ALPHA_TOLERANCE = 1e-3
SOLVE_ALPHA_SAMPLES = 50_000
SOLVE_ALPHA_LOWER_BOUND = 0.1
SOLVE_ALPHA_UPPER_BOUND = 10.0

MAX_DECIMALS = 6

LOGGER_NAME = "trace-generator"
LOG = logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------------
# Small state models
# ---------------------------------------------------------------------
@dataclass
class ClusterState:
    num_nodes: int
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
    p = argparse.ArgumentParser(description="Generate initial + trace workload JSON files.")

    # General
    p.add_argument("--output-dir", dest="output_dir", default="./output-traces")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", dest="log_level", default="INFO")

    # Cluster / horizon
    p.add_argument("--num-nodes", type=int, default=8)
    p.add_argument("--trace-time", type=str, default="3600s")
    p.add_argument("--target-util", type=float, required=True)

    # Inter-arrival (Pareto)
    p.add_argument("--xmin-arrival", type=float, default=0.01)
    p.add_argument("--xmax-arrival", type=float, default=None)
    p.add_argument("--mean-arrival", type=float, required=True)

    # Lifetime bounds (mean is inferred from util)
    p.add_argument("--xmin-life", type=float, default=10.0)
    p.add_argument("--xmax-life", type=float, default=None)

    # Requests (Pareto)
    p.add_argument("--xmin-req", type=float, default=0.01)
    p.add_argument("--xmax-req", type=float, default=1.0)
    p.add_argument("--mean-req", type=float, required=True)

    # Priority + replicas (geometric/uniform)
    p.add_argument("--priority-min", type=int, default=1)
    p.add_argument("--priority-max", type=int, default=3)
    p.add_argument("--priority-ratio", type=float, default=1.0)

    p.add_argument("--replicas-min", type=int, default=1)
    p.add_argument("--replicas-max", type=int, default=1)
    p.add_argument("--replicas-ratio", type=float, default=1.0)

    # Calibration controls
    p.add_argument(
        "--util-tol",
        type=float,
        default=0.01,
        help="Relative tolerance for hitting target utilization (default: 0.01 = 1%).",
    )
    p.add_argument("--calib-max-iter", type=int, default=200)

    # Initial snapshot size control (optional)
    p.add_argument(
        "--initial-fill-tol",
        type=float,
        default=0.002,
        help="How close initial snapshot util should be to target (fractional, default 0.002).",
    )
    p.add_argument(
        "--initial-max-pods",
        type=int,
        default=200_000,
        help="Safety cap when building initial snapshot (default 200k).",
    )

    return p


def round_float_args(args: argparse.Namespace, ndigits: int) -> None:
    for k, v in vars(args).items():
        if isinstance(v, float):
            setattr(args, k, round(v, ndigits))


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------
class TraceGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
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

        # For info yaml
        self._last_stats: Dict[str, float] = {}
        self._prepared = False

    # -------------------------
    # Logging / metadata
    # -------------------------
    def log_args(self) -> None:
        include = [
            "output_dir",
            "seed",
            "log_level",
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
            "util_tol",
            "calib_max_iter",
            "initial_fill_tol",
            "initial_max_pods",
        ]
        log_args_block(LOG, self.args, title="ARGS", include=include)

    def _write_info_file(self, extra: Dict[str, object]) -> None:
        """
        Single place for *all* generation metadata (no meta inside JSON files).
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
    def _expected_value(vals: np.ndarray, probs: Optional[np.ndarray]) -> float:
        return float(np.mean(vals)) if probs is None else float(np.sum(vals.astype(float) * probs.astype(float)))

    # -------------------------
    # Mean life inference
    # -------------------------
    def _infer_mean_life_from_target_util(self) -> float:
        mean_arrival = float(self.args.mean_arrival)
        if mean_arrival <= 0:
            raise ValueError("mean-arrival must be > 0.")
        lam = 1.0 / mean_arrival

        r_vals, r_probs = self._build_geometric_support(
            max(1, int(self.args.replicas_min)),
            max(1, int(self.args.replicas_max)),
            float(self.args.replicas_ratio),
        )
        e_rep = self._expected_value(r_vals, r_probs)

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

    # -------------------------
    # Fit alphas
    # -------------------------
    def _fit_alphas(self, rng: np.random.Generator) -> None:
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
    # Util metric (single truth)
    # -------------------------
    def _time_avg_req_util(self, pods: List[TraceRecord]) -> float:
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
    # Initial snapshot (steady state)
    # -------------------------
    def _build_initial_snapshot(
        self,
        rng: np.random.Generator,
        prio_vals: np.ndarray,
        prio_probs: Optional[np.ndarray],
        rep_vals: np.ndarray,
        rep_probs: Optional[np.ndarray],
        next_id: int,
    ) -> Tuple[List[TraceRecord], int]:
        assert self.alpha_req is not None and self.alpha_life is not None

        target_req_total = float(self.args.target_util) * float(self.args.num_nodes)
        tol = float(self.args.initial_fill_tol)
        max_pods = int(self.args.initial_max_pods)

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
        Generate trace pods with start_time >= 0, while tracking *live* state
        that includes the initial snapshot baseline at t=0 (for plotting only).
        """
        assert self.alpha_req is not None and self.alpha_arrival is not None and self.alpha_life is not None

        state = ClusterState(num_nodes=int(self.args.num_nodes))
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
    # Prep
    # -------------------------
    def _ensure_prepared(self) -> None:
        if self._prepared:
            return

        self.args.mean_life = self._infer_mean_life_from_target_util()
        LOG.info(
            "inferred mean_life=%.3fs from target-util=%.3f",
            float(self.args.mean_life),
            float(self.args.target_util),
        )
        self._prepared = True

    # -------------------------
    # Calibration loop
    # -------------------------
    def _generate_all_once(self, iter_seed: int) -> Tuple[List[TraceRecord], List[TraceRecord], Dict[str, float], Dict[str, object]]:
        rng = np.random.default_rng(iter_seed)

        self._fit_alphas(rng)

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

        initial_pods, next_id = self._build_initial_snapshot(
            rng,
            prio_vals,
            prio_probs,
            rep_vals,
            rep_probs,
            next_id=next_id,
        )
        trace_pods, next_id, times, u_hist, pods_hist = self._generate_trace_events(
            rng,
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
            "measured_util_time_avg": float(util),
            "target_util_time_avg": float(self.args.target_util),
            "util_tol": float(self.args.util_tol),
            "calib_max_iter": int(self.args.calib_max_iter),
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

    def _calibrate_mean_life(self) -> Tuple[List[TraceRecord], List[TraceRecord], Dict[str, object]]:
        target = float(self.args.target_util)
        tol = float(self.args.util_tol)
        max_iter = int(self.args.calib_max_iter)

        best_initial: List[TraceRecord] = []
        best_trace: List[TraceRecord] = []
        best_err = float("inf")
        best_extra: Dict[str, object] = {}

        for it in range(1, max_iter + 1):
            iter_seed = self.base_seed + 10_000 * it
            initial_pods, trace_pods, stats, extra_info = self._generate_all_once(iter_seed)

            measured = float(stats["util_time_avg"])
            err = abs(measured - target) / max(1e-12, target)
            best_err = min(best_err, err)

            LOG.info(
                "[calibrate] iter=%d measured=%.3f target=%.3f rel_err=%.2f%% (initial=%d trace=%d)",
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
                raise RuntimeError("Measured utilization is ~0; cannot calibrate mean_life.")

            new_mean_life = float(self.args.mean_life) * (target / measured)

            xmin = float(self.args.xmin_life)
            xmax = self.args.xmax_life
            if new_mean_life <= xmin:
                new_mean_life = xmin * 1.001
            if xmax is not None and new_mean_life >= float(xmax):
                new_mean_life = float(xmax) * 0.999

            self.args.mean_life = float(new_mean_life)
            LOG.info("[calibrate] updated mean_life=%.3fs", float(self.args.mean_life))

        LOG.warning("calibration did not reach util-tol; best rel_err=%.2f%%", 100.0 * best_err)
        return best_initial, best_trace, best_extra

    # -------------------------
    # Output writers
    # -------------------------
    def _write_json(self, path: Path, pods: List[TraceRecord]) -> None:
        # NO META in JSON anymore (kept in info_generate.yaml).
        obj = {"pods": [asdict(p) for p in pods]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        LOG.info("wrote %s (%d records)", path, len(pods))

    def _write_outputs(self, initial_pods: List[TraceRecord], trace_pods: List[TraceRecord], extra_info: Dict[str, object]) -> None:
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
    # Plots
    # -------------------------
    def _plot_utilization(self, all_pods: List[TraceRecord]) -> None:
        if not self.times:
            return

        t = np.asarray(self.times, dtype=float)
        u = np.asarray(self.u_req_hist, dtype=float)
        pods = np.asarray(self.pods_hist, dtype=float)

        max_time = float(max(t)) if len(t) else 0.0
        if max_time <= 7 * 3600:
            x_scale, x_label = (1.0 / 60.0), "Time (minutes)"
        elif max_time <= 7 * 24 * 3600:
            x_scale, x_label = (1.0 / 3600.0), "Time (hours)"
        else:
            x_scale, x_label = (1.0 / (24.0 * 3600.0)), "Time (days)"

        fig, ax1 = plt.subplots(figsize=(9, 4))

        ax1.plot(t, u, label="Utilization", color="tab:blue", linewidth=0.8)
        ax1.set_xlabel(x_label, labelpad=20)
        ax1.set_ylabel("Utilization (fraction of total capacity)")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = ax1.twinx()
        ax2.plot(t, pods, label="Number of pods", color="tab:orange", linewidth=0.8)
        ax2.set_ylabel("Number of pods")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        xticks = [x for x in ax1.get_xticks() if 0.0 <= x <= max_time]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f"{x * x_scale:.0f}" for x in xticks])

        events: List[Tuple[float, str]] = []
        for p in all_pods:
            events.append((float(p.start_time), "C"))
            events.append((float(p.end_time), "D"))
        events.sort(key=lambda e: e[0])

        ax1.text(
            0.0,
            -0.10,
            f"+{int(self.initial_pods_count)}",
            transform=ax1.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
        )

        if events and xticks:
            idx = 0
            n_events = len(events)
            for i, tick in enumerate(xticks):
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
                        -0.10,
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

    def _plot_histograms(self, all_pods: List[TraceRecord]) -> None:
        if not all_pods:
            return

        pods_sorted = sorted(all_pods, key=lambda p: p.start_time)
        start_times = np.array([p.start_time for p in pods_sorted], dtype=float)
        req_vals = np.array([p.cpu for p in pods_sorted], dtype=float)
        lifetimes = np.array([p.end_time - p.start_time for p in pods_sorted], dtype=float)
        prios = np.array([p.priority for p in pods_sorted], dtype=int)
        reps = np.array([p.replicas for p in pods_sorted], dtype=int)

        inter_arr = np.empty_like(start_times)
        if len(start_times) > 0:
            inter_arr[0] = start_times[0]
        if len(start_times) > 1:
            inter_arr[1:] = np.diff(start_times)

        fig, axes = plt.subplots(5, 1, figsize=(6, 10))
        axes = axes.flatten()

        plot_histogram_with_pareto(
            axes[0],
            inter_arr,
            title="Inter-arrival times (all records)",
            x_label="Δt (seconds)",
            y_label="Probability density",
            bins=80,
            log_y=True,
            x_max=self.args.xmax_arrival,
            scale=1.0,
            pareto_fit=True,
            pareto_alpha=float(self.args.alpha_arrival),
            pareto_xmin=float(self.args.xmin_arrival),
        )
        plot_histogram_with_pareto(
            axes[1],
            lifetimes,
            title="Lifetimes (all records)",
            x_label="Lifetime (seconds)",
            y_label="Probability density",
            bins=80,
            log_y=True,
            x_max=self.args.xmax_life,
            scale=1.0,
            pareto_fit=True,
            pareto_alpha=float(self.args.alpha_life),
            pareto_xmin=float(self.args.xmin_life),
        )
        plot_histogram_with_pareto(
            axes[2],
            req_vals,
            title="Requests (CPU = MEM)",
            x_label="Request (fraction of node capacity)",
            y_label="Probability density",
            bins=80,
            log_y=True,
            x_max=self.args.xmax_req,
            scale=1.0,
            pareto_fit=True,
            pareto_alpha=float(self.args.alpha_req),
            pareto_xmin=float(self.args.xmin_req),
        )
        plot_bar_with_geometric(
            axes[3],
            prios,
            title="Priorities",
            x_label="Priority",
            y_label="Probability mass",
            geom_fit=True,
            geom_ratio=float(self.args.priority_ratio),
            x_min=int(self.args.priority_min),
            x_max=int(self.args.priority_max),
        )
        plot_bar_with_geometric(
            axes[4],
            reps,
            title="Replicas",
            x_label="Replicas",
            y_label="Probability mass",
            geom_fit=True,
            geom_ratio=float(self.args.replicas_ratio),
            x_min=int(self.args.replicas_min),
            x_max=int(self.args.replicas_max),
        )

        fig.tight_layout()
        fig.savefig(self.hist_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOG.info("saved generated histograms to %s", self.hist_plot_path)

    # -------------------------
    # Runner
    # -------------------------
    def run(self) -> None:
        self._ensure_prepared()
        self.log_args()

        initial_pods, trace_pods, extra_info = self._calibrate_mean_life()

        all_pods = initial_pods + trace_pods
        self._write_outputs(initial_pods, trace_pods, extra_info)

        self._plot_utilization(all_pods)
        self._plot_histograms(all_pods)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    if os.getenv("TRACE_GENERATOR_NOOP") == "1":
        return
    args = build_arg_parser().parse_args()
    round_float_args(args, MAX_DECIMALS)
    setup_logging(name=LOGGER_NAME, prefix="[trace-generator] ", level=args.log_level)

    gen = TraceGenerator(args)
    gen.run()
    LOG.info("done.")


if __name__ == "__main__":
    main()
