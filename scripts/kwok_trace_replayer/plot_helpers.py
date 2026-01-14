#!/usr/bin/env python3
# scripts/kwok_trace_replayer/plot_helpers.py

from __future__ import annotations

import math
from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt

from scripts.kwok_trace_replayer.trace_helpers import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    estimate_pareto_params,
    TraceRecord,
)


# -----------------------------------------------------------------------------
# Shared time scaling (align with trace_generator)
# -----------------------------------------------------------------------------
def choose_time_unit(max_time_s: float) -> tuple[float, str]:
    """
    Match trace_generator behavior:
      <= 7h   -> minutes
      <= 7d   -> hours
      else    -> days

    Returns (scale, label) where x_plot = x_seconds * scale.
    """
    if max_time_s <= 7 * 3600:
        return 1.0 / 60.0, "time (minutes)"
    if max_time_s <= 7 * 24 * 3600:
        return 1.0 / 3600.0, "time (hours)"
    return 1.0 / (24.0 * 3600.0), "time (days)"


# -----------------------------------------------------------------------------
# Utilization time series
# -----------------------------------------------------------------------------
def plot_utilization_and_num_pods(
    *,
    times: List[float],
    u_req_hist: List[float],
    pods_hist: List[int],
    all_pods: List[TraceRecord],
    initial_pods_count: int,
    out_path: str,
    show_plots: bool = False,
    logger=None,
) -> None:
    """Plot effective utilization (max(cpu, mem)) and pod-count time series."""
    if not times:
        return

    t_s = np.asarray(times, dtype=float)
    u = np.asarray(u_req_hist, dtype=float)
    pods = np.asarray(pods_hist, dtype=float)

    max_time_s = float(np.nanmax(t_s)) if t_s.size else 0.0
    x_scale, x_label = choose_time_unit(max_time_s)

    x = t_s * x_scale  # x-axis is now truly in minutes/hours/days (aligned)
    max_x = float(np.nanmax(x)) if x.size else 0.0

    fig, ax1 = plt.subplots(figsize=(9, 4))

    # Pick two distinct colors from the active matplotlib cycle (theme-friendly)
    cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = cycle.by_key().get("color", []) if cycle is not None else []
    c_util = colors[0] if len(colors) > 0 else "C0"
    c_pods = colors[1] if len(colors) > 1 else "C1"

    l1, = ax1.plot(x, u, label="effective utilization, max(cpu, mem)", linewidth=0.9, color=c_util)
    ax1.set_xlabel(x_label, labelpad=20)
    ax1.set_ylabel("effective utilization, max(cpu, mem)", color=c_util)
    ax1.tick_params(axis="y", colors=c_util)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, pods, label="number of pods", linewidth=0.9, color=c_pods)
    ax2.set_ylabel("number of pods", color=c_pods)
    ax2.tick_params(axis="y", colors=c_pods)

    ax1.legend([l1, l2], ["effective utilization, max(cpu, mem)", "number of pods"], loc="lower right", frameon=False)

    # Build event stream in seconds (for counting), but render annotations in x-axis units.
    events: List[tuple[float, str]] = []
    for p in all_pods:
        events.append((float(p.start_time), "C"))
        events.append((float(p.end_time), "D"))
    events.sort(key=lambda e: e[0])

    # Annotate initial snapshot at time 0
    ax1.text(
        0.0,
        -0.10,
        f"+{int(initial_pods_count)}",
        transform=ax1.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
    )

    # Annotate per-tick creation/deletion counts.
    xticks_x = [v for v in ax1.get_xticks() if 0.0 <= v <= max_x]
    if events and xticks_x:
        idx = 0
        n_events = len(events)
        for i, tick_x in enumerate(xticks_x):
            left_x = 0.0 if i == 0 else xticks_x[i - 1]
            right_x = tick_x

            # Convert tick window back to seconds for comparisons
            left_s = left_x / x_scale if x_scale > 0 else 0.0
            right_s = right_x / x_scale if x_scale > 0 else 0.0

            c_count = 0
            d_count = 0

            while idx < n_events and events[idx][0] <= right_s:
                t_ev_s, kind = events[idx]
                idx += 1
                if t_ev_s > left_s:
                    if kind == "C":
                        c_count += 1
                    else:
                        d_count += 1

            if c_count or d_count:
                ax1.text(
                    tick_x,
                    -0.10,
                    f"+{c_count} -{d_count}",
                    transform=ax1.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, bbox_inches="tight")

    if show_plots:
        plt.show()
    plt.close(fig)

    if logger is not None:
        logger.info("saved utilization plot to %s", out_path)


# -----------------------------------------------------------------------------
# Generator histograms
# -----------------------------------------------------------------------------
def plot_generator_histograms(
    *,
    all_pods: List[TraceRecord],
    out_path: str,
    alpha_arrival: float,
    xmin_arrival: float,
    xmax_arrival: float | None,
    alpha_life: float,
    xmin_life: float,
    xmax_life: float | None,
    alpha_cpu: float,
    xmin_cpu: float,
    xmax_cpu: float | None,
    alpha_mem: float,
    xmin_mem: float,
    xmax_mem: float | None,
    priority_ratio: float,
    priority_min: int,
    priority_max: int,
    replicas_ratio: float,
    replicas_min: int,
    replicas_max: int,
    show_plots: bool = False,
    logger=None,
) -> None:
    """Plot generation histograms and save to disk."""
    if not all_pods:
        return

    # Inter-arrivals should ignore initial snapshot (many pods at t=0).
    trace_pods = [p for p in all_pods if float(p.start_time) > 0.0]
    trace_pods = sorted(trace_pods, key=lambda p: p.start_time)

    start_times = np.array([p.start_time for p in trace_pods], dtype=float)
    inter_arr = np.empty_like(start_times)
    if start_times.size:
        inter_arr[0] = start_times[0]
        if start_times.size > 1:
            inter_arr[1:] = np.diff(start_times)

    pods_sorted = sorted(all_pods, key=lambda p: p.start_time)
    req_vals = np.array([p.cpu for p in pods_sorted], dtype=float)
    lifetimes = np.array([p.end_time - p.start_time for p in pods_sorted], dtype=float)
    prios = np.array([p.priority for p in pods_sorted], dtype=int)
    reps = np.array([p.replicas for p in pods_sorted], dtype=int)

    fig, axes = plt.subplots(6, 1, figsize=(6, 10))
    axes = axes.flatten()

    plot_histogram_with_pareto(
        axes[0],
        inter_arr,
        title="Inter-arrival times",
        x_label="Δt (seconds)",
        y_label="probability density",
        bins=80,
        log_y=True,
        x_max=xmax_arrival,
        scale=1.0,
        pareto_fit=True,
        pareto_alpha=float(alpha_arrival),
        pareto_xmin=float(xmin_arrival),
        pareto_xmax=float(xmax_arrival) if xmax_arrival is not None else None,
    )
    plot_histogram_with_pareto(
        axes[1],
        lifetimes,
        title="Lifetimes",
        x_label="lifetime (seconds)",
        y_label="probability density",
        bins=80,
        log_y=True,
        x_max=xmax_life,
        scale=1.0,
        pareto_fit=True,
        pareto_alpha=float(alpha_life),
        pareto_xmin=float(xmin_life),
        pareto_xmax=float(xmax_life) if xmax_life is not None else None,
    )
    plot_histogram_with_pareto(
        axes[2],
        req_vals,
        title="CPU requests",
        x_label="request (fraction of node capacity)",
        y_label="probability density",
        bins=80,
        log_y=True,
        x_max=xmax_cpu,
        scale=1.0,
        pareto_fit=True,
        pareto_alpha=float(alpha_cpu),
        pareto_xmin=float(xmin_cpu),
        pareto_xmax=float(xmax_cpu) if xmax_cpu is not None else None,
    )
    plot_histogram_with_pareto(
        axes[3],
        req_vals,
        title="Memory requests",
        x_label="request (fraction of node capacity)",
        y_label="probability density",
        bins=80,
        log_y=True,
        x_max=xmax_mem,
        scale=1.0,
        pareto_fit=True,
        pareto_alpha=float(alpha_mem),
        pareto_xmin=float(xmin_mem),
        pareto_xmax=float(xmax_mem) if xmax_mem is not None else None,
    )
    plot_bar_with_geometric(
        axes[4],
        prios,
        title="Priorities",
        x_label="priority",
        y_label="probability mass",
        geom_fit=True,
        geom_ratio=float(priority_ratio),
        x_min=int(priority_min),
        x_max=int(priority_max),
    )
    plot_bar_with_geometric(
        axes[5],
        reps,
        title="Replicas",
        x_label="replicas", 
        y_label="probability mass",
        geom_fit=True,
        geom_ratio=float(replicas_ratio),
        x_min=int(replicas_min),
        x_max=int(replicas_max),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")

    if show_plots:
        plt.show()
    plt.close(fig)

    if logger is not None:
        logger.info("saved generated histograms to %s", out_path)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _bounded_pareto_pdf(x: np.ndarray, *, alpha: float, x_min: float, x_max: float) -> np.ndarray:
    """
    Bounded Pareto PDF on [x_min, x_max]:
      f(x) = (alpha * x_min^alpha / x^(alpha+1)) / (1 - (x_min/x_max)^alpha)
    """
    if not (alpha > 0 and x_min > 0 and x_max > x_min):
        return np.zeros_like(x, dtype=float)
    denom = 1.0 - (x_min / x_max) ** alpha
    denom = max(denom, 1e-300)
    return (alpha * (x_min ** alpha) / (x ** (alpha + 1.0))) / denom


def _pareto_pdf(x: np.ndarray, *, alpha: float, x_min: float) -> np.ndarray:
    """Unbounded Pareto PDF for x >= x_min."""
    if not (alpha > 0 and x_min > 0):
        return np.zeros_like(x, dtype=float)
    return alpha * (x_min ** alpha) / (x ** (alpha + 1.0))


def plot_histogram_with_pareto(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
    bins: int,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    log_y: bool = False,
    scale: float = 1.0,
    pareto_fit: bool = False,
    pareto_alpha: float | None = None,
    pareto_xmin: float | None = None,
    pareto_xmax: float | None = None,
) -> None:
    """
    Plot to an existing axes:
      - Histogram as probability density (area ≈ 1)
      - Optional (bounded) Pareto PDF overlay
      - Sample mean line
    """
    data_scaled = np.asarray(data, dtype=float) * float(scale)

    finite_mask = np.isfinite(data_scaled)
    data_scaled = data_scaled[finite_mask]

    if data_scaled.size == 0:
        ax.set_axis_off()
        return

    mean_val = float(np.mean(data_scaled))

    if x_max is not None and math.isfinite(float(x_max)):
        xm = float(x_max)
        data_for_hist = data_scaled[data_scaled <= xm]
        if data_for_hist.size == 0:
            data_for_hist = data_scaled
    else:
        data_for_hist = data_scaled

    n_points = data_for_hist.size
    n_unique = np.unique(data_for_hist).size
    max_bins_allowed = max(1, min(n_points, n_unique))
    bins_eff = min(int(bins), max_bins_allowed)

    ax.hist(data_for_hist, bins=bins_eff, density=True)

    plot_min = float(np.min(data_for_hist))
    plot_max = float(np.max(data_for_hist))

    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    if pareto_fit:
        if pareto_alpha is None or pareto_xmin is None:
            pareto_alpha, pareto_xmin = estimate_pareto_params(data_scaled[data_scaled > 0.0])

        a = float(pareto_alpha)
        xm = float(pareto_xmin)

        # If pareto_xmax is provided, use bounded Pareto overlay.
        xM = None
        if pareto_xmax is not None and math.isfinite(float(pareto_xmax)):
            xM = float(pareto_xmax)
            if not (xM > xm):
                xM = None

        lo = max(plot_min, xm)
        hi = plot_max
        if xM is not None:
            hi = min(hi, xM)

        if hi > lo and math.isfinite(lo) and math.isfinite(hi):
            x_fit = np.linspace(lo, hi, 400)
            if xM is not None:
                y_fit = _bounded_pareto_pdf(x_fit, alpha=a, x_min=xm, x_max=xM)
                label = rf"bounded Pareto: $\alpha={a:.3f}$, $x_{{\min}}={xm:.3g}$, $x_{{\max}}={xM:.3g}$"
            else:
                y_fit = _pareto_pdf(x_fit, alpha=a, x_min=xm)
                label = rf"Pareto: $\alpha={a:.3f}$, $x_{{\min}}={xm:.3g}$"

            line, = ax.plot(x_fit, y_fit, linewidth=1.5, linestyle="-")
            legend_handles.append(line)
            legend_labels.append(label)

    if plot_min <= mean_val <= plot_max:
        mean_line = ax.axvline(mean_val, linestyle="--", alpha=0.8)
        legend_handles.append(mean_line)
        legend_labels.append(f"mean={mean_val:.3f}")

    if log_y:
        ax.set_yscale("log")
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=LEGEND_FONTSIZE)


def plot_bar_with_geometric(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
    geom_fit: bool = False,
    geom_ratio: float | None = None,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    log_y: bool = False,
) -> None:
    """
    Plot a discrete distribution as a bar chart (probability mass),
    with optional geometric(-like) overlay.

    Priorities are treated as categorical buckets:
      - one bar per observed value (within [x_min, x_max] if given)
      - equally spaced buckets; numeric distance does not affect spacing
      - x-ticks are the true values.
    """
    data_int = np.asarray(data, dtype=int)
    finite_mask = np.isfinite(data_int)
    data_int = data_int[finite_mask]

    if data_int.size == 0:
        ax.set_axis_off()
        return

    if x_min is not None:
        data_int = data_int[data_int >= int(x_min)]
    if x_max is not None:
        data_int = data_int[data_int <= int(x_max)]

    if data_int.size == 0:
        ax.set_axis_off()
        return

    unique_vals = np.sort(np.unique(data_int))
    n_vals = unique_vals.size
    positions = np.arange(n_vals, dtype=float)

    counts = np.array([np.sum(data_int == v) for v in unique_vals], dtype=float)
    total = counts.sum()
    probs_emp = counts / total if total > 0.0 else np.zeros_like(counts)

    ax.bar(positions, probs_emp, width=0.8, align="center")

    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    if geom_fit and geom_ratio is not None and float(geom_ratio) > 0.0:
        r = float(geom_ratio)
        k0 = int(unique_vals[0])
        exponents = (unique_vals - k0).astype(float)

        if np.isclose(r, 1.0):
            weights = np.ones_like(exponents)
        else:
            weights = r ** exponents

        probs_theo = weights / weights.sum()

        line, = ax.plot(positions, probs_theo, linestyle="-", linewidth=1.0)
        legend_handles.append(line)
        legend_labels.append(
            r"geometric: "
            rf"$r={r:.3f}$, "
            rf"$k\in[{int(unique_vals[0])},{int(unique_vals[-1])}]$"
        )

    if log_y:
        ax.set_yscale("log")
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    ax.set_xlim(-0.5, n_vals - 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_vals, rotation=90, ha="center", fontsize=TICK_LABEL_FONTSIZE)

    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=LEGEND_FONTSIZE)
