#!/usr/bin/env python3
# plot_helpers.py

import math
from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scripts.kwok_trace_replayer.trace_helpers import (
    estimate_pareto_params,
    TraceRecord,
)

# -------------------------------------------------------------------
# Plot configuration
# -------------------------------------------------------------------

PLOT_AXIS_LABEL_FONTSIZE = 5.0
PLOT_TICK_LABEL_FONTSIZE = 4.0
PLOT_LEGEND_FONTSIZE = 4.0

# -----------------------------------------------------------------------------
# Time scaling
# -----------------------------------------------------------------------------

def choose_time_unit(max_time_s: float) -> tuple[float, str]:
    """
    <= 7h   -> minutes
    <= 7d   -> hours
    else    -> days
    """
    if max_time_s <= 7 * 3600:
        return 1.0 / 60.0, "time (minutes)"
    if max_time_s <= 7 * 24 * 3600:
        return 1.0 / 3600.0, "time (hours)"
    return 1.0 / (24.0 * 3600.0), "time (days)"

# -----------------------------------------------------------------------------
# Utilization
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
    """
    Plot usage (max(cpu, mem)) and pod-count time series.
    """
    if not times:
        return

    t_s = np.asarray(times, dtype=float)
    util = np.asarray(u_req_hist, dtype=float)   # still stored as fractions, e.g. 0.90
    pods = np.asarray(pods_hist, dtype=float)

    max_time_s = float(np.nanmax(t_s)) if t_s.size else 0.0
    x_scale, x_label = choose_time_unit(max_time_s)

    x = t_s * x_scale
    max_x = float(np.nanmax(x)) if x.size else 0.0

    fig, ax1 = plt.subplots(figsize=(9, 4))

    # Pick colors
    cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = cycle.by_key().get("color", []) if cycle is not None else []
    c_util = colors[0] if len(colors) > 0 else "C0"
    c_pods = colors[1] if len(colors) > 1 else "C1"

    l1, = ax1.plot(x, util, label="usage", linewidth=0.9, color=c_util)
    ax1.set_xlabel(x_label, labelpad=20)
    ax1.set_ylabel("usage (%)", color=c_util)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax1.tick_params(axis="y", colors=c_util)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, pods, label="number of pods", linewidth=0.9, color=c_pods)
    ax2.set_ylabel("number of pods", color=c_pods)
    ax2.tick_params(axis="y", colors=c_pods)

    ax1.legend([l1, l2], ["usage", "number of pods"], loc="upper right", frameon=True)

    # Build event stream in seconds (weighted by replicas, so counts are pods)
    events: List[tuple[float, str, int]] = []
    for p in all_pods:
        w = int(getattr(p, "replicas", 1))
        events.append((float(p.start_time), "C", w))  # created
        events.append((float(p.end_time), "D", w))    # deleted
    events.sort(key=lambda e: e[0])

    # Annotate initial snapshot at time 0
    ax1.annotate(
        f"+{int(initial_pods_count)}",
        xy=(0.0, -0.10),
        xycoords=ax1.get_xaxis_transform(),
        xytext=(-10, 0),              # move left by 10 points
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color="gray",
        clip_on=False,
    )

    # Annotate per-tick creation/deletion counts.
    # Important: include the final trace boundary (max_x), because matplotlib ticks
    # often stop before the exact endpoint.
    xticks_x = sorted(float(v) for v in ax1.get_xticks() if 0.0 <= float(v) <= max_x)

    if not xticks_x:
        xticks_x = [0.0]

    # Append explicit final boundary if it is not already a tick
    atol = max(1e-9, 1e-6 * max(1.0, max_x))
    if max_x > 0.0 and not np.isclose(xticks_x[-1], max_x, atol=atol, rtol=0.0):
        xticks_x.append(max_x)

    if events and xticks_x:
        idx = 0
        n_events = len(events)
        shown_interval_labels = 0  # add this before the for-loop

        for i, tick_x in enumerate(xticks_x):
            left_x = 0.0 if i == 0 else xticks_x[i - 1]
            right_x = tick_x

            # Convert tick window back to seconds
            left_s = left_x / x_scale if x_scale > 0 else 0.0
            right_s = right_x / x_scale if x_scale > 0 else 0.0

            c_count = 0
            d_count = 0

            # Consume all events up to the right boundary
            while idx < n_events and events[idx][0] <= right_s + 1e-12:
                t_event_s, kind, weight = events[idx]
                idx += 1

                # Keep initial t=0 events out of interval labels (shown separately above)
                if t_event_s > left_s + 1e-12:
                    if kind == "C":
                        c_count += weight
                    else:
                        d_count += weight

            is_final_boundary = np.isclose(right_x, max_x, atol=atol, rtol=0.0)

            if c_count or d_count or is_final_boundary:
                # Default placement for middle labels
                x_offset_pts = 0
                ha = "center"

                # First shown interval label: nudge a bit left
                if shown_interval_labels == 0 and not is_final_boundary:
                    x_offset_pts = -6
                    ha = "center"

                # Final label: nudge a bit right
                if is_final_boundary:
                    x_offset_pts = +16
                    ha = "right"

                ax1.annotate(
                    f"+{c_count} -{d_count}",
                    xy=(tick_x, -0.10),
                    xycoords=ax1.get_xaxis_transform(),
                    xytext=(x_offset_pts, 0),
                    textcoords="offset points",
                    ha=ha,
                    va="top",
                    fontsize=8,
                    color="gray",
                    clip_on=False,
                )
                shown_interval_labels += 1

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, bbox_inches="tight")

    if show_plots:
        plt.show()
    plt.close(fig)

    if logger is not None:
        logger.info("saved utilization plot to %s", out_path)

# -----------------------------------------------------------------------------
# Histograms
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
    """
    Plot histograms
    """
    if not all_pods:
        return
    
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
    replicas = np.array([p.replicas for p in pods_sorted], dtype=int)

    fig, axes = plt.subplots(6, 1, figsize=(6, 10))
    axes = axes.flatten()

    plot_histogram_with_pareto(
        axes[0],
        inter_arr,
        x_label="inter-arrival time (seconds)",
        y_label="% of samples",
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
        x_label="lifetime (seconds)",
        y_label="% of samples",
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
        x_label="requested CPU (fraction of node capacity)",
        y_label="% of samples",
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
        x_label="requested memory (fraction of node capacity)",
        y_label="% of samples",
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
        x_label="priority",
        y_label="% of samples",
        geom_fit=True,
        geom_ratio=float(priority_ratio),
        x_min=int(priority_min),
        x_max=int(priority_max),
    )
    plot_bar_with_geometric(
        axes[5],
        replicas,
        x_label="replicas", 
        y_label="% of samples",
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

def bounded_pareto_pdf(x: np.ndarray, *, alpha: float, x_min: float, x_max: float) -> np.ndarray:
    """
    Bounded Pareto PDF on [x_min, x_max]:
      f(x) = (alpha * x_min^alpha / x^(alpha+1)) / (1 - (x_min/x_max)^alpha)
    """
    if not (alpha > 0 and x_min > 0 and x_max > x_min):
        return np.zeros_like(x, dtype=float)
    denom = 1.0 - (x_min / x_max) ** alpha
    denom = max(denom, 1e-300)
    return (alpha * (x_min ** alpha) / (x ** (alpha + 1.0))) / denom

def pareto_pdf(x: np.ndarray, *, alpha: float, x_min: float) -> np.ndarray:
    """
    Unbounded Pareto PDF for x >= x_min.
    """
    if not (alpha > 0 and x_min > 0):
        return np.zeros_like(x, dtype=float)
    return alpha * (x_min ** alpha) / (x ** (alpha + 1.0))

def plot_histogram_with_pareto(
    ax: plt.Axes,
    data: np.ndarray,
    *,
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

    weights = np.ones_like(data_for_hist) * (100.0 / data_for_hist.size)
    _, bin_edges, _ = ax.hist(data_for_hist, bins=bins_eff, weights=weights)
    bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0

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
                y_fit = bounded_pareto_pdf(x_fit, alpha=a, x_min=xm, x_max=xM) * bin_width * 100.0
                label = rf"bounded Pareto: $\alpha={a:.3f}$, $x_{{\min}}={xm:.3g}$, $x_{{\max}}={xM:.3g}$"
            else:
                y_fit = pareto_pdf(x_fit, alpha=a, x_min=xm) * bin_width * 100.0
                label = rf"Pareto: $\alpha={a:.3f}$, $x_{{\min}}={xm:.3g}$"

            line = ax.plot(x_fit, y_fit, linewidth=1.5, linestyle="-")[0]
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

    ax.set_xlabel(x_label, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=PLOT_LEGEND_FONTSIZE)

def plot_bar_with_geometric(
    ax: plt.Axes,
    data: np.ndarray,
    *,
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
    Plot a discrete distribution as a bar chart, with optional geometric(-like) overlay.
    """
    data_integers = np.asarray(data, dtype=int)
    finite_mask = np.isfinite(data_integers)
    data_integers = data_integers[finite_mask]

    if data_integers.size == 0:
        ax.set_axis_off()
        return

    if x_min is not None:
        data_integers = data_integers[data_integers >= int(x_min)]
    if x_max is not None:
        data_integers = data_integers[data_integers <= int(x_max)]

    if data_integers.size == 0:
        ax.set_axis_off()
        return

    unique_vals = np.sort(np.unique(data_integers))
    n_vals = unique_vals.size
    positions = np.arange(n_vals, dtype=float)

    counts = np.array([np.sum(data_integers == v) for v in unique_vals], dtype=float)
    total = counts.sum()
    probs_empirical = (counts / total * 100.0) if total > 0.0 else np.zeros_like(counts)

    ax.bar(positions, probs_empirical, width=0.8, align="center")

    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    if geom_fit and geom_ratio is not None and float(geom_ratio) > 0.0:
        ratio = float(geom_ratio)
        k0 = int(unique_vals[0])
        exponents = (unique_vals - k0).astype(float)

        if np.isclose(ratio, 1.0):
            weights = np.ones_like(exponents)
        else:
            weights = ratio ** exponents

        probs_theoretical = weights / weights.sum() * 100.0

        line, = ax.plot(positions, probs_theoretical, linestyle="-", linewidth=1.0)
        legend_handles.append(line)
        legend_labels.append(
            r"geometric: "
            rf"$r={ratio:.3f}$, "
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
    ax.set_xticklabels(unique_vals, rotation=90, ha="center", fontsize=PLOT_TICK_LABEL_FONTSIZE)

    ax.set_xlabel(x_label, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=PLOT_TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=PLOT_LEGEND_FONTSIZE)
