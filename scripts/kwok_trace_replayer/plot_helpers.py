#!/usr/bin/env python3
# plot_helpers.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List

from scripts.kwok_trace_replayer.trace_helpers import (
    MC_MEAN_SEED, MC_MEAN_SAMPLES,
    AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE,
    TITLE_FONTSIZE, LEGEND_FONTSIZE,
    estimate_pareto_params,
    TraceRecord,
)


def plot_utilization_time_series(
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
    """Plot utilization and pod-count time series and save to disk."""
    if not times:
        return

    t = np.asarray(times, dtype=float)
    u = np.asarray(u_req_hist, dtype=float)
    pods = np.asarray(pods_hist, dtype=float)

    max_time = float(np.max(t)) if len(t) else 0.0
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

    events: List[tuple[float, str]] = []
    for p in all_pods:
        events.append((float(p.start_time), "C"))
        events.append((float(p.end_time), "D"))
    events.sort(key=lambda e: e[0])

    ax1.text(
        0.0,
        -0.10,
        f"+{int(initial_pods_count)}",
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
    plt.savefig(out_path)
    if show_plots:
        plt.show()
    plt.close(fig)
    if logger is not None:
        logger.info("saved utilization plot to %s", out_path)


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
    alpha_req: float,
    xmin_req: float,
    xmax_req: float | None,
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
        x_max=xmax_arrival,
        scale=1.0,
        pareto_fit=True,
        pareto_alpha=float(alpha_arrival),
        pareto_xmin=float(xmin_arrival),
    )
    plot_histogram_with_pareto(
        axes[1],
        lifetimes,
        title="Lifetimes (all records)",
        x_label="Lifetime (seconds)",
        y_label="Probability density",
        bins=80,
        log_y=True,
        x_max=xmax_life,
        scale=1.0,
        pareto_fit=True,
        pareto_alpha=float(alpha_life),
        pareto_xmin=float(xmin_life),
    )
    plot_histogram_with_pareto(
        axes[2],
        req_vals,
        title="Requests (CPU = MEM)",
        x_label="Request (fraction of node capacity)",
        y_label="Probability density",
        bins=80,
        log_y=True,
        x_max=xmax_req,
        scale=1.0,
        pareto_fit=True,
        pareto_alpha=float(alpha_req),
        pareto_xmin=float(xmin_req),
    )
    plot_bar_with_geometric(
        axes[3],
        prios,
        title="Priorities",
        x_label="Priority",
        y_label="Probability mass",
        geom_fit=True,
        geom_ratio=float(priority_ratio),
        x_min=int(priority_min),
        x_max=int(priority_max),
    )
    plot_bar_with_geometric(
        axes[4],
        reps,
        title="Replicas",
        x_label="Replicas",
        y_label="Probability mass",
        geom_fit=True,
        geom_ratio=float(replicas_ratio),
        x_min=int(replicas_min),
        x_max=int(replicas_max),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    if logger is not None:
        logger.info("saved generated histograms to %s", out_path)

# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

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
) -> None:
    """
    Plot to an existing axes:
    - Histogram as probability density (area ≈ 1)
    - Optional Pareto PDF
    - Sample mean line
    """
    # Scale data
    data_scaled = np.asarray(data, dtype=float) * scale

    # Drop NaN and non-finite values (inf, -inf)
    finite_mask = np.isfinite(data_scaled)
    data_scaled = data_scaled[finite_mask]

    # Sample mean over all data
    mean_val = float(np.mean(data_scaled))

    # Crop data to x_max for histogram and pareto fitting
    if x_max is not None:
        data_for_hist = data_scaled[data_scaled <= x_max]
        if data_for_hist.size == 0:
            # If everything is above x_max, fall back to all data
            data_for_hist = data_scaled
    else:
        data_for_hist = data_scaled

    # Safe bin count
    n_points = data_for_hist.size
    n_unique = np.unique(data_for_hist).size
    max_bins_allowed = max(1, min(n_points, n_unique))
    bins_eff = min(bins, max_bins_allowed)

    # Histogram as PDF
    ax.hist(data_for_hist, bins=bins_eff, density=True)

    plot_min = float(np.min(data_for_hist))
    plot_max = float(np.max(data_for_hist))

    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    # Fix rng for MC mean so result is deterministic
    rng_mc = np.random.default_rng(MC_MEAN_SEED)

    # --- Optional Pareto curve ---
    if pareto_fit:

        # Estimate pareto parameters if not provided
        if pareto_alpha is None or pareto_xmin is None:
            pareto_alpha, pareto_xmin = estimate_pareto_params(
                data_scaled[data_scaled > 0.0]
            )

        lo = max(plot_min, pareto_xmin)
        hi = plot_max
        x_fit = np.linspace(lo, hi, 400)

        # Pareto PDF: f(x) = alpha * x_min^alpha / x^(alpha+1), x >= x_min
        pareto_pdf_vals = (
            pareto_alpha
            * (pareto_xmin ** pareto_alpha)
            / (x_fit ** (pareto_alpha + 1.0))
        )
        pareto_line, = ax.plot(x_fit, pareto_pdf_vals, linewidth=1.5, linestyle="-")

        # Compute an MC mean
        if x_max is not None:  # Truncated/clamped case, matching sample_pareto()
            u = rng_mc.random(MC_MEAN_SAMPLES)
            u_min_tail = (pareto_xmin / x_max) ** pareto_alpha  # in (0, 1)
            u_min = max(1e-12, float(u_min_tail))  # avoid zero
            u = np.clip(u, u_min, 1.0 - 1e-12)  # avoid one
            samples = pareto_xmin / (u ** (1.0 / pareto_alpha))  # Inverse CDF
            samples = np.minimum(samples, x_max)  # clamp at x_max
        else:  # Unbounded Pareto (clipping only)
            u = rng_mc.random(MC_MEAN_SAMPLES)
            u = np.clip(u, 1e-12, 1.0 - 1e-12)
            samples = pareto_xmin / (u ** (1.0 / pareto_alpha))

        mc_mean = float(samples.mean())
        mean_str = f", MC-mean={mc_mean:.2f}"

        if pareto_line is not None:
            label = (
                r"Pareto: "
                rf"$\alpha\!=\!{pareto_alpha:.3f}$, "
                rf"$x_{{\min}}\!=\!{pareto_xmin:.3f}$"
                # f"{mean_str}"
            )
            legend_handles.append(pareto_line)
            legend_labels.append(label)

    # --- Sample mean line ---
    if plot_min <= mean_val <= plot_max:
        mean_line = ax.axvline(mean_val, linestyle="--", alpha=0.8)
        legend_handles.append(mean_line)

    mean_label = f"mean={mean_val:.3f}"
    legend_labels.append(mean_label)

    # Axis scales and labels
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
    - one bar per *observed* priority value (within [x_min, x_max] if given)
    - bars are equally spaced; numeric distance between priorities does not affect spacing
    - x-ticks are the true priority values.
    """
    # Convert and clean
    data_int = np.asarray(data, dtype=int)
    finite_mask = np.isfinite(data_int)
    data_int = data_int[finite_mask]

    if data_int.size == 0:
        ax.set_axis_off()
        return

    # Optional clipping on numeric value
    if x_min is not None:
        data_int = data_int[data_int >= int(x_min)]
    if x_max is not None:
        data_int = data_int[data_int <= int(x_max)]

    if data_int.size == 0:
        ax.set_axis_off()
        return

    # Unique priority values (sorted); treat as categories
    unique_vals = np.sort(np.unique(data_int))
    n_vals = unique_vals.size

    # Positions 0..n_vals-1 (equally spaced buckets)
    positions = np.arange(n_vals, dtype=float)

    # Empirical probabilities over these buckets
    counts = np.array([np.sum(data_int == v) for v in unique_vals], dtype=float)
    total = counts.sum()
    probs_emp = counts / total if total > 0.0 else np.zeros_like(counts)

    # Bar chart (no gaps in index space)
    ax.bar(positions, probs_emp, width=0.8, align="center")

    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    # Optional geometric overlay, defined over the same observed priorities
    if geom_fit and geom_ratio is not None and geom_ratio > 0.0:
        r = float(geom_ratio)

        # Use min observed priority as origin for exponents
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
            r"Geometric: "
            rf"$r\!=\!{r:.3f}$, "
            rf"$k\in[{int(unique_vals[0])},{int(unique_vals[-1])}]$"
        )

    # Axis scales and limits
    if log_y:
        ax.set_yscale("log")
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    # X-limits: just a bit of padding around first/last bucket
    ax.set_xlim(-0.5, n_vals - 0.5)

    # X ticks: one per bar, labeled with the *true* priority value
    ax.set_xticks(positions)
    ax.set_xticklabels(
        unique_vals,
        rotation=90,
        ha="center",
        fontsize=TICK_LABEL_FONTSIZE,
    )

    # Match fontsizes to histogram helper
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=LEGEND_FONTSIZE)
