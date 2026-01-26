#!/usr/bin/env python3
"""
Fit candidate distributions to columns in a CSV using MLE (SciPy) and rank via AIC/BIC,
optionally producing overlay plots (histogram + fitted PDFs), and optionally sweeping xmin
candidates (percentiles) for ALL distributions.

Example:
  python -m scripts.public_trace_analysis.fit_distributions \
    --csv data.csv \
    --columns inter_arrival_us life_time_us cpu_request mem_request \
    --criterion aic \
    --out fit_results.csv \
    --xmin-qs 0 0.9 0.95 0.97 0.99 0.995 \
    --plot-dir fit_plots

Notes:
- With huge n, p-values are often "always significant"; use AIC/BIC for ranking.
- We filter to x>0 (positive support).
- xmin sweep:
    For each column, we take a subsample x_eval (size --max-n-eval), compute xmin as quantile(x_eval, q),
    then fit each distribution on x_tail = {x_eval >= xmin}.

    Tail support handling:
      * expon/lognorm/gamma/weibull_min: fit with floc=xmin so support starts at xmin.
      * pareto (Type I): SciPy's support is x >= loc + scale, so we enforce xmin by fitting with
            floc=0 and fscale=xmin   (so loc+scale = xmin) and estimate shape only.

- Because n differs across xmin_q, AIC/BIC are not directly comparable across different xmin_q.
  Use avg_nll as the more meaningful comparison across different tail sizes.

- Plotting:
    For each column, overlays one curve per distribution: the best (lowest avg_nll) across xmin_q.
    Uses histogram-based y-limits to stay robust against Pareto spikes near xmin.
"""

import argparse
import ast
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError as e:
    raise SystemExit("This script requires scipy. Install with: pip install scipy") from e

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------
# Candidate distributions (positive-support)
# -----------------------------
CONT_DISTS: Dict[str, object] = {
    "expon": stats.expon,
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "pareto": stats.pareto,          # Type I in SciPy; support x >= loc + scale
    "weibull_min": stats.weibull_min,
}


@dataclass
class FitRow:
    column: str
    dist: str
    xmin_q: float     # percentile used for tail threshold (0.0 means full support)
    n: int
    k_params: int
    loglik: float
    aic: float
    bic: float
    xmin: float       # threshold used
    params: str
    notes: str


def _clean_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    return x


def _subsample(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if x.size <= max_n:
        return x
    idx = rng.choice(x.size, size=max_n, replace=False)
    return x[idx]


def _safe_logpdf(dist, x: np.ndarray, params: Tuple[float, ...]) -> Optional[np.ndarray]:
    try:
        lp = dist.logpdf(x, *params)
        if not np.all(np.isfinite(lp)):
            return None
        return lp
    except Exception:
        return None


def _count_estimated_params(params: Tuple[float, ...], fixed_loc: bool, fixed_scale: bool) -> int:
    k = len(params) - (1 if fixed_loc else 0) - (1 if fixed_scale else 0)
    return max(int(k), 1)


def _fit_continuous_with_xmin(
    col: str,
    x_eval: np.ndarray,
    dist_name: str,
    dist,
    xmin_q: float,
    xmin: float,
    fix_loc0_full: bool,
) -> Optional[FitRow]:
    """
    Fit dist on the tail subset determined by xmin. Returns FitRow.
    - For xmin_q == 0: "full-support" fit on all x_eval
    - For xmin_q > 0: fit on x_tail = x_eval[x_eval >= xmin] with support forced to start at xmin
    """
    if x_eval.size < 50:
        return None

    # Tail selection
    if xmin_q <= 0.0:
        x_fit = x_eval
        tail_mode = False
    else:
        x_fit = x_eval[x_eval >= xmin]
        tail_mode = True

    if x_fit.size < 50:
        return None

    # Fit parameters (support control)
    fixed_loc = False
    fixed_scale = False

    try:
        if not tail_mode:
            # full-support
            if fix_loc0_full:
                # fix loc=0 for positive-support distributions
                params = dist.fit(x_fit, floc=0.0)
                fixed_loc = True
                note = "full; fix_loc0"
            else:
                params = dist.fit(x_fit)
                note = "full; loc_free"
        else:
            # tail-support forced to start at xmin
            if dist_name == "pareto":
                # Pareto Type I in SciPy: support x >= loc + scale
                # Force xmin by fixing loc=0 and scale=xmin => support starts at xmin
                if not (xmin > 0.0):
                    return None
                params = dist.fit(x_fit, floc=0.0, fscale=xmin)
                fixed_loc = True
                fixed_scale = True
                note = f"tail; floc=0,fscale=xmin (support=xmin)"
            else:
                # For other dists, loc is a shift; force lower support at xmin via floc=xmin
                params = dist.fit(x_fit, floc=xmin)
                fixed_loc = True
                note = "tail; floc=xmin (support=xmin)"
    except Exception:
        return None

    # Log-likelihood
    lp = _safe_logpdf(dist, x_fit, params)
    if lp is None:
        return None
    ll = float(np.sum(lp))

    n = int(x_fit.size)
    k_params = _count_estimated_params(params, fixed_loc=fixed_loc, fixed_scale=fixed_scale)

    aic = 2.0 * k_params - 2.0 * ll
    bic = math.log(n) * k_params - 2.0 * ll

    return FitRow(
        column=col,
        dist=dist_name,
        xmin_q=float(xmin_q),
        n=n,
        k_params=k_params,
        loglik=float(ll),
        aic=float(aic),
        bic=float(bic),
        xmin=float(xmin if tail_mode else 0.0),
        params=str(tuple(float(p) for p in params)),
        notes=note,
    )


# -----------------------------
# Plotting
# -----------------------------
def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _parse_params(params_str: str) -> Tuple[float, ...]:
    v = ast.literal_eval(params_str)
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    return (float(v),)


def _make_overlay_plots_for_column(
    col: str,
    x_raw: np.ndarray,
    fit_rows: List[FitRow],
    plot_dir: Path,
    rng: np.random.Generator,
    plot_max_n: int,
    clip_q: float,
    bins: int,
    plot_ymax_q: float,
    plot_ymax_mult: float,
    plot_top_k: int,
) -> None:
    if plt is None:
        raise SystemExit("Plotting requested but matplotlib is not available. Install with: pip install matplotlib")

    x = _clean_positive(x_raw)
    if x.size < 50:
        return

    x_plot = _subsample(x, plot_max_n, rng)

    # clip for visibility on x-axis
    clip_q = float(clip_q)
    clip_q = min(max(clip_q, 0.9), 0.999999)
    xmax_vis = float(np.quantile(x_plot, clip_q))
    xmin_vis = float(np.min(x_plot))
    x_plot_vis = x_plot[(x_plot >= xmin_vis) & (x_plot <= xmax_vis)]
    if x_plot_vis.size < 50:
        x_plot_vis = x_plot

    xmin_grid = float(np.min(x_plot_vis))
    xmax_grid = float(np.max(x_plot_vis))
    if not (xmin_grid > 0 and xmax_grid > xmin_grid):
        return

    xg_lin = np.linspace(xmin_grid, xmax_grid, 600)

    # Select best-per-dist (by avg_nll) to avoid messy plots when sweeping xmin
    # and optionally keep only top_k curves overall.
    rows_df = pd.DataFrame([asdict(r) for r in fit_rows])
    rows_df["avg_nll"] = -rows_df["loglik"] / rows_df["n"]
    best_per_dist = rows_df.sort_values("avg_nll", ascending=True).groupby("dist", as_index=False).head(1)
    best_per_dist = best_per_dist.sort_values("avg_nll", ascending=True).head(int(max(plot_top_k, 1)))

    curves = []
    for _, rr in best_per_dist.iterrows():
        dist_name = str(rr["dist"])
        dist = CONT_DISTS.get(dist_name)
        if dist is None:
            continue

        try:
            p = _parse_params(str(rr["params"]))
        except Exception:
            continue

        xmin_q = float(rr["xmin_q"])
        xmin_used = float(rr["xmin"])
        aic = float(rr["aic"])

        def pdf_fn(xx, d=dist, pp=p):
            try:
                y = d.pdf(xx, *pp)
                y = np.asarray(y, dtype=float)
                y[~np.isfinite(y)] = np.nan
                return y
            except Exception:
                return np.full_like(xx, np.nan, dtype=float)

        if xmin_q <= 0.0:
            label = f"{dist_name} (full; AIC={aic:.1f})"
        else:
            label = f"{dist_name} (q={xmin_q:g}, xmin={xmin_used:.3g}; AIC={aic:.1f})"
        curves.append((label, pdf_fn))

    if not curves:
        return

    # Histogram-based y-limit (robust vs Pareto spikes)
    hist_y, _ = np.histogram(x_plot_vis, bins=bins, density=True)
    hist_y = np.asarray(hist_y, dtype=float)
    hist_y = hist_y[np.isfinite(hist_y) & (hist_y > 0)]

    q = float(plot_ymax_q)
    q = min(max(q, 0.5), 0.999999)

    y_max = None
    if hist_y.size > 0:
        y_ref = float(np.quantile(hist_y, q))
        if not np.isfinite(y_ref) or y_ref <= 0:
            y_ref = float(np.max(hist_y))
        y_max = float(plot_ymax_mult) * y_ref

    fig = plt.figure()
    ax = plt.gca()

    ax.hist(x_plot_vis, bins=bins, density=True)

    for label, f in curves:
        y = f(xg_lin)
        if np.all(~np.isfinite(y)):
            continue
        ax.plot(xg_lin, y, label=label)

    # y log-scale
    ax.set_yscale("log")

    
    ax.set_title(f"{col}: histogram + best fitted PDFs (x clipped at q={clip_q})")
    ax.set_xlabel(col)
    ax.set_ylabel("density")
    ax.legend(fontsize="small")
    fig.tight_layout()

    out1 = plot_dir / f"{_safe_name(col)}.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV (with header).")
    ap.add_argument("--columns", nargs="+", required=True, help="Columns to fit.")
    ap.add_argument("--criterion", choices=["aic", "bic"], default="aic", help="Ranking criterion.")
    ap.add_argument("--max-n-eval", type=int, default=200_000, help="Max samples used per column (subsampled).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling.")
    ap.add_argument("--out", default="fit_results.csv", help="Output CSV path.")
    ap.add_argument("--no-fix-loc0", action="store_true", help="Do not fix loc=0 for full-support fits.")
    ap.add_argument("--xmin-qs", nargs="+", type=float,
                    default=[0.0, 0.01, 0.1],
                    help="Sweep xmin candidates as percentiles in [0,1). 0 means full-support fit.")
    ap.add_argument("--xmin-min-n", type=int, default=200,
                    help="Minimum tail sample size required for an xmin_q fit to be attempted.")
    ap.add_argument("--print-top-k", type=int, default=12,
                    help="Print this many top-ranked fits per column.")

    # Plotting
    ap.add_argument("--plot-dir", default=None, help="If set, write overlay plots to this directory.")
    ap.add_argument("--plot-max-n", type=int, default=200_000, help="Max samples used per plot (subsampled).")
    ap.add_argument("--plot-clip-q", type=float, default=0.999, help="Clip data at this quantile for plot visibility.")
    ap.add_argument("--plot-bins", type=int, default=80, help="Histogram bins.")
    ap.add_argument("--plot-ymax-q", type=float, default=0.99,
                    help="Set y-limit from this quantile of histogram density heights (robust vs spikes).")
    ap.add_argument("--plot-ymax-mult", type=float, default=1.2,
                    help="Multiply the histogram-based y-limit by this factor for headroom.")
    ap.add_argument("--plot-top-k", type=int, default=5,
                    help="Max number of curves to overlay (best-per-dist by avg_nll, then top-k overall).")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv)
    rows: List[FitRow] = []

    fix_loc0_full = not args.no_fix_loc0

    # sanitize xmin-qs
    xmin_qs = []
    for q in args.xmin_qs:
        q = float(q)
        if q < 0.0:
            q = 0.0
        if q >= 1.0:
            q = 0.999999
        xmin_qs.append(q)
    xmin_qs = sorted(set(xmin_qs))

    for col in args.columns:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not in CSV; skipping.")
            continue

        x_clean = _clean_positive(df[col].to_numpy())
        if x_clean.size < 50:
            print(f"[WARN] Column '{col}' has <50 positive finite samples; skipping.")
            continue

        # One evaluation subsample per column for consistent sweep + speed
        x_eval = _subsample(x_clean, args.max_n_eval, rng)

        # Precompute all xmin thresholds from x_eval (so sweep is deterministic for this run)
        thresholds: Dict[float, float] = {}
        for q in xmin_qs:
            if q <= 0.0:
                thresholds[q] = 0.0
            else:
                thresholds[q] = float(np.quantile(x_eval, q))

        for xmin_q in xmin_qs:
            xmin = thresholds[xmin_q]

            # Ensure tail has enough points
            if xmin_q > 0.0:
                n_tail = int(np.sum(x_eval >= xmin))
                if n_tail < int(args.xmin_min_n):
                    continue

            for dist_name, dist in CONT_DISTS.items():
                r = _fit_continuous_with_xmin(
                    col=col,
                    x_eval=x_eval,
                    dist_name=dist_name,
                    dist=dist,
                    xmin_q=xmin_q,
                    xmin=xmin,
                    fix_loc0_full=fix_loc0_full,
                )
                if r is not None:
                    rows.append(r)

    if not rows:
        raise SystemExit("No fits succeeded. Try different columns, increase sample size, or disable fix_loc0.")

    out_df = pd.DataFrame([asdict(r) for r in rows])
    out_df["avg_nll"] = -out_df["loglik"] / out_df["n"]

    # Rank per column
    out_df["rank_score"] = out_df[args.criterion]
    out_df = out_df.sort_values(["column", "rank_score"], ascending=True)
    out_df.to_csv(args.out, index=False)

    # Print results per column (top-k)
    print("\n=== Fits per column (ranked) ===")
    print("NOTE: When xmin_q varies, n varies too; compare avg_nll more than raw AIC/BIC across different xmin_q.")
    for col, g in out_df.groupby("column", sort=False):
        g2 = g.sort_values("rank_score", ascending=True).head(int(max(args.print_top_k, 1)))
        print(f"\n[{col}]")
        print(g2[["dist", "xmin_q", "n", "k_params", "aic", "bic", "avg_nll", "xmin", "params", "notes"]].to_string(index=False))

    print(f"\n[OK] Wrote full results to {args.out}")

    # Plotting
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        FITROW_COLS = ["column", "dist", "xmin_q", "n", "k_params", "loglik", "aic", "bic", "xmin", "params", "notes"]

        for col in args.columns:
            if col not in df.columns:
                continue
            if col not in out_df["column"].values:
                continue

            records = out_df.loc[out_df["column"] == col, FITROW_COLS].to_dict(orient="records")
            col_rows = [FitRow(**r) for r in records]

            _make_overlay_plots_for_column(
                col=col,
                x_raw=df[col].to_numpy(),
                fit_rows=col_rows,
                plot_dir=plot_dir,
                rng=rng,
                plot_max_n=args.plot_max_n,
                clip_q=args.plot_clip_q,
                bins=args.plot_bins,
                plot_ymax_q=args.plot_ymax_q,
                plot_ymax_mult=args.plot_ymax_mult,
                plot_top_k=args.plot_top_k,
            )

        print(f"[OK] Wrote overlay plots under: {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
