#!/usr/bin/env python3
"""
Fit candidate distributions to columns in a CSV using MLE (SciPy) and rank via AIC/BIC.

Example:
  python fit_distributions.py \
    --csv data.csv \
    --columns inter_arrival_us life_time_us cpu_request mem_request priority \
    --criterion aic \
    --out fit_results.csv \
    --discrete-columns priority

Notes:
- With huge n, p-values are often "always significant"; use AIC/BIC for ranking.
- We fix loc=0 for positive-support distributions (common for these metrics).
- Discrete fitting is ONLY applied to columns listed in --discrete-columns (no heuristics).
"""

import argparse
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError as e:
    raise SystemExit("This script requires scipy. Install with: pip install scipy") from e


# -----------------------------
# Candidate distributions (popular + positive-support)
# -----------------------------
CONT_DISTS: Dict[str, object] = {
    "expon": stats.expon,
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "weibull_min": stats.weibull_min,
    "lomax": stats.lomax,      # Pareto Type II
    "pareto": stats.pareto,    # Pareto Type I
}

# Discrete fits (optional)
# Note: SciPy's geom is support {1,2,...}; your priorities may include 0.
# We'll implement a geometric-on-{0,1,2,...} ourselves.
DISC_NAMES = ["geometric0", "poisson", "zipf"]


@dataclass
class FitRow:
    column: str
    dist: str
    n: int
    k_params: int
    loglik: float
    aic: float
    bic: float
    xmin: float
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


def _fit_continuous(
    col: str,
    x: np.ndarray,
    dist_name: str,
    dist,
    fix_loc0: bool,
    max_n_eval: int,
    rng: np.random.Generator,
) -> Optional[FitRow]:
    """
    Fit a continuous distribution via MLE and return FitRow with AIC/BIC.
    Uses a subsample for evaluation (max_n_eval) to keep runtime sane.
    """
    x = _clean_positive(x)
    if x.size < 50:
        return None

    x_eval = _subsample(x, max_n_eval, rng)

    # Fit parameters
    try:
        if fix_loc0:
            params = dist.fit(x_eval, floc=0.0)
            # params includes loc even though fixed; count only estimated params
            k_params = max(len(params) - 1, 1)
        else:
            params = dist.fit(x_eval)
            k_params = len(params)
    except Exception:
        return None

    # Log-likelihood
    lp = _safe_logpdf(dist, x_eval, params)
    if lp is None:
        return None
    ll = float(np.sum(lp))

    # Scores
    n = int(x_eval.size)
    aic = 2.0 * k_params - 2.0 * ll
    bic = math.log(n) * k_params - 2.0 * ll

    return FitRow(
        column=col,
        dist=dist_name,
        n=n,
        k_params=k_params,
        loglik=ll,
        aic=float(aic),
        bic=float(bic),
        xmin=0.0,
        params=str(tuple(float(p) for p in params)),
        notes="fix_loc0" if fix_loc0 else "",
    )


# -----------------------------
# Discrete fits (only for user-specified columns)
# -----------------------------
def _clean_discrete(x: np.ndarray) -> np.ndarray:
    """
    Prepare discrete data: finite ints (no positivity filtering).
    Keeps zeros.
    """
    k = np.asarray(x)
    k = k[np.isfinite(k)].astype(int)
    return k


def _fit_geometric0(col: str, x: np.ndarray, max_n_eval: int, rng: np.random.Generator) -> Optional[FitRow]:
    """
    Geometric on support {0,1,2,...}: P(K=k)=p(1-p)^k
    MLE: p = 1/(mean(k)+1)
    """
    k = _clean_discrete(x)
    if k.size < 50:
        return None

    k_eval = _subsample(k, max_n_eval, rng)
    k_eval = k_eval - int(k_eval.min())  # shift to start at 0

    mean_k = float(np.mean(k_eval))
    p = 1.0 / (mean_k + 1.0)
    if not (0.0 < p < 1.0):
        return None

    ll = float(k_eval.size * np.log(p) + np.sum(k_eval) * np.log(1.0 - p))
    k_params = 1
    n = int(k_eval.size)
    aic = 2.0 * k_params - 2.0 * ll
    bic = math.log(n) * k_params - 2.0 * ll

    return FitRow(
        column=col,
        dist="geometric0",
        n=n,
        k_params=k_params,
        loglik=ll,
        aic=float(aic),
        bic=float(bic),
        xmin=0.0,
        params=str((p,)),
        notes="shift_to_zero",
    )


def _fit_poisson(col: str, x: np.ndarray, max_n_eval: int, rng: np.random.Generator) -> Optional[FitRow]:
    k = _clean_discrete(x)
    if k.size < 50:
        return None

    k_eval = _subsample(k, max_n_eval, rng)
    k_eval = k_eval - int(k_eval.min())  # shift to start at 0
    lam = float(np.mean(k_eval))
    if lam < 0:
        return None

    ll = float(np.sum(stats.poisson.logpmf(k_eval, mu=lam)))
    k_params = 1
    n = int(k_eval.size)
    aic = 2.0 * k_params - 2.0 * ll
    bic = math.log(n) * k_params - 2.0 * ll

    return FitRow(
        column=col,
        dist="poisson",
        n=n,
        k_params=k_params,
        loglik=ll,
        aic=float(aic),
        bic=float(bic),
        xmin=0.0,
        params=str((lam,)),
        notes="shift_to_zero",
    )


def _fit_zipf(col: str, x: np.ndarray, max_n_eval: int, rng: np.random.Generator) -> Optional[FitRow]:
    # Zipf is support {1,2,...}; shift to start at 1
    k = _clean_discrete(x)
    if k.size < 100:
        return None

    k_eval = _subsample(k, max_n_eval, rng)
    k_eval = k_eval - int(k_eval.min()) + 1
    if np.any(k_eval < 1):
        return None

    try:
        params = stats.zipf.fit(k_eval, floc=0)  # shape only + loc fixed
        a = float(params[0])
        ll = float(np.sum(stats.zipf.logpmf(k_eval, a=a)))
        k_params = 1
        n = int(k_eval.size)
        aic = 2.0 * k_params - 2.0 * ll
        bic = math.log(n) * k_params - 2.0 * ll
        return FitRow(
            column=col,
            dist="zipf",
            n=n,
            k_params=k_params,
            loglik=ll,
            aic=float(aic),
            bic=float(bic),
            xmin=0.0,
            params=str((a,)),
            notes="shift_to_one",
        )
    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV (with header).")
    ap.add_argument("--columns", nargs="+", required=True, help="Columns to fit.")
    ap.add_argument("--discrete-columns", nargs="*", default=[],
                    help="Columns to treat as discrete (e.g., priority). Only these get discrete fits.")
    ap.add_argument("--criterion", choices=["aic", "bic"], default="aic", help="Ranking criterion.")
    ap.add_argument("--max-n-eval", type=int, default=200_000, help="Max samples used per fit (subsampled).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling.")
    ap.add_argument("--out", default="fit_results.csv", help="Output CSV path.")
    ap.add_argument("--no-fix-loc0", action="store_true", help="Do not fix loc=0 for positive distributions.")
    ap.add_argument("--disable-continuous-for-discrete", action="store_true",
                    help="If set, do NOT run continuous fits for discrete columns; only discrete fits.")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv)
    rows: List[FitRow] = []

    fix_loc0 = not args.no_fix_loc0
    discrete_set = set(args.discrete_columns)

    for col in args.columns:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not in CSV; skipping.")
            continue

        x = df[col].to_numpy()
        is_discrete = col in discrete_set

        # Continuous fits
        if not (is_discrete and args.disable_continuous_for_discrete):
            for name, dist in CONT_DISTS.items():
                r = _fit_continuous(
                    col=col,
                    x=x,
                    dist_name=name,
                    dist=dist,
                    fix_loc0=fix_loc0,
                    max_n_eval=args.max_n_eval,
                    rng=rng,
                )
                if r is not None:
                    rows.append(r)

        # Discrete fits (ONLY for user-specified discrete columns)
        if is_discrete:
            for r in [
                _fit_geometric0(col, x, args.max_n_eval, rng),
                _fit_poisson(col, x, args.max_n_eval, rng),
                _fit_zipf(col, x, args.max_n_eval, rng),
            ]:
                if r is not None:
                    rows.append(r)

    if not rows:
        raise SystemExit("No fits succeeded. Try different columns, increase sample size, or disable fix_loc0.")

    out_df = pd.DataFrame([asdict(r) for r in rows])

    # Rank per column
    out_df["rank_score"] = out_df[args.criterion]
    out_df = out_df.sort_values(["column", "rank_score"], ascending=True)
    out_df.to_csv(args.out, index=False)

    # Print results per column (all)
    print("\n=== Fits per column (ranked) ===")
    for col, g in out_df.groupby("column", sort=False):
        g2 = g.sort_values("rank_score", ascending=True)
        print(f"\n[{col}]")
        print(g2[["dist", "n", "k_params", "aic", "bic", "xmin", "params", "notes"]].to_string(index=False))

    print(f"\n[OK] Wrote full results to {args.out}")


if __name__ == "__main__":
    main()
