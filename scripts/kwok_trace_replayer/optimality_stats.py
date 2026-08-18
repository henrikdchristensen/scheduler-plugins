#!/usr/bin/env python3
# optimality_stats.py
"""
Optimality-certification statistics for OPSche (kwok_trace_replayer experiments).

For every solver run the plugin records, in the aggregated
``analysis/kwok_trace_replayer/results_seeds.csv``, whether the best improving
plan was proven optimal by CP-SAT (``solver_optimal``), only feasible
(``solver_feasible``), or no improving plan was found (``solver_failed``).

This script pools all solver runs of the three main trigger modes (with
DefaultPreemption enabled, inter-arrivals 4/8/16 s) and reports, for the
non-blocking and blocking variants:
  * % of all solver runs that returned a certified-optimal improving plan, and
    * % of improving plans returned by the solver that were proven optimal.

These are the numbers cited in the evaluation ("Optimality Certification").

Usage:
    conda activate master
    python -m scripts.kwok_trace_replayer.optimality_stats
"""

import argparse
from pathlib import Path

import pandas as pd

RESULTS_CSV = Path("analysis/kwok_trace_replayer/results_seeds.csv")

# Primary modes reported in the main paper figure (8 s interval / idle window).
DEFAULT_MODES = ("schedulingfailure", "periodic8s", "stablequeue8s")

# Inter-arrival settings reported in the paper (tab:trace-params); the CSV also
# contains a 2 s setting that the paper does not report, so we exclude it.
DEFAULT_ARRIVALS = ("4s", "8s", "16s")


def _mode_of(plugin_config: str) -> str:
    return plugin_config.split("mode=", 1)[1].split("_blocking=", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(RESULTS_CSV))
    ap.add_argument("--defpreempt", default="1", choices=["0", "1"])
    ap.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    ap.add_argument("--arrivals", nargs="+", default=list(DEFAULT_ARRIVALS))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["mode"] = df["plugin_config"].map(_mode_of)
    df["arrival"] = df["job_name"].str.extract(r"arrival=(\d+s)")
    df = df[df["mode"].isin(args.modes) & df["arrival"].isin(args.arrivals)]

    print("Optimality certification, pooled over all solver runs "
          f"(modes={','.join(args.modes)}, arrivals={','.join(args.arrivals)}, "
          f"defpreempt={args.defpreempt})\n")

    for blocking, label in [("0", "non-blocking"), ("1", "blocking")]:
        suffix = f"_blocking={blocking}_defpreempt={args.defpreempt}"
        sub = df[df["plugin_config"].str.endswith(suffix)]
        att = sub["solver_attempts_mean"].sum()
        opt = sub["solver_optimal_mean"].sum()
        feasible = sub["solver_feasible_mean"].sum()
        improving = opt + feasible
        print(f"{label:12s}: {opt/att*100:.0f}% of all solver runs proven optimal, "
              f"{opt/improving*100:.0f}% of improving plans proven optimal "
              f"(runs={att:.0f}, optimal={opt:.0f}, feasible={feasible:.0f})")


if __name__ == "__main__":
    main()
