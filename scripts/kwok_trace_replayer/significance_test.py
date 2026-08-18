#!/usr/bin/env python3
"""Corrected significance analysis for the main OPSche comparison.

Each row in ``results_seeds.csv`` is already aggregated per experimental run.
For each trace setting (nodes, priorities, inter-arrival time), this script:

1. compares the five paired OPSche/default run differences with a two-sided
   Wilcoxon signed-rank test;
2. combines the three inter-arrival-level p-values within each fixed node count,
    mode, priority configuration, and metric using Fisher's method; and
3. within each node count, applies Holm correction across the 18 primary
    hypotheses (3 modes x 2 priority configurations x 3 metrics).

Usage:
    conda activate master
    python -m scripts.kwok_trace_replayer.significance_test
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import PermutationMethod, combine_pvalues, wilcoxon


RESULTS_CSV = Path("analysis/kwok_trace_replayer/results_seeds.csv")
DEFAULT_MODES = ("schedulingfailure", "periodic8s", "stablequeue8s")
DEFAULT_ARRIVALS = ("4s", "8s", "16s")
METRICS = (
    ("delta_U_pct_eff_mean", "usage", "higher"),
    ("delta_L_ms_total_mean", "latency", "lower"),
    ("delta_D_num_total_mean", "deletions", "lower"),
)


def _mode_of(plugin_config: str) -> str:
    return plugin_config.split("mode=", 1)[1].split("_blocking=", 1)[0]


def _wilcoxon_p(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) != 5:
        raise ValueError(f"expected 5 paired runs per trace setting, got {len(values)}")
    if (values == 0).all():
        return 1.0
    exact = PermutationMethod(n_resamples=float("inf"))
    return float(wilcoxon(values, alternative="two-sided", method=exact).pvalue)


def _holm_adjust(raw_p_values: list[float]) -> list[float]:
    count = len(raw_p_values)
    order = sorted(range(count), key=raw_p_values.__getitem__)
    adjusted = [1.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * raw_p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(RESULTS_CSV))
    parser.add_argument("--blocking", default="0", choices=["0", "1"])
    parser.add_argument("--defpreempt", default="1", choices=["0", "1"])
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    frame["mode"] = frame["plugin_config"].map(_mode_of)
    frame["nodes"] = frame["job_name"].str.extract(r"nodes=(\d+)").astype(int)
    frame["priorities"] = frame["job_name"].str.extract(r"prio=(\d+)").astype(int)
    frame["arrival"] = frame["job_name"].str.extract(r"arrival=(\d+s)")
    suffix = f"_blocking={args.blocking}_defpreempt={args.defpreempt}"
    frame = frame[
        frame["plugin_config"].str.endswith(suffix)
        & frame["mode"].isin(DEFAULT_MODES)
        & frame["arrival"].isin(DEFAULT_ARRIVALS)
    ]

    results = []
    for nodes in (16, 32):
        for priorities in (1, 4):
            for mode in DEFAULT_MODES:
                subset = frame[
                    (frame["nodes"] == nodes)
                    & (frame["priorities"] == priorities)
                    & (frame["mode"] == mode)
                ]
                for column, metric, improvement in METRICS:
                    setting_p_values = []
                    setting_means = []
                    for _arrival, setting in subset.groupby("arrival"):
                        setting_p_values.append(_wilcoxon_p(setting[column]))
                        setting_means.append(float(setting[column].mean()))
                    if len(setting_p_values) != 3:
                        raise ValueError(
                            f"expected 3 inter-arrival settings for "
                            f"{nodes}/{mode}/{priorities}/{metric}, got {len(setting_p_values)}"
                        )
                    combined_p = float(combine_pvalues(setting_p_values, method="fisher").pvalue)
                    mean_difference = sum(setting_means) / len(setting_means)
                    improves = mean_difference > 0 if improvement == "higher" else mean_difference < 0
                    results.append({
                        "nodes": nodes,
                        "priorities": priorities,
                        "mode": mode,
                        "metric": metric,
                        "mean_difference": mean_difference,
                        "direction": "improves" if improves else "worsens",
                        "combined_p": combined_p,
                    })

    for nodes in (16, 32):
        node_results = [result for result in results if result["nodes"] == nodes]
        adjusted = _holm_adjust([result["combined_p"] for result in node_results])
        for result, adjusted_p in zip(node_results, adjusted):
            result["adjusted_p"] = adjusted_p

    print("Run-level Wilcoxon tests; Fisher combination across 3 inter-arrival settings; ")
    print("Holm correction across 18 hypotheses within each node count\n")
    print(f"{'nodes':>5} {'prio':>4} {'mode':18s} {'metric':10s} {'mean diff':>11} {'raw p':>10} "
          f"{'Holm p':>10} {'result'}")
    for result in results:
        significant = result["adjusted_p"] < 0.05
        conclusion = result["direction"] if significant else "not significant"
        print(
            f"{result['nodes']:>5} {result['priorities']:>4} "
            f"{result['mode']:18s} {result['metric']:10s} "
            f"{result['mean_difference']:>+11.3f} {result['combined_p']:>10.3g} "
            f"{result['adjusted_p']:>10.3g} {conclusion}"
        )


if __name__ == "__main__":
    main()
