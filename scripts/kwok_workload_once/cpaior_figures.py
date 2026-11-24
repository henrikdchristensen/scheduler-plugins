"""Draw the CPAIOR figures from the same aggregates as the paper tables."""

from math import nan
from pathlib import Path

from matplotlib import colormaps, rc_context
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from scripts.kwok_workload_once.cpaior_tables import BUDGETS, LABELS, NODES, Dataset, Outcome, cluster_summary


OUTCOMES = (
    ("better_optimal", "Better & proven optimal", "#20713b"),
    ("better_feasible", "Better", "#72b774"),
    ("default_optimal", "Default proven optimal", "#4987b8"),
    ("no_call", "No call", "#d4d6d9"),
    ("no_gain", "No gain", "#cc785f"),
)
SOLVER_COLORS = {"cp_sat": "Blues", "cplex": "Oranges", "gurobi": "Purples"}
FIGURE_LABELS = {**LABELS, "cp_sat": "CP-SAT (OR-Tools)"}
Summaries = dict[str, dict[tuple[int, int], Outcome]]


def plot_outcomes(summaries: Summaries) -> Figure:
    count = len(summaries)
    fig = Figure(figsize=(4.60 if count == 2 else 6.5, 1.78))
    axes = fig.subplots(1, count, sharey=True, squeeze=False)[0]
    for i, (ax, (solver, groups)) in enumerate(zip(axes, summaries.items())):
        ax.set_title(FIGURE_LABELS[solver], fontsize=8, pad=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_xticks(range(len(NODES)), NODES)
        ax.set_xlabel("# nodes", labelpad=1)
        ax.grid(axis="y", alpha=.25, linewidth=.5)
        ax.set_axisbelow(True)
        for j, nodes in enumerate(NODES):
            for k, budget in enumerate(BUDGETS):
                x = j + (k - 1) * .24
                summary = groups[nodes, budget]
                bottom = 0.0
                for attr, label, color in OUTCOMES:
                    height = 100 * getattr(summary, attr) / summary.runs
                    ax.bar(x, height, .22, bottom=bottom, color=color, label=label if i == j == k == 0 else None)
                    bottom += height
                ax.text(x, 102, str(budget), ha="center", va="bottom", fontsize=5.3, clip_on=False)
    axes[0].set_ylabel("% of instances", labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=len(OUTCOMES), fontsize=5.7,
               frameon=False, columnspacing=1.1, handlelength=1.2, handletextpad=.35)
    fig.subplots_adjust(left=.105, right=.995, top=.78, bottom=.28, wspace=.105)
    return fig


def plot_runtime(summaries: Summaries) -> Figure:
    fig = Figure(figsize=(4.60, 1.70))
    proof, runtime = fig.subplots(1, 2)
    n_series = len(summaries) * len(BUDGETS)
    step = .76 / n_series
    for i, (solver, groups) in enumerate(summaries.items()):
        for j, budget in enumerate(BUDGETS):
            series = [groups[nodes, budget] for nodes in NODES]
            proved = [100 * group.proven / group.included if group.included else nan for group in series]
            # Zero or missing runtimes cannot appear on a logarithmic axis.
            durations = [value if value is not None and value > 0 else nan
                         for value in (group.mean_seconds for group in series)]
            offset = (i * len(BUDGETS) + j - (n_series - 1) / 2) * step
            x = [index + offset for index in range(len(NODES))]
            color = colormaps[SOLVER_COLORS[solver]]((.40, .60, .82)[j])
            proof.bar(x, proved, step * .94, color=color)
            runtime.bar(x, durations, step * .94, color=color, label=FIGURE_LABELS[solver] if j == 1 else None)
    proof.set(title="Proven optimal (1/10/20 s)", ylabel="% proven optimal", ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
    runtime.set(title="Mean time (1/10/20 s)", ylabel="mean optimiser time (s)", yscale="log")
    for ax in (proof, runtime):
        ax.set_xticks(range(len(NODES)), NODES)
        ax.set_xlabel("# nodes", labelpad=1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.25, linewidth=.5)
    handles, labels = runtime.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=len(summaries), fontsize=6.5, frameon=False)
    fig.subplots_adjust(left=.105, right=.995, top=.86, bottom=.28, wspace=.36)
    return fig


def generate(data: Dataset, out_dir: Path) -> None:
    summaries = {
        solver: {(nodes, budget): cluster_summary(results, nodes, budget)
                 for nodes in NODES for budget in BUDGETS}
        for solver, results in data.items()
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with rc_context({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .6, "pdf.fonttype": 42}):
        for name, plot in (("outcomes_by_nodes", plot_outcomes), ("runtime_proofs", plot_runtime)):
            path = out_dir / f"{name}.pdf"
            plot(summaries).savefig(path)
            print(f"[ok] wrote {path}")
