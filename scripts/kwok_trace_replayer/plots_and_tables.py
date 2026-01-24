#!/usr/bin/env python3
# scripts/kwok_trace_replayer/plots_and_tables.py
"""
python -m scripts.kwok_trace_replayer.plots_and_tables --in-results analysis/kwok_trace_replayer/results_paired.csv --out-dir analysis/kwok_trace_replayer

Produces:
  Tables (under <out-dir>/tables):
    - delta_U_eff_run.{txt,tex}
    - delta_R.{txt,tex}
    - delta_D.{txt,tex}
    - delta_L.{txt,tex}
    - solver_attempts.{txt,tex}
    - plans_activated.{txt,tex}
    - big_table_util_latency_deletions_optimizations.{txt,tex}
    - delta_D_without_defaultpreemption.{txt,tex}
    - delta_L_without_defaultpreemption.{txt,tex}
    - big_table_without_defaultpreemption_util_latency_deletions_optimizations.{txt,tex}

  Figures (under <out-dir>/figures):
    - delta_U_eff_run_kmax<K>.<png|pdf>
    - delta_D_kmax<K>.<png|pdf>
    - delta_L_kmax<K>.<png|pdf>
    - delta_D_kmax<K>_without_defaultpreemption.<png|pdf>
    - delta_L_kmax<K>_without_defaultpreemption.<png|pdf>

Key changes vs prior version:
  - Mode labels are now abbreviations: SF, PR-8-B, PR-8-NB, SQ-2-B, SQ-2-NB.
  - If defpreempt=0, add suffix "-NDF" (except SF, which stays "SF").
  - All LaTeX/ASCII tables use "arrivals at top; modes under arrivals; metrics as rows".
  - Each .tex file writes one tabular per kmax (kmax=1 and kmax=4), not mixed.
  - Only tabular environments are emitted (no table/table* wrapper).
"""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

TABLE_DECIMALS: int = 1
JOB_RE = re.compile(r"nodes=(\d+)_prio=(\d+)_arrival=([0-9.]+)s")

EXPECTED_COLS = [
    "job_name",
    "plugin_config",
    "delta_U_eff_run_mean",
    "delta_R_p1_mean", "delta_R_p2_mean", "delta_R_p3_mean", "delta_R_p4_mean", "delta_R_total_mean",
    "delta_D_p1_mean", "delta_D_p2_mean", "delta_D_p3_mean", "delta_D_p4_mean", "delta_D_total_mean",
    "delta_L_s_p1_mean", "delta_L_s_p2_mean", "delta_L_s_p3_mean", "delta_L_s_p4_mean", "delta_L_s_total_mean",
    "solver_attempts_mean", "plan_activated_mean",
]
KEY_COLS = ["nodes", "kmax", "arrival_s", "mode", "blocking", "defpreempt"]

# =============================================================================
# Plot styling
# =============================================================================

PLOT_MARKER_SIZE = 3.0
PLOT_ARRIVAL_X_SPACING = 0.6
PLOT_MODE_X_SPACING = 0.05

PLOT_FIGSIZE = (3.1, 2.2)
PLOT_TICK_FONTSIZE = 7
PLOT_LEGEND_FONTSIZE = 5
PLOT_AXIS_LABEL_FONTSIZE = 7
PLOT_Y_PADDING = 0.15

PLOT_LEGEND_HANDLE_LENGTH = 0.6
PLOT_LEGEND_COLUMN_SPACING = 0.5
PLOT_LEGEND_BORDER_AXES_PAD = 0.4
PLOT_LEGEND_LOC = "lower left"
PLOT_LEGEND_NCOL = 2


def add_standard_legend(
    *,
    ax: plt.Axes,
    handles: Sequence[object],
    labels: Sequence[str],
    loc: str = PLOT_LEGEND_LOC,
) -> None:
    """Standard legend inside the axes, deduped and compact."""
    clean: List[Tuple[object, str]] = []
    seen = set()
    for h, lbl in zip(handles, labels):
        s = str(lbl)
        if not s or s.startswith("_"):
            continue
        if s in seen:
            continue
        seen.add(s)
        clean.append((h, s))

    if not clean:
        return

    hh, ll = zip(*clean)
    ax.legend(
        hh,
        ll,
        loc=loc,
        ncol=min(int(PLOT_LEGEND_NCOL), len(ll)),
        frameon=True,
        fontsize=PLOT_LEGEND_FONTSIZE,
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
        borderaxespad=PLOT_LEGEND_BORDER_AXES_PAD,
    )

# =============================================================================
# Parsing helpers
# =============================================================================


def parse_job(job_name: str) -> Optional[Tuple[int, int, float]]:
    """
    Parse job_name format:
        nodes=<N>_prio=<kmax>_arrival=<a>s
    """
    m = JOB_RE.fullmatch(str(job_name).strip())
    if not m:
        return None
    n = int(m.group(1))
    k = int(m.group(2))
    a = float(m.group(3))
    return n, k, a


def parse_plugin_config(plugin_config: str) -> Dict[str, str]:
    """
    plugin_config format (from seal_results.py):
      mode=<mode>_blocking=<0/1>_defpreempt=<0/1>
    """
    kv: Dict[str, str] = {}
    for tok in str(plugin_config).split("_"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return kv


def canonical_mode(mode: str) -> str:
    """
    Canonicalize mode strings so ordering/labels are stable across inputs.
    """
    s = str(mode).strip().lower()
    if s in {"scheduling_failure", "scheduling-failure", "schedulingfailure", "sched-failure", "schedfailure"}:
        return "scheduling-failure"
    m = re.fullmatch(r"periodic-?([0-9.]+)s", s)
    if m:
        return f"periodic{m.group(1)}s"
    m = re.fullmatch(r"(?:stable-queue|stablequeue)-?([0-9.]+)s", s)
    if m:
        return f"stable-queue-{m.group(1)}s"
    return s


# Abbreviations (base, without -NDF suffix)
# Keyed by (canonical_mode, blocking) where blocking is 1/0
MODE_ABBR: Dict[Tuple[str, Optional[int]], str] = {
    ("scheduling-failure", None): "SF",
    ("periodic8s", 1): "PR-8-B",
    ("periodic8s", 0): "PR-8-NB",
    ("stable-queue-2s", 1): "SQ-2-B",
    ("stable-queue-2s", 0): "SQ-2-NB",
}

MODE_ORDER: List[Tuple[str, Optional[int]]] = [
    ("scheduling-failure", None),
    ("periodic8s", 1),
    ("periodic8s", 0),
    ("stable-queue-2s", 1),
    ("stable-queue-2s", 0),
]
MODE_RANK: Dict[Tuple[str, Optional[int]], int] = {k: i for i, k in enumerate(MODE_ORDER)}


@dataclass(frozen=True)
class RowKey:
    mode: str
    blocking: int
    defpreempt: int

    @staticmethod
    def from_plugin_config(plugin_config: str) -> "RowKey":
        kv = parse_plugin_config(plugin_config)
        mode = canonical_mode(kv.get("mode", "unknown"))
        if mode == "scheduling-failure":
            blocking = 0
        else:
            blocking = 1 if str(kv.get("blocking", "0")).lower() in {"1", "true", "yes"} else 0
        try:
            defpreempt = int(str(kv.get("defpreempt", "0")))
        except Exception:
            defpreempt = 0
        return RowKey(mode=mode, blocking=blocking, defpreempt=defpreempt)

    def sort_key(self) -> Tuple[int, int, str]:
        base = (self.mode, None) if self.mode == "scheduling-failure" else (self.mode, int(self.blocking))
        base_rank = MODE_RANK.get(base, 10_000)
        # Prefer defpreempt=1 before defpreempt=0 when both exist
        def_rank = 0 if self.defpreempt == 1 else 1
        return (base_rank, def_rank, self.mode)

    def abbr(self, include_defpreempt_suffix: bool = True) -> str:
        if self.mode == "scheduling-failure":
            return "SF"
        key = (self.mode, int(self.blocking))
        base = MODE_ABBR.get(key, f"{self.mode}:{self.blocking}")
        if include_defpreempt_suffix and self.defpreempt == 0:
            return f"{base}-NDF"
        return base


# =============================================================================
# Formatting
# =============================================================================

def is_finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def nan_str(latex: bool) -> str:
    return (r"\text{--}" if latex else "--")


def fmt_signed(x: object, decimals: int, nan_s: str) -> str:
    if not is_finite(x):
        return nan_s
    v = float(x)
    v = round(v, int(decimals))
    if v == 0.0:
        v = 0.0
    return f"{v:+.{decimals}f}"


def fmt_unsigned_int(x: object, nan_s: str) -> str:
    if not is_finite(x):
        return nan_s
    return f"{int(round(float(x))):d}"


def fmt_signed_int(x: object, nan_s: str) -> str:
    if not is_finite(x):
        return nan_s
    return f"{int(round(float(x))):+d}"


def arrival_group_label(a: float, *, latex: bool) -> str:
    a_i = int(a) if abs(a - round(a)) < 1e-9 else a
    if latex:
        return rf"$\mu_A={a_i}\,\mathrm{{s}}$"
    return f"muA={a_i}s"


def kmax_caption(kmax: int, *, latex: bool) -> str:
    if latex:
        if kmax == 1:
            return r"$k_{\max}=1$ (w/o priorities)"
        return rf"$k_{{\max}}={kmax}$ (w/ priorities)"
    if kmax == 1:
        return "kmax=1 (w/o priorities)"
    return f"kmax={kmax} (w/ priorities)"


# =============================================================================
# Data access
# =============================================================================

def load_results(results_csv: Path) -> pd.DataFrame:
    if not results_csv.exists():
        raise SystemExit(f"Not found: {results_csv}")

    df = pd.read_csv(results_csv)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{results_csv} missing columns: {', '.join(missing)}")

    parsed = df["job_name"].map(parse_job)
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), "job_name"].head(5).tolist()
        raise SystemExit(f"Could not parse some job_name values (examples): {bad}")

    df["nodes"] = parsed.map(lambda t: t[0] if t is not None else np.nan).astype(int)
    df["kmax"] = parsed.map(lambda t: t[1] if t is not None else np.nan).astype(int)
    df["arrival_s"] = parsed.map(lambda t: t[2] if t is not None else np.nan).astype(float)

    rks = df["plugin_config"].map(RowKey.from_plugin_config)
    df["mode"] = rks.map(lambda r: r.mode).astype(str)
    df["blocking"] = rks.map(lambda r: r.blocking).astype(int)
    df["defpreempt"] = rks.map(lambda r: r.defpreempt).astype(int)

    return df


def build_lookup(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.drop_duplicates(subset=KEY_COLS, keep="first")
    return d.set_index(KEY_COLS).sort_index()


def lookup_value(
    lookup: pd.DataFrame,
    *,
    nodes: int,
    kmax: int,
    arrival_s: float,
    rk: RowKey,
    col: str,
) -> float:
    key = (int(nodes), int(kmax), float(arrival_s), str(rk.mode), int(rk.blocking), int(rk.defpreempt))
    try:
        return float(lookup.at[key, col])
    except KeyError:
        return float("nan")


# =============================================================================
# defpreempt delta helpers (within-plugin)
# =============================================================================

def build_defpreempt_rows(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """
    Return (mode, blocking) pairs that have BOTH defpreempt=0 and defpreempt=1.
    Excludes scheduling-failure (no both variants).
    """
    seen: Dict[Tuple[str, int], set] = {}
    for pc in df["plugin_config"].unique().tolist():
        rk = RowKey.from_plugin_config(pc)
        if rk.mode == "scheduling-failure":
            continue
        seen.setdefault((rk.mode, rk.blocking), set()).add(rk.defpreempt)
    rows = [(m, b) for (m, b), defs in seen.items() if (0 in defs and 1 in defs)]
    rows.sort(key=lambda mb: RowKey(mode=mb[0], blocking=mb[1], defpreempt=1).sort_key())
    return rows


def defpreempt_delta(
    lookup: pd.DataFrame,
    *,
    nodes: int,
    kmax: int,
    arrival_s: float,
    mode: str,
    blocking: int,
    col: str,
) -> float:
    rk0 = RowKey(mode=mode, blocking=blocking, defpreempt=0)
    rk1 = RowKey(mode=mode, blocking=blocking, defpreempt=1)
    v0 = lookup_value(lookup, nodes=nodes, kmax=kmax, arrival_s=arrival_s, rk=rk0, col=col)
    v1 = lookup_value(lookup, nodes=nodes, kmax=kmax, arrival_s=arrival_s, rk=rk1, col=col)
    if not (is_finite(v0) and is_finite(v1)):
        return float("nan")
    return float(v0) - float(v1)


# =============================================================================
# Table model (metrics-as-rows; arrivals top; modes under arrivals)
# =============================================================================

MetricGetter = Callable[[int, RowKey, int, float], float]  # (kmax, rk, nodes, arrival) -> value


@dataclass(frozen=True)
class MetricRow:
    latex_label: str
    ascii_label: str
    getter: MetricGetter
    fmt_kind: str  # "signed_float", "unsigned_int", "signed_int"
    decimals: int = TABLE_DECIMALS
    scale: float = 1.0


def _fmt_metric_value(v: float, *, row: MetricRow, latex: bool) -> str:
    ns = nan_str(latex)
    if not is_finite(v):
        return ns
    vv = float(v) * float(row.scale)
    if row.fmt_kind == "signed_float":
        return fmt_signed(vv, row.decimals, ns)
    if row.fmt_kind == "unsigned_int":
        return fmt_unsigned_int(vv, ns)
    if row.fmt_kind == "signed_int":
        return fmt_signed_int(vv, ns)
    # fallback
    return fmt_signed(vv, row.decimals, ns)


def latex_metric_matrix_tables(
    *,
    out_path: Path,
    nodes_order: List[int],
    arrivals_order: List[float],
    modes_by_kmax: Dict[int, List[RowKey]],
    metrics_by_kmax: Dict[int, List[MetricRow]],
    include_defpreempt_suffix_in_header: bool,
) -> None:
    """
    Write one tabular per kmax, with:
      - columns grouped by arrival (mu_A) at top
      - modes under each arrival
      - rows are metrics
      - N shown as row blocks (N=16 then N=32)
    """
    lines: List[str] = []
    for kmax in sorted(metrics_by_kmax.keys()):
        if kmax not in modes_by_kmax:
            continue
        modes = modes_by_kmax[kmax]
        metrics = metrics_by_kmax[kmax]
        if not modes or not metrics:
            continue

        n_modes = len(modes)
        n_arr = len(arrivals_order)
        total_cols = 1 + n_modes * n_arr
        tab_spec = "l" + " c" * (total_cols - 1)

        # cmidrules per arrival group
        cmid = []
        for i in range(n_arr):
            start = 2 + i * n_modes
            end = start + n_modes - 1
            cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")

        # header rows
        arrival_hdr = " & " + " & ".join([rf"\multicolumn{{{n_modes}}}{{c}}{{{arrival_group_label(a, latex=True)}}}" for a in arrivals_order]) + r" \\"
        mode_labels = [rk.abbr(include_defpreempt_suffix_in_header) for _a in arrivals_order for rk in modes]
        mode_hdr = " & " + " & ".join(mode_labels) + r" \\"

        lines.append(rf"\begin{{tabular}}{{{tab_spec}}}")
        lines.append(r"\toprule")
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{{kmax_caption(kmax, latex=True)}}} \\")
        lines.append(r"\addlinespace[0.2em]")
        lines.append(arrival_hdr)
        lines.append("".join(cmid))
        lines.append(mode_hdr)
        lines.append(r"\midrule")

        for ni, n in enumerate(nodes_order):
            if ni > 0:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{${{N={n}}}$}} \\")
            lines.append(r"\midrule")
            for mrow in metrics:
                cells: List[str] = []
                for a in arrivals_order:
                    for rk in modes:
                        v = mrow.getter(kmax, rk, n, float(a))
                        cells.append(_fmt_metric_value(v, row=mrow, latex=True))
                lines.append(f"{mrow.latex_label} & " + " & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")  # spacing between tabulars

    out_path.write_text("\n".join(lines), encoding="utf-8")


def ascii_metric_matrix_tables(
    *,
    out_path: Path,
    nodes_order: List[int],
    arrivals_order: List[float],
    modes_by_kmax: Dict[int, List[RowKey]],
    metrics_by_kmax: Dict[int, List[MetricRow]],
    include_defpreempt_suffix_in_header: bool,
) -> None:
    def center(s: str, w: int) -> str:
        s = str(s)
        if len(s) >= w:
            return s
        pad = w - len(s)
        return " " * (pad // 2) + s + " " * (pad - pad // 2)

    lines: List[str] = []
    for kmax in sorted(metrics_by_kmax.keys()):
        if kmax not in modes_by_kmax:
            continue
        modes = modes_by_kmax[kmax]
        metrics = metrics_by_kmax[kmax]
        if not modes or not metrics:
            continue

        mode_labels = [rk.abbr(include_defpreempt_suffix_in_header) for _a in arrivals_order for rk in modes]
        arr_labels = [arrival_group_label(a, latex=False) for a in arrivals_order]

        # width calc
        row_w = max(12, max(len(m.ascii_label) for m in metrics) + 2)
        cell_w = 10
        total_cols = len(mode_labels)
        total_w = row_w + cell_w * total_cols
        sep = "-" * total_w

        lines.append(kmax_caption(kmax, latex=False))
        lines.append("")

        # arrival group header (approx centered over each group)
        hdr1 = " " * row_w
        for al in arr_labels:
            hdr1 += center(al, cell_w * len(modes))
        lines.append(hdr1.rstrip())

        # mode header
        hdr2 = " " * row_w
        for ml in mode_labels:
            hdr2 += center(ml, cell_w)
        lines.append(hdr2.rstrip())
        lines.append(sep)

        for n in nodes_order:
            lines.append(f"N={n}")
            lines.append(sep)
            for mrow in metrics:
                row = mrow.ascii_label.ljust(row_w)
                for a in arrivals_order:
                    for rk in modes:
                        v = mrow.getter(kmax, rk, n, float(a))
                        row += center(_fmt_metric_value(v, row=mrow, latex=False), cell_w)
                lines.append(row.rstrip())
            lines.append(sep)
            lines.append("")

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Plot helpers
# =============================================================================

YOfFn = Callable[[RowKey, int, float, int], float]      # (rk, nodes, arrival_s, kmax) -> y
LabelOfFn = Callable[[RowKey], str]                      # (rk) -> label


def symmetric_ylim_from_y_values(y_vals: List[float]) -> Tuple[float, float]:
    vals = [abs(float(y)) for y in y_vals if is_finite(y)]
    if not vals:
        return (-1.0, 1.0)
    m = max(vals)
    if m <= 0:
        return (-1.0, 1.0)
    pad = PLOT_Y_PADDING * m
    return (-(m + pad), (m + pad))


def plot_series(df: pd.DataFrame, *, kmax: int) -> List[RowKey]:
    """
    For plotting we keep:
      - defpreempt=1 configs
      - plus scheduling-failure (always defpreempt=0)
    """
    dff = df[df["kmax"].astype(int) == int(kmax)].copy()
    keep = (dff["defpreempt"].astype(int) == 1) | (dff["mode"].astype(str) == "scheduling-failure")
    dff = dff.loc[keep].drop_duplicates(
        subset=["job_name", "plugin_config", "mode", "blocking", "defpreempt", "nodes", "kmax", "arrival_s"],
        keep="first",
    )
    series = sorted(
        {
            RowKey(mode=m, blocking=int(b), defpreempt=int(d))
            for (m, b, d) in dff[["mode", "blocking", "defpreempt"]].drop_duplicates().itertuples(index=False, name=None)
        },
        key=lambda rk: rk.sort_key(),
    )
    return series


def collect_plot_y_vals(
    *,
    lookup: pd.DataFrame,
    df: pd.DataFrame,
    col: str,
    nodes_order: List[int],
    arrivals_order: List[float],
    kmax: int,
    scale: float = 1.0,
) -> List[float]:
    series = plot_series(df, kmax=kmax)
    y_vals: List[float] = []
    for rk in series:
        for a in arrivals_order:
            for n in nodes_order:
                v = lookup_value(lookup, nodes=n, kmax=kmax, arrival_s=a, rk=rk, col=col)
                if is_finite(v):
                    y_vals.append(float(v) * float(scale))
    return y_vals


def plot_dumbbell(
    *,
    out_dir: Path,
    filename_stem: str,
    y_label: str,
    nodes_order: List[int],
    arrivals_order: List[float],
    kmax: int,
    series: List[RowKey],
    y_of: YOfFn,
    label_of: LabelOfFn,
    ylim: Optional[Tuple[float, float]] = None,
    legend_loc: str = PLOT_LEGEND_LOC,
) -> None:
    if len(nodes_order) != 2:
        raise SystemExit(f"Dumbbell plots require exactly 2 distinct node values; found: {nodes_order}")

    out_dir.mkdir(parents=True, exist_ok=True)

    x_base = [i * PLOT_ARRIVAL_X_SPACING for i in range(len(arrivals_order))]
    m = max(1, len(series))

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    y_vals_for_limits: List[float] = []
    legend_handles: List[object] = []
    legend_labels: List[str] = []

    for i, rk in enumerate(series):
        mode_offset = (i - (m - 1) / 2.0) * PLOT_MODE_X_SPACING
        color = ax.plot([], [], linestyle="None")[0].get_color()  # advance cycle once per series
        label = label_of(rk)
        (h_legend,) = ax.plot([], [], linestyle="-", linewidth=1.8, color=color, label=label)
        legend_handles.append(h_legend)
        legend_labels.append(label)

        for xi, a in enumerate(arrivals_order):
            x = x_base[xi] + mode_offset
            y0 = y_of(rk, nodes_order[0], a, kmax)
            y1 = y_of(rk, nodes_order[1], a, kmax)
            y_vals_for_limits += [y0, y1]
            if is_finite(y0) and is_finite(y1):
                ax.plot([x, x], [y0, y1], linestyle="--", color=color, linewidth=1.0)
            if is_finite(y0):
                ax.plot([x], [y0], linestyle="None", marker="o", markersize=PLOT_MARKER_SIZE, color=color, label="_nolegend_")
            if is_finite(y1):
                ax.plot([x], [y1], linestyle="None", marker="s", markersize=PLOT_MARKER_SIZE, color=color, label="_nolegend_")

    ax.axhline(0.0, linewidth=0.8, color="black", linestyle="--", alpha=0.7)
    ax.set_ylabel(y_label, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_base)
    ax.set_xticklabels([arrival_group_label(a, latex=True) for a in arrivals_order])
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        ax.set_ylim(*symmetric_ylim_from_y_values([y for y in y_vals_for_limits if is_finite(y)]))

    yticks = ax.get_yticks()
    for y in yticks:
        if abs(y) < 1e-8:
            continue
        ax.axhline(y, linewidth=0.8, color="black", linestyle="--", alpha=0.25, zorder=0)

    add_standard_legend(ax=ax, handles=legend_handles, labels=legend_labels, loc=legend_loc)
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename_stem}.png", dpi=200)
    fig.savefig(out_dir / f"{filename_stem}.pdf")
    plt.close(fig)


# =============================================================================
# Main + CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate tables and figures from sealed results.")
    p.add_argument("--in-results", type=Path, required=True, help="Path to sealed results_paired.csv.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory (tables + figures subdirs).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.in_results))
    lookup = build_lookup(df)

    nodes_order = sorted(df["nodes"].dropna().astype(int).unique().tolist())
    arrivals_order = sorted(df["arrival_s"].dropna().astype(float).unique().tolist())
    kmaxs = sorted(df["kmax"].dropna().astype(int).unique().tolist())

    if len(nodes_order) != 2:
        raise SystemExit(f"Expected exactly 2 node values for plots; found {nodes_order}")
    if not arrivals_order:
        raise SystemExit("No arrivals inferred from results_paired.csv")
    if not kmaxs:
        raise SystemExit("No kmax values found")

    # Build mode columns (per table type) with stable ordering.
    def _sorted_modes(rks: List[RowKey]) -> List[RowKey]:
        rks2 = sorted(rks, key=lambda rk: rk.sort_key())
        # Ensure SF is first if present
        return rks2

    # Modes for "vs baseline" tables: defpreempt=1 plus SF (SF is defpreempt=0)
    modes_vs_baseline: Dict[int, List[RowKey]] = {}
    for k in kmaxs:
        dff = df[df["kmax"].astype(int) == k]
        rks = [RowKey.from_plugin_config(pc) for pc in dff["plugin_config"].unique().tolist()]
        keep = [rk for rk in rks if (rk.defpreempt == 1) or (rk.mode == "scheduling-failure")]
        modes_vs_baseline[k] = _sorted_modes(keep)

    # Modes for "w/o default preemption vs baseline" big table:
    # defpreempt=0 only, and (per your earlier preference) DO NOT include scheduling-failure here.
    modes_vs_baseline_nodf: Dict[int, List[RowKey]] = {}
    for k in kmaxs:
        dff = df[df["kmax"].astype(int) == k]
        rks = [RowKey.from_plugin_config(pc) for pc in dff["plugin_config"].unique().tolist()]
        keep = [rk for rk in rks if (rk.defpreempt == 0) and (rk.mode != "scheduling-failure")]
        modes_vs_baseline_nodf[k] = _sorted_modes(keep)

    # Modes for within-plugin deltas: only modes that have both defpreempt 0/1 (no SF)
    def_rows = build_defpreempt_rows(df)
    modes_def_delta: Dict[int, List[RowKey]] = {}
    for k in kmaxs:
        modes_def_delta[k] = _sorted_modes([RowKey(mode=m, blocking=b, defpreempt=1) for (m, b) in def_rows])

    # -----------------------------
    # Metric getters
    # -----------------------------
    def g(col: str) -> MetricGetter:
        return lambda kmax, rk, n, a: lookup_value(lookup, nodes=n, kmax=kmax, arrival_s=a, rk=rk, col=col)

    def gd(col: str) -> MetricGetter:
        # defpreempt delta: (def0)-(def1) for rk.mode/rk.blocking; rk.defpreempt ignored
        return lambda kmax, rk, n, a: defpreempt_delta(
            lookup, nodes=n, kmax=kmax, arrival_s=a, mode=rk.mode, blocking=rk.blocking, col=col
        )

    # -----------------------------
    # Metrics per table
    # -----------------------------
    def metrics_big(kmax: int, *, within_def_delta: bool = False) -> List[MetricRow]:
        # Getter set depends on whether it's within-plugin delta or vs-baseline.
        GG = gd if within_def_delta else g

        rows: List[MetricRow] = [
            MetricRow(r"$\Delta U$", "ΔU", GG("delta_U_eff_run_mean"), "signed_float", decimals=1, scale=100.0),
        ]

        # Latency rows
        rows.append(MetricRow(r"$\Delta L_{\mathrm{tot}}$", "ΔL_tot", GG("delta_L_s_total_mean"), "signed_float", decimals=1))
        if kmax != 1:
            for p in range(1, 5):
                rows.append(MetricRow(rf"$\Delta L_{{p_{p}}}$", f"ΔL_p{p}", GG(f"delta_L_s_p{p}_mean"), "signed_float", decimals=1))

        # Deletions rows
        rows.append(MetricRow(r"$\Delta D_{\mathrm{tot}}$", "ΔD_tot", GG("delta_D_total_mean"), "signed_float", decimals=1))
        if kmax != 1:
            for p in range(1, 5):
                rows.append(MetricRow(rf"$\Delta D_{{p_{p}}}$", f"ΔD_p{p}", GG(f"delta_D_p{p}_mean"), "signed_float", decimals=1))

        # Optimizations (#O): in vs-baseline tables it's a non-negative count. In within-plugin deltas it can be signed.
        if within_def_delta:
            rows.append(MetricRow(r"$\#O$", "#O", gd("plan_activated_mean"), "signed_int"))
        else:
            rows.append(MetricRow(r"$\#O$", "#O", g("plan_activated_mean"), "unsigned_int"))
        return rows

    def metrics_L_only(kmax: int) -> List[MetricRow]:
        rows = [MetricRow(r"$\Delta L_{\mathrm{tot}}$", "ΔL_tot", gd("delta_L_s_total_mean"), "signed_float", decimals=1)]
        if kmax != 1:
            for p in range(1, 5):
                rows.append(MetricRow(rf"$\Delta L_{{p_{p}}}$", f"ΔL_p{p}", gd(f"delta_L_s_p{p}_mean"), "signed_float", decimals=1))
        return rows

    def metrics_D_only(kmax: int) -> List[MetricRow]:
        rows = [MetricRow(r"$\Delta D_{\mathrm{tot}}$", "ΔD_tot", gd("delta_D_total_mean"), "signed_float", decimals=1)]
        if kmax != 1:
            for p in range(1, 5):
                rows.append(MetricRow(rf"$\Delta D_{{p_{p}}}$", f"ΔD_p{p}", gd(f"delta_D_p{p}_mean"), "signed_float", decimals=1))
        return rows

    def metrics_U_only(kmax: int) -> List[MetricRow]:
        return [MetricRow(r"$\Delta U$", "ΔU", g("delta_U_eff_run_mean"), "signed_float", decimals=1, scale=100.0)]

    def metrics_R(kmax: int) -> List[MetricRow]:
        rows = [MetricRow(r"$\Delta R_{\mathrm{tot}}$", "ΔR_tot", g("delta_R_total_mean"), "signed_float", decimals=1)]
        if kmax != 1:
            for p in range(1, 5):
                rows.append(MetricRow(rf"$\Delta R_{{p_{p}}}$", f"ΔR_p{p}", g(f"delta_R_p{p}_mean"), "signed_float", decimals=1))
        return rows

    def metrics_solver_attempts(_kmax: int) -> List[MetricRow]:
        return [MetricRow(r"$\#S$", "#S", g("solver_attempts_mean"), "unsigned_int")]

    def metrics_plans_activated(_kmax: int) -> List[MetricRow]:
        return [MetricRow(r"$\#O$", "#O", g("plan_activated_mean"), "unsigned_int")]

    # -----------------------------
    # Build metrics_by_kmax per output table
    # -----------------------------
    def build_metrics_map(builder: Callable[[int], List[MetricRow]]) -> Dict[int, List[MetricRow]]:
        return {k: builder(k) for k in kmaxs}

    # -----------------------------
    # Write tables (LaTeX + ASCII)
    # -----------------------------
    TABLE_JOBS = [
        # small tables (still same files, but new layout)
        ("delta_U_eff_run", modes_vs_baseline, build_metrics_map(metrics_U_only), True),
        ("delta_R", modes_vs_baseline, build_metrics_map(metrics_R), True),
        ("delta_D", modes_vs_baseline, build_metrics_map(lambda k: [MetricRow(r"$\Delta D_{\mathrm{tot}}$", "ΔD_tot", g("delta_D_total_mean"), "signed_float", decimals=1)]
                                                        + ([] if k == 1 else [MetricRow(rf"$\Delta D_{{p_{p}}}$", f"ΔD_p{p}", g(f"delta_D_p{p}_mean"), "signed_float", decimals=1) for p in range(1, 5)])), True),
        ("delta_L", modes_vs_baseline, build_metrics_map(lambda k: [MetricRow(r"$\Delta L_{\mathrm{tot}}$", "ΔL_tot", g("delta_L_s_total_mean"), "signed_float", decimals=1)]
                                                        + ([] if k == 1 else [MetricRow(rf"$\Delta L_{{p_{p}}}$", f"ΔL_p{p}", g(f"delta_L_s_p{p}_mean"), "signed_float", decimals=1) for p in range(1, 5)])), True),
        ("solver_attempts", modes_vs_baseline, build_metrics_map(metrics_solver_attempts), True),
        ("plans_activated", modes_vs_baseline, build_metrics_map(metrics_plans_activated), True),

        # big table vs baseline (defpreempt=1 + SF), rows are the full metric set
        ("big_table_util_latency_deletions_optimizations",
         modes_vs_baseline,
         {k: metrics_big(k, within_def_delta=False) for k in kmaxs},
         True),

        # within-plugin deltas: (def0)-(def1)
        ("delta_D_without_defaultpreemption",
         modes_def_delta,
         {k: metrics_D_only(k) for k in kmaxs},
         False),

        ("delta_L_without_defaultpreemption",
         modes_def_delta,
         {k: metrics_L_only(k) for k in kmaxs},
         False),

        # big table: defpreempt=0 vs baseline (no SF), show -NDF suffix in headers
        ("big_table_without_defaultpreemption_util_latency_deletions_optimizations",
         modes_vs_baseline_nodf,
         {k: metrics_big(k, within_def_delta=False) for k in kmaxs},
         True),
    ]

    for stem, modes_map, metrics_map, include_defpreempt_suffix in TABLE_JOBS:
        latex_metric_matrix_tables(
            out_path=tables_dir / f"{stem}.tex",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            modes_by_kmax=modes_map,
            metrics_by_kmax=metrics_map,
            include_defpreempt_suffix_in_header=include_defpreempt_suffix,
        )
        ascii_metric_matrix_tables(
            out_path=tables_dir / f"{stem}.txt",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            modes_by_kmax=modes_map,
            metrics_by_kmax=metrics_map,
            include_defpreempt_suffix_in_header=include_defpreempt_suffix,
        )

    # -----------------------------
    # Figures
    # -----------------------------
    util_y_vals_all: List[float] = []
    deletions_y_vals_all: List[float] = []
    latency_y_vals_all: List[float] = []

    for k in kmaxs:
        util_y_vals_all += collect_plot_y_vals(
            lookup=lookup, df=df, col="delta_U_eff_run_mean",
            nodes_order=nodes_order, arrivals_order=arrivals_order, kmax=k, scale=100.0,
        )
        deletions_y_vals_all += collect_plot_y_vals(
            lookup=lookup, df=df, col="delta_D_total_mean",
            nodes_order=nodes_order, arrivals_order=arrivals_order, kmax=k, scale=1.0,
        )
        latency_y_vals_all += collect_plot_y_vals(
            lookup=lookup, df=df, col="delta_L_s_total_mean",
            nodes_order=nodes_order, arrivals_order=arrivals_order, kmax=k, scale=1.0,
        )

    util_ylim = symmetric_ylim_from_y_values(util_y_vals_all)
    deletions_ylim = symmetric_ylim_from_y_values(deletions_y_vals_all)
    latency_ylim = symmetric_ylim_from_y_values(latency_y_vals_all)

    def y_from_col(*, col: str, scale: float) -> YOfFn:
        def _y(rk: RowKey, nodes: int, a: float, kmax: int) -> float:
            v = lookup_value(lookup, nodes=nodes, kmax=kmax, arrival_s=a, rk=rk, col=col)
            return float(v) * float(scale) if is_finite(v) else float("nan")
        return _y

    def y_defpreempt_delta(*, col: str) -> YOfFn:
        def _y(rk: RowKey, nodes: int, a: float, kmax: int) -> float:
            if rk.mode == "scheduling-failure":
                return float("nan")
            return defpreempt_delta(
                lookup, nodes=nodes, kmax=kmax, arrival_s=a,
                mode=rk.mode, blocking=rk.blocking, col=col,
            )
        return _y

    # Series for standard plots: defpreempt=1 (+SF)
    for plot_kmax in kmaxs:
        series = plot_series(df, kmax=plot_kmax)

        plot_dumbbell(
            out_dir=figures_dir,
            filename_stem=f"delta_U_eff_run_kmax{plot_kmax}",
            y_label=r"$\Delta U_{\mathrm{eff}}$ (pp)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=plot_kmax,
            series=series,
            y_of=y_from_col(col="delta_U_eff_run_mean", scale=100.0),
            label_of=lambda rk: rk.abbr(include_defpreempt_suffix=True),
            ylim=util_ylim,
        )

        plot_dumbbell(
            out_dir=figures_dir,
            filename_stem=f"delta_D_kmax{plot_kmax}",
            y_label=r"$\Delta D$ (#deletions)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=plot_kmax,
            series=series,
            y_of=y_from_col(col="delta_D_total_mean", scale=1.0),
            label_of=lambda rk: rk.abbr(include_defpreempt_suffix=True),
            ylim=deletions_ylim,
            legend_loc="upper right",
        )

        plot_dumbbell(
            out_dir=figures_dir,
            filename_stem=f"delta_L_kmax{plot_kmax}",
            y_label=r"$\Delta L$ (seconds)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=plot_kmax,
            series=series,
            y_of=y_from_col(col="delta_L_s_total_mean", scale=1.0),
            label_of=lambda rk: rk.abbr(include_defpreempt_suffix=True),
            ylim=latency_ylim,
        )

        # Within-plugin delta plots (def0 - def1) for the four modes (no SF)
        def_series = [RowKey(mode=m, blocking=b, defpreempt=1) for (m, b) in def_rows]
        def_series.sort(key=lambda rk: rk.sort_key())

        # Compute ylim per kmax/col for delta plots
        yvals_d = []
        yvals_l = []
        for rk in def_series:
            for a in arrivals_order:
                for n in nodes_order:
                    yvals_d.append(defpreempt_delta(lookup, nodes=n, kmax=plot_kmax, arrival_s=a, mode=rk.mode, blocking=rk.blocking, col="delta_D_total_mean"))
                    yvals_l.append(defpreempt_delta(lookup, nodes=n, kmax=plot_kmax, arrival_s=a, mode=rk.mode, blocking=rk.blocking, col="delta_L_s_total_mean"))
        ylim_d = symmetric_ylim_from_y_values(yvals_d)
        ylim_l = symmetric_ylim_from_y_values(yvals_l)

        plot_dumbbell(
            out_dir=figures_dir,
            filename_stem=f"delta_D_kmax{plot_kmax}_without_defaultpreemption",
            y_label=r"$\Delta D_{\mathrm{w/o\ defpreempt}}$ (#deletions)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=plot_kmax,
            series=def_series,
            y_of=y_defpreempt_delta(col="delta_D_total_mean"),
            label_of=lambda rk: rk.abbr(include_defpreempt_suffix=False),
            ylim=ylim_d,
            legend_loc="upper right",
        )

        plot_dumbbell(
            out_dir=figures_dir,
            filename_stem=f"delta_L_kmax{plot_kmax}_without_defaultpreemption",
            y_label=r"$\Delta L_{\mathrm{w/o\ defpreempt}}$ (seconds)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=plot_kmax,
            series=def_series,
            y_of=y_defpreempt_delta(col="delta_L_s_total_mean"),
            label_of=lambda rk: rk.abbr(include_defpreempt_suffix=False),
            ylim=ylim_l,
        )

    print(f"Wrote tables to:  {tables_dir}")
    print(f"Wrote figures to: {figures_dir}")


if __name__ == "__main__":
    main()
