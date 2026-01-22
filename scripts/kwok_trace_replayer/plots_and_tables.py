#!/usr/bin/env python3
# scripts/kwok_trace_replayer/plots_and_tables.py
"""
python -m scripts.kwok_trace_replayer.plots_and_tables --in-results analysis/kwok_trace_replayer/results_paired.csv --out-dir analysis/kwok_trace_replayer

Produces:
  Tables (under <out-dir>/tables):
    - delta_u_eff_run.{txt,tex}
    - delta_R.{txt,tex}
    - delta_D.{txt,tex}
    - delta_L.{txt,tex}
    - solver_attempts.{txt,tex}
    - plans_activated.{txt,tex}
    - big_table_latency_util.{txt,tex}
    - delta_D_without_defaultpreemption.{txt,tex}
    - delta_L_without_defaultpreemption.{txt,tex}
    - big_table_without_defaultpreemption_deletions_latency.{txt,tex}

  Figures (under <out-dir>/figures):
    - delta_u_eff_run_kmax<K>.<png|pdf>
    - delta_L_total_kmax<K>.<png|pdf>
    - delta_D_without_defaultpreemption_kmax<K>.<png|pdf>
    - delta_L_without_defaultpreemption_kmax<K>.<png|pdf>
"""

import argparse, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

TABLE_DECIMALS: int = 1 # for table formatting
JOB_RE = re.compile(r"nodes=(\d+)_prio=(\d+)_arrival=([0-9.]+)s")

MODE_ORDER: List[Tuple[str, Optional[int], str]] = [
    ("scheduling-failure", None, "Scheduling-failure"),
    ("periodic8s", 1, "Periodic-8s (blk)"),
    ("periodic8s", 0, "Periodic-8s (non-blk)"),
    ("stable-queue-2s", 1, "Stable-queue-2s (blk)"),
    ("stable-queue-2s", 0, "Stable-queue-2s (non-blk)"),
]
MODE_RANK: Dict[Tuple[str, Optional[int]], int] = {(m, b): i for i, (m, b, _lbl) in enumerate(MODE_ORDER)}
MODE_LABEL: Dict[Tuple[str, Optional[int]], str] = {(m, b): lbl for (m, b, lbl) in MODE_ORDER}

EXPECTED_COLS = [
    "job_name",
    "plugin_config",
    "delta_util_eff_run_mean",
    "delta_R_p1_mean", "delta_R_p2_mean", "delta_R_p3_mean", "delta_R_p4_mean", "delta_R_total_mean",
    "delta_D_p1_mean", "delta_D_p2_mean", "delta_D_p3_mean", "delta_D_p4_mean", "delta_D_total_mean",
    "delta_L_s_p1_mean", "delta_L_s_p2_mean", "delta_L_s_p3_mean", "delta_L_s_p4_mean", "delta_L_s_total_mean",
    "solver_attempts_mean", "plan_activated_mean",
]
KEY_COLS = ["nodes", "kmax", "arrival_s", "mode", "blocking", "defpreempt"]

# -----------------------------
# Plot styling
# -----------------------------

PLOT_MARKER_SIZE = 3.0
PLOT_ARRIVAL_X_SPACING = 0.6
PLOT_MODE_X_SPACING = 0.05

PLOT_FIGSIZE = (3.1, 2.2)
PLOT_TITLE_FONTSIZE = 8
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
    for handle, label in zip(handles, labels):
        s = str(label)
        if not s or s.startswith("_"):
            continue
        if s in seen:
            continue
        seen.add(s)
        clean.append((handle, s))

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

def display_mode(mode: str) -> str:
    """
    Better display label for mode strings.
    """
    s = canonical_mode(mode)
    if s == "scheduling-failure":
        return "Scheduling-failure"
    m = re.fullmatch(r"periodic([0-9.]+)s", s)
    if m:
        return f"Periodic-{m.group(1)}s"
    m = re.fullmatch(r"stable-queue-?([0-9.]+)s", s)
    if m:
        return f"Stable-queue-{m.group(1)}s"
    if s:
        return s[:1].upper() + s[1:]
    return "Unknown"


@dataclass(frozen=True)
class RowKey:
    mode: str
    blocking: int
    defpreempt: int

    @staticmethod
    def from_plugin_config(plugin_config: str) -> "RowKey":
        """
        Parse plugin_config into RowKey.
        """
        kv = parse_plugin_config(plugin_config)
        mode = canonical_mode(kv.get("mode", "unknown"))
        if mode == "scheduling-failure": # scheduling-failure has no blocking label; normalize to 0
            blocking = 0
        else:
            blocking = 1 if str(kv.get("blocking", "0")).lower() in {"1", "true", "yes"} else 0
        try:
            defpreempt = int(str(kv.get("defpreempt", "0")))
        except Exception:
            defpreempt = 0
        return RowKey(mode=mode, blocking=blocking, defpreempt=defpreempt)

    def base_label(self) -> str:
        if self.mode == "scheduling-failure":
            return "Scheduling-failure"
        label = MODE_LABEL.get((self.mode, self.blocking))
        if label is not None:
            return label
        base = display_mode(self.mode)
        blocking = "blk" if self.blocking == 1 else "non-blk"
        return f"{base} ({blocking})"

    def label(self, include_defpreempt: bool = True) -> str:
        """
        Full label for rows, optionally including defpreempt suffix.
        """
        base = self.base_label()
        if self.mode == "scheduling-failure":
            return base
        # Convention:
        #   defpreempt=1  => no suffix
        #   defpreempt=0  => ", w/o defpreempt"
        if include_defpreempt and self.defpreempt == 0:
            if base.endswith(")"):
                return base[:-1] + ", w/o defpreempt)"
            return base + " (w/o defpreempt)"
        return base

    def sort_key(self) -> Tuple[int, int, str]:
        """
        Sorting key for rows:
        1) by (mode, blocking) per MODE_RANK
        2) defpreempt=1 before defpreempt=0
        3) finally by mode string lex order
        """
        base_rank = MODE_RANK.get((self.mode, self.blocking))
        if base_rank is None:
            base_rank = MODE_RANK.get((self.mode, None))
        # defpreempt=1 before defpreempt=0
        def_rank = 0 if self.defpreempt == 1 else 1
        if base_rank is None:
            m = self.mode
            if m.startswith("periodic"):
                group = 10
                num = float(re.sub(r"[^0-9.]", "", m) or "0")
            elif m.startswith("stable-queue"):
                group = 11
                num = float(re.sub(r"[^0-9.]", "", m) or "0")
            else:
                group = 99
                num = 0.0
            base_rank = 1000 + group * 100 + int(round(num * 10))

        return (base_rank, def_rank, self.mode)

# =============================================================================
# Formatting
# =============================================================================

def is_finite(x: object) -> bool:
    """
    Return True if x is a finite number.
    """
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def normalize_neg_zero(v: float) -> float:
    """
    Normalize -0.0 to 0.0 for display purposes.
    """
    return 0.0 if abs(v) < 0.5e-12 else v

def nan_str(latex: bool) -> str:
    """
    Return NaN string for tables.
    """
    return (r"\text{--}" if latex else "--")

def fmt_signed(x: object, decimals: int, nan_s: str) -> str:
    """
    Format signed float with given decimals, normalizing -0.0 to 0.0.
    """
    if not is_finite(x):
        return nan_s
    v = normalize_neg_zero(float(x))
    return f"{v:+.{decimals}f}"

def fmt_count(x: object, nan_s: str) -> str:
    """
    Format count as signed integer, handling NaN.
    """
    if not is_finite(x):
        return nan_s
    return f"{int(round(float(x))):+d}"

def fmt_vec(values: Sequence[object], decimals: int, nan_s: str, latex: bool) -> str:
    """
    Format a vector of signed floats.
    """
    if any(not is_finite(v) for v in values):
        return nan_s
    parts = [fmt_signed(v, decimals=decimals, nan_s=nan_s) for v in values]
    inside = ",".join(parts)
    if latex:
        return r"{\scriptsize$\langle" + inside + r"\rangle$}"
    return "<" + inside + ">"

def arrival_label(a: float, *, latex: bool) -> str:
    """
    Return µ_A label for arrival s.
    """
    a_i = int(a) if abs(a - round(a)) < 1e-9 else a
    if latex:
        return rf"$\mu_A{{=}}{a_i}\,\mathrm{{s}}$"
    return f"µ_A={a_i}s"

# =============================================================================
# Data access
# =============================================================================

def load_results(results_csv: Path) -> pd.DataFrame:
    """
    Load and parse results CSV into DataFrame with extra parsed columns.
    """
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
    """
    Build a MultiIndex lookup table for O(1) cell access.
    """
    d = df.copy()
    d = d.drop_duplicates(subset=KEY_COLS, keep="first")
    return d.set_index(KEY_COLS).sort_index()

def lookup_value(
    lookup: pd.DataFrame,
    *,
    nodes: int,
    kmax: int,
    arrival_s: float,
    mode: str,
    blocking: int,
    defpreempt: int,
    col: str,
) -> float:
    """
    Lookup value from the DataFrame by key.
    """
    key = (int(nodes), int(kmax), float(arrival_s), str(mode), int(blocking), int(defpreempt))
    try:
        return float(lookup.at[key, col])
    except KeyError:
        return float("nan")

def value_at(
    lookup: pd.DataFrame,
    *,
    rk: RowKey,
    nodes: int,
    kmax: int,
    arrival_s: float,
    col: str,
) -> float:
    """
    Return value at given parameters.
    It uses rk to extract mode, blocking, defpreempt.
    """
    return lookup_value(
        lookup,
        nodes=nodes, kmax=kmax, arrival_s=arrival_s,
        mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
        col=col,
    )

# =============================================================================
# Special: defpreempt delta within plugin configs
# =============================================================================

def build_defpreempt_rows(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """
    Return (mode, blocking) pairs that have BOTH defpreempt=0 and defpreempt=1.
    Excludes scheduling-failure as it always disables defaultpreemption.
    """
    seen: Dict[Tuple[str, int], set] = {}
    for pc in df["plugin_config"].unique().tolist():
        rk = RowKey.from_plugin_config(pc)
        seen.setdefault((rk.mode, rk.blocking), set()).add(rk.defpreempt)
    rows = [
        (m, b)
        for (m, b), defs in seen.items()
        if (0 in defs and 1 in defs) and (m != "scheduling-failure")
    ]
    rows.sort(key=lambda mb: RowKey(mode=mb[0], blocking=mb[1], defpreempt=1).sort_key())
    return rows

def defpreempt_delta_total(
    lookup: pd.DataFrame,
    *,
    kmax: int,
    mode: str,
    blocking: int,
    nodes: int,
    arrival_s: float,
    total_col: str,
) -> float:
    """
    Return (defpreempt=0) - (defpreempt=1) for total_col.
    """
    v0 = lookup_value(lookup, nodes=nodes, kmax=kmax, arrival_s=arrival_s, mode=mode, blocking=blocking, defpreempt=0, col=total_col)
    v1 = lookup_value(lookup, nodes=nodes, kmax=kmax, arrival_s=arrival_s, mode=mode, blocking=blocking, defpreempt=1, col=total_col)
    return (v0 - v1) if (is_finite(v0) and is_finite(v1)) else float("nan")

def cell_defpreempt_delta_total_or_vec(
    lookup: pd.DataFrame,
    *,
    kmax: int,
    mode: str,
    blocking: int,
    nodes: int,
    arrival_s: float,
    total_col: str,
    part_col_tpl: str,
    decimals: int,
    latex: bool,
) -> str:
    """
    Return (defpreempt=0) - (defpreempt=1) for total or vector cell value depending on kmax.
    """
    ns = nan_str(latex)
    if kmax == 1:
        dd = defpreempt_delta_total(
            lookup,
            kmax=kmax, mode=mode, blocking=blocking,
            nodes=nodes, arrival_s=arrival_s,
            total_col=total_col,
        )
        return fmt_signed(dd, decimals, ns)
    vals0 = [
        lookup_value(lookup, nodes=nodes, kmax=kmax, arrival_s=arrival_s, mode=mode, blocking=blocking, defpreempt=0, col=part_col_tpl.format(p=p))
        for p in range(1, 5)
    ]
    vals1 = [
        lookup_value(lookup, nodes=nodes, kmax=kmax, arrival_s=arrival_s, mode=mode, blocking=blocking, defpreempt=1, col=part_col_tpl.format(p=p))
        for p in range(1, 5)
    ]
    if any(not is_finite(v) for v in vals0) or any(not is_finite(v) for v in vals1):
        return ns
    diff = [float(v0) - float(v1) for v0, v1 in zip(vals0, vals1)]
    return fmt_vec(diff, decimals, ns, latex=latex)

# =============================================================================
# Plot helpers
# =============================================================================

YOfFn = Callable[[RowKey, int, float, int], float]          # (rk, nodes, arrival_s, kmax) -> y
LabelOfFn = Callable[[RowKey], str]                         # (rk) -> legend label

def symmetric_ylim_from_y_values(y_vals: List[float]) -> Tuple[float, float]:
    """
    Return symmetric ylim (min, max) given y values.
    Adds 10% padding.
    """
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
    Return sorted series of RowKey for plotting for given kmax.
    Includes only rows with defpreempt=1 or scheduling-failure mode.
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
    """
    Collect y values for all series, nodes, arrivals for plotting.
    1) series ordered by plot_series()
    2) for each series, nodes in nodes_order
         for each nodes, arrivals in arrivals_order
    3) scale each y by scale factor
    """
    series = plot_series(df, kmax=kmax)
    y_vals: List[float] = []
    for rk in series:
        for a in arrivals_order:
            for n in nodes_order:
                y = value_at(lookup, rk=rk, nodes=n, kmax=kmax, arrival_s=a, col=col)
                if is_finite(y):
                    y_vals.append(float(y) * float(scale))
    return y_vals

def plot_dumbbell(
    *,
    out_dir: Path,
    filename_stem: str,
    title: str,
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
    """
    Dumbbell plot:
      - x-axis: arrivals
      - per series: two markers per x (nodes_order[0] as circle, nodes_order[1] as square)
      - optional dashed connector between the two markers.
    """
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
        color = ax.plot([], [], linestyle="None")[0].get_color() # advance the color cycle exactly once per series
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

    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_base)
    ax.set_xticklabels([arrival_label(a, latex=True) for a in arrivals_order])
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        ax.set_ylim(*symmetric_ylim_from_y_values([y for y in y_vals_for_limits if is_finite(y)]))

    add_standard_legend(ax=ax, handles=legend_handles, labels=legend_labels, loc=legend_loc)
    fig.tight_layout()

    fig.savefig(out_dir / f"{filename_stem}.png", dpi=200)
    fig.savefig(out_dir / f"{filename_stem}.pdf")
    plt.close(fig)

def collect_defpreempt_delta_total_y_vals(
    *,
    lookup: pd.DataFrame,
    nodes_order: List[int],
    arrivals_order: List[float],
    kmax: int,
    total_col: str,
    default_rows: List[Tuple[str, int]],
) -> List[float]:
    """
    Collect y values for defpreempt delta total for plotting.
    """
    y_vals: List[float] = []
    for mode, blocking in default_rows:
        for a in arrivals_order:
            for n in nodes_order:
                y = defpreempt_delta_total(
                    lookup,
                    kmax=kmax, mode=mode, blocking=blocking,
                    nodes=n, arrival_s=a, total_col=total_col,
                )
                if is_finite(y):
                    y_vals.append(float(y))
    return y_vals

# =============================================================================
# Table writers (ASCII + LaTeX tables)
# =============================================================================

RowLabelFn = Callable[[RowKey], str]

def latex_table(
    *,
    out_path: Path,
    title_math: str,
    subtitle: str,
    nodes_order: List[int],
    arrivals_order: List[float],
    rows_by_kmax: Dict[int, List[RowKey]],
    cell_fn,  # (kmax, rowkey, nodes, arrival) -> str
    row_label_fn: Optional[RowLabelFn] = None,
) -> None:
    """
    Write a LaTeX table to the specified output path.
    """
    row_label_fn = row_label_fn or (lambda rk: rk.label(include_defpreempt=True))

    num_nodes = len(nodes_order)
    num_arrivals = len(arrivals_order)
    if num_nodes < 1 or num_arrivals < 1:
        raise SystemExit("Need at least 1 node value and 1 arrival value to write tables.")

    total_cols = 1 + num_nodes * num_arrivals  # row label + data cells
    tab_spec = "l" + " c" * (total_cols - 1)

    # cmidrules for node groupings
    cmid = []
    for i in range(num_nodes):
        start = 2 + i * num_arrivals
        end = start + num_arrivals - 1
        cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")

    # node header row
    node_header_parts = [rf"\multicolumn{{{num_arrivals}}}{{c}}{{${{N={n}}}$}}" for n in nodes_order]
    node_header = "& " + " & ".join(node_header_parts) + r" \\"

    # arrival header row
    arrival_header_cells = [arrival_label(a, latex=True) for _n in nodes_order for a in arrivals_order]
    arrival_header = "& " + " & ".join(arrival_header_cells) + r" \\"

    lines: List[str] = []
    lines += [
        rf"\begin{{tabular}}{{{tab_spec}}}",
        r"\toprule",
        rf"\multicolumn{{{total_cols}}}{{l}}{{$\mathbf{{{title_math}}}$ {subtitle}}} \\",
        r"\addlinespace[0.2em]",
        node_header,
        "".join(cmid),
        arrival_header,
        r"\midrule",
    ]

    for kmax in sorted(rows_by_kmax.keys()):
        k_title = r"$k_{\max}=1$ (no priorities)" if kmax == 1 else rf"$k_{{\max}}={kmax}$ (priorities enabled)"
        lines += [
            rf"\multicolumn{{{total_cols}}}{{l}}{{{k_title}}} \\",
            r"\midrule",
        ]
        for rk in rows_by_kmax[kmax]:
            row_label = row_label_fn(rk)
            cells: List[str] = []
            for n in nodes_order:
                for a in arrivals_order:
                    cells.append(cell_fn(kmax, rk, n, a))
            lines.append(row_label + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    if lines and lines[-1] == r"\midrule":
        lines.pop()

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")

def ascii_table(
    *,
    out_path: Path,
    title: str,
    nodes_order: List[int],
    arrivals_order: List[float],
    rows_by_kmax: Dict[int, List[RowKey]],
    cell_fn,  # (kmax, rowkey, nodes, arrival) -> str
    row_w: Optional[int] = None,
    col_w: Optional[int] = None,
    row_label_fn: Optional[RowLabelFn] = None,
) -> None:
    """
    Write an ASCII table to the specified output path.
    """
    row_label_fn = row_label_fn or (lambda rk: rk.label(include_defpreempt=True))

    if not nodes_order or not arrivals_order:
        raise SystemExit("Need at least 1 node value and 1 arrival value to write tables.")

    # Width pass for ensuring proper alignment
    max_label = 0
    for rks in rows_by_kmax.values():
        for rk in rks:
            max_label = max(max_label, len(str(row_label_fn(rk))))
    required_row_w = max(24, max_label + 2)
    row_w = required_row_w if row_w is None else max(int(row_w), required_row_w)

    max_cell = 0
    for kmax, rks in rows_by_kmax.items():
        for rk in rks:
            for n in nodes_order:
                for a in arrivals_order:
                    max_cell = max(max_cell, len(str(cell_fn(kmax, rk, n, a))))
    max_hdr = max(len(arrival_label(a, latex=False)) for a in arrivals_order)
    required_col_w = max(10, max(max_cell, max_hdr) + 2)
    col_w = required_col_w if col_w is None else max(int(col_w), required_col_w)

    def center(s: str, w: int) -> str:
        s = str(s)
        if len(s) >= w:
            return s
        pad = w - len(s)
        return " " * (pad // 2) + s + " " * (pad - pad // 2)

    lines: List[str] = []
    lines.append(title)
    lines.append("")

    hdr1 = " " * row_w
    for n in nodes_order:
        hdr1 += center(f"N={n}", col_w * len(arrivals_order))
    lines.append(hdr1.rstrip())

    hdr2 = " " * row_w
    for _n in nodes_order:
        for a in arrivals_order:
            hdr2 += center(arrival_label(a, latex=False), col_w)
    lines.append(hdr2.rstrip())

    sep = "-" * (row_w + col_w * len(nodes_order) * len(arrivals_order))
    lines.append(sep)

    for kmax in sorted(rows_by_kmax.keys()):
        lines.append(f"kmax={kmax} ({'no priorities' if kmax == 1 else 'with priorities'})")
        lines.append(sep)
        for rk in rows_by_kmax[kmax]:
            label = str(row_label_fn(rk)).ljust(row_w)
            row = label
            for n in nodes_order:
                for a in arrivals_order:
                    row += center(cell_fn(kmax, rk, n, a), col_w)
            lines.append(row.rstrip())
        lines.append(sep)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_both(*, stem: str, out_dir: Path, latex_args: dict, ascii_args: dict) -> None:
    """
    Write both LaTeX and ASCII tables to out_dir with given stem.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    latex_table(out_path=out_dir / f"{stem}.tex", **latex_args)
    ascii_table(out_path=out_dir / f"{stem}.txt", **ascii_args)

# =============================================================================
# Cell builders for tables to get cell values.
# =============================================================================

def cell_signed(
    lookup: pd.DataFrame,
    *,
    kmax: int,
    rk: RowKey,
    nodes: int,
    arrival_s: float,
    col: str,
    decimals: int,
    latex: bool,
    scale: float = 1.0,
) -> str:
    """
    Return signed float cell value.
    """
    v = value_at(lookup, rk=rk, nodes=nodes, kmax=kmax, arrival_s=arrival_s, col=col)
    if is_finite(v):
        v = float(v) * float(scale)
    else:
        v = float("nan")
    return fmt_signed(v, decimals, nan_str(latex))

def cell_count(
    lookup: pd.DataFrame,
    *,
    kmax: int,
    rk: RowKey,
    nodes: int,
    arrival_s: float,
    col: str,
    latex: bool,
) -> str:
    """
    Return signed count cell value.
    """
    v = value_at(lookup, rk=rk, nodes=nodes, kmax=kmax, arrival_s=arrival_s, col=col)
    return fmt_count(v, nan_str(latex))

def cell_total_or_vec_prio(
    lookup: pd.DataFrame,
    *,
    kmax: int,
    rk: RowKey,
    nodes: int,
    arrival_s: float,
    total_col: str,
    part_col_tpl: str,  # e.g. "delta_D_p{p}_mean"
    decimals: int,
    latex: bool,
) -> str:
    """
    Return total or vector cell value depending on kmax.
    """
    if kmax == 1:
        v = value_at(lookup, rk=rk, nodes=nodes, kmax=kmax, arrival_s=arrival_s, col=total_col)
        return fmt_signed(v, decimals, nan_str(latex))

    vals = [
        value_at(lookup, rk=rk, nodes=nodes, kmax=kmax, arrival_s=arrival_s, col=part_col_tpl.format(p=p))
        for p in range(1, 5)
    ]
    return fmt_vec(vals, decimals, nan_str(latex), latex=latex)

def join_cells(a_cell, b_cell, *, sep: str = "; "):
    """
    Return a cell function that joins two cell functions with sep.
    """
    return lambda k, rk, n, a: f"{a_cell(k, rk, n, a)}{sep}{b_cell(k, rk, n, a)}"

# =============================================================================
# Table Construction Helpers
# =============================================================================

@dataclass(frozen=True)
class TableSpec:
    stem: str
    latex_title_math: str
    latex_subtitle: str
    ascii_title: str
    rows_by_kmax: Dict[int, List[RowKey]]
    make_cell_fn: Callable[[bool], Callable[[int, RowKey, int, float], str]]
    ascii_col_w: Optional[int] = None
    row_label_fn: Optional[RowLabelFn] = None

def write_table_spec(*, spec: TableSpec, out_dir: Path, nodes_order: List[int], arrivals_order: List[float]) -> None:
    """
    Write both LaTeX and ASCII tables for the given TableSpec.
    """
    latex_cell = spec.make_cell_fn(True)
    ascii_cell = spec.make_cell_fn(False)
    latex_args = dict(
        title_math=spec.latex_title_math,
        subtitle=spec.latex_subtitle,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        rows_by_kmax=spec.rows_by_kmax,
        cell_fn=lambda k, rk, n, a: latex_cell(k, rk, n, a),
    )
    ascii_args = dict(
        title=spec.ascii_title,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        rows_by_kmax=spec.rows_by_kmax,
        cell_fn=lambda k, rk, n, a: ascii_cell(k, rk, n, a),
    )
    if spec.ascii_col_w is not None:
        ascii_args["col_w"] = spec.ascii_col_w
    if spec.row_label_fn is not None:
        latex_args["row_label_fn"] = spec.row_label_fn
        ascii_args["row_label_fn"] = spec.row_label_fn
    write_both(stem=spec.stem, out_dir=out_dir, latex_args=latex_args, ascii_args=ascii_args)

def build_rows_by_kmax(df: pd.DataFrame) -> Dict[int, List[RowKey]]:
    """
    Build RowKey lists per kmax from DataFrame.
    """
    out: Dict[int, List[RowKey]] = {}
    for kmax in sorted(df["kmax"].dropna().astype(int).unique().tolist()):
        pcs = df.loc[df["kmax"].astype(int) == kmax, "plugin_config"].unique().tolist()
        rks = [RowKey.from_plugin_config(pc) for pc in pcs]
        rks.sort(key=lambda rk: rk.sort_key())
        out[kmax] = rks
    return out

def mk_cell_signed_col(*, lookup: pd.DataFrame, col: str, scale: float = 1.0, decimals: int = TABLE_DECIMALS):
    """
    Make cell function for signed float column.
    """
    def _mk(latex: bool):
        return lambda k, rk, n, a: cell_signed(
            lookup,
            kmax=k, rk=rk, nodes=n, arrival_s=a,
            col=col, decimals=decimals, latex=latex, scale=scale,
        )
    return _mk

def mk_cell_count_col(*, lookup: pd.DataFrame, col: str):
    """
    Make cell function for signed count column.
    """
    def _mk(latex: bool):
        return lambda k, rk, n, a: cell_count(
            lookup,
            kmax=k, rk=rk, nodes=n, arrival_s=a,
            col=col, latex=latex,
        )
    return _mk

def mk_cell_total_or_vec(*, lookup: pd.DataFrame, total_col: str, part_tpl: str, decimals: int = TABLE_DECIMALS):
    """
    Make cell function for total or vector column depending on kmax.
    """
    def _mk(latex: bool):
        return lambda k, rk, n, a: cell_total_or_vec_prio(
            lookup,
            kmax=k, rk=rk, nodes=n, arrival_s=a,
            total_col=total_col, part_col_tpl=part_tpl,
            decimals=decimals, latex=latex,
        )
    return _mk

def mk_cell_def_delta(*, lookup: pd.DataFrame, total_col: str, part_tpl: str, decimals: int = TABLE_DECIMALS):
    """
    Make cell function for defpreempt delta total or vector depending on kmax.
    """
    def _mk(latex: bool):
        return lambda k, rk, n, a: cell_defpreempt_delta_total_or_vec(
            lookup,
            kmax=k, mode=rk.mode, blocking=rk.blocking, nodes=n, arrival_s=a,
            total_col=total_col, part_col_tpl=part_tpl,
            decimals=decimals, latex=latex,
        )
    return _mk

# =============================================================================
# Main + CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ASCII+LaTeX tables and figures from sealed results.")
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

    # Infer nodes + arrivals from the CSV (replaces NODES and ARRIVALS_ORDER).
    nodes_order = sorted(df["nodes"].dropna().astype(int).unique().tolist())
    arrivals_order = sorted(df["arrival_s"].dropna().astype(float).unique().tolist())

    if not nodes_order:
        raise SystemExit("No node values inferred from results_paired.csv")
    if not arrivals_order:
        raise SystemExit("No arrival values inferred from results_paired.csv")

    # We only support dumbbell plots for exactly two node counts.
    if len(nodes_order) != 2:
        raise SystemExit(
            f"Expected exactly 2 distinct node values for dumbbell plots; found {len(nodes_order)}: {nodes_order}"
        )

    plot_kmaxs = sorted(df["kmax"].dropna().astype(int).unique().tolist())
    if not plot_kmaxs:
        raise SystemExit("No kmax values found in results_paired.csv")

    # Build RowKey lists per kmax (from plugin_config rows)
    rows_by_kmax = build_rows_by_kmax(df)

    # defpreempt-only (used by big_table_latency_util)
    rows_by_kmax_defpreempt_only: Dict[int, List[RowKey]] = {
        k: [rk for rk in rks if rk.defpreempt == 1]
        for k, rks in rows_by_kmax.items()
    }

    # Rows for defpreempt delta tables + plots (computed once)
    def_rows = build_defpreempt_rows(df)

    rows_by_kmax_def: Dict[int, List[RowKey]] = {}
    for kmax in sorted(rows_by_kmax.keys()):
        tmp = [RowKey(mode=m, blocking=b, defpreempt=1) for (m, b) in def_rows]  # defpreempt irrelevant for label
        tmp.sort(key=lambda rk: rk.sort_key())
        rows_by_kmax_def[kmax] = tmp

    # -------------------------------------------------------------------------
    # TABLES
    # -------------------------------------------------------------------------

    mk_cell_u_eff = mk_cell_signed_col(lookup=lookup, col="delta_util_eff_run_mean", scale=100.0)
    mk_cell_R = mk_cell_total_or_vec(lookup=lookup, total_col="delta_R_total_mean", part_tpl="delta_R_p{p}_mean")
    mk_cell_D = mk_cell_total_or_vec(lookup=lookup, total_col="delta_D_total_mean", part_tpl="delta_D_p{p}_mean")
    mk_cell_L = mk_cell_total_or_vec(lookup=lookup, total_col="delta_L_s_total_mean", part_tpl="delta_L_s_p{p}_mean")

    mk_cell_solver_attempts = mk_cell_count_col(lookup=lookup, col="solver_attempts_mean")
    mk_cell_plans_activated = mk_cell_count_col(lookup=lookup, col="plan_activated_mean")

    # Combined: latency ; util  (IMPORTANT: defpreempt=1 only)
    mk_cell_latency_vec_or_total = mk_cell_total_or_vec(lookup=lookup, total_col="delta_L_s_total_mean", part_tpl="delta_L_s_p{p}_mean")
    mk_cell_u_eff_signed = mk_cell_signed_col(lookup=lookup, col="delta_util_eff_run_mean", scale=100.0)

    def mk_cell_big_latency_util_opt(latex: bool):
        lat = mk_cell_latency_vec_or_total(latex)  # ΔL (vec or total)
        u = mk_cell_u_eff_signed(latex)            # Δu_eff (pp)
        opt = mk_cell_plans_activated(latex)       # #optimizations (count)
        return join_cells(join_cells(lat, u, sep="; "), opt, sep="; ")

    # defpreempt delta tables (within-plugin): (defpreempt=0)-(defpreempt=1)
    mk_cell_def_delta_D = mk_cell_def_delta(lookup=lookup, total_col="delta_D_total_mean", part_tpl="delta_D_p{p}_mean")
    mk_cell_def_delta_L = mk_cell_def_delta(lookup=lookup, total_col="delta_L_s_total_mean", part_tpl="delta_L_s_p{p}_mean")
    mk_cell_opt_mean = mk_cell_count_col(lookup=lookup, col="plan_activated_mean")

    def mk_cell_big_def_delta_D_L_opt(latex: bool):
        d = mk_cell_def_delta_D(latex)
        l = mk_cell_def_delta_L(latex)
        opt = mk_cell_opt_mean(latex)  # mean #optimizations (plans_activated)
        return join_cells(join_cells(d, l, sep="; "), opt, sep="; ")

    TABLE_SPECS: List[TableSpec] = [
        TableSpec(
            stem="delta_u_eff_run",
            latex_title_math=r"\Delta u_{\mathrm{eff}}(pp)",
            latex_subtitle=r"(mean effective utilization, max(cpu-util, mem-util)), plugin vs.\ baseline",
            ascii_title="Delta u_eff (pp), plugin vs baseline",
            rows_by_kmax=rows_by_kmax,
            make_cell_fn=mk_cell_u_eff,
        ),
        TableSpec(
            stem="delta_R",
            latex_title_math=r"\Delta R(count)",
            latex_subtitle=r"(mean running pods), plugin vs.\ baseline",
            ascii_title="Delta R(count), plugin vs baseline",
            rows_by_kmax=rows_by_kmax,
            make_cell_fn=mk_cell_R,
            ascii_col_w=28,
        ),
        TableSpec(
            stem="delta_D",
            latex_title_math=r"\Delta D(count)",
            latex_subtitle=r"(mean deletions), plugin vs.\ baseline",
            ascii_title="Delta D(count), plugin vs baseline",
            rows_by_kmax=rows_by_kmax,
            make_cell_fn=mk_cell_D,
            ascii_col_w=28,
        ),
        TableSpec(
            stem="delta_L",
            latex_title_math=r"\Delta L(s)",
            latex_subtitle=r"(mean latency), plugin vs.\ baseline",
            ascii_title="Delta L(s), plugin vs baseline",
            rows_by_kmax=rows_by_kmax,
            make_cell_fn=mk_cell_L,
            ascii_col_w=28,
        ),
        TableSpec(
            stem="solver_attempts",
            latex_title_math=r"\mathrm{solver\_attempts}",
            latex_subtitle=r"(mean solver attempts)",
            ascii_title="Solver attempts (mean)",
            rows_by_kmax=rows_by_kmax,
            make_cell_fn=mk_cell_solver_attempts,
        ),
        TableSpec(
            stem="plans_activated",
            latex_title_math=r"\mathrm{plans\_activated}",
            latex_subtitle=r"(mean plans activated)",
            ascii_title="Plans activated (mean)",
            rows_by_kmax=rows_by_kmax,
            make_cell_fn=mk_cell_plans_activated,
        ),
        TableSpec(
            stem="big_table_latency_util_optimizations",
            latex_title_math=r"\Delta L\ ;\ \Delta u_{\mathrm{eff}}\ ;\ \#\mathrm{optimizations}",
            latex_subtitle=r"(s; pp; count), plugin vs.\ baseline ($\Delta L$, $\Delta u_{\mathrm{eff}}$); \#optimizations mean",
            ascii_title="Combined (defpreempt enabled): ΔL (s) ; Δu_eff (pp) ; #optimizations",
            rows_by_kmax=rows_by_kmax_defpreempt_only,
            make_cell_fn=mk_cell_big_latency_util_opt,
            ascii_col_w=40,  # bump a bit since cells are longer now
        ),
        TableSpec(
            stem="delta_D_without_defaultpreemption",
            latex_title_math=r"\Delta D_{\mathrm{w/o\ defpreempt}}",
            latex_subtitle=r"(mean count), $(\mathrm{defpreempt}=0)-(\mathrm{defpreempt}=1)$",
            ascii_title="Delta deletions: (defpreempt=0) - (defpreempt=1)  [within plugin, baseline cancels]",
            rows_by_kmax=rows_by_kmax_def,
            make_cell_fn=mk_cell_def_delta_D,
            ascii_col_w=28,
            row_label_fn=lambda rk: rk.label(include_defpreempt=False),
        ),
        TableSpec(
            stem="delta_L_without_defaultpreemption",
            latex_title_math=r"\Delta L_{\mathrm{w/o\ defpreempt}}",
            latex_subtitle=r"(mean latency), $(\mathrm{defpreempt}=0)-(\mathrm{defpreempt}=1)$",
            ascii_title="Delta latency: (defpreempt=0) - (defpreempt=1)  [within plugin, baseline cancels]",
            rows_by_kmax=rows_by_kmax_def,
            make_cell_fn=mk_cell_def_delta_L,
            ascii_col_w=28,
            row_label_fn=lambda rk: rk.label(include_defpreempt=False),
        ),
        TableSpec(
            stem="big_table_without_defaultpreemption_deletions_latency_optimizations",
            latex_title_math=r"\Delta D_{\mathrm{w/o\ defpreempt}}\ ;\ \Delta L_{\mathrm{w/o\ defpreempt}}\ ;\ \#\mathrm{optimizations}",
            latex_subtitle=r"(count; s; count), $(\mathrm{defpreempt}=0)-(\mathrm{defpreempt}=1)$ for $\Delta D,\Delta L$; \#optimizations mean",
            ascii_title="Combined: Delta deletions ; Delta latency ; #optimizations  (ΔD,ΔL are (defpreempt=0)-(defpreempt=1))",
            rows_by_kmax=rows_by_kmax_def,
            make_cell_fn=mk_cell_big_def_delta_D_L_opt,
            ascii_col_w=52,
            row_label_fn=lambda rk: rk.label(include_defpreempt=False),
        ),
    ]

    for spec in TABLE_SPECS:
        write_table_spec(spec=spec, out_dir=tables_dir, nodes_order=nodes_order, arrivals_order=arrivals_order)

    # -------------------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------------------

    util_y_vals_all: List[float] = []
    latency_y_vals_all: List[float] = []
    for k in plot_kmaxs:
        util_y_vals_all += collect_plot_y_vals(
            lookup=lookup,
            df=df,
            col="delta_util_eff_run_mean",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=int(k),
            scale=100.0,
        )
        latency_y_vals_all += collect_plot_y_vals(
            lookup=lookup,
            df=df,
            col="delta_L_s_total_mean",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=int(k),
            scale=1.0,
        )

    util_ylim = symmetric_ylim_from_y_values(util_y_vals_all)
    latency_ylim = symmetric_ylim_from_y_values(latency_y_vals_all)

    def y_from_col(*, col: str, scale: float) -> YOfFn:
        def _y(rk: RowKey, nodes: int, a: float, kmax: int) -> float:
            v = value_at(lookup, rk=rk, nodes=nodes, kmax=kmax, arrival_s=a, col=col)
            return float(v) * float(scale) if is_finite(v) else float("nan")
        return _y

    def y_defpreempt_delta(*, total_col: str) -> YOfFn:
        def _y(rk: RowKey, nodes: int, a: float, kmax: int) -> float:
            return defpreempt_delta_total(
                lookup,
                kmax=kmax, mode=rk.mode, blocking=rk.blocking,
                nodes=nodes, arrival_s=a, total_col=total_col,
            )
        return _y

    def_series = [RowKey(mode=m, blocking=b, defpreempt=1) for (m, b) in def_rows]
    def_series.sort(key=lambda rk: rk.sort_key())

    def_ylim_by_kmax_col: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for k in plot_kmaxs:
        for total_col in ("delta_D_total_mean", "delta_L_s_total_mean"):
            y_vals = collect_defpreempt_delta_total_y_vals(
                lookup=lookup,
                nodes_order=nodes_order,
                arrivals_order=arrivals_order,
                kmax=int(k),
                total_col=total_col,
                default_rows=def_rows,
            )
            def_ylim_by_kmax_col[(int(k), total_col)] = symmetric_ylim_from_y_values(y_vals)

    for plot_kmax in plot_kmaxs:
        series = plot_series(df, kmax=int(plot_kmax))

        plot_dumbbell(
            out_dir=figures_dir,
            filename_stem=f"delta_u_eff_run_kmax{plot_kmax}",
            title=rf"$\Delta u_{{\mathrm{{eff}}}}$ vs $\mu_A$  (kmax={plot_kmax})",
            y_label=r"$\Delta u_{\mathrm{eff}}$ (pp)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=int(plot_kmax),
            series=series,
            y_of=y_from_col(col="delta_util_eff_run_mean", scale=100.0),
            label_of=lambda rk: rk.label(include_defpreempt=True),
            ylim=util_ylim,
        )

        plot_dumbbell(
            out_dir=figures_dir,
            filename_stem=f"delta_L_kmax{plot_kmax}",
            title=rf"$\Delta L$ vs $\mu_A$  (kmax={plot_kmax})",
            y_label=r"$\Delta L$ (seconds)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=int(plot_kmax),
            series=series,
            y_of=y_from_col(col="delta_L_s_total_mean", scale=1.0),
            label_of=lambda rk: rk.label(include_defpreempt=True),
            ylim=latency_ylim,
        )

        plot_dumbbell(
            out_dir=figures_dir,
            filename_stem=f"delta_D_kmax{plot_kmax}_without_defaultpreemption",
            title=rf"$\Delta D_{{\mathrm{{w/o\ defpreempt}}}}$ vs $\mu_A$ (kmax={plot_kmax})",
            y_label=r"$\Delta D_{\mathrm{w/o\ defpreempt}}$ (#deletions)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=int(plot_kmax),
            series=def_series,
            y_of=y_defpreempt_delta(total_col="delta_D_total_mean"),
            label_of=lambda rk: rk.label(include_defpreempt=False),
            ylim=def_ylim_by_kmax_col[(int(plot_kmax), "delta_D_total_mean")],
            legend_loc="upper right",
        )

        plot_dumbbell(
            out_dir=figures_dir,
            filename_stem=f"delta_L_kmax{plot_kmax}_without_defaultpreemption",
            title=rf"$\Delta L_{{\mathrm{{w/o\ defpreempt}}}}$ vs $\mu_A$  (kmax={plot_kmax})",
            y_label=r"$\Delta L_{\mathrm{w/o\ defpreempt}}$ (seconds)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=int(plot_kmax),
            series=def_series,
            y_of=y_defpreempt_delta(total_col="delta_L_s_total_mean"),
            label_of=lambda rk: rk.label(include_defpreempt=False),
            ylim=def_ylim_by_kmax_col[(int(plot_kmax), "delta_L_s_total_mean")],
        )

    print(f"Wrote tables to:  {tables_dir}")
    print(f"Wrote figures to: {figures_dir}")


if __name__ == "__main__":
    main()
