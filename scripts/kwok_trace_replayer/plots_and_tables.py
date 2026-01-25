#!/usr/bin/env python3
# scripts/kwok_trace_replayer/plots_and_tables.py
"""
python -m scripts.kwok_trace_replayer.plots_and_tables --in-results analysis/kwok_trace_replayer/results_paired.csv --out-dir analysis/kwok_trace_replayer
"""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

TABLE_DECIMALS: int = 1
JOB_RE = re.compile(r"nodes=(\d+)_prio=(\d+)_arrival=([0-9.]+)s")

# Convert latency from seconds -> milliseconds in BOTH tables and plots
LATENCY_SCALE: float = 1000.0

# --- Symmetric-log (symlog) settings for plots ---
SYMLOG_BASE: float = 10.0
# Linear region around 0 (in data units of the plotted metric):
SYMLOG_LINTHRESH_LAT_MS: float = 1.0      # 1 ms linear region
SYMLOG_LINTHRESH_DELETIONS: float = 1.0   # 1 deletion linear region
SYMLOG_LINSCALE: float = 1.0              # size of linear region (visual)

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

PLOT_MARKER_SIZE = 3.5
PLOT_ARRIVAL_X_SPACING = 0.35
PLOT_MODE_X_SPACING = 0.05

PLOT_TICK_FONTSIZE = 7
PLOT_TICK_PAD = 2.0
PLOT_TICK_LENGTH = 4.0

PLOT_AXIS_LABEL_FONTSIZE = 7
PLOT_Y_PADDING = 0.15

PLOT_LEGEND_FONTSIZE = 7
PLOT_LEGEND_HANDLE_LENGTH = 0.4
PLOT_LEGEND_COLUMN_SPACING = 0.5
PLOT_LEGEND_HANDLE_TEXT_PAD = 0.46
PLOT_LEGEND_FRAME_LINEWIDTH = 1.0
PLOT_LEGEND_NCOL = 5

GRID_FIGSIZE = (4.5, 5)
GRID_LEFT   = 0.125
GRID_RIGHT  = 0.99
GRID_BOTTOM = 0.04
GRID_TOP    = 0.92
GRID_WSPACE = 0.1
GRID_HSPACE = 0.1

GRID_LEGEND_PAD = 0.04

GRID_TITLE_FONTSIZE = 7
GRID_YLABEL_PAD_FIG = 0.095

# =============================================================================
# Parsing helpers
# =============================================================================

def parse_job(job_name: str) -> Optional[Tuple[int, int, float]]:
    """Parse job_name format: nodes=<N>_prio=<kmax>_arrival=<a>s"""
    m = JOB_RE.fullmatch(str(job_name).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def parse_plugin_config(plugin_config: str) -> Dict[str, str]:
    """plugin_config: mode=<mode>_blocking=<0/1>_defpreempt=<0/1>"""
    kv: Dict[str, str] = {}
    for tok in str(plugin_config).split("_"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return kv


def canonical_mode(mode: str) -> str:
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
        def_rank = 0 if self.defpreempt == 1 else 1
        return (base_rank, def_rank, self.mode)

    def abbr(self) -> str:
        if self.mode == "scheduling-failure":
            return "SF"
        return MODE_ABBR.get((self.mode, int(self.blocking)), f"{self.mode}:{self.blocking}")

# =============================================================================
# Formatting helpers (tables)
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
    v = round(float(x), int(decimals))
    if v == 0.0:
        v = 0.0
    return f"{v:+.{decimals}f}"


def fmt_unsigned_int(x: object, nan_s: str) -> str:
    if not is_finite(x):
        return nan_s
    return f"{int(round(float(x))):d}"


def arrival_group_label(a: float, *, latex: bool) -> str:
    a_i = int(a) if abs(a - round(a)) < 1e-9 else a
    return (rf"$\mu_A={a_i}\,\mathrm{{s}}$" if latex else f"muA={a_i}s")


def kmax_caption(kmax: int, *, latex: bool, without_default_preemption: bool) -> str:
    if latex:
        base = (r"$k_{\max}=1$ (w/o priorities)" if kmax == 1 else rf"$k_{{\max}}={kmax}$ (w/ priorities)")
        return base + (r" (w/o default preemption)" if without_default_preemption else "")
    base = ("kmax=1 (w/o priorities)" if kmax == 1 else f"kmax={kmax} (w/ priorities)")
    return base + (" (w/o default preemption)" if without_default_preemption else "")

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
    d = df.drop_duplicates(subset=KEY_COLS, keep="first").copy()
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
# Tables (metrics-as-rows; arrivals top; modes under arrivals)
# =============================================================================

MetricGetter = Callable[[int, RowKey, int, float], float]  # (kmax, rk, nodes, arrival) -> value


@dataclass(frozen=True)
class MetricRow:
    latex_label: str
    ascii_label: str
    getter: MetricGetter
    fmt_kind: str  # "signed_float" | "unsigned_int"
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
    return fmt_signed(vv, row.decimals, ns)


def latex_metric_matrix_tables(
    *,
    out_path: Path,
    nodes_order: List[int],
    arrivals_order: List[float],
    modes_by_kmax: Dict[int, List[RowKey]],
    metrics_by_kmax: Dict[int, List[MetricRow]],
    without_default_preemption: bool,
) -> None:
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

        cmid = []
        for i in range(n_arr):
            start = 2 + i * n_modes
            end = start + n_modes - 1
            cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")

        arrival_hdr = " & " + " & ".join(
            [rf"\multicolumn{{{n_modes}}}{{c}}{{{arrival_group_label(a, latex=True)}}}" for a in arrivals_order]
        ) + r" \\"
        mode_hdr = " & " + " & ".join([rk.abbr() for _a in arrivals_order for rk in modes]) + r" \\"

        lines.append(rf"\begin{{tabular}}{{{tab_spec}}}")
        lines.append(r"\toprule")
        lines.append(
            rf"\multicolumn{{{total_cols}}}{{l}}{{{kmax_caption(kmax, latex=True, without_default_preemption=without_default_preemption)}}} \\"
        )
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
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def ascii_metric_matrix_tables(
    *,
    out_path: Path,
    nodes_order: List[int],
    arrivals_order: List[float],
    modes_by_kmax: Dict[int, List[RowKey]],
    metrics_by_kmax: Dict[int, List[MetricRow]],
    without_default_preemption: bool,
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

        mode_labels = [rk.abbr() for _a in arrivals_order for rk in modes]
        arr_labels = [arrival_group_label(a, latex=False) for a in arrivals_order]

        row_w = max(12, max(len(m.ascii_label) for m in metrics) + 2)
        cell_w = 10
        total_cols = len(mode_labels)
        total_w = row_w + cell_w * total_cols
        sep = "-" * total_w

        lines.append(kmax_caption(kmax, latex=False, without_default_preemption=without_default_preemption))
        lines.append("")

        hdr1 = " " * row_w
        for al in arr_labels:
            hdr1 += center(al, cell_w * len(modes))
        lines.append(hdr1.rstrip())

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

YOfFn = Callable[[RowKey, int, float, int], float]  # (rk, nodes, arrival_s, kmax) -> y


def _default_color_cycle() -> List[str]:
    return list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"]))


def _build_color_map(labels: List[str]) -> Dict[str, str]:
    cycle = _default_color_cycle()
    return {lbl: cycle[i % len(cycle)] for i, lbl in enumerate(labels)}


def _make_mode_legend_handles(labels: List[str], color_map: Dict[str, str]) -> List[Line2D]:
    return [Line2D([0], [0], color=color_map[lbl], linewidth=1.8) for lbl in labels]


def symmetric_ylim_from_y_values(y_vals: List[float]) -> Tuple[float, float]:
    vals = [abs(float(y)) for y in y_vals if is_finite(y)]
    if not vals:
        return (-1.0, 1.0)
    m = max(vals)
    if m <= 0:
        return (-1.0, 1.0)
    pad = PLOT_Y_PADDING * m
    return (-(m + pad), (m + pad))


def plot_series(df: pd.DataFrame, *, kmax: int, defpreempt_value: int) -> List[RowKey]:
    """
    For plotting we keep:
      - requested defpreempt_value configs (0 or 1)
      - plus scheduling-failure (SF) always
    """
    dff = df[df["kmax"].astype(int) == int(kmax)].copy()
    keep = (dff["defpreempt"].astype(int) == int(defpreempt_value)) | (dff["mode"].astype(str) == "scheduling-failure")
    dff = dff.loc[keep].drop_duplicates(
        subset=["job_name", "plugin_config", "mode", "blocking", "defpreempt", "nodes", "kmax", "arrival_s"],
        keep="first",
    )
    return sorted(
        {
            RowKey(mode=m, blocking=int(b), defpreempt=int(d))
            for (m, b, d) in dff[["mode", "blocking", "defpreempt"]].drop_duplicates().itertuples(index=False, name=None)
        },
        key=lambda rk: rk.sort_key(),
    )


def collect_plot_y_vals(
    *,
    lookup: pd.DataFrame,
    df: pd.DataFrame,
    col: str,
    nodes_order: List[int],
    arrivals_order: List[float],
    kmax: int,
    defpreempt_value: int,
    scale: float = 1.0,
) -> List[float]:
    series = plot_series(df, kmax=kmax, defpreempt_value=defpreempt_value)
    y_vals: List[float] = []
    for rk in series:
        for a in arrivals_order:
            for n in nodes_order:
                v = lookup_value(lookup, nodes=n, kmax=kmax, arrival_s=a, rk=rk, col=col)
                if is_finite(v):
                    y_vals.append(float(v) * float(scale))
    return y_vals


def draw_dumbbell_on_ax(
    *,
    ax: plt.Axes,
    nodes_order: List[int],
    arrivals_order: List[float],
    kmax: int,
    series: List[RowKey],
    y_of: YOfFn,
    color_map: Dict[str, str],
    ylim: Tuple[float, float],
    show_xticklabels: bool,
    show_yticklabels: bool,
    y_scale: str = "linear",          # "linear" | "symlog"
    symlog_linthresh: float = 1.0,    # only used when y_scale="symlog"
) -> None:
    x_base = [i * PLOT_ARRIVAL_X_SPACING for i in range(len(arrivals_order))]
    m = max(1, len(series))

    for i, rk in enumerate(series):
        label = rk.abbr()
        color = color_map[label]
        mode_offset = (i - (m - 1) / 2.0) * PLOT_MODE_X_SPACING

        for xi, a in enumerate(arrivals_order):
            x = x_base[xi] + mode_offset
            y0 = y_of(rk, nodes_order[0], a, kmax)
            y1 = y_of(rk, nodes_order[1], a, kmax)

            if is_finite(y0) and is_finite(y1):
                ax.plot([x, x], [y0, y1], linestyle="--", color=color, linewidth=1.0)
            if is_finite(y0):
                ax.plot([x], [y0], linestyle="None", marker="o", markersize=PLOT_MARKER_SIZE, color=color)
            if is_finite(y1):
                ax.plot([x], [y1], linestyle="None", marker="s", markersize=PLOT_MARKER_SIZE, color=color)

    # Scale first (so tick locators are correct), then limits
    if y_scale == "symlog":
        ax.set_yscale(
            "symlog",
            base=SYMLOG_BASE,
            linthresh=symlog_linthresh,
            linscale=SYMLOG_LINSCALE,
        )
    else:
        ax.set_yscale("linear")

    ax.axhline(0.0, linewidth=0.8, color="black", linestyle="--", alpha=0.7)
    ax.set_ylim(float(ylim[0]), float(ylim[1]))

    pad = max(0.10, PLOT_MODE_X_SPACING * (m / 2.0 + 1.0))
    ax.set_xlim(min(x_base) - pad, max(x_base) + pad)

    ax.set_xticks(x_base)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=PLOT_TICK_FONTSIZE,
        pad=PLOT_TICK_PAD,
        length=PLOT_TICK_LENGTH,
    )

    if show_xticklabels:
        ax.set_xticklabels([arrival_group_label(a, latex=True) for a in arrivals_order])
    else:
        ax.tick_params(labelbottom=False)

    if not show_yticklabels:
        ax.tick_params(labelleft=False)

    for y in ax.get_yticks():
        if abs(y) < 1e-8:
            continue
        ax.axhline(y, linewidth=0.8, color="black", linestyle="--", alpha=0.15, zorder=0)

# =============================================================================
# Main + CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate BIG tables and grid figures from sealed results.")
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

    # Exactly two columns: kmax=1 and kmax=4
    kmax_cols = [k for k in (1, 4) if k in kmaxs]
    if len(kmax_cols) != 2:
        raise SystemExit(f"Expected kmax values [1,4] for the 2 columns; found kmaxs={kmaxs}")

    def _sorted_modes(rks: List[RowKey]) -> List[RowKey]:
        return sorted(rks, key=lambda rk: rk.sort_key())

    # Modes for tables:
    #  - defpreempt=1: EXCLUDE SF
    #  - defpreempt=0: include SF (and other defpreempt=0 modes)
    modes_def1_tables: Dict[int, List[RowKey]] = {}
    modes_def0_tables: Dict[int, List[RowKey]] = {}

    for k in kmaxs:
        dff = df[df["kmax"].astype(int) == k]
        rks = [RowKey.from_plugin_config(pc) for pc in dff["plugin_config"].unique().tolist()]

        modes_def1_tables[k] = _sorted_modes(
            [rk for rk in rks if (rk.mode != "scheduling-failure" and rk.defpreempt == 1)]
        )
        modes_def0_tables[k] = _sorted_modes(
            [rk for rk in rks if (rk.mode == "scheduling-failure") or (rk.mode != "scheduling-failure" and rk.defpreempt == 0)]
        )

    # Metric getter
    def g(col: str) -> MetricGetter:
        return lambda kmax, rk, n, a: lookup_value(lookup, nodes=n, kmax=kmax, arrival_s=a, rk=rk, col=col)

    # Table rows (latency shown in ms + labeled as ms)
    def metrics_big(kmax: int) -> List[MetricRow]:
        rows: List[MetricRow] = [
            MetricRow(r"$\Delta U$", "ΔU", g("delta_U_eff_run_mean"), "signed_float", decimals=1, scale=100.0),

            # seconds -> ms in tables, and label explicitly shows ms
            MetricRow(r"$\Delta L_{\mathrm{tot}}\;(\mathrm{ms})$", "ΔL_tot (ms)",
                      g("delta_L_s_total_mean"), "signed_float", decimals=0, scale=LATENCY_SCALE),
        ]
        if kmax != 1:
            for p in range(1, 5):
                rows.append(
                    MetricRow(
                        rf"$\Delta L_{{p_{p}}}\;(\mathrm{{ms}})$",
                        f"ΔL_p{p} (ms)",
                        g(f"delta_L_s_p{p}_mean"),
                        "signed_float",
                        decimals=0,
                        scale=LATENCY_SCALE,  # seconds -> ms
                    )
                )

        rows.append(MetricRow(r"$\Delta D_{\mathrm{tot}}$", "ΔD_tot", g("delta_D_total_mean"), "signed_float", decimals=1))
        if kmax != 1:
            for p in range(1, 5):
                rows.append(
                    MetricRow(
                        rf"$\Delta D_{{p_{p}}}$",
                        f"ΔD_p{p}",
                        g(f"delta_D_p{p}_mean"),
                        "signed_float",
                        decimals=1,
                    )
                )

        rows.append(MetricRow(r"$\#O$", "#O", g("plan_activated_mean"), "unsigned_int"))
        return rows

    metrics_map_big = {k: metrics_big(k) for k in kmaxs}

    # -----------------------------
    # Tables
    # -----------------------------
    for stem, modes_map, without_defpreempt in [
        ("table_util_latency_deletions_optimizations", modes_def1_tables, False),
        ("table_without_defaultpreemption_util_latency_deletions_optimizations", modes_def0_tables, True),
    ]:
        latex_metric_matrix_tables(
            out_path=tables_dir / f"{stem}.tex",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            modes_by_kmax=modes_map,
            metrics_by_kmax=metrics_map_big,
            without_default_preemption=without_defpreempt,
        )
        ascii_metric_matrix_tables(
            out_path=tables_dir / f"{stem}.txt",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            modes_by_kmax=modes_map,
            metrics_by_kmax=metrics_map_big,
            without_default_preemption=without_defpreempt,
        )

    # -----------------------------
    # Figures
    # -----------------------------

    util_y_vals_all: List[float] = []
    deletions_y_vals_all: List[float] = []
    latency_y_vals_all: List[float] = []

    for k in kmaxs:
        for dp in (0, 1):
            util_y_vals_all += collect_plot_y_vals(
                lookup=lookup, df=df, col="delta_U_eff_run_mean",
                nodes_order=nodes_order, arrivals_order=arrivals_order, kmax=k, defpreempt_value=dp, scale=100.0,
            )
            deletions_y_vals_all += collect_plot_y_vals(
                lookup=lookup, df=df, col="delta_D_total_mean",
                nodes_order=nodes_order, arrivals_order=arrivals_order, kmax=k, defpreempt_value=dp, scale=1.0,
            )
            latency_y_vals_all += collect_plot_y_vals(
                lookup=lookup, df=df, col="delta_L_s_total_mean",
                nodes_order=nodes_order, arrivals_order=arrivals_order, kmax=k, defpreempt_value=dp, scale=LATENCY_SCALE,
            )

    util_ylim = symmetric_ylim_from_y_values(util_y_vals_all)
    deletions_ylim = symmetric_ylim_from_y_values(deletions_y_vals_all)
    latency_ylim = symmetric_ylim_from_y_values(latency_y_vals_all)

    def y_from_col(*, col: str, scale: float) -> YOfFn:
        def _y(rk: RowKey, nodes: int, a: float, kmax: int) -> float:
            v = lookup_value(lookup, nodes=nodes, kmax=kmax, arrival_s=a, rk=rk, col=col)
            return float(v) * float(scale) if is_finite(v) else float("nan")
        return _y

    def plot_grid_figure(*, defpreempt_value: int, filename_stem: str) -> None:
        series_union = sorted(
            set(plot_series(df, kmax=kmax_cols[0], defpreempt_value=defpreempt_value))
            | set(plot_series(df, kmax=kmax_cols[1], defpreempt_value=defpreempt_value)),
            key=lambda rk: rk.sort_key(),
        )
        labels = [rk.abbr() for rk in series_union]
        color_map = _build_color_map(labels)

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=GRID_FIGSIZE, sharex=True)

        axes[0, 0].set_title(rf"$k_{{\max}}={kmax_cols[0]}$", fontsize=GRID_TITLE_FONTSIZE)
        axes[0, 1].set_title(rf"$k_{{\max}}={kmax_cols[1]}$", fontsize=GRID_TITLE_FONTSIZE)

        # Util (linear)
        for col_i, k in enumerate(kmax_cols):
            draw_dumbbell_on_ax(
                ax=axes[0, col_i],
                nodes_order=nodes_order,
                arrivals_order=arrivals_order,
                kmax=k,
                series=series_union,
                y_of=y_from_col(col="delta_U_eff_run_mean", scale=100.0),
                color_map=color_map,
                ylim=(-4.1, 4.1),
                show_xticklabels=False,
                show_yticklabels=(col_i == 0),
                y_scale="linear",
            )

        # Latency (symlog, symmetric around 0)
        for col_i, k in enumerate(kmax_cols):
            draw_dumbbell_on_ax(
                ax=axes[1, col_i],
                nodes_order=nodes_order,
                arrivals_order=arrivals_order,
                kmax=k,
                series=series_union,
                y_of=y_from_col(col="delta_L_s_total_mean", scale=LATENCY_SCALE),  # ms
                color_map=color_map,
                ylim=(-10e3 - 1.0, 10e3 + 1.0),
                show_xticklabels=False,
                show_yticklabels=(col_i == 0),
                y_scale="symlog",
                symlog_linthresh=SYMLOG_LINTHRESH_LAT_MS,
            )

        # Deletions (symlog, symmetric around 0)
        for col_i, k in enumerate(kmax_cols):
            draw_dumbbell_on_ax(
                ax=axes[2, col_i],
                nodes_order=nodes_order,
                arrivals_order=arrivals_order,
                kmax=k,
                series=series_union,
                y_of=y_from_col(col="delta_D_total_mean", scale=1.0),
                color_map=color_map,
                ylim=(-10e3 - 1.0, 10e3 + 1.0),
                show_xticklabels=True,
                show_yticklabels=(col_i == 0),
                y_scale="symlog",
                symlog_linthresh=SYMLOG_LINTHRESH_DELETIONS,
            )

        fig.subplots_adjust(
            left=GRID_LEFT, right=GRID_RIGHT, bottom=GRID_BOTTOM, top=GRID_TOP,
            wspace=GRID_WSPACE, hspace=GRID_HSPACE,
        )

        # Legend centered over the two plot columns (axes grid), not the whole figure
        handles = _make_mode_legend_handles(labels, color_map)
        bbox_l = axes[0, 0].get_position()
        bbox_r = axes[0, 1].get_position()
        x_center_grid = 0.5 * (bbox_l.x0 + bbox_r.x1)
        y_top_grid = max(bbox_l.y1, bbox_r.y1)
        legend_y = min(0.98, y_top_grid + GRID_LEGEND_PAD)

        leg = fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(x_center_grid, legend_y),
            ncol=min(int(PLOT_LEGEND_NCOL), len(labels)),
            frameon=True,
            fontsize=PLOT_LEGEND_FONTSIZE,
            handlelength=PLOT_LEGEND_HANDLE_LENGTH,
            handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
            columnspacing=PLOT_LEGEND_COLUMN_SPACING,
            borderaxespad=0.0,
        )
        leg.get_frame().set_linewidth(PLOT_LEGEND_FRAME_LINEWIDTH)

        # Figure-level ylabels aligned to the grid
        left_bbox = axes[0, 0].get_position()
        x_text = max(0.0, left_bbox.x0 - GRID_YLABEL_PAD_FIG)

        row_labels = [
            r"$\Delta U_{\mathrm{eff}}$ (pp)",
            r"$\Delta L$ (ms)",
            r"$\Delta D$ (#deletions)",
        ]
        for r, text in enumerate(row_labels):
            bbox = axes[r, 0].get_position()
            y_center = 0.5 * (bbox.y0 + bbox.y1)
            fig.text(x_text, y_center, text, rotation=90, va="center", ha="right", fontsize=PLOT_AXIS_LABEL_FONTSIZE)

        fig.savefig(figures_dir / f"{filename_stem}.png", dpi=200)
        fig.savefig(figures_dir / f"{filename_stem}.pdf")
        plt.close(fig)

    plot_grid_figure(defpreempt_value=1, filename_stem="grid_util_latency_deletions")
    plot_grid_figure(defpreempt_value=0, filename_stem="grid_util_latency_deletions_without_defaultpreemption")

    print(f"Wrote tables to:  {tables_dir}")
    print(f"Wrote figures to: {figures_dir}")


if __name__ == "__main__":
    main()
