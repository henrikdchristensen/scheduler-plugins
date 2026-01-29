#!/usr/bin/env python3
# scripts/kwok_trace_replayer/plots_and_tables.py
"""
python -m scripts.kwok_trace_replayer.plots_and_tables

Update:
- Supports results_paired.csv that includes *_std columns (seed-to-seed std).
- Big LaTeX tables now print "mean ± std" when std columns are present.
- Plots remain based on *_mean (unchanged).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.config.plot_config import (
    PLOT_AXIS_LABEL_FONTSIZE,
    PLOT_FIGURE_DPI,
    PLOT_LEGEND_COLUMN_SPACING,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_HANDLE_LENGTH,
    PLOT_LEGEND_HANDLE_TEXT_PAD,
    PLOT_TICK_FONTSIZE,
    PLOT_TITLE_FONTSIZE,
)

# =============================================================================
# Paths
# =============================================================================

IN_RESULTS = Path("analysis/kwok_trace_replayer/results_paired.csv")
OUT_DIR = Path("analysis/kwok_trace_replayer")
OUT_TABLES_DIR = OUT_DIR / "tables"
OUT_FIGURES_DIR = OUT_DIR / "figures"

# =============================================================================
# Constants / expectations
# =============================================================================

TABLE_DECIMALS: int = 1
JOB_RE = re.compile(r"nodes=(\d+)_prio=(\d+)_arrival=([0-9.]+)s")

# --- symlog settings ---
SYMLOG_BASE: float = 10.0
SYMLOG_LINTHRESH_LAT_MS: float = 1.0
SYMLOG_LINTHRESH_DELETIONS: float = 1.0
SYMLOG_LINSCALE: float = 1.0

# Columns we expect in results_paired.csv (means)
EXPECTED_COLS = [
    "job_name",
    "plugin_config",
    "delta_R_num_p1_mean",
    "delta_R_num_p2_mean",
    "delta_R_num_p3_mean",
    "delta_R_num_p4_mean",
    "delta_R_num_total_mean",
    "delta_D_num_p1_mean",
    "delta_D_num_p2_mean",
    "delta_D_num_p3_mean",
    "delta_D_num_p4_mean",
    "delta_D_num_total_mean",
    "delta_L_ms_p1_mean",
    "delta_L_ms_p2_mean",
    "delta_L_ms_p3_mean",
    "delta_L_ms_p4_mean",
    "delta_L_ms_total_mean",
    "solver_attempts_mean",
    "plan_activated_mean",
]

KEY_COLS = ["nodes", "priorities", "arrival_s", "mode", "blocking", "defpreempt"]

# =============================================================================
# Plot styling
# =============================================================================

PLOT_TICK_PAD = 2.0
PLOT_MARKER_SIZE = 4.0
PLOT_MARKER_LINEWIDTH = 0.4
PLOT_MIN_LINEAR_YTICKS = 5

PLOT_HEIGHT = 8.3

# ---- Layout for "all configs" figures ----
GRID_LEGEND_NCOL_ALL = 3
GRID_FIGSIZE = (5.4, PLOT_HEIGHT)
GRID_LEGEND_PAD_ALL = 0.033
GRID_LEFT_ALL = 0.08
GRID_RIGHT_ALL = 0.99
GRID_BOTTOM_ALL = 0.03
GRID_TOP_ALL = 0.925
GRID_WSPACE_ALL = 0.10
GRID_HSPACE_ALL = 0.10
GRID_YLABEL_PAD_PT_ALL = 22.0
PLOT_ARRIVAL_X_SPACING_ALL = 0.35
PLOT_MODE_X_SPACING_ALL = 0.05

# ---- Layout for "deltas" figures ----
GRID_LEGEND_NCOL_DELTAS = 1
GRID_FIGSIZE_DELTAS = (2.15, PLOT_HEIGHT)
GRID_LEFT_DELTAS = 0.2
PLOT_MODE_X_SPACING_DELTAS = 0.11


# =============================================================================
# View + modes configuration
# =============================================================================


@dataclass(frozen=True)
class ViewConfig:
    name: str
    defpreempt_value: int
    without_default_preemption_caption: bool
    table_stem: str
    figure_stem: str


VIEWS: List[ViewConfig] = [
    ViewConfig(
        name="with_default_preemption",
        defpreempt_value=1,
        without_default_preemption_caption=False,
        table_stem="table_main",
        figure_stem="grid_main",
    ),
    ViewConfig(
        name="without_default_preemption",
        defpreempt_value=0,
        without_default_preemption_caption=True,
        table_stem="table_main",
        figure_stem="grid_main",
    ),
]


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    blocking: int
    abbr: str
    label: str
    rank: int
    palette: str  # "set2" or "set3"
    color_idx: int


MODE_SPECS: List[ModeSpec] = [
    # Scheduling-failure (both variants)
    ModeSpec("schedulingfailure", 1, "SF-B", "Scheduling-failure (blocking)", rank=0, palette="set2", color_idx=0),
    ModeSpec("schedulingfailure", 0, "SF-NB", "Scheduling-failure (non-blocking)", rank=1, palette="set2", color_idx=1),
    # Periodic 8s
    ModeSpec("periodic8s", 1, "PR-8-B", "Periodic (blocking), 8s interval", rank=2, palette="set2", color_idx=2),
    ModeSpec("periodic8s", 0, "PR-8-NB", "Periodic (non-blocking), 8s interval", rank=3, palette="set2", color_idx=3),
    # Periodic 32s
    ModeSpec("periodic32s", 1, "PR-32-B", "Periodic (blocking), 32s interval", rank=4, palette="set2", color_idx=4),
    ModeSpec("periodic32s", 0, "PR-32-NB", "Periodic (non-blocking), 32s interval", rank=5, palette="set2", color_idx=5),
    # Stable-queue 2s
    ModeSpec("stable-queue-2s", 1, "SQ-2-B", "Stable-queue (blocking), 2s delay", rank=6, palette="set2", color_idx=6),
    ModeSpec("stable-queue-2s", 0, "SQ-2-NB", "Stable-queue (non-blocking), 2s delay", rank=7, palette="set2", color_idx=7),
    # Stable-queue 8s
    ModeSpec("stable-queue-8s", 1, "SQ-8-B", "Stable-queue (blocking), 8s delay", rank=8, palette="set3", color_idx=3),
    ModeSpec("stable-queue-8s", 0, "SQ-8-NB", "Stable-queue (non-blocking), 8s delay", rank=9, palette="set3", color_idx=4),
]

# Tables: include everything (filter only by defpreempt)
INCLUDE_TABLES_ALL: Dict[str, Optional[List[Tuple[str, int]]]] = {
    "with_default_preemption": None,
    "without_default_preemption": None,
}

# Plots (all configs): still only show subset (both blocking variants)
INCLUDE_PLOTS_ALL: Dict[str, Optional[List[Tuple[str, int]]]] = {
    "with_default_preemption": [
        ("schedulingfailure", 1),
        ("schedulingfailure", 0),
        ("periodic8s", 1),
        ("periodic8s", 0),
        ("stable-queue-2s", 1),
        ("stable-queue-2s", 0),
    ],
    "without_default_preemption": [
        ("schedulingfailure", 1),
        ("schedulingfailure", 0),
        ("periodic8s", 1),
        ("periodic8s", 0),
        ("stable-queue-2s", 1),
        ("stable-queue-2s", 0),
    ],
}

# Delta plots/table: periodic + stable-queue mode differences (right - left)
PERIODIC_STABLE_DELTA_PAIRS = [
    ("Periodic diff. (8s→32s) (blocking)", "periodic8s", "periodic32s", 1),
    ("Stable-queue diff. (2s→8s) (blocking)", "stable-queue-2s", "stable-queue-8s", 1),
]

# =============================================================================
# Plot y-axis configuration
# =============================================================================


@dataclass(frozen=True)
class YAxisConfig:
    scale: str  # "linear" or "symlog"
    ylim: Tuple[float, float]
    symlog_linthresh: float = 1.0  # only used for symlog


PLOT_Y: Dict[str, Dict[str, YAxisConfig]] = {
    "with_default_preemption": {
        "util": YAxisConfig(scale="linear", ylim=(-4.0, 4.0)),
        "latency": YAxisConfig(scale="symlog", ylim=(-1e5 - 1.0, 1e5 + 1.0), symlog_linthresh=SYMLOG_LINTHRESH_LAT_MS),
        "deletions": YAxisConfig(scale="symlog", ylim=(-1e4 - 1.0, 1e4 + 1.0), symlog_linthresh=SYMLOG_LINTHRESH_DELETIONS),
        # solver/plans configured dynamically (non-negative, not symmetric)
    },
    "without_default_preemption": {
        "util": YAxisConfig(scale="linear", ylim=(-4.0, 4.0)),
        "latency": YAxisConfig(scale="symlog", ylim=(-1e5 - 1.0, 1e5 + 1.0), symlog_linthresh=SYMLOG_LINTHRESH_LAT_MS),
        "deletions": YAxisConfig(scale="symlog", ylim=(-1e4 - 1.0, 1e4 + 1.0), symlog_linthresh=SYMLOG_LINTHRESH_DELETIONS),
    },
}

PLOT_Y_DELTAS: Dict[str, Dict[str, YAxisConfig]] = {
    "with_default_preemption": {
        "util": YAxisConfig(scale="linear", ylim=(-1.0, 1.0)),
        "latency": YAxisConfig(scale="symlog", ylim=(-1e4 - 1.0, 1e4 + 1.0), symlog_linthresh=SYMLOG_LINTHRESH_LAT_MS),
        "deletions": YAxisConfig(scale="symlog", ylim=(-1e3 - 1.0, 1e3 + 1.0), symlog_linthresh=SYMLOG_LINTHRESH_DELETIONS),
        # solver/plans deltas configured dynamically (symmetric around 0)
    },
    "without_default_preemption": {
        "util": YAxisConfig(scale="linear", ylim=(-1.0, 1.0)),
        "latency": YAxisConfig(scale="symlog", ylim=(-1e4 - 1.0, 1e4 + 1.0), symlog_linthresh=SYMLOG_LINTHRESH_LAT_MS),
        "deletions": YAxisConfig(scale="symlog", ylim=(-1e3 - 1.0, 1e3 + 1.0), symlog_linthresh=SYMLOG_LINTHRESH_DELETIONS),
    },
}

# =============================================================================
# Parsing helpers
# =============================================================================


def parse_job(job_name: str) -> Optional[Tuple[int, int, float]]:
    m = JOB_RE.fullmatch(str(job_name).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def parse_plugin_config(plugin_config: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for tok in str(plugin_config).split("_"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return kv


def canonical_mode(mode: str) -> str:
    s = str(mode).strip().lower()
    s_compact = s.replace("-", "").replace("_", "")
    if s_compact in {"schedulingfailure", "schedfailure"}:
        return "schedulingfailure"

    m = re.fullmatch(r"periodic-?([0-9.]+)s", s)
    if m:
        return f"periodic{m.group(1)}s"

    m = re.fullmatch(r"(?:stable-queue|stablequeue)-?([0-9.]+)s", s)
    if m:
        return f"stable-queue-{m.group(1)}s"

    return s


# =============================================================================
# RowKey + style derived from MODE_SPECS
# =============================================================================


@dataclass(frozen=True)
class RowKey:
    mode: str
    blocking: int
    defpreempt: int

    @staticmethod
    def from_plugin_config(plugin_config: str) -> "RowKey":
        kv = parse_plugin_config(plugin_config)
        mode = canonical_mode(kv.get("mode", "unknown"))
        blocking = 1 if str(kv.get("blocking", "0")).lower() in {"1", "true", "yes"} else 0
        try:
            defpreempt = int(str(kv.get("defpreempt", "0")))
        except Exception:
            defpreempt = 0
        return RowKey(mode=mode, blocking=blocking, defpreempt=defpreempt)


_SPEC_BY_MODE_BLOCK: Dict[Tuple[str, int], ModeSpec] = {(s.mode, int(s.blocking)): s for s in MODE_SPECS}


def rk_rank(rk: RowKey) -> int:
    spec = _SPEC_BY_MODE_BLOCK.get((rk.mode, int(rk.blocking)))
    return int(spec.rank) if spec else 10_000


def rk_abbr(rk: RowKey) -> str:
    spec = _SPEC_BY_MODE_BLOCK.get((rk.mode, int(rk.blocking)))
    return spec.abbr if spec else f"{rk.mode}:{int(rk.blocking)}"


def rk_label(rk: RowKey) -> str:
    spec = _SPEC_BY_MODE_BLOCK.get((rk.mode, int(rk.blocking)))
    return spec.label if spec else rk_abbr(rk)


def rk_color(rk: RowKey):
    set2 = plt.get_cmap("Set2").colors
    set3 = plt.get_cmap("Set3").colors
    spec = _SPEC_BY_MODE_BLOCK.get((rk.mode, int(rk.blocking)))
    if not spec:
        return set3[0]
    palette = set2 if spec.palette == "set2" else set3
    return palette[int(spec.color_idx) % len(palette)]


def sort_rks(rks: Iterable[RowKey]) -> List[RowKey]:
    return sorted(set(rks), key=lambda r: (rk_rank(r), r.mode, int(r.blocking), int(r.defpreempt)))


# =============================================================================
# Formatting helpers
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


def fmt_unsigned_float(x: object, decimals: int, nan_s: str) -> str:
    if not is_finite(x):
        return nan_s
    v = round(float(x), int(decimals))
    if v == 0.0:
        v = 0.0
    return f"{v:.{decimals}f}"


def fmt_pm(
    mean_v: object,
    std_v: object,
    *,
    mean_signed: bool,
    mean_dec: int,
    std_dec: int,
    latex: bool,
) -> str:
    ns = nan_str(latex)

    # If mean missing -> missing
    if not is_finite(mean_v):
        return ns

    m_val = float(mean_v)

    # If std missing, print mean only (no ±)
    if not is_finite(std_v):
        m = f"{m_val:+.{mean_dec}f}" if mean_signed else f"{m_val:.{mean_dec}f}"
        return rf"\ensuremath{{{m}}}" if latex else m

    s_val = abs(float(std_v))

    m = f"{m_val:+.{mean_dec}f}" if mean_signed else f"{m_val:.{mean_dec}f}"
    s = f"{s_val:.{std_dec}f}"
    return rf"\ensuremath{{{m}\,\pm\,{s}}}" if latex else f"{m} ± {s}"




def arrival_group_label(a: float, *, latex: bool) -> str:
    a_i = int(a) if abs(a - round(a)) < 1e-9 else a
    if latex:
        return rf"inter-arrival = {a_i}\,s"
    return f"inter-arrival = {a_i} s"


def priority_caption(priorities: int, *, latex: bool, without_default_preemption: bool) -> str:
    if latex:
        base = rf"\#priorities = {priorities}" + (r" (w/o priorities)" if priorities == 1 else r" (w/ priorities)")
        return base + (r" (w/o default preemption)" if without_default_preemption else "")
    base = f"#priorities = {priorities}" + (" (w/o priorities)" if priorities == 1 else " (w/ priorities)")
    return base + (" (w/o default preemption)" if without_default_preemption else "")


def rk_label_tex(rk: RowKey) -> str:
    return rf"\makecell[l]{{{rk_label(rk)}}}"


def view_suffix(view: ViewConfig) -> str:
    return "defaultpreemption=1" if int(view.defpreempt_value) == 1 else "defaultpreemption=0"


# =============================================================================
# Data access
# =============================================================================


def has_any_std_cols(df: pd.DataFrame) -> bool:
    return any(str(c).endswith("_std") for c in df.columns)


def std_col_of(mean_col: str) -> str:
    return mean_col[:-5] + "_std" if mean_col.endswith("_mean") else mean_col + "_std"


def load_results(results_csv: Path) -> pd.DataFrame:
    if not results_csv.exists():
        raise SystemExit(f"Not found: {results_csv}")

    df = pd.read_csv(results_csv)

    if "delta_U_pct_eff_mean" not in df.columns:
        raise SystemExit(
            f"{results_csv} missing usage column: expected 'delta_U_pct_eff_mean'"
        )

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{results_csv} missing columns: {', '.join(missing)}")

    parsed = df["job_name"].map(parse_job)
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), "job_name"].head(5).tolist()
        raise SystemExit(f"Could not parse some job_name values (examples): {bad}")

    df["nodes"] = parsed.map(lambda t: t[0] if t is not None else np.nan).astype(int)
    df["priorities"] = parsed.map(lambda t: t[1] if t is not None else np.nan).astype(int)
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
    priorities: int,
    arrival_s: float,
    rk: RowKey,
    col: str,
) -> float:
    key = (int(nodes), int(priorities), float(arrival_s), str(rk.mode), int(rk.blocking), int(rk.defpreempt))
    try:
        return float(lookup.at[key, col])
    except KeyError:
        return float("nan")


def lookup_mean_std(
    lookup: pd.DataFrame,
    *,
    nodes: int,
    priorities: int,
    arrival_s: float,
    rk: RowKey,
    mean_col: str,
) -> Tuple[float, float]:
    m = lookup_value(lookup, nodes=nodes, priorities=priorities, arrival_s=arrival_s, rk=rk, col=mean_col)
    s_col = std_col_of(mean_col)
    try:
        s = lookup_value(lookup, nodes=nodes, priorities=priorities, arrival_s=arrival_s, rk=rk, col=s_col)
    except Exception:
        s = float("nan")
    return m, s


# =============================================================================
# Inclusion helpers
# =============================================================================


def included_pairs_for_view(
    view: ViewConfig, include_map: Dict[str, Optional[List[Tuple[str, int]]]]
) -> Optional[set[Tuple[str, int]]]:
    items = include_map.get(view.name, None)
    if items is None:
        return None
    return {(canonical_mode(m), int(b)) for (m, b) in items}


def filter_rks_for_view(
    rks: Iterable[RowKey],
    view: ViewConfig,
    include_map: Dict[str, Optional[List[Tuple[str, int]]]],
) -> List[RowKey]:
    pairs = included_pairs_for_view(view, include_map)
    out: List[RowKey] = []
    for rk in rks:
        if int(rk.defpreempt) != int(view.defpreempt_value):
            continue
        if pairs is not None and (rk.mode, int(rk.blocking)) not in pairs:
            continue
        out.append(rk)
    return sort_rks(out)


# =============================================================================
# Tables (LaTeX only)
# =============================================================================

MetricGetter = Callable[[int, RowKey, int, float], Any]  # (priorities, rk, nodes, arrival) -> value


@dataclass(frozen=True)
class MetricRow:
    latex_label: str
    getter: MetricGetter
    fmt_kind: str  # "signed_float" | "unsigned_int" | "custom"
    decimals: int = TABLE_DECIMALS
    scale: float = 1.0
    formatter: Optional[Callable[[Any, bool], str]] = None  # (value, latex) -> str


def _fmt_metric_value(v: Any, *, row: MetricRow, latex: bool) -> str:
    ns = nan_str(latex)

    if row.formatter is not None:
        try:
            return row.formatter(v, latex)
        except Exception:
            return ns

    if not is_finite(v):
        return ns

    vv = float(v) * float(row.scale)
    if row.fmt_kind == "signed_float":
        return fmt_signed(vv, row.decimals, ns)
    if row.fmt_kind == "unsigned_int":
        return fmt_unsigned_int(vv, ns)
    return fmt_signed(vv, row.decimals, ns)


def metric_header_tex(label: str) -> str:
    """
    Use makecell to allow controlled line breaks (and keep top-aligned headers).
    IMPORTANT: when splitting math, restart math mode on each line.
    """
    s = str(label).strip()

    def _split_core(core: str) -> Optional[Tuple[str, str]]:
        core = core.strip()
        # LaTeX style: "... \;(\mathrm{ms})"
        if r"\;(" in core:
            left, right = core.split(r"\;(", 1)
            return left.strip(), ("(" + right.strip())
        # Plain text style: "... (ms)"
        if " (" in core and core.endswith(")"):
            left, right = core.rsplit(" (", 1)
            return left.strip(), ("(" + right.strip())
        return None

    # Math-wrapped: $ ... $
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        core = s[1:-1].strip()
        parts = _split_core(core)
        if parts is None:
            return rf"\makecell{{${core}$}}"
        left, right = parts
        return rf"\makecell{{${left}$\\${right}$}}"

    # Plain text
    parts = _split_core(s)
    if parts is None:
        return rf"\makecell{{{s}}}"

    left, right = parts
    return rf"\makecell{{{left}\\{right}}}"


def latex_metric_matrix_tables(
    *,
    out_path: Path,
    nodes_order: List[int],
    arrivals_order: List[float],
    modes: List[RowKey],
    metrics: List[MetricRow],
    priorities: int,
    without_default_preemption: bool,  # kept for compatibility with old signature
) -> None:
    """
    Columns: (empty) | metric1 | metric2 | ...
    Rows: mode1..modeM
    Blocks stacked for each (#nodes, inter-arrival).
    """
    _ = without_default_preemption  # intentionally unused; kept to preserve "logic surface"

    n_metrics = len(metrics)
    total_cols = 1 + n_metrics

    def _ai(a: float) -> str:
        return str(int(a)) if abs(a - round(a)) < 1e-9 else str(a)

    lines: List[str] = []
    colspec = "l" + (" " + "c" * n_metrics if n_metrics > 0 else "")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    metric_hdrs = [metric_header_tex(m.latex_label) for m in metrics]
    if metric_hdrs:
        lines.append(" & " + " & ".join(metric_hdrs) + r" \\")
    else:
        lines.append(r"\\")

    lines.append(r"\midrule")

    first_block = True
    for n in nodes_order:
        for a in arrivals_order:
            if not first_block:
                lines.append(r"\midrule")
            first_block = False

            lines.append(
                rf"\multicolumn{{{total_cols}}}{{l}}{{\#nodes = {int(n)}, inter-arrival = {_ai(float(a))}\,s}} \\"
            )
            lines.append(r"\midrule")

            for rk in modes:
                cells: List[str] = []
                for mrow in metrics:
                    v = mrow.getter(int(priorities), rk, int(n), float(a))
                    cells.append(_fmt_metric_value(v, row=mrow, latex=True))
                lines.append(f"{rk_label_tex(rk)} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Mode-delta table (ONE table, contains both views) - LaTeX only
# =============================================================================


@dataclass(frozen=True)
class ModeDeltaCol:
    abbr: str
    left_mode: str
    right_mode: str
    blocking: int


def _mode_delta_value(
    lookup: pd.DataFrame,
    *,
    col: str,
    nodes: int,
    priorities: int,
    arrival_s: float,
    defpreempt: int,
    left_mode: str,
    right_mode: str,
    blocking: int,
) -> float:
    rk_l = RowKey(mode=canonical_mode(left_mode), blocking=int(blocking), defpreempt=int(defpreempt))
    rk_r = RowKey(mode=canonical_mode(right_mode), blocking=int(blocking), defpreempt=int(defpreempt))
    vl = lookup_value(lookup, nodes=nodes, priorities=priorities, arrival_s=arrival_s, rk=rk_l, col=col)
    vr = lookup_value(lookup, nodes=nodes, priorities=priorities, arrival_s=arrival_s, rk=rk_r, col=col)
    if not is_finite(vl) or not is_finite(vr):
        return float("nan")
    return float(vr) - float(vl)


def write_mode_delta_table_tex(
    *,
    out_tex: Path,
    lookup: pd.DataFrame,
    views: Sequence[ViewConfig],
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_order: List[int],
) -> None:
    delta_cols: List[ModeDeltaCol] = [
        ModeDeltaCol("PR 8→32 (B)", "periodic8s", "periodic32s", blocking=1),
        ModeDeltaCol("SQ 2→8 (B)", "stable-queue-2s", "stable-queue-8s", blocking=1),
    ]

    # (col, latex_label, fmt_kind, decimals)
    metric_specs: List[Tuple[str, str, str, int]] = [
        ("delta_U_pct_eff_mean", r"$\Delta\ \mathrm{usage}\;(\%)$", "signed_float", 2),
        ("delta_L_ms_total_mean", r"$\Delta\ \mathrm{latency}_{\mathrm{total}}\;(\mathrm{ms})$", "signed_float", 0),
        ("delta_D_num_total_mean", r"$\Delta\ \mathrm{deletions}_{\mathrm{total}}$", "signed_float", 1),
        ("solver_attempts_mean", r"$\Delta\ \mathrm{solver\ runs}$", "signed_float", 1),
        ("plan_activated_mean", r"$\Delta\ \mathrm{plan\ activations}$", "signed_float", 1),
    ]

    tex_lines: List[str] = []
    for view in views:
        tex_lines.append(r"% ------------------------------------------------------------")
        tex_lines.append(rf"% Mode-delta table section: {view.name}")
        tex_lines.append(r"% ------------------------------------------------------------")
        tex_lines.append("")

        for priorities in sorted(priorities_order):
            n_modes = len(delta_cols)
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
            mode_hdr = " & " + " & ".join([c.abbr for _a in arrivals_order for c in delta_cols]) + r" \\"

            tex_lines.append(rf"\begin{{tabular}}{{{tab_spec}}}")
            tex_lines.append(r"\toprule")
            tex_lines.append(
                rf"\multicolumn{{{total_cols}}}{{l}}{{Mode $\Delta$ (PR 8$\rightarrow$32, SQ 2$\rightarrow$8), {priority_caption(priorities, latex=True, without_default_preemption=view.without_default_preemption_caption)}}} \\"
            )
            tex_lines.append(r"\addlinespace[0.2em]")
            tex_lines.append(arrival_hdr)
            tex_lines.append("".join(cmid))
            tex_lines.append(mode_hdr)
            tex_lines.append(r"\midrule")

            for ni, n in enumerate(nodes_order):
                if ni > 0:
                    tex_lines.append(r"\midrule")
                tex_lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\#nodes = {n}}} \\")
                tex_lines.append(r"\midrule")

                for (col, latex_lbl, _fmt_kind, dec) in metric_specs:
                    cells: List[str] = []
                    for a in arrivals_order:
                        for c in delta_cols:
                            v = _mode_delta_value(
                                lookup,
                                col=col,
                                nodes=n,
                                priorities=priorities,
                                arrival_s=float(a),
                                defpreempt=int(view.defpreempt_value),
                                left_mode=c.left_mode,
                                right_mode=c.right_mode,
                                blocking=int(c.blocking),
                            )
                            ns = nan_str(True)
                            cells.append(fmt_signed(v, dec, ns) if is_finite(v) else ns)
                    tex_lines.append(f"{latex_lbl} & " + " & ".join(cells) + r" \\")

            tex_lines.append(r"\bottomrule")
            tex_lines.append(r"\end{tabular}")
            tex_lines.append("")
        tex_lines.append("")

    out_tex.write_text("\n".join(tex_lines), encoding="utf-8")


# =============================================================================
# Plot helpers
# =============================================================================

def ycfg(view: ViewConfig, key: str, *, kind: str = "all") -> YAxisConfig:
    if kind == "deltas":
        d = PLOT_Y_DELTAS.get(view.name)
        if not d or key not in d:
            raise SystemExit(f"Missing PLOT_Y_DELTAS config for view={view.name} key={key}")
        return d[key]
    d = PLOT_Y.get(view.name)
    if not d or key not in d:
        raise SystemExit(f"Missing PLOT_Y config for view={view.name} key={key}")
    return d[key]


def arrival_tick_label(a: float) -> str:
    a_i = int(a) if abs(a - round(a)) < 1e-9 else a
    return f"{a_i}s"


def arrival_tick_label_with_axis(a: float, xi: int, n_arr: int) -> str:
    base = arrival_tick_label(a)
    mid = n_arr // 2
    if xi == mid:
        return base + "\ninter-arrival (s)"
    return base


YOfFn = Callable[[RowKey, int, float, int], float]  # (rk, nodes, arrival_s, priorities) -> y


def set_linear_yticks(ax: plt.Axes, ylim: Tuple[float, float], *, min_ticks: int = PLOT_MIN_LINEAR_YTICKS) -> None:
    ylo, yhi = float(ylim[0]), float(ylim[1])

    y0 = int(math.ceil(ylo))
    y1 = int(math.floor(yhi))
    ticks = list(range(y0, y1 + 1))

    if len(ticks) < int(min_ticks):
        span = max(1e-12, yhi - ylo)
        if span >= 4.0:
            dec = 0
        elif span >= 1.0:
            dec = 1
        else:
            dec = 2
        ticks = [round(float(t), dec) for t in np.linspace(ylo, yhi, int(min_ticks))]

    ax.set_yticks(ticks)


def nice_step(span: float, target_ticks: int) -> float:
    """
    Pick a 'nice' step size ~ span/target_ticks from {1,2,5} * 10^k.
    """
    span = float(span)
    if span <= 0:
        return 1.0
    raw = span / max(1, int(target_ticks))
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10**exp
    candidates = [1 * base, 2 * base, 5 * base, 10 * base]
    return min(candidates, key=lambda s: abs(s - raw))


def set_symmetric_count_yticks(ax: plt.Axes, ylim: Tuple[float, float], *, max_ticks_total: int = 7) -> None:
    """
    Sparse ticks symmetric around 0 for delta-count axes (can be negative).
    Ensures negative ticks exist so horizontal grid lines also show below 0.
    """
    ylo, yhi = float(ylim[0]), float(ylim[1])
    hi = max(abs(ylo), abs(yhi))
    if hi <= 0:
        ax.set_yticks([0])
        return

    max_ticks_total = max(3, int(max_ticks_total))
    per_side = max(1, (max_ticks_total - 1) // 2)

    step = nice_step(hi, per_side)
    step = max(1e-12, float(step))

    ticks: List[float] = [0.0]
    for i in range(1, per_side + 1):
        t = i * step
        ticks.extend([-t, t])

    ticks = sorted([t for t in ticks if t >= ylo - 1e-9 and t <= yhi + 1e-9])

    if all(abs(t - round(t)) < 1e-9 for t in ticks):
        ticks = [int(round(t)) for t in ticks]  # type: ignore[assignment]

    ax.set_yticks(ticks)


def set_count_yticks(ax: plt.Axes, ylim: Tuple[float, float], *, max_ticks: int = 5) -> None:
    """
    Sparse, integer-ish ticks for non-negative count axes.
    """
    ylo, yhi = float(ylim[0]), float(ylim[1])
    ylo = max(0.0, ylo)
    if yhi <= 0:
        ax.set_yticks([0])
        return

    step = nice_step(yhi - ylo, max_ticks - 1)
    step = max(1.0, step)

    ticks: List[float] = []
    t = 0.0
    for _ in range(50):
        ticks.append(t)
        t += step
        if t > yhi + 1e-9:
            break

    if abs(step - round(step)) < 1e-9:
        ticks = [int(round(x)) for x in ticks]  # type: ignore[assignment]

    ax.set_yticks(ticks)


def x_from_left_with_pad_points(fig: plt.Figure, left: float, pad_pt: float) -> float:
    pad_frac = float(pad_pt) / (72.0 * float(fig.get_figwidth()))
    return max(0.0, float(left) - pad_frac)


def draw_points_on_ax(
    *,
    ax: plt.Axes,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities: int,
    series: List[RowKey],
    y_of: YOfFn,
    ylim: Tuple[float, float],
    show_xticklabels: bool,
    show_yticklabels: bool,
    y_scale: str = "linear",
    symlog_linthresh: float = 1.0,
    arrival_x_spacing: float,
    mode_x_spacing: float,
    y_tick_strategy: str = "auto",
    color_of: Optional[Callable[[RowKey], Any]] = None,
) -> None:
    """
    Shared point plotting logic.

    - Two node counts are plotted with different markers:
        nodes_order[0] -> circles
        nodes_order[1] -> squares
    """
    n_arr = len(arrivals_order)
    step = float(arrival_x_spacing)

    boundaries = [i * step for i in range(n_arr + 1)]
    x_base = [(i + 0.5) * step for i in range(n_arr)]

    m = max(1, len(series))
    if m <= 1:
        mode_spacing = 0.0
    else:
        max_allowed = 0.45 * step
        mode_spacing = min(float(mode_x_spacing), (2.0 * max_allowed) / float(m - 1))

    _color_of = color_of if color_of is not None else rk_color

    for i, rk in enumerate(series):
        color = _color_of(rk)
        mode_offset = (i - (m - 1) / 2.0) * mode_spacing

        for xi, a in enumerate(arrivals_order):
            x = x_base[xi] + mode_offset
            y0 = y_of(rk, nodes_order[0], a, priorities)
            y1 = y_of(rk, nodes_order[1], a, priorities)

            if is_finite(y0):
                ax.plot(
                    [x],
                    [y0],
                    marker="o",
                    linestyle="None",
                    markersize=PLOT_MARKER_SIZE,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=PLOT_MARKER_LINEWIDTH,
                )
            if is_finite(y1):
                ax.plot(
                    [x],
                    [y1],
                    marker="s",
                    linestyle="None",
                    markersize=PLOT_MARKER_SIZE,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=PLOT_MARKER_LINEWIDTH,
                )

    if y_scale == "symlog":
        ax.set_yscale("symlog", base=SYMLOG_BASE, linthresh=symlog_linthresh, linscale=SYMLOG_LINSCALE)
    else:
        ax.set_yscale("linear")

    ax.axhline(0.0, linewidth=0.8, color="black", linestyle="-", alpha=0.7)
    ax.set_ylim(float(ylim[0]), float(ylim[1]))

    if y_scale == "linear":
        if y_tick_strategy == "count_sparse":
            set_count_yticks(ax, ylim, max_ticks=5)
        elif y_tick_strategy == "count_sparse_symmetric":
            set_symmetric_count_yticks(ax, ylim, max_ticks_total=7)
        else:
            set_linear_yticks(ax, ylim)

    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.margins(x=0)

    for bx in boundaries:
        ax.axvline(bx, linewidth=0.8, color="black", linestyle="--", alpha=0.7, zorder=0)

    ax.set_xticks(boundaries)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE, pad=PLOT_TICK_PAD)

    if show_xticklabels:
        ax.set_xticklabels([""] * len(boundaries))
        for xi, a in enumerate(arrivals_order):
            ls = 1.0
            if xi == len(arrivals_order) // 2:
                ls = 1.35
            ax.text(
                x_base[xi],
                -0.03,
                arrival_tick_label_with_axis(a, xi, len(arrivals_order)),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=PLOT_TICK_FONTSIZE,
                clip_on=False,
                linespacing=ls,
            )
    else:
        ax.tick_params(labelbottom=False)

    if not show_yticklabels:
        ax.tick_params(labelleft=False)

    for y in ax.get_yticks():
        if abs(y) < 1e-8:
            continue
        ax.axhline(y, linewidth=0.8, color="black", linestyle="--", alpha=0.15, zorder=0)


def compute_nonnegative_ylim_for_col(
    *,
    lookup: pd.DataFrame,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_order: List[int],
    series: List[RowKey],
    col: str,
    pad_frac: float = 0.08,
) -> Tuple[float, float]:
    vals: List[float] = []
    for rk in series:
        for n in nodes_order:
            for a in arrivals_order:
                for k in priorities_order:
                    v = lookup_value(lookup, nodes=n, priorities=k, arrival_s=a, rk=rk, col=col)
                    if is_finite(v):
                        vals.append(float(v))
    m = max(vals) if vals else 0.0
    hi = float(m) * (1.0 + float(pad_frac)) if m > 0.0 else 1.0
    return (0.0, hi)


def compute_symmetric_ylim_for_yfns(
    *,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_order: List[int],
    series: List[RowKey],
    y_fns: List[YOfFn],
    pad_frac: float = 0.08,
) -> Tuple[float, float]:
    vals: List[float] = []
    for i, rk in enumerate(series):
        y_of = y_fns[i]
        for n in nodes_order:
            for a in arrivals_order:
                for k in priorities_order:
                    v = y_of(rk, n, a, k)
                    if is_finite(v):
                        vals.append(abs(float(v)))
    m = max(vals) if vals else 0.0
    hi = float(m) * (1.0 + float(pad_frac)) if m > 0.0 else 1.0
    return (-hi, hi)


def make_grid_plot(
    *,
    lookup: pd.DataFrame,
    view: ViewConfig,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
    series: List[RowKey],
    out_stem: str,
    grid_left: float,
    grid_right: float,
    grid_bottom: float,
    grid_top: float,
    grid_wspace: float,
    grid_hspace: float,
    ylabel_pad_pt: float,
    legend_pad: float,
    legend_ncol: int,
    arrival_x_spacing: float,
    mode_x_spacing: float,
    ylim_solver: Tuple[float, float],
    ylim_plans: Tuple[float, float],
) -> None:
    def y_from_col(*, col: str, scale: float) -> YOfFn:
        def _y(rk: RowKey, nodes: int, a: float, priorities: int) -> float:
            v = lookup_value(lookup, nodes=nodes, priorities=priorities, arrival_s=a, rk=rk, col=col)
            return float(v) * float(scale) if is_finite(v) else float("nan")
        return _y

    present_labels = {rk_label(rk) for rk in series}
    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []
    for rk in series:
        lab = rk_label(rk)
        if lab in present_labels and lab not in legend_labels:
            legend_labels.append(lab)
            legend_handles.append(Line2D([0], [0], color=rk_color(rk), linewidth=1.8))

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=GRID_FIGSIZE, sharex=True)
    axes[0, 0].set_title(f"#priorities = {priorities_cols[0]}", fontsize=PLOT_TITLE_FONTSIZE)
    axes[0, 1].set_title(f"#priorities = {priorities_cols[1]}", fontsize=PLOT_TITLE_FONTSIZE)

    for col_i, k in enumerate(priorities_cols):
        yc = ycfg(view, "util", kind="all")
        draw_points_on_ax(
            ax=axes[0, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=series,
            y_of=y_from_col(col="delta_U_pct_eff_mean", scale=1.0),
            ylim=yc.ylim,
            show_xticklabels=False,
            show_yticklabels=(col_i == 0),
            y_scale=yc.scale,
            symlog_linthresh=yc.symlog_linthresh,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
        )

    for col_i, k in enumerate(priorities_cols):
        yc = ycfg(view, "latency", kind="all")
        draw_points_on_ax(
            ax=axes[1, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=series,
            y_of=y_from_col(col="delta_L_ms_total_mean", scale=1.0),
            ylim=yc.ylim,
            show_xticklabels=False,
            show_yticklabels=(col_i == 0),
            y_scale=yc.scale,
            symlog_linthresh=yc.symlog_linthresh,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
        )

    for col_i, k in enumerate(priorities_cols):
        yc = ycfg(view, "deletions", kind="all")
        draw_points_on_ax(
            ax=axes[2, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=series,
            y_of=y_from_col(col="delta_D_num_total_mean", scale=1.0),
            ylim=yc.ylim,
            show_xticklabels=False,
            show_yticklabels=(col_i == 0),
            y_scale=yc.scale,
            symlog_linthresh=yc.symlog_linthresh,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
        )

    for col_i, k in enumerate(priorities_cols):
        draw_points_on_ax(
            ax=axes[3, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=series,
            y_of=y_from_col(col="solver_attempts_mean", scale=1.0),
            ylim=ylim_solver,
            show_xticklabels=False,
            show_yticklabels=(col_i == 0),
            y_scale="linear",
            symlog_linthresh=1.0,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
            y_tick_strategy="count_sparse",
        )

    for col_i, k in enumerate(priorities_cols):
        draw_points_on_ax(
            ax=axes[4, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=series,
            y_of=y_from_col(col="plan_activated_mean", scale=1.0),
            ylim=ylim_plans,
            show_xticklabels=True,
            show_yticklabels=(col_i == 0),
            y_scale="linear",
            symlog_linthresh=1.0,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
            y_tick_strategy="count_sparse",
        )

    fig.subplots_adjust(
        left=grid_left,
        right=grid_right,
        bottom=grid_bottom,
        top=grid_top,
        wspace=grid_wspace,
        hspace=grid_hspace,
    )

    bbox_l = axes[0, 0].get_position()
    bbox_r = axes[0, 1].get_position()
    x_center_grid = 0.5 * (bbox_l.x0 + bbox_r.x1)
    y_top_grid = max(bbox_l.y1, bbox_r.y1)
    legend_y = min(0.98, y_top_grid + float(legend_pad))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(x_center_grid, legend_y),
        ncol=min(int(legend_ncol), len(legend_labels)),
        fontsize=PLOT_LEGEND_FONTSIZE,
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
    )

    x_text = x_from_left_with_pad_points(fig, grid_left, ylabel_pad_pt)
    row_labels = [
        r"$\mathrm{diff.}\ \mathrm{usage}\;(\%)$",
        r"$\mathrm{diff.}\ \mathrm{latency}\;(\mathrm{ms})$",
        r"$\mathrm{diff.}\ \mathrm{deletions}$",
        r"$\mathrm{solver\ runs}$",
        r"$\mathrm{plan\ activations}$",
    ]
    for r, text in enumerate(row_labels):
        bbox = axes[r, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(x_text, y_center, text, rotation=90, va="center", ha="right", fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    fig.savefig(OUT_FIGURES_DIR / f"{out_stem}.png", dpi=PLOT_FIGURE_DPI)
    fig.savefig(OUT_FIGURES_DIR / f"{out_stem}.pdf")
    plt.close(fig)


def make_mode_delta_series(
    *,
    lookup: pd.DataFrame,
    defpreempt: int,
    label: str,
    left_mode: str,
    right_mode: str,
    blocking: int,
) -> Tuple[str, YOfFn, YOfFn, YOfFn, YOfFn, YOfFn]:
    rk_l = RowKey(mode=canonical_mode(left_mode), blocking=int(blocking), defpreempt=int(defpreempt))
    rk_r = RowKey(mode=canonical_mode(right_mode), blocking=int(blocking), defpreempt=int(defpreempt))

    def _dd(col: str) -> YOfFn:
        def _y(_rk_unused: RowKey, nodes: int, a: float, priorities: int) -> float:
            vl = lookup_value(lookup, nodes=nodes, priorities=priorities, arrival_s=a, rk=rk_l, col=col)
            vr = lookup_value(lookup, nodes=nodes, priorities=priorities, arrival_s=a, rk=rk_r, col=col)
            if not is_finite(vl) or not is_finite(vr):
                return float("nan")
            return float(vr) - float(vl)
        return _y

    return (
        label,
        _dd("delta_U_pct_eff_mean"),
        _dd("delta_L_ms_total_mean"),
        _dd("delta_D_num_total_mean"),
        _dd("solver_attempts_mean"),
        _dd("plan_activated_mean"),
    )


def make_grid_plot_custom_series(
    *,
    view: ViewConfig,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_cols: List[int],
    series_labels: List[str],
    y_utils: List[YOfFn],
    y_lats: List[YOfFn],
    y_dels: List[YOfFn],
    y_solvers: List[YOfFn],
    y_plans: List[YOfFn],
    out_stem: str,
    figsize: Tuple[float, float],
    grid_left: float,
    grid_right: float,
    grid_bottom: float,
    grid_top: float,
    grid_wspace: float,
    grid_hspace: float,
    ylabel_pad_pt: float,
    legend_pad: float,
    legend_ncol: int,
    arrival_x_spacing: float,
    mode_x_spacing: float,
) -> None:
    cmap = plt.get_cmap("Set2").colors

    legend_handles: List[Line2D] = []
    for i, _lab in enumerate(series_labels):
        legend_handles.append(Line2D([0], [0], color=cmap[i % len(cmap)], linewidth=1.8))
    legend_labels = list(series_labels)

    fake_series = [RowKey(mode=f"custom{i}", blocking=0, defpreempt=view.defpreempt_value) for i in range(len(series_labels))]

    def color_override(rk: RowKey) -> Any:
        idx = int(rk.mode.replace("custom", "")) if rk.mode.startswith("custom") else 0
        return cmap[idx % len(cmap)]

    ylim_solver = compute_symmetric_ylim_for_yfns(
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_order=priorities_cols,
        series=fake_series,
        y_fns=y_solvers,
    )
    ylim_plans = compute_symmetric_ylim_for_yfns(
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_order=priorities_cols,
        series=fake_series,
        y_fns=y_plans,
    )

    def y_of_from_list(y_list: List[YOfFn]) -> Callable[[RowKey, int, float, int], float]:
        def _y(rk: RowKey, nodes: int, a: float, priorities: int) -> float:
            idx = int(rk.mode.replace("custom", "")) if rk.mode.startswith("custom") else 0
            return y_list[idx](rk, nodes, a, priorities)
        return _y

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=figsize, sharex=True)
    axes[0, 0].set_title(f"#priorities = {priorities_cols[0]}", fontsize=PLOT_TITLE_FONTSIZE)
    axes[0, 1].set_title(f"#priorities = {priorities_cols[1]}", fontsize=PLOT_TITLE_FONTSIZE)

    for col_i, k in enumerate(priorities_cols):
        yc = ycfg(view, "util", kind="deltas")
        draw_points_on_ax(
            ax=axes[0, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=fake_series,
            y_of=y_of_from_list(y_utils),
            ylim=yc.ylim,
            show_xticklabels=False,
            show_yticklabels=(col_i == 0),
            y_scale=yc.scale,
            symlog_linthresh=yc.symlog_linthresh,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
            color_of=color_override,
        )

    for col_i, k in enumerate(priorities_cols):
        yc = ycfg(view, "latency", kind="deltas")
        draw_points_on_ax(
            ax=axes[1, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=fake_series,
            y_of=y_of_from_list(y_lats),
            ylim=yc.ylim,
            show_xticklabels=False,
            show_yticklabels=(col_i == 0),
            y_scale=yc.scale,
            symlog_linthresh=yc.symlog_linthresh,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
            color_of=color_override,
        )

    for col_i, k in enumerate(priorities_cols):
        yc = ycfg(view, "deletions", kind="deltas")
        draw_points_on_ax(
            ax=axes[2, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=fake_series,
            y_of=y_of_from_list(y_dels),
            ylim=yc.ylim,
            show_xticklabels=False,
            show_yticklabels=(col_i == 0),
            y_scale=yc.scale,
            symlog_linthresh=yc.symlog_linthresh,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
            color_of=color_override,
        )

    for col_i, k in enumerate(priorities_cols):
        draw_points_on_ax(
            ax=axes[3, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=fake_series,
            y_of=y_of_from_list(y_solvers),
            ylim=ylim_solver,
            show_xticklabels=False,
            show_yticklabels=(col_i == 0),
            y_scale="linear",
            symlog_linthresh=1.0,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
            y_tick_strategy="count_sparse_symmetric",
            color_of=color_override,
        )

    for col_i, k in enumerate(priorities_cols):
        draw_points_on_ax(
            ax=axes[4, col_i],
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities=k,
            series=fake_series,
            y_of=y_of_from_list(y_plans),
            ylim=ylim_plans,
            show_xticklabels=True,
            show_yticklabels=(col_i == 0),
            y_scale="linear",
            symlog_linthresh=1.0,
            arrival_x_spacing=arrival_x_spacing,
            mode_x_spacing=mode_x_spacing,
            y_tick_strategy="count_sparse_symmetric",
            color_of=color_override,
        )

    fig.subplots_adjust(
        left=grid_left,
        right=grid_right,
        bottom=grid_bottom,
        top=grid_top,
        wspace=grid_wspace,
        hspace=grid_hspace,
    )

    bbox_l = axes[0, 0].get_position()
    bbox_r = axes[0, 1].get_position()
    x_center_grid = 0.5 * (bbox_l.x0 + bbox_r.x1)
    y_top_grid = max(bbox_l.y1, bbox_r.y1)
    legend_y = min(0.98, y_top_grid + float(legend_pad))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(x_center_grid, legend_y),
        ncol=min(int(legend_ncol), len(legend_labels)),
        fontsize=PLOT_LEGEND_FONTSIZE,
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
    )

    x_text = x_from_left_with_pad_points(fig, grid_left, ylabel_pad_pt)
    row_labels = [
        r"$\mathrm{diff.}\ \mathrm{usage}\;(\%)$",
        r"$\mathrm{diff.}\ \mathrm{latency}\;(\mathrm{ms})$",
        r"$\mathrm{diff.}\ \mathrm{deletions}$",
        r"$\mathrm{diff.}\ \mathrm{solver\ runs}$",
        r"$\mathrm{diff.}\ \mathrm{plan\ activations}$",
    ]
    for r, text in enumerate(row_labels):
        bbox = axes[r, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(x_text, y_center, text, rotation=90, va="center", ha="right", fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    fig.savefig(OUT_FIGURES_DIR / f"{out_stem}.png", dpi=PLOT_FIGURE_DPI)
    fig.savefig(OUT_FIGURES_DIR / f"{out_stem}.pdf")
    plt.close(fig)


# =============================================================================
# Metric builders (big tables)
# =============================================================================


def build_metrics_for_big_table(*, lookup: pd.DataFrame, has_std: bool) -> Callable[[int], List[MetricRow]]:
    """
    Returns metrics_big(priorities) -> List[MetricRow].

    If has_std=True, prints "mean ± std" using corresponding *_std columns
    (when present); otherwise prints the mean only.
    """

    def g_mean(col_mean: str) -> MetricGetter:
        return lambda priorities, rk, n, a: lookup_value(lookup, nodes=n, priorities=priorities, arrival_s=a, rk=rk, col=col_mean)

    def g_mean_std(col_mean: str) -> MetricGetter:
        return lambda priorities, rk, n, a: lookup_mean_std(lookup, nodes=n, priorities=priorities, arrival_s=a, rk=rk, mean_col=col_mean)

    def pm_formatter(*, mean_signed: bool, mean_dec: int, std_dec: int) -> Callable[[Any, bool], str]:
        def _f(v: Any, latex: bool) -> str:
            if not isinstance(v, (tuple, list)) or len(v) != 2:
                return nan_str(latex)
            m, s = v[0], v[1]
            return fmt_pm(m, s, mean_signed=mean_signed, mean_dec=mean_dec, std_dec=std_dec, latex=latex)
        return _f

    def metrics_big(priorities: int) -> List[MetricRow]:
        rows: List[MetricRow] = []

        # usage
        if has_std:
            rows.append(
                MetricRow(
                    latex_label=r"$\Delta\mathrm{usage}\;(\%)$",
                    getter=g_mean_std("delta_U_pct_eff_mean"),
                    fmt_kind="custom",
                    formatter=pm_formatter(mean_signed=True, mean_dec=2, std_dec=2),
                )
            )
        else:
            rows.append(
                MetricRow(
                    latex_label=r"$\Delta\mathrm{usage}\;(\%)$",
                    getter=g_mean("delta_U_pct_eff_mean"),
                    fmt_kind="signed_float",
                    decimals=2,
                )
            )

        # latency total
        if has_std:
            rows.append(
                MetricRow(
                    latex_label=r"$\Delta\mathrm{latency}_{\mathrm{total}}\;(\mathrm{ms})$",
                    getter=g_mean_std("delta_L_ms_total_mean"),
                    fmt_kind="custom",
                    formatter=pm_formatter(mean_signed=True, mean_dec=0, std_dec=0),
                )
            )
        else:
            rows.append(
                MetricRow(
                    latex_label=r"$\Delta\mathrm{latency}_{\mathrm{total}}\;(\mathrm{ms})$",
                    getter=g_mean("delta_L_ms_total_mean"),
                    fmt_kind="signed_float",
                    decimals=0,
                )
            )

        # latency per priority
        if priorities != 1:
            for p in (1, 2, 3, 4):
                col = f"delta_L_ms_p{p}_mean"
                if has_std:
                    rows.append(
                        MetricRow(
                            latex_label=rf"$\Delta\mathrm{{latency}}_{{p{p}}}\;(\mathrm{{ms}})$",
                            getter=g_mean_std(col),
                            fmt_kind="custom",
                            formatter=pm_formatter(mean_signed=True, mean_dec=0, std_dec=0),
                        )
                    )
                else:
                    rows.append(
                        MetricRow(
                            latex_label=rf"$\Delta\mathrm{{latency}}_{{p{p}}}\;(\mathrm{{ms}})$",
                            getter=g_mean(col),
                            fmt_kind="signed_float",
                            decimals=0,
                        )
                    )

        # deletions total
        if has_std:
            rows.append(
                MetricRow(
                    latex_label=r"$\Delta\mathrm{deletions}_{\mathrm{total}}$",
                    getter=g_mean_std("delta_D_num_total_mean"),
                    fmt_kind="custom",
                    formatter=pm_formatter(mean_signed=True, mean_dec=1, std_dec=1),
                )
            )
        else:
            rows.append(
                MetricRow(
                    latex_label=r"$\Delta\mathrm{deletions}_{\mathrm{total}}$",
                    getter=g_mean("delta_D_num_total_mean"),
                    fmt_kind="signed_float",
                    decimals=1,
                )
            )

        # deletions per priority
        if priorities != 1:
            for p in (1, 2, 3, 4):
                col = f"delta_D_num_p{p}_mean"
                if has_std:
                    rows.append(
                        MetricRow(
                            latex_label=rf"$\Delta\mathrm{{deletions}}_{{p{p}}}$",
                            getter=g_mean_std(col),
                            fmt_kind="custom",
                            formatter=pm_formatter(mean_signed=True, mean_dec=1, std_dec=1),
                        )
                    )
                else:
                    rows.append(
                        MetricRow(
                            latex_label=rf"$\Delta\mathrm{{deletions}}_{{p{p}}}$",
                            getter=g_mean(col),
                            fmt_kind="signed_float",
                            decimals=1,
                        )
                    )

        # plugin-only counters (non-negative; show mean ± std if available)
        if has_std:
            rows.append(
                MetricRow(
                    latex_label=r"\#solver\\runs",
                    getter=g_mean_std("solver_attempts_mean"),
                    fmt_kind="custom",
                    formatter=pm_formatter(mean_signed=False, mean_dec=0, std_dec=1),
                )
            )
            rows.append(
                MetricRow(
                    latex_label=r"\#plan\\activations",
                    getter=g_mean_std("plan_activated_mean"),
                    fmt_kind="custom",
                    formatter=pm_formatter(mean_signed=False, mean_dec=0, std_dec=1),
                )
            )
        else:
            rows.append(MetricRow(latex_label=r"\#solver\\runs", getter=g_mean("solver_attempts_mean"), fmt_kind="unsigned_int"))
            rows.append(MetricRow(latex_label=r"\#plan\\activations", getter=g_mean("plan_activated_mean"), fmt_kind="unsigned_int"))

        return rows

    return metrics_big


# =============================================================================
# Orchestration helpers
# =============================================================================


def infer_orders(df: pd.DataFrame) -> Tuple[List[int], List[float], List[int]]:
    nodes_order = sorted(df["nodes"].dropna().astype(int).unique().tolist())
    arrivals_order = sorted(df["arrival_s"].dropna().astype(float).unique().tolist())
    priorities_order = sorted(df["priorities"].dropna().astype(int).unique().tolist())
    return nodes_order, arrivals_order, priorities_order


def compute_shared_counter_ylims(
    *,
    lookup: pd.DataFrame,
    df: pd.DataFrame,
    nodes_order: List[int],
    arrivals_order: List[float],
    priorities_order: List[int],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    all_rks = [
        RowKey(mode=m, blocking=int(b), defpreempt=int(d))
        for (m, b, d) in df[["mode", "blocking", "defpreempt"]].drop_duplicates().itertuples(index=False, name=None)
    ]

    series_counters_both: List[RowKey] = []
    for v in VIEWS:
        series_counters_both.extend(filter_rks_for_view(all_rks, v, INCLUDE_PLOTS_ALL))
    series_counters_both = sort_rks(series_counters_both)

    ylim_solver_shared = compute_nonnegative_ylim_for_col(
        lookup=lookup,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_order=priorities_order,
        series=series_counters_both,
        col="solver_attempts_mean",
    )
    ylim_plans_shared = compute_nonnegative_ylim_for_col(
        lookup=lookup,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_order=priorities_order,
        series=series_counters_both,
        col="plan_activated_mean",
    )
    return ylim_solver_shared, ylim_plans_shared


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results(IN_RESULTS)
    lookup = build_lookup(df)
    nodes_order, arrivals_order, priorities_order = infer_orders(df)

    if len(nodes_order) != 2:
        raise SystemExit(f"Expected exactly 2 node values for plots; found {nodes_order}")
    if not arrivals_order:
        raise SystemExit("No arrivals inferred from results_paired.csv")
    if not priorities_order:
        raise SystemExit("No priority values found")

    priorities_cols = [k for k in (1, 4) if k in priorities_order]
    if len(priorities_cols) != 2:
        raise SystemExit(f"Expected priorities [1,4] for the 2 columns; found priorities={priorities_order}")


    has_std = has_any_std_cols(df)

    # Always produce the existing tables: mean-only (do NOT change filenames/format)
    metrics_big_mean_fn = build_metrics_for_big_table(lookup=lookup, has_std=False)
    metrics_map_big_mean = {k: metrics_big_mean_fn(k) for k in priorities_order}

    # Optionally produce new tables: mean ± std (new filenames)
    metrics_map_big_pm: Dict[int, List[MetricRow]] = {}
    if has_std:
        metrics_big_pm_fn = build_metrics_for_big_table(lookup=lookup, has_std=True)
        metrics_map_big_pm = {k: metrics_big_pm_fn(k) for k in priorities_order}

    all_rks = [
        RowKey(mode=m, blocking=int(b), defpreempt=int(d))
        for (m, b, d) in df[["mode", "blocking", "defpreempt"]].drop_duplicates().itertuples(index=False, name=None)
    ]

    ylim_solver_shared, ylim_plans_shared = compute_shared_counter_ylims(
        lookup=lookup,
        df=df,
        nodes_order=nodes_order,
        arrivals_order=arrivals_order,
        priorities_order=priorities_cols,
    )

    produced_tables: List[Path] = []
    produced_figs: List[Path] = []

    # -----------------------
    # Per-view tables + plots
    # -----------------------
    for view in VIEWS:
        suffix = view_suffix(view)

        modes_all = sort_rks(
            [RowKey(mode=s.mode, blocking=int(s.blocking), defpreempt=int(view.defpreempt_value)) for s in MODE_SPECS]
        )

        for k in priorities_cols:
            # 1) Existing table (mean only) — keep identical filename
            out_tex_mean = OUT_TABLES_DIR / f"{view.table_stem}_{suffix}_priorities={k}.tex"
            latex_metric_matrix_tables(
                out_path=out_tex_mean,
                nodes_order=nodes_order,
                arrivals_order=arrivals_order,
                modes=modes_all,
                metrics=metrics_map_big_mean[int(k)],
                priorities=int(k),
                without_default_preemption=view.without_default_preemption_caption,
            )
            produced_tables.append(out_tex_mean)

            # 2) New table (mean ± std) — NEW filename, only if std columns exist
            if has_std:
                out_tex_pm = OUT_TABLES_DIR / f"{view.table_stem}_{suffix}_priorities={k}_with_std.tex"
                latex_metric_matrix_tables(
                    out_path=out_tex_pm,
                    nodes_order=nodes_order,
                    arrivals_order=arrivals_order,
                    modes=modes_all,
                    metrics=metrics_map_big_pm[int(k)],
                    priorities=int(k),
                    without_default_preemption=view.without_default_preemption_caption,
                )
                produced_tables.append(out_tex_pm)

        series_all = filter_rks_for_view(all_rks, view, INCLUDE_PLOTS_ALL)
        out_stem_all = f"{view.figure_stem}_{suffix}"
        make_grid_plot(
            lookup=lookup,
            view=view,
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities_cols=priorities_cols,
            series=series_all,
            out_stem=out_stem_all,
            grid_left=GRID_LEFT_ALL,
            grid_right=GRID_RIGHT_ALL,
            grid_bottom=GRID_BOTTOM_ALL,
            grid_top=GRID_TOP_ALL,
            grid_wspace=GRID_WSPACE_ALL,
            grid_hspace=GRID_HSPACE_ALL,
            ylabel_pad_pt=GRID_YLABEL_PAD_PT_ALL,
            legend_pad=GRID_LEGEND_PAD_ALL,
            legend_ncol=GRID_LEGEND_NCOL_ALL,
            arrival_x_spacing=PLOT_ARRIVAL_X_SPACING_ALL,
            mode_x_spacing=PLOT_MODE_X_SPACING_ALL,
            ylim_solver=ylim_solver_shared,
            ylim_plans=ylim_plans_shared,
        )
        produced_figs.extend([OUT_FIGURES_DIR / f"{out_stem_all}.png", OUT_FIGURES_DIR / f"{out_stem_all}.pdf"])

        labels: List[str] = []
        y_utils: List[YOfFn] = []
        y_lats: List[YOfFn] = []
        y_dels: List[YOfFn] = []
        y_solvers: List[YOfFn] = []
        y_plans: List[YOfFn] = []

        for (lab, left_mode, right_mode, blocking) in PERIODIC_STABLE_DELTA_PAIRS:
            lab2, y_u, y_l, y_d, y_s, y_p = make_mode_delta_series(
                lookup=lookup,
                defpreempt=int(view.defpreempt_value),
                label=lab,
                left_mode=left_mode,
                right_mode=right_mode,
                blocking=int(blocking),
            )
            labels.append(lab2)
            y_utils.append(y_u)
            y_lats.append(y_l)
            y_dels.append(y_d)
            y_solvers.append(y_s)
            y_plans.append(y_p)

        out_stem_ps = f"grid_periodic_vs_stable_{suffix}"
        make_grid_plot_custom_series(
            view=view,
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities_cols=priorities_cols,
            series_labels=labels,
            y_utils=y_utils,
            y_lats=y_lats,
            y_dels=y_dels,
            y_solvers=y_solvers,
            y_plans=y_plans,
            out_stem=out_stem_ps,
            figsize=GRID_FIGSIZE_DELTAS,
            grid_left=GRID_LEFT_DELTAS,
            grid_right=GRID_RIGHT_ALL,
            grid_bottom=GRID_BOTTOM_ALL,
            grid_top=GRID_TOP_ALL,
            grid_wspace=GRID_WSPACE_ALL,
            grid_hspace=GRID_HSPACE_ALL,
            ylabel_pad_pt=GRID_YLABEL_PAD_PT_ALL,
            legend_pad=GRID_LEGEND_PAD_ALL,
            legend_ncol=GRID_LEGEND_NCOL_DELTAS,
            arrival_x_spacing=PLOT_ARRIVAL_X_SPACING_ALL,
            mode_x_spacing=PLOT_MODE_X_SPACING_DELTAS,
        )
        produced_figs.extend([OUT_FIGURES_DIR / f"{out_stem_ps}.png", OUT_FIGURES_DIR / f"{out_stem_ps}.pdf"])

    # -----------------------
    # One table: mode diffs periodic/stable, includes both views
    # -----------------------
    for k in priorities_cols:
        out_tex = OUT_TABLES_DIR / f"table_periodic_vs_stable_priorities={k}.tex"
        write_mode_delta_table_tex(
            out_tex=out_tex,
            lookup=lookup,
            views=VIEWS,
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            priorities_order=[k],
        )
        produced_tables.append(out_tex)

    # -----------------------
    # Summary
    # -----------------------
    print("Produced outputs:\n")
    print("Tables:")
    for p in produced_tables:
        print(f"  - {p}")
    print("\nFigures:")
    for p in produced_figs:
        print(f"  - {p}")
    print("")
    print(f"Wrote tables to:  {OUT_TABLES_DIR}")
    print(f"Wrote figures to: {OUT_FIGURES_DIR}")
    print(f"Std columns detected: {has_std}")
    if has_std:
        print("Also wrote additional tables with suffix: _with_std.tex")


if __name__ == "__main__":
    main()
