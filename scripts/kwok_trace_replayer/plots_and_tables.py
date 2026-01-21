# scripts/kwok_trace_replayer/plots_and_tables.py
#!/usr/bin/env python3
"""
scripts/kwok_trace_replayer/tables_from_sealed.py

Generate tables (plaintext + LaTeX) and plots from sealed outputs.

Inputs (under --in-dir):
  - results_paired.csv   (produced by seal_results.py)

Outputs (under --out-dir):
  Tables (BOTH):
    - <stem>.txt   (plaintext / easy to read)
    - <stem>.tex   (LaTeX)

    Plots (under --out-dir/plots by default):
        - delta_u_eff_run_kmax<K>.<png|pdf>           (Δu_eff in percentage points)
        - delta_latency_total_kmax<K>.<png|pdf>       (Δlatency_total in seconds)
        - delta_deletions_with_defaultpreemption_kmax<K>.<png|pdf> (Δ deletions)

Notes on plots:
  - x axis: arrival rates (μ_A = 4s, 8s, 16s)
  - y axis centered around 0 (symmetric limits)
  - for each series and each arrival: two dots (N=16 and N=32) connected with dashed line
  - by default plots use defpreempt=1 only (main configuration)
  - if --plot-include-defpreempt is set, include BOTH defpreempt=1 and defpreempt=0 (more dots)

Example:
  python -m scripts.kwok_trace_replayer.tables_from_sealed \
    --in-dir  analysis/kwok_trace_replayer/sealed \
    --out-dir analysis/kwok_trace_replayer/plots_and_tables \
    --float-decimals 1 \
        --plot-kmax 1,4
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# -----------------------------
# Plot styling
# -----------------------------

# Smaller markers improve readability when zooming.
MARKER_SIZE = 3.0

# Compact plots: reduce horizontal spacing between arrival groups and between modes.
# These are in data-x units (not pixels).
ARRIVAL_X_SPACING = 0.6
MODE_X_SPACING = 0.05

# Consistent plot size across all figures.
FIGSIZE = (3.5, 3.0)

# Font sizes (adjust to taste).
TITLE_FONTSIZE = 8
TICK_FONTSIZE = 7
LEGEND_FONTSIZE = 5
AXIS_LABEL_FONTSIZE = 7

LEGEND_HANDLE_LENGTH = 0.6
LEGEND_COLUMN_SPACING = 0.5
LEGEND_BORDER_AXES_PAD = 0.4

# Standard legend placement (inside plot).
LEGEND_LOC = "lower left"
LEGEND_NCOL = 2


def add_standard_legend(*, ax: plt.Axes, handles: Sequence[object], labels: Sequence[str]) -> None:
    """Add a standard legend inside the axes (lower-left)."""
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
        loc=LEGEND_LOC,
        ncol=min(int(LEGEND_NCOL), len(ll)),
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        columnspacing=LEGEND_COLUMN_SPACING,
        borderaxespad=LEGEND_BORDER_AXES_PAD,
    )



# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate plaintext+LaTeX tables and two plots from sealed results.")
    p.add_argument("--in-dir", required=True, help="Sealed output dir (contains results_paired.csv).")
    p.add_argument("--out-dir", required=True, help="Output dir (tables + plots subdir).")

    p.add_argument("--nodes", default="16,32", help="Comma-separated nodes values to include (default: 16,32).")
    p.add_argument("--arrivals", default="4,8,16", help="Comma-separated arrivals (seconds) to include (default: 4,8,16).")
    p.add_argument("--float-decimals", type=int, default=1, help="Decimals for float tables (default: 1).")

    # plots
    p.add_argument("--plots-dir", default=None, help="Plots output dir (default: <out-dir>/plots).")
    p.add_argument(
        "--plot-kmax",
        default="1,4",
        help="Comma-separated kmax values to plot (default: 1). Use 'auto' to plot all kmax values present.",
    )
    p.add_argument(
        "--plot-include-defpreempt",
        action="store_true",
        help="If set, include BOTH defpreempt=1 and defpreempt=0 in plots. Default: defpreempt=1 only.",
    )
    return p.parse_args()


# -----------------------------
# Parsing helpers
# -----------------------------

JOB_RE = re.compile(r"nodes=(\d+)_prio=(\d+)_arrival=([0-9.]+)s")

def parse_job(job_name: str) -> Optional[Tuple[int, int, float]]:
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
    s = str(mode).strip().lower()

    # scheduling-failure variants
    if s in {
        "scheduling_failure",
        "scheduling-failure",
        "schedulingfailure",
        "sched-failure",
        "schedfailure",
    }:
        return "scheduling-failure"

    # periodic variants: periodic8s, periodic-8s, periodic8s, etc.
    m = re.fullmatch(r"periodic-?([0-9.]+)s", s)
    if m:
        return f"periodic{m.group(1)}s"

    # stable-queue variants: stable-queue2s, stablequeue2s, stable-queue-2s, stablequeue-2s, ...
    m = re.fullmatch(r"(?:stable-queue|stablequeue)-?([0-9.]+)s", s)
    if m:
        return f"stable-queue-{m.group(1)}s"

    return s



# Explicit base order + base labels (used in BOTH tables + plots)
_MODE_BLOCK_ORDER: List[Tuple[str, Optional[int], str]] = [
    ("scheduling-failure", None, "Scheduling-failure"),
    ("periodic8s", 1, "Periodic-8s (blk)"),
    ("periodic8s", 0, "Periodic-8s (non-blk)"),
    ("stable-queue-2s", 1, "Stable-queue-2s (blk)"),
    ("stable-queue-2s", 0, "Stable-queue-2s (non-blk)"),
]
_MODE_BLOCK_RANK: Dict[Tuple[str, Optional[int]], int] = {
    (m, b): i for i, (m, b, _lbl) in enumerate(_MODE_BLOCK_ORDER)
}
_MODE_BLOCK_LABEL: Dict[Tuple[str, Optional[int]], str] = {
    (m, b): lbl for (m, b, lbl) in _MODE_BLOCK_ORDER
}


def pretty_mode(mode: str) -> str:
    s = canonical_mode(mode)

    if s == "scheduling-failure":
        return "Scheduling-failure"

    m = re.fullmatch(r"periodic([0-9.]+)s", s)
    if m:
        return f"Periodic-{m.group(1)}s"

    m = re.fullmatch(r"stable-queue-?([0-9.]+)s", s)
    if m:
        # canonical_mode() already forces "stable-queue-<x>s"
        return f"Stable-queue-{m.group(1)}s"

    # fallback: keep readable
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
        kv = parse_plugin_config(plugin_config)
        mode = canonical_mode(kv.get("mode", "unknown"))

        # scheduling-failure has no meaningful blocking label; normalize to 0
        if mode == "scheduling-failure":
            blocking = 0
        else:
            blocking = 1 if str(kv.get("blocking", "0")).lower() in {"1", "true", "yes"} else 0

        try:
            defpreempt = int(str(kv.get("defpreempt", "0")))
        except Exception:
            defpreempt = 0

        return RowKey(mode=mode, blocking=blocking, defpreempt=defpreempt)

    def _base_label(self) -> str:
        if self.mode == "scheduling-failure":
            return "Scheduling-failure"

        lbl = _MODE_BLOCK_LABEL.get((self.mode, self.blocking))
        if lbl is not None:
            return lbl

        # fallback for other modes
        base = pretty_mode(self.mode)
        blk = "blk" if self.blocking == 1 else "non-blk"
        return f"{base} ({blk})"

    def label(self, include_defpreempt: bool = True) -> str:
        base = self._base_label()

        # User requested exact label for scheduling-failure (no suffix)
        if self.mode == "scheduling-failure":
            return base

        # New convention:
        #   defpreempt=1  => no suffix
        #   defpreempt=0  => ", w/o defpreempt"
        if include_defpreempt and self.defpreempt == 0:
            if base.endswith(")"):
                return base[:-1] + ", w/o defpreempt)"
            return base + " (w/o defpreempt)"
        return base

    def sort_key(self) -> Tuple[int, int, str]:
        # Primary: explicit base order
        base_rank = _MODE_BLOCK_RANK.get((self.mode, self.blocking))
        if base_rank is None:
            base_rank = _MODE_BLOCK_RANK.get((self.mode, None))

        # Secondary: defpreempt=1 before defpreempt=0
        def_rank = 0 if self.defpreempt == 1 else 1

        # Fallback modes go after the known ones, but stay deterministic
        if base_rank is None:
            # reuse your old grouping idea, but push after the fixed list
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
            # compress into an int-ish rank
            base_rank = 1000 + group * 100 + int(round(num * 10))

        return (base_rank, def_rank, self.mode)



# -----------------------------
# Formatting
# -----------------------------

def is_finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _normalize_neg_zero(v: float) -> float:
    return 0.0 if abs(v) < 0.5e-12 else v


def fmt_signed(x: object, decimals: int, nan_str: str) -> str:
    if not is_finite(x):
        return nan_str
    v = _normalize_neg_zero(float(x))
    return f"{v:+.{decimals}f}"


def fmt_count(x: object, nan_str: str) -> str:
    if not is_finite(x):
        return nan_str
    return f"{int(round(float(x))):+d}"


def fmt_vec(values: Sequence[object], decimals: int, nan_str: str, latex: bool) -> str:
    if any(not is_finite(v) for v in values):
        return nan_str
    parts = [fmt_signed(v, decimals=decimals, nan_str=nan_str) for v in values]
    inside = ",".join(parts)
    if latex:
        return r"{\scriptsize$\langle" + inside + r"\rangle$}"
    return "<" + inside + ">"


# -----------------------------
# Data access
# -----------------------------

EXPECTED_COLS = [
    "job_name",
    "plugin_config",
    "delta_util_eff_run_mean",
    "delta_R_p1_mean", "delta_R_p2_mean", "delta_R_p3_mean", "delta_R_p4_mean",
    "delta_R_total_mean",
    "delta_D_p1_mean", "delta_D_p2_mean", "delta_D_p3_mean", "delta_D_p4_mean",
    "delta_D_total_mean",
    "delta_latency_s_p1_mean", "delta_latency_s_p2_mean", "delta_latency_s_p3_mean", "delta_latency_s_p4_mean",
    "delta_latency_s_total_mean",
    "solver_attempts_mean",
    "plan_activated_mean",
]


def load_results(in_dir: Path) -> pd.DataFrame:
    path = in_dir / "results_paired.csv"
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns: {', '.join(missing)}")

    parsed = df["job_name"].map(parse_job)
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), "job_name"].head(5).tolist()
        raise SystemExit(f"Could not parse some job_name values (examples): {bad}")

    df["nodes"] = parsed.map(lambda t: t[0] if t is not None else np.nan)
    df["kmax"] = parsed.map(lambda t: t[1] if t is not None else np.nan)
    df["arrival_s"] = parsed.map(lambda t: t[2] if t is not None else np.nan)

    rks = df["plugin_config"].map(RowKey.from_plugin_config)
    df["mode"] = rks.map(lambda r: r.mode)
    df["blocking"] = rks.map(lambda r: r.blocking)
    df["defpreempt"] = rks.map(lambda r: r.defpreempt)

    return df


def subset_value(
    df: pd.DataFrame,
    *,
    nodes: int,
    kmax: int,
    arrival_s: float,
    mode: str,
    blocking: int,
    defpreempt: int,
    col: str,
) -> float:
    g = df[
        (df["nodes"].astype(int) == int(nodes)) &
        (df["kmax"].astype(int) == int(kmax)) &
        (df["arrival_s"].astype(float) == float(arrival_s)) &
        (df["mode"].astype(str) == str(mode)) &
        (df["blocking"].astype(int) == int(blocking)) &
        (df["defpreempt"].astype(int) == int(defpreempt))
    ]
    if g.empty:
        return float("nan")
    return float(g.iloc[0][col])


# -----------------------------
# Table writers
# -----------------------------

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
    assert len(nodes_order) == 2 and len(arrivals_order) == 3, "Assumes 2 nodes x 3 arrivals."
    row_label_fn = row_label_fn or (lambda rk: rk.label(include_defpreempt=True))

    def mu(a: float) -> str:
        a_i = int(a) if abs(a - round(a)) < 1e-9 else a
        return f"$\\mu_A{{=}}{a_i}\\,\\mathrm{{s}}$"

    lines: List[str] = []
    lines += [
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        rf"\multicolumn{{7}}{{l}}{{$\mathbf{{{title_math}}}$ {subtitle}}} \\",
        r"\addlinespace[0.2em]",
        rf"& \multicolumn{{3}}{{c}}{{$N={nodes_order[0]}$}} & \multicolumn{{3}}{{c}}{{$N={nodes_order[1]}$}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        rf"& {mu(arrivals_order[0])} & {mu(arrivals_order[1])} & {mu(arrivals_order[2])}"
        rf" & {mu(arrivals_order[0])} & {mu(arrivals_order[1])} & {mu(arrivals_order[2])} \\",
        r"\midrule",
    ]

    for kmax in sorted(rows_by_kmax.keys()):
        k_title = r"$k_{\max}=1$ (no priorities)" if kmax == 1 else rf"$k_{{\max}}={kmax}$ (priorities enabled)"
        lines += [
            rf"\multicolumn{{7}}{{l}}{{{k_title}}} \\",
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
    row_label_fn = row_label_fn or (lambda rk: rk.label(include_defpreempt=True))

    def a_label(a: float) -> str:
        a_i = int(a) if abs(a - round(a)) < 1e-9 else a
        return f"µ_A={a_i}s"

    # -------- auto width pass --------
    # row_w: treat provided row_w as a MINIMUM (never truncate)
    max_label = 0
    for rks in rows_by_kmax.values():
        for rk in rks:
            max_label = max(max_label, len(str(row_label_fn(rk))))
    required_row_w = max(24, max_label + 2)
    row_w = required_row_w if row_w is None else max(int(row_w), required_row_w)

    # col_w: treat provided col_w as a MINIMUM (never truncate)
    max_cell = 0
    for kmax, rks in rows_by_kmax.items():
        for rk in rks:
            for n in nodes_order:
                for a in arrivals_order:
                    max_cell = max(max_cell, len(str(cell_fn(kmax, rk, n, a))))
    max_hdr = max(len(a_label(a)) for a in arrivals_order)
    required_col_w = max(10, max(max_cell, max_hdr) + 2)
    col_w = required_col_w if col_w is None else max(int(col_w), required_col_w)


    def center(s: str, w: int) -> str:
        s = str(s)
        if len(s) >= w:
            return s  # never truncate
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
            hdr2 += center(a_label(a), col_w)
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



def write_both(
    *,
    stem: str,
    out_dir: Path,
    latex_args: dict,
    ascii_args: dict,
) -> None:
    latex_table(out_path=out_dir / f"{stem}.tex", **latex_args)
    ascii_table(out_path=out_dir / f"{stem}.txt", **ascii_args)


# -----------------------------
# Special table: defpreempt delta deletions (within plugin, not vs baseline)
# -----------------------------

def build_defpreempt_rows(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """Return list of (mode, blocking) pairs that have BOTH defpreempt=0 and defpreempt=1."""
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


def cell_defpreempt_delta_deletions(
    df: pd.DataFrame,
    *,
    kmax: int,
    mode: str,
    blocking: int,
    nodes: int,
    arrival_s: float,
    float_decimals: int,
    latex: bool,
) -> str:
    # (defpreempt=1) - (defpreempt=0), baseline cancels
    nan_str = (r"\text{--}" if latex else "--")

    if kmax == 1:
        d0 = subset_value(df, nodes=nodes, kmax=kmax, arrival_s=arrival_s, mode=mode, blocking=blocking, defpreempt=0, col="delta_D_total_mean")
        d1 = subset_value(df, nodes=nodes, kmax=kmax, arrival_s=arrival_s, mode=mode, blocking=blocking, defpreempt=1, col="delta_D_total_mean")
        dd = (d1 - d0) if (is_finite(d0) and is_finite(d1)) else float("nan")
        return fmt_signed(dd, float_decimals, nan_str)

    vals0 = [
        subset_value(df, nodes=nodes, kmax=kmax, arrival_s=arrival_s, mode=mode, blocking=blocking, defpreempt=0, col=f"delta_D_p{p}_mean")
        for p in range(1, 5)
    ]
    vals1 = [
        subset_value(df, nodes=nodes, kmax=kmax, arrival_s=arrival_s, mode=mode, blocking=blocking, defpreempt=1, col=f"delta_D_p{p}_mean")
        for p in range(1, 5)
    ]
    if any(not is_finite(v) for v in vals0) or any(not is_finite(v) for v in vals1):
        return nan_str
    diff = [float(v1) - float(v0) for v0, v1 in zip(vals0, vals1)]
    return fmt_vec(diff, float_decimals, nan_str, latex=latex)


def defpreempt_delta_deletions_total(
    df: pd.DataFrame,
    *,
    kmax: int,
    mode: str,
    blocking: int,
    nodes: int,
    arrival_s: float,
) -> float:
    """(defpreempt=1) - (defpreempt=0) for TOTAL deletions; baseline cancels."""
    d0 = subset_value(
        df,
        nodes=nodes,
        kmax=kmax,
        arrival_s=arrival_s,
        mode=mode,
        blocking=blocking,
        defpreempt=0,
        col="delta_D_total_mean",
    )
    d1 = subset_value(
        df,
        nodes=nodes,
        kmax=kmax,
        arrival_s=arrival_s,
        mode=mode,
        blocking=blocking,
        defpreempt=1,
        col="delta_D_total_mean",
    )
    return (d1 - d0) if (is_finite(d0) and is_finite(d1)) else float("nan")


def collect_defpreempt_delta_deletions_total_ys(
    *,
    df: pd.DataFrame,
    nodes_order: List[int],
    arrivals_order: List[float],
    kmax: int,
) -> List[float]:
    """Collect y values for defpreempt deletions-delta plot (scalar total)."""
    ys: List[float] = []
    for mode, blocking in build_defpreempt_rows(df):
        for a in arrivals_order:
            y16 = defpreempt_delta_deletions_total(
                df,
                kmax=int(kmax),
                mode=str(mode),
                blocking=int(blocking),
                nodes=int(nodes_order[0]),
                arrival_s=float(a),
            )
            y32 = defpreempt_delta_deletions_total(
                df,
                kmax=int(kmax),
                mode=str(mode),
                blocking=int(blocking),
                nodes=int(nodes_order[1]),
                arrival_s=float(a),
            )
            if is_finite(y16):
                ys.append(float(y16))
            if is_finite(y32):
                ys.append(float(y32))
    return ys


def plot_defpreempt_delta_deletions_total(
    *,
    df: pd.DataFrame,
    out_dir: Path,
    filename_stem: str,
    title: str,
    y_label: str,
    nodes_order: List[int],
    arrivals_order: List[float],
    kmax: int,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot (defpreempt=1) - (defpreempt=0) for TOTAL deletions; baseline cancels."""
    out_dir.mkdir(parents=True, exist_ok=True)

    series = [RowKey(mode=str(m), blocking=int(b), defpreempt=1) for (m, b) in build_defpreempt_rows(df)]
    series.sort(key=lambda rk: rk.sort_key())

    x_base = [i * ARRIVAL_X_SPACING for i in range(len(arrivals_order))]
    m = max(1, len(series))
    mode_spacing = MODE_X_SPACING

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ys_for_limits: List[float] = []
    legend_handles: List[object] = []
    legend_labels: List[str] = []

    for i, rk in enumerate(series):
        mode_offset = (i - (m - 1) / 2.0) * mode_spacing

        color = ax.plot([], [], linestyle="None")[0].get_color()
        (h_legend,) = ax.plot([], [], linestyle="-", linewidth=1.8, color=color, label=rk.label(include_defpreempt=False))
        legend_handles.append(h_legend)
        legend_labels.append(rk.label(include_defpreempt=False))

        for xi, a in enumerate(arrivals_order):
            x = x_base[xi] + mode_offset

            y16 = defpreempt_delta_deletions_total(
                df,
                kmax=int(kmax),
                mode=rk.mode,
                blocking=int(rk.blocking),
                nodes=int(nodes_order[0]),
                arrival_s=float(a),
            )
            y32 = defpreempt_delta_deletions_total(
                df,
                kmax=int(kmax),
                mode=rk.mode,
                blocking=int(rk.blocking),
                nodes=int(nodes_order[1]),
                arrival_s=float(a),
            )

            ys_for_limits += [y16, y32]

            if is_finite(y16) and is_finite(y32):
                ax.plot([x, x], [y16, y32], linestyle="--", color=color, linewidth=1.0)

            if is_finite(y16):
                ax.plot(
                    [x],
                    [y16],
                    linestyle="None",
                    marker="o",
                    markersize=MARKER_SIZE,
                    color=color,
                    label="_nolegend_",
                )
            if is_finite(y32):
                ax.plot(
                    [x],
                    [y32],
                    linestyle="None",
                    marker="s",
                    markersize=MARKER_SIZE,
                    color=color,
                    label="_nolegend_",
                )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_base)
    ax.set_xticklabels([
        rf"$\mu_A={int(a) if abs(a-round(a)) < 1e-9 else a}\,\mathrm{{s}}$"
        for a in arrivals_order
    ])

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        _set_symmetric_ylim(ax, ys_for_limits)

    add_standard_legend(ax=ax, handles=legend_handles, labels=legend_labels)
    fig.tight_layout()

    fig.savefig(out_dir / f"{filename_stem}.png", dpi=200)
    fig.savefig(out_dir / f"{filename_stem}.pdf")
    plt.close(fig)


# -----------------------------
# Plots
# -----------------------------

def _set_symmetric_ylim(ax, ys: List[float]) -> None:
    vals = [abs(float(y)) for y in ys if is_finite(y)]
    if not vals:
        ax.set_ylim(-1.0, 1.0)
        return
    m = max(vals)
    if m <= 0:
        ax.set_ylim(-1.0, 1.0)
        return
    pad = 0.10 * m
    ax.set_ylim(-(m + pad), (m + pad))


def symmetric_ylim_from_ys(ys: List[float]) -> Tuple[float, float]:
    """Return a symmetric (low, high) ylim with 10% padding."""
    vals = [abs(float(y)) for y in ys if is_finite(y)]
    if not vals:
        return (-1.0, 1.0)
    m = max(vals)
    if m <= 0:
        return (-1.0, 1.0)
    pad = 0.10 * m
    return (-(m + pad), (m + pad))


def collect_plot_ys(
    *,
    df: pd.DataFrame,
    col: str,
    nodes_order: List[int],
    arrivals_order: List[float],
    kmax: int,
    scale: float = 1.0,
    include_defpreempt: bool = False,
) -> List[float]:
    """Collect the y values that would be plotted (used for consistent scaling across kmax)."""
    dff = df[df["kmax"].astype(int) == int(kmax)].copy()

    if not include_defpreempt:
        dff_main = dff[dff["defpreempt"].astype(int) == 1].copy()
        sf = dff[dff["mode"].astype(str) == "scheduling-failure"].copy()
        if not sf.empty:
            dff_main = pd.concat([dff_main, sf], ignore_index=True)
            dff_main = dff_main.drop_duplicates(
                subset=["job_name", "plugin_config", "mode", "blocking", "defpreempt", "nodes", "kmax", "arrival_s"],
                keep="first",
            )
        dff = dff_main

    series = sorted(
        {
            RowKey(mode=str(r["mode"]), blocking=int(r["blocking"]), defpreempt=int(r["defpreempt"]))
            for _, r in dff.iterrows()
        },
        key=lambda rk: rk.sort_key(),
    )

    ys: List[float] = []
    for rk in series:
        for a in arrivals_order:
            y16 = subset_value(
                df,
                nodes=nodes_order[0],
                kmax=kmax,
                arrival_s=a,
                mode=rk.mode,
                blocking=rk.blocking,
                defpreempt=rk.defpreempt,
                col=col,
            )
            y32 = subset_value(
                df,
                nodes=nodes_order[1],
                kmax=kmax,
                arrival_s=a,
                mode=rk.mode,
                blocking=rk.blocking,
                defpreempt=rk.defpreempt,
                col=col,
            )
            if is_finite(y16):
                ys.append(float(y16) * float(scale))
            if is_finite(y32):
                ys.append(float(y32) * float(scale))

    return ys


def plot_two_node_dots(
    *,
    df: pd.DataFrame,
    out_dir: Path,
    filename_stem: str,
    title: str,
    y_label: str,
    col: str,
    nodes_order: List[int],
    arrivals_order: List[float],
    kmax: int,
    float_decimals: int,
    scale: float = 1.0,
    include_defpreempt: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    dff = df[df["kmax"].astype(int) == int(kmax)].copy()

    if not include_defpreempt:
        # main config is defpreempt=1
        dff_main = dff[dff["defpreempt"].astype(int) == 1].copy()

        # BUT: keep scheduling-failure even if it only exists for defpreempt=0
        sf = dff[dff["mode"].astype(str) == "scheduling-failure"].copy()
        if not sf.empty:
            dff_main = pd.concat([dff_main, sf], ignore_index=True)

            # de-dup on the identifying cols we care about
            dff_main = dff_main.drop_duplicates(
                subset=["job_name", "plugin_config", "mode", "blocking", "defpreempt", "nodes", "kmax", "arrival_s"],
                keep="first",
            )

        dff = dff_main


    # series are full RowKey (so include_defpreempt=True doubles dots by adding defpreempt=0 rows too)
    series = sorted(
        {
            RowKey(mode=str(r["mode"]), blocking=int(r["blocking"]), defpreempt=int(r["defpreempt"]))
            for _, r in dff.iterrows()
        },
        key=lambda rk: rk.sort_key(),
    )

    x_base = [i * ARRIVAL_X_SPACING for i in range(len(arrivals_order))]

    m = max(1, len(series))
    mode_spacing = MODE_X_SPACING

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ys_for_limits: List[float] = []
    legend_handles: List[object] = []
    legend_labels: List[str] = []

    for i, rk in enumerate(series):
        mode_offset = (i - (m - 1) / 2.0) * mode_spacing

        # consume a fresh cycle color for this series
        color = ax.plot([], [], linestyle="None")[0].get_color()

        # legend: show a LINE (not dots)
        (h_legend,) = ax.plot([], [], linestyle="-", linewidth=1.8, color=color, label=rk.label(include_defpreempt=True))
        legend_handles.append(h_legend)
        legend_labels.append(rk.label(include_defpreempt=True))

        for xi, a in enumerate(arrivals_order):
            x = x_base[xi] + mode_offset

            y16 = subset_value(
                df, nodes=nodes_order[0], kmax=kmax, arrival_s=a,
                mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt, col=col
            )
            y32 = subset_value(
                df, nodes=nodes_order[1], kmax=kmax, arrival_s=a,
                mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt, col=col
            )

            y16 = y16 * scale if is_finite(y16) else float("nan")
            y32 = y32 * scale if is_finite(y32) else float("nan")
            ys_for_limits += [y16, y32]

            if is_finite(y16) and is_finite(y32):
                ax.plot([x, x], [y16, y32], linestyle="--", color=color, linewidth=1.0)

            if is_finite(y16):
                ax.plot(
                    [x],
                    [y16],
                    linestyle="None",
                    marker="o",
                    markersize=MARKER_SIZE,
                    color=color,
                    label="_nolegend_",
                )
            if is_finite(y32):
                ax.plot(
                    [x],
                    [y32],
                    linestyle="None",
                    marker="s",
                    markersize=MARKER_SIZE,
                    color=color,
                    label="_nolegend_",
                )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_base)
    ax.set_xticklabels([
        rf"$\mu_A={int(a) if abs(a-round(a)) < 1e-9 else a}\,\mathrm{{s}}$"
        for a in arrivals_order
    ])

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        _set_symmetric_ylim(ax, ys_for_limits)
    add_standard_legend(ax=ax, handles=legend_handles, labels=legend_labels)
    fig.tight_layout()

    fig.savefig(out_dir / f"{filename_stem}.png", dpi=200)
    fig.savefig(out_dir / f"{filename_stem}.pdf")
    plt.close(fig)



# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.plots_dir) if args.plots_dir else (out_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    nodes_order = [int(x.strip()) for x in str(args.nodes).split(",") if x.strip()]
    arrivals_order = [float(x.strip()) for x in str(args.arrivals).split(",") if x.strip()]
    if len(nodes_order) != 2 or len(arrivals_order) != 3:
        raise SystemExit("This script assumes exactly 2 nodes values and 3 arrival values.")

    float_decimals = int(args.float_decimals)
    plot_kmax_arg = str(args.plot_kmax).strip().lower()

    df = load_results(in_dir)

    available_kmax = sorted(df["kmax"].dropna().astype(int).unique().tolist())
    if not available_kmax:
        raise SystemExit("No kmax values found in results_paired.csv")

    if plot_kmax_arg == "auto":
        plot_kmaxs = available_kmax
    else:
        plot_kmaxs: List[int] = []
        for tok in plot_kmax_arg.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                plot_kmaxs.append(int(tok))
            except Exception:
                raise SystemExit(f"Invalid --plot-kmax value: {args.plot_kmax}")
        plot_kmaxs = sorted(set(plot_kmaxs))

    missing = [k for k in plot_kmaxs if k not in available_kmax]
    if missing:
        raise SystemExit(f"Requested kmax not in results: {missing}. Available: {available_kmax}")

    # Shared y-axis scaling across kmax values (per metric).
    util_ys_all: List[float] = []
    lat_ys_all: List[float] = []
    for k in plot_kmaxs:
        util_ys_all += collect_plot_ys(
            df=df,
            col="delta_util_eff_run_mean",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=int(k),
            scale=100.0,
            include_defpreempt=bool(args.plot_include_defpreempt),
        )
        lat_ys_all += collect_plot_ys(
            df=df,
            col="delta_latency_s_total_mean",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=int(k),
            scale=1.0,
            include_defpreempt=bool(args.plot_include_defpreempt),
        )

    util_ylim = symmetric_ylim_from_ys(util_ys_all)
    lat_ylim = symmetric_ylim_from_ys(lat_ys_all)

    # Build RowKey lists per kmax from actual plugin_config rows
    rows_by_kmax: Dict[int, List[RowKey]] = {}
    for kmax in sorted(df["kmax"].dropna().astype(int).unique().tolist()):
        pcs = sorted(df.loc[df["kmax"].astype(int) == kmax, "plugin_config"].unique().tolist())
        rks = [RowKey.from_plugin_config(pc) for pc in pcs]
        rks.sort(key=lambda rk: rk.sort_key())
        rows_by_kmax[kmax] = rks

    # Filtered rows for "main" tables where we only want defpreempt=1
    rows_by_kmax_defpreempt_only: Dict[int, List[RowKey]] = {
        k: [rk for rk in rks if rk.defpreempt == 1]
        for k, rks in rows_by_kmax.items()
    }

    # ----------------
    # TABLES
    # ----------------

    def cell_ueff(kmax: int, rk: RowKey, n: int, a: float, latex: bool) -> str:
        nan_str = (r"\text{--}" if latex else "--")
        v = subset_value(df, nodes=n, kmax=kmax, arrival_s=a, mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                         col="delta_util_eff_run_mean")
        v = v * 100.0 if is_finite(v) else float("nan")
        return fmt_signed(v, float_decimals, nan_str)

    write_both(
        stem="delta_u_eff_run",
        out_dir=out_dir,
        latex_args=dict(
            title_math=r"\Delta u_{\mathrm{eff}}",
            subtitle=r"(pp), mean paired difference vs.\ baseline",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_ueff(k, rk, n, a, True),
        ),
        ascii_args=dict(
            title="Delta u_eff (pp), mean paired difference vs baseline",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_ueff(k, rk, n, a, False),
        ),
    )

    def cell_delta_R(kmax: int, rk: RowKey, n: int, a: float, latex: bool) -> str:
        nan_str = (r"\text{--}" if latex else "--")
        if kmax == 1:
            v = subset_value(
                df, nodes=n, kmax=kmax, arrival_s=a,
                mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                col="delta_R_total_mean",
            )
            return fmt_signed(v, float_decimals, nan_str)
        vals = [
            subset_value(
                df, nodes=n, kmax=kmax, arrival_s=a,
                mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                col=f"delta_R_p{p}_mean",
            )
            for p in range(1, 5)
        ]
        return fmt_vec(vals, float_decimals, nan_str, latex=latex)

    write_both(
        stem="delta_R",
        out_dir=out_dir,
        latex_args=dict(
            title_math=r"\Delta R",
            subtitle=r"(mean running pods), mean paired difference vs.\ baseline",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_delta_R(k, rk, n, a, True),
        ),
        ascii_args=dict(
            title="Delta R (mean running pods), mean paired difference vs baseline",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_delta_R(k, rk, n, a, False),
            col_w=28,
        ),
    )


    def cell_delta_D(kmax: int, rk: RowKey, n: int, a: float, latex: bool) -> str:
        nan_str = (r"\text{--}" if latex else "--")
        if kmax == 1:
            v = subset_value(
                df, nodes=n, kmax=kmax, arrival_s=a,
                mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                col="delta_D_total_mean",
            )
            return fmt_signed(v, float_decimals, nan_str)
        vals = [
            subset_value(
                df, nodes=n, kmax=kmax, arrival_s=a,
                mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                col=f"delta_D_p{p}_mean",
            )
            for p in range(1, 5)
        ]
        return fmt_vec(vals, float_decimals, nan_str, latex=latex)

    write_both(
        stem="delta_D",
        out_dir=out_dir,
        latex_args=dict(
            title_math=r"\Delta D(T)",
            subtitle=r"(deletions), mean paired difference vs.\ baseline",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_delta_D(k, rk, n, a, True),
        ),
        ascii_args=dict(
            title="Delta D(T) (deletions), mean paired difference vs baseline",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_delta_D(k, rk, n, a, False),
            col_w=28,
        ),
    )


    def cell_delta_latency(kmax: int, rk: RowKey, n: int, a: float, latex: bool) -> str:
        nan_str = (r"\text{--}" if latex else "--")
        if kmax == 1:
            v = subset_value(
                df, nodes=n, kmax=kmax, arrival_s=a,
                mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                col="delta_latency_s_total_mean",
            )
            return fmt_signed(v, float_decimals, nan_str)
        vals = [
            subset_value(
                df, nodes=n, kmax=kmax, arrival_s=a,
                mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                col=f"delta_latency_s_p{p}_mean",
            )
            for p in range(1, 5)
        ]
        return fmt_vec(vals, float_decimals, nan_str, latex=latex)

    write_both(
        stem="delta_latency",
        out_dir=out_dir,
        latex_args=dict(
            title_math=r"\Delta \mathrm{latency}",
            subtitle=r"(s), mean paired difference vs.\ baseline",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_delta_latency(k, rk, n, a, True),
        ),
        ascii_args=dict(
            title="Delta latency (s), mean paired difference vs baseline",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_delta_latency(k, rk, n, a, False),
            col_w=28,
        ),
    )


    def cell_solver_attempts(kmax: int, rk: RowKey, n: int, a: float, latex: bool) -> str:
        nan_str = (r"\text{--}" if latex else "--")
        v = subset_value(
            df, nodes=n, kmax=kmax, arrival_s=a,
            mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
            col="solver_attempts_mean",
        )
        return fmt_count(v, nan_str)

    write_both(
        stem="solver_attempts",
        out_dir=out_dir,
        latex_args=dict(
            title_math=r"\mathrm{solver\_attempts}",
            subtitle=r"(count), mean across seeds",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_solver_attempts(k, rk, n, a, True),
        ),
        ascii_args=dict(
            title="solver_attempts (count), mean across seeds",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_solver_attempts(k, rk, n, a, False),
        ),
    )

    def cell_plans_activated(kmax: int, rk: RowKey, n: int, a: float, latex: bool) -> str:
        nan_str = (r"\text{--}" if latex else "--")
        v = subset_value(
            df, nodes=n, kmax=kmax, arrival_s=a,
            mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
            col="plan_activated_mean",
        )
        return fmt_count(v, nan_str)

    write_both(
        stem="plans_activated",
        out_dir=out_dir,
        latex_args=dict(
            title_math=r"\mathrm{plans\_activated}",
            subtitle=r"(count), mean across seeds",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_plans_activated(k, rk, n, a, True),
        ),
        ascii_args=dict(
            title="plans_activated (count), mean across seeds",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax,
            cell_fn=lambda k, rk, n, a: cell_plans_activated(k, rk, n, a, False),
        ),
    )



    # Big combined table: latency + util
    # IMPORTANT: show ONLY defpreempt=1 rows here
    def cell_big_latency_util(kmax: int, rk: RowKey, n: int, a: float, latex: bool) -> str:
        nan_str = (r"\text{--}" if latex else "--")

        u = subset_value(df, nodes=n, kmax=kmax, arrival_s=a, mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                         col="delta_util_eff_run_mean")
        u = u * 100.0 if is_finite(u) else float("nan")
        u_s = fmt_signed(u, float_decimals, nan_str)

        if kmax == 1:
            lat = subset_value(df, nodes=n, kmax=kmax, arrival_s=a, mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                               col="delta_latency_s_total_mean")
            lat_s = fmt_signed(lat, float_decimals, nan_str)
        else:
            vals = [
                subset_value(df, nodes=n, kmax=kmax, arrival_s=a, mode=rk.mode, blocking=rk.blocking, defpreempt=rk.defpreempt,
                             col=f"delta_latency_s_p{p}_mean")
                for p in range(1, 5)
            ]
            lat_s = fmt_vec(vals, float_decimals, nan_str, latex=latex)

        return f"{lat_s}; {u_s}"

    write_both(
        stem="big_table_latency_util",
        out_dir=out_dir,
        latex_args=dict(
            title_math=r"\Delta \mathrm{latency}_{p}\ ;\ \Delta u_{\mathrm{eff}}",
            subtitle=r"(s; pp), paired difference vs.\ baseline",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax_defpreempt_only,  # <-- filter here
            cell_fn=lambda k, rk, n, a: cell_big_latency_util(k, rk, n, a, True),
        ),
        ascii_args=dict(
            title="Combined (defpreempt enabled): Delta latency_p (or total if kmax=1) ; Delta u_eff (pp)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax_defpreempt_only,  # <-- filter here
            cell_fn=lambda k, rk, n, a: cell_big_latency_util(k, rk, n, a, False),
            col_w=30,
        ),
    )

    # defpreempt table: (defpreempt=1) - (defpreempt=0) for deletions
    def_rows = build_defpreempt_rows(df)
    rows_by_kmax_def: Dict[int, List[RowKey]] = {}
    for kmax in sorted(rows_by_kmax.keys()):
        tmp = [RowKey(mode=m, blocking=b, defpreempt=1) for (m, b) in def_rows]  # defpreempt value irrelevant for label
        tmp.sort(key=lambda rk: rk.sort_key())
        rows_by_kmax_def[kmax] = tmp

    def cell_def(kmax: int, rk: RowKey, n: int, a: float, latex: bool) -> str:
        return cell_defpreempt_delta_deletions(
            df,
            kmax=kmax,
            mode=rk.mode,
            blocking=rk.blocking,
            nodes=n,
            arrival_s=a,
            float_decimals=float_decimals,
            latex=latex,
        )

    write_both(
        stem="delta_deletions_with_defaultpreemption",
        out_dir=out_dir,
        latex_args=dict(
            title_math=r"\Delta D_{\mathrm{defpreempt}}",
            subtitle=r"(deletions), $(\mathrm{defpreempt}=1)-(\mathrm{defpreempt}=0)$",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax_def,
            cell_fn=lambda k, rk, n, a: cell_def(k, rk, n, a, True),
            row_label_fn=lambda rk: rk.label(include_defpreempt=False),  # <-- no suffix here
        ),
        ascii_args=dict(
            title="Delta deletions: (defpreempt=1) - (defpreempt=0)  [within plugin, baseline cancels]",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            rows_by_kmax=rows_by_kmax_def,
            cell_fn=lambda k, rk, n, a: cell_def(k, rk, n, a, False),
            col_w=28,
            row_label_fn=lambda rk: rk.label(include_defpreempt=False),  # <-- no suffix here
        ),
    )

    # ----------------
    # PLOTS
    # ----------------

    for plot_kmax in plot_kmaxs:
        plot_two_node_dots(
            df=df,
            out_dir=plots_dir,
            filename_stem=f"delta_u_eff_run_kmax{plot_kmax}",
            title=rf"$\Delta u_{{\mathrm{{eff}}}}$ (pp) vs $\mu_A$  (kmax={plot_kmax})",
            y_label=r"$\Delta u_{\mathrm{eff}}$ (pp)",
            col="delta_util_eff_run_mean",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=plot_kmax,
            float_decimals=float_decimals,
            scale=100.0,
            include_defpreempt=bool(args.plot_include_defpreempt),
            ylim=util_ylim,
        )

        plot_two_node_dots(
            df=df,
            out_dir=plots_dir,
            filename_stem=f"delta_latency_total_kmax{plot_kmax}",
            title=rf"$\Delta \mathrm{{latency}}_\mathrm{{total}}$ (s) vs $\mu_A$  (kmax={plot_kmax})",
            y_label=r"$\Delta \mathrm{latency}$ (s)",
            col="delta_latency_s_total_mean",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=plot_kmax,
            float_decimals=float_decimals,
            scale=1.0,
            include_defpreempt=bool(args.plot_include_defpreempt),
            ylim=lat_ylim,
        )

        plot_defpreempt_delta_deletions_total(
            df=df,
            out_dir=plots_dir,
            filename_stem=f"delta_deletions_with_defaultpreemption_kmax{plot_kmax}",
            title=rf"$\Delta D_{{\mathrm{{defpreempt}}}}$ (count) vs $\mu_A$  (kmax={plot_kmax})",
            y_label=r"$\Delta D_{\mathrm{defpreempt}}$ (deletions)",
            nodes_order=nodes_order,
            arrivals_order=arrivals_order,
            kmax=int(plot_kmax),
            ylim=symmetric_ylim_from_ys(
                collect_defpreempt_delta_deletions_total_ys(
                    df=df,
                    nodes_order=nodes_order,
                    arrivals_order=arrivals_order,
                    kmax=int(plot_kmax),
                )
            ),
        )

    print(f"Wrote tables to: {out_dir}")
    print(f"Wrote plots to:  {plots_dir}")


if __name__ == "__main__":
    main()
