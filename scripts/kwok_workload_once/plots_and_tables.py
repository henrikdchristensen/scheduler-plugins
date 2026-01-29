#!/usr/bin/env python3
# plots_and_tables.py
"""
python -m scripts.kwok_workload_once.plots_and_tables

Tables:
- Per-combo (old table_per_combo_prio** style):
  - table_per_combo_prio{K}.{csv,tex}
  - table_per_combo_prio{K}_with_std.{csv,tex}   (if per_combo_results_with_std.csv exists)

Restored "old saved files" layout (the two matrices you pasted):
- table_solver_duration_and_utils_prio{K}_timeout{T}{_cfg...}.{csv,tex}
- table_solver_duration_and_utils_prio{K}_timeout{T}{_cfg...}_with_std.{csv,tex} (optional)

- table_solver_better_vs_kwok_optimal_prio{K}_timeout{T}{_cfg...}.{csv,tex}
- table_solver_better_vs_kwok_optimal_prio{K}_timeout{T}{_cfg...}_with_std.{csv,tex} (optional)

The restored layout is:
  util | metric | (ppn=4, nodes=4/8/16/32) | (ppn=8, nodes=4/8/16/32)
with booktabs + multirow + cmidrule exactly like your snippet.

Std input:
- analysis/kwok_workload_once/per_combo_results_with_std.csv (optional)

Notes / assumptions:
- solver_duration_ms_mean is converted to seconds in the restored matrix tables.
- cpu_delta_mean and mem_delta_mean are assumed already in percentage points (as in your examples).
- If multiple config_dir values exist, tables are generated per (timeout, config_dir) and the config is encoded in filename.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Iterable

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

from scripts.config.plot_config import (
    PLOT_FIGURE_DPI,
    PLOT_TITLE_FONTSIZE,
    PLOT_AXIS_LABEL_FONTSIZE,
    PLOT_TICK_FONTSIZE,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_HANDLE_LENGTH,
    PLOT_LEGEND_COLUMN_SPACING,
    PLOT_LEGEND_HANDLE_TEXT_PAD,
)

#################################################################
# Load data
#################################################################
DF_PER_COMBO_PATH = Path("analysis/kwok_workload_once/per_combo_results.csv")
df_per_combo = pd.read_csv(DF_PER_COMBO_PATH)

OUT_DIR = Path("analysis/kwok_workload_once")

OUT_FIGURES_DIR = OUT_DIR / "figures"
OUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLES_DIR = OUT_DIR / "tables"
OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

DF_PER_COMBO_STD_PATH = Path("analysis/kwok_workload_once/per_combo_results_with_std.csv")
df_per_combo_std: Optional[pd.DataFrame] = None
if DF_PER_COMBO_STD_PATH.exists():
    df_per_combo_std = pd.read_csv(DF_PER_COMBO_STD_PATH)
else:
    print(f"[info] std file not found: {DF_PER_COMBO_STD_PATH} (only non-std tables will be written)")

#################################################################
# Table helpers (per-combo tables)
#################################################################
def pm_col(df: pd.DataFrame, mean_col: str, std_col: str, decimals: int) -> pd.Series:
    m = pd.to_numeric(df.get(mean_col), errors="coerce")
    s = pd.to_numeric(df.get(std_col), errors="coerce").abs()
    m_str = m.map(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "--")
    s_str = s.map(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "--")
    return m_str + " ± " + s_str


def _latex_escape(s: object) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("$", r"\$")
        .replace("^", r"\^{}")
        .replace("~", r"\~{}")
    )


def _makecell(lines: list[str]) -> str:
    return r"\makecell[l]{" + r"\\ ".join(lines) + r"}"


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _rate_to_pp(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * 100.0
    return out


def _write_latex_table(
    *,
    df: pd.DataFrame,
    headers: list[tuple[str, str]],
    title: str,
    out_path: Path,
    column_spec: str,
) -> None:
    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + column_spec + r"}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{" + title + r"} \\")
    lines.append(r"\addlinespace[0.25em]")
    lines.append(" & ".join([h for (_c, h) in headers]) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        cells: list[str] = []
        for c, _h in headers:
            cells.append(_latex_escape(row.get(c, "")))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote table (LaTeX): {out_path}")


def _table_layout_headers() -> list[tuple[str, str]]:
    return [
        ("pods_per_node", _makecell(["ppn"])),
        ("timeout_s", _makecell(["timeout", "(s)"])),
        ("nodes", _makecell(["nodes"])),
        ("util", _makecell(["util", r"(\%)"])),
        ("config_dir", _makecell(["config"])),
        ("default_all_running_rate", _makecell(["No Calls", r"(\%)"])),
        ("default_optimal_rate", _makecell(["KWOK Opt", r"(\%)"])),
        ("solver_optimal_rate", _makecell(["Better&Opt", r"(\%)"])),
        ("solver_feasible_rate", _makecell(["Better", r"(\%)"])),
        ("solver_failed_rate", _makecell(["Failures", r"(\%)"])),
        ("other_rate", _makecell(["Other", r"(\%)"])),
        ("solver_called_rate", _makecell(["Solver called", r"(\%)"])),
        ("solver_duration_ms_mean", _makecell(["Solver", "(ms)"])),
        ("cpu_delta_mean", _makecell([r"$\Delta$CPU"])),
        ("mem_delta_mean", _makecell([r"$\Delta$Mem"])),
        ("n_seeds", _makecell([r"$n$"])),
    ]


def _table_column_spec() -> str:
    return "r r r r l " + ("c " * 10) + "r"


def build_per_combo_table_no_std(df: pd.DataFrame, prio: int) -> pd.DataFrame:
    sub = df[df["priorities"].astype(int) == int(prio)].copy()
    if sub.empty:
        return sub

    sub = _ensure_numeric(sub, ["pods_per_node", "timeout_s", "nodes", "util", "n_seeds"])

    rate_cols = [
        "default_all_running_rate",
        "default_optimal_rate",
        "solver_optimal_rate",
        "solver_feasible_rate",
        "solver_failed_rate",
        "other_rate",
        "solver_called_rate",
    ]
    sub = _rate_to_pp(sub, rate_cols)

    for c in rate_cols:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notna(x) else "--")

    if "solver_duration_ms_mean" in sub.columns:
        sub["solver_duration_ms_mean"] = pd.to_numeric(sub["solver_duration_ms_mean"], errors="coerce").map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "--"
        )
    for c in ["cpu_delta_mean", "mem_delta_mean"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce").map(lambda x: f"{x:.3f}" if pd.notna(x) else "--")

    if "n_seeds" in sub.columns:
        sub["n_seeds"] = pd.to_numeric(sub["n_seeds"], errors="coerce").map(lambda x: str(int(x)) if pd.notna(x) else "--")

    sub = sub.sort_values(["pods_per_node", "timeout_s", "nodes", "util", "config_dir"], kind="mergesort")
    return sub


def build_per_combo_table_with_std(df_std: pd.DataFrame, prio: int) -> pd.DataFrame:
    sub = df_std[df_std["priorities"].astype(int) == int(prio)].copy()
    if sub.empty:
        return sub

    sub = _ensure_numeric(sub, ["pods_per_node", "timeout_s", "nodes", "util", "n_seeds"])

    required_std_cols = [
        "default_all_running_rate_std",
        "solver_called_rate_std",
        "solver_failed_rate_std",
        "default_optimal_rate_std",
        "solver_optimal_rate_std",
        "solver_feasible_rate_std",
        "other_rate_std",
        "solver_duration_ms_mean_std",
        "cpu_delta_mean_std",
        "mem_delta_mean_std",
    ]
    missing = [c for c in required_std_cols if c not in sub.columns]
    if missing:
        raise SystemExit(
            "[error] per_combo_results_with_std.csv is missing required std columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + "\n=> Fix seal_results.py to write these columns."
        )

    rate_pairs = [
        ("default_all_running_rate", "default_all_running_rate_std", 1),
        ("default_optimal_rate", "default_optimal_rate_std", 1),
        ("solver_optimal_rate", "solver_optimal_rate_std", 1),
        ("solver_feasible_rate", "solver_feasible_rate_std", 1),
        ("solver_failed_rate", "solver_failed_rate_std", 1),
        ("other_rate", "other_rate_std", 1),
        ("solver_called_rate", "solver_called_rate_std", 1),
    ]
    for mean_col, std_col, dec in rate_pairs:
        sub[mean_col] = pd.to_numeric(sub[mean_col], errors="coerce") * 100.0
        sub[std_col] = pd.to_numeric(sub[std_col], errors="coerce") * 100.0
        sub[mean_col] = pm_col(sub, mean_col, std_col, dec)

    sub["solver_duration_ms_mean"] = pm_col(sub, "solver_duration_ms_mean", "solver_duration_ms_mean_std", 1)
    sub["cpu_delta_mean"] = pm_col(sub, "cpu_delta_mean", "cpu_delta_mean_std", 3)
    sub["mem_delta_mean"] = pm_col(sub, "mem_delta_mean", "mem_delta_mean_std", 3)

    if "n_seeds" in sub.columns:
        sub["n_seeds"] = pd.to_numeric(sub["n_seeds"], errors="coerce").map(lambda x: str(int(x)) if pd.notna(x) else "--")

    sub = sub.sort_values(["pods_per_node", "timeout_s", "nodes", "util", "config_dir"], kind="mergesort")
    return sub


def write_tables_per_priority(*, df_base: pd.DataFrame, df_std: Optional[pd.DataFrame], out_dir: Path) -> None:
    inferred = sorted(pd.to_numeric(df_base.get("priorities"), errors="coerce").dropna().astype(int).unique().tolist())
    if not inferred:
        print("[warn] no priorities found in per_combo_results.csv; skipping tables")
        return

    headers = _table_layout_headers()
    colspec = _table_column_spec()

    for prio in inferred:
        tbl = build_per_combo_table_no_std(df_base, prio)
        if not tbl.empty:
            cols = [c for c, _h in headers if c in tbl.columns]
            tbl_out = tbl[cols].copy()

            out_csv = out_dir / f"table_per_combo_prio{prio}.csv"
            tbl_out.to_csv(out_csv, index=False)
            print(f"[ok] wrote table (CSV): {out_csv}")

            out_tex = out_dir / f"table_per_combo_prio{prio}.tex"
            title = rf"\textbf{{Per-combination results, \#priorities={prio}}}"
            _write_latex_table(
                df=tbl_out,
                headers=[(c, h) for c, h in headers if c in tbl_out.columns],
                title=title,
                out_path=out_tex,
                column_spec=colspec,
            )
        else:
            print(f"[skip] no rows for priorities={prio} (no-std)")

        if df_std is None:
            continue

        tbls = build_per_combo_table_with_std(df_std, prio)
        if not tbls.empty:
            cols_s = [c for c, _h in headers if c in tbls.columns]
            tbls_out = tbls[cols_s].copy()

            out_csv_s = out_dir / f"table_per_combo_prio{prio}_with_std.csv"
            tbls_out.to_csv(out_csv_s, index=False)
            print(f"[ok] wrote table (std CSV): {out_csv_s}")

            out_tex_s = out_dir / f"table_per_combo_prio{prio}_with_std.tex"
            title_s = rf"\textbf{{Per-combination results (mean $\pm$ std), \#priorities={prio}}}"
            _write_latex_table(
                df=tbls_out,
                headers=[(c, h) for c, h in headers if c in tbls_out.columns],
                title=title_s,
                out_path=out_tex_s,
                column_spec=colspec,
            )
        else:
            print(f"[skip] no rows for priorities={prio} (std)")


#################################################################
# Restored "old saved files" matrix tables
#################################################################
NODES_ORDER = [4, 8, 16, 32]
PPN_ORDER = [4, 8]

def _slug(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return s[:60] if s else "cfg"

def _dash_cell() -> str:
    # matches your example: \multicolumn{1}{c}{—}
    return r"\multicolumn{1}{c}{—}"

def _fmt_num(x: object, dec: int) -> str:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return _dash_cell()
    return f"{float(v):.{dec}f}"

def _fmt_pm(m: object, s: object, dec: int) -> str:
    m_v = pd.to_numeric(pd.Series([m]), errors="coerce").iloc[0]
    s_v = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
    if pd.isna(m_v) and pd.isna(s_v):
        return _dash_cell()
    m_str = "--" if pd.isna(m_v) else f"{float(m_v):.{dec}f}"
    s_str = "--" if pd.isna(s_v) else f"{abs(float(s_v)):.{dec}f}"
    # keep compact, no LaTeX math here (matches your snippet style)
    return f"{m_str} ± {s_str}"

def _select_row(
    df: pd.DataFrame,
    *,
    prio: int,
    timeout: int,
    util: float,
    ppn: int,
    nodes: int,
    config_dir: str,
) -> pd.Series | None:
    sub = df[
        (df["priorities"].astype(int) == int(prio))
        & (df["timeout_s"].astype(int) == int(timeout))
        & (pd.to_numeric(df["util"], errors="coerce") == float(util))
        & (df["pods_per_node"].astype(int) == int(ppn))
        & (df["nodes"].astype(int) == int(nodes))
        & (df["config_dir"].astype(str) == str(config_dir))
    ]
    if sub.empty:
        return None
    return sub.iloc[0]

def _write_matrix_tex(
    *,
    out_path: Path,
    body_lines: list[str],
) -> None:
    out_path.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote table (LaTeX): {out_path}")


# --- add near the top of the "Restored matrix tables" section ---

def _aggregate_over_config_dir(
    df: pd.DataFrame,
    *,
    group_keys: list[str],
    mean_cols: list[str],
) -> pd.DataFrame:
    """
    Collapse multiple config_dir rows into a single row per group by averaging.

    group_keys must NOT include config_dir.
    Only columns in mean_cols that exist are aggregated.
    """
    df2 = df.copy()
    for c in group_keys + mean_cols:
        if c in df2.columns and c != "config_dir":
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    cols_present = [c for c in mean_cols if c in df2.columns]
    if not cols_present:
        return df2.drop(columns=["config_dir"], errors="ignore").drop_duplicates(subset=group_keys)

    g = (
        df2.groupby(group_keys, as_index=False)[cols_present]
        .mean(numeric_only=True)
    )
    return g


def _select_row_agg(
    df: pd.DataFrame,
    *,
    prio: int,
    timeout: int,
    util: float,
    ppn: int,
    nodes: int,
) -> pd.Series | None:
    sub = df[
        (df["priorities"].astype(int) == int(prio))
        & (df["timeout_s"].astype(int) == int(timeout))
        & (pd.to_numeric(df["util"], errors="coerce") == float(util))
        & (df["pods_per_node"].astype(int) == int(ppn))
        & (df["nodes"].astype(int) == int(nodes))
    ]
    if sub.empty:
        return None
    return sub.iloc[0]

def write_table_solver_duration_and_utils_matrix(
    *,
    df: pd.DataFrame,
    df_std: Optional[pd.DataFrame],
    out_dir: Path,
) -> None:
    df = df.copy()
    for c in ["util", "nodes", "pods_per_node", "timeout_s", "priorities"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- collapse over config_dir (if present) ---
    group_keys = ["priorities", "timeout_s", "util", "pods_per_node", "nodes"]
    mean_cols = ["solver_duration_ms_mean", "cpu_delta_mean", "mem_delta_mean"]
    if "config_dir" in df.columns:
        df = _aggregate_over_config_dir(df, group_keys=group_keys, mean_cols=mean_cols)

    prios = sorted(df["priorities"].dropna().astype(int).unique().tolist())
    timeouts = sorted(df["timeout_s"].dropna().astype(int).unique().tolist())
    utils = sorted(df["util"].dropna().unique().tolist())

    # std prep (optional)
    if df_std is not None:
        df_std = df_std.copy()
        for c in ["util", "nodes", "pods_per_node", "timeout_s", "priorities"]:
            if c in df_std.columns:
                df_std[c] = pd.to_numeric(df_std[c], errors="coerce")

        needed = ["solver_duration_ms_mean_std", "cpu_delta_mean_std", "mem_delta_mean_std"]
        missing = [c for c in needed if c not in df_std.columns]
        if missing:
            raise SystemExit(
                "[error] per_combo_results_with_std.csv missing std columns for solver_duration_and_utils matrix:\n"
                + "\n".join(f"  - {c}" for c in missing)
            )

        mean_cols_std = mean_cols + needed
        if "config_dir" in df_std.columns:
            df_std = _aggregate_over_config_dir(df_std, group_keys=group_keys, mean_cols=mean_cols_std)

    for prio in prios:
        for timeout in timeouts:
            # -------- no-std LaTeX --------
            lines: list[str] = []
            lines.append(r"\begin{tabular}{@{}c l *{8}{c}@{}}")
            lines.append(r"\toprule")
            lines.append(
                r"\multirow{2}{*}{\textbf{util}} & \multirow{2}{*}{\textbf{metric}} &"
                r"\multicolumn{4}{c}{\textbf{ppn = 4}} & \multicolumn{4}{c}{\textbf{ppn = 8}}\\"
            )
            lines.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
            lines.append(
                r" &  & \textbf{4} & \textbf{8} & \textbf{16} & \textbf{32} & "
                r"\textbf{4} & \textbf{8} & \textbf{16} & \textbf{32}\\"
            )
            lines.append(r"\midrule")

            wrote_any = False
            for util in utils:
                util_lbl = f"{int(round(util))}\\%"

                row_vals_1, row_vals_2, row_vals_3 = [], [], []
                for ppn in PPN_ORDER:
                    for n in NODES_ORDER:
                        r = _select_row_agg(df, prio=prio, timeout=timeout, util=util, ppn=ppn, nodes=n)
                        if r is None:
                            row_vals_1.append(_dash_cell())
                            row_vals_2.append(_dash_cell())
                            row_vals_3.append(_dash_cell())
                        else:
                            # ms -> s
                            row_vals_1.append(_fmt_num(pd.to_numeric(r.get("solver_duration_ms_mean"), errors="coerce") / 1000.0, 1))
                            row_vals_2.append(_fmt_num(r.get("cpu_delta_mean"), 1))
                            row_vals_3.append(_fmt_num(r.get("mem_delta_mean"), 1))

                if all(v == _dash_cell() for v in (row_vals_1 + row_vals_2 + row_vals_3)):
                    continue

                wrote_any = True
                lines.append(rf"\multirow{{3}}{{*}}{{{util_lbl}}}")
                lines.append(r"& solver\,duration\,(s) & " + " & ".join(row_vals_1) + r" \\")
                lines.append(r"& $\Delta$\,cpu\,util\,(\%) & " + " & ".join(row_vals_2) + r" \\")
                lines.append(r"& $\Delta$\,mem\,util\,(\%) & " + " & ".join(row_vals_3) + r" \\")
                lines.append(r"\midrule")

            if wrote_any:
                if lines[-1] == r"\midrule":
                    lines[-1] = r"\bottomrule"
                else:
                    lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")

                out_tex = out_dir / f"table_solver_duration_and_utils_prio{prio}_timeout{timeout}.tex"
                _write_matrix_tex(out_path=out_tex, body_lines=lines)

            # -------- with-std LaTeX (optional) --------
            if df_std is None:
                continue

            lines_s: list[str] = []
            lines_s.append(r"\begin{tabular}{@{}c l *{8}{c}@{}}")
            lines_s.append(r"\toprule")
            lines_s.append(
                r"\multirow{2}{*}{\textbf{util}} & \multirow{2}{*}{\textbf{metric}} &"
                r"\multicolumn{4}{c}{\textbf{ppn = 4}} & \multicolumn{4}{c}{\textbf{ppn = 8}}\\"
            )
            lines_s.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
            lines_s.append(
                r" &  & \textbf{4} & \textbf{8} & \textbf{16} & \textbf{32} & "
                r"\textbf{4} & \textbf{8} & \textbf{16} & \textbf{32}\\"
            )
            lines_s.append(r"\midrule")

            wrote_any_s = False
            for util in utils:
                util_lbl = f"{int(round(util))}\\%"

                row_vals_1, row_vals_2, row_vals_3 = [], [], []
                for ppn in PPN_ORDER:
                    for n in NODES_ORDER:
                        r = _select_row_agg(df_std, prio=prio, timeout=timeout, util=util, ppn=ppn, nodes=n)
                        if r is None:
                            row_vals_1.append(_dash_cell())
                            row_vals_2.append(_dash_cell())
                            row_vals_3.append(_dash_cell())
                        else:
                            m_s = pd.to_numeric(r.get("solver_duration_ms_mean"), errors="coerce") / 1000.0
                            s_s = pd.to_numeric(r.get("solver_duration_ms_mean_std"), errors="coerce") / 1000.0
                            row_vals_1.append(_fmt_pm(m_s, s_s, 1))
                            row_vals_2.append(_fmt_pm(r.get("cpu_delta_mean"), r.get("cpu_delta_mean_std"), 1))
                            row_vals_3.append(_fmt_pm(r.get("mem_delta_mean"), r.get("mem_delta_mean_std"), 1))

                if all(v == _dash_cell() for v in (row_vals_1 + row_vals_2 + row_vals_3)):
                    continue

                wrote_any_s = True
                lines_s.append(rf"\multirow{{3}}{{*}}{{{util_lbl}}}")
                lines_s.append(r"& solver\,duration\,(s) & " + " & ".join(row_vals_1) + r" \\")
                lines_s.append(r"& $\Delta$\,cpu\,util\,(\%) & " + " & ".join(row_vals_2) + r" \\")
                lines_s.append(r"& $\Delta$\,mem\,util\,(\%) & " + " & ".join(row_vals_3) + r" \\")
                lines_s.append(r"\midrule")

            if wrote_any_s:
                if lines_s[-1] == r"\midrule":
                    lines_s[-1] = r"\bottomrule"
                else:
                    lines_s.append(r"\bottomrule")
                lines_s.append(r"\end{tabular}")

                out_tex_s = out_dir / f"table_solver_duration_and_utils_prio{prio}_timeout{timeout}_with_std.tex"
                _write_matrix_tex(out_path=out_tex_s, body_lines=lines_s)

def write_table_solver_better_vs_kwok_optimal_matrix(
    *,
    df: pd.DataFrame,
    df_std: Optional[pd.DataFrame],
    out_dir: Path,
) -> None:
    df = df.copy()
    for c in ["util", "nodes", "pods_per_node", "timeout_s", "priorities"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    group_keys = ["priorities", "timeout_s", "util", "pods_per_node", "nodes"]
    mean_cols = ["solver_optimal_rate", "solver_feasible_rate", "default_optimal_rate"]
    if "config_dir" in df.columns:
        df = _aggregate_over_config_dir(df, group_keys=group_keys, mean_cols=mean_cols)

    prios = sorted(df["priorities"].dropna().astype(int).unique().tolist())
    timeouts = sorted(df["timeout_s"].dropna().astype(int).unique().tolist())
    utils = sorted(df["util"].dropna().unique().tolist())

    if df_std is not None:
        df_std = df_std.copy()
        for c in ["util", "nodes", "pods_per_node", "timeout_s", "priorities"]:
            if c in df_std.columns:
                df_std[c] = pd.to_numeric(df_std[c], errors="coerce")

        needed = ["solver_optimal_rate_std", "solver_feasible_rate_std", "default_optimal_rate_std"]
        missing = [c for c in needed if c not in df_std.columns]
        if missing:
            raise SystemExit(
                "[error] per_combo_results_with_std.csv missing std columns for better_vs_kwok matrix:\n"
                + "\n".join(f"  - {c}" for c in missing)
            )

        mean_cols_std = mean_cols + needed
        if "config_dir" in df_std.columns:
            df_std = _aggregate_over_config_dir(df_std, group_keys=group_keys, mean_cols=mean_cols_std)

    def better_pp(row: pd.Series) -> float:
        opt = pd.to_numeric(row.get("solver_optimal_rate"), errors="coerce")
        feas = pd.to_numeric(row.get("solver_feasible_rate"), errors="coerce")
        if pd.isna(opt) or pd.isna(feas):
            return np.nan
        return float((opt + feas) * 100.0)

    def kwok_pp(row: pd.Series) -> float:
        kw = pd.to_numeric(row.get("default_optimal_rate"), errors="coerce")
        return np.nan if pd.isna(kw) else float(kw * 100.0)

    def better_std_pp(row: pd.Series) -> float:
        opt_s = pd.to_numeric(row.get("solver_optimal_rate_std"), errors="coerce")
        feas_s = pd.to_numeric(row.get("solver_feasible_rate_std"), errors="coerce")
        if pd.isna(opt_s) and pd.isna(feas_s):
            return np.nan
        opt_s0 = 0.0 if pd.isna(opt_s) else float(opt_s)
        feas_s0 = 0.0 if pd.isna(feas_s) else float(feas_s)
        return float(((opt_s0 ** 2 + feas_s0 ** 2) ** 0.5) * 100.0)

    def kwok_std_pp(row: pd.Series) -> float:
        kw_s = pd.to_numeric(row.get("default_optimal_rate_std"), errors="coerce")
        return np.nan if pd.isna(kw_s) else float(kw_s * 100.0)

    for prio in prios:
        for timeout in timeouts:
            # no-std
            lines: list[str] = []
            lines.append(r"\begin{tabular}{@{}c l *{8}{c}@{}}")
            lines.append(r"\toprule")
            lines.append(
                r"\multirow{2}{*}{\textbf{util}} & \multirow{2}{*}{\textbf{metric}} &"
                r"\multicolumn{4}{c}{\textbf{ppn = 4}} & \multicolumn{4}{c}{\textbf{ppn = 8}}\\"
            )
            lines.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
            lines.append(
                r" &  & \textbf{4} & \textbf{8} & \textbf{16} & \textbf{32} & "
                r"\textbf{4} & \textbf{8} & \textbf{16} & \textbf{32}\\"
            )
            lines.append(r"\midrule")

            wrote_any = False
            for util in utils:
                util_lbl = f"{int(round(util))}\\%"

                row_better, row_kwok = [], []
                for ppn in PPN_ORDER:
                    for n in NODES_ORDER:
                        r = _select_row_agg(df, prio=prio, timeout=timeout, util=util, ppn=ppn, nodes=n)
                        if r is None:
                            row_better.append(_dash_cell())
                            row_kwok.append(_dash_cell())
                        else:
                            row_better.append(_fmt_num(better_pp(r), 1))
                            row_kwok.append(_fmt_num(kwok_pp(r), 1))

                if all(v == _dash_cell() for v in (row_better + row_kwok)):
                    continue

                wrote_any = True
                lines.append(rf"\multirow{{2}}{{*}}{{{util_lbl}}}")
                lines.append(r"& Better\,(\%) & " + " & ".join(row_better) + r" \\")
                lines.append(r"& KWOK\,Optimal\,(\%) & " + " & ".join(row_kwok) + r" \\")
                lines.append(r"\midrule")

            if wrote_any:
                if lines[-1] == r"\midrule":
                    lines[-1] = r"\bottomrule"
                else:
                    lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")

                out_tex = out_dir / f"table_solver_better_vs_kwok_optimal_prio{prio}_timeout{timeout}.tex"
                _write_matrix_tex(out_path=out_tex, body_lines=lines)

            if df_std is None:
                continue

            # with-std
            lines_s: list[str] = []
            lines_s.append(r"\begin{tabular}{@{}c l *{8}{c}@{}}")
            lines_s.append(r"\toprule")
            lines_s.append(
                r"\multirow{2}{*}{\textbf{util}} & \multirow{2}{*}{\textbf{metric}} &"
                r"\multicolumn{4}{c}{\textbf{ppn = 4}} & \multicolumn{4}{c}{\textbf{ppn = 8}}\\"
            )
            lines_s.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
            lines_s.append(
                r" &  & \textbf{4} & \textbf{8} & \textbf{16} & \textbf{32} & "
                r"\textbf{4} & \textbf{8} & \textbf{16} & \textbf{32}\\"
            )
            lines_s.append(r"\midrule")

            wrote_any_s = False
            for util in utils:
                util_lbl = f"{int(round(util))}\\%"

                row_better, row_kwok = [], []
                for ppn in PPN_ORDER:
                    for n in NODES_ORDER:
                        r = _select_row_agg(df_std, prio=prio, timeout=timeout, util=util, ppn=ppn, nodes=n)
                        if r is None:
                            row_better.append(_dash_cell())
                            row_kwok.append(_dash_cell())
                        else:
                            row_better.append(_fmt_pm(better_pp(r), better_std_pp(r), 1))
                            row_kwok.append(_fmt_pm(kwok_pp(r), kwok_std_pp(r), 1))

                if all(v == _dash_cell() for v in (row_better + row_kwok)):
                    continue

                wrote_any_s = True
                lines_s.append(rf"\multirow{{2}}{{*}}{{{util_lbl}}}")
                lines_s.append(r"& Better\,(\%) & " + " & ".join(row_better) + r" \\")
                lines_s.append(r"& KWOK\,Optimal\,(\%) & " + " & ".join(row_kwok) + r" \\")
                lines_s.append(r"\midrule")

            if wrote_any_s:
                if lines_s[-1] == r"\midrule":
                    lines_s[-1] = r"\bottomrule"
                else:
                    lines_s.append(r"\bottomrule")
                lines_s.append(r"\end{tabular}")

                out_tex_s = out_dir / f"table_solver_better_vs_kwok_optimal_prio{prio}_timeout{timeout}_with_std.tex"
                _write_matrix_tex(out_path=out_tex_s, body_lines=lines_s)

#################################################################
# Generate tables
#################################################################
write_tables_per_priority(df_base=df_per_combo, df_std=df_per_combo_std, out_dir=OUT_TABLES_DIR)

# Restored matrix tables (exact layout like your snippet)
write_table_solver_duration_and_utils_matrix(df=df_per_combo, df_std=df_per_combo_std, out_dir=OUT_TABLES_DIR)
write_table_solver_better_vs_kwok_optimal_matrix(df=df_per_combo, df_std=df_per_combo_std, out_dir=OUT_TABLES_DIR)


#################################################################
# Global plotting settings
#################################################################

# filters
PLOT_PPNS       = [4, 8]
PLOT_PRIORITIES = [1, 2, 4]
PLOT_TIMEOUTS   = [1, 10, 20]

# precision
EPS = 1e-9

# fonts
ANNOT_FS  = 3.5
mpl.rcParams.update({
    "axes.titlesize": PLOT_TITLE_FONTSIZE,
    "axes.labelsize": PLOT_AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": PLOT_TICK_FONTSIZE,
    "ytick.labelsize": PLOT_TICK_FONTSIZE,
})

# labels
TARGET_UTIL_LABEL   = "target util (%)"
NODES_LABEL         = "# of nodes"
INSTANCES_LABEL     = "% of instances"
PODS_PER_NODE_LABEL = "pods/node"

# figure saving
FIGURE_FORMATS = ["pdf", "png"]

# 2d sizes
FIGSIZE_2D = (3.5, 2.5)
CELL_FIGSIZE_2D = (2.2, 1.6)
BAR_WIDTH_2D = 0.9

# 3d sizes
FIGSIZE_3D = (8, 4)
BAR_WIDTH_3D = 0.13
ELEV_3D, AZIM_3D = 20.0, -54.0

# colors / series (stack order = bottom -> top)
set2, set3 = plt.get_cmap("Set2").colors, plt.get_cmap("Set3").colors
CATEGORIES = [
    {"key": "other",             "label": "Other",          "col": "other_rate",               "color": set2[3]},
    {"key": "solver_optimal",    "label": "Better&Optimal", "col": "solver_optimal_rate",      "color": set2[0]},
    {"key": "solver_feasible",   "label": "Better",         "col": "solver_feasible_rate",     "color": set2[1]},
    {"key": "default_optimal",   "label": "KWOK Optimal",   "col": "default_optimal_rate",     "color": set2[2]},
    {"key": "default_all",       "label": "No Calls",       "col": "default_all_running_rate", "color": set3[11]},
    {"key": "solver_failed",     "label": "Failures",       "col": "solver_failed_rate",       "color": set2[7]},
]

#################################################################
# Plotting helpers
#################################################################
def save_figure(fig: mpl.figure.Figure, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        fname = out_path.with_suffix(f".{ext}")
        fig.savefig(fname, dpi=PLOT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved figure: {out_path} ({', '.join(FIGURE_FORMATS)})")

#################################################################
# 2D bar plot as a grid: ppn vs priorities w/ util aggregated
#################################################################
def aggregate_over_util(per_combo_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["pods_per_node","priorities","timeout_s","nodes"]

    g = (
        per_combo_df
        .groupby(keys, as_index=False)
        .agg({
            "n_seeds": "sum",
            "n_seeds_not_all_running": "sum",
            "n_default_all_running": "sum",
            "n_solver_called": "sum",
            "n_solver_failed": "sum",
            "n_default_optimal": "sum",
            "n_solver_optimal": "sum",
            "n_solver_feasible": "sum",
            "n_solver_improve": "sum",
            "n_other": "sum",
            "solver_duration_ms_sum": "sum",
            "cpu_delta_sum": "sum",
            "mem_delta_sum": "sum",
        })
    )

    def safe_div(num, den):
        return num.div(den.replace(0, np.nan)).fillna(0.0)

    g["default_all_running_rate"] = safe_div(g["n_default_all_running"], g["n_seeds"])
    g["solver_called_rate"]       = safe_div(g["n_solver_called"],        g["n_seeds"])
    g["solver_failed_rate"]       = safe_div(g["n_solver_failed"],        g["n_seeds"])
    g["default_optimal_rate"]     = safe_div(g["n_default_optimal"],      g["n_seeds"])
    g["solver_optimal_rate"]      = safe_div(g["n_solver_optimal"],       g["n_seeds"])
    g["solver_feasible_rate"]     = safe_div(g["n_solver_feasible"],      g["n_seeds"])
    g["solver_improve_rate"]      = safe_div(g["n_solver_improve"],       g["n_seeds"])
    g["other_rate"]               = safe_div(g["n_other"],                g["n_seeds"])
    g["solver_duration_ms_mean"]  = safe_div(g["solver_duration_ms_sum"], g["n_solver_called"])
    g["cpu_delta_mean"]           = safe_div(g["cpu_delta_sum"],          g["n_seeds"])
    g["mem_delta_mean"]           = safe_div(g["mem_delta_sum"],          g["n_seeds"])
    return g.copy()

def plot_2d_grid_ppn_prio_with_aggregated_util(
    df_util_agg: pd.DataFrame,
    ppns: list[int],
    priorities: list[int],
    out_path: Path,
    cell_figsize: tuple[float, float]
):
    nrows, ncols = len(ppns), len(priorities)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell_figsize[0], nrows * cell_figsize[1]),
        sharex=True, sharey=True, squeeze=False
    )

    fig.supylabel("% of instances", fontsize=PLOT_AXIS_LABEL_FONTSIZE, x=0.03)

    seen_keys = set()

    for r, ppn in enumerate(ppns):
        for c, prio in enumerate(priorities):
            ax = axes[r][c]
            panel = df_util_agg[(df_util_agg["pods_per_node"] == ppn) & (df_util_agg["priorities"] == prio)].copy()

            if panel.empty:
                ax.axis("off")
                continue

            panel = panel.groupby(
                ["nodes", "timeout_s"],
                as_index=False
            )[[s["col"] for s in CATEGORIES if s["col"].endswith("_rate")]].mean()

            nodes_vals = sorted(panel["nodes"].unique().tolist())
            ts = sorted(panel["timeout_s"].unique().tolist())
            bars_per_group = max(1, len(ts))
            width = BAR_WIDTH_2D / bars_per_group
            x = np.arange(len(nodes_vals))

            for j, timeout in enumerate(ts):
                sub = panel[panel["timeout_s"] == timeout].set_index("nodes")
                xj = x + (j - (bars_per_group - 1) / 2.0) * width
                bar_height = np.zeros(len(nodes_vals), dtype=float)

                for category in CATEGORIES:
                    vals = (sub.get(category["col"], pd.Series(0.0, index=sub.index)).reindex(nodes_vals).fillna(0.0).values)
                    h = vals * 100.0
                    if np.any(h > EPS):
                        ax.bar(
                            xj, h, width=width, bottom=bar_height,
                            color=category["color"], edgecolor="black", linewidth=0.35
                        )
                        bar_height += h
                        seen_keys.add(category["key"])

                for xi, top in zip(xj, bar_height):
                    ax.text(float(xi), float(top) + 1.0, f"{int(timeout)}s",
                            ha="center", va="bottom", fontsize=ANNOT_FS)

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in nodes_vals], fontsize=PLOT_TICK_FONTSIZE)
            ax.set_ylim(0, 110)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(100.0))
            ax.tick_params(axis='y', labelsize=PLOT_TICK_FONTSIZE)

            if r == 0:
                ax.set_title(rf"#priorities={prio}", fontsize=PLOT_TITLE_FONTSIZE)

            if c == 0:
                if nrows % 2 == 1 and r == nrows // 2:
                    axis_text = f"{INSTANCES_LABEL}\n{PODS_PER_NODE_LABEL} = {ppn}"
                else:
                    axis_text = f"\n{PODS_PER_NODE_LABEL} = {ppn}"
                lbl = ax.set_ylabel(axis_text, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=10)
                lbl.set_va('center')
                lbl.set_ha('center')
                lbl.set_linespacing(1.8)

            if r == nrows - 1:
                ax.set_xlabel(NODES_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE)

    legends = [s for s in CATEGORIES if s["key"] in seen_keys][::-1]
    legend_handles = [mpatches.Rectangle((0, 0), 1, 1, fc=s["color"]) for s in legends]
    legend_labels  = [s["label"] for s in legends]
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.02),
        fontsize=PLOT_LEGEND_FONTSIZE,
        ncol=len(legend_labels),
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
    )

    save_figure(fig, out_path)

plot_2d_grid_ppn_prio_with_aggregated_util(
    df_util_agg=aggregate_over_util(df_per_combo),
    ppns=[4, 8],
    priorities=[1, 2, 4],
    out_path=OUT_FIGURES_DIR / "2d_grid_ppn_prio",
    cell_figsize=CELL_FIGSIZE_2D,
)

###############################################################
# 3D bar plot: ppn vs priorities vs timeout
###############################################################
def plot_3d_ppn_prio_timeout(df: pd.DataFrame, title: str, out_path: Path):
    utils = sorted(df["util"].unique().tolist())
    nodes = sorted(int(n) for n in df["nodes"].unique().tolist())

    def get_rate(df: pd.DataFrame, util_val, nodes_val, col):
        df_idx = df.set_index(["util","nodes"])
        try:
            return float(df_idx.loc[(util_val, nodes_val), col])
        except KeyError:
            return 0.0

    x_index = {r: i for i, r in enumerate(utils)}
    y_index = {n: j for j, n in enumerate(nodes)}

    fig, ax = plt.subplots(
        figsize=FIGSIZE_3D,
        subplot_kw={"projection": "3d"}
    )
    ax.view_init(elev=ELEV_3D, azim=AZIM_3D)
    ax.set_proj_type("ortho")

    ax.set_xlim(0, len(utils))
    ax.set_ylim(0, len(nodes))
    ax.set_zlim(0, 100)

    ax.set_xticks([x_index[r] + 0.5 for r in utils])
    ax.set_yticks([y_index[n] + 0.5 for n in nodes])
    ax.set_zticks([0, 20, 40, 60, 80, 100])
    ax.zaxis.set_major_formatter(mtick.PercentFormatter(100.0))

    ax.set_xticklabels([f"{int(round(r))}%" for r in utils], fontsize=PLOT_TICK_FONTSIZE)
    ax.set_yticklabels([str(n) for n in nodes], fontsize=PLOT_TICK_FONTSIZE)

    ax.tick_params(axis='x', labelsize=PLOT_TICK_FONTSIZE, pad=-2)
    ax.tick_params(axis='y', labelsize=PLOT_TICK_FONTSIZE, pad=-2)
    ax.tick_params(axis='z', labelsize=PLOT_TICK_FONTSIZE, pad=-1)

    ax.set_xlabel(TARGET_UTIL_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=-4.0)
    ax.set_ylabel(NODES_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=-5.5)
    ax.set_zlabel(INSTANCES_LABEL, fontsize=PLOT_AXIS_LABEL_FONTSIZE, labelpad=-5.0)

    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE, y=1.01, pad=0)

    dx = max(0.05, min(1.0, BAR_WIDTH_3D))
    dy = max(0.05, min(1.0, BAR_WIDTH_3D))

    seen_keys = set()
    for u in utils:
        for n in nodes:
            x0 = x_index[u] + (1 - dx)/2
            y0 = y_index[n] + (1 - dy)/2
            z = 0.0

            rate_solver_opt    = get_rate(df, u, n, "solver_optimal_rate")
            rate_solver_feas   = get_rate(df, u, n, "solver_feasible_rate")
            rate_solver_fail   = get_rate(df, u, n, "solver_failed_rate")
            rate_default_opt   = get_rate(df, u, n, "default_optimal_rate")
            rate_default_all   = get_rate(df, u, n, "default_all_running_rate")
            rate_other         = get_rate(df, u, n, "other_rate")

            for key, rate in [
                ("other",           rate_other),
                ("solver_optimal",  rate_solver_opt),
                ("solver_feasible", rate_solver_feas),
                ("default_optimal", rate_default_opt),
                ("default_all",     rate_default_all),
                ("solver_failed",   rate_solver_fail),
            ]:
                h = rate * 100.0
                if h > EPS:
                    color = next(s["color"] for s in CATEGORIES if s["key"] == key)
                    ax.bar3d(
                        x0, y0, z, dx, dy, h,
                        color=color, edgecolor="black", linewidth=0.35, shade=False,
                    )
                    z += h
                    seen_keys.add(key)

    legends = [s for s in CATEGORIES if s["key"] in seen_keys][::-1]
    legend_handles = [mpatches.Rectangle((0,0),1,1, fc=s["color"]) for s in legends]
    legend_labels  = [s["label"] for s in legends]
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.545, 0.92),
        fontsize=PLOT_LEGEND_FONTSIZE,
        ncol=len(legend_labels),
        handlelength=PLOT_LEGEND_HANDLE_LENGTH,
        handletextpad=PLOT_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=PLOT_LEGEND_COLUMN_SPACING,
    )

    save_figure(fig, out_path)

for ppn in [4, 8]:
    for prio in [1, 2, 4]:
        for t in [1, 10, 20]:
            sub = df_per_combo[
                (df_per_combo["pods_per_node"] == ppn)
                & (df_per_combo["priorities"] == prio)
                & (df_per_combo["timeout_s"] == t)
            ]
            if sub.empty:
                print(f"[skip] no per-combo rows for ppn={ppn}, prio={prio}, t={t}")
                continue
            title = rf"{PODS_PER_NODE_LABEL}={ppn}, #priorities={prio}, timeout={t}s"
            out_file = OUT_FIGURES_DIR / f"3d_ppn{ppn}_prio{prio}_timeout{t:02d}"
            plot_3d_ppn_prio_timeout(sub, title, out_file)
