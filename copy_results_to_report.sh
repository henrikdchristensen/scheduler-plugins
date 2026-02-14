#!/usr/bin/env bash
# ./copy_results_to_report.sh /mnt/d/projects/master/SDU-Master-Thesis-Report/figures/generated/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="${SCRIPT_DIR}/analysis"
DEST_DIR="${1:?Usage: $0 <dest_dir>}"
DEST_DIR="${DEST_DIR%/}"

# source (relative to analysis/) → destination folder name
COPY_MAP=(
    "kwok_trace_replayer/figures            trace_replayer_figures"
    "kwok_trace_replayer/tables             trace_replayer_tables"
    "public_traces                          public_traces_figures"
    "kwok_workload_once/figures/cp_sat      optimizer_cp_sat_figures"
    "kwok_workload_once/tables/cp_sat       optimizer_cp_sat_tables"
    "kwok_workload_once/figures/gurobi      optimizer_gurobi_figures"
    "kwok_workload_once/tables/gurobi       optimizer_gurobi_tables"
)

for entry in "${COPY_MAP[@]}"; do
    read -r src_rel dest_name <<< "$entry"
    src="${ANALYSIS_DIR}/${src_rel}"
    dest="${DEST_DIR}/${dest_name}"

    if [[ ! -d "$src" ]]; then
        echo "[skip] ${src_rel}"
        continue
    fi

    [[ -d "$dest" ]] && rm -rf "$dest"
    mkdir -p "$(dirname "$dest")"
    cp -r "$src" "$dest"
    echo "[ok] ${src_rel} → ${dest_name}"
done
