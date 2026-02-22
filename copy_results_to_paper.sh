#!/usr/bin/env bash
# ./copy_results_to_paper.sh /mnt/c/projects/master/k8s-scheduler-opt-modes-paper/figures/generated/
# ./copy_results_to_paper.sh /mnt/d/projects/master/k8s-scheduler-opt-modes-paper/figures/generated/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="${SCRIPT_DIR}/analysis"
DEST_DIR="${1:?Usage: $0 <dest_dir>}"
DEST_DIR="${DEST_DIR%/}"

# source (relative to analysis/) → destination folder name
COPY_MAP=(
    "kwok_trace_replayer/figures    figures"
    "kwok_trace_replayer/tables     tables"
    "public_traces                  figures"
)

for entry in "${COPY_MAP[@]}"; do
    read -r src_rel dest_name <<< "$entry"
    src="${ANALYSIS_DIR}/${src_rel}"
    dest="${DEST_DIR}/${dest_name}"

    if [[ ! -d "$src" ]]; then
        echo "[skip] ${src_rel}"
        continue
    fi

    [[ -d "$dest" ]] || mkdir -p "$dest"
    cp -r "$src"/. "$dest"/
    echo "[ok] ${src_rel} → ${dest_name}"
done
