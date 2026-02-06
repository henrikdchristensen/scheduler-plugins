#!/usr/bin/env bash
# bootstrap.sh
# Can be run as: ./bootstrap.sh --content-dir <dir> --runner <test_runner|trace_replayer> --job-file <job-file.yaml>
set -euo pipefail

########################## Defaults ##########################
CONTENT_DIR="${CONTENT_DIR:-/workspace}"
CONTENT_DIR_WAIT_TIMEOUT_S="${CONTENT_DIR_WAIT_TIMEOUT:-30}"  # seconds
CONTENT_DIR_WAIT_INTERVAL_S="${CONTENT_DIR_WAIT_INTERVAL:-2}" # seconds

PYTHON_SOLVER_OUT_VENV_DIR="/opt/venv"
PYTHON_SOLVER_OUT_SCRIPT_DIR="/opt/solver"

# Solver selection: cp_sat (default), cbc
SOLVER_TYPE="${SOLVER_TYPE:-cp_sat}"

# Runner selection: test_runner (default) or trace_replayer
RUNNER="${RUNNER:-test_runner}"

# Shared config
CLUSTER_NAME="${CLUSTER_NAME:-}"
KWOK_RUNTIME="${KWOK_RUNTIME:-binary}"  # binary | docker
JOB_FILE="${JOB_FILE:-}"                # can be relative to CONTENT_DIR
LOG_LEVEL="${LOG_LEVEL:-}"
CLEAN_START="${CLEAN_START:-}"

# For test_runner
RESULTS_DIR="${RESULTS_DIR:-}" # can be relative to CONTENT_DIR
CONFIG_FILE="${CONFIG_FILE:-}" # can be relative to CONTENT_DIR
SEED_FILE="${SEED_FILE:-}"     # can be relative to CONTENT_DIR
SEED="${SEED:-}"
RE_RUN_SEEDS="${RE_RUN_SEEDS:-}"
DEFAULT_SCHEDULER="${DEFAULT_SCHEDULER:-}" # if true, use default kube-scheduler instead of custom one
SOLVER_TRIGGER="${SOLVER_TRIGGER:-}"
PAUSE="${PAUSE:-}"
SEEDS_NOT_ALL_RUNNING="${SEEDS_NOT_ALL_RUNNING:-}" # int
SAVE_SOLVER_STATS="${SAVE_SOLVER_STATS:-}"
SAVE_SCHEDULER_LOGS="${SAVE_SCHEDULER_LOGS:-}"

# For trace_replayer
TRACE_DIR="${TRACE_DIR:-}"                     # can be relative to CONTENT_DIR
KWOKCTL_CONFIG_FILE="${KWOKCTL_CONFIG_FILE:-}" # can be relative to CONTENT_DIR
NODE_CPU="${NODE_CPU:-}"
NODE_MEM="${NODE_MEM:-}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-}"

########################## Helpers ##########################
log(){ printf '[%s] %s\n' "$1" "$2"; }
die(){ log error "$1"; exit 1; }

run_root(){
  local cmd="$*"
  if [ "$(id -u)" -eq 0 ]; then
    bash -lc "${cmd}"
  else
    command -v sudo >/dev/null 2>&1 || die "need root (sudo not found)"
    # Preserve env so variables like CONTENT_DIR/RUNNER/etc survive.
    sudo -E bash -lc "${cmd}"
  fi
}

wait_for_dir() {
  local dir="${1:?}" timeout="${2:-}" interval="${3:-1}"
  local start elapsed
  start="$(date +%s)"
  log wait "waiting for CONTENT_DIR='${dir}'"
  while [ ! -d "$dir" ]; do
    sleep "$interval"
    if [ -n "$timeout" ]; then
      elapsed="$(( $(date +%s) - start ))"
      if [ "$elapsed" -ge "$timeout" ]; then
        die "CONTENT_DIR not found after ${elapsed}s: ${dir}"
      fi
    fi
  done
  log ok "CONTENT_DIR available: ${dir}"
}

to_abs_under_folder() {
  local path="${1:-}"
  if [ -z "$path" ]; then
    echo ""
  elif [[ "$path" = /* ]]; then
    echo "$path"
  else
    echo "${CONTENT_DIR%/}/$path"
  fi
}

resolve_paths_relative_to_folder() {
  CONFIG_FILE="$(to_abs_under_folder "$CONFIG_FILE")"
  RESULTS_DIR="$(to_abs_under_folder "$RESULTS_DIR")"
  SEED_FILE="$(to_abs_under_folder "$SEED_FILE")"
  JOB_FILE="$(to_abs_under_folder "$JOB_FILE")"
  TRACE_DIR="$(to_abs_under_folder "$TRACE_DIR")"
  KWOKCTL_CONFIG_FILE="$(to_abs_under_folder "$KWOKCTL_CONFIG_FILE")"
}

print_cfg() {
  log cfg "CONTENT_DIR=${CONTENT_DIR}"
  log cfg "KWOK_RUNTIME=${KWOK_RUNTIME}"
  log cfg "RUNNER=${RUNNER}"
  log cfg "SOLVER_TYPE=${SOLVER_TYPE}"

  if [ -n "${JOB_FILE}" ]; then
    log cfg "JOB_FILE=${JOB_FILE}"
  else
    log cfg "CLUSTER_NAME=${CLUSTER_NAME:-<unset>}"
    log cfg "CONFIG_FILE=${CONFIG_FILE:-<unset>}"
    log cfg "RESULTS_DIR=${RESULTS_DIR:-<unset>}"
    log cfg "SEED_FILE=${SEED_FILE:-<unset>}"
    log cfg "SEED=${SEED:-<unset>}"
    log cfg "RE_RUN_SEEDS=${RE_RUN_SEEDS:-<unset>}"
    log cfg "CLEAN_START=${CLEAN_START:-<unset>}"
    log cfg "LOG_LEVEL=${LOG_LEVEL:-<unset>}"
    log cfg "PAUSE=${PAUSE:-<unset>}"
  fi

  log cfg "DEFAULT_SCHEDULER=${DEFAULT_SCHEDULER:-<unset>}"
  log cfg "SEEDS_NOT_ALL_RUNNING=${SEEDS_NOT_ALL_RUNNING:-<unset>}"
  log cfg "SOLVER_TRIGGER=${SOLVER_TRIGGER:-<unset>}"
  log cfg "SAVE_SOLVER_STATS=${SAVE_SOLVER_STATS:-<unset>}"
  log cfg "SAVE_SCHEDULER_LOGS=${SAVE_SCHEDULER_LOGS:-<unset>}"

  log cfg "TRACE_DIR=${TRACE_DIR:-<unset>}"
  log cfg "KWOKCTL_CONFIG_FILE=${KWOKCTL_CONFIG_FILE:-<unset>}"
  log cfg "NODE_CPU=${NODE_CPU:-<unset>}"
  log cfg "NODE_MEM=${NODE_MEM:-<unset>}"
  log cfg "MONITOR_INTERVAL=${MONITOR_INTERVAL:-<unset>}"
}

require_file() { [ -e "$1" ] || die "missing required file: $1"; }

ensure_shipped_k8s_tools() {
  log init "verifying shipped kubernetes tools (kubectl/kwokctl/kwok) and scheduler"

  require_file "${CONTENT_DIR}/bin/kubectl"
  require_file "${CONTENT_DIR}/bin/kwokctl"
  require_file "${CONTENT_DIR}/bin/kwok"
  require_file "${CONTENT_DIR}/bin/kube-scheduler"
  require_file "${CONTENT_DIR}/bin/kube-apiserver"
  require_file "${CONTENT_DIR}/bin/kube-controller-manager"

  run_root "
    set -euo pipefail
    chmod +x \
      '${CONTENT_DIR}/bin/kubectl' \
      '${CONTENT_DIR}/bin/kwokctl' \
      '${CONTENT_DIR}/bin/kwok' \
      '${CONTENT_DIR}/bin/kube-scheduler' \
      '${CONTENT_DIR}/bin/kube-apiserver' \
      '${CONTENT_DIR}/bin/kube-controller-manager'
  "

  # Ensure shipped versions are used
  export PATH="${CONTENT_DIR}/bin:${PATH}"

  kubectl version --client=true >/dev/null 2>&1 || die "shipped kubectl not runnable"
  kwokctl --version >/dev/null 2>&1 || die "shipped kwokctl not runnable"
  kwok --version    >/dev/null 2>&1 || die "shipped kwok not runnable"

  log ok "shipped tools ready (PATH prefixed with ${CONTENT_DIR}/bin)"
}

ensure_system_python() {
  command -v python3 >/dev/null 2>&1 || die "python3 not found on system"
  python3 -V >/dev/null 2>&1 || die "python3 not runnable"

  if command -v pip3 >/dev/null 2>&1; then
    pip3 --version >/dev/null 2>&1 || die "pip3 not runnable"
  else
    python3 -m pip --version >/dev/null 2>&1 || die "pip not available for python3"
  fi
}

pip_install() {
  local pip="$1"
  local req="$2"
  run_root "
    set -euo pipefail
    '${pip}' install --no-cache-dir -r '${req}'
  "
}

stage_solver_and_venv() {
  log init "staging solver to ${PYTHON_SOLVER_OUT_SCRIPT_DIR} (venv @ ${PYTHON_SOLVER_OUT_VENV_DIR})"

  # Determine solver script based on SOLVER_TYPE
  local solver_script
  case "${SOLVER_TYPE}" in
    cbc)
      solver_script="solver_cbc.py"
      ;;
    cp_sat|*)
      solver_script="solver_cp_sat.py"
      ;;
  esac
  log cfg "SOLVER_TYPE=${SOLVER_TYPE} -> using ${solver_script}"

  require_file "${CONTENT_DIR}/scripts/python_solver/${solver_script}"
  require_file "${CONTENT_DIR}/scripts/python_solver/requirements.txt"

  ensure_system_python

  run_root "
    set -euo pipefail
    install -d -m 0755 '${PYTHON_SOLVER_OUT_SCRIPT_DIR}'
    install -d -m 0755 '${PYTHON_SOLVER_OUT_VENV_DIR}'
    cp -a '${CONTENT_DIR}/scripts/python_solver/${solver_script}' '${PYTHON_SOLVER_OUT_SCRIPT_DIR}/solver.py'

    python3 -m venv '${PYTHON_SOLVER_OUT_VENV_DIR}'
    '${PYTHON_SOLVER_OUT_VENV_DIR}/bin/python' -m ensurepip --upgrade || true
    '${PYTHON_SOLVER_OUT_VENV_DIR}/bin/python' -m pip install --upgrade pip setuptools wheel

    '${PYTHON_SOLVER_OUT_VENV_DIR}/bin/python' - <<'PY'
import sys
print('venv python:', sys.executable)
print('venv version:', sys.version.split()[0])
PY
  "

  pip_install "${PYTHON_SOLVER_OUT_VENV_DIR}/bin/pip" "${CONTENT_DIR}/scripts/python_solver/requirements.txt"

  log ok "staged solver + venv"
}

install_test_scripts_requirements() {
  log init "installing runner requirements (online pip)"

  if [ -f "${CONTENT_DIR}/scripts/kwok_workload_once/requirements.txt" ]; then
    pip_install "${PYTHON_SOLVER_OUT_VENV_DIR}/bin/pip" "${CONTENT_DIR}/scripts/kwok_workload_once/requirements.txt"
  fi
  if [ -f "${CONTENT_DIR}/scripts/kwok_trace_replayer/requirements.txt" ]; then
    pip_install "${PYTHON_SOLVER_OUT_VENV_DIR}/bin/pip" "${CONTENT_DIR}/scripts/kwok_trace_replayer/requirements.txt"
  fi

  log ok "runner requirements installed"
}

######################## Stages ########################
stage_setup() {
  log init "setup starting"
  resolve_paths_relative_to_folder
  print_cfg

  ensure_shipped_k8s_tools
  stage_solver_and_venv

  log ok "setup done"
}

stage_test() {
  log init "KWOK test starting"
  resolve_paths_relative_to_folder
  print_cfg

  # Install deps as root (pip into /opt/venv), then run the selected runner as root too.
  run_root "
    set -euo pipefail

    export PATH='${CONTENT_DIR}/bin':\"\$PATH\"

    # Debug
    echo '[dbg] root id:' \$(id)
    echo '[dbg] root PATH:' \"\$PATH\"
    command -v kwokctl >/dev/null 2>&1 || { echo '[dbg] kwokctl not in PATH'; ls -la '${CONTENT_DIR}/bin' || true; exit 1; }
    kwokctl --version >/dev/null 2>&1 || { echo '[dbg] kwokctl exists but not runnable'; exit 1; }

    # Install runner requirements (as root, using venv pip)
    if [ -f '${CONTENT_DIR}/scripts/kwok_workload_once/requirements.txt' ]; then
      '${PYTHON_SOLVER_OUT_VENV_DIR}/bin/pip' install --no-cache-dir -r '${CONTENT_DIR}/scripts/kwok_workload_once/requirements.txt'
    fi
    if [ -f '${CONTENT_DIR}/scripts/kwok_trace_replayer/requirements.txt' ]; then
      '${PYTHON_SOLVER_OUT_VENV_DIR}/bin/pip' install --no-cache-dir -r '${CONTENT_DIR}/scripts/kwok_trace_replayer/requirements.txt'
    fi

    cd '${CONTENT_DIR}'

    case '${RUNNER}' in
      trace_replayer)
        # Build args safely (bash array in root shell)
        args=()
        [ -n '${KWOK_RUNTIME}'        ] && args+=( --kwok-runtime '${KWOK_RUNTIME}' )
        [ -n '${CLUSTER_NAME}'        ] && args+=( --cluster-name '${CLUSTER_NAME}' )
        [ -n '${TRACE_DIR}'           ] && args+=( --trace-dir '${TRACE_DIR}' )
        [ -n '${KWOKCTL_CONFIG_FILE}' ] && args+=( --kwokctl-config-file '${KWOKCTL_CONFIG_FILE}' )
        [ -n '${NODE_CPU}'            ] && args+=( --node-cpu '${NODE_CPU}' )
        [ -n '${NODE_MEM}'            ] && args+=( --node-mem '${NODE_MEM}' )
        [ -n '${MONITOR_INTERVAL}'    ] && args+=( --monitor-interval '${MONITOR_INTERVAL}' )
        [ -n '${LOG_LEVEL}'           ] && args+=( --log-level '${LOG_LEVEL}' )
        [ -n '${JOB_FILE}'            ] && args+=( --job-file '${JOB_FILE}' )

        '${PYTHON_SOLVER_OUT_VENV_DIR}/bin/python' -m scripts.kwok_trace_replayer.trace_replayer \"\${args[@]}\"
        ;;

      test_runner|*)
        args=()
        [ -n '${KWOK_RUNTIME}'        ] && args+=( --kwok-runtime '${KWOK_RUNTIME}' )
        [ -n '${CLUSTER_NAME}'        ] && args+=( --cluster-name '${CLUSTER_NAME}' )
        [ -n '${CONFIG_FILE}'         ] && args+=( --config-file '${CONFIG_FILE}' )
        [ -n '${RESULTS_DIR}'         ] && args+=( --results-dir '${RESULTS_DIR}' )
        [ -n '${SEED_FILE}'           ] && args+=( --seed-file '${SEED_FILE}' )
        [ -n '${SEED}'                ] && args+=( --seed '${SEED}' )
        [ -n '${REPEATS:-}'           ] && args+=( --repeats '${REPEATS:-}' )
        [ -n '${LOG_LEVEL}'           ] && args+=( --log-level '${LOG_LEVEL}' )
        [ -n '${JOB_FILE}'            ] && args+=( --job-file '${JOB_FILE}' )
        [ -n '${SEEDS_NOT_ALL_RUNNING}' ] && args+=( --seeds-not-all-running '${SEEDS_NOT_ALL_RUNNING}' )
        [ -n '${DEFAULT_SCHEDULER}'   ] && args+=( --default-scheduler '${DEFAULT_SCHEDULER}' )
        [ -n '${KWOKCTL_CONFIG_FILE}' ] && args+=( --kwokctl-config-file '${KWOKCTL_CONFIG_FILE}' )
        [ -n '${SOLVER_TYPE}'         ] && args+=( --solver-type '${SOLVER_TYPE}' )

        # passthrough flags (computed outside root) isn't reliable; re-compute inside if you need it,
        # or just keep it simple for now.

        '${PYTHON_SOLVER_OUT_VENV_DIR}/bin/python' -m scripts.kwok_workload_once.test_runner \"\${args[@]}\"
        ;;
    esac
  "

  log ok "${RUNNER} done"
}

##################### Args and Dispatch #####################
# FORMAT per line: name|VAR|kind|pass
# kind: "flag" (boolean) or "value" (needs a value)
# pass: the flag to forward ("" to disable)
FLAGS_SPEC=(
  "content-dir|CONTENT_DIR|value|"
  "cluster-name|CLUSTER_NAME|value|"
  "kwok-runtime|KWOK_RUNTIME|value|"
  "config-file|CONFIG_FILE|value|"
  "results-dir|RESULTS_DIR|value|"
  "seed-file|SEED_FILE|value|"
  "seed|SEED|value|"
  "repeats|REPEATS|value|"
  "job-file|JOB_FILE|value|"
  "solver-type|SOLVER_TYPE|value|"
  "solver-trigger|SOLVER_TRIGGER|flag|--solver-trigger"
  "save-solver-stats|SAVE_SOLVER_STATS|flag|--save-solver-stats"
  "save-scheduler-logs|SAVE_SCHEDULER_LOGS|flag|--save-scheduler-logs"
  "seeds-not-all-running|SEEDS_NOT_ALL_RUNNING|value|"
  "re-run-seeds|RE_RUN_SEEDS|flag|--re-run-seeds"
  "clean-start|CLEAN_START|flag|--clean-start"
  "pause|PAUSE|flag|--pause"
  "log-level|LOG_LEVEL|value|"
  "default-scheduler|DEFAULT_SCHEDULER|flag|--default-scheduler"
  "runner|RUNNER|value|"
  "trace-dir|TRACE_DIR|value|"
  "kwokctl-config-file|KWOKCTL_CONFIG_FILE|value|"
  "node-cpu|NODE_CPU|value|"
  "node-mem|NODE_MEM|value|"
  "monitor-interval|MONITOR_INTERVAL|value|"
)

get_spec_field() { # usage: get_spec_field "<name>" <idx>
  local name="$1" idx="$2" row IFS='|'
  for row in "${FLAGS_SPEC[@]}"; do
    read -r n var kind pass <<<"$row"
    if [ "$n" = "$name" ]; then
      case "$idx" in
        0) echo "$n"   ;; 1) echo "$var" ;;
        2) echo "$kind";; 3) echo "$pass";;
      esac
      return 0
    fi
  done
  return 1
}

set_var() { local var="$1" val="$2"; printf -v "$var" '%s' "$val"; }

parse_cli_using_spec() {
  case "${1-}" in all|setup|test) cmd="$1"; shift;; esac

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --*=*)
        local opt="${1%%=*}" val="${1#*=}"
        opt="${opt#--}"
        local var
        var="$(get_spec_field "$opt" 1)" || die "unknown argument: --$opt"
        set_var "$var" "${val}"
        ;;
      --*)
        local opt="${1#--}"
        local var kind
        var="$(get_spec_field "$opt" 1)" || die "unknown argument: --$opt"
        kind="$(get_spec_field "$opt" 2)"
        if [ "$kind" = "flag" ]; then
          if [ "$#" -ge 2 ] && [[ "$2" != --* ]] && [[ "$2" =~ ^(true|false)$ ]]; then
            set_var "$var" "$2"; shift
          else
            set_var "$var" "true"
          fi
        else
          [ "$#" -ge 2 ] || die "missing value for --$opt"
          set_var "$var" "$2"; shift
        fi
        ;;
      --) shift; break ;;
      *) die "unknown argument: $1" ;;
    esac
    shift
  done
}

build_passthrough_flags() {
  local out=() row
  local OLDIFS="$IFS"
  IFS='|'
  for row in "${FLAGS_SPEC[@]}"; do
    read -r name var kind pass <<<"$row"
    [ -n "$pass" ] || continue
    if [ "$kind" = "flag" ]; then
      if [ -n "${!var:-}" ] && [ "${!var}" != "false" ]; then
        out+=("$pass")
      fi
    fi
  done
  IFS="$OLDIFS"
  printf '%s\n' "${out[@]}"
}

cmd="all"
parse_cli_using_spec "$@"

wait_for_dir "${CONTENT_DIR}" "${CONTENT_DIR_WAIT_TIMEOUT_S}" "${CONTENT_DIR_WAIT_INTERVAL_S}"

case "${cmd}" in
  setup) stage_setup ;;
  test)  stage_test ;;
  all)   stage_setup; stage_test ;;
  *)     die "unknown command: ${cmd} (expected: all|setup|test)" ;;
esac

exit 0
