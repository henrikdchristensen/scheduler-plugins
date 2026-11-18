#!/usr/bin/env bash
set -euo pipefail
# Run as: ./make_bootstrap_folder.sh

BOOTSTRAP_DIR=bootstrap
INCLUDE_CPLEX=false
OVERRIDE=false
while (( $# )); do
  case "$1" in
    --cplex) INCLUDE_CPLEX=true ;;
    --override) OVERRIDE=true ;;
    --suffix|--suffix=*)
      if [[ "$1" == --suffix ]]; then
        suffix="${2:-}"
        if (( $# > 1 )); then shift; fi
      else
        suffix="${1#*=}"
      fi
      if [[ ! "${suffix}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
        echo '--suffix requires a suffix starting with a letter or digit and containing only letters, digits, dots, underscores or hyphens' >&2
        exit 2
      fi
      BOOTSTRAP_DIR="bootstrap-${suffix}"
      ;;
    *) echo "Usage: $0 [--cplex] [--suffix SUFFIX] [--override]" >&2; exit 2 ;;
  esac
  shift
done
CONTENT_DIR="${BOOTSTRAP_DIR}/content"
if [[ "${OVERRIDE}" != true && ( -e "${BOOTSTRAP_DIR}" || -L "${BOOTSTRAP_DIR}" ) ]]; then
  echo "Output path already exists: ${BOOTSTRAP_DIR}. Use --override to replace it." >&2
  exit 1
fi
if [[ "${INCLUDE_CPLEX}" == true && ! -s cplex_installer.bin ]]; then
  echo 'Missing cplex_installer.bin in repo root' >&2
  exit 1
fi
ENV_FILE="opt-prio.env"
echo "Loading versions from ${ENV_FILE}..."
# shellcheck source=/dev/null
set -a
source "${ENV_FILE}"
set +a

# Normalize CRLF
KUBECTL_VERSION="${KUBECTL_VERSION//$'\r'/}"
KWOK_VERSION="${KWOK_VERSION//$'\r'/}"
SCHEDULER_VERSION="${SCHEDULER_VERSION//$'\r'/}"

# Normalize tags
KUBECTL_TAG="${KUBECTL_VERSION}"
[[ "${KUBECTL_TAG}" != v* ]] && KUBECTL_TAG="v${KUBECTL_TAG}"

KWOK_TAG="${KWOK_VERSION}"
[[ "${KWOK_TAG}" != v* ]] && KWOK_TAG="v${KWOK_TAG}"

if [[ "${OVERRIDE}" == true ]]; then
  rm -rf -- "${BOOTSTRAP_DIR}"
fi
echo "Creating ${BOOTSTRAP_DIR} directory..."
mkdir -- "${BOOTSTRAP_DIR}"
mkdir -- "${CONTENT_DIR}"

echo "Recording git commit in bootstrap/content/GIT_COMMIT..."
# safe.directory: the repo is owned by another user when building in a container
GIT_COMMIT="$(git -c safe.directory='*' rev-parse HEAD)"
[[ -n "$(git -c safe.directory='*' status --porcelain)" ]] && GIT_COMMIT="${GIT_COMMIT}-dirty"
echo "${GIT_COMMIT}" > "${CONTENT_DIR}/GIT_COMMIT"

echo "Copying ${ENV_FILE} into bootstrap..."
cp "${ENV_FILE}" "${BOOTSTRAP_DIR}/opt-prio.env"

echo "Copying data subfolder into bootstrap/content..."
rsync -a --exclude='__pycache__' data/ "${CONTENT_DIR}/data"

if [[ "${INCLUDE_CPLEX}" == true ]]; then
  echo "Copying cplex_installer.bin into ${CONTENT_DIR}..."
  cp cplex_installer.bin "${CONTENT_DIR}/cplex_installer.bin"
elif [[ -f gurobi.lic ]]; then
  echo "Copying gurobi.lic into bootstrap/content (git-ignored, needed for --solver-type gurobi)..."
  cp gurobi.lic "${CONTENT_DIR}/gurobi.lic"
fi

echo "Copying scripts subfolder into bootstrap/content/scripts..."
rsync -a --exclude='__pycache__' scripts/ "${CONTENT_DIR}/scripts/"

echo "Move bootstrap files to 'bootstrap' folder and set permissions..."
mv "${CONTENT_DIR}/scripts/bootstrap/"* "${BOOTSTRAP_DIR}/"
rmdir "${CONTENT_DIR}/scripts/bootstrap"
sed -i.bak $'s/\r$//' "${BOOTSTRAP_DIR}/"*.sh
rm "${BOOTSTRAP_DIR}/"*.sh.bak
chmod +x "${BOOTSTRAP_DIR}/"*.sh

echo "Copying manifests files into 'manifests' folder..."
mkdir -p "${CONTENT_DIR}/manifests/mypriorityoptimizer" "${CONTENT_DIR}/manifests/mydeterministicscore"
cp -r manifests/mypriorityoptimizer/* "${CONTENT_DIR}/manifests/mypriorityoptimizer/"
cp -r manifests/mydeterministicscore/* "${CONTENT_DIR}/manifests/mydeterministicscore/"

echo "Building scheduler binary + downloading kube-apiserver via build.sh..."
# The bootstrap folder always targets linux/amd64, built with the pinned Go version
TARGET_OS=linux TARGET_ARCH=amd64 GOTOOLCHAIN="go${GO_VERSION//$'\r'/}" bash ./build.sh
chmod +x bin/kube-scheduler

echo "Creating bootstrap/content/bin and shipping binaries..."
mkdir -p "${CONTENT_DIR}/bin"

# Ship kube-scheduler
cp bin/kube-scheduler "${CONTENT_DIR}/bin/kube-scheduler"
chmod +x "${CONTENT_DIR}/bin/kube-scheduler"

# Ship kube-apiserver (downloaded by build.sh)
cp bin/kube-apiserver "${CONTENT_DIR}/bin/kube-apiserver"
chmod +x "${CONTENT_DIR}/bin/kube-apiserver"

# Ship kube-controller-manager (downloaded by build.sh)
cp bin/kube-controller-manager "${CONTENT_DIR}/bin/kube-controller-manager"
chmod +x "${CONTENT_DIR}/bin/kube-controller-manager"

# Ship kubectl
echo "Downloading kubectl ${KUBECTL_TAG}..."
curl -fsSL -o "${CONTENT_DIR}/bin/kubectl" \
  "https://dl.k8s.io/release/${KUBECTL_TAG}/bin/linux/amd64/kubectl"
chmod +x "${CONTENT_DIR}/bin/kubectl"

# Ship kwokctl + kwok
echo "Downloading kwokctl/kwok ${KWOK_TAG}..."
curl -fsSL -o "${CONTENT_DIR}/bin/kwokctl" \
  "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_TAG}/kwokctl-linux-amd64"
curl -fsSL -o "${CONTENT_DIR}/bin/kwok" \
  "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_TAG}/kwok-linux-amd64"
chmod +x "${CONTENT_DIR}/bin/kwokctl" "${CONTENT_DIR}/bin/kwok"

echo "Verifying that all shipped binaries are linux/amd64..."
for bin in "${CONTENT_DIR}/bin/"*; do
  # ELF magic in bytes 0-3 and machine type x86-64 (0x3e) in bytes 18-19; works on any build host
  if [[ "$(od -An -tx1 -N20 "${bin}" | tr -d ' \n')" != 7f454c46*3e00 ]]; then
    echo "ERROR: ${bin} is not a linux/amd64 binary; delete bin/$(basename "${bin}") and rerun" >&2
    exit 1
  fi
done

echo "Done. Built from ${GIT_COMMIT}."
if [[ "${GIT_COMMIT}" == *-dirty ]]; then
  echo "WARNING: uncommitted changes; results will be labelled '-dirty'. Commit and rebuild before a real run." >&2
fi
