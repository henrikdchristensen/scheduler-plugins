#!/usr/bin/env bash
set -euo pipefail

# Load environment variables
ENV_FILE="opt-prio.env"
echo "=== Load environment variables from ${ENV_FILE} ==="
# shellcheck source=/dev/null
set -a
source "${ENV_FILE}"
set +a

# Normalize CRLF
SCHEDULER_VERSION="${SCHEDULER_VERSION//$'\r'/}"

echo "Environment variables loaded."

# Detect OS and architecture for cross-platform support
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$(uname -m)" in
  x86_64)  HOST_ARCH="amd64" ;;
  aarch64|arm64) HOST_ARCH="arm64" ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
echo "Detected platform: ${HOST_OS}/${HOST_ARCH}"

mkdir -p bin

echo "Building kube-scheduler with mypriorityoptimizer plugin (${SCHEDULER_VERSION})..."
make build-scheduler GO_BUILD_ENV="CGO_ENABLED=0 GOOS=${HOST_OS} GOARCH=${HOST_ARCH}" VERSION="${SCHEDULER_VERSION}"

APISERVER_BIN="bin/kube-apiserver"
APISERVER_URL="https://dl.k8s.io/release/${SCHEDULER_VERSION}/bin/${HOST_OS}/${HOST_ARCH}/kube-apiserver"

if [[ -x "${APISERVER_BIN}" ]] && [[ -s "${APISERVER_BIN}" ]]; then
  echo "kube-apiserver already present (${APISERVER_BIN}); skipping download."
else
  echo "Downloading kube-apiserver ${SCHEDULER_VERSION}..."
  if ! curl -fsSL -o "${APISERVER_BIN}" "${APISERVER_URL}"; then
    echo "WARNING: kube-apiserver not available for ${HOST_OS}/${HOST_ARCH} (not published by upstream). Skipping."
    rm -f "${APISERVER_BIN}"
  else
    chmod +x "${APISERVER_BIN}"
  fi
fi

KUBECONTROLLERMANAGER_BIN="bin/kube-controller-manager"
KUBECONTROLLERMANAGER_URL="https://dl.k8s.io/release/${SCHEDULER_VERSION}/bin/${HOST_OS}/${HOST_ARCH}/kube-controller-manager"

if [[ -x "${KUBECONTROLLERMANAGER_BIN}" ]] && [[ -s "${KUBECONTROLLERMANAGER_BIN}" ]]; then
  echo "kube-controller-manager already present (${KUBECONTROLLERMANAGER_BIN}); skipping download."
else
  echo "Downloading kube-controller-manager ${SCHEDULER_VERSION}..."
  if ! curl -fsSL -o "${KUBECONTROLLERMANAGER_BIN}" "${KUBECONTROLLERMANAGER_URL}"; then
    echo "WARNING: kube-controller-manager not available for ${HOST_OS}/${HOST_ARCH} (not published by upstream). Skipping."
    rm -f "${KUBECONTROLLERMANAGER_BIN}"
  else
    chmod +x "${KUBECONTROLLERMANAGER_BIN}"
  fi
fi

echo "Build completed."
