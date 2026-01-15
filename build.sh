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

mkdir -p bin

echo "Building kube-scheduler with mypriorityoptimizer plugin (${SCHEDULER_VERSION})..."
make build-scheduler GO_BUILD_ENV='CGO_ENABLED=0 GOOS=linux GOARCH=amd64' VERSION="${SCHEDULER_VERSION}"

APISERVER_BIN="bin/kube-apiserver"
APISERVER_URL="https://dl.k8s.io/release/${SCHEDULER_VERSION}/bin/linux/amd64/kube-apiserver"

if [[ -x "${APISERVER_BIN}" ]] && [[ -s "${APISERVER_BIN}" ]]; then
  echo "kube-apiserver already present (${APISERVER_BIN}); skipping download."
else
  echo "Downloading kube-apiserver ${SCHEDULER_VERSION}..."
  curl -fsSL -o "${APISERVER_BIN}" "${APISERVER_URL}"
  chmod +x "${APISERVER_BIN}"
fi

KUBECONTROLLERMANAGER_BIN="bin/kube-controller-manager"
KUBECONTROLLERMANAGER_URL="https://dl.k8s.io/release/${SCHEDULER_VERSION}/bin/linux/amd64/kube-controller-manager"

if [[ -x "${KUBECONTROLLERMANAGER_BIN}" ]] && [[ -s "${KUBECONTROLLERMANAGER_BIN}" ]]; then
  echo "kube-controller-manager already present (${KUBECONTROLLERMANAGER_BIN}); skipping download."
else
  echo "Downloading kube-controller-manager ${SCHEDULER_VERSION}..."
  curl -fsSL -o "${KUBECONTROLLERMANAGER_BIN}" "${KUBECONTROLLERMANAGER_URL}"
  chmod +x "${KUBECONTROLLERMANAGER_BIN}"
fi

echo "Build completed."
