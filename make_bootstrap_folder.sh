#!/usr/bin/env bash
set -euo pipefail

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

echo "Cleaning and creating bootstrap directory..."
rm -rf bootstrap
mkdir -p bootstrap/content

echo "Copying ${ENV_FILE} into bootstrap..."
cp "${ENV_FILE}" bootstrap/opt-prio.env

echo "Copying data subfolder into bootstrap/content..."
rsync -a --exclude='__pycache__' data/ bootstrap/content/data

echo "Copying scripts subfolder into bootstrap/content/scripts..."
rsync -a --exclude='__pycache__' scripts/ bootstrap/content/scripts/

echo "Move bootstrap files to 'bootstrap' folder and set permissions..."
mv bootstrap/content/scripts/bootstrap/* bootstrap/
rmdir bootstrap/content/scripts/bootstrap
sed -i 's/\r$//' bootstrap/*.sh
chmod +x bootstrap/*.sh

echo "Copying manifests files into 'manifests' folder..."
mkdir -p bootstrap/content/manifests/mypriorityoptimizer
mkdir -p bootstrap/content/manifests/mydeterministicscore
cp -r manifests/mypriorityoptimizer/*  bootstrap/content/manifests/mypriorityoptimizer/
cp -r manifests/mydeterministicscore/* bootstrap/content/manifests/mydeterministicscore/

echo "Building scheduler binary + downloading kube-apiserver via build.sh..."
bash ./build.sh
chmod +x bin/kube-scheduler

echo "Creating bootstrap/content/bin and shipping binaries..."
mkdir -p bootstrap/content/bin

# Ship kube-scheduler
cp bin/kube-scheduler bootstrap/content/bin/kube-scheduler
chmod +x bootstrap/content/bin/kube-scheduler

# Ship kube-apiserver (downloaded by build.sh)
cp bin/kube-apiserver bootstrap/content/bin/kube-apiserver
chmod +x bootstrap/content/bin/kube-apiserver

# Ship kube-controller-manager (downloaded by build.sh)
cp bin/kube-controller-manager bootstrap/content/bin/kube-controller-manager
chmod +x bootstrap/content/bin/kube-controller-manager

# Ship kubectl
echo "Downloading kubectl ${KUBECTL_TAG}..."
curl -fsSL -o bootstrap/content/bin/kubectl \
  "https://dl.k8s.io/release/${KUBECTL_TAG}/bin/linux/amd64/kubectl"
chmod +x bootstrap/content/bin/kubectl

# Ship kwokctl + kwok
echo "Downloading kwokctl/kwok ${KWOK_TAG}..."
curl -fsSL -o bootstrap/content/bin/kwokctl \
  "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_TAG}/kwokctl-linux-amd64"
curl -fsSL -o bootstrap/content/bin/kwok \
  "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_TAG}/kwok-linux-amd64"
chmod +x bootstrap/content/bin/kwokctl bootstrap/content/bin/kwok

# Verify binaries
bootstrap/content/bin/kubectl version --client=true >/dev/null
bootstrap/content/bin/kwokctl --version >/dev/null
bootstrap/content/bin/kwok --version >/dev/null

echo "Done."
