#!/usr/bin/env bash
set -euo pipefail

# Load environment variables
ENV_FILE="opt-prio.env"
echo "=== Load environment variables from ${ENV_FILE} ==="
# shellcheck source=/dev/null
set -a
source "${ENV_FILE}"
set +a
echo "Environment variables loaded."

echo "Building kube-scheduler with mypriorityoptimizer plugin..."
make build-scheduler GO_BUILD_ENV='CGO_ENABLED=0 GOOS=linux GOARCH=amd64' VERSION=${SCHEDULER_VERSION}
echo "Build completed."