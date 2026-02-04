#!/usr/bin/env bash
resourceUtilization() {
	# Collects resource utilization for display in the UI
	/opt/ucloud/ucmetrics viz &> /dev/null
}

resourceUtilization &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
exec /work/.script-generated-0.sh &> /work/stdout-$UCLOUD_RANK.log
