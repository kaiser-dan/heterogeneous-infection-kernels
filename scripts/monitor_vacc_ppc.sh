#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <remote_log_path> <sleep_seconds> <job_id> [job_id ...]" >&2
  exit 1
fi

remote_log_path="$1"
sleep_seconds="$2"
shift 2
job_ids_csv="$(IFS=,; echo "$*")"

while true; do
  printf '===== %s =====\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  ssh vacc "squeue -j ${job_ids_csv} -o \"%.18i %.9P %.25j %.8T %.10M %.9l %.6D %R\" || true"
  ssh vacc "tail -n 40 '${remote_log_path}' || true"
  printf '\n'
  sleep "${sleep_seconds}"
done
