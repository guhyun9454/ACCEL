#!/usr/bin/env bash
# Restore the H100 R7K2 profile after the clean RACE sweep reaches a terminal state.
set -u

CFG="${H100_SSH_CONFIG:-/home/shared/.h100/ssh_config}"
NODE="${H100_NODE:-h100-node1}"
REMOTE_RUN_ROOT="${ACCEL_RUN_ROOT:-/dataset/disc/run/accel-race-clean}"
POLL_SECONDS="${POLL_SECONDS:-300}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-65}"
HOLD_SCRIPT="${H100_HOLD_SCRIPT:-/home/shared/skills/h100/hold.sh}"
GPUSTAT_SCRIPT="${H100_GPUSTAT_SCRIPT:-/home/shared/skills/h100/gpustat.sh}"

echo "RESTORE_MONITOR_START time=$(date --iso-8601=seconds) node=$NODE run_root=$REMOTE_RUN_ROOT"

while true; do
    status="$(
        ssh -o BatchMode=yes -F "$CFG" "$NODE" \
            "if test -s '$REMOTE_RUN_ROOT/completed.worker_failures'; then printf 'COMPLETE '; cat '$REMOTE_RUN_ROOT/completed.worker_failures'; else echo RUNNING; fi" \
            2>&1
    )"
    ssh_status=$?
    echo "RESTORE_MONITOR_POLL time=$(date --iso-8601=seconds) ssh_status=$ssh_status status=$status"

    if [[ "$ssh_status" -eq 0 && "$status" == COMPLETE* ]]; then
        worker_failures="${status#COMPLETE }"
        echo "RACE_SWEEP_TERMINAL time=$(date --iso-8601=seconds) worker_failures=$worker_failures"
        sleep "$COOLDOWN_SECONDS"

        while true; do
            echo "R7K2_RESTORE_ATTEMPT time=$(date --iso-8601=seconds)"
            if "$HOLD_SCRIPT"; then
                echo "R7K2_RESTORE_STARTED time=$(date --iso-8601=seconds)"
                break
            fi
            echo "R7K2_RESTORE_RETRY time=$(date --iso-8601=seconds)"
            sleep 120
        done

        sleep "$COOLDOWN_SECONDS"
        "$GPUSTAT_SCRIPT" "$NODE" || true
        echo "RESTORE_MONITOR_END time=$(date --iso-8601=seconds) worker_failures=$worker_failures"
        exit 0
    fi

    sleep "$POLL_SECONDS"
done
