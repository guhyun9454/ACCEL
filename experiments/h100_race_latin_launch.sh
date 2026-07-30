#!/usr/bin/env bash
# Launch the clean 15-model RACE-H Latin 3-seed sweep on physical GPUs 4 and 5.
set -euo pipefail

REPO="${ACCEL_REPO:-/dataset/disc/workspace/ACCEL-race-clean}"
SIF="${ACCEL_SIF:-/dataset/singularity_images/pytorch271-cu128-devel.sif}"
HF_MODELS="${HF_MODELS:-/dataset/disc/cache/huggingface/models}"
RUN_ROOT="${ACCEL_RUN_ROOT:-/dataset/disc/run/accel-race-clean}"
LOG_ROOT="${ACCEL_LOG_ROOT:-/dataset/disc/logs/accel-race-clean}"
RESULT_TAG="${ACCEL_RESULT_TAG:-race_latin_single_options_3seed_h100_0730}"
MODEL_RUNNER="$REPO/experiments/h100_race_latin_model.sh"

models=(
    allenai/Olmo-3-7B-Instruct
    google/gemma-3-4b-it
    meta-llama/Llama-3.1-8B
    meta-llama/Llama-3.1-8B-Instruct
    meta-llama/Llama-3.2-3B
    meta-llama/Llama-3.2-3B-Instruct
    microsoft/Phi-4-mini-instruct
    microsoft/Phi-3-mini-4k-instruct
    Qwen/Qwen3-4B-Instruct-2507
    Qwen/Qwen2.5-7B
    Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-3B-Instruct
    mistralai/Mistral-7B-Instruct-v0.3
    mistralai/Ministral-8B-Instruct-2410
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
)

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
chmod 750 "$MODEL_RUNNER"
rm -f "$RUN_ROOT"/worker*.status "$RUN_ROOT"/completed.worker_failures

missing=0
for model in "${models[@]}"; do
    cache_dir="$HF_MODELS/models--${model//\//--}"
    if ! find -L "$cache_dir/snapshots" -type f -name config.json -print -quit 2>/dev/null | grep -q .; then
        echo "NOT_READY model=$model cache=$cache_dir" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    echo "REFUSING: all 15 model snapshots must be complete before launch." >&2
    exit 74
fi

if pgrep -f 'eval_clm.py.*--result_tag r7k2_' >/dev/null; then
    echo "REFUSING: R7K2 is still running; use the approved release procedure first." >&2
    exit 76
fi

for gpu in 4 5; do
    while read -r app_pid; do
        [[ -z "$app_pid" ]] && continue
        app_args="$(ps -o args= -p "$app_pid")"
        if [[ "$app_args" != *"--result_tag $RESULT_TAG"* ]]; then
            echo "REFUSING: GPU$gpu has unrelated compute pid=$app_pid: $app_args" >&2
            exit 75
        fi
    done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
done

launch_worker() {
    local worker_id="$1"
    local gpu="$2"
    local log="$LOG_ROOT/worker${worker_id}-gpu${gpu}.log"
    local pidfile="$RUN_ROOT/worker${worker_id}-gpu${gpu}.pid"
    local statusfile="$RUN_ROOT/worker${worker_id}-gpu${gpu}.status"

    if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "REFUSING: worker $worker_id already runs as pid $(cat "$pidfile")." >&2
        return 77
    fi

    : > "$log"
    nohup bash -c '
        set +e
        worker_id="$1"
        gpu="$2"
        result_tag="$3"
        statusfile="$4"
        shift 4
        failures=0
        echo "WORKER_START worker=$worker_id gpu=$gpu tag=$result_tag time=$(date --iso-8601=seconds)"
        for model in "$@"; do
            echo "WORKER_MODEL_START worker=$worker_id gpu=$gpu model=$model time=$(date --iso-8601=seconds)"
            env CUDA_VISIBLE_DEVICES="$gpu" ACCEL_RESULT_TAG="$result_tag" \
                singularity exec --nv --bind /dataset:/dataset "$ACCEL_SIF" \
                bash "$MODEL_RUNNER" "$model"
            status=$?
            echo "WORKER_MODEL_END worker=$worker_id gpu=$gpu model=$model status=$status time=$(date --iso-8601=seconds)"
            if [[ "$status" -ne 0 ]]; then
                failures=$((failures + 1))
            fi
        done
        echo "WORKER_END worker=$worker_id gpu=$gpu failures=$failures time=$(date --iso-8601=seconds)"
        printf "%s\n" "$failures" > "$statusfile"
        exit "$failures"
    ' bash "$worker_id" "$gpu" "$RESULT_TAG" "$statusfile" "$@" \
        >> "$log" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "$pid" > "$pidfile"
    echo "STARTED worker=$worker_id gpu=$gpu pid=$pid log=$log models=$*"
}

export ACCEL_SIF="$SIF"
export MODEL_RUNNER

new_pids=()
cleanup_failed_launch() {
    if [[ "${#new_pids[@]}" -gt 0 ]]; then
        kill -TERM "${new_pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup_failed_launch ERR

# Three workers per H100. Each worker processes its static queue sequentially.
launch_worker 0 4 "${models[0]}" "${models[6]}" "${models[12]}"
new_pids+=("$(cat "$RUN_ROOT/worker0-gpu4.pid")")
launch_worker 1 4 "${models[1]}" "${models[7]}" "${models[13]}"
new_pids+=("$(cat "$RUN_ROOT/worker1-gpu4.pid")")
launch_worker 2 4 "${models[2]}" "${models[8]}" "${models[14]}"
new_pids+=("$(cat "$RUN_ROOT/worker2-gpu4.pid")")
launch_worker 3 5 "${models[3]}" "${models[9]}"
new_pids+=("$(cat "$RUN_ROOT/worker3-gpu5.pid")")
launch_worker 4 5 "${models[4]}" "${models[10]}"
new_pids+=("$(cat "$RUN_ROOT/worker4-gpu5.pid")")
launch_worker 5 5 "${models[5]}" "${models[11]}"
new_pids+=("$(cat "$RUN_ROOT/worker5-gpu5.pid")")

supervisor_log="$LOG_ROOT/supervisor.log"
nohup bash -c '
    set +e
    run_root="$1"
    log_root="$2"
    result_tag="$3"
    failures=0
    echo "SUPERVISOR_START tag=$result_tag time=$(date --iso-8601=seconds)"
    for worker_id in 0 1 2 3 4 5; do
        if [[ "$worker_id" -lt 3 ]]; then gpu=4; else gpu=5; fi
        statusfile="$run_root/worker${worker_id}-gpu${gpu}.status"
        pidfile="$run_root/worker${worker_id}-gpu${gpu}.pid"
        while [[ ! -s "$statusfile" ]]; do
            pid="$(cat "$pidfile")"
            if ! kill -0 "$pid" 2>/dev/null; then
                printf "255\n" > "$statusfile"
                break
            fi
            sleep 15
        done
        status="$(cat "$statusfile")"
        echo "SUPERVISOR_WORKER status=$status statusfile=$statusfile"
        if [[ "$status" -ne 0 ]]; then
            failures=$((failures + 1))
        fi
    done
    echo "SUPERVISOR_END tag=$result_tag worker_failures=$failures time=$(date --iso-8601=seconds)"
    printf "%s\n" "$failures" > "$run_root/completed.worker_failures"
' bash "$RUN_ROOT" "$LOG_ROOT" "$RESULT_TAG" \
    >> "$supervisor_log" 2>&1 < /dev/null &
supervisor_pid=$!
printf '%s\n' "$supervisor_pid" > "$RUN_ROOT/supervisor.pid"

sleep 5
for pidfile in "$RUN_ROOT"/worker*-gpu*.pid; do
    pid="$(cat "$pidfile")"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "FAILED_TO_STAY_RUNNING pid=$pid pidfile=$pidfile" >&2
        exit 78
    fi
done

trap - ERR
echo "LAUNCH_COMPLETE supervisor_pid=$supervisor_pid tag=$RESULT_TAG"
nvidia-smi -i 4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
