#!/bin/bash
# Single-model RACE cause diagnosis with an isolated result tag.
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
TRANSITION_MODE="${2:-latin}"
SCOPE="${3:-test}"

REPO="${ACCEL_REPO:-/ceph_data/guhyun9454/g/ACCEL-race-cause}"
CONDA_SH="${ACCEL_CONDA_SH:-/nas2/data/guhyun9454/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${ACCEL_CONDA_ENV:-ai-action}"
HF_HOME="${HF_HOME:-/ceph_data/guhyun9454/hf}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
PROMPT_MODE="single_options"
if [[ "${ACCEL_REPEAT_OPTIONS:-0}" == "1" ]]; then
  PROMPT_MODE="repeat_options"
fi
RESULT_TAG="${ACCEL_RESULT_TAG:-race_cause_${TRANSITION_MODE}_${PROMPT_MODE}_${SCOPE}_0730}"
if [[ ! "$RESULT_TAG" =~ ^([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9])$ ]]; then
  echo "result tag must already be sanitized: $RESULT_TAG" >&2
  exit 2
fi

export HF_HOME
export HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO/.cache/matplotlib}"
export PYTHONPATH="$REPO/.deps${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MPLCONFIGDIR"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO/code"

args=(
  --pretrained_model_path "$MODEL"
  --eval_names race,0,full
  --option_id_set ABCD
  --cache_dir "$HF_HUB_CACHE"
  --force
  --pride_mix
  --skip_full
  --n_runs 1
  --empirical_pride
  --empirical_residual_model empirical
  --plot_empirical_prefix_fractions 2
  --plot_pride_ours_fractions 0,0.5,1,2,5,10,20,40,80,100
  --empirical_sweep_mode percentile
  --empirical_stage_schedule flat
  --empirical_transition_mode "$TRANSITION_MODE"
  --result_tag "$RESULT_TAG"
)

if [[ "$SCOPE" == "test" ]]; then
  args+=(--test)
elif [[ "$SCOPE" != "full" ]]; then
  echo "scope must be 'test' or 'full', got: $SCOPE" >&2
  exit 2
fi

if [[ "${ACCEL_REPEAT_OPTIONS:-0}" == "1" ]]; then
  args+=(--repeat_options)
fi

python eval_clm.py "${args[@]}"

MODEL_NAME="${MODEL##*/}"
analysis_dir="results_race/0s_${MODEL_NAME}/race_full_id-ABCD__${RESULT_TAG}/empirical_analysis"
mapfile -t trajectory_paths < <(
  find "$analysis_dir" -maxdepth 1 -type f -name '*alpha2_trajectories.jsonl' -print
)
if [[ "${#trajectory_paths[@]}" -ne 1 ]]; then
  echo "expected exactly one trajectory under $analysis_dir, found ${#trajectory_paths[@]}" >&2
  exit 3
fi
trajectory_path="${trajectory_paths[0]}"

report_path="${trajectory_path%_trajectories.jsonl}_cause_report.json"
python race_cause_report.py \
  "$trajectory_path" \
  --percentile 2 \
  --k 4 \
  --stage-schedule flat \
  --output "$report_path"
