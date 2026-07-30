#!/usr/bin/env bash
# Run the clean RACE-H Latin 3-seed protocol for one model on one visible GPU.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 <hf-model-id>" >&2
    exit 2
fi

MODEL="$1"
REPO="${ACCEL_REPO:-/dataset/disc/workspace/ACCEL-race-clean}"
VENV="${ACCEL_VENV:-/dataset/disc/envs/accel}"
HF_CACHE="${HF_CACHE:-/dataset/disc/cache/huggingface}"
RESULT_TAG="${ACCEL_RESULT_TAG:-race_latin_single_options_3seed_h100_0730}"

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/models"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/models"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/dataset/disc/cache/matplotlib}"
export PYTHONPATH="$REPO/code:$REPO/streamlit_app${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MPLCONFIGDIR"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO/code"

echo "MODEL_RUN_START model=$MODEL tag=$RESULT_TAG time=$(date --iso-8601=seconds)"
python eval_clm.py \
    --pretrained_model_path "$MODEL" \
    --cache_dir "$HF_CACHE/models" \
    --eval_names race,0,full \
    --option_id_set ABCD \
    --force \
    --pride_mix \
    --skip_full \
    --n_runs 3 \
    --empirical_pride \
    --empirical_residual_model empirical \
    --plot_empirical_prefix_fractions 2 \
    --plot_pride_ours_fractions 0,0.5,1,2,5,10,20,40,80,100 \
    --empirical_sweep_mode percentile \
    --empirical_stage_schedule flat \
    --empirical_transition_mode latin \
    --result_tag "$RESULT_TAG"
echo "MODEL_RUN_END model=$MODEL status=0 time=$(date --iso-8601=seconds)"
