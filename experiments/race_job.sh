#!/bin/bash
# RACE-H sweep job (NeurIPS #32713 rebuttal P1-3).
#
# Build the data once before the first submit:
#   python code/data_race/process.py --subset high
#
# Submit one model per GPU. The submit filter rejects --exclude, so the v-node
# must be pinned explicitly with -w or the job is refused with
# "SUBMISSION REJECTED: GPU type 'high_perf' is REQUIRED".
#
#   sbatch --job-name=race-<name> -w ariel-v6 --gres=gpu:1 --cpus-per-gpu=8 \
#          --mem-per-gpu=32G --time=6-0 --partition=batch_ugrad -o logs/%j.out \
#          experiments/race_job.sh <hf-model-id>
#
# Override the account-specific paths via the environment; the defaults are the
# jihye4118 layout used for the 2026-07-26 sweep.
set -euo pipefail

MODEL="$1"
REPO="${ACCEL_REPO:-/ceph_data/jihye4118/LLM-MCQ-Bias}"
CONDA_SH="${ACCEL_CONDA_SH:-/ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${ACCEL_CONDA_ENV:-llm}"
# Models live in the repo's code/models (HF hub layout), NOT under HF_HOME —
# point HF_HUB_CACHE there or every job re-downloads ~14GB.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO/code/models}"
export HF_HOME="${HF_HOME:-/ceph_data/jihye4118/hf_cache}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO/code"

python eval_clm.py \
  --pretrained_model_path "$MODEL" \
  --eval_names race,0,full \
  --option_id_set ABCD \
  --force --wandb --pride_mix --skip_full \
  --n_runs 3 \
  --wandb_project "${ACCEL_WANDB_PROJECT:-3_race_emp_latin_real}" \
  --empirical_pride \
  --empirical_residual_model empirical \
  --plot_empirical_prefix_fractions 2 \
  --empirical_sweep_mode percentile \
  --empirical_stage_schedule flat \
  --result_tag empirical_latin_flat_0502 \
  --empirical_transition_mode latin
