#!/bin/bash
# CGES companion-benchmark job (NeurIPS #32713 rebuttal, g9SK-W3 follow-up).
#
# Runs the canonical empirical-residual pipeline (base + PriDe + full-cyclic
# caches + ACCEL curves) on one model for one of the new benchmarks:
#   obqa     OpenBookQA test, 500 items, 4-option — homogeneous / ARC-like
#   medmcqa  MedMCQA validation, 4-option, 21 subjects — heterogeneous
# The <task>_full per-view caches these produce feed the offline CGES-vs-ours
# routing simulation (p11_cges2.py).
#
# Build the data once before the first submit (needs the `datasets` lib):
#   python code/data_obqa/process.py
#   python code/data_medmcqa/process.py
#
# Seraph (ariel jihye4118 defaults; pin the v-node explicitly):
#   sbatch --job-name=<task>-<name> -w ariel-v6 --gres=gpu:1 --cpus-per-gpu=8 \
#          --mem-per-gpu=32G --time=6-0 --partition=batch_ugrad -o logs/%j.out \
#          experiments/cges_bench_job.sh <task> <hf-model-id>
# Other accounts/clusters: override ACCEL_REPO / ACCEL_CONDA_SH /
# ACCEL_CONDA_ENV / HF_HUB_CACHE / HF_HOME via the environment.
set -euo pipefail

TASK="$1"
MODEL="$2"
case "$TASK" in
  obqa|medmcqa) ;;
  *) echo "unknown task: $TASK (expected obqa|medmcqa)" >&2; exit 2 ;;
esac

REPO="${ACCEL_REPO:-/ceph_data/jihye4118/LLM-MCQ-Bias-cges}"
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
  --eval_names "$TASK,0,full" \
  --option_id_set ABCD \
  --force --wandb --pride_mix --skip_full \
  --n_runs 3 \
  --wandb_project "${ACCEL_WANDB_PROJECT:-3_${TASK}_emp_latin}" \
  --empirical_pride \
  --empirical_residual_model empirical \
  --plot_empirical_prefix_fractions 2 \
  --empirical_sweep_mode percentile \
  --empirical_stage_schedule flat \
  --result_tag "${ACCEL_RESULT_TAG:-empirical_latin_flat_0801}" \
  --empirical_transition_mode latin
