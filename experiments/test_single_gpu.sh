#!/bin/bash
# Run from moana login node as briankim0809.
# Single-GPU smoke test for load_model_simple.py.
# Edit NODE and MODEL before running.

NODE=moana-y1                       # change to a free y-node
MODEL="Qwen/Qwen2.5-7B-Instruct"   # small model to validate loading code

REPO=/ceph_data/briankim0809/projects/LLM-MCQ-Bias
CONDA_SH=/ceph_data/briankim0809/anaconda3/etc/profile.d/conda.sh
CEPH_HF=/ceph_data/briankim0809/hf_cache

mkdir -p "$REPO/logs"

JOB_ID=$(sbatch --job-name=test_load \
    --nodelist="$NODE" \
    --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G \
    --time=6-0 --partition=batch_ce_ugrad \
    -o "$REPO/logs/test_load_%j.out" \
    --wrap="source $CONDA_SH && conda activate llm_clone && \
export HF_HOME=$CEPH_HF && \
export HUGGINGFACE_HUB_CACHE=$CEPH_HF && \
export TRANSFORMERS_CACHE=$CEPH_HF && \
cd $REPO && python code/load_model_simple.py --model $MODEL" \
    | awk '{print $4}')

echo "[$JOB_ID] test_load → $NODE (model: $MODEL)"
echo "Monitor: tail -f $REPO/logs/test_load_${JOB_ID}.out"
