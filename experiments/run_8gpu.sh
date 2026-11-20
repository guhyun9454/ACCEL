#!/bin/bash
# Run from moana login node as briankim0809.
# Loads a 70B-class model across 8x RTX 3090 (192GB total VRAM).
# Only moana-y[1-5] have 8 GPUs — y[6-7] have only 4.

NODE=moana-y1                                    # change to a free y-node (y1–y5 only)
MODEL="meta-llama/Llama-3.1-70B-Instruct"       # ~140GB bf16, fits in 8x24GB

REPO=/ceph_data/briankim0809/projects/LLM-MCQ-Bias
CONDA_SH=/ceph_data/briankim0809/anaconda3/etc/profile.d/conda.sh
CEPH_HF=/ceph_data/briankim0809/hf_cache

mkdir -p "$REPO/logs"

JOB_ID=$(sbatch --job-name=eval_70b \
    --nodelist="$NODE" \
    --gres=gpu:8 --cpus-per-gpu=8 --mem-per-gpu=32G \
    --time=6-0 --partition=batch_ce_ugrad \
    -o "$REPO/logs/eval_70b_%j.out" \
    --wrap="source $CONDA_SH && conda activate llm_clone && \
export HF_HOME=$CEPH_HF && \
export HUGGINGFACE_HUB_CACHE=$CEPH_HF && \
export TRANSFORMERS_CACHE=$CEPH_HF && \
cd $REPO && python code/load_model_simple.py --model $MODEL" \
    | awk '{print $4}')

echo "[$JOB_ID] eval_70b → $NODE (model: $MODEL)"
echo "Monitor: squeue -u briankim0809"
echo "         tail -f $REPO/logs/eval_70b_${JOB_ID}.out"
