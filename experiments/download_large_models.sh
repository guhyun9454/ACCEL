#!/bin/bash
# Run from moana login node as briankim0809.
# Submits 5 parallel download jobs (one per model, one per y-node).
# HF token must already be configured: huggingface-cli login
# Downloads skip duplicate non-HF weight formats (consolidated*, original/*.pth); transformers never reads them.

REPO=/ceph_data/briankim0809/projects/LLM-MCQ-Bias
CONDA_SH=/ceph_data/briankim0809/anaconda3/etc/profile.d/conda.sh
CEPH_HF=/ceph_data/briankim0809/hf_cache

mkdir -p "$REPO/logs"

NODES=(y1 y2 y3 y4 y5)
MODELS=(
    "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "google/gemma-3-27b-it"
)
JNAMES=(prep_llama31 prep_llama33 prep_qwen72 prep_qwen32 prep_gemma27)

for i in 0 1 2 3 4; do
    NODE="moana-${NODES[$i]}"
    MODEL="${MODELS[$i]}"
    JNAME="${JNAMES[$i]}"

    JOB_ID=$(sbatch --job-name="$JNAME" \
        --nodelist="$NODE" \
        --gres=gpu:1 --cpus-per-gpu=4 --mem-per-gpu=16G \
        --time=6-0 --partition=batch_ce_ugrad \
        -o "$REPO/logs/${JNAME}_%j.out" \
        --wrap="source $CONDA_SH && conda activate llm_clone && \
export HF_HOME=$CEPH_HF && \
export HUGGINGFACE_HUB_CACHE=$CEPH_HF && \
export TRANSFORMERS_CACHE=$CEPH_HF && \
huggingface-cli download $MODEL --exclude 'consolidated*' 'original/*' '*.pth'" \
        | awk '{print $4}')

    echo "[$JOB_ID] $JNAME → $NODE : $MODEL"
done

echo ""
echo "Monitor: squeue -u briankim0809"
