#!/bin/bash
# Usage: bash _run_model.sh <HF_MODEL_ID>
MODEL=$1
REPO=/ceph_data/briankim0809/projects/LLM-MCQ-Bias
. /ceph_data/briankim0809/anaconda3/etc/profile.d/conda.sh
conda activate llm_clone
export HF_HOME=/ceph_data/briankim0809/hf_cache
export HUGGINGFACE_HUB_CACHE=/ceph_data/briankim0809/hf_cache
export TRANSFORMERS_CACHE=/ceph_data/briankim0809/hf_cache
export HF_TOKEN=$(cat /ceph_data/briankim0809/hf_cache/token 2>/dev/null)
cd "$REPO"
python code/load_model_simple.py --model "$MODEL"
