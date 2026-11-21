#!/bin/bash
# Usage: bash _run_download.sh <HF_MODEL_ID>
MODEL=$1
. /ceph_data/briankim0809/anaconda3/etc/profile.d/conda.sh
conda activate llm_clone
export HF_HOME=/ceph_data/briankim0809/hf_cache
export HUGGINGFACE_HUB_CACHE=/ceph_data/briankim0809/hf_cache
export TRANSFORMERS_CACHE=/ceph_data/briankim0809/hf_cache
export HF_TOKEN=$(cat /ceph_data/briankim0809/hf_cache/token 2>/dev/null)
echo "Downloading $MODEL ..."
# Skip duplicate non-HF weight formats (Mistral consolidated*, Llama original/*.pth); transformers never reads them.
huggingface-cli download "$MODEL" --exclude "consolidated*" "original/*" "*.pth"
echo "Done: $MODEL"
