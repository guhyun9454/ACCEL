#!/bin/bash
# y1 8GPU 전체 선점 — 모델 다운로드 후 바로 8GPU 테스트
REPO=/ceph_data/briankim0809/projects/LLM-MCQ-Bias
. /ceph_data/briankim0809/anaconda3/etc/profile.d/conda.sh
conda activate llm_clone
export HF_HOME=/ceph_data/briankim0809/hf_cache
export HUGGINGFACE_HUB_CACHE=/ceph_data/briankim0809/hf_cache
export TRANSFORMERS_CACHE=/ceph_data/briankim0809/hf_cache
export HF_TOKEN=$(cat /ceph_data/briankim0809/hf_cache/token 2>/dev/null)

echo "=== [$(date)] 다운로드 시작 ==="

# 병렬 다운로드
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct &
huggingface-cli download Qwen/Qwen2.5-72B-Instruct &
huggingface-cli download Qwen/Qwen2.5-32B-Instruct &
huggingface-cli download google/gemma-4-31B-it &
huggingface-cli download Qwen/Qwen3.6-27B &
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct &
wait

echo "=== [$(date)] 다운로드 완료 — 8GPU 테스트 시작 ==="
cd "$REPO"
python code/load_model_simple.py --model meta-llama/Llama-3.1-70B-Instruct
