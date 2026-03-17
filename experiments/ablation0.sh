#!/bin/bash
#SBATCH -J token-exp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v9
#SBATCH -t 1-0 
#SBATCH -o logs/%A.out

source ~/.bashrc

DATA_ROOT="../LLM-MCQ-Bias_data"

EVAL_NAMES=("mmlu,0,cyclic_swap_text" "arc,0,cyclic_swap_text" "csqa,0,cyclic_swap_text") 

MODELS=(
  "mistralai/Ministral-8B-Instruct-2410"
)

for MODEL in "${MODELS[@]}"; do
  echo "Running evaluation for model: $MODEL"
  python ../code/eval_clm.py \
    --pretrained_model_path "$MODEL" \
    --eval_names "${EVAL_NAMES[@]}" \
    --data_root ../LLM-MCQ-Bias_data/LLM-MCQ-Bias_data
done
