#!/usr/bin/env bash

MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "google/gemma-3-1b-it"
)

TASKS=(
  "arc"
  "csqa"
)

OPTSETS_ARC=(abcd ABCD 1234)
OPTSETS_CSQA=(abcde ABCDE 12345)
FEWSHOT="0"

for task in "${TASKS[@]}"; do
  if [[ "$task" == "arc" ]]; then
    OPTSETS=("${OPTSETS_ARC[@]}")
  else
    OPTSETS=("${OPTSETS_CSQA[@]}")
  fi

  for model in "${MODELS[@]}"; do
    for rp in $(seq 5 5 100); do
      ratio=$(awk -v v="$rp" 'BEGIN{printf "%.2f", v/100}')
      echo -e "\033[34m[RUN] task=$task model=$model ratio=$ratio few_shot=$FEWSHOT\033[0m"
      python analyze_token_preference.py \
        --pretrained_model_path "$model" \
        --task "$task" \
        --num_few_shot "$FEWSHOT" \
        --ratio_prefix_samples "$ratio" \
        --option_id_sets "${OPTSETS[@]}" \
        --cache_dir "$../models"
    done
  done
done

echo "All runs completed."


