#!/usr/bin/env bash
# experiments/pride.bash
# Self-escalation routing for ARC(4지선다) + CSQA(5지선다)
# - USE_PRIDE=true 로 두면 PriDe 라우팅(router_with_pride.py), false면 router_no_pride.py
# - 필요시 MODELS를 공백구분 나열로 override 가능 (미지정 시 기본 리스트 사용)

set -euo pipefail

# ====== 환경 설정 ======
# source ~/.bashrc
# conda activate tensorflow_gpu   # <-- 당신 환경에 맞게 변경

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="data"
OUT_BASE="routes_out"
if [[ "$USE_PRIDE" == "true" ]]; then
  OUTDIR="${OUT_BASE}/pride_on"
else
  OUTDIR="${OUT_BASE}/pride_off"
fi
LAMBDA="0.80"                 # 평균 top-1 confidence 임계치
EXTRA_ARGS="--prompt_lang en" # 프롬프트 언어
KO_FLAG="false"               # 한국어 데이터면 true로 변경
USE_PRIDE="true"              # PriDe 라우팅 사용 여부
PRIDE_CFG="method=paraphrase,k=3,seed=42"

# 선택: 모델 리스트 override (공백 구분). 비우면 기본 리스트 사용
# MODELS='Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-3B-Instruct'
MODELS="${MODELS:-}"

# ====== 공통 함수 ======
run_route () {
  local EVAL_NAME="$1" NUM_SHIFTS="$2"

  local SCRIPT="code/router_no_pride.py"
  local PRIDE_OPT=()
  if [[ "$USE_PRIDE" == "true" ]]; then
    SCRIPT="code/router_with_pride.py"
    PRIDE_OPT=(--pride "$PRIDE_CFG")
  fi

  local KO_OPT=()
  if [[ "$KO_FLAG" == "true" ]]; then
    KO_OPT=(--ko)
  fi

  local MODEL_OPT=()
  if [[ -n "${MODELS}" ]]; then
    # 공백 분리된 MODELS를 --models 뒤에 그대로 전달
    MODEL_OPT=(--models ${MODELS})
  fi

  echo "[RUN] ${SCRIPT} | eval=${EVAL_NAME} | shifts=${NUM_SHIFTS} | pride=${USE_PRIDE}"
  python "${SCRIPT}"     --data_root "${DATA_ROOT}"     --eval_name "${EVAL_NAME}"     --num_shifts "${NUM_SHIFTS}"     --lambda "${LAMBDA}"     --root_out "${OUTDIR}"     --extra "${EXTRA_ARGS}"     "${KO_OPT[@]}"     "${PRIDE_OPT[@]}"     "${MODEL_OPT[@]}"
}

# ====== 실행: ARC(4), CSQA(5) ======
run_route "arc,0" 4
run_route "csqa,0" 5

echo "[DONE] routes saved under: ${OUTDIR}/"
