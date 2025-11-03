#!/usr/bin/env bash
# experiments/pride.bash
set -euo pipefail

# ====== 환경 설정 ======
# source ~/.bashrc
# conda activate tensorflow_gpu   # 환경에 맞게

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---- 기본값(환경변수로 덮어쓰기 가능) ----
DATA_ROOT="${DATA_ROOT:-data}"
USE_PRIDE="${USE_PRIDE:-true}"            # true/false
KO_FLAG="${KO_FLAG:-false}"               # true/false
PRIDE_CFG="${PRIDE_CFG:-method=paraphrase,k=3,seed=42}"
LAMBDA="${LAMBDA:-0.80}"
EXTRA_ARGS="${EXTRA_ARGS:---prompt_lang en}"
MODELS="${MODELS:-}"                      # 공백 구분 모델 목록 or 빈값(기본 리스트 사용)

# ---- 출력 디렉토리 분기 ----
OUT_BASE="routes_out"
if [[ "$USE_PRIDE" == "true" ]]; then
  OUTDIR="${OUT_BASE}/pride_on"
else
  OUTDIR="${OUT_BASE}/pride_off"
fi
mkdir -p "$OUTDIR"

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
    MODEL_OPT=(--models ${MODELS})
  fi

  echo "[RUN] ${SCRIPT} | eval=${EVAL_NAME} | shifts=${NUM_SHIFTS} | pride=${USE_PRIDE}"
  python "${SCRIPT}" \
    --data_root "${DATA_ROOT}" \
    --eval_name "${EVAL_NAME}" \
    --num_shifts "${NUM_SHIFTS}" \
    --lambda "${LAMBDA}" \
    --root_out "${OUTDIR}" \
    --extra "${EXTRA_ARGS}" \
    "${KO_OPT[@]}" \
    "${PRIDE_OPT[@]}" \
    "${MODEL_OPT[@]}"
}

# ====== 실행: ARC(4), CSQA(5) ======
run_route "arc,0" 4
run_route "csqa,0" 5

echo "[DONE] routes saved under: ${OUTDIR}/"
