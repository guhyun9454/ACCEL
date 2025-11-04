#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---- 기본값 ----
DATA_ROOT="${DATA_ROOT:-data}"
USE_PRIDE="${USE_PRIDE:-true}"
KO_FLAG="${KO_FLAG:-false}"
PRIDE_CFG="${PRIDE_CFG:-method=paraphrase,k=3,seed=42}"
LAMBDA="${LAMBDA:-0.80}"
EXTRA_ARGS="${EXTRA_ARGS:---prompt_lang en}"

# 문자열형(공백 구분)도 받고, 배열형도 받기
: "${MODELS_STR:=${MODELS_STR:-${MODELS:-}}}"   # 문자열형
# 배열형이면 길이가 0보다 큼
MODELS_ARR=${MODELS_ARR:-}                     # 외부에서 배열을 못 넘길 때 대비
EVAL_NAMES_ARR=${EVAL_NAMES_ARR:-}

# ---- 출력 디렉토리 ----
OUT_BASE="routes_out"
OUTDIR="${OUT_BASE}/$( [[ "$USE_PRIDE" == "true" ]] && echo pride_on || echo pride_off )"
mkdir -p "$OUTDIR"

run_route () {
  local EVAL_NAME="$1" NUM_SHIFTS="$2"

  local SCRIPT="code/router_no_pride.py"
  local PRIDE_OPT=()
  if [[ "$USE_PRIDE" == "true" ]]; then
    SCRIPT="code/router_with_pride.py"
    PRIDE_OPT=(--pride "$PRIDE_CFG")
  fi

  local KO_OPT=()
  [[ "$KO_FLAG" == "true" ]] && KO_OPT=(--ko)

  # 모델 인자 구성: 배열 > 문자열 순서로 우선
  local MODEL_OPT=()
  if [[ -n "${MODELS_ARR:-}" ]]; then
    # 쉘 변수에 배열이 이미 선언돼 있다고 가정
    # shellcheck disable=SC2154
    MODEL_OPT=(--models "${MODELS[@]}")
  elif [[ -n "${MODELS_STR:-}" ]]; then
    # 공백 구분 문자열
    read -r -a MODELS_FROM_STR <<<"${MODELS_STR}"
    MODEL_OPT=(--models "${MODELS_FROM_STR[@]}")
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

# ===== 실행: 배열이 있으면 그걸 루프, 없으면 기본 두 개 =====
if [[ -n "${EVAL_NAMES_ARR:-}" ]]; then
  # shellcheck disable=SC2154
  for name in "${EVAL_NAMES[@]}"; do
    case "$name" in
      arc,*)  run_route "$name" 4 ;;
      csqa,*) run_route "$name" 5 ;;
      *)      run_route "$name" 4 ;;  # 기본 4지
    esac
  done
else
  run_route "arc,0" 4
  run_route "csqa,0" 5
fi

echo "[DONE] routes saved under: ${OUTDIR}/"
