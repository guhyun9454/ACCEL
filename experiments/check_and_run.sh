#!/bin/bash
# Run from moana login node as briankim0809.
# Checks download/test job status and submits 8-GPU job when ready.
# Usage: bash experiments/check_and_run.sh

REPO=/ceph_data/briankim0809/projects/LLM-MCQ-Bias
LOGS=$REPO/logs
CONDA_SH=/ceph_data/briankim0809/anaconda3/etc/profile.d/conda.sh
CEPH_HF=/ceph_data/briankim0809/hf_cache
SETUP="source $REPO/setup_env.sh"

echo "=== $(date) ==="
echo ""

# ── 1. 전체 잡 상태 ──────────────────────────────────────────────────
echo "[squeue]"
squeue -u briankim0809 -o "%.10i %.15j %.8T %.10M %R"
echo ""

# ── 2. test_load 결과 확인 ───────────────────────────────────────────
TEST_LOG=$(ls -t $LOGS/test_load_*.out 2>/dev/null | head -1)
if [ -n "$TEST_LOG" ]; then
    echo "[test_load] $TEST_LOG"
    grep -E "GPU|Output|Error|Traceback|done" "$TEST_LOG" 2>/dev/null | tail -10
    echo ""
fi

# ── 3. 다운로드 잡 진행 상황 ─────────────────────────────────────────
echo "[download progress]"
for f in $LOGS/prep_*.out; do
    [ -f "$f" ] || continue
    NAME=$(basename "$f" | sed 's/_.*//')
    LAST=$(tail -3 "$f" 2>/dev/null | tr '\n' ' ')
    echo "  $NAME: $LAST"
done
echo ""

# ── 4. 다운로드 완료 여부 체크 ────────────────────────────────────────
echo "[hf_cache contents]"
ls $CEPH_HF/models--*/snapshots 2>/dev/null | grep -v "^$" \
    | sed 's|.*/models--||;s|/snapshots||' | sort -u
echo ""

# ── 5. 8GPU 잡 자동 제출 조건 확인 ──────────────────────────────────
# 조건: test_load 완료(RUNNING/COMPLETED 아님) + y1 idle + Llama-3.1-70B 캐시 존재
TEST_RUNNING=$(squeue -u briankim0809 -h -o "%j" | grep -c "test_load")
Y1_FREE=$(sinfo -N -h -o "%N %T" | grep "moana-y1" | grep -c "idle")
LLAMA31_CACHED=$(ls $CEPH_HF/models--meta-llama--Llama-3.1-70B-Instruct/snapshots 2>/dev/null | wc -l)
ALREADY_RUNNING=$(squeue -u briankim0809 -h -o "%j" | grep -c "eval_70b")

echo "[8GPU submit check]"
echo "  test_load still running: $TEST_RUNNING"
echo "  moana-y1 idle: $Y1_FREE"
echo "  Llama-3.1-70B cached: $LLAMA31_CACHED"
echo "  eval_70b already submitted: $ALREADY_RUNNING"

if [ "$TEST_RUNNING" -eq 0 ] && [ "$Y1_FREE" -eq 1 ] && [ "$LLAMA31_CACHED" -gt 0 ] && [ "$ALREADY_RUNNING" -eq 0 ]; then
    echo ""
    echo ">>> 조건 충족 — 8GPU 잡 제출합니다"
    JOB_8G=$(sbatch --job-name=eval_70b --nodelist=moana-y1 \
        --gres=gpu:8 --cpus-per-gpu=8 --mem-per-gpu=32G \
        --time=6-0 --partition=batch_ce_ugrad \
        -o $LOGS/eval_70b_%j.out \
        --wrap="$SETUP && cd $REPO && python code/load_model_simple.py --model meta-llama/Llama-3.1-70B-Instruct" \
        | awk '{print $4}')
    echo ">>> eval_70b submitted: $JOB_8G"
else
    echo "(아직 조건 미충족)"
fi
