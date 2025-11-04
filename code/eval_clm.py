#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import math
import gc
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig  # (옵션)
# 내부 유틸
from eval_clm_utils import (
    parse_arguments,
    prepare_eval,
)
from utils import (
    _orange, _blue, _purple,
    eval_all_samples,
    get_accuracy,
    get_bootstrap_accuracy_std,
    save_results,
    patch_open,
)

# ============ NVML (GPU 메모리 로깅) ============
_nvml_ok = False
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False

logger = logging.getLogger(__name__)

# ============ 덤프 버퍼 ============
_PRED_DUMP = []

def _softmax(vec):
    try:
        import numpy as np
        x = np.asarray(vec, dtype=float)
        x = x - np.max(x)
        e = np.exp(x)
        s = e.sum()
        if s <= 0:
            return [1.0 / len(x)] * len(x)
        return (e / s).tolist()
    except Exception:
        # 안전망
        if not vec:
            return []
        s = float(sum(max(1e-12, float(v)) for v in vec))
        return [float(v) / s for v in vec]

def dump_pred(qid, gold_idx, pred_idx, probs=None):
    """
    시각화/라우팅 호환을 위해 gold/pred (구버전), gold_idx/pred_idx(신버전) 모두 기록.
    probs가 logits로 들어오면 softmax해 확률로 변환.
    """
    rec = {
        "qid": qid,
        "gold": int(gold_idx),
        "pred": int(pred_idx),
        "gold_idx": int(gold_idx),
        "pred_idx": int(pred_idx),
    }
    if probs is not None:
        try:
            # 만약 logits라면 확률로 변환
            _p = list(probs)
            if any(abs(float(v)) > 1.0 for v in _p):  # 대충 logits 힌트
                _p = _softmax(_p)
            else:
                # 이미 확률 형태여도 합이 1이 아니면 정규화
                s = sum(float(v) for v in _p) or 1.0
                _p = [float(v) / s for v in _p]
            rec["probs"] = [float(x) for x in _p]
        except Exception:
            pass
    _PRED_DUMP.append(rec)

def flush_dump(path):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as w:
        for r in _PRED_DUMP:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    _PRED_DUMP.clear()

def logging_cuda_memory_usage():
    if not _nvml_ok:
        logger.info("******** Memory usage ******** (NVML unavailable)")
        return
    logger.info("******** Memory usage ********")
    try:
        n_gpus = pynvml.nvmlDeviceGetCount()
        for i in range(n_gpus):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            logger.info("GPU {}: {:.2f} GB / {:.2f} GB".format(i, m.used / 1024 ** 3, m.total / 1024 ** 3))
    except Exception:
        pass

# ============ 키 스키마 호환 유틸 ============
def _safe_get(rec, keys):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None

def _letter_to_idx(letter, toks):
    """
    letter('A','B','가','나','1','2'...) -> 인덱스.
    """
    try:
        s = str(letter).strip()
        if s in toks:
            return toks.index(s)
        # 혹시 정수 문자열이면 직접 인덱싱 시도 (e.g., "0","1")
        if s.isdigit():
            i = int(s)
            if 0 <= i < len(toks):
                return i
    except Exception:
        pass
    return None

def _coerce_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _resolve_gold_pred_indices(r, tokens4, tokens5):
    """
    결과 dict r에서 gold/pred 인덱스를 최대한 회복.
    - gold 키 후보: gold_idx, label_idx, label, gold, ideal(문자), answer, label_letter
    - pred 키 후보: pred_idx, prediction_idx, pred, prediction, sampled(문자)
    - probs/logits가 있으면 argmax로 pred_idx 보정
    """
    # 선택지 수 추정
    L = _safe_get(r, ["num_choices", "n_choices"])
    if not isinstance(L, int):
        for cand in ["choices", "options", "option_list"]:
            if cand in r and isinstance(r[cand], (list, tuple)):
                L = len(r[cand]); break
    if not isinstance(L, int):
        # 마지막 안전망
        L = 4
    toks = tokens4 if L == 4 else (tokens5 if L == 5 else [str(k) for k in range(L)])

    # ---- GOLD ----
    gold_idx = _safe_get(r, ["gold_idx", "label_idx", "label", "gold"])
    gold_idx = _coerce_int(gold_idx, default=None)
    if gold_idx is None:
        gold_letter = _safe_get(r, ["gold_letter", "answer", "label_letter", "ideal"])
        if gold_letter is not None:
            gold_idx = _letter_to_idx(gold_letter, toks)
    if gold_idx is None:
        gold_idx = -1

    # ---- PRED ----
    pred_idx = _safe_get(r, ["pred_idx", "prediction_idx", "pred", "prediction"])
    pred_idx = _coerce_int(pred_idx, default=None)
    pred_letter = None
    if pred_idx is None:
        pred_letter = _safe_get(r, ["pred_letter", "prediction_letter", "sampled"])
        if pred_letter is not None:
            pred_idx = _letter_to_idx(pred_letter, toks)

    # ---- PROBS/LOGITS ----
    probs = _safe_get(r, ["probs", "prob", "choice_probs", "choice_prob"])
    logits = _safe_get(r, ["logits"])
    if isinstance(logits, (list, tuple)) and len(logits) > 0:
        probs_from_logits = _softmax(logits)
        probs = probs_from_logits
    # pred_idx가 아직 없다면 probs로 보정
    if pred_idx is None and isinstance(probs, (list, tuple)) and len(probs) > 0:
        try:
            import numpy as np
            pred_idx = int(np.argmax(probs))
        except Exception:
            pred_idx = max(range(len(probs)), key=lambda i: probs[i])

    if pred_idx is None:
        pred_idx = -1

    # probs가 있다면 리스트로 강제
    if isinstance(probs, (list, tuple)):
        probs = [float(x) for x in probs]
    else:
        probs = None

    return gold_idx, pred_idx, probs, toks

# ============ 메인 ============
def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    # argparse는 eval_clm_utils.parse_arguments()에서 처리
    args = parse_arguments()
    if len(getattr(args, "eval_names", [])) == 0:
        return

    os.makedirs('models', exist_ok=True)

    # 토크나이저 로드 (fast 실패시 slow fallback)
    try:
        toker = AutoTokenizer.from_pretrained(
            args.pretrained_model_path,
            use_fast=True,
            add_bos_token=False, add_eos_token=False,
            trust_remote_code=True,
            cache_dir='models',
        )
        logger.info("Tokenizer loaded with use_fast=True")
    except Exception as e_fast:
        logger.warning(f"Failed to load tokenizer with use_fast=True: {e_fast}. Retrying with use_fast=False.")
        try:
            toker = AutoTokenizer.from_pretrained(
                args.pretrained_model_path,
                use_fast=False,
                add_bos_token=False, add_eos_token=False,
                trust_remote_code=True,
                cache_dir='models',
            )
            logger.info("Tokenizer loaded with use_fast=False")
        except Exception as e_slow:
            logger.exception(
                f"Failed to load tokenizer (use_fast=True/False) for {args.pretrained_model_path}: {e_slow}"
            )
            return

    # 모델 로드
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            device_map='auto',
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            cache_dir='models',
        )
    except Exception as e_model:
        logger.exception(f"Failed to load model for {args.pretrained_model_path}: {e_model}")
        return
    logging_cuda_memory_usage()

    # 옵션 라벨(토큰) 파싱: 덤프 시 gold/pred 인덱스 해석용
    tokens4 = getattr(args, "option_ids4", None)
    tokens5 = getattr(args, "option_ids5", None)
    tokens4 = tokens4.split(",") if isinstance(tokens4, str) and tokens4 else ["A","B","C","D"]
    tokens5 = tokens5.split(",") if isinstance(tokens5, str) and tokens5 else ["A","B","C","D","E"]

    save_preds_path = getattr(args, "save_preds", None)

    for eval_name in args.eval_names[::1]:
        (
            subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn
        ) = prepare_eval(args, eval_name)

        for subject in subjects[::1]:
            cache_path = f'{args.save_path}/{subject}.jsonl'
            save_preds_on = bool(save_preds_path)

            # 결과 파일 존재 & save_preds 미설정 → 스킵
            if os.path.exists(cache_path) and not save_preds_on:
                logger.info(f"Results already exist (and --save_preds not set): {cache_path} — skipping")
                continue
            if os.path.exists(cache_path) and save_preds_on:
                logger.info(f"Results exist but --save_preds is set — re-running to emit JSONL: {save_preds_path}")

            logger.info(_blue(f"Preparing: {subject}"))
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = prepare_eval_fn(model, toker, few_shot_samples)

            logger.info(_blue(f"Run started: {subject}"))
            results = eval_all_samples(
                eval_fn, eval_samples,
                name=f'{args.task},{args.num_few_shot},{args.setting},{subject}',
                threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
            )
            gc.collect()
            torch.cuda.empty_cache()

            # ===== JSONL 덤프: qid/gold_idx/pred_idx/probs =====
            for i, r in enumerate(results):
                qid = r.get("id", r.get("qid", f"{subject}:{i}"))

                gold_idx, pred_idx, probs, toks = _resolve_gold_pred_indices(
                    r, tokens4=tokens4, tokens5=tokens5
                )
                dump_pred(qid, gold_idx, pred_idx, probs)

            # subject 단위로 JSONL append 저장
            flush_dump(save_preds_path)

            # ===== 리포트/저장 (기존 로직 유지) =====
            metrics = None
            if args.setting not in ['perm', 'cyclic'] and len(results) > 0:
                metrics = {'type': 'metric', 'data': {}}
                metrics['data']['accuracy'] = get_accuracy(results)
                metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)
                logger.info("Final report:")
                for key, value in metrics['data'].items():
                    logger.info(f"{key}: {value}")

            logger.info(_orange(f"Run completed: {subject}"))
            save_results(f'{args.save_path}/{subject}.jsonl', results, metrics)
            logger.info(f"Results saved: {subject}")
            logging_cuda_memory_usage()

if __name__ == "__main__":
    main()
