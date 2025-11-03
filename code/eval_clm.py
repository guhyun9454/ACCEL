#!/usr/bin/env python3
import os
import sys
import json
import logging
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig  # (미사용 가능)

import gc

import pynvml
pynvml.nvmlInit()

logger = logging.getLogger(__name__)

# ===== JSONL 덤프 유틸 (변형/라우팅/시각화용) =====
_PRED_DUMP = []
def dump_pred(qid, gold_idx, pred_idx, probs=None):
    rec = {"qid": qid, "gold": int(gold_idx), "pred": int(pred_idx)}
    if probs is not None:
        try:
            rec["probs"] = [float(x) for x in probs]
        except Exception:
            # probs가 텐서/넘파이일 수도 있으니 best-effort 변환
            try:
                rec["probs"] = [float(x) for x in list(probs)]
            except Exception:
                pass
    _PRED_DUMP.append(rec)

def flush_dump(path):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # subject 단위 append 저장 (라우터가 여러 번 호출 가능)
    with open(path, "a", encoding="utf-8") as w:
        for r in _PRED_DUMP:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    _PRED_DUMP.clear()

def logging_cuda_memory_usage():
    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info("GPU {}: {:.2f} GB / {:.2f} GB".format(i, meminfo.used / 1024 ** 3, meminfo.total / 1024 ** 3))


def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    # ── 중요: 여기서 --save_preds / --permute_shift / --pride 를 받아야 함
    # (아래 2)번에서 eval_clm_utils.py에 실제 argparse 추가 필요)
    args = parse_arguments()
    if len(args.eval_names) == 0:
        exit()

    os.makedirs('models', exist_ok=True)

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

    for eval_name in args.eval_names[::1]:
        (
            subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn
        ) = prepare_eval(args, eval_name)
        for subject in subjects[::1]:
            # 결과 파일이 있으면 스킵 (원하면 아래 두 줄을 주석 처리해서 항상 재실행 가능)
            if os.path.exists(f'{args.save_path}/{subject}.jsonl'):
                logger.info(f"Results already exist: {args.save_path}/{subject}.jsonl")
                continue

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

            # ===== JSONL 덤프: qid/gold/pred/(probs) =====
            def _safe_get_idx(rec, keys):
                for k in keys:
                    if k in rec and rec[k] is not None:
                        return rec[k]
                return None

            def _letter_to_idx(letter, toks):
                try:
                    return toks.index(str(letter))
                except Exception:
                    return None

            for i, r in enumerate(results):
                qid = r.get("id", r.get("qid", f"{subject}:{i}"))

                # 선택지 개수 추정
                L = _safe_get_idx(r, ["num_choices", "n_choices"])
                if not isinstance(L, int):
                    for cand in ["choices", "options", "option_list"]:
                        if cand in r and isinstance(r[cand], (list, tuple)):
                            L = len(r[cand]); break
                if not isinstance(L, int):
                    L = 4
                toks = tokens4 if L == 4 else (tokens5 if L == 5 else [str(k) for k in range(L)])

                # gold / pred 인덱스 해석 (여러 키명 호환)
                gold_idx = _safe_get_idx(r, ["gold_idx","label_idx","label","gold"])
                if gold_idx is None:
                    gold_letter = _safe_get_idx(r, ["gold_letter","answer","label_letter"])
                    if gold_letter is not None:
                        gold_idx = _letter_to_idx(gold_letter, toks)
                if gold_idx is None: gold_idx = -1

                pred_idx = _safe_get_idx(r, ["pred_idx","prediction_idx","pred","prediction"])
                if pred_idx is None:
                    pred_letter = _safe_get_idx(r, ["pred_letter","prediction_letter"])
                    if pred_letter is not None:
                        pred_idx = _letter_to_idx(pred_letter, toks)
                if pred_idx is None: pred_idx = -1

                probs = _safe_get_idx(r, ["probs","prob","choice_probs","choice_prob","logits"])
                if not isinstance(probs, (list, tuple)):
                    probs = None

                dump_pred(qid, gold_idx, pred_idx, probs)

            # subject 단위로 JSONL에 append 저장
            flush_dump(getattr(args, "save_preds", None))

            # ===== 기본 리포트/저장(기존 로직 유지) =====
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
