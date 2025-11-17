# eval_clm.py
# -*- coding: utf-8 -*-

import os
import sys
import gc
import json
import copy
import logging
import argparse
import random
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

# ---- local utils (project-provided) ----
from utils import (
    _norm,
    shuffle_options_with_ids,
    move_answer,
    cycle_options,
    permute_options,
    _orange, _blue, _purple,
    eval_all_samples,
    get_accuracy,
    get_bootstrap_accuracy_std,
    save_results,
    patch_open,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Logging / CUDA env helpers
# ----------------------------------------------------

def setup_logging(to_stdout: bool = True, level: int = logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout if to_stdout else sys.stderr)]
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(process)d] %(levelname)s %(filename)s:%(lineno)d | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)
    hf_logging.set_verbosity_error()
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def log_cuda_env():
    logger.info("python: %s", sys.version.replace("\n", " "))
    logger.info("torch: %s; cuda available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.info("cuda device count: %d", n)
        for i in range(n):
            try:
                logger.info("cuda[%d] name: %s", i, torch.cuda.get_device_name(i))
            except Exception:
                pass
        logger.info("bf16 supported: %s", torch.cuda.is_bf16_supported())
    else:
        logger.info("Running on CPU")


# ----------------------------------------------------
# Alias / token bucket utils
# ----------------------------------------------------

def _build_id_groups(option_ids_header: List[str], option_id_set: Optional[str]
                     ) -> Tuple[List[List[str]], List[str], List[str]]:
    """
    header=['A','B','C','D'](ARC) / ['A','B','C','D','E'](CSQA)
    option_id_set 예: 'ABCDqw' -> 그룹: [A/q, B/w, C, D]
    return: id_groups, display_ids('A/q'...), primary_ids('A'...)
    """
    k = len(option_ids_header)
    if not option_id_set:
        id_groups = [[h] for h in option_ids_header]
    else:
        custom = list(str(option_id_set))
        if len(custom) < k:
            raise ValueError(f"option_id_set length must be >= {k}; got {len(custom)}: {option_id_set}")
        prim = custom[:k]
        extras = custom[k:]
        id_groups = [[p] for p in prim]
        for i, extra in enumerate(extras):
            id_groups[i % k].append(extra)
    display_ids = ["/".join(g) for g in id_groups]
    primary_ids = [g[0] for g in id_groups]
    return id_groups, display_ids, primary_ids


def _token_variants_for_label(toker, label: str) -> List[int]:
    """Collect candidate token ids for ': X' / ':X' / fallback to the label itself."""
    ids: List[Optional[int]] = []
    try:
        ids.append(toker(f": {label}").input_ids[-1])
    except Exception:
        ids.append(None)
    try:
        cand = toker(f":{label}").input_ids[-1]
        if cand != ids[-1]:
            ids.append(cand)
    except Exception:
        ids.append(None)
    # fallback: label itself
    if all(x is None for x in ids) and label:
        try:
            ids = [toker(label).input_ids[-1]]
        except Exception:
            ids = []
    return [x for x in ids if x is not None]


def _group_token_id_buckets(toker, id_groups: List[List[str]]):
    """
    Return per-option token-id buckets, flattened list and spans.
    """
    buckets = []
    for group in id_groups:
        token_ids = []
        for label in group:
            token_ids.extend(_token_variants_for_label(toker, label))
        if not token_ids:  # safety: never empty bucket
            token_ids = [toker(":").input_ids[-1]]
        # de-dup, keep order
        seen, uniq = set(), []
        for tid in token_ids:
            if tid not in seen:
                seen.add(tid)
                uniq.append(tid)
        buckets.append(uniq)

    flat, spans, cur = [], [], 0
    for b in buckets:
        s = cur
        flat.extend(b)
        cur += len(b)
        spans.append((s, cur))
    return buckets, flat, spans


# ----------------------------------------------------
# Argparse
# ----------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--eval_names", type=str, nargs='+', default=[],
                        help="예: arc,0,full / csqa,0,shuffle_both / mmlu,0,perm")
    parser.add_argument("--option_id_set", type=str, default=None,
                        help=">= #options 길이 허용. 예: ABCDqw (A/q, B/w)")
    parser.add_argument("--option_id_sets", type=str, nargs='+', default=None,
                        help='(선택) 두 세트를 비교할 때 사용. 예: \"ABCD abcd\"')
    parser.add_argument("--print_prompt_example", action="store_true",
                        help="옵션/별칭 적용된 프롬프트 예시 1건 로깅")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data",
                        help="(참고) 각 태스크는 data_<task> 폴더를 사용")
    parser.add_argument("--save_path", type=str, default=None,
                        help="(선택) 결과 저장 루트. 기본은 results_<task>/...")
    parser.add_argument("--model_name", type=str, default=None,
                        help="출력 폴더 이름. 기본값은 pretrained 경로의 마지막 토큰")
    parser.add_argument("--lowconf_threshold", type=float, default=0.0,
                        help="top 확률이 이 값보다 낮으면 low_confidence=True")
    parser.add_argument("--perm_reduce", type=str, default="mean",
                        choices=["mean", "max", "vote", "first"],
                        help="perm/cyclic에서 여러 변형의 확률을 합치는 방법")
    args = parser.parse_args()

    # validate eval_names entries
    for eval_name in args.eval_names:
        parts = eval_name.split(',')
        if len(parts) < 2:
            raise ValueError(f"--eval_names 형식 오류: {eval_name} (예: arc,0,full)")
        task = parts[0]
        if task not in ["mmlu", "arc", "csqa"]:
            raise ValueError(f"Unknown task: {task}")
        _ = int(parts[1])  # few-shot must be int
        setting = parts[2] if len(parts) > 2 and parts[2] else None
        if setting is not None and not (
            setting in ["noid", "perm", "cyclic", "shuffle_both", "full"] or
            (setting.startswith("move") and setting[-1] in ["a", "b", "c", "d"]) 
        ):
            raise ValueError(f"Unknown setting: {setting}")
    return args


# ----------------------------------------------------
# Data / prompt preparation
# ----------------------------------------------------

def prepare_eval(args, eval_name):
    parts = eval_name.split(',')
    task = parts[0]
    num_few_shot = int(parts[1])
    setting = parts[2] if len(parts) > 2 and parts[2] else None
    moved_answer = None
    if setting is not None and setting.startswith('move'):
        moved_answer = setting[-1].upper()

    # save path
    model_dir = _model_dir_name(args)
    save_path = args.save_path
    if not save_path:
        save_path = f"results_{task}/{num_few_shot}s_{model_dir}/{task}"
        if setting is not None:
            save_path += f"_{setting}"
        if getattr(args, 'option_id_set', None):
            save_path += f"_id-{args.option_id_set}"
    os.makedirs(save_path, exist_ok=True)

    # headers
    option_ids_header = list('ABCDE' if task == 'csqa' else 'ABCD')

    # alias groups
    id_groups, display_ids, primary_ids = _build_id_groups(option_ids_header, args.option_id_set)
    printed_ids = display_ids if setting == 'full' else primary_ids

    # subjects
    data_path = f"data_{task}"
    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(f"{data_path}/test") if f.endswith("_test.csv")
    ])

    # system message
    if 'mmlu' in task:
        sys_msg = 'The following are multiple choice questions about {}.'
    else:
        sys_msg = 'The following are multiple choice questions.'
    sys_msg += ' You should directly answer the question by choosing the correct option.'

    # prompt builder
    def create_user_prompt(question: str, options: List[str]) -> str:
        if setting == 'noid':
            user_prompt = (
                f"Question: {question.strip()}\nOptions:\n" +
                "\n".join([f"{answer}".strip() for answer in options]) +
                "\nAnswer:"
            )
        elif setting == 'shuffle_both':
            shuffled_ids, shuffled_options = shuffle_options_with_ids(printed_ids, options)
            user_prompt = (
                f"Question: {question.strip()}\nOptions:\n" +
                "\n".join([f"{oid}. {ans}".strip() for oid, ans in zip(shuffled_ids, shuffled_options)]) +
                "\nAnswer:"
            )
        else:
            user_prompt = (
                f"Question: {question.strip()}\nOptions:\n" +
                "\n".join([f"{oid}. {ans}".strip() for oid, ans in zip(printed_ids, options)]) +
                "\nAnswer:"
            )
        return user_prompt

    # few-shot strings
    def prepare_few_shot_samples(subject):
        df = pd.read_csv(
            f"{data_path}/dev/{subject}_dev.csv",
            names=("Question", *option_ids_header, "Answer"),
            dtype=str
        )
        if setting == 'noid':
            few = df.apply(
                lambda x: create_user_prompt(x["Question"], [x[e] for e in option_ids_header]) +
                          ' ' + str(x[x["Answer"]]),
                axis=1
            ).to_list()
        else:
            few = df.apply(
                lambda x: create_user_prompt(x["Question"], [x[e] for e in option_ids_header]) +
                          ' ' + printed_ids[option_ids_header.index(x["Answer"])],
                axis=1
            ).to_list()
        return few

    # eval samples
    def prepare_eval_samples(subject):
        df = pd.read_csv(
            open(f"{data_path}/test/{subject}_test.csv"),
            names=("Question", *option_ids_header, "Answer"),
            dtype=str
        )
        if moved_answer is not None:
            df = df.apply(lambda x: move_answer(x, moved_answer), axis=1)

        def sys_pair(msg, q, opts):
            return [msg.format(subject.replace('_', ' ')), create_user_prompt(q, opts)]

        inputs = []
        if setting == 'perm':
            # 각 permutation에 대해 라벨→원래보기 인덱스 매핑(perm_map)을 함께 저장
            for _, row in df.iterrows():
                orig_opts = [row[e] for e in option_ids_header]    # 원래 A,B,C,D 보기
                per_variants = []
                for po in permute_options(orig_opts):
                    perm_map = [orig_opts.index(po[j]) for j in range(len(orig_opts))]
                    per_variants.append([sys_pair(sys_msg, row["Question"], po), perm_map])
                inputs.append(per_variants)
        elif setting == 'cyclic':
            for _, row in df.iterrows():
                orig_opts = [row[e] for e in option_ids_header]
                per_variants = []
                for co in cycle_options(orig_opts):
                    perm_map = [orig_opts.index(co[j]) for j in range(len(orig_opts))]
                    per_variants.append([sys_pair(sys_msg, row["Question"], co), perm_map])
                inputs.append(per_variants)
        else:
            inputs = df.apply(
                lambda x: [sys_msg.format(subject.replace('_', ' ')),
                           create_user_prompt(x["Question"], [x[e] for e in option_ids_header])],
                axis=1
            ).to_list()

        options = df.apply(lambda x: [str(x[e]) for e in option_ids_header], axis=1).to_list()
        ideal_idx = df.apply(lambda x: option_ids_header.index(x["Answer"]), axis=1).to_list()
        ideals_disp = [printed_ids[i] for i in ideal_idx]  # for logging
        return list(zip(inputs, options, ideal_idx, ideals_disp))

    # eval_fn selector
    if setting == 'noid':
        prepare_eval_fn = partial(prepare_eval_fn_noid, num_few_shot=num_few_shot)
    else:
        prepare_eval_fn = partial(
            prepare_eval_fn_tokenpick,
            num_few_shot=num_few_shot,
            id_groups=id_groups,
            printed_ids=printed_ids,
            lowconf_threshold=float(getattr(args, 'lowconf_threshold', 0.0)),
            reduce_mode=str(getattr(args, 'perm_reduce', 'mean')),
        )

    return subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn, save_path


# ----------------------------------------------------
# Evaluation functions
# ----------------------------------------------------

def prepare_eval_fn_tokenpick(model, toker, few_shot_samples, num_few_shot,
                              id_groups: List[List[str]],
                              printed_ids: List[str],
                              lowconf_threshold: float,
                              reduce_mode: str = "mean"):
    """
    Token aggregation over alias buckets for each option.
    Supports base / shuffle_both / full / perm / cyclic.

    perm/cyclic의 경우:
      - 각 변형별 라벨(A/B/C/D) 확률을 "원래 보기 순서(정답 축)"로 역정렬(canonicalize)하여 모음
      - reduce_mode(mean/max/vote/first)로 합친 뒤 최종 pred/acc 계산
    """
    bpe_has_space_prefix = (toker(': A').input_ids[-1] != toker(':A').input_ids[-1])
    _, flat_ids, spans = _group_token_id_buckets(toker, id_groups)

    def _pick_for_single_text(input_text: str):
        input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids.to(model.device)
        input_ids = input_ids[..., -1536:]  # context trim
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[:, -1].view(-1)  # (vocab,)
        cand_logits = logits[flat_ids]
        cand_probs = F.softmax(cand_logits, dim=-1).detach().to(torch.float32).cpu().numpy()

        group_probs = []
        for s, e in spans:
            group_probs.append(float(np.sum(cand_probs[s:e])) if e > s else 0.0)
        group_probs = np.asarray(group_probs, dtype=np.float32)
        denom = float(group_probs.sum()) + 1e-10
        group_probs = group_probs / denom
        pred_idx = int(np.argmax(group_probs))
        conf = float(group_probs[pred_idx])
        return group_probs, pred_idx, conf

    def _reduce(arrs: List[np.ndarray]) -> np.ndarray:
        # arrs: list of [K] canonical probs (원래 보기 순서 축)
        if reduce_mode == "mean":
            return np.mean(arrs, axis=0)
        if reduce_mode == "max":
            return np.max(arrs, axis=0)
        if reduce_mode == "vote":
            votes = np.bincount([int(np.argmax(a)) for a in arrs], minlength=len(arrs[0])).astype(np.float32)
            return votes / (votes.sum() + 1e-10)
        # "first"
        return arrs[0]

    def eval_fn(sample, rng: random.Random):
        idx, (inp, options, ideal_idx, _ideal_disp) = sample
        K = len(options)

        # perm/cyclic: inp = [ [ [sys,eval], perm_map ], ... ]
        if isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], list) and len(inp[0]) == 2:
            canonical_lists = []
            shown_prompt = None
            for (sys_msg, eval_sample), perm_map in inp:
                t = sys_msg + '\n\n'
                if num_few_shot > 0:
                    for s in few_shot_samples[:num_few_shot]:
                        t += s + '\n\n'
                t += eval_sample
                if not bpe_has_space_prefix:
                    t += ' '
                if shown_prompt is None:
                    shown_prompt = t

                probs_label, _, _ = _pick_for_single_text(t)   # [K] (라벨 A/B/C/D 순서)
                # 라벨 → 원래 보기 순서로 역정렬
                canonical = np.zeros((K,), dtype=np.float32)
                for j, orig_idx in enumerate(perm_map):  # j: 라벨 위치, orig_idx: 원래 보기 위치
                    canonical[orig_idx] += probs_label[j]
                s = float(canonical.sum())
                if s > 0:
                    canonical /= s
                canonical_lists.append(canonical)

            agg = _reduce(canonical_lists)                 # [K] 원래 보기 순서 기준
            pred_idx = int(np.argmax(agg))
            conf = float(agg[pred_idx] / (agg.sum() + 1e-10))
            correct = (pred_idx == int(ideal_idx))
            low_conf = bool(conf < float(lowconf_threshold))
            return {
                'type': 'result',
                'data': {
                    'idx': idx,
                    'prompt': shown_prompt,
                    'options': options,
                    'agg_probs': agg.tolist(),
                    'sampled': printed_ids[pred_idx],
                    'ideal': printed_ids[ideal_idx],
                    'correct': bool(correct),
                    'conf': float(conf),
                    'low_confidence': low_conf,
                    'perm_reduce': reduce_mode,
                },
            }

        # base/shuffle_both/full: single input
        sys_msg, eval_sample = inp
        text = sys_msg + '\n\n'
        if num_few_shot > 0:
            for s in few_shot_samples[:num_few_shot]:
                text += s + '\n\n'
        text += eval_sample
        if not bpe_has_space_prefix:
            text += ' '

        probs, pred_idx, conf = _pick_for_single_text(text)
        correct = (pred_idx == int(ideal_idx))
        low_conf = bool(conf < float(lowconf_threshold))
        return {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': text,
                'options': options,
                'probs': probs.tolist(),
                'sampled': printed_ids[pred_idx],
                'ideal': printed_ids[ideal_idx],
                'correct': bool(correct),
                'conf': float(conf),
                'low_confidence': low_conf,
            },
        }

    return eval_fn


def prepare_eval_fn_noid(model, toker, few_shot_samples, num_few_shot):
    """no-id: choose by option-wise NLL."""
    toker.padding_side = 'right'

    def eval_fn(sample, rng: random.Random):
        idx, (inp, options, ideal_idx, ideal_disp) = sample
        sys_msg, eval_sample = inp
        input_text = sys_msg + '\n\n'
        if num_few_shot > 0:
            for s in few_shot_samples[:num_few_shot]:
                input_text += s + '\n\n'
        input_text += eval_sample

        prefix_input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids
        losses, lengths = [], []
        for option in options:
            text = input_text + ' ' + option.strip()
            input_ids = toker(text, truncation=False, return_tensors="pt").input_ids.to(model.device)
            lengths.append(input_ids.size(1) - prefix_input_ids.size(1))

            labels = input_ids.clone()
            labels[:, :prefix_input_ids.size(1)] = -100

            input_ids = input_ids[..., -1536:]
            labels = labels[..., -1536:]

            with torch.no_grad():
                loss = model(input_ids=input_ids, labels=labels).loss.detach().to(torch.float32).cpu().item()
            losses.append(loss)

        nll = -np.array(losses, dtype=np.float32)
        probs = np.exp(nll - np.max(nll))
        probs = probs / (probs.sum() + 1e-10)

        pred_idx = int(np.argmin(losses))
        conf = float(probs[pred_idx])
        correct = (pred_idx == int(ideal_idx))

        return {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'lengths': lengths,
                'losses': losses,
                'probs': probs.tolist(),
                'sampled': str(pred_idx),  # index notation for no-id
                'ideal': str(ideal_idx),
                'correct': bool(correct),
                'conf': conf,
                'low_confidence': False,
            },
        }

    return eval_fn


# ----------------------------------------------------
# main
# ----------------------------------------------------

def _model_dir_name(args):
    base = args.model_name or str(getattr(args, "pretrained_model_path", "model")).split("/")[-1]
    return base.replace("/", "_").replace(" ", "_")


def main():
    # strong, unbuffered logging for SLURM
    setup_logging(to_stdout=True, level=logging.INFO)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    patch_open()
    log_cuda_env()

    args = parse_arguments()
    logger.info("Parsed args: %s", args)

    # Load HF
    try:
        toker = AutoTokenizer.from_pretrained(
            args.pretrained_model_path,
            use_fast=False,
            add_bos_token=False,
            add_eos_token=False,
            cache_dir=args.cache_dir,
        )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            device_map='auto',
            use_safetensors=True,
            torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
            cache_dir=args.cache_dir,
        )
        model.eval()
        logger.info("Model & tokenizer loaded: %s", _model_dir_name(args))
    except Exception as e:
        logger.exception("Failed to load model/tokenizer: %s", e)
        sys.exit(1)

    printed_example = False

    for eval_name in args.eval_names:
        subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn, save_path = prepare_eval(args, eval_name)

        for subject in subjects:
            out_path = f"{save_path}/{subject}.jsonl"
            if os.path.exists(out_path):
                logger.info("Results already exist, skip: %s", out_path)
                continue

            logger.info(_blue(f"Preparing subject: {subject}"))
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = prepare_eval_fn(model, toker, few_shot_samples)

            # Print a single prompt example
            if args.print_prompt_example and not printed_example and len(eval_samples) > 0:
                try:
                    first_input, _first_options, _ideal_idx, _ideal_disp = eval_samples[0]

                    def build_text(inp):
                        # If perm/cyclic, show the first transformed variant
                        if isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], list):
                            (sys_msg, eval_sample), _perm_map = inp[0]
                        else:
                            sys_msg, eval_sample = inp
                        text = sys_msg + '\n\n'
                        # few-shot
                        nfs = int(eval_name.split(',')[1]) if len(eval_name.split(',')) > 1 else 0
                        if nfs > 0:
                            for s in few_shot_samples[:nfs]:
                                text += s + '\n\n'
                        text += eval_sample
                        # tokenizer space-prefix fix
                        bpe_has_space_prefix = (toker(': A').input_ids[-1] != toker(':A').input_ids[-1])
                        if not bpe_has_space_prefix:
                            text += ' '
                        return text

                    example_text = build_text(first_input)
                    logger.info(_purple("==== Prompt example ====\n%s\n==== End prompt example ===="), example_text)
                    printed_example = True
                except Exception as e:
                    logger.warning("Failed to build prompt example: %s", e)

            logger.info(_blue(f"Run started: {subject}"))
            results = eval_all_samples(
                eval_fn, eval_samples,
                name=f"{eval_name},{subject}",
                threads=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 이제 perm/cyclic도 성능 계산을 수행 (우리가 'correct'를 채워줌)
            metrics = None
            if len(results) > 0:
                metrics = {'type': 'metric', 'data': {}}
                metrics['data']['accuracy'] = get_accuracy(results)
                metrics['data']['bootstrap_std'] = get_bootstrap_accuracy_std(results)
                logger.info("Final report for %s:", subject)
                for key, value in metrics['data'].items():
                    logger.info("  %s: %s", key, value)

            logger.info(_orange(f"Run completed: {subject}"))
            save_results(out_path, results, metrics)
            logger.info("Results saved: %s", out_path)

    logger.info("All done.")


if __name__ == "__main__":
    main()
