# eval_clm_utils.py
import os
import sys
import random
import copy
import json
import argparse
import logging
from typing import List
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math

from utils import (
    _norm,
    shuffle_options_with_ids,
    move_answer,
    cycle_options,
    permute_options,
)

logger = logging.getLogger(__name__)


# -----------------------------
# Args
# -----------------------------
def parse_arguments():
    logger.info(f'cuda is available {torch.cuda.is_available()}')
    logger.info(f'cuda device count {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        logger.info(f'cuda device name {torch.cuda.get_device_name(0)}')

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--eval_names", type=str, nargs='+', default=[],
                        help='eval tasks and settings')
    parser.add_argument("--option_id_set", type=str, default=None,
                        help='custom option ID string (e.g., ABCD / abcd / 1234). Length must match #options')
    parser.add_argument("--print_prompt_example", action="store_true",
                        help='Print exactly one example prompt (after applying option ids) and continue evaluation')
    parser.add_argument("--cache_dir", type=str, default="models",
                        help='Hugging Face cache directory for model/tokenizer downloads')
    parser.add_argument("--option_id_sets", type=str, nargs='+', default=None,
                        help='Provide exactly two option ID sets to compare (e.g., "ABCD abcd")')
    parser.add_argument("--test", action="store_true",
                        help='Test mode: evaluate only 100 samples instead of all samples')

    # ---- Dummy / Noise correction ----
    parser.add_argument("--dummy_id_set", type=str, default="XZ",
                        help="더미 라벨 2개(예: XZ). 프롬프트엔 표기하지 않고 확률 보정용으로만 사용")
    parser.add_argument("--noise_mode", type=str, default="off",
                        choices=["off", "dummy_avg", "dummy_max"],
                        help="off: 보정 안 함, dummy_avg: 더미 확률 평균을 빼기, dummy_max: 더미 확률 최대치를 빼기")
    parser.add_argument("--noise_alpha", type=float, default=1.0,
                        help="노이즈 가중치 α. 최종 q = ReLU(p - α·noise), 이후 재정규화")

    # ---- W&B ----
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--wandb_sample_idx", type=int, default=None,
                        help="Sample idx to log detailed prompts/probs; default: first sample")

    # ---- Ours(low-conf cascade) ----
    parser.add_argument("--ours_low_conf_percent", type=float, default=10.0,
                        help="Bottom percentile (e.g., 10.0) of confidence (from beta subset) to trigger cascading ensemble")

    # ---- Overwrite ----
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results files if they exist")

    # ---- Cascading export (추가 저장물) ----
    parser.add_argument("--cascade_export", action="store_true",
                        help="FULL 또는 CYCLIC 결과로부터 cascading/switch-* 예측을 산출/저장")
    parser.add_argument("--cascade_beta", type=float, default=0.0,
                        help="0~1. 베타 커브 정의 그대로: 앞쪽 beta 비율 샘플은 기본만, 나머지는 cascade")
    parser.add_argument("--cascade_policy", type=str, default="ours",
                        choices=["ours", "switch_full", "switch_cyclic"],
                        help="ours=confidence 기반 순차 집계, switch_full=low-conf면 전부 집계, switch_cyclic=low-conf면 k회전만 집계")

    args = parser.parse_args()

    args.model_name = args.pretrained_model_path.split('/')[-1]

    for eval_name in args.eval_names:
        eval_args = eval_name.split(',')
        task = eval_args[0]
        if task not in ['mmlu', 'arc', 'csqa']:
            raise ValueError(f"Unknown task: {task}")

        num_few_shot = int(eval_args[1])

        setting = eval_args[2] if len(eval_args) > 2 else None
        if setting is not None and not (
            setting in [
                'noid',
                'perm', 'cyclic', 'full',
                'shuffle_both',
            ] or (setting.startswith('move') and setting[-1] in ['a', 'b', 'c', 'd'])
        ):
            raise ValueError(f"Unknown setting: {setting}")

    return args


# -----------------------------
# Data prep
# -----------------------------
def prepare_eval(args, eval_name):
    # task and setting
    eval_args = eval_name.split(',')
    args.task = task = eval_args[0]
    args.num_few_shot = num_few_shot = int(eval_args[1])
    args.setting = setting = eval_args[2] if len(eval_args) > 2 and eval_args[2] else None
    if setting is not None and setting.startswith('move'):
        moved_answer = setting[-1].upper()

    # save_path
    save_path = f'results_{task}/{num_few_shot}s_{args.model_name}/{task}'
    if setting is not None:
        save_path += f'_{setting}'
    if getattr(args, 'option_id_set', None):
        save_path += f'_id-{args.option_id_set}'
    args.save_path = save_path
    os.makedirs(args.save_path, exist_ok=True)

    option_ids = list('ABCD')
    option_ids_header = list('ABCD')
    if task in ['csqa']:
        option_ids = list('ABCDE')
        option_ids_header = list('ABCDE')

    # custom option ids
    if getattr(args, 'option_id_set', None):
        k = len(option_ids_header)
        custom = list(args.option_id_set)
        if len(custom) != k:
            raise ValueError(f"option_id_set length must be {k} for task '{task}', got {len(custom)}: {args.option_id_set}")
        option_ids = custom

    data_path = f'data_{task}'
    subjects = sorted([f.split("_test.csv")[0]
                       for f in os.listdir(f'{data_path}/test') if "_test.csv" in f])

    # sys_msg
    if 'mmlu' in task:
        sys_msg = 'The following are multiple choice questions about {}.'
    else:  # task in ['arc', 'tqa']
        sys_msg = 'The following are multiple choice questions.'
    sys_msg += ' You should directly answer the question by choosing the correct option.'

    # prompt builder
    def create_user_prompt(question: str, options: List[str]):
        if setting in ['noid']:
            user_prompt = f"Question: {question.strip()}\nOptions:\n" + \
                "\n".join([f"{answer}".strip()
                           for option_id, answer in zip(option_ids, options)]) + \
                "\nAnswer:"
        elif setting in ['shuffle_both']:
            shuffled_option_ids, shuffled_options = shuffle_options_with_ids(option_ids, options)
            user_prompt = f"Question: {question.strip()}\nOptions:\n" + \
                "\n".join([f"{option_id}. {answer}".strip()
                           for option_id, answer in zip(shuffled_option_ids, shuffled_options)]) + \
                "\nAnswer:"
        else:
            user_prompt = f"Question: {question.strip()}\nOptions:\n" + \
                "\n".join([f"{option_id}. {answer}".strip()
                           for option_id, answer in zip(option_ids, options)]) + \
                "\nAnswer:"
        return user_prompt

    # few-shot
    def prepare_few_shot_samples(subject):
        df = pd.read_csv(f'{data_path}/dev/{subject}_dev.csv', names=("Question", *option_ids_header, "Answer"), dtype=str)
        if setting in ['noid']:
            few_shot_samples = df.apply(lambda x:
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header])
                + ' ' + x[x["Answer"]]
            , axis=1).to_list()
        else:
            few_shot_samples = df.apply(lambda x:
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header])
                + ' ' + option_ids[option_ids_header.index(x["Answer"])]
            , axis=1).to_list()
        return few_shot_samples

    # eval samples
    def prepare_eval_samples(subject):
        df = pd.read_csv(open(f'{data_path}/test/{subject}_test.csv'), names=("Question", *option_ids_header, "Answer"), dtype=str)

        if setting is not None and setting.startswith('move'):
            df = df.apply(lambda x: move_answer(x, moved_answer), axis=1)

        if setting in ['perm', 'full']:
            inputs = df.apply(lambda x: [
                [
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], permuted_options),
                ] for permuted_options in permute_options([x[e] for e in option_ids_header])
            ], axis=1).to_list()
        elif setting in ['cyclic']:
            inputs = df.apply(lambda x: [
                [
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], cycled_options),
                ] for cycled_options in cycle_options([x[e] for e in option_ids_header])
            ], axis=1).to_list()
        else:
            inputs = df.apply(lambda x: [
                sys_msg.format(subject.replace('_', ' ')),
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header]),
            ], axis=1).to_list()
        options = df.apply(lambda x: [str(x[e]) for e in option_ids_header], axis=1).to_list()
        ideals = df.apply(lambda x: option_ids[option_ids_header.index(x["Answer"])], axis=1).to_list()
        return list(zip(inputs, options, ideals))

    # which eval fn
    if setting in ['noid']:
        prepare_eval_fn = partial(prepare_eval_fn_noid, num_few_shot=num_few_shot, option_ids=option_ids)
    elif setting in ['perm', 'cyclic', 'full']:
        prepare_eval_fn = partial(prepare_eval_fn_perm, num_few_shot=num_few_shot, option_ids=option_ids)
    else:
        prepare_eval_fn = partial(prepare_eval_fn_base, num_few_shot=num_few_shot, option_ids=option_ids)

    return subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn


# -----------------------------
# Noise correction helper
# -----------------------------
def _apply_noise(letter_probs: np.ndarray,
                 dummy_probs: np.ndarray,
                 mode: str,
                 alpha: float) -> np.ndarray:
    """q = ReLU(p - alpha * noise); then renormalize."""
    if mode == "off" or dummy_probs is None or dummy_probs.size == 0:
        p = letter_probs.astype(np.float64)
        s = float(p.sum()) + 1e-12
        return (p / s)
    if mode == "dummy_avg":
        noise = float(np.mean(dummy_probs))
    elif mode == "dummy_max":
        noise = float(np.max(dummy_probs))
    else:
        noise = 0.0
    q = np.maximum(letter_probs.astype(np.float64) - alpha * noise, 0.0)
    s = float(q.sum())
    if s <= 0:
        # fallback: just normalize original p
        p = letter_probs.astype(np.float64)
        s = float(p.sum()) + 1e-12
        return (p / s)
    return (q / s)


# -----------------------------
# Eval fns
# -----------------------------
def prepare_eval_fn_base(model, toker, few_shot_samples, num_few_shot, option_ids):
    bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]

    def eval_fn(sample, rng: random.Random, *,
                noise_mode=None, noise_alpha=None, dummy_id_set=None):
        idx, (input, options, ideal) = sample
        sys_msg, eval_sample = input.copy()
        input_text = sys_msg + '\n\n'
        if num_few_shot > 0:
            for s in few_shot_samples[:num_few_shot]:
                input_text += s + '\n\n'
        input_text += eval_sample
        if not bpe_has_space_prefix:
            input_text += ' '

        input_ids = toker(input_text, return_tensors="pt").input_ids.to(model.device)
        input_ids = input_ids[..., -1536:]
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[:, -1].view(-1)

        # token indices (letters + dummies), with/without space
        letters = list(option_ids)
        dummies = list(dummy_id_set or "")
        # collect indices
        letter_idx_space = [toker(f': {e}').input_ids[-1] for e in letters]
        letter_idx_nospace = [toker(f':{e}').input_ids[-1] for e in letters]
        dummy_idx_space = [toker(f': {e}').input_ids[-1] for e in dummies] if len(dummies) > 0 else []
        dummy_idx_nospace = [toker(f':{e}').input_ids[-1] for e in dummies] if len(dummies) > 0 else []

        all_indices = letter_idx_space + letter_idx_nospace + dummy_idx_space + dummy_idx_nospace
        probs_all = F.softmax(logits[..., all_indices], dim=-1).detach().cpu().to(torch.float32).numpy()

        K = len(letters)
        D = len(dummies)
        # reshape to (2, K+D) → sum over space/no-space
        probs_all = probs_all.reshape(2, K + D).sum(axis=0) if (K + D) > 0 else probs_all
        letter_probs = probs_all[:K]
        dummy_probs = probs_all[K:] if D > 0 else None

        # apply noise
        mode = noise_mode or "off"
        alpha = float(noise_alpha if noise_alpha is not None else 1.0)
        probs = _apply_noise(letter_probs, dummy_probs, mode, alpha)

        sampled = option_ids[np.argmax(probs)]
        correct = (sampled == ideal)
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'probs': probs.tolist(),  # noise-corrected letters only
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_noid(model, toker, few_shot_samples, num_few_shot, option_ids):
    toker.padding_side = 'right'
    bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]

    # noid는 토큰 확률이 아니라 LM loss 기반이라 noise 보정 적용하지 않음
    def eval_fn(sample, rng: random.Random, **kwargs):
        idx, (input, options, ideal) = sample
        sys_msg, eval_sample = input.copy()
        input_text = sys_msg + '\n\n'
        if num_few_shot > 0:
            for s in few_shot_samples[:num_few_shot]:
                input_text += s + '\n\n'
        input_text += eval_sample
        prefix_input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids

        losses = []
        lengths = []
        for option in options:
            prefix_and_option_text = input_text + ' ' + option.strip()
            input_ids = toker(prefix_and_option_text, truncation=False, return_tensors="pt").input_ids.to(model.device)
            lengths.append(input_ids.size(1) - prefix_input_ids.size(1))

            labels = input_ids.clone()
            labels[:, :prefix_input_ids.size(1)] = -100

            input_ids = input_ids[..., -1536:]
            labels = labels[..., -1536:]

            with torch.no_grad():
                loss = model(input_ids=input_ids, labels=labels).loss.detach().to(torch.float32).cpu().item()
            losses.append(loss)

        nll = - np.array(losses)
        probs = np.exp(nll - np.max(nll))
        probs = probs / (probs.sum() + 1e-10)

        sampled = option_ids[np.argmin(losses)]
        correct = (sampled == ideal)
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'lengths': lengths,
                'losses': losses,
                'probs': probs.tolist(),
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_perm(model, toker, few_shot_samples, num_few_shot, option_ids):
    toker.padding_side = 'left'
    bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]

    def eval_fn(sample, rng: random.Random, *,
                noise_mode=None, noise_alpha=None, dummy_id_set=None):
        idx, (probing_inputs, options, ideal) = sample
        # cyclic=k, full=k! permutations
        num_options = len(option_ids)
        expected_counts = {num_options, math.factorial(num_options)}
        assert len(probing_inputs) in expected_counts

        input_texts = []
        for probing_input in probing_inputs:
            sys_msg, eval_sample = probing_input.copy()
            input_text = sys_msg + '\n\n'
            if num_few_shot > 0:
                for s in few_shot_samples[:num_few_shot]:
                    input_text += s + '\n\n'
            input_text += eval_sample
            if not bpe_has_space_prefix:
                input_text += ' '
            input_texts.append(input_text)

        letters = list(option_ids)
        dummies = list(dummy_id_set or "")
        K = len(letters); D = len(dummies)

        # Precompute token indices (letters + dummies), with/without space
        letter_idx_space = [toker(f': {e}').input_ids[-1] for e in letters]
        letter_idx_nospace = [toker(f':{e}').input_ids[-1] for e in letters]
        dummy_idx_space = [toker(f': {e}').input_ids[-1] for e in dummies] if D > 0 else []
        dummy_idx_nospace = [toker(f':{e}').input_ids[-1] for e in dummies] if D > 0 else []
        all_indices = letter_idx_space + letter_idx_nospace + dummy_idx_space + dummy_idx_nospace

        all_probs = []
        for input_text in input_texts:
            input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids.to(model.device)
            input_ids = input_ids[..., -1536:]
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[:, -1]

            probs_all = F.softmax(logits[..., all_indices], dim=-1).detach().to(torch.float32).cpu().numpy()
            # (1, 2*(K+D)) → (2, K+D) → sum axis=0
            probs_all = probs_all.reshape(input_ids.size(0), 2, K + D).sum(axis=1)[0] if (K + D) > 0 else probs_all[0]
            letter_probs = probs_all[:K]
            dummy_probs = probs_all[K:] if D > 0 else None

            mode = noise_mode or "off"
            alpha = float(noise_alpha if noise_alpha is not None else 1.0)
            corrected = _apply_noise(letter_probs, dummy_probs, mode, alpha)
            all_probs.append(corrected.tolist())

        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_texts[0],
                'prompts': input_texts,
                'options': options,
                'probs': all_probs,  # list of noise-corrected probs (letters only)
                'ideal': ideal,
            },
        }
        return result
    return eval_fn
