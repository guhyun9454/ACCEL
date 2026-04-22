
import os
import sys
import random
import copy
import json
import argparse
import logging
from tqdm import tqdm
from typing import List, Tuple
from functools import partial
from itertools import permutations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import (
    _norm,
    _purple,
    shuffle_options_with_ids,
    move_answer,
    cycle_options,
    permute_options,
)

logger = logging.getLogger(__name__)


def _build_toprank_latin_rank_rows(first_rank_by_slot: List[int]) -> List[Tuple[int, ...]]:
    """
    Build a small Latin-square schedule in rank space.

    The first row is the base displayed rank order. The second row swaps rank1
    and rank2 positions, then rotates the remaining ranks so no slot keeps the
    same rank. Remaining rows are filled by deterministic backtracking.
    """
    first = tuple(int(x) for x in first_rank_by_slot)
    k = len(first)
    if k <= 1:
        return [first]
    if sorted(first) != list(range(k)):
        raise ValueError(f"invalid rank row for Latin schedule: {first}")

    pos_r1 = first.index(0)
    pos_r2 = first.index(1)
    second = list(first)
    second[pos_r1] = 1
    second[pos_r2] = 0
    remaining_positions = [j for j in range(k) if j not in {pos_r1, pos_r2}]
    remaining_ranks = [first[j] for j in remaining_positions]
    if len(remaining_ranks) > 1:
        remaining_ranks = remaining_ranks[1:] + remaining_ranks[:1]
    for j, rank in zip(remaining_positions, remaining_ranks):
        second[j] = int(rank)
    seed_rows = [first, tuple(second)]

    all_rows = list(permutations(range(k)))
    rows: List[Tuple[int, ...]] = []
    used_by_col = [set() for _ in range(k)]

    def _add_row(row: Tuple[int, ...]) -> None:
        rows.append(row)
        for col, symbol in enumerate(row):
            used_by_col[col].add(symbol)

    for row in seed_rows:
        if row in rows:
            continue
        if any(row[col] in used_by_col[col] for col in range(k)):
            continue
        _add_row(row)

    def _search() -> bool:
        if len(rows) >= k:
            return True
        for cand in all_rows:
            if cand in rows:
                continue
            if any(cand[col] in used_by_col[col] for col in range(k)):
                continue
            _add_row(cand)
            if _search():
                return True
            removed = rows.pop()
            for col, symbol in enumerate(removed):
                used_by_col[col].remove(symbol)
        return False

    if _search() and len(rows) == k:
        return rows

    # Conservative fallback: a standard cyclic Latin square anchored to first.
    return [tuple(first[(col + shift) % k] for col in range(k)) for shift in range(k)]


def _build_toprank_latin_perms(base_probs: np.ndarray, k: int) -> List[Tuple[int, ...]]:
    """
    Return displayed-slot -> content-index permutations for a base-rank Latin
    schedule. Row 0 is identity; every base-rank content appears once per slot.
    """
    bp = np.asarray(base_probs, dtype=np.float64)
    rank_order = np.argsort(bp)[::-1].tolist()
    rank_of_content = {int(content_idx): int(rank) for rank, content_idx in enumerate(rank_order)}
    rank_to_content = {int(rank): int(content_idx) for rank, content_idx in enumerate(rank_order)}
    first_rank_by_slot = [rank_of_content[int(slot)] for slot in range(int(k))]
    rank_rows = _build_toprank_latin_rank_rows(first_rank_by_slot)
    return [tuple(rank_to_content[int(rank)] for rank in row) for row in rank_rows]


def parse_arguments():
    # ---- Safe CUDA logging (GPU 없을 때도 안 터지게) ----
    try:
        cuda_ok = torch.cuda.is_available()
        n_dev = torch.cuda.device_count() if cuda_ok else 0
        dev_name = torch.cuda.get_device_name(0) if (cuda_ok and n_dev > 0) else "N/A"
        logger.info(f'cuda is available {cuda_ok}')
        logger.info(f'cuda device count {n_dev}')
        logger.info(f'cuda device name {dev_name}')
    except Exception as e:
        logger.warning(f'CUDA info logging failed: {e}')

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

    # W&B logging flags
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_entity", type=str, default="capde",
                        help='W&B entity (default: "capde")')
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--wandb_sample_idx", type=int, default=None,
                        help="Sample idx to log detailed prompts/probs; default: first sample")

    # Ours method hyperparameters (existing)
    parser.add_argument("--ours_low_conf_percent", type=float, default=30.0,
                        help="Bottom percentile (e.g., 30.0) of confidence (from beta subset) to trigger cascading ensemble")
    parser.add_argument("--ours_low_conf_percent_list", type=str, default="5,10,20,30",
                        help="Comma-separated percentile list for derived-policy reports/plots (default: '5,10,20,30'). Overrides --ours_low_conf_percent. Use empty string \"\" to disable and fall back to --ours_low_conf_percent.")
    parser.add_argument("--ours_th1_tradeoff", type=str, default="5,10,20,30",
                        help="th1 percentile values for trade-off plot (e.g., 5,10,20,30). 각 th1에 대해 th2 curve를 그림.")
    parser.add_argument("--ours_th2_tradeoff", type=str,
                        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
                        help="th2 percentile values for trade-off plot. Default: 1..30 (각 th1에 대해 th2를 1~30까지 변화시킴).")
    parser.add_argument("--ours_top2_gap_frac", type=float, default=0.3,
                        help="If (top1 - top2) / top1 <= this, treat as 'very ambiguous' and send directly to cyclic (use 0.0 to disable).")

    # =========================================================
    # [ADD] Simple PRIDE (PriDe) mixing knobs (online-friendly)
    # =========================================================
    parser.add_argument("--pride_mix", action="store_true",
                        help="Enable PRIDE debiasing then run OUR online policies on debiased probs (for comparison).")
    parser.add_argument("--pride_prefix_ratio", type=float, default=0.02,
                        help="Random prefix ratio used to estimate PRIDE prior (default=0.02).")
    parser.add_argument("--skip_full", action="store_true",
                        help="Use cyclic instead of full permutations when setting=full (e.g., MMLU 4-choice: 4x instead of 24x).")
    parser.add_argument("--pride_seed", type=int, default=0,
                        help="Seed for PRIDE random prefix sampling (default=0).")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of runs for derived policies (PRIDE prior, cyclic_random). Results averaged over runs, like debiase_pride.py (default=1).")

    # Three-curves plot: fraction lists (modify to change number of points)
    parser.add_argument("--plot_cyclic_fractions", type=str, default="0,10,20,30,40,50,60,70,80,90,100",
                        help="Cyclic (no PRIDE) curve fractions, comma-separated (e.g. 0,10,20,...,100).")
    parser.add_argument("--plot_pride_ours_fractions", type=str, default="0.5,1,2,5,10,20,30,40,50,60,70,80,90,100",
                        help="OURS th1 curve (x-axis). Default+PRIDE도 동일 p 사용.")
    parser.add_argument("--plot_pride_prefix_fractions", type=str, default="0.5,1,2,5,10,20,30,40,50,60,70,80,90,100",
                        help="Ours+PRIDE에서 PriDe prefix(alpha) 값. 선택 가능한 α 목록.")
    parser.add_argument("--rank_slot_summary_json", type=str, default=None,
                        help="Path to multi_model_rank_slot_delta_summary.json used for Latin-Gaussian slot noise std lookup.")

    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose logs (extra summaries).")

    # =========================================================
    # [ADD] AvgGap(PSEUDO, ONLINE) knobs (eval_clm.py에서 getattr로 쓰는 것들)
    # =========================================================
    # debug: allow usage like --ours_debug_mad 1
    parser.add_argument("--ours_debug_mad", type=int, default=0,
                        help="(0/1) Print MAD/threshold debug logs.")
    parser.add_argument("--ours_debug_mad_n", type=int, default=5,
                        help="Max debug prints per beta.")
    
    parser.add_argument("--ours_th1_ema", type=float, default=0.01, help="EMA rate used in entropy probe policy: (1) th2 teacher update step size, (2) th1 tracking to th2.")

    # MAD EMA alpha
    parser.add_argument("--ours_mad_alpha", type=float, default=0.10,
                        help="EMA alpha for MAD update.")

    # th1 online update lr
    parser.add_argument("--ours_th_lr_up", type=float, default=0.05,
                        help="th1 increase lr (when tc==t1).")
    parser.add_argument("--ours_th_lr_dn", type=float, default=0.05,
                        help="th1 decrease lr (when tc!=t1).")

    # th2 = th1 (+/-) MAD
    parser.add_argument("--ours_th2_mode", type=str, default="plus",
                        choices=["plus", "minus", "sub", "-"],
                        help="th2 = th1 + MAD (plus) or th1 - MAD (minus).")

    # th1 init / clamp range (all in [0,1])
    # NOTE: eval_clm.py에서 None이면 perc로 자동 설정하도록 처리하는 걸 권장
    parser.add_argument("--ours_th1_init", type=float, default=None,
                        help="Initial th1 in [0,1]. If omitted, auto=ours_low_conf_percent/100.")
    parser.add_argument("--ours_th1_min", type=float, default=0.0,
                        help="Min clamp for th1.")
    parser.add_argument("--ours_th1_max", type=float, default=1.0,
                        help="Max clamp for th1.")

    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results files if they exist")
    parser.add_argument("--develop", action="store_true",
                        help="Dev mode: skip model/data eval and generate dummy numeric logs/plots (for W&B/Streamlit pipeline test).")

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

    # Override displayed option IDs by user-specified token set (for probing token preference)
    # Keep headers (ground-truth labels) as uppercase letters to match dataset files.
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
    else: # task in ['arc', 'tqa']
        sys_msg = 'The following are multiple choice questions.'

    sys_msg += ' You should directly answer the question by choosing the correct option.'

    # create_user_prompt
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

    # prepare_few_shot_samples
    # few_shot_seed: None이면 기존 동작(첫 N개). 정수면 해당 seed로 shuffle 후 사용 (n_runs 시 run별 다른 few-shot)
    def prepare_few_shot_samples(subject, few_shot_seed=None):
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
        if few_shot_seed is not None:
            rng = random.Random(int(few_shot_seed))
            rng.shuffle(few_shot_samples)
        return few_shot_samples

    if getattr(args, 'skip_full', False) and setting == 'full':
        k_opts = len(option_ids_header)
        logger.info(_purple(f"[skip_full] Using cyclic instead of full permutations (k={k_opts} → {k_opts}x inferences)"))

    # prepare_eval_samples
    def prepare_eval_samples(subject):
        df = pd.read_csv(open(f'{data_path}/test/{subject}_test.csv'), names=("Question", *option_ids_header, "Answer"), dtype=str)

        if setting is not None and setting.startswith('move'):
            df = df.apply(lambda x: move_answer(x, moved_answer), axis=1)

        # NOTE: full permutation은 k!로 급증하므로, k>=5에서는 full이라도 cyclic만 생성(자동 다운그레이드)
        # --skip_full: full이어도 cyclic만 사용 (MMLU 등 오래 걸릴 때)
        k_opts = len(option_ids_header)
        full_permutation_disabled = (setting == 'full' and k_opts >= 5) or bool(getattr(args, 'skip_full', False))

        if setting in ['perm'] or (setting == 'full' and not full_permutation_disabled):
            permuted_indices = list(sorted(permutations(range(len(option_ids_header)))))
            inputs = df.apply(lambda x: [
                [
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], [x[e] for e in [option_ids_header[idx] for idx in perm_tup]]),
                ] for perm_tup in permuted_indices
            ], axis=1).to_list()
            perm_tuples = [list(permuted_indices) for _ in range(len(df))]
        elif setting in ['cyclic'] or (setting == 'full' and full_permutation_disabled):
            cyc_indices = [tuple((i + s) % k_opts for i in range(k_opts)) for s in range(k_opts)]
            inputs = df.apply(lambda x: [
                [
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], [x[e] for e in [option_ids_header[idx] for idx in perm_tup]]),
                ] for perm_tup in cyc_indices
            ], axis=1).to_list()
            perm_tuples = [list(cyc_indices) for _ in range(len(df))]
        else:
            inputs = df.apply(lambda x: [
                sys_msg.format(subject.replace('_', ' ')),
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header]),
            ], axis=1).to_list()
            perm_tuples = None
        options = df.apply(lambda x: [str(x[e]) for e in option_ids_header], axis=1).to_list()
        ideals = df.apply(lambda x: option_ids[option_ids_header.index(x["Answer"])], axis=1).to_list()
        if perm_tuples is None:
            return list(zip(inputs, options, ideals))
        return list(zip(inputs, options, ideals, perm_tuples))
    
    

    # prepare_eval_fn
    if setting in ['noid']:
        prepare_eval_fn = partial(prepare_eval_fn_noid, num_few_shot=num_few_shot, option_ids=option_ids)
    elif setting in ['perm', 'cyclic', 'full']:
        prepare_eval_fn = partial(prepare_eval_fn_perm, num_few_shot=num_few_shot, option_ids=option_ids)
    else:
        prepare_eval_fn = partial(prepare_eval_fn_base, num_few_shot=num_few_shot, option_ids=option_ids)

    return subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn


def prepare_eval_fn_base(model, toker, few_shot_samples, num_few_shot, option_ids):
    # NOTE: 일부 Seq2Seq 토크나이저(T5 등)는 기본적으로 EOS 같은 special token을 붙여서
    # `input_ids[-1]`가 항상 EOS가 되는 문제가 있음. 반드시 special token 없이 비교/추출한다.
    bpe_has_space_prefix = (
        toker(": A", add_special_tokens=False).input_ids[-1]
        != toker(":A", add_special_tokens=False).input_ids[-1]
    )

    def eval_fn(sample, rng: random.Random):
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
        is_seq2seq = getattr(getattr(model, "config", None), "is_encoder_decoder", False)
        with torch.no_grad():
            if is_seq2seq:
                cfg = getattr(model, "config", None)
                dec_start = getattr(cfg, "decoder_start_token_id", None) or getattr(cfg, "pad_token_id", 0)
                dec_ids = torch.full((input_ids.size(0), 1), dec_start, dtype=torch.long, device=model.device)
                logits = model(input_ids=input_ids, decoder_input_ids=dec_ids).logits[:, -1].view(-1)
            else:
                logits = model(input_ids=input_ids).logits[:, -1].view(-1)

        option_indices = (
            [toker(f": {e}", add_special_tokens=False).input_ids[-1] for e in option_ids]
            + [toker(f":{e}", add_special_tokens=False).input_ids[-1] for e in option_ids]
        )
        probs = F.softmax(
            logits[..., option_indices], dim=-1
        ).detach().cpu().to(torch.float32).numpy()
        probs = probs.reshape(2, len(option_ids)).sum(axis=0)
        sampled = option_ids[np.argmax(probs)]

        correct = (sampled == ideal)
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'probs': probs.tolist(),
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_noid(model, toker, few_shot_samples, num_few_shot, option_ids):
    toker.padding_side = 'right'
    bpe_has_space_prefix = (
        toker(": A", add_special_tokens=False).input_ids[-1]
        != toker(":A", add_special_tokens=False).input_ids[-1]
    )

    def eval_fn(sample, rng: random.Random):
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
                loss = model(
                    input_ids=input_ids,
                    labels=labels,
                ).loss.detach().to(torch.float32).cpu().item()
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
    bpe_has_space_prefix = (
        toker(": A", add_special_tokens=False).input_ids[-1]
        != toker(":A", add_special_tokens=False).input_ids[-1]
    )

    def _rebuild_user_prompt_like(eval_sample_text: str, question_text: str, reordered_options: List[str]) -> str:
        text = str(eval_sample_text or "")
        use_option_labels = True
        if "\nOptions:\n" in text:
            try:
                options_block = text.split("\nOptions:\n", 1)[1].split("\nAnswer:", 1)[0]
                option_lines = [ln.strip() for ln in options_block.splitlines() if ln.strip()]
                if option_lines:
                    use_option_labels = all(
                        any(ln.startswith(f"{oid}.") for oid in option_ids)
                        for ln in option_lines[: min(len(option_ids), len(option_lines))]
                    )
            except Exception:
                use_option_labels = True
        if use_option_labels:
            options_text = "\n".join(
                f"{option_id}. {answer}".strip()
                for option_id, answer in zip(option_ids, reordered_options)
            )
        else:
            options_text = "\n".join(str(answer).strip() for answer in reordered_options)
        return f"Question: {str(question_text).strip()}\nOptions:\n{options_text}\nAnswer:"

    def eval_fn(sample, rng: random.Random):
        idx, payload = sample
        if len(payload) == 4:
            probing_inputs, options, ideal, perm_tuples = payload
        else:
            probing_inputs, options, ideal = payload
            perm_tuples = None
        # Allow either cyclic count (= #options) or full permutation count (= factorial(#options))
        num_options = len(option_ids)
        expected_counts = {num_options, math.factorial(num_options)}
        assert len(probing_inputs) in expected_counts
        if perm_tuples is None:
            if len(probing_inputs) == num_options:
                perm_tuples = [tuple((i + s) % num_options for i in range(num_options)) for s in range(num_options)]
            else:
                perm_tuples = list(sorted(permutations(range(num_options))))
        perm_tuples = [tuple(int(x) for x in p) for p in perm_tuples]

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

        is_seq2seq = getattr(getattr(model, "config", None), "is_encoder_decoder", False)
        decoder_start = None
        if is_seq2seq:
            cfg = getattr(model, "config", None)
            decoder_start = getattr(cfg, "decoder_start_token_id", None) or getattr(cfg, "pad_token_id", 0)

        option_indices = (
            [toker(f": {e}", add_special_tokens=False).input_ids[-1] for e in option_ids]
            + [toker(f":{e}", add_special_tokens=False).input_ids[-1] for e in option_ids]
        )

        def _score_input_text(input_text: str) -> List[float]:
            input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids.to(model.device)
            input_ids = input_ids[..., -1536:]
            with torch.no_grad():
                if is_seq2seq:
                    dec_ids = torch.full(
                        (input_ids.size(0), 1),
                        decoder_start,
                        dtype=torch.long,
                        device=model.device,
                    )
                    logits = model(input_ids=input_ids, decoder_input_ids=dec_ids).logits[:, -1]
                else:
                    logits = model(input_ids=input_ids).logits[:, -1]
            probs = F.softmax(
                logits[..., option_indices], dim=-1,
            ).detach().to(torch.float32).cpu().numpy()
            probs = probs.reshape(input_ids.size(0), 2, len(option_ids)).sum(axis=1)[0]
            return probs.tolist()

        all_probs = []
        for input_text in input_texts:
            all_probs.append(_score_input_text(input_text))

        identity_perm = tuple(range(num_options))
        try:
            identity_idx = perm_tuples.index(identity_perm)
        except ValueError:
            identity_idx = 0
        base_probs = np.asarray(all_probs[identity_idx], dtype=np.float64)
        top_order = np.argsort(base_probs)[::-1]
        top1_slot = int(top_order[0]) if top_order.size > 0 else 0
        top2_slot = int(top_order[1]) if top_order.size > 1 else int(top1_slot)
        swap_perm = list(identity_perm)
        swap_perm[top1_slot], swap_perm[top2_slot] = swap_perm[top2_slot], swap_perm[top1_slot]
        swap_prompt = None
        swap_probs = None
        swap_sampled = None
        if top1_slot != top2_slot:
            swap_options = [options[idx] for idx in swap_perm]
            if "Question: " in eval_sample and "\nOptions:\n" in eval_sample:
                question_text = eval_sample.split("Question: ", 1)[1].split("\nOptions:\n", 1)[0]
            else:
                question_text = eval_sample
            swap_prompt = [
                sys_msg,
                _rebuild_user_prompt_like(eval_sample, question_text, swap_options),
            ]
            swap_input_text = sys_msg + '\n\n'
            if num_few_shot > 0:
                for s in few_shot_samples[:num_few_shot]:
                    swap_input_text += s + '\n\n'
            swap_input_text += swap_prompt[1]
            if not bpe_has_space_prefix:
                swap_input_text += ' '
            # The production method now uses the Latin-square schedule below.
            # Keep the exact top1/top2 swap metadata for compatibility, but
            # avoid spending an extra forward pass that is no longer consumed.

        if "Question: " in eval_sample and "\nOptions:\n" in eval_sample:
            question_text = eval_sample.split("Question: ", 1)[1].split("\nOptions:\n", 1)[0]
        else:
            question_text = eval_sample
        latin_perms = _build_toprank_latin_perms(base_probs, num_options)
        latin_probs = []
        latin_prompts = []
        perm_to_existing_idx = {tuple(p): idx for idx, p in enumerate(perm_tuples)}
        for latin_perm in latin_perms:
            latin_perm = tuple(int(x) for x in latin_perm)
            existing_idx = perm_to_existing_idx.get(latin_perm)
            if existing_idx is not None:
                latin_probs.append(all_probs[existing_idx])
                latin_prompts.append(None)
                continue
            latin_options = [options[idx] for idx in latin_perm]
            latin_prompt = [
                sys_msg,
                _rebuild_user_prompt_like(eval_sample, question_text, latin_options),
            ]
            latin_input_text = sys_msg + '\n\n'
            if num_few_shot > 0:
                for s in few_shot_samples[:num_few_shot]:
                    latin_input_text += s + '\n\n'
            latin_input_text += latin_prompt[1]
            if not bpe_has_space_prefix:
                latin_input_text += ' '
            latin_probs.append(_score_input_text(latin_input_text))
            latin_prompts.append(latin_prompt[1])

        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_texts[0],
                'prompts': input_texts,
                'options': options,
                'probs': all_probs,
                'perm_tuples': [list(map(int, p)) for p in perm_tuples],
                'swap_top12_perm': list(map(int, swap_perm)),
                'swap_top12_slots': [int(top1_slot), int(top2_slot)],
                'swap_top12_prompt': swap_prompt[1] if swap_prompt is not None else None,
                'swap_top12_probs': swap_probs,
                'swap_top12_sampled': swap_sampled,
                'latin_toprank_perms': [list(map(int, p)) for p in latin_perms],
                'latin_toprank_probs': latin_probs,
                'latin_toprank_prompts': latin_prompts,
                'ideal': ideal,
            },
        }
        return result
    return eval_fn
