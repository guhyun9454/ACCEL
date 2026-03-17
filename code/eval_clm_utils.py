import os
import sys
import random
import copy
import json
import argparse
import logging
import unicodedata
from tqdm import tqdm
from typing import List, Optional
from functools import partial
from itertools import permutations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    _norm,
    shuffle_options_with_ids,
    shuffle_option_texts,
    ids_in_positions_to_permuted_indices,
    permuted_indices_to_ids_in_positions,
    cyclic_shift,
    move_answer,
    cycle_options,
)

logger = logging.getLogger(__name__)

_DASH_CHARS = {
    '\u2010',  # hyphen
    '\u2011',  # non-breaking hyphen
    '\u2012',  # figure dash
    '\u2013',  # en dash
    '\u2014',  # em dash
    '\u2015',  # horizontal bar
    '\u2212',  # minus sign
    '\ufe58',  # small em dash
    '\ufe63',  # small hyphen-minus
    '\uff0d',  # fullwidth hyphen-minus
}
_ZERO_WIDTH_CHARS = {
    '\u200b',  # zero width space
    '\u200c',  # zero width non-joiner
    '\u200d',  # zero width joiner
    '\ufeff',  # zero width no-break space
    '\u2060',  # word joiner
}


def _normalize_model_path(model_path: str) -> str:
    normalized = unicodedata.normalize("NFKC", model_path)
    normalized = ''.join('-' if ch in _DASH_CHARS else ch for ch in normalized)
    normalized = ''.join(ch for ch in normalized if ch not in _ZERO_WIDTH_CHARS)
    return normalized.strip()


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
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            "Root directory that contains data_<task> folders (e.g., data_mmlu, data_arc, ...). "
            "If not provided, the code will try common locations such as ./data_<task>, ./code/data_<task>, "
            "and ../LLM-MCQ-Bias_data/data_<task>."
        ),
    )
    parser.add_argument("--eval_names", type=str, nargs='+', default=[],
                        help='eval tasks and settings')
    args = parser.parse_args()

    original_model_path = args.pretrained_model_path
    args.pretrained_model_path = _normalize_model_path(args.pretrained_model_path)
    if args.pretrained_model_path != original_model_path:
        logger.warning(
            "Normalized pretrained_model_path from %r to %r",
            original_model_path,
            args.pretrained_model_path,
        )

    model_path = args.pretrained_model_path.rstrip('/\\')
    args.model_name = model_path.replace('\\', '/').split('/')[-1] if model_path else args.pretrained_model_path

    for eval_name in args.eval_names:
        eval_args = eval_name.split(',')
        task = eval_args[0]
        if task not in [
            'mmlu', 'arc', 'csqa', 'hellaswag',
        ]:
            raise ValueError(f"Unknown task: {task}")

        num_few_shot = int(eval_args[1])

        setting = eval_args[2] if len(eval_args) > 2 else None
        if setting is not None and not (
            setting in [
                'noid',
                'perm', 'cyclic',
                'shuffle_both',
                # Ablations for disentangling token/position effects
                'swap_text', 'swap_id',
                'cyclic_swap_text', 'cyclic_swap_id',
            ] or (setting.startswith('move') and setting[-1] in ['a', 'b', 'c', 'd'])
        ):
            raise ValueError(f"Unknown setting: {setting}")

    return args


def _resolve_data_path(task: str, data_root: Optional[str] = None) -> str:
    """Resolve the folder path that contains dev/ and test/ for the given task.

    We support multiple repo layouts:
      1) Running from `code/` with local folders: ./data_<task>/dev, ./data_<task>/test
      2) Running from repo root: ./code/data_<task>/dev, ./code/data_<task>/test
      3) Cloning the official data repo under repo root: ./LLM-MCQ-Bias_data/data_<task>/...
      4) User-provided --data_root that contains data_<task>/...
    """
    candidates: List[str] = []
    if data_root:
        candidates.extend([
            os.path.join(data_root, f'data_{task}'),
            os.path.join(data_root, task),
        ])
    candidates.extend([
        f'data_{task}',
        os.path.join('code', f'data_{task}'),
        os.path.join('..', f'data_{task}'),
        os.path.join('..', 'LLM-MCQ-Bias_data', f'data_{task}'),
        os.path.join('..', 'LLM-MCQ-Bias_data', task),
        os.path.join('..', '..', 'LLM-MCQ-Bias_data', f'data_{task}'),
        os.path.join('..', '..', 'LLM-MCQ-Bias_data', task),
    ])

    for cand in candidates:
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, 'test')):
            return cand

    msg = (
        f"Cannot find dataset folder for task='{task}'. Tried:\n" +
        "\n".join([f"  - {c}" for c in candidates]) +
        "\n\n" +
        "Fix: pass --data_root to point to the directory that contains data_<task>/dev and data_<task>/test, "
        "or create a symlink named data_<task> in your working directory."
    )
    raise FileNotFoundError(msg)


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
    args.save_path = save_path
    os.makedirs(args.save_path, exist_ok=True)

    option_ids = list('ABCD')
    option_ids_header = list('ABCD')
    if task in ['csqa']:
        option_ids = list('ABCDE')
        option_ids_header = list('ABCDE')

    data_path = _resolve_data_path(task, getattr(args, 'data_root', None))
    subjects = sorted([f.split("_test.csv")[0]
                       for f in os.listdir(f'{data_path}/test') if "_test.csv" in f])

    # sys_msg
    if 'mmlu' in task:
        sys_msg = 'The following are multiple choice questions about {}.'
    else: # task in ['arc', 'tqa']
        sys_msg = 'The following are multiple choice questions.'

    sys_msg += ' You should directly answer the question by choosing the correct option.'

    # create_user_prompt
    def create_user_prompt(question: str, options: List[str], display_option_ids: List[str] = None):
        """Create the user prompt.

        `display_option_ids` controls which IDs are printed next to the options
        *in the shown order*.
        """
        if display_option_ids is None:
            display_option_ids = option_ids

        if setting in ['noid']:
            user_prompt = f"Question: {question.strip()}\nOptions:\n" + \
                "\n".join([f"{answer}".strip()
                           for _, answer in zip(option_ids, options)]) + \
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
                           for option_id, answer in zip(display_option_ids, options)]) + \
                "\nAnswer:"
        return user_prompt

    def _apply_single_prompt_transform(options: List[str]):
        """Apply single-prompt manipulations and return prompt components.

        Returns:
            shown_options: List[str]
            ids_in_positions: List[str] (IDs shown next to each shown option)
            permuted_indices: Tuple[int] mapping output label index -> original option index
        """
        n = len(options)
        if setting in ['swap_text']:
            permuted_indices_list, shown_options = shuffle_option_texts(options)
            ids_in_positions = list(option_ids)
            permuted_indices = tuple(permuted_indices_list)
            return shown_options, ids_in_positions, permuted_indices
        if setting in ['swap_id', 'cyclic_swap_id']:
            # IMPORTANT: to make "swap_text" vs "swap_id" comparable, we use the *same*
            # underlying label->option mapping (permutation) as swap_text.
            #
            # swap_text: texts move, IDs stay in A/B/C/D order.
            # swap_id:   texts stay, IDs move so that label->option mapping is identical.
            base_perm, _ = shuffle_option_texts(options)
            base_perm = tuple(int(x) for x in base_perm)
            ids_in_positions = permuted_indices_to_ids_in_positions(option_ids, base_perm)
            shown_options = list(options)
            permuted_indices = base_perm
            return shown_options, ids_in_positions, permuted_indices
        # default / shuffle_both / noid
        return list(options), list(option_ids), tuple(range(n))

    # prepare_few_shot_samples
    def prepare_few_shot_samples(subject):
        df = pd.read_csv(
            f'{data_path}/dev/{subject}_dev.csv',
            names=("Question", *option_ids_header, "Answer"),
            dtype=str,
            encoding='utf-8',
        )

        def _make_one(x):
            raw_options = [x[e] for e in option_ids_header]
            shown_options, ids_in_positions, permuted_indices = _apply_single_prompt_transform(raw_options)
            prompt = create_user_prompt(x["Question"], shown_options, display_option_ids=ids_in_positions)

            # The *canonical* correct option index is determined by the dataset label.
            correct_original_idx = option_ids_header.index(x["Answer"])
            if setting in ['noid']:
                # Option-ID removal: answer is the option text itself.
                answer = str(x[x["Answer"]])
            else:
                # Find which output ID corresponds to the canonical correct option.
                correct_label_index = permuted_indices.index(correct_original_idx)
                answer = option_ids[correct_label_index]
            return prompt + ' ' + answer

        few_shot_samples = df.apply(_make_one, axis=1).to_list()
        return few_shot_samples

    # prepare_eval_samples
    def prepare_eval_samples(subject):
        df = pd.read_csv(
            f'{data_path}/test/{subject}_test.csv',
            names=("Question", *option_ids_header, "Answer"),
            dtype=str,
            encoding='utf-8',
        )

        if setting is not None and setting.startswith('move'):
            df = df.apply(lambda x: move_answer(x, moved_answer), axis=1)


        def _make_one_base(x):
            raw_options = [x[e] for e in option_ids_header]
            shown_options, ids_in_positions, permuted_indices = _apply_single_prompt_transform(raw_options)
            inp = [
                sys_msg.format(subject.replace('_', ' ')),
                create_user_prompt(x["Question"], shown_options, display_option_ids=ids_in_positions),
            ]
            meta = {
                'permuted_indices': permuted_indices,
                'ids_in_positions': ids_in_positions,
                'shown_options': shown_options,
            }
            return inp, [str(e) for e in raw_options], meta

        def _make_one_perm(x):
            raw_options = [x[e] for e in option_ids_header]
            probing_inputs = []
            permuted_indices_list = []
            for permuted_idx in list(sorted(permutations(range(len(raw_options))))):
                permuted_options = [raw_options[i] for i in permuted_idx]
                probing_inputs.append([
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], permuted_options),
                ])
                permuted_indices_list.append(tuple(permuted_idx))
            meta = {
                'permuted_indices_list': permuted_indices_list,
            }
            return probing_inputs, [str(e) for e in raw_options], meta

        def _make_one_cyclic(x):
            raw_options = [x[e] for e in option_ids_header]
            probing_inputs = []
            permuted_indices_list = []
            n = len(raw_options)
            base = list(range(n))
            for k, cycled_options in enumerate(cycle_options(raw_options)):
                probing_inputs.append([
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], cycled_options),
                ])
                permuted_indices_list.append(tuple(cyclic_shift(base, k)))
            meta = {
                'permuted_indices_list': permuted_indices_list,
            }
            return probing_inputs, [str(e) for e in raw_options], meta

        def _make_one_cyclic_swap_text(x):
            raw_options = [x[e] for e in option_ids_header]
            n = len(raw_options)
            base = list(range(n))  # identity: original dataset order
            probing_inputs = []
            permuted_indices_list = []
            for k in range(n):
                # Cyclic shift texts only; IDs stay as ABCD
                perm_k = cyclic_shift(base, k)
                opts_k = [raw_options[i] for i in perm_k]
                probing_inputs.append([
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], opts_k),
                ])
                # permuted_indices[label_idx] = original_option_idx
                # k=0: (0,1,2,3) identity, k=1: (1,2,3,0) etc.
                permuted_indices_list.append(tuple(perm_k))
            meta = {
                'permuted_indices_list': permuted_indices_list,
            }
            return probing_inputs, [str(e) for e in raw_options], meta

        def _make_one_cyclic_swap_id(x):
            raw_options = [x[e] for e in option_ids_header]
            # Use the same base permutation as cyclic_swap_text for fair comparison.
            base_perm, _ = shuffle_option_texts(raw_options)
            p0 = list(int(v) for v in base_perm)

            probing_inputs = []
            permuted_indices_list = []
            for k in range(len(raw_options)):
                pk = cyclic_shift(p0, k)
                # Convert label->option mapping pk into IDs shown per position.
                ids_in_positions = permuted_indices_to_ids_in_positions(option_ids, pk)
                probing_inputs.append([
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], raw_options, display_option_ids=ids_in_positions),
                ])
                permuted_indices_list.append(tuple(pk))
            meta = {
                'permuted_indices_list': permuted_indices_list,
                'base_permuted_indices': tuple(p0),
            }
            return probing_inputs, [str(e) for e in raw_options], meta

        if setting in ['perm']:
            triples = df.apply(_make_one_perm, axis=1).to_list()
            inputs = [t[0] for t in triples]
            options = [t[1] for t in triples]
            metas = [t[2] for t in triples]
        elif setting in ['cyclic']:
            triples = df.apply(_make_one_cyclic, axis=1).to_list()
            inputs = [t[0] for t in triples]
            options = [t[1] for t in triples]
            metas = [t[2] for t in triples]
        elif setting in ['cyclic_swap_text']:
            triples = df.apply(_make_one_cyclic_swap_text, axis=1).to_list()
            inputs = [t[0] for t in triples]
            options = [t[1] for t in triples]
            metas = [t[2] for t in triples]
        elif setting in ['cyclic_swap_id']:
            triples = df.apply(_make_one_cyclic_swap_id, axis=1).to_list()
            inputs = [t[0] for t in triples]
            options = [t[1] for t in triples]
            metas = [t[2] for t in triples]
        else:
            triples = df.apply(_make_one_base, axis=1).to_list()
            inputs = [t[0] for t in triples]
            options = [t[1] for t in triples]
            metas = [t[2] for t in triples]

        ideals = df.apply(lambda x: option_ids[option_ids_header.index(x["Answer"])], axis=1).to_list()
        return list(zip(inputs, options, ideals, metas))

    # prepare_eval_fn
    if setting in ['noid']:
        prepare_eval_fn = partial(
            prepare_eval_fn_noid,
            num_few_shot=num_few_shot,
            option_ids=option_ids,
            is_encoder_decoder=bool(getattr(args, 'is_encoder_decoder', False)),
        )
    elif setting in ['perm', 'cyclic', 'cyclic_swap_text', 'cyclic_swap_id']:
        prepare_eval_fn = partial(
            prepare_eval_fn_perm,
            num_few_shot=num_few_shot,
            option_ids=option_ids,
            is_encoder_decoder=bool(getattr(args, 'is_encoder_decoder', False)),
        )
    else:
        prepare_eval_fn = partial(
            prepare_eval_fn_base,
            num_few_shot=num_few_shot,
            option_ids=option_ids,
            is_encoder_decoder=bool(getattr(args, 'is_encoder_decoder', False)),
        )

    return subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn


def _get_model_device(model):
    try:
        return model.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device('cpu')


def _build_input_text(sys_msg, eval_sample, few_shot_samples, num_few_shot):
    input_text = sys_msg + '\n\n'
    if num_few_shot > 0:
        for s in few_shot_samples[:num_few_shot]:
            input_text += s + '\n\n'
    input_text += eval_sample
    return input_text


def _probs_from_losses(losses):
    nll = -np.asarray(losses, dtype=np.float64)
    probs = np.exp(nll - np.max(nll))
    probs = probs / (probs.sum() + 1e-10)
    return probs.astype(np.float32)


def _score_clm_label_candidates(model, toker, input_text, option_ids):
    device = _get_model_device(model)
    input_ids = toker(input_text, return_tensors="pt").input_ids.to(device)
    input_ids = input_ids[..., -1536:]
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
        ).logits[:, -1].view(-1)

    option_indices = [toker(f': {e}').input_ids[-1] for e in option_ids] + \
        [toker(f':{e}').input_ids[-1] for e in option_ids]
    probs = F.softmax(
        logits[..., option_indices], dim=-1
    ).detach().cpu().to(torch.float32).numpy()
    probs = probs.reshape(2, len(option_ids)).sum(axis=0)
    return probs


def _score_clm_text_candidates(model, toker, input_text, options):
    device = _get_model_device(model)
    prefix_input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids

    losses = []
    lengths = []
    for option in options:
        prefix_and_option_text = input_text + ' ' + option.strip()
        input_ids = toker(prefix_and_option_text, truncation=False, return_tensors="pt").input_ids.to(device)
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

    probs = _probs_from_losses(losses)
    return losses, lengths, probs


def _score_seq2seq_target_nll(model, toker, input_text, target_text):
    device = _get_model_device(model)

    encoded = toker(input_text, truncation=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)[..., -1536:]
    attention_mask = encoded.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)[..., -1536:]

    target_ids = toker(str(target_text), truncation=False, return_tensors="pt").input_ids.to(device)
    target_ids = target_ids[..., :256]
    labels = target_ids.clone()
    if toker.pad_token_id is not None:
        labels[labels == toker.pad_token_id] = -100

    with torch.no_grad():
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss.detach().to(torch.float32).cpu().item()

    length = int((labels != -100).sum().item())
    if length <= 0:
        length = int(labels.numel())
    return float(loss), length


def _score_seq2seq_label_candidates(model, toker, input_text, option_ids):
    losses = []
    lengths = []
    for label in option_ids:
        variant_losses = []
        variant_lengths = []
        for target in (label, f' {label}'):
            loss, length = _score_seq2seq_target_nll(model, toker, input_text, target)
            variant_losses.append(loss)
            variant_lengths.append(length)
        best_idx = int(np.argmin(variant_losses))
        losses.append(float(variant_losses[best_idx]))
        lengths.append(int(variant_lengths[best_idx]))

    probs = _probs_from_losses(losses)
    return losses, lengths, probs


def _score_seq2seq_text_candidates(model, toker, input_text, options):
    losses = []
    lengths = []
    for option in options:
        target = str(option).strip()
        loss, length = _score_seq2seq_target_nll(model, toker, input_text, target)
        losses.append(loss)
        lengths.append(length)
    probs = _probs_from_losses(losses)
    return losses, lengths, probs


def prepare_eval_fn_base(model, toker, few_shot_samples, num_few_shot, option_ids, is_encoder_decoder=False):
    bpe_has_space_prefix = None
    if not is_encoder_decoder:
        bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]

    def eval_fn(sample, rng: random.Random):
        idx, (input, options, ideal, meta) = sample
        permuted_indices = tuple(meta.get('permuted_indices', tuple(range(len(option_ids)))))
        sys_msg, eval_sample = input.copy()
        input_text = _build_input_text(sys_msg, eval_sample, few_shot_samples, num_few_shot)
        if (not is_encoder_decoder) and (not bpe_has_space_prefix):
            input_text += ' '

        if is_encoder_decoder:
            _, _, probs = _score_seq2seq_label_candidates(model, toker, input_text, option_ids)
        else:
            probs = _score_clm_label_candidates(model, toker, input_text, option_ids)

        sampled_label_index = int(np.argmax(probs))
        sampled = option_ids[sampled_label_index]

        # Evaluate correctness in the *canonical* option space.
        # permuted_indices maps output label index -> original option index.
        ideal_original_idx = option_ids.index(ideal)
        pred_original_idx = permuted_indices[sampled_label_index]
        correct = (pred_original_idx == ideal_original_idx)
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'meta': meta,
                'probs': probs.tolist(),
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_noid(model, toker, few_shot_samples, num_few_shot, option_ids, is_encoder_decoder=False):
    toker.padding_side = 'right'

    def eval_fn(sample, rng: random.Random):
        idx, (input, options, ideal, meta) = sample
        permuted_indices = tuple(meta.get('permuted_indices', tuple(range(len(option_ids)))))
        sys_msg, eval_sample = input.copy()
        input_text = _build_input_text(sys_msg, eval_sample, few_shot_samples, num_few_shot)

        if is_encoder_decoder:
            losses, lengths, probs = _score_seq2seq_text_candidates(model, toker, input_text, options)
        else:
            losses, lengths, probs = _score_clm_text_candidates(model, toker, input_text, options)

        sampled_pos = int(np.argmin(losses))
        sampled = option_ids[sampled_pos]

        ideal_original_idx = option_ids.index(ideal)
        pred_original_idx = permuted_indices[sampled_pos]
        correct = (pred_original_idx == ideal_original_idx)
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'meta': meta,
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


def prepare_eval_fn_perm(model, toker, few_shot_samples, num_few_shot, option_ids, is_encoder_decoder=False):
    toker.padding_side = 'left'
    bpe_has_space_prefix = None
    if not is_encoder_decoder:
        bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]

    def eval_fn(sample, rng: random.Random):
        idx, (probing_inputs, options, ideal, meta) = sample

        input_texts = []
        for probing_input in probing_inputs:
            sys_msg, eval_sample = probing_input.copy()
            input_text = _build_input_text(sys_msg, eval_sample, few_shot_samples, num_few_shot)
            if (not is_encoder_decoder) and (not bpe_has_space_prefix):
                input_text += ' '
            input_texts.append(input_text)

        all_probs = []
        for input_text in input_texts:
            if is_encoder_decoder:
                _, _, probs = _score_seq2seq_label_candidates(model, toker, input_text, option_ids)
            else:
                probs = _score_clm_label_candidates(model, toker, input_text, option_ids)
            all_probs.append(probs.tolist())

        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_texts[0],
                'options': options,
                'meta': meta,
                'probs': all_probs,
                'ideal': ideal,
            },
        }
        return result
    return eval_fn