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

import math
from itertools import permutations
import matplotlib.pyplot as plt

import pynvml
pynvml.nvmlInit()

logger = logging.getLogger(__name__)


def logging_cuda_memory_usage():
    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info("GPU {}: {:.2f} GB / {:.2f} GB".format(i, meminfo.used / 1024 ** 3, meminfo.total / 1024 ** 3))


def _rotations(k: int):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(probs_seq, permuted_indices, k: int):
    """
    Map letter-indexed probs from each permutation to content-indexed probs,
    then average across permutations.
    probs_seq: list of length (#perms), each a length-k list of probs for letters
    permuted_indices: list of tuples, permutation p: letter j corresponds to content index p[j]
    """
    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            agg[p[j]] += letter_probs[j]
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


def _read_results_file(file_path):
    try:
        lines = [json.loads(line) for line in open(file_path)]
        lines = [e for e in lines if e.get('type') == 'result']
        lines = sorted(lines, key=lambda x: int(x['data']['idx']))
        return lines
    except FileNotFoundError:
        return None
    except Exception:
        return None


def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    args = parse_arguments()
    if len(args.eval_names) == 0:
        exit()

    # Optional: W&B init
    wandb_run = None
    if getattr(args, 'wandb', False):
        try:
            import wandb
            project = args.wandb_project
            run_name = args.wandb_run_name or f"{args.model_name}-{args.eval_names[0]}"
            wandb_run = wandb.init(entity="capde", project=project, name=run_name, config={
                "pretrained_model_path": args.pretrained_model_path,
                "model_name": args.model_name,
                "eval_names": args.eval_names,
                "option_id_set": args.option_id_set,
            })
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            wandb_run = None

    toker = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
        cache_dir=args.cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        device_map='auto',
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        cache_dir=args.cache_dir,
    )
    logging_cuda_memory_usage()

    printed_example = False

    # Comparative mode: run with two option ID sets and compute matching ratio and flip stats
    if getattr(args, 'option_id_sets', None) and len(args.option_id_sets) == 2:
        from types import SimpleNamespace

        id_set_a, id_set_b = args.option_id_sets[0], args.option_id_sets[1]

        overall_total = 0
        overall_matches = 0
        overall_c2i = 0
        overall_i2c = 0
        overall_both_correct = 0
        overall_both_incorrect = 0

        for eval_name in args.eval_names[::1]:
            eval_args = eval_name.split(',')
            setting = eval_args[2] if len(eval_args) > 2 else None
            if setting in ['perm', 'cyclic']:
                logger.info(_orange(f"Skipping compare for setting '{setting}' (not supported)."))
                continue

            args_a = SimpleNamespace(pretrained_model_path=args.pretrained_model_path, model_name=args.model_name, option_id_set=id_set_a)
            args_b = SimpleNamespace(pretrained_model_path=args.pretrained_model_path, model_name=args.model_name, option_id_set=id_set_b)

            (subjects_a, prepare_fewshot_a, prepare_eval_samples_a, prepare_eval_fn_a) = prepare_eval(args_a, eval_name)
            (subjects_b, prepare_fewshot_b, prepare_eval_samples_b, prepare_eval_fn_b) = prepare_eval(args_b, eval_name)

            assert subjects_a == subjects_b, "Subjects mismatch across option ID sets"

            for subject in subjects_a[::1]:
                logger.info(_blue(f"Preparing (compare): {subject}"))
                few_a = prepare_fewshot_a(subject)
                few_b = prepare_fewshot_b(subject)
                eval_samples_a = prepare_eval_samples_a(subject)
                eval_samples_b = prepare_eval_samples_b(subject)
                eval_fn_a = prepare_eval_fn_a(model, toker, few_a)
                eval_fn_b = prepare_eval_fn_b(model, toker, few_b)

                # Prompt example (only once, from set A)
                if getattr(args, 'print_prompt_example', False) and not printed_example and len(eval_samples_a) > 0:
                    try:
                        first_input, _first_options, _first_ideal = eval_samples_a[0]

                        def build_input_text(pair):
                            sys_msg, eval_sample = pair
                            text = sys_msg + '\n\n'
                            if args_a.num_few_shot > 0:
                                for s in few_a[:args_a.num_few_shot]:
                                    text += s + '\n\n'
                            text += eval_sample
                            return text

                        if isinstance(first_input, list) and len(first_input) > 0 and isinstance(first_input[0], list):
                            input_text = build_input_text(first_input[0])
                            bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]
                            if not bpe_has_space_prefix:
                                input_text += ' '
                        else:
                            input_text = build_input_text(first_input)
                            bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]
                            if not bpe_has_space_prefix:
                                input_text += ' '
                        logger.info(_purple("==== Prompt example ===="))
                        logger.info("\n" + input_text)
                        logger.info(_purple("==== End prompt example ===="))
                        printed_example = True
                    except Exception as e:
                        logger.warning(f"Failed to build prompt example: {e}")

                # Try cached results first
                path_a = f'{args_a.save_path}/{subject}.jsonl'
                path_b = f'{args_b.save_path}/{subject}.jsonl'
                results_a = None if getattr(args, 'force', False) else _read_results_file(path_a)
                results_b = None if getattr(args, 'force', False) else _read_results_file(path_b)

                if results_a is not None:
                    logger.info(_blue(f"Using cached results (A): {path_a}"))
                else:
                    logger.info(_blue(f"Run started (A): {subject} [{id_set_a}]"))
                    max_samples = 100 if getattr(args, 'test', False) else None
                    results_a = eval_all_samples(
                        eval_fn_a, eval_samples_a,
                        name=f'{args_a.task},{args_a.num_few_shot},{args_a.setting},{subject},{id_set_a}',
                        threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
                        max_num_samples=max_samples,
                    )
                    save_results(path_a, results_a, metrics=None)
                    logger.info(f"Results saved (A): {subject}")

                if results_b is not None:
                    logger.info(_blue(f"Using cached results (B): {path_b}"))
                else:
                    logger.info(_blue(f"Run started (B): {subject} [{id_set_b}]"))
                    max_samples = 100 if getattr(args, 'test', False) else None
                    results_b = eval_all_samples(
                        eval_fn_b, eval_samples_b,
                        name=f'{args_b.task},{args_b.num_few_shot},{args_b.setting},{subject},{id_set_b}',
                        threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
                        max_num_samples=max_samples,
                    )
                    save_results(path_b, results_b, metrics=None)
                    logger.info(f"Results saved (B): {subject}")

                # Align by idx
                map_a = {int(r['data']['idx']): r for r in results_a if r['type'] == 'result'}
                map_b = {int(r['data']['idx']): r for r in results_b if r['type'] == 'result'}
                common = sorted(set(map_a.keys()) & set(map_b.keys()))

                matches = 0
                total = 0
                c2i = 0
                i2c = 0
                both_correct = 0
                both_incorrect = 0

                for idx in common:
                    ra = map_a[idx]['data']
                    rb = map_b[idx]['data']
                    pa = int(np.argmax(np.array(ra['probs'])))
                    pb = int(np.argmax(np.array(rb['probs'])))
                    if pa == pb:
                        matches += 1
                    ca = bool(ra['correct'])
                    cb = bool(rb['correct'])
                    if ca and not cb:
                        c2i += 1
                    elif (not ca) and cb:
                        i2c += 1
                    elif ca and cb:
                        both_correct += 1
                    elif (not ca) and (not cb):
                        both_incorrect += 1
                    total += 1

                overall_total += total
                overall_matches += matches
                overall_c2i += c2i
                overall_i2c += i2c
                overall_both_correct += both_correct
                overall_both_incorrect += both_incorrect

                acc_a = get_accuracy(results_a)
                acc_b = get_accuracy(results_b)
                mr = (matches / total) if total > 0 else float('nan')
                logger.info(_purple(f"[{subject}] matching_ratio={mr:.4f} (matches/total={matches}/{total})"))
                logger.info(f"accuracy_A({id_set_a})={acc_a:.4f}, accuracy_B({id_set_b})={acc_b:.4f}")
                logger.info(f"flip correct->incorrect={c2i}, incorrect->correct={i2c}, both_correct={both_correct}, both_incorrect={both_incorrect}")

                gc.collect()
                torch.cuda.empty_cache()

        if overall_total > 0:
            overall_mr = overall_matches / overall_total
            logger.info(_purple(
                "==== Overall compare summary ====\n"
                f"matching_ratio={overall_mr:.4f} (matches/total={overall_matches}/{overall_total})\n"
                f"correct->incorrect={overall_c2i}, incorrect->correct={overall_i2c}, "
                f"both_correct={overall_both_correct}, both_incorrect={overall_both_incorrect}"
            ))
        return

    # Single-run mode
    for eval_name in args.eval_names[::1]:
        (
            subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn
        ) = prepare_eval(args, eval_name)

        for subject in subjects[::1]:
            cached_path = f'{args.save_path}/{subject}.jsonl'
            use_cached = (not getattr(args, 'force', False)) and os.path.exists(cached_path)

            logger.info(_blue(f"Preparing: {subject}"))
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = prepare_eval_fn(model, toker, few_shot_samples)

            # Print one prompt example if requested
            if getattr(args, 'print_prompt_example', False) and not printed_example and len(eval_samples) > 0:
                try:
                    first_input, _first_options, _first_ideal = eval_samples[0]

                    def build_input_text(pair):
                        sys_msg, eval_sample = pair
                        text = sys_msg + '\n\n'
                        if args.num_few_shot > 0:
                            for s in few_shot_samples[:args.num_few_shot]:
                                text += s + '\n\n'
                        text += eval_sample
                        return text

                    if isinstance(first_input, list) and len(first_input) > 0 and isinstance(first_input[0], list):
                        input_text = build_input_text(first_input[0])
                        bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]
                        if not bpe_has_space_prefix:
                            input_text += ' '
                    else:
                        input_text = build_input_text(first_input)
                        if args.setting not in ['noid']:
                            bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]
                            if not bpe_has_space_prefix:
                                input_text += ' '

                    logger.info(_purple("==== Prompt example ===="))
                    logger.info("\n" + input_text)
                    logger.info(_purple("==== End prompt example ===="))
                    printed_example = True
                except Exception as e:
                    logger.warning(f"Failed to build prompt example: {e}")

            if use_cached:
                logger.info(_blue(f"Using cached results: {cached_path}"))
                results = _read_results_file(cached_path) or []
                ran_eval = False
            else:
                logger.info(_blue(f"Run started: {subject}"))
                max_samples = 100 if getattr(args, 'test', False) else None
                results = eval_all_samples(
                    eval_fn, eval_samples,
                    name=f'{args.task},{args.num_few_shot},{args.setting},{subject}',
                    threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
                    max_num_samples=max_samples,
                )
                gc.collect()
                torch.cuda.empty_cache()
                ran_eval = True

            metrics = None
            if len(results) > 0:
                if args.setting in ['perm', 'full', 'cyclic']:
                    # Ensemble over permutations/cycles with content mapping
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        k_guess = len(results[0]['data']['options'])
                        option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                    k = len(option_ids)

                    if args.setting in ['perm', 'full']:
                        perm_list = list(sorted(permutations(range(k))))
                    else:
                        perm_list = _rotations(k)

                    total = 0
                    corrects = 0
                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data.get('probs', None)
                        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                            continue
                        agg = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                        pred_letter = option_ids[int(np.argmax(agg))]
                        if pred_letter == data['ideal']:
                            corrects += 1
                        total += 1
                    acc = (corrects / total) if total > 0 else float('nan')
                    metrics = {'type': 'metric', 'data': {'accuracy': acc}}
                    logger.info(_purple(f"==== Ensemble report ({args.setting}) ===="))
                    logger.info(f"accuracy: {acc:.4f}")
                else:
                    metrics = {'type': 'metric', 'data': {}}
                    metrics['data']['accuracy'] = get_accuracy(results)
                    metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)
                    logger.info("Final report:")
                    for key, value in metrics['data'].items():
                        logger.info(f"{key}: {value}")
            logger.info(_orange(f"Run completed: {subject}"))

            if not use_cached:
                save_results(f'{args.save_path}/{subject}.jsonl', results, metrics)
                logger.info(f"Results saved: {subject}")

            # Derive cyclic and default (base) outputs automatically from FULL permutation runs
            if args.setting == 'full':
                try:
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        k_guess = len(results[0]['data']['options']) if len(results) > 0 and results[0]['type'] == 'result' else 4
                        option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                    k = len(option_ids)

                    perm_list = list(sorted(permutations(range(k))))
                    identity_idx = perm_list.index(tuple(range(k)))
                    cyclic_indices = [perm_list.index(tuple((i + s) % k for i in range(k))) for s in range(k)]

                    cyclic_results = []
                    base_results = []
                    full_total = 0
                    full_corrects = 0
                    cyclic_total = 0
                    cyclic_corrects = 0
                    base_correct_list = []
                    cyclic_correct_list = []
                    full_correct_list = []

                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data['probs']
                        if not isinstance(probs_seq, list) or len(probs_seq) <= identity_idx:
                            continue

                        # Cyclic subset
                        cyclic_probs = [probs_seq[idx] for idx in cyclic_indices]
                        cyclic_results.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': cyclic_probs,
                                'ideal': data['ideal'],
                            },
                        })
                        agg_cyc = _aggregate_probs_over_permutations(
                            cyclic_probs,
                            [tuple((i + s) % k for i in range(k)) for s in range(k)],
                            k,
                        )
                        pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                        correct_cyc = (pred_cyc == data['ideal'])
                        cyclic_correct_list.append(correct_cyc)
                        if correct_cyc:
                            cyclic_corrects += 1
                        cyclic_total += 1

                        # Default (identity)
                        base_probs = probs_seq[identity_idx]
                        sampled = option_ids[int(np.argmax(np.array(base_probs)))]
                        correct = (sampled == data['ideal'])
                        base_correct_list.append(correct)
                        base_results.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': base_probs,
                                'sampled': sampled,
                                'ideal': data['ideal'],
                                'correct': correct,
                            },
                        })

                        # Full ensemble
                        agg_full = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                        pred_full = option_ids[int(np.argmax(agg_full))]
                        correct_full = (pred_full == data['ideal'])
                        full_correct_list.append(correct_full)
                        if correct_full:
                            full_corrects += 1
                        full_total += 1

                    # Save cyclic
                    cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                    if getattr(args, 'option_id_set', None):
                        cyclic_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(cyclic_save_path, exist_ok=True)
                    cyclic_metrics = None
                    if cyclic_total > 0:
                        cyclic_acc = cyclic_corrects / cyclic_total
                        cyclic_metrics = {'type': 'metric', 'data': {'accuracy': cyclic_acc}}
                        logger.info(_purple(f"[{subject}] Cyclic ensemble accuracy: {cyclic_acc:.4f}"))
                    save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results, metrics=cyclic_metrics)
                    logger.info(_orange(f"Derived and saved cyclic results (with metrics): {subject}"))

                    # Save base
                    base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                    if getattr(args, 'option_id_set', None):
                        base_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(base_save_path, exist_ok=True)
                    base_metrics = None
                    if len(base_results) > 0:
                        base_metrics = {'type': 'metric', 'data': {}}
                        base_metrics['data']['accuracy'] = get_accuracy(base_results)
                        base_metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(base_results)
                        logger.info("Derived base report:")
                        for key, value in base_metrics['data'].items():
                            logger.info(f"{key}: {value}")
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results, base_metrics)
                    logger.info(_orange(f"Derived and saved base results (with metrics): {subject}"))

                    if full_total > 0:
                        full_acc = full_corrects / full_total
                        logger.info(_purple(f"[{subject}] Full permutation ensemble accuracy: {full_acc:.4f}"))
                    else:
                        full_acc = float('nan')

                    summary_full = full_acc
                    summary_cyc = cyclic_metrics['data']['accuracy'] if cyclic_metrics is not None else float('nan')
                    summary_base = base_metrics['data']['accuracy'] if (base_metrics is not None and 'accuracy' in base_metrics['data']) else float('nan')
                    logger.info(_purple(f"[{subject}] Accuracies — Full: {summary_full:.4f}, Cyclic: {summary_cyc:.4f}, Default: {summary_base:.4f}"))

                    # Beta curves
                    if len(base_correct_list) == len(cyclic_correct_list) == len(full_correct_list) and len(base_correct_list) > 0:
                        N = len(base_correct_list)
                        betas = [i / 10.0 for i in range(11)]
                        C_cyc = float(k)
                        C_full = float(math.factorial(k))
                        curve_cyc = []
                        curve_full = []

                        for beta in betas:
                            n = int(N * beta + 1e-9)

                            # cyclic mix
                            if n > 0:
                                acc_cyc = (sum(cyclic_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                            else:
                                acc_cyc = sum(base_correct_list) / float(N)
                            cost_cyc = (beta * C_cyc) + ((1.0 - beta) * 1.0)
                            curve_cyc.append((cost_cyc, acc_cyc))

                            # full mix
                            if n > 0:
                                acc_full_mix = (sum(full_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                            else:
                                acc_full_mix = sum(base_correct_list) / float(N)
                            cost_full = (beta * C_full) + ((1.0 - beta) * 1.0)
                            curve_full.append((cost_full, acc_full_mix))

                        logger.info(_purple("[{}] Beta curve (Cyclic): ".format(subject) +
                                            ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_cyc])))
                        logger.info(_purple("[{}] Beta curve (Full): ".format(subject) +
                                            ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_full])))

                        # Ours (dynamic cascading)
                        curve_ours = []
                        ours_cascade_counts_list = []
                        try:
                            order_indices = list(range(len(perm_list)))
                            if identity_idx != 0:
                                order_indices = [identity_idx] + [i for i in order_indices if i != identity_idx]

                            per_sample_probs = []
                            base_probs_list = []
                            ideals = []
                            for r in results:
                                if r.get('type') != 'result':
                                    continue
                                data = r['data']
                                probs_seq = np.asarray(data['probs'], dtype=np.float64)
                                per_sample_probs.append(probs_seq)
                                base_probs_list.append(probs_seq[identity_idx])
                                ideals.append(data['ideal'])

                            def _conf_gap(pvec: np.ndarray) -> float:
                                vals = np.sort(pvec)[::-1]
                                if vals.shape[0] < 2:
                                    return 0.0
                                return float(vals[0] - vals[1])

                            default_conf = np.array([_conf_gap(bp) for bp in base_probs_list], dtype=np.float64)
                            perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0

                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                total_cost = 0.0
                                corrects = 0
                                cascade_counts = []

                                # beta subset: default only
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects += 1
                                    total_cost += 1.0
                                    cascade_counts.append(1)

                                # (1-beta) subset: cascading
                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    selected = [order_indices[0]]
                                    agg = _aggregate_probs_over_permutations(
                                        [probs_seq[j].tolist() for j in selected],
                                        [perm_list[j] for j in selected],
                                        k,
                                    )
                                    current_conf = _conf_gap(agg)
                                    t = 1
                                    while (current_conf < thresh) and (t < len(order_indices)):
                                        selected.append(order_indices[t])
                                        agg = _aggregate_probs_over_permutations(
                                            [probs_seq[j].tolist() for j in selected],
                                            [perm_list[j] for j in selected],
                                            k,
                                        )
                                        current_conf = _conf_gap(agg)
                                        t += 1
                                    pred_letter = option_ids[int(np.argmax(agg))]
                                    if pred_letter == ideals[i]:
                                        corrects += 1
                                    total_cost += float(len(selected))
                                    cascade_counts.append(int(len(selected)))

                                acc_ours = (corrects / float(N)) if N > 0 else float('nan')
                                cost_ours = (total_cost / float(N)) if N > 0 else float('nan')
                                curve_ours.append((cost_ours, acc_ours))
                                ours_cascade_counts_list.append(cascade_counts)

                            logger.info(_purple(
                                "[{}] Beta curve (Ours): ".format(subject) +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours])
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours curve: {e}")
                            curve_ours = []
                            ours_cascade_counts_list = []

                        # Switch-full & switch-cyclic
                        curve_ours_switch_full = []
                        curve_ours_switch_cyc = []
                        try:
                            if 'per_sample_probs' not in locals():
                                per_sample_probs = []
                                base_probs_list = []
                                ideals = []
                                for r in results:
                                    if r.get('type') != 'result':
                                        continue
                                    data = r['data']
                                    probs_seq = np.asarray(data['probs'], dtype=np.float64)
                                    per_sample_probs.append(probs_seq)
                                    base_probs_list.append(probs_seq[identity_idx])
                                    ideals.append(data['ideal'])

                            if 'default_conf' not in locals():
                                def _conf_gap2(pvec: np.ndarray) -> float:
                                    vals = np.sort(pvec)[::-1]
                                    if vals.shape[0] < 2:
                                        return 0.0
                                    return float(vals[0] - vals[1])

                                default_conf = np.array([_conf_gap2(bp) for bp in base_probs_list], dtype=np.float64)

                            perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0

                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                # switch-full
                                total_cost_sf = 0.0
                                corrects_sf = 0
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects_sf += 1
                                    total_cost_sf += 1.0

                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    if default_conf[i] < thresh:
                                        agg = _aggregate_probs_over_permutations(
                                            [probs_seq[j].tolist() for j in range(len(perm_list))],
                                            [perm_list[j] for j in range(len(perm_list))],
                                            k,
                                        )
                                        total_cost_sf += float(len(perm_list))
                                    else:
                                        agg = probs_seq[identity_idx]
                                        total_cost_sf += 1.0
                                    pred_letter = option_ids[int(np.argmax(agg))]
                                    if pred_letter == ideals[i]:
                                        corrects_sf += 1

                                acc_sf = (corrects_sf / float(N)) if N > 0 else float('nan')
                                cost_sf = (total_cost_sf / float(N)) if N > 0 else float('nan')
                                curve_ours_switch_full.append((cost_sf, acc_sf))

                                # switch-cyclic
                                total_cost_sc = 0.0
                                corrects_sc = 0
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects_sc += 1
                                    total_cost_sc += 1.0

                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    if default_conf[i] < thresh:
                                        agg = _aggregate_probs_over_permutations(
                                            [probs_seq[j].tolist() for j in cyclic_indices],
                                            [perm_list[j] for j in cyclic_indices],
                                            k,
                                        )
                                        total_cost_sc += float(k)
                                    else:
                                        agg = probs_seq[identity_idx]
                                        total_cost_sc += 1.0
                                    pred_letter = option_ids[int(np.argmax(agg))]
                                    if pred_letter == ideals[i]:
                                        corrects_sc += 1

                                acc_sc = (corrects_sc / float(N)) if N > 0 else float('nan')
                                cost_sc = (total_cost_sc / float(N)) if N > 0 else float('nan')
                                curve_ours_switch_cyc.append((cost_sc, acc_sc))

                            logger.info(_purple(
                                "[{}] Beta curve (Ours switch-full): ".format(subject) +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_switch_full])
                            ))
                            logger.info(_purple(
                                "[{}] Beta curve (Ours switch-cyclic): ".format(subject) +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_switch_cyc])
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours ablation curves: {e}")
                            curve_ours_switch_full = []
                            curve_ours_switch_cyc = []

                        # --------- NEW: Ours top2 -> cyclic (with conf threshold + label-flip) ----------
                        curve_ours_top2cyc = []
                        try:
                            if 'per_sample_probs' not in locals():
                                per_sample_probs = []
                                base_probs_list = []
                                ideals = []
                                for r in results:
                                    if r.get('type') != 'result':
                                        continue
                                    data = r['data']
                                    probs_seq = np.asarray(data['probs'], dtype=np.float64)
                                    per_sample_probs.append(probs_seq)
                                    base_probs_list.append(probs_seq[identity_idx])
                                    ideals.append(data['ideal'])

                            if 'default_conf' not in locals():
                                def _conf_gap3(pvec: np.ndarray) -> float:
                                    vals = np.sort(pvec)[::-1]
                                    if vals.shape[0] < 2:
                                        return 0.0
                                    return float(vals[0] - vals[1])
                                default_conf = np.array([_conf_gap3(bp) for bp in base_probs_list], dtype=np.float64)

                            perm_to_idx = {perm_list[i]: i for i in range(len(perm_list))}
                            id_perm = tuple(range(k))

                            # 1) 모든 샘플에 대해 "identity + top2 ensemble" conf 미리 계산
                            top2_conf_all = []
                            top2_pred_all = []
                            base_pred_all = []
                            for i in range(N):
                                bp = np.asarray(base_probs_list[i], dtype=np.float64)
                                probs_seq = np.asarray(per_sample_probs[i], dtype=np.float64)

                                order_desc = np.argsort(bp)[::-1]
                                idx1, idx2 = int(order_desc[0]), int(order_desc[1])

                                perm_top2 = list(range(k))
                                perm_top2[idx1], perm_top2[idx2] = perm_top2[idx2], perm_top2[idx1]
                                perm_top2 = tuple(perm_top2)
                                idx_top2 = perm_to_idx[perm_top2]

                                agg_t2 = _aggregate_probs_over_permutations(
                                    [bp.tolist(), probs_seq[idx_top2].tolist()],
                                    [id_perm, perm_top2],
                                    k,
                                )
                                vals = np.sort(agg_t2)[::-1]
                                conf_t2 = float(vals[0] - vals[1]) if vals.shape[0] >= 2 else 0.0
                                top2_conf_all.append(conf_t2)
                                top2_pred_all.append(int(np.argmax(agg_t2)))
                                base_pred_all.append(int(np.argmax(bp)))

                            top2_conf_all = np.asarray(top2_conf_all, dtype=np.float64)
                            top2_pred_all = np.asarray(top2_pred_all, dtype=np.int64)
                            base_pred_all = np.asarray(base_pred_all, dtype=np.int64)

                            perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0
                            if perc > 0.0:
                                top2_conf_thresh = float(np.quantile(top2_conf_all, perc))
                            else:
                                top2_conf_thresh = -1.0

                            # 2) beta별로 base / top2 / cyclic 조합
                            for beta in betas:
                                n = int(N * beta + 1e-9)

                                total_cost_t2c = 0.0
                                corrects_t2c = 0

                                # (1) beta 구간: 기본 default만
                                for i in range(0, n):
                                    bp = np.asarray(base_probs_list[i], dtype=np.float64)
                                    pred_idx = int(np.argmax(bp))
                                    if option_ids[pred_idx] == ideals[i]:
                                        corrects_t2c += 1
                                    total_cost_t2c += 1.0

                                # (2) 나머지 구간: base + top2는 항상 쓰고, 필요시 cyclic 승급
                                for i in range(n, N):
                                    probs_seq = np.asarray(per_sample_probs[i], dtype=np.float64)
                                    bp = np.asarray(base_probs_list[i], dtype=np.float64)

                                    order_desc = np.argsort(bp)[::-1]
                                    idx1, idx2 = int(order_desc[0]), int(order_desc[1])

                                    perm_top2 = list(range(k))
                                    perm_top2[idx1], perm_top2[idx2] = perm_top2[idx2], perm_top2[idx1]
                                    perm_top2 = tuple(perm_top2)
                                    idx_top2 = perm_to_idx[perm_top2]

                                    agg_t2 = _aggregate_probs_over_permutations(
                                        [bp.tolist(), probs_seq[idx_top2].tolist()],
                                        [id_perm, perm_top2],
                                        k,
                                    )
                                    vals = np.sort(agg_t2)[::-1]
                                    conf_t2 = float(vals[0] - vals[1]) if vals.shape[0] >= 2 else 0.0
                                    pred_base = int(np.argmax(bp))
                                    pred_t2 = int(np.argmax(agg_t2))

                                    need_cyclic = False
                                    # (a) base vs top2 label flip → ID bias 강하게 의심
                                    if pred_t2 != pred_base:
                                        need_cyclic = True
                                    # (b) top2조차 플랫(bottom perc%)이면 cyclic
                                    if conf_t2 <= top2_conf_thresh:
                                        need_cyclic = True

                                    if not need_cyclic:
                                        pred_idx = pred_t2
                                        total_cost_t2c += 2.0  # base + top2
                                    else:
                                        cyc_probs_seq = [probs_seq[j].tolist() for j in cyclic_indices]
                                        cyc_perm = [perm_list[j] for j in cyclic_indices]
                                        agg_cyc = _aggregate_probs_over_permutations(cyc_probs_seq, cyc_perm, k)
                                        pred_idx = int(np.argmax(agg_cyc))
                                        total_cost_t2c += float(k)  # cyclic: k번 호출

                                    if option_ids[pred_idx] == ideals[i]:
                                        corrects_t2c += 1

                                acc_t2c = (corrects_t2c / float(N)) if N > 0 else float('nan')
                                cost_t2c = (total_cost_t2c / float(N)) if N > 0 else float('nan')
                                curve_ours_top2cyc.append((cost_t2c, acc_t2c))

                            logger.info(_purple(
                                "[{}] Beta curve (Ours top2->cyclic): ".format(subject) +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_top2cyc])
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours top2->cyclic curve: {e}")
                            curve_ours_top2cyc = []

                        # Save curve data
                        curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                        if getattr(args, 'option_id_set', None):
                            curve_save_path += f'_id-{args.option_id_set}'
                        os.makedirs(curve_save_path, exist_ok=True)
                        curve_obj = {
                            'subject': subject,
                            'k': k,
                            'betas': betas,
                            'default_accuracy': summary_base,
                            'cyclic': {
                                'costs': [c for c, _ in curve_cyc],
                                'accuracies': [a for _, a in curve_cyc],
                            },
                            'full': {
                                'costs': [c for c, _ in curve_full],
                                'accuracies': [a for _, a in curve_full],
                            },
                        }
                        if len(curve_ours) == len(betas):
                            curve_obj['ours'] = {
                                'costs': [c for c, _ in curve_ours],
                                'accuracies': [a for _, a in curve_ours],
                            }
                        if len(curve_ours_switch_full) == len(betas):
                            curve_obj['ours_switch_full'] = {
                                'costs': [c for c, _ in curve_ours_switch_full],
                                'accuracies': [a for _, a in curve_ours_switch_full],
                            }
                        if len(curve_ours_switch_cyc) == len(betas):
                            curve_obj['ours_switch_cyclic'] = {
                                'costs': [c for c, _ in curve_ours_switch_cyc],
                                'accuracies': [a for _, a in curve_ours_switch_cyc],
                            }
                        if len(curve_ours_top2cyc) == len(betas):
                            curve_obj['ours_top2_cyclic'] = {
                                'costs': [c for c, _ in curve_ours_top2cyc],
                                'accuracies': [a for _, a in curve_ours_top2cyc],
                            }

                        # Oracle low-confidence analysis
                        try:
                            def _conf_gap_oracle(pvec: np.ndarray) -> float:
                                vals = np.sort(pvec)[::-1]
                                if vals.shape[0] < 2:
                                    return 0.0
                                return float(vals[0] - vals[1])

                            default_confs = []
                            default_corrects = []
                            for r in results:
                                if r.get('type') != 'result':
                                    continue
                                data = r['data']
                                probs_seq = np.asarray(data['probs'], dtype=np.float64)
                                base_probs = probs_seq[identity_idx]
                                default_confs.append(_conf_gap_oracle(base_probs))
                                pred_letter = option_ids[int(np.argmax(base_probs))]
                                default_corrects.append(int(pred_letter == data['ideal']))

                            default_confs = np.asarray(default_confs, dtype=np.float64)
                            default_corrects = np.asarray(default_corrects, dtype=np.int32)
                            order = np.argsort(default_confs)  # ascending (low conf first)
                            oracle_percentiles = list(range(1, 101))
                            oracle_bottom_accs = []
                            for p in oracle_percentiles:
                                n = max(1, int(N * (p / 100.0) + 1e-9))
                                sel = order[:n]
                                acc_bottom = float(default_corrects[sel].mean())
                                oracle_bottom_accs.append(acc_bottom)
                            bottom10 = oracle_bottom_accs[9] if len(oracle_bottom_accs) >= 10 else float('nan')
                            logger.info(_purple(f"[{subject}] Oracle bottom-10% accuracy (default): {bottom10:.4f}"))
                            curve_obj['oracle_low_conf'] = {
                                'percentiles': oracle_percentiles,
                                'accuracies': oracle_bottom_accs,
                                'bottom10_acc': bottom10,
                            }
                        except Exception as e:
                            logger.warning(f"Failed to compute oracle low-confidence accuracy curve: {e}")

                        save_results(f'{curve_save_path}/{subject}_beta_curve.jsonl', [curve_obj], metrics=None)

                        # W&B logging
                        if wandb_run is not None:
                            try:
                                import wandb

                                target_idx = args.wandb_sample_idx
                                chosen = None
                                for r in results:
                                    if r.get('type') != 'result':
                                        continue
                                    if target_idx is None or int(r['data']['idx']) == int(target_idx):
                                        chosen = r['data']
                                        break
                                if chosen is None and len(results) > 0 and results[0].get('type') == 'result':
                                    chosen = results[0]['data']

                                if chosen is not None and 'prompts' in chosen and 'probs' in chosen:
                                    prompts_list = chosen['prompts']
                                    probs_seq = chosen['probs']
                                    cols = ['perm_idx', 'ideal', 'prompt'] + [f'prob_{oid}' for oid in option_ids]
                                    rows = []
                                    for pi, (ptext, pvec) in enumerate(zip(prompts_list, probs_seq)):
                                        rows.append([pi, chosen['ideal'], ptext] + [float(x) for x in pvec])
                                    table = wandb.Table(columns=cols, data=rows)
                                    wandb.log({f"{subject}/sample_prompts": table})

                                fig = plt.figure(figsize=(7.5, 5.0), dpi=160)
                                cyc_costs = [c for c, _ in curve_cyc]
                                cyc_accs = [a for _, a in curve_cyc]
                                full_costs = [c for c, _ in curve_full]
                                full_accs = [a for _, a in curve_full]
                                plt.plot(cyc_costs, cyc_accs, marker='o', label='Cyclic (k rotations)')
                                plt.plot(full_costs, full_accs, marker='o', label='Full (k! permutations)')
                                if len(curve_ours) == len(betas):
                                    ours_costs = [c for c, _ in curve_ours]
                                    ours_accs = [a for _, a in curve_ours]
                                    plt.plot(ours_costs, ours_accs, marker='o', label='Ours (cascading)')
                                if len(curve_ours_switch_full) == len(betas):
                                    sf_costs = [c for c, _ in curve_ours_switch_full]
                                    sf_accs = [a for _, a in curve_ours_switch_full]
                                    plt.plot(sf_costs, sf_accs, marker='o', label='Ours (switch-full)')
                                if len(curve_ours_switch_cyc) == len(betas):
                                    sc_costs = [c for c, _ in curve_ours_switch_cyc]
                                    sc_accs = [a for _, a in curve_ours_switch_cyc]
                                    plt.plot(sc_costs, sc_accs, marker='o', label='Ours (switch-cyclic)')
                                if len(curve_ours_top2cyc) == len(betas):
                                    t2_costs = [c for c, _ in curve_ours_top2cyc]
                                    t2_accs = [a for _, a in curve_ours_top2cyc]
                                    plt.plot(t2_costs, t2_accs, marker='o', label='Ours (top2->cyclic)')
                                plt.scatter([1.0], [summary_base], marker='*', s=180, c='black', label='Default')
                                plt.xlabel("Computational Cost (× of default)")
                                plt.ylabel("Accuracy")
                                plt.title(f"Accuracy vs. Cost — {subject}")
                                plt.grid(True, linestyle='--', alpha=0.4)
                                plt.legend()
                                plt.tight_layout()
                                out_png = f"{curve_save_path}/{subject}_beta_curve.png"
                                fig.savefig(out_png, dpi=160, bbox_inches='tight')
                                wandb.log({f"{subject}/beta_curve": wandb.Image(out_png)})
                                plt.close(fig)

                                if 'oracle_low_conf' in curve_obj:
                                    fig2 = plt.figure(figsize=(7.5, 5.0), dpi=160)
                                    xs = curve_obj['oracle_low_conf']['percentiles']
                                    ys = curve_obj['oracle_low_conf']['accuracies']
                                    plt.plot(xs, ys, marker='o', label='Bottom-p% (default acc)')
                                    plt.xlabel("p (Bottom p% by default confidence)")
                                    plt.ylabel("Accuracy on bottom-p% subset")
                                    plt.title(f"Oracle: Low-confidence subset accuracy — {subject}")
                                    plt.grid(True, linestyle='--', alpha=0.4)
                                    plt.legend()
                                    plt.tight_layout()
                                    out_png2 = f"{curve_save_path}/{subject}_oracle_low_conf_acc.png"
                                    fig2.savefig(out_png2, dpi=160, bbox_inches='tight')
                                    wandb.log({f"{subject}/oracle_low_conf_curve": wandb.Image(out_png2)})
                                    plt.close(fig2)

                                if len(curve_ours) == len(betas) and isinstance(ours_cascade_counts_list, list) and len(ours_cascade_counts_list) == len(betas):
                                    try:
                                        for bi, beta in enumerate(betas):
                                            counts = ours_cascade_counts_list[bi]
                                            if isinstance(counts, list) and len(counts) > 0:
                                                wandb.log({f"{subject}/ours_cascade_hist_beta_{beta:.1f}": wandb.Histogram(counts)})
                                    except Exception as e:
                                        logger.warning(f"W&B cascade histogram logging failed: {e}")

                                payload = {
                                    f"{subject}/acc_full": summary_full,
                                    f"{subject}/acc_cyclic": summary_cyc,
                                    f"{subject}/acc_default": summary_base,
                                }
                                if len(curve_ours_top2cyc) == len(betas):
                                    payload[f"{subject}/acc_top2_cyclic_best"] = max(a for _, a in curve_ours_top2cyc)
                                wandb.log(payload)
                            except Exception as e:
                                logger.warning(f"W&B logging failed: {e}")
                except Exception as e:
                    logger.warning(f"Failed to derive cyclic/base from full for subject '{subject}': {e}")

            logging_cuda_memory_usage()

    try:
        if wandb_run is not None:
            import wandb
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
