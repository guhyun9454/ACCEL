# eval_clm.py
# -*- coding: utf-8 -*-

import os
import sys
import gc
import json
import copy
import logging
import random
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

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

import pynvml
pynvml.nvmlInit()

logger = logging.getLogger(__name__)


def logging_cuda_memory_usage():
    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(
            "GPU {}: {:.2f} GB / {:.2f} GB".format(
                i,
                meminfo.used / 1024 ** 3,
                meminfo.total / 1024 ** 3
            )
        )


def _rotations(k: int):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(probs_seq, permuted_indices, k: int):
    """
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
    hf_logging.set_verbosity_error()

    args = parse_arguments()
    if len(args.eval_names) == 0:
        return

    # Optional: W&B init
    wandb_run = None
    if getattr(args, 'wandb', False):
        try:
            import wandb
            project = args.wandb_project
            run_name = args.wandb_run_name or f"{args.model_name}-{args.eval_names[0]}"
            wandb_run = wandb.init(
                entity="capde",
                project=project,
                name=run_name,
                config={
                    "pretrained_model_path": args.pretrained_model_path,
                    "model_name": args.model_name,
                    "eval_names": args.eval_names,
                    "option_id_set": args.option_id_set,
                },
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            wandb_run = None

    # Tokenizer / Model
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

    # ---------------------------------------------------------
    # 1) Comparative mode: option_id_sets 두 개 비교
    # ---------------------------------------------------------
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
            if setting in ['perm', 'cyclic', 'full']:
                logger.info(_orange(f"Skipping compare for setting '{setting}' (not supported)."))
                continue

            args_a = SimpleNamespace(
                pretrained_model_path=args.pretrained_model_path,
                model_name=args.model_name,
                option_id_set=id_set_a,
            )
            args_b = SimpleNamespace(
                pretrained_model_path=args.pretrained_model_path,
                model_name=args.model_name,
                option_id_set=id_set_b,
            )

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
            logger.info(
                _purple(
                    "==== Overall compare summary ====\n"
                    f"matching_ratio={overall_mr:.4f} (matches/total={overall_matches}/{overall_total})\n"
                    f"correct->incorrect={overall_c2i}, incorrect->correct={overall_i2c}, "
                    f"both_correct={overall_both_correct}, both_incorrect={overall_both_incorrect}"
                )
            )
        return

    # ---------------------------------------------------------
    # 2) Single-run mode
    # ---------------------------------------------------------
    for eval_name in args.eval_names[::1]:
        (subjects, prepare_few_shot_samples,
         prepare_eval_samples, prepare_eval_fn) = prepare_eval(args, eval_name)

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
                        from itertools import permutations
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
                save_results(cached_path, results, metrics)
                logger.info(f"Results saved: {subject}")

            # -------------------------------------------------
            # FULL permutation에서 cyclic / base / beta curve 등 파생 결과 계산
            # -------------------------------------------------
            if args.setting == 'full' and len(results) > 0:
                try:
                    # Determine option ids
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        k_guess = len(results[0]['data']['options'])
                        option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                    k = len(option_ids)

                    from itertools import permutations
                    perm_list = list(sorted(permutations(range(k))))
                    perm_index_map = {p: idx for idx, p in enumerate(perm_list)}
                    identity_idx = perm_list.index(tuple(range(k)))
                    identity_perm = perm_list[identity_idx]
                    cyclic_indices = [
                        perm_list.index(tuple((i + s) % k for i in range(k)))
                        for s in range(k)
                    ]

                    # Derived results
                    cyclic_results = []
                    base_results = []
                    base_correct_list = []
                    cyclic_correct_list = []
                    full_correct_list = []

                    full_total = 0
                    full_corrects = 0
                    cyclic_total = 0
                    cyclic_corrects = 0

                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data['probs']
                        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                            continue

                        # Cyclic probs subset
                        cyc_probs = [probs_seq[idx] for idx in cyclic_indices]
                        cyclic_results.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': cyc_probs,
                                'ideal': data['ideal'],
                            },
                        })
                        cyc_perms = [tuple((i + s) % k for i in range(k)) for s in range(k)]
                        agg_cyc = _aggregate_probs_over_permutations(
                            cyc_probs, cyc_perms, k
                        )
                        pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                        corr_cyc = (pred_cyc == data['ideal'])
                        cyclic_correct_list.append(corr_cyc)
                        if corr_cyc:
                            cyclic_corrects += 1
                        cyclic_total += 1

                        # Base (identity)
                        base_probs = np.asarray(probs_seq[identity_idx], dtype=np.float64)
                        pred_base = option_ids[int(np.argmax(base_probs))]
                        corr_base = (pred_base == data['ideal'])
                        base_correct_list.append(corr_base)
                        base_results.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': base_probs.tolist(),
                                'sampled': pred_base,
                                'ideal': data['ideal'],
                                'correct': corr_base,
                            },
                        })

                        # Full ensemble
                        agg_full = _aggregate_probs_over_permutations(
                            probs_seq, perm_list, k
                        )
                        pred_full = option_ids[int(np.argmax(agg_full))]
                        corr_full = (pred_full == data['ideal'])
                        full_correct_list.append(corr_full)
                        if corr_full:
                            full_corrects += 1
                        full_total += 1

                    # Save cyclic-derived results
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

                    # Save base-derived results
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

                    # Full ensemble accuracy
                    if full_total > 0:
                        full_acc = full_corrects / full_total
                        logger.info(_purple(f"[{subject}] Full permutation ensemble accuracy: {full_acc:.4f}"))

                    summary_full = full_acc if full_total > 0 else float('nan')
                    summary_cyc = cyclic_acc if cyclic_total > 0 else float('nan')
                    summary_base = base_metrics['data']['accuracy'] if (base_metrics is not None and 'accuracy' in base_metrics['data']) else float('nan')
                    logger.info(_purple(f"[{subject}] Accuracies — Full: {summary_full:.4f}, Cyclic: {summary_cyc:.4f}, Default: {summary_base:.4f}"))

                    # ---------------- Beta curves (Cyclic / Full / Ours / switch / top2 variants) ------------
                    if len(base_correct_list) == len(cyclic_correct_list) == len(full_correct_list) and len(base_correct_list) > 0:
                        N = len(base_correct_list)
                        betas = [i / 10.0 for i in range(11)]
                        C_cyc = float(k)
                        C_full = float(math.factorial(k))

                        # Cyclic / Full beta curves (deterministic subset)
                        curve_cyc = []
                        curve_full = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)

                            # cyclic mix
                            if n > 0:
                                acc_cyc_mix = (sum(cyclic_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                            else:
                                acc_cyc_mix = sum(base_correct_list) / float(N)
                            cost_cyc = beta * C_cyc + (1.0 - beta) * 1.0
                            curve_cyc.append((cost_cyc, acc_cyc_mix))

                            # full mix
                            if n > 0:
                                acc_full_mix = (sum(full_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                            else:
                                acc_full_mix = sum(base_correct_list) / float(N)
                            cost_full_mix = beta * C_full + (1.0 - beta) * 1.0
                            curve_full.append((cost_full_mix, acc_full_mix))

                        logger.info(_purple(f"[{subject}] Beta curve (Cyclic): " +
                                            ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_cyc])))
                        logger.info(_purple(f"[{subject}] Beta curve (Full): " +
                                            ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_full])))

                        # Helper for confidence gap
                        def _conf_gap(pvec: np.ndarray) -> float:
                            vals = np.sort(pvec)[::-1]
                            if vals.shape[0] < 2:
                                return 0.0
                            return float(vals[0] - vals[1])

                        # per-sample structures
                        per_sample_probs = []
                        base_probs_list = []
                        ideals = []
                        default_conf = []
                        for r in results:
                            if r.get('type') != 'result':
                                continue
                            data_r = r['data']
                            probs_seq_r = np.asarray(data_r['probs'], dtype=np.float64)
                            per_sample_probs.append(probs_seq_r)
                            base_probs = probs_seq_r[identity_idx]
                            base_probs_list.append(base_probs)
                            ideals.append(data_r['ideal'])
                            default_conf.append(_conf_gap(base_probs))
                        default_conf = np.asarray(default_conf, dtype=np.float64)

                        perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0

                        # ---------------- Ours (cascading ensemble) ----------------
                        curve_ours = []
                        ours_cascade_counts_list = []
                        try:
                            order_indices = list(range(len(perm_list)))
                            if identity_idx != 0:
                                order_indices = [identity_idx] + [i for i in order_indices if i != identity_idx]

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

                                # (1-beta) subset: dynamic cascade
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
                                    cascade_counts.append(len(selected))

                                acc_ours = (corrects / float(N)) if N > 0 else float('nan')
                                cost_ours = (total_cost / float(N)) if N > 0 else float('nan')
                                curve_ours.append((cost_ours, acc_ours))
                                ours_cascade_counts_list.append(cascade_counts)

                            logger.info(_purple(f"[{subject}] Beta curve (Ours): " +
                                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours])))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours curve: {e}")
                            curve_ours = []
                            ours_cascade_counts_list = []

                        # ---------------- switch-full / switch-cyclic ----------------
                        curve_ours_switch_full = []
                        curve_ours_switch_cyc = []
                        try:
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

                            logger.info(_purple(f"[{subject}] Beta curve (Ours switch-full): " +
                                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_switch_full])))
                            logger.info(_purple(f"[{subject}] Beta curve (Ours switch-cyclic): " +
                                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_switch_cyc])))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours ablation curves: {e}")
                            curve_ours_switch_full = []
                            curve_ours_switch_cyc = []

                        # ---------------- Ours top2 -> cyclic (low-conf only) ----------------
                        curve_ours_top2_cyc = []
                        try:
                            top2_gap_frac = 0.05

                            for beta in betas:
                                n = int(N * beta + 1e-9)

                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                total_cost_t2 = 0.0
                                corrects_t2 = 0

                                # beta subset: default만
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects_t2 += 1
                                    total_cost_t2 += 1.0

                                # (1-beta) subset
                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    base_probs = base_probs_list[i]
                                    pred_base_letter = option_ids[int(np.argmax(base_probs))]

                                    # high-conf → default
                                    if default_conf[i] >= thresh:
                                        total_cost_t2 += 1.0
                                        if pred_base_letter == ideals[i]:
                                            corrects_t2 += 1
                                        continue

                                    # base에서 top2 swap
                                    sorted_idx = np.argsort(base_probs)[::-1]
                                    top1_idx = int(sorted_idx[0])
                                    top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx
                                    top1_val = float(base_probs[top1_idx])
                                    top2_val = float(base_probs[top2_idx])

                                    if top1_val - top2_val >= top2_gap_frac * max(top1_val, 1e-8):
                                        perm_swap = list(identity_perm)
                                        perm_swap[top1_idx], perm_swap[top2_idx] = perm_swap[top2_idx], perm_swap[top1_idx]
                                        perm_swap_t = tuple(perm_swap)
                                        if perm_swap_t in perm_list:
                                            swap_idx = perm_list.index(perm_swap_t)
                                            probs_base = probs_seq[identity_idx]
                                            probs_swap = probs_seq[swap_idx]
                                            agg_top2 = _aggregate_probs_over_permutations(
                                                [probs_base.tolist(), probs_swap.tolist()],
                                                [perm_list[identity_idx], perm_list[swap_idx]],
                                                k,
                                            )
                                        else:
                                            agg_top2 = base_probs.copy()
                                    else:
                                        agg_top2 = base_probs.copy()

                                    pred_top2_letter = option_ids[int(np.argmax(agg_top2))]

                                    # 답 바뀌었으면 cyclic, 아니면 top2까지만
                                    if pred_top2_letter == pred_base_letter:
                                        # top2까지 (혹은 base와 동일)
                                        if np.allclose(agg_top2, base_probs):
                                            total_cost_t2 += 1.0
                                            final_letter = pred_base_letter
                                        else:
                                            total_cost_t2 += 2.0
                                            final_letter = pred_top2_letter
                                    else:
                                        # 여기서는 cyclic으로 승급
                                        cyc_probs = [probs_seq[j].tolist() for j in cyclic_indices]
                                        cyc_perms = [perm_list[j] for j in cyclic_indices]
                                        agg_cyc = _aggregate_probs_over_permutations(
                                            cyc_probs, cyc_perms, k
                                        )
                                        pred_cyc_letter = option_ids[int(np.argmax(agg_cyc))]
                                        final_letter = pred_cyc_letter
                                        total_cost_t2 += float(k)

                                    if final_letter == ideals[i]:
                                        corrects_t2 += 1

                                acc_t2 = (corrects_t2 / float(N)) if N > 0 else float('nan')
                                cost_t2 = (total_cost_t2 / float(N)) if N > 0 else float('nan')
                                curve_ours_top2_cyc.append((cost_t2, acc_t2))

                            logger.info(_purple(f"[{subject}] Beta curve (Ours top2->cyclic): " +
                                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_top2_cyc])))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours top2->cyclic curve: {e}")
                            curve_ours_top2_cyc = []

                        # ---------------- Ours top2flip -> cyclic (low-conf + flip only) ----------------
                        curve_ours_top2flip_cyc = []
                        try:
                            top2_gap_frac2 = 0.1

                            for beta in betas:
                                n = int(N * beta + 1e-9)

                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                total_cost_t2f = 0.0
                                corrects_t2f = 0

                                # beta subset: default
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects_t2f += 1
                                    total_cost_t2f += 1.0

                                # (1-beta) subset
                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    base_probs = base_probs_list[i]
                                    pred_base_letter = option_ids[int(np.argmax(base_probs))]

                                    # high-conf → default
                                    if default_conf[i] >= thresh:
                                        total_cost_t2f += 1.0
                                        if pred_base_letter == ideals[i]:
                                            corrects_t2f += 1
                                        continue

                                    # base에서 top2 swap
                                    sorted_idx = np.argsort(base_probs)[::-1]
                                    top1_idx = int(sorted_idx[0])
                                    top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx
                                    top1_val = float(base_probs[top1_idx])
                                    top2_val = float(base_probs[top2_idx])

                                    if top1_val - top2_val >= top2_gap_frac2 * max(top1_val, 1e-8):
                                        perm_swap = list(identity_perm)
                                        perm_swap[top1_idx], perm_swap[top2_idx] = perm_swap[top2_idx], perm_swap[top1_idx]
                                        perm_swap_t = tuple(perm_swap)
                                        if perm_swap_t in perm_list:
                                            swap_idx = perm_list.index(perm_swap_t)
                                            probs_base = probs_seq[identity_idx]
                                            probs_swap = probs_seq[swap_idx]
                                            agg_top2 = _aggregate_probs_over_permutations(
                                                [probs_base.tolist(), probs_swap.tolist()],
                                                [perm_list[identity_idx], perm_list[swap_idx]],
                                                k,
                                            )
                                        else:
                                            agg_top2 = base_probs.copy()
                                    else:
                                        agg_top2 = base_probs.copy()

                                    pred_top2_letter = option_ids[int(np.argmax(agg_top2))]

                                    # flip 없으면 default에 그대로
                                    if pred_top2_letter == pred_base_letter:
                                        total_cost_t2f += 1.0
                                        if pred_base_letter == ideals[i]:
                                            corrects_t2f += 1
                                        continue

                                    # flip + low-conf인 애들만 cyclic 승급
                                    cyc_probs = [probs_seq[j].tolist() for j in cyclic_indices]
                                    cyc_perms = [perm_list[j] for j in cyclic_indices]
                                    agg_cyc = _aggregate_probs_over_permutations(
                                        cyc_probs, cyc_perms, k
                                    )
                                    pred_cyc_letter = option_ids[int(np.argmax(agg_cyc))]
                                    total_cost_t2f += float(k)
                                    if pred_cyc_letter == ideals[i]:
                                        corrects_t2f += 1

                                acc_t2f = (corrects_t2f / float(N)) if N > 0 else float('nan')
                                cost_t2f = (total_cost_t2f / float(N)) if N > 0 else float('nan')
                                curve_ours_top2flip_cyc.append((cost_t2f, acc_t2f))

                            logger.info(_purple(f"[{subject}] Beta curve (Ours top2flip->cyclic): " +
                                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_top2flip_cyc])))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours top2flip->cyclic curve: {e}")
                            curve_ours_top2flip_cyc = []

                        # ---------------- Ours top2 -> top3 -> cyclic (low-conf + flip만 deeper) ----------------
                        curve_ours_top2_3_cyc = []
                        try:
                            top2_gap_frac_adapt = 0.1

                            for beta in betas:
                                n = int(N * beta + 1e-9)

                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                total_cost_t4 = 0.0
                                corrects_t4 = 0

                                # beta subset: 항상 default
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects_t4 += 1
                                    total_cost_t4 += 1.0

                                # (1-beta) subset
                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    base_probs = base_probs_list[i]
                                    pred_base_letter = option_ids[int(np.argmax(base_probs))]

                                    # high-conf → default
                                    if default_conf[i] >= thresh:
                                        total_cost_t4 += 1.0
                                        if pred_base_letter == ideals[i]:
                                            corrects_t4 += 1
                                        continue

                                    # top1/top2/top3 index
                                    sorted_idx = np.argsort(base_probs)[::-1]
                                    top1_idx = int(sorted_idx[0])
                                    top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx
                                    top3_idx = int(sorted_idx[2]) if len(sorted_idx) > 2 else top2_idx

                                    top1_val = float(base_probs[top1_idx])
                                    top2_val = float(base_probs[top2_idx])

                                    # 먼저 top2 swap
                                    if top1_val - top2_val >= top2_gap_frac_adapt * max(top1_val, 1e-8):
                                        perm_swap = list(identity_perm)
                                        perm_swap[top1_idx], perm_swap[top2_idx] = perm_swap[top2_idx], perm_swap[top1_idx]
                                        perm_swap_t = tuple(perm_swap)
                                        swap_idx = perm_index_map.get(perm_swap_t, identity_idx)
                                        idxs_top2 = [identity_idx, swap_idx]
                                        agg_top2 = _aggregate_probs_over_permutations(
                                            [probs_seq[j].tolist() for j in idxs_top2],
                                            [perm_list[j] for j in idxs_top2],
                                            k,
                                        )
                                    else:
                                        agg_top2 = base_probs.copy()
                                        idxs_top2 = [identity_idx]

                                    pred_top2_letter = option_ids[int(np.argmax(agg_top2))]

                                    # flip 없으면 ID 민감도 낮다고 보고 default로 고정
                                    if pred_top2_letter == pred_base_letter:
                                        total_cost_t4 += 1.0
                                        if pred_base_letter == ideals[i]:
                                            corrects_t4 += 1
                                        continue

                                    # flip 발생 → ID bias 민감한 샘플
                                    conf_top2 = _conf_gap(agg_top2)

                                    # top2에서 conf 회복되면 여기서 stop (cost≈2)
                                    if conf_top2 >= thresh:
                                        total_cost_t4 += float(len(idxs_top2))
                                        if pred_top2_letter == ideals[i]:
                                            corrects_t4 += 1
                                        continue

                                    # top3 partial cyclic
                                    S = [top1_idx, top2_idx, top3_idx]
                                    S = list(dict.fromkeys(S))
                                    if len(S) < 2:
                                        # 안전장치: 그냥 cyclic
                                        cyc_probs = [probs_seq[j].tolist() for j in cyclic_indices]
                                        cyc_perms = [perm_list[j] for j in cyclic_indices]
                                        agg_cyc = _aggregate_probs_over_permutations(
                                            cyc_probs, cyc_perms, k
                                        )
                                        pred_cyc_letter = option_ids[int(np.argmax(agg_cyc))]
                                        total_cost_t4 += float(k)
                                        if pred_cyc_letter == ideals[i]:
                                            corrects_t4 += 1
                                        continue

                                    # identity + S에 대한 두 번 rotation (3개 perm)
                                    p0 = identity_perm
                                    p1_list = list(identity_perm)
                                    p2_list = list(identity_perm)
                                    for idx_c in S:
                                        pos = S.index(idx_c)
                                        p1_list[idx_c] = S[(pos + 1) % len(S)]
                                        p2_list[idx_c] = S[(pos + 2) % len(S)]
                                    p1 = tuple(p1_list)
                                    p2 = tuple(p2_list)

                                    idxs_part = []
                                    for p in (p0, p1, p2):
                                        idx_p = perm_index_map.get(p, None)
                                        if idx_p is not None:
                                            idxs_part.append(idx_p)

                                    if len(idxs_part) < 2:
                                        # perm을 제대로 못 찾으면 cyclic으로 fallback
                                        cyc_probs = [probs_seq[j].tolist() for j in cyclic_indices]
                                        cyc_perms = [perm_list[j] for j in cyclic_indices]
                                        agg_cyc = _aggregate_probs_over_permutations(
                                            cyc_probs, cyc_perms, k
                                        )
                                        pred_cyc_letter = option_ids[int(np.argmax(agg_cyc))]
                                        total_cost_t4 += float(k)
                                        if pred_cyc_letter == ideals[i]:
                                            corrects_t4 += 1
                                        continue

                                    agg_top3 = _aggregate_probs_over_permutations(
                                        [probs_seq[j].tolist() for j in idxs_part],
                                        [perm_list[j] for j in idxs_part],
                                        k,
                                    )
                                    pred_top3_letter = option_ids[int(np.argmax(agg_top3))]
                                    conf_top3 = _conf_gap(agg_top3)

                                    # top3에서 conf 회복되면 stop (cost≈3)
                                    if conf_top3 >= thresh:
                                        total_cost_t4 += float(len(idxs_part))
                                        if pred_top3_letter == ideals[i]:
                                            corrects_t4 += 1
                                        continue

                                    # 마지막 단계: full cyclic
                                    cyc_probs = [probs_seq[j].tolist() for j in cyclic_indices]
                                    cyc_perms = [perm_list[j] for j in cyclic_indices]
                                    agg_cyc = _aggregate_probs_over_permutations(
                                        cyc_probs, cyc_perms, k
                                    )
                                    pred_cyc_letter = option_ids[int(np.argmax(agg_cyc))]
                                    total_cost_t4 += float(k)
                                    if pred_cyc_letter == ideals[i]:
                                        corrects_t4 += 1

                                acc_t4 = (corrects_t4 / float(N)) if N > 0 else float('nan')
                                cost_t4 = (total_cost_t4 / float(N)) if N > 0 else float('nan')
                                curve_ours_top2_3_cyc.append((cost_t4, acc_t4))

                            logger.info(_purple(f"[{subject}] Beta curve (Ours top2->top3->cyclic): " +
                                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_top2_3_cyc])))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours top2->top3->cyclic curve: {e}")
                            curve_ours_top2_3_cyc = []

                        # ---------------- Oracle low-confidence accuracy ----------------
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
                        if len(curve_ours_top2_cyc) == len(betas):
                            curve_obj['ours_top2_to_cyclic'] = {
                                'costs': [c for c, _ in curve_ours_top2_cyc],
                                'accuracies': [a for _, a in curve_ours_top2_cyc],
                            }
                        if len(curve_ours_top2flip_cyc) == len(betas):
                            curve_obj['ours_top2flip_to_cyclic'] = {
                                'costs': [c for c, _ in curve_ours_top2flip_cyc],
                                'accuracies': [a for _, a in curve_ours_top2flip_cyc],
                            }
                        if len(curve_ours_top2_3_cyc) == len(betas):
                            curve_obj['ours_top2_to_top3_to_cyclic'] = {
                                'costs': [c for c, _ in curve_ours_top2_3_cyc],
                                'accuracies': [a for _, a in curve_ours_top2_3_cyc],
                            }

                        try:
                            default_confs = default_conf.copy()
                            default_corrects = np.array(
                                [1 if c else 0 for c in base_correct_list],
                                dtype=np.int32
                            )
                            order = np.argsort(default_confs)
                            oracle_percentiles = list(range(1, 101))
                            oracle_bottom_accs = []
                            for p in oracle_percentiles:
                                nn = max(1, int(N * (p / 100.0) + 1e-9))
                                sel = order[:nn]
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

                        curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                        if getattr(args, 'option_id_set', None):
                            curve_save_path += f'_id-{args.option_id_set}'
                        os.makedirs(curve_save_path, exist_ok=True)
                        save_results(f'{curve_save_path}/{subject}_beta_curve.jsonl', [curve_obj], metrics=None)

                except Exception as e:
                    logger.warning(f"Failed to derive cyclic/base from full for subject '{subject}': {e}")

            logging_cuda_memory_usage()

    # Finalize W&B
    try:
        if wandb_run is not None:
            import wandb
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
