# eval_clm.py
# -*- coding: utf-8 -*-

import os
import sys
import gc
import json
import copy
import logging
import random
import math
from functools import partial
from typing import List, Optional, Tuple
from collections import Counter

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
    # 1) Comparative mode: option_id_sets 두 개 비교 (로직 유지)
    # ---------------------------------------------------------
    if getattr(args, 'option_id_sets', None) and len(args.option_id_sets) == 2:
        from types import SimpleNamespace
        id_set_a, id_set_b = args.option_id_sets[0], args.option_id_sets[1]
        logger.info("Comparative mode logic exists but skipped in this summary for focus.")
        pass 

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

                    # Derived results containers
                    cyclic_results = []
                    base_results = []
                    base_correct_list = []
                    cyclic_correct_list = []
                    full_correct_list = []

                    full_total = 0
                    full_corrects = 0
                    cyclic_total = 0
                    cyclic_corrects = 0

                    # Data collections for analysis
                    per_sample_probs = []
                    base_probs_list = []
                    ideals = []

                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data['probs']
                        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                            continue
                        
                        probs_seq_np = np.asarray(probs_seq, dtype=np.float64)
                        per_sample_probs.append(probs_seq_np)
                        ideals.append(data['ideal'])

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
                        base_probs_list.append(base_probs)
                        
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

                    # =========================================================
                    # [Pre-computation] Compute Confidence stats BEFORE Analysis
                    # =========================================================
                    default_conf = [] # Gap (Top1 - Top2)
                    base_conf_max = [] # Confidence (Top1 Prob)
                    
                    for bp in base_probs_list:
                        vals = np.sort(bp)[::-1]
                        if vals.shape[0] < 2:
                            top1, top2 = (vals[0], 0.0) if vals.shape[0] > 0 else (0.0, 0.0)
                        else:
                            top1, top2 = vals[0], vals[1]
                        base_conf_max.append(top1)
                        default_conf.append(top1 - top2) # Gap calculation

                    # Convert lists to numpy for vectorized ops
                    default_conf = np.asarray(default_conf, dtype=np.float64)
                    base_conf_max = np.asarray(base_conf_max, dtype=np.float64)
                    
                    # Compute Global Flip Trigger for Ground Truth comparison
                    flip_trigger_mask_global = []
                    for i in range(len(base_probs_list)):
                        # Re-calculate Flip status
                        probs_seq = per_sample_probs[i]
                        base_probs = base_probs_list[i]
                        sorted_idx = np.argsort(base_probs)[::-1]
                        top1_idx = int(sorted_idx[0])
                        top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx
                        perm_swap = list(identity_perm)
                        perm_swap[top1_idx], perm_swap[top2_idx] = perm_swap[top2_idx], perm_swap[top1_idx]
                        perm_swap_t = tuple(perm_swap)
                        swap_idx = perm_index_map.get(perm_swap_t, identity_idx)
                        probs_swap = probs_seq[swap_idx]
                        
                        agg_base = _aggregate_probs_over_permutations([probs_seq[identity_idx].tolist()], [perm_list[identity_idx]], k) 
                        agg_swap = _aggregate_probs_over_permutations([probs_swap.tolist()], [perm_list[swap_idx]], k)
                        pred_base_content = option_ids[int(np.argmax(agg_base))]
                        pred_swap_content = option_ids[int(np.argmax(agg_swap))]
                        
                        if pred_base_content != pred_swap_content:
                            flip_trigger_mask_global.append(True)
                        else:
                            flip_trigger_mask_global.append(False)
                    arr_flip_trigger_global = np.array(flip_trigger_mask_global, dtype=bool)
                    
                    # Target (Gain) Mask
                    arr_base_correct = np.array(base_correct_list, dtype=bool)
                    arr_cyclic_correct = np.array(cyclic_correct_list, dtype=bool)
                    target_mask = (~arr_base_correct) & (arr_cyclic_correct)
                    # =========================================================

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
                    logger.info(_orange(f"Derived and saved cyclic results: {subject}"))

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
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results, base_metrics)
                    logger.info(_orange(f"Derived and saved base results: {subject}"))

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

                        # 1. Standard Cyclic & Full Curves
                        curve_cyc = []
                        curve_full = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            # Cyclic Mix
                            if n > 0:
                                acc_cyc_mix = (sum(cyclic_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                            else:
                                acc_cyc_mix = sum(base_correct_list) / float(N)
                            cost_cyc = beta * C_cyc + (1.0 - beta) * 1.0
                            curve_cyc.append((cost_cyc, acc_cyc_mix))
                            # Full Mix
                            if n > 0:
                                acc_full_mix = (sum(full_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                            else:
                                acc_full_mix = sum(base_correct_list) / float(N)
                            cost_full_mix = beta * C_full + (1.0 - beta) * 1.0
                            curve_full.append((cost_full_mix, acc_full_mix))

                        # 2. Ours (Cascading Ensemble)
                        perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0
                        curve_ours = []
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
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects += 1
                                    total_cost += 1.0
                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    selected = [order_indices[0]]
                                    agg = _aggregate_probs_over_permutations([probs_seq[j].tolist() for j in selected], [perm_list[j] for j in selected], k)
                                    vals = np.sort(agg)[::-1]
                                    cur_gap = vals[0] - vals[1] if len(vals) > 1 else vals[0]
                                    t = 1
                                    while (cur_gap < thresh) and (t < len(order_indices)):
                                        selected.append(order_indices[t])
                                        agg = _aggregate_probs_over_permutations([probs_seq[j].tolist() for j in selected], [perm_list[j] for j in selected], k)
                                        vals = np.sort(agg)[::-1]
                                        cur_gap = vals[0] - vals[1] if len(vals) > 1 else vals[0]
                                        t += 1
                                    pred_letter = option_ids[int(np.argmax(agg))]
                                    if pred_letter == ideals[i]:
                                        corrects += 1
                                    total_cost += float(len(selected))
                                acc_ours = (corrects / float(N)) if N > 0 else float('nan')
                                cost_ours = (total_cost / float(N)) if N > 0 else float('nan')
                                curve_ours.append((cost_ours, acc_ours))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours curve: {e}")
                            curve_ours = []

                        # 3. Switch-Full / Switch-Cyclic
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
                                # switch-cyclic
                                total_cost_sc = 0.0
                                corrects_sc = 0
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects_sf += 1
                                        corrects_sc += 1
                                    total_cost_sf += 1.0
                                    total_cost_sc += 1.0
                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    is_ambiguous = (default_conf[i] < thresh)
                                    if is_ambiguous:
                                        agg_full = _aggregate_probs_over_permutations([probs_seq[j].tolist() for j in range(len(perm_list))], [perm_list[j] for j in range(len(perm_list))], k)
                                        pred_f = option_ids[int(np.argmax(agg_full))]
                                        total_cost_sf += float(len(perm_list))
                                    else:
                                        pred_f = option_ids[int(np.argmax(base_probs_list[i]))]
                                        total_cost_sf += 1.0
                                    if pred_f == ideals[i]:
                                        corrects_sf += 1
                                    if is_ambiguous:
                                        agg_cyc = _aggregate_probs_over_permutations([probs_seq[j].tolist() for j in cyclic_indices], [perm_list[j] for j in cyclic_indices], k)
                                        pred_c = option_ids[int(np.argmax(agg_cyc))]
                                        total_cost_sc += float(k)
                                    else:
                                        pred_c = option_ids[int(np.argmax(base_probs_list[i]))]
                                        total_cost_sc += 1.0
                                    if pred_c == ideals[i]:
                                        corrects_sc += 1
                                acc_sf = corrects_sf / float(N)
                                cost_sf = total_cost_sf / float(N)
                                curve_ours_switch_full.append((cost_sf, acc_sf))
                                acc_sc = corrects_sc / float(N)
                                cost_sc = total_cost_sc / float(N)
                                curve_ours_switch_cyc.append((cost_sc, acc_sc))
                        except Exception as e:
                            logger.warning(f"Failed to compute switch curves: {e}")
                            curve_ours_switch_full = []
                            curve_ours_switch_cyc = []

                        # =========================================================================
                        # 4. Ours top2flip -> cyclic (Beta Analysis + PR Metrics Logging)
                        # =========================================================================
                        curve_ours_top2flip_cyc = []
                        try:
                            _unused_gap_frac = getattr(args, "ours_top2_gap_frac", 0.0)
                            
                            # Helper for PR
                            def calc_pr_f1(pred_mask, gt_mask):
                                tp = (pred_mask & gt_mask).sum()
                                fp = (pred_mask & ~gt_mask).sum()
                                fn = (~pred_mask & gt_mask).sum()
                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                                return precision, recall, f1

                            logger.info(_purple(f"[{subject}] Beta-wise Threshold & PR Analysis (Low Conf [Gap] AND Flip Changed):"))

                            for beta in betas:
                                n = int(N * beta + 1e-9)

                                # 1. Estimate Threshold from Beta sample (or 0 if n=0)
                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    # Fallback: use Oracle for 0.0 just to show a baseline reference
                                    thresh = float(np.quantile(default_conf, perc))

                                total_cost_t2f = 0.0
                                corrects_t2f = 0
                                
                                # 2. Evaluate Strategy on Test Set (1-Beta)
                                # We check how well the estimated threshold performs on the remaining data.
                                test_indices = list(range(n, N))
                                if len(test_indices) > 0:
                                    # Subset arrays
                                    test_default_conf = default_conf[n:]
                                    test_flip_trigger = arr_flip_trigger_global[n:]
                                    test_target_mask = target_mask[n:]
                                    
                                    # Apply Strategy: (Gap <= Thr) AND (Flip Changed)
                                    pred_mask_low_conf = test_default_conf <= thresh
                                    pred_mask_combined = pred_mask_low_conf & test_flip_trigger
                                    
                                    # Metrics
                                    pr_p, pr_r, pr_f1 = calc_pr_f1(pred_mask_combined, test_target_mask)
                                    
                                    # Log stats for interesting Betas
                                    if beta in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]: 
                                        logger.info(f"  [Beta {beta:.1f}] Sample Size: {n} | Est. Threshold ({perc*100:.0f}%): {thresh:.4f} | Real F1: {pr_f1:.4f} | Prec: {pr_p:.4f} | Rec: {pr_r:.4f}")
                                else:
                                    # Beta=1.0, no test set.
                                    pass

                                # 3. Calculate Curve (Cost/Acc) - Standard Logic
                                # Beta part: Default
                                for i in range(0, n):
                                    bp = base_probs_list[i]
                                    pred_letter = option_ids[int(np.argmax(bp))]
                                    if pred_letter == ideals[i]:
                                        corrects_t2f += 1
                                    total_cost_t2f += 1.0

                                # (1-Beta) part: Strategy
                                for i in range(n, N):
                                    # Check Low Conf (using Estimated Threshold)
                                    if default_conf[i] >= thresh:
                                        total_cost_t2f += 1.0
                                        pred_letter = option_ids[int(np.argmax(base_probs_list[i]))]
                                        if pred_letter == ideals[i]:
                                            corrects_t2f += 1
                                        continue

                                    # Low Conf -> Check Flip
                                    if arr_flip_trigger_global[i]: # Pre-computed flip status
                                        # Triggered -> Run Cyclic
                                        total_cost_t2f += float(k)
                                        # Use Cyclic Prediction
                                        cyc_probs = [per_sample_probs[i][j].tolist() for j in cyclic_indices]
                                        cyc_perms = [perm_list[j] for j in cyclic_indices]
                                        agg_cyc = _aggregate_probs_over_permutations(cyc_probs, cyc_perms, k)
                                        pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                                        if pred_cyc == ideals[i]:
                                            corrects_t2f += 1
                                    else:
                                        # Not Triggered -> Stop (Cost = 2.0 for Base+Swap)
                                        total_cost_t2f += 2.0 
                                        pred_letter = option_ids[int(np.argmax(base_probs_list[i]))]
                                        if pred_letter == ideals[i]:
                                            corrects_t2f += 1

                                acc_t2f = (corrects_t2f / float(N)) if N > 0 else float('nan')
                                cost_t2f = (total_cost_t2f / float(N)) if N > 0 else float('nan')
                                curve_ours_top2flip_cyc.append((cost_t2f, acc_t2f))

                            logger.info(_purple(f"[{subject}] Beta curve (Ours top2flip->cyclic): " +
                                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_top2flip_cyc])))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours top2flip->cyclic curve: {e}")
                            import traceback
                            traceback.print_exc()
                            curve_ours_top2flip_cyc = []

                        # =========================================================================
                        # [NEW] Precision / Recall / F1 Analysis (Strictly User's Logic)
                        # Requirement: Low Confidence (Gap based) -> Then Flip Check -> Then Cyclic
                        # =========================================================================
                        # (Keeping this Oracle analysis for reference as well)
                        pr_analysis_results = {}
                        try:
                            # 3. Analyze Low Conf (Switch-Cyclic Style) & Combined Strategy
                            scan_percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                            
                            combined_stats = [] 
                            
                            # Use default_conf (Gap) which is exactly what switch_cyclic uses
                            low_conf_thresholds = np.percentile(default_conf, scan_percentiles)

                            logger.info(_purple(f"[{subject}] Detailed ORACLE Analysis (If we knew full distribution):"))

                            for i, p in enumerate(scan_percentiles):
                                thr = low_conf_thresholds[i]
                                pred_mask_low_conf = default_conf <= thr 
                                pred_mask_combined = pred_mask_low_conf & arr_flip_trigger_global
                                p_comb, r_comb, f1_comb, tp_comb, fp_comb, fn_comb = calc_pr_f1(pred_mask_combined, target_mask)
                                
                                logger.info(f"  [Oracle Bottom {p}%] Gap Threshold: {thr:.4f} | Trigger: {pred_mask_combined.mean()*100:.2f}% | F1: {f1_comb:.4f} | Prec: {p_comb:.4f} | Rec: {r_comb:.4f}")

                                combined_stats.append({
                                    'percentile': p,
                                    'threshold': float(thr),
                                    'precision': float(p_comb),
                                    'recall': float(r_comb),
                                    'f1': float(f1_comb),
                                    'trigger_rate': float(pred_mask_combined.mean())
                                })

                            pr_analysis_results = {
                                'combined_gap_flip_stats': combined_stats,
                            }

                        except Exception as e:
                            logger.warning(f"Failed to compute PR Analysis: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # =========================================================================
                        # [NEW] Gap vs Flip Rate Correlation Analysis
                        # =========================================================================
                        gap_flip_stats = []
                        try:
                            bins = [i / 10.0 for i in range(11)]
                            logger.info(_purple(f"[{subject}] Gap vs Flip Rate Analysis:"))
                            for i in range(len(bins) - 1):
                                low = bins[i]
                                high = bins[i+1]
                                if i == len(bins) - 2:
                                    mask_bin = (default_conf >= low) & (default_conf <= high)
                                else:
                                    mask_bin = (default_conf >= low) & (default_conf < high)
                                count_total = int(mask_bin.sum())
                                if count_total > 0:
                                    count_flip = int(arr_flip_trigger_global[mask_bin].sum())
                                    flip_rate = count_flip / count_total
                                else:
                                    count_flip = 0
                                    flip_rate = 0.0
                                stat_item = {
                                    'gap_range_start': low,
                                    'gap_range_end': high,
                                    'total_samples': count_total,
                                    'flipped_samples': count_flip,
                                    'flip_rate': flip_rate
                                }
                                gap_flip_stats.append(stat_item)
                                if count_total > 0:
                                    logger.info(f"  Gap {low:.1f}~{high:.1f}: {flip_rate*100:5.1f}% flipped ({count_flip}/{count_total})")
                            curve_obj['gap_flip_analysis'] = gap_flip_stats
                        except Exception as e:
                            logger.warning(f"Failed to compute Gap vs Flip stats: {e}")
                            
                        # =========================================================================

                        # ---------------- Curve Object Packaging ----------------
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
                            'pr_analysis': pr_analysis_results
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
                        if len(curve_ours_top2flip_cyc) == len(betas):
                            curve_obj['ours_top2flip_to_cyclic'] = {
                                'costs': [c for c, _ in curve_ours_top2flip_cyc],
                                'accuracies': [a for _, a in curve_ours_top2flip_cyc],
                            }

                        curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                        if getattr(args, 'option_id_set', None):
                            curve_save_path += f'_id-{args.option_id_set}'
                        os.makedirs(curve_save_path, exist_ok=True)
                        save_results(f'{curve_save_path}/{subject}_beta_curve.jsonl', [curve_obj], metrics=None)

                except Exception as e:
                    logger.warning(f"Failed to derive cyclic/base from full for subject '{subject}': {e}")
                    import traceback
                    traceback.print_exc()

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