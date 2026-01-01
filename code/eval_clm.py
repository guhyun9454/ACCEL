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

# -------------------------
# [FIX] Safe NVML init
# -------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

logger = logging.getLogger(__name__)


def logging_cuda_memory_usage():
    if not _NVML_OK:
        logger.info("******** Memory usage ********")
        logger.info("NVML unavailable; skipping GPU memory usage logging.")
        return

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
            import wandb  # type: ignore
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

            if args.setting == 'full' and len(results) > 0:
                try:
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
                    
                    full_total = 0
                    full_corrects = 0
                    cyclic_total = 0
                    cyclic_corrects = 0

                    per_sample_probs = []
                    base_probs_list = []
                    ideals = []
                    
                    # [FIXED] Initialize correctness lists
                    base_correct_list = []
                    cyclic_correct_list = []
                    # [FIXED] This was missing and caused NameError
                    full_correct_list = []

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

                        # Cyclic
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
                        agg_cyc = _aggregate_probs_over_permutations(cyc_probs, cyc_perms, k)
                        pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                        corr_cyc = (pred_cyc == data['ideal'])
                        
                        cyclic_correct_list.append(corr_cyc)
                        if corr_cyc:
                            cyclic_corrects += 1
                        cyclic_total += 1

                        # Base
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

                        # Full
                        agg_full = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                        pred_full = option_ids[int(np.argmax(agg_full))]
                        corr_full = (pred_full == data['ideal'])

                        # [FIXED] Append full correctness for later beta-curve logic
                        full_correct_list.append(corr_full)

                        if corr_full:
                            full_corrects += 1
                        full_total += 1

                    # =========================================================
                    # 1. Pre-compute ALL Confidence stats & Triggers
                    # =========================================================
                    default_conf = [] # Gap (Top1 - Top2)
                    base_conf_max = [] # Confidence (Top1 Prob)
                    mean_gap_list = [] # Avg Gap of (Base + Flip)
                    flip_trigger_mask_global = []

                    for i, bp in enumerate(base_probs_list):
                        # Base Stats
                        vals = np.sort(bp)[::-1]
                        if vals.shape[0] < 2:
                            top1, top2 = (vals[0], 0.0) if vals.shape[0] > 0 else (0.0, 0.0)
                        else:
                            top1, top2 = vals[0], vals[1]
                        base_conf_max.append(top1)
                        default_conf.append(top1 - top2) # Gap calculation

                        # Avg Gap & Flip Trigger
                        sorted_idx = np.argsort(bp)[::-1]
                        top1_idx = int(sorted_idx[0])
                        top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx
                        perm_swap = list(identity_perm)
                        perm_swap[top1_idx], perm_swap[top2_idx] = perm_swap[top2_idx], perm_swap[top1_idx]
                        swap_idx = perm_index_map.get(tuple(perm_swap), identity_idx)
                        
                        probs_base_raw = per_sample_probs[i][identity_idx]
                        probs_swap_raw = per_sample_probs[i][swap_idx]
                        agg_base = _aggregate_probs_over_permutations([probs_base_raw.tolist()], [perm_list[identity_idx]], k) 
                        agg_swap = _aggregate_probs_over_permutations([probs_swap_raw.tolist()], [perm_list[swap_idx]], k)
                        
                        mean_probs = (agg_base + agg_swap) / 2.0
                        vals_mean = np.sort(mean_probs)[::-1]
                        mean_gap = vals_mean[0] - vals_mean[1] if len(vals_mean) > 1 else 0.0
                        mean_gap_list.append(mean_gap)

                        # Flip Check
                        pred_base_content = option_ids[int(np.argmax(agg_base))]
                        pred_swap_content = option_ids[int(np.argmax(agg_swap))]
                        if pred_base_content != pred_swap_content:
                            flip_trigger_mask_global.append(True)
                        else:
                            flip_trigger_mask_global.append(False)

                    # Convert to numpy
                    default_conf = np.asarray(default_conf, dtype=np.float64)
                    base_conf_max = np.asarray(base_conf_max, dtype=np.float64)
                    mean_conf = np.asarray(mean_gap_list, dtype=np.float64)
                    arr_flip_trigger_global = np.array(flip_trigger_mask_global, dtype=bool)
                    
                    # Target Mask (Gain)
                    arr_base_correct = np.array(base_correct_list, dtype=bool)
                    arr_cyclic_correct = np.array(cyclic_correct_list, dtype=bool)
                    target_mask = (~arr_base_correct) & (arr_cyclic_correct)
                    
                    total_positives = int(target_mask.sum())
                    # =========================================================

                    # Save cyclic-derived results
                    cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                    if getattr(args, 'option_id_set', None):
                        cyclic_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(cyclic_save_path, exist_ok=True)
                    
                    # Calculate cyclic_acc here BEFORE using it in metrics
                    if cyclic_total > 0:
                        cyclic_acc = cyclic_corrects / cyclic_total
                        cyclic_metrics = {'type': 'metric', 'data': {'accuracy': cyclic_acc}}
                        logger.info(_purple(f"[{subject}] Cyclic ensemble accuracy: {cyclic_acc:.4f}"))
                    else:
                        cyclic_acc = float('nan')
                        cyclic_metrics = None

                    save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results, metrics=cyclic_metrics)
                    logger.info(_orange(f"Derived and saved cyclic results: {subject}"))

                    # Save base-derived results
                    base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                    if getattr(args, 'option_id_set', None):
                        base_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(base_save_path, exist_ok=True)
                    base_metrics = {'type': 'metric', 'data': {'accuracy': get_accuracy(base_results)}}
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

                        # 4. Ours top2flip -> cyclic
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
                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                total_cost_t2f = 0.0
                                corrects_t2f = 0
                                
                                # Log stats for interesting Betas
                                test_indices = list(range(n, N))
                                if len(test_indices) > 0 and beta in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
                                    test_default_conf = default_conf[n:]
                                    test_flip_trigger = arr_flip_trigger_global[n:]
                                    test_target_mask = target_mask[n:]
                                    
                                    pred_mask_low_conf = test_default_conf <= thresh
                                    pred_mask_combined = pred_mask_low_conf & test_flip_trigger
                                    
                                    pr_p, pr_r, pr_f1 = calc_pr_f1(pred_mask_combined, test_target_mask)
                                    logger.info(f"  [Beta {beta:.1f}] Sample Size: {n} | Est. Threshold ({perc*100:.0f}%): {thresh:.4f} | Real F1: {pr_f1:.4f} | Prec: {pr_p:.4f} | Rec: {pr_r:.4f}")

                                for i in range(0, n):
                                    if base_correct_list[i]: corrects_t2f += 1
                                    total_cost_t2f += 1.0

                                for i in range(n, N):
                                    if default_conf[i] >= thresh:
                                        total_cost_t2f += 1.0
                                        if base_correct_list[i]: corrects_t2f += 1
                                        continue

                                    if arr_flip_trigger_global[i]: 
                                        total_cost_t2f += float(k)
                                        if cyclic_correct_list[i]: corrects_t2f += 1
                                    else:
                                        total_cost_t2f += 2.0 
                                        if base_correct_list[i]: corrects_t2f += 1

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

                        # 5. Ours AvgGap -> Cyclic (Cost Analysis)
                        curve_ours_avg_gap_cyc = []
                        try:
                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                total_cost_avg = 0.0
                                corrects_avg = 0

                                for i in range(0, n):
                                    if base_correct_list[i]: corrects_avg += 1
                                    total_cost_avg += 1.0

                                for i in range(n, N):
                                    if default_conf[i] >= thresh:
                                        total_cost_avg += 1.0
                                        if base_correct_list[i]: corrects_avg += 1
                                        continue
                                    
                                    if mean_conf[i] < thresh:
                                        total_cost_avg += float(k)
                                        if cyclic_correct_list[i]: corrects_avg += 1
                                    else:
                                        total_cost_avg += 2.0
                                        if base_correct_list[i]: corrects_avg += 1

                                acc_avg = (corrects_avg / float(N)) if N > 0 else float('nan')
                                cost_avg = (total_cost_avg / float(N)) if N > 0 else float('nan')
                                curve_ours_avg_gap_cyc.append((cost_avg, acc_avg))

                            logger.info(_purple(f"[{subject}] Beta curve (Ours AvgGap->cyclic): " +
                                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_avg_gap_cyc])))

                        except Exception as e:
                            logger.warning(f"Failed to compute Ours AvgGap->cyclic curve: {e}")
                            curve_ours_avg_gap_cyc = []

                        # =========================================================
                        # [NEW] Absolute-threshold grid (thr1/thr2) for AvgGap Strategy
                        #   - x-axis: thr1 (base gap threshold)
                        #   - y-axis: thr2 (avg gap threshold)
                        #   - values: accuracy of policy:
                        #       if default_gap > thr1: base
                        #       else if mean_gap < thr2: cyclic
                        #       else: base
                        # =========================================================
                        abs_thr_grid = None
                        try:
                            thr_step = 0.1  # 10% unit in absolute probability gap space
                            thr1_vals = np.round(np.arange(0.0, 1.0 + 1e-9, thr_step), 10)
                            thr2_vals = np.round(np.arange(0.0, 1.0 + 1e-9, thr_step), 10)

                            acc_grid = np.zeros((len(thr2_vals), len(thr1_vals)), dtype=np.float64)
                            # Cost model (consistent with earlier AvgGap->cyclic cost analysis):
                            # - base-only: 1
                            # - if low-conf then compute flip-check (base+swap): 2
                            # - if triggered -> cyclic: k
                            cost_grid = np.zeros((len(thr2_vals), len(thr1_vals)), dtype=np.float64)
                            cell_rows = []
                            for yi, thr2 in enumerate(thr2_vals):
                                for xi, thr1 in enumerate(thr1_vals):
                                    # Policy:
                                    # - if default_gap > thr1: base
                                    # - else if mean_gap < thr2: cyclic
                                    # - else: base (after flip-check)
                                    low_conf = (default_conf <= thr1)
                                    use_cyclic = low_conf & (mean_conf < thr2)
                                    use_flipcheck_only = low_conf & (~use_cyclic)

                                    acc_grid[yi, xi] = float(np.mean(np.where(use_cyclic, arr_cyclic_correct, arr_base_correct)))
                                    # Cost per-sample:
                                    # - cyclic -> k
                                    # - low_conf but not cyclic -> 2 (base + swap check)
                                    # - otherwise -> 1
                                    per_sample_cost = np.where(use_cyclic, float(k), np.where(use_flipcheck_only, 2.0, 1.0))
                                    cost_grid[yi, xi] = float(np.mean(per_sample_cost))
                                    cell_rows.append({
                                        'thr1': float(thr1),
                                        'thr2': float(thr2),
                                        'acc': float(acc_grid[yi, xi]),
                                        'cost': float(cost_grid[yi, xi]),
                                        'xi': int(xi),
                                        'yi': int(yi),
                                    })

                            # Best by accuracy; break ties by lower cost (and then smaller thresholds for determinism)
                            max_acc = float(np.max(acc_grid))
                            cand_ys, cand_xs = np.where(acc_grid == max_acc)
                            if len(cand_ys) == 0:
                                best_y, best_x = 0, 0
                            else:
                                cand_costs = cost_grid[cand_ys, cand_xs]
                                min_cost = float(np.min(cand_costs))
                                keep = (cand_costs == min_cost)
                                cand_ys2 = cand_ys[keep]
                                cand_xs2 = cand_xs[keep]
                                # deterministic: pick smallest thr2 then thr1
                                order = np.lexsort((cand_xs2, cand_ys2))
                                best_y = int(cand_ys2[order[0]])
                                best_x = int(cand_xs2[order[0]])

                            best_thr1 = float(thr1_vals[best_x])
                            best_thr2 = float(thr2_vals[best_y])
                            best_acc = float(acc_grid[best_y, best_x])
                            best_cost = float(cost_grid[best_y, best_x])

                            abs_thr_grid = {
                                'thr_step': float(thr_step),
                                'thr1_vals': [float(v) for v in thr1_vals.tolist()],
                                'thr2_vals': [float(v) for v in thr2_vals.tolist()],
                                'acc_grid': acc_grid.tolist(),
                                'cost_grid': cost_grid.tolist(),
                                'cells': cell_rows,
                                'cost_model': {'base': 1.0, 'base_plus_flipcheck': 2.0, 'cyclic': float(k)},
                                'best': {'thr1': best_thr1, 'thr2': best_thr2, 'acc': best_acc, 'cost': best_cost},
                            }

                            logger.info(_purple(
                                f"[{subject}] AvgGap abs-threshold grid best: "
                                f"acc={best_acc:.4f}, cost={best_cost:.2f} @ (thr1={best_thr1:.1f}, thr2={best_thr2:.1f})"
                            ))

                            # W&B heatmap logging (image) - only if enabled
                            if wandb_run is not None:
                                import matplotlib
                                # Avoid backend switching errors across repeated subject loops
                                if "matplotlib.pyplot" not in sys.modules:
                                    matplotlib.use("Agg")
                                import matplotlib.pyplot as plt

                                fig_w = max(8.0, 0.55 * len(thr1_vals) + 3.0)
                                fig_h = max(8.0, 0.55 * len(thr2_vals) + 3.0)
                                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                                # Relative color scaling (auto) so small differences are visible.
                                # Use robust percentiles to avoid a single outlier flattening the colormap.
                                flat = acc_grid[np.isfinite(acc_grid)].reshape(-1)
                                if flat.size > 0:
                                    vmin = float(np.percentile(flat, 5))
                                    vmax = float(np.percentile(flat, 95))
                                    if vmax - vmin < 1e-8:
                                        vmin = float(np.min(flat))
                                        vmax = float(np.max(flat))
                                    if vmax - vmin < 1e-8:
                                        vmin = max(0.0, vmin - 0.01)
                                        vmax = min(1.0, vmax + 0.01)
                                else:
                                    vmin, vmax = 0.0, 1.0
                                im = ax.imshow(
                                    acc_grid,
                                    origin='lower',
                                    vmin=vmin,
                                    vmax=vmax,
                                    cmap='viridis',
                                    aspect='auto',
                                )
                                ax.set_xticks(np.arange(len(thr1_vals)))
                                ax.set_xticklabels([f"{v:.1f}" for v in thr1_vals], rotation=45, ha='right')
                                ax.set_yticks(np.arange(len(thr2_vals)))
                                ax.set_yticklabels([f"{v:.1f}" for v in thr2_vals])
                                ax.set_xlabel("thr1 (base gap)")
                                ax.set_ylabel("thr2 (avg gap)")
                                ax.set_title(f"{subject} | AvgGap abs-threshold grid (acc / cost)")
                                cbar = fig.colorbar(im, ax=ax)
                                cbar.set_label("accuracy")

                                # Annotate each cell with accuracy + cost
                                for yi in range(acc_grid.shape[0]):
                                    for xi in range(acc_grid.shape[1]):
                                        val_acc = float(acc_grid[yi, xi])
                                        val_cost = float(cost_grid[yi, xi])
                                        text_color = "white" if im.norm(val_acc) < 0.5 else "black"
                                        ax.text(
                                            xi, yi, f"{val_acc:.3f}\n{val_cost:.2f}",
                                            ha="center", va="center",
                                            color=text_color,
                                            fontsize=7,
                                        )

                                # Highlight best cell
                                ax.scatter([best_x], [best_y], s=140, facecolors='none', edgecolors='red', linewidths=2)

                                fig.tight_layout()

                                import wandb  # type: ignore
                                wandb.log({
                                    f"{subject}/avggap_abs_thr_acc_grid": wandb.Image(fig),
                                    f"{subject}/avggap_abs_thr_best_acc": best_acc,
                                    f"{subject}/avggap_abs_thr_best_cost": best_cost,
                                    f"{subject}/avggap_abs_thr_best_thr1": best_thr1,
                                    f"{subject}/avggap_abs_thr_best_thr2": best_thr2,
                                })
                                plt.close(fig)

                        except Exception as e:
                            logger.warning(f"Failed to compute/log AvgGap abs-threshold grid: {e}")
                            import traceback
                            traceback.print_exc()
                            abs_thr_grid = None

                        curve_obj = {
                            'subject': subject,
                            'k': k,
                            'betas': betas,
                            'default_accuracy': base_metrics['data']['accuracy'],
                            'cyclic': {'costs': [c for c, _ in curve_cyc], 'accuracies': [a for _, a in curve_cyc]},
                            'ours_top2flip': {'costs': [c for c, _ in curve_ours_top2flip_cyc], 'accuracies': [a for _, a in curve_ours_top2flip_cyc]},
                            'ours_avggap': {'costs': [c for c, _ in curve_ours_avg_gap_cyc], 'accuracies': [a for _, a in curve_ours_avg_gap_cyc]},
                        }

                        if abs_thr_grid is not None:
                            curve_obj['abs_threshold_grid_avggap'] = abs_thr_grid

                        # =========================================================
                        # [NEW] Gain/Loss & Net Accuracy Analysis (AvgGap Strategy)
                        # =========================================================
                        try:
                            logger.info(_purple(f"[{subject}] Detailed PR & Gain/Loss Analysis (Oracle Bottom %):"))
                            
                            scan_percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                            low_conf_thresholds = np.percentile(default_conf, scan_percentiles)
                            
                            combined_stats = []
                            
                            for i, p in enumerate(scan_percentiles):
                                thr = low_conf_thresholds[i]
                                
                                # AvgGap Strategy: Trigger if (Low Conf Original) AND (Mean Gap < Thr)
                                pred_mask_low = default_conf <= thr
                                pred_mask_avg = pred_mask_low & (mean_conf <= thr)
                                
                                trigger_count = int(pred_mask_avg.sum())
                                gain_mask = (~arr_base_correct) & (arr_cyclic_correct) & pred_mask_avg
                                gain_count = int(gain_mask.sum())
                                loss_mask = (arr_base_correct) & (~arr_cyclic_correct) & pred_mask_avg
                                loss_count = int(loss_mask.sum())
                                net_gain = gain_count - loss_count
                                
                                precision = gain_count / trigger_count if trigger_count > 0 else 0.0
                                recall = gain_count / total_positives if total_positives > 0 else 0.0
                                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                                
                                logger.info(f"  [Bottom {p}%] (Thr={thr:.4f}) Trigger: {trigger_count} | Gain: +{gain_count} | Loss: -{loss_count} | Net: {net_gain:+d} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
                                
                                combined_stats.append({
                                    'percentile': p,
                                    'threshold': float(thr),
                                    'trigger_count': trigger_count,
                                    'gain': gain_count,
                                    'loss': loss_count,
                                    'net_gain': net_gain,
                                    'f1': f1,
                                    'recall': recall,
                                    'precision': precision
                                })
                            
                            curve_obj['pr_analysis'] = combined_stats

                            bins = [i / 10.0 for i in range(11)]
                            gap_flip_stats = []
                            logger.info(_purple(f"[{subject}] Gap vs Flip Rate:"))
                            for i in range(len(bins) - 1):
                                low, high = bins[i], bins[i+1]
                                mask = (default_conf >= low) & (default_conf <= high) if i == 9 else (default_conf >= low) & (default_conf < high)
                                count = int(mask.sum())
                                flipped = int(arr_flip_trigger_global[mask].sum()) if count > 0 else 0
                                rate = flipped / count if count > 0 else 0.0
                                if count > 0:
                                    logger.info(f"  Gap {low:.1f}~{high:.1f}: {rate*100:.1f}% flipped ({flipped}/{count})")
                                gap_flip_stats.append({'low': low, 'high': high, 'count': count, 'flipped': flipped, 'rate': rate})
                            
                            curve_obj['gap_flip_analysis'] = gap_flip_stats

                        except Exception as e:
                            logger.warning(f"Analysis Failed: {e}")
                            import traceback
                            traceback.print_exc()

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
            import wandb  # type: ignore
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()