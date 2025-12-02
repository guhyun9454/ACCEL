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

            # -------------------------------------------------
            # FULL permutation에서 cyclic / base / beta curve 등 파생 결과 계산
            # -------------------------------------------------
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
                    base_correct_list = []
                    cyclic_correct_list = []
                    full_correct_list = []

                    full_total = 0
                    full_corrects = 0
                    cyclic_total = 0
                    cyclic_corrects = 0

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
                        
                        # Mean Gap
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
                    mean_conf = np.asarray(mean_gap_list, dtype=np.float64)
                    arr_flip_trigger_global = np.array(flip_trigger_mask_global, dtype=bool)
                    
                    # Target Mask (Gain)
                    arr_base_correct = np.array(base_correct_list, dtype=bool)
                    arr_cyclic_correct = np.array(cyclic_correct_list, dtype=bool)
                    target_mask = (~arr_base_correct) & (arr_cyclic_correct)
                    # =========================================================

                    # Save intermediate results
                    cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                    if getattr(args, 'option_id_set', None):
                        cyclic_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(cyclic_save_path, exist_ok=True)
                    cyclic_metrics = {'type': 'metric', 'data': {'accuracy': cyclic_acc}}
                    logger.info(_purple(f"[{subject}] Cyclic ensemble accuracy: {cyclic_acc:.4f}"))
                    save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results, metrics=cyclic_metrics)

                    base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                    if getattr(args, 'option_id_set', None):
                        base_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(base_save_path, exist_ok=True)
                    base_metrics = {'type': 'metric', 'data': {'accuracy': get_accuracy(base_results)}}
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results, base_metrics)

                    logger.info(_purple(f"[{subject}] Accuracies — Full: {full_acc:.4f}, Cyclic: {cyclic_acc:.4f}, Default: {base_metrics['data']['accuracy']:.4f}"))

                    # =========================================================
                    # 2. Compute Beta Curves
                    # =========================================================
                    N = len(base_correct_list)
                    betas = [i / 10.0 for i in range(11)]
                    C_cyc = float(k)
                    
                    # Standard Cyclic
                    curve_cyc = []
                    for beta in betas:
                        n = int(N * beta + 1e-9)
                        if n > 0:
                            acc = (sum(cyclic_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                        else:
                            acc = sum(base_correct_list) / float(N)
                        cost = beta * C_cyc + (1.0 - beta) * 1.0
                        curve_cyc.append((cost, acc))

                    # Ours Top2Flip -> Cyclic
                    curve_ours_top2flip_cyc = []
                    perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0
                    
                    try:
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            if n > 0:
                                thresh = float(np.quantile(default_conf[:n], perc))
                            else:
                                thresh = float(np.quantile(default_conf, perc))

                            total_cost = 0.0
                            corrects = 0
                            
                            # Beta part: Default
                            for i in range(0, n):
                                if base_correct_list[i]: corrects += 1
                                total_cost += 1.0
                            
                            # Test part
                            for i in range(n, N):
                                if default_conf[i] >= thresh:
                                    # High Conf -> Default
                                    total_cost += 1.0
                                    if base_correct_list[i]: corrects += 1
                                else:
                                    # Low Conf -> Check Flip
                                    if arr_flip_trigger_global[i]:
                                        # Trigger -> Cyclic
                                        total_cost += float(k)
                                        if cyclic_correct_list[i]: corrects += 1
                                    else:
                                        # No Trigger -> Stop (Cost 2.0 for check)
                                        total_cost += 2.0
                                        if base_correct_list[i]: corrects += 1
                            
                            acc = corrects / float(N)
                            cost = total_cost / float(N)
                            curve_ours_top2flip_cyc.append((cost, acc))
                        
                        logger.info(_purple(f"[{subject}] Beta curve (Top2Flip): " + ", ".join([f"(c={c:.2f}, a={a:.4f})" for c, a in curve_ours_top2flip_cyc])))
                    except Exception as e:
                        logger.warning(f"Error in Top2Flip curve: {e}")

                    # Ours AvgGap -> Cyclic
                    curve_ours_avg_gap_cyc = []
                    try:
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            if n > 0:
                                thresh = float(np.quantile(default_conf[:n], perc))
                            else:
                                thresh = float(np.quantile(default_conf, perc))

                            total_cost = 0.0
                            corrects = 0
                            
                            for i in range(0, n):
                                if base_correct_list[i]: corrects += 1
                                total_cost += 1.0
                            
                            for i in range(n, N):
                                if default_conf[i] >= thresh:
                                    total_cost += 1.0
                                    if base_correct_list[i]: corrects += 1
                                else:
                                    # Low Conf -> Check Avg Gap
                                    if mean_conf[i] < thresh:
                                        total_cost += float(k)
                                        if cyclic_correct_list[i]: corrects += 1
                                    else:
                                        total_cost += 2.0
                                        if base_correct_list[i]: corrects += 1
                            
                            acc = corrects / float(N)
                            cost = total_cost / float(N)
                            curve_ours_avg_gap_cyc.append((cost, acc))

                        logger.info(_purple(f"[{subject}] Beta curve (AvgGap): " + ", ".join([f"(c={c:.2f}, a={a:.4f})" for c, a in curve_ours_avg_gap_cyc])))
                    except Exception as e:
                        logger.warning(f"Error in AvgGap curve: {e}")

                    # =========================================================
                    # 3. Create Curve Object & Run PR Analysis
                    # =========================================================
                    curve_obj = {
                        'subject': subject,
                        'k': k,
                        'betas': betas,
                        'default_accuracy': base_metrics['data']['accuracy'],
                        'cyclic': {'costs': [c for c, _ in curve_cyc], 'accuracies': [a for _, a in curve_cyc]},
                        'ours_top2flip': {'costs': [c for c, _ in curve_ours_top2flip_cyc], 'accuracies': [a for _, a in curve_ours_top2flip_cyc]},
                        'ours_avggap': {'costs': [c for c, _ in curve_ours_avg_gap_cyc], 'accuracies': [a for _, a in curve_ours_avg_gap_cyc]},
                    }

                    # PR Analysis
                    try:
                        def calc_pr_f1(pred_mask, gt_mask):
                            tp = (pred_mask & gt_mask).sum()
                            fp = (pred_mask & ~gt_mask).sum()
                            fn = (~pred_mask & gt_mask).sum()
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                            return precision, recall, f1

                        logger.info(_purple(f"[{subject}] Detailed PR Analysis (Oracle Bottom %):"))
                        
                        scan_percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                        low_conf_thresholds = np.percentile(default_conf, scan_percentiles)
                        
                        combined_stats = []
                        
                        for i, p in enumerate(scan_percentiles):
                            thr = low_conf_thresholds[i]
                            
                            # Standard: Low Conf & Flip Trigger
                            pred_mask_low = default_conf <= thr
                            pred_mask_std = pred_mask_low & arr_flip_trigger_global
                            _, _, f1_std = calc_pr_f1(pred_mask_std, target_mask)
                            
                            # AvgGap: Low Conf & (Flip Trigger OR MeanGap Low)
                            # Actually AvgGap strategy logic is: if Original Gap is low -> Check Mean Gap. 
                            # If Mean Gap < Thr -> Trigger.
                            pred_mask_avg = pred_mask_low & (mean_conf <= thr)
                            p_avg, r_avg, f1_avg = calc_pr_f1(pred_mask_avg, target_mask)
                            
                            logger.info(f"  [Bottom {p}%] Std F1: {f1_std:.4f} | AvgGap F1: {f1_avg:.4f} | AvgGap Rec: {r_avg:.4f} | AvgGap Prec: {p_avg:.4f}")
                            
                            combined_stats.append({
                                'percentile': p,
                                'threshold': float(thr),
                                'f1_standard': float(f1_std),
                                'f1_avggap': float(f1_avg),
                                'recall_avggap': float(r_avg),
                                'prec_avggap': float(p_avg)
                            })
                        
                        curve_obj['pr_analysis'] = combined_stats

                        # Gap vs Flip Analysis
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
                        logger.warning(f"PR Analysis Failed: {e}")
                        import traceback
                        traceback.print_exc()

                    # Save Curve Object
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