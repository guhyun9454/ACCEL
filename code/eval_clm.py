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

                    # -----------------------------------------------------------------
                    # [Pre-Computation Phase]
                    # Gather all necessary metrics for every sample
                    # -----------------------------------------------------------------
                    
                    full_data_records = []
                    
                    full_total = 0
                    full_corrects = 0
                    
                    # For quick base/cyclic results saving
                    cyclic_results_to_save = []
                    base_results_to_save = []
                    base_correct_count = 0
                    cyclic_correct_count = 0

                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data['probs']
                        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                            continue
                        
                        probs_seq_np = np.asarray(probs_seq, dtype=np.float64)
                        ideal = data['ideal']

                        # 1. Base Logic
                        base_probs = probs_seq_np[identity_idx]
                        pred_base_idx = int(np.argmax(base_probs))
                        pred_base = option_ids[pred_base_idx]
                        is_base_correct = (pred_base == ideal)
                        
                        base_vals = np.sort(base_probs)[::-1]
                        top1_val = base_vals[0]
                        top2_val = base_vals[1] if len(base_vals) > 1 else 0.0
                        gap_val = top1_val - top2_val

                        # 2. Cyclic Logic
                        cyc_probs_list = [probs_seq[idx] for idx in cyclic_indices]
                        cyc_perms = [tuple((i + s) % k for i in range(k)) for s in range(k)]
                        agg_cyc = _aggregate_probs_over_permutations(cyc_probs_list, cyc_perms, k)
                        pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                        is_cyclic_correct = (pred_cyc == ideal)

                        # 3. Full Logic
                        agg_full = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                        pred_full = option_ids[int(np.argmax(agg_full))]
                        is_full_correct = (pred_full == ideal)
                        if is_full_correct: full_corrects += 1
                        full_total += 1

                        # 4. Flip & Swap Logic
                        sorted_idx = np.argsort(base_probs)[::-1]
                        t1, t2 = sorted_idx[0], sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
                        perm_swap = list(identity_perm)
                        perm_swap[t1], perm_swap[t2] = perm_swap[t2], perm_swap[t1]
                        swap_idx = perm_index_map.get(tuple(perm_swap), identity_idx)
                        
                        probs_swap = probs_seq_np[swap_idx]
                        agg_swap = _aggregate_probs_over_permutations([probs_swap.tolist()], [perm_list[swap_idx]], k)
                        pred_swap = option_ids[int(np.argmax(agg_swap))]
                        
                        is_flipped = (pred_base != pred_swap)
                        
                        # Swap Gap
                        vals_swap = np.sort(probs_swap)[::-1]
                        gap_swap = vals_swap[0] - vals_swap[1] if len(vals_swap) > 1 else vals_swap[0]
                        
                        # Avg Gap
                        avg_gap_val = (gap_val + gap_swap) / 2.0

                        # Store Record
                        full_data_records.append({
                            'ideal': ideal,
                            'is_base_correct': is_base_correct,
                            'is_cyclic_correct': is_cyclic_correct,
                            'is_full_correct': is_full_correct,
                            'gap': gap_val,
                            'top1': top1_val,
                            'is_flipped': is_flipped,
                            'avg_gap': avg_gap_val
                        })

                        if is_base_correct: base_correct_count += 1
                        if is_cyclic_correct: cyclic_correct_count += 1
                        
                        base_results_to_save.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': base_probs.tolist(),
                                'sampled': pred_base,
                                'ideal': ideal,
                                'correct': is_base_correct,
                            }
                        })
                        cyclic_results_to_save.append({
                            'type': 'result',
                            'data': {
                                'idx': data['idx'],
                                'prompt': data.get('prompt'),
                                'options': data['options'],
                                'probs': cyc_probs_list,
                                'ideal': ideal,
                            }
                        })

                    # -----------------------------------------------------------------
                    # Save Derived Results (Base / Cyclic)
                    # -----------------------------------------------------------------
                    N = len(full_data_records)
                    
                    cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                    if getattr(args, 'option_id_set', None): cyclic_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(cyclic_save_path, exist_ok=True)
                    cyc_acc = cyclic_correct_count / N if N > 0 else 0.0
                    save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results_to_save, 
                                 metrics={'type': 'metric', 'data': {'accuracy': cyc_acc}})
                    
                    base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                    if getattr(args, 'option_id_set', None): base_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(base_save_path, exist_ok=True)
                    base_acc = base_correct_count / N if N > 0 else 0.0
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results_to_save, 
                                 metrics={'type': 'metric', 'data': {'accuracy': base_acc}})

                    full_acc = full_corrects / full_total if full_total > 0 else 0.0
                    logger.info(_purple(f"[{subject}] Base: {base_acc:.4f}, Cyclic: {cyc_acc:.4f}, Full: {full_acc:.4f}"))

                    # =================================================================
                    # Multi-Strategy Beta Curve Analysis
                    # =================================================================
                    logger.info(_blue(f"Starting Multi-Strategy Analysis for {subject}..."))
                    
                    betas = [i / 10.0 for i in range(11)]
                    
                    # Convert list of dicts to easy-access arrays
                    arr_gap = np.array([r['gap'] for r in full_data_records])
                    arr_top1 = np.array([r['top1'] for r in full_data_records])
                    arr_avggap = np.array([r['avg_gap'] for r in full_data_records])
                    arr_flip = np.array([r['is_flipped'] for r in full_data_records], dtype=bool)
                    
                    arr_base_corr = np.array([r['is_base_correct'] for r in full_data_records], dtype=float)
                    arr_cyc_corr = np.array([r['is_cyclic_correct'] for r in full_data_records], dtype=float)
                    arr_full_corr = np.array([r['is_full_correct'] for r in full_data_records], dtype=float)

                    # --- 1. Existing Baselines ---
                    results_random_cyclic = []   # (a)
                    results_random_full = []     # (b)
                    results_switch_full = []     # (c)
                    results_switch_cyclic = []   # (d) == Pure Gap
                    
                    # --- 2. New Experiments ---
                    results_top1_cyclic = []     # top1 -> cyclic
                    results_top1_flip_cyclic = []# top1+flip -> cyclic
                    results_avg_gap_cyclic = []  # pure avggap sort -> cyclic
                    results_gap_guided_avg_gap = [] # [NEW] Gap -> AvgGap -> Cyclic

                    # Costs
                    C_cyc = float(k)
                    C_full = float(math.factorial(k))

                    for beta in betas:
                        n_budget = int(N * beta + 1e-9)
                        
                        # (A) Random Baselines
                        cost_rnd_cyc = beta * C_cyc + (1.0 - beta) * 1.0
                        acc_rnd_cyc = beta * cyc_acc + (1.0 - beta) * base_acc
                        results_random_cyclic.append({'beta': beta, 'cost': cost_rnd_cyc, 'acc': acc_rnd_cyc})
                        
                        cost_rnd_full = beta * C_full + (1.0 - beta) * 1.0
                        acc_rnd_full = beta * full_acc + (1.0 - beta) * base_acc
                        results_random_full.append({'beta': beta, 'cost': cost_rnd_full, 'acc': acc_rnd_full})

                        # (B) Thresholding Logic
                        # Gap Threshold
                        if n_budget == 0: thr_gap = -1.0
                        elif n_budget == N: thr_gap = 1.1
                        else: thr_gap = np.sort(arr_gap)[n_budget-1]
                        mask_gap = (arr_gap <= thr_gap)

                        # Top1 Threshold
                        if n_budget == 0: thr_top1 = -1.0
                        elif n_budget == N: thr_top1 = 1.1
                        else: thr_top1 = np.sort(arr_top1)[n_budget-1]
                        mask_top1 = (arr_top1 <= thr_top1)

                        # AvgGap Threshold
                        if n_budget == 0: thr_avg = -1.0
                        elif n_budget == N: thr_avg = 1.1
                        else: thr_avg = np.sort(arr_avggap)[n_budget-1]
                        mask_avg = (arr_avggap <= thr_avg)

                        # (C) Gap-Based Switching (Existing)
                        # Switch -> Cyclic (Pure Gap)
                        cost_sw_cyc = np.sum(np.where(mask_gap, C_cyc, 1.0)) / N
                        acc_sw_cyc = (np.sum(arr_cyc_corr[mask_gap]) + np.sum(arr_base_corr[~mask_gap])) / N
                        results_switch_cyclic.append({'beta': beta, 'cost': cost_sw_cyc, 'acc': acc_sw_cyc, 'thresh': float(thr_gap)})

                        # Switch -> Full
                        cost_sw_full = np.sum(np.where(mask_gap, C_full, 1.0)) / N
                        acc_sw_full = (np.sum(arr_full_corr[mask_gap]) + np.sum(arr_base_corr[~mask_gap])) / N
                        results_switch_full.append({'beta': beta, 'cost': cost_sw_full, 'acc': acc_sw_full, 'thresh': float(thr_gap)})

                        # (D) New Strategy 1: Top1 -> Cyclic
                        cost_top1 = np.sum(np.where(mask_top1, C_cyc, 1.0)) / N
                        acc_top1 = (np.sum(arr_cyc_corr[mask_top1]) + np.sum(arr_base_corr[~mask_top1])) / N
                        results_top1_cyclic.append({'beta': beta, 'cost': cost_top1, 'acc': acc_top1, 'thresh': float(thr_top1)})

                        # (E) New Strategy 2: Top1 + Flip -> Cyclic
                        mask_low_conf = mask_top1
                        mask_flip_triggered = (~mask_low_conf) & (arr_flip) # High Conf but Flip
                        mask_high_conf_no_flip = (~mask_low_conf) & (~arr_flip)
                        
                        acc_hybrid = (np.sum(arr_cyc_corr[mask_low_conf]) + 
                                      np.sum(arr_cyc_corr[mask_flip_triggered]) + 
                                      np.sum(arr_base_corr[mask_high_conf_no_flip])) / N
                        
                        c_low = np.sum(mask_low_conf) * C_cyc
                        c_flip = np.sum(mask_flip_triggered) * C_cyc 
                        c_safe = np.sum(mask_high_conf_no_flip) * 2.0 
                        cost_hybrid = (c_low + c_flip + c_safe) / N
                        results_top1_flip_cyclic.append({'beta': beta, 'cost': cost_hybrid, 'acc': acc_hybrid, 'thresh': float(thr_top1)})

                        # (F) New Strategy 3: Avg Gap -> Cyclic (Pure Sort)
                        cost_avg = np.sum(np.where(mask_avg, C_cyc, 2.0)) / N
                        acc_avg = (np.sum(arr_cyc_corr[mask_avg]) + np.sum(arr_base_corr[~mask_avg])) / N
                        results_avg_gap_cyclic.append({'beta': beta, 'cost': cost_avg, 'acc': acc_avg, 'thresh': float(thr_avg)})

                        # (G) [NEW!] Gap Guided AvgGap -> Cyclic (User Snippet Logic)
                        # Logic:
                        # 1. Check Gap. If Gap > thresh: Stop (Cost 1.0)
                        # 2. If Gap <= thresh: Check AvgGap.
                        #    - If AvgGap <= thresh: Cyclic (Cost k)
                        #    - Else: Stop (Cost 2.0)
                        # Note: Uses same threshold 'thr_gap' for consistency with snippet
                        
                        mask_gap_low = mask_gap # Gap <= thresh
                        mask_avggap_low = (arr_avggap <= thr_gap) # AvgGap <= thresh
                        
                        mask_trigger_cyclic = mask_gap_low & mask_avggap_low
                        mask_check_but_no_cyclic = mask_gap_low & (~mask_avggap_low)
                        mask_no_check = ~mask_gap_low
                        
                        acc_guided = (np.sum(arr_cyc_corr[mask_trigger_cyclic]) + 
                                      np.sum(arr_base_corr[mask_check_but_no_cyclic]) + 
                                      np.sum(arr_base_corr[mask_no_check])) / N
                        
                        c_trigger = np.sum(mask_trigger_cyclic) * C_cyc
                        c_checked = np.sum(mask_check_but_no_cyclic) * 2.0
                        c_base = np.sum(mask_no_check) * 1.0
                        cost_guided = (c_trigger + c_checked + c_base) / N
                        
                        results_gap_guided_avg_gap.append({'beta': beta, 'cost': cost_guided, 'acc': acc_guided, 'thresh': float(thr_gap)})
                        
                        logger.info(f"  [Beta {beta:.1f}] SwCyc({acc_sw_cyc:.4f}) | SwFull({acc_sw_full:.4f}) | Top1({acc_top1:.4f}) | T1Flip({acc_hybrid:.4f}) | AvgGap({acc_avg:.4f}) | Guided({acc_guided:.4f})")

                    # Save All Curves
                    final_obj = {
                        'subject': subject,
                        'k': k,
                        'betas': betas,
                        'curves': {
                            # Existing Baselines
                            'random_cyclic': results_random_cyclic,
                            'random_full': results_random_full,
                            'switch_cyclic': results_switch_cyclic,
                            'switch_full': results_switch_full,
                            # New Experiments
                            'top1_cyclic': results_top1_cyclic,
                            'top1_flip_cyclic': results_top1_flip_cyclic,
                            'avg_gap_cyclic': results_avg_gap_cyclic,
                            'gap_guided_avg_gap': results_gap_guided_avg_gap
                        }
                    }
                    
                    curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                    if getattr(args, 'option_id_set', None): curve_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(curve_save_path, exist_ok=True)
                    save_results(f'{curve_save_path}/{subject}_multi_strategy.jsonl', [final_obj], metrics=None)
                    logger.info(_orange(f"Saved Multi-Strategy curves for: {subject}"))

                except Exception as e:
                    logger.warning(f"Failed to derive analysis for subject '{subject}': {e}")
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