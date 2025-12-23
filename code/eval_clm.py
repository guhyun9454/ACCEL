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
    """
    probs_seq: list of length = len(permuted_indices)
              each element is length-k probs in *presented option order*.
    permuted_indices: list of permutations p, where p[j] = original index of j-th presented option.
    Output: aggregated probs in original-option index space.
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

            # =========================================================
            # Full mode: derive base/cyclic/full + multi-strategy curves
            # =========================================================
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

                    # correctness lists
                    base_correct_list = []
                    cyclic_correct_list = []
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

                        full_correct_list.append(corr_full)
                        if corr_full:
                            full_corrects += 1
                        full_total += 1

                    # =========================================================
                    # Precompute confidence + flip/avg
                    #
                    # - default_conf[i] = base gap = top1(base_probs)-top2(base_probs)
                    # - base_conf_max[i] = base top1 confidence
                    # - mean_probs = (agg_base + agg_swap)/2  (content-space aligned)
                    # - mean_conf[i] = gap(mean_probs)        (THIS is your "AvgGap")
                    # - mean_correct_list[i] = correctness if we predict with argmax(mean_probs)
                    # =========================================================
                    default_conf = []               # base gap
                    base_conf_max = []              # base top1 confidence
                    mean_gap_list = []              # gap of mean_probs (base+top2flip averaged)
                    flip_trigger_mask_global = []   # pred(base) != pred(swap)
                    mean_correct_list = []          # pred(mean_probs) correctness

                    for i, bp in enumerate(base_probs_list):
                        # Base Stats
                        vals = np.sort(bp)[::-1]
                        if vals.shape[0] < 2:
                            top1, top2 = (vals[0], 0.0) if vals.shape[0] > 0 else (0.0, 0.0)
                        else:
                            top1, top2 = vals[0], vals[1]
                        base_conf_max.append(top1)
                        default_conf.append(top1 - top2)

                        # Build swap permutation based on top1/top2 indices in BASE presented order
                        sorted_idx = np.argsort(bp)[::-1]
                        top1_idx = int(sorted_idx[0])
                        top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx

                        perm_swap = list(identity_perm)
                        perm_swap[top1_idx], perm_swap[top2_idx] = perm_swap[top2_idx], perm_swap[top1_idx]
                        swap_idx = perm_index_map.get(tuple(perm_swap), identity_idx)

                        probs_base_raw = per_sample_probs[i][identity_idx]
                        probs_swap_raw = per_sample_probs[i][swap_idx]

                        # content-space aligned probabilities for base and swap
                        agg_base = _aggregate_probs_over_permutations(
                            [probs_base_raw.tolist()],
                            [perm_list[identity_idx]],
                            k
                        )
                        agg_swap = _aggregate_probs_over_permutations(
                            [probs_swap_raw.tolist()],
                            [perm_list[swap_idx]],
                            k
                        )

                        # mean_probs: avg probs /2 (네가 말한 그대로)
                        mean_probs = (agg_base + agg_swap) / 2.0

                        # mean_gap: gap computed FROM mean_probs
                        vals_mean = np.sort(mean_probs)[::-1]
                        mean_gap = vals_mean[0] - vals_mean[1] if len(vals_mean) > 1 else 0.0
                        mean_gap_list.append(mean_gap)

                        pred_base_content = option_ids[int(np.argmax(agg_base))]
                        pred_swap_content = option_ids[int(np.argmax(agg_swap))]
                        flip_trigger_mask_global.append(pred_base_content != pred_swap_content)

                        pred_mean = option_ids[int(np.argmax(mean_probs))]
                        mean_correct_list.append(pred_mean == ideals[i])

                    # Convert to numpy
                    default_conf = np.asarray(default_conf, dtype=np.float64)
                    base_conf_max = np.asarray(base_conf_max, dtype=np.float64)   # top1 confidence
                    mean_conf = np.asarray(mean_gap_list, dtype=np.float64)       # AvgGap
                    arr_flip_trigger_global = np.asarray(flip_trigger_mask_global, dtype=bool)
                    arr_mean_correct = np.asarray(mean_correct_list, dtype=bool)

                    arr_base_correct = np.asarray(base_correct_list, dtype=bool)
                    arr_cyclic_correct = np.asarray(cyclic_correct_list, dtype=bool)

                    # Gain target: cases where cyclic fixes base mistakes
                    target_mask = (~arr_base_correct) & (arr_cyclic_correct)
                    total_positives = int(target_mask.sum())

                    # Save cyclic-derived results
                    cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                    if getattr(args, 'option_id_set', None):
                        cyclic_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(cyclic_save_path, exist_ok=True)

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

                    base_acc = get_accuracy(base_results)
                    base_metrics = {'type': 'metric', 'data': {'accuracy': base_acc}}
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results, base_metrics)
                    logger.info(_orange(f"Derived and saved base results: {subject}"))

                    # Full ensemble accuracy
                    full_acc = (full_corrects / full_total) if full_total > 0 else float('nan')
                    logger.info(_purple(f"[{subject}] Accuracies — Full: {full_acc:.4f}, Cyclic: {cyclic_acc:.4f}, Default: {base_acc:.4f}"))

                    # ---------------- Multi-strategy curves ------------
                    if len(base_correct_list) == len(cyclic_correct_list) == len(full_correct_list) and len(base_correct_list) > 0:
                        N = len(base_correct_list)
                        betas = [i / 10.0 for i in range(11)]

                        C_cyc = float(k)
                        C_full = float(math.factorial(k))

                        # Helper arrays as float for sum()
                        base_corr_f = arr_base_correct.astype(np.float64)
                        cyc_corr_f = arr_cyclic_correct.astype(np.float64)
                        full_corr_f = np.asarray(full_correct_list, dtype=np.float64)
                        mean_corr_f = arr_mean_correct.astype(np.float64)
                        top1_conf = base_conf_max.astype(np.float64)

                        # Threshold percentile
                        perc = max(min(getattr(args, 'ours_low_conf_percent', 30.0), 100.0), 0.0) / 100.0

                        # ------------------------------------------------------------------
                        # (1) Random-mix baselines: Cyclic / Full
                        # ------------------------------------------------------------------
                        curve_cyc = []
                        curve_full = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)

                            # Cyclic mix
                            if n > 0:
                                acc_cyc_mix = (cyc_corr_f[:n].sum() + base_corr_f[n:].sum()) / float(N)
                            else:
                                acc_cyc_mix = base_corr_f.sum() / float(N)
                            cost_cyc = beta * C_cyc + (1.0 - beta) * 1.0
                            curve_cyc.append((float(cost_cyc), float(acc_cyc_mix)))

                            # Full mix
                            if n > 0:
                                acc_full_mix = (full_corr_f[:n].sum() + base_corr_f[n:].sum()) / float(N)
                            else:
                                acc_full_mix = base_corr_f.sum() / float(N)
                            cost_full_mix = beta * C_full + (1.0 - beta) * 1.0
                            curve_full.append((float(cost_full_mix), float(acc_full_mix)))

                        # ------------------------------------------------------------------
                        # (2) Ours: Cascading Ensemble (kept as-is)
                        # ------------------------------------------------------------------
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
                                corrects = 0.0

                                # prefix: base only
                                total_cost += float(n) * 1.0
                                corrects += base_corr_f[:n].sum()

                                for i in range(n, N):
                                    probs_seq = per_sample_probs[i]
                                    selected = [order_indices[0]]

                                    agg = _aggregate_probs_over_permutations(
                                        [probs_seq[j].tolist() for j in selected],
                                        [perm_list[j] for j in selected],
                                        k
                                    )
                                    vals = np.sort(agg)[::-1]
                                    cur_gap = vals[0] - vals[1] if len(vals) > 1 else vals[0]
                                    t = 1

                                    while (cur_gap < thresh) and (t < len(order_indices)):
                                        selected.append(order_indices[t])
                                        agg = _aggregate_probs_over_permutations(
                                            [probs_seq[j].tolist() for j in selected],
                                            [perm_list[j] for j in selected],
                                            k
                                        )
                                        vals = np.sort(agg)[::-1]
                                        cur_gap = vals[0] - vals[1] if len(vals) > 1 else vals[0]
                                        t += 1

                                    pred_letter = option_ids[int(np.argmax(agg))]
                                    corrects += 1.0 if pred_letter == ideals[i] else 0.0
                                    total_cost += float(len(selected))

                                curve_ours.append((total_cost / float(N), corrects / float(N)))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours curve: {e}")
                            curve_ours = []

                        # ------------------------------------------------------------------
                        # (3) Switch-Full / Switch-Cyclic (gap threshold)
                        # ------------------------------------------------------------------
                        curve_ours_switch_full = []
                        curve_ours_switch_cyc = []
                        try:
                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                is_amb = (default_conf < thresh)

                                # switch-full
                                cost_sf = (np.where(is_amb, C_full, 1.0).sum()) / float(N)
                                acc_sf = (full_corr_f[is_amb].sum() + base_corr_f[~is_amb].sum()) / float(N)
                                curve_ours_switch_full.append((float(cost_sf), float(acc_sf)))

                                # switch-cyclic
                                cost_sc = (np.where(is_amb, C_cyc, 1.0).sum()) / float(N)
                                acc_sc = (cyc_corr_f[is_amb].sum() + base_corr_f[~is_amb].sum()) / float(N)
                                curve_ours_switch_cyc.append((float(cost_sc), float(acc_sc)))

                        except Exception as e:
                            logger.warning(f"Failed to compute switch curves: {e}")
                            curve_ours_switch_full = []
                            curve_ours_switch_cyc = []

                        # ------------------------------------------------------------------
                        # (4) Switch-Cyclic using TOP1 confidence threshold
                        # ------------------------------------------------------------------
                        curve_switch_cyc_top1 = []
                        try:
                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                if n > 0:
                                    thr_top1 = float(np.quantile(top1_conf[:n], perc))
                                else:
                                    thr_top1 = float(np.quantile(top1_conf, perc))

                                is_amb = (top1_conf < thr_top1)
                                cost = (np.where(is_amb, C_cyc, 1.0).sum()) / float(N)
                                acc = (cyc_corr_f[is_amb].sum() + base_corr_f[~is_amb].sum()) / float(N)
                                curve_switch_cyc_top1.append((float(cost), float(acc)))
                        except Exception as e:
                            logger.warning(f"Failed to compute switch-cyclic-top1 curve: {e}")
                            curve_switch_cyc_top1 = []

                        # ------------------------------------------------------------------
                        # (5) Gap-based Top2Flip + Average prediction (NO cyclic)
                        # ------------------------------------------------------------------
                        curve_top2avg_only = []
                        try:
                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                high = (default_conf >= thresh)
                                low = ~high

                                total_cost = (high.sum() * 1.0 + low.sum() * 2.0) / float(N)
                                corrects = (base_corr_f[high].sum() + mean_corr_f[low].sum()) / float(N)
                                curve_top2avg_only.append((float(total_cost), float(corrects)))
                        except Exception as e:
                            logger.warning(f"Failed to compute Top2Avg-only curve: {e}")
                            curve_top2avg_only = []

                        # ------------------------------------------------------------------
                        # (6) AvgGap만 (no top2flip): Avg(Identity + Rotation1) only
                        # ------------------------------------------------------------------
                        curve_avg2rot_only = []
                        try:
                            rot1_idx = cyclic_indices[1] if len(cyclic_indices) > 1 else cyclic_indices[0]

                            mean2rot_correct = np.zeros(N, dtype=bool)

                            for i in range(N):
                                probs_seq = per_sample_probs[i]

                                agg_id = _aggregate_probs_over_permutations(
                                    [probs_seq[identity_idx].tolist()],
                                    [perm_list[identity_idx]],
                                    k
                                )
                                agg_r1 = _aggregate_probs_over_permutations(
                                    [probs_seq[rot1_idx].tolist()],
                                    [perm_list[rot1_idx]],
                                    k
                                )

                                mean2 = (agg_id + agg_r1) / 2.0
                                pred = option_ids[int(np.argmax(mean2))]
                                mean2rot_correct[i] = (pred == ideals[i])

                            mean2rot_corr_f = mean2rot_correct.astype(np.float64)

                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                high = (default_conf >= thresh)
                                low = ~high

                                total_cost = (high.sum() * 1.0 + low.sum() * 2.0) / float(N)
                                corrects = (base_corr_f[high].sum() + mean2rot_corr_f[low].sum()) / float(N)
                                curve_avg2rot_only.append((float(total_cost), float(corrects)))
                        except Exception as e:
                            logger.warning(f"Failed to compute Avg2Rot-only curve: {e}")
                            curve_avg2rot_only = []

                        # ------------------------------------------------------------------
                        # (7) Ours top2flip -> cyclic (kept same COST MODEL as your code)
                        # ------------------------------------------------------------------
                        curve_ours_top2flip_cyc = []
                        try:
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

                                # logging PR on test region
                                if beta in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0] and N > n:
                                    test_default_conf = default_conf[n:]
                                    test_flip_trigger = arr_flip_trigger_global[n:]
                                    test_target_mask = target_mask[n:]

                                    pred_mask_low_conf = test_default_conf <= thresh
                                    pred_mask_combined = pred_mask_low_conf & test_flip_trigger

                                    pr_p, pr_r, pr_f1 = calc_pr_f1(pred_mask_combined, test_target_mask)
                                    logger.info(
                                        f"  [Beta {beta:.1f}] Sample Size: {n} | "
                                        f"Est. Threshold ({perc*100:.0f}%): {thresh:.4f} | "
                                        f"Real F1: {pr_f1:.4f} | Prec: {pr_p:.4f} | Rec: {pr_r:.4f}"
                                    )

                                total_cost = 0.0
                                corrects = 0.0

                                # prefix
                                total_cost += float(n) * 1.0
                                corrects += base_corr_f[:n].sum()

                                # test region
                                for i in range(n, N):
                                    if default_conf[i] >= thresh:
                                        total_cost += 1.0
                                        corrects += 1.0 if arr_base_correct[i] else 0.0
                                        continue

                                    if arr_flip_trigger_global[i]:
                                        total_cost += float(k)
                                        corrects += 1.0 if arr_cyclic_correct[i] else 0.0
                                    else:
                                        total_cost += 2.0
                                        corrects += 1.0 if arr_base_correct[i] else 0.0

                                curve_ours_top2flip_cyc.append((total_cost / float(N), corrects / float(N)))

                            logger.info(_purple(
                                f"[{subject}] Beta curve (Ours top2flip->cyclic): " +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_top2flip_cyc])
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours top2flip->cyclic curve: {e}")
                            import traceback
                            traceback.print_exc()
                            curve_ours_top2flip_cyc = []

                        # ------------------------------------------------------------------
                        # (8) Ours AvgGap -> Cyclic (YOUR intended semantics)
                        # ------------------------------------------------------------------
                        curve_ours_avg_gap_cyc = []
                        try:
                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                if n > 0:
                                    thresh = float(np.quantile(default_conf[:n], perc))
                                else:
                                    thresh = float(np.quantile(default_conf, perc))

                                total_cost = 0.0
                                corrects = 0.0

                                # prefix
                                total_cost += float(n) * 1.0
                                corrects += base_corr_f[:n].sum()

                                for i in range(n, N):
                                    if default_conf[i] >= thresh:
                                        total_cost += 1.0
                                        corrects += 1.0 if arr_base_correct[i] else 0.0
                                        continue

                                    # low base-gap:
                                    # if mean_gap still low -> cyclic else stop with mean_pred
                                    if mean_conf[i] < thresh:
                                        total_cost += float(k)
                                        corrects += 1.0 if arr_cyclic_correct[i] else 0.0
                                    else:
                                        total_cost += 2.0
                                        corrects += 1.0 if arr_mean_correct[i] else 0.0

                                curve_ours_avg_gap_cyc.append((total_cost / float(N), corrects / float(N)))

                            logger.info(_purple(
                                f"[{subject}] Beta curve (Ours AvgGap->cyclic): " +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_ours_avg_gap_cyc])
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to compute Ours AvgGap->cyclic curve: {e}")
                            curve_ours_avg_gap_cyc = []

                        # ------------------------------------------------------------------
                        # (9) TOP1-threshold: flip once, then either cyclic (if flip changes) or stop at mean_pred
                        # ------------------------------------------------------------------
                        curve_top1flip_then = []
                        try:
                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                if n > 0:
                                    thr_top1 = float(np.quantile(top1_conf[:n], perc))
                                else:
                                    thr_top1 = float(np.quantile(top1_conf, perc))

                                total_cost = 0.0
                                corrects = 0.0

                                # prefix base
                                total_cost += float(n) * 1.0
                                corrects += base_corr_f[:n].sum()

                                for i in range(n, N):
                                    if top1_conf[i] >= thr_top1:
                                        total_cost += 1.0
                                        corrects += 1.0 if arr_base_correct[i] else 0.0
                                        continue

                                    # low top1 => do one flip-run
                                    if arr_flip_trigger_global[i]:
                                        total_cost += float(k)
                                        corrects += 1.0 if arr_cyclic_correct[i] else 0.0
                                    else:
                                        total_cost += 2.0
                                        corrects += 1.0 if arr_mean_correct[i] else 0.0

                                curve_top1flip_then.append((total_cost / float(N), corrects / float(N)))
                        except Exception as e:
                            logger.warning(f"Failed to compute top1flip_then curve: {e}")
                            curve_top1flip_then = []

                        # ------------------------------------------------------------------
                        # Save curves (모든 전략 cost/acc 저장)
                        # ------------------------------------------------------------------
                        curve_obj = {
                            'subject': subject,
                            'k': k,
                            'betas': betas,
                            'default_accuracy': float(base_acc),
                            'full_accuracy': float(full_acc),
                            'cyclic_accuracy': float(cyclic_acc),

                            # (A) random mix
                            'random_cyclic': {'costs': [c for c, _ in curve_cyc], 'accuracies': [a for _, a in curve_cyc]},
                            'random_full': {'costs': [c for c, _ in curve_full], 'accuracies': [a for _, a in curve_full]},

                            # (C) gap-switch
                            'switch_cyclic_gap': {'costs': [c for c, _ in curve_ours_switch_cyc], 'accuracies': [a for _, a in curve_ours_switch_cyc]},
                            'switch_full_gap': {'costs': [c for c, _ in curve_ours_switch_full], 'accuracies': [a for _, a in curve_ours_switch_full]},

                            # (D) top1-switch
                            'switch_cyclic_top1': {'costs': [c for c, _ in curve_switch_cyc_top1], 'accuracies': [a for _, a in curve_switch_cyc_top1]},

                            # (B) ours cascade
                            'ours_cascade': {'costs': [c for c, _ in curve_ours], 'accuracies': [a for _, a in curve_ours]},

                            # (5) gap-based top2flip+avg (no cyclic)
                            'gap_top2avg_only': {'costs': [c for c, _ in curve_top2avg_only], 'accuracies': [a for _, a in curve_top2avg_only]},

                            # (6) AvgGap-only (no top2flip) : avg(identity + rot1)
                            'avggap_only_2rot': {'costs': [c for c, _ in curve_avg2rot_only], 'accuracies': [a for _, a in curve_avg2rot_only]},

                            # (7) ours top2flip -> cyclic
                            'ours_top2flip_to_cyclic': {'costs': [c for c, _ in curve_ours_top2flip_cyc], 'accuracies': [a for _, a in curve_ours_top2flip_cyc]},

                            # (8) ours AvgGap -> cyclic
                            'ours_avggap_to_cyclic': {'costs': [c for c, _ in curve_ours_avg_gap_cyc], 'accuracies': [a for _, a in curve_ours_avg_gap_cyc]},

                            # (9) top1-based flip once then (cyclic if changed else mean)
                            'top1flip_then': {'costs': [c for c, _ in curve_top1flip_then], 'accuracies': [a for _, a in curve_top1flip_then]},
                        }

                        # =========================================================
                        # Gain/Loss & Net Analysis (AvgGap Strategy)
                        # - uses mean_conf (gap of mean_probs) and base-gap threshold
                        # =========================================================
                        try:
                            logger.info(_purple(f"[{subject}] Detailed PR & Gain/Loss Analysis (Oracle Bottom %):"))

                            scan_percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                            low_conf_thresholds = np.percentile(default_conf, scan_percentiles)

                            combined_stats = []

                            for i, p in enumerate(scan_percentiles):
                                thr = float(low_conf_thresholds[i])

                                # Trigger rule (AvgGap Strategy):
                                #   base_gap low AND mean_gap low  => cyclic trigger
                                pred_mask_low = (default_conf <= thr)
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

                                logger.info(
                                    f"  [Bottom {p}%] (Thr={thr:.4f}) Trigger: {trigger_count} | "
                                    f"Gain: +{gain_count} | Loss: -{loss_count} | Net: {net_gain:+d} | "
                                    f"F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}"
                                )

                                combined_stats.append({
                                    'percentile': p,
                                    'threshold': thr,
                                    'trigger_count': trigger_count,
                                    'gain': gain_count,
                                    'loss': loss_count,
                                    'net_gain': net_gain,
                                    'f1': f1,
                                    'recall': recall,
                                    'precision': precision
                                })

                            curve_obj['pr_analysis'] = combined_stats

                            # Gap vs Flip rate (debug)
                            bins = [i / 10.0 for i in range(11)]
                            gap_flip_stats = []
                            logger.info(_purple(f"[{subject}] Gap vs Flip Rate:"))
                            for i in range(len(bins) - 1):
                                low, high = bins[i], bins[i + 1]
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
                        logger.info(_orange(f"Saved multi-strategy curves for: {subject}"))

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
