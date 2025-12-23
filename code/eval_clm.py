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


def _curve_pairs_to_str(curve_pairs):
    if not curve_pairs:
        return "(empty)"
    return ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_pairs])


def _print_all_curves(subject: str, curves: dict):
    """
    curves: {name: [(cost, acc), ...]}  (length == len(betas))
    """
    logger.info(_purple(f"[{subject}] ===== Multi-Strategy Beta Curves (cost/acc) ====="))
    for name, pairs in curves.items():
        logger.info(_purple(f"[{subject}] {name}: {_curve_pairs_to_str(pairs)}"))


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

                    cyclic_results = []
                    base_results = []

                    full_total = 0
                    full_corrects = 0
                    cyclic_total = 0
                    cyclic_corrects = 0

                    per_sample_probs = []
                    base_probs_list = []
                    ideals = []

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

                        # Cyclic (k rotations)
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
                        cyclic_corrects += int(corr_cyc)
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

                        # Full
                        agg_full = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                        pred_full = option_ids[int(np.argmax(agg_full))]
                        corr_full = (pred_full == data['ideal'])

                        full_correct_list.append(corr_full)
                        full_corrects += int(corr_full)
                        full_total += 1

                    # ------------------- precompute base-gap / top1 / top2flip mean-gap -------------------
                    default_conf = []     # base gap = top1-top2 (base presented order)
                    base_conf_max = []    # base top1 confidence
                    mean_gap_list = []    # gap(mean_probs) where mean_probs = avg(content-aligned base & swap)
                    flip_trigger_mask = []  # pred(base_content) != pred(swap_content)
                    mean_correct_list = []  # pred(mean_probs) correctness

                    for i, bp in enumerate(base_probs_list):
                        vals = np.sort(bp)[::-1]
                        top1 = float(vals[0]) if len(vals) > 0 else 0.0
                        top2 = float(vals[1]) if len(vals) > 1 else 0.0
                        base_conf_max.append(top1)
                        default_conf.append(top1 - top2)

                        sorted_idx = np.argsort(bp)[::-1]
                        top1_idx = int(sorted_idx[0])
                        top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx

                        perm_swap = list(identity_perm)
                        perm_swap[top1_idx], perm_swap[top2_idx] = perm_swap[top2_idx], perm_swap[top1_idx]
                        swap_idx = perm_index_map.get(tuple(perm_swap), identity_idx)

                        probs_base_raw = per_sample_probs[i][identity_idx]
                        probs_swap_raw = per_sample_probs[i][swap_idx]

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
                        mean_probs = (agg_base + agg_swap) / 2.0

                        vals_mean = np.sort(mean_probs)[::-1]
                        mean_gap = float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0
                        mean_gap_list.append(mean_gap)

                        pred_base_content = option_ids[int(np.argmax(agg_base))]
                        pred_swap_content = option_ids[int(np.argmax(agg_swap))]
                        flip_trigger_mask.append(pred_base_content != pred_swap_content)

                        pred_mean = option_ids[int(np.argmax(mean_probs))]
                        mean_correct_list.append(pred_mean == ideals[i])

                    default_conf = np.asarray(default_conf, dtype=np.float64)     # base gap
                    top1_conf = np.asarray(base_conf_max, dtype=np.float64)      # top1 confidence
                    mean_conf = np.asarray(mean_gap_list, dtype=np.float64)      # AvgGap
                    flip_trigger = np.asarray(flip_trigger_mask, dtype=bool)     # did base<->swap change argmax?
                    mean_correct = np.asarray(mean_correct_list, dtype=bool)

                    arr_base_correct = np.asarray(base_correct_list, dtype=bool)
                    arr_cyclic_correct = np.asarray(cyclic_correct_list, dtype=bool)
                    arr_full_correct = np.asarray(full_correct_list, dtype=bool)

                    # Save cyclic/base derived (optional but kept)
                    cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                    if getattr(args, 'option_id_set', None):
                        cyclic_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(cyclic_save_path, exist_ok=True)

                    cyclic_acc = (cyclic_corrects / cyclic_total) if cyclic_total > 0 else float('nan')
                    save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results,
                                 metrics={'type': 'metric', 'data': {'accuracy': cyclic_acc}})
                    logger.info(_orange(f"Derived and saved cyclic results: {subject}"))

                    base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                    if getattr(args, 'option_id_set', None):
                        base_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(base_save_path, exist_ok=True)

                    base_acc = get_accuracy(base_results)
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results,
                                 metrics={'type': 'metric', 'data': {'accuracy': float(base_acc)}})
                    logger.info(_orange(f"Derived and saved base results: {subject}"))

                    full_acc = (full_corrects / full_total) if full_total > 0 else float('nan')
                    logger.info(_purple(
                        f"[{subject}] Accuracies — Full: {full_acc:.4f}, Cyclic: {cyclic_acc:.4f}, Default: {base_acc:.4f}"
                    ))

                    # ---------------- Multi-strategy curves (PRINT ALL) ----------------
                    N = len(arr_base_correct)
                    if N > 0 and len(arr_cyclic_correct) == N and len(arr_full_correct) == N:
                        betas = [i / 10.0 for i in range(11)]
                        perc = max(min(getattr(args, 'ours_low_conf_percent', 30.0), 100.0), 0.0) / 100.0

                        C_cyc = float(k)                 # cyclic = k rotations
                        C_full = float(math.factorial(k))  # full = k!

                        base_corr_f = arr_base_correct.astype(np.float64)
                        cyc_corr_f = arr_cyclic_correct.astype(np.float64)
                        full_corr_f = arr_full_correct.astype(np.float64)
                        mean_corr_f = mean_correct.astype(np.float64)

                        # (1) random_cyclic mix
                        curve_random_cyclic = []
                        # (2) random_full mix
                        curve_random_full = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            if n > 0:
                                acc_cyc_mix = (cyc_corr_f[:n].sum() + base_corr_f[n:].sum()) / float(N)
                                acc_full_mix = (full_corr_f[:n].sum() + base_corr_f[n:].sum()) / float(N)
                            else:
                                acc_cyc_mix = base_corr_f.sum() / float(N)
                                acc_full_mix = base_corr_f.sum() / float(N)

                            cost_cyc = beta * C_cyc + (1.0 - beta) * 1.0
                            cost_full = beta * C_full + (1.0 - beta) * 1.0
                            curve_random_cyclic.append((float(cost_cyc), float(acc_cyc_mix)))
                            curve_random_full.append((float(cost_full), float(acc_full_mix)))

                        # (3) switch_cyclic (gap threshold)
                        curve_switch_cyclic_gap = []
                        # (4) switch_full (gap threshold)
                        curve_switch_full_gap = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))
                            is_amb = (default_conf < thresh)

                            cost_sc = (np.where(is_amb, C_cyc, 1.0).sum()) / float(N)
                            acc_sc = (cyc_corr_f[is_amb].sum() + base_corr_f[~is_amb].sum()) / float(N)
                            curve_switch_cyclic_gap.append((float(cost_sc), float(acc_sc)))

                            cost_sf = (np.where(is_amb, C_full, 1.0).sum()) / float(N)
                            acc_sf = (full_corr_f[is_amb].sum() + base_corr_f[~is_amb].sum()) / float(N)
                            curve_switch_full_gap.append((float(cost_sf), float(acc_sf)))

                        # (5) switch_cyclic (top1 confidence threshold)
                        curve_switch_cyclic_top1 = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            thr_top1 = float(np.quantile(top1_conf[:n], perc)) if n > 0 else float(np.quantile(top1_conf, perc))
                            is_amb = (top1_conf < thr_top1)

                            cost = (np.where(is_amb, C_cyc, 1.0).sum()) / float(N)
                            acc = (cyc_corr_f[is_amb].sum() + base_corr_f[~is_amb].sum()) / float(N)
                            curve_switch_cyclic_top1.append((float(cost), float(acc)))

                        # (6) gap_top2avg_only (NO cyclic): base-gap low -> run swap+avg (cost2) and predict mean
                        curve_gap_top2avg_only = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))

                            high = (default_conf >= thresh)
                            low = ~high
                            total_cost = (high.sum() * 1.0 + low.sum() * 2.0) / float(N)
                            acc = (base_corr_f[high].sum() + mean_corr_f[low].sum()) / float(N)
                            curve_gap_top2avg_only.append((float(total_cost), float(acc)))

                        # (7) avggap_only_2rot (no top2flip): base-gap low -> avg(identity + rot1) (cost2)
                        curve_avggap_only_2rot = []
                        try:
                            rot1_idx = cyclic_indices[1] if len(cyclic_indices) > 1 else cyclic_indices[0]
                            mean2rot_correct = np.zeros(N, dtype=bool)
                            for i in range(N):
                                probs_seq = per_sample_probs[i]
                                agg_id = _aggregate_probs_over_permutations([probs_seq[identity_idx].tolist()], [perm_list[identity_idx]], k)
                                agg_r1 = _aggregate_probs_over_permutations([probs_seq[rot1_idx].tolist()], [perm_list[rot1_idx]], k)
                                mean2 = (agg_id + agg_r1) / 2.0
                                pred = option_ids[int(np.argmax(mean2))]
                                mean2rot_correct[i] = (pred == ideals[i])
                            mean2rot_corr_f = mean2rot_correct.astype(np.float64)

                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))
                                high = (default_conf >= thresh)
                                low = ~high

                                total_cost = (high.sum() * 1.0 + low.sum() * 2.0) / float(N)
                                acc = (base_corr_f[high].sum() + mean2rot_corr_f[low].sum()) / float(N)
                                curve_avggap_only_2rot.append((float(total_cost), float(acc)))
                        except Exception as e:
                            logger.warning(f"avggap_only_2rot failed: {e}")
                            curve_avggap_only_2rot = []

                        # (8) ours_cascade (perm-by-perm growing until gap >= thresh)
                        curve_ours_cascade = []
                        try:
                            order_indices = list(range(len(perm_list)))
                            if identity_idx != 0:
                                order_indices = [identity_idx] + [i for i in order_indices if i != identity_idx]

                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))

                                total_cost = float(n) * 1.0
                                corrects = base_corr_f[:n].sum()

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

                                curve_ours_cascade.append((float(total_cost / float(N)), float(corrects / float(N))))
                        except Exception as e:
                            logger.warning(f"ours_cascade failed: {e}")
                            curve_ours_cascade = []

                        # (9) ours_top2flip_to_cyclic: base-gap low -> if flip_trigger then cyclic else base (cost2 if not flipped)
                        curve_ours_top2flip_to_cyclic = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))

                            total_cost = float(n) * 1.0
                            corrects = base_corr_f[:n].sum()

                            for i in range(n, N):
                                if default_conf[i] >= thresh:
                                    total_cost += 1.0
                                    corrects += 1.0 if arr_base_correct[i] else 0.0
                                else:
                                    # we conceptually ran swap too
                                    if flip_trigger[i]:
                                        total_cost += float(k)   # treat as cyclic cost
                                        corrects += 1.0 if arr_cyclic_correct[i] else 0.0
                                    else:
                                        total_cost += 2.0
                                        corrects += 1.0 if arr_base_correct[i] else 0.0

                            curve_ours_top2flip_to_cyclic.append((float(total_cost / float(N)), float(corrects / float(N))))

                        # (10) ours_avggap_to_cyclic: base-gap low -> if AvgGap still low then cyclic else mean_pred
                        curve_ours_avggap_to_cyclic = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))

                            total_cost = float(n) * 1.0
                            corrects = base_corr_f[:n].sum()

                            for i in range(n, N):
                                if default_conf[i] >= thresh:
                                    total_cost += 1.0
                                    corrects += 1.0 if arr_base_correct[i] else 0.0
                                else:
                                    # base_gap low: we have mean_conf from (base+swap)/2
                                    if mean_conf[i] < thresh:
                                        total_cost += float(k)
                                        corrects += 1.0 if arr_cyclic_correct[i] else 0.0
                                    else:
                                        total_cost += 2.0
                                        corrects += 1.0 if mean_correct[i] else 0.0

                            curve_ours_avggap_to_cyclic.append((float(total_cost / float(N)), float(corrects / float(N))))

                        # (11) NEW: flip_then_avggap_to_cyclic (네가 말한 “flip 바뀐 애에 대해 AvgGap도 보자”)
                        # base-gap low -> run swap(=cost2)
                        #   if flip_trigger:
                        #       if AvgGap low -> cyclic
                        #       else -> mean_pred
                        #   else:
                        #       mean_pred (or base_pred) (어차피 동일한 쪽이 대부분이라 mean_pred로 통일)
                        curve_flip_then_avggap_to_cyclic = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))

                            total_cost = float(n) * 1.0
                            corrects = base_corr_f[:n].sum()

                            for i in range(n, N):
                                if default_conf[i] >= thresh:
                                    total_cost += 1.0
                                    corrects += 1.0 if arr_base_correct[i] else 0.0
                                else:
                                    # we ran swap to know flip/mean_conf => baseline cost 2 unless escalated
                                    if flip_trigger[i] and (mean_conf[i] < thresh):
                                        total_cost += float(k)
                                        corrects += 1.0 if arr_cyclic_correct[i] else 0.0
                                    else:
                                        total_cost += 2.0
                                        corrects += 1.0 if mean_correct[i] else 0.0

                            curve_flip_then_avggap_to_cyclic.append((float(total_cost / float(N)), float(corrects / float(N))))

                        # (12) top1flip_then:
                        # top1 low -> do swap(=cost2); if flip_trigger then cyclic else mean_pred
                        curve_top1flip_then = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)
                            thr_top1 = float(np.quantile(top1_conf[:n], perc)) if n > 0 else float(np.quantile(top1_conf, perc))

                            total_cost = float(n) * 1.0
                            corrects = base_corr_f[:n].sum()

                            for i in range(n, N):
                                if top1_conf[i] >= thr_top1:
                                    total_cost += 1.0
                                    corrects += 1.0 if arr_base_correct[i] else 0.0
                                else:
                                    if flip_trigger[i]:
                                        total_cost += float(k)
                                        corrects += 1.0 if arr_cyclic_correct[i] else 0.0
                                    else:
                                        total_cost += 2.0
                                        corrects += 1.0 if mean_correct[i] else 0.0

                            curve_top1flip_then.append((float(total_cost / float(N)), float(corrects / float(N))))

                        # ---- PRINT ALL curves immediately (this was missing before) ----
                        curves_to_print = {
                            "random_cyclic": curve_random_cyclic,
                            "random_full": curve_random_full,
                            "switch_cyclic_gap": curve_switch_cyclic_gap,
                            "switch_full_gap": curve_switch_full_gap,
                            "switch_cyclic_top1": curve_switch_cyclic_top1,
                            "gap_top2avg_only(no_cyclic)": curve_gap_top2avg_only,
                            "avggap_only_2rot(no_top2flip)": curve_avggap_only_2rot,
                            "ours_cascade": curve_ours_cascade,
                            "ours_top2flip_to_cyclic": curve_ours_top2flip_to_cyclic,
                            "ours_avggap_to_cyclic": curve_ours_avggap_to_cyclic,
                            "flip_then_avggap_to_cyclic(NEW)": curve_flip_then_avggap_to_cyclic,
                            "top1flip_then": curve_top1flip_then,
                        }
                        _print_all_curves(subject, curves_to_print)

                        # ---- Save curves ----
                        curve_obj = {
                            'subject': subject,
                            'k': k,
                            'betas': betas,
                            'default_accuracy': float(base_acc),
                            'full_accuracy': float(full_acc),
                            'cyclic_accuracy': float(cyclic_acc),

                            'random_cyclic': {'costs': [c for c, _ in curve_random_cyclic], 'accuracies': [a for _, a in curve_random_cyclic]},
                            'random_full': {'costs': [c for c, _ in curve_random_full], 'accuracies': [a for _, a in curve_random_full]},

                            'switch_cyclic_gap': {'costs': [c for c, _ in curve_switch_cyclic_gap], 'accuracies': [a for _, a in curve_switch_cyclic_gap]},
                            'switch_full_gap': {'costs': [c for c, _ in curve_switch_full_gap], 'accuracies': [a for _, a in curve_switch_full_gap]},
                            'switch_cyclic_top1': {'costs': [c for c, _ in curve_switch_cyclic_top1], 'accuracies': [a for _, a in curve_switch_cyclic_top1]},

                            'gap_top2avg_only': {'costs': [c for c, _ in curve_gap_top2avg_only], 'accuracies': [a for _, a in curve_gap_top2avg_only]},
                            'avggap_only_2rot': {'costs': [c for c, _ in curve_avggap_only_2rot], 'accuracies': [a for _, a in curve_avggap_only_2rot]},

                            'ours_cascade': {'costs': [c for c, _ in curve_ours_cascade], 'accuracies': [a for _, a in curve_ours_cascade]},
                            'ours_top2flip_to_cyclic': {'costs': [c for c, _ in curve_ours_top2flip_to_cyclic], 'accuracies': [a for _, a in curve_ours_top2flip_to_cyclic]},
                            'ours_avggap_to_cyclic': {'costs': [c for c, _ in curve_ours_avggap_to_cyclic], 'accuracies': [a for _, a in curve_ours_avggap_to_cyclic]},
                            'flip_then_avggap_to_cyclic': {'costs': [c for c, _ in curve_flip_then_avggap_to_cyclic], 'accuracies': [a for _, a in curve_flip_then_avggap_to_cyclic]},
                            'top1flip_then': {'costs': [c for c, _ in curve_top1flip_then], 'accuracies': [a for _, a in curve_top1flip_then]},
                        }

                        curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                        if getattr(args, 'option_id_set', None):
                            curve_save_path += f'_id-{args.option_id_set}'
                        os.makedirs(curve_save_path, exist_ok=True)

                        save_results(f'{curve_save_path}/{subject}_multi_strategy.jsonl', [curve_obj], metrics=None)
                        logger.info(_orange(f"Saved multi-strategy curves for: {subject}"))

                except Exception as e:
                    logger.warning(f"Failed to derive multi-strategy curves for subject '{subject}': {e}")
                    import traceback
                    traceback.print_exc()

            logging_cuda_memory_usage()

    try:
        if wandb_run is not None:
            import wandb
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
