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
                    "ours_low_conf_percent": getattr(args, "ours_low_conf_percent", None),
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
            # Derived policies: only when args.setting == 'full'
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

                    cyc_perms = [tuple((i + s) % k for i in range(k)) for s in range(k)]  # reuse

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
                    # Confidence stats & triggers (precompute)
                    # =========================================================
                    default_conf = []               # base gap: top1 - top2
                    mean_gap_list = []              # gap(mean(base, swap))
                    flip_trigger_mask_global = []   # base argmax != swap argmax

                    for i, bp in enumerate(base_probs_list):
                        vals = np.sort(bp)[::-1]
                        if vals.shape[0] < 2:
                            top1, top2 = (vals[0], 0.0) if vals.shape[0] > 0 else (0.0, 0.0)
                        else:
                            top1, top2 = vals[0], vals[1]
                        default_conf.append(float(top1 - top2))

                        # swap (top1 <-> top2) permutation index
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
                        flip_trigger_mask_global.append(pred_base_content != pred_swap_content)

                    default_conf = np.asarray(default_conf, dtype=np.float64)
                    mean_conf = np.asarray(mean_gap_list, dtype=np.float64)
                    arr_flip_trigger_global = np.asarray(flip_trigger_mask_global, dtype=bool)

                    arr_base_correct = np.asarray(base_correct_list, dtype=bool)
                    arr_cyclic_correct = np.asarray(cyclic_correct_list, dtype=bool)

                    # =========================================================
                    # Save cyclic/base derived results
                    # =========================================================
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

                    base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                    if getattr(args, 'option_id_set', None):
                        base_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(base_save_path, exist_ok=True)
                    base_metrics = {'type': 'metric', 'data': {'accuracy': get_accuracy(base_results)}}
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results, base_metrics)
                    logger.info(_orange(f"Derived and saved base results: {subject}"))

                    # Full accuracy summary
                    if full_total > 0:
                        full_acc = full_corrects / full_total
                        logger.info(_purple(f"[{subject}] Full permutation ensemble accuracy: {full_acc:.4f}"))
                    else:
                        full_acc = float('nan')

                    summary_full = full_acc
                    summary_cyc = cyclic_acc if cyclic_total > 0 else float('nan')
                    summary_base = base_metrics['data']['accuracy'] if (base_metrics is not None and 'accuracy' in base_metrics['data']) else float('nan')
                    logger.info(_purple(f"[{subject}] Accuracies — Full: {summary_full:.4f}, Cyclic: {summary_cyc:.4f}, Default: {summary_base:.4f}"))

                    # =========================================================
                    # Beta curves
                    # =========================================================
                    if len(base_correct_list) == len(cyclic_correct_list) == len(full_correct_list) and len(base_correct_list) > 0:
                        N = len(base_correct_list)
                        betas = [i / 10.0 for i in range(11)]
                        C_cyc = float(k)
                        C_full = float(math.factorial(k))

                        perc = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0

                        # -------------------------
                        # 1) Standard cyclic/full mix
                        # -------------------------
                        curve_cyc = []
                        curve_full = []
                        for beta in betas:
                            n = int(N * beta + 1e-9)

                            if n > 0:
                                acc_cyc_mix = (sum(cyclic_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                                acc_full_mix = (sum(full_correct_list[:n]) + sum(base_correct_list[n:])) / float(N)
                            else:
                                acc_cyc_mix = sum(base_correct_list) / float(N)
                                acc_full_mix = sum(base_correct_list) / float(N)

                            cost_cyc = beta * C_cyc + (1.0 - beta) * 1.0
                            cost_full_mix = beta * C_full + (1.0 - beta) * 1.0
                            curve_cyc.append((cost_cyc, acc_cyc_mix))
                            curve_full.append((cost_full_mix, acc_full_mix))

                        # -------------------------
                        # 2) switch-full / switch-cyclic
                        # -------------------------
                        curve_switch_full = []
                        curve_switch_cyc = []
                        try:
                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))

                                total_cost_sf = 0.0
                                corrects_sf = 0
                                total_cost_sc = 0.0
                                corrects_sc = 0

                                for i in range(0, n):
                                    total_cost_sf += 1.0
                                    total_cost_sc += 1.0
                                    if base_correct_list[i]:
                                        corrects_sf += 1
                                        corrects_sc += 1

                                for i in range(n, N):
                                    is_ambiguous = (float(default_conf[i]) < thresh)

                                    if is_ambiguous:
                                        total_cost_sf += float(len(perm_list))
                                        if full_correct_list[i]:
                                            corrects_sf += 1
                                    else:
                                        total_cost_sf += 1.0
                                        if base_correct_list[i]:
                                            corrects_sf += 1

                                    if is_ambiguous:
                                        total_cost_sc += float(k)
                                        if cyclic_correct_list[i]:
                                            corrects_sc += 1
                                    else:
                                        total_cost_sc += 1.0
                                        if base_correct_list[i]:
                                            corrects_sc += 1

                                curve_switch_full.append((total_cost_sf / float(N), corrects_sf / float(N)))
                                curve_switch_cyc.append((total_cost_sc / float(N), corrects_sc / float(N)))

                            logger.info(_purple(
                                f"[{subject}] Beta curve (Switch->cyclic): " +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_switch_cyc])
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to compute switch curves: {e}")
                            curve_switch_full = []
                            curve_switch_cyc = []

                        # -------------------------
                        # 3) top2flip -> cyclic
                        # -------------------------
                        curve_top2flip_cyc = []
                        try:
                            base_plus_flip_cost = 2.0
                            extra_cyclic_cost = float(k - 1)
                            cyc_after_flip_cost = base_plus_flip_cost + extra_cyclic_cost  # k+1

                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))

                                total_cost = 0.0
                                corrects = 0

                                for i in range(0, n):
                                    total_cost += 1.0
                                    if base_correct_list[i]:
                                        corrects += 1

                                for i in range(n, N):
                                    if float(default_conf[i]) >= thresh:
                                        total_cost += 1.0
                                        if base_correct_list[i]:
                                            corrects += 1
                                        continue

                                    if bool(arr_flip_trigger_global[i]):
                                        total_cost += cyc_after_flip_cost
                                        if cyclic_correct_list[i]:
                                            corrects += 1
                                    else:
                                        total_cost += base_plus_flip_cost
                                        if base_correct_list[i]:
                                            corrects += 1

                                curve_top2flip_cyc.append((total_cost / float(N), corrects / float(N)))

                            logger.info(_purple(
                                f"[{subject}] Beta curve (Ours top2flip->cyclic): " +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_top2flip_cyc])
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to compute top2flip->cyclic curve: {e}")
                            curve_top2flip_cyc = []

                        # -------------------------
                        # 4) AvgGap(static) -> cyclic
                        # -------------------------
                        curve_avggap_static = []
                        try:
                            base_plus_flip_cost = 2.0
                            extra_cyclic_cost = float(k - 1)
                            cyc_after_flip_cost = base_plus_flip_cost + extra_cyclic_cost  # k+1

                            for beta in betas:
                                n = int(N * beta + 1e-9)
                                thresh = float(np.quantile(default_conf[:n], perc)) if n > 0 else float(np.quantile(default_conf, perc))

                                total_cost = 0.0
                                corrects = 0

                                for i in range(0, n):
                                    total_cost += 1.0
                                    if base_correct_list[i]:
                                        corrects += 1

                                for i in range(n, N):
                                    if float(default_conf[i]) >= thresh:
                                        total_cost += 1.0
                                        if base_correct_list[i]:
                                            corrects += 1
                                        continue

                                    total_cost += base_plus_flip_cost
                                    if float(mean_conf[i]) < thresh:
                                        total_cost += extra_cyclic_cost
                                        if cyclic_correct_list[i]:
                                            corrects += 1
                                    else:
                                        if base_correct_list[i]:
                                            corrects += 1

                                curve_avggap_static.append((total_cost / float(N), corrects / float(N)))

                            logger.info(_purple(
                                f"[{subject}] Beta curve (Ours AvgGap(static)->cyclic): " +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_avggap_static])
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to compute AvgGap(static)->cyclic curve: {e}")
                            curve_avggap_static = []

                        # -------------------------
                        # 5) AvgGap(dynamic, ONLINE, th1+MAD) -> cyclic
                        #   - NO quantile/percentile thresholding
                        #   - th1: online-updated (oracle-based)
                        #   - th2: th1 + gamma * mad
                        #   - mad: EMA of |gap - mean_gap|
                        # -------------------------
                        curve_avggap_dynamic = []
                        try:
                            base_plus_flip_cost = 2.0
                            extra_cyclic_cost = float(k - 1)

                            # knobs (추천 시작값)
                            mad_alpha = 0.10   # EMA speed
                            mad_gamma = 1.00   # th2 = th1 + gamma*mad  (너 요청: th1+MAD -> gamma=1)
                            th_lr_up = 0.05    # cyclic이 base를 살렸을 때 th1 올리는 속도
                            th_lr_dn = 0.05    # cyclic이 base를 망쳤을 때 th1 내리는 속도
                            th_lr_decay = 0.005

                            def _clamp01(x: float) -> float:
                                return max(0.0, min(1.0, float(x)))

                            # percentile/quantile 없이 초기값 잡기:
                            # - perc 자체를 th1로 쓰면 scale mismatch가 날 수 있음(0.3이 너무 큼 등)
                            # - 그래서 "작은 값"으로 시작해서 online으로 키우는 걸 추천
                            init_th1 = _clamp01(getattr(args, "ours_th1_init", 0.05))  # CLI로 조절 가능하게
                            th1_min = _clamp01(getattr(args, "ours_th1_min", 0.00))
                            th1_max = _clamp01(getattr(args, "ours_th1_max", 1.00))

                            for beta in betas:
                                n = int(N * beta + 1e-9)

                                th1 = init_th1
                                mad = th1 / 2.0  # 초기 MAD

                                total_cost = 0.0
                                corrects = 0

                                # prefix: base only (기존 beta curve 정의 유지)
                                for i in range(0, n):
                                    total_cost += 1.0
                                    if base_correct_list[i]:
                                        corrects += 1

                                # online region
                                for i in range(n, N):
                                    gap = float(default_conf[i])

                                    # gate1: confident => base only
                                    if gap >= th1:
                                        total_cost += 1.0
                                        if base_correct_list[i]:
                                            corrects += 1
                                        # (optional) 너무 보수적으로 굳는 걸 막는 약한 decay
                                        th1 = _clamp01(th1 - th_lr_decay * th1)
                                        th1 = min(max(th1, th1_min), th1_max)
                                        continue

                                    # low-conf: pay base+flip to compute mean_conf
                                    total_cost += base_plus_flip_cost

                                    # update MAD
                                    diff_abs = abs(gap - float(mean_conf[i]))
                                    mad = (1.0 - mad_alpha) * mad + mad_alpha * diff_abs
                                    mad = max(0.0, mad)

                                    # gate2 threshold: th2 = th1 + MAD
                                    th2 = _clamp01(th1 + mad_gamma * mad)

                                    # decision
                                    go_cyclic = (float(mean_conf[i]) < th2)

                                    if go_cyclic:
                                        total_cost += extra_cyclic_cost
                                        if cyclic_correct_list[i]:
                                            corrects += 1

                                        # ===== oracle online update of th1 =====
                                        # base wrong & cyclic right => 더 많이 cyclic 보내고 싶다 => th1 ↑
                                        if (not base_correct_list[i]) and cyclic_correct_list[i]:
                                            # gap이 th1보다 얼마나 작은지 기반으로 step
                                            th1 = _clamp01(th1 + th_lr_up * (th1 - gap + 1e-6))

                                        # base right & cyclic wrong => cyclic 줄이고 싶다 => th1 ↓
                                        elif base_correct_list[i] and (not cyclic_correct_list[i]):
                                            th1 = _clamp01(th1 - th_lr_dn * max(th1, 1e-6))

                                        # 둘 다 같으면(둘 다 맞/틀) => 약한 decay
                                        else:
                                            th1 = _clamp01(th1 - th_lr_decay * max(th1, 1e-6))

                                    else:
                                        # stop after flip, return base pred
                                        if base_correct_list[i]:
                                            corrects += 1

                                        # cyclic 안 갔는데 base가 틀렸으면 -> th1을 조금 올려서 다음에 더 잡게(선택)
                                        if not base_correct_list[i]:
                                            th1 = _clamp01(th1 + 0.25 * th_lr_up * (th1 - gap + 1e-6))
                                        else:
                                            th1 = _clamp01(th1 - th_lr_decay * th1)

                                    th1 = min(max(th1, th1_min), th1_max)

                                curve_avggap_dynamic.append((total_cost / float(N), corrects / float(N)))

                            logger.info(_purple(
                                f"[{subject}] Beta curve (Ours AvgGap(ONLINE,th1+MAD)->cyclic): " +
                                ", ".join([f"(cost={c:.2f}, acc={a:.4f})" for c, a in curve_avggap_dynamic])
                            ))

                        except Exception as e:
                            logger.warning(f"Failed to compute AvgGap(dynamic online th1+MAD)->cyclic curve: {e}")
                            curve_avggap_dynamic = []

                        # Save curves
                        curve_obj = {
                            'subject': subject,
                            'k': k,
                            'betas': betas,
                            'default_accuracy': float(summary_base),

                            'cyclic': {
                                'costs': [float(c) for c, _ in curve_cyc],
                                'accuracies': [float(a) for _, a in curve_cyc]
                            },
                            'full': {
                                'costs': [float(c) for c, _ in curve_full],
                                'accuracies': [float(a) for _, a in curve_full]
                            },

                            'switch_full': {
                                'costs': [float(c) for c, _ in curve_switch_full],
                                'accuracies': [float(a) for _, a in curve_switch_full]
                            },
                            'switch_cyclic': {
                                'costs': [float(c) for c, _ in curve_switch_cyc],
                                'accuracies': [float(a) for _, a in curve_switch_cyc]
                            },

                            'ours_top2flip': {
                                'costs': [float(c) for c, _ in curve_top2flip_cyc],
                                'accuracies': [float(a) for _, a in curve_top2flip_cyc]
                            },
                            'ours_avggap_static': {
                                'costs': [float(c) for c, _ in curve_avggap_static],
                                'accuracies': [float(a) for _, a in curve_avggap_static]
                            },
                            'ours_avggap_dynamic': {
                                'costs': [float(c) for c, _ in curve_avggap_dynamic],
                                'accuracies': [float(a) for _, a in curve_avggap_dynamic]
                            },

                            # backward-compat alias
                            'ours_avggap': {
                                'costs': [float(c) for c, _ in curve_avggap_static],
                                'accuracies': [float(a) for _, a in curve_avggap_static]
                            },

                            'ours_low_conf_percent': float(getattr(args, 'ours_low_conf_percent', 10.0)),
                            'ours_low_conf_frac': float(perc),
                            'dynamic_init_th1': float(init_th1_fixed),
                            'dynamic_perc2': float(min(1.0, max(0.0, perc * perc))),
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
