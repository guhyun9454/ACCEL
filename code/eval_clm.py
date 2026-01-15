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
import zlib
from functools import partial
from typing import List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

# Matplotlib (for saving beta-curve PNGs in headless envs)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_clm_utils import (
    parse_arguments,
    prepare_eval,
)

from debias_utils import simple as debias_simple

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
    probs_seq: list of length = (#permutations used)
      each element: list/array of length k (letter-space probs)
    permuted_indices: list of permutations p where p[j] is content-index at letter position j.
    Returns:
      agg: length k, content-space aggregated probabilities (mean over permutations)
    """
    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            agg[p[j]] += letter_probs[j]
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


# ---- helper: normalized negative entropy confidence in [0,1] ----
def _nent_conf(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / s
    H = -float(np.sum(p * np.log(p)))
    # higher = more confident
    return float(1.0 - H / max(1e-12, np.log(len(p))))


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


def _stable_u32_seed(s: str, base_seed: int = 0) -> int:
    """Deterministic per-string seed (independent of Python hash randomization)."""
    return (int(zlib.crc32(s.encode("utf-8"))) + int(base_seed)) & 0xFFFFFFFF


def _apply_pride_global_prior_to_probs_seq(probs_seq: np.ndarray,
                                          prior: np.ndarray,
                                          eps: float = 1e-12) -> np.ndarray:
    """
    probs_seq: (num_perms, k) observed letter-space probs for a single sample.
    prior: (k,) global prior over option IDs (letter tokens).
    Returns: corrected probs_seq with row-wise renormalization.
    """
    p = np.asarray(probs_seq, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64).reshape(1, -1)
    adj = p / (pr + eps)
    adj = adj / (adj.sum(axis=1, keepdims=True) + eps)
    return adj


def _estimate_pride_global_prior(per_sample_probs: List[np.ndarray],
                                 cyclic_indices: List[int],
                                 ratio_prefix_samples: float,
                                 prefix_selector: str,
                                 base_seed: int,
                                 subject_key: str,
                                 base_conf: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    Estimate global prior (PriDe) from a small subset of samples using cyclic permutations only.
    Returns (prior_global, meta).
    """
    N = len(per_sample_probs)
    ratio = float(max(0.0, min(1.0, ratio_prefix_samples)))
    num_prefix = max(1, int(round(N * ratio))) if N > 0 else 0
    num_prefix = min(num_prefix, N)

    if num_prefix <= 0:
        prior = np.ones((len(cyclic_indices),), dtype=np.float64)
        prior = prior / prior.sum()
        return prior, {"num_prefix_samples": 0, "selector": prefix_selector}

    if prefix_selector == "low_conf" and base_conf is not None and len(base_conf) == N:
        prefix_ids = np.argsort(np.asarray(base_conf, dtype=np.float64))[:num_prefix].tolist()
        selector_used = "low_conf"
    else:
        seed = _stable_u32_seed(subject_key, base_seed=base_seed)
        rng = random.Random(seed)
        ids = list(range(N))
        rng.shuffle(ids)
        prefix_ids = ids[:num_prefix]
        selector_used = "random"

    priors = []
    for i in prefix_ids:
        # observed matrix using cyclic rotations (k x k)
        observed = np.asarray([per_sample_probs[i][j] for j in cyclic_indices], dtype=np.float64)
        _, _, prior_i = debias_simple(observed)
        priors.append(prior_i)

    prior_global = np.mean(np.asarray(priors, dtype=np.float64), axis=0)
    prior_global = prior_global / (prior_global.sum() + 1e-12)

    meta = {
        "ratio_prefix_samples": float(ratio),
        "num_prefix_samples": int(num_prefix),
        "selector": selector_used,
        "seed_key": str(subject_key),
        "base_seed": int(base_seed),
    }
    return prior_global, meta


def _plot_compare_beta_curve_png(baseline_curve_obj: dict,
                                 pride_curve_obj: dict,
                                 out_path: str,
                                 title: str = "Accuracy vs. Computational Cost") -> None:
    """
    baseline_curve_obj/pride_curve_obj: curve objects saved by eval_clm.py (single subject)
    Saves a single overlay plot (baseline vs pride) to out_path.
    """
    def _extract(obj: dict):
        cyc_costs = obj["cyclic"]["costs"]
        cyc_accs = obj["cyclic"]["accuracies"]
        full_costs = obj["full"]["costs"]
        full_accs = obj["full"]["accuracies"]
        default_acc = float(obj.get("default_accuracy", float("nan")))
        return cyc_costs, cyc_accs, full_costs, full_accs, default_acc

    b_cyc_costs, b_cyc_accs, b_full_costs, b_full_accs, b_def = _extract(baseline_curve_obj)
    p_cyc_costs, p_cyc_accs, p_full_costs, p_full_accs, p_def = _extract(pride_curve_obj)

    plt.figure(figsize=(7.5, 5.0), dpi=160)
    plt.plot(b_cyc_costs, b_cyc_accs, marker='o', label='Baseline Cyclic')
    plt.plot(b_full_costs, b_full_accs, marker='o', label='Baseline Full')
    plt.scatter([1.0], [b_def], marker='*', s=180, c='black', label='Baseline Default')

    plt.plot(p_cyc_costs, p_cyc_accs, marker='o', linestyle='--', label='PRIDE Cyclic')
    plt.plot(p_full_costs, p_full_accs, marker='o', linestyle='--', label='PRIDE Full')
    plt.scatter([1.0], [p_def], marker='*', s=180, c='gray', label='PRIDE Default')

    plt.xlabel("Computational Cost (× of default)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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
                    "ours_th1_ema": getattr(args, "ours_th1_ema", None),
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
                    sample_idxs = []
                    sample_prompts = []
                    sample_options_list = []

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
                        sample_idxs.append(data.get('idx'))
                        sample_prompts.append(data.get('prompt'))
                        sample_options_list.append(data.get('options'))

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

                    # ---------------------------------------------------------
                    # helper: compute+save beta curves (baseline or pride)
                    # ---------------------------------------------------------
                    def _compute_and_save_beta_curve(
                        curve_save_path_: str,
                        tag: str,
                        per_sample_probs_: List[np.ndarray],
                        base_probs_list_: List[np.ndarray],
                        ideals_: List[str],
                        base_correct_list_: List[bool],
                        cyclic_correct_list_: List[bool],
                        full_correct_list_: List[bool],
                        default_conf_: np.ndarray,
                        mean_conf_: np.ndarray,
                        arr_flip_trigger_global_: np.ndarray,
                        base_nent_conf_: np.ndarray,
                        extra_meta: Optional[dict] = None,
                    ):
                        if len(base_correct_list_) == 0:
                            return None
                        if not (len(base_correct_list_) == len(cyclic_correct_list_) == len(full_correct_list_)):
                            return None

                        N_ = len(base_correct_list_)
                        betas_ = [i / 10.0 for i in range(11)]
                        C_cyc_ = float(k)
                        C_full_ = float(math.factorial(k))
                        perc_ = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0

                        # -------------------------
                        # 1) Standard cyclic/full mix
                        # -------------------------
                        curve_cyc_ = []
                        curve_full_ = []
                        for beta in betas_:
                            n = int(N_ * beta + 1e-9)
                            if n > 0:
                                acc_cyc_mix = (sum(cyclic_correct_list_[:n]) + sum(base_correct_list_[n:])) / float(N_)
                                acc_full_mix = (sum(full_correct_list_[:n]) + sum(base_correct_list_[n:])) / float(N_)
                            else:
                                acc_cyc_mix = sum(base_correct_list_) / float(N_)
                                acc_full_mix = sum(base_correct_list_) / float(N_)

                            cost_cyc = beta * C_cyc_ + (1.0 - beta) * 1.0
                            cost_full_mix = beta * C_full_ + (1.0 - beta) * 1.0
                            curve_cyc_.append((cost_cyc, acc_cyc_mix))
                            curve_full_.append((cost_full_mix, acc_full_mix))

                        # -------------------------
                        # 2) switch-full / switch-cyclic (legacy, gap 기반)
                        # -------------------------
                        curve_switch_full_ = []
                        curve_switch_cyc_ = []
                        try:
                            for beta in betas_:
                                n = int(N_ * beta + 1e-9)
                                thresh = float(np.quantile(default_conf_[:n], perc_)) if n > 0 else float(np.quantile(default_conf_, perc_))

                                total_cost_sf = 0.0
                                corrects_sf = 0
                                total_cost_sc = 0.0
                                corrects_sc = 0

                                for i in range(0, n):
                                    total_cost_sf += 1.0
                                    total_cost_sc += 1.0
                                    if base_correct_list_[i]:
                                        corrects_sf += 1
                                        corrects_sc += 1

                                for i in range(n, N_):
                                    is_ambiguous = (float(default_conf_[i]) < thresh)

                                    if is_ambiguous:
                                        total_cost_sf += float(len(perm_list))
                                        if full_correct_list_[i]:
                                            corrects_sf += 1
                                    else:
                                        total_cost_sf += 1.0
                                        if base_correct_list_[i]:
                                            corrects_sf += 1

                                    if is_ambiguous:
                                        total_cost_sc += float(k)
                                        if cyclic_correct_list_[i]:
                                            corrects_sc += 1
                                    else:
                                        total_cost_sc += 1.0
                                        if base_correct_list_[i]:
                                            corrects_sc += 1

                                curve_switch_full_.append((total_cost_sf / float(N_), corrects_sf / float(N_)))
                                curve_switch_cyc_.append((total_cost_sc / float(N_), corrects_sc / float(N_)))
                        except Exception as e:
                            logger.warning(f"[{subject}] ({tag}) Failed to compute switch curves: {e}")
                            curve_switch_full_ = []
                            curve_switch_cyc_ = []

                        # -------------------------
                        # 3) top2flip -> cyclic (legacy)
                        # -------------------------
                        curve_top2flip_cyc_ = []
                        try:
                            base_plus_flip_cost = 2.0
                            extra_cyclic_cost = float(k - 1)
                            cyc_after_flip_cost = base_plus_flip_cost + extra_cyclic_cost  # k+1

                            for beta in betas_:
                                n = int(N_ * beta + 1e-9)
                                thresh = float(np.quantile(default_conf_[:n], perc_)) if n > 0 else float(np.quantile(default_conf_, perc_))

                                total_cost = 0.0
                                corrects = 0

                                for i in range(0, n):
                                    total_cost += 1.0
                                    if base_correct_list_[i]:
                                        corrects += 1

                                for i in range(n, N_):
                                    if float(default_conf_[i]) >= thresh:
                                        total_cost += 1.0
                                        if base_correct_list_[i]:
                                            corrects += 1
                                        continue

                                    if bool(arr_flip_trigger_global_[i]):
                                        total_cost += cyc_after_flip_cost
                                        if cyclic_correct_list_[i]:
                                            corrects += 1
                                    else:
                                        total_cost += base_plus_flip_cost
                                        if base_correct_list_[i]:
                                            corrects += 1

                                curve_top2flip_cyc_.append((total_cost / float(N_), corrects / float(N_)))
                        except Exception as e:
                            logger.warning(f"[{subject}] ({tag}) Failed to compute top2flip->cyclic curve: {e}")
                            curve_top2flip_cyc_ = []

                        # -------------------------
                        # 4) AvgGap(static) -> cyclic (legacy)
                        # -------------------------
                        curve_avggap_static_ = []
                        try:
                            base_plus_flip_cost = 2.0
                            extra_cyclic_cost = float(k - 1)

                            for beta in betas_:
                                n = int(N_ * beta + 1e-9)
                                thresh = float(np.quantile(default_conf_[:n], perc_)) if n > 0 else float(np.quantile(default_conf_, perc_))

                                total_cost = 0.0
                                corrects = 0

                                for i in range(0, n):
                                    total_cost += 1.0
                                    if base_correct_list_[i]:
                                        corrects += 1

                                for i in range(n, N_):
                                    if float(default_conf_[i]) >= thresh:
                                        total_cost += 1.0
                                        if base_correct_list_[i]:
                                            corrects += 1
                                        continue

                                    total_cost += base_plus_flip_cost
                                    if float(mean_conf_[i]) < thresh:
                                        total_cost += extra_cyclic_cost
                                        if cyclic_correct_list_[i]:
                                            corrects += 1
                                    else:
                                        if base_correct_list_[i]:
                                            corrects += 1

                                curve_avggap_static_.append((total_cost / float(N_), corrects / float(N_)))
                        except Exception as e:
                            logger.warning(f"[{subject}] ({tag}) Failed to compute AvgGap(static)->cyclic curve: {e}")
                            curve_avggap_static_ = []

                        # =========================================================
                        # 5) NEW: Entropy-based 3-stage policy with minimal HP
                        # =========================================================
                        curve_avggap_dynamic_ = []
                        init_th1_fixed_ = float('nan')
                        init_th2_fixed_ = float('nan')
                        th_trace_by_beta_ = []
                        try:
                            base_plus_probe_cost = 2.0
                            extra_cyclic_cost = float(k - 2)

                            def _clamp01(x: float) -> float:
                                return max(0.0, min(1.0, float(x)))

                            ema = float(getattr(args, "ours_th1_ema", getattr(args, "ours_mad_alpha", 0.01)))
                            ema = max(0.0, min(1.0, ema))

                            # precompute base top1/top2 indices
                            base_top12_idx = []
                            for i in range(N_):
                                bp = np.asarray(base_probs_list_[i], dtype=np.float64)
                                sidx = np.argsort(bp)[::-1]
                                t1 = int(sidx[0])
                                t2 = int(sidx[1]) if len(sidx) > 1 else int(sidx[0])
                                base_top12_idx.append((t1, t2))

                            # ideal indices (content-space == option_ids index)
                            ideal_idx_list = []
                            for i in range(N_):
                                try:
                                    ideal_idx_list.append(int(option_ids.index(ideals_[i])))
                                except Exception:
                                    ideal_idx_list.append(-1)

                            # cyclic teacher precompute
                            cyc_conf_list = []
                            cyclic_pred_idx = []
                            for i in range(N_):
                                cyc_probs = [per_sample_probs_[i][idx] for idx in cyclic_indices]
                                agg_cyc = _aggregate_probs_over_permutations(cyc_probs, cyc_perms, k)
                                cyclic_pred_idx.append(int(np.argmax(agg_cyc)))
                                cyc_conf_list.append(_nent_conf(agg_cyc))
                            cyc_conf_list = np.asarray(cyc_conf_list, dtype=np.float64)
                            cyclic_pred_idx = np.asarray(cyclic_pred_idx, dtype=np.int64)

                            debug_teacher = bool(getattr(args, "ours_debug_teacher", False))
                            debug_teacher_n = int(getattr(args, "ours_debug_teacher_n", 5))

                            TRACE_STRIDE = 200
                            TRACE_MAX_PER_BETA = 300

                            for beta in betas_:
                                n = int(N_ * beta + 1e-9)
                                th1 = float(np.quantile(base_nent_conf_[:n], perc_)) if n > 0 else float(np.quantile(base_nent_conf_, perc_))
                                th1 = _clamp01(th1)

                                if hasattr(args, "ours_th1_init") and args.ours_th1_init is not None:
                                    th1 = _clamp01(float(args.ours_th1_init))

                                th2 = float(th1)
                                if hasattr(args, "ours_th2_init") and args.ours_th2_init is not None:
                                    th2 = _clamp01(float(args.ours_th2_init))

                                th1_init_beta = float(th1)
                                th2_init_beta = float(th2)

                                if math.isnan(init_th1_fixed_):
                                    init_th1_fixed_ = float(th1)
                                    init_th2_fixed_ = float(th2)

                                total_cost = 0.0
                                corrects = 0
                                dbg_printed = 0

                                cnt_stage1_stop = 0
                                cnt_stage2_stop = 0
                                cnt_stage3_cyc = 0
                                cnt_teacher_agree = 0
                                cnt_teacher_disagree = 0
                                cnt_th2_updates = 0

                                stage3_total = 0
                                stage3_pred2_correct = 0
                                stage3_predcyc_correct = 0

                                beta_trace = []

                                def _push_trace(i: int, stage: int, decision: str,
                                                c1: float = None, c2: float = None, shift: int = None,
                                                pred2: int = None, pred_cyc: int = None,
                                                w: float = None, y: float = None,
                                                force: bool = False):
                                    if len(beta_trace) >= TRACE_MAX_PER_BETA:
                                        return
                                    if (not force) and (i >= n) and ((i - n) % TRACE_STRIDE != 0):
                                        return
                                    rec = {"i": int(i), "stage": int(stage), "decision": str(decision),
                                           "th1": float(th1), "th2": float(th2)}
                                    if c1 is not None: rec["c1"] = float(c1)
                                    if c2 is not None: rec["c2"] = float(c2)
                                    if shift is not None: rec["shift"] = int(shift)
                                    if pred2 is not None: rec["pred2"] = int(pred2)
                                    if pred_cyc is not None: rec["pred_cyc"] = int(pred_cyc)
                                    if w is not None: rec["w"] = float(w)
                                    if y is not None: rec["y"] = float(y)
                                    beta_trace.append(rec)

                                _push_trace(i=n, stage=0, decision="init", force=True)

                                # prefix: base only
                                for i in range(0, n):
                                    total_cost += 1.0
                                    if base_correct_list_[i]:
                                        corrects += 1

                                # online region
                                for i in range(n, N_):
                                    c1 = float(base_nent_conf_[i])
                                    if c1 >= th1:
                                        total_cost += 1.0
                                        cnt_stage1_stop += 1
                                        if base_correct_list_[i]:
                                            corrects += 1
                                        _push_trace(i=i, stage=1, decision="stop_base", c1=c1)
                                        continue

                                    # stage2 probe
                                    total_cost += base_plus_probe_cost
                                    t1, t2 = base_top12_idx[i]
                                    if int(t1) != 0:
                                        shift = int(t1)
                                    else:
                                        shift = int(t2)
                                        if shift == 0:
                                            shift = 1

                                    probs_id = per_sample_probs_[i][identity_idx]
                                    probs_probe = per_sample_probs_[i][cyclic_indices[shift]]

                                    agg2 = _aggregate_probs_over_permutations(
                                        [probs_id.tolist(), probs_probe.tolist()],
                                        [cyc_perms[0], cyc_perms[shift]],
                                        k
                                    )

                                    pred2 = int(np.argmax(agg2))
                                    c2 = float(_nent_conf(agg2))
                                    go_cyclic = (c2 < th2)

                                    if not go_cyclic:
                                        cnt_stage2_stop += 1
                                        ideal_idx = ideal_idx_list[i]
                                        if ideal_idx >= 0 and pred2 == ideal_idx:
                                            corrects += 1
                                        th1 = _clamp01((1.0 - ema) * th1 + ema * th2)
                                        _push_trace(i=i, stage=2, decision="stop_probe", c1=c1, c2=c2, shift=shift, pred2=pred2)
                                        continue

                                    # stage3 full cyclic
                                    total_cost += extra_cyclic_cost
                                    cnt_stage3_cyc += 1
                                    if cyclic_correct_list_[i]:
                                        corrects += 1

                                    stage3_total += 1
                                    ideal_idx = ideal_idx_list[i]
                                    if ideal_idx >= 0:
                                        if pred2 == ideal_idx:
                                            stage3_pred2_correct += 1
                                        pred_cyc_idx = int(cyclic_pred_idx[i])
                                        if pred_cyc_idx == ideal_idx:
                                            stage3_predcyc_correct += 1

                                    pred_cyc = int(cyclic_pred_idx[i])
                                    y = 1.0 if pred_cyc != pred2 else 0.0
                                    w = float(cyc_conf_list[i])

                                    if y > 0.5:
                                        cnt_teacher_disagree += 1
                                    else:
                                        cnt_teacher_agree += 1

                                    th2_prev = th2
                                    th2 = _clamp01(th2 + ema * w * ((2.0 * y) - 1.0) * (th2 - c2))
                                    if abs(th2 - th2_prev) > 0.0:
                                        cnt_th2_updates += 1

                                    th1 = _clamp01((1.0 - ema) * th1 + ema * th2)
                                    _push_trace(i=i, stage=3, decision="go_cyclic_update",
                                                c1=c1, c2=c2, shift=shift, pred2=pred2, pred_cyc=pred_cyc, w=w, y=y, force=True)

                                    if debug_teacher and dbg_printed < debug_teacher_n:
                                        logger.info(_purple(
                                            f"[{subject}] [{tag}] [TEACH-MIN] beta={beta:.1f} i={i} "
                                            f"c1={c1:.4f} th1={th1:.4f} "
                                            f"c2={c2:.4f} th2={th2:.4f} "
                                            f"shift={shift} w={w:.4f} y={y:.0f} "
                                            f"pred2={pred2} pred_cyc={pred_cyc} "
                                            f"{'DIS' if y > 0.5 else 'AGR'}"
                                        ))
                                        dbg_printed += 1

                                curve_avggap_dynamic_.append((total_cost / float(N_), corrects / float(N_)))

                                if stage3_total > 0:
                                    acc_pred2_stage3 = stage3_pred2_correct / float(stage3_total)
                                    acc_cyc_stage3 = stage3_predcyc_correct / float(stage3_total)
                                    gain_stage3 = acc_cyc_stage3 - acc_pred2_stage3
                                else:
                                    acc_pred2_stage3 = float('nan')
                                    acc_cyc_stage3 = float('nan')
                                    gain_stage3 = float('nan')

                                th_trace_by_beta_.append({
                                    "beta": float(beta),
                                    "n": int(n),
                                    "N": int(N_),
                                    "ema": float(ema),
                                    "perc": float(perc_),
                                    "init_th1": float(th1_init_beta),
                                    "init_th2": float(th2_init_beta),
                                    "final_th1": float(th1),
                                    "final_th2": float(th2),
                                    "stage1_stop": int(cnt_stage1_stop),
                                    "stage2_stop": int(cnt_stage2_stop),
                                    "stage3_cyc": int(cnt_stage3_cyc),
                                    "teacher_disagree": int(cnt_teacher_disagree),
                                    "teacher_agree": int(cnt_teacher_agree),
                                    "th2_updates": int(cnt_th2_updates),
                                    "stage3_total": int(stage3_total),
                                    "stage3_acc_pred2": float(acc_pred2_stage3),
                                    "stage3_acc_cyc": float(acc_cyc_stage3),
                                    "stage3_gain": float(gain_stage3),
                                    "trace_stride": int(TRACE_STRIDE),
                                    "trace_max": int(TRACE_MAX_PER_BETA),
                                    "trace": beta_trace,
                                })
                        except Exception as e:
                            logger.warning(f"[{subject}] ({tag}) Failed to compute NEW dynamic curve: {e}")
                            curve_avggap_dynamic_ = []
                            init_th1_fixed_ = float('nan')
                            init_th2_fixed_ = float('nan')
                            th_trace_by_beta_ = []

                        curve_obj_ = {
                            'subject': subject,
                            'tag': str(tag),
                            'k': k,
                            'betas': betas_,
                            'default_accuracy': float(np.mean(np.asarray(base_correct_list_, dtype=np.float64))),
                            'cyclic': {
                                'costs': [float(c) for c, _ in curve_cyc_],
                                'accuracies': [float(a) for _, a in curve_cyc_]
                            },
                            'full': {
                                'costs': [float(c) for c, _ in curve_full_],
                                'accuracies': [float(a) for _, a in curve_full_]
                            },
                            'switch_full': {
                                'costs': [float(c) for c, _ in curve_switch_full_],
                                'accuracies': [float(a) for _, a in curve_switch_full_]
                            },
                            'switch_cyclic': {
                                'costs': [float(c) for c, _ in curve_switch_cyc_],
                                'accuracies': [float(a) for _, a in curve_switch_cyc_]
                            },
                            'ours_top2flip': {
                                'costs': [float(c) for c, _ in curve_top2flip_cyc_],
                                'accuracies': [float(a) for _, a in curve_top2flip_cyc_]
                            },
                            'ours_avggap_static': {
                                'costs': [float(c) for c, _ in curve_avggap_static_],
                                'accuracies': [float(a) for _, a in curve_avggap_static_]
                            },
                            'ours_avggap_dynamic': {
                                'costs': [float(c) for c, a in curve_avggap_dynamic_],
                                'accuracies': [float(a) for c, a in curve_avggap_dynamic_]
                            },
                            'ours_avggap': {  # backward-compat alias
                                'costs': [float(c) for c, _ in curve_avggap_static_],
                                'accuracies': [float(a) for _, a in curve_avggap_static_]
                            },
                            'ours_low_conf_percent': float(getattr(args, 'ours_low_conf_percent', 10.0)),
                            'ours_low_conf_frac': float(perc_),
                            'dynamic_init_th1': float(init_th1_fixed_),
                            'dynamic_init_th2': float(init_th2_fixed_),
                            'dynamic_ema': float(getattr(args, "ours_th1_ema", getattr(args, "ours_mad_alpha", 0.01))),
                            'dynamic_note': "entropy_conf stage1+stage2, probe uses cyclic shift(top1->A, else top2->A), th2 updated by teacher(cyclic vs pred2) weighted by cyclic entropy-conf",
                            'dynamic_th_trace': th_trace_by_beta_,
                        }
                        if extra_meta:
                            curve_obj_.update(extra_meta)

                        os.makedirs(curve_save_path_, exist_ok=True)
                        save_results(f'{curve_save_path_}/{subject}_beta_curve.jsonl', [curve_obj_], metrics=None)
                        return curve_obj_


                    # =========================================================
                    # Confidence stats & triggers (precompute)
                    # =========================================================
                    default_conf = []               # base gap: top1 - top2 (legacy curves)
                    mean_gap_list = []              # (legacy) gap(mean(base, swap)) used by AvgGap(static)
                    flip_trigger_mask_global = []   # (legacy) base argmax != swap argmax

                    # ✅ entropy conf for base (used in NEW dynamic)
                    base_nent_conf = []

                    for i, bp in enumerate(base_probs_list):
                        # gap
                        vals = np.sort(bp)[::-1]
                        if vals.shape[0] < 2:
                            top1, top2 = (vals[0], 0.0) if vals.shape[0] > 0 else (0.0, 0.0)
                        else:
                            top1, top2 = vals[0], vals[1]
                        default_conf.append(float(top1 - top2))

                        # base entropy-conf
                        base_nent_conf.append(_nent_conf(bp))

                        # swap (top1 <-> top2) permutation index (legacy)
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
                    base_nent_conf = np.asarray(base_nent_conf, dtype=np.float64)

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
                    curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                    if getattr(args, 'option_id_set', None):
                        curve_save_path += f'_id-{args.option_id_set}'
                    baseline_curve_obj = _compute_and_save_beta_curve(
                        curve_save_path_=curve_save_path,
                        tag="baseline",
                        per_sample_probs_=per_sample_probs,
                        base_probs_list_=base_probs_list,
                        ideals_=ideals,
                        base_correct_list_=base_correct_list,
                        cyclic_correct_list_=cyclic_correct_list,
                        full_correct_list_=full_correct_list,
                        default_conf_=default_conf,
                        mean_conf_=mean_conf,
                        arr_flip_trigger_global_=arr_flip_trigger_global,
                        base_nent_conf_=base_nent_conf,
                        extra_meta=None,
                    )

                    # =========================================================
                    # PRIDE: estimate global prior and re-run all methods on corrected probs
                    # =========================================================
                    if not bool(getattr(args, "disable_pride", False)) and len(per_sample_probs) > 0:
                        subject_key = f"{args.task}|{args.num_few_shot}|{args.model_name}|{subject}|{getattr(args,'option_id_set', '')}"
                        prior_global, prior_meta = _estimate_pride_global_prior(
                            per_sample_probs=per_sample_probs,
                            cyclic_indices=cyclic_indices,
                            ratio_prefix_samples=float(getattr(args, "pride_ratio_prefix_samples", 0.05)),
                            prefix_selector=str(getattr(args, "pride_prefix_selector", "random")),
                            base_seed=int(getattr(args, "pride_seed", 0)),
                            subject_key=subject_key,
                            base_conf=base_nent_conf,
                        )

                        # apply correction to all permutations
                        per_sample_probs_pride = [
                            _apply_pride_global_prior_to_probs_seq(ps, prior_global)
                            for ps in per_sample_probs
                        ]

                        # recompute derived lists for pride
                        base_probs_list_pride = []
                        base_correct_list_pride = []
                        cyclic_correct_list_pride = []
                        full_correct_list_pride = []

                        default_conf_pride = []
                        mean_gap_list_pride = []
                        flip_trigger_mask_pride = []
                        base_nent_conf_pride = []

                        # pride-derived base/cyclic results (optional but useful)
                        base_results_pride = []
                        cyclic_results_pride = []

                        for i in range(len(per_sample_probs_pride)):
                            probs_seq_np = per_sample_probs_pride[i]
                            s_idx = sample_idxs[i] if i < len(sample_idxs) else i
                            s_prompt = sample_prompts[i] if i < len(sample_prompts) else None
                            s_options = sample_options_list[i] if i < len(sample_options_list) else None

                            # cyclic
                            cyc_probs = [probs_seq_np[idx] for idx in cyclic_indices]
                            cyclic_results_pride.append({
                                'type': 'result',
                                'data': {
                                    'idx': s_idx,
                                    'prompt': s_prompt,
                                    'options': s_options,
                                    'probs': [cp.tolist() for cp in cyc_probs],
                                    'ideal': ideals[i],
                                },
                            })
                            agg_cyc = _aggregate_probs_over_permutations([cp.tolist() for cp in cyc_probs], cyc_perms, k)
                            pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                            cyclic_correct_list_pride.append(pred_cyc == ideals[i])

                            # base
                            base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
                            base_probs_list_pride.append(base_probs)
                            pred_base = option_ids[int(np.argmax(base_probs))]
                            corr_base = (pred_base == ideals[i])
                            base_correct_list_pride.append(corr_base)
                            base_results_pride.append({
                                'type': 'result',
                                'data': {
                                    'idx': s_idx,
                                    'prompt': s_prompt,
                                    'options': s_options,
                                    'probs': base_probs.tolist(),
                                    'sampled': pred_base,
                                    'ideal': ideals[i],
                                    'correct': corr_base,
                                },
                            })

                            # full
                            agg_full = _aggregate_probs_over_permutations(probs_seq_np, perm_list, k)
                            pred_full = option_ids[int(np.argmax(agg_full))]
                            full_correct_list_pride.append(pred_full == ideals[i])

                            # conf stats (pride)
                            vals = np.sort(base_probs)[::-1]
                            if vals.shape[0] < 2:
                                top1, top2 = (vals[0], 0.0) if vals.shape[0] > 0 else (0.0, 0.0)
                            else:
                                top1, top2 = vals[0], vals[1]
                            default_conf_pride.append(float(top1 - top2))
                            base_nent_conf_pride.append(_nent_conf(base_probs))

                            # swap (legacy)
                            sorted_idx = np.argsort(base_probs)[::-1]
                            top1_idx = int(sorted_idx[0])
                            top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx
                            perm_swap = list(identity_perm)
                            perm_swap[top1_idx], perm_swap[top2_idx] = perm_swap[top2_idx], perm_swap[top1_idx]
                            swap_idx = perm_index_map.get(tuple(perm_swap), identity_idx)

                            probs_base_raw = probs_seq_np[identity_idx]
                            probs_swap_raw = probs_seq_np[swap_idx]
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
                            mean_gap_list_pride.append(mean_gap)

                            pred_base_content = option_ids[int(np.argmax(agg_base))]
                            pred_swap_content = option_ids[int(np.argmax(agg_swap))]
                            flip_trigger_mask_pride.append(pred_base_content != pred_swap_content)

                        default_conf_pride = np.asarray(default_conf_pride, dtype=np.float64)
                        mean_conf_pride = np.asarray(mean_gap_list_pride, dtype=np.float64)
                        arr_flip_trigger_pride = np.asarray(flip_trigger_mask_pride, dtype=bool)
                        base_nent_conf_pride = np.asarray(base_nent_conf_pride, dtype=np.float64)

                        # save pride base/cyclic derived results
                        cyclic_save_path_pride = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic_pride'
                        if getattr(args, 'option_id_set', None):
                            cyclic_save_path_pride += f'_id-{args.option_id_set}'
                        os.makedirs(cyclic_save_path_pride, exist_ok=True)
                        save_results(f'{cyclic_save_path_pride}/{subject}.jsonl', cyclic_results_pride, metrics=None)

                        base_save_path_pride = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_pride'
                        if getattr(args, 'option_id_set', None):
                            base_save_path_pride += f'_id-{args.option_id_set}'
                        os.makedirs(base_save_path_pride, exist_ok=True)
                        base_metrics_pride = {'type': 'metric', 'data': {'accuracy': float(np.mean(np.asarray(base_correct_list_pride, dtype=np.float64)))}}
                        save_results(f'{base_save_path_pride}/{subject}.jsonl', base_results_pride, base_metrics_pride)

                        curve_save_path_pride = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full_pride'
                        if getattr(args, 'option_id_set', None):
                            curve_save_path_pride += f'_id-{args.option_id_set}'

                        pride_curve_obj = _compute_and_save_beta_curve(
                            curve_save_path_=curve_save_path_pride,
                            tag="pride",
                            per_sample_probs_=per_sample_probs_pride,
                            base_probs_list_=base_probs_list_pride,
                            ideals_=ideals,
                            base_correct_list_=base_correct_list_pride,
                            cyclic_correct_list_=cyclic_correct_list_pride,
                            full_correct_list_=full_correct_list_pride,
                            default_conf_=default_conf_pride,
                            mean_conf_=mean_conf_pride,
                            arr_flip_trigger_global_=arr_flip_trigger_pride,
                            base_nent_conf_=base_nent_conf_pride,
                            extra_meta={
                                "pride": {
                                    "prior": [float(x) for x in prior_global.tolist()],
                                    "prior_map": {str(k_): float(v) for k_, v in zip(option_ids, prior_global.tolist())},
                                    "meta": prior_meta,
                                }
                            },
                        )

                        # ---------------------------------------------------------
                        # Save + upload compare PNG to W&B (same run) if enabled
                        # ---------------------------------------------------------
                        try:
                            if (wandb_run is not None) and (baseline_curve_obj is not None) and (pride_curve_obj is not None):
                                compare_png_path = os.path.join(curve_save_path_pride, f"{subject}_beta_curve_compare.png")
                                _plot_compare_beta_curve_png(
                                    baseline_curve_obj=baseline_curve_obj,
                                    pride_curve_obj=pride_curve_obj,
                                    out_path=compare_png_path,
                                    title=f"{args.task} {subject} — Baseline vs PRIDE",
                                )
                                import wandb  # safe: only when wandb_run exists
                                wandb_run.log({
                                    f"beta_curve_compare/{subject}": wandb.Image(compare_png_path),
                                })
                                logger.info(_orange(f"W&B uploaded compare PNG: {compare_png_path}"))
                        except Exception as e:
                            logger.warning(f"W&B compare PNG upload failed: {e}")

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
