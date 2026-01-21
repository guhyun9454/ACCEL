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
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

# Matplotlib (headless)
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
    """cyclic rotations: ABCD, BCDA, CDAB, DABC"""
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


def _probe_shift_cyclic_put_top2_into_top1_slot(base_probs: np.ndarray, k: int) -> Tuple[int, int, int]:
    """
    핵심 규칙(너가 원한 것):
      - base에서 top1 인덱스 t1, top2 인덱스 t2를 찾고,
      - cyclic rotations(ABCD/BCDA/CDAB/DABC) 안에서
        "top2가 top1 슬롯(원래 top1 위치)으로 오도록" shift s를 고른다.

    수식:
      shift s = (t2 - t1) mod k

    예시(k=4):
      - top1=A(0), top2=C(2) => s=(2-0)=2 => CDAB
      - top1=D(3), top2=C(2) => s=(2-3)=3 => DABC

    return: (shift s, top1_idx t1, top2_idx t2)
    """
    bp = np.asarray(base_probs, dtype=np.float64)
    sidx = np.argsort(bp)[::-1]
    t1 = int(sidx[0])
    t2 = int(sidx[1]) if len(sidx) > 1 else int(sidx[0])

    s = int((t2 - t1) % int(k))
    # s==0이면 probe가 base랑 똑같아서 의미 없음 → 그냥 1로 밀기(가능한 경우)
    if s == 0:
        s = 1 if k > 1 else 0
    return s, t1, t2


def _read_results_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
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

    prefix_selector:
      - "random": deterministic random by subject_key+seed
      - "low_conf": base_conf(=base gap) 작은 샘플들로 prefix 구성
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
        # observed matrix using cyclic rotations only (k x k)
        observed = np.asarray([per_sample_probs[i][j] for j in cyclic_indices], dtype=np.float64)
        _, _, prior_i = debias_simple(observed)  # returns (debiased, ... , prior)
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
    Overlay baseline vs PRIDE (cyclic/full/default) and save PNG.
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


def _plot_compare_method_curve_png(baseline_curve_obj: dict,
                                   pride_curve_obj: dict,
                                   method_key: str,
                                   out_path: str,
                                   title: str) -> None:
    """
    Overlay a specific method curve (cost vs accuracy) for baseline vs PRIDE.
    method_key: 'ours_avggap' or 'ours_top2flip'
    """
    if method_key not in baseline_curve_obj or method_key not in pride_curve_obj:
        return

    b = baseline_curve_obj[method_key]
    p = pride_curve_obj[method_key]
    b_costs = b.get("costs", [])
    b_accs = b.get("accuracies", [])
    p_costs = p.get("costs", [])
    p_accs = p.get("accuracies", [])
    if len(b_costs) == 0 or len(p_costs) == 0:
        return

    plt.figure(figsize=(7.5, 5.0), dpi=160)
    plt.plot(b_costs, b_accs, marker='o', label=f'Baseline {method_key}')
    plt.plot(p_costs, p_accs, marker='o', linestyle='--', label=f'PRIDE {method_key}')
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

    # Optional W&B init
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
                    "option_id_set": getattr(args, "option_id_set", None),
                    "ours_low_conf_percent": getattr(args, "ours_low_conf_percent", None),
                    "disable_pride": getattr(args, "disable_pride", False),
                    "pride_ratio_prefix_samples": getattr(args, "pride_ratio_prefix_samples", 0.05),
                    "pride_prefix_selector": getattr(args, "pride_prefix_selector", "random"),
                    "pride_seed": getattr(args, "pride_seed", 0),
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

            # -------------------------
            # metrics (for existing settings)
            # -------------------------
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

            logger.info(_orange(f"Run completed: {subject}"))

            if not use_cached:
                save_results(cached_path, results, metrics)
                logger.info(f"Results saved: {subject}")

            # =========================================================
            # Derived policies & beta curves (ONLY when args.setting == 'full')
            # =========================================================
            if args.setting == 'full' and len(results) > 0:
                try:
                    # option IDs / k
                    if getattr(args, 'option_id_set', None):
                        option_ids = list(args.option_id_set)
                    else:
                        k_guess = len(results[0]['data']['options'])
                        option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                    k = len(option_ids)

                    from itertools import permutations
                    perm_list = list(sorted(permutations(range(k))))
                    identity_idx = perm_list.index(tuple(range(k)))

                    cyclic_indices = [
                        perm_list.index(tuple((i + s) % k for i in range(k)))
                        for s in range(k)
                    ]
                    cyc_perms = [tuple((i + s) % k for i in range(k)) for s in range(k)]

                    cyclic_results = []
                    base_results = []

                    per_sample_probs = []
                    base_probs_list = []
                    ideals = []
                    sample_idxs = []
                    sample_prompts = []
                    sample_options_list = []

                    base_correct_list = []
                    cyclic_correct_list = []
                    full_correct_list = []

                    full_total = 0
                    full_corrects = 0
                    cyclic_total = 0
                    cyclic_corrects = 0

                    # -------------------------
                    # Build base/cyclic/full correctness lists
                    # -------------------------
                    for r in results:
                        if r.get('type') != 'result':
                            continue
                        data = r['data']
                        probs_seq = data.get('probs')
                        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                            continue

                        probs_seq_np = np.asarray(probs_seq, dtype=np.float64)
                        per_sample_probs.append(probs_seq_np)
                        ideals.append(data['ideal'])
                        sample_idxs.append(data.get('idx'))
                        sample_prompts.append(data.get('prompt'))
                        sample_options_list.append(data.get('options'))

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
                        agg_cyc = _aggregate_probs_over_permutations(cyc_probs, cyc_perms, k)
                        pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                        corr_cyc = (pred_cyc == data['ideal'])
                        cyclic_correct_list.append(corr_cyc)
                        if corr_cyc:
                            cyclic_corrects += 1
                        cyclic_total += 1

                        # Base (identity only)
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

                        # Full (all perms)
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
                        base_correct_list_: List[bool],
                        cyclic_correct_list_: List[bool],
                        full_correct_list_: List[bool],
                        default_conf_: np.ndarray,          # base gap
                        mean_conf_: np.ndarray,             # gap(mean(base,probe))
                        arr_flip_trigger_global_: np.ndarray,
                        arr_probe2_correct_: np.ndarray,    # correctness when stopping after probe
                        extra_meta: Optional[dict] = None,
                    ):
                        if len(base_correct_list_) == 0:
                            return None
                        if not (len(base_correct_list_) == len(cyclic_correct_list_) == len(full_correct_list_) == len(arr_probe2_correct_)):
                            return None

                        N_ = len(base_correct_list_)
                        betas_ = [i / 10.0 for i in range(11)]
                        perc_ = max(min(getattr(args, 'ours_low_conf_percent', 10.0), 100.0), 0.0) / 100.0

                        C_cyc_ = float(k)
                        C_full_ = float(math.factorial(k))

                        # 1) beta mix curves (cyclic/full)
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

                        # 2) switch curves (threshold=quantile, per beta)
                        curve_switch_full_ = []
                        curve_switch_cyc_ = []
                        for beta in betas_:
                            n = int(N_ * beta + 1e-9)
                            thresh = float(np.quantile(default_conf_[:n], perc_)) if n > 0 else float(np.quantile(default_conf_, perc_))

                            total_cost_sf = 0.0
                            corrects_sf = 0
                            total_cost_sc = 0.0
                            corrects_sc = 0

                            # online prefix
                            for i in range(0, n):
                                total_cost_sf += 1.0
                                total_cost_sc += 1.0
                                if base_correct_list_[i]:
                                    corrects_sf += 1
                                    corrects_sc += 1

                            # decision region
                            for i in range(n, N_):
                                is_ambiguous = (float(default_conf_[i]) < thresh)

                                # switch_full
                                if is_ambiguous:
                                    total_cost_sf += float(len(perm_list))  # full perms
                                    if full_correct_list_[i]:
                                        corrects_sf += 1
                                else:
                                    total_cost_sf += 1.0
                                    if base_correct_list_[i]:
                                        corrects_sf += 1

                                # switch_cyclic
                                if is_ambiguous:
                                    total_cost_sc += float(k)  # cyclic rotations
                                    if cyclic_correct_list_[i]:
                                        corrects_sc += 1
                                else:
                                    total_cost_sc += 1.0
                                    if base_correct_list_[i]:
                                        corrects_sc += 1

                            curve_switch_full_.append((total_cost_sf / float(N_), corrects_sf / float(N_)))
                            curve_switch_cyc_.append((total_cost_sc / float(N_), corrects_sc / float(N_)))

                        # 3) ours_top2flip: base -> probe(shift s) -> (flip? cyclic : stop at probe)
                        #    - th1 = quantile(default_conf) per beta
                        #    - if cyclic => total cost = k (NOT k+1)
                        curve_top2flip_cyc_ = []
                        for beta in betas_:
                            n = int(N_ * beta + 1e-9)
                            th1 = float(np.quantile(default_conf_[:n], perc_)) if n > 0 else float(np.quantile(default_conf_, perc_))

                            total_cost = 0.0
                            corrects = 0

                            # online prefix
                            for i in range(0, n):
                                total_cost += 1.0
                                if base_correct_list_[i]:
                                    corrects += 1

                            # decision region
                            for i in range(n, N_):
                                if float(default_conf_[i]) >= th1:
                                    total_cost += 1.0
                                    if base_correct_list_[i]:
                                        corrects += 1
                                    continue

                                # base+probe already done conceptually (cost=2),
                                # but if go cyclic, we define "total cost=k" (remaining rotations)
                                if bool(arr_flip_trigger_global_[i]):
                                    total_cost += float(k)
                                    if cyclic_correct_list_[i]:
                                        corrects += 1
                                else:
                                    total_cost += 2.0
                                    if bool(arr_probe2_correct_[i]):
                                        corrects += 1

                            curve_top2flip_cyc_.append((total_cost / float(N_), corrects / float(N_)))

                        # 4) ours_avggap: base -> probe -> avggap(mean) compare with th2
                        #    - th1 = quantile(default_conf), th2 = quantile(mean_conf), both per beta
                        #    - if cyclic => total cost = k
                        curve_avggap_ = []
                        for beta in betas_:
                            n = int(N_ * beta + 1e-9)
                            th1 = float(np.quantile(default_conf_[:n], perc_)) if n > 0 else float(np.quantile(default_conf_, perc_))
                            th2 = float(np.quantile(mean_conf_[:n], perc_)) if n > 0 else float(np.quantile(mean_conf_, perc_))

                            total_cost = 0.0
                            corrects = 0

                            for i in range(0, n):
                                total_cost += 1.0
                                if base_correct_list_[i]:
                                    corrects += 1

                            for i in range(n, N_):
                                if float(default_conf_[i]) >= th1:
                                    total_cost += 1.0
                                    if base_correct_list_[i]:
                                        corrects += 1
                                    continue

                                # probe done
                                if float(mean_conf_[i]) < th2:
                                    total_cost += float(k)
                                    if cyclic_correct_list_[i]:
                                        corrects += 1
                                else:
                                    total_cost += 2.0
                                    if bool(arr_probe2_correct_[i]):
                                        corrects += 1

                            curve_avggap_.append((total_cost / float(N_), corrects / float(N_)))

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
                            'ours_avggap': {
                                'costs': [float(c) for c, _ in curve_avggap_],
                                'accuracies': [float(a) for _, a in curve_avggap_]
                            },
                        }
                        if extra_meta:
                            curve_obj_.update(extra_meta)

                        os.makedirs(curve_save_path_, exist_ok=True)
                        save_results(f'{curve_save_path_}/{subject}_beta_curve.jsonl', [curve_obj_], metrics=None)
                        return curve_obj_

                    # =========================================================
                    # Precompute confidence stats & probe triggers (BASELINE)
                    # =========================================================
                    default_conf = []                 # base gap: top1-top2 (letter-space)
                    mean_gap_list = []                # gap(mean(base,probe)) (content-space)
                    flip_trigger_mask = []            # pred_base != pred_probe (content-space)
                    probe2_correct_list = []          # correctness of argmax(mean_probs)

                    for i, bp in enumerate(base_probs_list):
                        bp = np.asarray(bp, dtype=np.float64)

                        # base gap
                        vals = np.sort(bp)[::-1]
                        top1 = float(vals[0]) if vals.shape[0] > 0 else 0.0
                        top2 = float(vals[1]) if vals.shape[0] > 1 else 0.0
                        default_conf.append(top1 - top2)

                        # cyclic probe shift: put top2 into top1 slot
                        shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(bp, k)
                        probe_perm_idx = cyclic_indices[shift]

                        # aggregate base into content-space (single perm)
                        probs_base_raw = per_sample_probs[i][identity_idx]
                        agg_base = _aggregate_probs_over_permutations(
                            [probs_base_raw.tolist()],
                            [tuple(range(k))],
                            k
                        )

                        # aggregate probe into content-space (single rotation perm)
                        probs_probe_raw = per_sample_probs[i][probe_perm_idx]
                        agg_probe = _aggregate_probs_over_permutations(
                            [probs_probe_raw.tolist()],
                            [cyc_perms[shift]],
                            k
                        )

                        mean_probs = (agg_base + agg_probe) / 2.0
                        vals_mean = np.sort(mean_probs)[::-1]
                        mean_gap = float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0
                        mean_gap_list.append(mean_gap)

                        pred_base = option_ids[int(np.argmax(agg_base))]
                        pred_probe = option_ids[int(np.argmax(agg_probe))]
                        flip_trigger_mask.append(pred_base != pred_probe)

                        pred2 = option_ids[int(np.argmax(mean_probs))]
                        probe2_correct_list.append(pred2 == ideals[i])

                    default_conf = np.asarray(default_conf, dtype=np.float64)
                    mean_conf = np.asarray(mean_gap_list, dtype=np.float64)
                    arr_flip_trigger = np.asarray(flip_trigger_mask, dtype=bool)
                    arr_probe2_correct = np.asarray(probe2_correct_list, dtype=bool)

                    # =========================================================
                    # Save cyclic/base derived results
                    # =========================================================
                    cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                    if getattr(args, 'option_id_set', None):
                        cyclic_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(cyclic_save_path, exist_ok=True)

                    cyclic_acc = (cyclic_corrects / cyclic_total) if cyclic_total > 0 else float('nan')
                    cyclic_metrics = {'type': 'metric', 'data': {'accuracy': cyclic_acc}} if cyclic_total > 0 else None
                    save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results, metrics=cyclic_metrics)
                    logger.info(_orange(f"Derived and saved cyclic results: {subject}"))
                    logger.info(_purple(f"[{subject}] Cyclic ensemble accuracy: {cyclic_acc:.4f}"))

                    base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                    if getattr(args, 'option_id_set', None):
                        base_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(base_save_path, exist_ok=True)
                    base_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64))) if len(base_correct_list) else float('nan')
                    base_metrics = {'type': 'metric', 'data': {'accuracy': base_acc}}
                    save_results(f'{base_save_path}/{subject}.jsonl', base_results, base_metrics)
                    logger.info(_orange(f"Derived and saved base results: {subject}"))

                    full_acc = (full_corrects / full_total) if full_total > 0 else float('nan')
                    logger.info(_purple(f"[{subject}] Full permutation ensemble accuracy: {full_acc:.4f}"))
                    logger.info(_purple(f"[{subject}] Accuracies — Full: {full_acc:.4f}, Cyclic: {cyclic_acc:.4f}, Default: {base_acc:.4f}"))

                    # =========================================================
                    # Beta curves (baseline)
                    # =========================================================
                    curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                    if getattr(args, 'option_id_set', None):
                        curve_save_path += f'_id-{args.option_id_set}'

                    baseline_curve_obj = _compute_and_save_beta_curve(
                        curve_save_path_=curve_save_path,
                        tag="baseline",
                        base_correct_list_=base_correct_list,
                        cyclic_correct_list_=cyclic_correct_list,
                        full_correct_list_=full_correct_list,
                        default_conf_=default_conf,
                        mean_conf_=mean_conf,
                        arr_flip_trigger_global_=arr_flip_trigger,
                        arr_probe2_correct_=arr_probe2_correct,
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
                            base_conf=default_conf,  # base gap as confidence
                        )

                        # apply correction to all permutations
                        per_sample_probs_pride = [
                            _apply_pride_global_prior_to_probs_seq(ps, prior_global)
                            for ps in per_sample_probs
                        ]

                        # recompute derived lists for pride
                        base_correct_pride = []
                        cyclic_correct_pride = []
                        full_correct_pride = []

                        default_conf_pride = []
                        mean_conf_pride_list = []
                        flip_trigger_pride = []
                        probe2_correct_pride_list = []

                        base_results_pride = []
                        cyclic_results_pride = []
                        cyclic_corrects_pride = 0

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
                            corr_cyc = (pred_cyc == ideals[i])
                            cyclic_correct_pride.append(corr_cyc)
                            if corr_cyc:
                                cyclic_corrects_pride += 1

                            # base
                            base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
                            pred_base = option_ids[int(np.argmax(base_probs))]
                            corr_base = (pred_base == ideals[i])
                            base_correct_pride.append(corr_base)
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
                            full_correct_pride.append(pred_full == ideals[i])

                            # pride base gap
                            vals = np.sort(base_probs)[::-1]
                            top1 = float(vals[0]) if vals.shape[0] > 0 else 0.0
                            top2 = float(vals[1]) if vals.shape[0] > 1 else 0.0
                            default_conf_pride.append(top1 - top2)

                            # probe shift from PRIDE base_probs
                            shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(base_probs, k)
                            probe_perm_idx = cyclic_indices[shift]

                            # base/probe aggregate in content-space
                            probs_base_raw = probs_seq_np[identity_idx]
                            agg_base = _aggregate_probs_over_permutations([probs_base_raw.tolist()], [tuple(range(k))], k)

                            probs_probe_raw = probs_seq_np[probe_perm_idx]
                            agg_probe = _aggregate_probs_over_permutations([probs_probe_raw.tolist()], [cyc_perms[shift]], k)

                            mean_probs = (agg_base + agg_probe) / 2.0
                            vals_mean = np.sort(mean_probs)[::-1]
                            mean_gap = float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0
                            mean_conf_pride_list.append(mean_gap)

                            pred_base2 = option_ids[int(np.argmax(agg_base))]
                            pred_probe2 = option_ids[int(np.argmax(agg_probe))]
                            flip_trigger_pride.append(pred_base2 != pred_probe2)

                            pred2 = option_ids[int(np.argmax(mean_probs))]
                            probe2_correct_pride_list.append(pred2 == ideals[i])

                        default_conf_pride = np.asarray(default_conf_pride, dtype=np.float64)
                        mean_conf_pride = np.asarray(mean_conf_pride_list, dtype=np.float64)
                        arr_flip_pride = np.asarray(flip_trigger_pride, dtype=bool)
                        arr_probe2_correct_pride = np.asarray(probe2_correct_pride_list, dtype=bool)

                        # save pride base/cyclic derived results
                        cyclic_save_path_pride = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic_pride'
                        if getattr(args, 'option_id_set', None):
                            cyclic_save_path_pride += f'_id-{args.option_id_set}'
                        os.makedirs(cyclic_save_path_pride, exist_ok=True)
                        cyclic_acc_pride = float(cyclic_corrects_pride) / float(len(cyclic_correct_pride)) if len(cyclic_correct_pride) else float('nan')
                        save_results(f'{cyclic_save_path_pride}/{subject}.jsonl', cyclic_results_pride, metrics={'type':'metric','data':{'accuracy':cyclic_acc_pride}})

                        base_save_path_pride = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_pride'
                        if getattr(args, 'option_id_set', None):
                            base_save_path_pride += f'_id-{args.option_id_set}'
                        os.makedirs(base_save_path_pride, exist_ok=True)
                        base_acc_pride = float(np.mean(np.asarray(base_correct_pride, dtype=np.float64))) if len(base_correct_pride) else float('nan')
                        save_results(f'{base_save_path_pride}/{subject}.jsonl', base_results_pride, metrics={'type':'metric','data':{'accuracy':base_acc_pride}})

                        curve_save_path_pride = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full_pride'
                        if getattr(args, 'option_id_set', None):
                            curve_save_path_pride += f'_id-{args.option_id_set}'

                        pride_curve_obj = _compute_and_save_beta_curve(
                            curve_save_path_=curve_save_path_pride,
                            tag="pride",
                            base_correct_list_=base_correct_pride,
                            cyclic_correct_list_=cyclic_correct_pride,
                            full_correct_list_=full_correct_pride,
                            default_conf_=default_conf_pride,
                            mean_conf_=mean_conf_pride,
                            arr_flip_trigger_global_=arr_flip_pride,
                            arr_probe2_correct_=arr_probe2_correct_pride,
                            extra_meta={
                                "pride": {
                                    "prior": [float(x) for x in prior_global.tolist()],
                                    "prior_map": {str(k_): float(v) for k_, v in zip(option_ids, prior_global.tolist())},
                                    "meta": prior_meta,
                                }
                            },
                        )

                        # compare plot upload
                        try:
                            if (wandb_run is not None) and (baseline_curve_obj is not None) and (pride_curve_obj is not None):
                                compare_png_path = os.path.join(curve_save_path_pride, f"{subject}_beta_curve_compare.png")
                                _plot_compare_beta_curve_png(
                                    baseline_curve_obj=baseline_curve_obj,
                                    pride_curve_obj=pride_curve_obj,
                                    out_path=compare_png_path,
                                    title=f"{args.task} {subject} — Baseline vs PRIDE",
                                )
                                import wandb
                                wandb_run.log({f"beta_curve_compare/{subject}": wandb.Image(compare_png_path)})
                        except Exception as e:
                            logger.warning(f"W&B compare PNG upload failed: {e}")

                        # optional: ours_avggap compare
                        try:
                            if (wandb_run is not None) and (baseline_curve_obj is not None) and (pride_curve_obj is not None):
                                avggap_png_path = os.path.join(curve_save_path_pride, f"{subject}_avggap_compare.png")
                                _plot_compare_method_curve_png(
                                    baseline_curve_obj=baseline_curve_obj,
                                    pride_curve_obj=pride_curve_obj,
                                    method_key="ours_avggap",
                                    out_path=avggap_png_path,
                                    title=f"{args.task} {subject} — ours_avggap Baseline vs PRIDE",
                                )
                                import wandb
                                wandb_run.log({f"avggap_compare/{subject}": wandb.Image(avggap_png_path)})
                        except Exception as e:
                            logger.warning(f"W&B ours_avggap compare PNG upload failed: {e}")

                except Exception as e:
                    logger.warning(f"Failed to derive beta curves for subject '{subject}': {e}")
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
