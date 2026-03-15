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
import time
import atexit
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
from transformers import logging as hf_logging

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# PriDe (PRIDE) helper: estimates option-token prior
from debias_utils import simple as debias_simple

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

def _pride_correct_row(row: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """PriDe correction: divide by prior then renormalize."""
    r = np.asarray(row, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64)
    adj = r / (pr + eps)
    adj = adj / (adj.sum() + eps)
    return adj


def _recall_std(labels: List[int], preds: List[int], k: int) -> float:
    """
    Simple recall std over classes 0..k-1.
    recall(c) = TP_c / P_c. Classes with P_c==0 are ignored.
    """
    if k <= 0:
        return float("nan")
    P = [0] * int(k)
    TP = [0] * int(k)
    for y, p in zip(labels, preds):
        if 0 <= int(y) < k:
            P[int(y)] += 1
            if int(p) == int(y):
                TP[int(y)] += 1
    recalls = []
    for c in range(int(k)):
        if P[c] > 0:
            recalls.append(TP[c] / float(P[c]))
    if len(recalls) == 0:
        return float("nan")
    return float(np.std(np.asarray(recalls, dtype=np.float64)))


def _run_online_avggap_policy_with_preds(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_pred_idx: List[int],
    cyclic_pred_idx: List[int],
    probe2_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    th1_percent: float,
    th2_percent: float,
    offline_prefix_n: int = 0,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, List[int]]:
    """
    Same decision rule as `_run_online_avggap_policy`, but returns predicted class indices.
    Used for recall-std analysis vs percentile p.
    """
    N = len(labels_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0
    preds: List[int] = []
    past_dc: List[float] = []
    past_mc: List[float] = []
    q1 = float(th1_percent) / 100.0
    q2 = float(th2_percent) / 100.0

    for i in range(N):
        gap_i = float(dc[i])
        mgap_i = float(mc[i])

        if i < int(offline_prefix_n):
            pred_i = int(base_pred_idx[i])
            preds.append(pred_i)
            total_cost += 1.0
            corrects += 1 if (pred_i == int(labels_idx[i])) else 0
            past_dc.append(gap_i)
            past_mc.append(mgap_i)
            continue

        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q1)) if len(past_dc) > 0 else 0.0
        th2_val = float(np.quantile(np.asarray(past_mc, dtype=np.float64), q2)) if len(past_mc) > 0 else 0.0

        if gap_i >= th1_val:
            c_step = 1.0
            pred_i = int(base_pred_idx[i])
        else:
            if mgap_i < th2_val:
                c_step = float(k)
                pred_i = int(cyclic_pred_idx[i])
            else:
                c_step = 2.0
                pred_i = int(probe2_pred_idx[i])

        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))

        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
        past_dc.append(gap_i)
        past_mc.append(mgap_i)

    return total_cost / float(N), corrects / float(N), preds


def _run_online_th1_quantile_th2_from_th1_rule_with_preds(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_pred_idx: List[int],
    cyclic_pred_idx: List[int],
    probe2_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    th1_percent: float,
    th2_rule_from_th1_value,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, float, List[int]]:
    """
    Same as `_run_online_th1_quantile_th2_from_th1_rule`, but returns preds.
    Returns: (cost, acc, th2_percent_estimate_over_all_samples, preds)
    """
    N = len(labels_idx)
    if N == 0:
        return float("nan"), float("nan"), float("nan"), []
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    q1 = float(th1_percent) / 100.0

    total_cost = 0.0
    corrects = 0
    preds: List[int] = []
    past_dc: List[float] = []
    final_th2_val = 0.0

    for i in range(N):
        gap_i = float(dc[i])
        mgap_i = float(mc[i])
        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q1)) if len(past_dc) > 0 else 0.0
        th2_val = float(th2_rule_from_th1_value(float(th1_val)))
        if th2_val < 0.0:
            th2_val = 0.0
        if th2_val > 1.0:
            th2_val = 1.0
        final_th2_val = th2_val

        if gap_i >= th1_val:
            c_step = 1.0
            pred_i = int(base_pred_idx[i])
        else:
            if mgap_i < th2_val:
                c_step = float(k)
                pred_i = int(cyclic_pred_idx[i])
            else:
                c_step = 2.0
                pred_i = int(probe2_pred_idx[i])

        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))
            pred_i = int(cyclic_pred_idx[i])  # forced → cyclic prediction

        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
        past_dc.append(gap_i)

    th2_perc = (np.sum(mc < float(final_th2_val)) / float(len(mc))) * 100.0 if len(mc) > 0 else 0.0
    return total_cost / float(N), corrects / float(N), float(th2_perc), preds


def _run_online_switch_cyclic_with_preds(
    default_conf: np.ndarray,
    base_pred_idx: List[int],
    cyclic_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    th1_percent: float,
    offline_prefix_n: int = 0,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, List[int]]:
    """
    Online switch_cyclic:
    - th1: online running-quantile over PAST default_conf gaps (percentile=th1_percent)
    - if dc >= th1 -> base, else -> cyclic
    Returns (cost, acc, preds_idx).
    """
    N = len(labels_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    dc = np.asarray(default_conf, dtype=np.float64)
    q1 = float(th1_percent) / 100.0

    total_cost = 0.0
    corrects = 0
    preds: List[int] = []
    past_dc: List[float] = []

    for i in range(N):
        gap_i = float(dc[i])

        if i < int(offline_prefix_n):
            pred_i = int(base_pred_idx[i])
            preds.append(pred_i)
            total_cost += 1.0
            corrects += 1 if (pred_i == int(labels_idx[i])) else 0
            past_dc.append(gap_i)
            continue

        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q1)) if len(past_dc) > 0 else 0.0
        if gap_i >= th1_val:
            c_step = 1.0
            pred_i = int(base_pred_idx[i])
        else:
            c_step = float(k)
            pred_i = int(cyclic_pred_idx[i])

        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))
            pred_i = int(cyclic_pred_idx[i])  # forced → cyclic prediction

        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
        past_dc.append(gap_i)

    return total_cost / float(N), corrects / float(N), preds


def _run_online_switch_cyclic_with_stats(
    default_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    k: int,
    th1_percent: float,
    offline_prefix_n: int = 0,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, Dict[str, int]]:
    """Returns (cost, acc, {n_base, n_cyclic})."""
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan"), {"n_base": 0, "n_cyclic": 0}
    dc = np.asarray(default_conf, dtype=np.float64)
    q1 = float(th1_percent) / 100.0
    total_cost, corrects = 0.0, 0
    n_base, n_cyclic = 0, 0
    past_dc: List[float] = []
    for i in range(N):
        gap_i = float(dc[i])
        if i < int(offline_prefix_n):
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            n_base += 1
            past_dc.append(gap_i)
            continue
        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q1)) if len(past_dc) > 0 else 0.0
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)
        if is_forced or gap_i < th1_val:
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
            n_cyclic += 1
        else:
            c_step = 1.0
            corrects += 1 if base_correct[i] else 0
            n_base += 1
        total_cost += float(c_step)
        past_dc.append(gap_i)
    return total_cost / float(N), corrects / float(N), {"n_base": int(n_base), "n_cyclic": int(n_cyclic)}


def _run_online_sqrt_policy_with_preds(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_pred_idx: List[int],
    cyclic_pred_idx: List[int],
    probe2_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    th1_percent: float,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, List[int]]:
    """
    Pred-returning version of `_run_online_sqrt_policy` (Online Sqrt All).
    """
    N = len(labels_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0
    preds: List[int] = []
    running_gap_sum = 0.0
    running_cnt = 0
    past_gaps: List[float] = []

    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), float(th1_percent) / 100.0)) if len(past_gaps) > 0 else 0.0
        current_avg_gap = (running_gap_sum / running_cnt) if running_cnt > 0 else 0.0
        safe_avg = min(1.0, max(0.0, float(current_avg_gap)))
        th2_val = float(th1_val) * float(np.sqrt(1.0 - safe_avg))

        if gap_i >= th1_val:
            c_step = 1.0
            pred_i = int(base_pred_idx[i])
        else:
            if float(mc[i]) < float(th2_val):
                c_step = float(k)
                pred_i = int(cyclic_pred_idx[i])
            else:
                c_step = 2.0
                pred_i = int(probe2_pred_idx[i])

        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))
            pred_i = int(cyclic_pred_idx[i])  # forced → cyclic prediction

        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
        running_gap_sum += gap_i
        running_cnt += 1
        past_gaps.append(gap_i)

    return total_cost / float(N), corrects / float(N), preds


def _run_online_sqrt_policy_lowconf_update_with_preds(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_pred_idx: List[int],
    cyclic_pred_idx: List[int],
    probe2_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    th1_percent: float,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, List[int]]:
    """
    Pred-returning version of `_run_online_sqrt_policy_lowconf_update`.
    """
    N = len(labels_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0
    preds: List[int] = []
    low_sum = 0.0
    low_cnt = 0
    past_gaps: List[float] = []

    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), float(th1_percent) / 100.0)) if len(past_gaps) > 0 else 0.0
        current_avg_gap = (low_sum / low_cnt) if low_cnt > 0 else 0.0
        safe_avg = min(1.0, max(0.0, float(current_avg_gap)))
        th2_val = float(th1_val) * float(np.sqrt(1.0 - safe_avg))

        if gap_i >= th1_val:
            c_step = 1.0
            pred_i = int(base_pred_idx[i])
        else:
            if float(mc[i]) < float(th2_val):
                c_step = float(k)
                pred_i = int(cyclic_pred_idx[i])
            else:
                c_step = 2.0
                pred_i = int(probe2_pred_idx[i])
            # update low-conf running stats (low-conf only update)
            low_sum += gap_i
            low_cnt += 1

        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))
            pred_i = int(cyclic_pred_idx[i])  # forced → cyclic prediction

        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
        past_gaps.append(gap_i)

    return total_cost / float(N), corrects / float(N), preds


def _stable_u32_seed(s: str, base_seed: int = 0) -> int:
    return (int(zlib.crc32(s.encode("utf-8"))) + int(base_seed)) & 0xFFFFFFFF


def _estimate_pride_prior_random_prefix_mean(
    per_sample_probs: List[np.ndarray],
    cyclic_indices: List[int],
    k: int,
    prefix_ratio: float,
    seed: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Estimate global prior over option letters using a random prefix subset (ratio),
    WITHOUT EMA: compute per-sample prior_i and take their mean.
    """
    N = len(per_sample_probs)
    if N <= 0:
        prior = np.ones((k,), dtype=np.float64) / float(k)
        return prior, {"N": 0, "m": 0, "used": 0, "ratio": float(prefix_ratio), "seed": int(seed), "prefix_ids": []}

    ratio = float(max(0.0, min(1.0, prefix_ratio)))
    m = int(max(1, int(round(N * ratio))))

    rng = np.random.default_rng(int(seed))
    prefix_ids = rng.choice(np.arange(N, dtype=np.int64), size=m, replace=False)
    prefix_ids = [int(x) for x in prefix_ids.tolist()]

    priors = []
    used = 0
    for i in prefix_ids:
        ps = np.asarray(per_sample_probs[i], dtype=np.float64)
        observed = np.asarray([ps[j] for j in cyclic_indices], dtype=np.float64)  # (k,k)
        try:
            _, _, prior_i = debias_simple(observed)
        except Exception:
            continue
        prior_i = np.asarray(prior_i, dtype=np.float64)
        prior_i = prior_i / (prior_i.sum() + eps)
        priors.append(prior_i)
        used += 1

    if len(priors) == 0:
        prior = np.ones((k,), dtype=np.float64) / float(k)
    else:
        prior = np.mean(np.asarray(priors, dtype=np.float64), axis=0)
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior / (prior.sum() + eps)

    meta = {"N": int(N), "m": int(m), "used": int(used), "ratio": float(ratio), "seed": int(seed), "prefix_ids": prefix_ids}
    return prior, meta


def _run_prefix_cyclic_postfix_base(
    base_correct: List[bool],
    cyclic_correct: List[bool],
    k: int,
    prefix_ids: set,
) -> Tuple[float, float]:
    """
    Default+PRIDE: prefix=cyclic, postfix=base.
    alpha=2 → 앞 2% cyclic, 뒤 98% 보정된 base로 측정.
    Returns (cost, acc).
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan")
    total_cost, corrects = 0.0, 0
    for i in range(N):
        if int(i) in prefix_ids:
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
    return total_cost / float(N), corrects / float(N)


def _run_cyclic_random_fraction(
    base_correct: List[bool],
    cyclic_correct: List[bool],
    k: int,
    fraction_pct: float,
    seed: int,
) -> Tuple[float, float]:
    """
    Randomly select fraction_pct% of samples to run cyclic; rest use base.
    Returns (cost, acc).
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan")
    frac = max(0.0, min(1.0, float(fraction_pct) / 100.0))
    m = int(round(frac * N))
    rng = np.random.default_rng(int(seed))
    cyclic_indices = set(rng.choice(np.arange(N, dtype=np.int64), size=min(m, N), replace=False))
    total_cost, corrects = 0.0, 0
    for i in range(N):
        if i in cyclic_indices:
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
    return total_cost / float(N), corrects / float(N)


def _run_cyclic_random_fraction_with_preds(
    base_pred_idx: List[int],
    cyclic_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    fraction_pct: float,
    seed: int,
) -> Tuple[float, float, List[int]]:
    """Returns (cost, acc, preds) for recall_std."""
    N = len(base_pred_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    frac = max(0.0, min(1.0, float(fraction_pct) / 100.0))
    m = int(round(frac * N))
    rng = np.random.default_rng(int(seed))
    cyclic_indices = set(rng.choice(np.arange(N, dtype=np.int64), size=min(m, N), replace=False))
    total_cost, corrects = 0.0, 0
    preds: List[int] = []
    for i in range(N):
        if i in cyclic_indices:
            pred_i = int(cyclic_pred_idx[i])
            total_cost += float(k)
        else:
            pred_i = int(base_pred_idx[i])
            total_cost += 1.0
        preds.append(pred_i)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
    return total_cost / float(N), corrects / float(N), preds


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
    probs_seq: list/array of length = (#permutations used)
      each element: length k (letter-space probs)
    permuted_indices: list of permutations p where p[j] is content-index at letter position j.
    Returns: agg (k,) content-space aggregated probs (mean over permutations)
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
    규칙:
      - base(letter-space)에서 top1=t1, top2=t2를 찾고,
      - cyclic rotations 중 "원래 top1 슬롯(=t1 위치)에 top2(t2)가 오도록" shift s 선택
    shift s = (t2 - t1) mod k
    """
    bp = np.asarray(base_probs, dtype=np.float64)
    sidx = np.argsort(bp)[::-1]
    t1 = int(sidx[0])
    t2 = int(sidx[1]) if len(sidx) > 1 else int(sidx[0])
    s = int((t2 - t1) % int(k))
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


def _quantile(arr: np.ndarray, p01: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    p01 = float(max(0.0, min(1.0, p01)))
    return float(np.quantile(arr, p01))


def _plot_confidence_distribution(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    out_path: str,
    title: str
):
    """
    default_conf (Base Gap)와 mean_conf (Avg Gap)의 분포(Histogram)를 그리고
    주요 Percentile 지점(10, 20, 30)을 표시
    """
    plt.figure(figsize=(10, 6), dpi=160)
    
    # Histogram
    plt.hist(default_conf, bins=50, range=(0, 1), alpha=0.5, label='Base Gap (default_conf)', color='gray', density=True)
    plt.hist(mean_conf, bins=50, range=(0, 1), alpha=0.5, label='Avg Gap (mean_conf)', color='blue', density=True)
    
    # Percentiles
    percs = [10, 20, 30]
    colors = ['red', 'green', 'purple']
    
    # Base Gap Percentiles
    for p, c in zip(percs, colors):
        val = np.percentile(default_conf, p)
        plt.axvline(val, color=c, linestyle='--', alpha=0.7, label=f'Base p{p}: {val:.3f}')
        
    # Avg Gap Percentiles
    for p, c in zip(percs, colors):
        val = np.percentile(mean_conf, p)
        plt.axvline(val, color=c, linestyle=':', alpha=0.9, linewidth=2, label=f'Avg p{p}: {val:.3f}')

    plt.xlabel("Confidence Gap")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_baseline_points_scatter(
    curve_obj: dict,
    out_path: str,
    title: str,
    extra_points: List[dict] = None
):
    """
    Baseline report에 나오는 각 정책들의 Cost vs Accuracy를 Point로 찍어서 비교
    extra_points: [{'cost': float, 'acc': float, 'label': str, 'marker': str, 'color': str}, ...]
    """
    plt.figure(figsize=(8, 6), dpi=160)
    
    # 1. Always Points (Reference)
    always = curve_obj.get("always", {})
    if "default" in always:
        plt.scatter(always["default"]["cost"], always["default"]["acc"], 
                   marker='*', s=300, color='gray', label='Default', zorder=10)
    if "cyclic" in always:
        plt.scatter(always["cyclic"]["cost"], always["cyclic"]["acc"], 
                   marker='d', s=150, color='purple', label='Cyclic', zorder=10)
    # Full removed as per user request
    # if "full" in always:
    #     plt.scatter(always["full"]["cost"], always["full"]["acc"], 
    #                marker='X', s=150, color='black', label='Full', zorder=10)

    # 2. Policy Points (REAL-WORLD online; single point)
    policies = ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]
    markers = ['s', '^', 'v', 'o']
    colors = ['orange', 'brown', 'green', 'blue']
    
    for key, m, c in zip(policies, markers, colors):
        if key in curve_obj:
            # single point (length-1)
            cost = float(curve_obj[key]["costs"][0])
            acc = float(curve_obj[key]["accuracies"][0])
            plt.scatter(cost, acc, marker=m, s=120, color=c, label=key, alpha=0.9)

    # 3. Extra Points (th1/sqrt(k), th1^2, th1^1.5, ...)
    if extra_points:
        for p in extra_points:
            plt.scatter(p['cost'], p['acc'], marker=p['marker'], s=150, color=p['color'], 
                       edgecolors='black', label=p['label'], zorder=15)

    plt.xlabel("Computational Cost (× of default)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_baseline_vs_pride_points_scatter(
    baseline_obj: dict,
    pride_obj: dict,
    out_path: str,
    title: str,
):
    """
    Overlay baseline vs PRIDE+OURS (same p) on one scatter.
    Baseline: filled markers
    PRIDE+OURS: same markers but hollow (edge only)
    """
    plt.figure(figsize=(8, 6), dpi=160)

    def _plot_one(obj: dict, prefix: str, hollow: bool):
        always = obj.get("always", {}) if isinstance(obj, dict) else {}
        if "default" in always:
            plt.scatter(
                float(always["default"]["cost"]),
                float(always["default"]["acc"]),
                marker="*",
                s=260,
                facecolors="none" if hollow else "gray",
                edgecolors="gray",
                linewidths=1.8 if hollow else 1.0,
                label=f"{prefix}Default",
                zorder=10,
            )
        if "cyclic" in always:
            plt.scatter(
                float(always["cyclic"]["cost"]),
                float(always["cyclic"]["acc"]),
                marker="d",
                s=140,
                facecolors="none" if hollow else "purple",
                edgecolors="purple",
                linewidths=1.8 if hollow else 1.0,
                label=f"{prefix}Cyclic",
                zorder=10,
            )

        policies = ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]
        markers = ['s', '^', 'v', 'o']
        colors = ['orange', 'brown', 'green', 'blue']
        for key, m, c in zip(policies, markers, colors):
            if key not in obj:
                continue
            cost = float(obj[key]["costs"][0])
            acc = float(obj[key]["accuracies"][0])
            plt.scatter(
                cost,
                acc,
                marker=m,
                s=110,
                facecolors="none" if hollow else c,
                edgecolors=c,
                linewidths=1.8 if hollow else 1.0,
                alpha=0.95,
                label=f"{prefix}{key}",
            )

        # heuristic points, if present (use stored marker/color when available)
        for hp in (obj.get("heuristic_points", []) or []):
            if not isinstance(hp, dict):
                continue
            cost = float(hp.get("cost", float("nan")))
            acc = float(hp.get("acc", float("nan")))
            lab = str(hp.get("label", "heuristic"))
            mk = str(hp.get("marker", "o"))
            col = str(hp.get("color", "black"))
            if np.isnan(cost) or np.isnan(acc):
                continue
            plt.scatter(
                cost,
                acc,
                marker=mk,
                s=80,
                facecolors="none" if hollow else col,
                edgecolors=col,
                linewidths=1.8 if hollow else 1.0,
                alpha=0.45,
                label=f"{prefix}{lab}" if prefix else lab,
            )

    _plot_one(baseline_obj, prefix="BASE_", hollow=False)
    _plot_one(pride_obj, prefix="PRIDE_", hollow=True)

    plt.xlabel("Computational Cost (× of default)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right', fontsize=7, ncol=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_three_curves_acc_recall_std(
    derived_records_by_p: Dict[float, List[dict]],
    derived_records_pride_by_p: Dict[float, List[dict]],
    derived_records_pride_by_alpha: Dict[float, List[dict]],
    out_dir: str,
    task: str,
    cyclic_fractions: List[int],
    pride_ours_fractions: List[float],
    pride_prefix_list: List[float],
    wandb_ok: bool = False,
    wandb_run: Any = None,
):
    """
    Three curves: (1) Cyclic (no PRIDE), (2) Default+PRIDE, (3) OURS (th1/2, no PRIDE).
    X=Cost, Y=Accuracy or Recall_std. Fractions configurable via args.
    """
    if not derived_records_by_p:
        logger.debug("three-curves plot skipped: derived_records_by_p empty")
        return
    color_cyclic = "#F39C12"
    color_pride = "#27AE60"
    color_ours = "#5DADE2"
    n_subjects = len(next(iter(derived_records_by_p.values()), []))
    macro_note = f" (macro over {n_subjects} subjects)" if n_subjects > 1 else ""

    def _agg_cyclic(by_p, fracs):
        costs, accs, rstds, acc_stds, rstd_stds = [], [], [], [], []
        p_any = next((float(p) for p in fracs if float(p) in by_p), None) or next(iter(by_p.keys()), None)
        cobjs = by_p.get(float(p_any), []) if p_any is not None else []
        for fp in fracs:
            key = f"cyclic_random_{fp}"
            cbs = [float(c[key]["costs"][0]) for c in cobjs if key in c]
            abs_ = [float(c[key]["accuracies"][0]) * 100.0 for c in cobjs if key in c]
            rbs = [float(c.get(f"{key}_recall_std", float("nan"))) for c in cobjs if key in c]
            costs.append(np.mean(cbs) if cbs else float("nan"))
            accs.append(np.mean(abs_) if abs_ else float("nan"))
            rstds.append(np.nanmean(rbs) if rbs else float("nan"))
            acc_stds.append(float(np.nanstd(abs_)) if len(abs_) > 1 else 0.0)
            rstd_stds.append(float(np.nanstd(rbs)) if len(rbs) > 1 else 0.0)
        return costs, accs, rstds, acc_stds, rstd_stds

    def _agg_heur(by_p, label, fracs):
        costs, accs, rstds, acc_stds, rstd_stds = [], [], [], [], []
        for p in fracs:
            pts = by_p.get(float(p), [])
            cl, al, rl = [], [], []
            for c in pts:
                hp_map = {str(h.get("label")): h for h in (c.get("heuristic_points", []) or []) if isinstance(h, dict)}
                h = hp_map.get(label, {})
                if h and "cost" in h:
                    cl.append(float(h["cost"]))
                if h and "acc" in h:
                    al.append(float(h["acc"]) * 100.0)
                if h and "recall_std" in h:
                    rl.append(float(h["recall_std"]))
            costs.append(np.mean(cl) if cl else float("nan"))
            accs.append(np.mean(al) if al else float("nan"))
            rstds.append(np.nanmean(rl) if rl else float("nan"))
            acc_stds.append(float(np.nanstd(al)) if len(al) > 1 else 0.0)
            rstd_stds.append(float(np.nanstd(rl)) if len(rl) > 1 else 0.0)
        return costs, accs, rstds, acc_stds, rstd_stds

    def _agg_pride_default(by_p, fracs):
        costs, accs, rstds, acc_stds, rstd_stds = [], [], [], [], []
        for p in fracs:
            pts = by_p.get(float(p), [])
            cl, al, rl = [], [], []
            for c in pts:
                pf = float(p)
                key_candidates = [
                    f"cyclic_random_{pf}",
                    f"cyclic_random_{pf:g}",  # e.g., 1.0 -> "1"
                ]
                key = next((kk for kk in key_candidates if kk in c), None)
                if key is not None:
                    cl.append(float(c[key]["costs"][0]))
                    al.append(float(c[key]["accuracies"][0]) * 100.0)
                    rkey = f"{key}_recall_std"
                    if rkey in c and isinstance(c.get(rkey), (int, float)):
                        rl.append(float(c[rkey]))
            costs.append(np.mean(cl) if cl else float("nan"))
            accs.append(np.mean(al) if al else float("nan"))
            rstds.append(np.nanmean(rl) if rl else float("nan"))
            acc_stds.append(float(np.nanstd(al)) if len(al) > 1 else 0.0)
            rstd_stds.append(float(np.nanstd(rl)) if len(rl) > 1 else 0.0)
        return costs, accs, rstds, acc_stds, rstd_stds

    def _agg_heur_by_th1_p(cobjs_list, th1_list, label_filter="online_sqrt_all"):
        """cobjs have heuristic_points with th1_p and label; extract by p and label."""
        costs, accs, rstds, acc_stds, rstd_stds = [], [], [], [], []
        for p in th1_list:
            cl, al, rl = [], [], []
            for c in cobjs_list:
                for h in (c.get("heuristic_points", []) or []):
                    if isinstance(h, dict) and h.get("th1_p") == p and h.get("label") == label_filter:
                        if "cost" in h:
                            cl.append(float(h["cost"]))
                        if "acc" in h:
                            al.append(float(h["acc"]) * 100.0)
                        if "recall_std" in h:
                            rl.append(float(h["recall_std"]))
                        break
            costs.append(np.mean(cl) if cl else float("nan"))
            accs.append(np.mean(al) if al else float("nan"))
            rstds.append(np.nanmean(rl) if rl else float("nan"))
            acc_stds.append(float(np.nanstd(al)) if len(al) > 1 else 0.0)
            rstd_stds.append(float(np.nanstd(rl)) if len(rl) > 1 else 0.0)
        return costs, accs, rstds, acc_stds, rstd_stds

    cost_cyc, acc_cyc, rstd_cyc, acc_std_cyc, rstd_std_cyc = _agg_cyclic(derived_records_by_p, cyclic_fractions)
    _n = len(pride_prefix_list) if pride_prefix_list else len(pride_ours_fractions)
    _def5 = ([float("nan")] * _n, [float("nan")] * _n, [float("nan")] * _n, [0.0] * _n, [0.0] * _n)

    # Default+PRIDE: alpha=2 → prefix 2% cyclic, postfix 98% base. 각 alpha당 1점.
    pride_fracs_for_plot = pride_ours_fractions
    if derived_records_pride_by_alpha:
        fracs_pride = [p for p in pride_prefix_list if p in derived_records_pride_by_alpha]
        by_p_def = {float(alpha): derived_records_pride_by_alpha[alpha] for alpha in fracs_pride}
        cost_pride, acc_pride, rstd_pride, acc_std_pride, rstd_std_pride = _agg_pride_default(by_p_def, fracs_pride) if by_p_def else _def5
        pride_fracs_for_plot = fracs_pride
    else:
        cost_pride, acc_pride, rstd_pride, acc_std_pride, rstd_std_pride = _agg_pride_default(derived_records_pride_by_p, pride_ours_fractions) if derived_records_pride_by_p else _def5

    cost_ours, acc_ours, rstd_ours, acc_std_ours, rstd_std_ours = _agg_heur(derived_records_by_p, "th1/2", pride_ours_fractions)

    # Ours+PRIDE: use derived_records_pride_by_alpha (first alpha for plot) or legacy by_p
    if derived_records_pride_by_alpha:
        alpha_ours = pride_prefix_list[0] if pride_prefix_list else 10
        cobjs_op = derived_records_pride_by_alpha.get(alpha_ours, [])
        cost_ours_pride, acc_ours_pride, rstd_ours_pride, acc_std_ours_pride, rstd_std_ours_pride = _agg_heur_by_th1_p(cobjs_op, pride_ours_fractions, "online_sqrt_all") if cobjs_op else _def5
    else:
        cost_ours_pride, acc_ours_pride, rstd_ours_pride, acc_std_ours_pride, rstd_std_ours_pride = _agg_heur(derived_records_pride_by_p, "th1/2", pride_ours_fractions) if derived_records_pride_by_p else _def5

    # Baseline = cyclic at fraction 0 (default, cost~1.0)
    default_acc = float(acc_cyc[0]) if acc_cyc and np.isfinite(acc_cyc[0]) else float("nan")
    default_recall_std = float(rstd_cyc[0]) if rstd_cyc and np.isfinite(rstd_cyc[0]) else float("nan")

    # Delta for plots: acc - default; recall_std: default - method (higher = better)
    delta_acc_cyc = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_cyc]
    delta_acc_pride = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_pride]
    delta_acc_ours = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_ours]
    delta_acc_ours_pride = [float(a - default_acc) if np.isfinite(a) and np.isfinite(default_acc) else float("nan") for a in acc_ours_pride]

    delta_rstd_cyc = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_cyc]
    delta_rstd_pride = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_pride]
    delta_rstd_ours = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_ours]
    delta_rstd_ours_pride = [float(default_recall_std - r) if np.isfinite(r) and np.isfinite(default_recall_std) else float("nan") for r in rstd_ours_pride]

    def _plot_curve(ax, costs, yvals, marker, color, linestyle, label):
        valid = [(c, y) for c, y in zip(costs, yvals) if np.isfinite(c) and np.isfinite(y)]
        if not valid:
            return
        xs, ys = zip(*sorted(valid, key=lambda t: t[0]))
        ax.plot(xs, ys, marker=marker, color=color, linestyle=linestyle, linewidth=2, markersize=8, label=label)

    os.makedirs(out_dir, exist_ok=True)
    # Plot 1: Delta Accuracy (cyclic, default_pride, ours)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax, cost_cyc, delta_acc_cyc, "o", color_cyclic, "-", "Cyclic")
    _plot_curve(ax, cost_pride, delta_acc_pride, "s", color_pride, "--", "PriDe")
    _plot_curve(ax, cost_ours, delta_acc_ours, "^", color_ours, "-.", "Ours")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax.set_title(f"{task} — Δ Accuracy{macro_note}", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_acc = os.path.join(out_dir, f"{task}_three_curves_acc.png")
    fig.savefig(out_acc, bbox_inches="tight")
    plt.close(fig)
    logger.info(_purple(f"Saved three-curves delta acc: {out_acc}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({f"plots/{task}/three_curves_acc": wandb.Image(out_acc)})
        except Exception:
            pass

    # Plot 2: Delta Recall std (default - method, higher = better)
    fig2, ax2 = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax2, cost_cyc, delta_rstd_cyc, "o", color_cyclic, "-", "Cyclic")
    _plot_curve(ax2, cost_pride, delta_rstd_pride, "s", color_pride, "--", "PriDe")
    _plot_curve(ax2, cost_ours, delta_rstd_ours, "^", color_ours, "-.", "Ours")
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax2.set_ylabel("Δ Recall std", fontsize=11)
    ax2.set_title(f"{task} — Δ Recall std{macro_note}", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout()
    out_rstd = os.path.join(out_dir, f"{task}_three_curves_recall_std.png")
    fig2.savefig(out_rstd, bbox_inches="tight")
    plt.close(fig2)
    logger.info(_purple(f"Saved three-curves delta recall_std: {out_rstd}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({f"plots/{task}/three_curves_recall_std": wandb.Image(out_rstd)})
        except Exception:
            pass

    # Plot 3-4: Ours vs Ours+PRIDE only (delta acc, delta recall_std)
    fig3, ax3 = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax3, cost_ours, delta_acc_ours, "^", color_ours, "-.", "Ours")
    _plot_curve(ax3, cost_ours_pride, delta_acc_ours_pride, "D", color_pride, "--", "Ours (with PriDe)")
    ax3.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax3.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax3.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax3.set_title(f"{task} — Ours vs Ours+PRIDE Δ Accuracy{macro_note}", fontsize=12)
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.4)
    fig3.tight_layout()
    out_ours_acc = os.path.join(out_dir, f"{task}_ours_vs_ours_pride_acc.png")
    fig3.savefig(out_ours_acc, bbox_inches="tight")
    plt.close(fig3)
    logger.info(_purple(f"Saved ours vs ours_pride delta acc: {out_ours_acc}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({f"plots/{task}/ours_vs_ours_pride_acc": wandb.Image(out_ours_acc)})
        except Exception:
            pass

    fig4, ax4 = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax4, cost_ours, delta_rstd_ours, "^", color_ours, "-.", "Ours")
    _plot_curve(ax4, cost_ours_pride, delta_rstd_ours_pride, "D", color_pride, "--", "Ours (with PriDe)")
    ax4.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax4.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax4.set_ylabel("Δ Recall std", fontsize=11)
    ax4.set_title(f"{task} — Ours vs Ours+PRIDE Δ Recall std{macro_note}", fontsize=12)
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.4)
    fig4.tight_layout()
    out_ours_rstd = os.path.join(out_dir, f"{task}_ours_vs_ours_pride_recall_std.png")
    fig4.savefig(out_ours_rstd, bbox_inches="tight")
    plt.close(fig4)
    logger.info(_purple(f"Saved ours vs ours_pride delta recall_std: {out_ours_rstd}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({f"plots/{task}/ours_vs_ours_pride_recall_std": wandb.Image(out_ours_rstd)})
        except Exception:
            pass

    # -------- Save numeric curve points for downstream averaging --------
    def _build_ours_pride_payload(
        by_alpha, prefix_list, th1_fracs, def_acc, def_rstd, agg_fn,
        cost_legacy, acc_legacy, rstd_legacy, dacc_legacy, drstd_legacy, acc_std_legacy, rstd_std_legacy,
    ):
        if by_alpha:
            by_alpha_out = {}
            for alpha in prefix_list:
                cobjs = by_alpha.get(alpha, [])
                if not cobjs:
                    continue
                entry = {}
                for variant in ("th1/2", "online_sqrt_all"):
                    co, ac, rs, asd, rsd = agg_fn(cobjs, th1_fracs, variant)
                    entry[variant] = {
                        "p": [float(x) for x in th1_fracs],
                        "cost": [float(x) if np.isfinite(x) else float("nan") for x in co],
                        "acc": [float(x) if np.isfinite(x) else float("nan") for x in ac],
                        "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rs],
                        "delta_acc": [float(a - def_acc) if np.isfinite(a) and np.isfinite(def_acc) else float("nan") for a in ac],
                        "delta_recall_std": [float(def_rstd - r) if np.isfinite(r) and np.isfinite(def_rstd) else float("nan") for r in rs],
                        "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in asd],
                        "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rsd],
                    }
                alpha_key = f"{float(alpha):g}"
                by_alpha_out[alpha_key] = entry
            return {
                "pride_prefix_fractions": [float(a) for a in prefix_list],
                "p": [float(x) for x in th1_fracs],
                "by_alpha": by_alpha_out,
            }
        return {
            "pride_prefix_fractions": [],
            "p": [int(x) for x in th1_fracs],
            "by_alpha": {},
            "cost": [float(x) if np.isfinite(x) else float("nan") for x in cost_legacy],
            "acc": [float(x) if np.isfinite(x) else float("nan") for x in acc_legacy],
            "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rstd_legacy],
            "delta_acc": [float(x) if np.isfinite(x) else float("nan") for x in dacc_legacy],
            "delta_recall_std": [float(x) if np.isfinite(x) else float("nan") for x in drstd_legacy],
            "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in acc_std_legacy],
            "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rstd_std_legacy],
        }

    try:
        points_path = os.path.join(out_dir, f"{task}_three_curves_points.json")
        payload = {
            "version": 2,
            "task": str(task),
            "default_acc": float(default_acc),
            "default_recall_std": float(default_recall_std),
            "cyclic_fractions": [int(x) for x in cyclic_fractions],
            "pride_ours_fractions": [float(x) for x in pride_ours_fractions],
            "curves": {
                "cyclic": {
                    "fraction": [int(x) for x in cyclic_fractions],
                    "cost": [float(x) if np.isfinite(x) else float("nan") for x in cost_cyc],
                    "acc": [float(x) if np.isfinite(x) else float("nan") for x in acc_cyc],
                    "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rstd_cyc],
                    "delta_acc": [float(x) if np.isfinite(x) else float("nan") for x in delta_acc_cyc],
                    "delta_recall_std": [float(x) if np.isfinite(x) else float("nan") for x in delta_rstd_cyc],
                    "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in acc_std_cyc],
                    "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rstd_std_cyc],
                },
                "default_pride": {
                    "p": [float(x) for x in pride_fracs_for_plot],
                    "cost": [float(x) if np.isfinite(x) else float("nan") for x in cost_pride],
                    "acc": [float(x) if np.isfinite(x) else float("nan") for x in acc_pride],
                    "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rstd_pride],
                    "delta_acc": [float(x) if np.isfinite(x) else float("nan") for x in delta_acc_pride],
                    "delta_recall_std": [float(x) if np.isfinite(x) else float("nan") for x in delta_rstd_pride],
                    "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in acc_std_pride],
                    "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rstd_std_pride],
                },
                "ours": {
                    "p": [float(x) for x in pride_ours_fractions],
                    "cost": [float(x) if np.isfinite(x) else float("nan") for x in cost_ours],
                    "acc": [float(x) if np.isfinite(x) else float("nan") for x in acc_ours],
                    "recall_std": [float(x) if np.isfinite(x) else float("nan") for x in rstd_ours],
                    "delta_acc": [float(x) if np.isfinite(x) else float("nan") for x in delta_acc_ours],
                    "delta_recall_std": [float(x) if np.isfinite(x) else float("nan") for x in delta_rstd_ours],
                    "delta_acc_std": [float(x) if np.isfinite(x) else 0.0 for x in acc_std_ours],
                    "delta_recall_std_std": [float(x) if np.isfinite(x) else 0.0 for x in rstd_std_ours],
                },
                "ours_pride": _build_ours_pride_payload(
                    derived_records_pride_by_alpha, pride_prefix_list, pride_ours_fractions,
                    default_acc, default_recall_std, _agg_heur_by_th1_p, cost_ours_pride,
                    acc_ours_pride, rstd_ours_pride, delta_acc_ours_pride, delta_rstd_ours_pride,
                    acc_std_ours_pride, rstd_std_ours_pride,
                ),
            },
        }
        with open(points_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(_purple(f"Saved three-curves points: {points_path}"))

        if wandb_ok and wandb_run is not None:
            try:
                import wandb
                # Put points into summary for easy retrieval via API (supports multi-task per run)
                existing = wandb_run.summary.get("three_curves_points_v1", {})
                if not isinstance(existing, dict):
                    existing = {}
                existing = dict(existing)
                existing[str(task)] = payload
                wandb_run.summary["three_curves_points_v1"] = existing

                # Also log as an artifact for durability
                art_name = f"three-curves-points-{str(task)}-{wandb_run.id}"
                art = wandb.Artifact(name=art_name, type="three_curves_points")
                art.add_file(points_path)
                wandb_run.log_artifact(art)
            except Exception as e:
                logger.warning(f"W&B three-curves points logging failed: {e}")
    except Exception as e:
        logger.warning(f"Failed to save three-curves points json: {e}")


def _plot_delta_cost_bars_by_p(
    delta_cost_by_p: Dict[float, Dict[str, float]],
    out_path: str,
    title: str,
    ylabel: str = "Δ Cost (PRIDE+OURS - BASELINE)",
):
    """
    Grouped bar chart: x-axis = p values, bars = policies/heuristics.
    delta_cost_by_p[p][label] = mean delta cost.
    """
    if not isinstance(delta_cost_by_p, dict) or len(delta_cost_by_p) == 0:
        return

    ps = sorted([float(p) for p in delta_cost_by_p.keys()])
    # collect labels that have at least one finite value
    all_labels = set()
    for p in ps:
        for lab, v in (delta_cost_by_p.get(p, {}) or {}).items():
            all_labels.add(str(lab))
    labels = sorted(list(all_labels))
    if len(labels) == 0:
        return

    # filter labels with any finite
    filt_labels = []
    for lab in labels:
        vs = []
        for p in ps:
            v = float((delta_cost_by_p.get(p, {}) or {}).get(lab, float("nan")))
            vs.append(v)
        if any(np.isfinite(v) for v in vs):
            filt_labels.append(lab)
    labels = filt_labels
    if len(labels) == 0:
        return

    x = np.arange(len(ps), dtype=np.float64)
    width = 0.80 / float(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)

    for i, lab in enumerate(labels):
        vals = []
        for p in ps:
            v = float((delta_cost_by_p.get(p, {}) or {}).get(lab, float("nan")))
            vals.append(0.0 if (not np.isfinite(v)) else v)
        offset = (i - (len(labels) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=str(lab))

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"p{int(round(p))}" for p in ps])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7, ncol=3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_scatter(
    delta_points: List[dict],
    out_path: str,
    title: str,
    xlabel: str = "Δ Cost (PRIDE+OURS - BASELINE)",
    ylabel: str = "Δ Accuracy (PRIDE+OURS - BASELINE)",
):
    """
    delta_points: [{'label': str, 'dcost': float, 'dacc': float, 'marker': str, 'color': str}, ...]
    """
    if not delta_points:
        return
    plt.figure(figsize=(8.4, 6.0), dpi=180)
    for p in delta_points:
        dcost = float(p.get("dcost", float("nan")))
        dacc = float(p.get("dacc", float("nan")))
        if not (np.isfinite(dcost) and np.isfinite(dacc)):
            continue
        lab = str(p.get("label", ""))
        mk = str(p.get("marker", "o"))
        col = str(p.get("color", "black"))
        plt.scatter(dcost, dacc, marker=mk, s=110, color=col, edgecolors="black", alpha=0.85)
        if lab:
            plt.annotate(lab, (dcost, dacc), textcoords="offset points", xytext=(6, 4), fontsize=7, alpha=0.85)
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    plt.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_th2_tradeoff_curve_compare(
    subject: str,
    curve_save_path: str,
    th1_list: List[float],
    default_conf_base: np.ndarray,
    mean_conf_base: np.ndarray,
    base_correct_base: List[bool],
    cyclic_correct_base: List[bool],
    probe2_correct_base: np.ndarray,
    default_conf_pr: np.ndarray,
    mean_conf_pr: np.ndarray,
    base_correct_pr: List[bool],
    cyclic_correct_pr: List[bool],
    probe2_correct_pr: np.ndarray,
    k: int,
    args: Any,
    wandb_ok: bool = False,
    wandb_run: Any = None,
    fname_tag: str = "PRIDE_COMPARE",
    forced_cyclic_ids_pr: Optional[set] = None,
):
    """
    Compare dense th2-sweep curves (baseline vs PRIDE+OURS debiased stats).
    Lines: baseline solid, PRIDE dashed. Colors encode th1.
    """
    dense_th2_list = list(range(1, 31))
    default_acc_base = float(np.mean(np.asarray(base_correct_base, dtype=np.float64))) if len(base_correct_base) else float("nan")
    # PRIDE default: prefix->cyclic, postfix->base (debias_pride.py와 동일)
    Npr = len(base_correct_pr)
    if forced_cyclic_ids_pr is not None and Npr > 0:
        default_corrects_pr = [
            cyclic_correct_pr[i] if i in forced_cyclic_ids_pr else base_correct_pr[i]
            for i in range(Npr)
        ]
        default_acc_pr = float(np.mean(np.asarray(default_corrects_pr, dtype=np.float64)))
    else:
        default_acc_pr = float(np.mean(np.asarray(base_correct_pr, dtype=np.float64))) if Npr else float("nan")
    # Anchor for PRIDE curves: use BASELINE default as reference (requested)
    anchor_default_acc = float(default_acc_base)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    suffix = f"_{str(fname_tag).strip()}" if str(fname_tag).strip() else ""

    # Plot A: Cost vs th2
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        costs_b = []
        costs_p = []
        for th2p in dense_th2_list:
            cb, _ = _run_online_avggap_policy(
                default_conf=default_conf_base,
                mean_conf=mean_conf_base,
                base_correct=base_correct_base,
                cyclic_correct=cyclic_correct_base,
                probe2_correct=probe2_correct_base,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
            )
            cp, _ = _run_online_avggap_policy(
                default_conf=default_conf_pr,
                mean_conf=mean_conf_pr,
                base_correct=base_correct_pr,
                cyclic_correct=cyclic_correct_pr,
                probe2_correct=probe2_correct_pr,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids_pr,
            )
            costs_b.append(float(cb))
            costs_p.append(float(cp))
        ax1.plot(dense_th2_list, costs_b, color=color, linewidth=1.6, alpha=0.75)
        ax1.plot(dense_th2_list, costs_p, color=color, linewidth=1.6, alpha=0.75, linestyle="--")

        # Heuristic points on Cost-vs-th2 compare
        try:
            def _pt(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_base,
                    mean_conf=mean_conf_base,
                    base_correct=base_correct_base,
                    cyclic_correct=cyclic_correct_base,
                    probe2_correct=probe2_correct_base,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=None,
                )

            def _pt_pr(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_pr,
                    mean_conf=mean_conf_pr,
                    base_correct=base_correct_pr,
                    cyclic_correct=cyclic_correct_pr,
                    probe2_correct=probe2_correct_pr,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=forced_cyclic_ids_pr,
                )

            pts = [
                ("*", lambda x: x / 2.0),
                ("P", lambda x, kk=k: x / math.sqrt(float(kk))),
                ("s", lambda x: x ** 2),
                ("^", lambda x: x ** 1.5),
            ]
            for mk, rf in pts:
                cb, _, p_b = _pt(rf)
                cp2, _, p_p2 = _pt_pr(rf)
                ax1.scatter([float(p_b)], [float(cb)], marker=mk, s=70, color=color, edgecolors="black", zorder=7)
                ax1.scatter([float(p_p2)], [float(cp2)], marker=mk, s=70, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_s, _, p_b_s = _run_online_sqrt_policy(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_s, _, p_p_s = _run_online_sqrt_policy(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax1.scatter([float(p_b_s)], [float(cb_s)], marker="D", s=60, color=color, edgecolors="black", zorder=7)
            ax1.scatter([float(p_p_s)], [float(cp_s)], marker="D", s=60, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_lc, _, p_b_lc = _run_online_sqrt_policy_lowconf_update(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_lc, _, p_p_lc = _run_online_sqrt_policy_lowconf_update(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax1.scatter([float(p_b_lc)], [float(cb_lc)], marker="X", s=60, color=color, edgecolors="black", zorder=7)
            ax1.scatter([float(p_p_lc)], [float(cp_lc)], marker="X", s=60, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)
        except Exception:
            pass
    ax1.set_xlabel("th2 (percentile, avg gap)")
    ax1.set_ylabel("Computational Cost (× of default)")
    ax1.set_title(f"{getattr(args,'task','task')} {subject} — Cost vs th2 (BASELINE solid vs PRIDE dashed)")
    try:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="BASELINE (solid)"),
            Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="PRIDE+OURS (dashed)"),
        ]
        ax1.legend(handles=handles, loc="best", fontsize=9)
    except Exception:
        pass
    ax1.grid(True, linestyle="--", alpha=0.35)
    out_cost = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST_compare{suffix}.png")
    os.makedirs(os.path.dirname(out_cost), exist_ok=True)
    fig1.tight_layout()
    fig1.savefig(out_cost, bbox_inches="tight")
    plt.close(fig1)

    # Plot B: ΔAcc vs th2 (reference = BASELINE default; PRIDE also uses BASELINE default)
    fig2, ax2 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        da_b = []
        da_p = []
        for th2p in dense_th2_list:
            _, ab = _run_online_avggap_policy(
                default_conf=default_conf_base,
                mean_conf=mean_conf_base,
                base_correct=base_correct_base,
                cyclic_correct=cyclic_correct_base,
                probe2_correct=probe2_correct_base,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
            )
            _, ap = _run_online_avggap_policy(
                default_conf=default_conf_pr,
                mean_conf=mean_conf_pr,
                base_correct=base_correct_pr,
                cyclic_correct=cyclic_correct_pr,
                probe2_correct=probe2_correct_pr,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids_pr,
            )
            da_b.append((float(ab) - float(anchor_default_acc)) * 100.0)
            da_p.append((float(ap) - float(anchor_default_acc)) * 100.0)
        ax2.plot(dense_th2_list, da_b, color=color, linewidth=1.6, alpha=0.75)
        ax2.plot(dense_th2_list, da_p, color=color, linewidth=1.6, alpha=0.75, linestyle="--")

        # Heuristic points on ΔAcc-vs-th2 compare
        try:
            def _pt(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_base,
                    mean_conf=mean_conf_base,
                    base_correct=base_correct_base,
                    cyclic_correct=cyclic_correct_base,
                    probe2_correct=probe2_correct_base,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=None,
                )

            def _pt_pr(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_pr,
                    mean_conf=mean_conf_pr,
                    base_correct=base_correct_pr,
                    cyclic_correct=cyclic_correct_pr,
                    probe2_correct=probe2_correct_pr,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=forced_cyclic_ids_pr,
                )

            pts = [
                ("*", lambda x: x / 2.0),
                ("P", lambda x, kk=k: x / math.sqrt(float(kk))),
                ("s", lambda x: x ** 2),
                ("^", lambda x: x ** 1.5),
            ]
            for mk, rf in pts:
                cb, ab, p_b = _pt(rf)
                cp2, ap2, p_p2 = _pt_pr(rf)
                ax2.scatter([float(p_b)], [(float(ab) - float(anchor_default_acc)) * 100.0], marker=mk, s=70, color=color, edgecolors="black", zorder=7)
                ax2.scatter([float(p_p2)], [(float(ap2) - float(anchor_default_acc)) * 100.0], marker=mk, s=70, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_s, ab_s, p_b_s = _run_online_sqrt_policy(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_s, ap_s, p_p_s = _run_online_sqrt_policy(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax2.scatter([float(p_b_s)], [(float(ab_s) - float(anchor_default_acc)) * 100.0], marker="D", s=60, color=color, edgecolors="black", zorder=7)
            ax2.scatter([float(p_p_s)], [(float(ap_s) - float(anchor_default_acc)) * 100.0], marker="D", s=60, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_lc, ab_lc, p_b_lc = _run_online_sqrt_policy_lowconf_update(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_lc, ap_lc, p_p_lc = _run_online_sqrt_policy_lowconf_update(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax2.scatter([float(p_b_lc)], [(float(ab_lc) - float(anchor_default_acc)) * 100.0], marker="X", s=60, color=color, edgecolors="black", zorder=7)
            ax2.scatter([float(p_p_lc)], [(float(ap_lc) - float(anchor_default_acc)) * 100.0], marker="X", s=60, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)
        except Exception:
            pass
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.set_xlabel("th2 (percentile, avg gap)")
    ax2.set_ylabel("Δ Accuracy (%)")
    ax2.set_title(f"{getattr(args,'task','task')} {subject} — ΔAcc vs th2 (ref=BASE default; BASE solid vs PRIDE dashed)")
    try:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="BASELINE (solid)"),
            Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="PRIDE+OURS (dashed)"),
        ]
        ax2.legend(handles=handles, loc="best", fontsize=9)
    except Exception:
        pass
    ax2.grid(True, linestyle="--", alpha=0.35)
    out_da = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_DELTA_ACC_compare{suffix}.png")
    fig2.tight_layout()
    fig2.savefig(out_da, bbox_inches="tight")
    plt.close(fig2)

    # Plot C: Cost vs ΔAcc (trade-off; reference = BASELINE default)
    fig3, ax3 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        xs_b, ys_b = [], []
        xs_p, ys_p = [], []
        for th2p in dense_th2_list:
            cb, ab = _run_online_avggap_policy(
                default_conf=default_conf_base,
                mean_conf=mean_conf_base,
                base_correct=base_correct_base,
                cyclic_correct=cyclic_correct_base,
                probe2_correct=probe2_correct_base,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
            )
            cp, ap = _run_online_avggap_policy(
                default_conf=default_conf_pr,
                mean_conf=mean_conf_pr,
                base_correct=base_correct_pr,
                cyclic_correct=cyclic_correct_pr,
                probe2_correct=probe2_correct_pr,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids_pr,
            )
            xs_b.append(float(cb)); ys_b.append((float(ab) - float(anchor_default_acc)) * 100.0)
            xs_p.append(float(cp)); ys_p.append((float(ap) - float(anchor_default_acc)) * 100.0)
        ax3.plot(xs_b, ys_b, color=color, linewidth=1.4, alpha=0.55)
        ax3.plot(xs_p, ys_p, color=color, linewidth=1.4, alpha=0.55, linestyle="--")

        # Heuristic points on the compare trade-off plot (same marker set as th2_tradeoff)
        try:
            # baseline (filled)
            def _pt(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_base,
                    mean_conf=mean_conf_base,
                    base_correct=base_correct_base,
                    cyclic_correct=cyclic_correct_base,
                    probe2_correct=probe2_correct_base,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=None,
                )

            def _pt_pr(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_pr,
                    mean_conf=mean_conf_pr,
                    base_correct=base_correct_pr,
                    cyclic_correct=cyclic_correct_pr,
                    probe2_correct=probe2_correct_pr,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=forced_cyclic_ids_pr,
                )

            pts = [
                ("*", lambda x: x / 2.0),
                ("P", lambda x, kk=k: x / math.sqrt(float(kk))),
                ("s", lambda x: x ** 2),
                ("^", lambda x: x ** 1.5),
            ]

            for mk, rf in pts:
                cb, ab, _ = _pt(rf)
                cp2, ap2, _ = _pt_pr(rf)
                ax3.scatter([float(cb)], [(float(ab) - float(anchor_default_acc)) * 100.0], marker=mk, s=75, color=color, edgecolors="black", zorder=7)
                ax3.scatter([float(cp2)], [(float(ap2) - float(anchor_default_acc)) * 100.0], marker=mk, s=75, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            # Online sqrt points
            cb_s, ab_s, _ = _run_online_sqrt_policy(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_s, ap_s, _ = _run_online_sqrt_policy(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax3.scatter([float(cb_s)], [(float(ab_s) - float(anchor_default_acc)) * 100.0], marker="D", s=65, color=color, edgecolors="black", zorder=7)
            ax3.scatter([float(cp_s)], [(float(ap_s) - float(anchor_default_acc)) * 100.0], marker="D", s=65, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_lc, ab_lc, _ = _run_online_sqrt_policy_lowconf_update(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_lc, ap_lc, _ = _run_online_sqrt_policy_lowconf_update(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax3.scatter([float(cb_lc)], [(float(ab_lc) - float(anchor_default_acc)) * 100.0], marker="X", s=65, color=color, edgecolors="black", zorder=7)
            ax3.scatter([float(cp_lc)], [(float(ap_lc) - float(anchor_default_acc)) * 100.0], marker="X", s=65, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)
        except Exception:
            pass
    ax3.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax3.set_xlabel("Computational Cost (× of default)")
    ax3.set_ylabel("Δ Accuracy (%)")
    ax3.set_title(f"{getattr(args,'task','task')} {subject} — Trade-off (ΔAcc ref=BASE default; BASE solid vs PRIDE dashed)")
    try:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="BASELINE (solid)"),
            Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="PRIDE+OURS (dashed)"),
        ]
        ax3.legend(handles=handles, loc="best", fontsize=9)
    except Exception:
        pass
    ax3.grid(True, linestyle="--", alpha=0.35)
    try:
        ax3.text(
            0.01,
            0.01,
            "Heuristic markers: * th1/2, P th1/sqrt(k), s th1^2, ^ th1^1.5, D Sqrt(All), X Sqrt(LowConf)\n"
            "PRIDE heuristic pts are hollow (edge=color), BASELINE are filled.",
            transform=ax3.transAxes,
            fontsize=7,
            alpha=0.85,
            va="bottom",
        )
    except Exception:
        pass
    out_tr = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST_vs_DELTA_compare{suffix}.png")
    fig3.tight_layout()
    fig3.savefig(out_tr, bbox_inches="tight")
    plt.close(fig3)

    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({
                f"plots/{subject}/th2_tradeoff_COST_compare{suffix}": wandb.Image(out_cost),
                f"plots/{subject}/th2_tradeoff_DELTA_ACC_compare{suffix}": wandb.Image(out_da),
                f"plots/{subject}/th2_tradeoff_COST_vs_DELTA_compare{suffix}": wandb.Image(out_tr),
            })
        except Exception:
            pass


def _run_online_sqrt_policy(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, float]:
    """
    [Online Sqrt Policy] with Forced Prefix Logic
    """
    N = len(base_correct)
    if N == 0:
        return 0.0, 0.0, 0.0

    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0
    
    running_gap_sum = 0.0
    running_cnt = 0
    past_gaps: List[float] = []
    
    final_th2_val = 0.0

    for i in range(N):
        gap_i = float(dc[i])

        # 1. Update Stats
        if len(past_gaps) > 0:
            th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), float(th1_percent) / 100.0))
        else:
            th1_val = 0.0
        
        if running_cnt > 0:
            current_avg_gap = running_gap_sum / running_cnt
        else:
            current_avg_gap = 0.0
            
        safe_avg = min(1.0, max(0.0, current_avg_gap))
        current_th2_val = th1_val * np.sqrt(1.0 - safe_avg)
        final_th2_val = current_th2_val

        # 2. Check Forced
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)

        if is_forced:
            # [Prefix] Force Cyclic
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
            # [Postfix] Policy
            if gap_i >= th1_val:
                c_step = 1.0
                corrects += 1 if base_correct[i] else 0
            else:
                if float(mc[i]) < current_th2_val:
                    c_step = float(k)
                    corrects += 1 if cyclic_correct[i] else 0
                else:
                    c_step = 2.0
                    corrects += 1 if bool(probe2_correct[i]) else 0

        # 3. Accumulate
        total_cost += float(c_step)
        running_gap_sum += gap_i
        running_cnt += 1
        past_gaps.append(gap_i)

    if len(mc) > 0:
        final_th2_perc = (np.sum(mc < final_th2_val) / len(mc)) * 100.0
    else:
        final_th2_perc = 0.0

    return total_cost / float(N), corrects / float(N), final_th2_perc


def _run_online_sqrt_policy_with_stats(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, float, Dict[str, int]]:
    """Returns (cost, acc, final_th2_perc, {n_base, n_probe2, n_cyclic})."""
    N = len(base_correct)
    if N == 0:
        return 0.0, 0.0, 0.0, {"n_base": 0, "n_probe2": 0, "n_cyclic": 0}
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    total_cost, corrects = 0.0, 0
    n_base, n_probe2, n_cyclic = 0, 0, 0
    running_gap_sum, running_cnt = 0.0, 0
    past_gaps: List[float] = []
    final_th2_val = 0.0
    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), float(th1_percent) / 100.0)) if len(past_gaps) > 0 else 0.0
        current_avg_gap = (running_gap_sum / running_cnt) if running_cnt > 0 else 0.0
        safe_avg = min(1.0, max(0.0, current_avg_gap))
        current_th2_val = th1_val * np.sqrt(1.0 - safe_avg)
        final_th2_val = current_th2_val
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)
        if is_forced:
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
            n_cyclic += 1
        elif gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            n_base += 1
        elif float(mc[i]) < current_th2_val:
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
            n_cyclic += 1
        else:
            total_cost += 2.0
            corrects += 1 if bool(probe2_correct[i]) else 0
            n_probe2 += 1
        running_gap_sum += gap_i
        running_cnt += 1
        past_gaps.append(gap_i)
    final_th2_perc = (np.sum(mc < final_th2_val) / len(mc)) * 100.0 if len(mc) > 0 else 0.0
    return total_cost / float(N), corrects / float(N), float(final_th2_perc), {"n_base": int(n_base), "n_probe2": int(n_probe2), "n_cyclic": int(n_cyclic)}


def _run_online_sqrt_policy_lowconf_update(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, float]:
    """
    [Online Sqrt Policy — LowConf-only update]
    th1 = online percentile (running quantile over observed default_conf gaps)
    th2 = th1 * sqrt(1 - CurrentAvgGapLowConf) (Online)

    - CurrentAvgGapLowConf: 과거(t-1)까지의 "th1 gate를 통과한(low-conf: dc < th1)" 샘플들의 default_conf 평균
      (즉, 2-stage로 들어온 샘플들만으로 AvgGap을 업데이트)
    """
    N = len(base_correct)
    if N == 0:
        return 0.0, 0.0, 0.0

    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0

    low_sum = 0.0
    low_cnt = 0
    final_th2_val = 0.0
    past_gaps: List[float] = []

    for i in range(N):
        gap_i = float(dc[i])

        # 0) Online th1 (past gaps only)
        if len(past_gaps) > 0:
            th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), float(th1_percent) / 100.0))
        else:
            th1_val = 0.0

        # 1) Running average based on PAST low-conf samples only
        if low_cnt > 0:
            current_avg_gap = low_sum / low_cnt
        else:
            # low-conf 샘플을 아직 못 봤으면 보수적으로(Avg=0) 시작 -> th2 ≈ th1
            current_avg_gap = 0.0

        # 2) Dynamic th2 (based on low-conf-only avg)
        safe_avg = min(1.0, max(0.0, current_avg_gap))
        current_th2_val = th1_val * np.sqrt(1.0 - safe_avg)
        final_th2_val = current_th2_val

        # 3) Execute policy
        if gap_i >= th1_val:
            c_step = 1.0
            corrects += 1 if base_correct[i] else 0
        else:
            if float(mc[i]) < current_th2_val:
                c_step = float(k)
                corrects += 1 if cyclic_correct[i] else 0
            else:
                c_step = 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0

        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))
        total_cost += float(c_step)

        # 4) Update low-conf-only stats AFTER decision
        if gap_i < th1_val:
            low_sum += gap_i
            low_cnt += 1
        past_gaps.append(gap_i)

    if len(mc) > 0:
        final_th2_perc = (np.sum(mc < final_th2_val) / len(mc)) * 100.0
    else:
        final_th2_perc = 0.0

    return total_cost / float(N), corrects / float(N), final_th2_perc


def _run_online_sqrt_policy_lowconf_update_with_stats(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, float, Dict[str, int]]:
    """Returns (cost, acc, final_th2_perc, {n_base, n_probe2, n_cyclic})."""
    N = len(base_correct)
    if N == 0:
        return 0.0, 0.0, 0.0, {"n_base": 0, "n_probe2": 0, "n_cyclic": 0}
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    total_cost, corrects = 0.0, 0
    n_base, n_probe2, n_cyclic = 0, 0, 0
    low_sum, low_cnt = 0.0, 0
    past_gaps: List[float] = []
    final_th2_val = 0.0
    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), float(th1_percent) / 100.0)) if len(past_gaps) > 0 else 0.0
        current_avg_gap = (low_sum / low_cnt) if low_cnt > 0 else 0.0
        safe_avg = min(1.0, max(0.0, current_avg_gap))
        current_th2_val = th1_val * np.sqrt(1.0 - safe_avg)
        final_th2_val = current_th2_val
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)
        if is_forced:
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
            n_cyclic += 1
        elif gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            n_base += 1
        elif float(mc[i]) < current_th2_val:
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
            n_cyclic += 1
        else:
            total_cost += 2.0
            corrects += 1 if bool(probe2_correct[i]) else 0
            n_probe2 += 1
        if gap_i < th1_val:
            low_sum += gap_i
            low_cnt += 1
        past_gaps.append(gap_i)
    final_th2_perc = (np.sum(mc < final_th2_val) / len(mc)) * 100.0 if len(mc) > 0 else 0.0
    return total_cost / float(N), corrects / float(N), float(final_th2_perc), {"n_base": int(n_base), "n_probe2": int(n_probe2), "n_cyclic": int(n_cyclic)}


def _run_online_th1_quantile_th2_from_th1_rule(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    th2_rule_from_th1_value,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, float]:
    """
    [Real-world Online Policy Template] with Forced Prefix Logic
    """
    N = len(base_correct)
    if N == 0:
        return 0.0, 0.0, 0.0

    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0
    past_gaps: List[float] = []
    final_th2_val = 0.0

    q = float(th1_percent) / 100.0

    for i in range(N):
        gap_i = float(dc[i])

        if len(past_gaps) > 0:
            th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), q))
        else:
            th1_val = 0.0

        th2_val = float(th2_rule_from_th1_value(th1_val))
        final_th2_val = th2_val

        # Check Forced
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)

        if is_forced:
             # [Prefix] Force Cyclic
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
            # [Postfix] Policy
            if gap_i >= th1_val:
                c_step = 1.0
                corrects += 1 if base_correct[i] else 0
            else:
                if float(mc[i]) < th2_val:
                    c_step = float(k)
                    corrects += 1 if cyclic_correct[i] else 0
                else:
                    c_step = 2.0
                    corrects += 1 if bool(probe2_correct[i]) else 0

        total_cost += float(c_step)
        past_gaps.append(gap_i)

    if len(mc) > 0:
        final_th2_perc = (np.sum(mc < final_th2_val) / len(mc)) * 100.0
    else:
        final_th2_perc = 0.0

    return total_cost / float(N), corrects / float(N), final_th2_perc


def _run_online_th1_quantile_th2_from_th1_rule_with_stats(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    th2_rule_from_th1_value,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, float, Dict[str, int]]:
    """
    Same as `_run_online_th1_quantile_th2_from_th1_rule`, but returns decision counts.
    """
    N = len(base_correct)
    if N == 0:
        return 0.0, 0.0, 0.0, {"n_base": 0, "n_probe2": 0, "n_cyclic": 0}

    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0
    past_gaps: List[float] = []
    final_th2_val = 0.0

    n_base = 0
    n_probe2 = 0
    n_cyclic = 0

    q = float(th1_percent) / 100.0

    for i in range(N):
        gap_i = float(dc[i])

        # Online th1 (past only)
        if len(past_gaps) > 0:
            th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), q))
        else:
            th1_val = 0.0

        th2_val = float(th2_rule_from_th1_value(float(th1_val)))
        if th2_val < 0.0:
            th2_val = 0.0
        if th2_val > 1.0:
            th2_val = 1.0
        final_th2_val = th2_val

        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)
        if is_forced:
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
            n_cyclic += 1
        elif gap_i >= th1_val:
            c_step = 1.0
            corrects += 1 if base_correct[i] else 0
            n_base += 1
        else:
            if float(mc[i]) < th2_val:
                c_step = float(k)
                corrects += 1 if cyclic_correct[i] else 0
                n_cyclic += 1
            else:
                c_step = 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0
                n_probe2 += 1

        total_cost += float(c_step)

        past_gaps.append(gap_i)

    if len(mc) > 0:
        final_th2_perc = (np.sum(mc < final_th2_val) / len(mc)) * 100.0
    else:
        final_th2_perc = 0.0

    return total_cost / float(N), corrects / float(N), float(final_th2_perc), {"n_base": int(n_base), "n_probe2": int(n_probe2), "n_cyclic": int(n_cyclic)}


def _run_online_top2flip_policy_with_preds(
    default_conf: np.ndarray,
    flip_trigger: np.ndarray,
    base_pred_idx: List[int],
    cyclic_pred_idx: List[int],
    probe2_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    th1_percent: float,
    offline_prefix_n: int = 0,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, List[int]]:
    """Returns (cost, acc, preds)."""
    N = len(labels_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    dc = np.asarray(default_conf, dtype=np.float64)
    flip = np.asarray(flip_trigger, dtype=bool)
    q = float(th1_percent) / 100.0
    total_cost, corrects = 0.0, 0
    preds: List[int] = []
    past_dc: List[float] = []
    for i in range(N):
        gap_i = float(dc[i])
        if i < int(offline_prefix_n):
            pred_i = int(base_pred_idx[i])
            preds.append(pred_i)
            total_cost += 1.0
            corrects += 1 if (pred_i == int(labels_idx[i])) else 0
            past_dc.append(gap_i)
            continue
        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q)) if len(past_dc) > 0 else 0.0
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)
        if is_forced:
            pred_i = int(cyclic_pred_idx[i])
        elif gap_i >= th1_val:
            pred_i = int(base_pred_idx[i])
        elif bool(flip[i]):
            pred_i = int(cyclic_pred_idx[i])
        else:
            pred_i = int(probe2_pred_idx[i])
        preds.append(pred_i)
        c_step = float(k) if (is_forced or (gap_i < th1_val and bool(flip[i]))) else (1.0 if gap_i >= th1_val else 2.0)
        total_cost += c_step
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
        past_dc.append(gap_i)
    return total_cost / float(N), corrects / float(N), preds


def _run_online_top2flip_policy(
    default_conf: np.ndarray,
    flip_trigger: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    offline_prefix_n: int = 0,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float]:
    """
    [Real-world Online top2flip] with Forced Prefix Logic
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan")

    dc = np.asarray(default_conf, dtype=np.float64)
    flip = np.asarray(flip_trigger, dtype=bool)

    total_cost = 0.0
    corrects = 0
    past_gaps: List[float] = []
    q = float(th1_percent) / 100.0

    for i in range(N):
        gap_i = float(dc[i])

        if i < int(offline_prefix_n):
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            past_gaps.append(gap_i)
            continue

        if len(past_gaps) > 0:
            th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), q))
        else:
            th1_val = 0.0

        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)

        if is_forced:
            # [Prefix] Force Cyclic
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
            # [Postfix] Policy
            if gap_i >= th1_val:
                c_step = 1.0
                corrects += 1 if base_correct[i] else 0
            else:
                if bool(flip[i]):
                    c_step = float(k)
                    corrects += 1 if cyclic_correct[i] else 0
                else:
                    c_step = 2.0
                    corrects += 1 if bool(probe2_correct[i]) else 0

        total_cost += float(c_step)
        past_gaps.append(gap_i)

    return total_cost / float(N), corrects / float(N)


def _run_online_top2flip_policy_with_stats(
    default_conf: np.ndarray,
    flip_trigger: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    offline_prefix_n: int = 0,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, Dict[str, int]]:
    """Returns (cost, acc, {n_base, n_cyclic, n_probe2}). th1 pass->base; th1 fail & flip->cyclic; th1 fail & no flip->probe2."""
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan"), {"n_base": 0, "n_cyclic": 0, "n_probe2": 0}
    dc = np.asarray(default_conf, dtype=np.float64)
    flip = np.asarray(flip_trigger, dtype=bool)
    q = float(th1_percent) / 100.0
    total_cost, corrects = 0.0, 0
    n_base, n_cyclic, n_probe2 = 0, 0, 0
    past_dc: List[float] = []
    for i in range(N):
        gap_i = float(dc[i])
        if i < int(offline_prefix_n):
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            n_base += 1
            past_dc.append(gap_i)
            continue
        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q)) if len(past_dc) > 0 else 0.0
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)
        if is_forced:
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
            n_cyclic += 1
        elif gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            n_base += 1
        elif bool(flip[i]):
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
            n_cyclic += 1
        else:
            total_cost += 2.0
            corrects += 1 if bool(probe2_correct[i]) else 0
            n_probe2 += 1
        past_dc.append(gap_i)
    return total_cost / float(N), corrects / float(N), {"n_base": int(n_base), "n_cyclic": int(n_cyclic), "n_probe2": int(n_probe2)}


def _run_online_avggap_policy(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    th2_percent: float,
    offline_prefix_n: int = 0,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float]:
    """
    [Real-world Online avggap]
    [Modified] If forced_cyclic_ids matches, force Cost=k and Acc=Cyclic.
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan")

    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0
    past_dc: List[float] = []
    past_mc: List[float] = []
    q1 = float(th1_percent) / 100.0
    q2 = float(th2_percent) / 100.0

    for i in range(N):
        gap_i = float(dc[i])
        mgap_i = float(mc[i])

        if i < int(offline_prefix_n):
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            past_dc.append(gap_i)
            past_mc.append(mgap_i)
            continue

        # 1. Update Thresholds from Past
        if len(past_dc) > 0:
            th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q1))
        else:
            th1_val = 0.0

        if len(past_mc) > 0:
            th2_val = float(np.quantile(np.asarray(past_mc, dtype=np.float64), q2))
        else:
            th2_val = 0.0

        # 2. Check Forced (Prefix)
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)

        if is_forced:
            # [Prefix] 무조건 Cyclic
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
            # [Postfix] Policy Decision
            if gap_i >= th1_val:
                c_step = 1.0
                corrects += 1 if base_correct[i] else 0
            else:
                if mgap_i < th2_val:
                    c_step = float(k)
                    corrects += 1 if cyclic_correct[i] else 0
                else:
                    c_step = 2.0
                    corrects += 1 if bool(probe2_correct[i]) else 0

        # 3. Accumulate & Update Stats
        total_cost += float(c_step)
        past_dc.append(gap_i)
        past_mc.append(mgap_i)

    return total_cost / float(N), corrects / float(N)


def _run_online_avggap_policy_with_stats(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    th2_percent: float,
    offline_prefix_n: int = 0,
    forced_cyclic_ids: Optional[set] = None,
) -> Tuple[float, float, Dict[str, int]]:
    """
    Same as `_run_online_avggap_policy`, but returns decision counts:
    - n_base: used base (dc >= th1)
    - n_probe2: used probe2 (dc < th1 and mc >= th2)
    - n_cyclic: used cyclic (dc < th1 and mc < th2)
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan"), {"n_base": 0, "n_probe2": 0, "n_cyclic": 0}

    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0
    past_dc: List[float] = []
    past_mc: List[float] = []
    q1 = float(th1_percent) / 100.0
    q2 = float(th2_percent) / 100.0

    n_base = 0
    n_probe2 = 0
    n_cyclic = 0

    for i in range(N):
        gap_i = float(dc[i])
        mgap_i = float(mc[i])

        if i < int(offline_prefix_n):
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            n_base += 1
            past_dc.append(gap_i)
            past_mc.append(mgap_i)
            continue

        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q1)) if len(past_dc) > 0 else 0.0
        th2_val = float(np.quantile(np.asarray(past_mc, dtype=np.float64), q2)) if len(past_mc) > 0 else 0.0

        if gap_i >= th1_val:
            c_step = 1.0
            corrects += 1 if base_correct[i] else 0
            n_base += 1
        else:
            if mgap_i < th2_val:
                c_step = float(k)
                corrects += 1 if cyclic_correct[i] else 0
                n_cyclic += 1
            else:
                c_step = 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0
                n_probe2 += 1

        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))
        total_cost += float(c_step)

        past_dc.append(gap_i)
        past_mc.append(mgap_i)

    return total_cost / float(N), corrects / float(N), {"n_base": int(n_base), "n_probe2": int(n_probe2), "n_cyclic": int(n_cyclic)}


def _run_online_dynamic_policy(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int
) -> Tuple[float, float, float]:
    """
    [Fully Dynamic Online Policy]
    - th1 = 1.0 - OnlineAvgGap
    - th2 = OnlineAvgGap - OnlineStdDev (clipped at 0)
    """
    N = len(base_correct)
    if N == 0:
        return 0.0, 0.0, 0.0

    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)

    total_cost = 0.0
    corrects = 0
    
    running_sum = 0.0
    running_sq_sum = 0.0
    running_cnt = 0
    
    final_th2_val = 0.0

    for i in range(N):
        gap_i = float(dc[i])
        
        if running_cnt > 1:
            mu = running_sum / running_cnt
            var = (running_sq_sum / running_cnt) - (mu ** 2)
            sigma = np.sqrt(max(0.0, var))
        else:
            mu = 0.0
            sigma = 0.0
            
        current_th1 = 1.0 - mu
        current_th2 = mu - sigma
        if current_th2 < 0.0: current_th2 = 0.0
        final_th2_val = current_th2

        if gap_i >= current_th1:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
        else:
            if float(mc[i]) < current_th2:
                total_cost += float(k)
                corrects += 1 if cyclic_correct[i] else 0
            else:
                total_cost += 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0
        
        running_sum += gap_i
        running_sq_sum += (gap_i ** 2)
        running_cnt += 1

    if len(mc) > 0:
        final_th2_perc = (np.sum(mc < final_th2_val) / len(mc)) * 100.0
    else:
        final_th2_perc = 0.0

    return total_cost / float(N), corrects / float(N), final_th2_perc


def _compute_and_plot_th2_tradeoff(
    subject: str,
    curve_save_path: str,
    th1_list: List[float],
    th2_list: List[float],
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct_list: List[bool],
    cyclic_correct_list: List[bool],
    arr_probe2_correct: np.ndarray,
    k: int,
    args: Any,
    wandb_ok: bool = False,
    wandb_run: Any = None,
    plot_tag: str = "BASELINE",
    fname_tag: str = "",
    forced_cyclic_ids: Optional[set] = None,
):
    """
    th1/th2 trade-off plot with heuristic points:
    1. th1/2 (*)  (fixed divide-by-2 baseline)
    2. th1/sqrt(k) (P)  (k-aware scaling: k=4 -> th1/2, k=5 -> th1/sqrt(5), ...)
    2. th1^2 (s)
    3. th1^1.5 (^)
    4. Online Sqrt (All) (D)
    5. Online Sqrt (LowConf-only update) (X)
    """
    # Default: prefix->cyclic, postfix->base (debias_pride.py와 동일)
    N = len(base_correct_list)
    if forced_cyclic_ids is not None:
        default_corrects = [
            cyclic_correct_list[i] if i in forced_cyclic_ids else base_correct_list[i]
            for i in range(N)
        ]
        default_acc = float(np.mean(np.asarray(default_corrects, dtype=np.float64)))
    else:
        default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    
    # Dense curve range (requested: 1..30)
    dense_th2_list = list(range(1, 31))
    
    # Real-world online points: th1 is online-quantile (past-only), th2 derived from th1 value
    def _online_point(th1_p: float, rule_func):
        return _run_online_th1_quantile_th2_from_th1_rule(
            default_conf=default_conf,
            mean_conf=mean_conf,
            base_correct=base_correct_list,
            cyclic_correct=cyclic_correct_list,
            probe2_correct=arr_probe2_correct,
            k=k,
            th1_percent=float(th1_p),
            th2_rule_from_th1_value=rule_func,
            forced_cyclic_ids=forced_cyclic_ids,
        )

    # Plot 1: Cost vs th2
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        # 1) Dense Curve
        costs = []
        for th2p in dense_th2_list:
            # REAL-WORLD online curve: sweep th2_percent while th1_percent fixed
            c, _ = _run_online_avggap_policy(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids,
            )
            costs.append(c)
        ax1.plot(dense_th2_list, costs, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.6)

        # 2) Heuristics Points
        # (A) th1 / 2  (fixed baseline)
        c_half, _, p_half = _online_point(th1p, lambda x: x / 2.0)
        ax1.scatter([p_half], [c_half], marker='*', s=120, color=color, edgecolors='black', zorder=6, label='th1/2' if idx==0 else "")

        # (A2) th1 / sqrt(k)  (k-aware)
        c_sqrtk, _, p_sqrtk = _online_point(th1p, lambda x, kk=k: x / math.sqrt(float(kk)))
        ax1.scatter([p_sqrtk], [c_sqrtk], marker='P', s=110, color=color, edgecolors='black', zorder=6, label='th1/sqrt(k)' if idx==0 else "")
        
        # (B) th1 ^ 2
        c_sq, _, p_sq = _online_point(th1p, lambda x: x ** 2)
        ax1.scatter([p_sq], [c_sq], marker='s', s=80, color=color, edgecolors='black', zorder=6, label='th1^2' if idx==0 else "")

        # (C) th1 ^ 1.5
        c_pow, _, p_pow = _online_point(th1p, lambda x: x ** 1.5)
        ax1.scatter([p_pow], [c_pow], marker='^', s=90, color=color, edgecolors='black', zorder=6, label='th1^1.5' if idx==0 else "")

        # (D) Online Sqrt
        c_sqrt, _, p_sqrt = _run_online_sqrt_policy(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p, forced_cyclic_ids=forced_cyclic_ids
        )
        ax1.scatter([p_sqrt], [c_sqrt], marker='D', s=80, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (All)' if idx==0 else "")

        # (E) Online Sqrt (LowConf-only update)
        c_sqrt_lc, _, p_sqrt_lc = _run_online_sqrt_policy_lowconf_update(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p, forced_cyclic_ids=forced_cyclic_ids
        )
        ax1.scatter([p_sqrt_lc], [c_sqrt_lc], marker='X', s=70, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (LowConf-only)' if idx==0 else "")

    ax1.set_xlabel("th2 (percentile, avg gap)", fontsize=11)
    ax1.set_ylabel("Computational Cost (× of default)", fontsize=11)
    ax1.set_title(f"{getattr(args, 'task', 'task')} {subject} — Cost vs th2 ({plot_tag})", fontsize=12)
    ax1.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax1.set_xticklabels([str(t) for t in [1, 5, 10, 15, 20, 25, 30]])
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.4)
    suffix = f"_{str(fname_tag).strip()}" if str(fname_tag).strip() else ""
    out_cost = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST{suffix}.png")
    os.makedirs(os.path.dirname(out_cost), exist_ok=True)
    fig1.tight_layout()
    fig1.savefig(out_cost, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: Δ Accuracy (%) vs Cost   (trade-off axis)
    fig2, ax2 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        # 1) Dense Curve (sweep th2 -> (cost, Δacc))
        costs_line = []
        delta_accs_line = []
        for th2p in dense_th2_list:
            c, a = _run_online_avggap_policy(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids,
            )
            costs_line.append(float(c))
            delta_accs_line.append((float(a) - float(default_acc)) * 100.0)
        ax2.plot(costs_line, delta_accs_line, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.5)

        # 2) Heuristics (points on cost axis)
        # (A) th1 / 2
        c_half, a_half, _ = _online_point(th1p, lambda x: x / 2.0)
        ax2.scatter([c_half], [(a_half-default_acc)*100], marker='*', s=120, color=color, edgecolors='black', zorder=6)

        # (A2) th1 / sqrt(k)
        c_sqrtk, a_sqrtk, _ = _online_point(th1p, lambda x, kk=k: x / math.sqrt(float(kk)))
        ax2.scatter([c_sqrtk], [(a_sqrtk-default_acc)*100], marker='P', s=110, color=color, edgecolors='black', zorder=6)
        
        # (B) th1 ^ 2
        c_sq, a_sq, _ = _online_point(th1p, lambda x: x ** 2)
        ax2.scatter([c_sq], [(a_sq-default_acc)*100], marker='s', s=80, color=color, edgecolors='black', zorder=6)

        # (C) th1 ^ 1.5
        c_pow, a_pow, _ = _online_point(th1p, lambda x: x ** 1.5)
        ax2.scatter([c_pow], [(a_pow-default_acc)*100], marker='^', s=90, color=color, edgecolors='black', zorder=6)

        # (D) Online Sqrt
        c_sqrt, a_sqrt, _ = _run_online_sqrt_policy(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        ax2.scatter([c_sqrt], [(a_sqrt-default_acc)*100], marker='D', s=80, color=color, edgecolors='black', zorder=6)

        # (E) Online Sqrt (LowConf-only update)
        c_sqrt_lc, a_sqrt_lc, _ = _run_online_sqrt_policy_lowconf_update(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        ax2.scatter([c_sqrt_lc], [(a_sqrt_lc-default_acc)*100], marker='X', s=70, color=color, edgecolors='black', zorder=6)

    ax2.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("Computational Cost (× of default)", fontsize=11)
    ax2.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax2.set_title(f"{getattr(args, 'task', 'task')} {subject} — Δ Accuracy vs Cost ({plot_tag})", fontsize=12)
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.4)
    out_delta = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_DELTA_ACC{suffix}.png")
    fig2.tight_layout()
    fig2.savefig(out_delta, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: Cost vs Δ Accuracy
    fig3, ax3 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        # Curve
        costs_line = []
        delta_accs_line = []
        for th2p in dense_th2_list:
            c, a = _run_online_avggap_policy(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids,
            )
            costs_line.append(c)
            delta_accs_line.append((a - default_acc) * 100.0)
        ax3.plot(costs_line, delta_accs_line, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.3)
        
        # Points
        # (A) th1 / 2
        c_h, a_h, p_h = _online_point(th1p, lambda x: x / 2.0)
        d_h = (a_h - default_acc) * 100.0
        ax3.scatter([c_h], [d_h], marker='*', s=120, color=color, edgecolors='black', zorder=6, label='th1/2' if idx==0 else "")

        # (A2) th1 / sqrt(k)
        c_hk, a_hk, p_hk = _online_point(th1p, lambda x, kk=k: x / math.sqrt(float(kk)))
        d_hk = (a_hk - default_acc) * 100.0
        ax3.scatter([c_hk], [d_hk], marker='P', s=110, color=color, edgecolors='black', zorder=6, label='th1/sqrt(k)' if idx==0 else "")

        # (B) th1 ^ 2
        c_s, a_s, p_s = _online_point(th1p, lambda x: x ** 2)
        d_s = (a_s - default_acc) * 100.0
        ax3.scatter([c_s], [d_s], marker='s', s=80, color=color, edgecolors='black', zorder=6, label='th1^2' if idx==0 else "")

        # (C) th1 ^ 1.5
        c_p, a_p, p_p = _online_point(th1p, lambda x: x ** 1.5)
        d_p = (a_p - default_acc) * 100.0
        ax3.scatter([c_p], [d_p], marker='^', s=90, color=color, edgecolors='black', zorder=6, label='th1^1.5' if idx==0 else "")

        # (D) Online Sqrt
        c_sqt, a_sqt, p_sqt = _run_online_sqrt_policy(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        d_sqt = (a_sqt - default_acc) * 100.0
        ax3.scatter([c_sqt], [d_sqt], marker='D', s=80, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (All)' if idx==0 else "")

        # (E) Online Sqrt (LowConf-only update)
        c_sqt_lc, a_sqt_lc, p_sqt_lc = _run_online_sqrt_policy_lowconf_update(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        d_sqt_lc = (a_sqt_lc - default_acc) * 100.0
        ax3.scatter([c_sqt_lc], [d_sqt_lc], marker='X', s=70, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (LowConf-only)' if idx==0 else "")

        # Log (verbose only; this block is very long)
        if bool(getattr(args, "verbose", False)):
            logger.info(_purple(f"==== TH2 online-point report (th1={int(th1p)}) ===="))
            logger.info(f"default              : cost=1.000, acc={default_acc:.4f}")
            logger.info(f"th1/2                : cost={c_h:.3f}, acc={a_h:.4f}, th2≈p{p_h:.1f}")
            logger.info(f"th1/sqrt(k)          : cost={c_hk:.3f}, acc={a_hk:.4f}, th2≈p{p_hk:.1f}")
            logger.info(f"th1^2                : cost={c_s:.3f}, acc={a_s:.4f}, th2≈p{p_s:.1f}")
            logger.info(f"th1^1.5              : cost={c_p:.3f}, acc={a_p:.4f}, th2≈p{p_p:.1f}")
            logger.info(f"Online Sqrt (All)    : cost={c_sqt:.3f}, acc={a_sqt:.4f}, th2≈p{p_sqt:.1f}")
            logger.info(f"Online Sqrt (LowConf): cost={c_sqt_lc:.3f}, acc={a_sqt_lc:.4f}, th2≈p{p_sqt_lc:.1f}")

    ax3.scatter([1.0], [0.0], marker='*', s=200, label='default', color='gray', zorder=5)
    ax3.set_xlabel("Computational Cost (× of default)", fontsize=11)
    ax3.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax3.set_title(f"{getattr(args, 'task', 'task')} {subject} — Trade-off (All Heuristics, {plot_tag})", fontsize=12)
    ax3.legend(loc='best', fontsize=9, ncol=2)
    ax3.grid(True, linestyle='--', alpha=0.4)
    out_trade = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST_vs_DELTA{suffix}.png")
    fig3.tight_layout()
    fig3.savefig(out_trade, bbox_inches="tight")
    plt.close(fig3)

    logger.info(_purple(f"th2 trade-off plots saved ({plot_tag}): {subject}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({
                f"plots/{subject}/th2_tradeoff_COST{suffix}": wandb.Image(out_cost),
                f"plots/{subject}/th2_tradeoff_DELTA_ACC{suffix}": wandb.Image(out_delta),
                f"plots/{subject}/th2_tradeoff_COST_vs_DELTA{suffix}": wandb.Image(out_trade),
            })
        except Exception:
            pass


def _parse_percent_value_list(v) -> List[float]:
    if v is None:
        return [30.0]
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                pass
        return out if len(out) > 0 else [30.0]
    if isinstance(v, str):
        s = v.strip()
        if "," in s:
            out = []
            for t in s.split(","):
                t = t.strip()
                if t == "":
                    continue
                try:
                    out.append(float(t))
                except Exception:
                    pass
            return out if len(out) > 0 else [30.0]
        try:
            return [float(s)]
        except Exception:
            return [30.0]
    return [30.0]


# =========================================================
# Curves: baseline (all methods)
# =========================================================
def _merge_curve_objs_over_runs(cobjs: List[dict]) -> Optional[dict]:
    """
    Average cost, acc, recall_std over multiple curve_objs (from n_runs).
    Returns one merged cobj with averaged numeric values.
    """
    if not cobjs or len(cobjs) == 0:
        return None
    if len(cobjs) == 1:
        return copy.deepcopy(cobjs[0])
    ref = cobjs[0]
    out = copy.deepcopy(ref)
    n = float(len(cobjs))
    # always
    for key in ["default", "cyclic", "full"]:
        if key in (out.get("always") or {}):
            costs = [float(c.get("always", {}).get(key, {}).get("cost", float("nan"))) for c in cobjs]
            accs = [float(c.get("always", {}).get(key, {}).get("acc", float("nan"))) for c in cobjs]
            costs = [x for x in costs if np.isfinite(x)]
            accs = [x for x in accs if np.isfinite(x)]
            if costs and accs:
                out["always"][key]["cost"] = float(np.mean(costs))
                out["always"][key]["acc"] = float(np.mean(accs))
    # cyclic_random_{fp}
    for k, v in list((out or {}).items()):
        if isinstance(k, str) and k.startswith("cyclic_random_") and not k.endswith("_recall_std"):
            if isinstance(v, dict) and "costs" in v and "accuracies" in v:
                costs = []
                accs = []
                for c in cobjs:
                    if k in c and isinstance(c[k], dict):
                        costs.append(float(c[k].get("costs", [float("nan")])[0]))
                        accs.append(float(c[k].get("accuracies", [float("nan")])[0]))
                costs = [x for x in costs if np.isfinite(x)]
                accs = [x for x in accs if np.isfinite(x)]
                if costs and accs:
                    out[k]["costs"] = [float(np.mean(costs))]
                    out[k]["accuracies"] = [float(np.mean(accs))]
    # *_recall_std
    for k in list((out or {}).keys()):
        if isinstance(k, str) and k.endswith("_recall_std") and isinstance(out.get(k), (int, float)):
            vals = [float(c.get(k, float("nan"))) for c in cobjs if k in c]
            vals = [x for x in vals if np.isfinite(x)]
            if vals:
                out[k] = float(np.mean(vals))
    # heuristic_points (group by (label, th1_p) when th1_p present, else by label only)
    hp_ref = ref.get("heuristic_points") or []
    if hp_ref:
        def _hp_key(h):
            lab = str(h.get("label")) if h.get("label") else ""
            th1_p = h.get("th1_p")
            return (lab, th1_p) if th1_p is not None else (lab, None)

        keys_seen = set()
        for h in hp_ref:
            if isinstance(h, dict) and h.get("label"):
                keys_seen.add(_hp_key(h))
        merged_hp = []
        for (lab, th1_p) in sorted(keys_seen, key=lambda k: (k[0], (k[1] if k[1] is not None else -1))):
            costs, accs, rstds, nbs, np2s, ncs = [], [], [], [], [], []
            marker_ref, color_ref = "o", "black"
            for c in cobjs:
                for h in (c.get("heuristic_points") or []):
                    if not isinstance(h, dict):
                        continue
                    if str(h.get("label")) != lab:
                        continue
                    if th1_p is not None and h.get("th1_p") != th1_p:
                        continue
                    if th1_p is None and h.get("th1_p") is not None:
                        continue
                    costs.append(float(h.get("cost", float("nan"))))
                    accs.append(float(h.get("acc", float("nan"))))
                    if "recall_std" in h:
                        rstds.append(float(h["recall_std"]))
                    if "n_base" in h:
                        nbs.append(int(h.get("n_base", 0)))
                    if "n_probe2" in h:
                        np2s.append(int(h.get("n_probe2", 0)))
                    if "n_cyclic" in h:
                        ncs.append(int(h.get("n_cyclic", 0)))
                    marker_ref = str(h.get("marker", "o"))
                    color_ref = str(h.get("color", "black"))
                    break
            costs = [x for x in costs if np.isfinite(x)]
            accs = [x for x in accs if np.isfinite(x)]
            if costs and accs:
                entry = {"label": lab, "cost": float(np.mean(costs)), "acc": float(np.mean(accs)), "marker": marker_ref, "color": color_ref}
                if th1_p is not None:
                    entry["th1_p"] = th1_p
                if rstds:
                    entry["recall_std"] = float(np.mean(rstds))
                if nbs:
                    entry["n_base"] = int(np.mean(nbs))
                if np2s:
                    entry["n_probe2"] = int(np.mean(np2s))
                if ncs:
                    entry["n_cyclic"] = int(np.mean(ncs))
                merged_hp.append(entry)
        out["heuristic_points"] = merged_hp
    return out


def _compute_curves_for_one_percentile(
    subject: str,
    tag: str,
    k: int,
    perm_list: List[Tuple[int, ...]],
    base_correct_list: List[bool],
    cyclic_correct_list: List[bool],
    full_correct_list: List[bool],
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    flip_trigger: np.ndarray,
    probe2_correct: np.ndarray,
    perc_value: float,
    full_enabled: bool = True,
    forced_cyclic_ids: Optional[set] = None,
    labels_idx: Optional[List[int]] = None,
    base_pred_idx: Optional[List[int]] = None,
    cyclic_pred_idx: Optional[List[int]] = None,
    probe2_pred_idx: Optional[List[int]] = None,
    full_pred_idx: Optional[List[int]] = None,
    cyclic_fractions: Optional[List[float]] = None,
    run_seed_offset: int = 0,
) -> dict:
    """
    REAL-WORLD online evaluation (no beta / no offline prefix).
    [Modified] If forced_cyclic_ids (Prefix) is active, force Cost=k and Acc=Cyclic.
    """
    N = len(base_correct_list)
    if N == 0:
        return {}

    perc01 = float(max(0.0, min(100.0, perc_value))) / 100.0

    C_cyc = float(k)
    C_full = float(len(perm_list)) if full_enabled else float("nan")

    # Default ensemble: prefix -> cyclic, postfix -> base (debias_pride.py와 동일)
    if forced_cyclic_ids is not None:
        default_corrects = [
            cyclic_correct_list[i] if i in forced_cyclic_ids else base_correct_list[i]
            for i in range(N)
        ]
        default_acc = float(np.mean(np.asarray(default_corrects, dtype=np.float64)))
    else:
        default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    cyclic_acc_always = float(np.mean(np.asarray(cyclic_correct_list, dtype=np.float64)))
    full_acc_always = float(np.mean(np.asarray(full_correct_list, dtype=np.float64))) if full_enabled and len(full_correct_list) == N else float("nan")

    # 1) switch policies (REAL-WORLD online)
    total_cost_sc = 0.0
    corrects_sc = 0
    total_cost_sf = 0.0
    corrects_sf = 0
    past_gaps: List[float] = []

    for i in range(N):
        # 1. Calculate Threshold based on PAST data
        if len(past_gaps) > 0:
            thresh = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), perc01))
        else:
            thresh = float("-inf")
        
        # 2. Check if current sample is in Prefix (Investment Phase)
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)
        
        # 3. Policy Decision (Ambiguous?)
        # Even if forced, we calculate this to simulate what the policy *would* have thought
        amb = (float(default_conf[i]) < thresh)

        # -----------------------------------------------------
        # Logic: switch_cyclic
        # -----------------------------------------------------
        if is_forced:
            # [Prefix] 무조건 Cyclic 수행 (Cost=k, Acc=Cyclic)
            c_step_sc = C_cyc
            corrects_sc += 1 if cyclic_correct_list[i] else 0
        else:
            # [Postfix] Policy 판단에 따름
            if amb:
                c_step_sc = C_cyc
                corrects_sc += 1 if cyclic_correct_list[i] else 0
            else:
                c_step_sc = 1.0
                corrects_sc += 1 if base_correct_list[i] else 0

        # -----------------------------------------------------
        # Logic: switch_full (if enabled)
        # -----------------------------------------------------
        if full_enabled and len(full_correct_list) == N:
            if is_forced:
                # [Prefix] Full 수행 (Cost=Full/Cyclic, Acc=Full)
                c_step_sf = C_full 
                corrects_sf += 1 if full_correct_list[i] else 0
            else:
                # [Postfix]
                if amb:
                    c_step_sf = C_full
                    corrects_sf += 1 if full_correct_list[i] else 0
                else:
                    c_step_sf = 1.0
                    corrects_sf += 1 if base_correct_list[i] else 0
        else:
            c_step_sf = 0.0

        # Update History (Observe the gap regardless of decision)
        total_cost_sc += float(c_step_sc)
        if full_enabled and len(full_correct_list) == N:
            total_cost_sf += float(c_step_sf)
            
        past_gaps.append(float(default_conf[i]))

    switch_cyclic_cost = total_cost_sc / float(N)
    switch_cyclic_acc = corrects_sc / float(N)
    switch_full_cost = (total_cost_sf / float(N)) if (full_enabled and len(full_correct_list) == N) else float("nan")
    switch_full_acc = (corrects_sf / float(N)) if (full_enabled and len(full_correct_list) == N) else float("nan")

    # 2) ours_top2flip / ours_avggap (REAL-WORLD online) + stats
    _, _, top2_stats = _run_online_top2flip_policy_with_stats(
        default_conf=default_conf,
        flip_trigger=flip_trigger,
        base_correct=base_correct_list,
        cyclic_correct=cyclic_correct_list,
        probe2_correct=probe2_correct,
        k=k,
        th1_percent=perc_value,
        offline_prefix_n=0,
        forced_cyclic_ids=forced_cyclic_ids,
    )
    c_top2, a_top2 = _run_online_top2flip_policy(
        default_conf=default_conf,
        flip_trigger=flip_trigger,
        base_correct=base_correct_list,
        cyclic_correct=cyclic_correct_list,
        probe2_correct=probe2_correct,
        k=k,
        th1_percent=perc_value,
        offline_prefix_n=0,
        forced_cyclic_ids=forced_cyclic_ids,
    )
    _, _, sc_stats = _run_online_switch_cyclic_with_stats(
        default_conf=default_conf,
        base_correct=base_correct_list,
        cyclic_correct=cyclic_correct_list,
        k=k,
        th1_percent=perc_value,
        offline_prefix_n=0,
        forced_cyclic_ids=forced_cyclic_ids,
    )
    c_avg, a_avg, avg_stats = _run_online_avggap_policy_with_stats(
        default_conf=default_conf,
        mean_conf=mean_conf,
        base_correct=base_correct_list,
        cyclic_correct=cyclic_correct_list,
        probe2_correct=probe2_correct,
        k=k,
        th1_percent=perc_value,
        th2_percent=perc_value,
        offline_prefix_n=0,
        forced_cyclic_ids=forced_cyclic_ids,
    )

    # Cyclic random fraction (for three-curves plot)
    # Default+PRIDE: alpha<100 → prefix cyclic, postfix base(보정). alpha>=100 → Cyclic과 동일(원본).
    cyclic_fractions = cyclic_fractions or [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    seed_base = _stable_u32_seed(str(subject), int(run_seed_offset))
    cyclic_random_costs: Dict[str, float] = {}
    cyclic_random_accs: Dict[str, float] = {}
    prefix_ids_set = set(int(x) for x in forced_cyclic_ids) if forced_cyclic_ids else set()
    for fp in cyclic_fractions:
        fp_f = float(fp)
        if prefix_ids_set and float(fp_f) == float(perc_value):
            c_r, a_r = _run_prefix_cyclic_postfix_base(
                base_correct_list, cyclic_correct_list, k, prefix_ids_set
            )
        else:
            fp_seed_off = int(fp_f) if float(fp_f).is_integer() else int(round(fp_f * 1000.0))
            c_r, a_r = _run_cyclic_random_fraction(
                base_correct_list, cyclic_correct_list, k, fp_f, seed_base + fp_seed_off
            )
        cyclic_random_costs[f"cyclic_random_{fp}"] = float(c_r)
        cyclic_random_accs[f"cyclic_random_{fp}"] = float(a_r)

    # Prefix overhead accounting for "always" ensembles
    default_cost_always = 1.0
    if forced_cyclic_ids is not None and N > 0:
        m = int(len(forced_cyclic_ids))
        if m > 0:
            # Default ensemble also paid 'k' for the prefix samples to estimate prior
            default_cost_always = 1.0 + (float(m) * (float(k) - 1.0)) / float(N)

    curve_obj = {
        "subject": subject,
        "tag": str(tag),
        "k": int(k),
        "percentile": float(perc_value),
        "n_samples": int(N),
        "default_accuracy": float(default_acc),

        "always": {
            "default": {"cost": float(default_cost_always), "acc": float(default_acc)},
            "cyclic": {"cost": float(C_cyc), "acc": float(cyclic_acc_always)},
        },

        "cyclic": {"costs": [float(C_cyc)], "accuracies": [float(cyclic_acc_always)]},
        **{key: {"costs": [cyclic_random_costs[key]], "accuracies": [cyclic_random_accs[key]]} for key in cyclic_random_costs},
        "switch_cyclic": {"costs": [float(switch_cyclic_cost)], "accuracies": [float(switch_cyclic_acc)], "stats": dict(sc_stats)},
        "ours_top2flip": {"costs": [float(c_top2)], "accuracies": [float(a_top2)], "stats": dict(top2_stats)},
        "ours_avggap": {"costs": [float(c_avg)], "accuracies": [float(a_avg)]},
        "ours_avggap_stats": dict(avg_stats),
    }
    # Optional: add recall_std when labels_idx and preds available
    if labels_idx is not None and base_pred_idx is not None and cyclic_pred_idx is not None and probe2_pred_idx is not None:
        try:
            _, _, preds_sc = _run_online_switch_cyclic_with_preds(
                default_conf, base_pred_idx, cyclic_pred_idx, labels_idx, k, perc_value, 0, forced_cyclic_ids
            )
            _, _, preds_top2 = _run_online_top2flip_policy_with_preds(
                default_conf, flip_trigger, base_pred_idx, cyclic_pred_idx, probe2_pred_idx, labels_idx, k, perc_value, 0, forced_cyclic_ids
            )
            _, _, preds_avg = _run_online_avggap_policy_with_preds(
                default_conf, mean_conf, base_pred_idx, cyclic_pred_idx, probe2_pred_idx, labels_idx, k, perc_value, perc_value, 0, forced_cyclic_ids
            )
            curve_obj["switch_cyclic_recall_std"] = float(_recall_std(labels_idx, preds_sc, k))
            curve_obj["ours_top2flip_recall_std"] = float(_recall_std(labels_idx, preds_top2, k))
            curve_obj["ours_avggap_recall_std"] = float(_recall_std(labels_idx, preds_avg, k))
            # Default: prefix->cyclic, postfix->base (debias_pride.py와 동일)
            default_pred_idx = (
                [cyclic_pred_idx[i] if i in forced_cyclic_ids else base_pred_idx[i] for i in range(N)]
                if forced_cyclic_ids is not None
                else base_pred_idx
            )
            curve_obj["default_recall_std"] = float(_recall_std(labels_idx, default_pred_idx, k))
            curve_obj["cyclic_recall_std"] = float(_recall_std(labels_idx, cyclic_pred_idx, k))
            for fp in cyclic_fractions:
                fp_f = float(fp)
                if forced_cyclic_ids is not None and float(fp_f) == float(perc_value):
                    # Default+PRIDE: prefix=cyclic, postfix=base (same as default_pred_idx)
                    preds_r = default_pred_idx
                else:
                    fp_seed_off = int(fp_f) if float(fp_f).is_integer() else int(round(fp_f * 1000.0))
                    _, _, preds_r = _run_cyclic_random_fraction_with_preds(
                        base_pred_idx, cyclic_pred_idx, labels_idx, k, fp_f, seed_base + fp_seed_off
                    )  # seed_base already includes run_seed_offset
                curve_obj[f"cyclic_random_{fp}_recall_std"] = float(_recall_std(labels_idx, preds_r, k))
            if full_enabled and full_pred_idx is not None and len(full_pred_idx) == len(labels_idx):
                curve_obj["full_recall_std"] = float(_recall_std(labels_idx, full_pred_idx, k))
        except Exception:
            pass
    if full_enabled:
        curve_obj["always"]["full"] = {"cost": float(C_full), "acc": float(full_acc_always)}
        curve_obj["full"] = {"costs": [float(C_full)], "accuracies": [float(full_acc_always)]}
        if full_enabled and len(full_correct_list) == N:
            curve_obj["switch_full"] = {"costs": [float(switch_full_cost)], "accuracies": [float(switch_full_acc)]}
    return curve_obj


def _log_baseline_report(curve_obj: dict):
    """
    BASELINE은 풀로 찍고,
    PRIDE_FREE는 (아래 main에서) 한 줄만 찍는다.
    """
    p = curve_obj.get("percentile")
    logger.info(_purple(f"==== BASELINE Derived policy report (REAL-WORLD online, p={p}) ===="))

    always = curve_obj.get("always", {})
    def _recall_str(obj):
        return f", recall_std={obj:.4f}" if isinstance(obj, (int, float)) else ""
    logger.info(f"BASELINE default(ensemble) : cost={always['default']['cost']:.3f}, acc={always['default']['acc']:.4f}{_recall_str(curve_obj.get('default_recall_std'))}")
    logger.info(f"BASELINE cyclic(ensemble)  : cost={always['cyclic']['cost']:.3f}, acc={always['cyclic']['acc']:.4f}{_recall_str(curve_obj.get('cyclic_recall_std'))}")
    if "full" in always:
        logger.info(f"BASELINE full(ensemble)    : cost={always['full']['cost']:.3f}, acc={always['full']['acc']:.4f}{_recall_str(curve_obj.get('full_recall_std'))}")
    else:
        logger.info("BASELINE full(ensemble)    : (disabled)")

    for key in ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]:
        if key in curve_obj:
            c0 = float(curve_obj[key]["costs"][0])
            a0 = float(curve_obj[key]["accuracies"][0])
            st = curve_obj.get("ours_avggap_stats", {}) if key == "ours_avggap" else curve_obj[key].get("stats", {})
            if not isinstance(st, dict):
                st = {}
            nb = int(st.get("n_base", 0))
            np2 = int(st.get("n_probe2", 0))
            nc = int(st.get("n_cyclic", 0))
            extra = f", n_base={nb}, n_probe2={np2}, n_cyclic={nc}"
            recall_key = f"{key}_recall_std"
            rstd = curve_obj.get(recall_key)
            if isinstance(rstd, (int, float)):
                extra += f", recall_std={rstd:.4f}"
            logger.info(f"BASELINE {key:<12} : cost={c0:.3f}, acc={a0:.4f}{extra}")

    # Cyclic random fractions (plot에 쓰이는 것과 동일)
    def _log_cyclic_random(obj: dict, prefix: str):
        keys = [
            k for k in (obj or {}).keys()
            if isinstance(k, str) and k.startswith("cyclic_random_") and not k.endswith("_recall_std")
        ]
        def _k_to_float(s: str) -> float:
            try:
                return float(s.replace("cyclic_random_", ""))
            except Exception:
                return float("inf")
        keys = sorted(keys, key=_k_to_float)
        if keys:
            logger.info(_purple(f"---- Cyclic random fractions (plot과 동일) [{prefix}] ----"))
            for k in keys:
                if k in obj and isinstance(obj[k], dict) and "costs" in obj[k] and "accuracies" in obj[k]:
                    c = float(obj[k]["costs"][0])
                    a = float(obj[k]["accuracies"][0])
                    rstd = obj.get(f"{k}_recall_std")
                    extra = f", recall_std={rstd:.4f}" if isinstance(rstd, (int, float)) else ""
                    logger.info(f"BASELINE {k:<16}: cost={c:.3f}, acc={a:.4f}{extra}")
    _log_cyclic_random(curve_obj, "BASELINE")


def _log_named_report(name: str, curve_obj: dict):
    """Same format as baseline report, but with custom header prefix."""
    p = curve_obj.get("percentile")
    logger.info(_purple(f"==== {name} Derived policy report (REAL-WORLD online, p={p}) ===="))

    always = curve_obj.get("always", {})
    def _recall_str(obj):
        return f", recall_std={obj:.4f}" if isinstance(obj, (int, float)) else ""
    if "default" in always:
        logger.info(f"{name} default(ensemble) : cost={always['default']['cost']:.3f}, acc={always['default']['acc']:.4f}{_recall_str(curve_obj.get('default_recall_std'))}")
    if "cyclic" in always:
        logger.info(f"{name} cyclic(ensemble)  : cost={always['cyclic']['cost']:.3f}, acc={always['cyclic']['acc']:.4f}{_recall_str(curve_obj.get('cyclic_recall_std'))}")
    if "full" in always:
        logger.info(f"{name} full(ensemble)    : cost={always['full']['cost']:.3f}, acc={always['full']['acc']:.4f}{_recall_str(curve_obj.get('full_recall_std'))}")

    for key in ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]:
        if key in curve_obj:
            c0 = float(curve_obj[key]["costs"][0])
            a0 = float(curve_obj[key]["accuracies"][0])
            st = curve_obj.get("ours_avggap_stats", {}) if key == "ours_avggap" else curve_obj[key].get("stats", {})
            if not isinstance(st, dict):
                st = {}
            nb, np2, nc = int(st.get("n_base", 0)), int(st.get("n_probe2", 0)), int(st.get("n_cyclic", 0))
            extra = f", n_base={nb}, n_probe2={np2}, n_cyclic={nc}"
            rstd = curve_obj.get(f"{key}_recall_std")
            if isinstance(rstd, (int, float)):
                extra += f", recall_std={rstd:.4f}"
            logger.info(f"{name} {key:<12} : cost={c0:.3f}, acc={a0:.4f}{extra}")

    # Cyclic random fractions (plot Default+PRIDE curve와 동일)
    fracs = sorted([int(k.replace("cyclic_random_", "")) for k in (curve_obj or {}).keys()
                    if isinstance(k, str) and k.startswith("cyclic_random_") and not k.endswith("_recall_std")],
                   key=lambda x: x)
    if fracs:
        logger.info(_purple(f"---- Cyclic random fractions (plot Default+PRIDE와 동일) [{name}] ----"))
        for fp in fracs:
            k = f"cyclic_random_{fp}"
            if k in curve_obj and isinstance(curve_obj[k], dict) and "costs" in curve_obj[k] and "accuracies" in curve_obj[k]:
                c = float(curve_obj[k]["costs"][0])
                a = float(curve_obj[k]["accuracies"][0])
                rstd = curve_obj.get(f"{k}_recall_std")
                extra = f", recall_std={rstd:.4f}" if isinstance(rstd, (int, float)) else ""
                logger.info(f"{name} cyclic_{fp}%      : cost={c:.3f}, acc={a:.4f}{extra}")


def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    hf_logging.set_verbosity_error()

    args = parse_arguments()
    if len(getattr(args, "eval_names", [])) == 0:
        return

    # -------- W&B init (optional) --------
    wandb_run = None
    wandb_ok = False
    if bool(getattr(args, "wandb", False)):
        try:
            import wandb
            wandb_ok = True
            project = getattr(args, "wandb_project", None) or "eval_clm"
            run_name = getattr(args, "wandb_run_name", None) or f"{getattr(args,'model_name','model')}-{args.eval_names[0]}"
            entity = getattr(args, "wandb_entity", None) or "capde"
            cfg = {
                "pretrained_model_path": getattr(args, "pretrained_model_path", None),
                "model_name": getattr(args, "model_name", None),
                "eval_names": getattr(args, "eval_names", None),
                "option_id_set": getattr(args, "option_id_set", None),
                "ours_low_conf_percent": getattr(args, "ours_low_conf_percent", None),
            }
            wandb_run = wandb.init(project=project, entity=entity, name=run_name, config=cfg)
            logger.info(_blue(f"W&B init ok: project={project}, entity={entity}, name={run_name}"))
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            wandb_run = None
            wandb_ok = False

    # -------- DEVELOP MODE: generate dummy curve points only --------
    if bool(getattr(args, "develop", False)):
        try:
            # Use the same fractions as real plotting
            cyclic_fracs = [int(x) for x in _parse_percent_value_list(getattr(args, "plot_cyclic_fractions", "0,10,20,30,40,50,60,70,80,90,100")) if 0 <= int(x) <= 100]
            pride_fracs = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_ours_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0]
        except Exception:
            cyclic_fracs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            pride_fracs = [0.5, 1.0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        logger.info(_orange("[develop] Skipping model/data eval. Writing dummy three-curves plots/points."))

        # Always-random seed per process execution (time/pid/wandb_run.id mixed)
        try:
            wid = (wandb_run.id if wandb_run is not None else "no_wandb")
            wid_u32 = int(zlib.adler32(str(wid).encode("utf-8"))) & 0xFFFFFFFF
        except Exception:
            wid_u32 = 0
        base_seed = (int(time.time_ns()) ^ (int(os.getpid()) << 16) ^ int(wid_u32)) & 0xFFFFFFFF

        for eval_name in (getattr(args, "eval_names", None) or []):
            try:
                parts = str(eval_name).split(",")
                task = str(parts[0]).strip()
                num_few_shot = int(parts[1]) if len(parts) > 1 and str(parts[1]).strip() else 0
                args.task = task
                args.num_few_shot = num_few_shot

                out_dir = f"results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full"
                if getattr(args, "option_id_set", None):
                    out_dir += f"_id-{args.option_id_set}"
                os.makedirs(out_dir, exist_ok=True)

                # Randomness per eval_name, mixed with base_seed
                seed = (int(base_seed) ^ (int(zlib.adler32(str(eval_name).encode("utf-8"))) & 0xFFFFFFFF)) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)

                # Build a single "subject" curve object that contains cyclic_random_{fp} keys.
                # (three-curves plotting expects these keys inside one list element)
                cobj_base = {}
                k = 4.0  # pretend 4-choice
                for fp in cyclic_fracs:
                    frac = float(fp) / 100.0
                    cost = 1.0 + frac * (k - 1.0) + float(rng.normal(0.0, 0.02))
                    acc = 0.45 + 0.25 * frac + float(rng.normal(0.0, 0.01))
                    acc = float(np.clip(acc, 0.0, 1.0))
                    rstd = 0.20 - 0.10 * frac + float(rng.normal(0.0, 0.005))
                    rstd = float(np.clip(rstd, 0.0, 1.0))
                    cobj_base[f"cyclic_random_{fp}"] = {"costs": [float(cost)], "accuracies": [float(acc)]}
                    cobj_base[f"cyclic_random_{fp}_recall_std"] = float(rstd)

                derived_records_by_p = {}
                for p in pride_fracs:
                    frac = float(p) / 100.0
                    # Heuristic point for OURS at each p
                    ours_cost = 1.0 + frac * 1.5 + float(rng.normal(0.0, 0.02))
                    ours_acc = 0.50 + 0.20 * frac + float(rng.normal(0.0, 0.01))
                    ours_acc = float(np.clip(ours_acc, 0.0, 1.0))
                    ours_rstd = 0.18 - 0.08 * frac + float(rng.normal(0.0, 0.006))
                    ours_rstd = float(np.clip(ours_rstd, 0.0, 1.0))

                    cobj = dict(cobj_base)
                    cobj["heuristic_points"] = [{
                        "label": "th1/2",
                        "cost": float(ours_cost),
                        "acc": float(ours_acc),
                        "recall_std": float(ours_rstd),
                        "n_base": int(800 + rng.integers(0, 50)),
                        "n_probe2": int(150 + rng.integers(0, 50)),
                        "n_cyclic": int(50 + rng.integers(0, 50)),
                    }]
                    derived_records_by_p[float(p)] = [cobj]  # 1 "subject"

                pride_prefix = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_prefix_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0] or [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
                derived_records_pride_by_p = {}
                derived_records_pride_by_alpha = {}
                for alpha in pride_prefix:
                    cobj_pr = {}
                    for p in pride_fracs:
                        frac = float(p) / 100.0
                        pride_cost = 1.0 + frac * 2.2 + float(rng.normal(0.0, 0.02))
                        pride_acc = 0.52 + 0.18 * frac + float(rng.normal(0.0, 0.01))
                        pride_acc = float(np.clip(pride_acc, 0.0, 1.0))
                        pride_rstd = 0.16 - 0.07 * frac + float(rng.normal(0.0, 0.006))
                        pride_rstd = float(np.clip(pride_rstd, 0.0, 1.0))
                        key = f"cyclic_random_{p}"
                        cobj_pr[key] = {"costs": [float(pride_cost)], "accuracies": [float(pride_acc)]}
                        cobj_pr[f"{key}_recall_std"] = float(pride_rstd)
                        cobj_pr.setdefault("heuristic_points", []).append({
                            "label": "th1/2", "th1_p": p, "cost": float(pride_cost),
                            "acc": float(pride_acc), "recall_std": float(pride_rstd),
                            "n_base": 800, "n_probe2": 150, "n_cyclic": 50,
                        })
                        sqrt_cost = 1.0 + frac * 1.9 + float(rng.normal(0.0, 0.02))
                        sqrt_acc = 0.53 + 0.19 * frac + float(rng.normal(0.0, 0.01))
                        sqrt_rstd = 0.15 - 0.06 * frac + float(rng.normal(0.0, 0.006))
                        cobj_pr["heuristic_points"].append({
                            "label": "online_sqrt_all", "th1_p": p, "cost": float(sqrt_cost),
                            "acc": float(np.clip(sqrt_acc, 0.0, 1.0)), "recall_std": float(np.clip(sqrt_rstd, 0.0, 1.0)),
                            "n_base": 800, "n_probe2": 150, "n_cyclic": 50,
                        })
                    derived_records_pride_by_alpha[alpha] = [cobj_pr]

                _plot_three_curves_acc_recall_std(
                    derived_records_by_p,
                    derived_records_pride_by_p,
                    derived_records_pride_by_alpha,
                    out_dir,
                    args.task,
                    cyclic_fractions=cyclic_fracs,
                    pride_ours_fractions=pride_fracs,
                    pride_prefix_list=pride_prefix,
                    wandb_ok=wandb_ok,
                    wandb_run=wandb_run,
                )
            except Exception as e:
                logger.warning(f"[develop] Failed to write dummy plots for eval_name='{eval_name}': {e}")

        # Finish W&B early (since we skip the rest of main)
        if wandb_ok and wandb_run is not None:
            try:
                import wandb
                logger.info(_blue("W&B: syncing and finishing run (develop)..."))
                wandb.finish()
                time.sleep(2)
            except Exception:
                pass
        return

    # -------- Tokenizer / Model --------
    toker = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
        cache_dir=getattr(args, "cache_dir", None),
    )

    use_bf16 = bool(torch.cuda.is_available()) and bool(torch.cuda.is_bf16_supported())
    config = AutoConfig.from_pretrained(
        args.pretrained_model_path,
        cache_dir=getattr(args, "cache_dir", None),
    )
    model_type = getattr(config, "model_type", "").lower()
    if model_type in ("t5", "mt5", "umt5"):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.pretrained_model_path,
            device_map='auto',
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            cache_dir=getattr(args, "cache_dir", None),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            device_map='auto',
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            cache_dir=getattr(args, "cache_dir", None),
        )

    logging_cuda_memory_usage()

    for eval_name in args.eval_names[::1]:
        (subjects, prepare_few_shot_samples,
         prepare_eval_samples, prepare_eval_fn) = prepare_eval(args, eval_name)

        # =========================
        # Aggregate (MMLU-style) summary over subjects
        # =========================
        eval_acc_records: List[dict] = []  # [{'subject':str,'corrects':int,'total':int,'acc':float}]
        derived_records_by_p: Dict[float, List[dict]] = {}  # p -> list of curve_obj (per subject)
        derived_records_pride_by_p: Dict[float, List[dict]] = {}  # legacy: p->list (Default+PRIDE 단일 alpha용)
        derived_records_pride_by_alpha: Dict[float, List[dict]] = {}  # alpha(0.5,1.0,2,5,...) -> list of PRIDE curve_obj
        pride_recall_std_records: List[dict] = []  # [{'subject':str,'rstd':float,'m':int,'N':int}]
        recall_std_vs_p_records: List[dict] = []  # [{'subject':str,'p':float,'method':str,'kind':str,'rstd':float}]
        n_runs = max(1, int(getattr(args, "n_runs", 1)))
        skip_per_subject_plots = (args.task == "mmlu" and len(subjects) > 1)

        # 논문 테이블용 Base T/F 기준 트랜지션 기록 (Cyclic & Full)
        transition_records_cyclic: List[dict] = []
        transition_records_full: List[dict] = []
        # Default+PRIDE, Ours+PRIDE, Ours (per perc 2~100) - 논문 Experiments/Analysis용
        transition_records_default_pride_by_p: Dict[float, List[dict]] = {}
        transition_records_ours_pride_by_p: Dict[float, List[dict]] = {}
        transition_records_ours_by_p: Dict[float, List[dict]] = {}

        def _make_transition_record_from_preds(base_correct, pred_idx, labels_idx, conf_arr, subject):
            """base_correct: List[bool], pred_idx: List[int], labels_idx: List[int], conf_arr: np.ndarray"""
            N = len(base_correct)
            pred_correct = [int(pred_idx[i]) == int(labels_idx[i]) for i in range(N)]
            base_t_gaps, base_f_gaps = [], []
            t_to_f, f_to_t = 0, 0
            conf_flat = np.asarray(conf_arr, dtype=np.float64).ravel()
            for i in range(N):
                c = float(conf_flat[i]) if i < len(conf_flat) else 0.0
                if base_correct[i]:
                    base_t_gaps.append(c)
                    if not pred_correct[i]:
                        t_to_f += 1
                else:
                    base_f_gaps.append(c)
                    if pred_correct[i]:
                        f_to_t += 1
            return {"subject": str(subject), "base_t_gaps": base_t_gaps, "base_f_gaps": base_f_gaps,
                    "t_to_f_count": t_to_f, "f_to_t_count": f_to_t}

        for subject in subjects[::1]:
            # n_runs > 1이면 run별 다른 seed로 평가 (few-shot, run_seed 고정 해제)
            run_indices = list(range(n_runs))
            for run_idx in run_indices:
                use_run_suffix = (n_runs > 1)
                cached_path = (f'{args.save_path}/{subject}_run{run_idx}.jsonl' if use_run_suffix
                               else f'{args.save_path}/{subject}.jsonl')
                use_cached = (not bool(getattr(args, 'force', False))) and os.path.exists(cached_path)

                few_shot_seed = run_idx if use_run_suffix else None
                run_seed = run_idx if use_run_suffix else None

                logger.info(_blue(f"Preparing: {subject}" + (f" [run {run_idx+1}/{n_runs}]" if use_run_suffix else "")))
                few_shot_samples = prepare_few_shot_samples(subject, few_shot_seed=few_shot_seed)
                eval_samples = prepare_eval_samples(subject)
                # n_runs > 1: 매 run마다 데이터 순서를 다르게 shuffle (재현 가능한 시드)
                if n_runs > 1:
                    shuffler = random.Random(int(run_idx) + 42)
                    shuffler.shuffle(eval_samples)
                eval_fn = prepare_eval_fn(model, toker, few_shot_samples)

                if use_cached:
                    logger.info(_blue(f"Using cached results: {cached_path}"))
                    results = _read_results_file(cached_path) or []
                else:
                    logger.info(_blue(f"Run started: {subject}" + (f" [run {run_idx+1}/{n_runs}]" if use_run_suffix else "")))
                    max_samples = 100 if bool(getattr(args, 'test', False)) else None
                    n_threads = torch.cuda.device_count()
                    n_threads = max(1, int(n_threads)) if 'falcon' not in args.pretrained_model_path else 1
                    results = eval_all_samples(
                        eval_fn, eval_samples,
                        name=f'{args.task},{args.num_few_shot},{args.setting},{subject}' + (f',run{run_idx}' if use_run_suffix else ''),
                        threads=n_threads,
                        max_num_samples=max_samples,
                        run_seed=run_seed,
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

                        # If results contain only k rotations (e.g., k>=5 full-permutation disabled),
                        # aggregate with rotations instead of factorial permutations.
                        probs_len0 = None
                        try:
                            probs_len0 = len(results[0]['data'].get('probs', []))
                        except Exception:
                            probs_len0 = None

                        if args.setting in ['perm', 'full']:
                            if probs_len0 == k:
                                logger.info(_orange(f"[Auto] Full permutation disabled or not provided (k={k}). Using cyclic rotations for aggregation."))
                                perm_list = _rotations(k)
                            else:
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

                        # aggregate: (micro uses correct/total; macro uses per-subject acc mean)
                        if total > 0:
                            eval_acc_records.append({
                                "subject": str(subject),
                                "corrects": int(corrects),
                                "total": int(total),
                                "acc": float(acc),
                            })
                    else:
                        metrics = {'type': 'metric', 'data': {}}
                        metrics['data']['accuracy'] = get_accuracy(results)
                        metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)

                        # aggregate (base/noid/shuffle etc)
                        total_b = 0
                        corrects_b = 0
                        for r in results:
                            if r.get("type") != "result":
                                continue
                            data = r.get("data", {}) or {}
                            corr = data.get("correct", None)
                            if corr is None:
                                if ("sampled" in data) and ("ideal" in data):
                                    corr = (data.get("sampled") == data.get("ideal"))
                                else:
                                    continue
                            total_b += 1
                            corrects_b += 1 if bool(corr) else 0
                        if total_b > 0:
                            eval_acc_records.append({
                                "subject": str(subject),
                                "corrects": int(corrects_b),
                                "total": int(total_b),
                                "acc": float(corrects_b) / float(total_b),
                            })

                logger.info(_orange(f"Run completed: {subject}" + (f" [run {run_idx+1}/{n_runs}]" if use_run_suffix else "")))

                if not use_cached:
                    save_results(cached_path, results, metrics)
                    logger.info(f"Results saved: {subject}" + (f" [run {run_idx}]" if use_run_suffix else ""))

                # =========================================================
                # Derived policies & PRIDE_FREE (full or cyclic for MMLU aggregate plots)
                # =========================================================
                if args.setting in ('full', 'cyclic') and len(results) > 0:
                    try:
                        if getattr(args, 'option_id_set', None):
                            option_ids = list(args.option_id_set)
                        else:
                            k_guess = len(results[0]['data']['options'])
                            option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                        k = len(option_ids)

                        # Determine whether full permutations exist in cached results.
                        probs_len0 = None
                        try:
                            probs_len0 = len(results[0]['data'].get('probs', []))
                        except Exception:
                            probs_len0 = None

                        full_enabled = (probs_len0 is not None and probs_len0 == math.factorial(k))
                        if full_enabled:
                            from itertools import permutations
                            perm_list = list(sorted(permutations(range(k))))
                        else:
                            # fallback to cyclic rotations only
                            logger.info(_orange(f"[Auto] k={k} full permutations not available. Running derived policies with cyclic rotations only."))
                            perm_list = _rotations(k)
                        identity_idx = perm_list.index(tuple(range(k)))

                        cyclic_indices = [
                            perm_list.index(tuple((i + s) % k for i in range(k)))
                            for s in range(k)
                        ]
                        cyc_perms = [tuple((i + s) % k for i in range(k)) for s in range(k)]

                        # ---------- collect per-sample raw probs ----------
                        per_sample_probs = []
                        ideals = []

                        # ---------- derived correctness lists (baseline) ----------
                        base_correct_list = []
                        cyclic_correct_list = []
                        full_correct_list = []
                        full_pred_idx_list = []  # argmax(agg_full) for recall_std when full_enabled

                        base_probs_list = []  # identity row (letter-space)
                        base_pred_idx_list = []     # argmax(base_probs) as index
                        cyclic_pred_idx_list = []   # argmax(agg_cyc) as index
                        probe2_pred_idx_list = []   # argmax(mean_probs) as index

                        cyclic_results = []
                        base_results = []

                        full_total = 0
                        full_corrects = 0
                        cyclic_total = 0
                        cyclic_corrects = 0

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

                            # cyclic (k rotations)
                            cyc_probs = [probs_seq_np[idx] for idx in cyclic_indices]
                            cyclic_results.append({
                                'type': 'result',
                                'data': {
                                    'idx': data['idx'],
                                    'prompt': data.get('prompt'),
                                    'options': data['options'],
                                    'probs': [cp.tolist() for cp in cyc_probs],
                                    'ideal': data['ideal'],
                                },
                            })
                            agg_cyc = _aggregate_probs_over_permutations([cp.tolist() for cp in cyc_probs], cyc_perms, k)
                            pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                            cyclic_pred_idx_list.append(int(np.argmax(agg_cyc)))
                            corr_cyc = (pred_cyc == data['ideal'])
                            cyclic_correct_list.append(corr_cyc)
                            cyclic_corrects += 1 if corr_cyc else 0
                            cyclic_total += 1

                            # base (identity only)
                            base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
                            base_probs_list.append(base_probs)
                            pred_base = option_ids[int(np.argmax(base_probs))]
                            base_pred_idx_list.append(int(np.argmax(base_probs)))
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

                            # full (all perms) - only if available
                            if full_enabled:
                                agg_full = _aggregate_probs_over_permutations(probs_seq_np, perm_list, k)
                                pred_full_idx = int(np.argmax(agg_full))
                                pred_full = option_ids[pred_full_idx]
                                full_pred_idx_list.append(pred_full_idx)
                                corr_full = (pred_full == data['ideal'])
                                full_correct_list.append(corr_full)
                                full_corrects += 1 if corr_full else 0
                                full_total += 1

                        # ---------- confidence stats & probe triggers (baseline) ----------
                        default_conf = []          # base gap (letter-space)
                        mean_gap_list = []         # gap(mean(base,probe)) (content-space)
                        flip_trigger_mask = []     # pred_base != pred_probe (content-space)
                        probe2_correct_list = []   # correctness of argmax(mean_probs)

                        for i, bp in enumerate(base_probs_list):
                            bp = np.asarray(bp, dtype=np.float64)
                            vals = np.sort(bp)[::-1]
                            top1 = float(vals[0]) if vals.shape[0] > 0 else 0.0
                            top2 = float(vals[1]) if vals.shape[0] > 1 else 0.0
                            default_conf.append(top1 - top2)

                            shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(bp, k)
                            probe_perm_idx = cyclic_indices[shift]

                            probs_base_raw = per_sample_probs[i][identity_idx]
                            agg_base = _aggregate_probs_over_permutations([probs_base_raw.tolist()], [tuple(range(k))], k)

                            probs_probe_raw = per_sample_probs[i][probe_perm_idx]
                            agg_probe = _aggregate_probs_over_permutations([probs_probe_raw.tolist()], [cyc_perms[shift]], k)

                            mean_probs = (agg_base + agg_probe) / 2.0
                            vals_mean = np.sort(mean_probs)[::-1]
                            mean_gap = float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0
                            mean_gap_list.append(mean_gap)

                            pred_base_cs = option_ids[int(np.argmax(agg_base))]
                            pred_probe_cs = option_ids[int(np.argmax(agg_probe))]
                            flip_trigger_mask.append(pred_base_cs != pred_probe_cs)

                            pred2 = option_ids[int(np.argmax(mean_probs))]
                            probe2_pred_idx_list.append(int(np.argmax(mean_probs)))
                            probe2_correct_list.append(pred2 == ideals[i])

                        default_conf = np.asarray(default_conf, dtype=np.float64)
                        mean_conf = np.asarray(mean_gap_list, dtype=np.float64)
                        arr_flip_trigger = np.asarray(flip_trigger_mask, dtype=bool)
                        arr_probe2_correct = np.asarray(probe2_correct_list, dtype=bool)

                        # Base T/F 그룹별 Gap 및 트랜지션 카운트 수집 (Cyclic & Full)
                        base_t_gaps_cyc, base_f_gaps_cyc = [], []
                        t_to_f_count_cyc, f_to_t_count_cyc = 0, 0
                        for bc, cc, conf in zip(base_correct_list, cyclic_correct_list, default_conf):
                            if bc:
                                base_t_gaps_cyc.append(float(conf))
                                if not cc:
                                    t_to_f_count_cyc += 1
                            else:
                                base_f_gaps_cyc.append(float(conf))
                                if cc:
                                    f_to_t_count_cyc += 1
                        transition_records_cyclic.append({
                            "subject": str(subject),
                            "base_t_gaps": base_t_gaps_cyc,
                            "base_f_gaps": base_f_gaps_cyc,
                            "t_to_f_count": t_to_f_count_cyc,
                            "f_to_t_count": f_to_t_count_cyc,
                        })

                        if full_enabled and len(full_correct_list) == len(base_correct_list):
                            base_t_gaps_full, base_f_gaps_full = [], []
                            t_to_f_count_full, f_to_t_count_full = 0, 0
                            for bc, fc, conf in zip(base_correct_list, full_correct_list, default_conf):
                                if bc:
                                    base_t_gaps_full.append(float(conf))
                                    if not fc:
                                        t_to_f_count_full += 1
                                else:
                                    base_f_gaps_full.append(float(conf))
                                    if fc:
                                        f_to_t_count_full += 1
                            transition_records_full.append({
                                "subject": str(subject),
                                "base_t_gaps": base_t_gaps_full,
                                "base_f_gaps": base_f_gaps_full,
                                "t_to_f_count": t_to_f_count_full,
                                "f_to_t_count": f_to_t_count_full,
                            })

                        # ---------- optional: PRIDE debiasing + n_runs averaging (like debiase_pride.py) ----------
                        pride_enabled = bool(getattr(args, "pride_mix", False))
                        by_perc_baseline: Dict[float, List[dict]] = defaultdict(list)
                        by_pride_alpha: Dict[float, List[dict]] = defaultdict(list)  # alpha(0.5,1.0,2,5,...) -> [cobj]
                        curve_objs_baseline = []
                        curve_objs_pride = []

                        pride_prefix_list = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_prefix_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0]
                        if not pride_prefix_list:
                            pride_prefix_list = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
                        ours_th1_list = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_ours_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0]
                        if not ours_th1_list:
                            ours_th1_list = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

                        # n_runs>1이면 외부 run_idx당 1회만 (이미 다른 결과). n_runs==1이면 n_runs번 curve variation
                        inner_run_indices = [run_idx] if use_run_suffix else list(range(n_runs))
                        for run_idx_inner in inner_run_indices:
                            cyclic_fracs_run = [int(x) for x in _parse_percent_value_list(getattr(args, "plot_cyclic_fractions", "0,10,20,30,40,50,60,70,80,90,100")) if 0 <= x <= 100]

                            for perc in ours_th1_list:
                                perc = float(perc)
                                labels_idx_for_curves = [option_ids.index(str(x)) for x in ideals]

                                # --- 1) BASELINE Curves 계산 ---
                                cobj = _compute_curves_for_one_percentile(
                                    subject=subject, tag="baseline", k=k, perm_list=perm_list,
                                    base_correct_list=base_correct_list, cyclic_correct_list=cyclic_correct_list,
                                    full_correct_list=full_correct_list if full_enabled else [],
                                    default_conf=default_conf, mean_conf=mean_conf, flip_trigger=arr_flip_trigger,
                                    probe2_correct=arr_probe2_correct, perc_value=perc, full_enabled=bool(full_enabled),
                                    labels_idx=labels_idx_for_curves, base_pred_idx=base_pred_idx_list,
                                    cyclic_pred_idx=cyclic_pred_idx_list, probe2_pred_idx=probe2_pred_idx_list,
                                    full_pred_idx=full_pred_idx_list if full_enabled and len(full_pred_idx_list) == len(ideals) else None,
                                    cyclic_fractions=cyclic_fracs_run, run_seed_offset=run_idx_inner,
                                )

                                if cobj:
                                    by_perc_baseline[perc].append(cobj)
                                    def _get_static_pt(th1_p, rule_func):
                                        c, a, th2p, st = _run_online_th1_quantile_th2_from_th1_rule_with_stats(
                                            default_conf, mean_conf, base_correct_list, cyclic_correct_list,
                                            arr_probe2_correct, k, th1_p, rule_func, None)
                                        out = {'cost': c, 'acc': a, 'label': 'th1/2', 'marker': '*', 'color': 'gray'}
                                        try:
                                            _, _, _, preds = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                                default_conf, mean_conf, base_pred_idx_list, cyclic_pred_idx_list, probe2_pred_idx_list,
                                                labels_idx_for_curves, k, th1_p, rule_func, None)
                                            out['recall_std'] = float(_recall_std(labels_idx_for_curves, preds, k))
                                            out['n_base'], out['n_probe2'], out['n_cyclic'] = st['n_base'], st['n_probe2'], st['n_cyclic']
                                        except Exception:
                                            pass
                                        return out
                                    if "heuristic_points" not in cobj:
                                        cobj["heuristic_points"] = [_get_static_pt(perc, lambda x: x / 2.0)]

                                    # Ours (baseline) transition 기록
                                    try:
                                        _, _, _, preds_ours = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                            default_conf, mean_conf, base_pred_idx_list, cyclic_pred_idx_list,
                                            probe2_pred_idx_list, labels_idx_for_curves, k, perc, lambda x: x / 2.0, None)
                                        rec = _make_transition_record_from_preds(
                                            base_correct_list, preds_ours, labels_idx_for_curves, default_conf, subject)
                                        transition_records_ours_by_p.setdefault(perc, []).append(rec)
                                    except Exception:
                                        pass

                        # --- 2) PRIDE Curves: pride_alpha(2,5,10,20) x ours_th1(2..100) 분리 ---
                        if pride_enabled:
                            for pride_alpha in pride_prefix_list:
                                seed = _stable_u32_seed(str(subject), int(getattr(args, "pride_seed", 0)) + run_idx_inner)
                                pride_prior, pride_meta = _estimate_pride_prior_random_prefix_mean(
                                    per_sample_probs=per_sample_probs,
                                    cyclic_indices=cyclic_indices,
                                    k=k,
                                    prefix_ratio=float(pride_alpha) / 100.0,
                                    seed=seed,
                                )
                                prefix_ids_set = set(int(x) for x in (pride_meta.get("prefix_ids") or []))

                                base_correct_list_pr = []
                                cyclic_correct_list_pr = []
                                full_correct_list_pr = []
                                full_pred_idx_list_pr = []
                                default_conf_pr = []
                                mean_gap_list_pr = []
                                flip_trigger_mask_pr = []
                                probe2_correct_list_pr = []
                                base_pred_idx_list_pr = []
                                cyclic_pred_idx_list_pr = []
                                probe2_pred_idx_list_pr = []

                                for i in range(len(per_sample_probs)):
                                    ps = np.asarray(per_sample_probs[i], dtype=np.float64)
                                    ps_corr = np.asarray([_pride_correct_row(ps[j], pride_prior) for j in range(ps.shape[0])], dtype=np.float64)

                                    # Cyclic
                                    cyc_probs_corr = [ps_corr[idx] for idx in cyclic_indices]
                                    agg_cyc_corr = _aggregate_probs_over_permutations([cp.tolist() for cp in cyc_probs_corr], cyc_perms, k)
                                    pred_cyc_corr = option_ids[int(np.argmax(agg_cyc_corr))]
                                    cyclic_pred_idx_list_pr.append(int(np.argmax(agg_cyc_corr)))
                                    cyclic_correct_list_pr.append(pred_cyc_corr == ideals[i])

                                    # Base
                                    base_row_corr = np.asarray(ps_corr[identity_idx], dtype=np.float64)
                                    pred_base_corr = option_ids[int(np.argmax(base_row_corr))]
                                    base_pred_idx_list_pr.append(int(np.argmax(base_row_corr)))
                                    base_correct_list_pr.append(pred_base_corr == ideals[i])

                                    # Full
                                    if full_enabled:
                                        agg_full_corr = _aggregate_probs_over_permutations(ps_corr, perm_list, k)
                                        pred_full_idx_pr = int(np.argmax(agg_full_corr))
                                        full_pred_idx_list_pr.append(pred_full_idx_pr)
                                        full_correct_list_pr.append(option_ids[pred_full_idx_pr] == ideals[i])

                                    # Gaps
                                    vals = np.sort(base_row_corr)[::-1]
                                    default_conf_pr.append((float(vals[0]) if len(vals) > 0 else 0.0) - (float(vals[1]) if len(vals) > 1 else 0.0))

                                    shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(base_row_corr, k)
                                    probe_perm_idx = cyclic_indices[shift]
                                    agg_base = _aggregate_probs_over_permutations([base_row_corr.tolist()], [tuple(range(k))], k)
                                    probe_row_corr = np.asarray(ps_corr[probe_perm_idx], dtype=np.float64)
                                    agg_probe = _aggregate_probs_over_permutations([probe_row_corr.tolist()], [cyc_perms[shift]], k)

                                    mean_probs = (np.asarray(agg_base, dtype=np.float64) + np.asarray(agg_probe, dtype=np.float64)) / 2.0
                                    vals_mean = np.sort(mean_probs)[::-1]
                                    mean_gap_list_pr.append(float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0)

                                    pred_base_cs = option_ids[int(np.argmax(agg_base))]
                                    pred_probe_cs = option_ids[int(np.argmax(agg_probe))]
                                    flip_trigger_mask_pr.append(pred_base_cs != pred_probe_cs)

                                    pred2 = option_ids[int(np.argmax(mean_probs))]
                                    probe2_pred_idx_list_pr.append(int(np.argmax(mean_probs)))
                                    probe2_correct_list_pr.append(pred2 == ideals[i])

                                default_conf_pr = np.asarray(default_conf_pr, dtype=np.float64)
                                mean_conf_pr = np.asarray(mean_gap_list_pr, dtype=np.float64)
                                arr_flip_trigger_pr = np.asarray(flip_trigger_mask_pr, dtype=bool)
                                arr_probe2_correct_pr = np.asarray(probe2_correct_list_pr, dtype=bool)

                                # alpha>=100: prefix=전체 → 보정 불가. Cyclic permutation과 동일 (원본 사용)
                                base_for_dp = base_correct_list if pride_alpha >= 100 else base_correct_list_pr
                                cyclic_for_dp = cyclic_correct_list if pride_alpha >= 100 else cyclic_correct_list_pr
                                base_pred_dp = base_pred_idx_list if pride_alpha >= 100 else base_pred_idx_list_pr
                                cyclic_pred_dp = cyclic_pred_idx_list if pride_alpha >= 100 else cyclic_pred_idx_list_pr

                                cobj_pr = _compute_curves_for_one_percentile(
                                    subject=subject, tag="pride_mix", k=k, perm_list=perm_list,
                                    base_correct_list=base_for_dp, cyclic_correct_list=cyclic_for_dp,
                                    full_correct_list=full_correct_list_pr if full_enabled else [],
                                    default_conf=default_conf_pr, mean_conf=mean_conf_pr,
                                    flip_trigger=arr_flip_trigger_pr, probe2_correct=arr_probe2_correct_pr,
                                    perc_value=float(pride_alpha), full_enabled=bool(full_enabled),
                                    forced_cyclic_ids=prefix_ids_set, labels_idx=labels_idx_for_curves,
                                    base_pred_idx=base_pred_dp, cyclic_pred_idx=cyclic_pred_dp,
                                    probe2_pred_idx=probe2_pred_idx_list_pr,
                                    full_pred_idx=full_pred_idx_list_pr if full_enabled and len(full_pred_idx_list_pr) == len(ideals) else None,
                                    cyclic_fractions=[pride_alpha],
                                    run_seed_offset=run_idx_inner,
                                )
                                if cobj_pr:
                                    # alpha>=100: cost/acc/recall_std 모두 Cyclic과 동일하게 원본 사용
                                    bc_use, cc_use = (base_for_dp, cyclic_for_dp) if pride_alpha >= 100 else (base_correct_list_pr, cyclic_correct_list_pr)
                                    bp_use, cp_use = (base_pred_dp, cyclic_pred_dp) if pride_alpha >= 100 else (base_pred_idx_list_pr, cyclic_pred_idx_list_pr)
                                    def _get_static_pt_pride(th1_p, rule_func, label_key):
                                        c, a, th2p, st = _run_online_th1_quantile_th2_from_th1_rule_with_stats(
                                            default_conf_pr, mean_conf_pr, bc_use, cc_use,
                                            arr_probe2_correct_pr, k, th1_p, rule_func, prefix_ids_set)
                                        out = {'cost': c, 'acc': a, 'label': label_key, 'th1_p': th1_p, 'marker': '*', 'color': 'gray'}
                                        try:
                                            _, _, _, preds = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                                default_conf_pr, mean_conf_pr, bp_use, cp_use,
                                                probe2_pred_idx_list_pr, labels_idx_for_curves, k, th1_p, rule_func, prefix_ids_set)
                                            out['recall_std'] = float(_recall_std(labels_idx_for_curves, preds, k))
                                            out['n_base'], out['n_probe2'], out['n_cyclic'] = st['n_base'], st['n_probe2'], st['n_cyclic']
                                        except Exception:
                                            pass
                                        return out
                                    def _get_static_pt_online_sqrt(th1_p):
                                        c, a, th2p, st = _run_online_sqrt_policy_with_stats(
                                            default_conf_pr, mean_conf_pr, bc_use, cc_use,
                                            arr_probe2_correct_pr, k, th1_p, prefix_ids_set)
                                        out = {'cost': c, 'acc': a, 'label': 'online_sqrt_all', 'th1_p': th1_p, 'marker': 'D', 'color': 'gray'}
                                        try:
                                            _, _, preds = _run_online_sqrt_policy_with_preds(
                                                default_conf_pr, mean_conf_pr, bp_use, cp_use,
                                                probe2_pred_idx_list_pr, labels_idx_for_curves, k, th1_p, prefix_ids_set)
                                            out['recall_std'] = float(_recall_std(labels_idx_for_curves, preds, k))
                                            out['n_base'], out['n_probe2'], out['n_cyclic'] = st['n_base'], st['n_probe2'], st['n_cyclic']
                                        except Exception:
                                            pass
                                        return out
                                    pts_th12 = [_get_static_pt_pride(float(th1), lambda x: x / 2.0, "th1/2") for th1 in ours_th1_list]
                                    pts_sqrt = [_get_static_pt_online_sqrt(float(th1)) for th1 in ours_th1_list]
                                    cobj_pr["heuristic_points"] = pts_th12 + pts_sqrt
                                    by_pride_alpha[pride_alpha].append(cobj_pr)

                                for ours_th1 in ours_th1_list:
                                    try:
                                        th1_f = float(ours_th1)
                                        th1_seed_off = int(th1_f) if th1_f.is_integer() else int(round(th1_f * 1000.0))
                                        seed_cyc = _stable_u32_seed(str(subject), int(run_idx_inner)) + int(th1_seed_off)
                                        _, _, preds_dp = _run_cyclic_random_fraction_with_preds(
                                            base_pred_idx_list_pr, cyclic_pred_idx_list_pr,
                                            labels_idx_for_curves, k, th1_f, seed_cyc)
                                        rec_dp = _make_transition_record_from_preds(
                                            base_correct_list_pr, preds_dp, labels_idx_for_curves, default_conf_pr, subject)
                                        transition_records_default_pride_by_p.setdefault(float(th1_f), []).append(rec_dp)
                                    except Exception:
                                        pass
                                    try:
                                        _, _, _, preds_op = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                            default_conf_pr, mean_conf_pr, base_pred_idx_list_pr, cyclic_pred_idx_list_pr,
                                            probe2_pred_idx_list_pr, labels_idx_for_curves, k, ours_th1, lambda x: x / 2.0, prefix_ids_set)
                                        rec_op = _make_transition_record_from_preds(
                                            base_correct_list_pr, preds_op, labels_idx_for_curves, default_conf_pr, subject)
                                        if float(pride_alpha) == 2.0:
                                            transition_records_ours_pride_by_p.setdefault(float(th1_f), []).append(rec_op)
                                    except Exception:
                                        pass

                        # Merge over runs and append to derived_records
                        for perc in ours_th1_list:
                            perc = float(perc)
                            cobjs_b = by_perc_baseline.get(perc, [])
                            merged_b = _merge_curve_objs_over_runs(cobjs_b) if cobjs_b else None
                            if merged_b:
                                derived_records_by_p.setdefault(perc, []).append(merged_b)
                                curve_objs_baseline.append(merged_b)

                        if pride_enabled:
                            for pride_alpha in pride_prefix_list:
                                cobjs_p = by_pride_alpha.get(pride_alpha, [])
                                merged_p = _merge_curve_objs_over_runs(cobjs_p) if cobjs_p else None
                                if merged_p:
                                    derived_records_pride_by_alpha.setdefault(pride_alpha, []).append(merged_p)
                                    curve_objs_pride.append(merged_p)

                        # ---------- save cyclic/base derived results ----------
                        cyclic_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_cyclic'
                        if getattr(args, 'option_id_set', None):
                            cyclic_save_path += f'_id-{args.option_id_set}'
                        os.makedirs(cyclic_save_path, exist_ok=True)

                        cyclic_acc = (cyclic_corrects / cyclic_total) if cyclic_total > 0 else float('nan')
                        save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results,
                                 metrics={'type': 'metric', 'data': {'accuracy': cyclic_acc}})

                        base_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}'
                        if getattr(args, 'option_id_set', None):
                            base_save_path += f'_id-{args.option_id_set}'
                        os.makedirs(base_save_path, exist_ok=True)

                        base_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64))) if len(base_correct_list) else float('nan')
                        save_results(f'{base_save_path}/{subject}.jsonl', base_results,
                                 metrics={'type': 'metric', 'data': {'accuracy': base_acc}})

                        full_acc = (full_corrects / full_total) if full_total > 0 else float('nan')

                        # ---------- curve save path (for per-subject plots when not MMLU) ----------
                        curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                        if getattr(args, 'option_id_set', None):
                            curve_save_path += f'_id-{args.option_id_set}'
                        os.makedirs(curve_save_path, exist_ok=True)

                        # (per-subject report removed — FINAL CONDENSED REPORT only)
                        save_results(f'{curve_save_path}/{subject}_curve.jsonl', curve_objs_baseline, metrics=None)
                        if pride_enabled and len(curve_objs_pride) > 0:
                            save_results(f'{curve_save_path}/{subject}_pride_curve.jsonl', curve_objs_pride, metrics=None)

                        # (th2 tradeoff plot removed — only macro three-curves acc/recall_std at end)

                    except Exception as e:
                        logger.warning(f"Failed to derive curves for subject '{subject}': {e}")
                        import traceback
                        traceback.print_exc()

            logging_cuda_memory_usage()

        # 논문 작성용 T->F/F->T Empirical Analysis (이미지 업로드 전에 먼저 출력)
        def _print_transition_analysis(records, name):
            if not records:
                return
            all_base_t_gaps, all_base_f_gaps = [], []
            tot_t_to_f, tot_f_to_t = 0, 0
            t_to_f_ratios_per_subj, f_to_t_ratios_per_subj = [], []
            for rec in records:
                base_t = len(rec["base_t_gaps"])
                base_f = len(rec["base_f_gaps"])
                t_to_f = rec["t_to_f_count"]
                f_to_t = rec["f_to_t_count"]
                all_base_t_gaps.extend(rec["base_t_gaps"])
                all_base_f_gaps.extend(rec["base_f_gaps"])
                tot_t_to_f += t_to_f
                tot_f_to_t += f_to_t
                if base_t > 0:
                    t_to_f_ratios_per_subj.append(t_to_f / base_t * 100.0)
                if base_f > 0:
                    f_to_t_ratios_per_subj.append(f_to_t / base_f * 100.0)
            tot_base_t = len(all_base_t_gaps)
            tot_base_f = len(all_base_f_gaps)
            n_records = len(records)
            avg_gap_t = float(np.mean(all_base_t_gaps)) if tot_base_t > 0 else 0.0
            avg_gap_f = float(np.mean(all_base_f_gaps)) if tot_base_f > 0 else 0.0
            if n_records > 1 or len(t_to_f_ratios_per_subj) > 1 or len(f_to_t_ratios_per_subj) > 1:
                t_to_f_ratio = float(np.mean(t_to_f_ratios_per_subj)) if t_to_f_ratios_per_subj else 0.0
                f_to_t_ratio = float(np.mean(f_to_t_ratios_per_subj)) if f_to_t_ratios_per_subj else 0.0
                t_to_f_std = float(np.std(t_to_f_ratios_per_subj)) if len(t_to_f_ratios_per_subj) > 1 else 0.0
                f_to_t_std = float(np.std(f_to_t_ratios_per_subj)) if len(f_to_t_ratios_per_subj) > 1 else 0.0
                ratio_note = f" (macro avg over {n_records} records)"
                ratio_std = f" ± {t_to_f_std:.2f}" if t_to_f_std > 0 else ""
                ratio_std_ft = f" ± {f_to_t_std:.2f}" if f_to_t_std > 0 else ""
            else:
                t_to_f_ratio = (tot_t_to_f / tot_base_t * 100.0) if tot_base_t > 0 else 0.0
                f_to_t_ratio = (tot_f_to_t / tot_base_f * 100.0) if tot_base_f > 0 else 0.0
                ratio_note = ""
                ratio_std = ""
                ratio_std_ft = ""
            logger.info(_purple(f"\n==== EMPIRICAL ANALYSIS: {name} Permutation ===="))
            logger.info(f"[Initial Prediction: TRUE (원본 정답 그룹)]")
            logger.info(f" - Total Samples : {tot_base_t}")
            logger.info(f" - Avg Confidence: {avg_gap_t:.4f} (High Confidence)")
            logger.info(f" - Effect : T -> F (훼손) = {tot_t_to_f} / {tot_base_t} ({t_to_f_ratio:.2f}%{ratio_std}{ratio_note})")
            logger.info(f"\n[Initial Prediction: FALSE (원본 오답 그룹)]")
            logger.info(f" - Total Samples : {tot_base_f}")
            logger.info(f" - Avg Confidence: {avg_gap_f:.4f} (Low Confidence)")
            logger.info(f" - Effect : F -> T (교정) = {tot_f_to_t} / {tot_base_f} ({f_to_t_ratio:.2f}%{ratio_std_ft}{ratio_note})")
            logger.info("======================================================\n")

        _print_transition_analysis(transition_records_cyclic, "Cyclic")
        _print_transition_analysis(transition_records_full, "Full")

        # 논문 Experiments/Analysis: Default+PRIDE, Ours+PRIDE, Ours (per perc 2~100) — 정답/오답 avg conf, T→F, F→T
        def _print_transition_analysis_by_perc(records_by_p: Dict[float, List[dict]], method_name: str):
            if not records_by_p:
                return
            pride_fracs_sorted = sorted([p for p in records_by_p.keys() if isinstance(p, (int, float))])
            suffix = " (α=2% 고정)" if method_name == "Ours+PRIDE" else ""
            logger.info(_purple(f"\n==== EMPIRICAL ANALYSIS: {method_name} (per perc){suffix} [{args.task}] ===="))
            logger.info("perc | avg_conf_T(정답) | avg_conf_F(오답) | T→F(훼손) | F→T(교정) | T→F% | F→T%")
            for p in pride_fracs_sorted:
                recs = records_by_p.get(float(p), [])
                if not recs:
                    continue
                all_t, all_f, tot_t_to_f, tot_f_to_t = [], [], 0, 0
                for r in recs:
                    all_t.extend(r.get("base_t_gaps", []))
                    all_f.extend(r.get("base_f_gaps", []))
                    tot_t_to_f += r.get("t_to_f_count", 0)
                    tot_f_to_t += r.get("f_to_t_count", 0)
                nt, nf = len(all_t), len(all_f)
                avg_t = float(np.mean(all_t)) if nt > 0 else float("nan")
                avg_f = float(np.mean(all_f)) if nf > 0 else float("nan")
                t_to_f_pct = (tot_t_to_f / nt * 100.0) if nt > 0 else 0.0
                f_to_t_pct = (tot_f_to_t / nf * 100.0) if nf > 0 else 0.0
                p_str = f"{float(p):g}%"
                logger.info(f"{p_str:>6} | {avg_t:.4f} | {avg_f:.4f} | {tot_t_to_f} | {tot_f_to_t} | {t_to_f_pct:.2f}% | {f_to_t_pct:.2f}%")
            logger.info("======================================================\n")

        if transition_records_default_pride_by_p:
            _print_transition_analysis_by_perc(transition_records_default_pride_by_p, "Default+PRIDE")
        if transition_records_ours_pride_by_p:
            _print_transition_analysis_by_perc(transition_records_ours_pride_by_p, "Ours+PRIDE")
        if transition_records_ours_by_p:
            _print_transition_analysis_by_perc(transition_records_ours_by_p, "Ours")

        # Three-curves: Cost vs Acc, Cost vs Recall_std (Cyclic / Default+PRIDE / OURS th1/2)
        if len(derived_records_by_p) > 0:
            try:
                out_dir = f"results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full"
                if getattr(args, 'option_id_set', None):
                    out_dir += f"_id-{args.option_id_set}"
                os.makedirs(out_dir, exist_ok=True)
                cyclic_fracs = [int(x) for x in _parse_percent_value_list(getattr(args, "plot_cyclic_fractions", "0,10,20,30,40,50,60,70,80,90,100")) if 0 <= x <= 100]
                pride_fracs = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_ours_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0]
                pride_prefix = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_prefix_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0] or [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
                _plot_three_curves_acc_recall_std(
                    derived_records_by_p,
                    derived_records_pride_by_p if len(derived_records_pride_by_p) > 0 else {},
                    derived_records_pride_by_alpha if len(derived_records_pride_by_alpha) > 0 else {},
                    out_dir,
                    args.task,
                    cyclic_fractions=cyclic_fracs or [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    pride_ours_fractions=pride_fracs or [2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                    pride_prefix_list=pride_prefix,
                    wandb_ok=wandb_ok,
                    wandb_run=wandb_run,
                )
            except Exception as ex:
                logger.warning(f"Three-curves plot failed: {ex}")

        # =========================================================
        # 커스텀 최종 요약 리포트 (사용자 맞춤형 포맷)
        # =========================================================
        if len(derived_records_by_p) > 0:
            logger.info(_purple("==== FINAL CONDENSED REPORT ===="))
            n_subjects = len(subjects)

            def _macro_mean_std_over_runs(vals_list, n_subj, n_run):
                """57개 과목 macro 평균 → 5 run에 대해 mean ± std (MMLU 스타일)"""
                if n_subj <= 1 or n_run <= 1 or len(vals_list) != n_subj * n_run:
                    m = float(np.mean(vals_list)) if vals_list else float("nan")
                    s = float(np.std(vals_list)) if len(vals_list) > 1 else float("nan")
                    return m, s
                # cobjs 순서: s0r0,s0r1,...,s0r(n_run-1), s1r0,..., s(n_subj-1)r(n_run-1)
                run_means = []
                for r in range(n_run):
                    run_vals = [vals_list[r + i * n_run] for i in range(n_subj)]
                    run_vals = [x for x in run_vals if np.isfinite(x)]
                    run_means.append(float(np.mean(run_vals)) if run_vals else float("nan"))
                run_means = [x for x in run_means if np.isfinite(x)]
                mean = float(np.mean(run_means)) if run_means else float("nan")
                std = float(np.std(run_means)) if len(run_means) > 1 else float("nan")
                return mean, std

            def get_cyclic_stats(cobjs, p):
                pf = float(p)
                key_candidates = [f"cyclic_random_{pf}", f"cyclic_random_{pf:g}"]
                # pick first key that exists in at least one cobj
                key = None
                for kk in key_candidates:
                    if any((kk in c) for c in (cobjs or [])):
                        key = kk
                        break
                if key is None:
                    key = key_candidates[0]
                costs, accs, rstds = [], [], []
                for c in cobjs:
                    if key in c and "costs" in c[key] and "accuracies" in c[key]:
                        costs.append(c[key]["costs"][0])
                        accs.append(c[key]["accuracies"][0])
                    rk = f"{key}_recall_std"
                    if rk in c:
                        rstds.append(c[rk])
                if not accs:
                    return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
                mean_c, std_c = _macro_mean_std_over_runs(costs, n_subjects, n_runs)
                mean_a, std_a = _macro_mean_std_over_runs(accs, n_subjects, n_runs)
                mean_r, std_r = _macro_mean_std_over_runs(rstds, n_subjects, n_runs)
                return mean_c, mean_a, mean_r, std_c, std_a, std_r

            def get_heur_stats(cobjs, label="th1/2"):
                costs, accs, rstds, nb, np2, nc = [], [], [], [], [], []
                for c in cobjs:
                    hps = {str(h.get("label")): h for h in (c.get("heuristic_points") or []) if isinstance(h, dict)}
                    if label in hps:
                        h = hps[label]
                        if "cost" in h:
                            costs.append(h["cost"])
                        accs.append(h.get("acc", float("nan")))
                        if "recall_std" in h:
                            rstds.append(h["recall_std"])
                        if "n_base" in h:
                            nb.append(h["n_base"])
                        if "n_probe2" in h:
                            np2.append(h["n_probe2"])
                        if "n_cyclic" in h:
                            nc.append(h["n_cyclic"])
                if not accs:
                    return float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan")
                mean_c, std_c = _macro_mean_std_over_runs(costs, n_subjects, n_runs)
                mean_a, std_a = _macro_mean_std_over_runs(accs, n_subjects, n_runs)
                mean_r, std_r = _macro_mean_std_over_runs(rstds, n_subjects, n_runs)
                mean_nb = float(np.mean(nb)) if nb else 0.0
                mean_np2 = float(np.mean(np2)) if np2 else 0.0
                mean_nc = float(np.mean(nc)) if nc else 0.0
                return mean_c, mean_a, mean_r, mean_nb, mean_np2, mean_nc, std_c, std_a, std_r

            def get_heur_stats_by_th1_p(cobjs, th1_p, label_filter="online_sqrt_all"):
                costs, accs, rstds, nb, np2, nc = [], [], [], [], [], []
                for c in cobjs:
                    for h in (c.get("heuristic_points") or []):
                        if isinstance(h, dict) and h.get("th1_p") == th1_p and h.get("label") == label_filter:
                            if "cost" in h:
                                costs.append(h["cost"])
                            accs.append(h.get("acc", float("nan")))
                            if "recall_std" in h:
                                rstds.append(h["recall_std"])
                            if "n_base" in h:
                                nb.append(h["n_base"])
                            if "n_probe2" in h:
                                np2.append(h["n_probe2"])
                            if "n_cyclic" in h:
                                nc.append(h["n_cyclic"])
                            break
                if not accs:
                    return float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan")
                mean_c, std_c = _macro_mean_std_over_runs(costs, n_subjects, n_runs)
                mean_a, std_a = _macro_mean_std_over_runs(accs, n_subjects, n_runs)
                mean_r, std_r = _macro_mean_std_over_runs(rstds, n_subjects, n_runs)
                mean_nb = float(np.mean(nb)) if nb else 0.0
                mean_np2 = float(np.mean(np2)) if np2 else 0.0
                mean_nc = float(np.mean(nc)) if nc else 0.0
                return mean_c, mean_a, mean_r, mean_nb, mean_np2, mean_nc, std_c, std_a, std_r

            pride_fracs = [float(x) for x in _parse_percent_value_list(
                getattr(args, "plot_pride_ours_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")
            ) if 0.0 <= float(x) <= 100.0] or [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            # Cyclic 레포트: 항상 0,10,20,...,100 전체 구간 출력 (plot_cyclic_fractions와 무관)
            cyclic_fracs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            pride_alphas = sorted(derived_records_pride_by_alpha.keys()) if derived_records_pride_by_alpha else []

            _fmt = (lambda m, s: f"{m:.3f}±{s:.3f}" if np.isfinite(s) and s > 0 else f"{m:.3f}") if n_runs > 1 else (lambda m, s: f"{m:.3f}")
            _fmt4 = (lambda m, s: f"{m:.4f}±{s:.4f}" if np.isfinite(s) and s > 0 else f"{m:.4f}") if n_runs > 1 else (lambda m, s: f"{m:.4f}")

            # 1. default + pride (per alpha only — alpha와 cyclic fraction p 동일 개념)
            logger.info("---- default + pride ----")
            for alpha in pride_alphas:
                cobjs = derived_records_pride_by_alpha[alpha]
                p = alpha  # Default+PRIDE: prefix α% = cyclic fraction, 하나의 파라미터만 사용
                cost, acc, rstd, std_c, std_a, std_r = get_cyclic_stats(cobjs, p)
                a_str = f"{float(alpha):g}"
                logger.info(f"default_pride_α{a_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}")

            # 2. ours + pride (per alpha): th1/2 and Online Sqrt
            logger.info("---- ours + pride (th1/2) ----")
            for alpha in pride_alphas:
                cobjs = derived_records_pride_by_alpha[alpha]
                for p in pride_fracs:
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats_by_th1_p(cobjs, p, "th1/2")
                    a_str = f"{float(alpha):g}"
                    p_str = f"{float(p):g}"
                    logger.info(f"ours_pride_th12_α{a_str}_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")
            logger.info("---- ours + pride (Online Sqrt) ----")
            for alpha in pride_alphas:
                cobjs = derived_records_pride_by_alpha[alpha]
                for p in pride_fracs:
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats_by_th1_p(cobjs, p, "online_sqrt_all")
                    a_str = f"{float(alpha):g}"
                    p_str = f"{float(p):g}"
                    logger.info(f"ours_pride_sqrt_α{a_str}_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")

            # 3. ours
            logger.info("---- ours ----")
            for p in pride_fracs:
                if float(p) in derived_records_by_p:
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats(derived_records_by_p[float(p)], "th1/2")
                    p_str = f"{float(p):g}"
                    logger.info(f"ours_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")

            # 4. cyclic
            logger.info("---- cyclic ----")
            base_any_cobjs = next(iter(derived_records_by_p.values()), []) if derived_records_by_p else []
            for p in cyclic_fracs:
                cost, acc, rstd, std_c, std_a, std_r = get_cyclic_stats(base_any_cobjs, p)
                logger.info(f"cyclic_{p:03d}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}")

    # -------- finalize W&B --------
    _wandb_done = {"done": False}
    def _wandb_finish():
        if _wandb_done["done"] or not wandb_ok or wandb_run is None:
            return
        try:
            import wandb
            logger.info(_blue("W&B: syncing and finishing run..."))
            wandb.finish()
            time.sleep(5)  # 업로드 스레드 완료 대기 (업로드 중 프로세스 죽는 문제 완화)
            logger.info(_blue("W&B: run finished."))
        except Exception as e:
            logger.warning(f"W&B finish failed: {e}")
        finally:
            _wandb_done["done"] = True

    if wandb_ok and wandb_run is not None:
        atexit.register(_wandb_finish)
    try:
        _wandb_finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
