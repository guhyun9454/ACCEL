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
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import zlib
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

        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
        past_gaps.append(gap_i)

    return total_cost / float(N), corrects / float(N), preds


def _stable_u32_seed(s: str, base_seed: int = 0) -> int:
    return (int(zlib.crc32(s.encode("utf-8"))) + int(base_seed)) & 0xFFFFFFFF


def _run_cyclic_random_fraction(
    base_correct: List[bool],
    cyclic_correct: List[bool],
    k: int,
    fraction_pct: int,
    seed: int,
) -> Tuple[float, float]:
    """
    Randomly select fraction_pct% of samples to run cyclic; rest use base.
    fraction_pct in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].
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
    fraction_pct: int,
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


def _plot_main_figure_cost_vs_acc(
    derived_records_by_p: Dict[float, List[dict]],
    out_path: str,
    task: str,
):
    """
    Main figure: Cost vs Accuracy trade-off — SLM improvement via cascading policy.
    Shows default (cost≈1), cyclic (cost=k), and our policies achieving Pareto improvement.
    """
    if not derived_records_by_p or len(derived_records_by_p) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=180)
    cyclic_random_keys = [f"cyclic_random_{fp}" for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    keys = ["default", "cyclic"] + cyclic_random_keys
    colors = {"default": "#2ca02c", "cyclic": "#1f77b4"}
    markers = {"default": "o", "cyclic": "s"}
    for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        k = f"cyclic_random_{fp}"
        colors[k] = plt.cm.viridis((fp - 10) / 90.0)
        markers[k] = "."
    for key in keys:
        costs_all, accs_all = [], []
        for p, cobjs in sorted(derived_records_by_p.items(), key=lambda t: t[0]):
            for cobj in cobjs:
                n = int(cobj.get("n_samples", 0)) or 1
                if key in ["default", "cyclic"]:
                    always = cobj.get("always", {}) or {}
                    if key not in always:
                        continue
                    costs_all.append(float(always[key]["cost"]))
                    accs_all.append(float(always[key]["acc"]) * 100.0)
                elif key in cobj:
                    costs_all.append(float(cobj[key]["costs"][0]))
                    accs_all.append(float(cobj[key]["accuracies"][0]) * 100.0)
        if len(costs_all) > 0 and len(accs_all) > 0:
            c_mean = float(np.mean(costs_all))
            a_mean = float(np.mean(accs_all))
            ax.scatter(c_mean, a_mean, marker=markers.get(key, "o"), s=120, c=colors.get(key, "gray"),
                       label=key, zorder=5, edgecolors="black", linewidths=0.8)
    ax.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"{task} — Cost vs Accuracy (SLM cascading policy)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_cyclic_random_pride_vs_baseline_curves(
    derived_records_by_p: Dict[float, List[dict]],
    derived_records_pride_by_p: Dict[float, List[dict]],
    out_dir: str,
    task: str,
):
    """Plot 3 curves: X=cost, Y=acc/recall_std. Macro avg over subjects (MMLU: 57 subj, ARC/CSQA: 1). (1) Cyclic no PRIDE 5~100%, (2) PRIDE+OURS Online Sqrt All p=5/10/20/30, (3) OURS th1/2 no PRIDE p=5/10/20/30."""
    if not derived_records_by_p:
        logger.debug("three-curves plot skipped: derived_records_by_p empty")
        return
    fractions = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ps_heuristic = [5, 10, 20, 30]
    # Curve 1: Cyclic (no PRIDE) — 11 points, macro avg (np.mean) over subjects
    p_cyclic = next((float(p) for p in ps_heuristic if float(p) in derived_records_by_p), None)
    if p_cyclic is None:
        logger.warning("three-curves plot skipped: no p in {5,10,20,30} in derived_records_by_p (keys=%s)", list(derived_records_by_p.keys()))
        return
    cobjs_b = derived_records_by_p.get(p_cyclic, [])
    n_subjects = len(cobjs_b)
    cost_cyclic, acc_cyclic, rstd_cyclic = [], [], []
    for fp in fractions:
        key = f"cyclic_random_{fp}"
        cbs = [float(c[key]["costs"][0]) for c in cobjs_b if key in c]
        abs_ = [float(c[key]["accuracies"][0]) * 100.0 for c in cobjs_b if key in c]
        rbs = [float(c.get(f"{key}_recall_std", float("nan"))) for c in cobjs_b if key in c]
        cost_cyclic.append(np.mean(cbs) if cbs else float("nan"))
        acc_cyclic.append(np.mean(abs_) if abs_ else float("nan"))
        rstd_cyclic.append(np.nanmean(rbs) if rbs else float("nan"))

    # Curve 2 & 3: heuristic points — (cost, acc, recall_std) per p
    def _agg_heur(by_p, label):
        costs, accs, rstds = [], [], []
        for p in ps_heuristic:
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
        return costs, accs, rstds

    cobjs_p = derived_records_pride_by_p if derived_records_pride_by_p else {}
    cost_pride, acc_pride, rstd_pride = _agg_heur(cobjs_p, "Online Sqrt (All)") if cobjs_p else ([float("nan")] * 4, [float("nan")] * 4, [float("nan")] * 4)
    cost_ours, acc_ours, rstd_ours = _agg_heur(derived_records_by_p, "th1/2")

    def _plot_curve(ax, costs, accs_or_rstds, fractions_or_ps, marker, color, linestyle, label):
        valid = [(c, y) for c, y in zip(costs, accs_or_rstds) if np.isfinite(c) and np.isfinite(y)]
        if not valid:
            return
        xs, ys = zip(*sorted(valid, key=lambda t: t[0]))
        ax.plot(xs, ys, marker=marker, color=color, linestyle=linestyle, linewidth=2, markersize=8, label=label)

    # Colors: light blue (Full Perm), light orange (Cyclic Perm), green (PriDe)
    color_cyclic = "#F39C12"   # light orange (Cyclic Perm)
    color_pride = "#27AE60"    # green (PriDe)
    color_ours = "#5DADE2"     # light blue (Full Perm)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax, cost_cyclic, acc_cyclic, fractions, "o", color_cyclic, "-", "Cyclic (no PRIDE)")
    _plot_curve(ax, cost_pride, acc_pride, ps_heuristic, "s", color_pride, "--", "PRIDE+OURS (Online Sqrt All)")
    _plot_curve(ax, cost_ours, acc_ours, ps_heuristic, "^", color_ours, "-.", "OURS (th1/2, no PRIDE)")
    macro_note = f" (macro over {n_subjects} subjects)" if n_subjects > 1 else ""
    ax.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"{task} — Accuracy{macro_note}", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{task}_three_curves_acc.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(_purple(f"Saved three-curves acc: {out_path}"))

    fig2, ax2 = plt.subplots(figsize=(10, 6.5), dpi=160)
    _plot_curve(ax2, cost_cyclic, rstd_cyclic, fractions, "o", color_cyclic, "-", "Cyclic (no PRIDE)")
    _plot_curve(ax2, cost_pride, rstd_pride, ps_heuristic, "s", color_pride, "--", "PRIDE+OURS (Online Sqrt All)")
    _plot_curve(ax2, cost_ours, rstd_ours, ps_heuristic, "^", color_ours, "-.", "OURS (th1/2, no PRIDE)")
    ax2.set_xlabel("Computational Cost (× of default forward pass)", fontsize=11)
    ax2.set_ylabel("Recall std", fontsize=11)
    ax2.set_title(f"{task} — Recall std{macro_note}", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, f"{task}_three_curves_recall_std.png")
    fig2.savefig(out_path2, bbox_inches="tight")
    plt.close(fig2)
    logger.info(_purple(f"Saved three-curves recall_std: {out_path2}"))


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
    default_acc_pr = float(np.mean(np.asarray(base_correct_pr, dtype=np.float64))) if len(base_correct_pr) else float("nan")
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

        if gap_i >= th1_val:
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

        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))
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

    default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    cyclic_acc_always = float(np.mean(np.asarray(cyclic_correct_list, dtype=np.float64)))
    full_acc_always = float(np.mean(np.asarray(full_correct_list, dtype=np.float64))) if full_enabled and len(full_correct_list) == N else float("nan")

    # Cyclic random fraction (10, 20, ..., 100%)
    seed_base = _stable_u32_seed(str(subject), 0)
    cyclic_random_costs: Dict[str, float] = {}
    cyclic_random_accs: Dict[str, float] = {}
    for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        c_r, a_r = _run_cyclic_random_fraction(
            base_correct_list, cyclic_correct_list, k, fp, seed_base + int(fp)
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
    }
    # Optional: add recall_std when labels_idx and preds available
    if labels_idx is not None and base_pred_idx is not None and cyclic_pred_idx is not None:
        try:
            curve_obj["default_recall_std"] = float(_recall_std(labels_idx, base_pred_idx, k))
            curve_obj["cyclic_recall_std"] = float(_recall_std(labels_idx, cyclic_pred_idx, k))
            for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                _, _, preds_r = _run_cyclic_random_fraction_with_preds(
                    base_pred_idx, cyclic_pred_idx, labels_idx, k, fp, seed_base + int(fp)
                )
                curve_obj[f"cyclic_random_{fp}_recall_std"] = float(_recall_std(labels_idx, preds_r, k))
            if full_enabled and full_pred_idx is not None and len(full_pred_idx) == len(labels_idx):
                curve_obj["full_recall_std"] = float(_recall_std(labels_idx, full_pred_idx, k))
        except Exception:
            pass
    if full_enabled:
        curve_obj["always"]["full"] = {"cost": float(C_full), "acc": float(full_acc_always)}
        curve_obj["full"] = {"costs": [float(C_full)], "accuracies": [float(full_acc_always)]}
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

    cyclic_random_keys = [f"cyclic_random_{fp}" for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    for key in cyclic_random_keys:
        if key in curve_obj:
            c0 = float(curve_obj[key]["costs"][0])
            a0 = float(curve_obj[key]["accuracies"][0])
            extra = ""
            rstd = curve_obj.get(f"{key}_recall_std")
            if isinstance(rstd, (int, float)):
                extra = f", recall_std={rstd:.4f}"
            logger.info(f"BASELINE {key:<18} : cost={c0:.3f}, acc={a0:.4f}{extra}")


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

    cyclic_random_keys = [f"cyclic_random_{fp}" for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    for key in cyclic_random_keys:
        if key in curve_obj:
            c0 = float(curve_obj[key]["costs"][0])
            a0 = float(curve_obj[key]["accuracies"][0])
            extra = ""
            rstd = curve_obj.get(f"{key}_recall_std")
            if isinstance(rstd, (int, float)):
                extra = f", recall_std={rstd:.4f}"
            logger.info(f"{name} {key:<18} : cost={c0:.3f}, acc={a0:.4f}{extra}")


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
            entity = getattr(args, "wandb_entity", None)
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

    # -------- Tokenizer / Model --------
    toker = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
        cache_dir=getattr(args, "cache_dir", None),
    )

    use_bf16 = bool(torch.cuda.is_available()) and bool(torch.cuda.is_bf16_supported())
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
        derived_records_pride_by_p: Dict[float, List[dict]] = {}  # p -> list of PRIDE+OURS curve_obj (per subject)
        pride_recall_std_records: List[dict] = []  # [{'subject':str,'rstd':float,'m':int,'N':int}]
        recall_std_vs_p_records: List[dict] = []  # [{'subject':str,'p':float,'method':str,'kind':str,'rstd':float}]

        for subject in subjects[::1]:
            cached_path = f'{args.save_path}/{subject}.jsonl'
            use_cached = (not bool(getattr(args, 'force', False))) and os.path.exists(cached_path)

            logger.info(_blue(f"Preparing: {subject}"))
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = prepare_eval_fn(model, toker, few_shot_samples)

            if use_cached:
                logger.info(_blue(f"Using cached results: {cached_path}"))
                results = _read_results_file(cached_path) or []
            else:
                logger.info(_blue(f"Run started: {subject}"))
                max_samples = 100 if bool(getattr(args, 'test', False)) else None
                n_threads = torch.cuda.device_count()
                n_threads = max(1, int(n_threads)) if 'falcon' not in args.pretrained_model_path else 1
                results = eval_all_samples(
                    eval_fn, eval_samples,
                    name=f'{args.task},{args.num_few_shot},{args.setting},{subject}',
                    threads=n_threads,
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

            logger.info(_orange(f"Run completed: {subject}"))

            if not use_cached:
                save_results(cached_path, results, metrics)
                logger.info(f"Results saved: {subject}")

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

                    # ---------- optional: PRIDE debiasing then run OUR policies on debiased probs ----------
                    pride_enabled = bool(getattr(args, "pride_mix", False))
                    pride_prior = None
                    pride_meta = None
                    if pride_enabled:
                        # deterministic seed per subject for reproducibility
                        seed = _stable_u32_seed(str(subject), int(getattr(args, "pride_seed", 0)))
                        pride_prior, pride_meta = _estimate_pride_prior_random_prefix_mean(
                            per_sample_probs=per_sample_probs,
                            cyclic_indices=cyclic_indices,
                            k=k,
                            prefix_ratio=float(getattr(args, "pride_prefix_ratio", 0.02)),
                            seed=seed,
                        )
                        prefix_ids_set = set(int(x) for x in (pride_meta.get("prefix_ids") or []))
                        logger.info(_purple(f"==== PRIDE prior estimated (random prefix {pride_meta.get('m')}/{pride_meta.get('N')}) ===="))
                        logger.info(f"prior: {[float(x) for x in np.asarray(pride_prior, dtype=np.float64).tolist()]}")

                        # Recall test: base-only predictions with debiased base row (argmax)
                        try:
                            base_labels = [option_ids.index(str(x)) for x in ideals]
                            base_preds = []
                            for i in range(len(per_sample_probs)):
                                ps = np.asarray(per_sample_probs[i], dtype=np.float64)
                                base_row = np.asarray(ps[identity_idx], dtype=np.float64)
                                base_row_corr = _pride_correct_row(base_row, pride_prior)
                                base_preds.append(int(np.argmax(base_row_corr)))
                            rstd = _recall_std(base_labels, base_preds, k=k)
                            logger.info(_purple(f"PRIDE recall_std (base-only, over labels): {rstd:.4f}"))
                            pride_recall_std_records.append({"subject": str(subject), "rstd": float(rstd), "m": int(pride_meta.get("m", 0)), "N": int(pride_meta.get("N", 0))})
                        except Exception:
                            pass

                        # build debiased correctness + gaps (same pipeline, but on corrected probs)
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

                            # cyclic ensemble (k rotations)
                            cyc_probs_corr = [ps_corr[idx] for idx in cyclic_indices]
                            agg_cyc_corr = _aggregate_probs_over_permutations([cp.tolist() for cp in cyc_probs_corr], cyc_perms, k)
                            pred_cyc_corr = option_ids[int(np.argmax(agg_cyc_corr))]
                            cyclic_pred_idx_list_pr.append(int(np.argmax(agg_cyc_corr)))
                            corr_cyc_corr = (pred_cyc_corr == ideals[i])
                            cyclic_correct_list_pr.append(corr_cyc_corr)

                            # base (identity)
                            base_row_corr = np.asarray(ps_corr[identity_idx], dtype=np.float64)
                            pred_base_corr = option_ids[int(np.argmax(base_row_corr))]
                            base_pred_idx_list_pr.append(int(np.argmax(base_row_corr)))
                            corr_base_corr = (pred_base_corr == ideals[i])
                            base_correct_list_pr.append(corr_base_corr)

                            # full (if available)
                            if full_enabled:
                                agg_full_corr = _aggregate_probs_over_permutations(ps_corr, perm_list, k)
                                pred_full_idx_pr = int(np.argmax(agg_full_corr))
                                full_pred_idx_list_pr.append(pred_full_idx_pr)
                                full_correct_list_pr.append(option_ids[pred_full_idx_pr] == ideals[i])

                            # gaps + probe2
                            vals = np.sort(base_row_corr)[::-1]
                            top1 = float(vals[0]) if vals.shape[0] > 0 else 0.0
                            top2 = float(vals[1]) if vals.shape[0] > 1 else 0.0
                            default_conf_pr.append(top1 - top2)

                            shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(base_row_corr, k)
                            probe_perm_idx = cyclic_indices[shift]

                            agg_base = _aggregate_probs_over_permutations([base_row_corr.tolist()], [tuple(range(k))], k)
                            probe_row_corr = np.asarray(ps_corr[probe_perm_idx], dtype=np.float64)
                            agg_probe = _aggregate_probs_over_permutations([probe_row_corr.tolist()], [cyc_perms[shift]], k)

                            mean_probs = (np.asarray(agg_base, dtype=np.float64) + np.asarray(agg_probe, dtype=np.float64)) / 2.0
                            vals_mean = np.sort(mean_probs)[::-1]
                            mean_gap = float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0
                            mean_gap_list_pr.append(mean_gap)

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

                    logger.info(_orange(f"Derived and saved cyclic results: {subject}"))
                    logger.info(_orange(f"Derived and saved base results: {subject}"))
                    # (optional) verbose summary
                    if bool(getattr(args, "verbose", False)):
                        if full_enabled:
                            logger.info(_purple(f"[{subject}] Accuracies — Full: {full_acc:.4f}, Cyclic: {cyclic_acc:.4f}, Default: {base_acc:.4f}"))
                        else:
                            logger.info(_purple(f"[{subject}] Accuracies — Full: (disabled), Cyclic: {cyclic_acc:.4f}, Default: {base_acc:.4f}"))

                    # ---------- compute & save REAL-WORLD online curves (baseline; single point per policy) ----------
                    curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                    if getattr(args, 'option_id_set', None):
                        curve_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(curve_save_path, exist_ok=True)

                    perc_src = getattr(args, "ours_low_conf_percent_list", None)
                    if perc_src is None or (isinstance(perc_src, str) and perc_src.strip() == ""):
                        perc_src = "5,10,20,30"
                    perc_list = _parse_percent_value_list(perc_src)
                    curve_objs_baseline = []
                    baseline_by_p = {}
                    curve_objs_pride = []

                    for perc in perc_list:
                        perc = float(perc)
                        labels_idx_for_curves = [option_ids.index(str(x)) for x in ideals]
                        cobj = _compute_curves_for_one_percentile(
                            subject=subject,
                            tag="baseline",
                            k=k,
                            perm_list=perm_list,
                            base_correct_list=base_correct_list,
                            cyclic_correct_list=cyclic_correct_list,
                            full_correct_list=full_correct_list if full_enabled else [],
                            default_conf=default_conf,
                            mean_conf=mean_conf,
                            flip_trigger=arr_flip_trigger,
                            probe2_correct=arr_probe2_correct,
                            perc_value=perc,
                            full_enabled=bool(full_enabled),
                            labels_idx=labels_idx_for_curves,
                            base_pred_idx=base_pred_idx_list,
                            cyclic_pred_idx=cyclic_pred_idx_list,
                            probe2_pred_idx=probe2_pred_idx_list,
                            full_pred_idx=full_pred_idx_list if full_enabled and len(full_pred_idx_list) == len(ideals) else None,
                        )
                        if cobj:
                            curve_objs_baseline.append(cobj)
                            baseline_by_p[perc] = cobj
                            _log_baseline_report(cobj)

                            # aggregate derived policy report over subjects (per p)
                            derived_records_by_p.setdefault(float(perc), []).append(cobj)

                            # PRIDE+OURS (debiased probs) for the same p
                            if pride_enabled and pride_prior is not None:
                                cobj_pr = _compute_curves_for_one_percentile(
                                    subject=subject,
                                    tag="pride_mix",
                                    k=k,
                                    perm_list=perm_list,
                                    base_correct_list=base_correct_list_pr,
                                    cyclic_correct_list=cyclic_correct_list_pr,
                                    full_correct_list=full_correct_list_pr if full_enabled else [],
                                    default_conf=default_conf_pr,
                                    mean_conf=mean_conf_pr,
                                    flip_trigger=arr_flip_trigger_pr,
                                    probe2_correct=arr_probe2_correct_pr,
                                    perc_value=perc,
                                    full_enabled=bool(full_enabled),
                                    forced_cyclic_ids=prefix_ids_set,
                                    labels_idx=labels_idx_for_curves,
                                    base_pred_idx=base_pred_idx_list_pr,
                                    cyclic_pred_idx=cyclic_pred_idx_list_pr,
                                    probe2_pred_idx=probe2_pred_idx_list_pr,
                                    full_pred_idx=full_pred_idx_list_pr if full_enabled and len(full_pred_idx_list_pr) == len(ideals) else None,
                                )
                                if cobj_pr:
                                    curve_objs_pride.append(cobj_pr)
                                    # For fair comparison: keep ensemble baselines (cyclic/full) identical to BASELINE.
                                    # PRIDE+OURS is meant to debias before running our *policies*; the "cyclic/full(ensemble)"
                                    # lines are anchors and should not move across PRIDE on/off.
                                    try:
                                        if isinstance(cobj, dict) and isinstance(cobj_pr, dict):
                                            if "always" in cobj and "always" in cobj_pr:
                                                if "cyclic" in cobj["always"]:
                                                    cobj_pr["always"]["cyclic"] = dict(cobj["always"]["cyclic"])
                                                if "full" in (cobj["always"] or {}) and "full" in (cobj_pr["always"] or {}):
                                                    cobj_pr["always"]["full"] = dict(cobj["always"]["full"])
                                            if "cyclic" in cobj:
                                                cobj_pr["cyclic"] = dict(cobj["cyclic"])
                                            if "full" in cobj and "full" in cobj_pr:
                                                cobj_pr["full"] = dict(cobj["full"])
                                    except Exception:
                                        pass
                                    _log_named_report("PRIDE+OURS", cobj_pr)
                                    derived_records_pride_by_p.setdefault(float(perc), []).append(cobj_pr)
                                    # NOTE: heuristic_points for PRIDE+OURS are filled below (after baseline_points helpers are defined)

                            # ---- Recall std vs p (BASELINE vs PRIDE) — macro over subjects (MMLU-friendly) ----
                            try:
                                labels_idx = [option_ids.index(str(x)) for x in ideals]

                                # BASELINE: default / cyclic / full (always policies)
                                recall_std_vs_p_records.append({
                                    "subject": str(subject), "p": float(perc), "method": "default", "kind": "BASELINE",
                                    "rstd": float(_recall_std(labels_idx, base_pred_idx_list, k=k))
                                })
                                recall_std_vs_p_records.append({
                                    "subject": str(subject), "p": float(perc), "method": "cyclic", "kind": "BASELINE",
                                    "rstd": float(_recall_std(labels_idx, cyclic_pred_idx_list, k=k))
                                })
                                if full_enabled and len(full_pred_idx_list) == len(labels_idx):
                                    recall_std_vs_p_records.append({
                                        "subject": str(subject), "p": float(perc), "method": "full", "kind": "BASELINE",
                                        "rstd": float(_recall_std(labels_idx, full_pred_idx_list, k=k))
                                    })

                                # BASELINE: cyclic_random (10, 20, ..., 100%)
                                seed_base = _stable_u32_seed(str(subject), 0)
                                for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                                    _, _, preds_r = _run_cyclic_random_fraction_with_preds(
                                        base_pred_idx_list, cyclic_pred_idx_list, labels_idx, k, fp, seed_base + int(fp)
                                    )
                                    recall_std_vs_p_records.append({"subject": str(subject), "p": float(perc), "method": f"cyclic_random_{fp}", "kind": "BASELINE", "rstd": float(_recall_std(labels_idx, preds_r, k=k))})

                                # BASELINE: heuristic points (th1/2, Online Sqrt All only)
                                _, _, _, preds_h1 = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                    default_conf=default_conf,
                                    mean_conf=mean_conf,
                                    base_pred_idx=base_pred_idx_list,
                                    cyclic_pred_idx=cyclic_pred_idx_list,
                                    probe2_pred_idx=probe2_pred_idx_list,
                                    labels_idx=labels_idx,
                                    k=k,
                                    th1_percent=float(perc),
                                    th2_rule_from_th1_value=lambda x: x / 2.0,
                                    forced_cyclic_ids=None,
                                )
                                recall_std_vs_p_records.append({"subject": str(subject), "p": float(perc), "method": "th1/2", "kind": "BASELINE", "rstd": float(_recall_std(labels_idx, preds_h1, k=k))})

                                _, _, preds_sqrt = _run_online_sqrt_policy_with_preds(
                                    default_conf=default_conf,
                                    mean_conf=mean_conf,
                                    base_pred_idx=base_pred_idx_list,
                                    cyclic_pred_idx=cyclic_pred_idx_list,
                                    probe2_pred_idx=probe2_pred_idx_list,
                                    labels_idx=labels_idx,
                                    k=k,
                                    th1_percent=float(perc),
                                    forced_cyclic_ids=None,
                                )
                                recall_std_vs_p_records.append({"subject": str(subject), "p": float(perc), "method": "Online Sqrt (All)", "kind": "BASELINE", "rstd": float(_recall_std(labels_idx, preds_sqrt, k=k))})

                                # PRIDE versions (if enabled)
                                if pride_enabled and pride_prior is not None:
                                    # PRIDE: default / cyclic (always)
                                    recall_std_vs_p_records.append({
                                        "subject": str(subject), "p": float(perc), "method": "default", "kind": "PRIDE+OURS",
                                        "rstd": float(_recall_std(labels_idx, base_pred_idx_list_pr, k=k))
                                    })
                                    recall_std_vs_p_records.append({
                                        "subject": str(subject), "p": float(perc), "method": "cyclic", "kind": "PRIDE+OURS",
                                        "rstd": float(_recall_std(labels_idx, cyclic_pred_idx_list_pr, k=k))
                                    })
                                    if full_enabled and len(full_pred_idx_list_pr) == len(labels_idx):
                                        recall_std_vs_p_records.append({
                                            "subject": str(subject), "p": float(perc), "method": "full", "kind": "PRIDE+OURS",
                                            "rstd": float(_recall_std(labels_idx, full_pred_idx_list_pr, k=k))
                                        })

                                    # PRIDE: cyclic_random (10, 20, ..., 100%)
                                    seed_base_pr = _stable_u32_seed(str(subject), 0)
                                    for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                                        _, _, preds_r_pr = _run_cyclic_random_fraction_with_preds(
                                            base_pred_idx_list_pr, cyclic_pred_idx_list_pr, labels_idx, k, fp, seed_base_pr + int(fp)
                                        )
                                        recall_std_vs_p_records.append({"subject": str(subject), "p": float(perc), "method": f"cyclic_random_{fp}", "kind": "PRIDE+OURS", "rstd": float(_recall_std(labels_idx, preds_r_pr, k=k))})

                                    # PRIDE: heuristic points (th1/2, Online Sqrt All only)
                                    _, _, _, preds_h1_pr = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                        default_conf=default_conf_pr,
                                        mean_conf=mean_conf_pr,
                                        base_pred_idx=base_pred_idx_list_pr,
                                        cyclic_pred_idx=cyclic_pred_idx_list_pr,
                                        probe2_pred_idx=probe2_pred_idx_list_pr,
                                        labels_idx=labels_idx,
                                        k=k,
                                        th1_percent=float(perc),
                                        th2_rule_from_th1_value=lambda x: x / 2.0,
                                        forced_cyclic_ids=prefix_ids_set,
                                    )
                                    recall_std_vs_p_records.append({"subject": str(subject), "p": float(perc), "method": "th1/2", "kind": "PRIDE+OURS", "rstd": float(_recall_std(labels_idx, preds_h1_pr, k=k))})

                                    _, _, preds_sqrt_pr = _run_online_sqrt_policy_with_preds(
                                        default_conf=default_conf_pr,
                                        mean_conf=mean_conf_pr,
                                        base_pred_idx=base_pred_idx_list_pr,
                                        cyclic_pred_idx=cyclic_pred_idx_list_pr,
                                        probe2_pred_idx=probe2_pred_idx_list_pr,
                                        labels_idx=labels_idx,
                                        k=k,
                                        th1_percent=float(perc),
                                        forced_cyclic_ids=prefix_ids_set,
                                    )
                                    recall_std_vs_p_records.append({"subject": str(subject), "p": float(perc), "method": "Online Sqrt (All)", "kind": "PRIDE+OURS", "rstd": float(_recall_std(labels_idx, preds_sqrt_pr, k=k))})
                            except Exception:
                                pass

                            # [ADD] Baseline Point Plot with 3 Rules
                            ptag = f"p{int(round(perc))}"
                            out_pts = os.path.join(curve_save_path, f"{subject}_{ptag}_baseline_points.png")
                            
                            # 3가지 룰 포인트 계산
                            th1p = float(perc)
                            extra_pts = []
                            
                            # Helper to calc static rule points for baseline plot (cost, acc, stats, recall)
                            def _get_static_pt(th1_p, rule_func, label, marker, color):
                                c, a, th2p, st = _run_online_th1_quantile_th2_from_th1_rule_with_stats(
                                    default_conf=default_conf,
                                    mean_conf=mean_conf,
                                    base_correct=base_correct_list,
                                    cyclic_correct=cyclic_correct_list,
                                    probe2_correct=arr_probe2_correct,
                                    k=k,
                                    th1_percent=float(th1_p),
                                    th2_rule_from_th1_value=rule_func,
                                )
                                out = {'cost': c, 'acc': a, 'th2_p': float(th2p), 'label': label, 'marker': marker, 'color': color,
                                       'n_base': int(st.get('n_base', 0)), 'n_probe2': int(st.get('n_probe2', 0)), 'n_cyclic': int(st.get('n_cyclic', 0))}
                                try:
                                    _, _, _, preds = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                        default_conf, mean_conf, base_pred_idx_list, cyclic_pred_idx_list, probe2_pred_idx_list,
                                        labels_idx, k, float(th1_p), rule_func, None
                                    )
                                    out['recall_std'] = float(_recall_std(labels_idx, preds, k))
                                except Exception:
                                    pass
                                return out

                            # Helper for PRIDE+OURS heuristic points (same rules, but debiased stats + prefix overhead)
                            def _get_static_pt_pride(th1_p, rule_func, label, marker, color):
                                c, a, th2p, st = _run_online_th1_quantile_th2_from_th1_rule_with_stats(
                                    default_conf=default_conf_pr,
                                    mean_conf=mean_conf_pr,
                                    base_correct=base_correct_list_pr,
                                    cyclic_correct=cyclic_correct_list_pr,
                                    probe2_correct=arr_probe2_correct_pr,
                                    k=k,
                                    th1_percent=float(th1_p),
                                    th2_rule_from_th1_value=rule_func,
                                    forced_cyclic_ids=prefix_ids_set,
                                )
                                out = {'cost': c, 'acc': a, 'th2_p': float(th2p), 'label': label, 'marker': marker, 'color': color,
                                       'n_base': int(st.get('n_base', 0)), 'n_probe2': int(st.get('n_probe2', 0)), 'n_cyclic': int(st.get('n_cyclic', 0))}
                                try:
                                    _, _, _, preds = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                        default_conf_pr, mean_conf_pr, base_pred_idx_list_pr, cyclic_pred_idx_list_pr, probe2_pred_idx_list_pr,
                                        labels_idx, k, float(th1_p), rule_func, prefix_ids_set
                                    )
                                    out['recall_std'] = float(_recall_std(labels_idx, preds, k))
                                except Exception:
                                    pass
                                return out

                            # 1. Static Heuristic th1/2
                            extra_pts.append(_get_static_pt(th1p, lambda x: x / 2.0, 'th1/2', '*', 'gray'))

                            # 2. [ONLINE Sqrt All]
                            def _sqrt_pt(c_sqrt, a_sqrt, p_sqrt, st, label, marker):
                                out = {'cost': c_sqrt, 'acc': a_sqrt, 'th2_p': float(p_sqrt), 'label': label, 'marker': marker, 'color': 'orange',
                                       'n_base': int(st.get('n_base', 0)), 'n_probe2': int(st.get('n_probe2', 0)), 'n_cyclic': int(st.get('n_cyclic', 0))}
                                return out
                            c_sqrt, a_sqrt, p_sqrt, st_sqrt = _run_online_sqrt_policy_with_stats(
                                default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1_percent=perc
                            )
                            sq_pt = _sqrt_pt(c_sqrt, a_sqrt, p_sqrt, st_sqrt, 'Online Sqrt (All)', 'D')
                            try:
                                _, _, preds_sqrt = _run_online_sqrt_policy_with_preds(
                                    default_conf, mean_conf, base_pred_idx_list, cyclic_pred_idx_list, probe2_pred_idx_list,
                                    labels_idx, k, perc, None
                                )
                                sq_pt['recall_std'] = float(_recall_std(labels_idx, preds_sqrt, k))
                            except Exception:
                                pass
                            extra_pts.append(sq_pt)

                            # Save heuristic points into curve_obj for aggregate reporting (incl. n_base, n_probe2, n_cyclic, recall_std)
                            try:
                                def _hp_entry(hp):
                                    e = {"label": str(hp.get("label")), "cost": float(hp.get("cost")), "acc": float(hp.get("acc")),
                                         "th2_p": float(hp.get("th2_p", float("nan"))), "marker": str(hp.get("marker", "o")), "color": str(hp.get("color", "black"))}
                                    for k in ["n_base", "n_probe2", "n_cyclic", "recall_std"]:
                                        if k in hp and hp[k] is not None:
                                            e[k] = int(hp[k]) if k.startswith("n_") else float(hp[k])
                                    return e
                                cobj["heuristic_points"] = [_hp_entry(hp) for hp in (extra_pts or [])]
                            except Exception:
                                pass

                            # PRIDE+OURS: attach heuristic points + overlay plot
                            if pride_enabled and pride_prior is not None and 'cobj_pr' in locals() and cobj_pr:
                                extra_pts_pr = []
                                extra_pts_pr.append(_get_static_pt_pride(th1p, lambda x: x / 2.0, 'th1/2', '*', 'gray'))

                                c_sqrt_pr, a_sqrt_pr, p_sqrt_pr, st_sqrt_pr = _run_online_sqrt_policy_with_stats(
                                    default_conf_pr, mean_conf_pr, base_correct_list_pr, cyclic_correct_list_pr, arr_probe2_correct_pr,
                                    k, th1_percent=perc, forced_cyclic_ids=prefix_ids_set
                                )
                                sq_pr_pt = _sqrt_pt(c_sqrt_pr, a_sqrt_pr, p_sqrt_pr, st_sqrt_pr, 'Online Sqrt (All)', 'D')
                                try:
                                    _, _, preds_sqrt_pr = _run_online_sqrt_policy_with_preds(
                                        default_conf_pr, mean_conf_pr, base_pred_idx_list_pr, cyclic_pred_idx_list_pr, probe2_pred_idx_list_pr,
                                        labels_idx, k, perc, prefix_ids_set
                                    )
                                    sq_pr_pt['recall_std'] = float(_recall_std(labels_idx, preds_sqrt_pr, k))
                                except Exception:
                                    pass
                                extra_pts_pr.append(sq_pr_pt)

                                try:
                                    cobj_pr["heuristic_points"] = [_hp_entry(hp) for hp in (extra_pts_pr or [])]
                                except Exception:
                                    pass

                                out_cmp = os.path.join(curve_save_path, f"{subject}_{ptag}_baseline_vs_pride_points.png")
                                _plot_baseline_vs_pride_points_scatter(
                                    baseline_obj=cobj,
                                    pride_obj=cobj_pr,
                                    out_path=out_cmp,
                                    title=f"{args.task} {subject} — Baseline vs PRIDE+OURS (REAL-WORLD online, {ptag})",
                                )
                                if wandb_ok and wandb_run is not None:
                                    try:
                                        import wandb
                                        wandb_run.log({f"plots/{subject}/{ptag}/baseline_vs_pride_points": wandb.Image(out_cmp)})
                                    except Exception:
                                        pass

                            # [ADD] log heuristic point performances (cost, acc, n_base, n_probe2, n_cyclic, recall_std)
                            def _hp_log_line(hp):
                                acc = float(hp.get("acc", float("nan")))
                                cost = float(hp.get("cost", float("nan")))
                                line = f"cost={cost:.3f}, acc={acc:.4f}"
                                nb, np2, nc = hp.get("n_base"), hp.get("n_probe2"), hp.get("n_cyclic")
                                if nb is not None and np2 is not None and nc is not None:
                                    line += f", n_base={int(nb)}, n_probe2={int(np2)}, n_cyclic={int(nc)}"
                                rstd = hp.get("recall_std")
                                if isinstance(rstd, (int, float)):
                                    line += f", recall_std={rstd:.4f}"
                                return line

                            try:
                                base_acc0 = float(np.mean(np.asarray(base_correct_list, dtype=np.float64))) if len(base_correct_list) else float("nan")
                                logger.info(_purple(f"==== HEURISTIC Point report (p={int(round(perc))}) ===="))
                                logger.info(f"{'default':<18}: cost=1.000, acc={base_acc0:.4f}")
                                for hp in extra_pts:
                                    logger.info(f"{hp.get('label','?'):<18}: {_hp_log_line(hp)}")
                            except Exception:
                                pass

                            # PRIDE+OURS heuristic point report (same p)
                            if pride_enabled and pride_prior is not None and 'cobj_pr' in locals() and cobj_pr and isinstance(cobj_pr, dict):
                                try:
                                    logger.info(_purple(f"==== PRIDE+OURS HEURISTIC Point report (p={int(round(perc))}) ===="))
                                    for hp in (cobj_pr.get("heuristic_points", []) or []):
                                        logger.info(f"{hp.get('label','?'):<18}: {_hp_log_line(hp)}")
                                except Exception:
                                    pass

                            # Δ plot/log (PRIDE+OURS - BASELINE) including heuristics (verbose only)
                            if bool(getattr(args, "verbose", False)) and pride_enabled and pride_prior is not None and 'cobj_pr' in locals() and cobj_pr and isinstance(cobj_pr, dict):
                                try:
                                    delta_pts = []
                                    # policies
                                    cyclic_random_keys = [f"cyclic_random_{fp}" for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
                                    for key in cyclic_random_keys:
                                        if key in cobj and key in cobj_pr:
                                            dcost = float(cobj_pr[key]["costs"][0]) - float(cobj[key]["costs"][0])
                                            dacc = float(cobj_pr[key]["accuracies"][0]) - float(cobj[key]["accuracies"][0])
                                            delta_pts.append({"label": key, "dcost": dcost, "dacc": dacc, "marker": "o", "color": "black"})

                                    # heuristics
                                    bmap = {str(h.get("label")): h for h in (cobj.get("heuristic_points", []) or []) if isinstance(h, dict)}
                                    pmap = {str(h.get("label")): h for h in (cobj_pr.get("heuristic_points", []) or []) if isinstance(h, dict)}
                                    for lab in sorted(set(bmap.keys()) & set(pmap.keys())):
                                        dcost = float(pmap[lab].get("cost")) - float(bmap[lab].get("cost"))
                                        dacc = float(pmap[lab].get("acc")) - float(bmap[lab].get("acc"))
                                        delta_pts.append({
                                            "label": lab,
                                            "dcost": dcost,
                                            "dacc": dacc,
                                            "marker": str(pmap[lab].get("marker", "o")),
                                            "color": str(pmap[lab].get("color", "black")),
                                        })

                                    logger.info(_purple(f"==== DELTA report (PRIDE+OURS - BASELINE, p={int(round(perc))}) ===="))
                                    for dp in delta_pts:
                                        logger.info(f"{dp['label']:<22}: Δcost={dp['dcost']:+.3f}, Δacc={dp['dacc']:+.4f}")

                                    out_d = os.path.join(curve_save_path, f"{subject}_{ptag}_pride_delta_scatter.png")
                                    _plot_delta_scatter(
                                        delta_points=delta_pts,
                                        out_path=out_d,
                                        title=f"{args.task} {subject} — Δ (PRIDE+OURS - BASELINE), {ptag}",
                                    )
                                    if wandb_ok and wandb_run is not None:
                                        try:
                                            import wandb
                                            wandb_run.log({f"plots/{subject}/{ptag}/pride_delta_scatter": wandb.Image(out_d)})
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                            # 3. [ONLINE DYNAMIC] REMOVED
                            # c_dyn, a_dyn, _ = _run_online_dynamic_policy(
                            #     default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k
                            # )
                            # extra_pts.append({'cost': c_dyn, 'acc': a_dyn, 'label': 'Online Dynamic', 'marker': 'H', 'color': 'purple'})

                            _plot_baseline_points_scatter(
                                curve_obj=cobj,
                                out_path=out_pts,
                                title=f"{args.task} {subject} — Baseline Policies (REAL-WORLD online, {ptag}, Heuristics 6)",
                                extra_points=extra_pts
                            )
                            if wandb_ok and wandb_run is not None:
                                try:
                                    import wandb
                                    wandb_run.log({f"plots/{subject}/{ptag}/baseline_points": wandb.Image(out_pts)})
                                except Exception:
                                    pass

                    save_results(f'{curve_save_path}/{subject}_curve.jsonl', curve_objs_baseline, metrics=None)
                    if pride_enabled and len(curve_objs_pride) > 0:
                        save_results(f'{curve_save_path}/{subject}_pride_curve.jsonl', curve_objs_pride, metrics=None)

                    # =========================================================
                    # th2 trade-off plot: th1 (5, 10, 20, 30)에 대해 각각 th2 (5, 10, 20, 30) curve
                    # =========================================================
                    th1_tradeoff_list = _parse_percent_value_list(
                        getattr(args, "ours_th1_tradeoff", "5,10,20,30")
                    )
                    th2_tradeoff_list = _parse_percent_value_list(
                        getattr(args, "ours_th2_tradeoff", ",".join([str(i) for i in range(1, 31)]))
                    )
                    if len(th1_tradeoff_list) > 0 and len(th2_tradeoff_list) > 0 and len(per_sample_probs) > 0:
                        _compute_and_plot_th2_tradeoff(
                            subject=subject,
                            curve_save_path=curve_save_path,
                            th1_list=th1_tradeoff_list,
                            th2_list=th2_tradeoff_list,
                            default_conf=default_conf,
                            mean_conf=mean_conf,
                            base_correct_list=base_correct_list,
                            cyclic_correct_list=cyclic_correct_list,
                            arr_probe2_correct=arr_probe2_correct,
                            k=k,
                            args=args,
                            wandb_ok=wandb_ok,
                            wandb_run=wandb_run,
                            plot_tag="BASELINE",
                            fname_tag="BASE",
                            forced_cyclic_ids=None,
                        )

                        # PRIDE-only curves (debiased stats) + overlay compare
                        if pride_enabled and pride_prior is not None:
                            _compute_and_plot_th2_tradeoff(
                                subject=subject,
                                curve_save_path=curve_save_path,
                                th1_list=th1_tradeoff_list,
                                th2_list=th2_tradeoff_list,
                                default_conf=default_conf_pr,
                                mean_conf=mean_conf_pr,
                                base_correct_list=base_correct_list_pr,
                                cyclic_correct_list=cyclic_correct_list_pr,
                                arr_probe2_correct=arr_probe2_correct_pr,
                                k=k,
                                args=args,
                                wandb_ok=wandb_ok,
                                wandb_run=wandb_run,
                                plot_tag="PRIDE+OURS (debiased)",
                                fname_tag="PRIDE",
                                forced_cyclic_ids=prefix_ids_set,
                            )
                            _plot_th2_tradeoff_curve_compare(
                                subject=subject,
                                curve_save_path=curve_save_path,
                                th1_list=th1_tradeoff_list,
                                default_conf_base=default_conf,
                                mean_conf_base=mean_conf,
                                base_correct_base=base_correct_list,
                                cyclic_correct_base=cyclic_correct_list,
                                probe2_correct_base=arr_probe2_correct,
                                default_conf_pr=default_conf_pr,
                                mean_conf_pr=mean_conf_pr,
                                base_correct_pr=base_correct_list_pr,
                                cyclic_correct_pr=cyclic_correct_list_pr,
                                probe2_correct_pr=arr_probe2_correct_pr,
                                k=k,
                                args=args,
                                wandb_ok=wandb_ok,
                                wandb_run=wandb_run,
                                forced_cyclic_ids_pr=prefix_ids_set,
                            )

                except Exception as e:
                    logger.warning(f"Failed to derive curves for subject '{subject}': {e}")
                    import traceback
                    traceback.print_exc()

            logging_cuda_memory_usage()

        # ---------- end subjects loop: print aggregate summary ----------
        if len(eval_acc_records) > 0:
            macro_acc = float(np.mean([float(r["acc"]) for r in eval_acc_records]))
            micro_total = int(np.sum([int(r["total"]) for r in eval_acc_records]))
            micro_corrects = int(np.sum([int(r["corrects"]) for r in eval_acc_records]))
            micro_acc = (float(micro_corrects) / float(micro_total)) if micro_total > 0 else float("nan")
            logger.info(_purple(f"==== AGGREGATE report over subjects ({args.task}, setting={args.setting}) ===="))
            logger.info(f"subjects: {len(eval_acc_records)}/{len(subjects)}")
            if len(eval_acc_records) <= 1:
                logger.info(f"accuracy: {macro_acc:.4f}")
            else:
                logger.info(f"accuracy (macro mean over subjects): {macro_acc:.4f}")
                logger.info(f"accuracy (micro = sum correct / sum total): {micro_acc:.4f}")

        # Recall std histogram plot (PRIDE)
        if len(pride_recall_std_records) > 0:
            try:
                out_dir = f"results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full"
                if getattr(args, 'option_id_set', None):
                    out_dir += f"_id-{args.option_id_set}"
                os.makedirs(out_dir, exist_ok=True)
                rstds = [float(r.get("rstd", float("nan"))) for r in pride_recall_std_records]
                rstds = [r for r in rstds if np.isfinite(r)]
                if len(rstds) > 0:
                    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=180)
                    ax.hist(rstds, bins=20, color="#4c78a8", alpha=0.85)
                    ax.set_title(f"{args.task} — PRIDE recall std (base-only debiased), n={len(rstds)}")
                    ax.set_xlabel("recall_std")
                    ax.set_ylabel("count")
                    ax.grid(True, linestyle="--", alpha=0.35)
                    out_png = os.path.join(out_dir, f"{args.task}_aggregate_pride_recall_std_hist.png")
                    fig.tight_layout()
                    fig.savefig(out_png, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(_purple(f"Saved PRIDE recall_std histogram: {out_png}"))
                    if wandb_ok and wandb_run is not None:
                        try:
                            import wandb
                            wandb_run.log({f"plots/{args.task}/aggregate/pride_recall_std_hist": wandb.Image(out_png)})
                        except Exception:
                            pass
            except Exception:
                pass

        # Recall std vs p plot (BASELINE vs PRIDE) — includes heuristics
        if len(recall_std_vs_p_records) > 0:
            try:
                out_dir = f"results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full"
                if getattr(args, 'option_id_set', None):
                    out_dir += f"_id-{args.option_id_set}"
                os.makedirs(out_dir, exist_ok=True)

                methods = sorted(list({str(r.get("method")) for r in recall_std_vs_p_records if r.get("method") is not None}))
                kinds = ["BASELINE", "PRIDE+OURS"]

                # build mean rstd per (method, kind, p) over subjects
                def _mean_rstd(method: str, kind: str, p: float) -> float:
                    vals = [float(r.get("rstd", float("nan"))) for r in recall_std_vs_p_records
                            if str(r.get("method")) == method and str(r.get("kind")) == kind and float(r.get("p")) == float(p)]
                    vals = [v for v in vals if np.isfinite(v)]
                    return float(np.mean(vals)) if len(vals) > 0 else float("nan")

                ps = sorted(list({float(r.get("p")) for r in recall_std_vs_p_records if r.get("p") is not None}))
                if len(ps) > 0 and len(methods) > 0:
                    def _plot_overlay(method_subset: List[str], out_png: str, title_suffix: str, wandb_key: str):
                        if method_subset is None or len(method_subset) == 0:
                            return
                        fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=200)
                        cmap = plt.get_cmap("tab20")
                        method_colors = {m: cmap(i % 20) for i, m in enumerate(method_subset)}

                        for m in method_subset:
                            col = method_colors.get(m, "black")
                            ys_b = [_mean_rstd(m, "BASELINE", p) for p in ps]
                            ys_p = [_mean_rstd(m, "PRIDE+OURS", p) for p in ps]
                            ax.plot(ps, ys_b, color=col, linestyle="-", marker="o", linewidth=2.0, alpha=0.9, label=m)
                            ax.plot(ps, ys_p, color=col, linestyle="--", marker="o", linewidth=2.0, alpha=0.9)

                        # For single-subject tasks (e.g., csqa), "macro" wording is confusing.
                        uniq_subj = sorted(list({str(r.get("subject")) for r in recall_std_vs_p_records if r.get("subject") is not None}))
                        macro_note = "(macro over subjects)" if len(uniq_subj) > 1 else ""
                        ax.set_title(f"{args.task} — Recall std vs p {title_suffix} {macro_note}".strip())
                        ax.set_xlabel("p (th1 percentile)")
                        ax.set_ylabel("recall_std")
                        ax.grid(True, linestyle="--", alpha=0.35)
                        ax.set_xticks(ps)
                        ax.set_xticklabels([f"p{int(round(p))}" for p in ps])

                        # Legend for methods (colors)
                        leg1 = ax.legend(title="Method (color)", loc="upper left", fontsize=8, title_fontsize=9, ncol=2)
                        ax.add_artist(leg1)

                        # Legend for line styles (meaning)
                        try:
                            from matplotlib.lines import Line2D
                            style_handles = [
                                Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="BASELINE (solid)"),
                                Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="PRIDE+OURS (dashed)"),
                            ]
                            ax.legend(handles=style_handles, title="Line style", loc="upper right", fontsize=9, title_fontsize=9)
                        except Exception:
                            pass

                        fig.tight_layout()
                        fig.savefig(out_png, bbox_inches="tight")
                        plt.close(fig)
                        logger.info(_purple(f"Saved recall_std vs p overlay plot: {out_png}"))
                        if wandb_ok and wandb_run is not None:
                            try:
                                import wandb
                                wandb_run.log({wandb_key: wandb.Image(out_png)})
                            except Exception:
                                pass

                    cyclic_random_list = [f"cyclic_random_{fp}" for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
                    policy_methods = [m for m in methods if m in {"default", "cyclic", "full"} or m in cyclic_random_list]
                    # Heuristics plot: default, cyclic + all heuristic points (MMLU/ARC/CSQA)
                    heur_methods = [m for m in methods if m in {"th1/2", "Online Sqrt (All)"}]

                    out_png_pol = os.path.join(out_dir, f"{args.task}_aggregate_recall_std_vs_p_overlay_POLICIES.png")
                    _plot_overlay(policy_methods, out_png_pol, "(Policies)", f"plots/{args.task}/aggregate/recall_std_vs_p_overlay_policies")

                    out_png_heur = os.path.join(out_dir, f"{args.task}_aggregate_recall_std_vs_p_overlay_HEURISTICS.png")
                    _plot_overlay(heur_methods, out_png_heur, "(default, cyclic + Heuristics)", f"plots/{args.task}/aggregate/recall_std_vs_p_overlay_heuristics")

                    # Δ recall_std (OURS without PRIDE - default) bar plot — always (pride 안 붙인 경우에도)
                    # reference = default (baseline anchor); delta = recall_std(method) - recall_std(default)
                    rstd_delta_from_default_by_p: Dict[float, Dict[str, float]] = {}
                    for p in ps:
                        rstd_delta_from_default_by_p[float(p)] = {}
                        r_default = _mean_rstd("default", "BASELINE", p)
                        for m in methods:
                            if m == "default":
                                continue
                            r_b = _mean_rstd(m, "BASELINE", p)
                            if np.isfinite(r_b) and np.isfinite(r_default):
                                rstd_delta_from_default_by_p[float(p)][m] = float(r_b - r_default)
                            elif np.isfinite(r_b):
                                rstd_delta_from_default_by_p[float(p)][m] = float(r_b)
                    rstd_delta_pol = {p: {m: v for m, v in (d or {}).items() if m in set(policy_methods)} for p, d in rstd_delta_from_default_by_p.items()}
                    rstd_delta_heur = {p: {m: v for m, v in (d or {}).items() if m in set(heur_methods)} for p, d in rstd_delta_from_default_by_p.items()}
                    if any(len(d) > 0 for d in rstd_delta_pol.values()):
                        out_rstd_pol = os.path.join(out_dir, f"{args.task}_aggregate_recall_std_delta_from_default_POLICIES.png")
                        _plot_delta_cost_bars_by_p(rstd_delta_pol, out_rstd_pol, f"{args.task} — Δ recall_std (OURS - default) (Policies)", ylabel="Δ recall_std (vs default)")
                        logger.info(_purple(f"Saved Δ recall_std bar plot (OURS vs default, policies): {out_rstd_pol}"))
                        if wandb_ok and wandb_run is not None:
                            try:
                                import wandb
                                wandb_run.log({f"plots/{args.task}/aggregate/recall_std_delta_from_default_policies": wandb.Image(out_rstd_pol)})
                            except Exception:
                                pass
                    if any(len(d) > 0 for d in rstd_delta_heur.values()):
                        out_rstd_heur = os.path.join(out_dir, f"{args.task}_aggregate_recall_std_delta_from_default_HEURISTICS.png")
                        _plot_delta_cost_bars_by_p(rstd_delta_heur, out_rstd_heur, f"{args.task} — Δ recall_std (OURS - default) (default, cyclic + Heuristics)", ylabel="Δ recall_std (vs default)")
                        logger.info(_purple(f"Saved Δ recall_std bar plot (OURS vs default, heuristics): {out_rstd_heur}"))
                        if wandb_ok and wandb_run is not None:
                            try:
                                import wandb
                                wandb_run.log({f"plots/{args.task}/aggregate/recall_std_delta_from_default_heuristics": wandb.Image(out_rstd_heur)})
                            except Exception:
                                pass

                    # Δ recall_std (PRIDE+OURS - BASELINE) bar plot (only when PRIDE data exists)
                    has_pride = any(str(r.get("kind")) == "PRIDE+OURS" for r in recall_std_vs_p_records)
                    delta_rstd_by_p: Dict[float, Dict[str, float]] = {}
                    if has_pride:
                        for p in ps:
                            delta_rstd_by_p[float(p)] = {}
                            for m in methods:
                                r_b = _mean_rstd(m, "BASELINE", p)
                                r_p = _mean_rstd(m, "PRIDE+OURS", p)
                                if np.isfinite(r_b) and np.isfinite(r_p):
                                    delta_rstd_by_p[float(p)][m] = float(r_p - r_b)
                    if has_pride:
                        drpol_subset = {p: {m: v for m, v in (d or {}).items() if m in set(policy_methods)} for p, d in delta_rstd_by_p.items()}
                        drheur_subset = {p: {m: v for m, v in (d or {}).items() if m in set(heur_methods)} for p, d in delta_rstd_by_p.items()}
                    if has_pride and any(len(d) > 0 for d in drpol_subset.values()):
                        out_png_drpol = os.path.join(out_dir, f"{args.task}_aggregate_recall_std_delta_POLICIES.png")
                        _plot_delta_cost_bars_by_p(drpol_subset, out_png_drpol, f"{args.task} — Δ recall_std (PRIDE+OURS - BASELINE) (Policies)", ylabel="Δ recall_std")
                        logger.info(_purple(f"Saved Δ recall_std bar plot (policies): {out_png_drpol}"))
                        if wandb_ok and wandb_run is not None:
                            try:
                                import wandb
                                wandb_run.log({f"plots/{args.task}/aggregate/recall_std_delta_policies": wandb.Image(out_png_drpol)})
                            except Exception:
                                pass
                    if has_pride and any(len(d) > 0 for d in drheur_subset.values()):
                        out_png_drheur = os.path.join(out_dir, f"{args.task}_aggregate_recall_std_delta_HEURISTICS.png")
                        _plot_delta_cost_bars_by_p(drheur_subset, out_png_drheur, f"{args.task} — Δ recall_std (PRIDE+OURS - BASELINE) (default, cyclic + Heuristics)", ylabel="Δ recall_std")
                        logger.info(_purple(f"Saved Δ recall_std bar plot (heuristics): {out_png_drheur}"))
                        if wandb_ok and wandb_run is not None:
                            try:
                                import wandb
                                wandb_run.log({f"plots/{args.task}/aggregate/recall_std_delta_heuristics": wandb.Image(out_png_drheur)})
                            except Exception:
                                pass

                    # Δ recall_std (OURS+PRIDE - default) bar plot
                    if has_pride:
                        delta_rstd_pride_vs_default_by_p: Dict[float, Dict[str, float]] = {}
                        for p in ps:
                            r_default = _mean_rstd("default", "BASELINE", p)
                            delta_rstd_pride_vs_default_by_p[float(p)] = {}
                            for m in methods:
                                r_p = _mean_rstd(m, "PRIDE+OURS", p)
                                if np.isfinite(r_p) and np.isfinite(r_default):
                                    delta_rstd_pride_vs_default_by_p[float(p)][m] = float(r_p - r_default)
                        drpd_pol = {p: {m: v for m, v in (d or {}).items() if m in set(policy_methods)} for p, d in delta_rstd_pride_vs_default_by_p.items()}
                        drpd_heur = {p: {m: v for m, v in (d or {}).items() if m in set(heur_methods)} for p, d in delta_rstd_pride_vs_default_by_p.items()}
                        if any(len(d) > 0 for d in drpd_pol.values()):
                            out_drpd_pol = os.path.join(out_dir, f"{args.task}_aggregate_recall_std_delta_pride_vs_default_POLICIES.png")
                            _plot_delta_cost_bars_by_p(drpd_pol, out_drpd_pol, f"{args.task} — Δ recall_std (OURS+PRIDE - default) (Policies)", ylabel="Δ recall_std (vs default)")
                            logger.info(_purple(f"Saved Δ recall_std bar plot (OURS+PRIDE vs default, policies): {out_drpd_pol}"))
                            if wandb_ok and wandb_run is not None:
                                try:
                                    import wandb
                                    wandb_run.log({f"plots/{args.task}/aggregate/recall_std_delta_pride_vs_default_policies": wandb.Image(out_drpd_pol)})
                                except Exception:
                                    pass
                        if any(len(d) > 0 for d in drpd_heur.values()):
                            out_drpd_heur = os.path.join(out_dir, f"{args.task}_aggregate_recall_std_delta_pride_vs_default_HEURISTICS.png")
                            _plot_delta_cost_bars_by_p(drpd_heur, out_drpd_heur, f"{args.task} — Δ recall_std (OURS+PRIDE - default) (Heuristics)", ylabel="Δ recall_std (vs default)")
                            logger.info(_purple(f"Saved Δ recall_std bar plot (OURS+PRIDE vs default, heuristics): {out_drpd_heur}"))
                            if wandb_ok and wandb_run is not None:
                                try:
                                    import wandb
                                    wandb_run.log({f"plots/{args.task}/aggregate/recall_std_delta_pride_vs_default_heuristics": wandb.Image(out_drpd_heur)})
                                except Exception:
                                    pass
            except Exception:
                pass

        # Aggregate derived-policy summary (only available when args.setting == 'full')
        if len(derived_records_by_p) > 0:
            for p, cobjs in sorted(derived_records_by_p.items(), key=lambda t: t[0]):
                cyclic_random_keys = [f"cyclic_random_{fp}" for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
                keys = ["default", "cyclic", "full"] + cyclic_random_keys
                logger.info(_purple(f"==== AGGREGATE Derived policy report (REAL-WORLD online, p={p}) ===="))
                single_subject = (len(cobjs) <= 1)
                for key in keys:
                    accs = []
                    costs = []
                    rstds = []
                    for cobj in cobjs:
                        if key in ["default", "cyclic", "full"]:
                            always = cobj.get("always", {}) or {}
                            if key not in always:
                                continue
                            costs.append(float(always[key]["cost"]))
                            accs.append(float(always[key]["acc"]))
                            rkey = f"{key}_recall_std"
                            if rkey in cobj and isinstance(cobj.get(rkey), (int, float)):
                                rstds.append(float(cobj[rkey]))
                        else:
                            if key not in cobj:
                                continue
                            costs.append(float(cobj[key]["costs"][0]))
                            accs.append(float(cobj[key]["accuracies"][0]))
                            rkey = f"{key}_recall_std"
                            if rkey in cobj and isinstance(cobj.get(rkey), (int, float)):
                                rstds.append(float(cobj[rkey]))

                    if len(accs) == 0:
                        continue
                    cost_macro = float(np.mean(costs))
                    acc_macro = float(np.mean(accs))
                    line = f"{key:<14}: cost≈{cost_macro:.3f}, acc={acc_macro:.4f}"
                    if len(rstds) > 0:
                        line += f", recall_std={float(np.mean(rstds)):.4f}"
                    logger.info(line)

                # Heuristic points (e.g., th1/2, th1/sqrt(k), Online Sqrt) if available
                heuristic_labels = set()
                for cobj in cobjs:
                    for hp in (cobj.get("heuristic_points", []) or []):
                        if isinstance(hp, dict) and hp.get("label") is not None:
                            heuristic_labels.add(str(hp.get("label")))

                if len(heuristic_labels) > 0:
                    logger.info(_purple(f"---- Heuristic points (aggregated over subjects, p={p}) ----"))
                    for lab in sorted(heuristic_labels):
                        costs = []
                        accs = []
                        ws = []
                        n_bases, n_probe2s, n_cyclics, rstds = [], [], [], []
                        for cobj in cobjs:
                            n = int(cobj.get("n_samples", 0)) or 0
                            hp_map = {str(h.get("label")): h for h in (cobj.get("heuristic_points", []) or []) if isinstance(h, dict)}
                            if lab not in hp_map:
                                continue
                            h = hp_map[lab]
                            costs.append(float(h.get("cost", float("nan"))))
                            accs.append(float(h.get("acc", float("nan"))))
                            ws.append(n if n > 0 else 1)
                            if "n_base" in h and h["n_base"] is not None:
                                n_bases.append(int(h["n_base"]))
                            if "n_probe2" in h and h["n_probe2"] is not None:
                                n_probe2s.append(int(h["n_probe2"]))
                            if "n_cyclic" in h and h["n_cyclic"] is not None:
                                n_cyclics.append(int(h["n_cyclic"]))
                            if "recall_std" in h and isinstance(h.get("recall_std"), (int, float)):
                                rstds.append(float(h["recall_std"]))
                        # filter out nans
                        filt = [(c, a, w) for c, a, w in zip(costs, accs, ws) if (not np.isnan(c)) and (not np.isnan(a))]
                        if len(filt) == 0:
                            continue
                        costs_f = [t[0] for t in filt]
                        accs_f = [t[1] for t in filt]
                        cost_macro = float(np.mean(costs_f))
                        acc_macro = float(np.mean(accs_f))
                        line = f"{lab:<14}: cost≈{cost_macro:.3f}, acc={acc_macro:.4f}"
                        if len(n_bases) == len(costs_f) and len(n_probe2s) == len(costs_f) and len(n_cyclics) == len(costs_f):
                            line += f", n_base≈{int(np.mean(n_bases)):.0f}, n_probe2≈{int(np.mean(n_probe2s)):.0f}, n_cyclic≈{int(np.mean(n_cyclics)):.0f}"
                        if len(rstds) > 0:
                            line += f", recall_std={float(np.mean(rstds)):.4f}"
                        logger.info(line)

        # Δacc (OURS without PRIDE - default) bar plot — from BASELINE derived_records
        if len(derived_records_by_p) > 0:
            try:
                out_dir = f"results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full"
                if getattr(args, 'option_id_set', None):
                    out_dir += f"_id-{args.option_id_set}"
                os.makedirs(out_dir, exist_ok=True)
                policy_keys_baseline = ["cyclic", "full"] + [f"cyclic_random_{fp}" for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
                delta_acc_from_default_pol: Dict[float, Dict[str, float]] = {}
                delta_acc_from_default_heur: Dict[float, Dict[str, float]] = {}
                for p, cobjs in sorted(derived_records_by_p.items(), key=lambda t: t[0]):
                    acc_defaults = []
                    for cobj in cobjs:
                        always = cobj.get("always", {}) or {}
                        if "default" in always:
                            acc_defaults.append(float(always["default"]["acc"]))
                    default_acc_mean = float(np.mean(acc_defaults)) if len(acc_defaults) > 0 else float("nan")
                    if not np.isfinite(default_acc_mean):
                        continue
                    delta_acc_from_default_pol[float(p)] = {}
                    for key in policy_keys_baseline:
                        accs = []
                        for cobj in cobjs:
                            if key in ["cyclic", "full"]:
                                always = cobj.get("always", {}) or {}
                                if key in always:
                                    accs.append(float(always[key]["acc"]))
                            elif key in cobj:
                                accs.append(float(cobj[key]["accuracies"][0]))
                        if len(accs) > 0:
                            delta_acc_from_default_pol[float(p)][key] = float(np.mean(accs) - default_acc_mean)
                    heur_labels = set()
                    for cobj in cobjs:
                        for hp in (cobj.get("heuristic_points", []) or []):
                            if isinstance(hp, dict) and hp.get("label") is not None:
                                heur_labels.add(str(hp.get("label")))
                    delta_acc_from_default_heur[float(p)] = {}
                    for lab in heur_labels:
                        accs = []
                        for cobj in cobjs:
                            hp_map = {str(h.get("label")): h for h in (cobj.get("heuristic_points", []) or []) if isinstance(h, dict)}
                            if lab in hp_map and "acc" in hp_map[lab]:
                                accs.append(float(hp_map[lab]["acc"]))
                        if len(accs) > 0:
                            delta_acc_from_default_heur[float(p)][lab] = float(np.mean(accs) - default_acc_mean)
                if any(len(d) > 0 for d in delta_acc_from_default_pol.values()):
                    out_acc_pol = os.path.join(out_dir, f"{args.task}_aggregate_delta_acc_from_default_POLICIES.png")
                    _plot_delta_cost_bars_by_p(delta_acc_from_default_pol, out_acc_pol, f"{args.task} — ΔAcc (OURS - default) (Policies)", ylabel="Δ Accuracy (vs default)")
                    logger.info(_purple(f"Saved Δacc bar plot (OURS vs default, policies): {out_acc_pol}"))
                    if wandb_ok and wandb_run is not None:
                        try:
                            import wandb
                            wandb_run.log({f"plots/{args.task}/aggregate/delta_acc_from_default_policies": wandb.Image(out_acc_pol)})
                        except Exception:
                            pass
                if any(len(d) > 0 for d in delta_acc_from_default_heur.values()):
                    out_acc_heur = os.path.join(out_dir, f"{args.task}_aggregate_delta_acc_from_default_HEURISTICS.png")
                    _plot_delta_cost_bars_by_p(delta_acc_from_default_heur, out_acc_heur, f"{args.task} — ΔAcc (OURS - default) (Heuristics)", ylabel="Δ Accuracy (vs default)")
                    logger.info(_purple(f"Saved Δacc bar plot (OURS vs default, heuristics): {out_acc_heur}"))
                    if wandb_ok and wandb_run is not None:
                        try:
                            import wandb
                            wandb_run.log({f"plots/{args.task}/aggregate/delta_acc_from_default_heuristics": wandb.Image(out_acc_heur)})
                        except Exception:
                            pass
            except Exception as ex:
                logger.warning(f"Δacc from default plot failed: {ex}")

        # Main figure: Cost vs Accuracy (SLM improvement)
        if len(derived_records_by_p) > 0:
            try:
                out_dir = f"results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full"
                if getattr(args, 'option_id_set', None):
                    out_dir += f"_id-{args.option_id_set}"
                os.makedirs(out_dir, exist_ok=True)
                main_fig_path = os.path.join(out_dir, f"{args.task}_main_figure_cost_vs_acc.png")
                _plot_main_figure_cost_vs_acc(derived_records_by_p, main_fig_path, args.task)
                logger.info(_purple(f"Saved main figure (cost vs acc): {main_fig_path}"))
                # Three-curves plot (Cyclic / PRIDE+OURS / OURS) — runs with derived_records_by_p; PRIDE optional
                try:
                    _plot_cyclic_random_pride_vs_baseline_curves(
                        derived_records_by_p,
                        derived_records_pride_by_p,
                        out_dir,
                        args.task,
                    )
                except Exception as ex:
                    logger.warning(f"Cyclic random PRIDE vs BASELINE plot failed: {ex}")
            except Exception as ex:
                logger.warning(f"Main figure plot failed: {ex}")

        # Aggregate PRIDE+OURS derived-policy summary (if enabled)
        if len(derived_records_pride_by_p) > 0:
            # For plotting: delta cost per p (policies + heuristics)
            delta_cost_policies_by_p: Dict[float, Dict[str, float]] = {}
            delta_cost_heur_by_p: Dict[float, Dict[str, float]] = {}
            delta_acc_policies_by_p: Dict[float, Dict[str, float]] = {}
            delta_acc_heur_by_p: Dict[float, Dict[str, float]] = {}

            for p, cobjs in sorted(derived_records_pride_by_p.items(), key=lambda t: t[0]):
                cyclic_random_keys = [f"cyclic_random_{fp}" for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
                keys = ["default", "cyclic", "full"] + cyclic_random_keys
                logger.info(_purple(f"==== AGGREGATE PRIDE+OURS Derived policy report (REAL-WORLD online, p={p}) ===="))
                single_subject = (len(cobjs) <= 1)
                for key in keys:
                    accs = []
                    costs = []
                    rstds = []
                    for cobj in cobjs:
                        if key in ["default", "cyclic", "full"]:
                            always = cobj.get("always", {}) or {}
                            if key not in always:
                                continue
                            costs.append(float(always[key]["cost"]))
                            accs.append(float(always[key]["acc"]))
                            rkey = f"{key}_recall_std"
                            if rkey in cobj and isinstance(cobj.get(rkey), (int, float)):
                                rstds.append(float(cobj[rkey]))
                        else:
                            if key not in cobj:
                                continue
                            costs.append(float(cobj[key]["costs"][0]))
                            accs.append(float(cobj[key]["accuracies"][0]))
                            rkey = f"{key}_recall_std"
                            if rkey in cobj and isinstance(cobj.get(rkey), (int, float)):
                                rstds.append(float(cobj[rkey]))

                    if len(accs) == 0:
                        continue
                    cost_macro = float(np.mean(costs))
                    acc_macro = float(np.mean(accs))
                    line = f"{key:<14}: cost≈{cost_macro:.3f}, acc={acc_macro:.4f}"
                    if len(rstds) > 0:
                        line += f", recall_std={float(np.mean(rstds)):.4f}"
                    logger.info(line)

                # Heuristic points averages (PRIDE+OURS)
                heuristic_labels = set()
                for cobj in cobjs:
                    for hp in (cobj.get("heuristic_points", []) or []):
                        if isinstance(hp, dict) and hp.get("label") is not None:
                            heuristic_labels.add(str(hp.get("label")))

                if len(heuristic_labels) > 0:
                    logger.info(_purple(f"---- Heuristic points (PRIDE+OURS aggregated over subjects, p={p}) ----"))
                    for lab in sorted(heuristic_labels):
                        costs = []
                        accs = []
                        ws = []
                        n_bases, n_probe2s, n_cyclics, rstds = [], [], [], []
                        for cobj in cobjs:
                            n = int(cobj.get("n_samples", 0)) or 0
                            hp_map = {str(h.get("label")): h for h in (cobj.get("heuristic_points", []) or []) if isinstance(h, dict)}
                            if lab not in hp_map:
                                continue
                            h = hp_map[lab]
                            costs.append(float(h.get("cost", float("nan"))))
                            accs.append(float(h.get("acc", float("nan"))))
                            ws.append(n if n > 0 else 1)
                            if "n_base" in h and h["n_base"] is not None:
                                n_bases.append(int(h["n_base"]))
                            if "n_probe2" in h and h["n_probe2"] is not None:
                                n_probe2s.append(int(h["n_probe2"]))
                            if "n_cyclic" in h and h["n_cyclic"] is not None:
                                n_cyclics.append(int(h["n_cyclic"]))
                            if "recall_std" in h and isinstance(h.get("recall_std"), (int, float)):
                                rstds.append(float(h["recall_std"]))
                        filt = [(c, a, w) for c, a, w in zip(costs, accs, ws) if (not np.isnan(c)) and (not np.isnan(a))]
                        if len(filt) == 0:
                            continue
                        costs_f = [t[0] for t in filt]
                        accs_f = [t[1] for t in filt]
                        cost_macro = float(np.mean(costs_f))
                        acc_macro = float(np.mean(accs_f))
                        line = f"{lab:<14}: cost≈{cost_macro:.3f}, acc={acc_macro:.4f}"
                        if len(n_bases) == len(costs_f) and len(n_probe2s) == len(costs_f) and len(n_cyclics) == len(costs_f):
                            line += f", n_base≈{int(np.mean(n_bases)):.0f}, n_probe2≈{int(np.mean(n_probe2s)):.0f}, n_cyclic≈{int(np.mean(n_cyclics)):.0f}"
                        if len(rstds) > 0:
                            line += f", recall_std={float(np.mean(rstds)):.4f}"
                        logger.info(line)

                # cost/acc overhead vs BASELINE (Δ data always for plots; extra log when verbose)
                if float(p) in derived_records_by_p:
                    try:
                        base_cobjs = derived_records_by_p[float(p)]
                        if bool(getattr(args, "verbose", False)):
                            logger.info(_purple(f"---- Prefix overhead Δcost (PRIDE+OURS - BASELINE), p={p} ----"))
                        for key in keys:
                            base_costs = []
                            pride_costs = []
                            for b, pr in zip(base_cobjs, cobjs):
                                if key in ["default", "cyclic", "full"]:
                                    if key in (b.get("always", {}) or {}) and key in (pr.get("always", {}) or {}):
                                        base_costs.append(float(b["always"][key]["cost"]))
                                        pride_costs.append(float(pr["always"][key]["cost"]))
                                else:
                                    if key in b and key in pr:
                                        base_costs.append(float(b[key]["costs"][0]))
                                        pride_costs.append(float(pr[key]["costs"][0]))
                            if len(base_costs) == 0 or len(pride_costs) == 0:
                                continue
                            d = float(np.mean(np.asarray(pride_costs, dtype=np.float64) - np.asarray(base_costs, dtype=np.float64)))
                            if bool(getattr(args, "verbose", False)):
                                logger.info(f"{key:<14}: Δcost≈{d:+.3f}")
                            delta_cost_policies_by_p.setdefault(float(p), {})[str(key)] = float(d)

                            # Δacc (macro over subjects)
                            try:
                                base_accs = []
                                pride_accs = []
                                for b, pr in zip(base_cobjs, cobjs):
                                    if key in ["default", "cyclic", "full"]:
                                        if key in (b.get("always", {}) or {}) and key in (pr.get("always", {}) or {}):
                                            base_accs.append(float(b["always"][key]["acc"]))
                                            pride_accs.append(float(pr["always"][key]["acc"]))
                                    else:
                                        if key in b and key in pr:
                                            base_accs.append(float(b[key]["accuracies"][0]))
                                            pride_accs.append(float(pr[key]["accuracies"][0]))
                                if len(base_accs) > 0 and len(pride_accs) > 0:
                                    da = float(np.mean(np.asarray(pride_accs, dtype=np.float64) - np.asarray(base_accs, dtype=np.float64)))
                                    delta_acc_policies_by_p.setdefault(float(p), {})[str(key)] = float(da)
                            except Exception:
                                pass

                        # Also print Δacc for policies (same section)
                        if float(p) in delta_acc_policies_by_p:
                            logger.info(_purple(f"---- Prefix overhead Δacc (PRIDE+OURS - BASELINE), p={p} ----"))
                            for key in keys:
                                if key in (delta_acc_policies_by_p.get(float(p), {}) or {}):
                                    logger.info(f"{key:<14}: Δacc≈{float(delta_acc_policies_by_p[float(p)][key]):+.4f}")

                        # Heuristic delta-costs (if present on both)
                        heur_labels = set()
                        for b, pr in zip(base_cobjs, cobjs):
                            for hp in (b.get("heuristic_points", []) or []):
                                if isinstance(hp, dict) and hp.get("label") is not None:
                                    heur_labels.add(str(hp.get("label")))
                        for lab in sorted(list(heur_labels)):
                            bcosts = []
                            pcosts = []
                            for b, pr in zip(base_cobjs, cobjs):
                                bmap = {str(h.get("label")): h for h in (b.get("heuristic_points", []) or []) if isinstance(h, dict)}
                                pmap = {str(h.get("label")): h for h in (pr.get("heuristic_points", []) or []) if isinstance(h, dict)}
                                if lab not in bmap or lab not in pmap:
                                    continue
                                bcosts.append(float(bmap[lab].get("cost", float("nan"))))
                                pcosts.append(float(pmap[lab].get("cost", float("nan"))))
                            filt = [(bb, pp) for bb, pp in zip(bcosts, pcosts) if np.isfinite(bb) and np.isfinite(pp)]
                            if len(filt) == 0:
                                continue
                            bb = np.asarray([t[0] for t in filt], dtype=np.float64)
                            pp = np.asarray([t[1] for t in filt], dtype=np.float64)
                            d2 = float(np.mean(pp - bb))
                            delta_cost_heur_by_p.setdefault(float(p), {})[str(lab)] = float(d2)

                            # heuristic Δacc (macro)
                            try:
                                baccs = []
                                paccs = []
                                for b, pr in zip(base_cobjs, cobjs):
                                    bmap = {str(h.get("label")): h for h in (b.get("heuristic_points", []) or []) if isinstance(h, dict)}
                                    pmap = {str(h.get("label")): h for h in (pr.get("heuristic_points", []) or []) if isinstance(h, dict)}
                                    if lab in bmap and lab in pmap:
                                        baccs.append(float(bmap[lab].get("acc", float("nan"))))
                                        paccs.append(float(pmap[lab].get("acc", float("nan"))))
                                filt2 = [(ba, pa) for ba, pa in zip(baccs, paccs) if np.isfinite(ba) and np.isfinite(pa)]
                                if len(filt2) > 0:
                                    ba = np.asarray([t[0] for t in filt2], dtype=np.float64)
                                    pa = np.asarray([t[1] for t in filt2], dtype=np.float64)
                                    da2 = float(np.mean(pa - ba))
                                    delta_acc_heur_by_p.setdefault(float(p), {})[str(lab)] = float(da2)
                            except Exception:
                                pass

                        # Print heuristics Δcost/Δacc
                        if float(p) in delta_cost_heur_by_p:
                            logger.info(_purple(f"---- Prefix overhead Δcost (Heuristics), p={p} ----"))
                            for lab, v in sorted((delta_cost_heur_by_p.get(float(p), {}) or {}).items(), key=lambda t: str(t[0])):
                                logger.info(f"{str(lab):<18}: Δcost≈{float(v):+.3f}")
                        if float(p) in delta_acc_heur_by_p:
                            logger.info(_purple(f"---- Prefix overhead Δacc (Heuristics), p={p} ----"))
                            for lab, v in sorted((delta_acc_heur_by_p.get(float(p), {}) or {}).items(), key=lambda t: str(t[0])):
                                logger.info(f"{str(lab):<18}: Δacc≈{float(v):+.4f}")
                    except Exception:
                        pass

            # ---- Save Δacc bar plots ----
            try:
                out_dir = f"results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full"
                if getattr(args, 'option_id_set', None):
                    out_dir += f"_id-{args.option_id_set}"
                os.makedirs(out_dir, exist_ok=True)

                # Δacc bar plots (PRIDE+OURS - BASELINE)
                if len(delta_acc_policies_by_p) > 0:
                    out_png = os.path.join(out_dir, f"{args.task}_aggregate_pride_prefix_delta_acc_POLICIES.png")
                    _plot_delta_cost_bars_by_p(
                        delta_cost_by_p=delta_acc_policies_by_p,
                        out_path=out_png,
                        title=f"{args.task} — ΔAcc by p (Policies) [PRIDE+OURS - BASELINE]",
                        ylabel="Δ Accuracy (PRIDE+OURS - BASELINE)",
                    )
                    logger.info(_purple(f"Saved Δacc bar plot (policies): {out_png}"))
                    if wandb_ok and wandb_run is not None:
                        try:
                            import wandb
                            wandb_run.log({f"plots/{args.task}/aggregate/pride_delta_acc_policies": wandb.Image(out_png)})
                        except Exception:
                            pass

                if len(delta_acc_heur_by_p) > 0:
                    out_png = os.path.join(out_dir, f"{args.task}_aggregate_pride_prefix_delta_acc_HEURISTICS.png")
                    _plot_delta_cost_bars_by_p(
                        delta_cost_by_p=delta_acc_heur_by_p,
                        out_path=out_png,
                        title=f"{args.task} — ΔAcc by p (Heuristics) [PRIDE+OURS - BASELINE]",
                        ylabel="Δ Accuracy (PRIDE+OURS - BASELINE)",
                    )
                    logger.info(_purple(f"Saved Δacc bar plot (heuristics): {out_png}"))
                    if wandb_ok and wandb_run is not None:
                        try:
                            import wandb
                            wandb_run.log({f"plots/{args.task}/aggregate/pride_delta_acc_heuristics": wandb.Image(out_png)})
                        except Exception:
                            pass

                # Δacc (OURS+PRIDE - default) bar plots
                delta_acc_pride_vs_default_pol: Dict[float, Dict[str, float]] = {}
                delta_acc_pride_vs_default_heur: Dict[float, Dict[str, float]] = {}
                for p, cobjs in sorted(derived_records_pride_by_p.items(), key=lambda t: t[0]):
                    if float(p) not in derived_records_by_p:
                        continue
                    base_cobjs = derived_records_by_p[float(p)]
                    default_accs = [float(b.get("always", {}).get("default", {}).get("acc", float("nan"))) for b in base_cobjs if "default" in (b.get("always") or {})]
                    default_acc_mean = float(np.mean(default_accs)) if len(default_accs) > 0 else float("nan")
                    if not np.isfinite(default_acc_mean):
                        continue
                    delta_acc_pride_vs_default_pol[float(p)] = {}
                    for key in ["default", "cyclic", "full"] + [f"cyclic_random_{fp}" for fp in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]:
                        accs = []
                        for cobj in cobjs:
                            if key in ["default", "cyclic", "full"]:
                                always = cobj.get("always", {}) or {}
                                if key in always:
                                    accs.append(float(always[key]["acc"]))
                            elif key in cobj:
                                accs.append(float(cobj[key]["accuracies"][0]))
                        if len(accs) > 0:
                            delta_acc_pride_vs_default_pol[float(p)][key] = float(np.mean(accs) - default_acc_mean)
                    delta_acc_pride_vs_default_heur[float(p)] = {}
                    heur_labels_pvd = set()
                    for cobj in cobjs:
                        for hp in (cobj.get("heuristic_points", []) or []):
                            if isinstance(hp, dict) and hp.get("label") is not None:
                                heur_labels_pvd.add(str(hp.get("label")))
                    for lab in heur_labels_pvd:
                        accs = []
                        for cobj in cobjs:
                            hp_map = {str(h.get("label")): h for h in (cobj.get("heuristic_points", []) or []) if isinstance(h, dict)}
                            if lab in hp_map and "acc" in hp_map[lab]:
                                accs.append(float(hp_map[lab]["acc"]))
                        if len(accs) > 0:
                            delta_acc_pride_vs_default_heur[float(p)][lab] = float(np.mean(accs) - default_acc_mean)
                if any(len(d) > 0 for d in delta_acc_pride_vs_default_pol.values()):
                    out_acc_pvd_pol = os.path.join(out_dir, f"{args.task}_aggregate_delta_acc_pride_vs_default_POLICIES.png")
                    _plot_delta_cost_bars_by_p(delta_acc_pride_vs_default_pol, out_acc_pvd_pol, f"{args.task} — ΔAcc (OURS+PRIDE - default) (Policies)", ylabel="Δ Accuracy (vs default)")
                    logger.info(_purple(f"Saved Δacc bar plot (OURS+PRIDE vs default, policies): {out_acc_pvd_pol}"))
                    if wandb_ok and wandb_run is not None:
                        try:
                            import wandb
                            wandb_run.log({f"plots/{args.task}/aggregate/delta_acc_pride_vs_default_policies": wandb.Image(out_acc_pvd_pol)})
                        except Exception:
                            pass
                if any(len(d) > 0 for d in delta_acc_pride_vs_default_heur.values()):
                    out_acc_pvd_heur = os.path.join(out_dir, f"{args.task}_aggregate_delta_acc_pride_vs_default_HEURISTICS.png")
                    _plot_delta_cost_bars_by_p(delta_acc_pride_vs_default_heur, out_acc_pvd_heur, f"{args.task} — ΔAcc (OURS+PRIDE - default) (Heuristics)", ylabel="Δ Accuracy (vs default)")
                    logger.info(_purple(f"Saved Δacc bar plot (OURS+PRIDE vs default, heuristics): {out_acc_pvd_heur}"))
                    if wandb_ok and wandb_run is not None:
                        try:
                            import wandb
                            wandb_run.log({f"plots/{args.task}/aggregate/delta_acc_pride_vs_default_heuristics": wandb.Image(out_acc_pvd_heur)})
                        except Exception:
                            pass

                # Cyclic random PRIDE vs BASELINE curves (5/10/20/30)
                try:
                    _plot_cyclic_random_pride_vs_baseline_curves(
                        derived_records_by_p,
                        derived_records_pride_by_p,
                        out_dir,
                        args.task,
                    )
                except Exception as ex:
                    logger.warning(f"Cyclic random PRIDE vs BASELINE plot failed: {ex}")
            except Exception:
                pass

    # -------- finalize W&B --------
    try:
        if wandb_ok and wandb_run is not None:
            import wandb
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
