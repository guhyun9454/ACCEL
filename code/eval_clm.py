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

    # 2. Policy Points (beta=0)
    policies = ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]
    markers = ['s', '^', 'v', 'o']
    colors = ['orange', 'brown', 'green', 'blue']
    
    for key, m, c in zip(policies, markers, colors):
        if key in curve_obj:
            # beta=0 is the first element
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


def _run_online_sqrt_policy(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float
) -> Tuple[float, float, float]:
    """
    [Online Sqrt Policy]
    th1 = online percentile (running quantile over observed default_conf gaps)
    th2 = th1 * sqrt(1 - CurrentAvgGap) (Online)
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

        # 0) Online th1 (decision uses past observed gaps only)
        if len(past_gaps) > 0:
            th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), float(th1_percent) / 100.0))
        else:
            # 초기에는 보수적/공격적 선택이 필요함. 여기서는 "일단 통과" 쪽(0.0)으로 둠.
            th1_val = 0.0
        
        # 1. Update Running Average
        if running_cnt > 0:
            current_avg_gap = running_gap_sum / running_cnt
        else:
            current_avg_gap = 0.0
            
        # 2. Dynamic th2
        safe_avg = min(1.0, max(0.0, current_avg_gap))
        current_th2_val = th1_val * np.sqrt(1.0 - safe_avg)
        final_th2_val = current_th2_val

        # 3. Execution
        if gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
        else:
            if float(mc[i]) < current_th2_val:
                total_cost += float(k)
                corrects += 1 if cyclic_correct[i] else 0
            else:
                total_cost += 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0
        
        running_gap_sum += gap_i
        running_cnt += 1
        past_gaps.append(gap_i)

    if len(mc) > 0:
        final_th2_perc = (np.sum(mc < final_th2_val) / len(mc)) * 100.0
    else:
        final_th2_perc = 0.0

    return total_cost / float(N), corrects / float(N), final_th2_perc


def _run_online_sqrt_policy_lowconf_update(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float
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
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
        else:
            if float(mc[i]) < current_th2_val:
                total_cost += float(k)
                corrects += 1 if cyclic_correct[i] else 0
            else:
                total_cost += 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0

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


def _run_online_th1_quantile_th2_from_th1_rule(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    th2_rule_from_th1_value,
) -> Tuple[float, float, float]:
    """
    [Real-world Online Policy Template]
    - th1: online running-quantile over PAST default_conf gaps (percentile = th1_percent)
    - th2: derived from current th1 value by a deterministic rule: th2 = f(th1)

    NOTE: returned th2_percentile is only for plotting (maps final th2 value onto *full* mean_conf distribution).
    Decision itself never uses future information.
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

        # Online th1 (past only)
        if len(past_gaps) > 0:
            th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), q))
        else:
            th1_val = 0.0

        # th2 derived from th1 value
        th2_val = float(th2_rule_from_th1_value(th1_val))
        final_th2_val = th2_val

        # Execute
        if gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
        else:
            if float(mc[i]) < th2_val:
                total_cost += float(k)
                corrects += 1 if cyclic_correct[i] else 0
            else:
                total_cost += 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0

        past_gaps.append(gap_i)

    if len(mc) > 0:
        final_th2_perc = (np.sum(mc < final_th2_val) / len(mc)) * 100.0
    else:
        final_th2_perc = 0.0

    return total_cost / float(N), corrects / float(N), final_th2_perc


def _run_online_top2flip_policy(
    default_conf: np.ndarray,
    flip_trigger: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    offline_prefix_n: int = 0,
) -> Tuple[float, float]:
    """
    [Real-world Online top2flip]
    - th1: online running-quantile over PAST default_conf gaps (percentile = th1_percent)
    - low-conf (dc < th1): if flip_trigger -> cyclic else probe2
    - offline_prefix_n: first n samples are base-only, but still update running stats.
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

        # prefix: base only
        if i < int(offline_prefix_n):
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            past_gaps.append(gap_i)
            continue

        # online th1 from past only
        if len(past_gaps) > 0:
            th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), q))
        else:
            th1_val = 0.0

        if gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
        else:
            if bool(flip[i]):
                total_cost += float(k)
                corrects += 1 if cyclic_correct[i] else 0
            else:
                total_cost += 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0

        past_gaps.append(gap_i)

    return total_cost / float(N), corrects / float(N)


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
) -> Tuple[float, float]:
    """
    [Real-world Online avggap]
    - th1: online running-quantile over PAST default_conf gaps (percentile = th1_percent)
    - th2: online running-quantile over PAST mean_conf gaps (percentile = th2_percent)
    - low-conf (dc < th1): if mc < th2 -> cyclic else probe2
    - offline_prefix_n: first n samples are base-only, but still update running stats.

    Note: mean_conf values are precomputed for analysis; decision uses only past thresholds (no future).
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

        if len(past_dc) > 0:
            th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q1))
        else:
            th1_val = 0.0

        if len(past_mc) > 0:
            th2_val = float(np.quantile(np.asarray(past_mc, dtype=np.float64), q2))
        else:
            th2_val = 0.0

        if gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
        else:
            if mgap_i < th2_val:
                total_cost += float(k)
                corrects += 1 if cyclic_correct[i] else 0
            else:
                total_cost += 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0

        past_dc.append(gap_i)
        past_mc.append(mgap_i)

    return total_cost / float(N), corrects / float(N)


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
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        ax1.scatter([p_sqrt], [c_sqrt], marker='D', s=80, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (All)' if idx==0 else "")

        # (E) Online Sqrt (LowConf-only update)
        c_sqrt_lc, _, p_sqrt_lc = _run_online_sqrt_policy_lowconf_update(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        ax1.scatter([p_sqrt_lc], [c_sqrt_lc], marker='X', s=70, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (LowConf-only)' if idx==0 else "")

    ax1.set_xlabel("th2 (percentile, avg gap)", fontsize=11)
    ax1.set_ylabel("Computational Cost (× of default)", fontsize=11)
    ax1.set_title(f"{getattr(args, 'task', 'task')} {subject} — Cost vs th2 (Heuristics Comparison)", fontsize=12)
    ax1.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax1.set_xticklabels([str(t) for t in [1, 5, 10, 15, 20, 25, 30]])
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.4)
    out_cost = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST.png")
    os.makedirs(os.path.dirname(out_cost), exist_ok=True)
    fig1.tight_layout()
    fig1.savefig(out_cost, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: Δ Accuracy (%) vs th2
    fig2, ax2 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        # 1) Dense Curve
        accs = []
        for th2p in dense_th2_list:
            _, a = _run_online_avggap_policy(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
            )
            accs.append(a)
        delta_accs = [(a - default_acc) * 100.0 for a in accs]
        ax2.plot(dense_th2_list, delta_accs, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.6)

        # 2) Heuristics
        # (A) th1 / 2
        _, a_half, p_half = _online_point(th1p, lambda x: x / 2.0)
        ax2.scatter([p_half], [(a_half-default_acc)*100], marker='*', s=120, color=color, edgecolors='black', zorder=6)

        # (A2) th1 / sqrt(k)
        _, a_sqrtk, p_sqrtk = _online_point(th1p, lambda x, kk=k: x / math.sqrt(float(kk)))
        ax2.scatter([p_sqrtk], [(a_sqrtk-default_acc)*100], marker='P', s=110, color=color, edgecolors='black', zorder=6)
        
        # (B) th1 ^ 2
        _, a_sq, p_sq = _online_point(th1p, lambda x: x ** 2)
        ax2.scatter([p_sq], [(a_sq-default_acc)*100], marker='s', s=80, color=color, edgecolors='black', zorder=6)

        # (C) th1 ^ 1.5
        _, a_pow, p_pow = _online_point(th1p, lambda x: x ** 1.5)
        ax2.scatter([p_pow], [(a_pow-default_acc)*100], marker='^', s=90, color=color, edgecolors='black', zorder=6)

        # (D) Online Sqrt
        _, a_sqrt, p_sqrt = _run_online_sqrt_policy(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        ax2.scatter([p_sqrt], [(a_sqrt-default_acc)*100], marker='D', s=80, color=color, edgecolors='black', zorder=6)

        # (E) Online Sqrt (LowConf-only update)
        _, a_sqrt_lc, p_sqrt_lc = _run_online_sqrt_policy_lowconf_update(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        ax2.scatter([p_sqrt_lc], [(a_sqrt_lc-default_acc)*100], marker='X', s=70, color=color, edgecolors='black', zorder=6)

    ax2.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("th2 (percentile, avg gap)", fontsize=11)
    ax2.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax2.set_title(f"{getattr(args, 'task', 'task')} {subject} — Δ Accuracy vs th2", fontsize=12)
    ax2.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax2.set_xticklabels([str(t) for t in [1, 5, 10, 15, 20, 25, 30]])
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.4)
    out_delta = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_DELTA_ACC.png")
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

        # Log
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
    ax3.set_title(f"{getattr(args, 'task', 'task')} {subject} — Trade-off (All Heuristics)", fontsize=12)
    ax3.legend(loc='best', fontsize=9, ncol=2)
    ax3.grid(True, linestyle='--', alpha=0.4)
    out_trade = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST_vs_DELTA.png")
    fig3.tight_layout()
    fig3.savefig(out_trade, bbox_inches="tight")
    plt.close(fig3)

    logger.info(_purple(f"th2 trade-off plots saved (dense th2=1..30 + online points): {subject}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({
                f"plots/{subject}/th2_tradeoff_COST": wandb.Image(out_cost),
                f"plots/{subject}/th2_tradeoff_DELTA_ACC": wandb.Image(out_delta),
                f"plots/{subject}/th2_tradeoff_COST_vs_DELTA": wandb.Image(out_trade),
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
    betas: Optional[List[float]] = None
) -> dict:
    """
    baseline curves:
      cyclic, full, switch_full, switch_cyclic, ours_top2flip, ours_avggap
    + always(default/cyclic/full ensemble) points

    beta 의미: offline prefix 비율 (n = beta*N)
      - first n samples: base only (cost=1)
      - remaining N-n: policy applied (potentially expensive)
    => beta↑ (offline↑)  → cost↓
    """
    if betas is None:
        betas = [i / 10.0 for i in range(11)]

    N = len(base_correct_list)
    if N == 0:
        return {}

    perc01 = float(max(0.0, min(100.0, perc_value))) / 100.0

    C_cyc = float(k)
    C_full = float(len(perm_list)) if full_enabled else float("nan")

    default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    cyclic_acc_always = float(np.mean(np.asarray(cyclic_correct_list, dtype=np.float64)))
    full_acc_always = float(np.mean(np.asarray(full_correct_list, dtype=np.float64))) if full_enabled and len(full_correct_list) == N else float("nan")

    # 1) cyclic/full beta curves
    curve_cyc = []
    curve_full = []
    for beta in betas:
        n = int(N * beta + 1e-9)

        acc_cyc = (sum(base_correct_list[:n]) + sum(cyclic_correct_list[n:])) / float(N)
        acc_full = (sum(base_correct_list[:n]) + sum(full_correct_list[n:])) / float(N) if full_enabled and len(full_correct_list) == N else float("nan")

        cost_cyc = beta * 1.0 + (1.0 - beta) * C_cyc
        cost_full = beta * 1.0 + (1.0 - beta) * C_full if full_enabled else float("nan")

        curve_cyc.append((cost_cyc, acc_cyc))
        if full_enabled:
            curve_full.append((cost_full, acc_full))

    # 2) switch curves
    curve_switch_full = []
    curve_switch_cyc = []
    for beta in betas:
        n = int(N * beta + 1e-9)

        total_cost_sf = 0.0
        corrects_sf = 0
        total_cost_sc = 0.0
        corrects_sc = 0
        past_gaps: List[float] = []

        # offline prefix: base only
        for i in range(0, n):
            total_cost_sf += 1.0
            total_cost_sc += 1.0
            if base_correct_list[i]:
                corrects_sf += 1
                corrects_sc += 1
            # online thresholding uses observed gaps only (past)
            past_gaps.append(float(default_conf[i]))

        # online: apply switch
        for i in range(n, N):
            # Online th1 threshold = running-quantile over PAST gaps only
            if len(past_gaps) > 0:
                thresh = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), perc01))
            else:
                # no past info yet -> don't treat as ambiguous (avoid oracle / avoid over-trigger)
                thresh = float("-inf")

            amb = (float(default_conf[i]) < thresh)

            if full_enabled and len(full_correct_list) == N:
                if amb:
                    total_cost_sf += C_full
                    corrects_sf += 1 if full_correct_list[i] else 0
                else:
                    total_cost_sf += 1.0
                    corrects_sf += 1 if base_correct_list[i] else 0

            if amb:
                total_cost_sc += C_cyc
                corrects_sc += 1 if cyclic_correct_list[i] else 0
            else:
                total_cost_sc += 1.0
                corrects_sc += 1 if base_correct_list[i] else 0

            # Update past gaps AFTER decision
            past_gaps.append(float(default_conf[i]))

        if full_enabled and len(full_correct_list) == N:
            curve_switch_full.append((total_cost_sf / float(N), corrects_sf / float(N)))
        curve_switch_cyc.append((total_cost_sc / float(N), corrects_sc / float(N)))

    # 3) ours_top2flip
    curve_top2flip = []
    for beta in betas:
        n = int(N * beta + 1e-9)
        c, a = _run_online_top2flip_policy(
            default_conf=default_conf,
            flip_trigger=flip_trigger,
            base_correct=base_correct_list,
            cyclic_correct=cyclic_correct_list,
            probe2_correct=probe2_correct,
            k=k,
            th1_percent=perc_value,
            offline_prefix_n=n,
        )
        curve_top2flip.append((float(c), float(a)))

    # 4) ours_avggap
    curve_avggap = []
    for beta in betas:
        n = int(N * beta + 1e-9)
        c, a = _run_online_avggap_policy(
            default_conf=default_conf,
            mean_conf=mean_conf,
            base_correct=base_correct_list,
            cyclic_correct=cyclic_correct_list,
            probe2_correct=probe2_correct,
            k=k,
            th1_percent=perc_value,
            th2_percent=perc_value,
            offline_prefix_n=n,
        )
        curve_avggap.append((float(c), float(a)))

    curve_obj = {
        "subject": subject,
        "tag": str(tag),
        "k": int(k),
        "percentile": float(perc_value),
        "betas": [float(b) for b in betas],
        "default_accuracy": float(default_acc),

        "always": {
            "default": {"cost": 1.0, "acc": float(default_acc)},
            "cyclic": {"cost": float(C_cyc), "acc": float(cyclic_acc_always)},
        },

        "cyclic": {"costs": [float(c) for c, _ in curve_cyc], "accuracies": [float(a) for _, a in curve_cyc]},
        "switch_cyclic": {"costs": [float(c) for c, _ in curve_switch_cyc], "accuracies": [float(a) for _, a in curve_switch_cyc]},
        "ours_top2flip": {"costs": [float(c) for c, _ in curve_top2flip], "accuracies": [float(a) for _, a in curve_top2flip]},
        "ours_avggap": {"costs": [float(c) for c, _ in curve_avggap], "accuracies": [float(a) for _, a in curve_avggap]},
    }
    if full_enabled:
        curve_obj["always"]["full"] = {"cost": float(C_full), "acc": float(full_acc_always)}
        curve_obj["full"] = {"costs": [float(c) for c, _ in curve_full], "accuracies": [float(a) for _, a in curve_full]}
        if len(curve_switch_full) > 0:
            curve_obj["switch_full"] = {"costs": [float(c) for c, _ in curve_switch_full], "accuracies": [float(a) for _, a in curve_switch_full]}
    return curve_obj


def _compute_curve_for_single_policy(
    subject: str,
    tag: str,
    policy_key: str,
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
    betas: Optional[List[float]] = None
) -> dict:
    """
    Returns curve_obj containing ONLY:
      - default_accuracy
      - always(default/cyclic/full)
      - <policy_key> curve
    """
    if betas is None:
        betas = [i / 10.0 for i in range(11)]

    N = len(base_correct_list)
    if N == 0:
        return {}

    perc01 = float(max(0.0, min(100.0, perc_value))) / 100.0
    C_cyc = float(k)
    C_full = float(len(perm_list))

    default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    cyclic_acc_always = float(np.mean(np.asarray(cyclic_correct_list, dtype=np.float64)))
    full_acc_always = float(np.mean(np.asarray(full_correct_list, dtype=np.float64)))

    curve = []

    if policy_key == "switch_cyclic":
        for beta in betas:
            n = int(N * beta + 1e-9)
            th1 = _quantile(default_conf[:n], perc01) if n > 0 else _quantile(default_conf, perc01)

            total_cost = 0.0
            corrects = 0

            for i in range(0, n):
                total_cost += 1.0
                corrects += 1 if base_correct_list[i] else 0

            for i in range(n, N):
                if float(default_conf[i]) < th1:
                    total_cost += C_cyc
                    corrects += 1 if cyclic_correct_list[i] else 0
                else:
                    total_cost += 1.0
                    corrects += 1 if base_correct_list[i] else 0

            curve.append((total_cost / float(N), corrects / float(N)))

    elif policy_key == "ours_top2flip":
        for beta in betas:
            n = int(N * beta + 1e-9)
            th1 = _quantile(default_conf[:n], perc01) if n > 0 else _quantile(default_conf, perc01)

            total_cost = 0.0
            corrects = 0

            for i in range(0, n):
                total_cost += 1.0
                corrects += 1 if base_correct_list[i] else 0

            for i in range(n, N):
                if float(default_conf[i]) >= th1:
                    total_cost += 1.0
                    corrects += 1 if base_correct_list[i] else 0
                else:
                    if bool(flip_trigger[i]):
                        total_cost += C_cyc
                        corrects += 1 if cyclic_correct_list[i] else 0
                    else:
                        total_cost += 2.0
                        corrects += 1 if bool(probe2_correct[i]) else 0

            curve.append((total_cost / float(N), corrects / float(N)))

    elif policy_key == "ours_avggap":
        for beta in betas:
            n = int(N * beta + 1e-9)
            th1 = _quantile(default_conf[:n], perc01) if n > 0 else _quantile(default_conf, perc01)
            th2 = _quantile(mean_conf[:n], perc01) if n > 0 else _quantile(mean_conf, perc01)

            total_cost = 0.0
            corrects = 0

            for i in range(0, n):
                total_cost += 1.0
                corrects += 1 if base_correct_list[i] else 0

            for i in range(n, N):
                if float(default_conf[i]) >= th1:
                    total_cost += 1.0
                    corrects += 1 if base_correct_list[i] else 0
                else:
                    if float(mean_conf[i]) < th2:
                        total_cost += C_cyc
                        corrects += 1 if cyclic_correct_list[i] else 0
                    else:
                        total_cost += 2.0
                        corrects += 1 if bool(probe2_correct[i]) else 0

            curve.append((total_cost / float(N), corrects / float(N)))
    else:
        raise ValueError(f"Unknown policy_key: {policy_key}")

    curve_obj = {
        "subject": subject,
        "tag": str(tag),
        "k": int(k),
        "percentile": float(perc_value),
        "betas": [float(b) for b in betas],
        "default_accuracy": float(default_acc),

        "always": {
            "default": {"cost": 1.0, "acc": float(default_acc)},
            "cyclic": {"cost": float(C_cyc), "acc": float(cyclic_acc_always)},
            "full": {"cost": float(C_full), "acc": float(full_acc_always)},
        },

        policy_key: {"costs": [float(c) for c, _ in curve], "accuracies": [float(a) for _, a in curve]},
    }
    return curve_obj


def _log_baseline_report(curve_obj: dict):
    """
    BASELINE은 풀로 찍고,
    PRIDE_FREE는 (아래 main에서) 한 줄만 찍는다.
    """
    p = curve_obj.get("percentile")
    logger.info(_purple(f"==== BASELINE Derived policy report (REAL-WORLD online, p={p}) ===="))

    always = curve_obj.get("always", {})
    logger.info(f"BASELINE default(ensemble) : cost={always['default']['cost']:.3f}, acc={always['default']['acc']:.4f}")
    logger.info(f"BASELINE cyclic(ensemble)  : cost={always['cyclic']['cost']:.3f}, acc={always['cyclic']['acc']:.4f}")
    if "full" in always:
        logger.info(f"BASELINE full(ensemble)    : cost={always['full']['cost']:.3f}, acc={always['full']['acc']:.4f}")
    else:
        logger.info("BASELINE full(ensemble)    : (disabled)")

    for key in ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]:
        if key in curve_obj:
            c0 = float(curve_obj[key]["costs"][0])
            a0 = float(curve_obj[key]["accuracies"][0])
            logger.info(f"BASELINE {key:<12} : cost={c0:.3f}, acc={a0:.4f}")


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
                else:
                    metrics = {'type': 'metric', 'data': {}}
                    metrics['data']['accuracy'] = get_accuracy(results)
                    metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)

            logger.info(_orange(f"Run completed: {subject}"))

            if not use_cached:
                save_results(cached_path, results, metrics)
                logger.info(f"Results saved: {subject}")

            # =========================================================
            # Derived policies & PRIDE_FREE (historically ONLY when args.setting == 'full')
            # =========================================================
            if args.setting == 'full' and len(results) > 0:
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

                    base_probs_list = []  # identity row (letter-space)

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
                        corr_cyc = (pred_cyc == data['ideal'])
                        cyclic_correct_list.append(corr_cyc)
                        cyclic_corrects += 1 if corr_cyc else 0
                        cyclic_total += 1

                        # base (identity only)
                        base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
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

                        # full (all perms) - only if available
                        if full_enabled:
                            agg_full = _aggregate_probs_over_permutations(probs_seq_np, perm_list, k)
                            pred_full = option_ids[int(np.argmax(agg_full))]
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
                        probe2_correct_list.append(pred2 == ideals[i])

                    default_conf = np.asarray(default_conf, dtype=np.float64)
                    mean_conf = np.asarray(mean_gap_list, dtype=np.float64)
                    arr_flip_trigger = np.asarray(flip_trigger_mask, dtype=bool)
                    arr_probe2_correct = np.asarray(probe2_correct_list, dtype=bool)

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
                    if full_enabled:
                        logger.info(_purple(f"[{subject}] Accuracies — Full: {full_acc:.4f}, Cyclic: {cyclic_acc:.4f}, Default: {base_acc:.4f}"))
                    else:
                        logger.info(_purple(f"[{subject}] Accuracies — Full: (disabled), Cyclic: {cyclic_acc:.4f}, Default: {base_acc:.4f}"))

                    # ---------- compute & save beta curves (baseline) ----------
                    curve_save_path = f'results_{args.task}/{args.num_few_shot}s_{args.model_name}/{args.task}_full'
                    if getattr(args, 'option_id_set', None):
                        curve_save_path += f'_id-{args.option_id_set}'
                    os.makedirs(curve_save_path, exist_ok=True)

                    perc_list = _parse_percent_value_list(getattr(args, "ours_low_conf_percent", 10.0))
                    curve_objs_baseline = []
                    baseline_by_p = {}

                    for perc in perc_list:
                        perc = float(perc)
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
                        )
                        if cobj:
                            curve_objs_baseline.append(cobj)
                            baseline_by_p[perc] = cobj
                            _log_baseline_report(cobj)

                            # [ADD] Baseline Point Plot with 3 Rules
                            ptag = f"p{int(round(perc))}"
                            out_pts = os.path.join(curve_save_path, f"{subject}_{ptag}_baseline_points.png")
                            
                            # 3가지 룰 포인트 계산
                            th1p = float(perc)
                            extra_pts = []
                            
                            # Helper to calc static rule points for baseline plot
                            def _get_static_pt(th1_p, rule_func, label, marker, color):
                                c, a, _ = _run_online_th1_quantile_th2_from_th1_rule(
                                    default_conf=default_conf,
                                    mean_conf=mean_conf,
                                    base_correct=base_correct_list,
                                    cyclic_correct=cyclic_correct_list,
                                    probe2_correct=arr_probe2_correct,
                                    k=k,
                                    th1_percent=float(th1_p),
                                    th2_rule_from_th1_value=rule_func,
                                )
                                return {'cost': c, 'acc': a, 'label': label, 'marker': marker, 'color': color}

                            # 1. Static Heuristics (division heuristic scaled by #choices)
                            # th2 = th1 / sqrt(k)  (k=4 -> th1/2, k=5 -> th1/sqrt(5), ...)
                            extra_pts.append(_get_static_pt(th1p, lambda x: x / 2.0, 'th1/2', '*', 'gray'))
                            extra_pts.append(_get_static_pt(th1p, lambda x, kk=k: x / math.sqrt(float(kk)), 'th1/sqrt(k)', 'P', 'gray'))
                            extra_pts.append(_get_static_pt(th1p, lambda x: x ** 2, 'th1^2', 's', 'gray'))
                            extra_pts.append(_get_static_pt(th1p, lambda x: x ** 1.5, 'th1^1.5', '^', 'gray'))

                            # 2. [ONLINE Sqrt]
                            c_sqrt, a_sqrt, _ = _run_online_sqrt_policy(
                                default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1_percent=perc
                            )
                            extra_pts.append({'cost': c_sqrt, 'acc': a_sqrt, 'label': 'Online Sqrt', 'marker': 'D', 'color': 'orange'})

                            # [ADD] log heuristic point performances (no plot annotation)
                            try:
                                base_acc0 = float(np.mean(np.asarray(base_correct_list, dtype=np.float64))) if len(base_correct_list) else float("nan")
                                logger.info(_purple(f"==== HEURISTIC Point report (p={int(round(perc))}) ===="))
                                logger.info(f"{'default':<18}: cost=1.000, acc={base_acc0:.4f}")
                                for hp in extra_pts:
                                    acc = float(hp.get("acc", float("nan")))
                                    cost = float(hp.get("cost", float("nan")))
                                    logger.info(f"{hp.get('label','?'):<18}: cost={cost:.3f}, acc={acc:.4f}")
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
                                title=f"{args.task} {subject} — Baseline Policies (REAL-WORLD online, {ptag}, Heuristics 5)",
                                extra_points=extra_pts
                            )
                            if wandb_ok and wandb_run is not None:
                                try:
                                    import wandb
                                    wandb_run.log({f"plots/{subject}/{ptag}/baseline_points": wandb.Image(out_pts)})
                                except Exception:
                                    pass

                    # [ADD] Confidence Distribution Plot (th1/th2 thresholds for p=10,20,30)
                    out_dist = os.path.join(curve_save_path, f"{subject}_confidence_distribution.png")
                    _plot_confidence_distribution(
                        default_conf=default_conf,
                        mean_conf=mean_conf,
                        out_path=out_dist,
                        title=f"{args.task} {subject} — Confidence Gap Distribution"
                    )
                    if wandb_ok and wandb_run is not None:
                        try:
                            import wandb
                            wandb_run.log({f"plots/{subject}/confidence_distribution": wandb.Image(out_dist)})
                        except Exception:
                            pass

                    save_results(f'{curve_save_path}/{subject}_beta_curve.jsonl', curve_objs_baseline, metrics=None)

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
                        )

                except Exception as e:
                    logger.warning(f"Failed to derive beta curves for subject '{subject}': {e}")
                    import traceback
                    traceback.print_exc()

            logging_cuda_memory_usage()

    # -------- finalize W&B --------
    try:
        if wandb_ok and wandb_run is not None:
            import wandb
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
