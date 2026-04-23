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
from eval_clm_online import (
    _recall_std,
    _gaussian_base_posterior_prob,
    _gaussian_swap_posterior_prob,
    _run_cyclic_random_fraction,
    _run_cyclic_random_fraction_with_preds,
    _run_online_avggap_policy,
    _run_online_avggap_policy_with_preds,
    _run_online_avggap_policy_with_stats,
    _run_online_sqrt_policy,
    _run_online_sqrt_policy_lowconf_update,
    _run_online_sqrt_policy_with_preds,
    _run_online_sqrt_policy_with_stats,
    _run_online_switch_cyclic_with_preds,
    _run_online_switch_cyclic_with_stats,
    _run_online_swap_posterior_conf_policy,
    _run_online_swap_posterior_conf_policy_with_preds,
    _run_online_swap_posterior_conf_policy_with_stats,
    _run_online_swap_gaussian_policy,
    _run_online_swap_gaussian_policy_with_preds,
    _run_online_swap_gaussian_policy_with_stats,
    _run_online_th1_quantile_th2_from_th1_rule,
    _run_online_th1_quantile_th2_from_th1_rule_with_preds,
    _run_online_th1_quantile_th2_from_th1_rule_with_stats,
    _run_online_top2flip_policy,
    _run_online_top2flip_policy_with_preds,
    _run_online_top2flip_policy_with_stats,
    _run_prefix_cyclic_postfix_base,
)
from eval_clm_plots import _plot_three_curves_acc_recall_std
from eval_clm_reporting import _log_baseline_report, _log_named_report

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

LEGACY_OURS_LABEL = "th1/2"
SWAP_GAUSSIAN_LABEL = "swap_gaussian"
SWAP_GAUSSIAN_SQRT_LABEL = "swap_gaussian_sqrt"
SWAP_GAUSSIAN_POSTERIOR_LABEL = "swap_gaussian_posterior"
SWAP_GAUSSIAN_POSTERIOR_LATIN_LABEL = "swap_gaussian_posterior_latin"
PRIMARY_OURS_LABEL = SWAP_GAUSSIAN_LABEL


def _rule_th1_half(th1_val: float) -> float:
    return float(th1_val) / 2.0


def _rule_th1_sqrt2(th1_val: float) -> float:
    return float(th1_val) / math.sqrt(2.0)


def _gap_of_distribution(probs: np.ndarray) -> float:
    vals = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    if vals.size <= 1:
        return 0.0
    return float(vals[0] - vals[1])


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(xa) & np.isfinite(ya)
    if np.sum(m) < 2:
        return float("nan")
    xa = xa[m]
    ya = ya[m]
    if float(np.std(xa)) <= 1e-12 or float(np.std(ya)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _build_sigma_analysis_record(
    subject: str,
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    cyclic_gap_mean: np.ndarray,
    cyclic_gap_std: np.ndarray,
    flip_mask: np.ndarray,
) -> Dict[str, float]:
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    mu = np.asarray(cyclic_gap_mean, dtype=np.float64)
    sg = np.asarray(cyclic_gap_std, dtype=np.float64)
    flip = np.asarray(flip_mask, dtype=bool)

    resid_single = dc - mu
    resid_two_view = mc - mu
    sigma_single = float(np.std(resid_single)) if resid_single.size > 0 else float("nan")
    sigma_two_view = float(np.std(resid_two_view)) if resid_two_view.size > 0 else float("nan")
    sigma_ratio = float(sigma_two_view / sigma_single) if np.isfinite(sigma_single) and sigma_single > 1e-12 else float("nan")

    q_low = float(np.quantile(dc, 0.30)) if dc.size > 0 else 0.0
    q_high = float(np.quantile(dc, 0.70)) if dc.size > 0 else 0.0
    low_conf_mask = dc <= q_low
    high_conf_mask = dc >= q_high

    q_sigma_low = float(np.quantile(sg, 0.30)) if sg.size > 0 else 0.0
    q_sigma_high = float(np.quantile(sg, 0.70)) if sg.size > 0 else 0.0
    low_sigma_mask = sg <= q_sigma_low
    high_sigma_mask = sg >= q_sigma_high

    def _mean_or_nan(arr: np.ndarray, mask: np.ndarray) -> float:
        if arr.size == 0 or np.sum(mask) == 0:
            return float("nan")
        return float(np.mean(arr[mask]))

    flip_float = flip.astype(np.float64)
    return {
        "subject": str(subject),
        "n": int(dc.size),
        "default_gap_mean": float(np.mean(dc)) if dc.size > 0 else float("nan"),
        "two_view_gap_mean": float(np.mean(mc)) if mc.size > 0 else float("nan"),
        "cyclic_gap_mean": float(np.mean(mu)) if mu.size > 0 else float("nan"),
        "sigma_mean": float(np.mean(sg)) if sg.size > 0 else float("nan"),
        "sigma_std": float(np.std(sg)) if sg.size > 0 else float("nan"),
        "sigma_single": sigma_single,
        "sigma_two_view": sigma_two_view,
        "sigma_ratio": sigma_ratio,
        "sigma_ratio_target": float(1.0 / math.sqrt(2.0)),
        "corr_default_gap_sigma": _safe_corr(dc, sg),
        "corr_flip_sigma": _safe_corr(flip_float, sg),
        "sigma_low_conf_mean": _mean_or_nan(sg, low_conf_mask),
        "sigma_high_conf_mean": _mean_or_nan(sg, high_conf_mask),
        "flip_low_conf": _mean_or_nan(flip_float, low_conf_mask),
        "flip_high_conf": _mean_or_nan(flip_float, high_conf_mask),
        "flip_low_sigma": _mean_or_nan(flip_float, low_sigma_mask),
        "flip_high_sigma": _mean_or_nan(flip_float, high_sigma_mask),
    }

def _pride_correct_row(row: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """PriDe correction: divide by prior then renormalize."""
    r = np.asarray(row, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64)
    adj = r / (pr + eps)
    adj = adj / (adj.sum() + eps)
    return adj


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


def _slot_labels_for_k(k: int) -> List[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 < int(k) <= len(alphabet):
        return list(alphabet[: int(k)])
    return [str(i) for i in range(int(k))]


def _load_rank_slot_std_lookup(summary_path: str, k: int) -> Dict[Tuple[int, str], float]:
    if not summary_path or not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"rank-slot summary json not found: {summary_path}. "
            f"Please provide --rank_slot_summary_json."
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    by_k = obj.get("by_k", {}) or {}
    k_obj = by_k.get(str(int(k)))
    if not isinstance(k_obj, dict):
        raise ValueError(f"rank-slot summary missing by_k[{k}] in {summary_path}")
    raw = k_obj.get("raw", {}) or {}
    labels = _slot_labels_for_k(int(k))
    lookup: Dict[Tuple[int, str], float] = {}
    for r in range(1, int(k) + 1):
        rank_key = f"rank{r}"
        slot_map = raw.get(rank_key)
        if not isinstance(slot_map, dict):
            raise ValueError(f"rank-slot summary missing raw[{rank_key}] in {summary_path}")
        for slot in labels:
            fit = slot_map.get(slot, {}) or {}
            std_val = float(fit.get("std", float("nan")))
            if not np.isfinite(std_val) or std_val <= 0.0:
                raise ValueError(f"invalid std for ({rank_key},{slot}) in {summary_path}: {std_val}")
            lookup[(int(r), str(slot))] = std_val
    return lookup


def _load_rank_slot_summary_obj(summary_path: str, k: int) -> Dict[str, Any]:
    if not summary_path or not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"rank-slot summary json not found: {summary_path}. "
            f"Please provide --rank_slot_summary_json."
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    by_k = obj.get("by_k", {}) or {}
    k_obj = by_k.get(str(int(k)))
    if not isinstance(k_obj, dict):
        raise ValueError(f"rank-slot summary missing by_k[{k}] in {summary_path}")
    raw = k_obj.get("raw", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"rank-slot summary missing by_k[{k}]['raw'] in {summary_path}")
    return raw


def _rank_slot_stat_from_entry(entry: Dict[str, Any], stat_source: str = "empirical") -> Dict[str, float]:
    src = str(stat_source).lower().strip()
    if src in {"fit", "gaussian", "gaussian_fit"}:
        fit = (entry.get("gaussian_fit", {}) or {})
        mean_val = float(fit.get("mean", float("nan")))
        std_val = float(fit.get("std", float("nan")))
    else:
        mean_val = float(entry.get("mean", float("nan")))
        std_val = float(entry.get("std", float("nan")))
    n_val = int(entry.get("n", 0))
    return {"n": n_val, "mean": mean_val, "std": std_val}


def _combine_gaussian_stats(stats: List[Dict[str, float]]) -> Dict[str, float]:
    valid = []
    for st in (stats or []):
        n = int(st.get("n", 0))
        mean = float(st.get("mean", float("nan")))
        std = float(st.get("std", float("nan")))
        if n > 0 and np.isfinite(mean) and np.isfinite(std) and std >= 0.0:
            valid.append({"n": n, "mean": mean, "std": std})
    if not valid:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    n_total = int(sum(int(st["n"]) for st in valid))
    if n_total <= 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    mean_total = float(sum(float(st["n"]) * float(st["mean"]) for st in valid) / float(n_total))
    second_moment = float(
        sum(
            float(st["n"]) * ((float(st["std"]) ** 2) + (float(st["mean"]) ** 2))
            for st in valid
        ) / float(n_total)
    )
    var_total = max(0.0, second_moment - mean_total ** 2)
    return {"n": int(n_total), "mean": float(mean_total), "std": float(math.sqrt(var_total))}


def _load_slot_gaussian_stats_lookup(
    summary_path: str,
    k: int,
    *,
    stat_source: str = "empirical",
    rank_mode: str = "rank12_pooled",
) -> Dict[str, Dict[str, float]]:
    raw = _load_rank_slot_summary_obj(summary_path, int(k))
    labels = _slot_labels_for_k(int(k))
    rank_mode_norm = str(rank_mode).lower().strip()
    if rank_mode_norm not in {"rank1", "rank2", "rank12", "rank12_pooled"}:
        raise ValueError(f"Unsupported rank_mode={rank_mode}. Expected one of rank1/rank2/rank12_pooled.")

    lookup: Dict[str, Dict[str, float]] = {}
    for slot in labels:
        stats_to_merge: List[Dict[str, float]] = []
        if rank_mode_norm == "rank1":
            stats_to_merge.append(_rank_slot_stat_from_entry((raw.get("rank1", {}) or {}).get(slot, {}) or {}, stat_source))
        elif rank_mode_norm == "rank2":
            stats_to_merge.append(_rank_slot_stat_from_entry((raw.get("rank2", {}) or {}).get(slot, {}) or {}, stat_source))
        else:
            stats_to_merge.append(_rank_slot_stat_from_entry((raw.get("rank1", {}) or {}).get(slot, {}) or {}, stat_source))
            stats_to_merge.append(_rank_slot_stat_from_entry((raw.get("rank2", {}) or {}).get(slot, {}) or {}, stat_source))
        combined = _combine_gaussian_stats(stats_to_merge)
        if not np.isfinite(float(combined.get("mean", float("nan")))):
            raise ValueError(f"Invalid pooled mean for slot={slot} in {summary_path}")
        if not np.isfinite(float(combined.get("std", float("nan")))) or float(combined.get("std", 0.0)) <= 0.0:
            raise ValueError(f"Invalid pooled std for slot={slot} in {summary_path}")
        lookup[str(slot)] = combined
    return lookup


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
    policies = [SWAP_GAUSSIAN_LABEL]
    markers = ['v']
    colors = ['green']
    
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

        policies = [SWAP_GAUSSIAN_LABEL]
        markers = ['v']
        colors = ['green']
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
            costs, accs, rstds, nbs, np2s, nswaps, ncs = [], [], [], [], [], [], []
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
                    if "n_swap" in h:
                        nswaps.append(int(h.get("n_swap", 0)))
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
                if nswaps:
                    entry["n_swap"] = int(np.mean(nswaps))
                if ncs:
                    entry["n_cyclic"] = int(np.mean(ncs))
                merged_hp.append(entry)
        out["heuristic_points"] = merged_hp
    # transition (Ours baseline): sum counts over runs
    trans_list = [c.get("transition") for c in cobjs if isinstance(c.get("transition"), dict)]
    if trans_list:
        out["transition"] = {
            "t_to_f_count": sum(int(t.get("t_to_f_count", 0)) for t in trans_list),
            "f_to_t_count": sum(int(t.get("f_to_t_count", 0)) for t in trans_list),
            "base_t_count": sum(int(t.get("base_t_count", 0)) for t in trans_list),
            "base_f_count": sum(int(t.get("base_f_count", 0)) for t in trans_list),
        }
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
    swap_posterior_prob: Optional[np.ndarray] = None,
    swap_cand1_correct: Optional[List[bool]] = None,
    swap_cand2_correct: Optional[List[bool]] = None,
    swap_cand1_pred_idx: Optional[List[int]] = None,
    swap_cand2_pred_idx: Optional[List[int]] = None,
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

    # 2) new production rule: exact top1-top2 swap + Gaussian posterior
    swap_cost = float("nan")
    swap_acc = float("nan")
    swap_stats: Dict[str, int] = {"n_base": 0, "n_swap": 0}
    swap_sqrt_cost = float("nan")
    swap_sqrt_acc = float("nan")
    swap_sqrt_stats: Dict[str, int] = {"n_base": 0, "n_swap": 0, "n_cyclic": 0}
    if (
        swap_posterior_prob is not None
        and swap_cand1_correct is not None
        and swap_cand2_correct is not None
        and len(swap_cand1_correct) == N
        and len(swap_cand2_correct) == N
    ):
        swap_cost, swap_acc, swap_stats = _run_online_swap_gaussian_policy_with_stats(
            default_conf=default_conf,
            swap_posterior_prob=swap_posterior_prob,
            cand1_correct=swap_cand1_correct,
            cand2_correct=swap_cand2_correct,
            k=k,
            th1_percent=perc_value,
            cyclic_correct=cyclic_correct_list,
        )
        swap_sqrt_cost, swap_sqrt_acc, swap_sqrt_stats = _run_online_swap_gaussian_policy_with_stats(
            default_conf=default_conf,
            swap_posterior_prob=swap_posterior_prob,
            cand1_correct=swap_cand1_correct,
            cand2_correct=swap_cand2_correct,
            k=k,
            th1_percent=perc_value,
            cyclic_correct=cyclic_correct_list,
            th2_mode="sqrt",
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
        SWAP_GAUSSIAN_LABEL: {"costs": [float(swap_cost)], "accuracies": [float(swap_acc)], "stats": dict(swap_stats)},
        SWAP_GAUSSIAN_SQRT_LABEL: {"costs": [float(swap_sqrt_cost)], "accuracies": [float(swap_sqrt_acc)], "stats": dict(swap_sqrt_stats)},
    }
    # Optional: add recall_std when labels_idx and preds available
    if (
        labels_idx is not None
        and base_pred_idx is not None
        and cyclic_pred_idx is not None
    ):
        try:
            if (
                swap_posterior_prob is not None
                and swap_cand1_pred_idx is not None
                and swap_cand2_pred_idx is not None
            ):
                _, _, preds_swap = _run_online_swap_gaussian_policy_with_preds(
                    default_conf,
                    swap_posterior_prob,
                    swap_cand1_pred_idx,
                    swap_cand2_pred_idx,
                    labels_idx,
                    k,
                    perc_value,
                    cyclic_pred_idx=cyclic_pred_idx,
                )
                curve_obj[f"{SWAP_GAUSSIAN_LABEL}_recall_std"] = float(_recall_std(labels_idx, preds_swap, k))
                _, _, preds_swap_sqrt = _run_online_swap_gaussian_policy_with_preds(
                    default_conf,
                    swap_posterior_prob,
                    swap_cand1_pred_idx,
                    swap_cand2_pred_idx,
                    labels_idx,
                    k,
                    perc_value,
                    cyclic_pred_idx=cyclic_pred_idx,
                    th2_mode="sqrt",
                )
                curve_obj[f"{SWAP_GAUSSIAN_SQRT_LABEL}_recall_std"] = float(_recall_std(labels_idx, preds_swap_sqrt, k))
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
    return curve_obj


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
                        "label": PRIMARY_OURS_LABEL,
                        "cost": float(ours_cost + 0.05),
                        "acc": float(np.clip(ours_acc + 0.01, 0.0, 1.0)),
                        "recall_std": float(np.clip(ours_rstd - 0.01, 0.0, 1.0)),
                        "n_base": int(780 + rng.integers(0, 50)),
                        "n_swap": int(170 + rng.integers(0, 50)),
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
                            "label": LEGACY_OURS_LABEL, "th1_p": p, "cost": float(pride_cost),
                            "acc": float(pride_acc), "recall_std": float(pride_rstd),
                            "n_base": 800, "n_probe2": 150, "n_cyclic": 50,
                        })
                    derived_records_pride_by_alpha[alpha] = [cobj_pr]

                _plot_three_curves_acc_recall_std(
                    derived_records_by_p,
                    derived_records_pride_by_p,
                    derived_records_pride_by_alpha,
                    {},
                    {},
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
    # 일부 LLaMA/Mistral 계열은 slow(SentencePiece) tokenizer 파일이 없어서
    # use_fast=False 로 강제하면 \"TypeError: not a string\" 가 발생한다.
    # 따라서 기본값(use_fast=True)에 맡기고 BOS/EOS 토큰만 제어한다.
    toker = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
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
        sigma_analysis_baseline_records: List[dict] = []
        sigma_analysis_pride_by_alpha: Dict[float, List[dict]] = {}
        rank_slot_std_lookup_cache: Dict[int, Dict[Tuple[int, str], float]] = {}
        slot_gaussian_stats_cache: Dict[Tuple[int, str, str], Dict[str, Dict[str, float]]] = {}
        swap_posterior_conf_records: Dict[float, List[dict]] = {}
        swap_posterior_latin_conf_records: Dict[float, List[dict]] = {}

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
                        latin_correct_list = []
                        latin_pred_idx_list = []

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
                        if int(k) not in rank_slot_std_lookup_cache:
                            rank_slot_std_lookup_cache[int(k)] = _load_rank_slot_std_lookup(
                                getattr(args, "rank_slot_summary_json", None), int(k)
                            )
                        rank_slot_std_lookup = rank_slot_std_lookup_cache[int(k)]
                        slot_gaussian_cache_key = (
                            int(k),
                            str(getattr(args, "swap_posterior_stat_source", "empirical")),
                            str(getattr(args, "swap_posterior_rank_mode", "rank12_pooled")),
                        )
                        if slot_gaussian_cache_key not in slot_gaussian_stats_cache:
                            slot_gaussian_stats_cache[slot_gaussian_cache_key] = _load_slot_gaussian_stats_lookup(
                                getattr(args, "rank_slot_summary_json", None),
                                int(k),
                                stat_source=str(getattr(args, "swap_posterior_stat_source", "empirical")),
                                rank_mode=str(getattr(args, "swap_posterior_rank_mode", "rank12_pooled")),
                            )
                        slot_gaussian_stats = slot_gaussian_stats_cache[slot_gaussian_cache_key]
                        slot_labels = _slot_labels_for_k(int(k))

                        swap_posterior_prob_list = []
                        base_posterior_prob_list = []
                        swap_slot_posterior_prob_list = []
                        swap_cand1_pred_idx_list = []
                        swap_cand2_pred_idx_list = []
                        swap_cand1_correct_list = []
                        swap_cand2_correct_list = []

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

                            latin_perm_rows = [tuple(int(x) for x in p) for p in (data.get("latin_top12_perm_tuples") or [])]
                            latin_probs_rows = data.get("latin_top12_probs") or []
                            if (
                                isinstance(latin_probs_rows, list)
                                and len(latin_perm_rows) == int(k)
                                and len(latin_probs_rows) == len(latin_perm_rows)
                            ):
                                agg_latin = _aggregate_probs_over_permutations(latin_probs_rows, latin_perm_rows, k)
                            else:
                                agg_latin = np.asarray(agg_cyc, dtype=np.float64)
                            pred_latin_idx = int(np.argmax(agg_latin))
                            pred_latin = option_ids[pred_latin_idx]
                            latin_pred_idx_list.append(pred_latin_idx)
                            latin_correct_list.append(pred_latin == data["ideal"])

                            # base (identity only)
                            base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
                            base_probs_list.append(base_probs)
                            pred_base = option_ids[int(np.argmax(base_probs))]
                            base_pred_idx_list.append(int(np.argmax(base_probs)))
                            corr_base = (pred_base == data['ideal'])
                            base_correct_list.append(corr_base)
                            top_order_base = np.argsort(base_probs)[::-1]
                            cand1_idx = int(top_order_base[0]) if top_order_base.size > 0 else 0
                            cand2_idx = int(top_order_base[1]) if top_order_base.size > 1 else int(cand1_idx)
                            swap_cand1_pred_idx_list.append(int(cand1_idx))
                            swap_cand2_pred_idx_list.append(int(cand2_idx))
                            swap_cand1_correct_list.append(int(cand1_idx) == int(option_ids.index(str(data['ideal']))))
                            swap_cand2_correct_list.append(int(cand2_idx) == int(option_ids.index(str(data['ideal']))))

                            swap_perm = list(range(k))
                            swap_perm[cand1_idx], swap_perm[cand2_idx] = swap_perm[cand2_idx], swap_perm[cand1_idx]
                            swap_probs = data.get("swap_top12_probs", None)
                            if not isinstance(swap_probs, list):
                                if full_enabled:
                                    try:
                                        swap_idx = perm_list.index(tuple(int(x) for x in swap_perm))
                                        swap_probs = probs_seq_np[swap_idx].tolist()
                                    except ValueError:
                                        swap_probs = None
                                if not isinstance(swap_probs, list):
                                    raise ValueError(
                                        f"Sample idx={data.get('idx')} missing swap_top12_probs; "
                                        f"rerun eval_clm.py with updated cache generation or provide full permutations."
                                    )
                            swap_probs_np = np.asarray(swap_probs, dtype=np.float64)
                            rank_order = np.argsort(np.asarray(agg_cyc, dtype=np.float64))[::-1].tolist()
                            rank_map = {int(content_idx): int(pos + 1) for pos, content_idx in enumerate(rank_order)}
                            rank1 = rank_map[int(cand1_idx)]
                            rank2 = rank_map[int(cand2_idx)]
                            slot1_base = str(slot_labels[int(cand1_idx)])
                            slot1_swap = str(slot_labels[int(cand2_idx)])
                            slot2_base = str(slot_labels[int(cand2_idx)])
                            slot2_swap = str(slot_labels[int(cand1_idx)])
                            base_slot_stat_1 = slot_gaussian_stats[slot1_base]
                            base_slot_stat_2 = slot_gaussian_stats[slot2_base]
                            swap_slot_stat_1 = slot_gaussian_stats[slot1_swap]
                            swap_slot_stat_2 = slot_gaussian_stats[slot2_swap]
                            p_base = _gaussian_base_posterior_prob(
                                y1=float(base_probs[int(cand1_idx)]),
                                mean1=float(base_slot_stat_1["mean"]),
                                std1=float(base_slot_stat_1["std"]),
                                y2=float(base_probs[int(cand2_idx)]),
                                mean2=float(base_slot_stat_2["mean"]),
                                std2=float(base_slot_stat_2["std"]),
                            )
                            p_swap = _gaussian_swap_posterior_prob(
                                y1_base=float(base_probs[int(cand1_idx)]),
                                y1_swap=float(swap_probs_np[int(cand2_idx)]),
                                std1_base=float(rank_slot_std_lookup[(int(rank1), slot1_base)]),
                                std1_swap=float(rank_slot_std_lookup[(int(rank1), slot1_swap)]),
                                y2_base=float(base_probs[int(cand2_idx)]),
                                y2_swap=float(swap_probs_np[int(cand1_idx)]),
                                std2_base=float(rank_slot_std_lookup[(int(rank2), slot2_base)]),
                                std2_swap=float(rank_slot_std_lookup[(int(rank2), slot2_swap)]),
                            )
                            p_swap_slot = _gaussian_swap_posterior_prob(
                                y1_base=float(base_probs[int(cand1_idx)]),
                                y1_swap=float(swap_probs_np[int(cand2_idx)]),
                                std1_base=float(base_slot_stat_1["std"]),
                                std1_swap=float(swap_slot_stat_1["std"]),
                                y2_base=float(base_probs[int(cand2_idx)]),
                                y2_swap=float(swap_probs_np[int(cand1_idx)]),
                                std2_base=float(base_slot_stat_2["std"]),
                                std2_swap=float(swap_slot_stat_2["std"]),
                            )
                            base_posterior_prob_list.append(float(p_base))
                            swap_posterior_prob_list.append(float(p_swap))
                            swap_slot_posterior_prob_list.append(float(p_swap_slot))
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
                        cyclic_gap_mean_list = []  # per-sample mean gap over cyclic rotations
                        cyclic_gap_std_list = []   # per-sample sigma over cyclic rotations

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

                            cyc_gaps_i = []
                            for cyc_local_idx, perm_idx in enumerate(cyclic_indices):
                                agg_cyc_single = _aggregate_probs_over_permutations(
                                    [per_sample_probs[i][perm_idx].tolist()],
                                    [cyc_perms[cyc_local_idx]],
                                    k,
                                )
                                cyc_gaps_i.append(_gap_of_distribution(np.asarray(agg_cyc_single, dtype=np.float64)))
                            cyc_gaps_arr = np.asarray(cyc_gaps_i, dtype=np.float64)
                            cyclic_gap_mean_list.append(float(np.mean(cyc_gaps_arr)) if cyc_gaps_arr.size > 0 else 0.0)
                            cyclic_gap_std_list.append(float(np.std(cyc_gaps_arr)) if cyc_gaps_arr.size > 0 else 0.0)

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
                        arr_base_posterior = np.asarray(base_posterior_prob_list, dtype=np.float64)
                        arr_swap_posterior = np.asarray(swap_posterior_prob_list, dtype=np.float64)
                        arr_swap_slot_posterior = np.asarray(swap_slot_posterior_prob_list, dtype=np.float64)
                        cyclic_gap_mean = np.asarray(cyclic_gap_mean_list, dtype=np.float64)
                        cyclic_gap_std = np.asarray(cyclic_gap_std_list, dtype=np.float64)
                        sigma_analysis_baseline_records.append(
                            _build_sigma_analysis_record(
                                subject=subject,
                                default_conf=default_conf,
                                mean_conf=mean_conf,
                                cyclic_gap_mean=cyclic_gap_mean,
                                cyclic_gap_std=cyclic_gap_std,
                                flip_mask=arr_flip_trigger,
                            )
                        )
                        try:
                            conf_levels = [
                                float(x)
                                for x in _parse_percent_value_list(getattr(args, "swap_posterior_conf_levels", "80,90,95"))
                                if 50.0 <= float(x) <= 100.0
                            ]
                        except Exception:
                            conf_levels = [80.0, 90.0, 95.0]
                        for conf_level in conf_levels:
                            try:
                                sp_cost, sp_acc, sp_stats = _run_online_swap_posterior_conf_policy_with_stats(
                                    base_posterior_prob=arr_base_posterior,
                                    swap_posterior_prob=arr_swap_slot_posterior,
                                    cand1_correct=swap_cand1_correct_list,
                                    cand2_correct=swap_cand2_correct_list,
                                    k=k,
                                    conf_percent=float(conf_level),
                                    cyclic_correct=cyclic_correct_list,
                                )
                                _, _, sp_preds = _run_online_swap_posterior_conf_policy_with_preds(
                                    base_posterior_prob=arr_base_posterior,
                                    swap_posterior_prob=arr_swap_slot_posterior,
                                    cand1_pred_idx=swap_cand1_pred_idx_list,
                                    cand2_pred_idx=swap_cand2_pred_idx_list,
                                    labels_idx=[option_ids.index(str(x)) for x in ideals],
                                    k=k,
                                    conf_percent=float(conf_level),
                                    cyclic_pred_idx=cyclic_pred_idx_list,
                                )
                                swap_posterior_conf_records.setdefault(float(conf_level), []).append({
                                    "subject": str(subject),
                                    "cost": float(sp_cost),
                                    "acc": float(sp_acc),
                                    "recall_std": float(_recall_std([option_ids.index(str(x)) for x in ideals], sp_preds, k)),
                                    "n_base": int(sp_stats.get("n_base", 0)),
                                    "n_swap": int(sp_stats.get("n_swap", 0)),
                                    "n_cyclic": int(sp_stats.get("n_cyclic", 0)),
                                })
                                sp_l_cost, sp_l_acc, sp_l_stats = _run_online_swap_posterior_conf_policy_with_stats(
                                    base_posterior_prob=arr_base_posterior,
                                    swap_posterior_prob=arr_swap_slot_posterior,
                                    cand1_correct=swap_cand1_correct_list,
                                    cand2_correct=swap_cand2_correct_list,
                                    k=k,
                                    conf_percent=float(conf_level),
                                    cyclic_correct=latin_correct_list,
                                )
                                _, _, sp_l_preds = _run_online_swap_posterior_conf_policy_with_preds(
                                    base_posterior_prob=arr_base_posterior,
                                    swap_posterior_prob=arr_swap_slot_posterior,
                                    cand1_pred_idx=swap_cand1_pred_idx_list,
                                    cand2_pred_idx=swap_cand2_pred_idx_list,
                                    labels_idx=[option_ids.index(str(x)) for x in ideals],
                                    k=k,
                                    conf_percent=float(conf_level),
                                    cyclic_pred_idx=latin_pred_idx_list,
                                )
                                swap_posterior_latin_conf_records.setdefault(float(conf_level), []).append({
                                    "subject": str(subject),
                                    "cost": float(sp_l_cost),
                                    "acc": float(sp_l_acc),
                                    "recall_std": float(_recall_std([option_ids.index(str(x)) for x in ideals], sp_l_preds, k)),
                                    "n_base": int(sp_l_stats.get("n_base", 0)),
                                    "n_swap": int(sp_l_stats.get("n_swap", 0)),
                                    "n_cyclic": int(sp_l_stats.get("n_cyclic", 0)),
                                })
                            except Exception:
                                pass

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
                                    swap_posterior_prob=arr_swap_posterior,
                                    swap_cand1_correct=swap_cand1_correct_list,
                                    swap_cand2_correct=swap_cand2_correct_list,
                                    swap_cand1_pred_idx=swap_cand1_pred_idx_list,
                                    swap_cand2_pred_idx=swap_cand2_pred_idx_list,
                                    full_pred_idx=full_pred_idx_list if full_enabled and len(full_pred_idx_list) == len(ideals) else None,
                                    cyclic_fractions=cyclic_fracs_run, run_seed_offset=run_idx_inner,
                                )

                                if cobj:
                                    by_perc_baseline[perc].append(cobj)
                                    if "heuristic_points" not in cobj:
                                        c_swap, a_swap, st_swap = _run_online_swap_gaussian_policy_with_stats(
                                            default_conf=default_conf,
                                            swap_posterior_prob=arr_swap_posterior,
                                            cand1_correct=swap_cand1_correct_list,
                                            cand2_correct=swap_cand2_correct_list,
                                            k=k,
                                            th1_percent=perc,
                                            cyclic_correct=cyclic_correct_list,
                                        )
                                        hp_swap = {
                                            "label": PRIMARY_OURS_LABEL,
                                            "cost": float(c_swap),
                                            "acc": float(a_swap),
                                            "marker": "v",
                                            "color": "gray",
                                            "n_base": int(st_swap.get("n_base", 0)),
                                            "n_swap": int(st_swap.get("n_swap", 0)),
                                        }
                                        try:
                                            _, _, preds_swap = _run_online_swap_gaussian_policy_with_preds(
                                                default_conf,
                                                arr_swap_posterior,
                                                swap_cand1_pred_idx_list,
                                                swap_cand2_pred_idx_list,
                                                labels_idx_for_curves,
                                                k,
                                                perc,
                                                cyclic_pred_idx=cyclic_pred_idx_list,
                                            )
                                            hp_swap["recall_std"] = float(_recall_std(labels_idx_for_curves, preds_swap, k))
                                        except Exception:
                                            pass
                                        cobj["heuristic_points"] = [hp_swap]
                                        try:
                                            c_swap_sqrt, a_swap_sqrt, st_swap_sqrt = _run_online_swap_gaussian_policy_with_stats(
                                                default_conf=default_conf,
                                                swap_posterior_prob=arr_swap_posterior,
                                                cand1_correct=swap_cand1_correct_list,
                                                cand2_correct=swap_cand2_correct_list,
                                                k=k,
                                                th1_percent=perc,
                                                cyclic_correct=cyclic_correct_list,
                                                th2_mode="sqrt",
                                            )
                                            hp_swap_sqrt = {
                                                "label": SWAP_GAUSSIAN_SQRT_LABEL,
                                                "cost": float(c_swap_sqrt),
                                                "acc": float(a_swap_sqrt),
                                                "marker": "P",
                                                "color": "gray",
                                                "n_base": int(st_swap_sqrt.get("n_base", 0)),
                                                "n_swap": int(st_swap_sqrt.get("n_swap", 0)),
                                                "n_cyclic": int(st_swap_sqrt.get("n_cyclic", 0)),
                                            }
                                            _, _, preds_swap_sqrt = _run_online_swap_gaussian_policy_with_preds(
                                                default_conf,
                                                arr_swap_posterior,
                                                swap_cand1_pred_idx_list,
                                                swap_cand2_pred_idx_list,
                                                labels_idx_for_curves,
                                                k,
                                                perc,
                                                cyclic_pred_idx=cyclic_pred_idx_list,
                                                th2_mode="sqrt",
                                            )
                                            hp_swap_sqrt["recall_std"] = float(_recall_std(labels_idx_for_curves, preds_swap_sqrt, k))
                                            cobj["heuristic_points"].append(hp_swap_sqrt)
                                        except Exception:
                                            pass

                                    # Ours (baseline) transition 기록
                                    try:
                                        _, _, preds_ours = _run_online_swap_gaussian_policy_with_preds(
                                            default_conf,
                                            arr_swap_posterior,
                                            swap_cand1_pred_idx_list,
                                            swap_cand2_pred_idx_list,
                                            labels_idx_for_curves,
                                            k,
                                            perc,
                                            cyclic_pred_idx=cyclic_pred_idx_list,
                                        )
                                        rec = _make_transition_record_from_preds(
                                            base_correct_list, preds_ours, labels_idx_for_curves, default_conf, subject)
                                        transition_records_ours_by_p.setdefault(perc, []).append(rec)
                                        # curve에 transition counts만 저장 (routing 리포트용)
                                        cobj["transition"] = {
                                            "t_to_f_count": rec["t_to_f_count"],
                                            "f_to_t_count": rec["f_to_t_count"],
                                            "base_t_count": len(rec["base_t_gaps"]),
                                            "base_f_count": len(rec["base_f_gaps"]),
                                        }
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
                                cyclic_gap_mean_list_pr = []
                                cyclic_gap_std_list_pr = []

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

                                    cyc_gaps_i_pr = []
                                    for cyc_local_idx, perm_idx in enumerate(cyclic_indices):
                                        agg_cyc_single_pr = _aggregate_probs_over_permutations(
                                            [ps_corr[perm_idx].tolist()],
                                            [cyc_perms[cyc_local_idx]],
                                            k,
                                        )
                                        cyc_gaps_i_pr.append(_gap_of_distribution(np.asarray(agg_cyc_single_pr, dtype=np.float64)))
                                    cyc_gaps_arr_pr = np.asarray(cyc_gaps_i_pr, dtype=np.float64)
                                    cyclic_gap_mean_list_pr.append(float(np.mean(cyc_gaps_arr_pr)) if cyc_gaps_arr_pr.size > 0 else 0.0)
                                    cyclic_gap_std_list_pr.append(float(np.std(cyc_gaps_arr_pr)) if cyc_gaps_arr_pr.size > 0 else 0.0)

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
                                cyclic_gap_mean_pr = np.asarray(cyclic_gap_mean_list_pr, dtype=np.float64)
                                cyclic_gap_std_pr = np.asarray(cyclic_gap_std_list_pr, dtype=np.float64)
                                sigma_analysis_pride_by_alpha.setdefault(float(pride_alpha), []).append(
                                    _build_sigma_analysis_record(
                                        subject=subject,
                                        default_conf=default_conf_pr,
                                        mean_conf=mean_conf_pr,
                                        cyclic_gap_mean=cyclic_gap_mean_pr,
                                        cyclic_gap_std=cyclic_gap_std_pr,
                                        flip_mask=arr_flip_trigger_pr,
                                    )
                                )

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
                                    def _get_static_pt_pride(th1_p, rule_func, label_key, marker_key):
                                        c, a, th2p, st = _run_online_th1_quantile_th2_from_th1_rule_with_stats(
                                            default_conf_pr, mean_conf_pr, bc_use, cc_use,
                                            arr_probe2_correct_pr, k, th1_p, rule_func, prefix_ids_set)
                                        out = {'cost': c, 'acc': a, 'label': label_key, 'th1_p': th1_p, 'marker': marker_key, 'color': 'gray'}
                                        try:
                                            _, _, _, preds = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                                default_conf_pr, mean_conf_pr, bp_use, cp_use,
                                                probe2_pred_idx_list_pr, labels_idx_for_curves, k, th1_p, rule_func, prefix_ids_set)
                                            out['recall_std'] = float(_recall_std(labels_idx_for_curves, preds, k))
                                            out['n_base'], out['n_probe2'], out['n_cyclic'] = st['n_base'], st['n_probe2'], st['n_cyclic']
                                        except Exception:
                                            pass
                                        return out
                                    pts_th12 = [_get_static_pt_pride(float(th1), _rule_th1_half, LEGACY_OURS_LABEL, "*") for th1 in ours_th1_list]
                                    cobj_pr["heuristic_points"] = pts_th12
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
                                            probe2_pred_idx_list_pr, labels_idx_for_curves, k, ours_th1, _rule_th1_half, prefix_ids_set)
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

        def _sigma_summary_slug(name: str) -> str:
            s = str(name).strip().lower()
            out = []
            for ch in s:
                if ch.isalnum():
                    out.append(ch)
                else:
                    out.append("_")
            slug = "".join(out)
            while "__" in slug:
                slug = slug.replace("__", "_")
            return slug.strip("_") or "sigma"

        def _print_sigma_analysis(records: List[dict], name: str):
            if not records:
                return None
            keys = [
                "sigma_mean",
                "sigma_std",
                "sigma_single",
                "sigma_two_view",
                "sigma_ratio",
                "corr_default_gap_sigma",
                "corr_flip_sigma",
                "sigma_low_conf_mean",
                "sigma_high_conf_mean",
                "flip_low_conf",
                "flip_high_conf",
                "flip_low_sigma",
                "flip_high_sigma",
            ]
            agg = {}
            for key in keys:
                vals = [float(r.get(key, float("nan"))) for r in records if np.isfinite(float(r.get(key, float("nan"))))]
                agg[key] = float(np.mean(vals)) if vals else float("nan")
            logger.info(_purple(f"\n==== SIGMA ANALYSIS: {name} ===="))
            logger.info(
                f"records={len(records)} | "
                f"sigma(mean)={agg['sigma_mean']:.4f}, sigma(std)={agg['sigma_std']:.4f}, "
                f"single_resid_sigma={agg['sigma_single']:.4f}, two_view_resid_sigma={agg['sigma_two_view']:.4f}, "
                f"ratio={agg['sigma_ratio']:.4f} (target={1.0 / math.sqrt(2.0):.4f})"
            )
            logger.info(
                f"corr(default_gap,sigma)={agg['corr_default_gap_sigma']:.4f}, "
                f"corr(flip,sigma)={agg['corr_flip_sigma']:.4f}"
            )
            logger.info(
                f"low_conf_sigma={agg['sigma_low_conf_mean']:.4f}, high_conf_sigma={agg['sigma_high_conf_mean']:.4f}, "
                f"flip_low_conf={agg['flip_low_conf']:.4f}, flip_high_conf={agg['flip_high_conf']:.4f}"
            )
            logger.info(
                f"flip_low_sigma={agg['flip_low_sigma']:.4f}, flip_high_sigma={agg['flip_high_sigma']:.4f}"
            )
            logger.info("========================================\n")
            agg["records"] = int(len(records))
            agg["sigma_ratio_target"] = float(1.0 / math.sqrt(2.0))
            return agg

        sigma_summary_payload = {}
        baseline_sigma_summary = _print_sigma_analysis(sigma_analysis_baseline_records, "Baseline")
        if baseline_sigma_summary is not None:
            sigma_summary_payload[_sigma_summary_slug("Baseline")] = baseline_sigma_summary
        for alpha in sorted(sigma_analysis_pride_by_alpha.keys()):
            name = f"PriDe(alpha={float(alpha):g}%)"
            pride_sigma_summary = _print_sigma_analysis(sigma_analysis_pride_by_alpha[alpha], name)
            if pride_sigma_summary is not None:
                sigma_summary_payload[_sigma_summary_slug(name)] = pride_sigma_summary
        if wandb_ok and wandb_run is not None and sigma_summary_payload:
            try:
                existing_sigma = wandb_run.summary.get("sigma_analysis_v1", {})
                if not isinstance(existing_sigma, dict):
                    existing_sigma = {}
                existing_sigma = dict(existing_sigma)
                task_key = f"{str(args.task)}_{int(args.num_few_shot)}shot"
                existing_sigma[task_key] = sigma_summary_payload
                wandb_run.summary["sigma_analysis_v1"] = existing_sigma
            except Exception as e:
                logger.warning(f"W&B sigma summary update failed: {e}")

        # Three-curves: Cost vs Acc, Cost vs Recall_std (Cyclic / Default+PRIDE / OURS th1/sqrt2)
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
                    swap_posterior_conf_records if len(swap_posterior_conf_records) > 0 else {},
                    swap_posterior_latin_conf_records if len(swap_posterior_latin_conf_records) > 0 else {},
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

            def get_heur_stats(cobjs, label=PRIMARY_OURS_LABEL):
                costs, accs, rstds, nb, naux, nc = [], [], [], [], [], []
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
                        if "n_swap" in h:
                            naux.append(h["n_swap"])
                        elif "n_probe2" in h:
                            naux.append(h["n_probe2"])
                        if "n_cyclic" in h:
                            nc.append(h["n_cyclic"])
                if not accs:
                    return float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan")
                mean_c, std_c = _macro_mean_std_over_runs(costs, n_subjects, n_runs)
                mean_a, std_a = _macro_mean_std_over_runs(accs, n_subjects, n_runs)
                mean_r, std_r = _macro_mean_std_over_runs(rstds, n_subjects, n_runs)
                mean_nb = float(np.mean(nb)) if nb else 0.0
                mean_np2 = float(np.mean(naux)) if naux else 0.0
                mean_nc = float(np.mean(nc)) if nc else 0.0
                return mean_c, mean_a, mean_r, mean_nb, mean_np2, mean_nc, std_c, std_a, std_r

            def get_heur_stats_by_th1_p(cobjs, th1_p, label_filter=LEGACY_OURS_LABEL):
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

            # 2. ours + pride (legacy heuristic comparison baseline)
            logger.info("---- ours + pride ----")
            for alpha in pride_alphas:
                cobjs = derived_records_pride_by_alpha[alpha]
                for p in pride_fracs:
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats_by_th1_p(cobjs, p, LEGACY_OURS_LABEL)
                    a_str = f"{float(alpha):g}"
                    p_str = f"{float(p):g}"
                    logger.info(f"ours_pride_α{a_str}_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")

            # 3. ours
            logger.info("---- ours ----")
            for p in pride_fracs:
                if float(p) in derived_records_by_p:
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats(derived_records_by_p[float(p)], PRIMARY_OURS_LABEL)
                    p_str = f"{float(p):g}"
                    logger.info(f"ours_swap_gaussian_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_swap={np2:.0f}, n_cyclic={nc:.0f}")
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats(derived_records_by_p[float(p)], SWAP_GAUSSIAN_SQRT_LABEL)
                    logger.info(f"ours_swap_gaussian_sqrt_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_swap={np2:.0f}, n_cyclic={nc:.0f}")

            # 3.5 calibrated Gaussian posterior routing
            if swap_posterior_conf_records:
                logger.info("---- ours posterior confidence ----")
                for conf_level in sorted(swap_posterior_conf_records.keys()):
                    rows = swap_posterior_conf_records[float(conf_level)]
                    costs = [float(r.get("cost", float("nan"))) for r in rows if np.isfinite(float(r.get("cost", float("nan"))))]
                    accs = [float(r.get("acc", float("nan"))) for r in rows if np.isfinite(float(r.get("acc", float("nan"))))]
                    rstds = [float(r.get("recall_std", float("nan"))) for r in rows if np.isfinite(float(r.get("recall_std", float("nan"))))]
                    mean_c, std_c = _macro_mean_std_over_runs(costs, n_subjects, n_runs)
                    mean_a, std_a = _macro_mean_std_over_runs(accs, n_subjects, n_runs)
                    mean_r, std_r = _macro_mean_std_over_runs(rstds, n_subjects, n_runs)
                    mean_nb = float(np.mean([float(r.get("n_base", 0)) for r in rows])) if rows else 0.0
                    mean_ns = float(np.mean([float(r.get("n_swap", 0)) for r in rows])) if rows else 0.0
                    mean_nc = float(np.mean([float(r.get("n_cyclic", 0)) for r in rows])) if rows else 0.0
                    conf_str = f"{float(conf_level):g}"
                    logger.info(
                        f"ours_swap_gaussian_posterior_{conf_str}% : "
                        f"cost={_fmt(mean_c, std_c)}, acc={_fmt4(mean_a, std_a)}, "
                        f"recall_std={_fmt4(mean_r, std_r)}, "
                        f"n_base={mean_nb:.0f}, n_swap={mean_ns:.0f}, n_cyclic={mean_nc:.0f}"
                    )
            if swap_posterior_latin_conf_records:
                logger.info("---- ours posterior confidence (latin fallback) ----")
                for conf_level in sorted(swap_posterior_latin_conf_records.keys()):
                    rows = swap_posterior_latin_conf_records[float(conf_level)]
                    costs = [float(r.get("cost", float("nan"))) for r in rows if np.isfinite(float(r.get("cost", float("nan"))))]
                    accs = [float(r.get("acc", float("nan"))) for r in rows if np.isfinite(float(r.get("acc", float("nan"))))]
                    rstds = [float(r.get("recall_std", float("nan"))) for r in rows if np.isfinite(float(r.get("recall_std", float("nan"))))]
                    mean_c, std_c = _macro_mean_std_over_runs(costs, n_subjects, n_runs)
                    mean_a, std_a = _macro_mean_std_over_runs(accs, n_subjects, n_runs)
                    mean_r, std_r = _macro_mean_std_over_runs(rstds, n_subjects, n_runs)
                    mean_nb = float(np.mean([float(r.get("n_base", 0)) for r in rows])) if rows else 0.0
                    mean_ns = float(np.mean([float(r.get("n_swap", 0)) for r in rows])) if rows else 0.0
                    mean_nc = float(np.mean([float(r.get("n_cyclic", 0)) for r in rows])) if rows else 0.0
                    conf_str = f"{float(conf_level):g}"
                    logger.info(
                        f"ours_swap_gaussian_posterior_latin_{conf_str}% : "
                        f"cost={_fmt(mean_c, std_c)}, acc={_fmt4(mean_a, std_a)}, "
                        f"recall_std={_fmt4(mean_r, std_r)}, "
                        f"n_base={mean_nb:.0f}, n_swap={mean_ns:.0f}, n_latin={mean_nc:.0f}"
                    )

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
