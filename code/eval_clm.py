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


def _stable_u32_seed(s: str, base_seed: int = 0) -> int:
    """Deterministic per-string seed (independent of Python hash randomization)."""
    return (int(zlib.crc32(s.encode("utf-8"))) + int(base_seed)) & 0xFFFFFFFF


def _apply_pride_global_prior_to_probs_seq(
    probs_seq: np.ndarray,
    prior: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    probs_seq: (num_perms, k) letter-space probs
    prior: (k,) global prior over option IDs (letter tokens)
    returns: corrected probs_seq with row-wise renormalization

    PriDe correction: divide by prior then renormalize.
    """
    p = np.asarray(probs_seq, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64).reshape(1, -1)
    adj = p / (pr + eps)
    adj = adj / (adj.sum(axis=1, keepdims=True) + eps)
    return adj


# =========================================================
# PRIDE_FREE (policy-pool based, "free" prior)
# =========================================================
def _quantile(arr: np.ndarray, p01: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    p01 = float(max(0.0, min(1.0, p01)))
    return float(np.quantile(arr, p01))


def _policy_action_beta0(
    policy: str,
    dc_i: float,
    mc_i: float,
    flip_i: bool,
    th1: float,
    th2: float
) -> str:
    """
    Returns action in {"base", "probe2", "cyclic"} for beta=0 (no offline prefix).

    - switch_cyclic:
        if dc >= th1 -> base
        else -> cyclic

    - ours_top2flip:
        if dc >= th1 -> base
        else:
            if flip -> cyclic else probe2

    - ours_avggap:
        if dc >= th1 -> base
        else:
            if mc < th2 -> cyclic else probe2
    """
    if policy == "switch_cyclic":
        return "base" if dc_i >= th1 else "cyclic"
    if policy == "ours_top2flip":
        if dc_i >= th1:
            return "base"
        return "cyclic" if bool(flip_i) else "probe2"
    if policy == "ours_avggap":
        if dc_i >= th1:
            return "base"
        return "cyclic" if (mc_i < th2) else "probe2"
    raise ValueError(f"Unknown policy: {policy}")

def _policy_action(
    policy: str,
    dc_i: float,
    mc_i: float,
    flip_i: bool,
    th1: float,
    th2: float
) -> str:
    """
    Returns action in {"base", "probe2", "cyclic"} given fixed thresholds th1/th2.
    (beta에 따라 th1/th2가 달라지므로, beta0 전용이 아니라 일반 버전으로 분리)
    """
    if policy == "switch_cyclic":
        return "base" if dc_i >= th1 else "cyclic"
    if policy == "ours_top2flip":
        if dc_i >= th1:
            return "base"
        return "cyclic" if bool(flip_i) else "probe2"
    if policy == "ours_avggap":
        if dc_i >= th1:
            return "base"
        return "cyclic" if (mc_i < th2) else "probe2"
    raise ValueError(f"Unknown policy: {policy}")


def _estimate_prior_from_precomputed_priors(
    per_sample_priors: List[np.ndarray],
    pool_ids: List[int],
    k: int,
    eps: float = 1e-12
) -> np.ndarray:
    """
    pool_ids에 해당하는 per-sample prior들을 평균내서 global prior로 사용.
    pool이 비면 uniform.
    """
    if pool_ids is None or len(pool_ids) == 0:
        prior = np.ones((k,), dtype=np.float64)
        return prior / (prior.sum() + eps)

    arr = np.asarray([per_sample_priors[i] for i in pool_ids], dtype=np.float64)  # (m,k)
    prior = np.mean(arr, axis=0)
    prior = np.asarray(prior, dtype=np.float64)
    return prior / (prior.sum() + eps)


def _pride_correct_row(row: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    PriDe: divide by prior then renormalize (row-wise).
    row: (k,)
    prior: (k,)
    """
    r = np.asarray(row, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64)
    adj = r / (pr + eps)
    adj = adj / (adj.sum() + eps)
    return adj


def _predict_cyclic_with_pride(
    ps: np.ndarray,
    prior: np.ndarray,
    k: int,
    option_ids: List[str],
    ideal: str,
    cyclic_indices: List[int],
    cyc_perms: List[Tuple[int, ...]],
) -> bool:
    # cyclic rows만 pride correction 적용
    corrected = []
    for idx in cyclic_indices:
        corrected.append(_pride_correct_row(ps[idx], prior).tolist())
    agg = _aggregate_probs_over_permutations(corrected, cyc_perms, k)
    pred = option_ids[int(np.argmax(agg))]
    return (pred == ideal)


def _predict_probe2_with_pride(
    ps: np.ndarray,
    prior: np.ndarray,
    k: int,
    option_ids: List[str],
    ideal: str,
    identity_idx: int,
    cyclic_indices: List[int],
    cyc_perms: List[Tuple[int, ...]],
) -> bool:
    # probe shift는 "원래 알고리즘"대로 base raw에서 결정(정책과 분리)
    base_raw = np.asarray(ps[identity_idx], dtype=np.float64)
    shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(base_raw, k)
    probe_perm_idx = cyclic_indices[shift]

    base_corr = _pride_correct_row(ps[identity_idx], prior)
    probe_corr = _pride_correct_row(ps[probe_perm_idx], prior)

    agg_base = _aggregate_probs_over_permutations([base_corr.tolist()], [tuple(range(k))], k)
    agg_probe = _aggregate_probs_over_permutations([probe_corr.tolist()], [cyc_perms[shift]], k)

    mean_probs = (agg_base + agg_probe) / 2.0
    pred = option_ids[int(np.argmax(mean_probs))]
    return (pred == ideal)


def _compute_pride_curve_dynamic_prior(
    subject: str,
    tag: str,
    policy_key: str,
    k: int,
    option_ids: List[str],
    perm_list: List[Tuple[int, ...]],
    identity_idx: int,
    cyclic_indices: List[int],
    cyc_perms: List[Tuple[int, ...]],
    per_sample_probs: List[np.ndarray],         # raw (num_perms,k)
    per_sample_priors: List[np.ndarray],        # precomputed prior per sample (k,)
    ideals: List[str],
    base_correct_list: List[bool],              # raw base correctness
    default_conf: np.ndarray,                   # raw dc
    mean_conf: np.ndarray,                      # raw mc
    flip_trigger: np.ndarray,                   # raw flip
    perc_value: float,
    betas: Optional[List[float]] = None,
) -> dict:
    """
    ✅ 너가 원하는 PRIDE:
      - 각 beta마다 online 구간(n..N-1)에서 "정책이 cyclic로 보낸 샘플"로 pool 구성 → prior 추정
      - 같은 beta에서 online 구간의 low-conf(dc<th1) 샘플에만 divide-by-prior 보정 적용 후 평가
      - offline prefix(0..n-1)는 base only + 보정 없음
    """
    if betas is None:
        betas = [i / 10.0 for i in range(11)]

    N = len(base_correct_list)
    if N == 0:
        return {}

    perc01 = float(max(0.0, min(100.0, perc_value))) / 100.0
    C_cyc = float(k)

    costs = []
    accs = []
    meta_per_beta = []

    for beta in betas:
        n = int(N * float(beta) + 1e-9)

        # thresholds는 "offline prefix"에서 추정(기존 코드와 동일한 규칙)
        th1 = _quantile(default_conf[:n], perc01) if n > 0 else _quantile(default_conf, perc01)
        # ours_avggap만 th2 필요
        th2 = _quantile(mean_conf[:n], perc01) if n > 0 else _quantile(mean_conf, perc01)

        # online에서 low-conf + action 분류
        pool_ids = []  # online에서 policy가 cyclic로 보낸 샘플들
        action_map = {}  # i -> action (online, low-conf만 기록)

        for i in range(n, N):
            if float(default_conf[i]) >= float(th1):
                continue  # high-conf, 보정/풀 대상 아님
            act = _policy_action(
                policy=policy_key,
                dc_i=float(default_conf[i]),
                mc_i=float(mean_conf[i]),
                flip_i=bool(flip_trigger[i]),
                th1=float(th1),
                th2=float(th2),
            )
            action_map[i] = act
            if act == "cyclic":
                pool_ids.append(i)

        # prior(beta) = mean of per-sample priors over pool
        prior_beta = _estimate_prior_from_precomputed_priors(per_sample_priors, pool_ids, k)

        # ---- 평가: offline prefix는 base only ----
        total_cost = float(n) * 1.0
        corrects = int(np.sum(np.asarray(base_correct_list[:n], dtype=np.int64)))

        # ---- online 평가 ----
        for i in range(n, N):
            if float(default_conf[i]) >= float(th1):
                # base, 보정 없음
                total_cost += 1.0
                corrects += 1 if base_correct_list[i] else 0
                continue

            # low-conf: prior로 나눠서 보정 적용
            ps = np.asarray(per_sample_probs[i], dtype=np.float64)
            act = action_map.get(i, "base")  # low-conf면 항상 들어있어야 함

            if act == "cyclic":
                total_cost += C_cyc
                if _predict_cyclic_with_pride(ps, prior_beta, k, option_ids, ideals[i], cyclic_indices, cyc_perms):
                    corrects += 1
            elif act == "probe2":
                total_cost += 2.0
                if _predict_probe2_with_pride(ps, prior_beta, k, option_ids, ideals[i], identity_idx, cyclic_indices, cyc_perms):
                    corrects += 1
            else:
                # 이 케이스는 low-conf인데 base로 떨어지는 경우라 거의 없어야 함(정책 정의상)
                total_cost += 1.0
                corrects += 1 if base_correct_list[i] else 0

        costs.append(total_cost / float(N))
        accs.append(float(corrects) / float(N))

        meta_per_beta.append({
            "beta": float(beta),
            "n_offline": int(n),
            "th1": float(th1),
            "th2": float(th2),
            "pool_size": int(len(pool_ids)),
            "prior": [float(x) for x in prior_beta.tolist()],
            "prior_map": {str(opt): float(v) for opt, v in zip(option_ids, prior_beta.tolist())},
        })

    curve_obj = {
        "subject": subject,
        "tag": str(tag),
        "k": int(k),
        "percentile": float(perc_value),
        "betas": [float(b) for b in betas],
        "default_accuracy": float(np.mean(np.asarray(base_correct_list, dtype=np.float64))),

        "always": {
            "default": {"cost": 1.0, "acc": float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))},
            "cyclic": {"cost": float(k), "acc": float("nan")},  # 필요하면 넣고, 아니면 plot에서 안 써도 됨
            "full": {"cost": float(len(perm_list)), "acc": float("nan")},
        },

        policy_key: {"costs": [float(c) for c in costs], "accuracies": [float(a) for a in accs]},

        "pride_free_dynamic": {
            "pool_policy": policy_key,
            "pool_from": "online_only_and_action==cyclic",
            "apply_to": "online_only_and_dc<th1 (low-conf)",
            "per_beta_meta": meta_per_beta,
        }
    }
    return curve_obj


def _collect_policy_pool_ids_beta0(
    
    policy: str,
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    flip_trigger: np.ndarray,
    perc_value: float
) -> Tuple[List[int], Dict[str, float]]:
    """
    Pool IDs = EXACTLY those samples that THIS policy sends to CYCLIC at beta=0.
    percentile p로 th1/th2를 잡는다 (전체 데이터 기반, beta=0).
    """
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    fl = np.asarray(flip_trigger, dtype=bool)

    perc01 = float(max(0.0, min(100.0, float(perc_value)))) / 100.0
    th1 = _quantile(dc, perc01)
    th2 = _quantile(mc, perc01)

    pool = []
    for i in range(len(dc)):
        act = _policy_action_beta0(policy, float(dc[i]), float(mc[i]), bool(fl[i]), th1, th2)
        if act == "cyclic":
            pool.append(i)

    meta = {"th1": float(th1), "th2": float(th2), "perc": float(perc_value)}
    return pool, meta


def _estimate_pride_prior_from_pool_ids(
    per_sample_probs: List[np.ndarray],
    cyclic_indices: List[int],
    pool_ids: List[int],
    ema_alpha: float = 0.0,   # NOTE: we will force no-EMA in main (no alpha).
    eps: float = 1e-12
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    per_sample_probs[i] : (num_perms, k)
    pool_ids만 대상으로, 그 중에서도 cyclic_indices(=k rotations) rows만 사용해서 prior를 추정.

    - ema_alpha>0: pool 순서대로 EMA 업데이트
    - ema_alpha==0: pool priors 평균

    Returns:
      prior_global (k,)
      meta dict
    """
    if len(per_sample_probs) == 0:
        return np.array([], dtype=np.float64), {"pool_size": 0, "method": "empty"}

    k = int(np.asarray(per_sample_probs[0], dtype=np.float64).shape[-1])

    if pool_ids is None or len(pool_ids) == 0:
        prior = np.ones((k,), dtype=np.float64)
        prior = prior / (prior.sum() + eps)
        return prior, {"pool_size": 0, "method": "uniform"}

    ema_alpha = float(max(0.0, min(1.0, float(ema_alpha))))

    if ema_alpha > 0.0:
        prior = np.ones((k,), dtype=np.float64)
        prior = prior / (prior.sum() + eps)
        used = 0
        for i in pool_ids:
            ps = np.asarray(per_sample_probs[i], dtype=np.float64)
            observed = np.asarray([ps[j] for j in cyclic_indices], dtype=np.float64)  # (k,k)
            _, _, prior_i = debias_simple(observed)
            prior_i = np.asarray(prior_i, dtype=np.float64)
            prior_i = prior_i / (prior_i.sum() + eps)
            prior = (1.0 - ema_alpha) * prior + ema_alpha * prior_i
            prior = prior / (prior.sum() + eps)
            used += 1
        meta = {"pool_size": int(len(pool_ids)), "used": int(used), "method": f"ema(alpha={ema_alpha:.3f})"}
        return prior, meta

    priors = []
    for i in pool_ids:
        ps = np.asarray(per_sample_probs[i], dtype=np.float64)
        observed = np.asarray([ps[j] for j in cyclic_indices], dtype=np.float64)  # (k,k)
        _, _, prior_i = debias_simple(observed)
        prior_i = np.asarray(prior_i, dtype=np.float64)
        prior_i = prior_i / (prior_i.sum() + eps)
        priors.append(prior_i)

    prior_global = np.mean(np.asarray(priors, dtype=np.float64), axis=0)
    prior_global = prior_global / (prior_global.sum() + eps)

    meta = {"pool_size": int(len(pool_ids)), "used": int(len(priors)), "method": "mean"}
    return prior_global, meta


# =========================================================
# Plot helpers (5 figures only)
# =========================================================
def _plot_pride_core_plot(
    baseline_curve_obj: dict,
    pride_curve_by_policy: Dict[str, dict],
    out_path: str,
    title: str,
):
    """
    (1) PRIDE core plot 1장 (single figure):
      - baseline lines: cyclic + switch_cyclic + ours_top2flip
      - PRIDE lines   : switch_cyclic(PRIDE) + ours_top2flip(PRIDE) + ours_avggap(PRIDE)
      - points        : default(ens) + cyclic_ensemble point
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=200)

    # ----- points: default + cyclic ensemble (no annotate, legend only) -----
    always = baseline_curve_obj.get("always", {})
    if "default" in always:
        ax.scatter(
            [float(always["default"]["cost"])],
            [float(always["default"]["acc"])],
            marker="*",
            s=140,
            label="default",
        )
    if "cyclic" in always:
        ax.scatter(
            [float(always["cyclic"]["cost"])],
            [float(always["cyclic"]["acc"])],
            marker="D",
            s=70,
            label="cyclic_ensemble",
        )

    def _plot_curve_obj(curve_obj: dict, key: str, label: str,
                        lw: float = 1.8, ls: str = "-", marker: str = "o"):
        if curve_obj is None or key not in curve_obj:
            return
        xs = [float(v) for v in curve_obj[key]["costs"]]
        ys = [float(v) for v in curve_obj[key]["accuracies"]]
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, linestyle=ls, marker=marker, linewidth=lw, markersize=4, label=label)

    # ----- baseline (solid) -----
    _plot_curve_obj(baseline_curve_obj, "cyclic", "cyclic", ls="-", lw=1.8)
    _plot_curve_obj(baseline_curve_obj, "switch_cyclic", "switch_cyclic", ls="-", lw=1.8)
    _plot_curve_obj(baseline_curve_obj, "ours_top2flip", "ours_top2flip", ls="-", lw=1.8)

    # ----- PRIDE (dashed) -----
    if "switch_cyclic" in pride_curve_by_policy:
        _plot_curve_obj(pride_curve_by_policy["switch_cyclic"], "switch_cyclic", "switch_cyclic(PRIDE)", ls="--", lw=2.2)
    if "ours_top2flip" in pride_curve_by_policy:
        _plot_curve_obj(pride_curve_by_policy["ours_top2flip"], "ours_top2flip", "ours_top2flip(PRIDE)", ls="--", lw=2.2)
    if "ours_avggap" in pride_curve_by_policy:
        _plot_curve_obj(pride_curve_by_policy["ours_avggap"], "ours_avggap", "ours_avggap(PRIDE)", ls="--", lw=2.2)

    ax.set_xlabel("Computational Cost (× of default)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.30)

    ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=1.0, fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])  # legend 공간 확보
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _plot_avggap_baseline_vs_pride_points(
    baseline_curve_obj: dict,
    pride_curve_obj: dict,
    out_path: str,
    title: str
):
    """
    (2) AvgGap baseline vs PRIDE 1장:
      - two lines: ours_avggap(base) vs ours_avggap(PRIDE)
      - reference: baseline_default + pride_default (points) + cyclic baseline (optional)
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=200)

    # baseline default point
    always_b = baseline_curve_obj.get("always", {})
    if "default" in always_b:
        ax.scatter(
            [float(always_b["default"]["cost"])],
            [float(always_b["default"]["acc"])],
            marker="*",
            s=140,
            label="baseline_default",
        )

    # pride default point (있으면 같이 표시)
    always_p = pride_curve_obj.get("always", {}) if isinstance(pride_curve_obj, dict) else {}
    if "default" in always_p:
        ax.scatter(
            [float(always_p["default"]["cost"])],
            [float(always_p["default"]["acc"])],
            marker="*",
            s=140,
            label="pride_default",
        )

    def _plot_curve(obj: dict, key: str, label: str, lw: float, ls: str):
        if obj is None or key not in obj:
            return
        xs = [float(v) for v in obj[key]["costs"]]
        ys = [float(v) for v in obj[key]["accuracies"]]
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, linestyle=ls, marker="o", linewidth=lw, markersize=6, label=label)

    _plot_curve(baseline_curve_obj, "ours_avggap", "baseline_ours_avggap", lw=2.0, ls="-")
    _plot_curve(pride_curve_obj, "ours_avggap", "pride_ours_avggap", lw=2.0, ls="--")

    ax.set_xlabel("Computational Cost (× of default)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.30)

    # ✅ 네 예시처럼 좌상단 legend 박스
    ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=1.0, fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _plot_heatmap_with_text(
    acc_grid: np.ndarray,
    cost_grid: np.ndarray,
    x_ticks: List[float],
    y_ticks: List[float],
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    mode: str = "base"
):
    """
    (3)(4)(5) heatmap 3장:
      - baseline: cell "acc\\nc=cost"
      - pride   : cell "acc\\nc=cost"
      - delta   : cell "Δacc\\nΔc"

    NOTE: y_ticks includes 0 -> th2=0 row exists.
    """
    plt.figure(figsize=(8.8, 7.2), dpi=220)
    im = plt.imshow(acc_grid, aspect='auto', origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    xt = list(range(len(x_ticks)))
    yt = list(range(len(y_ticks)))
    plt.xticks(xt, [f"{x:.0f}" for x in x_ticks], rotation=45, ha='right')
    plt.yticks(yt, [f"{y:.0f}" for y in y_ticks])

    ny, nx = acc_grid.shape
    for iy in range(ny):
        for ix in range(nx):
            a = acc_grid[iy, ix]
            c = cost_grid[iy, ix]
            if np.isnan(a) or np.isnan(c):
                txt = "nan"
            else:
                if mode == "delta":
                    txt = f"{a:+.3f}\n{c:+.2f}"
                else:
                    txt = f"{a:.3f}\nc={c:.2f}"
            plt.text(ix, iy, txt, ha='center', va='center', fontsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


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

    # 3. Extra Points (th1/2, th1^2, th1^1.5)
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
    [Online Adaptive Policy]
    th2 = th1 * sqrt(1 - CurrentAvgGap)
    
    - CurrentAvgGap: 현재 시점(t)까지 관측된 default_conf들의 평균 (Running Average)
    - 미래 정보(Oracle)를 사용하지 않음.
    
    Returns:
        avg_cost, accuracy, final_converged_th2_percentile
    """
    N = len(base_correct)
    if N == 0:
        return 0.0, 0.0, 0.0

    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    th1_val = _quantile(dc, float(th1_percent) / 100.0)

    total_cost = 0.0
    corrects = 0
    
    running_gap_sum = 0.0
    running_cnt = 0
    
    final_th2_val = 0.0

    for i in range(N):
        gap_i = float(dc[i])
        
        # 1. Update Running Average (Decision based on PAST info)
        if running_cnt > 0:
            current_avg_gap = running_gap_sum / running_cnt
        else:
            current_avg_gap = 0.0
            
        # 2. Dynamic th2 Calculation
        safe_avg = min(1.0, max(0.0, current_avg_gap))
        current_th2_val = th1_val * np.sqrt(1.0 - safe_avg)
        final_th2_val = current_th2_val

        # 3. Policy Execution
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
        
        # 4. Update Stats
        running_gap_sum += gap_i
        running_cnt += 1

    # Calculate equivalent percentile for the final th2 value for plotting
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
    th1/th2 trade-off plot: 
    - th1을 고정하고, th2를 0부터 촘촘하게 변화시켜 부드러운 곡선 생성
    - Online Sqrt(1-Gap) 포인트 표시
    """
    default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    
    # 촘촘한 th2 범위 생성 (0 ~ max(th2_list) + 여유분, step=1)
    max_th2 = max(th2_list) if len(th2_list) > 0 else 30
    dense_th2_list = np.arange(0, max_th2 + 2, 1.0).tolist()
    
    # Plot 1: Cost vs th2
    fig1, ax1 = plt.subplots(figsize=(8.0, 5.5), dpi=160)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        # 1) Dense Curve
        costs = []
        for th2p in dense_th2_list:
            c, _ = _policy_metrics_avggap_beta0(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
            )
            costs.append(c)
        ax1.plot(dense_th2_list, costs, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.8)

        # 2) [ONLINE] th1 * sqrt(1 - AvgGap)
        c_sqrt, a_sqrt, final_th2_perc = _run_online_sqrt_policy(
            default_conf=default_conf,
            mean_conf=mean_conf,
            base_correct=base_correct_list,
            cyclic_correct=cyclic_correct_list,
            probe2_correct=arr_probe2_correct,
            k=k,
            th1_percent=th1p
        )
        ax1.scatter([final_th2_perc], [c_sqrt], marker='D', s=100, color=color, edgecolors='black', zorder=5, label='Online Sqrt' if idx==0 else "")

    ax1.set_xlabel("th2 (percentile, avg gap)", fontsize=11)
    ax1.set_ylabel("Computational Cost (× of default)", fontsize=11)
    ax1.set_title(f"{getattr(args, 'task', 'task')} {subject} — Cost vs th2 (Online)", fontsize=12)
    ax1.set_xticks(th2_list) # 사용자가 지정한 주요 틱만 표시
    ax1.set_xticklabels([f"{int(t)}" for t in th2_list])
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    out_cost = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST.png")
    os.makedirs(os.path.dirname(out_cost), exist_ok=True)
    fig1.tight_layout()
    fig1.savefig(out_cost, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: Δ Accuracy (%) vs th2
    fig2, ax2 = plt.subplots(figsize=(8.0, 5.5), dpi=160)
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        # 1) Dense Curve
        accs = []
        for th2p in dense_th2_list:
            _, a = _policy_metrics_avggap_beta0(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
            )
            accs.append(a)
        
        delta_accs = [(a - default_acc) * 100.0 for a in accs]
        ax2.plot(dense_th2_list, delta_accs, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.8)

        # 2) [ONLINE] th1 * sqrt(1 - AvgGap)
        c_sqrt, a_sqrt, final_th2_perc = _run_online_sqrt_policy(
            default_conf=default_conf,
            mean_conf=mean_conf,
            base_correct=base_correct_list,
            cyclic_correct=cyclic_correct_list,
            probe2_correct=arr_probe2_correct,
            k=k,
            th1_percent=th1p
        )
        d_sqrt = (a_sqrt - default_acc) * 100.0
        ax2.scatter([final_th2_perc], [d_sqrt], marker='D', s=100, color=color, edgecolors='black', zorder=5)

    ax2.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("th2 (percentile, avg gap)", fontsize=11)
    ax2.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax2.set_title(f"{getattr(args, 'task', 'task')} {subject} — Δ Accuracy vs th2 (Online)", fontsize=12)
    ax2.set_xticks(th2_list)
    ax2.set_xticklabels([f"{int(t)}" for t in th2_list])
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    out_delta = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_DELTA_ACC.png")
    fig2.tight_layout()
    fig2.savefig(out_delta, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: Cost vs Δ Accuracy
    fig3, ax3 = plt.subplots(figsize=(8.0, 5.5), dpi=160)
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        costs_line = []
        delta_accs_line = []
        for th2p in dense_th2_list:
            c, a = _policy_metrics_avggap_beta0(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
            )
            costs_line.append(c)
            delta_accs_line.append((a - default_acc) * 100.0)
        
        ax3.plot(costs_line, delta_accs_line, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.4)
        
        # --- Single Point Comparison (Online Sqrt(1-Gap)) ---
        
        # [ONLINE] th1 * sqrt(1 - AvgGap)
        c4, a4, _ = _run_online_sqrt_policy(
            default_conf=default_conf,
            mean_conf=mean_conf,
            base_correct=base_correct_list,
            cyclic_correct=cyclic_correct_list,
            probe2_correct=arr_probe2_correct,
            k=k,
            th1_percent=th1p
        )
        d4 = (a4 - default_acc) * 100.0
        ax3.scatter([c4], [d4], marker='D', s=100, color=color, edgecolors='black', zorder=5, label='Online Sqrt' if idx==0 else "")
        
        ax3.annotate(f"{d4:.2f}", (c4, d4), xytext=(0, 6), textcoords='offset points', fontsize=8, ha='center', color=color, fontweight='bold')

        # Report Log
        logger.info(f"[th1={int(th1p):<2}] Online Sqrt : cost={c4:.3f}, acc={a4:.4f} (+{d4:.2f}%)")

    ax3.scatter([1.0], [0.0], marker='*', s=200, label='default', color='gray', zorder=5)
    ax3.set_xlabel("Computational Cost (× of default)", fontsize=11)
    ax3.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax3.set_title(f"{getattr(args, 'task', 'task')} {subject} — Trade-off (Online Sqrt)", fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.4)
    out_trade = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST_vs_DELTA.png")
    fig3.tight_layout()
    fig3.savefig(out_trade, bbox_inches="tight")
    plt.close(fig3)

    logger.info(_purple(f"th2 trade-off plots saved (dense curve + half-th1 star): {subject}"))
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


def _policy_metrics_avggap_beta0(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct: List[bool],
    cyclic_correct: List[bool],
    probe2_correct: np.ndarray,
    k: int,
    th1_percent: float,
    th2_percent: float
) -> Tuple[float, float]:
    """
    ours_avggap 정책을 beta=0에서 (th1, th2 percentile)로 평가.
    return: (avg_cost, acc)
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan")
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    th1 = _quantile(dc, float(th1_percent) / 100.0)
    th2 = _quantile(mc, float(th2_percent) / 100.0)

    total_cost = 0.0
    corrects = 0
    for i in range(N):
        if float(dc[i]) >= th1:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
        else:
            if float(mc[i]) < th2:
                total_cost += float(k)
                corrects += 1 if cyclic_correct[i] else 0
            else:
                total_cost += 2.0
                corrects += 1 if bool(probe2_correct[i]) else 0

    return total_cost / float(N), corrects / float(N)

def _policy_metrics_avggap_pride_dynamic(
    # raw gating stats
    default_conf: np.ndarray,
    mean_conf: np.ndarray,

    # raw base correctness (offline/highconf는 보정 안 하니까 그대로 씀)
    base_correct: List[bool],

    # raw per-sample info
    per_sample_priors: List[np.ndarray],   # (N,k) list of priors (cyclic rows 기반)
    per_sample_probs: List[np.ndarray],    # (N,num_perms,k) raw probs
    ideals: List[str],

    # permutation helpers
    k: int,
    option_ids: List[str],
    identity_idx: int,
    cyclic_indices: List[int],
    cyc_perms: List[Tuple[int, ...]],

    # threshold percentiles (heatmap axes)
    th1_percent: float,
    th2_percent: float,

    # beta control (heatmap을 beta=0 말고도 만들 수 있게)
    beta: float = 0.0,
    eps: float = 1e-12
) -> Tuple[float, float, int]:
    """
    PRIDE_FREE_DYNAMIC 정의에 맞춘 avggap 정책 평가:

    - offline prefix (0..n-1): base only, 보정 없음
    - online (n..N-1):
        - dc>=th1 : base, 보정 없음
        - dc<th1  : (low-conf)
            - pool = (action==cyclic)인 low-conf 샘플들로 prior(beta,th1,th2) 추정
            - action==cyclic : cyclic prediction에만 보정 적용
            - action==probe2 : base/probe2 mean prediction에만 보정 적용
    return: (avg_cost, acc, pool_size)
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan"), 0

    n = int(N * float(beta) + 1e-9)

    # thresholds are estimated from offline prefix (기존 규칙과 동일)
    th1 = _quantile(default_conf[:n], float(th1_percent) / 100.0) if n > 0 else _quantile(default_conf, float(th1_percent) / 100.0)
    th2 = _quantile(mean_conf[:n], float(th2_percent) / 100.0) if n > 0 else _quantile(mean_conf, float(th2_percent) / 100.0)

    # --------------------------
    # 1) build pool from online low-conf samples that go to cyclic
    # --------------------------
    pool_ids = []
    for i in range(n, N):
        if float(default_conf[i]) >= float(th1):
            continue
        # ours_avggap: low-conf -> cyclic if mc < th2 else probe2
        if float(mean_conf[i]) < float(th2):
            pool_ids.append(i)

    # prior(beta,cell) = mean of per-sample priors in pool (empty -> uniform)
    prior = _estimate_prior_from_precomputed_priors(per_sample_priors, pool_ids, k, eps=eps)

    # --------------------------
    # 2) evaluate with "apply correction only to online low-conf"
    # --------------------------
    total_cost = 0.0
    corrects = 0
    C_cyc = float(k)

    # offline prefix
    for i in range(0, n):
        total_cost += 1.0
        corrects += 1 if base_correct[i] else 0

    # online
    for i in range(n, N):
        dc = float(default_conf[i])
        if dc >= float(th1):
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
            continue

        # low-conf: PRIDE 적용
        mc = float(mean_conf[i])
        ps = np.asarray(per_sample_probs[i], dtype=np.float64)

        if mc < float(th2):
            # action == cyclic
            total_cost += C_cyc
            # cyclic rows만 correction
            corrected = []
            for idx in cyclic_indices:
                corrected.append(_pride_correct_row(ps[idx], prior, eps=eps).tolist())
            agg = _aggregate_probs_over_permutations(corrected, cyc_perms, k)
            pred = option_ids[int(np.argmax(agg))]
            corrects += 1 if (pred == ideals[i]) else 0
        else:
            # action == probe2
            total_cost += 2.0

            base_raw = np.asarray(ps[identity_idx], dtype=np.float64)
            shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(base_raw, k)
            probe_perm_idx = cyclic_indices[shift]

            base_corr = _pride_correct_row(ps[identity_idx], prior, eps=eps)  # identity라 content-space 그대로
            probe_corr = _pride_correct_row(ps[probe_perm_idx], prior, eps=eps)
            agg_probe = _aggregate_probs_over_permutations([probe_corr.tolist()], [cyc_perms[shift]], k)

            mean_probs = (base_corr + agg_probe) / 2.0
            pred = option_ids[int(np.argmax(mean_probs))]
            corrects += 1 if (pred == ideals[i]) else 0

    return total_cost / float(N), corrects / float(N), int(len(pool_ids))

# =========================================================
# Curves: baseline (all methods) + PRIDE_FREE (single policy)
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
    C_full = float(len(perm_list))

    default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    cyclic_acc_always = float(np.mean(np.asarray(cyclic_correct_list, dtype=np.float64)))
    full_acc_always = float(np.mean(np.asarray(full_correct_list, dtype=np.float64)))

    # 1) cyclic/full beta curves
    curve_cyc = []
    curve_full = []
    for beta in betas:
        n = int(N * beta + 1e-9)

        acc_cyc = (sum(base_correct_list[:n]) + sum(cyclic_correct_list[n:])) / float(N)
        acc_full = (sum(base_correct_list[:n]) + sum(full_correct_list[n:])) / float(N)

        cost_cyc = beta * 1.0 + (1.0 - beta) * C_cyc
        cost_full = beta * 1.0 + (1.0 - beta) * C_full

        curve_cyc.append((cost_cyc, acc_cyc))
        curve_full.append((cost_full, acc_full))

    # 2) switch curves
    curve_switch_full = []
    curve_switch_cyc = []
    for beta in betas:
        n = int(N * beta + 1e-9)
        thresh = _quantile(default_conf[:n], perc01) if n > 0 else _quantile(default_conf, perc01)

        total_cost_sf = 0.0
        corrects_sf = 0
        total_cost_sc = 0.0
        corrects_sc = 0

        # offline prefix: base only
        for i in range(0, n):
            total_cost_sf += 1.0
            total_cost_sc += 1.0
            if base_correct_list[i]:
                corrects_sf += 1
                corrects_sc += 1

        # online: apply switch
        for i in range(n, N):
            amb = (float(default_conf[i]) < thresh)

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

        curve_switch_full.append((total_cost_sf / float(N), corrects_sf / float(N)))
        curve_switch_cyc.append((total_cost_sc / float(N), corrects_sc / float(N)))

    # 3) ours_top2flip
    curve_top2flip = []
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

        curve_top2flip.append((total_cost / float(N), corrects / float(N)))

    # 4) ours_avggap
    curve_avggap = []
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

        curve_avggap.append((total_cost / float(N), corrects / float(N)))

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

        "cyclic": {"costs": [float(c) for c, _ in curve_cyc], "accuracies": [float(a) for _, a in curve_cyc]},
        "full": {"costs": [float(c) for c, _ in curve_full], "accuracies": [float(a) for _, a in curve_full]},
        "switch_full": {"costs": [float(c) for c, _ in curve_switch_full], "accuracies": [float(a) for _, a in curve_switch_full]},
        "switch_cyclic": {"costs": [float(c) for c, _ in curve_switch_cyc], "accuracies": [float(a) for _, a in curve_switch_cyc]},
        "ours_top2flip": {"costs": [float(c) for c, _ in curve_top2flip], "accuracies": [float(a) for _, a in curve_top2flip]},
        "ours_avggap": {"costs": [float(c) for c, _ in curve_avggap], "accuracies": [float(a) for _, a in curve_avggap]},
    }
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
    logger.info(_purple(f"==== BASELINE Derived policy report (beta=0, p={p}) ===="))

    always = curve_obj.get("always", {})
    logger.info(f"BASELINE default(ensemble) : cost={always['default']['cost']:.3f}, acc={always['default']['acc']:.4f}")
    logger.info(f"BASELINE cyclic(ensemble)  : cost={always['cyclic']['cost']:.3f}, acc={always['cyclic']['acc']:.4f}")
    logger.info(f"BASELINE full(ensemble)    : cost={always['full']['cost']:.3f}, acc={always['full']['acc']:.4f}")

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
                "disable_pride": getattr(args, "disable_pride", False),
                # We FORCE no alpha (no EMA) for PRIDE in this script.
                "pride_ema_alpha_forced": 0.0,
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
            # Derived policies & PRIDE_FREE (ONLY when args.setting == 'full')
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

                        # full (all perms)
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
                    logger.info(_purple(f"[{subject}] Accuracies — Full: {full_acc:.4f}, Cyclic: {cyclic_acc:.4f}, Default: {base_acc:.4f}"))

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
                            full_correct_list=full_correct_list,
                            default_conf=default_conf,
                            mean_conf=mean_conf,
                            flip_trigger=arr_flip_trigger,
                            probe2_correct=arr_probe2_correct,
                            perc_value=perc,
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
                            
                            # 3. th1 * sqrt(1 - AvgGap) [ONLINE] ONLY
                            c4, a4, _ = _run_online_sqrt_policy(
                                default_conf=default_conf,
                                mean_conf=mean_conf,
                                base_correct=base_correct_list,
                                cyclic_correct=cyclic_correct_list,
                                probe2_correct=arr_probe2_correct,
                                k=k,
                                th1_percent=th1p
                            )
                            extra_pts.append({'cost': c4, 'acc': a4, 'label': 'Online Sqrt', 'marker': 'D', 'color': 'orange'})

                            _plot_baseline_points_scatter(
                                curve_obj=cobj,
                                out_path=out_pts,
                                title=f"{args.task} {subject} — Baseline Policies (beta=0, {ptag}, Online)",
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
                        getattr(args, "ours_th2_tradeoff", "5,10,20,30")
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
