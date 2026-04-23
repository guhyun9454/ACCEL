from typing import Dict, List, Optional, Tuple

import numpy as np
import math


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


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _swap_gaussian_th2_value(th1_val: float, mode: str = "half") -> float:
    th1 = max(0.0, min(1.0, float(th1_val)))
    mode_norm = str(mode).lower()
    if mode_norm in {"sqrt", "root", "root_th1"}:
        return float(math.sqrt(th1))
    if mode_norm in {"same", "equal", "identity", "th1"}:
        return float(th1)
    return float(th1 / 2.0)


def _gaussian_swap_posterior_prob(
    y1_base: float,
    y1_swap: float,
    std1_base: float,
    std1_swap: float,
    y2_base: float,
    y2_swap: float,
    std2_base: float,
    std2_swap: float,
) -> float:
    eps = 1e-12
    v11 = max(float(std1_base) ** 2, eps)
    v12 = max(float(std1_swap) ** 2, eps)
    v21 = max(float(std2_base) ** 2, eps)
    v22 = max(float(std2_swap) ** 2, eps)

    prec1 = (1.0 / v11) + (1.0 / v12)
    prec2 = (1.0 / v21) + (1.0 / v22)
    mu1 = ((float(y1_base) / v11) + (float(y1_swap) / v12)) / prec1
    mu2 = ((float(y2_base) / v21) + (float(y2_swap) / v22)) / prec2
    var1 = 1.0 / prec1
    var2 = 1.0 / prec2
    z_std = math.sqrt(max(var1 + var2, eps))
    return float(_normal_cdf((mu1 - mu2) / z_std))


def _gaussian_latin_posterior_from_views(
    latin_probs: np.ndarray,
    latin_perms: List[Tuple[int, ...]],
    rank_map: Dict[int, int],
    rank_slot_std_lookup: Dict[Tuple[int, str], float],
    slot_labels: List[str],
) -> Tuple[int, float, List[float], List[float]]:
    """
    Combine a Latin-square set of observations into per-content Gaussian
    posterior means. Each content is observed once in every displayed slot.

    Returns: (pred_content_idx, confidence_between_top2, posterior_means, posterior_vars)
    """
    probs = np.asarray(latin_probs, dtype=np.float64)
    k = int(len(slot_labels))
    eps = 1e-12
    if probs.ndim != 2 or probs.shape[1] != k or len(latin_perms) != probs.shape[0]:
        raise ValueError("invalid Latin probs/perms shape for Gaussian posterior")

    weighted_sum = np.zeros(k, dtype=np.float64)
    precision_sum = np.zeros(k, dtype=np.float64)
    for row_idx, perm in enumerate(latin_perms):
        if len(perm) != k:
            raise ValueError("invalid Latin permutation length")
        for slot_idx, content_idx_raw in enumerate(perm):
            content_idx = int(content_idx_raw)
            rank = int(rank_map[content_idx])
            slot = str(slot_labels[int(slot_idx)])
            std = float(rank_slot_std_lookup[(rank, slot)])
            var = max(std * std, eps)
            precision = 1.0 / var
            weighted_sum[content_idx] += float(probs[row_idx, slot_idx]) * precision
            precision_sum[content_idx] += precision

    posterior_means = weighted_sum / np.maximum(precision_sum, eps)
    posterior_vars = 1.0 / np.maximum(precision_sum, eps)
    order = np.argsort(posterior_means)[::-1]
    pred_idx = int(order[0]) if order.size > 0 else 0
    if order.size <= 1:
        return pred_idx, 1.0, posterior_means.tolist(), posterior_vars.tolist()
    runner_up = int(order[1])
    denom = math.sqrt(max(float(posterior_vars[pred_idx] + posterior_vars[runner_up]), eps))
    p_top = _normal_cdf(float(posterior_means[pred_idx] - posterior_means[runner_up]) / denom)
    confidence = float(2.0 * abs(p_top - 0.5))
    return pred_idx, confidence, posterior_means.tolist(), posterior_vars.tolist()


def _run_online_latin_gaussian_policy_with_preds(
    default_conf: np.ndarray,
    flip_confidence: np.ndarray,
    flip_pred_idx: List[int],
    latin_pred_idx: List[int],
    base_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    th1_percent: float,
    th2_mode: str = "half",
) -> Tuple[float, float, List[int]]:
    N = len(labels_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    dc = np.asarray(default_conf, dtype=np.float64)
    conf = np.asarray(flip_confidence, dtype=np.float64)
    q = float(th1_percent) / 100.0
    total_cost = 0.0
    corrects = 0
    preds: List[int] = []
    past_dc: List[float] = []
    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q)) if len(past_dc) > 0 else 0.0
        th2_val = _swap_gaussian_th2_value(th1_val, th2_mode)
        if gap_i >= th1_val:
            pred_i = int(base_pred_idx[i])
            c_step = 1.0
        elif float(conf[i]) >= th2_val:
            pred_i = int(flip_pred_idx[i])
            c_step = 2.0
        else:
            pred_i = int(latin_pred_idx[i])
            c_step = float(k)
        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if int(pred_i) == int(labels_idx[i]) else 0
        past_dc.append(gap_i)
    return total_cost / float(N), corrects / float(N), preds


def _run_online_latin_gaussian_policy_with_stats(
    default_conf: np.ndarray,
    flip_confidence: np.ndarray,
    flip_correct: List[bool],
    latin_correct: List[bool],
    base_correct: List[bool],
    k: int,
    th1_percent: float,
    th2_mode: str = "half",
) -> Tuple[float, float, Dict[str, int]]:
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan"), {"n_base": 0, "n_flip": 0, "n_latin": 0, "n_swap": 0, "n_cyclic": 0}
    dc = np.asarray(default_conf, dtype=np.float64)
    conf = np.asarray(flip_confidence, dtype=np.float64)
    q = float(th1_percent) / 100.0
    total_cost = 0.0
    corrects = 0
    n_base = 0
    n_flip = 0
    n_latin = 0
    past_dc: List[float] = []
    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q)) if len(past_dc) > 0 else 0.0
        th2_val = _swap_gaussian_th2_value(th1_val, th2_mode)
        if gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if bool(base_correct[i]) else 0
            n_base += 1
        elif float(conf[i]) >= th2_val:
            total_cost += 2.0
            corrects += 1 if bool(flip_correct[i]) else 0
            n_flip += 1
        else:
            total_cost += float(k)
            corrects += 1 if bool(latin_correct[i]) else 0
            n_latin += 1
        past_dc.append(gap_i)
    return (
        total_cost / float(N),
        corrects / float(N),
        {
            "n_base": int(n_base),
            "n_flip": int(n_flip),
            "n_latin": int(n_latin),
            "n_swap": int(n_flip),
            "n_cyclic": 0,
        },
    )


def _run_online_swap_gaussian_policy_with_preds(
    default_conf: np.ndarray,
    swap_posterior_prob: np.ndarray,
    cand1_pred_idx: List[int],
    cand2_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    th1_percent: float,
    cyclic_pred_idx: Optional[List[int]] = None,
    th2_mode: str = "half",
) -> Tuple[float, float, List[int]]:
    N = len(labels_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    dc = np.asarray(default_conf, dtype=np.float64)
    pp = np.asarray(swap_posterior_prob, dtype=np.float64)
    q = float(th1_percent) / 100.0
    total_cost = 0.0
    corrects = 0
    preds: List[int] = []
    past_dc: List[float] = []
    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q)) if len(past_dc) > 0 else 0.0
        th2_val = _swap_gaussian_th2_value(th1_val, th2_mode)
        posterior_conf = 2.0 * abs(float(pp[i]) - 0.5)
        if gap_i >= th1_val:
            pred_i = int(cand1_pred_idx[i])
            c_step = 1.0
        elif cyclic_pred_idx is not None and posterior_conf < th2_val:
            pred_i = int(cyclic_pred_idx[i])
            c_step = float(k)
        else:
            pred_i = int(cand1_pred_idx[i]) if float(pp[i]) >= 0.5 else int(cand2_pred_idx[i])
            c_step = 2.0
        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if int(pred_i) == int(labels_idx[i]) else 0
        past_dc.append(gap_i)
    return total_cost / float(N), corrects / float(N), preds


def _run_online_swap_gaussian_policy(
    default_conf: np.ndarray,
    swap_posterior_prob: np.ndarray,
    cand1_correct: List[bool],
    cand2_correct: List[bool],
    k: int,
    th1_percent: float,
    cyclic_correct: Optional[List[bool]] = None,
    th2_mode: str = "half",
) -> Tuple[float, float]:
    N = len(cand1_correct)
    if N == 0:
        return float("nan"), float("nan")
    dc = np.asarray(default_conf, dtype=np.float64)
    pp = np.asarray(swap_posterior_prob, dtype=np.float64)
    q = float(th1_percent) / 100.0
    total_cost = 0.0
    corrects = 0
    past_dc: List[float] = []
    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q)) if len(past_dc) > 0 else 0.0
        th2_val = _swap_gaussian_th2_value(th1_val, th2_mode)
        posterior_conf = 2.0 * abs(float(pp[i]) - 0.5)
        if gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if bool(cand1_correct[i]) else 0
        elif cyclic_correct is not None and posterior_conf < th2_val:
            total_cost += float(k)
            corrects += 1 if bool(cyclic_correct[i]) else 0
        else:
            total_cost += 2.0
            corrects += 1 if (bool(cand1_correct[i]) if float(pp[i]) >= 0.5 else bool(cand2_correct[i])) else 0
        past_dc.append(gap_i)
    return total_cost / float(N), corrects / float(N)


def _run_online_swap_gaussian_policy_with_stats(
    default_conf: np.ndarray,
    swap_posterior_prob: np.ndarray,
    cand1_correct: List[bool],
    cand2_correct: List[bool],
    k: int,
    th1_percent: float,
    cyclic_correct: Optional[List[bool]] = None,
    th2_mode: str = "half",
) -> Tuple[float, float, Dict[str, int]]:
    N = len(cand1_correct)
    if N == 0:
        return float("nan"), float("nan"), {"n_base": 0, "n_swap": 0, "n_cyclic": 0}
    dc = np.asarray(default_conf, dtype=np.float64)
    pp = np.asarray(swap_posterior_prob, dtype=np.float64)
    q = float(th1_percent) / 100.0
    total_cost = 0.0
    corrects = 0
    n_base = 0
    n_swap = 0
    n_cyclic = 0
    past_dc: List[float] = []
    for i in range(N):
        gap_i = float(dc[i])
        th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q)) if len(past_dc) > 0 else 0.0
        th2_val = _swap_gaussian_th2_value(th1_val, th2_mode)
        posterior_conf = 2.0 * abs(float(pp[i]) - 0.5)
        if gap_i >= th1_val:
            total_cost += 1.0
            corrects += 1 if bool(cand1_correct[i]) else 0
            n_base += 1
        elif cyclic_correct is not None and posterior_conf < th2_val:
            total_cost += float(k)
            corrects += 1 if bool(cyclic_correct[i]) else 0
            n_cyclic += 1
        else:
            total_cost += 2.0
            corrects += 1 if (bool(cand1_correct[i]) if float(pp[i]) >= 0.5 else bool(cand2_correct[i])) else 0
            n_swap += 1
        past_dc.append(gap_i)
    return total_cost / float(N), corrects / float(N), {"n_base": int(n_base), "n_swap": int(n_swap), "n_cyclic": int(n_cyclic)}


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
            pred_i = int(cyclic_pred_idx[i])

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
            pred_i = int(cyclic_pred_idx[i])

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
            pred_i = int(cyclic_pred_idx[i])

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
            low_sum += gap_i
            low_cnt += 1

        if forced_cyclic_ids is not None and int(i) in forced_cyclic_ids:
            c_step = max(float(c_step), float(k))
            pred_i = int(cyclic_pred_idx[i])

        preds.append(pred_i)
        total_cost += float(c_step)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
        past_gaps.append(gap_i)

    return total_cost / float(N), corrects / float(N), preds


def _run_prefix_cyclic_postfix_base(
    base_correct: List[bool],
    cyclic_correct: List[bool],
    k: int,
    prefix_ids: set,
) -> Tuple[float, float]:
    """
    Default+PRIDE: prefix=cyclic, postfix=base.
    alpha=2 -> front 2% cyclic, remaining debiased base.
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

        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)

        if is_forced:
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
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
    [Online Sqrt Policy - LowConf-only update]
    th1 = online percentile (running quantile over observed default_conf gaps)
    th2 = th1 * sqrt(1 - CurrentAvgGapLowConf) (Online)

    - CurrentAvgGapLowConf: average default_conf over prior low-confidence samples only.
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

        if len(past_gaps) > 0:
            th1_val = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), float(th1_percent) / 100.0))
        else:
            th1_val = 0.0

        if low_cnt > 0:
            current_avg_gap = low_sum / low_cnt
        else:
            current_avg_gap = 0.0

        safe_avg = min(1.0, max(0.0, current_avg_gap))
        current_th2_val = th1_val * np.sqrt(1.0 - safe_avg)
        final_th2_val = current_th2_val

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

        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)

        if is_forced:
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
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
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
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
    """Returns (cost, acc, {n_base, n_cyclic, n_probe2})."""
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

        if len(past_dc) > 0:
            th1_val = float(np.quantile(np.asarray(past_dc, dtype=np.float64), q1))
        else:
            th1_val = 0.0

        if len(past_mc) > 0:
            th2_val = float(np.quantile(np.asarray(past_mc, dtype=np.float64), q2))
        else:
            th2_val = 0.0

        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)

        if is_forced:
            c_step = float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
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
    Same as `_run_online_avggap_policy`, but returns decision counts.
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
    k: int,
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
        if current_th2 < 0.0:
            current_th2 = 0.0
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
