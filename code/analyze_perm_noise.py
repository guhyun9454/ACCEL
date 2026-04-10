#!/usr/bin/env python3
"""
Permutation noise / ensemble diversity analysis.

This script is meant to be run AFTER `eval_clm.py` produced cached jsonl results:
  - <cache_dir>/<subject>.jsonl               (n_runs=1)
  - <cache_dir>/<subject>_run{i}.jsonl        (n_runs>1)

Each jsonl line is expected to contain:
  {"type":"result","data":{"idx":..., "options":[...], "probs":[[...],[...],...], "ideal":"A"|...}}

We compute:
  - Original (identity) accuracy
  - Per-permutation accuracies (cyclic rotations subset + full permutations)
  - Mean/variance over permutation accuracies
  - Pairwise correlation between permutation correctness vectors (simple diversity proxy)
  - Ensemble accuracy vs number of permutations mixed (random subset sampling)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import zlib
from dataclasses import dataclass
from itertools import combinations, permutations
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _try_import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def _rotations(k: int) -> List[Tuple[int, ...]]:
    """cyclic rotations: ABCD, BCDA, CDAB, DABC"""
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(
    probs_seq: Sequence[Sequence[float]],
    permuted_indices: Sequence[Tuple[int, ...]],
    k: int,
) -> np.ndarray:
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


def _gap_of_distribution(probs: Sequence[float]) -> float:
    vals = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    if vals.size <= 1:
        return 0.0
    return float(vals[0] - vals[1])


def _top2_indices(probs: Sequence[float]) -> Tuple[int, int]:
    vals = np.argsort(np.asarray(probs, dtype=np.float64))[::-1]
    if vals.size == 0:
        return 0, 0
    if vals.size == 1:
        return int(vals[0]), int(vals[0])
    return int(vals[0]), int(vals[1])


def _perm_label(perm: Sequence[int], k: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if int(k) <= len(alphabet):
        symbols = alphabet[: int(k)]
    else:
        symbols = "".join(str(i) for i in range(int(k)))
    return "".join(symbols[int(x)] for x in perm)


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
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


def _gaussian_fit_report(values: Sequence[float], bins: int = 60) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    out: Dict[str, float] = {
        "n": int(arr.size),
        "mean": float("nan"),
        "std": float("nan"),
        "skew": float("nan"),
        "excess_kurtosis": float("nan"),
        "ks_to_fit": float("nan"),
        "kl_hist_to_fit": float("nan"),
    }
    if arr.size == 0:
        return out

    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    out["mean"] = mu
    out["std"] = sigma

    if not np.isfinite(sigma) or sigma <= 1e-12 or arr.size < 2:
        return out

    z = (arr - mu) / sigma
    out["skew"] = float(np.mean(z ** 3))
    out["excess_kurtosis"] = float(np.mean(z ** 4) - 3.0)

    xs = np.sort(arr)
    nd = NormalDist(mu=mu, sigma=sigma)
    fit_cdf = np.asarray([nd.cdf(float(v)) for v in xs], dtype=np.float64)
    n = xs.size
    ecdf_hi = np.arange(1, n + 1, dtype=np.float64) / float(n)
    ecdf_lo = np.arange(0, n, dtype=np.float64) / float(n)
    ks = max(float(np.max(np.abs(ecdf_hi - fit_cdf))), float(np.max(np.abs(fit_cdf - ecdf_lo))))
    out["ks_to_fit"] = ks

    hist_counts, edges = np.histogram(arr, bins=max(10, int(bins)))
    total = int(np.sum(hist_counts))
    if total > 0:
        p = hist_counts.astype(np.float64) / float(total)
        q = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            q.append(max(1e-12, float(nd.cdf(float(hi)) - nd.cdf(float(lo)))))
        q = np.asarray(q, dtype=np.float64)
        q = q / float(np.sum(q))
        mask = p > 0.0
        out["kl_hist_to_fit"] = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return out


def _laplace_fit_report(values: Sequence[float], bins: int = 60) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    out: Dict[str, float] = {
        "n": int(arr.size),
        "loc": float("nan"),
        "scale": float("nan"),
        "ks_to_fit": float("nan"),
        "kl_hist_to_fit": float("nan"),
    }
    if arr.size == 0:
        return out

    loc = float(np.median(arr))
    scale = float(np.mean(np.abs(arr - loc)))
    out["loc"] = loc
    out["scale"] = scale
    if not np.isfinite(scale) or scale <= 1e-12 or arr.size < 2:
        return out

    def _cdf(x: float) -> float:
        if x < loc:
            return 0.5 * math.exp((x - loc) / scale)
        return 1.0 - 0.5 * math.exp(-(x - loc) / scale)

    xs = np.sort(arr)
    fit_cdf = np.asarray([_cdf(float(v)) for v in xs], dtype=np.float64)
    n = xs.size
    ecdf_hi = np.arange(1, n + 1, dtype=np.float64) / float(n)
    ecdf_lo = np.arange(0, n, dtype=np.float64) / float(n)
    out["ks_to_fit"] = max(float(np.max(np.abs(ecdf_hi - fit_cdf))), float(np.max(np.abs(fit_cdf - ecdf_lo))))

    hist_counts, edges = np.histogram(arr, bins=max(10, int(bins)))
    total = int(np.sum(hist_counts))
    if total > 0:
        p = hist_counts.astype(np.float64) / float(total)
        q = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            q.append(max(1e-12, float(_cdf(float(hi)) - _cdf(float(lo)))))
        q = np.asarray(q, dtype=np.float64)
        q = q / float(np.sum(q))
        mask = p > 0.0
        out["kl_hist_to_fit"] = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return out


def _cauchy_fit_report(values: Sequence[float], bins: int = 60) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    out: Dict[str, float] = {
        "n": int(arr.size),
        "loc": float("nan"),
        "scale": float("nan"),
        "ks_to_fit": float("nan"),
        "kl_hist_to_fit": float("nan"),
    }
    if arr.size == 0:
        return out

    loc = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    scale = float(max(1e-12, 0.5 * (q75 - q25)))
    out["loc"] = loc
    out["scale"] = scale
    if not np.isfinite(scale) or scale <= 1e-12 or arr.size < 2:
        return out

    def _cdf(x: float) -> float:
        return 0.5 + math.atan((x - loc) / scale) / math.pi

    xs = np.sort(arr)
    fit_cdf = np.asarray([_cdf(float(v)) for v in xs], dtype=np.float64)
    n = xs.size
    ecdf_hi = np.arange(1, n + 1, dtype=np.float64) / float(n)
    ecdf_lo = np.arange(0, n, dtype=np.float64) / float(n)
    out["ks_to_fit"] = max(float(np.max(np.abs(ecdf_hi - fit_cdf))), float(np.max(np.abs(fit_cdf - ecdf_lo))))

    hist_counts, edges = np.histogram(arr, bins=max(10, int(bins)))
    total = int(np.sum(hist_counts))
    if total > 0:
        p = hist_counts.astype(np.float64) / float(total)
        q = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            q.append(max(1e-12, float(_cdf(float(hi)) - _cdf(float(lo)))))
        q = np.asarray(q, dtype=np.float64)
        q = q / float(np.sum(q))
        mask = p > 0.0
        out["kl_hist_to_fit"] = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return out


def _laplace_pdf(xs: np.ndarray, loc: float, scale: float) -> np.ndarray:
    xs_arr = np.asarray(xs, dtype=np.float64)
    if not np.isfinite(loc) or not np.isfinite(scale) or scale <= 1e-12:
        return np.full_like(xs_arr, np.nan, dtype=np.float64)
    return 0.5 * np.exp(-np.abs(xs_arr - loc) / scale) / scale


def _cauchy_pdf(xs: np.ndarray, loc: float, scale: float) -> np.ndarray:
    xs_arr = np.asarray(xs, dtype=np.float64)
    if not np.isfinite(loc) or not np.isfinite(scale) or scale <= 1e-12:
        return np.full_like(xs_arr, np.nan, dtype=np.float64)
    return 1.0 / (math.pi * scale * (1.0 + ((xs_arr - loc) / scale) ** 2))


def _summarize_fit_reports_by_group(
    values: Sequence[float],
    labels: Sequence[object],
    top_n: int = 8,
) -> List[dict]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    lab_list = [str(x) for x in labels]
    n = min(int(arr.size), len(lab_list))
    if n <= 0:
        return []
    arr = arr[:n]
    lab_list = lab_list[:n]
    grouped: Dict[str, List[float]] = {}
    for val, label in zip(arr.tolist(), lab_list):
        if not np.isfinite(float(val)):
            continue
        grouped.setdefault(str(label), []).append(float(val))
    items: List[dict] = []
    for label, vals in grouped.items():
        vals_arr = np.asarray(vals, dtype=np.float64)
        if vals_arr.size <= 0:
            continue
        items.append({
            "label": str(label),
            "count": int(vals_arr.size),
            "gaussian_fit": _gaussian_fit_report(vals_arr),
            "laplace_fit": _laplace_fit_report(vals_arr),
            "cauchy_fit": _cauchy_fit_report(vals_arr),
        })
    items.sort(key=lambda x: (-int(x.get("count", 0)), str(x.get("label", ""))))
    return items[: int(top_n)]


def _format_group_fit_reports(items: Sequence[dict], top_n: int = 8) -> str:
    parts: List[str] = []
    for item in list(items or [])[: int(top_n)]:
        label = str(item.get("label", ""))
        g = item.get("gaussian_fit", {}) or {}
        l = item.get("laplace_fit", {}) or {}
        c = item.get("cauchy_fit", {}) or {}
        parts.append(
            "{}: G({:.3f}/{:.3f}) L({:.3f}/{:.3f}) C({:.3f}/{:.3f})".format(
                label,
                float(g.get("ks_to_fit", float("nan"))),
                float(g.get("kl_hist_to_fit", float("nan"))),
                float(l.get("ks_to_fit", float("nan"))),
                float(l.get("kl_hist_to_fit", float("nan"))),
                float(c.get("ks_to_fit", float("nan"))),
                float(c.get("kl_hist_to_fit", float("nan"))),
            )
        )
    return ", ".join(parts)


def _merge_count_maps(dst: Dict[str, object], src: Dict[str, object]) -> None:
    for key, val in (src or {}).items():
        key_s = str(key)
        if isinstance(val, dict):
            cur = dst.setdefault(key_s, {})
            if not isinstance(cur, dict):
                cur = {}
                dst[key_s] = cur
            _merge_count_maps(cur, val)
        else:
            dst[key_s] = int(dst.get(key_s, 0)) + int(val)


def _make_margin_noise_bucket(with_correctness: bool = False) -> Dict[str, object]:
    bucket: Dict[str, object] = {
        "residuals": [],
        "residual_perm_labels": [],
        "residual_ideal_labels": [],
        "residual_correct_slot_labels": [],
        "residual_top1_slot_labels": [],
        "residual_top2_slot_labels": [],
        "residual_is_correct_flags": [],
        "z_scores": [],
        "z_perm_labels": [],
        "z_ideal_labels": [],
        "z_correct_slot_labels": [],
        "z_top1_slot_labels": [],
        "z_top2_slot_labels": [],
        "z_is_correct_flags": [],
        "sample_ref_margins": [],
        "sample_sigmas": [],
        "sample_base_gaps": [],
        "t_residuals": {},
        "t_z_scores": {},
        "standardized_bin_edges": [],
        "standardized_bin_label_counts": {},
        "negative_tail_ideal_counts": {},
        "negative_tail_correct_slot_counts": {},
        "negative_tail_perm_by_ideal": {},
        "negative_tail_perm_by_correct_slot": {},
        "negative_tail_top1_slot_counts": {},
        "negative_tail_perm_by_top1_slot": {},
        "negative_tail_top2_slot_counts": {},
        "negative_tail_perm_by_top2_slot": {},
        "perm_total_counts": {},
        "perm_correct_counts": {},
        "perm_ideal_total_counts": {},
        "perm_ideal_correct_counts": {},
        "negative_tail_perm_total_counts": {},
        "negative_tail_perm_correct_counts": {},
        "negative_tail_perm_ideal_total_counts": {},
        "negative_tail_perm_ideal_correct_counts": {},
        "ideal_total_counts": {},
        "ideal_correct_counts": {},
        "negative_tail_ideal_correct_counts": {},
        "standardized_bin_ideal_total_counts": {},
        "standardized_bin_ideal_correct_counts": {},
        "standardized_bin_total_counts": {},
        "standardized_bin_correct_counts": {},
        "standardized_bin_perm_total_counts": {},
        "standardized_bin_perm_correct_counts": {},
        "standardized_bin_perm_ideal_total_counts": {},
        "standardized_bin_perm_ideal_correct_counts": {},
        "standardized_bin_top1_slot_counts": {},
        "standardized_bin_top1_slot_total_counts": {},
        "standardized_bin_top1_slot_correct_counts": {},
        "standardized_bin_top1_slot_ideal_total_counts": {},
        "standardized_bin_top1_slot_ideal_correct_counts": {},
        "standardized_bin_top2_slot_counts": {},
        "standardized_bin_top2_slot_total_counts": {},
        "standardized_bin_top2_slot_correct_counts": {},
        "standardized_bin_top2_slot_ideal_total_counts": {},
        "standardized_bin_top2_slot_ideal_correct_counts": {},
        "entry_total_count": 0,
        "entry_correct_count": 0,
        "negative_tail_total_count": 0,
        "negative_tail_correct_count": 0,
        "correct_slot_total_counts": {},
        "correct_slot_correct_counts": {},
        "correct_slot_ideal_total_counts": {},
        "correct_slot_ideal_correct_counts": {},
        "negative_tail_correct_slot_total_counts": {},
        "negative_tail_correct_slot_correct_counts": {},
        "top1_slot_total_counts": {},
        "top1_slot_correct_counts": {},
        "top1_slot_ideal_total_counts": {},
        "top1_slot_ideal_correct_counts": {},
        "negative_tail_top1_slot_total_counts": {},
        "negative_tail_top1_slot_correct_counts": {},
        "top2_slot_total_counts": {},
        "top2_slot_correct_counts": {},
        "top2_slot_ideal_total_counts": {},
        "top2_slot_ideal_correct_counts": {},
        "negative_tail_top2_slot_total_counts": {},
        "negative_tail_top2_slot_correct_counts": {},
        "negative_tail_correct_slot_ideal_total_counts": {},
        "negative_tail_correct_slot_ideal_correct_counts": {},
        "negative_tail_top1_slot_ideal_total_counts": {},
        "negative_tail_top1_slot_ideal_correct_counts": {},
        "negative_tail_top2_slot_ideal_total_counts": {},
        "negative_tail_top2_slot_ideal_correct_counts": {},
    }
    if with_correctness:
        bucket["correctness_buckets"] = {
            "correct": _make_margin_noise_bucket(with_correctness=False),
            "incorrect": _make_margin_noise_bucket(with_correctness=False),
        }
    return bucket


def _summarize_peak_bins(
    *,
    bin_edges: Sequence[float],
    bin_label_counts: Dict[str, object],
    top_n_bins: int = 5,
    top_n_labels: int = 8,
) -> List[dict]:
    summaries: List[dict] = []
    indexed = []
    for bin_idx_s, counts in (bin_label_counts or {}).items():
        if not isinstance(counts, dict):
            continue
        total = int(sum(int(v) for v in counts.values()))
        try:
            bin_idx = int(bin_idx_s)
        except Exception:
            continue
        indexed.append((total, bin_idx, counts))
    indexed.sort(key=lambda x: (-x[0], x[1]))
    edges = list(bin_edges)
    for total, bin_idx, counts in indexed[: int(top_n_bins)]:
        if not (0 <= int(bin_idx) < len(edges) - 1):
            continue
        items = sorted(((str(k), int(v)) for k, v in counts.items()), key=lambda x: (-x[1], x[0]))
        summaries.append({
            "bin_index": int(bin_idx),
            "range_left": float(edges[int(bin_idx)]),
            "range_right": float(edges[int(bin_idx) + 1]),
            "count": int(total),
            "top_perm_labels": [
                {
                    "perm_label": str(label),
                    "count": int(cnt),
                    "fraction": float(cnt / total) if total > 0 else float("nan"),
                }
                for label, cnt in items[: int(top_n_labels)]
            ],
        })
    return summaries


def _summarize_flat_counts(counts: Dict[str, object], top_n: int = 8) -> List[dict]:
    total = int(sum(int(v) for v in (counts or {}).values()))
    items = sorted(((str(k), int(v)) for k, v in (counts or {}).items()), key=lambda x: (-x[1], x[0]))
    return [
        {
            "label": str(label),
            "count": int(cnt),
            "fraction": float(cnt / total) if total > 0 else float("nan"),
        }
        for label, cnt in items[: int(top_n)]
    ]


def _summarize_grouped_counts(group_counts: Dict[str, object], top_n_groups: int = 8, top_n_labels: int = 8) -> List[dict]:
    groups = []
    for group, counts in (group_counts or {}).items():
        if not isinstance(counts, dict):
            continue
        total = int(sum(int(v) for v in counts.values()))
        if total <= 0:
            continue
        items = sorted(((str(k), int(v)) for k, v in counts.items()), key=lambda x: (-x[1], x[0]))
        groups.append({
            "group": str(group),
            "count": int(total),
            "top_labels": [
                {
                    "label": str(label),
                    "count": int(cnt),
                    "fraction": float(cnt / total) if total > 0 else float("nan"),
                }
                for label, cnt in items[: int(top_n_labels)]
            ],
        })
    groups.sort(key=lambda x: (-int(x["count"]), str(x["group"])))
    return groups[: int(top_n_groups)]


def _summarize_accuracy_by_group(
    total_counts: Dict[str, object],
    correct_counts: Dict[str, object],
    top_n: int = 8,
) -> List[dict]:
    items = []
    for label, total in (total_counts or {}).items():
        total_i = int(total)
        if total_i <= 0:
            continue
        correct_i = int((correct_counts or {}).get(str(label), 0))
        items.append({
            "label": str(label),
            "count": int(total_i),
            "correct": int(correct_i),
            "accuracy": float(correct_i / total_i),
        })
    items.sort(key=lambda x: (-x["count"], x["label"]))
    return items[: int(top_n)]


def _summarize_accuracy_rstd_by_group(
    total_counts: Dict[str, object],
    correct_counts: Dict[str, object],
    ideal_total_counts: Dict[str, object],
    ideal_correct_counts: Dict[str, object],
    top_n: int = 8,
) -> List[dict]:
    items = []
    labels = set(str(k) for k in (total_counts or {}).keys())
    for label in labels:
        total_i = int((total_counts or {}).get(str(label), 0))
        if total_i <= 0:
            continue
        correct_i = int((correct_counts or {}).get(str(label), 0))
        items.append({
            "label": str(label),
            "count": int(total_i),
            "correct": int(correct_i),
            "accuracy": float(correct_i / total_i),
            "recall_std": _recall_std_from_counts(
                (ideal_total_counts or {}).get(str(label), {}) if isinstance((ideal_total_counts or {}).get(str(label), {}), dict) else {},
                (ideal_correct_counts or {}).get(str(label), {}) if isinstance((ideal_correct_counts or {}).get(str(label), {}), dict) else {},
            ),
        })
    items.sort(key=lambda x: (-x["count"], x["label"]))
    return items[: int(top_n)]


def _summarize_rate_by_group(
    numerator_counts: Dict[str, object],
    denominator_counts: Dict[str, object],
    top_n: int = 8,
) -> List[dict]:
    items = []
    labels = set(str(k) for k in (denominator_counts or {}).keys()) | set(str(k) for k in (numerator_counts or {}).keys())
    for label in labels:
        denom = int((denominator_counts or {}).get(str(label), 0))
        numer = int((numerator_counts or {}).get(str(label), 0))
        if denom <= 0:
            continue
        items.append({
            "label": str(label),
            "numerator": int(numer),
            "denominator": int(denom),
            "rate": float(numer / denom),
        })
    items.sort(key=lambda x: (-x["denominator"], x["label"]))
    return items[: int(top_n)]


def _recall_std_from_counts(
    total_counts: Dict[str, object],
    correct_counts: Dict[str, object],
) -> float:
    recalls = []
    labels = sorted(set(str(k) for k in (total_counts or {}).keys()) | set(str(k) for k in (correct_counts or {}).keys()))
    for label in labels:
        total = int((total_counts or {}).get(str(label), 0))
        if total <= 0:
            continue
        correct = int((correct_counts or {}).get(str(label), 0))
        recalls.append(float(correct / total))
    if not recalls:
        return float("nan")
    return float(np.std(np.asarray(recalls, dtype=np.float64)))


def _summarize_tail_bins(
    *,
    bin_edges: Sequence[float],
    bin_label_counts: Dict[str, object],
    max_right_edge: float = -1.5,
    min_left_edge: Optional[float] = None,
    top_n_bins: int = 5,
    top_n_labels: int = 8,
) -> Dict[str, object]:
    edges = list(bin_edges)
    tail_bins: List[Tuple[int, int, Dict[str, object]]] = []
    aggregate_counts: Dict[str, int] = {}
    total_tail = 0
    for bin_idx_s, counts in (bin_label_counts or {}).items():
        if not isinstance(counts, dict):
            continue
        try:
            bin_idx = int(bin_idx_s)
        except Exception:
            continue
        if not (0 <= bin_idx < len(edges) - 1):
            continue
        right_edge = float(edges[bin_idx + 1])
        left_edge = float(edges[bin_idx])
        if min_left_edge is not None:
            if left_edge < float(min_left_edge):
                continue
        else:
            if right_edge > float(max_right_edge):
                continue
        total = int(sum(int(v) for v in counts.values()))
        if total <= 0:
            continue
        total_tail += total
        tail_bins.append((total, bin_idx, counts))
        for label, cnt in counts.items():
            label_s = str(label)
            aggregate_counts[label_s] = int(aggregate_counts.get(label_s, 0)) + int(cnt)

    tail_bins.sort(key=lambda x: (-x[0], x[1]))
    top_bins = []
    for total, bin_idx, counts in tail_bins[: int(top_n_bins)]:
        items = sorted(((str(k), int(v)) for k, v in counts.items()), key=lambda x: (-x[1], x[0]))
        top_bins.append({
            "bin_index": int(bin_idx),
            "range_left": float(edges[int(bin_idx)]),
            "range_right": float(edges[int(bin_idx) + 1]),
            "count": int(total),
            "top_perm_labels": [
                {
                    "perm_label": str(label),
                    "count": int(cnt),
                    "fraction": float(cnt / total) if total > 0 else float("nan"),
                }
                for label, cnt in items[: int(top_n_labels)]
            ],
        })

    aggregate_items = sorted(
        ((str(k), int(v)) for k, v in aggregate_counts.items()),
        key=lambda x: (-x[1], x[0]),
    )
    return {
        "z_cutoff": float(min_left_edge if min_left_edge is not None else max_right_edge),
        "count": int(total_tail),
        "top_perm_labels": [
            {
                "perm_label": str(label),
                "count": int(cnt),
                "fraction": float(cnt / total_tail) if total_tail > 0 else float("nan"),
            }
            for label, cnt in aggregate_items[: int(top_n_labels)]
        ],
        "top_bins": top_bins,
    }


def _count_labels(labels: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        label_s = str(label)
        counts[label_s] = int(counts.get(label_s, 0)) + 1
    return counts


def _count_labels_by_ideal(labels: Sequence[str], ideal_labels: Sequence[str]) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    total_counts: Dict[str, int] = {}
    ideal_counts: Dict[str, Dict[str, int]] = {}
    for label, ideal in zip(labels, ideal_labels):
        label_s = str(label)
        ideal_s = str(ideal)
        total_counts[label_s] = int(total_counts.get(label_s, 0)) + 1
        ideal_counts.setdefault(label_s, {})
        ideal_counts[label_s][ideal_s] = int(ideal_counts[label_s].get(ideal_s, 0)) + 1
    return total_counts, ideal_counts


def _count_correct_by_label_and_ideal(
    labels: Sequence[str],
    ideal_labels: Sequence[str],
    correct_flags: Sequence[bool],
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    correct_counts: Dict[str, int] = {}
    ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    for label, ideal, is_correct in zip(labels, ideal_labels, correct_flags):
        if not bool(is_correct):
            continue
        label_s = str(label)
        ideal_s = str(ideal)
        correct_counts[label_s] = int(correct_counts.get(label_s, 0)) + 1
        ideal_correct_counts.setdefault(label_s, {})
        ideal_correct_counts[label_s][ideal_s] = int(ideal_correct_counts[label_s].get(ideal_s, 0)) + 1
    return correct_counts, ideal_correct_counts


def _summarize_entry_subset(
    *,
    ideal_labels: Sequence[str],
    perm_labels: Sequence[str],
    correct_slot_labels: Sequence[str],
    top1_slot_labels: Sequence[str],
    top2_slot_labels: Sequence[str],
    correct_flags: Sequence[bool],
    denom_correct_slot_counts: Dict[str, object],
    denom_top1_slot_counts: Dict[str, object],
    denom_top2_slot_counts: Dict[str, object],
) -> Dict[str, object]:
    ideal_counts_map = _count_labels(ideal_labels)
    ideal_correct_counts_map = _count_labels([label for label, flag in zip(ideal_labels, correct_flags) if flag])
    perm_total_counts, perm_ideal_total_counts = _count_labels_by_ideal(perm_labels, ideal_labels)
    perm_correct_counts, perm_ideal_correct_counts = _count_correct_by_label_and_ideal(perm_labels, ideal_labels, correct_flags)
    correct_slot_total_counts, correct_slot_ideal_total_counts = _count_labels_by_ideal(correct_slot_labels, ideal_labels)
    correct_slot_correct_counts, correct_slot_ideal_correct_counts = _count_correct_by_label_and_ideal(
        correct_slot_labels, ideal_labels, correct_flags
    )
    top1_slot_total_counts, top1_slot_ideal_total_counts = _count_labels_by_ideal(top1_slot_labels, ideal_labels)
    top1_slot_correct_counts, top1_slot_ideal_correct_counts = _count_correct_by_label_and_ideal(
        top1_slot_labels, ideal_labels, correct_flags
    )
    top2_slot_total_counts, top2_slot_ideal_total_counts = _count_labels_by_ideal(top2_slot_labels, ideal_labels)
    top2_slot_correct_counts, top2_slot_ideal_correct_counts = _count_correct_by_label_and_ideal(
        top2_slot_labels, ideal_labels, correct_flags
    )
    count = int(len(correct_flags))
    correct = int(sum(1 for x in correct_flags if x))
    return {
        "count": count,
        "correct": correct,
        "accuracy": float(correct / count) if count > 0 else float("nan"),
        "recall_std": _recall_std_from_counts(ideal_counts_map, ideal_correct_counts_map),
        "ideal_counts": _summarize_flat_counts(ideal_counts_map),
        "perm_accuracy": _summarize_accuracy_rstd_by_group(
            perm_total_counts,
            perm_correct_counts,
            perm_ideal_total_counts,
            perm_ideal_correct_counts,
        ),
        "correct_slot_counts": _summarize_flat_counts(correct_slot_total_counts),
        "slot_incidence": _summarize_rate_by_group(correct_slot_total_counts, denom_correct_slot_counts),
        "correct_slot_accuracy": _summarize_accuracy_rstd_by_group(
            correct_slot_total_counts,
            correct_slot_correct_counts,
            correct_slot_ideal_total_counts,
            correct_slot_ideal_correct_counts,
        ),
        "top1_slot_counts": _summarize_flat_counts(top1_slot_total_counts),
        "top1_slot_incidence": _summarize_rate_by_group(top1_slot_total_counts, denom_top1_slot_counts),
        "top1_slot_accuracy": _summarize_accuracy_rstd_by_group(
            top1_slot_total_counts,
            top1_slot_correct_counts,
            top1_slot_ideal_total_counts,
            top1_slot_ideal_correct_counts,
        ),
        "top2_slot_counts": _summarize_flat_counts(top2_slot_total_counts),
        "top2_slot_incidence": _summarize_rate_by_group(top2_slot_total_counts, denom_top2_slot_counts),
        "top2_slot_accuracy": _summarize_accuracy_rstd_by_group(
            top2_slot_total_counts,
            top2_slot_correct_counts,
            top2_slot_ideal_total_counts,
            top2_slot_ideal_correct_counts,
        ),
    }


def _analyze_cyclic_margin_noise(
    results: List[dict],
    perm_list: List[Tuple[int, ...]],
    option_ids: List[str],
    subject: str,
    run_idx: int,
    use_full_reference: bool = False,
    combo_sample_limit: int = 2048,
    combo_seed: int = 0,
    negative_tail_z_cutoff: float = -1.5,
    right_tail_z_cutoff: float = 1.5,
) -> Tuple[Dict[str, object], List[dict], Dict[str, object]]:
    """
    Treat the selected ensemble (cyclic or full) content-space top1/top2 margin as a per-sample reference M_ref.
    For each selected single view, compute a signed margin on the same reference pair:
        M^(pi) = p_pi(top1_ref) - p_pi(top2_ref)
    and define nuisance residual xi^(pi) = M^(pi) - M_ref.
    """
    k = len(option_ids)
    identity_perm = tuple(range(k))
    identity_idx = perm_list.index(identity_perm) if identity_perm in perm_list else 0
    cyc_perms = _rotations(k)
    cyc_idxs = [perm_list.index(p) for p in cyc_perms if p in perm_list]
    full_available = len(perm_list) > len(cyc_idxs)
    selected_perm_idxs = list(range(len(perm_list))) if use_full_reference and full_available else list(cyc_idxs)
    reference_mode = "full" if use_full_reference and full_available else "cyclic"
    n_views = len(selected_perm_idxs)
    selected_perm_labels = [_perm_label(perm_list[idx], k) for idx in selected_perm_idxs]
    selected_perm_tuples = [tuple(int(x) for x in perm_list[idx]) for idx in selected_perm_idxs]

    if n_views <= 0:
        return (
            {
                "reference_mode": reference_mode,
                "n_samples": 0,
                "n_views": 0,
            },
            [],
            _make_margin_noise_bucket(with_correctness=True),
        )

    def _t_values_for_mode() -> List[int]:
        if n_views <= 8:
            return list(range(1, n_views + 1))
        vals = [1, 2, 4, 8, 16, 32, 64, n_views]
        return sorted({int(t) for t in vals if 1 <= int(t) <= n_views})

    base_seed = (int(zlib.crc32(f"{subject}:{run_idx}:{reference_mode}".encode("utf-8"))) + int(combo_seed)) & 0xFFFFFFFF
    combo_rng = np.random.default_rng(base_seed)
    t_values = _t_values_for_mode()
    combo_cache: Dict[int, List[Tuple[int, ...]]] = {}
    for t in t_values:
        total = math.comb(int(n_views), int(t))
        if total <= int(combo_sample_limit):
            combo_cache[int(t)] = list(combinations(range(n_views), int(t)))
            continue
        seen = set()
        combos: List[Tuple[int, ...]] = []
        target = int(combo_sample_limit)
        while len(combos) < target:
            chosen = tuple(sorted(int(x) for x in combo_rng.choice(np.arange(n_views), size=int(t), replace=False).tolist()))
            if chosen in seen:
                continue
            seen.add(chosen)
            combos.append(chosen)
        combo_cache[int(t)] = combos

    sample_records: List[dict] = []
    pooled_residuals: List[float] = []
    residual_perm_labels_all: List[str] = []
    residual_ideal_labels_all: List[str] = []
    residual_correct_slot_labels_all: List[str] = []
    residual_top1_slot_labels_all: List[str] = []
    residual_top2_slot_labels_all: List[str] = []
    residual_is_correct_flags_all: List[bool] = []
    pooled_z_scores: List[float] = []
    z_perm_labels_all: List[str] = []
    z_ideal_labels_all: List[str] = []
    z_correct_slot_labels_all: List[str] = []
    z_top1_slot_labels_all: List[str] = []
    z_top2_slot_labels_all: List[str] = []
    z_is_correct_flags_all: List[bool] = []
    sample_ref_margins: List[float] = []
    sample_sigmas: List[float] = []
    sample_base_gaps: List[float] = []
    t_residuals: Dict[int, List[float]] = {t: [] for t in range(1, n_views + 1)}
    t_z_scores: Dict[int, List[float]] = {t: [] for t in range(1, n_views + 1)}
    std_bin_edges = np.linspace(-4.0, 4.0, 81, dtype=np.float64)
    standardized_bin_label_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_ideal_counts: Dict[str, int] = {}
    negative_tail_correct_slot_counts: Dict[str, int] = {}
    negative_tail_perm_by_ideal: Dict[str, Dict[str, int]] = {}
    negative_tail_perm_by_correct_slot: Dict[str, Dict[str, int]] = {}
    negative_tail_top1_slot_counts: Dict[str, int] = {}
    negative_tail_perm_by_top1_slot: Dict[str, Dict[str, int]] = {}
    negative_tail_top2_slot_counts: Dict[str, int] = {}
    negative_tail_perm_by_top2_slot: Dict[str, Dict[str, int]] = {}
    perm_total_counts: Dict[str, int] = {}
    perm_correct_counts: Dict[str, int] = {}
    perm_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    perm_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_perm_total_counts: Dict[str, int] = {}
    negative_tail_perm_correct_counts: Dict[str, int] = {}
    negative_tail_perm_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_perm_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    ideal_total_counts: Dict[str, int] = {}
    ideal_correct_counts: Dict[str, int] = {}
    negative_tail_ideal_correct_counts: Dict[str, int] = {}
    standardized_bin_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_total_counts: Dict[str, int] = {}
    standardized_bin_correct_counts: Dict[str, int] = {}
    standardized_bin_perm_total_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_perm_correct_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_perm_ideal_total_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    standardized_bin_perm_ideal_correct_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    standardized_bin_top1_slot_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_top1_slot_total_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_top1_slot_correct_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_top1_slot_ideal_total_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    standardized_bin_top1_slot_ideal_correct_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    standardized_bin_top2_slot_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_top2_slot_total_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_top2_slot_correct_counts: Dict[str, Dict[str, int]] = {}
    standardized_bin_top2_slot_ideal_total_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    standardized_bin_top2_slot_ideal_correct_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    entry_total_count = 0
    entry_correct_count = 0
    negative_tail_total_count = 0
    negative_tail_correct_count = 0
    correct_slot_total_counts: Dict[str, int] = {}
    correct_slot_correct_counts: Dict[str, int] = {}
    negative_tail_correct_slot_total_counts: Dict[str, int] = {}
    negative_tail_correct_slot_correct_counts: Dict[str, int] = {}
    correct_slot_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    correct_slot_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    top1_slot_total_counts: Dict[str, int] = {}
    top1_slot_correct_counts: Dict[str, int] = {}
    top1_slot_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    top1_slot_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_top1_slot_total_counts: Dict[str, int] = {}
    negative_tail_top1_slot_correct_counts: Dict[str, int] = {}
    top2_slot_total_counts: Dict[str, int] = {}
    top2_slot_correct_counts: Dict[str, int] = {}
    top2_slot_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    top2_slot_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_top2_slot_total_counts: Dict[str, int] = {}
    negative_tail_top2_slot_correct_counts: Dict[str, int] = {}
    negative_tail_correct_slot_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_correct_slot_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_top1_slot_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_top1_slot_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_top2_slot_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    negative_tail_top2_slot_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    correctness_buckets = {
        "correct": _make_margin_noise_bucket(with_correctness=False),
        "incorrect": _make_margin_noise_bucket(with_correctness=False),
    }

    eps = 1e-12

    def _finite_view_scale(t: int) -> float:
        if n_views <= 1:
            return float("nan")
        t = int(t)
        numer = float(max(0, n_views - t))
        denom = float(max(1, t) * max(1, n_views - 1))
        return math.sqrt(numer / denom)

    for r in results:
        d = r.get("data", {}) or {}
        probs_seq = d.get("probs", None)
        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
            continue

        selected_dists = []
        for perm_idx in selected_perm_idxs:
            dist = _aggregate_probs_over_permutations([probs_seq[perm_idx]], [perm_list[perm_idx]], k)
            selected_dists.append(np.asarray(dist, dtype=np.float64))
        if not selected_dists:
            continue
        selected_dists_arr = np.asarray(selected_dists, dtype=np.float64)
        ref_dist = np.mean(selected_dists_arr, axis=0)
        ref_top1, ref_top2 = _top2_indices(ref_dist)
        ref_margin = float(ref_dist[ref_top1] - ref_dist[ref_top2])
        ref_pred = str(option_ids[ref_top1]) if 0 <= int(ref_top1) < len(option_ids) else str(ref_top1)
        ideal_label = str(d.get("ideal"))
        ideal_idx = int(option_ids.index(ideal_label)) if ideal_label in option_ids else -1
        reference_correct = bool(ref_pred == ideal_label)
        single_view_correct_slot_indices = [
            int(next((j for j, content_idx in enumerate(perm) if int(content_idx) == ideal_idx), -1))
            for perm in selected_perm_tuples
        ] if ideal_idx >= 0 else [-1 for _ in selected_perm_tuples]
        single_view_correct_slot_labels = [
            str(option_ids[idx]) if 0 <= int(idx) < len(option_ids) else "UNK"
            for idx in single_view_correct_slot_indices
        ]
        single_view_preds = [str(option_ids[int(np.argmax(dist))]) for dist in selected_dists_arr]
        single_view_correct_flags = [bool(pred == ideal_label) for pred in single_view_preds]
        single_view_top1_slot_indices = [int(np.argmax(np.asarray(probs_seq[perm_idx], dtype=np.float64))) for perm_idx in selected_perm_idxs]
        single_view_top1_slot_labels = [
            str(option_ids[idx]) if 0 <= int(idx) < len(option_ids) else "UNK"
            for idx in single_view_top1_slot_indices
        ]
        single_view_top2_slot_indices = []
        for perm_idx in selected_perm_idxs:
            letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
            top1_idx, top2_idx = _top2_indices(letter_probs)
            single_view_top2_slot_indices.append(int(top2_idx))
        single_view_top2_slot_labels = [
            str(option_ids[idx]) if 0 <= int(idx) < len(option_ids) else "UNK"
            for idx in single_view_top2_slot_indices
        ]

        single_margins = selected_dists_arr[:, ref_top1] - selected_dists_arr[:, ref_top2]
        residuals = single_margins - ref_margin
        sigma_i = float(np.std(residuals))

        identity_dist = _aggregate_probs_over_permutations([probs_seq[identity_idx]], [perm_list[identity_idx]], k)
        base_gap = _gap_of_distribution(identity_dist)
        base_margin_on_ref_pair = float(identity_dist[ref_top1] - identity_dist[ref_top2])

        pooled_residuals.extend([float(x) for x in residuals.tolist()])
        pooled_payload_perm_labels = [str(x) for x in selected_perm_labels]
        pooled_payload_ideal_labels = [str(ideal_label) for _ in residuals.tolist()]
        pooled_payload_correct_slot_labels = [str(x) for x in single_view_correct_slot_labels]
        pooled_payload_top1_slot_labels = [str(x) for x in single_view_top1_slot_labels]
        pooled_payload_top2_slot_labels = [str(x) for x in single_view_top2_slot_labels]
        pooled_payload_correct_flags = [bool(x) for x in single_view_correct_flags]
        residual_perm_labels_all.extend(pooled_payload_perm_labels)
        residual_ideal_labels_all.extend(pooled_payload_ideal_labels)
        residual_correct_slot_labels_all.extend(pooled_payload_correct_slot_labels)
        residual_top1_slot_labels_all.extend(pooled_payload_top1_slot_labels)
        residual_top2_slot_labels_all.extend(pooled_payload_top2_slot_labels)
        residual_is_correct_flags_all.extend(pooled_payload_correct_flags)
        sample_ref_margins.append(ref_margin)
        sample_sigmas.append(sigma_i)
        sample_base_gaps.append(base_gap)
        corr_bucket = correctness_buckets["correct" if reference_correct else "incorrect"]
        corr_bucket.setdefault("residual_perm_labels", []).extend(pooled_payload_perm_labels)
        corr_bucket.setdefault("residual_ideal_labels", []).extend(pooled_payload_ideal_labels)
        corr_bucket.setdefault("residual_correct_slot_labels", []).extend(pooled_payload_correct_slot_labels)
        corr_bucket.setdefault("residual_top1_slot_labels", []).extend(pooled_payload_top1_slot_labels)
        corr_bucket.setdefault("residual_top2_slot_labels", []).extend(pooled_payload_top2_slot_labels)
        corr_bucket.setdefault("residual_is_correct_flags", []).extend(pooled_payload_correct_flags)

        if sigma_i > eps:
            z_vals = [float(x / sigma_i) for x in residuals.tolist()]
            pooled_z_scores.extend(z_vals)
            for z_val, perm_label in zip(z_vals, selected_perm_labels):
                bin_idx = int(np.digitize(z_val, std_bin_edges) - 1)
                bin_idx = max(0, min(bin_idx, len(std_bin_edges) - 2))
                counts = standardized_bin_label_counts.setdefault(str(bin_idx), {})
                counts[str(perm_label)] = int(counts.get(str(perm_label), 0)) + 1
        else:
            z_vals = []

        corr_bucket["residuals"].extend([float(x) for x in residuals.tolist()])
        corr_bucket["sample_ref_margins"].extend([float(ref_margin)])
        corr_bucket["sample_sigmas"].extend([float(sigma_i)])
        corr_bucket["sample_base_gaps"].extend([float(base_gap)])
        for perm_label, correct_slot_label, top1_slot_label, top2_slot_label, is_correct in zip(
            selected_perm_labels,
            single_view_correct_slot_labels,
            single_view_top1_slot_labels,
            single_view_top2_slot_labels,
            single_view_correct_flags,
        ):
            entry_total_count += 1
            ideal_total_counts[str(ideal_label)] = int(ideal_total_counts.get(str(ideal_label), 0)) + 1
            perm_total_counts[str(perm_label)] = int(perm_total_counts.get(str(perm_label), 0)) + 1
            perm_ideal_total_counts.setdefault(str(perm_label), {})
            perm_ideal_total_counts[str(perm_label)][str(ideal_label)] = int(
                perm_ideal_total_counts[str(perm_label)].get(str(ideal_label), 0)
            ) + 1
            if is_correct:
                entry_correct_count += 1
                ideal_correct_counts[str(ideal_label)] = int(ideal_correct_counts.get(str(ideal_label), 0)) + 1
                perm_correct_counts[str(perm_label)] = int(perm_correct_counts.get(str(perm_label), 0)) + 1
                perm_ideal_correct_counts.setdefault(str(perm_label), {})
                perm_ideal_correct_counts[str(perm_label)][str(ideal_label)] = int(
                    perm_ideal_correct_counts[str(perm_label)].get(str(ideal_label), 0)
                ) + 1
            corr_bucket["entry_total_count"] = int(corr_bucket.get("entry_total_count", 0)) + 1
            corr_bucket.setdefault("ideal_total_counts", {})[str(ideal_label)] = int(
                corr_bucket.setdefault("ideal_total_counts", {}).get(str(ideal_label), 0)
            ) + 1
            corr_bucket.setdefault("perm_total_counts", {})[str(perm_label)] = int(
                corr_bucket.setdefault("perm_total_counts", {}).get(str(perm_label), 0)
            ) + 1
            corr_bucket.setdefault("perm_ideal_total_counts", {}).setdefault(str(perm_label), {})
            corr_bucket["perm_ideal_total_counts"][str(perm_label)][str(ideal_label)] = int(
                corr_bucket["perm_ideal_total_counts"][str(perm_label)].get(str(ideal_label), 0)
            ) + 1
            if is_correct:
                corr_bucket["entry_correct_count"] = int(corr_bucket.get("entry_correct_count", 0)) + 1
                corr_bucket.setdefault("ideal_correct_counts", {})[str(ideal_label)] = int(
                    corr_bucket.setdefault("ideal_correct_counts", {}).get(str(ideal_label), 0)
                ) + 1
                corr_bucket.setdefault("perm_correct_counts", {})[str(perm_label)] = int(
                    corr_bucket.setdefault("perm_correct_counts", {}).get(str(perm_label), 0)
                ) + 1
                corr_bucket.setdefault("perm_ideal_correct_counts", {}).setdefault(str(perm_label), {})
                corr_bucket["perm_ideal_correct_counts"][str(perm_label)][str(ideal_label)] = int(
                    corr_bucket["perm_ideal_correct_counts"][str(perm_label)].get(str(ideal_label), 0)
                ) + 1
            correct_slot_total_counts[str(correct_slot_label)] = int(correct_slot_total_counts.get(str(correct_slot_label), 0)) + 1
            if is_correct:
                correct_slot_correct_counts[str(correct_slot_label)] = int(correct_slot_correct_counts.get(str(correct_slot_label), 0)) + 1
            corr_bucket.setdefault("correct_slot_total_counts", {})[str(correct_slot_label)] = int(
                corr_bucket.setdefault("correct_slot_total_counts", {}).get(str(correct_slot_label), 0)
            ) + 1
            corr_bucket.setdefault("correct_slot_ideal_total_counts", {}).setdefault(str(correct_slot_label), {})
            corr_bucket["correct_slot_ideal_total_counts"][str(correct_slot_label)][str(ideal_label)] = int(
                corr_bucket["correct_slot_ideal_total_counts"][str(correct_slot_label)].get(str(ideal_label), 0)
            ) + 1
            correct_slot_ideal_total_counts.setdefault(str(correct_slot_label), {})
            correct_slot_ideal_total_counts[str(correct_slot_label)][str(ideal_label)] = int(
                correct_slot_ideal_total_counts[str(correct_slot_label)].get(str(ideal_label), 0)
            ) + 1
            if is_correct:
                corr_bucket.setdefault("correct_slot_correct_counts", {})[str(correct_slot_label)] = int(
                    corr_bucket.setdefault("correct_slot_correct_counts", {}).get(str(correct_slot_label), 0)
                ) + 1
                corr_bucket.setdefault("correct_slot_ideal_correct_counts", {}).setdefault(str(correct_slot_label), {})
                corr_bucket["correct_slot_ideal_correct_counts"][str(correct_slot_label)][str(ideal_label)] = int(
                    corr_bucket["correct_slot_ideal_correct_counts"][str(correct_slot_label)].get(str(ideal_label), 0)
                ) + 1
                correct_slot_ideal_correct_counts.setdefault(str(correct_slot_label), {})
                correct_slot_ideal_correct_counts[str(correct_slot_label)][str(ideal_label)] = int(
                    correct_slot_ideal_correct_counts[str(correct_slot_label)].get(str(ideal_label), 0)
                ) + 1
            top1_slot_total_counts[str(top1_slot_label)] = int(top1_slot_total_counts.get(str(top1_slot_label), 0)) + 1
            if is_correct:
                top1_slot_correct_counts[str(top1_slot_label)] = int(top1_slot_correct_counts.get(str(top1_slot_label), 0)) + 1
            corr_bucket.setdefault("top1_slot_total_counts", {})[str(top1_slot_label)] = int(
                corr_bucket.setdefault("top1_slot_total_counts", {}).get(str(top1_slot_label), 0)
            ) + 1
            corr_bucket.setdefault("top1_slot_ideal_total_counts", {}).setdefault(str(top1_slot_label), {})
            corr_bucket["top1_slot_ideal_total_counts"][str(top1_slot_label)][str(ideal_label)] = int(
                corr_bucket["top1_slot_ideal_total_counts"][str(top1_slot_label)].get(str(ideal_label), 0)
            ) + 1
            top1_slot_ideal_total_counts.setdefault(str(top1_slot_label), {})
            top1_slot_ideal_total_counts[str(top1_slot_label)][str(ideal_label)] = int(
                top1_slot_ideal_total_counts[str(top1_slot_label)].get(str(ideal_label), 0)
            ) + 1
            if is_correct:
                corr_bucket.setdefault("top1_slot_correct_counts", {})[str(top1_slot_label)] = int(
                    corr_bucket.setdefault("top1_slot_correct_counts", {}).get(str(top1_slot_label), 0)
                ) + 1
                corr_bucket.setdefault("top1_slot_ideal_correct_counts", {}).setdefault(str(top1_slot_label), {})
                corr_bucket["top1_slot_ideal_correct_counts"][str(top1_slot_label)][str(ideal_label)] = int(
                    corr_bucket["top1_slot_ideal_correct_counts"][str(top1_slot_label)].get(str(ideal_label), 0)
                ) + 1
                top1_slot_ideal_correct_counts.setdefault(str(top1_slot_label), {})
                top1_slot_ideal_correct_counts[str(top1_slot_label)][str(ideal_label)] = int(
                    top1_slot_ideal_correct_counts[str(top1_slot_label)].get(str(ideal_label), 0)
                ) + 1
            top2_slot_total_counts[str(top2_slot_label)] = int(top2_slot_total_counts.get(str(top2_slot_label), 0)) + 1
            if is_correct:
                top2_slot_correct_counts[str(top2_slot_label)] = int(top2_slot_correct_counts.get(str(top2_slot_label), 0)) + 1
            corr_bucket.setdefault("top2_slot_total_counts", {})[str(top2_slot_label)] = int(
                corr_bucket.setdefault("top2_slot_total_counts", {}).get(str(top2_slot_label), 0)
            ) + 1
            corr_bucket.setdefault("top2_slot_ideal_total_counts", {}).setdefault(str(top2_slot_label), {})
            corr_bucket["top2_slot_ideal_total_counts"][str(top2_slot_label)][str(ideal_label)] = int(
                corr_bucket["top2_slot_ideal_total_counts"][str(top2_slot_label)].get(str(ideal_label), 0)
            ) + 1
            top2_slot_ideal_total_counts.setdefault(str(top2_slot_label), {})
            top2_slot_ideal_total_counts[str(top2_slot_label)][str(ideal_label)] = int(
                top2_slot_ideal_total_counts[str(top2_slot_label)].get(str(ideal_label), 0)
            ) + 1
            if is_correct:
                corr_bucket.setdefault("top2_slot_correct_counts", {})[str(top2_slot_label)] = int(
                    corr_bucket.setdefault("top2_slot_correct_counts", {}).get(str(top2_slot_label), 0)
                ) + 1
                corr_bucket.setdefault("top2_slot_ideal_correct_counts", {}).setdefault(str(top2_slot_label), {})
                corr_bucket["top2_slot_ideal_correct_counts"][str(top2_slot_label)][str(ideal_label)] = int(
                    corr_bucket["top2_slot_ideal_correct_counts"][str(top2_slot_label)].get(str(ideal_label), 0)
                ) + 1
                top2_slot_ideal_correct_counts.setdefault(str(top2_slot_label), {})
                top2_slot_ideal_correct_counts[str(top2_slot_label)][str(ideal_label)] = int(
                    top2_slot_ideal_correct_counts[str(top2_slot_label)].get(str(ideal_label), 0)
                ) + 1
        if z_vals:
            corr_bucket["z_scores"].extend(z_vals)
            z_perm_labels_all.extend([str(x) for x in selected_perm_labels])
            z_ideal_labels_all.extend([str(ideal_label) for _ in z_vals])
            z_correct_slot_labels_all.extend([str(x) for x in single_view_correct_slot_labels])
            z_top1_slot_labels_all.extend([str(x) for x in single_view_top1_slot_labels])
            z_top2_slot_labels_all.extend([str(x) for x in single_view_top2_slot_labels])
            z_is_correct_flags_all.extend([bool(x) for x in single_view_correct_flags])
            corr_bucket.setdefault("z_perm_labels", []).extend([str(x) for x in selected_perm_labels])
            corr_bucket.setdefault("z_ideal_labels", []).extend([str(ideal_label) for _ in z_vals])
            corr_bucket.setdefault("z_correct_slot_labels", []).extend([str(x) for x in single_view_correct_slot_labels])
            corr_bucket.setdefault("z_top1_slot_labels", []).extend([str(x) for x in single_view_top1_slot_labels])
            corr_bucket.setdefault("z_top2_slot_labels", []).extend([str(x) for x in single_view_top2_slot_labels])
            corr_bucket.setdefault("z_is_correct_flags", []).extend([bool(x) for x in single_view_correct_flags])
            if not corr_bucket.get("standardized_bin_edges"):
                corr_bucket["standardized_bin_edges"] = [float(x) for x in std_bin_edges.tolist()]
            for z_val, perm_label, correct_slot_label, top1_slot_label, top2_slot_label, is_correct in zip(
                z_vals,
                selected_perm_labels,
                single_view_correct_slot_labels,
                single_view_top1_slot_labels,
                single_view_top2_slot_labels,
                single_view_correct_flags,
            ):
                bin_idx = int(np.digitize(z_val, std_bin_edges) - 1)
                bin_idx = max(0, min(bin_idx, len(std_bin_edges) - 2))
                standardized_bin_total_counts[str(bin_idx)] = int(standardized_bin_total_counts.get(str(bin_idx), 0)) + 1
                standardized_bin_ideal_total_counts.setdefault(str(bin_idx), {})
                standardized_bin_ideal_total_counts[str(bin_idx)][str(ideal_label)] = int(
                    standardized_bin_ideal_total_counts[str(bin_idx)].get(str(ideal_label), 0)
                ) + 1
                if is_correct:
                    standardized_bin_correct_counts[str(bin_idx)] = int(standardized_bin_correct_counts.get(str(bin_idx), 0)) + 1
                    standardized_bin_ideal_correct_counts.setdefault(str(bin_idx), {})
                    standardized_bin_ideal_correct_counts[str(bin_idx)][str(ideal_label)] = int(
                        standardized_bin_ideal_correct_counts[str(bin_idx)].get(str(ideal_label), 0)
                    ) + 1
                counts = corr_bucket.setdefault("standardized_bin_label_counts", {}).setdefault(str(bin_idx), {})
                counts[str(perm_label)] = int(counts.get(str(perm_label), 0)) + 1
                corr_bucket.setdefault("standardized_bin_total_counts", {})[str(bin_idx)] = int(
                    corr_bucket.setdefault("standardized_bin_total_counts", {}).get(str(bin_idx), 0)
                ) + 1
                corr_bucket.setdefault("standardized_bin_ideal_total_counts", {}).setdefault(str(bin_idx), {})
                corr_bucket["standardized_bin_ideal_total_counts"][str(bin_idx)][str(ideal_label)] = int(
                    corr_bucket["standardized_bin_ideal_total_counts"][str(bin_idx)].get(str(ideal_label), 0)
                ) + 1
                standardized_bin_perm_total_counts.setdefault(str(bin_idx), {})
                standardized_bin_perm_total_counts[str(bin_idx)][str(perm_label)] = int(
                    standardized_bin_perm_total_counts[str(bin_idx)].get(str(perm_label), 0)
                ) + 1
                standardized_bin_perm_ideal_total_counts.setdefault(str(bin_idx), {}).setdefault(str(perm_label), {})
                standardized_bin_perm_ideal_total_counts[str(bin_idx)][str(perm_label)][str(ideal_label)] = int(
                    standardized_bin_perm_ideal_total_counts[str(bin_idx)][str(perm_label)].get(str(ideal_label), 0)
                ) + 1
                corr_bucket.setdefault("standardized_bin_perm_total_counts", {}).setdefault(str(bin_idx), {})
                corr_bucket["standardized_bin_perm_total_counts"][str(bin_idx)][str(perm_label)] = int(
                    corr_bucket["standardized_bin_perm_total_counts"][str(bin_idx)].get(str(perm_label), 0)
                ) + 1
                corr_bucket.setdefault("standardized_bin_perm_ideal_total_counts", {}).setdefault(str(bin_idx), {}).setdefault(str(perm_label), {})
                corr_bucket["standardized_bin_perm_ideal_total_counts"][str(bin_idx)][str(perm_label)][str(ideal_label)] = int(
                    corr_bucket["standardized_bin_perm_ideal_total_counts"][str(bin_idx)][str(perm_label)].get(str(ideal_label), 0)
                ) + 1
                if is_correct:
                    corr_bucket.setdefault("standardized_bin_correct_counts", {})[str(bin_idx)] = int(
                        corr_bucket.setdefault("standardized_bin_correct_counts", {}).get(str(bin_idx), 0)
                    ) + 1
                    corr_bucket.setdefault("standardized_bin_ideal_correct_counts", {}).setdefault(str(bin_idx), {})
                    corr_bucket["standardized_bin_ideal_correct_counts"][str(bin_idx)][str(ideal_label)] = int(
                        corr_bucket["standardized_bin_ideal_correct_counts"][str(bin_idx)].get(str(ideal_label), 0)
                    ) + 1
                    standardized_bin_perm_correct_counts.setdefault(str(bin_idx), {})
                    standardized_bin_perm_correct_counts[str(bin_idx)][str(perm_label)] = int(
                        standardized_bin_perm_correct_counts[str(bin_idx)].get(str(perm_label), 0)
                    ) + 1
                    standardized_bin_perm_ideal_correct_counts.setdefault(str(bin_idx), {}).setdefault(str(perm_label), {})
                    standardized_bin_perm_ideal_correct_counts[str(bin_idx)][str(perm_label)][str(ideal_label)] = int(
                        standardized_bin_perm_ideal_correct_counts[str(bin_idx)][str(perm_label)].get(str(ideal_label), 0)
                    ) + 1
                    corr_bucket.setdefault("standardized_bin_perm_correct_counts", {}).setdefault(str(bin_idx), {})
                    corr_bucket["standardized_bin_perm_correct_counts"][str(bin_idx)][str(perm_label)] = int(
                        corr_bucket["standardized_bin_perm_correct_counts"][str(bin_idx)].get(str(perm_label), 0)
                    ) + 1
                    corr_bucket.setdefault("standardized_bin_perm_ideal_correct_counts", {}).setdefault(str(bin_idx), {}).setdefault(str(perm_label), {})
                    corr_bucket["standardized_bin_perm_ideal_correct_counts"][str(bin_idx)][str(perm_label)][str(ideal_label)] = int(
                        corr_bucket["standardized_bin_perm_ideal_correct_counts"][str(bin_idx)][str(perm_label)].get(str(ideal_label), 0)
                    ) + 1
                standardized_bin_top1_slot_counts.setdefault(str(bin_idx), {})
                standardized_bin_top1_slot_counts[str(bin_idx)][str(top1_slot_label)] = int(
                    standardized_bin_top1_slot_counts[str(bin_idx)].get(str(top1_slot_label), 0)
                ) + 1
                standardized_bin_top1_slot_total_counts.setdefault(str(bin_idx), {})
                standardized_bin_top1_slot_total_counts[str(bin_idx)][str(top1_slot_label)] = int(
                    standardized_bin_top1_slot_total_counts[str(bin_idx)].get(str(top1_slot_label), 0)
                ) + 1
                standardized_bin_top1_slot_ideal_total_counts.setdefault(str(bin_idx), {}).setdefault(str(top1_slot_label), {})
                standardized_bin_top1_slot_ideal_total_counts[str(bin_idx)][str(top1_slot_label)][str(ideal_label)] = int(
                    standardized_bin_top1_slot_ideal_total_counts[str(bin_idx)][str(top1_slot_label)].get(str(ideal_label), 0)
                ) + 1
                corr_bucket.setdefault("standardized_bin_top1_slot_counts", {}).setdefault(str(bin_idx), {})
                corr_bucket["standardized_bin_top1_slot_counts"][str(bin_idx)][str(top1_slot_label)] = int(
                    corr_bucket["standardized_bin_top1_slot_counts"][str(bin_idx)].get(str(top1_slot_label), 0)
                ) + 1
                corr_bucket.setdefault("standardized_bin_top1_slot_total_counts", {}).setdefault(str(bin_idx), {})
                corr_bucket["standardized_bin_top1_slot_total_counts"][str(bin_idx)][str(top1_slot_label)] = int(
                    corr_bucket["standardized_bin_top1_slot_total_counts"][str(bin_idx)].get(str(top1_slot_label), 0)
                ) + 1
                corr_bucket.setdefault("standardized_bin_top1_slot_ideal_total_counts", {}).setdefault(str(bin_idx), {}).setdefault(str(top1_slot_label), {})
                corr_bucket["standardized_bin_top1_slot_ideal_total_counts"][str(bin_idx)][str(top1_slot_label)][str(ideal_label)] = int(
                    corr_bucket["standardized_bin_top1_slot_ideal_total_counts"][str(bin_idx)][str(top1_slot_label)].get(str(ideal_label), 0)
                ) + 1
                standardized_bin_top2_slot_counts.setdefault(str(bin_idx), {})
                standardized_bin_top2_slot_counts[str(bin_idx)][str(top2_slot_label)] = int(
                    standardized_bin_top2_slot_counts[str(bin_idx)].get(str(top2_slot_label), 0)
                ) + 1
                standardized_bin_top2_slot_total_counts.setdefault(str(bin_idx), {})
                standardized_bin_top2_slot_total_counts[str(bin_idx)][str(top2_slot_label)] = int(
                    standardized_bin_top2_slot_total_counts[str(bin_idx)].get(str(top2_slot_label), 0)
                ) + 1
                standardized_bin_top2_slot_ideal_total_counts.setdefault(str(bin_idx), {}).setdefault(str(top2_slot_label), {})
                standardized_bin_top2_slot_ideal_total_counts[str(bin_idx)][str(top2_slot_label)][str(ideal_label)] = int(
                    standardized_bin_top2_slot_ideal_total_counts[str(bin_idx)][str(top2_slot_label)].get(str(ideal_label), 0)
                ) + 1
                corr_bucket.setdefault("standardized_bin_top2_slot_counts", {}).setdefault(str(bin_idx), {})
                corr_bucket["standardized_bin_top2_slot_counts"][str(bin_idx)][str(top2_slot_label)] = int(
                    corr_bucket["standardized_bin_top2_slot_counts"][str(bin_idx)].get(str(top2_slot_label), 0)
                ) + 1
                corr_bucket.setdefault("standardized_bin_top2_slot_total_counts", {}).setdefault(str(bin_idx), {})
                corr_bucket["standardized_bin_top2_slot_total_counts"][str(bin_idx)][str(top2_slot_label)] = int(
                    corr_bucket["standardized_bin_top2_slot_total_counts"][str(bin_idx)].get(str(top2_slot_label), 0)
                ) + 1
                corr_bucket.setdefault("standardized_bin_top2_slot_ideal_total_counts", {}).setdefault(str(bin_idx), {}).setdefault(str(top2_slot_label), {})
                corr_bucket["standardized_bin_top2_slot_ideal_total_counts"][str(bin_idx)][str(top2_slot_label)][str(ideal_label)] = int(
                    corr_bucket["standardized_bin_top2_slot_ideal_total_counts"][str(bin_idx)][str(top2_slot_label)].get(str(ideal_label), 0)
                ) + 1
                if is_correct:
                    standardized_bin_top1_slot_correct_counts.setdefault(str(bin_idx), {})
                    standardized_bin_top1_slot_correct_counts[str(bin_idx)][str(top1_slot_label)] = int(
                        standardized_bin_top1_slot_correct_counts[str(bin_idx)].get(str(top1_slot_label), 0)
                    ) + 1
                    standardized_bin_top1_slot_ideal_correct_counts.setdefault(str(bin_idx), {}).setdefault(str(top1_slot_label), {})
                    standardized_bin_top1_slot_ideal_correct_counts[str(bin_idx)][str(top1_slot_label)][str(ideal_label)] = int(
                        standardized_bin_top1_slot_ideal_correct_counts[str(bin_idx)][str(top1_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                    corr_bucket.setdefault("standardized_bin_top1_slot_correct_counts", {}).setdefault(str(bin_idx), {})
                    corr_bucket["standardized_bin_top1_slot_correct_counts"][str(bin_idx)][str(top1_slot_label)] = int(
                        corr_bucket["standardized_bin_top1_slot_correct_counts"][str(bin_idx)].get(str(top1_slot_label), 0)
                    ) + 1
                    corr_bucket.setdefault("standardized_bin_top1_slot_ideal_correct_counts", {}).setdefault(str(bin_idx), {}).setdefault(str(top1_slot_label), {})
                    corr_bucket["standardized_bin_top1_slot_ideal_correct_counts"][str(bin_idx)][str(top1_slot_label)][str(ideal_label)] = int(
                        corr_bucket["standardized_bin_top1_slot_ideal_correct_counts"][str(bin_idx)][str(top1_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                    standardized_bin_top2_slot_correct_counts.setdefault(str(bin_idx), {})
                    standardized_bin_top2_slot_correct_counts[str(bin_idx)][str(top2_slot_label)] = int(
                        standardized_bin_top2_slot_correct_counts[str(bin_idx)].get(str(top2_slot_label), 0)
                    ) + 1
                    standardized_bin_top2_slot_ideal_correct_counts.setdefault(str(bin_idx), {}).setdefault(str(top2_slot_label), {})
                    standardized_bin_top2_slot_ideal_correct_counts[str(bin_idx)][str(top2_slot_label)][str(ideal_label)] = int(
                        standardized_bin_top2_slot_ideal_correct_counts[str(bin_idx)][str(top2_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                    corr_bucket.setdefault("standardized_bin_top2_slot_correct_counts", {}).setdefault(str(bin_idx), {})
                    corr_bucket["standardized_bin_top2_slot_correct_counts"][str(bin_idx)][str(top2_slot_label)] = int(
                        corr_bucket["standardized_bin_top2_slot_correct_counts"][str(bin_idx)].get(str(top2_slot_label), 0)
                    ) + 1
                    corr_bucket.setdefault("standardized_bin_top2_slot_ideal_correct_counts", {}).setdefault(str(bin_idx), {}).setdefault(str(top2_slot_label), {})
                    corr_bucket["standardized_bin_top2_slot_ideal_correct_counts"][str(bin_idx)][str(top2_slot_label)][str(ideal_label)] = int(
                        corr_bucket["standardized_bin_top2_slot_ideal_correct_counts"][str(bin_idx)][str(top2_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                if float(z_val) <= float(negative_tail_z_cutoff):
                    negative_tail_total_count += 1
                    if is_correct:
                        negative_tail_correct_count += 1
                    negative_tail_ideal_counts[str(ideal_label)] = int(negative_tail_ideal_counts.get(str(ideal_label), 0)) + 1
                    if is_correct:
                        negative_tail_ideal_correct_counts[str(ideal_label)] = int(
                            negative_tail_ideal_correct_counts.get(str(ideal_label), 0)
                        ) + 1
                    negative_tail_perm_total_counts[str(perm_label)] = int(
                        negative_tail_perm_total_counts.get(str(perm_label), 0)
                    ) + 1
                    negative_tail_perm_ideal_total_counts.setdefault(str(perm_label), {})
                    negative_tail_perm_ideal_total_counts[str(perm_label)][str(ideal_label)] = int(
                        negative_tail_perm_ideal_total_counts[str(perm_label)].get(str(ideal_label), 0)
                    ) + 1
                    corr_bucket.setdefault("negative_tail_perm_total_counts", {})[str(perm_label)] = int(
                        corr_bucket.setdefault("negative_tail_perm_total_counts", {}).get(str(perm_label), 0)
                    ) + 1
                    corr_bucket.setdefault("negative_tail_perm_ideal_total_counts", {}).setdefault(str(perm_label), {})
                    corr_bucket["negative_tail_perm_ideal_total_counts"][str(perm_label)][str(ideal_label)] = int(
                        corr_bucket["negative_tail_perm_ideal_total_counts"][str(perm_label)].get(str(ideal_label), 0)
                    ) + 1
                    if is_correct:
                        negative_tail_perm_correct_counts[str(perm_label)] = int(
                            negative_tail_perm_correct_counts.get(str(perm_label), 0)
                        ) + 1
                        negative_tail_perm_ideal_correct_counts.setdefault(str(perm_label), {})
                        negative_tail_perm_ideal_correct_counts[str(perm_label)][str(ideal_label)] = int(
                            negative_tail_perm_ideal_correct_counts[str(perm_label)].get(str(ideal_label), 0)
                        ) + 1
                        corr_bucket.setdefault("negative_tail_perm_correct_counts", {})[str(perm_label)] = int(
                            corr_bucket.setdefault("negative_tail_perm_correct_counts", {}).get(str(perm_label), 0)
                        ) + 1
                        corr_bucket.setdefault("negative_tail_perm_ideal_correct_counts", {}).setdefault(str(perm_label), {})
                        corr_bucket["negative_tail_perm_ideal_correct_counts"][str(perm_label)][str(ideal_label)] = int(
                            corr_bucket["negative_tail_perm_ideal_correct_counts"][str(perm_label)].get(str(ideal_label), 0)
                        ) + 1
                    negative_tail_correct_slot_counts[str(correct_slot_label)] = int(
                        negative_tail_correct_slot_counts.get(str(correct_slot_label), 0)
                    ) + 1
                    negative_tail_top1_slot_counts[str(top1_slot_label)] = int(
                        negative_tail_top1_slot_counts.get(str(top1_slot_label), 0)
                    ) + 1
                    negative_tail_top2_slot_counts[str(top2_slot_label)] = int(
                        negative_tail_top2_slot_counts.get(str(top2_slot_label), 0)
                    ) + 1
                    ideal_perm_counts_all = negative_tail_perm_by_ideal.setdefault(str(ideal_label), {})
                    ideal_perm_counts_all[str(perm_label)] = int(ideal_perm_counts_all.get(str(perm_label), 0)) + 1
                    slot_perm_counts_all = negative_tail_perm_by_correct_slot.setdefault(str(correct_slot_label), {})
                    slot_perm_counts_all[str(perm_label)] = int(slot_perm_counts_all.get(str(perm_label), 0)) + 1
                    top1_slot_perm_counts_all = negative_tail_perm_by_top1_slot.setdefault(str(top1_slot_label), {})
                    top1_slot_perm_counts_all[str(perm_label)] = int(top1_slot_perm_counts_all.get(str(perm_label), 0)) + 1
                    top2_slot_perm_counts_all = negative_tail_perm_by_top2_slot.setdefault(str(top2_slot_label), {})
                    top2_slot_perm_counts_all[str(perm_label)] = int(top2_slot_perm_counts_all.get(str(perm_label), 0)) + 1
                    negative_tail_correct_slot_total_counts[str(correct_slot_label)] = int(
                        negative_tail_correct_slot_total_counts.get(str(correct_slot_label), 0)
                    ) + 1
                    negative_tail_correct_slot_ideal_total_counts.setdefault(str(correct_slot_label), {})
                    negative_tail_correct_slot_ideal_total_counts[str(correct_slot_label)][str(ideal_label)] = int(
                        negative_tail_correct_slot_ideal_total_counts[str(correct_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                    if is_correct:
                        negative_tail_correct_slot_correct_counts[str(correct_slot_label)] = int(
                            negative_tail_correct_slot_correct_counts.get(str(correct_slot_label), 0)
                        ) + 1
                        negative_tail_correct_slot_ideal_correct_counts.setdefault(str(correct_slot_label), {})
                        negative_tail_correct_slot_ideal_correct_counts[str(correct_slot_label)][str(ideal_label)] = int(
                            negative_tail_correct_slot_ideal_correct_counts[str(correct_slot_label)].get(str(ideal_label), 0)
                        ) + 1
                    negative_tail_top1_slot_total_counts[str(top1_slot_label)] = int(
                        negative_tail_top1_slot_total_counts.get(str(top1_slot_label), 0)
                    ) + 1
                    negative_tail_top1_slot_ideal_total_counts.setdefault(str(top1_slot_label), {})
                    negative_tail_top1_slot_ideal_total_counts[str(top1_slot_label)][str(ideal_label)] = int(
                        negative_tail_top1_slot_ideal_total_counts[str(top1_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                    if is_correct:
                        negative_tail_top1_slot_correct_counts[str(top1_slot_label)] = int(
                            negative_tail_top1_slot_correct_counts.get(str(top1_slot_label), 0)
                        ) + 1
                        negative_tail_top1_slot_ideal_correct_counts.setdefault(str(top1_slot_label), {})
                        negative_tail_top1_slot_ideal_correct_counts[str(top1_slot_label)][str(ideal_label)] = int(
                            negative_tail_top1_slot_ideal_correct_counts[str(top1_slot_label)].get(str(ideal_label), 0)
                        ) + 1
                    negative_tail_top2_slot_total_counts[str(top2_slot_label)] = int(
                        negative_tail_top2_slot_total_counts.get(str(top2_slot_label), 0)
                    ) + 1
                    negative_tail_top2_slot_ideal_total_counts.setdefault(str(top2_slot_label), {})
                    negative_tail_top2_slot_ideal_total_counts[str(top2_slot_label)][str(ideal_label)] = int(
                        negative_tail_top2_slot_ideal_total_counts[str(top2_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                    if is_correct:
                        negative_tail_top2_slot_correct_counts[str(top2_slot_label)] = int(
                            negative_tail_top2_slot_correct_counts.get(str(top2_slot_label), 0)
                        ) + 1
                        negative_tail_top2_slot_ideal_correct_counts.setdefault(str(top2_slot_label), {})
                        negative_tail_top2_slot_ideal_correct_counts[str(top2_slot_label)][str(ideal_label)] = int(
                            negative_tail_top2_slot_ideal_correct_counts[str(top2_slot_label)].get(str(ideal_label), 0)
                        ) + 1

                    corr_bucket.setdefault("negative_tail_ideal_counts", {})[str(ideal_label)] = int(
                        corr_bucket.setdefault("negative_tail_ideal_counts", {}).get(str(ideal_label), 0)
                    ) + 1
                    if is_correct:
                        corr_bucket.setdefault("negative_tail_ideal_correct_counts", {})[str(ideal_label)] = int(
                            corr_bucket.setdefault("negative_tail_ideal_correct_counts", {}).get(str(ideal_label), 0)
                        ) + 1
                    corr_bucket.setdefault("negative_tail_correct_slot_counts", {})[str(correct_slot_label)] = int(
                        corr_bucket.setdefault("negative_tail_correct_slot_counts", {}).get(str(correct_slot_label), 0)
                    ) + 1
                    corr_bucket.setdefault("negative_tail_top1_slot_counts", {})[str(top1_slot_label)] = int(
                        corr_bucket.setdefault("negative_tail_top1_slot_counts", {}).get(str(top1_slot_label), 0)
                    ) + 1
                    corr_bucket.setdefault("negative_tail_top2_slot_counts", {})[str(top2_slot_label)] = int(
                        corr_bucket.setdefault("negative_tail_top2_slot_counts", {}).get(str(top2_slot_label), 0)
                    ) + 1
                    corr_bucket["negative_tail_total_count"] = int(corr_bucket.get("negative_tail_total_count", 0)) + 1
                    if is_correct:
                        corr_bucket["negative_tail_correct_count"] = int(corr_bucket.get("negative_tail_correct_count", 0)) + 1
                    ideal_perm_counts = corr_bucket.setdefault("negative_tail_perm_by_ideal", {}).setdefault(str(ideal_label), {})
                    ideal_perm_counts[str(perm_label)] = int(ideal_perm_counts.get(str(perm_label), 0)) + 1
                    slot_perm_counts = corr_bucket.setdefault("negative_tail_perm_by_correct_slot", {}).setdefault(str(correct_slot_label), {})
                    slot_perm_counts[str(perm_label)] = int(slot_perm_counts.get(str(perm_label), 0)) + 1
                    top1_slot_perm_counts = corr_bucket.setdefault("negative_tail_perm_by_top1_slot", {}).setdefault(str(top1_slot_label), {})
                    top1_slot_perm_counts[str(perm_label)] = int(top1_slot_perm_counts.get(str(perm_label), 0)) + 1
                    top2_slot_perm_counts = corr_bucket.setdefault("negative_tail_perm_by_top2_slot", {}).setdefault(str(top2_slot_label), {})
                    top2_slot_perm_counts[str(perm_label)] = int(top2_slot_perm_counts.get(str(perm_label), 0)) + 1
                    corr_bucket.setdefault("negative_tail_correct_slot_total_counts", {})[str(correct_slot_label)] = int(
                        corr_bucket.setdefault("negative_tail_correct_slot_total_counts", {}).get(str(correct_slot_label), 0)
                    ) + 1
                    corr_bucket.setdefault("negative_tail_correct_slot_ideal_total_counts", {}).setdefault(str(correct_slot_label), {})
                    corr_bucket["negative_tail_correct_slot_ideal_total_counts"][str(correct_slot_label)][str(ideal_label)] = int(
                        corr_bucket["negative_tail_correct_slot_ideal_total_counts"][str(correct_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                    if is_correct:
                        corr_bucket.setdefault("negative_tail_correct_slot_correct_counts", {})[str(correct_slot_label)] = int(
                            corr_bucket.setdefault("negative_tail_correct_slot_correct_counts", {}).get(str(correct_slot_label), 0)
                        ) + 1
                        corr_bucket.setdefault("negative_tail_correct_slot_ideal_correct_counts", {}).setdefault(str(correct_slot_label), {})
                        corr_bucket["negative_tail_correct_slot_ideal_correct_counts"][str(correct_slot_label)][str(ideal_label)] = int(
                            corr_bucket["negative_tail_correct_slot_ideal_correct_counts"][str(correct_slot_label)].get(str(ideal_label), 0)
                        ) + 1
                    corr_bucket.setdefault("negative_tail_top1_slot_total_counts", {})[str(top1_slot_label)] = int(
                        corr_bucket.setdefault("negative_tail_top1_slot_total_counts", {}).get(str(top1_slot_label), 0)
                    ) + 1
                    corr_bucket.setdefault("negative_tail_top1_slot_ideal_total_counts", {}).setdefault(str(top1_slot_label), {})
                    corr_bucket["negative_tail_top1_slot_ideal_total_counts"][str(top1_slot_label)][str(ideal_label)] = int(
                        corr_bucket["negative_tail_top1_slot_ideal_total_counts"][str(top1_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                    if is_correct:
                        corr_bucket.setdefault("negative_tail_top1_slot_correct_counts", {})[str(top1_slot_label)] = int(
                            corr_bucket.setdefault("negative_tail_top1_slot_correct_counts", {}).get(str(top1_slot_label), 0)
                        ) + 1
                        corr_bucket.setdefault("negative_tail_top1_slot_ideal_correct_counts", {}).setdefault(str(top1_slot_label), {})
                        corr_bucket["negative_tail_top1_slot_ideal_correct_counts"][str(top1_slot_label)][str(ideal_label)] = int(
                            corr_bucket["negative_tail_top1_slot_ideal_correct_counts"][str(top1_slot_label)].get(str(ideal_label), 0)
                        ) + 1
                    corr_bucket.setdefault("negative_tail_top2_slot_total_counts", {})[str(top2_slot_label)] = int(
                        corr_bucket.setdefault("negative_tail_top2_slot_total_counts", {}).get(str(top2_slot_label), 0)
                    ) + 1
                    corr_bucket.setdefault("negative_tail_top2_slot_ideal_total_counts", {}).setdefault(str(top2_slot_label), {})
                    corr_bucket["negative_tail_top2_slot_ideal_total_counts"][str(top2_slot_label)][str(ideal_label)] = int(
                        corr_bucket["negative_tail_top2_slot_ideal_total_counts"][str(top2_slot_label)].get(str(ideal_label), 0)
                    ) + 1
                    if is_correct:
                        corr_bucket.setdefault("negative_tail_top2_slot_correct_counts", {})[str(top2_slot_label)] = int(
                            corr_bucket.setdefault("negative_tail_top2_slot_correct_counts", {}).get(str(top2_slot_label), 0)
                        ) + 1
                        corr_bucket.setdefault("negative_tail_top2_slot_ideal_correct_counts", {}).setdefault(str(top2_slot_label), {})
                        corr_bucket["negative_tail_top2_slot_ideal_correct_counts"][str(top2_slot_label)][str(ideal_label)] = int(
                            corr_bucket["negative_tail_top2_slot_ideal_correct_counts"][str(top2_slot_label)].get(str(ideal_label), 0)
                        ) + 1

        t_summary: List[dict] = []
        for t, combos in combo_cache.items():
            avg_margins = np.asarray(
                [float(np.mean(single_margins[list(combo)])) for combo in combos],
                dtype=np.float64,
            )
            resid_t = avg_margins - ref_margin
            t_residuals[t].extend([float(x) for x in resid_t.tolist()])
            if sigma_i > eps:
                denom = sigma_i / math.sqrt(float(t))
                if denom > eps:
                    t_z_scores[t].extend([float(x / denom) for x in resid_t.tolist()])
            t_summary.append({
                "t": int(t),
                "n_combinations": int(len(combos)),
                "resid_std": float(np.std(resid_t)) if resid_t.size > 0 else float("nan"),
                "resid_abs_mean": float(np.mean(np.abs(resid_t))) if resid_t.size > 0 else float("nan"),
            })

        sample_records.append({
            "subject": str(subject),
            "run": int(run_idx),
            "idx": int(d.get("idx", -1)),
            "ideal": ideal_label,
            "ideal_index": int(ideal_idx),
            "reference_mode": reference_mode,
            "n_views": int(n_views),
            "ref_pred": ref_pred,
            "reference_correct": bool(reference_correct),
            "ref_top1": int(ref_top1),
            "ref_top2": int(ref_top2),
            "ref_margin": ref_margin,
            "base_gap": float(base_gap),
            "base_margin_on_ref_pair": base_margin_on_ref_pair,
            "sigma_i": sigma_i,
            "single_view_perm_indices": [int(x) for x in selected_perm_idxs],
            "single_view_perm_tuples": [list(x) for x in selected_perm_tuples],
            "single_view_perm_labels": [str(x) for x in selected_perm_labels],
            "single_view_correct_slot_indices": [int(x) for x in single_view_correct_slot_indices],
            "single_view_correct_slot_labels": [str(x) for x in single_view_correct_slot_labels],
            "single_view_top1_slot_indices": [int(x) for x in single_view_top1_slot_indices],
            "single_view_top1_slot_labels": [str(x) for x in single_view_top1_slot_labels],
            "single_view_top2_slot_indices": [int(x) for x in single_view_top2_slot_indices],
            "single_view_top2_slot_labels": [str(x) for x in single_view_top2_slot_labels],
            "single_view_preds": [str(x) for x in single_view_preds],
            "single_view_correct_flags": [bool(x) for x in single_view_correct_flags],
            "single_view_margins": [float(x) for x in single_margins.tolist()],
            "single_view_residuals": [float(x) for x in residuals.tolist()],
            "single_view_z_scores": [float(x / sigma_i) if sigma_i > eps else float("nan") for x in residuals.tolist()],
            "t_summary": t_summary,
        })

    pooled_resid_arr = np.asarray(pooled_residuals, dtype=np.float64)
    pooled_z_arr = np.asarray(pooled_z_scores, dtype=np.float64)
    sample_ref_arr = np.asarray(sample_ref_margins, dtype=np.float64)
    sample_sigma_arr = np.asarray(sample_sigmas, dtype=np.float64)
    sample_base_gap_arr = np.asarray(sample_base_gaps, dtype=np.float64)

    t_view_summary = []
    t1_std = float(np.std(np.asarray(t_residuals[1], dtype=np.float64))) if 1 in t_residuals and t_residuals[1] else float("nan")
    for t in sorted(t_residuals.keys()):
        resid_arr = np.asarray(t_residuals[t], dtype=np.float64)
        z_arr = np.asarray(t_z_scores[t], dtype=np.float64)
        resid_std = float(np.std(resid_arr)) if resid_arr.size > 0 else float("nan")
        sqrt_target = float(t1_std / math.sqrt(float(t))) if np.isfinite(t1_std) else float("nan")
        t_view_summary.append({
            "t": int(t),
            "n_residuals": int(resid_arr.size),
            "resid_std": resid_std,
            "resid_abs_mean": float(np.mean(np.abs(resid_arr))) if resid_arr.size > 0 else float("nan"),
            "sqrt_target_from_t1": sqrt_target,
            "finite_view_target_from_t1": float(t1_std * _finite_view_scale(int(t))) if np.isfinite(t1_std) else float("nan"),
            "ratio_to_t1_sqrt": float(resid_std / sqrt_target) if np.isfinite(resid_std) and np.isfinite(sqrt_target) and sqrt_target > eps else float("nan"),
            "ratio_to_finite_view_target": (
                float(resid_std / (t1_std * _finite_view_scale(int(t))))
                if np.isfinite(resid_std) and np.isfinite(t1_std) and np.isfinite(_finite_view_scale(int(t))) and (t1_std * _finite_view_scale(int(t))) > eps
                else float("nan")
            ),
            "standardized_residual_fit": _gaussian_fit_report(z_arr),
        })

    summary: Dict[str, object] = {
        "reference_mode": reference_mode,
        "n_samples": int(len(sample_records)),
        "n_views": int(n_views),
        "mean_ref_margin": float(np.mean(sample_ref_arr)) if sample_ref_arr.size > 0 else float("nan"),
        "std_ref_margin": float(np.std(sample_ref_arr)) if sample_ref_arr.size > 0 else float("nan"),
        "mean_sigma_i": float(np.mean(sample_sigma_arr)) if sample_sigma_arr.size > 0 else float("nan"),
        "std_sigma_i": float(np.std(sample_sigma_arr)) if sample_sigma_arr.size > 0 else float("nan"),
        "mean_base_gap": float(np.mean(sample_base_gap_arr)) if sample_base_gap_arr.size > 0 else float("nan"),
        "corr_ref_margin_sigma": _safe_corr(sample_ref_arr, sample_sigma_arr),
        "corr_base_gap_sigma": _safe_corr(sample_base_gap_arr, sample_sigma_arr),
        "pooled_residual_fit": _gaussian_fit_report(pooled_resid_arr),
        "pooled_residual_laplace_fit": _laplace_fit_report(pooled_resid_arr),
        "pooled_residual_cauchy_fit": _cauchy_fit_report(pooled_resid_arr),
        "standardized_residual_fit": _gaussian_fit_report(pooled_z_arr),
        "peak_bin_summary": _summarize_peak_bins(
            bin_edges=std_bin_edges.tolist(),
            bin_label_counts=standardized_bin_label_counts,
        ),
        "negative_tail_summary": _summarize_tail_bins(
            bin_edges=std_bin_edges.tolist(),
            bin_label_counts=standardized_bin_label_counts,
            max_right_edge=float(negative_tail_z_cutoff),
        ),
        "negative_tail_ideal_counts": _summarize_flat_counts(negative_tail_ideal_counts),
        "negative_tail_correct_slot_counts": _summarize_flat_counts(negative_tail_correct_slot_counts),
        "negative_tail_by_ideal": _summarize_grouped_counts(negative_tail_perm_by_ideal),
        "negative_tail_by_correct_slot": _summarize_grouped_counts(negative_tail_perm_by_correct_slot),
        "negative_tail_top1_slot_counts": _summarize_flat_counts(negative_tail_top1_slot_counts),
        "negative_tail_by_top1_slot": _summarize_grouped_counts(negative_tail_perm_by_top1_slot),
        "negative_tail_top2_slot_counts": _summarize_flat_counts(negative_tail_top2_slot_counts),
        "negative_tail_by_top2_slot": _summarize_grouped_counts(negative_tail_perm_by_top2_slot),
        "negative_tail_slot_incidence": _summarize_rate_by_group(
            negative_tail_correct_slot_total_counts,
            correct_slot_total_counts,
        ),
        "negative_tail_top1_slot_incidence": _summarize_rate_by_group(
            negative_tail_top1_slot_total_counts,
            top1_slot_total_counts,
        ),
        "negative_tail_top2_slot_incidence": _summarize_rate_by_group(
            negative_tail_top2_slot_total_counts,
            top2_slot_total_counts,
        ),
        "correct_slot_accuracy": _summarize_accuracy_by_group(
            correct_slot_total_counts,
            correct_slot_correct_counts,
        ),
        "negative_tail_correct_slot_accuracy": _summarize_accuracy_by_group(
            negative_tail_correct_slot_total_counts,
            negative_tail_correct_slot_correct_counts,
        ),
        "top1_slot_accuracy": _summarize_accuracy_by_group(
            top1_slot_total_counts,
            top1_slot_correct_counts,
        ),
        "negative_tail_top1_slot_accuracy": _summarize_accuracy_by_group(
            negative_tail_top1_slot_total_counts,
            negative_tail_top1_slot_correct_counts,
        ),
        "top2_slot_accuracy": _summarize_accuracy_by_group(
            top2_slot_total_counts,
            top2_slot_correct_counts,
        ),
        "negative_tail_top2_slot_accuracy": _summarize_accuracy_by_group(
            negative_tail_top2_slot_total_counts,
            negative_tail_top2_slot_correct_counts,
        ),
        "correctness_split": {
            name: _summarize_margin_noise_bucket(
                bucket,
                n_views=n_views,
                reference_mode=reference_mode,
                negative_tail_z_cutoff=float(negative_tail_z_cutoff),
            )
            for name, bucket in correctness_buckets.items()
        },
        "t_view_summary": t_view_summary,
    }

    pooled_payload = {
        "residuals": [float(x) for x in pooled_residuals],
        "residual_perm_labels": residual_perm_labels_all,
        "residual_ideal_labels": residual_ideal_labels_all,
        "residual_correct_slot_labels": residual_correct_slot_labels_all,
        "residual_top1_slot_labels": residual_top1_slot_labels_all,
        "residual_top2_slot_labels": residual_top2_slot_labels_all,
        "residual_is_correct_flags": [bool(x) for x in residual_is_correct_flags_all],
        "z_scores": [float(x) for x in pooled_z_scores],
        "z_perm_labels": z_perm_labels_all,
        "z_ideal_labels": z_ideal_labels_all,
        "z_correct_slot_labels": z_correct_slot_labels_all,
        "z_top1_slot_labels": z_top1_slot_labels_all,
        "z_top2_slot_labels": z_top2_slot_labels_all,
        "z_is_correct_flags": [bool(x) for x in z_is_correct_flags_all],
        "sample_ref_margins": [float(x) for x in sample_ref_margins],
        "sample_sigmas": [float(x) for x in sample_sigmas],
        "sample_base_gaps": [float(x) for x in sample_base_gaps],
        "t_residuals": {int(t): [float(x) for x in vals] for t, vals in t_residuals.items()},
        "t_z_scores": {int(t): [float(x) for x in vals] for t, vals in t_z_scores.items()},
        "standardized_bin_edges": [float(x) for x in std_bin_edges.tolist()],
        "standardized_bin_label_counts": standardized_bin_label_counts,
        "negative_tail_ideal_counts": negative_tail_ideal_counts,
        "negative_tail_correct_slot_counts": negative_tail_correct_slot_counts,
        "negative_tail_perm_by_ideal": negative_tail_perm_by_ideal,
        "negative_tail_perm_by_correct_slot": negative_tail_perm_by_correct_slot,
        "negative_tail_top1_slot_counts": negative_tail_top1_slot_counts,
        "negative_tail_perm_by_top1_slot": negative_tail_perm_by_top1_slot,
        "negative_tail_top2_slot_counts": negative_tail_top2_slot_counts,
        "negative_tail_perm_by_top2_slot": negative_tail_perm_by_top2_slot,
        "perm_total_counts": perm_total_counts,
        "perm_correct_counts": perm_correct_counts,
        "perm_ideal_total_counts": perm_ideal_total_counts,
        "perm_ideal_correct_counts": perm_ideal_correct_counts,
        "negative_tail_perm_total_counts": negative_tail_perm_total_counts,
        "negative_tail_perm_correct_counts": negative_tail_perm_correct_counts,
        "negative_tail_perm_ideal_total_counts": negative_tail_perm_ideal_total_counts,
        "negative_tail_perm_ideal_correct_counts": negative_tail_perm_ideal_correct_counts,
        "ideal_total_counts": ideal_total_counts,
        "ideal_correct_counts": ideal_correct_counts,
        "negative_tail_ideal_correct_counts": negative_tail_ideal_correct_counts,
        "standardized_bin_ideal_total_counts": standardized_bin_ideal_total_counts,
        "standardized_bin_ideal_correct_counts": standardized_bin_ideal_correct_counts,
        "standardized_bin_total_counts": standardized_bin_total_counts,
        "standardized_bin_correct_counts": standardized_bin_correct_counts,
        "standardized_bin_perm_total_counts": standardized_bin_perm_total_counts,
        "standardized_bin_perm_correct_counts": standardized_bin_perm_correct_counts,
        "standardized_bin_perm_ideal_total_counts": standardized_bin_perm_ideal_total_counts,
        "standardized_bin_perm_ideal_correct_counts": standardized_bin_perm_ideal_correct_counts,
        "standardized_bin_top1_slot_counts": standardized_bin_top1_slot_counts,
        "standardized_bin_top1_slot_total_counts": standardized_bin_top1_slot_total_counts,
        "standardized_bin_top1_slot_correct_counts": standardized_bin_top1_slot_correct_counts,
        "standardized_bin_top1_slot_ideal_total_counts": standardized_bin_top1_slot_ideal_total_counts,
        "standardized_bin_top1_slot_ideal_correct_counts": standardized_bin_top1_slot_ideal_correct_counts,
        "standardized_bin_top2_slot_counts": standardized_bin_top2_slot_counts,
        "standardized_bin_top2_slot_total_counts": standardized_bin_top2_slot_total_counts,
        "standardized_bin_top2_slot_correct_counts": standardized_bin_top2_slot_correct_counts,
        "standardized_bin_top2_slot_ideal_total_counts": standardized_bin_top2_slot_ideal_total_counts,
        "standardized_bin_top2_slot_ideal_correct_counts": standardized_bin_top2_slot_ideal_correct_counts,
        "entry_total_count": int(entry_total_count),
        "entry_correct_count": int(entry_correct_count),
        "negative_tail_total_count": int(negative_tail_total_count),
        "negative_tail_correct_count": int(negative_tail_correct_count),
        "correct_slot_total_counts": correct_slot_total_counts,
        "correct_slot_correct_counts": correct_slot_correct_counts,
        "correct_slot_ideal_total_counts": correct_slot_ideal_total_counts,
        "correct_slot_ideal_correct_counts": correct_slot_ideal_correct_counts,
        "negative_tail_correct_slot_total_counts": negative_tail_correct_slot_total_counts,
        "negative_tail_correct_slot_correct_counts": negative_tail_correct_slot_correct_counts,
        "negative_tail_correct_slot_ideal_total_counts": negative_tail_correct_slot_ideal_total_counts,
        "negative_tail_correct_slot_ideal_correct_counts": negative_tail_correct_slot_ideal_correct_counts,
        "top1_slot_total_counts": top1_slot_total_counts,
        "top1_slot_correct_counts": top1_slot_correct_counts,
        "top1_slot_ideal_total_counts": top1_slot_ideal_total_counts,
        "top1_slot_ideal_correct_counts": top1_slot_ideal_correct_counts,
        "negative_tail_top1_slot_total_counts": negative_tail_top1_slot_total_counts,
        "negative_tail_top1_slot_correct_counts": negative_tail_top1_slot_correct_counts,
        "negative_tail_top1_slot_ideal_total_counts": negative_tail_top1_slot_ideal_total_counts,
        "negative_tail_top1_slot_ideal_correct_counts": negative_tail_top1_slot_ideal_correct_counts,
        "top2_slot_total_counts": top2_slot_total_counts,
        "top2_slot_correct_counts": top2_slot_correct_counts,
        "top2_slot_ideal_total_counts": top2_slot_ideal_total_counts,
        "top2_slot_ideal_correct_counts": top2_slot_ideal_correct_counts,
        "negative_tail_top2_slot_total_counts": negative_tail_top2_slot_total_counts,
        "negative_tail_top2_slot_correct_counts": negative_tail_top2_slot_correct_counts,
        "negative_tail_top2_slot_ideal_total_counts": negative_tail_top2_slot_ideal_total_counts,
        "negative_tail_top2_slot_ideal_correct_counts": negative_tail_top2_slot_ideal_correct_counts,
        "correctness_buckets": correctness_buckets,
    }
    return summary, sample_records, pooled_payload


def _infer_task_name_from_path(path: str) -> Optional[str]:
    s = str(path).lower()
    if "results_arc" in s or "/arc" in s:
        return "arc"
    if "results_mmlu" in s or "/mmlu" in s:
        return "mmlu"
    if "results_csqa" in s or "/csqa" in s:
        return "csqa"
    return None


def _guess_model_tag_from_results_dir(path: str) -> str:
    norm = os.path.normpath(str(path))
    parts = [p for p in norm.split(os.sep) if p]
    for part in reversed(parts):
        if "shot" in part or "results_" in part or part in {"arc", "mmlu", "csqa"}:
            continue
        if "s_" in part:
            return part.split("s_", 1)[-1] or part
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1] if parts else "model"


def _merge_margin_noise_payload_into_bucket(bucket: Dict[str, object], payload: Dict[str, object]) -> None:
    bucket.setdefault("residuals", [])
    bucket.setdefault("residual_perm_labels", [])
    bucket.setdefault("residual_ideal_labels", [])
    bucket.setdefault("residual_correct_slot_labels", [])
    bucket.setdefault("residual_top1_slot_labels", [])
    bucket.setdefault("residual_top2_slot_labels", [])
    bucket.setdefault("residual_is_correct_flags", [])
    bucket.setdefault("z_scores", [])
    bucket.setdefault("sample_ref_margins", [])
    bucket.setdefault("sample_sigmas", [])
    bucket.setdefault("sample_base_gaps", [])
    bucket.setdefault("t_residuals", {})
    bucket.setdefault("t_z_scores", {})
    bucket.setdefault("standardized_bin_edges", [])
    bucket.setdefault("standardized_bin_label_counts", {})

    bucket["residuals"].extend(payload.get("residuals", []))
    bucket["residual_perm_labels"].extend([str(x) for x in (payload.get("residual_perm_labels", []) or [])])
    bucket["residual_ideal_labels"].extend([str(x) for x in (payload.get("residual_ideal_labels", []) or [])])
    bucket["residual_correct_slot_labels"].extend([str(x) for x in (payload.get("residual_correct_slot_labels", []) or [])])
    bucket["residual_top1_slot_labels"].extend([str(x) for x in (payload.get("residual_top1_slot_labels", []) or [])])
    bucket["residual_top2_slot_labels"].extend([str(x) for x in (payload.get("residual_top2_slot_labels", []) or [])])
    bucket["residual_is_correct_flags"].extend([bool(x) for x in (payload.get("residual_is_correct_flags", []) or [])])
    bucket["z_scores"].extend(payload.get("z_scores", []))
    bucket.setdefault("z_perm_labels", [])
    bucket["z_perm_labels"].extend([str(x) for x in (payload.get("z_perm_labels", []) or [])])
    bucket.setdefault("z_ideal_labels", [])
    bucket["z_ideal_labels"].extend([str(x) for x in (payload.get("z_ideal_labels", []) or [])])
    bucket.setdefault("z_correct_slot_labels", [])
    bucket["z_correct_slot_labels"].extend([str(x) for x in (payload.get("z_correct_slot_labels", []) or [])])
    bucket.setdefault("z_top1_slot_labels", [])
    bucket["z_top1_slot_labels"].extend([str(x) for x in (payload.get("z_top1_slot_labels", []) or [])])
    bucket.setdefault("z_top2_slot_labels", [])
    bucket["z_top2_slot_labels"].extend([str(x) for x in (payload.get("z_top2_slot_labels", []) or [])])
    bucket.setdefault("z_is_correct_flags", [])
    bucket["z_is_correct_flags"].extend([bool(x) for x in (payload.get("z_is_correct_flags", []) or [])])
    bucket["sample_ref_margins"].extend(payload.get("sample_ref_margins", []))
    bucket["sample_sigmas"].extend(payload.get("sample_sigmas", []))
    bucket["sample_base_gaps"].extend(payload.get("sample_base_gaps", []))

    for t, vals in (payload.get("t_residuals", {}) or {}).items():
        t_int = int(t)
        bucket["t_residuals"].setdefault(t_int, [])
        bucket["t_residuals"][t_int].extend(vals)
    for t, vals in (payload.get("t_z_scores", {}) or {}).items():
        t_int = int(t)
        bucket["t_z_scores"].setdefault(t_int, [])
        bucket["t_z_scores"][t_int].extend(vals)
    if payload.get("standardized_bin_edges"):
        bucket["standardized_bin_edges"] = list(payload.get("standardized_bin_edges", []))
    _merge_count_maps(
        bucket.setdefault("standardized_bin_label_counts", {}),
        payload.get("standardized_bin_label_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_ideal_counts", {}),
        payload.get("negative_tail_ideal_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_correct_slot_counts", {}),
        payload.get("negative_tail_correct_slot_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_perm_by_ideal", {}),
        payload.get("negative_tail_perm_by_ideal", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_perm_by_correct_slot", {}),
        payload.get("negative_tail_perm_by_correct_slot", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_top1_slot_counts", {}),
        payload.get("negative_tail_top1_slot_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_perm_by_top1_slot", {}),
        payload.get("negative_tail_perm_by_top1_slot", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_top2_slot_counts", {}),
        payload.get("negative_tail_top2_slot_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_perm_by_top2_slot", {}),
        payload.get("negative_tail_perm_by_top2_slot", {}) or {},
    )
    _merge_count_maps(bucket.setdefault("perm_total_counts", {}), payload.get("perm_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("perm_correct_counts", {}), payload.get("perm_correct_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("perm_ideal_total_counts", {}), payload.get("perm_ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("perm_ideal_correct_counts", {}), payload.get("perm_ideal_correct_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("negative_tail_perm_total_counts", {}), payload.get("negative_tail_perm_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("negative_tail_perm_correct_counts", {}), payload.get("negative_tail_perm_correct_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("negative_tail_perm_ideal_total_counts", {}), payload.get("negative_tail_perm_ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("negative_tail_perm_ideal_correct_counts", {}), payload.get("negative_tail_perm_ideal_correct_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("ideal_total_counts", {}), payload.get("ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("ideal_correct_counts", {}), payload.get("ideal_correct_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("negative_tail_ideal_correct_counts", {}), payload.get("negative_tail_ideal_correct_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("standardized_bin_ideal_total_counts", {}), payload.get("standardized_bin_ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("standardized_bin_ideal_correct_counts", {}), payload.get("standardized_bin_ideal_correct_counts", {}) or {})
    _merge_count_maps(
        bucket.setdefault("standardized_bin_total_counts", {}),
        payload.get("standardized_bin_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_correct_counts", {}),
        payload.get("standardized_bin_correct_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_perm_total_counts", {}),
        payload.get("standardized_bin_perm_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_perm_correct_counts", {}),
        payload.get("standardized_bin_perm_correct_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_perm_ideal_total_counts", {}),
        payload.get("standardized_bin_perm_ideal_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_perm_ideal_correct_counts", {}),
        payload.get("standardized_bin_perm_ideal_correct_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top1_slot_counts", {}),
        payload.get("standardized_bin_top1_slot_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top1_slot_total_counts", {}),
        payload.get("standardized_bin_top1_slot_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top1_slot_correct_counts", {}),
        payload.get("standardized_bin_top1_slot_correct_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top1_slot_ideal_total_counts", {}),
        payload.get("standardized_bin_top1_slot_ideal_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top1_slot_ideal_correct_counts", {}),
        payload.get("standardized_bin_top1_slot_ideal_correct_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top2_slot_counts", {}),
        payload.get("standardized_bin_top2_slot_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top2_slot_total_counts", {}),
        payload.get("standardized_bin_top2_slot_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top2_slot_correct_counts", {}),
        payload.get("standardized_bin_top2_slot_correct_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top2_slot_ideal_total_counts", {}),
        payload.get("standardized_bin_top2_slot_ideal_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("standardized_bin_top2_slot_ideal_correct_counts", {}),
        payload.get("standardized_bin_top2_slot_ideal_correct_counts", {}) or {},
    )
    bucket["entry_total_count"] = int(bucket.get("entry_total_count", 0)) + int(payload.get("entry_total_count", 0))
    bucket["entry_correct_count"] = int(bucket.get("entry_correct_count", 0)) + int(payload.get("entry_correct_count", 0))
    bucket["negative_tail_total_count"] = int(bucket.get("negative_tail_total_count", 0)) + int(payload.get("negative_tail_total_count", 0))
    bucket["negative_tail_correct_count"] = int(bucket.get("negative_tail_correct_count", 0)) + int(payload.get("negative_tail_correct_count", 0))
    _merge_count_maps(
        bucket.setdefault("correct_slot_total_counts", {}),
        payload.get("correct_slot_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("correct_slot_correct_counts", {}),
        payload.get("correct_slot_correct_counts", {}) or {},
    )
    _merge_count_maps(bucket.setdefault("correct_slot_ideal_total_counts", {}), payload.get("correct_slot_ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("correct_slot_ideal_correct_counts", {}), payload.get("correct_slot_ideal_correct_counts", {}) or {})
    _merge_count_maps(
        bucket.setdefault("negative_tail_correct_slot_total_counts", {}),
        payload.get("negative_tail_correct_slot_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_correct_slot_correct_counts", {}),
        payload.get("negative_tail_correct_slot_correct_counts", {}) or {},
    )
    _merge_count_maps(bucket.setdefault("negative_tail_correct_slot_ideal_total_counts", {}), payload.get("negative_tail_correct_slot_ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("negative_tail_correct_slot_ideal_correct_counts", {}), payload.get("negative_tail_correct_slot_ideal_correct_counts", {}) or {})
    _merge_count_maps(
        bucket.setdefault("top1_slot_total_counts", {}),
        payload.get("top1_slot_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("top1_slot_correct_counts", {}),
        payload.get("top1_slot_correct_counts", {}) or {},
    )
    _merge_count_maps(bucket.setdefault("top1_slot_ideal_total_counts", {}), payload.get("top1_slot_ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("top1_slot_ideal_correct_counts", {}), payload.get("top1_slot_ideal_correct_counts", {}) or {})
    _merge_count_maps(
        bucket.setdefault("negative_tail_top1_slot_total_counts", {}),
        payload.get("negative_tail_top1_slot_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_top1_slot_correct_counts", {}),
        payload.get("negative_tail_top1_slot_correct_counts", {}) or {},
    )
    _merge_count_maps(bucket.setdefault("negative_tail_top1_slot_ideal_total_counts", {}), payload.get("negative_tail_top1_slot_ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("negative_tail_top1_slot_ideal_correct_counts", {}), payload.get("negative_tail_top1_slot_ideal_correct_counts", {}) or {})
    _merge_count_maps(
        bucket.setdefault("top2_slot_total_counts", {}),
        payload.get("top2_slot_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("top2_slot_correct_counts", {}),
        payload.get("top2_slot_correct_counts", {}) or {},
    )
    _merge_count_maps(bucket.setdefault("top2_slot_ideal_total_counts", {}), payload.get("top2_slot_ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("top2_slot_ideal_correct_counts", {}), payload.get("top2_slot_ideal_correct_counts", {}) or {})
    _merge_count_maps(
        bucket.setdefault("negative_tail_top2_slot_total_counts", {}),
        payload.get("negative_tail_top2_slot_total_counts", {}) or {},
    )
    _merge_count_maps(
        bucket.setdefault("negative_tail_top2_slot_correct_counts", {}),
        payload.get("negative_tail_top2_slot_correct_counts", {}) or {},
    )
    _merge_count_maps(bucket.setdefault("negative_tail_top2_slot_ideal_total_counts", {}), payload.get("negative_tail_top2_slot_ideal_total_counts", {}) or {})
    _merge_count_maps(bucket.setdefault("negative_tail_top2_slot_ideal_correct_counts", {}), payload.get("negative_tail_top2_slot_ideal_correct_counts", {}) or {})
    payload_corr = payload.get("correctness_buckets", {}) or {}
    if payload_corr:
        dst_corr = bucket.setdefault("correctness_buckets", {})
        for key, sub_payload in payload_corr.items():
            sub_bucket = dst_corr.setdefault(str(key), _make_margin_noise_bucket(with_correctness=False))
            _merge_margin_noise_payload_into_bucket(sub_bucket, sub_payload or {})


def _summarize_margin_noise_bucket(
    bucket: Dict[str, object],
    n_views: int,
    reference_mode: str = "cyclic",
    negative_tail_z_cutoff: float = -1.5,
    right_tail_z_cutoff: float = 1.5,
) -> Dict[str, object]:
    residuals = np.asarray(bucket.get("residuals", []), dtype=np.float64)
    residual_perm_labels = [str(x) for x in (bucket.get("residual_perm_labels", []) or [])]
    residual_ideal_labels = [str(x) for x in (bucket.get("residual_ideal_labels", []) or [])]
    residual_correct_slot_labels = [str(x) for x in (bucket.get("residual_correct_slot_labels", []) or [])]
    residual_top1_slot_labels = [str(x) for x in (bucket.get("residual_top1_slot_labels", []) or [])]
    residual_top2_slot_labels = [str(x) for x in (bucket.get("residual_top2_slot_labels", []) or [])]
    residual_is_correct_flags = [bool(x) for x in (bucket.get("residual_is_correct_flags", []) or [])]
    z_scores = np.asarray(bucket.get("z_scores", []), dtype=np.float64)
    z_perm_labels = [str(x) for x in (bucket.get("z_perm_labels", []) or [])]
    z_ideal_labels = [str(x) for x in (bucket.get("z_ideal_labels", []) or [])]
    z_correct_slot_labels = [str(x) for x in (bucket.get("z_correct_slot_labels", []) or [])]
    z_top1_slot_labels = [str(x) for x in (bucket.get("z_top1_slot_labels", []) or [])]
    z_top2_slot_labels = [str(x) for x in (bucket.get("z_top2_slot_labels", []) or [])]
    z_is_correct_flags = [bool(x) for x in (bucket.get("z_is_correct_flags", []) or [])]
    sample_ref = np.asarray(bucket.get("sample_ref_margins", []), dtype=np.float64)
    sample_sigma = np.asarray(bucket.get("sample_sigmas", []), dtype=np.float64)
    sample_base_gap = np.asarray(bucket.get("sample_base_gaps", []), dtype=np.float64)
    t_residuals = bucket.get("t_residuals", {}) or {}
    t_z_scores = bucket.get("t_z_scores", {}) or {}
    std_bin_edges = bucket.get("standardized_bin_edges", []) or []
    std_bin_label_counts = bucket.get("standardized_bin_label_counts", {}) or {}
    std_bin_total_counts = bucket.get("standardized_bin_total_counts", {}) or {}
    std_bin_correct_counts = bucket.get("standardized_bin_correct_counts", {}) or {}
    std_bin_ideal_total_counts = bucket.get("standardized_bin_ideal_total_counts", {}) or {}
    std_bin_ideal_correct_counts = bucket.get("standardized_bin_ideal_correct_counts", {}) or {}
    std_bin_perm_total_counts = bucket.get("standardized_bin_perm_total_counts", {}) or {}
    std_bin_perm_correct_counts = bucket.get("standardized_bin_perm_correct_counts", {}) or {}
    std_bin_perm_ideal_total_counts = bucket.get("standardized_bin_perm_ideal_total_counts", {}) or {}
    std_bin_perm_ideal_correct_counts = bucket.get("standardized_bin_perm_ideal_correct_counts", {}) or {}
    std_bin_top1_slot_counts = bucket.get("standardized_bin_top1_slot_counts", {}) or {}
    std_bin_top1_slot_total_counts = bucket.get("standardized_bin_top1_slot_total_counts", {}) or {}
    std_bin_top1_slot_correct_counts = bucket.get("standardized_bin_top1_slot_correct_counts", {}) or {}
    std_bin_top1_slot_ideal_total_counts = bucket.get("standardized_bin_top1_slot_ideal_total_counts", {}) or {}
    std_bin_top1_slot_ideal_correct_counts = bucket.get("standardized_bin_top1_slot_ideal_correct_counts", {}) or {}
    std_bin_top2_slot_counts = bucket.get("standardized_bin_top2_slot_counts", {}) or {}
    std_bin_top2_slot_total_counts = bucket.get("standardized_bin_top2_slot_total_counts", {}) or {}
    std_bin_top2_slot_correct_counts = bucket.get("standardized_bin_top2_slot_correct_counts", {}) or {}
    std_bin_top2_slot_ideal_total_counts = bucket.get("standardized_bin_top2_slot_ideal_total_counts", {}) or {}
    std_bin_top2_slot_ideal_correct_counts = bucket.get("standardized_bin_top2_slot_ideal_correct_counts", {}) or {}
    ideal_total_counts = bucket.get("ideal_total_counts", {}) or {}
    ideal_correct_counts = bucket.get("ideal_correct_counts", {}) or {}
    t1_resid = np.asarray(t_residuals.get(1, []), dtype=np.float64)
    t1_std = float(np.std(t1_resid)) if t1_resid.size > 0 else float("nan")
    fit_raw = _gaussian_fit_report(residuals)
    fit_raw_laplace = _laplace_fit_report(residuals)
    fit_raw_cauchy = _cauchy_fit_report(residuals)
    raw_mu = float(fit_raw.get("mean", float("nan")))
    raw_sigma = float(fit_raw.get("std", float("nan")))
    raw_bin_edges: List[float] = []
    raw_top_bin: Dict[str, object] = {}
    raw_top_bin_subset: Dict[str, object] = {}
    raw_left_tail_subset: Dict[str, object] = {}
    raw_right_tail_subset: Dict[str, object] = {}
    if residuals.size > 0:
        raw_min = float(np.min(residuals))
        raw_max = float(np.max(residuals))
        if np.isfinite(raw_min) and np.isfinite(raw_max):
            if raw_max > raw_min:
                raw_bin_edges = np.linspace(raw_min, raw_max, 61, dtype=np.float64).tolist()
            else:
                raw_bin_edges = [raw_min - 0.5, raw_max + 0.5]
    if residuals.size > 0 and len(raw_bin_edges) >= 2:
        raw_bin_idxs = []
        raw_bin_counts: Dict[str, Dict[str, int]] = {}
        for resid_val, perm_label in zip(residuals.tolist(), residual_perm_labels):
            raw_bin_idx = int(np.digitize(float(resid_val), np.asarray(raw_bin_edges, dtype=np.float64)) - 1)
            raw_bin_idx = max(0, min(raw_bin_idx, len(raw_bin_edges) - 2))
            raw_bin_idxs.append(raw_bin_idx)
            raw_bin_counts.setdefault(str(raw_bin_idx), {})
            raw_bin_counts[str(raw_bin_idx)][str(perm_label)] = int(raw_bin_counts[str(raw_bin_idx)].get(str(perm_label), 0)) + 1
        raw_peak_bins = _summarize_peak_bins(bin_edges=raw_bin_edges, bin_label_counts=raw_bin_counts)
        if raw_peak_bins:
            raw_top_bin = raw_peak_bins[0]
            raw_bin_idx = int(raw_top_bin.get("bin_index", -1))
            raw_top_indices = [i for i, cur in enumerate(raw_bin_idxs) if int(cur) == raw_bin_idx]
            raw_top_bin_subset = _summarize_entry_subset(
                ideal_labels=[residual_ideal_labels[i] for i in raw_top_indices if i < len(residual_ideal_labels)],
                perm_labels=[residual_perm_labels[i] for i in raw_top_indices if i < len(residual_perm_labels)],
                correct_slot_labels=[residual_correct_slot_labels[i] for i in raw_top_indices if i < len(residual_correct_slot_labels)],
                top1_slot_labels=[residual_top1_slot_labels[i] for i in raw_top_indices if i < len(residual_top1_slot_labels)],
                top2_slot_labels=[residual_top2_slot_labels[i] for i in raw_top_indices if i < len(residual_top2_slot_labels)],
                correct_flags=[residual_is_correct_flags[i] for i in raw_top_indices if i < len(residual_is_correct_flags)],
                denom_correct_slot_counts=bucket.get("correct_slot_total_counts", {}) or {},
                denom_top1_slot_counts=bucket.get("top1_slot_total_counts", {}) or {},
                denom_top2_slot_counts=bucket.get("top2_slot_total_counts", {}) or {},
            )
        if np.isfinite(raw_mu) and np.isfinite(raw_sigma) and raw_sigma > 1e-12:
            raw_left_cut = float(raw_mu - raw_sigma)
            raw_right_cut = float(raw_mu + raw_sigma)
            raw_left_indices = [i for i, val in enumerate(residuals.tolist()) if float(val) <= raw_left_cut]
            raw_right_indices = [i for i, val in enumerate(residuals.tolist()) if float(val) >= raw_right_cut]
            raw_left_tail_subset = _summarize_entry_subset(
                ideal_labels=[residual_ideal_labels[i] for i in raw_left_indices if i < len(residual_ideal_labels)],
                perm_labels=[residual_perm_labels[i] for i in raw_left_indices if i < len(residual_perm_labels)],
                correct_slot_labels=[residual_correct_slot_labels[i] for i in raw_left_indices if i < len(residual_correct_slot_labels)],
                top1_slot_labels=[residual_top1_slot_labels[i] for i in raw_left_indices if i < len(residual_top1_slot_labels)],
                top2_slot_labels=[residual_top2_slot_labels[i] for i in raw_left_indices if i < len(residual_top2_slot_labels)],
                correct_flags=[residual_is_correct_flags[i] for i in raw_left_indices if i < len(residual_is_correct_flags)],
                denom_correct_slot_counts=bucket.get("correct_slot_total_counts", {}) or {},
                denom_top1_slot_counts=bucket.get("top1_slot_total_counts", {}) or {},
                denom_top2_slot_counts=bucket.get("top2_slot_total_counts", {}) or {},
            )
            raw_left_tail_subset["residual_cutoff"] = raw_left_cut
            raw_right_tail_subset = _summarize_entry_subset(
                ideal_labels=[residual_ideal_labels[i] for i in raw_right_indices if i < len(residual_ideal_labels)],
                perm_labels=[residual_perm_labels[i] for i in raw_right_indices if i < len(residual_perm_labels)],
                correct_slot_labels=[residual_correct_slot_labels[i] for i in raw_right_indices if i < len(residual_correct_slot_labels)],
                top1_slot_labels=[residual_top1_slot_labels[i] for i in raw_right_indices if i < len(residual_top1_slot_labels)],
                top2_slot_labels=[residual_top2_slot_labels[i] for i in raw_right_indices if i < len(residual_top2_slot_labels)],
                correct_flags=[residual_is_correct_flags[i] for i in raw_right_indices if i < len(residual_is_correct_flags)],
                denom_correct_slot_counts=bucket.get("correct_slot_total_counts", {}) or {},
                denom_top1_slot_counts=bucket.get("top1_slot_total_counts", {}) or {},
                denom_top2_slot_counts=bucket.get("top2_slot_total_counts", {}) or {},
            )
            raw_right_tail_subset["residual_cutoff"] = raw_right_cut
    peak_bin_summary = _summarize_peak_bins(
        bin_edges=std_bin_edges,
        bin_label_counts=std_bin_label_counts,
    )
    top_bin_accuracy = {}
    overall_perm_accuracy = _summarize_accuracy_rstd_by_group(
        bucket.get("perm_total_counts", {}) or {},
        bucket.get("perm_correct_counts", {}) or {},
        bucket.get("perm_ideal_total_counts", {}) or {},
        bucket.get("perm_ideal_correct_counts", {}) or {},
    )
    negative_tail_perm_accuracy = _summarize_accuracy_rstd_by_group(
        bucket.get("negative_tail_perm_total_counts", {}) or {},
        bucket.get("negative_tail_perm_correct_counts", {}) or {},
        bucket.get("negative_tail_perm_ideal_total_counts", {}) or {},
        bucket.get("negative_tail_perm_ideal_correct_counts", {}) or {},
    )
    right_tail_idxs = [i for i, z in enumerate(z_scores.tolist()) if float(z) >= float(right_tail_z_cutoff)]
    right_tail_perm_labels = [z_perm_labels[i] for i in right_tail_idxs if i < len(z_perm_labels)]
    right_tail_ideal_labels = [z_ideal_labels[i] for i in right_tail_idxs if i < len(z_ideal_labels)]
    right_tail_correct_slot_labels = [z_correct_slot_labels[i] for i in right_tail_idxs if i < len(z_correct_slot_labels)]
    right_tail_top1_slot_labels = [z_top1_slot_labels[i] for i in right_tail_idxs if i < len(z_top1_slot_labels)]
    right_tail_top2_slot_labels = [z_top2_slot_labels[i] for i in right_tail_idxs if i < len(z_top2_slot_labels)]
    right_tail_correct_flags = [z_is_correct_flags[i] for i in right_tail_idxs if i < len(z_is_correct_flags)]
    right_tail_perm_total_counts, right_tail_perm_ideal_total_counts = _count_labels_by_ideal(right_tail_perm_labels, right_tail_ideal_labels)
    right_tail_perm_correct_counts, right_tail_perm_ideal_correct_counts = _count_correct_by_label_and_ideal(
        right_tail_perm_labels, right_tail_ideal_labels, right_tail_correct_flags
    )
    right_tail_perm_accuracy = _summarize_accuracy_rstd_by_group(
        right_tail_perm_total_counts,
        right_tail_perm_correct_counts,
        right_tail_perm_ideal_total_counts,
        right_tail_perm_ideal_correct_counts,
    )
    right_tail_ideal_counts_map = _count_labels(right_tail_ideal_labels)
    right_tail_ideal_correct_counts_map = _count_labels([label for label, flag in zip(right_tail_ideal_labels, right_tail_correct_flags) if flag])
    right_tail_correct_slot_total_counts, right_tail_correct_slot_ideal_total_counts = _count_labels_by_ideal(right_tail_correct_slot_labels, right_tail_ideal_labels)
    right_tail_correct_slot_correct_counts, right_tail_correct_slot_ideal_correct_counts = _count_correct_by_label_and_ideal(
        right_tail_correct_slot_labels, right_tail_ideal_labels, right_tail_correct_flags
    )
    right_tail_top1_slot_total_counts, right_tail_top1_slot_ideal_total_counts = _count_labels_by_ideal(right_tail_top1_slot_labels, right_tail_ideal_labels)
    right_tail_top1_slot_correct_counts, right_tail_top1_slot_ideal_correct_counts = _count_correct_by_label_and_ideal(
        right_tail_top1_slot_labels, right_tail_ideal_labels, right_tail_correct_flags
    )
    right_tail_top2_slot_total_counts, right_tail_top2_slot_ideal_total_counts = _count_labels_by_ideal(right_tail_top2_slot_labels, right_tail_ideal_labels)
    right_tail_top2_slot_correct_counts, right_tail_top2_slot_ideal_correct_counts = _count_correct_by_label_and_ideal(
        right_tail_top2_slot_labels, right_tail_ideal_labels, right_tail_correct_flags
    )
    top_bin_perm_accuracy = []
    top_bin_ideal_counts_map: Dict[str, int] = {}
    top_bin_correct_slot_total_counts: Dict[str, int] = {}
    top_bin_correct_slot_ideal_total_counts: Dict[str, Dict[str, int]] = {}
    top_bin_correct_slot_correct_counts: Dict[str, int] = {}
    top_bin_correct_slot_ideal_correct_counts: Dict[str, Dict[str, int]] = {}
    top_bin_top1_slot_incidence = []
    top_bin_top2_slot_incidence = []
    top_bin_top1_counts = []
    top_bin_top1_accuracy = []
    top_bin_top2_counts = []
    top_bin_top2_accuracy = []
    if peak_bin_summary:
        top_bin = peak_bin_summary[0]
        bin_idx = int(top_bin.get("bin_index", -1))
        total = int(std_bin_total_counts.get(str(bin_idx), 0))
        correct = int(std_bin_correct_counts.get(str(bin_idx), 0))
        top_bin_accuracy = {
            "bin_index": int(bin_idx),
            "range_left": float(top_bin.get("range_left", float("nan"))),
            "range_right": float(top_bin.get("range_right", float("nan"))),
            "count": int(total),
            "correct": int(correct),
            "accuracy": float(correct / total) if total > 0 else float("nan"),
            "recall_std": _recall_std_from_counts(
                std_bin_ideal_total_counts.get(str(bin_idx), {}) if isinstance(std_bin_ideal_total_counts.get(str(bin_idx), {}), dict) else {},
                std_bin_ideal_correct_counts.get(str(bin_idx), {}) if isinstance(std_bin_ideal_correct_counts.get(str(bin_idx), {}), dict) else {},
            ),
        }
        top_bin_idxs = [i for i, z in enumerate(z_scores.tolist()) if max(0, min(int(np.digitize(float(z), std_bin_edges) - 1), len(std_bin_edges) - 2)) == bin_idx]
        top_bin_ideal_labels = [z_ideal_labels[i] for i in top_bin_idxs if i < len(z_ideal_labels)]
        top_bin_correct_slot_labels = [z_correct_slot_labels[i] for i in top_bin_idxs if i < len(z_correct_slot_labels)]
        top_bin_correct_flags = [z_is_correct_flags[i] for i in top_bin_idxs if i < len(z_is_correct_flags)]
        top_bin_ideal_counts_map = _count_labels(top_bin_ideal_labels)
        top_bin_correct_slot_total_counts, top_bin_correct_slot_ideal_total_counts = _count_labels_by_ideal(
            top_bin_correct_slot_labels, top_bin_ideal_labels
        )
        top_bin_correct_slot_correct_counts, top_bin_correct_slot_ideal_correct_counts = _count_correct_by_label_and_ideal(
            top_bin_correct_slot_labels, top_bin_ideal_labels, top_bin_correct_flags
        )
        top_bin_perm_accuracy = _summarize_accuracy_rstd_by_group(
            std_bin_perm_total_counts.get(str(bin_idx), {}) if isinstance(std_bin_perm_total_counts.get(str(bin_idx), {}), dict) else {},
            std_bin_perm_correct_counts.get(str(bin_idx), {}) if isinstance(std_bin_perm_correct_counts.get(str(bin_idx), {}), dict) else {},
            std_bin_perm_ideal_total_counts.get(str(bin_idx), {}) if isinstance(std_bin_perm_ideal_total_counts.get(str(bin_idx), {}), dict) else {},
            std_bin_perm_ideal_correct_counts.get(str(bin_idx), {}) if isinstance(std_bin_perm_ideal_correct_counts.get(str(bin_idx), {}), dict) else {},
        )
        top_bin_top1_counts = _summarize_flat_counts(
            std_bin_top1_slot_counts.get(str(bin_idx), {}) if isinstance(std_bin_top1_slot_counts.get(str(bin_idx), {}), dict) else {}
        )
        top_bin_top1_slot_incidence = _summarize_rate_by_group(
            std_bin_top1_slot_total_counts.get(str(bin_idx), {}) if isinstance(std_bin_top1_slot_total_counts.get(str(bin_idx), {}), dict) else {},
            bucket.get("top1_slot_total_counts", {}) or {},
        )
        top_bin_top1_accuracy = _summarize_accuracy_rstd_by_group(
            std_bin_top1_slot_total_counts.get(str(bin_idx), {}) if isinstance(std_bin_top1_slot_total_counts.get(str(bin_idx), {}), dict) else {},
            std_bin_top1_slot_correct_counts.get(str(bin_idx), {}) if isinstance(std_bin_top1_slot_correct_counts.get(str(bin_idx), {}), dict) else {},
            std_bin_top1_slot_ideal_total_counts.get(str(bin_idx), {}) if isinstance(std_bin_top1_slot_ideal_total_counts.get(str(bin_idx), {}), dict) else {},
            std_bin_top1_slot_ideal_correct_counts.get(str(bin_idx), {}) if isinstance(std_bin_top1_slot_ideal_correct_counts.get(str(bin_idx), {}), dict) else {},
        )
        top_bin_top2_counts = _summarize_flat_counts(
            std_bin_top2_slot_counts.get(str(bin_idx), {}) if isinstance(std_bin_top2_slot_counts.get(str(bin_idx), {}), dict) else {}
        )
        top_bin_top2_slot_incidence = _summarize_rate_by_group(
            std_bin_top2_slot_total_counts.get(str(bin_idx), {}) if isinstance(std_bin_top2_slot_total_counts.get(str(bin_idx), {}), dict) else {},
            bucket.get("top2_slot_total_counts", {}) or {},
        )
        top_bin_top2_accuracy = _summarize_accuracy_rstd_by_group(
            std_bin_top2_slot_total_counts.get(str(bin_idx), {}) if isinstance(std_bin_top2_slot_total_counts.get(str(bin_idx), {}), dict) else {},
            std_bin_top2_slot_correct_counts.get(str(bin_idx), {}) if isinstance(std_bin_top2_slot_correct_counts.get(str(bin_idx), {}), dict) else {},
            std_bin_top2_slot_ideal_total_counts.get(str(bin_idx), {}) if isinstance(std_bin_top2_slot_ideal_total_counts.get(str(bin_idx), {}), dict) else {},
            std_bin_top2_slot_ideal_correct_counts.get(str(bin_idx), {}) if isinstance(std_bin_top2_slot_ideal_correct_counts.get(str(bin_idx), {}), dict) else {},
        )

    t_view_summary = []
    for t in sorted(int(x) for x in t_residuals.keys()):
        resid_arr = np.asarray(t_residuals.get(t, []), dtype=np.float64)
        z_arr = np.asarray(t_z_scores.get(t, []), dtype=np.float64)
        resid_std = float(np.std(resid_arr)) if resid_arr.size > 0 else float("nan")
        sqrt_target = float(t1_std / math.sqrt(float(t))) if np.isfinite(t1_std) else float("nan")
        finite_target = (
            float(t1_std * math.sqrt(float(max(0, int(n_views) - int(t))) / float(max(1, int(t)) * max(1, int(n_views) - 1))))
            if np.isfinite(t1_std) and int(n_views) > 1
            else float("nan")
        )
        t_view_summary.append({
            "t": int(t),
            "n_residuals": int(resid_arr.size),
            "resid_std": resid_std,
            "resid_abs_mean": float(np.mean(np.abs(resid_arr))) if resid_arr.size > 0 else float("nan"),
            "sqrt_target_from_t1": sqrt_target,
            "finite_view_target_from_t1": finite_target,
            "ratio_to_t1_sqrt": float(resid_std / sqrt_target) if np.isfinite(resid_std) and np.isfinite(sqrt_target) and sqrt_target > 1e-12 else float("nan"),
            "ratio_to_finite_view_target": float(resid_std / finite_target) if np.isfinite(resid_std) and np.isfinite(finite_target) and finite_target > 1e-12 else float("nan"),
            "standardized_residual_fit": _gaussian_fit_report(z_arr),
        })

    return {
        "reference_mode": str(reference_mode),
        "n_views": int(n_views),
        "n_residuals": int(residuals.size),
        "n_standardized_residuals": int(z_scores.size),
        "n_samples": int(sample_sigma.size),
        "pooled_residual_fit": fit_raw,
        "pooled_residual_laplace_fit": fit_raw_laplace,
        "pooled_residual_cauchy_fit": fit_raw_cauchy,
        "raw_correct_slot_fits": _summarize_fit_reports_by_group(
            bucket.get("residuals", []) or [],
            bucket.get("residual_correct_slot_labels", []) or [],
        ),
        "raw_top1_slot_fits": _summarize_fit_reports_by_group(
            bucket.get("residuals", []) or [],
            bucket.get("residual_top1_slot_labels", []) or [],
        ),
        "raw_top2_slot_fits": _summarize_fit_reports_by_group(
            bucket.get("residuals", []) or [],
            bucket.get("residual_top2_slot_labels", []) or [],
        ),
        "standardized_residual_fit": _gaussian_fit_report(z_scores),
        "raw_top_bin_summary": raw_top_bin,
        "raw_top_bin_subset": raw_top_bin_subset,
        "raw_left_tail_subset": raw_left_tail_subset,
        "raw_right_tail_subset": raw_right_tail_subset,
        "entry_accuracy": {
            "count": int(bucket.get("entry_total_count", 0)),
            "correct": int(bucket.get("entry_correct_count", 0)),
            "accuracy": (
                float(int(bucket.get("entry_correct_count", 0)) / int(bucket.get("entry_total_count", 0)))
                if int(bucket.get("entry_total_count", 0)) > 0
                else float("nan")
            ),
            "recall_std": _recall_std_from_counts(ideal_total_counts, ideal_correct_counts),
        },
        "negative_tail_accuracy": {
            "z_cutoff": float(negative_tail_z_cutoff),
            "count": int(bucket.get("negative_tail_total_count", 0)),
            "correct": int(bucket.get("negative_tail_correct_count", 0)),
            "accuracy": (
                float(int(bucket.get("negative_tail_correct_count", 0)) / int(bucket.get("negative_tail_total_count", 0)))
                if int(bucket.get("negative_tail_total_count", 0)) > 0
                else float("nan")
            ),
            "recall_std": _recall_std_from_counts(
                bucket.get("negative_tail_ideal_counts", {}) or {},
                bucket.get("negative_tail_ideal_correct_counts", {}) or {},
            ),
        },
        "right_tail_accuracy": {
            "z_cutoff": float(right_tail_z_cutoff),
            "count": int(len(right_tail_idxs)),
            "correct": int(sum(1 for x in right_tail_correct_flags if x)),
            "accuracy": (
                float(sum(1 for x in right_tail_correct_flags if x) / len(right_tail_correct_flags))
                if right_tail_correct_flags
                else float("nan")
            ),
            "recall_std": _recall_std_from_counts(
                right_tail_ideal_counts_map,
                right_tail_ideal_correct_counts_map,
            ),
        },
        "top_standardized_bin_accuracy": top_bin_accuracy,
        "top_standardized_bin_ideal_counts": _summarize_flat_counts(top_bin_ideal_counts_map),
        "top_standardized_bin_correct_slot_counts": _summarize_flat_counts(top_bin_correct_slot_total_counts),
        "top_standardized_bin_slot_incidence": _summarize_rate_by_group(
            top_bin_correct_slot_total_counts,
            bucket.get("correct_slot_total_counts", {}) or {},
        ),
        "top_standardized_bin_correct_slot_accuracy": _summarize_accuracy_rstd_by_group(
            top_bin_correct_slot_total_counts,
            top_bin_correct_slot_correct_counts,
            top_bin_correct_slot_ideal_total_counts,
            top_bin_correct_slot_ideal_correct_counts,
        ),
        "overall_perm_accuracy": overall_perm_accuracy,
        "negative_tail_perm_accuracy": negative_tail_perm_accuracy,
        "right_tail_perm_accuracy": right_tail_perm_accuracy,
        "top_standardized_bin_perm_accuracy": top_bin_perm_accuracy,
        "top_standardized_bin_top1_slot_counts": top_bin_top1_counts,
        "top_standardized_bin_top1_slot_incidence": top_bin_top1_slot_incidence,
        "top_standardized_bin_top1_slot_accuracy": top_bin_top1_accuracy,
        "top_standardized_bin_top2_slot_counts": top_bin_top2_counts,
        "top_standardized_bin_top2_slot_incidence": top_bin_top2_slot_incidence,
        "top_standardized_bin_top2_slot_accuracy": top_bin_top2_accuracy,
        "mean_sample_ref_margin": float(np.mean(sample_ref)) if sample_ref.size > 0 else float("nan"),
        "mean_sample_sigma_i": float(np.mean(sample_sigma)) if sample_sigma.size > 0 else float("nan"),
        "corr_ref_margin_sigma": _safe_corr(sample_ref, sample_sigma),
        "corr_base_gap_sigma": _safe_corr(sample_base_gap, sample_sigma),
        "peak_bin_summary": peak_bin_summary,
        "negative_tail_summary": _summarize_tail_bins(
            bin_edges=std_bin_edges,
            bin_label_counts=std_bin_label_counts,
            max_right_edge=float(negative_tail_z_cutoff),
        ),
        "right_tail_summary": _summarize_tail_bins(
            bin_edges=std_bin_edges,
            bin_label_counts=std_bin_label_counts,
            min_left_edge=float(right_tail_z_cutoff),
        ),
        "negative_tail_ideal_counts": _summarize_flat_counts(bucket.get("negative_tail_ideal_counts", {}) or {}),
        "right_tail_ideal_counts": _summarize_flat_counts(right_tail_ideal_counts_map),
        "negative_tail_correct_slot_counts": _summarize_flat_counts(bucket.get("negative_tail_correct_slot_counts", {}) or {}),
        "right_tail_correct_slot_counts": _summarize_flat_counts(right_tail_correct_slot_total_counts),
        "negative_tail_by_ideal": _summarize_grouped_counts(bucket.get("negative_tail_perm_by_ideal", {}) or {}),
        "negative_tail_by_correct_slot": _summarize_grouped_counts(bucket.get("negative_tail_perm_by_correct_slot", {}) or {}),
        "negative_tail_top1_slot_counts": _summarize_flat_counts(bucket.get("negative_tail_top1_slot_counts", {}) or {}),
        "negative_tail_by_top1_slot": _summarize_grouped_counts(bucket.get("negative_tail_perm_by_top1_slot", {}) or {}),
        "negative_tail_top2_slot_counts": _summarize_flat_counts(bucket.get("negative_tail_top2_slot_counts", {}) or {}),
        "negative_tail_by_top2_slot": _summarize_grouped_counts(bucket.get("negative_tail_perm_by_top2_slot", {}) or {}),
        "right_tail_top1_slot_counts": _summarize_flat_counts(right_tail_top1_slot_total_counts),
        "right_tail_top2_slot_counts": _summarize_flat_counts(right_tail_top2_slot_total_counts),
        "negative_tail_slot_incidence": _summarize_rate_by_group(
            bucket.get("negative_tail_correct_slot_total_counts", {}) or {},
            bucket.get("correct_slot_total_counts", {}) or {},
        ),
        "right_tail_slot_incidence": _summarize_rate_by_group(
            right_tail_correct_slot_total_counts,
            bucket.get("correct_slot_total_counts", {}) or {},
        ),
        "negative_tail_top1_slot_incidence": _summarize_rate_by_group(
            bucket.get("negative_tail_top1_slot_total_counts", {}) or {},
            bucket.get("top1_slot_total_counts", {}) or {},
        ),
        "right_tail_top1_slot_incidence": _summarize_rate_by_group(
            right_tail_top1_slot_total_counts,
            bucket.get("top1_slot_total_counts", {}) or {},
        ),
        "negative_tail_top2_slot_incidence": _summarize_rate_by_group(
            bucket.get("negative_tail_top2_slot_total_counts", {}) or {},
            bucket.get("top2_slot_total_counts", {}) or {},
        ),
        "right_tail_top2_slot_incidence": _summarize_rate_by_group(
            right_tail_top2_slot_total_counts,
            bucket.get("top2_slot_total_counts", {}) or {},
        ),
        "correct_slot_accuracy": _summarize_accuracy_rstd_by_group(
            bucket.get("correct_slot_total_counts", {}) or {},
            bucket.get("correct_slot_correct_counts", {}) or {},
            bucket.get("correct_slot_ideal_total_counts", {}) or {},
            bucket.get("correct_slot_ideal_correct_counts", {}) or {},
        ),
        "negative_tail_correct_slot_accuracy": _summarize_accuracy_rstd_by_group(
            bucket.get("negative_tail_correct_slot_total_counts", {}) or {},
            bucket.get("negative_tail_correct_slot_correct_counts", {}) or {},
            bucket.get("negative_tail_correct_slot_ideal_total_counts", {}) or {},
            bucket.get("negative_tail_correct_slot_ideal_correct_counts", {}) or {},
        ),
        "right_tail_correct_slot_accuracy": _summarize_accuracy_rstd_by_group(
            right_tail_correct_slot_total_counts,
            right_tail_correct_slot_correct_counts,
            right_tail_correct_slot_ideal_total_counts,
            right_tail_correct_slot_ideal_correct_counts,
        ),
        "top1_slot_accuracy": _summarize_accuracy_rstd_by_group(
            bucket.get("top1_slot_total_counts", {}) or {},
            bucket.get("top1_slot_correct_counts", {}) or {},
            bucket.get("top1_slot_ideal_total_counts", {}) or {},
            bucket.get("top1_slot_ideal_correct_counts", {}) or {},
        ),
        "negative_tail_top1_slot_accuracy": _summarize_accuracy_rstd_by_group(
            bucket.get("negative_tail_top1_slot_total_counts", {}) or {},
            bucket.get("negative_tail_top1_slot_correct_counts", {}) or {},
            bucket.get("negative_tail_top1_slot_ideal_total_counts", {}) or {},
            bucket.get("negative_tail_top1_slot_ideal_correct_counts", {}) or {},
        ),
        "right_tail_top1_slot_accuracy": _summarize_accuracy_rstd_by_group(
            right_tail_top1_slot_total_counts,
            right_tail_top1_slot_correct_counts,
            right_tail_top1_slot_ideal_total_counts,
            right_tail_top1_slot_ideal_correct_counts,
        ),
        "top2_slot_accuracy": _summarize_accuracy_rstd_by_group(
            bucket.get("top2_slot_total_counts", {}) or {},
            bucket.get("top2_slot_correct_counts", {}) or {},
            bucket.get("top2_slot_ideal_total_counts", {}) or {},
            bucket.get("top2_slot_ideal_correct_counts", {}) or {},
        ),
        "negative_tail_top2_slot_accuracy": _summarize_accuracy_rstd_by_group(
            bucket.get("negative_tail_top2_slot_total_counts", {}) or {},
            bucket.get("negative_tail_top2_slot_correct_counts", {}) or {},
            bucket.get("negative_tail_top2_slot_ideal_total_counts", {}) or {},
            bucket.get("negative_tail_top2_slot_ideal_correct_counts", {}) or {},
        ),
        "right_tail_top2_slot_accuracy": _summarize_accuracy_rstd_by_group(
            right_tail_top2_slot_total_counts,
            right_tail_top2_slot_correct_counts,
            right_tail_top2_slot_ideal_total_counts,
            right_tail_top2_slot_ideal_correct_counts,
        ),
        "correctness_split": {
            str(name): _summarize_margin_noise_bucket(
                sub_bucket,
                n_views=int(n_views),
                reference_mode=str(reference_mode),
                negative_tail_z_cutoff=float(negative_tail_z_cutoff),
                right_tail_z_cutoff=float(right_tail_z_cutoff),
            )
            for name, sub_bucket in (bucket.get("correctness_buckets", {}) or {}).items()
            if isinstance(sub_bucket, dict)
        },
        "t_view_summary": t_view_summary,
    }


def _save_raw_fit_by_label_plot(
    *,
    plt,
    values: Sequence[float],
    labels: Sequence[object],
    out_path: str,
    title: str,
) -> Optional[str]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    lab_list = [str(x) for x in labels]
    n = min(int(arr.size), len(lab_list))
    if n <= 0:
        return None
    arr = arr[:n]
    lab_list = lab_list[:n]
    grouped: Dict[str, np.ndarray] = {}
    label_arr = np.asarray(lab_list, dtype=object)
    for label in sorted(set(lab_list)):
        vals = arr[label_arr == label]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            grouped[str(label)] = vals
    if not grouped:
        return None

    labels_sorted = sorted(grouped.keys())
    n_panels = len(labels_sorted)
    ncols = min(3, max(1, n_panels))
    nrows = int(math.ceil(float(n_panels) / float(ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.8 * nrows), dpi=180)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, label in zip(axes_arr, labels_sorted):
        vals = grouped[label]
        fit_g = _gaussian_fit_report(vals)
        fit_l = _laplace_fit_report(vals)
        fit_c = _cauchy_fit_report(vals)
        ax.hist(vals, bins=40, density=True, alpha=0.68, color="#6baed6", label=f"{label} residuals")
        x_min = float(np.min(vals))
        x_max = float(np.max(vals))
        if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
            xs = np.linspace(x_min, x_max, 400)
            mu = float(fit_g.get("mean", float("nan")))
            sigma = float(fit_g.get("std", float("nan")))
            if np.isfinite(mu) and np.isfinite(sigma) and sigma > 1e-12:
                pdf_g = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
                ax.plot(
                    xs,
                    pdf_g,
                    color="#d62728",
                    lw=1.8,
                    label="G ({:.3f}/{:.3f})".format(
                        float(fit_g.get("ks_to_fit", float("nan"))),
                        float(fit_g.get("kl_hist_to_fit", float("nan"))),
                    ),
                )
            pdf_l = _laplace_pdf(xs, float(fit_l.get("loc", float("nan"))), float(fit_l.get("scale", float("nan"))))
            if np.all(np.isfinite(pdf_l)):
                ax.plot(
                    xs,
                    pdf_l,
                    color="#2ca02c",
                    lw=1.6,
                    ls="--",
                    label="L ({:.3f}/{:.3f})".format(
                        float(fit_l.get("ks_to_fit", float("nan"))),
                        float(fit_l.get("kl_hist_to_fit", float("nan"))),
                    ),
                )
            pdf_c = _cauchy_pdf(xs, float(fit_c.get("loc", float("nan"))), float(fit_c.get("scale", float("nan"))))
            if np.all(np.isfinite(pdf_c)):
                ax.plot(
                    xs,
                    pdf_c,
                    color="#9467bd",
                    lw=1.6,
                    ls=":",
                    label="C ({:.3f}/{:.3f})".format(
                        float(fit_c.get("ks_to_fit", float("nan"))),
                        float(fit_c.get("kl_hist_to_fit", float("nan"))),
                    ),
                )
        ax.set_title(f"slot={label}, n={int(vals.size)}", fontsize=10)
        ax.set_xlabel("raw residual")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes_arr[len(labels_sorted):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_raw_param_sweep_plot(
    *,
    plt,
    values: Sequence[float],
    out_path: str,
    title: str,
    laplace_loc: float,
    laplace_scale: float,
    cauchy_loc: float,
    cauchy_scale: float,
    scale_multipliers: Sequence[float] = (0.7, 1.0, 1.5),
) -> Optional[str]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return None
    x_min = float(np.min(arr))
    x_max = float(np.max(arr))
    if not (np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min):
        return None

    fig = plt.figure(figsize=(8.4, 5.2), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(arr, bins=60, density=True, alpha=0.68, color="#6baed6", label="Raw residuals")
    xs = np.linspace(x_min, x_max, 500)

    laplace_colors = ["#2ca02c", "#1b9e77", "#66a61e"]
    cauchy_colors = ["#9467bd", "#8c564b", "#e377c2"]
    for idx, mult in enumerate(scale_multipliers):
        scale = float(laplace_scale) * float(mult)
        pdf = _laplace_pdf(xs, float(laplace_loc), float(scale))
        if np.all(np.isfinite(pdf)):
            ax.plot(
                xs,
                pdf,
                color=laplace_colors[idx % len(laplace_colors)],
                lw=1.6,
                ls="--",
                label=f"Laplace x{float(mult):.1f}",
            )
    for idx, mult in enumerate(scale_multipliers):
        scale = float(cauchy_scale) * float(mult)
        pdf = _cauchy_pdf(xs, float(cauchy_loc), float(scale))
        if np.all(np.isfinite(pdf)):
            ax.plot(
                xs,
                pdf,
                color=cauchy_colors[idx % len(cauchy_colors)],
                lw=1.6,
                ls=":",
                label=f"Cauchy x{float(mult):.1f}",
            )

    ax.set_title(title)
    ax.set_xlabel("raw residual")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_multi_model_margin_plots(
    *,
    combined_by_k: Dict[int, Dict[str, object]],
    out_dir: str,
) -> List[str]:
    plt = _try_import_matplotlib()
    if plt is None:
        print("[warn] matplotlib not available; skipping multi-model plot saving.")
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    for k in sorted(combined_by_k.keys()):
        rec = combined_by_k[int(k)]
        pooled = rec.get("pooled_summary", {}) or {}
        per_model = rec.get("per_model", []) or []
        ref_mode = str((pooled or {}).get("reference_mode", rec.get("reference_mode", "cyclic")))
        mode_label = "Full-permutation" if ref_mode == "full" else "Cyclic"
        raw_residuals = np.asarray(rec.get("pooled_bucket", {}).get("residuals", []), dtype=np.float64)
        z_scores = np.asarray(rec.get("pooled_bucket", {}).get("z_scores", []), dtype=np.float64)
        t_residuals = rec.get("pooled_bucket", {}).get("t_residuals", {}) or {}

        if raw_residuals.size > 0:
            fig = plt.figure(figsize=(8.0, 5.0), dpi=180)
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(raw_residuals, bins=60, density=True, alpha=0.68, color="#6baed6", label="All models pooled residual")
            fit_raw = pooled.get("pooled_residual_fit", {}) or {}
            fit_laplace = pooled.get("pooled_residual_laplace_fit", {}) or {}
            fit_cauchy = pooled.get("pooled_residual_cauchy_fit", {}) or {}
            mu = float(fit_raw.get("mean", float("nan")))
            sigma = float(fit_raw.get("std", float("nan")))
            x_min = float(np.min(raw_residuals))
            x_max = float(np.max(raw_residuals))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                xs = np.linspace(x_min, x_max, 400)
                if np.isfinite(mu) and np.isfinite(sigma) and sigma > 1e-12:
                    pdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
                    ax.plot(
                        xs,
                        pdf,
                        color="#d62728",
                        lw=2.0,
                        label="Gaussian (KS={:.3f}, KL={:.3f})".format(
                            float(fit_raw.get("ks_to_fit", float("nan"))),
                            float(fit_raw.get("kl_hist_to_fit", float("nan"))),
                        ),
                    )
                laplace_pdf = _laplace_pdf(xs, float(fit_laplace.get("loc", float("nan"))), float(fit_laplace.get("scale", float("nan"))))
                if np.all(np.isfinite(laplace_pdf)):
                    ax.plot(
                        xs,
                        laplace_pdf,
                        color="#2ca02c",
                        lw=1.8,
                        ls="--",
                        label="Laplace (KS={:.3f}, KL={:.3f})".format(
                            float(fit_laplace.get("ks_to_fit", float("nan"))),
                            float(fit_laplace.get("kl_hist_to_fit", float("nan"))),
                        ),
                    )
                cauchy_pdf = _cauchy_pdf(xs, float(fit_cauchy.get("loc", float("nan"))), float(fit_cauchy.get("scale", float("nan"))))
                if np.all(np.isfinite(cauchy_pdf)):
                    ax.plot(
                        xs,
                        cauchy_pdf,
                        color="#9467bd",
                        lw=1.8,
                        ls=":",
                        label="Cauchy (KS={:.3f}, KL={:.3f})".format(
                            float(fit_cauchy.get("ks_to_fit", float("nan"))),
                            float(fit_cauchy.get("kl_hist_to_fit", float("nan"))),
                        ),
                    )
            ax.set_title(
                f"Multi-model pooled {ref_mode} residuals (raw) (k={k})\n"
                f"models={len(per_model)}, compare Gaussian / Laplace / Cauchy"
            )
            ax.set_xlabel("residual = margin - ref_margin")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = os.path.join(out_dir, f"multi_model_margin_noise_raw_hist_k{k}.png")
            fig.savefig(p)
            plt.close(fig)
            saved.append(p)
            p_slot = os.path.join(out_dir, f"multi_model_margin_noise_raw_hist_by_correct_slot_k{k}.png")
            saved_slot = _save_raw_fit_by_label_plot(
                plt=plt,
                values=raw_residuals,
                labels=rec.get("pooled_bucket", {}).get("residual_correct_slot_labels", []) or [],
                out_path=p_slot,
                title=f"Multi-model pooled {ref_mode} raw residuals by correct slot (k={k})",
            )
            if saved_slot:
                saved.append(saved_slot)
            p_top1 = os.path.join(out_dir, f"multi_model_margin_noise_raw_hist_by_top1_slot_k{k}.png")
            saved_top1 = _save_raw_fit_by_label_plot(
                plt=plt,
                values=raw_residuals,
                labels=rec.get("pooled_bucket", {}).get("residual_top1_slot_labels", []) or [],
                out_path=p_top1,
                title=f"Multi-model pooled {ref_mode} raw residuals by top1 slot (k={k})",
            )
            if saved_top1:
                saved.append(saved_top1)
            p_top2 = os.path.join(out_dir, f"multi_model_margin_noise_raw_hist_by_top2_slot_k{k}.png")
            saved_top2 = _save_raw_fit_by_label_plot(
                plt=plt,
                values=raw_residuals,
                labels=rec.get("pooled_bucket", {}).get("residual_top2_slot_labels", []) or [],
                out_path=p_top2,
                title=f"Multi-model pooled {ref_mode} raw residuals by top2 slot (k={k})",
            )
            if saved_top2:
                saved.append(saved_top2)
            p_sweep = os.path.join(out_dir, f"multi_model_margin_noise_raw_param_sweep_k{k}.png")
            saved_sweep = _save_raw_param_sweep_plot(
                plt=plt,
                values=raw_residuals,
                out_path=p_sweep,
                title=f"Multi-model pooled {ref_mode} raw residual parameter sweep (k={k})",
                laplace_loc=float(fit_laplace.get("loc", float("nan"))),
                laplace_scale=float(fit_laplace.get("scale", float("nan"))),
                cauchy_loc=float(fit_cauchy.get("loc", float("nan"))),
                cauchy_scale=float(fit_cauchy.get("scale", float("nan"))),
            )
            if saved_sweep:
                saved.append(saved_sweep)

        if z_scores.size > 0:
            fig = plt.figure(figsize=(8.0, 5.0), dpi=180)
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(z_scores, bins=60, density=True, alpha=0.68, color="#1f77b4", label="All models pooled z")
            x_min = float(np.min(z_scores))
            x_max = float(np.max(z_scores))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                xs = np.linspace(max(-4.0, x_min), min(4.0, x_max), 300)
                normal_pdf = np.exp(-0.5 * xs * xs) / math.sqrt(2.0 * math.pi)
                ax.plot(xs, normal_pdf, color="#d62728", lw=2.0, label="N(0,1)")
            fit = pooled.get("standardized_residual_fit", {}) or {}
            ax.set_title(
                f"Multi-model pooled {ref_mode} residuals (standardized) (k={k})\n"
                f"models={len(per_model)}, KS={float(fit.get('ks_to_fit', float('nan'))):.3f}, "
                f"KL={float(fit.get('kl_hist_to_fit', float('nan'))):.3f}"
            )
            ax.set_xlabel("z = residual / sigma_i")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = os.path.join(out_dir, f"multi_model_margin_noise_hist_k{k}.png")
            fig.savefig(p)
            plt.close(fig)
            saved.append(p)

        if per_model:
            labels = [str(x.get("model_tag", "model")) for x in per_model]
            ks_vals = [float(((x.get("summary", {}) or {}).get("standardized_residual_fit", {}) or {}).get("ks_to_fit", float("nan"))) for x in per_model]
            kl_vals = [float(((x.get("summary", {}) or {}).get("standardized_residual_fit", {}) or {}).get("kl_hist_to_fit", float("nan"))) for x in per_model]
            if labels:
                fig = plt.figure(figsize=(max(8.0, 0.6 * len(labels)), 5.0), dpi=180)
                ax = fig.add_subplot(1, 1, 1)
                x = np.arange(len(labels))
                width = 0.38
                ax.bar(x - width / 2, ks_vals, width=width, color="#1f77b4", alpha=0.85, label="KS")
                ax.bar(x + width / 2, kl_vals, width=width, color="#ff7f0e", alpha=0.85, label="Hist KL")
                ax.set_title(f"Per-model Gaussian fit metrics ({mode_label}, k={k})")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(fontsize=8)
                fig.tight_layout()
                p = os.path.join(out_dir, f"multi_model_fit_metrics_k{k}.png")
                fig.savefig(p)
                plt.close(fig)
                saved.append(p)

        if t_residuals:
            ts = sorted(int(t) for t in t_residuals.keys())
            emp_std = []
            for t in ts:
                arr = np.asarray(t_residuals.get(t, []), dtype=np.float64)
                emp_std.append(float(np.std(arr)) if arr.size > 0 else float("nan"))
            if emp_std:
                base_std = emp_std[0] if np.isfinite(emp_std[0]) else float("nan")
                sqrt_target = [float(base_std / math.sqrt(float(t))) if np.isfinite(base_std) else float("nan") for t in ts]
                n_views = int(rec.get("n_views", max(ts) if ts else 0))
                finite_target = [
                    float(base_std * math.sqrt(float(max(0, n_views - int(t))) / float(max(1, int(t)) * max(1, n_views - 1))))
                    if np.isfinite(base_std) and n_views > 1
                    else float("nan")
                    for t in ts
                ]
                fig = plt.figure(figsize=(6.8, 4.8), dpi=180)
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(ts, emp_std, "-o", color="#1f77b4", lw=2.0, ms=5, label="Pooled empirical std")
                ax.plot(ts, sqrt_target, "--s", color="#d62728", lw=1.6, ms=4, label="1/sqrt(T)")
                if n_views > 1:
                    ax.plot(ts, finite_target, ":^", color="#2ca02c", lw=1.6, ms=4, label="Finite-view target")
                ax.set_title(f"Multi-model pooled T-scaling ({ref_mode}, k={k})")
                ax.set_xlabel("T = #views averaged")
                ax.set_ylabel("Residual std")
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8)
                fig.tight_layout()
                p = os.path.join(out_dir, f"multi_model_t_scaling_k{k}.png")
                fig.savefig(p)
                plt.close(fig)
                saved.append(p)

    return saved


def _run_multi_results_analysis(
    *,
    results_dirs: List[str],
    subjects: Optional[List[str]],
    n_runs: int,
    max_samples: int,
    aggregate_out_dir: str,
    save_plots: bool,
    use_full_reference: bool,
    combo_sample_limit: int,
    negative_tail_z_cutoff: float,
    right_tail_z_cutoff: float,
) -> None:
    valid_dirs = [str(x) for x in results_dirs if str(x).strip()]
    if not valid_dirs:
        raise SystemExit("No valid results_dirs were provided.")

    combined_by_k: Dict[int, Dict[str, object]] = {}
    for results_dir in valid_dirs:
        cache_files = _discover_cache_files(results_dir, subjects, int(n_runs))
        if not cache_files:
            print(f"[warn] no cache files found under: {results_dir}")
            continue
        model_tag = _guess_model_tag_from_results_dir(results_dir)
        model_buckets: Dict[int, Dict[str, object]] = {}
        model_n_views: Dict[int, int] = {}

        for ci in cache_files:
            results = _read_results_file(ci.path)
            if int(max_samples) > 0:
                results = results[: int(max_samples)]
            if not results:
                continue

            data0 = results[0].get("data", {}) or {}
            options = data0.get("options", None)
            if not isinstance(options, list) or len(options) == 0:
                continue
            k = len(options)
            probs0 = data0.get("probs", None)
            if not isinstance(probs0, list):
                continue
            perm_count = len(probs0)
            perm_list, _ = _infer_perm_list(k, perm_count)
            option_ids = list("ABCDE" if k == 5 else "ABCD") if k in (4, 5) else [str(i) for i in range(k)]
            summary, _, payload = _analyze_cyclic_margin_noise(
                results=results,
                perm_list=perm_list,
                option_ids=option_ids,
                subject=ci.subject,
                run_idx=int(ci.run_idx),
                use_full_reference=bool(use_full_reference),
                combo_sample_limit=int(combo_sample_limit),
                negative_tail_z_cutoff=float(negative_tail_z_cutoff),
                right_tail_z_cutoff=float(right_tail_z_cutoff),
            )
            bucket = model_buckets.setdefault(
                int(k),
                {
                    "reference_mode": str(summary.get("reference_mode", "cyclic")),
                    **_make_margin_noise_bucket(with_correctness=True),
                },
            )
            _merge_margin_noise_payload_into_bucket(bucket, payload)
            model_n_views[int(k)] = int(summary.get("n_views", k))

        for k, bucket in model_buckets.items():
            n_views = int(model_n_views.get(int(k), int(k)))
            model_summary = _summarize_margin_noise_bucket(
                bucket,
                n_views=n_views,
                reference_mode=str(bucket.get("reference_mode", "cyclic")),
                negative_tail_z_cutoff=float(negative_tail_z_cutoff),
                right_tail_z_cutoff=float(right_tail_z_cutoff),
            )
            combined = combined_by_k.setdefault(
                int(k),
                {
                    "reference_mode": str(model_summary.get("reference_mode", "cyclic")),
                    "per_model": [],
                    "pooled_bucket": {
                        **_make_margin_noise_bucket(with_correctness=True),
                    },
                    "n_views": n_views,
                },
            )
            combined["per_model"].append({
                "model_tag": str(model_tag),
                "results_dir": str(results_dir),
                "summary": model_summary,
            })
            _merge_margin_noise_payload_into_bucket(combined["pooled_bucket"], bucket)

    if not combined_by_k:
        raise SystemExit("No valid multi-model records found.")

    output: Dict[str, object] = {"results_dirs": valid_dirs, "by_k": {}}
    mode_name = "Full-Permutation" if use_full_reference else "Cyclic"
    print(f"==== Multi-Model {mode_name} Margin Noise Analysis ====")
    for k in sorted(combined_by_k.keys()):
        rec = combined_by_k[int(k)]
        pooled_summary = _summarize_margin_noise_bucket(
            rec["pooled_bucket"],
            n_views=int(rec.get("n_views", int(k))),
            reference_mode=str(rec.get("reference_mode", "cyclic")),
            negative_tail_z_cutoff=float(negative_tail_z_cutoff),
            right_tail_z_cutoff=float(right_tail_z_cutoff),
        )
        rec["pooled_summary"] = pooled_summary
        per_model = rec.get("per_model", []) or []

        def _macro_metric(path: Sequence[str]) -> float:
            vals = []
            for item in per_model:
                cur = item.get("summary", {}) or {}
                ok = True
                for key in path:
                    if not isinstance(cur, dict) or key not in cur:
                        ok = False
                        break
                    cur = cur[key]
                if ok and isinstance(cur, (int, float)) and np.isfinite(float(cur)):
                    vals.append(float(cur))
            return float(np.mean(vals)) if vals else float("nan")

        macro_average = {
            "ks_to_fit": _macro_metric(["standardized_residual_fit", "ks_to_fit"]),
            "kl_hist_to_fit": _macro_metric(["standardized_residual_fit", "kl_hist_to_fit"]),
            "skew": _macro_metric(["standardized_residual_fit", "skew"]),
            "excess_kurtosis": _macro_metric(["standardized_residual_fit", "excess_kurtosis"]),
            "mean_sigma_i": _macro_metric(["mean_sample_sigma_i"]),
        }
        output["by_k"][str(int(k))] = {
            "n_models": int(len(per_model)),
            "n_views": int(rec.get("n_views", int(k))),
            "reference_mode": str((pooled_summary or {}).get("reference_mode", "cyclic")),
            "macro_average": macro_average,
            "pooled_summary": pooled_summary,
            "per_model": per_model,
        }

        pooled_fit = pooled_summary.get("standardized_residual_fit", {}) or {}
        raw_fit = pooled_summary.get("pooled_residual_fit", {}) or {}
        raw_laplace_fit = pooled_summary.get("pooled_residual_laplace_fit", {}) or {}
        raw_cauchy_fit = pooled_summary.get("pooled_residual_cauchy_fit", {}) or {}
        print(f"--- k={int(k)} ---")
        print(
            "models={}, pooled KS={:.4f}, pooled KL={:.4f}, pooled skew={:.4f}, pooled kurtosis={:.4f}".format(
                len(per_model),
                float(pooled_fit.get("ks_to_fit", float("nan"))),
                float(pooled_fit.get("kl_hist_to_fit", float("nan"))),
                float(pooled_fit.get("skew", float("nan"))),
                float(pooled_fit.get("excess_kurtosis", float("nan"))),
            )
        )
        print(
            "macro-average KS={:.4f}, KL={:.4f}, skew={:.4f}, kurtosis={:.4f}, mean_sigma_i={:.4f}".format(
                float(macro_average.get("ks_to_fit", float("nan"))),
                float(macro_average.get("kl_hist_to_fit", float("nan"))),
                float(macro_average.get("skew", float("nan"))),
                float(macro_average.get("excess_kurtosis", float("nan"))),
                float(macro_average.get("mean_sigma_i", float("nan"))),
            )
        )
        print(
            "raw pooled fits: Gaussian KS={:.4f}, KL={:.4f} | Laplace KS={:.4f}, KL={:.4f} | Cauchy KS={:.4f}, KL={:.4f}".format(
                float(raw_fit.get("ks_to_fit", float("nan"))),
                float(raw_fit.get("kl_hist_to_fit", float("nan"))),
                float(raw_laplace_fit.get("ks_to_fit", float("nan"))),
                float(raw_laplace_fit.get("kl_hist_to_fit", float("nan"))),
                float(raw_cauchy_fit.get("ks_to_fit", float("nan"))),
                float(raw_cauchy_fit.get("kl_hist_to_fit", float("nan"))),
            )
        )
        raw_slot_fits = (pooled_summary or {}).get("raw_correct_slot_fits", []) or []
        if raw_slot_fits:
            print("raw correct-slot fits:", _format_group_fit_reports(raw_slot_fits, top_n=8))
        raw_top1_fits = (pooled_summary or {}).get("raw_top1_slot_fits", []) or []
        if raw_top1_fits:
            print("raw top1-slot fits:", _format_group_fit_reports(raw_top1_fits, top_n=8))
        raw_top2_fits = (pooled_summary or {}).get("raw_top2_slot_fits", []) or []
        if raw_top2_fits:
            print("raw top2-slot fits:", _format_group_fit_reports(raw_top2_fits, top_n=8))
        peak_info = (pooled_summary or {}).get("peak_bin_summary", []) or []
        if peak_info:
            top_bin = peak_info[0]
            top_labels = top_bin.get("top_perm_labels", [])[:5]
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in top_labels) if top_labels else ""
            print(
                "top standardized bin [{:.2f}, {:.2f}] count={} | top perms: {}".format(
                    float(top_bin.get("range_left", float("nan"))),
                    float(top_bin.get("range_right", float("nan"))),
                    int(top_bin.get("count", 0)),
                    labels_str,
                )
            )
        top_bin_acc = (pooled_summary or {}).get("top_standardized_bin_accuracy", {}) or {}
        if top_bin_acc:
            print(
                "top standardized bin acc/rstd: {:.4f}/{:.4f} ({}/{})".format(
                    float(top_bin_acc.get("accuracy", float("nan"))),
                    float(top_bin_acc.get("recall_std", float("nan"))),
                    int(top_bin_acc.get("correct", 0)),
                    int(top_bin_acc.get("count", 0)),
                )
            )
        top_bin_perm_acc = (pooled_summary or {}).get("top_standardized_bin_perm_accuracy", []) or []
        if top_bin_perm_acc:
            print("top-standardized-bin perm acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top_bin_perm_acc[:5]))
        top_bin_ideals = (pooled_summary or {}).get("top_standardized_bin_ideal_counts", []) or []
        if top_bin_ideals:
            print("top standardized bin by ideal:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in top_bin_ideals[:5]))
        top_bin_slots = (pooled_summary or {}).get("top_standardized_bin_correct_slot_counts", []) or []
        if top_bin_slots:
            print("top standardized bin by correct slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in top_bin_slots[:5]))
        top_bin_slot_inc = (pooled_summary or {}).get("top_standardized_bin_slot_incidence", []) or []
        if top_bin_slot_inc:
            print("top-standardized-bin slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in top_bin_slot_inc[:5]))
        top_bin_slot_acc = (pooled_summary or {}).get("top_standardized_bin_correct_slot_accuracy", []) or []
        if top_bin_slot_acc:
            print("top-standardized-bin slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top_bin_slot_acc[:5]))
        top_bin_top1 = (pooled_summary or {}).get("top_standardized_bin_top1_slot_counts", []) or []
        if top_bin_top1:
            print("top standardized bin by top1 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in top_bin_top1[:5]))
        top_bin_top1_inc = (pooled_summary or {}).get("top_standardized_bin_top1_slot_incidence", []) or []
        if top_bin_top1_inc:
            print("top-standardized-bin top1-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in top_bin_top1_inc[:5]))
        top_bin_top1_acc = (pooled_summary or {}).get("top_standardized_bin_top1_slot_accuracy", []) or []
        if top_bin_top1_acc:
            print("top-standardized-bin top1-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top_bin_top1_acc[:5]))
        top_bin_top2 = (pooled_summary or {}).get("top_standardized_bin_top2_slot_counts", []) or []
        if top_bin_top2:
            print("top standardized bin by top2 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in top_bin_top2[:5]))
        top_bin_top2_inc = (pooled_summary or {}).get("top_standardized_bin_top2_slot_incidence", []) or []
        if top_bin_top2_inc:
            print("top-standardized-bin top2-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in top_bin_top2_inc[:5]))
        top_bin_top2_acc = (pooled_summary or {}).get("top_standardized_bin_top2_slot_accuracy", []) or []
        if top_bin_top2_acc:
            print("top-standardized-bin top2-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top_bin_top2_acc[:5]))
        neg_tail = (pooled_summary or {}).get("negative_tail_summary", {}) or {}
        neg_labels = (neg_tail.get("top_perm_labels", []) or [])[:5]
        if neg_labels:
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_labels)
            print(
                "negative tail z<={:.2f} count={} | top perms: {}".format(
                    float(neg_tail.get("z_cutoff", float("nan"))),
                    int(neg_tail.get("count", 0)),
                    labels_str,
                )
            )
        neg_tail_acc = (pooled_summary or {}).get("negative_tail_accuracy", {}) or {}
        if neg_tail_acc:
            print(
                "negative tail acc/rstd: {:.4f}/{:.4f} ({}/{})".format(
                    float(neg_tail_acc.get("accuracy", float("nan"))),
                    float(neg_tail_acc.get("recall_std", float("nan"))),
                    int(neg_tail_acc.get("correct", 0)),
                    int(neg_tail_acc.get("count", 0)),
                )
            )
        neg_tail_perm_acc = (pooled_summary or {}).get("negative_tail_perm_accuracy", []) or []
        if neg_tail_perm_acc:
            print("negative-tail perm acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in neg_tail_perm_acc[:5]))
        right_tail = (pooled_summary or {}).get("right_tail_summary", {}) or {}
        right_labels = (right_tail.get("top_perm_labels", []) or [])[:5]
        if right_labels:
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in right_labels)
            print(
                "right tail z>={:.2f} count={} | top perms: {}".format(
                    float(right_tail.get("z_cutoff", float("nan"))),
                    int(right_tail.get("count", 0)),
                    labels_str,
                )
            )
        right_tail_acc = (pooled_summary or {}).get("right_tail_accuracy", {}) or {}
        if right_tail_acc:
            print(
                "right tail acc/rstd: {:.4f}/{:.4f} ({}/{})".format(
                    float(right_tail_acc.get("accuracy", float("nan"))),
                    float(right_tail_acc.get("recall_std", float("nan"))),
                    int(right_tail_acc.get("correct", 0)),
                    int(right_tail_acc.get("count", 0)),
                )
            )
        right_tail_perm_acc = (pooled_summary or {}).get("right_tail_perm_accuracy", []) or []
        if right_tail_perm_acc:
            print("right-tail perm acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in right_tail_perm_acc[:5]))
        entry_acc = (pooled_summary or {}).get("entry_accuracy", {}) or {}
        if entry_acc:
            print(
                "overall view acc/rstd: {:.4f}/{:.4f} ({}/{})".format(
                    float(entry_acc.get("accuracy", float("nan"))),
                    float(entry_acc.get("recall_std", float("nan"))),
                    int(entry_acc.get("correct", 0)),
                    int(entry_acc.get("count", 0)),
                )
            )
        overall_perm_acc = (pooled_summary or {}).get("overall_perm_accuracy", []) or []
        if overall_perm_acc:
            print("overall perm acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in overall_perm_acc[:5]))
        neg_bins = (neg_tail.get("top_bins", []) or [])
        if neg_bins:
            top_bin = neg_bins[0]
            top_labels = (top_bin.get("top_perm_labels", []) or [])[:5]
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in top_labels) if top_labels else ""
            print(
                "top negative bin [{:.2f}, {:.2f}] count={} | top perms: {}".format(
                    float(top_bin.get("range_left", float("nan"))),
                    float(top_bin.get("range_right", float("nan"))),
                    int(top_bin.get("count", 0)),
                    labels_str,
                )
            )
        neg_ideals = (pooled_summary or {}).get("negative_tail_ideal_counts", []) or []
        if neg_ideals:
            print("negative tail by ideal:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_ideals[:5]))
        neg_slots = (pooled_summary or {}).get("negative_tail_correct_slot_counts", []) or []
        if neg_slots:
            print("negative tail by correct slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_slots[:5]))
        tail_slot_inc = (pooled_summary or {}).get("negative_tail_slot_incidence", []) or []
        if tail_slot_inc:
            print("negative-tail slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in tail_slot_inc[:5]))
        slot_acc = (pooled_summary or {}).get("correct_slot_accuracy", []) or []
        if slot_acc:
            print("correct slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in slot_acc[:5]))
        tail_slot_acc = (pooled_summary or {}).get("negative_tail_correct_slot_accuracy", []) or []
        if tail_slot_acc:
            print("negative-tail slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in tail_slot_acc[:5]))
        right_ideals = (pooled_summary or {}).get("right_tail_ideal_counts", []) or []
        if right_ideals:
            print("right tail by ideal:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in right_ideals[:5]))
        right_slots = (pooled_summary or {}).get("right_tail_correct_slot_counts", []) or []
        if right_slots:
            print("right tail by correct slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in right_slots[:5]))
        right_slot_inc = (pooled_summary or {}).get("right_tail_slot_incidence", []) or []
        if right_slot_inc:
            print("right-tail slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in right_slot_inc[:5]))
        right_slot_acc = (pooled_summary or {}).get("right_tail_correct_slot_accuracy", []) or []
        if right_slot_acc:
            print("right-tail slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in right_slot_acc[:5]))
        neg_top1_slots = (pooled_summary or {}).get("negative_tail_top1_slot_counts", []) or []
        if neg_top1_slots:
            print("negative tail by top1 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_top1_slots[:5]))
        tail_top1_inc = (pooled_summary or {}).get("negative_tail_top1_slot_incidence", []) or []
        if tail_top1_inc:
            print("negative-tail top1-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in tail_top1_inc[:5]))
        top1_acc = (pooled_summary or {}).get("top1_slot_accuracy", []) or []
        if top1_acc:
            print("top1-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top1_acc[:5]))
        tail_top1_acc = (pooled_summary or {}).get("negative_tail_top1_slot_accuracy", []) or []
        if tail_top1_acc:
            print("negative-tail top1-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in tail_top1_acc[:5]))
        right_top1_slots = (pooled_summary or {}).get("right_tail_top1_slot_counts", []) or []
        if right_top1_slots:
            print("right tail by top1 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in right_top1_slots[:5]))
        right_top1_inc = (pooled_summary or {}).get("right_tail_top1_slot_incidence", []) or []
        if right_top1_inc:
            print("right-tail top1-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in right_top1_inc[:5]))
        right_top1_acc = (pooled_summary or {}).get("right_tail_top1_slot_accuracy", []) or []
        if right_top1_acc:
            print("right-tail top1-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in right_top1_acc[:5]))
        neg_top2_slots = (pooled_summary or {}).get("negative_tail_top2_slot_counts", []) or []
        if neg_top2_slots:
            print("negative tail by top2 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_top2_slots[:5]))
        tail_top2_inc = (pooled_summary or {}).get("negative_tail_top2_slot_incidence", []) or []
        if tail_top2_inc:
            print("negative-tail top2-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in tail_top2_inc[:5]))
        top2_acc = (pooled_summary or {}).get("top2_slot_accuracy", []) or []
        if top2_acc:
            print("top2-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top2_acc[:5]))
        tail_top2_acc = (pooled_summary or {}).get("negative_tail_top2_slot_accuracy", []) or []
        if tail_top2_acc:
            print("negative-tail top2-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in tail_top2_acc[:5]))
        right_top2_slots = (pooled_summary or {}).get("right_tail_top2_slot_counts", []) or []
        if right_top2_slots:
            print("right tail by top2 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in right_top2_slots[:5]))
        right_top2_inc = (pooled_summary or {}).get("right_tail_top2_slot_incidence", []) or []
        if right_top2_inc:
            print("right-tail top2-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in right_top2_inc[:5]))
        right_top2_acc = (pooled_summary or {}).get("right_tail_top2_slot_accuracy", []) or []
        if right_top2_acc:
            print("right-tail top2-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in right_top2_acc[:5]))
        raw_top_bin = (pooled_summary or {}).get("raw_top_bin_summary", {}) or {}
        if raw_top_bin:
            raw_top_labels = (raw_top_bin.get("top_perm_labels", []) or [])[:5]
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in raw_top_labels) if raw_top_labels else ""
            print(
                "raw top bin [{:.4f}, {:.4f}] count={} | top perms: {}".format(
                    float(raw_top_bin.get("range_left", float("nan"))),
                    float(raw_top_bin.get("range_right", float("nan"))),
                    int(raw_top_bin.get("count", 0)),
                    labels_str,
                )
            )
        raw_top_subset = (pooled_summary or {}).get("raw_top_bin_subset", {}) or {}
        if raw_top_subset:
            print("raw top bin acc/rstd: {:.4f}/{:.4f} ({}/{})".format(float(raw_top_subset.get("accuracy", float("nan"))), float(raw_top_subset.get("recall_std", float("nan"))), int(raw_top_subset.get("correct", 0)), int(raw_top_subset.get("count", 0))))
            for key, title in (
                ("perm_accuracy", "raw-top-bin perm acc/rstd"),
                ("ideal_counts", "raw top bin by ideal"),
                ("correct_slot_counts", "raw top bin by correct slot"),
                ("slot_incidence", "raw-top-bin slot incidence"),
                ("correct_slot_accuracy", "raw-top-bin slot acc/rstd"),
                ("top1_slot_counts", "raw top bin by top1 slot"),
                ("top1_slot_incidence", "raw-top-bin top1-slot incidence"),
                ("top1_slot_accuracy", "raw-top-bin top1-slot acc/rstd"),
                ("top2_slot_counts", "raw top bin by top2 slot"),
                ("top2_slot_incidence", "raw-top-bin top2-slot incidence"),
                ("top2_slot_accuracy", "raw-top-bin top2-slot acc/rstd"),
            ):
                vals = raw_top_subset.get(key, []) or []
                if not vals:
                    continue
                if "acc/rstd" in title:
                    print(title + ":", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in vals[:5]))
                elif "incidence" in title:
                    print(title + ":", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in vals[:5]))
                else:
                    print(title + ":", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in vals[:5]))
        for raw_key, raw_title in (("raw_left_tail_subset", "raw left tail"), ("raw_right_tail_subset", "raw right tail")):
            raw_subset = (pooled_summary or {}).get(raw_key, {}) or {}
            if raw_subset:
                print(
                    f"{raw_title} residual {'<=' if 'left' in raw_key else '>='}{float(raw_subset.get('residual_cutoff', float('nan'))):.4f} acc/rstd: "
                    f"{float(raw_subset.get('accuracy', float('nan'))):.4f}/{float(raw_subset.get('recall_std', float('nan'))):.4f} "
                    f"({int(raw_subset.get('correct', 0))}/{int(raw_subset.get('count', 0))})"
                )
                for key, title in (
                    ("perm_accuracy", f"{raw_title} perm acc/rstd"),
                    ("ideal_counts", f"{raw_title} by ideal"),
                    ("correct_slot_counts", f"{raw_title} by correct slot"),
                    ("slot_incidence", f"{raw_title.replace(' ', '-')} slot incidence"),
                    ("correct_slot_accuracy", f"{raw_title.replace(' ', '-')} slot acc/rstd"),
                    ("top1_slot_counts", f"{raw_title} by top1 slot"),
                    ("top1_slot_incidence", f"{raw_title.replace(' ', '-')} top1-slot incidence"),
                    ("top1_slot_accuracy", f"{raw_title.replace(' ', '-')} top1-slot acc/rstd"),
                    ("top2_slot_counts", f"{raw_title} by top2 slot"),
                    ("top2_slot_incidence", f"{raw_title.replace(' ', '-')} top2-slot incidence"),
                    ("top2_slot_accuracy", f"{raw_title.replace(' ', '-')} top2-slot acc/rstd"),
                ):
                    vals = raw_subset.get(key, []) or []
                    if not vals:
                        continue
                    if "acc/rstd" in title:
                        print(title + ":", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in vals[:5]))
                    elif "incidence" in title:
                        print(title + ":", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in vals[:5]))
                    else:
                        print(title + ":", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in vals[:5]))
        corr_split = (pooled_summary or {}).get("correctness_split", {}) or {}
        for split_name in ("correct", "incorrect"):
            split = corr_split.get(split_name, {}) or {}
            if not split:
                continue
            neg = (split.get("negative_tail_summary", {}) or {})
            fit = (split.get("standardized_residual_fit", {}) or {})
            print(
                "{} refs: n_samples={}, mean_sigma_i={:.4f}, z_skew={:.4f}, left_tail_count={}".format(
                    split_name,
                    int(split.get("n_samples", 0)),
                    float(split.get("mean_sample_sigma_i", float("nan"))),
                    float(fit.get("skew", float("nan"))),
                    int(neg.get("count", 0)),
                )
            )
        print("")

    out_dir = str(aggregate_out_dir).strip() or str(valid_dirs[0])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multi_model_perm_margin_noise_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

    if save_plots:
        saved = _save_multi_model_margin_plots(combined_by_k=combined_by_k, out_dir=out_dir)
        for p in saved:
            print(f"Saved: {p}")


def _read_results_file(file_path: str) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    lines = [e for e in lines if e.get("type") == "result"]
    lines = sorted(lines, key=lambda x: int(x["data"]["idx"]))
    return lines


@dataclass(frozen=True)
class CacheInfo:
    subject: str
    run_idx: int
    path: str


def _discover_cache_files(cache_dir: str, subjects: Optional[List[str]], n_runs: int) -> List[CacheInfo]:
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"cache_dir not found: {cache_dir}")

    out: List[CacheInfo] = []
    if subjects:
        for subj in subjects:
            for r in range(max(1, n_runs)):
                p = os.path.join(cache_dir, f"{subj}_run{r}.jsonl") if n_runs > 1 else os.path.join(cache_dir, f"{subj}.jsonl")
                if os.path.exists(p):
                    out.append(CacheInfo(subject=subj, run_idx=r, path=p))
    else:
        # Infer subjects by listing jsonl files in cache_dir.
        for fn in sorted(os.listdir(cache_dir)):
            if not fn.endswith(".jsonl"):
                continue
            if fn.endswith("_curve.jsonl") or fn.endswith("_pride_curve.jsonl"):
                continue
            path = os.path.join(cache_dir, fn)
            # Use splitext so names like `csqa_run0.jsonl` become `csqa_run0`, not `csqa_run0.`
            base = os.path.splitext(fn)[0]
            if base.endswith("_run0") or "_run" in base:
                # subject_run{i}
                try:
                    subj, run_s = base.rsplit("_run", 1)
                    run_idx = int(run_s)
                except Exception:
                    continue
                out.append(CacheInfo(subject=subj, run_idx=run_idx, path=path))
            else:
                out.append(CacheInfo(subject=base, run_idx=0, path=path))

    # If n_runs>1, keep only run indices < n_runs when provided.
    if n_runs > 1:
        out = [x for x in out if 0 <= int(x.run_idx) < int(n_runs)]

    # Stable ordering: subject then run
    out = sorted(out, key=lambda x: (x.subject, x.run_idx))
    return out


def _compute_eval_save_path(eval_name: str, pretrained_model_path: str, option_id_set: Optional[str]) -> str:
    """
    Mirror `code/eval_clm_utils.py::prepare_eval` save_path logic.
    NOTE: This path is relative to the working directory where eval_clm.py is executed (typically `code/`).
    """
    eval_args = str(eval_name).split(",")
    task = str(eval_args[0]).strip()
    num_few_shot = int(eval_args[1])
    setting = str(eval_args[2]).strip() if len(eval_args) > 2 and str(eval_args[2]).strip() else None
    model_name = str(pretrained_model_path).split("/")[-1]

    save_path = f"results_{task}/{num_few_shot}s_{model_name}/{task}"
    if setting is not None:
        save_path += f"_{setting}"
    if option_id_set:
        save_path += f"_id-{option_id_set}"
    return save_path


def _infer_perm_list(k: int, perm_count: int) -> Tuple[List[Tuple[int, ...]], bool]:
    """
    Returns (perm_list, full_enabled).
    - If perm_count == factorial(k): full permutations
    - Else if perm_count == k: cyclic rotations
    - Else: fall back to "first perm_count permutations in sorted(permutations(range(k)))"
    """
    if perm_count == math.factorial(k):
        return list(sorted(permutations(range(k)))), True
    if perm_count == k:
        return _rotations(k), False
    # Unknown: try full-perm prefix (still lets analysis run)
    full = list(sorted(permutations(range(k))))
    if perm_count <= len(full):
        return full[:perm_count], False
    raise ValueError(f"Unsupported perm_count={perm_count} for k={k}")


def _run_analysis(
    *,
    cache_dir: str,
    subjects: Optional[List[str]],
    n_runs: int,
    seed: int,
    max_samples: int,
    subset_trials: int,
    subset_sizes: str,
    max_perms_corr: int,
    save_plots: bool,
    wandb_enabled: bool,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    wandb_run_name: Optional[str],
    wandb_tags: Optional[str],
    wandb_group: Optional[str],
    wandb_job_type: str,
    task_name: Optional[str],
    save_margin_samples: bool,
    use_full_reference: bool,
    combo_sample_limit: int,
    negative_tail_z_cutoff: float,
    right_tail_z_cutoff: float,
) -> None:
    subjects_list = [s.strip() for s in str(subjects or []) if str(s).strip()] if subjects else None
    cache_files = _discover_cache_files(str(cache_dir), subjects_list, int(n_runs))
    if not cache_files:
        raise SystemExit(f"No cache files found under: {cache_dir}")

    per_record_reports = []
    # For correlation heatmaps: accumulate correctness matrices by k when permutation ordering matches.
    corr_by_k: Dict[int, Dict[str, object]] = {}
    # For recall-by-option bar plot across all subjects: accumulate y_true/preds per k
    recall_by_k: Dict[int, Dict[str, List[str]]] = {}
    # For margin-noise / Gaussianity analysis
    margin_noise_by_k: Dict[int, Dict[str, object]] = {}
    margin_noise_sample_records: List[dict] = []

    for ci in cache_files:
        results = _read_results_file(ci.path)
        if int(max_samples) > 0:
            results = results[: int(max_samples)]
        if not results:
            continue

        data0 = results[0].get("data", {}) or {}
        options = data0.get("options", None)
        if not isinstance(options, list) or len(options) == 0:
            raise ValueError(f"Missing options in {ci.path}")
        k = len(options)
        option_ids = list("ABCDE" if k == 5 else "ABCD") if k in (4, 5) else [str(i) for i in range(k)]

        probs0 = data0.get("probs", None)
        if not isinstance(probs0, list):
            raise ValueError(f"Missing probs in {ci.path}")
        perm_count = len(probs0)
        perm_list, full_enabled = _infer_perm_list(k, perm_count)
        identity_idx = perm_list.index(tuple(range(k))) if tuple(range(k)) in perm_list else 0

        perm_accs, correct_mat = _perm_accs_and_correct_matrix(results, perm_list, option_ids)

        cyc_perms = _rotations(k)
        cyc_idxs = [perm_list.index(p) for p in cyc_perms if p in perm_list]
        non_id_cyc_idxs = [i for i in cyc_idxs if i != identity_idx]
        non_id_full_idxs = [i for i in range(len(perm_list)) if i != identity_idx]

        orig_acc = float(perm_accs[identity_idx]) if 0 <= identity_idx < len(perm_accs) else float("nan")

        def _mean_var(x: np.ndarray) -> Tuple[float, float]:
            x = np.asarray(x, dtype=np.float64)
            if x.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(x)), float(np.var(x))

        cyc_nonid_accs = perm_accs[non_id_cyc_idxs] if non_id_cyc_idxs else np.asarray([], dtype=np.float64)
        full_nonid_accs = perm_accs[non_id_full_idxs] if non_id_full_idxs else np.asarray([], dtype=np.float64)

        cyc_mean, cyc_var = _mean_var(cyc_nonid_accs)
        full_mean, full_var = _mean_var(full_nonid_accs)

        corr_cyc = _pairwise_corr_mean(correct_mat[cyc_idxs], int(max_perms_corr), int(seed)) if len(cyc_idxs) > 1 else float("nan")
        corr_full = _pairwise_corr_mean(correct_mat, int(max_perms_corr), int(seed))

        sizes = []
        for t in str(subset_sizes).split(","):
            t = t.strip()
            if not t:
                continue
            try:
                sizes.append(int(t))
            except Exception:
                pass
        sizes = sorted(set(sizes))
        subset_curve = _ensemble_acc_subset_curve(
            results,
            perm_list,
            option_ids,
            sizes=sizes,
            n_trials=int(subset_trials),
            seed=int(seed) + 1000 * int(ci.run_idx),
        )

        corrects_full = 0
        corrects_cyc = 0
        for r in results:
            d = r.get("data", {}) or {}
            probs_seq = d.get("probs", None)
            if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                continue
            ideal = str(d.get("ideal"))
            agg_all = _aggregate_probs_over_permutations(probs_seq, perm_list, len(option_ids))
            pred_all = option_ids[int(np.argmax(agg_all))]
            corrects_full += 1 if pred_all == ideal else 0
            if cyc_idxs:
                probs_c = [probs_seq[i] for i in cyc_idxs]
                perms_c = [perm_list[i] for i in cyc_idxs]
                agg_c = _aggregate_probs_over_permutations(probs_c, perms_c, len(option_ids))
                pred_c = option_ids[int(np.argmax(agg_c))]
                corrects_cyc += 1 if pred_c == ideal else 0
        ens_full_acc = corrects_full / float(len(results)) if results else float("nan")
        ens_cyc_acc = corrects_cyc / float(len(results)) if results else float("nan")
        margin_noise_summary, sample_noise_records, pooled_noise_payload = _analyze_cyclic_margin_noise(
            results=results,
            perm_list=perm_list,
            option_ids=option_ids,
            subject=ci.subject,
            run_idx=int(ci.run_idx),
            use_full_reference=bool(use_full_reference),
            combo_sample_limit=int(combo_sample_limit),
            combo_seed=int(seed),
            negative_tail_z_cutoff=float(negative_tail_z_cutoff),
            right_tail_z_cutoff=float(right_tail_z_cutoff),
        )

        per_record_reports.append({
            "subject": ci.subject,
            "run": int(ci.run_idx),
            "k": int(k),
            "perm_count": int(len(perm_list)),
            "full_enabled": bool(full_enabled),
            "orig_acc": float(orig_acc),
            "cyc_nonid_mean_acc": float(cyc_mean),
            "cyc_nonid_var_acc": float(cyc_var),
            "full_nonid_mean_acc": float(full_mean),
            "full_nonid_var_acc": float(full_var),
            "corr_cyclic_mean": float(corr_cyc),
            "corr_full_mean": float(corr_full),
            "ens_acc_all_perms": float(ens_full_acc),
            "ens_acc_cyclic_perms": float(ens_cyc_acc),
            "subset_curve": subset_curve,
            "cyclic_margin_noise": margin_noise_summary,
        })

        bucket = margin_noise_by_k.setdefault(
            int(k),
            {
                "reference_mode": str(margin_noise_summary.get("reference_mode", "cyclic")),
                "n_views": int(margin_noise_summary.get("n_views", k)),
                **_make_margin_noise_bucket(with_correctness=True),
            },
        )
        _merge_margin_noise_payload_into_bucket(bucket, pooled_noise_payload)
        if save_margin_samples and sample_noise_records:
            margin_noise_sample_records.extend(sample_noise_records)

        # Accumulate per-sample predictions for recall-by-option plot (all subjects combined)
        try:
            y_true_list = []
            y_orig_list = []
            y_cyc_list = []
            y_full_list = []

            # Precompute indices
            identity_perm = tuple(range(int(k)))
            identity_idx = perm_list.index(identity_perm) if identity_perm in perm_list else 0
            cyc_perms = _rotations(int(k))
            cyc_idxs = [perm_list.index(p) for p in cyc_perms if p in perm_list]
            cyc_perm_list = [perm_list[i] for i in cyc_idxs]

            for r in results:
                d = r.get("data", {}) or {}
                probs_seq = d.get("probs", None)
                if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                    continue
                ideal = str(d.get("ideal"))
                if ideal not in option_ids:
                    continue

                # Original (identity only)
                agg_o = _aggregate_probs_over_permutations([probs_seq[identity_idx]], [perm_list[identity_idx]], int(k))
                pred_o = option_ids[int(np.argmax(agg_o))]

                # Cyclic ensemble (all rotations available)
                if cyc_idxs:
                    probs_c = [probs_seq[i] for i in cyc_idxs]
                    agg_c = _aggregate_probs_over_permutations(probs_c, cyc_perm_list, int(k))
                    pred_c = option_ids[int(np.argmax(agg_c))]
                else:
                    pred_c = pred_o

                # Full/available ensemble (all perms in cache)
                agg_f = _aggregate_probs_over_permutations(probs_seq, perm_list, int(k))
                pred_f = option_ids[int(np.argmax(agg_f))]

                y_true_list.append(ideal)
                y_orig_list.append(pred_o)
                y_cyc_list.append(pred_c)
                y_full_list.append(pred_f)

            if y_true_list:
                if int(k) not in recall_by_k:
                    recall_by_k[int(k)] = {"y_true": [], "orig": [], "cyclic": [], "full": []}
                recall_by_k[int(k)]["y_true"].extend(y_true_list)
                recall_by_k[int(k)]["orig"].extend(y_orig_list)
                recall_by_k[int(k)]["cyclic"].extend(y_cyc_list)
                recall_by_k[int(k)]["full"].extend(y_full_list)
        except Exception:
            pass

        # Store correctness matrix for heatmap (only if we can keep a consistent perm order for this k)
        if (save_plots or wandb_enabled) and len(perm_list) > 1:
            if int(k) not in corr_by_k:
                corr_by_k[int(k)] = {"perm_list": perm_list, "mats": [correct_mat]}
            else:
                ref_perm_list = corr_by_k[int(k)].get("perm_list")
                if isinstance(ref_perm_list, list) and ref_perm_list == perm_list:
                    mats = corr_by_k[int(k)].get("mats")
                    if isinstance(mats, list):
                        mats.append(correct_mat)

    if not per_record_reports:
        raise SystemExit("No valid records after reading cache files.")

    print("==== Permutation Noise Analysis ====")
    print(f"cache_dir: {cache_dir}")
    print(f"records: {len(per_record_reports)} (subject-run)")
    print("")

    for k in sorted(set(int(r["k"]) for r in per_record_reports)):
        rs = [r for r in per_record_reports if int(r["k"]) == k]
        print(f"--- k={k} (records={len(rs)}) ---")
        for name, key in [
            ("Original acc (identity)", "orig_acc"),
            ("Cyclic ensemble acc (all rotations)", "ens_acc_cyclic_perms"),
            ("Full/available ensemble acc (all perms in cache)", "ens_acc_all_perms"),
            ("Mean corr (cyclic perms)", "corr_cyclic_mean"),
            ("Mean corr (all perms)", "corr_full_mean"),
            ("Mean acc over non-identity cyclic perms", "cyc_nonid_mean_acc"),
            ("Var  acc over non-identity cyclic perms", "cyc_nonid_var_acc"),
            ("Mean acc over non-identity perms (full if available)", "full_nonid_mean_acc"),
            ("Var  acc over non-identity perms (full if available)", "full_nonid_var_acc"),
        ]:
            vals = [x[key] for x in rs if np.isfinite(x.get(key, float("nan")))]
            if not vals:
                continue
            m = float(np.mean(vals))
            s = float(np.std(vals)) if len(vals) > 1 else 0.0
            print(f"{name}: {m:.4f} ± {s:.4f}")
        print("")

    mode_name = "Full-Permutation" if use_full_reference else "Cyclic"
    print(f"==== {mode_name} Margin Noise Analysis ====")
    for k in sorted(set(int(r["k"]) for r in per_record_reports)):
        rs = [r for r in per_record_reports if int(r["k"]) == k]
        margin_summaries = [r.get("cyclic_margin_noise", {}) for r in rs if isinstance(r.get("cyclic_margin_noise"), dict)]
        if not margin_summaries:
            continue
        print(f"--- k={k} (records={len(margin_summaries)}) ---")

        def _mean_nested(path: Sequence[str]) -> float:
            vals = []
            for rec in margin_summaries:
                cur = rec
                ok = True
                for key in path:
                    if not isinstance(cur, dict) or key not in cur:
                        ok = False
                        break
                    cur = cur[key]
                if ok and isinstance(cur, (int, float)) and np.isfinite(float(cur)):
                    vals.append(float(cur))
            return float(np.mean(vals)) if vals else float("nan")

        print(
            "ref_margin(mean)={:.4f}, sigma_i(mean)={:.4f}, corr(ref_margin,sigma_i)={:.4f}, corr(base_gap,sigma_i)={:.4f}".format(
                _mean_nested(["mean_ref_margin"]),
                _mean_nested(["mean_sigma_i"]),
                _mean_nested(["corr_ref_margin_sigma"]),
                _mean_nested(["corr_base_gap_sigma"]),
            )
        )
        print(
            "residual fit: std={:.4f}, ks={:.4f}, kl={:.4f}".format(
                _mean_nested(["pooled_residual_fit", "std"]),
                _mean_nested(["pooled_residual_fit", "ks_to_fit"]),
                _mean_nested(["pooled_residual_fit", "kl_hist_to_fit"]),
            )
        )
        print(
            "raw alt fits: Laplace KS={:.4f}, KL={:.4f} | Cauchy KS={:.4f}, KL={:.4f}".format(
                _mean_nested(["pooled_residual_laplace_fit", "ks_to_fit"]),
                _mean_nested(["pooled_residual_laplace_fit", "kl_hist_to_fit"]),
                _mean_nested(["pooled_residual_cauchy_fit", "ks_to_fit"]),
                _mean_nested(["pooled_residual_cauchy_fit", "kl_hist_to_fit"]),
            )
        )
        raw_slot_fits = next((rec.get("raw_correct_slot_fits", []) for rec in margin_summaries if rec.get("raw_correct_slot_fits")), [])
        if raw_slot_fits:
            print("raw correct-slot fits:", _format_group_fit_reports(raw_slot_fits, top_n=8))
        raw_top1_fits = next((rec.get("raw_top1_slot_fits", []) for rec in margin_summaries if rec.get("raw_top1_slot_fits")), [])
        if raw_top1_fits:
            print("raw top1-slot fits:", _format_group_fit_reports(raw_top1_fits, top_n=8))
        raw_top2_fits = next((rec.get("raw_top2_slot_fits", []) for rec in margin_summaries if rec.get("raw_top2_slot_fits")), [])
        if raw_top2_fits:
            print("raw top2-slot fits:", _format_group_fit_reports(raw_top2_fits, top_n=8))
        print(
            "standardized fit: mean={:.4f}, std={:.4f}, ks={:.4f}, kl={:.4f}".format(
                _mean_nested(["standardized_residual_fit", "mean"]),
                _mean_nested(["standardized_residual_fit", "std"]),
                _mean_nested(["standardized_residual_fit", "ks_to_fit"]),
                _mean_nested(["standardized_residual_fit", "kl_hist_to_fit"]),
            )
        )
        peak_info = next((rec.get("peak_bin_summary", []) for rec in margin_summaries if rec.get("peak_bin_summary")), [])
        if peak_info:
            top_bin = peak_info[0]
            top_labels = top_bin.get("top_perm_labels", [])[:3]
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in top_labels) if top_labels else ""
            print(
                "top standardized bin [{:.2f}, {:.2f}] count={} | top perms: {}".format(
                    float(top_bin.get("range_left", float("nan"))),
                    float(top_bin.get("range_right", float("nan"))),
                    int(top_bin.get("count", 0)),
                    labels_str,
                )
            )
        top_bin_acc = next((rec.get("top_standardized_bin_accuracy", {}) for rec in margin_summaries if rec.get("top_standardized_bin_accuracy")), {})
        if top_bin_acc:
            print(
                "top standardized bin acc/rstd: {:.4f}/{:.4f} ({}/{})".format(
                    float(top_bin_acc.get("accuracy", float("nan"))),
                    float(top_bin_acc.get("recall_std", float("nan"))),
                    int(top_bin_acc.get("correct", 0)),
                    int(top_bin_acc.get("count", 0)),
                )
            )
        top_bin_perm_acc = next((rec.get("top_standardized_bin_perm_accuracy", []) for rec in margin_summaries if rec.get("top_standardized_bin_perm_accuracy")), [])
        if top_bin_perm_acc:
            print("top-standardized-bin perm acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top_bin_perm_acc[:3]))
        top_bin_ideals = next((rec.get("top_standardized_bin_ideal_counts", []) for rec in margin_summaries if rec.get("top_standardized_bin_ideal_counts")), [])
        if top_bin_ideals:
            print("top standardized bin by ideal:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in top_bin_ideals[:3]))
        top_bin_slots = next((rec.get("top_standardized_bin_correct_slot_counts", []) for rec in margin_summaries if rec.get("top_standardized_bin_correct_slot_counts")), [])
        if top_bin_slots:
            print("top standardized bin by correct slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in top_bin_slots[:3]))
        top_bin_slot_inc = next((rec.get("top_standardized_bin_slot_incidence", []) for rec in margin_summaries if rec.get("top_standardized_bin_slot_incidence")), [])
        if top_bin_slot_inc:
            print("top-standardized-bin slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in top_bin_slot_inc[:3]))
        top_bin_slot_acc = next((rec.get("top_standardized_bin_correct_slot_accuracy", []) for rec in margin_summaries if rec.get("top_standardized_bin_correct_slot_accuracy")), [])
        if top_bin_slot_acc:
            print("top-standardized-bin slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top_bin_slot_acc[:3]))
        top_bin_top1 = next((rec.get("top_standardized_bin_top1_slot_counts", []) for rec in margin_summaries if rec.get("top_standardized_bin_top1_slot_counts")), [])
        if top_bin_top1:
            print("top standardized bin by top1 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in top_bin_top1[:3]))
        top_bin_top1_inc = next((rec.get("top_standardized_bin_top1_slot_incidence", []) for rec in margin_summaries if rec.get("top_standardized_bin_top1_slot_incidence")), [])
        if top_bin_top1_inc:
            print("top-standardized-bin top1-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in top_bin_top1_inc[:3]))
        top_bin_top1_acc = next((rec.get("top_standardized_bin_top1_slot_accuracy", []) for rec in margin_summaries if rec.get("top_standardized_bin_top1_slot_accuracy")), [])
        if top_bin_top1_acc:
            print("top-standardized-bin top1-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top_bin_top1_acc[:3]))
        top_bin_top2 = next((rec.get("top_standardized_bin_top2_slot_counts", []) for rec in margin_summaries if rec.get("top_standardized_bin_top2_slot_counts")), [])
        if top_bin_top2:
            print("top standardized bin by top2 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in top_bin_top2[:3]))
        top_bin_top2_inc = next((rec.get("top_standardized_bin_top2_slot_incidence", []) for rec in margin_summaries if rec.get("top_standardized_bin_top2_slot_incidence")), [])
        if top_bin_top2_inc:
            print("top-standardized-bin top2-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in top_bin_top2_inc[:3]))
        top_bin_top2_acc = next((rec.get("top_standardized_bin_top2_slot_accuracy", []) for rec in margin_summaries if rec.get("top_standardized_bin_top2_slot_accuracy")), [])
        if top_bin_top2_acc:
            print("top-standardized-bin top2-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top_bin_top2_acc[:3]))
        neg_tail = next((rec.get("negative_tail_summary", {}) for rec in margin_summaries if rec.get("negative_tail_summary")), {})
        neg_labels = (neg_tail.get("top_perm_labels", []) or [])[:3]
        if neg_labels:
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_labels)
            print(
                "negative tail z<={:.2f} count={} | top perms: {}".format(
                    float(neg_tail.get("z_cutoff", float("nan"))),
                    int(neg_tail.get("count", 0)),
                    labels_str,
                )
            )
        neg_tail_acc = next((rec.get("negative_tail_accuracy", {}) for rec in margin_summaries if rec.get("negative_tail_accuracy")), {})
        if neg_tail_acc:
            print(
                "negative tail acc/rstd: {:.4f}/{:.4f} ({}/{})".format(
                    float(neg_tail_acc.get("accuracy", float("nan"))),
                    float(neg_tail_acc.get("recall_std", float("nan"))),
                    int(neg_tail_acc.get("correct", 0)),
                    int(neg_tail_acc.get("count", 0)),
                )
            )
        neg_tail_perm_acc = next((rec.get("negative_tail_perm_accuracy", []) for rec in margin_summaries if rec.get("negative_tail_perm_accuracy")), [])
        if neg_tail_perm_acc:
            print("negative-tail perm acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in neg_tail_perm_acc[:3]))
        right_tail = next((rec.get("right_tail_summary", {}) for rec in margin_summaries if rec.get("right_tail_summary")), {})
        right_labels = (right_tail.get("top_perm_labels", []) or [])[:3]
        if right_labels:
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in right_labels)
            print(
                "right tail z>={:.2f} count={} | top perms: {}".format(
                    float(right_tail.get("z_cutoff", float("nan"))),
                    int(right_tail.get("count", 0)),
                    labels_str,
                )
            )
        right_tail_acc = next((rec.get("right_tail_accuracy", {}) for rec in margin_summaries if rec.get("right_tail_accuracy")), {})
        if right_tail_acc:
            print(
                "right tail acc/rstd: {:.4f}/{:.4f} ({}/{})".format(
                    float(right_tail_acc.get("accuracy", float("nan"))),
                    float(right_tail_acc.get("recall_std", float("nan"))),
                    int(right_tail_acc.get("correct", 0)),
                    int(right_tail_acc.get("count", 0)),
                )
            )
        right_tail_perm_acc = next((rec.get("right_tail_perm_accuracy", []) for rec in margin_summaries if rec.get("right_tail_perm_accuracy")), [])
        if right_tail_perm_acc:
            print("right-tail perm acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in right_tail_perm_acc[:3]))
        entry_acc = next((rec.get("entry_accuracy", {}) for rec in margin_summaries if rec.get("entry_accuracy")), {})
        if entry_acc:
            print(
                "overall view acc/rstd: {:.4f}/{:.4f} ({}/{})".format(
                    float(entry_acc.get("accuracy", float("nan"))),
                    float(entry_acc.get("recall_std", float("nan"))),
                    int(entry_acc.get("correct", 0)),
                    int(entry_acc.get("count", 0)),
                )
            )
        overall_perm_acc = next((rec.get("overall_perm_accuracy", []) for rec in margin_summaries if rec.get("overall_perm_accuracy")), [])
        if overall_perm_acc:
            print("overall perm acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in overall_perm_acc[:3]))
        neg_bins = (neg_tail.get("top_bins", []) or [])
        if neg_bins:
            top_bin = neg_bins[0]
            top_labels = (top_bin.get("top_perm_labels", []) or [])[:3]
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in top_labels) if top_labels else ""
            print(
                "top negative bin [{:.2f}, {:.2f}] count={} | top perms: {}".format(
                    float(top_bin.get("range_left", float("nan"))),
                    float(top_bin.get("range_right", float("nan"))),
                    int(top_bin.get("count", 0)),
                    labels_str,
                )
            )
        neg_ideals = next((rec.get("negative_tail_ideal_counts", []) for rec in margin_summaries if rec.get("negative_tail_ideal_counts")), [])
        if neg_ideals:
            print("negative tail by ideal:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_ideals[:3]))
        neg_slots = next((rec.get("negative_tail_correct_slot_counts", []) for rec in margin_summaries if rec.get("negative_tail_correct_slot_counts")), [])
        if neg_slots:
            print("negative tail by correct slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_slots[:3]))
        tail_slot_inc = next((rec.get("negative_tail_slot_incidence", []) for rec in margin_summaries if rec.get("negative_tail_slot_incidence")), [])
        if tail_slot_inc:
            print("negative-tail slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in tail_slot_inc[:3]))
        slot_acc = next((rec.get("correct_slot_accuracy", []) for rec in margin_summaries if rec.get("correct_slot_accuracy")), [])
        if slot_acc:
            print("correct slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in slot_acc[:3]))
        tail_slot_acc = next((rec.get("negative_tail_correct_slot_accuracy", []) for rec in margin_summaries if rec.get("negative_tail_correct_slot_accuracy")), [])
        if tail_slot_acc:
            print("negative-tail slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in tail_slot_acc[:3]))
        right_ideals = next((rec.get("right_tail_ideal_counts", []) for rec in margin_summaries if rec.get("right_tail_ideal_counts")), [])
        if right_ideals:
            print("right tail by ideal:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in right_ideals[:3]))
        right_slots = next((rec.get("right_tail_correct_slot_counts", []) for rec in margin_summaries if rec.get("right_tail_correct_slot_counts")), [])
        if right_slots:
            print("right tail by correct slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in right_slots[:3]))
        right_slot_inc = next((rec.get("right_tail_slot_incidence", []) for rec in margin_summaries if rec.get("right_tail_slot_incidence")), [])
        if right_slot_inc:
            print("right-tail slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in right_slot_inc[:3]))
        right_slot_acc = next((rec.get("right_tail_correct_slot_accuracy", []) for rec in margin_summaries if rec.get("right_tail_correct_slot_accuracy")), [])
        if right_slot_acc:
            print("right-tail slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in right_slot_acc[:3]))
        neg_top1_slots = next((rec.get("negative_tail_top1_slot_counts", []) for rec in margin_summaries if rec.get("negative_tail_top1_slot_counts")), [])
        if neg_top1_slots:
            print("negative tail by top1 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_top1_slots[:3]))
        tail_top1_inc = next((rec.get("negative_tail_top1_slot_incidence", []) for rec in margin_summaries if rec.get("negative_tail_top1_slot_incidence")), [])
        if tail_top1_inc:
            print("negative-tail top1-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in tail_top1_inc[:3]))
        top1_acc = next((rec.get("top1_slot_accuracy", []) for rec in margin_summaries if rec.get("top1_slot_accuracy")), [])
        if top1_acc:
            print("top1-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top1_acc[:3]))
        tail_top1_acc = next((rec.get("negative_tail_top1_slot_accuracy", []) for rec in margin_summaries if rec.get("negative_tail_top1_slot_accuracy")), [])
        if tail_top1_acc:
            print("negative-tail top1-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in tail_top1_acc[:3]))
        right_top1_slots = next((rec.get("right_tail_top1_slot_counts", []) for rec in margin_summaries if rec.get("right_tail_top1_slot_counts")), [])
        if right_top1_slots:
            print("right tail by top1 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in right_top1_slots[:3]))
        right_top1_inc = next((rec.get("right_tail_top1_slot_incidence", []) for rec in margin_summaries if rec.get("right_tail_top1_slot_incidence")), [])
        if right_top1_inc:
            print("right-tail top1-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in right_top1_inc[:3]))
        right_top1_acc = next((rec.get("right_tail_top1_slot_accuracy", []) for rec in margin_summaries if rec.get("right_tail_top1_slot_accuracy")), [])
        if right_top1_acc:
            print("right-tail top1-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in right_top1_acc[:3]))
        neg_top2_slots = next((rec.get("negative_tail_top2_slot_counts", []) for rec in margin_summaries if rec.get("negative_tail_top2_slot_counts")), [])
        if neg_top2_slots:
            print("negative tail by top2 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in neg_top2_slots[:3]))
        tail_top2_inc = next((rec.get("negative_tail_top2_slot_incidence", []) for rec in margin_summaries if rec.get("negative_tail_top2_slot_incidence")), [])
        if tail_top2_inc:
            print("negative-tail top2-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in tail_top2_inc[:3]))
        top2_acc = next((rec.get("top2_slot_accuracy", []) for rec in margin_summaries if rec.get("top2_slot_accuracy")), [])
        if top2_acc:
            print("top2-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in top2_acc[:3]))
        tail_top2_acc = next((rec.get("negative_tail_top2_slot_accuracy", []) for rec in margin_summaries if rec.get("negative_tail_top2_slot_accuracy")), [])
        if tail_top2_acc:
            print("negative-tail top2-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in tail_top2_acc[:3]))
        right_top2_slots = next((rec.get("right_tail_top2_slot_counts", []) for rec in margin_summaries if rec.get("right_tail_top2_slot_counts")), [])
        if right_top2_slots:
            print("right tail by top2 slot:", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in right_top2_slots[:3]))
        right_top2_inc = next((rec.get("right_tail_top2_slot_incidence", []) for rec in margin_summaries if rec.get("right_tail_top2_slot_incidence")), [])
        if right_top2_inc:
            print("right-tail top2-slot incidence:", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in right_top2_inc[:3]))
        right_top2_acc = next((rec.get("right_tail_top2_slot_accuracy", []) for rec in margin_summaries if rec.get("right_tail_top2_slot_accuracy")), [])
        if right_top2_acc:
            print("right-tail top2-slot acc/rstd:", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in right_top2_acc[:3]))
        raw_top_bin = next((rec.get("raw_top_bin_summary", {}) for rec in margin_summaries if rec.get("raw_top_bin_summary")), {})
        if raw_top_bin:
            raw_top_labels = (raw_top_bin.get("top_perm_labels", []) or [])[:3]
            labels_str = ", ".join(f"{x.get('perm_label')}:{x.get('fraction', float('nan')):.2f}" for x in raw_top_labels) if raw_top_labels else ""
            print(
                "raw top bin [{:.4f}, {:.4f}] count={} | top perms: {}".format(
                    float(raw_top_bin.get("range_left", float("nan"))),
                    float(raw_top_bin.get("range_right", float("nan"))),
                    int(raw_top_bin.get("count", 0)),
                    labels_str,
                )
            )
        raw_top_subset = next((rec.get("raw_top_bin_subset", {}) for rec in margin_summaries if rec.get("raw_top_bin_subset")), {})
        if raw_top_subset:
            print("raw top bin acc/rstd: {:.4f}/{:.4f} ({}/{})".format(float(raw_top_subset.get("accuracy", float("nan"))), float(raw_top_subset.get("recall_std", float("nan"))), int(raw_top_subset.get("correct", 0)), int(raw_top_subset.get("count", 0))))
            for key, title in (
                ("perm_accuracy", "raw-top-bin perm acc/rstd"),
                ("ideal_counts", "raw top bin by ideal"),
                ("correct_slot_counts", "raw top bin by correct slot"),
                ("slot_incidence", "raw-top-bin slot incidence"),
                ("correct_slot_accuracy", "raw-top-bin slot acc/rstd"),
                ("top1_slot_counts", "raw top bin by top1 slot"),
                ("top1_slot_incidence", "raw-top-bin top1-slot incidence"),
                ("top1_slot_accuracy", "raw-top-bin top1-slot acc/rstd"),
                ("top2_slot_counts", "raw top bin by top2 slot"),
                ("top2_slot_incidence", "raw-top-bin top2-slot incidence"),
                ("top2_slot_accuracy", "raw-top-bin top2-slot acc/rstd"),
            ):
                vals = raw_top_subset.get(key, []) or []
                if not vals:
                    continue
                if "acc/rstd" in title:
                    print(title + ":", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in vals[:3]))
                elif "incidence" in title:
                    print(title + ":", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in vals[:3]))
                else:
                    print(title + ":", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in vals[:3]))
        for raw_key, raw_title, comp in (
            ("raw_left_tail_subset", "raw left tail", "<="),
            ("raw_right_tail_subset", "raw right tail", ">="),
        ):
            raw_subset = next((rec.get(raw_key, {}) for rec in margin_summaries if rec.get(raw_key)), {})
            if raw_subset:
                print(
                    f"{raw_title} residual {comp}{float(raw_subset.get('residual_cutoff', float('nan'))):.4f} acc/rstd: "
                    f"{float(raw_subset.get('accuracy', float('nan'))):.4f}/{float(raw_subset.get('recall_std', float('nan'))):.4f} "
                    f"({int(raw_subset.get('correct', 0))}/{int(raw_subset.get('count', 0))})"
                )
                for key, title in (
                    ("perm_accuracy", f"{raw_title.replace(' ', '-')} perm acc/rstd"),
                    ("ideal_counts", f"{raw_title} by ideal"),
                    ("correct_slot_counts", f"{raw_title} by correct slot"),
                    ("slot_incidence", f"{raw_title.replace(' ', '-')} slot incidence"),
                    ("correct_slot_accuracy", f"{raw_title.replace(' ', '-')} slot acc/rstd"),
                    ("top1_slot_counts", f"{raw_title} by top1 slot"),
                    ("top1_slot_incidence", f"{raw_title.replace(' ', '-')} top1-slot incidence"),
                    ("top1_slot_accuracy", f"{raw_title.replace(' ', '-')} top1-slot acc/rstd"),
                    ("top2_slot_counts", f"{raw_title} by top2 slot"),
                    ("top2_slot_incidence", f"{raw_title.replace(' ', '-')} top2-slot incidence"),
                    ("top2_slot_accuracy", f"{raw_title.replace(' ', '-')} top2-slot acc/rstd"),
                ):
                    vals = raw_subset.get(key, []) or []
                    if not vals:
                        continue
                    if "acc/rstd" in title:
                        print(title + ":", ", ".join(f"{x.get('label')}:{x.get('accuracy', float('nan')):.3f}/{x.get('recall_std', float('nan')):.3f}" for x in vals[:3]))
                    elif "incidence" in title:
                        print(title + ":", ", ".join(f"{x.get('label')}:{x.get('rate', float('nan')):.3f}" for x in vals[:3]))
                    else:
                        print(title + ":", ", ".join(f"{x.get('label')}:{x.get('fraction', float('nan')):.2f}" for x in vals[:3]))
        corr_split = next((rec.get("correctness_split", {}) for rec in margin_summaries if rec.get("correctness_split")), {})
        for split_name in ("correct", "incorrect"):
            split = (corr_split.get(split_name, {}) or {})
            if not split:
                continue
            neg = (split.get("negative_tail_summary", {}) or {})
            fit = (split.get("standardized_residual_fit", {}) or {})
            print(
                "{} refs: n_samples={}, mean_sigma_i={:.4f}, z_skew={:.4f}, left_tail_count={}".format(
                    split_name,
                    int(split.get("n_samples", 0)),
                    float(split.get("mean_sample_sigma_i", float("nan"))),
                    float(fit.get("skew", float("nan"))),
                    int(neg.get("count", 0)),
                )
            )
        first_t_summary = next((rec.get("t_view_summary", []) for rec in margin_summaries if rec.get("t_view_summary")), [])
        if first_t_summary:
            ts = sorted({int(x.get("t")) for x in first_t_summary if isinstance(x, dict) and "t" in x})
            for t in ts:
                vals_std = []
                vals_ratio = []
                vals_ratio_finite = []
                for rec in margin_summaries:
                    for item in rec.get("t_view_summary", []) or []:
                        if int(item.get("t", -1)) != int(t):
                            continue
                        if isinstance(item.get("resid_std"), (int, float)) and np.isfinite(float(item["resid_std"])):
                            vals_std.append(float(item["resid_std"]))
                        if isinstance(item.get("ratio_to_t1_sqrt"), (int, float)) and np.isfinite(float(item["ratio_to_t1_sqrt"])):
                            vals_ratio.append(float(item["ratio_to_t1_sqrt"]))
                        if isinstance(item.get("ratio_to_finite_view_target"), (int, float)) and np.isfinite(float(item["ratio_to_finite_view_target"])):
                            vals_ratio_finite.append(float(item["ratio_to_finite_view_target"]))
                if vals_std:
                    ratio_str = f", ratio_to_1/sqrt(T)={float(np.mean(vals_ratio)):.4f}" if vals_ratio else ""
                    finite_str = f", ratio_to_finite-view={float(np.mean(vals_ratio_finite)):.4f}" if vals_ratio_finite else ""
                    print(f"T={int(t)}: resid_std={float(np.mean(vals_std)):.4f}{ratio_str}{finite_str}")
        print("")

    out_path = os.path.join(str(cache_dir), "perm_noise_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(per_record_reports, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

    margin_noise_summary_by_k: Dict[str, object] = {}
    for k in sorted(margin_noise_by_k.keys()):
        bucket = margin_noise_by_k[int(k)]
        margin_noise_summary_by_k[str(int(k))] = _summarize_margin_noise_bucket(
            bucket,
            n_views=int(bucket.get("n_views", int(k))),
            reference_mode=str(bucket.get("reference_mode", "cyclic")),
            negative_tail_z_cutoff=float(negative_tail_z_cutoff),
            right_tail_z_cutoff=float(right_tail_z_cutoff),
        )

    margin_noise_summary_path = os.path.join(str(cache_dir), "perm_margin_noise_summary.json")
    with open(margin_noise_summary_path, "w", encoding="utf-8") as f:
        json.dump(margin_noise_summary_by_k, f, ensure_ascii=False, indent=2)
    print(f"Saved: {margin_noise_summary_path}")

    if save_margin_samples and margin_noise_sample_records:
        samples_path = os.path.join(str(cache_dir), "perm_margin_noise_samples.jsonl")
        with open(samples_path, "w", encoding="utf-8") as f:
            for rec in margin_noise_sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved: {samples_path}")

    saved_plot_paths: List[str] = []
    if save_plots or wandb_enabled:
        saved_plot_paths = _save_noise_plots(
            per_record_reports,
            str(cache_dir),
            corr_by_k=corr_by_k,
            recall_by_k=recall_by_k,
            task_name=task_name,
            margin_noise_by_k=margin_noise_by_k,
        )

    if wandb_enabled:
        _wandb_log_report(
            per_record_reports=per_record_reports,
            out_dir=str(cache_dir),
            report_json_path=out_path,
            plot_paths=saved_plot_paths,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_tags=wandb_tags,
            wandb_group=wandb_group,
            wandb_job_type=wandb_job_type,
        )


def _save_noise_plots(
    per_record_reports: List[dict],
    out_dir: str,
    corr_by_k: Optional[Dict[int, Dict[str, object]]] = None,
    recall_by_k: Optional[Dict[int, Dict[str, List[str]]]] = None,
    task_name: Optional[str] = None,
    margin_noise_by_k: Optional[Dict[int, Dict[str, object]]] = None,
) -> List[str]:
    """
    Save a small set of PNG plots into out_dir.
    - subset size (m) vs ensemble accuracy (mean±std over subject-run records)
    - permutation-accuracy variance summary (cyclic vs all perms)
    """
    plt = _try_import_matplotlib()
    if plt is None:
        print("[warn] matplotlib not available; skipping plot saving.")
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    # Task color palette (for "which dataset" identity, not method identity)
    task_palette = {
        "arc": "#1f77b4",   # blue
        "mmlu": "#ff7f0e",  # orange
        "csqa": "#2ca02c",  # green
    }
    curve_color = task_palette.get(str(task_name).lower(), "#1f77b4" if str(task_name).lower() == "arc" else "#2E86C1")

    # Group records by k
    ks = sorted({int(r.get("k", -1)) for r in per_record_reports if isinstance(r.get("k"), (int, float)) and int(r.get("k")) > 0})
    if not ks:
        return

    for k in ks:
        rs = [r for r in per_record_reports if int(r.get("k", -1)) == k]
        if not rs:
            continue

        # Aggregate subset curve across records
        all_ms = sorted({int(m) for r in rs for m in (r.get("subset_curve") or {}).keys()})
        xs: List[int] = []
        ys: List[float] = []
        es: List[float] = []
        for m in all_ms:
            vals = []
            for r in rs:
                sc = r.get("subset_curve") or {}
                if m in sc:
                    vals.append(float(sc[m][0]))
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            xs.append(int(m))
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

        # Baselines
        def _m(key: str) -> float:
            v = [float(r.get(key, float("nan"))) for r in rs]
            v = [x for x in v if np.isfinite(x)]
            return float(np.mean(v)) if v else float("nan")

        orig = _m("orig_acc")
        ens_cyc = _m("ens_acc_cyclic_perms")
        ens_all = _m("ens_acc_all_perms")
        full_enabled_any = any(bool(r.get("full_enabled")) for r in rs)

        # Plot: subset curve
        if xs:
            fig = plt.figure(figsize=(8.5, 5.5), dpi=160)
            ax = fig.add_subplot(1, 1, 1)
            task_tag = f"{task_name}" if task_name else "task"
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                fmt="-o",
                lw=1.8,
                ms=4,
                capsize=3,
                color=curve_color,
                ecolor=curve_color,
                label=f"{task_tag}: Ensemble(m perms) mean±std over subject-run",
            )
            if np.isfinite(orig):
                ax.axhline(orig, color="gray", ls="--", lw=1.2, label="Original (identity) macro-mean")
            if np.isfinite(ens_cyc):
                ax.axhline(ens_cyc, color="#8E44AD", ls=":", lw=1.4, label="Cyclic ensemble (all rotations) macro-mean")
            if np.isfinite(ens_all):
                lab = "Full/available ensemble (all perms) macro-mean" if full_enabled_any else "Available ensemble (all perms in cache) macro-mean"
                ax.axhline(ens_all, color="#2E86C1", ls="-.", lw=1.2, label=lab)
            ax.set_title(f"Permutation-noise: ensemble size vs accuracy (k={k})")
            ax.set_xlabel("m = #permutations mixed")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            p = os.path.join(out_dir, f"perm_noise_subset_curve_k{k}.png")
            fig.savefig(p)
            plt.close(fig)
            saved.append(p)

        # Plot: recall-by-option bar chart (aggregate over all subjects)
        if recall_by_k and int(k) in recall_by_k:
            try:
                rec = recall_by_k[int(k)]
                y_true = rec.get("y_true") or []
                y_orig = rec.get("orig") or []
                y_cyc = rec.get("cyclic") or []
                y_full = rec.get("full") or []
                if y_true and len(y_true) == len(y_orig) == len(y_cyc) == len(y_full):
                    options = list("ABCDE"[: int(k)]) if int(k) in (4, 5) else sorted(list({*y_true}))

                    def _acc(y_pred):
                        return float(np.mean([1.0 if p == t else 0.0 for p, t in zip(y_pred, y_true)])) if y_true else float("nan")

                    def _recalls(y_pred):
                        out = []
                        for opt in options:
                            idxs = [i for i, t in enumerate(y_true) if t == opt]
                            if not idxs:
                                out.append(0.0)
                            else:
                                out.append(float(np.mean([1.0 if y_pred[i] == opt else 0.0 for i in idxs])))
                        return out

                    r_orig = _recalls(y_orig)
                    r_cyc = _recalls(y_cyc)
                    r_full = _recalls(y_full)
                    a_orig = _acc(y_orig)
                    a_cyc = _acc(y_cyc)
                    a_full = _acc(y_full)
                    s_orig = float(np.std(r_orig))
                    s_cyc = float(np.std(r_cyc))
                    s_full = float(np.std(r_full))

                    x = np.arange(int(k))
                    width = 0.25
                    fig = plt.figure(figsize=(10.0, 6.0), dpi=160)
                    ax = fig.add_subplot(1, 1, 1)
                    b1 = ax.bar(
                        x - width,
                        r_orig,
                        width,
                        label=f"Original (Acc: {a_orig:.3f} | Std: {s_orig:.3f})",
                        color="#d62728",
                        alpha=0.82,
                    )
                    b2 = ax.bar(
                        x,
                        r_cyc,
                        width,
                        label=f"Cyclic {int(k)} (Acc: {a_cyc:.3f} | Std: {s_cyc:.3f})",
                        color="#1f77b4",
                        edgecolor="black",
                        linewidth=1.2,
                        alpha=0.85,
                    )
                    b3 = ax.bar(
                        x + width,
                        r_full,
                        width,
                        label=f"Full {len(_rotations(int(k))) if int(k) != 4 else 24} (Acc: {a_full:.3f} | Std: {s_full:.3f})",
                        color="#ff7f0e",
                        alpha=0.82,
                    )
                    ax.set_ylabel("Recall (accuracy per option)", fontsize=12)
                    ax.set_xlabel("Option", fontsize=12)
                    ax.set_title("Recall robustness per option (all subjects combined)", fontsize=14, pad=12)
                    ax.set_xticks(x)
                    ax.set_xticklabels(options, fontsize=12)
                    ax.grid(axis="y", linestyle=":", alpha=0.5)
                    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, title="Metrics (Lower Std = More Robust)")

                    def _autolabel(rects):
                        for rect in rects:
                            h = rect.get_height()
                            ax.annotate(
                                f"{h:.3f}",
                                xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                rotation=90,
                            )

                    _autolabel(b1)
                    _autolabel(b2)
                    _autolabel(b3)
                    ymin = max(0.0, min(r_orig + r_cyc + r_full) - 0.08)
                    ymax = min(1.0, max(r_orig + r_cyc + r_full) + 0.10)
                    ax.set_ylim(ymin, ymax)

                    fig.tight_layout()
                    # If only one k exists, keep the user's preferred name.
                    fname = "analyze_perm_recall.png" if len(ks) == 1 else f"analyze_perm_recall_k{k}.png"
                    p = os.path.join(out_dir, fname)
                    fig.savefig(p)
                    plt.close(fig)
                    saved.append(p)
            except Exception:
                pass

        # Plot: correlation heatmap + boxplot (Cyclic first, then non-cyclic)
        if corr_by_k and int(k) in corr_by_k:
            perm_list = corr_by_k[int(k)].get("perm_list")
            mats = corr_by_k[int(k)].get("mats")
            if isinstance(perm_list, list) and isinstance(mats, list) and mats:
                try:
                    # concat along samples axis: (P, sum_N)
                    mat_concat = np.concatenate([np.asarray(m, dtype=np.float64) for m in mats], axis=1)
                    P = int(mat_concat.shape[0])
                    cyc_perms = _rotations(int(k))
                    cyc_idxs = [perm_list.index(p) for p in cyc_perms if p in perm_list]
                    non_cyc_idxs = [i for i in range(P) if i not in set(cyc_idxs)]
                    order = list(cyc_idxs) + list(non_cyc_idxs)
                    X = mat_concat[order]
                    # robust corr (avoid NaN when variance=0)
                    X = X - X.mean(axis=1, keepdims=True)
                    denom = np.sqrt((X * X).sum(axis=1, keepdims=True)) + 1e-12
                    Xn = X / denom
                    C = Xn @ Xn.T

                    # Enhanced heatmap: remove diagonal, tight scale
                    C_plot = np.asarray(C, dtype=np.float64).copy()
                    np.fill_diagonal(C_plot, np.nan)
                    vmin = float(np.nanmin(C_plot)) if np.isfinite(np.nanmin(C_plot)) else -1.0
                    vmax = float(np.nanmax(C_plot)) if np.isfinite(np.nanmax(C_plot)) else 1.0
                    # If all correlations are positive-ish, a sequential cmap shows contrast better
                    if vmin >= 0.0:
                        cmap = "YlOrRd"
                    else:
                        cmap = "coolwarm"

                    fig = plt.figure(figsize=(8.0, 6.8), dpi=200)
                    ax = fig.add_subplot(1, 1, 1)
                    im = ax.imshow(C_plot, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
                    ax.set_title(f"Enhanced corr heatmap (diag removed, tight scale) (k={k})")

                    # Axis labels: C1..Ck then N1..N(P-k)
                    n_c = len(cyc_idxs)
                    labels = [f"C{i+1}" for i in range(n_c)] + [f"N{i+1}" for i in range(P - n_c)]
                    ax.set_xticks(np.arange(P))
                    ax.set_yticks(np.arange(P))
                    ax.set_xticklabels(labels, fontsize=6, rotation=90)
                    ax.set_yticklabels(labels, fontsize=6)
                    # separators between cyclic and non-cyclic
                    if n_c > 0 and n_c < P:
                        ax.axhline(y=n_c - 0.5, color="black", lw=1.5, ls="--")
                        ax.axvline(x=n_c - 0.5, color="black", lw=1.5, ls="--")
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Pearson corr (correctness vectors)", rotation=90)
                    fig.tight_layout()
                    p = os.path.join(out_dir, f"perm_noise_corr_heatmap_k{k}.png")
                    fig.savefig(p)
                    plt.close(fig)
                    saved.append(p)

                    # Boxplot comparison: cyclic pairs vs non-cyclic pairs
                    if n_c >= 2 and (P - n_c) >= 2:
                        cyc_block = C_plot[:n_c, :n_c]
                        non_block = C_plot[n_c:, n_c:]
                        cyc_pairs = cyc_block[np.triu_indices(n_c, k=1)]
                        non_pairs = non_block[np.triu_indices(P - n_c, k=1)]
                        cyc_pairs = cyc_pairs[np.isfinite(cyc_pairs)]
                        non_pairs = non_pairs[np.isfinite(non_pairs)]
                        if cyc_pairs.size > 0 and non_pairs.size > 0:
                            fig = plt.figure(figsize=(7.2, 5.8), dpi=200)
                            ax = fig.add_subplot(1, 1, 1)
                            bp = ax.boxplot(
                                [cyc_pairs, non_pairs],
                                tick_labels=["Cyclic pairs (C vs C)", "Non-cyclic pairs (N vs N)"],
                                patch_artist=True,
                                widths=0.55,
                                showfliers=False,
                            )
                            colors = ["#1f77b4", "#ff7f0e"]
                            for patch, color in zip(bp["boxes"], colors):
                                patch.set_facecolor(color)
                                patch.set_alpha(0.55)
                            for med in bp["medians"]:
                                med.set(color="black", linewidth=2)

                            # jittered scatter for readability
                            rng = np.random.default_rng(42)
                            x1 = rng.normal(1.0, 0.045, size=cyc_pairs.size)
                            x2 = rng.normal(2.0, 0.045, size=non_pairs.size)
                            ax.scatter(x1, cyc_pairs, color="blue", alpha=0.55, s=12, zorder=3)
                            ax.scatter(x2, non_pairs, color="red", alpha=0.40, s=12, zorder=3)
                            ax.set_ylabel("Pearson correlation")
                            ax.set_title(f"Error-correlation distribution: cyclic vs non-cyclic (k={k})")
                            ax.grid(axis="y", linestyle="--", alpha=0.35)
                            fig.tight_layout()
                            p2 = os.path.join(out_dir, f"perm_noise_corr_boxplot_k{k}.png")
                            fig.savefig(p2)
                            plt.close(fig)
                            saved.append(p2)
                except Exception:
                    pass

        # Plot: variance bars
        cyc_vars = [float(r.get("cyc_nonid_var_acc", float("nan"))) for r in rs]
        all_vars = [float(r.get("full_nonid_var_acc", float("nan"))) for r in rs]
        cyc_vars = [v for v in cyc_vars if np.isfinite(v)]
        all_vars = [v for v in all_vars if np.isfinite(v)]
        if cyc_vars or all_vars:
            fig = plt.figure(figsize=(6.5, 4.2), dpi=160)
            ax = fig.add_subplot(1, 1, 1)
            labels = []
            vals = []
            errs = []
            if cyc_vars:
                labels.append("Cyclic perms (non-id)\nvar(acc)")
                vals.append(float(np.mean(cyc_vars)))
                errs.append(float(np.std(cyc_vars)) if len(cyc_vars) > 1 else 0.0)
            if all_vars:
                labels.append("All perms (non-id)\nvar(acc)")
                vals.append(float(np.mean(all_vars)))
                errs.append(float(np.std(all_vars)) if len(all_vars) > 1 else 0.0)
            x = np.arange(len(labels))
            ax.bar(x, vals, yerr=errs, capsize=4, color=["#8E44AD", "#2E86C1"][: len(labels)], alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Variance")
            ax.set_title(f"Permutation accuracy variance (k={k})")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            p = os.path.join(out_dir, f"perm_noise_perm_acc_variance_k{k}.png")
            fig.savefig(p)
            plt.close(fig)
            saved.append(p)

        if margin_noise_by_k and int(k) in margin_noise_by_k:
            try:
                margin_bucket = margin_noise_by_k[int(k)]
                ref_mode = str(margin_bucket.get("reference_mode", "cyclic"))
                mode_label = "Full-permutation" if ref_mode == "full" else "Cyclic"
                raw_residuals = np.asarray(margin_bucket.get("residuals", []), dtype=np.float64)
                z_scores = np.asarray(margin_bucket.get("z_scores", []), dtype=np.float64)
                sample_ref = np.asarray(margin_bucket.get("sample_ref_margins", []), dtype=np.float64)
                sample_sigma = np.asarray(margin_bucket.get("sample_sigmas", []), dtype=np.float64)
                t_residuals = margin_bucket.get("t_residuals", {}) or {}

                if raw_residuals.size > 0:
                    fig = plt.figure(figsize=(8.0, 5.0), dpi=180)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.hist(raw_residuals, bins=50, density=True, alpha=0.68, color="#6baed6", label="Raw residuals")
                    fit_raw = _gaussian_fit_report(raw_residuals)
                    fit_laplace = _laplace_fit_report(raw_residuals)
                    fit_cauchy = _cauchy_fit_report(raw_residuals)
                    mu = float(fit_raw.get("mean", float("nan")))
                    sigma = float(fit_raw.get("std", float("nan")))
                    x_min = float(np.min(raw_residuals))
                    x_max = float(np.max(raw_residuals))
                    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                        xs = np.linspace(x_min, x_max, 400)
                        if np.isfinite(mu) and np.isfinite(sigma) and sigma > 1e-12:
                            pdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
                            ax.plot(
                                xs,
                                pdf,
                                color="#d62728",
                                lw=2.0,
                                label="Gaussian (KS={:.3f}, KL={:.3f})".format(
                                    float(fit_raw.get("ks_to_fit", float("nan"))),
                                    float(fit_raw.get("kl_hist_to_fit", float("nan"))),
                                ),
                            )
                        laplace_pdf = _laplace_pdf(xs, float(fit_laplace.get("loc", float("nan"))), float(fit_laplace.get("scale", float("nan"))))
                        if np.all(np.isfinite(laplace_pdf)):
                            ax.plot(
                                xs,
                                laplace_pdf,
                                color="#2ca02c",
                                lw=1.8,
                                ls="--",
                                label="Laplace (KS={:.3f}, KL={:.3f})".format(
                                    float(fit_laplace.get("ks_to_fit", float("nan"))),
                                    float(fit_laplace.get("kl_hist_to_fit", float("nan"))),
                                ),
                            )
                        cauchy_pdf = _cauchy_pdf(xs, float(fit_cauchy.get("loc", float("nan"))), float(fit_cauchy.get("scale", float("nan"))))
                        if np.all(np.isfinite(cauchy_pdf)):
                            ax.plot(
                                xs,
                                cauchy_pdf,
                                color="#9467bd",
                                lw=1.8,
                                ls=":",
                                label="Cauchy (KS={:.3f}, KL={:.3f})".format(
                                    float(fit_cauchy.get("ks_to_fit", float("nan"))),
                                    float(fit_cauchy.get("kl_hist_to_fit", float("nan"))),
                                ),
                            )
                    ax.set_title(
                        f"{mode_label} margin residuals (raw) (k={k})\n"
                        f"mean={fit_raw['mean']:.3f}, std={fit_raw['std']:.3f}, compare Gaussian / Laplace / Cauchy"
                    )
                    ax.set_xlabel("residual = margin - ref_margin")
                    ax.set_ylabel("Density")
                    ax.grid(True, alpha=0.25)
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    p = os.path.join(out_dir, f"perm_margin_noise_raw_hist_k{k}.png")
                    fig.savefig(p)
                    plt.close(fig)
                    saved.append(p)
                    p_slot = os.path.join(out_dir, f"perm_margin_noise_raw_hist_by_correct_slot_k{k}.png")
                    saved_slot = _save_raw_fit_by_label_plot(
                        plt=plt,
                        values=raw_residuals,
                        labels=bucket.get("residual_correct_slot_labels", []) or [],
                        out_path=p_slot,
                        title=f"{mode_label} raw residuals by correct slot (k={k})",
                    )
                    if saved_slot:
                        saved.append(saved_slot)
                    p_top1 = os.path.join(out_dir, f"perm_margin_noise_raw_hist_by_top1_slot_k{k}.png")
                    saved_top1 = _save_raw_fit_by_label_plot(
                        plt=plt,
                        values=raw_residuals,
                        labels=bucket.get("residual_top1_slot_labels", []) or [],
                        out_path=p_top1,
                        title=f"{mode_label} raw residuals by top1 slot (k={k})",
                    )
                    if saved_top1:
                        saved.append(saved_top1)
                    p_top2 = os.path.join(out_dir, f"perm_margin_noise_raw_hist_by_top2_slot_k{k}.png")
                    saved_top2 = _save_raw_fit_by_label_plot(
                        plt=plt,
                        values=raw_residuals,
                        labels=bucket.get("residual_top2_slot_labels", []) or [],
                        out_path=p_top2,
                        title=f"{mode_label} raw residuals by top2 slot (k={k})",
                    )
                    if saved_top2:
                        saved.append(saved_top2)
                    p_sweep = os.path.join(out_dir, f"perm_margin_noise_raw_param_sweep_k{k}.png")
                    saved_sweep = _save_raw_param_sweep_plot(
                        plt=plt,
                        values=raw_residuals,
                        out_path=p_sweep,
                        title=f"{mode_label} raw residual parameter sweep (k={k})",
                        laplace_loc=float(fit_laplace.get("loc", float("nan"))),
                        laplace_scale=float(fit_laplace.get("scale", float("nan"))),
                        cauchy_loc=float(fit_cauchy.get("loc", float("nan"))),
                        cauchy_scale=float(fit_cauchy.get("scale", float("nan"))),
                    )
                    if saved_sweep:
                        saved.append(saved_sweep)

                if z_scores.size > 0:
                    fig = plt.figure(figsize=(8.0, 5.0), dpi=180)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.hist(z_scores, bins=50, density=True, alpha=0.68, color="#1f77b4", label="Standardized residuals")
                    x_min = float(np.min(z_scores))
                    x_max = float(np.max(z_scores))
                    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                        xs = np.linspace(max(-4.0, x_min), min(4.0, x_max), 300)
                        normal_pdf = np.exp(-0.5 * xs * xs) / math.sqrt(2.0 * math.pi)
                        ax.plot(xs, normal_pdf, color="#d62728", lw=2.0, label="N(0,1)")
                    fit = _gaussian_fit_report(z_scores)
                    ax.set_title(
                        f"{mode_label} margin residuals (standardized) (k={k})\n"
                        f"mean={fit['mean']:.3f}, std={fit['std']:.3f}, KS={fit['ks_to_fit']:.3f}, KL={fit['kl_hist_to_fit']:.3f}"
                    )
                    ax.set_xlabel("z = residual / sigma_i")
                    ax.set_ylabel("Density")
                    ax.grid(True, alpha=0.25)
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    p = os.path.join(out_dir, f"perm_margin_noise_hist_k{k}.png")
                    fig.savefig(p)
                    plt.close(fig)
                    saved.append(p)

                    fig = plt.figure(figsize=(5.6, 5.6), dpi=180)
                    ax = fig.add_subplot(1, 1, 1)
                    zs = np.sort(z_scores)
                    if zs.size > 4000:
                        idx = np.linspace(0, zs.size - 1, 4000).astype(np.int64)
                        zs = zs[idx]
                    probs = (np.arange(1, zs.size + 1, dtype=np.float64) - 0.5) / float(zs.size)
                    nd0 = NormalDist()
                    q_theory = np.asarray([nd0.inv_cdf(float(pv)) for pv in probs], dtype=np.float64)
                    ax.scatter(q_theory, zs, s=8, alpha=0.30, color="#1f77b4")
                    lo = float(min(np.min(q_theory), np.min(zs)))
                    hi = float(max(np.max(q_theory), np.max(zs)))
                    ax.plot([lo, hi], [lo, hi], color="#d62728", lw=1.8, ls="--")
                    ax.set_title(f"QQ plot vs N(0,1) (k={k})")
                    ax.set_xlabel("Theoretical quantiles")
                    ax.set_ylabel("Empirical quantiles")
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    p = os.path.join(out_dir, f"perm_margin_noise_qq_k{k}.png")
                    fig.savefig(p)
                    plt.close(fig)
                    saved.append(p)

                if sample_ref.size > 0 and sample_sigma.size > 0 and sample_ref.size == sample_sigma.size:
                    fig = plt.figure(figsize=(6.8, 5.2), dpi=180)
                    ax = fig.add_subplot(1, 1, 1)
                    ax.scatter(sample_ref, sample_sigma, s=12, alpha=0.22, color="#2ca02c")
                    corr = _safe_corr(sample_ref, sample_sigma)
                    ax.set_title(f"Sample sigma vs {ref_mode} reference margin (k={k})\nCorr={corr:.3f}")
                    ax.set_xlabel("Cyclic reference margin")
                    ax.set_ylabel("sigma_i")
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    p = os.path.join(out_dir, f"perm_margin_sigma_vs_margin_k{k}.png")
                    fig.savefig(p)
                    plt.close(fig)
                    saved.append(p)

                if t_residuals:
                    ts = sorted(int(t) for t in t_residuals.keys())
                    emp_std = []
                    for t in ts:
                        arr = np.asarray(t_residuals.get(t, []), dtype=np.float64)
                        emp_std.append(float(np.std(arr)) if arr.size > 0 else float("nan"))
                    if emp_std:
                        base_std = emp_std[0] if np.isfinite(emp_std[0]) else float("nan")
                        target = [float(base_std / math.sqrt(float(t))) if np.isfinite(base_std) else float("nan") for t in ts]
                        n_views = max(ts) if ts else 0
                        finite_target = [
                            float(base_std * math.sqrt(float(max(0, n_views - int(t))) / float(max(1, int(t)) * max(1, n_views - 1))))
                            if np.isfinite(base_std) and n_views > 1
                            else float("nan")
                            for t in ts
                        ]
                        fig = plt.figure(figsize=(6.8, 4.8), dpi=180)
                        ax = fig.add_subplot(1, 1, 1)
                        ax.plot(ts, emp_std, "-o", color="#1f77b4", lw=2.0, ms=5, label="Empirical residual std")
                        ax.plot(ts, target, "--s", color="#d62728", lw=1.6, ms=4, label="std(T=1) / sqrt(T)")
                        if n_views > 1:
                            ax.plot(ts, finite_target, ":^", color="#2ca02c", lw=1.6, ms=4, label="Finite-view target")
                        ax.set_title(f"View-count scaling of {ref_mode} residual std (k={k})")
                        ax.set_xlabel("T = #views averaged")
                        ax.set_ylabel("Residual std")
                        ax.grid(True, alpha=0.25)
                        ax.legend(fontsize=8)
                        fig.tight_layout()
                        p = os.path.join(out_dir, f"perm_margin_t_scaling_k{k}.png")
                        fig.savefig(p)
                        plt.close(fig)
                        saved.append(p)
            except Exception:
                pass

    return saved


def _wandb_log_report(
    *,
    per_record_reports: List[dict],
    out_dir: str,
    report_json_path: str,
    plot_paths: List[str],
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    wandb_run_name: Optional[str],
    wandb_tags: Optional[str],
    wandb_group: Optional[str],
    wandb_job_type: str,
) -> None:
    wandb = _try_import_wandb()
    if wandb is None:
        print("[warn] wandb not available; skipping W&B logging.")
        return

    tags = [t.strip() for t in (wandb_tags or "").split(",") if t.strip()] if wandb_tags else None
    try:
        run = wandb.init(
            project=wandb_project or None,
            entity=wandb_entity or None,
            name=wandb_run_name or None,
            group=wandb_group or None,
            job_type=str(wandb_job_type or "analysis"),
            tags=tags,
            reinit=True,
        )
    except Exception as e:
        print(f"[warn] wandb.init failed; skipping W&B logging: {e}")
        return
    try:
        # Helpful for debugging in cluster logs
        if getattr(run, "url", None):
            print(f"W&B run: {run.url}")
    except Exception:
        pass

    # Log key summary scalars (macro over records), per k
    try:
        ks = sorted({int(r.get("k", -1)) for r in per_record_reports if isinstance(r.get("k"), (int, float)) and int(r.get("k")) > 0})
        for k in ks:
            rs = [r for r in per_record_reports if int(r.get("k", -1)) == k]
            if not rs:
                continue
            def _mean(key: str) -> float:
                v = [float(x.get(key, float("nan"))) for x in rs]
                v = [x for x in v if np.isfinite(x)]
                return float(np.mean(v)) if v else float("nan")
            wandb.log({
                f"perm_noise/k{k}/orig_acc_mean": _mean("orig_acc"),
                f"perm_noise/k{k}/ens_cyclic_mean": _mean("ens_acc_cyclic_perms"),
                f"perm_noise/k{k}/ens_all_mean": _mean("ens_acc_all_perms"),
                f"perm_noise/k{k}/corr_cyclic_mean": _mean("corr_cyclic_mean"),
                f"perm_noise/k{k}/corr_all_mean": _mean("corr_full_mean"),
                f"perm_noise/k{k}/var_cyclic_nonid_mean": _mean("cyc_nonid_var_acc"),
                f"perm_noise/k{k}/var_all_nonid_mean": _mean("full_nonid_var_acc"),
                f"perm_noise/k{k}/n_records": float(len(rs)),
            })
    except Exception:
        pass

    # Log images
    imgs = []
    for p in plot_paths:
        if not p:
            continue
        if not os.path.exists(p):
            print(f"[warn] plot missing, not uploading: {p}")
            continue
        try:
            key = os.path.splitext(os.path.basename(p))[0]
            img = wandb.Image(p, caption=key)
            imgs.append(img)
            wandb.log({f"perm_noise/plots/{key}": img})
        except Exception as e:
            print(f"[warn] failed to upload image {p}: {e}")
    # Also log a single gallery key for convenience
    if imgs:
        try:
            wandb.log({"perm_noise/plots_gallery": imgs})
        except Exception:
            pass
    else:
        print("[warn] no plots to upload (did matplotlib fail? did --save_plots run?)")

    # Save JSON into the run (files panel)
    try:
        # wandb.save works best with paths relative to base_path
        wandb.save(os.path.basename(report_json_path), base_path=out_dir, policy="now")
    except Exception:
        pass

    try:
        run.finish()
    except Exception:
        pass


def _perm_accs_and_correct_matrix(
    results: List[dict],
    perm_list: List[Tuple[int, ...]],
    option_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - perm_accs: shape (P,)  (accuracy per permutation when used alone)
      - correct_mat: shape (P,N)  (0/1 correctness per permutation and sample)
    """
    P = len(perm_list)
    if P <= 0:
        raise ValueError("perm_list empty")
    N = len(results)
    correct_mat = np.zeros((P, N), dtype=np.int8)
    for i, r in enumerate(results):
        data = r.get("data", {}) or {}
        probs_seq = data.get("probs", None)
        if not isinstance(probs_seq, list) or len(probs_seq) != P:
            raise ValueError(f"Bad probs shape at sample {i}: expected {P}, got {len(probs_seq) if isinstance(probs_seq, list) else type(probs_seq)}")
        ideal = str(data.get("ideal"))
        for pidx, p in enumerate(perm_list):
            agg = _aggregate_probs_over_permutations([probs_seq[pidx]], [p], len(option_ids))
            pred = option_ids[int(np.argmax(agg))]
            correct_mat[pidx, i] = 1 if pred == ideal else 0
    perm_accs = correct_mat.mean(axis=1).astype(np.float64)
    return perm_accs, correct_mat


def _pairwise_corr_mean(correct_mat: np.ndarray, max_perms: int, seed: int) -> float:
    """
    Mean of off-diagonal Pearson correlations between permutation correctness vectors.
    correct_mat: (P,N) binary
    """
    P, N = correct_mat.shape
    if P <= 1 or N <= 1:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    idxs = np.arange(P)
    if max_perms > 0 and P > max_perms:
        idxs = rng.choice(idxs, size=int(max_perms), replace=False)
    X = correct_mat[idxs].astype(np.float64)
    # Center
    X = X - X.mean(axis=1, keepdims=True)
    denom = np.sqrt((X * X).sum(axis=1, keepdims=True)) + 1e-12
    Xn = X / denom
    C = Xn @ Xn.T
    # Off-diagonal mean
    m = int(C.shape[0])
    if m <= 1:
        return float("nan")
    off = C[np.triu_indices(m, k=1)]
    return float(np.mean(off)) if off.size else float("nan")


def _ensemble_acc_subset_curve(
    results: List[dict],
    perm_list: List[Tuple[int, ...]],
    option_ids: List[str],
    sizes: List[int],
    n_trials: int,
    seed: int,
) -> Dict[int, Tuple[float, float]]:
    """
    For each subset size m, sample n_trials random subsets of permutations,
    compute ensemble accuracy by averaging unpermuted probs over chosen perms.
    Returns: m -> (mean_acc, std_acc)
    """
    P = len(perm_list)
    N = len(results)
    rng = np.random.default_rng(int(seed))

    # Preload probs_seq + ideals once
    probs_all = []
    ideals = []
    for r in results:
        d = r.get("data", {}) or {}
        probs_seq = d.get("probs", None)
        if not isinstance(probs_seq, list) or len(probs_seq) != P:
            raise ValueError("Bad probs in results for subset-curve")
        probs_all.append(probs_seq)
        ideals.append(str(d.get("ideal")))

    out: Dict[int, Tuple[float, float]] = {}
    for m in sizes:
        m = int(m)
        if m <= 0 or m > P:
            continue
        accs = []
        for _ in range(int(n_trials)):
            chosen = rng.choice(np.arange(P), size=m, replace=False)
            chosen = np.sort(chosen)
            chosen_perms = [perm_list[int(i)] for i in chosen]
            corrects = 0
            for i in range(N):
                chosen_probs = [probs_all[i][int(j)] for j in chosen]
                agg = _aggregate_probs_over_permutations(chosen_probs, chosen_perms, len(option_ids))
                pred = option_ids[int(np.argmax(agg))]
                corrects += 1 if pred == ideals[i] else 0
            accs.append(corrects / float(N) if N > 0 else float("nan"))
        out[m] = (float(np.mean(accs)), float(np.std(accs)) if len(accs) > 1 else 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)

    # ===== Mode A: analysis-only (point to existing cached results) =====
    ap.add_argument("--results_dir", type=str, default="", help="Directory with eval_clm cached subject jsonl files.")
    ap.add_argument("--results_dirs", type=str, nargs="+", default=None,
                    help="Multiple cached result directories for multi-model aggregation.")
    ap.add_argument("--subjects", type=str, default="", help="Comma-separated subject list. If empty, infer from results_dir.")
    ap.add_argument("--aggregate_out_dir", type=str, default="",
                    help="Output directory for multi-model aggregation files. Defaults to the first results_dir.")

    # ===== Mode B: eval+analyze (run eval_clm.py first) =====
    ap.add_argument("--eval_clm_path", type=str, default="", help="Path to eval_clm.py. Default: sibling eval_clm.py.")
    ap.add_argument("--skip_eval", action="store_true", help="Skip running eval_clm, only analyze inferred results_dir.")

    # These mirror the common eval_clm args so you can run:
    # python analyze_perm_noise.py --pretrained_model_path ... --eval_names csqa,5,full ...
    ap.add_argument("--pretrained_model_path", type=str, default="", help="(wrapper) Hugging Face model id or path.")
    ap.add_argument("--eval_names", type=str, nargs="+", default=[], help="(wrapper) eval names, e.g. csqa,5,full")
    ap.add_argument("--option_id_set", type=str, default=None, help="(wrapper) option id set (e.g., ABCDE)")
    ap.add_argument("--n_runs", type=int, default=1, help="(wrapper/analysis) Number of runs.")

    # Analysis knobs
    ap.add_argument("--seed", type=int, default=0, help="Random seed (subset sampling / corr subsampling).")
    ap.add_argument("--max_samples", type=int, default=0, help="If >0, truncate to first N samples per subject-run.")
    ap.add_argument("--subset_trials", type=int, default=50, help="Trials per subset size for ensemble curve.")
    ap.add_argument("--subset_sizes", type=str, default="1,2,3,4,6,8,12,16,24",
                    help="Comma-separated subset sizes m (number of perms to mix). Sizes > P are ignored.")
    ap.add_argument("--max_perms_corr", type=int, default=24, help="Max permutations for correlation calc (subsample if larger).")
    ap.add_argument("--full", action="store_true",
                    help="Use the full-permutation ensemble as pseudo-GT/reference when full permutations are available.")
    ap.add_argument("--max_t_combinations", type=int, default=2048,
                    help="Max number of subset combinations sampled per T for margin-noise T-scaling.")
    ap.add_argument("--negative_tail_z_cutoff", type=float, default=-1.5,
                    help="Standardized z cutoff for summarizing adverse left-tail permutations.")
    ap.add_argument("--right_tail_z_cutoff", type=float, default=1.5,
                    help="Standardized z cutoff for summarizing favorable right-tail permutations.")
    ap.add_argument("--save_plots", action="store_true", help="Save PNG plots into results_dir (default: off).")
    ap.add_argument("--save_margin_samples", action="store_true",
                    help="Save sample-level cyclic margin residuals/sigma_i jsonl into results_dir.")

    # W&B logging for analysis (separate run)
    ap.add_argument("--wandb", action="store_true", help="Log analysis results (metrics/images) to Weights & Biases.")
    ap.add_argument("--wandb_project", type=str, default=None, help="W&B project name (analysis).")
    # Match eval_clm default entity ("capde") to avoid silent 'wrong entity' uploads.
    ap.add_argument("--wandb_entity", type=str, default="capde", help="W&B entity (analysis).")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (analysis).")
    ap.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags for W&B run.")
    ap.add_argument("--wandb_group", type=str, default=None, help="W&B group for analysis runs.")
    ap.add_argument("--wandb_job_type", type=str, default="perm_noise_analysis", help="W&B job_type.")

    # Parse known args and forward the rest to eval_clm.py
    args, unknown = ap.parse_known_args()

    # If results_dirs is explicitly given, do multi-model aggregation.
    if args.results_dirs:
        subjects = [s.strip() for s in str(args.subjects).split(",") if s.strip()] if args.subjects else None
        _run_multi_results_analysis(
            results_dirs=[str(x) for x in args.results_dirs],
            subjects=subjects,
            n_runs=int(args.n_runs),
            max_samples=int(args.max_samples),
            aggregate_out_dir=str(args.aggregate_out_dir),
            save_plots=bool(args.save_plots),
            use_full_reference=bool(args.full),
            combo_sample_limit=int(args.max_t_combinations),
            negative_tail_z_cutoff=float(args.negative_tail_z_cutoff),
            right_tail_z_cutoff=float(args.right_tail_z_cutoff),
        )
        return

    # If results_dir is explicitly given, do analysis-only.
    if str(args.results_dir).strip():
        # Infer task name from results_dir like "results_arc/..." when possible
        _td = str(args.results_dir)
        task_name = _infer_task_name_from_path(_td)
        subjects = [s.strip() for s in str(args.subjects).split(",") if s.strip()] if args.subjects else None
        _run_analysis(
            cache_dir=str(args.results_dir),
            subjects=subjects,
            n_runs=int(args.n_runs),
            seed=int(args.seed),
            max_samples=int(args.max_samples),
            subset_trials=int(args.subset_trials),
            subset_sizes=str(args.subset_sizes),
            max_perms_corr=int(args.max_perms_corr),
            save_plots=bool(args.save_plots),
            wandb_enabled=bool(args.wandb),
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_tags=args.wandb_tags,
            wandb_group=args.wandb_group,
            wandb_job_type=str(args.wandb_job_type),
            task_name=task_name,
            save_margin_samples=bool(args.save_margin_samples),
            use_full_reference=bool(args.full),
            combo_sample_limit=int(args.max_t_combinations),
            negative_tail_z_cutoff=float(args.negative_tail_z_cutoff),
            right_tail_z_cutoff=float(args.right_tail_z_cutoff),
        )
        return

    # Otherwise, wrapper mode: run eval_clm then analyze inferred results_dir(s).
    if not str(args.pretrained_model_path).strip() or not args.eval_names:
        raise SystemExit(
            "Provide either --results_dirs (multi-model), --results_dir (analysis-only), OR provide --pretrained_model_path and --eval_names (eval+analyze)."
        )

    eval_clm_path = str(args.eval_clm_path).strip()
    if not eval_clm_path:
        # Default to sibling eval_clm.py (same folder as this script)
        eval_clm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_clm.py")

    if not os.path.exists(eval_clm_path):
        raise SystemExit(f"eval_clm.py not found: {eval_clm_path}")

    code_dir = os.path.dirname(os.path.abspath(eval_clm_path))

    if not bool(args.skip_eval):
        cmd = [sys.executable, os.path.abspath(eval_clm_path)]
        # Forward wrapper-known args in eval_clm-compatible form + any unknown flags
        cmd += ["--pretrained_model_path", str(args.pretrained_model_path)]
        cmd += ["--eval_names"] + [str(x) for x in args.eval_names]
        if args.option_id_set:
            cmd += ["--option_id_set", str(args.option_id_set)]
        if int(args.n_runs) != 1:
            cmd += ["--n_runs", str(int(args.n_runs))]
        # Forward W&B flags to eval_clm as well (so eval logs are preserved)
        if bool(args.wandb):
            cmd += ["--wandb"]
        if args.wandb_project:
            cmd += ["--wandb_project", str(args.wandb_project)]
        if args.wandb_entity:
            cmd += ["--wandb_entity", str(args.wandb_entity)]
        if args.wandb_run_name:
            cmd += ["--wandb_run_name", str(args.wandb_run_name)]
        cmd += list(unknown)

        print("==== Running eval_clm.py ====")
        print("cwd:", code_dir)
        print("cmd:", " ".join(cmd))
        subprocess.run(cmd, cwd=code_dir, check=True)

    # Analyze each eval_name separately (results_dir differs by task/shot/setting)
    for eval_name in args.eval_names:
        # eval_name like "arc,0,full"
        try:
            task_name = str(eval_name).split(",")[0].strip()
        except Exception:
            task_name = None
        rel_results_dir = _compute_eval_save_path(str(eval_name), str(args.pretrained_model_path), args.option_id_set)
        results_dir = os.path.join(code_dir, rel_results_dir)
        print("")
        print(f"==== Analyze: {eval_name} ====")
        _run_analysis(
            cache_dir=results_dir,
            subjects=None,
            n_runs=int(args.n_runs),
            seed=int(args.seed),
            max_samples=int(args.max_samples),
            subset_trials=int(args.subset_trials),
            subset_sizes=str(args.subset_sizes),
            max_perms_corr=int(args.max_perms_corr),
            save_plots=bool(args.save_plots),
            wandb_enabled=bool(args.wandb),
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=(args.wandb_run_name or f"perm_noise/{os.path.basename(str(args.pretrained_model_path))}/{eval_name}"),
            wandb_tags=args.wandb_tags,
            wandb_group=args.wandb_group,
            wandb_job_type=str(args.wandb_job_type),
            task_name=task_name,
            save_margin_samples=bool(args.save_margin_samples),
            use_full_reference=bool(args.full),
            combo_sample_limit=int(args.max_t_combinations),
            negative_tail_z_cutoff=float(args.negative_tail_z_cutoff),
            right_tail_z_cutoff=float(args.right_tail_z_cutoff),
        )


if __name__ == "__main__":
    main()
