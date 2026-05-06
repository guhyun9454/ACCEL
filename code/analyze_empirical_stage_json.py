#!/usr/bin/env python3
"""
Analyze saved empirical-stage JSON outputs produced by eval_clm.py.

Primary inputs:
  - results_.../full.../empirical_analysis/{subject}_run{r}_empirical_alpha{a}_summary.json
  - results_.../full.../empirical_analysis/{subject}_run{r}_empirical_alpha{a}_trajectories.jsonl
  - results_.../full.../{task}_empirical_stage_analysis.json
  - results_.../full.../{task}_three_curves_points.json   (optional)

This script builds one consolidated report json/markdown file.  It can also
produce an extra cyclic_learned summary when requested.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _sanitize_result_tag(tag: Optional[str]) -> str:
    s = str(tag or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s.strip("-._")


def _build_results_dir(
    *,
    task: str,
    num_few_shot: int,
    model_name: str,
    option_id_set: Optional[str],
    result_tag: Optional[str],
    setting: str = "full",
) -> str:
    model_leaf = str(model_name).strip().split("/")[-1]
    out = f"results_{task}/{int(num_few_shot)}s_{model_leaf}/{task}"
    if setting:
        out += f"_{setting}"
    if option_id_set:
        out += f"_id-{option_id_set}"
    tag = _sanitize_result_tag(result_tag)
    if tag:
        out += f"__{tag}"
    return out


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_result_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = _load_jsonl(path)
    out = [row for row in rows if isinstance(row, dict) and row.get("type") == "result"]
    out = sorted(out, key=lambda x: int(((x.get("data") or {}).get("idx", -1))))
    return out


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _float_key(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("inf")


def _stats(values: Iterable[Any]) -> Dict[str, Any]:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if vals.shape[0] == 0 or m.shape[0] != vals.shape[0]:
        return float("nan")
    if not np.any(m):
        return float("nan")
    return float(np.mean(vals[m]))


def _recall_std(labels: List[int], preds: List[int], k: int) -> float:
    if int(k) <= 0:
        return float("nan")
    positives = [0] * int(k)
    true_pos = [0] * int(k)
    for y, p in zip(labels, preds):
        yi = int(y)
        pi = int(p)
        if 0 <= yi < int(k):
            positives[yi] += 1
            if pi == yi:
                true_pos[yi] += 1
    recalls = []
    for cls_idx in range(int(k)):
        if positives[cls_idx] > 0:
            recalls.append(true_pos[cls_idx] / float(positives[cls_idx]))
    if not recalls:
        return float("nan")
    return float(np.std(np.asarray(recalls, dtype=np.float64)))


def _compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correct, dtype=np.float64).ravel()
    mask = np.isfinite(conf) & np.isfinite(corr)
    if np.sum(mask) <= 0:
        return float("nan")
    conf = conf[mask]
    corr = corr[mask]
    n = float(conf.shape[0])
    edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1, dtype=np.float64)
    ece = 0.0
    for bin_idx in range(len(edges) - 1):
        lo = float(edges[bin_idx])
        hi = float(edges[bin_idx + 1])
        if bin_idx == len(edges) - 2:
            bmask = (conf >= lo) & (conf <= hi)
        else:
            bmask = (conf >= lo) & (conf < hi)
        if not np.any(bmask):
            continue
        acc_b = float(np.mean(corr[bmask]))
        conf_b = float(np.mean(conf[bmask]))
        ece += (float(np.sum(bmask)) / n) * abs(acc_b - conf_b)
    return float(ece)


def _normalize_alpha(alpha: Any) -> str:
    f = _safe_float(alpha)
    if not math.isfinite(f):
        return str(alpha)
    return f"{f:g}"


def _rotations(k: int) -> List[Tuple[int, ...]]:
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(
    probs_seq: List[List[float]],
    permuted_indices: List[Tuple[int, ...]],
    k: int,
) -> np.ndarray:
    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            agg[p[j]] += letter_probs[j]
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


def _extract_task_from_dir(results_dir: str) -> Optional[str]:
    task_jsons = glob.glob(os.path.join(results_dir, "*_empirical_stage_analysis.json"))
    if len(task_jsons) == 1:
        return os.path.basename(task_jsons[0]).replace("_empirical_stage_analysis.json", "")
    return None


def _load_records(task_analysis_path: Optional[str], summary_paths: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    task_payload: Dict[str, Any] = {}
    if task_analysis_path and os.path.exists(task_analysis_path):
        task_payload = _load_json(task_analysis_path)
        records = list(task_payload.get("records") or [])
        if records:
            return records, task_payload
    records = [_load_json(path) for path in summary_paths]
    return records, task_payload


def _record_key(subject: Any, run_idx: Any, alpha: Any) -> Tuple[str, int, str]:
    return (str(subject), int(run_idx), _normalize_alpha(alpha))


def _collect_stage_arrays(trajectories: List[Dict[str, Any]]) -> Dict[int, Dict[str, np.ndarray]]:
    by_stage: Dict[int, Dict[str, np.ndarray]] = {}
    sample_count = len(trajectories)
    for sample_idx, row in enumerate(trajectories):
        stages = [int(x) for x in (row.get("decision_stages") or [])]
        confs = np.asarray(row.get("conf_by_stage") or [], dtype=np.float64)
        corrs = np.asarray(row.get("correct_by_stage") or [], dtype=np.float64)
        true_probs = np.asarray(row.get("true_prob_by_stage") or [], dtype=np.float64)
        preds = np.asarray(row.get("pred_by_stage") or [], dtype=np.float64)
        for idx, stage_id in enumerate(stages):
            if stage_id not in by_stage:
                by_stage[stage_id] = {
                    "conf": np.full((sample_count,), np.nan, dtype=np.float64),
                    "corr": np.full((sample_count,), np.nan, dtype=np.float64),
                    "true_prob": np.full((sample_count,), np.nan, dtype=np.float64),
                    "pred": np.full((sample_count,), np.nan, dtype=np.float64),
                }
            if idx < confs.shape[0]:
                by_stage[stage_id]["conf"][sample_idx] = float(confs[idx])
            if idx < corrs.shape[0]:
                by_stage[stage_id]["corr"][sample_idx] = float(corrs[idx])
            if idx < true_probs.shape[0]:
                by_stage[stage_id]["true_prob"][sample_idx] = float(true_probs[idx])
            if idx < preds.shape[0]:
                by_stage[stage_id]["pred"][sample_idx] = float(preds[idx])
    return by_stage


def _reliability_bin_summary(trajectories: List[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    if not trajectories:
        return {}
    by_stage = _collect_stage_arrays(trajectories)
    edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1, dtype=np.float64)
    out: Dict[str, Any] = {}
    for stage_id in sorted(by_stage.keys()):
        conf = np.asarray(by_stage[stage_id]["conf"], dtype=np.float64)
        corr = np.asarray(by_stage[stage_id]["corr"], dtype=np.float64)
        valid = np.isfinite(conf) & np.isfinite(corr)
        conf = conf[valid]
        corr = corr[valid]
        n = int(conf.size)
        if n == 0:
            continue
        bins_payload: Dict[str, Any] = {}
        for bin_idx in range(len(edges) - 1):
            lo = float(edges[bin_idx])
            hi = float(edges[bin_idx + 1])
            if bin_idx == len(edges) - 2:
                mask = (conf >= lo) & (conf <= hi)
            else:
                mask = (conf >= lo) & (conf < hi)
            label = f"{int(round(lo * 100)):02d}-{int(round(hi * 100)):02d}"
            count = int(np.sum(mask))
            if count == 0:
                bins_payload[label] = {
                    "count": 0,
                    "ratio": 0.0,
                    "acc": float("nan"),
                    "avg_conf": float("nan"),
                    "gap": float("nan"),
                }
                continue
            acc_b = float(np.mean(corr[mask]))
            conf_b = float(np.mean(conf[mask]))
            bins_payload[label] = {
                "count": count,
                "ratio": float(count / n),
                "acc": acc_b,
                "avg_conf": conf_b,
                "gap": float(acc_b - conf_b),
            }
        out[str(stage_id)] = {
            "n_samples": n,
            "bin_edges": [float(x) for x in edges.tolist()],
            "bins": bins_payload,
        }
    return out


def _percentile_bin_gain_summary(trajectories: List[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    if not trajectories:
        return {}
    by_stage = _collect_stage_arrays(trajectories)
    ordered_stages = sorted(by_stage.keys())
    out: Dict[str, Any] = {}
    quantiles = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1, dtype=np.float64)
    for prev_stage, next_stage in zip(ordered_stages[:-1], ordered_stages[1:]):
        conf_prev = np.asarray(by_stage[prev_stage]["conf"], dtype=np.float64)
        corr_prev = np.asarray(by_stage[prev_stage]["corr"], dtype=np.float64)
        corr_next = np.asarray(by_stage[next_stage]["corr"], dtype=np.float64)
        valid = np.isfinite(conf_prev) & np.isfinite(corr_prev) & np.isfinite(corr_next)
        conf_prev = conf_prev[valid]
        corr_prev = corr_prev[valid]
        corr_next = corr_next[valid]
        if conf_prev.size == 0:
            continue
        cut_vals = np.quantile(conf_prev, quantiles)
        trans_key = f"{int(prev_stage)}->{int(next_stage)}"
        bins_payload: Dict[str, Any] = {}
        for bin_idx in range(len(quantiles) - 1):
            q_lo = float(quantiles[bin_idx])
            q_hi = float(quantiles[bin_idx + 1])
            th_lo = float(cut_vals[bin_idx])
            th_hi = float(cut_vals[bin_idx + 1])
            if bin_idx == len(quantiles) - 2:
                mask = (conf_prev >= th_lo) & (conf_prev <= th_hi)
            else:
                mask = (conf_prev >= th_lo) & (conf_prev < th_hi)
            label = f"{int(round(q_lo * 100)):02d}-{int(round(q_hi * 100)):02d}"
            count = int(np.sum(mask))
            if count == 0:
                bins_payload[label] = {
                    "count": 0,
                    "ratio": 0.0,
                    "threshold_lo": th_lo,
                    "threshold_hi": th_hi,
                    "avg_conf_from": float("nan"),
                    "acc_from": float("nan"),
                    "acc_to": float("nan"),
                    "delta_acc": float("nan"),
                    "w2c_count": 0,
                    "c2w_count": 0,
                    "w2c": float("nan"),
                    "c2w": float("nan"),
                }
                continue
            c_prev = corr_prev[mask]
            c_next = corr_next[mask]
            conf_bin = conf_prev[mask]
            w2c_mask = (c_prev < 0.5) & (c_next >= 0.5)
            c2w_mask = (c_prev >= 0.5) & (c_next < 0.5)
            bins_payload[label] = {
                "count": count,
                "ratio": float(count / conf_prev.size),
                "threshold_lo": th_lo,
                "threshold_hi": th_hi,
                "avg_conf_from": float(np.mean(conf_bin)),
                "acc_from": float(np.mean(c_prev)),
                "acc_to": float(np.mean(c_next)),
                "delta_acc": float(np.mean(c_next - c_prev)),
                "w2c_count": int(np.sum(w2c_mask)),
                "c2w_count": int(np.sum(c2w_mask)),
                "w2c": float(np.mean(w2c_mask.astype(np.float64))),
                "c2w": float(np.mean(c2w_mask.astype(np.float64))),
            }
        out[trans_key] = {
            "n_samples": int(conf_prev.size),
            "percentile_edges": [float(x) for x in quantiles.tolist()],
            "bins": bins_payload,
        }
    return out


def _percentile_reliability_summary(trajectories: List[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    if not trajectories:
        return {}
    by_stage = _collect_stage_arrays(trajectories)
    quantiles = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1, dtype=np.float64)
    out: Dict[str, Any] = {}
    for stage_id in sorted(by_stage.keys()):
        conf = np.asarray(by_stage[stage_id]["conf"], dtype=np.float64)
        corr = np.asarray(by_stage[stage_id]["corr"], dtype=np.float64)
        valid = np.isfinite(conf) & np.isfinite(corr)
        conf = conf[valid]
        corr = corr[valid]
        n = int(conf.size)
        if n == 0:
            continue
        cut_vals = np.quantile(conf, quantiles)
        bins_payload: Dict[str, Any] = {}
        ece_val = 0.0
        for bin_idx in range(len(quantiles) - 1):
            q_lo = float(quantiles[bin_idx])
            q_hi = float(quantiles[bin_idx + 1])
            th_lo = float(cut_vals[bin_idx])
            th_hi = float(cut_vals[bin_idx + 1])
            if bin_idx == len(quantiles) - 2:
                mask = (conf >= th_lo) & (conf <= th_hi)
            else:
                mask = (conf >= th_lo) & (conf < th_hi)
            label = f"{int(round(q_lo * 100)):02d}-{int(round(q_hi * 100)):02d}"
            count = int(np.sum(mask))
            if count == 0:
                bins_payload[label] = {
                    "count": 0,
                    "ratio": 0.0,
                    "threshold_lo": th_lo,
                    "threshold_hi": th_hi,
                    "acc": float("nan"),
                    "avg_conf": float("nan"),
                    "gap": float("nan"),
                }
                continue
            acc_b = float(np.mean(corr[mask]))
            conf_b = float(np.mean(conf[mask]))
            ratio = float(count / n)
            gap = float(acc_b - conf_b)
            ece_val += ratio * abs(gap)
            bins_payload[label] = {
                "count": count,
                "ratio": ratio,
                "threshold_lo": th_lo,
                "threshold_hi": th_hi,
                "acc": acc_b,
                "avg_conf": conf_b,
                "gap": gap,
            }
        out[str(stage_id)] = {
            "n_samples": n,
            "ece": float(ece_val),
            "percentile_edges": [float(x) for x in quantiles.tolist()],
            "bins": bins_payload,
        }
    return out


def _simulate_policy_stops_from_trajectories(
    trajectories: List[Dict[str, Any]],
    *,
    k: int,
    sweep_mode: str,
    sweep_value: float,
    stage_schedule: str = "flat",
    stage_gamma: float = 0.5,
    percentile_mode: str = "online",
    max_stop_stage: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = sorted(list(trajectories), key=lambda r: int(r.get("sample_pos", 0)))
    if not rows:
        return []
    max_stage = max(
        max([int(x) for x in (row.get("decision_stages") or [1])])
        for row in rows
    )
    stage_schedule = str(stage_schedule or "flat").strip().lower()
    if stage_schedule not in {"flat", "sqrt"}:
        stage_schedule = "flat"
    percentile_mode = str(percentile_mode or "online").strip().lower()
    if percentile_mode not in {"online", "fixed_prefix"}:
        percentile_mode = "online"
    gamma = float(stage_gamma) if np.isfinite(float(stage_gamma)) and float(stage_gamma) > 0.0 else 0.5
    histories: List[List[float]] = [[] for _ in range(int(max_stage))]
    fixed_thresholds: List[float] = [0.0 for _ in range(int(max_stage))]
    sweep_mode_norm = str(sweep_mode or "percentile").strip().lower()
    if sweep_mode_norm not in {"percentile", "confidence"}:
        sweep_mode_norm = "percentile"

    if sweep_mode_norm == "percentile" and percentile_mode == "fixed_prefix":
        fixed_histories: List[List[float]] = [[] for _ in range(int(max_stage))]
        for row in rows:
            if not bool(row.get("prefix_forced", False)):
                continue
            confs = [float(x) for x in (row.get("conf_by_stage") or [])]
            stages = [int(x) for x in (row.get("decision_stages") or [])]
            if len(confs) != len(stages):
                continue
            for local_idx, stage_id in enumerate(stages):
                fixed_histories[int(stage_id) - 1].append(float(confs[local_idx]))
        for stage_idx in range(int(max_stage)):
            stage_id = stage_idx + 1
            if stage_schedule == "sqrt":
                stage_value = float(sweep_value) / (float(stage_id) ** gamma)
            else:
                stage_value = float(sweep_value)
            q = max(0.0, min(1.0, stage_value / 100.0))
            hist = fixed_histories[stage_idx]
            fixed_thresholds[stage_idx] = float(np.quantile(np.asarray(hist, dtype=np.float64), q)) if hist else 0.0

    out: List[Dict[str, Any]] = []
    for row in rows:
        confs = [float(x) for x in (row.get("conf_by_stage") or [])]
        stages = [int(x) for x in (row.get("decision_stages") or [])]
        eligible_stages = [int(stage_id) for stage_id in stages if max_stop_stage is None or int(stage_id) <= int(max_stop_stage)]
        if not eligible_stages:
            eligible_stages = [int(stages[0])] if stages else [1]
        forced_prefix = bool(row.get("prefix_forced", False))
        stop_stage = int(eligible_stages[-1])
        if not forced_prefix:
            for local_idx, stage_id in enumerate(stages):
                if max_stop_stage is not None and int(stage_id) > int(max_stop_stage):
                    break
                if sweep_mode_norm == "percentile":
                    if percentile_mode == "fixed_prefix":
                        thr = float(fixed_thresholds[int(stage_id) - 1])
                    else:
                        hist = histories[int(stage_id) - 1]
                        if stage_schedule == "sqrt":
                            stage_value = float(sweep_value) / (float(stage_id) ** gamma)
                        else:
                            stage_value = float(sweep_value)
                        q = max(0.0, min(1.0, stage_value / 100.0))
                        thr = float(np.quantile(np.asarray(hist, dtype=np.float64), q)) if hist else 0.0
                else:
                    base_tau = float(sweep_value)
                    if stage_schedule == "sqrt":
                        chance = 1.0 / float(k)
                        thr = chance + (base_tau - chance) / (float(stage_id) ** gamma)
                    else:
                        thr = base_tau
                    thr = max(0.0, min(1.0, float(thr)))
                if float(confs[local_idx]) >= thr:
                    stop_stage = int(stage_id)
                    break
        stop_local_idx = stages.index(int(stop_stage))
        out.append({
            "sample_pos": int(row.get("sample_pos", len(out))),
            "sample_id": int(row.get("sample_id", len(out))),
            "stop_stage": int(stop_stage),
            "stop_local_idx": int(stop_local_idx),
        })
        if sweep_mode_norm == "percentile" and percentile_mode == "online":
            for local_idx, stage_id in enumerate(stages[: stop_local_idx + 1]):
                if max_stop_stage is not None and int(stage_id) > int(max_stop_stage):
                    break
                histories[int(stage_id) - 1].append(float(confs[local_idx]))
    return out


def _actual_routed_stage1_fourway_summary(
    trajectories: List[Dict[str, Any]],
    *,
    k: int,
    sweep_mode: str,
    sweep_values: List[float],
    stage_schedule: str = "flat",
    stage_gamma: float = 0.5,
    percentile_mode: str = "online",
) -> Dict[str, Any]:
    if not trajectories:
        return {}
    rows = sorted(list(trajectories), key=lambda r: int(r.get("sample_pos", 0)))
    stage_maps: List[Dict[int, int]] = []
    for row in rows:
        stages = [int(x) for x in (row.get("decision_stages") or [])]
        corrs = [int(x) for x in (row.get("correct_by_stage") or [])]
        stage_maps.append({int(stage_id): int(corrs[idx]) for idx, stage_id in enumerate(stages) if idx < len(corrs)})
    ordered_stages = sorted({stage_id for row in rows for stage_id in (row.get("decision_stages") or [])})
    if 1 not in ordered_stages:
        return {}
    sweep_mode_norm = str(sweep_mode or "percentile").strip().lower()
    if sweep_mode_norm not in {"percentile", "confidence"}:
        sweep_mode_norm = "percentile"
    out: Dict[str, Any] = {"sweep_key": "p" if sweep_mode_norm == "percentile" else "confidence", "points": {}}
    for sweep_value in sweep_values:
        policy_rows = _simulate_policy_stops_from_trajectories(
            rows,
            k=k,
            sweep_mode=sweep_mode_norm,
            sweep_value=float(sweep_value),
            stage_schedule=stage_schedule,
            stage_gamma=stage_gamma,
            percentile_mode=percentile_mode,
        )
        point_key = f"{float(sweep_value):g}"
        point_payload: Dict[str, Any] = {}
        for prev_stage, next_stage in zip(ordered_stages[:-1], ordered_stages[1:]):
            routed_indices = [i for i, prow in enumerate(policy_rows) if int(prow["stop_stage"]) > int(prev_stage)]
            if not routed_indices:
                point_payload[f"{int(prev_stage)}->{int(next_stage)}"] = {
                    "n_routed": 0,
                    "routed_ratio": 0.0,
                    "cc_count": 0,
                    "wc_count": 0,
                    "ww_count": 0,
                    "cw_count": 0,
                    "cc": float("nan"),
                    "wc": float("nan"),
                    "ww": float("nan"),
                    "cw": float("nan"),
                }
                continue
            c1 = np.asarray([stage_maps[i].get(1, 0) for i in routed_indices], dtype=np.float64)
            ct = np.asarray([stage_maps[i].get(int(next_stage), np.nan) for i in routed_indices], dtype=np.float64)
            valid = np.isfinite(c1) & np.isfinite(ct)
            c1 = c1[valid]
            ct = ct[valid]
            if c1.size == 0:
                continue
            cc_mask = (c1 >= 0.5) & (ct >= 0.5)
            wc_mask = (c1 < 0.5) & (ct >= 0.5)
            ww_mask = (c1 < 0.5) & (ct < 0.5)
            cw_mask = (c1 >= 0.5) & (ct < 0.5)
            denom = float(c1.size)
            point_payload[f"{int(prev_stage)}->{int(next_stage)}"] = {
                "n_routed": int(c1.size),
                "routed_ratio": float(c1.size / len(rows)),
                "cc_count": int(np.sum(cc_mask)),
                "wc_count": int(np.sum(wc_mask)),
                "ww_count": int(np.sum(ww_mask)),
                "cw_count": int(np.sum(cw_mask)),
                "cc": float(np.mean(cc_mask.astype(np.float64))),
                "wc": float(np.mean(wc_mask.astype(np.float64))),
                "ww": float(np.mean(ww_mask.astype(np.float64))),
                "cw": float(np.mean(cw_mask.astype(np.float64))),
            }
        out["points"][point_key] = point_payload
    return out


def _actual_policy_stage_metrics_summary(
    trajectories: List[Dict[str, Any]],
    *,
    k: int,
    sweep_mode: str,
    sweep_values: List[float],
    stage_schedule: str = "flat",
    stage_gamma: float = 0.5,
    percentile_mode: str = "online",
    ece_bins: int = 10,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    if not trajectories:
        return {}
    rows = sorted(list(trajectories), key=lambda r: int(r.get("sample_pos", 0)))
    ordered_stages = sorted({int(stage_id) for row in rows for stage_id in (row.get("decision_stages") or [])})
    sweep_mode_norm = str(sweep_mode or "percentile").strip().lower()
    if sweep_mode_norm not in {"percentile", "confidence"}:
        sweep_mode_norm = "percentile"
    out: Dict[str, Any] = {"sweep_key": "p" if sweep_mode_norm == "percentile" else "confidence", "points": {}}
    n_total = len(rows)
    for sweep_value in sweep_values:
        policy_rows = _simulate_policy_stops_from_trajectories(
            rows,
            k=k,
            sweep_mode=sweep_mode_norm,
            sweep_value=float(sweep_value),
            stage_schedule=stage_schedule,
            stage_gamma=stage_gamma,
            percentile_mode=percentile_mode,
        )
        point_key = f"{float(sweep_value):g}"
        stage_payload: Dict[str, Any] = {}
        final_conf = []
        final_corr = []
        final_true_prob = []
        for stage_id in ordered_stages:
            conf_vals: List[float] = []
            corr_vals: List[float] = []
            true_prob_vals: List[float] = []
            for row, prow in zip(rows, policy_rows):
                if int(prow["stop_stage"]) != int(stage_id):
                    continue
                stop_local_idx = int(prow["stop_local_idx"])
                conf_by_stage = row.get("conf_by_stage") or []
                corr_by_stage = row.get("correct_by_stage") or []
                true_prob_by_stage = row.get("true_prob_by_stage") or []
                if stop_local_idx >= len(conf_by_stage) or stop_local_idx >= len(corr_by_stage) or stop_local_idx >= len(true_prob_by_stage):
                    continue
                conf_val = float(conf_by_stage[stop_local_idx])
                corr_val = float(corr_by_stage[stop_local_idx])
                true_prob_val = float(true_prob_by_stage[stop_local_idx])
                conf_vals.append(conf_val)
                corr_vals.append(corr_val)
                true_prob_vals.append(true_prob_val)
                final_conf.append(conf_val)
                final_corr.append(corr_val)
                final_true_prob.append(true_prob_val)
            conf_arr = np.asarray(conf_vals, dtype=np.float64)
            corr_arr = np.asarray(corr_vals, dtype=np.float64)
            true_prob_arr = np.asarray(true_prob_vals, dtype=np.float64)
            count = int(conf_arr.size)
            stage_payload[str(stage_id)] = {
                "n_samples": count,
                "ratio": float(count / n_total) if n_total > 0 else float("nan"),
                "acc": float(np.mean(corr_arr)) if count > 0 else float("nan"),
                "nll": float(np.mean(-np.log(np.clip(true_prob_arr, eps, 1.0)))) if count > 0 else float("nan"),
                "avg_conf": float(np.mean(conf_arr)) if count > 0 else float("nan"),
                "ece": _compute_ece(conf_arr, corr_arr, n_bins=ece_bins) if count > 0 else float("nan"),
            }
        final_conf_arr = np.asarray(final_conf, dtype=np.float64)
        final_corr_arr = np.asarray(final_corr, dtype=np.float64)
        final_true_prob_arr = np.asarray(final_true_prob, dtype=np.float64)
        stage_payload["overall_final"] = {
            "n_samples": int(final_conf_arr.size),
            "ratio": 1.0 if final_conf_arr.size > 0 else float("nan"),
            "acc": float(np.mean(final_corr_arr)) if final_conf_arr.size > 0 else float("nan"),
            "nll": float(np.mean(-np.log(np.clip(final_true_prob_arr, eps, 1.0)))) if final_true_prob_arr.size > 0 else float("nan"),
            "avg_conf": float(np.mean(final_conf_arr)) if final_conf_arr.size > 0 else float("nan"),
            "ece": _compute_ece(final_conf_arr, final_corr_arr, n_bins=ece_bins) if final_conf_arr.size > 0 else float("nan"),
        }
        out["points"][point_key] = stage_payload
    return out


def _trajectory_top_tail_summary(
    trajectories: List[Dict[str, Any]],
    sweep_mode: str,
    sweep_values: List[float],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not trajectories:
        return grouped
    by_stage = _collect_stage_arrays(trajectories)
    ordered_stages = sorted(by_stage.keys())
    sweep_mode_norm = str(sweep_mode or "percentile").strip().lower()
    if sweep_mode_norm not in {"percentile", "confidence"}:
        sweep_mode_norm = "percentile"
    for prev_stage, next_stage in zip(ordered_stages[:-1], ordered_stages[1:]):
        conf_prev = np.asarray(by_stage[prev_stage]["conf"], dtype=np.float64)
        corr_prev = np.asarray(by_stage[prev_stage]["corr"], dtype=np.float64)
        corr_next = np.asarray(by_stage[next_stage]["corr"], dtype=np.float64)
        valid = np.isfinite(conf_prev) & np.isfinite(corr_prev) & np.isfinite(corr_next)
        conf_prev = conf_prev[valid]
        corr_prev = corr_prev[valid]
        corr_next = corr_next[valid]
        if conf_prev.size == 0:
            continue
        trans_key = f"{int(prev_stage)}->{int(next_stage)}"
        grouped[trans_key] = {}
        w2c_mask = (corr_prev < 0.5) & (corr_next >= 0.5)
        c2w_mask = (corr_prev >= 0.5) & (corr_next < 0.5)
        for sweep_value in sweep_values:
            sweep_f = float(sweep_value)
            if sweep_mode_norm == "percentile":
                q_low = max(0.0, min(1.0, sweep_f / 100.0))
                q_high = max(0.0, min(1.0, 1.0 - q_low))
                tau_low = float(np.quantile(conf_prev, q_low))
                tau_high = float(np.quantile(conf_prev, q_high))
                sweep_key = "p"
            else:
                tau_low = max(0.0, min(1.0, sweep_f))
                tau_high = max(0.0, min(1.0, 1.0 - sweep_f))
                sweep_key = "confidence"
            low_mask = conf_prev < tau_low
            top_mask = conf_prev >= tau_high
            grouped[trans_key][f"{sweep_f:g}"] = {
                sweep_key: float(sweep_f),
                "low_ratio": float(np.mean(low_mask.astype(np.float64))) if conf_prev.size > 0 else float("nan"),
                "top_ratio": float(np.mean(top_mask.astype(np.float64))) if conf_prev.size > 0 else float("nan"),
                "acc_low_from": _masked_mean(corr_prev, low_mask),
                "acc_low_to": _masked_mean(corr_next, low_mask),
                "delta_low": _masked_mean(corr_next - corr_prev, low_mask),
                "w2c_low": _masked_mean(w2c_mask.astype(np.float64), low_mask),
                "c2w_low": _masked_mean(c2w_mask.astype(np.float64), low_mask),
                "acc_top_from": _masked_mean(corr_prev, top_mask),
                "acc_top_to": _masked_mean(corr_next, top_mask),
                "delta_top": _masked_mean(corr_next - corr_prev, top_mask),
                "w2c_top": _masked_mean(w2c_mask.astype(np.float64), top_mask),
                "c2w_top": _masked_mean(c2w_mask.astype(np.float64), top_mask),
                "threshold_low": float(tau_low),
                "threshold_top": float(tau_high),
            }
    return grouped


def _parse_traj_file_key(path: str) -> Optional[Tuple[str, int, str]]:
    name = os.path.basename(path)
    m = re.match(r"(.+)_run(\d+)_empirical_alpha([^.]+)_trajectories\.jsonl$", name)
    if not m:
        return None
    subject = str(m.group(1))
    run_idx = int(m.group(2))
    alpha_key = _normalize_alpha(m.group(3))
    return (subject, run_idx, alpha_key)


def _stage1_reference_summary(trajectories: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not trajectories:
        return {}
    by_stage: Dict[int, Dict[str, np.ndarray]] = {}
    sample_count = len(trajectories)
    for sample_idx, row in enumerate(trajectories):
        stages = [int(x) for x in (row.get("decision_stages") or [])]
        corrs = np.asarray(row.get("correct_by_stage") or [], dtype=np.float64)
        for idx, stage_id in enumerate(stages):
            if stage_id not in by_stage:
                by_stage[stage_id] = {"corr": np.full((sample_count,), np.nan, dtype=np.float64)}
            by_stage[stage_id]["corr"][sample_idx] = float(corrs[idx])
    if 1 not in by_stage:
        return {}
    corr_1 = np.asarray(by_stage[1]["corr"], dtype=np.float64)
    out: Dict[str, Dict[str, float]] = {}
    for stage_id in sorted(by_stage.keys()):
        if stage_id == 1:
            continue
        corr_t = np.asarray(by_stage[stage_id]["corr"], dtype=np.float64)
        valid = np.isfinite(corr_1) & np.isfinite(corr_t)
        c1 = corr_1[valid]
        ct = corr_t[valid]
        if c1.size == 0:
            continue
        w2c_mask = (c1 < 0.5) & (ct >= 0.5)
        c2w_mask = (c1 >= 0.5) & (ct < 0.5)
        out[str(stage_id)] = {
            "n_samples": int(c1.size),
            "acc_from": float(np.mean(c1)),
            "acc_to": float(np.mean(ct)),
            "delta_acc": float(np.mean(ct - c1)),
            "w2c_count": int(np.sum(w2c_mask)),
            "c2w_count": int(np.sum(c2w_mask)),
            "w2c": float(np.mean(w2c_mask.astype(np.float64))),
            "c2w": float(np.mean(c2w_mask.astype(np.float64))),
        }
    return out


def _summarize_stage_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        alpha_key = _normalize_alpha(rec.get("alpha"))
        summary = rec.get("summary") or {}
        stage_metrics = summary.get("stage_metrics") or {}
        for stage_id, vals in stage_metrics.items():
            if not isinstance(vals, dict):
                continue
            for metric in ["acc", "nll", "ece", "avg_conf", "conf_correct", "conf_wrong"]:
                grouped[alpha_key][str(stage_id)][metric].append(vals.get(metric))

    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for stage_id in sorted(grouped[alpha_key].keys(), key=_float_key):
            out[alpha_key][stage_id] = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][stage_id].items()
            }
    return out


def _summarize_transitions(
    records: List[Dict[str, Any]],
    trajectory_map: Optional[Dict[Tuple[str, int, str], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    thresh_grouped: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    sweep_key_by_alpha_transition: Dict[Tuple[str, str], str] = {}
    for rec in records:
        alpha_key = _normalize_alpha(rec.get("alpha"))
        transitions = ((rec.get("summary") or {}).get("transitions") or [])
        for trans in transitions:
            trans_key = f"{int(trans.get('from_stage', -1))}->{int(trans.get('to_stage', -1))}"
            for metric in ["delta_acc", "w2c", "c2w", "acc_from", "acc_to"]:
                grouped[alpha_key][trans_key][metric].append(trans.get(metric))
            for row in trans.get("threshold_analysis") or []:
                if not isinstance(row, dict):
                    continue
                if "p" in row:
                    sweep_key = "p"
                    sweep_value = float(row["p"])
                elif "confidence" in row:
                    sweep_key = "confidence"
                    sweep_value = float(row["confidence"])
                else:
                    continue
                sweep_name = f"{sweep_value:g}"
                sweep_key_by_alpha_transition[(alpha_key, trans_key)] = sweep_key
                for metric in [
                    "threshold",
                    "low_ratio",
                    "coverage",
                    "accepted_accuracy",
                    "acc_low_from",
                    "acc_low_to",
                    "delta_low",
                    "w2c_low",
                    "c2w_low",
                    "acc_high_from",
                    "acc_high_to",
                    "delta_high",
                    "w2c_high",
                    "c2w_high",
                ]:
                    thresh_grouped[alpha_key][trans_key][sweep_name][metric].append(row.get(metric))

    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for trans_key in sorted(grouped[alpha_key].keys(), key=lambda x: tuple(int(p) for p in x.split("->"))):
            entry = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][trans_key].items()
            }
            thresh_payload = {}
            for sweep_name in sorted(thresh_grouped[alpha_key][trans_key].keys(), key=_float_key):
                thresh_payload[sweep_name] = {
                    metric: _stats(values)
                    for metric, values in thresh_grouped[alpha_key][trans_key][sweep_name].items()
                }
            if thresh_payload:
                entry["threshold_sweep_key"] = sweep_key_by_alpha_transition.get((alpha_key, trans_key), "p")
                entry["threshold_analysis"] = thresh_payload
            if trajectory_map is not None:
                matching_top_stats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for rec in records:
                    rec_alpha = _normalize_alpha(rec.get("alpha"))
                    if rec_alpha != alpha_key:
                        continue
                    rec_key = _record_key(rec.get("subject"), rec.get("run_idx"), rec.get("alpha"))
                    traj_rows = trajectory_map.get(rec_key)
                    if not traj_rows:
                        continue
                    summary = rec.get("summary") or {}
                    top_summary = _trajectory_top_tail_summary(
                        traj_rows,
                        sweep_mode=str(summary.get("sweep_mode", "percentile")),
                        sweep_values=list(summary.get("sweep_values") or []),
                    )
                    row = top_summary.get(trans_key)
                    if not row:
                        continue
                    for point_key, point_vals in row.items():
                        matching_top_stats[point_key].append(point_vals)
                if matching_top_stats:
                    top_payload = {}
                    for point_key in sorted(matching_top_stats.keys(), key=_float_key):
                        metric_lists: Dict[str, List[float]] = defaultdict(list)
                        for row in matching_top_stats[point_key]:
                            for metric, val in row.items():
                                if metric in {"p", "confidence"}:
                                    continue
                                metric_lists[metric].append(val)
                        top_payload[point_key] = {
                            metric: _stats(vals)
                            for metric, vals in metric_lists.items()
                        }
                    if top_payload:
                        entry["top_threshold_analysis"] = top_payload
            out[alpha_key][trans_key] = entry
    return out


def _summarize_adaptive_points(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sweep_key_by_alpha: Dict[str, str] = {}
    for rec in records:
        alpha_key = _normalize_alpha(rec.get("alpha"))
        points = ((rec.get("summary") or {}).get("adaptive_points") or [])
        for point in points:
            if not isinstance(point, dict):
                continue
            if "p" in point:
                sweep_key = "p"
                sweep_value = float(point["p"])
            elif "confidence" in point:
                sweep_key = "confidence"
                sweep_value = float(point["confidence"])
            else:
                continue
            sweep_key_by_alpha[alpha_key] = sweep_key
            sweep_name = f"{sweep_value:g}"
            for metric in ["cost", "acc", "recall_std"]:
                grouped[alpha_key][sweep_name][metric].append(point.get(metric))
            for key, value in point.items():
                if isinstance(key, str) and key.startswith("n_stage_"):
                    grouped[alpha_key][sweep_name][key].append(value)

    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {
            "sweep_key": sweep_key_by_alpha.get(alpha_key, "p"),
            "points": {},
        }
        for sweep_name in sorted(grouped[alpha_key].keys(), key=_float_key):
            out[alpha_key]["points"][sweep_name] = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][sweep_name].items()
            }
    return out


def _extract_stage_metric_row(
    trajectories: List[Dict[str, Any]],
    stage_id: int,
    *,
    k: int,
    ece_bins: int = 10,
    eps: float = 1e-12,
) -> Dict[str, float]:
    by_stage = _collect_stage_arrays(trajectories)
    payload = by_stage.get(int(stage_id))
    if not payload:
        return {
            "n_samples": 0,
            "acc": float("nan"),
            "nll": float("nan"),
            "ece": float("nan"),
            "avg_conf": float("nan"),
            "conf_correct": float("nan"),
            "conf_wrong": float("nan"),
            "cost": float(stage_id),
        }
    conf = np.asarray(payload["conf"], dtype=np.float64)
    corr = np.asarray(payload["corr"], dtype=np.float64)
    true_prob = np.asarray(payload["true_prob"], dtype=np.float64)
    pred = np.asarray(payload["pred"], dtype=np.float64)
    labels = []
    for row in trajectories:
        labels.append(int(row.get("label_idx", -1)))
    valid = np.isfinite(conf) & np.isfinite(corr) & np.isfinite(true_prob)
    conf = conf[valid]
    corr = corr[valid]
    true_prob = true_prob[valid]
    pred = pred[valid]
    labels_arr = np.asarray(labels, dtype=np.int64)[valid] if len(labels) == valid.shape[0] else np.asarray([], dtype=np.int64)
    correct_mask = corr >= 0.5
    wrong_mask = ~correct_mask
    n = int(conf.size)
    return {
        "n_samples": n,
        "acc": float(np.mean(corr)) if n > 0 else float("nan"),
        "nll": float(np.mean(-np.log(np.clip(true_prob, eps, 1.0)))) if n > 0 else float("nan"),
        "ece": _compute_ece(conf, corr, n_bins=ece_bins) if n > 0 else float("nan"),
        "avg_conf": float(np.mean(conf)) if n > 0 else float("nan"),
        "recall_std": _recall_std(labels_arr.tolist(), pred.astype(np.int64).tolist(), int(k)) if n > 0 and labels_arr.size == pred.size else float("nan"),
        "conf_correct": _masked_mean(conf, correct_mask),
        "conf_wrong": _masked_mean(conf, wrong_mask),
        "cost": float(stage_id),
    }


def _simulate_policy_overall_metrics(
    trajectories: List[Dict[str, Any]],
    *,
    k: int,
    sweep_mode: str,
    sweep_value: float,
    stage_schedule: str,
    stage_gamma: float,
    percentile_mode: str,
    max_stop_stage: Optional[int] = None,
    ece_bins: int = 10,
    eps: float = 1e-12,
) -> Dict[str, float]:
    rows = sorted(list(trajectories), key=lambda r: int(r.get("sample_pos", 0)))
    policy_rows = _simulate_policy_stops_from_trajectories(
        rows,
        k=k,
        sweep_mode=sweep_mode,
        sweep_value=float(sweep_value),
        stage_schedule=stage_schedule,
        stage_gamma=stage_gamma,
        percentile_mode=percentile_mode,
        max_stop_stage=max_stop_stage,
    )
    final_conf = []
    final_corr = []
    final_true_prob = []
    final_pred = []
    final_label = []
    stage_counts = Counter()
    total_cost = 0.0
    for row, prow in zip(rows, policy_rows):
        stop_stage = int(prow["stop_stage"])
        stop_local_idx = int(prow["stop_local_idx"])
        conf_by_stage = row.get("conf_by_stage") or []
        corr_by_stage = row.get("correct_by_stage") or []
        true_prob_by_stage = row.get("true_prob_by_stage") or []
        pred_by_stage = row.get("pred_by_stage") or []
        label_idx = int(row.get("label_idx", -1))
        if stop_local_idx >= len(conf_by_stage) or stop_local_idx >= len(corr_by_stage) or stop_local_idx >= len(true_prob_by_stage) or stop_local_idx >= len(pred_by_stage):
            continue
        final_conf.append(float(conf_by_stage[stop_local_idx]))
        final_corr.append(float(corr_by_stage[stop_local_idx]))
        final_true_prob.append(float(true_prob_by_stage[stop_local_idx]))
        final_pred.append(int(pred_by_stage[stop_local_idx]))
        final_label.append(int(label_idx))
        stage_counts[int(stop_stage)] += 1
        total_cost += float(stop_stage)
    conf_arr = np.asarray(final_conf, dtype=np.float64)
    corr_arr = np.asarray(final_corr, dtype=np.float64)
    true_prob_arr = np.asarray(final_true_prob, dtype=np.float64)
    n = int(conf_arr.size)
    return {
        "n_samples": n,
        "acc": float(np.mean(corr_arr)) if n > 0 else float("nan"),
        "nll": float(np.mean(-np.log(np.clip(true_prob_arr, eps, 1.0)))) if n > 0 else float("nan"),
        "ece": _compute_ece(conf_arr, corr_arr, n_bins=ece_bins) if n > 0 else float("nan"),
        "avg_conf": float(np.mean(conf_arr)) if n > 0 else float("nan"),
        "recall_std": _recall_std(final_label, final_pred, int(k)) if n > 0 else float("nan"),
        "cost": float(total_cost / n) if n > 0 else float("nan"),
        "routing": {f"n_stage_{int(stage_id)}": int(count) for stage_id, count in sorted(stage_counts.items())},
    }


def _summarize_ablation_metrics(
    records: List[Dict[str, Any]],
    trajectory_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    routing_grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sweep_key_by_alpha: Dict[str, str] = {}

    for rec in records:
        rec_key = _record_key(rec.get("subject"), rec.get("run_idx"), rec.get("alpha"))
        traj_rows = trajectory_map.get(rec_key)
        if not traj_rows:
            continue
        alpha_key = _normalize_alpha(rec.get("alpha"))
        summary = rec.get("summary") or {}
        sweep_mode = str(summary.get("sweep_mode", rec.get("sweep_mode", "percentile"))).strip().lower()
        sweep_values = list(summary.get("sweep_values") or [])
        k = int(summary.get("k", rec.get("k", 0)) or 0)
        ece_bins = int(summary.get("ece_bins", 10) or 10)
        if k <= 0:
            continue
        stage_schedule = str(rec.get("threshold_schedule", rec.get("stage_schedule", "flat")))
        stage_gamma = float(rec.get("threshold_gamma", rec.get("stage_gamma", 0.5)) or 0.5)
        percentile_mode = str(rec.get("percentile_mode", "online"))
        ordered_stages = sorted({int(x) for row in traj_rows for x in (row.get("decision_stages") or [])})
        if not ordered_stages:
            continue
        full_stage = int(ordered_stages[-1])
        flip_stage = 2 if 2 in ordered_stages else int(full_stage)

        always_flip = _extract_stage_metric_row(traj_rows, flip_stage, k=k, ece_bins=ece_bins)
        always_full = _extract_stage_metric_row(traj_rows, full_stage, k=k, ece_bins=ece_bins)
        for metric, val in always_flip.items():
            if metric == "n_samples":
                continue
            grouped[alpha_key]["always_flip"][metric].append(val)
        for metric, val in always_full.items():
            if metric == "n_samples":
                continue
            grouped[alpha_key]["always_full"][metric].append(val)

        if sweep_values:
            sweep_key = "p" if sweep_mode == "percentile" else "confidence"
            sweep_key_by_alpha[alpha_key] = sweep_key
            for sweep_value in sweep_values:
                point_key = f"{float(sweep_value):g}"
                adaptive_flip_only = _simulate_policy_overall_metrics(
                    traj_rows,
                    k=k,
                    sweep_mode=sweep_mode,
                    sweep_value=float(sweep_value),
                    stage_schedule=stage_schedule,
                    stage_gamma=stage_gamma,
                    percentile_mode=percentile_mode,
                    max_stop_stage=flip_stage,
                    ece_bins=ece_bins,
                )
                for metric, val in adaptive_flip_only.items():
                    if metric in {"n_samples", "routing"}:
                        continue
                    grouped[alpha_key][f"adaptive_flip_only::{point_key}"][metric].append(val)
                for rkey, rval in (adaptive_flip_only.get("routing") or {}).items():
                    routing_grouped[alpha_key][f"adaptive_flip_only::{point_key}"][rkey].append(rval)

    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        alpha_payload: Dict[str, Any] = {
            "always_flip": {
                metric: _stats(values)
                for metric, values in grouped[alpha_key].get("always_flip", {}).items()
            },
            "always_full": {
                metric: _stats(values)
                for metric, values in grouped[alpha_key].get("always_full", {}).items()
            },
        }
        adaptive_payload = {"sweep_key": sweep_key_by_alpha.get(alpha_key, "p"), "points": {}}
        for bucket, metrics in grouped[alpha_key].items():
            if not bucket.startswith("adaptive_flip_only::"):
                continue
            point_key = bucket.split("::", 1)[1]
            row = {
                metric: _stats(values)
                for metric, values in metrics.items()
            }
            for rkey, rvals in routing_grouped[alpha_key].get(bucket, {}).items():
                row[rkey] = _stats(rvals)
            adaptive_payload["points"][point_key] = row
        alpha_payload["adaptive_flip_only"] = adaptive_payload
        out[alpha_key] = alpha_payload
    return out


def _summarize_stage1_reference(
    records: List[Dict[str, Any]],
    trajectory_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        rec_key = _record_key(rec.get("subject"), rec.get("run_idx"), rec.get("alpha"))
        traj_rows = trajectory_map.get(rec_key)
        if not traj_rows:
            continue
        alpha_key = _normalize_alpha(rec.get("alpha"))
        per_rec = _stage1_reference_summary(traj_rows)
        for stage_key, row in per_rec.items():
            for metric, val in row.items():
                grouped[alpha_key][stage_key][metric].append(val)
    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for stage_key in sorted(grouped[alpha_key].keys(), key=_float_key):
            out[alpha_key][stage_key] = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][stage_key].items()
            }
    return out


def _summarize_reliability_bins(
    records: List[Dict[str, Any]],
    trajectory_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    sample_counts: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        rec_key = _record_key(rec.get("subject"), rec.get("run_idx"), rec.get("alpha"))
        traj_rows = trajectory_map.get(rec_key)
        if not traj_rows:
            continue
        alpha_key = _normalize_alpha(rec.get("alpha"))
        per_rec = _reliability_bin_summary(traj_rows, n_bins=n_bins)
        for stage_key, stage_payload in per_rec.items():
            sample_counts[alpha_key][stage_key].append(stage_payload.get("n_samples"))
            for bin_key, bin_row in (stage_payload.get("bins") or {}).items():
                for metric, val in bin_row.items():
                    grouped[alpha_key][stage_key][bin_key][metric].append(val)
    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for stage_key in sorted(grouped[alpha_key].keys(), key=_float_key):
            out[alpha_key][stage_key] = {
                "n_samples": _stats(sample_counts[alpha_key][stage_key]),
                "bins": {
                    bin_key: {
                        metric: _stats(values)
                        for metric, values in grouped[alpha_key][stage_key][bin_key].items()
                    }
                    for bin_key in sorted(grouped[alpha_key][stage_key].keys())
                },
            }
    return out


def _summarize_percentile_bin_gains(
    records: List[Dict[str, Any]],
    trajectory_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    sample_counts: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        rec_key = _record_key(rec.get("subject"), rec.get("run_idx"), rec.get("alpha"))
        traj_rows = trajectory_map.get(rec_key)
        if not traj_rows:
            continue
        alpha_key = _normalize_alpha(rec.get("alpha"))
        per_rec = _percentile_bin_gain_summary(traj_rows, n_bins=n_bins)
        for trans_key, trans_payload in per_rec.items():
            sample_counts[alpha_key][trans_key].append(trans_payload.get("n_samples"))
            for bin_key, bin_row in (trans_payload.get("bins") or {}).items():
                for metric, val in bin_row.items():
                    grouped[alpha_key][trans_key][bin_key][metric].append(val)
    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for trans_key in sorted(grouped[alpha_key].keys(), key=lambda x: tuple(int(p) for p in x.split("->"))):
            out[alpha_key][trans_key] = {
                "n_samples": _stats(sample_counts[alpha_key][trans_key]),
                "bins": {
                    bin_key: {
                        metric: _stats(values)
                        for metric, values in grouped[alpha_key][trans_key][bin_key].items()
                    }
                    for bin_key in sorted(grouped[alpha_key][trans_key].keys())
                },
            }
    return out


def _summarize_percentile_reliability(
    records: List[Dict[str, Any]],
    trajectory_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    ece_grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    sample_counts: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        rec_key = _record_key(rec.get("subject"), rec.get("run_idx"), rec.get("alpha"))
        traj_rows = trajectory_map.get(rec_key)
        if not traj_rows:
            continue
        alpha_key = _normalize_alpha(rec.get("alpha"))
        per_rec = _percentile_reliability_summary(traj_rows, n_bins=n_bins)
        for stage_key, stage_payload in per_rec.items():
            sample_counts[alpha_key][stage_key].append(stage_payload.get("n_samples"))
            ece_grouped[alpha_key][stage_key].append(stage_payload.get("ece"))
            for bin_key, bin_row in (stage_payload.get("bins") or {}).items():
                for metric, val in bin_row.items():
                    grouped[alpha_key][stage_key][bin_key][metric].append(val)
    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for stage_key in sorted(grouped[alpha_key].keys(), key=_float_key):
            out[alpha_key][stage_key] = {
                "n_samples": _stats(sample_counts[alpha_key][stage_key]),
                "ece": _stats(ece_grouped[alpha_key][stage_key]),
                "bins": {
                    bin_key: {
                        metric: _stats(values)
                        for metric, values in grouped[alpha_key][stage_key][bin_key].items()
                    }
                    for bin_key in sorted(grouped[alpha_key][stage_key].keys())
                },
            }
    return out


def _summarize_actual_policy_stage1_fourway(
    records: List[Dict[str, Any]],
    trajectory_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rec in records:
        rec_key = _record_key(rec.get("subject"), rec.get("run_idx"), rec.get("alpha"))
        traj_rows = trajectory_map.get(rec_key)
        if not traj_rows:
            continue
        alpha_key = _normalize_alpha(rec.get("alpha"))
        summary = rec.get("summary") or {}
        sweep_mode = str(summary.get("sweep_mode", rec.get("sweep_mode", "percentile")))
        sweep_values = list(summary.get("sweep_values") or [])
        k = int(summary.get("k", rec.get("k", 0)) or 0)
        if k <= 0:
            continue
        stage_schedule = str(rec.get("threshold_schedule", rec.get("stage_schedule", "flat")))
        stage_gamma = float(rec.get("threshold_gamma", rec.get("stage_gamma", 0.5)) or 0.5)
        percentile_mode = str(rec.get("percentile_mode", "online"))
        per_rec = _actual_routed_stage1_fourway_summary(
            traj_rows,
            k=k,
            sweep_mode=sweep_mode,
            sweep_values=sweep_values,
            stage_schedule=stage_schedule,
            stage_gamma=stage_gamma,
            percentile_mode=percentile_mode,
        )
        alpha_entry = out.setdefault(alpha_key, {
            "sweep_key": per_rec.get("sweep_key", "p"),
            "points": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        })
        alpha_entry["sweep_key"] = per_rec.get("sweep_key", alpha_entry.get("sweep_key", "p"))
        points_store = alpha_entry["points"]
        for point_key, point_payload in (per_rec.get("points") or {}).items():
            for trans_key, trans_row in (point_payload or {}).items():
                for metric, val in trans_row.items():
                    points_store[str(point_key)][str(trans_key)][str(metric)].append(val)
    finalized: Dict[str, Any] = {}
    for alpha_key in sorted(out.keys(), key=_float_key):
        alpha_entry = out[alpha_key]
        finalized[alpha_key] = {"sweep_key": alpha_entry.get("sweep_key", "p"), "points": {}}
        for point_key in sorted(alpha_entry["points"].keys(), key=_float_key):
            finalized[alpha_key]["points"][point_key] = {}
            for trans_key in sorted(alpha_entry["points"][point_key].keys(), key=lambda x: tuple(int(p) for p in x.split("->"))):
                finalized[alpha_key]["points"][point_key][trans_key] = {
                    metric: _stats(values)
                    for metric, values in alpha_entry["points"][point_key][trans_key].items()
                }
    return finalized


def _summarize_actual_policy_stage_metrics(
    records: List[Dict[str, Any]],
    trajectory_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rec in records:
        rec_key = _record_key(rec.get("subject"), rec.get("run_idx"), rec.get("alpha"))
        traj_rows = trajectory_map.get(rec_key)
        if not traj_rows:
            continue
        alpha_key = _normalize_alpha(rec.get("alpha"))
        summary = rec.get("summary") or {}
        sweep_mode = str(summary.get("sweep_mode", rec.get("sweep_mode", "percentile")))
        sweep_values = list(summary.get("sweep_values") or [])
        k = int(summary.get("k", rec.get("k", 0)) or 0)
        ece_bins = int(summary.get("ece_bins", 10) or 10)
        if k <= 0:
            continue
        stage_schedule = str(rec.get("threshold_schedule", rec.get("stage_schedule", "flat")))
        stage_gamma = float(rec.get("threshold_gamma", rec.get("stage_gamma", 0.5)) or 0.5)
        percentile_mode = str(rec.get("percentile_mode", "online"))
        per_rec = _actual_policy_stage_metrics_summary(
            traj_rows,
            k=k,
            sweep_mode=sweep_mode,
            sweep_values=sweep_values,
            stage_schedule=stage_schedule,
            stage_gamma=stage_gamma,
            percentile_mode=percentile_mode,
            ece_bins=ece_bins,
        )
        alpha_entry = out.setdefault(alpha_key, {
            "sweep_key": per_rec.get("sweep_key", "p"),
            "points": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        })
        alpha_entry["sweep_key"] = per_rec.get("sweep_key", alpha_entry.get("sweep_key", "p"))
        points_store = alpha_entry["points"]
        for point_key, point_payload in (per_rec.get("points") or {}).items():
            for stage_key, stage_row in (point_payload or {}).items():
                for metric, val in stage_row.items():
                    points_store[str(point_key)][str(stage_key)][str(metric)].append(val)
    finalized: Dict[str, Any] = {}
    for alpha_key in sorted(out.keys(), key=_float_key):
        alpha_entry = out[alpha_key]
        finalized[alpha_key] = {"sweep_key": alpha_entry.get("sweep_key", "p"), "points": {}}
        for point_key in sorted(alpha_entry["points"].keys(), key=_float_key):
            finalized[alpha_key]["points"][point_key] = {}
            for stage_key in sorted(alpha_entry["points"][point_key].keys(), key=lambda x: (x != "overall_final", _float_key(x))):
                finalized[alpha_key]["points"][point_key][stage_key] = {
                    metric: _stats(values)
                    for metric, values in alpha_entry["points"][point_key][stage_key].items()
                }
    return finalized


def _summarize_trajectories(traj_paths: List[str]) -> Dict[str, Any]:
    pattern_counts: Counter[str] = Counter()
    total_rows = 0
    prefix_forced = 0
    sample_id_counts: Counter[int] = Counter()
    for path in traj_paths:
        rows = _load_jsonl(path)
        total_rows += len(rows)
        for row in rows:
            if bool(row.get("prefix_forced", False)):
                prefix_forced += 1
            sample_id = row.get("sample_id")
            if isinstance(sample_id, int):
                sample_id_counts[int(sample_id)] += 1
            stages = row.get("decision_stages") or []
            pattern = "-".join(str(int(x)) for x in stages)
            pattern_counts[pattern] += 1
    return {
        "n_files": int(len(traj_paths)),
        "n_rows": int(total_rows),
        "prefix_forced_ratio": float(prefix_forced / total_rows) if total_rows > 0 else float("nan"),
        "n_unique_sample_ids": int(len(sample_id_counts)),
        "decision_stage_patterns": dict(sorted(pattern_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def _summarize_record_metadata(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_records": int(len(records)),
        "subjects": sorted({str(rec.get("subject", "")) for rec in records if rec.get("subject") is not None}),
        "run_indices": sorted({int(rec.get("run_idx", -1)) for rec in records if rec.get("run_idx") is not None}),
        "alphas": sorted({_safe_float(rec.get("alpha")) for rec in records if math.isfinite(_safe_float(rec.get("alpha")))}),
        "transition_modes": dict(Counter(str(rec.get("transition_mode", "")) for rec in records if rec.get("transition_mode") is not None)),
        "residual_models": dict(Counter(str(rec.get("residual_model", "")) for rec in records if rec.get("residual_model") is not None)),
        "threshold_schedules": dict(Counter(str(rec.get("threshold_schedule", "")) for rec in records if rec.get("threshold_schedule") is not None)),
        "percentile_modes": dict(Counter(str(rec.get("percentile_mode", "")) for rec in records if rec.get("percentile_mode") is not None)),
    }


def _summarize_points_payload(points_path: Optional[str]) -> Dict[str, Any]:
    if not points_path or not os.path.exists(points_path):
        return {}
    payload = _load_json(points_path)
    empirical = ((payload.get("curves") or {}).get("empirical_pride") or {})
    out = {
        "path": points_path,
        "version": payload.get("version"),
        "empirical_pride": {},
    }
    if empirical:
        out["empirical_pride"] = {
            "sweep_mode": empirical.get("sweep_mode"),
            "percentile_mode": empirical.get("percentile_mode", "online"),
            "residual_model": empirical.get("residual_model"),
            "transition_mode": empirical.get("transition_mode"),
            "selection_policy": empirical.get("selection_policy"),
            "selected_sequence_name": empirical.get("selected_sequence_name"),
            "selected_action_sequence": empirical.get("selected_action_sequence"),
            "threshold_schedule": empirical.get("threshold_schedule"),
            "threshold_gamma": empirical.get("threshold_gamma"),
            "empirical_prefix_fractions": empirical.get("empirical_prefix_fractions") or empirical.get("pride_prefix_fractions"),
            "by_alpha_selection": {
                alpha_key: ((alpha_payload or {}).get("selection") or {})
                for alpha_key, alpha_payload in sorted(((empirical.get("by_alpha") or {}).items()), key=lambda kv: _float_key(kv[0]))
            },
        }
    return out


def _find_baseline_transition_files(results_dir: str, task: str, result_tag: Optional[str]) -> List[Tuple[str, str, str]]:
    parent_dir = os.path.dirname(os.path.abspath(results_dir))
    base_name = os.path.basename(os.path.abspath(results_dir))
    prefix = f"{task}_full"
    suffix = ""
    if base_name.startswith(prefix):
        suffix = base_name[len(prefix):]
    elif base_name.startswith(task):
        suffix = base_name[len(task):]
    subject_pattern = "*.jsonl"

    def _glob_dir(mode_prefix: str) -> List[str]:
        pat = os.path.join(parent_dir, f"{task}{mode_prefix}{suffix}", subject_pattern)
        return sorted(glob.glob(pat))

    base_files = _glob_dir("")
    cyclic_files = _glob_dir("_cyclic")
    # Keep only subject files, not curve files.
    def _filter_subject_files(paths: List[str]) -> Dict[str, str]:
        out = {}
        for path in paths:
            name = os.path.basename(path)
            if name.endswith("_curve.jsonl") or name.endswith("_pride_curve.jsonl"):
                continue
            if name.startswith(f"{task}_") and (name.endswith("_curve.jsonl") or name.endswith("_pride_curve.jsonl")):
                continue
            subject = name[:-6]
            out[subject] = path
        return out
    base_map = _filter_subject_files(base_files)
    cyc_map = _filter_subject_files(cyclic_files)
    pairs = []
    for subject in sorted(set(base_map) & set(cyc_map)):
        pairs.append((subject, base_map[subject], cyc_map[subject]))
    return pairs


def _summarize_baseline_cyclic_transition(results_dir: str, task: str, result_tag: Optional[str]) -> Dict[str, Any]:
    pairs = _find_baseline_transition_files(results_dir, task, result_tag)
    if not pairs:
        return {}
    per_subject = []
    w2c_vals = []
    c2w_vals = []
    delta_vals = []
    acc_from_vals = []
    acc_to_vals = []
    for subject, base_path, cyc_path in pairs:
        base_rows = _load_result_jsonl(base_path)
        cyc_rows = _load_result_jsonl(cyc_path)
        if not base_rows or not cyc_rows:
            continue
        base_map = {int((row.get("data") or {}).get("idx", -1)): row for row in base_rows}
        cyc_map = {int((row.get("data") or {}).get("idx", -1)): row for row in cyc_rows}
        shared = sorted(set(base_map) & set(cyc_map))
        if not shared:
            continue
        base_corr = []
        cyc_corr = []
        for idx in shared:
            bdata = base_map[idx].get("data") or {}
            cdata = cyc_map[idx].get("data") or {}
            b_corr = bdata.get("correct")
            if b_corr is None:
                sampled = bdata.get("sampled")
                ideal = bdata.get("ideal")
                if sampled is None or ideal is None:
                    continue
                b_corr = (sampled == ideal)
            cyc_probs = cdata.get("probs") or []
            ideal = str(cdata.get("ideal", ""))
            if not isinstance(cyc_probs, list) or not cyc_probs:
                continue
            k = len(cyc_probs)
            perm_list = _rotations(k)
            agg = _aggregate_probs_over_permutations(cyc_probs, perm_list, k)
            pred_idx = int(np.argmax(agg))
            option_ids = list("ABCDE"[:k]) if k <= 5 else [str(i) for i in range(k)]
            c_corr = (option_ids[pred_idx] == ideal)
            base_corr.append(bool(b_corr))
            cyc_corr.append(bool(c_corr))
        if not base_corr:
            continue
        base_arr = np.asarray(base_corr, dtype=np.float64)
        cyc_arr = np.asarray(cyc_corr, dtype=np.float64)
        w2c = float(np.mean((base_arr < 0.5) & (cyc_arr >= 0.5)))
        c2w = float(np.mean((base_arr >= 0.5) & (cyc_arr < 0.5)))
        delta = float(np.mean(cyc_arr - base_arr))
        acc_from = float(np.mean(base_arr))
        acc_to = float(np.mean(cyc_arr))
        per_subject.append({
            "subject": subject,
            "n_samples": int(base_arr.size),
            "acc_from": acc_from,
            "acc_to": acc_to,
            "delta_acc": delta,
            "w2c": w2c,
            "c2w": c2w,
        })
        w2c_vals.append(w2c)
        c2w_vals.append(c2w)
        delta_vals.append(delta)
        acc_from_vals.append(acc_from)
        acc_to_vals.append(acc_to)
    if not per_subject:
        return {}
    return {
        "n_subjects": int(len(per_subject)),
        "per_subject": per_subject,
        "acc_from": _stats(acc_from_vals),
        "acc_to": _stats(acc_to_vals),
        "delta_acc": _stats(delta_vals),
        "w2c": _stats(w2c_vals),
        "c2w": _stats(c2w_vals),
    }


def _summarize_cyclic_learned(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    learned_records = [rec for rec in records if str(rec.get("transition_mode", "")).strip() == "cyclic_learned"]
    if not learned_records:
        return {}

    seq_counts: Counter[str] = Counter()
    seq_by_subject: Dict[str, Counter[str]] = defaultdict(Counter)
    seq_by_alpha: Dict[str, Counter[str]] = defaultdict(Counter)
    actions_by_name: Dict[str, List[str]] = {}
    cand_scores: Dict[str, List[float]] = defaultdict(list)

    for rec in learned_records:
        subject = str(rec.get("subject", ""))
        alpha_key = _normalize_alpha(rec.get("alpha"))
        seq_name = str(rec.get("selected_sequence_name", "")).strip()
        if seq_name:
            seq_counts[seq_name] += 1
            seq_by_subject[subject][seq_name] += 1
            seq_by_alpha[alpha_key][seq_name] += 1
            if seq_name not in actions_by_name:
                actions_by_name[seq_name] = list(rec.get("selected_action_sequence") or [])
        for row in rec.get("candidate_sequence_scores") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            score = _safe_float(row.get("score"))
            if name and math.isfinite(score):
                cand_scores[name].append(score)
            if name and name not in actions_by_name:
                actions_by_name[name] = list(row.get("actions") or [])

    return {
        "n_records": int(len(learned_records)),
        "selected_sequence_counts": dict(sorted(seq_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "selected_sequence_by_subject": {
            subject: dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
            for subject, counter in sorted(seq_by_subject.items())
        },
        "selected_sequence_by_alpha": {
            alpha_key: dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
            for alpha_key, counter in sorted(seq_by_alpha.items(), key=lambda kv: _float_key(kv[0]))
        },
        "actions_by_sequence": actions_by_name,
        "candidate_score_summary": {
            name: _stats(scores)
            for name, scores in sorted(cand_scores.items())
        },
    }


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    meta = report.get("metadata") or {}
    lines.append(f"# Empirical Analysis Report: {meta.get('task', 'unknown')}")
    lines.append("")
    lines.append(f"- `results_dir`: `{meta.get('results_dir', '')}`")
    lines.append(f"- `model_name`: `{meta.get('model_name', '')}`")
    lines.append(f"- `option_id_set`: `{meta.get('option_id_set', '')}`")
    lines.append(f"- `result_tag`: `{meta.get('result_tag', '')}`")
    lines.append("")

    overview = report.get("record_overview") or {}
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- `n_records`: {overview.get('n_records', 0)}")
    lines.append(f"- `subjects`: {', '.join(overview.get('subjects', []))}")
    lines.append(f"- `alphas`: {', '.join(str(x) for x in overview.get('alphas', []))}")
    lines.append(f"- `transition_modes`: `{json.dumps(overview.get('transition_modes', {}), ensure_ascii=False)}`")
    lines.append("")

    traj = report.get("trajectory_summary") or {}
    if traj:
        lines.append("## Trajectories")
        lines.append("")
        lines.append(f"- `n_files`: {traj.get('n_files', 0)}")
        lines.append(f"- `n_rows`: {traj.get('n_rows', 0)}")
        lines.append(f"- `prefix_forced_ratio`: {traj.get('prefix_forced_ratio', float('nan')):.4f}")
        lines.append(f"- `decision_stage_patterns`: `{json.dumps(traj.get('decision_stage_patterns', {}), ensure_ascii=False)}`")
        lines.append("")

    baseline = report.get("baseline_cyclic_transition") or {}
    if baseline:
        lines.append("## Baseline Cyclic")
        lines.append("")
        lines.append(
            f"- `delta_acc`: {((baseline.get('delta_acc') or {}).get('mean', float('nan'))):.4f}, "
            f"`w2c`: {((baseline.get('w2c') or {}).get('mean', float('nan'))):.4f}, "
            f"`c2w`: {((baseline.get('c2w') or {}).get('mean', float('nan'))):.4f}"
        )
        lines.append("")

    stage1_ref = report.get("stage1_reference_summary") or {}
    if stage1_ref:
        lines.append("## Stage1 Reference")
        lines.append("")
        for alpha_key, alpha_vals in sorted(stage1_ref.items(), key=lambda kv: _float_key(kv[0])):
            if "2" not in alpha_vals:
                continue
            row = alpha_vals["2"]
            lines.append(
                f"- alpha `{alpha_key}` | `1->2` w2c={((row.get('w2c') or {}).get('mean', float('nan'))):.4f}, "
                f"c2w={((row.get('c2w') or {}).get('mean', float('nan'))):.4f}, "
                f"w2c_count={((row.get('w2c_count') or {}).get('mean', float('nan'))):.2f}, "
                f"c2w_count={((row.get('c2w_count') or {}).get('mean', float('nan'))):.2f}"
            )
        lines.append("")

    learned = report.get("cyclic_learned_summary") or {}
    if learned:
        lines.append("## Cyclic Learned")
        lines.append("")
        lines.append(f"- `n_records`: {learned.get('n_records', 0)}")
        lines.append(f"- `selected_sequence_counts`: `{json.dumps(learned.get('selected_sequence_counts', {}), ensure_ascii=False)}`")
        lines.append("")

    lines.append("## Stage Metrics")
    lines.append("")
    for alpha_key, alpha_payload in sorted((report.get("stage_metric_summary") or {}).items(), key=lambda kv: _float_key(kv[0])):
        lines.append(f"### alpha={alpha_key}")
        for stage_id, stage_payload in sorted(alpha_payload.items(), key=lambda kv: _float_key(kv[0])):
            acc = ((stage_payload.get("acc") or {}).get("mean"))
            nll = ((stage_payload.get("nll") or {}).get("mean"))
            ece = ((stage_payload.get("ece") or {}).get("mean"))
            conf = ((stage_payload.get("avg_conf") or {}).get("mean"))
            lines.append(
                f"- stage `{stage_id}`: acc={acc:.4f} nll={nll:.4f} ece={ece:.4f} avg_conf={conf:.4f}"
                if all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in [acc, nll, ece, conf])
                else f"- stage `{stage_id}`"
            )
        lines.append("")
    rel_bins = report.get("reliability_bin_summary") or {}
    if rel_bins:
        lines.append("## Reliability Bins")
        lines.append("")
        for alpha_key, alpha_payload in sorted(rel_bins.items(), key=lambda kv: _float_key(kv[0])):
            stage_payload = (alpha_payload or {}).get("1") or {}
            bins_payload = stage_payload.get("bins") or {}
            if not bins_payload:
                continue
            lines.append(f"- alpha `{alpha_key}` stage `1` bins available: {', '.join(sorted(bins_payload.keys()))}")
        lines.append("")
    pct_rel = report.get("percentile_reliability_summary") or {}
    if pct_rel:
        lines.append("## Percentile Reliability")
        lines.append("")
        for alpha_key, alpha_payload in sorted(pct_rel.items(), key=lambda kv: _float_key(kv[0])):
            stage_payload = (alpha_payload or {}).get("1") or {}
            ece = ((stage_payload.get("ece") or {}).get("mean", float("nan")))
            if math.isfinite(_safe_float(ece)):
                lines.append(f"- alpha `{alpha_key}` stage `1` percentile-bin ece={float(ece):.4f}")
        lines.append("")
    pct_bins = report.get("percentile_bin_gain_summary") or {}
    if pct_bins:
        lines.append("## Percentile Bin Gains")
        lines.append("")
        for alpha_key, alpha_payload in sorted(pct_bins.items(), key=lambda kv: _float_key(kv[0])):
            trans_payload = (alpha_payload or {}).get("1->2") or {}
            bins_payload = trans_payload.get("bins") or {}
            if not bins_payload:
                continue
            row = bins_payload.get("00-10") or {}
            delta = ((row.get("delta_acc") or {}).get("mean", float("nan")))
            lines.append(
                f"- alpha `{alpha_key}` | `1->2` bottom decile delta={delta:.4f}"
                if math.isfinite(_safe_float(delta))
                else f"- alpha `{alpha_key}` | `1->2` bins available"
            )
        lines.append("")
    routed = report.get("actual_policy_stage1_fourway_summary") or {}
    if routed:
        lines.append("## Actual Routed Stage1 Four-Way")
        lines.append("")
        for alpha_key, alpha_payload in sorted(routed.items(), key=lambda kv: _float_key(kv[0])):
            sweep_key = str((alpha_payload or {}).get("sweep_key", "p"))
            points = (alpha_payload or {}).get("points") or {}
            if not points:
                continue
            focus_key = "80" if "80" in points else next(iter(sorted(points.keys(), key=_float_key)), None)
            if focus_key is None:
                continue
            trans_row = ((points.get(focus_key) or {}).get("1->2") or {})
            wc = ((trans_row.get("wc") or {}).get("mean", float("nan")))
            cw = ((trans_row.get("cw") or {}).get("mean", float("nan")))
            lines.append(
                f"- alpha `{alpha_key}` | `{sweep_key}={focus_key}` `1->2` wc={float(wc):.4f} cw={float(cw):.4f}"
                if all(math.isfinite(_safe_float(x)) for x in [wc, cw]) else
                f"- alpha `{alpha_key}` | `{sweep_key}={focus_key}` routed summary available"
            )
        lines.append("")
    policy_stage = report.get("actual_policy_stage_metrics_summary") or {}
    if policy_stage:
        lines.append("## Actual Policy Stage Metrics")
        lines.append("")
        for alpha_key, alpha_payload in sorted(policy_stage.items(), key=lambda kv: _float_key(kv[0])):
            sweep_key = str((alpha_payload or {}).get("sweep_key", "p"))
            points = (alpha_payload or {}).get("points") or {}
            if not points:
                continue
            focus_key = "80" if "80" in points else next(iter(sorted(points.keys(), key=_float_key)), None)
            if focus_key is None:
                continue
            overall = ((points.get(focus_key) or {}).get("overall_final") or {})
            acc = ((overall.get("acc") or {}).get("mean", float("nan")))
            nll = ((overall.get("nll") or {}).get("mean", float("nan")))
            ece = ((overall.get("ece") or {}).get("mean", float("nan")))
            if all(math.isfinite(_safe_float(x)) for x in [acc, nll, ece]):
                lines.append(f"- alpha `{alpha_key}` | `{sweep_key}={focus_key}` overall_final acc={float(acc):.4f}, nll={float(nll):.4f}, ece={float(ece):.4f}")
        lines.append("")
    ablation = report.get("ablation_summary") or {}
    if ablation:
        lines.append("## Ablation")
        lines.append("")
        for alpha_key, alpha_payload in sorted(ablation.items(), key=lambda kv: _float_key(kv[0])):
            lines.append(f"### alpha={alpha_key}")
            for tag, label in (("always_flip", "Always Flip"), ("always_full", "Always Flip+Latin")):
                row = alpha_payload.get(tag) or {}
                acc = ((row.get("acc") or {}).get("mean", float("nan")))
                nll = ((row.get("nll") or {}).get("mean", float("nan")))
                ece = ((row.get("ece") or {}).get("mean", float("nan")))
                cost = ((row.get("cost") or {}).get("mean", float("nan")))
                rstd = ((row.get("recall_std") or {}).get("mean", float("nan")))
                if all(math.isfinite(_safe_float(x)) for x in [acc, nll, ece, cost, rstd]):
                    lines.append(f"- {label}: cost={float(cost):.4f}, acc={float(acc):.4f}, rstd={float(rstd):.4f}, nll={float(nll):.4f}, ece={float(ece):.4f}")
            adaptive = alpha_payload.get("adaptive_flip_only") or {}
            sweep_key = str(adaptive.get("sweep_key", "p"))
            points = adaptive.get("points") or {}
            if points:
                focus_key = "80" if "80" in points else next(iter(sorted(points.keys(), key=_float_key)), None)
                if focus_key is not None:
                    row = points.get(focus_key) or {}
                    acc = ((row.get("acc") or {}).get("mean", float("nan")))
                    nll = ((row.get("nll") or {}).get("mean", float("nan")))
                    ece = ((row.get("ece") or {}).get("mean", float("nan")))
                    cost = ((row.get("cost") or {}).get("mean", float("nan")))
                    rstd = ((row.get("recall_std") or {}).get("mean", float("nan")))
                    if all(math.isfinite(_safe_float(x)) for x in [acc, nll, ece, cost, rstd]):
                        lines.append(f"- Adaptive Flip-Only (`{sweep_key}={focus_key}`): cost={float(cost):.4f}, acc={float(acc):.4f}, rstd={float(rstd):.4f}, nll={float(nll):.4f}, ece={float(ece):.4f}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Direct path to results_<task>/.../<task>_full[_id-...][__result_tag]. If omitted, reconstruct from task/model options.")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (e.g. arc, csqa, mmlu). Required unless --results_dir is provided or can be inferred.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name suffix used in results dir (e.g. Olmo-3-7B-Instruct). Required unless --results_dir is provided.")
    parser.add_argument("--num_few_shot", type=int, default=0,
                        help="Few-shot count used in results dir reconstruction.")
    parser.add_argument("--option_id_set", type=str, default=None,
                        help="Option-id suffix used in results dir reconstruction.")
    parser.add_argument("--result_tag", type=str, default=None,
                        help="Optional result_tag suffix used in results dir reconstruction.")
    parser.add_argument("--setting", type=str, default="full",
                        help="Setting suffix for results dir reconstruction. Empirical analysis normally uses full.")
    parser.add_argument("--analyze_cyclic_learned", action="store_true",
                        help="Add cyclic_learned-specific action-sequence analysis when available.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional explicit output json path. Default: <results_dir>/<task>_empirical_analysis_report.json")
    parser.add_argument("--output_md", type=str, default=None,
                        help="Optional explicit output markdown path. Default: <results_dir>/<task>_empirical_analysis_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir
    if not results_dir:
        if not args.task or not args.model_name:
            raise SystemExit("Either --results_dir or both --task and --model_name are required.")
        results_dir = _build_results_dir(
            task=str(args.task),
            num_few_shot=int(args.num_few_shot),
            model_name=str(args.model_name),
            option_id_set=args.option_id_set,
            result_tag=args.result_tag,
            setting=str(args.setting or "full"),
        )
    results_dir = os.path.abspath(results_dir)
    if not os.path.isdir(results_dir):
        raise SystemExit(f"Results directory not found: {results_dir}")

    task = str(args.task or _extract_task_from_dir(results_dir) or "").strip()
    if not task:
        raise SystemExit("Could not infer task from results dir. Please provide --task.")

    task_analysis_path = os.path.join(results_dir, f"{task}_empirical_stage_analysis.json")
    points_path = os.path.join(results_dir, f"{task}_three_curves_points.json")
    empirical_analysis_dir = os.path.join(results_dir, "empirical_analysis")
    summary_paths = sorted(glob.glob(os.path.join(empirical_analysis_dir, "*_summary.json")))
    traj_paths = sorted(glob.glob(os.path.join(empirical_analysis_dir, "*_trajectories.jsonl")))
    trajectory_map: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for path in traj_paths:
        key = _parse_traj_file_key(path)
        if key is None:
            continue
        trajectory_map[key] = _load_jsonl(path)

    records, task_payload = _load_records(task_analysis_path if os.path.exists(task_analysis_path) else None, summary_paths)
    if not records:
        raise SystemExit(f"No empirical analysis records found under {results_dir}")

    first_record = records[0]
    model_name = str(
        (task_payload.get("model_name") if isinstance(task_payload, dict) else None)
        or args.model_name
        or first_record.get("model_name", "")
    )

    report: Dict[str, Any] = {
        "metadata": {
            "task": task,
            "results_dir": results_dir,
            "model_name": model_name,
            "num_few_shot": int(args.num_few_shot),
            "option_id_set": args.option_id_set,
            "result_tag": _sanitize_result_tag(args.result_tag),
            "task_analysis_path": task_analysis_path if os.path.exists(task_analysis_path) else None,
            "points_path": points_path if os.path.exists(points_path) else None,
            "empirical_analysis_dir": empirical_analysis_dir if os.path.isdir(empirical_analysis_dir) else None,
        },
        "source_files": {
            "n_summary_files": int(len(summary_paths)),
            "n_trajectory_files": int(len(traj_paths)),
            "summary_files": summary_paths,
            "trajectory_files": traj_paths,
        },
        "record_overview": _summarize_record_metadata(records),
        "stage_metric_summary": _summarize_stage_metrics(records),
        "reliability_bin_summary": _summarize_reliability_bins(records, trajectory_map),
        "percentile_reliability_summary": _summarize_percentile_reliability(records, trajectory_map),
        "transition_summary": _summarize_transitions(records, trajectory_map=trajectory_map),
        "percentile_bin_gain_summary": _summarize_percentile_bin_gains(records, trajectory_map),
        "actual_policy_stage1_fourway_summary": _summarize_actual_policy_stage1_fourway(records, trajectory_map),
        "actual_policy_stage_metrics_summary": _summarize_actual_policy_stage_metrics(records, trajectory_map),
        "stage1_reference_summary": _summarize_stage1_reference(records, trajectory_map),
        "ablation_summary": _summarize_ablation_metrics(records, trajectory_map),
        "adaptive_point_summary": _summarize_adaptive_points(records),
        "trajectory_summary": _summarize_trajectories(traj_paths),
        "points_payload_summary": _summarize_points_payload(points_path if os.path.exists(points_path) else None),
        "baseline_cyclic_transition": _summarize_baseline_cyclic_transition(results_dir, task, args.result_tag),
    }
    if args.analyze_cyclic_learned:
        report["cyclic_learned_summary"] = _summarize_cyclic_learned(records)

    output_json = args.output_json or os.path.join(results_dir, f"{task}_empirical_analysis_report.json")
    output_md = args.output_md or os.path.join(results_dir, f"{task}_empirical_analysis_report.md")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print(f"Saved empirical analysis report: {output_json}")
    print(f"Saved empirical analysis markdown: {output_md}")
    if args.analyze_cyclic_learned:
        learned = report.get("cyclic_learned_summary") or {}
        if learned:
            print(
                "Cyclic learned sequence counts: "
                + json.dumps(learned.get("selected_sequence_counts", {}), ensure_ascii=False)
            )
        else:
            print("Cyclic learned analysis requested, but no cyclic_learned records were found.")


if __name__ == "__main__":
    main()
