#!/usr/bin/env python3
"""
Rank-slot probability-delta analysis.

For each sample, we use the cyclic permutation ensemble as a pseudo-GT reference.
Let ref_dist be the cyclic-ensemble content-space distribution and let
content_(r) be the content ranked r-th in ref_dist.

For each rank r and displayed slot s, we collect:

    delta(r, s) = p_view(content_(r)) - p_ref(content_(r))

where the view is the cyclic permutation in which content_(r) is placed at slot s.

For k-way multiple choice, this yields k x k distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analyze_perm_noise import (
    _apply_offline_pride_to_results,
    _aggregate_probs_over_permutations,
    _cauchy_fit_report,
    _discover_cache_files,
    _gaussian_fit_report,
    _infer_perm_list,
    _laplace_fit_report,
    _read_results_file,
    _rotations,
    _try_import_matplotlib,
)


def _slot_labels(k: int) -> List[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 < int(k) <= len(alphabet):
        return list(alphabet[: int(k)])
    return [str(i) for i in range(int(k))]


def _rank_slot_bucket(k: int) -> Dict[str, Dict[str, List[float]]]:
    labels = _slot_labels(k)
    return {f"rank{r}": {slot: [] for slot in labels} for r in range(1, int(k) + 1)}


def _merge_rank_slot_bucket(dst: Dict[str, Dict[str, List[float]]], src: Dict[str, Dict[str, List[float]]]) -> None:
    for rank_key, slot_map in (src or {}).items():
        dst.setdefault(str(rank_key), {})
        for slot, vals in (slot_map or {}).items():
            dst[str(rank_key)].setdefault(str(slot), [])
            dst[str(rank_key)][str(slot)].extend(float(x) for x in (vals or []))


def _summarize_values(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    out = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    out["gaussian_fit"] = _gaussian_fit_report(arr)
    out["laplace_fit"] = _laplace_fit_report(arr)
    out["cauchy_fit"] = _cauchy_fit_report(arr)
    return out


def _summarize_rank_slot_bucket(bucket: Dict[str, Dict[str, List[float]]], k: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    labels = _slot_labels(k)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in range(1, int(k) + 1):
        rank_key = f"rank{r}"
        out[rank_key] = {}
        for slot in labels:
            out[rank_key][slot] = _summarize_values((bucket.get(rank_key, {}) or {}).get(slot, []))
    return out


def _analyze_results_file(results: List[dict], subject: str, run_idx: int) -> Tuple[Optional[int], Dict[str, Dict[str, List[float]]]]:
    if not results:
        return None, {}

    data0 = results[0].get("data", {}) or {}
    options = data0.get("options", None)
    if not isinstance(options, list) or not options:
        return None, {}
    k = len(options)
    labels = _slot_labels(k)
    probs0 = data0.get("probs", None)
    if not isinstance(probs0, list):
        return None, {}
    perm_count = len(probs0)
    perm_list, _ = _infer_perm_list(k, perm_count)
    cyc_perms = _rotations(k)
    cyc_idxs = [perm_list.index(p) for p in cyc_perms if p in perm_list]
    if len(cyc_idxs) != k:
        return None, {}

    bucket = _rank_slot_bucket(k)
    selected_perm_tuples = [tuple(int(x) for x in perm_list[idx]) for idx in cyc_idxs]

    for r in results:
        d = r.get("data", {}) or {}
        probs_seq = d.get("probs", None)
        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
            continue

        selected_dists = []
        for perm_idx in cyc_idxs:
            dist = _aggregate_probs_over_permutations([probs_seq[perm_idx]], [perm_list[perm_idx]], k)
            selected_dists.append(np.asarray(dist, dtype=np.float64))
        if not selected_dists:
            continue

        selected_dists_arr = np.asarray(selected_dists, dtype=np.float64)
        ref_dist = np.mean(selected_dists_arr, axis=0)
        rank_order = np.argsort(ref_dist)[::-1].tolist()

        for rank_idx, content_idx in enumerate(rank_order, start=1):
            ref_prob = float(ref_dist[int(content_idx)])
            for view_idx, perm in enumerate(selected_perm_tuples):
                slot_idx = next((j for j, c_idx in enumerate(perm) if int(c_idx) == int(content_idx)), -1)
                if slot_idx < 0:
                    continue
                slot_label = str(labels[int(slot_idx)])
                view_prob = float(selected_dists_arr[view_idx, int(content_idx)])
                delta = float(view_prob - ref_prob)
                bucket[f"rank{int(rank_idx)}"][slot_label].append(delta)

    return int(k), bucket


def _save_rank_slot_grid_plot(
    *,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    bucket: Dict[str, Dict[str, List[float]]],
    out_path: str,
    title: str,
) -> Optional[str]:
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    rank_keys = list(summary.keys())
    if not rank_keys:
        return None
    k = len(rank_keys)
    labels = list((summary.get("rank1", {}) or {}).keys())
    if not labels:
        return None

    fig, axes = plt.subplots(k, len(labels), figsize=(3.2 * len(labels), 2.6 * k), dpi=160, squeeze=False)
    for r_idx, rank_key in enumerate(rank_keys):
        for s_idx, slot in enumerate(labels):
            ax = axes[r_idx][s_idx]
            vals = np.asarray((bucket.get(rank_key, {}) or {}).get(slot, []), dtype=np.float64)
            fit = (summary.get(rank_key, {}) or {}).get(slot, {}) or {}
            gfit = fit.get("gaussian_fit", {}) or {}
            lfit = fit.get("laplace_fit", {}) or {}
            cfit = fit.get("cauchy_fit", {}) or {}
            if vals.size > 0:
                ax.hist(vals, bins=40, density=True, alpha=0.68, color="#6baed6")
                x_min = float(np.min(vals))
                x_max = float(np.max(vals))
                if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                    xs = np.linspace(x_min, x_max, 300)
                    mu = float(gfit.get("mean", float("nan")))
                    sigma = float(gfit.get("std", float("nan")))
                    if np.isfinite(mu) and np.isfinite(sigma) and sigma > 1e-12:
                        gpdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
                        ax.plot(xs, gpdf, color="#d62728", lw=1.5)
                    loc_l = float(lfit.get("loc", float("nan")))
                    scale_l = float(lfit.get("scale", float("nan")))
                    if np.isfinite(loc_l) and np.isfinite(scale_l) and scale_l > 1e-12:
                        lpdf = np.exp(-np.abs(xs - loc_l) / scale_l) / (2.0 * scale_l)
                        ax.plot(xs, lpdf, color="#2ca02c", lw=1.2, ls="--")
                    loc_c = float(cfit.get("loc", float("nan")))
                    scale_c = float(cfit.get("scale", float("nan")))
                    if np.isfinite(loc_c) and np.isfinite(scale_c) and scale_c > 1e-12:
                        cpdf = 1.0 / (math.pi * scale_c * (1.0 + ((xs - loc_c) / scale_c) ** 2))
                        ax.plot(xs, cpdf, color="#9467bd", lw=1.2, ls=":")
            ax.axvline(0.0, color="black", lw=1.0, ls=":")
            ax.set_title(
                f"{rank_key}-{slot}\n"
                f"n={int(fit.get('n', 0))}, mean={float(fit.get('mean', float('nan'))):.3f}\n"
                f"G/L/C KS={float(gfit.get('ks_to_fit', float('nan'))):.2f}/"
                f"{float(lfit.get('ks_to_fit', float('nan'))):.2f}/"
                f"{float(cfit.get('ks_to_fit', float('nan'))):.2f}",
                fontsize=8,
            )
            ax.grid(True, alpha=0.2)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_rank_slot_heatmap(
    *,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    out_path: str,
    value_key: str,
    title: str,
) -> Optional[str]:
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    rank_keys = list(summary.keys())
    if not rank_keys:
        return None
    labels = list((summary.get("rank1", {}) or {}).keys())
    if not labels:
        return None
    mat = np.full((len(rank_keys), len(labels)), np.nan, dtype=np.float64)
    for i, rank_key in enumerate(rank_keys):
        for j, slot in enumerate(labels):
            mat[i, j] = float(((summary.get(rank_key, {}) or {}).get(slot, {}) or {}).get(value_key, float("nan")))
    fig = plt.figure(figsize=(1.5 + 1.2 * len(labels), 1.5 + 1.0 * len(rank_keys)), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(rank_keys)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(rank_keys)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _compare_rank_slot_summaries(
    raw_summary: Dict[str, Dict[str, Dict[str, float]]],
    pride_summary: Dict[str, Dict[str, Dict[str, float]]],
    *,
    value_key: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    rank_keys = sorted(set(raw_summary.keys()) | set(pride_summary.keys()))
    for rank_key in rank_keys:
        out[rank_key] = {}
        slot_keys = sorted(set((raw_summary.get(rank_key, {}) or {}).keys()) | set((pride_summary.get(rank_key, {}) or {}).keys()))
        for slot in slot_keys:
            raw_val = float(((raw_summary.get(rank_key, {}) or {}).get(slot, {}) or {}).get(value_key, float("nan")))
            pride_val = float(((pride_summary.get(rank_key, {}) or {}).get(slot, {}) or {}).get(value_key, float("nan")))
            delta = float(pride_val - raw_val) if np.isfinite(raw_val) and np.isfinite(pride_val) else float("nan")
            out[rank_key][slot] = {
                "raw": raw_val,
                "pride": pride_val,
                "delta": delta,
            }
    return out


def _print_rank_slot_block(title: str, summary: Dict[str, Dict[str, Dict[str, float]]], k: int) -> None:
    print(title)
    for r in range(1, int(k) + 1):
        rank_key = f"rank{r}"
        pieces = []
        for slot in _slot_labels(int(k)):
            fit = (summary.get(rank_key, {}) or {}).get(slot, {}) or {}
            pieces.append(f"{slot}:{float(fit.get('mean', float('nan'))):+.3f}/{float(fit.get('std', float('nan'))):.3f}")
        print(f"{rank_key}: " + ", ".join(pieces))
    print("")


def _print_rank_slot_comparison(title: str, comparison: Dict[str, Dict[str, Dict[str, float]]], k: int) -> None:
    print(title)
    for r in range(1, int(k) + 1):
        rank_key = f"rank{r}"
        pieces = []
        for slot in _slot_labels(int(k)):
            fit = (comparison.get(rank_key, {}) or {}).get(slot, {}) or {}
            pieces.append(
                f"{slot}:{float(fit.get('raw', float('nan'))):+.3f}->{float(fit.get('pride', float('nan'))):+.3f} ({float(fit.get('delta', float('nan'))):+.3f})"
            )
        print(f"{rank_key}: " + ", ".join(pieces))
    print("")


def _run_multi(
    *,
    results_dirs: List[str],
    subjects: Optional[List[str]],
    n_runs: int,
    max_samples: int,
    aggregate_out_dir: str,
    save_plots: bool,
    apply_pride_offline: bool,
    pride_prefix_percent: float,
    pride_seed: int,
) -> None:
    os.makedirs(aggregate_out_dir, exist_ok=True)
    combined_by_k: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    combined_pride_by_k: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    pride_infos_by_k: Dict[int, List[Dict[str, object]]] = {}

    for results_dir in results_dirs:
        cache_files = _discover_cache_files(results_dir, subjects, int(n_runs))
        if not cache_files:
            print(f"[warn] no cache files found under: {results_dir}")
            continue
        for ci in cache_files:
            results = _read_results_file(ci.path)
            if int(max_samples) > 0:
                results = results[: int(max_samples)]
            k, bucket = _analyze_results_file(results, ci.subject, int(ci.run_idx))
            if k is None:
                continue
            combined_by_k.setdefault(int(k), _rank_slot_bucket(int(k)))
            _merge_rank_slot_bucket(combined_by_k[int(k)], bucket)
            if bool(apply_pride_offline):
                data0 = results[0].get("data", {}) or {}
                probs0 = data0.get("probs", None)
                if isinstance(probs0, list):
                    perm_list, _ = _infer_perm_list(int(k), len(probs0))
                    pride_results, pride_info = _apply_offline_pride_to_results(
                        results=results,
                        perm_list=perm_list,
                        option_ids=_slot_labels(int(k)),
                        pride_prefix_percent=float(pride_prefix_percent),
                        pride_seed=int(pride_seed),
                    )
                    k_pride, pride_bucket = _analyze_results_file(pride_results, ci.subject, int(ci.run_idx))
                    if k_pride is not None:
                        combined_pride_by_k.setdefault(int(k_pride), _rank_slot_bucket(int(k_pride)))
                        _merge_rank_slot_bucket(combined_pride_by_k[int(k_pride)], pride_bucket)
                        pride_infos_by_k.setdefault(int(k_pride), []).append(pride_info)

    if not combined_by_k:
        raise SystemExit("No valid records found.")

    output: Dict[str, object] = {"results_dirs": results_dirs, "by_k": {}}
    print("==== Rank-Slot Delta Analysis ====")
    for k in sorted(combined_by_k.keys()):
        bucket = combined_by_k[int(k)]
        summary = _summarize_rank_slot_bucket(bucket, int(k))
        pride_summary = _summarize_rank_slot_bucket(combined_pride_by_k[int(k)], int(k)) if int(k) in combined_pride_by_k else {}
        comparison_mean = _compare_rank_slot_summaries(summary, pride_summary, value_key="mean") if pride_summary else {}
        comparison_std = _compare_rank_slot_summaries(summary, pride_summary, value_key="std") if pride_summary else {}
        output["by_k"][str(int(k))] = {
            "raw": summary,
            "pride": pride_summary,
            "comparison_mean": comparison_mean,
            "comparison_std": comparison_std,
            "pride_info": pride_infos_by_k.get(int(k), []),
        }
        print(f"--- k={int(k)} ---")
        _print_rank_slot_block("raw:", summary, int(k))
        if pride_summary:
            _print_rank_slot_block("pride:", pride_summary, int(k))
            _print_rank_slot_comparison("raw -> pride (mean delta):", comparison_mean, int(k))

        if save_plots:
            grid_path = os.path.join(aggregate_out_dir, f"multi_model_rank_slot_delta_grid_k{int(k)}.png")
            mean_path = os.path.join(aggregate_out_dir, f"multi_model_rank_slot_delta_mean_k{int(k)}.png")
            std_path = os.path.join(aggregate_out_dir, f"multi_model_rank_slot_delta_std_k{int(k)}.png")
            saved_grid = _save_rank_slot_grid_plot(
                summary=summary,
                bucket=bucket,
                out_path=grid_path,
                title=f"Multi-model cyclic rank-slot delta distributions (k={int(k)})",
            )
            saved_mean = _save_rank_slot_heatmap(
                summary=summary,
                out_path=mean_path,
                value_key="mean",
                title=f"Multi-model cyclic rank-slot delta mean (k={int(k)})",
            )
            saved_std = _save_rank_slot_heatmap(
                summary=summary,
                out_path=std_path,
                value_key="std",
                title=f"Multi-model cyclic rank-slot delta std (k={int(k)})",
            )
            for p in (saved_grid, saved_mean, saved_std):
                if p:
                    print(f"Saved: {p}")
            if pride_summary:
                pride_grid_path = os.path.join(aggregate_out_dir, f"multi_model_rank_slot_delta_pride_grid_k{int(k)}.png")
                pride_mean_path = os.path.join(aggregate_out_dir, f"multi_model_rank_slot_delta_pride_mean_k{int(k)}.png")
                pride_std_path = os.path.join(aggregate_out_dir, f"multi_model_rank_slot_delta_pride_std_k{int(k)}.png")
                pride_bucket = combined_pride_by_k[int(k)]
                saved_pride_grid = _save_rank_slot_grid_plot(
                    summary=pride_summary,
                    bucket=pride_bucket,
                    out_path=pride_grid_path,
                    title=f"Multi-model cyclic rank-slot delta distributions after PriDe (k={int(k)})",
                )
                saved_pride_mean = _save_rank_slot_heatmap(
                    summary=pride_summary,
                    out_path=pride_mean_path,
                    value_key="mean",
                    title=f"Multi-model cyclic rank-slot delta mean after PriDe (k={int(k)})",
                )
                saved_pride_std = _save_rank_slot_heatmap(
                    summary=pride_summary,
                    out_path=pride_std_path,
                    value_key="std",
                    title=f"Multi-model cyclic rank-slot delta std after PriDe (k={int(k)})",
                )
                delta_mean_path = os.path.join(aggregate_out_dir, f"multi_model_rank_slot_delta_pride_minus_raw_mean_k{int(k)}.png")
                delta_std_path = os.path.join(aggregate_out_dir, f"multi_model_rank_slot_delta_pride_minus_raw_std_k{int(k)}.png")
                saved_delta_mean = _save_rank_slot_heatmap(
                    summary=comparison_mean,
                    out_path=delta_mean_path,
                    value_key="delta",
                    title=f"Multi-model rank-slot delta mean change (PriDe - raw) (k={int(k)})",
                )
                saved_delta_std = _save_rank_slot_heatmap(
                    summary=comparison_std,
                    out_path=delta_std_path,
                    value_key="delta",
                    title=f"Multi-model rank-slot delta std change (PriDe - raw) (k={int(k)})",
                )
                for p in (saved_pride_grid, saved_pride_mean, saved_pride_std, saved_delta_mean, saved_delta_std):
                    if p:
                        print(f"Saved: {p}")

    out_path = os.path.join(aggregate_out_dir, "multi_model_rank_slot_delta_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


def _run_single(
    *,
    cache_dir: str,
    subjects: Optional[List[str]],
    n_runs: int,
    max_samples: int,
    save_plots: bool,
    apply_pride_offline: bool,
    pride_prefix_percent: float,
    pride_seed: int,
) -> None:
    cache_files = _discover_cache_files(cache_dir, subjects, int(n_runs))
    if not cache_files:
        raise SystemExit(f"No cache files found under: {cache_dir}")

    combined_by_k: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    combined_pride_by_k: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    pride_infos_by_k: Dict[int, List[Dict[str, object]]] = {}
    for ci in cache_files:
        results = _read_results_file(ci.path)
        if int(max_samples) > 0:
            results = results[: int(max_samples)]
        k, bucket = _analyze_results_file(results, ci.subject, int(ci.run_idx))
        if k is None:
            continue
        combined_by_k.setdefault(int(k), _rank_slot_bucket(int(k)))
        _merge_rank_slot_bucket(combined_by_k[int(k)], bucket)
        if bool(apply_pride_offline):
            data0 = results[0].get("data", {}) or {}
            probs0 = data0.get("probs", None)
            if isinstance(probs0, list):
                perm_list, _ = _infer_perm_list(int(k), len(probs0))
                pride_results, pride_info = _apply_offline_pride_to_results(
                    results=results,
                    perm_list=perm_list,
                    option_ids=_slot_labels(int(k)),
                    pride_prefix_percent=float(pride_prefix_percent),
                    pride_seed=int(pride_seed),
                )
                k_pride, pride_bucket = _analyze_results_file(pride_results, ci.subject, int(ci.run_idx))
                if k_pride is not None:
                    combined_pride_by_k.setdefault(int(k_pride), _rank_slot_bucket(int(k_pride)))
                    _merge_rank_slot_bucket(combined_pride_by_k[int(k_pride)], pride_bucket)
                    pride_infos_by_k.setdefault(int(k_pride), []).append(pride_info)

    if not combined_by_k:
        raise SystemExit("No valid records found.")

    output: Dict[str, object] = {"cache_dir": cache_dir, "by_k": {}}
    print("==== Rank-Slot Delta Analysis ====")
    print(f"cache_dir: {cache_dir}")
    for k in sorted(combined_by_k.keys()):
        bucket = combined_by_k[int(k)]
        summary = _summarize_rank_slot_bucket(bucket, int(k))
        pride_summary = _summarize_rank_slot_bucket(combined_pride_by_k[int(k)], int(k)) if int(k) in combined_pride_by_k else {}
        comparison_mean = _compare_rank_slot_summaries(summary, pride_summary, value_key="mean") if pride_summary else {}
        comparison_std = _compare_rank_slot_summaries(summary, pride_summary, value_key="std") if pride_summary else {}
        output["by_k"][str(int(k))] = {
            "raw": summary,
            "pride": pride_summary,
            "comparison_mean": comparison_mean,
            "comparison_std": comparison_std,
            "pride_info": pride_infos_by_k.get(int(k), []),
        }
        print(f"--- k={int(k)} ---")
        _print_rank_slot_block("raw:", summary, int(k))
        if pride_summary:
            _print_rank_slot_block("pride:", pride_summary, int(k))
            _print_rank_slot_comparison("raw -> pride (mean delta):", comparison_mean, int(k))

        if save_plots:
            grid_path = os.path.join(cache_dir, f"rank_slot_delta_grid_k{int(k)}.png")
            mean_path = os.path.join(cache_dir, f"rank_slot_delta_mean_k{int(k)}.png")
            std_path = os.path.join(cache_dir, f"rank_slot_delta_std_k{int(k)}.png")
            saved_grid = _save_rank_slot_grid_plot(
                summary=summary,
                bucket=bucket,
                out_path=grid_path,
                title=f"Cyclic rank-slot delta distributions (k={int(k)})",
            )
            saved_mean = _save_rank_slot_heatmap(
                summary=summary,
                out_path=mean_path,
                value_key="mean",
                title=f"Cyclic rank-slot delta mean (k={int(k)})",
            )
            saved_std = _save_rank_slot_heatmap(
                summary=summary,
                out_path=std_path,
                value_key="std",
                title=f"Cyclic rank-slot delta std (k={int(k)})",
            )
            for p in (saved_grid, saved_mean, saved_std):
                if p:
                    print(f"Saved: {p}")
            if pride_summary:
                pride_bucket = combined_pride_by_k[int(k)]
                pride_grid_path = os.path.join(cache_dir, f"rank_slot_delta_pride_grid_k{int(k)}.png")
                pride_mean_path = os.path.join(cache_dir, f"rank_slot_delta_pride_mean_k{int(k)}.png")
                pride_std_path = os.path.join(cache_dir, f"rank_slot_delta_pride_std_k{int(k)}.png")
                saved_pride_grid = _save_rank_slot_grid_plot(
                    summary=pride_summary,
                    bucket=pride_bucket,
                    out_path=pride_grid_path,
                    title=f"Cyclic rank-slot delta distributions after PriDe (k={int(k)})",
                )
                saved_pride_mean = _save_rank_slot_heatmap(
                    summary=pride_summary,
                    out_path=pride_mean_path,
                    value_key="mean",
                    title=f"Cyclic rank-slot delta mean after PriDe (k={int(k)})",
                )
                saved_pride_std = _save_rank_slot_heatmap(
                    summary=pride_summary,
                    out_path=pride_std_path,
                    value_key="std",
                    title=f"Cyclic rank-slot delta std after PriDe (k={int(k)})",
                )
                delta_mean_path = os.path.join(cache_dir, f"rank_slot_delta_pride_minus_raw_mean_k{int(k)}.png")
                delta_std_path = os.path.join(cache_dir, f"rank_slot_delta_pride_minus_raw_std_k{int(k)}.png")
                saved_delta_mean = _save_rank_slot_heatmap(
                    summary=comparison_mean,
                    out_path=delta_mean_path,
                    value_key="delta",
                    title=f"Rank-slot delta mean change (PriDe - raw) (k={int(k)})",
                )
                saved_delta_std = _save_rank_slot_heatmap(
                    summary=comparison_std,
                    out_path=delta_std_path,
                    value_key="delta",
                    title=f"Rank-slot delta std change (PriDe - raw) (k={int(k)})",
                )
                for p in (saved_pride_grid, saved_pride_mean, saved_pride_std, saved_delta_mean, saved_delta_std):
                    if p:
                        print(f"Saved: {p}")

    out_path = os.path.join(cache_dir, "rank_slot_delta_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze cyclic-ensemble rank-slot probability deltas.")
    ap.add_argument("--results_dir", type=str, default=None, help="Single cache/results directory.")
    ap.add_argument("--results_dirs", nargs="*", default=None, help="Multiple results directories to pool.")
    ap.add_argument("--aggregate_out_dir", type=str, default=None, help="Output dir for pooled multi-model analysis.")
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional subject list.")
    ap.add_argument("--n_runs", type=int, default=1, help="Number of run caches to read.")
    ap.add_argument("--max_samples", type=int, default=0, help="Optional cap per cache file.")
    ap.add_argument("--save_plots", action="store_true", help="Save rank-slot plots.")
    ap.add_argument("--apply_pride_offline", action="store_true", help="Apply PriDe offline and compare before/after.")
    ap.add_argument("--pride_prefix_percent", type=float, default=100.0, help="Percent of samples used for PriDe prior estimation.")
    ap.add_argument("--pride_seed", type=int, default=0, help="Seed for offline PriDe prefix sampling.")
    args = ap.parse_args()

    if args.results_dirs:
        if not args.aggregate_out_dir:
            raise SystemExit("--aggregate_out_dir is required with --results_dirs")
        _run_multi(
            results_dirs=[str(x) for x in (args.results_dirs or []) if str(x).strip()],
            subjects=[str(x) for x in (args.subjects or [])] if args.subjects else None,
            n_runs=int(args.n_runs),
            max_samples=int(args.max_samples),
            aggregate_out_dir=str(args.aggregate_out_dir),
            save_plots=bool(args.save_plots),
            apply_pride_offline=bool(args.apply_pride_offline),
            pride_prefix_percent=float(args.pride_prefix_percent),
            pride_seed=int(args.pride_seed),
        )
        return

    if not args.results_dir:
        raise SystemExit("Provide --results_dir or --results_dirs")
    _run_single(
        cache_dir=str(args.results_dir),
        subjects=[str(x) for x in (args.subjects or [])] if args.subjects else None,
        n_runs=int(args.n_runs),
        max_samples=int(args.max_samples),
        save_plots=bool(args.save_plots),
        apply_pride_offline=bool(args.apply_pride_offline),
        pride_prefix_percent=float(args.pride_prefix_percent),
        pride_seed=int(args.pride_seed),
    )


if __name__ == "__main__":
    main()
