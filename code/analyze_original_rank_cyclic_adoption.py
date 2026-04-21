#!/usr/bin/env python3
"""
Original-rank cyclic adoption analysis.

For each sample, we use the original/identity order prediction as the rank
reference. If content_(r) is ranked r-th by the original order distribution, we
ask how often that same content is selected as top-1 after cyclic permutations.

For k-way multiple choice, this yields k x k adoption rates:

    adoption(r, s) = P(content_(r) is top-1 | content_(r) is displayed at slot s)

We also report the mean probability assigned to each original-rank content under
each slot, which helps distinguish "chosen as answer" from "assigned more mass."
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analyze_perm_noise import (
    _aggregate_probs_over_permutations,
    _discover_cache_files,
    _infer_perm_list,
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


def _rank_bucket(k: int) -> Dict[str, List[float]]:
    return {f"rank{r}": [] for r in range(1, int(k) + 1)}


def _merge_rank_slot_bucket(dst: Dict[str, Dict[str, List[float]]], src: Dict[str, Dict[str, List[float]]]) -> None:
    for rank_key, slot_map in (src or {}).items():
        dst.setdefault(str(rank_key), {})
        for slot, vals in (slot_map or {}).items():
            dst[str(rank_key)].setdefault(str(slot), [])
            dst[str(rank_key)][str(slot)].extend(float(x) for x in (vals or []))


def _merge_rank_bucket(dst: Dict[str, List[float]], src: Dict[str, List[float]]) -> None:
    for rank_key, vals in (src or {}).items():
        dst.setdefault(str(rank_key), [])
        dst[str(rank_key)].extend(float(x) for x in (vals or []))


def _summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _summarize_rank_slot_bucket(bucket: Dict[str, Dict[str, List[float]]], k: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    labels = _slot_labels(k)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in range(1, int(k) + 1):
        rank_key = f"rank{r}"
        out[rank_key] = {}
        for slot in labels:
            out[rank_key][slot] = _summary((bucket.get(rank_key, {}) or {}).get(slot, []))
    return out


def _summarize_rank_bucket(bucket: Dict[str, List[float]], k: int) -> Dict[str, Dict[str, float]]:
    return {f"rank{r}": _summary((bucket or {}).get(f"rank{r}", [])) for r in range(1, int(k) + 1)}


def _identity_perm_index(perm_list: Sequence[Sequence[int]], k: int) -> Optional[int]:
    identity = tuple(range(int(k)))
    for idx, perm in enumerate(perm_list):
        if tuple(int(x) for x in perm) == identity:
            return int(idx)
    return None


def _analyze_results_file(
    results: List[dict],
) -> Tuple[
    Optional[int],
    Dict[str, Dict[str, List[float]]],
    Dict[str, Dict[str, List[float]]],
    Dict[str, List[float]],
]:
    if not results:
        return None, {}, {}, {}

    data0 = results[0].get("data", {}) or {}
    options = data0.get("options", None)
    if not isinstance(options, list) or not options:
        return None, {}, {}, {}
    k = len(options)
    labels = _slot_labels(k)
    probs0 = data0.get("probs", None)
    if not isinstance(probs0, list):
        return None, {}, {}, {}

    perm_list, _ = _infer_perm_list(k, len(probs0))
    identity_idx = _identity_perm_index(perm_list, k)
    if identity_idx is None:
        return None, {}, {}, {}

    cyc_perms = _rotations(k)
    cyc_idxs = [perm_list.index(p) for p in cyc_perms if p in perm_list]
    if len(cyc_idxs) != k:
        return None, {}, {}, {}
    selected_perm_tuples = [tuple(int(x) for x in perm_list[idx]) for idx in cyc_idxs]

    adoption_bucket = _rank_slot_bucket(k)
    prob_bucket = _rank_slot_bucket(k)
    original_prob_bucket = _rank_bucket(k)

    for rec in results:
        d = rec.get("data", {}) or {}
        probs_seq = d.get("probs", None)
        if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
            continue

        original_dist = np.asarray(
            _aggregate_probs_over_permutations([probs_seq[identity_idx]], [perm_list[identity_idx]], k),
            dtype=np.float64,
        )
        if original_dist.size != k:
            continue
        rank_order = np.argsort(original_dist)[::-1].tolist()
        for rank_idx, content_idx in enumerate(rank_order, start=1):
            original_prob_bucket[f"rank{int(rank_idx)}"].append(float(original_dist[int(content_idx)]))

        selected_dists = []
        for perm_idx in cyc_idxs:
            dist = _aggregate_probs_over_permutations([probs_seq[perm_idx]], [perm_list[perm_idx]], k)
            selected_dists.append(np.asarray(dist, dtype=np.float64))
        if not selected_dists:
            continue
        selected_dists_arr = np.asarray(selected_dists, dtype=np.float64)
        top1_contents = np.argmax(selected_dists_arr, axis=1).astype(int)

        for rank_idx, content_idx in enumerate(rank_order, start=1):
            rank_key = f"rank{int(rank_idx)}"
            for view_idx, perm in enumerate(selected_perm_tuples):
                slot_idx = next((j for j, c_idx in enumerate(perm) if int(c_idx) == int(content_idx)), -1)
                if slot_idx < 0:
                    continue
                slot_label = str(labels[int(slot_idx)])
                adopted = 1.0 if int(top1_contents[view_idx]) == int(content_idx) else 0.0
                adoption_bucket[rank_key][slot_label].append(adopted)
                prob_bucket[rank_key][slot_label].append(float(selected_dists_arr[view_idx, int(content_idx)]))

    return int(k), adoption_bucket, prob_bucket, original_prob_bucket


def _save_heatmap(
    *,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    out_path: str,
    value_key: str,
    title: str,
    cmap: str = "viridis",
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
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0.0 if value_key == "mean" else None)
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


def _save_rank_bar(
    *,
    summary: Dict[str, Dict[str, float]],
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
    vals = [float((summary.get(rank_key, {}) or {}).get(value_key, float("nan"))) for rank_key in rank_keys]
    xs = np.arange(len(rank_keys))
    fig = plt.figure(figsize=(1.2 + 1.1 * len(rank_keys), 3.2), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(xs, vals, color="#6baed6", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(rank_keys)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for x, val in zip(xs, vals):
        if np.isfinite(val):
            ax.text(float(x), float(val), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _print_rank_slot_block(title: str, summary: Dict[str, Dict[str, Dict[str, float]]], k: int) -> None:
    print(title)
    for r in range(1, int(k) + 1):
        rank_key = f"rank{r}"
        pieces = []
        for slot in _slot_labels(int(k)):
            fit = (summary.get(rank_key, {}) or {}).get(slot, {}) or {}
            pieces.append(f"{slot}:{float(fit.get('mean', float('nan'))):.3f}/{float(fit.get('std', float('nan'))):.3f}")
        print(f"{rank_key}: " + ", ".join(pieces))
    print("")


def _print_rank_block(title: str, summary: Dict[str, Dict[str, float]], k: int) -> None:
    pieces = []
    for r in range(1, int(k) + 1):
        rank_key = f"rank{r}"
        fit = (summary.get(rank_key, {}) or {})
        pieces.append(f"{rank_key}:{float(fit.get('mean', float('nan'))):.3f}/{float(fit.get('std', float('nan'))):.3f}")
    print(title)
    print(", ".join(pieces))
    print("")


def _run_multi(
    *,
    results_dirs: List[str],
    subjects: Optional[List[str]],
    n_runs: int,
    max_samples: int,
    aggregate_out_dir: str,
    save_plots: bool,
) -> None:
    os.makedirs(aggregate_out_dir, exist_ok=True)
    adoption_by_k: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    prob_by_k: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    original_prob_by_k: Dict[int, Dict[str, List[float]]] = {}

    for results_dir in results_dirs:
        cache_files = _discover_cache_files(results_dir, subjects, int(n_runs))
        if not cache_files:
            print(f"[warn] no cache files found under: {results_dir}")
            continue
        for ci in cache_files:
            results = _read_results_file(ci.path)
            if int(max_samples) > 0:
                results = results[: int(max_samples)]
            k, adoption_bucket, prob_bucket, original_prob_bucket = _analyze_results_file(results)
            if k is None:
                continue
            adoption_by_k.setdefault(int(k), _rank_slot_bucket(int(k)))
            prob_by_k.setdefault(int(k), _rank_slot_bucket(int(k)))
            original_prob_by_k.setdefault(int(k), _rank_bucket(int(k)))
            _merge_rank_slot_bucket(adoption_by_k[int(k)], adoption_bucket)
            _merge_rank_slot_bucket(prob_by_k[int(k)], prob_bucket)
            _merge_rank_bucket(original_prob_by_k[int(k)], original_prob_bucket)

    if not adoption_by_k:
        raise SystemExit("No valid records found.")

    output: Dict[str, object] = {"results_dirs": results_dirs, "by_k": {}}
    print("==== Original-Rank Cyclic Adoption Analysis ====")
    for k in sorted(adoption_by_k.keys()):
        adoption_summary = _summarize_rank_slot_bucket(adoption_by_k[int(k)], int(k))
        prob_summary = _summarize_rank_slot_bucket(prob_by_k[int(k)], int(k))
        original_prob_summary = _summarize_rank_bucket(original_prob_by_k[int(k)], int(k))
        output["by_k"][str(int(k))] = {
            "adoption_rate": adoption_summary,
            "cyclic_content_prob": prob_summary,
            "original_rank_prob": original_prob_summary,
        }
        print(f"--- k={int(k)} ---")
        _print_rank_block("original-order rank probs:", original_prob_summary, int(k))
        _print_rank_slot_block("cyclic adoption rate by original rank/slot:", adoption_summary, int(k))
        _print_rank_slot_block("cyclic content prob by original rank/slot:", prob_summary, int(k))

        if save_plots:
            paths = [
                _save_heatmap(
                    summary=adoption_summary,
                    out_path=os.path.join(aggregate_out_dir, f"multi_model_original_rank_cyclic_adoption_rate_k{int(k)}.png"),
                    value_key="mean",
                    title=f"Multi-model adoption rate by original rank/slot (k={int(k)})",
                ),
                _save_heatmap(
                    summary=prob_summary,
                    out_path=os.path.join(aggregate_out_dir, f"multi_model_original_rank_cyclic_content_prob_k{int(k)}.png"),
                    value_key="mean",
                    title=f"Multi-model cyclic probability by original rank/slot (k={int(k)})",
                ),
                _save_rank_bar(
                    summary=original_prob_summary,
                    out_path=os.path.join(aggregate_out_dir, f"multi_model_original_rank_prob_k{int(k)}.png"),
                    value_key="mean",
                    title=f"Multi-model original-order rank probability (k={int(k)})",
                ),
            ]
            for path in paths:
                if path:
                    print(f"Saved: {path}")

    out_path = os.path.join(aggregate_out_dir, "multi_model_original_rank_cyclic_adoption_summary.json")
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
) -> None:
    cache_files = _discover_cache_files(cache_dir, subjects, int(n_runs))
    if not cache_files:
        raise SystemExit(f"No cache files found under: {cache_dir}")

    adoption_by_k: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    prob_by_k: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    original_prob_by_k: Dict[int, Dict[str, List[float]]] = {}
    for ci in cache_files:
        results = _read_results_file(ci.path)
        if int(max_samples) > 0:
            results = results[: int(max_samples)]
        k, adoption_bucket, prob_bucket, original_prob_bucket = _analyze_results_file(results)
        if k is None:
            continue
        adoption_by_k.setdefault(int(k), _rank_slot_bucket(int(k)))
        prob_by_k.setdefault(int(k), _rank_slot_bucket(int(k)))
        original_prob_by_k.setdefault(int(k), _rank_bucket(int(k)))
        _merge_rank_slot_bucket(adoption_by_k[int(k)], adoption_bucket)
        _merge_rank_slot_bucket(prob_by_k[int(k)], prob_bucket)
        _merge_rank_bucket(original_prob_by_k[int(k)], original_prob_bucket)

    if not adoption_by_k:
        raise SystemExit("No valid records found.")

    output: Dict[str, object] = {"cache_dir": cache_dir, "by_k": {}}
    print("==== Original-Rank Cyclic Adoption Analysis ====")
    print(f"cache_dir: {cache_dir}")
    for k in sorted(adoption_by_k.keys()):
        adoption_summary = _summarize_rank_slot_bucket(adoption_by_k[int(k)], int(k))
        prob_summary = _summarize_rank_slot_bucket(prob_by_k[int(k)], int(k))
        original_prob_summary = _summarize_rank_bucket(original_prob_by_k[int(k)], int(k))
        output["by_k"][str(int(k))] = {
            "adoption_rate": adoption_summary,
            "cyclic_content_prob": prob_summary,
            "original_rank_prob": original_prob_summary,
        }
        print(f"--- k={int(k)} ---")
        _print_rank_block("original-order rank probs:", original_prob_summary, int(k))
        _print_rank_slot_block("cyclic adoption rate by original rank/slot:", adoption_summary, int(k))
        _print_rank_slot_block("cyclic content prob by original rank/slot:", prob_summary, int(k))

        if save_plots:
            paths = [
                _save_heatmap(
                    summary=adoption_summary,
                    out_path=os.path.join(cache_dir, f"original_rank_cyclic_adoption_rate_k{int(k)}.png"),
                    value_key="mean",
                    title=f"Adoption rate by original rank/slot (k={int(k)})",
                ),
                _save_heatmap(
                    summary=prob_summary,
                    out_path=os.path.join(cache_dir, f"original_rank_cyclic_content_prob_k{int(k)}.png"),
                    value_key="mean",
                    title=f"Cyclic probability by original rank/slot (k={int(k)})",
                ),
                _save_rank_bar(
                    summary=original_prob_summary,
                    out_path=os.path.join(cache_dir, f"original_rank_prob_k{int(k)}.png"),
                    value_key="mean",
                    title=f"Original-order rank probability (k={int(k)})",
                ),
            ]
            for path in paths:
                if path:
                    print(f"Saved: {path}")

    out_path = os.path.join(cache_dir, "original_rank_cyclic_adoption_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze how original-order ranked contents are adopted under cyclic permutations.")
    ap.add_argument("--results_dir", type=str, default=None, help="Single cache/results directory.")
    ap.add_argument("--results_dirs", nargs="*", default=None, help="Multiple results directories to pool.")
    ap.add_argument("--aggregate_out_dir", type=str, default=None, help="Output dir for pooled multi-model analysis.")
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional subject list.")
    ap.add_argument("--n_runs", type=int, default=1, help="Number of run caches to read.")
    ap.add_argument("--max_samples", type=int, default=0, help="Optional cap per cache file.")
    ap.add_argument("--save_plots", action="store_true", help="Save heatmap/bar plots.")
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
    )


if __name__ == "__main__":
    main()
