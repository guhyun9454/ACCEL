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
from dataclasses import dataclass
from itertools import permutations
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
            base = fn[:-5]
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
        })

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

    out_path = os.path.join(str(cache_dir), "perm_noise_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(per_record_reports, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

    saved_plot_paths: List[str] = []
    if save_plots or wandb_enabled:
        saved_plot_paths = _save_noise_plots(
            per_record_reports,
            str(cache_dir),
            corr_by_k=corr_by_k,
            recall_by_k=recall_by_k,
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
            ax.errorbar(xs, ys, yerr=es, fmt="-o", lw=1.6, ms=4, capsize=3, label="Ensemble(m perms): mean±std over subject-run")
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
                                labels=["Cyclic pairs (C vs C)", "Non-cyclic pairs (N vs N)"],
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
    ap.add_argument("--subjects", type=str, default="", help="Comma-separated subject list. If empty, infer from results_dir.")

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
    ap.add_argument("--save_plots", action="store_true", help="Save PNG plots into results_dir (default: off).")

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

    # If results_dir is explicitly given, do analysis-only.
    if str(args.results_dir).strip():
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
        )
        return

    # Otherwise, wrapper mode: run eval_clm then analyze inferred results_dir(s).
    if not str(args.pretrained_model_path).strip() or not args.eval_names:
        raise SystemExit(
            "Provide either --results_dir (analysis-only) OR provide --pretrained_model_path and --eval_names (eval+analyze)."
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
        )


if __name__ == "__main__":
    main()

