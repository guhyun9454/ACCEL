#!/usr/bin/env python3
"""
Plot per-option recall robustness (Original vs Cyclic vs Full/Available).

This is a standalone script extracted from the permutation-noise analysis work.
It reads eval_clm cached jsonl files (per subject, optionally per run), aggregates
all samples across subjects, and saves a PNG bar chart:
  - recall per option (A/B/C/D or A..E) for:
      (1) Original (identity-only)
      (2) Cyclic ensemble (all rotations available in the file)
      (3) Full/Available ensemble (all permutations present in the file)

It also optionally uploads the PNG to Weights & Biases via wandb.Image.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from itertools import permutations
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
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
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(
    probs_seq: Sequence[Sequence[float]],
    permuted_indices: Sequence[Tuple[int, ...]],
    k: int,
) -> np.ndarray:
    """
    probs_seq: list/array len = (#perms used), each element length k (letter-space probs)
    permuted_indices: perms p where p[j] is content-index at letter position j.
    Returns: agg (k,) content-space aggregated probs (mean over permutations)
    """
    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            # p[j] is the content-index that appears at letter position j in this permutation.
            # Convert letter-space probs -> content-space by adding letter_probs[j] into agg[p[j]].
            agg[p[j]] += letter_probs[j]
            
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


def _infer_perm_list(k: int, perm_count: int) -> Tuple[List[Tuple[int, ...]], bool]:
    """Return (perm_list, full_enabled)."""
    if perm_count == math.factorial(k):
        return list(sorted(permutations(range(k)))), True
    if perm_count == k:
        return _rotations(k), False
    # Unknown: allow prefix of full perms
    full = list(sorted(permutations(range(k))))
    if perm_count <= len(full):
        return full[:perm_count], False
    raise ValueError(f"Unsupported perm_count={perm_count} for k={k}")


def _read_jsonl_results(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "result":
                out.append(obj.get("data", {}) or {})
    out.sort(key=lambda d: int(d.get("idx", 0)))
    return out


def _collect_files(results_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    files = [
        p for p in files
        if p.endswith(".jsonl")
        and not p.endswith("_curve.jsonl")
        and not p.endswith("_pride_curve.jsonl")
    ]
    return files


def _compute_preds_for_one_sample(
    probs_seq: List[List[float]],
    perm_list: List[Tuple[int, ...]],
    option_ids: List[str],
) -> Tuple[str, str, str]:
    k = len(option_ids)
    identity = tuple(range(k))
    identity_idx = perm_list.index(identity) if identity in perm_list else 0

    cyc_perms = _rotations(k)
    cyc_idxs = [perm_list.index(p) for p in cyc_perms if p in perm_list]
    cyc_perm_list = [perm_list[i] for i in cyc_idxs]

    # Original (identity-only)
    agg_o = _aggregate_probs_over_permutations([probs_seq[identity_idx]], [perm_list[identity_idx]], k)
    pred_o = option_ids[int(np.argmax(agg_o))]

    # Cyclic ensemble
    if cyc_idxs:
        probs_c = [probs_seq[i] for i in cyc_idxs]
        agg_c = _aggregate_probs_over_permutations(probs_c, cyc_perm_list, k)
        pred_c = option_ids[int(np.argmax(agg_c))]
    else:
        pred_c = pred_o

    # Full/available ensemble (all perms present in file)
    agg_f = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
    pred_f = option_ids[int(np.argmax(agg_f))]

    return pred_o, pred_c, pred_f


def _recalls_by_option(y_true: List[str], y_pred: List[str], options: List[str]) -> List[float]:
    out = []
    for opt in options:
        idxs = [i for i, t in enumerate(y_true) if t == opt]
        if not idxs:
            out.append(0.0)
        else:
            out.append(float(np.mean([1.0 if y_pred[i] == opt else 0.0 for i in idxs])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)

    # ===== Mode A: plot-only (point to existing cached results) =====
    ap.add_argument("--results_dir", type=str, default="", help="e.g., results_mmlu/0s_MODEL/mmlu_full_id-ABCD")
    ap.add_argument("--glob", type=str, default="*.jsonl", help="File glob under results_dir (default: *.jsonl)")
    ap.add_argument("--max_files", type=int, default=0, help="If >0, limit number of subject files read.")
    ap.add_argument("--max_samples_per_file", type=int, default=0, help="If >0, truncate samples per subject file.")
    ap.add_argument("--out", type=str, default="analyze_perm_recall.png", help="Output PNG filename (saved into results_dir unless absolute).")
    ap.add_argument("--force_plot", action="store_true", help="Overwrite existing PNG even if present.")

    # ===== Mode B: eval+plot (run eval_clm.py if cache missing) =====
    ap.add_argument("--eval_clm_path", type=str, default="", help="Path to eval_clm.py. Default: sibling eval_clm.py.")
    ap.add_argument("--skip_eval", action="store_true", help="Skip running eval_clm even if cache missing.")
    ap.add_argument("--pretrained_model_path", type=str, default="", help="(wrapper) Hugging Face model id or path.")
    ap.add_argument("--eval_names", type=str, nargs="+", default=[], help="(wrapper) eval names, e.g. mmlu,0,perm")
    ap.add_argument("--n_runs", type=int, default=1, help="(wrapper) Number of runs (affects cache filenames).")
    ap.add_argument("--option_id_set", type=str, default=None, help="(wrapper) option id set, e.g. ABCD / ABCDE")
    ap.add_argument("--force", action="store_true", help="(wrapper) forward to eval_clm to overwrite cache.")

    ap.add_argument("--wandb", action="store_true", help="Upload PNG to W&B using wandb.Image.")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default="capde")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags")

    # parse_known_args so we can forward unknown flags to eval_clm.py
    args, unknown = ap.parse_known_args()

    def _compute_eval_save_path(eval_name: str, pretrained_model_path: str, option_id_set: Optional[str]) -> str:
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

    # Determine results_dir (explicit or inferred from eval args)
    results_dir = str(args.results_dir).strip()
    task_name: Optional[str] = None

    if not results_dir:
        if not str(args.pretrained_model_path).strip() or not args.eval_names:
            raise SystemExit(
                "Provide either --results_dir (plot-only) OR provide --pretrained_model_path and --eval_names (eval+plot)."
            )

        # Only plot for the first eval_name (most common use). If you pass multiple, we loop below.
        # We still compute task_name for W&B naming.
        task_name = str(args.eval_names[0]).split(",")[0].strip() if args.eval_names else None

        eval_clm_path = str(args.eval_clm_path).strip()
        if not eval_clm_path:
            eval_clm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_clm.py")
        if not os.path.exists(eval_clm_path):
            raise SystemExit(f"eval_clm.py not found: {eval_clm_path}")
        code_dir = os.path.dirname(os.path.abspath(eval_clm_path))

        # Evaluate each eval_name and plot each into its own results dir
        for eval_name in args.eval_names:
            task_name = str(eval_name).split(",")[0].strip() if eval_name else task_name
            rel = _compute_eval_save_path(str(eval_name), str(args.pretrained_model_path), args.option_id_set)
            results_dir = os.path.join(code_dir, rel)

            # If cache missing, run eval_clm
            files_probe = _collect_files(results_dir, str(args.glob))
            if int(args.n_runs) > 1 and str(args.glob).strip() == "*.jsonl":
                # Default to one run to avoid duplication
                files_probe = _collect_files(results_dir, "*_run0.jsonl")
            cache_ok = len(files_probe) > 0

            if (not cache_ok) and (not bool(args.skip_eval)):
                cmd = [sys.executable, os.path.abspath(eval_clm_path)]
                cmd += ["--pretrained_model_path", str(args.pretrained_model_path)]
                cmd += ["--eval_names", str(eval_name)]
                if args.option_id_set:
                    cmd += ["--option_id_set", str(args.option_id_set)]
                if int(args.n_runs) != 1:
                    cmd += ["--n_runs", str(int(args.n_runs))]
                if bool(args.force):
                    cmd += ["--force"]
                # Pass through W&B flags to eval if requested
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
            elif (not cache_ok) and bool(args.skip_eval):
                raise SystemExit(f"No cached jsonl found in {results_dir} and --skip_eval was set.")

            # Plot for this eval_name
            _plot_one_results_dir(args, results_dir, task_name=task_name)
        return

    # Plot-only mode for an explicit results_dir
    _plot_one_results_dir(args, results_dir, task_name=None)


def _plot_one_results_dir(args, results_dir: str, task_name: Optional[str] = None) -> None:
    # Decide file glob: if user didn't override and n_runs>1, default to run0
    file_glob = str(getattr(args, "glob", "*.jsonl"))
    if int(getattr(args, "n_runs", 1)) > 1 and file_glob.strip() == "*.jsonl":
        file_glob = "*_run0.jsonl"

    out = str(getattr(args, "out", "analyze_perm_recall.png"))
    out_path = out if os.path.isabs(out) else os.path.join(results_dir, out)
    if os.path.exists(out_path) and (not bool(getattr(args, "force_plot", False))):
        print(f"[skip] PNG already exists (use --force_plot to overwrite): {out_path}")
        return

    files = _collect_files(results_dir, file_glob)
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]
    if not files:
        raise SystemExit(f"No jsonl files found in: {results_dir} (glob={file_glob})")

    # Aggregate across all subjects
    y_true: List[str] = []
    y_orig: List[str] = []
    y_cyc: List[str] = []
    y_full: List[str] = []

    k: Optional[int] = None
    perm_list: Optional[List[Tuple[int, ...]]] = None
    option_ids: Optional[List[str]] = None

    for fp in files:
        rows = _read_jsonl_results(fp)
        if int(args.max_samples_per_file) > 0:
            rows = rows[: int(args.max_samples_per_file)]
        if not rows:
            continue

        d0 = rows[0]
        options = d0.get("options", None)
        if not isinstance(options, list) or len(options) == 0:
            continue
        k0 = int(len(options))

        probs0 = d0.get("probs", None)
        if not isinstance(probs0, list):
            continue
        perm_count = int(len(probs0))
        perm_list0, _ = _infer_perm_list(k0, perm_count)

        # Use standard labels A..E (by k) unless ideals contain something else
        option_ids0 = list("ABCDE"[:k0]) if k0 in (4, 5) else [str(i) for i in range(k0)]

        # Ensure consistency across files (same k and perm layout)
        if k is None:
            k = k0
            perm_list = perm_list0
            option_ids = option_ids0
        else:
            if int(k) != int(k0):
                continue
            if perm_list != perm_list0:
                # Different perm layout (e.g., some files cyclic-only). Skip for now.
                continue

        assert perm_list is not None and option_ids is not None
        for d in rows:
            probs_seq = d.get("probs", None)
            ideal = str(d.get("ideal"))
            if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                continue
            if ideal not in option_ids:
                continue
            p_o, p_c, p_f = _compute_preds_for_one_sample(probs_seq, perm_list, option_ids)
            y_true.append(ideal)
            y_orig.append(p_o)
            y_cyc.append(p_c)
            y_full.append(p_f)

    if not y_true or option_ids is None or perm_list is None or k is None:
        raise SystemExit("No valid samples collected (check results_dir / glob / perm availability).")

    # Metrics
    def _acc(y_pred: List[str]) -> float:
        return float(np.mean([1.0 if p == t else 0.0 for p, t in zip(y_pred, y_true)])) if y_true else float("nan")

    r_orig = _recalls_by_option(y_true, y_orig, option_ids)
    r_cyc = _recalls_by_option(y_true, y_cyc, option_ids)
    r_full = _recalls_by_option(y_true, y_full, option_ids)
    a_orig = _acc(y_orig)
    a_cyc = _acc(y_cyc)
    a_full = _acc(y_full)
    s_orig = float(np.std(r_orig))
    s_cyc = float(np.std(r_cyc))
    s_full = float(np.std(r_full))

    plt = _try_import_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib not available (cannot save PNG).")

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
    # Full count (available perms)
    b3 = ax.bar(
        x + width,
        r_full,
        width,
        label=f"Full/Avail {len(perm_list)} (Acc: {a_full:.3f} | Std: {s_full:.3f})",
        color="#ff7f0e",
        alpha=0.82,
    )

    ax.set_ylabel("Recall (accuracy per option)", fontsize=12)
    ax.set_xlabel("Option", fontsize=12)
    ax.set_title("Recall robustness per option (all subjects combined)", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(option_ids, fontsize=12)
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

    out = str(args.out)
    if not os.path.isabs(out):
        out = os.path.join(results_dir, out)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

    if bool(args.wandb):
        wandb = _try_import_wandb()
        if wandb is None:
            print("[warn] wandb not available; skipping upload.")
            return
        tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()] if args.wandb_tags else None
        run = wandb.init(
            project=args.wandb_project or None,
            entity=args.wandb_entity or None,
            name=(args.wandb_run_name or (f"perm_recall/{task_name}" if task_name else None)),
            tags=tags,
            job_type="perm_recall_plot",
            reinit=True,
        )
        wandb.log({
            "perm_recall/acc_original": a_orig,
            "perm_recall/acc_cyclic": a_cyc,
            "perm_recall/acc_full_or_avail": a_full,
            "perm_recall/recall_std_original": s_orig,
            "perm_recall/recall_std_cyclic": s_cyc,
            "perm_recall/recall_std_full_or_avail": s_full,
            "perm_recall/plot": wandb.Image(out_path, caption=os.path.basename(out_path)),
        })
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()

