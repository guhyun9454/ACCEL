#!/usr/bin/env python3
"""
Flip Rate (order sensitivity) analysis for cyclic rotations.

Goal: show that low-confidence samples are more unstable under option order rotations.

We define per-sample cyclic predictions as:
  for each cyclic rotation perm p (letter-position -> content-index),
  pred_letter = argmax(probs_letter_space)
  pred_content = p[pred_letter]

Flip metrics (per sample):
  - flip_any: 1 if not all pred_content across rotations are identical
  - pairwise_disagree: average_{i<j} 1[pred_i != pred_j]

We then bin samples by base confidence gap (top1-top2) into equal-count deciles (per file),
and average flip metrics across files (mean±std).
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

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


def _compute_results_dir(code_dir: str, eval_name: str, pretrained_model_path: str, option_id_set: Optional[str]):
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
    return os.path.join(code_dir, save_path)


def _discover_jsonl_files(results_dir: str, jsonl_glob: str = "*.jsonl", run_idx: Optional[int] = None) -> List[str]:
    pats = [
        os.path.join(results_dir, str(jsonl_glob)),
        os.path.join(results_dir, "**", str(jsonl_glob)),
    ]
    files: List[str] = []
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    files = [
        p for p in files
        if p.endswith(".jsonl")
        and (not p.endswith("_curve.jsonl"))
        and (not p.endswith("_pride_curve.jsonl"))
    ]
    if run_idx is not None:
        tok = f"run{int(run_idx)}"
        files = [p for p in files if tok in os.path.basename(p)]
    return files


def _iter_result_rows(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "result":
                continue
            data = obj.get("data", {}) or {}
            if isinstance(data, dict):
                yield data


def _rotations(k: int):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _infer_perm_list(k: int, perm_count: int):
    from itertools import permutations

    if perm_count == math.factorial(k):
        return list(sorted(permutations(range(k))))
    if perm_count == k:
        return _rotations(k)
    full = list(sorted(permutations(range(k))))
    if perm_count <= len(full):
        return full[:perm_count]
    return full


def _gap_top1_top2(bp: np.ndarray) -> float:
    s = np.sort(np.asarray(bp, dtype=np.float64))[::-1]
    if s.size <= 1:
        return 0.0
    return float(s[0] - s[1])


def _equal_count_bins(order_idx: np.ndarray, n_bins: int) -> List[np.ndarray]:
    chunks = np.array_split(order_idx, int(max(1, int(n_bins))))
    return [c.astype(np.int64) for c in chunks]


def _pairwise_disagree(preds: List[int]) -> float:
    k = len(preds)
    if k <= 1:
        return 0.0
    tot = 0
    diff = 0
    for i in range(k):
        for j in range(i + 1, k):
            tot += 1
            diff += 1 if int(preds[i]) != int(preds[j]) else 0
    return float(diff) / float(tot) if tot > 0 else 0.0


def _load_one_file(fp: str, option_id_set: Optional[str], require_cyclic: bool = True):
    """
    Returns dict with arrays:
      conf_gap: (N,)
      flip_any: (N,)
      pair_disagree: (N,)
    """
    gaps: List[float] = []
    flip_any: List[float] = []
    pair_dis: List[float] = []

    k_local: Optional[int] = None
    option_ids_local: Optional[List[str]] = None
    perm_list = None
    identity_idx = None
    cyc_indices = None

    for d in _iter_result_rows(fp):
        probs = d.get("probs", None)
        if not isinstance(probs, list) or len(probs) == 0:
            continue
        if not isinstance(probs[0], list):
            # base-only cache can't measure flip under rotations
            continue

        row0 = probs[0]
        if not isinstance(row0, list) or len(row0) == 0:
            continue

        k = int(len(row0))
        perm_count = int(len(probs))
        if k_local is None:
            k_local = k
            if option_id_set is not None:
                option_ids_local = list(str(option_id_set))
                if len(option_ids_local) != k_local:
                    raise ValueError(f"--option_id_set length must be {k_local}, got {len(option_ids_local)}: {option_id_set}")
            else:
                option_ids_local = list("ABCDE"[:k_local]) if k_local in (4, 5) else [str(i) for i in range(k_local)]

            perm_list = _infer_perm_list(k_local, perm_count)
            identity = tuple(range(k_local))
            identity_idx = perm_list.index(identity) if identity in perm_list else 0

            # cyclic rotation indices
            cyc_perms = _rotations(k_local)
            cyc_indices = []
            for p in cyc_perms:
                if p in perm_list:
                    cyc_indices.append(perm_list.index(p))
            if require_cyclic and (len(cyc_indices) != k_local):
                # can't form full cyclic set for flip metric
                return None

        if k != k_local:
            continue
        assert perm_list is not None and identity_idx is not None and cyc_indices is not None

        bp = np.asarray(probs[identity_idx], dtype=np.float64)
        if bp.ndim != 1 or bp.size != k_local:
            continue
        gap = _gap_top1_top2(bp)

        # cyclic predictions in content-space
        preds_content = []
        for pi in cyc_indices:
            lp = np.asarray(probs[pi], dtype=np.float64)
            if lp.ndim != 1 or lp.size != k_local:
                preds_content = []
                break
            pred_letter = int(np.argmax(lp))
            p = perm_list[pi]  # letter -> content
            preds_content.append(int(p[pred_letter]))
        if not preds_content:
            continue

        gaps.append(float(gap))
        flip_any.append(1.0 if len(set(preds_content)) > 1 else 0.0)
        pair_dis.append(_pairwise_disagree(preds_content))

    if k_local is None or len(gaps) == 0:
        return None
    return {
        "file": fp,
        "k": int(k_local),
        "option_ids": option_ids_local,
        "gap": np.asarray(gaps, dtype=np.float64),
        "flip_any": np.asarray(flip_any, dtype=np.float64),
        "pair_disagree": np.asarray(pair_dis, dtype=np.float64),
    }


def _analyze_files(
    files: List[str],
    option_id_set: Optional[str],
    n_bins: int,
    min_bin_n: int,
):
    per_file = []
    skipped = 0
    for fp in files:
        obj = _load_one_file(fp, option_id_set=option_id_set, require_cyclic=True)
        if obj is None:
            skipped += 1
            continue
        per_file.append(obj)

    if not per_file:
        raise ValueError("No usable files. Make sure you point to perm/full/cyclic caches with permutation probs.")

    k_seen = int(per_file[0]["k"])
    option_ids = per_file[0]["option_ids"]
    for o in per_file[1:]:
        if int(o["k"]) != k_seen:
            raise ValueError("Mixed k across files; filter your selection.")
        if list(o["option_ids"]) != list(option_ids):
            raise ValueError("Mixed option_id_set across files; pass --option_id_set explicitly.")

    bins_by_i: Dict[int, List[dict]] = {i: [] for i in range(int(n_bins))}
    n_total = 0
    for o in per_file:
        gap = o["gap"]
        flip = o["flip_any"]
        pair = o["pair_disagree"]
        n_total += int(gap.size)
        order = np.argsort(gap, kind="mergesort")  # low conf first
        idx_bins = _equal_count_bins(order, int(n_bins))
        for i, idx in enumerate(idx_bins):
            N = int(idx.size)
            if N < int(min_bin_n):
                continue
            bins_by_i[i].append(
                {
                    "n": N,
                    "gap_mean": float(np.mean(gap[idx])),
                    "flip": float(np.mean(flip[idx])),
                    "pair": float(np.mean(pair[idx])),
                }
            )

    rows = []
    for i in range(int(n_bins)):
        items = bins_by_i.get(i, [])
        if not items:
            continue
        flips = np.asarray([it["flip"] for it in items], dtype=np.float64)
        pairs = np.asarray([it["pair"] for it in items], dtype=np.float64)
        gaps = np.asarray([it["gap_mean"] for it in items], dtype=np.float64)
        Ns = np.asarray([it["n"] for it in items], dtype=np.float64)
        rows.append(
            {
                "bin": int(i),
                "pct_hi": int((i + 1) * (100 / int(n_bins))),
                "n_files": int(len(items)),
                "n_mean": float(np.mean(Ns)),
                "gap_mean": float(np.mean(gaps)),
                "gap_std": float(np.std(gaps)) if len(gaps) > 1 else 0.0,
                "flip_mean": float(np.mean(flips)),
                "flip_std": float(np.std(flips)) if len(flips) > 1 else 0.0,
                "pair_mean": float(np.mean(pairs)),
                "pair_std": float(np.std(pairs)) if len(pairs) > 1 else 0.0,
            }
        )
    rows = sorted(rows, key=lambda r: int(r["bin"]))

    return {
        "k": int(k_seen),
        "option_ids": option_ids,
        "n_total": int(n_total),
        "n_files": int(len(per_file)),
        "n_files_skipped": int(skipped),
        "n_bins": int(n_bins),
        "bins": rows,
    }


def _print_table(rep: dict):
    print("pct_hi\tbin\tn_files\tN_mean\tgap_mean\tflip_mean±std\tpair_disagree_mean±std")
    for r in rep["bins"]:
        print(
            f"{r['pct_hi']}%\t{r['bin']}\t{r['n_files']}\t{r['n_mean']:.1f}\t{r['gap_mean']:.4f}\t"
            f"{r['flip_mean']:.4f}±{r['flip_std']:.4f}\t{r['pair_mean']:.4f}±{r['pair_std']:.4f}"
        )


def _save_plot(rep: dict, out_path: str, title: str):
    plt = _try_import_matplotlib()
    if plt is None:
        print("[warn] matplotlib not available; skipping plot.")
        return None
    bins = rep["bins"]
    if not bins:
        return None
    xs = np.asarray([b["pct_hi"] for b in bins], dtype=np.float64)
    y = np.asarray([b["flip_mean"] for b in bins], dtype=np.float64)
    ystd = np.asarray([b["flip_std"] for b in bins], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8), dpi=170)
    ax.plot(xs, y, marker="o", linewidth=2.0, color="#C0392B", label="Flip rate (any change across cyclic)")
    ax.fill_between(xs, y - ystd, y + ystd, color="#C0392B", alpha=0.18, linewidth=0)
    ax.set_xlabel("Confidence percentile bin (per-file, equal-count)")
    ax.set_ylabel("Flip rate")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--results_dir", type=str, default="", help="Directory containing cached *.jsonl results.")
    ap.add_argument("--jsonl_paths", type=str, nargs="+", default=None, help="Explicit jsonl file paths to analyze (overrides --results_dir).")
    ap.add_argument("--jsonl_glob", type=str, default="*.jsonl")
    ap.add_argument("--run_idx", type=int, default=None)
    ap.add_argument("--models", type=str, nargs="+", default=None,
                    help="Model names/ids. Auto-collect jsonl under computed results_dir for each model.")
    ap.add_argument("--results_root", type=str, default="", help="Root directory that contains results_* folders (default: this script's directory).")
    ap.add_argument("--eval_name", type=str, default="", help="Eval name like 'arc,0,full'. Used when --models is set.")
    ap.add_argument("--option_id_set", type=str, default=None)

    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--min_bin_n", type=int, default=1)
    ap.add_argument("--out", type=str, default="flip_rate_deciles.png")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--save_plot", action="store_true")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default="capde")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    # One-shot runner (optional)
    ap.add_argument("--eval_clm_path", type=str, default="")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--pretrained_model_path", type=str, default="")
    ap.add_argument("--eval_names", type=str, nargs="+", default=[])
    ap.add_argument("--n_runs", type=int, default=1)
    ap.add_argument("--force", action="store_true")

    args, unknown = ap.parse_known_args()

    results_dir = str(args.results_dir).strip()
    files: List[str] = []

    if args.models:
        results_root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        eval_name = str(args.eval_name).strip() or (str(args.eval_names[0]).strip() if args.eval_names else "")
        if not eval_name:
            raise SystemExit("When using --models, provide --eval_name (e.g., arc,0,full) or --eval_names.")
        model_list = [str(m).strip() for m in (args.models or []) if str(m).strip()]
        by_model: Dict[str, List[str]] = {}
        for m in model_list:
            rdir = _compute_results_dir(results_root, eval_name, m, args.option_id_set)
            f = _discover_jsonl_files(rdir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)
            by_model[m] = f
            files.extend(f)
            if not results_dir and os.path.isdir(rdir):
                results_dir = rdir
        files = sorted(set(files))
        print(f"[info] selected models={len(model_list)}, jsonl_files={len(files)}")
        for m in model_list:
            print(f"[info]  - {m.split('/')[-1]}: {len(by_model.get(m, []))} files")

    if (not files) and (not results_dir) and str(args.pretrained_model_path).strip() and args.eval_names:
        eval_clm_path = str(args.eval_clm_path).strip() or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_clm.py")
        code_dir = os.path.dirname(os.path.abspath(eval_clm_path))
        eval_name = str(args.eval_names[0])
        results_dir = _compute_results_dir(code_dir, eval_name, str(args.pretrained_model_path), args.option_id_set)
        files = _discover_jsonl_files(results_dir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)
        if (not files) and (not bool(args.skip_eval)):
            cmd = [sys.executable, os.path.abspath(eval_clm_path)]
            cmd += ["--pretrained_model_path", str(args.pretrained_model_path)]
            cmd += ["--eval_names", eval_name]
            if args.option_id_set:
                cmd += ["--option_id_set", str(args.option_id_set)]
            if int(args.n_runs) != 1:
                cmd += ["--n_runs", str(int(args.n_runs))]
            if bool(args.force):
                cmd += ["--force"]
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

    if not files:
        if args.jsonl_paths:
            files = [os.path.abspath(p) for p in (args.jsonl_paths or [])]
            if not results_dir and files:
                results_dir = os.path.dirname(os.path.abspath(files[0]))
        else:
            if not results_dir:
                raise SystemExit("Provide --models, --jsonl_paths, --results_dir, or eval_clm args.")
            files = _discover_jsonl_files(results_dir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)

    if not files:
        raise SystemExit("No jsonl files selected.")

    rep = _analyze_files(
        files=files,
        option_id_set=args.option_id_set,
        n_bins=int(args.n_bins),
        min_bin_n=int(args.min_bin_n),
    )
    _print_table(rep)

    out_plot = None
    out_path = str(args.out)
    if results_dir and (not os.path.isabs(out_path)):
        out_path = os.path.join(results_dir, out_path)
    if bool(args.save_plot):
        title = str(args.title) if args.title else "Flip rate vs confidence deciles (cyclic rotations)"
        out_plot = _save_plot(rep, out_path, title)
        if out_plot:
            print(f"Saved plot: {out_plot}")

    if bool(args.wandb):
        wandb = _try_import_wandb()
        if wandb is None:
            print("[warn] wandb not available; skipping W&B logging.")
            return
        run = wandb.init(
            project=getattr(args, "wandb_project", None) or None,
            entity=getattr(args, "wandb_entity", None) or None,
            name=getattr(args, "wandb_run_name", None) or None,
            job_type="flip_rate",
            reinit=True,
        )
        try:
            for r in rep["bins"]:
                wandb.log({
                    "pct_hi": int(r["pct_hi"]),
                    "n_files": int(r["n_files"]),
                    "n_mean": float(r["n_mean"]),
                    "gap_mean": float(r["gap_mean"]),
                    "flip_mean": float(r["flip_mean"]),
                    "flip_std": float(r["flip_std"]),
                    "pair_mean": float(r["pair_mean"]),
                    "pair_std": float(r["pair_std"]),
                })
            if out_plot:
                wandb.log({"plots/flip_rate": wandb.Image(out_plot)})
        except Exception as e:
            print(f"[warn] W&B logging failed: {e}")
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()

