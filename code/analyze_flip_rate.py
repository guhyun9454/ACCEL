#!/usr/bin/env python3
"""
Flip Rate (order sensitivity) analysis for cyclic rotations.

Goal: show that low-confidence samples are more unstable under option order rotations.
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
    num_few_shot = int(eval_args[1]) if len(eval_args) > 1 else 0
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


def _load_one_file(fp: str, option_id_set: Optional[str], require_cyclic: bool = True, flip_only: bool = False):
    gaps: List[float] = []
    flip_any: List[float] = []
    pair_dis: List[float] = []
    t_to_f: List[float] = []
    f_to_t_list: List[float] = []

    k_local: Optional[int] = None
    option_ids_local: Optional[List[str]] = None
    perm_list = None
    identity_idx = None
    cyc_indices = None

    for d in _iter_result_rows(fp):
        if not flip_only:
            ideal = d.get("ideal", None)
            if not isinstance(ideal, str):
                continue
        probs = d.get("probs", None)
        if not isinstance(probs, list) or len(probs) == 0:
            continue
        if not isinstance(probs[0], list):
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

            cyc_perms = _rotations(k_local)
            cyc_indices = []
            for p in cyc_perms:
                if p in perm_list:
                    cyc_indices.append(perm_list.index(p))
            if require_cyclic and (len(cyc_indices) != k_local):
                return None

        if k != k_local:
            continue
        assert perm_list is not None and identity_idx is not None and cyc_indices is not None

        bp = np.asarray(probs[identity_idx], dtype=np.float64)
        if bp.ndim != 1 or bp.size != k_local:
            continue
        gap = _gap_top1_top2(bp)

        preds_content = []
        for pi in cyc_indices:
            lp = np.asarray(probs[pi], dtype=np.float64)
            if lp.ndim != 1 or lp.size != k_local:
                preds_content = []
                break
            pred_letter = int(np.argmax(lp))
            p = perm_list[pi] 
            preds_content.append(int(p[pred_letter]))
        if not preds_content:
            continue

        if not flip_only:
            if ideal not in option_ids_local:
                continue
            ideal_idx = int(option_ids_local.index(ideal))
            correct_identity = int(preds_content[0]) == int(ideal_idx)
            any_rotation_correct = any(int(preds_content[i]) == int(ideal_idx) for i in range(len(preds_content)))
            any_rotation_wrong = any(int(preds_content[i]) != int(ideal_idx) for i in range(len(preds_content)))
            t_to_f.append(1.0 if (correct_identity and any_rotation_wrong) else 0.0)
            f_to_t_list.append(1.0 if ((not correct_identity) and any_rotation_correct) else 0.0)

        gaps.append(float(gap))
        flip_any.append(1.0 if len(set(preds_content)) > 1 else 0.0)
        pair_dis.append(_pairwise_disagree(preds_content))

    if k_local is None or len(gaps) == 0:
        return None
    out: Dict = {
        "file": fp,
        "k": int(k_local),
        "option_ids": option_ids_local,
        "gap": np.asarray(gaps, dtype=np.float64),
        "flip_any": np.asarray(flip_any, dtype=np.float64),
        "pair_disagree": np.asarray(pair_dis, dtype=np.float64),
    }
    if flip_only:
        out["t_to_f"] = None
        out["f_to_t"] = None
    else:
        out["t_to_f"] = np.asarray(t_to_f, dtype=np.float64)
        out["f_to_t"] = np.asarray(f_to_t_list, dtype=np.float64)
    return out


def _analyze_files(
    files: List[str],
    option_id_set: Optional[str],
    n_bins: int,
    min_bin_n: int,
    flip_only: bool = False,
):
    per_file = []
    skipped = 0
    for fp in files:
        obj = _load_one_file(fp, option_id_set=option_id_set, require_cyclic=True, flip_only=flip_only)
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
    has_tf_ft = bool(per_file and per_file[0].get("t_to_f") is not None)
    for o in per_file:
        gap = o["gap"]
        flip = o["flip_any"]
        pair = o["pair_disagree"]
        t2f = o.get("t_to_f")
        f2t = o.get("f_to_t")
        n_total += int(gap.size)
        order = np.argsort(gap, kind="mergesort") 
        idx_bins = _equal_count_bins(order, int(n_bins))
        for i, idx in enumerate(idx_bins):
            N = int(idx.size)
            if N < int(min_bin_n):
                continue
            row: Dict = {
                "n": N,
                "gap_mean": float(np.mean(gap[idx])),
                "flip": float(np.mean(flip[idx])),
                "pair": float(np.mean(pair[idx])),
            }
            if t2f is not None and f2t is not None:
                row["t_to_f"] = float(np.mean(t2f[idx]))
                row["f_to_t"] = float(np.mean(f2t[idx]))
            bins_by_i[i].append(row)

    rows = []
    for i in range(int(n_bins)):
        items = bins_by_i.get(i, [])
        if not items:
            continue
        flips = np.asarray([it["flip"] for it in items], dtype=np.float64)
        pairs = np.asarray([it["pair"] for it in items], dtype=np.float64)
        gaps = np.asarray([it["gap_mean"] for it in items], dtype=np.float64)
        Ns = np.asarray([it["n"] for it in items], dtype=np.float64)
        r: Dict = {
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
        if has_tf_ft and items[0].get("t_to_f") is not None:
            t2f = np.asarray([it["t_to_f"] for it in items], dtype=np.float64)
            f2t = np.asarray([it["f_to_t"] for it in items], dtype=np.float64)
            r["t_to_f_mean"] = float(np.mean(t2f))
            r["t_to_f_std"] = float(np.std(t2f)) if len(t2f) > 1 else 0.0
            r["f_to_t_mean"] = float(np.mean(f2t))
            r["f_to_t_std"] = float(np.std(f2t)) if len(f2t) > 1 else 0.0
        rows.append(r)
    rows = sorted(rows, key=lambda r: int(r["bin"]))

    return {
        "k": int(k_seen),
        "option_ids": option_ids,
        "n_total": int(n_total),
        "n_files": int(len(per_file)),
        "n_files_skipped": int(skipped),
        "n_bins": int(n_bins),
        "has_tf_ft": has_tf_ft,
        "bins": rows,
    }


def _print_table(rep: dict):
    has_tf_ft = rep.get("has_tf_ft", False)
    if has_tf_ft:
        print("pct_hi\tbin\tn_files\tN_mean\tgap_mean\tflip_mean±std\tpair_mean±std\tT→F_mean±std\tF→T_mean±std")
        for r in rep["bins"]:
            print(
                f"{r['pct_hi']}%\t{r['bin']}\t{r['n_files']}\t{r['n_mean']:.1f}\t{r['gap_mean']:.4f}\t"
                f"{r['flip_mean']:.4f}±{r['flip_std']:.4f}\t{r['pair_mean']:.4f}±{r['pair_std']:.4f}\t"
                f"{r['t_to_f_mean']:.4f}±{r['t_to_f_std']:.4f}\t{r['f_to_t_mean']:.4f}±{r['f_to_t_std']:.4f}"
            )
    else:
        print("pct_hi\tbin\tn_files\tN_mean\tgap_mean\tflip_mean±std\tpair_mean±std")
        for r in rep["bins"]:
            print(
                f"{r['pct_hi']}%\t{r['bin']}\t{r['n_files']}\t{r['n_mean']:.1f}\t{r['gap_mean']:.4f}\t"
                f"{r['flip_mean']:.4f}±{r['flip_std']:.4f}\t{r['pair_mean']:.4f}±{r['pair_std']:.4f}"
            )


def _save_plot(reports_dict: Dict[str, dict], out_path: str, title: str):
    plt = _try_import_matplotlib()
    if plt is None:
        print("[warn] matplotlib not available; skipping plot.")
        return None

    # 색상 지정 (빨강, 파랑, 초록, 보라, 주황)
    colors = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD", "#F39C12"]
    
    # T->F 등 부가 지표를 그릴지 결정 (데이터셋이 1개일 때만 복잡한 선들을 그림)
    plot_tf_ft = len(reports_dict) == 1

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.2) if plot_tf_ft else (8.5, 4.8), dpi=170)

    for idx, (name, rep) in enumerate(reports_dict.items()):
        bins = rep["bins"]
        if not bins:
            continue
        
        color = colors[idx % len(colors)]
        xs = np.asarray([b["pct_hi"] for b in bins], dtype=np.float64)
        y_flip = np.asarray([b["flip_mean"] for b in bins], dtype=np.float64)
        y_flip_std = np.asarray([b["flip_std"] for b in bins], dtype=np.float64)

        # Flip Rate 그리기
        label_name = f"Flip rate ({name})" if len(reports_dict) > 1 else "Flip rate"
        ax.plot(xs, y_flip, marker="o", linewidth=2.0, color=color, label=label_name)
        ax.fill_between(xs, y_flip - y_flip_std, y_flip + y_flip_std, color=color, alpha=0.18, linewidth=0)

        # 데이터셋이 1개일 때만 T->F / F->T 추가
        if plot_tf_ft and rep.get("has_tf_ft", False):
            y_t2f = np.asarray([b["t_to_f_mean"] for b in bins], dtype=np.float64)
            y_t2f_std = np.asarray([b["t_to_f_std"] for b in bins], dtype=np.float64)
            y_f2t = np.asarray([b["f_to_t_mean"] for b in bins], dtype=np.float64)
            y_f2t_std = np.asarray([b["f_to_t_std"] for b in bins], dtype=np.float64)
            ax.plot(xs, y_t2f, marker="s", linewidth=1.8, color="#8E44AD", label="T→F (correct→wrong)")
            ax.fill_between(xs, y_t2f - y_t2f_std, y_t2f + y_t2f_std, color="#8E44AD", alpha=0.18, linewidth=0)
            ax.plot(xs, y_f2t, marker="^", linewidth=1.8, color="#27AE60", label="F→T (wrong→correct)")
            ax.fill_between(xs, y_f2t - y_f2t_std, y_f2t + y_f2t_std, color="#27AE60", alpha=0.18, linewidth=0)

    ax.set_xlabel("Confidence percentile bin (top1-top2 probability gap)")
    ax.set_ylabel("Rate")
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
    ap.add_argument("--jsonl_paths", type=str, nargs="+", default=None, help="Explicit jsonl file paths to analyze.")
    ap.add_argument("--jsonl_glob", type=str, default="*.jsonl")
    ap.add_argument("--run_idx", type=int, default=None)
    ap.add_argument("--models", type=str, nargs="+", default=None)
    ap.add_argument("--results_root", type=str, default="",
                    help="Root dir containing results_<task>/ (e.g. results_csqa/, results_mmlu/). Default: script dir. Required for --models + --eval_names if results live elsewhere.")
    ap.add_argument("--eval_names", type=str, nargs="+", default=[], help="List of eval names (e.g., csqa,0,full mmlu,0,full).")
    ap.add_argument("--option_id_set", type=str, default=None,
                    help="Option IDs for all evals (e.g. ABCD). Path becomes .../task_full_id-ABCD.")
    ap.add_argument("--option_id_sets", type=str, nargs="+", default=None,
                    help="Per-eval option IDs, same order as --eval_names (e.g. ABCDE ABCD for csqa mmlu). Overrides --option_id_set.")

    ap.add_argument("--flip_only", action="store_true")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--min_bin_n", type=int, default=1)
    ap.add_argument("--out", type=str, default="flip_rate_deciles.png")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--save_plot", action="store_true")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default="capde")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args, unknown = ap.parse_known_args()

    reports = {}

    # 1. 모델과 여러 eval_names가 주어졌을 때 각각 분석
    if args.models and args.eval_names:
        results_root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        model_list = [str(m).strip() for m in args.models if str(m).strip()]
        
        print(f"[info] Running multi-dataset analysis for: {args.eval_names}")
        
        print(f"[info] results_root = {results_root}")
        id_sets = getattr(args, "option_id_sets", None)
        if id_sets is not None and len(id_sets) != len(args.eval_names):
            raise SystemExit("--option_id_sets length must match --eval_names (e.g. --option_id_sets ABCDE ABCD for two evals).")
        for idx, eval_n in enumerate(args.eval_names):
            option_id = (id_sets[idx] if id_sets is not None else args.option_id_set)
            files = []
            tried_dirs = []
            for m in model_list:
                rdir = _compute_results_dir(results_root, eval_n, m, option_id)
                tried_dirs.append(rdir)
                f = _discover_jsonl_files(rdir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)
                files.extend(f)
            files = sorted(set(files))
            if files:
                try:
                    rep = _analyze_files(
                        files=files,
                        option_id_set=option_id,
                        n_bins=int(args.n_bins),
                        min_bin_n=int(args.min_bin_n),
                        flip_only=bool(args.flip_only),
                    )
                    reports[eval_n] = rep
                except Exception as e:
                    print(f"[error] Failed to analyze {eval_n}: {e}")
            else:
                print(f"[warn] No files found for eval_name: {eval_n}")
                print(f"[warn]   Looked under results_root: {results_root}")
                for d in tried_dirs[:3]:
                    print(f"[warn]     - {d}")
                if len(tried_dirs) > 3:
                    print(f"[warn]     ... and {len(tried_dirs) - 3} more model paths.")
                if option_id is None:
                    print(f"[warn]   If your dirs are like .../csqa_full_id-ABCDE or .../mmlu_full_id-ABCD, pass --option_id_set or --option_id_sets.")

    # 2. 기존 방식 (results_dir 단일 디렉토리나 jsonl_paths 지정 시)
    else:
        results_dir = str(args.results_dir).strip()
        files = []
        if args.jsonl_paths:
            files = [os.path.abspath(p) for p in args.jsonl_paths]
            if not results_dir and files:
                results_dir = os.path.dirname(os.path.abspath(files[0]))
        elif results_dir:
            files = _discover_jsonl_files(results_dir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)
        
        if files:
            dataset_name = args.eval_names[0] if args.eval_names else "Dataset"
            rep = _analyze_files(
                files=files,
                option_id_set=args.option_id_set,
                n_bins=int(args.n_bins),
                min_bin_n=int(args.min_bin_n),
                flip_only=bool(args.flip_only),
            )
            reports[dataset_name] = rep

    if not reports:
        raise SystemExit("No data to analyze.")

    # 각 데이터셋별 테이블 출력
    for name, rep in reports.items():
        print(f"\n==== Results for: {name} ====")
        _print_table(rep)

    # 통합 Plot 저장
    out_plot = None
    if bool(args.save_plot):
        out_path = str(args.out)
        title = str(args.title) if args.title else "Flip rate vs confidence deciles (cyclic rotations)"
        out_plot = _save_plot(reports, out_path, title)
        if out_plot:
            print(f"\nSaved plot: {out_plot}")


if __name__ == "__main__":
    main()