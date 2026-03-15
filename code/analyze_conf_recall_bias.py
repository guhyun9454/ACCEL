#!/usr/bin/env python3
import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import math
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

def _set_font():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


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


def _extract_base_probs_and_pred(data: dict) -> Tuple[Optional[np.ndarray], Optional[int]]:
    probs = data.get("probs", None)
    if not isinstance(probs, list) or len(probs) == 0:
        return None, None
    if isinstance(probs[0], list):
        row0 = probs[0]
        if not isinstance(row0, list) or len(row0) == 0:
            return None, None
        k = int(len(row0))
        perm_count = int(len(probs))
        perm_list = _infer_perm_list(k, perm_count)
        identity = tuple(range(k))
        identity_idx = perm_list.index(identity) if identity in perm_list else 0
        bp = np.asarray(probs[identity_idx], dtype=np.float64)
    else:
        bp = np.asarray(probs, dtype=np.float64)
    if bp.ndim != 1 or bp.size == 0:
        return None, None
    pred_idx = int(np.argmax(bp))
    return bp, pred_idx


def _gap_top1_top2(bp: np.ndarray) -> float:
    s = np.sort(np.asarray(bp, dtype=np.float64))[::-1]
    if s.size <= 1:
        return 0.0
    return float(s[0] - s[1])


def _analyze(
    files: List[str],
    option_id_set: Optional[str],
    n_bins: int,
    min_bin_n: int,
):
    def _load_one_file(fp: str):
        confs1, yt1, yp1 = [], [], []
        k_local, option_ids_local = None, None
        
        for d in _iter_result_rows(fp):
            ideal = d.get("ideal", None)
            if not isinstance(ideal, str):
                continue
            options = d.get("options", None)
            k = int(len(options)) if isinstance(options, list) and len(options) > 0 else -1
            
            bp, pred_idx = _extract_base_probs_and_pred(d)
            if bp is None or pred_idx is None:
                continue
            k = int(bp.size) if k <= 0 else k
            if k <= 0: continue

            if k_local is None:
                k_local = k
                if option_id_set is not None:
                    option_ids_local = list(str(option_id_set))
                else:
                    option_ids_local = list("ABCDE"[:k_local]) if k_local in (4, 5) else [str(i) for i in range(k_local)]
            
            if k_local != k or ideal not in option_ids_local:
                continue
                
            confs1.append(_gap_top1_top2(bp))
            yt1.append(int(option_ids_local.index(ideal)))
            yp1.append(int(pred_idx))
            
        if k_local is None or len(confs1) == 0:
            return None
            
        return {
            "k": int(k_local),
            "option_ids": option_ids_local,
            "conf": np.asarray(confs1, dtype=np.float64),
            "yt": np.asarray(yt1, dtype=np.int64),
            "yp": np.asarray(yp1, dtype=np.int64),
            "file": fp,
        }

    per_file = [obj for fp in files if (obj := _load_one_file(fp)) is not None]
    if not per_file:
        raise ValueError("No usable samples found.")

    k_seen = int(per_file[0]["k"])
    option_ids = per_file[0]["option_ids"]

    # 글로벌 병합 데이터 (모든 모델/파일 샘플 통합)
    conf_all = np.concatenate([o["conf"] for o in per_file], axis=0)
    yt_all = np.concatenate([o["yt"] for o in per_file], axis=0)
    yp_all = np.concatenate([o["yp"] for o in per_file], axis=0)
    fidx_all = np.concatenate([np.full(o["conf"].size, f_idx, dtype=np.int64) for f_idx, o in enumerate(per_file)])

    # ==========================================================
    # 여기가 Percentile(백분위수)로 분할하는 핵심 로직입니다.
    # Confidence 순으로 정렬 후 정확히 n_bins 개수로 쪼갭니다 (예: 10등분 -> Decile)
    # ==========================================================
    order = np.argsort(conf_all, kind="mergesort")
    chunks = np.array_split(order, int(n_bins))

    rows = []
    
    # "mean_over_files" 모드 하드코딩 적용 (논문 테이블용 평균 산출)
    for i, idx_arr in enumerate(chunks):
        c_chunk = conf_all[idx_arr]
        yt_chunk = yt_all[idx_arr]
        yp_chunk = yp_all[idx_arr]
        fidx_chunk = fidx_all[idx_arr]
        
        accs, rstds, cms, Ns = [], [], [], []
        
        for f_idx in range(len(per_file)):
            m = (fidx_chunk == f_idx)
            n_local = int(np.sum(m))
            if n_local < 1:  # 최소 샘플 조건
                continue
                
            # Accuracy 계산
            accs.append(float(np.mean(yp_chunk[m] == yt_chunk[m])) * 100.0) # 퍼센트
            
            # Recall Std 계산 (작성자님 코드와 완벽히 동일한 수식)
            recalls = []
            for c in range(k_seen):
                class_mask = (yt_chunk[m] == c)
                denom = int(np.sum(class_mask))
                if denom > 0:
                    rec = np.mean((yp_chunk[m][class_mask] == c).astype(float)) * 100.0
                    recalls.append(rec)
            
            # 해당 파일, 해당 Bin의 recall_std 추가
            if len(recalls) > 0:
                rstds.append(float(np.std(recalls)))
                
            cms.append(float(np.mean(c_chunk[m])))
            Ns.append(n_local)
            
        if not accs or not rstds:
            continue
            
        # 해당 구간(Percentile Bin)에 대한 모델들의 평균값 산출
        rows.append({
            "bin": int(i),
            "n_files": len(accs),
            "n_mean": float(np.mean(Ns)),
            "conf_mean": float(np.mean(cms)),
            "conf_mean_std": float(np.std(cms)) if len(cms) > 1 else 0.0,
            "acc": float(np.mean(accs)),
            "acc_std": float(np.std(accs)) if len(accs) > 1 else 0.0,
            "recall_std": float(np.mean(rstds)),
            "recall_std_std": float(np.std(rstds)) if len(rstds) > 1 else 0.0,
        })

    return {
        "k": k_seen,
        "option_ids": option_ids,
        "bins": sorted(rows, key=lambda r: int(r["bin"]))
    }


def _print_table(rep: dict):
    n_bins = int(len(rep.get("bins", []) or [])) or 10
    print("Percentile\tN_mean\tAvg_Conf\tAcc(%)±std\tRecall_Std(%)±std")
    for r in rep["bins"]:
        # 하위 X 퍼센트 표시 (예: 10%, 20% ...)
        pct_hi = int((int(r["bin"]) + 1) * (100 / max(1, int(n_bins))))
        print(f"Top {100 - pct_hi + 10:02d}~{100 - pct_hi:02d}%\t{r['n_mean']:.1f}\t{r['conf_mean']:.4f}\t"
              f"{r['acc']:.2f}±{r['acc_std']:.2f}\t{r['recall_std']:.2f}±{r['recall_std_std']:.2f}")


def _save_plot(rep: dict, out_path: str, title: str):
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    _set_font()

    bins = rep["bins"]
    if not bins:
        return None

    xs = np.asarray([b["conf_mean"] for b in bins], dtype=np.float64)
    rstd = np.asarray([b["recall_std"] for b in bins], dtype=np.float64)
    rstd_std = np.asarray([b.get("recall_std_std", 0.0) for b in bins], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=180)
    
    # Recall Std Plot
    ax1.plot(xs, rstd, marker="o", linewidth=2.0, color="#8E44AD", label="Recall Std (%)")
    ax1.fill_between(xs, rstd - rstd_std, rstd + rstd_std, color="#8E44AD", alpha=0.15, linewidth=0)
    
    ax1.set_ylabel("Recall Standard Deviation (%)", fontsize=11)
    ax1.set_xlabel("Confidence Gap (Top 1 - Top 2)", fontsize=11)
    ax1.set_title(title, fontsize=13, pad=15)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--results_dir", type=str, default="")
    ap.add_argument("--jsonl_paths", type=str, nargs="+", default=None)
    ap.add_argument("--jsonl_glob", type=str, default="*.jsonl")
    ap.add_argument("--run_idx", type=int, default=None)
    ap.add_argument("--models", type=str, nargs="+", default=None)
    ap.add_argument("--results_root", type=str, default="")
    ap.add_argument("--eval_name", type=str, default="")
    ap.add_argument("--out", type=str, default="conf_recall_bias.png")
    ap.add_argument("--title", type=str, default="Recall Std vs Confidence Gap")
    ap.add_argument("--n_bins", type=int, default=10, help="Number of percentiles (10 = Decile)")
    ap.add_argument("--min_bin_n", type=int, default=10)
    ap.add_argument("--aggregate_mode", type=str, default="mean_over_files") # 호환성
    ap.add_argument("--table_mode", type=str, default="summary") # 호환성
    ap.add_argument("--save_plot", action="store_true")

    args, unknown = ap.parse_known_args()

    results_dir = str(args.results_dir).strip()
    files: List[str] = []

    if args.models:
        results_root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        eval_name = str(args.eval_name).strip() or (str(args.eval_names[0]).strip() if args.eval_names else "")
        model_list = [str(m).strip() for m in (args.models or []) if str(m).strip()]

        for m in model_list:
            rdir = _compute_results_dir(results_root, eval_name, m, args.option_id_set)
            f = _discover_jsonl_files(rdir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)
            files.extend(f)

        files = sorted(set(files))
        if not results_dir:
            for m in model_list:
                rdir = _compute_results_dir(results_root, eval_name, m, args.option_id_set)
                if os.path.isdir(rdir):
                    results_dir = rdir
                    break

    if not files:
        if args.jsonl_paths:
            files = [os.path.abspath(p) for p in (args.jsonl_paths or [])]
        elif results_dir:
            files = _discover_jsonl_files(results_dir, jsonl_glob=str(args.jsonl_glob), run_idx=args.run_idx)

    if not files:
        raise SystemExit("No jsonl files selected.")

    save_root = results_dir if results_dir else os.path.dirname(os.path.abspath(files[0]))

    rep = _analyze(
        files=files,
        option_id_set=args.option_id_set,
        n_bins=int(args.n_bins),
        min_bin_n=int(args.min_bin_n),
    )

    _print_table(rep)

    out_path = str(args.out)
    if save_root and (not os.path.isabs(out_path)):
        out_path = os.path.join(save_root, out_path)
        
    if bool(args.save_plot):
        out_plot = _save_plot(rep, out_path, str(args.title))
        if out_plot:
            print(f"Saved plot: {out_plot}")

if __name__ == "__main__":
    main()