#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ece_viz.py
- 디렉토리 구조: <root>/<token>/<dataset>/<model>/<dataset>.jsonl
- 각 jsonl 한 줄은 다음 둘 중 하나를 가정
  (A) {"type":"result","data":{ ... "probs":[...], "correct":true/false, "qid":... }}
  (B) 평면형 { "probs":[...], "correct":true/false, "qid":... }  (+ 일부 키 변형 허용)
- 출력:
  1) 히스토그램(정답/오답 confidence 분포, T0/T1/T2 색상 고정: R/B/G)
  2) 리라이어빌리티 다이어그램(토큰별)
  3) ECE/정확도 요약 TSV

사용 예)
  python code/ece_viz.py --root results --datasets arc csqa \
    --models "Qwen/Qwen2.5-1.5B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" \
    --outdir viz_out/ece
"""

import os, json, argparse, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# ───── 공통 유틸 ─────
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def flatten_record(rec: dict) -> dict:
    # {"type":"result","data":{...}} → {...}
    if isinstance(rec, dict) and "data" in rec and isinstance(rec["data"], dict):
        d = dict(rec["data"])
        if "qid" not in d and "idx" in d:
            d["qid"] = str(d["idx"])
        return d
    return rec

def fold_probs(p: np.ndarray, n_opts_hint: Optional[int] = None) -> np.ndarray:
    """p가 2D면 평균, 길이가 2*C면 접어서 C로 합산."""
    arr = np.array(p, dtype=float)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    if arr.ndim == 1 and arr.size % 2 == 0 and (n_opts_hint is None or arr.size == 2*n_opts_hint):
        arr = arr.reshape(2, arr.size//2).sum(axis=0)
    # 정규화
    arr = np.clip(arr, 1e-12, None)
    arr = arr / arr.sum()
    return arr

def ece_top1(probs: np.ndarray, correct: np.ndarray, n_bins: int=15) -> float:
    """multiclass top-1 ECE"""
    conf = probs.max(axis=1)
    bins = np.linspace(0., 1., n_bins+1)
    ece = 0.0
    N = len(conf)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else ((conf > lo) & (conf <= hi))
        if not np.any(mask): 
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece += (mask.sum()/N) * abs(acc_b - conf_b)
    return float(ece)

# ───── 로딩 ─────
def collect_models(root: str, tokens: list[str], datasets: list[str]) -> list[str]:
    mset = set()
    for t in tokens:
        for ds in datasets:
            base = Path(root)/t/ds
            if base.is_dir():
                for m in base.iterdir():
                    if m.is_dir():
                        mset.add(m.name)
    return sorted(mset)

def load_token_records(fp: Path):
    if not fp.is_file():
        return []
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rows.append(flatten_record(rec))
    return rows

def extract_conf_correct(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """records → (conf, correct). probs 없으면 해당 항목 스킵"""
    confs, rights = [], []
    for r in rows:
        p = r.get("probs", r.get("all_probs"))
        if p is None:
            continue
        probs = fold_probs(p)
        confs.append(float(np.max(probs)))
        # correct 플래그 우선
        c = r.get("correct", None)
        if isinstance(c, str):
            c = c.lower() in ("true","1","yes")
        if isinstance(c, bool):
            rights.append(1 if c else 0)
        else:
            # correct가 없으면 pred vs gold로 복원 시도
            # pred_idx 있으면 쓰고, 없으면 argmax
            pred_idx = None
            for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx","output_idx"):
                if k in r:
                    try:
                        vi = int(r[k]); 
                        if vi >= 0: pred_idx = vi
                        break
                    except Exception:
                        pass
            if pred_idx is None:
                pred_idx = int(np.argmax(probs))
            gold_idx = None
            for k in ("gold_idx","label_idx","correct_idx","target_idx","solution_idx"):
                if k in r:
                    try:
                        vi = int(r[k]); 
                        if vi >= 0: gold_idx = vi
                        break
                    except Exception:
                        pass
            if gold_idx is None:
                # gold 인덱스를 못 구하면 해당 샘플은 스킵
                confs.pop()
                continue
            rights.append(1 if pred_idx == gold_idx else 0)
    if not confs:
        return np.array([]), np.array([])
    return np.array(confs), np.array(rights, dtype=int)

# ───── 시각화 ─────
def plot_hist_by_token(model_tag, dataset, conf_by_tok, right_by_tok, outdir, bins=20):
    """
    conf_by_tok/right_by_tok: dict[token] -> ndarray
    토큰별(빨/파/초)로 정답/오답 히스토그램을 한 그림에 오버레이.
    """
    color_map = {"T0":"red", "T1":"blue", "T2":"green"}
    plt.figure()  # 단일 플롯
    for t in sorted(conf_by_tok.keys()):
        conf = conf_by_tok[t]; right = right_by_tok[t]
        if conf.size == 0: 
            continue
        # 정답/오답 분리
        conf_ok  = conf[right==1]
        conf_bad = conf[right==0]
        # 오버레이 히스토그램
        plt.hist(conf_bad, bins=bins, range=(0.0,1.0), alpha=0.35, color=color_map.get(t,"gray"),
                 label=f"{t} wrong (n={len(conf_bad)})", density=True)
        plt.hist(conf_ok,  bins=bins, range=(0.0,1.0), alpha=0.55, color=color_map.get(t,"gray"),
                 label=f"{t} correct (n={len(conf_ok)})", density=True)
    plt.xlabel("Confidence (top-1 prob)")
    plt.ylabel("Density")
    plt.title(f"{model_tag} — {dataset} | Confidence distribution by token")
    plt.legend(fontsize=9)
    out_png = Path(outdir)/f"{model_tag}_{dataset}_conf_hist.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_reliability_by_token(model_tag, dataset, conf_by_tok, right_by_tok, outdir, n_bins=15):
    """
    토큰별 리라이어빌리티 다이어그램 한 그림에(빨/파/초)로 그리기.
    """
    color_map = {"T0":"red", "T1":"blue", "T2":"green"}
    plt.figure()
    xs = np.linspace(0,1,101)
    plt.plot(xs, xs, linestyle="--", color="black", linewidth=1, label="Ideal")
    for t in sorted(conf_by_tok.keys()):
        conf = conf_by_tok[t]; right = right_by_tok[t]
        if conf.size == 0: 
            continue
        bins = np.linspace(0.,1.,n_bins+1)
        mids = 0.5*(bins[:-1]+bins[1:])
        accs = []
        for b in range(n_bins):
            lo,hi = bins[b], bins[b+1]
            mask = (conf >= lo) & (conf <= hi) if b==0 else ((conf > lo) & (conf <= hi))
            if mask.any():
                accs.append(right[mask].mean())
            else:
                accs.append(np.nan)
        accs = np.array(accs, dtype=float)
        plt.plot(mids, accs, marker="o", color=color_map.get(t,"gray"), label=f"{t}")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"{model_tag} — {dataset} | Reliability by token")
    plt.legend()
    out_png = Path(outdir)/f"{model_tag}_{dataset}_reliability.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ───── 메인 ─────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="결과 루트 (예: results)")
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--models", nargs="*", default=None,
                    help='특정 모델만 선택 (부분문자열 매칭 가능). 비우면 자동 탐색.')
    ap.add_argument("--outdir", default="viz_out/ece")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--n_bins_ece", type=int, default=15)
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.outdir)

    models = collect_models(args.root, args.tokens, args.datasets) if not args.models else sorted(set(args.models))
    if not models:
        print("[WARN] 모델 디렉토리를 찾지 못했습니다."); return

    # 요약 TSV 준비
    tsv_lines = ["model\tdataset\ttoken\tn\tacc\tece\n"]

    for model in models:
        for ds in args.datasets:
            conf_by_tok = {}; right_by_tok = {}
            for t in args.tokens:
                jf = Path(args.root)/t/ds/model/f"{ds}.jsonl"
                rows = load_token_records(jf)
                conf, right = extract_conf_correct(rows)
                conf_by_tok[t]  = conf
                right_by_tok[t] = right

            # 히스토그램/리라이어빌리티 저장
            plot_hist_by_token(model, ds, conf_by_tok, right_by_tok, args.outdir, bins=args.bins)
            plot_reliability_by_token(model, ds, conf_by_tok, right_by_tok, args.outdir, n_bins=args.n_bins_ece)

            # ECE/ACC 수치 요약
            for t in args.tokens:
                conf = conf_by_tok[t]; right = right_by_tok[t]
                if conf.size == 0: 
                    continue
                # ECE 계산은 probs가 필요하지만 conf만 있으면 top-1 conf에 대한 bin-acc로 동일하게 계산 가능
                # 여기선 conf/right로 직접 ECE 계산
                # (probs 전체가 있어야 더 정교하지만 top-1 ECE는 conf로 충분)
                # 재구현
                bins = np.linspace(0.,1.,args.n_bins_ece+1)
                N = len(conf); ece_val = 0.0
                acc = right.mean()
                for b in range(args.n_bins_ece):
                    lo,hi=bins[b],bins[b+1]
                    mask = (conf >= lo) & (conf <= hi) if b==0 else ((conf>lo)&(conf<=hi))
                    if not np.any(mask): 
                        continue
                    acc_b = right[mask].mean()
                    conf_b = conf[mask].mean()
                    ece_val += (mask.sum()/N) * abs(acc_b - conf_b)
                tsv_lines.append(f"{model}\t{ds}\t{t}\t{N}\t{acc:.4f}\t{ece_val:.4f}\n")

    # TSV 저장
    tsv_path = Path(args.outdir)/"ece_summary.tsv"
    with open(tsv_path, "w", encoding="utf-8") as w:
        w.writelines(tsv_lines)
    print(f"[DONE] Saved: {tsv_path}")

if __name__ == "__main__":
    main()
