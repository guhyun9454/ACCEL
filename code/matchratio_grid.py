#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer-match matrix (rows=models, cols=datasets) with verbose debug.
- 기준: "정답(qid) 집합" 간의 겹침을 계산
- 입력: <root>/<dataset>/<model>/<T*/P0.jsonl>
- 토큰쌍: --pair T0 T1  (또는 --pair ALL 로 T0,T1,T2 모든 쌍 평균)
- 지표(--metric): jaccard(기본), overlap, inter
"""

import os, json, argparse, numpy as np
import matplotlib
matplotlib.use("Agg")  # 백엔드 고정(무음 종료 방지)
import matplotlib.pyplot as plt
from typing import Dict, Set, Tuple, List

def vprint(verbose: bool, *msg):
    if verbose:
        print(*msg, flush=True)

def read_jsonl(path: str, verbose: bool=False) -> List[dict]:
    rows=[]
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln=ln.strip()
                if ln:
                    rows.append(json.loads(ln))
        vprint(verbose, f"[read_jsonl] OK {path}  (n={len(rows)})")
    except FileNotFoundError:
        vprint(verbose, f"[read_jsonl] MISSING {path}")
    except Exception as e:
        vprint(verbose, f"[read_jsonl] ERROR {path} :: {type(e).__name__}: {e}")
    return rows

def parse_pred_gold(rec: dict,
                    tokens4=("A","B","C","D"),
                    tokens5=("A","B","C","D","E"),
                    verbose: bool=False) -> Tuple[str, int, int]:
    qid = rec.get("qid", rec.get("id"))
    if qid is None:
        return None, None, None

    # 선택지 개수 추정
    L = rec.get("num_choices") or rec.get("n_choices")
    if not isinstance(L, int):
        for k in ("choices","options","option_list"):
            if k in rec and isinstance(rec[k], (list,tuple)):
                L = len(rec[k]); break
    if not isinstance(L, int):
        L = 4
    toks = tokens4 if L==4 else (tokens5 if L==5 else [str(i) for i in range(L)])

    def letter_to_idx(letter):
        try:
            return toks.index(str(letter))
        except Exception:
            return None

    # gold
    gold = rec.get("gold_idx") or rec.get("label_idx") or rec.get("label") or rec.get("gold")
    if gold is None:
        gold_letter = rec.get("gold_letter") or rec.get("answer") or rec.get("label_letter")
        gold = letter_to_idx(gold_letter) if gold_letter is not None else None
    # pred
    pred = rec.get("pred_idx") or rec.get("prediction_idx") or rec.get("pred") or rec.get("prediction")
    if pred is None:
        pred_letter = rec.get("pred_letter") or rec.get("prediction_letter")
        pred = letter_to_idx(pred_letter) if pred_letter is not None else None

    try:
        gold = int(gold) if gold is not None else None
        pred = int(pred) if pred is not None else None
    except Exception:
        gold, pred = None, None

    return qid, gold, pred

def load_correct_set(dir_model_token: str, verbose: bool=False) -> Set[str]:
    """<dir_model>/<T*/P0.jsonl> -> {qid | 정답 맞춘 qid}"""
    f = os.path.join(dir_model_token, "P0.jsonl")
    if not os.path.exists(f):
        vprint(verbose, f"[load_correct_set] MISS {f}")
        return set()
    ok=set()
    rows = read_jsonl(f, verbose)
    miss = 0
    for rec in rows:
        qid, gold, pred = parse_pred_gold(rec, verbose=verbose)
        if qid is None or gold is None or pred is None:
            miss += 1
            continue
        if gold == pred:
            ok.add(str(qid))
    vprint(verbose, f"[load_correct_set] {dir_model_token}: |ok|={len(ok)}  (skipped={miss})")
    return ok

def jaccard(A: Set[str], B: Set[str]) -> float:
    if not A and not B: return np.nan
    return len(A & B)/len(A | B) if (A|B) else np.nan

def overlap_coeff(A: Set[str], B: Set[str]) -> float:
    if not A or not B: return np.nan
    return len(A & B)/min(len(A),len(B))

def inter_ratio(A: Set[str], B: Set[str]) -> float:
    # |A∩B| / |A|, A 기준
    if not A: return np.nan
    return len(A & B)/len(A)

def pairwise_pairs(tokens: List[str]) -> List[Tuple[str,str]]:
    out=[]
    for i in range(len(tokens)):
        for j in range(i+1,len(tokens)):
            out.append((tokens[i],tokens[j]))
    return out

def simple_model_name(raw: str) -> str:
    s = raw.replace("meta-llama_", "Meta ").replace("google_", "Google ") \
           .replace("Qwen_", "Qwen ").replace("naver-hyperclovax_", "Naver ") \
           .replace("kakaocorp_", "Kakao ").replace("LGAI-EXAONE_", "LG AI ") \
           .replace("K-intelligence_", "KT ").replace("_", " ")
    return s.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--pair", nargs="+", default=["ALL"], help="예: T0 T1 또는 ALL")
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--metric", choices=["jaccard","overlap","inter"], default="jaccard")
    ap.add_argument("--out_png", default="viz_out/answer_match_matrix.png")
    ap.add_argument("--out_csv", default="viz_out/answer_match_matrix.csv")
    ap.add_argument("--min_models", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    verbose = args.verbose
    vprint(verbose, f"[ARGS] root={args.root} datasets={args.datasets} pair={args.pair} tokens={args.tokens} metric={args.metric}")
    vprint(verbose, f"[CWD]  {os.getcwd()}")

    # 출력 경로 준비
    out_png_dir = os.path.dirname(args.out_png) or "."
    out_csv_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_png_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)

    # metric fn
    metric_map = {"jaccard": jaccard, "overlap": overlap_coeff, "inter": inter_ratio}
    metric_fn = metric_map[args.metric]

    # 모델 탐색
    model_set=set()
    per_ds_models={}
    for ds in args.datasets:
        dsd=os.path.join(args.root, ds)
        vprint(verbose, f"[SCAN] dataset_dir={dsd} exists={os.path.isdir(dsd)}")
        if not os.path.isdir(dsd):
            continue
        ms=[m for m in os.listdir(dsd) if os.path.isdir(os.path.join(dsd,m))]
        per_ds_models[ds]=sorted(ms); model_set.update(ms)
        vprint(verbose, f"[SCAN] {ds}: models_found={len(ms)}  sample={ms[:5]}")
    models=sorted(model_set)
    vprint(verbose, f"[SCAN] union_models={len(models)}")
    if not models:
        print(f"[FATAL] No models under {args.root}/<dataset>/*")
        return

    # 토큰쌍
    if len(args.pair)==1 and args.pair[0].upper()=="ALL":
        pairs = pairwise_pairs(args.tokens)
        pair_name = "mean(T-pairs)"
    else:
        if len(args.pair)!=2:
            print("[FATAL] --pair must be two tokens or 'ALL'")
            return
        pairs = [tuple(args.pair)]
        pair_name = f"{args.pair[0]} vs {args.pair[1]}"
    vprint(verbose, f"[PAIRS] {pairs}  ({pair_name})")

    # 매트릭스
    M = np.full((len(models), len(args.datasets)), np.nan, float)

    for j, ds in enumerate(args.datasets):
        dsd=os.path.join(args.root, ds)
        if not os.path.isdir(dsd):
            vprint(verbose, f"[SKIP] dataset {ds} dir missing: {dsd}")
            continue
        mods=[m for m in os.listdir(dsd) if os.path.isdir(os.path.join(dsd,m))]
        if len(mods)<args.min_models:
            vprint(verbose, f"[SKIP] dataset {ds}: models<{args.min_models}")
            continue

        for i, model in enumerate(models):
            dir_model=os.path.join(dsd, model)
            if not os.path.isdir(dir_model):
                vprint(verbose, f"[MISS] model_dir not found for dataset {ds}: {dir_model}")
                continue

            # 필요 토큰의 정답집합 캐시
            need_tokens = sorted(set([x for ab in pairs for x in ab]))
            correct: Dict[str, Set[str]] = {}
            for t in need_tokens:
                dir_model_token = os.path.join(dir_model, t)
                correct[t] = load_correct_set(dir_model_token, verbose=verbose)

            vals=[]
            for a,b in pairs:
                A, B = correct.get(a,set()), correct.get(b,set())
                val = metric_fn(A,B)
                vals.append(val)
                vprint(verbose, f"[METRIC] ds={ds} model={model} pair=({a},{b}) |A|={len(A)} |B|={len(B)} -> {args.metric}={val}")

            if vals:
                M[i,j]=float(np.nanmean(vals))
                vprint(verbose, f"[FILL] ds={ds} model={model} mean={M[i,j]}")

    # 정렬: 평균 큰 모델부터
    row_means=np.nanmean(M, axis=1)
    order=np.argsort(-row_means)
    M=M[order,:]; models_sorted=[models[k] for k in order]
    vprint(verbose, f"[SHAPE] M={M.shape}  nan_count={np.isnan(M).sum()}")

    # CSV 저장
    try:
        with open(args.out_csv, "w", encoding="utf-8") as w:
            w.write("Model," + ",".join(args.datasets) + "\n")
            for i, m in enumerate(models_sorted):
                row=[simple_model_name(m)] + [("" if np.isnan(M[i,j]) else f"{M[i,j]:.4f}") for j in range(M.shape[1])]
                w.write(",".join(row) + "\n")
        print(f"[DONE] CSV: {args.out_csv}")
    except Exception as e:
        print(f"[ERROR] writing CSV {args.out_csv}: {type(e).__name__}: {e}")

    # 히트맵
    try:
        plt.figure(figsize=(8, max(3, 0.35*len(models_sorted)+1)))
        plt.imshow(M, vmin=0.0, vmax=1.0, aspect="auto")
        plt.title(f"Correct-Overlap ({args.metric}) — {pair_name}", fontsize=14, pad=8)
        plt.xticks(range(len(args.datasets)), [ds.upper() for ds in args.datasets])
        plt.yticks(range(len(models_sorted)), [simple_model_name(m) for m in models_sorted])
        plt.colorbar()
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i,j]):
                    plt.text(j, i, f"{100*M[i,j]:.1f}%", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=180)
        plt.close()
        print(f"[DONE] PNG: {args.out_png}")
    except Exception as e:
        print(f"[ERROR] drawing PNG {args.out_png}: {type(e).__name__}: {e}")

    print(f"[INFO] Root: {args.root} | Pair: {pair_name} | Metric: {args.metric}")
