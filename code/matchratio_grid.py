#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer-match matrix (rows=models, cols=datasets).
- 기준: "정답(qid) 집합" 간의 겹침을 계산
- 입력: <root>/<dataset>/<model>/<T*/P0.jsonl> (각 jsonl은 qid, gold, pred, probs 등 포함)
- 토큰쌍: --pair T0 T1  (또는 --pair ALL 로 T0,T1,T2 모든 쌍 평균)
- 지표(--metric): jaccard(기본), overlap, inter

예시:
  python code/matchratio_grid.py \
    --root routes_out/token_only/pride_on \
    --datasets arc csqa \
    --pair ALL \
    --metric jaccard \
    --out_png viz_out/answer_match_matrix_on_ALL.png \
    --out_csv viz_out/answer_match_matrix_on_ALL.csv
"""
import os, json, argparse, numpy as np
import matplotlib.pyplot as plt

def read_jsonl(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def parse_pred_gold(rec, tokens4=("A","B","C","D"), tokens5=("A","B","C","D","E")):
    # qid
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

def load_correct_set(dir_model_token):
    """<dir_model>/<T*/P0.jsonl> -> {qid | 정답 맞춘 qid}"""
    f = os.path.join(dir_model_token, "P0.jsonl")
    if not os.path.exists(f):
        return set()
    ok=set()
    for rec in read_jsonl(f):
        qid, gold, pred = parse_pred_gold(rec)
        if qid is None or gold is None or pred is None:
            continue
        if gold == pred:
            ok.add(str(qid))
    return ok

def jaccard(A,B):
    if not A and not B: return np.nan
    return len(A & B)/len(A | B) if (A|B) else np.nan

def overlap_coeff(A,B):
    if not A or not B: return np.nan
    return len(A & B)/min(len(A),len(B))

def inter_ratio(A,B):
    # |A∩B| / |A|, A 기준
    if not A: return np.nan
    return len(A & B)/len(A)

def pairwise_pairs(tokens):
    out=[]
    for i in range(len(tokens)):
        for j in range(i+1,len(tokens)):
            out.append((tokens[i],tokens[j]))
    return out

def simple_model_name(raw):
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
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # metric fn
    metric_fn = {"jaccard": jaccard, "overlap": overlap_coeff, "inter": inter_ratio}[args.metric]

    # 모델 발견 (모든 ds의 합집합)
    model_set=set()
    per_ds_models={}
    for ds in args.datasets:
        dsd=os.path.join(args.root, ds)
        if not os.path.isdir(dsd): continue
        ms=[m for m in os.listdir(dsd) if os.path.isdir(os.path.join(dsd,m))]
        per_ds_models[ds]=sorted(ms); model_set.update(ms)
    models=sorted(model_set)
    if not models:
        raise SystemExit(f"No models under {args.root}/<dataset>/*")

    # 토큰쌍
    if len(args.pair)==1 and args.pair[0].upper()=="ALL":
        pairs = pairwise_pairs(args.tokens)
        pair_name = "mean(T-pairs)"
    else:
        if len(args.pair)!=2:
            raise SystemExit("--pair must be two tokens or 'ALL'")
        pairs = [tuple(args.pair)]
        pair_name = f"{args.pair[0]} vs {args.pair[1]}"

    # 매트릭스 채우기
    M = np.full((len(models), len(args.datasets)), np.nan, float)

    for j, ds in enumerate(args.datasets):
        dsd=os.path.join(args.root, ds)
        if not os.path.isdir(dsd): continue
        mods=[m for m in os.listdir(dsd) if os.path.isdir(os.path.join(dsd,m))]
        if len(mods)<args.min_models: continue

        for i, model in enumerate(models):
            dir_model=os.path.join(dsd, model)
            if not os.path.isdir(dir_model): continue

            # 필요 토큰의 정답집합 캐시
            correct = {}
            for t in set(sum(([a,b] for (a,b) in pairs), [])):
                correct[t] = load_correct_set(os.path.join(dir_model, t))

            vals=[]
            for a,b in pairs:
                A, B = correct.get(a,set()), correct.get(b,set())
                vals.append(metric_fn(A,B))
            if vals:
                M[i,j]=float(np.nanmean(vals))

    # 정렬: 평균 큰 모델부터
    row_means=np.nanmean(M, axis=1)
    order=np.argsort(-row_means)
    M=M[order,:]; models_sorted=[models[k] for k in order]

    # CSV 저장
    with open(args.out_csv, "w", encoding="utf-8") as w:
        w.write("Model," + ",".join(args.datasets) + "\n")
        for i, m in enumerate(models_sorted):
            row=[simple_model_name(m)] + [("" if np.isnan(M[i,j]) else f"{M[i,j]:.4f}") for j in range(M.shape[1])]
            w.write(",".join(row) + "\n")

    # 히트맵
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
    print(f"[DONE] CSV: {args.out_csv}")
    print(f"[INFO] Root: {args.root} | Pair: {pair_name} | Metric: {args.metric}")
