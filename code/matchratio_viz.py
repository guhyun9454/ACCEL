#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer Match Rate (AMR) 시각화 (토큰-우선 디렉토리 레이아웃)

ROOT/
  T0/
    arc/<model>/arc/arc.jsonl  (또는 arc/<model>/arc.jsonl)
    csqa/<model>/csqa/csqa.jsonl (또는 csqa/<model>/csqa.jsonl)
  T1/...
  T2/...

기능
- 변형(토큰/시프트)별 예측을 qid로 정렬
- 변형 간 Matching Ratio(MR = P(pred_i == pred_j)) 행렬 계산
- 모델×데이터셋 격자에 "AMR = 모든 변형쌍 MR의 평균"을 채워 히트맵 저장
- 데이터셋별 수평 BAR로 모델 AMR 분포 저장
- (옵션) 변형 정확도/개수 CSV 저장

사용 예:
python matchratio_viz.py --root results --datasets arc csqa --tokens T0 T1 T2 --outdir viz_out
"""

import argparse
import json
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt


# ---------------------- utils ----------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def normalize_letter(x):
    if x is None: return None
    if not isinstance(x, str): x = str(x)
    x = x.strip()
    if not x: return None
    ch = x[0]
    if ch in "([{": ch = x[1] if len(x)>1 else ch
    return ch.upper() if ch.isalpha() else ch

def choose_letters_for(token_name, C):
    if C == 4:
        return list("1234") if token_name.upper().startswith("T2") else list("ABCD")
    if C == 5:
        return list("12345") if token_name.upper().startswith("T2") else list("ABCDE")
    # fallback
    return list("ABCD") if C is None else (list("ABCDE")[:C] if C<=5 else [f"C{i}" for i in range(C)])

def probs_to_idx_and_letter(probs, letters):
    if probs is None or not letters: return None, None
    p = np.array(probs, dtype=float)
    if p.ndim == 2: p = p.mean(axis=0)
    if p.ndim != 1 or p.size != len(letters): return None, None
    idx = int(np.argmax(p))
    return idx, letters[idx]

def simple_model_name(raw):
    return (raw.replace("meta-llama_", "Meta ")
              .replace("google_", "Google ")
              .replace("Qwen_", "Qwen ")
              .replace("naver-hyperclovax_", "Naver ")
              .replace("kakaocorp_", "Kakao ")
              .replace("LGAI-EXAONE_", "LG AI ")
              .replace("K-intelligence_", "KT ")
              .replace("_", " ")).strip()

def flatten_record(rec):
    if isinstance(rec, dict) and "data" in rec and isinstance(rec["data"], dict):
        flat = dict(rec["data"])
        if "qid" not in flat and "idx" in flat:
            flat["qid"] = str(flat["idx"])
        return flat
    return rec

def parse_record_generic(rec, token_letters):
    r = flatten_record(rec)

    # qid
    qid = r.get("qid") or r.get("id") or r.get("uid") or r.get("question_id") or r.get("idx")
    if qid is None: return None
    qid = str(qid)

    # probs
    P = None
    if "probs" in r:
        try:
            arr = np.array(r["probs"], dtype=float)
            if arr.ndim == 2: arr = arr.mean(axis=0)
            if arr.ndim == 1: P = arr
        except Exception:
            P = None

    # gold
    gold_letter = None
    for k in ("ideal","gold_letter","label_letter","answer","gold","label","solution","target","correct_letter"):
        if k in r:
            gold_letter = normalize_letter(r[k]); break
    gold_idx = None
    if gold_letter and token_letters:
        up = gold_letter.upper()
        lut = [t.upper() if t.isalpha() else t for t in token_letters]
        if up in lut: gold_idx = lut.index(up)
    if gold_idx is None:
        for k in ("gold_idx","label_idx","answer_idx","correct_idx","solution_idx","target_idx","gold","label","answer"):
            if k in r:
                try: 
                    vi = int(r[k]); 
                    if vi >= 0: gold_idx = vi; break
                except Exception:
                    pass
    if gold_idx is None and P is not None:
        gold_idx = None  # gold를 확률로 추정하진 않음

    # pred
    pred_idx = None
    pred_letter = None
    for k in ("sampled","pred_letter","prediction_letter","answer","model_answer","choice","selected","output","final_answer","pred","prediction"):
        if k in r:
            pred_letter = normalize_letter(r[k]); break
    if pred_letter and token_letters:
        up = pred_letter.upper()
        lut = [t.upper() if t.isalpha() else t for t in token_letters]
        if up in lut: pred_idx = lut.index(up)
    if pred_idx is None:
        for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx","output_idx","model_pred_idx","pred","prediction"):
            if k in r:
                try:
                    vi = int(r[k]); 
                    if vi >= 0: pred_idx = vi; break
                except Exception:
                    pass
    if pred_idx is None and P is not None and token_letters:
        pred_idx, _ = probs_to_idx_and_letter(P, token_letters)

    # correct (있으면 보존)
    correct_flag = r.get("correct", None)
    if isinstance(correct_flag, str):
        correct_flag = correct_flag.lower() in ("true","yes","1")

    return {
        "qid": qid,
        "pred": pred_idx,
        "gold": gold_idx,
        "probs": (P.tolist() if isinstance(P, np.ndarray) else None),
        "correct": correct_flag
    }


# ---------------------- loading ----------------------

def load_token_first_layout(root, datasets, tokens, model_filter=None):
    """
    반환:
      store[(model, dataset)][variant_id] -> list[rec_std]
      variant_id: 'T0' (또는 'T0P0' 등, 여기선 토큰만 사용)
    """
    store = defaultdict(lambda: defaultdict(list))

    for token in tokens:
        tok_dir = os.path.join(root, token)
        if not os.path.isdir(tok_dir): continue

        for ds in datasets:
            ds_dir = os.path.join(tok_dir, ds)
            if not os.path.isdir(ds_dir): continue

            for mdir in sorted(os.listdir(ds_dir)):
                model_dir = os.path.join(ds_dir, mdir)
                if not os.path.isdir(model_dir): continue
                if model_filter and mdir != model_filter.replace("/", "_"): continue

                # jsonl 후보
                cand1 = os.path.join(model_dir, ds, f"{ds}.jsonl")
                cand2 = os.path.join(model_dir, f"{ds}.jsonl")
                jf = cand1 if os.path.isfile(cand1) else (cand2 if os.path.isfile(cand2) else None)
                if jf is None: continue

                # 레터셋 미리 감지
                pre_letters = None
                with open(jf, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            rec = flatten_record(rec)
                            if "probs" in rec:
                                arr = np.array(rec["probs"], dtype=float)
                                C = arr.shape[-1] if arr.ndim>=1 else None
                                if C and C>=2:
                                    pre_letters = choose_letters_for(token, C)
                                    break
                        except Exception:
                            continue
                if pre_letters is None: pre_letters = choose_letters_for(token, 4)

                variant_id = token
                with open(jf, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            raw = json.loads(line)
                        except Exception:
                            continue
                        letters = pre_letters
                        _r = flatten_record(raw)
                        if "probs" in _r:
                            try:
                                C = np.array(_r["probs"]).shape[-1]
                                if C and C>=2: letters = choose_letters_for(token, C)
                            except Exception:
                                pass
                        std = parse_record_generic(raw, letters)
                        if std is None: continue
                        store[(mdir, ds)][variant_id].append(std)
    return store


# ---------------------- metrics ----------------------

def pairwise_mr(preds_dict):
    """
    preds_dict: dict[vid] -> np.array of pred indices (aligned length)
    반환: (V,V) matching ratio matrix
    """
    vids = sorted(preds_dict.keys())
    V = len(vids)
    mr = np.full((V, V), np.nan, float)
    for i, vi in enumerate(vids):
        pi = preds_dict[vi]
        for j, vj in enumerate(vids):
            pj = preds_dict[vj]
            n = min(len(pi), len(pj))
            if n == 0: continue
            mr[i, j] = (pi[:n] == pj[:n]).mean()
    return vids, mr

def compute_group_stats(group):
    """
    group: dict[variant_id] -> list[std rec]
    반환:
      amr: 모든 변형쌍 MR 평균(대각 제외)
      acc_per_variant: dict[vid] -> accuracy
      n_q: 공통 qid 개수
    """
    order = sorted(group.keys())
    aligned = {}
    all_qids = set()
    for vid in order:
        m = {r["qid"]: (r["pred"], r["gold"]) for r in group[vid]}
        aligned[vid] = m
        all_qids |= set(m.keys())
    # 공통 qid만 사용
    qids = [q for q in all_qids if all(q in aligned[v] for v in order)]
    if len(qids) < 1 or len(order) < 2:
        return np.nan, {}, 0

    preds = {}
    accs = {}
    for vid in order:
        p, g = [], []
        for q in qids:
            pred, gold = aligned[vid][q]
            if pred is None or gold is None:
                continue
            p.append(int(pred)); g.append(int(gold))
        p = np.array(p, dtype=int)
        g = np.array(g, dtype=int)
        preds[vid] = p
        accs[vid] = float((p == g).mean()) if len(p)>0 else np.nan

    # MR 행렬 & AMR(오프대각 평균)
    vids, mr = pairwise_mr(preds)
    if mr.size == 0: return np.nan, accs, len(qids)
    off = mr[~np.eye(len(vids), dtype=bool)]
    amr = float(np.nanmean(off)) if np.isfinite(off).any() else np.nan
    return amr, accs, len(qids)


# ---------------------- plots ----------------------

def plot_amr_heatmap(matrix, models, datasets, out_png):
    plt.figure(figsize=(8, max(3, 0.45*len(models)+1)))
    im = plt.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    plt.title("Answer Match Rate")
    plt.xticks(range(len(datasets)), [d.upper() for d in datasets])
    plt.yticks(range(len(models)), [simple_model_name(m) for m in models])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{100*v:.1f}%", ha="center", va="center", fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_dataset_bars(values, models, dataset, out_png):
    # values: list of floats (AMR) length == len(models)
    pairs = [(simple_model_name(m), v) for m, v in zip(models, values) if np.isfinite(v)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    plt.figure(figsize=(7, max(3, 0.45*len(labels)+1)))
    y = np.arange(len(labels))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    for yi, vi in zip(y, vals):
        plt.text(vi + 0.005, yi, f"{100*vi:.1f}%", va="center", fontsize=8)
    plt.xlim(0.0, 1.0)
    plt.xlabel("Answer Match Rate")
    plt.title(f"AMR by Model — {dataset.upper()}")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--model", default=None, help="특정 모델만 보려면 지정(슬래시→언더스코어)")
    ap.add_argument("--outdir", default="viz_out")
    ap.add_argument("--write_per_variant_csv", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    store = load_token_first_layout(args.root, args.datasets, args.tokens, args.model)
    if not store:
        print("[WARN] 결과가 없습니다. 경로/토큰/데이터셋을 확인하세요.")
        return

    # 모델/데이터셋 목록
    pairs = sorted(store.keys(), key=lambda x: (x[1], x[0]))
    models = sorted({m for (m, _) in pairs})
    datasets = list(args.datasets)  # 입력 순서 유지

    # AMR 매트릭스
    M = np.full((len(models), len(datasets)), np.nan, float)

    # (옵션) 변형별 정확도 CSV 수집
    rows_acc = []

    for j, ds in enumerate(datasets):
        for i, m in enumerate(models):
            group = store.get((m, ds), {})
            if not group: 
                continue
            amr, accs, n_q = compute_group_stats(group)
            M[i, j] = amr
            if args.write_per_variant_csv and accs:
                for vid, a in accs.items():
                    rows_acc.append({"model": m, "dataset": ds, "variant": vid, "accuracy": a, "n_q": n_q})

    # 히트맵 저장
    heat_png = os.path.join(args.outdir, "answer_match_rate_heatmap.png")
    plot_amr_heatmap(M, models, datasets, heat_png)
    print("Saved:", heat_png)

    # 데이터셋별 바 차트 저장
    for j, ds in enumerate(datasets):
        vals = [M[i, j] for i in range(len(models))]
        bar_png = os.path.join(args.outdir, f"amr_bar_{ds}.png")
        plot_dataset_bars(vals, models, ds, bar_png)
        print("Saved:", bar_png)

    # 테이블 CSV 저장 (모델×데이터셋 AMR)
    tbl = pd.DataFrame(M, index=[simple_model_name(m) for m in models], columns=[d.upper() for d in datasets])
    csv_path = os.path.join(args.outdir, "answer_match_rate_table.csv")
    tbl.to_csv(csv_path, float_format="%.4f")
    print("Saved:", csv_path)

    # (옵션) 변형별 정확도 CSV
    if rows_acc:
        acc_df = pd.DataFrame(rows_acc).sort_values(["dataset","model","variant"])
        acc_csv = os.path.join(args.outdir, "per_variant_accuracy.csv")
        acc_df.to_csv(acc_csv, index=False, float_format="%.4f")
        print("Saved:", acc_csv)


if __name__ == "__main__":
    main()
