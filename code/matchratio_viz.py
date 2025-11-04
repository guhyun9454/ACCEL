#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matchratio_viz.py  (token-first directory layout 지원)

디렉토리 예)
ROOT/
  T0/
    arc/
      Qwen2.5-1.5B-Instruct/
        arc/arc.jsonl   (또는 arc.jsonl)
    csqa/
      Qwen2.5-1.5B-Instruct/
        csqa/csqa.jsonl (또는 csqa.jsonl)
  T1/...
  T2/...

기능
- 토큰별 변형(T0/T1/T2[+P?])을 읽어 공통 qid 교집합으로 정렬
- 변형별 정확도, 변형 간 Matching Ratio(MR: pred 일치율) 행렬
- (있다면) 토큰 내 시프트간 평균 MR, 같은 시프트에서 토큰간 평균 MR
- (있다면) probs로 단순 앙상블(ECE 포함)

사용 예
python matchratio_viz.py \
  --root /nas2/.../results \
  --datasets arc csqa \
  --tokens T0 T1 T2 \
  --outdir viz_out
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Utilities
# ----------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def imsave_heatmap(matrix, labels, title, out_png):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, interpolation='nearest', vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def normalize_letter(x):
    """문자 선택지 표기 정규화: 'a'->'A', '(b'->'B', '1)->'1'"""
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    if not x:
        return None
    ch = x[0]
    if ch in "([{":
        ch = x[1] if len(x) > 1 else ch
    return ch.upper() if ch.isalpha() else ch

def choose_letters_for(token_name, C):
    """
    토큰명과 클래스 수(C)에 맞춰 표준 레터 시퀀스 반환.
    - T2는 숫자(4/5) 사용 가정
    - 그 외는 대문자(ABCD/ABCDE) 사용
    """
    if C == 4:
        if token_name.upper().startswith("T2"):
            return list("1234")
        else:
            return list("ABCD")
    elif C == 5:
        if token_name.upper().startswith("T2"):
            return list("12345")
        else:
            return list("ABCDE")
    else:
        # 알 수 없는 C → ABC..., 또는 123... 생성
        if token_name.upper().startswith("T2"):
            return [str(i) for i in range(1, C+1)]
        else:
            base = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if C <= len(base):
                return list(base[:C])
            return [f"C{i}" for i in range(C)]

def parse_record_generic(rec, token_letters):
    """
    jsonl 한 줄을 표준 스키마로 변환:
    반환: (qid:str, pred:int, gold:int, probs:np.ndarray or None)
    - 지원 입력 키:
      - 최상위 또는 data{} 내부: idx/qid/id, sampled/ideal/probs/pred/gold
    - pred/gold은 token_letters에서의 index로 변환
    """
    if "data" in rec and isinstance(rec["data"], dict):
        rec = rec["data"]

    # qid 후보
    qid = None
    for k in ("qid", "id", "uid", "question_id", "idx"):
        if k in rec:
            qid = str(rec[k])
            break
    if qid is None:
        return None  # 스킵

    # probs
    probs = rec.get("probs", None)
    if probs is not None:
        try:
            P = np.array(probs, dtype=float)
            if P.ndim == 2:        # (C,)
                pass
            elif P.ndim == 3:      # (?, C) → 평균
                P = P.mean(axis=0)
            elif P.ndim == 1:
                P = P
            else:
                P = None
        except Exception:
            P = None
    else:
        P = None

    # pred/gold (문자표기를 token_letters 인덱스로 변환)
    # 우선 'sampled' / 'ideal' 문자 우선, 없으면 pred/gold 인덱스 시도
    pred_idx = None
    gold_idx = None

    # 문자 → 인덱스
    def letter_to_idx(letter):
        letter = normalize_letter(letter)
        if letter is None:
            return None
        # 대문자 기준 비교
        up = letter.upper()
        # token_letters가 숫자면 그대로 비교
        tl = [t.upper() if t.isalpha() else t for t in token_letters]
        try:
            return tl.index(up)
        except ValueError:
            return None

    # 문자 키들
    pred_letter = None
    for k in ("pred_letter","prediction_letter","sampled","answer","model_answer","choice","selected","output","final_answer"):
        if k in rec:
            pred_letter = rec[k]
            break

    gold_letter = None
    for k in ("gold_letter","label_letter","ideal","answer","gold","label","solution","target","correct_letter"):
        if k in rec:
            gold_letter = rec[k]
            break

    if pred_letter is not None:
        pred_idx = letter_to_idx(pred_letter)
    if gold_letter is not None:
        gold_idx = letter_to_idx(gold_letter)

    # 인덱스 키들 (문자 변환 실패시 대체)
    if pred_idx is None:
        for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx","output_idx","model_pred_idx","pred","prediction"):
            if k in rec:
                try:
                    vi = int(rec[k])
                    pred_idx = vi if vi >= 0 else None
                    break
                except Exception:
                    pass
    if gold_idx is None:
        for k in ("gold_idx","label_idx","answer_idx","correct_idx","solution_idx","target_idx","gold","label","answer"):
            if k in rec:
                try:
                    vi = int(rec[k])
                    gold_idx = vi if vi >= 0 else None
                    break
                except Exception:
                    pass

    # 마지막 보정: probs가 있으면 argmax로 pred 보정
    if pred_idx is None and P is not None and P.ndim == 2:
        pred_idx = int(np.argmax(P[0])) if P.shape[0] == 1 else int(np.argmax(P))
    if pred_idx is None and P is not None and P.ndim == 1:
        pred_idx = int(np.argmax(P))

    # gold가 여전히 None이면 스킵 (정렬이 안 됨)
    if gold_idx is None:
        return None

    # probs 정형화: (C,) 로 맞춤
    if P is not None:
        if P.ndim == 2 and P.shape[0] == 1:
            P = P[0]
        if P.ndim != 1:
            P = None

    return (qid, pred_idx, gold_idx, P)

# ----------------------
# Loading
# ----------------------

def load_token_first_layout(root, datasets, tokens, model_filter=None):
    """
    token-first 레이아웃에서 결과를 적재.
    반환:
      store[(model_tag, dataset)][variation_id] -> list[rec_std]
      variation_id = 'T0', 'T1', ... (또는 'T0P0' 형식도 허용)
    """
    store = defaultdict(lambda: defaultdict(list))

    # 토큰 디렉토리 순회
    for token in tokens:
        tok_dir = os.path.join(root, token)
        if not os.path.isdir(tok_dir):
            continue

        for ds in datasets:
            ds_dir = os.path.join(tok_dir, ds)
            if not os.path.isdir(ds_dir):
                continue

            # 모델 디렉토리들
            for mdir in sorted(os.listdir(ds_dir)):
                model_dir = os.path.join(ds_dir, mdir)
                if not os.path.isdir(model_dir):
                    continue
                if model_filter is not None and mdir != model_filter.replace("/", "_"):
                    continue

                # jsonl 경로 후보 (둘 다 지원)
                cand1 = os.path.join(model_dir, ds, f"{ds}.jsonl")
                cand2 = os.path.join(model_dir, f"{ds}.jsonl")
                jf = cand1 if os.path.isfile(cand1) else cand2 if os.path.isfile(cand2) else None
                if jf is None:
                    continue

                # probs 길이에 맞는 토큰 문자셋 만들기 위해 먼저 한 줄 미리 훑어봄
                # (없으면 기본 4로 가정했다가 실제 레코드마다 재조정)
                pre_letters = None

                with open(jf, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if "data" in rec and isinstance(rec["data"], dict):
                                rec = rec["data"]
                            probs = rec.get("probs", None)
                            if probs is not None:
                                P = np.array(probs, dtype=float)
                                C = P.shape[-1] if P.ndim >= 1 else None
                                if C is not None and C >= 2:
                                    pre_letters = choose_letters_for(token, C)
                                    break
                        except Exception:
                            continue
                if pre_letters is None:
                    pre_letters = choose_letters_for(token, 4)

                # variation id: 토큰( + 가능하면 P시프트 )
                # 여기서는 파일명으로 P를 구분하지 않으므로 토큰만 사용
                variation_id = token

                # 본격 적재
                with open(jf, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec_raw = json.loads(line)
                        except Exception:
                            continue
                        # 각 레코드마다 probs 길이가 다르면 letters를 재산정
                        _rec = rec_raw.get("data", rec_raw)
                        probs = _rec.get("probs", None)
                        letters = pre_letters
                        if probs is not None:
                            try:
                                P = np.array(probs, dtype=float)
                                C = P.shape[-1] if P.ndim >= 1 else None
                                if C is not None and C >= 2:
                                    letters = choose_letters_for(token, C)
                            except Exception:
                                pass

                        parsed = parse_record_generic(rec_raw, letters)
                        if parsed is None:
                            continue
                        qid, pred, gold, P = parsed
                        store[(mdir, ds)][variation_id].append({
                            "qid": qid,
                            "pred": pred,
                            "gold": gold,
                            "probs": (P.tolist() if isinstance(P, np.ndarray) else None),
                        })

    return store

# ----------------------
# Metrics & Ensembles
# ----------------------

def compute_metrics(group):
    """
    group: dict[variation_id] -> list[rec_std{qid,pred,gold,probs}]
    반환:
      acc: dict[vid] -> float
      mr: (V,V) ndarray
      order: list[vid]
      df: per-variant summary DataFrame
      aligned_probs: dict[vid] -> (N,C) or {}
      golds: (N,) or None
    """
    order = sorted(group.keys())
    aligned = {}
    all_qids = set()
    for vid in order:
        m = {}
        for r in group[vid]:
            m[str(r["qid"])] = (int(r["pred"]) if r["pred"] is not None else None,
                                int(r["gold"]) if r["gold"] is not None else None,
                                r.get("probs", None))
        aligned[vid] = m
        all_qids |= set(m.keys())

    common_qids = [q for q in all_qids if all(q in aligned[vid] for vid in order)]
    if not common_qids:
        return {}, np.zeros((len(order), len(order))), order, pd.DataFrame(), {}, None

    acc = {}
    preds = {}
    probs = {}
    golds = None
    for vid in order:
        c = 0
        p_list = []
        P_list = []
        for q in common_qids:
            pred, gold, pr = aligned[vid][q]
            if pred is None or gold is None:
                continue
            p_list.append(pred)
            c += 1 if pred == gold else 0
            if pr is not None:
                P_list.append(np.array(pr, dtype=float))
        n = len(p_list)
        acc[vid] = (c / n) if n > 0 else 0.0
        preds[vid] = np.array(p_list, dtype=int)
        if P_list:
            probs[vid] = np.stack(P_list, axis=0)
        if golds is None:
            golds = np.array([aligned[order[0]][q][1] for q in common_qids], dtype=int)

    V = len(order)
    mr = np.zeros((V, V), dtype=float)
    for i, vi in enumerate(order):
        for j, vj in enumerate(order):
            common_len = min(len(preds[vi]), len(preds[vj]))
            if common_len == 0:
                mr[i, j] = np.nan
            else:
                mr[i, j] = (preds[vi][:common_len] == preds[vj][:common_len]).mean()

    rows = [{"variation_id": vid, "accuracy": acc[vid], "n": len(preds[vid])} for vid in order]
    df = pd.DataFrame(rows).sort_values("variation_id").reset_index(drop=True)
    return acc, mr, order, df, probs, golds

def ensemble_probs(variant_probs):
    mats = [variant_probs[vid] for vid in sorted(variant_probs.keys())]
    P = np.stack(mats, axis=0).mean(axis=0)
    P = np.clip(P, 1e-12, None)
    P /= P.sum(axis=1, keepdims=True)
    return P

def geometric_ensemble_probs(variant_probs):
    mats = [np.clip(variant_probs[vid], 1e-12, None) for vid in sorted(variant_probs.keys())]
    logsum = np.sum(np.log(mats), axis=0) / len(mats)
    P = np.exp(logsum)
    P /= P.sum(axis=1, keepdims=True)
    return P

def ece(probs, correct, n_bins=10):
    top_conf = probs.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece_val = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (top_conf > lo) & (top_conf <= hi) if b>0 else (top_conf >= lo) & (top_conf <= hi)
        if mask.sum()==0: 
            continue
        acc = correct[mask].mean()
        conf = top_conf[mask].mean()
        ece_val += (mask.mean()) * abs(acc - conf)
    return float(ece_val)

# ----------------------
# Main
# ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="토큰-우선 결과 루트")
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--model", default=None, help="특정 모델 폴더명(슬래시→언더스코어 치환된 형태)")
    ap.add_argument("--outdir", default="viz_out")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    store = load_token_first_layout(
        root=args.root,
        datasets=args.datasets,
        tokens=args.tokens,
        model_filter=args.model
    )
    if not store:
        print("[WARN] 결과를 찾지 못했습니다. root/Tx/<dataset>/<model>/*.jsonl 확인하세요.")
        return

    # (model, dataset) 그룹 단위로 분석
    for (model_tag, eval_name), group in store.items():
        print(f"\n== Model: {model_tag} | Eval: {eval_name} ==")
        acc, mr, order, df, probs, golds = compute_metrics(group)
        if not acc:
            print("공통 qid가 없거나 비어 있습니다.")
            continue

        # 표 저장
        tsv_path = os.path.join(args.outdir, f"{model_tag}_{eval_name}_variant_acc.tsv")
        df.to_csv(tsv_path, sep="\t", index=False)
        print("Saved:", tsv_path)

        # MR 히트맵
        png_mr = os.path.join(args.outdir, f"{model_tag}_{eval_name}_mr.png")
        imsave_heatmap(mr, order, f"Matching Ratio — {model_tag} / {eval_name}", png_mr)
        print("Saved:", png_mr)

        # 토큰 내 시프트 / 시프트 내 토큰 요약
        # 변형명이 'T0P0' 같은 형태면 P시프트 추출, 아니면 토큰만 존재
        token_sets = sorted({vid.split('P')[0] for vid in order})
        shifts = sorted({int(vid.split('P')[1]) for vid in order if 'P' in vid and vid.split('P')[1].isdigit()})

        # 같은 토큰 내 서로 다른 시프트 사이 평균 MR
        if any(order.count(t) > 1 for t in token_sets):
            rows = []
            for t in token_sets:
                vids = [v for v in order if v.startswith(t)]
                idx = [order.index(v) for v in vids]
                sub = mr[np.ix_(idx, idx)]
                if sub.size > len(idx):  # 2개 이상일 때
                    mrr = (np.nansum(sub) - np.nansum(np.diag(sub))) / (sub.size - len(idx))
                    rows.append({"token_set": t, "avg_MR_within_token": float(mrr)})
            if rows:
                df_cyc = pd.DataFrame(rows)
                cyc_tsv = os.path.join(args.outdir, f"{model_tag}_{eval_name}_cyclic_within_token.tsv")
                df_cyc.to_csv(cyc_tsv, sep="\t", index=False)
                print("Saved:", cyc_tsv)

        # 같은 시프트 내 서로 다른 토큰 사이 평균 MR
        if shifts:
            rows = []
            for p in shifts:
                vids = [v for v in order if v.endswith(f"P{p}")]
                idx = [order.index(v) for v in vids]
                sub = mr[np.ix_(idx, idx)]
                if sub.size > len(idx) and len(idx) > 1:
                    mrr = (np.nansum(sub) - np.nansum(np.diag(sub))) / (sub.size - len(idx))
                    rows.append({"permute_shift": p, "avg_MR_across_tokens": float(mrr)})
            if rows:
                df_tok = pd.DataFrame(rows)
                tok_tsv = os.path.join(args.outdir, f"{model_tag}_{eval_name}_token_across_shifts.tsv")
                df_tok.to_csv(tok_tsv, sep="\t", index=False)
                print("Saved:", tok_tsv)

        # 앙상블 (probs가 모든 변형에 존재할 때)
        if probs and golds is not None:
            # 변형별 probs 길이 다르면 공통 길이로 맞춤
            C = min(p.shape[1] for p in probs.values())
            trimmed = {k: v[:, :C] for k, v in probs.items()}
            P_avg = ensemble_probs(trimmed)
            P_geo = geometric_ensemble_probs(trimmed)

            ens_acc_avg = float((P_avg.argmax(axis=1) == golds[:P_avg.shape[0]]).mean())
            ens_acc_geo = float((P_geo.argmax(axis=1) == golds[:P_geo.shape[0]]).mean())

            def _ece(prob):
                top_conf = prob.max(axis=1)
                corr = (prob.argmax(axis=1) == golds[:prob.shape[0]])
                return ece(prob, corr)

            ece_avg = _ece(P_avg)
            ece_geo = _ece(P_geo)

            with open(os.path.join(args.outdir, f"{model_tag}_{eval_name}_ensemble.txt"), "w") as f:
                f.write(f"Arithmetic-mean ensemble accuracy: {ens_acc_avg:.4f}\n")
                f.write(f"Geometric-mean ensemble accuracy : {ens_acc_geo:.4f}\n")
                f.write(f"ECE (arith): {ece_avg:.4f}\n")
                f.write(f"ECE (geom) : {ece_geo:.4f}\n")
            print("Saved:", os.path.join(args.outdir, f"{model_tag}_{eval_name}_ensemble.txt"))
        else:
            print("[Note] probs가 없어 앙상블/ECE 생략.")

if __name__ == "__main__":
    main()
