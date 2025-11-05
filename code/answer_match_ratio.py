#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
answer_match_rate.py — 모델×데이터셋 Answer Match Rate(토큰 간 일치율) 히트맵

입력 구조(둘 다 지원):
  <root>/<token>/<dataset>/<model>/<dataset>.jsonl
  <root>/<token>/<dataset>/<model>/<dataset>/<dataset>.jsonl

JSONL 예시:
  {"type":"result","data":{"qid":"...", "probs":[...], "sampled":"C", "ideal":"B"}}
  { "qid":"...", "pred_idx":2, "gold_idx":1, ... }  (평면형도 OK)

정의:
  - 쌍별 매칭율(pairwise match rate): 두 토큰이 공통으로 예측한 qid 집합에서
    pred_idx가 같은 비율
  - Answer Match Rate(AMR): 모든 토큰쌍 매칭율의 집계값
      * --agg weighted : 각 쌍의 표본수(공통 qid 수)로 가중 평균 (기본)
      * --agg mean     : 단순 평균

필터(--mode):
  - all     : 모든 공통 qid
  - correct : 두 토큰이 모두 정답인 qid만
  - wrong   : 두 토큰이 모두 오답인 qid만

출력:
  - Heatmap PNG  (행=모델, 열=데이터셋)
  - TSV (모델,데이터셋,amr, pairs_used, qids_used)

사용 예:
  python answer_match_rate.py --root results --datasets arc csqa --tokens T0 T1 T2 \
    --outdir viz_out/answer_match --mode all --agg weighted
"""

import os, json, argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

# Headless 안전
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

# ───────────────────────── 유틸 ─────────────────────────
TOKEN_LABEL = {"T0": "ABCD", "T1": "abcd", "T2": "1234"}

def safe_tag(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").strip()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def flatten_record(rec: dict) -> dict:
    if isinstance(rec, dict) and "data" in rec and isinstance(rec["data"], dict):
        d = dict(rec["data"])
        if "qid" not in d and "idx" in d:
            d["qid"] = str(d["idx"])
        return d
    return rec

def normalize_letter(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    s = str(x).strip()
    if not s: return None
    ch = s[0]
    if ch in "([{": ch = s[1] if len(s) > 1 else ch
    if ch.isalpha(): ch = ch.upper()
    return ch

def token_letters(token: str) -> List[str]:
    if token in ("T0","T1"): return list("ABCD")
    if token == "T2":        return list("1234")
    return list("ABCD")

def letter_to_idx(token: str, letter: str) -> Optional[int]:
    if token == "T2":
        ch = str(letter).strip()[:1] if letter else None
    else:
        ch = normalize_letter(letter)
    if not ch: return None
    L = token_letters(token)
    return L.index(ch) if ch in L else None

def fold_probs(arr, n_opts: int) -> np.ndarray:
    p = np.array(arr, dtype=float)
    if p.ndim == 2: p = p.mean(axis=0)
    if p.ndim == 1 and p.size == 2*n_opts:
        p = p.reshape(2, n_opts).sum(axis=0)
    p = np.clip(p, 1e-12, None); p /= p.sum()
    return p

def locate_jsonl(root: str, token: str, dataset: str, model: str) -> Optional[Path]:
    base = Path(root) / token / dataset / model
    c1 = base / f"{dataset}.jsonl"
    c2 = base / dataset / f"{dataset}.jsonl"
    if c1.is_file(): return c1
    if c2.is_file(): return c2
    return None

def collect_models(root: str, tokens: List[str], datasets: List[str]) -> List[str]:
    mset = set()
    for t in tokens:
        for ds in datasets:
            base = Path(root) / t / ds
            if base.is_dir():
                for m in base.iterdir():
                    if m.is_dir(): mset.add(m.name)
    return sorted(mset)

def filter_models(all_models: List[str], wants: Optional[List[str]]) -> List[str]:
    if not wants: return all_models
    wants_l = [w.lower() for w in wants]
    wants_s = [safe_tag(w).lower() for w in wants]
    out = []
    for m in all_models:
        ml = m.lower()
        if any(w in ml for w in wants_l) or any(w in ml for w in wants_s):
            out.append(m)
    return sorted(set(out))

# ─────────────────────── 로딩 & 파싱 ───────────────────────
def load_records(fp: Optional[Path]) -> List[dict]:
    if not fp or not fp.is_file(): return []
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                rows.append(flatten_record(json.loads(s)))
            except Exception:
                continue
    return rows

def parse_one_rec(rec: dict, token: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[bool]]:
    """
    반환: (qid, pred_idx, gold_idx, correct_flag)
    """
    qid = rec.get("qid") or rec.get("id") or rec.get("uid") or rec.get("question_id")
    if qid is None and "idx" in rec: qid = str(rec["idx"])
    if qid is None: return None, None, None, None

    n_opts = len(token_letters(token))

    # gold
    gold_idx = None
    g_letter = rec.get("ideal") or rec.get("gold_letter") or rec.get("label_letter") or rec.get("answer") \
               or rec.get("gold") or rec.get("label") or rec.get("solution") or rec.get("target")
    gi = letter_to_idx(token, g_letter) if g_letter is not None else None
    if gi is not None:
        gold_idx = gi
    else:
        for k in ("gold_idx","label_idx","correct_idx","target_idx","solution_idx","answer_idx"):
            if k in rec:
                try:
                    vi = int(rec[k]); 
                    if vi >= 0: gold_idx = vi; break
                except Exception:
                    pass

    # pred
    pred_idx = None
    p_letter = rec.get("sampled") or rec.get("pred_letter") or rec.get("prediction_letter") or rec.get("model_answer") \
               or rec.get("choice") or rec.get("selected") or rec.get("final_answer") \
               or rec.get("pred") or rec.get("prediction") or rec.get("answer")
    pi = letter_to_idx(token, p_letter) if p_letter is not None else None
    if pi is not None:
        pred_idx = pi
    else:
        for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx","output_idx","model_pred_idx"):
            if k in rec:
                try:
                    vi = int(rec[k]); 
                    if vi >= 0: pred_idx = vi; break
                except Exception:
                    pass
        if pred_idx is None and "probs" in rec:
            p = fold_probs(rec["probs"], n_opts)
            pred_idx = int(np.argmax(p))

    # correct
    cf = rec.get("correct", None)
    if isinstance(cf, str): cf = cf.lower() in ("true","1","yes")
    if isinstance(cf, bool):
        correct_flag = cf
    else:
        correct_flag = (pred_idx is not None and gold_idx is not None and pred_idx == gold_idx)

    return str(qid), pred_idx, gold_idx, correct_flag

def load_token_dicts(root: str, token: str, dataset: str, model: str):
    """
    preds_by_qid: {qid -> pred_idx}
    correct_by_qid: {qid -> 0/1}
    """
    fp = locate_jsonl(root, token, dataset, model)
    rows = load_records(fp)
    preds, rights = {}, {}
    for r in rows:
        qid, pi, gi, cf = parse_one_rec(r, token)
        if qid is None or pi is None: 
            continue
        preds[qid] = int(pi)
        rights[qid] = 1 if (cf is True) else 0
    return preds, rights

# ───────────────────── 매칭율 계산 ─────────────────────
def pair_order(tokens: List[str]) -> List[Tuple[str,str]]:
    return [(tokens[i], tokens[j]) for i in range(len(tokens)) for j in range(i+1, len(tokens))]

def pair_match_rate(pred_i: Dict[str,int], pred_j: Dict[str,int],
                    corr_i: Dict[str,int], corr_j: Dict[str,int], mode: str):
    keys = set(pred_i.keys()) & set(pred_j.keys())
    if mode == "correct":
        keys = {q for q in keys if corr_i.get(q,0)==1 and corr_j.get(q,0)==1}
    elif mode == "wrong":
        keys = {q for q in keys if corr_i.get(q,0)==0 and corr_j.get(q,0)==0}
    if not keys: 
        return np.nan, 0
    same = sum(1 for q in keys if pred_i[q] == pred_j[q])
    return same/float(len(keys)), len(keys)

def amr_for_model_dataset(root: str, tokens: List[str], dataset: str, model: str,
                          mode: str, agg: str) -> Tuple[float, int, int]:
    """
    반환: (AMR, 총쌍표본수, 사용된쌍수)
    """
    preds_by_tok, corr_by_tok = {}, {}
    for t in tokens:
        p, c = load_token_dicts(root, t, dataset, model)
        preds_by_tok[t] = p
        corr_by_tok[t]  = c

    pairs = pair_order(tokens)
    vals, counts = [], []
    for a,b in pairs:
        mr, n = pair_match_rate(preds_by_tok[a], preds_by_tok[b],
                                corr_by_tok[a],  corr_by_tok[b], mode)
        if n > 0 and np.isfinite(mr):
            vals.append(mr); counts.append(n)
    if not vals:
        return float("nan"), 0, 0

    if agg == "mean":
        amr = float(np.mean(vals))
        total_q = int(sum(counts))
    else:  # weighted
        amr = float(np.average(vals, weights=counts))
        total_q = int(sum(counts))

    return amr, total_q, len(vals)

# ───────────────────── 그림 ─────────────────────
def draw_heatmap(models: List[str], datasets: List[str], M: np.ndarray,
                 out_png: Path, title="Answer Match Rate", vmin=0.0, vmax=1.0, dpi=220):
    fig, ax = plt.subplots(figsize=(max(6, 1.8*len(datasets)), max(3.2, 0.9*len(models))))
    im = ax.imshow(M, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto", interpolation="none")
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_yticks(range(len(models)));   ax.set_yticklabels(models)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # 셀 주석
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i,j]):
                ax.text(j, i, f"{100*M[i,j]:.1f}%", ha="center", va="center", color="black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[SAVE] {out_png}")

# ───────────────────── 메인 ─────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--tokens",   nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--models",   nargs="*", default=None,
                    help="부분문자열 매칭 필터. 비우면 자동 탐색.")
    ap.add_argument("--outdir",   default="viz_out/answer_match")
    ap.add_argument("--mode",     choices=["all","correct","wrong"], default="all")
    ap.add_argument("--agg",      choices=["weighted","mean"], default="weighted")
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--dpi",  type=int, default=220)
    ap.add_argument("--title", type=str, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); ensure_dir(outdir)

    # 모델 자동수집 + 필터
    auto_models = collect_models(args.root, args.tokens, args.datasets)
    models = filter_models(auto_models, args.models)
    if not models:
        print("[WARN] 모델 디렉토리를 찾지 못했습니다."); return
    models = sorted(models)

    # 매트릭스 계산
    M = np.full((len(models), len(args.datasets)), np.nan, float)
    tsv_lines = ["model\tdataset\tmode\tagg\tamr\tpairs_used\tqids_used\n"]
    for i, m in enumerate(models):
        for j, ds in enumerate(args.datasets):
            amr, qids_used, pairs_used = amr_for_model_dataset(
                args.root, args.tokens, ds, m, args.mode, args.agg
            )
            M[i,j] = amr
            tsv_lines.append(f"{m}\t{ds}\t{args.mode}\t{args.agg}\t{amr:.6f}\t{pairs_used}\t{qids_used}\n")

    # 그림
    title = args.title or "Answer Match Rate"
    out_png = outdir / f"answer_match_{args.mode}_{args.agg}.png"
    draw_heatmap(models, args.datasets, M, out_png, title=title, vmin=args.vmin, vmax=args.vmax, dpi=args.dpi)

    # TSV
    tsv_path = outdir / f"answer_match_{args.mode}_{args.agg}.tsv"
    with tsv_path.open("w", encoding="utf-8") as w:
        w.writelines(tsv_lines)
    print(f"[DONE] Saved: {tsv_path}")

if __name__ == "__main__":
    main()
