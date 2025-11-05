#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE & Matching-Ratio visualizer

입력 구조(둘 다 지원):
  <root>/<token>/<dataset>/<model>/<dataset>.jsonl
  <root>/<token>/<dataset>/<model>/<dataset>/<dataset>.jsonl

JSONL 레코드 예:
  {"type":"result","data":{ "qid": "...", "probs":[...], "sampled":"C", "ideal":"B", "correct":true }}
  혹은 평면형 { "qid":..., "probs":[...], "pred_idx":2, "gold_idx":1, ... }

출력:
  - 히스토그램(정답/오답 분리, T0/T1/T2 색 고정 R/B/G)
  - 리라이어빌리티 다이어그램(토큰별)
  - ECE/정확도 요약 TSV
  - 매칭비율 히트맵: ALL / CORRECT-ONLY / WRONG-ONLY (칸에 % 주석)

사용 예:
  python ece_viz.py --root results --datasets arc csqa \
    --models "0s_Llama-3.2-1B-Instruct" "0s_Llama-3.2-3B-Instruct" \
    --outdir viz_out/ece
"""

import os, json, argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

# 헤드리스 안전
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt


# ───────────── 공통 유틸 ─────────────

TOKEN_LABEL = {"T0": "ABCD", "T1": "abcd", "T2": "1234"}
TOKEN_COL   = {"T0": "red",  "T1": "blue", "T2": "green"}

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
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    ch = s[0]
    if ch in "([{":
        ch = s[1] if len(s) > 1 else ch
    if ch.isalpha():
        ch = ch.upper()
    return ch

def token_letters(token: str) -> List[str]:
    """토큰별 표준 선택지 시퀀스"""
    if token == "T0": return list("ABCD")
    if token == "T1": return list("ABCD")  # 비교는 대문자 기준(= abcd → ABCD)
    if token == "T2": return list("1234")
    return list("ABCD")

def letter_to_idx(token: str, letter: str) -> Optional[int]:
    ch = normalize_letter(letter) if token != "T2" else (str(letter).strip()[:1] if letter else None)
    if ch is None: return None
    L = token_letters(token)
    if ch in L:
        return L.index(ch)
    return None

def fold_probs(arr: np.ndarray, n_opts: int) -> np.ndarray:
    """2D이면 평균, 2*n_opts이면 접어서 n_opts로, 그리고 정규화"""
    p = np.array(arr, dtype=float)
    if p.ndim == 2:
        p = p.mean(axis=0)
    if p.ndim == 1 and p.size == 2*n_opts:
        p = p.reshape(2, n_opts).sum(axis=0)
    p = np.clip(p, 1e-12, None)
    p /= p.sum()
    return p

def locate_jsonl(root: str, token: str, dataset: str, model: str) -> Optional[Path]:
    base = Path(root) / token / dataset / model
    c1 = base / f"{dataset}.jsonl"
    c2 = base / dataset / f"{dataset}.jsonl"
    if c1.is_file(): return c1
    if c2.is_file(): return c2
    return None


# ───────────── 로딩 & 파싱 ─────────────

def collect_models(root: str, tokens: List[str], datasets: List[str]) -> List[str]:
    mset = set()
    for t in tokens:
        for ds in datasets:
            base = Path(root) / t / ds
            if base.is_dir():
                for m in base.iterdir():
                    if m.is_dir():
                        mset.add(m.name)
    return sorted(mset)

def filter_models(all_models: List[str], wants: Optional[List[str]]) -> List[str]:
    if not wants:
        return all_models
    wants_s = [safe_tag(w).lower() for w in wants]
    wants_r = [w.lower() for w in wants]
    out = []
    for m in all_models:
        ml = m.lower()
        if any(w in ml for w in wants_s) or any(w in ml for w in wants_r):
            out.append(m)
    return sorted(set(out))

def load_records(fp: Optional[Path]) -> List[dict]:
    if not fp or not fp.is_file():
        return []
    rows: List[dict] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                rows.append(flatten_record(json.loads(s)))
            except Exception:
                continue
    return rows

def parse_one_rec(rec: dict, token: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[float], Optional[bool]]:
    """
    반환: (qid, pred_idx, gold_idx, top1_conf, correct_flag)
    - pred_idx는 sampled/pred_letter → pred_idx → probs(argmax) 순으로 결정 (gold는 절대 사용 X)
    - gold_idx는 ideal/gold_letter → gold_idx 순으로 환산
    - conf는 probs의 top-1
    """
    qid = rec.get("qid") or rec.get("id") or rec.get("uid") or rec.get("question_id")
    if qid is None and "idx" in rec: qid = str(rec["idx"])
    if qid is None: return None, None, None, None, None

    L = token_letters(token)
    n_opts = len(L)

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
                    if vi >= 0: gold_idx = vi
                    break
                except Exception:
                    pass

    # pred
    pred_idx = None
    p_letter = rec.get("sampled") or rec.get("pred_letter") or rec.get("prediction_letter") or rec.get("model_answer") \
               or rec.get("choice") or rec.get("selected") or rec.get("output") or rec.get("final_answer") \
               or rec.get("pred") or rec.get("prediction") or rec.get("answer")
    pi = letter_to_idx(token, p_letter) if p_letter is not None else None
    if pi is not None:
        pred_idx = pi
    else:
        for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx","output_idx","model_pred_idx"):
            if k in rec:
                try:
                    vi = int(rec[k]); 
                    if vi >= 0: pred_idx = vi
                    break
                except Exception:
                    pass
        if pred_idx is None and "probs" in rec:
            p = fold_probs(rec["probs"], n_opts)
            pred_idx = int(np.argmax(p))

    # conf (가능할 때만)
    top1_conf = None
    if "probs" in rec:
        p = fold_probs(rec["probs"], n_opts)
        top1_conf = float(p.max())

    # correct 플래그(있으면 그대로)
    cf = rec.get("correct", None)
    if isinstance(cf, str): cf = cf.lower() in ("true","1","yes")
    if isinstance(cf, bool):
        correct_flag = cf
    else:
        correct_flag = (pred_idx is not None and gold_idx is not None and pred_idx == gold_idx)

    return str(qid), pred_idx, gold_idx, top1_conf, correct_flag

def load_arrays_for_token(root: str, token: str, dataset: str, model: str):
    """해당 토큰 결과에서 (qid, pred_idx, gold_idx, conf, correct)을 모아 numpy로 반환"""
    fp = locate_jsonl(root, token, dataset, model)
    rows = load_records(fp)
    qids: List[str] = []
    preds: List[int] = []
    golds: List[int] = []
    confs: List[float] = []
    rights: List[int] = []
    for r in rows:
        qid, pi, gi, conf, cf = parse_one_rec(r, token)
        if qid is None or pi is None or gi is None or conf is None:
            continue
        qids.append(qid)
        preds.append(int(pi))
        golds.append(int(gi))
        confs.append(float(conf))
        rights.append(1 if cf else 0)
    return np.array(qids), np.array(preds), np.array(golds), np.array(confs), np.array(rights, dtype=int)


# ───────────── ECE/히스토그램/리라이어빌리티 ─────────────

def ece_from_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0., 1., n_bins + 1)
    N = len(conf)
    e = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else ((conf > lo) & (conf <= hi))
        if not np.any(mask): 
            continue
        e += (mask.sum()/N) * abs(correct[mask].mean() - conf[mask].mean())
    return float(e)

def plot_hist_separated(model: str, dataset: str, conf_by_tok: Dict[str, np.ndarray], right_by_tok: Dict[str, np.ndarray], outdir: Path, bins=20):
    fig = plt.figure()
    for t, conf in conf_by_tok.items():
        if conf.size == 0: continue
        right = right_by_tok[t]
        ok  = conf[right==1]
        bad = conf[right==0]
        # 오답(연한), 정답(진한)
        plt.hist(bad, bins=bins, range=(0,1), alpha=0.35, color=TOKEN_COL.get(t,"gray"),
                 density=True, label=f"{t} {TOKEN_LABEL.get(t,t)} wrong (n={len(bad)})")
        plt.hist(ok,  bins=bins, range=(0,1), alpha=0.65, color=TOKEN_COL.get(t,"gray"),
                 density=True, label=f"{t} {TOKEN_LABEL.get(t,t)} correct (n={len(ok)})")
    plt.xlabel("Confidence (top-1 prob)")
    plt.ylabel("Density")
    plt.title(f"{model} — {dataset} | Confidence distribution (correct vs wrong)")
    plt.legend(fontsize=9)
    out_png = outdir / f"{safe_tag(model)}_{dataset}_conf_hist.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_reliability(model: str, dataset: str, conf_by_tok: Dict[str, np.ndarray], right_by_tok: Dict[str, np.ndarray], outdir: Path, n_bins=15):
    fig = plt.figure()
    xs = np.linspace(0,1,101)
    plt.plot(xs, xs, "--", color="black", linewidth=1, label="Ideal")
    for t, conf in conf_by_tok.items():
        if conf.size == 0: continue
        right = right_by_tok[t]
        bins = np.linspace(0.,1.,n_bins+1)
        mids = 0.5*(bins[:-1]+bins[1:])
        accs = []
        for b in range(n_bins):
            lo,hi = bins[b], bins[b+1]
            mask = (conf >= lo) & (conf <= hi) if b==0 else ((conf > lo) & (conf <= hi))
            accs.append(right[mask].mean() if np.any(mask) else np.nan)
        plt.plot(mids, np.array(accs, float), marker="o", color=TOKEN_COL.get(t,"gray"),
                 label=f"{t} {TOKEN_LABEL.get(t,t)}")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.title(f"{model} — {dataset} | Reliability by token")
    plt.legend()
    out_png = outdir / f"{safe_tag(model)}_{dataset}_reliability.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


# ───────────── 매칭 비율(MR) ─────────────

def mr_matrix(preds_by_tok: Dict[str, Dict[str,int]], mask_qids: Optional[set]=None, tok_order: Optional[List[str]]=None):
    """preds_by_tok[t][qid]=pred_idx → VxV 매칭 비율 행렬"""
    order = tok_order or sorted(preds_by_tok.keys())
    V = len(order)
    M = np.zeros((V,V), dtype=float)
    for i, ti in enumerate(order):
        for j, tj in enumerate(order):
            Qi = set(preds_by_tok[ti].keys())
            Qj = set(preds_by_tok[tj].keys())
            keys = Qi & Qj
            if mask_qids is not None:
                keys &= mask_qids
            if not keys:
                M[i,j] = np.nan
            else:
                same = sum(1 for q in keys if preds_by_tok[ti][q] == preds_by_tok[tj][q])
                M[i,j] = same / float(len(keys))
    return order, M

def plot_mr(model: str, dataset: str, tokens: List[str],
            preds_by_tok: Dict[str, Dict[str,int]],
            correct_by_tok: Dict[str, Dict[str,int]],
            outdir: Path, tag: str):

    order = tokens
    # ALL
    ord_all, M_all = mr_matrix(preds_by_tok, tok_order=order)
    # CORRECT-ONLY (두 토큰 모두 정답인 qid만)
    both_correct = None
    for t in order:
        q_ok = {q for q,v in correct_by_tok[t].items() if v==1}
        both_correct = q_ok if both_correct is None else (both_correct & q_ok)
    ord_c, M_c = mr_matrix(preds_by_tok, mask_qids=both_correct, tok_order=order)
    # WRONG-ONLY (두 토큰 모두 오답인 qid만)
    both_wrong = None
    for t in order:
        q_bad = {q for q,v in correct_by_tok[t].items() if v==0}
        both_wrong = q_bad if both_wrong is None else (both_wrong & q_bad)
    ord_w, M_w = mr_matrix(preds_by_tok, mask_qids=both_wrong, tok_order=order)

    def _draw(M, suffix):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(M, vmin=0, vmax=1, aspect="equal", cmap="viridis")
        ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
        ax.set_xticklabels(order); ax.set_yticklabels(order)
        plt.title(f"Matching Ratio — {model} / {dataset} ({suffix})")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i,j]):
                    ax.text(j, i, f"{100*M[i,j]:.1f}%", ha="center", va="center", color="white" if M[i,j]<0.5 else "black")
        fig.colorbar(im, ax=ax)
        out_png = outdir / f"{safe_tag(model)}_{dataset}_mr_{suffix}.png"
        fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

    _draw(M_all, "all")
    _draw(M_c,   "correct")
    _draw(M_w,   "wrong")


# ───────────── 메인 ─────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--tokens",   nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--models",   nargs="*", default=None)
    ap.add_argument("--outdir",   default="viz_out/ece")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--n_bins_ece", type=int, default=15)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); ensure_dir(outdir)

    auto_models = collect_models(args.root, args.tokens, args.datasets)
    models = filter_models(auto_models, args.models)

    if not models:
        print("[WARN] 모델 디렉토리 없음"); return

    tsv_lines = ["model\tdataset\ttoken\tn\tacc\tece\n"]

    for model in models:
        for ds in args.datasets:
            conf_by_tok : Dict[str,np.ndarray] = {}
            right_by_tok: Dict[str,np.ndarray] = {}
            # MR용: qid 단위 pred/정오 보관
            preds_by_tok   : Dict[str,Dict[str,int]] = {}
            correct_by_tok : Dict[str,Dict[str,int]] = {}

            for t in args.tokens:
                qid, pred, gold, conf, right = load_arrays_for_token(args.root, t, ds, model)
                conf_by_tok[t]  = conf
                right_by_tok[t] = right

                # MR용 dict
                preds_by_tok[t]   = {q:int(pi) for q,pi in zip(qid, pred)}
                correct_by_tok[t] = {q:int(r)  for q,r  in zip(qid, right)}

                if conf.size > 0:
                    ece_val = ece_from_conf(conf, right, n_bins=args.n_bins_ece)
                    tsv_lines.append(f"{model}\t{ds}\t{t}\t{len(conf)}\t{right.mean():.4f}\t{ece_val:.4f}\n")

            # 시각화
            plot_hist_separated(model, ds, conf_by_tok, right_by_tok, outdir, bins=args.bins)
            plot_reliability(model, ds, conf_by_tok, right_by_tok, outdir, n_bins=args.n_bins_ece)
            # MR 히트맵 3종
            plot_mr(model, ds, args.tokens, preds_by_tok, correct_by_tok, outdir, tag="")

            # 디버그: 토큰 간 전부 동일하면 경고
            if len(args.tokens) >= 2:
                first = args.tokens[0]
                for other in args.tokens[1:]:
                    inter = set(preds_by_tok[first]).intersection(preds_by_tok[other])
                    if inter:
                        same = sum(1 for q in inter if preds_by_tok[first][q]==preds_by_tok[other][q])
                        if same == len(inter):
                            print(f"[WARN] {model}/{ds}: {first} vs {other} 예측이 교집합 {len(inter)}개에서 100% 동일합니다. (경로/파서 확인 필요)")

    # TSV 저장
    tsv_path = outdir / "ece_summary.tsv"
    with tsv_path.open("w", encoding="utf-8") as w:
        w.writelines(tsv_lines)
    print(f"[DONE] Saved: {tsv_path}")

if __name__ == "__main__":
    main()
