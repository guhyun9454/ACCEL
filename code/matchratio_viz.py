#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ece_viz.py  — confidence 분포 + reliability + matching ratio (MR)

지원 경로:
  1) <root>/<token>/<dataset>/<model>/<dataset>.jsonl
  2) <root>/<token>/<dataset>/<model>/<dataset>/<dataset>.jsonl  ← 현재 구조

출력:
  - 히스토그램(정답만 / 오답만)  [토큰별 색상: T0=red, T1=blue, T2=green; 범례명 T0→ABCD, T1→abcd, T2→1234]
  - 리라이어빌리티 다이어그램(토큰별 한 장)
  - ECE/정확도 요약 TSV
  - Matching Ratio 히트맵 3종:
      (a) MR_all: 전 샘플에서 pred_idx 일치 비율
      (b) MR_wrong: 두 토큰이 모두 오답인 샘플에서 같은 오답을 골랐는지 비율
      (c) Correct-set Jaccard: 정답 qid 집합 간 Jaccard(겹침) 비율

예시:
  python code/ece_viz.py \
    --root results \
    --datasets arc csqa \
    --models "0s_Llama-3.2-1B-Instruct" "0s_Llama-3.2-3B-Instruct" "0s_Qwen2.5-1.5B-Instruct" "0s_gemma-3-1b-it" \
    --outdir viz_out/ece \
    --per_model_dir --xmin 0.3 --xmax 1.0 --also_combined
"""

import os, json, argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set

import numpy as np

# 헤드리스 안전
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt


# ────────────── 유틸 ──────────────
TOKEN_LABEL_MAP = {"T0": "ABCD", "T1": "abcd", "T2": "1234"}
COLOR_MAP = {"T0": "red", "T1": "blue", "T2": "green"}

def safe_name(s: str) -> str:
    return str(s).replace("/", "__").replace("\\", "__").strip()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def locate_jsonl(root: str, token: str, dataset: str, model: str) -> Optional[Path]:
    base = Path(root) / token / dataset / model
    cand1 = base / f"{dataset}.jsonl"
    cand2 = base / dataset / f"{dataset}.jsonl"
    if cand1.is_file(): return cand1
    if cand2.is_file(): return cand2
    return None

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

def fold_probs(p: np.ndarray, n_opts_hint: Optional[int] = None) -> np.ndarray:
    arr = np.array(p, dtype=float)
    if arr.ndim == 2:  # 여러 뷰 평균
        arr = arr.mean(axis=0)
    if arr.ndim == 1 and arr.size % 2 == 0 and (n_opts_hint is None or arr.size == 2*n_opts_hint):
        arr = arr.reshape(2, arr.size//2).sum(axis=0)
    arr = np.clip(arr, 1e-12, None)
    arr = arr / arr.sum()
    return arr

def ece_from_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0., 1., n_bins+1)
    N = len(conf); ece_val = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else ((conf > lo) & (conf <= hi))
        if not np.any(mask): continue
        acc_b  = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece_val += (mask.sum()/N) * abs(acc_b - conf_b)
    return float(ece_val)


# ────────────── 로딩 ──────────────
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
    if not wants: return all_models
    wants_a = [w.lower() for w in wants]
    wants_b = [safe_name(w).lower() for w in wants]
    out = []
    for m in all_models:
        ml = m.lower()
        if any(w in ml for w in wants_a) or any(w in ml for w in wants_b):
            out.append(m)
    return sorted(set(out))

def load_token_records(fp: Path) -> List[dict]:
    if not fp or not fp.is_file(): return []
    rows: List[dict] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            rows.append(flatten_record(rec))
    return rows

def extract_conf_correct(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    confs: List[float] = []; rights: List[int] = []
    for r in rows:
        p = r.get("probs") or r.get("all_probs")
        if p is None: continue
        probs = fold_probs(p)
        confs.append(float(np.max(probs)))

        c = r.get("correct", None)
        if isinstance(c, str): c = c.lower() in ("true","1","yes")
        if isinstance(c, bool):
            rights.append(1 if c else 0); continue

        p_letter = normalize_letter(r.get("sampled"))
        g_letter = normalize_letter(r.get("ideal"))
        if p_letter is not None and g_letter is not None:
            rights.append(1 if p_letter == g_letter else 0); continue

        pred_idx = None
        for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx","output_idx"):
            if k in r:
                try:
                    vi = int(r[k])
                    if vi >= 0: pred_idx = vi; break
                except Exception: pass
        gold_idx = None
        for k in ("gold_idx","label_idx","correct_idx","target_idx","solution_idx"):
            if k in r:
                try:
                    vi = int(r[k])
                    if vi >= 0: gold_idx = vi; break
                except Exception: pass
        if pred_idx is None: pred_idx = int(np.argmax(probs))
        if gold_idx is None:
            confs.pop(); continue
        rights.append(1 if pred_idx == gold_idx else 0)

    if not confs: return np.array([]), np.array([])
    return np.array(confs, dtype=float), np.array(rights, dtype=int)

def build_qid_pred_right(rows: List[dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    각 qid에 대해 (pred_idx, correct) 추출.
    pred_idx는 'pred_idx' 없으면 probs argmax 사용.
    """
    pred: Dict[str, int] = {}
    right: Dict[str, int] = {}
    for r in rows:
        qid = r.get("qid") or r.get("id") or r.get("uid") or r.get("question_id")
        if qid is None:
            if "idx" in r: qid = str(r["idx"])
            else: continue
        qid = str(qid)

        probs = r.get("probs") or r.get("all_probs")
        pred_idx = None
        for k in ("pred_idx","prediction_idx","answer_idx","choice_idx","selected_idx","output_idx"):
            if k in r:
                try:
                    vi = int(r[k])
                    if vi >= 0: pred_idx = vi; break
                except Exception: pass
        if pred_idx is None and probs is not None:
            pred_idx = int(np.argmax(fold_probs(probs)))
        if pred_idx is None:  # pred를 전혀 못구하면 스킵
            continue

        # correct
        c = r.get("correct", None)
        if isinstance(c, str): c = c.lower() in ("true","1","yes")
        if isinstance(c, bool):
            is_right = 1 if c else 0
        else:
            # gold 비교
            gold_idx = None
            for k in ("gold_idx","label_idx","correct_idx","target_idx","solution_idx"):
                if k in r:
                    try:
                        vi = int(r[k])
                        if vi >= 0: gold_idx = vi; break
                    except Exception: pass
            if gold_idx is None:
                # 문자 비교 마지막 시도
                p_letter = normalize_letter(r.get("sampled"))
                g_letter = normalize_letter(r.get("ideal"))
                is_right = 1 if (p_letter is not None and g_letter is not None and p_letter==g_letter) else 0
            else:
                is_right = 1 if pred_idx == gold_idx else 0

        pred[qid] = int(pred_idx)
        right[qid] = int(is_right)
    return pred, right


# ────────────── 시각화 ──────────────
def _plot_hist(ax, data_list, labels, colors, bins, rng, title):
    for (vec, lab, col) in zip(data_list, labels, colors):
        if vec.size == 0: continue
        ax.hist(vec, bins=bins, range=rng, density=True, alpha=0.65, color=col, label=lab)
    ax.set_xlabel("Confidence (top-1 prob)"); ax.set_ylabel("Density"); ax.set_title(title)
    ax.legend(fontsize=9)

def plot_hist_split_by_token(model_tag, dataset, conf_by_tok, right_by_tok, outdir, bins=20, per_model_dir=False, xlim=(0.0,1.0)):
    base = outdir / safe_name(model_tag) if per_model_dir else outdir
    ensure_dir(base)

    # Correct only
    data, labels, colors = [], [], []
    for t in sorted(conf_by_tok.keys()):
        conf = conf_by_tok[t]; right = right_by_tok[t]
        vec = conf[right == 1]
        if vec.size == 0: continue
        data.append(vec); labels.append(f"{TOKEN_LABEL_MAP.get(t,t)} (n={len(vec)})"); colors.append(COLOR_MAP.get(t,"gray"))
    if data:
        fig, ax = plt.subplots()
        _plot_hist(ax, data, labels, colors, bins, xlim, f"{model_tag} — {dataset} | Confidence (correct only)")
        out_png = base / f"{safe_name(model_tag)}_{dataset}_conf_hist_correct.png"
        fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
        print(f"[SAVE] {out_png}")

    # Wrong only
    data, labels, colors = [], [], []
    for t in sorted(conf_by_tok.keys()):
        conf = conf_by_tok[t]; right = right_by_tok[t]
        vec = conf[right == 0]
        if vec.size == 0: continue
        data.append(vec); labels.append(f"{TOKEN_LABEL_MAP.get(t,t)} (n={len(vec)})"); colors.append(COLOR_MAP.get(t,"gray"))
    if data:
        fig, ax = plt.subplots()
        _plot_hist(ax, data, labels, colors, bins, xlim, f"{model_tag} — {dataset} | Confidence (wrong only)")
        out_png = base / f"{safe_name(model_tag)}_{dataset}_conf_hist_wrong.png"
        fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
        print(f"[SAVE] {out_png}")

def plot_hist_combined_overlay(model_tag, dataset, conf_by_tok, right_by_tok, outdir, bins=20, per_model_dir=False, xlim=(0.0,1.0)):
    base = outdir / safe_name(model_tag) if per_model_dir else outdir
    ensure_dir(base)
    fig = plt.figure(); any_data=False
    for t in sorted(conf_by_tok.keys()):
        conf = conf_by_tok[t]; right = right_by_tok[t]
        if conf.size == 0: continue
        any_data=True
        conf_ok  = conf[right==1]; conf_bad = conf[right==0]
        plt.hist(conf_bad, bins=bins, range=xlim, alpha=0.35, color=COLOR_MAP.get(t,"gray"),
                 label=f"{TOKEN_LABEL_MAP.get(t,t)} wrong (n={len(conf_bad)})", density=True)
        plt.hist(conf_ok,  bins=bins, range=xlim, alpha=0.55, color=COLOR_MAP.get(t,"gray"),
                 label=f"{TOKEN_LABEL_MAP.get(t,t)} correct (n={len(conf_ok)})", density=True)
    if any_data:
        plt.xlabel("Confidence (top-1 prob)"); plt.ylabel("Density")
        plt.title(f"{model_tag} — {dataset} | Confidence distribution by token")
        plt.legend(fontsize=9)
        out_png = base / f"{safe_name(model_tag)}_{dataset}_conf_hist.png"
        plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
        print(f"[SAVE] {out_png}")

def plot_reliability_by_token(model_tag, dataset, conf_by_tok, right_by_tok, outdir, n_bins=15, per_model_dir=False):
    base = outdir / safe_name(model_tag) if per_model_dir else outdir
    ensure_dir(base)
    fig = plt.figure()
    xs = np.linspace(0,1,101)
    plt.plot(xs,xs,linestyle="--",color="black",linewidth=1,label="Ideal")
    for t in sorted(conf_by_tok.keys()):
        conf = conf_by_tok[t]; right = right_by_tok[t]
        if conf.size == 0: continue
        bins = np.linspace(0.,1.,n_bins+1); mids = 0.5*(bins[:-1]+bins[1:])
        accs=[]
        for b in range(n_bins):
            lo,hi=bins[b],bins[b+1]
            mask = (conf >= lo) & (conf <= hi) if b==0 else ((conf>lo)&(conf<=hi))
            accs.append(right[mask].mean() if np.any(mask) else np.nan)
        accs = np.array(accs, float)
        plt.plot(mids, accs, marker="o", color=COLOR_MAP.get(t,"gray"),
                 label=f"{TOKEN_LABEL_MAP.get(t,t)}")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.ylim(0.0,1.0)
    plt.title(f"{model_tag} — {dataset} | Reliability by token")
    plt.legend()
    out_png = base / f"{safe_name(model_tag)}_{dataset}_reliability.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    print(f"[SAVE] {out_png}")

def _plot_matrix(M: np.ndarray, labels: List[str], title: str, out_png: Path, vmin=0.0, vmax=1.0, fmt="{:.1%}"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(M, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    # 숫자 표기
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i,j]):
                ax.text(j, i, fmt.format(M[i,j]), ha="center", va="center", fontsize=9)
    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[SAVE] {out_png}")

# ── MR 계산
def pairwise_pairs(tokens: List[str]) -> List[Tuple[str,str]]:
    return [(tokens[i], tokens[j]) for i in range(len(tokens)) for j in range(i+1, len(tokens))]

def compute_mr_matrices(pred_by_tok: Dict[str, Dict[str,int]],
                        right_by_tok: Dict[str, Dict[str,int]],
                        tokens: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    반환:
      MR_all      : 전 샘플에서 pred_idx 일치 비율
      MR_wrong    : 두 토큰이 모두 오답인 샘플에서 pred_idx 일치 비율(같은 오답을 골랐는지)
      Jacc_correct: 정답 qid 집합 Jaccard(겹침/합집합)
    """
    T = len(tokens)
    MR_all = np.full((T,T), np.nan, float)
    MR_wrong = np.full((T,T), np.nan, float)
    Jacc = np.full((T,T), np.nan, float)

    for i, ti in enumerate(tokens):
        for j, tj in enumerate(tokens):
            if i == j:
                MR_all[i,j] = 1.0; MR_wrong[i,j] = 1.0; Jacc[i,j] = 1.0
                continue
            Pi, Ri = pred_by_tok.get(ti, {}), right_by_tok.get(ti, {})
            Pj, Rj = pred_by_tok.get(tj, {}), right_by_tok.get(tj, {})
            keys = set(Pi.keys()) & set(Pj.keys())
            if not keys: continue

            # 전체 MR
            eq = [1 for k in keys if Pi[k] == Pj[k]]
            MR_all[i,j] = np.mean(eq) if eq else np.nan

            # 오답 MR (두 토큰 모두 오답인 공통 qid)
            wrong_keys = [k for k in keys if (Ri.get(k,0)==0 and Rj.get(k,0)==0)]
            if wrong_keys:
                eqw = [1 for k in wrong_keys if Pi[k] == Pj[k]]
                MR_wrong[i,j] = np.mean(eqw) if eqw else np.nan

            # 정답 세트 Jaccard
            Ai = {k for k,v in Ri.items() if v==1}
            Aj = {k for k,v in Rj.items() if v==1}
            u = len(Ai|Aj)
            Jacc[i,j] = (len(Ai&Aj)/u) if u>0 else np.nan

    labels = [t for t in tokens]
    return MR_all, MR_wrong, Jacc, labels


# ────────────── 메인 ──────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="부분문자열 매칭. 비우면 자동 탐색.")
    ap.add_argument("--outdir", default="viz_out/ece")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--n_bins_ece", type=int, default=15)
    ap.add_argument("--per_model_dir", action="store_true")
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=1.0)
    ap.add_argument("--skip_reliability", action="store_true")
    ap.add_argument("--also_combined", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); ensure_dir(outdir)
    xlim = (args.xmin, args.xmax)

    auto_models = collect_models(args.root, args.tokens, args.datasets)
    models = filter_models(auto_models, args.models)
    if not models:
        print("[WARN] 모델 디렉토리를 찾지 못했습니다."); return

    # TSV 헤더
    tsv_lines = ["model\tdataset\ttoken\tn\tacc\tece\n"]

    for model in models:
        for ds in args.datasets:
            conf_by_tok: Dict[str, np.ndarray] = {}
            right_by_tok_arr: Dict[str, np.ndarray] = {}

            # MR 계산을 위한 qid 맵
            pred_by_tok: Dict[str, Dict[str,int]] = {}
            right_by_tok_qid: Dict[str, Dict[str,int]] = {}

            for t in args.tokens:
                jf = locate_jsonl(args.root, t, ds, model)
                rows = load_token_records(jf) if jf is not None else []
                conf, right = extract_conf_correct(rows)
                conf_by_tok[t] = conf
                right_by_tok_arr[t] = right

                pred_qid, right_qid = build_qid_pred_right(rows)
                pred_by_tok[t] = pred_qid
                right_by_tok_qid[t] = right_qid

            # 히스토그램
            plot_hist_split_by_token(model, ds, conf_by_tok, right_by_tok_arr, outdir,
                                     bins=args.bins, per_model_dir=args.per_model_dir, xlim=xlim)
            if args.also_combined:
                plot_hist_combined_overlay(model, ds, conf_by_tok, right_by_tok_arr, outdir,
                                           bins=args.bins, per_model_dir=args.per_model_dir, xlim=xlim)
            # 리라이어빌리티
            if not args.skip_reliability:
                plot_reliability_by_token(model, ds, conf_by_tok, right_by_tok_arr, outdir,
                                          n_bins=args.n_bins_ece, per_model_dir=args.per_model_dir)

            # ECE/ACC 요약
            for t in args.tokens:
                conf = conf_by_tok[t]; right = right_by_tok_arr[t]
                if conf.size == 0: continue
                ece_val = ece_from_conf(conf, right, n_bins=args.n_bins_ece)
                acc_val = float(right.mean())
                tsv_lines.append(f"{model}\t{ds}\t{t}\t{len(conf)}\t{acc_val:.4f}\t{ece_val:.4f}\n")

            # ── Matching Ratio 히트맵 3종 ──
            MR_all, MR_wrong, Jacc, labels = compute_mr_matrices(pred_by_tok, right_by_tok_qid, args.tokens)
            base = outdir / safe_name(model) if args.per_model_dir else outdir

            _plot_matrix(MR_all,   labels, f"Matching Ratio — {model} / {ds}", base / f"{safe_name(model)}_{ds}_mr_all.png")
            _plot_matrix(MR_wrong, labels, f"Matching Ratio (wrong-only) — {model} / {ds}", base / f"{safe_name(model)}_{ds}_mr_wrong.png")
            _plot_matrix(Jacc,     labels, f"Correct-set Overlap (Jaccard) — {model} / {ds}", base / f"{safe_name(model)}_{ds}_correct_jaccard.png")

    # TSV 저장
    tsv_path = outdir / "ece_summary.tsv"
    ensure_dir(tsv_path.parent)
    with tsv_path.open("w", encoding="utf-8") as w:
        w.writelines(tsv_lines)
    print(f"[DONE] Saved: {tsv_path}")

if __name__ == "__main__":
    main()
