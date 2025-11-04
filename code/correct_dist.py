#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
맞춘 데이터( pred == gold )만 모아서 분포 PNG 저장:
- 입력 트리: <root>/<dataset>/<model>/<T*/P0.jsonl>
- 출력: out_dir/correct_idx_dist_<DATASET>.png (+ 요약 CSV/로그)
"""

import os, sys, json, argparse, numpy as np
from pathlib import Path
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

# -------- 공통 파서 --------
def read_jsonl(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def normalize_letter(x: str):
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
    if ch.isalpha():
        ch = ch.upper()
    return ch

def parse_pred_gold(rec):
    """
    다양한 키를 받아서 (qid, pred_idx, gold_idx, pred_letter, gold_letter) 반환
    - 인덱스가 있으면 그걸 우선 사용
    - 없을 때는 레터 비교도 가능하도록 레터 병행 추출
    """
    qid = rec.get("qid", rec.get("id"))
    if qid is None:
        return None, None, None, None, None

    pred_raw = (rec.get("pred_idx") or rec.get("prediction_idx") or
                rec.get("pred") or rec.get("prediction"))
    gold_raw = (rec.get("gold_idx") or rec.get("label_idx") or
                rec.get("label") or rec.get("gold") or rec.get("answer"))

    def to_int_or_none(v):
        try:
            s = str(v)
            if s.lstrip("-").isdigit():
                return int(s)
        except Exception:
            pass
        return None

    pred_idx = to_int_or_none(pred_raw)
    gold_idx = to_int_or_none(gold_raw)

    pred_letter = None
    for k in ("pred_letter", "prediction_letter", "pred", "prediction"):
        if k in rec and rec[k] is not None:
            pred_letter = normalize_letter(rec[k]); break

    gold_letter = None
    for k in ("gold_letter", "label_letter", "answer", "gold", "label"):
        if k in rec and rec[k] is not None:
            gold_letter = normalize_letter(rec[k]); break

    return str(qid), pred_idx, gold_idx, pred_letter, gold_letter

# -------- 데이터 수집 --------
def collect_correct_records(model_dir, token):
    """
    model_dir/<token>/P0.jsonl 에서 '맞춘' 행만 골라 반환
    - pred_idx == gold_idx (둘 다 >=0) 우선
    - 아니면 pred_letter == gold_letter 로 대체 판단
    """
    jf = os.path.join(model_dir, token, "P0.jsonl")
    if not os.path.isfile(jf):
        return [], {"total":0, "usable":0, "skip_neg1":0}

    rows = read_jsonl(jf)
    out = []
    total = len(rows)
    usable = 0
    skip_neg1 = 0
    for rec in rows:
        qid, pi, gi, pl, gl = parse_pred_gold(rec)
        if qid is None:
            continue

        # -1(미응답) 제외
        if pi is not None and pi < 0:
            skip_neg1 += 1
            continue
        if gi is not None and gi < 0:
            # 금상첨화로 gold가 -1인 경우도 제외
            continue

        decided_correct = None
        # 인덱스 기준이 가장 견고함
        if pi is not None and gi is not None:
            usable += 1
            decided_correct = (pi == gi)
            if decided_correct:
                out.append({"qid": qid, "gold_idx": gi, "pred_idx": pi})
            continue

        # 인덱스 없으면 레터로 판단 (일부 포맷 대비)
        if pl and gl:
            usable += 1
            decided_correct = (pl == gl)
            if decided_correct:
                # 인덱스 모르면 None으로 둠 (분포는 인덱스 없는 건 집계 제외)
                out.append({"qid": qid, "gold_idx": None, "pred_idx": None})

    return out, {"total": total, "usable": usable, "skip_neg1": skip_neg1}

# -------- 그리기 --------
def draw_bar_counts(counts, title, out_png, xticklabels=None):
    xs = np.arange(len(counts))
    plt.figure(figsize=(8,4))
    plt.bar(xs, counts)
    plt.title(title)
    plt.xlabel("Answer index (gold_idx)")
    plt.ylabel("Count of CORRECT")
    if xticklabels is not None:
        plt.xticks(xs, xticklabels)
    else:
        plt.xticks(xs, [str(i) for i in xs])
    for i, c in enumerate(counts):
        plt.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()

# -------- 메인 --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["arc","csqa"])
    ap.add_argument("--tokens", nargs="+", default=["T0","T1","T2"])
    ap.add_argument("--out_dir", default="viz_out/correct_dist")
    ap.add_argument("--models", nargs="*", default=None, help="모델명 부분문자열 필터")
    ap.add_argument("--max_idx", type=int, default=4, help="선택지 개수(인덱스 축 길이) 기본 4")
    ap.add_argument("--save_csv", action="store_true")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary_lines = []

    for ds in args.datasets:
        dsd = os.path.join(args.root, ds)
        if not os.path.isdir(dsd):
            print(f"[WARN] dataset dir missing: {dsd}")
            continue

        models = [m for m in os.listdir(dsd) if os.path.isdir(os.path.join(dsd,m))]
        if args.models:
            filt = []
            for m in models:
                for s in args.models:
                    if s in m:
                        filt.append(m); break
            models = sorted(set(filt))

        # 집계: gold_idx 분포(맞춘 것만)
        correct_gold_counts = np.zeros(args.max_idx, dtype=int)
        total_rows = usable_rows = skipped_neg1 = 0

        # 원하면 per-item CSV도 생성
        csv_rows = []

        for m in models:
            model_dir = os.path.join(dsd, m)
            for t in args.tokens:
                recs, stat = collect_correct_records(model_dir, t)
                total_rows += stat["total"]
                usable_rows += stat["usable"]
                skipped_neg1 += stat["skip_neg1"]

                for r in recs:
                    gi = r["gold_idx"]
                    if gi is not None and 0 <= gi < args.max_idx:
                        correct_gold_counts[gi] += 1
                    if args.save_csv:
                        csv_rows.append([ds, m, t, r["qid"], r["pred_idx"], r["gold_idx"]])

        title = f"[{ds.upper()}] CORRECT distribution by gold_idx (aggregated)"
        out_png = os.path.join(args.out_dir, f"correct_idx_dist_{ds}.png")
        draw_bar_counts(correct_gold_counts, title, out_png)
        print(f"[DONE] {out_png}  counts={correct_gold_counts.tolist()}")

        if args.save_csv and csv_rows:
            csv_path = os.path.join(args.out_dir, f"correct_items_{ds}.csv")
            with open(csv_path, "w", encoding="utf-8") as w:
                w.write("dataset,model,token,qid,pred_idx,gold_idx\n")
                for row in csv_rows:
                    w.write(",".join("" if v is None else str(v) for v in row) + "\n")
            print(f"[DONE] {csv_path}  (rows={len(csv_rows)})")

        summary_lines.append(
            f"{ds}: total={total_rows}, usable={usable_rows}, skip_pred_-1={skipped_neg1}, "
            f"correct_counts={correct_gold_counts.tolist()}"
        )

    # 전체 요약 저장
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as w:
        w.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()
