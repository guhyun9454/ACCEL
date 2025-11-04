#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, math, os, sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

LETTER2IDX = {c:i for i,c in enumerate(list("ABCDE"))}

def read_jsonl(paths):
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                yield json.loads(line)

def get_idx(rec, gold=True):
    """다양한 키(gold_idx/gold/gold_letter, pred_idx/pred/pred_letter)에서 인덱스 추출."""
    if gold:
        for k in ("gold_idx","gold","answer_idx","label_idx"):
            if k in rec and rec[k] is not None:
                return rec[k]
        if "gold_letter" in rec and rec["gold_letter"]:
            return LETTER2IDX.get(str(rec["gold_letter"]).strip().upper(), -1)
    else:
        for k in ("pred_idx","pred"):
            if k in rec and rec[k] is not None:
                return rec[k]
        if "pred_letter" in rec and rec["pred_letter"]:
            return LETTER2IDX.get(str(rec["pred_letter"]).strip().upper(), -1)
    return -1

def detect_num_options(records):
    """qid / 레터 사용 / 최댓값을 바탕으로 4 또는 5 자동 추정."""
    max_idx = -1
    sample_qids = set()
    for r in records:
        g = get_idx(r, gold=True)
        p = get_idx(r, gold=False)
        max_idx = max(max_idx, g, p)
        if "qid" in r and isinstance(r["qid"], str):
            sample_qids.add(r["qid"])
    # 힌트: csqa 포함 → 5, arc 포함 → 4
    hint5 = any("csqa" in q.lower() for q in sample_qids)
    hint4 = any("arc" in q.lower() for q in sample_qids)
    if hint5 and not hint4:
        return 5
    if hint4 and not hint5:
        return 4
    # 값 기반 추정
    if max_idx >= 4:
        return 5
    return 4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", help="prediction jsonl(s)")
    ap.add_argument("--num-options", type=int, default=None, help="4 or 5 (자동감지 기본)")
    ap.add_argument("--mode", choices=["count","acc"], default="count",
                    help="count: 정답 위치별 '맞춘 개수', acc: 정답 위치별 '정확도'")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    records = list(read_jsonl(args.jsonl))
    if not records:
        print("입력에 레코드가 없습니다.")
        sys.exit(1)

    if args.num_options is None:
        num_options = detect_num_options(records)
    else:
        num_options = args.num_options

    totals = Counter()    # 정답 위치별 전체 문항 수
    corrects = Counter()  # 정답 위치별 맞춘 개수

    n_bad = 0
    for r in records:
        g = get_idx(r, gold=True)
        p = get_idx(r, gold=False)
        if g is None or g == -1:
            n_bad += 1
            continue
        if not (0 <= g < num_options):
            # 예: csqa인데 4지로 잘못 지정된 경우 등
            n_bad += 1
            continue
        totals[g] += 1
        if p is not None and p != -1 and p == g:
            corrects[g] += 1

    n_total = sum(totals.values())
    n_correct = sum(corrects.values())

    if n_total == 0:
        print("유효한 gold 인덱스가 하나도 없습니다. (모두 -1 또는 범위 밖)")
        print(f"총 레코드: {len(records)}, 무효: {n_bad}")
        sys.exit(2)

    print(f"[요약] 총 레코드: {len(records)} | 유효 gold: {n_total} | 맞춘 개수: {n_correct}")
    if n_bad > 0:
        print(f"  경고: 무효 레코드 {n_bad}개(키 불일치/-1/범위 밖). 그래프에서 제외됨.")

    xs = list(range(num_options))
    if args.mode == "count":
        ys = [corrects[i] for i in xs]
        ylabel = "Count of CORRECT"
    else:
        ys = [(corrects[i]/totals[i]) if totals[i] else 0.0 for i in xs]
        ylabel = "Accuracy (correct/total)"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(xs, ys, width=0.6)
    for i, y in enumerate(ys):
        ax.text(i, y + (0.02 if args.mode=="acc" else (max(ys)*0.02 if max(ys)>0 else 0.02)),
                f"{y:.2f}" if args.mode=="acc" else str(y),
                ha="center", va="bottom", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xlabel("Answer index (gold_idx)")
    ax.set_ylabel(ylabel)

    title = args.title or f"CORRECT distribution by gold_idx ({'acc' if args.mode=='acc' else 'count'})"
    ax.set_title(title)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"저장됨: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
