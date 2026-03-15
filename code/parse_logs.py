#!/usr/bin/env python3
"""
eval_clm.py FINAL CONDENSED REPORT 로그에서 Cyclic / PriDe / Ours(α=2) 구간별 수치를 파싱합니다.
여러 로그(15개 모델)를 넣으면 CSV로 뽑아서 직접 평균 내기 좋게 하고, 옵션으로 N개 평균 요약도 출력합니다.

구간:
  - Cyclic: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
  - PriDe:   2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100  (default_pride_αX%)
  - Ours:    2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100  (ours_pride_sqrt_α2_X%, α=2 고정)

실행:
  python parse_logs.py --logs model1.log model2.log ...
  python parse_logs.py --logs my_logs/*.log --csv parsed.csv
  python parse_logs.py --logs my_logs/*.log --aggregate   # N개 로그에 대해 평균 요약 출력
"""

import argparse
import csv
import re
import sys
from collections import defaultdict

import numpy as np


# 목표 구간 (사용자 지정과 동일)
CYCLIC_FRACS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PRIDE_OURS_FRACS = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
OURS_ALPHA = 2


def parse_val(s):
    """'0.8035±0.0012' → (0.8035, 0.0012), '5.000' → (5.0, 0.0)"""
    s = (s or "").strip()
    if not s:
        return None, None
    if "±" in s:
        parts = s.split("±", 1)
        return float(parts[0]), float(parts[1])
    try:
        return float(s), 0.0
    except ValueError:
        return None, None


def parse_eval_clm_log(content: str, log_name: str):
    """
    eval_clm.py 로그 본문에서 다음 형식 파싱:
      ---- default + pride ----
      default_pride_α2% : cost=..., acc=..., recall_std=...
      ---- ours + pride (Online Sqrt) ----
      ours_pride_sqrt_α2_002% : cost=..., acc=..., recall_std=...
      ---- ours ----
      ours_002% : ...
      ---- cyclic ----
      cyclic_010% : cost=..., acc=..., recall_std=...

    반환: list of dict { "model": log_name, "method": "Cyclic"|"PriDe"|"Ours", "p": int|float, "acc": float, "acc_std": float, "recall_std": float, "recall_std_std": float, "cost": float, "cost_std": float }
    """
    rows = []
    # 로그 앞에 [timestamp] [eval_clm.py:xxxx] 가 올 수 있음
    # 패턴: 라인 안에 cyclic_010% : cost=1.401, acc=0.8035±0.0007, recall_std=0.0364±0.0022
    cyclic_re = re.compile(
        r"cyclic_(\d{3})%\s*:\s*cost=([\d\.±]+)(?:,\s*acc=([\d\.±]+))?(?:,\s*recall_std=([\d\.±]+))?"
    )
    default_pride_re = re.compile(
        r"default_pride_α([\d\.]+)%\s*:\s*cost=([\d\.±]+)(?:,\s*acc=([\d\.±]+))?(?:,\s*recall_std=([\d\.±]+))?"
    )
    ours_pride_sqrt_re = re.compile(
        r"ours_pride_sqrt_α2_(\d+)%\s*:\s*cost=([\d\.±]+)(?:,\s*acc=([\d\.±]+))?(?:,\s*recall_std=([\d\.±]+))?"
    )

    for line in content.splitlines():
        line = line.strip()
        # Cyclic: cyclic_010%, cyclic_020%, ...
        m = cyclic_re.search(line)
        if m:
            p = int(m.group(1))
            if p in CYCLIC_FRACS:
                cost_m, cost_s = parse_val(m.group(2))
                acc_m, acc_s = parse_val(m.group(3)) if m.group(3) else (None, None)
                r_m, r_s = parse_val(m.group(4)) if m.group(4) else (None, None)
                if acc_m is not None or cost_m is not None:
                    rows.append({
                        "model": log_name,
                        "method": "Cyclic",
                        "p": p,
                        "acc": acc_m,
                        "acc_std": acc_s if acc_s is not None else 0.0,
                        "recall_std": r_m,
                        "recall_std_std": r_s if r_s is not None else 0.0,
                        "cost": cost_m,
                        "cost_std": cost_s if cost_s is not None else 0.0,
                    })
            continue

        # PriDe: default_pride_α2%, α5%, ...
        m = default_pride_re.search(line)
        if m:
            p = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            if p in PRIDE_OURS_FRACS:
                cost_m, cost_s = parse_val(m.group(2))
                acc_m, acc_s = parse_val(m.group(3)) if m.group(3) else (None, None)
                r_m, r_s = parse_val(m.group(4)) if m.group(4) else (None, None)
                if acc_m is not None or cost_m is not None:
                    rows.append({
                        "model": log_name,
                        "method": "PriDe",
                        "p": p,
                        "acc": acc_m,
                        "acc_std": acc_s if acc_s is not None else 0.0,
                        "recall_std": r_m,
                        "recall_std_std": r_s if r_s is not None else 0.0,
                        "cost": cost_m,
                        "cost_std": cost_s if cost_s is not None else 0.0,
                    })
            continue

        # Ours (α=2): ours_pride_sqrt_α2_002%, ...
        m = ours_pride_sqrt_re.search(line)
        if m:
            p = int(m.group(1))
            if p in PRIDE_OURS_FRACS:
                cost_m, cost_s = parse_val(m.group(2))
                acc_m, acc_s = parse_val(m.group(3)) if m.group(3) else (None, None)
                r_m, r_s = parse_val(m.group(4)) if m.group(4) else (None, None)
                if acc_m is not None or cost_m is not None:
                    rows.append({
                        "model": log_name,
                        "method": "Ours",
                        "p": p,
                        "acc": acc_m,
                        "acc_std": acc_s if acc_s is not None else 0.0,
                        "recall_std": r_m,
                        "recall_std_std": r_s if r_s is not None else 0.0,
                        "cost": cost_m,
                        "cost_std": cost_s if cost_s is not None else 0.0,
                    })
            continue

    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Parse eval_clm.py FINAL CONDENSED REPORT from log files. Cyclic 10..100, PriDe 2..100, Ours(α=2) 2..100."
    )
    ap.add_argument("--logs", nargs="+", required=True, help="로그 파일 경로 (예: logs/*.log)")
    ap.add_argument("--csv", type=str, default="", help="CSV로 저장 (열: model, method, p, acc, acc_std, recall_std, recall_std_std, cost, cost_std)")
    ap.add_argument("--aggregate", action="store_true", help="N개 로그에 대해 (method, p)별 평균 요약 출력")
    args = ap.parse_args()

    all_rows = []
    for log_path in args.logs:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            print(f"[warn] Skip {log_path}: {e}", file=sys.stderr)
            continue
        name = log_path.split("/")[-1].split("\\")[-1]
        rows = parse_eval_clm_log(content, name)
        all_rows.extend(rows)

    if not all_rows:
        print("Error: No lines parsed from any log.", file=sys.stderr)
        sys.exit(1)

    # CSV 저장 (15개 평균 내기 좋게)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["model", "method", "p", "acc", "acc_std", "recall_std", "recall_std_std", "cost", "cost_std"],
                extrasaction="ignore",
            )
            w.writeheader()
            for r in all_rows:
                w.writerow({k: ("" if v is None else v) for k, v in r.items()})
        print(f"Wrote {len(all_rows)} rows to {args.csv}", file=sys.stderr)

    # 집계 요약 (--aggregate 시)
    if args.aggregate:
        # (method, p) -> lists of acc, recall_std, cost
        agg = defaultdict(lambda: {"acc": [], "acc_std": [], "recall_std": [], "recall_std_std": [], "cost": [], "cost_std": []})
        for r in all_rows:
            k = (r["method"], r["p"])
            if r.get("acc") is not None:
                agg[k]["acc"].append(r["acc"])
                agg[k]["acc_std"].append(r.get("acc_std") or 0.0)
            if r.get("recall_std") is not None:
                agg[k]["recall_std"].append(r["recall_std"])
                agg[k]["recall_std_std"].append(r.get("recall_std_std") or 0.0)
            if r.get("cost") is not None:
                agg[k]["cost"].append(r["cost"])
                agg[k]["cost_std"].append(r.get("cost_std") or 0.0)

        n_models = len(set(r["model"] for r in all_rows))
        print(f"\n==== AGGREGATED OVER {n_models} MODELS (eval_clm log parse) ====")
        method_order = ["Cyclic", "PriDe", "Ours"]
        for method in method_order:
            fracs = CYCLIC_FRACS if method == "Cyclic" else PRIDE_OURS_FRACS
            print(f"\n--- {method} ---")
            for p in fracs:
                k = (method, p)
                if k not in agg:
                    continue
                d = agg[k]
                acc_list = d["acc"]
                if not acc_list:
                    continue
                acc_m = float(np.mean(acc_list))
                acc_s = float(np.mean(d["acc_std"])) if d["acc_std"] else 0.0
                r_m = float(np.mean(d["recall_std"])) if d["recall_std"] else None
                r_s = float(np.mean(d["recall_std_std"])) if d["recall_std_std"] else 0.0
                c_m = float(np.mean(d["cost"])) if d["cost"] else None
                c_s = float(np.mean(d["cost_std"])) if d["cost_std"] else 0.0
                acc_str = f"{acc_m:.4f}±{acc_s:.4f}"
                rstr = f", recall_std={r_m:.4f}±{r_s:.4f}" if r_m is not None else ""
                cstr = f", cost={c_m:.4f}±{c_s:.4f}" if c_m is not None else ""
                print(f"  {p}% : acc={acc_str}{rstr}{cstr}")
        print("\n=========================================================================\n")
    else:
        n_models = len(set(r["model"] for r in all_rows))
        print(f"Parsed {len(all_rows)} rows from {n_models} log(s). Use --csv out.csv to save, --aggregate to print mean.", file=sys.stderr)


if __name__ == "__main__":
    main()
