#!/usr/bin/env python3
"""
15개 모델의 터미널 로그(.log / .txt)에서 report_three_methods 출력을 파싱해
모델별 평균·표준편차를 다시 평균 내어 한 번에 출력합니다.
JSON/curve를 다시 뽑지 않아도 로그만 있으면 논문용 테이블 수치를 얻을 수 있습니다.

실행: python parse_logs.py --logs model1.log model2.log ...
      python parse_logs.py --logs my_logs/*.txt
"""

import re
import sys
import argparse
import numpy as np
from collections import defaultdict


def parse_val(s):
    """'0.8035±0.0012' → (0.8035, 0.0012), '5.000' → (5.0, 0.0)"""
    s = (s or "").strip()
    if not s:
        return None, None
    if "±" in s:
        parts = s.split("±", 1)
        return float(parts[0]), float(parts[1])
    return float(s), 0.0


def main():
    ap = argparse.ArgumentParser(description="Parse report_three_methods log output from multiple models and aggregate.")
    ap.add_argument("--logs", nargs="+", required=True, help="로그 파일 경로들 (예: logs/*.log)")
    args = ap.parse_args()

    # data[method][p] = { "acc_m": [], "acc_s": [], "rstd_m": [], "rstd_s": [], "cost_m": [], "cost_s": [] }
    data = defaultdict(lambda: defaultdict(lambda: {"acc_m": [], "acc_s": [], "rstd_m": [], "rstd_s": [], "cost_m": [], "cost_s": []}))

    # report_three_methods 출력: "  10% : acc=0.8035±0.0007, recall_std=0.0364±0.0022, cost=1.4013±0.0000"
    line_pattern = re.compile(
        r"^\s*(\d+)%\s*:\s*acc=([\d\.±]+)"
        r"(?:,\s*recall_std=([\d\.±]+))?"
        r"(?:,\s*cost=([\d\.±]+))?\s*$"
    )

    section_pattern = re.compile(r"^---\s*(Cyclic|PriDe|Ours \(Online Sqrt, α=2)\)\s*---")

    parsed_models = 0
    for log_path in args.logs:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[warn] Skip {log_path}: {e}", file=sys.stderr)
            continue

        current_method = None
        for line in lines:
            line = line.rstrip()
            m_section = section_pattern.search(line)
            if m_section:
                current_method = m_section.group(1)
                continue
            if current_method is None:
                continue
            m = line_pattern.search(line)
            if m:
                p_key = int(m.group(1))
                acc_m, acc_s = parse_val(m.group(2))
                rstd_m, rstd_s = parse_val(m.group(3)) if m.group(3) else (None, None)
                cost_m, cost_s = parse_val(m.group(4)) if m.group(4) else (None, None)

                if acc_m is None:
                    continue
                data[current_method][p_key]["acc_m"].append(acc_m)
                data[current_method][p_key]["acc_s"].append(acc_s)
                if rstd_m is not None:
                    data[current_method][p_key]["rstd_m"].append(rstd_m)
                    data[current_method][p_key]["rstd_s"].append(rstd_s)
                if cost_m is not None:
                    data[current_method][p_key]["cost_m"].append(cost_m)
                    data[current_method][p_key]["cost_s"].append(cost_s)

        parsed_models += 1

    if parsed_models == 0:
        print("Error: No log file could be read.", file=sys.stderr)
        return

    print(f"\n==== 15 MODELS AGGREGATED REPORT (parsed from {parsed_models} logs) ====")
    print("목표: 각 모델에서 뽑힌 (평균±표준편차)에서, 15개 모델에 대해 평균들의 평균, 표준편차들의 평균.\n")

    method_order = ["Cyclic", "PriDe", "Ours (Online Sqrt, α=2)"]
    cyclic_fracs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    pride_fracs = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for method in method_order:
        if method not in data or not data[method]:
            continue
        print(f"--- {method} ---")
        fracs = cyclic_fracs if method == "Cyclic" else pride_fracs
        for p in fracs:
            if p not in data[method]:
                continue
            d = data[method][p]
            acc_m_list = d["acc_m"]
            if not acc_m_list:
                continue
            n_count = len(acc_m_list)

            acc_m_avg = float(np.mean(acc_m_list))
            acc_s_avg = float(np.mean(d["acc_s"])) if d["acc_s"] else 0.0
            rstd_m_avg = float(np.mean(d["rstd_m"])) if d["rstd_m"] else None
            rstd_s_avg = float(np.mean(d["rstd_s"])) if d["rstd_s"] else 0.0
            cost_m_avg = float(np.mean(d["cost_m"])) if d["cost_m"] else None
            cost_s_avg = float(np.mean(d["cost_s"])) if d["cost_s"] else 0.0

            acc_str = f"{acc_m_avg:.4f}±{acc_s_avg:.4f}"
            rstr = f", recall_std={rstd_m_avg:.4f}±{rstd_s_avg:.4f}" if rstd_m_avg is not None else ""
            cstr = f", cost={cost_m_avg:.4f}±{cost_s_avg:.4f}" if cost_m_avg is not None else ""

            print(f"  {p}% : acc={acc_str}{rstr}{cstr}")
        print()

    print("=========================================================================\n")


if __name__ == "__main__":
    main()
