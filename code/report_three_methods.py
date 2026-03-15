#!/usr/bin/env python3
"""
Cyclic, PriDe, Ours (Online Sqrt All) 3개 방법에 대해
subject별 curve/pride_curve JSONL에서 구간별 acc/recall_std를 읽어
N개 subject 평균(mean만, run std 없음)으로 리포트 출력.
Cyclic: 10~100%, PriDe/Ours: 2,5,10,...,100%. Ours는 alpha=2 고정.

Usage:
  python report_three_methods.py --results_dir results_mmlu/0s_Model/mmlu_full_id-ABCD
  python report_three_methods.py --results_root . --models M1 M2 ... M15 --eval_name mmlu,0,full --option_id_set ABCD
  (여러 모델 지정 시 15개 모델 등 모든 모델에서 구간별 평균을 내어 한 번에 출력)
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def _compute_results_dir(code_dir: str, eval_name: str, model_path: str, option_id_set: Optional[str]) -> str:
    parts = str(eval_name).strip().split(",")
    task = parts[0].strip()
    num_few = int(parts[1]) if len(parts) > 1 else 0
    setting = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    model_name = str(model_path).split("/")[-1]
    path = f"results_{task}/{num_few}s_{model_name}/{task}"
    if setting:
        path += f"_{setting}"
    if option_id_set:
        path += f"_id-{option_id_set}"
    return os.path.join(code_dir, path)


def _discover_curve_files(results_dir: str) -> Tuple[List[str], List[str]]:
    base = os.path.join(results_dir, "*_curve.jsonl")
    pride = os.path.join(results_dir, "*_pride_curve.jsonl")
    base_files = sorted(glob.glob(base))
    pride_files = sorted(glob.glob(pride))
    return base_files, pride_files


def _subject_from_path(path: str, suffix: str) -> str:
    basename = os.path.basename(path)
    if basename.endswith(suffix):
        return basename[: -len(suffix)]
    return basename


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


# Cyclic: 10~100%
CYCLIC_FRACS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# PriDe & Ours: 2, 5, 10, ..., 100%
PRIDE_OURS_FRACS = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
OURS_ALPHA = 2  # Ours (Online Sqrt)는 alpha=2 고정


def _get_cyclic_at_fracs(lines: List[dict], fracs: List[int]) -> Dict[int, Tuple[float, Optional[float], Optional[float]]]:
    """Baseline curve: any line has cyclic_random_{p}; take first line and extract all p in fracs. Returns (acc, rstd, cost)."""
    out = {}
    for obj in lines:
        for p in fracs:
            if p in out:
                continue
            key = f"cyclic_random_{p}"
            rkey = f"cyclic_random_{p}_recall_std"
            if key not in obj or not isinstance(obj[key], dict):
                continue
            acc_list = obj[key].get("accuracies") or obj[key].get("acc")
            if isinstance(acc_list, list) and acc_list:
                acc = float(acc_list[0])
            elif isinstance(acc_list, (int, float)):
                acc = float(acc_list)
            else:
                continue
            rstd = float(obj[rkey]) if rkey in obj and isinstance(obj[rkey], (int, float)) else None
            cost_list = obj[key].get("costs")
            cost = float(cost_list[0]) if isinstance(cost_list, list) and cost_list and isinstance(cost_list[0], (int, float)) else None
            out[p] = (acc, rstd, cost)
        if len(out) == len(fracs):
            break
    return out


def _get_pride_at_fracs(lines: List[dict], fracs: List[float]) -> Dict[float, Tuple[float, Optional[float], Optional[float]]]:
    """Pride curve: each line has one cyclic_random_{alpha}; collect (p, acc, rstd, cost) for p in fracs."""
    out = {}
    for obj in lines:
        for p in fracs:
            if p in out:
                continue
            keys_try = list({f"cyclic_random_{p}", f"cyclic_random_{p:g}", f"cyclic_random_{int(p)}"})
            for key in keys_try:
                if key not in obj or not isinstance(obj[key], dict):
                    continue
                acc_list = obj[key].get("accuracies") or obj[key].get("acc")
                if isinstance(acc_list, list) and acc_list:
                    acc = float(acc_list[0])
                elif isinstance(acc_list, (int, float)):
                    acc = float(acc_list)
                else:
                    continue
                rkey = f"{key}_recall_std"
                rstd = float(obj.get(rkey)) if isinstance(obj.get(rkey), (int, float)) else None
                cost_list = obj[key].get("costs")
                cost = float(cost_list[0]) if isinstance(cost_list, list) and cost_list and isinstance(cost_list[0], (int, float)) else None
                out[p] = (acc, rstd, cost)
                break
    return out


def _get_ours_online_sqrt_at_fracs(lines: List[dict], fracs: List[float], alpha: float = 2.0) -> Dict[float, Tuple[float, Optional[float], Optional[float]]]:
    """Ours (Online Sqrt): use line that corresponds to alpha=2 (line with cyclic_random_2). Returns (acc, rstd, cost)."""
    alpha_line = None
    for obj in lines:
        key2 = "cyclic_random_2" in obj or "cyclic_random_2.0" in obj
        if key2:
            alpha_line = obj
            break
    if not alpha_line:
        return {}
    out = {}
    for h in alpha_line.get("heuristic_points") or []:
        if not isinstance(h, dict) or str(h.get("label")) != "online_sqrt_all":
            continue
        th1_p = h.get("th1_p")
        if th1_p is None:
            continue
        p = float(th1_p) if th1_p != int(th1_p) else int(th1_p)
        if p not in fracs:
            continue
        acc = h.get("acc")
        if acc is None:
            continue
        rstd = h.get("recall_std")
        cost = h.get("cost")
        out[p] = (float(acc), float(rstd) if rstd is not None else None, float(cost) if cost is not None else None)
    return out


def run(
    results_dir: str,
    max_subjects: Optional[int] = None,
) -> Tuple[Dict[str, Dict], int]:
    """Returns ({"Cyclic": {p: (acc_mean, rstd_mean, cost_mean)}, ...}, n_subjects)."""
    base_files, pride_files = _discover_curve_files(results_dir)
    if not base_files:
        return {}, 0

    subjects = sorted({_subject_from_path(p, "_curve.jsonl") for p in base_files})
    if max_subjects is not None:
        subjects = subjects[: max_subjects]

    base_by_subj = {_subject_from_path(p, "_curve.jsonl"): p for p in base_files}
    pride_by_subj = {_subject_from_path(p, "_pride_curve.jsonl"): p for p in pride_files}

    # per-subject per-frac lists: { p: [(acc, rstd, cost), ...] }
    cyc_by_p: Dict[int, List[Tuple[float, Optional[float], Optional[float]]]] = {p: [] for p in CYCLIC_FRACS}
    pride_by_p: Dict[float, List[Tuple[float, Optional[float], Optional[float]]]] = {p: [] for p in PRIDE_OURS_FRACS}
    ours_by_p: Dict[float, List[Tuple[float, Optional[float], Optional[float]]]] = {p: [] for p in PRIDE_OURS_FRACS}

    for subj in subjects:
        base_path = base_by_subj.get(subj)
        pride_path = pride_by_subj.get(subj)
        if not base_path:
            continue
        base_lines = _read_jsonl(base_path)
        if not base_lines:
            continue
        cyc_vals = _get_cyclic_at_fracs(base_lines, CYCLIC_FRACS)
        for p, (acc, rstd, cost) in cyc_vals.items():
            cyc_by_p[p].append((acc, rstd, cost))

        if pride_path:
            pride_lines = _read_jsonl(pride_path)
            pride_vals = _get_pride_at_fracs(pride_lines, PRIDE_OURS_FRACS)
            for p, (acc, rstd, cost) in pride_vals.items():
                p_key = int(p) if p == int(p) else p
                if p_key in pride_by_p:
                    pride_by_p[p_key].append((acc, rstd, cost))
            ours_vals = _get_ours_online_sqrt_at_fracs(pride_lines, PRIDE_OURS_FRACS, alpha=float(OURS_ALPHA))
            for p, (acc, rstd, cost) in ours_vals.items():
                p_key = int(p) if p == int(p) else p
                if p_key in ours_by_p:
                    ours_by_p[p_key].append((acc, rstd, cost))

    n_used = max(len(cyc_by_p.get(p, [])) for p in CYCLIC_FRACS) if CYCLIC_FRACS else 0

    def _mean_over_subjects(by_p: Dict) -> Dict:
        result = {}
        for p, lst in by_p.items():
            if not lst:
                continue
            accs = [x[0] for x in lst]
            rstds = [x[1] for x in lst if x[1] is not None]
            costs = [x[2] for x in lst if len(x) > 2 and x[2] is not None]
            result[p] = (
                float(np.mean(accs)),
                float(np.mean(rstds)) if rstds else None,
                float(np.mean(costs)) if costs else None,
            )
        return result

    # 항상 Cyclic, PriDe, Ours 세 섹션 포함 (데이터 없으면 빈 dict)
    out = {
        "Cyclic": _mean_over_subjects(cyc_by_p) if any(cyc_by_p.get(p) for p in CYCLIC_FRACS) else {},
        "PriDe": _mean_over_subjects(pride_by_p) if any(pride_by_p.get(p) for p in PRIDE_OURS_FRACS) else {},
        "Ours (Online Sqrt, α=2)": _mean_over_subjects(ours_by_p) if any(ours_by_p.get(p) for p in PRIDE_OURS_FRACS) else {},
    }
    return out, n_used


def main():
    ap = argparse.ArgumentParser(description="Cyclic, PriDe, Ours (Online Sqrt) 3-method report from curve JSONL (mean over subjects, no run std).")
    ap.add_argument("--results_dir", type=str, default="", help="Directory containing *_curve.jsonl and *_pride_curve.jsonl")
    ap.add_argument("--results_root", type=str, default="", help="Root for results_* (used with --models)")
    ap.add_argument("--models", type=str, nargs="+", default=None, help="Model names; with --eval_name to resolve results_dir")
    ap.add_argument("--eval_name", type=str, default="", help="e.g. mmlu,0,full")
    ap.add_argument("--option_id_set", type=str, default=None, help="e.g. ABCD")
    ap.add_argument("--max_subjects", type=int, default=None, help="Cap number of subjects (e.g. 15)")
    args = ap.parse_args()

    results_dir = str(args.results_dir).strip()
    if results_dir:
        model_list = None
    elif args.models and args.eval_name:
        root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        model_list = [str(m).strip() for m in args.models if str(m).strip()]
        if not model_list:
            model_list = None
    else:
        model_list = None
    if not results_dir and not model_list:
        print("Error: need --results_dir or (--results_root + --models + --eval_name)", file=sys.stderr)
        sys.exit(1)

    if results_dir:
        dirs_to_run = [(results_dir, "results")]
    else:
        root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        dirs_to_run = [
            (_compute_results_dir(root, args.eval_name, m, args.option_id_set), m.split("/")[-1])
            for m in model_list
        ]

    # 모든 모델에서 run() 결과 수집
    reports_per_model: List[Tuple[dict, int, str]] = []
    for res_dir, label in dirs_to_run:
        if not os.path.isdir(res_dir):
            print("[skip] not a directory: {} ({})".format(res_dir, label), file=sys.stderr)
            continue
        report, n_subj = run(res_dir, max_subjects=args.max_subjects)
        if not report or not any(report.values()):
            print("[skip] No data in {} ({})".format(res_dir, label), file=sys.stderr)
            continue
        reports_per_model.append((report, n_subj, label))

    if not reports_per_model:
        print("Error: No data from any directory.", file=sys.stderr)
        sys.exit(1)

    n_models = len(reports_per_model)
    n_subj_avg = int(np.mean([x[1] for x in reports_per_model]))
    # 단일 디렉터리면 그대로 출력; 여러 모델이면 모델별 평균으로 한 번만 출력
    if n_models == 1:
        report, n_subj, label = reports_per_model[0]
        print("==== THREE-METHOD REPORT: {} (mean over {} subjects, no run std) ====".format(label, n_subj))
        print("results_dir:", dirs_to_run[0][0])
    else:
        # 모델별 평균: method별 p별로 (acc, rstd, cost) 리스트의 평균
        method_names = ["Cyclic", "PriDe", "Ours (Online Sqrt, α=2)"]
        all_ps = sorted(set().union(*(set(reports_per_model[i][0].get(m, {}).keys()) for i in range(n_models) for m in method_names)), key=lambda x: (float(x), x))
        report = {}
        for method_name in method_names:
            by_p = {}
            for p in all_ps:
                accs, rstds, costs = [], [], []
                for rep, _, _ in reports_per_model:
                    if method_name not in rep or p not in rep[method_name]:
                        continue
                    t = rep[method_name][p]
                    accs.append(t[0])
                    if t[1] is not None:
                        rstds.append(t[1])
                    if len(t) > 2 and t[2] is not None:
                        costs.append(t[2])
                if not accs:
                    continue
                by_p[p] = (
                    float(np.mean(accs)),
                    float(np.mean(rstds)) if rstds else None,
                    float(np.mean(costs)) if costs else None,
                )
            report[method_name] = by_p
        print("==== THREE-METHOD REPORT: mean over {} models (avg {} subjects per model) ====".format(n_models, n_subj_avg))
        print("results_root:", str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__)))
        print("eval_name:", args.eval_name, "| option_id_set:", args.option_id_set)
        print("models (n={}):".format(n_models), ", ".join(x[2] for x in reports_per_model[:5]) + (" ..." if n_models > 5 else ""))

    for method_name, by_p in report.items():
        print("\n--- {} ---".format(method_name))
        if not by_p:
            print("  (no data — need *_pride_curve.jsonl from eval_clm with --pride_mix)" if "PriDe" in method_name or "Ours" in method_name else "  (no data)")
        else:
            for p in sorted(by_p.keys(), key=lambda x: (float(x), x)):
                acc, rstd, cost = by_p[p][0], by_p[p][1], by_p[p][2] if len(by_p[p]) > 2 else None
                rstr = f", recall_std={rstd:.4f}" if rstd is not None else ""
                cstr = f", cost={cost:.4f}" if cost is not None else ""
                print("  {}% : acc={:.4f}{}{}".format(p, acc, rstr, cstr))
    print("\n======================================================")


if __name__ == "__main__":
    main()
