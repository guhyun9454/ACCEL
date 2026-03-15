#!/usr/bin/env python3
"""
Cyclic, PriDe, Ours (Online Sqrt All) 3개 방법에 대해
subject별 curve/pride_curve JSONL에서 100% 지점을 읽어
N개 subject 평균(mean만, run std 없음)으로 리포트 출력.

Usage:
  python report_three_methods.py --results_dir results_mmlu/0s_Model/mmlu_full_id-ABCD
  python report_three_methods.py --results_root . --models ModelName --eval_name mmlu,0,full --option_id_set ABCD
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


def _get_cyclic_100(lines: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    key = "cyclic_random_100"
    rkey = "cyclic_random_100_recall_std"
    accs, rstds = [], []
    for obj in lines:
        if key in obj and isinstance(obj[key], dict):
            acc_list = obj[key].get("accuracies") or obj[key].get("acc")
            if isinstance(acc_list, list) and acc_list:
                accs.append(float(acc_list[0]))
            elif isinstance(acc_list, (int, float)):
                accs.append(float(acc_list))
        if rkey in obj and isinstance(obj[rkey], (int, float)):
            rstds.append(float(obj[rkey]))
    if not accs:
        return None, None
    acc_mean = float(np.mean(accs))
    rstd_mean = float(np.mean(rstds)) if rstds else None
    return acc_mean, rstd_mean


def _get_pride_cyclic_100(lines: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """PriDe (Default+PRIDE) at 100%: line that has cyclic_random_100 (alpha=100)."""
    key = "cyclic_random_100"
    rkey = "cyclic_random_100_recall_std"
    for obj in lines:
        if key in obj and isinstance(obj[key], dict):
            acc_list = obj[key].get("accuracies") or obj[key].get("acc")
            if isinstance(acc_list, list) and acc_list:
                acc = float(acc_list[0])
            elif isinstance(acc_list, (int, float)):
                acc = float(acc_list)
            else:
                continue
            rstd = float(obj[rkey]) if rkey in obj and isinstance(obj[rkey], (int, float)) else None
            return acc, rstd
    return None, None


def _get_ours_online_sqrt_100(lines: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """Ours (Online Sqrt All) at th1_p=100 from heuristic_points (use first line that has it, e.g. alpha=2)."""
    for obj in lines:
        for h in obj.get("heuristic_points") or []:
            if not isinstance(h, dict):
                continue
            if str(h.get("label")) != "online_sqrt_all":
                continue
            if h.get("th1_p") != 100 and h.get("th1_p") != 100.0:
                continue
            acc = h.get("acc")
            rstd = h.get("recall_std")
            if acc is not None:
                return float(acc), float(rstd) if rstd is not None else None
    return None, None


def run(
    results_dir: str,
    max_subjects: Optional[int] = None,
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], int]:
    base_files, pride_files = _discover_curve_files(results_dir)
    if not base_files:
        return {}, 0

    subjects = sorted({_subject_from_path(p, "_curve.jsonl") for p in base_files})
    if max_subjects is not None:
        subjects = subjects[: max_subjects]

    base_by_subj = {_subject_from_path(p, "_curve.jsonl"): p for p in base_files}
    pride_by_subj = {_subject_from_path(p, "_pride_curve.jsonl"): p for p in pride_files}

    cyc_accs, cyc_rstds = [], []
    pride_accs, pride_rstds = [], []
    ours_accs, ours_rstds = [], []

    for subj in subjects:
        base_path = base_by_subj.get(subj)
        pride_path = pride_by_subj.get(subj)
        if not base_path:
            continue
        base_lines = _read_jsonl(base_path)
        if not base_lines:
            continue
        acc_c, rstd_c = _get_cyclic_100(base_lines)
        if acc_c is not None:
            cyc_accs.append(acc_c)
            if rstd_c is not None:
                cyc_rstds.append(rstd_c)

        if pride_path:
            pride_lines = _read_jsonl(pride_path)
            acc_p, rstd_p = _get_pride_cyclic_100(pride_lines)
            if acc_p is not None:
                pride_accs.append(acc_p)
                if rstd_p is not None:
                    pride_rstds.append(rstd_p)
            acc_o, rstd_o = _get_ours_online_sqrt_100(pride_lines)
            if acc_o is not None:
                ours_accs.append(acc_o)
                if rstd_o is not None:
                    ours_rstds.append(rstd_o)

    n_used = len(cyc_accs)
    out = {}
    if cyc_accs:
        out["Cyclic (100%)"] = (float(np.mean(cyc_accs)), float(np.mean(cyc_rstds)) if cyc_rstds else None)
    if pride_accs:
        out["PriDe (100%)"] = (float(np.mean(pride_accs)), float(np.mean(pride_rstds)) if pride_rstds else None)
    if ours_accs:
        out["Ours (Online Sqrt 100%)"] = (float(np.mean(ours_accs)), float(np.mean(ours_rstds)) if ours_rstds else None)
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
    if not results_dir and args.models and args.eval_name:
        root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        # use first model for single report
        model = args.models[0]
        results_dir = _compute_results_dir(root, args.eval_name, model, args.option_id_set)
    if not results_dir or not os.path.isdir(results_dir):
        print("Error: need --results_dir or (--results_root + --models + --eval_name)", file=sys.stderr)
        sys.exit(1)

    report, n_subj = run(results_dir, max_subjects=args.max_subjects)
    if not report:
        print("No data found in", results_dir)
        sys.exit(0)

    print("==== THREE-METHOD REPORT (mean over {} subjects, no run std) ====".format(n_subj))
    print("results_dir:", results_dir)
    for name, (acc, rstd) in report.items():
        rstr = f", recall_std={rstd:.4f}" if rstd is not None else ""
        print("{} : acc={:.4f}{}".format(name, acc, rstr))
    print("======================================================")


if __name__ == "__main__":
    main()
