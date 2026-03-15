#!/usr/bin/env python3
"""
Cyclic, PriDe, Ours (Online Sqrt All) 3개 방법에 대해
subject별 curve/pride_curve JSONL에서 구간별 acc/recall_std를 읽어
N개 subject 평균(mean만, run std 없음)으로 리포트 출력.
Cyclic: 10~100%, PriDe/Ours: 2,5,10,...,100%. Ours는 alpha=2 고정.

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


# Cyclic: 10~100%
CYCLIC_FRACS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# PriDe & Ours: 2, 5, 10, ..., 100%
PRIDE_OURS_FRACS = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
OURS_ALPHA = 2  # Ours (Online Sqrt)는 alpha=2 고정


def _get_cyclic_at_fracs(lines: List[dict], fracs: List[int]) -> Dict[int, Tuple[float, Optional[float]]]:
    """Baseline curve: any line has cyclic_random_{p}; take first line and extract all p in fracs."""
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
            out[p] = (acc, rstd)
        if len(out) == len(fracs):
            break
    return out


def _get_pride_at_fracs(lines: List[dict], fracs: List[float]) -> Dict[float, Tuple[float, Optional[float]]]:
    """Pride curve: each line has one cyclic_random_{alpha}; collect (p, acc, rstd) for p in fracs."""
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
                out[p] = (acc, rstd)
                break
    return out


def _get_ours_online_sqrt_at_fracs(lines: List[dict], fracs: List[float], alpha: float = 2.0) -> Dict[float, Tuple[float, Optional[float]]]:
    """Ours (Online Sqrt): use line that corresponds to alpha=2 (line with cyclic_random_2)."""
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
        out[p] = (float(acc), float(rstd) if rstd is not None else None)
    return out


def run(
    results_dir: str,
    max_subjects: Optional[int] = None,
) -> Tuple[Dict[str, Dict], int]:
    """Returns ({"Cyclic": {p: (acc_mean, rstd_mean), ...}, "PriDe": {...}, "Ours (Online Sqrt, α=2)": {...}}, n_subjects)."""
    base_files, pride_files = _discover_curve_files(results_dir)
    if not base_files:
        return {}, 0

    subjects = sorted({_subject_from_path(p, "_curve.jsonl") for p in base_files})
    if max_subjects is not None:
        subjects = subjects[: max_subjects]

    base_by_subj = {_subject_from_path(p, "_curve.jsonl"): p for p in base_files}
    pride_by_subj = {_subject_from_path(p, "_pride_curve.jsonl"): p for p in pride_files}

    # per-subject per-frac lists: { p: [(acc, rstd), ...] }
    cyc_by_p: Dict[int, List[Tuple[float, Optional[float]]]] = {p: [] for p in CYCLIC_FRACS}
    pride_by_p: Dict[float, List[Tuple[float, Optional[float]]]] = {p: [] for p in PRIDE_OURS_FRACS}
    ours_by_p: Dict[float, List[Tuple[float, Optional[float]]]] = {p: [] for p in PRIDE_OURS_FRACS}

    for subj in subjects:
        base_path = base_by_subj.get(subj)
        pride_path = pride_by_subj.get(subj)
        if not base_path:
            continue
        base_lines = _read_jsonl(base_path)
        if not base_lines:
            continue
        cyc_vals = _get_cyclic_at_fracs(base_lines, CYCLIC_FRACS)
        for p, (acc, rstd) in cyc_vals.items():
            cyc_by_p[p].append((acc, rstd))

        if pride_path:
            pride_lines = _read_jsonl(pride_path)
            pride_vals = _get_pride_at_fracs(pride_lines, PRIDE_OURS_FRACS)
            for p, (acc, rstd) in pride_vals.items():
                p_key = int(p) if p == int(p) else p
                if p_key in pride_by_p:
                    pride_by_p[p_key].append((acc, rstd))
            ours_vals = _get_ours_online_sqrt_at_fracs(pride_lines, PRIDE_OURS_FRACS, alpha=float(OURS_ALPHA))
            for p, (acc, rstd) in ours_vals.items():
                p_key = int(p) if p == int(p) else p
                if p_key in ours_by_p:
                    ours_by_p[p_key].append((acc, rstd))

    n_used = max(len(cyc_by_p.get(p, [])) for p in CYCLIC_FRACS) if CYCLIC_FRACS else 0
    out = {}

    def _mean_over_subjects(by_p: Dict) -> Dict:
        result = {}
        for p, lst in by_p.items():
            if not lst:
                continue
            accs = [x[0] for x in lst]
            rstds = [x[1] for x in lst if x[1] is not None]
            result[p] = (float(np.mean(accs)), float(np.mean(rstds)) if rstds else None)
        return result

    if any(cyc_by_p[p] for p in CYCLIC_FRACS):
        out["Cyclic"] = _mean_over_subjects(cyc_by_p)
    if any(pride_by_p[p] for p in PRIDE_OURS_FRACS):
        out["PriDe"] = _mean_over_subjects(pride_by_p)
    if any(ours_by_p[p] for p in PRIDE_OURS_FRACS):
        out["Ours (Online Sqrt, α=2)"] = _mean_over_subjects(ours_by_p)

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
    for method_name, by_p in report.items():
        print("\n--- {} ---".format(method_name))
        for p in sorted(by_p.keys(), key=lambda x: (float(x), x)):
            acc, rstd = by_p[p]
            rstr = f", recall_std={rstd:.4f}" if rstd is not None else ""
            print("  {}% : acc={:.4f}{}".format(p, acc, rstr))
    print("\n======================================================")


if __name__ == "__main__":
    main()
