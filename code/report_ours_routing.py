#!/usr/bin/env python3
"""
Ours (th1/2) 정책만 대상: 구간별(2,5,10,20,...,100%) routing 비율과 T→F, F→T, F→F, T→T 비율 리포트.
*_curve.jsonl (baseline)에서 heuristic_points(label=th1/2)와 transition을 읽어 subject 평균으로 출력.

Usage:
  python report_ours_routing.py --results_dir results_mmlu/0s_Model/mmlu_full_id-ABCD
  python report_ours_routing.py --results_root . --models ModelName --eval_name mmlu,0,full --option_id_set ABCD
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

OURS_PERCENTS = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


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


def _discover_curve_files(results_dir: str) -> List[str]:
    base = os.path.join(results_dir, "*_curve.jsonl")
    return sorted(glob.glob(base))


def _subject_from_path(path: str, suffix: str = "_curve.jsonl") -> str:
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


def _get_ours_th12_per_line(obj: dict) -> Optional[Tuple[float, int, int, int, Optional[dict]]]:
    """한 줄(curve obj)에서 label=th1/2인 heuristic_point와 transition 추출. (p, n_base, n_probe2, n_cyclic, transition)."""
    for h in obj.get("heuristic_points") or []:
        if not isinstance(h, dict) or str(h.get("label")) != "th1/2":
            continue
        th1_p = h.get("th1_p")
        if th1_p is None:
            continue
        p = float(th1_p) if th1_p != int(th1_p) else int(th1_p)
        n_base = int(h.get("n_base", 0))
        n_probe2 = int(h.get("n_probe2", 0))
        n_cyclic = int(h.get("n_cyclic", 0))
        trans = obj.get("transition") if isinstance(obj.get("transition"), dict) else None
        return (p, n_base, n_probe2, n_cyclic, trans)
    return None


def run(
    results_dir: str,
    max_subjects: Optional[int] = None,
) -> Tuple[Dict[float, Dict], int]:
    """
    Returns (by_p, n_subjects).
    by_p[p] = {
      "routing": {"base_ratio": float, "probe2_ratio": float, "cyclic_ratio": float},
      "transition": {"t_to_t_ratio": float, "t_to_f_ratio": float, "f_to_t_ratio": float, "f_to_f_ratio": float} or None
    }
    """
    base_files = _discover_curve_files(results_dir)
    if not base_files:
        return {}, 0

    subjects = sorted({_subject_from_path(p, "_curve.jsonl") for p in base_files})
    if max_subjects is not None:
        subjects = subjects[: max_subjects]

    base_by_subj = {_subject_from_path(p, "_curve.jsonl"): p for p in base_files}

    # per p: list of (base_ratio, probe2_ratio, cyclic_ratio) per subject
    routing_by_p: Dict[float, List[Tuple[float, float, float]]] = {p: [] for p in OURS_PERCENTS}
    # per p: list of (t_to_t, t_to_f, f_to_t, f_to_f) counts per subject (then we'll normalize)
    transition_by_p: Dict[float, List[Tuple[int, int, int, int]]] = {p: [] for p in OURS_PERCENTS}

    for subj in subjects:
        path = base_by_subj.get(subj)
        if not path:
            continue
        lines = _read_jsonl(path)
        for obj in lines:
            row = _get_ours_th12_per_line(obj)
            if row is None:
                continue
            p, n_base, n_probe2, n_cyclic, trans = row
            if p not in routing_by_p:
                continue
            N = n_base + n_probe2 + n_cyclic
            if N > 0:
                routing_by_p[p].append((n_base / N, n_probe2 / N, n_cyclic / N))
            if trans is not None:
                bt = int(trans.get("base_t_count", 0))
                bf = int(trans.get("base_f_count", 0))
                t_to_f = int(trans.get("t_to_f_count", 0))
                f_to_t = int(trans.get("f_to_t_count", 0))
                t_to_t = bt - t_to_f
                f_to_f = bf - f_to_t
                transition_by_p[p].append((t_to_t, t_to_f, f_to_t, f_to_f))

    n_used = max(len(routing_by_p.get(p, [])) for p in OURS_PERCENTS) if OURS_PERCENTS else 0

    out = {}
    for p in OURS_PERCENTS:
        rlist = routing_by_p.get(p, [])
        if not rlist:
            out[p] = {"routing": None, "transition": None}
            continue
        base_ratios = [x[0] for x in rlist]
        probe2_ratios = [x[1] for x in rlist]
        cyclic_ratios = [x[2] for x in rlist]
        routing = {
            "base_ratio": float(np.mean(base_ratios)),
            "probe2_ratio": float(np.mean(probe2_ratios)),
            "cyclic_ratio": float(np.mean(cyclic_ratios)),
        }
        tlist = transition_by_p.get(p, [])
        if not tlist:
            out[p] = {"routing": routing, "transition": None}
            continue
        # 비율: 전체 샘플 대비 T→T, T→F, F→T, F→F (subject별 비율의 평균)
        trans_ratios = []
        for (t_to_t, t_to_f, f_to_t, f_to_f) in tlist:
            total = t_to_t + t_to_f + f_to_t + f_to_f
            if total > 0:
                trans_ratios.append((
                    t_to_t / total, t_to_f / total, f_to_t / total, f_to_f / total
                ))
        if trans_ratios:
            transition = {
                "t_to_t_ratio": float(np.mean([x[0] for x in trans_ratios])),
                "t_to_f_ratio": float(np.mean([x[1] for x in trans_ratios])),
                "f_to_t_ratio": float(np.mean([x[2] for x in trans_ratios])),
                "f_to_f_ratio": float(np.mean([x[3] for x in trans_ratios])),
            }
        else:
            transition = None
        out[p] = {"routing": routing, "transition": transition}
    return out, n_used


def main():
    ap = argparse.ArgumentParser(description="Ours (th1/2) routing & T→F/F→T/F→F/T→T report from baseline curve JSONL.")
    ap.add_argument("--results_dir", type=str, default="", help="Directory containing *_curve.jsonl")
    ap.add_argument("--results_root", type=str, default="", help="Root for results_* (used with --models)")
    ap.add_argument("--models", type=str, nargs="+", default=None, help="Model names")
    ap.add_argument("--eval_name", type=str, default="", help="e.g. mmlu,0,full")
    ap.add_argument("--option_id_set", type=str, default=None, help="e.g. ABCD")
    ap.add_argument("--max_subjects", type=int, default=None, help="Cap number of subjects")
    args = ap.parse_args()

    results_dir = str(args.results_dir).strip()
    if results_dir:
        dirs_to_run = [(results_dir, "results")]
    elif args.models and args.eval_name:
        root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        model_list = [str(m).strip() for m in args.models if str(m).strip()]
        if not model_list:
            print("Error: --models required with --eval_name", file=sys.stderr)
            sys.exit(1)
        dirs_to_run = [
            (_compute_results_dir(root, args.eval_name, m, args.option_id_set), m.split("/")[-1])
            for m in model_list
        ]
    else:
        print("Error: need --results_dir or (--results_root + --models + --eval_name)", file=sys.stderr)
        sys.exit(1)

    for res_dir, label in dirs_to_run:
        if not os.path.isdir(res_dir):
            print("[skip] not a directory: {} ({})".format(res_dir, label))
            continue
        by_p, n_subj = run(res_dir, max_subjects=args.max_subjects)
        if not by_p:
            print("[skip] No data in {} ({})".format(res_dir, label))
            continue
        print("==== OURS (th1/2) ROUTING & TRANSITION REPORT: {} (mean over {} subjects) ====".format(label, n_subj))
        print("results_dir:", res_dir)
        print("\n--- Routing 비율 (base / probe2 / cyclic) ---")
        for p in OURS_PERCENTS:
            d = by_p.get(p, {})
            r = d.get("routing")
            if r is None:
                print("  {}% : (no data)".format(p))
            else:
                print("  {}% : base={:.4f}, probe2={:.4f}, cyclic={:.4f}".format(
                    p, r["base_ratio"], r["probe2_ratio"], r["cyclic_ratio"]))
        print("\n--- Transition 비율 (T→T, T→F, F→T, F→F) ---")
        for p in OURS_PERCENTS:
            d = by_p.get(p, {})
            t = d.get("transition")
            if t is None:
                print("  {}% : (no data)".format(p))
            else:
                print("  {}% : T→T={:.4f}, T→F={:.4f}, F→T={:.4f}, F→F={:.4f}".format(
                    p, t["t_to_t_ratio"], t["t_to_f_ratio"], t["f_to_t_ratio"], t["f_to_f_ratio"]))
        print("\n======================================================")


if __name__ == "__main__":
    main()
