#!/usr/bin/env python3
"""
Cyclic, PriDe, Ours (Online Sqrt All) 3개 방법에 대해
run0, run1, run2 등 여러 Run 간의 평균(mean)과 표준편차(std)를 계산하는 리포트 스크립트.
여러 모델(--models)을 입력하면 15개 모델의 평균을 낸 뒤, Run 간의 표준편차를 보여줍니다.
"""

import argparse
import glob
import json
import os
import re
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


def _get_mmlu_subject_set() -> Optional[set]:
    try:
        from mmlu_categories import subcategories  # type: ignore
        return set(subcategories.keys())
    except Exception:
        return None


def _discover_curve_files(results_dir: str) -> Tuple[List[str], List[str]]:
    base = os.path.join(results_dir, "*_curve.jsonl")
    pride = os.path.join(results_dir, "*_pride_curve.jsonl")
    return sorted(glob.glob(base)), sorted(glob.glob(pride))


def _parse_subject_and_run(filename: str, suffix: str) -> Tuple[str, int]:
    """파일명에서 subject 이름과 run 번호를 분리합니다 (예: csqa_run2 -> 'csqa', 2)"""
    stem = filename
    if filename.endswith(suffix):
        stem = filename[:-len(suffix)]
    m = re.search(r'_run(\d+)$', stem)
    if m:
        return stem[:m.start()], int(m.group(1))
    return stem, 0


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


CYCLIC_FRACS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PRIDE_OURS_FRACS = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
OURS_ALPHA = 2


def _get_cyclic_at_fracs(lines: List[dict], fracs: List[int]) -> Dict[int, Tuple[float, Optional[float], Optional[float]]]:
    out = {}
    for obj in lines:
        for p in fracs:
            if p in out: continue
            key = f"cyclic_random_{p}"
            rkey = f"{key}_recall_std"
            if key not in obj or not isinstance(obj[key], dict): continue
            acc_list = obj[key].get("accuracies") or obj[key].get("acc")
            if isinstance(acc_list, list) and acc_list: acc = float(acc_list[0])
            elif isinstance(acc_list, (int, float)): acc = float(acc_list)
            else: continue
            rstd = float(obj[rkey]) if rkey in obj and isinstance(obj[rkey], (int, float)) else None
            cost_list = obj[key].get("costs")
            cost = float(cost_list[0]) if isinstance(cost_list, list) and cost_list and isinstance(cost_list[0], (int, float)) else None
            out[p] = (acc, rstd, cost)
        if len(out) == len(fracs): break
    return out


def _get_pride_at_fracs(lines: List[dict], fracs: List[float]) -> Dict[float, Tuple[float, Optional[float], Optional[float]]]:
    out = {}
    for obj in lines:
        for p in fracs:
            if p in out: continue
            keys_try = list({f"cyclic_random_{p}", f"cyclic_random_{p:g}", f"cyclic_random_{int(p)}"})
            for key in keys_try:
                if key not in obj or not isinstance(obj[key], dict): continue
                acc_list = obj[key].get("accuracies") or obj[key].get("acc")
                if isinstance(acc_list, list) and acc_list: acc = float(acc_list[0])
                elif isinstance(acc_list, (int, float)): acc = float(acc_list)
                else: continue
                rkey = f"{key}_recall_std"
                rstd = float(obj.get(rkey)) if isinstance(obj.get(rkey), (int, float)) else None
                cost_list = obj[key].get("costs")
                cost = float(cost_list[0]) if isinstance(cost_list, list) and cost_list and isinstance(cost_list[0], (int, float)) else None
                out[p] = (acc, rstd, cost)
                break
    return out


def _get_ours_online_sqrt_at_fracs(lines: List[dict], fracs: List[float], alpha: float) -> Dict[float, Tuple[float, Optional[float], Optional[float]]]:
    alpha_line = None
    for obj in lines:
        if "cyclic_random_2" in obj or "cyclic_random_2.0" in obj:
            alpha_line = obj
            break
    if not alpha_line: return {}
    out = {}
    for h in alpha_line.get("heuristic_points") or []:
        if not isinstance(h, dict) or str(h.get("label")) != "online_sqrt_all": continue
        th1_p = h.get("th1_p")
        if th1_p is None: continue
        p = float(th1_p) if th1_p != int(th1_p) else int(th1_p)
        if p not in fracs: continue
        acc = h.get("acc")
        if acc is None: continue
        rstd = h.get("recall_std")
        cost = h.get("cost")
        out[p] = (float(acc), float(rstd) if rstd is not None else None, float(cost) if cost is not None else None)
    return out


def process_model(results_dir: str, max_subjects: Optional[int]) -> Dict[int, Dict[str, Dict[float, Tuple[float, float, float]]]]:
    base_files, pride_files = _discover_curve_files(results_dir)
    
    # Run 번호(run_idx) 기준으로 파일 분류
    runs = {}
    for f in base_files:
        subj, r_idx = _parse_subject_and_run(os.path.basename(f), "_curve.jsonl")
        runs.setdefault(r_idx, {}).setdefault("base", []).append((subj, f))
    for f in pride_files:
        subj, r_idx = _parse_subject_and_run(os.path.basename(f), "_pride_curve.jsonl")
        runs.setdefault(r_idx, {}).setdefault("pride", []).append((subj, f))

    mmlu_set = _get_mmlu_subject_set() if "mmlu" in results_dir.lower() else None
    model_run_data = {}

    for r_idx, files_dict in runs.items():
        base_list = files_dict.get("base", [])
        pride_list = files_dict.get("pride", [])

        def _filter_subj(lst):
            valid = [(s, f) for s, f in lst if mmlu_set is None or s in mmlu_set]
            valid.sort(key=lambda x: x[0])
            return valid[:max_subjects] if max_subjects else valid

        base_list = _filter_subj(base_list)
        pride_list = _filter_subj(pride_list)

        cyc_by_p = {p: [] for p in CYCLIC_FRACS}
        pride_by_p = {p: [] for p in PRIDE_OURS_FRACS}
        ours_by_p = {p: [] for p in PRIDE_OURS_FRACS}

        for subj, f in base_list:
            cyc_vals = _get_cyclic_at_fracs(_read_jsonl(f), CYCLIC_FRACS)
            for p, v in cyc_vals.items(): cyc_by_p[p].append(v)
        for subj, f in pride_list:
            lines = _read_jsonl(f)
            pv = _get_pride_at_fracs(lines, PRIDE_OURS_FRACS)
            ov = _get_ours_online_sqrt_at_fracs(lines, PRIDE_OURS_FRACS, OURS_ALPHA)
            for p, v in pv.items(): pride_by_p[p].append(v)
            for p, v in ov.items(): ours_by_p[p].append(v)

        def _mean_tuples(lst):
            if not lst: return None
            accs = [x[0] for x in lst if x[0] is not None]
            rstds = [x[1] for x in lst if x[1] is not None]
            costs = [x[2] for x in lst if x[2] is not None]
            if not accs: return None
            return (
                float(np.mean(accs)),
                float(np.mean(rstds)) if rstds else None,
                float(np.mean(costs)) if costs else None
            )

        run_res = {"Cyclic": {}, "PriDe": {}, "Ours (Online Sqrt, α=2)": {}}
        for p in CYCLIC_FRACS:
            m = _mean_tuples(cyc_by_p[p])
            if m: run_res["Cyclic"][p] = m
        for p in PRIDE_OURS_FRACS:
            m = _mean_tuples(pride_by_p[p])
            if m: run_res["PriDe"][p] = m
            m = _mean_tuples(ours_by_p[p])
            if m: run_res["Ours (Online Sqrt, α=2)"][p] = m

        model_run_data[r_idx] = run_res

    return model_run_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="")
    ap.add_argument("--results_root", type=str, default="")
    ap.add_argument("--models", type=str, nargs="+", default=None)
    ap.add_argument("--eval_name", type=str, default="")
    ap.add_argument("--option_id_set", type=str, default=None)
    ap.add_argument("--max_subjects", type=int, default=None)
    args = ap.parse_args()

    results_dir = str(args.results_dir).strip()
    if results_dir:
        dirs_to_run = [(results_dir, os.path.basename(results_dir))]
    else:
        root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        model_list = [str(m).strip() for m in args.models if str(m).strip()]
        dirs_to_run = [(_compute_results_dir(root, args.eval_name, m, args.option_id_set), m.split("/")[-1]) for m in model_list]

    all_models_data = []
    for res_dir, label in dirs_to_run:
        if not os.path.isdir(res_dir):
            continue
        m_data = process_model(res_dir, args.max_subjects)
        if m_data:
            all_models_data.append(m_data)

    if not all_models_data:
        print("Error: No data from any directory.", file=sys.stderr)
        sys.exit(1)

    # 1. 모든 Run 인덱스 추출 (보통 0, 1, 2)
    all_run_idxs = set()
    for m_data in all_models_data:
        all_run_idxs.update(m_data.keys())

    # 2. 모델들 간의 평균 내기 (특정 Run 기준)
    ensemble_run_data = {}
    for r_idx in all_run_idxs:
        ensemble_run_data[r_idx] = {"Cyclic": {}, "PriDe": {}, "Ours (Online Sqrt, α=2)": {}}
        for method in ["Cyclic", "PriDe", "Ours (Online Sqrt, α=2)"]:
            fracs = CYCLIC_FRACS if method == "Cyclic" else PRIDE_OURS_FRACS
            for p in fracs:
                vals = []
                for m_data in all_models_data:
                    if r_idx in m_data and method in m_data[r_idx] and p in m_data[r_idx][method]:
                        vals.append(m_data[r_idx][method][p])
                if vals:
                    accs = [x[0] for x in vals if x[0] is not None]
                    rstds = [x[1] for x in vals if x[1] is not None]
                    costs = [x[2] for x in vals if x[2] is not None]
                    ensemble_run_data[r_idx][method][p] = (
                        float(np.mean(accs)) if accs else None,
                        float(np.mean(rstds)) if rstds else None,
                        float(np.mean(costs)) if costs else None
                    )

    # 3. Run들 간의 평균 및 표준편차 구하기!
    final_report = {"Cyclic": {}, "PriDe": {}, "Ours (Online Sqrt, α=2)": {}}
    for method in ["Cyclic", "PriDe", "Ours (Online Sqrt, α=2)"]:
        fracs = CYCLIC_FRACS if method == "Cyclic" else PRIDE_OURS_FRACS
        for p in fracs:
            vals = []
            for r_idx in all_run_idxs:
                if method in ensemble_run_data[r_idx] and p in ensemble_run_data[r_idx][method]:
                    vals.append(ensemble_run_data[r_idx][method][p])
            if vals:
                accs = [x[0] for x in vals if x[0] is not None]
                rstds = [x[1] for x in vals if x[1] is not None]
                costs = [x[2] for x in vals if x[2] is not None]

                final_report[method][p] = (
                    float(np.mean(accs)) if accs else None,
                    float(np.std(accs)) if len(accs) > 1 else None,  # std across runs!
                    float(np.mean(rstds)) if rstds else None,
                    float(np.std(rstds)) if len(rstds) > 1 else None,
                    float(np.mean(costs)) if costs else None,
                    float(np.std(costs)) if len(costs) > 1 else None,
                    len(accs) # number of runs
                )

    # 결과 출력
    print("\n==== THREE-METHOD REPORT ====")
    if len(all_models_data) > 1:
        print(f"Aggregated average across {len(all_models_data)} models")
    print(f"Variance (±) is computed across {len(all_run_idxs)} distinct runs (e.g. run0, run1, run2)")
    
    for method, by_p in final_report.items():
        print(f"\n--- {method} ---")
        if not by_p:
            print("  (no data)")
            continue
        for p in sorted(by_p.keys(), key=lambda x: (float(x), x)):
            t = by_p[p]
            acc_m, acc_s, rstd_m, rstd_s, cost_m, cost_s, n_runs = t

            acc_str = f"{acc_m:.4f}±{acc_s:.4f}" if acc_s is not None else f"{acc_m:.4f} (n={n_runs})"
            rstr = f", recall_std={rstd_m:.4f}±{rstd_s:.4f}" if rstd_s is not None else (f", recall_std={rstd_m:.4f}" if rstd_m is not None else "")
            cstr = f", cost={cost_m:.4f}±{cost_s:.4f}" if cost_s is not None else (f", cost={cost_m:.4f}" if cost_m is not None else "")

            print(f"  {p}% : acc={acc_str}{rstr}{cstr}")
    print("\n======================================================")


if __name__ == "__main__":
    main()