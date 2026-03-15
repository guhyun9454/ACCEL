#!/usr/bin/env python3
"""
Cyclic, PriDe, Ours (Online Sqrt All) 3개 방법에 대해
run0, run1, run2 등 여러 Run 간의 평균(mean)과 표준편차(std)를 계산하는 리포트 스크립트.

사용 예:
  --models M1 ... M15 --results_root /path/to/results   (모델명으로 run* JSONL 디렉터리 자동 탐색)
  --models M1 ... M15 --eval_name mmlu,0,full [--option_id_set ABCD]   (경로 규칙으로 계산)
  --results_dirs dir1 dir2 ...   또는  --results_dir 단일경로
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
    # 저장 폴더 컨벤션: 0s_<model>, 5s_<model>, ...
    # (만약 과거 실험이 0_shot_<model> 형태로 저장되어 있어도, 자동 탐색 fallback에서 잡히도록 유지)
    few_dir = f"{num_few}s_{model_name}"
    path = f"results_{task}/{few_dir}/{task}"
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
    pride_files = sorted(glob.glob(pride))
    base_files = sorted(glob.glob(base))
    # 중요: "*_curve.jsonl"은 "*_pride_curve.jsonl"도 매칭하므로 제외해야 run 파싱이 망가지지 않음
    base_files = [p for p in base_files if not p.endswith("_pride_curve.jsonl")]
    return base_files, pride_files


def _has_run_curve_files(results_dir: str) -> bool:
    """run0/run1/run2 형태의 curve 파일이 실제로 존재하는지 체크."""
    if not results_dir or not os.path.isdir(results_dir):
        return False
    return bool(glob.glob(os.path.join(results_dir, "*_run*_curve.jsonl"))) or bool(
        glob.glob(os.path.join(results_dir, "*_run*_pride_curve.jsonl"))
    )


def _parse_eval_name(eval_name: str) -> Tuple[str, int, Optional[str]]:
    parts = str(eval_name).strip().split(",")
    task = parts[0].strip()
    num_few = int(parts[1]) if len(parts) > 1 and str(parts[1]).strip() != "" else 0
    setting = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    return task, num_few, setting


def _find_results_dir_for_model_and_eval(
    results_root: str,
    model_name: str,
    eval_name: str,
    option_id_set: Optional[str],
) -> Optional[str]:
    """results_root 아래에서 (model, task, setting, option_id_set)까지 맞는 run-curve 디렉터리를 탐색."""
    model_key = model_name.split("/")[-1].strip()
    root = os.path.abspath(results_root)
    if not os.path.isdir(root):
        return None
    task, _, setting = _parse_eval_name(eval_name)
    req = f"{task}" + (f"_{setting}" if setting else "") + (f"_id-{option_id_set}" if option_id_set else "")
    req_l = req.lower()

    found_dirs = set()
    for pattern in ["*_run*_curve.jsonl", "*_run*_pride_curve.jsonl"]:
        for path in glob.glob(os.path.join(root, "**", pattern), recursive=True):
            found_dirs.add(os.path.dirname(path))

    candidates = []
    for d in found_dirs:
        if model_key not in d:
            continue
        base = os.path.basename(d.rstrip(os.sep)).lower()
        if req_l not in base:
            continue
        candidates.append(d)
    if not candidates:
        return None

    def score(d: str) -> Tuple[int, int, int]:
        # 0s_ModelName/... 또는 0_shot_ModelName/... 형태 우선
        prefer_0s = 1 if (("0s_" + model_key in d) or ("0_shot_" + model_key in d)) else 0
        base, pride = _discover_curve_files(d)
        n = len(base) + len(pride)
        depth = d.count(os.sep)
        return (prefer_0s, n, -depth)

    return max(candidates, key=score)


def _find_results_dir_for_model(results_root: str, model_name: str) -> Optional[str]:
    """
    results_root 아래에서 0s_<model>/<task>_full_id-ABCD 형태의 디렉터리 중
    run0/run1/run2 curve JSONL(*_run*_curve.jsonl, *_run*_pride_curve.jsonl)이 있는 곳만 찾습니다.
    """
    model_key = model_name.split("/")[-1].strip()
    root = os.path.abspath(results_root)
    if not os.path.isdir(root):
        return None
    # run이 꼭 있는 경우만: *_run*_curve.jsonl / *_run*_pride_curve.jsonl
    found_dirs = set()
    for pattern in ["*_run*_curve.jsonl", "*_run*_pride_curve.jsonl"]:
        for path in glob.glob(os.path.join(root, "**", pattern), recursive=True):
            found_dirs.add(os.path.dirname(path))
    # 0s_Llama-3.2-3B/arc_full_id-ABCD 형태: 경로에 모델명(0s_ 포함)이 있어야 함
    candidates = [d for d in found_dirs if model_key in d]
    if not candidates:
        return None

    def score(d: str) -> Tuple[int, int, int]:
        base, pride = _discover_curve_files(d)
        n = len(base) + len(pride)
        depth = d.count(os.sep)
        # 0s_<model>/... 또는 0_shot_<model>/... 형태 우선
        prefer_0s = 1 if (("0s_" + model_key in d) or ("0_shot_" + model_key in d)) else 0
        return (prefer_0s, n, -depth)

    return max(candidates, key=score)


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
    ap.add_argument("--results_dir", type=str, default="", help="단일 결과 디렉터리 (예: .../mmlu_full_id-ABCD)")
    ap.add_argument("--results_dirs", type=str, nargs="+", default=None,
                    help="run0/run1/run2 JSONL이 있는 디렉터리 여러 개 (15개 모델이면 15개 디렉터리)")
    ap.add_argument("--results_root", type=str, default="",
                    help="모델별 디렉터리 탐색 시 루트 (--models만 줄 때 사용)")
    ap.add_argument("--models", type=str, nargs="+", default=None,
                    help="모델 이름 목록; results_root 아래에서 run* curve JSONL 디렉터리 자동 탐색")
    ap.add_argument("--eval_name", type=str, default="",
                    help="지정 시 경로 규칙으로 계산 (예: mmlu,0,full)")
    ap.add_argument("--option_id_set", type=str, default=None)
    ap.add_argument("--max_subjects", type=int, default=None)
    args = ap.parse_args()

    results_dir = str(args.results_dir).strip()
    results_dirs = getattr(args, "results_dirs", None) or []
    results_dirs = [d.strip() for d in results_dirs if d.strip()]

    if results_dir:
        dirs_to_run = [(results_dir, os.path.basename(results_dir.rstrip(os.sep)))]
    elif results_dirs:
        dirs_to_run = [(d, os.path.basename(d.rstrip(os.sep)) or f"dir_{i}") for i, d in enumerate(results_dirs)]
    elif args.models:
        root = str(args.results_root).strip() or os.path.dirname(os.path.abspath(__file__))
        model_list = [str(m).strip() for m in args.models if str(m).strip()]
        if args.eval_name:
            dirs_to_run = []
            for m in model_list:
                computed = _compute_results_dir(root, args.eval_name, m, args.option_id_set)
                if _has_run_curve_files(computed) or os.path.isdir(computed):
                    # run curve가 있으면 그대로, (혹시 run 없이도 존재하면) 일단 사용
                    dirs_to_run.append((computed, m.split("/")[-1]))
                    continue
                # fallback: results_root 아래에서 task/setting/id까지 맞는 디렉터리 자동 탐색
                found = _find_results_dir_for_model_and_eval(root, m, args.eval_name, args.option_id_set)
                if found:
                    dirs_to_run.append((found, m.split("/")[-1]))
                else:
                    print(f"Warning: 디렉터리를 찾지 못함 (model={m}, computed={computed})", file=sys.stderr)
            if not dirs_to_run:
                print("Error: 어떤 모델에 대해서도 디렉터리를 찾지 못했습니다.", file=sys.stderr)
                sys.exit(1)
        else:
            dirs_to_run = []
            for m in model_list:
                found = _find_results_dir_for_model(root, m)
                if found:
                    dirs_to_run.append((found, m.split("/")[-1]))
                else:
                    print(f"Warning: run* curve 디렉터리를 찾지 못함 (model={m}, root={root})", file=sys.stderr)
            if not dirs_to_run:
                print("Error: 어떤 모델에 대해서도 디렉터리를 찾지 못했습니다.", file=sys.stderr)
                sys.exit(1)
    else:
        print("Error: --results_dir, --results_dirs, 또는 --models 중 하나를 지정하세요.", file=sys.stderr)
        sys.exit(1)

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

            # 항상 acc=0.8035±0.0007, recall_std=0.0364±0.0022 형식 (std 없으면 ±0.0000)
            acc_str = f"{acc_m:.4f}±{acc_s:.4f}" if acc_s is not None else f"{acc_m:.4f}±0.0000"
            rstr = (f", recall_std={rstd_m:.4f}±{rstd_s:.4f}" if rstd_s is not None else f", recall_std={rstd_m:.4f}±0.0000") if rstd_m is not None else ""
            cstr = (f", cost={cost_m:.4f}±{cost_s:.4f}" if cost_s is not None else f", cost={cost_m:.4f}±0.0000") if cost_m is not None else ""

            print(f"  {p}% : acc={acc_str}{rstr}{cstr}")
    print("\n======================================================")


if __name__ == "__main__":
    main()