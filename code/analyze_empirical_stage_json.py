#!/usr/bin/env python3
"""
Analyze saved empirical-stage JSON outputs produced by eval_clm.py.

Primary inputs:
  - results_.../full.../empirical_analysis/{subject}_run{r}_empirical_alpha{a}_summary.json
  - results_.../full.../empirical_analysis/{subject}_run{r}_empirical_alpha{a}_trajectories.jsonl
  - results_.../full.../{task}_empirical_stage_analysis.json
  - results_.../full.../{task}_three_curves_points.json   (optional)

This script builds one consolidated report json/markdown file.  It can also
produce an extra cyclic_learned summary when requested.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _sanitize_result_tag(tag: Optional[str]) -> str:
    s = str(tag or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s.strip("-._")


def _build_results_dir(
    *,
    task: str,
    num_few_shot: int,
    model_name: str,
    option_id_set: Optional[str],
    result_tag: Optional[str],
    setting: str = "full",
) -> str:
    out = f"results_{task}/{int(num_few_shot)}s_{model_name}/{task}"
    if setting:
        out += f"_{setting}"
    if option_id_set:
        out += f"_id-{option_id_set}"
    tag = _sanitize_result_tag(result_tag)
    if tag:
        out += f"__{tag}"
    return out


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _float_key(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("inf")


def _stats(values: Iterable[Any]) -> Dict[str, Any]:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _normalize_alpha(alpha: Any) -> str:
    f = _safe_float(alpha)
    if not math.isfinite(f):
        return str(alpha)
    return f"{f:g}"


def _extract_task_from_dir(results_dir: str) -> Optional[str]:
    task_jsons = glob.glob(os.path.join(results_dir, "*_empirical_stage_analysis.json"))
    if len(task_jsons) == 1:
        return os.path.basename(task_jsons[0]).replace("_empirical_stage_analysis.json", "")
    return None


def _load_records(task_analysis_path: Optional[str], summary_paths: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    task_payload: Dict[str, Any] = {}
    if task_analysis_path and os.path.exists(task_analysis_path):
        task_payload = _load_json(task_analysis_path)
        records = list(task_payload.get("records") or [])
        if records:
            return records, task_payload
    records = [_load_json(path) for path in summary_paths]
    return records, task_payload


def _summarize_stage_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        alpha_key = _normalize_alpha(rec.get("alpha"))
        summary = rec.get("summary") or {}
        stage_metrics = summary.get("stage_metrics") or {}
        for stage_id, vals in stage_metrics.items():
            if not isinstance(vals, dict):
                continue
            for metric in ["acc", "nll", "ece", "avg_conf", "conf_correct", "conf_wrong"]:
                grouped[alpha_key][str(stage_id)][metric].append(vals.get(metric))

    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for stage_id in sorted(grouped[alpha_key].keys(), key=_float_key):
            out[alpha_key][stage_id] = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][stage_id].items()
            }
    return out


def _summarize_transitions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        alpha_key = _normalize_alpha(rec.get("alpha"))
        transitions = ((rec.get("summary") or {}).get("transitions") or [])
        for trans in transitions:
            trans_key = f"{int(trans.get('from_stage', -1))}->{int(trans.get('to_stage', -1))}"
            for metric in ["delta_acc", "w2c", "c2w", "acc_from", "acc_to"]:
                grouped[alpha_key][trans_key][metric].append(trans.get(metric))

    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for trans_key in sorted(grouped[alpha_key].keys(), key=lambda x: tuple(int(p) for p in x.split("->"))):
            out[alpha_key][trans_key] = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][trans_key].items()
            }
    return out


def _summarize_adaptive_points(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sweep_key_by_alpha: Dict[str, str] = {}
    for rec in records:
        alpha_key = _normalize_alpha(rec.get("alpha"))
        points = ((rec.get("summary") or {}).get("adaptive_points") or [])
        for point in points:
            if not isinstance(point, dict):
                continue
            if "p" in point:
                sweep_key = "p"
                sweep_value = float(point["p"])
            elif "confidence" in point:
                sweep_key = "confidence"
                sweep_value = float(point["confidence"])
            else:
                continue
            sweep_key_by_alpha[alpha_key] = sweep_key
            sweep_name = f"{sweep_value:g}"
            for metric in ["cost", "acc", "recall_std"]:
                grouped[alpha_key][sweep_name][metric].append(point.get(metric))
            for key, value in point.items():
                if isinstance(key, str) and key.startswith("n_stage_"):
                    grouped[alpha_key][sweep_name][key].append(value)

    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {
            "sweep_key": sweep_key_by_alpha.get(alpha_key, "p"),
            "points": {},
        }
        for sweep_name in sorted(grouped[alpha_key].keys(), key=_float_key):
            out[alpha_key]["points"][sweep_name] = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][sweep_name].items()
            }
    return out


def _summarize_trajectories(traj_paths: List[str]) -> Dict[str, Any]:
    pattern_counts: Counter[str] = Counter()
    total_rows = 0
    prefix_forced = 0
    sample_id_counts: Counter[int] = Counter()
    for path in traj_paths:
        rows = _load_jsonl(path)
        total_rows += len(rows)
        for row in rows:
            if bool(row.get("prefix_forced", False)):
                prefix_forced += 1
            sample_id = row.get("sample_id")
            if isinstance(sample_id, int):
                sample_id_counts[int(sample_id)] += 1
            stages = row.get("decision_stages") or []
            pattern = "-".join(str(int(x)) for x in stages)
            pattern_counts[pattern] += 1
    return {
        "n_files": int(len(traj_paths)),
        "n_rows": int(total_rows),
        "prefix_forced_ratio": float(prefix_forced / total_rows) if total_rows > 0 else float("nan"),
        "n_unique_sample_ids": int(len(sample_id_counts)),
        "decision_stage_patterns": dict(sorted(pattern_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def _summarize_record_metadata(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_records": int(len(records)),
        "subjects": sorted({str(rec.get("subject", "")) for rec in records if rec.get("subject") is not None}),
        "run_indices": sorted({int(rec.get("run_idx", -1)) for rec in records if rec.get("run_idx") is not None}),
        "alphas": sorted({_safe_float(rec.get("alpha")) for rec in records if math.isfinite(_safe_float(rec.get("alpha")))}),
        "transition_modes": dict(Counter(str(rec.get("transition_mode", "")) for rec in records if rec.get("transition_mode") is not None)),
        "residual_models": dict(Counter(str(rec.get("residual_model", "")) for rec in records if rec.get("residual_model") is not None)),
        "threshold_schedules": dict(Counter(str(rec.get("threshold_schedule", "")) for rec in records if rec.get("threshold_schedule") is not None)),
        "percentile_modes": dict(Counter(str(rec.get("percentile_mode", "")) for rec in records if rec.get("percentile_mode") is not None)),
    }


def _summarize_points_payload(points_path: Optional[str]) -> Dict[str, Any]:
    if not points_path or not os.path.exists(points_path):
        return {}
    payload = _load_json(points_path)
    empirical = ((payload.get("curves") or {}).get("empirical_pride") or {})
    out = {
        "path": points_path,
        "version": payload.get("version"),
        "empirical_pride": {},
    }
    if empirical:
        out["empirical_pride"] = {
            "sweep_mode": empirical.get("sweep_mode"),
            "percentile_mode": empirical.get("percentile_mode", "online"),
            "residual_model": empirical.get("residual_model"),
            "transition_mode": empirical.get("transition_mode"),
            "selection_policy": empirical.get("selection_policy"),
            "selected_sequence_name": empirical.get("selected_sequence_name"),
            "selected_action_sequence": empirical.get("selected_action_sequence"),
            "threshold_schedule": empirical.get("threshold_schedule"),
            "threshold_gamma": empirical.get("threshold_gamma"),
            "empirical_prefix_fractions": empirical.get("empirical_prefix_fractions") or empirical.get("pride_prefix_fractions"),
            "by_alpha_selection": {
                alpha_key: ((alpha_payload or {}).get("selection") or {})
                for alpha_key, alpha_payload in sorted(((empirical.get("by_alpha") or {}).items()), key=lambda kv: _float_key(kv[0]))
            },
        }
    return out


def _summarize_cyclic_learned(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    learned_records = [rec for rec in records if str(rec.get("transition_mode", "")).strip() == "cyclic_learned"]
    if not learned_records:
        return {}

    seq_counts: Counter[str] = Counter()
    seq_by_subject: Dict[str, Counter[str]] = defaultdict(Counter)
    seq_by_alpha: Dict[str, Counter[str]] = defaultdict(Counter)
    actions_by_name: Dict[str, List[str]] = {}
    cand_scores: Dict[str, List[float]] = defaultdict(list)

    for rec in learned_records:
        subject = str(rec.get("subject", ""))
        alpha_key = _normalize_alpha(rec.get("alpha"))
        seq_name = str(rec.get("selected_sequence_name", "")).strip()
        if seq_name:
            seq_counts[seq_name] += 1
            seq_by_subject[subject][seq_name] += 1
            seq_by_alpha[alpha_key][seq_name] += 1
            if seq_name not in actions_by_name:
                actions_by_name[seq_name] = list(rec.get("selected_action_sequence") or [])
        for row in rec.get("candidate_sequence_scores") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            score = _safe_float(row.get("score"))
            if name and math.isfinite(score):
                cand_scores[name].append(score)
            if name and name not in actions_by_name:
                actions_by_name[name] = list(row.get("actions") or [])

    return {
        "n_records": int(len(learned_records)),
        "selected_sequence_counts": dict(sorted(seq_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "selected_sequence_by_subject": {
            subject: dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
            for subject, counter in sorted(seq_by_subject.items())
        },
        "selected_sequence_by_alpha": {
            alpha_key: dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
            for alpha_key, counter in sorted(seq_by_alpha.items(), key=lambda kv: _float_key(kv[0]))
        },
        "actions_by_sequence": actions_by_name,
        "candidate_score_summary": {
            name: _stats(scores)
            for name, scores in sorted(cand_scores.items())
        },
    }


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    meta = report.get("metadata") or {}
    lines.append(f"# Empirical Analysis Report: {meta.get('task', 'unknown')}")
    lines.append("")
    lines.append(f"- `results_dir`: `{meta.get('results_dir', '')}`")
    lines.append(f"- `model_name`: `{meta.get('model_name', '')}`")
    lines.append(f"- `option_id_set`: `{meta.get('option_id_set', '')}`")
    lines.append(f"- `result_tag`: `{meta.get('result_tag', '')}`")
    lines.append("")

    overview = report.get("record_overview") or {}
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- `n_records`: {overview.get('n_records', 0)}")
    lines.append(f"- `subjects`: {', '.join(overview.get('subjects', []))}")
    lines.append(f"- `alphas`: {', '.join(str(x) for x in overview.get('alphas', []))}")
    lines.append(f"- `transition_modes`: `{json.dumps(overview.get('transition_modes', {}), ensure_ascii=False)}`")
    lines.append("")

    traj = report.get("trajectory_summary") or {}
    if traj:
        lines.append("## Trajectories")
        lines.append("")
        lines.append(f"- `n_files`: {traj.get('n_files', 0)}")
        lines.append(f"- `n_rows`: {traj.get('n_rows', 0)}")
        lines.append(f"- `prefix_forced_ratio`: {traj.get('prefix_forced_ratio', float('nan')):.4f}")
        lines.append(f"- `decision_stage_patterns`: `{json.dumps(traj.get('decision_stage_patterns', {}), ensure_ascii=False)}`")
        lines.append("")

    learned = report.get("cyclic_learned_summary") or {}
    if learned:
        lines.append("## Cyclic Learned")
        lines.append("")
        lines.append(f"- `n_records`: {learned.get('n_records', 0)}")
        lines.append(f"- `selected_sequence_counts`: `{json.dumps(learned.get('selected_sequence_counts', {}), ensure_ascii=False)}`")
        lines.append("")

    lines.append("## Stage Metrics")
    lines.append("")
    for alpha_key, alpha_payload in sorted((report.get("stage_metric_summary") or {}).items(), key=lambda kv: _float_key(kv[0])):
        lines.append(f"### alpha={alpha_key}")
        for stage_id, stage_payload in sorted(alpha_payload.items(), key=lambda kv: _float_key(kv[0])):
            acc = ((stage_payload.get("acc") or {}).get("mean"))
            nll = ((stage_payload.get("nll") or {}).get("mean"))
            ece = ((stage_payload.get("ece") or {}).get("mean"))
            conf = ((stage_payload.get("avg_conf") or {}).get("mean"))
            lines.append(
                f"- stage `{stage_id}`: acc={acc:.4f} nll={nll:.4f} ece={ece:.4f} avg_conf={conf:.4f}"
                if all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in [acc, nll, ece, conf])
                else f"- stage `{stage_id}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Direct path to results_<task>/.../<task>_full[_id-...][__result_tag]. If omitted, reconstruct from task/model options.")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (e.g. arc, csqa, mmlu). Required unless --results_dir is provided or can be inferred.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name suffix used in results dir (e.g. Olmo-3-7B-Instruct). Required unless --results_dir is provided.")
    parser.add_argument("--num_few_shot", type=int, default=0,
                        help="Few-shot count used in results dir reconstruction.")
    parser.add_argument("--option_id_set", type=str, default=None,
                        help="Option-id suffix used in results dir reconstruction.")
    parser.add_argument("--result_tag", type=str, default=None,
                        help="Optional result_tag suffix used in results dir reconstruction.")
    parser.add_argument("--setting", type=str, default="full",
                        help="Setting suffix for results dir reconstruction. Empirical analysis normally uses full.")
    parser.add_argument("--analyze_cyclic_learned", action="store_true",
                        help="Add cyclic_learned-specific action-sequence analysis when available.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional explicit output json path. Default: <results_dir>/<task>_empirical_analysis_report.json")
    parser.add_argument("--output_md", type=str, default=None,
                        help="Optional explicit output markdown path. Default: <results_dir>/<task>_empirical_analysis_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir
    if not results_dir:
        if not args.task or not args.model_name:
            raise SystemExit("Either --results_dir or both --task and --model_name are required.")
        results_dir = _build_results_dir(
            task=str(args.task),
            num_few_shot=int(args.num_few_shot),
            model_name=str(args.model_name),
            option_id_set=args.option_id_set,
            result_tag=args.result_tag,
            setting=str(args.setting or "full"),
        )
    results_dir = os.path.abspath(results_dir)
    if not os.path.isdir(results_dir):
        raise SystemExit(f"Results directory not found: {results_dir}")

    task = str(args.task or _extract_task_from_dir(results_dir) or "").strip()
    if not task:
        raise SystemExit("Could not infer task from results dir. Please provide --task.")

    task_analysis_path = os.path.join(results_dir, f"{task}_empirical_stage_analysis.json")
    points_path = os.path.join(results_dir, f"{task}_three_curves_points.json")
    empirical_analysis_dir = os.path.join(results_dir, "empirical_analysis")
    summary_paths = sorted(glob.glob(os.path.join(empirical_analysis_dir, "*_summary.json")))
    traj_paths = sorted(glob.glob(os.path.join(empirical_analysis_dir, "*_trajectories.jsonl")))

    records, task_payload = _load_records(task_analysis_path if os.path.exists(task_analysis_path) else None, summary_paths)
    if not records:
        raise SystemExit(f"No empirical analysis records found under {results_dir}")

    first_record = records[0]
    model_name = str(
        (task_payload.get("model_name") if isinstance(task_payload, dict) else None)
        or args.model_name
        or first_record.get("model_name", "")
    )

    report: Dict[str, Any] = {
        "metadata": {
            "task": task,
            "results_dir": results_dir,
            "model_name": model_name,
            "num_few_shot": int(args.num_few_shot),
            "option_id_set": args.option_id_set,
            "result_tag": _sanitize_result_tag(args.result_tag),
            "task_analysis_path": task_analysis_path if os.path.exists(task_analysis_path) else None,
            "points_path": points_path if os.path.exists(points_path) else None,
            "empirical_analysis_dir": empirical_analysis_dir if os.path.isdir(empirical_analysis_dir) else None,
        },
        "source_files": {
            "n_summary_files": int(len(summary_paths)),
            "n_trajectory_files": int(len(traj_paths)),
            "summary_files": summary_paths,
            "trajectory_files": traj_paths,
        },
        "record_overview": _summarize_record_metadata(records),
        "stage_metric_summary": _summarize_stage_metrics(records),
        "transition_summary": _summarize_transitions(records),
        "adaptive_point_summary": _summarize_adaptive_points(records),
        "trajectory_summary": _summarize_trajectories(traj_paths),
        "points_payload_summary": _summarize_points_payload(points_path if os.path.exists(points_path) else None),
    }
    if args.analyze_cyclic_learned:
        report["cyclic_learned_summary"] = _summarize_cyclic_learned(records)

    output_json = args.output_json or os.path.join(results_dir, f"{task}_empirical_analysis_report.json")
    output_md = args.output_md or os.path.join(results_dir, f"{task}_empirical_analysis_report.md")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print(f"Saved empirical analysis report: {output_json}")
    print(f"Saved empirical analysis markdown: {output_md}")
    if args.analyze_cyclic_learned:
        learned = report.get("cyclic_learned_summary") or {}
        if learned:
            print(
                "Cyclic learned sequence counts: "
                + json.dumps(learned.get("selected_sequence_counts", {}), ensure_ascii=False)
            )
        else:
            print("Cyclic learned analysis requested, but no cyclic_learned records were found.")


if __name__ == "__main__":
    main()
