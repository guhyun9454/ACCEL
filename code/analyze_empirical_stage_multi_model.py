#!/usr/bin/env python3
"""
Aggregate empirical-stage analysis reports over multiple models.

Preferred input:
  - per-model reports produced by analyze_empirical_stage_json.py

This script averages model-level metrics rather than pooling samples across
models, which is usually the safer summary for cross-model comparisons.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
    model_leaf = str(model_name).strip().split("/")[-1]
    out = f"results_{task}/{int(num_few_shot)}s_{model_leaf}/{task}"
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


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


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


def _parse_csv_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _collect_report_paths(
    *,
    report_paths: Sequence[str],
    results_dirs: Sequence[str],
    task: Optional[str],
    model_names: Sequence[str],
    num_few_shot: int,
    option_id_set: Optional[str],
    result_tag: Optional[str],
    setting: str,
) -> List[str]:
    paths: List[str] = []
    for path in report_paths:
        if path:
            paths.append(os.path.abspath(path))
    for results_dir in results_dirs:
        if not results_dir:
            continue
        rdir = os.path.abspath(results_dir)
        if not task:
            matches = glob.glob(os.path.join(rdir, "*_empirical_analysis_report.json"))
            if len(matches) == 1:
                paths.append(matches[0])
                continue
            raise SystemExit(f"Could not infer report file from results dir without --task: {rdir}")
        paths.append(os.path.join(rdir, f"{task}_empirical_analysis_report.json"))
    if task and model_names:
        for model_name in model_names:
            rdir = _build_results_dir(
                task=task,
                num_few_shot=int(num_few_shot),
                model_name=model_name,
                option_id_set=option_id_set,
                result_tag=result_tag,
                setting=setting,
            )
            paths.append(os.path.join(os.path.abspath(rdir), f"{task}_empirical_analysis_report.json"))
    unique_paths: List[str] = []
    seen = set()
    for path in paths:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique_paths.append(norm)
    return unique_paths


def _maybe_generate_report(
    *,
    report_path: str,
    task: Optional[str],
    include_cyclic_learned: bool,
    force: bool,
) -> bool:
    if os.path.exists(report_path) and not bool(force):
        return True
    results_dir = os.path.dirname(os.path.abspath(report_path))
    if not os.path.isdir(results_dir):
        return False
    report_name = os.path.basename(report_path)
    if task and report_name != f"{task}_empirical_analysis_report.json":
        return False

    script_path = os.path.join(os.path.dirname(__file__), "analyze_empirical_stage_json.py")
    cmd = [
        sys.executable,
        script_path,
        "--results_dir",
        results_dir,
    ]
    if task:
        cmd.extend(["--task", str(task)])
    if include_cyclic_learned:
        cmd.append("--analyze_cyclic_learned")
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return False
    return os.path.exists(report_path)


def _aggregate_stage_metrics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for report in reports:
        payload = report.get("stage_metric_summary") or {}
        for alpha_key, alpha_vals in payload.items():
            for stage_id, stage_vals in (alpha_vals or {}).items():
                for metric, metric_stats in (stage_vals or {}).items():
                    grouped[str(alpha_key)][str(stage_id)][str(metric)].append((metric_stats or {}).get("mean"))
    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for stage_id in sorted(grouped[alpha_key].keys(), key=_float_key):
            out[alpha_key][stage_id] = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][stage_id].items()
            }
    return out


def _aggregate_transition_metrics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    thresh_grouped: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    top_grouped: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    sweep_key_by_alpha_transition: Dict[tuple[str, str], str] = {}
    for report in reports:
        payload = report.get("transition_summary") or {}
        for alpha_key, alpha_vals in payload.items():
            for trans_key, trans_vals in (alpha_vals or {}).items():
                for metric, metric_stats in (trans_vals or {}).items():
                    if metric in {"threshold_sweep_key", "threshold_analysis"}:
                        continue
                    grouped[str(alpha_key)][str(trans_key)][str(metric)].append((metric_stats or {}).get("mean"))
                sweep_key = str((trans_vals or {}).get("threshold_sweep_key", "p"))
                for point_key, point_vals in ((trans_vals or {}).get("threshold_analysis") or {}).items():
                    sweep_key_by_alpha_transition[(str(alpha_key), str(trans_key))] = sweep_key
                    for metric, metric_stats in (point_vals or {}).items():
                        thresh_grouped[str(alpha_key)][str(trans_key)][str(point_key)][str(metric)].append(
                            (metric_stats or {}).get("mean")
                        )
                for point_key, point_vals in ((trans_vals or {}).get("top_threshold_analysis") or {}).items():
                    for metric, metric_stats in (point_vals or {}).items():
                        top_grouped[str(alpha_key)][str(trans_key)][str(point_key)][str(metric)].append(
                            (metric_stats or {}).get("mean")
                        )
    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {}
        for trans_key in sorted(grouped[alpha_key].keys(), key=lambda x: tuple(int(p) for p in str(x).split("->"))):
            entry = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][trans_key].items()
            }
            thresh_payload = {}
            for point_key in sorted(thresh_grouped[alpha_key][trans_key].keys(), key=_float_key):
                thresh_payload[point_key] = {
                    metric: _stats(values)
                    for metric, values in thresh_grouped[alpha_key][trans_key][point_key].items()
                }
            if thresh_payload:
                entry["threshold_sweep_key"] = sweep_key_by_alpha_transition.get((alpha_key, trans_key), "p")
                entry["threshold_analysis"] = thresh_payload
            top_payload = {}
            for point_key in sorted(top_grouped[alpha_key][trans_key].keys(), key=_float_key):
                top_payload[point_key] = {
                    metric: _stats(values)
                    for metric, values in top_grouped[alpha_key][trans_key][point_key].items()
                }
            if top_payload:
                entry["top_threshold_analysis"] = top_payload
            out[alpha_key][trans_key] = entry
    return out


def _aggregate_adaptive_points(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sweep_key_by_alpha: Dict[str, str] = {}
    for report in reports:
        payload = report.get("adaptive_point_summary") or {}
        for alpha_key, alpha_vals in payload.items():
            sweep_key_by_alpha[str(alpha_key)] = str((alpha_vals or {}).get("sweep_key", "p"))
            for point_key, point_vals in ((alpha_vals or {}).get("points") or {}).items():
                for metric, metric_stats in (point_vals or {}).items():
                    grouped[str(alpha_key)][str(point_key)][str(metric)].append((metric_stats or {}).get("mean"))
    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        out[alpha_key] = {"sweep_key": sweep_key_by_alpha.get(alpha_key, "p"), "points": {}}
        for point_key in sorted(grouped[alpha_key].keys(), key=_float_key):
            out[alpha_key]["points"][point_key] = {
                metric: _stats(values)
                for metric, values in grouped[alpha_key][point_key].items()
            }
    return out


def _aggregate_curve_points(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    curve_meta: Dict[str, Dict[str, Any]] = {}
    for report in reports:
        empirical = (((report.get("points_payload_summary") or {}).get("empirical_pride")) or {})
        if not empirical:
            continue
        for alpha_key, selection in (empirical.get("by_alpha_selection") or {}).items():
            grouped[str(alpha_key)]["selection"]["sequence_counts"]["_models"].append(1.0)
            if selection and not curve_meta.get(str(alpha_key)):
                curve_meta[str(alpha_key)] = {
                    "sweep_mode": empirical.get("sweep_mode"),
                    "percentile_mode": empirical.get("percentile_mode", "online"),
                    "residual_model": empirical.get("residual_model"),
                    "transition_mode": empirical.get("transition_mode"),
                    "threshold_schedule": empirical.get("threshold_schedule"),
                    "threshold_gamma": empirical.get("threshold_gamma"),
                }
            for seq_name, count in ((selection or {}).get("sequence_counts") or {}).items():
                grouped[str(alpha_key)]["selection"]["sequence_counts"][str(seq_name)].append(count)

    out: Dict[str, Any] = {}
    for alpha_key in sorted(grouped.keys(), key=_float_key):
        seq_counts = {}
        for seq_name, values in grouped[alpha_key]["selection"]["sequence_counts"].items():
            if seq_name == "_models":
                continue
            seq_counts[seq_name] = _stats(values)
        out[alpha_key] = {
            "meta": curve_meta.get(alpha_key, {}),
            "sequence_counts": seq_counts,
        }
    return out


def _aggregate_cyclic_learned(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    model_rows = []
    total_selected = Counter()
    by_sequence_scores: Dict[str, List[float]] = defaultdict(list)
    actions_by_sequence: Dict[str, List[str]] = {}
    for report in reports:
        learned = report.get("cyclic_learned_summary") or {}
        if not learned:
            continue
        model_name = str((report.get("metadata") or {}).get("model_name", ""))
        sel_counts = Counter({str(k): int(v) for k, v in (learned.get("selected_sequence_counts") or {}).items()})
        total_selected.update(sel_counts)
        for seq_name, action_seq in (learned.get("actions_by_sequence") or {}).items():
            if seq_name not in actions_by_sequence:
                actions_by_sequence[str(seq_name)] = list(action_seq or [])
        for seq_name, score_stats in (learned.get("candidate_score_summary") or {}).items():
            score_mean = _safe_float((score_stats or {}).get("mean"))
            if math.isfinite(score_mean):
                by_sequence_scores[str(seq_name)].append(score_mean)
        model_rows.append({
            "model_name": model_name,
            "selected_sequence_counts": dict(sel_counts),
        })
    if not model_rows:
        return {}
    return {
        "n_models": int(len(model_rows)),
        "selected_sequence_counts_total": dict(sorted(total_selected.items(), key=lambda kv: (-kv[1], kv[0]))),
        "candidate_score_summary_over_models": {
            seq_name: _stats(values)
            for seq_name, values in sorted(by_sequence_scores.items())
        },
        "actions_by_sequence": actions_by_sequence,
        "per_model": model_rows,
    }


def _aggregate_baseline_cyclic_transition(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals: Dict[str, List[float]] = defaultdict(list)
    per_model = []
    for report in reports:
        payload = report.get("baseline_cyclic_transition") or {}
        if not payload:
            continue
        model_name = str((report.get("metadata") or {}).get("model_name", ""))
        row = {"model_name": model_name}
        for metric in ["acc_from", "acc_to", "delta_acc", "w2c", "c2w"]:
            mean_val = _safe_float((payload.get(metric) or {}).get("mean"))
            row[metric] = mean_val
            if math.isfinite(mean_val):
                vals[metric].append(mean_val)
        per_model.append(row)
    if not per_model:
        return {}
    return {
        "n_models": int(len(per_model)),
        "acc_from": _stats(vals.get("acc_from", [])),
        "acc_to": _stats(vals.get("acc_to", [])),
        "delta_acc": _stats(vals.get("delta_acc", [])),
        "w2c": _stats(vals.get("w2c", [])),
        "c2w": _stats(vals.get("c2w", [])),
        "per_model": per_model,
    }


def _aggregate_overview(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    tasks = sorted({str((r.get("metadata") or {}).get("task", "")) for r in reports})
    models = [str((r.get("metadata") or {}).get("model_name", "")) for r in reports]
    transitions = Counter()
    residuals = Counter()
    schedules = Counter()
    percentiles = Counter()
    alphas = set()
    for r in reports:
        ov = r.get("record_overview") or {}
        transitions.update({str(k): int(v) for k, v in (ov.get("transition_modes") or {}).items()})
        residuals.update({str(k): int(v) for k, v in (ov.get("residual_models") or {}).items()})
        schedules.update({str(k): int(v) for k, v in (ov.get("threshold_schedules") or {}).items()})
        percentiles.update({str(k): int(v) for k, v in (ov.get("percentile_modes") or {}).items()})
        for alpha in ov.get("alphas") or []:
            fa = _safe_float(alpha)
            if math.isfinite(fa):
                alphas.add(fa)
    return {
        "n_models": int(len(reports)),
        "tasks": tasks,
        "model_names": models,
        "alphas": sorted(alphas),
        "transition_modes": dict(transitions),
        "residual_models": dict(residuals),
        "threshold_schedules": dict(schedules),
        "percentile_modes": dict(percentiles),
    }


def _build_markdown(report: Dict[str, Any]) -> str:
    meta = report.get("metadata") or {}
    lines = [
        f"# Multi-Model Empirical Analysis Report: {meta.get('task', 'unknown')}",
        "",
        f"- `n_models`: {meta.get('n_models', 0)}",
        f"- `model_names`: {', '.join(meta.get('model_names', []))}",
        f"- `result_tag`: `{meta.get('result_tag', '')}`",
        "",
        "## Overview",
        "",
        f"- `transition_modes`: `{json.dumps((report.get('overview') or {}).get('transition_modes', {}), ensure_ascii=False)}`",
        f"- `residual_models`: `{json.dumps((report.get('overview') or {}).get('residual_models', {}), ensure_ascii=False)}`",
        "",
    ]
    baseline = report.get("baseline_cyclic_transition") or {}
    if baseline:
        lines.extend([
            "## Baseline Cyclic",
            "",
            f"- `delta_acc`: {((baseline.get('delta_acc') or {}).get('mean', float('nan'))):.4f}, "
            f"`w2c`: {((baseline.get('w2c') or {}).get('mean', float('nan'))):.4f}, "
            f"`c2w`: {((baseline.get('c2w') or {}).get('mean', float('nan'))):.4f}",
            "",
        ])
    lines.extend([
        "## Stage Metrics",
        "",
    ])
    for alpha_key, alpha_payload in sorted((report.get("stage_metric_summary") or {}).items(), key=lambda kv: _float_key(kv[0])):
        lines.append(f"### alpha={alpha_key}")
        for stage_id, stage_payload in sorted((alpha_payload or {}).items(), key=lambda kv: _float_key(kv[0])):
            acc = ((stage_payload.get("acc") or {}).get("mean"))
            nll = ((stage_payload.get("nll") or {}).get("mean"))
            ece = ((stage_payload.get("ece") or {}).get("mean"))
            conf = ((stage_payload.get("avg_conf") or {}).get("mean"))
            if all(math.isfinite(_safe_float(x)) for x in [acc, nll, ece, conf]):
                lines.append(f"- stage `{stage_id}`: acc={float(acc):.4f}, nll={float(nll):.4f}, ece={float(ece):.4f}, avg_conf={float(conf):.4f}")
            else:
                lines.append(f"- stage `{stage_id}`")
        lines.append("")
    learned = report.get("cyclic_learned_summary") or {}
    if learned:
        lines.extend([
            "## Cyclic Learned",
            "",
            f"- `selected_sequence_counts_total`: `{json.dumps(learned.get('selected_sequence_counts_total', {}), ensure_ascii=False)}`",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_paths", type=str, default=None,
                        help="Comma-separated list of per-model *_empirical_analysis_report.json paths.")
    parser.add_argument("--results_dirs", type=str, default=None,
                        help="Comma-separated list of results dirs. Each should already contain <task>_empirical_analysis_report.json.")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name, used for report path reconstruction.")
    parser.add_argument("--model_names", type=str, default=None,
                        help="Comma-separated model names for results dir reconstruction.")
    parser.add_argument("--num_few_shot", type=int, default=0,
                        help="Few-shot count used in results dir reconstruction.")
    parser.add_argument("--option_id_set", type=str, default=None,
                        help="Option-id suffix used in results dir reconstruction.")
    parser.add_argument("--result_tag", type=str, default=None,
                        help="Result tag suffix used in results dir reconstruction.")
    parser.add_argument("--setting", type=str, default="full",
                        help="Setting suffix for results dir reconstruction. Usually full.")
    parser.add_argument("--include_cyclic_learned", action="store_true",
                        help="Aggregate cyclic_learned sequence-selection summaries when present.")
    parser.set_defaults(auto_generate_missing_reports=True)
    parser.add_argument("--no_auto_generate_missing_reports", dest="auto_generate_missing_reports", action="store_false",
                        help="Do not auto-run analyze_empirical_stage_json.py for missing per-model report json files.")
    parser.add_argument("--force", action="store_true",
                        help="Force-regenerate per-model empirical analysis reports before multi-model aggregation, even if they already exist.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Explicit output path for aggregated json.")
    parser.add_argument("--output_md", type=str, default=None,
                        help="Explicit output path for aggregated markdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_paths = _collect_report_paths(
        report_paths=_parse_csv_list(args.report_paths),
        results_dirs=_parse_csv_list(args.results_dirs),
        task=args.task,
        model_names=_parse_csv_list(args.model_names),
        num_few_shot=int(args.num_few_shot),
        option_id_set=args.option_id_set,
        result_tag=args.result_tag,
        setting=str(args.setting or "full"),
    )
    if not report_paths:
        raise SystemExit("No report paths resolved. Provide --report_paths, --results_dirs, or --task with --model_names.")
    if args.auto_generate_missing_reports or args.force:
        for path in report_paths:
            _maybe_generate_report(
                report_path=path,
                task=args.task,
                include_cyclic_learned=bool(args.include_cyclic_learned),
                force=bool(args.force),
            )
    missing = [path for path in report_paths if not os.path.exists(path)]
    if missing:
        raise SystemExit("Missing report files:\n" + "\n".join(missing))

    reports = [_load_json(path) for path in report_paths]
    task = str(args.task or (reports[0].get("metadata") or {}).get("task", "")).strip()
    if not task:
        raise SystemExit("Could not infer task. Please provide --task.")

    model_names = [str((r.get("metadata") or {}).get("model_name", "")) for r in reports]
    output_root = os.path.dirname(report_paths[0])
    output_json = args.output_json or os.path.join(output_root, f"{task}_empirical_analysis_multi_model.json")
    output_md = args.output_md or os.path.join(output_root, f"{task}_empirical_analysis_multi_model.md")

    report: Dict[str, Any] = {
        "metadata": {
            "task": task,
            "n_models": int(len(reports)),
            "model_names": model_names,
            "report_paths": report_paths,
            "result_tag": _sanitize_result_tag(args.result_tag),
        },
        "overview": _aggregate_overview(reports),
        "stage_metric_summary": _aggregate_stage_metrics(reports),
        "transition_summary": _aggregate_transition_metrics(reports),
        "adaptive_point_summary": _aggregate_adaptive_points(reports),
        "points_payload_selection_summary": _aggregate_curve_points(reports),
        "baseline_cyclic_transition": _aggregate_baseline_cyclic_transition(reports),
    }
    if args.include_cyclic_learned:
        report["cyclic_learned_summary"] = _aggregate_cyclic_learned(reports)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print(f"Saved multi-model empirical report: {output_json}")
    print(f"Saved multi-model empirical markdown: {output_md}")
    if args.include_cyclic_learned:
        learned = report.get("cyclic_learned_summary") or {}
        if learned:
            print(
                "Cyclic learned total sequence counts: "
                + json.dumps(learned.get("selected_sequence_counts_total", {}), ensure_ascii=False)
            )


if __name__ == "__main__":
    main()
