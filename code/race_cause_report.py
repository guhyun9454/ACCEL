"""Decompose an empirical ACCEL trajectory against its per-item PriDe reference."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def trace_online_percentile(
    trajectories: Iterable[Dict[str, Any]],
    percentile: float,
    k: int,
    stage_schedule: str = "flat",
    stage_gamma: float = 0.5,
) -> List[Dict[str, Any]]:
    """Replay the production online-percentile router with per-item stop traces."""
    schedule = str(stage_schedule or "flat").strip().lower()
    if schedule not in {"flat", "sqrt"}:
        raise ValueError(f"unsupported stage schedule: {stage_schedule}")

    base_percentile = min(max(float(percentile), 0.0), 100.0)
    gamma = float(stage_gamma)
    histories: List[List[float]] = [[] for _ in range(int(k))]
    traced = []

    for row in trajectories:
        decision_stages = [int(value) for value in row["decision_stages"]]
        confidences = [float(value) for value in row["conf_by_stage"]]
        predictions = [int(value) for value in row["pred_by_stage"]]
        if not (
            len(decision_stages) == len(confidences) == len(predictions)
            and decision_stages
        ):
            raise ValueError(f"inconsistent trajectory for sample {row.get('sample_id')}")
        if decision_stages[0] != 1:
            raise ValueError(
                f"trajectory must start at stage 1 for sample {row.get('sample_id')}"
            )

        forced_prefix = bool(row.get("prefix_forced", False))
        stop_stage = int(decision_stages[-1])
        if not forced_prefix:
            for local_idx, stage_id in enumerate(decision_stages):
                stage_percentile = (
                    base_percentile / (float(stage_id) ** gamma)
                    if schedule == "sqrt"
                    else base_percentile
                )
                quantile = min(max(stage_percentile / 100.0, 0.0), 1.0)
                history = histories[stage_id - 1]
                threshold = (
                    float(np.quantile(np.asarray(history, dtype=np.float64), quantile))
                    if history
                    else 0.0
                )
                if confidences[local_idx] >= threshold:
                    stop_stage = int(stage_id)
                    break

        stop_local_idx = decision_stages.index(stop_stage)
        for local_idx, stage_id in enumerate(decision_stages[: stop_local_idx + 1]):
            histories[stage_id - 1].append(confidences[local_idx])

        label_idx = int(row["label_idx"])
        accel_pred_idx = int(predictions[stop_local_idx])
        pride_pred_idx = int(row["pride_pred_idx"])
        traced.append({
            **row,
            "stop_stage": stop_stage,
            "accel_pred_idx": accel_pred_idx,
            "accel_correct": int(accel_pred_idx == label_idx),
            "pride_correct": int(pride_pred_idx == label_idx),
            "stage1_correct": int(predictions[0] == label_idx),
        })

    return traced


def _group_summary(rows: List[Dict[str, Any]], total_n: int) -> Dict[str, Any]:
    n_rows = len(rows)
    accel_correct = sum(int(row["accel_correct"]) for row in rows)
    pride_correct = sum(int(row["pride_correct"]) for row in rows)
    stage1_correct = sum(int(row["stage1_correct"]) for row in rows)
    return {
        "n": n_rows,
        "fraction": float(n_rows / total_n) if total_n else float("nan"),
        "accel_accuracy": float(accel_correct / n_rows) if n_rows else float("nan"),
        "pride_accuracy": float(pride_correct / n_rows) if n_rows else float("nan"),
        "stage1_accuracy": float(stage1_correct / n_rows) if n_rows else float("nan"),
        "accel_minus_pride": (
            float((accel_correct - pride_correct) / n_rows)
            if n_rows
            else float("nan")
        ),
        "weighted_contribution": (
            float((accel_correct - pride_correct) / total_n)
            if total_n
            else float("nan")
        ),
        "prediction_disagreement": (
            float(np.mean([
                int(row["accel_pred_idx"]) != int(row["pride_pred_idx"])
                for row in rows
            ]))
            if n_rows
            else float("nan")
        ),
    }


def _stage1_correction_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare empirical residual-mixture stage 1 with PriDe base correction."""
    n_rows = len(rows)
    empirical_correct = sum(
        int(row["pred_by_stage"][0]) == int(row["label_idx"])
        for row in rows
    )
    pride_correct = sum(
        int(row["pride_base_pred_idx"]) == int(row["label_idx"])
        for row in rows
    )
    empirical_wins = sum(
        int(row["pred_by_stage"][0]) == int(row["label_idx"])
        and int(row["pride_base_pred_idx"]) != int(row["label_idx"])
        for row in rows
    )
    pride_wins = sum(
        int(row["pred_by_stage"][0]) != int(row["label_idx"])
        and int(row["pride_base_pred_idx"]) == int(row["label_idx"])
        for row in rows
    )
    return {
        "n": n_rows,
        "empirical_mixture_stage1_accuracy": (
            float(empirical_correct / n_rows) if n_rows else float("nan")
        ),
        "pride_base_accuracy": (
            float(pride_correct / n_rows) if n_rows else float("nan")
        ),
        "empirical_minus_pride": (
            float((empirical_correct - pride_correct) / n_rows)
            if n_rows
            else float("nan")
        ),
        "prediction_disagreement": (
            float(np.mean([
                int(row["pred_by_stage"][0]) != int(row["pride_base_pred_idx"])
                for row in rows
            ]))
            if n_rows
            else float("nan")
        ),
        "empirical_wins": int(empirical_wins),
        "pride_wins": int(pride_wins),
        "net_empirical_wins": int(empirical_wins - pride_wins),
    }


def _router_stage_summary(traced: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Expose schedule potential separately from the online router decision."""
    rows = [
        row for row in traced if not bool(row.get("prefix_forced", False))
    ]
    n_rows = len(rows)
    if not rows:
        return {
            "scope": "nonprefix",
            "n_nonprefix": 0,
            "stage1_accuracy": float("nan"),
            "router_accuracy": float("nan"),
            "last_stage_accuracy": float("nan"),
            "oracle_any_stage_accuracy": float("nan"),
            "router_net_corrections_vs_stage1_count": 0,
            "available_oracle_corrections_vs_stage1_count": 0,
            "router_regret_to_oracle_count": 0,
            "stop_stage_histogram": {},
            "stagewise": {},
        }

    stage_ids = sorted({
        int(stage_id)
        for row in rows
        for stage_id in row["decision_stages"]
    })
    stagewise = {}
    for stage_id in stage_ids:
        available = []
        for row in rows:
            stage_to_pred = {
                int(current_stage): int(prediction)
                for current_stage, prediction in zip(
                    row["decision_stages"], row["pred_by_stage"]
                )
            }
            if stage_id in stage_to_pred:
                available.append((row, stage_to_pred[stage_id]))
        stage_correct = sum(
            int(prediction) == int(row["label_idx"])
            for row, prediction in available
        )
        corrected = sum(
            int(row["stage1_correct"]) == 0
            and int(prediction) == int(row["label_idx"])
            for row, prediction in available
        )
        broken = sum(
            int(row["stage1_correct"]) == 1
            and int(prediction) != int(row["label_idx"])
            for row, prediction in available
        )
        stagewise[str(stage_id)] = {
            "n": len(available),
            "accuracy": (
                float(stage_correct / len(available))
                if available
                else float("nan")
            ),
            "w2c_vs_stage1_count": int(corrected),
            "c2w_vs_stage1_count": int(broken),
            "net_corrections_vs_stage1_count": int(corrected - broken),
        }

    oracle_correct = sum(
        any(
            int(prediction) == int(row["label_idx"])
            for prediction in row["pred_by_stage"]
        )
        for row in rows
    )
    last_stage_correct = sum(
        int(row["pred_by_stage"][-1]) == int(row["label_idx"])
        for row in rows
    )
    router_correct = sum(int(row["accel_correct"]) for row in rows)
    stage1_correct = sum(int(row["stage1_correct"]) for row in rows)
    stop_histogram = defaultdict(int)
    for row in rows:
        stop_histogram[str(int(row["stop_stage"]))] += 1

    return {
        "scope": "nonprefix",
        "n_nonprefix": n_rows,
        "stage1_accuracy": float(stage1_correct / n_rows),
        "router_accuracy": float(router_correct / n_rows),
        "last_stage_accuracy": float(last_stage_correct / n_rows),
        "oracle_any_stage_accuracy": float(oracle_correct / n_rows),
        "router_net_corrections_vs_stage1_count": int(
            router_correct - stage1_correct
        ),
        "available_oracle_corrections_vs_stage1_count": int(
            oracle_correct - stage1_correct
        ),
        "router_regret_to_oracle_count": int(oracle_correct - router_correct),
        "stop_stage_histogram": dict(sorted(stop_histogram.items())),
        "stagewise": stagewise,
    }


def build_decomposition(
    trajectories: List[Dict[str, Any]],
    percentile: float,
    k: int,
    stage_schedule: str = "flat",
    stage_gamma: float = 0.5,
) -> Dict[str, Any]:
    traced = trace_online_percentile(
        trajectories=trajectories,
        percentile=percentile,
        k=k,
        stage_schedule=stage_schedule,
        stage_gamma=stage_gamma,
    )
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in traced:
        if bool(row.get("prefix_forced", False)):
            group = "P_prefix"
        elif int(row["stop_stage"]) > 1:
            group = "E_escalated"
        else:
            group = "U_stage1"
        grouped[group].append(row)

    total_n = len(traced)
    groups = {
        name: _group_summary(grouped.get(name, []), total_n)
        for name in ("P_prefix", "E_escalated", "U_stage1")
    }
    accel_correct = sum(int(row["accel_correct"]) for row in traced)
    pride_correct = sum(int(row["pride_correct"]) for row in traced)
    routed = grouped.get("E_escalated", [])
    routed_w2c = sum(
        int(row["stage1_correct"]) == 0 and int(row["accel_correct"]) == 1
        for row in routed
    )
    routed_c2w = sum(
        int(row["stage1_correct"]) == 1 and int(row["accel_correct"]) == 0
        for row in routed
    )
    nonprefix = [
        row for row in traced if not bool(row.get("prefix_forced", False))
    ]
    stage1_wrong = sum(int(row["stage1_correct"]) == 0 for row in nonprefix)
    routed_stage1_wrong = sum(int(row["stage1_correct"]) == 0 for row in routed)

    return {
        "n_samples": total_n,
        "percentile": float(percentile),
        "k": int(k),
        "stage_schedule": stage_schedule,
        "accel_accuracy": float(accel_correct / total_n) if total_n else float("nan"),
        "pride_accuracy": float(pride_correct / total_n) if total_n else float("nan"),
        "accel_minus_pride": (
            float((accel_correct - pride_correct) / total_n)
            if total_n
            else float("nan")
        ),
        "groups": groups,
        "stage1_correction": {
            "all": _stage1_correction_summary(traced),
            "nonprefix": _stage1_correction_summary(nonprefix),
        },
        "routing": {
            "n_escalated_nonprefix": len(routed),
            "stage1_wrong_precision": (
                float(routed_stage1_wrong / len(routed))
                if routed
                else float("nan")
            ),
            "stage1_wrong_recall": (
                float(routed_stage1_wrong / stage1_wrong)
                if stage1_wrong
                else float("nan")
            ),
            "w2c": int(routed_w2c),
            "c2w": int(routed_c2w),
            "net_corrections": int(routed_w2c - routed_c2w),
        },
        "router_stage_analysis": _router_stage_summary(traced),
        "contribution_check": float(sum(
            group["weighted_contribution"] for group in groups.values()
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--percentile", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--stage-schedule", choices=["flat", "sqrt"], default="flat")
    parser.add_argument("--stage-gamma", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_decomposition(
        trajectories=load_jsonl(args.trajectories),
        percentile=args.percentile,
        k=args.k,
        stage_schedule=args.stage_schedule,
        stage_gamma=args.stage_gamma,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
