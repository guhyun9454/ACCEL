"""Offline stage-1 ablation for PriDe versus ACCEL's empirical correction."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def load_results(path: Path, option_ids: Sequence[str]) -> tuple:
    per_sample_probs = []
    labels = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if row.get("type") != "result":
                continue
            data = row["data"]
            per_sample_probs.append(
                np.asarray(data["probs"], dtype=np.float64)
            )
            labels.append(option_ids.index(str(data["ideal"])))
    if not per_sample_probs:
        raise ValueError(f"no result rows found in {path}")
    return per_sample_probs, np.asarray(labels, dtype=np.int64)


def recall_std(labels: np.ndarray, predictions: np.ndarray, k: int) -> float:
    recalls = []
    for class_idx in range(int(k)):
        mask = labels == class_idx
        if np.any(mask):
            recalls.append(float(np.mean(predictions[mask] == class_idx)))
    return float(np.std(np.asarray(recalls, dtype=np.float64)))


def _variant_summary(
    labels: np.ndarray,
    predictions: np.ndarray,
    reference_predictions: np.ndarray,
    mask: np.ndarray,
    k: int,
) -> Dict[str, Any]:
    labels_use = labels[mask]
    predictions_use = predictions[mask]
    reference_use = reference_predictions[mask]
    n_rows = int(labels_use.size)
    correct = predictions_use == labels_use
    reference_correct = reference_use == labels_use
    return {
        "n": n_rows,
        "accuracy": float(np.mean(correct)),
        "recall_std": recall_std(labels_use, predictions_use, k),
        "accuracy_delta_vs_pride": float(
            np.mean(correct) - np.mean(reference_correct)
        ),
        "prediction_disagreement_vs_pride": float(
            np.mean(predictions_use != reference_use)
        ),
        "w2c_vs_pride_count": int(
            np.sum(np.logical_and(correct, np.logical_not(reference_correct)))
        ),
        "c2w_vs_pride_count": int(
            np.sum(np.logical_and(np.logical_not(correct), reference_correct))
        ),
    }


def build_prediction_report(
    labels: np.ndarray,
    predictions_by_variant: Dict[str, np.ndarray],
    prefix_ids: Iterable[int],
    k: int,
) -> Dict[str, Any]:
    if "pride_arithmetic" not in predictions_by_variant:
        raise ValueError("pride_arithmetic predictions are required")
    labels = np.asarray(labels, dtype=np.int64)
    normalized_predictions = {
        name: np.asarray(predictions, dtype=np.int64)
        for name, predictions in predictions_by_variant.items()
    }
    if any(len(predictions) != len(labels) for predictions in normalized_predictions.values()):
        raise ValueError("prediction length mismatch")

    prefix_mask = np.zeros(len(labels), dtype=bool)
    for sample_pos in prefix_ids:
        prefix_mask[int(sample_pos)] = True
    masks = {
        "all": np.ones(len(labels), dtype=bool),
        "nonprefix": np.logical_not(prefix_mask),
    }
    pride_predictions = normalized_predictions["pride_arithmetic"]
    return {
        scope: {
            name: _variant_summary(
                labels=labels,
                predictions=predictions,
                reference_predictions=pride_predictions,
                mask=mask,
                k=k,
            )
            for name, predictions in normalized_predictions.items()
        }
        for scope, mask in masks.items()
    }


def run_ablation(
    results_path: Path,
    task: str,
    alpha: float,
    seed: int,
    option_ids: Sequence[str],
) -> Dict[str, Any]:
    from eval_clm import (
        _compute_empirical_stage_posteriors,
        _estimate_empirical_pride_bank,
        _estimate_pride_prior_random_prefix_mean,
        _pride_correct_row,
        _stable_u32_seed,
    )

    per_sample_probs, labels = load_results(results_path, option_ids)
    k = len(option_ids)
    if any(probs.shape != (k, k) for probs in per_sample_probs):
        shapes = sorted({tuple(probs.shape) for probs in per_sample_probs})
        raise ValueError(
            f"stage-1 ablation requires exactly {k} cyclic views, got {shapes}"
        )

    cyclic_indices = list(range(k))
    run_seed = _stable_u32_seed(str(task), int(seed))
    pride_prior, pride_meta = _estimate_pride_prior_random_prefix_mean(
        per_sample_probs=per_sample_probs,
        cyclic_indices=cyclic_indices,
        k=k,
        prefix_ratio=float(alpha) / 100.0,
        seed=run_seed,
    )
    mean_prior, mu_hat, residual_bank, empirical_meta = (
        _estimate_empirical_pride_bank(
            per_sample_probs=per_sample_probs,
            cyclic_indices=cyclic_indices,
            k=k,
            prefix_ratio=float(alpha) / 100.0,
            seed=run_seed,
        )
    )
    pride_prefix_ids = [int(value) for value in pride_meta["prefix_ids"]]
    empirical_prefix_ids = [int(value) for value in empirical_meta["prefix_ids"]]
    if set(pride_prefix_ids) != set(empirical_prefix_ids):
        raise ValueError(
            "PriDe and empirical prefix selections differ: "
            f"pride={len(pride_prefix_ids)}, empirical={len(empirical_prefix_ids)}"
        )
    if not np.allclose(pride_prior, mean_prior, rtol=0.0, atol=1e-12):
        raise ValueError("PriDe arithmetic prior does not match empirical mean prior")

    base_probs = np.asarray(
        [np.asarray(probs[0], dtype=np.float64) for probs in per_sample_probs],
        dtype=np.float64,
    )
    predictions_by_variant = {
        "raw": np.argmax(base_probs, axis=1).astype(np.int64),
        "pride_arithmetic": np.asarray([
            int(np.argmax(_pride_correct_row(row, pride_prior)))
            for row in base_probs
        ], dtype=np.int64),
    }
    identity_schedule = [tuple(range(k))]
    zero_residual = np.zeros((1, k), dtype=np.float64)
    zero_mu = np.zeros((k,), dtype=np.float64)
    for name, mu_use, residual_use in (
        ("accel_geometric_mu_only", mu_hat, zero_residual),
        ("accel_residual_only", zero_mu, residual_bank),
        ("accel_empirical_mixture", mu_hat, residual_bank),
    ):
        predictions = []
        for row in base_probs:
            _, stage_predictions, _ = _compute_empirical_stage_posteriors(
                stage_probs=row.reshape(1, -1),
                slot_to_content_schedule=identity_schedule,
                mu_hat=mu_use,
                residual_bank=residual_use,
            )
            predictions.append(int(stage_predictions[0]))
        predictions_by_variant[name] = np.asarray(
            predictions, dtype=np.int64
        )

    geometric_prior = np.exp(mu_hat - float(np.max(mu_hat)))
    geometric_prior = geometric_prior / float(np.sum(geometric_prior))
    scopes = build_prediction_report(
        labels=labels,
        predictions_by_variant=predictions_by_variant,
        prefix_ids=empirical_prefix_ids,
        k=k,
    )
    all_scope = scopes["all"]
    return {
        "task": str(task),
        "results_path": str(results_path),
        "n_samples": int(len(labels)),
        "alpha": float(alpha),
        "prefix_count": int(len(empirical_prefix_ids)),
        "prefix_ids": empirical_prefix_ids,
        "prior_geometry": {
            "arithmetic_prior": np.asarray(pride_prior).tolist(),
            "geometric_prior": np.asarray(geometric_prior).tolist(),
            "l1_distance": float(np.sum(np.abs(pride_prior - geometric_prior))),
            "linf_distance": float(np.max(np.abs(pride_prior - geometric_prior))),
            "residual_mean_abs": float(np.mean(np.abs(residual_bank))),
            "residual_max_abs": float(np.max(np.abs(residual_bank))),
        },
        "scopes": scopes,
        "accuracy_delta_decomposition_all": {
            "arithmetic_to_geometric": float(
                all_scope["accel_geometric_mu_only"]["accuracy"]
                - all_scope["pride_arithmetic"]["accuracy"]
            ),
            "geometric_to_residual_mixture": float(
                all_scope["accel_empirical_mixture"]["accuracy"]
                - all_scope["accel_geometric_mu_only"]["accuracy"]
            ),
            "total_empirical_minus_pride": float(
                all_scope["accel_empirical_mixture"]["accuracy"]
                - all_scope["pride_arithmetic"]["accuracy"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--task", default="race")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--option-ids", default="ABCD")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_ablation(
        results_path=args.results,
        task=args.task,
        alpha=args.alpha,
        seed=args.seed,
        option_ids=list(args.option_ids),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
