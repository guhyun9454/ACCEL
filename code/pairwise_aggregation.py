"""Aggregation and metrics for pairwise MCQ verification.

The inference script emits a probability ``P(i > j)`` for each unordered pair
of answer options.  This module deliberately has no torch/transformers
dependency so the aggregation can be unit-tested and rerun offline.
"""

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


PairProbabilities = Dict[Tuple[int, int], float]


def accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    if len(labels) == 0:
        return float("nan")
    return float(np.mean(np.asarray(labels) == np.asarray(predictions)))


def recall_std(labels: Sequence[int], predictions: Sequence[int], k: int) -> float:
    recalls = []
    labels_arr = np.asarray(labels, dtype=np.int64)
    preds_arr = np.asarray(predictions, dtype=np.int64)
    for option in range(k):
        mask = labels_arr == option
        if np.any(mask):
            recalls.append(float(np.mean(preds_arr[mask] == option)))
    return float(np.std(recalls)) if recalls else float("nan")


def _validate_pairs(pair_probs: PairProbabilities, k: int) -> None:
    expected = {(i, j) for i in range(k) for j in range(i + 1, k)}
    if set(pair_probs) != expected:
        missing = sorted(expected - set(pair_probs))
        extra = sorted(set(pair_probs) - expected)
        raise ValueError(f"pair set mismatch: missing={missing}, extra={extra}")
    for pair, probability in pair_probs.items():
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid probability for {pair}: {probability}")


def copeland_scores(pair_probs: PairProbabilities, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return hard-win counts and summed win probabilities for every option."""
    _validate_pairs(pair_probs, k)
    wins = np.zeros(k, dtype=np.float64)
    soft = np.zeros(k, dtype=np.float64)
    for (i, j), p_i in pair_probs.items():
        soft[i] += p_i
        soft[j] += 1.0 - p_i
        if p_i > 0.5:
            wins[i] += 1.0
        elif p_i < 0.5:
            wins[j] += 1.0
        else:
            wins[i] += 0.5
            wins[j] += 0.5
    return wins, soft


def copeland_predict(pair_probs: PairProbabilities, k: int) -> int:
    """Copeland winner, with summed win probability as a deterministic tie-break."""
    wins, soft = copeland_scores(pair_probs, k)
    candidates = np.flatnonzero(wins == wins.max())
    return int(candidates[np.argmax(soft[candidates])])


def bradley_terry_scores(
    pair_probs: PairProbabilities,
    k: int,
    max_iter: int = 100,
    tolerance: float = 1e-10,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Fit soft-label Bradley--Terry scores with damped Newton updates.

    The final option is fixed to zero to remove the additive-score degeneracy.
    Each observed pair probability is treated as a fractional Bernoulli target.
    """
    _validate_pairs(pair_probs, k)
    theta = np.zeros(k, dtype=np.float64)
    for _ in range(max_iter):
        gradient = np.zeros(k, dtype=np.float64)
        hessian = np.zeros((k, k), dtype=np.float64)
        for (i, j), target in pair_probs.items():
            delta = float(np.clip(theta[i] - theta[j], -40.0, 40.0))
            predicted = 1.0 / (1.0 + np.exp(-delta))
            residual = predicted - target
            weight = max(predicted * (1.0 - predicted), 1e-8)
            gradient[i] += residual
            gradient[j] -= residual
            hessian[i, i] += weight
            hessian[j, j] += weight
            hessian[i, j] -= weight
            hessian[j, i] -= weight

        reduced_hessian = hessian[:-1, :-1] + ridge * np.eye(k - 1)
        step = np.linalg.solve(reduced_hessian, gradient[:-1])
        theta[:-1] -= step
        theta[-1] = 0.0
        if float(np.max(np.abs(step))) < tolerance:
            break
    theta -= theta.mean()
    return theta


def bradley_terry_predict(pair_probs: PairProbabilities, k: int) -> int:
    return int(np.argmax(bradley_terry_scores(pair_probs, k)))


def count_condorcet_cycle(pair_probs: PairProbabilities, k: int) -> bool:
    """Whether the strict majority graph contains a directed cycle."""
    _validate_pairs(pair_probs, k)
    edges = [[] for _ in range(k)]
    for (i, j), p_i in pair_probs.items():
        if p_i > 0.5:
            edges[i].append(j)
        elif p_i < 0.5:
            edges[j].append(i)

    state = [0] * k

    def visit(node: int) -> bool:
        state[node] = 1
        for child in edges[node]:
            if state[child] == 1 or (state[child] == 0 and visit(child)):
                return True
        state[node] = 2
        return False

    return any(state[node] == 0 and visit(node) for node in range(k))

