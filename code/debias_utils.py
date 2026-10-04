
import os
import random
import numpy as np
from itertools import permutations


def argsort(l):
    ranking = np.argsort(l)
    for rdx in range(len(ranking)):
        if rdx == 0:
            continue
        if l[ranking[rdx]] == l[ranking[rdx-1]]:
            ranking[rdx] = ranking[rdx-1]
    return ranking


def simple(observed, permuted_indices=None):
    observed = np.array(observed)
    observed = observed / (observed.sum(axis=1, keepdims=True) + 1e-10)
    # Square (k, k) inputs are the k cyclic rotations for any k. Before
    # 2026-10-04 only (4,4)/(5,5)/(24,4) were accepted; for k=2 and k=10 the
    # assertion fired, the eval_clm callers swallowed it, and the prior silently
    # fell back to uniform (no PriDe correction, empty ACCEL prefix).
    assert observed.shape in [(24, 4)] or observed.shape[0] == observed.shape[1], observed.shape

    if permuted_indices is None:
        if observed.shape[1] not in (4, 5):
            k = observed.shape[1]
            permuted_indices = [tuple((i + s) % k for i in range(k)) for s in range(k)]
        elif observed.shape[1] == 4:
            permuted_indices = [
                (0, 1, 2, 3),
                (1, 2, 3, 0),
                (2, 3, 0, 1),
                (3, 0, 1, 2),
            ]
        else:
            permuted_indices = [
                (0, 1, 2, 3, 4),
                (1, 2, 3, 4, 0),
                (2, 3, 4, 0, 1),
                (3, 4, 0, 1, 2),
                (4, 0, 1, 2, 3),
            ]

    if observed.shape[0] != observed.shape[1]:
        observed = observed[[cantor_expansion(e) for e in permuted_indices]]
    debiased = gather_probs(observed, permuted_indices)
    debiased = np.mean(debiased, axis=1)
    prior = softmax(np.log(observed + 1e-10).mean(axis=0))

    return observed, debiased, prior


def full(observed, permuted_indices=None):
    observed = np.array(observed)
    observed = observed / (observed.sum(axis=1, keepdims=True) + 1e-10)
    assert observed.shape in [(24, 4)], observed.shape

    if permuted_indices is not None:
        observed = observed[[cantor_expansion(e) for e in permuted_indices]]
    debiased = gather_probs(observed, permuted_indices)
    debiased = np.mean(debiased, axis=1)
    prior = softmax(np.log(observed + 1e-10).mean(axis=0))

    return observed, debiased, prior


def gather_probs(observed, permuted_indices=None):
    if permuted_indices is None:
        permuted_indices = sorted(permutations(range(observed.shape[1])))
    assert len(permuted_indices) == observed.shape[0]
    gathered_probs = [[] for _ in range(observed.shape[1])]

    for pdx, indices in enumerate(permuted_indices):
        for idx, index in enumerate(indices):
            gathered_probs[index].append(observed[pdx, idx])
    return gathered_probs


def cantor_expansion(p):
    n = len(p)
    code = 0
    for i in range(n):
        smaller_count = sum(1 for j in range(i+1, n) if p[j] < p[i])
        if smaller_count > 0:
            code += smaller_count * factorial(n - i - 1)
    return code


def factorial(num):
    if num == 0:
        return 1
    return num * factorial(num - 1)


def softmax(x):
    x = np.array(x)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)
    return x
