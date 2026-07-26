"""CalibraEval (Li et al., ACL 2025) adapted to k-way MCQ, run on cached probabilities.

Why this file exists
--------------------
The released implementation (https://github.com/CSHaitao/CalibraEval) is hard-wired
to *binary* pairwise LLM-as-judge data: `normalize.py` renormalizes over exactly
{A, B}, `CalibraEval.py` consumes one scalar per prompt (`prompt_i_logit['A']`),
and `eval.py` scores pairwise agreement against labels in {-1, +1}. None of that
runs on 4-option MCQ, so the method is re-implemented here against the same
objective, generalized from 2 options / 3 prompt variants to k options / m
cyclic permutations.

What is kept from the original
------------------------------
* The calibration map g is a monotone step function on observed probabilities,
  parameterized as a normalized cumulative sum of exp(theta) — this is what makes
  the Non-parametric Order-Preserving Algorithm order-preserving by construction.
* The label-free objective: calibrated distributions should (a) sum to one within
  a view, (b) agree across views of the same item, while (c) a regularizer keeps
  the map from collapsing everything onto the uniform 1/k.
* An isotonic fit of the learned map, so it can be applied to unseen values.

Where this deviates, and why
----------------------------
* Gradients are derived analytically here (and checked against finite differences
  in tests/test_calibraeval_mcq.py) rather than transcribed from the released
  `calculate_loss_and_gradient`, which reuses `exps[s1 - 1]` inside the s2 and s3
  gradient branches where the chain rule calls for `exps[s2 - 1]` / `exps[s3 - 1]`.
  Its `calculate_loss` also omits the `lam` factor that its gradient applies to the
  second term. Reproducing those would reproduce bugs, not the published method.
* Binary consistency `g(p1) + g(p3) = 1` generalizes to `sum_i g(q[c][i]) = 1`.
* Binary invariance `g(p1) = g(p2)` generalizes to the across-view variance of
  `g(q[c][i])` for fixed content option i.

Notation
--------
`P[c][j]` is the cached probability of option-ID slot j under cyclic permutation c.
Permutation c places content option i at slot (i - c) mod k, so the probability the
model assigned to *content* option i under view c is `q[c][i] = P[c][(i - c) % k]`.
Gold labels are content indices, matching `labels_idx_for_curves` in eval_clm.py,
so accuracy and recall_std here are computed in the same space as the cached
Baseline / PriDe / ACCEL numbers.
"""

import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Long enough for both the 4-choice tasks and 5-choice CSQA; k itself is read
# off the cached probability block rather than assumed.
OPTION_IDS = "ABCDE"


# --------------------------------------------------------------------------
# metrics (mirrors eval_clm_online._recall_std so numbers are comparable)
# --------------------------------------------------------------------------

def recall_std(labels: Sequence[int], preds: Sequence[int], k: int) -> float:
    """Std of per-class recall. Classes never appearing as gold are ignored."""
    if k <= 0:
        return float("nan")
    positives = [0] * k
    true_pos = [0] * k
    for y, p in zip(labels, preds):
        y, p = int(y), int(p)
        if 0 <= y < k:
            positives[y] += 1
            if p == y:
                true_pos[y] += 1
    recalls = [true_pos[c] / float(positives[c]) for c in range(k) if positives[c] > 0]
    if not recalls:
        return float("nan")
    return float(np.std(np.asarray(recalls, dtype=np.float64)))


def accuracy(labels: Sequence[int], preds: Sequence[int]) -> float:
    if not len(labels):
        return float("nan")
    return float(np.mean([int(y) == int(p) for y, p in zip(labels, preds)]))


# --------------------------------------------------------------------------
# cache loading
# --------------------------------------------------------------------------

def to_content_space(probs: Sequence[Sequence[float]], k: int) -> np.ndarray:
    """(views, slots) cached probabilities -> (views, content options).

    Content option i sits at slot (i - c) % k under cyclic permutation c.
    """
    arr = np.asarray(probs, dtype=np.float64)
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        for i in range(k):
            out[c, i] = arr[c, (i - c) % k]
    return out


def load_cached_run(path: str, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Read one <task>_run<r>.jsonl -> (Q, y).

    Q has shape (n_items, n_views, k) in **slot** space — exactly as cached, before
    any un-rotation — and y holds gold content indices. Use `to_content_space` to
    un-rotate a view when content-space scores are needed.
    `k` is inferred from the first usable record unless given — ARC/MMLU/RACE are
    4-choice but CSQA is 5-choice, so it must not be hardcoded. Records whose
    probability block is not (>=2 views, k options) are skipped: a truncated or
    single-view record cannot support a cross-view consistency objective.
    """
    Q, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") != "result":
                continue
            data = rec.get("data") or {}
            probs = data.get("probs")
            ideal = data.get("ideal")
            if not probs or ideal is None:
                continue
            arr = np.asarray(probs, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] < 2:
                continue
            if k is None:
                k = int(arr.shape[1])
            if arr.shape[1] != k:
                continue
            gold = OPTION_IDS.index(str(ideal)) if str(ideal) in OPTION_IDS else None
            if gold is None:
                continue
            Q.append(arr)
            y.append(gold)
    if not Q:
        raise ValueError(f"no usable records in {path}")
    views = min(q.shape[0] for q in Q)
    return np.stack([q[:views] for q in Q]), np.asarray(y, dtype=np.int64)


# --------------------------------------------------------------------------
# Non-parametric Order-Preserving Algorithm
# --------------------------------------------------------------------------

class OrderPreservingCalibrator:
    """One monotone map per option slot, fitted label-free to make views agree.

    Each g_j is a step function over the sorted unique observed probabilities: with
    e = exp(theta_j), S_m = sum_{n<m} e_n and T = S_N, the map sends the m-th
    smallest observed value to S_m / T. Monotonicity is structural (e_n > 0) — the
    "order-preserving" property — so g_j never reorders two values *within* slot j.

    Why one map *per slot* and not a single shared map
    --------------------------------------------------
    A single shared monotone map cannot change a k-way argmax at all: ranking is
    invariant under any strictly increasing elementwise transform, with or without
    renormalization. The original method escapes this only because its decision is
    binary — it compares a calibrated marginal p(A) against the fixed threshold
    0.5, so shifting p(A) across 0.5 flips the answer. That is precisely a
    slot-specific correction. Generalizing it faithfully to k options therefore
    means k maps, one per option-ID position: `argmax_j g_j(P[c][j])` is not
    rank-invariant across j, and it is exactly the position bias PriDe and ACCEL
    target. Fitting a shared map instead reproduces the baseline verbatim, which
    is how this was caught.
    """

    def __init__(self, lam: float = 0.05, lr: float = 10.0, epochs: int = 30, seed: int = 0):
        self.lam = float(lam)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.values_: Optional[np.ndarray] = None
        self.mapped_: Optional[np.ndarray] = None

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _g_from_theta(theta: np.ndarray) -> np.ndarray:
        """theta (..., N) -> g (..., N+1), the normalized cumulative sum.

        g[..., 0] = 0 and g[..., N] = 1, so every row is a monotone map onto [0,1].
        """
        theta = np.atleast_2d(theta)
        e = np.exp(theta - theta.max(axis=-1, keepdims=True))
        cs = np.concatenate([np.zeros((theta.shape[0], 1)), np.cumsum(e, axis=-1)], axis=-1)
        return cs / cs[:, -1:]

    @staticmethod
    def _theta_grad(theta: np.ndarray, dg: np.ndarray) -> np.ndarray:
        """Chain dL/dg (rows, N+1) back to dL/dtheta (rows, N).

        With g_m = S_m / T:  dg_m/dtheta_n = e_n * (1{n < m} - g_m) / T.
        So dL/dtheta_n = (e_n / T) * (sum_{m > n} dg_m - sum_m dg_m * g_m).
        Each row (option slot) has its own independent map, so rows do not mix.
        """
        theta = np.atleast_2d(theta)
        dg = np.atleast_2d(dg)
        e = np.exp(theta - theta.max(axis=-1, keepdims=True))
        T = e.sum(axis=-1, keepdims=True)
        g = OrderPreservingCalibrator._g_from_theta(theta)
        # rev[..., n] = sum_{m >= n} dg_m, so rev[..., 1:] = sum_{m > n} dg_m
        rev = np.cumsum(dg[:, ::-1], axis=-1)[:, ::-1]
        inner = np.sum(dg * g, axis=-1, keepdims=True)
        return (e / T) * (rev[:, 1:] - inner)

    def _loss_and_dg(self, g: np.ndarray, idx: np.ndarray) -> Tuple[float, np.ndarray]:
        """Objective and dL/dg.

        `idx` has shape (items, views, k) in **slot** space: idx[n, c, j] indexes the
        observed probability that view c put on option-ID slot j. `g` has shape
        (k, N+1) — row j is slot j's map.
        """
        n_items, n_views, k = idx.shape
        slots = np.arange(k)[None, None, :]
        gi = g[slots, idx]                            # (items, views, k), calibrated
        dgi = np.zeros_like(gi)
        n = float(n_items)

        # (a) each calibrated view should still be a distribution over slots
        resid = gi.sum(axis=2) - 1.0                  # (items, views)
        loss_sum = float(np.sum(resid ** 2)) / n
        dgi += (2.0 * resid / n)[:, :, None]

        # (b) the same *content* option, seen at different slots across views, should
        #     receive the same calibrated score. This is what ties the per-slot maps
        #     to each other, and what removes position bias.
        #     Under view c, content option i sits at slot (i - c) % k.
        content = np.empty_like(gi)
        back = np.empty((n_views, k), dtype=np.int64)
        for c in range(n_views):
            for i in range(k):
                back[c, i] = (i - c) % k
            content[:, c, :] = gi[:, c, back[c]]
        dev = content - content.mean(axis=1, keepdims=True)
        loss_inv = float(np.sum(dev ** 2)) / n
        for c in range(n_views):                      # scatter back to slot space;
            # back[c] is a permutation, so each slot is written exactly once
            dgi[:, c, back[c]] += 2.0 * dev[:, c, :] / n

        # (c) keep the maps away from the uniform collapse (subtracted, as in the paper)
        spread = gi - 1.0 / k
        loss_reg = float(np.sum(spread ** 2)) / n
        dgi += -2.0 * self.lam * spread / n

        dg = np.zeros_like(g)
        for j in range(k):
            np.add.at(dg[j], idx[:, :, j].reshape(-1), dgi[:, :, j].reshape(-1))

        return loss_sum + loss_inv - self.lam * loss_reg, dg

    # -- public ------------------------------------------------------------
    def fit(self, Q: np.ndarray, verbose: bool = False) -> "OrderPreservingCalibrator":
        """Fit on observed probabilities Q, shape (items, views, k) in slot space.

        Labels are never touched — this is the label-free part of the method.
        """
        k = Q.shape[2]
        values = np.unique(np.concatenate([Q.reshape(-1), [0.0, 1.0]]))
        idx = np.searchsorted(values, Q)              # exact hits by construction
        theta = np.zeros((k, len(values) - 1), dtype=np.float64)

        for epoch in range(self.epochs):
            g = self._g_from_theta(theta)
            loss, dg = self._loss_and_dg(g, idx)
            theta = theta - self.lr * self._theta_grad(theta, dg)
            theta -= theta.mean(axis=-1, keepdims=True)   # fix the per-row shift degeneracy
            if verbose:
                print(f"  epoch {epoch:3d}  loss={loss:.6f}")

        self.values_ = values
        # values[m] -> g[j, m] for slot j; the endpoints pin g(0)=0 and g(1)=1.
        mapped = self._g_from_theta(theta)
        # The isotonic step the original code applies at the end. Rows are already
        # monotone by construction, so this only guards against numerical drift.
        self.mapped_ = np.clip(np.maximum.accumulate(mapped, axis=-1), 0.0, 1.0)
        return self

    def transform(self, Q: np.ndarray) -> np.ndarray:
        """Apply each slot's map along the last axis, interpolating between values."""
        if self.values_ is None:
            raise RuntimeError("fit() first")
        Q = np.asarray(Q, dtype=np.float64)
        out = np.empty_like(Q)
        for j in range(Q.shape[-1]):
            out[..., j] = np.interp(Q[..., j], self.values_, self.mapped_[j])
        return out

    def calibrate(self, Q: np.ndarray) -> np.ndarray:
        """Apply g and renormalize each view to a distribution."""
        out = self.transform(Q)
        total = out.sum(axis=-1, keepdims=True)
        return np.divide(out, total, out=np.full_like(out, 1.0 / out.shape[-1]), where=total > 0)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(parts, calib_frac: float = 0.02, lam: float = 0.05,
             epochs: int = 30, seed: int = 0) -> dict:
    """Head-to-head numbers for one run.

    `parts` is a list of (Q, y) blocks — one per cached file. MMLU stores a file
    per subject, so the calibration prefix is taken *within each block* and then
    pooled; taking it from the concatenation instead would draw the whole
    calibration set from whichever subject sorts first. For ARC/CSQA there is a
    single block and this reduces to a plain prefix.

    The calibrator never sees labels, and every reported number is computed on the
    held-out remainder only.
    """
    if isinstance(parts, np.ndarray):  # single (Q, y) passed directly
        raise TypeError("evaluate() takes a list of (Q, y) blocks")

    calib_blocks, test_Q, test_y = [], [], []
    for Q, y in parts:
        n_calib = max(1, int(round(calib_frac * len(Q))))
        calib_blocks.append(Q[:n_calib])
        test_Q.append(Q[n_calib:])
        test_y.append(y[n_calib:])

    Q_calib = np.concatenate(calib_blocks, axis=0)
    Q_test = np.concatenate(test_Q, axis=0)
    y_test = np.concatenate(test_y, axis=0)
    n_items = sum(len(Q) for Q, _ in parts)
    n_views, k = Q_test.shape[1], Q_test.shape[2]

    cal = OrderPreservingCalibrator(lam=lam, epochs=epochs, seed=seed).fit(Q_calib)

    out = {
        "n_items_total": int(n_items),
        "n_calib": int(len(Q_calib)),
        "n_test": int(len(y_test)),
        "n_blocks": len(parts),
        "n_views": int(n_views),
        "k": int(k),
        "methods": {},
    }

    def record(name: str, preds: np.ndarray, cost: float):
        out["methods"][name] = {
            "cost": float(cost),
            "acc": accuracy(y_test, preds),
            "recall_std": recall_std(y_test, preds, k),
        }

    def content(views_slot: np.ndarray) -> np.ndarray:
        """(items, views, k) slot space -> content space, un-rotating each view."""
        return np.stack([to_content_space(item, k) for item in views_slot])

    # View 0 presents options in their original order, so slot index == content index.
    record("baseline", np.argmax(Q_test[:, 0, :], axis=1), 1.0)

    # CalibraEval at the same budget as the baseline: one view, per-slot calibrated.
    record("calibraeval@1", np.argmax(cal.calibrate(Q_test[:, 0, :]), axis=1), 1.0)

    # Full-budget ensembling, without calibration and with it.
    record("cyclic", np.argmax(content(Q_test).mean(axis=1), axis=1), float(n_views))
    record(f"calibraeval@{n_views}",
           np.argmax(content(cal.calibrate(Q_test)).mean(axis=1), axis=1), float(n_views))

    return out


def group_by_run_index(paths: Sequence[str]) -> dict:
    """Bucket cached files by the seed-run index in their name.

    ARC/CSQA store `<task>_run<r>.jsonl`; MMLU stores `<subject>_run<r>.jsonl`, one
    per subject, so all 57 subjects of a given r belong to the same run.
    """
    groups: dict = {}
    for path in paths:
        stem = os.path.basename(path).rsplit(".", 1)[0]
        marker = stem.rfind("run")
        key = stem[marker + 3:] if marker >= 0 else stem
        groups.setdefault(key, []).append(path)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_files", nargs="+", required=True,
                        help="cached <task>_run<r>.jsonl paths")
    parser.add_argument("--label", default="", help="tag written into the output json")
    parser.add_argument("--calib_frac", type=float, default=0.02)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", default=None, help="write results json here")
    args = parser.parse_args()

    groups = group_by_run_index(args.run_files)
    per_run = []
    for run_key in sorted(groups):
        files = sorted(groups[run_key])
        parts = [load_cached_run(p) for p in files]
        res = evaluate(parts, calib_frac=args.calib_frac, lam=args.lam, epochs=args.epochs)
        res["run"] = run_key
        res["n_files"] = len(files)
        per_run.append(res)
        print(f"[run{run_key}] files={len(files)} n_test={res['n_test']} views={res['n_views']} k={res['k']}")
        for name, m in res["methods"].items():
            print(f"    {name:>18s}  cost={m['cost']:.2f}  acc={m['acc']:.4f}  rstd={m['recall_std']:.4f}")

    # Average across the seed runs, which is how the paper reports these.
    summary = {}
    for name in per_run[0]["methods"]:
        summary[name] = {
            key: float(np.mean([r["methods"][name][key] for r in per_run]))
            for key in ("cost", "acc", "recall_std")
        }
    print("\n=== mean over runs ===")
    for name, m in summary.items():
        print(f"  {name:>18s}  cost={m['cost']:.2f}  acc={m['acc']:.4f}  rstd={m['recall_std']:.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "per_run": per_run, "mean": summary}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
