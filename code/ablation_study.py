import argparse
import json
import os
import random
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from debias_utils import simple as debias_simple
from debias_utils import cantor_expansion
from utils import (
    BAD_OPTIONS,
    REFER_OPTIONS,
    _norm,
    cyclic_shift,
    ids_in_positions_to_permuted_indices,
    shuffle_option_ids,
    shuffle_option_texts,
)


@dataclass
class Sample:
    """One MCQ sample aggregated across subjects."""

    sample_id: Tuple[str, int]  # (subject, idx)
    options: List[str]          # canonical/original option texts in order A,B,C,D,(E)
    ideal_label: str            # canonical correct label, e.g., 'A'
    observed: np.ndarray        # shape (m, n) or (n,) depending on source
    permuted_indices_list: Optional[List[Tuple[int, ...]]] = None  # for m prompts, mapping label->orig option index


def _iter_result_lines(load_path: str) -> Iterable[Tuple[str, Dict]]:
    """Yield (subject, data) for each `type=='result'` line in a results folder."""
    for record_file in sorted(os.listdir(load_path)):
        if not record_file.endswith('.jsonl'):
            continue
        subject = record_file[:-6]
        record_path = os.path.join(load_path, record_file)
        with open(record_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                if obj.get('type') != 'result':
                    continue
                yield subject, obj['data']


def _is_bad_option_set(options: List[str]) -> bool:
    if any(e in option for e in BAD_OPTIONS for option in options):
        return True
    if any(e in option.lower() for e in REFER_OPTIONS for option in options):
        return True
    return False


def load_samples(load_path: str, prob_key: str = 'probs', skip_bad_options: bool = True) -> List[Sample]:
    """Load result jsonl files into `Sample`s.

    Supports both base results (vector probs + correct) and perm/cyclic results
    (matrix probs/observed).
    """
    samples: List[Sample] = []
    for subject, data in _iter_result_lines(load_path):
        idx = int(data['idx'])
        options = [str(x) for x in data['options']]
        if skip_bad_options and _is_bad_option_set(options):
            continue

        ideal = str(data['ideal'])
        observed = np.array(data.get(prob_key))

        meta = data.get('meta', {}) or {}
        perm_list = meta.get('permuted_indices_list', None)
        if perm_list is not None:
            perm_list = [tuple(map(int, p)) for p in perm_list]

        samples.append(
            Sample(
                sample_id=(subject, idx),
                options=options,
                ideal_label=ideal,
                observed=observed,
                permuted_indices_list=perm_list,
            )
        )

    # Stable ordering for reproducibility.
    samples.sort(key=lambda s: (s.sample_id[0], s.sample_id[1]))
    return samples


def infer_permuted_indices_list(observed: np.ndarray, n_options: int) -> List[Tuple[int, ...]]:
    """Infer permuted_indices_list when it's missing (for legacy result files)."""
    m = int(observed.shape[0])
    if m == n_options:
        # cyclic shifts of identity
        base = list(range(n_options))
        return [tuple(cyclic_shift(base, k)) for k in range(n_options)]

    fact = 1
    for i in range(2, n_options + 1):
        fact *= i
    if m == fact:
        return [tuple(p) for p in sorted(permutations(range(n_options)))]

    raise ValueError(f"Cannot infer permutation list: observed has {m} rows for n={n_options}")


def probs_labels_to_canonical(probs_labels: np.ndarray, permuted_indices: Tuple[int, ...]) -> np.ndarray:
    """Reorder probs over labels into probs over canonical option indices."""
    n = len(permuted_indices)
    probs_canon = np.zeros(n, dtype=float)
    for label_idx in range(n):
        probs_canon[permuted_indices[label_idx]] = float(probs_labels[label_idx])
    return probs_canon


def bootstrap_std(corrects: List[int], n_boot: int = 1000, seed: int = 0) -> float:
    if len(corrects) == 0:
        return float('nan')
    rng = np.random.default_rng(seed)
    arr = np.asarray(corrects, dtype=int)
    accs = []
    for _ in range(n_boot):
        idxs = rng.integers(0, len(arr), size=len(arr))
        accs.append(float(arr[idxs].mean()))
    return float(np.std(accs, ddof=1))


def accuracy_from_corrects(corrects: List[int]) -> float:
    if len(corrects) == 0:
        return float('nan')
    return float(np.mean(corrects))


def find_global_worst_permutation(samples: List[Sample], option_ids: str) -> Tuple[Tuple[int, ...], float]:
    """Return (perm, acc) for the permutation that minimizes base accuracy globally."""
    n = len(option_ids)
    all_perms = [tuple(p) for p in sorted(permutations(range(n)))]

    # Build mapping from perm -> row index for each sample (assuming full perm results)
    # We use Cantor expansion (row order = sorted permutations).
    perm_to_row = {p: cantor_expansion(p) for p in all_perms}

    best_perm = None
    worst_acc = 1.0
    for p in all_perms:
        corrects = []
        row_idx = perm_to_row[p]
        for s in samples:
            obs = s.observed
            probs_labels = obs[row_idx]
            probs_canon = probs_labels_to_canonical(probs_labels, p)
            pred = int(np.argmax(probs_canon))
            ideal = option_ids.index(s.ideal_label)
            corrects.append(1 if pred == ideal else 0)
        acc = accuracy_from_corrects(corrects)
        if acc < worst_acc:
            worst_acc = acc
            best_perm = p

    assert best_perm is not None
    return best_perm, worst_acc


def select_row_for_perm(permuted_indices_list: List[Tuple[int, ...]], perm: Tuple[int, ...]) -> int:
    try:
        return permuted_indices_list.index(tuple(perm))
    except ValueError:
        # If the list is full permutations in sorted order, use Cantor expansion.
        # This is more efficient and robust.
        return cantor_expansion(tuple(perm))


def scenario_base_perm_identity(n: int) -> Tuple[int, ...]:
    return tuple(range(n))


def scenario_base_perm_swap_text(options: List[str]) -> Tuple[int, ...]:
    perm_idx, _ = shuffle_option_texts(options)
    return tuple(int(x) for x in perm_idx)


def scenario_base_perm_worst_per_sample(
    sample: Sample,
    option_ids: str,
    perm_list: List[Tuple[int, ...]],
) -> Tuple[int, ...]:
    """Pick the permutation that minimizes P(correct option) for this sample."""
    ideal = option_ids.index(sample.ideal_label)
    obs = sample.observed

    worst_p = None
    worst_prob = float('inf')

    for row_idx, p in enumerate(perm_list):
        # p[label_idx] = original option index
        # Find the label whose mapped option is the correct one.
        label_idx = p.index(ideal)
        prob_correct = float(obs[row_idx][label_idx])
        if prob_correct < worst_prob:
            worst_prob = prob_correct
            worst_p = p

    assert worst_p is not None
    return worst_p


def evaluate_base(
    samples: List[Sample],
    option_ids: str,
    base_perm_selector,
    perm_list_cache: Optional[List[Tuple[int, ...]]] = None,
) -> Tuple[float, float, int]:
    corrects: List[int] = []

    for s in samples:
        n = len(option_ids)
        perm_list = s.permuted_indices_list
        if perm_list is None:
            perm_list = perm_list_cache
        if perm_list is None:
            perm_list = infer_permuted_indices_list(s.observed, n)

        base_perm = base_perm_selector(s)
        row_idx = select_row_for_perm(perm_list, base_perm)
        probs_labels = np.array(s.observed[row_idx], dtype=float)
        probs_canon = probs_labels_to_canonical(probs_labels, base_perm)

        pred = int(np.argmax(probs_canon))
        ideal = option_ids.index(s.ideal_label)
        corrects.append(1 if pred == ideal else 0)

    acc = accuracy_from_corrects(corrects)
    std = bootstrap_std(corrects)
    return acc, std, len(corrects)


def evaluate_cyclic(
    samples: List[Sample],
    option_ids: str,
    base_perm_selector,
    perm_list_cache: Optional[List[Tuple[int, ...]]] = None,
) -> Tuple[float, float, int]:
    corrects: List[int] = []
    n = len(option_ids)

    for s in samples:
        perm_list = s.permuted_indices_list
        if perm_list is None:
            perm_list = perm_list_cache
        if perm_list is None:
            perm_list = infer_permuted_indices_list(s.observed, n)

        base_perm = base_perm_selector(s)
        cyclic_perms = [tuple(cyclic_shift(base_perm, k)) for k in range(n)]

        _, debiased, _ = debias_simple(np.array(s.observed, dtype=float), permuted_indices=cyclic_perms)
        pred = int(np.argmax(debiased))
        ideal = option_ids.index(s.ideal_label)
        corrects.append(1 if pred == ideal else 0)

    acc = accuracy_from_corrects(corrects)
    std = bootstrap_std(corrects)
    return acc, std, len(corrects)


def evaluate_pride(
    samples: List[Sample],
    option_ids: str,
    base_perm_selector,
    prefix_ratio: float = 0.05,
    n_iters: int = 20,
    seed: int = 0,
    perm_list_cache: Optional[List[Tuple[int, ...]]] = None,
) -> Tuple[float, float, int]:
    """Evaluate PriDe.

    We follow the repo's implementation philosophy:
      - Use `debias_simple` on prefix samples to estimate a global prior.
      - Debias single-query probabilities via log-ratio with that prior.

    Returns:
      mean_acc, std_over_iters, n_samples
    """
    n = len(option_ids)

    # Pre-infer permutation lists for all samples (legacy support)
    perm_lists: List[List[Tuple[int, ...]]] = []
    for s in samples:
        pl = s.permuted_indices_list
        if pl is None:
            if perm_list_cache is not None:
                pl = perm_list_cache
            else:
                pl = infer_permuted_indices_list(s.observed, n)
        perm_lists.append(pl)

    N = len(samples)
    prefix_size = max(1, int(round(N * prefix_ratio)))

    iter_accs: List[float] = []

    for it in range(n_iters):
        rng = random.Random(f"{seed}:{it}")
        all_idx = list(range(N))
        rng.shuffle(all_idx)
        prefix_idx = set(all_idx[:prefix_size])

        priors = []
        # 1) Estimate prior
        for i in prefix_idx:
            s = samples[i]
            obs = np.array(s.observed, dtype=float)

            # If obs is full permutations (e.g., 24x4), the default simple() will
            # select cyclic shifts of identity.
            #
            # If obs is already cyclic (n x n) but corresponds to *non-identity*
            # cyclic shifts (e.g., cyclic_swap_id), we must pass the per-sample
            # permutation list to align rows correctly.
            if s.permuted_indices_list is not None and obs.shape[0] == obs.shape[1]:
                _, _, prior = debias_simple(obs, permuted_indices=s.permuted_indices_list)
            else:
                _, _, prior = debias_simple(obs)
            priors.append(prior)

        prior_global = np.mean(np.stack(priors, axis=0), axis=0)

        # 2) Debias single-query probabilities for all samples
        corrects: List[int] = []
        for i, s in enumerate(samples):
            base_perm = base_perm_selector(s)
            perm_list = perm_lists[i]
            row_idx = select_row_for_perm(perm_list, base_perm)
            probs_labels = np.array(s.observed[row_idx], dtype=float)
            probs_canon = probs_labels_to_canonical(probs_labels, base_perm)

            debiased = np.log(probs_canon + 1e-10) - np.log(prior_global + 1e-10)
            pred = int(np.argmax(debiased))
            ideal = option_ids.index(s.ideal_label)
            corrects.append(1 if pred == ideal else 0)

        iter_accs.append(accuracy_from_corrects(corrects))

    return float(np.mean(iter_accs)), float(np.std(iter_accs, ddof=1)), len(samples)


def _load_callable(spec: str):
    """Load a callable from `module:function` or `/path/to/file.py:function`."""
    if ':' not in spec:
        raise ValueError("--ours_callable must be in the form module:function")
    mod, fn = spec.split(':', 1)

    if mod.endswith('.py') and os.path.exists(mod):
        import importlib.util

        spec_obj = importlib.util.spec_from_file_location('_ablation_ours', mod)
        if spec_obj is None or spec_obj.loader is None:
            raise ValueError(f"Failed to load module from {mod}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)  # type: ignore
    else:
        import importlib

        module = importlib.import_module(mod)

    if not hasattr(module, fn):
        raise ValueError(f"Callable {fn} not found in {mod}")
    return getattr(module, fn)


def evaluate_ours(
    samples: List[Sample],
    option_ids: str,
    base_perm_selector,
    ours_predict,
    perm_list_cache: Optional[List[Tuple[int, ...]]] = None,
) -> Tuple[float, float, int]:
    """Evaluate a user-provided Ours method.

    The callable is expected to implement:

        ours_predict(
            observed: np.ndarray,
            permuted_indices_list: List[Tuple[int,...]],
            base_perm: Tuple[int,...],
            option_ids: str,
            ideal_label: str,
            scenario: str,
            **kwargs,
        ) -> int | Sequence[float]

    It should return either:
      - the predicted *canonical option index* (int), or
      - a length-n score/probability vector over canonical option indices.
    """
    corrects: List[int] = []
    n = len(option_ids)

    for s in samples:
        perm_list = s.permuted_indices_list
        if perm_list is None:
            if perm_list_cache is not None:
                perm_list = perm_list_cache
            else:
                perm_list = infer_permuted_indices_list(s.observed, n)

        base_perm = base_perm_selector(s)
        pred = ours_predict(
            observed=np.array(s.observed, dtype=float),
            permuted_indices_list=perm_list,
            base_perm=tuple(base_perm),
            option_ids=option_ids,
            ideal_label=s.ideal_label,
        )

        if isinstance(pred, (list, tuple, np.ndarray)):
            pred_idx = int(np.argmax(np.asarray(pred, dtype=float)))
        else:
            pred_idx = int(pred)

        ideal = option_ids.index(s.ideal_label)
        corrects.append(1 if pred_idx == ideal else 0)

    return accuracy_from_corrects(corrects), bootstrap_std(corrects), len(corrects)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('--tasks', type=str, nargs='+', default=['mmlu', 'arc', 'csqa', 'hellaswag'])
    parser.add_argument('--model', type=str, required=True, help='Model name used in the results folder, e.g., llama-7b')
    parser.add_argument('--num_shot', type=int, default=0)

    parser.add_argument('--results_root', type=str, default='.', help='Repo root where results_<task>/... folders live')

    parser.add_argument('--prefix_ratio', type=float, default=0.05)
    parser.add_argument('--pride_iters', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--output_csv', type=str, default='ablation_results.csv')
    parser.add_argument('--make_plots', action='store_true')
    parser.add_argument('--plots_dir', type=str, default='ablation_plots')

    parser.add_argument(
        '--ours_callable',
        type=str,
        default=None,
        help=(
            "Optional: include Ours in the comparison by providing a callable as "
            "`module:function` or `/path/to/file.py:function`. "
            "See `evaluate_ours()` docstring for the expected signature."
        ),
    )

    args = parser.parse_args()
    
    # Matches the model folder naming logic in eval_clm_utils.py
    args.model_name = args.model.replace('\\', '/').split('/')[-1]

    ours_predict = None
    if args.ours_callable is not None:
        ours_predict = _load_callable(args.ours_callable)

    rows: List[Dict] = []

    for task in args.tasks:
        # Determine which result folder to use for permutation-based analysis.
        if task in ['csqa']:
            # The original repo typically uses cyclic for CSQA.
            perm_setting = 'cyclic'
            option_ids = 'ABCDE'
        else:
            perm_setting = 'perm'
            option_ids = 'ABCD'

        perm_path = os.path.join(args.results_root, f'results_{task}', f'{args.num_shot}s_{args.model_name}', f'{task}_{perm_setting}')
        if not os.path.isdir(perm_path):
            print(f"[WARN] Missing permutation results for task={task}: {perm_path}")
            print("       Skipping permutation-based scenarios (standard / worst / swap_text) for this task.")
            perm_samples: List[Sample] = []
        else:
            perm_samples = load_samples(perm_path, prob_key='probs')

        # Cache the inferred perm list for legacy results (same for all samples).
        perm_list_cache: Optional[List[Tuple[int, ...]]] = None
        if len(perm_samples) > 0 and perm_samples[0].permuted_indices_list is None:
            perm_list_cache = infer_permuted_indices_list(perm_samples[0].observed, len(option_ids))

        # --- Scenario A: Standard (identity permutation) ---
        if len(perm_samples) > 0:
            base_perm_selector = lambda s, n=len(option_ids): scenario_base_perm_identity(n)
            acc, std, n_samp = evaluate_base(perm_samples, option_ids, base_perm_selector, perm_list_cache=perm_list_cache)
            rows.append({'task': task, 'scenario': 'standard', 'method': 'base', 'accuracy': acc, 'std': std, 'n': n_samp})

            acc, std, n_samp = evaluate_cyclic(perm_samples, option_ids, base_perm_selector, perm_list_cache=perm_list_cache)
            rows.append({'task': task, 'scenario': 'standard', 'method': 'cyclic', 'accuracy': acc, 'std': std, 'n': n_samp})

            acc_m, acc_s, n_samp = evaluate_pride(
                perm_samples,
                option_ids,
                base_perm_selector,
                prefix_ratio=args.prefix_ratio,
                n_iters=args.pride_iters,
                seed=args.seed,
                perm_list_cache=perm_list_cache,
            )
            rows.append({'task': task, 'scenario': 'standard', 'method': 'pride', 'accuracy': acc_m, 'std': acc_s, 'n': n_samp})

            if ours_predict is not None:
                acc, std, n_samp = evaluate_ours(perm_samples, option_ids, base_perm_selector, ours_predict, perm_list_cache=perm_list_cache)
                rows.append({'task': task, 'scenario': 'standard', 'method': 'ours', 'accuracy': acc, 'std': std, 'n': n_samp})

        # --- Scenario B1: Worst permutation (global) ---
        if len(perm_samples) > 0 and perm_setting == 'perm':
            worst_p, worst_acc = find_global_worst_permutation(perm_samples, option_ids)

            base_perm_selector = lambda s, p=worst_p: p
            acc, std, n_samp = evaluate_base(perm_samples, option_ids, base_perm_selector, perm_list_cache=perm_list_cache)
            rows.append({'task': task, 'scenario': f'worst_perm_global_{worst_p}', 'method': 'base', 'accuracy': acc, 'std': std, 'n': n_samp})

            acc, std, n_samp = evaluate_cyclic(perm_samples, option_ids, base_perm_selector, perm_list_cache=perm_list_cache)
            rows.append({'task': task, 'scenario': f'worst_perm_global_{worst_p}', 'method': 'cyclic', 'accuracy': acc, 'std': std, 'n': n_samp})

            acc_m, acc_s, n_samp = evaluate_pride(
                perm_samples,
                option_ids,
                base_perm_selector,
                prefix_ratio=args.prefix_ratio,
                n_iters=args.pride_iters,
                seed=args.seed,
                perm_list_cache=perm_list_cache,
            )
            rows.append({'task': task, 'scenario': f'worst_perm_global_{worst_p}', 'method': 'pride', 'accuracy': acc_m, 'std': acc_s, 'n': n_samp})

            if ours_predict is not None:
                acc, std, n_samp = evaluate_ours(perm_samples, option_ids, base_perm_selector, ours_predict, perm_list_cache=perm_list_cache)
                rows.append({'task': task, 'scenario': f'worst_perm_global_{worst_p}', 'method': 'ours', 'accuracy': acc, 'std': std, 'n': n_samp})

        # --- Scenario B2: Worst permutation (per sample) ---
        if len(perm_samples) > 0 and perm_setting == 'perm':
            # Precompute perm list once.
            perm_list = perm_samples[0].permuted_indices_list or perm_list_cache
            if perm_list is None:
                perm_list = infer_permuted_indices_list(perm_samples[0].observed, len(option_ids))

            base_perm_selector = lambda s, option_ids=option_ids, perm_list=perm_list: scenario_base_perm_worst_per_sample(s, option_ids, perm_list)

            acc, std, n_samp = evaluate_base(perm_samples, option_ids, base_perm_selector, perm_list_cache=perm_list_cache)
            rows.append({'task': task, 'scenario': 'worst_perm_per_sample', 'method': 'base', 'accuracy': acc, 'std': std, 'n': n_samp})

            acc, std, n_samp = evaluate_cyclic(perm_samples, option_ids, base_perm_selector, perm_list_cache=perm_list_cache)
            rows.append({'task': task, 'scenario': 'worst_perm_per_sample', 'method': 'cyclic', 'accuracy': acc, 'std': std, 'n': n_samp})

            acc_m, acc_s, n_samp = evaluate_pride(
                perm_samples,
                option_ids,
                base_perm_selector,
                prefix_ratio=args.prefix_ratio,
                n_iters=args.pride_iters,
                seed=args.seed,
                perm_list_cache=perm_list_cache,
            )
            rows.append({'task': task, 'scenario': 'worst_perm_per_sample', 'method': 'pride', 'accuracy': acc_m, 'std': acc_s, 'n': n_samp})

            if ours_predict is not None:
                acc, std, n_samp = evaluate_ours(perm_samples, option_ids, base_perm_selector, ours_predict, perm_list_cache=perm_list_cache)
                rows.append({'task': task, 'scenario': 'worst_perm_per_sample', 'method': 'ours', 'accuracy': acc, 'std': std, 'n': n_samp})

        # --- Scenario C: Swap option text only (IDs fixed; answer changes) ---
        # NOTE: This scenario is only meaningful if the model is actually queried with prompts
        # whose option *texts* are swapped while the displayed IDs remain in A/B/C/D order.
        #
        # In this repo, `setting=perm` already runs exactly those swapped-text prompts for
        # *all* permutations. So, if you have perm results, you can "reuse" them by selecting
        # a particular permutation row.
        #
        # However, running full permutations is expensive. If you run `setting=cyclic_swap_text`,
        # we will prefer that folder here (only n prompts per sample, e.g. 4x for 4-choice).

        swap_text_path = os.path.join(
            args.results_root,
            f'results_{task}',
            f'{args.num_shot}s_{args.model_name}',
            f'{task}_cyclic_swap_text',
        )

        if os.path.isdir(swap_text_path):
            swap_text_samples = load_samples(swap_text_path, prob_key='probs')
            base_perm_selector = lambda s: (s.permuted_indices_list[0] if s.permuted_indices_list else scenario_base_perm_identity(len(option_ids)))

            acc, std, n_samp = evaluate_base(swap_text_samples, option_ids, base_perm_selector)
            rows.append({'task': task, 'scenario': 'swap_text', 'method': 'base', 'accuracy': acc, 'std': std, 'n': n_samp})

            # cyclic debias for swap_text: the observed matrix is already cyclic and meta holds the perm list.
            corrects: List[int] = []
            for s in swap_text_samples:
                perms = s.permuted_indices_list
                if perms is None:
                    perms = infer_permuted_indices_list(s.observed, len(option_ids))
                _, debiased, _ = debias_simple(np.array(s.observed, dtype=float), permuted_indices=perms)
                pred = int(np.argmax(debiased))
                ideal = option_ids.index(s.ideal_label)
                corrects.append(1 if pred == ideal else 0)
            rows.append({
                'task': task,
                'scenario': 'swap_text',
                'method': 'cyclic',
                'accuracy': accuracy_from_corrects(corrects),
                'std': bootstrap_std(corrects),
                'n': len(corrects),
            })

            acc_m, acc_s, n_samp = evaluate_pride(
                swap_text_samples,
                option_ids,
                base_perm_selector,
                prefix_ratio=args.prefix_ratio,
                n_iters=args.pride_iters,
                seed=args.seed,
            )
            rows.append({'task': task, 'scenario': 'swap_text', 'method': 'pride', 'accuracy': acc_m, 'std': acc_s, 'n': n_samp})

            if ours_predict is not None:
                acc, std, n_samp = evaluate_ours(swap_text_samples, option_ids, base_perm_selector, ours_predict)
                rows.append({'task': task, 'scenario': 'swap_text', 'method': 'ours', 'accuracy': acc, 'std': std, 'n': n_samp})

        elif len(perm_samples) > 0 and perm_setting == 'perm':
            # Fallback: reuse full-permutation results by selecting the deterministic shuffle permutation.
            base_perm_selector = lambda s: scenario_base_perm_swap_text(s.options)

            acc, std, n_samp = evaluate_base(perm_samples, option_ids, base_perm_selector, perm_list_cache=perm_list_cache)
            rows.append({'task': task, 'scenario': 'swap_text', 'method': 'base', 'accuracy': acc, 'std': std, 'n': n_samp})

            acc, std, n_samp = evaluate_cyclic(perm_samples, option_ids, base_perm_selector, perm_list_cache=perm_list_cache)
            rows.append({'task': task, 'scenario': 'swap_text', 'method': 'cyclic', 'accuracy': acc, 'std': std, 'n': n_samp})

            acc_m, acc_s, n_samp = evaluate_pride(
                perm_samples,
                option_ids,
                base_perm_selector,
                prefix_ratio=args.prefix_ratio,
                n_iters=args.pride_iters,
                seed=args.seed,
                perm_list_cache=perm_list_cache,
            )
            rows.append({'task': task, 'scenario': 'swap_text', 'method': 'pride', 'accuracy': acc_m, 'std': acc_s, 'n': n_samp})

            if ours_predict is not None:
                acc, std, n_samp = evaluate_ours(perm_samples, option_ids, base_perm_selector, ours_predict, perm_list_cache=perm_list_cache)
                rows.append({'task': task, 'scenario': 'swap_text', 'method': 'ours', 'accuracy': acc, 'std': std, 'n': n_samp})

        # --- Scenario D: Swap option ID only (texts fixed; answer changes) ---
        # This requires running `eval_clm.py` with `--setting cyclic_swap_id` for the task.
        swap_id_path = os.path.join(args.results_root, f'results_{task}', f'{args.num_shot}s_{args.model_name}', f'{task}_cyclic_swap_id')
        if os.path.isdir(swap_id_path):
            swap_id_samples = load_samples(swap_id_path, prob_key='probs')

            # For swap_id, the observed matrix is already cyclic (n x n) and meta contains permuted_indices_list.
            # Base permutation is always the first row's permutation.
            base_perm_selector = lambda s: (s.permuted_indices_list[0] if s.permuted_indices_list else scenario_base_perm_identity(len(option_ids)))

            acc, std, n_samp = evaluate_base(swap_id_samples, option_ids, base_perm_selector)
            rows.append({'task': task, 'scenario': 'swap_id', 'method': 'base', 'accuracy': acc, 'std': std, 'n': n_samp})

            # cyclic debias uses the already-cyclic observations (just pass perm list)
            corrects: List[int] = []
            for s in swap_id_samples:
                perms = s.permuted_indices_list
                if perms is None:
                    perms = infer_permuted_indices_list(s.observed, len(option_ids))
                _, debiased, _ = debias_simple(np.array(s.observed, dtype=float), permuted_indices=perms)
                pred = int(np.argmax(debiased))
                ideal = option_ids.index(s.ideal_label)
                corrects.append(1 if pred == ideal else 0)
            rows.append({
                'task': task,
                'scenario': 'swap_id',
                'method': 'cyclic',
                'accuracy': accuracy_from_corrects(corrects),
                'std': bootstrap_std(corrects),
                'n': len(corrects),
            })

            # Pride on swap_id: estimate prior from cyclic observations for prefix samples
            # (we reuse evaluate_pride, but it needs correct base_perm_selector and perm lists).
            acc_m, acc_s, n_samp = evaluate_pride(
                swap_id_samples,
                option_ids,
                base_perm_selector,
                prefix_ratio=args.prefix_ratio,
                n_iters=args.pride_iters,
                seed=args.seed,
            )
            rows.append({'task': task, 'scenario': 'swap_id', 'method': 'pride', 'accuracy': acc_m, 'std': acc_s, 'n': n_samp})

            if ours_predict is not None:
                acc, std, n_samp = evaluate_ours(swap_id_samples, option_ids, base_perm_selector, ours_predict)
                rows.append({'task': task, 'scenario': 'swap_id', 'method': 'ours', 'accuracy': acc, 'std': std, 'n': n_samp})
        else:
            print(f"[INFO] swap_id results not found for task={task}. To enable: run eval_clm with setting=cyclic_swap_id.")

        # --- PriDe-style ablations ---
        # Option-ID shuffle ablation (token bias vs position bias disentangling)
        shuffle_path = os.path.join(args.results_root, f'results_{task}', f'{args.num_shot}s_{args.model_name}', f'{task}_shuffle_both')
        if os.path.isdir(shuffle_path):
            # Base results already contain correctness.
            # We re-load and compute accuracy from the `correct` field.
            corrects = []
            for _, data in _iter_result_lines(shuffle_path):
                options = [str(x) for x in data['options']]
                if _is_bad_option_set(options):
                    continue
                corrects.append(1 if data.get('correct') else 0)
            rows.append({
                'task': task,
                'scenario': 'option_id_shuffle',
                'method': 'base',
                'accuracy': accuracy_from_corrects(corrects),
                'std': bootstrap_std(corrects),
                'n': len(corrects),
            })

        remove_path = os.path.join(args.results_root, f'results_{task}', f'{args.num_shot}s_{args.model_name}', f'{task}_noid')
        if os.path.isdir(remove_path):
            corrects = []
            for _, data in _iter_result_lines(remove_path):
                options = [str(x) for x in data['options']]
                if _is_bad_option_set(options):
                    continue
                corrects.append(1 if data.get('correct') else 0)
            rows.append({
                'task': task,
                'scenario': 'option_id_remove',
                'method': 'base',
                'accuracy': accuracy_from_corrects(corrects),
                'std': bootstrap_std(corrects),
                'n': len(corrects),
            })

    # --- Save CSV ---
    import pandas as pd

    df = pd.DataFrame(rows)
    # Add run metadata columns.
    df['model'] = args.model
    df['num_shot'] = args.num_shot
    df.to_csv(args.output_csv, index=False)
    print(f"Saved CSV to {args.output_csv}")

    # --- Plots ---
    if args.make_plots:
        if 'task' not in df.columns or len(df) == 0:
            print("[WARN] No rows available for plotting. Skipping plot generation.")
            return
        os.makedirs(args.plots_dir, exist_ok=True)
        import matplotlib.pyplot as plt

        # Plot per task: scenario x method bar chart
        for task in sorted(df['task'].unique()):
            sub = df[df['task'] == task].copy()
            scenarios = list(sorted(sub['scenario'].unique()))
            methods = list(sorted(sub['method'].unique()))

            x = np.arange(len(scenarios))
            width = 0.8 / max(1, len(methods))

            fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 1.2), 4))
            for i, m in enumerate(methods):
                msub = sub[sub['method'] == m]
                # Align by scenario
                ys = []
                es = []
                for sc in scenarios:
                    row = msub[msub['scenario'] == sc]
                    if len(row) == 0:
                        ys.append(np.nan)
                        es.append(0.0)
                    else:
                        ys.append(float(row.iloc[0]['accuracy']))
                        es.append(float(row.iloc[0]['std']))

                ax.bar(x + i * width, ys, width, yerr=es, label=m)

            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(scenarios, rotation=30, ha='right')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Accuracy')
            ax.set_title(f"{task} ablation study ({args.model}, {args.num_shot}-shot)")
            ax.legend()
            fig.tight_layout()

            out_path = os.path.join(args.plots_dir, f"{task}_ablation.png")
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

        print(f"Saved plots to {args.plots_dir}/")


if __name__ == '__main__':
    main()
