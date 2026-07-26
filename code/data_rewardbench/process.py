"""Convert RewardBench into the headerless CSV layout the eval loader expects.

Why this dataset
----------------
CalibraEval (ACL 2025) is evaluated on pairwise preference judging: a query plus
two candidate answers, pick the better one. Its `src/process_data.py` reads
records with `prompt` / `chosen` / `rejected` / `chosen_model` / `rejected_model`
/ `subset` / `id`, which is the RewardBench schema. Running ACCEL on that
benchmark is the mirror image of `code/calibraeval_mcq.py`: instead of adapting
their method to our 4-option setting, it meets their method on its own ground,
where the position bias is exactly the A-vs-B preference the paper targets.

It is a k=2 task. The loader picks that up from the `rewardbench` branch in
`eval_clm_utils.prepare_eval`; cyclic rotation and the targeted Latin schedule
both degrade correctly to two views (identity + swap).

Layout matches data_arc / data_csqa / data_race:
    dev/rewardbench_dev.csv    few-shot pool
    test/rewardbench_test.csv  evaluation set
Columns are positional, no header: Question, A, B, Answer.

Which slot holds the better answer is randomized per item with a fixed seed, so
the gold labels are balanced rather than always "A" — otherwise a model that
always answers A would score 100%.

Usage:
    python process.py                      # full test split
    python process.py --limit 1200         # match the ARC/CSQA scale
"""

import argparse
import csv
import os
import random

HF_DATASET = "allenai/reward-bench"
OPTION_IDS = "AB"


def collapse_ws(text: str) -> str:
    """Flatten to one line: the prompt puts one option per line."""
    return " ".join(str(text).split())


def to_row(sample, rng):
    """Return a CSV row, or None if the item is malformed.

    The better answer is placed in slot A or B at random so the label
    distribution stays balanced.
    """
    query = collapse_ws(sample.get("prompt", ""))
    chosen = collapse_ws(sample.get("chosen", ""))
    rejected = collapse_ws(sample.get("rejected", ""))
    if not query or not chosen or not rejected or chosen == rejected:
        return None
    if rng.random() < 0.5:
        return [query, chosen, rejected, "A"]
    return [query, rejected, chosen, "B"]


def collect(dataset, limit=None, seed=42):
    rng = random.Random(seed)
    rows, skipped = [], 0
    for sample in dataset:
        row = to_row(sample, rng)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows, skipped


def write_split(rows, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="subsample the test split (default: keep all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_few_shot_pool", type=int, default=5)
    parser.add_argument("--split", default="filtered",
                        help="RewardBench split to read (it ships a single 'filtered' split)")
    args = parser.parse_args()

    # Imported here so the row helpers stay importable (and testable) without the
    # HF datasets dependency.
    from datasets import load_dataset

    here = os.path.dirname(os.path.abspath(__file__))
    data = list(load_dataset(HF_DATASET, split=args.split))

    # Hold the few-shot pool out of the evaluation set.
    dev_src, test_src = data[: args.num_few_shot_pool], data[args.num_few_shot_pool:]

    test_rows, skipped = collect(test_src, args.limit, args.seed)
    test_path = write_split(test_rows, os.path.join(here, "test"), "rewardbench_test.csv")
    print(f"test : {len(test_rows)} rows (skipped {skipped}) -> {test_path}")

    dev_rows, dev_skipped = collect(dev_src, args.num_few_shot_pool, args.seed)
    dev_path = write_split(dev_rows, os.path.join(here, "dev"), "rewardbench_dev.csv")
    print(f"dev  : {len(dev_rows)} rows (skipped {dev_skipped}) -> {dev_path}")

    balance = {oid: sum(1 for r in test_rows if r[-1] == oid) for oid in OPTION_IDS}
    print(f"label balance: {balance}")


if __name__ == "__main__":
    main()
