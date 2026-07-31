"""Convert OpenBookQA into the headerless CSV layout the eval loader expects.

OpenBookQA (Mihaylov et al., 2018) is a 4-option elementary-science MCQ
benchmark in the ARC family (same option count, same science-fact style, no
per-item subject structure). It joins the CGES routing-baseline study
(NeurIPS 2026 rebuttal, g9SK-W3 follow-up) as the *homogeneous* companion
benchmark: the working hypothesis is that fixed-τ posterior-mass halting
matches the online-percentile gate on ARC-like benches and loses on
heterogeneous ones.

Output matches ``data_arc`` / ``data_race``:
    dev/obqa_dev.csv    few-shot pool (from the validation split)
    test/obqa_test.csv  evaluation set (test split, 500 items)
Columns are positional, no header: Question, A, B, C, D, Answer.

Usage:
    python process.py
"""

import argparse
import csv
import os
import random

HF_DATASET = "allenai/openbookqa"
HF_CONFIG = "main"
NUM_OPTIONS = 4
OPTION_IDS = "ABCD"


def collapse_ws(text: str) -> str:
    """Flatten to a single line (see data_race/process.py for why)."""
    return " ".join(str(text).split())


def to_row(sample):
    """Return a CSV row, or None if the item is malformed."""
    labels = [str(l).strip().upper() for l in sample["choices"]["label"]]
    texts = [collapse_ws(t) for t in sample["choices"]["text"]]
    if len(labels) != NUM_OPTIONS or any(not t for t in texts):
        return None
    # Options are stored in label order; reorder defensively anyway.
    by_label = dict(zip(labels, texts))
    if sorted(by_label) != list(OPTION_IDS):
        return None
    answer = str(sample["answerKey"]).strip().upper()
    if answer not in OPTION_IDS:
        return None
    return [collapse_ws(sample["question_stem"]), *[by_label[c] for c in OPTION_IDS], answer]


def write_split(rows, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


def collect(dataset, limit=None, seed=42):
    rows, skipped = [], 0
    for sample in dataset:
        row = to_row(sample)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="subsample the test split (default: keep all 500)")
    parser.add_argument("--seed", type=int, default=42, help="subsampling seed")
    parser.add_argument("--num_few_shot_pool", type=int, default=5,
                        help="rows written to dev/ as the few-shot pool (matches data_arc)")
    args = parser.parse_args()

    from datasets import load_dataset

    here = os.path.dirname(os.path.abspath(__file__))

    test_rows, skipped = collect(
        load_dataset(HF_DATASET, HF_CONFIG, split="test"), args.limit, args.seed
    )
    test_path = write_split(test_rows, os.path.join(here, "test"), "obqa_test.csv")
    print(f"test : {len(test_rows)} rows (skipped {skipped}) -> {test_path}")

    dev_rows, dev_skipped = collect(
        load_dataset(HF_DATASET, HF_CONFIG, split="validation"),
        args.num_few_shot_pool, args.seed,
    )
    dev_path = write_split(dev_rows, os.path.join(here, "dev"), "obqa_dev.csv")
    print(f"dev  : {len(dev_rows)} rows (skipped {dev_skipped}) -> {dev_path}")


if __name__ == "__main__":
    main()
