"""Convert MedMCQA into the headerless CSV layout the eval loader expects.

MedMCQA (Pal et al., 2022) is a 4-option medical-entrance MCQ benchmark
spanning 21 medical subjects. It joins the CGES routing-baseline study
(NeurIPS 2026 rebuttal, g9SK-W3 follow-up) as the *heterogeneous* companion
benchmark (the MMLU-like side of the homogeneous/heterogeneous contrast;
OpenBookQA is the ARC-like side).

We use the labeled ``validation`` split (the ``test`` split ships without
answers) and keep only ``choice_type == "single"`` items with a valid
correct-option index.

Output matches ``data_arc`` / ``data_race``:
    dev/medmcqa_dev.csv    few-shot pool (from the train split)
    test/medmcqa_test.csv  evaluation set (validation split)
Columns are positional, no header: Question, A, B, C, D, Answer.

Usage:
    python process.py
"""

import argparse
import csv
import os
import random

HF_DATASET = "openlifescienceai/medmcqa"
NUM_OPTIONS = 4
OPTION_IDS = "ABCD"


def collapse_ws(text: str) -> str:
    """Flatten to a single line (see data_race/process.py for why)."""
    return " ".join(str(text).split())


def to_row(sample):
    """Return a CSV row, or None if the item is malformed."""
    if str(sample.get("choice_type", "single")).strip().lower() != "single":
        return None
    options = [collapse_ws(sample[k]) for k in ("opa", "opb", "opc", "opd")]
    if any(not o for o in options):
        return None
    try:
        cop = int(sample["cop"])
    except (TypeError, ValueError):
        return None
    if not 0 <= cop < NUM_OPTIONS:
        return None
    question = collapse_ws(sample["question"])
    if not question:
        return None
    return [question, *options, OPTION_IDS[cop]]


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
                        help="subsample the eval split (default: keep all)")
    parser.add_argument("--seed", type=int, default=42, help="subsampling seed")
    parser.add_argument("--num_few_shot_pool", type=int, default=5,
                        help="rows written to dev/ as the few-shot pool (matches data_arc)")
    args = parser.parse_args()

    from datasets import load_dataset

    here = os.path.dirname(os.path.abspath(__file__))

    test_rows, skipped = collect(
        load_dataset(HF_DATASET, split="validation"), args.limit, args.seed
    )
    test_path = write_split(test_rows, os.path.join(here, "test"), "medmcqa_test.csv")
    print(f"test : {len(test_rows)} rows (skipped {skipped}) -> {test_path}")

    dev_rows, dev_skipped = collect(
        load_dataset(HF_DATASET, split="train"), args.num_few_shot_pool, args.seed
    )
    dev_path = write_split(dev_rows, os.path.join(here, "dev"), "medmcqa_dev.csv")
    print(f"dev  : {len(dev_rows)} rows (skipped {dev_skipped}) -> {dev_path}")


if __name__ == "__main__":
    main()
