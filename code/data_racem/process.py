"""Convert RACE into the headerless CSV layout the eval loader expects.

RACE (Lai et al., 2017) is a reading-comprehension MCQ benchmark: each item is a
passage plus a question with 4 options. The loader (``eval_clm_utils.py``) reads a
single ``Question`` text column, so the passage is folded into that column as

    Article:
    <passage>

    Question: <question>

and ``eval_clm_utils`` skips its usual ``Question: `` prefix for this task (see
``question_prefix`` there).

Output matches ``data_arc`` / ``data_csqa``:
    dev/race_dev.csv    few-shot pool (first N validation items)
    test/race_test.csv  evaluation set
Columns are positional, no header: Question, A, B, C, D, Answer.

Usage (default reproduces the NeurIPS 2026 rebuttal P1-3 setup, RACE middle split for task 'racem'):
    python process.py
    python process.py --subset all --limit 1200 --seed 42
"""

import argparse
import csv
import os
import random

HF_DATASET = "ehovy/race"
NUM_OPTIONS = 4
OPTION_IDS = "ABCD"


def collapse_ws(text: str) -> str:
    """Flatten to a single line.

    The prompt puts the question on one line and one option per line, so a stray
    newline inside either would break that structure (and, for an option, shift
    the option-ID token the scorer reads). Passage newlines are kept — the
    article is meant to render as paragraphs.
    """
    return " ".join(str(text).split())


def build_question(article: str, question: str) -> str:
    return f"Article:\n{article.strip()}\n\nQuestion: {collapse_ws(question)}"


def to_row(sample):
    """Return a CSV row, or None if the item is malformed."""
    options = [collapse_ws(o) for o in sample["options"]]
    if len(options) != NUM_OPTIONS or any(not o for o in options):
        return None
    answer = str(sample["answer"]).strip().upper()
    if answer not in OPTION_IDS:
        return None
    return [build_question(sample["article"], sample["question"]), *options, answer]


def write_split(rows, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    # newline="" keeps embedded newlines in the passage quoted rather than mangled.
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
        # Fixed seed so the subsample is reproducible across machines and reruns.
        rows = random.Random(seed).sample(rows, limit)
    return rows, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="middle", choices=["high", "middle", "all"],
                        help="RACE config: high (default, RACE-H), middle, or all (M+H)")
    parser.add_argument("--limit", type=int, default=None,
                        help="subsample the test split to this many items (default: keep all)")
    parser.add_argument("--seed", type=int, default=42, help="subsampling seed")
    parser.add_argument("--num_few_shot_pool", type=int, default=5,
                        help="rows written to dev/ as the few-shot pool (matches data_arc)")
    args = parser.parse_args()

    # Imported here, not at module scope, so the row-building helpers above stay
    # importable (and unit-testable) without the HF datasets dependency.
    from datasets import load_dataset

    here = os.path.dirname(os.path.abspath(__file__))

    test_rows, skipped = collect(
        load_dataset(HF_DATASET, args.subset, split="test"), args.limit, args.seed
    )
    test_path = write_split(test_rows, os.path.join(here, "test"), "racem_test.csv")
    print(f"test : {len(test_rows)} rows (skipped {skipped}) -> {test_path}")

    dev_rows, dev_skipped = collect(
        load_dataset(HF_DATASET, args.subset, split="validation"), args.num_few_shot_pool, args.seed
    )
    dev_path = write_split(dev_rows, os.path.join(here, "dev"), "racem_dev.csv")
    print(f"dev  : {len(dev_rows)} rows (skipped {dev_skipped}) -> {dev_path}")


if __name__ == "__main__":
    main()
