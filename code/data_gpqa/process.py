"""Build data_gpqa/{dev,test}/gpqa_{dev,test}.csv from Idavidrein/gpqa.

Mirrors data_arc/process.py: emit headerless CSV rows
[Question, A, B, C, D, Answer] that prepare_eval() reads via
    pd.read_csv(..., names=("Question", "A", "B", "C", "D", "Answer"))

Config: gpqa_main (448 items). PA-GRPO's Appendix C does not say which GPQA
subset it used ("curated subsets ... including a higher-quality, harder
subset"), so we take the standard reference set rather than Diamond.

Two things that matter for a bias study:

1. GPQA stores the answer as "Correct Answer" + "Incorrect Answer 1..3", i.e.
   the correct option is ALWAYS first. Writing them in that order would put the
   answer key at 'A' for every item and make recall-balance metrics meaningless.
   Each item therefore gets a deterministic per-index shuffle (seeded by
   SHUFFLE_SEED + row index) so the answer key is spread across A-D and the
   file is reproducible.

2. GPQA ships a single split. ARC draws few-shot examples from its official Dev
   file, so we carve the first N_DEV items off for dev and keep the rest as
   test. The canonical ACCEL protocol is 0-shot, where the dev rows are never
   rendered into the prompt, but carving them out keeps the split leak-free if
   anyone later runs few-shot. Cost: test is 443 items instead of 448.

The GPQA card asks that the canary string not be redistributed; only Question
and the four option texts are written out.
"""

import csv
import os
import random

from datasets import load_dataset

CONFIG = "gpqa_main"
SHUFFLE_SEED = 20260802
N_DEV = 5

OPTION_IDS = "ABCD"
INCORRECT_KEYS = ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]


def build_rows(dataset):
    rows = []
    for idx, item in enumerate(dataset):
        question = " ".join(str(item["Question"]).split())
        correct = " ".join(str(item["Correct Answer"]).split())
        options = [correct] + [" ".join(str(item[k]).split()) for k in INCORRECT_KEYS]

        if any(not o for o in options) or not question:
            print(f"skip {idx}: empty field")
            continue

        order = list(range(4))
        random.Random(SHUFFLE_SEED + idx).shuffle(order)
        shuffled = [options[i] for i in order]
        answer = OPTION_IDS[order.index(0)]

        rows.append([question, *shuffled, answer])
    return rows


def main():
    data = load_dataset("Idavidrein/gpqa", CONFIG)["train"]
    rows = build_rows(data)

    for split, subset in (("dev", rows[:N_DEV]), ("test", rows[N_DEV:])):
        os.makedirs(split, exist_ok=True)
        path = f"{split}/gpqa_{split}.csv"
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerows(subset)
        print(f"wrote {path}: {len(subset)} rows")

    dist = {oid: sum(r[5] == oid for r in rows[N_DEV:]) for oid in OPTION_IDS}
    print(f"test answer-key distribution: {dist}")


if __name__ == "__main__":
    main()
