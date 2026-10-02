"""Build data_medmcqa/{dev,test}/medmcqa_{dev,test}.csv from openlifescienceai/medmcqa.

Mirrors data_arc/process.py: emit headerless CSV rows
[Question, A, B, C, D, Answer] that prepare_eval() reads via
    pd.read_csv(..., names=("Question", "A", "B", "C", "D", "Answer"))

Splits:
- test = the official validation split (4,183 items). The official test split
  ships without answer keys (cop = -1), so validation is the standard
  evaluation set.
- dev = the first N_DEV items of the official train split. The canonical ACCEL
  protocol is 0-shot, where dev rows are never rendered into the prompt, but a
  separate split keeps few-shot runs leak-free.

Everything goes into one file (micro-average, like ARC/CSQA). MedMCQA has 21
subjects, but several have fewer than 50 validation items, which is too few for
a per-subject 2% PriDe prefix.

Options are kept in the published order, as in ARC. The answer key is
therefore NOT balanced: validation is A 1348 / B 1085 / C 925 / D 825.
21 items have two options with identical text; they are kept as published.

usage: python process.py                 # via `datasets`
       python process.py <parquet_dir>   # from local {validation,train}-*.parquet
"""

import csv
import glob
import os
import sys

N_DEV = 5
OPTION_IDS = "ABCD"
OPTION_KEYS = ["opa", "opb", "opc", "opd"]


def clean(text):
    return " ".join(str(text).split())


def build_rows(items):
    rows = []
    for idx, item in enumerate(items):
        question = clean(item["question"])
        options = [clean(item[k]) for k in OPTION_KEYS]
        cop = int(item["cop"])
        if not question or any(not o for o in options) or cop not in range(4):
            print(f"skip {idx}: empty field or cop={cop}")
            continue
        rows.append([question, *options, OPTION_IDS[cop]])
    return rows


def load_splits():
    if len(sys.argv) > 1:
        import pyarrow.parquet as pq
        read = lambda s: pq.read_table(glob.glob(os.path.join(sys.argv[1], f"{s}-*.parquet"))[0]).to_pylist()
        return read("validation"), read("train")[:N_DEV]
    from datasets import load_dataset
    data = load_dataset("openlifescienceai/medmcqa")
    return data["validation"], data["train"].select(range(N_DEV))


def main():
    test_items, dev_items = load_splits()
    for split, items in (("dev", dev_items), ("test", test_items)):
        rows = build_rows(items)
        os.makedirs(split, exist_ok=True)
        path = f"{split}/medmcqa_{split}.csv"
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"wrote {path}: {len(rows)} rows")
        if split == "test":
            print("test answer-key distribution:", {o: sum(r[5] == o for r in rows) for o in OPTION_IDS})


if __name__ == "__main__":
    main()
