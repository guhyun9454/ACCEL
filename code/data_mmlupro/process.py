"""Build data_mmlupro/{dev,test}/mmlupro_{dev,test}.csv from TIGER-Lab/MMLU-Pro.

Mirrors data_arc/process.py, but with k=10: headerless CSV rows
[Question, A, B, C, D, E, F, G, H, I, J, Answer], read by prepare_eval() with
option_ids_header = A..J.

Splits:
- test = the official test split, restricted to items with exactly 10 options
  (9,981 of 12,032). The cyclic / Latin permutation machinery needs a fixed k;
  the 2,051 items with 3-9 options (mostly from the original MMLU) are dropped.
- dev = the first N_DEV items of the official validation split (70 items, all
  10-option). The canonical ACCEL protocol is 0-shot, so dev is never rendered.

Everything goes into one file (micro-average, the MMLU-Pro reporting
convention). The source is sorted by category, but eval_clm.py shuffles the
item order per run (seed run_idx + 42), so the online prefix and percentile
gate do not see category blocks.

Text is kept as published apart from stripping leading/trailing whitespace:
558 questions contain newlines (code, tables), which carry meaning. Options
keep the published order; the answer key is already balanced (945-1,048 per
letter). 16 items have two options with identical text and 7 options are a
literal "None"-like string; both are kept (read with keep_default_na=False).

usage: python process.py                 # via `datasets`
       python process.py <parquet_dir>   # from local {test,validation}-*.parquet
"""

import csv
import glob
import os
import sys

K = 10
N_DEV = 5
OPTION_IDS = "ABCDEFGHIJ"


def build_rows(items):
    rows, skipped = [], 0
    for item in items:
        options = [str(o).strip() for o in item["options"]]
        question = str(item["question"]).strip()
        if len(options) != K:
            skipped += 1
            continue
        assert OPTION_IDS[int(item["answer_index"])] == item["answer"], item["question_id"]
        if not question or any(not o for o in options):
            print(f"skip {item['question_id']}: empty field")
            skipped += 1
            continue
        rows.append([question, *options, item["answer"]])
    return rows, skipped


def load_splits():
    if len(sys.argv) > 1:
        import pyarrow.parquet as pq
        read = lambda s: pq.read_table(glob.glob(os.path.join(sys.argv[1], f"{s}-*.parquet"))[0]).to_pylist()
        return read("test"), read("validation")
    from datasets import load_dataset
    data = load_dataset("TIGER-Lab/MMLU-Pro")
    return data["test"], data["validation"]


def main():
    test_items, dev_items = load_splits()
    for split, items in (("dev", dev_items), ("test", test_items)):
        rows, skipped = build_rows(items)
        if split == "dev":
            rows = rows[:N_DEV]
        os.makedirs(split, exist_ok=True)
        path = f"{split}/mmlupro_{split}.csv"
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"wrote {path}: {len(rows)} rows (skipped {skipped} non-{K}-option items)")
        if split == "test":
            print("test answer-key distribution:", {o: sum(r[-1] == o for r in rows) for o in OPTION_IDS})


if __name__ == "__main__":
    main()
