"""Build data_logiqa2/{dev,test}/logiqa2_{dev,test}.csv from LogiQA 2.0 (MRC, English).

Source: csitfun/LogiQA2.0 (Liu et al., TASLP 2023), license CC BY-NC-SA 4.0 (the
HF mirror datatune/LogiQA2.0 labels it MIT; the original repository's license
applies). Files: MRC/test.txt and MRC/dev.txt, one JSON object per line with
text / question / options[4] / answer (0-3).

Mirrors data_race/process.py: the passage is folded into the Question column
with its own headings, and eval_clm_utils drops the generic "Question: "
prefix for this task, so a prompt renders as
    Passage:\n...\n\nQuestion: ...\nOptions:\nA. ...   (same layout as RACE's "Article:")

Splits:
- test = the official MRC test split (1,572 items, all 4-option).
- dev = the first N_DEV items of the official MRC dev split. The canonical
  ACCEL protocol is 0-shot, so dev is never rendered.

Kept as published: options in the published order (answer key A/B/C/D =
347/384/417/424), and 14 exact duplicate items (same passage, question,
options and answer) that the official split contains.

usage: python process.py                  # downloads from the HF mirror
       python process.py <dir_with_txts>  # local test.txt / dev.txt
"""

import csv
import json
import os
import sys
import urllib.request

N_DEV = 5
OPTION_IDS = "ABCD"
MIRROR = "https://huggingface.co/datasets/datatune/LogiQA2.0/resolve/main/MRC/{}.txt"


def read_split(name):
    if len(sys.argv) > 1:
        lines = open(os.path.join(sys.argv[1], f"{name}.txt"), encoding="utf-8").read().splitlines()
    else:
        lines = urllib.request.urlopen(MIRROR.format(name)).read().decode("utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def build_rows(items):
    rows = []
    for item in items:
        options = [str(o).strip() for o in item["options"]]
        assert len(options) == 4 and all(options), item["id"]
        question = f"Passage:\n{str(item['text']).strip()}\n\nQuestion: {str(item['question']).strip()}"
        rows.append([question, *options, OPTION_IDS[int(item["answer"])]])
    return rows


def main():
    for split, src in (("dev", "dev"), ("test", "test")):
        rows = build_rows(read_split(src))
        if split == "dev":
            rows = rows[:N_DEV]
        os.makedirs(split, exist_ok=True)
        path = f"{split}/logiqa2_{split}.csv"
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"wrote {path}: {len(rows)} rows")
        if split == "test":
            print("test answer-key distribution:", {o: sum(r[-1] == o for r in rows) for o in OPTION_IDS})


if __name__ == "__main__":
    main()
