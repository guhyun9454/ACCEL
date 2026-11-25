"""X-CSQA (Lin et al. 2021, "Common Sense Beyond English") -> data_xcsqa_<lang>/{dev,test}/xcsqa_<lang>_{dev,test}.csv.

Source: HF dataset INK-USC/xcsr, revision f691f311a11f84773a899e9bfc7da2bae51b0b02, files
X-CSQA-<lang>/validation-00000-of-00001.parquet (1,000 Q per language; the test split's answerKey is hidden).
Items are parallel across languages (same id), but each language has its own option order and answerKey.

A parallel subset is kept: an id is dropped in every language if, in any processed language, its stem is empty
or its five option texts are not all non-empty and distinct (translation collapses some options; 95 of 1,000 ids
for en/zh/de/ar/hi). Then, as in data_csqa/process.py, the kept ids are shuffled with Random(23): the first 5 go
to dev (few-shot pool, unused at 0-shot), the rest to test.

usage: python process.py <dir with <lang>.validation.parquet> [langs...]   (needs pandas + pyarrow)
"""
import csv
import os
import random
import sys

import pandas as pd

RAW = sys.argv[1]
LANGS = sys.argv[2:] or ["en", "zh", "de", "ar", "hi"]
HERE = os.path.dirname(os.path.abspath(__file__))

data = {}
for lang in LANGS:
    df = pd.read_parquet(os.path.join(RAW, f"{lang}.validation.parquet"))
    rows = {}
    for i, q, a in zip(df["id"], df["question"], df["answerKey"]):
        labels = list(q["choices"]["label"])
        if labels != ["A", "B", "C", "D", "E"] or a not in labels:
            raise ValueError(f"{lang} {i}: labels {labels} answer {a}")
        rows[i] = (q["stem"].strip(), [t.strip() for t in q["choices"]["text"]], a)
    data[lang] = rows

ids = set.intersection(*(set(r) for r in data.values()))
bad = {i for i in ids for lang in LANGS
       if not data[lang][i][0] or not all(data[lang][i][1]) or len(set(data[lang][i][1])) < 5}
kept = sorted(ids - bad)
random.Random(23).shuffle(kept)
split = {"dev": kept[:5], "test": kept[5:]}
print(f"ids {len(ids)}, dropped {len(bad)}, dev {len(split['dev'])}, test {len(split['test'])}")

for lang in LANGS:
    task = f"xcsqa_{lang}"
    for key, sel in split.items():
        out = os.path.join(HERE, "..", f"data_{task}", key)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"{task}_{key}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            for i in sel:
                stem, opts, a = data[lang][i]
                w.writerow([stem, *opts, a])
with open(os.path.join(HERE, "ids_test.txt"), "w") as f:
    f.write("\n".join(split["test"]) + "\n")
