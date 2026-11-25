"""MMMLU (OpenAI; MMLU test set professionally translated) -> data_mmmlu_<lang>/{test,dev}/<subject>_{test,dev}.csv.

Source: HF dataset openai/MMMLU, revision 325a01dc3e173cac1578df94120499aaca2e2504, files test/mmlu_<LOCALE>.csv
(14,042 rows each, columns Question, A-D, Answer, Subject). Rows follow MMLU's test order subject by subject, so the
57 per-subject files line up one to one with data_mmlu/test (checked: same subject sequence; the answer key differs
from English on a few rows, e.g. 176 in ZH-CN, so each language keeps its own Answer). All rows are kept, as for MMLU.
dev/ holds a copy of the English data_mmlu/dev files: the loader always opens a dev file, but only few-shot runs use
it, and the multilingual check is 0-shot. Generated files are not committed (about 10 MB per language).

usage: python process.py <dir with mmlu_<LOCALE>.csv> [lang ...]   (needs pandas)
"""
import csv
import os
import shutil
import sys

import pandas as pd

LOCALE = {"zh": "ZH-CN", "de": "DE-DE", "ar": "AR-XY", "hi": "HI-IN"}
RAW = sys.argv[1]
LANGS = sys.argv[2:] or list(LOCALE)
HERE = os.path.dirname(os.path.abspath(__file__))
MMLU = os.path.join(HERE, "..", "data_mmlu")
subjects = sorted(f[:-len("_test.csv")] for f in os.listdir(os.path.join(MMLU, "test")) if f.endswith("_test.csv"))

for lang in LANGS:
    df = pd.read_csv(os.path.join(RAW, f"mmlu_{LOCALE[lang]}.csv"), dtype=str, keep_default_na=False)
    if sorted(df["Subject"].unique()) != subjects:
        raise ValueError(f"{lang}: subject set differs from data_mmlu")
    out = os.path.join(HERE, "..", f"data_mmmlu_{lang}")
    os.makedirs(os.path.join(out, "test"), exist_ok=True)
    os.makedirs(os.path.join(out, "dev"), exist_ok=True)
    n = 0
    for s in subjects:
        rows = df[df["Subject"] == s]
        n_en = sum(1 for _ in open(os.path.join(MMLU, "test", f"{s}_test.csv"), newline=""))
        with open(os.path.join(out, "test", f"{s}_test.csv"), "w", newline="") as f:
            w = csv.writer(f)
            for _, r in rows.iterrows():
                if r["Answer"] not in "ABCD" or len(r["Answer"]) != 1:
                    raise ValueError(f"{lang} {s}: answer {r['Answer']!r}")
                w.writerow([r["Question"], r["A"], r["B"], r["C"], r["D"], r["Answer"]])
        if len(rows) != len(pd.read_csv(os.path.join(MMLU, "test", f"{s}_test.csv"), header=None)):
            raise ValueError(f"{lang} {s}: {len(rows)} rows vs English")
        shutil.copy(os.path.join(MMLU, "dev", f"{s}_dev.csv"), os.path.join(out, "dev", f"{s}_dev.csv"))
        n += len(rows)
    print(lang, n, "rows,", len(subjects), "subjects")
