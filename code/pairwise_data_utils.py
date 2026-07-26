"""Shared conversion helpers for the pairwise (k=2) preference benchmarks.

RewardBench, MT-Bench and Preference-Bench all reduce to the same item: a query
plus two candidate answers, one of which is the gold choice. They differ only in
field names and in how gold is recorded, so the per-dataset `process.py` scripts
supply an iterator of (query, chosen, rejected) and everything else lives here.

The CSV layout matches data_arc / data_csqa / data_race: positional columns with
no header — Question, A, B, Answer.
"""

import csv
import os
import random

OPTION_IDS = "AB"


def collapse_ws(text: str) -> str:
    """Flatten to one line — the prompt renders one option per line."""
    return " ".join(str(text).split())


def to_row(query: str, chosen: str, rejected: str, rng: random.Random):
    """Return a CSV row, or None if the item carries no usable preference.

    Which slot holds the better answer is randomized, so gold is not always "A".
    Without this a model that always answers "A" would score 100% and the
    benchmark would measure nothing.
    """
    query = collapse_ws(query)
    chosen = collapse_ws(chosen)
    rejected = collapse_ws(rejected)
    if not query or not chosen or not rejected or chosen == rejected:
        return None
    if rng.random() < 0.5:
        return [query, chosen, rejected, "A"]
    return [query, rejected, chosen, "B"]


def collect(triples, limit=None, seed=42):
    """(query, chosen, rejected) iterable -> (rows, n_skipped)."""
    rng = random.Random(seed)
    rows, skipped = [], 0
    for query, chosen, rejected in triples:
        row = to_row(query, chosen, rejected, rng)
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


def write_dataset(task: str, triples, here: str, limit=None, seed=42, num_few_shot_pool=5):
    """Write dev/ and test/ for one task and report what was produced."""
    triples = list(triples)
    dev_src, test_src = triples[:num_few_shot_pool], triples[num_few_shot_pool:]

    test_rows, skipped = collect(test_src, limit, seed)
    test_path = write_split(test_rows, os.path.join(here, "test"), f"{task}_test.csv")
    print(f"test : {len(test_rows)} rows (skipped {skipped}) -> {test_path}")

    dev_rows, dev_skipped = collect(dev_src, num_few_shot_pool, seed)
    dev_path = write_split(dev_rows, os.path.join(here, "dev"), f"{task}_dev.csv")
    print(f"dev  : {len(dev_rows)} rows (skipped {dev_skipped}) -> {dev_path}")

    balance = {oid: sum(1 for r in test_rows if r[-1] == oid) for oid in OPTION_IDS}
    print(f"label balance: {balance}")
    return test_rows, dev_rows


def majority_vote(votes):
    """Strict majority winner among votes, or None on a tie / no clear majority.

    MT-Bench records one row per human judge, so the same pair appears several
    times and the judges do not always agree.
    """
    if not votes:
        return None
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else None
