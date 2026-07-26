import csv
import io
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "data_rewardbench"))

from process import collapse_ws, collect, to_row  # noqa: E402

SAMPLE = {
    "prompt": "What is 2+2?  ",
    "chosen": "Four,\nexactly",
    "rejected": "Five",
    "subset": "math",
}


class TestRowBuilding(unittest.TestCase):
    def test_gold_slot_holds_the_chosen_answer(self):
        for seed in range(20):
            row = to_row(SAMPLE, random.Random(seed))
            gold = row[1] if row[-1] == "A" else row[2]
            self.assertEqual(gold, "Four, exactly")

    def test_two_options_plus_query_and_label(self):
        row = to_row(SAMPLE, random.Random(0))
        self.assertEqual(len(row), 4)
        self.assertIn(row[-1], ("A", "B"))

    def test_options_are_single_line(self):
        row = to_row(SAMPLE, random.Random(0))
        for field in row[:-1]:
            self.assertNotIn("\n", field)
        self.assertEqual(collapse_ws("a\n b"), "a b")

    def test_malformed_items_are_dropped(self):
        rng = random.Random(0)
        self.assertIsNone(to_row({**SAMPLE, "prompt": "   "}, rng))
        self.assertIsNone(to_row({**SAMPLE, "chosen": ""}, rng))
        # an item whose two candidates are identical carries no preference signal
        self.assertIsNone(to_row({**SAMPLE, "rejected": SAMPLE["chosen"]}, rng))


class TestCollect(unittest.TestCase):
    def test_labels_are_balanced_not_always_a(self):
        # If the better answer always sat in slot A, a model that always answers A
        # would score 100% and the benchmark would measure nothing.
        rows, _ = collect([SAMPLE] * 200, seed=42)
        counts = {oid: sum(1 for r in rows if r[-1] == oid) for oid in "AB"}
        self.assertEqual(sum(counts.values()), 200)
        self.assertGreater(min(counts.values()), 70)

    def test_counts_skipped(self):
        rows, skipped = collect([SAMPLE, {**SAMPLE, "chosen": ""}, SAMPLE], seed=1)
        self.assertEqual((len(rows), skipped), (2, 1))

    def test_subsample_is_seed_reproducible(self):
        items = [{**SAMPLE, "prompt": f"q{i}"} for i in range(50)]
        first, _ = collect(items, limit=10, seed=7)
        second, _ = collect(items, limit=10, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)


class TestCsvRoundTrip(unittest.TestCase):
    def test_fields_survive_round_trip(self):
        import pandas as pd

        rows, _ = collect([SAMPLE] * 3, seed=3)
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        df = pd.read_csv(io.StringIO(buf.getvalue()),
                         names=("Question", "A", "B", "Answer"), dtype=str)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0]["Question"], "What is 2+2?")
        self.assertTrue(set(df["Answer"]) <= {"A", "B"})


if __name__ == "__main__":
    unittest.main()
