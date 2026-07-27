import csv
import io
from pathlib import Path
import sys
import unittest

# Every data_*/process.py is named "process", so a plain `import process` binds
# whichever directory happened to be imported first and the other test modules
# silently get the wrong one. Load this one by path under a unique name.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "race_process", Path(__file__).resolve().parents[1] / "code" / "data_race" / "process.py")
_race = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_race)

NUM_OPTIONS = _race.NUM_OPTIONS
build_question = _race.build_question
collapse_ws = _race.collapse_ws
collect = _race.collect
to_row = _race.to_row

# A passage carrying every character class that breaks naive CSV handling:
# embedded commas, double quotes, single newlines and a blank-line paragraph break.
ARTICLE = 'Tom said, "Hi!"\nHe left, quickly.\n\nThen, later, he came back.'
SAMPLE = {
    "article": ARTICLE,
    "question": "What did Tom do,\naccording to the passage?",
    "options": ["He left", "He, stayed", 'He said "no"', "He\nslept"],
    "answer": "b",
}


def read_back(rows):
    """Write rows exactly as process.py does, then parse them as the loader does."""
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    import pandas as pd

    return pd.read_csv(
        io.StringIO(buf.getvalue()),
        names=("Question", *list("ABCD"), "Answer"),
        dtype=str,
    )


class TestRowBuilding(unittest.TestCase):
    def test_row_shape_and_answer_normalized(self):
        row = to_row(SAMPLE)
        self.assertEqual(len(row), NUM_OPTIONS + 2)
        self.assertEqual(row[-1], "B")

    def test_article_keeps_paragraphs_question_flattened(self):
        question = build_question(ARTICLE, SAMPLE["question"])
        self.assertIn("Article:\n", question)
        self.assertIn("\n\nQuestion: ", question)
        # the passage's own paragraph break survives...
        self.assertIn("He left, quickly.\n\nThen", question)
        # ...but the question itself is a single line
        self.assertTrue(question.split("\n\nQuestion: ")[1].count("\n") == 0)

    def test_options_are_single_line(self):
        # An option containing a newline would put the next option's ID at the
        # start of a line, shifting the token the scorer reads.
        for option in to_row(SAMPLE)[1:-1]:
            self.assertNotIn("\n", option)
        self.assertEqual(collapse_ws("He\nslept"), "He slept")

    def test_malformed_items_are_dropped(self):
        self.assertIsNone(to_row({**SAMPLE, "options": ["a", "b", "c"]}))
        self.assertIsNone(to_row({**SAMPLE, "answer": "E"}))
        self.assertIsNone(to_row({**SAMPLE, "options": ["a", "", "c", "d"]}))


class TestCsvRoundTrip(unittest.TestCase):
    def test_multiline_article_stays_one_row(self):
        df = read_back([to_row(SAMPLE)])
        self.assertEqual(len(df), 1)

    def test_fields_survive_round_trip(self):
        row = to_row(SAMPLE)
        df = read_back([row])
        self.assertEqual(df.iloc[0]["Question"], row[0])
        self.assertEqual(df.iloc[0]["B"], "He, stayed")
        self.assertEqual(df.iloc[0]["C"], 'He said "no"')
        self.assertEqual(df.iloc[0]["D"], "He slept")
        self.assertEqual(df.iloc[0]["Answer"], "B")


class TestCollect(unittest.TestCase):
    def test_counts_skipped_and_keeps_good_rows(self):
        rows, skipped = collect([SAMPLE, {**SAMPLE, "answer": "Z"}, SAMPLE])
        self.assertEqual((len(rows), skipped), (2, 1))

    def test_subsample_is_seed_reproducible(self):
        items = [{**SAMPLE, "question": f"q{i}"} for i in range(20)]
        first, _ = collect(items, limit=5, seed=42)
        second, _ = collect(items, limit=5, seed=42)
        other, _ = collect(items, limit=5, seed=7)
        self.assertEqual(len(first), 5)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_limit_above_size_keeps_everything(self):
        rows, _ = collect([SAMPLE, SAMPLE], limit=99)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
