import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from calibrate_cache import probs_of, read_records  # noqa: E402


class TestProbsOf(unittest.TestCase):
    def test_accepts_a_multi_view_result(self):
        rec = {"type": "result", "data": {"probs": [[0.1, 0.2, 0.3, 0.4]] * 4}}
        self.assertEqual(probs_of(rec).shape, (4, 4))

    def test_rejects_non_result_records(self):
        self.assertIsNone(probs_of({"type": "meta", "data": {"probs": [[0.5, 0.5]] * 2}}))

    def test_rejects_single_view(self):
        # a one-view record cannot support the cross-view objective and must be
        # passed through untouched rather than silently calibrated
        self.assertIsNone(probs_of({"type": "result", "data": {"probs": [[0.25] * 4]}}))

    def test_rejects_missing_probs(self):
        self.assertIsNone(probs_of({"type": "result", "data": {"ideal": "A"}}))


class TestRoundTrip(unittest.TestCase):
    """The rewritten cache must stay readable by eval_clm.py."""

    def test_records_keep_schema_and_order(self):
        recs = [
            {"type": "meta", "data": {"note": "keep me"}},
            {"type": "result", "data": {"idx": 0, "ideal": "B",
                                        "probs": [[0.1, 0.6, 0.2, 0.1]] * 4,
                                        "options": ["a", "b", "c", "d"]}},
        ]
        path = os.path.join(tempfile.mkdtemp(), "arc_run0.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

        back = read_records(path)
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0]["type"], "meta")
        self.assertEqual(back[1]["data"]["ideal"], "B")
        self.assertEqual(back[1]["data"]["options"], ["a", "b", "c", "d"])

    def test_calibrated_block_is_still_a_distribution(self):
        # eval_clm reads these as option-ID probabilities; if a view stopped
        # summing to 1 every downstream metric would shift for the wrong reason.
        from calibraeval_mcq import OrderPreservingCalibrator

        rng = np.random.default_rng(0)
        Q = rng.random((30, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        cal = OrderPreservingCalibrator(max_epochs=5).fit(Q)
        out = cal.calibrate(Q[0])
        self.assertEqual(out.shape, (4, 4))
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=1e-9)
        self.assertTrue(np.all(out >= 0.0))


if __name__ == "__main__":
    unittest.main()
