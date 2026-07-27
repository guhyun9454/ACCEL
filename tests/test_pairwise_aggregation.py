import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pairwise_aggregation import (  # noqa: E402
    bradley_terry_predict,
    bradley_terry_scores,
    copeland_predict,
    copeland_scores,
    count_condorcet_cycle,
)


class TestPairwiseAggregation(unittest.TestCase):
    def test_consistent_ranking_recovers_top_option(self):
        pair_probs = {
            (0, 1): 0.8,
            (0, 2): 0.9,
            (0, 3): 0.95,
            (1, 2): 0.7,
            (1, 3): 0.8,
            (2, 3): 0.7,
        }
        self.assertEqual(copeland_predict(pair_probs, 4), 0)
        self.assertEqual(bradley_terry_predict(pair_probs, 4), 0)
        scores = bradley_terry_scores(pair_probs, 4)
        self.assertTrue(np.all(np.diff(scores) < 0))
        self.assertFalse(count_condorcet_cycle(pair_probs, 4))

    def test_copeland_uses_soft_score_to_break_tie(self):
        pair_probs = {
            (0, 1): 0.9,
            (0, 2): 0.9,
            (0, 3): 0.49,
            (1, 2): 0.9,
            (1, 3): 0.9,
            (2, 3): 0.6,
        }
        wins, soft = copeland_scores(pair_probs, 4)
        self.assertEqual(wins[0], wins[1])
        self.assertGreater(soft[0], soft[1])
        self.assertEqual(copeland_predict(pair_probs, 4), 0)

    def test_detects_condorcet_cycle(self):
        pair_probs = {
            (0, 1): 0.8,  # 0 > 1
            (0, 2): 0.2,  # 2 > 0
            (0, 3): 0.8,
            (1, 2): 0.8,  # 1 > 2
            (1, 3): 0.8,
            (2, 3): 0.8,
        }
        self.assertTrue(count_condorcet_cycle(pair_probs, 4))

    def test_rejects_incomplete_pair_graph(self):
        with self.assertRaises(ValueError):
            copeland_predict({(0, 1): 0.6}, 4)


if __name__ == "__main__":
    unittest.main()
