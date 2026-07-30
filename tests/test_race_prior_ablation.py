from pathlib import Path
import importlib.util
import unittest

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "race_prior_ablation",
    Path(__file__).resolve().parents[1] / "code" / "race_prior_ablation.py",
)
_ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ablation)


class TestRacePriorAblation(unittest.TestCase):
    def test_prediction_report_separates_prefix_and_pairwise_wins(self):
        labels = np.asarray([0, 1, 2, 3])
        predictions = {
            "raw": np.asarray([0, 0, 2, 0]),
            "pride_arithmetic": np.asarray([0, 0, 1, 3]),
            "accel_geometric_mu_only": np.asarray([0, 1, 1, 0]),
            "accel_empirical_mixture": np.asarray([0, 1, 2, 0]),
        }

        report = _ablation.build_prediction_report(
            labels=labels,
            predictions_by_variant=predictions,
            prefix_ids=[0],
            k=4,
        )

        empirical = report["all"]["accel_empirical_mixture"]
        self.assertEqual(empirical["w2c_vs_pride_count"], 2)
        self.assertEqual(empirical["c2w_vs_pride_count"], 1)
        self.assertAlmostEqual(empirical["accuracy_delta_vs_pride"], 0.25)
        self.assertEqual(
            report["nonprefix"]["pride_arithmetic"]["n"],
            3,
        )

    def test_prediction_report_requires_matching_lengths(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            _ablation.build_prediction_report(
                labels=np.asarray([0, 1]),
                predictions_by_variant={
                    "pride_arithmetic": np.asarray([0]),
                },
                prefix_ids=[],
                k=2,
            )


if __name__ == "__main__":
    unittest.main()
