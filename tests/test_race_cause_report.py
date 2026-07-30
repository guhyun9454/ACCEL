from pathlib import Path
import importlib.util
import unittest


_spec = importlib.util.spec_from_file_location(
    "race_cause_report",
    Path(__file__).resolve().parents[1] / "code" / "race_cause_report.py",
)
_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_report)


def trajectory(
    sample_id,
    label,
    predictions,
    confidences,
    pride_prediction,
    prefix=False,
):
    return {
        "sample_id": sample_id,
        "label_idx": label,
        "prefix_forced": prefix,
        "decision_stages": [1, 2, 3, 4],
        "pred_by_stage": predictions,
        "conf_by_stage": confidences,
        "true_prob_by_stage": [0.25, 0.25, 0.25, 0.25],
        "pride_pred_idx": pride_prediction,
    }


class TestRaceCauseReport(unittest.TestCase):
    def test_decomposition_groups_sum_to_overall_delta(self):
        rows = [
            trajectory(0, 0, [1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4], 1, prefix=True),
            trajectory(1, 1, [0, 1, 1, 1], [0.05, 0.3, 0.4, 0.5], 0),
            trajectory(2, 2, [2, 2, 2, 2], [0.9, 0.9, 0.9, 0.9], 2),
            trajectory(3, 3, [2, 2, 2, 2], [0.8, 0.8, 0.8, 0.8], 3),
        ]

        report = _report.build_decomposition(rows, percentile=50, k=4)

        self.assertEqual(report["groups"]["P_prefix"]["n"], 1)
        self.assertEqual(report["groups"]["E_escalated"]["n"], 1)
        self.assertEqual(report["groups"]["U_stage1"]["n"], 2)
        self.assertAlmostEqual(
            report["accel_minus_pride"],
            report["contribution_check"],
        )
        self.assertEqual(report["routing"]["w2c"], 1)
        self.assertEqual(report["routing"]["c2w"], 0)


if __name__ == "__main__":
    unittest.main()
