import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from compare_cost_axis import (  # noqa: E402
    _as_fraction,
    _nearest_index,
    find_sweep,
    load_calibraeval,
    load_sweep,
)


def sweep_payload():
    """Shaped like a real three_curves_points.json, with acc in PERCENT."""
    return {"curves": {
        "cyclic": {
            "cost": [1.0, 1.6, 4.0],
            "acc": [78.88, 79.26, 81.20],          # percent
            "recall_std": [0.0300, 0.0205, 0.0226],
        },
        "default_pride": {
            "p": [0.5, 1.0, 2.0, 5.0],
            "cost": [1.015, 1.031, 1.059, 1.149],
            "acc": [79.40, 79.43, 79.20, 79.60],
            "recall_std": [0.0350, 0.0385, 0.0355, 0.0299],
        },
        # ACCEL lives here. The ours_pride block below is the earlier
        # threshold-cascade line and must not be mistaken for it.
        "empirical_pride": {
            "threshold_schedule": "flat", "percentile_mode": "online",
            "residual_model": "empirical", "transition_mode": "latin",
            "by_alpha": {"2": {"primary": {
                "p": [0.5, 1.0, 2.0, 5.0],
                "cost": [1.007, 1.011, 1.021, 1.050],
                "acc": [67.19, 67.16, 67.43, 67.91],
                "recall_std": [0.0313, 0.0332, 0.0382, 0.0440],
            }}},
        },
        "ours_pride": {"by_alpha": {
            # keyed by PriDe PREFIX; `p` inside is the th1 PERCENTILE.
            "2": {
                # the paper's setting: prefix alpha=2%, primary variant
                "th1/sqrt2": {
                    "p": [0.5, 1.0, 2.0, 5.0],
                    "cost": [1.0210, 1.0305, 1.0410, 1.0800],
                    "acc": [78.900, 78.950, 79.010, 79.100],
                    "recall_std": [0.03700, 0.03600, 0.03450, 0.03300],
                },
                # decoy: the superseded variant at the same indices
                "th1/2": {
                    "p": [0.5, 1.0, 2.0, 5.0],
                    "cost": [9.91, 9.92, 9.93, 9.94],
                    "acc": [1.0, 1.0, 1.0, 1.0],
                    "recall_std": [9.9, 9.9, 9.9, 9.9],
                },
            },
            # decoy: reading the prefix level wrong lands here
            "0.5": {"th1/sqrt2": {
                "p": [0.5, 1.0, 2.0],
                "cost": [1.0730, 1.0807, 1.0904],
                "acc": [79.084, 78.970, 78.970],
                "recall_std": [0.03529, 0.03390, 0.03258],
            }},
        }},
    }}


class TestUnits(unittest.TestCase):
    def test_percent_accuracy_becomes_a_fraction(self):
        self.assertAlmostEqual(_as_fraction(78.88), 0.7888)

    def test_fractional_accuracy_is_left_alone(self):
        self.assertAlmostEqual(_as_fraction(0.7888), 0.7888)

    def test_boundary_of_one_is_treated_as_a_fraction(self):
        self.assertAlmostEqual(_as_fraction(1.0), 1.0)


class TestNearestIndex(unittest.TestCase):
    def test_exact_and_approximate(self):
        self.assertEqual(_nearest_index([0.5, 1.0, 2.0, 5.0], 2.0), 2)
        self.assertEqual(_nearest_index([0.5, 1.0, 2.0, 5.0], 1.9), 2)


class TestLoadSweep(unittest.TestCase):
    def setUp(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(sweep_payload(), f)
        f.close()
        self.methods = load_sweep(f.name)

    def test_baseline_is_the_cost_one_end_of_the_cyclic_curve(self):
        b = self.methods["baseline"]
        self.assertAlmostEqual(b["cost"], 1.0)
        self.assertAlmostEqual(b["acc"], 0.7888)

    def test_cyclic_is_the_full_budget_end(self):
        c = self.methods["cyclic"]
        self.assertAlmostEqual(c["cost"], 4.0)
        self.assertAlmostEqual(c["acc"], 0.8120)

    def test_pride_is_taken_at_alpha_2pct(self):
        p = self.methods["pride"]
        self.assertAlmostEqual(p["alpha_pct"], 2.0)
        self.assertAlmostEqual(p["cost"], 1.059)
        self.assertAlmostEqual(p["acc"], 0.7920)

    def test_accel_comes_from_the_empirical_latin_curve(self):
        # ACCEL is the empirical-residual + Latin-square method in
        # curves.empirical_pride. curves.ours_pride holds the threshold-cascade
        # variants from the earlier line -- same "ours" wording, different method.
        a = self.methods["accel"]
        self.assertAlmostEqual(a["prefix_pct"], 2.0)
        self.assertAlmostEqual(a["beta"], 2.0)
        self.assertAlmostEqual(a["cost"], 1.021)
        self.assertAlmostEqual(a["acc"], 0.6743)
        self.assertEqual(a["schedule"], "flat")
        self.assertEqual(a["transition_mode"], "latin")
        for wrong in (1.0410, 1.0730, 9.93):
            self.assertNotAlmostEqual(a["cost"], wrong, msg="read a cascade variant")

    def test_all_recall_stds_stay_fractions(self):
        for name, m in self.methods.items():
            self.assertLess(m["recall_std"], 1.0, msg=name)


class TestLoadCalibraeval(unittest.TestCase):
    def test_keeps_only_calibraeval_rows(self):
        payload = {"mean": {
            "baseline": {"cost": 1.0, "acc": 0.78, "recall_std": 0.03},
            "calibraeval@1": {"cost": 1.0, "acc": 0.79, "recall_std": 0.024},
            "cyclic": {"cost": 4.0, "acc": 0.81, "recall_std": 0.022},
            "calibraeval@4": {"cost": 4.0, "acc": 0.81, "recall_std": 0.025},
        }}
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, f)
        f.close()
        out = load_calibraeval(f.name)
        self.assertEqual(set(out), {"calibraeval@1", "calibraeval@4"})
        self.assertAlmostEqual(out["calibraeval@1"]["recall_std"], 0.024)


class TestFindSweep(unittest.TestCase):
    @staticmethod
    def _tree(with_empirical):
        """results_arc/0s_M/{untagged, tagged}/ where only one holds ACCEL."""
        root = tempfile.mkdtemp()
        for sub, has in (("arc_full_id-ABCD", False),
                         ("arc_full_id-ABCD__empirical_latin_flat_0502", with_empirical)):
            d = os.path.join(root, "results_arc", "0s_M", sub)
            os.makedirs(d)
            curves = {"cyclic": {}, "ours_pride": {}}
            if has:
                curves["empirical_pride"] = {"by_alpha": {}}
            with open(os.path.join(d, "arc_three_curves_points.json"), "w") as f:
                json.dump({"curves": curves}, f)
        return root

    def test_picks_the_file_that_actually_holds_the_empirical_curve(self):
        # The untagged directory is an older run with no empirical_pride curve.
        # Selecting by directory naming picked that one and hid ACCEL entirely,
        # so selection is by content.
        hit = find_sweep(self._tree(with_empirical=True), "arc", "M")
        self.assertIn("__empirical_latin_flat_0502", hit)

    def test_returns_none_when_no_file_has_the_curve(self):
        self.assertIsNone(find_sweep(self._tree(with_empirical=False), "arc", "M"))

    def test_can_fall_back_to_any_sweep_when_not_required(self):
        hit = find_sweep(self._tree(with_empirical=False), "arc", "M",
                         require_empirical=False)
        self.assertTrue(hit.endswith("arc_three_curves_points.json"))

    def test_returns_none_when_absent(self):
        self.assertIsNone(find_sweep(tempfile.mkdtemp(), "arc", "nope"))


if __name__ == "__main__":
    unittest.main()
