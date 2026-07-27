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
        "ours_pride": {"by_alpha": {
            "2": {"th1/2": {
                "p": [0.5, 1.0, 2.0],
                "cost": [1.0730, 1.0807, 1.0904],
                "acc": [79.084, 78.970, 78.970],
                "recall_std": [0.03529, 0.03390, 0.03258],
            }},
            "5": {"th1/2": {"p": [0.5], "cost": [9.9], "acc": [1.0], "recall_std": [9.9]}},
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

    def test_accel_reads_the_2pct_prefix_block_at_th1_0p5(self):
        # by_alpha is keyed by PriDe prefix; p inside it is the th1 percentile.
        # Picking the wrong level silently returns another configuration.
        a = self.methods["accel"]
        self.assertAlmostEqual(a["cost"], 1.0730)
        self.assertAlmostEqual(a["th1_pct"], 0.5)
        self.assertAlmostEqual(a["acc"], 0.79084)

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
    def test_skips_result_tag_variant_directories(self):
        # Only the untagged directory is the canonical sweep; the __<tag> ones are
        # separate experiment variants and must not be picked up by accident.
        root = tempfile.mkdtemp()
        for sub in ("arc_full_id-ABCD", "arc_full_id-ABCD__empirical_latin_flat_0502"):
            d = os.path.join(root, "results_arc", "0s_M", sub)
            os.makedirs(d)
            with open(os.path.join(d, "arc_three_curves_points.json"), "w") as f:
                f.write("{}")
        hit = find_sweep(root, "arc", "M")
        self.assertTrue(hit.endswith("arc_full_id-ABCD/arc_three_curves_points.json"))

    def test_returns_none_when_absent(self):
        self.assertIsNone(find_sweep(tempfile.mkdtemp(), "arc", "nope"))


if __name__ == "__main__":
    unittest.main()
