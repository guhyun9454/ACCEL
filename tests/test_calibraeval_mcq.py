import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from calibraeval_mcq import (  # noqa: E402
    OrderPreservingCalibrator,
    accuracy,
    evaluate,
    load_cached_run,
    recall_std,
    to_content_space,
)


class TestContentSpace(unittest.TestCase):
    def test_cyclic_unrotation_recovers_the_same_option(self):
        # Content option 3 is correct; under permutation c it sits at slot (3-c)%4,
        # which is exactly the layout observed in the cached ARC records.
        probs = [
            [0.01, 0.00, 0.00, 0.99],  # perm 0 -> slot D
            [0.01, 0.00, 0.98, 0.01],  # perm 1 -> slot C
            [0.02, 0.96, 0.01, 0.01],  # perm 2 -> slot B
            [0.98, 0.01, 0.00, 0.01],  # perm 3 -> slot A
        ]
        Q = to_content_space(probs, 4)
        # after un-rotating, every view should put its mass on content option 3
        self.assertTrue(np.all(np.argmax(Q, axis=1) == 3))

    def test_identity_for_first_view(self):
        probs = [[0.1, 0.2, 0.3, 0.4]] * 4
        Q = to_content_space(probs, 4)
        np.testing.assert_allclose(Q[0], [0.1, 0.2, 0.3, 0.4])


class TestMetrics(unittest.TestCase):
    def test_recall_std_zero_when_uniformly_correct(self):
        labels = [0, 1, 2, 3, 0, 1, 2, 3]
        self.assertAlmostEqual(recall_std(labels, labels, 4), 0.0)

    def test_recall_std_flags_position_bias(self):
        # always predicts class 0 -> recall is 1 for class 0 and 0 elsewhere
        labels = [0, 1, 2, 3]
        preds = [0, 0, 0, 0]
        self.assertAlmostEqual(recall_std(labels, preds, 4), float(np.std([1.0, 0, 0, 0])))

    def test_absent_gold_classes_are_ignored(self):
        # only classes 0 and 1 appear as gold; 2 and 3 must not count as zero recall
        self.assertAlmostEqual(recall_std([0, 1], [0, 1], 4), 0.0)

    def test_accuracy(self):
        self.assertAlmostEqual(accuracy([1, 2, 3], [1, 0, 3]), 2 / 3)


class TestCalibratorInternals(unittest.TestCase):
    """The gradient is derived by hand, so it is checked numerically."""

    def setUp(self):
        rng = np.random.default_rng(0)
        Q = rng.random((6, 3, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        self.cal = OrderPreservingCalibrator(lam=0.05)
        values = np.unique(np.concatenate([Q.reshape(-1), [0.0, 1.0]]))
        self.values = values
        self.idx = np.searchsorted(values, Q)
        self.theta = rng.normal(size=len(values) - 1) * 0.3

    def _loss(self, theta):
        g = OrderPreservingCalibrator._g_from_theta(theta)
        return self.cal._loss_and_dg(g, self.idx)[0]

    def test_analytic_gradient_matches_finite_differences(self):
        g = OrderPreservingCalibrator._g_from_theta(self.theta)
        _, dg = self.cal._loss_and_dg(g, self.idx)
        analytic = OrderPreservingCalibrator._theta_grad(self.theta, dg)

        eps = 1e-6
        for n in range(0, len(self.theta), 7):  # sample coordinates; full sweep is slow
            plus, minus = self.theta.copy(), self.theta.copy()
            plus[n] += eps
            minus[n] -= eps
            numeric = (self._loss(plus) - self._loss(minus)) / (2 * eps)
            self.assertAlmostEqual(
                analytic[n], numeric, places=5,
                msg=f"gradient mismatch at {n}: analytic={analytic[n]} numeric={numeric}",
            )

    def test_map_is_monotone_and_bounded(self):
        g = OrderPreservingCalibrator._g_from_theta(self.theta)
        self.assertAlmostEqual(g[0], 0.0)
        self.assertAlmostEqual(g[-1], 1.0)
        self.assertTrue(np.all(np.diff(g) >= 0))

    def test_fit_preserves_order_within_a_view(self):
        # the defining property: calibration may not reorder options
        rng = np.random.default_rng(1)
        Q = rng.random((40, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        cal = OrderPreservingCalibrator(epochs=5).fit(Q)
        out = cal.calibrate(Q)
        for before, after in zip(Q.reshape(-1, 4), out.reshape(-1, 4)):
            np.testing.assert_array_equal(np.argsort(before), np.argsort(after))

    def test_calibrated_views_are_distributions(self):
        rng = np.random.default_rng(2)
        Q = rng.random((20, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        cal = OrderPreservingCalibrator(epochs=5).fit(Q)
        np.testing.assert_allclose(cal.calibrate(Q).sum(axis=-1), 1.0, rtol=1e-9)

    def test_loss_decreases(self):
        rng = np.random.default_rng(3)
        Q = rng.random((30, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        values = np.unique(np.concatenate([Q.reshape(-1), [0.0, 1.0]]))
        idx = np.searchsorted(values, Q)
        cal = OrderPreservingCalibrator(lam=0.05)
        theta = np.zeros(len(values) - 1)
        first = cal._loss_and_dg(cal._g_from_theta(theta), idx)[0]
        for _ in range(20):
            g = cal._g_from_theta(theta)
            _, dg = cal._loss_and_dg(g, idx)
            theta -= cal.lr * cal._theta_grad(theta, dg)
            theta -= theta.mean()
        last = cal._loss_and_dg(cal._g_from_theta(theta), idx)[0]
        self.assertLess(last, first)


class TestCacheLoading(unittest.TestCase):
    def _write(self, records):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def test_reads_records_and_skips_malformed(self):
        good = {"type": "result", "data": {
            "idx": 0, "ideal": "C",
            "probs": [[0.1, 0.2, 0.6, 0.1]] * 4, "options": ["a", "b", "c", "d"]}}
        wrong_shape = {"type": "result", "data": {"idx": 1, "ideal": "A", "probs": [[0.5, 0.5]]}}
        not_result = {"type": "meta", "data": {}}
        no_ideal = {"type": "result", "data": {"idx": 2, "probs": [[0.25] * 4] * 4}}
        path = self._write([good, wrong_shape, not_result, no_ideal, good])
        Q, y = load_cached_run(path)
        self.assertEqual(Q.shape, (2, 4, 4))
        self.assertEqual(list(y), [2, 2])

    def test_raises_when_nothing_usable(self):
        path = self._write([{"type": "meta", "data": {}}])
        with self.assertRaises(ValueError):
            load_cached_run(path)

    def test_infers_five_options_for_csqa(self):
        # CSQA is 5-choice; k must come from the data, not a hardcoded 4.
        rec = {"type": "result", "data": {
            "idx": 0, "ideal": "E",
            "probs": [[0.1, 0.1, 0.1, 0.1, 0.6]] * 5}}
        Q, y = load_cached_run(self._write([rec]))
        self.assertEqual(Q.shape, (1, 5, 5))
        self.assertEqual(list(y), [4])


class TestEvaluate(unittest.TestCase):
    def test_reports_all_regimes_on_heldout_items(self):
        rng = np.random.default_rng(4)
        n = 200
        y = rng.integers(0, 4, size=n)
        Q = rng.random((n, 4, 4)) * 0.2
        for i in range(n):
            Q[i, :, y[i]] += 1.0
        Q /= Q.sum(axis=2, keepdims=True)

        res = evaluate(Q, y, calib_frac=0.02, epochs=3)
        self.assertEqual(res["n_calib"] + res["n_test"], n)
        self.assertEqual(set(res["methods"]), {"baseline", "calibraeval@1", "cyclic", "calibraeval@4"})
        # signal is strong and position-independent here, so everything should be near-perfect
        for name, m in res["methods"].items():
            self.assertGreater(m["acc"], 0.9, msg=name)
        self.assertEqual(res["methods"]["baseline"]["cost"], 1.0)
        self.assertEqual(res["methods"]["calibraeval@4"]["cost"], 4.0)

    def test_calibration_split_is_excluded_from_scoring(self):
        rng = np.random.default_rng(5)
        Q = rng.random((100, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        y = rng.integers(0, 4, size=100)
        res = evaluate(Q, y, calib_frac=0.1, epochs=2)
        self.assertEqual(res["n_calib"], 10)
        self.assertEqual(res["n_test"], 90)


if __name__ == "__main__":
    unittest.main()
