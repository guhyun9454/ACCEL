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
        self.theta = rng.normal(size=(4, len(values) - 1)) * 0.3

    def _loss(self, theta):
        g = OrderPreservingCalibrator._g_from_theta(theta)
        return self.cal._loss_and_dg(g, self.idx)[0]

    def test_analytic_gradient_matches_finite_differences(self):
        g = OrderPreservingCalibrator._g_from_theta(self.theta)
        _, dg = self.cal._loss_and_dg(g, self.idx)
        analytic = OrderPreservingCalibrator._theta_grad(self.theta, dg)

        eps = 1e-6
        rows, cols = self.theta.shape
        for j in range(rows):
            for n in range(0, cols, 7):  # sample coordinates; a full sweep is slow
                plus, minus = self.theta.copy(), self.theta.copy()
                plus[j, n] += eps
                minus[j, n] -= eps
                numeric = (self._loss(plus) - self._loss(minus)) / (2 * eps)
                self.assertAlmostEqual(
                    analytic[j, n], numeric, places=5,
                    msg=f"gradient mismatch at slot {j} value {n}: "
                        f"analytic={analytic[j, n]} numeric={numeric}",
                )

    def test_every_slot_map_is_monotone_and_bounded(self):
        g = OrderPreservingCalibrator._g_from_theta(self.theta)
        for row in g:
            self.assertAlmostEqual(row[0], 0.0)
            self.assertAlmostEqual(row[-1], 1.0)
            self.assertTrue(np.all(np.diff(row) >= 0))

    def test_each_slot_map_preserves_order_within_its_own_slot(self):
        # Order preservation now means: within one slot, a larger observed
        # probability still maps to a larger calibrated one. It deliberately does
        # NOT hold across slots — that is what lets calibration correct position bias.
        rng = np.random.default_rng(1)
        Q = rng.random((40, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        cal = OrderPreservingCalibrator(max_epochs=5).fit(Q)
        out = cal.transform(Q)
        for j in range(4):
            before = Q[:, :, j].reshape(-1)
            after = out[:, :, j].reshape(-1)
            np.testing.assert_array_equal(np.argsort(before), np.argsort(after))

    def test_per_slot_maps_can_change_a_single_view_argmax(self):
        # The reason the calibrator is per-slot at all. A single shared monotone map
        # cannot change a k-way argmax, so fitting one would reproduce the baseline
        # exactly; distinct per-slot maps can and do move the decision.
        cal = OrderPreservingCalibrator()
        cal.values_ = np.array([0.0, 0.4, 0.6, 1.0])
        cal.mapped_ = np.array([
            [0.0, 0.05, 0.10, 1.0],   # slot 0 heavily damped (a biased position)
            [0.0, 0.90, 0.95, 1.0],
            [0.0, 0.90, 0.95, 1.0],
            [0.0, 0.90, 0.95, 1.0],
        ])
        P = np.array([[0.6, 0.4, 0.0, 0.0]])
        self.assertEqual(int(np.argmax(P, axis=1)[0]), 0)
        self.assertEqual(int(np.argmax(cal.calibrate(P), axis=1)[0]), 1)

    def test_calibrated_views_are_distributions(self):
        rng = np.random.default_rng(2)
        Q = rng.random((20, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        cal = OrderPreservingCalibrator(max_epochs=5).fit(Q)
        np.testing.assert_allclose(cal.calibrate(Q).sum(axis=-1), 1.0, rtol=1e-9)

    def test_loss_decreases(self):
        rng = np.random.default_rng(3)
        Q = rng.random((30, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        values = np.unique(np.concatenate([Q.reshape(-1), [0.0, 1.0]]))
        idx = np.searchsorted(values, Q)
        cal = OrderPreservingCalibrator(lam=0.05)
        theta = np.zeros((4, len(values) - 1))
        first = cal._loss_and_dg(cal._g_from_theta(theta), idx)[0]
        for _ in range(20):
            g = cal._g_from_theta(theta)
            _, dg = cal._loss_and_dg(g, idx)
            theta = theta - cal.lr * cal._theta_grad(theta, dg)
            theta -= theta.mean(axis=-1, keepdims=True)
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
    @staticmethod
    def _biased_cache(n=600, k=4, slot0_boost=1.0, seed=4):
        """Synthetic cache in slot space with a controllable slot-0 preference.

        Content option i is placed at slot (i - c) % k under view c, matching the
        real cache, then slot 0's mass is inflated to simulate position bias.
        """
        rng = np.random.default_rng(seed)
        y = rng.integers(0, k, size=n)
        P = np.empty((n, k, k))
        for item in range(n):
            for c in range(k):
                row = rng.random(k) * 0.3
                row[(y[item] - c) % k] += 1.0     # the correct option, at its slot
                row[0] += slot0_boost              # position bias, always on slot 0
                P[item, c] = row / row.sum()
        return P, y

    def test_macro_and_pooled_recall_std_differ_on_multi_subject_data(self):
        # eval_clm.py macro-averages recall_std over MMLU's 57 subjects; pooling
        # instead gives a different quantity, and comparing one against the other
        # is the mistake this pair of fields exists to prevent.
        # The signal has to be weak enough that the model actually errs; with a
        # clean signal every recall is 1.0 and both figures are trivially 0.
        rng = np.random.default_rng(11)
        parts = []
        for _ in range(20):                      # 20 "subjects", 40 items each
            n, k = 40, 4
            y = rng.integers(0, k, size=n)
            P = np.empty((n, k, k))
            for item in range(n):
                for c in range(k):
                    row = rng.random(k)
                    row[(y[item] - c) % k] += 0.25   # weak preference for the answer
                    P[item, c] = row / row.sum()
            parts.append((P, y))
        res = evaluate(parts, estimation="full", max_epochs=10)
        m = res["methods"]["baseline"]
        self.assertEqual(m["recall_std"], m["recall_std_macro"])
        self.assertEqual(m["n_blocks_scored"], 20)
        self.assertGreater(m["recall_std_pooled"], 0.0, "fixture produced no errors")
        # ~10 items per gold position per subject: sampling noise alone lifts the
        # macro average well above the pooled figure.
        self.assertGreater(m["recall_std_macro"], m["recall_std_pooled"])

    def test_single_block_makes_macro_equal_pooled(self):
        P, y = self._biased_cache(n=200)
        m = evaluate([(P, y)], estimation="full", max_epochs=5)["methods"]["baseline"]
        self.assertAlmostEqual(m["recall_std_macro"], m["recall_std_pooled"])

    def test_reports_all_regimes_on_heldout_items(self):
        P, y = self._biased_cache(slot0_boost=0.0)
        res = evaluate([(P, y)], estimation='prefix', calib_frac=0.1, max_epochs=5)
        self.assertEqual(res["n_calib"] + res["n_test"], len(y))
        self.assertEqual(set(res["methods"]),
                         {"baseline", "calibraeval@1", "prior_division@1",
                          "cyclic", "calibraeval@4"})
        self.assertEqual(res["methods"]["baseline"]["cost"], 1.0)
        self.assertEqual(res["methods"]["calibraeval@4"]["cost"], 4.0)
        # no position bias and a clean signal -> the uncalibrated routes are strong
        self.assertGreater(res["methods"]["baseline"]["acc"], 0.9)
        self.assertGreater(res["methods"]["cyclic"]["acc"], 0.9)

    def test_calibration_reduces_position_bias_when_it_exists(self):
        # The point of the method: with a slot-0 preference baked in, the
        # uncalibrated baseline over-predicts whichever content option sits at slot
        # 0, inflating recall_std. Per-slot calibration should pull that back.
        P, y = self._biased_cache(slot0_boost=0.9)
        res = evaluate([(P, y)], estimation='prefix', calib_frac=0.2, max_epochs=40)
        base = res["methods"]["baseline"]
        cal = res["methods"]["calibraeval@1"]
        self.assertGreater(base["recall_std"], 0.05, "fixture failed to inject bias")
        self.assertLess(cal["recall_std"], base["recall_std"])

    def test_calibration_split_is_excluded_from_scoring(self):
        rng = np.random.default_rng(5)
        Q = rng.random((100, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        y = rng.integers(0, 4, size=100)
        res = evaluate([(Q, y)], estimation='prefix', calib_frac=0.1, max_epochs=2)
        self.assertEqual(res["n_calib"], 10)
        self.assertEqual(res["n_test"], 90)

    def test_calibration_prefix_is_taken_per_block(self):
        # MMLU ships one file per subject. The calibration set must draw from every
        # subject, not just whichever sorts first.
        rng = np.random.default_rng(6)
        parts = []
        for _ in range(5):
            Q = rng.random((40, 4, 4))
            Q /= Q.sum(axis=2, keepdims=True)
            parts.append((Q, rng.integers(0, 4, size=40)))
        res = evaluate(parts, estimation='prefix', calib_frac=0.1, max_epochs=2)
        self.assertEqual(res["n_blocks"], 5)
        self.assertEqual(res["n_calib"], 5 * 4)     # 10% of each 40-item block
        self.assertEqual(res["n_test"], 5 * 36)

    def test_rejects_bare_arrays(self):
        with self.assertRaises(TypeError):
            evaluate(np.zeros((4, 4, 4)), np.zeros(4))

    def test_rejects_unknown_estimation_mode(self):
        P, y = self._biased_cache(n=40)
        with self.assertRaises(ValueError):
            evaluate([(P, y)], estimation="half")

    def test_full_estimation_uses_and_scores_every_item(self):
        # Paper §4.3: the calibration map is fitted on the whole test set without
        # gold labels, and results are reported over that same set.
        P, y = self._biased_cache(n=120, slot0_boost=0.9)
        res = evaluate([(P, y)], estimation="full", max_epochs=20)
        self.assertEqual(res["estimation"], "full")
        self.assertEqual(res["n_calib"], len(y))
        self.assertEqual(res["n_test"], len(y))

    def test_prefix_mode_fits_on_a_small_holdout_and_scores_the_rest(self):
        # Deliberately not asserting that one mode scores better than the other:
        # which wins is an empirical question about real data, and a synthetic
        # fixture can be made to favour either.
        P, y = self._biased_cache(n=600, slot0_boost=0.9)
        prefix = evaluate([(P, y)], estimation="prefix", calib_frac=0.02, max_epochs=30)
        self.assertEqual(prefix["n_calib"], 12)
        self.assertEqual(prefix["n_test"], 588)
        self.assertEqual(prefix["estimation"], "prefix")

    def test_relative_loss_stopping_triggers(self):
        # A loose tolerance must stop early; the epoch cap is only a backstop.
        P, y = self._biased_cache(n=200)
        res = evaluate([(P, y)], estimation="full", max_epochs=500, tol=1e-2)
        self.assertTrue(res["converged"])
        self.assertLess(res["epochs_run"], 500)

    def test_default_lambda_is_the_paper_value(self):
        self.assertEqual(OrderPreservingCalibrator().lam, 0.5)


if __name__ == "__main__":
    unittest.main()


class TestPriorDivisionControl(unittest.TestCase):
    """The control that tells us what the competitor column is actually measuring."""

    def test_it_is_reported(self):
        P, y = TestEvaluate._biased_cache(n=120)
        res = evaluate([(P, y)], estimation="full", max_epochs=10)
        self.assertIn("prior_division@1", res["methods"])
        self.assertEqual(res["methods"]["prior_division@1"]["cost"], 1.0)

    def test_it_removes_an_injected_slot_prior(self):
        # With a slot-0 preference baked in, dividing by the estimated per-slot
        # prior must cut recall_std relative to the raw baseline -- otherwise the
        # control is not doing its job and cannot rule anything out.
        P, y = TestEvaluate._biased_cache(n=600, slot0_boost=0.9)
        res = evaluate([(P, y)], estimation="full", max_epochs=10)
        self.assertLess(res["methods"]["prior_division@1"]["recall_std"],
                        res["methods"]["baseline"]["recall_std"])


class TestCheckpointedFit(unittest.TestCase):
    """One pass must give the same maps as separate runs to each stopping point."""

    @staticmethod
    def _data(seed=0, n=40):
        rng = np.random.default_rng(seed)
        Q = rng.random((n, 4, 4))
        Q /= Q.sum(axis=2, keepdims=True)
        return Q

    def test_yields_every_requested_checkpoint_in_order(self):
        Q = self._data()
        seen = [e for e, _ in OrderPreservingCalibrator().fit_with_checkpoints(Q, [3, 1, 7])]
        self.assertEqual(seen, [1, 3, 7])

    def test_checkpoint_matches_an_independent_run_to_that_epoch(self):
        # If these diverged, a multi-checkpoint sweep would not be reporting the
        # same thing as running each stopping point separately.
        Q = self._data(seed=5)
        target = 6
        via_ckpt = None
        for epoch, cal in OrderPreservingCalibrator().fit_with_checkpoints(Q, [2, target]):
            if epoch == target:
                via_ckpt = cal.calibrate(Q[:, 0, :]).copy()
        direct = OrderPreservingCalibrator(max_epochs=target, tol=0.0).fit(Q)
        np.testing.assert_allclose(via_ckpt, direct.calibrate(Q[:, 0, :]), rtol=1e-10)

    def test_map_is_usable_at_each_checkpoint(self):
        Q = self._data(seed=7)
        for _, cal in OrderPreservingCalibrator().fit_with_checkpoints(Q, [2, 5]):
            out = cal.calibrate(Q[:, 0, :])
            np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=1e-9)
