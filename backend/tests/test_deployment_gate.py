import unittest

import numpy as np

from backend.serving_app.monitoring.deployment_gate import check_gate
from backend.serving_app.monitoring.metrics import pr_auc


class GateTests(unittest.TestCase):
    def setUp(self):
        self.y = np.r_[np.ones(40), np.zeros(960)]
        self.scores = np.where(self.y == 1, 0.9, 0.1)
        self.policy = {"r_min": 0.95, "alert_cap": 0.06}

    def gate(self, scores=None, current_scores=None, current_tau=0.5):
        # 기본 운영 모델은 경보를 내지 않아 F2가 0이므로, 절대 기준만 따로 확인할 수 있습니다.
        current = np.full_like(self.scores, 0.1) if current_scores is None else current_scores
        return check_gate(self.y, self.scores if scores is None else scores, 0.5, self.policy,
                          current_scores=current, current_tau=current_tau)

    def test_candidate_equal_to_current_passes(self):
        result = self.gate(current_scores=self.scores)
        self.assertTrue(result["passed"])
        self.assertEqual(result["current"]["f2"], result["candidate"]["f2"])

    def test_current_model_is_required(self):
        with self.assertRaises(TypeError):
            check_gate(self.y, self.scores, 0.5, self.policy)

    def test_recall_boundary_and_failure(self):
        scores = self.scores.copy()
        scores[:2] = 0.1
        self.assertTrue(self.gate(scores)["passed"])
        scores[2] = 0.1
        self.assertIn("recall", self.gate(scores)["failed_checks"])

    def test_direct_alert_limit_blocks_precision_proxy_loophole(self):
        scores = self.scores.copy()
        scores[40:60] = 0.9
        self.assertTrue(self.gate(scores)["passed"])
        scores[60] = 0.9
        result = self.gate(scores)
        self.assertGreater(result["candidate"]["precision"], 0.57)
        self.assertIn("alert_rate", result["failed_checks"])

    def test_f2_regression_blocks_otherwise_acceptable_candidate(self):
        scores = self.scores.copy()
        scores[40:50] = 0.9
        result = self.gate(scores, current_scores=self.scores, current_tau=0.5)
        self.assertEqual(result["failed_checks"], ["f2_no_regression"])

    def test_each_model_uses_its_own_threshold_on_same_labels(self):
        result = self.gate(current_scores=self.scores, current_tau=0.95)
        self.assertEqual(result["current"]["recall"], 0)
        self.assertTrue(result["passed"])

    def test_insufficient_evaluation_is_not_promoted(self):
        small = check_gate(self.y[:100], self.scores[:100], 0.5, self.policy,
                           current_scores=self.scores[:100], current_tau=0.5)
        self.assertIn("enough_samples", small["failed_checks"])
        y = np.r_[np.ones(19), np.zeros(981)]
        scores = np.where(y, 0.9, 0.1)
        few = check_gate(y, scores, 0.5, self.policy, current_scores=scores, current_tau=0.5)
        self.assertIn("enough_frauds", few["failed_checks"])

    def test_invalid_predictions_fail_closed(self):
        for value in [np.nan, np.inf, -0.1, 1.1]:
            scores = self.scores.copy()
            scores[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.gate(scores)
        with self.assertRaises(ValueError):
            self.gate(self.scores[:-1])

    def test_average_precision_groups_tied_scores(self):
        self.assertAlmostEqual(pr_auc(self.y, np.ones(1000)), 0.04)
        self.assertAlmostEqual(pr_auc(self.y[::-1], np.ones(1000)), 0.04)


if __name__ == "__main__":
    unittest.main()
