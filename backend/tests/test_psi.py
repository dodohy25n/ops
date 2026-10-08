import unittest

import numpy as np

from backend.serving_app.monitoring.metrics import psi, psi_reference


class PsiTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.base = (rng.random(100_000) < 0.01).astype("float32")
        self.same = (rng.random(1_000) < 0.01).astype("float32")
        self.shifted = (rng.random(1_000) < 0.5).astype("float32")

    def test_categorical_detects_binary_ratio_change(self):
        reference = psi_reference(self.base, categorical=True)
        self.assertLess(psi(reference, self.same), 0.1)
        self.assertGreater(psi(reference, self.shifted), 0.25)

    def test_quantile_bins_miss_binary_change(self):
        # 이전 방식의 한계: 0이 대부분이면 0과 1이 같은 구간에 들어가 변화가 0으로 보입니다.
        self.assertEqual(psi(psi_reference(self.base), self.shifted), 0.0)

    def test_unseen_category_counts_as_shift(self):
        reference = psi_reference(np.array([0.0, 0.25, 0.5] * 100), categorical=True)
        self.assertGreater(psi(reference, np.full(300, 1.0)), 0.25)


if __name__ == "__main__":
    unittest.main()
