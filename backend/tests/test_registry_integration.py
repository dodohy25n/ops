"""별도 임시 DB와 합성 시험용 모델로 승격, 실패, 캐시 교체를 검증합니다."""
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tensorflow import keras

from backend.serving_app import model_loader
from backend.serving_app.registry import MODEL_NAME, PROJECT_ROOT, active_version, configure_registry
from backend.serving_app.train_and_register import register_candidate


def fixture_model(always_alert=False):
    model = keras.Sequential([
        keras.layers.Input((20, 17)),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    weights = np.zeros((17, 1), dtype="float32")
    if not always_alert:
        weights[0, 0] = 10
    model.layers[-1].set_weights([weights, np.array([10 if always_alert else -5], dtype="float32")])
    return model


class RegistryIntegrationTests(unittest.TestCase):
    def test_initial_rejection_replacement_and_cache(self):
        y = np.r_[np.ones(40), np.zeros(960)]
        X = np.zeros((1000, 20, 17), dtype="float32")
        X[:40, :, 0] = 1
        settings = json.loads((PROJECT_ROOT / "serving_app/models/thresholds.json").read_text())
        scaler = PROJECT_ROOT / "serving_app/models/scaler.pkl"
        evaluation = {"name": "synthetic-control-flow-test", "usage": "not a trained v2 result"}
        old_cache = model_loader._model_cache
        try:
            with tempfile.TemporaryDirectory(prefix="fraud-registry-test-") as tmp, patch.dict(
                os.environ, {"FRAUD_MLFLOW_DIR": tmp, "MODEL_SOURCE": "mlflow"}
            ):
                first = register_candidate(fixture_model(), settings, scaler, X, y, evaluation)
                self.assertTrue(first["promoted"])
                self.assertEqual(first["version"], "1")
                cached = model_loader.reload_model()
                np.testing.assert_array_equal(cached.classify(X), y)
                client = configure_registry()

                failed = register_candidate(fixture_model(True), settings, scaler, X, y, evaluation)
                self.assertFalse(failed["promoted"])
                self.assertEqual(failed["version"], "2")
                self.assertIn("alert_rate", failed["gate"]["failed_checks"])
                self.assertEqual(str(active_version(client).version), "1")
                self.assertEqual(client.get_run(failed["run_id"]).data.tags["gate_passed"], "false")
                self.assertEqual(len(client.search_model_versions(f"name='{MODEL_NAME}'")), 2)
                self.assertEqual(client.get_model_version(MODEL_NAME, "2").tags["gate_passed"], "false")
                self.assertIs(model_loader.get_model(), cached)

                newer_settings = copy.deepcopy(settings)
                newer_settings["tau"] = 0.6
                second = register_candidate(fixture_model(), newer_settings, scaler, X, y, evaluation)
                self.assertTrue(second["promoted"])
                self.assertEqual(second["version"], "3")
                self.assertEqual(model_loader.get_model().version, "1")
                reloaded = model_loader.reload_model()
                self.assertEqual(reloaded.version, "3")
                self.assertEqual(reloaded.tau, 0.6)
                np.testing.assert_array_equal(reloaded.scaler.lo, cached.scaler.lo)
                np.testing.assert_array_equal(reloaded.classify(X), y)
                with patch.object(model_loader, "_load_model", side_effect=ValueError("damaged bundle")):
                    with self.assertRaises(ValueError):
                        model_loader.reload_model()
                self.assertIs(model_loader.get_model(), reloaded)
        finally:
            model_loader._model_cache = old_cache
            configure_registry()


if __name__ == "__main__":
    unittest.main()
