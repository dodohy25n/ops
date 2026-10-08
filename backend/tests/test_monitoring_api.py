"""/predict/batch가 감시 기록을 남기고, 연속 미달이면 재학습을 백그라운드로 요청하는지 HTTP로 검증합니다."""
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from data.features import FraudScaler, encode
from backend.scripts.make_api_sample import sample_rows
from backend.serving_app import model_loader, retrain
from backend.serving_app.main import create_app
from backend.serving_app.registry import PROJECT_ROOT
from backend.tests.test_csv_api import csv_bytes


class MonitoredModel:
    """실제 thresholds.json 형식의 기준을 가진 시험용 모델입니다. 창 크기만 4건으로 줄였고 알림을 내지 않습니다."""
    version = "1"
    tau = 0.5

    def __init__(self, rows):
        self.scaler = FraudScaler().fit(encode(pd.DataFrame(rows)))
        self.settings = copy.deepcopy(json.loads((PROJECT_ROOT / "serving_app/models/thresholds.json").read_text()))
        self.settings["trigger"].update(window_size=4, min_alerts=1, min_frauds=1)

    def predict_scores(self, X):
        return X[:, -1, 0] * 0


def labelled_rows():
    """카드 2장 × 25일. 판정 거래 12건 중 sample-card-01의 6건이 사기라 4건 창마다 사기가 2건입니다."""
    rows = sample_rows()
    for row in rows:
        row["이상거래여부"] = "1" if row["카드KEY"] == "sample-card-01" else "0"
    return rows


class MonitoringApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="monitoring-api-test-")
        self.env = patch.dict(os.environ, {"FRAUD_API_DIR": self.tmp.name, "FRAUD_MLFLOW_DIR": self.tmp.name,
                                          "MODEL_SOURCE": "mlflow", "LOADING_MODE": "lazy"})
        self.env.start()
        self.cache = model_loader._model_cache
        model_loader._model_cache = None
        self.client = TestClient(create_app())
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        model_loader._model_cache = self.cache
        self.env.stop()
        self.tmp.cleanup()

    def analyze(self, upload_id, model):
        with patch.object(model_loader, "get_model", return_value=model):
            response = self.client.post("/predict/batch", json={"upload_id": upload_id})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_status_before_any_analysis(self):
        status = self.client.get("/monitoring/status").json()
        self.assertEqual(status, {"state": None, "windows": [], "events": []})
        self.assertEqual(self.client.get("/monitoring/status?limit=0").status_code, 422)

    def test_batch_closes_windows_requests_retrain_once_and_reports_status(self):
        rows = labelled_rows()
        model = MonitoredModel(rows)
        upload = self.client.post("/data/upload", files={"file": ("cards.csv", csv_bytes(rows, label=True), "text/csv")})
        upload_id = upload.json()["upload_id"]

        with patch.object(retrain, "run") as run:
            result = self.analyze(upload_id, model)
            run.assert_called_once_with("1")
            monitoring = result["monitoring"]
            self.assertEqual((monitoring["model_version"], monitoring["added"], monitoring["pending"]), ("1", 12, 0))
            self.assertEqual([w["window"] for w in monitoring["windows_closed"]], [1, 2, 3])
            self.assertEqual([w["status"] for w in monitoring["windows_closed"]],
                             [{"precision": "insufficient", "recall": "below"}] * 3)
            self.assertEqual([w["recall"] for w in monitoring["windows_closed"]], [0, 0, 0])
            self.assertTrue(monitoring["retrain_requested"])
            self.assertEqual((monitoring["retrain_reason"], monitoring["retrain_window"]), (["recall"], 2))
            self.assertEqual(monitoring["consecutive"], {"precision": 0, "recall": 3})
            self.assertEqual(self.client.get("/predict/results/" + result["analysis_id"]).json()["monitoring"],
                             monitoring)

            # 같은 CSV를 다시 분석하면 이미 센 거래라 창이 늘지 않고, 재학습을 기다리는 동안 다시 요청하지 않습니다.
            again = self.analyze(upload_id, model)["monitoring"]
            self.assertEqual((again["added"], again["skipped_duplicates"], again["windows_closed"]), (0, 12, []))
            self.assertFalse(again["retrain_requested"])
            run.assert_called_once()

        status = self.client.get("/monitoring/status").json()
        self.assertEqual(set(status), {"state", "windows", "events"})
        state = status["state"]
        self.assertEqual((state["model_version"], state["observations"], state["windows"]), ("1", 12, 3))
        self.assertEqual(state["retrain"]["status"], "requested")
        self.assertEqual((state["retrain"]["reason"], state["retrain"]["window"]), (["recall"], 2))
        self.assertEqual([w["window"] for w in status["windows"]], [1, 2, 3])
        self.assertEqual([w["retrain_requested"] for w in status["windows"]], [False, True, False])
        for window in status["windows"]:
            self.assertEqual(window["samples"], 4)
            self.assertEqual(set(window["psi"]), set(model.settings["psi"]["features"]) | {"score"})
            self.assertIn(window["psi_level"], {"ok", "warn", "alert"})
        requested = [e for e in status["events"] if e["type"] == "retrain_requested"]
        self.assertEqual(len(requested), 1)
        self.assertEqual((requested[0]["window"], requested[0]["model_version"]), (2, "1"))
        self.assertEqual(len(self.client.get("/monitoring/status?limit=1").json()["windows"]), 1)

    def test_model_without_monitoring_settings_skips_monitoring(self):
        rows = labelled_rows()
        model = MonitoredModel(rows)
        model.settings = {}
        upload_id = self.client.post("/data/upload", files={
            "file": ("cards.csv", csv_bytes(rows, label=True), "text/csv")}).json()["upload_id"]
        with patch.object(retrain, "run") as run:
            result = self.analyze(upload_id, model)
        self.assertIsNone(result["monitoring"])
        run.assert_not_called()
        self.assertIsNone(self.client.get("/monitoring/status").json()["state"])


if __name__ == "__main__":
    unittest.main()
