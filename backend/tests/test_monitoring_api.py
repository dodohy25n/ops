"""/predict/batch의 감시 기록과 재학습 요청, 재학습 도중 꺼진 서버를 다시 시작할 때의 상태 복구를 HTTP로 검증합니다."""
import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
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


def labelled_rows(prefix="sample-card"):
    """카드 2장 × 25일. 판정 거래 12건 중 첫 카드의 6건이 사기라 4건 창마다 사기가 2건입니다."""
    rows = sample_rows()
    for row in rows:
        fraud = row["카드KEY"] == "sample-card-01"
        row["카드KEY"] = row["카드KEY"].replace("sample-card", prefix)
        row["이상거래여부"] = "1" if fraud else "0"
    return rows


class ApiTestCase(unittest.TestCase):
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

    def restart(self):
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app())
        self.client.__enter__()

    def upload(self, rows):
        return self.client.post("/data/upload", files={
            "file": ("cards.csv", csv_bytes(rows, label=True), "text/csv")}).json()["upload_id"]

    def analyze(self, upload_id, model):
        with patch.object(model_loader, "get_model", return_value=model):
            response = self.client.post("/predict/batch", json={"upload_id": upload_id})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()


class MonitoringApiTests(ApiTestCase):
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



class InterruptedRetrainRecoveryTests(ApiTestCase):
    """서버가 재학습 도중 꺼진 뒤 다시 시작하면 남은 진행 중 상태를 failed로 바꿔 재학습 요청이 다시 가능해야 합니다."""

    def state_path(self):
        return monitoring_dir(self.tmp.name) / "state.json"

    def stuck_with(self, status):
        """실제 분석으로 재학습 요청 상태를 만든 뒤 상태만 status로 바꿉니다. 재학습은 실행되지 않습니다."""
        model = MonitoredModel(labelled_rows())
        with patch.object(retrain, "run"):
            self.analyze(self.upload(labelled_rows()), model)
        state = json.loads(self.state_path().read_text(encoding="utf-8"))
        state["retrain"]["status"] = status
        self.state_path().write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(state["consecutive"]["recall"], 3)
        return model

    def test_restart_marks_running_or_requested_retrain_failed(self):
        for status in ("requested", "running"):
            with self.subTest(status=status):
                self.restart_clean()
                self.stuck_with(status)
                self.restart()
                status_response = self.client.get("/monitoring/status?limit=500").json()
                state = status_response["state"]
                self.assertEqual(state["retrain"]["status"], "failed")
                self.assertEqual(state["retrain"]["error"], "서버 재시작으로 중단")
                self.assertIn("finished_at", state["retrain"])
                self.assertEqual(state["consecutive"], {"precision": 0, "recall": 0})
                self.assertEqual((state["model_version"], state["observations"], state["windows"]), ("1", 12, 3))
                failed = [e for e in status_response["events"] if e["type"] == "retrain_failed"]
                self.assertEqual(len(failed), 1)
                self.assertEqual((failed[0]["interrupted"], failed[0]["from_version"], failed[0]["error"]),
                                 (status, "1", "서버 재시작으로 중단"))
                self.restart()
                again = self.client.get("/monitoring/status?limit=500").json()
                self.assertEqual(again["state"], state)
                self.assertEqual(len([e for e in again["events"] if e["type"] == "retrain_failed"]), 1)

    def test_restart_leaves_finished_states_untouched(self):
        for status in ("idle", "promoted", "rejected", "failed"):
            with self.subTest(status=status):
                self.restart_clean()
                self.stuck_with(status)
                events = monitoring_dir(self.tmp.name) / "events.jsonl"
                before = (self.state_path().read_bytes(), events.read_bytes())
                self.restart()
                self.assertEqual((self.state_path().read_bytes(), events.read_bytes()), before)

    def test_restart_without_state_creates_nothing(self):
        self.restart()
        self.assertFalse(monitoring_dir(self.tmp.name).exists())
        self.assertIsNone(self.client.get("/monitoring/status").json()["state"])

    def test_retrain_can_be_requested_again_after_recovery(self):
        model = self.stuck_with("running")
        with patch.object(retrain, "run") as run:
            self.analyze(self.upload(labelled_rows("other-card")), model)
            run.assert_not_called()
        self.restart()
        with patch.object(retrain, "run") as run:
            monitoring = self.analyze(self.upload(labelled_rows("third-card")), model)["monitoring"]
            run.assert_called_once_with("1")
        self.assertTrue(monitoring["retrain_requested"])
        self.assertEqual(monitoring["retrain_window"], 8)
        state = self.client.get("/monitoring/status").json()["state"]
        self.assertEqual(state["retrain"]["status"], "requested")
        self.assertEqual(state["windows"], 9)

    def restart_clean(self):
        root = monitoring_dir(self.tmp.name)
        if root.exists():
            shutil.rmtree(root)


def monitoring_dir(api_dir):
    return Path(api_dir).resolve() / "monitoring"


if __name__ == "__main__":
    unittest.main()
