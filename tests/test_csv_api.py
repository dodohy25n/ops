"""CSV 경계·학습 전처리 일치·오류 응답·결과 저장을 실제 HTTP 라우터로 검증합니다."""
import csv
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from data.csv_input import REQUIRED_COLUMNS, parse_csv
from data.features import FraudScaler, build_sequences, encode, sort_transactions
from scripts.make_api_sample import sample_rows
from serving_app import model_loader
from serving_app.main import create_app


def csv_bytes(rows, label=False, encoding="utf-8-sig"):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=REQUIRED_COLUMNS + (["이상거래여부"] if label else []))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode(encoding)


class FixtureModel:
    """HTTP 연결 검증용. 실제 학습 모델이나 게이트 통과 모델을 뜻하지 않습니다."""
    version = "v1-local"
    tau = 0.9

    def __init__(self, rows):
        self.scaler = FraudScaler().fit(encode(pd.DataFrame(rows)))
        self.seen = []

    def predict_scores(self, X):
        self.seen.append(X.copy())
        return X[:, -1, 0]


class CsvApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="csv-api-test-")
        self.env = patch.dict(os.environ, {"FRAUD_API_DIR": self.tmp.name,
                                          "MODEL_SOURCE": "local", "LOADING_MODE": "lazy",
                                          "FRAUD_CORS_ORIGINS": "http://localhost:5173"})
        self.env.start()
        self.cache = model_loader._model_cache
        self.error = model_loader._last_load_error
        model_loader._model_cache = None
        model_loader._last_load_error = None
        self.client = TestClient(create_app())
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        model_loader._model_cache = self.cache
        model_loader._last_load_error = self.error
        self.env.stop()
        self.tmp.cleanup()

    def upload(self, rows=None, label=False, encoding="utf-8-sig"):
        return self.client.post("/data/upload", files={
            "file": ("cards.csv", csv_bytes(sample_rows() if rows is None else rows, label, encoding), "text/csv")
        })

    def test_startup_health_swagger_and_screen_without_model(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertFalse(self.client.get("/health").json()["model_loaded"])
        self.assertEqual(self.client.get("/docs").status_code, 200)
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/predict/batch", paths)
        self.assertNotIn("/predict", paths)
        self.assertIn("카드 거래 CSV 분석", self.client.get("/").text)

    def test_upload_validation_metadata_and_cp949(self):
        response = self.upload(encoding="cp949")
        self.assertEqual(response.status_code, 201, response.text)
        meta = response.json()
        self.assertEqual((meta["rows"], meta["cards"], meta["predictable_rows"], meta["excluded_rows"]), (50, 2, 12, 38))
        self.assertEqual(self.client.get("/data/uploads").json()[0], meta)
        self.assertEqual(self.client.get("/data/status").json()["latest"], meta)
        self.assertEqual(self.client.get("/data/uploads/" + meta["upload_id"]).json(), meta)

    def test_bad_values_columns_and_duplicates_are_rejected_before_storage(self):
        for column, bad in [("통합승인금액", "NaN"), ("승인일자", "20240230"),
                            ("승인시간대", "24"), ("국내해외여부", "9"), ("승인SEQ", "1.5")]:
            rows = sample_rows()
            rows[0][column] = bad
            with self.subTest(column=column):
                self.assertEqual(self.upload(rows).status_code, 422)
        rows = sample_rows()
        rows.append(rows[0].copy())
        self.assertEqual(self.upload(rows).status_code, 422)
        self.assertEqual(self.client.post("/data/upload", files={"file": ("x.csv", b"Date,Close\n2024,123\n")}).status_code, 422)
        self.assertEqual(self.client.get("/data/uploads").json(), [])

    def test_missing_model_returns_503_and_keeps_upload(self):
        meta = self.upload().json()
        with patch.object(model_loader, "get_model", side_effect=FileNotFoundError("모델 파일 없음")):
            response = self.client.post("/predict/batch", json={"upload_id": meta["upload_id"]})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "model_unavailable")
        self.assertEqual(len(self.client.get("/data/uploads").json()), 1)
        self.assertEqual(self.client.get("/predict/results").json(), [])

    def test_batch_matches_training_sequences_and_persists_paginated_results(self):
        rows = sample_rows()[::-1]  # CSV 순서와 카드별 시간순이 달라도 동일한 입력이어야 합니다.
        for row in rows:
            row["이상거래여부"] = "1" if int(row["승인SEQ"]) >= 23 else "0"
        model = FixtureModel(rows)
        meta = self.upload(rows, label=True).json()
        with patch.object(model_loader, "get_model", return_value=model):
            response = self.client.post("/predict/batch", json={"upload_id": meta["upload_id"]})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        expected, _, _, _ = build_sequences(sort_transactions(pd.DataFrame(rows)), model.scaler)
        np.testing.assert_array_equal(np.concatenate(model.seen), expected)
        self.assertEqual(result["predictions"], 12)
        self.assertEqual(result["model_role"], "local_candidate")
        self.assertEqual(result["labelled_predictions"], 12)
        self.assertEqual(result["metrics"]["samples"], 12)
        path = "/predict/results/" + result["analysis_id"]
        self.assertEqual(self.client.get(path).json(), result)
        page = self.client.get(path + "/transactions?offset=2&limit=3").json()
        self.assertEqual(len(page["items"]), 3)
        self.assertEqual(page["total"], 12)
        self.assertIsInstance(page["items"][0]["is_fraud"], bool)
        download = self.client.get(path + "/download")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(download.text)))), 12)
        self.assertEqual(self.client.get(path + "/transactions?offset=20").json()["items"], [])

    def test_period_uses_earlier_rows_as_context_without_scoring_them(self):
        rows = sample_rows()
        model = FixtureModel(rows)
        meta = self.upload(rows).json()
        with patch.object(model_loader, "get_model", return_value=model):
            result = self.client.post("/predict/batch", json={"upload_id": meta["upload_id"],
                                      "start_date": "2024-08-23", "end_date": "2024-08-24"}).json()
        self.assertEqual((result["input_rows"], result["period_rows"], result["predictions"]), (50, 4, 4))
        self.assertEqual(result["excluded_rows"], 0)
        self.assertIsNone(result["metrics"])
        self.assertEqual(np.concatenate(model.seen).shape, (4, 20, 17))

    def test_insufficient_history_and_bad_period_do_not_load_model(self):
        meta = self.upload(sample_rows()[:19]).json()
        with patch.object(model_loader, "get_model") as loader:
            self.assertEqual(self.client.post("/predict/batch", json={"upload_id": meta["upload_id"]}).status_code, 422)
            loader.assert_not_called()
        self.assertEqual(self.client.post("/predict/batch", json={"upload_id": meta["upload_id"],
                         "start_date": "2024-08-25", "end_date": "2024-08-01"}).status_code, 422)

    def test_partial_labels_only_evaluate_known_predictions(self):
        rows = sample_rows()
        for row in rows:
            row["이상거래여부"] = "0" if row["카드KEY"] == "sample-card-01" else ""
        meta = self.upload(rows, label=True).json()
        with patch.object(model_loader, "get_model", return_value=FixtureModel(rows)):
            result = self.client.post("/predict/batch", json={"upload_id": meta["upload_id"]}).json()
        self.assertEqual(result["labelled_predictions"], 6)
        self.assertIsNone(result["metrics"]["recall"])
        self.assertIsNone(result["metrics"]["pr_auc"])

    def test_unknown_ids_and_query_limits(self):
        self.assertEqual(self.client.post("/predict/batch", json={"upload_id": "f" * 32}).status_code, 404)
        self.assertEqual(self.client.post("/predict/batch", json={"upload_id": "../bad"}).status_code, 422)
        self.assertEqual(self.client.get("/predict/results/bad").status_code, 404)
        self.assertEqual(self.client.get("/predict/results/" + "f" * 32 + "/transactions?limit=1001").status_code, 422)

    def test_metrics_logging_and_upload_size_limit(self):
        self.client.get("/health")
        summary = self.client.get("/metrics/summary").json()
        self.assertGreaterEqual(summary["request_count"], 1)
        self.assertEqual(summary["error_rate"], 0)
        log = self.client.get("/logs/requests.jsonl").json()
        self.assertIn('"duration_ms"', log["content"])
        for line in log["content"].splitlines():
            self.assertIsInstance(json.loads(line)["status"], int)
        with patch("serving_app.routers.data.max_upload_bytes", return_value=10):
            self.assertEqual(self.upload().status_code, 413)

    def test_frontend_cors_preflight_and_error_responses(self):
        headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                   "Access-Control-Request-Headers": "Content-Type"}
        response = self.client.options("/predict/batch", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], headers["Origin"])
        headers["Origin"] = "http://unlisted.example"
        self.assertEqual(self.client.options("/predict/batch", headers=headers).status_code, 400)
        response = self.client.post("/predict/batch", json={"upload_id": "f" * 32},
                                    headers={"Origin": "http://localhost:5173"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5173")

    def test_eager_missing_model_keeps_health_available(self):
        with patch.dict(os.environ, {"LOADING_MODE": "eager"}), patch.object(
            model_loader, "_load_model", side_effect=FileNotFoundError("모델 파일 없음")
        ), TestClient(create_app()) as client:
            state = client.get("/health").json()
            self.assertEqual(state["status"], "ok")
            self.assertFalse(state["model_loaded"])
            self.assertEqual(state["model"]["error"], "모델 파일 없음")

    def test_model_cache_reuse_and_failed_reload_preserves_existing_model(self):
        fixture = FixtureModel(sample_rows())
        with patch.object(model_loader, "_load_model", return_value=fixture) as loader:
            self.assertIs(model_loader.get_model(), fixture)
            self.assertIs(model_loader.get_model(), fixture)
            loader.assert_called_once()
        with patch.object(model_loader, "_load_model", side_effect=ValueError("damaged bundle")):
            with self.assertRaises(ValueError):
                model_loader.reload_model()
        self.assertIs(model_loader.get_model(), fixture)


if __name__ == "__main__":
    unittest.main()
