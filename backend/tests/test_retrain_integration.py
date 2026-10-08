"""감시 기록에서 재학습 → 게이트 → 교체까지 retrain.run 한 번의 흐름을 임시 저장소와 합성 모델로 검증합니다.

합성 모델은 마지막 거래의 금액 피처만 봅니다. 금액이 큰 거래를 사기로 표시한 합성 데이터에서
기준 τ가 너무 높은 v1은 사기를 하나도 못 잡고, 재학습이 τ를 다시 고르면 게이트를 통과합니다.
모든 거래에 알림을 내는 v1에서 출발한 후보는 알림 비율 상한에 걸려 거절됩니다.
"""
import copy
import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from tensorflow import keras

from data.storage import save_upload
from backend.scripts.make_api_sample import sample_rows
from backend.serving_app import model_loader, retrain
from backend.serving_app.batch_service import analyze_batch, prepare_batch
from backend.serving_app.monitoring import drift_monitor
from backend.serving_app.registry import MODEL_NAME, PROJECT_ROOT, active_version, configure_registry
from backend.serving_app.schemas import BatchRequest
from backend.serving_app.train_and_register import register_baseline
from backend.tests.fixtures import write_test_scaler

CARDS = 50
DAYS = 119
FRAUD_AMOUNT = "5000000"


def last_amount_model(always_alert=False):
    """마지막 거래의 금액 피처 하나로 점수를 내는 시험용 모델입니다. 실제 학습 결과가 아닙니다."""
    model = keras.Sequential([
        keras.layers.Input((20, 17)),
        keras.layers.Cropping1D((19, 0)),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    weights = np.zeros((17, 1), dtype="float32")
    if not always_alert:
        weights[0, 0] = 20
    model.layers[-1].set_weights([weights, np.array([10 if always_alert else -16], dtype="float32")])
    return model


def synthetic_transactions():
    """카드 50장이 하루 한 건씩 119일 거래합니다. 판정 가능한 거래는 100일 × 50장 = 5,000건입니다.

    날짜마다 판정 거래 수가 같아 60/20/20 경계가 날짜 경계와 맞습니다. 사기 비율은 약 3%입니다.
    """
    template = sample_rows()[0]
    rows = []
    for day in range(DAYS):
        approved = (date(2024, 7, 1) + timedelta(days=day)).strftime("%Y%m%d")
        for card in range(CARDS):
            fraud = (card + day) % 30 == 0
            rows.append({**template, "카드KEY": f"card-{card:02d}", "승인일자": approved,
                         "승인시간대": "12", "승인SEQ": str(day + 1),
                         "통합승인금액": FRAUD_AMOUNT if fraud else str(10000 + (card * 37 + day) % 50 * 100),
                         "이상거래여부": "1" if fraud else "0"})
    return pd.DataFrame(rows)


class RetrainIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="retrain-test-")
        self.env = patch.dict(os.environ, {"FRAUD_API_DIR": self.tmp.name, "FRAUD_MLFLOW_DIR": self.tmp.name,
                                          "MODEL_SOURCE": "mlflow"})
        self.env.start()
        self.cache = model_loader._model_cache
        model_loader._model_cache = None
        self.settings = json.loads((PROJECT_ROOT / "serving_app/models/thresholds.json").read_text())
        self.scaler = write_test_scaler(Path(self.tmp.name) / "scaler.pkl")

    def tearDown(self):
        model_loader._model_cache = self.cache
        self.env.stop()
        configure_registry()
        self.tmp.cleanup()

    def start_with(self, model, tau):
        settings = copy.deepcopy(self.settings)
        settings["tau"] = tau
        register_baseline(model, settings, self.scaler, np.zeros((4, 20, 17), dtype="float32"))
        current = model_loader.get_model()
        self.assertEqual(current.version, "1")
        upload = save_upload(synthetic_transactions(), "synthetic.csv", {"rows": CARDS * DAYS})
        request = BatchRequest(upload_id=upload["upload_id"])
        summary, observations = analyze_batch(request, prepare_batch(request), current)
        self.assertEqual(summary["predictions"], 5000)
        monitoring = drift_monitor.record(summary, observations, current)
        self.assertTrue(monitoring["retrain_requested"])
        return current

    def events(self, kind):
        return [e for e in drift_monitor.read_jsonl("events") if e["type"] == kind]

    def assert_ordered_disjoint_periods(self, periods):
        names = ["fine_tune", "tau", "holdout"]
        self.assertEqual(list(periods), names)
        self.assertEqual([periods[n]["samples"] for n in names], [3000, 1000, 1000])
        for name in names:
            self.assertLessEqual(periods[name]["start"], periods[name]["end"])
        # 날짜 경계와 구간 경계가 맞으므로 앞 구간의 마지막 날은 뒤 구간의 첫날보다 엄격히 앞서야 합니다.
        self.assertLess(periods["fine_tune"]["end"], periods["tau"]["start"])
        self.assertLess(periods["tau"]["end"], periods["holdout"]["start"])
        self.assertEqual((periods["fine_tune"]["start"], periods["holdout"]["end"]), ("20240720", "20241027"))

    def test_candidate_passing_gate_becomes_champion_and_resets_monitoring(self):
        self.start_with(last_amount_model(), tau=0.99)

        summary = retrain.run("1")

        self.assertEqual(summary["status"], "promoted")
        self.assertEqual(summary["candidate_version"], "2")
        self.assertEqual(summary["failed_checks"], [])
        self.assertEqual(summary["current"]["recall"], 0)
        self.assertGreaterEqual(summary["candidate"]["recall"], 0.95)
        self.assertLess(summary["tau"], 0.99)
        self.assertEqual(str(active_version(configure_registry()).version), "2")
        self.assertEqual(configure_registry().get_model_version(MODEL_NAME, "2").tags["gate_passed"], "true")
        self.assertEqual(model_loader._model_cache.version, "2")
        self.assertEqual(model_loader.get_model().version, "2")
        self.assertEqual(model_loader.get_model().tau, summary["tau"])
        self.assertEqual(drift_monitor.load_state()["retrain"]["status"], "promoted")

        event = self.events("retrain_promoted")[-1]
        self.assertEqual((event["from_version"], event["candidate_version"]), ("1", "2"))
        self.assert_ordered_disjoint_periods(event["periods"])

        # 다음 분석이 감시 상태를 열면 v1 기록은 보관 파일로 옮겨지고 v2 창은 처음부터 셉니다.
        state = drift_monitor.current_state(model_loader.get_model().version)
        self.assertEqual((state["model_version"], state["observations"], state["windows"]), ("2", 0, 0))
        self.assertEqual(state["retrain"], {"status": "idle"})
        root = drift_monitor.monitor_dir()
        self.assertTrue((root / "observations-v1.csv").is_file())
        self.assertTrue((root / "windows-v1.jsonl").is_file())
        self.assertFalse((root / "observations.csv").exists())
        self.assertEqual(self.events("window_reset")[-1]["to_version"], "2")

    def test_candidate_failing_gate_is_rejected_and_champion_stays(self):
        current = self.start_with(last_amount_model(always_alert=True), tau=0.28)

        summary = retrain.run("1")

        self.assertEqual(summary["status"], "rejected")
        self.assertEqual(summary["candidate_version"], "2")
        self.assertIn("alert_rate", summary["failed_checks"])
        self.assertGreater(summary["candidate"]["alert_rate"], self.settings["gate"]["alert_cap"])
        client = configure_registry()
        self.assertEqual(str(active_version(client).version), "1")
        self.assertEqual(client.get_model_version(MODEL_NAME, "2").tags["gate_passed"], "false")
        self.assertIs(model_loader.get_model(), current)

        state = drift_monitor.load_state()
        self.assertEqual(state["model_version"], "1")
        self.assertEqual(state["retrain"]["status"], "rejected")
        self.assertEqual(state["consecutive"], {"precision": 0, "recall": 0})
        event = self.events("retrain_rejected")[-1]
        self.assertEqual(event["failed_checks"], summary["failed_checks"])
        self.assert_ordered_disjoint_periods(event["periods"])
        self.assertEqual(self.events("retrain_promoted"), [])


if __name__ == "__main__":
    unittest.main()
