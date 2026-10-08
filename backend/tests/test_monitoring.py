"""판정 창 감시: 창 닫기, 판정 보류, 연속 미달 시 재학습 요청, 중복 제외, 운영 모델 교체 시 초기화."""
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backend.serving_app.monitoring import drift_monitor
from backend.serving_app.monitoring.metrics import psi_reference


class FakeModel:
    def __init__(self, version="1"):
        self.version = version
        self.tau = 0.5
        self.settings = {
            "trigger": {"window_size": 10, "min_alerts": 1, "min_frauds": 1, "consecutive": 2,
                        "precision": {"minus_2sigma": 0.5}, "recall": {"minus_2sigma": 0.5}},
            "psi": {"warn": 0.1, "alert": 0.25,
                    "features": {"overseas": psi_reference([0.0] * 99 + [1.0], categorical=True)},
                    "score": psi_reference([i / 100 for i in range(100)])},
        }


def window(start, caught, frauds=2, overseas=0.0):
    """사기 frauds건 중 caught건을 잡은 10건 창을 만듭니다."""
    rows = []
    for i in range(10):
        fraud = i < frauds
        rows.append({"upload_id": "u", "analysis_id": "a", "card_key": f"c{start + i}",
                     "approval_date": "20241001", "hour": 1, "approval_seq": str(start + i),
                     "actual": "1" if fraud else "0", "score": 0.9 if (fraud and i < caught) else 0.1,
                     "alert": fraud and i < caught, "psi_overseas": overseas})
    return pd.DataFrame(rows)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="monitor-test-")
        self.env = patch.dict(os.environ, {"FRAUD_API_DIR": self.tmp.name})
        self.env.start()
        self.model = FakeModel()
        self.next = 0

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def feed(self, *windows, model=None):
        frames = []
        for caught, frauds in windows:
            frames.append(window(self.next, caught, frauds))
            self.next += 10
        return drift_monitor.record({"analysis_id": "a"}, pd.concat(frames, ignore_index=True), model or self.model)

    def test_windows_close_every_window_size(self):
        result = self.feed((2, 2), (2, 2))
        self.assertEqual(len(result["windows_closed"]), 2)
        self.assertEqual(result["pending"], 0)
        partial = drift_monitor.record({"analysis_id": "b"}, window(self.next, 2).iloc[:5], self.model)
        self.assertEqual((len(partial["windows_closed"]), partial["pending"]), (0, 5))

    def test_trigger_at_second_consecutive_window_even_if_later_window_recovers(self):
        result = self.feed((2, 2), (0, 2), (0, 2), (2, 2))
        self.assertTrue(result["retrain_requested"])
        self.assertEqual(result["retrain_reason"], ["recall"])
        self.assertEqual(result["retrain_window"], 3)
        self.assertEqual(result["consecutive"]["recall"], 0)

    def test_insufficient_window_holds_counter(self):
        result = self.feed((0, 2), (0, 0), (0, 2))
        self.assertEqual([w["status"]["recall"] for w in result["windows_closed"]], ["below", "insufficient", "below"])
        self.assertTrue(result["retrain_requested"])

    def test_no_second_request_while_waiting_and_rejection_resets(self):
        self.assertTrue(self.feed((0, 2), (0, 2))["retrain_requested"])
        self.assertFalse(self.feed((0, 2), (0, 2))["retrain_requested"])
        drift_monitor.update_retrain("1", status="rejected")
        self.assertEqual(drift_monitor.load_state()["consecutive"], {"precision": 0, "recall": 0})
        self.assertTrue(self.feed((0, 2), (0, 2))["retrain_requested"])

    def test_duplicates_are_skipped(self):
        first = window(0, 2)
        drift_monitor.record({"analysis_id": "a"}, first, self.model)
        again = drift_monitor.record({"analysis_id": "b"}, first, self.model)
        self.assertEqual((again["added"], again["skipped_duplicates"]), (0, 10))

    def test_new_champion_resets_windows_and_keeps_old_records(self):
        self.feed((0, 2))
        result = self.feed((0, 2), model=FakeModel("2"))
        self.assertEqual(result["consecutive"]["recall"], 1)
        self.assertEqual(drift_monitor.load_state()["model_version"], "2")
        self.assertTrue((drift_monitor.monitor_dir() / "observations-v1.csv").is_file())
        self.assertIn("window_reset", [e["type"] for e in drift_monitor.read_jsonl("events")])

    def test_psi_alert_is_logged_without_retrain(self):
        frame = window(0, 2, overseas=1.0)
        result = drift_monitor.record({"analysis_id": "a"}, frame, self.model)
        self.assertEqual(result["windows_closed"][0]["psi_level"], "alert")
        self.assertFalse(result["retrain_requested"])
        self.assertEqual(drift_monitor.read_jsonl("events")[-1]["type"], "psi_alert")


class SplitTests(unittest.TestCase):
    def test_time_split_is_ordered_and_disjoint(self):
        from backend.serving_app.retrain import split_by_time

        train, tune, hold = split_by_time(40_000)
        self.assertEqual((train.stop, tune.start, tune.stop, hold.start, hold.stop), (24_000, 24_000, 32_000, 32_000, 40_000))


if __name__ == "__main__":
    unittest.main()
