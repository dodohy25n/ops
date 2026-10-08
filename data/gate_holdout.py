"""2024년 7월을 최초 게이트에, 8~12월을 이후 운영 시연에 씁니다."""
import hashlib

import numpy as np
import pandas as pd

from data.features import FraudScaler, build_sequences, sort_transactions
from backend.serving_app.registry import PROJECT_ROOT

GATE_START = "20240701"
GATE_END = "20240731"
OPS_START = "20240801"


def load_gate_holdout():
    rows = pd.read_csv(PROJECT_ROOT / "data/processed/ops_rows.csv", dtype=str)
    rows = sort_transactions(rows[rows["승인일자"] <= GATE_END])
    scaler = FraudScaler.load(PROJECT_ROOT / "serving_app/models/scaler.pkl")
    X, y, dates, _ = build_sequences(rows, scaler)
    mask = (dates >= GATE_START) & (dates <= GATE_END)
    X, y, dates = X[mask], y[mask], dates[mask].astype("U8")
    if not len(y):
        raise ValueError("7월 홀드아웃이 없습니다. prepare_data.py의 운영 데이터를 확인하세요.")
    digest = hashlib.sha256()
    for array in (X, y, dates):
        digest.update(np.ascontiguousarray(array).tobytes())
    return X, y, {
        "name": "initial-gate-2024-07",
        "start": GATE_START,
        "end": GATE_END,
        "sha256": digest.hexdigest(),
        "usage": "evaluation only; not training or threshold tuning",
    }
