"""
v1을 2024년 상반기(valid)에 돌려 운영 기준값을 측정하고 backend/serving_app/models/thresholds.json에 저장합니다.
실행(프로젝트 최상위): python backend/scripts/measure_v1.py

설계 문서(docs/설계지표.md)의 [측정] 값이 모두 여기서 나옵니다.
  1) 분류 기준점 tau: F2가 가장 높은 확률값
  2) 배포 게이트 하한: R_min(v1 Recall 내림), P_min(사기율 x R_min / 경보 상한)
  3) 드리프트 트리거 기준: 1,000건 창마다 Precision, Recall을 재서 평균 - 2시그마, 평균 - 1시그마
  4) PSI 기준 분포: 학습 기간 거래와 v1 예측 확률의 분포, 그리고 정상 창의 PSI 범위
  5) beta=2의 근거: 이상거래와 정상 거래의 평균 승인 금액
"""
import json
import os
import sys
from datetime import date

import numpy as np
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.features import FEATURES, FraudScaler
from backend.serving_app.config import DATA_DIR, MODEL_DIR
from backend.serving_app.monitoring.metrics import (
    BETA,
    best_threshold,
    evaluate,
    precision_recall,
    psi,
    psi_reference,
)

MODEL_PATH = MODEL_DIR / "fraud_v1.keras"
OUT_PATH = MODEL_DIR / "thresholds.json"

ALERT_CAP = 0.06
WINDOW_SIZE = 1000
MIN_ALERTS = 30
MIN_FRAUDS = 20
CONSECUTIVE = 2
PSI_WARN = 0.1
PSI_ALERT = 0.25
PSI_FEATURES = ["amount_log", "hour", "overseas", "merchant_sales_bin", "installment"]


def window_stats(y, scores, tau, order):
    rows = []
    for start in range(0, len(order) - WINDOW_SIZE + 1, WINDOW_SIZE):
        idx = order[start : start + WINDOW_SIZE]
        y_w, pred_w = y[idx], (scores[idx] >= tau).astype(int)
        precision, recall = precision_recall(y_w, pred_w)
        rows.append(
            {
                "precision": precision if pred_w.sum() >= MIN_ALERTS else None,
                "recall": recall if y_w.sum() >= MIN_FRAUDS else None,
                "idx": idx,
            }
        )
    return rows


def summarize(values):
    v = np.array([x for x in values if x is not None])
    mean, std = float(v.mean()), float(v.std())
    return {
        "windows": int(len(v)),
        "mean": mean,
        "std": std,
        "min": float(v.min()),
        "minus_2sigma": mean - 2 * std,
        "minus_1sigma": mean - std,
    }


def amount_won(scaled_amount, scaler):
    i = FEATURES.index("amount_log")
    return np.expm1(scaled_amount * (scaler.hi[i] - scaler.lo[i]) + scaler.lo[i])


def main():
    model = keras.models.load_model(MODEL_PATH)
    scaler = FraudScaler.load(MODEL_DIR / "scaler.pkl")
    valid = np.load(DATA_DIR / "processed/valid.npz")
    X, y, dates = valid["X"], valid["y"], valid["dates"]
    scores = model.predict(X, batch_size=4096, verbose=0).flatten()

    tau = best_threshold(y, scores)
    m = evaluate(y, scores, tau)
    fraud_rate = float(y.mean())
    r_min = float(np.floor(m["recall"] * 100) / 100)
    p_min = fraud_rate * r_min / ALERT_CAP

    order = np.argsort(dates, kind="stable")
    windows = window_stats(y, scores, tau, order)
    precision_stats = summarize([w["precision"] for w in windows])
    recall_stats = summarize([w["recall"] for w in windows])

    train = np.load(DATA_DIR / "processed/train.npz")
    train_last, train_y = train["X"][:, -1, :], train["y"]
    feature_refs = {f: psi_reference(train_last[:, FEATURES.index(f)]) for f in PSI_FEATURES}
    score_ref = psi_reference(scores)
    last = X[:, -1, :]
    normal_psi = {
        f: max(psi(feature_refs[f], last[w["idx"], FEATURES.index(f)]) for w in windows) for f in PSI_FEATURES
    }
    normal_psi["score"] = max(psi(score_ref, scores[w["idx"]]) for w in windows)

    amounts = amount_won(train_last[:, FEATURES.index("amount_log")], scaler)
    fraud_amount = float(amounts[train_y == 1].mean())
    normal_amount = float(amounts[train_y == 0].mean())

    result = {
        "model": "v1",
        "measured_on": "valid (2024-01-01 ~ 2024-06-30)",
        "measured_at": date.today().isoformat(),
        "beta": BETA,
        "tau": tau,
        "valid": {**m, "fraud_rate": fraud_rate, "sequences": int(len(y))},
        "gate": {"r_min": r_min, "p_min": p_min, "alert_cap": ALERT_CAP},
        "trigger": {
            "window_size": WINDOW_SIZE,
            "min_alerts": MIN_ALERTS,
            "min_frauds": MIN_FRAUDS,
            "consecutive": CONSECUTIVE,
            "precision": precision_stats,
            "recall": recall_stats,
        },
        "psi": {
            "warn": PSI_WARN,
            "alert": PSI_ALERT,
            "features": feature_refs,
            "score": score_ref,
            "normal_window_max": normal_psi,
        },
        "amount": {"fraud_mean": fraud_amount, "normal_mean": normal_amount},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[1] 기준점 tau = {tau:.2f} (F2 최대)")
    print(
        f"[2] valid 성능: F2 {m['f2']:.4f} | Recall {m['recall']:.4f} | Precision {m['precision']:.4f} | "
        f"PR-AUC {m['pr_auc']:.4f} (기준선 {fraud_rate:.4f}) | 경보 비율 {m['alert_rate']:.4f}"
    )
    print(f"[3] 게이트 하한: R_min {r_min:.2f} | P_min {p_min:.4f} (경보 상한 {ALERT_CAP:.0%} 가정)")
    for name, s in [("Precision", precision_stats), ("Recall", recall_stats)]:
        print(
            f"[4] {name:9s} 창 {s['windows']}개: 평균 {s['mean']:.4f}, 표준편차 {s['std']:.4f}, "
            f"최솟값 {s['min']:.4f} -> 기준 -2시그마 {s['minus_2sigma']:.4f} / -1시그마 {s['minus_1sigma']:.4f}"
        )
    print("[5] 정상 창 PSI 최댓값: " + ", ".join(f"{k} {v:.4f}" for k, v in normal_psi.items()))
    print(f"[6] 평균 승인 금액: 이상거래 {fraud_amount:,.0f}원 / 정상 {normal_amount:,.0f}원")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
