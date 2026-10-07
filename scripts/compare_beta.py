"""
β를 1~5로 바꿔 가며 v1의 기준점 τ와 그 결과를 비교합니다.

β는 무엇을 더 중요하게 볼지 정하는 기준이라서, 점수가 가장 높은 β를 고를 수는 없습니다.
β마다 자기 기준으로는 항상 최고점이 나오기 때문입니다. 그래서 β를 한 단계 올릴 때
사기를 몇 건 더 잡고, 그 대가로 정상 고객을 몇 명 더 막는지를 비교해 β를 정합니다.

실행: python scripts/compare_beta.py   (train_baseline_v1.py 이후)
"""
import json
import os
import sys

import numpy as np
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving_app.monitoring.metrics import f_beta, precision_recall

MODEL_PATH = "serving_app/models/fraud_v1.keras"
THRESHOLDS_PATH = "serving_app/models/thresholds.json"
BETAS = [1, 2, 3, 4, 5]


def main():
    with open(THRESHOLDS_PATH) as f:
        alert_cap = json.load(f)["gate"]["alert_cap"]
    valid = np.load("data/processed/valid.npz")
    X, y = valid["X"], valid["y"]
    scores = keras.models.load_model(MODEL_PATH).predict(X, batch_size=4096, verbose=0).flatten()

    taus = np.round(np.arange(0.01, 1.0, 0.01), 2)
    pr = [precision_recall(y, (scores >= t).astype(int)) for t in taus]

    rows = []
    for beta in BETAS:
        i = int(np.argmax([f_beta(p, r, beta) for p, r in pr]))
        pred = scores >= taus[i]
        rows.append(
            {
                "beta": beta,
                "tau": float(taus[i]),
                "recall": pr[i][1],
                "precision": pr[i][0],
                "alert_rate": float(pred.mean()),
                "missed": int((~pred & (y == 1)).sum()),
                "blocked": int((pred & (y == 0)).sum()),
            }
        )

    print(f"검증 {len(y):,}건, 실제 사기 {int(y.sum()):,}건, 경보 상한 {alert_cap:.0%}\n")
    print(" β |  τ   | Recall Precision | 경보 비율 | 놓친 사기 | 잘못 막은 정상 | 이전 β 대비: 더 잡은 사기 / 더 막은 정상 (사기 1건당)")
    prev = None
    for r in rows:
        line = (
            f" {r['beta']} | {r['tau']:.2f} | {r['recall']:.3f}  {r['precision']:.3f}    | "
            f"{r['alert_rate'] * 100:5.2f}%{' 초과' if r['alert_rate'] > alert_cap else '     '} | "
            f"{r['missed']:>7,} | {r['blocked']:>10,} |"
        )
        if prev:
            caught = prev["missed"] - r["missed"]
            blocked = r["blocked"] - prev["blocked"]
            line += f" {caught:>4} / {blocked:>5,} ({blocked / caught:.1f}명)"
        print(line)
        prev = r


if __name__ == "__main__":
    main()
