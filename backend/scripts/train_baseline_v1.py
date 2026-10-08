"""
첫 번째 버전 모델(v1)을 학습해 serving_app/models/fraud_v1.keras로 저장합니다.

실행 순서
  1) python scripts/prepare_data.py      시퀀스와 스케일러 만들기
  2) python scripts/train_baseline_v1.py 이 파일
  3) python scripts/measure_v1.py        검증 구간에서 기준값 측정

학습에는 2021~2023년 시퀀스만 씁니다. 2024년 상반기(valid)는 기준값을 재는 시험지라서
학습 중에는 열어 보지 않습니다. 대신 학습 데이터 끝쪽 카드 10%를 떼어 학습이 나아지는지만
확인하고, 두 epoch 연속 나아지지 않으면 멈춘 뒤 가장 좋았던 가중치로 되돌립니다.
"""
import os
import sys
import time

import numpy as np
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.serving_app.lstm_model import build_model

MODEL_PATH = "serving_app/models/fraud_v1.keras"
SEED = 42
MAX_EPOCHS = 15
BATCH_SIZE = 512


def main():
    keras.utils.set_random_seed(SEED)
    data = np.load("data/processed/train.npz")
    X, y = data["X"], data["y"]
    print(f"학습 시퀀스 {len(y):,}개, 이상거래 비율 {y.mean():.4f}")

    model = build_model()
    stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=2, restore_best_weights=True
    )
    started = time.time()
    history = model.fit(
        X,
        y,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        validation_split=0.1,
        callbacks=[stop],
        verbose=2,
    )
    best = int(np.argmax(history.history["val_pr_auc"]))
    print(
        f"학습 종료: {len(history.history['loss'])} epoch, {time.time() - started:.0f}초 | "
        f"최고 val PR-AUC {history.history['val_pr_auc'][best]:.4f} (epoch {best + 1})"
    )

    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
