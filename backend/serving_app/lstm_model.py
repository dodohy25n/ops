"""
카드 이상거래 판별용 LSTM 아키텍처입니다. 학습, MLflow 등록, fine-tuning이 함께 씁니다.

HAIC는 다음날 종가(숫자 하나)를 맞히는 회귀라서 마지막 층이 Dense(1)이고 손실이 mse였습니다.
여기서는 이상거래일 확률을 내야 하므로 마지막 층에 sigmoid를 붙여 0~1 사이 값을 내고,
손실은 정답이 0 또는 1인 문제에 쓰는 binary crossentropy로 바꿉니다.
LSTM 층 구성(32 -> 32 -> 16)은 HAIC 스켈레톤을 그대로 유지합니다.
"""
from tensorflow import keras

from data.features import N_FEATURES, SEQ_LEN

LEARNING_RATE = 1e-3


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    compile_model(model, LEARNING_RATE)
    return model


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model
