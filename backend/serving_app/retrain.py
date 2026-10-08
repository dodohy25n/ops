"""감시가 요청한 재학습: 운영 모델에서 이어서 학습한 후보(v2)를 게이트에 올립니다.

데이터는 현재 운영 모델이 판정한 거래 중 정답이 있는 최근 4만 건(2024 하반기 한 달 분량)입니다.
기간을 넓히면 시간순 앞부분이 변화가 시작되기 전 거래로 채워져 후보가 새 패턴을 배우지 못합니다.
이를 시간순으로 fine-tune 60% → τ 선택 20% → 게이트 홀드아웃 20%로 나눕니다. 공부한 거래로
시험을 보거나 시험지로 τ를 고르면 점수가 낙관적으로 나오므로 세 구간은 겹치지 않습니다.
학습률 1e-3·3 epoch는 시뮬레이션 10월 데이터의 τ 구간 F2로 골랐고, 홀드아웃은 보지 않았습니다.
"""
import copy
import logging
import tempfile
import time
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from data.features import SEQ_LEN, encode, sort_transactions
from data.storage import load_upload
from backend.serving_app import model_loader
from backend.serving_app.config import model_source
from backend.serving_app.monitoring import drift_monitor
from backend.serving_app.monitoring.metrics import best_threshold, evaluate, psi_reference

MAX_SAMPLES = 40_000
SPLIT = (0.6, 0.2, 0.2)
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LEARNING_RATE = 1e-3
BATCH_SIZE = 512

logger = logging.getLogger("aiops")
_running = Lock()


def _key(card, date, hour, seq):
    return f"{card}|{date}|{int(hour)}|{int(seq)}"


def labelled_sequences(observations, scaler):
    """감시 기록의 정답 있는 거래를 업로드 원본에서 다시 20건 시퀀스로 만듭니다."""
    labelled = observations[observations["actual"].isin({"0", "1"})].copy()
    labelled["_key"] = [_key(*k) for k in labelled[drift_monitor.KEY_COLUMNS].to_numpy()]
    X_parts, keys = [], []
    for upload_id, wanted in labelled.groupby("upload_id"):
        df = sort_transactions(load_upload(upload_id))
        scaled = scaler.transform(encode(df))
        pos = df.groupby("카드KEY").cumcount().to_numpy()
        row_keys = np.array([_key(*k) for k in df[["카드KEY", "승인일자", "승인시간대", "승인SEQ"]].to_numpy()])
        ends = np.flatnonzero((pos >= SEQ_LEN - 1) & pd.Series(row_keys).isin(wanted["_key"]).to_numpy())
        X_parts.append(scaled[ends[:, None] - (SEQ_LEN - 1) + np.arange(SEQ_LEN)])
        keys.extend(row_keys[ends])
    index = pd.Series(np.arange(len(keys)), index=keys)
    labelled = labelled[labelled["_key"].isin(index.index)]
    labelled = labelled.sort_values(["approval_date", "hour", "approval_seq", "card_key"], key=lambda c: (
        c.astype(int) if c.name in {"hour", "approval_seq"} else c)).tail(MAX_SAMPLES)
    X = np.concatenate(X_parts)[index.loc[labelled["_key"]].to_numpy()]
    return X, labelled["actual"].astype(int).to_numpy(), labelled["approval_date"].to_numpy()


def split_by_time(n):
    """시간순 인덱스를 fine-tune·τ 선택·홀드아웃 세 구간으로 나눕니다."""
    first = int(n * SPLIT[0])
    second = first + int(n * SPLIT[1])
    return slice(0, first), slice(first, second), slice(second, n)


def fine_tune(current, X, y):
    from tensorflow import keras

    from backend.serving_app.lstm_model import compile_model

    import tensorflow as tf

    # 같은 데이터면 같은 후보가 나오도록 난수와 연산 순서를 고정합니다. 시연 결과가 실행마다 바뀌지 않게 합니다.
    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    model = keras.models.clone_model(current.keras_model)
    model.set_weights(current.keras_model.get_weights())
    compile_model(model, FINE_TUNE_LEARNING_RATE)
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=FINE_TUNE_EPOCHS, shuffle=True, verbose=0)
    return model


def run(version):
    """백그라운드에서 한 번에 하나만 실행합니다. 결과는 감시 상태와 이벤트 로그에 남습니다."""
    if not _running.acquire(blocking=False):
        return None
    started = time.time()
    try:
        drift_monitor.update_retrain(version, status="running", started_at=started)
        if model_source() != "mlflow":
            raise RuntimeError("재학습한 후보는 MLflow Registry로만 교체합니다. MODEL_SOURCE=mlflow가 필요합니다.")
        from backend.serving_app.train_and_register import register_candidate

        current = model_loader.get_model()
        if current.version != version:
            raise RuntimeError("재학습 요청 이후 운영 모델이 바뀌었습니다.")
        X, y, dates = labelled_sequences(drift_monitor.load_observations(), current.scaler)
        train, tune, hold = split_by_time(len(y))
        candidate = fine_tune(current, X[train], y[train])
        tune_scores = candidate.predict(X[tune], batch_size=4096, verbose=0).reshape(-1)
        tau = best_threshold(y[tune], tune_scores)
        settings = copy.deepcopy(current.settings)
        settings.update(
            model=f"fine-tuned from {version}", tau=tau, measured_at=time.strftime("%Y-%m-%d"),
            measured_on=f"retrain tau split ({dates[tune][0]} ~ {dates[tune][-1]})",
            valid={**evaluate(y[tune], tune_scores, tau), "fraud_rate": float(y[tune].mean()),
                   "sequences": int(len(y[tune]))},
        )
        # 거래 피처의 기준 분포는 데이터의 성질이라 그대로 두고, 사기 확률 분포는 모델마다 달라 후보 점수로 다시 잽니다.
        settings["psi"]["score"] = psi_reference(tune_scores)
        periods = {name: {"start": str(dates[part][0]), "end": str(dates[part][-1]),
                          "samples": int(len(y[part])), "frauds": int(y[part].sum())}
                   for name, part in (("fine_tune", train), ("tau", tune), ("holdout", hold))}
        evaluation = {"name": f"holdout-{periods['holdout']['start']}-{periods['holdout']['end']}",
                      **periods["holdout"], "periods": periods,
                      "usage": "evaluation only; not fine-tuning or threshold tuning"}
        # 운영 모델과 함께 Registry에서 내려받은 스케일러를 그대로 후보 번들에 넣습니다.
        # 로컬 models/scaler.pkl을 다시 참조하면 새 서버나 컨테이너에서 파일이 없어
        # 재학습이 조용히 실패할 수 있습니다.
        with tempfile.TemporaryDirectory(prefix="fraud-retrain-bundle-") as tmp:
            scaler_path = Path(tmp) / "scaler.pkl"
            current.scaler.save(scaler_path)
            result = register_candidate(candidate, settings, scaler_path, X[hold], y[hold],
                                        evaluation, run_name=f"retrain-from-v{version}")
        outcome = "promoted" if result["promoted"] else "rejected"
        summary = {"status": outcome, "finished_at": time.time(), "candidate_version": result["version"],
                   "tau": tau, "periods": periods, "failed_checks": result["gate"]["failed_checks"],
                   "candidate": result["gate"]["candidate"], "current": result["gate"]["current"]}
        drift_monitor.append_jsonl("events", {"ts": time.time(), "type": f"retrain_{outcome}",
                                              "from_version": version, **summary})
        drift_monitor.update_retrain(version, **summary)
        if result["promoted"]:
            model_loader.reload_model()
        return summary
    except Exception as exc:
        logger.exception("retrain failed")
        drift_monitor.append_jsonl("events", {"ts": time.time(), "type": "retrain_failed",
                                              "from_version": version, "error": str(exc)})
        drift_monitor.update_retrain(version, status="failed", finished_at=time.time(), error=str(exc))
        return None
    finally:
        _running.release()
