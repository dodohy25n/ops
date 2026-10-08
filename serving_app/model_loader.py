"""모델, 스케일러, 판정 기준을 같은 버전에서 함께 불러옵니다."""
import json
import os
import tempfile
from pathlib import Path
from threading import RLock

import numpy as np

from data.features import FEATURES, N_FEATURES, SEQ_LEN, FraudScaler, sequence_from_rows
from serving_app.config import PROJECT_ROOT

_model_cache = None
_load_lock = RLock()
_last_load_error = None


class LoadedModel:
    def __init__(self, keras_model, scaler, settings, version):
        self.keras_model = keras_model
        self.scaler = scaler
        self.settings = settings
        self.tau = float(settings["tau"])
        self.version = str(version)
        if settings.get("features", FEATURES) != FEATURES:
            raise ValueError("등록된 모델과 현재 코드의 피처 순서가 다릅니다.")
        if tuple(keras_model.input_shape[1:]) != (SEQ_LEN, N_FEATURES):
            raise ValueError("모델 입력 모양이 (20, 17)과 다릅니다.")
        if not np.isfinite(self.tau) or not 0 < self.tau < 1:
            raise ValueError("등록된 판정 기준이 올바르지 않습니다.")

    def predict_scores(self, X):
        X = np.asarray(X, dtype="float32")
        if X.ndim != 3 or X.shape[1:] != (SEQ_LEN, N_FEATURES) or not len(X):
            raise ValueError("모델 입력은 (거래 수, 20, 17)이어야 합니다.")
        if not np.isfinite(X).all():
            raise ValueError("모델 입력에 NaN 또는 무한대가 있습니다.")
        scores = self.keras_model.predict(X, batch_size=4096, verbose=0).reshape(-1)
        if len(scores) != len(X) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("모델이 올바른 이상거래 점수를 반환하지 않았습니다.")
        return scores

    def predict_one(self, sequence: list[dict]) -> float:
        if len(sequence) != SEQ_LEN:
            raise ValueError(f"거래가 정확히 {SEQ_LEN}건 필요합니다.")
        X = sequence_from_rows(sequence, self.scaler)[None, ...]
        return float(self.predict_scores(X)[0])

    def classify(self, X):
        return (self.predict_scores(X) >= self.tau).astype(int)


def load_run_bundle(client, run_id, model_uri, version):
    import mlflow.tensorflow

    with tempfile.TemporaryDirectory(prefix="fraud-bundle-") as tmp:
        bundle = Path(client.download_artifacts(run_id, "bundle", tmp))
        settings = json.loads((bundle / "settings.json").read_text())
        scaler = FraudScaler.load(bundle / "scaler.pkl")
        model = mlflow.tensorflow.load_model(
            model_uri, keras_model_kwargs={"compile": False}
        )
    return LoadedModel(model, scaler, settings, version)


def load_registered_version(client, version):
    from serving_app.registry import MODEL_NAME

    return load_run_bundle(client, version.run_id, f"models:/{MODEL_NAME}/{version.version}", version.version)


def _load_from_mlflow():
    from serving_app.registry import active_version, configure_registry

    client = configure_registry()
    version = active_version(client)
    if version is None:
        raise RuntimeError("운영 모델이 없습니다. 먼저 train_and_register.py를 실행하세요.")
    return load_registered_version(client, version)


def _load_from_local():
    root = PROJECT_ROOT / "serving_app/models"
    missing = [p.name for p in (root / "fraud_v1.keras", root / "scaler.pkl", root / "thresholds.json")
               if not p.is_file()]
    if missing:
        raise FileNotFoundError("모델 준비가 필요합니다: " + ", ".join(missing))
    from tensorflow import keras

    settings = json.loads((root / "thresholds.json").read_text())
    model = keras.models.load_model(root / "fraud_v1.keras", compile=False)
    return LoadedModel(model, FraudScaler.load(root / "scaler.pkl"), settings, "v1-local")


def _load_model():
    source = os.getenv("MODEL_SOURCE", "local")
    if source not in {"local", "mlflow"}:
        raise ValueError("MODEL_SOURCE는 local 또는 mlflow여야 합니다.")
    return _load_from_mlflow() if source == "mlflow" else _load_from_local()


def reload_model():
    global _model_cache, _last_load_error
    with _load_lock:
        try:
            loaded = _load_model()
        except Exception as exc:
            _last_load_error = str(exc)
            raise
        _model_cache = loaded
        _last_load_error = None
        return loaded


def load_eager():
    return reload_model()


def get_model():
    with _load_lock:
        if _model_cache is None:
            return reload_model()
        return _model_cache


def model_state():
    """상태 조회만으로 모델을 로드하거나 Registry DB를 생성하지 않습니다."""
    model = _model_cache
    source = os.getenv("MODEL_SOURCE", "local")
    if model is not None:
        role = "local_candidate" if model.version == "v1-local" else "champion"
        return {"state": "loaded", "source": source, "version": model.version,
                "tau": model.tau, "role": role, "error": _last_load_error}
    root = PROJECT_ROOT / "serving_app/models"
    missing = ([p.name for p in (root / "fraud_v1.keras", root / "scaler.pkl", root / "thresholds.json")
                if not p.is_file()] if source == "local" else [])
    return {"state": "unavailable" if missing or _last_load_error else "unloaded",
            "source": source, "version": None, "tau": None, "role": None,
            "error": _last_load_error, "missing_files": missing}
