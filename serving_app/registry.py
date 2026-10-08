"""실습 폴더 안의 SQLite와 파일 저장소로 MLflow 위치를 고정합니다."""
import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from serving_app.config import PROJECT_ROOT

MODEL_NAME = "CardFraudLSTM"
ACTIVE_ALIAS = "champion"
EXPERIMENT_NAME = "card-fraud-aiops"


def configure_registry() -> MlflowClient:
    root = Path(os.getenv("FRAUD_MLFLOW_DIR", str(PROJECT_ROOT))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{root / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient()


def experiment_id(client: MlflowClient) -> str:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    root = Path(os.getenv("FRAUD_MLFLOW_DIR", str(PROJECT_ROOT))).resolve()
    return client.create_experiment(EXPERIMENT_NAME, artifact_location=(root / "mlruns").as_uri())


def active_version(client: MlflowClient):
    try:
        registered = client.get_registered_model(MODEL_NAME)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None
    if ACTIVE_ALIAS not in registered.aliases:
        return None
    return client.get_model_version(MODEL_NAME, registered.aliases[ACTIVE_ALIAS])
