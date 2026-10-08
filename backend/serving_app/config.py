"""서버 파일 위치. 학습 라이브러리 없이도 읽을 수 있습니다."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
MODEL_DIR = PROJECT_ROOT / "serving_app/models"
DATA_DIR = REPO_ROOT / "data"


def model_source():
    """운영 모델은 MLflow의 champion입니다. local은 등록 전 v1 파일을 직접 읽을 때만 씁니다."""
    source = os.getenv("MODEL_SOURCE", "mlflow")
    if source not in {"local", "mlflow"}:
        raise ValueError("MODEL_SOURCE는 local 또는 mlflow여야 합니다.")
    return source


def frontend_dir():
    return Path(os.getenv("FRAUD_FRONTEND_DIR", str(REPO_ROOT / "frontend"))).resolve()


def runtime_dir():
    return Path(os.getenv("FRAUD_API_DIR", str(PROJECT_ROOT))).resolve()


def upload_dir():
    return runtime_dir() / "data/uploads"


def result_dir():
    return runtime_dir() / "data/results"


def monitor_dir():
    return runtime_dir() / "monitoring"


def log_dir():
    return runtime_dir() / "logs"


def max_upload_bytes():
    return int(os.getenv("FRAUD_MAX_UPLOAD_MB", "128")) * 1024 * 1024
