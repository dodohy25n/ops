"""서버 파일 위치. 학습 라이브러리 없이도 읽을 수 있습니다."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def frontend_dir():
    return Path(os.getenv("FRAUD_FRONTEND_DIR", str(PROJECT_ROOT.parent / "frontend"))).resolve()


def runtime_dir():
    return Path(os.getenv("FRAUD_API_DIR", str(PROJECT_ROOT))).resolve()


def upload_dir():
    return runtime_dir() / "data/uploads"


def result_dir():
    return runtime_dir() / "data/results"


def log_dir():
    return runtime_dir() / "logs"


def max_upload_bytes():
    return int(os.getenv("FRAUD_MAX_UPLOAD_MB", "128")) * 1024 * 1024
