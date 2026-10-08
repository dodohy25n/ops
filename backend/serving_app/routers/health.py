"""프로세스 생존과 모델의 준비 상태를 구분합니다."""
import os

from fastapi import APIRouter

from backend.serving_app import model_loader
from backend.serving_app.schemas import HealthResponse

router = APIRouter(tags=["서버 상태"])


@router.get("/health", response_model=HealthResponse)
def health():
    state = model_loader.model_state()
    return {"status": "ok", "model_loaded": state["state"] == "loaded",
            "loading_mode": os.getenv("LOADING_MODE", "lazy"), "model": state}
