"""운영 감시 상태: 판정 창, PSI 알림, 재학습 요청과 게이트 결과."""
from fastapi import APIRouter, Query

from backend.serving_app.monitoring import drift_monitor
from backend.serving_app.schemas import MonitoringStatus

router = APIRouter(prefix="/monitoring", tags=["운영 감시"])


@router.get("/status", response_model=MonitoringStatus)
def status(limit: int = Query(20, ge=1, le=500)):
    return drift_monitor.summary(limit)
