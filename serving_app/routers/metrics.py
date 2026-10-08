"""강의 스켈레톤의 /metrics/summary: HTTP 요청 지표 조회."""
from fastapi import APIRouter, Request

from serving_app.schemas import RequestMetrics

router = APIRouter(prefix="/metrics", tags=["서버 상태"])


@router.get("/summary", response_model=RequestMetrics)
def summary(request: Request):
    return request.app.state.monitor.summary()
