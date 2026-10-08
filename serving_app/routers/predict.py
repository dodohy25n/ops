"""기간별 CSV 배치 분석과 결과 조회. 이번 단계에서는 재학습을 실행하지 않습니다."""
import logging
import time

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from data.storage import get_result, list_records, record_path
from serving_app import model_loader
from serving_app.batch_service import analyze_batch, prepare_batch
from serving_app.config import result_dir
from serving_app.schemas import (
    BatchRequest, BatchSummary, ErrorResponse, ModelUnavailableResponse, PredictionPage,
)

router = APIRouter(prefix="/predict", tags=["CSV 배치 분석"])
logger = logging.getLogger("aiops")


@router.post("/batch", response_model=BatchSummary,
             responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
                        503: {"model": ModelUnavailableResponse}})
def batch(request: BatchRequest):
    started = time.perf_counter()
    try:
        prepared = prepare_batch(request)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        model = model_loader.get_model()
    except Exception as exc:
        logger.warning("model unavailable: %s", exc)
        raise HTTPException(503, {"code": "model_unavailable", "message": str(exc),
                                  "model": model_loader.model_state()}) from exc
    try:
        return analyze_batch(request, prepared, model, started=started)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/results", response_model=list[BatchSummary])
def results():
    return list_records(result_dir())


@router.get("/results/{analysis_id}", response_model=BatchSummary, responses={404: {"model": ErrorResponse}})
def summary(analysis_id: str):
    try:
        return get_result(analysis_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/results/{analysis_id}/transactions", response_model=PredictionPage,
            responses={404: {"model": ErrorResponse}})
def transactions(analysis_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    meta = summary(analysis_id)
    try:
        path = record_path(result_dir(), analysis_id, "csv")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    rows = pd.read_csv(path, dtype=str, keep_default_na=False,
                       skiprows=lambda i: 0 < i <= offset, nrows=limit)
    items = []
    for row in rows.to_dict("records"):
        row["is_fraud"] = row["is_fraud"] == "True"
        row["actual"] = int(row["actual"]) if row["actual"] else None
        items.append(row)
    return {"analysis_id": analysis_id, "total": meta["predictions"],
            "offset": offset, "limit": limit, "items": items}


@router.get("/results/{analysis_id}/download", response_class=FileResponse,
            responses={200: {"content": {"text/csv": {"schema": {"type": "string", "format": "binary"}}}},
                       404: {"model": ErrorResponse}})
def download(analysis_id: str):
    summary(analysis_id)
    try:
        path = record_path(result_dir(), analysis_id, "csv")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="text/csv", filename=f"predictions-{analysis_id}.csv")
