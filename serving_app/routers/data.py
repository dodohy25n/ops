"""기간별 카드 거래 CSV 업로드와 저장된 데이터 조회."""
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from data.csv_input import parse_csv, summarize_upload
from data.storage import get_upload, list_records, save_upload
from serving_app.config import max_upload_bytes, upload_dir
from serving_app.schemas import DataStatus, ErrorResponse, UploadSummary

router = APIRouter(prefix="/data", tags=["CSV 데이터"])


def store_csv(raw, filename):
    df = parse_csv(raw)
    return save_upload(df, filename, summarize_upload(df))


@router.post("/upload", response_model=UploadSummary, status_code=201,
             responses={413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def upload(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(422, "카드 거래 CSV 파일을 선택하세요.")
        raw = bytearray()
        while chunk := await file.read(1024 * 1024):
            raw.extend(chunk)
            if len(raw) > max_upload_bytes():
                raise HTTPException(413, "CSV가 업로드 크기 제한을 초과했습니다.")
        try:
            return await run_in_threadpool(store_csv, bytes(raw), file.filename)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


@router.get("/uploads", response_model=list[UploadSummary])
def uploads():
    return list_records(upload_dir())


@router.get("/uploads/{upload_id}", response_model=UploadSummary, responses={404: {"model": ErrorResponse}})
def uploaded_data(upload_id: str):
    try:
        return get_upload(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/status", response_model=DataStatus)
def status():
    records = list_records(upload_dir(), limit=1)
    return {"exists": bool(records), "latest": records[0] if records else None}
