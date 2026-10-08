"""서버가 생성한 로그 파일 조회."""
from pathlib import Path

from fastapi import APIRouter, HTTPException

from serving_app.config import log_dir
from serving_app.schemas import ErrorResponse, LogContent, LogFile

router = APIRouter(prefix="/logs", tags=["로그"])


@router.get("", response_model=list[LogFile])
def list_logs():
    if not log_dir().exists():
        return []
    return [{"name": p.name, "size": p.stat().st_size}
            for p in sorted(log_dir().iterdir()) if p.is_file() and p.suffix in {".log", ".jsonl"}]


@router.get("/{filename}", response_model=LogContent,
            responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
def read_log(filename: str):
    if filename != Path(filename).name or Path(filename).suffix not in {".log", ".jsonl"}:
        raise HTTPException(400, "잘못된 로그 파일명입니다.")
    directory = log_dir().resolve()
    path = (directory / filename).resolve()
    if path.parent != directory or not path.is_file():
        raise HTTPException(404, "로그 파일을 찾을 수 없습니다.")
    # 전체 로그 대신 마지막 100KB를 반환합니다.
    with path.open("rb") as stream:
        size = path.stat().st_size
        stream.seek(max(0, size - 100_000))
        raw = stream.read()
    return {"name": filename, "content": raw.decode("utf-8", errors="replace"), "truncated": size > 100_000}
