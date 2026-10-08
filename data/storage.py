"""검증한 카드 거래 CSV와 분석 결과를 UUID로 저장합니다."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from serving_app.config import result_dir, upload_dir


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def record_path(directory: Path, record_id: str, suffix: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", record_id):
        raise FileNotFoundError("잘못되었거나 존재하지 않는 ID입니다.")
    path = directory / f"{record_id}.{suffix}"
    if not path.is_file():
        raise FileNotFoundError("저장된 항목을 찾을 수 없습니다.")
    return path


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_upload(df: pd.DataFrame, filename: str, summary: dict) -> dict:
    directory = upload_dir()
    directory.mkdir(parents=True, exist_ok=True)
    upload_id = uuid4().hex
    path = directory / f"{upload_id}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    metadata = {"upload_id": upload_id, "filename": Path(filename).name,
                "created_at": timestamp(), **summary}
    write_json(path.with_suffix(".json"), metadata)
    return metadata


def get_upload(upload_id: str) -> dict:
    return json.loads(record_path(upload_dir(), upload_id, "json").read_text(encoding="utf-8"))


def load_upload(upload_id: str) -> pd.DataFrame:
    get_upload(upload_id)
    return pd.read_csv(record_path(upload_dir(), upload_id, "csv"), dtype=str, keep_default_na=False)


def list_records(directory: Path, limit: int = 50) -> list[dict]:
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return [json.loads(p.read_text(encoding="utf-8")) for p in files[:limit]]


def get_result(analysis_id: str) -> dict:
    return json.loads(record_path(result_dir(), analysis_id, "json").read_text(encoding="utf-8"))
