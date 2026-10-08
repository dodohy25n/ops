"""외부 입력은 기간별 CSV. 내부 모델 입력은 카드별 거래 20건입니다."""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ErrorResponse(BaseModel):
    detail: str | dict | list


class BaselineMetrics(BaseModel):
    period: str | None
    f2: float
    precision: float
    recall: float
    pr_auc: float | None
    alert_rate: float
    fraud_rate: float | None


class GateCriteria(BaseModel):
    recall_min: float
    alert_rate_max: float
    min_samples: int
    min_frauds: int


class TriggerCriteria(BaseModel):
    window_size: int
    min_alerts: int
    min_frauds: int
    consecutive: int
    precision_floor: float
    recall_floor: float


class PsiCriteria(BaseModel):
    warn: float
    alert: float


class ModelCriteria(BaseModel):
    """모델 버전과 함께 저장된 판정 기준. 배포 게이트는 후보 심사용, 트리거는 운영 감시용입니다."""
    beta: float
    tau: float
    baseline: BaselineMetrics
    gate: GateCriteria
    trigger: TriggerCriteria
    psi: PsiCriteria


class ModelState(BaseModel):
    state: Literal["loaded", "unloaded", "unavailable"]
    source: Literal["local", "mlflow"]
    version: str | None
    tau: float | None
    role: Literal["local_candidate", "champion"] | None
    error: str | None
    missing_files: list[str] = Field(default_factory=list)
    criteria: ModelCriteria | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    loading_mode: Literal["lazy", "eager"]
    model: ModelState


class RequestMetrics(BaseModel):
    request_count: int
    error_count: int
    error_rate: float
    avg_latency_ms: float
    scope: str


class ModelUnavailableDetail(BaseModel):
    code: Literal["model_unavailable"]
    message: str
    model: ModelState


class ModelUnavailableResponse(BaseModel):
    detail: ModelUnavailableDetail


class UploadSummary(BaseModel):
    upload_id: str
    filename: str
    created_at: str
    rows: int
    cards: int
    predictable_rows: int
    excluded_rows: int
    labelled_rows: int
    start_date: str
    end_date: str


class DataStatus(BaseModel):
    exists: bool
    latest: UploadSummary | None


class QualityMetrics(BaseModel):
    tau: float
    f2: float | None
    precision: float | None
    recall: float | None
    pr_auc: float | None
    alert_rate: float
    samples: int
    frauds: int


class BatchRequest(BaseModel):
    upload_id: str = Field(..., pattern=r"^[0-9a-f]{32}$", description="/data/upload가 반환한 ID")
    start_date: date | None = Field(None, description="판정 시작일. 앞선 CSV 행은 이력으로 사용합니다.")
    end_date: date | None = None

    @model_validator(mode="after")
    def check_period(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
        return self


class BatchSummary(BaseModel):
    analysis_id: str
    upload_id: str
    filename: str
    created_at: str
    model_version: str
    model_role: str
    tau: float
    input_rows: int
    period_rows: int
    predictions: int
    excluded_rows: int
    alerts: int
    alert_rate: float
    labelled_predictions: int
    metrics: QualityMetrics | None
    start_date: str
    end_date: str
    duration_seconds: float
    transactions_per_second: float


class TransactionPrediction(BaseModel):
    row_number: int
    card_key: str
    approval_date: str
    hour: int
    approval_seq: str
    fraud_score: float
    is_fraud: bool
    actual: int | None


class PredictionPage(BaseModel):
    analysis_id: str
    total: int
    offset: int
    limit: int
    items: list[TransactionPrediction]


class LogFile(BaseModel):
    name: str
    size: int


class LogContent(BaseModel):
    name: str
    content: str
    truncated: bool
