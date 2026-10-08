"""CSV 내부에서 시퀀스를 만들고 청크 단위로 추론합니다. 모델 버전은 분석 동안 고정합니다."""
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from data.features import SEQ_LEN, encode, sort_transactions
from data.storage import get_upload, load_upload, timestamp, write_json
from serving_app.config import result_dir
from serving_app.monitoring.metrics import evaluate

INFERENCE_CHUNK = 4096


def prepare_batch(request):
    metadata = get_upload(request.upload_id)
    df = load_upload(request.upload_id)
    df["_row_number"] = np.arange(len(df)) + 2
    df = sort_transactions(df)
    start = request.start_date.strftime("%Y%m%d") if request.start_date else df["승인일자"].min()
    end = request.end_date.strftime("%Y%m%d") if request.end_date else df["승인일자"].max()
    in_period = df["승인일자"].between(start, end).to_numpy()
    eligible = df.groupby("카드KEY").cumcount().to_numpy() >= SEQ_LEN - 1
    ends = np.flatnonzero(in_period & eligible)
    if not len(ends):
        raise ValueError("해당 기간에 판정 가능한 거래가 없습니다. 같은 카드의 거래가 20건 이상 필요합니다.")
    return metadata, df, ends, in_period, start, end


def analyze_batch(request, prepared, model, started=None):
    started = time.perf_counter() if started is None else started
    metadata, df, ends, in_period, start, end = prepared
    scaled = model.scaler.transform(encode(df))
    scores = np.empty(len(ends), dtype="float32")
    for offset in range(0, len(ends), INFERENCE_CHUNK):
        chunk = ends[offset:offset + INFERENCE_CHUNK]
        windows = chunk[:, None] - (SEQ_LEN - 1) + np.arange(SEQ_LEN)
        scores[offset:offset + len(chunk)] = model.predict_scores(scaled[windows])
    predicted = scores >= model.tau
    # 결과는 카드별 시퀀스 생성 순서 대신 판정 시점 순서로 조회합니다.
    targets = df.iloc[ends].copy()
    targets["fraud_score"] = scores
    targets["is_fraud"] = predicted
    targets["_seq"] = targets["승인SEQ"].astype("int64")
    targets["_hour"] = targets["승인시간대"].astype(int)
    targets = targets.sort_values(["승인일자", "_hour", "_seq", "카드KEY"])
    labels = targets["이상거래여부"] if "이상거래여부" in targets else pd.Series("", index=targets.index)
    known = labels.isin({"0", "1"})
    metrics = None
    if known.any():
        y = labels[known].astype(int).to_numpy()
        labelled_scores = targets.loc[known, "fraud_score"].to_numpy()
        metrics = evaluate(y, labelled_scores, model.tau)
        if not y.sum():
            metrics.update(recall=None, f2=None, pr_auc=None)
        if not (labelled_scores >= model.tau).sum():
            metrics["precision"] = None
        metrics.update(samples=len(y), frauds=int(y.sum()))
    result = pd.DataFrame({
        "row_number": targets["_row_number"], "card_key": targets["카드KEY"],
        "approval_date": targets["승인일자"], "hour": targets["_hour"],
        "approval_seq": targets["승인SEQ"], "fraud_score": targets["fraud_score"],
        "is_fraud": targets["is_fraud"], "actual": labels.where(known, ""),
    })
    directory = result_dir()
    directory.mkdir(parents=True, exist_ok=True)
    analysis_id = uuid4().hex
    output = directory / f"{analysis_id}.csv"
    result.to_csv(output, index=False, encoding="utf-8")
    duration = max(time.perf_counter() - started, 1e-6)
    summary = {
        "analysis_id": analysis_id, "upload_id": request.upload_id,
        "filename": metadata["filename"], "created_at": timestamp(),
        "model_version": model.version,
        "model_role": "local_candidate" if model.version == "v1-local" else "champion",
        "tau": model.tau, "input_rows": len(df), "period_rows": int(in_period.sum()),
        "predictions": len(ends), "excluded_rows": int(in_period.sum()) - len(ends),
        "alerts": int(predicted.sum()), "alert_rate": float(predicted.mean()),
        "labelled_predictions": int(known.sum()), "metrics": metrics,
        "start_date": start, "end_date": end,
        "duration_seconds": round(duration, 6),
        "transactions_per_second": round(len(ends) / duration, 2),
    }
    write_json(output.with_suffix(".json"), summary)
    return summary
