# 저장소 최상위에서: docker build -f backend/serving_app/Dockerfile -t card-fraud-api:dev .
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRAUD_API_DIR=/runtime FRAUD_MLFLOW_DIR=/runtime/registry \
    MODEL_SOURCE=mlflow LOADING_MODE=lazy
WORKDIR /app
COPY backend/requirements-api.txt backend/requirements.txt backend/
RUN pip install --no-cache-dir -r backend/requirements-api.txt
COPY backend/ backend/
COPY data/ data/
COPY frontend/ frontend/
RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /runtime/data/uploads /runtime/data/results /runtime/logs /runtime/registry \
    && chown -R appuser:appuser /runtime
USER appuser
EXPOSE 8077
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8077/health', timeout=2)"
CMD ["uvicorn", "backend.serving_app.main:app", "--host", "0.0.0.0", "--port", "8077"]

# 실제 모델로 판정하고 MLflow에 등록할 때 선택하는 런타임입니다. 기본 API 이미지에는 포함하지 않습니다.
FROM base AS model-runtime
USER root
RUN pip install --no-cache-dir -r backend/requirements.txt
USER appuser

# 기본 빌드는 CSV 서버입니다. 모델 학습·등록은 이미지 빌드 중 실행하지 않습니다.
FROM base AS api
