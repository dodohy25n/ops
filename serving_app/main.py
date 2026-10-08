"""카드 거래 CSV 배치 서버. 실행: uvicorn serving_app.main:app --port 8077"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from serving_app import model_loader
from serving_app.config import PROJECT_ROOT, log_dir
from serving_app.monitoring.logger import RequestMonitor
from serving_app.routers import data, health, logs, metrics, predict

DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"


def create_app():
    monitor = RequestMonitor()

    @asynccontextmanager
    async def lifespan(app):
        mode = os.getenv("LOADING_MODE", "lazy")
        if mode not in {"lazy", "eager"}:
            raise ValueError("LOADING_MODE는 lazy 또는 eager여야 합니다.")
        if os.getenv("MODEL_SOURCE", "local") not in {"local", "mlflow"}:
            raise ValueError("MODEL_SOURCE는 local 또는 mlflow여야 합니다.")
        monitor.start(log_dir())
        try:
            if mode == "eager":
                try:
                    model_loader.load_eager()
                except Exception as exc:
                    logging.getLogger("aiops").warning("eager load unavailable: %s", exc)
            yield
        finally:
            monitor.stop()

    app = FastAPI(title="카드 거래 CSV 배치 분석", version="0.1.0", lifespan=lifespan,
                  description="기간별 거래 CSV를 업로드하고 카드별 최근 20건으로 이상거래를 판정합니다.")
    app.state.monitor = monitor
    app.middleware("http")(monitor.middleware)
    origins = [s.strip() for s in os.getenv("FRAUD_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if s.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], expose_headers=["Content-Disposition"])
    for router in (health.router, data.router, predict.router, metrics.router, logs.router):
        app.include_router(router)
    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "serving_app/static", html=True), name="static")
    return app


app = create_app()
