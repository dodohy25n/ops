"""HTTP 요청 로그와 프로세스별 지표 수집. 모델 품질 지표와 별도로 관리합니다."""
import json
import logging
import time
from threading import Lock


class RequestMonitor:
    def __init__(self):
        self.lock = Lock()
        self.count = 0
        self.errors = 0
        self.duration_ms = 0.0
        self.logger = logging.getLogger(f"csv-api.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = None

    def start(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.handler = logging.FileHandler(directory / "requests.jsonl", encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self.handler)

    def stop(self):
        if self.handler:
            self.logger.removeHandler(self.handler)
            self.handler.close()
            self.handler = None

    async def middleware(self, request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            duration = (time.perf_counter() - started) * 1000
            with self.lock:
                self.count += 1
                self.errors += int(status >= 500)
                self.duration_ms += duration
            self.logger.info(json.dumps({"ts": time.time(), "method": request.method,
                                         "path": request.url.path, "status": status,
                                         "duration_ms": round(duration, 3)}, ensure_ascii=False))

    def summary(self):
        with self.lock:
            return {"request_count": self.count, "error_count": self.errors,
                    "error_rate": self.errors / self.count if self.count else 0.0,
                    "avg_latency_ms": self.duration_ms / self.count if self.count else 0.0,
                    "scope": "current server process; HTTP requests"}
