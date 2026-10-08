# Ops! Card Fraud AIOps

카드 거래 CSV를 업로드하고, 카드별 최근 20건을 LSTM으로 판정한 뒤
PSI·Precision·Recall을 감시하여 재학습과 배포 게이트까지 연결하는 B2B 운영 도구입니다.

## 폴더 구조

```text
ops-main 2/
├── frontend/                 # Ops! 상단 메뉴형 운영 화면
│   ├── index.html            # 화면 구조와 API 호출
│   └── theme.css             # 평면형 B2B 디자인
├── backend/
│   ├── serving_app/
│   │   ├── main.py           # FastAPI 진입점, 정적 프론트 제공
│   │   ├── routers/           # health, data, predict, monitoring, metrics, logs
│   │   ├── monitoring/        # PSI, 성능 창, 배포 게이트
│   │   ├── model_loader.py    # local/MLflow 모델 로딩
│   │   └── retrain.py         # fine-tuning과 champion 교체
│   ├── scripts/               # 데이터 준비, 학습, 시연
│   └── tests/                 # API·감시·재학습 통합 테스트
├── data/                     # CSV 검증, 피처 변환, 저장
└── docs/                     # API 명세와 설계 근거
```

## 처음 실행

프로젝트 최상위 폴더에서 실행합니다.

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
```

운영 모델이 MLflow Registry에 등록되어 있다면:

```bash
MODEL_SOURCE=mlflow LOADING_MODE=lazy uvicorn backend.serving_app.main:app --reload --port 8077
```

등록 전 로컬 모델 파일을 확인하려면:

```bash
MODEL_SOURCE=local LOADING_MODE=lazy uvicorn backend.serving_app.main:app --reload --port 8077
```

- 화면: <http://127.0.0.1:8077/>
- Swagger: <http://127.0.0.1:8077/docs>
- 상태 API: <http://127.0.0.1:8077/health>

## 모델 파일 주의

실제 분석에는 다음 산출물이 필요합니다.

- local 모드: `backend/serving_app/models/fraud_v1.keras`, `scaler.pkl`, `thresholds.json`
- MLflow 모드: Registry의 `CardFraudLSTM@champion`

현재 전달본에는 위 local 모델 세 파일이 포함되어 있고, `(20, 17)` 입력 형상·임계값 `0.28`·실제 배치 추론을 확인했습니다. `MODEL_SOURCE=local`로 실행하면 첫 분석 요청 때 로드됩니다.

모델이 없어도 서버 상태 확인, CSV 업로드, 데이터 조회, 로그와 지표 화면은 실행됩니다.
분석 요청은 잘못된 결과를 만들지 않고 HTTP 503으로 실패하도록 설계되어 있습니다.

## 검증

```bash
MLFLOW_DISABLE_AGENT_HINT=1 MPLCONFIGDIR=/tmp/ops-mpl \
python -m unittest discover -s backend/tests -t . -v
```

통합 테스트는 운영용 바이너리 파일에 의존하지 않도록 임시 스케일러와 합성 모델을 사용합니다.
