# 카드 거래 CSV 백엔드 서버

기간별 거래 CSV를 업로드하고, 카드별 최근 20건으로 배치 판정하는 FastAPI 서버입니다. 서버 루트(`/`)는 저장소의 `frontend/` 대시보드를 그대로 서비스하므로, 서버 하나만 띄우면 화면과 API를 함께 사용할 수 있습니다.

프론트 연동용 상세 요청·응답·오류 예시는 [API 명세](API명세.md), 코드에서 내보낸 기계용 스키마는 [OpenAPI JSON](openapi.json)에 있습니다.

## 빠른 시작: Docker

저장소 최상위 폴더에서 실행합니다. 빌드 컨텍스트가 저장소 최상위라서 `backend/`·`data/`·`frontend/`가 함께 이미지에 들어갑니다. 원본 데이터·모델 파일·실행 기록은 `.dockerignore`로 제외합니다.

```bash
docker compose -f backend/serving_app/docker-compose.yml up -d --build
```

- 백엔드와 대시보드: http://localhost:8099/
- Swagger API 문서: http://localhost:8099/docs
- 서버·모델 상태: http://localhost:8099/health
- 종료: `docker compose -f backend/serving_app/docker-compose.yml down`

기본 `api` 이미지는 CSV 검증·저장·상태 조회와 대시보드를 실행하며 TensorFlow·MLflow를 포함하지 않습니다. 그래서 분석 요청은 HTTP 503을 반환합니다. 실제 판정은 아래 [모델 연결](#모델-연결)의 `model-runtime` 이미지를 사용합니다.

업로드, 결과, 로그, 컨테이너의 MLflow 저장소는 `card-fraud-csv_runtime` Docker 볼륨에 보관합니다. 일반 재시작과 `down` 후 재기동에서도 유지됩니다. 모델 폴더는 호스트의 `backend/serving_app/models/`를 읽기 전용으로 연결합니다.

기본 이미지 빌드만 필요한 경우:

```bash
docker build -f backend/serving_app/Dockerfile --target api -t card-fraud-api:api .
```

## 로컬 실행

프로젝트 최상위 폴더에서 실행합니다. 실제 모델로 판정하려면 TensorFlow가 포함된 `backend/requirements.txt`를 설치합니다. CSV 검증·저장만 확인할 때는 `backend/requirements-api.txt`로 충분합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.serving_app.main:app --host 127.0.0.1 --port 8077
```

기본값은 Registry의 `champion`을 읽으므로, 새로 복제한 환경에서는 v1을 학습한 뒤 `python -m backend.serving_app.train_and_register`로 한 번 등록합니다. 등록 전에 v1 파일만 확인할 때는 `MODEL_SOURCE=local`로 실행합니다.

브라우저에서 http://localhost:8077/ 에 접속하면 대시보드가 열립니다. 업로드·결과·로그는 `backend/data/`와 `backend/logs/`에 저장되며 Git에서 제외합니다.

## 프론트 연결

서버가 서비스하는 대시보드는 같은 주소의 API를 호출합니다. 프론트를 따로 띄울 때 API 기본 주소는 Docker 실행 시 `http://localhost:8099`, 로컬 uvicorn은 `http://localhost:8077`입니다.

기본 CORS 허용 주소는 `localhost`와 `127.0.0.1`의 5173·3000 포트입니다. 다른 프론트 주소는 `FRAUD_CORS_ORIGINS`에 쉼표로 구분해 지정합니다. 쿠키 인증은 사용하지 않습니다.

```javascript
const API_BASE = "http://localhost:8099";
const form = new FormData();
form.append("file", selectedFile);
const uploadResponse = await fetch(`${API_BASE}/data/upload`, {
  method: "POST", body: form
});
const uploaded = await uploadResponse.json();
if (!uploadResponse.ok) throw new Error(uploaded.detail);

const analysisResponse = await fetch(`${API_BASE}/predict/batch`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ upload_id: uploaded.upload_id })
});
const analysis = await analysisResponse.json();
if (!analysisResponse.ok) {
  const detail = analysis.detail;
  throw new Error(typeof detail === "string" ? detail : detail.message);
}
```

파일 업로드에서는 브라우저가 multipart 경계를 지정하도록 `Content-Type`을 직접 설정하지 않습니다. 분석 응답을 받은 뒤 거래 목록을 페이지별로 조회합니다.

## API 명세

| Method | 경로 | 역할 |
|---|---|---|
| GET | `/health` | 프로세스 생존, 모델 로딩 상태, 버전·τ·역할 |
| POST | `/data/upload` | multipart `file`로 CSV 업로드·검증, 성공 201 |
| GET | `/data/uploads` | 최근 업로드 50개 메타데이터 |
| GET | `/data/uploads/{upload_id}` | 업로드 한 개의 메타데이터 |
| GET | `/data/status` | 최신 업로드 상태 |
| POST | `/predict/batch` | 업로드 CSV에서 시퀀스 생성·배치 추론·결과 저장, 감시 기록 누적, 조건 충족 시 백그라운드 재학습 요청 |
| GET | `/monitoring/status?limit=20` | 감시 상태(`state`), 최근 판정 창(`windows`), 최근 이벤트(`events`). `limit`은 1~500 |
| GET | `/predict/results` | 최근 분석 50개 요약 |
| GET | `/predict/results/{analysis_id}` | 저장된 분석 요약 |
| GET | `/predict/results/{analysis_id}/transactions` | `offset=0`, `limit=100` 기준 거래별 판정 조회; 최대 1,000건 |
| GET | `/predict/results/{analysis_id}/download` | 전체 판정 CSV 다운로드 |
| GET | `/metrics/summary` | 해당 서버 프로세스의 요청수·평균 응답시간·5xx 비율 |
| GET | `/logs`, `/logs/{filename}` | 로그 목록과 마지막 100KB 조회 |

분석 요청:

```json
{
  "upload_id": "업로드 응답의 32자리 ID",
  "start_date": "2024-08-01",
  "end_date": "2024-08-31"
}
```

날짜 필드는 선택 사항이며 생략하면 CSV 전체 기간을 판정합니다. 기간 앞의 CSV 거래도 시퀀스 이력으로 사용하되 기간 밖 거래는 결과·성능 집계에서 제외합니다. 다른 업로드 파일의 이력을 자동으로 합치지는 않습니다.

분석 요약에는 `analysis_id`, `model_version`, `model_role`, `tau`, 입력·기간 내·판정·제외·경보 건수, 경보 비율, 정답이 있는 판정 건수, 성능 지표, 처리시간, 감시 결과 `monitoring`이 있습니다. 거래 결과에는 CSV 행 번호, 카드 식별자, 승인일자·시간대·SEQ, `fraud_score`, `is_fraud`, 선택 정답 `actual`을 기록합니다.

실패 응답은 422(컬럼·값·기간·이력 부족), 413(파일 크기), 404(알 수 없는 ID), 503(모델 미준비)입니다. 503은 `detail.code = "model_unavailable"`로 구분합니다. 업로드는 모델 없이도 저장되므로 모델 준비 후 같은 ID로 다시 분석할 수 있습니다.

## 운영 감시와 재학습

분석이 끝나면 서버가 판정한 거래를 감시 기록에 이어 붙입니다. 재학습은 응답을 돌려준 뒤 백그라운드에서 실행되므로 분석 응답이 재학습 때문에 늦어지지 않습니다. 기준값과 근거는 [설계 지표](설계지표.md) 5·6절에 있습니다.

| 단계 | 동작 | 코드 |
|---|---|---|
| 기록 | 판정 거래를 날짜 → 카드 순으로 `monitoring/observations.csv`에 추가합니다. 같은 카드·날짜·시간대·승인SEQ가 이미 있으면 건너뜁니다 | `batch_service.py`, `monitoring/drift_monitor.py` |
| 창 | 누적 1,000건마다 창 하나를 닫고 Precision(경보 30건 이상일 때)·Recall(실제 사기 20건 이상일 때)·PSI를 계산합니다. 표본이 부족한 지표는 판정 보류이며 연속 횟수를 바꾸지 않습니다 | `drift_monitor.py` |
| PSI | 승인 금액·시간대·해외·가맹점 매출 구간·할부·사기 확률 중 하나라도 0.1 이상이면 `psi_warn`, 0.25 이상이면 `psi_alert` 이벤트만 남깁니다 | `drift_monitor.py` |
| 재학습 요청 | 창을 닫을 때마다 Precision 또는 Recall이 평균 − 2σ 미만으로 2개 창 연속인지 확인하고, 충족하면 그 창에서 요청합니다. 재학습이 요청·실행 중이면 새로 요청하지 않습니다 | `drift_monitor.py`, `routers/predict.py` |
| 재학습 | 현재 운영 모델이 판정한 정답 있는 최근 40,000건을 시간순 60% · 20% · 20%로 나눠 fine-tune, τ 선택, 게이트 홀드아웃에 씁니다. 한 번에 하나만 실행합니다 | `retrain.py` |
| 게이트 | `register_candidate()`가 후보와 현재 champion을 홀드아웃에서 비교합니다. 통과하면 champion을 옮기고 서버가 새 모델을 불러오며, 다음 분석에서 판정 창이 처음부터 다시 시작됩니다. 실패하면 현재 모델을 유지하고 연속 횟수를 비웁니다 | `train_and_register.py`, `retrain.py` |

감시 파일은 `FRAUD_API_DIR` 아래 `monitoring/`에 저장합니다(Docker는 `/runtime/monitoring`, 볼륨에 유지).

| 파일 | 내용 |
|---|---|
| `state.json` | 현재 운영 모델 버전, 누적 건수, 닫힌 창 수, 지표별 연속 미달 횟수, 재학습 상태 |
| `observations.csv` | 판정 거래 기록(점수, 경보, 정답, PSI 대상 피처) |
| `windows.jsonl` | 창별 Precision·Recall·판정 상태·PSI |
| `events.jsonl` | `psi_warn`, `psi_alert`, `retrain_requested`, `retrain_promoted`, `retrain_rejected`, `retrain_failed`, `window_reset` |
| `observations-v{버전}.csv`, `windows-v{버전}.jsonl` | 운영 모델이 바뀔 때 보관한 이전 모델의 기록 |

재학습한 후보는 MLflow Registry로만 교체하므로 `MODEL_SOURCE=mlflow`가 필요합니다. `local` 모드에서도 감시 기록은 쌓이지만, 재학습 요청은 `retrain_failed`로 끝납니다. 상태는 `GET /monitoring/status`, 분석 응답의 `monitoring`, 대시보드의 「운영 감시 · 재학습」 카드와 「데이터 분포 (PSI)」 카드에서 확인합니다([프론트 README](../frontend/README.md)).

## CSV 입력

UTF-8(BOM 포함)과 CP949를 지원하며 기본 크기 제한은 128MB입니다. 필수 컬럼은 다음과 같습니다.

```text
카드KEY, 승인일자, 승인시간대, 승인SEQ, 통합승인금액, 국내해외여부,
카드이용한도금액, 연령, 가맹점누적매출금액_구간화, 개인법인구분코드_가맹점,
가맹점여부_신규, 인터넷판매여부, 일시불할부구분코드,
전월_매출건수, 전월_매출금액, 카드구분코드
```

날짜는 YYYYMMDD, 시간대는 정수 0~23, 승인SEQ는 음수가 아닌 정수, 승인금액·카드한도는 0 이상입니다. 카드한도 빈칸은 0으로 처리합니다. 학습 데이터에 금액·한도 0과 한도 빈칸이 있었으므로 입구 검사도 같은 범위를 받습니다. 국내/해외는 0/1, 카드 종류는 1~5, 개인/법인은 1/2, 신규·인터넷은 0/1, 일시불/할부는 A/B입니다. 선택적 가맹점 코드·통계·연령의 빈 값과 `_`는 기존 전처리 규칙에 따라 0으로 처리합니다. 전월 매출건수·금액의 음수는 취소 초과를 표현할 수 있어 허용하고 인코딩 때 0으로 자릅니다.

같은 카드·날짜·시간대·승인SEQ의 중복 거래, 잘못된 날짜, NaN·무한대·잘못된 코드는 거부합니다. `이상거래여부`는 선택 컬럼이며 0/1만 성능 평가에 사용하고 빈 값은 미확정으로 취급합니다. 이상거래유형·설명과 기타 추가 컬럼은 모델 입력과 업로드 저장에서 제외합니다.

각 카드의 첫 19건은 이력이 부족해 판정에서 제외합니다. 기간 시작부터 모두 판정하려면 앞선 19건을 CSV에 포함하고 판정 시작일을 지정합니다. 정답이 있는 판정만 Precision·Recall·F2·average precision에 사용합니다. 실제 사기가 없으면 Recall 등을 `null`로, 경보가 없으면 Precision을 `null`로 표시합니다.

업로드 확인용 가상 CSV는 `data/sample_card_transactions.csv`입니다. 카드 2장·거래 50건으로 판정 가능 12건, 이력 부족 38건입니다. 실제 탐지 성능 평가를 위한 데이터는 아닙니다.

## 모델 연결

기본값 `MODEL_SOURCE=mlflow`는 MLflow Registry의 운영 모델 `champion`을 읽고 결과에 `model_role=champion`을 기록합니다. 분석할 때마다 `champion`이 가리키는 버전을 확인해, 게이트를 통과한 새 버전으로 옮겨졌으면 서버 재시작 없이 그 버전을 불러옵니다. 새 버전을 읽다가 실패하면 기존 모델로 계속 판정합니다. 현재 `champion`은 v1(버전 1)입니다.

`MODEL_SOURCE=local`은 Registry 없이 `backend/serving_app/models/`의 `fraud_v1.keras`, `scaler.pkl`, `thresholds.json`을 직접 읽고 `model_role=local`을 기록합니다. 등록 전 v1 확인용이며, 게이트를 통과한 모델로 교체되지 않습니다.

실제 모델로 판정하려면 `model-runtime` 이미지를 띄우고, 컨테이너의 빈 Registry에 v1을 기준 운영 모델로 한 번 등록합니다. 등록 기록은 Docker 볼륨에 남으므로 재시작해도 다시 등록할 필요가 없습니다.

```bash
FRAUD_DOCKER_TARGET=model-runtime docker compose -f backend/serving_app/docker-compose.yml up -d --build
docker compose -f backend/serving_app/docker-compose.yml exec serving-api python -m backend.serving_app.train_and_register
```

호스트의 `backend/mlflow.db`를 컨테이너로 복사하지 않는 이유는 DB가 호스트의 절대 경로로 아티팩트를 가리키기 때문입니다. 컨테이너에서 같은 v1 파일로 다시 등록하면 Registry 위치와 아티팩트 경로가 일치합니다.

모델·스케일러·τ는 기존 로더를 재사용합니다. 서버 요청에서 스케일러를 새로 fit하지 않습니다. 분석 중에는 한 모델 객체를 사용하며, 시퀀스는 4,096개씩 만들어 추론하므로 전체 3차원 입력을 한꺼번에 만들지 않습니다. CSV 파싱·정렬과 피처 변환은 메모리에서 처리합니다.

## 강의 스켈레톤과 파일 구조

추가 강의 PDF p.107·128의 구조를 따르되 CSV 처리 모듈을 덧붙였습니다.

```text
serving_app/
├── main.py                  앱, lifespan, CORS, 라우터 조립
├── schemas.py               업로드·분석·결과 응답 스키마
├── model_loader.py          모델·스케일러·τ, Lazy/Eager 로딩
├── config.py                파일 경로와 업로드 제한
├── batch_service.py         CSV 내부 시퀀스 생성·배치 판정·결과 저장·감시용 판정 기록
├── retrain.py               감시가 요청한 fine-tuning → 배포 게이트 → 교체
├── train_and_register.py    v1 기준 등록, 후보 게이트 심사·조건부 champion 교체
├── routers/
│   ├── data.py              CSV 업로드·데이터 상태
│   ├── predict.py           분석 실행·결과 조회·다운로드, 감시 기록과 재학습 요청
│   ├── monitoring.py        감시 상태 조회
│   ├── health.py            서버·모델 상태
│   ├── metrics.py           요청 지표 API
│   └── logs.py              로그 조회
├── monitoring/
│   ├── logger.py            구조화 요청 로그·지표 수집
│   ├── metrics.py           모델 품질 지표·PSI 계산
│   ├── drift_monitor.py     1,000건 판정 창, PSI 알림, 연속 미달과 재학습 요청
│   └── deployment_gate.py   후보와 운영 모델 비교
└── Dockerfile, docker-compose.yml
data/
├── csv_input.py             CSV 검증
├── features.py              기존 학습·판정 공용 전처리
└── storage.py               CSV·메타데이터 저장과 조회
```

HAIC 실습의 RMSE 기반 감시·재학습 모듈은 카드 구조와 맞지 않아 삭제하고, 카드 거래용 창 감시·재학습을 새로 만들었습니다(`drift_monitor.py`, `retrain.py`). 감시 기록은 파일로 유지됩니다. 요청 운영 지표는 프로세스 누적값이고 재시작 시 초기화되며, 요청 로그는 파일로 유지됩니다.

## 환경변수

| 변수 | 기본값 | 용도 |
|---|---|---|
| `MODEL_SOURCE` | `mlflow` | `mlflow`는 Registry의 champion, `local`은 v1 파일 직접 읽기 |
| `LOADING_MODE` | `lazy` | `eager`는 기동 시 로딩 시도. 실패해도 상태·업로드 API 기동 |
| `FRAUD_API_DIR` | 프로젝트 루트; Docker는 `/runtime` | 업로드·결과·로그·감시 기록(`monitoring/`) 저장 루트 |
| `FRAUD_MAX_UPLOAD_MB` | `128` | CSV 파일 크기 제한 |
| `FRAUD_CORS_ORIGINS` | localhost/127.0.0.1의 5173·3000 | 허용할 프론트 주소 |
| `FRAUD_API_PORT` | `8099` | Compose의 호스트 포트 |
| `FRAUD_DOCKER_TARGET` | `api` | Compose의 빌드 target. 실제 판정은 `model-runtime` |
| `FRAUD_MLFLOW_DIR` | 프로젝트 루트; Docker는 `/runtime/registry` | MLflow DB·아티팩트 저장 위치 |

## 검증

```bash
pip install httpx2==2.13.0
python -m unittest discover -s backend/tests -t . -v
```

테스트 45개(CSV/API 15개, 배포 게이트 9개, 판정 창 감시·시간 분할 8개, 분석 API 감시 연결·재시작 복구 7개, PSI 3개, 재학습 통합 2개, Registry 흐름 1개)가 통과했습니다. 감시 테스트는 가짜 모델과 임시 폴더로 창 닫기, 판정 보류, 연속 미달 시 요청, 중복 제외, 모델 교체 시 초기화, PSI 알림만 기록을 확인합니다. 재학습 통합 테스트는 합성 모델과 임시 Registry로 `retrain.run`의 통과·미달 경로를 끝까지 확인합니다. 목록은 [검증 결과](results/검증결과.md)에 있습니다. API 테스트는 가상 시험 모델로 HTTP 연결, 학습 전처리 일치, 카드·기간 경계, 정답 선택 평가, 결과 저장·페이지 조회·다운로드, 오류, CORS, 캐시, Eager 실패 대응을 확인합니다. 실제 학습 모델의 성능 검증을 뜻하지 않습니다.

2026-10-08 폴더 구조 변경 후 Docker 구성을 다시 검증했습니다(ARM64).

| 항목 | 결과 |
|---|---|
| `api` 이미지 | healthy, 대시보드 200, CSV 업로드 201, 모델 런타임 없음으로 분석 503 |
| `model-runtime` 이미지 | Registry가 비어 있을 때 503 → 컨테이너에서 v1 등록 후 `champion` 버전 1로 분석 |
| 로컬과 컨테이너 판정 일치 | 운영 구간 카드 300장 CSV, 판정 43,503건에서 점수 최대 차이 0, 판정 전건 일치 |
| 재시작 | 업로드 기록과 운영 버전 1 유지 |

로컬과 컨테이너는 `backend/requirements-api.txt`의 같은 버전(pandas 3.0.6, numpy 2.4.4, fastapi 0.141.1)을 사용합니다. 위 표는 감시·재학습 연결(커밋 `dfabda7`) 이전에 실행했습니다. 연결 이후에는 새 볼륨의 `model-runtime` 컨테이너에서 6개월 시연 전체(감시 → 재학습 → 게이트 → 교체)를 HTTP로 돌렸고, 월별 결과가 [시연 실행 로그](results/시연-실행-로그.txt)와 같았습니다([검증 결과](results/검증결과.md#docker-컨테이너-시연)).

## 6개월 시연 실행

2024년 하반기를 월별 CSV로 나눠 정상 → 명절 → 신종 사기 순서로 서버에 넣습니다. 시나리오와 결과는 [발표자 가이드](발표자-가이드.md) 6절, 실제 출력은 [시연 실행 로그](results/시연-실행-로그.txt)에 있습니다.

### Docker로 한 번에 실행 (권장)

```bash
bash backend/scripts/demo_docker.sh
```

`model-runtime` 컨테이너를 새 볼륨으로 띄우고(`down -v` → `up -d --build`), healthy를 기다린 뒤 컨테이너 안의 빈 Registry에 v1을 등록하고, 시나리오 CSV를 확인한 다음 `run_demo.py --api http://127.0.0.1:8099`를 실행합니다. 시나리오 CSV가 없으면 생성 명령(`.venv/bin/python backend/scripts/make_scenarios.py`, `data/processed/ops_rows.csv` 필요)을 안내하고 멈추며, 서버는 켜 둡니다. 컨테이너는 볼륨 안의 Registry만 쓰므로 호스트의 `backend/mlflow.db`·`backend/mlruns`는 바뀌지 않습니다. 처음 실행은 이미지 빌드 때문에 4~5분, 시연 부분은 약 37초 걸립니다. 환경변수 `FRAUD_API_PORT`(기본 8099), `PYTHON`(기본 `python3`)으로 포트와 실행할 파이썬을 바꿀 수 있습니다. 대시보드는 `http://127.0.0.1:8099/`, 정리는 `docker compose -f backend/serving_app/docker-compose.yml down`입니다.

### 로컬에서 실행

```bash
# 1) 시나리오 CSV 생성 (data/processed/ops_rows.csv 필요 → data/scenarios/2024-07.csv ~ 2024-12.csv)
python backend/scripts/make_scenarios.py

# 2) 실제 Registry를 건드리지 않도록 빈 저장소를 따로 지정하고 v1을 등록
export FRAUD_API_DIR=/tmp/fraud-demo FRAUD_MLFLOW_DIR=/tmp/fraud-demo/registry
python -m backend.serving_app.train_and_register

# 3) 같은 환경변수로 서버 실행
uvicorn backend.serving_app.main:app --host 127.0.0.1 --port 8077

# 4) 다른 터미널에서 월별 CSV를 순서대로 업로드·분석
python backend/scripts/run_demo.py --api http://127.0.0.1:8077
```

`run_demo.py`는 표준 라이브러리만 사용하며, 재학습이 요청되면 `/monitoring/status`로 게이트 결과가 나올 때까지 기다린 뒤 다음 달로 넘어갑니다. 마지막에 월별 요약 표(월, 시나리오, 모델, Recall, F2, 재학습 결과)와 대시보드 주소를 출력합니다. 경로 `/tmp/fraud-demo`는 예시이며 빈 폴더면 됩니다. 같은 폴더를 다시 쓰면 이전 감시 기록과 후보 버전이 남아 결과가 달라집니다.

서버를 시작할 때 감시 상태에 재학습이 「요청됨/실행 중」으로 남아 있으면(재학습 도중 서버가 꺼진 경우) 이를 `failed`(서버 재시작으로 중단)로 바꾸고 `retrain_failed` 이벤트를 남깁니다. 그래야 다음 하락 때 재학습을 다시 요청할 수 있습니다(`drift_monitor.recover_interrupted_retrain()`).

구현 참고: [FastAPI 파일 업로드](https://fastapi.tiangolo.com/tutorial/request-files/), [앱 lifespan](https://fastapi.tiangolo.com/advanced/events/), [HTTP 테스트](https://fastapi.tiangolo.com/tutorial/testing/).
