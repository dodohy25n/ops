# 카드 거래 CSV 백엔드 서버

기간별 거래 CSV를 업로드하고, 카드별 최근 20건으로 배치 판정하는 FastAPI 서버입니다. 서버 루트(`/`)는 저장소의 `frontend/` 대시보드를 그대로 서비스하므로, 서버 하나만 띄우면 화면과 API를 함께 사용할 수 있습니다.

프론트 연동용 상세 요청·응답·오류 예시는 [API 명세](API명세.md), 코드에서 내보낸 기계용 스키마는 [OpenAPI JSON](openapi.json)에 있습니다.

## 빠른 시작: Docker

> 2026-10-08 폴더 구조를 `backend/`·`frontend/`·`data/`로 나눈 뒤 Dockerfile·compose의 경로는 아직 갱신하지 않았습니다. 아래 명령은 구조 변경 전 기준이며 다시 검증해야 합니다.

프로젝트 루트에서 실행합니다.

```bash
docker compose -f serving_app/docker-compose.yml up -d --build
```

- 백엔드와 대시보드: http://localhost:8099/
- Swagger API 문서: http://localhost:8099/docs
- 서버·모델 상태: http://localhost:8099/health
- 종료: `docker compose -f serving_app/docker-compose.yml down`

기본 `api` 이미지는 CSV 검증·저장·상태 조회와 프론트 연동을 실행합니다. 모델이나 TensorFlow 없이도 기동됩니다. 현재 저장소에는 실제 모델·스케일러가 없으므로 분석 요청은 HTTP 503을 반환합니다. 이 상태를 성공 판정이나 운영 승격으로 표시하지 않습니다.

업로드, 결과, 로그는 `card-fraud-csv_runtime` Docker 볼륨에 보관합니다. 일반 재시작과 `down` 후 재기동에서도 유지됩니다. 모델 폴더는 호스트의 `serving_app/models/`를 읽기 전용으로 연결합니다. PDF·원본 데이터·모델 파일·Git 정보는 이미지 빌드 컨텍스트에서 제외합니다.

기본 이미지 빌드만 필요한 경우:

```bash
docker build -f serving_app/Dockerfile -t card-fraud-api:dev .
```

## 로컬 실행

프로젝트 최상위 폴더에서 실행합니다. 실제 모델로 판정하려면 TensorFlow가 포함된 `backend/requirements.txt`를 설치합니다. CSV 검증·저장만 확인할 때는 `backend/requirements-api.txt`로 충분합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.serving_app.main:app --host 127.0.0.1 --port 8077
```

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
| POST | `/predict/batch` | 업로드 CSV에서 시퀀스 생성·배치 추론·결과 저장 |
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

분석 요약에는 `analysis_id`, `model_version`, `model_role`, `tau`, 입력·기간 내·판정·제외·경보 건수, 경보 비율, 정답이 있는 판정 건수, 성능 지표, 처리시간이 있습니다. 거래 결과에는 CSV 행 번호, 카드 식별자, 승인일자·시간대·SEQ, `fraud_score`, `is_fraud`, 선택 정답 `actual`을 기록합니다.

실패 응답은 422(컬럼·값·기간·이력 부족), 413(파일 크기), 404(알 수 없는 ID), 503(모델 미준비)입니다. 503은 `detail.code = "model_unavailable"`로 구분합니다. 업로드는 모델 없이도 저장되므로 모델 준비 후 같은 ID로 다시 분석할 수 있습니다.

## CSV 입력

UTF-8(BOM 포함)과 CP949를 지원하며 기본 크기 제한은 128MB입니다. 필수 컬럼은 다음과 같습니다.

```text
카드KEY, 승인일자, 승인시간대, 승인SEQ, 통합승인금액, 국내해외여부,
카드이용한도금액, 연령, 가맹점누적매출금액_구간화, 개인법인구분코드_가맹점,
가맹점여부_신규, 인터넷판매여부, 일시불할부구분코드,
전월_매출건수, 전월_매출금액, 카드구분코드
```

날짜는 YYYYMMDD, 시간대는 정수 0~23, 승인SEQ는 음수가 아닌 정수, 승인금액·카드한도는 양수입니다. 국내/해외는 0/1, 카드 종류는 1~5, 개인/법인은 1/2, 신규·인터넷은 0/1, 일시불/할부는 A/B입니다. 선택적 가맹점 코드·통계·연령의 빈 값과 `_`는 기존 전처리 규칙에 따라 0으로 처리합니다. 전월 매출금액의 음수는 취소 초과를 표현할 수 있어 허용하고 인코딩 때 0으로 자릅니다.

같은 카드·날짜·시간대·승인SEQ의 중복 거래, 잘못된 날짜, NaN·무한대·잘못된 코드는 거부합니다. `이상거래여부`는 선택 컬럼이며 0/1만 성능 평가에 사용하고 빈 값은 미확정으로 취급합니다. 이상거래유형·설명과 기타 추가 컬럼은 모델 입력과 업로드 저장에서 제외합니다.

각 카드의 첫 19건은 이력이 부족해 판정에서 제외합니다. 기간 시작부터 모두 판정하려면 앞선 19건을 CSV에 포함하고 판정 시작일을 지정합니다. 정답이 있는 판정만 Precision·Recall·F2·average precision에 사용합니다. 실제 사기가 없으면 Recall 등을 `null`로, 경보가 없으면 Precision을 `null`로 표시합니다.

업로드 확인용 가상 CSV는 `data/sample_card_transactions.csv`입니다. 카드 2장·거래 50건으로 판정 가능 12건, 이력 부족 38건입니다. 실제 탐지 성능 평가를 위한 데이터는 아닙니다.

## 모델 연결

현재 `MODEL_SOURCE=local`은 `serving_app/models/fraud_v1.keras`, `scaler.pkl`, `thresholds.json`을 읽고 결과에 `model_role=local_candidate`를 기록합니다. `MODEL_SOURCE=mlflow`는 게이트를 통과한 `champion`을 읽습니다. 로컬 후보 분석과 운영 승격은 별개입니다.

실제 추론 런타임이 필요한 경우 제공된 `model-runtime` Docker target을 선택하고 모델·스케일러를 준비합니다.

```bash
FRAUD_DOCKER_TARGET=model-runtime docker compose -f serving_app/docker-compose.yml up -d --build
```

이 target에는 TensorFlow·MLflow 설치 설정이 있습니다. 이번에는 기본 `api` target만 빌드·검증했고 실제 모델이 없어 `model-runtime`의 LSTM 판정은 검증하지 않았습니다. MLflow 저장소는 컨테이너의 `/runtime/registry` 경로를 사용합니다. 과거 환경의 절대 경로를 참조하는 DB를 복사하는 것만으로 아티팩트 접근이 보장되지는 않습니다.

모델·스케일러·τ는 기존 로더를 재사용합니다. 서버 요청에서 스케일러를 새로 fit하지 않습니다. 분석 중에는 한 모델 객체를 사용하며, 시퀀스는 4,096개씩 만들어 추론하므로 전체 3차원 입력을 한꺼번에 만들지 않습니다. CSV 파싱·정렬과 피처 변환은 메모리에서 처리합니다.

## 강의 스켈레톤과 파일 구조

추가 강의 PDF p.107·128의 구조를 따르되 CSV 처리 모듈을 덧붙였습니다.

```text
serving_app/
├── main.py                  앱, lifespan, CORS, 라우터 조립
├── schemas.py               업로드·분석·결과 응답 스키마
├── model_loader.py          모델·스케일러·τ, Lazy/Eager 로딩
├── config.py                파일 경로와 업로드 제한
├── batch_service.py         CSV 내부 시퀀스 생성·배치 판정·결과 저장
├── routers/
│   ├── data.py              CSV 업로드·데이터 상태
│   ├── predict.py           분석 실행·결과 조회·다운로드
│   ├── health.py            서버·모델 상태
│   ├── metrics.py           요청 지표 API
│   └── logs.py              로그 조회
├── monitoring/
│   ├── logger.py            구조화 요청 로그·지표 수집
│   ├── metrics.py           모델 품질 지표 계산
│   ├── deployment_gate.py   기존 배포 게이트
│   ├── drift_detector.py    기존 HAIC 코드; 카드 감시 연결 전
│   └── retrain_trigger.py   기존 HAIC 코드; 카드 재학습 연결 전
└── Dockerfile, docker-compose.yml
data/
├── csv_input.py             CSV 검증
├── features.py              기존 학습·판정 공용 전처리
└── storage.py               CSV·메타데이터 저장과 조회
```

서버는 HAIC 감시·재학습 모듈을 import하지 않습니다. 자동 재학습은 이번 서버 구축 범위 이후에 연결합니다. 운영 지표는 프로세스 누적값이고 재시작 시 초기화되며, 요청 로그는 파일로 유지됩니다.

## 환경변수

| 변수 | 기본값 | 용도 |
|---|---|---|
| `MODEL_SOURCE` | `local` | `local` 또는 `mlflow` |
| `LOADING_MODE` | `lazy` | `eager`는 기동 시 로딩 시도. 실패해도 상태·업로드 API 기동 |
| `FRAUD_API_DIR` | 프로젝트 루트; Docker는 `/runtime` | 업로드·결과·로그 저장 루트 |
| `FRAUD_MAX_UPLOAD_MB` | `128` | CSV 파일 크기 제한 |
| `FRAUD_CORS_ORIGINS` | localhost/127.0.0.1의 5173·3000 | 허용할 프론트 주소 |
| `FRAUD_API_PORT` | `8099` | Compose의 호스트 포트 |
| `FRAUD_DOCKER_TARGET` | `api` | Compose의 빌드 target |
| `FRAUD_MLFLOW_DIR` | 프로젝트 루트; Docker는 `/runtime/registry` | MLflow DB·아티팩트 저장 위치 |

## 검증

```bash
pip install httpx==0.28.1
python -m unittest discover -s tests -p test_csv_api.py -v
python -m unittest discover -s tests -p test_deployment_gate.py -v
```

CSV/API 13개와 기존 게이트 8개 테스트가 통과했습니다. API 테스트는 가상 시험 모델로 HTTP 연결, 학습 전처리 일치, 카드·기간 경계, 정답 선택 평가, 결과 저장·페이지 조회·다운로드, 오류, CORS, 캐시, Eager 실패 대응을 확인합니다. 실제 학습 모델의 성능 검증을 뜻하지 않습니다.

기본 Docker 이미지의 ARM64 빌드·healthy 상태, Swagger, CSV 업로드, 모델 미준비 503, CORS, 컨테이너 재시작 후 업로드 유지도 확인했습니다. 실제 LSTM·MLflow 전체 검증은 모델과 데이터 준비 후 별도로 수행해야 합니다.

구현 참고: [FastAPI 파일 업로드](https://fastapi.tiangolo.com/tutorial/request-files/), [앱 lifespan](https://fastapi.tiangolo.com/advanced/events/), [HTTP 테스트](https://fastapi.tiangolo.com/tutorial/testing/).
