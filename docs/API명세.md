# 카드 거래 CSV 배치 API 명세

버전: 0.1.0. 실제 코드의 OpenAPI 스키마는 [openapi.json](openapi.json), 서버 구성·Docker 실행은 [FastAPI CSV 서버](FastAPI-CSV서버.md)에 있습니다. 아래 분석 성공 예시는 응답 형식을 설명하며 실제 v1의 분석 성능을 뜻하지 않습니다.

## 기본 주소와 호출 순서

| 환경 | 기본 주소 |
|---|---|
| Docker Compose | `http://localhost:8099` |
| 로컬 uvicorn | `http://localhost:8077` |
| 실행 중인 Swagger | 기본 주소 + `/docs` |
| 실행 중인 OpenAPI | 기본 주소 + `/openapi.json` |

```text
GET /health
  → POST /data/upload (기간별 CSV)
  → POST /predict/batch (upload_id, 선택 판정 기간)
  → GET /predict/results/{analysis_id}
  → GET /predict/results/{analysis_id}/transactions
  → GET /predict/results/{analysis_id}/download
```

기본 CORS는 `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`, `http://127.0.0.1:3000`을 허용합니다. 추가 주소는 `FRAUD_CORS_ORIGINS`로 설정합니다. JSON은 `Content-Type: application/json`, 업로드는 multipart를 사용합니다. 날짜 범위 입력은 ISO 날짜(YYYY-MM-DD), CSV·결과의 거래 날짜는 YYYYMMDD입니다.

## 1. GET /health

서버 프로세스 상태와 모델 준비 상태를 별도로 반환합니다. 모델이 없어도 HTTP 200이며 `model_loaded=false`입니다. 조회 자체로 모델 로딩을 시작하지 않습니다.

```json
{
  "status": "ok",
  "model_loaded": false,
  "loading_mode": "lazy",
  "model": {
    "state": "unavailable",
    "source": "local",
    "version": null,
    "tau": null,
    "role": null,
    "error": null,
    "missing_files": ["fraud_v1.keras", "scaler.pkl"]
  }
}
```

| 필드 | 타입·값 | 의미 |
|---|---|---|
| `model.state` | `loaded` / `unloaded` / `unavailable` | 메모리 로딩 완료 / 아직 로딩 전 / 파일 부족 또는 로딩 실패 |
| `model.source` | `local` / `mlflow` | 모델 선택 소스 |
| `model.version` | string 또는 null | 실제로 로드한 모델 버전 |
| `model.tau` | number 또는 null | 이상거래 판정 기준 |
| `model.role` | `local` / `champion` / null | 로컬 파일의 v1 기준 모델 또는 MLflow 운영 모델 |
| `model.error` | string 또는 null | 마지막 로딩 오류 |
| `model.missing_files` | string[] | 로컬 모드에서 확인한 누락 파일 |
| `model.criteria` | object 또는 null | 모델 버전과 함께 저장된 판정 기준. 로컬 모드는 로딩 전에도 `thresholds.json`에서 읽고, MLflow 모드는 로딩 후 bundle의 `settings.json`에서 읽습니다 |

`model.criteria`는 다음 값을 담습니다. 대시보드는 이 값으로 운영 결과를 비교하므로, 모델이 교체되면 기준도 함께 바뀝니다.

| 필드 | 의미 |
|---|---|
| `beta`, `tau` | τ 선택에 쓴 F-beta의 β와 선택된 τ |
| `baseline` | τ를 고른 검증 구간의 F2·Precision·Recall·PR-AUC·경보 비율·이상거래 비율과 기간 |
| `gate` | 후보 모델 심사용 배포 게이트: `recall_min`, `alert_rate_max`, `min_samples`, `min_frauds` |
| `trigger` | 운영 감시용 재학습 트리거: `window_size`, `min_alerts`, `min_frauds`, `consecutive`, `precision_floor`·`recall_floor`(정상 창 평균 − 2σ) |
| `psi` | PSI 주의·경고 기준 `warn`, `alert` |

`unloaded` 상태는 Registry 모델이 존재한다는 보장이 아닙니다. 실제 로딩 성공 여부는 분석 실행 시 확인합니다. `eager` 모드에서도 로딩 실패 시 상태·업로드 기능은 사용할 수 있습니다.

## 2. POST /data/upload

multipart의 `file` 필드로 카드 거래 CSV를 업로드합니다. 필수 컬럼·날짜·코드·숫자 범위·중복 키를 검증하고 성공하면 HTTP 201을 반환합니다. UTF-8(BOM 포함)과 CP949를 지원합니다.

```bash
curl -F 'file=@data/sample_card_transactions.csv' http://localhost:8099/data/upload
```

```json
{
  "upload_id": "8bfce083ebd94843a4fb02ddf2eff106",
  "filename": "sample_card_transactions.csv",
  "created_at": "2026-10-08T00:41:43.260934+00:00",
  "rows": 50,
  "cards": 2,
  "predictable_rows": 12,
  "excluded_rows": 38,
  "labelled_rows": 0,
  "start_date": "20240801",
  "end_date": "20240825"
}
```

| 필드 | 타입 | 의미 |
|---|---|---|
| `upload_id` | string | 이 업로드를 지정하는 32자리 소문자 16진수 ID |
| `filename` | string | 원래 업로드 파일명 |
| `created_at` | string | UTC 시각, ISO 8601 |
| `rows` / `cards` | integer | 거래 건수 / 고유 카드 수 |
| `predictable_rows` | integer | 카드별 이력 19건을 확보한 판정 가능 거래 수 |
| `excluded_rows` | integer | 이력 부족 거래 수 |
| `labelled_rows` | integer | CSV 전체 중 정답 0/1이 확정된 거래 수 |
| `start_date` / `end_date` | string | CSV 거래 날짜 최솟값 / 최댓값 |

원본 추가 컬럼 중 모델에 쓰지 않는 이상거래유형·설명 등은 저장에서 제외합니다. 업로드는 모델 없이도 가능합니다.

오류: 파일 크기 초과 413, 컬럼·형식·값·중복 키 오류 422. 기본 파일 한도는 128MB이며 `FRAUD_MAX_UPLOAD_MB`로 변경합니다. 필수 컬럼과 허용 코드는 [CSV 입력 규칙](FastAPI-CSV서버.md#csv-입력)을 참고하세요.

## 3. 업로드 조회

| Method·경로 | 성공 응답 |
|---|---|
| `GET /data/uploads` | 업로드 요약 배열, 최신순 최대 50개; 없으면 `[]` |
| `GET /data/uploads/{upload_id}` | 2번과 같은 업로드 요약 객체 |
| `GET /data/status` | 아래 객체 |

```json
{"exists": false, "latest": null}
```

업로드가 있으면 `exists=true`이며 `latest`에 최신 업로드 요약을 반환합니다. 알 수 없거나 잘못된 ID 조회는 404입니다.

## 4. POST /predict/batch

저장된 CSV로 동기 배치 분석을 실행합니다. 요청이 완료되면 분석 요약을 반환하고 거래별 결과를 파일에 저장합니다. 프론트는 분석 완료까지 로딩 상태를 유지해야 합니다.

| 요청 필드 | 필수 | 타입 | 의미 |
|---|---|---|---|
| `upload_id` | 예 | string | 업로드 응답의 32자리 소문자 16진수 ID |
| `start_date` | 아니요 | ISO date 또는 null | 판정 시작일; 기본 CSV 첫 날짜 |
| `end_date` | 아니요 | ISO date 또는 null | 판정 종료일; 기본 CSV 마지막 날짜 |

```json
{
  "upload_id": "8bfce083ebd94843a4fb02ddf2eff106",
  "start_date": "2024-08-20",
  "end_date": "2024-08-25"
}
```

기간 앞의 CSV 거래는 과거 이력으로 사용합니다. 카드별 최근 20건의 마지막 거래가 판정 대상이며, 미래 거래와 다른 카드의 거래를 시퀀스에 섞지 않습니다. 다른 CSV에 저장된 이력을 자동으로 합치지는 않습니다.

성공 HTTP 200 응답 예시:

```json
{
  "analysis_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "upload_id": "8bfce083ebd94843a4fb02ddf2eff106",
  "filename": "sample_card_transactions.csv",
  "created_at": "2026-10-08T01:00:00+00:00",
  "model_version": "v1-local",
  "model_role": "local",
  "tau": 0.28,
  "input_rows": 50,
  "period_rows": 12,
  "predictions": 12,
  "excluded_rows": 0,
  "alerts": 2,
  "alert_rate": 0.1666666667,
  "labelled_predictions": 0,
  "metrics": null,
  "start_date": "20240820",
  "end_date": "20240825",
  "duration_seconds": 0.12,
  "transactions_per_second": 100.0
}
```

| 응답 필드 | 타입 | 의미 |
|---|---|---|
| `analysis_id` | string | 이 분석을 지정하는 32자리 ID |
| `upload_id` / `filename` | string | 분석에 사용한 CSV |
| `model_version` / `model_role` | string | 분석 전체에 사용한 모델 버전과 후보/운영 역할 |
| `tau` | number | 판정 기준. `fraud_score >= tau`일 때 이상거래 |
| `input_rows` | integer | 이력을 포함한 CSV 전체 거래 수 |
| `period_rows` | integer | 지정한 판정 기간의 전체 거래 수 |
| `predictions` | integer | 실제 판정한 거래 수 |
| `excluded_rows` | integer | 판정 기간 중 이력 부족으로 제외한 수 |
| `alerts` / `alert_rate` | integer / number | 판정 거래 중 경보 건수 / 비율(0~1) |
| `labelled_predictions` | integer | 판정 거래 중 정답이 확정된 수 |
| `metrics` | object 또는 null | 정답이 있는 판정만으로 계산한 성능 지표 |
| `start_date` / `end_date` | string | 적용한 판정 범위 |
| `duration_seconds` | number | 데이터 준비·모델 로딩·판정·결과 CSV 저장에 걸린 초 |
| `transactions_per_second` | number | 판정 건수 ÷ 처리시간 |

정답이 있는 경우 `metrics` 구조:

```json
{
  "tau": 0.28,
  "precision": 0.8,
  "recall": 1.0,
  "f2": 0.9523809524,
  "pr_auc": 0.9,
  "alert_rate": 0.05,
  "samples": 1000,
  "frauds": 40
}
```

`metrics.alert_rate`는 정답이 있는 판정의 경보 비율이고, 바깥 `alert_rate`는 모든 판정의 경보 비율입니다. 정답이 없으면 `metrics=null`; 실제 사기가 없으면 Recall·F2·PR-AUC가 null; 경보가 없으면 Precision이 null입니다. PR-AUC는 average precision 방식입니다. 이 지표만으로 자동 재학습·승격을 실행하지 않습니다.

오류: 존재하지 않는 업로드 404, ID·기간 형식이나 기간 내 판정 가능한 거래 부족 422, 모델 미준비 503.

## 5. 분석 요약 조회

| Method·경로 | 성공 응답 |
|---|---|
| `GET /predict/results` | 4번의 분석 요약 배열, 최신순 최대 50개; 없으면 `[]` |
| `GET /predict/results/{analysis_id}` | 4번의 분석 요약 객체 |

알 수 없거나 잘못된 분석 ID는 404입니다. 로컬 후보로 분석한 결과는 운영 승격 결과가 아닙니다.

## 6. GET /predict/results/{analysis_id}/transactions

| 쿼리 | 기본값 | 범위 |
|---|---|---|
| `offset` | 0 | 0 이상 |
| `limit` | 100 | 1~1000 |

```json
{
  "analysis_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "total": 12,
  "offset": 0,
  "limit": 100,
  "items": [
    {
      "row_number": 21,
      "card_key": "sample-card-01",
      "approval_date": "20240820",
      "hour": 12,
      "approval_seq": "20",
      "fraud_score": 0.87,
      "is_fraud": true,
      "actual": null
    }
  ]
}
```

거래 목록은 판정 시각 순(날짜·시간대·승인SEQ·카드)입니다. `row_number`는 검증·저장된 CSV의 헤더를 1행으로 본 행 번호입니다. 빈 행을 제거하므로 원본 파일의 물리적 줄 번호와 다를 수 있습니다. `fraud_score`는 0~1, `is_fraud`는 boolean, `actual`은 확정 정답 0/1 또는 null입니다. 전체 결과 크기보다 큰 offset이면 `items=[]`입니다.

오류: 없는 분석 404, 잘못된 쿼리 범위 422.

## 7. GET /predict/results/{analysis_id}/download

HTTP 200, `Content-Type: text/csv`로 전체 판정 CSV를 내려줍니다. 파일명은 `predictions-{analysis_id}.csv`이며 `Content-Disposition` 헤더를 프론트에서 읽을 수 있습니다. CSV 컬럼은 6번 items와 같습니다. 없는 분석은 404입니다.

## 8. GET /metrics/summary

```json
{
  "request_count": 100,
  "error_count": 2,
  "error_rate": 0.02,
  "avg_latency_ms": 18.5,
  "scope": "current server process; HTTP requests"
}
```

해당 프로세스에서 집계한 HTTP 요청 지표입니다. 서버 재시작 시 초기화됩니다. 거래 건수 기준 모델 품질 지표와 별개이며, 요청 경로별 지표나 5분 창 집계는 아직 제공하지 않습니다. 5xx만 `error_count`에 포함하므로 422 입력 오류는 서버 에러율에 포함하지 않습니다.

## 9. 로그 조회

`GET /logs`는 `[{"name":"requests.jsonl","size":1024}]` 형식의 배열을 반환합니다. 로그가 없으면 `[]`입니다.

`GET /logs/requests.jsonl` 응답:

```json
{
  "name": "requests.jsonl",
  "content": "{\"ts\":1791417600,\"method\":\"GET\",\"path\":\"/health\",\"status\":200,\"duration_ms\":1.2}\n",
  "truncated": false
}
```

100KB를 넘는 로그는 마지막 100KB만 반환하고 `truncated=true`입니다. 잘못된 파일명·확장자는 400, 없는 로그는 404입니다.

## 공통 오류 처리

CSV 검증 등 업무 오류는 문자열 detail을 반환합니다.

```json
{"detail":"CSV 2행 통합승인금액: 값의 숫자 형식 또는 허용 범위를 확인하세요."}
```

JSON 바디·쿼리의 Pydantic 검증 오류는 `detail` 배열을 반환합니다. 각 항목의 `loc`, `msg`, `type`으로 문제 필드를 표시합니다.

모델 미준비는 객체 detail을 반환합니다.

```json
{
  "detail": {
    "code": "model_unavailable",
    "message": "모델 준비가 필요합니다: fraud_v1.keras, scaler.pkl",
    "model": {
      "state": "unavailable",
      "source": "local",
      "version": null,
      "tau": null,
      "role": null,
      "error": "모델 준비가 필요합니다: fraud_v1.keras, scaler.pkl",
      "missing_files": ["fraud_v1.keras", "scaler.pkl"]
    }
  }
}
```

프론트는 응답 상태를 먼저 확인하고 문자열·배열·객체 detail을 구분합니다. 예상하지 못한 500은 JSON이 아닐 수 있으므로 JSON 파싱 실패도 처리합니다. CSV 업로드 성공과 모델 분석 성공을 별도 상태로 표시합니다.
