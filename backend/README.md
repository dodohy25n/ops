# 카드 거래 이상탐지 AIOps 조별 실습

W13 모델 서빙 및 AIOps 과정에서 카드별 최근 20건의 거래로 이상거래를 판정하는 LSTM과 모델 운영 흐름을 구현합니다. 과제 마감은 2026-10-08입니다.

운영 입력은 일정 기간의 카드 거래가 담긴 CSV입니다. CSV를 업로드하면 시스템이 카드별 시간순으로 20건 시퀀스를 만들어 배치로 판정하고, PSI·성능 감시와 조건부 재학습으로 이어지는 방식입니다. 20건은 CSV에서 생성하는 모델 입력 단위입니다. 현재 구현 상태와 남은 작업은 [CSV 배치 처리 기준 정리](../docs/현황과-남은작업.md)에 있습니다.

**2026-10-08에는 카드 거래 CSV FastAPI 서버와 Docker 실행 구성을 추가했습니다.** CSV 업로드·배치 판정·결과 조회, 프론트용 CORS, 헬스체크·요청 지표·로그를 연결했습니다. 기본 Docker 서버는 모델 없이도 기동하며, 실제 분석은 모델 준비 후 가능합니다. [서버 실행 및 API 명세](../docs/FastAPI-CSV서버.md)

문서: [서버 구성·Docker 실행](../docs/FastAPI-CSV서버.md) · [API 상세 명세](../docs/API명세.md) · [OpenAPI JSON](../docs/openapi.json)

v1은 MLflow Registry 버전 1로 등록되어 운영 모델(`champion`)입니다. 배포 게이트는 재학습한 후보 v2를 내보내도 되는지 판단할 때만 사용합니다. 다음 우선순위는 감시·재학습 연결입니다.

## 현재 진행 상태

| 작업 | 상태 |
|---|---|
| 원본 분석, 시간 분할, 공용 전처리 | 완료 |
| 전체 카드 사용, 시퀀스 20건, v1 학습 | 완료 |
| τ·운영 지표 측정과 β 비교 | 완료 |
| PSI 범주형 값별 비율 계산과 기준 분포 재측정 | 완료 (커밋 `a1b5631`) |
| MLflow Tracking·Registry, 배포 게이트, 모델 로딩 | 구현·검증 완료 |
| v1 기준 운영 모델 등록 | 완료 — 버전 1이 `champion` |
| 카드 거래 CSV 업로드·배치 분석 API | 구현·검증 완료 |
| PSI·성능 감시와 재학습 연결 | 구현 중 (검증 전) |
| CSV 서버·대시보드·Docker | 구현. 컨테이너에서 champion으로 분석, 로컬과 판정 일치 확인 |
| 드리프트·자동 재학습 전체 시연 | 미완료 |

프론트 개발용 백엔드는 아래 명령으로 실행합니다. 기본 포트는 8099이며 Swagger는 `/docs`입니다. 업로드·결과·로그는 Docker 볼륨에 유지합니다. 기본 이미지는 모델 런타임이 없어 분석이 HTTP 503이며, 실제 판정은 [서버 가이드](../docs/FastAPI-CSV서버.md#모델-연결)의 `model-runtime` 절차를 따릅니다.

```bash
docker compose -f backend/serving_app/docker-compose.yml up -d --build
```

## 데이터와 모델

AI Hub 금융거래 합성데이터 1,952,871건, 카드 2,636장, 이상거래 비율 3.69%를 사용합니다. 정답을 포함한 이상거래유형·설명은 입력에서 제외하고 17개 피처를 만듭니다.

LSTM 구조는 수업 스켈레톤의 32 → 32 → 16을 유지하고, 출력과 손실 함수를 이상거래 분류용 sigmoid·binary crossentropy로 바꿨습니다. 2021~2023년 시퀀스 1,436,555개로 학습했습니다.

| 데이터에 기록된 거래 기간 | 역할 |
|---|---|
| 2021~2023 | 모델 학습, 스케일러 fit |
| 2024년 상반기 | τ 선택, β 비교, 감시 기준 측정 |
| 2024년 하반기 | 운영 시연. 재학습 시 fine-tune 구간과 게이트 홀드아웃을 이 안에서 시간순으로 나눔 |

## 4단계 결과

v1은 운영을 시작하는 기준 모델이므로 게이트 없이 `CardFraudLSTM` 버전 1로 등록하고 `champion` 별칭을 받았습니다. 모델·스케일러·τ는 같은 실행에서 불러옵니다. MLflow에서 다시 불러온 운영 모델은 2024년 상반기 검증 시퀀스 20,000건에서 로컬 v1과 점수 최대 차이 0, 판정 일치를 보였습니다.

배포 게이트는 재학습한 후보 v2를 현재 운영 모델과 같은 홀드아웃에서 비교합니다. 셋 다 통과해야 `champion`이 v2로 옮겨가고, 하나라도 실패하면 기존 운영 모델을 유지합니다.

1. F2가 현재 운영 모델보다 낮아지지 않아야 합니다.
2. Recall은 0.95 이상이어야 합니다.
3. 경보 비율은 6% 이하여야 합니다. 6%는 탐지팀 처리량에 대한 가정입니다.

홀드아웃은 fine-tuning과 τ 선택에 쓰지 않은 최근 구간입니다. 임시 DB와 합성 시험용 모델로 기준 등록·후보 실패·후보 교체·캐시 재로딩 흐름을 검증했습니다.

## 앞으로 할 일 — 작업 순서와 완료 기준

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| 1 | 운영 감시 연결 | PSI 주의·경고와 1,000건 창의 Precision·Recall을 기록하고, 성능 기준 미달이 2개 창 연속일 때 재학습을 요청합니다 |
| 2 | fine-tuning과 게이트·재로딩 연결 | 운영 모델에서 이어서 학습해 v2를 만들고, 같은 홀드아웃에서 운영 모델과 비교해 합격 시 모델·τ를 함께 교체하며 실패 시 기존 운영 모델을 유지합니다 |
| 3 | Docker·대시보드·시연 정리 | 정상·분포 변화·성능 저하 시나리오를 실행하고 API 응답·로그·Registry 기록을 남깁니다 |
| 4 | 제출 문서와 재현 점검 | 실제 구현 상태, 가정, 측정값을 맞추고 새 환경에서 실행 순서를 확인합니다 |

Recall 0.95와 경보 상한 6%는 v1의 상반기 측정값과 처리량 가정에서 정했습니다. 단순히 τ를 낮추면 경보량도 함께 늘 수 있으므로 두 기준을 동시에 확인합니다.

## 검증 자료

- [CSV 서버 실행·API·Docker 검증](../docs/FastAPI-CSV서버.md): 테스트 28개, 컨테이너 기동·업로드·champion 분석·로컬과 판정 일치·재시작 유지 검증을 정리했습니다.
- [검증 범위와 결과](../docs/results/검증결과.md): 테스트와 실제 모델 검증을 구분했습니다.

원본 데이터와 MLflow DB는 이 저장소에 포함되지 않으며, 복제한 환경에서는 실행 명령으로 새 기록을 생성합니다.

## 실행 방법

CSV 서버만 실행할 때는 [서버 가이드](../docs/FastAPI-CSV서버.md)의 `requirements-api.txt` 또는 Docker 명령을 사용합니다. 아래는 원본 데이터 준비부터 실제 모델 학습·등록까지의 기존 실행 순서입니다.

아래 명령은 이 저장소의 최상위 폴더에서 실행합니다. 원본 CSV는 별도로 `data/raw/train/`, `data/raw/valid/`에 준비합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt httpx2==2.13.0

python backend/scripts/prepare_data.py
python backend/scripts/train_baseline_v1.py
python backend/scripts/measure_v1.py
python backend/scripts/compare_beta.py
python -m backend.serving_app.train_and_register
python -m unittest discover -s backend/tests -t . -v
```

데이터 준비와 학습을 이미 실행했다면 `train_and_register`부터 실행합니다. 이 명령은 v1을 버전 1로 등록하고 `champion`으로 지정하며, 운영 모델이 이미 있으면 거부합니다.

MLflow 화면은 같은 폴더에서 실행합니다.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

MLflow 기록은 `mlflow.db`와 `mlruns/`에 저장됩니다. 다른 실행 폴더의 영향을 받지 않도록 코드에서는 저장 위치를 절대 경로로 고정했습니다. 테스트는 임시 저장소를 사용합니다.

## 핵심 파일

| 파일 | 역할 |
|---|---|
| `data/features.py` | 학습·서빙의 공용 전처리 |
| `serving_app/monitoring/metrics.py` | Precision, Recall, F2, average precision, PSI |
| `serving_app/monitoring/deployment_gate.py` | 후보와 운영 모델 비교 |
| `serving_app/registry.py` | MLflow 저장 위치와 운영 별칭 |
| `serving_app/train_and_register.py` | v1 기준 등록, 후보 평가·기록·조건부 교체 |
| `serving_app/model_loader.py` | 같은 버전의 모델·스케일러·τ 로딩 |
| `tests/` | 게이트와 Registry 흐름 검증 |

원본 데이터, 학습 모델, 스케일러, MLflow DB·아티팩트는 Git에서 제외합니다. 실행 전에 데이터 준비·학습이 필요합니다.

## 설계와 발표 자료

- [데이터 분석](../docs/데이터분석.md)
- [설계 지표](../docs/설계지표.md)
- [4단계 구현과 검증](../docs/4단계-MLflow와-배포게이트.md)
- [발표 노트](../docs/발표노트.md)

## 설계 가정과 확인할 사항

- 경보 상한 6%는 실제 탐지팀 인력으로 확인한 수치가 아니라 이번 실습의 처리량 가정입니다.
- 합성 시험용 모델의 승격·실패 테스트는 제어 흐름 검증입니다. 실제 새 모델 v2의 학습 성능을 보여주는 결과는 아닙니다.
- τ를 선택한 데이터와 게이트 홀드아웃을 분리합니다. 조정한 기준을 같은 데이터에서 다시 평가하면 성능이 낙관적으로 보일 수 있기 때문입니다.
- 실서비스의 이상거래 정답은 지연될 수 있습니다. 이후 시연에서 정답을 즉시 제공한다면 이 가정을 함께 명시합니다.
