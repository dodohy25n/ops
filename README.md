# 카드 거래 이상탐지 AIOps 조별 실습

W13 모델 서빙 및 AIOps 과정에서 카드별 최근 20건의 거래로 이상거래를 판정하는 LSTM과 모델 운영 흐름을 구현합니다. 과제 마감은 2026-10-08입니다.

## 현재 진행 상태

| 작업 | 상태 |
|---|---|
| 원본 분석, 시간 분할, 공용 전처리 | 완료 |
| 전체 카드 사용, 시퀀스 20건, v1 학습 | 완료 |
| τ·운영 지표 측정과 β 비교 | 완료 |
| MLflow Tracking·Registry, 배포 게이트, 모델 로딩 | 구현·검증 완료 |
| 실제 v1 운영 승격 | Recall 기준 미달로 보류 |
| 카드 거래 FastAPI, 운영 감시, 재학습, 컨테이너 | 다음 단계 |

현재 FastAPI 라우터와 대시보드, Docker 설정은 원본 HAIC 실습 스켈레톤이 남아 있습니다. 카드 거래 서버의 실행 방법은 해당 단계가 완료되면 추가합니다.

## 데이터와 모델

AI Hub 금융거래 합성데이터 1,952,871건, 카드 2,636장, 이상거래 비율 3.69%를 사용합니다. 정답을 포함한 이상거래유형·설명은 입력에서 제외하고 17개 피처를 만듭니다.

LSTM 구조는 수업 스켈레톤의 32 → 32 → 16을 유지하고, 출력과 손실 함수를 이상거래 분류용 sigmoid·binary crossentropy로 바꿨습니다. 2021~2023년 시퀀스 1,436,555개로 학습했습니다.

| 데이터에 기록된 거래 기간 | 역할 |
|---|---|
| 2021~2023 | 모델 학습, 스케일러 fit |
| 2024년 상반기 | τ 선택, β 비교, 감시 기준 측정 |
| 2024년 7월 | 최초 배포 게이트의 독립 평가 |
| 2024년 8~12월 | 이후 운영 시연 |

## 4단계 결과

배포 게이트는 후보를 현재 운영 모델과 같은 데이터에서 비교합니다.

1. F2가 현재 운영 모델보다 낮아지지 않아야 합니다. 최초 모델은 이 비교를 생략합니다.
2. Recall은 0.95 이상이어야 합니다.
3. 경보 비율은 6% 이하여야 합니다. 6%는 탐지팀 처리량에 대한 가정입니다.

실제 v1은 τ=0.28을 고정하고 7월 40,324건(실제 이상거래 1,496건)에 적용했습니다.

| 지표 | 값 |
|---|---|
| Recall | 0.9104 |
| Precision | 0.7864 |
| F2 | 0.8826 |
| PR-AUC (average precision) | 0.9310 |
| 경보 비율 | 4.2952% |
| 판정 | 후보 등록 완료, 운영 승격 보류 |

후보는 `CardFraudLSTM`의 Registry 버전으로 보관합니다. 게이트를 통과해야 운영 모델을 선택하는 `champion` 별칭을 받습니다. 최초 v1은 기준에 미달했으므로 현재 운영 모델은 없습니다.

모델·스케일러·τ는 같은 실행에서 불러옵니다. 등록 전후 40,324건의 점수와 판정이 일치했습니다. 임시 DB와 합성 시험용 모델로 통과·실패·버전 교체·캐시 재로딩도 검증했습니다.

## 실행 방법

아래 명령은 이 저장소의 최상위 폴더에서 실행합니다. 원본 CSV는 별도로 `data/raw/train/`, `data/raw/valid/`에 준비합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py
python scripts/train_baseline_v1.py
python scripts/measure_v1.py
python scripts/compare_beta.py
python -m serving_app.train_and_register
python scripts/verify_stage4.py
python -m unittest discover -s tests -v
```

데이터 준비와 학습을 이미 실행했다면 `train_and_register`부터 실행합니다. 최초 게이트 미달 시에도 후보와 지표는 저장되며, 등록 명령은 종료 코드 2로 배포 보류를 알립니다. 이는 현재 실제 측정 결과입니다.

MLflow 화면은 같은 폴더에서 실행합니다.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

MLflow 기록은 `mlflow.db`와 `mlruns/`에 저장됩니다. 다른 실행 폴더의 영향을 받지 않도록 코드에서는 저장 위치를 절대 경로로 고정했습니다. 테스트는 임시 저장소를 사용합니다.

## 핵심 파일

| 파일 | 역할 |
|---|---|
| `data/features.py` | 학습·서빙의 공용 전처리 |
| `data/gate_holdout.py` | 7월 독립 평가 데이터 생성 |
| `serving_app/monitoring/metrics.py` | Precision, Recall, F2, average precision, PSI |
| `serving_app/monitoring/deployment_gate.py` | 배포 조건 검사 |
| `serving_app/registry.py` | MLflow 저장 위치와 운영 별칭 |
| `serving_app/train_and_register.py` | 후보 평가, 기록, 등록, 조건부 승격 |
| `serving_app/model_loader.py` | 같은 버전의 모델·스케일러·τ 로딩 |
| `scripts/verify_stage4.py` | 실제 v1의 저장 전후 일치 검증 |
| `tests/` | 게이트와 Registry 흐름 검증 |

원본 데이터, 학습 모델, 스케일러, MLflow DB·아티팩트는 Git에서 제외합니다. 실행 전에 데이터 준비·학습이 필요합니다.

## 설계와 발표 자료

- [데이터 분석](docs/데이터분석.md)
- [설계 지표](docs/설계지표.md)
- [4단계 구현과 검증](docs/4단계-MLflow와-배포게이트.md)
- [발표 노트](docs/발표노트.md)

## 작성자가 채울 설명

<!-- TODO: τ 선택 데이터와 배포 평가 데이터를 나눠야 하는 이유를 본인이 이해한 말로 두세 문장 작성해 주세요. -->
<!-- TODO: 경보 상한 6%를 어떤 탐지팀 상황으로 가정했는지 팀에서 정한 근거를 작성해 주세요. -->
<!-- TODO: v1이 독립 평가에서 실패한 결과를 보고 다음에 무엇을 개선하려는지 본인의 판단을 작성해 주세요. -->
