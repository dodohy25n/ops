# HAIC AIOps 프론트엔드

교수님 AIOps 운영 도구의 화면 구성을 참고하여 만든 프론트엔드입니다.

## 포함 파일

- `index.html`: Dashboard, Simulation, Datasets, System 화면과 API 호출 코드
- `favicon.ico`: 브라우저 아이콘

## 프로젝트에 적용하는 위치

두 파일을 FastAPI 프로젝트의 다음 경로에 넣습니다.

```text
serving_app/static/
├── index.html
└── favicon.ico
```

## 프론트에서 사용하는 API

- `GET /health`: 서버 및 모델 상태
- `GET /data/status`: 활성 데이터셋 정보
- `POST /data/upload`: CSV 업로드
- `POST /predict`: 20일 시퀀스 단건 예측
- `POST /predict/batch-test`: 정상·드리프트 배치 시뮬레이션
- `GET /logs`: 로그 파일 목록
- `GET /logs/{filename}`: 로그 내용

`index.html`은 FastAPI와 같은 주소에서 서비스되는 것을 기준으로 상대 경로 API를 호출합니다. 파일을 더블클릭하여 여는 방식이 아니라 FastAPI 서버를 실행한 뒤 접속해야 합니다.

예시:

```bash
uvicorn serving_app.main:app --reload --port 8077
```

접속 주소:

```text
http://127.0.0.1:8077/
```

## 현재 확인된 범위

- 탭 이동, 상태 조회, 데이터셋 조회, 로그 조회 정상
- 단건 예측 `POST /predict` 정상
- CSV 업로드 및 드리프트 재학습은 백엔드와 MLflow 환경이 준비되어 있어야 전체 실행 가능
