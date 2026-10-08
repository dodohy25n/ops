#!/usr/bin/env bash
# 발표 시연을 한 번에 실행합니다. 실행(어디서든): bash backend/scripts/demo_docker.sh
# model-runtime 컨테이너를 새 볼륨으로 띄우고 v1 등록 → 6개월 시나리오 분석까지 진행합니다.
# 컨테이너는 Docker 볼륨 안의 Registry만 쓰므로 호스트의 backend/mlflow.db와 backend/mlruns는 바뀌지 않습니다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export FRAUD_DOCKER_TARGET=model-runtime
PORT="${FRAUD_API_PORT:-8099}"
API="http://127.0.0.1:${PORT}"
PYTHON="${PYTHON:-python3}"
COMPOSE=(docker compose -f backend/serving_app/docker-compose.yml)

echo "[1/5] 이전 컨테이너와 볼륨을 지우고 새로 띄웁니다."
"${COMPOSE[@]}" down -v
"${COMPOSE[@]}" up -d --build

echo "[2/5] 서버가 healthy가 될 때까지 기다립니다."
CONTAINER="$("${COMPOSE[@]}" ps -q serving-api)"
for _ in $(seq 60); do
  STATUS="$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER")"
  [ "$STATUS" = "healthy" ] && break
  sleep 2
done
if [ "$STATUS" != "healthy" ]; then
  echo "서버가 healthy가 되지 않았습니다(상태: $STATUS). 로그: ${COMPOSE[*]} logs serving-api" >&2
  exit 1
fi

echo "[3/5] 컨테이너 안의 빈 Registry에 v1을 기준 운영 모델로 등록합니다."
if ! OUTPUT="$("${COMPOSE[@]}" exec -T serving-api python -m backend.serving_app.train_and_register 2>&1)"; then
  echo "$OUTPUT" >&2
  exit 1
fi
echo "$OUTPUT" | grep "등록"

echo "[4/5] 시나리오 CSV를 확인합니다."
for MONTH in 07 08 09 10 11 12; do
  if [ ! -f "data/scenarios/2024-${MONTH}.csv" ]; then
    echo "data/scenarios/2024-${MONTH}.csv가 없습니다. 원본 데이터가 있는 환경에서 먼저 생성하세요." >&2
    echo "  .venv/bin/python backend/scripts/make_scenarios.py" >&2
    echo "서버는 켜져 있으므로 생성 후 아래 명령만 실행하면 됩니다." >&2
    echo "  $PYTHON backend/scripts/run_demo.py --api $API" >&2
    exit 1
  fi
done

echo "[5/5] 7월부터 12월까지 월별 CSV를 순서대로 분석합니다."
"$PYTHON" backend/scripts/run_demo.py --api "$API"
echo "정리: ${COMPOSE[*]} down"
