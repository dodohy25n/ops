"""외부 모델 산출물 없이 통합 테스트를 재현하기 위한 작은 시험용 도구."""
from pathlib import Path

import numpy as np

from data.features import FEATURES, N_FEATURES, FraudScaler


def write_test_scaler(path: str | Path) -> Path:
    """합성 거래 테스트에 필요한 고정 스케일러를 임시 경로에 저장합니다.

    실제 운영용 ``scaler.pkl``은 원본 학습 데이터로 만들어야 합니다. 테스트에서는
    금액 피처만 정상 거래와 고액 거래를 구분할 수 있으면 되므로 최소 범위만 정의합니다.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scaler = FraudScaler()
    scaler.lo = np.zeros(N_FEATURES, dtype="float32")
    scaler.hi = np.ones(N_FEATURES, dtype="float32")
    scaler.hi[FEATURES.index("amount_log")] = np.log1p(5_000_000).astype("float32")
    scaler.save(output)
    return output
