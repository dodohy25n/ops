"""시연용 월별 CSV를 만듭니다. 실행(프로젝트 최상위): python backend/scripts/make_scenarios.py

2024 하반기 실제 운영 거래를 월별로 나누고, 각 카드의 직전 19건을 이력으로 붙입니다.
분석할 때 판정 시작일을 그 달 1일로 주면 이력 행은 판정하지 않고 시퀀스에만 씁니다.

  07·08월  ① 정상: 원본 그대로
  09월     ② 명절: 9/13~9/19 거래 금액을 1.5배로 (정답은 그대로) → 분포만 바뀜
  10~12월  ③ 신종 사기: 매달 카드 80장에 해외·소액·인터넷 새벽 결제 4건 연속을 사기로 추가

②와 ③은 시뮬레이션이며 발표에서 그렇게 밝힙니다. 생성 파일은 data/scenarios/에 저장하고 Git에서 제외합니다.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.features import SEQ_LEN, sort_transactions
from backend.serving_app.config import DATA_DIR

OUT_DIR = DATA_DIR / "scenarios"
MONTHS = ["07", "08", "09", "10", "11", "12"]
HOLIDAY = ("20240913", "20240919")
HOLIDAY_AMOUNT = 1.5
NEW_FRAUD_MONTHS = {"10", "11", "12"}
NEW_FRAUD_CARDS = 80
NEW_FRAUD_BURST = 4
SEED = 7


def new_fraud_rows(month_rows, month, rng):
    rows = []
    for card in rng.choice(month_rows["카드KEY"].unique(), NEW_FRAUD_CARDS, replace=False):
        base = month_rows[month_rows["카드KEY"] == card].iloc[-1].to_dict()
        day = f"2024{month}{rng.integers(1, 29):02d}"
        for i in range(NEW_FRAUD_BURST):
            row = dict(base)
            row.update({"승인일자": day, "승인시간대": str(2 + i), "승인SEQ": str(9000 + i),
                        "국내해외여부": "1", "통합승인금액": str(1000 + 300 * i), "인터넷판매여부": "1",
                        "이상거래여부": "1"})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = pd.read_csv(DATA_DIR / "processed/ops_rows.csv", dtype=str, keep_default_na=False)
    rows = rows.drop(columns=["이상거래유형", "이상거래설명"], errors="ignore")
    in_holiday = rows["승인일자"].between(*HOLIDAY)
    amount = pd.to_numeric(rows.loc[in_holiday, "통합승인금액"])
    rows.loc[in_holiday, "통합승인금액"] = (amount * HOLIDAY_AMOUNT).astype(int).astype(str)
    added = [new_fraud_rows(rows[rows["승인일자"].str[:6] == f"2024{m}"], m, rng) for m in sorted(NEW_FRAUD_MONTHS)]
    rows = sort_transactions(pd.concat([rows, *added], ignore_index=True))
    position = rows.groupby("카드KEY").cumcount()
    for month in MONTHS:
        in_month = rows["승인일자"].str[:6] == f"2024{month}"
        first = position.where(in_month).groupby(rows["카드KEY"]).transform("min")
        keep = first.notna() & (position >= first - (SEQ_LEN - 1)) & (rows["승인일자"] <= f"2024{month}31")
        path = OUT_DIR / f"2024-{month}.csv"
        rows[keep].to_csv(path, index=False)
        print(f"{path.name}: {int(keep.sum()):,}행 (판정 대상 {int(in_month.sum()):,}건)")


if __name__ == "__main__":
    main()
