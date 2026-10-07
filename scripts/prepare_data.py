"""
카드 거래 원본 CSV를 학습, 검증, 운영 구간으로 나눠 저장합니다.

실행: python scripts/prepare_data.py

만들어지는 파일
  serving_app/models/scaler.pkl   학습 기간으로 fit한 스케일러 (서빙, 재학습이 그대로 재사용)
  data/processed/train.npz        학습 시퀀스 (2021~2023)
  data/processed/valid.npz        검증과 측정용 시퀀스 (2024년 상반기)
  data/processed/ops_rows.csv     운영 시연용 원본 거래 (2024년 하반기 + 카드별 직전 19건)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import (
    SEQ_LEN,
    TRAIN_END,
    VALID_END,
    FraudScaler,
    build_sequences,
    encode,
    load_transactions,
    time_split,
)

OUT_DIR = "data/processed"


def ops_rows_with_history(df):
    pos = df.groupby("카드KEY").cumcount()
    in_ops = df["승인일자"] > VALID_END
    first_ops = pos.where(in_ops).groupby(df["카드KEY"]).transform("min")
    keep = first_ops.notna() & (pos >= first_ops - (SEQ_LEN - 1))
    return df[keep]


def describe(name, y, cards):
    print(f"  {name:5s} 시퀀스 {len(y):>9,}개 | 이상거래 {int(y.sum()):>7,}건 ({y.mean():.4f}) | 카드 {len(set(cards)):,}장")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_transactions()
    print(f"[1] 원본 거래 {len(df):,}건, 이상거래 비율 {df['이상거래여부'].astype(int).mean():.4f}")

    scaler = FraudScaler().fit(encode(df[df["승인일자"] <= TRAIN_END]))
    scaler.save()
    print(f"[2] 스케일러를 학습 기간(~{TRAIN_END}) 거래로 fit -> serving_app/models/scaler.pkl")

    X, y, dates, cards = build_sequences(df, scaler)
    split = time_split(dates)
    print(f"[3] 시퀀스 {len(y):,}개 생성 (카드별 첫 {SEQ_LEN - 1}건은 앞선 거래가 부족해 제외)")

    print("[4] 시간 분할")
    describe("train", y[split["train"]], cards[split["train"]])
    describe("valid", y[split["valid"]], cards[split["valid"]])
    describe("ops", y[split["ops"]], cards[split["ops"]])

    np.savez(f"{OUT_DIR}/train.npz", X=X[split["train"]], y=y[split["train"]])
    np.savez(f"{OUT_DIR}/valid.npz", X=X[split["valid"]], y=y[split["valid"]], dates=dates[split["valid"]].astype("U8"))
    ops = ops_rows_with_history(df)
    ops.to_csv(f"{OUT_DIR}/ops_rows.csv", index=False)
    print(f"[5] 저장 완료 -> {OUT_DIR}/ (ops_rows.csv {len(ops):,}행)")


if __name__ == "__main__":
    main()
