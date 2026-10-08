"""카드 거래 CSV 검증. 전처리에서 잘못된 값을 0으로 덮기 전에 거부합니다."""
import csv
import io

import numpy as np
import pandas as pd

from data.features import CARD_TYPES, SEQ_LEN

REQUIRED_COLUMNS = [
    "카드KEY", "승인일자", "승인시간대", "승인SEQ", "통합승인금액", "국내해외여부",
    "카드이용한도금액", "연령", "가맹점누적매출금액_구간화", "개인법인구분코드_가맹점",
    "가맹점여부_신규", "인터넷판매여부", "일시불할부구분코드", "전월_매출건수",
    "전월_매출금액", "카드구분코드",
]
OPTIONAL_ZERO = {"", "_"}


def parse_csv(raw: bytes) -> pd.DataFrame:
    if not raw:
        raise ValueError("CSV 파일이 비어 있습니다.")
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("UTF-8 또는 CP949로 인코딩된 CSV가 필요합니다.")
    reader = csv.reader(io.StringIO(text), strict=True)
    try:
        header = [s.strip() for s in next(reader)]
        if len(set(header)) != len(header):
            raise ValueError("CSV에 중복된 컬럼명이 있습니다.")
        missing = sorted(set(REQUIRED_COLUMNS) - set(header))
        if missing:
            raise ValueError("필수 컬럼이 없습니다: " + ", ".join(missing))
        rows = []
        for row in reader:
            if not row or not any(s.strip() for s in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV {reader.line_num}행의 컬럼 수가 헤더와 다릅니다.")
            rows.append([s.strip() for s in row])
    except (csv.Error, StopIteration) as exc:
        raise ValueError("CSV 형식을 확인하세요.") from exc
    if not rows:
        raise ValueError("CSV에 거래가 없습니다.")
    columns = REQUIRED_COLUMNS + (["이상거래여부"] if "이상거래여부" in header else [])
    df = pd.DataFrame(rows, columns=header)[columns].copy()
    validate_transactions(df)
    return df


def validate_transactions(df: pd.DataFrame):
    def fail(column, bad, message):
        if bad.any():
            row = int(np.flatnonzero(bad.to_numpy())[0]) + 2
            raise ValueError(f"CSV {row}행 {column}: {message}")

    fail("카드KEY", df["카드KEY"].isin(OPTIONAL_ZERO), "카드 식별자가 필요합니다.")
    date_ok = df["승인일자"].str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(df["승인일자"], format="%Y%m%d", errors="coerce")
    fail("승인일자", ~date_ok | parsed.isna(), "YYYYMMDD 형식의 유효한 날짜가 필요합니다.")
    for column, lo, hi, integer, optional in [
        ("통합승인금액", 0, None, False, False),
        ("카드이용한도금액", 0, None, False, False),
        ("승인시간대", 0, 23, True, False),
        ("승인SEQ", 0, None, True, False),
        ("연령", 0, 120, True, True),
        ("가맹점누적매출금액_구간화", 0, None, True, True),
        ("전월_매출건수", None, None, True, True),
        ("전월_매출금액", None, None, False, True),
    ]:
        values = df[column].replace({s: "0" for s in OPTIONAL_ZERO}) if optional else df[column]
        numeric = pd.to_numeric(values, errors="coerce")
        bad = ~np.isfinite(numeric)
        if lo is not None:
            bad |= numeric < lo
        if hi is not None:
            bad |= numeric > hi
        if integer:
            bad |= numeric != np.floor(numeric)
        fail(column, bad, "값의 숫자 형식 또는 허용 범위를 확인하세요.")
        if column in {"승인SEQ", "승인시간대"}:
            fail(column, numeric > 2**53 - 1, "정수 범위를 초과했습니다.")
            df[column] = numeric.astype("int64").astype(str)
    for column, allowed in {
        "국내해외여부": {"0", "1"}, "카드구분코드": set(CARD_TYPES),
        "개인법인구분코드_가맹점": {"1", "2"} | OPTIONAL_ZERO,
        "가맹점여부_신규": {"0", "1"} | OPTIONAL_ZERO,
        "인터넷판매여부": {"0", "1"} | OPTIONAL_ZERO,
        "일시불할부구분코드": {"A", "B"} | OPTIONAL_ZERO,
    }.items():
        fail(column, ~df[column].isin(allowed), "허용된 코드가 아닙니다.")
    if "이상거래여부" in df:
        fail("이상거래여부", ~df["이상거래여부"].isin({"0", "1"} | OPTIONAL_ZERO), "0, 1 또는 미확정 빈 값이어야 합니다.")
    keys = ["카드KEY", "승인일자", "승인시간대", "승인SEQ"]
    fail("거래 키", df.duplicated(keys, keep=False), "같은 카드·날짜·시간대·승인SEQ의 거래가 중복되었습니다.")


def summarize_upload(df: pd.DataFrame) -> dict:
    counts = df.groupby("카드KEY").size()
    predictable = int((counts - (SEQ_LEN - 1)).clip(lower=0).sum())
    labelled = int(df["이상거래여부"].isin({"0", "1"}).sum()) if "이상거래여부" in df else 0
    return {"rows": len(df), "cards": len(counts), "predictable_rows": predictable,
            "excluded_rows": len(df) - predictable, "labelled_rows": labelled,
            "start_date": df["승인일자"].min(), "end_date": df["승인일자"].max()}
