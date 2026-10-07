"""
카드 거래를 LSTM 입력용 시퀀스로 바꾸는 공용 모듈입니다.

데이터 준비(scripts/prepare_data.py), 학습(serving_app/train_and_register.py),
서빙(serving_app/model_loader.py), 재학습(monitoring/retrain_trigger.py)이 모두
이 모듈을 거칩니다. 학습할 때와 서빙할 때 입력을 만드는 방식이 조금이라도 다르면
서버는 정상 응답하면서 틀린 확률을 내놓기 때문에, 변환 규칙을 이 파일 한 곳에만 둡니다.

입력 시퀀스: 같은 카드의 최근 SEQ_LEN(20)건 거래 (마지막 건이 판정 대상 거래)
타깃: 마지막 거래의 이상거래여부 (0 또는 1)
"""
import glob
import pickle

import numpy as np
import pandas as pd

SEQ_LEN = 20

TRAIN_END = "20231231"
VALID_END = "20240630"

LEAK_COLUMNS = ["이상거래유형", "이상거래설명"]
CARD_TYPES = ["1", "2", "3", "4", "5"]

FEATURES = [
    "amount_log",
    "hour",
    "overseas",
    "limit_log",
    "age",
    "merchant_sales_bin",
    "merchant_corp",
    "merchant_new",
    "internet",
    "installment",
    "merchant_prev_cnt_log",
    "merchant_prev_amt_log",
] + [f"card_type_{t}" for t in CARD_TYPES]
N_FEATURES = len(FEATURES)


def load_transactions(pattern: str = "data/raw/*/*.csv") -> pd.DataFrame:
    """CSV를 모두 읽어 합치고, 카드별 시간순으로 정렬합니다.

    AI Hub의 Training/Validation은 같은 기간에서 무작위로 나뉘어 있어서
    두 폴더를 합친 뒤 시간 기준으로 다시 나눕니다(time_split).
    """
    files = sorted(glob.glob(pattern))
    df = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    df = df.drop(columns=LEAK_COLUMNS, errors="ignore")
    return sort_transactions(df)


def sort_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_seq"] = df["승인SEQ"].astype(int)
    df = df.sort_values(["카드KEY", "승인일자", "승인시간대", "_seq"]).drop(columns="_seq")
    return df.reset_index(drop=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _log(s: pd.Series) -> pd.Series:
    return np.log1p(_num(s).clip(lower=0))


def _flag(s: pd.Series, value: str) -> pd.Series:
    return (s == value).astype(float)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼을 모델이 읽을 수 있는 숫자 피처로 바꿉니다.

    금액처럼 범위가 수천 원에서 수천만 원까지 벌어지는 값은 로그를 씌워 간격을 줄입니다.
    가맹점 전월 매출에는 취소가 매출보다 많아 음수인 값이 있어서, 로그 전에 0으로 자릅니다.
    코드값은 이상거래 유형 정의에 나오는 것(법인 가맹점, 신규 가맹점, 할부, 가족카드)만
    0/1 플래그로 꺼냅니다. 빈칸과 '_'는 0으로 처리합니다.
    """
    out = pd.DataFrame(index=df.index)
    out["amount_log"] = _log(df["통합승인금액"])
    out["hour"] = _num(df["승인시간대"])
    out["overseas"] = _flag(df["국내해외여부"], "1")
    out["limit_log"] = _log(df["카드이용한도금액"])
    out["age"] = _num(df["연령"])
    out["merchant_sales_bin"] = _num(df["가맹점누적매출금액_구간화"])
    out["merchant_corp"] = _flag(df["개인법인구분코드_가맹점"], "2")
    out["merchant_new"] = _flag(df["가맹점여부_신규"], "1")
    out["internet"] = _flag(df["인터넷판매여부"], "1")
    out["installment"] = _flag(df["일시불할부구분코드"], "B")
    out["merchant_prev_cnt_log"] = _log(df["전월_매출건수"])
    out["merchant_prev_amt_log"] = _log(df["전월_매출금액"])
    for t in CARD_TYPES:
        out[f"card_type_{t}"] = _flag(df["카드구분코드"], t)
    out = out[FEATURES].astype("float32")
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("피처에 NaN 또는 무한대가 있습니다. 원본 값의 범위를 확인하세요.")
    return out


class FraudScaler:
    """피처마다 학습 기간의 최솟값과 최댓값으로 [0, 1] 범위에 맞추는 min-max 스케일러입니다.

    학습 기간 데이터로 한 번만 fit하고 serving_app/models/scaler.pkl에 저장합니다.
    fine-tuning 때 다시 fit하면 같은 금액이 다른 숫자로 들어가 기존 가중치와 어긋나므로
    서빙과 재학습 모두 저장된 스케일러를 그대로 씁니다.
    """

    def __init__(self):
        self.lo = None
        self.hi = None

    def fit(self, encoded: pd.DataFrame) -> "FraudScaler":
        self.lo = encoded.min().to_numpy(dtype="float32")
        self.hi = encoded.max().to_numpy(dtype="float32")
        return self

    def transform(self, encoded: pd.DataFrame) -> np.ndarray:
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        scaled = (encoded.to_numpy(dtype="float32") - self.lo) / span
        return np.clip(scaled, 0.0, 1.0).astype("float32")

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"lo": self.lo, "hi": self.hi, "features": FEATURES}, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "FraudScaler":
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state["features"] != FEATURES:
            raise ValueError("scaler.pkl의 피처 목록이 현재 코드와 다릅니다. prepare_data.py를 다시 실행하세요.")
        scaler = cls()
        scaler.lo, scaler.hi = state["lo"], state["hi"]
        return scaler


def build_sequences(df: pd.DataFrame, scaler: FraudScaler, seq_len: int = SEQ_LEN):
    """카드별로 최근 seq_len건을 묶어 시퀀스를 만듭니다.

    df는 sort_transactions로 정렬된 상태여야 합니다. 카드의 첫 seq_len - 1건은
    앞선 거래가 모자라 시퀀스를 만들 수 없으므로 판정 대상에서 빠집니다.

    반환: X (n, seq_len, N_FEATURES), y (n,), dates (n,) 판정 대상 거래의 승인일자,
          cards (n,) 판정 대상 거래의 카드KEY
    """
    scaled = scaler.transform(encode(df))
    cards = df["카드KEY"].to_numpy()
    pos_in_card = df.groupby("카드KEY").cumcount().to_numpy()
    ends = np.flatnonzero(pos_in_card >= seq_len - 1)
    window = ends[:, None] - (seq_len - 1) + np.arange(seq_len)
    X = scaled[window]
    y = df["이상거래여부"].astype(int).to_numpy()[ends]
    dates = df["승인일자"].to_numpy()[ends]
    return X, y, dates, cards[ends]


def sequence_from_rows(rows: list[dict], scaler: FraudScaler) -> np.ndarray:
    """서빙용: 요청으로 들어온 거래 seq_len건을 모델 입력 한 개로 바꿉니다."""
    df = pd.DataFrame(rows).astype(str)
    return scaler.transform(encode(df))


def time_split(dates: np.ndarray) -> dict:
    """판정 대상 거래의 날짜로 학습, 검증, 운영 구간을 나눕니다.

    학습: 2021~2023 / 검증과 측정: 2024년 상반기 / 운영 시연: 2024년 하반기
    시퀀스 안의 앞선 거래는 경계를 넘어 과거 구간에 걸쳐도 됩니다. 판정 시점보다
    미래의 거래가 들어가지 않기 때문입니다.
    """
    return {
        "train": dates <= TRAIN_END,
        "valid": (dates > TRAIN_END) & (dates <= VALID_END),
        "ops": dates > VALID_END,
    }
