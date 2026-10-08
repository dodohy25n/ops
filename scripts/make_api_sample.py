"""CSV 업로드 확인용 가상 거래 50건. 실제 모델의 성능 평가 데이터가 아닙니다."""
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.csv_input import REQUIRED_COLUMNS


def sample_rows():
    rows = []
    for card in ("sample-card-01", "sample-card-02"):
        for i in range(25):
            rows.append({
                "카드KEY": card, "승인일자": (date(2024, 8, 1) + timedelta(days=i)).strftime("%Y%m%d"),
                "승인시간대": "12", "승인SEQ": str(i + 1), "통합승인금액": str(10000 + i * 100),
                "국내해외여부": "0", "카드이용한도금액": "3000000", "연령": "35",
                "가맹점누적매출금액_구간화": "2", "개인법인구분코드_가맹점": "1",
                "가맹점여부_신규": "0", "인터넷판매여부": "0", "일시불할부구분코드": "A",
                "전월_매출건수": "100", "전월_매출금액": "1000000", "카드구분코드": "1",
            })
    return rows


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "data/sample_card_transactions.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sample_rows())
    print(path)
