"""실행 중인 서버에 월별 시나리오 CSV를 순서대로 넣어 감시 → 재학습 → 게이트 흐름을 보여줍니다.

실행(프로젝트 최상위): python backend/scripts/run_demo.py --api http://127.0.0.1:8077
먼저 python backend/scripts/make_scenarios.py로 data/scenarios/를 만듭니다.
표준 라이브러리만 쓰므로 서버와 다른 환경에서도 실행할 수 있습니다.
"""
import argparse
import calendar
import json
import time
import unicodedata
import urllib.request
import uuid
from pathlib import Path

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "data/scenarios"
LABELS = {"07": "① 정상", "08": "① 정상", "09": "② 명절(금액 1.5배)", "10": "③ 신종 사기",
          "11": "③ 신종 사기", "12": "③ 신종 사기"}
DONE = {"promoted", "rejected", "failed"}


def call(api, path, body=None, content_type="application/json"):
    request = urllib.request.Request(api + path, data=body, method="POST" if body is not None else "GET")
    if body is not None:
        request.add_header("Content-Type", content_type)
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def upload(api, path):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            "Content-Type: text/csv\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return call(api, "/data/upload", body, f"multipart/form-data; boundary={boundary}")


def wait_for_retrain(api, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        retrain = call(api, "/monitoring/status?limit=1")["state"]["retrain"]
        if retrain.get("status") in DONE:
            return retrain
        time.sleep(3)
    raise TimeoutError("재학습이 제한 시간 안에 끝나지 않았습니다.")


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def pad(text, width):
    """한글은 터미널에서 두 칸을 차지하므로 화면 폭 기준으로 채웁니다."""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - shown, 0)


def print_summary(api, rows):
    columns = (("월", 9), ("시나리오", 20), ("모델", 6), ("Recall", 8), ("F2", 7), ("재학습 결과", 0))
    print("\n[요약] 월별 판정과 재학습 결과")
    print("  " + "".join(pad(name, width) for name, width in columns))
    for row in rows:
        print("  " + "".join(pad(value, width) for value, (_, width) in zip(row, columns)))
    print(f"\n대시보드: {api}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8077")
    parser.add_argument("--months", nargs="+", default=list(LABELS))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    rows = []
    for month in args.months:
        path = SCENARIO_DIR / f"2024-{month}.csv"
        uploaded = upload(args.api, path)
        last = calendar.monthrange(2024, int(month))[1]
        result = call(args.api, "/predict/batch", json.dumps({
            "upload_id": uploaded["upload_id"], "start_date": f"2024-{month}-01",
            "end_date": f"2024-{month}-{last}"}).encode())
        m, mon = result["metrics"], result["monitoring"]
        windows = mon["windows_closed"]
        below = {k: sum(w["status"][k] == "below" for w in windows) for k in ("precision", "recall")}
        psi = {level: sum(w["psi_level"] == level for w in windows) for level in ("warn", "alert")}
        outcome = "-"
        print(f"\n[2024-{month}] {LABELS.get(month, '')} | 모델 v{result['model_version']} τ {result['tau']}")
        print(f"  판정 {result['predictions']:,}건 | Recall {fmt(m['recall'])} Precision {fmt(m['precision'])} "
              f"F2 {fmt(m['f2'])} 경보 {m['alert_rate']:.2%}")
        print(f"  닫힌 창 {len(windows)}개 | 기준 미달 창 Precision {below['precision']} Recall {below['recall']} | "
              f"PSI 주의 {psi['warn']} 경고 {psi['alert']} | 연속 미달 {mon['consecutive']}")
        if mon["retrain_requested"]:
            print(f"  → 재학습 요청 ({', '.join(mon['retrain_reason'])} 2개 창 연속 미달). 게이트 결과를 기다립니다…")
            retrain = wait_for_retrain(args.api, args.timeout)
            print(f"  → 재학습 결과: {retrain['status']}")
            outcome = retrain["status"]
            if retrain.get("candidate_version"):
                outcome += f" (후보 v{retrain['candidate_version']})"
            if retrain.get("candidate"):
                c, cur = retrain["candidate"], retrain["current"]
                print(f"     후보 v{retrain.get('candidate_version')} τ {retrain['tau']}: Recall {c['recall']:.3f} "
                      f"F2 {c['f2']:.3f} 경보 {c['alert_rate']:.2%}")
                print(f"     운영 모델: Recall {cur['recall']:.3f} F2 {cur['f2']:.3f} 경보 {cur['alert_rate']:.2%}")
                print(f"     실패한 조건: {retrain['failed_checks'] or '없음'} | 구간 {retrain['periods']}")
            if retrain.get("error"):
                print(f"     오류: {retrain['error']}")
        rows.append((f"2024-{month}", LABELS.get(month, ""), f"v{result['model_version']}",
                     fmt(m["recall"]), fmt(m["f2"]), outcome))
    print_summary(args.api, rows)


if __name__ == "__main__":
    main()
