"""판정 결과를 1,000건 창으로 묶어 PSI와 Precision·Recall을 감시하고 재학습 여부를 정합니다.

CSV 분석이 끝날 때마다 판정 대상 거래를 시간순으로 이어 붙이고, 창이 찰 때마다 평가합니다.
기록은 현재 운영 모델 버전 단위로 쌓습니다. champion이 바뀌면 판정 창을 처음부터 다시 셉니다.
PSI는 알림만 보내고, 재학습은 성능 지표가 평소 범위(평균 − 2σ) 아래로 연속해서 떨어질 때만 요청합니다.
"""
import json
import time
from threading import RLock

import pandas as pd

from backend.serving_app.config import monitor_dir
from backend.serving_app.monitoring.metrics import psi

METRICS = ("precision", "recall")
KEY_COLUMNS = ["card_key", "approval_date", "hour", "approval_seq"]
_lock = RLock()


def _paths():
    root = monitor_dir()
    root.mkdir(parents=True, exist_ok=True)
    return {"state": root / "state.json", "observations": root / "observations.csv",
            "windows": root / "windows.jsonl", "events": root / "events.jsonl", "root": root}


def _new_state(version):
    return {"model_version": version, "observations": 0, "windows": 0,
            "consecutive": {m: 0 for m in METRICS}, "retrain": {"status": "idle"}}


def load_state():
    path = _paths()["state"]
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def save_state(state):
    path = _paths()["state"]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(name, record):
    with _paths()[name].open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(name, limit=None):
    path = _paths()[name]
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in (lines[-limit:] if limit else lines)]


def load_observations():
    path = _paths()["observations"]
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"card_key": str, "approval_date": str, "approval_seq": str,
                                    "actual": str, "upload_id": str, "analysis_id": str},
                       keep_default_na=False)


def _start_generation(paths, state, version):
    """운영 모델이 바뀌면 이전 기록을 버전 이름으로 보관하고 판정 창을 초기화합니다."""
    if state is not None:
        for name in ("observations", "windows"):
            if paths[name].is_file():
                paths[name].replace(paths["root"] / f"{paths[name].stem}-v{state['model_version']}{paths[name].suffix}")
        append_jsonl("events", {"ts": time.time(), "type": "window_reset",
                                "from_version": state["model_version"], "to_version": version})
    state = _new_state(version)
    save_state(state)
    return state


def current_state(version):
    with _lock:
        state = load_state()
        if state is None or state["model_version"] != version:
            state = _start_generation(_paths(), state, version)
        return state


def evaluate_window(rows, criteria, psi_refs):
    trigger = criteria["trigger"]
    labelled = rows[rows["actual"].isin({"0", "1"})]
    y = labelled["actual"].astype(int).to_numpy()
    alert = labelled["alert"].astype(bool).to_numpy()
    tp, alerts, frauds = int((alert & (y == 1)).sum()), int(alert.sum()), int(y.sum())
    values = {
        "precision": tp / alerts if alerts >= trigger["min_alerts"] else None,
        "recall": tp / frauds if frauds >= trigger["min_frauds"] else None,
    }
    floors = {m: trigger[m]["minus_2sigma"] for m in METRICS}
    status = {m: "insufficient" if values[m] is None else ("below" if values[m] < floors[m] else "ok")
              for m in METRICS}
    psi_values = {f: psi(ref, rows[f"psi_{f}"].to_numpy(dtype="float64")) for f, ref in psi_refs["features"].items()}
    psi_values["score"] = psi(psi_refs["score"], rows["score"].to_numpy(dtype="float64"))
    worst = max(psi_values.values())
    level = "alert" if worst >= psi_refs["alert"] else "warn" if worst >= psi_refs["warn"] else "ok"
    return {"start": f"{rows['approval_date'].iloc[0]}", "end": f"{rows['approval_date'].iloc[-1]}",
            "samples": len(rows), "labelled": len(labelled), "alerts": alerts, "frauds": frauds,
            **values, "floors": floors, "status": status, "psi": psi_values, "psi_level": level}


def record(analysis, observations, model):
    """분석 한 건의 판정 거래를 누적하고, 새로 닫힌 창을 평가합니다."""
    settings = model.settings
    trigger = settings["trigger"]
    with _lock:
        paths = _paths()
        state = current_state(model.version)
        existing = load_observations()
        new = observations.copy()
        new["hour"] = new["hour"].astype(int)
        if len(existing):
            seen = set(map(tuple, existing[KEY_COLUMNS].astype(str).to_numpy()))
            fresh = [tuple(k) not in seen for k in new[KEY_COLUMNS].astype(str).to_numpy()]
            new = new[fresh]
        skipped = len(observations) - len(new)
        if len(new):
            new.to_csv(paths["observations"], mode="a", header=not paths["observations"].is_file(), index=False)
        total = state["observations"] + len(new)
        closed, reasons, trigger_window = [], [], None
        waiting = state["retrain"]["status"] in {"requested", "running"}
        if total // trigger["window_size"] > state["windows"]:
            rows = pd.concat([existing, new], ignore_index=True) if len(existing) else new.reset_index(drop=True)
            while (state["windows"] + 1) * trigger["window_size"] <= total:
                index = state["windows"]
                window_rows = rows.iloc[index * trigger["window_size"]:(index + 1) * trigger["window_size"]]
                result = evaluate_window(window_rows, settings, settings["psi"])
                for m in METRICS:
                    if result["status"][m] == "below":
                        state["consecutive"][m] += 1
                    elif result["status"][m] == "ok":
                        state["consecutive"][m] = 0
                # 표본이 부족한 창은 판정을 보류하고 연속 횟수를 그대로 둡니다.
                result.update(window=index + 1, model_version=model.version, tau=model.tau,
                              analysis_id=analysis["analysis_id"], consecutive=dict(state["consecutive"]))
                # 조건을 충족한 창에서 바로 재학습을 요청합니다. 뒤의 정상 창이 연속 횟수를 지우기 전에 판단해야 합니다.
                hit = [m for m in METRICS if state["consecutive"][m] >= trigger["consecutive"]]
                if hit and not reasons and not waiting:
                    reasons, trigger_window = hit, index + 1
                result["retrain_requested"] = trigger_window == index + 1
                append_jsonl("windows", result)
                if result["psi_level"] != "ok":
                    append_jsonl("events", {"ts": time.time(), "type": f"psi_{result['psi_level']}",
                                            "window": index + 1, "psi": result["psi"]})
                state["windows"] += 1
                closed.append(result)
        state["observations"] = total
        triggered = bool(reasons)
        if triggered:
            state["retrain"] = {"status": "requested", "reason": reasons, "requested_at": time.time(),
                                "from_version": model.version, "window": trigger_window}
            append_jsonl("events", {"ts": time.time(), "type": "retrain_requested", "reason": reasons,
                                    "window": trigger_window, "model_version": model.version})
        save_state(state)
        return {"model_version": model.version, "added": len(new), "skipped_duplicates": skipped,
                "pending": total - state["windows"] * trigger["window_size"],
                "windows_closed": [{k: w[k] for k in ("window", "start", "end", "precision", "recall",
                                                       "status", "psi_level")} for w in closed],
                "consecutive": dict(state["consecutive"]), "retrain_requested": triggered,
                "retrain_reason": reasons, "retrain_window": trigger_window}


def update_retrain(version, **fields):
    with _lock:
        state = load_state()
        if state is None or state["model_version"] != version:
            return
        state["retrain"].update(fields)
        if fields.get("status") in {"rejected", "failed"}:
            # 같은 하락으로 곧바로 다시 재학습하지 않도록 연속 횟수를 비웁니다. 운영자가 원인을 확인합니다.
            state["consecutive"] = {m: 0 for m in METRICS}
        save_state(state)


def summary(limit=20):
    state = load_state()
    return {"state": state, "windows": read_jsonl("windows", limit), "events": read_jsonl("events", limit)}

