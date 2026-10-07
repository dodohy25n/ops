"""같은 홀드아웃에서 후보와 현재 모델을 비교하는 배포 심사입니다."""
import numpy as np

from serving_app.monitoring.metrics import evaluate

MIN_SAMPLES = 1000
MIN_FRAUDS = 20


def validate_predictions(y, scores, tau):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("정답과 점수는 같은 길이의 비어 있지 않은 1차원 배열이어야 합니다.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("정답은 0 또는 1이어야 합니다.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("모델 점수는 유한한 0~1 값이어야 합니다.")
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau는 0과 1 사이여야 합니다.")
    return y, scores


def check_gate(y, candidate_scores, candidate_tau, policy, *, current_scores=None, current_tau=None):
    y, scores = validate_predictions(y, candidate_scores, candidate_tau)
    for key in ("r_min", "alert_cap"):
        if not np.isfinite(policy[key]) or not 0 < policy[key] <= 1:
            raise ValueError(f"{key}는 0보다 크고 1 이하여야 합니다.")
    candidate = evaluate(y, scores, candidate_tau)
    current = None
    if current_scores is not None:
        _, current_scores = validate_predictions(y, current_scores, current_tau)
        current = evaluate(y, current_scores, current_tau)
    checks = {
        "enough_samples": len(y) >= MIN_SAMPLES,
        "enough_frauds": int(y.sum()) >= MIN_FRAUDS,
        "has_normal_transactions": bool(np.any(y == 0)),
        "recall": candidate["recall"] >= policy["r_min"],
        "alert_rate": candidate["alert_rate"] <= policy["alert_cap"],
        "f2_no_regression": current is None or candidate["f2"] >= current["f2"],
    }
    return {
        "passed": all(checks.values()),
        "mode": "initial" if current is None else "replacement",
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "candidate": candidate,
        "current": current,
        "samples": int(len(y)),
        "frauds": int(y.sum()),
        "policy": {"r_min": policy["r_min"], "alert_cap": policy["alert_cap"]},
    }
