"""
측정, 배포 게이트, 드리프트 감시가 함께 쓰는 지표 계산 모듈입니다.

같은 지표를 곳곳에서 따로 계산하면 측정할 때와 운영할 때 기준이 어긋날 수 있어서
계산식을 이 파일 한 곳에 둡니다. 컨테이너 이미지를 가볍게 유지하려고 numpy만 씁니다.
"""
import numpy as np

BETA = 2.0
PSI_EPS = 1e-4


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, fn


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp, fp, fn = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """정밀도-재현율 곡선 아래 면적을 average precision 방식으로 계산합니다."""
    order = np.argsort(-scores, kind="stable")
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    total = y.sum()
    ends = np.r_[scores[order][1:] != scores[order][:-1], True]
    recall_increments = np.diff(np.r_[0, tp[ends]])
    return float(np.sum(precision[ends] * recall_increments) / total) if total else 0.0


def evaluate(y_true: np.ndarray, scores: np.ndarray, tau: float) -> dict:
    y_pred = (scores >= tau).astype(int)
    precision, recall = precision_recall(y_true, y_pred)
    return {
        "tau": round(float(tau), 2),
        "f2": f_beta(precision, recall),
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc(y_true, scores),
        "alert_rate": float(y_pred.mean()),
    }


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """0.01 간격으로 기준점을 바꿔 가며 F2가 가장 높은 값을 고릅니다."""
    taus = np.round(np.arange(0.01, 1.0, 0.01), 2)
    best_tau, best_f2 = 0.5, -1.0
    for tau in taus:
        precision, recall = precision_recall(y_true, (scores >= tau).astype(int))
        score = f_beta(precision, recall)
        if score > best_f2:
            best_tau, best_f2 = float(tau), score
    return best_tau


def psi_reference(values: np.ndarray, n_bins: int = 10, categorical: bool = False) -> dict:
    """기준 분포를 저장합니다. 수치형은 10분위 구간, 범주형은 값별 비율입니다.

    0이 대부분인 0/1 피처를 분위수로 나누면 경계가 0 하나만 남아 0과 1이 같은 구간에 들어가므로,
    범주형은 구간 대신 값마다 비율을 셉니다.
    """
    values = np.asarray(values)
    if categorical:
        categories, counts = np.unique(np.round(values, 6), return_counts=True)
        return {"categories": categories.tolist(), "ratios": (counts / counts.sum()).tolist()}
    edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, n_bins - 1)))
    counts = np.bincount(np.digitize(values, edges), minlength=len(edges) + 1)
    return {"edges": edges.tolist(), "ratios": (counts / counts.sum()).tolist()}


def _bin_counts(reference: dict, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    expected = np.array(reference["ratios"])
    if "categories" in reference:
        categories = np.array(reference["categories"])
        index = np.searchsorted(categories, np.round(values, 6))
        known = (index < len(categories)) & (categories[np.minimum(index, len(categories) - 1)] == np.round(values, 6))
        # 기준 기간에 없던 값은 마지막 칸에 모읍니다.
        counts = np.bincount(np.where(known, index, len(categories)), minlength=len(categories) + 1)
        return counts, np.r_[expected, 0.0]
    edges = np.array(reference["edges"])
    return np.bincount(np.digitize(values, edges), minlength=len(edges) + 1), expected


def psi(reference: dict, values: np.ndarray) -> float:
    counts, expected = _bin_counts(reference, values)
    actual = counts / counts.sum()
    expected = np.clip(expected, PSI_EPS, None)
    actual = np.clip(actual, PSI_EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
