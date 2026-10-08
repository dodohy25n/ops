"""v1을 기준 운영 모델로 등록하고, 재학습 후보는 게이트를 거쳐 교체합니다.

실행(프로젝트 최상위): python -m backend.serving_app.train_and_register
"""
import copy
import json
from importlib.metadata import version as package_version

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.models import infer_signature

from data.features import FEATURES, FraudScaler
from backend.serving_app.model_loader import LoadedModel, _load_from_local, load_registered_version
from backend.serving_app.monitoring.deployment_gate import check_gate
from backend.serving_app.config import MODEL_DIR, PROJECT_ROOT
from backend.serving_app.registry import (
    ACTIVE_ALIAS, MODEL_NAME, active_version, configure_registry, experiment_id,
)


def _prepare(model, settings, scaler_path):
    settings = copy.deepcopy(settings)
    settings["features"] = FEATURES
    if settings["beta"] != 2:
        raise ValueError("이번 실습의 배포 기준은 F2(beta=2)입니다.")
    return settings, LoadedModel(model, FraudScaler.load(scaler_path), settings, "candidate")


def _log_and_register(client, model, settings, scaler_path, X, scores):
    mlflow.log_dict(settings, "bundle/settings.json")
    mlflow.log_artifact(str(scaler_path), "bundle")
    requirements = [f"{p}=={package_version(p)}" for p in ("tensorflow", "keras", "numpy")]
    info = mlflow.tensorflow.log_model(
        model, name="model", signature=infer_signature(X[:2], scores[:2, None]),
        input_example=X[:2], pip_requirements=requirements,
    )
    registered = mlflow.register_model(info.model_uri, MODEL_NAME)
    return info.model_uri, registered.version


def _assert_reload_matches(model_uri, model, X):
    reloaded = mlflow.tensorflow.load_model(model_uri, keras_model_kwargs={"compile": False})
    sample = X[:64]
    np.testing.assert_allclose(
        reloaded(sample, training=False).numpy(), model(sample, training=False).numpy(),
        atol=1e-6, rtol=1e-5,
    )


def register_baseline(model, settings, scaler_path, X, *, run_name="v1-baseline"):
    """운영 모델이 없을 때 v1을 게이트 없이 기준 운영 모델로 지정합니다. X는 서명과 저장 확인용입니다."""
    settings, baseline = _prepare(model, settings, scaler_path)
    client = configure_registry()
    if active_version(client) is not None:
        raise RuntimeError("이미 운영 모델이 있습니다. 새 모델은 register_candidate로 게이트를 거쳐야 합니다.")
    scores = baseline.predict_scores(X)
    with mlflow.start_run(experiment_id=experiment_id(client), run_name=run_name) as run:
        mlflow.log_params({"beta": 2, "tau": baseline.tau, "sequence_length": 20})
        mlflow.log_metrics({f"valid_{k}": v for k, v in settings["valid"].items()})
        model_uri, version = _log_and_register(client, model, settings, scaler_path, X, scores)
        _assert_reload_matches(model_uri, model, X)
        client.set_registered_model_alias(MODEL_NAME, ACTIVE_ALIAS, version)
        mlflow.set_tag("model_role", "baseline")
        result = {"promoted": True, "version": str(version), "run_id": run.info.run_id,
                  "model_uri": model_uri, "tau": baseline.tau}
        mlflow.log_dict(result, "registration.json")
    return result


def register_candidate(model, settings, scaler_path, X, y, evaluation, *, run_name="candidate"):
    """재학습한 후보를 현재 운영 모델과 같은 홀드아웃에서 비교합니다.

    tau는 미리 고정해야 합니다. X, y는 학습, tau 선택에 쓰지 않은 홀드아웃입니다.
    """
    settings, candidate = _prepare(model, settings, scaler_path)
    policy = json.loads((MODEL_DIR / "thresholds.json").read_text())["gate"]
    client = configure_registry()
    previous = active_version(client)
    if previous is None:
        raise RuntimeError("운영 모델이 없습니다. 먼저 v1을 기준 운영 모델로 등록하세요.")
    current = load_registered_version(client, previous)
    if not (np.array_equal(current.scaler.lo, candidate.scaler.lo)
            and np.array_equal(current.scaler.hi, candidate.scaler.hi)):
        raise ValueError("후보와 운영 모델은 같은 고정 스케일러를 사용해야 합니다.")
    scores = candidate.predict_scores(X)
    gate = check_gate(y, scores, candidate.tau, policy,
                      current_scores=current.predict_scores(X), current_tau=current.tau)
    result = {"promoted": False, "version": None, "previous_version": str(previous.version), "gate": gate}
    with mlflow.start_run(experiment_id=experiment_id(client), run_name=run_name) as run:
        result["run_id"] = run.info.run_id
        mlflow.log_params({"beta": 2, "tau": candidate.tau, "sequence_length": 20,
                           "r_min": policy["r_min"], "alert_cap": policy["alert_cap"],
                           "evaluation": evaluation["name"]})
        mlflow.log_metrics({f"candidate_{k}": v for k, v in gate["candidate"].items()})
        mlflow.log_metrics({f"current_{k}": v for k, v in gate["current"].items()})
        mlflow.log_dict(evaluation, "evaluation.json")
        mlflow.log_dict(gate, "gate.json")
        model_uri, version = _log_and_register(client, model, settings, scaler_path, X, scores)
        result["model_uri"], result["version"] = model_uri, str(version)
        mlflow.set_tags({"gate_passed": str(gate["passed"]).lower(), "model_role": "candidate"})
        client.set_model_version_tag(MODEL_NAME, version, "gate_passed", str(gate["passed"]).lower())
        if gate["passed"]:
            _assert_reload_matches(model_uri, model, X)
            latest = active_version(client)
            if (str(latest.version) if latest else None) != result["previous_version"]:
                raise RuntimeError("평가 중 운영 모델이 바뀌었습니다. 새 운영 모델과 다시 비교해야 합니다.")
            client.set_registered_model_alias(MODEL_NAME, ACTIVE_ALIAS, version)
            result["promoted"] = True
            mlflow.set_tag("model_role", "promoted")
        mlflow.log_dict(result, "registration.json")
    return result


def main():
    valid = np.load(PROJECT_ROOT.parent / "data/processed/valid.npz")
    X = valid["X"][:1024]
    local = _load_from_local()
    result = register_baseline(local.keras_model, local.settings, MODEL_DIR / "scaler.pkl", X)
    output = PROJECT_ROOT / "logs/baseline-registration.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"v1 기준 운영 모델 등록: 버전 {result['version']}, τ {result['tau']}, 별칭 {ACTIVE_ALIAS}")


if __name__ == "__main__":
    main()
