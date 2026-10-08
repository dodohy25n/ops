"""저장된 v1을 재학습 없이 심사, 등록합니다. 실행: python -m serving_app.train_and_register"""
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
from backend.serving_app.registry import (
    ACTIVE_ALIAS, MODEL_NAME, PROJECT_ROOT, active_version, configure_registry, experiment_id,
)


def register_candidate(model, settings, scaler_path, X, y, evaluation, *, run_name="candidate"):
    """tau는 미리 고정해야 합니다. X, y는 학습, tau 선택에 쓰지 않은 홀드아웃입니다."""
    settings = copy.deepcopy(settings)
    settings["features"] = FEATURES
    if settings["beta"] != 2:
        raise ValueError("이번 실습의 배포 기준은 F2(beta=2)입니다.")
    policy = json.loads((PROJECT_ROOT / "serving_app/models/thresholds.json").read_text())["gate"]
    candidate = LoadedModel(model, FraudScaler.load(scaler_path), settings, "candidate")
    scores = candidate.predict_scores(X)
    client = configure_registry()
    previous = active_version(client)
    current = load_registered_version(client, previous) if previous is not None else None
    current_scores = current.predict_scores(X) if current else None
    if current is not None:
        if not (np.array_equal(current.scaler.lo, candidate.scaler.lo)
                and np.array_equal(current.scaler.hi, candidate.scaler.hi)):
            raise ValueError("후보와 운영 모델은 같은 고정 스케일러를 사용해야 합니다.")
    gate = check_gate(y, scores, candidate.tau, policy,
                      current_scores=current_scores, current_tau=current.tau if current else None)
    result = {"promoted": False, "version": None,
              "previous_version": str(previous.version) if previous else None, "gate": gate}
    with mlflow.start_run(experiment_id=experiment_id(client), run_name=run_name) as run:
        result["run_id"] = run.info.run_id
        mlflow.log_params({"beta": 2, "tau": candidate.tau, "sequence_length": 20,
                           "r_min": policy["r_min"], "alert_cap": policy["alert_cap"],
                           "evaluation": evaluation["name"], "gate_mode": gate["mode"]})
        mlflow.log_metrics({f"candidate_{k}": v for k, v in gate["candidate"].items()})
        if gate["current"]:
            mlflow.log_metrics({f"current_{k}": v for k, v in gate["current"].items()})
        mlflow.log_dict(evaluation, "evaluation.json")
        mlflow.log_dict(gate, "gate.json")
        mlflow.log_dict(settings, "bundle/settings.json")
        mlflow.log_artifact(str(scaler_path), "bundle")
        requirements = [f"{p}=={package_version(p)}" for p in ("tensorflow", "keras", "numpy")]
        info = mlflow.tensorflow.log_model(
            model, name="model", signature=infer_signature(X[:2], scores[:2, None]),
            input_example=X[:2], pip_requirements=requirements,
        )
        result["model_uri"] = info.model_uri
        mlflow.set_tags({"gate_passed": str(gate["passed"]).lower(), "model_role": "candidate"})
        registered = mlflow.register_model(info.model_uri, MODEL_NAME)
        result["version"] = str(registered.version)
        client.set_model_version_tag(
            MODEL_NAME, registered.version, "gate_passed", str(gate["passed"]).lower()
        )
        if gate["passed"]:
            reloaded = mlflow.tensorflow.load_model(info.model_uri, keras_model_kwargs={"compile": False})
            sample = X[:64]
            np.testing.assert_allclose(
                reloaded(sample, training=False).numpy(), model(sample, training=False).numpy(),
                atol=1e-6, rtol=1e-5,
            )
            latest = active_version(client)
            if (str(latest.version) if latest else None) != result["previous_version"]:
                raise RuntimeError("평가 중 운영 모델이 바뀌었습니다. 새 운영 모델과 다시 비교해야 합니다.")
            client.set_registered_model_alias(MODEL_NAME, ACTIVE_ALIAS, registered.version)
            result["promoted"] = True
            mlflow.set_tag("model_role", "promoted")
        mlflow.log_dict(result, "registration.json")
    return result


def main():
    from data.gate_holdout import load_gate_holdout

    X, y, evaluation = load_gate_holdout()
    local = _load_from_local()
    print(f"[1] 7월 독립 평가: {len(y):,}건, 이상거래 {int(y.sum()):,}건", flush=True)
    result = register_candidate(
        local.keras_model, local.settings, PROJECT_ROOT / "serving_app/models/scaler.pkl",
        X, y, evaluation, run_name="v1-initial-registration",
    )
    output = PROJECT_ROOT / "logs/stage4-registration.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    m = result["gate"]["candidate"]
    print(f"[2] Recall {m['recall']:.4f} / 경보 {m['alert_rate']:.4%} / F2 {m['f2']:.4f}")
    print(f"[3] {'승격 완료' if result['promoted'] else '배포 보류'}: {result}")
    if not result["promoted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
