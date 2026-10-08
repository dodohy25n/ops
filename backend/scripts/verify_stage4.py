"""실제 v1의 로컬 파일과 MLflow 기록이 같은 예측, 판정을 내는지 확인합니다.

실행(프로젝트 최상위): python backend/scripts/verify_stage4.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.gate_holdout import load_gate_holdout
from backend.serving_app.model_loader import _load_from_local, load_registered_version, load_run_bundle
from backend.serving_app.monitoring.deployment_gate import check_gate
from backend.serving_app.registry import MODEL_NAME, PROJECT_ROOT, active_version, configure_registry


def main():
    X, y, evaluation = load_gate_holdout()
    registration = json.loads((PROJECT_ROOT / "logs/stage4-registration.json").read_text())
    run_id = registration["run_id"]
    client = configure_registry()
    local = _load_from_local()
    if registration["version"] is not None:
        version = client.get_model_version(MODEL_NAME, registration["version"])
        if version.run_id != run_id:
            raise ValueError("Registry 버전과 검증하려는 run이 다릅니다.")
        registered = load_registered_version(client, version)
    else:
        registered = load_run_bundle(client, run_id,
                                    registration.get("model_uri", f"runs:/{run_id}/model"),
                                    "unregistered-v1")
    before, after = local.predict_scores(X), registered.predict_scores(X)
    np.testing.assert_allclose(before, after, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(before >= local.tau, after >= registered.tau)
    np.testing.assert_array_equal(local.scaler.lo, registered.scaler.lo)
    np.testing.assert_array_equal(local.scaler.hi, registered.scaler.hi)
    operating = active_version(client)
    result = {
        "version": registered.version,
        "run_id": run_id,
        "operating_version": str(operating.version) if operating else None,
        "tau": registered.tau,
        "evaluation": evaluation,
        "max_score_difference": float(np.max(np.abs(before - after))),
        "classification_matches": True,
        "scaler_matches": True,
        "gate": check_gate(y, after, registered.tau, local.settings["gate"]),
    }
    (PROJECT_ROOT / "logs/stage4-verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
