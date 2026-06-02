from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.regression_model import (
    LinearModelArtifact,
    apply_standard_scaler,
    compute_cost_j,
    create_standard_scaler,
    load_model_artifact,
    predict_from_artifact,
    save_model_artifact,
)


def test_compute_cost_j_uses_course_formula() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    prediction = np.array([12.0, 17.0, 36.0])

    assert compute_cost_j(actual, prediction) == 49 / 6


def test_standard_scaler_round_trip_uses_training_statistics() -> None:
    frame = pd.DataFrame(
        {
            "面積_平方英尺": [1000.0, 1500.0, 2000.0],
            "房間數": [2.0, 3.0, 4.0],
        }
    )

    scaler = create_standard_scaler(frame, ["面積_平方英尺", "房間數"])
    scaled = apply_standard_scaler(frame, scaler)

    assert scaler["method"] == "zscore"
    assert scaler["features"] == ["面積_平方英尺", "房間數"]
    np.testing.assert_allclose(scaled.mean().to_numpy(), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(scaled.std(ddof=0).to_numpy(), [1.0, 1.0], atol=1e-12)


def test_model_artifact_json_can_be_loaded_for_inference(tmp_path: Path) -> None:
    artifact = LinearModelArtifact(
        model_kind="multiple_linear_regression",
        features=["面積_平方英尺", "房間數"],
        target="房價_千美元",
        weights=[10.0, 5.0],
        intercept=100.0,
        scaler={
            "method": "zscore",
            "features": ["面積_平方英尺", "房間數"],
            "mean": {"面積_平方英尺": 1000.0, "房間數": 2.0},
            "scale": {"面積_平方英尺": 100.0, "房間數": 1.0},
        },
        training_cost=12.5,
        data_source="內建範例資料：房價",
    )
    path = tmp_path / "model.json"

    save_model_artifact(artifact, path)
    loaded = load_model_artifact(path)
    prediction = predict_from_artifact(
        loaded,
        pd.DataFrame({"面積_平方英尺": [1100.0], "房間數": [3.0]}),
    )

    assert prediction.tolist() == [115.0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["features"] == ["面積_平方英尺", "房間數"]
