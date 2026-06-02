from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearModelArtifact:
    model_kind: str
    features: list[str]
    target: str
    weights: list[float]
    intercept: float
    scaler: dict[str, Any] | None
    training_cost: float
    data_source: str
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def compute_cost_j(actual: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if actual_array.shape != prediction_array.shape:
        raise ValueError("actual and prediction must have the same shape")
    if actual_array.size == 0:
        raise ValueError("actual and prediction must not be empty")
    errors = prediction_array - actual_array
    return float(np.sum(errors**2) / (2 * actual_array.size))


def create_standard_scaler(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    if not features:
        raise ValueError("features must not be empty")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    means = numeric.mean()
    scales = numeric.std(ddof=0)
    invalid = [str(column) for column, scale in scales.items() if pd.isna(scale) or scale == 0]
    if invalid:
        raise ValueError("cannot scale constant or empty columns: " + ", ".join(invalid))
    return {
        "method": "zscore",
        "features": [str(feature) for feature in features],
        "mean": {str(column): float(value) for column, value in means.items()},
        "scale": {str(column): float(value) for column, value in scales.items()},
    }


def apply_standard_scaler(frame: pd.DataFrame, scaler: dict[str, Any]) -> pd.DataFrame:
    features = [str(feature) for feature in scaler["features"]]
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    result = numeric.copy()
    for feature in features:
        result[feature] = (numeric[feature] - float(scaler["mean"][feature])) / float(
            scaler["scale"][feature]
        )
    return result


def save_model_artifact(artifact: LinearModelArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(artifact), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_model_artifact(path: Path) -> LinearModelArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LinearModelArtifact(
        model_kind=str(payload["model_kind"]),
        features=[str(feature) for feature in payload["features"]],
        target=str(payload["target"]),
        weights=[float(weight) for weight in payload["weights"]],
        intercept=float(payload["intercept"]),
        scaler=payload.get("scaler"),
        training_cost=float(payload["training_cost"]),
        data_source=str(payload["data_source"]),
        schema_version=int(payload.get("schema_version", 1)),
        created_at=str(payload["created_at"]),
    )


def predict_from_artifact(artifact: LinearModelArtifact, frame: pd.DataFrame) -> pd.Series:
    features = artifact.features
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    if artifact.scaler is not None:
        numeric = apply_standard_scaler(numeric, artifact.scaler)
    values = numeric.to_numpy(dtype=float)
    weights = np.asarray(artifact.weights, dtype=float)
    predictions = values @ weights + artifact.intercept
    return pd.Series(predictions, index=frame.index, name="prediction")
