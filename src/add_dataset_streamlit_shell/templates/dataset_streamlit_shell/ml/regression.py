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


@dataclass(frozen=True)
class GradientDescentStep:
    iteration: int
    weights: list[float]
    intercept: float
    cost: float


def compute_cost_j(actual: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if actual_array.shape != prediction_array.shape:
        raise ValueError("actual and prediction must have the same shape")
    if actual_array.size == 0:
        raise ValueError("actual and prediction must not be empty")
    errors = prediction_array - actual_array
    return float(np.sum(errors**2) / (2 * actual_array.size))


def predict_with_parameters(
    feature_frame: pd.DataFrame | np.ndarray,
    weights: list[float] | np.ndarray,
    intercept: float,
) -> pd.Series:
    values = np.asarray(feature_frame, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    predictions = values @ weight_array + float(intercept)
    index = feature_frame.index if isinstance(feature_frame, pd.DataFrame) else None
    return pd.Series(predictions, index=index, name="prediction")


def gradient_descent_steps(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    *,
    learning_rate: float,
    epochs: int,
    initial_weights: list[float] | None = None,
    initial_intercept: float = 0.0,
) -> list[GradientDescentStep]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")

    x = feature_frame.to_numpy(dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("feature_frame must contain at least one row")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature_frame and target must have the same row count")

    weights = (
        np.zeros(x.shape[1], dtype=float)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=float)
    )
    if weights.shape != (x.shape[1],):
        raise ValueError("initial_weights must match feature count")
    intercept = float(initial_intercept)

    steps = [_gradient_step_snapshot(0, x, y, weights, intercept)]
    m = float(x.shape[0])
    for iteration in range(1, epochs + 1):
        prediction = x @ weights + intercept
        error = prediction - y
        dj_dw = (x.T @ error) / m
        dj_db = float(np.sum(error) / m)
        weights = weights - learning_rate * dj_dw
        intercept = intercept - learning_rate * dj_db
        steps.append(_gradient_step_snapshot(iteration, x, y, weights, intercept))
    return steps


def _gradient_step_snapshot(
    iteration: int,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    intercept: float,
) -> GradientDescentStep:
    prediction = x @ weights + intercept
    return GradientDescentStep(
        iteration=iteration,
        weights=[float(weight) for weight in weights],
        intercept=float(intercept),
        cost=compute_cost_j(y, prediction),
    )


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


def format_prediction_formula(artifact: LinearModelArtifact) -> str:
    terms = [
        f"{weight:g} × {feature}"
        for feature, weight in zip(artifact.features, artifact.weights)
    ]
    formula = " + ".join(terms) + f" + {artifact.intercept:g}"
    if len(artifact.features) == 1:
        return f"Y = {artifact.weights[0]:g} × {artifact.features[0]} + {artifact.intercept:g}"
    return f"Y = {formula}"


def build_regression_agent_context(
    *,
    page_name: str,
    data_source: str,
    features: list[str],
    target: str,
    learning_rate: float | None,
    epochs: int | None,
    row_count: int,
    artifact: LinearModelArtifact | None = None,
    note: str = "",
) -> str:
    parts = [
        f"目前頁面：{page_name}。",
        f"資料來源：{data_source}。",
        f"可用訓練資料筆數：{row_count}。",
        "目前 features：" + "、".join(features) + "。",
        f"目前 target：{target}。",
    ]
    if learning_rate is not None:
        parts.append(f"learning rate α：{learning_rate:g}。")
    if epochs is not None:
        parts.append(f"epoch：{epochs}。")
    if artifact is None:
        parts.append("目前尚未完成本組設定的訓練，請引導學生先按「開始訓練」觀察 Cost 與模型演進。")
    else:
        weights = "、".join(
            f"{feature}={weight:g}" for feature, weight in zip(artifact.features, artifact.weights)
        )
        parts.extend(
            [
                f"最後 intercept/B：{artifact.intercept:g}。",
                f"最後 Cost J：{artifact.training_cost:g}。",
                f"weights：{weights}。",
            ]
        )
        if artifact.scaler is not None:
            parts.append("本模型使用 Z-score 特徵縮放；inference 需使用 JSON 內保存的 mean/scale。")
    if note:
        parts.append(note)
    return "".join(parts)
