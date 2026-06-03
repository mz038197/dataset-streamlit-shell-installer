from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset_streamlit_shell.ml.regression import (
    GradientDescentStep,
    apply_standard_scaler,
    create_standard_scaler,
)

MODEL_KIND_LOGISTIC = "logistic_regression"
MODEL_KIND_REGULARIZED = "regularized_logistic_regression"

CONTOUR_U_MIN = -1.0
CONTOUR_U_MAX = 1.5
DEFAULT_MAP_DEGREE = 6


@dataclass(frozen=True)
class LogisticModelArtifact:
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
class RegularizedLogisticModelArtifact:
    model_kind: str
    base_features: list[str]
    mapped_features: list[str]
    target: str
    weights: list[float]
    intercept: float
    map_degree: int
    lambda_: float
    training_cost: float
    data_source: str
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


ClassificationArtifact = LogisticModelArtifact | RegularizedLogisticModelArtifact


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z_array = np.asarray(z, dtype=float)
    clipped = np.clip(z_array, -500, 500)
    result = 1.0 / (1.0 + np.exp(-clipped))
    if np.ndim(z_array) == 0:
        return float(result)
    return result


def log_1pexp(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    mask = x <= 20
    out[mask] = np.log(1.0 + np.exp(x[mask]))
    out[~mask] = x[~mask]
    return out


def compute_cost_logistic(
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    weights: list[float] | np.ndarray,
    intercept: float,
) -> float:
    x = _as_feature_matrix(feature_frame)
    y = np.asarray(target, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature_frame and target must have the same row count")
    if x.shape[0] == 0:
        raise ValueError("feature_frame must contain at least one row")

    z = x @ w + float(intercept)
    cost_terms = -(y * z) + log_1pexp(z)
    return float(np.sum(cost_terms) / x.shape[0])


def compute_cost_logistic_reg(
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    weights: list[float] | np.ndarray,
    intercept: float,
    lambda_: float,
) -> float:
    logistic_cost = compute_cost_logistic(feature_frame, target, weights, intercept)
    w = np.asarray(weights, dtype=float).reshape(-1)
    m = _as_feature_matrix(feature_frame).shape[0]
    reg_cost = float(lambda_) / (2.0 * m) * float(np.sum(w**2))
    return logistic_cost + reg_cost


def predict_proba(
    feature_frame: pd.DataFrame | np.ndarray,
    weights: list[float] | np.ndarray,
    intercept: float,
) -> pd.Series:
    x = _as_feature_matrix(feature_frame)
    w = np.asarray(weights, dtype=float).reshape(-1)
    values = sigmoid(x @ w + float(intercept))
    index = feature_frame.index if isinstance(feature_frame, pd.DataFrame) else None
    return pd.Series(values, index=index, name="probability")


def predict_class_from_proba(probability: pd.Series | np.ndarray, threshold: float) -> pd.Series:
    prob_array = np.asarray(probability, dtype=float)
    classes = (prob_array >= float(threshold)).astype(int)
    index = probability.index if isinstance(probability, pd.Series) else None
    return pd.Series(classes, index=index, name="predicted_class")


def logistic_gradient_descent_steps(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    *,
    learning_rate: float,
    epochs: int,
    initial_weights: list[float] | np.ndarray | None = None,
    initial_intercept: float = 0.0,
    lambda_: float = 0.0,
    regularized: bool = False,
) -> list[GradientDescentStep]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")

    x = feature_frame.to_numpy(dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("feature_frame must contain at least one row")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature_frame and target must have the same row count")

    weights = (
        np.zeros(x.shape[1], dtype=float)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=float).reshape(-1)
    )
    if weights.shape != (x.shape[1],):
        raise ValueError("initial_weights must match feature count")
    intercept = float(initial_intercept)
    m = float(x.shape[0])

    steps = [_logistic_step_snapshot(0, x, y, weights, intercept, lambda_, regularized)]
    for iteration in range(1, epochs + 1):
        prediction = sigmoid(x @ weights + intercept)
        error = prediction - y
        dj_dw = (x.T @ error) / m
        if regularized:
            dj_dw = dj_dw + (float(lambda_) / m) * weights
        dj_db = float(np.sum(error) / m)
        weights = weights - learning_rate * dj_dw
        intercept = intercept - learning_rate * dj_db
        steps.append(
            _logistic_step_snapshot(iteration, x, y, weights, intercept, lambda_, regularized)
        )
    return steps


def _logistic_step_snapshot(
    iteration: int,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    intercept: float,
    lambda_: float,
    regularized: bool,
) -> GradientDescentStep:
    if regularized:
        cost = compute_cost_logistic_reg(x, y, weights, intercept, lambda_)
    else:
        cost = compute_cost_logistic(x, y, weights, intercept)
    return GradientDescentStep(
        iteration=iteration,
        weights=[float(weight) for weight in weights],
        intercept=float(intercept),
        cost=float(cost),
    )


def map_feature(
    frame: pd.DataFrame,
    base_features: list[str],
    *,
    degree: int = DEFAULT_MAP_DEGREE,
) -> tuple[pd.DataFrame, list[str]]:
    if len(base_features) != 2:
        raise ValueError("map_feature requires exactly two base features")
    x1 = frame[base_features[0]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x2 = frame[base_features[1]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    columns: list[str] = []
    values: list[np.ndarray] = []
    for total_degree in range(1, degree + 1):
        for power_x2 in range(total_degree + 1):
            power_x1 = total_degree - power_x2
            name = f"{base_features[0]}^{power_x1}*{base_features[1]}^{power_x2}"
            columns.append(name)
            values.append((x1**power_x1) * (x2**power_x2))
    mapped = pd.DataFrame(np.column_stack(values), columns=columns, index=frame.index)
    return mapped, columns


def map_feature_row(base_features: list[str], values: dict[str, float], *, degree: int) -> pd.DataFrame:
    frame = pd.DataFrame([{feature: float(values[feature]) for feature in base_features}])
    mapped, _ = map_feature(frame, base_features, degree=degree)
    return mapped


def validate_binary_target(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    unique = set(np.unique(numeric.astype(int)))
    return unique.issubset({0, 1}) and len(unique) >= 1


def training_accuracy(
    actual: pd.Series,
    probability: pd.Series,
    threshold: float,
) -> float:
    predicted = predict_class_from_proba(probability, threshold)
    actual_array = pd.to_numeric(actual, errors="coerce").astype(int)
    return float(np.mean(predicted.to_numpy() == actual_array.to_numpy()) * 100.0)


def save_classification_artifact(artifact: ClassificationArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(artifact), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_classification_artifact(
    path: Path,
    *,
    expected_kind: str | None = None,
) -> ClassificationArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return artifact_from_payload(payload, expected_kind=expected_kind)


def artifact_from_payload(
    payload: dict[str, Any],
    *,
    expected_kind: str | None = None,
) -> ClassificationArtifact:
    model_kind = str(payload["model_kind"])
    if expected_kind is not None and model_kind != expected_kind:
        raise ValueError(f"expected model_kind {expected_kind}, got {model_kind}")
    if model_kind == MODEL_KIND_LOGISTIC:
        return LogisticModelArtifact(
            model_kind=model_kind,
            features=[str(feature) for feature in payload["features"]],
            target=str(payload["target"]),
            weights=[float(weight) for weight in payload["weights"]],
            intercept=float(payload["intercept"]),
            scaler=payload.get("scaler"),
            training_cost=float(payload["training_cost"]),
            data_source=str(payload["data_source"]),
            schema_version=int(payload.get("schema_version", 1)),
            created_at=str(payload.get("created_at", datetime.now().isoformat(timespec="seconds"))),
        )
    if model_kind == MODEL_KIND_REGULARIZED:
        return RegularizedLogisticModelArtifact(
            model_kind=model_kind,
            base_features=[str(feature) for feature in payload["base_features"]],
            mapped_features=[str(feature) for feature in payload["mapped_features"]],
            target=str(payload["target"]),
            weights=[float(weight) for weight in payload["weights"]],
            intercept=float(payload["intercept"]),
            map_degree=int(payload["map_degree"]),
            lambda_=float(payload["lambda_"]),
            training_cost=float(payload["training_cost"]),
            data_source=str(payload["data_source"]),
            schema_version=int(payload.get("schema_version", 1)),
            created_at=str(payload.get("created_at", datetime.now().isoformat(timespec="seconds"))),
        )
    raise ValueError(f"unsupported model_kind: {model_kind}")


def predict_proba_from_logistic_artifact(
    artifact: LogisticModelArtifact,
    frame: pd.DataFrame,
) -> pd.Series:
    numeric = frame[artifact.features].apply(pd.to_numeric, errors="coerce")
    if artifact.scaler is not None:
        numeric = apply_standard_scaler(numeric, artifact.scaler)
    return predict_proba(numeric, artifact.weights, artifact.intercept)


def predict_proba_from_regularized_artifact(
    artifact: RegularizedLogisticModelArtifact,
    frame: pd.DataFrame,
) -> pd.Series:
    mapped, _ = map_feature(frame, artifact.base_features, degree=artifact.map_degree)
    return predict_proba(mapped[artifact.mapped_features], artifact.weights, artifact.intercept)


def build_classification_agent_context(
    *,
    page_name: str,
    data_source: str,
    features: list[str],
    target: str,
    learning_rate: float | None,
    epochs: int | None,
    row_count: int,
    artifact: ClassificationArtifact | None = None,
    lambda_: float | None = None,
    map_degree: int | None = None,
    threshold: float | None = None,
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
    if lambda_ is not None:
        parts.append(f"正則化 λ：{lambda_:g}。")
    if map_degree is not None:
        parts.append(f"特徵映射 degree：{map_degree}。")
    if artifact is None:
        parts.append("目前尚未完成本組設定的訓練，請引導學生先按「開始訓練」觀察 Cost 與模型演進。")
    else:
        parts.append(f"最後 intercept/B：{artifact.intercept:g}。")
        parts.append(f"最後 Cost J：{artifact.training_cost:g}。")
        if isinstance(artifact, LogisticModelArtifact):
            weights = "、".join(
                f"{feature}={weight:g}"
                for feature, weight in zip(artifact.features, artifact.weights)
            )
            parts.append(f"weights：{weights}。")
            if artifact.scaler is not None:
                parts.append("本模型使用 Z-score 特徵縮放。")
        else:
            parts.append(
                "base features："
                + "、".join(artifact.base_features)
                + f"；映射後 {len(artifact.mapped_features)} 維。"
            )
    if threshold is not None:
        parts.append(f"目前 UI 分類 threshold：{threshold:g}（僅影響類別預測，不參與訓練）。")
    if note:
        parts.append(note)
    return "".join(parts)


def _as_feature_matrix(feature_frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(feature_frame, pd.DataFrame):
        return feature_frame.to_numpy(dtype=float)
    return np.asarray(feature_frame, dtype=float)
