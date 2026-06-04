from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from dataset_streamlit_shell.ml.regression import GradientDescentStep, apply_standard_scaler

MODEL_KIND_LINEAR_SVM = "linear_svm"


@dataclass(frozen=True)
class LinearSvmArtifact:
    model_kind: str
    features: list[str]
    target: str
    C: float
    coef: list[float]
    intercept: float
    support_vectors: list[list[float]]
    training_accuracy: float
    data_source: str
    scaler: dict[str, Any] | None
    n_support: int
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def validate_svm_target(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    unique = set(np.unique(numeric.astype(int)))
    return unique == {-1, 1}


def fit_linear_svc(
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    C: float,
) -> SVC:
    x = _as_feature_matrix(feature_frame)
    y = np.asarray(target, dtype=float).reshape(-1)
    if len(np.unique(y)) < 2:
        raise ValueError("target must contain at least two classes")
    if not validate_svm_target(pd.Series(y)):
        raise ValueError("target must contain exactly -1 and 1")
    clf = SVC(kernel="linear", C=float(C))
    clf.fit(x, y)
    return clf


def training_accuracy(clf: SVC, feature_frame: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray) -> float:
    x = _as_feature_matrix(feature_frame)
    y = np.asarray(target, dtype=float).reshape(-1)
    predicted = clf.predict(x)
    return float(np.mean(predicted == y) * 100.0)


def build_linear_svm_artifact(
    clf: SVC,
    *,
    features: list[str],
    target: str,
    C: float,
    scaler: dict[str, Any] | None,
    data_source: str,
    feature_frame: pd.DataFrame | np.ndarray,
    target_series: pd.Series | np.ndarray,
) -> LinearSvmArtifact:
    support_vectors = clf.support_vectors_.tolist()
    return LinearSvmArtifact(
        model_kind=MODEL_KIND_LINEAR_SVM,
        features=list(features),
        target=target,
        C=float(C),
        coef=[float(value) for value in clf.coef_[0]],
        intercept=float(clf.intercept_[0]),
        support_vectors=support_vectors,
        training_accuracy=training_accuracy(clf, feature_frame, target_series),
        data_source=data_source,
        scaler=scaler,
        n_support=int(clf.support_vectors_.shape[0]),
    )


def save_svm_artifact(artifact: LinearSvmArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(artifact), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_svm_artifact(path: Path, *, expected_kind: str | None = MODEL_KIND_LINEAR_SVM) -> LinearSvmArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return artifact_from_payload(payload, expected_kind=expected_kind)


def artifact_from_payload(
    payload: dict[str, Any],
    *,
    expected_kind: str | None = MODEL_KIND_LINEAR_SVM,
) -> LinearSvmArtifact:
    model_kind = str(payload["model_kind"])
    if expected_kind is not None and model_kind != expected_kind:
        raise ValueError(f"expected model_kind {expected_kind}, got {model_kind}")
    if model_kind != MODEL_KIND_LINEAR_SVM:
        raise ValueError(f"unsupported model_kind: {model_kind}")
    return LinearSvmArtifact(
        model_kind=model_kind,
        features=[str(feature) for feature in payload["features"]],
        target=str(payload["target"]),
        C=float(payload["C"]),
        coef=[float(value) for value in payload["coef"]],
        intercept=float(payload["intercept"]),
        support_vectors=[
            [float(value) for value in row]
            for row in payload["support_vectors"]
        ],
        training_accuracy=float(payload["training_accuracy"]),
        data_source=str(payload["data_source"]),
        scaler=payload.get("scaler"),
        n_support=int(payload.get("n_support", len(payload["support_vectors"]))),
        schema_version=int(payload.get("schema_version", 1)),
        created_at=str(payload.get("created_at", datetime.now().isoformat(timespec="seconds"))),
    )


def compute_hinge_loss(
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    weights: list[float] | np.ndarray,
    intercept: float,
    *,
    C: float,
) -> float:
    x = _as_feature_matrix(feature_frame)
    y = np.asarray(target, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if x.shape[0] == 0:
        raise ValueError("feature_frame must contain at least one row")
    margins = y * (x @ w + float(intercept))
    hinge = np.maximum(0.0, 1.0 - margins)
    return float(0.5 * np.dot(w, w) + float(C) * np.mean(hinge))


def linear_svm_gradient_descent_steps(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    *,
    learning_rate: float,
    C: float,
    epochs: int,
    initial_weights: list[float] | np.ndarray | None = None,
    initial_intercept: float = 0.0,
) -> list[GradientDescentStep]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if C <= 0:
        raise ValueError("C must be greater than 0")

    x = _as_feature_matrix(feature_frame)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("feature_frame must contain at least one row")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature_frame and target must have the same row count")
    if not validate_svm_target(pd.Series(y)):
        raise ValueError("target must contain exactly -1 and 1")

    weights = (
        np.zeros(x.shape[1], dtype=float)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=float).reshape(-1)
    )
    if weights.shape != (x.shape[1],):
        raise ValueError("initial_weights must match feature count")
    intercept = float(initial_intercept)

    steps = [_svm_step_snapshot(0, x, y, weights, intercept, C)]
    for iteration in range(1, epochs + 1):
        for xi, yi in zip(x, y):
            condition = yi * (np.dot(weights, xi) + intercept)
            if condition >= 1.0:
                weights = weights - learning_rate * weights
            else:
                weights = weights - learning_rate * (weights - float(C) * yi * xi)
                intercept = intercept + learning_rate * float(C) * yi
        steps.append(_svm_step_snapshot(iteration, x, y, weights, intercept, C))
    return steps


def sample_gradient_descent_steps_for_animation(
    steps: list[GradientDescentStep],
    *,
    update_every: int,
    max_frames: int = 80,
) -> list[GradientDescentStep]:
    """Subsample steps for Streamlit animation; larger update_every yields fewer frames."""
    if not steps:
        return []
    stride = max(int(update_every), 1)
    if len(steps) > max_frames:
        cap_stride = max(1, (len(steps) + max_frames - 1) // max_frames)
        stride = max(stride, cap_stride)
    selected = steps[::stride]
    if selected[-1] is not steps[-1]:
        selected.append(steps[-1])
    return selected


def support_vector_candidates(
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    weights: list[float] | np.ndarray,
    intercept: float,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    x = _as_feature_matrix(feature_frame)
    y = np.asarray(target, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    margins = y * (x @ w + float(intercept))
    return margins <= (1.0 + float(tolerance))


def _prepare_features(frame: pd.DataFrame, features: list[str], scaler: dict[str, Any] | None) -> pd.DataFrame:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    if scaler is not None:
        numeric = apply_standard_scaler(numeric, scaler)
    return numeric


def decision_function_from_artifact(artifact: LinearSvmArtifact, frame: pd.DataFrame) -> np.ndarray:
    numeric = _prepare_features(frame, artifact.features, artifact.scaler)
    x = _as_feature_matrix(numeric)
    coef = np.asarray(artifact.coef, dtype=float).reshape(1, -1)
    return (x @ coef.T + artifact.intercept).reshape(-1)


def predict_binary_class(scores: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    return np.where(values >= 0, 1, -1).astype(int)


def predict_class_from_artifact(artifact: LinearSvmArtifact, frame: pd.DataFrame) -> np.ndarray:
    scores = decision_function_from_artifact(artifact, frame)
    return predict_binary_class(scores)


def build_svm_agent_context(
    *,
    page_name: str,
    data_source: str,
    features: list[str],
    target: str,
    C: float,
    row_count: int,
    artifact: LinearSvmArtifact | None = None,
    note: str = "",
) -> str:
    parts = [
        f"目前頁面：{page_name}。",
        f"資料來源：{data_source}。",
        f"可用訓練資料筆數：{row_count}。",
        "目前 features：" + "、".join(features) + "。",
        f"目前 target：{target}（需為 -1 / +1）。",
        f"懲罰係數 C：{C:g}。",
    ]
    if artifact is None:
        parts.append("目前尚未完成本組設定的訓練，請引導學生先按「開始訓練」觀察決策邊界與 support vectors。")
    else:
        parts.append(f"intercept：{artifact.intercept:g}。")
        weights = "、".join(
            f"{feature}={weight:g}"
            for feature, weight in zip(artifact.features, artifact.coef)
        )
        parts.append(f"coef：{weights}。")
        parts.append(f"support vectors 數量：{artifact.n_support}。")
        parts.append(f"訓練集正確率：{artifact.training_accuracy:.2f}%。")
        if artifact.scaler is not None:
            parts.append("訓練時對 features 做了 Z-score，推論需使用相同縮放。")
    if note:
        parts.append(note)
    return "\n".join(parts)


def _svm_step_snapshot(
    iteration: int,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    intercept: float,
    C: float,
) -> GradientDescentStep:
    return GradientDescentStep(
        iteration=iteration,
        weights=[float(weight) for weight in weights],
        intercept=float(intercept),
        cost=compute_hinge_loss(x, y, weights, intercept, C=C),
    )


def _as_feature_matrix(feature_frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(feature_frame, pd.DataFrame):
        return feature_frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.asarray(feature_frame, dtype=float)


__all__ = [
    "MODEL_KIND_LINEAR_SVM",
    "LinearSvmArtifact",
    "artifact_from_payload",
    "build_linear_svm_artifact",
    "build_svm_agent_context",
    "compute_hinge_loss",
    "decision_function_from_artifact",
    "fit_linear_svc",
    "linear_svm_gradient_descent_steps",
    "sample_gradient_descent_steps_for_animation",
    "load_svm_artifact",
    "predict_binary_class",
    "predict_class_from_artifact",
    "save_svm_artifact",
    "support_vector_candidates",
    "training_accuracy",
    "validate_svm_target",
]
