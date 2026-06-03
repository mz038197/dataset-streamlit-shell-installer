from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from dataset_streamlit_shell.ml.classification import validate_binary_target
from dataset_streamlit_shell.ml.regression import apply_standard_scaler

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


def predict_class_from_artifact(artifact: LinearSvmArtifact, frame: pd.DataFrame) -> np.ndarray:
    scores = decision_function_from_artifact(artifact, frame)
    return np.where(scores >= 0, 1, 0).astype(int)


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
        f"目前 target：{target}。",
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
    "decision_function_from_artifact",
    "fit_linear_svc",
    "load_svm_artifact",
    "predict_class_from_artifact",
    "save_svm_artifact",
    "training_accuracy",
    "validate_binary_target",
]
