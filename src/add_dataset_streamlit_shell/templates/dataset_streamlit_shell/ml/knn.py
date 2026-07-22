"""K-近鄰分類：訓練、鄰居查詢、agent context（pure helpers）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from dataset_streamlit_shell.ml.regression import apply_standard_scaler, create_standard_scaler

MODEL_KIND_KNN = "knn_classifier"
DEFAULT_K = 5
K_MIN = 1
K_MAX = 15
K_STEP = 2


@dataclass(frozen=True)
class KnnArtifact:
    model_kind: str
    features: list[str]
    target: str
    k: int
    training_accuracy: float
    data_source: str
    standardize: bool
    scaler: dict[str, Any] | None
    train_x: list[list[float]]
    train_y: list[int]
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def odd_k_values(*, k_min: int = K_MIN, k_max: int = K_MAX, step: int = K_STEP) -> list[int]:
    start = k_min if k_min % 2 == 1 else k_min + 1
    return list(range(start, k_max + 1, step))


def _as_feature_matrix(feature_frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(feature_frame, pd.DataFrame):
        return feature_frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.asarray(feature_frame, dtype=float)


def fit_knn_classifier(
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    k: int,
) -> KNeighborsClassifier:
    x = _as_feature_matrix(feature_frame)
    y = np.asarray(target, dtype=int).reshape(-1)
    if len(np.unique(y)) < 2:
        raise ValueError("target 至少要有兩個類別")
    if int(k) < 1:
        raise ValueError("k 必須 ≥ 1")
    if int(k) > len(y):
        raise ValueError(f"k={k} 不可大於訓練筆數 {len(y)}")
    clf = KNeighborsClassifier(n_neighbors=int(k), metric="euclidean", weights="uniform")
    clf.fit(x, y)
    return clf


def training_accuracy(
    clf: KNeighborsClassifier,
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
) -> float:
    x = _as_feature_matrix(feature_frame)
    y = np.asarray(target, dtype=int).reshape(-1)
    return float(np.mean(clf.predict(x) == y) * 100.0)


def build_knn_artifact(
    clf: KNeighborsClassifier,
    *,
    features: list[str],
    target: str,
    k: int,
    standardize: bool,
    scaler: dict[str, Any] | None,
    data_source: str,
    feature_frame: pd.DataFrame | np.ndarray,
    target_series: pd.Series | np.ndarray,
) -> KnnArtifact:
    x = _as_feature_matrix(feature_frame)
    y = np.asarray(target_series, dtype=int).reshape(-1)
    return KnnArtifact(
        model_kind=MODEL_KIND_KNN,
        features=list(features),
        target=target,
        k=int(k),
        training_accuracy=training_accuracy(clf, feature_frame, target_series),
        data_source=data_source,
        standardize=bool(standardize),
        scaler=scaler,
        train_x=x.tolist(),
        train_y=y.tolist(),
    )


def prepare_feature_matrix(
    frame: pd.DataFrame,
    features: list[str],
    *,
    standardize: bool,
    scaler: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    if not standardize:
        return numeric, None
    if scaler is None:
        scaler = create_standard_scaler(numeric, features)
    return apply_standard_scaler(numeric, scaler), scaler


def clf_from_artifact(artifact: KnnArtifact) -> KNeighborsClassifier:
    clf = KNeighborsClassifier(
        n_neighbors=int(artifact.k), metric="euclidean", weights="uniform"
    )
    clf.fit(np.asarray(artifact.train_x, dtype=float), np.asarray(artifact.train_y, dtype=int))
    return clf


def predict_class_from_artifact(artifact: KnnArtifact, frame: pd.DataFrame) -> np.ndarray:
    features = artifact.features
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    if artifact.scaler is not None:
        numeric = apply_standard_scaler(numeric, artifact.scaler)
    clf = clf_from_artifact(artifact)
    return clf.predict(numeric.to_numpy(dtype=float))


def nearest_neighbor_indices(
    artifact: KnnArtifact,
    query_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (indices, distances)；query 為原始特徵座標，會套用 artifact.scaler。"""
    query = pd.DataFrame(
        [{artifact.features[0]: query_xy[0], artifact.features[1]: query_xy[1]}]
    )
    numeric = query[artifact.features].apply(pd.to_numeric, errors="coerce")
    if artifact.scaler is not None:
        numeric = apply_standard_scaler(numeric, artifact.scaler)
    clf = clf_from_artifact(artifact)
    distances, indices = clf.kneighbors(numeric.to_numpy(dtype=float), n_neighbors=artifact.k)
    return indices[0], distances[0]


def decision_mesh_predictions(
    artifact: KnnArtifact,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_size: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = pd.DataFrame(
        {
            artifact.features[0]: xx.ravel(),
            artifact.features[1]: yy.ravel(),
        }
    )
    pred = predict_class_from_artifact(artifact, grid).reshape(xx.shape)
    return xx, yy, pred


def artifact_from_payload(payload: dict[str, Any]) -> KnnArtifact:
    return KnnArtifact(
        model_kind=str(payload["model_kind"]),
        features=[str(f) for f in payload["features"]],
        target=str(payload["target"]),
        k=int(payload["k"]),
        training_accuracy=float(payload["training_accuracy"]),
        data_source=str(payload["data_source"]),
        standardize=bool(payload.get("standardize", False)),
        scaler=payload.get("scaler"),
        train_x=[list(map(float, row)) for row in payload["train_x"]],
        train_y=[int(v) for v in payload["train_y"]],
        schema_version=int(payload.get("schema_version", 1)),
        created_at=str(payload.get("created_at", "")),
    )


def build_knn_agent_context(
    *,
    page_name: str,
    data_source: str,
    features: list[str],
    target: str,
    k: int,
    standardize: bool,
    row_count: int,
    artifact: KnnArtifact | None = None,
    note: str = "",
    expose_k: bool = True,
    prompt_train: bool = True,
) -> str:
    parts = [
        f"目前頁面：{page_name}。",
        f"資料來源：{data_source}。",
        f"可用訓練資料筆數：{row_count}。",
        "目前 features：" + "、".join(features) + "。",
        f"目前 target：{target}（二元 0/1）。",
        "距離：歐氏；投票：多數決（uniform）。",
    ]
    if expose_k:
        parts.append(f"目前 k：{k}。")
        parts.append(f"特徵標準化（Z-score）：{'開' if standardize else '關'}。")
    else:
        parts.append("本階段聚焦鄰居與多數決；k 固定且不在 UI 調整。")
        parts.append("特徵已做標準化後再算距離。")
    if artifact is None:
        if prompt_train:
            parts.append("尚未完成本組訓練；引導學生按「開始訓練」觀察決策邊界與查詢點鄰居。")
        else:
            parts.append(
                "「開始訓練」尚未解鎖；先協助完成訓練前預測，不要建議按該按鈕，也不要直接講正解。"
            )
    else:
        parts.append(f"已訓練 k={artifact.k}，訓練集正確率 {artifact.training_accuracy:.2f}%。")
        parts.append(
            f"訓練時標準化：{'有' if artifact.scaler is not None else '無'}。"
        )
    if note:
        parts.append(note)
    return "\n".join(parts)


def artifact_as_dict(artifact: KnnArtifact) -> dict[str, Any]:
    return asdict(artifact)
