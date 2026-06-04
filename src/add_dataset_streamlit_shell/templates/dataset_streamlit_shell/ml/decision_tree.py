from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 55
TRAIN_SIZE = 0.8

CAT_FEATURES = ["耳朵形狀", "臉型", "鬍鬚"]
CAT_TARGET = "是否為貓"

HEART_TARGET = "心臟病"
HEART_CAT_COLUMNS = ["性別", "胸痛類型", "靜息心電圖", "運動心絞痛", "ST斜率"]

CRITERION_CHOICES = {
    "Gini": "gini",
    "Entropy": "entropy",
}


def compute_entropy(y: np.ndarray | pd.Series) -> float:
    values = np.asarray(y, dtype=float).reshape(-1)
    if len(values) == 0:
        return 0.0
    positive_rate = float(np.mean(values == 1))
    if positive_rate in (0.0, 1.0):
        return 0.0
    return float(
        -positive_rate * np.log2(positive_rate)
        - (1.0 - positive_rate) * np.log2(1.0 - positive_rate)
    )


def split_dataset(
    feature_matrix: np.ndarray,
    node_indices: list[int],
    feature_index: int,
) -> tuple[list[int], list[int]]:
    left_indices: list[int] = []
    right_indices: list[int] = []
    for index in node_indices:
        if feature_matrix[index, feature_index] == 1:
            left_indices.append(index)
        else:
            right_indices.append(index)
    return left_indices, right_indices


def compute_information_gain(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    node_indices: list[int],
    feature_index: int,
) -> float:
    left_indices, right_indices = split_dataset(feature_matrix, node_indices, feature_index)
    node_target = target[node_indices]
    left_target = target[left_indices]
    right_target = target[right_indices]
    node_entropy = compute_entropy(node_target)
    if len(node_target) == 0:
        return 0.0
    weight_left = len(left_target) / len(node_target)
    weight_right = len(right_target) / len(node_target)
    weighted_entropy = weight_left * compute_entropy(left_target) + weight_right * compute_entropy(
        right_target
    )
    return float(node_entropy - weighted_entropy)


def information_gain_table(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    labels = frame[target].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    node_indices = list(range(len(frame)))
    rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(features):
        gain = compute_information_gain(matrix, labels, node_indices, feature_index)
        rows.append({"欄位": feature_name, "資訊增益": gain})
    return pd.DataFrame(rows).sort_values("資訊增益", ascending=False, ignore_index=True)


def fit_decision_tree(
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    max_depth: int,
    criterion: str,
) -> DecisionTreeClassifier:
    x = np.asarray(feature_frame, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if criterion not in {"gini", "entropy"}:
        raise ValueError("criterion must be 'gini' or 'entropy'")
    if len(np.unique(y)) < 2:
        raise ValueError("target must contain at least two classes")
    model = DecisionTreeClassifier(max_depth=int(max_depth), criterion=criterion)
    model.fit(x, y)
    return model


def training_accuracy(
    model: DecisionTreeClassifier,
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
) -> float:
    x = np.asarray(feature_frame, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    predicted = model.predict(x)
    return float(np.mean(predicted == y) * 100.0)


def one_hot_encode_heart(frame: pd.DataFrame) -> pd.DataFrame:
    if HEART_TARGET not in frame.columns:
        raise ValueError(f"missing target column: {HEART_TARGET}")
    working = frame.copy()
    missing_cats = [column for column in HEART_CAT_COLUMNS if column not in working.columns]
    if missing_cats:
        raise ValueError("missing categorical columns: " + ", ".join(missing_cats))
    return pd.get_dummies(
        working,
        prefix=HEART_CAT_COLUMNS,
        columns=HEART_CAT_COLUMNS,
    )


def prepare_heart_splits(
    frame: pd.DataFrame,
    *,
    random_state: int = RANDOM_STATE,
    train_size: float = TRAIN_SIZE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    encoded = one_hot_encode_heart(frame)
    features = [column for column in encoded.columns if column != HEART_TARGET]
    x = encoded[features]
    y = encoded[HEART_TARGET]
    return train_test_split(
        x,
        y,
        train_size=train_size,
        random_state=random_state,
    )


def build_decision_tree_agent_context(
    *,
    features: list[str],
    target: str,
    max_depth: int,
    criterion_label: str,
    training_accuracy_pct: float,
    row_count: int,
) -> str:
    return (
        f"決策樹概念頁：{row_count} 筆樣本；features={', '.join(features)}；"
        f"target={target}；max_depth={max_depth}；criterion={criterion_label}；"
        f"訓練集正確率={training_accuracy_pct:.2f}%。"
    )
