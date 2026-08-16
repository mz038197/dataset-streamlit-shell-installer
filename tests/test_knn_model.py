from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.knn import (  # noqa: E402
    DEFAULT_K,
    DEMO_QUERY_COUNT,
    build_knn_artifact,
    build_knn_agent_context,
    demo_query_points,
    fit_knn_classifier,
    majority_label,
    nearest_neighbor_indices,
    odd_k_values,
    predict_class_from_artifact,
    prepare_feature_matrix,
    vote_tally,
)

DEMO = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "classification"
)


def test_odd_k_values() -> None:
    assert odd_k_values() == [1, 3, 5, 7, 9, 11, 13, 15]
    assert DEFAULT_K in odd_k_values()


def test_fit_and_neighbors_on_blobs() -> None:
    df = pd.read_csv(DEMO / "knn_blobs_80.csv")
    features = ["特徵1", "特徵2"]
    matrix, scaler = prepare_feature_matrix(df, features, standardize=True)
    clf = fit_knn_classifier(matrix, df["類別"], k=5)
    artifact = build_knn_artifact(
        clf,
        features=features,
        target="類別",
        k=5,
        standardize=True,
        scaler=scaler,
        data_source="test",
        feature_frame=matrix,
        target_series=df["類別"],
    )
    assert artifact.training_accuracy > 80.0
    assert artifact.scaler is not None
    qx = float(df["特徵1"].median())
    qy = float(df["特徵2"].median())
    idx, dist = nearest_neighbor_indices(artifact, (qx, qy))
    assert len(idx) == 5
    assert len(dist) == 5
    pred = predict_class_from_artifact(
        artifact, pd.DataFrame([{"特徵1": qx, "特徵2": qy}])
    )
    assert pred[0] in (0, 1)


def test_scale_trap_without_standardize_still_fits() -> None:
    df = pd.read_csv(DEMO / "knn_scale_trap_80.csv")
    features = ["特徵1", "特徵2"]
    matrix, scaler = prepare_feature_matrix(df, features, standardize=False)
    assert scaler is None
    clf = fit_knn_classifier(matrix, df["類別"], k=5)
    artifact = build_knn_artifact(
        clf,
        features=features,
        target="類別",
        k=5,
        standardize=False,
        scaler=None,
        data_source="trap",
        feature_frame=matrix,
        target_series=df["類別"],
    )
    assert artifact.scaler is None
    assert artifact.k == 5


def test_demo_query_points_and_vote() -> None:
    import numpy as np

    pts = demo_query_points(np.array([0.0, 10.0]), np.array([0.0, 20.0]), count=DEMO_QUERY_COUNT)
    assert len(pts) == 3
    assert all(0.0 <= p[0] <= 10.0 and 0.0 <= p[1] <= 20.0 for p in pts)
    assert vote_tally([0, 0, 1]) == {0: 2, 1: 1}
    assert majority_label([0, 0, 1]) == 0
    assert majority_label([1, 1, 0, 0]) == 0  # 平票取較小標籤


def test_agent_context_uses_prediction_demo_wording() -> None:
    locked = build_knn_agent_context(
        page_name="K-近鄰分類",
        data_source="test",
        features=["特徵1", "特徵2"],
        target="類別",
        k=5,
        standardize=True,
        row_count=80,
        artifact=None,
        prompt_train=False,
    )
    assert "開始預測演示" in locked
    assert "開始訓練" not in locked
    unlocked = build_knn_agent_context(
        page_name="K-近鄰分類",
        data_source="test",
        features=["特徵1", "特徵2"],
        target="類別",
        k=5,
        standardize=True,
        row_count=80,
        artifact=None,
        prompt_train=True,
    )
    assert "下一步" in unlocked
    assert "開始訓練" not in unlocked
