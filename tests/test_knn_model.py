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
    build_knn_artifact,
    fit_knn_classifier,
    nearest_neighbor_indices,
    odd_k_values,
    predict_class_from_artifact,
    prepare_feature_matrix,
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
