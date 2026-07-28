from __future__ import annotations

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

from dataset_streamlit_shell.ml.clustering import (  # noqa: E402
    CLUSTERING_FEATURES,
    CLUSTERING_TRUTH_LABEL,
    DEFAULT_N_CLUSTERS,
    TRUE_CLUSTER_COUNT,
    cut_ward_clusters,
    fit_kmeans,
    load_builtin_clustering_frame,
    ward_linkage,
)

BUILTIN = (
    TEMPLATE_ROOT
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "clustering"
)


def test_builtin_csv_contract() -> None:
    frame = load_builtin_clustering_frame()
    assert list(frame.columns) == [*CLUSTERING_FEATURES, CLUSTERING_TRUTH_LABEL]
    assert CLUSTERING_FEATURES == ["特徵1", "特徵2"]
    assert CLUSTERING_TRUTH_LABEL == "真實群"
    assert TRUE_CLUSTER_COUNT == 3
    assert DEFAULT_N_CLUSTERS == 3
    assert set(frame[CLUSTERING_TRUTH_LABEL].unique()) == {0, 1, 2}
    assert len(frame) >= 60
    assert (BUILTIN / "clustering_blobs_90.csv").is_file()


def test_fit_kmeans_default_recovers_three_clusters() -> None:
    frame = load_builtin_clustering_frame()
    result = fit_kmeans(frame, n_clusters=DEFAULT_N_CLUSTERS)
    assert set(result.labels) == {0, 1, 2}
    assert result.centers.shape == (3, 2)
    assert len(result.labels) == len(frame)


def test_ward_linkage_and_cut_default() -> None:
    frame = load_builtin_clustering_frame()
    linkage = ward_linkage(frame)
    assert linkage.shape[0] == len(frame) - 1
    labels = cut_ward_clusters(linkage, n_clusters=DEFAULT_N_CLUSTERS)
    assert len(labels) == len(frame)
    assert set(np.unique(labels).tolist()) == {0, 1, 2}
