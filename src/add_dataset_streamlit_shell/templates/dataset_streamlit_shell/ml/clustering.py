"""非監督式分群：內建資料載入、K-Means、Ward 階層切群（pure helpers）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

SHELL_ROOT = Path(__file__).resolve().parents[1]
CLUSTERING_DEMO_DIR = SHELL_ROOT / "built-in-data" / "clustering"
CLUSTERING_BLOBS_PATH = CLUSTERING_DEMO_DIR / "clustering_blobs_90.csv"

CLUSTERING_FEATURES = ["特徵1", "特徵2"]
CLUSTERING_TRUTH_LABEL = "真實群"
TRUE_CLUSTER_COUNT = 3
DEFAULT_N_CLUSTERS = 3
N_CLUSTERS_MIN = 2
N_CLUSTERS_MAX = 8

SOURCE_LABEL = "內建範例資料：兩特徵三群分群（尺度相近，90 筆）"


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    n_clusters: int


def load_builtin_clustering_frame(
    path: Path | None = None,
) -> pd.DataFrame:
    csv_path = path or CLUSTERING_BLOBS_PATH
    frame = pd.read_csv(csv_path)
    expected = [*CLUSTERING_FEATURES, CLUSTERING_TRUTH_LABEL]
    if list(frame.columns) != expected:
        raise ValueError(f"預期欄位 {expected}，實際為 {list(frame.columns)}")
    return frame


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[CLUSTERING_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def fit_kmeans(
    frame: pd.DataFrame,
    *,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> KMeansResult:
    x = feature_matrix(frame)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(x)
    return KMeansResult(
        labels=np.asarray(labels, dtype=int),
        centers=np.asarray(model.cluster_centers_, dtype=float),
        n_clusters=int(n_clusters),
    )


def ward_linkage(frame: pd.DataFrame) -> np.ndarray:
    x = feature_matrix(frame)
    return linkage(x, method="ward")


def cut_ward_clusters(
    linkage_matrix: np.ndarray,
    *,
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> np.ndarray:
    """切成 n_clusters 群；回傳 0-based 標籤（與 K-Means 一致）。"""
    raw = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    labels = np.asarray(raw, dtype=int) - 1
    return labels
