"""非監督式分群：內建資料載入、K-Means、Ward 階層切群（pure helpers）。"""

from __future__ import annotations

from dataclasses import dataclass, replace
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

BEAT_INIT = "init"
BEAT_ASSIGN = "assign"
BEAT_UPDATE = "update"
KMEANS_MAX_ROUNDS = 15
KMEANS_INIT_RANDOM_STATE = 42


@dataclass(frozen=True)
class KMeansEvolution:
    locked_k: int
    initial_centers: tuple[tuple[float, ...], ...]
    max_rounds: int = KMEANS_MAX_ROUNDS
    centers: tuple[tuple[float, ...], ...] | None = None
    previous_centers: tuple[tuple[float, ...], ...] | None = None
    labels: tuple[int, ...] | None = None
    beat: str | None = None
    round_index: int = 0
    sse: float | None = None
    empty_cluster: bool = False
    converged: bool = False
    hit_round_cap: bool = False


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


def _as_centers(arr: np.ndarray) -> tuple[tuple[float, ...], ...]:
    rows = np.asarray(arr, dtype=float)
    return tuple(tuple(float(v) for v in row) for row in rows)


def _as_labels(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(arr, dtype=int).reshape(-1))


def sample_initial_centers(
    x: np.ndarray,
    *,
    n_clusters: int,
    random_state: int = KMEANS_INIT_RANDOM_STATE,
) -> np.ndarray:
    points = np.asarray(x, dtype=float)
    rng = np.random.RandomState(int(random_state))
    k = int(n_clusters)
    n = len(points)
    if k > n:
        raise ValueError(f"K={k} 不可大於筆數 {n}")
    idx = rng.choice(n, size=k, replace=False)
    return points[idx].copy()


def _assign_labels(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    delta = x[:, None, :] - centers[None, :, :]
    return np.linalg.norm(delta, axis=2).argmin(axis=1)


def _cluster_sse(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    return float(np.sum((x - centers[labels]) ** 2))


def _has_empty_cluster(labels: np.ndarray, n_clusters: int) -> bool:
    present = {int(v) for v in np.unique(labels)}
    return any(i not in present for i in range(int(n_clusters)))


def _move_centers(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    moved = np.array(centers, dtype=float, copy=True)
    for i in range(len(centers)):
        mask = labels == i
        if np.any(mask):
            moved[i] = x[mask].mean(axis=0)
    return moved


def new_kmeans_evolution(
    *,
    locked_k: int,
    initial_centers: np.ndarray,
    max_rounds: int = KMEANS_MAX_ROUNDS,
) -> KMeansEvolution:
    return KMeansEvolution(
        locked_k=int(locked_k),
        initial_centers=_as_centers(initial_centers),
        max_rounds=int(max_rounds),
    )


def can_advance_kmeans(evo: KMeansEvolution) -> bool:
    return not evo.converged and not evo.hit_round_cap


def advance_kmeans_evolution(evo: KMeansEvolution, x: np.ndarray) -> KMeansEvolution:
    if not can_advance_kmeans(evo):
        return evo
    points = np.asarray(x, dtype=float)
    if evo.beat is None:
        return replace(
            evo,
            beat=BEAT_INIT,
            centers=evo.initial_centers,
            previous_centers=None,
            labels=None,
            sse=None,
        )
    if evo.beat in (BEAT_INIT, BEAT_UPDATE):
        centers = np.asarray(evo.centers, dtype=float)
        labels = _assign_labels(points, centers)
        prev = evo.labels
        converged = prev is not None and _as_labels(labels) == prev
        return replace(
            evo,
            beat=BEAT_ASSIGN,
            labels=_as_labels(labels),
            sse=_cluster_sse(points, labels, centers),
            empty_cluster=_has_empty_cluster(labels, evo.locked_k),
            previous_centers=None,
            round_index=evo.round_index + 1,
            converged=converged,
        )
    centers = np.asarray(evo.centers, dtype=float)
    labels = np.asarray(evo.labels, dtype=int)
    moved = _move_centers(points, labels, centers)
    return replace(
        evo,
        beat=BEAT_UPDATE,
        previous_centers=evo.centers,
        centers=_as_centers(moved),
        sse=_cluster_sse(points, labels, moved),
        empty_cluster=_has_empty_cluster(labels, evo.locked_k),
        hit_round_cap=evo.round_index >= evo.max_rounds,
    )


def kmeans_evolution_status(evo: KMeansEvolution) -> str:
    if evo.beat is None:
        return "已鎖定 K。按「下一步」放出初始群中心。"
    extras: list[str] = []
    if evo.sse is not None:
        extras.append(f"群內距離平方和 {evo.sse:.1f}")
    if evo.empty_cluster:
        extras.append("有一群沒有點")
    if evo.converged:
        extras.append("已收斂")
    if evo.hit_round_cap:
        extras.append("已達輪數上限")
    suffix = "。" + "。".join(extras) if extras else ""
    if evo.beat == BEAT_INIT:
        return "初始群中心已放上，點尚未分群。"
    if evo.beat == BEAT_ASSIGN:
        return "① 每個點歸到最近的群中心" + suffix
    if evo.beat == BEAT_UPDATE:
        return "② 群中心移到該群平均" + suffix
    return ""


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
