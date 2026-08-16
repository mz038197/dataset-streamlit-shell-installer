"""K-近鄰分類：訓練、鄰居查詢、agent context（pure helpers）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
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


DEMO_QUERY_COUNT = 3
BEAT_APPEAR = 1
BEAT_RANK = 2
BEAT_LINES = 3
BEAT_VOTE = 4
RANK_TABLE_LIMIT = 10


def demo_query_points(
    x: np.ndarray,
    y: np.ndarray,
    *,
    count: int = DEMO_QUERY_COUNT,
) -> list[tuple[float, float]]:
    """依訓練散點範圍產出固定相對位置的示範查詢點（新點，非訓練列）。"""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    # 相對 bbox 的固定比例：偏邊界／偏一側，課堂可對照
    fracs = (
        (0.28, 0.55),
        (0.52, 0.78),
        (0.74, 0.32),
        (0.40, 0.40),
        (0.62, 0.58),
    )
    pts: list[tuple[float, float]] = []
    for i in range(max(1, int(count))):
        fx, fy = fracs[i % len(fracs)]
        pts.append((x_min + fx * (x_max - x_min), y_min + fy * (y_max - y_min)))
    return pts


def vote_tally(labels: list[int] | np.ndarray) -> dict[int, int]:
    arr = np.asarray(labels, dtype=int).reshape(-1)
    return {int(lab): int(np.sum(arr == lab)) for lab in sorted(np.unique(arr).tolist())}


def majority_label(labels: list[int] | np.ndarray) -> int:
    """多數決；平票時取標籤數值較小者（與 sklearn uniform + 奇數 k 實務對齊即可）。"""
    tally = vote_tally(labels)
    if not tally:
        raise ValueError("labels 不可為空")
    best = max(tally.values())
    winners = [lab for lab, n in tally.items() if n == best]
    return int(min(winners))


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
        f"目前 k：{k}。",
        "特徵標準化（Z-score）固定開啟。",
    ]
    if artifact is None:
        if prompt_train:
            parts.append(
                "尚未完成本組預測演示；引導學生按「開始預測演示」後用「下一步」走四拍預測過程演進。"
            )
        else:
            parts.append(
                "「開始預測演示」尚未解鎖；先協助完成訓練前預測，不要建議按該按鈕，也不要直接講正解。"
            )
    else:
        parts.append(
            f"已鎖定鄰居池 k={artifact.k}，訓練集正確率 {artifact.training_accuracy:.2f}%。"
        )
        parts.append("用「下一步」推進：查詢點出現 → 距離排序 → k 條連線 → 多數決。")
    if note:
        parts.append(note)
    return "\n".join(parts)


def artifact_as_dict(artifact: KnnArtifact) -> dict[str, Any]:
    return asdict(artifact)


@dataclass(frozen=True)
class RankRow:
    rank: int
    index: int
    distance: float
    label: int


@dataclass(frozen=True)
class KnnEvolution:
    locked_k: int
    demo_remaining: tuple[tuple[float, float], ...]
    labeled: tuple[tuple[float, float, int], ...]
    active_xy: tuple[float, float] | None
    beat: int | None
    distances_by_index: tuple[float, ...] | None
    order_near_to_far: tuple[int, ...] | None


def new_evolution(
    demos: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    locked_k: int,
) -> KnnEvolution:
    return KnnEvolution(
        locked_k=int(locked_k),
        demo_remaining=tuple((float(x), float(y)) for x, y in demos),
        labeled=(),
        active_xy=None,
        beat=None,
        distances_by_index=None,
        order_near_to_far=None,
    )


def shows_boundary(evo: KnnEvolution) -> bool:
    return len(evo.labeled) >= DEMO_QUERY_COUNT


def can_click(evo: KnnEvolution) -> bool:
    return shows_boundary(evo) and evo.beat in (None, BEAT_VOTE)


def can_advance(evo: KnnEvolution) -> bool:
    if evo.beat is None:
        return bool(evo.demo_remaining)
    if evo.beat in (BEAT_APPEAR, BEAT_RANK, BEAT_LINES):
        return True
    if evo.beat == BEAT_VOTE:
        return bool(evo.demo_remaining)
    return False


def neighbor_indices_for_view(evo: KnnEvolution) -> list[int] | None:
    if evo.beat not in (BEAT_LINES, BEAT_VOTE) or evo.order_near_to_far is None:
        return None
    return list(evo.order_near_to_far[: evo.locked_k])


def _query_vector(artifact: KnnArtifact, query_xy: tuple[float, float]) -> np.ndarray:
    query = pd.DataFrame(
        [{artifact.features[0]: query_xy[0], artifact.features[1]: query_xy[1]}]
    )
    numeric = query[artifact.features].apply(pd.to_numeric, errors="coerce")
    if artifact.scaler is not None:
        numeric = apply_standard_scaler(numeric, artifact.scaler)
    return numeric.to_numpy(dtype=float).reshape(-1)


def query_distance_profile(
    artifact: KnnArtifact, query_xy: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(artifact.train_x, dtype=float)
    query = _query_vector(artifact, query_xy)
    distances = np.linalg.norm(train - query, axis=1)
    order = np.argsort(distances, kind="mergesort")
    return distances, order


def distance_marker_styles(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(distances, dtype=float).reshape(-1)
    lo = float(np.min(values)) if values.size else 0.0
    hi = float(np.max(values)) if values.size else 0.0
    if values.size == 0 or hi - lo < 1e-12:
        n = int(values.size)
        return np.full(n, 10.0), np.full(n, 0.8)
    t = (values - lo) / (hi - lo)
    sizes = 16.0 - t * 10.0
    opacities = 1.0 - t * 0.75
    return sizes, opacities


def rank_table_rows(
    evo: KnnEvolution,
    train_y: list[int] | np.ndarray,
    *,
    limit: int = RANK_TABLE_LIMIT,
) -> tuple[RankRow, ...]:
    if evo.order_near_to_far is None or evo.distances_by_index is None:
        return ()
    labels = np.asarray(train_y, dtype=int).reshape(-1)
    rows: list[RankRow] = []
    for rank, idx in enumerate(evo.order_near_to_far[: int(limit)], start=1):
        rows.append(
            RankRow(
                rank=rank,
                index=int(idx),
                distance=float(evo.distances_by_index[idx]),
                label=int(labels[idx]),
            )
        )
    return tuple(rows)


def advance_evolution(evo: KnnEvolution, artifact: KnnArtifact) -> KnnEvolution:
    if not can_advance(evo):
        return evo
    if evo.beat is None or evo.beat == BEAT_VOTE:
        nxt = evo.demo_remaining[0]
        return replace(
            evo,
            demo_remaining=evo.demo_remaining[1:],
            active_xy=nxt,
            beat=BEAT_APPEAR,
            distances_by_index=None,
            order_near_to_far=None,
        )
    if evo.beat == BEAT_APPEAR:
        if evo.active_xy is None:
            return evo
        distances, order = query_distance_profile(artifact, evo.active_xy)
        return replace(
            evo,
            beat=BEAT_RANK,
            distances_by_index=tuple(float(v) for v in distances),
            order_near_to_far=tuple(int(i) for i in order),
        )
    if evo.beat == BEAT_RANK:
        return replace(evo, beat=BEAT_LINES)
    if evo.active_xy is None:
        return evo
    qx, qy = evo.active_xy
    pred = int(
        predict_class_from_artifact(
            artifact, pd.DataFrame([{artifact.features[0]: qx, artifact.features[1]: qy}])
        )[0]
    )
    return replace(
        evo,
        beat=BEAT_VOTE,
        labeled=evo.labeled + ((qx, qy, pred),),
    )


def click_query(evo: KnnEvolution, xy: tuple[float, float]) -> KnnEvolution:
    if not can_click(evo):
        return evo
    return replace(
        evo,
        active_xy=(float(xy[0]), float(xy[1])),
        beat=BEAT_APPEAR,
        distances_by_index=None,
        order_near_to_far=None,
    )


def accepted_chart_click(
    *,
    click_enabled: bool,
    selected_xy: tuple[float, float] | None,
    active_xy: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """儀式進行中或重複同一點的 selection 不當作新查詢。"""
    if not click_enabled or selected_xy is None:
        return None
    if active_xy is not None:
        if abs(active_xy[0] - selected_xy[0]) < 1e-9 and abs(active_xy[1] - selected_xy[1]) < 1e-9:
            return None
    return (float(selected_xy[0]), float(selected_xy[1]))


def evolution_status_caption(evo: KnnEvolution) -> str:
    if evo.beat is None:
        return "鄰居池已就緒。按「下一步」讓第 1 個查詢點進來。"
    labels = {
        BEAT_APPEAR: "① 新查詢點進來",
        BEAT_RANK: "② 依距離排序（越近越大／越實）",
        BEAT_LINES: "③ 取最近 k 個鄰居",
        BEAT_VOTE: "④ 多數決分類",
    }
    text = labels.get(evo.beat, "")
    if evo.beat == BEAT_VOTE and can_advance(evo):
        return f"{text}。再按「下一步」進入下一筆。"
    if evo.beat == BEAT_VOTE and can_click(evo):
        return f"{text}。示範 3 筆已完成，可在圖上點一下加演。"
    return text
