from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 線性回歸教學頁固定欄位（對應 built-in-data/regression CSV）
SIMPLE_REGRESSION_FEATURE = "城市人口_萬人"
SIMPLE_REGRESSION_TARGET = "餐廳獲利_萬美元"
MULTIPLE_REGRESSION_FEATURES = ("面積_平方英尺", "房間數", "樓層數", "屋齡_年")
MULTIPLE_REGRESSION_TARGET = "房價_千美元"

COST_J_LATEX = r"J(w,b) = \frac{1}{2m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})^2"
COST_GD_W_LATEX = r"w = w - \alpha \frac{\partial J(w,b)}{\partial w}"
COST_GD_B_LATEX = r"b = b - \alpha \frac{\partial J(w,b)}{\partial b}"
COST_DJ_DW_LATEX = (
    r"\frac{\partial J(w,b)}{\partial w} = "
    r"\frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}"
)
COST_DJ_DB_LATEX = (
    r"\frac{\partial J(w,b)}{\partial b} = "
    r"\frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})"
)


@dataclass(frozen=True)
class LinearModelArtifact:
    model_kind: str
    features: list[str]
    target: str
    weights: list[float]
    intercept: float
    scaler: dict[str, Any] | None
    training_cost: float
    data_source: str
    test_cost: float | None = None
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass(frozen=True)
class GradientDescentStep:
    iteration: int
    weights: list[float]
    intercept: float
    cost: float
    test_cost: float | None = None
    prev_weights: list[float] | None = None
    prev_intercept: float | None = None
    prev_cost: float | None = None
    dj_dw: list[float] | None = None
    dj_db: float | None = None
    delta_w: list[float] | None = None
    delta_b: float | None = None


def compute_cost_j(actual: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if actual_array.shape != prediction_array.shape:
        raise ValueError("actual and prediction must have the same shape")
    if actual_array.size == 0:
        raise ValueError("actual and prediction must not be empty")
    errors = prediction_array - actual_array
    return float(np.sum(errors**2) / (2 * actual_array.size))


def predict_with_parameters(
    feature_frame: pd.DataFrame | np.ndarray,
    weights: list[float] | np.ndarray,
    intercept: float,
) -> pd.Series:
    values = np.asarray(feature_frame, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    predictions = values @ weight_array + float(intercept)
    index = feature_frame.index if isinstance(feature_frame, pd.DataFrame) else None
    return pd.Series(predictions, index=index, name="prediction")


def gradient_descent_steps(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    *,
    learning_rate: float,
    epochs: int,
    initial_weights: list[float] | None = None,
    initial_intercept: float = 0.0,
) -> list[GradientDescentStep]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")

    x = feature_frame.to_numpy(dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("feature_frame must contain at least one row")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature_frame and target must have the same row count")

    weights = (
        np.zeros(x.shape[1], dtype=float)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=float)
    )
    if weights.shape != (x.shape[1],):
        raise ValueError("initial_weights must match feature count")
    intercept = float(initial_intercept)

    steps = [_gradient_step_snapshot(0, x, y, weights, intercept)]
    m = float(x.shape[0])
    for iteration in range(1, epochs + 1):
        prev_weights = weights.copy()
        prev_intercept = float(intercept)
        prediction = x @ weights + intercept
        prev_cost = compute_cost_j(y, prediction)
        error = prediction - y
        dj_dw = (x.T @ error) / m
        dj_db = float(np.sum(error) / m)
        delta_w = -learning_rate * dj_dw
        delta_b = -learning_rate * dj_db
        weights = weights + delta_w
        intercept = intercept + delta_b
        steps.append(
            _gradient_step_snapshot(
                iteration,
                x,
                y,
                weights,
                intercept,
                prev_weights=[float(weight) for weight in prev_weights],
                prev_intercept=prev_intercept,
                prev_cost=prev_cost,
                dj_dw=[float(value) for value in dj_dw],
                dj_db=dj_db,
                delta_w=[float(value) for value in delta_w],
                delta_b=float(delta_b),
            )
        )
    return steps


def attach_test_costs(
    steps: list[GradientDescentStep],
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> list[GradientDescentStep]:
    annotated: list[GradientDescentStep] = []
    for step in steps:
        prediction = predict_with_parameters(test_features, step.weights, step.intercept)
        annotated.append(
            replace(step, test_cost=compute_cost_j(test_target, prediction))
        )
    return annotated


def _gradient_step_snapshot(
    iteration: int,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    intercept: float,
    *,
    prev_weights: list[float] | None = None,
    prev_intercept: float | None = None,
    prev_cost: float | None = None,
    dj_dw: list[float] | None = None,
    dj_db: float | None = None,
    delta_w: list[float] | None = None,
    delta_b: float | None = None,
) -> GradientDescentStep:
    prediction = x @ weights + intercept
    return GradientDescentStep(
        iteration=iteration,
        weights=[float(weight) for weight in weights],
        intercept=float(intercept),
        cost=compute_cost_j(y, prediction),
        prev_weights=prev_weights,
        prev_intercept=prev_intercept,
        prev_cost=prev_cost,
        dj_dw=dj_dw,
        dj_db=dj_db,
        delta_w=delta_w,
        delta_b=delta_b,
    )


def create_standard_scaler(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    return create_feature_scaler(frame, features, "zscore")


def create_feature_scaler(
    frame: pd.DataFrame,
    features: list[str],
    method: str,
) -> dict[str, Any]:
    if not features:
        raise ValueError("features must not be empty")
    if method not in {"zscore", "maxdiv", "minmax", "mean"}:
        raise ValueError(f"unknown scale method: {method}")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    names = [str(feature) for feature in features]
    if method == "maxdiv":
        negatives = [
            name for name in names if bool((numeric[name].astype(float) < 0).any())
        ]
        if negatives:
            raise ValueError(
                "正規化（除以最大）要求訓練特徵 x≥0，但 "
                + "、".join(negatives)
                + " 出現負值"
            )
        xmax = numeric.max()
        invalid = [name for name in names if pd.isna(xmax[name]) or float(xmax[name]) == 0]
        if invalid:
            raise ValueError("cannot scale constant or empty columns: " + ", ".join(invalid))
        return {
            "method": "maxdiv",
            "features": names,
            "max": {name: float(xmax[name]) for name in names},
        }
    if method == "minmax":
        xmin = numeric.min()
        xmax = numeric.max()
        invalid = [
            name
            for name in names
            if pd.isna(xmin[name]) or pd.isna(xmax[name]) or float(xmax[name]) == float(xmin[name])
        ]
        if invalid:
            raise ValueError("cannot scale constant or empty columns: " + ", ".join(invalid))
        return {
            "method": "minmax",
            "features": names,
            "min": {name: float(xmin[name]) for name in names},
            "max": {name: float(xmax[name]) for name in names},
        }
    if method == "mean":
        means = numeric.mean()
        xmin = numeric.min()
        xmax = numeric.max()
        invalid = [
            name
            for name in names
            if pd.isna(means[name])
            or pd.isna(xmin[name])
            or pd.isna(xmax[name])
            or float(xmax[name]) == float(xmin[name])
        ]
        if invalid:
            raise ValueError("cannot scale constant or empty columns: " + ", ".join(invalid))
        return {
            "method": "mean",
            "features": names,
            "mean": {name: float(means[name]) for name in names},
            "min": {name: float(xmin[name]) for name in names},
            "max": {name: float(xmax[name]) for name in names},
        }
    means = numeric.mean()
    scales = numeric.std(ddof=0)
    invalid = [str(column) for column, scale in scales.items() if pd.isna(scale) or scale == 0]
    if invalid:
        raise ValueError("cannot scale constant or empty columns: " + ", ".join(invalid))
    return {
        "method": "zscore",
        "features": names,
        "mean": {str(column): float(value) for column, value in means.items()},
        "scale": {str(column): float(value) for column, value in scales.items()},
    }


def apply_standard_scaler(frame: pd.DataFrame, scaler: dict[str, Any]) -> pd.DataFrame:
    return apply_feature_scaler(frame, scaler)


def apply_feature_scaler(frame: pd.DataFrame, scaler: dict[str, Any]) -> pd.DataFrame:
    features = [str(feature) for feature in scaler["features"]]
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    result = numeric.copy()
    method = str(scaler.get("method") or "zscore")
    for feature in features:
        values = numeric[feature].astype(float)
        result[feature] = _transform_feature_values(values.to_numpy(), feature, scaler, method)
    return result


def _transform_feature_values(
    values: np.ndarray,
    feature: str,
    scaler: dict[str, Any],
    method: str,
) -> np.ndarray:
    key = str(feature)
    if method == "maxdiv":
        xmax = float(scaler["max"][key])
        if xmax == 0:
            raise ValueError(f"scaler max for {feature!r} must be non-zero")
        return values / xmax
    if method == "minmax":
        xmin = float(scaler["min"][key])
        xmax = float(scaler["max"][key])
        span = xmax - xmin
        if span == 0:
            raise ValueError(f"scaler range for {feature!r} must be non-zero")
        return (values - xmin) / span
    if method == "mean":
        mean = float(scaler["mean"][key])
        xmin = float(scaler["min"][key])
        xmax = float(scaler["max"][key])
        span = xmax - xmin
        if span == 0:
            raise ValueError(f"scaler range for {feature!r} must be non-zero")
        return (values - mean) / span
    mean = float(scaler["mean"][key])
    scale = float(scaler["scale"][key])
    if scale == 0:
        raise ValueError(f"scaler scale for {feature!r} must be non-zero")
    return (values - mean) / scale


def predict_line_on_original_x(
    raw_x: np.ndarray | pd.Series,
    *,
    weight: float,
    intercept: float,
    feature: str,
    scaler: dict[str, Any],
) -> np.ndarray:
    """將縮放空間的 w／b 映射回原始特徵橫軸上的回歸線 ŷ。"""
    values = np.asarray(raw_x, dtype=float)
    method = str(scaler.get("method") or "zscore")
    transformed = _transform_feature_values(values, str(feature), scaler, method)
    return transformed * float(weight) + float(intercept)


def save_model_artifact(artifact: LinearModelArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(artifact), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_model_artifact(path: Path) -> LinearModelArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LinearModelArtifact(
        model_kind=str(payload["model_kind"]),
        features=[str(feature) for feature in payload["features"]],
        target=str(payload["target"]),
        weights=[float(weight) for weight in payload["weights"]],
        intercept=float(payload["intercept"]),
        scaler=payload.get("scaler"),
        training_cost=float(payload["training_cost"]),
        data_source=str(payload["data_source"]),
        test_cost=(
            float(payload["test_cost"]) if payload.get("test_cost") is not None else None
        ),
        schema_version=int(payload.get("schema_version", 1)),
        created_at=str(payload["created_at"]),
    )


def predict_from_artifact(artifact: LinearModelArtifact, frame: pd.DataFrame) -> pd.Series:
    features = artifact.features
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    if artifact.scaler is not None:
        numeric = apply_feature_scaler(numeric, artifact.scaler)
    values = numeric.to_numpy(dtype=float)
    weights = np.asarray(artifact.weights, dtype=float)
    predictions = values @ weights + artifact.intercept
    return pd.Series(predictions, index=frame.index, name="prediction")


def format_prediction_formula(artifact: LinearModelArtifact) -> str:
    terms = [
        f"{weight:g} × {feature}"
        for feature, weight in zip(artifact.features, artifact.weights)
    ]
    formula = " + ".join(terms) + f" + {artifact.intercept:g}"
    if len(artifact.features) == 1:
        return f"Y = {artifact.weights[0]:g} × {artifact.features[0]} + {artifact.intercept:g}"
    return f"Y = {formula}"


def build_regression_agent_context(
    *,
    page_name: str,
    data_source: str,
    features: list[str],
    target: str,
    learning_rate: float | None,
    epochs: int | None,
    row_count: int,
    artifact: LinearModelArtifact | None = None,
    note: str = "",
    prompt_train: bool = True,
) -> str:
    parts = [
        f"目前頁面：{page_name}。",
        f"資料來源：{data_source}。",
        f"可用訓練資料筆數：{row_count}。",
        "目前 features：" + "、".join(features) + "。",
        f"目前 target：{target}。",
    ]
    if learning_rate is not None:
        parts.append(f"learning rate α：{learning_rate:g}。")
    if epochs is not None:
        parts.append(f"epoch：{epochs}。")
    if artifact is None:
        if prompt_train:
            parts.append(
                "目前尚未完成本組設定的訓練，請引導學生先按「開始訓練」觀察 Cost 與模型演進。"
            )
        else:
            parts.append(
                "目前尚未完成本組設定的訓練；「開始訓練」尚未解鎖，"
                "請先確認決策槽已齊且訓練前預測已過關，不要建議按該按鈕。"
            )
    else:
        weights = "、".join(
            f"{feature}={weight:g}" for feature, weight in zip(artifact.features, artifact.weights)
        )
        parts.extend(
            [
                f"最後 intercept/B：{artifact.intercept:g}。",
                f"最後訓練 Cost J：{artifact.training_cost:g}。",
            ]
        )
        if artifact.test_cost is not None:
            parts.append(f"最後測試 Cost J：{artifact.test_cost:g}。")
        parts.extend(
            [
                f"weights：{weights}。",
            ]
        )
        if artifact.scaler is not None:
            method = str((artifact.scaler or {}).get("method") or "zscore")
            parts.append(f"本模型使用特徵縮放 method={method}；inference 需使用保存的 scaler 統計量。")
    if note:
        parts.append(note)
    return "".join(parts)
