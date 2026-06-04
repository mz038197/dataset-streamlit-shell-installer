from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from dataset_streamlit_shell.ml.decision_tree import (
    RANDOM_STATE,
    prepare_heart_splits,
)

N_ESTIMATORS_LIST = [10, 50, 100, 500]
LEARNING_RATE_LIST = [0.01, 0.05, 0.1, 0.3]
SWEEP_N_ESTIMATORS_FIXED = 100
SWEEP_LEARNING_RATE_FIXED = 0.1
FINAL_N_ESTIMATORS = 500
FINAL_LEARNING_RATE = 0.1
EARLY_STOPPING_ROUNDS = 10


def training_and_validation_accuracy(
    model: XGBClassifier,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[float, float]:
    train_predictions = model.predict(x_train)
    val_predictions = model.predict(x_val)
    train_acc = float(accuracy_score(y_train, train_predictions) * 100.0)
    val_acc = float(accuracy_score(y_val, val_predictions) * 100.0)
    return train_acc, val_acc


def sweep_xgboost_hyperparam(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    param_name: str,
    values: list[float | int],
    fixed_n_estimators: int = SWEEP_N_ESTIMATORS_FIXED,
    fixed_learning_rate: float = SWEEP_LEARNING_RATE_FIXED,
    random_state: int = RANDOM_STATE,
) -> dict[str, list[float]]:
    train_scores: list[float] = []
    val_scores: list[float] = []
    for value in values:
        kwargs: dict[str, Any] = {
            "random_state": random_state,
            "verbosity": 0,
            "n_estimators": fixed_n_estimators,
            "learning_rate": fixed_learning_rate,
        }
        if param_name == "n_estimators":
            kwargs["n_estimators"] = int(value)
        elif param_name == "learning_rate":
            kwargs["learning_rate"] = float(value)
        else:
            raise ValueError(f"unsupported param_name: {param_name}")
        model = XGBClassifier(**kwargs)
        model.fit(x_train, y_train)
        train_acc, val_acc = training_and_validation_accuracy(model, x_train, y_train, x_val, y_val)
        train_scores.append(train_acc)
        val_scores.append(val_acc)
    return {
        "values": [float(value) for value in values],
        "train_accuracy": train_scores,
        "val_accuracy": val_scores,
    }


def fit_xgboost_final(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = FINAL_N_ESTIMATORS,
    learning_rate: float = FINAL_LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> XGBClassifier:
    split_index = int(len(x_train) * 0.8)
    x_fit = x_train.iloc[:split_index]
    y_fit = y_train.iloc[:split_index]
    x_eval = x_train.iloc[split_index:]
    y_eval = y_train.iloc[split_index:]
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        random_state=int(random_state),
        verbosity=0,
    )
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(x_eval, y_eval)],
        "verbose": False,
    }
    try:
        model.fit(x_fit, y_fit, early_stopping_rounds=int(early_stopping_rounds), **fit_kwargs)
    except TypeError:
        model.set_params(early_stopping_rounds=int(early_stopping_rounds))
        model.fit(x_fit, y_fit, **fit_kwargs)
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        return model
    return model


def best_iteration(model: XGBClassifier) -> int | None:
    value = getattr(model, "best_iteration", None)
    if value is None:
        return None
    return int(value)


def prepare_encoded_heart(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    return prepare_heart_splits(frame, random_state=RANDOM_STATE, train_size=0.8)


def build_xgboost_agent_context(
    *,
    train_rows: int,
    val_rows: int,
    feature_count: int,
    best_iteration_value: int | None,
    train_accuracy: float,
    val_accuracy: float,
) -> str:
    best_iter_text = "未知" if best_iteration_value is None else str(best_iteration_value)
    return (
        f"XGBoost 頁：訓練 {train_rows} 筆、驗證 {val_rows} 筆；"
        f"編碼後 features={feature_count}；best_iteration={best_iter_text}；"
        f"最終 train 準確率={train_accuracy:.2f}%；val 準確率={val_accuracy:.2f}%。"
    )
