from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from dataset_streamlit_shell.ml.decision_tree import RANDOM_STATE

DEFAULT_FOREST_N_ESTIMATORS = 50
BASELINE_FOREST_N_ESTIMATORS = 50


def fit_random_forest(
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    n_estimators: int,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    if int(n_estimators) < 1:
        raise ValueError("n_estimators must be >= 1")
    y = target if isinstance(target, pd.Series) else np.asarray(target, dtype=float).reshape(-1)
    if len(np.unique(np.asarray(y))) < 2:
        raise ValueError("target must contain at least two classes")
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
    )
    model.fit(feature_frame, y)
    return model


def _accuracy_pct(
    model: DecisionTreeClassifier | RandomForestClassifier,
    feature_frame: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
) -> float:
    y = target if isinstance(target, pd.Series) else np.asarray(target, dtype=float).reshape(-1)
    return float(accuracy_score(y, model.predict(feature_frame)) * 100.0)


def compare_single_tree_vs_forest(
    x_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    x_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    *,
    n_estimators: int,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    tree = DecisionTreeClassifier(random_state=int(random_state))
    tree.fit(x_train, y_train)
    forest = fit_random_forest(
        x_train,
        y_train,
        n_estimators=int(n_estimators),
        random_state=int(random_state),
    )
    return {
        "tree": tree,
        "forest": forest,
        "n_estimators": int(n_estimators),
        "tree_train_accuracy": _accuracy_pct(tree, x_train, y_train),
        "tree_val_accuracy": _accuracy_pct(tree, x_val, y_val),
        "forest_train_accuracy": _accuracy_pct(forest, x_train, y_train),
        "forest_val_accuracy": _accuracy_pct(forest, x_val, y_val),
    }


def forest_validation_baseline(
    x_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    x_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    *,
    n_estimators: int = BASELINE_FOREST_N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> float:
    result = compare_single_tree_vs_forest(
        x_train,
        y_train,
        x_val,
        y_val,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    return float(result["forest_val_accuracy"])
