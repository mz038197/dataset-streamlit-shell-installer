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

from dataset_streamlit_shell.ml.decision_tree import prepare_heart_splits
from dataset_streamlit_shell.ml.random_forest import (
    compare_single_tree_vs_forest,
    fit_random_forest,
)

HEART_PATH = (
    TEMPLATE_ROOT
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "classification"
    / "heart_disease.csv"
)


def test_fit_random_forest_smoke() -> None:
    frame = pd.read_csv(HEART_PATH)
    x_train, x_val, y_train, y_val = prepare_heart_splits(frame)
    model = fit_random_forest(x_train, y_train, n_estimators=10)
    predictions = model.predict(x_val)
    assert len(predictions) == len(y_val)


def test_compare_single_tree_vs_forest_returns_both_val_accuracies() -> None:
    frame = pd.read_csv(HEART_PATH)
    x_train, x_val, y_train, y_val = prepare_heart_splits(frame)
    result = compare_single_tree_vs_forest(
        x_train,
        y_train,
        x_val,
        y_val,
        n_estimators=20,
    )
    assert "tree_val_accuracy" in result
    assert "forest_val_accuracy" in result
    assert 0.0 <= result["tree_val_accuracy"] <= 100.0
    assert 0.0 <= result["forest_val_accuracy"] <= 100.0
    assert result["n_estimators"] == 20
    assert len(result["forest"].estimators_) == 20
