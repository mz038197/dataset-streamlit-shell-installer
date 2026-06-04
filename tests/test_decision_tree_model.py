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

from dataset_streamlit_shell.ml.decision_tree import (
    CAT_FEATURES,
    CAT_TARGET,
    compute_entropy,
    fit_decision_tree,
    information_gain_table,
    one_hot_encode_heart,
    prepare_heart_splits,
)

CAT_PATH = (
    TEMPLATE_ROOT
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "classification"
    / "cat_toy_10.csv"
)
HEART_PATH = (
    TEMPLATE_ROOT
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "classification"
    / "heart_disease.csv"
)


def test_compute_entropy_pure_and_mixed() -> None:
    assert compute_entropy([0, 0, 0]) == 0.0
    assert compute_entropy([1, 1, 1]) == 0.0
    assert compute_entropy([0, 1]) > 0.0


def test_information_gain_table_best_feature_is_ear_shape() -> None:
    frame = pd.read_csv(CAT_PATH)
    table = information_gain_table(frame, list(CAT_FEATURES), CAT_TARGET)
    assert table.iloc[0]["欄位"] == "耳朵形狀"
    assert float(table.iloc[0]["資訊增益"]) > 0.2


def test_fit_decision_tree_both_criteria() -> None:
    frame = pd.read_csv(CAT_PATH)
    x = frame[CAT_FEATURES]
    y = frame[CAT_TARGET]
    for criterion in ("gini", "entropy"):
        model = fit_decision_tree(x, y, max_depth=1, criterion=criterion)
        assert model.tree_.max_depth <= 1


def test_one_hot_encode_heart_has_no_object_columns() -> None:
    frame = pd.read_csv(HEART_PATH)
    encoded = one_hot_encode_heart(frame)
    assert "心臟病" in encoded.columns
    assert encoded.select_dtypes(include="object").empty


def test_prepare_heart_splits_sizes() -> None:
    frame = pd.read_csv(HEART_PATH)
    x_train, x_val, y_train, y_val = prepare_heart_splits(frame)
    total = len(x_train) + len(x_val)
    assert total == len(frame)
    assert len(x_train) > len(x_val)
