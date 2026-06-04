from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

pytest.importorskip("xgboost")

from dataset_streamlit_shell.ml.xgboost_model import (
    LEARNING_RATE_LIST,
    N_ESTIMATORS_LIST,
    fit_xgboost_final,
    prepare_encoded_heart,
    sweep_xgboost_hyperparam,
)

HEART_PATH = (
    TEMPLATE_ROOT
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "classification"
    / "heart_disease.csv"
)


def test_sweep_lengths() -> None:
    frame = pd.read_csv(HEART_PATH)
    x_train, x_val, y_train, y_val = prepare_encoded_heart(frame)
    n_result = sweep_xgboost_hyperparam(
        x_train,
        y_train,
        x_val,
        y_val,
        param_name="n_estimators",
        values=N_ESTIMATORS_LIST,
    )
    assert len(n_result["train_accuracy"]) == len(N_ESTIMATORS_LIST)
    assert len(n_result["val_accuracy"]) == len(N_ESTIMATORS_LIST)
    lr_result = sweep_xgboost_hyperparam(
        x_train,
        y_train,
        x_val,
        y_val,
        param_name="learning_rate",
        values=LEARNING_RATE_LIST,
    )
    assert len(lr_result["values"]) == len(LEARNING_RATE_LIST)


def test_fit_xgboost_final_smoke() -> None:
    frame = pd.read_csv(HEART_PATH)
    x_train, x_val, y_train, y_val = prepare_encoded_heart(frame)
    model = fit_xgboost_final(x_train, y_train, n_estimators=50, learning_rate=0.1)
    predictions = model.predict(x_val)
    assert len(predictions) == len(y_val)
