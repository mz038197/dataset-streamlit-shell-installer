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

from dataset_streamlit_shell.ml.regression import (
    COST_DJ_DB_LATEX,
    COST_DJ_DW_LATEX,
    COST_GD_B_LATEX,
    COST_GD_W_LATEX,
    COST_J_LATEX,
    MULTIPLE_REGRESSION_FEATURES,
    MULTIPLE_REGRESSION_TARGET,
    SIMPLE_REGRESSION_FEATURE,
    SIMPLE_REGRESSION_TARGET,
)

BUILTIN = (
    TEMPLATE_ROOT
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "regression"
)


def test_restaurant_csv_matches_fixed_simple_columns() -> None:
    frame = pd.read_csv(BUILTIN / "restaurant_profit.csv")
    assert list(frame.columns) == [SIMPLE_REGRESSION_FEATURE, SIMPLE_REGRESSION_TARGET]


def test_house_prices_csv_matches_fixed_multiple_columns() -> None:
    frame = pd.read_csv(BUILTIN / "house_prices.csv")
    assert list(frame.columns) == [
        *MULTIPLE_REGRESSION_FEATURES,
        MULTIPLE_REGRESSION_TARGET,
    ]


def test_cost_formula_latex_uses_zero_based_sum_index() -> None:
    assert r"\sum_{i=0}^{m-1}" in COST_J_LATEX
    assert r"\sum_{i=0}^{m-1}" in COST_DJ_DW_LATEX
    assert r"\sum_{i=0}^{m-1}" in COST_DJ_DB_LATEX
    assert r"\alpha" in COST_GD_W_LATEX
    assert r"\alpha" in COST_GD_B_LATEX
