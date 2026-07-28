from __future__ import annotations

import sys
from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.classification import (  # noqa: E402
    COST_DJ_DB_LOGISTIC_LATEX,
    COST_DJ_DW_LOGISTIC_LATEX,
    COST_DJ_DW_LOGISTIC_REG_LATEX,
    COST_GD_B_LOGISTIC_LATEX,
    COST_GD_W_LOGISTIC_LATEX,
    COST_J_LOGISTIC_LATEX,
    COST_J_LOGISTIC_REG_LATEX,
)


def test_logistic_cost_latex_matches_linear_regression_index_style() -> None:
    assert r"\sum_{i=0}^{m-1}" in COST_J_LOGISTIC_LATEX
    assert r"f_{w,b}(x^{(i)})" in COST_J_LOGISTIC_LATEX
    assert r"\mathrm{sigmoid}" not in COST_J_LOGISTIC_LATEX
    assert r"\alpha" in COST_GD_W_LOGISTIC_LATEX
    assert r"\alpha" in COST_GD_B_LOGISTIC_LATEX
    assert r"\sum_{i=0}^{m-1}" in COST_DJ_DW_LOGISTIC_LATEX
    assert r"\sum_{i=0}^{m-1}" in COST_DJ_DB_LOGISTIC_LATEX
    assert r"f_{w,b}(x^{(i)})" in COST_DJ_DW_LOGISTIC_LATEX


def test_regularized_cost_latex_includes_lambda_terms() -> None:
    assert r"\lambda" in COST_J_LOGISTIC_REG_LATEX
    assert r"\sum_{i=0}^{m-1}" in COST_J_LOGISTIC_REG_LATEX
    assert r"f_{w,b}(x^{(i)})" in COST_J_LOGISTIC_REG_LATEX
    assert r"\lambda" in COST_DJ_DW_LOGISTIC_REG_LATEX
    assert r"\frac{\lambda}{m}" in COST_DJ_DW_LOGISTIC_REG_LATEX or r"\frac{\lambda}{m}" in COST_DJ_DW_LOGISTIC_REG_LATEX.replace(
        " ", ""
    )
