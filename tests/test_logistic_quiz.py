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

from dataset_streamlit_shell.ui.logistic_quiz import (  # noqa: E402
    COST_CORRECT,
    COST_FOLLOWS_THRESHOLD,
    LAMBDA_CORRECT,
    LAMBDA_FIT_HARDER,
    LEARNING_STAGES,
    MAP_CORRECT,
    MAP_FEWER_FEATURES,
    PLEASE_SELECT,
    SIGMOID_CORRECT,
    SIGMOID_PM_LABEL,
    STAGE_BOUNDARY,
    STAGE_POLY_LAMBDA,
    both_boundary_quiz_correct,
    both_poly_quiz_correct,
    is_cost_correct,
    is_lambda_correct,
    is_map_correct,
    is_sigmoid_correct,
)


def test_learning_stages_order() -> None:
    assert LEARNING_STAGES == (STAGE_BOUNDARY, STAGE_POLY_LAMBDA)


def test_boundary_quiz_gate() -> None:
    assert both_boundary_quiz_correct(SIGMOID_CORRECT, COST_CORRECT)
    assert not both_boundary_quiz_correct(PLEASE_SELECT, COST_CORRECT)
    assert not both_boundary_quiz_correct(SIGMOID_CORRECT, PLEASE_SELECT)
    assert not both_boundary_quiz_correct(SIGMOID_PM_LABEL, COST_CORRECT)
    assert not both_boundary_quiz_correct(SIGMOID_CORRECT, COST_FOLLOWS_THRESHOLD)
    assert is_sigmoid_correct(SIGMOID_CORRECT)
    assert is_cost_correct(COST_CORRECT)


def test_poly_quiz_gate() -> None:
    assert both_poly_quiz_correct(MAP_CORRECT, LAMBDA_CORRECT)
    assert not both_poly_quiz_correct(PLEASE_SELECT, LAMBDA_CORRECT)
    assert not both_poly_quiz_correct(MAP_CORRECT, PLEASE_SELECT)
    assert not both_poly_quiz_correct(MAP_FEWER_FEATURES, LAMBDA_CORRECT)
    assert not both_poly_quiz_correct(MAP_CORRECT, LAMBDA_FIT_HARDER)
    assert is_map_correct(MAP_CORRECT)
    assert is_lambda_correct(LAMBDA_CORRECT)
