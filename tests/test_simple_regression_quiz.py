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

from dataset_streamlit_shell.ui.simple_regression_quiz import (  # noqa: E402
    ALPHA_CONVERGE,
    ALPHA_CORRECT,
    ALPHA_EXPLODE,
    PLEASE_SELECT,
    SLOPE_NEAR_ZERO,
    SLOPE_NEGATIVE,
    SLOPE_POSITIVE,
    both_quiz_correct,
    build_quiz_agent_appendix,
    build_regression_frame_hint_summary,
    can_send_hint,
    expected_slope_direction,
    is_alpha_correct,
    is_slope_correct,
    needs_quiz_reset,
    quiz_choice_status,
    quiz_identity,
    slope_label_from_weight,
)


def test_expected_slope_positive_negative_near_zero() -> None:
    pos = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
    neg = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [8.0, 6.0, 4.0, 2.0]})
    flat = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [5.0, 5.0, 5.0, 5.0]})
    weak = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 1.02, 0.99, 1.01]})
    assert expected_slope_direction(pos, "x", "y") == SLOPE_POSITIVE
    assert expected_slope_direction(neg, "x", "y") == SLOPE_NEGATIVE
    assert expected_slope_direction(flat, "x", "y") == SLOPE_NEAR_ZERO
    assert expected_slope_direction(weak, "x", "y") == SLOPE_NEAR_ZERO


def test_both_quiz_correct_gate() -> None:
    assert both_quiz_correct(SLOPE_POSITIVE, ALPHA_EXPLODE, expected_slope=SLOPE_POSITIVE)
    assert not both_quiz_correct(PLEASE_SELECT, ALPHA_EXPLODE, expected_slope=SLOPE_POSITIVE)
    assert not both_quiz_correct(SLOPE_POSITIVE, ALPHA_CONVERGE, expected_slope=SLOPE_POSITIVE)
    assert not both_quiz_correct(SLOPE_NEGATIVE, ALPHA_EXPLODE, expected_slope=SLOPE_POSITIVE)
    assert is_alpha_correct(ALPHA_CORRECT)
    assert is_slope_correct(SLOPE_NEGATIVE, SLOPE_NEGATIVE)


def test_needs_quiz_reset_includes_source_and_data_signature() -> None:
    frame_a = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    frame_b = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [6.0, 4.0, 2.0]})
    id_builtin = quiz_identity("a", "b", source_label="內建", frame=frame_a)
    id_ready_same_cols = quiz_identity("a", "b", source_label="ready.csv", frame=frame_a)
    id_ready_new_data = quiz_identity("a", "b", source_label="ready.csv", frame=frame_b)

    assert needs_quiz_reset(None, id_builtin) is False
    assert needs_quiz_reset(id_builtin, id_builtin) is False
    assert needs_quiz_reset(id_builtin, id_ready_same_cols) is True
    assert needs_quiz_reset(id_ready_same_cols, id_ready_new_data) is True
    assert needs_quiz_reset(("a", "b"), id_builtin) is True


def test_slope_label_from_weight_matches_quiz_correlation_scale() -> None:
    # w=0.01 with sx=sy=1 → corr_proxy=0.01 ≤ CORR_EPS → 接近 0
    assert slope_label_from_weight(0.01, x_std=1.0, y_std=1.0) == SLOPE_NEAR_ZERO
    assert slope_label_from_weight(-0.01, x_std=1.0, y_std=1.0) == SLOPE_NEAR_ZERO
    # Same raw w, but larger x scale → stronger corr_proxy
    assert slope_label_from_weight(0.01, x_std=10.0, y_std=1.0) == SLOPE_POSITIVE
    assert slope_label_from_weight(-0.01, x_std=10.0, y_std=1.0) == SLOPE_NEGATIVE
    assert slope_label_from_weight(1.2, x_std=1.0, y_std=1.0) == SLOPE_POSITIVE
    assert slope_label_from_weight(-0.3, x_std=1.0, y_std=1.0) == SLOPE_NEGATIVE
    assert slope_label_from_weight(0.0, x_std=1.0, y_std=1.0) == SLOPE_NEAR_ZERO

    weak = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 1.02, 0.99, 1.01]})
    assert expected_slope_direction(weak, "x", "y") == SLOPE_NEAR_ZERO
    # Trained OLS-ish slope on weak noise should still classify as 接近 0
    x = weak["x"].astype(float)
    y = weak["y"].astype(float)
    w = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    assert (
        slope_label_from_weight(
            w,
            x_std=float(x.std(ddof=0)),
            y_std=float(y.std(ddof=0)),
        )
        == SLOPE_NEAR_ZERO
    )

    assert quiz_choice_status(PLEASE_SELECT, correct=False) == "未選"
    assert quiz_choice_status(SLOPE_POSITIVE, correct=True) == "正確"
    assert quiz_choice_status(SLOPE_POSITIVE, correct=False) == "錯誤"


def test_hint_summary_uses_frame_without_answer_leak() -> None:
    frame = pd.DataFrame(
        {
            "城市人口_萬人": [1.0, 2.0, 3.0, 4.0],
            "餐廳獲利_萬美元": [2.0, 4.0, 6.0, 8.0],
        }
    )
    text = build_regression_frame_hint_summary(frame, "城市人口_萬人", "餐廳獲利_萬美元")
    assert "實際用於本頁的資料筆數：4" in text
    assert "城市人口_萬人" in text
    assert "餐廳獲利_萬美元" in text
    assert SLOPE_POSITIVE not in text
    assert SLOPE_NEGATIVE not in text
    assert SLOPE_NEAR_ZERO not in text
    assert ALPHA_EXPLODE not in text


def test_hint_cooldown_and_appendix_no_answer_leak() -> None:
    assert can_send_hint(None, 100.0) is True
    assert can_send_hint(100.0, 101.0, cooldown=2.5) is False
    assert can_send_hint(100.0, 103.0, cooldown=2.5) is True
    text = build_quiz_agent_appendix(
        slope_status="錯誤",
        alpha_status="未選",
        focus_qid="slope",
        feature="城市人口_萬人",
        target="餐廳獲利_萬美元",
        learning_rate=0.01,
        unlocked=False,
    )
    assert "勿直接告訴學生" in text or "不要直接" in text or "請勿直接" in text
    assert ALPHA_EXPLODE not in text
    assert "訓練是否已解鎖：否" in text
