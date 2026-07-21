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

from dataset_streamlit_shell.ui.multiple_regression_quiz import (  # noqa: E402
    PLEASE_SELECT,
    PURPOSE_ALWAYS_BETTER,
    PURPOSE_CORRECT,
    PURPOSE_MULTI,
    WEIGHTS_CLASS,
    WEIGHTS_CORRECT,
    both_quiz_correct,
    build_quiz_agent_appendix,
    can_send_hint,
    is_purpose_correct,
    is_weights_correct,
    needs_quiz_reset,
    pair_key,
    quiz_choice_status,
)


def test_both_quiz_correct_gate() -> None:
    assert both_quiz_correct(PURPOSE_CORRECT, WEIGHTS_CORRECT)
    assert not both_quiz_correct(PLEASE_SELECT, WEIGHTS_CORRECT)
    assert not both_quiz_correct(PURPOSE_CORRECT, WEIGHTS_CLASS)
    assert not both_quiz_correct(PURPOSE_ALWAYS_BETTER, WEIGHTS_CORRECT)
    assert is_purpose_correct(PURPOSE_MULTI)
    assert is_weights_correct(WEIGHTS_CORRECT)


def test_needs_quiz_reset_on_pair_change() -> None:
    features = ["面積_平方英尺", "房間數"]
    assert needs_quiz_reset(None, features, "房價_千美元") is False
    assert needs_quiz_reset(pair_key(features, "房價_千美元"), features, "房價_千美元") is False
    assert needs_quiz_reset(pair_key(features, "房價_千美元"), features, "其他") is True
    assert needs_quiz_reset(
        pair_key(features, "房價_千美元"),
        ["面積_平方英尺", "屋齡_年"],
        "房價_千美元",
    ) is True


def test_hint_cooldown_and_appendix() -> None:
    assert can_send_hint(None, 10.0) is True
    assert can_send_hint(10.0, 11.0, cooldown=2.5) is False
    assert can_send_hint(10.0, 13.0, cooldown=2.5) is True
    housing = build_quiz_agent_appendix(
        purpose_status="錯誤",
        weights_status="未選",
        focus_qid="purpose",
        features=["面積_平方英尺", "房間數"],
        target="房價_千美元",
        learning_rate=0.1,
        unlocked=False,
        use_housing_example=True,
    )
    assert "請勿直接告訴學生" in housing
    assert "訓練是否已解鎖：否" in housing
    assert "房價例子" in housing
    custom = build_quiz_agent_appendix(
        purpose_status="未選",
        weights_status="未選",
        focus_qid="purpose",
        features=["temp", "humidity"],
        target="yield",
        learning_rate=0.01,
        unlocked=False,
        use_housing_example=False,
    )
    assert "房價例子" not in custom
    assert "temp" in custom and "yield" in custom
    assert quiz_choice_status(PLEASE_SELECT, correct=False) == "未選"
    assert quiz_choice_status(PURPOSE_MULTI, correct=True) == "正確"
