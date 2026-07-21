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

from dataset_streamlit_shell.ui.svm_quiz import (  # noqa: E402
    C_CORRECT,
    C_WIDER_MARGIN,
    HARD_CORRECT,
    HARD_ALWAYS_OK,
    NORM_CORRECT,
    NORM_LARGER,
    OBJ_CORRECT,
    OBJ_BEST_SPLIT,
    PLEASE_SELECT,
    SKLEARN_HARD_EXAMPLE,
    SKLEARN_SOFT_EXAMPLE,
    both_hard_quiz_correct,
    both_soft_quiz_correct,
    build_hard_quiz_agent_appendix,
    build_soft_quiz_agent_appendix,
    can_send_hint,
    is_c_correct,
    is_hard_limit_correct,
    is_norm_correct,
    is_obj_correct,
    needs_quiz_reset,
    pair_key,
    quiz_choice_status,
)


def test_hard_quiz_gate() -> None:
    assert both_hard_quiz_correct(OBJ_CORRECT, NORM_CORRECT)
    assert not both_hard_quiz_correct(PLEASE_SELECT, NORM_CORRECT)
    assert not both_hard_quiz_correct(OBJ_CORRECT, NORM_LARGER)
    assert not both_hard_quiz_correct(OBJ_BEST_SPLIT, NORM_CORRECT)
    assert is_obj_correct(OBJ_CORRECT)
    assert is_norm_correct(NORM_CORRECT)


def test_soft_quiz_gate() -> None:
    assert both_soft_quiz_correct(HARD_CORRECT, C_CORRECT)
    assert not both_soft_quiz_correct(PLEASE_SELECT, C_CORRECT)
    assert not both_soft_quiz_correct(HARD_CORRECT, C_WIDER_MARGIN)
    assert not both_soft_quiz_correct(HARD_ALWAYS_OK, C_CORRECT)
    assert is_hard_limit_correct(HARD_CORRECT)
    assert is_c_correct(C_CORRECT)


def test_needs_quiz_reset_on_pair_change() -> None:
    features = ["特徵1", "特徵2"]
    assert (
        needs_quiz_reset(
            None, features, "類別", source_label="builtin", tab="hard"
        )
        is False
    )
    stored = pair_key(features, "類別", source_label="builtin", tab="hard")
    assert needs_quiz_reset(stored, features, "類別", source_label="builtin", tab="hard") is False
    assert needs_quiz_reset(stored, features, "其他", source_label="builtin", tab="hard") is True
    assert needs_quiz_reset(stored, features, "類別", source_label="ready", tab="hard") is True


def test_hint_cooldown_and_appendix() -> None:
    assert can_send_hint(None, 10.0) is True
    assert can_send_hint(10.0, 11.0, cooldown=2.5) is False
    assert can_send_hint(10.0, 13.0, cooldown=2.5) is True
    hard = build_hard_quiz_agent_appendix(
        obj_status="錯誤",
        norm_status="未選",
        focus_qid="objective",
        features=["特徵1", "特徵2"],
        target="類別",
        unlocked=False,
    )
    assert "請勿直接告訴學生" in hard
    assert "訓練是否已解鎖：否" in hard
    assert "硬間隔" in hard
    assert "C" not in hard
    soft = build_soft_quiz_agent_appendix(
        hard_status="未選",
        c_status="未選",
        focus_qid="c_tradeoff",
        features=["特徵1", "特徵2"],
        target="類別",
        C=1.0,
        unlocked=False,
    )
    assert "請勿直接告訴學生" in soft
    assert "C=1" in soft
    assert quiz_choice_status(PLEASE_SELECT, correct=False) == "未選"
    assert quiz_choice_status(OBJ_CORRECT, correct=True) == "正確"


def test_sklearn_examples_c_policy() -> None:
    assert "C=" not in SKLEARN_HARD_EXAMPLE
    assert "不設定 C" not in SKLEARN_HARD_EXAMPLE
    assert 'SVC(kernel="linear")' in SKLEARN_HARD_EXAMPLE
    assert "C=" in SKLEARN_SOFT_EXAMPLE
    assert "svm_soft_margin_80.csv" in SKLEARN_SOFT_EXAMPLE
