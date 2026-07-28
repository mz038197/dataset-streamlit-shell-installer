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

from dataset_streamlit_shell.ui.tree_ensemble_quiz import (  # noqa: E402
    BAGGING_CORRECT,
    BAGGING_SEQ,
    BOOST_CORRECT,
    BOOST_PARALLEL,
    CONTRAST_BAGGING,
    CONTRAST_CORRECT,
    ENTROPY_CORRECT,
    ENTROPY_GAIN,
    IG_ALWAYS_ZERO,
    IG_CORRECT,
    PLEASE_SELECT,
    STAGE1_LABEL,
    STAGE2_LABEL,
    STAGE3_LABEL,
    VOTE_AVG,
    VOTE_CORRECT,
    both_bagging_quiz_correct,
    both_boost_quiz_correct,
    both_tree_quiz_correct,
    build_bagging_quiz_agent_appendix,
    build_boost_quiz_agent_appendix,
    build_tree_quiz_agent_appendix,
    can_send_hint,
    is_bagging_correct,
    is_boost_correct,
    is_contrast_correct,
    is_entropy_correct,
    is_ig_correct,
    is_vote_correct,
    learning_stage_labels,
    quiz_choice_status,
)


def test_learning_stage_labels_match_glossary() -> None:
    assert learning_stage_labels() == (STAGE1_LABEL, STAGE2_LABEL, STAGE3_LABEL)
    assert "Bagging" in STAGE2_LABEL
    assert "Boosting" in STAGE3_LABEL


def test_tree_quiz_gate() -> None:
    assert both_tree_quiz_correct(ENTROPY_CORRECT, IG_CORRECT)
    assert not both_tree_quiz_correct(PLEASE_SELECT, IG_CORRECT)
    assert not both_tree_quiz_correct(ENTROPY_CORRECT, IG_ALWAYS_ZERO)
    assert not both_tree_quiz_correct(ENTROPY_GAIN, IG_CORRECT)
    assert is_entropy_correct(ENTROPY_CORRECT)
    assert is_ig_correct(IG_CORRECT)


def test_bagging_quiz_gate() -> None:
    assert both_bagging_quiz_correct(BAGGING_CORRECT, VOTE_CORRECT)
    assert not both_bagging_quiz_correct(BAGGING_SEQ, VOTE_CORRECT)
    assert not both_bagging_quiz_correct(BAGGING_CORRECT, VOTE_AVG)
    assert is_bagging_correct(BAGGING_CORRECT)
    assert is_vote_correct(VOTE_CORRECT)


def test_boost_quiz_gate() -> None:
    assert both_boost_quiz_correct(BOOST_CORRECT, CONTRAST_CORRECT)
    assert not both_boost_quiz_correct(BOOST_PARALLEL, CONTRAST_CORRECT)
    assert not both_boost_quiz_correct(BOOST_CORRECT, CONTRAST_BAGGING)
    assert is_boost_correct(BOOST_CORRECT)
    assert is_contrast_correct(CONTRAST_CORRECT)


def test_hint_and_appendix() -> None:
    assert can_send_hint(None, 10.0) is True
    assert can_send_hint(10.0, 11.0, cooldown=2.5) is False
    tree_app = build_tree_quiz_agent_appendix(
        entropy_status="未選",
        ig_status="未選",
        focus_qid="entropy",
        unlocked=False,
    )
    assert "請勿直接告訴學生" in tree_app
    bag_app = build_bagging_quiz_agent_appendix(
        bagging_status="錯誤",
        vote_status="未選",
        focus_qid="bagging",
        unlocked=False,
        n_estimators=30,
    )
    assert "n_estimators=30" in bag_app
    boost_app = build_boost_quiz_agent_appendix(
        boost_status="未選",
        contrast_status="未選",
        focus_qid="contrast",
        unlocked=False,
        learning_rate=0.1,
    )
    assert "learning_rate=0.1" in boost_app
    assert quiz_choice_status(PLEASE_SELECT, correct=False) == "未選"
    assert quiz_choice_status(ENTROPY_CORRECT, correct=True) == "正確"
