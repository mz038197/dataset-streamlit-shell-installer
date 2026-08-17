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

from dataset_streamlit_shell.ui.knn_quiz import (  # noqa: E402
    INST_CORRECT,
    INST_HAS_W,
    NEIGHBORS_FORMULA_CAPTION,
    PLEASE_SELECT,
    SKLEARN_NEIGHBORS_EXAMPLE,
    VOTE_B,
    VOTE_CORRECT,
    both_neighbors_quiz_correct,
    build_neighbors_quiz_agent_appendix,
    can_send_hint,
    is_inst_correct,
    is_vote_correct,
    needs_quiz_reset,
    pair_key,
    query_prediction_caption,
    quiz_choice_status,
    result_chart_caption,
    stage1_ui_leaks_k,
    vote_progress_caption,
)


def test_neighbors_quiz_gate() -> None:
    assert both_neighbors_quiz_correct(INST_CORRECT, VOTE_CORRECT)
    assert not both_neighbors_quiz_correct(PLEASE_SELECT, VOTE_CORRECT)
    assert not both_neighbors_quiz_correct(INST_CORRECT, VOTE_B)
    assert not both_neighbors_quiz_correct(INST_HAS_W, VOTE_CORRECT)
    assert is_inst_correct(INST_CORRECT)
    assert is_vote_correct(VOTE_CORRECT)


def test_needs_quiz_reset_on_pair_change() -> None:
    features = ["特徵1", "特徵2"]
    assert (
        needs_quiz_reset(
            None, features, "類別", source_label="builtin", tab="neighbors"
        )
        is False
    )
    stored = pair_key(features, "類別", source_label="builtin", tab="neighbors")
    assert (
        needs_quiz_reset(stored, features, "類別", source_label="builtin", tab="neighbors")
        is False
    )
    assert needs_quiz_reset(stored, features, "其他", source_label="builtin", tab="neighbors")


def test_hint_cooldown_and_appendix() -> None:
    assert can_send_hint(None, 10.0) is True
    assert can_send_hint(10.0, 11.0, cooldown=2.5) is False
    neighbors = build_neighbors_quiz_agent_appendix(
        inst_status="錯誤",
        vote_status="未選",
        focus_qid="instance",
        features=["特徵1", "特徵2"],
        target="類別",
        unlocked=False,
    )
    assert "請勿直接告訴學生" in neighbors
    assert "預測演示是否已解鎖：否" in neighbors
    assert "階段2" not in neighbors
    assert quiz_choice_status(PLEASE_SELECT, correct=False) == "未選"
    assert quiz_choice_status(INST_CORRECT, correct=True) == "正確"


def test_sklearn_example_uses_blobs_and_standardize() -> None:
    assert "knn_blobs_80.csv" in SKLEARN_NEIGHBORS_EXAMPLE
    assert "StandardScaler" in SKLEARN_NEIGHBORS_EXAMPLE
    assert "knn_scale_trap_80.csv" not in SKLEARN_NEIGHBORS_EXAMPLE


def test_surface_copy_uses_next_step_not_autoplay() -> None:
    chart = result_chart_caption()
    pred = query_prediction_caption(0.1, 0.2, 1, k=5)
    assert "開始預測演示" in chart
    assert "下一步" in chart
    assert "自動演" not in chart
    assert "k=" in pred
    assert not stage1_ui_leaks_k(NEIGHBORS_FORMULA_CAPTION)
    assert "5" not in NEIGHBORS_FORMULA_CAPTION


def test_vote_progress_caption() -> None:
    assert "等待" in vote_progress_caption({})
    mid = vote_progress_caption({0: 2, 1: 1})
    assert "類別 0×2" in mid and "類別 1×1" in mid
    assert "預測" not in mid
    done = vote_progress_caption({0: 3, 1: 2}, finalized_pred=0)
    assert "預測類別 **0**" in done


def test_knn_page_is_single_stage_stepper() -> None:
    ui_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "add_dataset_streamlit_shell"
        / "templates"
        / "dataset_streamlit_shell"
        / "ui"
        / "knn_ui.py"
    )
    src = ui_path.read_text(encoding="utf-8")
    assert "knn_learning_stage" not in src
    assert "_render_k_stage" not in src
    assert "knn_k_standardize" not in src
    assert "KNN_TRAP_PATH" not in src
    assert "time.sleep" not in src
    assert "開始預測演示" in src
    assert "開始訓練" not in src
    assert "下一步" in src
    assert "knn_k_slider" in src
    assert "預測過程演進" in src
    assert 'key="train_knn_neighbors"' in src
    assert "knn_neighbors_plotly_step" in src
    assert 'else "#7c3aed"' in src
    assert "size=18" in src
    assert 'else "#111827"' not in src
