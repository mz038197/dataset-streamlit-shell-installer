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
    K_CORRECT,
    K_SHARP,
    NEIGHBORS_FORMULA_CAPTION,
    PLEASE_SELECT,
    SCALE_CORRECT,
    SCALE_EQUAL,
    SKLEARN_K_EXAMPLE,
    SKLEARN_NEIGHBORS_EXAMPLE,
    VOTE_CORRECT,
    VOTE_B,
    both_k_quiz_correct,
    both_neighbors_quiz_correct,
    build_k_quiz_agent_appendix,
    build_neighbors_quiz_agent_appendix,
    can_send_hint,
    is_inst_correct,
    is_k_correct,
    is_scale_correct,
    is_vote_correct,
    needs_quiz_reset,
    pair_key,
    query_prediction_caption,
    quiz_choice_status,
    result_chart_caption,
    stage1_ui_leaks_k,
)


def test_neighbors_quiz_gate() -> None:
    assert both_neighbors_quiz_correct(INST_CORRECT, VOTE_CORRECT)
    assert not both_neighbors_quiz_correct(PLEASE_SELECT, VOTE_CORRECT)
    assert not both_neighbors_quiz_correct(INST_CORRECT, VOTE_B)
    assert not both_neighbors_quiz_correct(INST_HAS_W, VOTE_CORRECT)
    assert is_inst_correct(INST_CORRECT)
    assert is_vote_correct(VOTE_CORRECT)


def test_k_quiz_gate() -> None:
    assert both_k_quiz_correct(K_CORRECT, SCALE_CORRECT)
    assert not both_k_quiz_correct(PLEASE_SELECT, SCALE_CORRECT)
    assert not both_k_quiz_correct(K_CORRECT, SCALE_EQUAL)
    assert not both_k_quiz_correct(K_SHARP, SCALE_CORRECT)
    assert is_k_correct(K_CORRECT)
    assert is_scale_correct(SCALE_CORRECT)


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
    assert "訓練是否已解鎖：否" in neighbors
    k_app = build_k_quiz_agent_appendix(
        k_status="未選",
        scale_status="未選",
        focus_qid="k_large",
        features=["特徵1", "特徵2"],
        target="類別",
        k=5,
        standardize=True,
        unlocked=False,
    )
    assert "請勿直接告訴學生" in k_app
    assert "k=5" in k_app
    assert quiz_choice_status(PLEASE_SELECT, correct=False) == "未選"
    assert quiz_choice_status(INST_CORRECT, correct=True) == "正確"


def test_sklearn_examples_mention_files() -> None:
    assert "knn_blobs_80.csv" in SKLEARN_NEIGHBORS_EXAMPLE
    assert "n_neighbors=5" in SKLEARN_NEIGHBORS_EXAMPLE
    assert "knn_scale_trap_80.csv" in SKLEARN_K_EXAMPLE
    assert "StandardScaler" in SKLEARN_K_EXAMPLE


def test_stage1_surface_copy_hides_k() -> None:
    chart = result_chart_caption(expose_k=False)
    pred = query_prediction_caption(0.1, 0.2, 1, k=5, expose_k=False)
    assert not stage1_ui_leaks_k(NEIGHBORS_FORMULA_CAPTION, chart, pred)
    assert "5" not in NEIGHBORS_FORMULA_CAPTION
    assert "k=" not in pred
    assert "k 個" not in chart
    # 階段2 仍可露 k
    assert "k=" in query_prediction_caption(0.1, 0.2, 1, k=5, expose_k=True)
    assert "k 個" in result_chart_caption(expose_k=True)
    assert stage1_ui_leaks_k("本階段 k 固定為 5")


def test_neighbors_stage_wires_expose_k_false() -> None:
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
    assert "本階段 k 固定為 5" not in src
    assert "expose_k=False" in src
    assert 'key="train_knn_neighbors"' in src
    # 階段1 結果區不得寫死 k metric 文案為唯一路徑；須走 expose_k 分支
    neighbors_fn = src.split("def _render_neighbors_stage")[1].split("def _render_k_stage")[0]
    assert "expose_k=False" in neighbors_fn
    assert "expose_k=True" not in neighbors_fn
