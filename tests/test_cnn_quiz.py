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

from dataset_streamlit_shell.ui.cnn_quiz import (  # noqa: E402
    KERNEL_CORRECT,
    KERNEL_SLIDE_ONLY,
    LEARNING_STAGES,
    PLEASE_SELECT,
    POOL_CORRECT,
    POOL_AVERAGE_ONLY,
    STAGE_CONV,
    STAGE_FLOW,
    STAGE_HANDS_ON,
    STAGE_MATRIX,
    STAGE_RELU_POOL,
    STAGE_WHY,
    VIDEO_ID,
    both_quiz_correct,
    is_kernel_correct,
    is_pool_correct,
    stage_video_start_sec,
    youtube_watch_url,
)


def test_learning_stages_order() -> None:
    assert LEARNING_STAGES == (
        STAGE_MATRIX,
        STAGE_WHY,
        STAGE_CONV,
        STAGE_RELU_POOL,
        STAGE_FLOW,
        STAGE_HANDS_ON,
    )


def test_stage1_has_no_video_embed() -> None:
    assert stage_video_start_sec(STAGE_MATRIX) is None


def test_later_stages_have_video_starts() -> None:
    for stage in (STAGE_WHY, STAGE_CONV, STAGE_RELU_POOL, STAGE_FLOW, STAGE_HANDS_ON):
        start = stage_video_start_sec(stage)
        assert isinstance(start, int)
        assert start >= 0


def test_youtube_watch_url_includes_start() -> None:
    url = youtube_watch_url(VIDEO_ID, start_sec=87)
    assert VIDEO_ID in url
    assert "t=87" in url


def test_youtube_embed_url_includes_start() -> None:
    from dataset_streamlit_shell.ui.cnn_quiz import youtube_embed_url

    url = youtube_embed_url(VIDEO_ID, start_sec=87)
    assert VIDEO_ID in url
    assert "start=87" in url


def test_pretrain_quiz_gate() -> None:
    assert both_quiz_correct(KERNEL_CORRECT, POOL_CORRECT)
    assert not both_quiz_correct(PLEASE_SELECT, POOL_CORRECT)
    assert not both_quiz_correct(KERNEL_CORRECT, PLEASE_SELECT)
    assert not both_quiz_correct(KERNEL_SLIDE_ONLY, POOL_CORRECT)
    assert not both_quiz_correct(KERNEL_CORRECT, POOL_AVERAGE_ONLY)
    assert is_kernel_correct(KERNEL_CORRECT)
    assert is_pool_correct(POOL_CORRECT)
