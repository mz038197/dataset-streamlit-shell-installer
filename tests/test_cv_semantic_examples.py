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

from dataset_streamlit_shell.cv.image_io import (
    EXAMPLES_DIR,
    semantic_demo_specs,
    semantic_examples_ready,
)


def test_semantic_demo_specs_uses_bundled_voc_images() -> None:
    names = [spec.filename for spec in semantic_demo_specs()]
    assert names == [
        "street_scene.jpg",
        "dog.jpg",
        "three_cats.jpg",
        "horse_rider.jpg",
    ]
    hints = [spec.hint for spec in semantic_demo_specs()]
    assert "background" in hints[0]
    assert "前景" in hints[1]
    assert "同類" in hints[2]
    assert "horse" in hints[3]


def test_semantic_bundled_examples_exist() -> None:
    for spec in semantic_demo_specs():
        path = EXAMPLES_DIR / spec.filename
        assert path.exists(), f"missing bundled example: {spec.filename}"


def test_semantic_examples_ready_with_bundled_data() -> None:
    assert semantic_examples_ready() is True


def test_semantic_page_does_not_ask_to_download_samples() -> None:
    src = (
        TEMPLATE_ROOT
        / "dataset_streamlit_shell"
        / "ui"
        / "semantic_segmentation_ui.py"
    ).read_text(encoding="utf-8")
    assert "download_sample_data" not in src
    assert "找不到內建範例圖" in src
    assert "請先下載範例資料" not in src
    assert "下載範例資料" not in src
