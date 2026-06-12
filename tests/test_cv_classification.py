from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.cv.image_io import demo_image_specs, examples_ready, overlay_heatmap, pil_to_rgb_array


def test_format_top_prediction_summary_mentions_runner_up() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("tensorflow")
    from dataset_streamlit_shell.cv.classification import (
        PredictionItem,
        format_top_prediction_summary,
    )

    items = (
        PredictionItem(1, "n02108422", 0, "bull mastiff", 0.82),
        PredictionItem(2, "n02109047", 1, "Great Dane", 0.10),
    )
    summary = format_top_prediction_summary(items)
    assert "bull mastiff" in summary
    assert "Great Dane" in summary


def test_overlay_heatmap_resizes_to_image_shape() -> None:
    rgb = np.zeros((120, 200, 3), dtype=np.uint8)
    heatmap = np.linspace(0.0, 1.0, num=16 * 16, dtype=np.float32).reshape(16, 16)
    blended = overlay_heatmap(rgb, heatmap)
    assert blended.shape == rgb.shape
    assert blended.dtype == np.uint8


def test_pil_to_rgb_array_resize() -> None:
    from PIL import Image

    image = Image.new("RGB", (300, 100), color=(10, 20, 30))
    array = pil_to_rgb_array(image, size=(64, 64))
    assert array.shape == (64, 64, 3)


def test_demo_image_specs_has_four_entries() -> None:
    assert len(demo_image_specs()) == 4


def test_classification_bundled_examples_exist() -> None:
    from dataset_streamlit_shell.cv.image_io import EXAMPLES_DIR

    for spec in demo_image_specs():
        path = EXAMPLES_DIR / spec.filename
        assert path.exists(), f"missing bundled example: {spec.filename}"


def test_classification_examples_ready_with_bundled_data() -> None:
    assert examples_ready() is True


def test_build_mini_cnn_output_shape() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("tensorflow")
    from dataset_streamlit_shell.cv.mini_cnn import build_mini_cnn

    model = build_mini_cnn()
    output = model.output_shape
    assert output[-1] == 2


def test_default_backbone_is_resnet50() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("tensorflow")
    from dataset_streamlit_shell.cv.classification import DEFAULT_BACKBONE

    assert DEFAULT_BACKBONE == "resnet50"
