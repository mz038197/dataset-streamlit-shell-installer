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

from dataset_streamlit_shell.cv.image_io import semantic_demo_specs, semantic_examples_ready
from dataset_streamlit_shell.cv.semantic_segmentation import (
    DEFAULT_MODEL,
    ClassCoverage,
    blend_overlay,
    build_color_overlay,
    compute_class_coverages,
    format_semantic_summary,
    highlight_class,
    isolate_class_mask,
)


def test_default_model_name() -> None:
    assert DEFAULT_MODEL == "deeplabv3_resnet50_coco"


def test_compute_class_coverages_sorts_by_coverage() -> None:
    label_map = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [1, 1, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    class_names = ["__background__", "person", "car"]
    items = compute_class_coverages(label_map, class_names)
    assert items[0].label == "person"
    assert items[0].coverage > items[1].coverage
    labels = [item.label for item in items]
    assert "car" in labels


def test_blend_overlay_preserves_shape() -> None:
    rgb = np.full((40, 50, 3), 200, dtype=np.uint8)
    overlay = np.zeros_like(rgb)
    overlay[:, :25] = (10, 120, 30)
    blended = blend_overlay(rgb, overlay, alpha=0.5)
    assert blended.shape == rgb.shape
    assert blended.dtype == np.uint8


def test_isolate_class_mask_binary_values() -> None:
    label_map = np.array([[0, 1], [1, 2]], dtype=np.int32)
    mask = isolate_class_mask(label_map, 1)
    assert mask.tolist() == [[0, 1], [1, 0]]


def test_highlight_class_dims_non_target_pixels() -> None:
    rgb = np.full((2, 2, 3), 200, dtype=np.uint8)
    label_map = np.array([[0, 1], [1, 0]], dtype=np.int32)
    highlighted = highlight_class(rgb, label_map, 1)
    assert highlighted[0, 0].mean() < rgb[0, 0].mean()
    assert highlighted[0, 1].mean() == rgb[0, 1].mean()


def test_build_color_overlay_skips_background() -> None:
    label_map = np.array([[0, 1], [1, 0]], dtype=np.int32)
    items = (
        ClassCoverage(0, "__background__", 2, 0.5, (0, 0, 0)),
        ClassCoverage(1, "person", 2, 0.5, (26, 115, 232)),
    )
    overlay = build_color_overlay(label_map, items)
    assert overlay[0, 0].tolist() == [0, 0, 0]
    assert overlay[0, 1].tolist() == [26, 115, 232]


def test_format_semantic_summary_mentions_top_regions() -> None:
    items = (
        ClassCoverage(0, "__background__", 10, 0.5, (0, 0, 0)),
        ClassCoverage(15, "person", 6, 0.3, (1, 2, 3)),
        ClassCoverage(2, "car", 4, 0.2, (4, 5, 6)),
    )
    summary = format_semantic_summary(items)
    assert "person" in summary
    assert "car" in summary


def test_semantic_demo_specs_has_four_entries() -> None:
    assert len(semantic_demo_specs()) == 4


def test_semantic_examples_ready_false_when_missing() -> None:
    assert semantic_examples_ready() is False
