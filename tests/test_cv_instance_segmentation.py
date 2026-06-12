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

from dataset_streamlit_shell.cv.image_io import instance_demo_specs, instance_examples_ready
from dataset_streamlit_shell.cv.instance_segmentation import (
    DEFAULT_MODEL,
    InstanceItem,
    blend_overlay,
    build_color_overlay,
    crop_instance,
    draw_instances,
    filter_instances,
    format_instance_summary,
    highlight_instance,
    isolate_instance_mask,
)


def _sample_item(
    *,
    rank: int = 1,
    label: str = "dog",
    confidence: float = 0.9,
    mask: np.ndarray | None = None,
) -> InstanceItem:
    if mask is None:
        mask = np.zeros((60, 80), dtype=bool)
        mask[10:40, 20:60] = True
    area = int(mask.sum())
    return InstanceItem(
        rank=rank,
        label=label,
        confidence=confidence,
        x1=20,
        y1=10,
        x2=60,
        y2=40,
        mask=mask,
        area=area,
        coverage=float(area) / float(mask.size),
        color=(26, 115, 232),
    )


def test_default_model_is_yolov8n_seg() -> None:
    assert DEFAULT_MODEL == "yolov8n-seg.pt"


def test_filter_instances_applies_threshold_and_reranks() -> None:
    items = (
        _sample_item(rank=1, label="person", confidence=0.82),
        _sample_item(rank=2, label="car", confidence=0.18),
    )
    filtered = filter_instances(items, conf_threshold=0.25)
    assert len(filtered) == 1
    assert filtered[0].label == "person"
    assert filtered[0].rank == 1


def test_build_color_overlay_matches_image_shape() -> None:
    items = (_sample_item(),)
    overlay = build_color_overlay((60, 80), items)
    assert overlay.shape == (60, 80, 3)
    assert overlay.dtype == np.uint8


def test_blend_overlay_preserves_shape() -> None:
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    overlay = build_color_overlay((60, 80), (_sample_item(),))
    blended = blend_overlay(rgb, overlay, alpha=0.5)
    assert blended.shape == rgb.shape


def test_draw_instances_preserves_shape() -> None:
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    drawn = draw_instances(rgb, (_sample_item(),))
    assert drawn.shape == rgb.shape
    assert drawn.dtype == np.uint8


def test_isolate_instance_mask_returns_binary_array() -> None:
    item = _sample_item()
    binary = isolate_instance_mask(item.mask)
    assert binary.dtype == np.uint8
    assert binary.max() == 1


def test_highlight_instance_preserves_shape() -> None:
    rgb = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
    item = _sample_item()
    highlighted = highlight_instance(rgb, item.mask, item.color)
    assert highlighted.shape == rgb.shape


def test_crop_instance_returns_region() -> None:
    rgb = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
    item = _sample_item()
    crop = crop_instance(rgb, item)
    assert crop.shape == (30, 40, 3)


def test_format_instance_summary_lists_unique_labels() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:20, 5:20] = True
    mask2 = np.zeros((40, 40), dtype=bool)
    mask2[22:35, 22:35] = True
    items = (
        _sample_item(rank=1, label="person", confidence=0.9, mask=mask),
        _sample_item(rank=2, label="person", confidence=0.8, mask=mask2),
        _sample_item(rank=3, label="car", confidence=0.7, mask=mask),
    )
    summary = format_instance_summary(items)
    assert "person" in summary
    assert "car" in summary
    assert "3 instance" in summary


def test_instance_demo_specs_has_four_entries() -> None:
    assert len(instance_demo_specs()) == 4


def test_instance_bundled_examples_exist() -> None:
    from dataset_streamlit_shell.cv.image_io import EXAMPLES_DIR

    for spec in instance_demo_specs():
        path = EXAMPLES_DIR / spec.filename
        assert path.exists(), f"missing bundled example: {spec.filename}"


def test_instance_examples_ready_with_bundled_data() -> None:
    assert instance_examples_ready() is True
