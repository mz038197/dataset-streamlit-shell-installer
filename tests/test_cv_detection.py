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

from dataset_streamlit_shell.cv.detection import (
    DEFAULT_MODEL,
    DetectionItem,
    crop_detection,
    draw_detections,
    filter_detections,
    format_detection_summary,
)
from dataset_streamlit_shell.cv.image_io import detection_demo_specs, detection_examples_ready


def test_default_model_is_yolov8n() -> None:
    assert DEFAULT_MODEL == "yolov8n.pt"


def test_filter_detections_applies_threshold() -> None:
    items = (
        DetectionItem(1, "person", 0.82, 10, 20, 100, 200),
        DetectionItem(2, "car", 0.18, 120, 40, 260, 180),
    )
    filtered = filter_detections(items, conf_threshold=0.25)
    assert len(filtered) == 1
    assert filtered[0].label == "person"
    assert filtered[0].rank == 1


def test_draw_detections_preserves_shape() -> None:
    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    items = (DetectionItem(1, "dog", 0.91, 5, 8, 60, 70),)
    drawn = draw_detections(rgb, items)
    assert drawn.shape == rgb.shape
    assert drawn.dtype == np.uint8


def test_crop_detection_returns_region() -> None:
    rgb = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
    item = DetectionItem(1, "cat", 0.77, 10, 5, 40, 25)
    crop = crop_detection(rgb, item)
    assert crop.shape == (20, 30, 3)


def test_format_detection_summary_lists_unique_labels() -> None:
    items = (
        DetectionItem(1, "person", 0.9, 0, 0, 10, 10),
        DetectionItem(2, "person", 0.8, 20, 20, 40, 40),
        DetectionItem(3, "car", 0.7, 50, 50, 80, 80),
    )
    summary = format_detection_summary(items)
    assert "person" in summary
    assert "car" in summary
    assert "3 object" in summary


def test_detection_demo_specs_has_four_entries() -> None:
    assert len(detection_demo_specs()) == 4


def test_detection_examples_ready_false_when_missing() -> None:
    assert detection_examples_ready() is False
