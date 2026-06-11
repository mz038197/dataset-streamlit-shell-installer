from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_MODEL = "deeplabv3_resnet50_coco"

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (26, 115, 232),
    2: (219, 68, 55),
    3: (15, 157, 88),
    4: (244, 180, 0),
    5: (171, 71, 188),
    6: (0, 172, 193),
    7: (121, 85, 72),
    8: (233, 30, 99),
    9: (63, 81, 181),
    10: (0, 150, 136),
    11: (205, 220, 57),
    12: (255, 87, 34),
    13: (96, 125, 139),
    14: (158, 157, 36),
    15: (103, 58, 183),
    16: (0, 188, 212),
    17: (255, 193, 7),
    18: (109, 76, 65),
    19: (3, 169, 244),
    20: (139, 195, 74),
}


@dataclass(frozen=True)
class ClassCoverage:
    class_id: int
    label: str
    pixel_count: int
    coverage: float
    color: tuple[int, int, int]


@dataclass(frozen=True)
class SemanticResult:
    label_map: np.ndarray
    class_items: tuple[ClassCoverage, ...]
    color_overlay: np.ndarray
    model_name: str


def load_segmentation_model():
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        deeplabv3_resnet50,
    )

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval()
    return model, weights


def class_names_from_weights(weights) -> list[str]:
    categories = list(weights.meta.get("categories", ()))
    if categories:
        return [str(name) for name in categories]
    return [f"class_{index}" for index in range(21)]


def predict_semantic_mask(
    image: np.ndarray,
    *,
    model=None,
    weights=None,
) -> SemanticResult:
    if model is None or weights is None:
        model, weights = load_segmentation_model()

    class_names = class_names_from_weights(weights)
    preprocess = weights.transforms()
    pil_image = Image.fromarray(image)
    batch = preprocess(pil_image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(batch)["out"][0]
        label_map_small = logits.argmax(0)

    resized = F.interpolate(
        label_map_small.unsqueeze(0).unsqueeze(0).float(),
        size=(image.shape[0], image.shape[1]),
        mode="nearest",
    )[0, 0]
    label_map = resized.cpu().numpy().astype(np.int32)
    class_items = compute_class_coverages(label_map, class_names)
    color_overlay = build_color_overlay(label_map, class_items)
    return SemanticResult(
        label_map=label_map,
        class_items=class_items,
        color_overlay=color_overlay,
        model_name=DEFAULT_MODEL,
    )


def compute_class_coverages(
    label_map: np.ndarray,
    class_names: list[str],
) -> tuple[ClassCoverage, ...]:
    total_pixels = int(label_map.size)
    unique, counts = np.unique(label_map, return_counts=True)
    items: list[ClassCoverage] = []
    for class_id, pixel_count in zip(unique.tolist(), counts.tolist(), strict=True):
        class_id = int(class_id)
        label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        items.append(
            ClassCoverage(
                class_id=class_id,
                label=label,
                pixel_count=int(pixel_count),
                coverage=float(pixel_count) / float(total_pixels),
                color=color_for_class(class_id),
            )
        )
    items.sort(key=lambda item: item.coverage, reverse=True)
    return tuple(items)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    if class_id in CLASS_COLORS:
        return CLASS_COLORS[class_id]
    seed = int(class_id) + 17
    return (
        (seed * 37) % 180 + 40,
        (seed * 57) % 180 + 40,
        (seed * 79) % 180 + 40,
    )


def build_color_overlay(
    label_map: np.ndarray,
    class_items: tuple[ClassCoverage, ...],
) -> np.ndarray:
    overlay = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for item in class_items:
        if item.class_id == 0:
            continue
        overlay[label_map == item.class_id] = item.color
    return overlay


def blend_overlay(
    rgb: np.ndarray,
    color_overlay: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    mask = np.any(color_overlay > 0, axis=-1)
    blended = rgb.copy().astype(np.float32)
    overlay = color_overlay.astype(np.float32)
    blended[mask] = alpha * overlay[mask] + (1.0 - alpha) * blended[mask]
    return blended.astype(np.uint8)


def isolate_class_mask(label_map: np.ndarray, class_id: int) -> np.ndarray:
    return (label_map == class_id).astype(np.uint8)


def highlight_class(rgb: np.ndarray, label_map: np.ndarray, class_id: int) -> np.ndarray:
    mask = label_map == class_id
    highlighted = rgb.copy()
    dimmed = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    highlighted[~mask] = dimmed[~mask]
    return highlighted


def format_semantic_summary(class_items: tuple[ClassCoverage, ...], *, top_n: int = 3) -> str:
    visible = [item for item in class_items if item.class_id != 0]
    if not visible:
        return "No foreground semantic classes detected."
    top = visible[:top_n]
    parts = [f"{item.label} {item.coverage:.1%}" for item in top]
    return "Main semantic regions: " + ", ".join(parts)
