from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dataset_streamlit_shell.cv.detection import BOX_COLORS, DetectionItem, crop_detection

DEFAULT_MODEL = "yolov8n-seg.pt"


@dataclass(frozen=True)
class InstanceItem:
    rank: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    mask: np.ndarray
    area: int
    coverage: float
    color: tuple[int, int, int]

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class InstanceResult:
    items: tuple[InstanceItem, ...]
    raw_count: int
    model_name: str
    conf_threshold: float
    color_overlay: np.ndarray


def load_yolo_seg_model(model_name: str = DEFAULT_MODEL):
    from ultralytics import YOLO

    return YOLO(model_name)


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def predict_instances(
    image: np.ndarray,
    *,
    conf_threshold: float = 0.25,
    model=None,
) -> InstanceResult:
    segmenter = model or load_yolo_seg_model()
    outputs = segmenter.predict(image, conf=conf_threshold, verbose=False)
    boxes = outputs[0].boxes
    masks = outputs[0].masks
    height, width = image.shape[:2]
    total_pixels = int(height * width)

    if boxes is None or len(boxes) == 0 or masks is None:
        empty_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        return InstanceResult(
            items=tuple(),
            raw_count=0,
            model_name=DEFAULT_MODEL,
            conf_threshold=conf_threshold,
            color_overlay=empty_overlay,
        )

    order = np.argsort(-boxes.conf.cpu().numpy())
    items: list[InstanceItem] = []
    for rank, index in enumerate(order, start=1):
        xyxy = boxes.xyxy[index].cpu().numpy().astype(int).tolist()
        class_id = int(boxes.cls[index].cpu().numpy())
        label = str(outputs[0].names[class_id])
        confidence = float(boxes.conf[index].cpu().numpy())
        mask_tensor = masks.data[index].cpu().numpy()
        mask = _resize_mask(mask_tensor, height, width)
        area = int(mask.sum())
        color = BOX_COLORS[(rank - 1) % len(BOX_COLORS)]
        items.append(
            InstanceItem(
                rank=rank,
                label=label,
                confidence=confidence,
                x1=int(xyxy[0]),
                y1=int(xyxy[1]),
                x2=int(xyxy[2]),
                y2=int(xyxy[3]),
                mask=mask,
                area=area,
                coverage=float(area) / float(total_pixels),
                color=color,
            )
        )

    color_overlay = build_color_overlay(image.shape[:2], items)
    return InstanceResult(
        items=tuple(items),
        raw_count=len(items),
        model_name=DEFAULT_MODEL,
        conf_threshold=conf_threshold,
        color_overlay=color_overlay,
    )


def filter_instances(
    items: tuple[InstanceItem, ...],
    *,
    conf_threshold: float,
) -> tuple[InstanceItem, ...]:
    kept = [item for item in items if item.confidence >= conf_threshold]
    return tuple(
        InstanceItem(
            rank=index,
            label=item.label,
            confidence=item.confidence,
            x1=item.x1,
            y1=item.y1,
            x2=item.x2,
            y2=item.y2,
            mask=item.mask,
            area=item.area,
            coverage=item.coverage,
            color=BOX_COLORS[(index - 1) % len(BOX_COLORS)],
        )
        for index, item in enumerate(kept, start=1)
    )


def build_color_overlay(shape: tuple[int, int], items: tuple[InstanceItem, ...]) -> np.ndarray:
    height, width = shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    for item in items:
        overlay[item.mask] = item.color
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


def draw_instances(
    rgb: np.ndarray,
    items: tuple[InstanceItem, ...],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    canvas = blend_overlay(rgb, build_color_overlay(rgb.shape[:2], items), alpha=alpha)
    height, width = canvas.shape[:2]
    for item in items:
        color = item.color
        x1 = max(0, min(item.x1, width - 1))
        y1 = max(0, min(item.y1, height - 1))
        x2 = max(0, min(item.x2, width - 1))
        y2 = max(0, min(item.y2, height - 1))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{item.label} {item.confidence:.0%}"
        text_origin = (x1, max(18, y1 - 8))
        cv2.putText(
            canvas,
            caption,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def isolate_instance_mask(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8)


def highlight_instance(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    highlighted = rgb.copy()
    dimmed = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    highlighted[~mask] = dimmed[~mask]
    tint = np.zeros_like(highlighted)
    tint[mask] = color
    highlighted[mask] = (
        0.55 * tint[mask].astype(np.float32) + 0.45 * highlighted[mask].astype(np.float32)
    ).astype(np.uint8)
    return highlighted


def crop_instance(rgb: np.ndarray, item: InstanceItem) -> np.ndarray:
    detection_item = DetectionItem(
        rank=item.rank,
        label=item.label,
        confidence=item.confidence,
        x1=item.x1,
        y1=item.y1,
        x2=item.x2,
        y2=item.y2,
    )
    return crop_detection(rgb, detection_item)


def format_instance_summary(items: tuple[InstanceItem, ...]) -> str:
    if not items:
        return "No instances detected at the current confidence threshold."
    labels = [item.label for item in items]
    unique = list(dict.fromkeys(labels))
    joined = ", ".join(unique)
    return f"Detected {len(items)} instance(s): {joined}"
