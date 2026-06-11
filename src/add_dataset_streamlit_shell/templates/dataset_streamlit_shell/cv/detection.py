from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_MODEL = "yolov8n.pt"

BOX_COLORS = (
    (26, 115, 232),
    (219, 68, 55),
    (15, 157, 88),
    (244, 180, 0),
    (171, 71, 188),
    (0, 172, 193),
)


@dataclass(frozen=True)
class DetectionItem:
    rank: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class DetectionResult:
    items: tuple[DetectionItem, ...]
    raw_count: int
    model_name: str
    conf_threshold: float


def load_yolo_model(model_name: str = DEFAULT_MODEL):
    from ultralytics import YOLO

    return YOLO(model_name)


def predict_detections(
    image: np.ndarray,
    *,
    conf_threshold: float = 0.25,
    model=None,
) -> DetectionResult:
    detector = model or load_yolo_model()
    outputs = detector.predict(image, conf=conf_threshold, verbose=False)
    boxes = outputs[0].boxes
    items: list[DetectionItem] = []
    if boxes is None or len(boxes) == 0:
        return DetectionResult(
            items=tuple(),
            raw_count=0,
            model_name=DEFAULT_MODEL,
            conf_threshold=conf_threshold,
        )

    order = np.argsort(-boxes.conf.cpu().numpy())
    for rank, index in enumerate(order, start=1):
        xyxy = boxes.xyxy[index].cpu().numpy().astype(int).tolist()
        class_id = int(boxes.cls[index].cpu().numpy())
        label = str(outputs[0].names[class_id])
        confidence = float(boxes.conf[index].cpu().numpy())
        items.append(
            DetectionItem(
                rank=rank,
                label=label,
                confidence=confidence,
                x1=int(xyxy[0]),
                y1=int(xyxy[1]),
                x2=int(xyxy[2]),
                y2=int(xyxy[3]),
            )
        )
    return DetectionResult(
        items=tuple(items),
        raw_count=len(items),
        model_name=DEFAULT_MODEL,
        conf_threshold=conf_threshold,
    )


def filter_detections(
    items: tuple[DetectionItem, ...],
    *,
    conf_threshold: float,
) -> tuple[DetectionItem, ...]:
    kept = [item for item in items if item.confidence >= conf_threshold]
    return tuple(
        DetectionItem(
            rank=index,
            label=item.label,
            confidence=item.confidence,
            x1=item.x1,
            y1=item.y1,
            x2=item.x2,
            y2=item.y2,
        )
        for index, item in enumerate(kept, start=1)
    )


def draw_detections(rgb: np.ndarray, items: tuple[DetectionItem, ...]) -> np.ndarray:
    canvas = rgb.copy()
    height, width = canvas.shape[:2]
    for index, item in enumerate(items):
        color = BOX_COLORS[index % len(BOX_COLORS)]
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


def crop_detection(rgb: np.ndarray, item: DetectionItem) -> np.ndarray:
    height, width = rgb.shape[:2]
    x1 = max(0, min(item.x1, width - 1))
    y1 = max(0, min(item.y1, height - 1))
    x2 = max(0, min(item.x2, width))
    y2 = max(0, min(item.y2, height))
    if x2 <= x1 or y2 <= y1:
        return rgb.copy()
    return rgb[y1:y2, x1:x2].copy()


def format_detection_summary(items: tuple[DetectionItem, ...]) -> str:
    if not items:
        return "No objects detected at the current confidence threshold."
    labels = [item.label for item in items]
    unique = list(dict.fromkeys(labels))
    joined = ", ".join(unique)
    return f"Detected {len(items)} object(s): {joined}"
