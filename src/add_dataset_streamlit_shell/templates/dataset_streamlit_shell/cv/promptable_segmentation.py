from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dataset_streamlit_shell.cv.image_io import (
    SAM3_MODELS_DIR,
    sam3_weights_path,
    sam3_weights_ready,
)

DEFAULT_MODEL = "sam3.pt"
DEFAULT_CONF = 0.5

MASK_COLOR_RGB = (0, 120, 255)
BBOX_COLOR_RGB = (0, 255, 0)

SHELL_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TextPromptItem:
    rank: int
    prompt: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    mask: np.ndarray
    area: int
    coverage: float

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class PromptableResult:
    items: tuple[TextPromptItem, ...]
    raw_count: int
    prompts: tuple[str, ...]
    conf_threshold: float
    model_name: str
    annotated_image: np.ndarray


def parse_text_prompts(raw: str) -> tuple[str, ...]:
    prompts = tuple(line.strip() for line in raw.splitlines() if line.strip())
    return prompts


def load_sam3_predictor(
    *,
    conf_threshold: float = DEFAULT_CONF,
    weights_path: Path | None = None,
):
    from ultralytics.models.sam import SAM3SemanticPredictor

    model_path = weights_path or sam3_weights_path()
    if model_path is None:
        raise FileNotFoundError(
            "sam3.pt not found. Place weights at "
            f"{SAM3_MODELS_DIR / DEFAULT_MODEL} or the shell root."
        )
    overrides = {
        "conf": conf_threshold,
        "task": "segment",
        "mode": "predict",
        "model": str(model_path),
        "save": False,
    }
    return SAM3SemanticPredictor(overrides=overrides)


def _image_path_for_predictor(image: np.ndarray, *, source_path: Path | None = None) -> str:
    if source_path is not None and source_path.exists():
        return str(source_path)
    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(temp_path), bgr)
    return str(temp_path)


def predict_text_masks(
    image: np.ndarray,
    prompts: tuple[str, ...],
    *,
    predictor=None,
    conf_threshold: float = DEFAULT_CONF,
    source_path: Path | None = None,
) -> PromptableResult:
    if not prompts:
        raise ValueError("At least one text prompt is required.")

    model = predictor or load_sam3_predictor(conf_threshold=conf_threshold)
    image_path = _image_path_for_predictor(image, source_path=source_path)
    model.set_image(image_path)
    outputs = model(text=list(prompts), return_masks=True, save=False)
    return build_promptable_result(
        outputs,
        prompts=prompts,
        image=image,
        conf_threshold=conf_threshold,
    )


def build_promptable_result(
    outputs,
    *,
    prompts: tuple[str, ...],
    image: np.ndarray,
    conf_threshold: float,
) -> PromptableResult:
    height, width = image.shape[:2]
    total_pixels = int(height * width)
    items: list[TextPromptItem] = []
    raw_count = 0

    if not isinstance(outputs, list):
        outputs = [outputs]

    for result_index, result in enumerate(outputs):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        raw_count += len(result.boxes)
        prompt = prompts[result_index] if result_index < len(prompts) else prompts[-1]
        for box_index in range(len(result.boxes)):
            confidence = float(result.boxes.conf[box_index].cpu().numpy())
            if confidence < conf_threshold:
                continue
            xyxy = result.boxes.xyxy[box_index].cpu().numpy().astype(int).tolist()

            mask_bool = np.zeros((height, width), dtype=bool)
            if result.masks is not None and box_index < len(result.masks):
                mask_data = result.masks.data[box_index].cpu().numpy().astype(np.float32)
                mask_resized = cv2.resize(mask_data, (width, height))
                mask_bool = mask_resized > 0.5

            area = int(mask_bool.sum())
            items.append(
                TextPromptItem(
                    rank=len(items) + 1,
                    prompt=prompt,
                    confidence=confidence,
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    mask=mask_bool,
                    area=area,
                    coverage=float(area) / float(total_pixels) if total_pixels else 0.0,
                )
            )

    annotated = draw_promptable_results(image, tuple(items))
    return PromptableResult(
        items=tuple(items),
        raw_count=raw_count,
        prompts=prompts,
        conf_threshold=conf_threshold,
        model_name=DEFAULT_MODEL,
        annotated_image=annotated,
    )


def filter_promptable_items(
    items: tuple[TextPromptItem, ...],
    *,
    conf_threshold: float,
) -> tuple[TextPromptItem, ...]:
    kept = [item for item in items if item.confidence >= conf_threshold]
    return tuple(
        TextPromptItem(
            rank=index,
            prompt=item.prompt,
            confidence=item.confidence,
            x1=item.x1,
            y1=item.y1,
            x2=item.x2,
            y2=item.y2,
            mask=item.mask,
            area=item.area,
            coverage=item.coverage,
        )
        for index, item in enumerate(kept, start=1)
    )


def draw_promptable_results(
    image: np.ndarray,
    items: tuple[TextPromptItem, ...],
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    canvas = image.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    mask_color = np.array(MASK_COLOR_RGB, dtype=np.float32)
    bbox_color = BBOX_COLOR_RGB

    for item in items:
        if item.mask.any():
            blended = canvas[item.mask].astype(np.float32)
            canvas[item.mask] = (
                blended * (1.0 - alpha) + mask_color * alpha
            ).astype(np.uint8)

    for item in items:
        x1 = max(0, min(item.x1, canvas.shape[1] - 1))
        y1 = max(0, min(item.y1, canvas.shape[0] - 1))
        x2 = max(0, min(item.x2, canvas.shape[1] - 1))
        y2 = max(0, min(item.y2, canvas.shape[0] - 1))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bbox_color, 2)
        caption = f"{item.confidence:.2f}"
        cv2.putText(
            canvas,
            caption,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            bbox_color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def highlight_prompt_item(image: np.ndarray, item: TextPromptItem) -> np.ndarray:
    highlighted = image.copy()
    dimmed = (image.astype(np.float32) * 0.35).astype(np.uint8)
    highlighted[~item.mask] = dimmed[~item.mask]
    tint = np.zeros_like(highlighted)
    tint[item.mask] = MASK_COLOR_RGB
    highlighted[item.mask] = (
        0.55 * tint[item.mask].astype(np.float32) + 0.45 * highlighted[item.mask].astype(np.float32)
    ).astype(np.uint8)
    return highlighted


def crop_prompt_item(image: np.ndarray, item: TextPromptItem) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = max(0, min(item.x1, width - 1))
    y1 = max(0, min(item.y1, height - 1))
    x2 = max(0, min(item.x2, width))
    y2 = max(0, min(item.y2, height))
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    return image[y1:y2, x1:x2].copy()


def format_promptable_summary(items: tuple[TextPromptItem, ...]) -> str:
    if not items:
        return "No concepts matched the text prompts at the current confidence threshold."
    prompts = list(dict.fromkeys(item.prompt for item in items))
    joined = ", ".join(prompts[:4])
    return f"Matched {len(items)} region(s) for: {joined}"
