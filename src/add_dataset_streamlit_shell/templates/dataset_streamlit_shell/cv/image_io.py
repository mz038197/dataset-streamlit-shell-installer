from __future__ import annotations

import random
import shutil
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from collections.abc import Callable
from typing import Literal
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image

SHELL_ROOT = Path(__file__).resolve().parents[1]
CV_DATA_DIR = SHELL_ROOT / "built-in-data" / "computer-vision"
MINI_TRAIN_DIR = CV_DATA_DIR / "mini-train"
EXAMPLES_DIR = CV_DATA_DIR / "examples"

CATS_DOGS_ZIP_URL = (
    "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
)

EXTRA_DEMO_URLS: dict[str, str] = {
    "coffee_mug.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_cup_of_coffee.JPG"
    ),
    "street_scene.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/"
        "Good_Food_Display_-_NCI_Visuals_Online.jpg/1280px-"
        "Good_Food_Display_-_NCI_Visuals_Online.jpg"
    ),
}

DEMO_MANIFEST: tuple[tuple[str, str, str], ...] = (
    ("dog.jpg", "Single dog — expect high-confidence canine label", "dog"),
    ("tabby_cat.jpg", "Single cat — expect feline label", "cat"),
    ("coffee_mug.jpg", "Everyday object — mug or cup-like label", "extra"),
    ("street_scene.jpg", "Busy scene — lower confidence / mixed context", "extra"),
)


@dataclass(frozen=True)
class DemoImageSpec:
    filename: str
    hint: str
    source: Literal["dataset", "extra"]


def demo_image_specs() -> list[DemoImageSpec]:
    return [DemoImageSpec(filename, hint, source) for filename, hint, source in DEMO_MANIFEST]


def examples_ready() -> bool:
    return all((EXAMPLES_DIR / spec.filename).exists() for spec in demo_image_specs())


def mini_train_ready() -> bool:
    cats = list((MINI_TRAIN_DIR / "cats").glob("*.jpg")) if (MINI_TRAIN_DIR / "cats").exists() else []
    dogs = list((MINI_TRAIN_DIR / "dogs").glob("*.jpg")) if (MINI_TRAIN_DIR / "dogs").exists() else []
    return len(cats) >= 10 and len(dogs) >= 10


def dataset_ready() -> bool:
    return examples_ready() and mini_train_ready()


def load_image_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def load_image_path(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_rgb_array(image: Image.Image, *, size: tuple[int, int] | None = None) -> np.ndarray:
    working = image.convert("RGB")
    if size is not None:
        working = working.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(working, dtype=np.uint8)


def overlay_heatmap(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    if heatmap.shape[:2] != rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = (alpha * colored + (1.0 - alpha) * rgb).astype(np.uint8)
    return blended


def download_sample_data(
    *,
    progress_callback: Callable[[str, float], None] | None = None,
) -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    MINI_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    (MINI_TRAIN_DIR / "cats").mkdir(parents=True, exist_ok=True)
    (MINI_TRAIN_DIR / "dogs").mkdir(parents=True, exist_ok=True)

    zip_path = CV_DATA_DIR / "cats_and_dogs_filtered.zip"
    extract_root = CV_DATA_DIR / "_cats_dogs_extract"

    if progress_callback:
        progress_callback("下載 cats_and_dogs_filtered 資料集…", 0.1)
    if not zip_path.exists():
        urlretrieve(CATS_DOGS_ZIP_URL, zip_path)

    if progress_callback:
        progress_callback("解壓縮並整理示範圖…", 0.35)
    if extract_root.exists():
        shutil.rmtree(extract_root)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_root)

    train_cats = sorted((extract_root / "cats_and_dogs_filtered" / "train" / "cats").glob("*.jpg"))
    train_dogs = sorted((extract_root / "cats_and_dogs_filtered" / "train" / "dogs").glob("*.jpg"))
    if not train_cats or not train_dogs:
        raise FileNotFoundError("cats_and_dogs_filtered archive did not contain expected train images.")

    shutil.copy2(train_dogs[0], EXAMPLES_DIR / "dog.jpg")
    shutil.copy2(train_cats[0], EXAMPLES_DIR / "tabby_cat.jpg")

    if progress_callback:
        progress_callback("下載額外示範圖…", 0.55)
    for filename, url in EXTRA_DEMO_URLS.items():
        target = EXAMPLES_DIR / filename
        if not target.exists():
            urlretrieve(url, target)

    if progress_callback:
        progress_callback("建立迷你訓練子集…", 0.75)
    _populate_mini_train(train_cats, train_dogs)

    if progress_callback:
        progress_callback("完成", 1.0)


def _populate_mini_train(train_cats: list[Path], train_dogs: list[Path], *, per_class: int = 40) -> None:
    rng = random.Random(42)
    cat_pick = rng.sample(train_cats, k=min(per_class, len(train_cats)))
    dog_pick = rng.sample(train_dogs, k=min(per_class, len(train_dogs)))

    cats_dir = MINI_TRAIN_DIR / "cats"
    dogs_dir = MINI_TRAIN_DIR / "dogs"
    for existing in cats_dir.glob("*.jpg"):
        existing.unlink()
    for existing in dogs_dir.glob("*.jpg"):
        existing.unlink()

    for index, path in enumerate(cat_pick, start=1):
        shutil.copy2(path, cats_dir / f"cat_{index:03d}.jpg")
    for index, path in enumerate(dog_pick, start=1):
        shutil.copy2(path, dogs_dir / f"dog_{index:03d}.jpg")


def list_example_images() -> list[Path]:
    return [EXAMPLES_DIR / spec.filename for spec in demo_image_specs() if (EXAMPLES_DIR / spec.filename).exists()]


def list_mini_train_images() -> tuple[list[Path], list[Path]]:
    cats = sorted((MINI_TRAIN_DIR / "cats").glob("*.jpg"))
    dogs = sorted((MINI_TRAIN_DIR / "dogs").glob("*.jpg"))
    return cats, dogs
