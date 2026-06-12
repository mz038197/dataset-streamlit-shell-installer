from __future__ import annotations

import random
import shutil
import threading
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
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/"
        "City_of_London_1.jpg/1280px-City_of_London_1.jpg"
    ),
    "desk_objects.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/"
        "Desk-lamp-and-laptop.jpg/1280px-Desk-lamp-and-laptop.jpg"
    ),
}

DEMO_MANIFEST: tuple[tuple[str, str, str], ...] = (
    ("dog.jpg", "Single dog — expect high-confidence canine label", "dog"),
    ("tabby_cat.jpg", "Single cat — expect feline label", "cat"),
    ("coffee_mug.jpg", "Everyday object — mug or cup-like label", "extra"),
    ("street_scene.jpg", "Busy scene — lower confidence / mixed context", "extra"),
)

DETECTION_DEMO_MANIFEST: tuple[tuple[str, str], ...] = (
    ("cattle_show.jpg", "Livestock show — multiple persons and cows"),
    ("car_and_bus.jpg", "Road scene — car and bus"),
    ("horse_rider.jpg", "Equestrian scene — person and horse"),
    ("airplanes.jpg", "Air show — multiple airplanes"),
)

SEMANTIC_DEMO_MANIFEST: tuple[tuple[str, str], ...] = (
    ("street_scene.jpg", "Street scene — person / car / road regions"),
    ("desk_objects.jpg", "Desk scene — furniture and object regions"),
    ("dog.jpg", "Single dog — foreground vs background"),
    ("cat_and_dog.jpg", "Two animals — same-class pixels merge by label"),
)

INSTANCE_DEMO_MANIFEST: tuple[tuple[str, str], ...] = (
    ("dog.jpg", "Three puppies — multiple dog instances"),
    ("three_cats.jpg", "Three cats — multiple cat instances"),
    ("street_scene.jpg", "Street scene — multiple persons / cars"),
    ("coffee_mug.jpg", "Coffee set — cup, saucer, and spoon"),
)

SAM_DEMO_MANIFEST: tuple[tuple[str, str, str], ...] = (
    ("motorcycle_family.jpg", "Family with motorcycle — try prompts: motorcycle / person / helmet", "motorcycle\nperson\nhelmet"),
    ("wine_dinner.jpg", "Dinner scene — try prompts: person / bottle / wine glass", "person\nbottle\nwine glass"),
    ("soccer_stadium.jpg", "Soccer stadium — try prompts: person / jersey", "person\njersey"),
    ("pony_ride.jpg", "Pony ride — try prompts: horse / person", "horse\nperson"),
)

SAM3_MODELS_DIR = CV_DATA_DIR / "models"
SAM3_WEIGHTS_PATH = SAM3_MODELS_DIR / "sam3.pt"
SAM3_GDRIVE_FILE_ID = "18aap-5Ky9gQ8DJoh15LnK8JBeOAWi-cj"
SAM3_MIN_WEIGHT_BYTES = 3_000_000_000
SAM3_EXPECTED_WEIGHT_BYTES = 3_450_000_000


@dataclass(frozen=True)
class DemoImageSpec:
    filename: str
    hint: str
    source: Literal["dataset", "extra"]


def demo_image_specs() -> list[DemoImageSpec]:
    return [DemoImageSpec(filename, hint, source) for filename, hint, source in DEMO_MANIFEST]


@dataclass(frozen=True)
class DetectionDemoSpec:
    filename: str
    hint: str


def detection_demo_specs() -> list[DetectionDemoSpec]:
    return [DetectionDemoSpec(filename, hint) for filename, hint in DETECTION_DEMO_MANIFEST]


def detection_examples_ready() -> bool:
    return all((EXAMPLES_DIR / spec.filename).exists() for spec in detection_demo_specs())


@dataclass(frozen=True)
class SemanticDemoSpec:
    filename: str
    hint: str


def semantic_demo_specs() -> list[SemanticDemoSpec]:
    return [SemanticDemoSpec(filename, hint) for filename, hint in SEMANTIC_DEMO_MANIFEST]


def semantic_examples_ready() -> bool:
    return all((EXAMPLES_DIR / spec.filename).exists() for spec in semantic_demo_specs())


@dataclass(frozen=True)
class InstanceDemoSpec:
    filename: str
    hint: str


def instance_demo_specs() -> list[InstanceDemoSpec]:
    return [InstanceDemoSpec(filename, hint) for filename, hint in INSTANCE_DEMO_MANIFEST]


def instance_examples_ready() -> bool:
    return all((EXAMPLES_DIR / spec.filename).exists() for spec in instance_demo_specs())


@dataclass(frozen=True)
class SamDemoSpec:
    filename: str
    hint: str
    suggested_prompts: str


def sam_demo_specs() -> list[SamDemoSpec]:
    return [
        SamDemoSpec(filename, hint, suggested_prompts)
        for filename, hint, suggested_prompts in SAM_DEMO_MANIFEST
    ]


def sam_examples_ready() -> bool:
    return all((EXAMPLES_DIR / spec.filename).exists() for spec in sam_demo_specs())


def sam3_weights_ready() -> bool:
    return sam3_weights_path() is not None


def sam3_weights_path() -> Path | None:
    if _sam3_weight_file_valid(SAM3_WEIGHTS_PATH):
        return SAM3_WEIGHTS_PATH
    fallback = SHELL_ROOT / "sam3.pt"
    if _sam3_weight_file_valid(fallback):
        return fallback
    return None


def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.1f} GB"


def _sam3_weight_file_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size < SAM3_MIN_WEIGHT_BYTES:
        return False
    with path.open("rb") as handle:
        return handle.read(2) == b"PK"


def _download_gdrive_file(
    file_id: str,
    output_path: Path,
    progress_callback: Callable[[str, float], None] | None,
) -> None:
    import gdown

    if progress_callback:
        progress_callback("連線 Google Drive…", 0.05)

    url = f"https://drive.google.com/uc?id={file_id}"
    done = threading.Event()
    errors: list[BaseException] = []

    def run_download() -> None:
        try:
            gdown.download(url, str(output_path), quiet=True, resume=True)
        except BaseException as exc:  # noqa: BLE001 - propagate download failures
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run_download, daemon=True)
    thread.start()

    last_value = -1.0
    while not done.wait(timeout=0.5):
        if progress_callback is None:
            continue
        downloaded = output_path.stat().st_size if output_path.exists() else 0
        ratio = min(downloaded / SAM3_EXPECTED_WEIGHT_BYTES, 0.89)
        value = 0.05 + ratio * 0.85
        pct = int(ratio * 100)
        message = (
            f"下載中… {_format_gib(downloaded)} / "
            f"{_format_gib(SAM3_EXPECTED_WEIGHT_BYTES)}（{pct}%）"
        )
        if value - last_value >= 0.01:
            progress_callback(message, value)
            last_value = value

    thread.join()
    if errors:
        output_path.unlink(missing_ok=True)
        raise errors[0]

    if progress_callback:
        progress_callback("整理檔案…", 0.92)


def download_sam3_weights(
    *,
    progress_callback: Callable[[str, float], None] | None = None,
    gdrive_file_id: str = SAM3_GDRIVE_FILE_ID,
) -> None:
    if _sam3_weight_file_valid(SAM3_WEIGHTS_PATH):
        if progress_callback:
            progress_callback("SAM 3 權重已存在", 1.0)
        return

    SAM3_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = Path(f"{SAM3_WEIGHTS_PATH}.partial")
    if partial_path.exists():
        partial_path.unlink()

    _download_gdrive_file(gdrive_file_id, partial_path, progress_callback)

    if progress_callback:
        progress_callback("驗證檔案…", 0.97)
    if not _sam3_weight_file_valid(partial_path):
        partial_path.unlink(missing_ok=True)
        raise ValueError("下載的 sam3.pt 大小或格式不正確，請稍後重試或手動放置權重。")

    partial_path.replace(SAM3_WEIGHTS_PATH)
    if progress_callback:
        progress_callback("下載完成", 1.0)


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
    _save_cat_and_dog_composite(train_cats[1], train_dogs[1], EXAMPLES_DIR / "cat_and_dog.jpg")

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


def _save_cat_and_dog_composite(cat_path: Path, dog_path: Path, target: Path) -> None:
    cat = Image.open(cat_path).convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
    dog = Image.open(dog_path).convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
    composite = Image.new("RGB", (640, 320))
    composite.paste(cat, (0, 0))
    composite.paste(dog, (320, 0))
    composite.save(target, format="JPEG")


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
