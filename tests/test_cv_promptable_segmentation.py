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

from dataset_streamlit_shell.cv.image_io import (
    SAM3_GDRIVE_FILE_ID,
    download_sam3_weights,
    sam3_weights_path,
    sam3_weights_ready,
    sam_demo_specs,
)
from dataset_streamlit_shell.cv.promptable_segmentation import (
    DEFAULT_CONF,
    DEFAULT_MODEL,
    TextPromptItem,
    draw_promptable_results,
    filter_promptable_items,
    format_promptable_summary,
    parse_text_prompts,
)


def _sample_item(*, rank: int = 1, prompt: str = "dog") -> TextPromptItem:
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:40, 20:60] = True
    area = int(mask.sum())
    return TextPromptItem(
        rank=rank,
        prompt=prompt,
        confidence=0.82,
        x1=20,
        y1=10,
        x2=60,
        y2=40,
        mask=mask,
        area=area,
        coverage=float(area) / float(mask.size),
    )


def test_default_model_is_sam3() -> None:
    assert DEFAULT_MODEL == "sam3.pt"
    assert DEFAULT_CONF == 0.5


def test_parse_text_prompts_splits_lines() -> None:
    prompts = parse_text_prompts("dog\n\nperson\ncar\n")
    assert prompts == ("dog", "person", "car")


def test_filter_promptable_items_applies_threshold() -> None:
    items = (
        _sample_item(rank=1, prompt="person"),
        TextPromptItem(
            rank=2,
            prompt="car",
            confidence=0.18,
            x1=0,
            y1=0,
            x2=10,
            y2=10,
            mask=np.zeros((60, 80), dtype=bool),
            area=0,
            coverage=0.0,
        ),
    )
    filtered = filter_promptable_items(items, conf_threshold=0.25)
    assert len(filtered) == 1
    assert filtered[0].prompt == "person"
    assert filtered[0].rank == 1


def test_draw_promptable_results_preserves_shape() -> None:
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    drawn = draw_promptable_results(rgb, (_sample_item(),))
    assert drawn.shape == rgb.shape
    assert drawn.dtype == np.uint8


def test_format_promptable_summary_lists_prompts() -> None:
    items = (_sample_item(prompt="dog"), _sample_item(rank=2, prompt="person"))
    summary = format_promptable_summary(items)
    assert "dog" in summary
    assert "person" in summary
    assert "2 region" in summary


def test_sam_demo_specs_has_six_entries() -> None:
    assert len(sam_demo_specs()) == 6


def test_sam3_weights_ready_false_when_missing(tmp_path: Path, monkeypatch) -> None:
    from dataset_streamlit_shell.cv import image_io

    monkeypatch.setattr(image_io, "SAM3_WEIGHTS_PATH", tmp_path / "models" / "sam3.pt")
    monkeypatch.setattr(image_io, "SHELL_ROOT", tmp_path / "shell")
    assert sam3_weights_ready() is False
    assert sam3_weights_path() is None


def test_sam3_gdrive_file_id_is_configured() -> None:
    assert SAM3_GDRIVE_FILE_ID == "18aap-5Ky9gQ8DJoh15LnK8JBeOAWi-cj"


def test_download_sam3_weights_skips_when_valid(tmp_path: Path, monkeypatch) -> None:
    from dataset_streamlit_shell.cv import image_io

    weights = tmp_path / "sam3.pt"
    monkeypatch.setattr(image_io, "SAM3_MODELS_DIR", tmp_path)
    monkeypatch.setattr(image_io, "SAM3_WEIGHTS_PATH", weights)
    monkeypatch.setattr(image_io, "_sam3_weight_file_valid", lambda path: path == weights)

    messages: list[tuple[str, float]] = []
    download_sam3_weights(progress_callback=lambda message, value: messages.append((message, value)))
    assert messages == [("SAM 3 權重已存在", 1.0)]


def test_download_sam3_weights_writes_file(tmp_path: Path, monkeypatch) -> None:
    from dataset_streamlit_shell.cv import image_io

    models_dir = tmp_path / "models"
    weights = models_dir / "sam3.pt"
    monkeypatch.setattr(image_io, "SAM3_MODELS_DIR", models_dir)
    monkeypatch.setattr(image_io, "SAM3_WEIGHTS_PATH", weights)
    monkeypatch.setattr(image_io, "SAM3_MIN_WEIGHT_BYTES", 10)

    def fake_download(
        file_id: str,
        output_path: Path,
        progress_callback,
    ) -> None:
        assert file_id == SAM3_GDRIVE_FILE_ID
        output_path.write_bytes(b"PK" + b"x" * 10)
        if progress_callback:
            progress_callback("整理檔案…", 0.92)

    monkeypatch.setattr(image_io, "_download_gdrive_file", fake_download)
    download_sam3_weights()
    assert weights.read_bytes().startswith(b"PK")
