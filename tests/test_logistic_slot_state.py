from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ui.logistic_slot_state import (  # noqa: E402
    CHOICE_UNSET,
    DEFAULT_LAMBDA,
    LOSS_LOGLOSS,
    SCALE_MINMAX,
    SCALE_ZSCORE,
    STAGE_BOUNDARY,
    STAGE_POLY,
    apply_slot_write,
    can_write_train_request,
    consume_train_request,
    default_slot_state,
    empty_slot_state,
    empty_workspace_state,
    logistic_host_context_fragment,
    load_workspace_state,
    model_code_preview,
    model_download_files,
    model_sequential_compile_preview,
    save_workspace_state,
    slot_inspect_rows,
    slot_is_filled,
    slot_label,
    slots_are_complete,
    slot_signature,
    train_request_is_set,
    write_train_request,
)


def _complete(stage: str = STAGE_BOUNDARY, train_pct: int = 80) -> dict:
    incoming: dict = {"choices": {"split": train_pct}}
    if stage == STAGE_POLY:
        incoming["lambda_"] = DEFAULT_LAMBDA
    return apply_slot_write(default_slot_state(stage), incoming, stage=stage)


def test_empty_slot_state_locks_data_and_leaves_other_slots_unset() -> None:
    boundary = empty_slot_state(STAGE_BOUNDARY)
    poly = empty_slot_state(STAGE_POLY)
    assert boundary["choices"]["data"] == "admission"
    assert poly["choices"]["data"] == "microchip"
    assert boundary["choices"]["split"] is None
    assert boundary["choices"]["scale"] is None
    assert boundary["choices"]["linear"] is None
    assert boundary["choices"]["loss"] is None
    assert boundary["choices"]["opt"] is None
    assert boundary["alpha"] is None
    assert boundary["epochs"] is None
    assert boundary["lambda_"] is None
    assert slots_are_complete(boundary) is False
    assert slot_label("data", boundary, STAGE_BOUNDARY) == "內建大學錄取"
    assert slot_label("split", boundary, STAGE_BOUNDARY) == CHOICE_UNSET
    assert slot_is_filled("data", boundary) is True
    assert slot_is_filled("loss", boundary) is False


def test_default_slot_state_writes_zscore_without_split() -> None:
    boundary = default_slot_state(STAGE_BOUNDARY)
    poly = default_slot_state(STAGE_POLY)
    assert boundary["choices"]["split"] is None
    assert boundary["choices"]["scale"] == SCALE_ZSCORE
    assert boundary["choices"]["loss"] == LOSS_LOGLOSS
    assert boundary["alpha"] == 0.001
    assert boundary["epochs"] == 10000
    assert boundary["lambda_"] is None
    assert slots_are_complete(boundary) is False
    assert slot_label("loss", boundary, STAGE_BOUNDARY) == "對數損失"
    assert "from_logits" not in slot_label("loss", boundary, STAGE_BOUNDARY)
    assert "BinaryCrossentropy" not in slot_label("loss", boundary, STAGE_BOUNDARY)
    assert "Dense(1, linear)" in slot_label("linear", boundary, STAGE_BOUNDARY)
    assert poly["lambda_"] == DEFAULT_LAMBDA
    assert "λ=0.01" in slot_label("loss", poly, STAGE_POLY)
    assert slot_is_filled("loss", poly) is True
    assert slots_are_complete(poly) is False


def test_apply_slot_write_accepts_scale_alpha_lambda_rejects_locked() -> None:
    assembled = _complete(STAGE_POLY)
    changed = apply_slot_write(
        assembled,
        {
            "choices": {
                "scale": SCALE_MINMAX,
                "data": "upload",
                "linear": "dense8",
                "loss": "mse",
                "opt": "adam",
            },
            "alpha": 0.02,
            "epochs": 200,
            "lambda_": 1.0,
        },
        stage=STAGE_POLY,
    )
    assert changed["choices"]["scale"] == SCALE_MINMAX
    assert changed["choices"]["data"] == "microchip"
    assert changed["choices"]["linear"] == "dense1"
    assert changed["choices"]["loss"] == LOSS_LOGLOSS
    assert changed["choices"]["opt"] == "sgd"
    assert changed["alpha"] == pytest.approx(0.02)
    assert changed["epochs"] == 200
    assert changed["lambda_"] == pytest.approx(1.0)
    assert slot_signature(changed) != slot_signature(assembled)


def test_stage1_ignores_lambda_write() -> None:
    assembled = _complete(STAGE_BOUNDARY)
    changed = apply_slot_write(assembled, {"lambda_": 1.0}, stage=STAGE_BOUNDARY)
    assert changed["lambda_"] is None
    assert slots_are_complete(changed) is True


def test_poly_incomplete_without_lambda() -> None:
    state = default_slot_state(STAGE_POLY)
    state = apply_slot_write(state, {"choices": {"split": 80}, "lambda_": None}, stage=STAGE_POLY)
    state["lambda_"] = None
    assert slots_are_complete(state) is False
    assert slot_is_filled("loss", state) is False


def test_model_preview_uses_from_logits_binary_crossentropy() -> None:
    state = _complete(STAGE_BOUNDARY)
    sequential = model_sequential_compile_preview(state, stage=STAGE_BOUNDARY)
    preview = model_code_preview(state, stage=STAGE_BOUNDARY)
    assert 'Dense(1, activation="linear")' in sequential
    assert "BinaryCrossentropy(from_logits=True)" in sequential
    assert 'activation="sigmoid"' not in sequential
    assert sequential in preview
    assert "from_logits" not in slot_label("loss", state, STAGE_BOUNDARY)


def test_model_download_files_poly_includes_regularizer_and_phi() -> None:
    files = model_download_files(_complete(STAGE_POLY), stage=STAGE_POLY)
    assert files is not None
    assert "microchip_test.csv" in files
    assert "university_admission.csv" not in files
    script = files["train.py"].decode("utf-8")
    sequential = model_sequential_compile_preview(_complete(STAGE_POLY), stage=STAGE_POLY)
    assert sequential in script
    assert "kernel_regularizer" in script
    assert "l2(" in script
    assert "λ/(2m)" in script or "2m" in script
    assert "map_feature" in script or "degree=6" in script
    assert "BinaryCrossentropy(from_logits=True)" in script
    assert "train_test_split" in script
    assert "test precision" in script
    assert "test recall" in script
    assert "test F1" in script
    ast.parse(script)


def test_model_download_missing_when_incomplete() -> None:
    assert model_download_files(empty_slot_state(STAGE_BOUNDARY), stage=STAGE_BOUNDARY) is None
    assert model_download_files(default_slot_state(STAGE_BOUNDARY), stage=STAGE_BOUNDARY) is None


def test_workspace_round_trip_and_train_request(tmp_path: Path) -> None:
    workspace = empty_workspace_state()
    workspace[STAGE_BOUNDARY] = default_slot_state(STAGE_BOUNDARY)
    save_workspace_state(tmp_path, workspace)
    loaded = load_workspace_state(tmp_path)
    assert loaded[STAGE_BOUNDARY]["choices"]["scale"] == SCALE_ZSCORE
    assert loaded[STAGE_POLY]["choices"]["scale"] is None
    assert train_request_is_set(tmp_path) is False
    write_train_request(tmp_path)
    assert consume_train_request(tmp_path, allowed=True) is True
    assert train_request_is_set(tmp_path) is False


def test_can_write_train_request_needs_quiz_and_slots() -> None:
    state = _complete()
    assert can_write_train_request(state, quiz_unlocked=False) is False
    assert can_write_train_request(state, quiz_unlocked=True) is True
    assert can_write_train_request(state, quiz_unlocked=True, scale_errors=["bad"]) is False


def test_host_fragment_matches_slot_contract() -> None:
    text = logistic_host_context_fragment(
        slots_path="logistic_slots.json",
        request_path="logistic_train_request.json",
    )
    assert "組模型預設不要寫 split" in text
    assert "boundary" in text
    assert "poly" in text
    assert "logloss" in text
    assert "lambda_" in text
    assert "BinaryCrossentropy(from_logits=True)" in text
    assert "測試集混淆矩陣" in text
    assert "測試集機率對照" not in text
    assert "測試集 precision／recall／F1" in text
    assert "禁止說看不到測試 Cost" in text
    assert "禁止說看不到測試集 precision／recall／F1" in text
    assert "測試 Cost 只算對數損失" in text


def test_linear_inspect_has_phi_row_only_on_poly_stage() -> None:
    poly = dict(slot_inspect_rows("linear", _complete(STAGE_POLY), stage=STAGE_POLY))
    boundary = dict(slot_inspect_rows("linear", _complete(), stage=STAGE_BOUNDARY))
    assert poly["φ"].startswith("degree=6")
    assert "φ" not in boundary
    assert "logits" not in str(poly)
    assert "logits" not in str(boundary)
    assert poly["z"] == r"z = w·φ(x)+b"
    assert boundary["z"] == r"z = w·x+b"


def test_download_zip_is_stage_scoped() -> None:
    files = model_download_files(_complete(), stage=STAGE_BOUNDARY)
    assert files is not None
    assert "university_admission.csv" in files
    toml = files["pyproject.toml"].decode("utf-8")
    assert "tensorflow-cpu" in toml
    assert "streamlit" not in toml
