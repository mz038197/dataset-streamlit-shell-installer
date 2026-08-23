from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ui.lr_slot_state import (  # noqa: E402
    CHOICE_UNSET,
    SCALE_MAXDIV,
    SCALE_MEAN,
    SCALE_MINMAX,
    SCALE_ZSCORE,
    SLOT_IDS,
    STAGE_MULTIPLE,
    STAGE_SIMPLE,
    apply_slot_write,
    build_lr_page_snapshot,
    can_write_train_request,
    consume_train_request,
    default_slot_state,
    empty_slot_state,
    empty_workspace_state,
    load_workspace_state,
    lr_host_context_fragment,
    lr_slots_path,
    lr_train_request_path,
    model_code_preview,
    normalize_slot_state,
    save_workspace_state,
    scale_inspect,
    scale_method_errors,
    slot_button_label,
    slot_inspect_rows,
    slot_is_filled,
    slot_label,
    slots_are_complete,
    slot_signature,
    split_frame_by_train_pct,
    train_request_is_set,
    write_train_request,
)


def _complete(stage: str = STAGE_SIMPLE, train_pct: int = 80) -> dict:
    return apply_slot_write(
        default_slot_state(stage),
        {"choices": {"split": train_pct}},
        stage=stage,
    )


def test_empty_slot_state_locks_data_and_leaves_other_slots_unset() -> None:
    simple = empty_slot_state(STAGE_SIMPLE)
    multiple = empty_slot_state(STAGE_MULTIPLE)
    assert simple["choices"]["data"] == "restaurant"
    assert multiple["choices"]["data"] == "housing"
    assert simple["choices"]["split"] is None
    assert simple["choices"]["scale"] is None
    assert simple["choices"]["linear"] is None
    assert simple["choices"]["loss"] is None
    assert simple["choices"]["opt"] is None
    assert simple["alpha"] is None
    assert simple["epochs"] is None
    assert slots_are_complete(simple) is False
    assert slot_label("data", simple, STAGE_SIMPLE) == "內建餐廳獲利"
    assert slot_label("split", simple, STAGE_SIMPLE) == CHOICE_UNSET
    assert slot_label("scale", simple, STAGE_SIMPLE) == CHOICE_UNSET
    assert slot_is_filled("data", simple) is True
    assert slot_is_filled("split", simple) is False
    assert slot_is_filled("scale", simple) is False


def test_default_slot_state_assembles_zscore_and_locked_kinds() -> None:
    simple = default_slot_state(STAGE_SIMPLE)
    multiple = default_slot_state(STAGE_MULTIPLE)
    assert simple["choices"] == {
        "data": "restaurant",
        "split": None,
        "scale": SCALE_ZSCORE,
        "linear": "dense1",
        "loss": "mse",
        "opt": "sgd",
    }
    assert simple["alpha"] == 0.01
    assert simple["epochs"] == 1500
    assert slots_are_complete(simple) is False
    assert multiple["choices"]["data"] == "housing"
    assert multiple["choices"]["scale"] == SCALE_ZSCORE
    assert multiple["alpha"] == 0.1
    assert multiple["epochs"] == 1000
    assert slot_label("scale", simple, STAGE_SIMPLE) == "Z分數正規化"
    assert "Dense(1, linear)" in slot_label("linear", simple, STAGE_SIMPLE)
    assert "α=0.01" in slot_label("opt", simple, STAGE_SIMPLE)


def test_default_write_does_not_set_train_request(tmp_path: Path) -> None:
    workspace = empty_workspace_state()
    workspace[STAGE_SIMPLE] = default_slot_state(STAGE_SIMPLE)
    save_workspace_state(tmp_path, workspace)
    loaded = load_workspace_state(tmp_path)
    assert loaded[STAGE_SIMPLE]["choices"]["scale"] == SCALE_ZSCORE
    assert loaded[STAGE_MULTIPLE]["choices"]["scale"] is None
    assert train_request_is_set(tmp_path) is False


def test_apply_slot_write_accepts_scale_and_alpha_rejects_locked_kinds() -> None:
    assembled = _complete()
    changed = apply_slot_write(
        assembled,
        {
            "choices": {
                "scale": SCALE_MINMAX,
                "data": "upload",
                "linear": "dense8",
                "loss": "mae",
                "opt": "adam",
            },
            "alpha": 0.2,
            "epochs": 80,
        },
        stage=STAGE_SIMPLE,
    )
    assert changed["choices"]["scale"] == SCALE_MINMAX
    assert changed["choices"]["data"] == "restaurant"
    assert changed["choices"]["linear"] == "dense1"
    assert changed["choices"]["loss"] == "mse"
    assert changed["choices"]["opt"] == "sgd"
    assert changed["alpha"] == pytest.approx(0.2)
    assert changed["epochs"] == 80
    assert slot_signature(changed) != slot_signature(assembled)


def test_model_code_preview_follows_scale_and_feature_count() -> None:
    empty = empty_slot_state()
    assert "還沒選齊" in model_code_preview(empty, stage=STAGE_SIMPLE)

    simple = _complete()
    zscore = model_code_preview(simple, stage=STAGE_SIMPLE)
    assert "StandardScaler().fit_transform(x)" in zscore
    assert "Input(shape=(1,))" in zscore
    assert 'Dense(1, activation="linear")' in zscore
    assert 'loss="mse"' in zscore
    assert "SGD(learning_rate=0.01)" in zscore

    simple["choices"]["scale"] = SCALE_MAXDIV
    assert "x = x / x.max()" in model_code_preview(simple, stage=STAGE_SIMPLE)
    simple["choices"]["scale"] = SCALE_MINMAX
    assert "MinMaxScaler().fit_transform(x)" in model_code_preview(simple, stage=STAGE_SIMPLE)
    simple["choices"]["scale"] = SCALE_MEAN
    assert "(x - x.mean()) / (x.max() - x.min())" in model_code_preview(
        simple, stage=STAGE_SIMPLE
    )

    multiple = _complete(STAGE_MULTIPLE)
    multi_preview = model_code_preview(multiple, stage=STAGE_MULTIPLE)
    assert "Input(shape=(4,))" in multi_preview
    assert "SGD(learning_rate=0.1)" in multi_preview
    assert "train_test_split" not in zscore
    assert "train_test_split" not in multi_preview


def test_scale_inspect_has_formula_range_and_condition() -> None:
    info = scale_inspect(SCALE_MAXDIV)
    assert "x / x_max" in info["formula"].replace(" ", "") or "x / x_max" in info["formula"]
    assert "0" in info["range"]
    assert "≥ 0" in info["condition"] or ">= 0" in info["condition"]
    rows = slot_inspect_rows("scale", _complete(), stage=STAGE_SIMPLE)
    keys = [key for key, _ in rows]
    assert "公式" in keys
    assert "範圍" in keys
    assert "條件" in keys


def test_can_write_train_request_needs_complete_slots_and_quiz() -> None:
    empty = empty_slot_state()
    assembled = _complete()
    assert can_write_train_request(empty, quiz_unlocked=True) is False
    assert can_write_train_request(assembled, quiz_unlocked=False) is False
    assert can_write_train_request(assembled, quiz_unlocked=True) is True
    assert (
        can_write_train_request(
            assembled,
            quiz_unlocked=True,
            scale_errors=["正規化（除以最大）要求訓練特徵 x≥0"],
        )
        is False
    )


def test_consume_train_request_only_when_allowed(tmp_path: Path) -> None:
    write_train_request(tmp_path)
    assert consume_train_request(tmp_path, allowed=False) is False
    assert train_request_is_set(tmp_path) is False

    write_train_request(tmp_path)
    assert consume_train_request(tmp_path, allowed=True) is True
    assert train_request_is_set(tmp_path) is False


def test_maxdiv_rejects_negative_training_features() -> None:
    frame = pd.DataFrame({"x": [-1.0, 0.0, 2.0], "z": [1.0, 2.0, 3.0]})
    errors = scale_method_errors(SCALE_MAXDIV, frame, ["x"])
    assert errors
    assert "負值" in errors[0]
    assert scale_method_errors(SCALE_MAXDIV, frame, ["z"]) == []
    assert scale_method_errors(SCALE_ZSCORE, frame, ["x"]) == []


def test_two_stages_keep_independent_state_and_signatures(tmp_path: Path) -> None:
    workspace = empty_workspace_state()
    workspace[STAGE_SIMPLE] = default_slot_state(STAGE_SIMPLE)
    save_workspace_state(tmp_path, workspace)
    loaded = load_workspace_state(tmp_path)
    loaded[STAGE_SIMPLE] = apply_slot_write(
        loaded[STAGE_SIMPLE],
        {"choices": {"scale": SCALE_MEAN}, "alpha": 0.05, "epochs": 100},
        stage=STAGE_SIMPLE,
    )
    save_workspace_state(tmp_path, loaded)
    again = load_workspace_state(tmp_path)
    assert again[STAGE_SIMPLE]["choices"]["scale"] == SCALE_MEAN
    assert again[STAGE_MULTIPLE]["choices"]["scale"] is None
    assert slot_signature(again[STAGE_SIMPLE]) != slot_signature(again[STAGE_MULTIPLE])


def test_load_keeps_other_stage_when_file_omits_it(tmp_path: Path) -> None:
    previous = empty_workspace_state()
    previous[STAGE_MULTIPLE] = default_slot_state(STAGE_MULTIPLE)
    lr_slots_path(tmp_path).write_text(
        json.dumps({"simple": default_slot_state(STAGE_SIMPLE)}),
        encoding="utf-8",
    )
    loaded = load_workspace_state(tmp_path, previous)
    assert loaded[STAGE_SIMPLE]["choices"]["scale"] == SCALE_ZSCORE
    assert loaded[STAGE_MULTIPLE]["choices"]["data"] == "housing"


def test_page_snapshot_includes_open_slot_and_preview() -> None:
    text = build_lr_page_snapshot(
        stage=STAGE_SIMPLE,
        state=_complete(),
        open_slot="scale",
        quiz_unlocked=False,
        scale_errors=[],
        artifact_note="此學習階段目前沒有訓練結果。",
        slots_path="workspace/lr_slots.json",
        request_path="workspace/lr_train_request.json",
    )
    assert "學生正打開的決策槽：特徵縮放" in text
    assert "可否寫訓練請求：否" in text
    assert "StandardScaler().fit_transform(x)" in text


def test_host_context_forbids_exec_and_locked_kind_changes() -> None:
    text = lr_host_context_fragment(
        slots_path="workspace/lr_slots.json",
        request_path="workspace/lr_train_request.json",
    )
    assert "lr_slots.json" in text
    assert "lr_train_request.json" in text
    assert "不要同時寫訓練請求" in text
    assert "不要自行 exec" in text
    assert "鎖定槽" in text
    assert "nn_form.json" in text
    assert "代填" in text
    assert "全是 null" not in text
    assert "輸入資料進頁即為該階段鎖定值" in text
    assert "不必為了槽齊而寫 data" in text
    assert "六個決策槽" in text
    assert "訓練／測試切分" in text
    assert "組模型預設不要寫 split" in text
    assert "choices.split" in text
    assert "沒有下拉選單" in text
    assert "不要叫學生自己選" in text
    assert "打勾符號" in text
    assert "✅" not in text


def test_slot_button_label_shows_current_choice_only_when_complete() -> None:
    empty = empty_slot_state(STAGE_SIMPLE)
    assert slot_button_label("data", empty, STAGE_SIMPLE) == "輸入資料\n內建餐廳獲利"
    assert slot_button_label("split", empty, STAGE_SIMPLE) == "訓練／測試切分"
    assert slot_button_label("scale", empty, STAGE_SIMPLE) == "特徵縮放"
    assert "尚未選擇" not in slot_button_label("scale", empty, STAGE_SIMPLE)
    assembled = _complete()
    assert slot_button_label("scale", assembled, STAGE_SIMPLE) == "特徵縮放\nZ分數正規化"
    assert slot_button_label("split", assembled, STAGE_SIMPLE) == "訓練／測試切分\n訓練 80%"
    assert "α=0.01" in slot_button_label("opt", assembled, STAGE_SIMPLE)


def test_opt_is_incomplete_until_alpha_and_epochs_are_written() -> None:
    state = empty_slot_state(STAGE_SIMPLE)
    state["choices"]["opt"] = "sgd"
    assert slot_is_filled("opt", state) is False
    assert slot_button_label("opt", state, STAGE_SIMPLE) == "優化器"
    state["alpha"] = 0.01
    state["epochs"] = 1500
    assert slot_is_filled("opt", state) is True


def test_data_selection_detail_shows_builtin_table_and_row_count_before_assemble() -> None:
    rows = dict(
        slot_inspect_rows(
            "data",
            empty_slot_state(STAGE_SIMPLE),
            stage=STAGE_SIMPLE,
            row_count=97,
        )
    )
    assert rows["目前選擇"] == "內建餐廳獲利"
    assert rows["列數"] == "97"
    assert rows["x"] == "城市人口_萬人"
    assert "尚未選擇" not in rows["目前選擇"]
    housing = dict(
        slot_inspect_rows(
            "data",
            empty_slot_state(STAGE_MULTIPLE),
            stage=STAGE_MULTIPLE,
            row_count=47,
        )
    )
    assert housing["目前選擇"] == "內建房價四特徵"
    assert housing["列數"] == "47"


def test_normalize_backfills_locked_data_when_json_has_null() -> None:
    raw = {
        "choices": {
            "data": None,
            "scale": None,
            "linear": None,
            "loss": None,
            "opt": None,
        },
        "alpha": None,
        "epochs": None,
    }
    simple = normalize_slot_state(raw, stage=STAGE_SIMPLE)
    multiple = normalize_slot_state(raw, stage=STAGE_MULTIPLE)
    assert simple["choices"]["data"] == "restaurant"
    assert multiple["choices"]["data"] == "housing"
    assert simple["choices"]["scale"] is None


def test_load_backfills_data_from_legacy_null_json(tmp_path: Path) -> None:
    lr_slots_path(tmp_path).write_text(
        json.dumps(
            {
                "simple": {
                    "choices": {
                        "data": None,
                        "scale": None,
                        "linear": None,
                        "loss": None,
                        "opt": None,
                    }
                },
                "multiple": {
                    "choices": {
                        "data": None,
                        "scale": None,
                        "linear": None,
                        "loss": None,
                        "opt": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = load_workspace_state(tmp_path)
    assert loaded[STAGE_SIMPLE]["choices"]["data"] == "restaurant"
    assert loaded[STAGE_MULTIPLE]["choices"]["data"] == "housing"
    assert loaded[STAGE_SIMPLE]["choices"]["scale"] is None
    assert loaded[STAGE_SIMPLE]["choices"]["split"] is None


def test_slot_order_puts_split_after_data() -> None:
    assert SLOT_IDS == ("data", "split", "scale", "linear", "loss", "opt")


def test_apply_slot_write_accepts_train_pct_and_rejects_empty_sets() -> None:
    empty = empty_slot_state(STAGE_SIMPLE)
    written = apply_slot_write(empty, {"choices": {"split": 73}}, stage=STAGE_SIMPLE)
    assert written["choices"]["split"] == 73
    assert slot_label("split", written, STAGE_SIMPLE) == "訓練 73%"
    assert slot_is_filled("split", written) is True

    for rejected in (0, 100, -1, 101, 80.5, "nope"):
        bounced = apply_slot_write(written, {"choices": {"split": rejected}}, stage=STAGE_SIMPLE)
        assert bounced["choices"]["split"] == 73

    eighty = apply_slot_write(written, {"choices": {"split": "80"}}, stage=STAGE_SIMPLE)
    assert eighty["choices"]["split"] == 80
    assert slot_signature(eighty) != slot_signature(written)


def test_split_inspect_shows_train_percent_and_row_counts() -> None:
    empty_rows = dict(slot_inspect_rows("split", empty_slot_state(STAGE_SIMPLE), stage=STAGE_SIMPLE))
    assert "尚未選擇" in empty_rows["目前選擇"]

    rows = dict(
        slot_inspect_rows("split", _complete(train_pct=80), stage=STAGE_SIMPLE, row_count=97)
    )
    assert rows["目前選擇"] == "訓練 80%"
    assert rows["測試"] == "20%"
    assert rows["訓練列數"] == "78"
    assert rows["測試列數"] == "19"


def test_split_frame_uses_rounded_train_count_and_fixed_seed() -> None:
    frame = pd.DataFrame({"x": range(97), "y": range(97)})
    first_train, first_test = split_frame_by_train_pct(frame, 80)
    again_train, again_test = split_frame_by_train_pct(frame, 80)
    assert len(first_train) == 78
    assert len(first_test) == 19
    assert first_train["x"].tolist() == again_train["x"].tolist()
    one_pct_train, one_pct_test = split_frame_by_train_pct(frame, 1)
    assert len(one_pct_train) == 1
    assert len(one_pct_test) == 96


def test_stages_keep_independent_split() -> None:
    workspace = empty_workspace_state()
    workspace[STAGE_SIMPLE] = apply_slot_write(
        default_slot_state(STAGE_SIMPLE),
        {"choices": {"split": 50}},
        stage=STAGE_SIMPLE,
    )
    assert workspace[STAGE_SIMPLE]["choices"]["split"] == 50
    assert workspace[STAGE_MULTIPLE]["choices"]["split"] is None
