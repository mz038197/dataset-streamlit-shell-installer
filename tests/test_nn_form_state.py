from __future__ import annotations

import json
import sys
from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.coffee_nn import (  # noqa: E402
    TrainConfig,
    TrainResult,
    lab02_default_compile_spec,
    lab02_default_network_spec,
    load_builtin_frame,
    validate_network_spec,
)
from dataset_streamlit_shell.ml.coffee_nn import BUILTIN_DATA_PATH_SUFFIX  # noqa: E402
from dataset_streamlit_shell.ui.nn_form_state import (  # noqa: E402
    AGENT_EPOCHS_MAX,
    DEFAULT_MAX_RUNS,
    LOOP_STATUS_NEED_TRAIN,
    apply_state_to_session,
    build_nn_page_agent_context,
    clamp_agent_epochs,
    clamp_max_runs,
    consume_train_request,
    default_nn_form_state,
    get_loop_state,
    load_nn_form_state,
    nn_form_path,
    nn_train_request_path,
    normalize_nn_form_state,
    remaining_runs,
    reset_loop_budget,
    save_nn_form_state,
    set_loop_state,
    specs_to_state,
    state_to_specs,
    train_request_is_set,
    write_last_run,
)

BUILTIN_PATH = TEMPLATE_ROOT / "dataset_streamlit_shell" / Path(*BUILTIN_DATA_PATH_SUFFIX)


class _FakeSession(dict):
    """最小 session_state 替身。"""


def test_default_round_trip(tmp_path: Path) -> None:
    state = default_nn_form_state()
    save_nn_form_state(tmp_path, state)
    loaded = load_nn_form_state(tmp_path)
    assert loaded["hidden_layers"][0]["units"] == 3
    assert loaded["optimizer"] == "Adam"
    spec, compile_spec, train_config = state_to_specs(loaded)
    frame = load_builtin_frame(BUILTIN_PATH)
    assert validate_network_spec(spec, frame) == []
    assert compile_spec.learning_rate == 0.01
    assert train_config.epochs == 100


def test_normalize_clamps_invalid_values() -> None:
    raw = {
        "input_features": ["溫度", "時間", "幽靈"],
        "hidden_layers": [
            {"units": 999, "activation": "nope"},
            {"units": 2, "activation": "relu"},
            {"units": 2, "activation": "relu"},
            {"units": 2, "activation": "relu"},
            {"units": 2, "activation": "relu"},
            {"units": 2, "activation": "relu"},
            {"units": 2, "activation": "relu"},
            {"units": 2, "activation": "relu"},
            {"units": 2, "activation": "relu"},
            {"units": 2, "activation": "relu"},
        ],
        "output_units": 99,
        "output_activation": "softmax",
        "learning_rate": 9.0,
        "epochs": 9999,
        "optimizer": "Nope",
        "loss_choice": "???",
    }
    normalized = normalize_nn_form_state(raw)
    assert normalized["input_features"] == ["溫度", "時間"]
    assert len(normalized["hidden_layers"]) == 8
    assert normalized["hidden_layers"][0]["units"] == 32
    assert normalized["hidden_layers"][0]["activation"] == "relu"
    assert normalized["output_units"] == 10
    assert normalized["learning_rate"] == 1.0
    assert normalized["epochs"] == 500
    assert normalized["optimizer"] == "Adam"


def test_lab02_specs_to_state_validates() -> None:
    state = specs_to_state(
        lab02_default_network_spec(),
        lab02_default_compile_spec(),
        TrainConfig(epochs=100),
    )
    spec, _, _ = state_to_specs(state)
    frame = load_builtin_frame(BUILTIN_PATH)
    assert validate_network_spec(spec, frame) == []


def test_apply_state_to_session_keys() -> None:
    session = _FakeSession()
    state = default_nn_form_state()
    apply_state_to_session(session, state)
    assert session["nn_hidden_count"] == 1
    assert session["nn_hidden_units_1"] == 3
    assert session["nn_optimizer"] == "Adam"


def test_consume_train_request_respects_budget(tmp_path: Path) -> None:
    session = _FakeSession()
    set_loop_state(session, run_index=3, max_runs=3, status="idle")
    nn_train_request_path(tmp_path).write_text(
        json.dumps({"requested": True}),
        encoding="utf-8",
    )
    assert consume_train_request(tmp_path, session) is False
    assert get_loop_state(session)["status"] == "idle"
    assert not nn_train_request_path(tmp_path).exists()

    set_loop_state(session, run_index=0, max_runs=3)
    nn_train_request_path(tmp_path).write_text(
        json.dumps({"requested": True}),
        encoding="utf-8",
    )
    assert consume_train_request(tmp_path, session) is True
    assert get_loop_state(session)["status"] == LOOP_STATUS_NEED_TRAIN


def test_train_request_requires_explicit_true(tmp_path: Path) -> None:
    request_path = nn_train_request_path(tmp_path)
    for content in (
        "{not-json",
        "{}",
        "[]",
        json.dumps({"requested": False}),
        json.dumps({"requested": 1}),
    ):
        request_path.write_text(content, encoding="utf-8")
        assert train_request_is_set(tmp_path) is False

    request_path.write_text(json.dumps({"requested": True}), encoding="utf-8")
    assert train_request_is_set(tmp_path) is True


def test_reset_loop_budget_keeps_max_runs() -> None:
    session = _FakeSession()
    set_loop_state(session, run_index=2, max_runs=5, status=LOOP_STATUS_NEED_TRAIN)
    reset_loop_budget(session)
    loop = get_loop_state(session)
    assert loop["run_index"] == 0
    assert loop["max_runs"] == 5
    assert loop["status"] == "idle"


def test_clamp_helpers() -> None:
    assert clamp_max_runs(0) == 1
    assert clamp_max_runs(99) == 5
    assert clamp_max_runs(None) == DEFAULT_MAX_RUNS
    assert clamp_agent_epochs(200) == AGENT_EPOCHS_MAX
    assert remaining_runs({"run_index": 1, "max_runs": 3}) == 2


def test_write_last_run_and_context(tmp_path: Path) -> None:
    form = default_nn_form_state()
    result = TrainResult(
        history={"loss": [0.5]},
        final_loss=0.5,
        train_accuracy=80.0,
        parameter_count=12,
    )
    payload = write_last_run(
        tmp_path,
        form_state=form,
        train_result=result,
        run_index=1,
        epochs_used=80,
        note="test",
    )
    assert payload["train_accuracy"] == 80.0
    assert (tmp_path / "nn_last_run.json").is_file()
    assert (tmp_path / "nn_agent_runs.jsonl").is_file()

    session = _FakeSession()
    loop = get_loop_state(session)
    ctx = build_nn_page_agent_context(
        form_state=form,
        row_count=400,
        loop=loop,
        last_run=payload,
        form_path="workspace/nn_form.json",
        request_path="workspace/nn_train_request.json",
        last_run_path="workspace/nn_last_run.json",
    )
    assert "類神經網路" in ctx
    assert "剩餘" in ctx
    assert "0.5000" in ctx or "0.5" in ctx
    assert nn_form_path(tmp_path).name == "nn_form.json"


def test_nn_host_context_fragment_mentions_paths() -> None:
    from dataset_streamlit_shell.ui.nn_form_state import nn_host_context_fragment

    text = nn_host_context_fragment(
        form_path="workspace/nn_form.json",
        request_path="workspace/nn_train_request.json",
        last_run_path="workspace/nn_last_run.json",
    )
    assert "nn_form.json" in text
    assert "nn_train_request.json" in text
    assert "requested" in text
    assert "max_runs" in text
