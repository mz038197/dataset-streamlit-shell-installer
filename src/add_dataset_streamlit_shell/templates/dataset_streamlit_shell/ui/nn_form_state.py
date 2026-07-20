"""類神經網路左欄表單 JSON 與 Agent 實驗迴圈狀態（對齊 Studio writeback）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_streamlit_shell.ml.coffee_nn import (
    ACTIVATION_CHOICES,
    CompileSpec,
    FEATURE_OPTIONS,
    HiddenLayerSpec,
    LOSS_AUTO,
    LOSS_CHOICES,
    MAX_HIDDEN_LAYERS,
    MAX_OUTPUT_UNITS,
    MAX_UNITS_PER_LAYER,
    NetworkSpec,
    OPTIMIZER_CHOICES,
    OUTPUT_ACTIVATION_CHOICES,
    TrainConfig,
    TrainResult,
    lab02_default_compile_spec,
    lab02_default_network_spec,
)

DEFAULT_MAX_RUNS = 3
MAX_RUNS_HARD_CAP = 5
AGENT_EPOCHS_MAX = 80

LOOP_STATUS_IDLE = "idle"
LOOP_STATUS_NEED_TRAIN = "need_train"
LOOP_STATUS_NEED_AGENT = "need_agent_decision"

FORM_MTIME_KEY = "nn_form_applied_mtime"
LOOP_KEY = "nn_agent_loop"


def nn_form_path(workspace_dir: Path) -> Path:
    return workspace_dir / "nn_form.json"


def nn_train_request_path(workspace_dir: Path) -> Path:
    return workspace_dir / "nn_train_request.json"


def nn_last_run_path(workspace_dir: Path) -> Path:
    return workspace_dir / "nn_last_run.json"


def nn_agent_runs_path(workspace_dir: Path) -> Path:
    return workspace_dir / "nn_agent_runs.jsonl"


def default_nn_form_state() -> dict[str, Any]:
    spec = lab02_default_network_spec()
    compile_spec = lab02_default_compile_spec()
    return {
        "input_features": list(spec.input_features),
        "hidden_layers": [
            {"units": layer.units, "activation": layer.activation}
            for layer in spec.hidden_layers
        ],
        "output_units": spec.output_units,
        "output_activation": spec.output_activation,
        "use_normalization_layer": spec.use_normalization_layer,
        "loss_choice": spec.loss_choice,
        "optimizer": compile_spec.optimizer_name,
        "learning_rate": compile_spec.learning_rate,
        "epochs": 100,
    }


def normalize_nn_form_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    base = default_nn_form_state()
    if not isinstance(raw, dict):
        return base

    features = raw.get("input_features", base["input_features"])
    if not isinstance(features, list):
        features = list(base["input_features"])
    features = [f for f in features if f in FEATURE_OPTIONS]
    if not features:
        features = list(FEATURE_OPTIONS)

    hidden_raw = raw.get("hidden_layers", base["hidden_layers"])
    hidden_layers: list[dict[str, Any]] = []
    if isinstance(hidden_raw, list):
        for item in hidden_raw[:MAX_HIDDEN_LAYERS]:
            if not isinstance(item, dict):
                continue
            try:
                units = int(item.get("units", 3))
            except (TypeError, ValueError):
                units = 3
            units = max(1, min(MAX_UNITS_PER_LAYER, units))
            activation = str(item.get("activation", "relu"))
            if activation not in ACTIVATION_CHOICES:
                activation = "relu"
            hidden_layers.append({"units": units, "activation": activation})

    try:
        output_units = int(raw.get("output_units", base["output_units"]))
    except (TypeError, ValueError):
        output_units = 1
    output_units = max(1, min(MAX_OUTPUT_UNITS, output_units))

    output_activation = str(raw.get("output_activation", base["output_activation"]))
    allowed_out = ACTIVATION_CHOICES if output_units == 1 else OUTPUT_ACTIVATION_CHOICES
    if output_activation not in allowed_out:
        output_activation = "sigmoid" if output_units == 1 else "softmax"

    loss_choice = str(raw.get("loss_choice", base["loss_choice"]))
    if loss_choice not in LOSS_CHOICES:
        loss_choice = LOSS_AUTO

    optimizer = str(raw.get("optimizer", base["optimizer"]))
    if optimizer not in OPTIMIZER_CHOICES:
        optimizer = "Adam"

    try:
        learning_rate = float(raw.get("learning_rate", base["learning_rate"]))
    except (TypeError, ValueError):
        learning_rate = 0.01
    learning_rate = max(0.0001, min(1.0, learning_rate))

    try:
        epochs = int(raw.get("epochs", base["epochs"]))
    except (TypeError, ValueError):
        epochs = 100
    epochs = max(1, min(500, epochs))

    return {
        "input_features": features,
        "hidden_layers": hidden_layers,
        "output_units": output_units,
        "output_activation": output_activation,
        "use_normalization_layer": bool(raw.get("use_normalization_layer", False)),
        "loss_choice": loss_choice,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }


def load_nn_form_state(workspace_dir: Path) -> dict[str, Any]:
    path = nn_form_path(workspace_dir)
    if not path.is_file():
        state = default_nn_form_state()
        save_nn_form_state(workspace_dir, state)
        return state
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_nn_form_state()
    return normalize_nn_form_state(raw if isinstance(raw, dict) else None)


def save_nn_form_state(workspace_dir: Path, state: dict[str, Any]) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_nn_form_state(state)
    path = nn_form_path(workspace_dir)
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def form_file_mtime(workspace_dir: Path) -> float:
    path = nn_form_path(workspace_dir)
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def should_rerun_after_nn_chat(
    *,
    requested: bool,
    form_mtime: float,
    applied_mtime: float,
) -> bool:
    """Chat 後是否整頁重跑：有訓練請求，或 form 檔比已套用版本更新。"""
    return bool(requested) or form_mtime > applied_mtime


def state_to_specs(state: dict[str, Any]) -> tuple[NetworkSpec, CompileSpec, TrainConfig]:
    normalized = normalize_nn_form_state(state)
    hidden = tuple(
        HiddenLayerSpec(int(layer["units"]), str(layer["activation"]))
        for layer in normalized["hidden_layers"]
    )
    spec = NetworkSpec(
        input_features=tuple(normalized["input_features"]),
        hidden_layers=hidden,
        output_units=int(normalized["output_units"]),
        output_activation=str(normalized["output_activation"]),
        loss_choice=str(normalized["loss_choice"]),
        use_normalization_layer=bool(normalized["use_normalization_layer"]),
    )
    compile_spec = CompileSpec(
        optimizer_name=str(normalized["optimizer"]),
        learning_rate=float(normalized["learning_rate"]),
    )
    train_config = TrainConfig(epochs=int(normalized["epochs"]))
    return spec, compile_spec, train_config


def specs_to_state(
    spec: NetworkSpec,
    compile_spec: CompileSpec,
    train_config: TrainConfig,
) -> dict[str, Any]:
    return normalize_nn_form_state(
        {
            "input_features": list(spec.input_features),
            "hidden_layers": [
                {"units": layer.units, "activation": layer.activation}
                for layer in spec.hidden_layers
            ],
            "output_units": spec.output_units,
            "output_activation": spec.output_activation,
            "use_normalization_layer": spec.use_normalization_layer,
            "loss_choice": spec.loss_choice,
            "optimizer": compile_spec.optimizer_name,
            "learning_rate": compile_spec.learning_rate,
            "epochs": train_config.epochs,
        }
    )


def apply_state_to_session(session_state: Any, state: dict[str, Any]) -> None:
    normalized = normalize_nn_form_state(state)
    session_state["nn_features"] = list(normalized["input_features"])
    session_state["nn_hidden_count"] = len(normalized["hidden_layers"])
    for index, layer in enumerate(normalized["hidden_layers"], start=1):
        session_state[f"nn_hidden_units_{index}"] = int(layer["units"])
        session_state[f"nn_hidden_activation_{index}"] = str(layer["activation"])
    session_state["nn_output_units"] = int(normalized["output_units"])
    session_state["nn_output_activation"] = str(normalized["output_activation"])
    session_state["nn_use_norm_layer"] = bool(normalized["use_normalization_layer"])
    session_state["nn_loss_choice"] = str(normalized["loss_choice"])
    session_state["nn_optimizer"] = str(normalized["optimizer"])
    session_state["nn_learning_rate"] = float(normalized["learning_rate"])
    session_state["nn_epochs"] = int(normalized["epochs"])


def session_to_state(session_state: Any) -> dict[str, Any]:
    hidden_count = int(session_state.get("nn_hidden_count", 0) or 0)
    hidden_count = max(0, min(MAX_HIDDEN_LAYERS, hidden_count))
    hidden_layers = []
    for index in range(1, hidden_count + 1):
        units = session_state.get(f"nn_hidden_units_{index}", 3)
        activation = session_state.get(f"nn_hidden_activation_{index}", "relu")
        hidden_layers.append({"units": units, "activation": activation})
    return normalize_nn_form_state(
        {
            "input_features": session_state.get("nn_features", list(FEATURE_OPTIONS)),
            "hidden_layers": hidden_layers,
            "output_units": session_state.get("nn_output_units", 1),
            "output_activation": session_state.get("nn_output_activation", "sigmoid"),
            "use_normalization_layer": session_state.get("nn_use_norm_layer", False),
            "loss_choice": session_state.get("nn_loss_choice", LOSS_AUTO),
            "optimizer": session_state.get("nn_optimizer", "Adam"),
            "learning_rate": session_state.get("nn_learning_rate", 0.01),
            "epochs": session_state.get("nn_epochs", 100),
        }
    )


def clamp_max_runs(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_RUNS
    return max(1, min(MAX_RUNS_HARD_CAP, parsed))


def clamp_agent_epochs(epochs: int) -> int:
    return max(1, min(AGENT_EPOCHS_MAX, int(epochs)))


def default_loop_state() -> dict[str, Any]:
    return {
        "status": LOOP_STATUS_IDLE,
        "run_index": 0,
        "max_runs": DEFAULT_MAX_RUNS,
    }


def get_loop_state(session_state: Any) -> dict[str, Any]:
    raw = session_state.get(LOOP_KEY)
    if not isinstance(raw, dict):
        state = default_loop_state()
        session_state[LOOP_KEY] = state
        return state
    state = {
        "status": str(raw.get("status", LOOP_STATUS_IDLE)),
        "run_index": max(0, int(raw.get("run_index", 0) or 0)),
        "max_runs": clamp_max_runs(raw.get("max_runs", DEFAULT_MAX_RUNS)),
    }
    if state["status"] not in {
        LOOP_STATUS_IDLE,
        LOOP_STATUS_NEED_TRAIN,
        LOOP_STATUS_NEED_AGENT,
    }:
        state["status"] = LOOP_STATUS_IDLE
    session_state[LOOP_KEY] = state
    return state


def set_loop_state(session_state: Any, **updates: Any) -> dict[str, Any]:
    state = get_loop_state(session_state)
    state.update(updates)
    if "max_runs" in updates:
        state["max_runs"] = clamp_max_runs(state["max_runs"])
    if "run_index" in updates:
        state["run_index"] = max(0, int(state["run_index"]))
    session_state[LOOP_KEY] = state
    return state


def reset_loop_budget(session_state: Any) -> dict[str, Any]:
    state = get_loop_state(session_state)
    return set_loop_state(
        session_state,
        status=LOOP_STATUS_IDLE,
        run_index=0,
        max_runs=state["max_runs"],
    )


def remaining_runs(loop: dict[str, Any]) -> int:
    return max(0, int(loop["max_runs"]) - int(loop["run_index"]))


def train_request_is_set(workspace_dir: Path) -> bool:
    path = nn_train_request_path(workspace_dir)
    if not path.is_file():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and raw.get("requested") is True


def clear_train_request(workspace_dir: Path) -> None:
    path = nn_train_request_path(workspace_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def consume_train_request(workspace_dir: Path, session_state: Any) -> bool:
    """若有訓練請求且仍有預算，設為 need_train 並清除請求檔。回傳是否進入訓練。"""
    if not train_request_is_set(workspace_dir):
        return False
    clear_train_request(workspace_dir)
    loop = get_loop_state(session_state)
    if loop["run_index"] >= loop["max_runs"]:
        return False
    set_loop_state(session_state, status=LOOP_STATUS_NEED_TRAIN)
    return True


def write_last_run(
    workspace_dir: Path,
    *,
    form_state: dict[str, Any],
    train_result: TrainResult,
    run_index: int,
    epochs_used: int,
    note: str = "",
) -> dict[str, Any]:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_index": run_index,
        "epochs_used": epochs_used,
        "final_loss": float(train_result.final_loss),
        "train_accuracy": float(train_result.train_accuracy),
        "parameter_count": int(train_result.parameter_count),
        "form": normalize_nn_form_state(form_state),
        "note": note,
    }
    nn_last_run_path(workspace_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with nn_agent_runs_path(workspace_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def load_last_run(workspace_dir: Path) -> dict[str, Any] | None:
    path = nn_last_run_path(workspace_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def format_last_run_summary(last_run: dict[str, Any] | None) -> str:
    if not last_run:
        return "尚無 Agent 實驗訓練結果。"
    form = last_run.get("form") if isinstance(last_run.get("form"), dict) else {}
    hidden = form.get("hidden_layers") if isinstance(form.get("hidden_layers"), list) else []
    hidden_txt = ", ".join(
        f"{layer.get('units')}({layer.get('activation')})"
        for layer in hidden
        if isinstance(layer, dict)
    ) or "無"
    return (
        f"第 {last_run.get('run_index', '?')} 輪："
        f"loss={float(last_run.get('final_loss', float('nan'))):.4f}，"
        f"accuracy≈{float(last_run.get('train_accuracy', 0.0)):.2f}%，"
        f"epochs={last_run.get('epochs_used', '?')}，"
        f"隱藏層[{hidden_txt}]，"
        f"lr={form.get('learning_rate', '?')}"
    )


def nn_host_context_fragment(
    *,
    form_path: str,
    request_path: str,
    last_run_path: str,
) -> str:
    """寫進 dataset_base_context 的類神經網路 writeback／實驗規則。"""
    return (
        "【類神經網路頁】左欄架構／超參存在共享 JSON："
        f"{form_path}。若使用者要求改架構或超參，請 read_file 後以 edit_file／write_file 更新該檔，"
        "保留既有鍵名與型別（input_features、hidden_layers、output_units、output_activation、"
        "use_normalization_layer、loss_choice、optimizer、learning_rate、epochs）；不確定先 read_file。"
        "若要讓左欄播放與手動相同的訓練動畫，請在更新架構後寫入 "
        f'{request_path}，內容為 {{"requested": true}}。不要自行 exec 訓練、不要假裝已訓練完成。'
        f"訓練結束後系統會寫入 {last_run_path}；決策時請先 read_file 該結果，用繁體中文解釋取捨，"
        "若要繼續實驗再改 nn_form.json 並再次寫入訓練請求。達實驗上限必須停止並總結。"
        "禁止一開口無解釋貼完整 Lab02 標準答案；max_runs 與預算僅能由左欄 UI 調整，請勿改高或清除。"
    )


def build_nn_page_agent_context(
    *,
    form_state: dict[str, Any],
    row_count: int,
    loop: dict[str, Any],
    last_run: dict[str, Any] | None,
    form_path: str,
    request_path: str,
    last_run_path: str,
) -> str:
    spec, compile_spec, train_config = state_to_specs(form_state)
    train_result = None
    if last_run is not None:
        try:
            train_result = TrainResult(
                history={},
                final_loss=float(last_run.get("final_loss", float("nan"))),
                train_accuracy=float(last_run.get("train_accuracy", 0.0)),
                parameter_count=int(last_run.get("parameter_count", 0)),
            )
        except (TypeError, ValueError):
            train_result = None

    from dataset_streamlit_shell.ml.coffee_nn import build_nn_agent_context

    base = build_nn_agent_context(
        spec=spec,
        compile_spec=compile_spec,
        train_result=train_result,
        row_count=row_count,
    )
    lines = [
        base,
        f"epochs（表單）={train_config.epochs}；Agent 觸發訓練時會 clamp ≤ {AGENT_EPOCHS_MAX}。",
        f"共享架構檔：{form_path}",
        f"訓練請求檔：{request_path}（寫入 {{\"requested\": true}} 才會開始左欄動畫訓練）",
        f"最近結果檔：{last_run_path}",
        f"Agent 實驗進度：已完成 {loop['run_index']}／上限 {loop['max_runs']} 輪，剩餘 {remaining_runs(loop)} 輪。",
        f"迴圈狀態：{loop['status']}",
        f"最近一次實驗：{format_last_run_summary(last_run)}",
        "max_runs 僅能由左欄 UI 調整，請勿改高上限或清除預算。",
    ]
    return "\n".join(lines)
