"""線性回歸決策槽狀態、模型程式碼預覽與訓練請求（對齊類神經網路寫檔協議）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STAGE_SIMPLE = "simple"
STAGE_MULTIPLE = "multiple"
STAGES = (STAGE_SIMPLE, STAGE_MULTIPLE)

SLOT_IDS = ("data", "scale", "linear", "loss", "opt")
SLOT_TITLES = {
    "data": "輸入資料",
    "scale": "特徵縮放",
    "linear": "線性層",
    "loss": "損失函數",
    "opt": "優化器",
}

SCALE_MAXDIV = "maxdiv"
SCALE_MINMAX = "minmax"
SCALE_MEAN = "mean"
SCALE_ZSCORE = "zscore"
SCALE_METHODS = (SCALE_MAXDIV, SCALE_MINMAX, SCALE_MEAN, SCALE_ZSCORE)

DATA_RESTAURANT = "restaurant"
DATA_HOUSING = "housing"
LINEAR_DENSE1 = "dense1"
LOSS_MSE = "mse"
OPT_SGD = "sgd"

CHOICE_UNSET = "尚未選擇"
DATA_PREVIEW_ROWS = 10

SCALE_LABELS = {
    SCALE_MAXDIV: "正規化",
    SCALE_MINMAX: "最小-最大正規化",
    SCALE_MEAN: "平均值正規化",
    SCALE_ZSCORE: "Z分數正規化",
}

DATA_LABELS = {
    DATA_RESTAURANT: "內建餐廳獲利",
    DATA_HOUSING: "內建房價四特徵",
}

LOCKED_DATA = {
    STAGE_SIMPLE: DATA_RESTAURANT,
    STAGE_MULTIPLE: DATA_HOUSING,
}

DEFAULT_ALPHA = {
    STAGE_SIMPLE: 0.01,
    STAGE_MULTIPLE: 0.1,
}
DEFAULT_EPOCHS = {
    STAGE_SIMPLE: 1500,
    STAGE_MULTIPLE: 1000,
}

ALPHA_MIN = 0.0001
ALPHA_MAX = 1.0
EPOCHS_MIN = 1
EPOCHS_MAX = 5000

SCALE_INSPECT = {
    SCALE_MAXDIV: {
        "formula": "x′ = x / x_max",
        "range": "0 ≤ x′ ≤ 1",
        "condition": "x 必須 ≥ 0",
        "code": "x = x / x.max()",
    },
    SCALE_MINMAX: {
        "formula": "x′ = (x − x_min) / (x_max − x_min)",
        "range": "0 ≤ x′ ≤ 1",
        "condition": "",
        "code": "x = MinMaxScaler().fit_transform(x)",
    },
    SCALE_MEAN: {
        "formula": "x′ = (x − μ) / (x_max − x_min)",
        "range": "−1 ≤ x′ ≤ 1",
        "condition": "μ 為平均",
        "code": "x = (x - x.mean()) / (x.max() - x.min())",
    },
    SCALE_ZSCORE: {
        "formula": "x′ = (x − μ) / σ",
        "range": "實務上 −3 ≤ x′ ≤ 3",
        "condition": "μ 為平均，σ 為標準差",
        "code": "x = StandardScaler().fit_transform(x)",
    },
}


def empty_slot_state(stage: str = STAGE_SIMPLE) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return {
        "choices": {
            slot: LOCKED_DATA[stage] if slot == "data" else None for slot in SLOT_IDS
        },
        "alpha": None,
        "epochs": None,
    }


def empty_workspace_state() -> dict[str, Any]:
    return {stage: empty_slot_state(stage) for stage in STAGES}


def default_slot_state(stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return {
        "choices": {
            "data": LOCKED_DATA[stage],
            "scale": SCALE_ZSCORE,
            "linear": LINEAR_DENSE1,
            "loss": LOSS_MSE,
            "opt": OPT_SGD,
        },
        "alpha": DEFAULT_ALPHA[stage],
        "epochs": DEFAULT_EPOCHS[stage],
    }


def _clamp_alpha(value: Any, fallback: float | None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(ALPHA_MIN, min(ALPHA_MAX, parsed))


def _clamp_epochs(value: Any, fallback: int | None) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(EPOCHS_MIN, min(EPOCHS_MAX, parsed))


def normalize_slot_state(raw: dict[str, Any] | None, *, stage: str) -> dict[str, Any]:
    base = empty_slot_state(stage)
    if not isinstance(raw, dict):
        return base
    incoming = raw.get("choices") if isinstance(raw.get("choices"), dict) else {}

    scale = incoming.get("scale")
    if scale in SCALE_METHODS:
        base["choices"]["scale"] = scale

    if incoming.get("linear") == LINEAR_DENSE1:
        base["choices"]["linear"] = LINEAR_DENSE1
    if incoming.get("loss") == LOSS_MSE:
        base["choices"]["loss"] = LOSS_MSE
    if incoming.get("opt") == OPT_SGD:
        base["choices"]["opt"] = OPT_SGD

    if raw.get("alpha") is not None:
        base["alpha"] = _clamp_alpha(raw.get("alpha"), None)
    if raw.get("epochs") is not None:
        base["epochs"] = _clamp_epochs(raw.get("epochs"), None)
    return base


def apply_slot_write(
    previous: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
    *,
    stage: str,
) -> dict[str, Any]:
    """套用 Agent 寫入：可改特徵縮放與 α／epochs；鎖定槽種類拒改。"""
    current = normalize_slot_state(previous, stage=stage)
    raw = incoming if isinstance(incoming, dict) else {}
    choices = raw.get("choices") if isinstance(raw.get("choices"), dict) else {}

    next_state = {
        "choices": dict(current["choices"]),
        "alpha": current["alpha"],
        "epochs": current["epochs"],
    }

    locked_data = LOCKED_DATA[stage]
    if "data" in choices:
        if choices["data"] == locked_data:
            next_state["choices"]["data"] = locked_data
        elif choices["data"] is not None:
            next_state["choices"]["data"] = current["choices"]["data"]

    if "scale" in choices:
        if choices["scale"] in SCALE_METHODS:
            next_state["choices"]["scale"] = choices["scale"]
        elif choices["scale"] is not None:
            next_state["choices"]["scale"] = current["choices"]["scale"]

    for slot, allowed in (
        ("linear", LINEAR_DENSE1),
        ("loss", LOSS_MSE),
        ("opt", OPT_SGD),
    ):
        if slot not in choices:
            continue
        if choices[slot] == allowed:
            next_state["choices"][slot] = allowed
        elif choices[slot] is not None:
            next_state["choices"][slot] = current["choices"][slot]

    if "alpha" in raw:
        next_state["alpha"] = _clamp_alpha(raw.get("alpha"), current["alpha"])
    if "epochs" in raw:
        next_state["epochs"] = _clamp_epochs(raw.get("epochs"), current["epochs"])
    return next_state


def slots_are_complete(state: dict[str, Any]) -> bool:
    choices = state.get("choices") if isinstance(state.get("choices"), dict) else {}
    if any(choices.get(slot) is None for slot in SLOT_IDS):
        return False
    if state.get("alpha") is None or state.get("epochs") is None:
        return False
    return True


def slot_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    choices = state.get("choices") if isinstance(state.get("choices"), dict) else {}
    return (
        choices.get("data"),
        choices.get("scale"),
        choices.get("linear"),
        choices.get("loss"),
        choices.get("opt"),
        state.get("alpha"),
        state.get("epochs"),
    )


def slot_label(slot_id: str, state: dict[str, Any], stage: str) -> str:
    choices = state.get("choices") if isinstance(state.get("choices"), dict) else {}
    choice = choices.get(slot_id)
    if choice is None:
        return CHOICE_UNSET
    if slot_id == "data":
        return DATA_LABELS.get(str(choice), str(choice))
    if slot_id == "scale":
        return SCALE_LABELS.get(str(choice), str(choice))
    if slot_id == "linear":
        return "Dense(1, linear)"
    if slot_id == "loss":
        return "MSE"
    if slot_id == "opt":
        alpha = state.get("alpha")
        epochs = state.get("epochs")
        if alpha is None or epochs is None:
            return "SGD"
        return f"SGD · α={float(alpha):g} · {int(epochs)} epochs"
    return str(choice)


def slot_is_filled(slot_id: str, state: dict[str, Any]) -> bool:
    choices = state.get("choices") if isinstance(state.get("choices"), dict) else {}
    if slot_id == "opt":
        return (
            choices.get("opt") is not None
            and state.get("alpha") is not None
            and state.get("epochs") is not None
        )
    return choices.get(slot_id) is not None


def slot_button_label(slot_id: str, state: dict[str, Any], stage: str) -> str:
    title = SLOT_TITLES[slot_id]
    if not slot_is_filled(slot_id, state):
        return title
    return f"{title}\n{slot_label(slot_id, state, stage)}"


def scale_inspect(method: str | None) -> dict[str, str]:
    if method not in SCALE_INSPECT:
        return {"formula": "", "range": "", "condition": "", "code": ""}
    return dict(SCALE_INSPECT[method])


def slot_inspect_rows(
    slot_id: str,
    state: dict[str, Any],
    *,
    stage: str,
    row_count: int | None = None,
) -> list[tuple[str, str]]:
    choices = state.get("choices") if isinstance(state.get("choices"), dict) else {}
    choice = choices.get(slot_id)
    if slot_id == "data":
        if stage == STAGE_SIMPLE:
            rows = [
                ("目前選擇", DATA_LABELS[DATA_RESTAURANT]),
                ("x", "城市人口_萬人"),
                ("y", "餐廳獲利_萬美元"),
            ]
        else:
            rows = [
                ("目前選擇", DATA_LABELS[DATA_HOUSING]),
                ("x", "面積_平方英尺、房間數、樓層數、屋齡_年"),
                ("y", "房價_千美元"),
            ]
        if row_count is not None:
            rows.append(("列數", str(row_count)))
        rows.append(("可否改", "本頁鎖內建資料"))
        return rows
    if choice is None:
        return [("目前選擇", "尚未選擇。跟右側 Agent 說你要哪一個。")]
    if slot_id == "scale":
        info = scale_inspect(str(choice))
        rows = [
            ("目前選擇", SCALE_LABELS.get(str(choice), str(choice))),
            ("公式", info["formula"]),
            ("範圍", info["range"]),
        ]
        if info["condition"]:
            rows.append(("條件", info["condition"]))
        rows.append(
            ("可討論", "正規化、最小-最大正規化、平均值正規化、Z分數正規化")
        )
        return rows
    if slot_id == "linear":
        n_features = 1 if stage == STAGE_SIMPLE else 4
        return [
            ("目前選擇", "Dense(1, linear)"),
            ("特徵數", str(n_features)),
            ("可否改", "第一版鎖線性層種類"),
        ]
    if slot_id == "loss":
        return [
            ("目前選擇", "MSE"),
            ("對應", "Cost J = (1/2m) Σ (ŷ − y)²"),
            ("可否改", "第一版鎖 MSE"),
        ]
    return [
        ("目前選擇", "SGD"),
        ("α", "尚未寫入" if state.get("alpha") is None else f"{float(state['alpha']):g}"),
        ("epochs", "尚未寫入" if state.get("epochs") is None else str(int(state["epochs"]))),
        ("對應", "w := w − α ∂J/∂w"),
        ("可否改", "種類鎖 SGD；可請 Agent 改 α 與 epochs"),
    ]


def model_code_preview(state: dict[str, Any], *, stage: str) -> str:
    if not slots_are_complete(state):
        return "# 決策槽還沒選齊，沒有模型程式碼預覽。"
    n_features = 1 if stage == STAGE_SIMPLE else 4
    scale = str(state["choices"]["scale"])
    info = scale_inspect(scale)
    alpha = float(state["alpha"])
    lines = [
        info["code"],
        "model = Sequential([",
        f"    Input(shape=({n_features},)),",
        '    Dense(1, activation="linear"),',
        "])",
        "model.compile(",
        '    loss="mse",',
        f"    optimizer=SGD(learning_rate={alpha:g}),",
        ")",
    ]
    return "\n".join(lines)


def scale_method_errors(method: str | None, frame: Any, features: list[str]) -> list[str]:
    if method != SCALE_MAXDIV:
        return []
    numeric = frame[features]
    negatives = [
        str(column)
        for column in features
        if (numeric[column].astype(float) < 0).any()
    ]
    if not negatives:
        return []
    joined = "、".join(negatives)
    return [
        f"正規化（除以最大）要求訓練特徵 x≥0，但 {joined} 出現負值，不能開訓。"
    ]


def can_write_train_request(
    state: dict[str, Any],
    *,
    quiz_unlocked: bool,
    scale_errors: list[str] | None = None,
) -> bool:
    if not slots_are_complete(state):
        return False
    if not quiz_unlocked:
        return False
    if scale_errors:
        return False
    return True


def lr_slots_path(workspace_dir: Path) -> Path:
    return workspace_dir / "lr_slots.json"


def lr_train_request_path(workspace_dir: Path) -> Path:
    return workspace_dir / "lr_train_request.json"


def save_workspace_state(workspace_dir: Path, state: dict[str, Any]) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        stage: normalize_slot_state(state.get(stage), stage=stage) for stage in STAGES
    }
    lr_slots_path(workspace_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_raw_workspace(workspace_dir: Path) -> dict[str, Any] | None:
    path = lr_slots_path(workspace_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def load_workspace_state(
    workspace_dir: Path,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = _read_raw_workspace(workspace_dir)
    if raw is None:
        return previous if previous is not None else empty_workspace_state()
    return merge_workspace_write(previous or empty_workspace_state(), raw)


def merge_workspace_write(
    previous: dict[str, Any],
    incoming: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = incoming if isinstance(incoming, dict) else {}
    merged = {}
    for stage in STAGES:
        if stage in raw:
            merged[stage] = apply_slot_write(
                previous.get(stage),
                raw.get(stage),
                stage=stage,
            )
        else:
            merged[stage] = normalize_slot_state(previous.get(stage), stage=stage)
    return merged


def slots_file_mtime(workspace_dir: Path) -> float:
    path = lr_slots_path(workspace_dir)
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def should_rerun_after_lr_chat(
    *,
    requested: bool,
    slots_mtime: float,
    applied_mtime: float,
) -> bool:
    return bool(requested) or slots_mtime > applied_mtime


def train_request_is_set(workspace_dir: Path) -> bool:
    path = lr_train_request_path(workspace_dir)
    if not path.is_file():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and raw.get("requested") is True


def write_train_request(workspace_dir: Path) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    lr_train_request_path(workspace_dir).write_text(
        json.dumps({"requested": True}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def clear_train_request(workspace_dir: Path) -> None:
    path = lr_train_request_path(workspace_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def consume_train_request(
    workspace_dir: Path,
    *,
    allowed: bool,
) -> bool:
    """若有訓練請求則清檔。僅 allowed 時回傳 True（開始訓練）。"""
    if not train_request_is_set(workspace_dir):
        return False
    clear_train_request(workspace_dir)
    return bool(allowed)


def lr_host_context_fragment(
    *,
    slots_path: str,
    request_path: str,
) -> str:
    return (
        "【線性回歸頁】主教學欄是決策槽列，不是類神經網路 form。"
        "對學生講五個決策槽：輸入資料、特徵縮放、線性層、損失函數、優化器。"
        "輸入資料進頁即完成；其餘四個由你寫入決策槽狀態，不要叫學生自己選。"
        "主教學欄沒有下拉選單，也不要用打勾符號標完成；完成是綠框、第二行目前選擇；點框只看只讀選擇明細。"
        f"決策槽狀態在共享 JSON：{slots_path}，鍵為 simple／multiple 兩學習階段，"
        "每階段含 choices（data、scale、linear、loss、opt）與 alpha、epochs。"
        "輸入資料進頁即為該階段鎖定值（單變量 restaurant、多變量 housing），載入時也回成該值；"
        "不必為了槽齊而寫 data，寫鎖定值可以，寫其他來源拒絕。"
        "該階段尚未組過時，scale／linear／loss／opt 與 alpha、epochs 為尚未選擇。"
        "若使用者要求組一個線性回歸，請 read_file 後以 edit_file／write_file 寫入該階段預設："
        "scale=zscore、linear=dense1、loss=mse、opt=sgd，"
        "以及該階段預設 alpha／epochs（單變量 0.01／1500，多變量 0.1／1000）。"
        "組模型只寫決策槽狀態，不要同時寫訓練請求，也不要自行開始訓練。"
        "可改的只有 scale（maxdiv／minmax／mean／zscore）與 alpha、epochs。"
        "data／linear／loss／opt 是鎖定槽，種類拒絕改成其他值。"
        "write_file 時必須保留另一學習階段的鍵，不要清掉另一側。"
        "若要讓主教學欄播放與「開始訓練」相同的動畫，另寫 "
        f'{request_path}，內容為 {{"requested": true}}。'
        "決策槽未齊、訓練前預測未過關、或正規化（除以最大）遇上負值時，不准寫訓練請求。"
        "不要自行 exec 訓練、不要假裝已訓完、不要代填訓練前預測選項。"
        "不要改 nn_form.json 或類神經網路的訓練請求。"
    )


def build_lr_page_snapshot(
    *,
    stage: str,
    state: dict[str, Any],
    open_slot: str | None,
    quiz_unlocked: bool,
    scale_errors: list[str],
    artifact_note: str,
    slots_path: str,
    request_path: str,
) -> str:
    labels = {
        slot: slot_label(slot, state, stage) for slot in SLOT_IDS
    }
    open_title = SLOT_TITLES.get(open_slot or "", "")
    open_line = (
        f"學生正打開的決策槽：{open_title}。"
        if open_title
        else "學生目前沒有打開選擇明細。"
    )
    complete = slots_are_complete(state)
    allowed = can_write_train_request(
        state,
        quiz_unlocked=quiz_unlocked,
        scale_errors=scale_errors,
    )
    error_line = " ".join(scale_errors) if scale_errors else "特徵縮放驗證通過。"
    return "\n".join(
        [
            f"目前學習階段：{'單變量' if stage == STAGE_SIMPLE else '多變量'}。",
            "目前選擇："
            + "、".join(f"{SLOT_TITLES[slot]}={labels[slot]}" for slot in SLOT_IDS)
            + "。",
            open_line,
            f"決策槽是否已齊：{'是' if complete else '否'}。",
            f"訓練前預測是否過關：{'是' if quiz_unlocked else '否'}。",
            f"可否寫訓練請求：{'是' if allowed else '否'}。",
            error_line,
            artifact_note,
            f"共享決策槽狀態檔：{slots_path}",
            f"訓練請求檔：{request_path}（寫入 {{\"requested\": true}} 才會開始主教學欄動畫訓練）",
            model_code_preview(state, stage=stage),
        ]
    )
