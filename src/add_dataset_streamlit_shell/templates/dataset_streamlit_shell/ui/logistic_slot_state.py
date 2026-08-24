"""邏輯迴歸決策槽狀態、模型程式碼預覽、模型下載程式碼與訓練請求。"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_streamlit_shell.ml.classification import DEFAULT_MAP_DEGREE
from dataset_streamlit_shell.ui.lr_slot_state import (
    ALPHA_MAX,
    ALPHA_MIN,
    CHOICE_UNSET,
    DATA_PREVIEW_ROWS,
    LINEAR_DENSE1,
    OPT_SGD,
    SCALE_INSPECT,
    SCALE_LABELS,
    SCALE_MAXDIV,
    SCALE_MEAN,
    SCALE_METHODS,
    SCALE_MINMAX,
    SCALE_ZSCORE,
    SLOT_IDS,
    SLOT_TITLES,
    SPLIT_RANDOM_STATE,
    TRAIN_PCT_MAX,
    TRAIN_PCT_MIN,
    parse_train_pct,
    scale_inspect,
    scale_method_errors,
    split_frame_by_train_pct,
    train_test_row_counts,
)

STAGE_BOUNDARY = "boundary"
STAGE_POLY = "poly"
STAGES = (STAGE_BOUNDARY, STAGE_POLY)

DATA_ADMISSION = "admission"
DATA_MICROCHIP = "microchip"
LOSS_LOGLOSS = "logloss"

DATA_LABELS = {
    DATA_ADMISSION: "內建大學錄取",
    DATA_MICROCHIP: "內建微晶片檢測",
}

LOCKED_DATA = {
    STAGE_BOUNDARY: DATA_ADMISSION,
    STAGE_POLY: DATA_MICROCHIP,
}

DEFAULT_ALPHA = {
    STAGE_BOUNDARY: 0.001,
    STAGE_POLY: 0.01,
}
DEFAULT_EPOCHS = {
    STAGE_BOUNDARY: 10000,
    STAGE_POLY: 10000,
}
DEFAULT_LAMBDA = 0.01

EPOCHS_MIN = 1
EPOCHS_MAX = 20000
LAMBDA_MIN = 0.0
LAMBDA_MAX = 100.0

ADMISSION_FEATURES = ["考試1分數", "考試2分數"]
ADMISSION_TARGET = "是否錄取"
MICROCHIP_FEATURES = ["檢測分數1", "檢測分數2"]
MICROCHIP_TARGET = "是否通過"

_CLASSIFICATION_DATA_DIR = Path(__file__).resolve().parents[1] / "built-in-data" / "classification"
_DOWNLOAD_CSV = {
    STAGE_BOUNDARY: "university_admission.csv",
    STAGE_POLY: "microchip_test.csv",
}

_DOWNLOAD_PYPROJECT = """\
[project]
name = "logistic-model-download"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas",
    "scikit-learn",
    "tensorflow-cpu",
]
"""

_DOWNLOAD_README = """\
# 模型下載程式碼

這份專案可在本頁以外執行。主教學欄動畫走頁面自己的梯度下降，不是跑這支程式。

```
uv sync
uv run python train.py
```
"""


def empty_slot_state(stage: str = STAGE_BOUNDARY) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return {
        "choices": {
            slot: LOCKED_DATA[stage] if slot == "data" else None for slot in SLOT_IDS
        },
        "alpha": None,
        "epochs": None,
        "lambda_": None,
    }


def empty_workspace_state() -> dict[str, Any]:
    return {stage: empty_slot_state(stage) for stage in STAGES}


def default_slot_state(stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return {
        "choices": {
            "data": LOCKED_DATA[stage],
            "split": None,
            "scale": SCALE_ZSCORE,
            "linear": LINEAR_DENSE1,
            "loss": LOSS_LOGLOSS,
            "opt": OPT_SGD,
        },
        "alpha": DEFAULT_ALPHA[stage],
        "epochs": DEFAULT_EPOCHS[stage],
        "lambda_": DEFAULT_LAMBDA if stage == STAGE_POLY else None,
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


def parse_lambda(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < LAMBDA_MIN or parsed > LAMBDA_MAX:
        return None
    return parsed


def normalize_slot_state(raw: dict[str, Any] | None, *, stage: str) -> dict[str, Any]:
    base = empty_slot_state(stage)
    if not isinstance(raw, dict):
        return base
    incoming = raw.get("choices") if isinstance(raw.get("choices"), dict) else {}

    split = parse_train_pct(incoming.get("split"))
    if split is not None:
        base["choices"]["split"] = split

    scale = incoming.get("scale")
    if scale in SCALE_METHODS:
        base["choices"]["scale"] = scale

    if incoming.get("linear") == LINEAR_DENSE1:
        base["choices"]["linear"] = LINEAR_DENSE1
    if incoming.get("loss") == LOSS_LOGLOSS:
        base["choices"]["loss"] = LOSS_LOGLOSS
    if incoming.get("opt") == OPT_SGD:
        base["choices"]["opt"] = OPT_SGD

    if raw.get("alpha") is not None:
        base["alpha"] = _clamp_alpha(raw.get("alpha"), None)
    if raw.get("epochs") is not None:
        base["epochs"] = _clamp_epochs(raw.get("epochs"), None)
    if stage == STAGE_POLY and raw.get("lambda_") is not None:
        parsed = parse_lambda(raw.get("lambda_"))
        if parsed is not None:
            base["lambda_"] = parsed
    return base


def apply_slot_write(
    previous: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
    *,
    stage: str,
) -> dict[str, Any]:
    current = normalize_slot_state(previous, stage=stage)
    raw = incoming if isinstance(incoming, dict) else {}
    choices = raw.get("choices") if isinstance(raw.get("choices"), dict) else {}

    next_state = {
        "choices": dict(current["choices"]),
        "alpha": current["alpha"],
        "epochs": current["epochs"],
        "lambda_": current["lambda_"],
    }

    locked_data = LOCKED_DATA[stage]
    if "data" in choices:
        if choices["data"] == locked_data:
            next_state["choices"]["data"] = locked_data
        elif choices["data"] is not None:
            next_state["choices"]["data"] = current["choices"]["data"]

    if "split" in choices:
        parsed = parse_train_pct(choices["split"])
        if parsed is not None:
            next_state["choices"]["split"] = parsed
        elif choices["split"] is not None:
            next_state["choices"]["split"] = current["choices"]["split"]

    if "scale" in choices:
        if choices["scale"] in SCALE_METHODS:
            next_state["choices"]["scale"] = choices["scale"]
        elif choices["scale"] is not None:
            next_state["choices"]["scale"] = current["choices"]["scale"]

    for slot, allowed in (
        ("linear", LINEAR_DENSE1),
        ("loss", LOSS_LOGLOSS),
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
    if stage == STAGE_POLY and "lambda_" in raw:
        parsed = parse_lambda(raw.get("lambda_"))
        if parsed is not None:
            next_state["lambda_"] = parsed
        elif raw.get("lambda_") is None:
            next_state["lambda_"] = current["lambda_"]
    else:
        next_state["lambda_"] = None if stage != STAGE_POLY else next_state["lambda_"]
    if stage != STAGE_POLY:
        next_state["lambda_"] = None
    return next_state


def slots_are_complete(state: dict[str, Any]) -> bool:
    choices = state.get("choices") if isinstance(state.get("choices"), dict) else {}
    if any(choices.get(slot) is None for slot in SLOT_IDS):
        return False
    if state.get("alpha") is None or state.get("epochs") is None:
        return False
    if choices.get("data") == DATA_MICROCHIP and state.get("lambda_") is None:
        return False
    return True


def slot_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    choices = state.get("choices") if isinstance(state.get("choices"), dict) else {}
    return (
        choices.get("data"),
        choices.get("split"),
        choices.get("scale"),
        choices.get("linear"),
        choices.get("loss"),
        choices.get("opt"),
        state.get("alpha"),
        state.get("epochs"),
        state.get("lambda_"),
    )


def slot_label(slot_id: str, state: dict[str, Any], stage: str) -> str:
    choices = state.get("choices") if isinstance(state.get("choices"), dict) else {}
    choice = choices.get(slot_id)
    if choice is None:
        return CHOICE_UNSET
    if slot_id == "data":
        return DATA_LABELS.get(str(choice), str(choice))
    if slot_id == "split":
        return f"訓練 {int(choice)}%"
    if slot_id == "scale":
        return SCALE_LABELS.get(str(choice), str(choice))
    if slot_id == "linear":
        return "Dense(1, linear)"
    if slot_id == "loss":
        if stage == STAGE_POLY and state.get("lambda_") is not None:
            return f"對數損失，λ={float(state['lambda_']):g}"
        return "對數損失"
    if slot_id == "opt":
        alpha = state.get("alpha")
        epochs = state.get("epochs")
        if alpha is None or epochs is None:
            return "SGD"
        return f"SGD · α={float(alpha):g} · {int(epochs)} epochs"
    return str(choice)


def slot_is_filled(slot_id: str, state: dict[str, Any], stage: str | None = None) -> bool:
    choices = state.get("choices") if isinstance(state.get("choices"), dict) else {}
    if slot_id == "opt":
        return (
            choices.get("opt") is not None
            and state.get("alpha") is not None
            and state.get("epochs") is not None
        )
    if slot_id == "loss":
        if choices.get("loss") is None:
            return False
        data = choices.get("data")
        if data == DATA_MICROCHIP or stage == STAGE_POLY:
            return state.get("lambda_") is not None
        return True
    return choices.get(slot_id) is not None


def slot_button_label(slot_id: str, state: dict[str, Any], stage: str) -> str:
    title = SLOT_TITLES[slot_id]
    if not slot_is_filled(slot_id, state, stage):
        return title
    return f"{title}\n{slot_label(slot_id, state, stage)}"


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
        if stage == STAGE_BOUNDARY:
            rows = [
                ("目前選擇", DATA_LABELS[DATA_ADMISSION]),
                ("x", "考試1分數、考試2分數"),
                ("y", "是否錄取"),
            ]
        else:
            rows = [
                ("目前選擇", DATA_LABELS[DATA_MICROCHIP]),
                ("x", "檢測分數1、檢測分數2"),
                ("y", "是否通過"),
                ("多項式映射", f"degree={DEFAULT_MAP_DEGREE}（鎖定；先縮放再映射）"),
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
        if stage == STAGE_POLY:
            rows.append(("順序", "先縮放原始兩特徵，再做多項式映射"))
        rows.append(
            ("可討論", "正規化、最小-最大正規化、平均值正規化、Z分數正規化")
        )
        return rows
    if slot_id == "split":
        pct = int(choice)
        rows = [
            ("目前選擇", f"訓練 {pct}%"),
            ("測試", f"{100 - pct}%"),
        ]
        if row_count is not None:
            train_n, test_n = train_test_row_counts(int(row_count), pct)
            rows.append(("訓練列數", str(train_n)))
            rows.append(("測試列數", str(test_n)))
        rows.append(("可否改", "請 Agent 寫訓練占比 1–99"))
        return rows
    if slot_id == "linear":
        rows = [
            ("目前選擇", "Dense(1, linear)"),
            ("輸出", "網路輸出 z；圖與表的 ŷ 為 σ(z)"),
        ]
        if stage == STAGE_POLY:
            rows.append(("φ", f"degree={DEFAULT_MAP_DEGREE} 多項式特徵映射（鎖定；先縮放再映射）"))
        rows.extend(
            [
                ("z", r"z = w·φ(x)+b" if stage == STAGE_POLY else r"z = w·x+b"),
                ("ŷ", r"ŷ = σ(z) = 1/(1+e^{-z})"),
                ("可否改", "第一版鎖線性層種類"),
            ]
        )
        return rows
    if slot_id == "loss":
        rows = [
            ("目前選擇", slot_label("loss", state, stage)),
            ("對應", "對數損失；compile 見模型程式碼預覽"),
            ("可否改", "種類鎖對數損失" + ("；可請 Agent 改 λ" if stage == STAGE_POLY else "")),
        ]
        if stage == STAGE_POLY:
            rows.append(
                (
                    "λ",
                    "尚未寫入"
                    if state.get("lambda_") is None
                    else f"{float(state['lambda_']):g}",
                )
            )
        return rows
    return [
        ("目前選擇", "SGD"),
        ("α", "尚未寫入" if state.get("alpha") is None else f"{float(state['alpha']):g}"),
        ("epochs", "尚未寫入" if state.get("epochs") is None else str(int(state["epochs"]))),
        ("可否改", "種類鎖 SGD；可請 Agent 改 α 與 epochs"),
    ]


def _feature_count(stage: str) -> int:
    if stage == STAGE_POLY:
        return 27
    return 2


def model_sequential_compile_preview(state: dict[str, Any], *, stage: str) -> str:
    if not slots_are_complete(state):
        return ""
    n_features = _feature_count(stage)
    alpha = float(state["alpha"])
    dense_line = '    Dense(1, activation="linear"),'
    if stage == STAGE_POLY:
        lam = float(state["lambda_"])
        dense_line = (
            f'    Dense(1, activation="linear", kernel_regularizer=l2({lam:g})),'
        )
    return "\n".join(
        [
            "model = Sequential([",
            f"    Input(shape=({n_features},)),",
            dense_line,
            "])",
            "model.compile(",
            "    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),",
            f"    optimizer=SGD(learning_rate={alpha:g}),",
            ")",
        ]
    )


def model_code_preview(state: dict[str, Any], *, stage: str) -> str:
    if not slots_are_complete(state):
        return "# 決策槽還沒選齊，沒有模型程式碼預覽。"
    scale = str(state["choices"]["scale"])
    info = scale_inspect(scale)
    extra = ""
    if stage == STAGE_POLY:
        extra = (
            f"# 先縮放原始兩特徵，再做 degree={DEFAULT_MAP_DEGREE} 多項式映射（27 維）。\n"
        )
    return f"{info['code']}\n{extra}{model_sequential_compile_preview(state, stage=stage)}"


def _download_columns(stage: str) -> tuple[list[str], str]:
    if stage == STAGE_BOUNDARY:
        return list(ADMISSION_FEATURES), ADMISSION_TARGET
    return list(MICROCHIP_FEATURES), MICROCHIP_TARGET


def _csv_training_row_count(csv_path: Path, features: list[str], target: str) -> int:
    frame = pd.read_csv(csv_path)
    working = frame[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    return len(working)


def _download_scale_block(method: str) -> tuple[str, str]:
    if method == SCALE_MINMAX:
        return (
            "from sklearn.preprocessing import MinMaxScaler\n",
            "scaler = MinMaxScaler()\n"
            "x_scaled = scaler.fit_transform(x_train)\n"
            "x_test_scaled = scaler.transform(x_test)\n",
        )
    if method == SCALE_MAXDIV:
        return (
            "",
            "x_train_n = x_train.to_numpy(dtype=float)\n"
            "x_test_n = x_test.to_numpy(dtype=float)\n"
            "xmax = x_train_n.max(axis=0)\n"
            "x_scaled = x_train_n / xmax\n"
            "x_test_scaled = x_test_n / xmax\n",
        )
    if method == SCALE_MEAN:
        return (
            "",
            "x_train_n = x_train.to_numpy(dtype=float)\n"
            "x_test_n = x_test.to_numpy(dtype=float)\n"
            "mean = x_train_n.mean(axis=0)\n"
            "span = x_train_n.max(axis=0) - x_train_n.min(axis=0)\n"
            "x_scaled = (x_train_n - mean) / span\n"
            "x_test_scaled = (x_test_n - mean) / span\n",
        )
    return (
        "from sklearn.preprocessing import StandardScaler\n",
        "scaler = StandardScaler()\n"
        "x_scaled = scaler.fit_transform(x_train)\n"
        "x_test_scaled = scaler.transform(x_test)\n",
    )


def _map_feature_helper() -> str:
    return (
        "def map_feature(x, degree=6):\n"
        "    x1 = x[:, 0]\n"
        "    x2 = x[:, 1]\n"
        "    cols = []\n"
        "    for total in range(1, degree + 1):\n"
        "        for p2 in range(total + 1):\n"
        "            p1 = total - p2\n"
        "            cols.append((x1 ** p1) * (x2 ** p2))\n"
        "    import numpy as np\n"
        "    return np.column_stack(cols)\n"
        "\n"
    )


def _download_train_script(state: dict[str, Any], *, stage: str, csv_name: str, row_count: int) -> str:
    features, target = _download_columns(stage)
    train_pct = parse_train_pct((state.get("choices") or {}).get("split"))
    assert train_pct is not None
    train_n, _ = train_test_row_counts(row_count, train_pct)
    method = str(state["choices"]["scale"])
    scale_import, scale_block = _download_scale_block(method)
    sequential = model_sequential_compile_preview(state, stage=stage)
    feature_list = ", ".join(repr(name) for name in features)
    epochs = int(state["epochs"])
    regularizer_import = (
        "from tensorflow.keras.regularizers import l2\n" if stage == STAGE_POLY else ""
    )
    map_block = ""
    if stage == STAGE_POLY:
        map_block = (
            _map_feature_helper()
            + "x = map_feature(x_scaled, degree=6)\n"
            + "x_test = map_feature(x_test_scaled, degree=6)\n"
        )
    else:
        map_block = "x = x_scaled\nx_test = x_test_scaled\n"
    note = (
        "# 階段「多項式與 λ」的 kernel_regularizer=l2(λ) 與課堂 λ/(2m) 係數可能不同。\n"
        if stage == STAGE_POLY
        else ""
    )
    return (
        "# 模型下載程式碼：可在本頁以外執行。\n"
        "# 主教學欄動畫走本頁梯度下降，與這支 Keras 程式不同運算。\n"
        f"{note}"
        "import tensorflow as tf\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        f"{scale_import}"
        "from tensorflow.keras.layers import Dense, Input\n"
        "from tensorflow.keras.models import Sequential\n"
        "from tensorflow.keras.optimizers import SGD\n"
        f"{regularizer_import}"
        "\n"
        f'df = pd.read_csv("{csv_name}")\n'
        f"features = [{feature_list}]\n"
        f"target = {target!r}\n"
        "x_all = df[features]\n"
        "y_all = df[target]\n"
        "x_train, x_test, y_train, y_test = train_test_split(\n"
        f"    x_all, y_all, train_size={train_n}, random_state={SPLIT_RANDOM_STATE}, shuffle=True\n"
        ")\n"
        f"{scale_block}"
        f"{map_block}"
        "y = y_train.to_numpy(dtype=float)\n"
        "y_test = y_test.to_numpy(dtype=float)\n"
        "\n"
        f"{sequential}\n"
        f"model.fit(x, y, epochs={epochs}, verbose=0)\n"
        'print("train log-loss:", float(model.evaluate(x, y, verbose=0)))\n'
        'print("test log-loss:", float(model.evaluate(x_test, y_test, verbose=0)))\n'
        "logits = model.predict(x, verbose=0).reshape(-1)\n"
        "pred = (tf.sigmoid(logits).numpy() >= 0.5).astype(int)\n"
        'print("train accuracy:", float((pred == y).mean()))\n'
        "logits_t = model.predict(x_test, verbose=0).reshape(-1)\n"
        "pred_t = (tf.sigmoid(logits_t).numpy() >= 0.5).astype(int)\n"
        'print("test accuracy:", float((pred_t == y_test).mean()))\n'
        "y_true = y_test.astype(int)\n"
        "y_hat = pred_t.astype(int)\n"
        "tp = int(((y_true == 1) & (y_hat == 1)).sum())\n"
        "fp = int(((y_true == 0) & (y_hat == 1)).sum())\n"
        "fn = int(((y_true == 1) & (y_hat == 0)).sum())\n"
        "precision = (100.0 * tp / (tp + fp)) if (tp + fp) else None\n"
        "recall = (100.0 * tp / (tp + fn)) if (tp + fn) else None\n"
        "f1 = None if precision is None or recall is None else (\n"
        "    0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)\n"
        ")\n"
        'print("test precision:", f"{precision:.1f}%" if precision is not None else "—")\n'
        'print("test recall:", f"{recall:.1f}%" if recall is not None else "—")\n'
        'print("test F1:", f"{f1:.1f}%" if f1 is not None else "—")\n'
        "weights, intercept = model.layers[-1].get_weights()\n"
        'print("w:", weights.reshape(-1).tolist())\n'
        'print("b:", float(intercept.reshape(-1)[0]))\n'
    )


def model_download_files(state: dict[str, Any], *, stage: str) -> dict[str, bytes] | None:
    if stage not in STAGES or not slots_are_complete(state):
        return None
    csv_name = _DOWNLOAD_CSV[stage]
    csv_path = _CLASSIFICATION_DATA_DIR / csv_name
    features, target = _download_columns(stage)
    row_count = _csv_training_row_count(csv_path, features, target)
    script = _download_train_script(state, stage=stage, csv_name=csv_name, row_count=row_count)
    return {
        "pyproject.toml": _DOWNLOAD_PYPROJECT.encode("utf-8"),
        "README.md": _DOWNLOAD_README.encode("utf-8"),
        "train.py": script.encode("utf-8"),
        csv_name: csv_path.read_bytes(),
    }


def model_download_zip_name(stage: str) -> str:
    label = "線性邊界" if stage == STAGE_BOUNDARY else "多項式與λ"
    return f"模型下載程式碼-{label}.zip"


def model_download_zip_bytes(state: dict[str, Any], *, stage: str) -> bytes | None:
    files = model_download_files(state, stage=stage)
    if files is None:
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


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


def logistic_slots_path(workspace_dir: Path) -> Path:
    return workspace_dir / "logistic_slots.json"


def logistic_train_request_path(workspace_dir: Path) -> Path:
    return workspace_dir / "logistic_train_request.json"


def save_workspace_state(workspace_dir: Path, state: dict[str, Any]) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        stage: normalize_slot_state(state.get(stage), stage=stage) for stage in STAGES
    }
    logistic_slots_path(workspace_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_raw_workspace(workspace_dir: Path) -> dict[str, Any] | None:
    path = logistic_slots_path(workspace_dir)
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
    path = logistic_slots_path(workspace_dir)
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def should_rerun_after_logistic_chat(
    *,
    requested: bool,
    slots_mtime: float,
    applied_mtime: float,
) -> bool:
    return bool(requested) or slots_mtime > applied_mtime


def train_request_is_set(workspace_dir: Path) -> bool:
    path = logistic_train_request_path(workspace_dir)
    if not path.is_file():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and raw.get("requested") is True


def write_train_request(workspace_dir: Path) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    logistic_train_request_path(workspace_dir).write_text(
        json.dumps({"requested": True}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def clear_train_request(workspace_dir: Path) -> None:
    path = logistic_train_request_path(workspace_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def consume_train_request(
    workspace_dir: Path,
    *,
    allowed: bool,
) -> bool:
    if not train_request_is_set(workspace_dir):
        return False
    clear_train_request(workspace_dir)
    return bool(allowed)


def logistic_host_context_fragment(
    *,
    slots_path: str,
    request_path: str,
) -> str:
    return (
        "【邏輯迴歸頁】主教學欄是決策槽列，不是教學流程圖，也不是類神經網路 form。"
        "對學生講六個決策槽：輸入資料、訓練／測試切分、特徵縮放、線性層、損失函數、優化器。"
        "輸入資料進頁即完成。其餘各格只在學生對該格提出要求後才寫入；不要叫學生在頁上自己選。"
        "主教學欄沒有下拉選單，也不要用打勾符號標完成；完成是綠框、第二行目前選擇；點框只看只讀選擇明細。"
        f"決策槽狀態在共享 JSON：{slots_path}，鍵為 boundary／poly 兩學習階段，"
        "每階段含 choices（data、split、scale、linear、loss、opt）與 alpha、epochs；"
        "poly 另含 lambda_。"
        "輸入資料進頁即為該階段鎖定值（線性邊界 admission、多項式與 λ microchip），載入時也回成該值；"
        "不必為了槽齊而寫 data，寫鎖定值可以，寫其他來源拒絕。"
        "尚未被要求的 split／scale／linear／loss／opt 與 alpha、epochs 為尚未選擇；poly 的 lambda_ 亦然。"
        "只說組模型：列出仍尚未選擇的格與各格可寫入的值，當輪不寫入。"
        "對學生用課堂名稱：線性層 Dense(1, linear)、損失對數損失、優化器 SGD、特徵縮放建議 Z分數正規化；"
        "寫入 JSON 才用 linear=dense1、loss=logloss、opt=sgd、scale=zscore；"
        "優化器課堂建議 alpha／epochs 線性邊界 0.001／10000，多項式與 λ 0.01／10000；poly 損失建議 λ=0.01（JSON lambda_）。"
        "只講訓練／測試切分占比時只寫 choices.split，不得同時寫其他格"
        "（訓練集整數占比 1–99，測試為其餘；0 與 100 拒絕）。"
        "同一句已說其餘／剩下用預設，或學生確認已列出的格之後，才寫被涵蓋的格；"
        "回覆須點名各格寫入值。切分沒數字就不寫 split。"
        "學生同一句要求多格時可一次寫入被涵蓋的格。"
        "鎖定格由你講明鎖定值，學生確認該格後才寫；不要要學生背 Dense(1, linear)。"
        "寫決策槽狀態時不要同時寫訓練請求，也不要自行開始訓練。"
        "可改的有 split（1–99）與 scale（maxdiv／minmax／mean／zscore）與 alpha、epochs，以及 poly 的 lambda_。"
        "data／linear／loss／opt 是鎖定槽，種類拒絕改成其他值。"
        "損失框上只寫「對數損失」（poly 併 λ），不要把 BinaryCrossentropy 或 from_logits 寫進框上。"
        "模型程式碼預覽與下載的 compile 才是 BinaryCrossentropy(from_logits=True)，線性層 Dense(1, linear)。"
        "poly 先縮放原始兩特徵再做 degree=6 映射；degree 鎖定。"
        "頁上沒有切分旋鈕、α／epochs／λ 旋鈕。"
        "訓練畫面是決策邊界或 contour、訓練／測試 Cost，下方為測試集混淆矩陣與其下 precision／recall／F1；"
        "開訓後三張圖與這三個數逐幀一起更新。"
        "快照含與動畫同一套的訓練／測試 Cost 曲線（約 80 點）與最後數字、測試集混淆矩陣四格計數、"
        "測試集 precision／recall／F1 與訓練／測試正確率。"
        "禁止說看不到測試 Cost。"
        "禁止說看不到測試集 precision／recall／F1。"
        "階段「多項式與 λ」的訓練 Cost 含課堂 λ/(2m)||w||²；測試 Cost 只算對數損失、不含 λ。"
        "階段「多項式與 λ」的訓練 Cost 含 λ 項，通常會高過測試 Cost；看分叉與測試是否回升，不要只比誰比較小。"
        "測試 Cost 相對訓練上升或分叉時，可用課堂說法講過擬合傾向。"
        "測試集 precision／recall／F1 以 y=1 為正類，百分比一位小數；無法計算為 —。"
        "分類 threshold 不是決策槽，訓後才出現，只改測試集混淆矩陣與其下三數與表，不重畫決策邊界。"
        "write_file 時必須保留另一學習階段的鍵，不要清掉另一側。"
        "若要讓主教學欄播放與「開始訓練」相同的動畫，另寫 "
        f'{request_path}，內容為 {{"requested": true}}。'
        "決策槽未齊、訓練前預測未過關、或正規化（除以最大）遇上負值時，不准寫訓練請求。"
        "不要自行 exec 訓練、不要假裝已訓完、不要代填訓練前預測選項。"
        "不要改 nn_form.json、lr_slots.json 或線性回歸的訓練請求。"
        "六槽齊後主教學欄預覽 expander 可下載該階段模型下載程式碼"
        "（獨立 Keras 專案，含內建表；不是模型程式碼預覽，也不是已訓權重）。"
        "下載不必過關、不必先訓；訓練請求仍要過關。"
        "可告訴學生何時能下、以及與預覽／本頁梯度下降的差別。"
        "不要另貼一份程式，不要宣稱頁面 exec 或在跑模型下載程式碼，也不要把它寫進 workspace。"
    )


def build_logistic_page_snapshot(
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
    labels = {slot: slot_label(slot, state, stage) for slot in SLOT_IDS}
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
    stage_name = "線性邊界" if stage == STAGE_BOUNDARY else "多項式與 λ"
    return "\n".join(
        [
            f"目前學習階段：{stage_name}。",
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
