from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FEATURE_OPTIONS = ("特徵1", "特徵2")
TARGET_COLUMN = "類別"
BUILTIN_DATA_PATH_SUFFIX = ("built-in-data", "classification", "nn_binary_400.csv")

MAX_HIDDEN_LAYERS = 8
MAX_UNITS_PER_LAYER = 32
MAX_OUTPUT_UNITS = 10
PARAM_WARN_THRESHOLD = 5000

ACTIVATION_CHOICES = ("relu", "sigmoid", "tanh", "linear")
OUTPUT_ACTIVATION_CHOICES = ACTIVATION_CHOICES + ("softmax",)
OPTIMIZER_CHOICES = ("Adam", "SGD", "RMSprop")

LOSS_AUTO = "自動"
LOSS_CHOICES = (
    LOSS_AUTO,
    "BinaryCrossentropy",
    "BinaryCrossentropy(from_logits=True)",
    "CategoricalCrossentropy",
    "CategoricalCrossentropy(from_logits=True)",
    "SparseCategoricalCrossentropy",
    "SparseCategoricalCrossentropy(from_logits=True)",
)

AXIS_LABELS = {
    "特徵1": "Temperature (Celsius)",
    "特徵2": "Duration (minutes)",
}


@dataclass(frozen=True)
class HiddenLayerSpec:
    units: int
    activation: str


@dataclass(frozen=True)
class NetworkSpec:
    input_features: tuple[str, ...]
    hidden_layers: tuple[HiddenLayerSpec, ...]
    output_units: int
    output_activation: str
    loss_choice: str = LOSS_AUTO
    use_normalization_layer: bool = False


@dataclass(frozen=True)
class CompileSpec:
    optimizer_name: str
    learning_rate: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    tile_factor: int
    random_seed: int


@dataclass(frozen=True)
class TrainResult:
    history: dict[str, list[float]]
    final_loss: float
    train_accuracy: float
    parameter_count: int


@dataclass(frozen=True)
class TrainArtifacts:
    model: Any
    result: TrainResult
    feature_normalizer: Any | None


def lab02_default_network_spec() -> NetworkSpec:
    return NetworkSpec(
        input_features=("特徵1", "特徵2"),
        hidden_layers=(HiddenLayerSpec(3, "sigmoid"),),
        output_units=1,
        output_activation="sigmoid",
        loss_choice=LOSS_AUTO,
        use_normalization_layer=False,
    )


def lab02_default_compile_spec() -> CompileSpec:
    return CompileSpec(optimizer_name="Adam", learning_rate=0.01)


def load_builtin_frame(path: Any) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = list(FEATURE_OPTIONS) + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"內建資料缺少欄位：{', '.join(missing)}")
    return frame[required].apply(pd.to_numeric, errors="coerce").dropna()


def class_count(frame: pd.DataFrame) -> int:
    labels = frame[TARGET_COLUMN].astype(int)
    return int(labels.nunique())


def validate_network_spec(
    spec: NetworkSpec,
    frame: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if not spec.input_features:
        errors.append("請至少選擇一個輸入特徵。")
    for feature in spec.input_features:
        if feature not in FEATURE_OPTIONS:
            errors.append(f"不支援的輸入特徵：{feature}")
    if len(spec.hidden_layers) > MAX_HIDDEN_LAYERS:
        errors.append(f"隱藏層數不可超過 {MAX_HIDDEN_LAYERS}。")
    for index, layer in enumerate(spec.hidden_layers, start=1):
        if layer.units < 1 or layer.units > MAX_UNITS_PER_LAYER:
            errors.append(f"第 {index} 隱藏層神經元數須在 1～{MAX_UNITS_PER_LAYER}。")
        if layer.activation not in ACTIVATION_CHOICES:
            errors.append(f"第 {index} 隱藏層激勵函數不支援：{layer.activation}")
    if spec.output_units < 1 or spec.output_units > MAX_OUTPUT_UNITS:
        errors.append(f"輸出神經元數須在 1～{MAX_OUTPUT_UNITS}。")
    if spec.output_activation not in OUTPUT_ACTIVATION_CHOICES:
        errors.append(f"輸出激勵函數不支援：{spec.output_activation}")

    n_classes = class_count(frame)
    if spec.output_units > n_classes:
        errors.append(
            f"內建資料的「{TARGET_COLUMN}」只有 {n_classes} 類，輸出神經元數 {spec.output_units} 不相容。"
        )
    if spec.output_units > 2 and spec.output_units != n_classes:
        errors.append("輸出神經元數需等於類別數，或二元任務請設為 1。")

    if not errors and spec.loss_choice != LOSS_AUTO:
        loss_errors = _validate_manual_loss(spec)
        errors.extend(loss_errors)
    return errors


def _validate_manual_loss(spec: NetworkSpec) -> list[str]:
    choice = spec.loss_choice
    out_act = spec.output_activation
    units = spec.output_units
    errors: list[str] = []

    if choice.startswith("BinaryCrossentropy"):
        if units != 1:
            errors.append("BinaryCrossentropy 僅適用輸出神經元數 = 1。")
        if "from_logits" in choice:
            if out_act != "linear":
                errors.append("BinaryCrossentropy(from_logits=True) 需搭配輸出激勵函數 linear。")
        elif out_act != "sigmoid":
            errors.append("BinaryCrossentropy() 需搭配輸出激勵函數 sigmoid。")
        return errors

    if units < 2:
        errors.append(f"{choice} 需輸出神經元數 ≥ 2（建議 softmax + 類別數一致）。")
    if "from_logits" in choice:
        if out_act != "linear":
            errors.append(f"{choice}（from_logits=True）需搭配輸出激勵函數 linear。")
    elif out_act != "softmax":
        errors.append(f"{choice}() 需搭配輸出激勵函數 softmax。")
    return errors


def resolve_loss(spec: NetworkSpec):
    tf = _import_tf()
    if spec.loss_choice != LOSS_AUTO:
        return _loss_from_choice(spec.loss_choice)

    if spec.output_units == 1:
        if spec.output_activation == "linear":
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return tf.keras.losses.BinaryCrossentropy()

    if spec.output_activation == "linear":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tf.keras.losses.SparseCategoricalCrossentropy()


def _loss_from_choice(choice: str):
    tf = _import_tf()
    mapping = {
        "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy(),
        "BinaryCrossentropy(from_logits=True)": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "CategoricalCrossentropy": tf.keras.losses.CategoricalCrossentropy(),
        "CategoricalCrossentropy(from_logits=True)": tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        ),
        "SparseCategoricalCrossentropy": tf.keras.losses.SparseCategoricalCrossentropy(),
        "SparseCategoricalCrossentropy(from_logits=True)": tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
    }
    if choice not in mapping:
        raise ValueError(f"不支援的 loss：{choice}")
    return mapping[choice]


def build_optimizer(compile_spec: CompileSpec):
    tf = _import_tf()
    learning_rate = float(compile_spec.learning_rate)
    name = compile_spec.optimizer_name
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    if name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    raise ValueError(f"不支援的優化器：{name}")


def build_sequential_model(spec: NetworkSpec):
    tf = _import_tf()
    n_features = len(spec.input_features)
    layers: list[Any] = [tf.keras.Input(shape=(n_features,))]
    if spec.use_normalization_layer:
        layers.append(tf.keras.layers.Normalization())
    for index, hidden in enumerate(spec.hidden_layers, start=1):
        layers.append(
            tf.keras.layers.Dense(
                hidden.units,
                activation=hidden.activation,
                name=f"hidden_{index}",
            )
        )
    layers.append(
        tf.keras.layers.Dense(
            spec.output_units,
            activation=spec.output_activation,
            name="output",
        )
    )
    return tf.keras.Sequential(layers)


def estimate_parameter_count(spec: NetworkSpec) -> int:
    model = build_sequential_model(spec)
    return int(model.count_params())


def encode_labels(y: np.ndarray, spec: NetworkSpec) -> np.ndarray:
    labels = np.asarray(y, dtype=float).reshape(-1)
    if spec.output_units == 1:
        return labels.astype(np.float32)
    return labels.astype(np.int32)


def tile_training_arrays(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tile_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    factor = max(int(tile_factor), 1)
    x_tiled = np.tile(x, (factor, 1))
    if y.ndim == 1:
        y_tiled = np.tile(y, factor)
    else:
        y_tiled = np.tile(y, (factor, 1))
    return x_tiled, y_tiled


def transform_features(x: np.ndarray, feature_normalizer: Any | None, spec: NetworkSpec) -> np.ndarray:
    x_array = np.asarray(x, dtype=np.float32)
    if spec.use_normalization_layer:
        return x_array
    if feature_normalizer is None:
        raise ValueError("缺少訓練時的特徵正規化器。")
    return np.asarray(feature_normalizer(x_array).numpy(), dtype=np.float32)


def fit_feature_normalizer(x: np.ndarray):
    tf = _import_tf()
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x)
    return normalizer


def train_model(
    spec: NetworkSpec,
    compile_spec: CompileSpec,
    train_config: TrainConfig,
    x: np.ndarray,
    y: np.ndarray,
) -> TrainArtifacts:
    tf = _import_tf()
    tf.random.set_seed(int(train_config.random_seed))

    x_array = np.asarray(x, dtype=np.float32)
    y_encoded = encode_labels(y, spec)
    feature_normalizer = None

    if spec.use_normalization_layer:
        x_fit = x_array
    else:
        feature_normalizer = fit_feature_normalizer(x_array)
        x_fit = feature_normalizer(x_array).numpy()

    x_fit, y_fit = tile_training_arrays(x_fit, y_encoded, tile_factor=train_config.tile_factor)

    model = build_sequential_model(spec)
    if spec.use_normalization_layer:
        model.layers[1].adapt(x_array)

    loss = resolve_loss(spec)
    optimizer = build_optimizer(compile_spec)
    metrics = ["accuracy"] if spec.output_units == 1 else ["sparse_categorical_accuracy"]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    history_obj = model.fit(
        x_fit,
        y_fit,
        epochs=int(train_config.epochs),
        verbose=0,
    )
    history = {key: [float(value) for value in values] for key, values in history_obj.history.items()}
    final_loss = float(history.get("loss", [float("nan")])[-1])
    metric_key = metrics[0]
    train_accuracy = float(history.get(metric_key, [0.0])[-1]) * 100.0

    result = TrainResult(
        history=history,
        final_loss=final_loss,
        train_accuracy=train_accuracy,
        parameter_count=int(model.count_params()),
    )
    return TrainArtifacts(model=model, result=result, feature_normalizer=feature_normalizer)


def predict_scores(
    model: Any,
    x: np.ndarray,
    spec: NetworkSpec,
    *,
    feature_normalizer: Any | None = None,
) -> np.ndarray:
    x_transformed = transform_features(x, feature_normalizer, spec)
    return np.asarray(model.predict(x_transformed, verbose=0))


def predict_class_labels(scores: np.ndarray, spec: NetworkSpec) -> np.ndarray:
    if spec.output_units == 1:
        if spec.output_activation == "linear":
            return (scores.reshape(-1) >= 0.0).astype(int)
        return (scores.reshape(-1) >= 0.5).astype(int)
    return np.argmax(scores, axis=1).astype(int)


def format_model_code(spec: NetworkSpec, compile_spec: CompileSpec) -> str:
    n_features = len(spec.input_features)
    lines = ["model = Sequential(["]
    if spec.use_normalization_layer:
        lines.append(f"    Input(shape=({n_features},)),")
        lines.append("    Normalization(),")
    else:
        lines.append(f"    Input(shape=({n_features},)),")
    for hidden in spec.hidden_layers:
        lines.append(
            f"    Dense({hidden.units}, activation={hidden.activation!r}),"
        )
    lines.append(
        f"    Dense({spec.output_units}, activation={spec.output_activation!r}),"
    )
    lines.append("])")
    loss_label = _loss_label_for_code(spec)
    lines.append("model.compile(")
    lines.append(f"    loss={loss_label},")
    lines.append(
        f"    optimizer={compile_spec.optimizer_name}(learning_rate={compile_spec.learning_rate}),"
    )
    lines.append(")")
    return "\n".join(lines)


def _loss_label_for_code(spec: NetworkSpec) -> str:
    if spec.loss_choice == LOSS_AUTO:
        if spec.output_units == 1:
            if spec.output_activation == "linear":
                return "BinaryCrossentropy(from_logits=True)"
            return "BinaryCrossentropy()"
        if spec.output_activation == "linear":
            return "SparseCategoricalCrossentropy(from_logits=True)"
        return "SparseCategoricalCrossentropy()"
    return spec.loss_choice


def build_nn_agent_context(
    *,
    spec: NetworkSpec,
    compile_spec: CompileSpec,
    train_result: TrainResult | None,
    row_count: int,
) -> str:
    hidden = ", ".join(f"{layer.units}({layer.activation})" for layer in spec.hidden_layers) or "無"
    lines = [
        "頁面：類神經網路",
        f"資料：內建 {row_count} 筆，輸入特徵 {', '.join(spec.input_features)}，目標 {TARGET_COLUMN}（0/1）",
        f"架構：Input({len(spec.input_features)}) → 隱藏[{hidden}] → Dense({spec.output_units}, {spec.output_activation})",
        f"正規化：{'Normalization 層' if spec.use_normalization_layer else '訓練前 adapt+transform'}",
        f"compile：{compile_spec.optimizer_name}(lr={compile_spec.learning_rate}), loss={_loss_label_for_code(spec)}",
    ]
    if train_result is not None:
        lines.append(
            f"最後訓練 loss={train_result.final_loss:.4f}，accuracy≈{train_result.train_accuracy:.2f}%"
        )
    return "\n".join(lines)


def _import_tf():
    import tensorflow as tf

    return tf
