from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

INPUT_SIZE = (64, 64)


@dataclass(frozen=True)
class MiniTrainImage:
    path: Path
    label: int
    label_name: str


@dataclass(frozen=True)
class MiniTrainBatch:
    images: np.ndarray
    labels: np.ndarray
    items: tuple[MiniTrainImage, ...]


@dataclass(frozen=True)
class MiniTrainEpochSnapshot:
    epoch: int
    loss: float
    accuracy: float
    feature_maps: np.ndarray
    validation_predictions: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class MiniTrainResult:
    history: dict[str, list[float]]
    snapshots: tuple[MiniTrainEpochSnapshot, ...]


def load_mini_train_batch(cat_paths: list[Path], dog_paths: list[Path]) -> MiniTrainBatch:
    items: list[MiniTrainImage] = []
    images: list[np.ndarray] = []
    labels: list[int] = []
    for path in cat_paths:
        array = _load_resized(path)
        images.append(array)
        labels.append(0)
        items.append(MiniTrainImage(path=path, label=0, label_name="cat"))
    for path in dog_paths:
        array = _load_resized(path)
        images.append(array)
        labels.append(1)
        items.append(MiniTrainImage(path=path, label=1, label_name="dog"))
    stacked = np.stack(images, axis=0).astype(np.float32) / 255.0
    return MiniTrainBatch(
        images=stacked,
        labels=np.array(labels, dtype=np.int32),
        items=tuple(items),
    )


def build_mini_cnn() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*INPUT_SIZE, 3))
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="feature_conv")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="mini_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_with_history(
    batch: MiniTrainBatch,
    *,
    epochs: int = 10,
    validation_items: tuple[MiniTrainImage, ...] | None = None,
) -> MiniTrainResult:
    model = build_mini_cnn()
    feature_layer = model.get_layer("feature_conv")
    feature_extractor = tf.keras.Model(model.input, feature_layer.output)

    history: dict[str, list[float]] = {"loss": [], "accuracy": []}
    snapshots: list[MiniTrainEpochSnapshot] = []
    snapshot_epochs = _snapshot_epochs(epochs)
    validation_items = validation_items or tuple(batch.items[:2])

    for epoch in range(1, epochs + 1):
        record = model.fit(
            batch.images,
            batch.labels,
            epochs=1,
            batch_size=8,
            verbose=0,
        )
        loss = float(record.history["loss"][0])
        accuracy = float(record.history["accuracy"][0])
        history["loss"].append(loss)
        history["accuracy"].append(accuracy)

        if epoch in snapshot_epochs:
            feature_maps = _first_feature_maps(feature_extractor, batch.images[0])
            validation_predictions = _validation_predictions(model, validation_items)
            snapshots.append(
                MiniTrainEpochSnapshot(
                    epoch=epoch,
                    loss=loss,
                    accuracy=accuracy,
                    feature_maps=feature_maps,
                    validation_predictions=validation_predictions,
                )
            )

    return MiniTrainResult(history=history, snapshots=tuple(snapshots))


def _snapshot_epochs(epochs: int) -> set[int]:
    if epochs <= 1:
        return {1}
    middle = max(2, epochs // 2)
    return {1, middle, epochs}


def _load_resized(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize(INPUT_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _first_feature_maps(feature_extractor: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    batch = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)
    maps = feature_extractor.predict(batch, verbose=0)[0]
    channels = min(16, maps.shape[-1])
    normalized = []
    for index in range(channels):
        channel = maps[..., index]
        minimum = float(np.min(channel))
        maximum = float(np.max(channel))
        if maximum - minimum < 1e-8:
            scaled = np.zeros_like(channel)
        else:
            scaled = (channel - minimum) / (maximum - minimum)
        normalized.append(scaled)
    return np.stack(normalized, axis=-1)


def _validation_predictions(
    model: tf.keras.Model,
    validation_items: tuple[MiniTrainImage, ...],
) -> tuple[tuple[str, float], ...]:
    predictions: list[tuple[str, float]] = []
    for item in validation_items:
        image = _load_resized(item.path).astype(np.float32) / 255.0
        batch = np.expand_dims(image, axis=0)
        probs = model.predict(batch, verbose=0)[0]
        label_index = int(np.argmax(probs))
        label_name = "dog" if label_index == 1 else "cat"
        predictions.append((label_name, float(probs[label_index])))
    return tuple(predictions)
