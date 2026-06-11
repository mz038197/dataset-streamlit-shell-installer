from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.imagenet_utils import CLASS_INDEX
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input as resnet_preprocess

BackboneName = Literal["resnet50", "mobilenet_v2"]
DEFAULT_BACKBONE: BackboneName = "resnet50"
INPUT_SIZE = (224, 224)

RESNET50_STAGES: tuple[tuple[str, str, str], ...] = (
    ("Stage 0", "original", "Original RGB input (224×224)."),
    ("Stage 1", "conv1_relu", "Early edges and color blobs."),
    ("Stage 2", "pool1_pool", "Downsampled activations after max pooling."),
    ("Stage 3", "conv2_block3_out", "Textures and simple parts."),
    ("Stage 4", "conv3_block4_out", "Mid-level object parts."),
    ("Stage 5", "conv4_block6_out", "Richer semantic structure."),
    ("Stage 6", "conv5_block3_out", "High-level semantics before pooling."),
    ("Stage 7", "avg_pool", "Global pooled feature vector."),
    ("Stage 8", "predictions", "Softmax probabilities over ImageNet classes."),
)

RESNET50_LAST_CONV = "conv5_block3_out"
MOBILENET_LAST_CONV = "block_16_expand_relu"


@dataclass(frozen=True)
class PredictionItem:
    rank: int
    synset_id: str
    class_index: int
    label: str
    probability: float


@dataclass(frozen=True)
class ClassificationResult:
    backbone: BackboneName
    top_items: tuple[PredictionItem, ...]
    preprocessed_preview: np.ndarray


@dataclass(frozen=True)
class StageActivation:
    stage_id: str
    title: str
    caption: str
    shape_label: str
    feature_maps: np.ndarray | None
    vector_preview: np.ndarray | None


def _preprocess(image: np.ndarray, backbone: BackboneName) -> np.ndarray:
    resized = tf.image.resize(image, INPUT_SIZE)
    batch = tf.expand_dims(resized, axis=0)
    if backbone == "resnet50":
        return resnet_preprocess(batch)
    return mobilenet_preprocess(batch)


def load_classifier(backbone: BackboneName) -> tf.keras.Model:
    if backbone == "resnet50":
        return ResNet50(weights="imagenet")
    return MobileNetV2(weights="imagenet")


def last_conv_layer_name(backbone: BackboneName) -> str:
    return RESNET50_LAST_CONV if backbone == "resnet50" else MOBILENET_LAST_CONV


def predict_top_k(
    image: np.ndarray,
    backbone: BackboneName,
    *,
    k: int = 5,
    model: tf.keras.Model | None = None,
) -> ClassificationResult:
    classifier = model or load_classifier(backbone)
    batch = _preprocess(image, backbone)
    predictions = classifier.predict(batch, verbose=0)
    decoded = decode_predictions(predictions, top=max(k, 1))[0]
    items = tuple(
        PredictionItem(
            rank=index + 1,
            synset_id=synset_id,
            class_index=int(CLASS_INDEX[synset_id]),
            label=label.replace("_", " "),
            probability=float(score),
        )
        for index, (synset_id, label, score) in enumerate(decoded)
    )
    preview = np.clip(batch[0] + 1.0, 0.0, 2.0) / 2.0 if backbone == "resnet50" else batch[0]
    if backbone == "mobilenet_v2":
        preview = np.clip((preview + 1.0) / 2.0, 0.0, 1.0)
    preview_rgb = (preview * 255.0).astype(np.uint8)
    return ClassificationResult(
        backbone=backbone,
        top_items=items,
        preprocessed_preview=preview_rgb,
    )


def format_top_prediction_summary(items: tuple[PredictionItem, ...]) -> str:
    if not items:
        return "No predictions available."
    top = items[0]
    if len(items) == 1:
        return f"Top prediction: {top.label} ({top.probability:.1%})."
    second = items[1]
    if top.label.split()[0] == second.label.split()[0]:
        return (
            f"Model predicts {top.label} at {top.probability:.1%}; "
            f"runner-up {second.label} at {second.probability:.1%} suggests a related category."
        )
    return (
        f"Model predicts {top.label} at {top.probability:.1%}; "
        f"runner-up {second.label} at {second.probability:.1%}."
    )


def extract_stage_activations(
    image: np.ndarray,
    backbone: BackboneName,
    *,
    model: tf.keras.Model | None = None,
    max_channels: int = 16,
) -> list[StageActivation]:
    if backbone != "resnet50":
        backbone = "resnet50"
    classifier = model or load_classifier(backbone)
    batch = _preprocess(image, backbone)

    layer_names = [stage[1] for stage in RESNET50_STAGES if stage[1] not in {"original", "predictions"}]
    outputs = [classifier.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(classifier.input, outputs)
    activations = activation_model.predict(batch, verbose=0)
    predictions = classifier.predict(batch, verbose=0)

    stages: list[StageActivation] = []
    for title, stage_id, caption in RESNET50_STAGES:
        if stage_id == "original":
            original = tf.image.resize(image, INPUT_SIZE).numpy().astype(np.uint8)
            stages.append(
                StageActivation(
                    stage_id=stage_id,
                    title=title,
                    caption=caption,
                    shape_label=f"{original.shape[0]}×{original.shape[1]}×{original.shape[2]}",
                    feature_maps=None,
                    vector_preview=original,
                )
            )
            continue
        if stage_id == "predictions":
            decoded = decode_predictions(predictions, top=5)[0]
            stages.append(
                StageActivation(
                    stage_id=stage_id,
                    title=title,
                    caption=caption,
                    shape_label=f"top-5 from {predictions.shape[-1]} classes",
                    feature_maps=None,
                    vector_preview=np.array(
                        [CLASS_INDEX[synset_id] for synset_id, _, _ in decoded],
                        dtype=np.int32,
                    ),
                )
            )
            continue

        layer_index = layer_names.index(stage_id)
        tensor = activations[layer_index][0]
        if tensor.ndim == 3:
            maps = tensor[..., :max_channels]
            shape_label = f"{maps.shape[0]}×{maps.shape[1]}×{tensor.shape[-1]}"
            stages.append(
                StageActivation(
                    stage_id=stage_id,
                    title=title,
                    caption=caption,
                    shape_label=shape_label,
                    feature_maps=_normalize_feature_maps(maps),
                    vector_preview=None,
                )
            )
        else:
            stages.append(
                StageActivation(
                    stage_id=stage_id,
                    title=title,
                    caption=caption,
                    shape_label=str(tuple(tensor.shape)),
                    feature_maps=None,
                    vector_preview=tensor,
                )
            )
    return stages


def _normalize_feature_maps(maps: np.ndarray) -> np.ndarray:
    normalized = []
    for channel in range(maps.shape[-1]):
        feature = maps[..., channel]
        minimum = float(np.min(feature))
        maximum = float(np.max(feature))
        if maximum - minimum < 1e-8:
            scaled = np.zeros_like(feature)
        else:
            scaled = (feature - minimum) / (maximum - minimum)
        normalized.append(scaled)
    return np.stack(normalized, axis=-1)


def compute_grad_cam(
    image: np.ndarray,
    backbone: BackboneName,
    class_index: int,
    *,
    model: tf.keras.Model | None = None,
) -> np.ndarray:
    classifier = model or load_classifier(backbone)
    batch = _preprocess(image, backbone)
    last_conv = last_conv_layer_name(backbone)

    grad_model = tf.keras.Model(
        [classifier.inputs],
        [classifier.get_layer(last_conv).output, classifier.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    if float(maximum.numpy()) > 0:
        heatmap = heatmap / maximum
    heatmap_np = heatmap.numpy()
    resized = tf.image.resize(heatmap_np[..., np.newaxis], INPUT_SIZE).numpy()[..., 0]
    return resized


