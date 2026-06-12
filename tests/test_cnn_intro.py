from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.cnn_intro import (
    EDGE_KERNEL,
    PATCH_BAD,
    PATCH_GOOD,
    PATCH_MEDIUM,
    POOLING_INPUT,
    conv2d_valid,
    max_pool2x2,
    patch_response,
)
from dataset_streamlit_shell.ml.cnn_pytorch import (
    SimpleCNN,
    load_digits_tensors,
)


def test_conv2d_valid_shape() -> None:
    image = np.arange(36, dtype=float).reshape(6, 6)
    kernel = np.ones((3, 3))
    output = conv2d_valid(image, kernel)
    assert output.shape == (4, 4)


def test_patch_responses_match_notebook() -> None:
    assert patch_response(PATCH_GOOD, EDGE_KERNEL) == 3.0
    assert patch_response(PATCH_MEDIUM, EDGE_KERNEL) == 1.5
    assert patch_response(PATCH_BAD, EDGE_KERNEL) == -3.0


def test_max_pool2x2_matches_notebook() -> None:
    expected = np.array([[6.0, 5.0], [2.0, 4.0]])
    np.testing.assert_array_equal(max_pool2x2(POOLING_INPUT), expected)


def test_simple_cnn_forward_shape() -> None:
    model = SimpleCNN()
    sample = torch.zeros(1, 1, 8, 8)
    logits = model(sample)
    assert logits.shape == (1, 10)


def test_load_digits_tensors_shapes() -> None:
    images, labels, train_ds, test_ds = load_digits_tensors()
    assert images.shape == (1797, 8, 8)
    assert labels.shape == (1797,)
    assert train_ds.tensors[0].shape[1:] == (1, 8, 8)
    assert test_ds.tensors[0].shape[1:] == (1, 8, 8)
