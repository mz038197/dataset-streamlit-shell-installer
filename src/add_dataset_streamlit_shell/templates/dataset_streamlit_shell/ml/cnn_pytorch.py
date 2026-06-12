from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class CnnTrainResult:
    model: nn.Module
    epoch_losses: tuple[float, ...]
    test_accuracy: float
    test_images: np.ndarray
    test_labels: np.ndarray


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_digits_tensors(*, test_size: float = 0.2, random_state: int = 42):
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
    )
    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    return digits.images, digits.target, train_ds, test_ds


def train_simple_cnn(
    *,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 64,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> CnnTrainResult:
    _, _, train_ds, test_ds = load_digits_tensors()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        last_loss = 0.0
        for x_batch, y_batch in train_loader:
            loss = loss_fn(model(x_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        epoch_losses.append(last_loss)
        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, last_loss)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model(x_batch).argmax(dim=1)
            correct += int((pred == y_batch).sum().item())
            total += int(y_batch.size(0))

    test_images = test_ds.tensors[0].numpy()[:, 0]
    test_labels = test_ds.tensors[1].numpy()
    return CnnTrainResult(
        model=model,
        epoch_losses=tuple(epoch_losses),
        test_accuracy=correct / total if total else 0.0,
        test_images=test_images,
        test_labels=test_labels,
    )


def extract_first_conv_maps(model: SimpleCNN, sample: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        feature_maps = model.conv(sample)[0].numpy()
    return feature_maps
