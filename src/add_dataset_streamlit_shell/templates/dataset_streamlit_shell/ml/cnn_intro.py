from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

BASE_RGB_IMAGE = np.zeros((8, 8, 3))
BASE_RGB_IMAGE[2:6, 2:6] = [1, 0.5, 0]

DEMO_IMAGE_6X6 = np.array(
    [
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9],
    ],
    dtype=float,
)

SLIDE_KERNEL = np.array(
    [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ],
    dtype=float,
)

EDGE_KERNEL = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ],
    dtype=float,
)

PATCH_GOOD = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
    ],
    dtype=float,
)

PATCH_MEDIUM = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ],
    dtype=float,
)

PATCH_BAD = np.array(
    [
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
    ],
    dtype=float,
)

CONV_OUTPUT_DEMO = np.array(
    [
        [-2, -1, 0, 2],
        [-3, 1, 3, -1],
        [-2, 0, 4, 1],
    ],
    dtype=float,
)

POOLING_INPUT = np.array(
    [
        [1, 3, 2, 1],
        [4, 6, 5, 2],
        [0, 1, 3, 4],
        [2, 2, 1, 0],
    ],
    dtype=float,
)


def patch_response(patch: np.ndarray, kernel: np.ndarray) -> float:
    return float(np.sum(patch * kernel))


def conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    out_h = image.shape[0] - kh + 1
    out_w = image.shape[1] - kw + 1
    output = np.zeros((out_h, out_w), dtype=float)
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i : i + kh, j : j + kw] * kernel)
    return output


def max_pool2x2(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [x[0:2, 0:2].max(), x[0:2, 2:4].max()],
            [x[2:4, 0:2].max(), x[2:4, 2:4].max()],
        ],
        dtype=float,
    )


def draw_numbers_on_imshow(ax, matrix: np.ndarray, *, fmt: str = "{:.0f}") -> None:
    im = ax.images[-1]
    cmap = im.get_cmap()
    norm = im.norm
    for (row, col), value in np.ndenumerate(matrix):
        if np.isnan(value):
            continue
        rgba = cmap(norm(value))
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "black" if luminance > 0.6 else "white"
        ax.text(
            col,
            row,
            fmt.format(value),
            ha="center",
            va="center",
            color=text_color,
            fontsize=12,
            fontweight="bold",
        )


def _import_plt():
    from dataset_streamlit_shell.plotting import configure_matplotlib_for_traditional_chinese

    configure_matplotlib_for_traditional_chinese()
    import matplotlib.pyplot as plt

    return plt


def build_rgb_image_figure() -> Figure:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(BASE_RGB_IMAGE)
    ax.set_title("標準範例影像")
    ax.axis("off")
    fig.tight_layout()
    return fig


def build_conv_step_figure(
    image: np.ndarray,
    kernel: np.ndarray,
    step: int,
) -> Figure:
    plt = _import_plt()
    kh, kw = kernel.shape
    out_h, out_w = image.shape[0] - kh + 1, image.shape[1] - kw + 1
    step = max(0, min(int(step), out_h * out_w - 1))
    row = step // out_w
    col = step % out_w
    patch = image[row : row + kh, col : col + kw]
    response = float(np.sum(patch * kernel))

    partial = np.full((out_h, out_w), np.nan)
    for t in range(step + 1):
        ti = t // out_w
        tj = t % out_w
        partial[ti, tj] = np.sum(image[ti : ti + kh, tj : tj + kw] * kernel)

    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    ax1.imshow(image, cmap="viridis")
    ax1.set_title(f"影像（patch 位於 i={row}, j={col}）")
    ax1.set_xticks([])
    ax1.set_yticks([])
    for (r, c), val in np.ndenumerate(image):
        ax1.text(c, r, f"{val:.0f}", ha="center", va="center", color="black", fontsize=11)
    ax1.add_patch(
        plt.Rectangle((col - 0.5, row - 0.5), kw, kh, fill=False, edgecolor="red", linewidth=3)
    )

    ax2.imshow(patch, cmap="viridis")
    ax2.set_title("Patch")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for (r, c), val in np.ndenumerate(patch):
        ax2.text(c, r, f"{val:.0f}", ha="center", va="center", color="black", fontsize=12)

    ax3.imshow(kernel, cmap="viridis")
    ax3.set_title(f"Kernel（加總 = {response:.0f}）")
    ax3.set_xticks([])
    ax3.set_yticks([])
    for (r, c), val in np.ndenumerate(kernel):
        ax3.text(c, r, f"{val:.0f}", ha="center", va="center", color="black", fontsize=12)

    ax4.imshow(partial, cmap="viridis")
    ax4.set_title("特徵圖（目前已計算）")
    ax4.set_xticks([])
    ax4.set_yticks([])
    draw_numbers_on_imshow(ax4, partial, fmt="{:.0f}")
    ax4.add_patch(
        plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3)
    )

    fig.tight_layout()
    return fig


def build_patch_similarity_figure() -> Figure:
    plt = _import_plt()
    patches = [
        ("非常相似", PATCH_GOOD),
        ("有點相似", PATCH_MEDIUM),
        ("不相似", PATCH_BAD),
    ]
    fig = plt.figure(figsize=(9, 3))
    for index, (title, patch) in enumerate(patches, start=1):
        value = patch_response(patch, EDGE_KERNEL)
        ax = fig.add_subplot(1, 3, index)
        ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\n點積 = {value:.1f}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def build_feature_maps_demo_figure(*, seed: int = 42) -> Figure:
    plt = _import_plt()
    rng = np.random.default_rng(seed)
    feature_maps = rng.random((6, 6, 4))
    fig = plt.figure(figsize=(8, 4))
    for index in range(4):
        ax = fig.add_subplot(1, 4, index + 1)
        ax.imshow(feature_maps[:, :, index], cmap="gray")
        ax.set_title(f"特徵圖 {index}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def build_relu_curve_figure() -> Figure:
    plt = _import_plt()
    x = np.linspace(-5, 5, 100)
    y = np.maximum(0, x)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, linewidth=3)
    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title("ReLU 函式：max(0, x)")
    ax.set_xlabel("輸入")
    ax.set_ylabel("輸出")
    ax.grid(True)
    fig.tight_layout()
    return fig


def build_relu_image_figure(conv_output: np.ndarray | None = None) -> Figure:
    plt = _import_plt()
    conv = conv_output if conv_output is not None else CONV_OUTPUT_DEMO
    relu = np.maximum(0, conv)
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(conv, cmap="gray")
    ax1.set_title("ReLU 前（卷積輸出）")
    ax1.axis("off")
    ax2.imshow(relu, cmap="gray")
    ax2.set_title("ReLU 後")
    ax2.axis("off")
    fig.tight_layout()
    return fig


def build_pooling_demo_figure(x: np.ndarray | None = None) -> Figure:
    plt = _import_plt()
    source = x if x is not None else POOLING_INPUT
    pooled = max_pool2x2(source)
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(source, cmap="plasma")
    ax1.set_title("Pooling 前")
    ax1.axis("off")
    ax2.imshow(pooled, cmap="plasma")
    ax2.set_title("2×2 Max Pooling 後")
    ax2.axis("off")
    fig.tight_layout()
    return fig


def build_digits_preview_figure(images: np.ndarray, labels: np.ndarray, count: int = 10) -> Figure:
    plt = _import_plt()
    fig = plt.figure(figsize=(8, 3))
    for index in range(min(count, len(images))):
        ax = fig.add_subplot(1, count, index + 1)
        ax.imshow(images[index], cmap="gray")
        ax.set_title(str(labels[index]))
        ax.axis("off")
    fig.tight_layout()
    return fig


def build_first_conv_maps_figure(feature_maps: np.ndarray) -> Figure:
    plt = _import_plt()
    count = feature_maps.shape[0]
    cols = min(4, count)
    rows = int(np.ceil(count / cols))
    fig = plt.figure(figsize=(10, 2.5 * rows))
    for index in range(count):
        ax = fig.add_subplot(rows, cols, index + 1)
        ax.imshow(feature_maps[index], cmap="gray")
        ax.set_title(f"濾鏡 {index}")
        ax.axis("off")
    fig.suptitle("第一層 CNN 眼中的世界")
    fig.tight_layout()
    return fig
