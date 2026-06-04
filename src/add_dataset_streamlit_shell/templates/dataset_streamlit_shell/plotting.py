from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


TRADITIONAL_CHINESE_FONT_CANDIDATES = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "Noto Sans TC",
    "Noto Sans CJK TC",
    "MingLiU",
    "PingFang TC",
    "Heiti TC",
    "SimHei",
    "Arial Unicode MS",
]


def configure_matplotlib_for_traditional_chinese() -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    font_manager = importlib.import_module("matplotlib.font_manager")

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    installed_candidates = [
        font_name
        for font_name in TRADITIONAL_CHINESE_FONT_CANDIDATES
        if font_name in available_fonts
    ]
    if installed_candidates:
        plt.rcParams["font.family"] = installed_candidates[0]
        plt.rcParams["font.sans-serif"] = installed_candidates + ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = TRADITIONAL_CHINESE_FONT_CANDIDATES + [
            "DejaVu Sans"
        ]
    plt.rcParams["axes.unicode_minus"] = False


CLASS_NEGATIVE_STYLE = {
    "marker": "o",
    "c": "#f4b400",
    "edgecolors": "#5f4330",
    "linewidths": 0.6,
    "label": "y=-1",
}
CLASS_POSITIVE_STYLE = {
    "marker": "x",
    "c": "#202124",
    "linewidths": 1.2,
    "label": "y=+1",
}


def scatter_binary_classes(
    ax,
    x1,
    x2,
    *,
    positives: np.ndarray,
    negatives: np.ndarray,
) -> None:
    ax.scatter(x1[negatives], x2[negatives], **CLASS_NEGATIVE_STYLE)
    ax.scatter(x1[positives], x2[positives], **CLASS_POSITIVE_STYLE)


def build_regression_data_figures(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> list[tuple[str, Figure]]:
    plt = importlib.import_module("matplotlib.pyplot")
    configure_matplotlib_for_traditional_chinese()
    figures: list[tuple[str, Figure]] = []
    y = pd.to_numeric(frame[target], errors="coerce")

    if len(features) == 1:
        feature = features[0]
        x = pd.to_numeric(frame[feature], errors="coerce")
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.scatter(x, y, alpha=0.75)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title(f"{feature} vs {target}")
        figures.append((f"{feature} 與 {target}", fig))
        return figures

    count = min(len(features), 4)
    cols = 2 if count > 1 else 1
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4.2 * rows), constrained_layout=True)
    flat_axes = np.atleast_1d(axes).ravel()
    for index, feature in enumerate(features[:count]):
        x = pd.to_numeric(frame[feature], errors="coerce")
        ax = flat_axes[index]
        ax.scatter(x, y, alpha=0.75)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title(f"{feature} vs {target}")
    for ax in flat_axes[count:]:
        ax.axis("off")
    figures.append((f"各 feature 與 {target}", fig))
    return figures


def build_classification_data_figures(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> list[tuple[str, Figure]]:
    plt = importlib.import_module("matplotlib.pyplot")
    configure_matplotlib_for_traditional_chinese()
    figures: list[tuple[str, Figure]] = []
    labels = pd.to_numeric(frame[target], errors="coerce")
    positives = labels == 1
    negatives = labels == 0

    if len(features) == 1:
        feature = features[0]
        x = pd.to_numeric(frame[feature], errors="coerce")
        jitter = np.where(positives, 1.02, -0.02) + np.random.default_rng(0).uniform(
            -0.04, 0.04, size=len(frame)
        )
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.scatter(x[negatives], jitter[negatives], **CLASS_NEGATIVE_STYLE)
        ax.scatter(x[positives], jitter[positives], **CLASS_POSITIVE_STYLE)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_yticks([0, 1])
        ax.set_title(f"{feature} 與 {target}")
        ax.legend()
        figures.append((f"{feature} 分佈", fig))
        return figures

    plot_features = features[:2]
    x1 = pd.to_numeric(frame[plot_features[0]], errors="coerce")
    x2 = pd.to_numeric(frame[plot_features[1]], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    scatter_binary_classes(ax, x1, x2, positives=positives, negatives=negatives)
    ax.set_xlabel(plot_features[0])
    ax.set_ylabel(plot_features[1])
    title = f"{plot_features[0]} vs {plot_features[1]}"
    if len(features) > 2:
        title += "（訓練使用全部 features）"
    ax.set_title(title)
    ax.legend()
    figures.append(("特徵空間分佈", fig))
    return figures


def build_svm_paired_data_figure(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> Figure:
    plt = importlib.import_module("matplotlib.pyplot")
    configure_matplotlib_for_traditional_chinese()
    x1 = pd.to_numeric(frame[features[0]], errors="coerce")
    x2 = pd.to_numeric(frame[features[1]], errors="coerce")
    labels = pd.to_numeric(frame[target], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.scatter(x1, x2, c=labels, cmap=plt.cm.Paired)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(f"{features[0]} vs {features[1]}（Paired 色圖）")
    return fig


def build_linear_svm_result_figure(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    coef: np.ndarray | list[float],
    intercept: float,
    support_vectors: np.ndarray | list[list[float]],
    mesh_points: int = 30,
    paired_scatter: bool = False,
) -> Figure:
    plt = importlib.import_module("matplotlib.pyplot")
    configure_matplotlib_for_traditional_chinese()
    x1_name, x2_name = features[0], features[1]
    x1 = pd.to_numeric(frame[x1_name], errors="coerce").to_numpy(dtype=float)
    x2 = pd.to_numeric(frame[x2_name], errors="coerce").to_numpy(dtype=float)
    labels = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    coef_array = np.asarray(coef, dtype=float).reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    if paired_scatter:
        ax.scatter(x1, x2, c=labels, cmap=plt.cm.Paired, zorder=2)
    else:
        positives = labels == 1
        negatives = labels == -1
        scatter_binary_classes(ax, x1, x2, positives=positives, negatives=negatives)

    x_min, x_max = float(np.min(x1)), float(np.max(x1))
    y_min, y_max = float(np.min(x2)), float(np.max(x2))
    pad_x = (x_max - x_min) * 0.08 or 1.0
    pad_y = (y_max - y_min) * 0.08 or 1.0
    grid_x = np.linspace(x_min - pad_x, x_max + pad_x, mesh_points)
    grid_y = np.linspace(y_min - pad_y, y_max + pad_y, mesh_points)
    mesh_xx, mesh_yy = np.meshgrid(grid_x, grid_y)
    grid = np.c_[mesh_xx.ravel(), mesh_yy.ravel()]
    scores = grid @ coef_array + float(intercept)
    mesh_zz = scores.reshape(mesh_xx.shape)
    ax.contourf(mesh_xx, mesh_yy, mesh_zz, levels=20, alpha=0.35, cmap="coolwarm")
    ax.contour(mesh_xx, mesh_yy, mesh_zz, levels=[0.0], colors="black", linewidths=1.5)

    sv = np.asarray(support_vectors, dtype=float)
    if sv.size:
        ax.scatter(
            sv[:, 0],
            sv[:, 1],
            s=140,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            label="support vectors",
            zorder=4,
        )

    ax.set_xlabel(x1_name)
    ax.set_ylabel(x2_name)
    ax.set_title("線性 SVM 決策邊界與 support vectors")
    ax.legend(loc="best")
    return fig


def build_sigmoid_figure(
    *,
    z_min: float = -10.0,
    z_max: float = 10.0,
    highlight_z: float | None = None,
) -> Figure:
    plt = importlib.import_module("matplotlib.pyplot")
    configure_matplotlib_for_traditional_chinese()
    z = np.linspace(z_min, z_max, 400)
    clipped = np.clip(z, -500, 500)
    y = 1.0 / (1.0 + np.exp(-clipped))
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(z, y, color="#1a73e8", linewidth=2, label=r"$\sigma(z)=1/(1+e^{-z})$")
    ax.axhline(0.5, color="#9aa0a6", linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(0, color="#9aa0a6", linestyle="--", linewidth=1, alpha=0.8)
    ax.scatter([0], [0.5], color="#ea4335", zorder=5, label="z=0 → σ(z)=0.5")
    if highlight_z is not None:
        hz = float(np.clip(highlight_z, z_min, z_max))
        hy = float(1.0 / (1.0 + np.exp(-np.clip(hz, -500, 500))))
        ax.scatter([hz], [hy], color="#34a853", s=80, zorder=6, label=f"z={hz:g}")
        ax.vlines(hz, 0, hy, colors="#34a853", linestyles=":", linewidth=1)
        ax.hlines(hy, z_min, hz, colors="#34a853", linestyles=":", linewidth=1)
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$z=\mathbf{w}\cdot\mathbf{x}+b$")
    ax.set_ylabel(r"$\sigma(z)$（預測機率）")
    ax.set_title("Sigmoid 函數")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    return fig


def build_hyperparam_sweep_figure(
    *,
    param_label: str,
    values: list[float | int],
    train_accuracy: list[float],
    val_accuracy: list[float],
) -> Figure:
    plt = importlib.import_module("matplotlib.pyplot")
    configure_matplotlib_for_traditional_chinese()
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    x_positions = list(range(len(values)))
    ax.plot(x_positions, train_accuracy, marker="o", label="訓練集")
    ax.plot(x_positions, val_accuracy, marker="o", label="驗證集")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(value) for value in values])
    ax.set_xlabel(param_label)
    ax.set_ylabel("準確率（%）")
    ax.set_title(f"準確率 vs {param_label}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return fig


def build_decision_tree_figure(
    model,
    *,
    feature_names: list[str],
    class_names: list[str],
) -> Figure:
    plt = importlib.import_module("matplotlib.pyplot")
    tree = importlib.import_module("sklearn.tree")
    configure_matplotlib_for_traditional_chinese()
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax,
    )
    ax.set_title("決策樹結構")
    return fig


def render_figures_in_streamlit(figures: list[tuple[str, Figure]]) -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    if not figures:
        return
    st.markdown("##### 資料視覺化")
    for caption, fig in figures:
        st.caption(caption)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
