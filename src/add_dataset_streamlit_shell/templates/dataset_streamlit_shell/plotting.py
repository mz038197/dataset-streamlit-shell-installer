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

    fig_balance, ax_balance = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    class_counts = labels.astype(int).value_counts().sort_index()
    ax_balance.bar(
        [f"y={index}" for index in class_counts.index],
        class_counts.to_numpy(),
        color=["#f4b400", "#1a73e8"][: len(class_counts)],
    )
    ax_balance.set_xlabel(target)
    ax_balance.set_ylabel("樣本數")
    ax_balance.set_title(f"{target} 類別分佈")
    figures.append(("類別分佈", fig_balance))

    if len(features) == 1:
        feature = features[0]
        x = pd.to_numeric(frame[feature], errors="coerce")
        jitter = np.where(positives, 1.02, -0.02) + np.random.default_rng(0).uniform(
            -0.04, 0.04, size=len(frame)
        )
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        ax.scatter(x[negatives], jitter[negatives], marker="o", facecolors="none", label="y=0")
        ax.scatter(x[positives], jitter[positives], marker="x", label="y=1")
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
    ax.scatter(x1[negatives], x2[negatives], marker="o", facecolors="none", label="y=0")
    ax.scatter(x1[positives], x2[positives], marker="x", label="y=1")
    ax.set_xlabel(plot_features[0])
    ax.set_ylabel(plot_features[1])
    title = f"{plot_features[0]} vs {plot_features[1]}"
    if len(features) > 2:
        title += "（訓練使用全部 features）"
    ax.set_title(title)
    ax.legend()
    figures.append(("特徵空間分佈", fig))
    return figures


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
