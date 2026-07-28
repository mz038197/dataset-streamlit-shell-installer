"""非監督式分群頁：K-Means 與 Ward's Method（內建範例、開箱探索）。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from dataset_streamlit_shell.ml.clustering import (
    CLUSTERING_FEATURES,
    CLUSTERING_TRUTH_LABEL,
    DEFAULT_N_CLUSTERS,
    N_CLUSTERS_MAX,
    N_CLUSTERS_MIN,
    SOURCE_LABEL,
    cut_ward_clusters,
    fit_kmeans,
    load_builtin_clustering_frame,
    ward_linkage,
)
from dataset_streamlit_shell.plotting import configure_matplotlib_for_traditional_chinese
from dataset_streamlit_shell.ui.data_ui import render_chat_panel, render_dataset_metrics

configure_matplotlib_for_traditional_chinese()

KMEANS_TITLE = "K-Means 分群"
WARDS_TITLE = "Ward's Method（階層分群）"
_CLUSTER_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed", "#0891b2", "#ea580c", "#4b5563")


def render_kmeans_page() -> None:
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title(KMEANS_TITLE)
        st.caption("用內建範例走通指定 K 的分群；不依賴 ready.csv。")

        frame = load_builtin_clustering_frame()
        render_dataset_metrics(frame)
        st.caption(SOURCE_LABEL)
        st.caption(f"固定特徵：{'、'.join(CLUSTERING_FEATURES)}（不提供欄位下拉）")

        n_clusters = st.slider(
            "群數 K",
            min_value=N_CLUSTERS_MIN,
            max_value=N_CLUSTERS_MAX,
            value=DEFAULT_N_CLUSTERS,
            step=1,
            key="kmeans_n_clusters",
        )
        show_truth = st.checkbox(
            "對照教學用真實群標籤",
            value=False,
            key="kmeans_show_truth",
        )

        result = fit_kmeans(frame, n_clusters=n_clusters)
        st.markdown("##### 分群結果")
        _plot_cluster_scatter(
            frame,
            labels=result.labels,
            centers=result.centers,
            show_truth=show_truth,
            title=f"K-Means（K={n_clusters}）",
        )

        st.markdown("##### 建議問 Agent")
        st.code(
            "請說明 K-Means 分群在做什麼，以及為什麼第一版範例資料尺度要設計得相近。",
            language="text",
        )

    with side:
        render_chat_panel(
            extra_context=(
                f"目前頁面：{KMEANS_TITLE}。資料來源：{SOURCE_LABEL}。"
                f"固定特徵 {CLUSTERING_FEATURES}；目前 K={st.session_state.get('kmeans_n_clusters', DEFAULT_N_CLUSTERS)}。"
                "演算法不使用教學用真實群標籤；該欄僅供對照。"
            ),
            page_name=KMEANS_TITLE,
        )


def render_wards_page() -> None:
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title(WARDS_TITLE)
        st.caption("用內建範例看 Ward 合併過程與切群；不依賴 ready.csv。")

        frame = load_builtin_clustering_frame()
        render_dataset_metrics(frame)
        st.caption(SOURCE_LABEL)
        st.caption(f"固定特徵：{'、'.join(CLUSTERING_FEATURES)}（不提供欄位下拉）")

        n_clusters = st.slider(
            "切成幾群",
            min_value=N_CLUSTERS_MIN,
            max_value=N_CLUSTERS_MAX,
            value=DEFAULT_N_CLUSTERS,
            step=1,
            key="wards_n_clusters",
        )
        show_truth = st.checkbox(
            "對照教學用真實群標籤",
            value=False,
            key="wards_show_truth",
        )

        linkage_matrix = ward_linkage(frame)
        labels = cut_ward_clusters(linkage_matrix, n_clusters=n_clusters)

        st.markdown("##### 樹狀圖（dendrogram）")
        _plot_dendrogram(linkage_matrix)

        st.markdown("##### 切群散點")
        _plot_cluster_scatter(
            frame,
            labels=labels,
            centers=None,
            show_truth=show_truth,
            title=f"Ward 切群（k={n_clusters}）",
        )

        st.markdown("##### 建議問 Agent")
        st.code(
            "請說明 Ward's Method 和 K-Means 的差異，以及樹狀圖如何幫助決定切幾群。",
            language="text",
        )

    with side:
        render_chat_panel(
            extra_context=(
                f"目前頁面：{WARDS_TITLE}。資料來源：{SOURCE_LABEL}。"
                f"固定特徵 {CLUSTERING_FEATURES}；目前切群數="
                f"{st.session_state.get('wards_n_clusters', DEFAULT_N_CLUSTERS)}。"
                "演算法不使用教學用真實群標籤；該欄僅供對照。"
            ),
            page_name=WARDS_TITLE,
        )


def _cluster_color(label: int) -> str:
    return _CLUSTER_COLORS[int(label) % len(_CLUSTER_COLORS)]


def _draw_labeled_scatter(
    ax,
    *,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray | None,
    title: str,
    legend_prefix: str,
) -> None:
    for lab in sorted(np.unique(labels).tolist()):
        mask = labels == lab
        ax.scatter(
            x[mask],
            y[mask],
            s=36,
            c=_cluster_color(lab),
            edgecolors="#111827",
            linewidths=0.4,
            alpha=0.9,
            label=f"{legend_prefix} {lab}",
        )
    if centers is not None:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=160,
            c="#111827",
            marker="X",
            label="群中心",
            zorder=5,
        )
    ax.set_xlabel(CLUSTERING_FEATURES[0])
    ax.set_ylabel(CLUSTERING_FEATURES[1])
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=True)


def _plot_cluster_scatter(
    frame,
    *,
    labels: np.ndarray,
    centers: np.ndarray | None,
    show_truth: bool,
    title: str,
) -> None:
    x = frame[CLUSTERING_FEATURES[0]].to_numpy(dtype=float)
    y = frame[CLUSTERING_FEATURES[1]].to_numpy(dtype=float)

    # 對照時並排：演算法標籤與真實群編號不可直接當同一色（排列可置換）
    if show_truth:
        truth = frame[CLUSTERING_TRUTH_LABEL].to_numpy(dtype=int)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
        _draw_labeled_scatter(
            axes[0],
            x=x,
            y=y,
            labels=labels,
            centers=centers,
            title=title,
            legend_prefix="演算法群",
        )
        _draw_labeled_scatter(
            axes[1],
            x=x,
            y=y,
            labels=truth,
            centers=None,
            title="教學用真實群標籤",
            legend_prefix="真實群",
        )
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.caption("左右圖的群編號各自獨立；相同數字不代表同一群（分群標籤可置換）。")
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _draw_labeled_scatter(
        ax,
        x=x,
        y=y,
        labels=labels,
        centers=centers,
        title=title,
        legend_prefix="演算法群",
    )
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _plot_dendrogram(linkage_matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    dendrogram(linkage_matrix, ax=ax, no_labels=True, color_threshold=None)
    ax.set_xlabel("樣本（葉節點）")
    ax.set_ylabel("合併距離")
    ax.set_title("Ward 階層合併")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
