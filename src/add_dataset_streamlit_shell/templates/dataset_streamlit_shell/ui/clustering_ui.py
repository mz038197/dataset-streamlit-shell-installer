"""非監督式分群頁：K-Means 分群過程演進與 Ward's Method（內建範例）。"""

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
    KMeansEvolution,
    advance_kmeans_evolution,
    can_advance_kmeans,
    cut_ward_clusters,
    feature_matrix,
    kmeans_evolution_status,
    load_builtin_clustering_frame,
    new_kmeans_evolution,
    sample_initial_centers,
    ward_linkage,
)
from dataset_streamlit_shell.plotting import configure_matplotlib_for_traditional_chinese
from dataset_streamlit_shell.ui.data_ui import render_chat_panel, render_dataset_metrics
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane

configure_matplotlib_for_traditional_chinese()

KMEANS_TITLE = "K-Means 分群"
WARDS_TITLE = "Ward's Method（階層分群）"
KMEANS_EVO_KEY = "kmeans_evolution"
_CLUSTER_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed", "#0891b2", "#ea580c", "#4b5563")
_UNASSIGNED_COLOR = "#9ca3af"


def render_kmeans_page() -> None:
    teaching, agent = open_content_dual_pane()
    with teaching:
        st.title(KMEANS_TITLE)
        st.caption("用內建範例走通指定 K 的分群；按「下一步」走分群過程演進。不依賴 ready.csv。")

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

        st.markdown("##### 分群過程演進")
        if st.button(
            "開始分群演示",
            type="primary",
            width="stretch",
            key="kmeans_start_demo",
        ):
            points = feature_matrix(frame)
            st.session_state[KMEANS_EVO_KEY] = new_kmeans_evolution(
                locked_k=int(n_clusters),
                initial_centers=sample_initial_centers(points, n_clusters=int(n_clusters)),
            )

        evo = st.session_state.get(KMEANS_EVO_KEY)
        if isinstance(evo, KMeansEvolution):
            if int(n_clusters) != evo.locked_k:
                st.caption("slider 的 K 與這一輪鎖定值不同；要套用新 K 請重新按「開始分群演示」。")
            st.metric("鎖定的 K", str(evo.locked_k))
            if st.button(
                "下一步",
                type="primary",
                width="stretch",
                key="kmeans_next_step",
                disabled=not can_advance_kmeans(evo),
            ):
                evo = advance_kmeans_evolution(evo, feature_matrix(frame))
                st.session_state[KMEANS_EVO_KEY] = evo
            st.caption(kmeans_evolution_status(evo))
            _plot_kmeans_evolution(frame, evo, show_truth=show_truth, n_clusters=int(n_clusters))
        else:
            st.caption("請按「開始分群演示」鎖定 K，再按「下一步」放出初始群中心。")
            _plot_kmeans_evolution(frame, None, show_truth=show_truth, n_clusters=int(n_clusters))

    with agent:
        evo = st.session_state.get(KMEANS_EVO_KEY)
        locked = evo.locked_k if isinstance(evo, KMeansEvolution) else "尚未開始"
        status = kmeans_evolution_status(evo) if isinstance(evo, KMeansEvolution) else "尚未開始分群演示"
        render_chat_panel(
            extra_context=(
                f"目前頁面：{KMEANS_TITLE}。資料來源：{SOURCE_LABEL}。"
                f"固定特徵 {CLUSTERING_FEATURES}；slider K="
                f"{st.session_state.get('kmeans_n_clusters', DEFAULT_N_CLUSTERS)}；鎖定的 K={locked}。"
                f"分群過程演進：{status}。"
                "演算法不使用教學用真實群標籤；該欄僅供對照。"
            ),
            page_name=KMEANS_TITLE,
        )


def render_wards_page() -> None:
    teaching, agent = open_content_dual_pane()
    with teaching:
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

    with agent:
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
    labels: np.ndarray | None,
    centers: np.ndarray | None,
    title: str,
    legend_prefix: str,
    ghost_centers: np.ndarray | None = None,
) -> None:
    if labels is None:
        ax.scatter(
            x,
            y,
            s=36,
            c=_UNASSIGNED_COLOR,
            edgecolors="#111827",
            linewidths=0.4,
            alpha=0.9,
            label="未分群",
        )
    else:
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
    if ghost_centers is not None:
        ax.scatter(
            ghost_centers[:, 0],
            ghost_centers[:, 1],
            s=160,
            c="#111827",
            marker="X",
            alpha=0.28,
            label="上一拍中心",
            zorder=4,
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


def _plot_kmeans_evolution(
    frame,
    evo: KMeansEvolution | None,
    *,
    show_truth: bool,
    n_clusters: int,
) -> None:
    labels = None if evo is None or evo.labels is None else np.asarray(evo.labels, dtype=int)
    centers = None if evo is None or evo.centers is None else np.asarray(evo.centers, dtype=float)
    ghost = None
    if evo is not None and evo.previous_centers is not None:
        ghost = np.asarray(evo.previous_centers, dtype=float)
    k = evo.locked_k if evo is not None else n_clusters
    _plot_cluster_scatter(
        frame,
        labels=labels,
        centers=centers,
        show_truth=show_truth,
        title=f"K-Means（K={k}）",
        ghost_centers=ghost,
    )


def _plot_cluster_scatter(
    frame,
    *,
    labels: np.ndarray | None,
    centers: np.ndarray | None,
    show_truth: bool,
    title: str,
    ghost_centers: np.ndarray | None = None,
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
            ghost_centers=ghost_centers,
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
        ghost_centers=ghost_centers,
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
