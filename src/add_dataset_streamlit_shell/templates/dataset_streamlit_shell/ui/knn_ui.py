"""K-近鄰分類頁：單頁訓練前預測＋下一步預測過程演進（Plotly）。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataset_streamlit_shell.ml.knn import (
    BEAT_LINES,
    BEAT_RANK,
    BEAT_VOTE,
    DEFAULT_K,
    KnnArtifact,
    KnnEvolution,
    accepted_chart_click,
    advance_evolution,
    build_knn_agent_context,
    build_knn_artifact,
    can_advance,
    can_click,
    click_query,
    decision_mesh_predictions,
    demo_query_points,
    distance_marker_styles,
    evolution_status_caption,
    fit_knn_classifier,
    neighbor_indices_for_view,
    new_evolution,
    odd_k_values,
    prepare_feature_matrix,
    rank_table_rows,
    shows_boundary,
    vote_tally,
)
from dataset_streamlit_shell.ui import knn_quiz as quiz
from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    invoke_data_agent,
    render_chat_panel,
    render_dataset_metrics,
)
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
KNN_BLOBS_PATH = CLASSIFICATION_DEMO_DIR / "knn_blobs_80.csv"

KNN_FEATURES = ["特徵1", "特徵2"]
KNN_TARGET = "類別"
PAGE_TITLE = "K-近鄰分類"
CONTEXT_KEY = f"{PAGE_TITLE}_agent_context"
RESULT_KEY = "knn_neighbors_last_artifact"
EVO_KEY = "knn_neighbors_evolution"

_CLASS_COLORS = {0: "#2563eb", 1: "#dc2626"}


def render_knn_page() -> None:
    teaching, agent = open_content_dual_pane()
    with teaching:
        st.title(PAGE_TITLE)
        st.caption("先搞懂鄰居與多數決，解鎖後再調 k，用「下一步」看一個新點怎麼被分類。")
        _render_page()
        _compose_agent_context()

    with agent:
        render_chat_panel(
            extra_context=str(st.session_state.get(CONTEXT_KEY, f"目前頁面：{PAGE_TITLE}。")),
            page_name=PAGE_TITLE,
        )


def _render_page() -> None:
    st.markdown("##### 這一頁在問什麼")
    st.info(
        "K-近鄰是**實例型**學習：不先學一組權重 w，預測時找出距離最近的 k 個訓練點，再**多數決**。"
    )
    prepared = _prepare_data()
    if prepared is None:
        return
    working, features, target, source_label = prepared

    st.markdown("##### 預測怎麼做")
    st.latex(r"\hat{y}=\mathrm{majority}\{y_{(1)},\ldots,y_{(k)}\}")
    st.caption(quiz.NEIGHBORS_FORMULA_CAPTION)

    unlocked = _render_pretrain_quiz(
        features=features, target=target, source_label=source_label, row_count=len(working)
    )

    k_choices = odd_k_values()
    k = DEFAULT_K
    if unlocked:
        k = int(
            st.select_slider(
                "鄰居數 k",
                options=k_choices,
                value=DEFAULT_K if DEFAULT_K in k_choices else k_choices[0],
                key="knn_k_slider",
                help="奇數步進。按「開始預測演示」當下的 k 會鎖定這一輪。",
            )
        )

    st.markdown("##### 預測演示")
    demo_clicked = st.button(
        "開始預測演示",
        type="primary",
        width="stretch",
        key="train_knn_neighbors",
        disabled=not unlocked,
    )
    if not unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始預測演示。卡住時可按各題「Agent 提示」。")

    signature = (source_label, tuple(features), target, True, len(working), "single")
    artifact = _resolve_artifact(
        demo_clicked=demo_clicked and unlocked,
        working=working,
        features=features,
        target=target,
        source_label=source_label,
        k=k,
        signature=signature,
    )

    if artifact is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("鎖定的 k", str(artifact.k))
        c2.metric("標準化", "開")
        c3.metric("訓練集正確率", f"{artifact.training_accuracy:.2f}%")
        if unlocked and k != artifact.k:
            st.caption("slider 的 k 與這一輪鎖定值不同；要套用新 k 請重新按「開始預測演示」。")

        st.markdown("##### 預測過程演進")
        st.caption(quiz.result_chart_caption())
        evo = st.session_state.get(EVO_KEY)
        if isinstance(evo, KnnEvolution):
            next_clicked = st.button(
                "下一步",
                type="primary",
                width="stretch",
                key="knn_next_step",
                disabled=not can_advance(evo),
            )
            if next_clicked and can_advance(evo):
                evo = advance_evolution(evo, artifact)
                st.session_state[EVO_KEY] = evo
            _render_evolution_chart(working, artifact, evo)
        else:
            st.caption("請按「開始預測演示」準備鄰居池。")

    focus = st.session_state.get(quiz.SESSION_NEIGHBORS_FOCUS)
    inst_choice = str(st.session_state.get(quiz.SESSION_INST, quiz.PLEASE_SELECT))
    vote_choice = str(st.session_state.get(quiz.SESSION_VOTE, quiz.PLEASE_SELECT))
    appendix = quiz.build_neighbors_quiz_agent_appendix(
        inst_status=quiz.quiz_choice_status(inst_choice, correct=quiz.is_inst_correct(inst_choice)),
        vote_status=quiz.quiz_choice_status(vote_choice, correct=quiz.is_vote_correct(vote_choice)),
        focus_qid=focus,
        features=features,
        target=target,
        unlocked=unlocked,
    )
    st.session_state["_knn_ctx_frag"] = build_knn_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        k=int(artifact.k) if artifact is not None else k,
        standardize=True,
        row_count=len(working),
        artifact=artifact,
        expose_k=True,
        prompt_train=unlocked,
        note=appendix,
    )

    with st.expander("完整 sklearn 範例程式", expanded=False):
        st.code(quiz.SKLEARN_NEIGHBORS_EXAMPLE, language="python")


def _prepare_data() -> tuple[pd.DataFrame, list[str], str, str] | None:
    df = pd.read_csv(KNN_BLOBS_PATH)
    render_dataset_metrics(df)
    features = list(KNN_FEATURES)
    target = KNN_TARGET
    working = df[features + [target]].dropna().copy()
    working[target] = pd.to_numeric(working[target], errors="coerce").astype(int)
    if len(working) < 2:
        st.warning("可用樣本少於 2 筆，無法進行預測演示。")
        return None
    st.caption("內建範例資料：兩特徵二元分類（尺度相近，80 筆）")
    return working, features, target, "內建範例資料：兩特徵二元分類（尺度相近，80 筆）"


def _resolve_artifact(
    *,
    demo_clicked: bool,
    working: pd.DataFrame,
    features: list[str],
    target: str,
    source_label: str,
    k: int,
    signature: tuple,
) -> KnnArtifact | None:
    if demo_clicked:
        try:
            feature_matrix, scaler = prepare_feature_matrix(
                working, features, standardize=True
            )
            clf = fit_knn_classifier(feature_matrix, working[target], k=k)
        except ValueError as exc:
            st.error(str(exc))
            return None
        artifact = build_knn_artifact(
            clf,
            features=list(features),
            target=target,
            k=k,
            standardize=True,
            scaler=scaler,
            data_source=source_label,
            feature_frame=feature_matrix,
            target_series=working[target],
        )
        st.session_state[RESULT_KEY] = {"signature": signature, "artifact": artifact}
        f0, f1 = features[0], features[1]
        demos = demo_query_points(
            working[f0].to_numpy(dtype=float),
            working[f1].to_numpy(dtype=float),
        )
        st.session_state[EVO_KEY] = new_evolution(demos, locked_k=artifact.k)
        return artifact

    stored = st.session_state.get(RESULT_KEY)
    if isinstance(stored, dict) and stored.get("signature") == signature:
        return stored["artifact"]
    st.session_state.pop(EVO_KEY, None)
    return None


def _render_evolution_chart(
    working: pd.DataFrame, artifact: KnnArtifact, evo: KnnEvolution
) -> None:
    status_ph = st.empty()
    chart_ph = st.empty()
    table_ph = st.empty()

    sizes = opacities = None
    if evo.beat is not None and evo.beat >= BEAT_RANK and evo.distances_by_index is not None:
        sizes, opacities = distance_marker_styles(np.asarray(evo.distances_by_index, dtype=float))

    active_pred = None
    if evo.beat == BEAT_VOTE and evo.labeled:
        active_pred = int(evo.labeled[-1][2])

    fig = _build_knn_figure(
        working,
        artifact,
        labeled=evo.labeled,
        active_xy=evo.active_xy,
        active_pred=active_pred,
        neighbor_indices=neighbor_indices_for_view(evo),
        show_boundary=shows_boundary(evo),
        show_click_layer=can_click(evo),
        marker_sizes=sizes,
        marker_opacities=opacities,
        distances_by_index=evo.distances_by_index,
        chart_key="knn_neighbors_plotly",
    )
    # 可點與逐步用不同 widget key，避免儀式中的 selection 在第 3 點 ④ 後被當成新查詢。
    if can_click(evo):
        event = chart_ph.plotly_chart(
            fig,
            width="stretch",
            key="knn_neighbors_plotly",
            on_select="rerun",
            selection_mode="points",
        )
    else:
        event = None
        chart_ph.plotly_chart(
            fig,
            width="stretch",
            key="knn_neighbors_plotly_step",
        )
    status_ph.caption(evolution_status_caption(evo) + _labeled_suffix(evo, artifact))

    if evo.beat is not None and evo.beat >= BEAT_RANK:
        rows = rank_table_rows(evo, artifact.train_y)
        if rows:
            table_ph.dataframe(
                _rank_frame(rows, k=artifact.k, beat=int(evo.beat)),
                hide_index=True,
                width="stretch",
            )

    if event is None:
        return
    points = getattr(getattr(event, "selection", None), "points", None) or []
    if not points:
        return
    pt = points[0]
    if "x" not in pt or "y" not in pt:
        return
    accepted = accepted_chart_click(
        click_enabled=True,
        selected_xy=(float(pt["x"]), float(pt["y"])),
        active_xy=evo.active_xy,
    )
    if accepted is None:
        return
    st.session_state[EVO_KEY] = click_query(evo, accepted)
    st.rerun()


def _labeled_suffix(evo: KnnEvolution, artifact: KnnArtifact) -> str:
    if evo.beat != BEAT_VOTE or not evo.labeled:
        tally_bit = ""
        idx = neighbor_indices_for_view(evo)
        if idx:
            labels = [int(artifact.train_y[i]) for i in idx]
            tally_bit = " · " + quiz.vote_progress_caption(vote_tally(labels))
        return tally_bit
    qx, qy, pred = evo.labeled[-1]
    return " · " + quiz.query_prediction_caption(qx, qy, pred, k=artifact.k)


def _rank_frame(rows, *, k: int, beat: int) -> pd.DataFrame:
    taken = beat >= BEAT_LINES
    return pd.DataFrame(
        {
            "名次": [row.rank for row in rows],
            "距離": [round(row.distance, 4) for row in rows],
            "類別": [row.label for row in rows],
            "進入 k 鄰居": [("是" if row.rank <= k else "否") if taken else "尚未取 k" for row in rows],
        }
    )


def _build_knn_figure(
    working: pd.DataFrame,
    artifact: KnnArtifact,
    *,
    labeled: tuple[tuple[float, float, int], ...],
    active_xy: tuple[float, float] | None,
    active_pred: int | None,
    neighbor_indices: list[int] | None,
    show_boundary: bool,
    show_click_layer: bool,
    marker_sizes: np.ndarray | None,
    marker_opacities: np.ndarray | None,
    distances_by_index: tuple[float, ...] | None,
    chart_key: str = "knn_chart",
) -> go.Figure:
    f0, f1 = artifact.features
    x = working[f0].to_numpy(dtype=float)
    y = working[f1].to_numpy(dtype=float)
    labels = working[artifact.target].to_numpy(dtype=int)
    raw = working[artifact.features].to_numpy(dtype=float)

    pad_x = (x.max() - x.min()) * 0.08 + 1e-6
    pad_y = (y.max() - y.min()) * 0.08 + 1e-6
    x_min, x_max = float(x.min() - pad_x), float(x.max() + pad_x)
    y_min, y_max = float(y.min() - pad_y), float(y.max() + pad_y)

    fig = go.Figure()
    if show_boundary:
        xx, yy, zz = decision_mesh_predictions(
            artifact, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, grid_size=70
        )
        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=zz,
                colorscale=[[0, "#dbeafe"], [1, "#fee2e2"]],
                opacity=0.55,
                showscale=False,
                contours_coloring="fill",
                line_width=0,
                hoverinfo="skip",
                name="邊界",
            )
        )

    sizes = np.full(len(x), 9.0) if marker_sizes is None else np.asarray(marker_sizes, dtype=float)
    opacities = (
        np.full(len(x), 0.9) if marker_opacities is None else np.asarray(marker_opacities, dtype=float)
    )
    for cls, color, name in ((0, _CLASS_COLORS[0], "類別 0"), (1, _CLASS_COLORS[1], "類別 1")):
        mask = labels == cls
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                marker=dict(
                    size=sizes[mask],
                    color=color,
                    opacity=opacities[mask],
                    line=dict(width=0.5, color="white"),
                ),
                name=name,
            )
        )

    if show_click_layer:
        grid_n = 35
        gx = np.linspace(x_min, x_max, grid_n)
        gy = np.linspace(y_min, y_max, grid_n)
        gxx, gyy = np.meshgrid(gx, gy)
        fig.add_trace(
            go.Scatter(
                x=gxx.ravel(),
                y=gyy.ravel(),
                mode="markers",
                marker=dict(size=14, opacity=0),
                name="_click_layer",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for i, (lx, ly, pred) in enumerate(labeled):
        if active_xy is not None and abs(lx - active_xy[0]) < 1e-12 and abs(ly - active_xy[1]) < 1e-12:
            continue
        color = _CLASS_COLORS.get(int(pred), "#111827")
        fig.add_trace(
            go.Scatter(
                x=[float(lx)],
                y=[float(ly)],
                mode="markers",
                marker=dict(size=14, color=color, symbol="x", line=dict(width=2, color="white")),
                name="已預測新點" if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate=(
                    "已預測<br>" + f0 + "=%{x:.3f}<br>" + f1 + "=%{y:.3f}"
                    + f"<br>類別={pred}<extra></extra>"
                ),
            )
        )

    if active_xy is not None:
        qx, qy = float(active_xy[0]), float(active_xy[1])
        if neighbor_indices:
            for i in neighbor_indices:
                fig.add_trace(
                    go.Scatter(
                        x=[qx, float(raw[i, 0])],
                        y=[qy, float(raw[i, 1])],
                        mode="lines",
                        line=dict(color="#6b7280", width=1.5, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[float(raw[i, 0])],
                        y=[float(raw[i, 1])],
                        mode="markers",
                        marker=dict(
                            size=14, color="#f59e0b", symbol="circle-open", line=dict(width=2)
                        ),
                        showlegend=False,
                        hovertemplate=(
                            "鄰居<br>" + f0 + "=%{x:.3f}<br>" + f1 + "=%{y:.3f}"
                            + (
                                f"<br>距離={float(distances_by_index[i]):.3f}<extra></extra>"
                                if distances_by_index is not None
                                else "<extra></extra>"
                            )
                        ),
                    )
                )
        marker_color = (
            _CLASS_COLORS.get(int(active_pred), "#111827")
            if active_pred is not None
            else "#111827"
        )
        fig.add_trace(
            go.Scatter(
                x=[qx],
                y=[qy],
                mode="markers",
                marker=dict(size=14, color=marker_color, symbol="x"),
                name="查詢中",
                hovertemplate="查詢點<br>" + f0 + "=%{x:.3f}<br>" + f1 + "=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=480,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title=f0,
        yaxis_title=f1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        clickmode="event+select",
        uirevision=chart_key,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _render_pretrain_quiz(
    *, features: list[str], target: str, source_label: str, row_count: int
) -> bool:
    if quiz.needs_quiz_reset(
        st.session_state.get(quiz.SESSION_NEIGHBORS_PAIR),
        features,
        target,
        source_label=source_label,
        tab="neighbors",
    ):
        st.session_state[quiz.SESSION_INST] = quiz.PLEASE_SELECT
        st.session_state[quiz.SESSION_VOTE] = quiz.PLEASE_SELECT
        st.session_state[quiz.SESSION_NEIGHBORS_FOCUS] = quiz.QID_INST
    st.session_state[quiz.SESSION_NEIGHBORS_PAIR] = quiz.pair_key(
        features, target, source_label=source_label, tab="neighbors"
    )
    st.session_state.setdefault(quiz.SESSION_INST, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_VOTE, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_NEIGHBORS_FOCUS, quiz.QID_INST)

    st.markdown("##### 訓練前先猜一下")
    st.caption(
        "兩題都答對後，「開始預測演示」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。"
    )
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        inst_choice = st.radio(
            "題1：用 K-近鄰做預測時，模型有沒有先學一組權重 $w$？",
            [quiz.PLEASE_SELECT, *quiz.INST_OPTIONS],
            key=quiz.SESSION_INST,
        )
    with h1_col:
        st.write("")
        if st.button("Agent 提示", key="knn_hint_inst", disabled=not agent_ready, width="stretch"):
            _send_hint(
                quiz.QID_INST,
                features=features,
                target=target,
                source_label=source_label,
                row_count=row_count,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    inst_ok = quiz.is_inst_correct(str(inst_choice))
    if str(inst_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_NEIGHBORS_FOCUS] = quiz.QID_INST
    elif inst_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想：邏輯迴歸／SVM 會學 w，K-近鄰呢？可按「Agent 提示」。")
        st.session_state[quiz.SESSION_NEIGHBORS_FOCUS] = quiz.QID_INST

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        vote_choice = st.radio(
            "題2：三個鄰居的標籤是 A、A、B，多數決會預測？",
            [quiz.PLEASE_SELECT, *quiz.VOTE_OPTIONS],
            key=quiz.SESSION_VOTE,
        )
    with h2_col:
        st.write("")
        if st.button("Agent 提示", key="knn_hint_vote", disabled=not agent_ready, width="stretch"):
            _send_hint(
                quiz.QID_VOTE,
                features=features,
                target=target,
                source_label=source_label,
                row_count=row_count,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    vote_ok = quiz.is_vote_correct(str(vote_choice))
    if str(vote_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if inst_ok:
            st.session_state[quiz.SESSION_NEIGHBORS_FOCUS] = quiz.QID_VOTE
    elif vote_ok:
        st.caption("題2 OK。")
    else:
        st.caption("題2 再想想「出現次數比較多的標籤」，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_NEIGHBORS_FOCUS] = quiz.QID_VOTE

    unlocked = quiz.both_neighbors_quiz_correct(str(inst_choice), str(vote_choice))
    if unlocked:
        st.success("2／2 題已準備好預測演示。")
    else:
        st.info(f"進度：{int(inst_ok) + int(vote_ok)}／2 題答對（需全部正確才解鎖預測演示）。")
    return unlocked


def _send_hint(
    qid: str,
    *,
    features: list[str],
    target: str,
    source_label: str,
    row_count: int,
) -> None:
    ts_key = f"knn_neighbors_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示發送中，請稍候再按。")
        return
    if not st.session_state.get("data_agent_connected"):
        st.warning("請先在右側啟用資料 Agent，再按「Agent 提示」。")
        return
    st.session_state[quiz.SESSION_NEIGHBORS_FOCUS] = qid
    st.session_state[ts_key] = now
    inst_choice = str(st.session_state.get(quiz.SESSION_INST, quiz.PLEASE_SELECT))
    vote_choice = str(st.session_state.get(quiz.SESSION_VOTE, quiz.PLEASE_SELECT))
    unlocked = quiz.both_neighbors_quiz_correct(inst_choice, vote_choice)
    extra = build_knn_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        k=DEFAULT_K,
        standardize=True,
        row_count=row_count,
        artifact=None,
        expose_k=True,
        prompt_train=unlocked,
        note=quiz.build_neighbors_quiz_agent_appendix(
            inst_status=quiz.quiz_choice_status(inst_choice, correct=quiz.is_inst_correct(inst_choice)),
            vote_status=quiz.quiz_choice_status(vote_choice, correct=quiz.is_vote_correct(vote_choice)),
            focus_qid=qid,
            features=features,
            target=target,
            unlocked=unlocked,
        ),
    )
    with st.spinner("正在詢問 Agent…"):
        invoke_data_agent(
            quiz.neighbors_hint_user_text(qid, features=features, target=target),
            extra_context=extra,
            display_user_text=quiz.neighbors_hint_display_text(qid),
        )
    st.rerun()


def _compose_agent_context() -> None:
    frag = str(st.session_state.get("_knn_ctx_frag", ""))
    unlocked = quiz.both_neighbors_quiz_correct(
        str(st.session_state.get(quiz.SESSION_INST, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_VOTE, quiz.PLEASE_SELECT)),
    )
    parts = [
        f"目前頁面：{PAGE_TITLE}（單頁：鄰居與投票，解鎖後可調 k）。",
        f"訓練前預測是否解鎖：{'是' if unlocked else '否'}。",
        "未解鎖前請勿直接告訴學生訓練前預測的正解選項。",
    ]
    if frag:
        parts.append(frag)
    st.session_state[CONTEXT_KEY] = "\n".join(parts)
