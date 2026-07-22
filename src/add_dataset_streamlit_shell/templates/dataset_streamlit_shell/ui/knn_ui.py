"""K-近鄰分類頁：雙階段＋訓練前預測＋Plotly 查詢點 click。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataset_streamlit_shell.ml.knn import (
    DEFAULT_K,
    KnnArtifact,
    build_knn_agent_context,
    build_knn_artifact,
    decision_mesh_predictions,
    fit_knn_classifier,
    nearest_neighbor_indices,
    odd_k_values,
    predict_class_from_artifact,
    prepare_feature_matrix,
)
from dataset_streamlit_shell.ui import knn_quiz as quiz
from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    invoke_data_agent,
    render_chat_panel,
    render_dataset_metrics,
)

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
KNN_BLOBS_PATH = CLASSIFICATION_DEMO_DIR / "knn_blobs_80.csv"
KNN_TRAP_PATH = CLASSIFICATION_DEMO_DIR / "knn_scale_trap_80.csv"

KNN_FEATURES = ["特徵1", "特徵2"]
KNN_TARGET = "類別"
PAGE_TITLE = "K-近鄰分類"
CONTEXT_KEY = f"{PAGE_TITLE}_agent_context"
STAGE_NEIGHBORS_LABEL = "鄰居與投票"
STAGE_K_LABEL = "選擇 k"


def render_knn_page() -> None:
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title(PAGE_TITLE)
        st.caption("先搞懂鄰居與多數決，再調 k（並看特徵尺度如何影響距離）。")

        stage = st.radio(
            "學習階段",
            [STAGE_NEIGHBORS_LABEL, STAGE_K_LABEL],
            horizontal=True,
            key="knn_learning_stage",
        )
        if stage == STAGE_NEIGHBORS_LABEL:
            st.session_state[quiz.SESSION_PAGE_FOCUS] = "neighbors"
            _render_neighbors_stage()
        else:
            st.session_state[quiz.SESSION_PAGE_FOCUS] = "k"
            _render_k_stage()
        _compose_agent_context()

    with side:
        render_chat_panel(
            extra_context=str(st.session_state.get(CONTEXT_KEY, f"目前頁面：{PAGE_TITLE}。")),
            page_name=PAGE_TITLE,
        )


def _render_neighbors_stage() -> None:
    st.markdown("##### 這一階段在問什麼")
    st.info(
        "K-近鄰是**實例型**學習：不先學一組權重 w，預測時找出距離最近的 k 個訓練點，再**多數決**。"
    )
    prepared = _prepare_stage_data(
        builtin_path=KNN_BLOBS_PATH,
        builtin_label="內建範例資料：兩特徵二元分類（尺度相近，80 筆）",
    )
    if prepared is None:
        return
    working, features, target, source_label = prepared

    st.markdown("##### 預測怎麼做")
    st.latex(r"\hat{y}=\mathrm{majority}\{y_{(1)},\ldots,y_{(k)}\}")
    st.caption(quiz.NEIGHBORS_FORMULA_CAPTION)

    unlocked = _render_neighbors_pretrain_quiz(
        features=features, target=target, source_label=source_label, row_count=len(working)
    )

    st.markdown("##### 訓練")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_knn_neighbors",
        disabled=not unlocked,
    )
    if not unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")

    k = DEFAULT_K
    standardize = True
    result_key = "knn_neighbors_last_artifact"
    signature = (source_label, tuple(features), target, k, standardize, len(working), "neighbors")

    artifact = _resolve_or_train_artifact(
        train_clicked=train_clicked and unlocked,
        working=working,
        features=features,
        target=target,
        source_label=source_label,
        k=k,
        standardize=standardize,
        result_key=result_key,
        signature=signature,
        stale_caption="顯示最近一次訓練結果；換階段或資料後請重新訓練。",
        query_key="knn_neighbors_query_xy",
        chart_key="knn_neighbors_plotly",
        expose_k=False,
    )

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
    _merge_agent_context(
        tab="neighbors",
        source_label=source_label,
        features=features,
        target=target,
        k=k,
        standardize=standardize,
        row_count=len(working),
        artifact=artifact,
        expose_k=False,
        prompt_train=unlocked,
        note="目前階段：鄰居與投票。\n" + appendix,
    )

    with st.expander("完整 sklearn 範例程式", expanded=False):
        st.code(quiz.SKLEARN_NEIGHBORS_EXAMPLE, language="python")


def _render_k_stage() -> None:
    st.markdown("##### 這一階段在問什麼")
    st.info(
        "同一套鄰居投票，**k** 會改變邊界平滑程度；"
        "若某一軸數值範圍特別大又**沒標準化**，距離會被那一軸主導。"
    )
    prepared = _prepare_stage_data(
        builtin_path=KNN_TRAP_PATH,
        builtin_label="內建範例資料：刻意拉大特徵2尺度（80 筆，用來對照標準化）",
    )
    if prepared is None:
        return
    working, features, target, source_label = prepared

    st.markdown("##### 訓練設定")
    k_choices = odd_k_values()
    k = st.select_slider(
        "鄰居數 k",
        options=k_choices,
        value=DEFAULT_K if DEFAULT_K in k_choices else k_choices[0],
        key="knn_k_slider",
        help="奇數步進，減少平票。k 變大通常邊界更平滑。",
    )
    standardize = st.toggle(
        "特徵標準化（Z-score）",
        value=True,
        key="knn_k_standardize",
        help="關閉後可觀察大尺度特徵如何主導歐氏距離。",
    )

    unlocked = _render_k_pretrain_quiz(
        features=features,
        target=target,
        source_label=source_label,
        k=int(k),
        standardize=bool(standardize),
        row_count=len(working),
    )

    st.markdown("##### 訓練")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_knn_k",
        disabled=not unlocked,
    )
    if not unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")
    else:
        st.caption("解鎖後可改 k／標準化再訓練，對照邊界與鄰居。")

    result_key = "knn_k_last_artifact"
    signature = (
        source_label,
        tuple(features),
        target,
        int(k),
        bool(standardize),
        len(working),
        "k",
    )

    artifact = _resolve_or_train_artifact(
        train_clicked=train_clicked and unlocked,
        working=working,
        features=features,
        target=target,
        source_label=source_label,
        k=int(k),
        standardize=bool(standardize),
        result_key=result_key,
        signature=signature,
        stale_caption="顯示最近一次訓練結果；調整 k／標準化後請重新按「開始訓練」。",
        query_key="knn_k_query_xy",
        chart_key="knn_k_plotly",
        expose_k=True,
    )

    focus = st.session_state.get(quiz.SESSION_K_FOCUS)
    k_choice = str(st.session_state.get(quiz.SESSION_K, quiz.PLEASE_SELECT))
    scale_choice = str(st.session_state.get(quiz.SESSION_SCALE, quiz.PLEASE_SELECT))
    appendix = quiz.build_k_quiz_agent_appendix(
        k_status=quiz.quiz_choice_status(k_choice, correct=quiz.is_k_correct(k_choice)),
        scale_status=quiz.quiz_choice_status(
            scale_choice, correct=quiz.is_scale_correct(scale_choice)
        ),
        focus_qid=focus,
        features=features,
        target=target,
        k=int(k),
        standardize=bool(standardize),
        unlocked=unlocked,
    )
    _merge_agent_context(
        tab="k",
        source_label=source_label,
        features=features,
        target=target,
        k=int(k),
        standardize=bool(standardize),
        row_count=len(working),
        artifact=artifact,
        expose_k=True,
        prompt_train=unlocked,
        note="目前階段：選擇 k。\n" + appendix,
    )

    with st.expander("完整 sklearn 範例程式", expanded=False):
        st.code(quiz.SKLEARN_K_EXAMPLE, language="python")


def _prepare_stage_data(
    *,
    builtin_path: Path,
    builtin_label: str,
) -> tuple[pd.DataFrame, list[str], str, str] | None:
    df = pd.read_csv(builtin_path)
    render_dataset_metrics(df)
    features = list(KNN_FEATURES)
    target = KNN_TARGET
    working = df[features + [target]].dropna().copy()
    working[target] = pd.to_numeric(working[target], errors="coerce").astype(int)
    if len(working) < 2:
        st.warning("可用樣本少於 2 筆，無法訓練。")
        return None
    st.caption(builtin_label)
    return working, features, target, builtin_label


def _resolve_or_train_artifact(
    *,
    train_clicked: bool,
    working: pd.DataFrame,
    features: list[str],
    target: str,
    source_label: str,
    k: int,
    standardize: bool,
    result_key: str,
    signature: tuple,
    stale_caption: str,
    query_key: str,
    chart_key: str,
    expose_k: bool,
) -> KnnArtifact | None:
    artifact: KnnArtifact | None = None
    if train_clicked:
        try:
            feature_matrix, scaler = prepare_feature_matrix(
                working, features, standardize=standardize
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
            standardize=standardize,
            scaler=scaler,
            data_source=source_label,
            feature_frame=feature_matrix,
            target_series=working[target],
        )
        st.session_state[result_key] = {"signature": signature, "artifact": artifact}
    else:
        stored = st.session_state.get(result_key)
        if isinstance(stored, dict) and stored.get("signature") == signature:
            artifact = stored["artifact"]
            st.caption(stale_caption)

    if artifact is None:
        return None

    if expose_k:
        c1, c2, c3 = st.columns(3)
        c1.metric("k", str(artifact.k))
        c2.metric("標準化", "開" if artifact.scaler is not None else "關")
        c3.metric("訓練集正確率", f"{artifact.training_accuracy:.2f}%")
    else:
        c1, c2 = st.columns(2)
        c1.metric("標準化", "開" if artifact.scaler is not None else "關")
        c2.metric("訓練集正確率", f"{artifact.training_accuracy:.2f}%")

    st.markdown("##### 決策邊界與查詢點")
    st.caption(quiz.result_chart_caption(expose_k=expose_k))
    _render_knn_plotly(
        working,
        artifact,
        query_key=query_key,
        chart_key=chart_key,
        expose_k=expose_k,
    )
    return artifact


def _render_knn_plotly(
    working: pd.DataFrame,
    artifact: KnnArtifact,
    *,
    query_key: str,
    chart_key: str,
    expose_k: bool,
) -> None:
    f0, f1 = artifact.features
    x = working[f0].to_numpy(dtype=float)
    y = working[f1].to_numpy(dtype=float)
    labels = working[artifact.target].to_numpy(dtype=int)

    pad_x = (x.max() - x.min()) * 0.08 + 1e-6
    pad_y = (y.max() - y.min()) * 0.08 + 1e-6
    x_min, x_max = float(x.min() - pad_x), float(x.max() + pad_x)
    y_min, y_max = float(y.min() - pad_y), float(y.max() + pad_y)

    xx, yy, zz = decision_mesh_predictions(
        artifact, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, grid_size=70
    )

    fig = go.Figure()
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

    for cls, color, name in ((0, "#2563eb", "類別 0"), (1, "#dc2626", "類別 1")):
        mask = labels == cls
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                marker=dict(size=9, color=color, line=dict(width=0.5, color="white")),
                name=name,
            )
        )

    # 透明點陣：讓「點空白處」也能選到近似座標（matplotlib 做不到 click）
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

    query = st.session_state.get(query_key)
    if query is None:
        query = (float(np.median(x)), float(np.median(y)))
        st.session_state[query_key] = query
    qx, qy = float(query[0]), float(query[1])

    try:
        neighbor_idx, _dist = nearest_neighbor_indices(artifact, (qx, qy))
        # train_x 為縮放後空間；圖上連線用原始座標（與 working 列順序一致）
        raw = working[artifact.features].to_numpy(dtype=float)
        for i in neighbor_idx:
            i = int(i)
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
                        "鄰居<br>" + f0 + "=%{x:.3f}<br>" + f1 + "=%{y:.3f}<extra></extra>"
                    ),
                )
            )
        pred = int(
            predict_class_from_artifact(
                artifact, pd.DataFrame([{f0: qx, f1: qy}])
            )[0]
        )
        st.caption(
            quiz.query_prediction_caption(
                qx, qy, pred, k=artifact.k, expose_k=expose_k
            )
        )
    except Exception as exc:  # noqa: BLE001 — 顯示給學生看即可
        st.caption(f"查詢點鄰居暫無法計算：{exc}")

    fig.add_trace(
        go.Scatter(
            x=[qx],
            y=[qy],
            mode="markers",
            marker=dict(size=14, color="#111827", symbol="x"),
            name="查詢點",
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

    event = st.plotly_chart(
        fig,
        width="stretch",
        key=chart_key,
        on_select="rerun",
        selection_mode="points",
    )
    points = getattr(getattr(event, "selection", None), "points", None) or []
    if points:
        pt = points[0]
        if "x" in pt and "y" in pt:
            new_q = (float(pt["x"]), float(pt["y"]))
            prev = st.session_state.get(query_key)
            if prev is None or abs(prev[0] - new_q[0]) > 1e-9 or abs(prev[1] - new_q[1]) > 1e-9:
                st.session_state[query_key] = new_q
                st.rerun()


def _render_neighbors_pretrain_quiz(
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
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        inst_choice = st.radio(
            "題1：用 K-近鄰做預測時，模型有沒有先學一組權重 \(w\)？",
            [quiz.PLEASE_SELECT, *quiz.INST_OPTIONS],
            key=quiz.SESSION_INST,
        )
    with h1_col:
        st.write("")
        if st.button("Agent 提示", key="knn_hint_inst", disabled=not agent_ready, width="stretch"):
            _send_neighbors_hint(
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
            _send_neighbors_hint(
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
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(inst_ok) + int(vote_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _render_k_pretrain_quiz(
    *,
    features: list[str],
    target: str,
    source_label: str,
    k: int,
    standardize: bool,
    row_count: int,
) -> bool:
    if quiz.needs_quiz_reset(
        st.session_state.get(quiz.SESSION_K_PAIR),
        features,
        target,
        source_label=source_label,
        tab="k",
    ):
        st.session_state[quiz.SESSION_K] = quiz.PLEASE_SELECT
        st.session_state[quiz.SESSION_SCALE] = quiz.PLEASE_SELECT
        st.session_state[quiz.SESSION_K_FOCUS] = quiz.QID_K
    st.session_state[quiz.SESSION_K_PAIR] = quiz.pair_key(
        features, target, source_label=source_label, tab="k"
    )
    st.session_state.setdefault(quiz.SESSION_K, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_SCALE, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_K_FOCUS, quiz.QID_K)

    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        k_choice = st.radio(
            "題1：k 變得很接近訓練筆數時，邊界／預測通常會？",
            [quiz.PLEASE_SELECT, *quiz.K_OPTIONS],
            key=quiz.SESSION_K,
        )
    with h1_col:
        st.write("")
        if st.button("Agent 提示", key="knn_hint_k", disabled=not agent_ready, width="stretch"):
            _send_k_hint(
                quiz.QID_K,
                features=features,
                target=target,
                source_label=source_label,
                k=k,
                standardize=standardize,
                row_count=row_count,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    k_ok = quiz.is_k_correct(str(k_choice))
    if str(k_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_K_FOCUS] = quiz.QID_K
    elif k_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想「幾乎全班來投票」會怎樣，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_K_FOCUS] = quiz.QID_K

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        scale_choice = st.radio(
            "題2：某一特徵數值範圍遠大於另一個、又沒做標準化時，歐氏距離主要被誰主導？",
            [quiz.PLEASE_SELECT, *quiz.SCALE_OPTIONS],
            key=quiz.SESSION_SCALE,
        )
    with h2_col:
        st.write("")
        if st.button("Agent 提示", key="knn_hint_scale", disabled=not agent_ready, width="stretch"):
            _send_k_hint(
                quiz.QID_SCALE,
                features=features,
                target=target,
                source_label=source_label,
                k=k,
                standardize=standardize,
                row_count=row_count,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    scale_ok = quiz.is_scale_correct(str(scale_choice))
    if str(scale_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if k_ok:
            st.session_state[quiz.SESSION_K_FOCUS] = quiz.QID_SCALE
    elif scale_ok:
        st.caption("題2 OK。")
    else:
        st.caption("題2 再想想平方差誰比較大，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_K_FOCUS] = quiz.QID_SCALE

    unlocked = quiz.both_k_quiz_correct(str(k_choice), str(scale_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(k_ok) + int(scale_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _send_neighbors_hint(
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
    st.session_state[quiz.SESSION_PAGE_FOCUS] = "neighbors"
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
        expose_k=False,
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


def _send_k_hint(
    qid: str,
    *,
    features: list[str],
    target: str,
    source_label: str,
    k: int,
    standardize: bool,
    row_count: int,
) -> None:
    ts_key = f"knn_k_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示發送中，請稍候再按。")
        return
    if not st.session_state.get("data_agent_connected"):
        st.warning("請先在右側啟用資料 Agent，再按「Agent 提示」。")
        return
    st.session_state[quiz.SESSION_K_FOCUS] = qid
    st.session_state[quiz.SESSION_PAGE_FOCUS] = "k"
    st.session_state[ts_key] = now
    k_choice = str(st.session_state.get(quiz.SESSION_K, quiz.PLEASE_SELECT))
    scale_choice = str(st.session_state.get(quiz.SESSION_SCALE, quiz.PLEASE_SELECT))
    unlocked = quiz.both_k_quiz_correct(k_choice, scale_choice)
    extra = build_knn_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        k=k,
        standardize=standardize,
        row_count=row_count,
        artifact=None,
        expose_k=True,
        prompt_train=unlocked,
        note=quiz.build_k_quiz_agent_appendix(
            k_status=quiz.quiz_choice_status(k_choice, correct=quiz.is_k_correct(k_choice)),
            scale_status=quiz.quiz_choice_status(
                scale_choice, correct=quiz.is_scale_correct(scale_choice)
            ),
            focus_qid=qid,
            features=features,
            target=target,
            k=k,
            standardize=standardize,
            unlocked=unlocked,
        ),
    )
    with st.spinner("正在詢問 Agent…"):
        invoke_data_agent(
            quiz.k_hint_user_text(
                qid, features=features, target=target, k=k, standardize=standardize
            ),
            extra_context=extra,
            display_user_text=quiz.k_hint_display_text(qid),
        )
    st.rerun()


def _merge_agent_context(
    *,
    tab: str,
    source_label: str,
    features: list[str],
    target: str,
    k: int,
    standardize: bool,
    row_count: int,
    artifact: KnnArtifact | None,
    expose_k: bool,
    prompt_train: bool,
    note: str,
) -> None:
    st.session_state[f"_knn_ctx_frag_{tab}"] = build_knn_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        k=k,
        standardize=standardize,
        row_count=row_count,
        artifact=artifact,
        note=note,
        expose_k=expose_k,
        prompt_train=prompt_train,
    )


def _compose_agent_context() -> None:
    neighbors = str(st.session_state.get("_knn_ctx_frag_neighbors", ""))
    k_frag = str(st.session_state.get("_knn_ctx_frag_k", ""))
    neighbors_unlocked = quiz.both_neighbors_quiz_correct(
        str(st.session_state.get(quiz.SESSION_INST, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_VOTE, quiz.PLEASE_SELECT)),
    )
    k_unlocked = quiz.both_k_quiz_correct(
        str(st.session_state.get(quiz.SESSION_K, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_SCALE, quiz.PLEASE_SELECT)),
    )
    focus = st.session_state.get(quiz.SESSION_PAGE_FOCUS, "neighbors")
    parts = [
        f"目前頁面：{PAGE_TITLE}（雙階段：鄰居與投票 → 選擇 k）。",
        f"目前學習階段焦點：{focus}。",
        f"階段1訓練前預測是否解鎖：{'是' if neighbors_unlocked else '否'}。",
        f"階段2訓練前預測是否解鎖：{'是' if k_unlocked else '否'}。",
        "未解鎖前請勿直接告訴學生訓練前預測的正解選項。",
    ]
    if focus == "k" and k_frag:
        parts.append(k_frag)
    elif neighbors:
        parts.append(neighbors)
    elif k_frag:
        parts.append(k_frag)
    st.session_state[CONTEXT_KEY] = "\n".join(parts)
