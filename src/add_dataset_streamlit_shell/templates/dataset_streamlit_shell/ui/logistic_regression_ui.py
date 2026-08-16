from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.ml.classification import (
    CONTOUR_U_MAX,
    CONTOUR_U_MIN,
    COST_DJ_DB_LOGISTIC_LATEX,
    COST_DJ_DW_LOGISTIC_LATEX,
    COST_DJ_DW_LOGISTIC_REG_LATEX,
    COST_GD_B_LOGISTIC_LATEX,
    COST_GD_W_LOGISTIC_LATEX,
    COST_J_LOGISTIC_LATEX,
    COST_J_LOGISTIC_REG_LATEX,
    DEFAULT_MAP_DEGREE,
    MODEL_KIND_LOGISTIC,
    MODEL_KIND_REGULARIZED,
    ClassificationArtifact,
    LogisticModelArtifact,
    RegularizedLogisticModelArtifact,
    build_classification_agent_context,
    logistic_gradient_descent_steps,
    map_feature,
    map_feature_row,
    predict_class_from_proba,
    predict_proba,
    predict_proba_from_logistic_artifact,
    predict_proba_from_regularized_artifact,
    training_accuracy,
)
from dataset_streamlit_shell.ml.regression import (
    GradientDescentStep,
    apply_standard_scaler,
    create_standard_scaler,
)
from dataset_streamlit_shell.plotting import (
    build_classification_data_figures,
    build_sigmoid_figure,
    configure_matplotlib_for_traditional_chinese,
    render_figures_in_streamlit,
    scatter_binary_classes,
)
from dataset_streamlit_shell.ui import logistic_quiz as quiz
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane
from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    invoke_data_agent,
    render_chat_panel,
    render_dataset_metrics,
)
from dataset_streamlit_shell.ui.teaching_flow import (
    CLASSIFICATION_FLOW_VIEW_INPUT,
    CLASSIFICATION_FLOW_VIEW_LABELS,
    CLASSIFICATION_FLOW_VIEW_MODEL,
    CLASSIFICATION_FLOW_VIEW_OUTPUT,
    MICRO_COST,
    MICRO_GRAD,
    MICRO_PREDICT,
    MICRO_STEP_LABELS,
    SAMPLE_OPS_HEAD,
    SAMPLE_OPS_POLY_NOTE,
    SAMPLE_OPS_SCALE_NOTE,
    TEACHING_FLOW_CSS,
    TrainingMicroFrame,
    build_training_micro_frames,
    classification_flow_svg,
    gradient_board_rows,
    live_fit_caption,
    logistic_sample_ops_table_rows,
    micro_stepper_html,
    regularized_compact_board_lines,
    sample_ops_cost_caption,
    sample_ops_table_visible,
    symbolic_logistic_yhat_latex,
    symbolic_logistic_z_latex,
    training_flow_state,
)

configure_matplotlib_for_traditional_chinese()

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
UNIVERSITY_ADMISSION_PATH = CLASSIFICATION_DEMO_DIR / "university_admission.csv"
MICROCHIP_TEST_PATH = CLASSIFICATION_DEMO_DIR / "microchip_test.csv"

ADMISSION_FEATURES = ["考試1分數", "考試2分數"]
ADMISSION_TARGET = "是否錄取"
MICROCHIP_FEATURES = ["檢測分數1", "檢測分數2"]
MICROCHIP_TARGET = "是否通過"

PAGE_TITLE = "邏輯迴歸"
CONTEXT_KEY = f"{PAGE_TITLE}_agent_context"


def render_logistic_regression_page() -> None:
    teaching, agent = open_content_dual_pane()
    with teaching:
        st.title(PAGE_TITLE)
        st.caption(
            "沿著教學流程圖：輸入資料 → 分類模型 → 輸出呈現；用「目前查看」一次看一節。"
            "訓練時會展開微步驟，演示梯度如何更新 w。"
        )
        st.markdown(TEACHING_FLOW_CSS, unsafe_allow_html=True)
        stage = st.radio(
            "學習階段",
            list(quiz.LEARNING_STAGES),
            horizontal=True,
            key="logistic_learning_stage",
        )
        if stage == quiz.STAGE_BOUNDARY:
            _render_boundary_stage()
        else:
            _render_poly_lambda_stage()
    with agent:
        render_chat_panel(
            extra_context=str(st.session_state.get(CONTEXT_KEY, f"目前頁面：{PAGE_TITLE}。")),
            page_name=PAGE_TITLE,
        )


def _paint_teaching_flow(
    placeholder,
    *,
    finished: bool = False,
    training: bool = False,
    live_caption: str = "",
) -> None:
    if training or finished:
        state = training_flow_state(finished=finished)
        html = classification_flow_svg(
            hot=state.hot,
            done=state.done,
            live_caption=live_caption,
        )
    else:
        html = classification_flow_svg()
    placeholder.markdown(html, unsafe_allow_html=True)


def _paint_micro_stepper(placeholder, *, hot: str | None) -> None:
    placeholder.markdown(micro_stepper_html(hot=hot), unsafe_allow_html=True)


def _render_view_radio(key: str) -> str:
    if key not in st.session_state:
        st.session_state[key] = CLASSIFICATION_FLOW_VIEW_MODEL
    return str(
        st.radio(
            "目前查看",
            CLASSIFICATION_FLOW_VIEW_LABELS,
            horizontal=True,
            key=key,
        )
    )


def _render_logistic_prediction_formula(*, mapped: bool) -> None:
    st.markdown("**預測公式**")
    if mapped:
        st.latex(rf"x \mapsto \phi(x)\ (\mathrm{{degree}}={DEFAULT_MAP_DEGREE})")
    st.latex(symbolic_logistic_z_latex(mapped=mapped))
    st.latex(symbolic_logistic_yhat_latex())
    st.caption("ŷ = σ(z) ∈ (0,1)，不是 0/1 類別。訓練中主公式保持符號式。")


def _micro_live_caption(
    frame: TrainingMicroFrame,
    *,
    compact: bool = False,
    lambda_: float | None = None,
) -> str:
    if not compact:
        return live_fit_caption(
            iteration=frame.iteration,
            total_iterations=frame.total_iterations,
            weights=frame.chart_weights,
            intercept=frame.chart_intercept,
            cost=frame.chart_cost,
        )
    w_sq = sum(float(weight) * float(weight) for weight in frame.chart_weights)
    extra = f"λ = {lambda_:g}，" if lambda_ is not None else ""
    return (
        f"Iteration {frame.iteration:,} / {frame.total_iterations:,}，"
        f"{extra}‖w‖² = {w_sq:.4f}，B = {frame.chart_intercept:.4f}，"
        f"Cost J = {frame.chart_cost:.4f}"
    )


def _cost_history_for_frame(
    steps: list[GradientDescentStep],
    frame: TrainingMicroFrame,
) -> list[GradientDescentStep]:
    if frame.micro_step == MICRO_UPDATE:
        return steps[: frame.iteration + 1]
    return steps[: frame.iteration]


def _render_gradient_board(
    frame: TrainingMicroFrame,
    *,
    compact: bool,
    lambda_: float | None = None,
) -> None:
    st.markdown("##### 梯度演算板")
    if compact:
        for line in regularized_compact_board_lines(frame, lambda_=float(lambda_ or 0.0)):
            st.markdown(f"- {line}")
        if frame.micro_step in {MICRO_GRAD, MICRO_UPDATE}:
            st.latex(COST_DJ_DW_LOGISTIC_REG_LATEX)
        with st.expander("全部 w 與 b", expanded=False):
            st.dataframe(
                pd.DataFrame(gradient_board_rows(frame)),
                width="stretch",
                hide_index=True,
            )
        return
    st.caption(
        f"目前步驟：{MICRO_STEP_LABELS[frame.micro_step]}｜"
        f"Iteration {frame.iteration:,} / {frame.total_iterations:,}｜"
        f"α = {frame.learning_rate:g}｜"
        f"Cost J（更新前）= {frame.cost_before:.6g}"
    )
    if frame.micro_step == MICRO_PREDICT:
        st.caption("用當前 w、b 計算 ŷ = σ(z)，z = w·x + b。")
    elif frame.micro_step == MICRO_COST:
        st.caption("看整批 log-loss J；逐筆是 ℓ，不是平方誤差。")
    st.dataframe(pd.DataFrame(gradient_board_rows(frame)), width="stretch", hide_index=True)


def _render_logistic_sample_ops_table(
    frame: TrainingMicroFrame,
    *,
    model_x: pd.DataFrame,
    target: pd.Series,
    show_x: bool,
) -> None:
    if not sample_ops_table_visible(frame.micro_step):
        return
    st.markdown("##### 樣本運算表")
    st.caption(SAMPLE_OPS_SCALE_NOTE if show_x else SAMPLE_OPS_POLY_NOTE)
    feature_names = list(frame.feature_names)
    head_x = model_x[feature_names].head(SAMPLE_OPS_HEAD)
    head_y = target.reindex(head_x.index)
    rows = logistic_sample_ops_table_rows(
        frame,
        model_x_rows=head_x.to_numpy(dtype=float).tolist(),
        y_rows=[float(value) for value in head_y.to_numpy(dtype=float)],
        show_x=show_x,
    )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    cost_caption = sample_ops_cost_caption(frame)
    if cost_caption is not None:
        st.caption(cost_caption)


def _render_boundary_stage() -> None:
    df = pd.read_csv(UNIVERSITY_ADMISSION_PATH)
    source_label = "內建範例資料：大學錄取（ex2data1）"
    render_dataset_metrics(df)

    features = list(ADMISSION_FEATURES)
    target = ADMISSION_TARGET
    working = _classification_training_frame(df, features, target)
    if len(working) < 2:
        st.warning("可用樣本少於 2 筆，無法訓練。")
        return

    scaler = create_standard_scaler(working, features)
    feature_matrix = apply_standard_scaler(working[features], scaler)

    view_key = "logistic_view"
    anim_key = "logistic_anim"
    result_key = "logistic_regression_last_artifact"
    lr_key = "logistic_learning_rate"
    epochs_key = "logistic_epochs"
    step_mode_key = "logistic_step_mode"

    learning_rate = float(st.session_state.get(lr_key, 0.001))
    epochs = int(st.session_state.get(epochs_key, 10000))
    signature = (
        source_label,
        tuple(features),
        target,
        learning_rate,
        epochs,
        len(working),
    )
    stored = st.session_state.get(result_key)
    artifact: LogisticModelArtifact | None = None
    if isinstance(stored, dict) and stored.get("signature") == signature:
        artifact = stored["artifact"]

    quiz_unlocked = quiz.both_boundary_quiz_correct(
        str(st.session_state.get(quiz.SESSION_SIGMOID, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_COST, quiz.PLEASE_SELECT)),
    )
    anim = st.session_state.get(anim_key)
    training_active = isinstance(anim, dict) and not anim.get("finished", True)

    flow_placeholder = st.empty()
    micro_placeholder = st.empty()

    if training_active:
        _run_training_session(
            working,
            model_x=feature_matrix,
            features=features,
            target=target,
            source_label=source_label,
            signature=signature,
            result_key=result_key,
            anim_key=anim_key,
            view_key=view_key,
            flow_placeholder=flow_placeholder,
            micro_placeholder=micro_placeholder,
            compact=False,
            scaler=scaler,
            mapped_features=None,
        )
        stored_after = st.session_state.get(result_key)
        if isinstance(stored_after, dict) and stored_after.get("signature") == signature:
            artifact = stored_after["artifact"]
        anim = st.session_state.get(anim_key)
        training_active = isinstance(anim, dict) and not anim.get("finished", True)

    if not training_active:
        _paint_teaching_flow(flow_placeholder, finished=artifact is not None)
        micro_placeholder.empty()
        view = _render_view_radio(view_key)
        if view == CLASSIFICATION_FLOW_VIEW_INPUT:
            st.markdown("###### 輸入資料")
            _render_classification_data_intro(
                working,
                features=features,
                target=target,
                dataset_note="每一列是一位申請者：兩科筆試為 x，是否錄取為 y（1=錄取、0=未錄取）。",
            )
        elif view == CLASSIFICATION_FLOW_VIEW_MODEL:
            st.markdown("###### 分類模型")
            st.markdown("##### 訓練設定")
            c1, c2 = st.columns(2)
            learning_rate = float(
                c1.number_input(
                    "學習率 α",
                    min_value=0.0001,
                    max_value=1.0,
                    value=learning_rate,
                    step=0.001,
                    format="%.4f",
                    key=lr_key,
                )
            )
            epochs = int(
                c2.number_input(
                    "Epoch / 迭代次數",
                    min_value=1,
                    max_value=20000,
                    value=epochs,
                    step=1,
                    key=epochs_key,
                )
            )
            st.caption(
                "訓練前會對特徵做 Z-score 縮放。教案參考：α=0.001、10000 次迭代，Cost 約可降至 0.30。"
            )
            signature = (
                source_label,
                tuple(features),
                target,
                learning_rate,
                epochs,
                len(working),
            )
            stored = st.session_state.get(result_key)
            if isinstance(stored, dict) and stored.get("signature") == signature:
                artifact = stored["artifact"]
            _render_logistic_prediction_formula(mapped=False)
            _render_sigmoid_visualization()
            _render_logistic_cost_formula(regularized=False)
            quiz_unlocked = _render_boundary_pretrain_quiz()
            step_mode = st.toggle("逐步模式", value=False, key=step_mode_key)
            if not quiz_unlocked:
                st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")
            train_clicked = st.button(
                "開始訓練",
                type="primary",
                width="stretch",
                key="train_logistic_regression",
                disabled=not quiz_unlocked,
            )
            if train_clicked and quiz_unlocked:
                steps = logistic_gradient_descent_steps(
                    feature_matrix,
                    working[target],
                    learning_rate=learning_rate,
                    epochs=epochs,
                )
                frames = build_training_micro_frames(
                    _animation_steps(steps),
                    learning_rate=learning_rate,
                    feature_names=features,
                )
                st.session_state[anim_key] = {
                    "frames": frames,
                    "steps": steps,
                    "index": 0,
                    "finished": False,
                    "step_mode": bool(step_mode),
                    "signature": signature,
                }
                st.rerun()
        else:
            _render_output_view(
                working,
                features=features,
                target=target,
                artifact=artifact,
                result_key=result_key,
                quiz_unlocked=quiz_unlocked,
                scaler=scaler,
                mapped_features=None,
                threshold_key="logistic",
            )

    _update_boundary_agent_context(
        source_label=source_label,
        features=features,
        target=target,
        learning_rate=float(st.session_state.get(lr_key, learning_rate)),
        epochs=int(st.session_state.get(epochs_key, epochs)),
        row_count=len(working),
        artifact=artifact,
    )


def _render_poly_lambda_stage() -> None:
    df = pd.read_csv(MICROCHIP_TEST_PATH)
    source_label = "內建範例資料：晶片檢測（ex2data2）"
    render_dataset_metrics(df)

    base_features = list(MICROCHIP_FEATURES)
    target = MICROCHIP_TARGET
    working = _classification_training_frame(df, base_features, target)
    if len(working) < 2:
        st.warning("可用樣本少於 2 筆，無法訓練。")
        return

    mapped, mapped_features = map_feature(working, base_features, degree=DEFAULT_MAP_DEGREE)

    view_key = "regularized_logistic_view"
    anim_key = "regularized_logistic_anim"
    result_key = "regularized_logistic_last_artifact"
    lr_key = "regularized_learning_rate"
    epochs_key = "regularized_epochs"
    lambda_key = "regularized_lambda"
    step_mode_key = "regularized_step_mode"

    learning_rate = float(st.session_state.get(lr_key, 0.01))
    epochs = int(st.session_state.get(epochs_key, 10000))
    lambda_ = float(st.session_state.get(lambda_key, 0.01))
    signature = (
        source_label,
        tuple(base_features),
        target,
        learning_rate,
        epochs,
        lambda_,
        len(working),
    )
    stored = st.session_state.get(result_key)
    artifact: RegularizedLogisticModelArtifact | None = None
    if isinstance(stored, dict) and stored.get("signature") == signature:
        artifact = stored["artifact"]

    quiz_unlocked = quiz.both_poly_quiz_correct(
        str(st.session_state.get(quiz.SESSION_MAP, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_LAMBDA, quiz.PLEASE_SELECT)),
    )
    anim = st.session_state.get(anim_key)
    training_active = isinstance(anim, dict) and not anim.get("finished", True)

    flow_placeholder = st.empty()
    micro_placeholder = st.empty()

    if training_active:
        _run_training_session(
            working,
            model_x=mapped,
            features=base_features,
            target=target,
            source_label=source_label,
            signature=signature,
            result_key=result_key,
            anim_key=anim_key,
            view_key=view_key,
            flow_placeholder=flow_placeholder,
            micro_placeholder=micro_placeholder,
            compact=True,
            scaler=None,
            mapped_features=mapped_features,
            lambda_=lambda_,
        )
        stored_after = st.session_state.get(result_key)
        if isinstance(stored_after, dict) and stored_after.get("signature") == signature:
            artifact = stored_after["artifact"]
        anim = st.session_state.get(anim_key)
        training_active = isinstance(anim, dict) and not anim.get("finished", True)

    if not training_active:
        _paint_teaching_flow(flow_placeholder, finished=artifact is not None)
        micro_placeholder.empty()
        view = _render_view_radio(view_key)
        if view == CLASSIFICATION_FLOW_VIEW_INPUT:
            st.markdown("###### 輸入資料")
            _render_classification_data_intro(
                working,
                features=base_features,
                target=target,
                dataset_note=(
                    f"原始 2 個 features 會映射為 {len(mapped_features)} 維多項式特徵"
                    f"（degree={DEFAULT_MAP_DEGREE}），再以 λ 做正則化邏輯迴歸。"
                ),
            )
        elif view == CLASSIFICATION_FLOW_VIEW_MODEL:
            st.markdown("###### 分類模型")
            st.markdown("##### 訓練設定")
            c1, c2, c3 = st.columns(3)
            learning_rate = float(
                c1.number_input(
                    "學習率 α",
                    min_value=0.0001,
                    max_value=1.0,
                    value=learning_rate,
                    step=0.001,
                    format="%.4f",
                    key=lr_key,
                )
            )
            epochs = int(
                c2.number_input(
                    "Epoch / 迭代次數",
                    min_value=1,
                    max_value=20000,
                    value=epochs,
                    step=1,
                    key=epochs_key,
                )
            )
            lambda_ = float(
                c3.number_input(
                    "正則化 λ",
                    min_value=0.0,
                    max_value=10.0,
                    value=lambda_,
                    step=0.001,
                    format="%.4f",
                    key=lambda_key,
                )
            )
            st.caption("教案參考：α=0.01、λ=0.01、10000 次迭代（預設已對齊）。")
            signature = (
                source_label,
                tuple(base_features),
                target,
                learning_rate,
                epochs,
                lambda_,
                len(working),
            )
            stored = st.session_state.get(result_key)
            if isinstance(stored, dict) and stored.get("signature") == signature:
                artifact = stored["artifact"]
            _render_logistic_prediction_formula(mapped=True)
            _render_logistic_cost_formula(regularized=True)
            quiz_unlocked = _render_poly_pretrain_quiz()
            step_mode = st.toggle("逐步模式", value=False, key=step_mode_key)
            if not quiz_unlocked:
                st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")
            train_clicked = st.button(
                "開始訓練",
                type="primary",
                width="stretch",
                key="train_regularized_logistic",
                disabled=not quiz_unlocked,
            )
            if train_clicked and quiz_unlocked:
                rng = np.random.default_rng(1)
                initial_w = rng.random(len(mapped_features)) - 0.5
                steps = logistic_gradient_descent_steps(
                    mapped,
                    working[target],
                    learning_rate=learning_rate,
                    epochs=epochs,
                    initial_weights=initial_w.tolist(),
                    initial_intercept=1.0,
                    lambda_=lambda_,
                    regularized=True,
                )
                frames = build_training_micro_frames(
                    _animation_steps(steps),
                    learning_rate=learning_rate,
                    feature_names=mapped_features,
                )
                st.session_state[anim_key] = {
                    "frames": frames,
                    "steps": steps,
                    "index": 0,
                    "finished": False,
                    "step_mode": bool(step_mode),
                    "signature": signature,
                    "lambda_": lambda_,
                }
                st.rerun()
        else:
            _render_output_view(
                working,
                features=base_features,
                target=target,
                artifact=artifact,
                result_key=result_key,
                quiz_unlocked=quiz_unlocked,
                scaler=None,
                mapped_features=mapped_features,
                threshold_key="regularized",
            )

    _update_poly_agent_context(
        source_label=source_label,
        features=base_features,
        target=target,
        learning_rate=float(st.session_state.get(lr_key, learning_rate)),
        epochs=int(st.session_state.get(epochs_key, epochs)),
        lambda_=float(st.session_state.get(lambda_key, lambda_)),
        row_count=len(working),
        artifact=artifact,
    )


def _run_training_session(
    working: pd.DataFrame,
    *,
    model_x: pd.DataFrame,
    features: list[str],
    target: str,
    source_label: str,
    signature: tuple,
    result_key: str,
    anim_key: str,
    view_key: str,
    flow_placeholder,
    micro_placeholder,
    compact: bool,
    scaler: dict | None,
    mapped_features: list[str] | None,
    lambda_: float | None = None,
) -> None:
    anim = st.session_state.get(anim_key)
    if not isinstance(anim, dict) or anim.get("finished"):
        return
    frames: list[TrainingMicroFrame] = list(anim["frames"])
    steps: list[GradientDescentStep] = list(anim["steps"])
    if not frames:
        anim["finished"] = True
        return

    last_plot_key: bool | None = None
    board_col, chart_col = st.columns(2, gap="medium")
    with board_col:
        board_placeholder = st.empty()
        next_placeholder = st.empty()
    with chart_col:
        plot_placeholder = st.empty()
        cost_placeholder = st.empty()
        status_placeholder = st.empty()

    def _show_frame(frame: TrainingMicroFrame) -> None:
        nonlocal last_plot_key
        caption = _micro_live_caption(frame, compact=compact, lambda_=lambda_)
        _paint_teaching_flow(flow_placeholder, training=True, live_caption=caption)
        _paint_micro_stepper(micro_placeholder, hot=frame.micro_step)
        with board_placeholder.container():
            _render_gradient_board(frame, compact=compact, lambda_=lambda_)
            _render_logistic_sample_ops_table(
                frame,
                model_x=model_x,
                target=working[target],
                show_x=not compact,
            )
        chart_step = GradientDescentStep(
            iteration=frame.iteration,
            weights=list(frame.chart_weights),
            intercept=float(frame.chart_intercept),
            cost=float(frame.chart_cost),
        )
        # 階段2 contour 很貴；同一組 w 的預測／Cost／梯度微步重用上一張圖。
        redraw_plot = (not compact) or last_plot_key is None or frame.micro_step == MICRO_UPDATE
        if redraw_plot:
            if compact and mapped_features is not None:
                _render_regularized_contour_plot(
                    working,
                    features,
                    target,
                    mapped_features,
                    chart_step,
                    plot_placeholder,
                    grid_size=30,
                )
            else:
                _render_logistic_boundary_plot(
                    working,
                    features,
                    target,
                    chart_step,
                    plot_placeholder,
                    scaler=scaler,
                )
            last_plot_key = True
        _render_cost_history_plot(_cost_history_for_frame(steps, frame), cost_placeholder)
        status_placeholder.caption(caption)

    def _finalize() -> None:
        final_step = steps[-1]
        if compact and mapped_features is not None:
            artifact: ClassificationArtifact = RegularizedLogisticModelArtifact(
                model_kind=MODEL_KIND_REGULARIZED,
                base_features=list(features),
                mapped_features=list(mapped_features),
                target=target,
                weights=[float(value) for value in final_step.weights],
                intercept=float(final_step.intercept),
                map_degree=DEFAULT_MAP_DEGREE,
                lambda_=float(lambda_ or 0.0),
                training_cost=float(final_step.cost),
                data_source=source_label,
            )
        else:
            artifact = LogisticModelArtifact(
                model_kind=MODEL_KIND_LOGISTIC,
                features=list(features),
                target=target,
                weights=[float(value) for value in final_step.weights],
                intercept=float(final_step.intercept),
                scaler=scaler,
                training_cost=float(final_step.cost),
                data_source=source_label,
            )
        st.session_state[result_key] = {
            "signature": signature,
            "artifact": artifact,
            "steps": steps,
        }
        anim["finished"] = True
        st.session_state[view_key] = CLASSIFICATION_FLOW_VIEW_OUTPUT
        _paint_teaching_flow(flow_placeholder, finished=True)
        micro_placeholder.empty()

    if anim.get("step_mode"):
        index = int(anim.get("index", 0))
        index = min(max(index, 0), len(frames) - 1)
        _show_frame(frames[index])
        with next_placeholder.container():
            next_key = "regularized_logistic_next_micro" if compact else "logistic_next_micro"
            if st.button("下一步", type="primary", key=next_key):
                if index + 1 >= len(frames):
                    _finalize()
                else:
                    anim["index"] = index + 1
                st.rerun()
        return

    for frame in frames:
        _show_frame(frame)
        time.sleep(0.015)
    _finalize()
    st.rerun()


def _render_output_view(
    working: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    artifact: ClassificationArtifact | None,
    result_key: str,
    quiz_unlocked: bool,
    scaler: dict | None,
    mapped_features: list[str] | None,
    threshold_key: str,
) -> None:
    st.markdown("###### 輸出呈現")
    st.caption("圖上決策邊界／contour 維持 ŷ=0.5；threshold 只切表上的類別與正確率。")
    chart_left, chart_right = st.columns(2)
    plot_placeholder = chart_left.empty()
    cost_placeholder = chart_right.empty()
    status_placeholder = st.empty()
    threshold = _classification_threshold_slider(threshold_key, enabled=artifact is not None)
    if artifact is None:
        st.caption("訓練後這裡會出現決策邊界、Cost 曲線與 y／ŷ／predicted_class 表。")
        if quiz_unlocked:
            status_placeholder.info(
                "兩題已過關。到「分類模型」按下「開始訓練」，觀察梯度如何更新 w。"
            )
        else:
            status_placeholder.info("先到「分類模型」完成兩題訓練前預測，再開始訓練。")
        return

    status_placeholder.caption("顯示最近一次訓練結果；調整設定後請重新按「開始訓練」。")
    chart_step = GradientDescentStep(
        iteration=0,
        weights=[float(value) for value in artifact.weights],
        intercept=float(artifact.intercept),
        cost=float(artifact.training_cost),
    )
    if mapped_features is not None:
        _render_regularized_contour_plot(
            working,
            features,
            target,
            mapped_features,
            chart_step,
            plot_placeholder,
            grid_size=30,
        )
    else:
        _render_logistic_boundary_plot(
            working,
            features,
            target,
            chart_step,
            plot_placeholder,
            scaler=scaler,
        )
    result_bundle = st.session_state.get(result_key)
    cached_steps = result_bundle.get("steps") if isinstance(result_bundle, dict) else None
    if isinstance(cached_steps, list) and cached_steps:
        _render_cost_history_plot(cached_steps, cost_placeholder)
    if isinstance(artifact, LogisticModelArtifact):
        probability = predict_proba_from_logistic_artifact(artifact, working[artifact.features])
    else:
        probability = predict_proba_from_regularized_artifact(artifact, working)
    _render_logistic_training_results(artifact, working, target, probability, threshold)


def _update_boundary_agent_context(
    *,
    source_label: str,
    features: list[str],
    target: str,
    learning_rate: float,
    epochs: int,
    row_count: int,
    artifact: LogisticModelArtifact | None,
) -> None:
    quiz_unlocked = quiz.both_boundary_quiz_correct(
        str(st.session_state.get(quiz.SESSION_SIGMOID, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_COST, quiz.PLEASE_SELECT)),
    )
    focus = st.session_state.get(quiz.SESSION_FOCUS_BOUNDARY)
    threshold = float(st.session_state.get("logistic_threshold", 0.5))
    appendix = _boundary_quiz_appendix(unlocked=quiz_unlocked, focus_qid=focus)
    base_context = build_classification_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        learning_rate=learning_rate,
        epochs=epochs,
        row_count=row_count,
        artifact=artifact,
        threshold=threshold if artifact is not None else None,
    )
    st.session_state[CONTEXT_KEY] = f"{base_context}\n{appendix}"
    _render_classification_prompts(
        quiz.focus_prompt_lines(focus, stage=quiz.STAGE_BOUNDARY, unlocked=quiz_unlocked)
    )


def _update_poly_agent_context(
    *,
    source_label: str,
    features: list[str],
    target: str,
    learning_rate: float,
    epochs: int,
    lambda_: float,
    row_count: int,
    artifact: RegularizedLogisticModelArtifact | None,
) -> None:
    quiz_unlocked = quiz.both_poly_quiz_correct(
        str(st.session_state.get(quiz.SESSION_MAP, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_LAMBDA, quiz.PLEASE_SELECT)),
    )
    focus = st.session_state.get(quiz.SESSION_FOCUS_POLY)
    threshold = float(st.session_state.get("regularized_threshold", 0.5))
    appendix = _poly_quiz_appendix(unlocked=quiz_unlocked, focus_qid=focus)
    base_context = build_classification_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        learning_rate=learning_rate,
        epochs=epochs,
        row_count=row_count,
        artifact=artifact,
        lambda_=lambda_,
        map_degree=DEFAULT_MAP_DEGREE,
        threshold=threshold if artifact is not None else None,
    )
    st.session_state[CONTEXT_KEY] = f"{base_context}\n{appendix}"
    _render_classification_prompts(
        quiz.focus_prompt_lines(focus, stage=quiz.STAGE_POLY_LAMBDA, unlocked=quiz_unlocked)
    )


def _render_boundary_pretrain_quiz() -> bool:
    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    st.session_state.setdefault(quiz.SESSION_SIGMOID, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_COST, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_FOCUS_BOUNDARY, quiz.QID_SIGMOID)

    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        sigmoid_choice = st.radio(
            "題1：sigmoid 輸出大致代表什麼？",
            [quiz.PLEASE_SELECT, *quiz.SIGMOID_OPTIONS],
            key=quiz.SESSION_SIGMOID,
        )
    with h1_col:
        st.write("")
        if st.button("Agent 提示", key="logistic_hint_sigmoid", disabled=not agent_ready, width="stretch"):
            _send_boundary_quiz_hint(quiz.QID_SIGMOID)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    sigmoid_ok = quiz.is_sigmoid_correct(str(sigmoid_choice))
    if str(sigmoid_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_FOCUS_BOUNDARY] = quiz.QID_SIGMOID
    elif sigmoid_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想：輸出是不是 0～1 的機率？可按「Agent 提示」。")
        st.session_state[quiz.SESSION_FOCUS_BOUNDARY] = quiz.QID_SIGMOID

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        cost_choice = st.radio(
            "題2：訓練時的 Cost 與分類 threshold 的關係比較接近？",
            [quiz.PLEASE_SELECT, *quiz.COST_OPTIONS],
            key=quiz.SESSION_COST,
        )
    with h2_col:
        st.write("")
        if st.button("Agent 提示", key="logistic_hint_cost", disabled=not agent_ready, width="stretch"):
            _send_boundary_quiz_hint(quiz.QID_COST)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    cost_ok = quiz.is_cost_correct(str(cost_choice))
    if str(cost_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if sigmoid_ok:
            st.session_state[quiz.SESSION_FOCUS_BOUNDARY] = quiz.QID_COST
    elif cost_ok:
        st.caption("題2 OK。")
    else:
        st.caption("題2 再想想 Cost 看的是機率擬合還是 threshold，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_FOCUS_BOUNDARY] = quiz.QID_COST

    unlocked = quiz.both_boundary_quiz_correct(str(sigmoid_choice), str(cost_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(sigmoid_ok) + int(cost_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _render_poly_pretrain_quiz() -> bool:
    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    st.session_state.setdefault(quiz.SESSION_MAP, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_LAMBDA, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_FOCUS_POLY, quiz.QID_MAP)

    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        map_choice = st.radio(
            "題1：為什麼常把兩個檢測分數做成多項式特徵映射？",
            [quiz.PLEASE_SELECT, *quiz.MAP_OPTIONS],
            key=quiz.SESSION_MAP,
        )
    with h1_col:
        st.write("")
        if st.button("Agent 提示", key="logistic_hint_map", disabled=not agent_ready, width="stretch"):
            _send_poly_quiz_hint(quiz.QID_MAP)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    map_ok = quiz.is_map_correct(str(map_choice))
    if str(map_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_FOCUS_POLY] = quiz.QID_MAP
    elif map_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想：直線邊界夠不夠用？可按「Agent 提示」。")
        st.session_state[quiz.SESSION_FOCUS_POLY] = quiz.QID_MAP

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        lambda_choice = st.radio(
            "題2：λ 變大時，決策邊界／過擬合傾向比較可能？",
            [quiz.PLEASE_SELECT, *quiz.LAMBDA_OPTIONS],
            key=quiz.SESSION_LAMBDA,
        )
    with h2_col:
        st.write("")
        if st.button("Agent 提示", key="logistic_hint_lambda", disabled=not agent_ready, width="stretch"):
            _send_poly_quiz_hint(quiz.QID_LAMBDA)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    lambda_ok = quiz.is_lambda_correct(str(lambda_choice))
    if str(lambda_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if map_ok:
            st.session_state[quiz.SESSION_FOCUS_POLY] = quiz.QID_LAMBDA
    elif lambda_ok:
        st.caption("題2 OK。")
    else:
        st.caption("題2 再想想 λ 對權重大小與邊界彎曲的影響，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_FOCUS_POLY] = quiz.QID_LAMBDA

    unlocked = quiz.both_poly_quiz_correct(str(map_choice), str(lambda_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(map_ok) + int(lambda_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _boundary_quiz_appendix(*, unlocked: bool, focus_qid: str | None) -> str:
    sigmoid_choice = str(st.session_state.get(quiz.SESSION_SIGMOID, quiz.PLEASE_SELECT))
    cost_choice = str(st.session_state.get(quiz.SESSION_COST, quiz.PLEASE_SELECT))
    return quiz.build_boundary_quiz_agent_appendix(
        sigmoid_status=quiz.quiz_choice_status(
            sigmoid_choice, correct=quiz.is_sigmoid_correct(sigmoid_choice)
        ),
        cost_status=quiz.quiz_choice_status(
            cost_choice, correct=quiz.is_cost_correct(cost_choice)
        ),
        focus_qid=focus_qid,
        unlocked=unlocked,
    )


def _poly_quiz_appendix(*, unlocked: bool, focus_qid: str | None) -> str:
    map_choice = str(st.session_state.get(quiz.SESSION_MAP, quiz.PLEASE_SELECT))
    lambda_choice = str(st.session_state.get(quiz.SESSION_LAMBDA, quiz.PLEASE_SELECT))
    return quiz.build_poly_quiz_agent_appendix(
        map_status=quiz.quiz_choice_status(map_choice, correct=quiz.is_map_correct(map_choice)),
        lambda_status=quiz.quiz_choice_status(
            lambda_choice, correct=quiz.is_lambda_correct(lambda_choice)
        ),
        focus_qid=focus_qid,
        unlocked=unlocked,
    )


def _send_boundary_quiz_hint(qid: str) -> None:
    ts_key = f"logistic_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示冷卻中，稍后再試。")
        return
    st.session_state[ts_key] = now
    st.session_state[quiz.SESSION_FOCUS_BOUNDARY] = qid
    sigmoid_choice = str(st.session_state.get(quiz.SESSION_SIGMOID, quiz.PLEASE_SELECT))
    cost_choice = str(st.session_state.get(quiz.SESSION_COST, quiz.PLEASE_SELECT))
    unlocked = quiz.both_boundary_quiz_correct(sigmoid_choice, cost_choice)
    invoke_data_agent(
        quiz.hint_user_text(qid),
        extra_context=_boundary_quiz_appendix(unlocked=unlocked, focus_qid=qid),
        display_user_text=quiz.hint_display_text(qid),
    )
    st.rerun()


def _send_poly_quiz_hint(qid: str) -> None:
    ts_key = f"logistic_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示冷卻中，稍后再試。")
        return
    st.session_state[ts_key] = now
    st.session_state[quiz.SESSION_FOCUS_POLY] = qid
    map_choice = str(st.session_state.get(quiz.SESSION_MAP, quiz.PLEASE_SELECT))
    lambda_choice = str(st.session_state.get(quiz.SESSION_LAMBDA, quiz.PLEASE_SELECT))
    unlocked = quiz.both_poly_quiz_correct(map_choice, lambda_choice)
    invoke_data_agent(
        quiz.hint_user_text(qid),
        extra_context=_poly_quiz_appendix(unlocked=unlocked, focus_qid=qid),
        display_user_text=quiz.hint_display_text(qid),
    )
    st.rerun()


def _classification_training_frame(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    columns = list(features) + [target]
    return df[columns].apply(pd.to_numeric, errors="coerce").dropna()


def _render_classification_data_intro(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    dataset_note: str,
) -> None:
    st.markdown("##### Data 資訊")
    st.info(dataset_note)
    role_rows = []
    for column in features + [target]:
        series = pd.to_numeric(frame[column], errors="coerce")
        role_rows.append(
            {
                "欄位": column,
                "角色": "target（y）" if column == target else "feature（x）",
                "資料型態": str(frame[column].dtype),
                "缺失值": int(frame[column].isna().sum()),
                "最小值": float(series.min()),
                "最大值": float(series.max()),
                "平均值": float(series.mean()),
            }
        )
    st.dataframe(
        pd.DataFrame(role_rows).style.format(
            {"最小值": "{:.4f}", "最大值": "{:.4f}", "平均值": "{:.4f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("資料預覽", expanded=True):
        st.dataframe(frame[features + [target]].head(10), width="stretch", hide_index=True)
    render_figures_in_streamlit(build_classification_data_figures(frame, features, target))


def _render_sigmoid_visualization() -> None:
    with st.expander("Sigmoid 函數視覺化", expanded=True):
        st.caption(
            "邏輯迴歸先把特徵線性組合成 z，再經 sigmoid 壓到 0～1，作為屬於類別 1 的機率。"
        )
        highlight_z = st.slider(
            "在曲線上標示 z",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            key="logistic_sigmoid_highlight_z",
        )
        fig = build_sigmoid_figure(highlight_z=highlight_z)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        prob = float(1.0 / (1.0 + np.exp(-np.clip(highlight_z, -500, 500))))
        st.caption(f"當 z = {highlight_z:g} 時，σ(z) ≈ {prob:.4f}")


def _render_logistic_cost_formula(*, regularized: bool = False) -> None:
    with st.expander("成本與梯度下降", expanded=False):
        if regularized:
            st.latex(COST_J_LOGISTIC_REG_LATEX)
            st.caption("Cost 依 sigmoid 機率 f 計算，另加 λ 對 w 的正則化；與分類 threshold 無關。")
        else:
            st.latex(COST_J_LOGISTIC_LATEX)
            st.caption("Cost 只依 sigmoid 機率 f 計算，與分類 threshold 無關。")
        left, right = st.columns(2)
        with left:
            st.markdown("**梯度下降演算法**")
            st.markdown("`repeat until convergence:` `{`")
            st.latex(COST_GD_W_LOGISTIC_LATEX)
            st.latex(COST_GD_B_LOGISTIC_LATEX)
            st.markdown("`}`")
        with right:
            st.markdown("**導數項**")
            if regularized:
                st.latex(COST_DJ_DW_LOGISTIC_REG_LATEX)
            else:
                st.latex(COST_DJ_DW_LOGISTIC_LATEX)
            st.latex(COST_DJ_DB_LOGISTIC_LATEX)
            if regularized:
                st.caption("對 b 的導數不加 λ 項。")


def _classification_threshold_slider(page_key: str, *, enabled: bool) -> float:
    return st.slider(
        "分類 threshold（訓練後調整）",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        disabled=not enabled,
        key=f"{page_key}_threshold",
    )


def _animation_steps(steps: list[GradientDescentStep]) -> list[GradientDescentStep]:
    if len(steps) <= 80:
        return steps
    stride = max(len(steps) // 80, 1)
    selected = steps[::stride]
    if selected[-1] != steps[-1]:
        selected.append(steps[-1])
    return selected


def _render_logistic_boundary_plot(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    step: GradientDescentStep,
    placeholder,
    *,
    scaler: dict | None,
) -> None:
    x1_name, x2_name = features[0], features[1]
    plot_frame = frame[[x1_name, x2_name]]
    w1, w2 = float(step.weights[0]), float(step.weights[1])
    intercept = float(step.intercept)
    if scaler is not None:
        m1 = float(scaler["mean"][x1_name])
        s1 = float(scaler["scale"][x1_name])
        m2 = float(scaler["mean"][x2_name])
        s2 = float(scaler["scale"][x2_name])
        w1_plot = w1 / s1
        w2_plot = w2 / s2
        intercept_plot = intercept - w1 * m1 / s1 - w2 * m2 / s2
    else:
        w1_plot, w2_plot, intercept_plot = w1, w2, intercept
    x1 = plot_frame[x1_name]
    x2 = plot_frame[x2_name]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    positives = frame[target] == 1
    negatives = frame[target] == 0
    scatter_binary_classes(ax, x1, x2, positives=positives, negatives=negatives)
    if abs(w2_plot) > 1e-12:
        line_x = np.linspace(float(x1.min()), float(x1.max()), 100)
        line_y = -(w1_plot * line_x + intercept_plot) / w2_plot
        ax.plot(line_x, line_y, color="blue", label="決策邊界")
    ax.set_xlabel(x1_name)
    ax.set_ylabel(x2_name)
    ax.set_title(f"決策邊界（iteration {step.iteration}）")
    ax.legend()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_regularized_contour_plot(
    frame: pd.DataFrame,
    base_features: list[str],
    target: str,
    mapped_features: list[str],
    step: GradientDescentStep,
    placeholder,
    *,
    grid_size: int,
) -> None:
    x1_name, x2_name = base_features
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    positives = frame[target] == 1
    negatives = frame[target] == 0
    scatter_binary_classes(
        ax,
        frame[x1_name].to_numpy(dtype=float),
        frame[x2_name].to_numpy(dtype=float),
        positives=positives,
        negatives=negatives,
    )
    u = np.linspace(CONTOUR_U_MIN, CONTOUR_U_MAX, grid_size)
    v = np.linspace(CONTOUR_U_MIN, CONTOUR_U_MAX, grid_size)
    z_grid = np.zeros((len(u), len(v)))
    weights = np.asarray(step.weights, dtype=float)
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            mapped_row = map_feature_row(
                base_features,
                {x1_name: float(ui), x2_name: float(vj)},
                degree=DEFAULT_MAP_DEGREE,
            )
            z_val = float(
                predict_proba(mapped_row[mapped_features], weights, step.intercept).iloc[0]
            )
            z_grid[i, j] = z_val
    ax.contour(u, v, z_grid.T, levels=[0.5], colors="green")
    ax.set_xlabel(x1_name)
    ax.set_ylabel(x2_name)
    ax.set_title(f"決策邊界 contour f=0.5（iteration {step.iteration}）")
    ax.legend()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_cost_history_plot(steps: list[GradientDescentStep], placeholder) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot([step.iteration for step in steps], [step.cost for step in steps], color="orange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J")
    ax.set_title("Cost vs Iteration")
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_logistic_training_results(
    artifact: ClassificationArtifact,
    working: pd.DataFrame,
    target: str,
    probability: pd.Series,
    threshold: float,
) -> None:
    st.markdown("##### 訓練結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("最後 B", f"{artifact.intercept:.4f}")
    c2.metric("最後 Cost J", f"{artifact.training_cost:.4f}")
    c3.metric(
        "訓練集正確率",
        f"{training_accuracy(working[target], probability, threshold):.2f}%",
    )
    if isinstance(artifact, RegularizedLogisticModelArtifact):
        w_sq = sum(float(weight) * float(weight) for weight in artifact.weights)
        st.caption(f"λ = {artifact.lambda_:g}，‖w‖² = {w_sq:.4g}（輸出主路徑不列 27 個 w）。")
    predicted = predict_class_from_proba(probability, threshold)
    result = pd.DataFrame(
        {
            "y": working[target],
            "ŷ": probability,
            "predicted_class": predicted,
        }
    )
    st.dataframe(result.head(30).style.format({"ŷ": "{:.4f}"}), width="stretch")


def _render_classification_prompts(prompts: list[str]) -> None:
    st.markdown("##### 建議問 Agent")
    for prompt in prompts:
        st.code(prompt, language="text")
