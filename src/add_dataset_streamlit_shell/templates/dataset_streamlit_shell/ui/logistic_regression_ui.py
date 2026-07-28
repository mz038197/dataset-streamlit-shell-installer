from __future__ import annotations

import time
from pathlib import Path

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
from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    invoke_data_agent,
    render_chat_panel,
    render_dataset_metrics,
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
MODEL_FORMULA_LATEX = r"f_{w,b}(x)=\frac{1}{1+e^{-(w\cdot x+b)}}"


def render_logistic_regression_page() -> None:
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title(PAGE_TITLE)
        st.caption("先用直線決策邊界建立 sigmoid／Cost，再看多項式映射與 λ 正則化。")
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
    with side:
        render_chat_panel(
            extra_context=str(st.session_state.get(CONTEXT_KEY, f"目前頁面：{PAGE_TITLE}。")),
            page_name=PAGE_TITLE,
        )


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

    _render_classification_data_intro(
        working,
        features=features,
        target=target,
        dataset_note="每一列是一位申請者：兩科筆試為 x，是否錄取為 y（1=錄取、0=未錄取）。",
    )
    # 考試分數尺度約 30–100；未縮放時 α≈0.01 會讓 Cost 爆炸，教案 α=0.001 也幾乎不降。
    scaler = create_standard_scaler(working, features)
    feature_matrix = apply_standard_scaler(working[features], scaler)

    st.markdown("##### 訓練設定")
    c1, c2 = st.columns(2)
    learning_rate = c1.number_input(
        "學習率 α",
        min_value=0.0001,
        max_value=1.0,
        value=0.001,
        step=0.001,
        format="%.4f",
        key="logistic_learning_rate",
    )
    epochs = c2.number_input(
        "Epoch / 迭代次數",
        min_value=1,
        max_value=20000,
        value=10000,
        step=1,
        key="logistic_epochs",
    )
    st.caption(
        "訓練前會對特徵做 Z-score 縮放。教案參考：α=0.001、10000 次迭代，Cost 約可降至 0.30。"
    )
    st.markdown("##### 模型公式")
    st.latex(MODEL_FORMULA_LATEX)
    _render_sigmoid_visualization()
    _render_logistic_cost_formula(regularized=False)

    quiz_unlocked = _render_boundary_pretrain_quiz()
    result_key = "logistic_regression_last_artifact"
    signature = (
        source_label,
        tuple(features),
        target,
        float(learning_rate),
        int(epochs),
        len(working),
    )
    if not quiz_unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_logistic_regression",
        disabled=not quiz_unlocked,
    )
    artifact: LogisticModelArtifact | None = None
    if train_clicked and quiz_unlocked:
        steps = logistic_gradient_descent_steps(
            feature_matrix,
            working[target],
            learning_rate=float(learning_rate),
            epochs=int(epochs),
        )
        chart_left, chart_right = st.columns(2)
        boundary_placeholder = chart_left.empty()
        cost_placeholder = chart_right.empty()
        status_placeholder = st.empty()
        _animate_logistic_boundary(
            working,
            features,
            target,
            steps,
            boundary_placeholder,
            cost_placeholder,
            status_placeholder,
            scaler=scaler,
        )
        final_step = steps[-1]
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
        st.session_state[result_key] = {"signature": signature, "artifact": artifact}
    else:
        stored = st.session_state.get(result_key)
        if isinstance(stored, dict) and stored.get("signature") == signature:
            artifact = stored["artifact"]
            st.caption("顯示最近一次訓練結果；調整設定後請重新按「開始訓練」。")
        elif quiz_unlocked:
            st.info("兩題已過關。按下「開始訓練」觀察決策邊界與 Cost 的演進。")

    threshold = _classification_threshold_slider("logistic", enabled=artifact is not None)
    focus = st.session_state.get(quiz.SESSION_FOCUS_BOUNDARY)
    appendix = _boundary_quiz_appendix(unlocked=quiz_unlocked, focus_qid=focus)
    base_context = build_classification_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        row_count=len(working),
        artifact=artifact,
        threshold=threshold if artifact is not None else None,
    )
    st.session_state[CONTEXT_KEY] = f"{base_context}\n{appendix}"
    if artifact is not None:
        probability = predict_proba_from_logistic_artifact(artifact, working[artifact.features])
        _render_logistic_training_results(artifact, working, target, probability, threshold)
    _render_classification_prompts(
        quiz.focus_prompt_lines(focus, stage=quiz.STAGE_BOUNDARY, unlocked=quiz_unlocked)
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
    _render_classification_data_intro(
        working,
        features=base_features,
        target=target,
        dataset_note=(
            f"原始 2 個 features 會映射為 {len(mapped_features)} 維多項式特徵"
            f"（degree={DEFAULT_MAP_DEGREE}），再以 λ 做正則化邏輯迴歸。"
        ),
    )

    st.markdown("##### 訓練設定")
    c1, c2, c3 = st.columns(3)
    learning_rate = c1.number_input(
        "學習率 α",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.4f",
        key="regularized_learning_rate",
    )
    epochs = c2.number_input(
        "Epoch / 迭代次數",
        min_value=1,
        max_value=20000,
        value=10000,
        step=1,
        key="regularized_epochs",
    )
    lambda_ = c3.number_input(
        "正則化 λ",
        min_value=0.0,
        max_value=10.0,
        value=0.01,
        step=0.001,
        format="%.4f",
        key="regularized_lambda",
    )
    st.caption("教案參考：α=0.01、λ=0.01、10000 次迭代（預設已對齊）。")
    st.markdown("##### 模型公式")
    st.latex(rf"x \mapsto \phi(x)\ (\mathrm{{degree}}={DEFAULT_MAP_DEGREE})")
    st.latex(r"f_{w,b}(x)=\frac{1}{1+e^{-(w\cdot \phi(x)+b)}}")
    _render_logistic_cost_formula(regularized=True)

    quiz_unlocked = _render_poly_pretrain_quiz()
    result_key = "regularized_logistic_last_artifact"
    signature = (
        source_label,
        tuple(base_features),
        target,
        float(learning_rate),
        int(epochs),
        float(lambda_),
        len(working),
    )
    if not quiz_unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_regularized_logistic",
        disabled=not quiz_unlocked,
    )
    artifact: RegularizedLogisticModelArtifact | None = None
    if train_clicked and quiz_unlocked:
        rng = np.random.default_rng(1)
        initial_w = rng.random(len(mapped_features)) - 0.5
        steps = logistic_gradient_descent_steps(
            mapped,
            working[target],
            learning_rate=float(learning_rate),
            epochs=int(epochs),
            initial_weights=initial_w.tolist(),
            initial_intercept=1.0,
            lambda_=float(lambda_),
            regularized=True,
        )
        chart_left, chart_right = st.columns(2)
        contour_placeholder = chart_left.empty()
        cost_placeholder = chart_right.empty()
        status_placeholder = st.empty()
        _animate_regularized_contour(
            working,
            base_features,
            target,
            mapped_features,
            steps,
            contour_placeholder,
            cost_placeholder,
            status_placeholder,
        )
        final_step = steps[-1]
        artifact = RegularizedLogisticModelArtifact(
            model_kind=MODEL_KIND_REGULARIZED,
            base_features=list(base_features),
            mapped_features=list(mapped_features),
            target=target,
            weights=[float(value) for value in final_step.weights],
            intercept=float(final_step.intercept),
            map_degree=DEFAULT_MAP_DEGREE,
            lambda_=float(lambda_),
            training_cost=float(final_step.cost),
            data_source=source_label,
        )
        st.session_state[result_key] = {"signature": signature, "artifact": artifact}
    else:
        stored = st.session_state.get(result_key)
        if isinstance(stored, dict) and stored.get("signature") == signature:
            artifact = stored["artifact"]
            st.caption("顯示最近一次訓練結果；調整設定後請重新按「開始訓練」。")
        elif quiz_unlocked:
            st.info("兩題已過關。按下「開始訓練」觀察 contour 與 Cost 的演進。")

    threshold = _classification_threshold_slider("regularized", enabled=artifact is not None)
    focus = st.session_state.get(quiz.SESSION_FOCUS_POLY)
    appendix = _poly_quiz_appendix(unlocked=quiz_unlocked, focus_qid=focus)
    base_context = build_classification_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=base_features,
        target=target,
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        row_count=len(working),
        artifact=artifact,
        lambda_=float(lambda_),
        map_degree=DEFAULT_MAP_DEGREE,
        threshold=threshold if artifact is not None else None,
    )
    st.session_state[CONTEXT_KEY] = f"{base_context}\n{appendix}"
    if artifact is not None:
        probability = predict_proba_from_regularized_artifact(artifact, working)
        _render_logistic_training_results(artifact, working, target, probability, threshold)
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


def _animate_logistic_boundary(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    steps: list[GradientDescentStep],
    boundary_placeholder,
    cost_placeholder,
    status_placeholder,
    *,
    scaler: dict | None,
) -> None:
    for step in _animation_steps(steps):
        _render_logistic_boundary_plot(
            frame, features, target, step, boundary_placeholder, scaler=scaler
        )
        _render_cost_history_plot(steps[: step.iteration + 1], cost_placeholder)
        status_placeholder.caption(
            f"Iteration {step.iteration:,} / {steps[-1].iteration:,}，"
            f"Cost J = {step.cost:.4f}"
        )
        time.sleep(0.02)


def _animate_regularized_contour(
    frame: pd.DataFrame,
    base_features: list[str],
    target: str,
    mapped_features: list[str],
    steps: list[GradientDescentStep],
    contour_placeholder,
    cost_placeholder,
    status_placeholder,
) -> None:
    for step in _animation_steps(steps):
        _render_regularized_contour_plot(
            frame,
            base_features,
            target,
            mapped_features,
            step,
            contour_placeholder,
            grid_size=30,
        )
        _render_cost_history_plot(steps[: step.iteration + 1], cost_placeholder)
        status_placeholder.caption(
            f"Iteration {step.iteration:,} / {steps[-1].iteration:,}，Cost J = {step.cost:.4f}"
        )
        time.sleep(0.02)


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
    # 散點維持原始分數尺度；若訓練用 Z-score，把邊界反變換回原始座標
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
    predicted = predict_class_from_proba(probability, threshold)
    result = pd.DataFrame(
        {
            "actual": working[target],
            "probability": probability,
            "predicted_class": predicted,
        }
    )
    st.dataframe(result.head(30).style.format({"probability": "{:.4f}"}), width="stretch")


def _render_classification_prompts(prompts: list[str]) -> None:
    st.markdown("##### 建議問 Agent")
    for prompt in prompts:
        st.code(prompt, language="text")
