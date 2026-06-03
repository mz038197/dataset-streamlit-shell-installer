from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.data_ui import (
    READY_DATASET_PATH,
    SHELL_ROOT,
    _display_path,
    load_ready_dataset,
    render_chat_panel,
    render_dataset_metrics,
)
from dataset_streamlit_shell.ml.classification import (
    CONTOUR_U_MAX,
    CONTOUR_U_MIN,
    DEFAULT_MAP_DEGREE,
    MODEL_KIND_LOGISTIC,
    MODEL_KIND_REGULARIZED,
    ClassificationArtifact,
    LogisticModelArtifact,
    RegularizedLogisticModelArtifact,
    artifact_from_payload,
    build_classification_agent_context,
    logistic_gradient_descent_steps,
    map_feature,
    map_feature_row,
    predict_class_from_proba,
    predict_proba,
    predict_proba_from_logistic_artifact,
    predict_proba_from_regularized_artifact,
    save_classification_artifact,
    training_accuracy,
    validate_binary_target,
)
from dataset_streamlit_shell.ml.regression import (
    GradientDescentStep,
    apply_standard_scaler,
    create_standard_scaler,
)
from dataset_streamlit_shell.plotting import configure_matplotlib_for_traditional_chinese

configure_matplotlib_for_traditional_chinese()

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
CLASSIFICATION_MODEL_DIR = SHELL_ROOT / "workspace" / "models" / "classification"
UNIVERSITY_ADMISSION_PATH = CLASSIFICATION_DEMO_DIR / "university_admission.csv"
MICROCHIP_TEST_PATH = CLASSIFICATION_DEMO_DIR / "microchip_test.csv"

ADMISSION_FEATURES = ["考試1分數", "考試2分數"]
ADMISSION_TARGET = "是否錄取"
MICROCHIP_FEATURES = ["檢測分數1", "檢測分數2"]
MICROCHIP_TARGET = "是否通過"


def render_logistic_regression_page() -> None:
    def body(df: pd.DataFrame, source_label: str, *, builtin: bool) -> None:
        st.markdown("##### 邏輯迴歸")
        st.info(
            "依兩科考試成績預測是否錄取。訓練使用 logistic Cost J 與梯度下降；"
            "分類 threshold 在訓練完成後才用於解讀預測類別。"
        )
        if builtin:
            features = list(ADMISSION_FEATURES)
            target = ADMISSION_TARGET
            working = _classification_training_frame(df, features, target)
        else:
            numeric_columns = _numeric_columns(df)
            if len(numeric_columns) < 2:
                st.warning("至少需要 1 個 target 與 1 個 feature。")
                return
            default_target = _default_column(numeric_columns, ADMISSION_TARGET)
            target = st.selectbox(
                "選擇 target（y，0/1）",
                numeric_columns,
                index=numeric_columns.index(default_target)
                if default_target in numeric_columns
                else 0,
                key="logistic_target",
            )
            if not validate_binary_target(df[target]):
                st.warning("target 必須為 0/1。請在編碼頁先整理，或改用內建範例資料。")
                return
            feature_options = [column for column in numeric_columns if column != target]
            default_features = [column for column in ADMISSION_FEATURES if column in feature_options]
            if not default_features:
                default_features = feature_options[: min(2, len(feature_options))]
            features = st.multiselect(
                "選擇 features（x）",
                feature_options,
                default=default_features,
                key="logistic_features",
            )
            if not features:
                st.warning("請至少選擇 1 個 feature。")
                return
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
        feature_matrix, scaler = _prepare_logistic_features(working, features, builtin=builtin)

        st.markdown("##### 訓練設定")
        c1, c2 = st.columns(2)
        learning_rate = c1.number_input(
            "學習率 α",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.4f",
            key="logistic_learning_rate",
        )
        epochs = c2.number_input(
            "Epoch / 迭代次數",
            min_value=1,
            max_value=5000,
            value=10,
            step=1,
            key="logistic_epochs",
        )
        st.caption("教案參考：α=0.001、10000 次迭代，Cost 約可降至 0.30。")
        st.markdown("##### 模型公式")
        st.latex(r"f_{\mathbf{w},b}(\mathbf{x})=\mathrm{sigmoid}(\mathbf{w}\cdot\mathbf{x}+b)")
        _render_logistic_cost_formula()

        result_key = "logistic_regression_last_artifact"
        context_key = "邏輯迴歸_agent_context"
        signature = (
            source_label,
            tuple(features),
            target,
            float(learning_rate),
            int(epochs),
            len(working),
            builtin,
        )
        train_clicked = st.button(
            "開始訓練",
            type="primary",
            use_container_width=True,
            key="train_logistic_regression",
        )
        artifact: LogisticModelArtifact | None = None
        if train_clicked:
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
            if len(features) == 2:
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
            else:
                _animate_logistic_proba(
                    working[target],
                    feature_matrix,
                    steps,
                    boundary_placeholder,
                    cost_placeholder,
                    status_placeholder,
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
            else:
                st.info("設定 α 與 epoch 後，按下「開始訓練」觀察決策邊界與 Cost 的演進。")

        threshold = _classification_threshold_slider("logistic", enabled=artifact is not None)
        st.session_state[context_key] = build_classification_agent_context(
            page_name="邏輯迴歸",
            data_source=source_label,
            features=features,
            target=target,
            learning_rate=float(learning_rate),
            epochs=int(epochs),
            row_count=len(working),
            artifact=artifact,
            threshold=threshold if artifact is not None else None,
        )
        if artifact is not None:
            probability = predict_proba_from_logistic_artifact(artifact, working[artifact.features])
            _render_logistic_training_results(artifact, working, target, probability, threshold)
            _render_classification_save_section(
                artifact,
                filename_prefix="logistic_regression",
                page_key="logistic",
            )
        _render_logistic_inference_section(
            page_key="logistic",
            trained_artifact=artifact,
            threshold=threshold,
        )
        _render_classification_prompts(
            [
                "請解釋這條決策邊界代表什麼，以及錄取機率如何隨考試分數改變。",
                "請用 Cost J 說明模型目前擬合得好不好。",
                "調整 threshold 後，訓練集正確率如何變化？",
            ]
        )

    _classification_page_shell(
        "邏輯迴歸",
        "使用內建大學錄取資料或目前 ready.csv，練習二元邏輯迴歸與決策邊界。",
        "內建範例資料：大學錄取（ex2data1）",
        UNIVERSITY_ADMISSION_PATH,
        lambda df, label, builtin: body(df, label, builtin=builtin),
    )


def render_regularized_logistic_regression_page() -> None:
    def body(df: pd.DataFrame, source_label: str, *, builtin: bool) -> None:
        st.markdown("##### 正則化邏輯迴歸")
        st.info(
            "晶片兩項檢測分數預測是否通過。訓練前會將 2 個 features 映射為 6 次多項式（27 維），"
            "並以 λ 做正則化；threshold 僅用於訓練後的類別預測。"
        )
        if builtin:
            base_features = list(MICROCHIP_FEATURES)
            target = MICROCHIP_TARGET
            working = _classification_training_frame(df, base_features, target)
        else:
            numeric_columns = _numeric_columns(df)
            if len(numeric_columns) < 3:
                st.warning("至少需要 2 個 features 與 1 個 target。")
                return
            default_target = _default_column(numeric_columns, MICROCHIP_TARGET)
            target = st.selectbox(
                "選擇 target（y，0/1）",
                numeric_columns,
                index=numeric_columns.index(default_target)
                if default_target in numeric_columns
                else 0,
                key="regularized_target",
            )
            if not validate_binary_target(df[target]):
                st.warning("target 必須為 0/1。")
                return
            feature_options = [column for column in numeric_columns if column != target]
            default_features = [column for column in MICROCHIP_FEATURES if column in feature_options]
            selected = st.multiselect(
                "選擇 2 個原始 features（x1, x2）",
                feature_options,
                default=default_features if len(default_features) == 2 else feature_options[:2],
                key="regularized_features",
            )
            if len(selected) != 2:
                st.warning("正則化邏輯迴歸需要恰好 2 個原始 features 才能做特徵映射。")
                return
            base_features = selected
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
                f"原始 2 個 features 會映射為 {len(mapped_features)} 維多項式特徵（degree={DEFAULT_MAP_DEGREE}），"
                "再進行正則化邏輯迴歸。"
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
            max_value=5000,
            value=10,
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
        st.caption("教案參考：α=0.01、λ=0.01、10000 次迭代。")
        st.markdown("##### 模型公式")
        st.latex(
            r"J(\mathbf{w},b)=-\frac{1}{m}\sum loss + \frac{\lambda}{2m}\sum_j w_j^2"
        )
        _render_logistic_cost_formula(regularized=True)

        result_key = "regularized_logistic_last_artifact"
        context_key = "正則化邏輯迴歸_agent_context"
        signature = (
            source_label,
            tuple(base_features),
            target,
            float(learning_rate),
            int(epochs),
            float(lambda_),
            len(working),
            builtin,
        )
        train_clicked = st.button(
            "開始訓練",
            type="primary",
            use_container_width=True,
            key="train_regularized_logistic",
        )
        artifact: RegularizedLogisticModelArtifact | None = None
        if train_clicked:
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
            else:
                st.info("設定 α、epoch、λ 後，按下「開始訓練」觀察 contour 與 Cost 的演進。")

        threshold = _classification_threshold_slider("regularized", enabled=artifact is not None)
        st.session_state[context_key] = build_classification_agent_context(
            page_name="正則化邏輯迴歸",
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
        if artifact is not None:
            probability = predict_proba_from_regularized_artifact(artifact, working)
            _render_logistic_training_results(artifact, working, target, probability, threshold)
            _render_classification_save_section(
                artifact,
                filename_prefix="regularized_logistic_regression",
                page_key="regularized",
            )
        _render_regularized_inference_section(
            page_key="regularized",
            trained_artifact=artifact,
            threshold=threshold,
        )
        _render_classification_prompts(
            [
                "請解釋為什麼晶片資料需要多項式特徵映射與正則化。",
                "λ 變大時，決策邊界與 Cost 可能如何改變？",
                "請找出被判錯的樣本，推測可能原因。",
            ]
        )

    _classification_page_shell(
        "正則化邏輯迴歸",
        "使用內建晶片檢測資料或 ready.csv（需 2 個 features），練習特徵映射與 λ。",
        "內建範例資料：晶片檢測（ex2data2）",
        MICROCHIP_TEST_PATH,
        lambda df, label, builtin: body(df, label, builtin=builtin),
    )


def _classification_page_shell(
    title: str,
    caption: str,
    builtin_label: str,
    builtin_path: Path,
    render_main,
) -> None:
    main, side = st.columns([5, 3], gap="large")
    context_key = f"{title}_agent_context"
    with main:
        st.title(title)
        st.caption(caption)
        source = st.radio(
            "資料來源",
            ["內建範例資料", "目前 ready.csv"],
            horizontal=True,
            key=f"{title}_data_source",
        )
        builtin = source == "內建範例資料"
        if builtin:
            df = pd.read_csv(builtin_path)
            source_label = builtin_label
            st.success("目前使用本頁內建教學資料。")
        else:
            df = load_ready_dataset()
            source_label = f"目前 ready.csv：{_display_path(READY_DATASET_PATH)}"
            if df is None:
                st.warning("尚未建立 ready.csv，或改用內建範例資料。")
                return
            st.info(f"目前使用 `{_display_path(READY_DATASET_PATH)}`。")
        render_dataset_metrics(df)
        render_main(df, source_label, builtin=builtin)
    with side:
        render_chat_panel(
            extra_context=str(st.session_state.get(context_key, f"目前頁面：{title}。")),
            page_name=title,
        )


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.select_dtypes(include="number").columns]


def _default_column(columns: list[str], preferred: str, *, exclude: set[str] | None = None) -> str:
    exclude = exclude or set()
    if preferred in columns and preferred not in exclude:
        return preferred
    for column in columns:
        if column not in exclude:
            return column
    return columns[0]


def _classification_training_frame(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    columns = list(features) + [target]
    return df[columns].apply(pd.to_numeric, errors="coerce").dropna()


def _prepare_logistic_features(
    working: pd.DataFrame,
    features: list[str],
    *,
    builtin: bool,
) -> tuple[pd.DataFrame, dict | None]:
    if builtin:
        return working[features], None
    scaler = create_standard_scaler(working, features)
    return apply_standard_scaler(working, scaler), scaler


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
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("資料預覽", expanded=True):
        st.dataframe(frame[features + [target]].head(10), use_container_width=True, hide_index=True)


def _render_logistic_cost_formula(*, regularized: bool = False) -> None:
    with st.expander("成本函數 J(w,b)", expanded=False):
        if regularized:
            st.latex(
                r"J=-\frac{1}{m}\sum_i loss + \frac{\lambda}{2m}\sum_j w_j^2"
            )
        else:
            st.latex(
                r"J=-\frac{1}{m}\sum_i\Big[y^{(i)}\log f^{(i)}+(1-y^{(i)})\log(1-f^{(i)})\Big]"
            )
        st.caption("Cost 只依 sigmoid 機率 f 計算，與分類 threshold 無關。")


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


def _animate_logistic_proba(
    actual: pd.Series,
    feature_matrix: pd.DataFrame,
    steps: list[GradientDescentStep],
    plot_placeholder,
    cost_placeholder,
    status_placeholder,
) -> None:
    for step in _animation_steps(steps):
        probability = predict_proba(feature_matrix, step.weights, step.intercept)
        _render_actual_probability_plot(actual, probability, plot_placeholder)
        _render_cost_history_plot(steps[: step.iteration + 1], cost_placeholder)
        status_placeholder.caption(
            f"Iteration {step.iteration:,} / {steps[-1].iteration:,}，Cost J = {step.cost:.4f}"
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
    if scaler is not None:
        scaled = apply_standard_scaler(plot_frame, scaler)
        w1, w2 = step.weights[0], step.weights[1]
        x1 = scaled[x1_name]
        x2 = scaled[x2_name]
    else:
        w1, w2 = step.weights[0], step.weights[1]
        x1 = plot_frame[x1_name]
        x2 = plot_frame[x2_name]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    positives = frame[target] == 1
    negatives = frame[target] == 0
    ax.scatter(x1[negatives], x2[negatives], marker="o", facecolors="none", label="y=0")
    ax.scatter(x1[positives], x2[positives], marker="x", label="y=1")
    if abs(w2) > 1e-12:
        line_x = np.linspace(float(x1.min()), float(x1.max()), 100)
        line_y = -(w1 * line_x + step.intercept) / w2
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
    ax.scatter(
        frame.loc[negatives, x1_name],
        frame.loc[negatives, x2_name],
        marker="o",
        facecolors="none",
        label="y=0",
    )
    ax.scatter(
        frame.loc[positives, x1_name],
        frame.loc[positives, x2_name],
        marker="x",
        label="y=1",
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


def _render_actual_probability_plot(
    actual: pd.Series,
    probability: pd.Series,
    placeholder,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    ax.scatter(actual, probability, alpha=0.75)
    ax.set_xlabel("實際 y")
    ax.set_ylabel("預測機率 f(x)")
    ax.set_title("實際標籤 vs 預測機率")
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
    st.dataframe(result.head(30).style.format({"probability": "{:.4f}"}), use_container_width=True)


def _render_classification_save_section(
    artifact: ClassificationArtifact,
    filename_prefix: str,
    page_key: str,
) -> None:
    st.markdown("##### 保存模型 JSON")
    st.caption("檔案保存至 `dataset_streamlit_shell/workspace/models/classification/`。")
    if st.button(
        "保存模型 JSON",
        type="primary",
        use_container_width=True,
        key=f"save_{page_key}",
    ):
        CLASSIFICATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CLASSIFICATION_MODEL_DIR / f"{filename_prefix}_{stamp}.json"
        save_classification_artifact(artifact, path)
        st.success(f"已保存模型：`{_display_path(path)}`")


def _render_logistic_inference_section(
    *,
    page_key: str,
    trained_artifact: LogisticModelArtifact | None,
    threshold: float,
) -> None:
    st.markdown("##### 手動預測")
    st.caption("上傳的 JSON 必須為 `logistic_regression`。")
    active = _resolve_classification_artifact(
        page_key=page_key,
        trained_artifact=trained_artifact,
        expected_kind=MODEL_KIND_LOGISTIC,
    )
    if active is None:
        return
    input_values: dict[str, float] = {}
    cols = st.columns(min(len(active.features), 3) or 1)
    for index, feature in enumerate(active.features):
        with cols[index % len(cols)]:
            default_value = 0.0
            if active.scaler is not None:
                default_value = float(active.scaler["mean"].get(feature, 0.0))
            input_values[feature] = st.number_input(feature, value=default_value, key=f"{page_key}_{feature}")
    if st.button("計算預測", type="primary", key=f"{page_key}_predict"):
        frame = pd.DataFrame([input_values])
        prob = float(predict_proba_from_logistic_artifact(active, frame).iloc[0])
        pred_class = int(prob >= threshold)
        st.metric("預測機率", f"{prob:.4f}")
        st.metric("預測類別", str(pred_class))


def _render_regularized_inference_section(
    *,
    page_key: str,
    trained_artifact: RegularizedLogisticModelArtifact | None,
    threshold: float,
) -> None:
    st.markdown("##### 手動預測")
    st.caption("上傳的 JSON 必須為 `regularized_logistic_regression`；請輸入 2 個原始 features。")
    active = _resolve_classification_artifact(
        page_key=page_key,
        trained_artifact=trained_artifact,
        expected_kind=MODEL_KIND_REGULARIZED,
    )
    if active is None or not isinstance(active, RegularizedLogisticModelArtifact):
        return
    input_values: dict[str, float] = {}
    cols = st.columns(2)
    for index, feature in enumerate(active.base_features):
        with cols[index]:
            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                key=f"{page_key}_{feature}",
            )
    if st.button("計算預測", type="primary", key=f"{page_key}_predict"):
        frame = pd.DataFrame([input_values])
        prob = float(predict_proba_from_regularized_artifact(active, frame).iloc[0])
        pred_class = int(prob >= threshold)
        st.metric("預測機率", f"{prob:.4f}")
        st.metric("預測類別", str(pred_class))


def _resolve_classification_artifact(
    *,
    page_key: str,
    trained_artifact: ClassificationArtifact | None,
    expected_kind: str,
) -> ClassificationArtifact | None:
    options: list[str] = []
    if trained_artifact is not None:
        options.append("本次訓練結果")
    options.append("上傳模型 JSON")
    source = options[0] if len(options) == 1 else st.radio(
        "預測使用的模型",
        options,
        horizontal=True,
        key=f"{page_key}_inference_source",
    )
    if source == "本次訓練結果" and trained_artifact is not None:
        if trained_artifact.model_kind != expected_kind:
            st.error("本次訓練模型類型與此頁不符。")
            return None
        return trained_artifact
    uploaded = st.file_uploader("上傳模型 JSON", type=["json"], key=f"{page_key}_upload")
    if uploaded is None:
        st.info("請上傳先前保存的模型 JSON。")
        return None
    try:
        artifact = artifact_from_payload(
            json.loads(uploaded.getvalue().decode("utf-8")),
            expected_kind=expected_kind,
        )
        st.success(f"已載入模型：{artifact.model_kind}")
        return artifact
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        st.error(f"無法讀取模型 JSON：{exc}")
        return None


def _render_classification_prompts(prompts: list[str]) -> None:
    st.markdown("##### 建議問 Agent")
    for prompt in prompts:
        st.code(prompt, language="text")
