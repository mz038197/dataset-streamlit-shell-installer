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
from dataset_streamlit_shell.ml.regression import (
    GradientDescentStep,
    apply_standard_scaler,
    create_standard_scaler,
)
from dataset_streamlit_shell.ml.svm import (
    MODEL_KIND_LINEAR_SVM,
    LinearSvmArtifact,
    artifact_from_payload,
    build_linear_svm_artifact,
    build_svm_agent_context,
    compute_hinge_loss,
    decision_function_from_artifact,
    fit_linear_svc,
    linear_svm_gradient_descent_steps,
    predict_binary_class,
    predict_class_from_artifact,
    save_svm_artifact,
    support_vector_candidates,
    validate_svm_target,
)
from dataset_streamlit_shell.plotting import (
    build_classification_data_figures,
    build_linear_svm_result_figure,
    build_svm_paired_data_figure,
    configure_matplotlib_for_traditional_chinese,
    plot_linear_svm_hyperplanes,
    render_figures_in_streamlit,
)

configure_matplotlib_for_traditional_chinese()

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
CLASSIFICATION_MODEL_DIR = SHELL_ROOT / "workspace" / "models" / "classification"
SVM_BLOBS_PATH = CLASSIFICATION_DEMO_DIR / "svm_blobs_80.csv"

SVM_FEATURES = ["特徵1", "特徵2"]
SVM_TARGET = "類別"
SVM_DEMO_FEATURES = ["x1", "x2"]
SVM_DEMO_TARGET = "y"


def render_linear_svm_page() -> None:
    def body(df: pd.DataFrame, source_label: str, *, builtin: bool) -> None:
        st.markdown("##### 線性 SVM")
        mode = st.radio(
            "模式",
            ["標準 SVM（sklearn）", "教學示意（手寫 hinge loss 更新）"],
            horizontal=True,
            key="svm_mode",
        )
        if mode == "教學示意（手寫 hinge loss 更新）":
            demo_frame = _svm_teaching_demo_frame()
            st.info("教學示意模式固定使用 6 個二維樣本點，不使用目前頁面的資料來源或 ready.csv。")
            _render_svm_data_intro(
                demo_frame,
                features=SVM_DEMO_FEATURES,
                target=SVM_DEMO_TARGET,
                builtin=True,
                note="固定示意資料：3 個 -1 類別樣本與 3 個 +1 類別樣本。",
            )
            _render_teaching_mode(
                working=demo_frame,
                feature_matrix=demo_frame[SVM_DEMO_FEATURES],
                features=SVM_DEMO_FEATURES,
                target=SVM_DEMO_TARGET,
                scaler=None,
                source_label="固定 6 點教學示意資料",
                builtin=True,
            )
            return

        if builtin:
            features = list(SVM_FEATURES)
            target = SVM_TARGET
            working = _svm_training_frame(df, features, target, builtin=builtin)
        else:
            numeric_columns = _numeric_columns(df)
            if len(numeric_columns) < 3:
                st.warning("至少需要 2 個 features 與 1 個 target。")
                return
            default_target = _default_column(numeric_columns, SVM_TARGET)
            target = st.selectbox(
                "選擇 target（y，-1/+1）",
                numeric_columns,
                index=numeric_columns.index(default_target) if default_target in numeric_columns else 0,
                key="svm_target",
            )
            if not validate_svm_target(df[target]):
                st.warning("target 必須剛好包含 -1 與 +1。請先在前處理頁完成轉碼。")
                return
            feature_options = [column for column in numeric_columns if column != target]
            default_features = [column for column in SVM_FEATURES if column in feature_options]
            if not default_features:
                default_features = feature_options[: min(2, len(feature_options))]
            features = st.multiselect(
                "選擇 features（x）",
                feature_options,
                default=default_features,
                key="svm_features",
            )
            if len(features) < 1:
                st.warning("請至少選擇 1 個 feature。")
                return
            working = _svm_training_frame(df, features, target, builtin=builtin)

        if len(working) < 2:
            st.warning("可用樣本少於 2 筆，無法訓練。")
            return

        _render_svm_data_intro(working, features=features, target=target, builtin=builtin)
        feature_matrix, scaler = _prepare_svm_features(working, features, builtin=builtin)

        _render_sklearn_mode(
            working=working,
            feature_matrix=feature_matrix,
            features=features,
            target=target,
            scaler=scaler,
            source_label=source_label,
            builtin=builtin,
        )

    _svm_page_shell(
        "線性 SVM",
        "使用內建範例資料或 ready.csv，練習線性支持向量分類。",
        "內建範例資料：兩特徵二元分類（80 筆）",
        SVM_BLOBS_PATH,
        lambda df, label, builtin: body(df, label, builtin=builtin),
    )


def _render_sklearn_mode(
    *,
    working: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    features: list[str],
    target: str,
    scaler: dict | None,
    source_label: str,
    builtin: bool,
) -> None:
    st.info(
        "這個模式使用 scikit-learn 的 `SVC(kernel='linear')`。"
        "資料標記在本頁統一為 -1 / +1，圖上的 support vectors 為求解器回傳的正式結果。"
    )
    st.markdown("##### 訓練設定")
    C = st.number_input(
        "懲罰係數 C",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        key="svm_C",
    )
    st.markdown("##### 模型公式")
    st.latex(r"f_{\mathbf{w},b}(\mathbf{x})=\mathbf{w}\cdot\mathbf{x}+b")
    st.caption("預測類別由 f(x) 的正負決定：f(x) ≥ 0 判為 +1，f(x) < 0 判為 -1。")
    _render_svm_loss_formula()

    result_key = "linear_svm_last_artifact"
    context_key = "線性 SVM_agent_context"
    signature = (source_label, tuple(features), target, float(C), len(working), builtin, "sklearn")
    can_plot_2d = len(features) == 2
    if not can_plot_2d:
        st.caption("目前選超過 2 個 features，訓練後無法繪製 2D 決策邊界圖。")

    train_clicked = st.button("開始訓練", type="primary", use_container_width=True, key="train_linear_svm")
    artifact: LinearSvmArtifact | None = None
    if train_clicked:
        try:
            clf = fit_linear_svc(feature_matrix, working[target], C=float(C))
        except ValueError as exc:
            st.error(str(exc))
            return
        artifact = build_linear_svm_artifact(
            clf,
            features=list(features),
            target=target,
            C=float(C),
            scaler=scaler,
            data_source=source_label,
            feature_frame=feature_matrix,
            target_series=working[target],
        )
        st.session_state[result_key] = {"signature": signature, "artifact": artifact}
        if can_plot_2d:
            st.markdown("##### 訓練結果圖")
            fig = build_linear_svm_result_figure(
                working,
                features,
                target,
                coef=artifact.coef,
                intercept=artifact.intercept,
                support_vectors=artifact.support_vectors,
                paired_scatter=builtin,
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
    else:
        stored = st.session_state.get(result_key)
        if isinstance(stored, dict) and stored.get("signature") == signature:
            artifact = stored["artifact"]
            st.caption("顯示最近一次訓練結果；調整 C 後請重新按「開始訓練」。")
            if can_plot_2d:
                st.markdown("##### 訓練結果圖")
                fig = build_linear_svm_result_figure(
                    working,
                    features,
                    target,
                    coef=artifact.coef,
                    intercept=artifact.intercept,
                    support_vectors=artifact.support_vectors,
                    paired_scatter=builtin,
                )
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
        else:
            st.info("設定 C 後，按下「開始訓練」以顯示決策邊界與 support vectors。")

    st.session_state[context_key] = build_svm_agent_context(
        page_name="線性 SVM",
        data_source=source_label,
        features=features,
        target=target,
        C=float(C),
        row_count=len(working),
        artifact=artifact,
        note="目前模式：標準 SVM（sklearn）。",
    )
    if artifact is not None:
        _render_svm_training_results(artifact, working, target)
        _render_svm_save_section(artifact)
    _render_svm_inference_section(trained_artifact=artifact)
    _render_svm_prompts(
        [
            "請解釋 support vector 在這張圖上代表什麼。",
            "若 C 變大或變小，決策邊界與 margin 可能如何改變？",
            "請比較線性 SVM 與邏輯迴歸在這份資料上的差異。",
        ]
    )


def _render_teaching_mode(
    *,
    working: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    features: list[str],
    target: str,
    scaler: dict | None,
    source_label: str,
    builtin: bool,
) -> None:
    st.warning(
        "這個模式是教學示意。它用簡化的 hinge loss 逐筆更新來幫助理解 margin 與邊界移動，"
        "不是標準 SVM 求解器；圖上只標示 support vector candidates。"
    )
    if len(features) != 2:
        st.info("教學示意模式需要恰好 2 個 features 才能顯示決策邊界與 margin。")
        return

    st.markdown("##### 訓練設定")
    c1, c2, c3, c4 = st.columns(4)
    learning_rate = c1.number_input(
        "學習率 α",
        min_value=0.0001,
        max_value=1.0,
        value=0.001,
        step=0.0005,
        format="%.4f",
        key="svm_demo_lr",
    )
    C = c2.number_input(
        "懲罰係數 C",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        key="svm_demo_C",
    )
    epochs = c3.number_input(
        "Epoch / 迭代次數",
        min_value=1,
        max_value=5000,
        value=200,
        step=10,
        key="svm_demo_epochs",
    )
    update_every = c4.number_input(
        "更新步長",
        min_value=1,
        max_value=200,
        value=10,
        step=1,
        key="svm_demo_update_every",
    )
    st.markdown("##### 模型公式")
    st.latex(r"y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)}+b)\ge 1")
    _render_svm_loss_formula(teaching=True)

    result_key = "linear_svm_demo_last_result"
    context_key = "線性 SVM_agent_context"
    signature = (
        source_label,
        tuple(features),
        target,
        float(learning_rate),
        float(C),
        int(epochs),
        len(working),
        builtin,
        "demo",
    )
    train_clicked = st.button(
        "開始示意訓練",
        type="primary",
        use_container_width=True,
        key="train_linear_svm_demo",
    )
    final_step: GradientDescentStep | None = None
    candidate_mask: np.ndarray | None = None
    if train_clicked:
        steps = linear_svm_gradient_descent_steps(
            feature_matrix,
            working[target],
            learning_rate=float(learning_rate),
            C=float(C),
            epochs=int(epochs),
        )
        chart_left, chart_right = st.columns(2)
        boundary_placeholder = chart_left.empty()
        cost_placeholder = chart_right.empty()
        status_placeholder = st.empty()
        _animate_teaching_svm(
            working,
            feature_matrix,
            features,
            target,
            steps,
            boundary_placeholder,
            cost_placeholder,
            status_placeholder,
            update_every=int(update_every),
            scaler=scaler,
        )
        final_step = steps[-1]
        candidate_mask = support_vector_candidates(
            feature_matrix,
            working[target],
            final_step.weights,
            final_step.intercept,
        )
        st.session_state[result_key] = {
            "signature": signature,
            "step": final_step,
            "candidate_mask": candidate_mask.tolist(),
        }
    else:
        stored = st.session_state.get(result_key)
        if isinstance(stored, dict) and stored.get("signature") == signature:
            final_step = stored["step"]
            candidate_mask = np.asarray(stored["candidate_mask"], dtype=bool)
            st.caption("顯示最近一次示意訓練結果；調整設定後請重新按「開始示意訓練」。")
            fig = _build_teaching_svm_figure(
                working,
                feature_matrix,
                features,
                target,
                final_step,
                scaler=scaler,
                candidate_mask=candidate_mask,
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.info("設定 α、C、epoch 後，按下「開始示意訓練」觀察邊界與 hinge loss 的演進。")

    st.session_state[context_key] = build_svm_agent_context(
        page_name="線性 SVM",
        data_source=source_label,
        features=features,
        target=target,
        C=float(C),
        row_count=len(working),
        artifact=None,
        note="目前模式：教學示意（手寫 hinge loss 更新）。",
    )
    if final_step is not None and candidate_mask is not None:
        _render_teaching_results(
            working,
            feature_matrix,
            target,
            final_step,
            C=float(C),
            candidate_mask=candidate_mask,
        )
    _render_svm_prompts(
        [
            "請解釋為什麼這個模式只叫教學示意，而不是正式 SVM 求解器。",
            "margin 條件 y(wx+b) >= 1 在這組資料上代表什麼？",
            "哪些樣本被標成 support vector candidates，原因是什麼？",
        ]
    )


def _svm_page_shell(
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


def _default_column(columns: list[str], preferred: str) -> str:
    return preferred if preferred in columns else columns[0]


def _svm_training_frame(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    builtin: bool,
) -> pd.DataFrame:
    columns = features + [target]
    frame = df[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().reset_index(drop=True)
    if builtin:
        frame[target] = np.where(frame[target].to_numpy(dtype=int) == 1, 1, -1)
    return frame


def _prepare_svm_features(
    working: pd.DataFrame,
    features: list[str],
    *,
    builtin: bool,
) -> tuple[pd.DataFrame, dict | None]:
    if builtin:
        return working[features], None
    scaler = create_standard_scaler(working, features)
    return apply_standard_scaler(working, scaler), scaler


def _render_svm_data_intro(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    builtin: bool,
    note: str | None = None,
) -> None:
    st.markdown("##### Data 資訊")
    st.info(note or "每一列是一個樣本：特徵為 x，類別為 y（本頁固定使用 -1 / +1）。")
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
        pd.DataFrame(role_rows).style.format({"最小值": "{:.4f}", "最大值": "{:.4f}", "平均值": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("資料預覽", expanded=True):
        st.dataframe(frame[features + [target]].head(10), use_container_width=True, hide_index=True)
    st.markdown("##### 資料視覺化")
    if builtin and len(features) == 2:
        st.caption("特徵空間分佈（Paired 色圖）")
        fig = build_svm_paired_data_figure(frame, features, target)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    else:
        render_figures_in_streamlit(build_classification_data_figures(_classification_view(frame, target), features, target))


def _classification_view(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    view = frame.copy()
    view[target] = np.where(view[target].to_numpy(dtype=int) == 1, 1, 0)
    return view


def _render_svm_training_results(
    artifact: LinearSvmArtifact,
    working: pd.DataFrame,
    target: str,
) -> None:
    st.markdown("##### 訓練結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("intercept", f"{artifact.intercept:.4f}")
    c2.metric("Support vectors", str(artifact.n_support))
    c3.metric("訓練集正確率", f"{artifact.training_accuracy:.2f}%")
    scores = decision_function_from_artifact(artifact, working)
    predicted = predict_class_from_artifact(artifact, working)
    preview = pd.DataFrame({"actual": working[target], "decision_function": scores, "predicted_class": predicted})
    st.dataframe(preview.head(30).style.format({"decision_function": "{:.4f}"}), use_container_width=True)


def _render_svm_save_section(artifact: LinearSvmArtifact) -> None:
    st.markdown("##### 保存模型 JSON")
    st.caption("檔案保存至 `dataset_streamlit_shell/workspace/models/classification/`。")
    if st.button("保存模型 JSON", type="primary", use_container_width=True, key="save_svm"):
        CLASSIFICATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CLASSIFICATION_MODEL_DIR / f"linear_svm_{stamp}.json"
        save_svm_artifact(artifact, path)
        st.success(f"已保存模型：`{_display_path(path)}`")


def _render_svm_inference_section(*, trained_artifact: LinearSvmArtifact | None) -> None:
    st.markdown("##### 手動預測")
    st.caption("上傳的 JSON 必須為 `linear_svm`。輸出類別固定為 -1 / +1。")
    active = _resolve_svm_artifact(trained_artifact=trained_artifact)
    if active is None:
        return
    input_values: dict[str, float] = {}
    cols = st.columns(min(len(active.features), 3) or 1)
    for index, feature in enumerate(active.features):
        with cols[index % len(cols)]:
            default_value = 0.0
            if active.scaler is not None:
                default_value = float(active.scaler["mean"].get(feature, 0.0))
            input_values[feature] = st.number_input(feature, value=default_value, key=f"svm_{feature}")
    if st.button("計算預測", type="primary", key="svm_predict"):
        frame = pd.DataFrame([input_values])
        score = float(decision_function_from_artifact(active, frame)[0])
        pred_class = int(predict_binary_class(np.array([score]))[0])
        st.metric("decision function", f"{score:.4f}")
        st.metric("預測類別", str(pred_class))


def _resolve_svm_artifact(*, trained_artifact: LinearSvmArtifact | None) -> LinearSvmArtifact | None:
    options: list[str] = []
    if trained_artifact is not None:
        options.append("本次訓練結果")
    options.append("上傳模型 JSON")
    source = options[0] if len(options) == 1 else st.radio(
        "預測使用的模型",
        options,
        horizontal=True,
        key="svm_inference_source",
    )
    if source == "本次訓練結果" and trained_artifact is not None:
        if trained_artifact.model_kind != MODEL_KIND_LINEAR_SVM:
            st.error("本次訓練模型類型與此頁不符。")
            return None
        return trained_artifact
    uploaded = st.file_uploader("上傳模型 JSON", type=["json"], key="svm_upload")
    if uploaded is None:
        st.info("請上傳先前保存的模型 JSON。")
        return None
    try:
        artifact = artifact_from_payload(
            json.loads(uploaded.getvalue().decode("utf-8")),
            expected_kind=MODEL_KIND_LINEAR_SVM,
        )
        st.success(f"已載入模型：{artifact.model_kind}")
        return artifact
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        st.error(f"無法讀取模型 JSON：{exc}")
        return None


def _render_svm_loss_formula(*, teaching: bool = False) -> None:
    with st.expander("目標函數 Loss(w,b)", expanded=False):
        st.latex(
            r"\mathrm{Loss}=\frac{1}{2}\|\mathbf{w}\|^2"
            r"+ C\sum_i \max\bigl(0,\,1-y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)}+b)\bigr)"
        )
        if teaching:
            st.caption("教學示意模式以逐筆更新近似這個目標函數，用來理解 margin，並非標準 SVM 求解器。")
        else:
            st.caption("前半項縮小 ‖w‖ 以拉大 margin；後半項為 hinge loss。C 越大，越重視分對樣本。")


def _render_svm_prompts(prompts: list[str]) -> None:
    st.markdown("##### 建議問 Agent")
    for prompt in prompts:
        st.code(prompt, language="text")


def _animate_teaching_svm(
    frame: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    features: list[str],
    target: str,
    steps: list[GradientDescentStep],
    boundary_placeholder,
    cost_placeholder,
    status_placeholder,
    *,
    update_every: int,
    scaler: dict | None,
) -> None:
    for step in _sample_svm_steps(steps, update_every=update_every):
        candidate_mask = support_vector_candidates(feature_matrix, frame[target], step.weights, step.intercept)
        fig = _build_teaching_svm_figure(
            frame,
            feature_matrix,
            features,
            target,
            step,
            scaler=scaler,
            candidate_mask=candidate_mask,
        )
        boundary_placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)
        _render_svm_cost_history_plot(steps[: step.iteration + 1], cost_placeholder)
        status_placeholder.caption(
            f"Iteration {step.iteration:,} / {steps[-1].iteration:,}，"
            f"Loss = {step.cost:.4f}，w = [{step.weights[0]:.3f}, {step.weights[1]:.3f}]，b = {step.intercept:.3f}"
        )
        time.sleep(0.02)


def _sample_svm_steps(steps: list[GradientDescentStep], *, update_every: int) -> list[GradientDescentStep]:
    if len(steps) <= 80:
        return steps
    stride = max(int(update_every), len(steps) // 80, 1)
    selected = steps[::stride]
    if selected[-1] != steps[-1]:
        selected.append(steps[-1])
    return selected


def _build_teaching_svm_figure(
    frame: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    features: list[str],
    target: str,
    step: GradientDescentStep,
    *,
    scaler: dict | None,
    candidate_mask: np.ndarray,
):
    x1_name, x2_name = features
    if scaler is not None:
        plot_frame = apply_standard_scaler(frame[features], scaler)
        x1_label = f"{x1_name}（scaled）"
        x2_label = f"{x2_name}（scaled）"
    else:
        plot_frame = frame[features]
        x1_label = x1_name
        x2_label = x2_name
    x1 = plot_frame[x1_name].to_numpy(dtype=float)
    x2 = plot_frame[x2_name].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=int)
    positives = y == 1
    negatives = y == -1

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(x1[negatives], x2[negatives], label="-1", c="#f4b400", edgecolors="#5f4330", linewidths=0.6)
    ax.scatter(x1[positives], x2[positives], label="+1", c="#202124", marker="x", linewidths=1.2)

    weights = np.asarray(step.weights, dtype=float)
    plot_linear_svm_hyperplanes(ax, weights, step.intercept, x1, x2)

    if np.any(candidate_mask):
        ax.scatter(
            x1[candidate_mask],
            x2[candidate_mask],
            s=140,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            label="SV candidates",
            zorder=4,
        )

    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)
    ax.set_title(f"教學示意 SVM（iteration {step.iteration}）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig


def _render_svm_cost_history_plot(steps: list[GradientDescentStep], placeholder) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    ax.plot([step.iteration for step in steps], [step.cost for step in steps], color="#ff7043")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hinge Loss")
    ax.set_title("Loss vs Iteration")
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_teaching_results(
    working: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    target: str,
    step: GradientDescentStep,
    *,
    C: float,
    candidate_mask: np.ndarray,
) -> None:
    st.markdown("##### 示意訓練結果")
    scores = feature_matrix.to_numpy(dtype=float) @ np.asarray(step.weights, dtype=float) + float(step.intercept)
    predicted = predict_binary_class(scores)
    result = pd.DataFrame(
        {
            "actual": working[target],
            "decision_function": scores,
            "predicted_class": predicted,
            "sv_candidate": candidate_mask,
        }
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("最後 b", f"{step.intercept:.4f}")
    c2.metric("Support vector candidates", str(int(np.sum(candidate_mask))))
    c3.metric("最後 hinge loss", f"{compute_hinge_loss(feature_matrix, working[target], step.weights, step.intercept, C=C):.4f}")
    st.dataframe(result.head(30).style.format({"decision_function": "{:.4f}"}), use_container_width=True)


def _svm_teaching_demo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SVM_DEMO_FEATURES[0]: [1.0, 2.0, 2.0, 6.0, 7.0, 8.0],
            SVM_DEMO_FEATURES[1]: [2.0, 3.0, 1.0, 5.0, 7.0, 6.0],
            SVM_DEMO_TARGET: [-1, -1, -1, 1, 1, 1],
        }
    )
