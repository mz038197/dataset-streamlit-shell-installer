from __future__ import annotations

import json
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
from dataset_streamlit_shell.ml.classification import validate_binary_target
from dataset_streamlit_shell.ml.regression import (
    apply_standard_scaler,
    create_standard_scaler,
)
from dataset_streamlit_shell.ml.svm import (
    MODEL_KIND_LINEAR_SVM,
    LinearSvmArtifact,
    artifact_from_payload,
    build_linear_svm_artifact,
    build_svm_agent_context,
    decision_function_from_artifact,
    fit_linear_svc,
    predict_class_from_artifact,
    save_svm_artifact,
)
from dataset_streamlit_shell.plotting import (
    build_classification_data_figures,
    build_linear_svm_result_figure,
    build_svm_paired_data_figure,
    configure_matplotlib_for_traditional_chinese,
    render_figures_in_streamlit,
)

configure_matplotlib_for_traditional_chinese()

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
CLASSIFICATION_MODEL_DIR = SHELL_ROOT / "workspace" / "models" / "classification"
SVM_BLOBS_PATH = CLASSIFICATION_DEMO_DIR / "svm_blobs_80.csv"

SVM_FEATURES = ["特徵1", "特徵2"]
SVM_TARGET = "類別"


def render_linear_svm_page() -> None:
    def body(df: pd.DataFrame, source_label: str, *, builtin: bool) -> None:
        st.markdown("##### 線性 SVM")
        st.info(
            "使用 scikit-learn 的 `SVC(kernel='linear')` 擬合二元分類；"
            "訓練完成後顯示決策邊界與 support vectors（對齊《用 Python 學 AI》p53）。"
        )
        if builtin:
            features = list(SVM_FEATURES)
            target = SVM_TARGET
            working = _training_frame(df, features, target)
        else:
            numeric_columns = _numeric_columns(df)
            if len(numeric_columns) < 2:
                st.warning("至少需要 1 個 target 與 1 個 feature。")
                return
            default_target = _default_column(numeric_columns, SVM_TARGET)
            target = st.selectbox(
                "選擇 target（y，0/1）",
                numeric_columns,
                index=numeric_columns.index(default_target)
                if default_target in numeric_columns
                else 0,
                key="svm_target",
            )
            if not validate_binary_target(df[target]):
                st.warning("target 必須為 0/1。請在編碼頁先整理，或改用內建範例資料。")
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
            if not features:
                st.warning("請至少選擇 1 個 feature。")
                return
            working = _training_frame(df, features, target)
            if len(working) < 2:
                st.warning("可用樣本少於 2 筆，無法訓練。")
                return

        _render_svm_data_intro(working, features=features, target=target, builtin=builtin)
        feature_matrix, scaler = _prepare_svm_features(working, features, builtin=builtin)

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
        st.markdown("##### 模型")
        st.code("SVC(kernel='linear', C=...)", language="python")

        result_key = "linear_svm_last_artifact"
        context_key = "線性 SVM_agent_context"
        signature = (
            source_label,
            tuple(features),
            target,
            float(C),
            len(working),
            builtin,
        )
        can_plot_2d = len(features) == 2
        if not can_plot_2d:
            st.caption("目前選超過 2 個 features，訓練後無法繪製 2D 決策邊界圖。")

        train_clicked = st.button(
            "開始訓練",
            type="primary",
            use_container_width=True,
            key="train_linear_svm",
        )
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
                with st.expander("與課本相同的 5×5 採樣格點示意", expanded=False):
                    st.caption("教材以資料極值各取 5 個等分點建立 meshgrid。")
                    demo_fig, demo_ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
                    x1 = working[features[0]].to_numpy(dtype=float)
                    x2 = working[features[1]].to_numpy(dtype=float)
                    labels = working[target].to_numpy(dtype=float)
                    demo_ax.scatter(x1, x2, c=labels, cmap=plt.cm.Paired)
                    sample_x = np.linspace(np.min(x1), np.max(x1), 5)
                    sample_y = np.linspace(np.min(x2), np.max(x2), 5)
                    grid_x, grid_y = np.meshgrid(sample_x, sample_y)
                    demo_ax.scatter(grid_x, grid_y, c="k", s=20)
                    demo_ax.set_title("5×5 採樣格點（課本示意）")
                    st.pyplot(demo_fig, clear_figure=True)
                    plt.close(demo_fig)
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

    _svm_page_shell(
        "線性 SVM",
        "使用內建 make_blobs 教學資料或 ready.csv，練習線性支持向量分類。",
        "內建範例資料：make_blobs（random_state=7, n=80）",
        SVM_BLOBS_PATH,
        lambda df, label, builtin: body(df, label, builtin=builtin),
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


def _training_frame(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    columns = features + [target]
    frame = df[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().reset_index(drop=True)


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
) -> None:
    st.markdown("##### Data 資訊")
    note = (
        "每一列是一個二維樣本：兩個特徵為 x，類別為 y（0/1）。"
        "內建資料由 `make_blobs(n_samples=80, centers=2, random_state=7)` 產生。"
    )
    st.info(note)
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
    st.markdown("##### 資料視覺化")
    if builtin and len(features) == 2:
        st.caption("特徵空間分佈（Paired 色圖）")
        fig = build_svm_paired_data_figure(frame, features, target)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    else:
        render_figures_in_streamlit(build_classification_data_figures(frame, features, target))


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
    preview = pd.DataFrame(
        {
            "actual": working[target],
            "decision_function": scores,
            "predicted_class": predicted,
        }
    )
    st.dataframe(
        preview.head(30).style.format({"decision_function": "{:.4f}"}),
        use_container_width=True,
    )


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
    st.caption("上傳的 JSON 必須為 `linear_svm`。")
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
            input_values[feature] = st.number_input(
                feature,
                value=default_value,
                key=f"svm_{feature}",
            )
    if st.button("計算預測", type="primary", key="svm_predict"):
        frame = pd.DataFrame([input_values])
        score = float(decision_function_from_artifact(active, frame)[0])
        pred_class = int(score >= 0)
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


def _render_svm_prompts(prompts: list[str]) -> None:
    st.markdown("##### 建議問 Agent")
    for prompt in prompts:
        st.code(prompt, language="text")
