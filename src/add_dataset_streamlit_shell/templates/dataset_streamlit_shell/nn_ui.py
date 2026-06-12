from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.data_ui import SHELL_ROOT, render_chat_panel, render_dataset_metrics
from dataset_streamlit_shell.ml.coffee_nn import (
    ACTIVATION_CHOICES,
    AXIS_LABELS,
    BUILTIN_DATA_PATH_SUFFIX,
    CompileSpec,
    FEATURE_OPTIONS,
    HiddenLayerSpec,
    LOSS_AUTO,
    LOSS_CHOICES,
    MAX_HIDDEN_LAYERS,
    MAX_UNITS_PER_LAYER,
    NetworkSpec,
    OPTIMIZER_CHOICES,
    OUTPUT_ACTIVATION_CHOICES,
    PARAM_WARN_THRESHOLD,
    TARGET_COLUMN,
    TrainConfig,
    build_nn_agent_context,
    configure_tensorflow_runtime,
    estimate_parameter_count,
    format_model_code,
    lab02_default_compile_spec,
    lab02_default_network_spec,
    load_builtin_frame,
    append_epoch_history,
    build_train_artifacts_from_history,
    fit_one_training_epoch,
    predict_class_labels,
    predict_scores,
    prepare_training_runtime,
    validate_network_spec,
)
from dataset_streamlit_shell.plotting import (
    activation_curve_y,
    build_nn_1d_probability_figure,
    build_single_activation_curve_figure,
    build_nn_data_figures,
    build_nn_decision_boundary_figure,
    build_training_loss_figure,
    configure_matplotlib_for_traditional_chinese,
    linear_svm_data_axis_limits,
    render_figures_in_streamlit,
)

configure_matplotlib_for_traditional_chinese()

BUILTIN_PATH = SHELL_ROOT.joinpath(*BUILTIN_DATA_PATH_SUFFIX)
RESULT_KEY = "nn_last_result"
NN_TRAIN_PROGRESS_KEY = "nn_train_progress"
CONTEXT_KEY = "類神經網路_agent_context"

ACTIVATION_Z_MIN = -5.0
ACTIVATION_Z_MAX = 5.0
ACTIVATION_DEMO_ITEMS: tuple[tuple[str, str, str | None], ...] = (
    ("ReLU", r"f(z) = \max(0,\, z)", None),
    (
        "Leaky ReLU",
        r"f(z) = \begin{cases} z & z > 0 \\ \alpha z & z \le 0 \end{cases}",
        "α 為負斜率係數；TensorFlow 預設 α = 0.01。",
    ),
    ("Sigmoid", r"f(z) = \dfrac{1}{1 + e^{-z}}", None),
    ("Tanh", r"f(z) = \tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}", None),
    ("Linear", r"f(z) = z", "亦稱恆等（identity）活化，不做非線性變換。"),
)


def render_neural_network_page() -> None:
    configure_tensorflow_runtime()
    frame = load_builtin_frame(BUILTIN_PATH)
    tab_activation, tab_train = st.tabs(["活化函數", "神經網路訓練"])
    with tab_activation:
        _render_activation_tab()
    with tab_train:
        _render_training_tab(frame)


def _render_activation_tab() -> None:
    st.markdown("##### 常見活化函數")
    st.caption(
        "觀察 ReLU、Leaky ReLU、Sigmoid、Tanh、Linear 在輸入 z 上的輸出形狀；"
        "展開各函數下方的「公式」可對照數學定義。"
    )
    z_grid = np.linspace(ACTIVATION_Z_MIN, ACTIVATION_Z_MAX, 400)

    for name, latex, note in ACTIVATION_DEMO_ITEMS:
        st.markdown(f"###### {name}")
        leaky_alpha = 0.01
        if name == "Leaky ReLU":
            leaky_alpha = st.slider(
                "Leaky ReLU 的 α（負半軸斜率）",
                min_value=0.01,
                max_value=0.5,
                value=0.01,
                step=0.01,
                key="nn_leaky_relu_alpha",
            )
        values = activation_curve_y(name, z_grid, leaky_alpha=leaky_alpha)
        fig = build_single_activation_curve_figure(name, z_grid, values)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        with st.expander("公式", expanded=False):
            st.latex(latex)
            if note:
                st.caption(note)


def _sanitize_feature_session() -> None:
    stored = st.session_state.get("nn_features")
    if not isinstance(stored, list):
        return
    valid = [feature for feature in stored if feature in FEATURE_OPTIONS]
    if valid != stored:
        st.session_state["nn_features"] = valid or list(FEATURE_OPTIONS)


def _render_training_tab(frame: pd.DataFrame) -> None:
    context_key = CONTEXT_KEY
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title("類神經網路")
        st.caption(
            "使用內建雙特徵二元分類資料，以 TensorFlow Sequential 建立並訓練神經網路。"
        )
        st.success("目前使用本頁內建教學資料。")
        render_dataset_metrics(frame)
        render_figures_in_streamlit(
            build_nn_data_figures(
                frame,
                list(FEATURE_OPTIONS),
                TARGET_COLUMN,
                axis_labels=AXIS_LABELS,
            )
        )

        if st.button("套用 Lab02 預設", key="nn_apply_lab02"):
            _apply_lab02_defaults()

        spec, compile_spec, train_config = _render_network_form(frame)
        st.markdown("##### 模型程式碼預覽")
        st.code(format_model_code(spec, compile_spec), language="python")

        param_count = estimate_parameter_count(spec)
        if param_count > PARAM_WARN_THRESHOLD:
            st.warning(f"參數數量約 {param_count:,}，超過建議上限 {PARAM_WARN_THRESHOLD:,}，訓練可能較慢。")
        else:
            st.caption(f"參數數量約 {param_count:,}。")

        validation_errors = validate_network_spec(spec, frame)
        if validation_errors:
            for message in validation_errors:
                st.error(message)

        signature = _training_signature(spec, compile_spec, train_config, len(frame))
        train_disabled = bool(validation_errors)
        progress = st.session_state.get(NN_TRAIN_PROGRESS_KEY)
        if progress and progress.get("signature") != signature:
            del st.session_state[NN_TRAIN_PROGRESS_KEY]
            progress = None

        if progress and progress["status"] == "running":
            stop_clicked = st.button(
                "停止訓練",
                type="primary",
                use_container_width=True,
                key="stop_neural_network",
            )
            if stop_clicked:
                progress["stop_requested"] = True
                st.rerun()
        else:
            start_clicked = st.button(
                "開始訓練",
                type="primary",
                use_container_width=True,
                key="train_neural_network",
                disabled=train_disabled,
            )
            if start_clicked:
                if _ensure_tensorflow_available():
                    _init_training_progress(
                        frame,
                        spec=spec,
                        compile_spec=compile_spec,
                        train_config=train_config,
                        signature=signature,
                    )
                    st.rerun()

        if progress and progress["status"] == "running":
            _render_live_training_view(progress, frame)
            if progress["stop_requested"] or progress["current_epoch"] >= progress["total_epochs"]:
                _finalize_training_progress(progress, frame, context_key=context_key)
                st.rerun()
            _advance_training_epoch(progress)
            st.rerun()
        elif RESULT_KEY in st.session_state and st.session_state[RESULT_KEY]["signature"] == signature:
            _render_training_results(st.session_state[RESULT_KEY], frame=frame, spec=spec)
        else:
            st.info(
                "設定 epochs 與 learning rate 後，按下「開始訓練」觀察 loss 與決策區域的演進。"
            )

    with side:
        render_chat_panel(page_name="類神經網路")


def _apply_lab02_defaults() -> None:
    spec = lab02_default_network_spec()
    compile_spec = lab02_default_compile_spec()
    st.session_state["nn_features"] = list(spec.input_features)
    st.session_state["nn_hidden_count"] = len(spec.hidden_layers)
    for index, layer in enumerate(spec.hidden_layers, start=1):
        st.session_state[f"nn_hidden_units_{index}"] = layer.units
        st.session_state[f"nn_hidden_activation_{index}"] = layer.activation
    st.session_state["nn_output_units"] = spec.output_units
    st.session_state["nn_output_activation"] = spec.output_activation
    st.session_state["nn_use_norm_layer"] = spec.use_normalization_layer
    st.session_state["nn_loss_choice"] = spec.loss_choice
    st.session_state["nn_optimizer"] = compile_spec.optimizer_name
    st.session_state["nn_learning_rate"] = compile_spec.learning_rate
    st.session_state["nn_epochs"] = 100


def _render_network_form(frame: pd.DataFrame) -> tuple[NetworkSpec, CompileSpec, TrainConfig]:
    st.markdown("##### 輸入與架構")
    _sanitize_feature_session()
    selected_features = st.multiselect(
        "輸入特徵（1～2 個）",
        list(FEATURE_OPTIONS),
        default=list(FEATURE_OPTIONS),
        key="nn_features",
    )
    hidden_count = st.number_input(
        "隱藏層數",
        min_value=0,
        max_value=MAX_HIDDEN_LAYERS,
        value=1,
        step=1,
        key="nn_hidden_count",
    )
    hidden_layers: list[HiddenLayerSpec] = []
    for index in range(1, int(hidden_count) + 1):
        col_units, col_act = st.columns(2)
        units = col_units.number_input(
            f"第 {index} 層神經元數",
            min_value=1,
            max_value=MAX_UNITS_PER_LAYER,
            value=3 if index == 1 else 4,
            step=1,
            key=f"nn_hidden_units_{index}",
        )
        activation = col_act.selectbox(
            f"第 {index} 層活化函數",
            ACTIVATION_CHOICES,
            index=1 if index == 1 else 0,
            key=f"nn_hidden_activation_{index}",
        )
        hidden_layers.append(HiddenLayerSpec(int(units), activation))

    out_col1, out_col2 = st.columns(2)
    output_units = out_col1.number_input(
        "輸出神經元數",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="nn_output_units",
    )
    output_activation_options = _output_activation_options(int(output_units))
    default_out_act = "sigmoid" if int(output_units) == 1 else "softmax"
    current_out_act = st.session_state.get("nn_output_activation", default_out_act)
    if current_out_act not in output_activation_options:
        current_out_act = default_out_act
    output_activation = out_col2.selectbox(
        "輸出活化函數",
        output_activation_options,
        index=output_activation_options.index(current_out_act),
        key="nn_output_activation",
    )

    with st.expander("進階：正規化與 loss", expanded=False):
        use_norm_layer = st.checkbox(
            "在 Sequential 內加入 Normalization 層（與訓練前 adapt 二擇一）",
            value=False,
            key="nn_use_norm_layer",
        )
        if use_norm_layer:
            st.info("已啟用層內 Normalization；訓練前不另做 adapt+transform。")
        else:
            st.caption("預設：訓練前以 Normalization.adapt 正規化特徵，不放入 Sequential。")
        loss_choice = st.selectbox("loss", LOSS_CHOICES, key="nn_loss_choice")
    use_norm_layer = bool(st.session_state.get("nn_use_norm_layer", False))
    loss_choice = st.session_state.get("nn_loss_choice", LOSS_AUTO)

    st.markdown("##### 編譯與訓練")
    c1, c2, c3 = st.columns(3)
    optimizer_name = c1.selectbox("優化器", OPTIMIZER_CHOICES, key="nn_optimizer")
    learning_rate = c2.number_input(
        "learning_rate",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        format="%.4f",
        key="nn_learning_rate",
    )
    epochs = c3.number_input("epochs", min_value=1, max_value=500, value=100, step=1, key="nn_epochs")

    spec = NetworkSpec(
        input_features=tuple(selected_features),
        hidden_layers=tuple(hidden_layers),
        output_units=int(output_units),
        output_activation=output_activation,
        loss_choice=loss_choice,
        use_normalization_layer=use_norm_layer,
    )
    compile_spec = CompileSpec(
        optimizer_name=optimizer_name,
        learning_rate=float(learning_rate),
    )
    train_config = TrainConfig(epochs=int(epochs))
    return spec, compile_spec, train_config


def _output_activation_options(output_units: int) -> tuple[str, ...]:
    if output_units == 1:
        return ACTIVATION_CHOICES
    return OUTPUT_ACTIVATION_CHOICES


def _training_signature(
    spec: NetworkSpec,
    compile_spec: CompileSpec,
    train_config: TrainConfig,
    row_count: int,
) -> tuple:
    hidden = tuple((layer.units, layer.activation) for layer in spec.hidden_layers)
    return (
        spec.input_features,
        hidden,
        spec.output_units,
        spec.output_activation,
        spec.loss_choice,
        spec.use_normalization_layer,
        compile_spec.optimizer_name,
        compile_spec.learning_rate,
        train_config.epochs,
        row_count,
    )


def _ensure_tensorflow_available() -> bool:
    try:
        configure_tensorflow_runtime()
        import tensorflow  # noqa: F401
    except ImportError:
        st.error("找不到 TensorFlow。請重新執行安裝工具以安裝 tensorflow-cpu。")
        return False
    return True


def _scores_to_probabilities(scores: np.ndarray, spec: NetworkSpec) -> np.ndarray:
    if spec.output_units == 1:
        if spec.output_activation == "linear":
            return 1.0 / (1.0 + np.exp(-np.clip(scores.reshape(-1), -500, 500)))
        return scores.reshape(-1)
    if scores.ndim > 1 and scores.shape[1] > 1:
        return scores[:, 1]
    return scores.reshape(-1)


def _build_decision_mesh(
    frame: pd.DataFrame,
    features: list[str],
    *,
    mesh_points: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1_name, x2_name = features[0], features[1]
    x1 = frame[x1_name].to_numpy(dtype=float)
    x2 = frame[x2_name].to_numpy(dtype=float)
    x_lo, x_hi, y_lo, y_hi = linear_svm_data_axis_limits(x1, x2)
    grid_x = np.linspace(x_lo, x_hi, mesh_points)
    grid_y = np.linspace(y_lo, y_hi, mesh_points)
    mesh_xx, mesh_yy = np.meshgrid(grid_x, grid_y)
    grid = np.c_[mesh_xx.ravel(), mesh_yy.ravel()].astype(np.float32)
    return x1, x2, mesh_xx, mesh_yy, grid


def _render_epoch_loss_plot(
    history: dict[str, list[float]],
    *,
    epoch: int,
    container,
) -> None:
    fig = build_training_loss_figure(history, title=f"訓練 loss 曲線（epoch {epoch}）")
    container.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def _render_epoch_boundary_plot(
    *,
    frame: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
    spec: NetworkSpec,
    model,
    feature_normalizer,
    mesh_xx: np.ndarray,
    mesh_yy: np.ndarray,
    grid: np.ndarray,
    epoch: int,
    container,
) -> None:
    x1_name, x2_name = features[0], features[1]
    x1 = frame[x1_name].to_numpy(dtype=float)
    x2 = frame[x2_name].to_numpy(dtype=float)
    mesh_scores = predict_scores(model, grid, spec, feature_normalizer=feature_normalizer)
    prob_grid = _scores_to_probabilities(mesh_scores, spec)
    fig = build_nn_decision_boundary_figure(
        x1,
        x2,
        labels,
        mesh_xx,
        mesh_yy,
        prob_grid,
        x1_label=AXIS_LABELS.get(x1_name, x1_name),
        x2_label=AXIS_LABELS.get(x2_name, x2_name),
        title=f"神經網路決策區域（epoch {epoch}）",
    )
    container.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def _render_epoch_probability_plot(
    *,
    x: np.ndarray,
    labels: np.ndarray,
    spec: NetworkSpec,
    model,
    feature_normalizer,
    feature: str,
    epoch: int,
    container,
) -> None:
    scores = predict_scores(model, x, spec, feature_normalizer=feature_normalizer)
    probs = _scores_to_probabilities(scores, spec)
    x_label = AXIS_LABELS.get(feature, feature)
    fig = build_nn_1d_probability_figure(
        x[:, 0],
        probs,
        labels,
        x_label=x_label,
        title=f"單特徵分類機率曲線（epoch {epoch}）",
    )
    container.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def _init_training_progress(
    frame: pd.DataFrame,
    *,
    spec: NetworkSpec,
    compile_spec: CompileSpec,
    train_config: TrainConfig,
    signature: tuple,
) -> None:
    features = list(spec.input_features)
    x = frame[features].to_numpy(dtype=np.float32)
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    try:
        runtime = prepare_training_runtime(spec, compile_spec, x, y)
    except Exception as exc:
        st.error(f"訓練初始化失敗：{exc}")
        return

    st.session_state[NN_TRAIN_PROGRESS_KEY] = {
        "status": "running",
        "current_epoch": 0,
        "total_epochs": int(train_config.epochs),
        "history": {},
        "model": runtime.model,
        "feature_normalizer": runtime.feature_normalizer,
        "x_fit": runtime.x_fit,
        "y_fit": runtime.y_fit,
        "metric_key": runtime.metric_key,
        "spec": spec,
        "compile_spec": compile_spec,
        "signature": signature,
        "features": features,
        "stop_requested": False,
    }


def _advance_training_epoch(progress: dict) -> None:
    completed = int(progress["current_epoch"])
    try:
        logs = fit_one_training_epoch(
            progress["model"],
            progress["x_fit"],
            progress["y_fit"],
            completed_epochs=completed,
        )
    except Exception as exc:
        st.error(f"訓練失敗：{exc}")
        del st.session_state[NN_TRAIN_PROGRESS_KEY]
        return

    progress["history"] = append_epoch_history(progress["history"], logs)
    progress["current_epoch"] = completed + 1


def _render_live_training_view(progress: dict, frame: pd.DataFrame) -> None:
    epoch = int(progress["current_epoch"])
    if epoch <= 0:
        st.caption("訓練準備中…")
        return

    spec = progress["spec"]
    features = progress["features"]
    labels = frame[TARGET_COLUMN].to_numpy(dtype=float)
    history = progress["history"]
    model = progress["model"]
    feature_normalizer = progress["feature_normalizer"]
    metric_key = progress["metric_key"]
    total_epochs = int(progress["total_epochs"])

    chart_left, chart_right = st.columns(2)
    if len(features) == 2:
        _, _, mesh_xx, mesh_yy, grid = _build_decision_mesh(frame, features)
        _render_epoch_boundary_plot(
            frame=frame,
            features=features,
            labels=labels,
            spec=spec,
            model=model,
            feature_normalizer=feature_normalizer,
            mesh_xx=mesh_xx,
            mesh_yy=mesh_yy,
            grid=grid,
            epoch=epoch,
            container=chart_left,
        )
    elif len(features) == 1:
        x = frame[features].to_numpy(dtype=np.float32)
        _render_epoch_probability_plot(
            x=x,
            labels=labels,
            spec=spec,
            model=model,
            feature_normalizer=feature_normalizer,
            feature=features[0],
            epoch=epoch,
            container=chart_left,
        )

    _render_epoch_loss_plot(history, epoch=epoch, container=chart_right)

    loss_value = history.get("loss", [float("nan")])[-1]
    accuracy_value = history.get(metric_key, [0.0])[-1] * 100.0
    if progress.get("stop_requested"):
        status = f"已於 epoch {epoch:,} 停止"
    elif epoch >= total_epochs:
        status = "訓練完成"
    else:
        status = f"Epoch {epoch:,} / {total_epochs:,}"
    st.caption(f"{status} · loss = {loss_value:.4f} · accuracy = {accuracy_value:.2f}%")


def _finalize_training_progress(
    progress: dict,
    frame: pd.DataFrame,
    *,
    context_key: str,
) -> None:
    spec = progress["spec"]
    compile_spec = progress["compile_spec"]
    signature = progress["signature"]
    features = progress["features"]
    stopped_early = bool(progress.get("stop_requested"))

    artifacts = build_train_artifacts_from_history(
        progress["model"],
        progress["history"],
        progress["feature_normalizer"],
        spec,
        progress["metric_key"],
    )
    st.session_state[RESULT_KEY] = {
        "signature": signature,
        "artifacts": artifacts,
        "spec": spec,
        "compile_spec": compile_spec,
        "features": features,
        "stopped_early": stopped_early,
        "stopped_at_epoch": int(progress["current_epoch"]),
    }
    st.session_state[context_key] = build_nn_agent_context(
        spec=spec,
        compile_spec=compile_spec,
        train_result=artifacts.result,
        row_count=len(frame),
    )
    del st.session_state[NN_TRAIN_PROGRESS_KEY]


def _render_training_results(
    cached: dict,
    *,
    frame: pd.DataFrame,
    spec: NetworkSpec,
) -> None:
    artifacts = cached["artifacts"]
    result = artifacts.result
    features = cached["features"]

    st.markdown("##### 訓練結果")
    if cached.get("stopped_early"):
        st.caption(f"已於 epoch {cached.get('stopped_at_epoch', 0):,} 提早停止。")
    st.metric("最終 loss", f"{result.final_loss:.4f}")
    st.metric("訓練準確率", f"{result.train_accuracy:.2f}%")
    st.caption(f"參數數量：{result.parameter_count:,}")

    loss_fig = build_training_loss_figure(result.history)
    st.pyplot(loss_fig, clear_figure=True)
    plt.close(loss_fig)

    x = frame[features].to_numpy(dtype=np.float32)
    labels = frame[TARGET_COLUMN].to_numpy(dtype=float)
    scores = predict_scores(
        artifacts.model,
        x,
        spec,
        feature_normalizer=artifacts.feature_normalizer,
    )
    predicted = predict_class_labels(scores, spec)

    if len(features) == 2:
        _render_decision_map(
            frame,
            features=features,
            labels=labels,
            spec=spec,
            artifacts=cached,
        )
    elif len(features) == 1:
        feature = features[0]
        probs = _scores_to_probabilities(scores, spec)
        x_vals = x[:, 0]
        x_label = AXIS_LABELS.get(feature, feature)
        prob_fig = build_nn_1d_probability_figure(x_vals, probs, labels, x_label=x_label)
        st.pyplot(prob_fig, clear_figure=True)
        plt.close(prob_fig)

    mismatch = int(np.sum(predicted != labels.astype(int)))
    st.caption(f"訓練集預測與標籤不一致：{mismatch} 筆。")


def _render_decision_map(
    frame: pd.DataFrame,
    *,
    features: list[str],
    labels: np.ndarray,
    spec: NetworkSpec,
    artifacts: dict,
) -> None:
    x1, x2, mesh_xx, mesh_yy, grid = _build_decision_mesh(frame, features)
    x1_name, x2_name = features[0], features[1]
    mesh_scores = predict_scores(
        artifacts["artifacts"].model,
        grid,
        spec,
        feature_normalizer=artifacts["artifacts"].feature_normalizer,
    )
    prob_grid = _scores_to_probabilities(mesh_scores, spec)
    boundary_fig = build_nn_decision_boundary_figure(
        x1,
        x2,
        labels,
        mesh_xx,
        mesh_yy,
        prob_grid,
        x1_label=AXIS_LABELS.get(x1_name, x1_name),
        x2_label=AXIS_LABELS.get(x2_name, x2_name),
    )
    st.pyplot(boundary_fig, clear_figure=True)
    plt.close(boundary_fig)
