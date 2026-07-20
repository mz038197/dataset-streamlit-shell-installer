from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    WORKSPACE_DIR,
    _display_path,
    invoke_data_agent,
    render_chat_panel,
    render_dataset_metrics,
)
from dataset_streamlit_shell.ui.nn_form_state import (
    AGENT_EPOCHS_MAX,
    DEFAULT_MAX_RUNS,
    FORM_MTIME_KEY,
    LOOP_STATUS_IDLE,
    LOOP_STATUS_NEED_AGENT,
    LOOP_STATUS_NEED_TRAIN,
    MAX_RUNS_HARD_CAP,
    apply_state_to_session,
    build_nn_page_agent_context,
    clamp_agent_epochs,
    clamp_max_runs,
    clear_train_request,
    consume_train_request,
    format_last_run_summary,
    form_file_mtime,
    get_loop_state,
    load_last_run,
    load_nn_form_state,
    nn_form_path,
    nn_last_run_path,
    nn_train_request_path,
    remaining_runs,
    reset_loop_budget,
    save_nn_form_state,
    session_to_state,
    set_loop_state,
    specs_to_state,
    write_last_run,
)
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
    estimate_parameter_count,
    format_model_code,
    lab02_default_compile_spec,
    lab02_default_network_spec,
    load_builtin_frame,
    predict_class_labels,
    predict_scores,
    train_model,
    validate_network_spec,
)
from dataset_streamlit_shell.ml.tf_runtime import configure_tensorflow_runtime
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
    """左欄分頁、右欄固定 Agent（與 CNN 等教學頁同一版面）。"""
    configure_tensorflow_runtime()
    frame = load_builtin_frame(BUILTIN_PATH)
    _sync_form_from_disk_if_newer()

    main, side = st.columns([5, 3], gap="large")
    with main:
        tab_activation, tab_train = st.tabs(["活化函數", "神經網路訓練"])
        with tab_activation:
            _render_activation_tab()
        with tab_train:
            _render_training_tab(frame)

    with side:
        form_state = session_to_state(st.session_state)
        extra_context = _nn_extra_context(form_state, len(frame))
        st.session_state[CONTEXT_KEY] = extra_context
        loop = get_loop_state(st.session_state)

        if loop["status"] == LOOP_STATUS_NEED_AGENT:
            _run_agent_decision_turn(extra_context=extra_context, row_count=len(frame))
            if consume_train_request(WORKSPACE_DIR, st.session_state):
                st.rerun()
            set_loop_state(st.session_state, status=LOOP_STATUS_IDLE)
            # 下一輪會在 widget 建立前同步 Agent 可能更新的 form。
            st.rerun()

        render_chat_panel(extra_context=extra_context, page_name="類神經網路")
        if st.session_state.pop("data_chat_just_replied", False):
            consume_train_request(WORKSPACE_DIR, st.session_state)
            # 不可在本輪 widget 建立後寫回其 session_state；交由下一輪開頭同步。
            st.rerun()


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


def _sync_form_from_disk_if_newer() -> dict:
    """若 Agent 改過 nn_form.json，在 widget 建立前灌回 session_state。"""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    state = load_nn_form_state(WORKSPACE_DIR)
    mtime = form_file_mtime(WORKSPACE_DIR)
    applied = float(st.session_state.get(FORM_MTIME_KEY, -1.0))
    if "nn_features" not in st.session_state or mtime > applied:
        apply_state_to_session(st.session_state, state)
        st.session_state[FORM_MTIME_KEY] = mtime
    return state


def _persist_form_from_session() -> dict:
    state = session_to_state(st.session_state)
    save_nn_form_state(WORKSPACE_DIR, state)
    st.session_state[FORM_MTIME_KEY] = form_file_mtime(WORKSPACE_DIR)
    return state


def _nn_extra_context(form_state: dict, row_count: int) -> str:
    loop = get_loop_state(st.session_state)
    last_run = load_last_run(WORKSPACE_DIR)
    return build_nn_page_agent_context(
        form_state=form_state,
        row_count=row_count,
        loop=loop,
        last_run=last_run,
        form_path=_display_path(nn_form_path(WORKSPACE_DIR)),
        request_path=_display_path(nn_train_request_path(WORKSPACE_DIR)),
        last_run_path=_display_path(nn_last_run_path(WORKSPACE_DIR)),
    )


def _render_agent_experiment_controls() -> None:
    loop = get_loop_state(st.session_state)
    st.markdown("##### Agent 實驗")
    st.caption(
        "請在右側 chat 請 Agent 設計架構並寫入訓練請求；"
        "請求後左欄會播放與手動相同的訓練動畫。次數上限可調（1～5）。"
    )
    if "nn_agent_max_runs" not in st.session_state:
        st.session_state["nn_agent_max_runs"] = clamp_max_runs(
            loop.get("max_runs", DEFAULT_MAX_RUNS)
        )
    max_runs = st.number_input(
        "Agent 實驗最多幾輪",
        min_value=1,
        max_value=MAX_RUNS_HARD_CAP,
        step=1,
        key="nn_agent_max_runs",
    )
    set_loop_state(st.session_state, max_runs=int(max_runs))
    loop = get_loop_state(st.session_state)
    if loop["run_index"] >= loop["max_runs"] and loop["status"] != LOOP_STATUS_IDLE:
        set_loop_state(st.session_state, status=LOOP_STATUS_IDLE)

    progress_col, reset_col = st.columns([3, 1])
    progress_col.info(
        f"實驗進度：已完成 {loop['run_index']}／{loop['max_runs']} 輪"
        f"（剩餘 {remaining_runs(loop)}）· 狀態 `{loop['status']}`"
    )
    if reset_col.button("重置次數", key="nn_reset_agent_budget", help="清除 Agent 實驗進度，不改架構"):
        reset_loop_budget(st.session_state)
        clear_train_request(WORKSPACE_DIR)
        st.rerun()

    last_run = load_last_run(WORKSPACE_DIR)
    st.caption(f"Agent 最近一次實驗：{format_last_run_summary(last_run)}")

    st.markdown("##### 建議問 Agent")
    for question in (
        "請設計一個不過大的二元分類網路，寫回左欄架構，並寫入訓練請求開始第一輪動畫訓練。",
        "請先說明你選的層數與 activation 理由，再更新 nn_form.json 並請求訓練。",
        "依最近一次訓練結果調整 learning rate 或隱藏層寬度，解釋取捨後再請求下一輪訓練。",
    ):
        st.markdown(f"- `{question}`")


def _render_training_tab(frame: pd.DataFrame) -> None:
    """訓練分頁主內容（右欄 Agent 由 render_neural_network_page 固定渲染）。"""
    context_key = CONTEXT_KEY
    loop = get_loop_state(st.session_state)
    st.title("類神經網路")
    st.caption(
        "使用內建雙特徵二元分類資料，以 TensorFlow Sequential 建立並訓練神經網路。"
        "也可請右側 Agent 改左欄架構並寫入訓練請求，由系統播放訓練動畫。"
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

    _render_agent_experiment_controls()
    # 控制項可能已調整上限或把超出預算的迴圈停回 idle。
    loop = get_loop_state(st.session_state)

    if st.button("套用 Lab02 預設", key="nn_apply_lab02"):
        _apply_lab02_defaults()
        st.rerun()

    spec, compile_spec, train_config = _render_network_form(frame)
    form_state = _persist_form_from_session()
    st.session_state[context_key] = _nn_extra_context(form_state, len(frame))

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

    agent_train_config = TrainConfig(epochs=clamp_agent_epochs(train_config.epochs))
    if agent_train_config.epochs != train_config.epochs and loop["status"] == LOOP_STATUS_NEED_TRAIN:
        st.caption(
            f"Agent 本輪 epochs 已 clamp 為 {agent_train_config.epochs}"
            f"（上限 {AGENT_EPOCHS_MAX}）。"
        )

    signature = _training_signature(spec, compile_spec, train_config, len(frame))
    agent_signature = _training_signature(spec, compile_spec, agent_train_config, len(frame))
    train_disabled = bool(validation_errors)
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_neural_network",
        disabled=train_disabled,
    )

    if loop["status"] == LOOP_STATUS_NEED_TRAIN:
        if validation_errors:
            st.error("架構驗證失敗，已取消本輪 Agent 訓練。請修正後再請 Agent 寫入訓練請求。")
            set_loop_state(st.session_state, status=LOOP_STATUS_IDLE)
        elif not _ensure_tensorflow_available():
            set_loop_state(st.session_state, status=LOOP_STATUS_IDLE)
        else:
            with st.status("Agent 實驗訓練中…", expanded=True):
                ok = _run_training(
                    frame,
                    spec=spec,
                    compile_spec=compile_spec,
                    train_config=agent_train_config,
                    signature=agent_signature,
                    context_key=context_key,
                )
            if ok:
                cached = st.session_state.get(RESULT_KEY, {})
                artifacts = cached.get("artifacts")
                if artifacts is not None:
                    new_index = int(loop["run_index"]) + 1
                    write_last_run(
                        WORKSPACE_DIR,
                        form_state=form_state,
                        train_result=artifacts.result,
                        run_index=new_index,
                        epochs_used=agent_train_config.epochs,
                        note="Agent 實驗動畫訓練",
                    )
                    set_loop_state(
                        st.session_state,
                        run_index=new_index,
                        status=LOOP_STATUS_NEED_AGENT,
                    )
                    st.rerun()
            else:
                set_loop_state(st.session_state, status=LOOP_STATUS_IDLE)
    elif train_clicked:
        if _ensure_tensorflow_available():
            with st.status("訓練中…", expanded=True):
                _run_training(
                    frame,
                    spec=spec,
                    compile_spec=compile_spec,
                    train_config=train_config,
                    signature=signature,
                    context_key=context_key,
                )
    elif RESULT_KEY in st.session_state and st.session_state[RESULT_KEY]["signature"] in {
        signature,
        agent_signature,
    }:
        _render_training_results(st.session_state[RESULT_KEY], frame=frame, spec=spec)
    else:
        st.info(
            "可按「開始訓練」自行觀察動畫，或請右側 Agent 寫入訓練請求啟動實驗迴圈。"
        )


def _run_agent_decision_turn(*, extra_context: str, row_count: int) -> None:
    loop = get_loop_state(st.session_state)
    last_run = load_last_run(WORKSPACE_DIR)
    summary = format_last_run_summary(last_run)
    remaining = remaining_runs(loop)
    user_text = (
        "【系統｜Agent 實驗決策】"
        f"剛剛完成第 {loop['run_index']}／{loop['max_runs']} 輪左欄動畫訓練（資料 {row_count} 筆）。"
        f"結果：{summary}。剩餘可實驗 {remaining} 輪。"
        "請用繁體中文解釋結果與取捨；若要繼續，先更新 "
        f"{_display_path(nn_form_path(WORKSPACE_DIR))}，再寫入 "
        f'{_display_path(nn_train_request_path(WORKSPACE_DIR))} 為 {{"requested": true}}；'
        "若停止實驗，請總結理由且不要寫訓練請求。"
        "不要改高 max_runs，不要清除預算。"
    )
    with st.spinner("Agent 正在依訓練結果決策…"):
        answer = invoke_data_agent(
            user_text,
            extra_context=extra_context,
            display_user_text="（系統）請依剛剛的訓練結果決定是否調整參數並繼續實驗。",
        )
    st.info(answer[:500] + ("…" if len(answer) > 500 else ""))


def _apply_lab02_defaults() -> None:
    state = specs_to_state(
        lab02_default_network_spec(),
        lab02_default_compile_spec(),
        TrainConfig(epochs=100),
    )
    save_nn_form_state(WORKSPACE_DIR, state)
    apply_state_to_session(st.session_state, state)
    st.session_state[FORM_MTIME_KEY] = form_file_mtime(WORKSPACE_DIR)


def _render_network_form(frame: pd.DataFrame) -> tuple[NetworkSpec, CompileSpec, TrainConfig]:
    del frame  # 表單不直接依賴 frame；驗證在外層
    st.markdown("##### 輸入與架構")
    stored = st.session_state.get("nn_features")
    if isinstance(stored, list):
        valid = [feature for feature in stored if feature in FEATURE_OPTIONS]
        if valid != stored:
            st.session_state["nn_features"] = valid or list(FEATURE_OPTIONS)

    selected_features = st.multiselect(
        "輸入特徵（1～2 個）",
        list(FEATURE_OPTIONS),
        key="nn_features",
    )
    hidden_count = st.number_input(
        "隱藏層數",
        min_value=0,
        max_value=MAX_HIDDEN_LAYERS,
        step=1,
        key="nn_hidden_count",
    )
    hidden_layers: list[HiddenLayerSpec] = []
    for index in range(1, int(hidden_count) + 1):
        if f"nn_hidden_units_{index}" not in st.session_state:
            st.session_state[f"nn_hidden_units_{index}"] = 3 if index == 1 else 4
        if f"nn_hidden_activation_{index}" not in st.session_state:
            st.session_state[f"nn_hidden_activation_{index}"] = (
                "sigmoid" if index == 1 else "relu"
            )
        col_units, col_act = st.columns(2)
        units = col_units.number_input(
            f"第 {index} 層神經元數",
            min_value=1,
            max_value=MAX_UNITS_PER_LAYER,
            step=1,
            key=f"nn_hidden_units_{index}",
        )
        activation = col_act.selectbox(
            f"第 {index} 層活化函數",
            ACTIVATION_CHOICES,
            key=f"nn_hidden_activation_{index}",
        )
        hidden_layers.append(HiddenLayerSpec(int(units), activation))

    out_col1, out_col2 = st.columns(2)
    output_units = out_col1.number_input(
        "輸出神經元數",
        min_value=1,
        max_value=10,
        step=1,
        key="nn_output_units",
    )
    output_activation_options = _output_activation_options(int(output_units))
    current_out_act = st.session_state.get("nn_output_activation", "sigmoid")
    if current_out_act not in output_activation_options:
        st.session_state["nn_output_activation"] = (
            "sigmoid" if int(output_units) == 1 else "softmax"
        )
    output_activation = out_col2.selectbox(
        "輸出活化函數",
        output_activation_options,
        key="nn_output_activation",
    )

    with st.expander("進階：正規化與 loss", expanded=False):
        use_norm_layer = st.checkbox(
            "在 Sequential 內加入 Normalization 層（與訓練前 adapt 二擇一）",
            key="nn_use_norm_layer",
        )
        if use_norm_layer:
            st.info("已啟用層內 Normalization；訓練前不另做 adapt+transform。")
        else:
            st.caption("預設：訓練前以 Normalization.adapt 正規化特徵，不放入 Sequential。")
        st.selectbox("loss", LOSS_CHOICES, key="nn_loss_choice")
    use_norm_layer = bool(st.session_state.get("nn_use_norm_layer", False))
    loss_choice = st.session_state.get("nn_loss_choice", LOSS_AUTO)

    st.markdown("##### 編譯與訓練")
    c1, c2, c3 = st.columns(3)
    optimizer_name = c1.selectbox("優化器", OPTIMIZER_CHOICES, key="nn_optimizer")
    learning_rate = c2.number_input(
        "learning_rate",
        min_value=0.0001,
        max_value=1.0,
        format="%.4f",
        key="nn_learning_rate",
    )
    epochs = c3.number_input("epochs", min_value=1, max_value=500, step=1, key="nn_epochs")

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


def _nn_animation_epochs(total_epochs: int) -> set[int]:
    if total_epochs <= 80:
        return set(range(1, total_epochs + 1))
    stride = max(total_epochs // 80, 1)
    selected = set(range(1, total_epochs + 1, stride))
    selected.add(total_epochs)
    return selected


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
    container.pyplot(fig, clear_figure=True, width="stretch")
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
    container.pyplot(fig, clear_figure=True, width="stretch")
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
    container.pyplot(fig, clear_figure=True, width="stretch")
    plt.close(fig)


def _run_training(
    frame: pd.DataFrame,
    *,
    spec: NetworkSpec,
    compile_spec: CompileSpec,
    train_config: TrainConfig,
    signature: tuple,
    context_key: str,
) -> bool:
    features = list(spec.input_features)
    x = frame[features].to_numpy(dtype=np.float32)
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    labels = frame[TARGET_COLUMN].to_numpy(dtype=float)

    chart_left, chart_right = st.columns(2)
    boundary_placeholder = chart_left.empty()
    loss_placeholder = chart_right.empty()
    status_placeholder = st.empty()

    mesh_xx = mesh_yy = grid = None
    if len(features) == 2:
        _, _, mesh_xx, mesh_yy, grid = _build_decision_mesh(frame, features)

    animation_epochs = _nn_animation_epochs(train_config.epochs)
    metric_key = "accuracy" if spec.output_units == 1 else "sparse_categorical_accuracy"

    def on_epoch_end(
        epoch: int,
        history: dict[str, list[float]],
        model,
        feature_normalizer,
    ) -> None:
        if epoch not in animation_epochs:
            return
        _render_epoch_loss_plot(history, epoch=epoch, container=loss_placeholder)
        if len(features) == 2 and grid is not None:
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
                container=boundary_placeholder,
            )
        elif len(features) == 1:
            _render_epoch_probability_plot(
                x=x,
                labels=labels,
                spec=spec,
                model=model,
                feature_normalizer=feature_normalizer,
                feature=features[0],
                epoch=epoch,
                container=boundary_placeholder,
            )
        loss_value = history.get("loss", [float("nan")])[-1]
        accuracy_value = history.get(metric_key, [0.0])[-1] * 100.0
        status_placeholder.caption(
            f"Epoch {epoch:,} / {train_config.epochs:,} · "
            f"loss = {loss_value:.4f} · accuracy = {accuracy_value:.2f}%"
        )
        time.sleep(0.02)

    try:
        artifacts = train_model(
            spec,
            compile_spec,
            train_config,
            x,
            y,
            epoch_callback=on_epoch_end,
        )
    except Exception as exc:
        st.error(f"訓練失敗：{exc}")
        return False

    st.session_state[RESULT_KEY] = {
        "signature": signature,
        "artifacts": artifacts,
        "spec": spec,
        "compile_spec": compile_spec,
        "features": features,
    }
    st.session_state[context_key] = build_nn_agent_context(
        spec=spec,
        compile_spec=compile_spec,
        train_result=artifacts.result,
        row_count=len(frame),
    )
    _render_training_metrics(
        artifacts,
        frame=frame,
        spec=spec,
        features=features,
    )
    return True


def _render_training_metrics(
    artifacts,
    *,
    frame: pd.DataFrame,
    spec: NetworkSpec,
    features: list[str],
) -> None:
    result = artifacts.result
    st.markdown("##### 訓練結果")
    st.metric("最終 loss", f"{result.final_loss:.4f}")
    st.metric("訓練準確率", f"{result.train_accuracy:.2f}%")
    st.caption(f"參數數量：{result.parameter_count:,}")

    x = frame[features].to_numpy(dtype=np.float32)
    labels = frame[TARGET_COLUMN].to_numpy(dtype=float)
    scores = predict_scores(
        artifacts.model,
        x,
        spec,
        feature_normalizer=artifacts.feature_normalizer,
    )
    predicted = predict_class_labels(scores, spec)
    mismatch = int(np.sum(predicted != labels.astype(int)))
    st.caption(f"訓練集預測與標籤不一致：{mismatch} 筆。")


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
