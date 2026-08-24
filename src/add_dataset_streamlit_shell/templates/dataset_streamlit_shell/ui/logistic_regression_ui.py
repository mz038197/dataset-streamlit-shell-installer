"""邏輯迴歸教學頁：決策槽列、Agent 寫入、決策邊界與 Cost。"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.ml.classification import (
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
    attach_logistic_test_costs,
    build_classification_agent_context,
    logistic_gradient_descent_steps,
    map_feature,
    predict_class_from_proba,
    predict_proba,
    predict_proba_from_logistic_artifact,
    predict_proba_from_regularized_artifact,
    training_accuracy,
)
from dataset_streamlit_shell.ml.regression import (
    GradientDescentStep,
    apply_feature_scaler,
    create_feature_scaler,
)
from dataset_streamlit_shell.plotting import (
    CLASS_NEGATIVE_STYLE,
    CLASS_POSITIVE_STYLE,
    build_sigmoid_figure,
    configure_matplotlib_for_traditional_chinese,
    scatter_binary_classes,
)
from dataset_streamlit_shell.ui import logistic_quiz as quiz
from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    WORKSPACE_DIR,
    _display_path,
    invoke_data_agent,
    render_chat_panel,
    teaching_page_host_context,
)
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane
from dataset_streamlit_shell.ui.logistic_slot_state import (
    ADMISSION_FEATURES,
    ADMISSION_TARGET,
    DATA_PREVIEW_ROWS,
    MICROCHIP_FEATURES,
    MICROCHIP_TARGET,
    SLOT_IDS,
    SLOT_TITLES,
    STAGE_BOUNDARY,
    STAGE_POLY,
    build_logistic_page_snapshot,
    can_write_train_request,
    consume_train_request,
    empty_workspace_state,
    load_workspace_state,
    logistic_host_context_fragment,
    logistic_slots_path,
    logistic_train_request_path,
    model_code_preview,
    model_download_zip_bytes,
    model_download_zip_name,
    parse_train_pct,
    scale_method_errors,
    should_rerun_after_logistic_chat,
    slot_button_label,
    slot_inspect_rows,
    slot_is_filled,
    slot_signature,
    slots_are_complete,
    slots_file_mtime,
    split_frame_by_train_pct,
    train_request_is_set,
)
from dataset_streamlit_shell.ui.teaching_flow import (
    live_fit_caption,
    symbolic_logistic_yhat_latex,
    symbolic_logistic_z_latex,
)

configure_matplotlib_for_traditional_chinese()

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
UNIVERSITY_ADMISSION_PATH = CLASSIFICATION_DEMO_DIR / "university_admission.csv"
MICROCHIP_TEST_PATH = CLASSIFICATION_DEMO_DIR / "microchip_test.csv"

PAGE_TITLE = "邏輯迴歸"
CONTEXT_KEY = f"{PAGE_TITLE}_agent_context"
LG_TRAINING_CHART_FIGSIZE = (6.5, 6.5)

WORKSPACE_KEY = "logistic_workspace"
MTIME_KEY = "logistic_slots_applied_mtime"
INSPECT_KEY = "logistic_inspect_slot"

DECISION_SLOT_CSS = """
<style>
.lr-slot-row { margin: 0.2rem 0 0.55rem; }
[class*="st-key-lg_inspect_"] {
  border: 1px solid rgba(90,160,255,.35);
  border-radius: 10px;
  background: rgba(90,160,255,.08);
  padding: 0.65rem 0.75rem;
  margin: 0 0 0.8rem;
}
.lr-inspect { margin: 0; }
.lr-inspect h3 { margin: 0 0 0.35rem; font-size: 14px; }
.lr-inspect dl { margin: 0 0 0.55rem; display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 0.7rem; font-size: 13px; }
.lr-inspect dt { color: rgba(250,250,250,.58); }
.lr-inspect dd { margin: 0; }
.lr-inspect .formula {
  margin: 0;
  padding: 0.2rem 0.4rem;
  border-radius: 6px;
  background: #0b0d12;
  font: 15px/1.45 "Cambria Math", "Times New Roman", serif;
}
[class*="st-key-lg_slotbox_"][class*="_done"] button {
  border: 2px solid #3dd68c !important;
}
[class*="st-key-lg_slotbox_"][class*="_todo"] button {
  border: 2px solid #e85d5d !important;
}
</style>
"""


def _lg_host() -> str:
    return teaching_page_host_context(
        logistic_host_context_fragment(
            slots_path=_display_path(logistic_slots_path(WORKSPACE_DIR)),
            request_path=_display_path(logistic_train_request_path(WORKSPACE_DIR)),
        )
    )


def render_logistic_regression_page() -> None:
    workspace = _sync_workspace()
    teaching, agent = open_content_dual_pane()
    with teaching:
        st.title(PAGE_TITLE)
        st.caption(
            "框名是決策槽，框上是目前選擇。請資料 Agent 欄組模型；"
            "點框看只讀選擇明細。訓練畫面是決策邊界、訓練／測試 Cost，下方為測試集機率對照。"
        )
        st.markdown(DECISION_SLOT_CSS, unsafe_allow_html=True)
        stage_label = st.radio(
            "學習階段",
            list(quiz.LEARNING_STAGES),
            horizontal=True,
            key="logistic_learning_stage",
        )
        if stage_label == quiz.STAGE_BOUNDARY:
            _render_stage(
                workspace[STAGE_BOUNDARY],
                stage=STAGE_BOUNDARY,
                csv_path=UNIVERSITY_ADMISSION_PATH,
                features=list(ADMISSION_FEATURES),
                target=ADMISSION_TARGET,
                source_label="內建範例資料：大學錄取兩科成績",
                mapped=False,
            )
        else:
            _render_stage(
                workspace[STAGE_POLY],
                stage=STAGE_POLY,
                csv_path=MICROCHIP_TEST_PATH,
                features=list(MICROCHIP_FEATURES),
                target=MICROCHIP_TARGET,
                source_label="內建範例資料：微晶片兩項檢測",
                mapped=True,
            )
    with agent:
        extra = str(st.session_state.get(CONTEXT_KEY, f"目前頁面：{PAGE_TITLE}。"))
        render_chat_panel(
            extra_context=extra,
            page_name=PAGE_TITLE,
            host_context=_lg_host(),
            skip_working_snapshot=True,
            after_reply=_after_logistic_chat,
        )


def _sync_workspace() -> dict:
    previous = st.session_state.get(WORKSPACE_KEY)
    if not isinstance(previous, dict):
        previous = empty_workspace_state()
    workspace = load_workspace_state(WORKSPACE_DIR, previous)
    _drop_stale_results(previous, workspace)
    st.session_state[WORKSPACE_KEY] = workspace
    st.session_state[MTIME_KEY] = slots_file_mtime(WORKSPACE_DIR)
    return workspace


def _drop_stale_results(previous: dict, workspace: dict) -> None:
    mapping = {
        STAGE_BOUNDARY: "logistic_boundary_last_artifact",
        STAGE_POLY: "logistic_poly_last_artifact",
    }
    anim_keys = {
        STAGE_BOUNDARY: "logistic_boundary_anim",
        STAGE_POLY: "logistic_poly_anim",
    }
    for stage, result_key in mapping.items():
        if slot_signature(previous.get(stage, {})) != slot_signature(workspace.get(stage, {})):
            st.session_state.pop(result_key, None)
            st.session_state.pop(anim_keys[stage], None)


def _after_logistic_chat() -> None:
    requested = train_request_is_set(WORKSPACE_DIR)
    applied = float(st.session_state.get(MTIME_KEY, -1.0))
    if should_rerun_after_logistic_chat(
        requested=requested,
        slots_mtime=slots_file_mtime(WORKSPACE_DIR),
        applied_mtime=applied,
    ):
        st.rerun()


def _render_slot_row(state: dict, *, stage: str) -> str | None:
    inspect_key = f"{INSPECT_KEY}_{stage}"
    current = st.session_state.get(inspect_key)
    cols = st.columns(6)
    for column, slot_id in zip(cols, SLOT_IDS):
        filled = slot_is_filled(slot_id, state, stage)
        status = "done" if filled else "todo"
        with column:
            with st.container(key=f"lg_slotbox_{stage}_{slot_id}_{status}"):
                if st.button(
                    slot_button_label(slot_id, state, stage),
                    key=f"lg_slot_{stage}_{slot_id}",
                    width="stretch",
                    type="secondary",
                    help="點一下看選擇明細，再點收起。不能在這裡改選擇。",
                ):
                    st.session_state[inspect_key] = None if current == slot_id else slot_id
                    st.rerun()
    return st.session_state.get(inspect_key)


def _render_inspect(
    state: dict,
    *,
    stage: str,
    open_slot: str | None,
    frame: pd.DataFrame | None = None,
    features: list[str] | None = None,
    target: str | None = None,
) -> None:
    if not open_slot:
        st.caption("點上面的框看目前選擇。只讀；要換切分、縮放、α／epochs 或 λ，跟資料 Agent 欄說。")
        return
    row_count = len(frame) if frame is not None else None
    rows = slot_inspect_rows(open_slot, state, stage=stage, row_count=row_count)
    items = []
    for key, value in rows:
        cls = ' class="formula"' if key in {"公式", "φ", "z", "ŷ"} else ""
        items.append(f"<dt>{key}</dt><dd{cls}>{value}</dd>")
    with st.container(key=f"lg_inspect_{stage}_{open_slot}"):
        st.markdown(
            f'<aside class="lr-inspect"><h3>{SLOT_TITLES[open_slot]}</h3>'
            f"<dl>{''.join(items)}</dl></aside>",
            unsafe_allow_html=True,
        )
        if open_slot == "linear":
            st.latex(symbolic_logistic_z_latex(mapped=stage == STAGE_POLY))
            st.latex(symbolic_logistic_yhat_latex())
        if open_slot == "loss":
            _render_logistic_cost_formula(regularized=stage == STAGE_POLY)
        if open_slot != "data" or frame is None or not features or not target:
            return
        preview_cols = [column for column in [*features, target] if column in frame.columns]
        if not preview_cols:
            return
        st.dataframe(
            frame[preview_cols].head(DATA_PREVIEW_ROWS),
            width="stretch",
            hide_index=True,
        )


def _render_code_preview(state: dict, *, stage: str) -> None:
    with st.expander("模型程式碼預覽", expanded=False):
        st.code(model_code_preview(state, stage=stage), language="python")
        st.caption("只讀預覽，對應目前決策槽狀態。主教學欄訓練走本頁梯度下降，不是 exec 這段。")
        if not slots_are_complete(state):
            return
        payload = model_download_zip_bytes(state, stage=stage)
        if payload is None:
            return
        st.download_button(
            "下載模型下載程式碼",
            data=payload,
            file_name=model_download_zip_name(stage),
            mime="application/zip",
            width="stretch",
            key=f"lg_model_download_{stage}_{slot_signature(state)}",
        )


def _training_frame(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    return df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()


def _train_test_frames(working: pd.DataFrame, state: dict):
    pct = parse_train_pct((state.get("choices") or {}).get("split"))
    if pct is None:
        return None
    return split_frame_by_train_pct(working, pct)


def _scale_check_frame(working: pd.DataFrame, state: dict) -> pd.DataFrame:
    split = _train_test_frames(working, state)
    return split[0] if split is not None else working


def _animation_steps(steps: list[GradientDescentStep]) -> list[GradientDescentStep]:
    if len(steps) <= 80:
        return steps
    stride = max(len(steps) // 80, 1)
    selected = steps[::stride]
    if selected[-1] != steps[-1]:
        selected.append(steps[-1])
    return selected


def _fit_status_caption(step: GradientDescentStep, *, total_iterations: int, mapped: bool) -> str:
    if mapped:
        weights = [float(w) for w in step.weights]
        norm_sq = float(np.dot(weights, weights))
        train_label = "訓練 Cost J" if step.test_cost is not None else "Cost J"
        test_part = f"，測試 Cost J = {step.test_cost:.4f}" if step.test_cost is not None else ""
        return (
            f"Iteration {step.iteration:,} / {total_iterations:,}，"
            f"‖w‖² = {norm_sq:.4f}，B = {step.intercept:.4f}，"
            f"{train_label} = {step.cost:.4f}{test_part}"
        )
    return live_fit_caption(
        iteration=step.iteration,
        total_iterations=total_iterations,
        weights=step.weights,
        intercept=step.intercept,
        cost=step.cost,
        test_cost=step.test_cost,
    )


def _maybe_start_from_request(
    state: dict,
    *,
    quiz_unlocked: bool,
    scale_errors: list[str],
    start,
) -> None:
    allowed = can_write_train_request(
        state,
        quiz_unlocked=quiz_unlocked,
        scale_errors=scale_errors,
    )
    had_request = train_request_is_set(WORKSPACE_DIR)
    started = consume_train_request(WORKSPACE_DIR, allowed=allowed)
    if started:
        start()
    elif had_request:
        st.warning("訓練請求已忽略：決策槽未齊、訓練前預測未過關，或縮放條件不成立。")


def _render_stage(
    state: dict,
    *,
    stage: str,
    csv_path,
    features: list[str],
    target: str,
    source_label: str,
    mapped: bool,
) -> None:
    df = pd.read_csv(csv_path)
    missing = [column for column in features + [target] if column not in df.columns]
    if missing:
        st.warning(f"內建資料缺少固定欄位：{', '.join(missing)}。")
        return
    working = _training_frame(df, features, target)
    if len(working) < 2:
        st.warning("可用樣本少於 2 筆，無法訓練邏輯迴歸。")
        return

    anim_key = f"logistic_{stage}_anim"
    result_key = f"logistic_{stage}_last_artifact"
    quiz_unlocked = (
        _boundary_quiz_unlocked() if stage == STAGE_BOUNDARY else _poly_quiz_unlocked()
    )
    scale_errors = scale_method_errors(
        state["choices"].get("scale"),
        _scale_check_frame(working, state),
        features,
    )
    signature = slot_signature(state)
    stored = st.session_state.get(result_key)
    artifact = None
    if isinstance(stored, dict) and stored.get("signature") == signature:
        artifact = stored["artifact"]

    open_slot = _render_slot_row(state, stage=stage)
    _render_inspect(
        state,
        stage=stage,
        open_slot=open_slot,
        frame=working,
        features=features,
        target=target,
    )
    _render_code_preview(state, stage=stage)
    if stage == STAGE_BOUNDARY:
        _render_sigmoid_visualization()
        quiz_unlocked = _render_boundary_pretrain_quiz()
    else:
        quiz_unlocked = _render_poly_pretrain_quiz()

    def _start() -> None:
        _queue_training(
            working,
            features=features,
            target=target,
            state=state,
            source_label=source_label,
            result_key=result_key,
            anim_key=anim_key,
            mapped=mapped,
        )

    anim = st.session_state.get(anim_key)
    training_active = isinstance(anim, dict) and not anim.get("finished", True)
    if not training_active:
        _maybe_start_from_request(
            state,
            quiz_unlocked=quiz_unlocked,
            scale_errors=scale_errors,
            start=_start,
        )
        anim = st.session_state.get(anim_key)
        training_active = isinstance(anim, dict) and not anim.get("finished", True)

    allowed = can_write_train_request(
        state,
        quiz_unlocked=quiz_unlocked,
        scale_errors=scale_errors,
    )
    if scale_errors:
        st.error(scale_errors[0])
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key=f"train_logistic_{stage}",
        disabled=not allowed or training_active,
    )
    if not slots_are_complete(state):
        st.caption("先請資料 Agent 欄組齊決策槽。")
    elif not quiz_unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")
    if train_clicked and allowed:
        _start()
        training_active = True

    chart_left, chart_right = st.columns(2)
    left_placeholder = chart_left.empty()
    cost_placeholder = chart_right.empty()
    pred_placeholder = st.empty()
    status_placeholder = st.empty()

    if training_active:
        _run_training(
            working,
            features=features,
            target=target,
            result_key=result_key,
            anim_key=anim_key,
            left_placeholder=left_placeholder,
            cost_placeholder=cost_placeholder,
            pred_placeholder=pred_placeholder,
            status_placeholder=status_placeholder,
            mapped=mapped,
        )
        stored = st.session_state.get(result_key)
        if isinstance(stored, dict) and stored.get("signature") == signature:
            artifact = stored["artifact"]
        anim = st.session_state.get(anim_key)
        training_active = isinstance(anim, dict) and not anim.get("finished", True)

    threshold = 0.5
    if not training_active:
        if artifact is not None:
            threshold = _classification_threshold_slider(stage, enabled=True)
            _show_result(
                working,
                features=features,
                target=target,
                artifact=artifact,
                result_key=result_key,
                left_placeholder=left_placeholder,
                cost_placeholder=cost_placeholder,
                pred_placeholder=pred_placeholder,
                status_placeholder=status_placeholder,
                state=state,
                mapped=mapped,
                threshold=threshold,
            )
        else:
            status_placeholder.caption(
                "訓練後這裡會出現決策邊界、訓練／測試 Cost 與測試集機率對照。"
            )

    _set_agent_context(
        working,
        state=state,
        stage=stage,
        features=features,
        target=target,
        source_label=source_label,
        quiz_unlocked=quiz_unlocked,
        scale_errors=scale_errors,
        artifact=artifact,
        open_slot=open_slot,
        threshold=threshold if artifact is not None else None,
        mapped=mapped,
    )


def _queue_training(
    working: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    state: dict,
    source_label: str,
    result_key: str,
    anim_key: str,
    mapped: bool,
) -> None:
    split = _train_test_frames(working, state)
    if split is None:
        return
    train_df, test_df = split
    method = str(state["choices"]["scale"])
    scaler = create_feature_scaler(train_df, features, method)
    scaled_train = apply_feature_scaler(train_df, scaler)
    scaled_test = apply_feature_scaler(test_df, scaler)
    lambda_ = float(state["lambda_"] or 0.0)
    if mapped:
        train_x, mapped_features = map_feature(scaled_train, features, degree=DEFAULT_MAP_DEGREE)
        test_x, _ = map_feature(scaled_test, features, degree=DEFAULT_MAP_DEGREE)
        rng = np.random.default_rng(1)
        initial_weights = rng.normal(0.0, 0.01, size=len(mapped_features))
        initial_intercept = 1.0
        regularized = True
    else:
        train_x = scaled_train[features]
        test_x = scaled_test[features]
        mapped_features = list(features)
        initial_weights = None
        initial_intercept = 0.0
        regularized = False
    steps = attach_logistic_test_costs(
        logistic_gradient_descent_steps(
            train_x,
            train_df[target],
            learning_rate=float(state["alpha"]),
            epochs=int(state["epochs"]),
            initial_weights=initial_weights,
            initial_intercept=initial_intercept,
            lambda_=lambda_,
            regularized=regularized,
        ),
        test_x,
        test_df[target],
        lambda_=lambda_,
        regularized=regularized,
    )
    st.session_state[anim_key] = {
        "steps": steps,
        "sampled": _animation_steps(steps),
        "finished": False,
        "signature": slot_signature(state),
        "scaler": scaler,
        "source_label": source_label,
        "features": list(features),
        "mapped_features": list(mapped_features),
        "target": target,
        "train_frame": train_df,
        "test_frame": test_df,
        "train_x": train_x,
        "test_x": test_x,
        "mapped": mapped,
        "lambda_": lambda_,
    }
    st.session_state.pop(result_key, None)


def _run_training(
    working: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    result_key: str,
    anim_key: str,
    left_placeholder,
    cost_placeholder,
    pred_placeholder,
    status_placeholder,
    mapped: bool,
) -> None:
    anim = st.session_state.get(anim_key)
    if not isinstance(anim, dict) or anim.get("finished"):
        return
    steps: list[GradientDescentStep] = list(anim["steps"])
    sampled: list[GradientDescentStep] = list(anim["sampled"])
    if not sampled:
        anim["finished"] = True
        return

    for step in sampled:
        caption = _fit_status_caption(step, total_iterations=steps[-1].iteration, mapped=mapped)
        _render_boundary_or_contour(
            working,
            features=features,
            target=target,
            step=step,
            placeholder=left_placeholder,
            scaler=anim["scaler"],
            train_df=anim["train_frame"],
            test_df=anim["test_frame"],
            mapped=mapped,
            mapped_features=anim["mapped_features"],
        )
        history = [item for item in steps if item.iteration <= step.iteration]
        _render_cost_history_plot(history, cost_placeholder)
        _render_probability_plot(
            anim["test_frame"][target],
            predict_proba(anim["test_x"], step.weights, step.intercept),
            pred_placeholder,
            threshold=0.5,
        )
        status_placeholder.caption(caption)
        time.sleep(0.015)

    final_step = steps[-1]
    if mapped:
        artifact: ClassificationArtifact = RegularizedLogisticModelArtifact(
            model_kind=MODEL_KIND_REGULARIZED,
            base_features=list(features),
            mapped_features=list(anim["mapped_features"]),
            target=target,
            weights=[float(w) for w in final_step.weights],
            intercept=float(final_step.intercept),
            map_degree=DEFAULT_MAP_DEGREE,
            lambda_=float(anim["lambda_"]),
            training_cost=float(final_step.cost),
            data_source=str(anim["source_label"]),
            scaler=anim["scaler"],
        )
    else:
        artifact = LogisticModelArtifact(
            model_kind=MODEL_KIND_LOGISTIC,
            features=list(features),
            target=target,
            weights=[float(w) for w in final_step.weights],
            intercept=float(final_step.intercept),
            scaler=anim["scaler"],
            training_cost=float(final_step.cost),
            data_source=str(anim["source_label"]),
        )
    st.session_state[result_key] = {
        "signature": anim["signature"],
        "artifact": artifact,
        "steps": steps,
        "test_cost": final_step.test_cost,
        "train_frame": anim["train_frame"],
        "test_frame": anim["test_frame"],
        "test_x": anim["test_x"],
        "scaler": anim["scaler"],
        "mapped_features": anim["mapped_features"],
    }
    anim["finished"] = True


def _show_result(
    working: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    artifact: ClassificationArtifact,
    result_key: str,
    left_placeholder,
    cost_placeholder,
    pred_placeholder,
    status_placeholder,
    state: dict,
    mapped: bool,
    threshold: float,
) -> None:
    split = _train_test_frames(working, state)
    if split is None:
        status_placeholder.warning("訓練結果缺少切分，請重新訓練。")
        return
    train_df, test_df = split
    bundle = st.session_state.get(result_key)
    cached_steps = bundle.get("steps") if isinstance(bundle, dict) else None
    scaler = artifact.scaler if artifact.scaler is not None else (
        bundle.get("scaler") if isinstance(bundle, dict) else None
    )
    mapped_features = (
        list(artifact.mapped_features)
        if isinstance(artifact, RegularizedLogisticModelArtifact)
        else list(features)
    )
    step = GradientDescentStep(
        iteration=int(cached_steps[-1].iteration) if cached_steps else 0,
        weights=[float(w) for w in artifact.weights],
        intercept=float(artifact.intercept),
        cost=float(artifact.training_cost),
        test_cost=bundle.get("test_cost") if isinstance(bundle, dict) else None,
    )
    _render_boundary_or_contour(
        working,
        features=features,
        target=target,
        step=step,
        placeholder=left_placeholder,
        scaler=scaler,
        train_df=train_df,
        test_df=test_df,
        mapped=mapped,
        mapped_features=mapped_features,
    )
    if isinstance(cached_steps, list) and cached_steps:
        _render_cost_history_plot(cached_steps, cost_placeholder)
    test_proba = (
        predict_proba_from_regularized_artifact(artifact, test_df)
        if isinstance(artifact, RegularizedLogisticModelArtifact)
        else predict_proba_from_logistic_artifact(artifact, test_df)
    )
    _render_probability_plot(test_df[target], test_proba, pred_placeholder, threshold=threshold)
    train_proba = (
        predict_proba_from_regularized_artifact(artifact, train_df)
        if isinstance(artifact, RegularizedLogisticModelArtifact)
        else predict_proba_from_logistic_artifact(artifact, train_df)
    )
    train_acc = training_accuracy(train_df[target], train_proba, threshold)
    test_acc = training_accuracy(test_df[target], test_proba, threshold)
    status_placeholder.caption(
        _fit_status_caption(step, total_iterations=int(step.iteration), mapped=mapped)
        + f"，訓練正確率 {train_acc:.1f}%，測試正確率 {test_acc:.1f}%"
    )
    _render_test_table(test_df[target], test_proba, threshold)


def _render_boundary_or_contour(
    working: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    step: GradientDescentStep,
    placeholder,
    scaler: dict | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mapped: bool,
    mapped_features: list[str],
) -> None:
    x1_name, x2_name = features[0], features[1]
    fig, ax = plt.subplots(figsize=LG_TRAINING_CHART_FIGSIZE, constrained_layout=True)
    _scatter_split_classes(ax, train_df, test_df, x1_name, x2_name, target)
    _draw_decision_level(
        ax,
        working,
        features=features,
        weights=step.weights,
        intercept=step.intercept,
        scaler=scaler,
        mapped=mapped,
        mapped_features=mapped_features,
    )
    ax.set_xlabel(x1_name)
    ax.set_ylabel(x2_name)
    title = (
        f"決策邊界 contour f=0.5（iteration {step.iteration}）"
        if mapped
        else f"決策邊界（iteration {step.iteration}）"
    )
    ax.set_title(title)
    ax.legend()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _scatter_split_classes(
    ax,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x1_name: str,
    x2_name: str,
    target: str,
) -> None:
    train_pos = train_df[target] == 1
    train_neg = train_df[target] == 0
    scatter_binary_classes(
        ax,
        train_df[x1_name].to_numpy(dtype=float),
        train_df[x2_name].to_numpy(dtype=float),
        positives=train_pos.to_numpy(),
        negatives=train_neg.to_numpy(),
    )
    test_pos = test_df[target] == 1
    test_neg = test_df[target] == 0
    ax.scatter(
        test_df.loc[test_neg, x1_name],
        test_df.loc[test_neg, x2_name],
        marker=CLASS_NEGATIVE_STYLE["marker"],
        c="0.45",
        edgecolors="0.35",
        linewidths=0.6,
        alpha=0.4,
        label="測試點",
    )
    ax.scatter(
        test_df.loc[test_pos, x1_name],
        test_df.loc[test_pos, x2_name],
        marker=CLASS_POSITIVE_STYLE["marker"],
        c="0.45",
        linewidths=1.2,
        alpha=0.4,
        label="_nolegend_",
    )


def _draw_decision_level(
    ax,
    working: pd.DataFrame,
    *,
    features: list[str],
    weights,
    intercept: float,
    scaler: dict | None,
    mapped: bool,
    mapped_features: list[str],
) -> None:
    x1_name, x2_name = features
    x1 = working[x1_name].to_numpy(dtype=float)
    x2 = working[x2_name].to_numpy(dtype=float)
    pad1 = 0.05 * (float(x1.max()) - float(x1.min()) or 1.0)
    pad2 = 0.05 * (float(x2.max()) - float(x2.min()) or 1.0)
    u = np.linspace(float(x1.min()) - pad1, float(x1.max()) + pad1, 80)
    v = np.linspace(float(x2.min()) - pad2, float(x2.max()) + pad2, 80)
    uu, vv = np.meshgrid(u, v)
    grid = pd.DataFrame({x1_name: uu.ravel(), x2_name: vv.ravel()})
    if scaler is not None:
        grid = apply_feature_scaler(grid, scaler)
    if mapped:
        mapped_grid, cols = map_feature(grid, features, degree=DEFAULT_MAP_DEGREE)
        proba = predict_proba(mapped_grid[cols], weights, intercept)
    else:
        proba = predict_proba(grid[features], weights, intercept)
    z = proba.to_numpy(dtype=float).reshape(uu.shape)
    ax.contour(u, v, z, levels=[0.5], colors="blue" if not mapped else "green")


def _render_cost_history_plot(steps: list[GradientDescentStep], placeholder) -> None:
    fig, ax = plt.subplots(figsize=LG_TRAINING_CHART_FIGSIZE, constrained_layout=True)
    iterations = [step.iteration for step in steps]
    ax.plot(iterations, [step.cost for step in steps], color="orange", label="訓練 Cost")
    test_costs = [step.test_cost for step in steps if step.test_cost is not None]
    if len(test_costs) == len(steps):
        ax.plot(iterations, test_costs, color="tab:blue", label="測試 Cost")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J")
    ax.set_title("Cost vs Iteration")
    ax.legend()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_probability_plot(
    actual: pd.Series,
    probability: pd.Series,
    placeholder,
    *,
    threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=LG_TRAINING_CHART_FIGSIZE, constrained_layout=True)
    y = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    p = np.asarray(probability, dtype=float)
    ax.scatter(y, p, color="C0", alpha=0.8)
    ax.axhline(float(threshold), color="red", linestyle="--", label="分類 threshold")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("實際 y")
    ax.set_ylabel("預測機率 ŷ")
    ax.set_title("測試集機率對照")
    ax.legend()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_test_table(actual: pd.Series, probability: pd.Series, threshold: float) -> None:
    predicted = predict_class_from_proba(probability, threshold)
    result = pd.DataFrame(
        {
            "y": actual,
            "ŷ": probability.to_numpy(dtype=float),
            "predicted_class": predicted,
        }
    )
    st.dataframe(result.head(30).style.format({"ŷ": "{:.4f}"}), width="stretch")


def _classification_threshold_slider(stage: str, *, enabled: bool) -> float:
    return float(
        st.slider(
            "分類 threshold（訓練後調整）",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key=f"logistic_threshold_{stage}",
            disabled=not enabled,
            help="圖上決策邊界／contour 維持 ŷ=0.5；threshold 只切測試集表與測試集機率對照虛線。",
        )
    )


def _render_sigmoid_visualization() -> None:
    with st.expander("Sigmoid 函數視覺化", expanded=False):
        st.caption("邏輯迴歸先把特徵線性組合成 z，再經 sigmoid 壓到 0～1，作為屬於類別 1 的機率。")
        highlight_z = float(
            st.slider("在曲線上標示 z", min_value=-8.0, max_value=8.0, value=0.0, step=0.1, key="logistic_sigmoid_highlight_z")
        )
        fig = build_sigmoid_figure(highlight_z=highlight_z)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


def _render_logistic_cost_formula(*, regularized: bool = False) -> None:
    if regularized:
        st.caption("Cost 依 sigmoid 機率 f 計算，另加 λ 對 w 的正則化；與分類 threshold 無關。")
        st.latex(COST_J_LOGISTIC_REG_LATEX)
        st.markdown("導數項")
        st.latex(COST_DJ_DW_LOGISTIC_REG_LATEX)
        st.latex(COST_DJ_DB_LOGISTIC_LATEX)
    else:
        st.caption("Cost 只依 sigmoid 機率 f 計算，與分類 threshold 無關。")
        st.latex(COST_J_LOGISTIC_LATEX)
        st.markdown("導數項")
        st.latex(COST_DJ_DW_LOGISTIC_LATEX)
        st.latex(COST_DJ_DB_LOGISTIC_LATEX)
    st.markdown("梯度下降演算法")
    st.latex(COST_GD_W_LOGISTIC_LATEX)
    st.latex(COST_GD_B_LOGISTIC_LATEX)


def _boundary_quiz_unlocked() -> bool:
    return quiz.both_boundary_quiz_correct(
        str(st.session_state.get(quiz.SESSION_SIGMOID, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_COST, quiz.PLEASE_SELECT)),
    )


def _poly_quiz_unlocked() -> bool:
    return quiz.both_poly_quiz_correct(
        str(st.session_state.get(quiz.SESSION_MAP, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_LAMBDA, quiz.PLEASE_SELECT)),
    )


def _render_boundary_pretrain_quiz() -> bool:
    st.markdown("##### 訓練前先猜一下")
    agent_ready = True
    sigmoid_choice = st.radio(
        "題1：sigmoid 輸出大致代表什麼？",
        [quiz.PLEASE_SELECT, *quiz.SIGMOID_OPTIONS],
        key=quiz.SESSION_SIGMOID,
    )
    if st.button("Agent 提示", key="logistic_hint_sigmoid", disabled=not agent_ready, width="stretch"):
        _send_boundary_quiz_hint(quiz.QID_SIGMOID)
    sigmoid_ok = quiz.is_sigmoid_correct(str(sigmoid_choice))
    if str(sigmoid_choice) == quiz.PLEASE_SELECT:
        st.caption("先選一項。")
    elif sigmoid_ok:
        st.success("對。")
    else:
        st.error("再想一下。")

    cost_choice = st.radio(
        "題2：訓練時的 Cost 與分類 threshold 的關係比較接近？",
        [quiz.PLEASE_SELECT, *quiz.COST_OPTIONS],
        key=quiz.SESSION_COST,
    )
    if st.button("Agent 提示", key="logistic_hint_cost", disabled=not agent_ready, width="stretch"):
        _send_boundary_quiz_hint(quiz.QID_COST)
    cost_ok = quiz.is_cost_correct(str(cost_choice))
    if str(cost_choice) == quiz.PLEASE_SELECT:
        st.caption("先選一項。")
    elif cost_ok:
        st.success("對。")
    else:
        st.error("再想一下。")

    unlocked = quiz.both_boundary_quiz_correct(str(sigmoid_choice), str(cost_choice))
    st.info(f"進度：{int(sigmoid_ok) + int(cost_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _render_poly_pretrain_quiz() -> bool:
    st.markdown("##### 訓練前先猜一下")
    agent_ready = True
    map_choice = st.radio(
        "題1：為什麼常把兩個檢測分數做成多項式特徵映射？",
        [quiz.PLEASE_SELECT, *quiz.MAP_OPTIONS],
        key=quiz.SESSION_MAP,
    )
    if st.button("Agent 提示", key="logistic_hint_map", disabled=not agent_ready, width="stretch"):
        _send_poly_quiz_hint(quiz.QID_MAP)
    map_ok = quiz.is_map_correct(str(map_choice))
    if str(map_choice) == quiz.PLEASE_SELECT:
        st.caption("先選一項。")
    elif map_ok:
        st.success("對。")
    else:
        st.error("再想一下。")

    lambda_choice = st.radio(
        "題2：λ 變大時，決策邊界／過擬合傾向比較可能？",
        [quiz.PLEASE_SELECT, *quiz.LAMBDA_OPTIONS],
        key=quiz.SESSION_LAMBDA,
    )
    if st.button("Agent 提示", key="logistic_hint_lambda", disabled=not agent_ready, width="stretch"):
        _send_poly_quiz_hint(quiz.QID_LAMBDA)
    lambda_ok = quiz.is_lambda_correct(str(lambda_choice))
    if str(lambda_choice) == quiz.PLEASE_SELECT:
        st.caption("先選一項。")
    elif lambda_ok:
        st.success("對。")
    else:
        st.error("再想一下。")

    unlocked = quiz.both_poly_quiz_correct(str(map_choice), str(lambda_choice))
    st.info(f"進度：{int(map_ok) + int(lambda_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _send_boundary_quiz_hint(qid: str) -> None:
    invoke_data_agent(
        quiz.hint_user_text(qid),
        extra_context=_boundary_quiz_appendix(
            unlocked=_boundary_quiz_unlocked(),
            focus_qid=qid,
        ),
        display_user_text=quiz.hint_display_text(qid),
        skip_working_snapshot=True,
        host_context=_lg_host(),
        page_name=PAGE_TITLE,
    )


def _send_poly_quiz_hint(qid: str) -> None:
    invoke_data_agent(
        quiz.hint_user_text(qid),
        extra_context=_poly_quiz_appendix(
            unlocked=_poly_quiz_unlocked(),
            focus_qid=qid,
        ),
        display_user_text=quiz.hint_display_text(qid),
        skip_working_snapshot=True,
        host_context=_lg_host(),
        page_name=PAGE_TITLE,
    )


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


def _set_agent_context(
    working: pd.DataFrame,
    *,
    state: dict,
    stage: str,
    features: list[str],
    target: str,
    source_label: str,
    quiz_unlocked: bool,
    scale_errors: list[str],
    artifact: ClassificationArtifact | None,
    open_slot: str | None,
    threshold: float | None,
    mapped: bool,
) -> None:
    quiz_note = (
        _boundary_quiz_appendix(unlocked=quiz_unlocked, focus_qid=st.session_state.get(quiz.SESSION_FOCUS_BOUNDARY))
        if stage == STAGE_BOUNDARY
        else _poly_quiz_appendix(unlocked=quiz_unlocked, focus_qid=st.session_state.get(quiz.SESSION_FOCUS_POLY))
    )
    snapshot = build_logistic_page_snapshot(
        stage=stage,
        state=state,
        open_slot=open_slot,
        quiz_unlocked=quiz_unlocked,
        scale_errors=scale_errors,
        artifact_note=_artifact_note(artifact),
        slots_path=_display_path(logistic_slots_path(WORKSPACE_DIR)),
        request_path=_display_path(logistic_train_request_path(WORKSPACE_DIR)),
    )
    stage_label = "線性邊界" if stage == STAGE_BOUNDARY else "多項式與 λ"
    st.session_state[CONTEXT_KEY] = (
        build_classification_agent_context(
            page_name=f"{PAGE_TITLE}（{stage_label}）",
            data_source=source_label,
            features=features,
            target=target,
            learning_rate=state.get("alpha"),
            epochs=state.get("epochs"),
            row_count=len(working),
            artifact=artifact,
            lambda_=state.get("lambda_") if mapped else None,
            map_degree=DEFAULT_MAP_DEGREE if mapped else None,
            threshold=threshold,
        )
        + "\n"
        + snapshot
        + "\n"
        + quiz_note
    )


def _artifact_note(artifact: ClassificationArtifact | None) -> str:
    if artifact is None:
        return "此學習階段目前沒有訓練結果。"
    return f"此學習階段有訓練結果：B={artifact.intercept:g}，訓練 J={artifact.training_cost:g}。"
