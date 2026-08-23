"""線性回歸教學頁：決策槽列、Agent 寫入、回歸線與 Cost。"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.ml.regression import (
    GradientDescentStep,
    LinearModelArtifact,
    MULTIPLE_REGRESSION_FEATURES,
    MULTIPLE_REGRESSION_TARGET,
    SIMPLE_REGRESSION_FEATURE,
    SIMPLE_REGRESSION_TARGET,
    apply_feature_scaler,
    build_regression_agent_context,
    create_feature_scaler,
    gradient_descent_steps,
    predict_from_artifact,
    predict_line_on_original_x,
    predict_with_parameters,
)
from dataset_streamlit_shell.plotting import configure_matplotlib_for_traditional_chinese
from dataset_streamlit_shell.ui import multiple_regression_quiz as multi_quiz
from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    WORKSPACE_DIR,
    _display_path,
    invoke_data_agent,
    render_chat_panel,
)
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane
from dataset_streamlit_shell.ui.lr_slot_state import (
    DATA_PREVIEW_ROWS,
    SLOT_IDS,
    SLOT_TITLES,
    STAGE_MULTIPLE,
    STAGE_SIMPLE,
    build_lr_page_snapshot,
    can_write_train_request,
    consume_train_request,
    empty_workspace_state,
    load_workspace_state,
    lr_slots_path,
    lr_train_request_path,
    model_code_preview,
    scale_method_errors,
    should_rerun_after_lr_chat,
    slots_file_mtime,
    slot_button_label,
    slot_inspect_rows,
    slot_is_filled,
    slot_signature,
    slots_are_complete,
    train_request_is_set,
)
from dataset_streamlit_shell.ui.simple_regression_quiz import (
    ALPHA_OPTIONS,
    PLEASE_SELECT,
    QID_ALPHA,
    QID_SLOPE,
    SESSION_ALPHA,
    SESSION_FOCUS,
    SESSION_PAIR,
    SESSION_SLOPE,
    SLOPE_OPTIONS,
    both_quiz_correct,
    build_quiz_agent_appendix,
    build_regression_frame_hint_summary,
    can_send_hint,
    expected_slope_direction,
    hint_display_text,
    hint_user_text,
    is_alpha_correct,
    is_slope_correct,
    needs_quiz_reset,
    quiz_choice_status,
    quiz_identity,
)
from dataset_streamlit_shell.ui.teaching_flow import live_fit_caption

configure_matplotlib_for_traditional_chinese()

REGRESSION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "regression"
RESTAURANT_PROFIT_PATH = REGRESSION_DEMO_DIR / "restaurant_profit.csv"
HOUSE_PRICES_PATH = REGRESSION_DEMO_DIR / "house_prices.csv"

LR_PAGE_TITLE = "線性回歸"
LR_CONTEXT_KEY = f"{LR_PAGE_TITLE}_agent_context"
LR_STAGE_SIMPLE = "單變量"
LR_STAGE_MULTIPLE = "多變量"
SIMPLE_SOURCE_LABEL = "內建範例資料：城市人口與餐廳獲利"
MULTIPLE_SOURCE_LABEL = "內建範例資料：房價預測"

WORKSPACE_KEY = "lr_workspace"
MTIME_KEY = "lr_slots_applied_mtime"
INSPECT_KEY = "lr_inspect_slot"

DECISION_SLOT_CSS = """
<style>
.lr-slot-row { margin: 0.2rem 0 0.55rem; }
[class*="st-key-lr_inspect_"] {
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
[class*="st-key-lr_slotbox_"][class*="_done"] button {
  border: 2px solid #3dd68c !important;
}
[class*="st-key-lr_slotbox_"][class*="_todo"] button {
  border: 2px solid #e85d5d !important;
}
</style>
"""


def render_linear_regression_page() -> None:
    workspace = _sync_workspace()
    teaching, agent = open_content_dual_pane()
    with teaching:
        st.title(LR_PAGE_TITLE)
        st.caption(
            "框名是決策槽，框上是目前選擇。請資料 Agent 欄組模型；"
            "點框看只讀選擇明細。訓練畫面只有回歸線與 Cost。"
        )
        st.markdown(DECISION_SLOT_CSS, unsafe_allow_html=True)
        stage_label = st.radio(
            "學習階段",
            [LR_STAGE_SIMPLE, LR_STAGE_MULTIPLE],
            horizontal=True,
            key="lr_learning_stage",
        )
        stage = STAGE_SIMPLE if stage_label == LR_STAGE_SIMPLE else STAGE_MULTIPLE
        if stage == STAGE_SIMPLE:
            _render_simple_stage(workspace[STAGE_SIMPLE])
        else:
            _render_multiple_stage(workspace[STAGE_MULTIPLE])
    with agent:
        extra = str(st.session_state.get(LR_CONTEXT_KEY, f"目前頁面：{LR_PAGE_TITLE}。"))
        render_chat_panel(
            extra_context=extra,
            page_name=LR_PAGE_TITLE,
            after_reply=_after_lr_chat,
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
        STAGE_SIMPLE: "simple_regression_last_artifact",
        STAGE_MULTIPLE: "multiple_regression_last_artifact",
    }
    for stage, result_key in mapping.items():
        if slot_signature(previous.get(stage, {})) != slot_signature(workspace.get(stage, {})):
            st.session_state.pop(result_key, None)
            anim_key = (
                "simple_regression_anim"
                if stage == STAGE_SIMPLE
                else "multiple_regression_anim"
            )
            st.session_state.pop(anim_key, None)


def _after_lr_chat() -> None:
    requested = train_request_is_set(WORKSPACE_DIR)
    applied = float(st.session_state.get(MTIME_KEY, -1.0))
    if should_rerun_after_lr_chat(
        requested=requested,
        slots_mtime=slots_file_mtime(WORKSPACE_DIR),
        applied_mtime=applied,
    ):
        st.rerun()


def _render_slot_row(state: dict, *, stage: str) -> str | None:
    inspect_key = f"{INSPECT_KEY}_{stage}"
    current = st.session_state.get(inspect_key)
    cols = st.columns(5)
    for column, slot_id in zip(cols, SLOT_IDS):
        filled = slot_is_filled(slot_id, state)
        status = "done" if filled else "todo"
        with column:
            with st.container(key=f"lr_slotbox_{stage}_{slot_id}_{status}"):
                if st.button(
                    slot_button_label(slot_id, state, stage),
                    key=f"lr_slot_{stage}_{slot_id}",
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
        st.caption("點上面的框看目前選擇。只讀；要換縮放或 α／epochs，跟資料 Agent 欄說。")
        return
    row_count = len(frame) if frame is not None else None
    rows = slot_inspect_rows(open_slot, state, stage=stage, row_count=row_count)
    items = []
    for key, value in rows:
        cls = ' class="formula"' if key == "公式" else ""
        items.append(f"<dt>{key}</dt><dd{cls}>{value}</dd>")
    with st.container(key=f"lr_inspect_{stage}_{open_slot}"):
        st.markdown(
            f'<aside class="lr-inspect"><h3>{SLOT_TITLES[open_slot]}</h3>'
            f"<dl>{''.join(items)}</dl></aside>",
            unsafe_allow_html=True,
        )
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


def _training_frame(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    return df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()


def _animation_steps(steps: list[GradientDescentStep]) -> list[GradientDescentStep]:
    if len(steps) <= 80:
        return steps
    stride = max(len(steps) // 80, 1)
    selected = steps[::stride]
    if selected[-1] != steps[-1]:
        selected.append(steps[-1])
    return selected


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


def _render_simple_stage(state: dict) -> None:
    df = pd.read_csv(RESTAURANT_PROFIT_PATH)
    source_label = SIMPLE_SOURCE_LABEL
    feature = SIMPLE_REGRESSION_FEATURE
    target = SIMPLE_REGRESSION_TARGET
    if feature not in df.columns or target not in df.columns:
        st.warning("內建餐廳資料缺少固定欄位，無法開始單變量線性回歸。")
        return
    working = _training_frame(df, [feature], target)
    if len(working) < 2:
        st.warning("可用樣本少於 2 筆，無法訓練線性回歸。")
        return

    anim_key = "simple_regression_anim"
    result_key = "simple_regression_last_artifact"
    expected_slope = expected_slope_direction(working, feature, target)
    slope_choice = str(st.session_state.get(SESSION_SLOPE, PLEASE_SELECT))
    alpha_choice = str(st.session_state.get(SESSION_ALPHA, PLEASE_SELECT))
    quiz_unlocked = both_quiz_correct(
        slope_choice,
        alpha_choice,
        expected_slope=expected_slope,
    )
    scale_errors = scale_method_errors(
        state["choices"].get("scale"),
        working,
        [feature],
    )
    signature = slot_signature(state)
    stored = st.session_state.get(result_key)
    artifact = None
    if isinstance(stored, dict) and stored.get("signature") == signature:
        artifact = stored["artifact"]

    open_slot = _render_slot_row(state, stage=STAGE_SIMPLE)
    _render_inspect(
        state,
        stage=STAGE_SIMPLE,
        open_slot=open_slot,
        frame=working,
        features=[feature],
        target=target,
    )
    _render_code_preview(state, stage=STAGE_SIMPLE)
    quiz_unlocked = _render_simple_quiz(
        working,
        feature=feature,
        target=target,
        learning_rate=float(state["alpha"] or 0.01),
        expected_slope=expected_slope,
        source_label=source_label,
        epochs=int(state["epochs"] or 1500),
    )

    def _start() -> None:
        _queue_training(
            working,
            features=[feature],
            target=target,
            state=state,
            source_label=source_label,
            result_key=result_key,
            anim_key=anim_key,
            model_kind="simple_linear_regression",
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
        key="train_simple_regression",
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
    line_placeholder = chart_left.empty()
    cost_placeholder = chart_right.empty()
    status_placeholder = st.empty()

    if training_active:
        _run_simple_training(
            working,
            feature=feature,
            target=target,
            result_key=result_key,
            anim_key=anim_key,
            line_placeholder=line_placeholder,
            cost_placeholder=cost_placeholder,
            status_placeholder=status_placeholder,
        )
        stored = st.session_state.get(result_key)
        if isinstance(stored, dict) and stored.get("signature") == signature:
            artifact = stored["artifact"]
        anim = st.session_state.get(anim_key)
        training_active = isinstance(anim, dict) and not anim.get("finished", True)

    if not training_active:
        if artifact is not None:
            _show_simple_result(
                working,
                feature=feature,
                target=target,
                artifact=artifact,
                result_key=result_key,
                line_placeholder=line_placeholder,
                cost_placeholder=cost_placeholder,
                status_placeholder=status_placeholder,
            )
        else:
            status_placeholder.caption("訓練後這裡只會出現回歸線與 Cost。")

    _set_simple_agent_context(
        working,
        state=state,
        feature=feature,
        target=target,
        source_label=source_label,
        quiz_unlocked=quiz_unlocked,
        scale_errors=scale_errors,
        artifact=artifact,
        open_slot=open_slot,
        expected_slope=expected_slope,
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
    model_kind: str,
) -> None:
    method = str(state["choices"]["scale"])
    scaler = create_feature_scaler(working, features, method)
    scaled = apply_feature_scaler(working, scaler)
    learning_rate = float(state["alpha"])
    epochs = int(state["epochs"])
    steps = gradient_descent_steps(
        scaled,
        working[target],
        learning_rate=learning_rate,
        epochs=epochs,
    )
    st.session_state[anim_key] = {
        "steps": steps,
        "sampled": _animation_steps(steps),
        "finished": False,
        "signature": slot_signature(state),
        "scaler": scaler,
        "source_label": source_label,
        "model_kind": model_kind,
        "features": list(features),
        "target": target,
        "scaled": scaled,
    }
    st.session_state.pop(result_key, None)


def _run_simple_training(
    working: pd.DataFrame,
    *,
    feature: str,
    target: str,
    result_key: str,
    anim_key: str,
    line_placeholder,
    cost_placeholder,
    status_placeholder,
) -> None:
    anim = st.session_state.get(anim_key)
    if not isinstance(anim, dict) or anim.get("finished"):
        return
    steps: list[GradientDescentStep] = list(anim["steps"])
    sampled: list[GradientDescentStep] = list(anim["sampled"])
    scaler = anim["scaler"]
    if not sampled:
        anim["finished"] = True
        return

    for step in sampled:
        caption = live_fit_caption(
            iteration=step.iteration,
            total_iterations=steps[-1].iteration,
            weights=step.weights,
            intercept=step.intercept,
            cost=step.cost,
        )
        _render_simple_step_plot(
            working,
            feature,
            target,
            step,
            line_placeholder,
            scaler=scaler,
        )
        history = [item for item in steps if item.iteration <= step.iteration]
        _render_cost_history_plot(history, cost_placeholder)
        status_placeholder.caption(caption)
        time.sleep(0.015)

    final_step = steps[-1]
    artifact = LinearModelArtifact(
        model_kind="simple_linear_regression",
        features=[feature],
        target=target,
        weights=[float(final_step.weights[0])],
        intercept=float(final_step.intercept),
        scaler=scaler,
        training_cost=float(final_step.cost),
        data_source=str(anim["source_label"]),
    )
    st.session_state[result_key] = {
        "signature": anim["signature"],
        "artifact": artifact,
        "steps": steps,
    }
    anim["finished"] = True


def _show_simple_result(
    working: pd.DataFrame,
    *,
    feature: str,
    target: str,
    artifact: LinearModelArtifact,
    result_key: str,
    line_placeholder,
    cost_placeholder,
    status_placeholder,
) -> None:
    if artifact.scaler is None:
        status_placeholder.warning("訓練結果缺少 scaler，請重新訓練。")
        return
    _render_simple_step_plot(
        working,
        feature,
        target,
        GradientDescentStep(
            iteration=0,
            weights=[float(w) for w in artifact.weights],
            intercept=float(artifact.intercept),
            cost=float(artifact.training_cost),
        ),
        line_placeholder,
        scaler=artifact.scaler,
    )
    result_bundle = st.session_state.get(result_key)
    cached_steps = result_bundle.get("steps") if isinstance(result_bundle, dict) else None
    if isinstance(cached_steps, list) and cached_steps:
        _render_cost_history_plot(cached_steps, cost_placeholder)
    status_placeholder.caption(
        live_fit_caption(
            iteration=int(cached_steps[-1].iteration) if cached_steps else 0,
            total_iterations=int(cached_steps[-1].iteration) if cached_steps else 0,
            weights=artifact.weights,
            intercept=float(artifact.intercept),
            cost=float(artifact.training_cost),
        )
    )


def _render_multiple_stage(state: dict) -> None:
    df = pd.read_csv(HOUSE_PRICES_PATH)
    source_label = MULTIPLE_SOURCE_LABEL
    selected_features = list(MULTIPLE_REGRESSION_FEATURES)
    target = MULTIPLE_REGRESSION_TARGET
    missing = [column for column in selected_features + [target] if column not in df.columns]
    if missing:
        st.warning(f"內建房價資料缺少固定欄位：{', '.join(missing)}。")
        return
    working = _training_frame(df, selected_features, target)
    if len(working) < 2:
        st.warning("可用樣本少於 2 筆，無法訓練線性回歸。")
        return

    anim_key = "multiple_regression_anim"
    result_key = "multiple_regression_last_artifact"
    purpose_choice = str(st.session_state.get(multi_quiz.SESSION_PURPOSE, multi_quiz.PLEASE_SELECT))
    weights_choice = str(st.session_state.get(multi_quiz.SESSION_WEIGHTS, multi_quiz.PLEASE_SELECT))
    quiz_unlocked = multi_quiz.both_quiz_correct(purpose_choice, weights_choice)
    scale_errors = scale_method_errors(
        state["choices"].get("scale"),
        working,
        selected_features,
    )
    signature = slot_signature(state)
    stored = st.session_state.get(result_key)
    artifact = None
    if isinstance(stored, dict) and stored.get("signature") == signature:
        artifact = stored["artifact"]

    open_slot = _render_slot_row(state, stage=STAGE_MULTIPLE)
    _render_inspect(
        state,
        stage=STAGE_MULTIPLE,
        open_slot=open_slot,
        frame=working,
        features=selected_features,
        target=target,
    )
    _render_code_preview(state, stage=STAGE_MULTIPLE)
    quiz_unlocked = _render_multiple_quiz(
        features=selected_features,
        target=target,
        learning_rate=float(state["alpha"] or 0.1),
        source_label=source_label,
        epochs=int(state["epochs"] or 1000),
        row_count=len(working),
    )

    def _start() -> None:
        _queue_training(
            working,
            features=selected_features,
            target=target,
            state=state,
            source_label=source_label,
            result_key=result_key,
            anim_key=anim_key,
            model_kind="multiple_linear_regression",
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
        key="train_multiple_regression",
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
    pred_placeholder = chart_left.empty()
    cost_placeholder = chart_right.empty()
    status_placeholder = st.empty()

    if training_active:
        _run_multiple_training(
            working,
            target=target,
            result_key=result_key,
            anim_key=anim_key,
            pred_placeholder=pred_placeholder,
            cost_placeholder=cost_placeholder,
            status_placeholder=status_placeholder,
        )
        stored = st.session_state.get(result_key)
        if isinstance(stored, dict) and stored.get("signature") == signature:
            artifact = stored["artifact"]
        anim = st.session_state.get(anim_key)
        training_active = isinstance(anim, dict) and not anim.get("finished", True)

    if not training_active:
        if artifact is not None:
            prediction = predict_from_artifact(artifact, working[artifact.features])
            _render_actual_prediction_plot(working[target], prediction, target, pred_placeholder)
            result_bundle = st.session_state.get(result_key)
            cached_steps = (
                result_bundle.get("steps") if isinstance(result_bundle, dict) else None
            )
            if isinstance(cached_steps, list) and cached_steps:
                _render_cost_history_plot(cached_steps, cost_placeholder)
            status_placeholder.caption(
                live_fit_caption(
                    iteration=int(cached_steps[-1].iteration) if cached_steps else 0,
                    total_iterations=int(cached_steps[-1].iteration) if cached_steps else 0,
                    weights=artifact.weights,
                    intercept=float(artifact.intercept),
                    cost=float(artifact.training_cost),
                )
            )
        else:
            status_placeholder.caption("訓練後這裡只會出現預測對照與 Cost。")

    _set_multiple_agent_context(
        working,
        state=state,
        features=selected_features,
        target=target,
        source_label=source_label,
        quiz_unlocked=quiz_unlocked,
        scale_errors=scale_errors,
        artifact=artifact,
        open_slot=open_slot,
    )


def _run_multiple_training(
    working: pd.DataFrame,
    *,
    target: str,
    result_key: str,
    anim_key: str,
    pred_placeholder,
    cost_placeholder,
    status_placeholder,
) -> None:
    anim = st.session_state.get(anim_key)
    if not isinstance(anim, dict) or anim.get("finished"):
        return
    steps: list[GradientDescentStep] = list(anim["steps"])
    sampled: list[GradientDescentStep] = list(anim["sampled"])
    scaled = anim["scaled"]
    features = list(anim["features"])
    scaler = anim["scaler"]
    if not sampled:
        anim["finished"] = True
        return

    for step in sampled:
        caption = live_fit_caption(
            iteration=step.iteration,
            total_iterations=steps[-1].iteration,
            weights=step.weights,
            intercept=step.intercept,
            cost=step.cost,
        )
        prediction = predict_with_parameters(scaled, step.weights, step.intercept)
        _render_actual_prediction_plot(working[target], prediction, target, pred_placeholder)
        history = [item for item in steps if item.iteration <= step.iteration]
        _render_cost_history_plot(history, cost_placeholder)
        status_placeholder.caption(caption)
        time.sleep(0.015)

    final_step = steps[-1]
    artifact = LinearModelArtifact(
        model_kind="multiple_linear_regression",
        features=features,
        target=target,
        weights=[float(value) for value in final_step.weights],
        intercept=float(final_step.intercept),
        scaler=scaler,
        training_cost=float(final_step.cost),
        data_source=str(anim["source_label"]),
    )
    st.session_state[result_key] = {
        "signature": anim["signature"],
        "artifact": artifact,
        "steps": steps,
    }
    anim["finished"] = True


def _render_simple_step_plot(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    step: GradientDescentStep,
    placeholder,
    *,
    scaler: dict,
) -> None:
    x_values = frame[feature]
    line_x = np.linspace(float(x_values.min()), float(x_values.max()), 100)
    line_y = predict_line_on_original_x(
        line_x,
        weight=float(step.weights[0]),
        intercept=float(step.intercept),
        feature=feature,
        scaler=scaler,
    )
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.scatter(frame[feature], frame[target], alpha=0.75, label="資料點")
    ax.plot(line_x, line_y, color="red", label="回歸線")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"回歸線（iteration {step.iteration}）")
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


def _render_actual_prediction_plot(
    actual: pd.Series,
    prediction: pd.Series,
    target: str,
    placeholder,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    ax.scatter(actual, prediction, alpha=0.75)
    lower = float(min(actual.min(), prediction.min()))
    upper = float(max(actual.max(), prediction.max()))
    ax.plot([lower, upper], [lower, upper], color="red", linestyle="--", label="完全預測正確")
    ax.set_xlabel(f"實際 {target}")
    ax.set_ylabel(f"預測 {target}")
    ax.set_title("實際值 vs 預測值")
    ax.legend()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _set_simple_agent_context(
    working: pd.DataFrame,
    *,
    state: dict,
    feature: str,
    target: str,
    source_label: str,
    quiz_unlocked: bool,
    scale_errors: list[str],
    artifact: LinearModelArtifact | None,
    open_slot: str | None,
    expected_slope: str,
) -> None:
    slope_choice = str(st.session_state.get(SESSION_SLOPE, PLEASE_SELECT))
    alpha_choice = str(st.session_state.get(SESSION_ALPHA, PLEASE_SELECT))
    quiz_note = build_quiz_agent_appendix(
        slope_status=quiz_choice_status(
            slope_choice,
            correct=is_slope_correct(slope_choice, expected_slope),
        ),
        alpha_status=quiz_choice_status(
            alpha_choice,
            correct=is_alpha_correct(alpha_choice),
        ),
        focus_qid=st.session_state.get(SESSION_FOCUS),
        feature=feature,
        target=target,
        learning_rate=float(state["alpha"] or 0.01),
        unlocked=quiz_unlocked,
    )
    snapshot = build_lr_page_snapshot(
        stage=STAGE_SIMPLE,
        state=state,
        open_slot=open_slot,
        quiz_unlocked=quiz_unlocked,
        scale_errors=scale_errors,
        artifact_note=_artifact_note(artifact),
        slots_path=_display_path(lr_slots_path(WORKSPACE_DIR)),
        request_path=_display_path(lr_train_request_path(WORKSPACE_DIR)),
    )
    st.session_state[LR_CONTEXT_KEY] = (
        build_regression_agent_context(
            page_name=f"{LR_PAGE_TITLE}（{LR_STAGE_SIMPLE}）",
            data_source=source_label,
            features=[feature],
            target=target,
            learning_rate=state["alpha"],
            epochs=state["epochs"],
            row_count=len(working),
            artifact=artifact,
            prompt_train=quiz_unlocked and slots_are_complete(state),
        )
        + "\n"
        + snapshot
        + "\n"
        + quiz_note
    )


def _set_multiple_agent_context(
    working: pd.DataFrame,
    *,
    state: dict,
    features: list[str],
    target: str,
    source_label: str,
    quiz_unlocked: bool,
    scale_errors: list[str],
    artifact: LinearModelArtifact | None,
    open_slot: str | None,
) -> None:
    purpose_choice = str(st.session_state.get(multi_quiz.SESSION_PURPOSE, multi_quiz.PLEASE_SELECT))
    weights_choice = str(st.session_state.get(multi_quiz.SESSION_WEIGHTS, multi_quiz.PLEASE_SELECT))
    quiz_note = multi_quiz.build_quiz_agent_appendix(
        purpose_status=multi_quiz.quiz_choice_status(
            purpose_choice,
            correct=multi_quiz.is_purpose_correct(purpose_choice),
        ),
        weights_status=multi_quiz.quiz_choice_status(
            weights_choice,
            correct=multi_quiz.is_weights_correct(weights_choice),
        ),
        focus_qid=st.session_state.get(multi_quiz.SESSION_FOCUS),
        features=features,
        target=target,
        learning_rate=float(state["alpha"] or 0.1),
        unlocked=quiz_unlocked,
        use_housing_example="ready.csv" not in source_label,
    )
    snapshot = build_lr_page_snapshot(
        stage=STAGE_MULTIPLE,
        state=state,
        open_slot=open_slot,
        quiz_unlocked=quiz_unlocked,
        scale_errors=scale_errors,
        artifact_note=_artifact_note(artifact),
        slots_path=_display_path(lr_slots_path(WORKSPACE_DIR)),
        request_path=_display_path(lr_train_request_path(WORKSPACE_DIR)),
    )
    st.session_state[LR_CONTEXT_KEY] = (
        build_regression_agent_context(
            page_name=f"{LR_PAGE_TITLE}（{LR_STAGE_MULTIPLE}）",
            data_source=source_label,
            features=features,
            target=target,
            learning_rate=state["alpha"],
            epochs=state["epochs"],
            row_count=len(working),
            artifact=artifact,
            prompt_train=quiz_unlocked and slots_are_complete(state),
        )
        + "\n"
        + snapshot
        + "\n"
        + quiz_note
    )


def _artifact_note(artifact: LinearModelArtifact | None) -> str:
    if artifact is None:
        return "此學習階段目前沒有訓練結果。"
    return (
        f"此學習階段有訓練結果：B={artifact.intercept:g}，"
        f"J={artifact.training_cost:g}。"
    )


def _reset_simple_quiz() -> None:
    st.session_state[SESSION_SLOPE] = PLEASE_SELECT
    st.session_state[SESSION_ALPHA] = PLEASE_SELECT
    st.session_state[SESSION_FOCUS] = QID_SLOPE


def _simple_quiz_extra(
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    learning_rate: float,
    epochs: int,
    source_label: str,
    expected_slope: str,
) -> str:
    slope_choice = str(st.session_state.get(SESSION_SLOPE, PLEASE_SELECT))
    alpha_choice = str(st.session_state.get(SESSION_ALPHA, PLEASE_SELECT))
    unlocked = both_quiz_correct(
        slope_choice,
        alpha_choice,
        expected_slope=expected_slope,
    )
    workspace = st.session_state.get(WORKSPACE_KEY) or empty_workspace_state()
    can_train = unlocked and slots_are_complete(workspace[STAGE_SIMPLE])
    appendix = build_quiz_agent_appendix(
        slope_status=quiz_choice_status(
            slope_choice,
            correct=is_slope_correct(slope_choice, expected_slope),
        ),
        alpha_status=quiz_choice_status(
            alpha_choice,
            correct=is_alpha_correct(alpha_choice),
        ),
        focus_qid=st.session_state.get(SESSION_FOCUS),
        feature=feature,
        target=target,
        learning_rate=learning_rate,
        unlocked=unlocked,
    )
    return (
        build_regression_agent_context(
            page_name="單變量線性回歸",
            data_source=source_label,
            features=[feature],
            target=target,
            learning_rate=learning_rate,
            epochs=epochs,
            row_count=len(frame),
            artifact=None,
            prompt_train=can_train,
        )
        + "\n"
        + build_regression_frame_hint_summary(frame, feature, target)
        + "\n"
        + appendix
    )


def _send_simple_hint(
    qid: str,
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    learning_rate: float,
    epochs: int,
    source_label: str,
    expected_slope: str,
) -> None:
    ts_key = f"simple_reg_hint_ts_{qid}"
    now = time.time()
    if not can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示發送中，請稍候再按。")
        return
    if not st.session_state.get("data_agent_connected"):
        st.warning("請先在右側啟用資料 Agent，再按「Agent 提示」。")
        return
    st.session_state[SESSION_FOCUS] = qid
    st.session_state[ts_key] = now
    extra = _simple_quiz_extra(
        frame,
        feature=feature,
        target=target,
        learning_rate=learning_rate,
        epochs=epochs,
        source_label=source_label,
        expected_slope=expected_slope,
    )
    with st.spinner("正在詢問 Agent…"):
        invoke_data_agent(
            hint_user_text(
                qid,
                feature=feature,
                target=target,
                learning_rate=learning_rate,
            ),
            extra_context=extra,
            display_user_text=hint_display_text(qid),
        )
    st.rerun()


def _render_simple_quiz(
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    learning_rate: float,
    expected_slope: str,
    source_label: str,
    epochs: int,
) -> bool:
    identity = quiz_identity(
        feature,
        target,
        source_label=source_label,
        frame=frame,
    )
    if needs_quiz_reset(st.session_state.get(SESSION_PAIR), identity):
        _reset_simple_quiz()
    st.session_state[SESSION_PAIR] = identity
    if SESSION_SLOPE not in st.session_state:
        st.session_state[SESSION_SLOPE] = PLEASE_SELECT
    if SESSION_ALPHA not in st.session_state:
        st.session_state[SESSION_ALPHA] = PLEASE_SELECT
    if SESSION_FOCUS not in st.session_state:
        st.session_state[SESSION_FOCUS] = QID_SLOPE

    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        slope_choice = st.radio(
            "題1：依目前散點，擬合後的斜率 w 比較可能是？",
            [PLEASE_SELECT, *SLOPE_OPTIONS],
            key=SESSION_SLOPE,
            horizontal=True,
        )
    with h1_col:
        st.write("")
        if st.button(
            "Agent 提示",
            key="simple_reg_hint_slope",
            disabled=not agent_ready,
            width="stretch",
            help="一鍵請資料 Agent 欄給斜率方向的觀察线索",
        ):
            _send_simple_hint(
                QID_SLOPE,
                frame,
                feature=feature,
                target=target,
                learning_rate=learning_rate,
                epochs=epochs,
                source_label=source_label,
                expected_slope=expected_slope,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    slope_ok = is_slope_correct(str(slope_choice), expected_slope)
    if str(slope_choice) == PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[SESSION_FOCUS] = QID_SLOPE
    elif slope_ok:
        st.caption("題1 OK，訓練後可用實際的 w 對照。")
    else:
        st.caption("題1 與散點方向不符，可按「Agent 提示」或問資料 Agent 欄。")
        st.session_state[SESSION_FOCUS] = QID_SLOPE

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        alpha_choice = st.radio(
            "題2：若學習率 α 明顯偏大，Cost 曲線比較可能？",
            [PLEASE_SELECT, *ALPHA_OPTIONS],
            key=SESSION_ALPHA,
            horizontal=True,
        )
    with h2_col:
        st.write("")
        if st.button(
            "Agent 提示",
            key="simple_reg_hint_alpha",
            disabled=not agent_ready,
            width="stretch",
            help="一鍵請資料 Agent 欄給 α 與 Cost 的线索",
        ):
            _send_simple_hint(
                QID_ALPHA,
                frame,
                feature=feature,
                target=target,
                learning_rate=learning_rate,
                epochs=epochs,
                source_label=source_label,
                expected_slope=expected_slope,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    alpha_ok = is_alpha_correct(str(alpha_choice))
    if str(alpha_choice) == PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if slope_ok:
            st.session_state[SESSION_FOCUS] = QID_ALPHA
    elif alpha_ok:
        st.caption("題2 OK。")
        if not slope_ok:
            st.session_state[SESSION_FOCUS] = QID_SLOPE
    else:
        st.caption("題2 再想想 α 對更新步長的影響，可按「Agent 提示」。")
        st.session_state[SESSION_FOCUS] = QID_ALPHA

    unlocked = both_quiz_correct(
        str(slope_choice),
        str(alpha_choice),
        expected_slope=expected_slope,
    )
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(slope_ok) + int(alpha_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _reset_multiple_quiz() -> None:
    st.session_state[multi_quiz.SESSION_PURPOSE] = multi_quiz.PLEASE_SELECT
    st.session_state[multi_quiz.SESSION_WEIGHTS] = multi_quiz.PLEASE_SELECT
    st.session_state[multi_quiz.SESSION_FOCUS] = multi_quiz.QID_PURPOSE


def _multiple_quiz_extra(
    *,
    features: list[str],
    target: str,
    learning_rate: float,
    epochs: int,
    row_count: int,
    source_label: str,
) -> str:
    purpose_choice = str(st.session_state.get(multi_quiz.SESSION_PURPOSE, multi_quiz.PLEASE_SELECT))
    weights_choice = str(st.session_state.get(multi_quiz.SESSION_WEIGHTS, multi_quiz.PLEASE_SELECT))
    unlocked = multi_quiz.both_quiz_correct(purpose_choice, weights_choice)
    workspace = st.session_state.get(WORKSPACE_KEY) or empty_workspace_state()
    can_train = unlocked and slots_are_complete(workspace[STAGE_MULTIPLE])
    appendix = multi_quiz.build_quiz_agent_appendix(
        purpose_status=multi_quiz.quiz_choice_status(
            purpose_choice,
            correct=multi_quiz.is_purpose_correct(purpose_choice),
        ),
        weights_status=multi_quiz.quiz_choice_status(
            weights_choice,
            correct=multi_quiz.is_weights_correct(weights_choice),
        ),
        focus_qid=st.session_state.get(multi_quiz.SESSION_FOCUS),
        features=features,
        target=target,
        learning_rate=learning_rate,
        unlocked=unlocked,
        use_housing_example="ready.csv" not in source_label,
    )
    return (
        build_regression_agent_context(
            page_name="多變量線性回歸",
            data_source=source_label,
            features=features,
            target=target,
            learning_rate=learning_rate,
            epochs=epochs,
            row_count=row_count,
            artifact=None,
            prompt_train=can_train,
        )
        + "\n"
        + appendix
    )


def _send_multiple_hint(
    qid: str,
    *,
    features: list[str],
    target: str,
    learning_rate: float,
    epochs: int,
    row_count: int,
    source_label: str,
) -> None:
    ts_key = f"multiple_reg_hint_ts_{qid}"
    now = time.time()
    if not multi_quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示發送中，請稍候再按。")
        return
    if not st.session_state.get("data_agent_connected"):
        st.warning("請先在右側啟用資料 Agent，再按「Agent 提示」。")
        return
    st.session_state[multi_quiz.SESSION_FOCUS] = qid
    st.session_state[ts_key] = now
    extra = _multiple_quiz_extra(
        features=features,
        target=target,
        learning_rate=learning_rate,
        epochs=epochs,
        row_count=row_count,
        source_label=source_label,
    )
    with st.spinner("正在詢問 Agent…"):
        invoke_data_agent(
            multi_quiz.hint_user_text(qid, features=features, target=target),
            extra_context=extra,
            display_user_text=multi_quiz.hint_display_text(qid),
        )
    st.rerun()


def _render_multiple_quiz(
    *,
    features: list[str],
    target: str,
    learning_rate: float,
    source_label: str,
    epochs: int,
    row_count: int,
) -> bool:
    if multi_quiz.needs_quiz_reset(st.session_state.get(multi_quiz.SESSION_PAIR), features, target):
        _reset_multiple_quiz()
    st.session_state[multi_quiz.SESSION_PAIR] = multi_quiz.pair_key(features, target)
    if multi_quiz.SESSION_PURPOSE not in st.session_state:
        st.session_state[multi_quiz.SESSION_PURPOSE] = multi_quiz.PLEASE_SELECT
    if multi_quiz.SESSION_WEIGHTS not in st.session_state:
        st.session_state[multi_quiz.SESSION_WEIGHTS] = multi_quiz.PLEASE_SELECT
    if multi_quiz.SESSION_FOCUS not in st.session_state:
        st.session_state[multi_quiz.SESSION_FOCUS] = multi_quiz.QID_PURPOSE

    st.markdown("##### 訓練前先猜一下")
    st.caption(
        "兩題都答對後，「開始訓練」才會啟用。"
        "題目聚焦多變量的目的與 w／b 意義；卡住時可按「Agent 提示」。"
    )
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        purpose_choice = st.radio(
            "題1：相對於單變量只用一個 x，多變量的主要目的比較接近？",
            [multi_quiz.PLEASE_SELECT, *multi_quiz.PURPOSE_OPTIONS],
            key=multi_quiz.SESSION_PURPOSE,
        )
    with h1_col:
        st.write("")
        if st.button(
            "Agent 提示",
            key="multiple_reg_hint_purpose",
            disabled=not agent_ready,
            width="stretch",
            help="一鍵請資料 Agent 欄給「為什麼多 feature」的线索",
        ):
            _send_multiple_hint(
                multi_quiz.QID_PURPOSE,
                features=features,
                target=target,
                learning_rate=learning_rate,
                epochs=epochs,
                row_count=row_count,
                source_label=source_label,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    purpose_ok = multi_quiz.is_purpose_correct(str(purpose_choice))
    if str(purpose_choice) == multi_quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[multi_quiz.SESSION_FOCUS] = multi_quiz.QID_PURPOSE
    elif purpose_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想「多個 x 一起做什麼」，可按「Agent 提示」。")
        st.session_state[multi_quiz.SESSION_FOCUS] = multi_quiz.QID_PURPOSE

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        weights_choice = st.radio(
            "題2：訓練完成後，這組 w、b 主要代表？",
            [multi_quiz.PLEASE_SELECT, *multi_quiz.WEIGHTS_OPTIONS],
            key=multi_quiz.SESSION_WEIGHTS,
        )
    with h2_col:
        st.write("")
        if st.button(
            "Agent 提示",
            key="multiple_reg_hint_weights",
            disabled=not agent_ready,
            width="stretch",
            help="一鍵請資料 Agent 欄給 w／b 意義的线索",
        ):
            _send_multiple_hint(
                multi_quiz.QID_WEIGHTS,
                features=features,
                target=target,
                learning_rate=learning_rate,
                epochs=epochs,
                row_count=row_count,
                source_label=source_label,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    weights_ok = multi_quiz.is_weights_correct(str(weights_choice))
    if str(weights_choice) == multi_quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if purpose_ok:
            st.session_state[multi_quiz.SESSION_FOCUS] = multi_quiz.QID_WEIGHTS
    elif weights_ok:
        st.caption("題2 OK。")
        if not purpose_ok:
            st.session_state[multi_quiz.SESSION_FOCUS] = multi_quiz.QID_PURPOSE
    else:
        st.caption("題2 再想想多個 w 與 b 在表達什麼，可按「Agent 提示」。")
        st.session_state[multi_quiz.SESSION_FOCUS] = multi_quiz.QID_WEIGHTS

    unlocked = multi_quiz.both_quiz_correct(str(purpose_choice), str(weights_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(purpose_ok) + int(weights_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked
