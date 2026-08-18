"""專案展示頁：Challenge 資料檢視＋模型區／成果區空殼。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    WORKSPACE_DIR,
    _display_path,
    render_chat_panel,
    render_dataset_metrics,
    reset_agent_scope_session,
)
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane
from dataset_streamlit_shell.ui.startup_challenge_context import (
    CHALLENGE_ARTIFACT_KEY,
    CHALLENGE_COMPANIES,
    CHALLENGE_SPLIT_SIGNATURE_KEY,
    COMPANY_SWITCH_CANCEL_LABEL,
    COMPANY_SWITCH_CONFIRM_LABEL,
    COMPANY_SWITCH_DIALOG_TITLE,
    available_data_views,
    challenge_host_context,
    challenge_page_snapshot,
    challenge_paths,
    clear_challenge_runtime,
    company_changed_should_reset,
    company_switch_dialog_body,
    csv_for_view,
    default_data_view,
    model_zone_unlocked,
    resolve_company_switch,
    restore_startup_challenge_ui,
    result_zone_unlocked,
    split_signature,
    sync_split_if_working_stale,
)

EMPTY_SHELL_PATH = SHELL_ROOT / "ui" / "startup_challenge_empty_shell.py"
LIVE_UI_PATH = SHELL_ROOT / "ui" / "startup_challenge_ui.py"
COMMITTED_COMPANY_KEY = "challenge_selected_company"
SELECT_COMPANY_KEY = "challenge_company_select"
PENDING_COMPANY_KEY = "challenge_pending_company"
APPLY_COMPANY_KEY = "challenge_apply_company"


def render_model_zone() -> None:
    """有 Challenge 訓練資料與測試資料後，在此寫選型與訓練。不要放成果圖表。"""
    return


def render_result_zone() -> None:
    """有 Challenge 模型產物後，在此寫指標、圖與一次演示。"""
    return


def _column_overview_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "資料型態": [str(df[column].dtype) for column in df.columns],
            "非空值筆數": df.notna().sum(),
            "空值筆數": df.isna().sum(),
            "不同值數量": df.nunique(dropna=True),
        },
        index=df.columns,
    )


def _on_company_change(previous: str | None, company: str, paths) -> None:
    if not company_changed_should_reset(previous, company):
        return
    clear_challenge_runtime(paths)
    st.session_state.pop(CHALLENGE_ARTIFACT_KEY, None)
    st.session_state.pop(CHALLENGE_SPLIT_SIGNATURE_KEY, None)
    restore_startup_challenge_ui(EMPTY_SHELL_PATH, LIVE_UI_PATH)
    reset_agent_scope_session(scope="challenge")
    st.session_state["challenge_data_view"] = default_data_view(paths)
    st.session_state["challenge_company_cleared_notice"] = (
        f"已切換為 **{company}**：工作資料與切分已清除，頁面已還原空殼。"
    )


def _clear_pending_company() -> None:
    st.session_state.pop(PENDING_COMPANY_KEY, None)


def _on_company_select_change() -> None:
    selected = str(st.session_state[SELECT_COMPANY_KEY])
    committed = st.session_state.get(COMMITTED_COMPANY_KEY)
    _, pending = resolve_company_switch(committed, selected)
    if pending:
        st.session_state[PENDING_COMPANY_KEY] = pending
        st.session_state[SELECT_COMPANY_KEY] = committed


@st.dialog(COMPANY_SWITCH_DIALOG_TITLE, on_dismiss=_clear_pending_company)
def _confirm_company_switch(pending: str) -> None:
    st.markdown(company_switch_dialog_body(pending))
    with st.container(horizontal=True):
        if st.button(COMPANY_SWITCH_CONFIRM_LABEL, key="challenge_confirm_switch"):
            st.session_state[APPLY_COMPANY_KEY] = pending
            _clear_pending_company()
            st.rerun()
        if st.button(COMPANY_SWITCH_CANCEL_LABEL, key="challenge_cancel_switch"):
            _clear_pending_company()
            st.rerun()


def _render_data_view(paths) -> None:
    views = available_data_views(paths)
    if not views:
        st.warning("尚未放入該公司的 Challenge 起點 CSV。")
        return
    view_key = "challenge_data_view"
    if view_key not in st.session_state or st.session_state[view_key] not in views:
        st.session_state[view_key] = default_data_view(paths)
    view = str(
        st.radio(
            "資料",
            views,
            horizontal=True,
            key=view_key,
        )
    )
    csv_path = csv_for_view(paths, view)
    if csv_path is None or not csv_path.is_file():
        return
    frame = pd.read_csv(csv_path)
    render_dataset_metrics(frame)
    st.dataframe(_column_overview_frame(frame), width="stretch")
    with st.expander("資料預覽", expanded=True):
        st.dataframe(frame.head(20), width="stretch", hide_index=True)


def _render_zone(title: str, caption: str, *, unlocked: bool, renderer) -> None:
    with st.container(border=True):
        st.markdown(f"##### {title}")
        st.caption(caption)
        if unlocked:
            renderer()


def render_startup_challenge_page() -> None:
    paths_probe = challenge_paths(WORKSPACE_DIR, CHALLENGE_COMPANIES[0])
    paths_probe.challenge_dir.mkdir(parents=True, exist_ok=True)

    previous = st.session_state.get(COMMITTED_COMPANY_KEY)
    apply_company = st.session_state.pop(APPLY_COMPANY_KEY, None)
    if apply_company in CHALLENGE_COMPANIES:
        paths = challenge_paths(WORKSPACE_DIR, apply_company)
        _on_company_change(previous, apply_company, paths)
        st.session_state[COMMITTED_COMPANY_KEY] = apply_company
        st.session_state[SELECT_COMPANY_KEY] = apply_company

    company = st.session_state.get(COMMITTED_COMPANY_KEY, CHALLENGE_COMPANIES[0])
    if company not in CHALLENGE_COMPANIES:
        company = CHALLENGE_COMPANIES[0]
        st.session_state[COMMITTED_COMPANY_KEY] = company

    teaching, agent = open_content_dual_pane()

    with teaching:
        st.title("專案展示")
        pending = st.session_state.get(PENDING_COMPANY_KEY)
        if pending:
            _confirm_company_switch(str(pending))
        selected = str(
            st.selectbox(
                "挑戰公司",
                options=list(CHALLENGE_COMPANIES),
                key=SELECT_COMPANY_KEY,
                on_change=_on_company_select_change,
            )
        )
        company, pending = resolve_company_switch(
            st.session_state.get(COMMITTED_COMPANY_KEY),
            selected,
            st.session_state.get(PENDING_COMPANY_KEY),
        )
        if pending:
            st.session_state[PENDING_COMPANY_KEY] = pending
        else:
            st.session_state.pop(PENDING_COMPANY_KEY, None)
        st.session_state[COMMITTED_COMPANY_KEY] = company
        paths = challenge_paths(WORKSPACE_DIR, company)

        notice = st.session_state.pop("challenge_company_cleared_notice", None)
        if notice:
            st.warning(notice)

        if sync_split_if_working_stale(paths):
            st.session_state.pop(CHALLENGE_ARTIFACT_KEY, None)
            st.session_state.pop(CHALLENGE_SPLIT_SIGNATURE_KEY, None)

        current_sig = split_signature(paths)
        stored_sig = st.session_state.get(CHALLENGE_SPLIT_SIGNATURE_KEY)
        if current_sig is None or (
            stored_sig is not None and stored_sig != current_sig
        ):
            st.session_state.pop(CHALLENGE_ARTIFACT_KEY, None)
        st.session_state[CHALLENGE_SPLIT_SIGNATURE_KEY] = current_sig

        _render_data_view(paths)

        artifact_present = bool(st.session_state.get(CHALLENGE_ARTIFACT_KEY))
        _render_zone(
            "模型區",
            "這裡之後放你們要展示的模型。",
            unlocked=model_zone_unlocked(paths),
            renderer=render_model_zone,
        )
        _render_zone(
            "成果區",
            "這裡之後放訓練後的成果。",
            unlocked=result_zone_unlocked(paths, artifact_present=artifact_present),
            renderer=render_result_zone,
        )

    paths = challenge_paths(WORKSPACE_DIR, company)
    host = challenge_host_context(
        company=company,
        start_csv=_display_path(paths.start_csv),
        handbook=_display_path(paths.handbook),
        working_csv=_display_path(paths.working_csv),
        train_csv=_display_path(paths.train_csv),
        test_csv=_display_path(paths.test_csv),
        scripts_dir=_display_path(SHELL_ROOT / "scripts"),
    )
    snapshot = challenge_page_snapshot(
        company=company,
        start_exists=paths.start_csv.is_file(),
        working_exists=paths.working_csv.is_file(),
        train_exists=paths.train_csv.is_file(),
        test_exists=paths.test_csv.is_file(),
        artifact_present=bool(st.session_state.get(CHALLENGE_ARTIFACT_KEY)),
    )

    with agent:
        render_chat_panel(
            extra_context=snapshot,
            page_name="專案展示",
            host_context=host,
            agent_scope="challenge",
            skip_working_snapshot=True,
        )
