"""專案展示頁：白板三塊空殼。成果以 AI coding 改本檔常數／②區塊為準。"""

from __future__ import annotations

import streamlit as st

from dataset_streamlit_shell.ui.data_ui import (
    WORKSPACE_DIR,
    SHELL_ROOT,
    _display_path,
    render_chat_panel,
    reset_agent_scope_session,
)
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane
from dataset_streamlit_shell.ui.startup_challenge_context import (
    CHALLENGE_COMPANIES,
    board_status_summary,
    challenge_host_context,
    challenge_page_snapshot,
    challenge_paths,
    clear_challenge_working,
    company_changed_should_clear_working,
)

# --- 白板①：學生／Agent 以 AI coding 填寫 ---
BOARD_CUSTOMER = ""  # TODO(challenge): 客戶是誰
BOARD_PROBLEM = ""  # TODO(challenge): 要解決的問題
BOARD_TASK_TYPE = ""  # TODO(challenge): 分類／回歸
BOARD_TARGET_COLUMN = ""  # TODO(challenge): 目標欄位名稱（須與 CSV 一致）

# --- 白板②：最小訓練成果（空殼）---
BOARD_MODEL_NAME = ""  # TODO(challenge): 模型名稱
BOARD_METRIC_LINE = ""  # TODO(challenge): 一個評估指標（含數值）

# --- 白板③：限制／倫理 ---
BOARD_LIMITS = ""  # TODO(challenge): 至少 2–3 句紅線（含該公司必講差異點）


def _board_filled(value: str) -> bool:
    return bool(value.strip())


def _render_board_block(title: str, body: str, empty_hint: str) -> None:
    st.markdown(f"### {title}")
    if _board_filled(body):
        st.markdown(body)
    else:
        st.info(empty_hint)


def _on_company_change(previous: str | None, company: str, working_csv) -> None:
    if not company_changed_should_clear_working(previous, company):
        return
    clear_challenge_working(working_csv)
    # 換公司重建 Challenge Agent session；不還原本檔白板常數（定案 Q9=A）
    reset_agent_scope_session(scope="challenge")
    st.session_state["challenge_company_cleared_notice"] = (
        f"已切換為 **{company}**：Challenge 工作資料已清除，Challenge Agent session 已重置。"
        "本頁 UI 程式（白板常數）不會自動還原。"
    )


def render_startup_challenge_page() -> None:
    paths_probe = challenge_paths(WORKSPACE_DIR, CHALLENGE_COMPANIES[0])
    paths_probe.challenge_dir.mkdir(parents=True, exist_ok=True)

    previous = st.session_state.get("challenge_selected_company")
    company = st.session_state.get("challenge_selected_company", CHALLENGE_COMPANIES[0])
    if company not in CHALLENGE_COMPANIES:
        company = CHALLENGE_COMPANIES[0]

    teaching, agent = open_content_dual_pane()

    with teaching:
        st.title("專案展示")
        st.caption("AI Startup Challenge｜成果展示")
        st.write(
            "接上公司資料，完成白板三塊。不要重做整條 ML 教學頁。"
            "成果請透過右側資料 Agent 以 AI coding 寫入本頁允許檔案。"
        )

        company = st.selectbox(
            "挑戰公司",
            options=list(CHALLENGE_COMPANIES),
            index=list(CHALLENGE_COMPANIES).index(company),
            key="challenge_company_select",
        )
        paths = challenge_paths(WORKSPACE_DIR, company)
        _on_company_change(previous, company, paths.working_csv)
        st.session_state["challenge_selected_company"] = company

        notice = st.session_state.pop("challenge_company_cleared_notice", None)
        if notice:
            st.warning(notice)

        start_ok = paths.start_csv.is_file()
        working_ok = paths.working_csv.is_file()
        c1, c2 = st.columns(2)
        c1.metric("Challenge 起點資料", "存在" if start_ok else "不存在")
        c2.metric("Challenge 工作資料", "存在" if working_ok else "不存在")
        st.caption(
            f"起點：`{_display_path(paths.start_csv)}`　"
            f"說明書：`{_display_path(paths.handbook)}`　"
            f"工作：`{_display_path(paths.working_csv)}`"
        )
        if not start_ok:
            st.warning("尚未放入該公司的 Challenge 起點 CSV。請老師先發放到 `workspace/challenge/`。")
        if not working_ok:
            st.info("尚無 Challenge 工作資料。訓練／預測前請先從起點複製並清理為 `working.csv`。")

        st.divider()
        _render_board_block(
            "① 我們在解決什麼？",
            "\n".join(
                part
                for part in (
                    f"**客戶：** {BOARD_CUSTOMER}" if _board_filled(BOARD_CUSTOMER) else "",
                    f"**問題：** {BOARD_PROBLEM}" if _board_filled(BOARD_PROBLEM) else "",
                    f"**任務類型：** {BOARD_TASK_TYPE}" if _board_filled(BOARD_TASK_TYPE) else "",
                    f"**目標欄：** {BOARD_TARGET_COLUMN}" if _board_filled(BOARD_TARGET_COLUMN) else "",
                )
                if part
            ),
            "空殼：請用資料 Agent 改本檔 `BOARD_CUSTOMER`／`BOARD_PROBLEM`／"
            "`BOARD_TASK_TYPE`／`BOARD_TARGET_COLUMN`。",
        )

        st.markdown("### ② 我們做出來的結果？")
        st.caption(
            f"使用資料：{'Challenge 工作資料' if working_ok else '尚未建立 Challenge 工作資料（請先清理）'}"
        )
        if _board_filled(BOARD_MODEL_NAME) or _board_filled(BOARD_METRIC_LINE):
            if _board_filled(BOARD_MODEL_NAME):
                st.write(f"**模型：** {BOARD_MODEL_NAME}")
            if _board_filled(BOARD_METRIC_LINE):
                st.write(f"**指標：** {BOARD_METRIC_LINE}")
        else:
            st.info(
                "空殼：最小 train／test＋一個模型＋一個指標＋一次試預測，"
                "由學生／Agent 實作。"
            )
        # TODO(challenge): 最小 train/test + 一個模型 + 一個指標
        # TODO(challenge): 一次試預測表單
        # TODO(challenge): 不要重構本頁以外的教學頁
        st.caption("TODO(challenge)：訓練與試預測實作區（AI coding 補齊）")

        _render_board_block(
            "③ 我們不能亂承諾什麼？",
            BOARD_LIMITS,
            "空殼：請用資料 Agent 改本檔 `BOARD_LIMITS`（至少 2–3 句，含該公司必講紅線）。",
        )

        st.divider()
        st.markdown("##### 上台 Gate（自評）")
        st.markdown(
            """
- ① 已能說明客戶、問題、任務類型、目標欄  
- 有做資料處理，訓練走 Challenge 工作資料（或能說明等價流程）  
- ② 能出示指標＋一次預測／演示  
- ③ 有具體紅線（含該公司必講差異點）  
- 能回答：Agent 幫了什麼、哪裡仍由人類決定  
- （軟規則）自選 1 個獨特畫面  
"""
        )
        st.markdown("##### 建議問 Agent")
        st.code(
            "請先讀我們公司的資料說明書，用三點說明客戶、預測目標、任務類型。",
            language="text",
        )
        st.code(
            "請檢查起點 CSV 的資料品質問題，先列出來，不要直接改檔。",
            language="text",
        )
        st.code(
            "請幫我建立 Challenge 工作資料（workspace/challenge/working.csv），"
            "並示範一種缺失值處理；改之前先問我要選哪種。",
            language="text",
        )
        st.code(
            "請幫我把白板①③寫進 startup_challenge_ui.py 的 BOARD_* 常數，並做第②塊最小訓練。",
            language="text",
        )

    board_summary = board_status_summary(
        customer=BOARD_CUSTOMER,
        problem=BOARD_PROBLEM,
        task_type=BOARD_TASK_TYPE,
        target_column=BOARD_TARGET_COLUMN,
        model_name=BOARD_MODEL_NAME,
        metric_line=BOARD_METRIC_LINE,
        limits_text=BOARD_LIMITS,
    )
    paths = challenge_paths(WORKSPACE_DIR, company)
    host = challenge_host_context(
        company=company,
        start_csv=_display_path(paths.start_csv),
        handbook=_display_path(paths.handbook),
        working_csv=_display_path(paths.working_csv),
        scripts_dir=_display_path(SHELL_ROOT / "scripts"),
    )

    snapshot = challenge_page_snapshot(
        company=company,
        start_exists=paths.start_csv.is_file(),
        working_exists=paths.working_csv.is_file(),
        board_summary=board_summary,
        target_column=BOARD_TARGET_COLUMN,
    )

    with agent:
        render_chat_panel(
            extra_context=snapshot,
            page_name="專案展示",
            host_context=host,
            agent_scope="challenge",
            skip_working_snapshot=True,
        )
