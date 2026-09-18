---
name: developing-with-streamlit
description: "Use for ALL Streamlit tasks: creating, editing, debugging, beautifying, styling, theming, optimizing, or deploying Streamlit apps. Also custom components, st.components.v2, HTML/JS/CSS work. Discovers and loads version-matched reference docs from the user's installed Streamlit (>=1.57). Triggers: streamlit, st., dashboard, app.py, beautify, style, CSS, color, background, theme, button, widget styling, custom component, st.components, CCv2, session state, performance, cache, fragment, slow rerun, deploy."
allowed-tools: Bash(python ${CLAUDE_SKILL_DIR}/scripts/discover.py:*) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/discover.py:*)
---

# Developing with Streamlit

Streamlit (>=1.57) ships detailed reference documentation for building Streamlit apps inside its pip package. The bundled skill is a routing `SKILL.md` plus a `references/` folder of topic-specific reference docs (dashboards, themes, layouts, session state, custom components, etc.).

## Usage

Run the discovery script with the user's project directory:

```bash
python <SKILL_DIR>/scripts/discover.py --project-dir <USER_PROJECT_DIR>
```

The script prints either:

- **A path on stdout** (exit 0) — the bundled `SKILL.md`. Read it; it points into `references/`.
- **An `ERROR:` block on stderr** (non-zero exit). Follow the printed instructions and re-run.

`<SKILL_DIR>` is the directory containing this file; `<USER_PROJECT_DIR>` is the absolute path to the user's project. Passing `--project-dir` matters because the script resolves `.venv`, `../.venv`, `Pipfile`, `poetry.lock`, `pdm.lock`, and `uv.lock` relative to it.

## 已知坑

- Windows 若 `python` 不在 PATH（exit 9009），改用 `py` 或 `uv run python` 跑 `discover.py`／pytest；本專案測試用 `uv run pytest`。
- Streamlit 1.61 的 `stLogo` 常掛在 `<img>` 上；對 img 寫 `::after` 瀏覽器不會畫。側欄品牌列字樣要掛在 `stSidebarHeader`（div）的 `::after`，不要掛在 `stLogo`。
- 側欄收合控制的 flex 兄弟是 `stSidebarCollapseButton`（div），裡面才有 `button`。`order`／`margin-left: auto` 必須打在 wrapper，打在巢狀 `button` 不會把收合鈕釘到右邊。
- Streamlit 1.61 `layout="wide"` 時 `.block-container` 左右是 `wideSidePadding`（5rem）。只覆寫 `padding-top` 會留下側欄與主教學欄之間的大空白；要縮這段距離須連 `padding-left`／`padding-right` 一起用 `!important` 蓋掉（本殼契約為左右各 1rem、對稱）。不要把側欄變窄當成解法。
- `@st.fragment` 會在資料 Agent 欄多包 VerticalBlock。雙欄殼若只把欄內「第一個」`stVerticalBlock` 設成 `height:100%` flex，內層仍依內容長，`.dss-agent-pane` 的 `overflow:hidden` 會裁掉 `stChatInput` 下半。要對 **含輸入框的那層** VB（`input.closest('[data-testid="stVerticalBlock"]')`）設 flex，並把祖先一路撐到欄高。
- `st.rerun(scope="fragment")` 只能在 **fragment rerun** 呼叫。學習階段 radio 等會整頁重跑；`@st.fragment` 函式仍會跟著跑，但 `fragment_ids_this_run` 是空的，此時呼叫會丟 `StreamlitAPIException`。對話紀錄 selectbox 在整頁重跑時的殘值也不算使用者改選（否則會誤切 session）。用 `_in_fragment_rerun()`／`_session_pick_action`／`_rerun_chat_panel`。
- 不要用 JS `insertBefore`／`appendChild` 把自製節點（例如 `.dss-resizer`）塞進 `st.columns` 那一列。React 重跑時 VDOM 仍以為只有兩欄，會丟前端 `NotFoundError: Failed to execute 'removeChild' on 'Node'`；學生畫面常只有紅框、沒有 Python traceback，重整就暫時好。`MutationObserver` 在 React 拆樹途中再 insert 會加重 race。分隔條改 overlay／欄內 `::before`，或完全不插 DOM。不要當 column 的 sibling。
- 不要拆成 `st.markdown('<div>')` … 內容 … `st.markdown('</div>')`。每個 markdown 是獨立 React 節點，不成對 HTML 會讓瀏覽器自己補 tag，unmount 時也可能 `removeChild`。卡片外框用 CSS class 或單一完整 HTML 字串。
- 專案展示頁骨架不要在模組頂層 import live `startup_challenge_ui`。Agent 把該檔寫成語法錯誤時，頂層 import 會讓整頁（含資料 Agent 欄）載入失敗。lazy import + `importlib.reload`，失敗與區函式例外都印在該區框，`with agent:` 仍要跑到。`except Exception` 必須把名稱含 `Rerun` 的例外再丟出，否則區函式裡的 `st.rerun()` 會被吃掉。`st.stop()` 可吞，避免它中斷右欄。
