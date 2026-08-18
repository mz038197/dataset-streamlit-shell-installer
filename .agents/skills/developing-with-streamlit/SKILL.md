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
