# 內容區雙欄殼：留在 Streamlit，自訂高保真殼

全站內容區要對齊 waku 式體驗（主教學欄獨立捲動、資料 Agent 欄可拖寬並釘在視窗內），但產品仍是 Dataset Streamlit Shell。決定**不**為此整站改純 HTML/JS，改以自訂**內容區雙欄殼**承載主教學欄｜資料 Agent 欄；導覽側欄與線性回歸頁內三欄不在本決策範圍。

**Considered Options**: 整站遷純 HTML/JS（版面最順，等於重寫教學產品，否決）；Streamlit 原生 `st.columns`＋固定高度／比例控制的務實近似（穩但沒有拖分隔線手感，否決）；Streamlit 引擎＋自訂雙欄殼／component 做拖曳與獨立捲動（選定）。

**Consequences**: 所有「主教學欄｜資料 Agent 欄」頁須走同一殼入口（`open_content_dual_pane`），避免再散落硬編碼 `[5, 3]`。第一版契約：Agent 預設寬 320px（min 260／max 560）、寬度全站共用並存瀏覽器 `localStorage`、聊天區吃滿欄內剩餘高度；不做 Agent 收合、窄螢幕堆疊、主教學欄 sticky 標題。殼層依賴 `st.html(..., unsafe_allow_javascript=True)`，安裝器最低 Streamlit 升為 `>=1.61`。詞彙見 `CONTEXT.md`（主教學欄、資料 Agent 欄、內容區雙欄殼）。
