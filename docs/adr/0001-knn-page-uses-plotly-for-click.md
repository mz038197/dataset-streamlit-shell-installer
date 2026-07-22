# K-近鄰分類頁使用 Plotly 以支援查詢點 click

其他監督式 ML 頁以 matplotlib + `st.pyplot` 繪圖，圖上無法點選座標。K-近鄰分類需要「click 查詢點 → 標出 k 個鄰居」的課堂儀式感，因此**僅此頁**改用 Plotly（含 selection／click），不強制全專案改繪圖棧。

**Considered Options**: 改回 number_input／預置點（維持 matplotlib）；或 matplotlib 邊界圖 + 小塊 Plotly 點選並存。前者犧牲 click；後者雙圖維護成本更高。選定單頁 Plotly 主路徑。

**Consequences**: 模板需能依賴 Plotly；中文字型／樣式可能與 matplotlib 頁不一致，以本頁可用為優先。日後若全專案統一互動圖，可再 revisit。
