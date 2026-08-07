# 線性回歸頁以自研 SVG 教學流程圖為主骨架

要把「輸入 → 帶公式的模型 → 輸出」做成可導覽且訓練時可點亮的路徑，決定線性回歸兩學習階段共用三節點教學流程圖：頂部自研 SVG 拓撲條，下方三欄掛內容；模型節點必顯預測公式（Cost／GD 收 expander）；開始訓練後隨動畫 iteration 更新高亮與輸出圖，訓練中保留符號式並旁附即時參數，結束定格數字式。不用 Mermaid／graphviz，也不用純卡片列（難呈現邊與逐步高亮）。

**Considered Options**: 三欄 CSS 卡片（好維護但較不像 graph）；Mermaid／graphviz（難逐步高亮）；自研 SVG 條＋三欄內容（選定）。粗階段點亮曾列為備選，定案為跟 iteration（與現有 GD placeholder 動畫同頻）。

**Consequences**: 流程圖邏輯宜落在可單測的純函式（SVG／公式／caption），UI 只負責掛欄與動畫；別頁若複用「教學流程圖」詞彙，拓撲仍各自定義。
