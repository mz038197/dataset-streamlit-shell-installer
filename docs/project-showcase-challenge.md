# 專案展示（AI Startup Challenge）定案

> 來源：grilling（2026-08-11～12、2026-08-16、2026-08-19）。以本文與 `CONTEXT.md` 為準。白板三塊為舊頁結構，見 ADR-0011。各公司進度見 ADR-0014。

## 產品

| 項目 | 定案 |
|---|---|
| 側欄分段 | **AI新創工作坊**（與「降維分析」同層） |
| 側欄／頁名 | **專案展示** |
| 頁內主標 | 專案展示 |
| 版面 | 上半 Challenge 資料檢視；下半模型區／成果區空輪廓。資料 Agent 欄＝既有 chat，專用 host／session |
| 頁上文字 | 幾乎只留標題、公司選擇、資料、兩個空框。無 Gate checklist、無建議問句 |

## 資料軌道

- Challenge 起點資料：`workspace/challenge/{company}.csv`（只讀、不覆寫）
- Challenge 資料說明書：`workspace/challenge/{company}_資料說明書.md`（不上頁）
- Challenge 工作資料：`workspace/challenge/{company}/working.csv`（各公司一份；模型不直接吃）
- Challenge 訓練資料／測試資料：`workspace/challenge/{company}/train.csv`、`test.csv`（Agent 從該公司 working 切出，預設 80／20，類別則分層；無 val）
- 不走 Ready；不寫根目錄 train／val／test
- 挑戰公司：`edupulse`／`vitalrisk`／`airsense`／`churnlab`／`flowcast`（模板內建五間起點 CSV 與說明書；教師用缺陷說明不進學生專案）。已確認的那一間寫進磁碟，重整後仍有效
- 改該公司 working：刪除該公司 train／test，該公司模型產物失效
- 換公司：先經過更換挑戰公司確認；確認後**不刪檔**，改存／載入 Challenge UI 快照（沒有則專案展示空殼），切換該公司的 Challenge Agent session
- 舊共用 `challenge/working.csv`（及 train／test）：只搬進確認的那一間。磁碟上還沒記住公司卻仍有舊檔時，第一次也要確認後才搬
- 安裝／更新／`--force`：不得覆蓋或刪除 Challenge 工作資料、訓練資料、測試資料（含各公司資料夾與舊共用檔）

## 空殼與成果

- 發佈＝專案展示空殼；某公司尚無 Challenge UI 快照時才載入 `ui/startup_challenge_empty_shell.py`
- Challenge UI 快照：`workspace/challenge/{company}/startup_challenge_ui.py`；Agent 只改 live `ui/startup_challenge_ui.py`
- 模型區／成果區框始終可見；無該公司 train+test 時不可填入
- 模型區＝選型與訓練；成果區＝訓練後指標／圖／演示（要有該公司 Challenge 模型產物才渲染；產物寫在該公司資料夾，重整與換公司後仍有效）
- 一次 AI coding 可寫兩區程式；成果區未訓練前仍是空輪廓
- 成果真相＝Agent **AI coding** 改允許檔案，不以表單 session 當唯一載體
- 允許改：`ui/startup_challenge_ui.py`、目前公司資料夾內的 working／train／test、必要時 `scripts/`
- 不准拆「無檔則顯示輪廓」；不准改 empty_shell 檔；不准直接改各公司快照
- 倫理紅線只在對話與口頭 Gate

## Agent

- **Challenge host context**：獨立組裝，**不**叠加 `dataset_base_context`（含 NN fragment）
- **Challenge Agent session**：與雙表整理線分開；各公司一條，換公司切換不清空，重整後不保留
- host＝通用挑戰規則＋當前公司加碼；不剧透教師缺陷清單
- 每輪 user 附加頁面快照（公司、起點／working／train／test、模型產物）

## Gate

- 人審／自評；**不印在頁上**
- 獨特畫面＝軟規則；第一版不做機器硬檢查

詞彙見根目錄 `CONTEXT.md`。ADR-0009（host 獨立）、ADR-0011（工作流空殼）、ADR-0014（各公司進度）。
