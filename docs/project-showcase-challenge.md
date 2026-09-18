# 專案展示（AI Startup Challenge）定案

> 來源：grilling（2026-08-11～12、2026-08-16、2026-08-19、2026-08-24）。以本文與 `CONTEXT.md` 為準。白板三塊為舊頁結構，見 ADR-0011。各公司進度見 ADR-0014。清除回起點見 ADR-0025。頁骨架與區函式隔離見 ADR-0028。

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
- 清除回起點：Challenge 資料檢視旁的按鈕；無工作／切分／產物且無該公司（區函式形狀的）快照時不畫。磁碟上的舊整頁檔不算快照，不因此畫鈕。經清除回起點確認後只動目前公司：刪 working／train／test 與該公司區函式快照，live UI 還原專案展示空殼；舊整頁檔不刪；起點與其他公司、對話不動（ADR-0025）
- 舊共用 `challenge/working.csv`（及 train／test）：只搬進確認的那一間。磁碟上還沒記住公司卻仍有舊檔時，第一次也要確認後才搬
- 安裝／更新／`--force`：不得覆蓋或刪除 Challenge 工作資料、訓練資料、測試資料（含各公司資料夾與舊共用檔）

## 空殼與成果

- 發佈＝專案展示空殼；某公司尚無 Challenge UI 快照時載入 `ui/startup_challenge_empty_shell.py`；清除回起點確認後亦載入（該公司快照已刪）
- 專案展示頁骨架：`ui/startup_challenge_page.py`（公司選擇、資料檢視、區框與無檔則顯示輪廓、host、資料 Agent 欄）。頁入口薄包裝 `pages/30_Startup_Challenge.py` 只呼叫此檔。Agent 不准改
- 專案展示空殼與 live UI：只含模型區／成果區兩個接收該公司路徑的函式，內建可抄的最小選型／訓練骨架。發佈時兩檔位元組相同。無 train／test 時頁骨架仍不呼叫，畫面維持空輪廓
- Challenge UI 快照：`workspace/challenge/{company}/startup_challenge_ui.py`，內容是區函式不是頁骨架。Agent 只改 live `ui/startup_challenge_ui.py`。進頁時若快照已是區函式形狀則抄回 live。舊整頁檔留在磁碟但不載入、不因此畫清除回起點、不自動剪函式
- 模型區／成果區框始終可見；無該公司 train+test 時不可填入。解鎖後骨架把該公司路徑傳進區函式
- 模型區＝選型與訓練；成果區＝訓練後指標／圖／演示（要有該公司 Challenge 模型產物才渲染；產物寫在該公司資料夾，重整與換公司後仍有效）
- 一次 AI coding 可寫兩區程式；成果區未訓練前仍是空輪廓
- 成果真相＝Agent **AI coding** 改允許檔案，不以表單 session 當唯一載體
- 允許改：`ui/startup_challenge_ui.py`、目前公司資料夾內的 working／train／test、必要時 `scripts/`
- 不准改專案展示頁骨架與空殼還原來源；不准直接改各公司快照；不准在區函式呼叫 Challenge host context 當路徑物件；不准把教學頁雙欄／聊天欄抄進區函式
- 頁骨架 lazy 載入 live UI，並把區函式例外印在該框；資料 Agent 欄仍畫。區函式裡的 `st.rerun()` 仍要生效（名稱含 Rerun 的例外要再丟出）
- 倫理紅線只在對話與口頭 Gate

## Agent

- **Challenge host context**：獨立組裝，**不**叠加 `dataset_base_context`
- **Challenge Agent session**：與雙表整理線分開；各公司一條，換公司切換不清空，重整後不保留
- host＝通用挑戰規則＋當前公司加碼；不剧透教師缺陷清單
- 每輪 user 附加頁面快照（公司、起點／working／train／test、模型產物）

## Gate

- 人審／自評；**不印在頁上**
- 獨特畫面＝軟規則；第一版不做機器硬檢查

詞彙見根目錄 `CONTEXT.md`。ADR-0009（host 獨立）、ADR-0011（工作流空殼）、ADR-0014（各公司進度）、ADR-0028（頁骨架與區函式）。
