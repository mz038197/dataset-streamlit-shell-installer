# 專案展示（AI Startup Challenge）定案

> 來源：grilling（2026-08-11～12）＋原 temp 草案。  
> temp 草案（`Challenge展示空頁規格.md`／`Challenge_host_prompt草案.md`）保留作歷史，以本文與 `CONTEXT.md` 為準。

## 產品

| 項目 | 定案 |
|---|---|
| 側欄分段 | **AI新創工作坊**（與「降維分析」同層） |
| 側欄／頁名 | **專案展示** |
| 頁內主標 | 專案展示 |
| 頁內副標 | AI Startup Challenge｜成果展示 |
| 版面 | 主教學欄＝白板三塊；資料 Agent 欄＝既有 chat，專用 host／session |

## 資料軌道

- Challenge 起點資料：`workspace/challenge/{company}.csv`（只讀、不覆寫）
- Challenge 資料說明書：`workspace/challenge/{company}_資料說明書.md`
- Challenge 工作資料：`workspace/challenge/working.csv`
- 不走 Ready／資料切分；不強制 cleaning_log
- 挑戰公司：`edupulse`／`vitalrisk`／`airsense`／`churnlab`／`flowcast`
- 換公司：清除或封存 Challenge 工作資料；**不**還原已改 UI 程式

## 空殼與成果

- 發佈可為空殼；② 可留 `TODO(challenge)`
- 成果真相＝Agent **AI coding** 改允許檔案，不以表單 session 當唯一載體
- 允許改：`ui/startup_challenge_ui.py`、`workspace/challenge/*`、必要時 `scripts/`
- 頁入口薄包裝與側欄導覽：老師預放

## Agent

- **Challenge host context**：獨立組裝，**不**叠加 `dataset_base_context`（含 NN fragment）
- **Challenge Agent session**：與雙表整理線分開；進頁或換公司時以挑戰 host 重建
- host＝通用挑戰規則＋當前公司加碼；不剧透教師缺陷清單
- 每輪 user 附加頁面快照（公司、起點／working 是否存在、白板概況）

## Gate

- 人審／自評；頁上 checklist 文案即可
- 獨特畫面＝軟規則；第一版不做機器硬檢查

## 與舊草案差異

1. 側欄名改為 AI新創工作坊／專案展示  
2. host 不叠 `dataset_base_context`  
3. ①③ 不以 text_input 當成果真相（AI coding）  
4. 換公司不還原 UI  

詞彙見根目錄 `CONTEXT.md`。
