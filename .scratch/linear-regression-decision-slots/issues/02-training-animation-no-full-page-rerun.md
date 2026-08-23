# 02 — 線性回歸訓練動畫不要整頁跟著每一幀重跑

**What to build:** 單變量與多變量的訓練動畫改成同一次開訓流程內逐幀更新回歸線、Cost 與一行狀態，播完才重跑一次以留下訓練結果。決策槽列播完前可見、框上有目前選擇、不可點；訓練前預測與資料 Agent 欄凍結。開始訓練與訓練請求走同一契約。

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [x] `_run_simple_training`／`_run_multiple_training` 對抽樣幀 loop + sleep，不再每幀 `st.rerun()`
- [x] 按鈕開訓與訓練請求都在同一 request 播完，中間不再為了開訓多 rerun 一次
- [x] 播完寫入該階段訓練結果後不 rerun，同一 request 接著畫訓練結果與資料 Agent 欄
- [x] 沒有把訓練微步驟、逐步模式或 fragment 加回來

見 `CONTEXT.md` 訓練動畫（線性回歸），以及 spec story 36。
