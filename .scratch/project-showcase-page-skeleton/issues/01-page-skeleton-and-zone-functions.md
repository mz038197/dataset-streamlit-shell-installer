# 01 — 專案展示頁骨架與區函式隔離

**What to build:** 專案展示拆成專案展示頁骨架與兩個區函式。骨架負責挑戰公司選擇、Challenge 資料檢視、區框、「無檔則顯示輪廓」、Challenge host context、資料 Agent 欄，並把該公司路徑傳進模型區與成果區。live UI／專案展示空殼／Challenge UI 快照只剩那兩個函式。頁入口薄包裝只呼叫骨架。舊整頁檔留在磁碟但不當快照：不載入、不畫清除回起點、不刪、不自動剪函式。區函式形狀的快照進頁時抄回 live。Host 文案禁止改骨架、禁止在區函式裡把 Challenge host context 當路徑物件。寫 ADR-0028，產品文件對齊 glossary。前端 `removeChild`、多分群 CSV 不做。

**Blocked by:** None — can start immediately

**Status:** resolved

- [x] 挑戰軌道模組能分辨區函式形狀與舊整頁；load／進頁抄回／清除回起點畫鈕共用這份判斷
- [x] 頁骨架有 chrome 與無檔則顯示輪廓；live 與空殼還原來源相同且只有兩個接收路徑的區函式；頁入口薄包裝呼叫骨架
- [x] 解鎖後模型區／成果區拿到該公司路徑，不必呼叫 Challenge host context 組裝
- [x] 舊整頁不載入、不因此畫清除回起點、不刪檔；視為沒有快照時用空殼
- [x] 已確認公司且快照是區函式形狀時，進頁先抄回 live
- [x] Host 禁止改頁骨架與空殼還原來源，禁止在區函式把 host 當路徑物件
- [x] ADR-0028 與產品文件與 `CONTEXT.md` 一致

## Answer

頁骨架在 `ui/startup_challenge_page.py`；live／空殼只留 `render_model_zone(paths)` 與 `render_result_zone(paths)`。挑戰軌道用 AST 判斷區函式形狀；舊整頁不載入、不畫清除鈕、清除時不刪。進頁只在 live 仍等於空殼時把區函式快照抄回 live，避免 F5／rerun 蓋掉未入快照的模型區。ADR-0028。
