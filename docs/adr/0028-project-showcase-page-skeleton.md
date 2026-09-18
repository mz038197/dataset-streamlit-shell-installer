# 專案展示拆成頁骨架與區函式；舊整頁不當快照

學生請 Agent 填模型區時，live UI 裡同時有 Challenge host context 組裝與區函式。Agent 把 host 當路徑物件空呼叫，頁面炸掉。區函式也沒被傳入該公司 train／test 路徑。定案：專案展示頁骨架（`ui/startup_challenge_page.py`）承接公司選擇、資料檢視、區框、「無檔則顯示輪廓」、host、資料 Agent 欄，並把 `ChallengePaths` 傳進模型區／成果區。live UI、專案展示空殼、Challenge UI 快照只留兩個接收路徑的區函式。頁入口薄包裝只呼叫骨架。

磁碟上若仍是舊的整頁 `startup_challenge_ui.py` 備份，不當快照：不載入、不因此畫清除回起點、不自動剪成區函式、不刪檔。視為沒有快照時用空殼。區函式形狀的快照，進頁時抄回 live，以免 `--update` 後空 live 在下次換公司蓋掉備份。

**Considered Options**：只在 host 加「不要呼叫 challenge_host_context」；自動把舊整頁剪成兩個函式；偵測到舊整頁就刪。前兩項留可改 chrome 或默默遷移錯檔，第三項讓學生對照上週程式時檔不見。

**Consequences**：ADR-0011 的「空殼含整頁工作流」改由頁骨架承擔；空殼還原來源變成兩個空函式。load／進頁抄回／清除回起點畫鈕共用挑戰軌道模組的區函式形狀判斷。Agent 預設不改頁骨架與空殼還原來源。頁骨架不在模組頂層 import live UI，並包區函式例外，避免語法錯誤或訓練失敗讓資料 Agent 欄畫不出來。
