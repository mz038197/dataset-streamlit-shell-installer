Status: ready-for-agent

# 專案展示頁骨架與區函式隔離

見 `CONTEXT.md`：專案展示頁骨架、專案展示空殼、模型區、成果區、Challenge UI 快照、Challenge host context、Challenge 允許改動範圍、清除回起點確認。換公司規則仍依 ADR-0014；清除回起點仍依 ADR-0025。實作時寫 ADR-0028 記錄拆檔與舊整頁不當快照。

## Problem Statement

學生在專案展示請 Agent 用 AI coding 填模型區時，Agent 打開同一份 live UI，看到 Challenge host context 的組裝函式，就把它當路徑物件空呼叫。頁面炸掉。Streamlit 前端跟著出現 `removeChild` 紅框。模型區函式沒有被傳入該公司 Challenge 訓練資料／測試資料的位置，Agent 只能猜。

老師要的不是再寫一句「請不要呼叫」。不能改的區塊必須不在 Agent 可改的檔裡。

## Solution

專案展示拆成兩層。專案展示頁骨架（Agent 不准改）負責挑戰公司選擇、Challenge 資料檢視、區框、「無檔則顯示輪廓」、Challenge host context、資料 Agent 欄，並把該公司路徑傳進模型區與成果區。專案展示空殼／live UI／Challenge UI 快照只剩兩個區函式。頁入口薄包裝仍只呼叫頁骨架。

舊的整頁備份可留在磁碟，但不當快照：不載入、不因此畫清除回起點、不自動剪函式、不默默刪檔。區函式形狀的快照，進頁時抄回 live，以免安裝更新後第一次換公司用空 live 蓋掉備份。

## User Stories

1. As a 學生, I want 模型區函式一打開就有該公司路徑, so that Agent 不必猜檔在哪
2. As a 學生, I want 成果區函式同樣拿到該公司路徑, so that 評估與演示吃同一對 Challenge 訓練資料／測試資料
3. As a 學生, I want 沒有該公司 train 與 test 時模型區仍是空輪廓、不能填, so that 不會先做模型再切分
4. As a 學生, I want 沒有 Challenge 模型產物時成果區仍是空輪廓, so that 未訓練不當成果
5. As a 學生, I want 「無檔則顯示輪廓」寫在專案展示頁骨架, so that Agent 填兩區時拆不掉這層判斷
6. As a 學生, I want live UI 裡看不到組 Challenge host context 的程式, so that Agent 不會把它當 `ctx` 來呼叫
7. As a 學生, I want Agent 只改 live UI 的兩個區函式, so that 公司下拉、資料檢視、聊天欄不會被改壞
8. As a 學生, I want 跟 Agent 說把模型寫進模型區, so that 它改的是專案展示空殼那份 live 檔，不是頁骨架
9. As a 學生, I want 跟 Agent 說先分群再各訓再整合, so that 它能寫在模型區函式裡，不必等殼層開分群區
10. As a 學生, I want Challenge 訓練資料／測試資料仍是各公司一對, so that 分群是算法不是第二套 CSV 生命週期
11. As a 學生, I want 換挑戰公司後看到該公司的模型區程式, so that 五間公司不會共用已填的 UI
12. As a 學生, I want 換公司時存的是兩個區函式, so that 快照裡沒有專案展示頁骨架
13. As a 學生, I want 某公司尚無快照時看到兩個空函式的空殼, so that 新公司從空輪廓開始
14. As a 學生, I want 清除回起點確認後 live 回到兩個空函式, so that 模型區程式與該公司工作／切分一起沒
15. As a 學生, I want 沒有工作／切分／產物、且沒有區函式形狀快照時看不到清除回起點, so that 空頁不會多一顆沒用的鈕
16. As a 學生, I want 磁碟上的舊整頁檔不要讓清除回起點出現, so that 更新後不像還有一筆可清的模型區進度
17. As a 學生, I want 舊整頁檔更新後仍留在公司資料夾, so that 若要對照上週程式還找得到
18. As a 學生, I want 頁面不要載入舊整頁檔, so that 公司選擇與聊天欄不會被整頁備份寫回 live
19. As a 老師, I want `--update` 後進頁若該公司快照已是區函式形狀就抄回 live, so that 學生接著換公司不會用空 live 蓋掉備份
20. As a 老師, I want `--update` 後舊整頁快照不抄回 live, so that 隔離不會被舊檔破掉
21. As a 老師, I want 安裝更新不要刪各公司工作／訓練／測試與快照檔, so that 學生資料軌道還在
22. As a 老師, I want 頁入口薄包裝仍只呼叫專案展示頁骨架, so that 30_ 那頁不會變厚
23. As a 老師, I want 發佈時 live 與空殼還原來源位元組相同且只有兩個空函式, so that 還原來源跟發佈畫面一致
24. As a 老師, I want Challenge host context 寫明不准改專案展示頁骨架、不准在模型區呼叫 host 組裝函式, so that 對話規則跟檔案邊界一致
25. As a 老師, I want Challenge 允許改動範圍不含專案展示頁骨架與空殼還原來源, so that Agent 預設工具邊界對
26. As a 學生, I want 重整頁面不改 live UI, so that F5 不會把模型區程式換掉
27. As a 學生, I want 資料 Agent 欄仍用該公司 Challenge Agent session 與 Challenge host context, so that 挑戰軌道不叠雙表整理線
28. As a 學生, I want 訓成功後仍設 Challenge 模型產物, so that 成果區才解鎖
29. As a 學生, I want 寫回該公司 Challenge 工作資料時切分與產物仍作廢, so that 模型不會吃過期切分
30. As a 學生, I want 前端 `removeChild` 這次不被當成修繕範圍, so that 這次只修 Python 路徑陷阱與隔離
31. As a 維護者, I want 判斷「這份檔算不算區函式形狀快照」寫在無 Streamlit 依賴的挑戰軌道模組, so that 測試不必開瀏覽器
32. As a 維護者, I want 載入快照、進頁抄回、清除回起點是否畫鈕都走同一份「可當快照」判斷, so that 舊整頁不會在一處被忽略、另一處又被當成進度
33. As a 維護者, I want 實作留下 ADR, so that 以後的人不會把 chrome 併回 live UI 當成整理

## Implementation Decisions

- 新增專案展示頁骨架模組，承接現在 live UI 裡的整頁渲染（公司選擇、資料檢視、區框解鎖、host、聊天欄）。頁入口薄包裝改呼叫此模組。
- 專案展示空殼與 live UI 縮成兩個函式：模型區與成果區，簽名都接收該公司路徑物件（挑戰軌道既有的路徑資料結構）。頁骨架在解鎖後呼叫它們並傳入路徑。
- 空殼還原來源仍是獨立檔，發佈時與 live UI 內容相同（兩個空函式）。清除回起點確認與「無快照」載入仍從此來源還原 live。
- Challenge UI 快照的檔名與公司資料夾位置不變。內容改為區函式，不是頁骨架。
- 在挑戰軌道模組（無 Streamlit）新增「這份文字算不算區函式形狀快照」：必須定義模型區與成果區且接收路徑參數；不得定義整頁渲染入口。舊整頁（含整頁入口）回傳否。
- `load` 快照：可當快照才用該檔，否則當沒有快照、改用空殼還原來源。不刪磁碟上的舊整頁檔。
- 進頁：已確認挑戰公司且該公司快照可當快照時，先抄回 live 再畫頁。不可當快照則不抄回；live 用空殼。
- 清除回起點是否可畫：工作／切分／產物仍算。快照只在「可當快照且內容與空殼還原來源不同」時算。舊整頁檔不算。
- Challenge host context 文案：可改 live 兩區函式與目前公司資料夾、必要時腳本目錄；禁止改頁骨架與空殼還原來源；禁止在模型區／成果區呼叫 host 組裝函式當路徑物件；讀檔用傳入的路徑。
- 實作時新增 ADR-0028：為何拆檔、為何舊整頁不當快照、為何不自動遷移、為何不刪舊檔。
- 不改雙欄殼 MutationObserver、不修前端 `removeChild`。
- 不新增多份 Challenge 訓練資料、不分群區、不改五間公司檔案契約。

## Testing Decisions

- 只測對外行為：檔案角色（誰含 chrome、誰只有區函式）、快照可不可用、載入／抄回／畫鈕是否把舊整頁當沒有。不測 Streamlit 畫面、不測 Agent 會不會聽話。
- 挑戰軌道純函式是主縫：可當快照判斷、load 在不可用時走空殼、進頁抄回只發生在可當快照、清除回起點可用性忽略舊整頁。
- 版面契約用既有「讀模板原文」測法：live 等於空殼還原來源；兩者都沒有整頁 chrome（公司 dialog、host 組裝、聊天欄注入）；頁骨架有那些 chrome；頁入口薄包裝呼叫頁骨架而不是 live 的整頁函式。
- Host 文案測禁止頁骨架與禁止在區函式呼叫 host 組裝。
- Prior art：`tests/test_startup_challenge.py`（空殼等於 live、快照 save/load、清除回起點可用性、host 文案）。

## Out of Scope

- 前端 `NotFoundError: removeChild`、雙欄殼 MutationObserver、思考過程 expander。
- 把舊整頁自動剪成兩個區函式。
- 偵測到舊整頁就刪檔。
- 殼層第一類支援多群切分、多份 train／test、頁上多分群區。
- 改教學頁（邏輯回歸等）、改雙表整理線、改 Challenge 上台 Gate 機器驗收。
- 為挑戰線做 F5 對話存盤。

## Further Notes

測試縫（若不符請先說，不要默默改縫）：

1. 挑戰軌道模組的快照可用性與 load／抄回／畫鈕（最高、無 UI）。
2. 模板原文契約：空殼＝live＝只有區函式；chrome 只在頁骨架。

實作後應同步 `docs/project-showcase-challenge.md` 的空殼／允許改動範圍段落，使其與 `CONTEXT.md` 一致。
