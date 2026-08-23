Status: ready-for-agent

# 線性回歸決策槽與 Agent 寫入

見 `CONTEXT.md` 與 [ADR 0015](../../docs/adr/0015-linear-regression-decision-slots.md)。詞彙：決策槽、目前選擇、選擇明細、鎖定槽、決策槽狀態、模型程式碼預覽、訓練請求（線性回歸）、訓練結果（線性回歸）、訓練前預測（線性回歸）、特徵縮放四種目前選擇。

## Problem Statement

線性回歸頁只讓學生看訓練微步驟與梯度表。右欄 Agent 給不了選擇、也開不了訓。特徵縮放被鎖成 Z-score，跟課堂要講的四種方法對不上。學生無法跟 Agent 一起組模型，只能旁觀。

## Solution

同一頁改成決策槽列：進頁尚未選擇，請 Agent 寫入決策槽狀態（預設 Z分數正規化、MSE、SGD）。點槽看只讀選擇明細（特徵縮放帶公式）。槽齊且訓練前預測過關後，學生或 Agent 可開訓；畫面只播回歸線與 Cost。改該階段選擇就清掉該階段訓練結果。

## User Stories

1. As a 學生, I want 側欄仍叫線性回歸, so that 我不用找新頁
2. As a 學生, I want 進頁時五個決策槽都是尚未選擇, so that 我知道要先跟 Agent 組模型
3. As a 學生, I want 看到輸入資料、特徵縮放、線性層、損失函數、優化器五格, so that 我知道這頁要決定什麼
4. As a 學生, I want 決策槽框上看到目前選擇, so that 不用點開就知道 Agent 寫了什麼
5. As a 學生, I want 尚未選擇的槽看起來與已選不同, so that 我看得出還沒組完
6. As a 學生, I want 點決策槽展開選擇明細, so that 我能對目前選擇與公式
7. As a 學生, I want 再點同一槽收起選擇明細, so that 左欄不會一直佔著
8. As a 學生, I want 特徵縮放明細標出公式、範圍與條件, so that 我能對課堂四種方法
9. As a 學生, I want 明細只讀、沒有下拉, so that 改選擇一定走右欄
10. As a 學生, I want 跟 Agent 說組一個線性回歸, so that 五槽被寫入預設目前選擇
11. As a 學生, I want Agent 組模型時預設特徵縮放是 Z分數正規化, so that 跟課堂主線一致
12. As a 學生, I want Agent 組模型時不要自動開始訓練, so that 我還能先看槽、先過關
13. As a 學生, I want 跟 Agent 改成正規化、最小-最大正規化或平均值正規化, so that 框上與明細跟著變
14. As a 學生, I want 跟 Agent 改 α 或 epochs, so that 優化器框上的目前選擇更新
15. As a 學生, I want Agent 拒絕改輸入資料來源、線性層種類、損失函數或優化器種類, so that 鎖定槽不會被講成可以換
16. As a 學生, I want 鎖定槽在組完後仍顯示目前選擇（內建表、Dense(1, linear)、MSE、SGD）, so that 它們看起來不像空槽
17. As a 學生, I want 單變量輸入資料鎖餐廳獲利欄位, so that 我不會去選欄
18. As a 學生, I want 多變量輸入資料鎖房價四特徵, so that 線性層維度跟著變成 4
19. As a 學生, I want 切學習階段後另一側的決策槽狀態與訓練結果還在, so that 兩階段可以對照
20. As a 學生, I want 尚未組過的學習階段仍是尚未選擇, so that 不會偷帶另一階段的選擇
21. As a 學生, I want 特徵縮放同一選擇套用該階段全部訓練特徵, so that 多變量不必每欄各選一種
22. As a 學生, I want 模型程式碼預覽跟決策槽狀態一致, so that 我看到的 Sequential 就是框上的選擇
23. As a 學生, I want 程式碼預覽只讀, so that 我不會以為改字就能跑
24. As a 學生, I want 預覽在縮放為 Z分數時對應 StandardScaler 寫法, so that 跟明細公式對得上
25. As a 學生, I want 換縮放後預覽的前處理行跟著換, so that 程式不是假的
26. As a 老師, I want 左欄沒有套用預設按鈕, so that 學生必須跟 Agent 建立選擇
27. As a 學生, I want 訓練前預測仍是舊的兩題, so that 第一版不必先改關卡
28. As a 學生, I want 關卡沒過也能請 Agent 組模型, so that 我可以先看槽再猜
29. As a 學生, I want 關卡沒過時開始訓練是鎖的, so that 我仍要先想斜率與 α
30. As a 學生, I want 關卡沒過時 Agent 拒絕寫訓練請求, so that 右欄不能當跳關後門
31. As a 學生, I want Agent 不直接講 radio 正解, so that 提示仍只是線索
32. As a 學生, I want 槽未齊時不能訓練, so that 空白狀態不會開跑
33. As a 學生, I want 槽齊且過關後按開始訓練, so that 我自己也能開訓
34. As a 學生, I want 跟 Agent 說開始訓練時左欄播同一套動畫, so that 口頭開訓與按鈕沒有兩套結果
35. As a 學生, I want 訓練中只看到回歸線、Cost 與一行狀態, so that 視線不被表打斷
36. As a 學生, I want 訓練中決策槽列還在且可點, so that 我仍看得到現在的 α 與縮放
37. As a 學生, I want 看不到訓練微步驟、梯度演算板、樣本運算表、逐步模式, so that 頁不再是看步驟
38. As a 學生, I want 看不到三節點 SVG 與目前查看, so that 頂部只有決策槽列
39. As a 學生, I want 散點用原始單位、線映回原尺度, so that 圖上的人口／房價還認得
40. As a 學生, I want 四種縮放都這樣畫線, so that 換方法不會把橫軸變成縮放後的數
41. As a 學生, I want 訓前圖區空白, so that 我不會以為已經訓過
42. As a 學生, I want 訓完留下該階段訓練結果, so that 重整或切走再回來還看得到線
43. As a 學生, I want 改該階段決策槽狀態後線與 Cost 立刻消失, so that 舊選擇的線不會騙人
44. As a 學生, I want 改單變量選擇不影響多變量訓練結果, so that 兩階段互不覆蓋
45. As a 學生, I want 改選擇後關卡若仍過關即可再訓, so that 不必重答兩題
46. As a 學生, I want Agent 寫入後右欄 snapshot 含目前選擇與我正打開的槽, so that Agent 能講「你點開的是特徵縮放」
47. As a 學生, I want Agent 只在本頁動決策槽狀態與訓練請求, so that 它不會去改類神經網路的 form
48. As a 學生, I want Agent 不准 exec 出一套自己的 w, so that 左欄數字跟右欄說法同一來源
49. As a 老師, I want 本頁不做保存已訓模型檔與手動預測輸入, so that 範圍仍停在組與訓
50. As a 老師, I want 邏輯迴歸維持舊教學流程圖, so that 這次只收線性回歸
51. As a 學生, I want 正規化（除以最大）在 x 出現負值時被拒絕或講清楚不能用, so that 課堂條件 x≥0 不會默默算錯
52. As a 維護者, I want 舊微步驟相關純函式若不再被本頁呼叫就不要掛在主路徑, so that 廢止不是只改文案

## Implementation Decisions

- 取代同一頁線性回歸，不新增側欄頁。
- 決策槽狀態與訓練請求走與類神經網路相同的寫檔協議：共享狀態、訓練請求訊號、讀寫既有檔案工具。不做自動實驗迴圈、不消耗實驗預算、不新開訓練 tool。
- 決策槽狀態按學習階段各一份。正規化後的形狀（來自原型，只留決策）：

```
choices: { data, scale, linear, loss, opt }
scale ∈ { maxdiv, minmax, mean, zscore }
linear = dense1
loss = mse
opt = sgd
alpha, epochs 屬 opt
```

- 進頁或該階段尚未組過：全部尚未選擇。Agent「組模型」寫入鎖定槽固定值＋ Z分數正規化＋預設 α／epochs。
- 模型程式碼預覽是決策槽狀態的純函式，與 UI 用同一輸出。
- 特徵縮放四種在訓練模組用同一 scaler 字典慣例擴充 `method`；`predict_line_on_original_x` 依 method 映回，不假設只有 zscore。
- 正規化（除以最大）在任一訓練特徵出現負值時驗證失敗，Agent／UI 顯示原因，不開訓。
- 決策槽狀態簽名改變（該階段 choices／α／epochs）即丟掉該階段訓練結果。
- 訓練動畫抽樣可沿用現有上限思維（長 epochs 抽幀），但畫面不得組微步驟 frame、不得切訓練雙欄。
- 訓練前預測沿用現有兩題與 Agent 提示；解鎖條件接到「可否寫訓練請求／可否按開始訓練」。
- host 規則用本頁 fragment 講清：組模型只寫決策槽狀態；開訓另寫訓練請求；未齊或未過關不准請求；禁止 exec 訓練；點開的槽寫進當頁 snapshot。
- 內容區雙欄殼契約不變。Agent 欄寬沿用全站可拖寬度，不為本頁另做預設更寬。
- 不把決策槽列抽成給邏輯迴歸用的共用頁殼。

## Testing Decisions

- 測對外行為：狀態是否齊、預設寫入、四種縮放的數值變換與映回原尺度的線、程式碼預覽字串、訓練請求能否寫／被消費、改狀態是否作廢訓練結果、負值＋正規化（除以最大）要失敗。
- 不測 Streamlit widget、點擊樣式、雙欄 CSS。
- 主測縫是決策槽狀態模組（正規化、預設、預覽、請求、簽名）。縮放與映回線延伸現有回歸模組測試（已有 zscore 映回線的先例），三種新 method 各補行為斷言。
- 關卡對錯沿用現有訓練前預測測試，只加「未解鎖不得視為可寫訓練請求」。
- 不為已廢止的微步驟 frame／演算板／樣本表加新測試；舊測試若只服務廢止 UI，改為刪除或不再從本頁引用。

## Out of Scope

- 邏輯迴歸或其他教學頁改決策槽
- 特徵縮放「不縮放」、損失 MAE、Adam 或其他優化器
- 左欄下拉、可編輯程式碼、exec 預覽、保存已訓模型、手動預測
- 類神經網路式自動實驗迴圈與 max_runs
- 重寫訓練前預測題幹
- 上傳資料或 ready.csv
- 把決策槽做成跨頁工作流引擎
- 正式合併 throwaway 原型檔（原型不當生產碼）

## Further Notes

- 原型在教學 UI 旁，只用來對過 C 的決策槽列與四種縮放明細，實作請依本規格與詞彙表重寫。
- ADR 0008 留下的圖表規則（散點原尺度、線映回）仍要測；樣本運算表本身不要回來。
- 資料轉換頁的「正規化」避免詞不變：槽名是特徵縮放，方法「正規化」指除以最大。
