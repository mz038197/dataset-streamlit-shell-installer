# Agent 交接：ML 頁學生–Agent 互動決策

> 給下一位 agent 快速上手。記錄截至 2026-07-21 的產品／教學決策與實作狀態。  
> 專案：`dataset-streamlit-shell-installer`（模板在 `src/add_dataset_streamlit_shell/templates/dataset_streamlit_shell/`）。

## 1. 問題背景

課堂觀察：進入**分類演算法／類神經網路**後，學生與右側 Agent 互動接近 0——調參、按「開始訓練」、看動畫就結束。  
Agent 多半是可選裝飾；建議問句是可忽略的 `st.code`。

對照已有效的模式：

| 頁面 | 為何會跟 Agent 講話 |
|---|---|
| 圖表探索 `2_Charts.py` | 答錯圖種解不了鎖，文案導向問 Agent |
| 資料清理 `workflow_ui.py` | Agent 能改 `working.csv`；「先討論再勾選」 |
| 分類／舊版 NN／回歸 | 訓練自給自足，無門檻 |

**原則**：不要每頁都抄同一套。組合空間大 → 可寫回／實驗迴圈；概念頁 → 訓練前預測＋提示按鈕。

### 資料來源（監督式教學頁）

- **page 14–18**（單／多變量回歸、邏輯迴歸、正則化邏輯迴歸、線性支持向量機）：**僅內建範例資料**，不提供 `ready.csv` 切換（2026-07-21 定案；對齊 page 19–20）。
- `ready.csv` 仍供清理流程、圖表探索、降維、分群等分析頁使用。

---

## 2. 已實作：類神經網路（寫回左欄＋動畫實驗迴圈）

### 決策

- 採用 **Agent Studio 同款**：共享 JSON + `read_file`／`edit_file` 寫回左欄（不新開專用 tool）。
- **不要**用 `exec` 無 UI 訓練當主路徑；Agent 觸發的訓練必須走與手動相同的 `_run_training` **動畫**。
- **啟動方式**：只靠 chat——Agent 寫入 `nn_train_request.json`（`{"requested": true}`）才開第一輪；**不做**「開始 Agent 實驗」按鈕。
- 左欄**可手動調**；手動「開始訓練」**不消耗** Agent 實驗預算、不進迴圈。
- 實驗次數：預設 **3**，左欄可調 **1～5**（硬上限 5）；Agent **不可**改高 `max_runs`。
- Agent 觸發訓練：`epochs` **clamp ≤ 80**。
- Shell 主導狀態機（一輪動畫 = 一次 rerun）：`need_train` → 動畫 → `need_agent_decision` → `invoke_data_agent` → 可再請求訓練。

### 關鍵路徑

| 角色 | 路徑 |
|---|---|
| UI | `ui/nn_ui.py` |
| Form／迴圈狀態 | `ui/nn_form_state.py` |
| Host 規則片段 | `nn_form_state.nn_host_context_fragment` → `data_ui.dataset_base_context` |
| 程式呼叫 Agent | `data_ui.invoke_data_agent` |
| 共享檔 | `workspace/nn_form.json`、`nn_train_request.json`、`nn_last_run.json`、`nn_agent_runs.jsonl` |
| 測試 | `tests/test_nn_form_state.py` |

### 課堂注意

若 Agent 在改 host 文案前就已啟用，需再按一次「啟用資料 Agent」才載入新 `host_context`。

### 計劃檔（參考）

`~/.cursor/plans/nn_agent_寫回左欄_7405161d.plan.md`（勿當唯一真相；以程式與本文件為準）。

---

## 3. 已實作：單變量線性回歸（訓練前預測＋Agent 提示）

### 決策

- **不**抄 NN 實驗迴圈（α／epochs 空間太小，自動連跑容易變旁觀）。
- 採用 **訓練前預測關卡**：**兩題都答對**才解鎖「開始訓練」。
- 每題一顆 **「Agent 提示」**：一鍵 `invoke_data_agent` 送**線索**問句（明確要求不要直接講正解）；未啟用 Agent 則 disabled。
- 點提示後 `st.rerun()`，讓右欄立刻看到對話；同題短防連點（約 2.5s）。
- 換 **feature／target**（內建欄位間切換）→ 重置兩題；改 α／epochs **不**強制重置。
- 訓後輕量 caption：對照「你猜的斜率方向」vs 實際 `w`。不做第三關卡。
- 底部建議問句改為焦點連動（未解鎖／已解鎖不同文案）。

### 兩題內容

1. **斜率方向**（正的／負的／接近 0）  
   正解：當前資料 `feature` 與 `target` 相關係數符號（`|corr|≤0.05` → 接近 0）；本頁僅使用內建餐廳獲利資料。
2. **α 明顯偏大時 Cost**  
   正解固定：劇烈震盪或往上爆。

### 關鍵路徑

| 角色 | 路徑 |
|---|---|
| Quiz helpers | `ui/simple_regression_quiz.py` |
| 頁面接入 | `ui/workflow_ui.py` → `render_simple_linear_regression_page` |
| 測試 | `tests/test_simple_regression_quiz.py` |

### 計劃檔（參考）

`~/.cursor/plans/單變量訓練前預測_bb4a2b3b.plan.md`

---

## 4. 已實作：多變量線性回歸（目的取向預測）

### 決策

- **同一套殼**（兩題解鎖＋每題 Agent 提示），題目**不抄**單變量斜率題。
- 問題方向：**多變量的目的／w·b 意義**；α／Z-score **不當**解鎖關卡。
- 僅使用內建房價資料（與單變量同：無 `ready.csv` 切換）。

### 兩題內容

1. **為什麼要用多個 feature**  
   正解：用多個因素一起解釋／預測同一個連續 target。
2. **模型在學什麼（w、b）**  
   正解：每個（縮放後）feature 對預測的貢獻方向與相對強度，再加上截距。

### 關鍵路徑

| 角色 | 路徑 |
|---|---|
| Quiz helpers | `ui/multiple_regression_quiz.py` |
| 頁面接入 | `ui/workflow_ui.py` → `render_multiple_linear_regression_page` |
| 測試 | `tests/test_multiple_regression_quiz.py` |

### 計劃檔（參考）

`~/.cursor/plans/多變量訓練前預測_purpose.plan.md`

---

## 5. 已實作：線性支持向量機（雙階段＋訓練前預測）

### 決策

- 側欄／頁面標題用 **「線性支持向量機」**（僅 linear kernel；kernel／非線性列為未做）。
- **雙階段敘事**：階段1 硬間隔／最大化 margin（**表面完全不露 C**，含頁面 caption 與階段選項標籤）；階段2 Soft Margin（**以 C 為主角**，進入後才出現 C）。
- UI 用 **horizontal radio「學習階段」**（非 `st.tabs`）：專案依賴 `streamlit>=1.50`，舊版 tabs 無法可靠回報作用中分頁，會導致 Agent `SESSION_PAGE_FOCUS` 過期。切換 radio 即更新焦點。
- 階段1 公式：預設只露完整 hard-margin 定義；平行線距離→等價轉換→限制條件的長推導放 **expander**。
- 階段2 公式：預設多露 hinge＋C 短說明，**不做** 階段1 同等級長推導。
- 兩階段各有完整 sklearn 範例（expander）；階段1 程式 **不出現 `C=`**，階段2 明確寫 `C=...`。
- **訓練前預測**：兩階段各兩題解鎖＋每題 Agent 提示（同回歸殼）；**不做** NN 實驗迴圈。
- **僅內建資料**：階段1 `svm_blobs_80.csv`（可分）；階段2 `svm_soft_margin_80.csv`（重疊）。
- 舊「手寫 hinge 教學示意」主路徑已拿掉（不掛 UI）。

### 階段1 兩題

1. 只有 \(\min\frac12\|w\|^2\)、沒有限制 → \(w=0\)，無有效邊界。  
2. Margin 變大 → \(\|w\|\) 變小。

### 階段2 兩題

1. 不可分時硬間隔 → 找不到滿足所有限制的解，需要 soft margin。  
2. C 變大 → 更在意分對，margin 常變窄。

### 關鍵路徑

| 角色 | 路徑 |
|---|---|
| UI | `ui/svm_ui.py` |
| Quiz helpers | `ui/svm_quiz.py` |
| 模型 | `ml/svm.py`（`build_svm_agent_context(..., include_C=)`） |
| 測試 | `tests/test_svm_quiz.py`、`tests/test_svm_model.py` |

### 計劃檔（參考）

`~/.cursor/plans/svm_雙_tab_重構_c636ad5b.plan.md`

---

## 6. 跨頁共用模式（給實作者）

```text
訓練前預測殼
  ├─ 兩題 radio（含「請選擇」）
  ├─ 每題「Agent 提示」→ invoke_data_agent + extra_context
  ├─ both_correct → 解鎖「開始訓練」
  ├─ pair 變更 → 重置作答
  └─ agent context：狀態＋「未解鎖勿洩漏正解」
```

```text
NN 實驗殼（僅 NN）
  ├─ nn_form.json 寫回左欄
  ├─ nn_train_request.json 請求動畫訓練
  ├─ status: idle | need_train | need_agent_decision
  └─ max_runs 1～5，UI 權威
```

複用優先：`invoke_data_agent`、`render_chat_panel(extra_context=...)`、`build_*_agent_context` 附加 appendix。  
**不要**為提示按鈕新開 MCP／自訂 tool。

---

## 7. 刻意未做／以後再說

- 其餘分類頁（邏輯迴歸／樹／XGBoost）全面加預測關卡或寫回（**SVM 已做雙 Tab＋預測**）。
- NN 級實驗迴圈套到回歸／SVM 頁。
- 硬鎖「一定要跟 Agent 講過才能訓練」（目前是答題解鎖；Agent 是輔助）。
- Kernel／非線性 SVM；Tab1 範例寫 `C=1e6`。
- 無上限搜參、背景長時間訓練。

---

## 8. 建議下一手順序

1. 若課堂仍冷：再評估邏輯迴歸是否用「決策邊界／α」預測關卡（仍不宜上 NN 迴圈）。
2. 抽共用 `pretrain_quiz` UI 殼（回歸＋SVM 已落地，可評估是否抽殼）。
3. 樹／XGBoost 是否加輕量預測關卡——按需，勿預設抄 NN 迴圈。

---

## 9. 相關 Cursor 計劃檔（本機）

| 主題 | 檔案 |
|---|---|
| NN 寫回＋動畫迴圈 | `C:\Users\mz038\.cursor\plans\nn_agent_寫回左欄_7405161d.plan.md` |
| 單變量預測 | `C:\Users\mz038\.cursor\plans\單變量訓練前預測_bb4a2b3b.plan.md` |
| 多變量預測（已實作） | `C:\Users\mz038\.cursor\plans\多變量訓練前預測_purpose.plan.md` |
| SVM 雙 Tab＋預測 | `C:\Users\mz038\.cursor\plans\svm_雙_tab_重構_c636ad5b.plan.md` |

本文件若與計劃檔衝突，以**程式現況 + 本文件最新決策**為準，並更新本文件。
