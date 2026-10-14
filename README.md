# Dataset Streamlit Shell Installer

把教學用的 `dataset_streamlit_shell/` 裝進學生的 Agent 專案。殼層 = **AI Lab 框架** + 已加裝的**課程包**：先裝框架（總覽、資料 Agent 欄、側欄品牌列），上課到哪一段再 `add` 那一段的課程包；沒裝的課程包不出現在側欄。

## 安裝 AI Lab 框架

在要加裝殼層的專案根目錄執行：

```powershell
uvx --from git+https://github.com/mz038197/dataset-streamlit-shell-installer.git add-dataset-streamlit-shell
```

本機開發：

```powershell
uvx --from . add-dataset-streamlit-shell
```

只裝框架時側欄只有總覽，完成提示會建議下一步 `add data-science`。框架預設用 `uv add` 裝自己的套件（Streamlit、pandas、openai-tts、peas-agent-runtime）；專案旁有 `../peas-agent-runtime` 時改以 editable 安裝。

啟動：

```powershell
uv run streamlit run dataset_streamlit_shell/app.py
```

## 加裝課程包

以下指令都可加 `-C <專案根目錄>` 指定專案。下面以 `add-dataset-streamlit-shell` 代表上面的 `uvx --from ... add-dataset-streamlit-shell`。

```powershell
add-dataset-streamlit-shell add data-science            # 加裝一包
add-dataset-streamlit-shell add supervised unsupervised # 一次加裝多包
add-dataset-streamlit-shell add all                     # 加裝全部課程包
add-dataset-streamlit-shell list                        # 列出課程包、側欄分段、是否已安裝
```

- `add`：還沒裝框架時會先裝框架；已裝過的課程包顯示「已安裝」並略過；名稱打錯會列出可用名稱。只新增檔案，不覆蓋 `workspace/` 等保留目錄裡既有的檔。完成後**重新整理 Streamlit 頁面**，側欄就會出現新的分段。
- 加裝的課程包記在 `dataset_streamlit_shell/course_packs.json`（安裝紀錄）。
- 每個課程包只裝自己需要的額外套件。

| 指令名稱 | 側欄分段 | 額外套件 |
| --- | --- | --- |
| `data-science` | AI 協作資料整理 | matplotlib、scikit-learn |
| `supervised` | 機器學習 · 監督式 | numpy、matplotlib、plotly、scikit-learn、xgboost |
| `unsupervised` | 機器學習 · 非監督式 | numpy、matplotlib、scipy、scikit-learn |
| `deep-learning` | 深度學習 | numpy、matplotlib、tensorflow-cpu |
| `computer-vision` | 電腦視覺 | numpy、matplotlib、scikit-learn、tensorflow-cpu、torch、torchvision、ultralytics、CLIP、opencv-python-headless、pillow、gdown |
| `pca` | 降維分析 | numpy |
| `startup-workshop` | AI新創工作坊 | numpy、matplotlib |

課程包之間不共用程式：兩包都要用的功能各帶一份副本，只裝其中一包也能完整使用。降維分析讀 AI 協作資料整理產生的 Working；只裝 `pca` 時頁面會提示先 `add data-science`。

## 更新與重裝

```powershell
add-dataset-streamlit-shell --update   # 更新框架與已裝的課程包（依安裝紀錄，不多裝）
add-dataset-streamlit-shell --force    # 備份成 dataset_streamlit_shell.bakN 後重裝
```

- `--update`：刪掉保留目錄以外的舊檔再重新複製，保留：
  - `dataset_streamlit_shell/workspace/`（含 `workspace/challenge/` 學生進度、`user_settings.json`）
  - `dataset_streamlit_shell/scripts/`、`uploads/`、`memory/`
  - `sessions/*.jsonl`（專案根；對話紀錄）
  - `built-in-data/computer-vision/models/` 下已下載的 `*.pt` 權重
- `--force`：依安裝紀錄重裝框架與已裝的課程包；Challenge 工作資料／訓練／測試會從備份還原，不會只留在 `.bak`。
- 舊版安裝（沒有 `course_packs.json`）：`--update` 會報錯並建議改用 `--force`；`--force` 備份後只重裝框架，再用 `add` 加裝需要的課程包。
- `--force` 與 `--update` 不能同時使用；殼層已存在又沒帶任一旗標時拒絕覆蓋。

## 其他選項

- `--no-install-deps`：安裝、`add`、`--update` 都不執行 `uv add`；目標專案須自備 Streamlit 1.61 以上與各課程包的套件。
- `--require-agent-contract`：安裝後必須通過 `create_agent` 契約檢查，否則非零結束。未接上 Agent 也能安裝，只會提示。
- `--require-agent-core`：舊旗標，請改用 `--require-agent-contract`。

## 資料 Agent 欄

- 學生的 `create_agent` 串流推理（Responses API）時，欄位顯示可收合的 **思考過程**；只留在 Streamlit session 直到重新整理，不寫入 `sessions/*.jsonl`。只支援 `on_token` 的 Agent 照常可用，只是只顯示回答。
- 資料 Agent 欄**不顯示也不播放 TTS**。`workspace/user_settings.json` 可能留有舊版的 `tts_*` 鍵；檔案不存在時殼層會建立，但不據此驅動介面或播放。

## Promptable Segmentation（SAM 3）

電腦視覺課程包的 **Promptable Segmentation** 頁使用 SAM 3（`sam3.pt`，約 3.4 GB），權重**不**放在 git。第一次使用時在該頁按 **Download SAM 3 weights**（課程用 Google Drive），或手動放到：

```text
dataset_streamlit_shell/built-in-data/computer-vision/models/sam3.pt
```

需要穩定網路、約 3.4 GB 磁碟空間與 `gdown`（隨 `computer-vision` 課程包安裝）。權重僅供課程使用；`--update` 會保留已下載的 `*.pt`。
