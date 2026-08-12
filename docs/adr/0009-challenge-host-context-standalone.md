# Challenge host context 獨立、不叠 dataset_base_context

專案展示頁的 Agent 必須鎖定 `workspace/challenge/` 軌道與白板三塊完成樣貌。若叠加既有 `dataset_base_context()`，system 仍會引導改根目錄 Working／Ready／NN 表單，與挑戰邊界衝突。定案：專案展示使用獨立 Challenge host context（含公司加碼），並搭配獨立 Challenge Agent session。

**Considered Options**：叠加 base 並靠 fragment「衝突以挑戰為準」覆寫；或挑戰頁專用精簡 host。前者省事但模型常兩邊都聽。選定後者。

**Consequences**：`render_chat_panel` 需能注入自訂 `host_context` 與獨立 session 狀態鍵；挑戰頁不得共用整理線的 `data_agent` 快取。
