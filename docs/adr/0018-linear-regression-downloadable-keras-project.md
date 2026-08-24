# 線性回歸模型下載程式碼是獨立 Keras 專案

學生要帶走可在本頁以外執行的完整 Sequential 程式，並接受與主教學欄不同運算。決定六槽齊後下載目前學習階段一份 zip：`pyproject.toml`、`train.py`、該階段內建表、短 README；學生在自己的機器 `uv sync`。預覽維持只讀短碼且不含切分；頁面訓練仍走本頁梯度下降，不 exec 下載檔。不帶 `.venv`、不強制 `uv.lock`（lock 綁平台輪子）。

**Considered Options**: 把切分寫進模型程式碼預覽（拒：ADR 0017，預覽不是腳本）；下載檔釘零初始化／全批次以逼近頁面曲線（拒：帶走的應是外面會寫的 Keras 預設）；zip 附下載當下的 `uv.lock` 或 `.venv`（拒：換 OS 常壞、幾乎不能帶走）。

**Consequences**: Sequential／compile 與預覽同一段，另補切分（與頁面同一刀）與 `fit`。跑完印 Keras MSE 與 w、b，不複製教學圖。資料 Agent 可說明存在與差別，不得另貼程式或宣稱頁面在跑這份。
