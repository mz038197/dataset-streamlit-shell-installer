# 專案展示改為資料工作流，廢止白板三塊

專案展示若做成上台講稿三塊（問題／結果／紅線），學生看不見公司資料、也無法在頁上走完清理→切分→訓練。定案改為工作流空殼：上半 Challenge 資料檢視，下半模型區／成果區空輪廓；模型吃 `workspace/challenge/train.csv` 與 `test.csv`，不走 Ready、不寫根目錄切分檔。換公司時還原 `startup_challenge_ui.py` 為專案展示空殼（推翻「不還原 UI」），以免上一組模型畫面留在錯的公司上。獨立 Challenge host context（ADR-0009）仍有效；0009 文中的白板三塊視為舊頁結構。

**Considered Options**：保留白板三塊只加資料預覽；換公司只清資料不還原 UI。前者仍是講稿頁；後者在 Agent 已寫入真 UI 後會錯題。選定工作流空殼＋還原空殼。

**Consequences**：`startup_challenge_empty_shell.py` 為還原來源，Agent 不准編輯、也不准拆掉「無檔則顯示輪廓」。改 working 即刪 train／test。倫理紅線不上頁。
