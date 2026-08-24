# 邏輯迴歸損失框只寫名稱，圖用原尺度

損失函數框上的目前選擇只寫「對數損失」（階段2 併 λ），不寫 `from_logits=True` 或 Keras 類名。模型程式碼預覽與模型下載程式碼的 compile 確定為 `BinaryCrossentropy(from_logits=True)`，線性層為 `Dense(1, linear)`；圖與表的 ŷ 仍是 σ(z)。不採用 `CategoricalCrossentropy`：本頁標籤是 0/1，不是 one-hot。兩階段都做特徵縮放，且為課堂四選一（組模型預設 Z分數正規化）。階段「多項式與 λ」先縮放原始兩特徵，再做鎖定的 degree=6。散點、決策邊界與 contour 畫在原始特徵尺度，訓練在縮放後空間再映回。

**Considered Options**: 框上也寫 from_logits=True（拒：課堂詞是對數損失，Keras 旗標留給程式）；Dense(1, sigmoid) 搭配 from_logits=True（拒：雙重壓縮，類神經網路頁已禁）；階段2 不縮放（拒：兩階段都要縮放）；兩階段鎖死 Z分數（拒：縮放槽變成假決策）；先展開 27 維再縮放（拒：四種公式打在 x²／xy 上）；圖改畫縮放後座標（拒：與線性回歸「散點原尺度、線映回」不一致）。
