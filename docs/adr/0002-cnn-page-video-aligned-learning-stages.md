# 卷積神經網路頁以觀念主軸影片對齊學習階段

學生反映不懂 CNN 原理；現有七個 tabs 資訊雖齊，敘事與淺顯影片不同步。決定以 DataMListic《CNNs - Explained》（`YGILT182T6w`）為觀念主軸：開頭保留「圖片＝矩陣」，其後跟影片動機與運算順序；導覽改為六段 horizontal radio「學習階段」（不用 tabs）；階段 1 不嵌片，階段 2～6 嵌同一支片並附建議時間戳；正文繁中。階段 6 兩題訓練前預測（kernel／卷積、pooling）解鎖訓練。Hubel／LeNet 史話降為 expander；第一版不灌 inductive bias 專名。

**Considered Options**: 只局部改「為什麼需要 CNN」與卷積兩 tab（改動小但難成主軸）；影片僅備課不進 UI（學生感受不到「帶出」）；維持 tabs 或單頁長捲（較難強制跟片順序）。選定影片主軸＋六階段＋分段嵌片。

**Consequences**: 本頁導覽與 K-近鄰分類一致（學習階段／訓練前預測），需從 `st.tabs` 殼遷出；時間戳若影片改版需人工對一次。日後若要加 inductive bias 或 Dense 動機當解鎖題，屬明確加料，不視為本 ADR 默認範圍。
