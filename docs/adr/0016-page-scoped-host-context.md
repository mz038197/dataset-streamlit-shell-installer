# 教學頁自帶 host，不叠 dataset_base_context

整理線與教學頁叠在同一條 `dataset_base_context()` 時，模型會把類神經網路的「左欄」講成線性回歸決策槽，並叫學生用下拉選單。定案與 ADR 0009 同一條教訓：衝突規則不叠。`dataset_base_context` 只留雙表／Working／Ready／腳本，給總覽、AI 協作資料整理、降維分析。監督式、非監督式、深度學習、電腦視覺各頁傳自己的教學頁 host context（內建範例、不准動根目錄 working／ready／original；有寫回協定的頁再接該頁 fragment），每一輪不叠 Working 列數。專案展示維持 Challenge host context。

**Considered Options**：繼續叠 fragment 只加「不要講下拉」；只拆線性回歸與類神經網路、其他教學頁仍吃 base。前者模型仍兩邊都聽。後者圖表頁還是看得到左欄寫回。選定各教學頁自傳 host。

**Consequences**：教學頁沒傳 `host_context=` 會退回整理線 base，屬迴歸。別頁不能改另一頁的模型寫回。
