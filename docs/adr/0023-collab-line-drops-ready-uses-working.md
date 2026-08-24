# 協作線分析頁改讀 Working，廢止 Ready

圖表探索、資料切分、PCA 曾鎖凍結的 `ready.csv`，學生必須多走「建立 Ready 分析就緒資料」頁；該頁也不是品質關卡，繼續清 Working 時圖表仍看舊快照。定案：刪除 Ready 概念與側欄頁；上述頁直接讀 Working 工作資料，沒有 Working 才空狀態，不因髒表硬鎖。寫回 Working 即作廢切分產物，與 Challenge 線相同。監督式／非監督式教學頁仍用內建範例，不改掛 Working。

**Considered Options**: 只解鎖但仍保留 ready.csv 匯出（拒：沒人讀的幽靈檔）；分析頁維持凍結快照（拒：Ready 頁已是阻力）；PCA 繼續鎖 Ready（拒：Ready 會從整理線消失卻活在降維）。

**Consequences**: ADR 0003 的分群內建資料仍成立；其中「圖表探索等分析頁仍讀 ready」不再適用。側欄只刪 Ready 頁，圖表探索不前移。總覽可下載 Working。寫回 Working 與清除回雙表起點時刪掉遺留 ready.csv。
