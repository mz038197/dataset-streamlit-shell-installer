# 非監督式分群教學頁改用內建範例資料

K-Means 分群與 Ward's Method 原走 `render_analysis_shell`，沒有 `ready.csv` 就鎖頁，與「用內建範例走通演算法」的課程目標衝突。定案：兩頁僅使用共用內建範例（不提供 ready 切換）。當時分析頁仍讀 ready；該句已被 ADR 0023 取代。

**Considered Options**: 沒有 ready 時 fallback 到內建、同時保留 ready 切換；或維持 ready 閘門。前者讓資料路徑分叉、後者阻擋未完成整理的學生。選定與監督式教學頁相同的「只吃內建」。

**Consequences**: 總覽文案需維持「分群教學頁不吃整理線工作資料」；日後若要讓學生對自有資料分群，應另開分析用途頁，而不是把教學頁掛回 Working。圖表探索等分析頁改讀 Working、廢止 Ready，見 ADR 0023。
