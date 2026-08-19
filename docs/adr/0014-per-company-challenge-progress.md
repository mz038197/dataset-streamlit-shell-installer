# 專案展示進度依挑戰公司分開，重整與換公司都不刪

重整頁面只把已確認的挑戰公司記在 session，學生切回自己的公司時會被當成更換挑戰公司並清掉 Challenge 工作資料。定案：已確認公司寫進磁碟、重整後仍有效；working／train／test、Challenge 模型產物、Challenge UI 快照、Challenge Agent session 皆各公司一份。換公司改存／載入快照，不再清除資料、也不再一律還原專案展示空殼。舊的共用 `workspace/challenge/working.csv`（及 train／test）只搬進學生確認的那一間；磁碟上還沒記住公司卻仍有舊檔時，第一次也要確認後才搬。

**Considered Options**：只記住公司、換公司仍清檔（治 F5、不能對照兩間）；換公司不清但五間共用一份 working（會串題）；舊檔複製到五間（四間帶錯表）。選定各公司一份並只把舊檔歸到確認的那一間。

**Consequences**：推翻 ADR-0011「換公司還原空殼」與「換公司清 runtime」。空殼只在該公司尚無快照時載入。Agent 只改 live UI 與目前公司資料夾。F5 不保存對話。安裝／`--force` 須把學生的 Challenge 工作資料／切分從備份還原，不可只留在 `.bak`。
