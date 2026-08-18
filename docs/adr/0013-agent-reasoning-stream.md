# 資料 Agent 欄顯示思考過程，語音隱藏

資料實驗室殼要讓學生看見模型在答之前的推理摘要，但不能把整套殼換成 peas-agent-core（課堂仍走學生 `create_agent`）。決定由學生 Agent 串流 `on_reasoning`／`on_stream_reset`，殼只顯示「思考過程」expander；舊 `chat()` 沒有這些參數就只顯示答案。思考過程不寫進 session jsonl。語音設定從資料 Agent 欄隱藏且不播放，但保留 `user_settings.json` 的 tts 鍵。

**Considered Options**: 資料 Agent 欄改接 peas-agent-core（與 Studio 同一套，否決：課堂契約仍是 `create_agent`）；思考過程寫進 jsonl（否決：過程不是作業紀錄，且 runtime 訊息契約會膨脹）；硬性要求 `on_reasoning` 寫進 peas-agent-runtime 契約（否決：未升級的學生專案會安裝失敗）。

**Consequences**: `agent.chat` 以 signature 探測 callback。沒有思考文字就不畫 expander。`invoke_data_agent` 不接即時思考過程。跑工具時答案區為「（執行工具中…）」。本殼第一版不做 Studio 的推理深度旋鈕。
