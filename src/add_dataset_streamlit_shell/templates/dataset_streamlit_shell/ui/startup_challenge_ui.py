"""專案展示 live UI：模型區與成果區兩個函式。

只改這兩個函式本體。不要複製線性回歸／邏輯迴歸頁的雙欄、聊天欄、決策槽。
不要開內容區雙欄殼、不要畫聊天欄、不要 st.stop、不要 st.set_page_config，
也不要把 Challenge host context 當路徑物件來呼叫。
"""

from __future__ import annotations

from dataset_streamlit_shell.ui.startup_challenge_context import ChallengePaths


def render_model_zone(paths: ChallengePaths) -> None:
    """有 Challenge 訓練資料與測試資料後，在此寫選型與訓練。不要放成果圖表。"""
    return


def render_result_zone(paths: ChallengePaths) -> None:
    """有 Challenge 模型產物後，在此寫指標、圖與一次演示。"""
    return
