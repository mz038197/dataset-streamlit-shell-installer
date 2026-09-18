"""專案展示 live UI：模型區與成果區兩個函式。

只改這兩個函式本體。不要複製線性回歸／邏輯迴歸頁的雙欄、聊天欄、決策槽。
不要開內容區雙欄殼、不要畫聊天欄、不要 st.stop、不要 st.set_page_config，
也不要把 Challenge host context 當路徑物件來呼叫。

頂層保留 ChallengePaths。需要訓練時再加（可刪沒用到的名字）：
import pandas as pd
import streamlit as st
from dataset_streamlit_shell.ml.regression import apply_feature_scaler, attach_test_costs, create_feature_scaler, gradient_descent_steps, predict_with_parameters
from dataset_streamlit_shell.ml.classification import attach_logistic_test_costs, confusion_matrix_counts, logistic_gradient_descent_steps, predict_class_from_proba, predict_proba, precision_recall_f1_from_counts, sample_gradient_steps
不要 from ml.xxx、不要 from .xxx、不要 import lr_ui 或 logistic_regression_ui。
"""

from __future__ import annotations

from dataset_streamlit_shell.ui.startup_challenge_context import ChallengePaths


def render_model_zone(paths: ChallengePaths) -> None:
    """有 Challenge 訓練資料與測試資料後，在此寫選型與訓練。不要放成果圖表。"""
    return


def render_result_zone(paths: ChallengePaths) -> None:
    """有 Challenge 模型產物後，在此寫指標、圖與一次演示。"""
    return
