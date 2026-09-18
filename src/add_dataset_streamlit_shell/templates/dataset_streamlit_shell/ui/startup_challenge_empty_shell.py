"""專案展示 live UI：模型區與成果區兩個函式。

只改這兩個函式本體。不要複製線性回歸／邏輯迴歸頁的雙欄、聊天欄、決策槽。
不要開內容區雙欄殼、不要畫聊天欄、不要 st.stop、不要 st.set_page_config，
也不要把 Challenge host context 當路徑物件來呼叫。
"""

from __future__ import annotations

import pickle

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression

from dataset_streamlit_shell.ui.startup_challenge_context import ChallengePaths

_FITTED_MODEL_NAME = "challenge_fitted_model.pkl"
_TARGET_KEY = "challenge_model_target"
_TRAIN_KEY = "challenge_model_train"


def _fitted_model_path(paths: ChallengePaths):
    return paths.working_csv.with_name(_FITTED_MODEL_NAME)


def _features_and_target(frame: pd.DataFrame, target: str, columns: list[str] | None = None):
    x = pd.get_dummies(frame.drop(columns=[target]), dummy_na=True)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0)
    return x, frame[target], list(x.columns)


def render_model_zone(paths: ChallengePaths) -> None:
    """選型與訓練。不要放成果圖表。"""
    train = pd.read_csv(paths.train_csv)
    if train.empty or len(train.columns) < 2:
        st.error("Challenge 訓練資料欄位不足。")
        return
    # ponytail: 目標欄預設最後一欄；依該公司說明書改。
    target = str(
        st.selectbox(
            "目標欄",
            list(train.columns),
            index=len(train.columns) - 1,
            key=_TARGET_KEY,
        )
    )
    if not st.button("開始訓練", type="primary", key=_TRAIN_KEY):
        return
    x, y, columns = _features_and_target(train, target)
    if pd.api.types.is_numeric_dtype(y) and int(y.nunique(dropna=True)) > 12:
        model: object = LinearRegression()
        task = "reg"
    else:
        y = y.astype(str)
        model = LogisticRegression(max_iter=500)
        task = "clf"
    model.fit(x, y)
    _fitted_model_path(paths).write_bytes(
        pickle.dumps({"model": model, "target": target, "columns": columns, "task": task})
    )
    st.session_state["challenge_model_artifact"] = True
    st.success("訓練完成。成果區會顯示指標。")


def render_result_zone(paths: ChallengePaths) -> None:
    """訓練後的指標與一次對照。沒有產物時頁骨架不會呼叫本函式。"""
    packed_path = _fitted_model_path(paths)
    if not packed_path.is_file():
        st.caption("訓練後這裡會出現測試集指標。")
        return
    packed = pickle.loads(packed_path.read_bytes())
    test = pd.read_csv(paths.test_csv)
    target = packed["target"]
    if target not in test.columns:
        st.error(f"Challenge 測試資料沒有目標欄 {target}。")
        return
    x, y, _ = _features_and_target(test, target, packed["columns"])
    if packed["task"] == "clf":
        y = y.astype(str)
        pred = packed["model"].predict(x)
        acc = float((pred == y).mean())
        st.metric("測試集正確率", f"{acc:.1%}")
    else:
        pred = packed["model"].predict(x)
        mae = float(abs(pred - y.astype(float)).mean())
        st.metric("測試集 MAE", f"{mae:.3f}")
    st.dataframe(
        pd.DataFrame({"實際": list(y), "預測": list(pred)}).head(20),
        width="stretch",
        hide_index=True,
    )
