from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.ui.data_ui import SHELL_ROOT, render_chat_panel, render_dataset_metrics
from dataset_streamlit_shell.ml.decision_tree import HEART_TARGET, RANDOM_STATE
from dataset_streamlit_shell.ml.xgboost_model import (
    FINAL_LEARNING_RATE,
    FINAL_N_ESTIMATORS,
    LEARNING_RATE_LIST,
    N_ESTIMATORS_LIST,
    best_iteration,
    build_xgboost_agent_context,
    fit_xgboost_final,
    prepare_encoded_heart,
    sweep_xgboost_hyperparam,
    training_and_validation_accuracy,
)
from dataset_streamlit_shell.plotting import (
    build_classification_data_figures,
    build_hyperparam_sweep_figure,
    configure_matplotlib_for_traditional_chinese,
    render_figures_in_streamlit,
)

configure_matplotlib_for_traditional_chinese()

HEART_PATH = SHELL_ROOT / "built-in-data" / "classification" / "heart_disease.csv"
HEART_VIZ_FEATURES = ["年齡", "膽固醇"]


def render_xgboost_page() -> None:
    df = pd.read_csv(HEART_PATH)
    context_key = "XGBoost_agent_context"
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title("XGBoost")
        st.caption(
            "使用心臟病資料練習 one-hot 編碼、訓練／驗證切分，"
            "並觀察 n_estimators、learning_rate 與 early stopping。"
        )
        st.success("目前使用本頁內建教學資料。")
        render_dataset_metrics(df)
        _render_heart_data_intro(df)
        _render_xgboost_model_notes()
        st.markdown("##### 訓練設定")
        st.caption(f"切分比例：訓練 80%／驗證 20%；random_state={RANDOM_STATE}。")
        result_key = "xgboost_last_result"
        train_clicked = st.button(
            "開始訓練",
            type="primary",
            width="stretch",
            key="train_xgboost",
        )
        if train_clicked:
            with st.spinner("正在 one-hot 編碼、切分資料並訓練多組 XGBoost（可能需要數十秒）..."):
                try:
                    x_train, x_val, y_train, y_val = prepare_encoded_heart(df)
                except ValueError as exc:
                    st.error(str(exc))
                    return
                n_sweep = sweep_xgboost_hyperparam(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    param_name="n_estimators",
                    values=N_ESTIMATORS_LIST,
                )
                lr_sweep = sweep_xgboost_hyperparam(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    param_name="learning_rate",
                    values=LEARNING_RATE_LIST,
                )
                final_model = fit_xgboost_final(x_train, y_train)
                train_acc, val_acc = training_and_validation_accuracy(
                    final_model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                )
            st.session_state[result_key] = {
                "n_sweep": n_sweep,
                "lr_sweep": lr_sweep,
                "train_rows": len(x_train),
                "val_rows": len(x_val),
                "feature_count": x_train.shape[1],
                "best_iter": best_iteration(final_model),
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
            st.session_state[context_key] = build_xgboost_agent_context(
                train_rows=len(x_train),
                val_rows=len(x_val),
                feature_count=x_train.shape[1],
                best_iteration_value=best_iteration(final_model),
                train_accuracy=train_acc,
                val_accuracy=val_acc,
            )
            _render_xgboost_results(st.session_state[result_key])
        elif result_key in st.session_state:
            st.caption("顯示最近一次訓練結果；請重新按「開始訓練」以更新。")
            _render_xgboost_results(st.session_state[result_key])
        else:
            st.info("按下「開始訓練」以產生超參數掃描圖與最終 XGBoost 模型。")
        _render_xgboost_prompts()
    with side:
        render_chat_panel(
            extra_context=str(st.session_state.get(context_key, "目前頁面：XGBoost。")),
            page_name="XGBoost",
        )


def _render_heart_data_intro(frame: pd.DataFrame) -> None:
    st.markdown("##### Data 資訊")
    st.info(
        "每一列是一位病患的檢查紀錄；target 為心臟病（1=有、0=無）。"
        "訓練前會對性別、胸痛類型等類別欄自動 one-hot 編碼。"
    )
    features = [column for column in frame.columns if column != HEART_TARGET]
    role_rows = []
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        is_target = column == HEART_TARGET
        numeric_ratio = float(series.notna().mean())
        if numeric_ratio > 0.9 and column not in {"性別", "胸痛類型", "靜息心電圖", "運動心絞痛", "ST斜率"}:
            stats = {
                "最小值": float(series.min()),
                "最大值": float(series.max()),
                "平均值": float(series.mean()),
            }
        else:
            stats = {"最小值": None, "最大值": None, "平均值": None}
        role_rows.append(
            {
                "欄位": column,
                "角色": "target（y）" if is_target else "feature（x）",
                "資料型態": str(frame[column].dtype),
                "缺失值": int(frame[column].isna().sum()),
                **stats,
            }
        )
    display = pd.DataFrame(role_rows)
    st.dataframe(display, width="stretch", hide_index=True)
    with st.expander("資料預覽", expanded=True):
        st.dataframe(frame.head(10), width="stretch", hide_index=True)
    viz_features = [column for column in HEART_VIZ_FEATURES if column in features]
    if len(viz_features) >= 2:
        numeric_frame = frame[viz_features + [HEART_TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
        render_figures_in_streamlit(
            build_classification_data_figures(numeric_frame, viz_features, HEART_TARGET)
        )


def _render_xgboost_model_notes() -> None:
    st.markdown("##### 模型說明")
    st.caption(
        "XGBoost 以多棵弱樹序列加總預測；本頁使用 XGBClassifier，"
        f"掃描後以最終 n_estimators={FINAL_N_ESTIMATORS}、learning_rate={FINAL_LEARNING_RATE} "
        "在訓練集內再切 eval_set 做 early stopping。"
    )


def _render_xgboost_results(cached: dict) -> None:
    st.markdown("##### 訓練結果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("訓練筆數", f"{cached['train_rows']:,}")
    c2.metric("驗證筆數", f"{cached['val_rows']:,}")
    c3.metric("最終訓練準確率", f"{cached['train_acc']:.2f}%")
    c4.metric("最終驗證準確率", f"{cached['val_acc']:.2f}%")
    if cached.get("best_iter") is not None:
        st.caption(f"early stopping 最佳迭代（best_iteration）：{cached['best_iter']}")
    with st.expander("編碼後 feature 數", expanded=False):
        st.write(f"one-hot 後共 {cached['feature_count']} 個 feature 欄位。")

    n_sweep = cached["n_sweep"]
    fig_n = build_hyperparam_sweep_figure(
        param_label="n_estimators",
        values=n_sweep["values"],
        train_accuracy=n_sweep["train_accuracy"],
        val_accuracy=n_sweep["val_accuracy"],
    )
    st.pyplot(fig_n, clear_figure=True)
    plt.close(fig_n)

    lr_sweep = cached["lr_sweep"]
    fig_lr = build_hyperparam_sweep_figure(
        param_label="learning_rate",
        values=lr_sweep["values"],
        train_accuracy=lr_sweep["train_accuracy"],
        val_accuracy=lr_sweep["val_accuracy"],
    )
    st.pyplot(fig_lr, clear_figure=True)
    plt.close(fig_lr)


def _render_xgboost_prompts() -> None:
    st.markdown("##### 建議問 Agent")
    prompts = [
        "驗證集準確率比訓練集低很多，是否代表過擬合？可以從掃描圖怎麼看？",
        "n_estimators 變大時，為什麼訓練準確率上升但驗證不一定？",
        "early stopping 的 eval_set 和畫面上的驗證集（20%）有什麼不同？",
    ]
    for prompt in prompts:
        st.code(prompt, language="text")
