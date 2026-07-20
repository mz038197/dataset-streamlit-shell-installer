from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.tree import export_text

from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    render_chat_panel,
    render_dataset_metrics,
)
from dataset_streamlit_shell.ml.decision_tree import (
    CAT_FEATURES,
    CAT_TARGET,
    CRITERION_CHOICES,
    build_decision_tree_agent_context,
    fit_decision_tree,
    information_gain_table,
    training_accuracy,
)
from dataset_streamlit_shell.plotting import (
    build_classification_data_figures,
    build_decision_tree_figure,
    configure_matplotlib_for_traditional_chinese,
    render_figures_in_streamlit,
)

configure_matplotlib_for_traditional_chinese()

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
CAT_TOY_PATH = CLASSIFICATION_DEMO_DIR / "cat_toy_10.csv"


def render_decision_tree_concepts_page() -> None:
    df = pd.read_csv(CAT_TOY_PATH)
    features = list(CAT_FEATURES)
    target = CAT_TARGET
    working = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()

    context_key = "決策樹概念_agent_context"
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title("決策樹概念")
        st.caption("使用 10 筆玩具資料理解熵、資訊增益，並以 sklearn 決策樹觀察分裂結果。")
        st.success("目前使用本頁內建教學資料。")
        render_dataset_metrics(df)
        _render_tree_data_intro(working, features=features, target=target)
        _render_tree_formulas()
        ig_table = information_gain_table(working, features, target)
        st.markdown("##### 各 feature 資訊增益表")
        st.caption("在根節點（全部樣本）計算；數值愈大代表分裂後愈能降低不純度。")
        st.dataframe(
            ig_table.style.format({"資訊增益": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("##### 訓練設定")
        c1, c2 = st.columns(2)
        criterion_label = c1.radio(
            "分裂準則 criterion",
            list(CRITERION_CHOICES.keys()),
            horizontal=True,
            index=0,
            key="dt_criterion",
        )
        max_depth = c2.number_input(
            "最大深度 max_depth",
            min_value=1,
            max_value=2,
            value=1,
            step=1,
            key="dt_max_depth",
        )
        criterion = CRITERION_CHOICES[criterion_label]
        result_key = "decision_tree_last_result"
        signature = (criterion, int(max_depth), len(working))
        train_clicked = st.button(
            "開始訓練",
            type="primary",
            use_container_width=True,
            key="train_decision_tree",
        )
        if train_clicked:
            try:
                model = fit_decision_tree(
                    working[features],
                    working[target],
                    max_depth=int(max_depth),
                    criterion=criterion,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            accuracy = training_accuracy(model, working[features], working[target])
            st.session_state[result_key] = {
                "signature": signature,
                "model": model,
                "criterion_label": criterion_label,
                "max_depth": int(max_depth),
                "accuracy": accuracy,
            }
            st.session_state[context_key] = build_decision_tree_agent_context(
                features=features,
                target=target,
                max_depth=int(max_depth),
                criterion_label=criterion_label,
                training_accuracy_pct=accuracy,
                row_count=len(working),
            )
            _render_tree_training_results(
                model,
                working=working,
                features=features,
                criterion_label=criterion_label,
                max_depth=int(max_depth),
                accuracy=accuracy,
            )
        elif result_key in st.session_state and st.session_state[result_key]["signature"] == signature:
            cached = st.session_state[result_key]
            st.caption("顯示最近一次訓練結果；調整設定後請重新按「開始訓練」。")
            _render_tree_training_results(
                cached["model"],
                working=working,
                features=features,
                criterion_label=cached["criterion_label"],
                max_depth=cached["max_depth"],
                accuracy=cached["accuracy"],
            )
        else:
            st.info("選擇分裂準則與 max_depth 後，按下「開始訓練」以顯示決策樹。")
        _render_tree_prompts()
    with side:
        render_chat_panel(
            extra_context=str(st.session_state.get(context_key, "目前頁面：決策樹概念。")),
            page_name="決策樹概念",
        )


def _render_tree_data_intro(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
) -> None:
    st.markdown("##### Data 資訊")
    st.info("每一列代表一隻動物：三個 0/1 特徵描述外觀，target 為是否為貓（1=是、0=否）。")
    role_rows = []
    for column in features + [target]:
        series = pd.to_numeric(frame[column], errors="coerce")
        role_rows.append(
            {
                "欄位": column,
                "角色": "target（y）" if column == target else "feature（x）",
                "資料型態": str(frame[column].dtype),
                "缺失值": int(frame[column].isna().sum()),
                "最小值": float(series.min()),
                "最大值": float(series.max()),
                "平均值": float(series.mean()),
            }
        )
    st.dataframe(
        pd.DataFrame(role_rows).style.format(
            {"最小值": "{:.4f}", "最大值": "{:.4f}", "平均值": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("資料預覽", expanded=True):
        st.dataframe(frame[features + [target]].head(10), use_container_width=True, hide_index=True)
    render_figures_in_streamlit(build_classification_data_figures(frame, features, target))


def _render_tree_formulas() -> None:
    st.markdown("##### 模型公式")
    with st.expander("熵與資訊增益（手算）", expanded=True):
        st.latex(
            r"H(p_1) = -p_1 \log_2(p_1) - (1-p_1)\log_2(1-p_1)"
        )
        st.latex(
            r"\mathrm{IG} = H(p_1^{\mathrm{node}}) - \big(w^{\mathrm{left}} H(p_1^{\mathrm{left}}) + w^{\mathrm{right}} H(p_1^{\mathrm{right}})\big)"
        )
        st.caption(
            "上方資訊增益表依 log₂ 熵計算；與下方選 Entropy 時 sklearn 的分裂精神相同，"
            "但 sklearn 實作使用自然對數計算不純度。"
        )
    with st.expander("分裂準則：Gini 與 Entropy（sklearn criterion）", expanded=True):
        st.latex(r"G = 1 - \sum_{k=1}^{K} p_k^2")
        st.caption("二元分類可化簡為 G = 2p₁(1-p₁)。")
        st.latex(r"H = -\sum_{k=1}^{K} p_k \ln p_k")
        st.caption(
            "訓練時 DecisionTreeClassifier 在每個節點選不純度下降最大的分裂；"
            "選 Gini 用 G，選 Entropy 用 H。"
        )


def _render_tree_training_results(
    model,
    *,
    working: pd.DataFrame,
    features: list[str],
    criterion_label: str,
    max_depth: int,
    accuracy: float,
) -> None:
    st.markdown("##### 訓練結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("分裂準則", criterion_label)
    c2.metric("max_depth", str(max_depth))
    c3.metric("訓練集正確率", f"{accuracy:.2f}%")
    st.markdown("##### 決策樹圖")
    fig = build_decision_tree_figure(
        model,
        feature_names=features,
        class_names=["非貓", "貓"],
    )
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    with st.expander("文字版決策樹（export_text）", expanded=False):
        tree_text = export_text(
            model,
            feature_names=features,
            class_names=["非貓", "貓"],
        )
        st.code(tree_text, language="text")


def _render_tree_prompts() -> None:
    st.markdown("##### 建議問 Agent")
    prompts = [
        "為什麼資訊增益表排名第一的 feature，可能和 sklearn 樹根節點選的 feature 不同？",
        "同一 max_depth 下，Gini 與 Entropy 畫出的樹會一樣嗎？請對照本頁結果說明。",
        "max_depth=1 和 max_depth=2 的葉節點數有什麼差別？",
    ]
    for prompt in prompts:
        st.code(prompt, language="text")
