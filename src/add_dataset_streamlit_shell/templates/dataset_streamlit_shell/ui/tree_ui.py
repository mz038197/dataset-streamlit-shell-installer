"""決策樹與集成：三學習階段＋訓練前預測＋Bagging／Boosting 對照。"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.tree import export_text

from dataset_streamlit_shell.ml.decision_tree import (
    CAT_FEATURES,
    CAT_TARGET,
    CRITERION_CHOICES,
    HEART_TARGET,
    RANDOM_STATE,
    build_decision_tree_agent_context,
    fit_decision_tree,
    information_gain_table,
    prepare_heart_splits,
    training_accuracy,
)
from dataset_streamlit_shell.ml.random_forest import (
    BASELINE_FOREST_N_ESTIMATORS,
    compare_single_tree_vs_forest,
    forest_validation_baseline,
)
from dataset_streamlit_shell.ml.xgboost_model import (
    STAGE_DEFAULT_LEARNING_RATE,
    STAGE_FIXED_N_ESTIMATORS,
    fit_xgboost_stage,
    training_and_validation_accuracy,
)
from dataset_streamlit_shell.plotting import (
    build_classification_data_figures,
    build_decision_tree_figure,
    configure_matplotlib_for_traditional_chinese,
    render_figures_in_streamlit,
)
from dataset_streamlit_shell.ui import tree_ensemble_quiz as quiz
from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    invoke_data_agent,
    render_chat_panel,
    render_dataset_metrics,
)

configure_matplotlib_for_traditional_chinese()

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
CAT_TOY_PATH = CLASSIFICATION_DEMO_DIR / "cat_toy_10.csv"
HEART_PATH = CLASSIFICATION_DEMO_DIR / "heart_disease.csv"

PAGE_TITLE = "決策樹與集成"
CONTEXT_KEY = f"{PAGE_TITLE}_agent_context"


def render_decision_tree_concepts_page() -> None:
    """保留函式名以相容既有 page 入口；頁面標題為決策樹與集成。"""
    render_tree_ensemble_page()


def render_tree_ensemble_page() -> None:
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title(PAGE_TITLE)
        st.caption("從單顆決策樹走到隨機森林與 XGBoost，對照 Bagging 與 Boosting。")

        stage = st.radio(
            "學習階段",
            list(quiz.learning_stage_labels()),
            horizontal=True,
            key="tree_ensemble_learning_stage",
        )
        if stage == quiz.STAGE1_LABEL:
            st.session_state[quiz.SESSION_PAGE_FOCUS] = "tree"
            _render_single_tree_stage()
        elif stage == quiz.STAGE2_LABEL:
            st.session_state[quiz.SESSION_PAGE_FOCUS] = "bagging"
            _render_random_forest_stage()
        else:
            st.session_state[quiz.SESSION_PAGE_FOCUS] = "boost"
            _render_xgboost_stage()

    with side:
        render_chat_panel(
            extra_context=str(st.session_state.get(CONTEXT_KEY, f"目前頁面：{PAGE_TITLE}。")),
            page_name=PAGE_TITLE,
        )


def _render_single_tree_stage() -> None:
    st.markdown("##### 這一階段在問什麼")
    st.info("一棵決策樹怎麼選分裂：用熵／資訊增益衡量不純度下降。")

    df = pd.read_csv(CAT_TOY_PATH)
    features = list(CAT_FEATURES)
    target = CAT_TARGET
    working = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()

    st.success("目前使用本階段內建玩具資料（10 筆）。")
    render_dataset_metrics(df)
    _render_tree_data_intro(working, features=features, target=target)
    _render_tree_formulas()
    ig_table = information_gain_table(working, features, target)
    st.markdown("##### 各 feature 資訊增益表")
    st.caption("在根節點（全部樣本）計算；數值愈大代表分裂後愈能降低不純度。")
    st.dataframe(
        ig_table.style.format({"資訊增益": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    unlocked = _render_tree_pretrain_quiz()

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

    st.markdown("##### 訓練")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_decision_tree",
        disabled=not unlocked,
    )
    if not unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")

    if train_clicked and unlocked:
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
        st.session_state[CONTEXT_KEY] = build_decision_tree_agent_context(
            features=features,
            target=target,
            max_depth=int(max_depth),
            criterion_label=criterion_label,
            training_accuracy_pct=accuracy,
            row_count=len(working),
        ) + "\n" + _tree_quiz_appendix(unlocked=True)

    if result_key in st.session_state and st.session_state[result_key]["signature"] == signature:
        cached = st.session_state[result_key]
        if not (train_clicked and unlocked):
            st.caption("顯示最近一次訓練結果；調整設定後請重新按「開始訓練」。")
        _render_tree_training_results(
            cached["model"],
            working=working,
            features=features,
            criterion_label=cached["criterion_label"],
            max_depth=cached["max_depth"],
            accuracy=cached["accuracy"],
        )
    elif unlocked:
        st.info("選擇分裂準則與 max_depth 後，按下「開始訓練」以顯示決策樹。")
    else:
        st.info("先完成上方兩題訓練前預測，再開始訓練。")

    _merge_stage_context(
        note="目前階段：單顆決策樹。\n" + _tree_quiz_appendix(unlocked=unlocked)
    )
    _render_tree_prompts()


def _render_random_forest_stage() -> None:
    st.markdown("##### 這一階段在問什麼")
    st.info(
        "隨機森林是 **Bagging**：多棵樹大致平行訓練，再用多數決彙總。"
        "對照單顆樹與森林的驗證準確率。"
    )

    df = pd.read_csv(HEART_PATH)
    st.success("目前使用本階段內建心臟病資料（與階段3共用切分）。")
    render_dataset_metrics(df)
    _render_heart_data_intro(df)

    n_estimators = st.slider(
        "樹的數量 n_estimators",
        min_value=5,
        max_value=100,
        value=50,
        step=5,
        key="rf_n_estimators",
    )

    unlocked = _render_bagging_pretrain_quiz(n_estimators=int(n_estimators))

    st.markdown("##### 訓練")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_random_forest",
        disabled=not unlocked,
    )
    if not unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")

    result_key = "random_forest_last_result"
    signature = (int(n_estimators), RANDOM_STATE, "heart")
    x_train, x_val, y_train, y_val = prepare_heart_splits(df)

    if train_clicked and unlocked:
        with st.spinner("正在訓練單顆決策樹與隨機森林…"):
            result = compare_single_tree_vs_forest(
                x_train,
                y_train,
                x_val,
                y_val,
                n_estimators=int(n_estimators),
            )
        st.session_state[result_key] = {
            "signature": signature,
            "tree_val_accuracy": result["tree_val_accuracy"],
            "forest_val_accuracy": result["forest_val_accuracy"],
            "tree_train_accuracy": result["tree_train_accuracy"],
            "forest_train_accuracy": result["forest_train_accuracy"],
            "n_estimators": result["n_estimators"],
            "train_rows": len(x_train),
            "val_rows": len(x_val),
        }

    if result_key in st.session_state and st.session_state[result_key]["signature"] == signature:
        cached = st.session_state[result_key]
        if not (train_clicked and unlocked):
            st.caption("顯示最近一次訓練結果；調整 n_estimators 後請重新按「開始訓練」。")
        _render_forest_results(cached)
    elif unlocked:
        st.info("調整 n_estimators 後，按下「開始訓練」以比較單顆樹與森林。")
    else:
        st.info("先完成上方兩題訓練前預測，再開始訓練。")

    with st.expander("Bagging 補充（bootstrap／特徵隨機）", expanded=False):
        st.caption(
            "實務上隨機森林還常對樣本做 bootstrap、對特徵隨機抽樣；"
            "本階段主菜是平行訓練＋多數決，細節可之後再挖。"
        )

    _merge_stage_context(
        note=(
            f"目前階段：隨機森林（Bagging）；n_estimators={int(n_estimators)}。\n"
            + quiz.build_bagging_quiz_agent_appendix(
                bagging_status=_status(quiz.SESSION_BAGGING, quiz.is_bagging_correct),
                vote_status=_status(quiz.SESSION_VOTE, quiz.is_vote_correct),
                focus_qid=st.session_state.get(quiz.SESSION_BAG_FOCUS),
                unlocked=unlocked,
                n_estimators=int(n_estimators),
            )
        )
    )
    for prompt in (
        "為什麼森林的驗證準確率常常比單顆樹穩？",
        "n_estimators 一直加大，驗證準確率一定會一直升嗎？",
    ):
        st.code(prompt, language="text")


def _render_xgboost_stage() -> None:
    st.markdown("##### 這一階段在問什麼")
    st.info(
        "XGBoost 代表 **Boosting**：弱學習器序列訓練，後者針對前者錯誤調整。"
        "主旋鈕是 learning_rate（步長），不是平行種幾棵樹。"
    )
    st.markdown(quiz.bagging_vs_boosting_contrast_markdown())

    df = pd.read_csv(HEART_PATH)
    st.success("目前使用本階段內建心臟病資料（與階段2同一切分）。")
    render_dataset_metrics(df)
    _render_heart_data_intro(df)

    learning_rate = st.slider(
        "學習率 learning_rate",
        min_value=0.01,
        max_value=0.5,
        value=float(STAGE_DEFAULT_LEARNING_RATE),
        step=0.01,
        key="xgb_stage_learning_rate",
        help="Boosting 每一步走多大幅；與階段2的 n_estimators（平行棵數）語意不同。",
    )
    st.caption(
        f"本階段固定 n_estimators={STAGE_FIXED_N_ESTIMATORS}（序列加幾輪弱樹），"
        "只調整 learning_rate。"
    )

    unlocked = _render_boost_pretrain_quiz(learning_rate=float(learning_rate))

    st.markdown("##### 訓練")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_xgboost_stage",
        disabled=not unlocked,
    )
    if not unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")

    result_key = "xgboost_stage_last_result"
    signature = (float(learning_rate), STAGE_FIXED_N_ESTIMATORS, RANDOM_STATE)
    x_train, x_val, y_train, y_val = prepare_heart_splits(df)

    if train_clicked and unlocked:
        with st.spinner("正在訓練 XGBoost，並計算同切分下的森林基線…"):
            model = fit_xgboost_stage(
                x_train,
                y_train,
                learning_rate=float(learning_rate),
            )
            train_acc, val_acc = training_and_validation_accuracy(
                model, x_train, y_train, x_val, y_val
            )
            forest_val = forest_validation_baseline(
                x_train,
                y_train,
                x_val,
                y_val,
                n_estimators=BASELINE_FOREST_N_ESTIMATORS,
            )
            forest_n = BASELINE_FOREST_N_ESTIMATORS
        st.session_state[result_key] = {
            "signature": signature,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "forest_val_accuracy": forest_val,
            "forest_n_estimators": forest_n,
            "learning_rate": float(learning_rate),
            "n_estimators": STAGE_FIXED_N_ESTIMATORS,
            "train_rows": len(x_train),
            "val_rows": len(x_val),
            "feature_count": x_train.shape[1],
        }

    if result_key in st.session_state and st.session_state[result_key]["signature"] == signature:
        cached = st.session_state[result_key]
        if not (train_clicked and unlocked):
            st.caption("顯示最近一次訓練結果；調整 learning_rate 後請重新按「開始訓練」。")
        _render_xgboost_stage_results(cached)
    elif unlocked:
        st.info("調整 learning_rate 後，按下「開始訓練」。")
    else:
        st.info("先完成上方兩題訓練前預測，再開始訓練。")

    _merge_stage_context(
        note=(
            f"目前階段：XGBoost（Boosting）；learning_rate={float(learning_rate):g}。\n"
            + quiz.build_boost_quiz_agent_appendix(
                boost_status=_status(quiz.SESSION_BOOST, quiz.is_boost_correct),
                contrast_status=_status(quiz.SESSION_CONTRAST, quiz.is_contrast_correct),
                focus_qid=st.session_state.get(quiz.SESSION_BOOST_FOCUS),
                unlocked=unlocked,
                learning_rate=float(learning_rate),
            )
        )
    )
    for prompt in (
        "Bagging 與 Boosting 差在平行多數決還是序列糾錯？請用本頁對照表說明。",
        "learning_rate 變小通常會怎樣？和隨機森林的 n_estimators 能直接類比嗎？",
    ):
        st.code(prompt, language="text")


def _status(session_key: str, checker) -> str:
    choice = str(st.session_state.get(session_key, quiz.PLEASE_SELECT))
    return quiz.quiz_choice_status(choice, correct=checker(choice))


def _tree_quiz_appendix(*, unlocked: bool) -> str:
    return quiz.build_tree_quiz_agent_appendix(
        entropy_status=_status(quiz.SESSION_ENTROPY, quiz.is_entropy_correct),
        ig_status=_status(quiz.SESSION_IG, quiz.is_ig_correct),
        focus_qid=st.session_state.get(quiz.SESSION_TREE_FOCUS),
        unlocked=unlocked,
    )


def _merge_stage_context(*, note: str) -> None:
    existing = str(st.session_state.get(CONTEXT_KEY, f"目前頁面：{PAGE_TITLE}。"))
    st.session_state[CONTEXT_KEY] = existing.split("\n【")[0].rstrip() + "\n" + note


def _render_tree_pretrain_quiz() -> bool:
    st.session_state.setdefault(quiz.SESSION_ENTROPY, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_IG, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_TREE_FOCUS, quiz.QID_ENTROPY)

    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        entropy_choice = st.radio(
            "題1：關於熵（不純度），下列哪個正確？",
            [quiz.PLEASE_SELECT, *quiz.ENTROPY_OPTIONS],
            key=quiz.SESSION_ENTROPY,
        )
    with h1_col:
        st.write("")
        if st.button("Agent 提示", key="tree_hint_entropy", disabled=not agent_ready, width="stretch"):
            _send_tree_hint(quiz.QID_ENTROPY)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    entropy_ok = quiz.is_entropy_correct(str(entropy_choice))
    if str(entropy_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_TREE_FOCUS] = quiz.QID_ENTROPY
    elif entropy_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想「全是同一類」時不純度如何，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_TREE_FOCUS] = quiz.QID_ENTROPY

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        ig_choice = st.radio(
            "題2：訓練時，決策樹傾向怎麼選分裂？",
            [quiz.PLEASE_SELECT, *quiz.IG_OPTIONS],
            key=quiz.SESSION_IG,
        )
    with h2_col:
        st.write("")
        if st.button("Agent 提示", key="tree_hint_ig", disabled=not agent_ready, width="stretch"):
            _send_tree_hint(quiz.QID_IG)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    ig_ok = quiz.is_ig_correct(str(ig_choice))
    if str(ig_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if entropy_ok:
            st.session_state[quiz.SESSION_TREE_FOCUS] = quiz.QID_IG
    elif ig_ok:
        st.caption("題2 OK。")
    else:
        st.caption("題2 再想想資訊增益／不純度下降，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_TREE_FOCUS] = quiz.QID_IG

    unlocked = quiz.both_tree_quiz_correct(str(entropy_choice), str(ig_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(entropy_ok) + int(ig_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _render_bagging_pretrain_quiz(*, n_estimators: int) -> bool:
    st.session_state.setdefault(quiz.SESSION_BAGGING, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_VOTE, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_BAG_FOCUS, quiz.QID_BAGGING)

    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        bagging_choice = st.radio(
            "題1：Bagging（隨機森林）多棵樹通常怎麼訓練？",
            [quiz.PLEASE_SELECT, *quiz.BAGGING_OPTIONS],
            key=quiz.SESSION_BAGGING,
        )
    with h1_col:
        st.write("")
        if st.button("Agent 提示", key="rf_hint_bagging", disabled=not agent_ready, width="stretch"):
            _send_bagging_hint(quiz.QID_BAGGING, n_estimators=n_estimators)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    bag_ok = quiz.is_bagging_correct(str(bagging_choice))
    if str(bagging_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_BAG_FOCUS] = quiz.QID_BAGGING
    elif bag_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想「平行還是序列」，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_BAG_FOCUS] = quiz.QID_BAGGING

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        vote_choice = st.radio(
            "題2：分類時，多棵樹的預測通常怎麼彙總？",
            [quiz.PLEASE_SELECT, *quiz.VOTE_OPTIONS],
            key=quiz.SESSION_VOTE,
        )
    with h2_col:
        st.write("")
        if st.button("Agent 提示", key="rf_hint_vote", disabled=not agent_ready, width="stretch"):
            _send_bagging_hint(quiz.QID_VOTE, n_estimators=n_estimators)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    vote_ok = quiz.is_vote_correct(str(vote_choice))
    if str(vote_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if bag_ok:
            st.session_state[quiz.SESSION_BAG_FOCUS] = quiz.QID_VOTE
    elif vote_ok:
        st.caption("題2 OK。")
    else:
        st.caption("題2 再想想多數決，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_BAG_FOCUS] = quiz.QID_VOTE

    unlocked = quiz.both_bagging_quiz_correct(str(bagging_choice), str(vote_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(bag_ok) + int(vote_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _render_boost_pretrain_quiz(*, learning_rate: float) -> bool:
    st.session_state.setdefault(quiz.SESSION_BOOST, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_CONTRAST, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_BOOST_FOCUS, quiz.QID_BOOST)

    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        boost_choice = st.radio(
            "題1：Boosting（XGBoost）弱學習器通常怎麼訓練？",
            [quiz.PLEASE_SELECT, *quiz.BOOST_OPTIONS],
            key=quiz.SESSION_BOOST,
        )
    with h1_col:
        st.write("")
        if st.button("Agent 提示", key="xgb_hint_boost", disabled=not agent_ready, width="stretch"):
            _send_boost_hint(quiz.QID_BOOST, learning_rate=learning_rate)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    boost_ok = quiz.is_boost_correct(str(boost_choice))
    if str(boost_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_BOOST_FOCUS] = quiz.QID_BOOST
    elif boost_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想序列糾錯，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_BOOST_FOCUS] = quiz.QID_BOOST

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        contrast_choice = st.radio(
            "題2：Bagging 與 Boosting 的主要差異是？",
            [quiz.PLEASE_SELECT, *quiz.CONTRAST_OPTIONS],
            key=quiz.SESSION_CONTRAST,
        )
    with h2_col:
        st.write("")
        if st.button(
            "Agent 提示", key="xgb_hint_contrast", disabled=not agent_ready, width="stretch"
        ):
            _send_boost_hint(quiz.QID_CONTRAST, learning_rate=learning_rate)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    contrast_ok = quiz.is_contrast_correct(str(contrast_choice))
    if str(contrast_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if boost_ok:
            st.session_state[quiz.SESSION_BOOST_FOCUS] = quiz.QID_CONTRAST
    elif contrast_ok:
        st.caption("題2 OK。")
    else:
        st.caption("題2 再對照上方表格：平行多數決 vs 序列糾錯。")
        st.session_state[quiz.SESSION_BOOST_FOCUS] = quiz.QID_CONTRAST

    unlocked = quiz.both_boost_quiz_correct(str(boost_choice), str(contrast_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(boost_ok) + int(contrast_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _send_tree_hint(qid: str) -> None:
    ts_key = f"tree_ens_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示發送中，請稍候再按。")
        return
    if not st.session_state.get("data_agent_connected"):
        st.warning("請先在右側啟用資料 Agent，再按「Agent 提示」。")
        return
    st.session_state[quiz.SESSION_TREE_FOCUS] = qid
    st.session_state[ts_key] = now
    unlocked = quiz.both_tree_quiz_correct(
        str(st.session_state.get(quiz.SESSION_ENTROPY, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_IG, quiz.PLEASE_SELECT)),
    )
    invoke_data_agent(
        quiz.tree_hint_user_text(qid),
        extra_context=_tree_quiz_appendix(unlocked=unlocked),
        display_user_text="（Agent 提示）請給熵／資訊增益線索，不要直接講正解。",
    )
    st.rerun()


def _send_bagging_hint(qid: str, *, n_estimators: int) -> None:
    ts_key = f"tree_ens_bag_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示發送中，請稍候再按。")
        return
    if not st.session_state.get("data_agent_connected"):
        st.warning("請先在右側啟用資料 Agent，再按「Agent 提示」。")
        return
    st.session_state[quiz.SESSION_BAG_FOCUS] = qid
    st.session_state[ts_key] = now
    unlocked = quiz.both_bagging_quiz_correct(
        str(st.session_state.get(quiz.SESSION_BAGGING, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_VOTE, quiz.PLEASE_SELECT)),
    )
    invoke_data_agent(
        quiz.bagging_hint_user_text(qid, n_estimators=n_estimators),
        extra_context=quiz.build_bagging_quiz_agent_appendix(
            bagging_status=_status(quiz.SESSION_BAGGING, quiz.is_bagging_correct),
            vote_status=_status(quiz.SESSION_VOTE, quiz.is_vote_correct),
            focus_qid=qid,
            unlocked=unlocked,
            n_estimators=n_estimators,
        ),
        display_user_text="（Agent 提示）請給 Bagging／多數決線索，不要直接講正解。",
    )
    st.rerun()


def _send_boost_hint(qid: str, *, learning_rate: float) -> None:
    ts_key = f"tree_ens_boost_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示發送中，請稍候再按。")
        return
    if not st.session_state.get("data_agent_connected"):
        st.warning("請先在右側啟用資料 Agent，再按「Agent 提示」。")
        return
    st.session_state[quiz.SESSION_BOOST_FOCUS] = qid
    st.session_state[ts_key] = now
    unlocked = quiz.both_boost_quiz_correct(
        str(st.session_state.get(quiz.SESSION_BOOST, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_CONTRAST, quiz.PLEASE_SELECT)),
    )
    invoke_data_agent(
        quiz.boost_hint_user_text(qid, learning_rate=learning_rate),
        extra_context=quiz.build_boost_quiz_agent_appendix(
            boost_status=_status(quiz.SESSION_BOOST, quiz.is_boost_correct),
            contrast_status=_status(quiz.SESSION_CONTRAST, quiz.is_contrast_correct),
            focus_qid=qid,
            unlocked=unlocked,
            learning_rate=learning_rate,
        ),
        display_user_text="（Agent 提示）請給 Boosting／對照線索，不要直接講正解。",
    )
    st.rerun()


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
        width="stretch",
        hide_index=True,
    )
    with st.expander("資料預覽", expanded=True):
        st.dataframe(frame[features + [target]].head(10), width="stretch", hide_index=True)
    render_figures_in_streamlit(build_classification_data_figures(frame, features, target))


def _render_heart_data_intro(frame: pd.DataFrame) -> None:
    st.markdown("##### Data 資訊")
    st.info(
        f"心臟病二分類資料；target 欄為「{HEART_TARGET}」。"
        "類別欄位會 one-hot；訓練／驗證 80／20，"
        f"random_state={RANDOM_STATE}（階段2與階段3同一切分）。"
    )
    with st.expander("資料預覽", expanded=False):
        st.dataframe(frame.head(8), width="stretch", hide_index=True)
    viz_features = [col for col in ("年齡", "膽固醇") if col in frame.columns]
    if len(viz_features) == 2 and HEART_TARGET in frame.columns:
        render_figures_in_streamlit(
            build_classification_data_figures(frame, viz_features, HEART_TARGET)
        )


def _render_tree_formulas() -> None:
    st.markdown("##### 模型公式")
    with st.expander("熵與資訊增益（手算）", expanded=True):
        st.latex(r"H(p_1) = -p_1 \log_2(p_1) - (1-p_1)\log_2(1-p_1)")
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


def _render_forest_results(cached: dict) -> None:
    st.markdown("##### 訓練結果（單顆樹 vs 隨機森林）")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("訓練筆數", f"{cached['train_rows']:,}")
    c2.metric("驗證筆數", f"{cached['val_rows']:,}")
    c3.metric("單顆樹驗證準確率", f"{cached['tree_val_accuracy']:.2f}%")
    c4.metric("森林驗證準確率", f"{cached['forest_val_accuracy']:.2f}%")
    st.caption(
        f"n_estimators={cached['n_estimators']}；"
        f"訓練準確率：樹 {cached['tree_train_accuracy']:.2f}%／"
        f"森林 {cached['forest_train_accuracy']:.2f}%。"
    )


def _render_xgboost_stage_results(cached: dict) -> None:
    st.markdown("##### 訓練結果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("訓練筆數", f"{cached['train_rows']:,}")
    c2.metric("驗證筆數", f"{cached['val_rows']:,}")
    c3.metric("XGBoost 驗證準確率", f"{cached['val_acc']:.2f}%")
    c4.metric(
        "森林驗證基線（同切分）",
        f"{cached['forest_val_accuracy']:.2f}%",
    )
    st.caption(
        f"learning_rate={cached['learning_rate']:g}；"
        f"固定 n_estimators={cached['n_estimators']}；"
        f"訓練準確率={cached['train_acc']:.2f}%；"
        f"森林基線 n_estimators={cached['forest_n_estimators']}；"
        f"one-hot 後 features={cached['feature_count']}。"
    )


def _render_tree_prompts() -> None:
    st.markdown("##### 建議問 Agent")
    prompts = [
        "為什麼資訊增益表排名第一的 feature，可能和 sklearn 樹根節點選的 feature 不同？",
        "同一 max_depth 下，Gini 與 Entropy 畫出的樹會一樣嗎？請對照本頁結果說明。",
        "max_depth=1 和 max_depth=2 的葉節點數有什麼差別？",
    ]
    for prompt in prompts:
        st.code(prompt, language="text")
