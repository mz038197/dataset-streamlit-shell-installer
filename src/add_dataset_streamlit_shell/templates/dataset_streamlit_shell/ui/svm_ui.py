from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.ui.data_ui import (
    READY_DATASET_PATH,
    SHELL_ROOT,
    _display_path,
    invoke_data_agent,
    load_ready_dataset,
    render_chat_panel,
    render_dataset_metrics,
)
from dataset_streamlit_shell.ml.regression import (
    apply_standard_scaler,
    create_standard_scaler,
)
from dataset_streamlit_shell.ml.svm import (
    MODEL_KIND_LINEAR_SVM,
    LinearSvmArtifact,
    artifact_from_payload,
    build_linear_svm_artifact,
    build_svm_agent_context,
    decision_function_from_artifact,
    fit_linear_svc,
    predict_binary_class,
    predict_class_from_artifact,
    save_svm_artifact,
    validate_svm_target,
)
from dataset_streamlit_shell.plotting import (
    build_classification_data_figures,
    build_linear_svm_result_figure,
    build_svm_paired_data_figure,
    configure_matplotlib_for_traditional_chinese,
    render_figures_in_streamlit,
)
from dataset_streamlit_shell.ui import svm_quiz as quiz

configure_matplotlib_for_traditional_chinese()

CLASSIFICATION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "classification"
CLASSIFICATION_MODEL_DIR = SHELL_ROOT / "workspace" / "models" / "classification"
SVM_BLOBS_PATH = CLASSIFICATION_DEMO_DIR / "svm_blobs_80.csv"
SVM_SOFT_PATH = CLASSIFICATION_DEMO_DIR / "svm_soft_margin_80.csv"

SVM_FEATURES = ["特徵1", "特徵2"]
SVM_TARGET = "類別"
HARD_DEFAULT_C = 1.0
PAGE_TITLE = "線性 SVM"
CONTEXT_KEY = f"{PAGE_TITLE}_agent_context"
STAGE_HARD_LABEL = "線性可分（最大化 margin）"
STAGE_SOFT_LABEL = "Soft Margin"


def render_linear_svm_page() -> None:
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title(PAGE_TITLE)
        st.caption("先學可分開時的最大 margin，再學分不開時的 Soft Margin。")
        source = st.radio(
            "資料來源",
            ["內建範例資料", "目前 ready.csv"],
            horizontal=True,
            key=f"{PAGE_TITLE}_data_source",
        )
        builtin = source == "內建範例資料"
        ready_df: pd.DataFrame | None = None
        if builtin:
            st.success("內建資料依階段切換：可分 blobs → 重疊 soft-margin。")
        else:
            ready_df = load_ready_dataset()
            if ready_df is None:
                st.warning("尚未建立 ready.csv，或改用內建範例資料。")
                return
            st.info(f"目前使用 `{_display_path(READY_DATASET_PATH)}`。")
            render_dataset_metrics(ready_df)

        # st.tabs 在 streamlit>=1.50 無法可靠回報作用中分頁；用 radio 追蹤焦點給 Agent。
        stage = st.radio(
            "學習階段",
            [STAGE_HARD_LABEL, STAGE_SOFT_LABEL],
            horizontal=True,
            key="svm_learning_stage",
        )
        if stage == STAGE_HARD_LABEL:
            st.session_state[quiz.SESSION_PAGE_FOCUS] = "hard"
            _render_hard_margin_tab(builtin=builtin, ready_df=ready_df)
        else:
            st.session_state[quiz.SESSION_PAGE_FOCUS] = "soft"
            _render_soft_margin_tab(builtin=builtin, ready_df=ready_df)
        _compose_agent_context()

    with side:
        render_chat_panel(
            extra_context=str(st.session_state.get(CONTEXT_KEY, f"目前頁面：{PAGE_TITLE}。")),
            page_name=PAGE_TITLE,
        )


def _render_hard_margin_tab(*, builtin: bool, ready_df: pd.DataFrame | None) -> None:
    st.markdown("##### 這一階段在問什麼")
    st.info(
        "兩類若分得開，最好的直線是 **margin 最大** 的那條；"
        "卡住邊界的點叫 support vector。"
    )
    prepared = _prepare_tab_data(
        builtin=builtin,
        ready_df=ready_df,
        builtin_path=SVM_BLOBS_PATH,
        builtin_label="內建範例資料：可線性分開的兩特徵二元分類（80 筆）",
        key_prefix="svm_hard",
        note="每一列是一個樣本：特徵為 x，類別為 y（本頁固定使用 -1 / +1）。資料大致可被直線分開。",
    )
    if prepared is None:
        return
    working, feature_matrix, features, target, scaler, source_label = prepared

    st.markdown("##### 模型公式")
    st.latex(r"f_{\mathbf{w},b}(\mathbf{x})=\mathbf{w}\cdot\mathbf{x}+b")
    st.caption("預測類別由 f(x) 的正負決定：f(x) ≥ 0 判為 +1，f(x) < 0 判為 -1。")
    _render_hard_margin_formula()

    unlocked = _render_hard_pretrain_quiz(
        features=features,
        target=target,
        source_label=source_label,
        row_count=len(working),
    )

    st.markdown("##### 訓練")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_linear_svm_hard",
        disabled=not unlocked,
    )
    if not unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")

    result_key = "linear_svm_hard_last_artifact"
    signature = (source_label, tuple(features), target, len(working), builtin, "hard")
    can_plot_2d = len(features) == 2
    if not can_plot_2d:
        st.caption("目前選超過 2 個 features，訓練後無法繪製 2D 決策邊界圖。")

    artifact = _resolve_or_train_artifact(
        train_clicked=train_clicked and unlocked,
        feature_matrix=feature_matrix,
        working=working,
        features=features,
        target=target,
        scaler=scaler,
        source_label=source_label,
        C=HARD_DEFAULT_C,
        result_key=result_key,
        signature=signature,
        can_plot_2d=can_plot_2d,
        builtin=builtin,
        idle_message="答完兩題後，按下「開始訓練」以顯示決策邊界與 support vectors。",
        stale_caption="顯示最近一次訓練結果；換資料或欄位後請重新訓練。",
    )

    focus = st.session_state.get(quiz.SESSION_HARD_FOCUS)
    obj_choice = str(st.session_state.get(quiz.SESSION_OBJ, quiz.PLEASE_SELECT))
    norm_choice = str(st.session_state.get(quiz.SESSION_NORM, quiz.PLEASE_SELECT))
    appendix = quiz.build_hard_quiz_agent_appendix(
        obj_status=quiz.quiz_choice_status(obj_choice, correct=quiz.is_obj_correct(obj_choice)),
        norm_status=quiz.quiz_choice_status(norm_choice, correct=quiz.is_norm_correct(norm_choice)),
        focus_qid=focus,
        features=features,
        target=target,
        unlocked=unlocked,
    )
    _merge_agent_context(
        tab="hard",
        source_label=source_label,
        features=features,
        target=target,
        C=HARD_DEFAULT_C,
        row_count=len(working),
        artifact=artifact,
        include_C=False,
        prompt_train=unlocked,
        note="目前階段：線性可分（最大化 margin）。" + "\n" + appendix,
    )

    if artifact is not None:
        _render_svm_training_results(artifact, working, target, show_C=False, key_prefix="hard")
        _render_svm_save_section(artifact, key="save_svm_hard")
        _render_svm_inference_section(trained_artifact=artifact, key_prefix="hard")

    with st.expander("完整 sklearn 範例程式", expanded=False):
        st.code(quiz.SKLEARN_HARD_EXAMPLE, language="python")
        st.caption("範例使用 `SVC(kernel=\"linear\")`。")

    _render_svm_prompts(quiz.hard_focus_prompt_lines(focus, unlocked=unlocked))


def _render_soft_margin_tab(*, builtin: bool, ready_df: pd.DataFrame | None) -> None:
    st.markdown("##### 這一階段在問什麼")
    st.info(
        "上一階段假設資料分得開。這裡類別常有重疊，直線無法完美分開時，"
        "需要 **Soft Margin**：允許一些點落入 margin／分錯；**C** 決定顧 margin 還是顧分對。"
    )
    prepared = _prepare_tab_data(
        builtin=builtin,
        ready_df=ready_df,
        builtin_path=SVM_SOFT_PATH,
        builtin_label="內建範例資料：類別重疊的兩特徵二元分類（soft margin，80 筆）",
        key_prefix="svm_soft",
        note="資料有重疊，不容易用一條直線完美分開——適合觀察 Soft Margin 與 C。",
    )
    if prepared is None:
        return
    working, feature_matrix, features, target, scaler, source_label = prepared

    st.markdown("##### 模型公式")
    st.latex(r"f_{\mathbf{w},b}(\mathbf{x})=\mathbf{w}\cdot\mathbf{x}+b")
    _render_soft_margin_formula()

    st.markdown("##### 訓練設定")
    C = st.number_input(
        "懲罰係數 C",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        key="svm_soft_C",
        help="C 愈大愈在意分對訓練點；C 愈小愈傾向較寬的 margin。",
    )

    unlocked = _render_soft_pretrain_quiz(
        features=features,
        target=target,
        source_label=source_label,
        C=float(C),
        row_count=len(working),
    )

    st.markdown("##### 訓練")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        width="stretch",
        key="train_linear_svm_soft",
        disabled=not unlocked,
    )
    if not unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")
    else:
        st.caption("解鎖後可改 C 再按「開始訓練」，對照邊界與 support vectors 的變化。")

    result_key = "linear_svm_soft_last_artifact"
    signature = (source_label, tuple(features), target, float(C), len(working), builtin, "soft")
    can_plot_2d = len(features) == 2
    if not can_plot_2d:
        st.caption("目前選超過 2 個 features，訓練後無法繪製 2D 決策邊界圖。")

    artifact = _resolve_or_train_artifact(
        train_clicked=train_clicked and unlocked,
        feature_matrix=feature_matrix,
        working=working,
        features=features,
        target=target,
        scaler=scaler,
        source_label=source_label,
        C=float(C),
        result_key=result_key,
        signature=signature,
        can_plot_2d=can_plot_2d,
        builtin=builtin,
        idle_message="設定 C 並答完兩題後，按下「開始訓練」。",
        stale_caption="顯示最近一次訓練結果；調整 C 後請重新按「開始訓練」。",
    )

    focus = st.session_state.get(quiz.SESSION_SOFT_FOCUS)
    hard_choice = str(st.session_state.get(quiz.SESSION_HARD, quiz.PLEASE_SELECT))
    c_choice = str(st.session_state.get(quiz.SESSION_C, quiz.PLEASE_SELECT))
    appendix = quiz.build_soft_quiz_agent_appendix(
        hard_status=quiz.quiz_choice_status(
            hard_choice, correct=quiz.is_hard_limit_correct(hard_choice)
        ),
        c_status=quiz.quiz_choice_status(c_choice, correct=quiz.is_c_correct(c_choice)),
        focus_qid=focus,
        features=features,
        target=target,
        C=float(C),
        unlocked=unlocked,
    )
    _merge_agent_context(
        tab="soft",
        source_label=source_label,
        features=features,
        target=target,
        C=float(C),
        row_count=len(working),
        artifact=artifact,
        include_C=True,
        prompt_train=unlocked,
        note="目前階段：Soft Margin（引入 C）。" + "\n" + appendix,
    )

    if artifact is not None:
        _render_svm_training_results(artifact, working, target, show_C=True, key_prefix="soft")
        _render_svm_save_section(artifact, key="save_svm_soft")
        _render_svm_inference_section(trained_artifact=artifact, key_prefix="soft")

    with st.expander("完整 sklearn 範例程式", expanded=False):
        st.code(quiz.SKLEARN_SOFT_EXAMPLE, language="python")
        st.caption("與階段 1 同骨架，差在資料與明確寫出 C。")

    _render_svm_prompts(quiz.soft_focus_prompt_lines(focus, unlocked=unlocked))


def _prepare_tab_data(
    *,
    builtin: bool,
    ready_df: pd.DataFrame | None,
    builtin_path: Path,
    builtin_label: str,
    key_prefix: str,
    note: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, dict | None, str] | None:
    if builtin:
        df = pd.read_csv(builtin_path)
        render_dataset_metrics(df)
        features = list(SVM_FEATURES)
        target = SVM_TARGET
        working = _svm_training_frame(df, features, target, builtin=True)
        source_label = builtin_label
    else:
        assert ready_df is not None
        df = ready_df
        numeric_columns = _numeric_columns(df)
        if len(numeric_columns) < 3:
            st.warning("至少需要 2 個 features 與 1 個 target。")
            return None
        default_target = _default_column(numeric_columns, SVM_TARGET)
        target = st.selectbox(
            "選擇 target（y，-1/+1）",
            numeric_columns,
            index=numeric_columns.index(default_target) if default_target in numeric_columns else 0,
            key=f"{key_prefix}_target",
        )
        if not validate_svm_target(df[target]):
            st.warning("target 必須剛好包含 -1 與 +1。請先在前處理頁完成轉碼。")
            return None
        feature_options = [column for column in numeric_columns if column != target]
        default_features = [column for column in SVM_FEATURES if column in feature_options]
        if not default_features:
            default_features = feature_options[: min(2, len(feature_options))]
        features = st.multiselect(
            "選擇 features（x）",
            feature_options,
            default=default_features,
            key=f"{key_prefix}_features",
        )
        if len(features) < 1:
            st.warning("請至少選擇 1 個 feature。")
            return None
        working = _svm_training_frame(df, features, target, builtin=False)
        source_label = f"目前 ready.csv：{_display_path(READY_DATASET_PATH)}"

    if len(working) < 2:
        st.warning("可用樣本少於 2 筆，無法訓練。")
        return None

    _render_svm_data_intro(working, features=features, target=target, builtin=builtin, note=note)
    feature_matrix, scaler = _prepare_svm_features(working, features, builtin=builtin)
    return working, feature_matrix, features, target, scaler, source_label


def _resolve_or_train_artifact(
    *,
    train_clicked: bool,
    feature_matrix: pd.DataFrame,
    working: pd.DataFrame,
    features: list[str],
    target: str,
    scaler: dict | None,
    source_label: str,
    C: float,
    result_key: str,
    signature: tuple,
    can_plot_2d: bool,
    builtin: bool,
    idle_message: str,
    stale_caption: str,
) -> LinearSvmArtifact | None:
    artifact: LinearSvmArtifact | None = None
    if train_clicked:
        try:
            clf = fit_linear_svc(feature_matrix, working[target], C=float(C))
        except ValueError as exc:
            st.error(str(exc))
            return None
        artifact = build_linear_svm_artifact(
            clf,
            features=list(features),
            target=target,
            C=float(C),
            scaler=scaler,
            data_source=source_label,
            feature_frame=feature_matrix,
            target_series=working[target],
        )
        st.session_state[result_key] = {"signature": signature, "artifact": artifact}
        if can_plot_2d:
            st.markdown("##### 訓練結果圖")
            fig = build_linear_svm_result_figure(
                working,
                features,
                target,
                coef=artifact.coef,
                intercept=artifact.intercept,
                support_vectors=artifact.support_vectors,
                paired_scatter=builtin,
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        return artifact

    stored = st.session_state.get(result_key)
    if isinstance(stored, dict) and stored.get("signature") == signature:
        artifact = stored["artifact"]
        st.caption(stale_caption)
        if can_plot_2d:
            st.markdown("##### 訓練結果圖")
            fig = build_linear_svm_result_figure(
                working,
                features,
                target,
                coef=artifact.coef,
                intercept=artifact.intercept,
                support_vectors=artifact.support_vectors,
                paired_scatter=builtin,
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        return artifact

    st.info(idle_message)
    return None


def _merge_agent_context(
    *,
    tab: str,
    source_label: str,
    features: list[str],
    target: str,
    C: float,
    row_count: int,
    artifact: LinearSvmArtifact | None,
    include_C: bool,
    prompt_train: bool,
    note: str,
) -> None:
    """各階段寫入片段；頁面結尾再 _compose_agent_context 合併（避免互蓋）。"""
    st.session_state[f"_svm_ctx_frag_{tab}"] = build_svm_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        C=C,
        row_count=row_count,
        artifact=artifact,
        note=note,
        include_C=include_C,
        prompt_train=prompt_train,
    )


def _compose_agent_context() -> None:
    hard = str(st.session_state.get("_svm_ctx_frag_hard", ""))
    soft = str(st.session_state.get("_svm_ctx_frag_soft", ""))
    hard_unlocked = quiz.both_hard_quiz_correct(
        str(st.session_state.get(quiz.SESSION_OBJ, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_NORM, quiz.PLEASE_SELECT)),
    )
    soft_unlocked = quiz.both_soft_quiz_correct(
        str(st.session_state.get(quiz.SESSION_HARD, quiz.PLEASE_SELECT)),
        str(st.session_state.get(quiz.SESSION_C, quiz.PLEASE_SELECT)),
    )
    focus = st.session_state.get(quiz.SESSION_PAGE_FOCUS, "hard")
    parts = [
        f"目前頁面：{PAGE_TITLE}（雙階段：硬間隔 → Soft Margin）。",
        f"目前焦點階段：{focus}。",
        f"硬間隔階段訓練解鎖：{'是' if hard_unlocked else '否'}；"
        f"Soft Margin 階段訓練解鎖：{'是' if soft_unlocked else '否'}。",
        "回答時依學生目前焦點階段說明；硬間隔階段勿主動引入 C，除非學生問到 Soft Margin。",
        "未解鎖前請勿直接告訴學生訓練前預測的正解選項。",
    ]
    if hard:
        parts.extend(["", "—— 階段 1 硬間隔狀態 ——", hard])
    if soft:
        parts.extend(["", "—— 階段 2 Soft Margin 狀態 ——", soft])
    st.session_state[CONTEXT_KEY] = "\n".join(parts)


def _render_hard_margin_formula() -> None:
    st.latex(
        r"""
        \begin{aligned}
        \min_{w,b}\quad & \tfrac12\|w\|^2 \\
        \text{s.t.}\quad & y_i(w^\top x_i+b)\ge 1,\quad \forall i
        \end{aligned}
        """
    )
    st.caption(
        "在「全部分對且點在 margin 外」的分界線中，找 ‖w‖ 最小（⇔ margin 最大）的那一條。"
        r" 其中 $\mathrm{Margin}=2/\|w\|$。"
    )
    with st.expander("為什麼目標會變成 ½‖w‖²？（逐步推導）", expanded=False):
        st.markdown("**1. 平行線距離**")
        st.markdown("兩條平行線 \(w^\top x+b=c_1\)、\(w^\top x+b=c_2\) 的距離為：")
        st.latex(r"\frac{|c_1-c_2|}{\|w\|}")
        st.markdown("SVM 邊界取 \(+1\) 與 \(-1\)：")
        st.latex(r"\mathrm{Margin}=\frac{|1-(-1)|}{\|w\|}=\frac{2}{\|w\|}")
        st.caption(r"決策邊界到其中一邊的距離是 $1/\|w\|$。")

        st.markdown("**2. 原始目標：讓 Margin 最大**")
        st.latex(r"\max_{w,b}\frac{2}{\|w\|}")
        st.markdown("分子是常數，因此等價於：")
        st.latex(r"\min\|w\|")

        st.markdown("**3. 為什麼變成平方？**")
        st.markdown(
            r"因為 $\|w\|\ge 0$，$\min\|w\|$ 與 $\min\|w\|^2$ 會在同一個 $w$ 取到最小值；"
            "平方可消去根號，微分較方便。"
        )
        st.dataframe(
            pd.DataFrame({"‖w‖": [1, 2, 3], "‖w‖²": [1, 4, 9]}),
            hide_index=True,
            width="stretch",
        )

        st.markdown(r"**4. 為什麼再乘上 $\frac12$？**")
        st.markdown(
            r"乘上正常數不改變最小值的位置。加入 $\frac12$ 只是讓微分更乾淨："
            r"$\partial_w(\frac12\|w\|^2)=w$，若沒有 $\frac12$ 則會得到 $2w$。"
        )

        st.warning(
            r"**關鍵：$\frac12\|w\|^2$ 不能單獨當成完整 Loss。**"
            r"若只有 $\min\frac12\|w\|^2$，最優是 $w=0$，形不成有效邊界。"
            "它是 **objective（目標函數）**，必須搭配分類限制一起看。"
        )

        st.markdown("**5. 分類限制**")
        st.markdown(r"標籤 $y_i\in\{+1,-1\}$。正類要求 $w^\top x_i+b\ge 1$，負類要求 $w^\top x_i+b\le -1$，合併為：")
        st.latex(r"y_i(w^\top x_i+b)\ge 1")
        st.caption(r"當 $y_i=+1$ 還原成 $w^\top x_i+b\ge 1$；當 $y_i=-1$ 還原成 $w^\top x_i+b\le -1$。")

        st.markdown("**6. Hard-margin 完整定義**")
        st.latex(
            r"""
            \begin{aligned}
            \min_{w,b}\quad & \tfrac12\|w\|^2 \\
            \text{s.t.}\quad & y_i(w^\top x_i+b)\ge 1,\quad \forall i
            \end{aligned}
            """
        )
        st.markdown(
            r"**演變一覽：** $\max\mathrm{Margin}$ → $\max 2/\|w\|$ → $\min\|w\|$ → "
            r"$\min\frac12\|w\|^2$，並搭配 $y_i(w^\top x_i+b)\ge 1$。"
        )


def _render_soft_margin_formula() -> None:
    st.latex(
        r"\mathrm{Loss}=\frac{1}{2}\|\mathbf{w}\|^2"
        r"+ C\sum_i \max\bigl(0,\,1-y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)}+b)\bigr)"
    )
    st.markdown(
        "- 前半項仍想拉大 margin（縮小 ‖w‖）。\n"
        "- 後半項是 **hinge**：點若掉進 margin 或分錯，就付出代價。\n"
        "- **C 愈大**：愈不容忍違規，較在意把訓練點分對，margin 常變窄。\n"
        "- **C 愈小**：較能容忍違規，傾向較寬的 margin。"
    )
    st.caption("這就是 Soft Margin：在「顧 margin」與「顧分對」之間用 C 做取捨。")


def _render_hard_pretrain_quiz(
    *, features: list[str], target: str, source_label: str, row_count: int
) -> bool:
    if quiz.needs_quiz_reset(
        st.session_state.get(quiz.SESSION_HARD_PAIR),
        features,
        target,
        source_label=source_label,
        tab="hard",
    ):
        st.session_state[quiz.SESSION_OBJ] = quiz.PLEASE_SELECT
        st.session_state[quiz.SESSION_NORM] = quiz.PLEASE_SELECT
        st.session_state[quiz.SESSION_HARD_FOCUS] = quiz.QID_OBJ
    st.session_state[quiz.SESSION_HARD_PAIR] = quiz.pair_key(
        features, target, source_label=source_label, tab="hard"
    )
    st.session_state.setdefault(quiz.SESSION_OBJ, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_NORM, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_HARD_FOCUS, quiz.QID_OBJ)

    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        obj_choice = st.radio(
            r"題1：若只有 $\min\frac12\|w\|^2$、沒有分類限制，會怎樣？",
            [quiz.PLEASE_SELECT, *quiz.OBJ_OPTIONS],
            key=quiz.SESSION_OBJ,
        )
    with h1_col:
        st.write("")
        if st.button(
            "Agent 提示",
            key="svm_hard_hint_obj",
            disabled=not agent_ready,
            width="stretch",
        ):
            _send_hard_hint(
                quiz.QID_OBJ,
                features=features,
                target=target,
                source_label=source_label,
                row_count=row_count,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    obj_ok = quiz.is_obj_correct(str(obj_choice))
    if str(obj_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_HARD_FOCUS] = quiz.QID_OBJ
    elif obj_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想：沒有限制時目標函數會往哪裡跑？可按「Agent 提示」。")
        st.session_state[quiz.SESSION_HARD_FOCUS] = quiz.QID_OBJ

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        norm_choice = st.radio(
            r"題2：Margin 變大時，$\|w\|$ 通常怎麼變？",
            [quiz.PLEASE_SELECT, *quiz.NORM_OPTIONS],
            key=quiz.SESSION_NORM,
        )
    with h2_col:
        st.write("")
        if st.button(
            "Agent 提示",
            key="svm_hard_hint_norm",
            disabled=not agent_ready,
            width="stretch",
        ):
            _send_hard_hint(
                quiz.QID_NORM,
                features=features,
                target=target,
                source_label=source_label,
                row_count=row_count,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    norm_ok = quiz.is_norm_correct(str(norm_choice))
    if str(norm_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if obj_ok:
            st.session_state[quiz.SESSION_HARD_FOCUS] = quiz.QID_NORM
    elif norm_ok:
        st.caption("題2 OK。")
    else:
        st.caption(r"題2 再想想 $\mathrm{Margin}=2/\|w\|$，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_HARD_FOCUS] = quiz.QID_NORM

    unlocked = quiz.both_hard_quiz_correct(str(obj_choice), str(norm_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(obj_ok) + int(norm_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _render_soft_pretrain_quiz(
    *,
    features: list[str],
    target: str,
    source_label: str,
    C: float,
    row_count: int,
) -> bool:
    if quiz.needs_quiz_reset(
        st.session_state.get(quiz.SESSION_SOFT_PAIR),
        features,
        target,
        source_label=source_label,
        tab="soft",
    ):
        st.session_state[quiz.SESSION_HARD] = quiz.PLEASE_SELECT
        st.session_state[quiz.SESSION_C] = quiz.PLEASE_SELECT
        st.session_state[quiz.SESSION_SOFT_FOCUS] = quiz.QID_HARD
    st.session_state[quiz.SESSION_SOFT_PAIR] = quiz.pair_key(
        features, target, source_label=source_label, tab="soft"
    )
    st.session_state.setdefault(quiz.SESSION_HARD, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_C, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_SOFT_FOCUS, quiz.QID_HARD)

    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        hard_choice = st.radio(
            "題1：資料無法用一條直線完美分開時，硬間隔假設會怎樣？",
            [quiz.PLEASE_SELECT, *quiz.HARD_OPTIONS],
            key=quiz.SESSION_HARD,
        )
    with h1_col:
        st.write("")
        if st.button(
            "Agent 提示",
            key="svm_soft_hint_hard",
            disabled=not agent_ready,
            width="stretch",
        ):
            _send_soft_hint(
                quiz.QID_HARD,
                features=features,
                target=target,
                source_label=source_label,
                C=C,
                row_count=row_count,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    hard_ok = quiz.is_hard_limit_correct(str(hard_choice))
    if str(hard_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_SOFT_FOCUS] = quiz.QID_HARD
    elif hard_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想「所有限制都要滿足」在不可分時會怎樣，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_SOFT_FOCUS] = quiz.QID_HARD

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        c_choice = st.radio(
            "題2：C 明顯變大時，模型比較偏向？",
            [quiz.PLEASE_SELECT, *quiz.C_OPTIONS],
            key=quiz.SESSION_C,
        )
    with h2_col:
        st.write("")
        if st.button(
            "Agent 提示",
            key="svm_soft_hint_c",
            disabled=not agent_ready,
            width="stretch",
        ):
            _send_soft_hint(
                quiz.QID_C,
                features=features,
                target=target,
                source_label=source_label,
                C=C,
                row_count=row_count,
            )
        elif not agent_ready:
            st.caption("先啟用 Agent")

    c_ok = quiz.is_c_correct(str(c_choice))
    if str(c_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if hard_ok:
            st.session_state[quiz.SESSION_SOFT_FOCUS] = quiz.QID_C
    elif c_ok:
        st.caption("題2 OK。")
    else:
        st.caption("題2 再想想 C 在「分對」與「margin」之間的取捨，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_SOFT_FOCUS] = quiz.QID_C

    unlocked = quiz.both_soft_quiz_correct(str(hard_choice), str(c_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(hard_ok) + int(c_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _send_hard_hint(
    qid: str,
    *,
    features: list[str],
    target: str,
    source_label: str,
    row_count: int,
) -> None:
    ts_key = f"svm_hard_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示發送中，請稍候再按。")
        return
    if not st.session_state.get("data_agent_connected"):
        st.warning("請先在右側啟用資料 Agent，再按「Agent 提示」。")
        return
    st.session_state[quiz.SESSION_HARD_FOCUS] = qid
    st.session_state[quiz.SESSION_PAGE_FOCUS] = "hard"
    st.session_state[ts_key] = now
    obj_choice = str(st.session_state.get(quiz.SESSION_OBJ, quiz.PLEASE_SELECT))
    norm_choice = str(st.session_state.get(quiz.SESSION_NORM, quiz.PLEASE_SELECT))
    unlocked = quiz.both_hard_quiz_correct(obj_choice, norm_choice)
    extra = build_svm_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        C=HARD_DEFAULT_C,
        row_count=row_count,
        artifact=None,
        include_C=False,
        prompt_train=unlocked,
        note=quiz.build_hard_quiz_agent_appendix(
            obj_status=quiz.quiz_choice_status(obj_choice, correct=quiz.is_obj_correct(obj_choice)),
            norm_status=quiz.quiz_choice_status(norm_choice, correct=quiz.is_norm_correct(norm_choice)),
            focus_qid=qid,
            features=features,
            target=target,
            unlocked=unlocked,
        ),
    )
    with st.spinner("正在詢問 Agent…"):
        invoke_data_agent(
            quiz.hard_hint_user_text(qid, features=features, target=target),
            extra_context=extra,
            display_user_text=quiz.hard_hint_display_text(qid),
        )
    st.rerun()


def _send_soft_hint(
    qid: str,
    *,
    features: list[str],
    target: str,
    source_label: str,
    C: float,
    row_count: int,
) -> None:
    ts_key = f"svm_soft_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示發送中，請稍候再按。")
        return
    if not st.session_state.get("data_agent_connected"):
        st.warning("請先在右側啟用資料 Agent，再按「Agent 提示」。")
        return
    st.session_state[quiz.SESSION_SOFT_FOCUS] = qid
    st.session_state[quiz.SESSION_PAGE_FOCUS] = "soft"
    st.session_state[ts_key] = now
    hard_choice = str(st.session_state.get(quiz.SESSION_HARD, quiz.PLEASE_SELECT))
    c_choice = str(st.session_state.get(quiz.SESSION_C, quiz.PLEASE_SELECT))
    unlocked = quiz.both_soft_quiz_correct(hard_choice, c_choice)
    extra = build_svm_agent_context(
        page_name=PAGE_TITLE,
        data_source=source_label,
        features=features,
        target=target,
        C=C,
        row_count=row_count,
        artifact=None,
        include_C=True,
        prompt_train=unlocked,
        note=quiz.build_soft_quiz_agent_appendix(
            hard_status=quiz.quiz_choice_status(
                hard_choice, correct=quiz.is_hard_limit_correct(hard_choice)
            ),
            c_status=quiz.quiz_choice_status(c_choice, correct=quiz.is_c_correct(c_choice)),
            focus_qid=qid,
            features=features,
            target=target,
            C=C,
            unlocked=unlocked,
        ),
    )
    with st.spinner("正在詢問 Agent…"):
        invoke_data_agent(
            quiz.soft_hint_user_text(qid, features=features, target=target, C=C),
            extra_context=extra,
            display_user_text=quiz.soft_hint_display_text(qid),
        )
    st.rerun()


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.select_dtypes(include="number").columns]


def _default_column(columns: list[str], preferred: str) -> str:
    return preferred if preferred in columns else columns[0]


def _svm_training_frame(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    builtin: bool,
) -> pd.DataFrame:
    columns = features + [target]
    frame = df[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().reset_index(drop=True)
    if builtin:
        frame[target] = np.where(frame[target].to_numpy(dtype=int) == 1, 1, -1)
    return frame


def _prepare_svm_features(
    working: pd.DataFrame,
    features: list[str],
    *,
    builtin: bool,
) -> tuple[pd.DataFrame, dict | None]:
    if builtin:
        return working[features], None
    scaler = create_standard_scaler(working, features)
    return apply_standard_scaler(working, scaler), scaler


def _render_svm_data_intro(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    builtin: bool,
    note: str | None = None,
) -> None:
    st.markdown("##### Data 資訊")
    st.info(note or "每一列是一個樣本：特徵為 x，類別為 y（本頁固定使用 -1 / +1）。")
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
        pd.DataFrame(role_rows).style.format({"最小值": "{:.4f}", "最大值": "{:.4f}", "平均值": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )
    with st.expander("資料預覽", expanded=False):
        st.dataframe(frame[features + [target]].head(10), width="stretch", hide_index=True)
    st.markdown("##### 資料視覺化")
    if builtin and len(features) == 2:
        st.caption("特徵空間分佈（Paired 色圖）")
        fig = build_svm_paired_data_figure(frame, features, target)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    else:
        render_figures_in_streamlit(
            build_classification_data_figures(_classification_view(frame, target), features, target)
        )


def _classification_view(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    view = frame.copy()
    view[target] = np.where(view[target].to_numpy(dtype=int) == 1, 1, 0)
    return view


def _render_svm_training_results(
    artifact: LinearSvmArtifact,
    working: pd.DataFrame,
    target: str,
    *,
    show_C: bool,
    key_prefix: str,
) -> None:
    del key_prefix  # reserved for future per-tab widgets
    st.markdown("##### 訓練結果")
    if show_C:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("C", f"{artifact.C:g}")
        c2.metric("intercept", f"{artifact.intercept:.4f}")
        c3.metric("Support vectors", str(artifact.n_support))
        c4.metric("訓練集正確率", f"{artifact.training_accuracy:.2f}%")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("intercept", f"{artifact.intercept:.4f}")
        c2.metric("Support vectors", str(artifact.n_support))
        c3.metric("訓練集正確率", f"{artifact.training_accuracy:.2f}%")
    scores = decision_function_from_artifact(artifact, working)
    predicted = predict_class_from_artifact(artifact, working)
    preview = pd.DataFrame(
        {"actual": working[target], "decision_function": scores, "predicted_class": predicted}
    )
    st.dataframe(preview.head(30).style.format({"decision_function": "{:.4f}"}), width="stretch")


def _render_svm_save_section(artifact: LinearSvmArtifact, *, key: str) -> None:
    st.markdown("##### 保存模型 JSON")
    st.caption("檔案保存至 `dataset_streamlit_shell/workspace/models/classification/`。")
    if st.button("保存模型 JSON", type="primary", width="stretch", key=key):
        CLASSIFICATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CLASSIFICATION_MODEL_DIR / f"linear_svm_{stamp}.json"
        save_svm_artifact(artifact, path)
        st.success(f"已保存模型：`{_display_path(path)}`")


def _render_svm_inference_section(*, trained_artifact: LinearSvmArtifact | None, key_prefix: str) -> None:
    st.markdown("##### 手動預測")
    st.caption("上傳的 JSON 必須為 `linear_svm`。輸出類別固定為 -1 / +1。")
    active = _resolve_svm_artifact(trained_artifact=trained_artifact, key_prefix=key_prefix)
    if active is None:
        return
    input_values: dict[str, float] = {}
    cols = st.columns(min(len(active.features), 3) or 1)
    for index, feature in enumerate(active.features):
        with cols[index % len(cols)]:
            default_value = 0.0
            if active.scaler is not None:
                default_value = float(active.scaler["mean"].get(feature, 0.0))
            input_values[feature] = st.number_input(
                feature, value=default_value, key=f"svm_{key_prefix}_{feature}"
            )
    if st.button("計算預測", type="primary", key=f"svm_predict_{key_prefix}"):
        frame = pd.DataFrame([input_values])
        score = float(decision_function_from_artifact(active, frame)[0])
        pred_class = int(predict_binary_class(np.array([score]))[0])
        st.metric("decision function", f"{score:.4f}")
        st.metric("預測類別", str(pred_class))


def _resolve_svm_artifact(
    *,
    trained_artifact: LinearSvmArtifact | None,
    key_prefix: str,
) -> LinearSvmArtifact | None:
    options: list[str] = []
    if trained_artifact is not None:
        options.append("本次訓練結果")
    options.append("上傳模型 JSON")
    source = (
        options[0]
        if len(options) == 1
        else st.radio(
            "預測使用的模型",
            options,
            horizontal=True,
            key=f"svm_inference_source_{key_prefix}",
        )
    )
    if source == "本次訓練結果" and trained_artifact is not None:
        if trained_artifact.model_kind != MODEL_KIND_LINEAR_SVM:
            st.error("本次訓練模型類型與此頁不符。")
            return None
        return trained_artifact
    uploaded = st.file_uploader("上傳模型 JSON", type=["json"], key=f"svm_upload_{key_prefix}")
    if uploaded is None:
        st.info("請上傳先前保存的模型 JSON。")
        return None
    try:
        artifact = artifact_from_payload(
            json.loads(uploaded.getvalue().decode("utf-8")),
            expected_kind=MODEL_KIND_LINEAR_SVM,
        )
        st.success(f"已載入模型：{artifact.model_kind}")
        return artifact
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        st.error(f"無法讀取模型 JSON：{exc}")
        return None


def _render_svm_prompts(prompts: list[str]) -> None:
    st.markdown("##### 建議問 Agent")
    for prompt in prompts:
        st.code(prompt, language="text")
