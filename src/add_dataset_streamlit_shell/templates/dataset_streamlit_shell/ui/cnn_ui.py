"""卷積神經網路頁：六學習階段＋觀念主軸影片＋訓練前預測。"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import streamlit as st
import torch

from dataset_streamlit_shell.ml.cnn_intro import (
    BASE_RGB_IMAGE,
    DEMO_IMAGE_6X6,
    SLIDE_KERNEL,
    build_conv_step_figure,
    build_digits_preview_figure,
    build_feature_maps_demo_figure,
    build_first_conv_maps_figure,
    build_patch_similarity_figure,
    build_pooling_demo_figure,
    build_relu_curve_figure,
    build_relu_image_figure,
    build_rgb_image_figure,
    conv2d_valid,
)
from dataset_streamlit_shell.ml.cnn_pytorch import (
    SimpleCNN,
    extract_first_conv_maps,
    load_digits_tensors,
    train_simple_cnn,
)
from dataset_streamlit_shell.plotting import configure_matplotlib_for_traditional_chinese
from dataset_streamlit_shell.ui import cnn_quiz as quiz
from dataset_streamlit_shell.ui.data_ui import invoke_data_agent, render_chat_panel
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane

configure_matplotlib_for_traditional_chinese()

CONTEXT_KEY = "卷積神經網路_agent_context"
RESULT_KEY = "cnn_last_result"
PAGE_TITLE = "卷積神經網路（CNN）"


def render_cnn_introduction_page() -> None:
    teaching, agent = open_content_dual_pane()
    with teaching:
        st.title(PAGE_TITLE)
        st.caption(
            "先搞懂圖片是矩陣，再跟著觀念主軸影片理解為什麼需要 CNN、卷積與 pooling；"
            "最後兩題訓練前預測通過後再動手訓練。"
        )
        stage = st.radio(
            "學習階段",
            list(quiz.LEARNING_STAGES),
            horizontal=True,
            key="cnn_learning_stage",
        )
        st.session_state[quiz.SESSION_PAGE_FOCUS] = stage
        _render_stage_video(stage)

        if stage == quiz.STAGE_MATRIX:
            _render_matrix_stage()
        elif stage == quiz.STAGE_WHY:
            _render_why_stage()
        elif stage == quiz.STAGE_CONV:
            _render_convolution_stage()
        elif stage == quiz.STAGE_RELU_POOL:
            _render_relu_pool_stage()
        elif stage == quiz.STAGE_FLOW:
            _render_flow_stage()
        else:
            _render_hands_on_stage()

        _compose_agent_context(stage)

    with agent:
        render_chat_panel(
            extra_context=str(st.session_state.get(CONTEXT_KEY, f"目前頁面：{PAGE_TITLE}。")),
            page_name=PAGE_TITLE,
        )


def _show_figure(fig) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_stage_video(stage: str) -> None:
    start = quiz.stage_video_start_sec(stage)
    if start is None:
        return
    embed = quiz.youtube_embed_url(quiz.VIDEO_ID, start_sec=start)
    watch = quiz.youtube_watch_url(quiz.VIDEO_ID, start_sec=start)
    st.markdown("##### 觀念主軸影片（建議從此段開始）")
    st.caption(
        f"{quiz.VIDEO_TITLE} · 建議起點約 {start // 60}:{start % 60:02d}"
        "（對照用，非必看完）"
    )
    # st.video 對 YouTube 的 t= 常無效；改用 embed?start= 才能從建議秒數起播
    st.components.v1.html(
        f'<iframe width="100%" height="360" src="{embed}" '
        'title="CNN explained" frameborder="0" allow="accelerometer; autoplay; '
        'clipboard-write; encrypted-media; gyroscope; picture-in-picture; '
        'web-share" allowfullscreen></iframe>',
        height=380,
    )
    st.markdown(f"[在 YouTube 開啟（含時間戳）]({watch})")


def _render_matrix_stage() -> None:
    st.markdown("## 電腦眼中的「圖片」是什麼？")
    st.markdown(
        """
在電腦裡：

👉 **圖片其實是一個數字表格（矩陣）**

- 每一個格子 = 一個像素（pixel）
- 數字大小 = 亮度（灰階）或顏色強度

先用一張「超小圖片」來看。後面跟影片時，會用到這個直覺。
"""
    )
    _show_figure(build_rgb_image_figure())
    st.markdown("##### RGB 三通道矩陣")
    st.text("紅色 (R) 的矩陣:")
    st.code(str(BASE_RGB_IMAGE[:, :, 0]), language="text")
    st.text("綠色 (G) 的矩陣:")
    st.code(str(BASE_RGB_IMAGE[:, :, 1]), language="text")
    st.text("藍色 (B) 的矩陣:")
    st.code(str(BASE_RGB_IMAGE[:, :, 2]), language="text")


def _render_why_stage() -> None:
    st.markdown("## 為什麼影像不能硬塞進一般神經網路？")
    st.markdown(
        """
如果我們用 **傳統全連接神經網路（Dense）** 來看圖片：

- 一張 64×64 圖片 = **4096 個輸入**
- 參數很多 👉 很容易爆炸
- 先把圖「攤平」成一長條數字時，**原本相鄰的像素會被拆散**，局部形狀變難學

👉 所以我們需要一個：
**「會看局部、同一組規則能在整張圖重複用」的模型** —— 這就是 CNN 的起手式。

> 白話：同一種「找邊緣／找形狀」的小規則，應該在圖的左上角、右下角都能用，不必為每個位置重學一套。
"""
    )
    with st.expander("想多知道一點歷史？（史話補充）"):
        st.markdown(
            """
科學家在研究視覺皮層時發現：眼睛不是一次看完整畫面，而是先看小區域；同一種找線條的方法會在不同位置重複使用（Hubel & Wiesel，1960s）。

1998 年 Yann LeCun 的 **LeNet-5** 用小視窗（kernel）掃描圖片、權重共用，先找線條邊緣再組成數字形狀 —— 成為後來許多 CNN 的祖先。

> 可記一句：**CNN ≈ 用小視窗在圖片上滑動，找重複出現的圖形特徵**
"""
        )


def _render_convolution_stage() -> None:
    st.markdown("## 什麼是卷積（Convolution）？")
    st.markdown(
        """
- 拿一個小方塊（**kernel**／濾鏡）
- 在圖片上從左到右、從上到下滑過去
- 每次只看一小塊（局部）
- 跟這一小塊做「對應相乘再加總」→ 得到一個分數

分數高 ≈ 「這裡很像 kernel 在找的形狀」。
"""
    )
    st.markdown(
        """
### kernel（濾鏡）範例

```text
[-1  -1  -1]
[ 0   0   0]
[ 1   1   1]
```

可以想成在問：

> **「這裡是不是有『上面比較暗、下面比較亮』的水平邊緣？」**
"""
    )
    st.markdown("### Kernel 在圖像上滑動")
    kh, kw = SLIDE_KERNEL.shape
    out_h = DEMO_IMAGE_6X6.shape[0] - kh + 1
    out_w = DEMO_IMAGE_6X6.shape[1] - kw + 1
    step = st.slider(
        "卷積步驟",
        min_value=0,
        max_value=out_h * out_w - 1,
        value=0,
        key="cnn_conv_step",
    )
    _show_figure(build_conv_step_figure(DEMO_IMAGE_6X6, SLIDE_KERNEL, step))

    st.markdown(
        """
### Kernel 尋找特徵

- 圖片切一小塊（patch）跟 kernel 一格一格相乘再加總
- **很像** → 分數大；**不像** → 分數小甚至是負的

> **Kernel 就像一張「形狀模板」**
"""
    )
    _show_figure(build_patch_similarity_figure())

    st.markdown(
        """
### 一個 Kernel ≠ 一個答案

CNN 會同時使用很多個 kernel（有的找直線、有的找角落…）。
每個 kernel 產生一張 **特徵圖（feature map）**。
"""
    )
    _show_figure(build_feature_maps_demo_figure())
    full_map = conv2d_valid(DEMO_IMAGE_6X6, SLIDE_KERNEL)
    st.caption(f"完整卷積輸出 shape：{full_map.shape}")


def _render_relu_pool_stage() -> None:
    st.markdown("## ReLU：把「不像」的負訊號關掉")
    st.caption("本段影片時間戳對準 pooling；ReLU 以下面圖示為主（該片幾乎不單獨講 ReLU）。")
    st.markdown(
        """
卷積結果可正可負。負數多半表示「方向相反或不像」。

```
ReLU(x) = max(0, x)
```

👉 正數留下（像特徵），負數變 0（關掉）。
"""
    )
    _show_figure(build_relu_curve_figure())
    _show_figure(build_relu_image_figure())

    st.markdown("## Pooling：縮小，留下重點")
    st.markdown(
        """
Pooling 像把照片縮小，但人還是看得出重點。
常用 **Max Pooling**：每個小區塊只留最大反應。
"""
    )
    _show_figure(build_pooling_demo_figure())


def _render_flow_stage() -> None:
    st.markdown("## CNN 在做什麼？整條流程")
    st.code(
        """
圖片
 ↓
Conv → ReLU
 ↓
Pooling
 ↓
Conv → ReLU
 ↓
Pooling
 ↓
Flatten
 ↓
全連接層
 ↓
分類結果
""",
        language="text",
    )
    st.markdown(
        """
👉 前面負責「看圖找特徵」（局部掃描、同一組規則重複用）
👉 後面負責「根據特徵做判斷」

空間尺寸通常愈來愈小，特徵通道（不同 kernel 的反應）愈來愈豐富，最後壓成向量再分類。
"""
    )


def _render_hands_on_stage() -> None:
    st.markdown("## 真正的 CNN：它看到的是一堆特徵圖")
    st.markdown("### a. 讀取資料集")
    st.caption("sklearn digits（8×8 手寫數字，0～9）")
    preview_images, preview_labels, _, _ = load_digits_tensors()
    _show_figure(build_digits_preview_figure(preview_images, preview_labels))

    st.markdown("### b. 搭建 CNN 模型")
    st.code(
        """
SimpleCNN(
  (conv): Conv2d(1, 8, kernel_size=(3, 3), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2)
  (fc): Linear(in_features=128, out_features=10)
)
""",
        language="text",
    )
    st.caption(str(SimpleCNN()))

    unlocked = _render_pretrain_quiz()

    st.markdown("### c. 訓練 CNN 模型")
    train_clicked = st.button(
        "開始訓練",
        type="primary",
        key="cnn_train_button",
        disabled=not unlocked,
    )
    if not unlocked:
        st.caption("兩題訓練前預測都答對後，才能開始訓練。卡住時可按各題「Agent 提示」。")

    if train_clicked and unlocked:
        progress = st.progress(0.0, text="準備訓練…")
        status = st.empty()

        def on_progress(epoch: int, total: int, loss: float) -> None:
            progress.progress(epoch / total, text=f"Epoch {epoch}/{total}")
            status.write(f"epoch {epoch}, loss = {loss:.4f}")

        with st.spinner("訓練中…"):
            result = train_simple_cnn(epochs=10, lr=0.01, progress_callback=on_progress)
        st.session_state[RESULT_KEY] = result
        progress.progress(1.0, text="訓練完成")
        st.success(f"訓練完成，測試準確率 = {result.test_accuracy:.3f}")

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        if unlocked:
            st.info("按「開始訓練」以在 sklearn digits 上訓練 SimpleCNN。")
        return

    st.markdown("### d. 進行 CNN 預測")
    st.write(f"accuracy = {result.test_accuracy:.3f}")

    st.markdown("### e. 訓練完模型後，我們來看看")
    st.markdown("👉 **第一層 CNN 眼中的世界**")
    sample = torch.tensor(result.test_images[0:1, None, :, :], dtype=torch.float32)
    feature_maps = extract_first_conv_maps(result.model, sample)
    _show_figure(build_first_conv_maps_figure(feature_maps))

    st.markdown("恭喜！你已經初步了解 CNN 架構與原理。")


def _render_pretrain_quiz() -> bool:
    st.markdown("##### 訓練前先猜一下")
    st.caption("兩題都答對後，「開始訓練」才會啟用。卡住時可按「Agent 提示」問線索（不會直接給正解）。")
    st.session_state.setdefault(quiz.SESSION_KERNEL, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_POOL, quiz.PLEASE_SELECT)
    st.session_state.setdefault(quiz.SESSION_FOCUS, quiz.QID_KERNEL)

    agent_ready = bool(st.session_state.get("data_agent_connected"))

    q1_col, h1_col = st.columns([4, 1])
    with q1_col:
        kernel_choice = st.radio(
            "題1：kernel／卷積在影像上大致在做什麼？",
            [quiz.PLEASE_SELECT, *quiz.KERNEL_OPTIONS],
            key=quiz.SESSION_KERNEL,
        )
    with h1_col:
        st.write("")
        if st.button("Agent 提示", key="cnn_hint_kernel", disabled=not agent_ready, width="stretch"):
            _send_quiz_hint(quiz.QID_KERNEL)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    kernel_ok = quiz.is_kernel_correct(str(kernel_choice))
    if str(kernel_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題1。")
        st.session_state[quiz.SESSION_FOCUS] = quiz.QID_KERNEL
    elif kernel_ok:
        st.caption("題1 OK。")
    else:
        st.caption("題1 再想想：是不是「小模板在圖上找相似局部」？可按「Agent 提示」。")
        st.session_state[quiz.SESSION_FOCUS] = quiz.QID_KERNEL

    q2_col, h2_col = st.columns([4, 1])
    with q2_col:
        pool_choice = st.radio(
            "題2：Pooling 大致在做什麼？",
            [quiz.PLEASE_SELECT, *quiz.POOL_OPTIONS],
            key=quiz.SESSION_POOL,
        )
    with h2_col:
        st.write("")
        if st.button("Agent 提示", key="cnn_hint_pool", disabled=not agent_ready, width="stretch"):
            _send_quiz_hint(quiz.QID_POOL)
        elif not agent_ready:
            st.caption("先啟用 Agent")

    pool_ok = quiz.is_pool_correct(str(pool_choice))
    if str(pool_choice) == quiz.PLEASE_SELECT:
        st.caption("請先選擇題2。")
        if kernel_ok:
            st.session_state[quiz.SESSION_FOCUS] = quiz.QID_POOL
    elif pool_ok:
        st.caption("題2 OK。")
    else:
        st.caption("題2 再想想「縮小後還留下區塊裡最強的反應」，可按「Agent 提示」。")
        st.session_state[quiz.SESSION_FOCUS] = quiz.QID_POOL

    unlocked = quiz.both_quiz_correct(str(kernel_choice), str(pool_choice))
    if unlocked:
        st.success("2／2 題已準備好訓練。")
    else:
        st.info(f"進度：{int(kernel_ok) + int(pool_ok)}／2 題答對（需全部正確才解鎖訓練）。")
    return unlocked


def _send_quiz_hint(qid: str) -> None:
    ts_key = f"cnn_hint_ts_{qid}"
    now = time.time()
    if not quiz.can_send_hint(st.session_state.get(ts_key), now):
        st.caption("提示冷卻中，稍后再試。")
        return
    st.session_state[ts_key] = now
    st.session_state[quiz.SESSION_FOCUS] = qid

    kernel_choice = str(st.session_state.get(quiz.SESSION_KERNEL, quiz.PLEASE_SELECT))
    pool_choice = str(st.session_state.get(quiz.SESSION_POOL, quiz.PLEASE_SELECT))
    unlocked = quiz.both_quiz_correct(kernel_choice, pool_choice)
    appendix = quiz.build_quiz_agent_appendix(
        kernel_status=quiz.quiz_choice_status(
            kernel_choice, correct=quiz.is_kernel_correct(kernel_choice)
        ),
        pool_status=quiz.quiz_choice_status(
            pool_choice, correct=quiz.is_pool_correct(pool_choice)
        ),
        focus_qid=qid,
        unlocked=unlocked,
    )
    invoke_data_agent(
        quiz.hint_user_text(qid),
        extra_context=appendix,
        display_user_text=quiz.hint_display_text(qid),
    )
    st.rerun()


def _compose_agent_context(stage: str) -> None:
    kernel_choice = str(st.session_state.get(quiz.SESSION_KERNEL, quiz.PLEASE_SELECT))
    pool_choice = str(st.session_state.get(quiz.SESSION_POOL, quiz.PLEASE_SELECT))
    unlocked = quiz.both_quiz_correct(kernel_choice, pool_choice)
    result = st.session_state.get(RESULT_KEY)
    accuracy_txt = (
        f"最近一次測試準確率 = {result.test_accuracy:.3f}。"
        if result is not None
        else "尚未訓練。"
    )
    quiz_note = ""
    if stage == quiz.STAGE_HANDS_ON:
        quiz_note = quiz.build_quiz_agent_appendix(
            kernel_status=quiz.quiz_choice_status(
                kernel_choice, correct=quiz.is_kernel_correct(kernel_choice)
            ),
            pool_status=quiz.quiz_choice_status(
                pool_choice, correct=quiz.is_pool_correct(pool_choice)
            ),
            focus_qid=st.session_state.get(quiz.SESSION_FOCUS),
            unlocked=unlocked,
        )
    st.session_state[CONTEXT_KEY] = "\n".join(
        [
            f"目前頁面：{PAGE_TITLE}。",
            f"目前學習階段：{stage}。",
            f"觀念主軸影片：{quiz.VIDEO_TITLE}（{quiz.VIDEO_ID}）。",
            accuracy_txt,
            quiz_note,
        ]
    ).strip()
