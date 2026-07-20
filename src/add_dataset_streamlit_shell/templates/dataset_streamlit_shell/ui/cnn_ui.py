from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
import torch

from dataset_streamlit_shell.ui.cv_layout import render_cv_tabbed_page
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

configure_matplotlib_for_traditional_chinese()

CONTEXT_KEY = "卷積神經網路_agent_context"
RESULT_KEY = "cnn_last_result"
PAGE_TITLE = "卷積神經網路（CNN）"

TAB_LABELS = (
    "圖片與矩陣",
    "為什麼需要 CNN",
    "CNN 流程",
    "卷積 Convolution",
    "ReLU 層",
    "Pooling",
    "動手做 CNN",
)


def render_cnn_introduction_page() -> None:
    render_cv_tabbed_page(
        page_title=PAGE_TITLE,
        context_key=CONTEXT_KEY,
        tab_labels=TAB_LABELS,
        tab_renderers=(
            _render_image_matrix_tab,
            _render_why_cnn_tab,
            _render_cnn_flow_tab,
            _render_convolution_tab,
            _render_relu_tab,
            _render_pooling_tab,
            _render_hands_on_tab,
        ),
    )


def _show_figure(fig) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_image_matrix_tab() -> None:
    st.markdown("## 1. 電腦眼中的「圖片」是什麼？")
    st.markdown(
        """
在電腦裡：

👉 **圖片其實是一個數字表格（矩陣）**

- 每一個格子 = 一個像素（pixel）
- 數字大小 = 亮度（灰階）或顏色強度

先用一張「超小圖片」來看。
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


def _render_why_cnn_tab() -> None:
    st.markdown("## 2. CNN 由來")
    st.markdown(
        """
如果我們用 **傳統全連接神經網路（Dense）** 來看圖片：

- 一張 64×64 圖片 = **4096 個輸入**
- 參數很多 👉 很容易爆炸 💥
- 邊緣、形狀這些「局部特徵」會被忽略

👉 所以我們需要一個：
**「會看局部、會重複使用同一組規則」的模型**

---

### 生物啟發：人類是怎麼「看」的？

科學家在研究人類與動物視覺皮層時發現：

- 👁️ 眼睛不是一次看完整畫面
- 🧩 大腦是「先看小區域」
- 🔁 同一種「找線條的方法」會在不同位置重複使用

👉 這個發現來自 **Hubel & Wiesel（1960s）**
他們發現：
- 有些神經元專門對「水平線」有反應
- 有些對「垂直線」有反應

這正是 **CNN kernel 的靈感來源**。

---

### ✨ CNN 的突破（LeNet-5）

1998 年，Yann LeCun 發表 **LeNet-5**：

- 用小視窗（kernel）掃描圖片
- 同一組權重在整張圖上重複使用
- 先找「線條、邊緣」，再組成「數字、形狀」

👉 這套設計，成為 **今天所有 CNN 的祖先**。

> **CNN = 用小視窗在圖片上滑動，找重複出現的圖形特徵**
"""
    )


def _render_cnn_flow_tab() -> None:
    st.markdown("## 3. CNN 在做什麼？")
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
👉 前面負責「看圖找特徵」
👉 後面負責「根據特徵做判斷」
"""
    )


def _render_convolution_tab() -> None:
    st.markdown("## 4. 什麼是 Convolution（卷積）？")
    st.markdown(
        """
- 🔍 拿一個小方塊（kernel / filter）
- 🔍 在圖片上「從左到右，從上到下滑過去」
- 🔍 每次只看一小塊（局部）
- 🔍 找出「常出現的形狀或顏色組合」
"""
    )
    st.markdown(
        """
### a. kernel（濾鏡）

```text
[-1  -1  -1]
[ 0   0   0]
[ 1   1   1]
```

這個 kernel 可以想成在問：

> **「這裡是不是有『上面比較暗、下面比較亮』的水平邊緣？」**
"""
    )
    st.markdown("### b. Kernel 在圖像上滑動")
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
### c. Kernel 尋找特徵

- 圖片從窗口切一小塊（patch）
- 跟 kernel 一格一格相乘
- 全部加起來 → 得到一個數字

**很像 kernel 的 patch**
  - 乘起來幾乎都在「加分」
  - 👉 數值會變得很大

**有點像的 patch**
  - 有些地方加分、有些地方普通
  - 👉 數值中等

**不像的 patch**
  - 正負亂七八糟、彼此抵消
  - 👉 數值接近 0，甚至是負的

> **Kernel 就像一張「形狀模板」**
> 當它滑到「很像自己要找的特徵」時，算出來的數值會 **很大**；
> 滑到「不像的地方」，數值就會 **很小，甚至是負的**。
"""
    )
    _show_figure(build_patch_similarity_figure())

    st.markdown(
        """
### d. 一個 Kernel ≠ 一個答案

CNN 會同時使用 **很多個 kernel**：

- 有的找「直線」
- 有的找「角落」
- 有的找「顏色變化」

👉 每個 kernel 都會產生一張 **Feature Map**
"""
    )
    _show_figure(build_feature_maps_demo_figure())

    full_map = conv2d_valid(DEMO_IMAGE_6X6, SLIDE_KERNEL)
    st.caption(f"完整卷積輸出 shape：{full_map.shape}")


def _render_relu_tab() -> None:
    st.markdown("## 5. ReLU 層：為什麼 CNN 需要「把負數變 0」？")
    st.markdown(
        """
在前面我們看到：
👉 **卷積的結果可能是正數、0、也可能是負數**

那問題來了：

> 這些「負數」對影像辨識有幫助嗎？

所以:
> 不重要的訊號直接丟掉，重要的留下來

數學寫法其實很簡單：

```
ReLU(x) = max(0, x)
```
"""
    )
    _show_figure(build_relu_curve_figure())
    st.markdown(
        """
### 👀 用影像直覺來理解 ReLU

- 正數 👉「這裡很像 kernel 在找的特徵」
- 負數 👉「方向相反或雜訊」

👉 **ReLU 會把不像的地方直接關掉（變成 0）**，
只留下值得後面層繼續使用的訊號。
"""
    )
    _show_figure(build_relu_image_figure())


def _render_pooling_tab() -> None:
    st.markdown("## 6. Pooling：留下重點，其它不管")
    st.markdown(
        """
Pooling 就像：
- 📸 把照片縮小
- 👀 但人還是看得出重點

下面用 **顏色深淺** 來表示 Max Pooling。
"""
    )
    _show_figure(build_pooling_demo_figure())


def _render_hands_on_tab() -> None:
    st.markdown("## 7. 真正的 CNN：它看到的不是圖片，而是「一堆特徵圖」")
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

    st.markdown("### c. 訓練 CNN 模型")
    if st.button("開始訓練", type="primary", key="cnn_train_button"):
        progress = st.progress(0.0, text="準備訓練…")
        status = st.empty()

        def on_progress(epoch: int, total: int, loss: float) -> None:
            progress.progress(epoch / total, text=f"Epoch {epoch}/{total}")
            status.write(f"epoch {epoch}, loss = {loss:.4f}")

        with st.spinner("訓練中…"):
            result = train_simple_cnn(epochs=10, lr=0.01, progress_callback=on_progress)
        st.session_state[RESULT_KEY] = result
        st.session_state[CONTEXT_KEY] = (
            f"SimpleCNN 訓練完成：test accuracy = {result.test_accuracy:.3f}，"
            f"最後 epoch loss = {result.epoch_losses[-1]:.4f}。"
        )
        progress.progress(1.0, text="訓練完成")
        st.success(f"訓練完成，測試準確率 = {result.test_accuracy:.3f}")

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.info("按「開始訓練」以在 sklearn digits 上訓練 SimpleCNN。")
        return

    st.markdown("### d. 進行 CNN 預測")
    st.write(f"accuracy = {result.test_accuracy:.3f}")

    st.markdown("### e. 訓練完模型後，我們來看看")
    st.markdown("👉 **第一層 CNN 眼中的世界**")
    sample = torch.tensor(result.test_images[0:1, None, :, :], dtype=torch.float32)
    feature_maps = extract_first_conv_maps(result.model, sample)
    _show_figure(build_first_conv_maps_figure(feature_maps))

    st.markdown("# 恭喜!")
    st.markdown("你已經初步了解 CNN 架構與原理!")
