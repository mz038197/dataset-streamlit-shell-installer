from __future__ import annotations

import importlib

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from dataset_streamlit_shell.ml.tf_runtime import configure_tensorflow_runtime

configure_tensorflow_runtime()
import tensorflow as tf

from dataset_streamlit_shell.cv.classification import (
    ClassificationResult,
    PredictionItem,
    StageActivation,
    compute_grad_cam,
    extract_stage_activations,
    format_top_prediction_summary,
    load_classifier,
    predict_top_k,
)
from dataset_streamlit_shell.cv.image_io import (
    EXAMPLES_DIR,
    demo_image_specs,
    examples_ready,
    load_image_bytes,
    load_image_path,
    overlay_heatmap,
    pil_to_rgb_array,
)
from dataset_streamlit_shell.ui.cv_layout import render_cv_tabbed_page
from dataset_streamlit_shell.plotting import configure_matplotlib_for_traditional_chinese

configure_matplotlib_for_traditional_chinese()

PAGE_TITLE = "影像分類（Image Classification）"
CONTEXT_KEY = "image_classification_agent_context"
RESULT_KEY = "image_classification_last_results"
STAGES_KEY = "image_classification_stage_cache"
GRADCAM_KEY = "image_classification_gradcam_cache"


@st.cache_resource(show_spinner="載入 ResNet50（ImageNet 預訓練）…")
def _cached_resnet50() -> tf.keras.Model:
    return load_classifier("resnet50")


@st.cache_resource(show_spinner="載入 MobileNetV2（ImageNet 預訓練）…")
def _cached_mobilenet_v2() -> tf.keras.Model:
    return load_classifier("mobilenet_v2")


def render_image_classification_page() -> None:
    render_cv_tabbed_page(
        page_title=PAGE_TITLE,
        context_key=CONTEXT_KEY,
        tab_labels=['概念說明', '分類推論', '特徵歷程'],
        tab_renderers=[_render_concept_tab, _render_inference_tab, _render_feature_tab],
    )


def _render_download_panel() -> bool:
    if examples_ready():
        return True
    st.warning(
        "找不到內建範例圖。請重新執行 add-dataset-streamlit-shell --update，"
        "或改用上傳影像。"
    )
    return False


def _resolve_image(
    *,
    source_mode: str,
    uploaded_file,
    selected_example: str | None,
) -> np.ndarray | None:
    if source_mode == "上傳影像":
        if uploaded_file is None:
            return None
        return pil_to_rgb_array(load_image_bytes(uploaded_file.getvalue()))
    if not selected_example:
        return None
    path = EXAMPLES_DIR / selected_example
    if not path.exists():
        return None
    return pil_to_rgb_array(load_image_path(path))


def _render_concept_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("整張圖輸出一個語意標籤；此頁是電腦視覺路徑的第一站。")
    st.markdown("##### 什麼是影像分類？")
    st.write(
        "模型讀入整張圖片，輸出它屬於哪個類別（例如 golden retriever、coffee mug）。"
        "它回答的是「這是什麼」，還不涉及「在哪裡」。"
    )
    st.info(
        "分類：整張圖 → 一個標籤\n\n"
        "物件偵測（下一頁）：還要回答在哪裡\n\n"
        "語意分割（下一頁）：進一步到像素層級"
    )
    cols = st.columns(4)
    steps = ["輸入影像", "CNN 特徵抽取", "Softmax 機率", "Top-1 預測"]
    for column, step in zip(cols, steps, strict=True):
        column.markdown(f"**{step}**")
    with st.expander("ImageNet 是什麼？", expanded=False):
        st.write(
            "ImageNet 是含約 1000 類日常物件的資料集／基準。"
            "本頁模型在其上預訓練，因此可直接辨識常見物件。"
        )
    with st.expander("ResNet50 vs MobileNetV2", expanded=False):
        st.write(
            "兩者都是 CNN 架構。ResNet50 為本頁預設主骨幹，教科書常見、通常較準；"
            "MobileNetV2 較輕量，適合快速對照。"
        )
    with st.expander("常見限制", expanded=False):
        st.write(
            "整圖只有一個主標籤；無法標出物件位置；"
            "多主體場景時可能混淆。可到「物件偵測」頁定位、「語意分割」頁看區域，或「實例分割」頁區分每個獨立物件。"
        )
    st.caption(
        "深度學習頁從零訓練小資料；此處使用遷移學習——"
        "在 ImageNet 上學過特徵的預訓練模型。"
    )


def _render_inference_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("使用 ImageNet 預訓練模型進行 Top-K 分類推論。")
    ready = _render_download_panel()
    source_mode = st.radio(
        "資料來源",
        ["內建範例圖片", "上傳影像"],
        horizontal=True,
        key="cv_infer_source_mode",
    )
    uploaded = None
    selected_example = None
    if source_mode == "內建範例圖片":
        if ready:
            specs = demo_image_specs()
            selected_example = st.selectbox(
                "選擇示範圖",
                options=[spec.filename for spec in specs],
                format_func=lambda name: next(
                    spec.hint for spec in specs if spec.filename == name
                ),
                key="cv_infer_example",
            )
        else:
            st.warning("找不到內建範例圖，請改用上傳影像。")
    else:
        uploaded = st.file_uploader(
            "上傳影像",
            type=["jpg", "jpeg", "png", "webp"],
            key="cv_infer_upload",
        )

    image = _resolve_image(
        source_mode=source_mode,
        uploaded_file=uploaded,
        selected_example=selected_example,
    )
    if image is not None:
        st.image(image, caption=f"{image.shape[1]}×{image.shape[0]}", width="stretch")

    backbone_mode = st.radio(
        "模型骨幹",
        ["ResNet50（預設）", "MobileNetV2（輕量對照）", "兩者比較"],
        horizontal=True,
        key="cv_infer_backbone_mode",
    )
    top_k = st.slider("Top-K", min_value=1, max_value=5, value=5, key="cv_infer_top_k")
    run = st.button("執行分類", key="cv_infer_run", disabled=image is None)

    if run and image is not None:
        results = _run_classification(image, backbone_mode, top_k)
        st.session_state[RESULT_KEY] = results
        st.session_state[STAGES_KEY] = None
        st.session_state[GRADCAM_KEY] = None
        _update_agent_context(image, results)

    results = st.session_state.get(RESULT_KEY)
    if image is not None and results:
        _render_classification_results(image, results, top_k)



def _run_classification(
    image: np.ndarray,
    backbone_mode: str,
    top_k: int,
) -> dict[str, ClassificationResult]:
    if backbone_mode == "MobileNetV2（輕量對照）":
        return {
            "mobilenet_v2": predict_top_k(
                image,
                "mobilenet_v2",
                k=top_k,
                model=_cached_mobilenet_v2(),
            )
        }
    if backbone_mode == "兩者比較":
        return {
            "resnet50": predict_top_k(image, "resnet50", k=top_k, model=_cached_resnet50()),
            "mobilenet_v2": predict_top_k(
                image,
                "mobilenet_v2",
                k=top_k,
                model=_cached_mobilenet_v2(),
            ),
        }
    return {
        "resnet50": predict_top_k(image, "resnet50", k=top_k, model=_cached_resnet50()),
    }


def _render_classification_results(
    image: np.ndarray,
    results: dict[str, ClassificationResult],
    top_k: int,
) -> None:
    if len(results) == 1:
        result = next(iter(results.values()))
        left, right = st.columns(2)
        with left:
            st.image(image, caption="原圖", width="stretch")
        with right:
            _render_prediction_card(result, top_k)
        with st.expander("進階資訊", expanded=False):
            st.image(
                result.preprocessed_preview,
                caption="模型輸入（224×224 前處理後）",
                width="stretch",
            )
            st.caption(
                f"骨幹：{result.backbone} · ImageNet 預訓練 · 輸出 Top-{top_k}"
            )
            st.dataframe(
                _prediction_dataframe(result.top_items[:top_k]),
                width="stretch",
                hide_index=True,
            )
        return

    st.markdown("##### 兩者比較")
    columns = st.columns(2)
    for column, (name, result) in zip(columns, results.items(), strict=True):
        with column:
            st.markdown(f"**{name}**")
            _render_prediction_card(result, top_k)
    first = results["resnet50"].top_items[0]
    second = results["mobilenet_v2"].top_items[0]
    if first.label == second.label:
        st.success(f"Top-1 一致：{first.label}")
    else:
        st.warning(
            f"Top-1 不同：ResNet50 → {first.label} ({first.probability:.1%})；"
            f"MobileNetV2 → {second.label} ({second.probability:.1%})"
        )


def _render_prediction_card(result: ClassificationResult, top_k: int) -> None:
    items = result.top_items[:top_k]
    if not items:
        st.warning("沒有預測結果。")
        return
    top = items[0]
    st.markdown(f"### {top.label}")
    st.markdown(f"**{top.probability:.1%}**")
    if top.probability < 0.30:
        st.warning("信心偏低；可能是多物件或主體不明顯的場景。")
    fig = _build_topk_figure(items)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption(format_top_prediction_summary(items))


def _prediction_dataframe(items: tuple[PredictionItem, ...]):
    pd = importlib.import_module("pandas")
    return pd.DataFrame(
        {
            "rank": [item.rank for item in items],
            "label": [item.label for item in items],
            "probability": [round(item.probability, 4) for item in items],
        }
    )


def _build_topk_figure(items: tuple[PredictionItem, ...]):
    labels = [item.label for item in reversed(items)]
    scores = [item.probability for item in reversed(items)]
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.55 * len(items))), constrained_layout=True)
    bars = ax.barh(labels, scores, color="#1a73e8")
    bars[-1].set_color("#174ea6")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("probability")
    ax.set_title("Top-K predictions")
    ax.grid(True, axis="x", alpha=0.25)
    return fig


def _render_feature_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("觀察 ResNet50 推論時的逐層特徵圖與 Grad-CAM 熱力圖。")
    ready = _render_download_panel()
    source_mode = st.radio(
        "資料來源",
        ["沿用分類推論頁", "內建範例圖片", "上傳影像"],
        horizontal=True,
        key="cv_feature_source_mode",
    )
    image = None
    if source_mode == "沿用分類推論頁":
        last = st.session_state.get(RESULT_KEY)
        if last:
            st.info("將使用分類推論頁最近一次執行時的影像設定。")
        source_mode = st.session_state.get("cv_infer_source_mode", "內建範例圖片")
        image = _resolve_image(
            source_mode=source_mode,
            uploaded_file=st.session_state.get("cv_infer_upload"),
            selected_example=st.session_state.get("cv_infer_example"),
        )
    elif source_mode == "內建範例圖片":
        if ready:
            specs = demo_image_specs()
            selected = st.selectbox(
                "選擇示範圖",
                options=[spec.filename for spec in specs],
                key="cv_feature_example",
            )
            image = _resolve_image(
                source_mode="內建範例圖片",
                uploaded_file=None,
                selected_example=selected,
            )
        else:
            st.warning("找不到內建範例圖，請改用上傳影像。")
    else:
        uploaded = st.file_uploader(
            "上傳影像",
            type=["jpg", "jpeg", "png", "webp"],
            key="cv_feature_upload",
        )
        image = _resolve_image(
            source_mode="上傳影像",
            uploaded_file=uploaded,
            selected_example=None,
        )

    if image is not None:
        st.image(image, width="stretch")

    if st.button("擷取特徵歷程", key="cv_feature_extract", disabled=image is None):
        model = _cached_resnet50()
        stages = extract_stage_activations(image, "resnet50", model=model)
        st.session_state[STAGES_KEY] = stages
        top = predict_top_k(image, "resnet50", k=5, model=model)
        class_index = top.top_items[0].class_index
        heatmap = compute_grad_cam(image, "resnet50", class_index, model=model)
        st.session_state[GRADCAM_KEY] = {
            "heatmap": heatmap,
            "class_index": class_index,
            "label": top.top_items[0].label,
            "top_items": top.top_items,
        }

    stages: list[StageActivation] | None = st.session_state.get(STAGES_KEY)
    if stages:
        stage_index = st.slider(
            "層級",
            min_value=0,
            max_value=len(stages) - 1,
            value=0,
            key="cv_feature_stage_slider",
        )
        stage = stages[stage_index]
        st.markdown(f"**{stage.title}** · `{stage.shape_label}`")
        st.caption(stage.caption)
        if stage.vector_preview is not None and stage.feature_maps is None:
            if stage.stage_id == "original":
                st.image(stage.vector_preview, width="stretch")
            else:
                st.write("Pooled feature vector preview:", stage.vector_preview[:8])
        if stage.feature_maps is not None:
            grid = _feature_map_grid(stage.feature_maps)
            st.image(grid, caption="前 16 個 channel（灰階）", width="stretch")

    gradcam = st.session_state.get(GRADCAM_KEY)
    if gradcam and image is not None:
        st.divider()
        st.markdown("##### Grad-CAM")
        labels = [item.label for item in gradcam["top_items"]]
        selected_label = st.selectbox(
            "目標類別",
            options=labels,
            key="cv_gradcam_label",
        )
        selected_item = next(
            item for item in gradcam["top_items"] if item.label == selected_label
        )
        if selected_item.class_index != gradcam["class_index"]:
            heatmap = compute_grad_cam(
                image,
                "resnet50",
                selected_item.class_index,
                model=_cached_resnet50(),
            )
        else:
            heatmap = gradcam["heatmap"]
        cols = st.columns(3)
        overlay = overlay_heatmap(image, heatmap)
        cols[0].image(image, caption="原圖")
        cols[1].image((heatmap * 255).astype(np.uint8), caption="熱力圖")
        cols[2].image(overlay, caption="疊加圖")
        st.caption("顏色越暖表示該區域對此預測貢獻越大。")



def _feature_map_grid(feature_maps: np.ndarray) -> np.ndarray:
    channels = feature_maps.shape[-1]
    columns = 4
    rows = int(np.ceil(channels / columns))
    tile_h, tile_w = feature_maps.shape[0], feature_maps.shape[1]
    grid = np.zeros((rows * tile_h, columns * tile_w), dtype=np.uint8)
    for index in range(channels):
        row = index // columns
        col = index % columns
        tile = (feature_maps[..., index] * 255.0).astype(np.uint8)
        grid[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    return np.stack([grid, grid, grid], axis=-1)


def _update_agent_context(image: np.ndarray, results: dict[str, ClassificationResult]) -> None:
    lines = [f"目前頁面：{PAGE_TITLE}。", f"影像尺寸：{image.shape[1]}×{image.shape[0]}。"]
    for backbone, result in results.items():
        tops = ", ".join(
            f"{item.label} {item.probability:.1%}" for item in result.top_items[:3]
        )
        lines.append(f"{backbone} Top-3：{tops}")
    st.session_state[CONTEXT_KEY] = "\n".join(lines)
