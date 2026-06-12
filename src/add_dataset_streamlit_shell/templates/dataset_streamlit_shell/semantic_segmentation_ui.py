from __future__ import annotations

import importlib

import numpy as np
import streamlit as st

from dataset_streamlit_shell.cv.image_io import (
    EXAMPLES_DIR,
    download_sample_data,
    load_image_bytes,
    load_image_path,
    pil_to_rgb_array,
    semantic_demo_specs,
    semantic_examples_ready,
)
from dataset_streamlit_shell.cv.semantic_segmentation import (
    DEFAULT_MODEL,
    ClassCoverage,
    SemanticResult,
    blend_overlay,
    format_semantic_summary,
    highlight_class,
    isolate_class_mask,
    load_segmentation_model,
    predict_semantic_mask,
)
from dataset_streamlit_shell.cv_layout import render_cv_tabbed_page

PAGE_TITLE = "語意分割（Semantic Segmentation）"
CONTEXT_KEY = "semantic_segmentation_agent_context"
RESULT_KEY = "semantic_segmentation_last_result"
IMAGE_KEY = "semantic_segmentation_last_image"


@st.cache_resource(show_spinner="載入 DeepLabv3-ResNet50（COCO 預訓練）…")
def _cached_segmentation_model():
    return load_segmentation_model()


def render_semantic_segmentation_page() -> None:
    render_cv_tabbed_page(
        page_title=PAGE_TITLE,
        context_key=CONTEXT_KEY,
        tab_labels=['概念說明', '分割推論', '結果解讀'],
        tab_renderers=[_render_concept_tab, _render_inference_tab, _render_interpret_tab],
    )


def _render_download_panel() -> bool:
    if semantic_examples_ready():
        return True
    st.info("首次使用請先下載教學用示範圖（需網路連線）。")
    if st.button("下載範例資料", key="cv_semantic_download_samples"):
        progress = st.progress(0.0, text="準備下載…")
        status = st.empty()

        def _callback(message: str, value: float) -> None:
            progress.progress(value, text=message)
            status.caption(message)

        try:
            download_sample_data(progress_callback=_callback)
            st.success("範例資料已下載並快取於本機。")
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - surface download issues in UI
            st.error(f"下載失敗：{exc}")
            st.warning("你仍可使用「上傳影像」進行語意分割。")
    return semantic_examples_ready()


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


def _foreground_items(class_items: tuple[ClassCoverage, ...]) -> tuple[ClassCoverage, ...]:
    return tuple(item for item in class_items if item.class_id != 0)


def _render_concept_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("為每個像素指派語意類別；此頁是電腦視覺路徑的第三站。")
    st.markdown("##### 什麼是語意分割？")
    st.write(
        "模型輸出與原圖同尺寸的像素級類別圖。"
        "同一語意類別的所有像素使用相同顏色，不區分不同個體。"
    )
    st.info(
        "影像分類：整張圖 → 一個標籤\n\n"
        "物件偵測：多個 bbox + 類別\n\n"
        "語意分割：每個像素 → 一個類別"
    )
    cols = st.columns(4)
    steps = ["輸入影像", "Encoder-Decoder", "Per-pixel logits", "Argmax 類別圖"]
    for column, step in zip(cols, steps, strict=True):
        column.markdown(f"**{step}**")
    with st.expander("與實例分割的差異", expanded=False):
        st.write(
            "語意分割中，兩個 person 會是同一顏色；"
            "實例分割則會分成兩塊不同 mask；可到「實例分割」頁觀察。"
        )
    with st.expander("DeepLabv3 簡介", expanded=False):
        st.write(
            "DeepLabv3 使用 encoder-decoder 與 atrous convolution，"
            "在 COCO 上預訓練，可輸出約 21 類語意區域（含 background）。"
        )
    with st.expander("與物件偵測的差異", expanded=False):
        st.write(
            "偵測框只能框住物件外圍；語意分割能標出道路、天空、人等區域的像素歸屬，"
            "適合街景與場景理解。"
        )
    st.caption(f"本頁使用 {DEFAULT_MODEL}（COCO 預訓練）。")


def _render_inference_tab() -> None:
    st.title(PAGE_TITLE)
    st.title(PAGE_TITLE)
    st.caption("使用 DeepLabv3-ResNet50 產生像素級語意遮罩。")
    ready = _render_download_panel()
    source_mode = st.radio(
        "資料來源",
        ["內建範例圖片", "上傳影像"],
        horizontal=True,
        key="cv_semantic_source_mode",
    )
    uploaded = None
    selected_example = None
    if source_mode == "內建範例圖片":
        if ready:
            specs = semantic_demo_specs()
            selected_example = st.selectbox(
                "選擇示範圖",
                options=[spec.filename for spec in specs],
                format_func=lambda name: next(
                    spec.hint for spec in specs if spec.filename == name
                ),
                key="cv_semantic_example",
            )
        else:
            st.warning("請先下載範例資料，或改用上傳影像。")
    else:
        uploaded = st.file_uploader(
            "上傳影像",
            type=["jpg", "jpeg", "png", "webp"],
            key="cv_semantic_upload",
        )

    image = _resolve_image(
        source_mode=source_mode,
        uploaded_file=uploaded,
        selected_example=selected_example,
    )
    if image is not None:
        st.image(image, caption=f"{image.shape[1]}×{image.shape[0]}", use_container_width=True)

    st.caption("模型：DeepLabv3-ResNet50（COCO 預訓練）")
    alpha = st.slider(
        "遮罩透明度",
        min_value=0.20,
        max_value=0.80,
        value=0.45,
        step=0.05,
        key="cv_semantic_alpha",
    )
    run = st.button("執行分割", key="cv_semantic_run", disabled=image is None)

    if run and image is not None:
        with st.spinner("分割中…"):
            model, weights = _cached_segmentation_model()
            result = predict_semantic_mask(image, model=model, weights=weights)
        st.session_state[RESULT_KEY] = result
        st.session_state[IMAGE_KEY] = image
        st.session_state["cv_semantic_alpha_saved"] = alpha
        _update_agent_context(image, result)

    result: SemanticResult | None = st.session_state.get(RESULT_KEY)
    cached_image = st.session_state.get(IMAGE_KEY)
    if result is not None and cached_image is not None:
        display_alpha = float(st.session_state.get("cv_semantic_alpha_saved", alpha))
        _render_segmentation_results(cached_image, result, display_alpha)



def _render_segmentation_results(
    image: np.ndarray,
    result: SemanticResult,
    alpha: float,
) -> None:
    blended = blend_overlay(image, result.color_overlay, alpha=alpha)
    left, right = st.columns(2)
    with left:
        st.image(blended, caption="語意分割疊圖", use_container_width=True)
    with right:
        st.markdown("##### 類別圖例")
        foreground = _foreground_items(result.class_items)
        if foreground:
            _render_legend(foreground)
            st.dataframe(
                _coverage_dataframe(result.class_items),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(format_semantic_summary(result.class_items))
        else:
            st.warning("此圖僅偵測到 background 或無明顯前景類別。")


def _render_legend(class_items: tuple[ClassCoverage, ...]) -> None:
    for item in class_items:
        color_hex = "#{:02x}{:02x}{:02x}".format(*item.color)
        st.markdown(
            f"<span style='color:{color_hex}; font-weight:600;'>■</span> "
            f"{item.label} · {item.coverage:.1%}",
            unsafe_allow_html=True,
        )


def _render_interpret_tab() -> None:
    st.title(PAGE_TITLE)
    st.title(PAGE_TITLE)
    st.caption("調整遮罩透明度、檢視單一類別 mask，不需重新推論。")
    result: SemanticResult | None = st.session_state.get(RESULT_KEY)
    image = st.session_state.get(IMAGE_KEY)
    if result is None or image is None:
        st.info("請先到「分割推論」Tab 執行分割。")
    else:
        alpha = st.slider(
            "遮罩透明度",
            min_value=0.20,
            max_value=0.80,
            value=float(st.session_state.get("cv_semantic_alpha_saved", 0.45)),
            step=0.05,
            key="cv_semantic_interpret_alpha",
        )
        foreground = _foreground_items(result.class_items)
        if not foreground:
            st.warning("沒有前景類別可供解讀。")
        else:
            blended = blend_overlay(image, result.color_overlay, alpha=alpha)
            cols = st.columns(3)
            cols[0].image(image, caption="原圖")
            cols[1].image(result.color_overlay, caption="純色塊 mask")
            cols[2].image(blended, caption="疊加圖")

            options = [
                f"{item.label} ({item.coverage:.1%})" for item in foreground
            ]
            selected = st.selectbox(
                "單類別檢視",
                options=options,
                key="cv_semantic_class_select",
            )
            selected_index = options.index(selected)
            selected_item = foreground[selected_index]
            binary_mask = isolate_class_mask(result.label_map, selected_item.class_id)
            highlighted = highlight_class(image, result.label_map, selected_item.class_id)
            mask_rgb = np.stack([binary_mask * 255] * 3, axis=-1)
            sub_cols = st.columns(2)
            sub_cols[0].image(mask_rgb, caption=f"{selected_item.label} binary mask")
            sub_cols[1].image(
                highlighted,
                caption=f"{selected_item.label} highlighted",
                use_container_width=True,
            )
            st.caption(
                f"{selected_item.label} 覆蓋 {selected_item.pixel_count:,} 像素 "
                f"（{selected_item.coverage:.1%}）"
            )
        with st.expander("為何需要語意分割？", expanded=False):
            st.write(
                "Bounding box 無法描述像素級歸屬；"
                "語意分割適合街景、道路、室內場景等需要理解「區域」的任務。"
            )


def _coverage_dataframe(class_items: tuple[ClassCoverage, ...]):
    pd = importlib.import_module("pandas")
    return pd.DataFrame(
        {
            "class": [item.label for item in class_items],
            "pixel_count": [item.pixel_count for item in class_items],
            "coverage": [round(item.coverage, 4) for item in class_items],
        }
    )


def _update_agent_context(image: np.ndarray, result: SemanticResult) -> None:
    lines = [
        f"目前頁面：{PAGE_TITLE}。",
        f"影像尺寸：{image.shape[1]}×{image.shape[0]}。",
        f"模型：{result.model_name}",
        format_semantic_summary(result.class_items),
    ]
    foreground = _foreground_items(result.class_items)
    if foreground:
        labels = ", ".join(f"{item.label} {item.coverage:.1%}" for item in foreground[:3])
        lines.append(f"Top regions：{labels}")
    st.session_state[CONTEXT_KEY] = "\n".join(lines)
