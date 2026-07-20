from __future__ import annotations

import importlib

import numpy as np
import streamlit as st

from dataset_streamlit_shell.cv.image_io import (
    EXAMPLES_DIR,
    instance_demo_specs,
    instance_examples_ready,
    load_image_bytes,
    load_image_path,
    pil_to_rgb_array,
)
from dataset_streamlit_shell.cv.instance_segmentation import (
    DEFAULT_MODEL,
    InstanceItem,
    InstanceResult,
    blend_overlay,
    build_color_overlay,
    crop_instance,
    draw_instances,
    filter_instances,
    format_instance_summary,
    highlight_instance,
    isolate_instance_mask,
    load_yolo_seg_model,
    predict_instances,
)
from dataset_streamlit_shell.ui.cv_layout import render_cv_tabbed_page

PAGE_TITLE = "實例分割（Instance Segmentation）"
CONTEXT_KEY = "instance_segmentation_agent_context"
RESULT_KEY = "instance_segmentation_last_result"
IMAGE_KEY = "instance_segmentation_last_image"


@st.cache_resource(show_spinner="載入 YOLOv8n-seg（COCO 預訓練）…")
def _cached_yolo_seg():
    return load_yolo_seg_model()


def render_instance_segmentation_page() -> None:
    render_cv_tabbed_page(
        page_title=PAGE_TITLE,
        context_key=CONTEXT_KEY,
        tab_labels=['概念說明', '分割推論', '結果解讀'],
        tab_renderers=[_render_concept_tab, _render_inference_tab, _render_interpret_tab],
    )


def _render_download_panel() -> bool:
    if instance_examples_ready():
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
    st.caption("為每個獨立物件產生像素級 mask；此頁是電腦視覺路徑的第四站，下一站為提示式分割。")
    st.markdown("##### 什麼是實例分割？")
    st.write(
        "模型同時輸出類別、位置與像素級遮罩。"
        "即使兩個物件屬於同一類別，也會得到不同顏色的獨立 mask。"
    )
    st.info(
        "影像分類：整張圖 → 一個標籤\n\n"
        "物件偵測：多個 bbox + 類別\n\n"
        "語意分割：每個像素 → 類別（同類同色）\n\n"
        "實例分割：每個物件 → 獨立 mask"
    )
    cols = st.columns(4)
    steps = ["輸入影像", "Backbone", "偵測 + mask head", "每實例獨立遮罩"]
    for column, step in zip(cols, steps, strict=True):
        column.markdown(f"**{step}**")
    with st.expander("與語意分割的差異", expanded=False):
        st.write(
            "語意分割中，兩個 person 會是同一顏色；"
            "實例分割則會分成兩塊不同 mask。"
            "建議用「Three puppies」或「Three cats」示範圖觀察多個同類實例的獨立輪廓。"
        )
    with st.expander("YOLOv8n-seg 簡介", expanded=False):
        st.write(
            "YOLOv8n-seg 在 COCO 上預訓練，"
            "結合物件偵測與實例 mask，"
            "適合即時互動示範。"
        )
    with st.expander("與物件偵測的關係", expanded=False):
        st.write(
            "偵測頁的 bbox 只能框住外圍；"
            "實例分割進一步描繪物件輪廓，"
            "適合需要精確形狀的場景。"
        )
    st.caption(f"本頁使用 {DEFAULT_MODEL}（COCO 預訓練）。")


def _render_inference_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("使用 YOLOv8n-seg 產生每個物件的獨立像素遮罩。")
    ready = _render_download_panel()
    source_mode = st.radio(
        "資料來源",
        ["內建範例圖片", "上傳影像"],
        horizontal=True,
        key="cv_instance_source_mode",
    )
    uploaded = None
    selected_example = None
    if source_mode == "內建範例圖片":
        if ready:
            specs = instance_demo_specs()
            selected_example = st.selectbox(
                "選擇示範圖",
                options=[spec.filename for spec in specs],
                format_func=lambda name: next(
                    spec.hint for spec in specs if spec.filename == name
                ),
                key="cv_instance_example",
            )
        else:
            st.warning("找不到內建範例圖，請改用上傳影像。")
    else:
        uploaded = st.file_uploader(
            "上傳影像",
            type=["jpg", "jpeg", "png", "webp"],
            key="cv_instance_upload",
        )

    image = _resolve_image(
        source_mode=source_mode,
        uploaded_file=uploaded,
        selected_example=selected_example,
    )
    if image is not None:
        st.image(image, caption=f"{image.shape[1]}×{image.shape[0]}", width="stretch")

    st.caption("模型：YOLOv8n-seg（COCO 預訓練）")
    conf_threshold = st.slider(
        "信心門檻",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="cv_instance_conf_threshold",
    )
    alpha = st.slider(
        "遮罩透明度",
        min_value=0.20,
        max_value=0.80,
        value=0.45,
        step=0.05,
        key="cv_instance_alpha",
    )
    run = st.button("執行分割", key="cv_instance_run", disabled=image is None)

    if run and image is not None:
        with st.spinner("分割中…"):
            result = predict_instances(
                image,
                conf_threshold=conf_threshold,
                model=_cached_yolo_seg(),
            )
        st.session_state[RESULT_KEY] = result
        st.session_state[IMAGE_KEY] = image
        st.session_state["cv_instance_alpha_saved"] = alpha
        _update_agent_context(image, result)

    result: InstanceResult | None = st.session_state.get(RESULT_KEY)
    cached_image = st.session_state.get(IMAGE_KEY)
    if result is not None and cached_image is not None:
        display_alpha = float(st.session_state.get("cv_instance_alpha_saved", alpha))
        _render_instance_results(cached_image, result, display_alpha)



def _render_instance_results(
    image: np.ndarray,
    result: InstanceResult,
    alpha: float,
) -> None:
    items = result.items
    if items:
        annotated = draw_instances(image, items, alpha=alpha)
        st.image(annotated, caption="實例分割結果（mask + bbox）", width="stretch")
    else:
        st.image(image, caption="原圖（無分割結果）", width="stretch")
        st.warning("未偵測到實例。試著降低信心門檻後重新執行分割。")
        return

    st.markdown("##### 實例清單")
    legend_col, table_col = st.columns([1, 2], gap="medium")
    with legend_col:
        _render_legend(items)
    with table_col:
        st.dataframe(_instance_dataframe(items), width="stretch", hide_index=True)
        st.caption(format_instance_summary(items))


def _render_legend(items: tuple[InstanceItem, ...]) -> None:
    for item in items:
        color_hex = "#{:02x}{:02x}{:02x}".format(*item.color)
        st.markdown(
            f"<span style='color:{color_hex}; font-weight:600;'>■</span> "
            f"{item.rank}. {item.label} · {item.coverage:.1%}",
            unsafe_allow_html=True,
        )


def _render_interpret_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("調整顯示門檻、檢視單一實例 mask，不需重新推論。")
    result: InstanceResult | None = st.session_state.get(RESULT_KEY)
    image = st.session_state.get(IMAGE_KEY)
    if result is None or image is None:
        st.info("請先到「分割推論」Tab 執行分割。")
    else:
        display_threshold = st.slider(
            "顯示門檻（過濾快取結果）",
            min_value=0.10,
            max_value=0.90,
            value=float(result.conf_threshold),
            step=0.05,
            key="cv_instance_display_threshold",
        )
        alpha = st.slider(
            "遮罩透明度",
            min_value=0.20,
            max_value=0.80,
            value=float(st.session_state.get("cv_instance_alpha_saved", 0.45)),
            step=0.05,
            key="cv_instance_interpret_alpha",
        )
        filtered = filter_instances(result.items, conf_threshold=display_threshold)
        st.caption(
            f"原始實例 {result.raw_count} 個；"
            f"門檻 {display_threshold:.2f} 後顯示 {len(filtered)} 個。"
        )
        if filtered:
            filtered_overlay = build_color_overlay(image.shape[:2], filtered)
            overlay = blend_overlay(image, filtered_overlay, alpha=alpha)
            st.image(overlay, caption="疊加圖", width="stretch")
            compare_cols = st.columns(2, gap="medium")
            compare_cols[0].image(image, caption="原圖", width="stretch")
            compare_cols[1].image(
                filtered_overlay,
                caption="純色塊 mask",
                width="stretch",
            )

            options = [
                f"{item.rank}. {item.label} ({item.confidence:.0%})"
                for item in filtered
            ]
            selected_option = st.selectbox(
                "單實例檢視",
                options=options,
                key="cv_instance_select",
            )
            selected_index = options.index(selected_option)
            selected_item = filtered[selected_index]
            binary_mask = isolate_instance_mask(selected_item.mask)
            highlighted = highlight_instance(image, selected_item.mask, selected_item.color)
            crop = crop_instance(image, selected_item)
            mask_rgb = np.stack([binary_mask * 255] * 3, axis=-1)
            sub_cols = st.columns(3)
            sub_cols[0].image(mask_rgb, caption=f"{selected_item.label} binary mask")
            sub_cols[1].image(
                highlighted,
                caption=f"{selected_item.label} highlighted",
                width="stretch",
            )
            sub_cols[2].image(
                crop,
                caption=f"{selected_item.label} bbox crop",
                width="stretch",
            )
            st.caption(
                f"{selected_item.label} 覆蓋 {selected_item.area:,} 像素 "
                f"（{selected_item.coverage:.1%}）"
            )
            st.dataframe(_instance_dataframe(filtered), width="stretch", hide_index=True)
        else:
            st.warning("此門檻下沒有實例。請降低顯示門檻。")
        with st.expander("mask area 與 bbox 的差異", expanded=False):
            st.write(
                "Bounding box 是矩形外框；"
                "instance mask 描繪實際輪廓，"
                "能更精確描述不規則形狀的物件。"
            )


def _instance_dataframe(items: tuple[InstanceItem, ...]):
    pd = importlib.import_module("pandas")
    return pd.DataFrame(
        {
            "rank": [item.rank for item in items],
            "label": [item.label for item in items],
            "confidence": [round(item.confidence, 4) for item in items],
            "area": [item.area for item in items],
            "coverage": [round(item.coverage, 4) for item in items],
            "x1": [item.x1 for item in items],
            "y1": [item.y1 for item in items],
            "x2": [item.x2 for item in items],
            "y2": [item.y2 for item in items],
        }
    )


def _update_agent_context(image: np.ndarray, result: InstanceResult) -> None:
    lines = [
        f"目前頁面：{PAGE_TITLE}。",
        f"影像尺寸：{image.shape[1]}×{image.shape[0]}。",
        f"模型：{result.model_name} · 門檻 {result.conf_threshold:.2f}",
        format_instance_summary(result.items),
    ]
    if result.items:
        top = ", ".join(
            f"{item.label} {item.confidence:.1%}" for item in result.items[:3]
        )
        lines.append(f"Top instances：{top}")
    st.session_state[CONTEXT_KEY] = "\n".join(lines)
