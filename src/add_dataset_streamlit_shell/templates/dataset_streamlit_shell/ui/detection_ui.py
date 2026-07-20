from __future__ import annotations

import importlib

import numpy as np
import streamlit as st

from dataset_streamlit_shell.cv.detection import (
    DEFAULT_MODEL,
    DetectionItem,
    DetectionResult,
    crop_detection,
    draw_detections,
    filter_detections,
    format_detection_summary,
    load_yolo_model,
    predict_detections,
)
from dataset_streamlit_shell.cv.image_io import (
    EXAMPLES_DIR,
    detection_demo_specs,
    detection_examples_ready,
    load_image_bytes,
    load_image_path,
    pil_to_rgb_array,
)
from dataset_streamlit_shell.ui.cv_layout import render_cv_tabbed_page

PAGE_TITLE = "物件偵測（Object Detection）"
CONTEXT_KEY = "object_detection_agent_context"
RESULT_KEY = "object_detection_last_result"
IMAGE_KEY = "object_detection_last_image"


@st.cache_resource(show_spinner="載入 YOLOv8n（COCO 預訓練）…")
def _cached_yolo():
    return load_yolo_model()


def render_object_detection_page() -> None:
    render_cv_tabbed_page(
        page_title=PAGE_TITLE,
        context_key=CONTEXT_KEY,
        tab_labels=["概念說明", "偵測推論", "結果解讀"],
        tab_renderers=[_render_concept_tab, _render_inference_tab, _render_interpret_tab],
    )



def _render_download_panel() -> bool:
    if detection_examples_ready():
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
    st.caption("找出每個物件在哪裡、是什麼；此頁是電腦視覺路徑的第二站。")
    st.markdown("##### 什麼是物件偵測？")
    st.write(
        "模型讀入影像，輸出多個 bounding box，"
        "每個框附帶類別與信心度。它同時回答「在哪裡」與「是什麼」。"
    )
    st.info(
        "影像分類：整張圖 → 一個標籤\n\n"
        "物件偵測：多個 bbox + 類別 + 信心度\n\n"
        "語意分割：像素級類別\n\n"
        "實例分割：每個物件獨立 mask"
    )
    cols = st.columns(4)
    steps = ["輸入影像", "Backbone 特徵", "預測框 + 類別", "NMS 過濾"]
    for column, step in zip(cols, steps, strict=True):
        column.markdown(f"**{step}**")
    with st.expander("COCO 是什麼？", expanded=False):
        st.write(
            "COCO 是常用的物件偵測基準，約 80 類日常物件。"
            "YOLOv8n 在其上預訓練，可直接偵測 person、car、dog 等。"
        )
    with st.expander("Bounding box 座標", expanded=False):
        st.write(
            "每個框以 (x1, y1, x2, y2) 表示左上角與右下角像素座標。"
            "信心度越高，模型越確定該框內有該類別物件。"
        )
    with st.expander("與影像分類的差異", expanded=False):
        st.write(
            "分類頁對 street_scene 可能只給一個整圖標籤；"
            "偵測頁可同時框出 person、car、bicycle 等多個物件。"
        )
    st.caption(f"本頁使用 {DEFAULT_MODEL}（COCO 預訓練）。")


def _render_inference_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("使用 YOLOv8n 進行多物件 bounding box 偵測。")
    ready = _render_download_panel()
    source_mode = st.radio(
        "資料來源",
        ["內建範例圖片", "上傳影像"],
        horizontal=True,
        key="cv_detection_source_mode",
    )
    uploaded = None
    selected_example = None
    if source_mode == "內建範例圖片":
        if ready:
            specs = detection_demo_specs()
            selected_example = st.selectbox(
                "選擇示範圖",
                options=[spec.filename for spec in specs],
                format_func=lambda name: next(
                    spec.hint for spec in specs if spec.filename == name
                ),
                key="cv_detection_example",
            )
        else:
            st.warning("找不到內建範例圖，請改用上傳影像。")
    else:
        uploaded = st.file_uploader(
            "上傳影像",
            type=["jpg", "jpeg", "png", "webp"],
            key="cv_detection_upload",
        )

    image = _resolve_image(
        source_mode=source_mode,
        uploaded_file=uploaded,
        selected_example=selected_example,
    )
    if image is not None:
        st.image(image, caption=f"{image.shape[1]}×{image.shape[0]}", width="stretch")

    st.caption(f"模型：YOLOv8n（COCO 預訓練）")
    conf_threshold = st.slider(
        "信心門檻",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="cv_detection_conf_threshold",
    )
    run = st.button("執行偵測", key="cv_detection_run", disabled=image is None)

    if run and image is not None:
        with st.spinner("偵測中…"):
            result = predict_detections(
                image,
                conf_threshold=conf_threshold,
                model=_cached_yolo(),
            )
        st.session_state[RESULT_KEY] = result
        st.session_state[IMAGE_KEY] = image
        _update_agent_context(image, result)

    result: DetectionResult | None = st.session_state.get(RESULT_KEY)
    cached_image = st.session_state.get(IMAGE_KEY)
    if result is not None and cached_image is not None and image is not None:
        _render_detection_results(cached_image, result)


def _render_detection_results(image: np.ndarray, result: DetectionResult) -> None:
    items = result.items
    if items:
        annotated = draw_detections(image, items)
        st.image(annotated, caption="偵測結果", width="stretch")
    else:
        st.image(image, caption="原圖（無偵測結果）", width="stretch")
        st.warning("未偵測到物件。試著降低信心門檻後重新執行偵測。")
        return

    st.markdown("##### 偵測清單")
    st.dataframe(_detection_dataframe(items), width="stretch", hide_index=True)
    st.caption(format_detection_summary(items))


def _render_interpret_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("調整顯示門檻、檢視單框裁切，不需重新推論。")
    result: DetectionResult | None = st.session_state.get(RESULT_KEY)
    image = st.session_state.get(IMAGE_KEY)
    if result is None or image is None:
        st.info("請先到「偵測推論」Tab 執行偵測。")
    else:
        display_threshold = st.slider(
            "顯示門檻（過濾快取結果）",
            min_value=0.10,
            max_value=0.90,
            value=float(result.conf_threshold),
            step=0.05,
            key="cv_detection_display_threshold",
        )
        filtered = filter_detections(result.items, conf_threshold=display_threshold)
        st.caption(
            f"原始偵測 {result.raw_count} 個；"
            f"門檻 {display_threshold:.2f} 後顯示 {len(filtered)} 個。"
        )
        if filtered:
            annotated = draw_detections(image, filtered)
            st.image(annotated, caption="過濾後結果", width="stretch")
            crop_options = [
                f"{item.rank}. {item.label} ({item.confidence:.0%})" for item in filtered
            ]
            selected_option = st.selectbox(
                "選擇框進行裁切預覽",
                options=crop_options,
                key="cv_detection_crop_label",
            )
            selected_index = crop_options.index(selected_option)
            selected_item = filtered[selected_index]
            crop = crop_detection(image, selected_item)
            st.image(
                crop,
                caption=f"{selected_item.label} · {selected_item.confidence:.1%}",
                width="stretch",
            )
            st.dataframe(_detection_dataframe(filtered), width="stretch", hide_index=True)
        else:
            st.warning("此門檻下沒有偵測框。請降低顯示門檻。")
        with st.expander("NMS 簡介", expanded=False):
            st.write(
                "Non-Maximum Suppression（NMS）會合併高度重疊的框，"
                "避免同一物件被重複標記。"
                "YOLO 推論時已套用；此處的門檻滑桿只過濾顯示，不重新跑模型。"
            )


def _detection_dataframe(items: tuple[DetectionItem, ...]):
    pd = importlib.import_module("pandas")
    return pd.DataFrame(
        {
            "rank": [item.rank for item in items],
            "label": [item.label for item in items],
            "confidence": [round(item.confidence, 4) for item in items],
            "x1": [item.x1 for item in items],
            "y1": [item.y1 for item in items],
            "x2": [item.x2 for item in items],
            "y2": [item.y2 for item in items],
        }
    )


def _update_agent_context(image: np.ndarray, result: DetectionResult) -> None:
    lines = [
        f"目前頁面：{PAGE_TITLE}。",
        f"影像尺寸：{image.shape[1]}×{image.shape[0]}。",
        f"模型：{result.model_name} · 門檻 {result.conf_threshold:.2f}",
        format_detection_summary(result.items),
    ]
    if result.items:
        top = ", ".join(
            f"{item.label} {item.confidence:.1%}" for item in result.items[:3]
        )
        lines.append(f"Top detections：{top}")
    st.session_state[CONTEXT_KEY] = "\n".join(lines)
