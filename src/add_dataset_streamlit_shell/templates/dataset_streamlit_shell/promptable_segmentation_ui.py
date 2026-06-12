from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import streamlit as st

from dataset_streamlit_shell.cv.image_io import (
    SamDemoSpec,
    EXAMPLES_DIR,
    SAM3_MODELS_DIR,
    download_sample_data,
    download_sam3_weights,
    load_image_bytes,
    load_image_path,
    pil_to_rgb_array,
    sam3_weights_path,
    sam3_weights_ready,
    sam_demo_specs,
    sam_examples_ready,
)
from dataset_streamlit_shell.cv.promptable_segmentation import (
    DEFAULT_CONF,
    DEFAULT_MODEL,
    PromptableResult,
    TextPromptItem,
    crop_prompt_item,
    draw_promptable_results,
    filter_promptable_items,
    format_promptable_summary,
    highlight_prompt_item,
    load_sam3_predictor,
    parse_text_prompts,
    predict_text_masks,
)
from dataset_streamlit_shell.cv_layout import render_cv_tabbed_page

PAGE_TITLE = "提示式分割（Promptable Segmentation / SAM）"
CONTEXT_KEY = "promptable_segmentation_agent_context"
RESULT_KEY = "promptable_segmentation_last_result"
IMAGE_KEY = "promptable_segmentation_last_image"
PROMPTS_KEY = "promptable_segmentation_last_prompts"


@st.cache_resource(show_spinner="載入 SAM 3（sam3.pt）…")
def _cached_sam3_predictor(weights_path: str, conf_threshold: float):
    return load_sam3_predictor(
        conf_threshold=conf_threshold,
        weights_path=Path(weights_path),
    )


def render_promptable_segmentation_page() -> None:
    render_cv_tabbed_page(
        page_title=PAGE_TITLE,
        context_key=CONTEXT_KEY,
        tab_labels=['概念說明', '分割推論', '結果解讀'],
        tab_renderers=[_render_concept_tab, _render_inference_tab, _render_interpret_tab],
    )


def _render_download_panel() -> bool:
    if sam_examples_ready():
        return True
    st.info("首次使用請先下載教學用示範圖（需網路連線）。")
    if st.button("下載範例資料", key="cv_sam_download_samples"):
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
            st.warning("你仍可使用「上傳影像」進行提示式分割。")
    return sam_examples_ready()


def _render_weights_panel() -> bool:
    weights = sam3_weights_path()
    if sam3_weights_ready() and weights is not None:
        st.success(f"SAM 3 權重已就緒：`{weights}`")
        return True

    st.info("首次使用請下載 SAM 3 權重（約 3.4 GB，需穩定網路，請勿關閉分頁）。")
    if st.button("下載 SAM 3 權重", key="cv_sam_download_weights"):
        progress = st.progress(0.0, text="準備下載…")
        status = st.empty()

        def _callback(message: str, value: float) -> None:
            progress.progress(value, text=message)
            status.caption(message)

        try:
            download_sam3_weights(progress_callback=_callback)
            st.success("SAM 3 權重已下載完成，可以開始分割。")
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - surface download issues in UI
            st.error(f"下載失敗：{exc}")
            st.warning(
                f"請手動將 `{DEFAULT_MODEL}` 放到 `{SAM3_MODELS_DIR / DEFAULT_MODEL}`。"
            )
    return sam3_weights_ready()


def _resolve_image(
    *,
    source_mode: str,
    uploaded_file,
    selected_example: str | None,
) -> tuple[np.ndarray | None, str | None]:
    if source_mode == "上傳影像":
        if uploaded_file is None:
            return None, None
        return pil_to_rgb_array(load_image_bytes(uploaded_file.getvalue())), None
    if not selected_example:
        return None, None
    path = EXAMPLES_DIR / selected_example
    if not path.exists():
        return None, None
    return pil_to_rgb_array(load_image_path(path)), str(path)


def _render_concept_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("輸入文字描述，SAM 3 找出並分割符合概念的區域；此頁是電腦視覺路徑的第五站。")
    st.markdown("##### 什麼是提示式分割？")
    st.write(
        "不需固定類別表或重新訓練，只要輸入英文提示詞（如 `dog`、`person`），"
        "模型就會在影像中找出符合描述的區域並輸出 mask 與 bbox。"
    )
    st.info(
        "影像分類：整張圖 → 一個標籤\n\n"
        "物件偵測：固定 COCO 類別 + bbox\n\n"
        "語意／實例分割：自動分割全部物件\n\n"
        "提示式分割：你說要找什麼，模型就分什麼"
    )
    cols = st.columns(4)
    steps = ["輸入影像", "輸入文字提示", "SAM 3 概念匹配", "mask + bbox"]
    for column, step in zip(cols, steps, strict=True):
        column.markdown(f"**{step}**")
    with st.expander("簡短詞 vs 描述句", expanded=False):
        st.write(
            "簡短名詞（`dog`、`cup`）適合入門；"
            "較長的描述句（如 `a large bed on the right`）在複雜場景可能更精準，"
            "但 CPU 推論會更慢，建議先從 1～3 個提示詞開始。"
        )
    with st.expander("SAM 3 與 SAM 2 點擊提示的差異", expanded=False):
        st.write(
            "SAM 2 常用手動點擊或框選；"
            "SAM 3 支援 **Promptable Concept Segmentation**，"
            "可直接用文字描述概念，一次找出多個符合的區域。"
        )
    with st.expander("CPU 推論與權重", expanded=False):
        st.write(
            f"本頁使用 `{DEFAULT_MODEL}`。"
            "首次推論可能需數十秒至數分鐘（CPU）。"
            f"權重請放在 `{SAM3_MODELS_DIR / DEFAULT_MODEL}`。"
        )
    st.caption(f"本頁使用 SAM 3（{DEFAULT_MODEL}）。")


def _render_inference_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("輸入英文文字提示詞，SAM 3 產生符合概念的 mask 與 bbox。")
    weights_ready = _render_weights_panel()
    ready = _render_download_panel()
    source_mode = st.radio(
        "資料來源",
        ["內建範例圖片", "上傳影像"],
        horizontal=True,
        key="cv_sam_source_mode",
    )
    uploaded = None
    selected_example = None
    selected_spec: SamDemoSpec | None = None
    if source_mode == "內建範例圖片":
        if ready:
            specs = sam_demo_specs()
            selected_example = st.selectbox(
                "選擇示範圖",
                options=[spec.filename for spec in specs],
                format_func=lambda name: next(
                    spec.hint for spec in specs if spec.filename == name
                ),
                key="cv_sam_example",
            )
            selected_spec = next(
                spec for spec in specs if spec.filename == selected_example
            )
        else:
            st.warning("請先下載範例資料，或改用上傳影像。")
    else:
        uploaded = st.file_uploader(
            "上傳影像",
            type=["jpg", "jpeg", "png", "webp"],
            key="cv_sam_upload",
        )

    image, source_path = _resolve_image(
        source_mode=source_mode,
        uploaded_file=uploaded,
        selected_example=selected_example,
    )
    if image is not None:
        st.image(image, caption=f"{image.shape[1]}×{image.shape[0]}", use_container_width=True)

    if "cv_sam_prompts_input" not in st.session_state:
        st.session_state["cv_sam_prompts_input"] = (
            selected_spec.suggested_prompts if selected_spec else "dog"
        )
    if selected_spec is not None and st.button("套用建議提示詞", key="cv_sam_apply_prompts"):
        st.session_state["cv_sam_prompts_input"] = selected_spec.suggested_prompts
        st.rerun()
    prompts_raw = st.text_area(
        "文字提示詞（一行一個，建議英文）",
        height=120,
        key="cv_sam_prompts_input",
    )

    st.caption("模型：SAM 3（sam3.pt）· CPU 推論請耐心等候")
    conf_threshold = st.slider(
        "信心門檻",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_CONF,
        step=0.05,
        key="cv_sam_conf_threshold",
    )
    prompts = parse_text_prompts(prompts_raw)
    run_disabled = image is None or not prompts or not weights_ready
    run = st.button("執行分割", key="cv_sam_run", disabled=run_disabled)
    if not prompts:
        st.warning("請至少輸入一個文字提示詞。")

    if run and image is not None and prompts and weights_ready:
        weights = sam3_weights_path()
        assert weights is not None
        with st.spinner("SAM 3 分割中…（CPU 可能較久，請稍候）"):
            predictor = _cached_sam3_predictor(str(weights), conf_threshold)
            result = predict_text_masks(
                image,
                prompts,
                predictor=predictor,
                conf_threshold=conf_threshold,
                source_path=Path(source_path) if source_path else None,
            )
        st.session_state[RESULT_KEY] = result
        st.session_state[IMAGE_KEY] = image
        st.session_state[PROMPTS_KEY] = prompts_raw
        _update_agent_context(image, result)

    result: PromptableResult | None = st.session_state.get(RESULT_KEY)
    cached_image = st.session_state.get(IMAGE_KEY)
    if result is not None and cached_image is not None:
        _render_promptable_results(cached_image, result)



def _render_promptable_results(image: np.ndarray, result: PromptableResult) -> None:
    items = result.items
    if items:
        st.image(
            result.annotated_image,
            caption="提示式分割結果（mask + bbox）",
            use_container_width=True,
        )
    else:
        st.image(image, caption="原圖（無分割結果）", use_container_width=True)
        st.warning("未找到符合提示詞的區域。試著換提示詞或降低信心門檻。")
        return

    st.markdown("##### 匹配清單")
    st.dataframe(_promptable_dataframe(items), use_container_width=True, hide_index=True)
    st.caption(format_promptable_summary(items))


def _render_interpret_tab() -> None:
    st.title(PAGE_TITLE)
    st.caption("調整顯示門檻、檢視單一匹配結果，不需重新推論。")
    result: PromptableResult | None = st.session_state.get(RESULT_KEY)
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
            key="cv_sam_display_threshold",
        )
        alpha = st.slider(
            "遮罩透明度",
            min_value=0.20,
            max_value=0.80,
            value=0.50,
            step=0.05,
            key="cv_sam_interpret_alpha",
        )
        filtered = filter_promptable_items(result.items, conf_threshold=display_threshold)
        st.caption(
            f"原始匹配 {result.raw_count} 個；"
            f"門檻 {display_threshold:.2f} 後顯示 {len(filtered)} 個。"
        )
        if filtered:
            annotated = draw_promptable_results(image, filtered, alpha=alpha)
            st.image(annotated, caption="提示式分割結果", use_container_width=True)
            st.image(image, caption="原圖", use_container_width=True)

            options = [
                f"{item.rank}. {item.prompt} ({item.confidence:.0%})"
                for item in filtered
            ]
            selected_option = st.selectbox(
                "單一匹配檢視",
                options=options,
                key="cv_sam_select",
            )
            selected_index = options.index(selected_option)
            selected_item = filtered[selected_index]
            mask_rgb = np.stack([selected_item.mask.astype(np.uint8) * 255] * 3, axis=-1)
            highlighted = highlight_prompt_item(image, selected_item)
            crop = crop_prompt_item(image, selected_item)
            sub_cols = st.columns(3)
            sub_cols[0].image(mask_rgb, caption=f"{selected_item.prompt} binary mask")
            sub_cols[1].image(
                highlighted,
                caption=f"{selected_item.prompt} highlighted",
                use_container_width=True,
            )
            sub_cols[2].image(
                crop,
                caption=f"{selected_item.prompt} bbox crop",
                use_container_width=True,
            )
            st.caption(
                f"提示詞「{selected_item.prompt}」覆蓋 {selected_item.area:,} 像素 "
                f"（{selected_item.coverage:.1%}）"
            )
            st.dataframe(_promptable_dataframe(filtered), use_container_width=True, hide_index=True)
        else:
            st.warning("此門檻下沒有匹配結果。請降低顯示門檻。")
        with st.expander("文字提示 vs 點擊提示", expanded=False):
            st.write(
                "SAM 2 風格需手動點選位置；"
                "SAM 3 可直接用文字描述要找的概念，"
                "適合 open-vocabulary 與快速標註流程。"
            )


def _promptable_dataframe(items: tuple[TextPromptItem, ...]):
    pd = importlib.import_module("pandas")
    return pd.DataFrame(
        {
            "rank": [item.rank for item in items],
            "prompt": [item.prompt for item in items],
            "confidence": [round(item.confidence, 4) for item in items],
            "area": [item.area for item in items],
            "coverage": [round(item.coverage, 4) for item in items],
            "x1": [item.x1 for item in items],
            "y1": [item.y1 for item in items],
            "x2": [item.x2 for item in items],
            "y2": [item.y2 for item in items],
        }
    )


def _update_agent_context(image: np.ndarray, result: PromptableResult) -> None:
    lines = [
        f"目前頁面：{PAGE_TITLE}。",
        f"影像尺寸：{image.shape[1]}×{image.shape[0]}。",
        f"模型：{result.model_name} · 門檻 {result.conf_threshold:.2f}",
        f"提示詞：{', '.join(result.prompts)}",
        format_promptable_summary(result.items),
    ]
    st.session_state[CONTEXT_KEY] = "\n".join(lines)
