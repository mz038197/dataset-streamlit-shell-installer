from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

SHELL_ROOT = Path(__file__).parent
PROJECT_ROOT = SHELL_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = SHELL_ROOT / "data"
SESSION_DIR = SHELL_ROOT / "sessions"
CHAT_IMAGE_DIR = SHELL_ROOT / "uploads" / "chat_images"
AGENT_ACTIVATION_MARKER_PATH = SHELL_ROOT / ".agent_core_activated"
ORIGINAL_DATASET_PATH = DATA_DIR / "original.csv"
WORKING_DATASET_PATH = DATA_DIR / "working.csv"
READY_DATASET_PATH = DATA_DIR / "ready.csv"
CLEANING_LOG_PATH = DATA_DIR / "cleaning_log.jsonl"
MAX_CHAT_IMAGE_BYTES = 5 * 1024 * 1024


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def inject_style() -> None:
    st.markdown(
        """
<style>
    .block-container { padding-top: 2rem; }
    .data-card {
        border: 1px solid rgba(250, 250, 250, 0.12);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(255, 255, 255, 0.035);
    }
    .data-muted { color: rgba(250, 250, 250, 0.65); }
    .data-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        background: rgba(90, 160, 255, 0.16);
        border: 1px solid rgba(90, 160, 255, 0.24);
        font-size: 0.8rem;
    }
    .data-agent-title-spacer {
        height: 0.75rem;
    }
    .data-agent-title-text {
        font-size: 1.25rem;
        font-weight: 800;
        line-height: 1.5;
        margin-bottom: 0.55rem;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def _ensure_session_dir() -> None:
    SESSION_DIR.mkdir(exist_ok=True)


def _ensure_chat_image_dir() -> None:
    CHAT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_agent_class(project_root: Path = PROJECT_ROOT) -> tuple[type[Any] | None, str | None]:
    agent_core_path = project_root / "agent_core.py"
    if not agent_core_path.exists():
        return None, "請確認 agent_core.py 已放在專案根目錄。"

    module_name = f"_dataset_shell_agent_core_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, agent_core_path)
    if spec is None or spec.loader is None:
        return None, "已找到 agent_core.py，但無法載入這個檔案。"

    inserted_path = False
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
        inserted_path = True

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        return None, f"已找到 agent_core.py，但無法匯入 Agent：{exc}"
    finally:
        sys.modules.pop(module_name, None)
        if inserted_path:
            with contextlib.suppress(ValueError):
                sys.path.remove(project_root_text)

    agent_class = getattr(module, "Agent", None)
    if agent_class is None:
        return None, "已找到 agent_core.py，但檔案內沒有 Agent 類別。"
    return agent_class, None


def _clear_agent_cache() -> None:
    st.session_state.pop("data_agent", None)
    st.session_state.pop("data_agent_session_path", None)
    st.session_state["data_agent_core_connected"] = False


def _write_activation_marker() -> None:
    AGENT_ACTIVATION_MARKER_PATH.write_text(
        datetime.now().isoformat(timespec="seconds"),
        encoding="utf-8",
    )


def _remove_activation_marker() -> None:
    if AGENT_ACTIVATION_MARKER_PATH.exists():
        AGENT_ACTIVATION_MARKER_PATH.unlink()


def _save_uploaded_chat_image(uploaded_file) -> tuple[str | None, str | None]:
    if uploaded_file is None:
        return None, None

    data = uploaded_file.getvalue()
    if len(data) > MAX_CHAT_IMAGE_BYTES:
        return None, "圖片超過 5 MB，請先壓縮後再上傳。"

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None, "只支援 PNG、JPG、JPEG、WEBP 圖片。"

    _ensure_chat_image_dir()
    filename = f"chat_image_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}{suffix}"
    target = CHAT_IMAGE_DIR / filename
    target.write_bytes(data)
    return target.relative_to(PROJECT_ROOT).as_posix(), None


def save_dataset(df: pd.DataFrame, *, working: bool = False) -> None:
    _ensure_data_dir()
    target = WORKING_DATASET_PATH if working else ORIGINAL_DATASET_PATH
    df.to_csv(target, index=False)
    if not working:
        st.session_state["dataset_df"] = df
        st.session_state.pop("working_dataset_df", None)
        st.session_state.pop("selected_columns", None)
        st.session_state.pop("filter_columns", None)
        st.session_state.pop("filter_signature", None)


def refresh_working_dataset_cache() -> None:
    st.session_state.pop("working_dataset_df", None)


def refresh_ready_dataset_cache() -> None:
    st.session_state.pop("ready_dataset_df", None)


def load_dataset() -> pd.DataFrame | None:
    if "dataset_df" in st.session_state:
        return st.session_state["dataset_df"]
    if ORIGINAL_DATASET_PATH.exists():
        df = pd.read_csv(ORIGINAL_DATASET_PATH)
        st.session_state["dataset_df"] = df
        return df
    return None


def load_working_dataset() -> pd.DataFrame | None:
    if WORKING_DATASET_PATH.exists():
        df = pd.read_csv(WORKING_DATASET_PATH)
        st.session_state["working_dataset_df"] = df
        return df
    return load_dataset()


def load_ready_dataset() -> pd.DataFrame | None:
    if READY_DATASET_PATH.exists():
        df = pd.read_csv(READY_DATASET_PATH)
        st.session_state["ready_dataset_df"] = df
        return df
    return None


def reset_working_dataset() -> None:
    st.session_state.pop("working_dataset_df", None)
    if WORKING_DATASET_PATH.exists():
        WORKING_DATASET_PATH.unlink()
    reset_ready_dataset()
    reset_cleaning_log()


def reset_ready_dataset() -> None:
    refresh_ready_dataset_cache()
    if READY_DATASET_PATH.exists():
        READY_DATASET_PATH.unlink()


def reset_cleaning_log() -> None:
    if CLEANING_LOG_PATH.exists():
        CLEANING_LOG_PATH.unlink()


def initialize_working_dataset(df: pd.DataFrame) -> None:
    save_dataset(df, working=True)
    st.session_state["working_dataset_df"] = df


def reset_working_dataset_from_source() -> bool:
    source = load_dataset()
    if source is None:
        return False
    save_dataset(source, working=True)
    st.session_state["working_dataset_df"] = source
    reset_ready_dataset()
    append_cleaning_log(
        action="重置工作資料",
        columns=source.columns,
        rows=len(source),
        note="使用 original.csv 重建 working.csv。",
        actor="ui",
    )
    return True


def create_ready_dataset(df: pd.DataFrame) -> None:
    _ensure_data_dir()
    df.to_csv(READY_DATASET_PATH, index=False)
    st.session_state["ready_dataset_df"] = df


def append_cleaning_log(
    *,
    action: str,
    columns: Iterable[str] | None = None,
    rows: int | None = None,
    note: str = "",
    actor: str = "ui",
) -> None:
    _ensure_data_dir()
    column_list = [] if columns is None else [str(column) for column in columns]
    normalized_actor = actor if actor in {"agent", "ui"} else "ui"
    entry = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "actor": normalized_actor,
        "action": action,
        "columns": column_list,
        "rows": rows,
        "note": note,
    }
    with CLEANING_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_cleaning_log(limit: int = 8) -> list[dict[str, object]]:
    if not CLEANING_LOG_PATH.exists():
        return []
    entries: list[dict[str, object]] = []
    with CLEANING_LOG_PATH.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries[-limit:][::-1]


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def dataset_context(df: pd.DataFrame | None) -> str:
    if df is None:
        return "目前尚未載入 CSV 資料。"

    columns = ", ".join(str(c) for c in df.columns)
    source = _display_path(ORIGINAL_DATASET_PATH)
    working = _display_path(WORKING_DATASET_PATH)
    ready = _display_path(READY_DATASET_PATH)
    cleaning_log = _display_path(CLEANING_LOG_PATH)
    return (
        "目前 Streamlit 畫面使用一份通用 CSV 資料集。"
        f"Original 原始資料路徑：{source}，只作為重置來源，請勿覆蓋。"
        f"Working 工作資料路徑：{working}。上傳資料時系統會先建立一份和原始資料相同的工作副本。"
        f"Ready 分析就緒資料路徑：{ready}，由工作資料凍結後供後續學習與分析使用。"
        f"資料共有 {len(df)} 筆、{len(df.columns)} 欄。欄位：{columns}。"
        "回答資料問題時，請使用你的 read_file 或 exec 工具實際讀取 CSV 後再回答。"
        f"如果使用者要求補值、清理資料、計算欄位或新增欄位，請預設讀取並更新 {working}，不要覆蓋 {source}。"
        "如果需要撰寫 Python 腳本來整理或檢查資料，請只建立在 dataset_streamlit_shell/scripts/ 底下，"
        "不要在專案根目錄建立臨時 Python 腳本。"
        f"修改 Working 工作資料後，請追加一筆 JSONL 紀錄到 {cleaning_log}，每行必須是單一 JSON 物件，"
        "欄位固定為 created_at、actor、action、columns、rows、note。"
        "actor 必須是 agent；action 使用簡短 snake_case；columns 是受影響欄位陣列；"
        "rows 是受影響筆數；note 用繁體中文一句話摘要修改內容。"
        "範例："
        '{"created_at":"2026-05-29T12:05:41","actor":"agent",'
        '"action":"fill_missing_age","columns":["Age"],"rows":177,'
        '"note":"以中位數補齊 Age 欄位的空值。"}'
    )


def metric_value(df: pd.DataFrame, kind: str) -> str:
    if kind == "missing":
        missing = int(df.isna().sum().sum())
        return f"{missing:,}"
    if kind == "numeric":
        return f"{len(df.select_dtypes(include='number').columns):,}"
    if kind == "text":
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        return f"{len(text_cols):,}"
    return "N/A"


def render_dataset_metrics(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("資料列數", f"{len(df):,}")
    c2.metric("欄位數", f"{len(df.columns):,}")
    c3.metric("缺失儲存格", metric_value(df, "missing"))
    c4.metric("數值欄位", metric_value(df, "numeric"))


def prepare_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for column in display.select_dtypes(include=["object", "category"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    return display


def render_column_pills(columns: Iterable[str]) -> None:
    pills = " ".join(f'<span class="data-pill">{column}</span>' for column in columns)
    st.markdown(pills, unsafe_allow_html=True)


def _new_session_path() -> Path:
    _ensure_session_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shortid = uuid.uuid4().hex[:6]
    path = SESSION_DIR / f"session_{stamp}_{shortid}.jsonl"
    path.touch(exist_ok=False)
    return path


def _session_sort_time(path: Path) -> datetime:
    created_at: str | None = None
    try:
        with path.open(encoding="utf-8") as f:
            first = f.readline().strip()
        if first:
            obj = json.loads(first)
            if isinstance(obj, dict):
                created_at = obj.get("created_at")
    except (OSError, json.JSONDecodeError):
        created_at = None

    if created_at:
        try:
            return datetime.fromisoformat(created_at)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def _session_label(path: Path) -> str:
    ts = _session_sort_time(path)
    shortid = path.stem.split("_")[-1]
    return f"{ts:%H:%M:%S} · 本機 · {shortid}"


def _list_sessions() -> list[Path]:
    _ensure_session_dir()
    return sorted(
        SESSION_DIR.glob("session_*.jsonl"),
        key=_session_sort_time,
        reverse=True,
    )


def _extract_display_user_text(text: str) -> str:
    marker = "\n\n學生問題："
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text


def _load_session_history(path: Path) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    if not path.exists():
        return history

    try:
        with path.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict) or obj.get("_type") == "metadata":
                    continue

                role = obj.get("role")
                content = str(obj.get("content", "") or "").strip()
                if role == "user" and content:
                    history.append(("user", _extract_display_user_text(content)))
                elif role == "assistant" and content:
                    history.append(("assistant", content))
    except OSError:
        return history
    return history


def _set_current_session(path: Path) -> None:
    session_path = str(path.relative_to(PROJECT_ROOT))
    st.session_state["session_path"] = session_path
    st.session_state["data_chat_history"] = _load_session_history(path)
    st.session_state.pop("data_agent", None)
    st.session_state.pop("data_agent_session_path", None)


def _reset_session_picker_widget() -> None:
    st.session_state["session_picker_version"] = (
        st.session_state.get("session_picker_version", 0) + 1
    )


def _create_agent_for_session(session_path: str) -> Any:
    agent_class, error = load_agent_class()
    if error is not None or agent_class is None:
        raise RuntimeError(error or "無法載入 Agent Core。")
    return agent_class.from_env(session_path=session_path)


def _get_agent_for_session(session_path: str) -> Any:
    if (
        "data_agent" not in st.session_state
        or st.session_state.get("data_agent_session_path") != session_path
    ):
        st.session_state["data_agent"] = _create_agent_for_session(session_path)
        st.session_state["data_agent_session_path"] = session_path
        st.session_state["data_agent_core_connected"] = True
    return st.session_state["data_agent"]


def _activate_agent_core(session_path: str) -> tuple[bool, str]:
    _clear_agent_cache()
    try:
        agent = _create_agent_for_session(session_path)
    except RuntimeError as exc:
        _remove_activation_marker()
        return False, str(exc)
    except Exception as exc:
        _remove_activation_marker()
        return False, f"Agent Core 啟用失敗：{exc}"

    st.session_state["data_agent"] = agent
    st.session_state["data_agent_session_path"] = session_path
    st.session_state["data_agent_core_connected"] = True
    _write_activation_marker()
    return True, "Agent Core 已連接。"


def _restore_agent_core_if_possible(session_path: str) -> tuple[bool, str | None]:
    if st.session_state.get("data_agent_core_connected"):
        return True, None
    if not AGENT_ACTIVATION_MARKER_PATH.exists():
        return False, None
    ok, message = _activate_agent_core(session_path)
    if ok:
        return True, None
    return False, message


def render_chat_panel(extra_context: str = "") -> None:
    st.markdown('<div class="data-agent-title-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="data-agent-title-text">資料 Agent</div>',
        unsafe_allow_html=True,
    )

    df = load_working_dataset()
    if df is None:
        st.caption("先在「資料上傳與預覽」頁上傳 CSV，右側 Agent 才能分析同一份資料。")
        st.chat_input("詢問 Agent...", disabled=True, key="data_chat_disabled")
        return

    sessions = _list_sessions()
    if "session_path" not in st.session_state and sessions:
        _set_current_session(sessions[0])
    current_session = st.session_state.get("session_path")

    if "data_chat_history" not in st.session_state:
        st.session_state["data_chat_history"] = [
            (
                "assistant",
                "請先按「啟用資料 Agent」。啟用後，我可以協助分析目前上傳的 CSV。",
            )
        ]

    labels = {str(path.relative_to(PROJECT_ROOT)): _session_label(path) for path in sessions}
    ids = list(labels)
    if current_session and current_session not in labels:
        ids.insert(0, current_session)
        labels[current_session] = "剛剛 · 目前對話"

    picker_key = f"session_picker_{st.session_state.get('session_picker_version', 0)}"

    def _on_pick() -> None:
        picked = PROJECT_ROOT / st.session_state[picker_key]
        _set_current_session(picked)

    selected_index = ids.index(current_session) if current_session in ids else 0

    pick_col, new_col, del_col = st.columns([6, 1, 1])
    pick_col.selectbox(
        "對話紀錄",
        ids,
        index=selected_index,
        format_func=lambda value: labels.get(value, value),
        disabled=not ids,
        label_visibility="collapsed",
        key=picker_key,
        on_change=_on_pick,
    )
    if new_col.button("", icon=":material/add:", help="新增對話", use_container_width=True):
        _set_current_session(_new_session_path())
        _reset_session_picker_widget()
        st.rerun()
    if del_col.button(
        "",
        icon=":material/delete:",
        help="刪除對話",
        use_container_width=True,
        disabled=not current_session,
    ):
        if current_session:
            target = PROJECT_ROOT / current_session
            if target.exists():
                target.unlink()
            st.session_state.pop("session_path", None)
            st.session_state.pop("data_chat_history", None)
            st.session_state.pop("data_agent", None)
            st.session_state.pop("data_agent_session_path", None)
            remaining = _list_sessions()
            if remaining:
                _set_current_session(remaining[0])
            _reset_session_picker_widget()
            st.rerun()

    current_session = st.session_state.get("session_path")
    if not current_session:
        st.caption("尚無對話紀錄，請按 **+** 新增對話。")
        st.chat_input("詢問...", disabled=True, key="data_chat_no_session")
        return

    restored, restore_error = _restore_agent_core_if_possible(current_session)
    connected = bool(st.session_state.get("data_agent_core_connected")) or restored
    status_text = ":green[●] Agent Core：已連接" if connected else ":red[●] Agent Core：未啟用"
    st.markdown(f"**{status_text}**")
    if not connected:
        st.caption("請按下方按鈕，將你的 WG-22 Agent Core 連接到 Data Shell。")
        if restore_error:
            st.warning(restore_error)
        if st.button("啟用資料 Agent", type="primary", use_container_width=True):
            ok, message = _activate_agent_core(current_session)
            if ok:
                st.success("Agent Core 已連接。你可以開始詢問資料 Agent。")
                st.rerun()
            else:
                st.error(message)
        st.code("請介紹你自己，並說明你會如何協助我整理 CSV 資料。", language="text")
        st.chat_input("請先啟用資料 Agent...", disabled=True, key="data_chat_not_activated")
        return

    with st.expander("技術資訊", expanded=False):
        st.caption(f"對話紀錄檔：`{current_session}`")
        st.caption(f"Working 工作資料檔：`{_display_path(WORKING_DATASET_PATH)}`")

    uploaded_image = st.file_uploader(
        "附加圖片（選填）",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"data_chat_image_{current_session}",
        help="圖片只會送給下一則訊息；支援 PNG/JPG/WEBP，大小上限 5 MB。",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="下一則訊息會附上這張圖片", use_container_width=True)

    try:
        agent = _get_agent_for_session(current_session)
    except RuntimeError as exc:
        st.error(str(exc))
        _clear_agent_cache()
        _remove_activation_marker()
        st.chat_input("詢問 Agent...", disabled=True, key="data_chat_no_key")
        return
    except Exception as exc:
        st.error(f"Agent Core 連線失敗：`{exc}`")
        _clear_agent_cache()
        _remove_activation_marker()
        st.chat_input("詢問 Agent...", disabled=True, key="data_chat_connect_failed")
        return

    chat = st.container(height=460, border=False)
    with chat:
        for role, text in st.session_state["data_chat_history"]:
            with st.chat_message(role):
                st.markdown(text)

    if user_text := st.chat_input("詢問資料 Agent...", key="data_chat"):
        image_path, image_error = _save_uploaded_chat_image(uploaded_image)
        display_user_text = user_text
        if image_error:
            st.warning(image_error)
        elif image_path:
            display_user_text = f"{user_text}\n\n（已附圖：{image_path}）"

        st.session_state["data_chat_history"].append(("user", display_user_text))
        context = dataset_context(df)
        if extra_context.strip():
            context = f"{context}\n\n【目前頁面狀態】{extra_context.strip()}"
        prompt = f"{context}\n\n學生問題：{user_text}"

        with chat:
            with st.chat_message("user"):
                st.markdown(user_text)
                if uploaded_image is not None and image_path:
                    st.image(uploaded_image, caption="已附圖", use_container_width=True)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                answer_parts: list[str] = []

                def on_token(token: str) -> None:
                    answer_parts.append(token)
                    placeholder.markdown("".join(answer_parts))

                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        final_text = agent.chat(
                            prompt,
                            image_path=image_path,
                            on_token=on_token,
                        )
                except Exception as exc:  # keep classroom UI alive during agent debugging
                    final_text = f"Agent 執行時發生錯誤：`{exc}`"
                    placeholder.error(final_text)

        answer = "".join(answer_parts).strip() or final_text
        st.session_state["data_chat_history"].append(("assistant", answer))
