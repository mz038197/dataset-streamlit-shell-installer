from __future__ import annotations

import contextlib
import inspect
import io
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai_tts import Settings
from openai_tts.settings import MAX_TTS_SPEED, MIN_TTS_SPEED

SHELL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SHELL_ROOT.parent
load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.agent_loader import (  # noqa: E402
    create_agent_for_session,
    load_create_agent,
)
from dataset_streamlit_shell.ml.integration import clear_dual_table_copies  # noqa: E402

BRAND_LOGO_PATH = SHELL_ROOT / "assets" / "brand-logo.png"
WORKSPACE_DIR = SHELL_ROOT / "workspace"
SESSION_DIR = PROJECT_ROOT / "sessions"
AGENT_ACTIVATION_MARKER_PATH = SHELL_ROOT / ".agent_activated"
ORIGINAL_DATASET_PATH = WORKSPACE_DIR / "original.csv"
WORKING_DATASET_PATH = WORKSPACE_DIR / "working.csv"
READY_DATASET_PATH = WORKSPACE_DIR / "ready.csv"
TRAIN_DATASET_PATH = WORKSPACE_DIR / "train.csv"
VAL_DATASET_PATH = WORKSPACE_DIR / "val.csv"
TEST_DATASET_PATH = WORKSPACE_DIR / "test.csv"
CLEANING_LOG_PATH = WORKSPACE_DIR / "cleaning_log.jsonl"
SPLIT_DATASET_PATHS = (TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH)
USER_SETTINGS_PATH = WORKSPACE_DIR / "user_settings.json"
_AGENT_SCOPE_KEYS: dict[str, dict[str, str]] = {
    "data": {
        "session_path": "session_path",
        "chat_history": "data_chat_history",
        "agent": "data_agent",
        "agent_session_path": "data_agent_session_path",
        "agent_factory_ref": "data_agent_factory_ref",
        "agent_connected": "data_agent_connected",
        "picker_version": "session_picker_version",
        "chat_just_replied": "data_chat_just_replied",
    },
    "challenge": {
        "session_path": "challenge_session_path",
        "chat_history": "challenge_chat_history",
        "agent": "challenge_agent",
        "agent_session_path": "challenge_agent_session_path",
        "agent_factory_ref": "challenge_agent_factory_ref",
        "agent_connected": "challenge_agent_connected",
        "picker_version": "challenge_session_picker_version",
        "chat_just_replied": "challenge_chat_just_replied",
    },
}


def _agent_keys(scope: str) -> dict[str, str]:
    try:
        return _AGENT_SCOPE_KEYS[scope]
    except KeyError as exc:
        raise ValueError(f"未知 Agent scope：{scope!r}") from exc
TTS_VOICE_OPTIONS = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
]


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def brand_page_icon() -> str:
    return str(BRAND_LOGO_PATH)


def inject_style() -> None:
    if BRAND_LOGO_PATH.is_file():
        st.logo(str(BRAND_LOGO_PATH), size="small", icon_image=str(BRAND_LOGO_PATH))
    st.markdown(
        """
<style>
    /* 4rem 讓開固定 header（Deploy／選單）；勿再用 0.5rem，標題會被裁切 */
    /* 左右 1rem：蓋掉 Streamlit wide 的 5rem，讓主教學欄貼近導覽側欄 */
    .block-container,
    .stMainBlockContainer,
    [data-testid="stMainBlockContainer"] {
        padding-top: 4rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* CSS 注入列不要佔導航列底下的垂直空間 */
    [data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] > style:only-child) {
        display: none !important;
    }
    [data-testid="stSidebarHeader"] {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 10px !important;
        overflow: visible !important;
    }
    /* stLogo 在 1.61 是 img；img 上的 ::after 不會畫出來 */
    [data-testid="stSidebarHeader"]::after {
        content: "VansCoding.AI";
        font-weight: 700;
        font-size: 14px;
        letter-spacing: -0.02em;
        white-space: nowrap;
        background: linear-gradient(135deg, #007070 0%, #009999 50%, #f8c000 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        order: 1;
    }
    [data-testid="stSidebarHeader"] img {
        box-sizing: border-box !important;
        width: 40px !important;
        height: 40px !important;
        max-height: 40px !important;
        padding: 3px;
        border-radius: 8px;
        background: #eef5f5;
        box-shadow: 0 0 0 1px rgba(0, 112, 112, 0.18);
        object-fit: contain;
        object-position: center;
        order: 0;
    }
    [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
        margin-left: auto !important;
        order: 2;
    }
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


def _ensure_workspace_dir() -> None:
    WORKSPACE_DIR.mkdir(exist_ok=True)


def _default_tts_preferences() -> dict[str, object]:
    env = Settings()
    return {
        "tts_enabled": False,
        "tts_voice": env.voice,
        "tts_instructions": env.instructions,
        "tts_speed": float(env.speed if env.speed is not None else 1.0),
    }


def _normalize_tts_preferences(
    raw: dict[str, object],
    defaults: dict[str, object],
) -> dict[str, object]:
    voice = str(raw.get("tts_voice", defaults["tts_voice"]))
    if voice not in TTS_VOICE_OPTIONS:
        voice = str(defaults["tts_voice"])

    try:
        speed = float(raw.get("tts_speed", defaults["tts_speed"]))
    except (TypeError, ValueError):
        speed = float(defaults["tts_speed"])
    speed = max(MIN_TTS_SPEED, min(MAX_TTS_SPEED, speed))

    return {
        "tts_enabled": bool(raw.get("tts_enabled", defaults["tts_enabled"])),
        "tts_voice": voice,
        "tts_instructions": str(raw.get("tts_instructions", defaults["tts_instructions"])),
        "tts_speed": speed,
    }


def _read_user_settings() -> tuple[dict[str, object], bool]:
    defaults = _default_tts_preferences()
    if not USER_SETTINGS_PATH.exists():
        return defaults, True

    try:
        with USER_SETTINGS_PATH.open(encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return defaults, True

    if not isinstance(raw, dict):
        return defaults, True

    normalized = _normalize_tts_preferences(raw, defaults)
    return normalized, raw != normalized


def _load_user_settings() -> dict[str, object]:
    prefs, _needs_repair = _read_user_settings()
    return prefs


def _save_user_settings(settings: dict[str, object]) -> str | None:
    payload = _normalize_tts_preferences(settings, _default_tts_preferences())
    try:
        _ensure_workspace_dir()
        USER_SETTINGS_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        return f"無法寫入語音設定檔：{exc}"
    return None


def _ensure_user_settings_file() -> str | None:
    prefs, needs_repair = _read_user_settings()
    if not needs_repair:
        return None
    return _save_user_settings(prefs)


def _should_reload_tts_for_page(last_page: str | None, page_name: str) -> bool:
    return last_page != page_name


def _sync_tts_preferences_for_page(page_name: str) -> str | None:
    settings_error = _ensure_user_settings_file()
    if settings_error is not None:
        return settings_error

    persist_error = _persist_tts_preferences_if_changed()
    if persist_error is not None:
        return persist_error

    _reload_tts_preferences_from_file()
    st.session_state["_data_tts_page_name"] = page_name
    return None


def _reload_tts_preferences_from_file() -> None:
    prefs = _load_user_settings()
    st.session_state["data_tts_enabled"] = prefs["tts_enabled"]
    st.session_state["data_tts_voice"] = prefs["tts_voice"]
    st.session_state["data_tts_instructions"] = prefs["tts_instructions"]
    st.session_state["data_tts_speed"] = prefs["tts_speed"]
    st.session_state["_data_user_settings_snapshot"] = dict(prefs)


def _persist_tts_preferences_if_changed() -> str | None:
    required_keys = {
        "data_tts_enabled",
        "data_tts_voice",
        "data_tts_instructions",
        "data_tts_speed",
    }
    if not required_keys.issubset(st.session_state):
        return None

    current = {
        "tts_enabled": bool(st.session_state.get("data_tts_enabled", False)),
        "tts_voice": str(st.session_state.get("data_tts_voice", "")),
        "tts_instructions": str(st.session_state.get("data_tts_instructions", "")),
        "tts_speed": float(st.session_state.get("data_tts_speed", 1.0)),
    }
    normalized = _normalize_tts_preferences(current, _default_tts_preferences())
    previous = st.session_state.get("_data_user_settings_snapshot")
    if previous is None:
        return None
    if previous == normalized:
        return None

    error = _save_user_settings(normalized)
    if error is not None:
        return error
    st.session_state["_data_user_settings_snapshot"] = dict(normalized)
    return None


def _ensure_session_dir() -> None:
    SESSION_DIR.mkdir(exist_ok=True)


def _clear_agent_cache(*, scope: str = "data") -> None:
    keys = _agent_keys(scope)
    st.session_state.pop(keys["agent"], None)
    st.session_state.pop(keys["agent_session_path"], None)
    st.session_state.pop(keys["agent_factory_ref"], None)
    st.session_state[keys["agent_connected"]] = False


def reset_agent_scope_session(*, scope: str) -> None:
    """清除指定 scope 的 Agent 快取與目前對話指標（不刪實體 session 檔）。"""
    keys = _agent_keys(scope)
    _clear_agent_cache(scope=scope)
    st.session_state.pop(keys["session_path"], None)
    st.session_state.pop(keys["chat_history"], None)


def _write_activation_marker() -> None:
    AGENT_ACTIVATION_MARKER_PATH.write_text(
        datetime.now().isoformat(timespec="seconds"),
        encoding="utf-8",
    )


def _remove_activation_marker() -> None:
    if AGENT_ACTIVATION_MARKER_PATH.exists():
        AGENT_ACTIVATION_MARKER_PATH.unlink()


def save_dataset(df: pd.DataFrame, *, working: bool = False) -> None:
    _ensure_workspace_dir()
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


def clear_split_datasets() -> None:
    for path in SPLIT_DATASET_PATHS:
        if path.exists():
            path.unlink()


def reset_ready_dataset() -> None:
    refresh_ready_dataset_cache()
    if READY_DATASET_PATH.exists():
        READY_DATASET_PATH.unlink()
    clear_split_datasets()


def working_dataset_file_exists() -> bool:
    return WORKING_DATASET_PATH.exists()


def clear_to_dual_start() -> None:
    """清除 working／ready／切分產物／original／雙表工作副本，回到雙表起點。"""
    clear_split_datasets()
    reset_working_dataset()
    st.session_state.pop("dataset_df", None)
    if ORIGINAL_DATASET_PATH.exists():
        ORIGINAL_DATASET_PATH.unlink()
    clear_dual_table_copies(WORKSPACE_DIR)


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
    _ensure_workspace_dir()
    clear_split_datasets()  # 重建 Ready 作廢既有切分產物
    df.to_csv(READY_DATASET_PATH, index=False)
    st.session_state["ready_dataset_df"] = df


def save_split_datasets(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    _ensure_workspace_dir()
    train.to_csv(TRAIN_DATASET_PATH, index=False)
    val.to_csv(VAL_DATASET_PATH, index=False)
    test.to_csv(TEST_DATASET_PATH, index=False)


def append_cleaning_log(
    *,
    action: str,
    columns: Iterable[str] | None = None,
    rows: int | None = None,
    note: str = "",
    actor: str = "ui",
) -> None:
    _ensure_workspace_dir()
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


def dataset_base_context() -> str:
    """穩定環境說明 → create_agent(host_context=...) → 學生 system（對齊 Studio）。"""
    source = _display_path(ORIGINAL_DATASET_PATH)
    working = _display_path(WORKING_DATASET_PATH)
    ready = _display_path(READY_DATASET_PATH)
    cleaning_log = _display_path(CLEANING_LOG_PATH)
    scripts = _display_path(SHELL_ROOT / "scripts")
    from dataset_streamlit_shell.ui.nn_form_state import nn_host_context_fragment

    nn_form = _display_path(WORKSPACE_DIR / "nn_form.json")
    nn_request = _display_path(WORKSPACE_DIR / "nn_train_request.json")
    nn_last = _display_path(WORKSPACE_DIR / "nn_last_run.json")
    copies = _display_path(WORKSPACE_DIR / "integration")
    return (
        "目前為 Dataset Streamlit Shell。"
        f"Original 原始資料路徑：{source}，只作為重置來源。"
        "合併完成前不存在；僅在資料整合時可與 Working 同批寫入，此外請勿覆蓋。"
        f"Working 工作資料路徑：{working}。"
        "資料整合成功時與 Original 一併寫入；Working 為之後整理用副本。"
        f"雙表工作副本目錄：{copies}（workspace/integration/passengers.csv、voyage.csv）。"
        "雙表起點進頁時已寫出這兩份；請直接改航程表副本的 passenger_id 為 PassengerId，不要另存新檔。"
        "內建雙表教材不得覆寫。"
        "合併前只准將航程表副本的 passenger_id 改名為 PassengerId；不要補值、刪欄或改乘客表。"
        "通過訓練前預測（資料整合）且鍵名已對齊後，才可依學生指定的 left 或 inner 合併"
        "（課堂主線建議 left），"
        f"同時寫入 {source} 與 {working}，並刪除雙表工作副本"
        "（可用 commit_dual_table_merge）。"
        "寫入後請告訴學生按「重新讀取工作資料」。"
        "未解鎖或未對齊鍵名時拒絕合併。"
        f"Ready 分析就緒資料路徑：{ready}，由工作資料凍結後供圖表探索、降維等分析頁使用；"
        "監督式與非監督式教學頁使用各頁內建範例資料。"
        "回答資料問題時，請使用你的 read_file 或 exec 工具實際讀取 CSV 後再回答。"
        f"如果使用者要求補值、清理資料、計算欄位或新增欄位，且 Working 已存在，請預設讀取並更新 {working}，不要覆蓋 {source}。"
        f"如果需要撰寫 Python 腳本來整理或檢查資料，請只建立在 {scripts} 底下，"
        "不要在專案根目錄建立臨時 Python 腳本。"
        f"修改 Working 工作資料後，請追加一筆 JSONL 紀錄到 {cleaning_log}，每行必須是單一 JSON 物件，"
        "欄位固定為 created_at、actor、action、columns、rows、note。"
        "actor 必須是 agent；action 使用簡短 snake_case；columns 是受影響欄位陣列；"
        "rows 是受影響筆數；note 用繁體中文一句話摘要修改內容。"
        "範例："
        '{"created_at":"2026-05-29T12:05:41","actor":"agent",'
        '"action":"fill_missing_age","columns":["Age"],"rows":177,'
        '"note":"以中位數補齊 Age 欄位的空值。"}'
        + nn_host_context_fragment(
            form_path=nn_form,
            request_path=nn_request,
            last_run_path=nn_last,
        )
    )


def dataset_page_snapshot(df: pd.DataFrame | None, extra_context: str = "") -> str:
    """當頁／當下快照 → 每輪 user（【目前頁面狀態】）。"""
    parts: list[str] = []
    if df is None:
        parts.append(
            "尚未建立工作資料。"
            "若詢問資料內容、清理、補值或欄位計算，請先提醒到「欄位與資料整合」查看雙表；"
            "合併前可請 Agent 對齊航程表鍵名，通過關卡後再請 Agent 合併，會同時寫入 Original 與 Working。"
            "一般概念問題可直接回答。"
        )
    else:
        columns = ", ".join(str(c) for c in df.columns)
        parts.append(f"工作資料目前有 {len(df)} 筆、{len(df.columns)} 欄。欄位：{columns}。")
    if extra_context.strip():
        parts.append(extra_context.strip())
    return "\n".join(parts)


def format_user_turn(user_text: str, *, extra_context: str = "") -> str:
    if extra_context.strip():
        return (
            f"【目前頁面狀態】\n{extra_context.strip()}\n\n"
            f"使用者問題：{user_text}"
        )
    return f"使用者問題：{user_text}"


def dataset_context(df: pd.DataFrame | None) -> str:
    """相容舊呼叫：host + 快照合併（新路徑請改用 dataset_base_context／snapshot）。"""
    return f"{dataset_base_context()}\n\n{dataset_page_snapshot(df)}"


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


def _session_relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _is_valid_session_relpath(value: str) -> bool:
    if not value or value.endswith(".jsonl") is False:
        return False
    if "sessions" not in Path(value).parts:
        return False
    return (PROJECT_ROOT / value).is_file()


def _resolve_session_relpath(value: str, labels: dict[str, str]) -> str | None:
    if _is_valid_session_relpath(value):
        return value

    for relpath, label in labels.items():
        if value == label:
            return relpath
    return None


def _build_session_picker_options(
    sessions: list[Path],
) -> tuple[list[str], dict[str, str]]:
    labels = {_session_relpath(path): _session_label(path) for path in sessions}
    return list(labels), labels


def _list_sessions() -> list[Path]:
    _ensure_session_dir()
    return sorted(
        SESSION_DIR.glob("session_*.jsonl"),
        key=_session_sort_time,
        reverse=True,
    )


def _extract_display_user_text(text: str) -> str:
    for marker in ("\n\n使用者問題：", "\n\n學生問題："):
        if marker in text:
            return text.rsplit(marker, 1)[-1].strip()
    return text


REASONING_ROUND_SEPARATOR = "\n\n---\n\n"
TOOL_RUN_PLACEHOLDER = "（執行工具中…）"


def _commit_reasoning_round(segments: list[str], current_parts: list[str]) -> None:
    text = "".join(current_parts).strip()
    if text:
        segments.append(text)
    current_parts.clear()


def _merged_reasoning_text(segments: list[str], current_parts: list[str]) -> str:
    parts = [segment for segment in segments if segment.strip()]
    current = "".join(current_parts).strip()
    if current:
        parts.append(current)
    return REASONING_ROUND_SEPARATOR.join(parts)


def _render_reasoning_expander(
    reasoning_slot: Any,
    text: str,
    *,
    expanded: bool,
    stream_ui: dict[str, Any],
) -> None:
    if not text.strip():
        return
    stream_ui["visible"] = True
    stream_ui["expanded"] = expanded
    with reasoning_slot.container():
        with st.expander("思考過程", expanded=expanded):
            if expanded:
                stream_ui["reasoning_ph"] = st.empty()
                stream_ui["reasoning_ph"].markdown(text)
            else:
                stream_ui["reasoning_ph"] = None
                st.markdown(text)


def _parse_history_entry(entry: tuple[str, ...]) -> tuple[str, str, str]:
    role = entry[0]
    text = entry[1] if len(entry) > 1 else ""
    reasoning = entry[2] if len(entry) > 2 else ""
    return role, text, reasoning


def _render_history_message(role: str, text: str, *, reasoning: str = "") -> None:
    with st.chat_message(role):
        if role == "assistant" and reasoning.strip():
            with st.expander("思考過程", expanded=False):
                st.markdown(reasoning)
        st.markdown(text)


def _accepts_chat_keyword(chat_fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(chat_fn)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _chat_callback_kwargs(
    chat_fn: Any,
    *,
    on_token: Any,
    on_reasoning: Any,
    on_stream_reset: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"image_path": None, "on_token": on_token}
    if _accepts_chat_keyword(chat_fn, "on_reasoning"):
        kwargs["on_reasoning"] = on_reasoning
    if _accepts_chat_keyword(chat_fn, "on_stream_reset"):
        kwargs["on_stream_reset"] = on_stream_reset
    return kwargs


def _load_session_history(path: Path) -> list[tuple[str, ...]]:
    history: list[tuple[str, ...]] = []
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
                    history.append(("assistant", content, ""))
    except OSError:
        return history
    return history


def _set_current_session(path: Path, *, scope: str = "data") -> None:
    keys = _agent_keys(scope)
    session_path = _session_relpath(path)
    st.session_state[keys["session_path"]] = session_path
    st.session_state[keys["chat_history"]] = _load_session_history(path)
    st.session_state.pop(keys["agent"], None)
    st.session_state.pop(keys["agent_session_path"], None)


def _ensure_valid_current_session(
    sessions: list[Path],
    *,
    scope: str = "data",
) -> str | None:
    keys = _agent_keys(scope)
    current_session = st.session_state.get(keys["session_path"])
    if current_session and _is_valid_session_relpath(current_session):
        return current_session

    st.session_state.pop(keys["session_path"], None)
    st.session_state.pop(keys["agent"], None)
    st.session_state.pop(keys["agent_session_path"], None)
    if sessions:
        _set_current_session(sessions[0], scope=scope)
        return st.session_state[keys["session_path"]]
    return None


def _reset_session_picker_widget(*, scope: str = "data") -> None:
    keys = _agent_keys(scope)
    st.session_state[keys["picker_version"]] = (
        st.session_state.get(keys["picker_version"], 0) + 1
    )


def _create_agent_for_session(
    session_path: str,
    *,
    host_context: str | None = None,
) -> tuple[Any, str]:
    if not _is_valid_session_relpath(session_path):
        raise RuntimeError(f"對話紀錄路徑無效：{session_path!r}")
    context = dataset_base_context() if host_context is None else host_context
    return create_agent_for_session(
        PROJECT_ROOT,
        session_path,
        context,
    )


def _get_agent_for_session(
    session_path: str,
    *,
    scope: str = "data",
    host_context: str | None = None,
) -> Any:
    keys = _agent_keys(scope)
    if (
        keys["agent"] not in st.session_state
        or st.session_state.get(keys["agent_session_path"]) != session_path
    ):
        agent, factory_ref = _create_agent_for_session(
            session_path,
            host_context=host_context,
        )
        st.session_state[keys["agent"]] = agent
        st.session_state[keys["agent_session_path"]] = session_path
        st.session_state[keys["agent_factory_ref"]] = factory_ref
        st.session_state[keys["agent_connected"]] = True
    return st.session_state[keys["agent"]]


def _activate_agent(
    session_path: str,
    *,
    scope: str = "data",
    host_context: str | None = None,
) -> tuple[bool, str]:
    keys = _agent_keys(scope)
    _clear_agent_cache(scope=scope)
    try:
        agent, factory_ref = _create_agent_for_session(
            session_path,
            host_context=host_context,
        )
    except RuntimeError as exc:
        _remove_activation_marker()
        return False, str(exc)
    except Exception as exc:
        _remove_activation_marker()
        return False, f"Agent 啟用失敗：{exc}"

    st.session_state[keys["agent"]] = agent
    st.session_state[keys["agent_session_path"]] = session_path
    st.session_state[keys["agent_factory_ref"]] = factory_ref
    st.session_state[keys["agent_connected"]] = True
    _write_activation_marker()
    return True, "Agent 已連接。"


def _restore_agent_if_possible(
    session_path: str,
    *,
    scope: str = "data",
    host_context: str | None = None,
) -> tuple[bool, str | None]:
    keys = _agent_keys(scope)
    if st.session_state.get(keys["agent_connected"]):
        return True, None
    if not AGENT_ACTIVATION_MARKER_PATH.exists():
        return False, None
    ok, message = _activate_agent(
        session_path,
        scope=scope,
        host_context=host_context,
    )
    if ok:
        return True, None
    return False, message


def invoke_data_agent(
    user_text: str,
    *,
    extra_context: str = "",
    display_user_text: str | None = None,
) -> str:
    """程式呼叫 Agent 一輪，並寫入右側 chat 歷史。失敗時回傳錯誤字串。"""
    current_session = st.session_state.get("session_path")
    if not current_session or not st.session_state.get("data_agent_connected"):
        return "Agent 尚未啟用，無法繼續實驗決策。"

    if "data_chat_history" not in st.session_state:
        st.session_state["data_chat_history"] = []

    shown = display_user_text if display_user_text is not None else user_text
    st.session_state["data_chat_history"].append(("user", shown))

    df = load_working_dataset()
    snapshot = dataset_page_snapshot(df, extra_context)
    prompt = format_user_turn(user_text, extra_context=snapshot)

    try:
        agent = _get_agent_for_session(current_session)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            final_text = agent.chat(prompt, image_path=None, on_token=lambda _token: None)
        answer = (final_text or "").strip() or "（Agent 未回傳文字）"
    except Exception as exc:  # keep classroom UI alive
        answer = f"Agent 執行時發生錯誤：`{exc}`"

    st.session_state["data_chat_history"].append(("assistant", answer, ""))
    return answer


def render_chat_panel(
    extra_context: str = "",
    page_name: str = "",
    *,
    host_context: str | None = None,
    agent_scope: str = "data",
    skip_working_snapshot: bool = False,
) -> None:
    keys = _agent_keys(agent_scope)
    # 從其他頁回到本 scope 時丟掉 Agent 實例（保留對話），以當前 host_context 重建
    last_page = st.session_state.get("last_chat_page")
    if page_name and last_page != page_name:
        st.session_state.pop(keys["agent"], None)
        st.session_state.pop(keys["agent_session_path"], None)
    if page_name:
        st.session_state["last_chat_page"] = page_name
    st.markdown(
        '<div class="data-agent-title-text">資料 Agent</div>',
        unsafe_allow_html=True,
    )

    df = None if skip_working_snapshot else load_working_dataset()

    sessions = _list_sessions()
    if keys["session_path"] not in st.session_state and sessions:
        _set_current_session(sessions[0], scope=agent_scope)
    current_session = _ensure_valid_current_session(sessions, scope=agent_scope)

    if keys["chat_history"] not in st.session_state:
        greeting = (
            "請先按「啟用資料 Agent」。啟用後，我會以專案展示規則協助你："
            "先讀說明書、檢查 Challenge 起點資料，清理 Challenge 工作資料，再切成訓練／測試資料；"
            "討論後用 AI coding 補齊模型區與成果區。"
            if agent_scope == "challenge"
            else (
                "請先按「啟用資料 Agent」。啟用後，我可以協助你理解資料整理流程；"
                "雙表合併建立工作資料後，也能一起分析 Working。"
            )
        )
        st.session_state[keys["chat_history"]] = [("assistant", greeting)]

    ids, labels = _build_session_picker_options(sessions)
    if current_session and current_session not in labels and _is_valid_session_relpath(current_session):
        ids.insert(0, current_session)
        labels[current_session] = "剛剛 · 目前對話"

    picker_key = (
        f"{agent_scope}_session_picker_{st.session_state.get(keys['picker_version'], 0)}"
    )

    selected_index = ids.index(current_session) if current_session in ids else 0

    pick_col, new_col, del_col = st.columns([6, 1, 1])
    picked_id = pick_col.selectbox(
        "對話紀錄",
        ids,
        index=selected_index,
        format_func=lambda value: labels.get(value, value),
        disabled=not ids,
        label_visibility="collapsed",
        key=picker_key,
    )
    resolved_pick = _resolve_session_relpath(picked_id, labels)
    if resolved_pick and resolved_pick != current_session:
        _set_current_session(PROJECT_ROOT / resolved_pick, scope=agent_scope)
        st.rerun()
    if new_col.button(
        "",
        icon=":material/add:",
        help="新增對話",
        width="stretch",
        key=f"{agent_scope}_session_new",
    ):
        _set_current_session(_new_session_path(), scope=agent_scope)
        _reset_session_picker_widget(scope=agent_scope)
        st.rerun()
    if del_col.button(
        "",
        icon=":material/delete:",
        help="刪除對話",
        width="stretch",
        disabled=not current_session,
        key=f"{agent_scope}_session_del",
    ):
        if current_session:
            target = PROJECT_ROOT / current_session
            if target.exists():
                target.unlink()
            st.session_state.pop(keys["session_path"], None)
            st.session_state.pop(keys["chat_history"], None)
            st.session_state.pop(keys["agent"], None)
            st.session_state.pop(keys["agent_session_path"], None)
            remaining = _list_sessions()
            if remaining:
                _set_current_session(remaining[0], scope=agent_scope)
            _reset_session_picker_widget(scope=agent_scope)
            st.rerun()

    _ensure_user_settings_file()

    current_session = st.session_state.get(keys["session_path"])
    if not current_session:
        st.caption("尚無對話紀錄，請按 **+** 新增對話。")
        st.chat_input("詢問...", disabled=True, key=f"{agent_scope}_chat_no_session")
        return

    restored, restore_error = _restore_agent_if_possible(
        current_session,
        scope=agent_scope,
        host_context=host_context,
    )
    connected = bool(st.session_state.get(keys["agent_connected"])) or restored
    status_text = (
        ":green[●] Agent：已連接"
        if connected
        else ":red[●] Agent：未啟用"
    )
    st.markdown(f"**{status_text}**")
    if not connected:
        probe = load_create_agent(PROJECT_ROOT)
        if probe.error and probe.factory is None:
            st.warning(probe.error)
        if restore_error:
            st.warning(restore_error)
        if st.button(
            "啟用資料 Agent",
            type="primary",
            width="stretch",
            key=f"{agent_scope}_activate_agent",
        ):
            ok, message = _activate_agent(
                current_session,
                scope=agent_scope,
                host_context=host_context,
            )
            if ok:
                st.success(f"{message} 你可以開始詢問資料 Agent。")
                st.rerun()
            else:
                st.error(message)
        st.chat_input(
            "請先啟用資料 Agent...",
            disabled=True,
            key=f"{agent_scope}_chat_not_activated",
        )
        return

    try:
        agent = _get_agent_for_session(
            current_session,
            scope=agent_scope,
            host_context=host_context,
        )
    except RuntimeError as exc:
        st.error(str(exc))
        _clear_agent_cache(scope=agent_scope)
        _remove_activation_marker()
        st.chat_input("詢問 Agent...", disabled=True, key=f"{agent_scope}_chat_no_key")
        return
    except Exception as exc:
        st.error(f"Agent 連線失敗：`{exc}`")
        _clear_agent_cache(scope=agent_scope)
        _remove_activation_marker()
        st.chat_input(
            "詢問 Agent...",
            disabled=True,
            key=f"{agent_scope}_chat_connect_failed",
        )
        return

    st.markdown('<div data-dss-chat style="display:none"></div>', unsafe_allow_html=True)
    # Short placeholder; dual-pane chrome JS grows this to fill space above input.
    # A tall default (e.g. 720) clips in-flow chat_input under Agent overflow:hidden.
    chat = st.container(height=240, border=False)
    with chat:
        for entry in st.session_state[keys["chat_history"]]:
            role, text, reasoning = _parse_history_entry(entry)
            _render_history_message(role, text, reasoning=reasoning)

    if user_text := st.chat_input("詢問資料 Agent...", key=f"{agent_scope}_chat"):
        st.session_state[keys["chat_history"]].append(("user", user_text))
        # host_context 已在建立 Agent 時注入 system；此處只組 user 層
        if skip_working_snapshot:
            snapshot = extra_context.strip()
        else:
            snapshot = dataset_page_snapshot(df, extra_context)
        prompt = format_user_turn(user_text, extra_context=snapshot)

        with chat:
            with st.chat_message("user"):
                st.markdown(user_text)
            with st.chat_message("assistant"):
                reasoning_slot = st.empty()
                placeholder = st.empty()
                answer_parts: list[str] = []
                reasoning_segments: list[str] = []
                reasoning_parts: list[str] = []
                stream_ui: dict[str, Any] = {
                    "reasoning_ph": None,
                    "visible": False,
                    "expanded": False,
                    "answer_started": False,
                }

                def _sync_reasoning_ui() -> None:
                    text = _merged_reasoning_text(reasoning_segments, reasoning_parts)
                    if not text:
                        return
                    expanded = not stream_ui["answer_started"]
                    if expanded:
                        if not stream_ui["visible"] or not stream_ui["expanded"]:
                            _render_reasoning_expander(
                                reasoning_slot,
                                text,
                                expanded=True,
                                stream_ui=stream_ui,
                            )
                        elif stream_ui["reasoning_ph"] is not None:
                            stream_ui["reasoning_ph"].markdown(text)
                        else:
                            _render_reasoning_expander(
                                reasoning_slot,
                                text,
                                expanded=True,
                                stream_ui=stream_ui,
                            )
                    else:
                        _render_reasoning_expander(
                            reasoning_slot,
                            text,
                            expanded=False,
                            stream_ui=stream_ui,
                        )

                def on_reasoning(token: str) -> None:
                    reasoning_parts.append(token)
                    _sync_reasoning_ui()

                def on_token(token: str) -> None:
                    if not stream_ui["answer_started"]:
                        stream_ui["answer_started"] = True
                        _sync_reasoning_ui()
                    answer_parts.append(token)
                    placeholder.markdown("".join(answer_parts))

                def on_stream_reset() -> None:
                    _commit_reasoning_round(reasoning_segments, reasoning_parts)
                    answer_parts.clear()
                    stream_ui["answer_started"] = False
                    placeholder.markdown(TOOL_RUN_PLACEHOLDER)
                    merged = _merged_reasoning_text(reasoning_segments, reasoning_parts)
                    if merged:
                        _render_reasoning_expander(
                            reasoning_slot,
                            merged,
                            expanded=True,
                            stream_ui=stream_ui,
                        )

                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        final_text = agent.chat(
                            prompt,
                            **_chat_callback_kwargs(
                                agent.chat,
                                on_token=on_token,
                                on_reasoning=on_reasoning,
                                on_stream_reset=on_stream_reset,
                            ),
                        )
                except Exception as exc:  # keep classroom UI alive during agent debugging
                    final_text = f"Agent 執行時發生錯誤：`{exc}`"
                    answer = final_text
                    placeholder.error(final_text)
                else:
                    answer = "".join(answer_parts).strip() or final_text.strip()

                _commit_reasoning_round(reasoning_segments, reasoning_parts)
                reasoning_text = _merged_reasoning_text(reasoning_segments, [])
                st.session_state[keys["chat_history"]].append(
                    ("assistant", answer, reasoning_text)
                )
                st.session_state[keys["chat_just_replied"]] = True
