from __future__ import annotations

import sys
import types
from importlib import util
from pathlib import Path

import pytest

TEMPLATE_ROOT = (
    Path(__file__).parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
DATA_UI_PATH = TEMPLATE_ROOT / "dataset_streamlit_shell" / "ui" / "data_ui.py"


class _FakeSettings:
    def __init__(
        self,
        voice: str = "nova",
        instructions: str = "default instructions",
        speed: float | None = None,
    ) -> None:
        self.voice = voice
        self.instructions = instructions
        self.speed = speed

    @classmethod
    def from_env(cls):
        return cls()


def _load_data_ui_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_streamlit = types.SimpleNamespace(session_state={})
    fake_dotenv = types.SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None)
    fake_pandas = types.SimpleNamespace(DataFrame=object)
    fake_openai_tts = types.SimpleNamespace(
        Settings=_FakeSettings,
        stream_tts_play=lambda *_args, **_kwargs: None,
    )
    fake_openai_tts_settings = types.SimpleNamespace(
        MIN_TTS_SPEED=0.25,
        MAX_TTS_SPEED=4.0,
    )

    if str(TEMPLATE_ROOT) not in sys.path:
        sys.path.insert(0, str(TEMPLATE_ROOT))

    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    monkeypatch.setitem(sys.modules, "openai_tts", fake_openai_tts)
    monkeypatch.setitem(sys.modules, "openai_tts.settings", fake_openai_tts_settings)

    spec = util.spec_from_file_location("data_ui_reasoning_test", DATA_UI_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(module, "USER_SETTINGS_PATH", tmp_path / "workspace" / "user_settings.json")
    return module


def test_commit_reasoning_round_appends_and_clears_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_data_ui_module(monkeypatch, tmp_path)
    segments: list[str] = []
    current = ["think", "ing"]

    module._commit_reasoning_round(segments, current)

    assert segments == ["thinking"]
    assert current == []


def test_commit_reasoning_round_skips_empty_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_data_ui_module(monkeypatch, tmp_path)
    segments: list[str] = []
    current: list[str] = []

    module._commit_reasoning_round(segments, current)

    assert segments == []


def test_merged_reasoning_text_joins_segments_with_separator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_data_ui_module(monkeypatch, tmp_path)

    merged = module._merged_reasoning_text(["round1"], ["round2"])

    assert merged == f"round1{module.REASONING_ROUND_SEPARATOR}round2"


def test_merged_reasoning_text_after_commit_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_data_ui_module(monkeypatch, tmp_path)
    segments: list[str] = []
    current = ["a"]

    module._commit_reasoning_round(segments, current)
    current.extend(["b"])
    merged = module._merged_reasoning_text(segments, current)

    assert merged == f"a{module.REASONING_ROUND_SEPARATOR}b"


def test_parse_history_entry_accepts_legacy_two_tuple(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_data_ui_module(monkeypatch, tmp_path)

    role, text, reasoning = module._parse_history_entry(("assistant", "答案"))

    assert role == "assistant"
    assert text == "答案"
    assert reasoning == ""


def test_parse_history_entry_reads_reasoning_third_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_data_ui_module(monkeypatch, tmp_path)

    role, text, reasoning = module._parse_history_entry(("assistant", "答案", "思考"))

    assert role == "assistant"
    assert text == "答案"
    assert reasoning == "思考"


def test_load_session_history_does_not_restore_reasoning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_data_ui_module(monkeypatch, tmp_path)
    path = tmp_path / "session.jsonl"
    path.write_text(
        '{"role":"user","content":"嗨"}\n{"role":"assistant","content":"好"}\n',
        encoding="utf-8",
    )

    history = module._load_session_history(path)

    assert history == [("user", "嗨"), ("assistant", "好", "")]


def test_chat_callback_kwargs_omits_reasoning_when_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_data_ui_module(monkeypatch, tmp_path)

    def chat(user_text, *, image_path=None, on_token=None):
        return user_text

    kwargs = module._chat_callback_kwargs(
        chat,
        on_token=lambda _t: None,
        on_reasoning=lambda _t: None,
        on_stream_reset=lambda: None,
    )

    assert "on_token" in kwargs
    assert "on_reasoning" not in kwargs
    assert "on_stream_reset" not in kwargs


def test_chat_callback_kwargs_includes_reasoning_when_supported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_data_ui_module(monkeypatch, tmp_path)

    def chat(
        user_text,
        *,
        image_path=None,
        on_token=None,
        on_reasoning=None,
        on_stream_reset=None,
    ):
        return user_text

    on_reasoning = lambda _t: None
    on_reset = lambda: None
    kwargs = module._chat_callback_kwargs(
        chat,
        on_token=lambda _t: None,
        on_reasoning=on_reasoning,
        on_stream_reset=on_reset,
    )

    assert kwargs["on_reasoning"] is on_reasoning
    assert kwargs["on_stream_reset"] is on_reset


def test_on_stream_reset_does_not_clear_reasoning_segments_in_source() -> None:
    source = DATA_UI_PATH.read_text(encoding="utf-8")
    block = source.split("def on_stream_reset() -> None:", 1)[1].split("try:", 1)[0]

    assert "reasoning_segments.clear()" not in block
    assert "reasoning_slot.empty()" not in block
    assert "reasoning_parts.clear()" not in block
    assert "_commit_reasoning_round(reasoning_segments, reasoning_parts)" in block
    assert "TOOL_RUN_PLACEHOLDER" in block


def test_chat_input_flow_saves_reasoning_and_hides_tts() -> None:
    source = DATA_UI_PATH.read_text(encoding="utf-8")
    user_message_flow = source.split(
        'if user_text := st.chat_input("詢問資料 Agent...", key=f"{agent_scope}_chat"):',
        1,
    )[1]

    assert '("assistant", answer, reasoning_text)' in user_message_flow
    assert "on_reasoning=on_reasoning" in user_message_flow or "on_reasoning" in user_message_flow
    assert "stream_tts_play" not in user_message_flow
    assert "思考過程" in source


def test_render_chat_panel_does_not_show_tts_ui() -> None:
    source = DATA_UI_PATH.read_text(encoding="utf-8")
    panel = source.split("def render_chat_panel(", 1)[1]

    assert "_render_tts_settings_ui" not in panel
    assert "stream_tts_play" not in panel


def test_invoke_data_agent_stores_empty_reasoning() -> None:
    source = DATA_UI_PATH.read_text(encoding="utf-8")
    block = source.split("def invoke_data_agent(", 1)[1].split("def render_chat_panel(", 1)[0]

    assert '("assistant", answer, "")' in block
    assert "on_reasoning" not in block


def test_context_defines_reasoning_and_hidden_tts() -> None:
    context = (
        Path(__file__).parents[1] / "CONTEXT.md"
    ).read_text(encoding="utf-8")
    assert "**思考過程**:" in context
    assert "**語音設定**:" in context
    assert "已隱藏" in context.split("**語音設定**:", 1)[1].split("**", 1)[0]
