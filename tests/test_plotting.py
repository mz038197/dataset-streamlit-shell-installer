from __future__ import annotations

import sys
import types
from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.plotting import configure_matplotlib_for_traditional_chinese


def test_configure_matplotlib_for_traditional_chinese_prefers_installed_cjk_font(
    monkeypatch,
) -> None:
    fake_pyplot = types.SimpleNamespace(rcParams={})
    fake_font_manager = types.SimpleNamespace(
        fontManager=types.SimpleNamespace(
            ttflist=[
                types.SimpleNamespace(name="DejaVu Sans"),
                types.SimpleNamespace(name="Microsoft JhengHei"),
            ]
        )
    )
    fake_matplotlib = types.SimpleNamespace(font_manager=fake_font_manager)
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib.font_manager", fake_font_manager)

    configure_matplotlib_for_traditional_chinese()

    assert fake_pyplot.rcParams["font.family"] == "Microsoft JhengHei"
    assert fake_pyplot.rcParams["axes.unicode_minus"] is False
