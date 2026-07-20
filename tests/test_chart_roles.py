"""Column-role inference and count-based defaults for the Charts quiz page."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)


def _load_charts_module():
    if str(TEMPLATE_ROOT) not in sys.path:
        sys.path.insert(0, str(TEMPLATE_ROOT))
    os.environ["DATASET_CHARTS_SKIP_RUN"] = "1"
    sys.modules.setdefault("streamlit", MagicMock())
    sys.modules.setdefault("dataset_streamlit_shell.ui.data_ui", MagicMock())
    sys.modules.setdefault("dataset_streamlit_shell.plotting", MagicMock())

    page_path = TEMPLATE_ROOT / "dataset_streamlit_shell" / "pages" / "2_Charts.py"
    module_name = "charts_page_under_test"
    spec = importlib.util.spec_from_file_location(module_name, page_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_infer_keeps_all_numerics_and_datetime() -> None:
    charts = _load_charts_module()
    n = 30
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "revenue": [100.0 + i * 3.5 for i in range(n)],
            "store_id": list(range(1, n + 1)),
            "region": [["北", "中", "南"][i % 3] for i in range(n)],
            "Name": [f"row-{i}" for i in range(n)],
        }
    )

    roles = charts._infer_column_roles(df)

    assert "date" in roles["datetime"]
    assert "revenue" in roles["measures"]
    assert "store_id" in roles["measures"]  # numerics stay selectable
    assert "region" in roles["categorical"]
    assert "Name" not in roles["categorical"]  # high-cardinality text skipped


def test_infer_works_with_chinese_column_names() -> None:
    charts = _load_charts_module()
    n = 24
    df = pd.DataFrame(
        {
            "日期": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "營收": [80 + i * 2 for i in range(n)],
            "名次": list(range(1, n + 1)),
            "地區": [["北", "中", "南"][i % 3] for i in range(n)],
        }
    )

    roles = charts._infer_column_roles(df)

    assert "日期" in roles["datetime"]
    assert "營收" in roles["measures"]
    assert "名次" in roles["measures"]
    assert "地區" in roles["categorical"]

    bar_cfg, bar_missing = charts._default_draw_config("bar", roles, df)
    line_cfg, line_missing = charts._default_draw_config("line", roles, df)
    assert bar_missing is None and bar_cfg is not None
    assert bar_cfg["y_col"] == charts.COUNT_ROWS
    assert bar_cfg["aggregation"] == "count"
    assert line_missing is None and line_cfg is not None
    # Unique dates → count would be flat; prefer a numeric measure.
    assert line_cfg["x_col"] == "日期"
    assert line_cfg["y_col"] == "營收"
    assert line_cfg["aggregation"] == "mean"


def test_defaults_avoid_surrogate_ids_but_keep_them_selectable() -> None:
    charts = _load_charts_module()
    n = 40
    df = pd.DataFrame(
        {
            "PassengerId": list(range(1, n + 1)),
            "Name": [f"Passenger {i}" for i in range(n)],
            "Sex": ["male" if i % 2 == 0 else "female" for i in range(n)],
            "Age": [20.0 + (i % 10) for i in range(n)],
            "Fare": [7.25 + i * 1.5 for i in range(n)],
            "Pclass": [(i % 3) + 1 for i in range(n)],
            "SibSp": [i % 3 for i in range(n)],
            "Parch": [i % 2 for i in range(n)],
        }
    )
    roles = charts._infer_column_roles(df)

    assert "PassengerId" in roles["measures"]

    for chart_type in ("bar", "pie", "stacked_bar"):
        config, missing = charts._default_draw_config(chart_type, roles, df)
        assert missing is None, chart_type
        assert config is not None
        assert config["y_col"] == charts.COUNT_ROWS
        assert config["aggregation"] == "count"

    line_cfg, line_missing = charts._default_draw_config("line", roles, df)
    assert line_missing is None and line_cfg is not None
    assert line_cfg["x_col"] != "PassengerId"

    hist_cfg, hist_missing = charts._default_draw_config("histogram", roles, df)
    assert hist_missing is None and hist_cfg is not None
    assert hist_cfg["column"] != "PassengerId"
    assert hist_cfg["column"] in {"Age", "Fare"}

    radar_cfg, radar_missing = charts._default_draw_config("radar", roles, df)
    assert radar_missing is None and radar_cfg is not None
    assert "PassengerId" not in radar_cfg["columns"]
    assert len(radar_cfg["columns"]) >= 3


def test_defaults_never_pad_with_surrogate_keys() -> None:
    """Auto-defaults must exclude IDs entirely — not merely rank them last."""
    charts = _load_charts_module()
    n = 36

    # id + height + weight: radar must not pad the 3rd axis with id
    body = pd.DataFrame(
        {
            "id": list(range(1, n + 1)),
            "height": [150.0 + (i % 20) for i in range(n)],
            "weight": [50.0 + (i % 15) * 1.5 for i in range(n)],
        }
    )
    body_roles = charts._infer_column_roles(body)
    assert "id" in body_roles["measures"]  # still selectable via roles/picker

    radar_cfg, radar_missing = charts._default_draw_config("radar", body_roles, body)
    assert radar_cfg is None
    assert radar_missing is not None

    hist_cfg, hist_missing = charts._default_draw_config("histogram", body_roles, body)
    assert hist_missing is None and hist_cfg is not None
    assert hist_cfg["column"] in {"height", "weight"}

    # date + id: line/hist must not default to averaging/plotting id
    dated = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "id": list(range(1, n + 1)),
        }
    )
    dated_roles = charts._infer_column_roles(dated)
    assert "id" in dated_roles["measures"]

    line_cfg, line_missing = charts._default_draw_config("line", dated_roles, dated)
    assert line_cfg is None
    assert line_missing is not None

    hist2_cfg, hist2_missing = charts._default_draw_config("histogram", dated_roles, dated)
    assert hist2_cfg is None
    assert hist2_missing is not None

    # only id: line/hist must refuse auto-draw instead of plotting the key
    only_id = pd.DataFrame({"id": list(range(1, n + 1))})
    only_roles = charts._infer_column_roles(only_id)
    assert "id" in only_roles["measures"]

    line3_cfg, line3_missing = charts._default_draw_config("line", only_roles, only_id)
    assert line3_cfg is None
    assert line3_missing is not None

    hist3_cfg, hist3_missing = charts._default_draw_config("histogram", only_roles, only_id)
    assert hist3_cfg is None
    assert hist3_missing is not None


def test_missing_safe_defaults_fall_back_to_explicit_manual_selection() -> None:
    charts = _load_charts_module()
    n = 30

    ranked = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "rank": list(range(1, n + 1)),
        }
    )
    ranked_roles = charts._infer_column_roles(ranked)

    line_cfg = charts._manual_selection_config("line", ranked_roles, ranked)
    assert line_cfg is not None
    assert line_cfg["x_col"] == "date"
    assert line_cfg["y_col"] == charts.PLEASE_SELECT

    hist_cfg = charts._manual_selection_config("histogram", ranked_roles, ranked)
    assert hist_cfg is not None
    assert hist_cfg["column"] == charts.PLEASE_SELECT

    only_id = pd.DataFrame({"id": list(range(1, n + 1))})
    only_roles = charts._infer_column_roles(only_id)
    only_line_cfg = charts._manual_selection_config("line", only_roles, only_id)
    assert only_line_cfg is not None
    assert only_line_cfg["x_col"] == charts.PLEASE_SELECT
    assert only_line_cfg["y_col"] == charts.PLEASE_SELECT

    body = pd.DataFrame(
        {
            "id": list(range(1, n + 1)),
            "height": [150.0 + (i % 20) for i in range(n)],
            "weight": [50.0 + (i % 15) * 1.5 for i in range(n)],
        }
    )
    body_roles = charts._infer_column_roles(body)
    radar_cfg = charts._manual_selection_config("radar", body_roles, body)
    assert radar_cfg is not None
    assert "id" not in radar_cfg["columns"]
    assert len(radar_cfg["columns"]) == 2


def test_agent_context_does_not_claim_waiting_selection_is_rendered() -> None:
    charts = _load_charts_module()
    item = charts.QUIZ_ITEMS[-1]
    charts.st.session_state = {
        charts._quiz_choice_key(item.id): charts.CHART_KEY_TO_LABEL[item.correct],
        charts._quiz_drawn_key(item.id): True,
    }

    context = charts._build_agent_context(
        source_label="Working 工作資料",
        focus_id=item.id,
        drawn_id=item.id,
        draw_summary={
            "題幹": item.prompt,
            "圖表": charts.CHART_KEY_TO_LABEL[item.correct],
            "狀態": "等待選擇欄位",
        },
    )

    assert "等待選擇欄位" in context
    assert "已畫出預覽" not in context
