"""Smoke-check chart quiz helpers (no Streamlit UI).

Default: in-memory Titanic-like fixture (CI / any machine).
Optional: pass a CSV path as the first CLI argument.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "add_dataset_streamlit_shell" / "templates"
sys.path.insert(0, str(TEMPLATE))
os.environ["DATASET_CHARTS_SKIP_RUN"] = "1"

sys.modules["streamlit"] = MagicMock()
sys.modules["dataset_streamlit_shell.ui.data_ui"] = MagicMock()
sys.modules["dataset_streamlit_shell.plotting"] = MagicMock()

import pandas as pd

page_path = TEMPLATE / "dataset_streamlit_shell" / "pages" / "2_Charts.py"
spec = importlib.util.spec_from_file_location("charts_page", page_path)
assert spec and spec.loader
charts = importlib.util.module_from_spec(spec)
sys.modules["charts_page"] = charts
spec.loader.exec_module(charts)


def _classroom_titanic_fixture(n: int = 40) -> pd.DataFrame:
    """Minimal Titanic-shaped frame for smoke checks without external files."""
    return pd.DataFrame(
        {
            "PassengerId": list(range(1, n + 1)),
            "Survived": [i % 2 for i in range(n)],
            "Pclass": [(i % 3) + 1 for i in range(n)],
            "Name": [f"Passenger {i}" for i in range(n)],
            "Sex": ["male" if i % 2 == 0 else "female" for i in range(n)],
            "Age": [20.0 + (i % 15) for i in range(n)],
            "SibSp": [i % 3 for i in range(n)],
            "Parch": [i % 2 for i in range(n)],
            "Ticket": [f"TICKET-{i}" for i in range(n)],
            "Fare": [7.25 + i * 1.5 for i in range(n)],
            "Cabin": [None if i % 4 else f"C{i}" for i in range(n)],
            "Embarked": [["S", "C", "Q"][i % 3] for i in range(n)],
        }
    )


class _State(dict):
    pass


def _load_frame(csv_path: Path | None) -> pd.DataFrame:
    if csv_path is None:
        return _classroom_titanic_fixture()
    if not csv_path.exists():
        raise SystemExit(f"missing CSV: {csv_path}")
    return pd.read_csv(csv_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke-check chart quiz helpers.")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=None,
        help="Optional CSV path (default: in-memory Titanic-like fixture).",
    )
    args = parser.parse_args(argv)

    df = _load_frame(args.csv)
    roles = charts._infer_column_roles(df)

    assert "PassengerId" not in roles["categorical"]
    assert "PassengerId" in roles["measures"]  # numerics stay selectable
    assert "Name" not in roles["categorical"]
    assert "Sex" in roles["categorical"] or "Pclass" in roles["categorical"]
    assert "Age" in roles["continuous"] or "Age" in roles["measures"]
    assert not roles["datetime"], "classroom CSV / fixture should have no datetime columns"

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
    radar_cfg, radar_missing = charts._default_draw_config("radar", roles, df)
    assert radar_missing is None and radar_cfg is not None
    assert "PassengerId" not in radar_cfg["columns"]

    for item in charts.QUIZ_ITEMS:
        wrong = next(
            label for label in charts.CHART_LABELS if charts.CHART_LABEL_TO_KEY[label] != item.correct
        )
        assert not charts._selection_is_correct(item, charts.PLEASE_SELECT)
        assert not charts._selection_is_correct(item, wrong)
        assert charts._selection_is_correct(item, charts.CHART_KEY_TO_LABEL[item.correct])

        config, missing = charts._default_draw_config(item.correct, roles, df)
        assert missing is None, f"{item.id}: {missing}"
        assert config is not None
        fig, table, summary, note = charts._render_guided_chart(df, item.correct, config)
        assert fig is not None, f"{item.id}: fig is None ({note})"
        assert table is not None, f"{item.id}: table is None"
        charts.plt.close(fig)
        print(f"OK {item.id}: {item.prompt} -> {charts.CHART_KEY_TO_LABEL[item.correct]} rows={len(table)}")

    state = _State()
    state[charts._quiz_choice_key("compare_category")] = "圓餅圖"
    state[charts._quiz_choice_key("overall_proportion")] = charts.PLEASE_SELECT
    state["chart_quiz_focus"] = "compare_category"
    charts.st.session_state = state
    ctx = charts._build_agent_context(
        source_label="Working 工作資料",
        focus_id="compare_category",
        drawn_id=None,
        draw_summary=None,
    )
    assert "比較類別數量" in ctx
    assert "圓餅圖" in ctx
    assert "與建議不符" in ctx
    print("OK agent context")
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
