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

import pandas as pd

from dataset_streamlit_shell.plotting import (
    CLASS_NEGATIVE_STYLE,
    CLASS_POSITIVE_STYLE,
    build_classification_data_figures,
    build_linear_svm_result_figure,
    build_regression_data_figures,
    build_sigmoid_figure,
    configure_matplotlib_for_traditional_chinese,
)


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


def test_build_regression_data_figures_single_feature() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    figures = build_regression_data_figures(frame, ["x"], "y")
    assert len(figures) == 1


def test_build_sigmoid_figure() -> None:
    import matplotlib.pyplot as plt

    fig = build_sigmoid_figure(highlight_z=2.0)
    assert fig.axes
    plt.close(fig)


def test_build_classification_data_figures_two_features() -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.5, 3.0, 1.0],
            "x2": [1.5, 0.5, 2.5],
            "label": [0, 1, 1],
        }
    )
    figures = build_classification_data_figures(frame, ["x1", "x2"], "label")
    assert len(figures) == 1


def test_binary_class_scatter_defaults_to_zero_one_labels() -> None:
    assert CLASS_NEGATIVE_STYLE["label"] == "y=0"
    assert CLASS_POSITIVE_STYLE["label"] == "y=1"


def test_classification_data_figure_legend_uses_zero_one() -> None:
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        {
            "x1": [0.5, 3.0, 1.0],
            "x2": [1.5, 0.5, 2.5],
            "label": [0, 1, 1],
        }
    )
    _, fig = build_classification_data_figures(frame, ["x1", "x2"], "label")[0]
    _, labels = fig.axes[0].get_legend_handles_labels()
    assert labels == ["y=0", "y=1"]
    plt.close(fig)


def test_linear_svm_result_figure_legend_keeps_plus_minus_one() -> None:
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.0, 1.0],
            "x2": [0.0, 0.0, 1.0, 1.0],
            "y": [-1, -1, 1, 1],
        }
    )
    fig = build_linear_svm_result_figure(
        frame,
        ["x1", "x2"],
        "y",
        coef=[1.0, 1.0],
        intercept=-0.5,
        support_vectors=[[0.0, 1.0], [1.0, 0.0]],
    )
    _, labels = fig.axes[0].get_legend_handles_labels()
    assert "y=-1" in labels
    assert "y=+1" in labels
    assert "y=0" not in labels
    plt.close(fig)
