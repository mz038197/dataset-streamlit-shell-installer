from __future__ import annotations

from pathlib import Path

PAGES = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "pages"
)
UI = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "ui"
)


def test_kmeans_and_wards_pages_use_builtin_not_ready_shell() -> None:
    for name in ("12_KMeans.py", "11_Wards_Method.py"):
        src = (PAGES / name).read_text(encoding="utf-8")
        assert "render_analysis_shell" not in src
        assert "load_ready_dataset" not in src
        assert "ready.csv" not in src.lower()


def test_clustering_ui_wires_builtin_and_truth_toggle() -> None:
    src = (UI / "clustering_ui.py").read_text(encoding="utf-8")
    assert "load_builtin_clustering_frame" in src
    assert "fit_kmeans" in src
    assert "ward_linkage" in src
    assert "cut_ward_clusters" in src
    assert "對照教學用真實群標籤" in src
    assert "DEFAULT_N_CLUSTERS" in src
    assert "render_kmeans_page" in src
    assert "render_wards_page" in src
    assert "標準化" not in src
    assert "相同數字不代表同一群" in src
