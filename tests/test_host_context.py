from __future__ import annotations

import sys
from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.host_context import (  # noqa: E402
    assemble_host_context,
    framework_host_context,
)

PACK_LAYER = "【課程包層】只寫 workspace/example.csv。"
PAGE_LAYER = "【頁面層】寫回協定。"


def test_assemble_orders_framework_then_pack_then_page() -> None:
    host = assemble_host_context(PACK_LAYER, PAGE_LAYER)
    framework = framework_host_context()
    assert host.startswith(framework)
    assert host.index(framework) < host.index(PACK_LAYER) < host.index(PAGE_LAYER)
    assert host.endswith(PAGE_LAYER)


def test_assemble_without_page_layer_ends_with_pack_layer() -> None:
    host = assemble_host_context(PACK_LAYER)
    assert host.startswith(framework_host_context())
    assert host.endswith(PACK_LAYER)


def test_assemble_without_pack_layer_is_framework_only() -> None:
    assert assemble_host_context() == framework_host_context()
    assert assemble_host_context(None) == framework_host_context()


PACK_VOCABULARY = (
    "Working",
    "working.csv",
    "Original",
    "original.csv",
    "雙表",
    "cleaning_log",
    "挑戰",
    "Challenge",
    "challenge",
    "ready.csv",
    "內建範例",
    "Titanic",
    "鐵達尼",
)


def test_framework_layer_has_no_course_pack_vocabulary() -> None:
    text = framework_host_context()
    for word in PACK_VOCABULARY:
        assert word not in text, word


def test_framework_layer_states_role_read_first_scripts_and_whitelist() -> None:
    text = framework_host_context()
    assert "資料 Agent" in text
    assert "read_file" in text
    assert "dataset_streamlit_shell/scripts" in text
    assert "可寫清單" in text


def test_overview_host_is_framework_layer_only() -> None:
    from dataset_streamlit_shell.ui.overview_ui import overview_host_context

    assert overview_host_context() == framework_host_context()


def _workshop_paths(company: str):
    from dataset_streamlit_shell.ui.startup_challenge_context import ChallengeHostPaths

    shell = "dataset_streamlit_shell"
    challenge = f"{shell}/workspace/challenge"
    return ChallengeHostPaths(
        start_csv=f"{challenge}/{company}.csv",
        handbook=f"{challenge}/{company}_資料說明書.md",
        working_csv=f"{challenge}/{company}/working.csv",
        train_csv=f"{challenge}/{company}/train.csv",
        test_csv=f"{challenge}/{company}/test.csv",
        scripts_dir=f"{shell}/scripts",
        live_ui=f"{shell}/ui/startup_challenge_ui.py",
        page_py=f"{shell}/ui/startup_challenge_page.py",
        empty_shell=f"{shell}/ui/startup_challenge_empty_shell.py",
    )


def _pack_layers() -> dict[str, str]:
    from dataset_streamlit_shell.ui.computer_vision_pack import computer_vision_pack_layer
    from dataset_streamlit_shell.ui.data_science import data_science_pack_layer
    from dataset_streamlit_shell.ui.deep_learning_pack import deep_learning_pack_layer
    from dataset_streamlit_shell.ui.pca_pack import pca_pack_layer
    from dataset_streamlit_shell.ui.startup_challenge_context import (
        startup_workshop_pack_layer,
    )
    from dataset_streamlit_shell.ui.supervised_pack import supervised_pack_layer
    from dataset_streamlit_shell.ui.unsupervised_pack import unsupervised_pack_layer

    return {
        "data-science": data_science_pack_layer(),
        "supervised": supervised_pack_layer(),
        "unsupervised": unsupervised_pack_layer(),
        "deep-learning": deep_learning_pack_layer(),
        "computer-vision": computer_vision_pack_layer(),
        "pca": pca_pack_layer(),
        "startup-workshop": startup_workshop_pack_layer(_workshop_paths("edupulse")),
    }


def _page_hosts() -> dict[str, tuple[str, str]]:
    """每頁（或同一 host 的頁組）→（所屬課程包, host）。"""
    from dataset_streamlit_shell.ui import knn_ui, logistic_regression_ui, lr_ui, nn_ui
    from dataset_streamlit_shell.ui import svm_ui, tree_ui
    from dataset_streamlit_shell.ui.computer_vision_pack import (
        computer_vision_host_context,
    )
    from dataset_streamlit_shell.ui.data_science import data_science_host_context
    from dataset_streamlit_shell.ui.pca_pack import pca_host_context
    from dataset_streamlit_shell.ui.startup_challenge_context import challenge_host_context
    from dataset_streamlit_shell.ui.unsupervised_pack import unsupervised_host_context

    return {
        "資料整理各頁": ("data-science", data_science_host_context()),
        "線性回歸": ("supervised", lr_ui.page_host_context()),
        "邏輯回歸": ("supervised", logistic_regression_ui.page_host_context()),
        "線性 SVM": ("supervised", svm_ui.page_host_context()),
        "KNN": ("supervised", knn_ui.page_host_context()),
        "決策樹": ("supervised", tree_ui.page_host_context()),
        "分群": ("unsupervised", unsupervised_host_context()),
        "類神經網路": ("deep-learning", nn_ui._nn_host()),
        "電腦視覺各頁": ("computer-vision", computer_vision_host_context()),
        "PCA": ("pca", pca_host_context()),
        "專案展示": (
            "startup-workshop",
            challenge_host_context(company="edupulse", paths=_workshop_paths("edupulse")),
        ),
    }


def test_every_page_host_carries_exactly_one_course_pack_layer() -> None:
    layers = _pack_layers()
    for page, (pack, host) in _page_hosts().items():
        assert host.startswith(framework_host_context()), page
        assert host.count(layers[pack]) == 1, page
        others = [name for name, layer in layers.items() if name != pack and layer in host]
        assert others == [], f"{page} 還帶了 {others}"


def test_old_default_and_teaching_hosts_are_gone() -> None:
    from dataset_streamlit_shell.ui import data_ui

    assert not hasattr(data_ui, "dataset_base_context")
    assert not hasattr(data_ui, "teaching_page_host_context")


DATA_SCIENCE_FILE_NAMES = (
    "original.csv",
    "working.csv",
    "ready.csv",
    "cleaning_log",
    "雙表",
)


def _without_challenge_track_working(layer: str) -> str:
    """挑戰軌道自己的工作副本也叫 working.csv，但一定指目前公司資料夾內那份。"""
    return layer.replace("edupulse/working.csv", "edupulse/<working>").replace(
        "公司的 working.csv", "公司的 <working>"
    )


def test_pack_layers_do_not_name_data_science_files() -> None:
    """課程包層是白名單、不引用其他課程包；只有 Data Science 與 PCA 的那份副本會提到 DS 檔名。"""
    for pack, layer in _pack_layers().items():
        if pack in {"data-science", "pca"}:
            continue
        text = _without_challenge_track_working(layer)
        for name in DATA_SCIENCE_FILE_NAMES:
            assert name not in text, f"{pack} 課程包層提到 {name}"


def test_assemble_drops_page_layer_without_pack_layer() -> None:
    """頁面層只能疊在課程包層上；沒有課程包層時頁面層被丟掉，只剩框架層。"""
    assert assemble_host_context(None, PAGE_LAYER) == framework_host_context()
    assert assemble_host_context(page_layer=PAGE_LAYER) == framework_host_context()
