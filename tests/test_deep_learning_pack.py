"""深度學習課程包可單獨安裝（#14）：裝出什麼、套件、host 三層。"""

from __future__ import annotations

from pathlib import Path

from pack_install_helpers import (
    TEMPLATE_SHELL,
    FakeDependencyRunner,
    installed_files,
    installed_shell,
    sidebar,
    third_party_imports,
    unresolved_internal_imports,
)

from add_dataset_streamlit_shell.course_packs import DEEP_LEARNING, pack_files
from add_dataset_streamlit_shell.installer import FRAMEWORK_DEPENDENCIES, add_packs, install_shell


def test_framework_plus_deep_learning_resolves_every_internal_import(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "deep-learning")

    files = installed_files(shell)
    assert "pages/21_Neural_Network.py" in files
    assert "built-in-data/classification/nn_binary_400.csv" in files
    assert not any(rel.startswith("cv/") for rel in files)
    assert "pages/22_Image_Classification.py" not in files
    assert unresolved_internal_imports(shell) == []


def test_framework_plus_deep_learning_sidebar_shows_neural_network(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "deep-learning")

    assert sidebar(shell) == {"": ["總覽"], "深度學習": ["類神經網路"]}


def test_add_deep_learning_installs_tensorflow_only_then(tmp_path: Path) -> None:
    framework_runner = FakeDependencyRunner()
    install_shell(tmp_path, dependency_runner=framework_runner)
    assert not any("tensorflow" in spec for spec in framework_runner.added)

    runner = FakeDependencyRunner()
    add_packs(tmp_path, ["deep-learning"], dependency_runner=runner)

    assert "tensorflow-cpu" in runner.added
    for cv_only in ("torchvision", "ultralytics>=8.3.237", "opencv-python-headless"):
        assert cv_only not in runner.added


def test_deep_learning_dependencies_match_its_third_party_imports(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "deep-learning")
    module_to_dist = {
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "pandas": "pandas",
        "streamlit": "streamlit>=1.61",
        "tensorflow": "tensorflow-cpu",
    }

    used = third_party_imports(shell, pack_files(TEMPLATE_SHELL, DEEP_LEARNING))

    assert used <= set(module_to_dist), used
    needed = {module_to_dist[name] for name in used} - set(FRAMEWORK_DEPENDENCIES)
    assert set(DEEP_LEARNING.dependencies) == needed


def test_deep_learning_owns_its_tensorflow_runtime_copy() -> None:
    dl_files = set(pack_files(TEMPLATE_SHELL, DEEP_LEARNING))
    assert "ml/tf_runtime.py" in dl_files
    assert "built-in-data/classification/nn_binary_400.csv" in dl_files
    assert "ui/deep_learning_pack.py" in dl_files


def _template_imports() -> None:
    import sys

    templates = str(TEMPLATE_SHELL.parent)
    if templates not in sys.path:
        sys.path.insert(0, templates)


def test_deep_learning_host_is_framework_plus_one_pack_layer() -> None:
    _template_imports()
    from dataset_streamlit_shell.host_context import (
        assemble_host_context,
        framework_host_context,
    )
    from dataset_streamlit_shell.ui.deep_learning_pack import (
        deep_learning_host_context,
        deep_learning_pack_layer,
    )

    layer = deep_learning_pack_layer()
    host = deep_learning_host_context()

    assert host == assemble_host_context(layer)
    assert host.startswith(framework_host_context())
    assert host.count(layer) == 1
    assert host.count("課程包】") == 1


def test_deep_learning_layer_is_builtin_data_with_a_whitelist() -> None:
    _template_imports()
    from dataset_streamlit_shell.ui.deep_learning_pack import deep_learning_pack_layer

    layer = deep_learning_pack_layer()

    assert "nn_binary_400.csv" in layer
    assert "可寫清單" in layer
    assert "dataset_streamlit_shell/workspace/nn_form.json" in layer
    assert "dataset_streamlit_shell/workspace/nn_train_request.json" in layer
    for foreign in ("working.csv", "original.csv", "Working", "雙表", "cleaning_log", "challenge"):
        assert foreign not in layer, foreign
    # 讀檔習慣與腳本位置已在框架層
    assert "scripts" not in layer


def test_neural_network_page_host_adds_its_writeback_page_layer() -> None:
    _template_imports()
    from dataset_streamlit_shell.ui import nn_ui
    from dataset_streamlit_shell.ui.deep_learning_pack import (
        deep_learning_host_context,
        deep_learning_pack_layer,
    )
    from dataset_streamlit_shell.ui.nn_form_state import nn_host_context_fragment

    page_layer = nn_host_context_fragment(
        form_path="dataset_streamlit_shell/workspace/nn_form.json",
        request_path="dataset_streamlit_shell/workspace/nn_train_request.json",
        last_run_path="dataset_streamlit_shell/workspace/nn_last_run.json",
    )

    host = nn_ui._nn_host()

    assert host == deep_learning_host_context(page_layer)
    assert host.endswith(page_layer)
    assert host.count(deep_learning_pack_layer()) == 1
    assert "【教學頁】" not in host
