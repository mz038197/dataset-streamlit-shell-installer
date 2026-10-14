from __future__ import annotations

import ast
from pathlib import Path

from add_dataset_streamlit_shell.course_packs import (
    COURSE_PACKS,
    framework_files,
    pack_files,
)

SHELL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
)
INTERNAL_PREFIX = "dataset_streamlit_shell."


def test_manifest_lists_the_seven_course_packs_in_sidebar_order() -> None:
    assert [pack.name for pack in COURSE_PACKS] == [
        "data-science",
        "supervised",
        "unsupervised",
        "deep-learning",
        "computer-vision",
        "pca",
        "startup-workshop",
    ]
    sections = {pack.name: pack.sidebar_section for pack in COURSE_PACKS}
    assert sections["data-science"] == "AI 協作資料整理"
    assert sections["startup-workshop"] == "AI新創工作坊"
    for pack in COURSE_PACKS:
        assert pack.sidebar_section
        assert pack.paths


def test_every_manifest_path_exists_in_template() -> None:
    for pack in COURSE_PACKS:
        for rel in pack.paths:
            assert (SHELL_ROOT / rel).exists(), f"{pack.name}: {rel}"


def test_example_data_follows_the_pack_that_uses_it() -> None:
    owners = {pack.name: set(pack_files(SHELL_ROOT, pack)) for pack in COURSE_PACKS}
    assert "built-in-data/integration/titanic_passengers.csv" in owners["data-science"]
    assert "built-in-data/classification/nn_binary_400.csv" in owners["deep-learning"]
    assert "built-in-data/clustering/clustering_blobs_90.csv" in owners["unsupervised"]
    assert "built-in-data/computer-vision/examples/dog.jpg" in owners["computer-vision"]
    assert "workspace/challenge/vitalrisk.csv" in owners["startup-workshop"]


def test_framework_is_every_template_file_outside_the_packs() -> None:
    framework = set(framework_files(SHELL_ROOT))
    assert "app.py" in framework
    assert "host_context.py" in framework
    assert "ui/data_ui.py" in framework
    assert "ui/overview_ui.py" in framework
    assert "pages/3_Field_Quality.py" not in framework
    assert "ui/workflow_ui.py" not in framework
    assert not any("__pycache__" in rel for rel in framework)


def test_extra_dependencies_split_by_pack() -> None:
    deps = {pack.name: pack.dependencies for pack in COURSE_PACKS}
    assert "xgboost" in deps["supervised"]
    assert "plotly" in deps["supervised"]
    assert "tensorflow-cpu" in deps["deep-learning"]
    assert "tensorflow-cpu" in deps["computer-vision"]
    assert "torchvision" in deps["computer-vision"]
    assert not any("torch" in dep for dep in deps["deep-learning"])
    assert "scikit-learn" in deps["data-science"]


def _template_files() -> set[str]:
    return {
        path.relative_to(SHELL_ROOT).as_posix()
        for path in SHELL_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def _internal_imports(rel: str) -> list[tuple[str, list[str]]]:
    tree = ast.parse((SHELL_ROOT / rel).read_text(encoding="utf-8"))
    imports: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(INTERNAL_PREFIX):
            imports.append((node.module, [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(INTERNAL_PREFIX):
                    imports.append((alias.name, []))
    return imports


def _resolved_targets(rel: str, template: set[str]) -> list[str]:
    """把 dataset_streamlit_shell.* 的 import 解析成模板內的檔案路徑。"""
    targets: list[str] = []
    for module, names in _internal_imports(rel):
        path = module[len(INTERNAL_PREFIX):].replace(".", "/")
        if f"{path}.py" in template:
            targets.append(f"{path}.py")
            continue
        package = f"{path}/__init__.py"
        if package not in template:
            targets.append(f"<missing {module}>")
            continue
        targets.append(package)
        targets.extend(
            f"{path}/{name}.py" for name in names if f"{path}/{name}.py" in template
        )
    return targets


def test_framework_internal_imports_only_reach_framework_files() -> None:
    framework = set(framework_files(SHELL_ROOT))
    template = _template_files()
    for rel in sorted(framework):
        if not rel.endswith(".py"):
            continue
        for target in _resolved_targets(rel, template):
            assert target in framework, f"{rel} → {target}"
