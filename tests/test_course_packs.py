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


def test_readme_pack_table_matches_manifest() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for pack in COURSE_PACKS:
        row = next(
            (line for line in readme.splitlines() if line.startswith(f"| `{pack.name}` |")),
            None,
        )
        assert row is not None, pack.name
        assert pack.sidebar_section in row, pack.name
    assert "uv add --upgrade-package" not in readme


def test_every_template_file_belongs_to_at_most_one_course_pack() -> None:
    owners: dict[str, list[str]] = {}
    for pack in COURSE_PACKS:
        for rel in pack_files(SHELL_ROOT, pack):
            owners.setdefault(rel, []).append(pack.name)
    shared = {rel: names for rel, names in owners.items() if len(names) > 1}
    assert shared == {}


def _app_sections() -> dict[str, list[str]]:
    """app.py 的 SECTIONS：分段名稱 → 頁檔路徑（只取字串頁；總覽是函式頁）。"""
    tree = ast.parse((SHELL_ROOT / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SECTIONS" for t in node.targets
        ):
            assert isinstance(node.value, ast.Dict)
            sections: dict[str, list[str]] = {}
            for key, value in zip(node.value.keys, node.value.values):
                assert isinstance(key, ast.Constant) and isinstance(value, ast.List)
                sections[key.value] = [
                    call.args[0].value
                    for call in value.elts
                    if isinstance(call, ast.Call)
                    and isinstance(call.args[0], ast.Constant)
                ]
            return sections
    raise AssertionError("app.py has no SECTIONS")


def test_app_sections_match_course_pack_manifest() -> None:
    """側欄每一頁屬於剛好一個課程包，且分段名稱等於該包的 sidebar_section；包內每頁都在側欄。"""
    sections = _app_sections()
    listed: set[str] = set()
    for section, pages in sections.items():
        if section == "":
            assert pages == [], "總覽分段只放框架的函式頁"
            continue
        for page in pages:
            owners = [pack for pack in COURSE_PACKS if page in pack_files(SHELL_ROOT, pack)]
            assert len(owners) == 1, f"{page} 屬於 {[p.name for p in owners]}"
            assert owners[0].sidebar_section == section, f"{page}: {section}"
            listed.add(page)
    for pack in COURSE_PACKS:
        for rel in pack_files(SHELL_ROOT, pack):
            if rel.startswith("pages/"):
                assert rel in listed, f"{pack.name}: {rel} 不在 SECTIONS"


def _imports_plotting(rel: str) -> bool:
    return any(module == "dataset_streamlit_shell.plotting" for module, _ in _internal_imports(rel))


def test_packs_using_framework_plotting_declare_matplotlib() -> None:
    """plotting 是框架、但只在函式內懶載入 matplotlib；用到它的課程包要自己宣告 matplotlib。"""
    users = []
    for pack in COURSE_PACKS:
        if any(_imports_plotting(rel) for rel in pack_files(SHELL_ROOT, pack) if rel.endswith(".py")):
            users.append(pack.name)
            assert "matplotlib" in pack.dependencies, pack.name
    assert users  # 確實有課程包用到 plotting


def test_plotting_module_imports_without_matplotlib() -> None:
    """只裝框架（沒有 matplotlib）時，import plotting 頂層不會失敗。"""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['matplotlib'] = None\n"
        "import dataset_streamlit_shell.plotting\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SHELL_ROOT.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_shell_root_is_defined_only_in_host_context() -> None:
    """框架與 ui/ 只有一個 SHELL_ROOT 定義（host_context），其他模組從那裡 import。"""
    candidates = [
        rel
        for rel in _template_files()
        if rel.endswith(".py") and (rel.startswith("ui/") or "/" not in rel)
    ]
    definers = []
    for rel in candidates:
        tree = ast.parse((SHELL_ROOT / rel).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SHELL_ROOT" for t in node.targets
            ):
                definers.append(rel)
    assert definers == ["host_context.py"]


def test_each_pack_internal_imports_only_reach_framework_or_itself() -> None:
    framework = set(framework_files(SHELL_ROOT))
    template = _template_files()
    for pack in COURSE_PACKS:
        own = set(pack_files(SHELL_ROOT, pack))
        for rel in sorted(own):
            if not rel.endswith(".py"):
                continue
            for target in _resolved_targets(rel, template):
                assert target in framework or target in own, (
                    f"{pack.name}: {rel} → {target}"
                )
