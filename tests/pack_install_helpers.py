"""課程包安裝測試共用工具（#11–#16）。

- ``FakeDependencyRunner``：取代 ``subprocess.run``，記錄每次 ``uv add`` 的參數。
- ``installed_shell(project_root, *packs)``：把框架（與指定課程包）裝進 tmp 專案，
  不真的安裝套件，回傳殼層資料夾。
- ``unresolved_internal_imports(shell_dir)``：用 ``ast`` 靜態解析已安裝殼層內每個
  ``.py`` 的 ``dataset_streamlit_shell.*`` import，回傳找不到檔案的項目（空 = 全可解析）。
- ``installed_files(shell_dir)``：殼層內所有檔案的相對路徑（posix）。
- ``sidebar(shell_dir)``：已安裝殼層的側欄分段 → 頁面標題（不啟動 Streamlit）。
- ``third_party_imports(shell_dir, rels)``：指定檔案用到的第三方頂層模組名。
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from pathlib import Path

from add_dataset_streamlit_shell.installer import add_packs, install_shell

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1] / "src" / "add_dataset_streamlit_shell" / "templates"
)
TEMPLATE_SHELL = TEMPLATE_ROOT / "dataset_streamlit_shell"
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

INTERNAL_PREFIX = "dataset_streamlit_shell."


class FakeDependencyRunner:
    """假的依賴安裝器：記錄呼叫，不執行任何指令。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], *, cwd: Path, check: bool) -> None:
        assert check is True
        self.calls.append(list(args))

    @property
    def added(self) -> list[str]:
        """所有 ``uv add`` 呼叫中的套件規格（去掉 ``uv add`` 與旗標）。"""
        specs: list[str] = []
        for args in self.calls:
            assert args[:2] == ["uv", "add"], args
            rest = args[2:]
            index = 0
            while index < len(rest):
                if rest[index] == "--upgrade-package":
                    index += 2
                    continue
                if rest[index].startswith("--"):
                    index += 1
                    continue
                specs.append(rest[index])
                index += 1
        return specs


def installed_shell(
    project_root: Path,
    *packs: str,
    runner: FakeDependencyRunner | None = None,
) -> Path:
    """裝框架（並 ``add`` 指定課程包）進 project_root，回傳殼層資料夾。"""
    runner = runner if runner is not None else FakeDependencyRunner()
    result = install_shell(project_root, dependency_runner=runner)
    if packs:
        add_packs(project_root, list(packs), dependency_runner=runner)
    return result.target


def installed_files(shell_dir: Path) -> set[str]:
    return {
        path.relative_to(shell_dir).as_posix()
        for path in shell_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def sidebar(shell_dir: Path) -> dict[str, list[str]]:
    """已安裝殼層的側欄：分段名稱 → 頁面標題（只含存在的頁，空分段剔除）。

    從已安裝的 ``app.py`` 用 ``ast`` 取出 ``SECTIONS``（不啟動 Streamlit），
    再交給殼層自己的 ``navigation.installed_sections`` 過濾。
    """
    from dataset_streamlit_shell.navigation import NavPage, installed_sections

    tree = ast.parse((shell_dir / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SECTIONS" for t in node.targets
        ):
            expr = ast.Expression(node.value)
            break
    else:
        raise AssertionError("app.py has no SECTIONS")

    class _Callables(dict):
        def __missing__(self, key: str) -> object:
            return lambda: None  # 非檔案頁（例如總覽）

    namespace = _Callables(NavPage=NavPage)
    sections = eval(compile(expr, "app.py", "eval"), {"__builtins__": {}}, namespace)
    return {
        section: [page.title for page in pages]
        for section, pages in installed_sections(shell_dir, sections).items()
    }


def third_party_imports(shell_dir: Path, rels: Iterable[str]) -> set[str]:
    """指定檔案 import 的第三方頂層模組名（排除標準庫與殼層內部）。"""
    modules: set[str] = set()
    for rel in rels:
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((shell_dir / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            modules.update(name.split(".")[0] for name in names)
    return {
        name
        for name in modules
        if name not in sys.stdlib_module_names
        and name != "__future__"
        and name != "dataset_streamlit_shell"
    }


def _internal_imports(source: str) -> list[tuple[str, list[str]]]:
    imports: list[tuple[str, list[str]]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            INTERNAL_PREFIX
        ):
            imports.append((node.module, [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(INTERNAL_PREFIX):
                    imports.append((alias.name, []))
    return imports


def _missing_targets(module: str, names: list[str], files: set[str]) -> list[str]:
    path = module[len(INTERNAL_PREFIX):].replace(".", "/")
    if f"{path}.py" in files:
        return []
    if f"{path}/__init__.py" not in files:
        return [module]
    # from package import name：name 可能是子模組或 __init__ 內的名稱；
    # 模板的 package __init__ 不定義名稱，所以一律視為子模組。
    return [
        f"{module}.{name}"
        for name in names
        if f"{path}/{name}.py" not in files and f"{path}/{name}/__init__.py" not in files
    ]


def unresolved_internal_imports(shell_dir: Path) -> list[str]:
    """回傳 ``"<file> -> <module>"``，列出找不到對應檔案的內部 import。"""
    files = installed_files(shell_dir)
    problems: list[str] = []
    for rel in sorted(files):
        if not rel.endswith(".py"):
            continue
        source = (shell_dir / rel).read_text(encoding="utf-8")
        for module, names in _internal_imports(source):
            problems.extend(
                f"{rel} -> {missing}"
                for missing in _missing_targets(module, names, files)
            )
    return problems
