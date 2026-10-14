"""AI新創工作坊課程包（#16）：只裝框架 + startup-workshop 即可完整使用專案展示。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from pack_install_helpers import (
    TEMPLATE_SHELL,
    installed_files,
    installed_shell,
    sidebar,
    third_party_imports,
    unresolved_internal_imports,
)

from add_dataset_streamlit_shell.course_packs import STARTUP_WORKSHOP, pack_files
from add_dataset_streamlit_shell.installer import FRAMEWORK_DEPENDENCIES

COMPANIES = ("edupulse", "vitalrisk", "airsense", "churnlab", "flowcast")


def _import_lines(text: str) -> list[str]:
    """抽出文字（host 或空殼 docstring）裡給學生貼上的 dataset_streamlit_shell import 行。"""
    return re.findall(r"^\s*(from dataset_streamlit_shell\.\S+ import .+)$", text, re.MULTILINE)


def _missing_import_targets(shell: Path, lines: list[str]) -> list[str]:
    files = installed_files(shell)
    missing: list[str] = []
    for line in lines:
        node = ast.parse(line.strip()).body[0]
        assert isinstance(node, ast.ImportFrom)
        rel = node.module.removeprefix("dataset_streamlit_shell.").replace(".", "/") + ".py"
        if rel not in files:
            missing.append(node.module)
    return missing


def test_framework_plus_workshop_resolves_every_internal_import(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "startup-workshop")

    assert unresolved_internal_imports(shell) == []
    assert sidebar(shell) == {"": ["總覽"], "AI新創工作坊": ["專案展示"]}


def test_empty_shell_import_instructions_resolve_without_supervised(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "startup-workshop")
    files = installed_files(shell)

    # 監督式的回歸／分類模組不隨工作坊安裝；工作坊自帶副本
    assert "ml/regression.py" not in files
    assert "ml/classification.py" not in files
    for rel in ("ui/startup_challenge_ui.py", "ui/startup_challenge_empty_shell.py"):
        lines = _import_lines((shell / rel).read_text(encoding="utf-8"))
        joined = "\n".join(lines)
        assert " gradient_descent_steps" in joined, rel
        assert "logistic_gradient_descent_steps" in joined, rel
        assert _missing_import_targets(shell, lines) == [], rel


def test_workshop_owns_every_company_start_csv_and_handbook() -> None:
    owned = set(pack_files(TEMPLATE_SHELL, STARTUP_WORKSHOP))

    for company in COMPANIES:
        assert f"workspace/challenge/{company}.csv" in owned
        assert f"workspace/challenge/{company}_資料說明書.md" in owned
    assert "workspace/challenge/README.md" in owned
    assert "pages/30_Startup_Challenge.py" in owned
    assert "ml/regression.py" not in owned
    assert "ml/classification.py" not in owned


def test_workshop_dependencies_match_its_third_party_imports(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "startup-workshop")
    module_to_dist = {
        "numpy": "numpy",
        "pandas": "pandas",
        "streamlit": "streamlit>=1.61",
    }

    used = third_party_imports(shell, pack_files(TEMPLATE_SHELL, STARTUP_WORKSHOP))

    assert used <= set(module_to_dist), used
    needed = {module_to_dist[name] for name in used} - set(FRAMEWORK_DEPENDENCIES)
    # host 允許學生在模型區／成果區用 pyplot，所以 matplotlib 也要隨包安裝
    assert set(STARTUP_WORKSHOP.dependencies) == needed | {"matplotlib"}


_SHELL = "dataset_streamlit_shell"


def _paths(company: str) -> dict[str, str]:
    challenge = f"{_SHELL}/workspace/challenge"
    return {
        "start_csv": f"{challenge}/{company}.csv",
        "handbook": f"{challenge}/{company}_資料說明書.md",
        "working_csv": f"{challenge}/{company}/working.csv",
        "train_csv": f"{challenge}/{company}/train.csv",
        "test_csv": f"{challenge}/{company}/test.csv",
        "scripts_dir": f"{_SHELL}/scripts",
        "live_ui": f"{_SHELL}/ui/startup_challenge_ui.py",
        "page_py": f"{_SHELL}/ui/startup_challenge_page.py",
        "empty_shell": f"{_SHELL}/ui/startup_challenge_empty_shell.py",
    }


def test_challenge_host_is_framework_plus_workshop_layer_plus_company_layer() -> None:
    from dataset_streamlit_shell.host_context import framework_host_context
    from dataset_streamlit_shell.ui.data_science import data_science_pack_layer
    from dataset_streamlit_shell.ui.startup_challenge_context import (
        challenge_company_page_layer,
        challenge_host_context,
        startup_workshop_pack_layer,
    )

    paths = _paths("edupulse")
    host = challenge_host_context(company="edupulse", **paths)
    pack_layer = startup_workshop_pack_layer(**paths)
    page_layer = challenge_company_page_layer("edupulse")

    assert host.startswith(framework_host_context())
    assert host.count(pack_layer) == 1
    assert host.index(pack_layer) < host.index(page_layer)
    assert host.rstrip().endswith(page_layer.rstrip())
    assert "【EduPulse 加碼】" in page_layer
    assert data_science_pack_layer() not in host
    assert "【VitalRisk 加碼】" not in host


def test_workshop_layer_is_a_whitelist_of_track_files_without_company_fragment() -> None:
    from dataset_streamlit_shell.ui.startup_challenge_context import (
        startup_workshop_pack_layer,
    )

    paths = _paths("airsense")
    layer = startup_workshop_pack_layer(**paths)

    assert "可寫清單" in layer
    whitelist = layer.split("可寫清單", 1)[1].split("【", 1)[0]
    for key in ("live_ui", "working_csv", "train_csv", "test_csv"):
        assert paths[key] in whitelist, key
    assert paths["start_csv"] not in whitelist
    assert paths["empty_shell"] not in whitelist
    assert paths["page_py"] not in whitelist
    assert "加碼" not in layer
    assert "-200" not in layer


def test_company_page_layer_carries_each_company_fragment() -> None:
    from dataset_streamlit_shell.ui.startup_challenge_context import (
        challenge_company_page_layer,
    )

    assert "G1" in challenge_company_page_layer("edupulse")
    assert "不是診斷" in challenge_company_page_layer("vitalrisk")
    assert "-200" in challenge_company_page_layer("airsense")
    assert "不做清單" in challenge_company_page_layer("churnlab")
    assert "casual/registered" in challenge_company_page_layer("flowcast")
    for company in COMPANIES:
        assert company in challenge_company_page_layer(company)


def test_workshop_layer_import_instructions_resolve_without_supervised(
    tmp_path: Path,
) -> None:
    from dataset_streamlit_shell.ui.startup_challenge_context import (
        startup_workshop_pack_layer,
    )

    shell = installed_shell(tmp_path, "startup-workshop")
    lines = _import_lines(startup_workshop_pack_layer(**_paths("edupulse")))

    joined = "\n".join(lines)
    assert " gradient_descent_steps" in joined
    assert "logistic_gradient_descent_steps" in joined
    assert _missing_import_targets(shell, lines) == []
