"""安裝器課程包化（#11）：只看裝出了什麼、紀錄內容與 uv add 參數。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pack_install_helpers import (
    TEMPLATE_SHELL,
    FakeDependencyRunner,
    installed_files,
    installed_shell,
    sidebar,
    third_party_imports,
    unresolved_internal_imports,
)

from add_dataset_streamlit_shell.course_packs import (
    COURSE_PACKS,
    DATA_SCIENCE,
    UnknownCoursePackError,
    pack_files,
)
from add_dataset_streamlit_shell.installer import (
    FRAMEWORK_DEPENDENCIES,
    InstallRecordMissingError,
    add_packs,
    course_pack_status,
    install_shell,
)

RECORD = "course_packs.json"


def _record(shell: Path) -> list[str]:
    return json.loads((shell / RECORD).read_text(encoding="utf-8"))["course_packs"]


def _all_pack_files() -> set[str]:
    return {rel for pack in COURSE_PACKS for rel in pack_files(TEMPLATE_SHELL, pack)}


def test_bare_install_copies_only_the_framework(tmp_path: Path) -> None:
    runner = FakeDependencyRunner()
    shell = install_shell(tmp_path, dependency_runner=runner).target

    files = installed_files(shell)
    assert "app.py" in files
    assert "ui/data_ui.py" in files
    assert not files & _all_pack_files()
    assert not any(rel.startswith("pages/") for rel in files)
    assert unresolved_internal_imports(shell) == []
    assert _record(shell) == []


def test_bare_install_adds_only_framework_dependencies(tmp_path: Path) -> None:
    runner = FakeDependencyRunner()
    install_shell(tmp_path, dependency_runner=runner)

    added = runner.added
    assert "streamlit>=1.61" in added
    assert "pandas" in added
    assert any(spec.startswith("openai-tts") for spec in added)
    assert any("peas-agent-runtime" in spec for spec in added)
    assert "charset-normalizer>=3.4.9" in added
    for heavy in ("matplotlib", "scikit-learn", "xgboost", "tensorflow-cpu", "torchvision"):
        assert heavy not in added


def test_add_data_science_installs_its_pages_and_dependencies(tmp_path: Path) -> None:
    install_shell(tmp_path, dependency_runner=FakeDependencyRunner())
    runner = FakeDependencyRunner()

    result = add_packs(tmp_path, ["data-science"], dependency_runner=runner)

    shell = result.target
    files = installed_files(shell)
    assert set(pack_files(TEMPLATE_SHELL, DATA_SCIENCE)) <= files
    assert "pages/3_Field_Quality.py" in files
    assert "pages/14_Simple_Linear_Regression.py" not in files
    assert unresolved_internal_imports(shell) == []
    assert _record(shell) == ["data-science"]
    assert runner.calls == [["uv", "add", "matplotlib", "scikit-learn"]]
    assert result.added == ("data-science",)


def test_data_science_dependencies_match_its_third_party_imports(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "data-science")
    module_to_dist = {
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "pandas": "pandas",
        "streamlit": "streamlit>=1.61",
        "numpy": "numpy",
    }

    used = third_party_imports(shell, pack_files(TEMPLATE_SHELL, DATA_SCIENCE))

    assert used <= set(module_to_dist), used
    needed = {module_to_dist[name] for name in used} - set(FRAMEWORK_DEPENDENCIES)
    assert set(DATA_SCIENCE.dependencies) == needed


def test_framework_only_sidebar_shows_only_overview(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path)

    assert sidebar(shell) == {"": ["總覽"]}


def test_framework_plus_data_science_sidebar_shows_overview_and_data_science(
    tmp_path: Path,
) -> None:
    shell = installed_shell(tmp_path, "data-science")

    sections = sidebar(shell)

    assert list(sections) == ["", "AI 協作資料整理"]
    assert sections["AI 協作資料整理"][0] == "欄位與資料整合"
    assert len(sections["AI 協作資料整理"]) == 11


def test_import_resolver_reports_a_missing_internal_module(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "data-science")
    (shell / "ui" / "data_science.py").unlink()

    problems = unresolved_internal_imports(shell)

    assert any(p.endswith("dataset_streamlit_shell.ui.data_science") for p in problems)


def test_add_all_installs_every_pack(tmp_path: Path) -> None:
    install_shell(tmp_path, dependency_runner=FakeDependencyRunner())
    runner = FakeDependencyRunner()

    result = add_packs(tmp_path, ["all"], dependency_runner=runner)

    assert _record(result.target) == [pack.name for pack in COURSE_PACKS]
    assert _all_pack_files() <= installed_files(result.target)
    assert len(runner.calls) == 1
    assert {"xgboost", "tensorflow-cpu", "torchvision"} <= set(runner.added)
    assert len(runner.added) == len(set(runner.added))


def test_add_several_names_records_them_in_manifest_order(tmp_path: Path) -> None:
    install_shell(tmp_path, dependency_runner=FakeDependencyRunner())
    runner = FakeDependencyRunner()

    result = add_packs(
        tmp_path, ["unsupervised", "data-science"], dependency_runner=runner
    )

    assert _record(result.target) == ["data-science", "unsupervised"]
    files = installed_files(result.target)
    assert "pages/3_Field_Quality.py" in files
    assert "pages/12_KMeans.py" in files
    assert runner.calls == [["uv", "add", "matplotlib", "scikit-learn", "scipy"]]


def test_add_unknown_name_lists_available_names_and_changes_nothing(
    tmp_path: Path,
) -> None:
    shell = installed_shell(tmp_path)
    before = installed_files(shell)
    runner = FakeDependencyRunner()

    with pytest.raises(UnknownCoursePackError) as excinfo:
        add_packs(tmp_path, ["data-science", "datascience"], dependency_runner=runner)

    message = str(excinfo.value)
    assert "datascience" in message
    for pack in COURSE_PACKS:
        assert pack.name in message
    assert "all" in message
    assert installed_files(shell) == before
    assert runner.calls == []


def test_add_already_installed_pack_copies_nothing_and_adds_no_dependency(
    tmp_path: Path,
) -> None:
    shell = installed_shell(tmp_path, "data-science")
    page = shell / "pages" / "3_Field_Quality.py"
    page.write_text("# student edit\n", encoding="utf-8")
    runner = FakeDependencyRunner()

    result = add_packs(tmp_path, ["data-science"], dependency_runner=runner)

    assert result.added == ()
    assert result.already_installed == ("data-science",)
    assert page.read_text(encoding="utf-8") == "# student edit\n"
    assert runner.calls == []
    assert _record(shell) == ["data-science"]


def test_add_into_empty_project_installs_framework_first(tmp_path: Path) -> None:
    runner = FakeDependencyRunner()

    result = add_packs(tmp_path, ["data-science"], dependency_runner=runner)

    assert result.framework_installed is True
    assert (result.target / "app.py").is_file()
    assert (result.target / "pages" / "3_Field_Quality.py").is_file()
    assert _record(result.target) == ["data-science"]
    assert runner.calls[0][:4] == ["uv", "add", "--upgrade-package", "openai-tts"]
    assert runner.calls[-1] == ["uv", "add", "matplotlib", "scikit-learn"]
    assert unresolved_internal_imports(result.target) == []


def test_add_without_install_deps_runs_no_dependency_installer(tmp_path: Path) -> None:
    runner = FakeDependencyRunner()

    result = add_packs(
        tmp_path,
        ["data-science"],
        install_dependencies=False,
        dependency_runner=runner,
    )

    assert runner.calls == []
    assert (result.target / "pages" / "3_Field_Quality.py").is_file()


def test_add_leaves_preserved_directories_untouched(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path)
    keep = {
        "workspace/working.csv": "a\n2\n",
        "workspace/challenge/vitalrisk.csv": "student-edited\n",
        "scripts/student.py": "# keep\n",
        "uploads/chat_images/x.png": "png",
        "memory/notes.jsonl": "{}\n",
        "sessions/session_old.jsonl": "{}\n",
    }
    for rel, text in keep.items():
        (shell / rel).parent.mkdir(parents=True, exist_ok=True)
        (shell / rel).write_text(text, encoding="utf-8")

    add_packs(tmp_path, ["all"], dependency_runner=FakeDependencyRunner())

    for rel, text in keep.items():
        assert (shell / rel).read_text(encoding="utf-8") == text, rel
    # 不存在的挑戰起始檔照樣補上
    assert (shell / "workspace" / "challenge" / "airsense.csv").is_file()


def test_update_recopies_framework_and_recorded_packs_only(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "data-science")
    (shell / "app.py").write_text("# stale\n", encoding="utf-8")
    (shell / "pages" / "3_Field_Quality.py").write_text("# stale\n", encoding="utf-8")
    (shell / "stale_module.py").write_text("# old\n", encoding="utf-8")
    (shell / "workspace" / "working.csv").write_text("a\n2\n", encoding="utf-8")
    runner = FakeDependencyRunner()

    result = install_shell(tmp_path, update=True, dependency_runner=runner)

    files = installed_files(shell)
    assert (shell / "app.py").read_text(encoding="utf-8") != "# stale\n"
    assert (shell / "pages" / "3_Field_Quality.py").read_text(encoding="utf-8") != "# stale\n"
    assert "stale_module.py" not in files
    assert "pages/14_Simple_Linear_Regression.py" not in files
    assert (shell / "workspace" / "working.csv").read_text(encoding="utf-8") == "a\n2\n"
    assert _record(shell) == ["data-science"]
    assert result.course_packs == ("data-science",)
    assert runner.calls[-1] == ["uv", "add", "matplotlib", "scikit-learn"]
    assert "xgboost" not in runner.added


def test_update_without_install_record_errors_and_suggests_force(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path)
    (shell / RECORD).unlink()
    (shell / "app.py").write_text("# old install\n", encoding="utf-8")
    runner = FakeDependencyRunner()

    with pytest.raises(InstallRecordMissingError, match="--force"):
        install_shell(tmp_path, update=True, dependency_runner=runner)

    assert (shell / "app.py").read_text(encoding="utf-8") == "# old install\n"
    assert runner.calls == []


def test_force_reinstalls_recorded_packs_and_restores_challenge_data(
    tmp_path: Path,
) -> None:
    shell = installed_shell(tmp_path, "data-science", "startup-workshop")
    company = shell / "workspace" / "challenge" / "vitalrisk"
    company.mkdir(parents=True, exist_ok=True)
    (company / "working.csv").write_text("student-working\n", encoding="utf-8")
    runner = FakeDependencyRunner()

    result = install_shell(tmp_path, force=True, dependency_runner=runner)

    assert result.backed_up_to is not None and result.backed_up_to.is_dir()
    assert _record(shell) == ["data-science", "startup-workshop"]
    files = installed_files(shell)
    assert "pages/3_Field_Quality.py" in files
    assert "pages/30_Startup_Challenge.py" in files
    assert "pages/14_Simple_Linear_Regression.py" not in files
    assert (company / "working.csv").read_text(encoding="utf-8") == "student-working\n"
    assert "matplotlib" in runner.added
    assert "xgboost" not in runner.added


def test_force_without_install_record_installs_framework_only(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "data-science")
    (shell / RECORD).unlink()

    result = install_shell(tmp_path, force=True, dependency_runner=FakeDependencyRunner())

    assert result.backed_up_to is not None
    assert not installed_files(shell) & _all_pack_files()
    assert _record(shell) == []


def test_course_pack_status_lists_every_pack_with_installed_flag(tmp_path: Path) -> None:
    assert all(not installed for _, installed in course_pack_status(tmp_path))
    installed_shell(tmp_path, "data-science")

    status = course_pack_status(tmp_path)

    assert [pack.name for pack, _ in status] == [pack.name for pack in COURSE_PACKS]
    assert [installed for _, installed in status] == [
        pack.name == "data-science" for pack in COURSE_PACKS
    ]
