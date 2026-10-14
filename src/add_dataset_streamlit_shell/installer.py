from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from add_dataset_streamlit_shell.course_packs import (
    COURSE_PACKS,
    CoursePack,
    framework_files,
    pack_files,
    resolve_course_packs,
)


SHELL_DIR_NAME = "dataset_streamlit_shell"
# sessions 仍保留：舊版裝在 shell 內的對話檔，--update 時不可刪。
# 新版對話檔寫在專案根 sessions/（見 _ensure_project_sessions_dir）。
PRESERVE_ON_UPDATE = frozenset(
    {"workspace", "sessions", "scripts", "uploads", "memory"}
)
_CHALLENGE_RUNTIME_FILE_NAMES = frozenset(
    {
        "working.csv",
        "train.csv",
        "test.csv",
        "startup_challenge_ui.py",
        "model_artifact.json",
        "committed_company",
    }
)
# AI Lab 框架的套件；課程包的額外套件見 course_packs.py。
FRAMEWORK_DEPENDENCIES = (
    "streamlit>=1.61",
    "pandas",
    "openai-tts @ git+https://github.com/mz038197/openai-tts.git",
)
# 安裝紀錄：殼層資料夾內，記錄已安裝的課程包名稱（ADR 0029）。
INSTALL_RECORD_NAME = "course_packs.json"
# requests 間接依賴；避開 PyPI yanked 的 3.4.8
CHARSET_NORMALIZER_CONSTRAINT = "charset-normalizer>=3.4.9"
RUNTIME_DEPENDENCY = (
    "peas-agent-runtime @ git+https://github.com/mz038197/peas-agent-runtime.git"
)
DependencyRunner = Callable[[list[str]], None]


class InstallRecordMissingError(FileNotFoundError):
    """殼層存在但沒有安裝紀錄（舊版安裝；ADR 0029 不相容舊安裝）。"""


@dataclass(frozen=True)
class InstallResult:
    target: Path
    backed_up_to: Path | None = None
    installed_dependencies: tuple[str, ...] = ()
    contract_ok: bool | None = None
    contract_messages: tuple[str, ...] = ()
    factory_ref: str | None = None
    course_packs: tuple[str, ...] = ()


@dataclass(frozen=True)
class AddResult:
    target: Path
    added: tuple[str, ...] = ()
    already_installed: tuple[str, ...] = ()
    installed_dependencies: tuple[str, ...] = ()
    framework_installed: bool = False


def _resolve_runtime_path(project_root: Path) -> Path | None:
    sibling = (project_root.parent / "peas-agent-runtime").resolve()
    if sibling.is_dir() and (sibling / "pyproject.toml").exists():
        return sibling
    return None


def _template():
    return resources.files("add_dataset_streamlit_shell").joinpath(
        "templates",
        SHELL_DIR_NAME,
    )


def _check_project_root(project_root: Path) -> Path:
    project_root = project_root.resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")
    return project_root


def read_install_record(target: Path) -> tuple[CoursePack, ...] | None:
    """殼層已安裝的課程包（清單順序）；沒有安裝紀錄回傳 None。"""
    record = target / INSTALL_RECORD_NAME
    if not record.is_file():
        return None
    data = json.loads(record.read_text(encoding="utf-8"))
    return resolve_course_packs(data.get("course_packs", []))


def _require_install_record(target: Path) -> tuple[CoursePack, ...]:
    packs = read_install_record(target)
    if packs is None:
        raise InstallRecordMissingError(
            f"{SHELL_DIR_NAME}/{INSTALL_RECORD_NAME} was not found "
            "(shell installed by an older version). "
            "Re-run with --force to back it up and reinstall."
        )
    return packs


def _write_install_record(target: Path, packs: Iterable[CoursePack]) -> None:
    chosen = set(packs)
    names = [pack.name for pack in COURSE_PACKS if pack in chosen]
    (target / INSTALL_RECORD_NAME).write_text(
        json.dumps({"course_packs": names}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def course_pack_status(project_root: Path) -> list[tuple[CoursePack, bool]]:
    """每個課程包與是否已安裝（未裝殼層或沒有安裝紀錄時皆為未安裝）。"""
    target = project_root.resolve() / SHELL_DIR_NAME
    installed = set(read_install_record(target) or ())
    return [(pack, pack in installed) for pack in COURSE_PACKS]


def install_shell(
    project_root: Path,
    *,
    force: bool = False,
    update: bool = False,
    require_agent_core: bool = False,
    require_agent_contract: bool = False,
    install_dependencies: bool = True,
    dependency_runner: Callable[..., object] = subprocess.run,
) -> InstallResult:
    """安裝 AI Lab 框架；--update／--force 依安裝紀錄一併帶上已裝的課程包。"""
    project_root = _check_project_root(project_root)

    if require_agent_core and not (project_root / "agent_core.py").exists():
        raise FileNotFoundError(
            "agent_core.py was not found. Dataset Shell 已改為 create_agent；"
            "請改用 --require-agent-contract，或省略此旗標。"
        )

    target = project_root / SHELL_DIR_NAME
    backup: Path | None = None
    packs: tuple[CoursePack, ...] = ()
    with resources.as_file(_template()) as template_path:
        if target.exists() and update:
            packs = _require_install_record(target)
            _update_existing(target, template_path, packs)
        else:
            if target.exists():
                if not force:
                    raise FileExistsError(
                        f"{SHELL_DIR_NAME}/ already exists. "
                        "Re-run with --force to replace it."
                    )
                packs = read_install_record(target) or ()
                backup = _backup_existing(target)
            _copy_shell_files(template_path, target, packs)
    if backup is not None:
        _restore_challenge_runtime(backup, target)
    _write_install_record(target, packs)
    _ensure_project_sessions_dir(project_root)

    installed = (
        _install_dependencies(project_root, dependency_runner, packs)
        if install_dependencies
        else ()
    )
    contract_ok, contract_messages, factory_ref = _check_contract(project_root)
    if require_agent_contract and not contract_ok:
        raise FileNotFoundError(
            "Agent contract check failed:\n" + "\n".join(contract_messages)
        )
    return InstallResult(
        target=target,
        backed_up_to=backup,
        installed_dependencies=installed,
        contract_ok=contract_ok,
        contract_messages=contract_messages,
        factory_ref=factory_ref,
        course_packs=tuple(pack.name for pack in packs),
    )


def add_packs(
    project_root: Path,
    names: Sequence[str],
    *,
    install_dependencies: bool = True,
    dependency_runner: Callable[..., object] = subprocess.run,
) -> AddResult:
    """加裝課程包：必要時先裝框架；已裝的略過；只新增檔案，不動保留目錄裡既有的檔。"""
    project_root = _check_project_root(project_root)
    packs = resolve_course_packs(names)
    target = project_root / SHELL_DIR_NAME

    installed: list[str] = []
    framework_installed = False
    if not target.exists():
        framework = install_shell(
            project_root,
            install_dependencies=install_dependencies,
            dependency_runner=dependency_runner,
        )
        installed.extend(framework.installed_dependencies)
        framework_installed = True

    recorded = _require_install_record(target)
    new = tuple(pack for pack in packs if pack not in recorded)
    already = tuple(pack for pack in packs if pack in recorded)
    if new:
        with resources.as_file(_template()) as template_path:
            for pack in new:
                _copy_files(
                    template_path,
                    target,
                    pack_files(template_path, pack),
                    overwrite_preserved=False,
                )
        _write_install_record(target, (*recorded, *new))
        if install_dependencies:
            installed.extend(
                _install_pack_dependencies(project_root, dependency_runner, new)
            )

    return AddResult(
        target=target,
        added=tuple(pack.name for pack in new),
        already_installed=tuple(pack.name for pack in already),
        installed_dependencies=tuple(installed),
        framework_installed=framework_installed,
    )


def _copy_files(
    template_path: Path,
    target: Path,
    rels: Iterable[str],
    *,
    overwrite_preserved: bool,
) -> None:
    for rel in rels:
        dest = target / rel
        if (
            not overwrite_preserved
            and Path(rel).parts[0] in PRESERVE_ON_UPDATE
            and dest.exists()
        ):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template_path / rel, dest)


def _copy_shell_files(
    template_path: Path, target: Path, packs: Iterable[CoursePack]
) -> None:
    """複製框架與指定課程包的模板檔（模板內有的檔一律覆蓋）。"""
    target.mkdir(parents=True, exist_ok=True)
    _copy_files(
        template_path,
        target,
        framework_files(template_path),
        overwrite_preserved=True,
    )
    for pack in packs:
        _copy_files(
            template_path,
            target,
            pack_files(template_path, pack),
            overwrite_preserved=True,
        )


def _ensure_project_sessions_dir(project_root: Path) -> None:
    """對話 JSONL 存專案根 sessions/（與 main_shell 預設對齊）。"""
    sessions = project_root / "sessions"
    sessions.mkdir(exist_ok=True)
    gitkeep = sessions / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")


def _install_pack_dependencies(
    project_root: Path,
    dependency_runner: Callable[..., object],
    packs: Iterable[CoursePack],
) -> tuple[str, ...]:
    dependencies = tuple(
        dict.fromkeys(dep for pack in packs for dep in pack.dependencies)
    )
    if dependencies:
        dependency_runner(
            ["uv", "add", *dependencies],
            cwd=project_root,
            check=True,
        )
    return dependencies


def _install_dependencies(
    project_root: Path,
    dependency_runner: Callable[..., object],
    packs: Iterable[CoursePack] = (),
) -> tuple[str, ...]:
    dependency_runner(
        ["uv", "add", "--upgrade-package", "openai-tts", *FRAMEWORK_DEPENDENCIES],
        cwd=project_root,
        check=True,
    )
    runtime_path = _resolve_runtime_path(project_root)
    if runtime_path is not None:
        dependency_runner(
            ["uv", "add", "--editable", str(runtime_path)],
            cwd=project_root,
            check=True,
        )
        runtime_label = f"peas-agent-runtime @ {runtime_path}"
    else:
        dependency_runner(
            ["uv", "add", RUNTIME_DEPENDENCY],
            cwd=project_root,
            check=True,
        )
        runtime_label = RUNTIME_DEPENDENCY
    dependency_runner(
        ["uv", "add", CHARSET_NORMALIZER_CONSTRAINT],
        cwd=project_root,
        check=True,
    )
    pack_dependencies = _install_pack_dependencies(
        project_root, dependency_runner, packs
    )
    return (
        *FRAMEWORK_DEPENDENCIES,
        runtime_label,
        CHARSET_NORMALIZER_CONSTRAINT,
        *pack_dependencies,
    )


# 在專案 uv/.venv 內執行，避免安裝器全域 Python 找不到 peas-agent-runtime。
_CONTRACT_CHECK_SCRIPT = """\
import json
import sys
try:
    from peas_agent_runtime import check_agent_factory
except ImportError:
    print(json.dumps({
        "skip": True,
        "reason": "peas-agent-runtime 尚未可 import；略過契約檢查。",
    }))
    raise SystemExit(0)
factory_ref = sys.argv[1]
result = check_agent_factory(factory_ref, project_root=".")
print(json.dumps({
    "skip": False,
    "ok": result.ok,
    "messages": list(result.messages),
    "factory_ref": factory_ref,
}))
"""


def _parse_contract_payload(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _check_contract(
    project_root: Path,
) -> tuple[bool | None, tuple[str, ...], str | None]:
    """盡力檢查 create_agent；專案環境無 runtime 時回傳 None（不阻一般安裝）。"""
    for factory_ref in ("main_shell:create_agent", "main:create_agent"):
        module_name = factory_ref.split(":", 1)[0]
        if not (project_root / f"{module_name}.py").exists():
            continue
        try:
            proc = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "-c",
                    _CONTRACT_CHECK_SCRIPT,
                    factory_ref,
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None, ("無法執行 uv run；略過契約檢查。",), None

        payload = _parse_contract_payload(proc.stdout or "")
        if payload is None:
            return None, ("peas-agent-runtime 尚未可 import；略過契約檢查。",), None
        if payload.get("skip"):
            reason = payload.get("reason") or (
                "peas-agent-runtime 尚未可 import；略過契約檢查。"
            )
            return None, (str(reason),), None

        messages = tuple(str(m) for m in payload.get("messages", ()))
        ok = bool(payload.get("ok"))
        hard = tuple(m for m in messages if not m.startswith("提示："))
        if ok:
            return True, messages + (f"契約通過：{factory_ref}",), factory_ref
        return False, hard or messages, factory_ref

    return (
        False,
        (
            "未找到 main_shell:create_agent 或 main:create_agent。"
            "請實作 create_agent(session_path=None, host_context=None)。",
        ),
        None,
    )


def _update_existing(
    target: Path, template_path: Path, packs: Iterable[CoursePack]
) -> None:
    """刪掉保留目錄以外的舊檔，再複製框架與已裝課程包（不多裝）。"""
    models_dir = target / "built-in-data" / "computer-vision" / "models"
    models_backup: Path | None = None
    if models_dir.exists() and any(models_dir.glob("*.pt")):
        models_backup = target.parent / f".{target.name}_models_backup"
        if models_backup.exists():
            shutil.rmtree(models_backup)
        shutil.copytree(models_dir, models_backup)

    for child in target.iterdir():
        if child.name in PRESERVE_ON_UPDATE:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    _copy_shell_files(template_path, target, packs)

    if models_backup is not None and models_backup.exists():
        restored = target / "built-in-data" / "computer-vision" / "models"
        restored.mkdir(parents=True, exist_ok=True)
        for weight_file in models_backup.glob("*.pt"):
            shutil.copy2(weight_file, restored / weight_file.name)
        shutil.rmtree(models_backup)


def _restore_challenge_runtime(backup: Path, target: Path) -> None:
    src_root = backup / "workspace" / "challenge"
    dest_root = target / "workspace" / "challenge"
    if not src_root.is_dir():
        return
    dest_root.mkdir(parents=True, exist_ok=True)
    for src_file in src_root.rglob("*"):
        if not src_file.is_file():
            continue
        if src_file.name not in _CHALLENGE_RUNTIME_FILE_NAMES:
            continue
        dest_file = dest_root / src_file.relative_to(src_root)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)


def _backup_existing(target: Path) -> Path:
    index = 1
    while True:
        backup = target.with_name(f"{target.name}.bak{index}")
        if not backup.exists():
            shutil.move(str(target), str(backup))
            return backup
        index += 1
