from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


SHELL_DIR_NAME = "dataset_streamlit_shell"
PRESERVE_ON_UPDATE = frozenset(
    {"workspace", "sessions", "scripts", "uploads", "memory"}
)
PROJECT_DEPENDENCIES = (
    "streamlit",
    "pandas",
    "matplotlib",
    "numpy",
    "scikit-learn",
    "xgboost",
    "tensorflow-cpu",
    "pillow",
    "opencv-python-headless",
    "ultralytics>=8.3.237",
    "clip @ git+https://github.com/ultralytics/CLIP.git",
    "torchvision",
    "gdown",
    "openai-tts @ git+https://github.com/mz038197/openai-tts.git",
)
DependencyRunner = Callable[[list[str]], None]


@dataclass(frozen=True)
class InstallResult:
    target: Path
    backed_up_to: Path | None = None
    installed_dependencies: tuple[str, ...] = ()
    contract_ok: bool | None = None
    contract_messages: tuple[str, ...] = ()
    factory_ref: str | None = None


def _resolve_runtime_path(project_root: Path) -> Path | None:
    sibling = (project_root.parent / "peas-agent-runtime").resolve()
    if sibling.is_dir() and (sibling / "pyproject.toml").exists():
        return sibling
    return None


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
    project_root = project_root.resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    if require_agent_core and not (project_root / "agent_core.py").exists():
        raise FileNotFoundError(
            "agent_core.py was not found. Dataset Shell 已改為 create_agent；"
            "請改用 --require-agent-contract，或省略此旗標。"
        )

    target = project_root / SHELL_DIR_NAME
    backup: Path | None = None
    template = resources.files("add_dataset_streamlit_shell").joinpath(
        "templates",
        SHELL_DIR_NAME,
    )
    if target.exists():
        if update:
            with resources.as_file(template) as template_path:
                _update_existing(target, template_path)
            installed = (
                _install_dependencies(project_root, dependency_runner)
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
                installed_dependencies=installed,
                contract_ok=contract_ok,
                contract_messages=contract_messages,
                factory_ref=factory_ref,
            )
        if not force:
            raise FileExistsError(
                f"{SHELL_DIR_NAME}/ already exists. Re-run with --force to replace it."
            )
        backup = _backup_existing(target)

    with resources.as_file(template) as template_path:
        shutil.copytree(template_path, target)

    installed = (
        _install_dependencies(project_root, dependency_runner)
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
    )


def _install_dependencies(
    project_root: Path,
    dependency_runner: Callable[..., object],
) -> tuple[str, ...]:
    dependency_runner(
        ["uv", "add", "--upgrade-package", "openai-tts", *PROJECT_DEPENDENCIES],
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
            ["uv", "add", "peas-agent-runtime"],
            cwd=project_root,
            check=True,
        )
        runtime_label = "peas-agent-runtime"
    return (*PROJECT_DEPENDENCIES, runtime_label)


def _check_contract(
    project_root: Path,
) -> tuple[bool | None, tuple[str, ...], str | None]:
    """盡力檢查 create_agent；runtime 未安裝時回傳 None（不阻一般安裝）。"""
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    try:
        from peas_agent_runtime import check_agent_factory
    except ImportError:
        return None, ("peas-agent-runtime 尚未可 import；略過契約檢查。",), None

    for factory_ref in ("main_shell:create_agent", "main:create_agent"):
        module_name = factory_ref.split(":", 1)[0]
        if not (project_root / f"{module_name}.py").exists():
            continue
        result = check_agent_factory(factory_ref, project_root=project_root)
        hard = tuple(m for m in result.messages if not m.startswith("提示："))
        if result.ok:
            return True, result.messages + (f"契約通過：{factory_ref}",), factory_ref
        return False, hard or result.messages, factory_ref

    return (
        False,
        (
            "未找到 main_shell:create_agent 或 main:create_agent。"
            "請實作 create_agent(session_path=None, host_context=None)。",
        ),
        None,
    )


def _update_existing(target: Path, template_path: Path) -> None:
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
    shutil.copytree(template_path, target, dirs_exist_ok=True)

    if models_backup is not None and models_backup.exists():
        restored = target / "built-in-data" / "computer-vision" / "models"
        restored.mkdir(parents=True, exist_ok=True)
        for weight_file in models_backup.glob("*.pt"):
            shutil.copy2(weight_file, restored / weight_file.name)
        shutil.rmtree(models_backup)


def _backup_existing(target: Path) -> Path:
    index = 1
    while True:
        backup = target.with_name(f"{target.name}.bak{index}")
        if not backup.exists():
            shutil.move(str(target), str(backup))
            return backup
        index += 1
