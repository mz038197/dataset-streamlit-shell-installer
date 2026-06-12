from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


SHELL_DIR_NAME = "dataset_streamlit_shell"
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


def install_shell(
    project_root: Path,
    *,
    force: bool = False,
    update: bool = False,
    require_agent_core: bool = False,
    install_dependencies: bool = True,
    dependency_runner: Callable[..., object] = subprocess.run,
) -> InstallResult:
    project_root = project_root.resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    if require_agent_core and not (project_root / "agent_core.py").exists():
        raise FileNotFoundError(
            "agent_core.py was not found. Run this from a WG-22 workshop project root, "
            "or omit --require-agent-core."
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
            installed = _install_dependencies(
                project_root,
                dependency_runner,
            ) if install_dependencies else ()
            return InstallResult(target=target, installed_dependencies=installed)
        if not force:
            raise FileExistsError(
                f"{SHELL_DIR_NAME}/ already exists. Re-run with --force to replace it."
            )
        backup = _backup_existing(target)

    with resources.as_file(template) as template_path:
        shutil.copytree(template_path, target)

    installed = _install_dependencies(project_root, dependency_runner) if install_dependencies else ()
    return InstallResult(target=target, backed_up_to=backup, installed_dependencies=installed)


def _install_dependencies(
    project_root: Path,
    dependency_runner: Callable[..., object],
) -> tuple[str, ...]:
    dependency_runner(
        ["uv", "add", "--upgrade-package", "openai-tts", *PROJECT_DEPENDENCIES],
        cwd=project_root,
        check=True,
    )
    return PROJECT_DEPENDENCIES


def _update_existing(target: Path, template_path: Path) -> None:
    models_dir = target / "built-in-data" / "computer-vision" / "models"
    models_backup: Path | None = None
    if models_dir.exists() and any(models_dir.glob("*.pt")):
        models_backup = target.parent / f".{target.name}_models_backup"
        if models_backup.exists():
            shutil.rmtree(models_backup)
        shutil.copytree(models_dir, models_backup)

    for child in target.iterdir():
        if child.name in {"workspace", "sessions", "scripts", "uploads"}:
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
