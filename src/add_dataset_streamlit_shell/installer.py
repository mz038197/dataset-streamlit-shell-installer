from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


SHELL_DIR_NAME = "dataset_streamlit_shell"


@dataclass(frozen=True)
class InstallResult:
    target: Path
    backed_up_to: Path | None = None


def install_shell(
    project_root: Path,
    *,
    force: bool = False,
    require_agent_core: bool = True,
) -> InstallResult:
    project_root = project_root.resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    if require_agent_core and not (project_root / "agent_core.py").exists():
        raise FileNotFoundError(
            "agent_core.py was not found. Run this from a WG-22 workshop project root, "
            "or pass --no-agent-core-check."
        )

    target = project_root / SHELL_DIR_NAME
    backup: Path | None = None
    if target.exists():
        if not force:
            raise FileExistsError(
                f"{SHELL_DIR_NAME}/ already exists. Re-run with --force to replace it."
            )
        backup = _backup_existing(target)

    template = resources.files("add_dataset_streamlit_shell").joinpath(
        "templates",
        SHELL_DIR_NAME,
    )
    with resources.as_file(template) as template_path:
        shutil.copytree(template_path, target)

    return InstallResult(target=target, backed_up_to=backup)


def _backup_existing(target: Path) -> Path:
    index = 1
    while True:
        backup = target.with_name(f"{target.name}.bak{index}")
        if not backup.exists():
            shutil.move(str(target), str(backup))
            return backup
        index += 1
