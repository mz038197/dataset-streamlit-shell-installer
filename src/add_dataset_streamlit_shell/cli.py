from __future__ import annotations

from pathlib import Path

import typer

from add_dataset_streamlit_shell.course_packs import UnknownCoursePackError
from add_dataset_streamlit_shell.installer import (
    add_packs,
    course_pack_status,
    install_shell,
)


app = typer.Typer(
    name="add-dataset-streamlit-shell",
    help=(
        "Install the AI Lab framework into the current project; "
        "use `add <pack...>` to add course packs."
    ),
    add_completion=False,
)

_COMMAND = "add-dataset-streamlit-shell"
_RUN_HINT = "  uv run streamlit run dataset_streamlit_shell/app.py"


def _project_root_option() -> Path:
    return typer.Option(
        Path.cwd(),
        "--project-root",
        "-C",
        help="Workshop project root to install into.",
    )


@app.callback(invoke_without_command=True)
def install(
    ctx: typer.Context,
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        "-C",
        help="Workshop project root to install into.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Replace an existing dataset_streamlit_shell/ after backing it up.",
    ),
    update: bool = typer.Option(
        False,
        "--update",
        "-u",
        help=(
            "Update shell code in place while preserving workspace, "
            "project-root sessions/, uploads, memory."
        ),
    ),
    require_agent_core: bool = typer.Option(
        False,
        "--require-agent-core",
        help="(舊旗標) 要求 agent_core.py；Dataset Shell 已改 create_agent，請改用 --require-agent-contract。",
    ),
    require_agent_contract: bool = typer.Option(
        False,
        "--require-agent-contract",
        help="安裝後必須通過 create_agent 契約檢查，否則非零結束。",
    ),
    install_dependencies: bool = typer.Option(
        True,
        "--install-deps/--no-install-deps",
        help="Install Streamlit shell dependencies into the target project with uv add.",
    ),
) -> None:
    """Install the AI Lab framework (dataset_streamlit_shell/) into a workshop project."""

    if ctx.invoked_subcommand is not None:
        return
    if force and update:
        typer.secho("Error: --force and --update cannot be used together.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        result = install_shell(
            project_root,
            force=force,
            update=update,
            require_agent_core=require_agent_core,
            require_agent_contract=require_agent_contract,
            install_dependencies=install_dependencies,
        )
    except (FileExistsError, FileNotFoundError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    action = "updated" if update else "installed"
    typer.secho(f"Dataset Streamlit shell {action}.", fg=typer.colors.GREEN)
    typer.echo(f"Target: {result.target}")
    typer.echo(
        "Course packs: " + (", ".join(result.course_packs) or "(none, AI Lab framework only)")
    )
    if result.backed_up_to is not None:
        typer.echo(f"Previous shell backed up to: {result.backed_up_to}")
    if result.installed_dependencies:
        typer.echo("Installed dependencies: " + ", ".join(result.installed_dependencies))

    if result.contract_ok is True:
        typer.secho(
            f"Agent contract OK ({result.factory_ref}).",
            fg=typer.colors.GREEN,
        )
    elif result.contract_ok is False:
        typer.secho("Agent contract: not connected yet.", fg=typer.colors.YELLOW)
        for msg in result.contract_messages:
            typer.echo(f"  - {msg}")
    elif result.contract_messages:
        for msg in result.contract_messages:
            typer.echo(f"Contract note: {msg}")

    typer.echo("")
    typer.echo("Next:")
    if not result.course_packs:
        typer.echo(f"  {_COMMAND} add data-science")
    typer.echo(_RUN_HINT)


@app.command("add")
def add(
    names: list[str] = typer.Argument(
        ...,
        help="Course pack names to add (see `list`), or `all`.",
    ),
    project_root: Path = _project_root_option(),
    install_dependencies: bool = typer.Option(
        True,
        "--install-deps/--no-install-deps",
        help="Install the course packs' dependencies with uv add.",
    ),
) -> None:
    """Add course packs (installs the AI Lab framework first if needed)."""

    try:
        result = add_packs(
            project_root,
            names,
            install_dependencies=install_dependencies,
        )
    except (UnknownCoursePackError, FileNotFoundError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if result.framework_installed:
        typer.secho("AI Lab framework installed.", fg=typer.colors.GREEN)
    typer.echo(f"Target: {result.target}")
    for name in result.added:
        typer.secho(f"Added: {name}", fg=typer.colors.GREEN)
    for name in result.already_installed:
        typer.echo(f"{name}: 已安裝")
    if result.installed_dependencies:
        typer.echo("Installed dependencies: " + ", ".join(result.installed_dependencies))
    typer.echo("")
    typer.echo("Next: refresh the Streamlit page, or start it with:")
    typer.echo(_RUN_HINT)


@app.command("list")
def list_packs(project_root: Path = _project_root_option()) -> None:
    """List course packs, their sidebar sections and whether they are installed."""

    for pack, installed in course_pack_status(project_root):
        state = "已安裝" if installed else "未安裝"
        typer.echo(f"{pack.name:<18} {pack.sidebar_section:<16} {state}")


def main() -> None:
    app()
