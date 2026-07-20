from __future__ import annotations

from pathlib import Path

import typer

from add_dataset_streamlit_shell.installer import install_shell


app = typer.Typer(
    name="add-dataset-streamlit-shell",
    help="Install the workshop dataset Streamlit shell into the current project.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def install(
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
    """Copy dataset_streamlit_shell/ into a workshop project."""

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
    typer.echo("  uv run streamlit run dataset_streamlit_shell/app.py")


def main() -> None:
    app()
