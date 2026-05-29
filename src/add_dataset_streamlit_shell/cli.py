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
        help="Update shell code in place while preserving data, sessions, and uploads.",
    ),
    require_agent_core: bool = typer.Option(
        False,
        "--require-agent-core",
        help="Require agent_core.py to exist before installing.",
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
        )
    except (FileExistsError, FileNotFoundError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    action = "updated" if update else "installed"
    typer.secho(f"Dataset Streamlit shell {action}.", fg=typer.colors.GREEN)
    typer.echo(f"Target: {result.target}")
    if result.backed_up_to is not None:
        typer.echo(f"Previous shell backed up to: {result.backed_up_to}")

    typer.echo("")
    typer.echo("Next:")
    typer.echo("  uv add streamlit pandas")
    typer.echo("  uv run streamlit run dataset_streamlit_shell/app.py")


def main() -> None:
    app()
