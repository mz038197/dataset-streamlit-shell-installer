# Dataset Streamlit Shell Installer

Install the workshop `dataset_streamlit_shell/` UI into a student agent project.

## Usage

From the project where you want to add the shell:

```powershell
uvx --from . add-dataset-streamlit-shell
```

To require `agent_core.py` during installation:

```powershell
uvx --from . add-dataset-streamlit-shell --require-agent-core
```

To update an existing shell while keeping runtime data:

```powershell
uvx --from . add-dataset-streamlit-shell --update
```

By default, installation and update also run this in the target project:

```powershell
uv add streamlit pandas matplotlib numpy
```

To copy or update the shell without changing project dependencies:

```powershell
uvx --from . add-dataset-streamlit-shell --no-install-deps
```

This preserves:

- `dataset_streamlit_shell/data/*.csv`
- `dataset_streamlit_shell/data/*.jsonl`
- `dataset_streamlit_shell/sessions/*.jsonl`
- `dataset_streamlit_shell/scripts/`
- `dataset_streamlit_shell/uploads/`

After installation:

```powershell
uv run streamlit run dataset_streamlit_shell/app.py
```

## What It Does

- Copies `dataset_streamlit_shell/` into the current project.
- Installs even before `agent_core.py` is connected; use `--require-agent-core` for strict checking.
- Installs required project dependencies with `uv add streamlit pandas matplotlib numpy` by default.
- Refuses to overwrite an existing shell unless `--force` is used.
- Supports `--update` to refresh shell code while preserving runtime data.
- Prints the Streamlit launch command.
