# Dataset Streamlit Shell Installer

Install the workshop `dataset_streamlit_shell/` UI into a student agent project.

## Usage

From a workshop project that contains `agent_core.py`:

```powershell
uvx --from . add-dataset-streamlit-shell
```

To update an existing shell while keeping runtime data:

```powershell
uvx --from . add-dataset-streamlit-shell --update
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
- Checks that `agent_core.py` exists by default.
- Refuses to overwrite an existing shell unless `--force` is used.
- Supports `--update` to refresh shell code while preserving runtime data.
- Prints the Streamlit launch command.
