# Dataset Streamlit Shell Installer

Install the workshop `dataset_streamlit_shell/` UI into a student agent project.

## Usage

From a workshop project that contains `agent_core.py`:

```powershell
uvx --from . add-dataset-streamlit-shell
```

After installation:

```powershell
uv run streamlit run dataset_streamlit_shell/app.py
```

## What It Does

- Copies `dataset_streamlit_shell/` into the current project.
- Checks that `agent_core.py` exists by default.
- Refuses to overwrite an existing shell unless `--force` is used.
- Prints the Streamlit launch command.
