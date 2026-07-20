# Dataset Streamlit Shell Installer

Install the workshop `dataset_streamlit_shell/` UI into a student agent project.

## Usage

From the project where you want to add the shell:

```powershell
uvx --from git+https://github.com/mz038197/dataset-streamlit-shell-installer.git add-dataset-streamlit-shell
```

Local development:

```powershell
uvx --from . add-dataset-streamlit-shell
```

To require `agent_core.py` during installation:

```powershell
uvx --from git+https://github.com/mz038197/dataset-streamlit-shell-installer.git add-dataset-streamlit-shell --require-agent-core
```

To update an existing shell while keeping runtime data:

```powershell
uvx --from git+https://github.com/mz038197/dataset-streamlit-shell-installer.git add-dataset-streamlit-shell --update
```

By default, installation and update also run this in the target project:

```powershell
uv add --upgrade-package openai-tts streamlit pandas matplotlib numpy scikit-learn xgboost "openai-tts @ git+https://github.com/mz038197/openai-tts.git"
```

To copy or update the shell without changing project dependencies:

```powershell
uvx --from git+https://github.com/mz038197/dataset-streamlit-shell-installer.git add-dataset-streamlit-shell --no-install-deps
```

This preserves:

- `dataset_streamlit_shell/workspace/*.csv`
- `dataset_streamlit_shell/workspace/*.jsonl`
- `dataset_streamlit_shell/workspace/user_settings.json`
- `sessions/*.jsonl`（專案根；對話紀錄）
- `dataset_streamlit_shell/scripts/`
- `dataset_streamlit_shell/uploads/`

After installation:

```powershell
uv run streamlit run dataset_streamlit_shell/app.py
```

## TTS settings

The right-side Agent panel reads and writes TTS preferences at:

```text
dataset_streamlit_shell/workspace/user_settings.json
```

The file is created automatically with defaults when the panel opens. It uses these keys:

```json
{
  "tts_enabled": false,
  "tts_voice": "nova",
  "tts_instructions": "用台灣繁體中文說話。",
  "tts_speed": 1.0
}
```

Student-facing TTS settings are loaded in this order:

1. `dataset_streamlit_shell/workspace/user_settings.json`
2. `openai-tts` built-in defaults

The visible UI defaults do not use `.env` values, so students only need to understand
the settings file and the right-side panel. `.env` can still hold API keys or advanced
runtime settings.

Switching Streamlit pages reloads TTS preferences from `user_settings.json`.
After manually editing that file, switch to another page and back, or restart the Streamlit session.
The TTS panel is available even before Agent Core is connected.

## Machine learning pages (shell template)

The installed `dataset_streamlit_shell/` template includes supervised learning pages aligned with the course labs:

- **Linear regression** (built-in restaurant profit and house price CSVs under `built-in-data/regression/`)
- **Logistic regression** — university admission demo from `built-in-data/classification/university_admission.csv` (Coursera ex2data1)
- **Regularized logistic regression** — microchip test demo from `built-in-data/classification/microchip_test.csv` (Coursera ex2data2, degree-6 feature map and λ)
- **Linear SVM** — built-in two-feature binary demo from `built-in-data/classification/svm_blobs_80.csv`. Uses `sklearn.svm.SVC(kernel='linear')`; shows the final decision boundary and support vectors. Requires `scikit-learn`.
- **Decision tree concepts** — built-in 10-row cat toy CSV from `built-in-data/classification/cat_toy_10.csv`. Shows entropy, information gain table, and `DecisionTreeClassifier` with student-chosen Gini/Entropy criterion and `max_depth` 1–2. Builtin data only; no model JSON save.
- **XGBoost** — built-in heart disease CSV from `built-in-data/classification/heart_disease.csv` (Traditional Chinese column names). Auto one-hot encoding, 80/20 train/validation split, hyperparameter sweep curves, and final `XGBClassifier` with early stopping. Requires `xgboost`. Builtin data only; no model JSON save.

Trained classification models are saved as portable JSON under `dataset_streamlit_shell/workspace/models/classification/`. Logistic pages use hand-written gradient descent and logistic Cost J; the linear SVM page uses scikit-learn `SVC`. Classification threshold is adjusted after training on logistic pages and is not stored in the model JSON (regularization λ is stored for the regularized page). Decision tree and XGBoost pages do not save models in this release.

### Promptable segmentation (SAM 3)

The **Promptable Segmentation** page uses SAM 3 (`sam3.pt`, ~3.4 GB). Weights are **not** bundled in git. On first use, open the page and click **Download SAM 3 weights** (course-hosted via Google Drive), or place `sam3.pt` manually at:

```text
dataset_streamlit_shell/built-in-data/computer-vision/models/sam3.pt
```

Requires stable network, ~3.4 GB free disk space, and `gdown` (installed with shell dependencies). Weights are for course use only. Running `add-dataset-streamlit-shell --update` preserves downloaded `*.pt` files under `built-in-data/computer-vision/models/`.

## What It Does

- Copies `dataset_streamlit_shell/` into the current project.
- Installs even before `agent_core.py` is connected; use `--require-agent-core` for strict checking.
- Installs required project dependencies with `uv add streamlit pandas matplotlib numpy scikit-learn xgboost` and `openai-tts` by default.
- Persists TTS preferences to `dataset_streamlit_shell/workspace/user_settings.json` across page changes and browser restarts.
- Refuses to overwrite an existing shell unless `--force` is used.
- Supports `--update` to refresh shell code while preserving runtime data.
- Prints the Streamlit launch command.
