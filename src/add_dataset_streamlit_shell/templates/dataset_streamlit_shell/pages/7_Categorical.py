from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.ui.data_ui import inject_style
from dataset_streamlit_shell.ui.workflow_ui import render_categorical_page

st.set_page_config(page_title="類別欄位整理", page_icon="CA", layout="wide")
inject_style()
render_categorical_page()
