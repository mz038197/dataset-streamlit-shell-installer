from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.data_ui import inject_style
from dataset_streamlit_shell.workflow_ui import render_simple_linear_regression_page

st.set_page_config(page_title="單變量線性回歸", page_icon="LR", layout="wide")
inject_style()
render_simple_linear_regression_page()
