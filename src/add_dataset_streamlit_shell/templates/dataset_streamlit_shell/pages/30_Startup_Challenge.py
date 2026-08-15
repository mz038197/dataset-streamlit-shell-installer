from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.ui.data_ui import brand_page_icon, inject_style
from dataset_streamlit_shell.ui.startup_challenge_ui import render_startup_challenge_page

st.set_page_config(page_title="專案展示", page_icon=brand_page_icon(), layout="wide")
inject_style()
render_startup_challenge_page()
