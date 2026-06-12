from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.cnn_ui import render_cnn_introduction_page
from dataset_streamlit_shell.data_ui import inject_style

st.set_page_config(page_title="卷積神經網路（CNN）", page_icon="CNN", layout="wide")
inject_style()
render_cnn_introduction_page()
