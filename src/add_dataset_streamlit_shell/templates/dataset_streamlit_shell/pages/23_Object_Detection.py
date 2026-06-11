from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.data_ui import inject_style
from dataset_streamlit_shell.detection_ui import render_object_detection_page

st.set_page_config(page_title="物件偵測（Object Detection）", page_icon="OD", layout="wide")
inject_style()
render_object_detection_page()
