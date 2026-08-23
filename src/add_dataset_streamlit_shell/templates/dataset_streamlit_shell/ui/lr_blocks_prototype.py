"""PROTOTYPE opener — throwaway. Opens lr_blocks_prototype.html?variant=A."""

from __future__ import annotations

import webbrowser
from pathlib import Path

html = Path(__file__).with_suffix(".html").resolve()
webbrowser.open(html.as_uri() + "?variant=C")
