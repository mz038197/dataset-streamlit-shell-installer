"""測試環境：強制非互動後端，避免本機／CI 的 tk 缺檔拖垮 matplotlib。"""

from __future__ import annotations

import os

# 必須在任何 pyplot import 之前；setdefault 允許外部覆寫。
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except ImportError:
    pass
