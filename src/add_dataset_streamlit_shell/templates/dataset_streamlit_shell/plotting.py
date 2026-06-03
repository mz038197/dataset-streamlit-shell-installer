from __future__ import annotations

import importlib


TRADITIONAL_CHINESE_FONT_CANDIDATES = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "Noto Sans TC",
    "Noto Sans CJK TC",
    "MingLiU",
    "PingFang TC",
    "Heiti TC",
    "SimHei",
    "Arial Unicode MS",
]


def configure_matplotlib_for_traditional_chinese() -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    font_manager = importlib.import_module("matplotlib.font_manager")

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    installed_candidates = [
        font_name
        for font_name in TRADITIONAL_CHINESE_FONT_CANDIDATES
        if font_name in available_fonts
    ]
    if installed_candidates:
        plt.rcParams["font.family"] = installed_candidates[0]
        plt.rcParams["font.sans-serif"] = installed_candidates + ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = TRADITIONAL_CHINESE_FONT_CANDIDATES + [
            "DejaVu Sans"
        ]
    plt.rcParams["axes.unicode_minus"] = False
