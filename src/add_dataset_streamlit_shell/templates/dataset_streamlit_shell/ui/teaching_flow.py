"""教學流程圖 — 純函式（SVG／公式／作動狀態），供線性回歸等頁掛載。"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable

FLOW_INPUT = "input"
FLOW_MODEL = "model"
FLOW_OUTPUT = "output"

FLOW_NODE_LABELS = {
    FLOW_INPUT: "輸入資料",
    FLOW_MODEL: "回歸模型",
    FLOW_OUTPUT: "輸出呈現",
}

FLOW_NODE_ORDER = (FLOW_INPUT, FLOW_MODEL, FLOW_OUTPUT)


@dataclass(frozen=True)
class FlowRenderState:
    hot: str | None
    done: frozenset[str]


def symbolic_prediction_latex(features: list[str]) -> str:
    if len(features) == 1:
        return r"Y = WX + B"
    terms = [rf"W_{i} X_{i}" for i in range(1, len(features) + 1)]
    return "Y = " + " + ".join(terms) + " + B"


def numeric_prediction_latex(
    features: list[str],
    weights: list[float],
    intercept: float,
) -> str:
    if len(features) != len(weights):
        raise ValueError("features and weights length mismatch")
    if len(features) == 1:
        return rf"Y = {weights[0]:g} X + {intercept:g}"
    terms = [rf"{weight:g} X_{i}" for i, weight in enumerate(weights, start=1)]
    return "Y = " + " + ".join(terms) + rf" + {intercept:g}"


def live_fit_caption(
    *,
    iteration: int,
    total_iterations: int,
    weights: Iterable[float],
    intercept: float,
    cost: float,
) -> str:
    weight_list = list(weights)
    if len(weight_list) == 1:
        weight_part = f"W = {weight_list[0]:.4f}"
    else:
        weight_part = "W = [" + ", ".join(f"{w:.3f}" for w in weight_list) + "]"
    return (
        f"Iteration {iteration:,} / {total_iterations:,}，"
        f"{weight_part}，B = {intercept:.4f}，Cost J = {cost:.4f}"
    )


def training_flow_state(*, finished: bool) -> FlowRenderState:
    if finished:
        return FlowRenderState(
            hot=None,
            done=frozenset(FLOW_NODE_ORDER),
        )
    return FlowRenderState(
        hot=FLOW_MODEL,
        done=frozenset({FLOW_INPUT}),
    )


def regression_flow_svg(
    *,
    hot: str | None = None,
    done: Iterable[str] = (),
    live_caption: str = "",
) -> str:
    done_set = set(done)
    # Layout: three boxes left → right with cubic edges (waku-style, small graph).
    width, height = 720, 110
    box_w, box_h = 168, 52
    gap = 74
    pad_x, pad_y = 20, 24
    positions = {
        FLOW_INPUT: (pad_x, pad_y),
        FLOW_MODEL: (pad_x + box_w + gap, pad_y),
        FLOW_OUTPUT: (pad_x + 2 * (box_w + gap), pad_y),
    }

    def node_class(name: str) -> str:
        classes = ["node"]
        if name == hot:
            classes.append("hot")
        elif name in done_set:
            classes.append("done")
        return " ".join(classes)

    edges = []
    for src, dst in ((FLOW_INPUT, FLOW_MODEL), (FLOW_MODEL, FLOW_OUTPUT)):
        x1 = positions[src][0] + box_w
        y1 = positions[src][1] + box_h / 2
        x2 = positions[dst][0]
        y2 = positions[dst][1] + box_h / 2
        mx = (x1 + x2) / 2
        edge_live = " live" if hot == dst or (hot is None and dst in done_set and src in done_set) else ""
        if hot == FLOW_MODEL and src == FLOW_INPUT:
            edge_live = " live"
        edges.append(
            f'<path class="flow{edge_live}" data-edge="{src}-{dst}" '
            f'd="M{x1} {y1} C{mx} {y1} {mx} {y2} {x2} {y2}" marker-end="url(#tf-arr)"/>'
        )

    nodes = []
    for name in FLOW_NODE_ORDER:
        x, y = positions[name]
        label = escape(FLOW_NODE_LABELS[name])
        nodes.append(
            f'<g class="{node_class(name)}" data-node="{name}">'
            f'<rect class="bx" x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="9"/>'
            f'<text class="nt" x="{x + box_w / 2}" y="{y + 32}" text-anchor="middle">{label}</text>'
            f"</g>"
        )

    caption = ""
    if live_caption:
        caption = (
            f'<text class="cap" x="{width / 2}" y="{height - 8}" '
            f'text-anchor="middle">{escape(live_caption)}</text>'
        )

    return (
        f'<div class="teaching-flow-wrap">'
        f'<svg viewBox="0 0 {width} {height}" class="teaching-flow" role="img" '
        f'aria-label="教學流程圖：輸入資料、回歸模型、輸出呈現">'
        f"<defs><marker id=\"tf-arr\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" "
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" class="head"/></marker></defs>'
        f"{''.join(edges)}{''.join(nodes)}{caption}"
        f"</svg></div>"
    )


TEACHING_FLOW_CSS = """
<style>
.teaching-flow-wrap { overflow-x: auto; margin: 0.4rem 0 0.8rem; }
.teaching-flow {
    width: 100%;
    max-width: 720px;
    height: auto;
    display: block;
    margin: 0 auto;
    font-family: "Segoe UI", system-ui, sans-serif;
}
.teaching-flow .bx {
    fill: rgba(255, 255, 255, 0.04);
    stroke: rgba(250, 250, 250, 0.28);
    stroke-width: 1.2;
    transition: stroke 0.15s ease, fill 0.15s ease, stroke-width 0.15s ease;
}
.teaching-flow .nt {
    fill: rgba(250, 250, 250, 0.92);
    font-size: 14px;
    font-weight: 600;
}
.teaching-flow .cap {
    fill: rgba(140, 180, 255, 0.95);
    font-size: 11px;
}
.teaching-flow .flow {
    fill: none;
    stroke: rgba(250, 250, 250, 0.35);
    stroke-width: 1.4;
}
.teaching-flow .flow.live {
    stroke: rgba(90, 160, 255, 0.95);
    stroke-width: 2.4;
    stroke-dasharray: 6 5;
}
.teaching-flow .head { fill: rgba(250, 250, 250, 0.45); }
.teaching-flow .node.hot .bx {
    stroke: rgba(90, 160, 255, 0.95) !important;
    stroke-width: 2.6;
    fill: rgba(90, 160, 255, 0.16) !important;
}
.teaching-flow .node.done .bx {
    stroke: rgba(80, 200, 140, 0.9) !important;
    stroke-width: 2.2;
    fill: rgba(80, 200, 140, 0.12) !important;
}
@media (prefers-reduced-motion: reduce) {
    .teaching-flow .bx { transition: none; }
}
</style>
"""
