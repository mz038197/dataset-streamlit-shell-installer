"""教學流程圖 — 純函式（SVG／公式／作動狀態），供線性回歸、邏輯迴歸等頁掛載。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
    from dataset_streamlit_shell.ml.regression import GradientDescentStep

FLOW_INPUT = "input"
FLOW_MODEL = "model"
FLOW_OUTPUT = "output"

FLOW_NODE_LABELS = {
    FLOW_INPUT: "輸入資料",
    FLOW_MODEL: "回歸模型",
    FLOW_OUTPUT: "輸出呈現",
}

FLOW_NODE_ORDER = (FLOW_INPUT, FLOW_MODEL, FLOW_OUTPUT)

FLOW_VIEW_INPUT = FLOW_NODE_LABELS[FLOW_INPUT]
FLOW_VIEW_MODEL = FLOW_NODE_LABELS[FLOW_MODEL]
FLOW_VIEW_OUTPUT = FLOW_NODE_LABELS[FLOW_OUTPUT]
FLOW_VIEW_LABELS = (FLOW_VIEW_INPUT, FLOW_VIEW_MODEL, FLOW_VIEW_OUTPUT)

CLASSIFICATION_FLOW_NODE_LABELS = {
    FLOW_INPUT: "輸入資料",
    FLOW_MODEL: "分類模型",
    FLOW_OUTPUT: "輸出呈現",
}
CLASSIFICATION_FLOW_VIEW_INPUT = CLASSIFICATION_FLOW_NODE_LABELS[FLOW_INPUT]
CLASSIFICATION_FLOW_VIEW_MODEL = CLASSIFICATION_FLOW_NODE_LABELS[FLOW_MODEL]
CLASSIFICATION_FLOW_VIEW_OUTPUT = CLASSIFICATION_FLOW_NODE_LABELS[FLOW_OUTPUT]
CLASSIFICATION_FLOW_VIEW_LABELS = (
    CLASSIFICATION_FLOW_VIEW_INPUT,
    CLASSIFICATION_FLOW_VIEW_MODEL,
    CLASSIFICATION_FLOW_VIEW_OUTPUT,
)

SAMPLE_OPS_POLY_NOTE = "ŷ = σ(w·φ(x)+b)；表上不展開 27 維 φ。"

MICRO_PREDICT = "predict"
MICRO_COST = "cost"
MICRO_GRAD = "grad"
MICRO_UPDATE = "update"

MICRO_STEP_LABELS = {
    MICRO_PREDICT: "預測 ŷ",
    MICRO_COST: "算 Cost J",
    MICRO_GRAD: "算梯度",
    MICRO_UPDATE: "更新參數",
}

MICRO_STEP_ORDER = (MICRO_PREDICT, MICRO_COST, MICRO_GRAD, MICRO_UPDATE)

SAMPLE_OPS_HEAD = 5
SAMPLE_OPS_SCALE_NOTE = "表上 x 為 Z-score 後、與當前 w 相乘的同一組輸入。"


@dataclass(frozen=True)
class TrainingMicroFrame:
    iteration: int
    total_iterations: int
    micro_step: str
    learning_rate: float
    feature_names: list[str]
    weights_before: list[float]
    intercept_before: float
    weights_after: list[float]
    intercept_after: float
    cost_before: float
    cost_after: float
    dj_dw: list[float]
    dj_db: float
    delta_w: list[float]
    delta_b: float

    @property
    def chart_weights(self) -> list[float]:
        return self.weights_after if self.micro_step == MICRO_UPDATE else self.weights_before

    @property
    def chart_intercept(self) -> float:
        return self.intercept_after if self.micro_step == MICRO_UPDATE else self.intercept_before

    @property
    def chart_cost(self) -> float:
        return self.cost_after if self.micro_step == MICRO_UPDATE else self.cost_before


@dataclass(frozen=True)
class FlowRenderState:
    hot: str | None
    done: frozenset[str]


def symbolic_logistic_z_latex(*, mapped: bool = False) -> str:
    if mapped:
        return r"z = w\cdot\phi(x)+b"
    return r"z = w\cdot x+b"


def symbolic_logistic_yhat_latex() -> str:
    return r"\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}}"


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


def build_training_micro_frames(
    steps: Iterable[GradientDescentStep],
    *,
    learning_rate: float,
    feature_names: list[str] | None = None,
) -> list[TrainingMicroFrame]:
    step_list = list(steps)
    if not step_list:
        return []
    total = step_list[-1].iteration
    frames: list[TrainingMicroFrame] = []
    for step in step_list:
        if (
            step.iteration < 1
            or step.prev_weights is None
            or step.prev_intercept is None
            or step.prev_cost is None
            or step.dj_dw is None
            or step.dj_db is None
            or step.delta_w is None
            or step.delta_b is None
        ):
            continue
        names = feature_names
        if names is None:
            names = [f"x{i}" for i in range(1, len(step.weights) + 1)]
            if len(names) == 1:
                names = ["x"]
        for micro in MICRO_STEP_ORDER:
            frames.append(
                TrainingMicroFrame(
                    iteration=step.iteration,
                    total_iterations=total,
                    micro_step=micro,
                    learning_rate=float(learning_rate),
                    feature_names=list(names),
                    weights_before=list(step.prev_weights),
                    intercept_before=float(step.prev_intercept),
                    weights_after=list(step.weights),
                    intercept_after=float(step.intercept),
                    cost_before=float(step.prev_cost),
                    cost_after=float(step.cost),
                    dj_dw=list(step.dj_dw),
                    dj_db=float(step.dj_db),
                    delta_w=list(step.delta_w),
                    delta_b=float(step.delta_b),
                )
            )
    return frames


def micro_stepper_html(*, hot: str | None) -> str:
    cells = []
    for index, name in enumerate(MICRO_STEP_ORDER):
        classes = ["micro-step"]
        if name == hot:
            classes.append("hot")
        label = escape(MICRO_STEP_LABELS[name])
        cells.append(
            f'<div class="{" ".join(classes)}" data-micro="{name}">'
            f'<span class="micro-idx">{index + 1}</span>'
            f'<span class="micro-label">{label}</span></div>'
        )
        if index < len(MICRO_STEP_ORDER) - 1:
            cells.append('<div class="micro-arrow" aria-hidden="true">→</div>')
    return (
        '<div class="training-micro-wrap" role="list" aria-label="訓練微步驟">'
        f'{"".join(cells)}</div>'
    )


def simple_gradient_board_lines(frame: TrainingMicroFrame) -> list[str]:
    w0 = frame.weights_before[0]
    lines = [
        f"目前步驟：{MICRO_STEP_LABELS[frame.micro_step]}",
        f"Iteration {frame.iteration:,} / {frame.total_iterations:,}",
        f"α = {frame.learning_rate:g}",
        f"w = {w0:.6g}",
        f"b = {frame.intercept_before:.6g}",
    ]
    if frame.micro_step == MICRO_PREDICT:
        lines.append("用當前 w、b 計算預測 ŷ = w·x + b")
        return lines
    if frame.micro_step in {MICRO_COST, MICRO_GRAD, MICRO_UPDATE}:
        lines.append(f"Cost J = {frame.cost_before:.6g}")
    if frame.micro_step in {MICRO_GRAD, MICRO_UPDATE}:
        lines.append(f"∂J/∂w = {frame.dj_dw[0]:.6g}")
        lines.append(f"∂J/∂b = {frame.dj_db:.6g}")
    if frame.micro_step == MICRO_UPDATE:
        lines.append(f"Δw = −α·∂J/∂w = {frame.delta_w[0]:.6g}")
        lines.append(f"Δb = −α·∂J/∂b = {frame.delta_b:.6g}")
        lines.append(f"w' = w + Δw = {frame.weights_after[0]:.6g}")
        lines.append(f"b' = b + Δb = {frame.intercept_after:.6g}")
        lines.append(f"更新後 Cost J = {frame.cost_after:.6g}")
    return lines


def gradient_board_rows(frame: TrainingMicroFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    show_grad = frame.micro_step in {MICRO_GRAD, MICRO_UPDATE}
    show_update = frame.micro_step == MICRO_UPDATE
    for name, w, dj, dw, w_after in zip(
        frame.feature_names,
        frame.weights_before,
        frame.dj_dw,
        frame.delta_w,
        frame.weights_after,
        strict=True,
    ):
        rows.append(
            {
                "參數": name,
                "當前": f"{w:.6g}",
                "梯度": f"{dj:.6g}" if show_grad else "—",
                "Δ": f"{dw:.6g}" if show_update else "—",
                "更新後": f"{w_after:.6g}" if show_update else "—",
            }
        )
    rows.append(
        {
            "參數": "b（截距）",
            "當前": f"{frame.intercept_before:.6g}",
            "梯度": f"{frame.dj_db:.6g}" if show_grad else "—",
            "Δ": f"{frame.delta_b:.6g}" if show_update else "—",
            "更新後": f"{frame.intercept_after:.6g}" if show_update else "—",
        }
    )
    return rows


def sample_ops_table_visible(micro_step: str) -> bool:
    return micro_step in {MICRO_PREDICT, MICRO_COST}


def sample_ops_x_labels(feature_names: list[str]) -> list[str]:
    if len(feature_names) == 1:
        return ["x（Z-score）"]
    return [f"{name}（Z-score）" for name in feature_names]


def sample_ops_cost_caption(frame: TrainingMicroFrame) -> str | None:
    if frame.micro_step != MICRO_COST:
        return None
    return f"Cost J（整批）= {frame.cost_before:.6g}"


def sample_ops_table_rows(
    frame: TrainingMicroFrame,
    *,
    scaled_x_rows: list[list[float]],
    y_rows: list[float],
) -> list[dict[str, str]] | None:
    if not sample_ops_table_visible(frame.micro_step):
        return None
    if len(scaled_x_rows) != len(y_rows):
        raise ValueError("scaled_x_rows and y_rows length mismatch")
    x_labels = sample_ops_x_labels(frame.feature_names)
    n_features = len(frame.feature_names)
    show_error = frame.micro_step == MICRO_COST
    rows: list[dict[str, str]] = []
    for x_row, y in zip(scaled_x_rows, y_rows, strict=True):
        if len(x_row) != n_features:
            raise ValueError("scaled_x_rows feature width mismatch")
        y_hat = sum(
            float(w) * float(x) for w, x in zip(frame.weights_before, x_row, strict=True)
        ) + float(frame.intercept_before)
        row: dict[str, str] = {
            label: f"{float(x):.6g}" for label, x in zip(x_labels, x_row, strict=True)
        }
        row["ŷ"] = f"{y_hat:.6g}"
        row["y"] = f"{float(y):.6g}"
        if show_error:
            err = y_hat - float(y)
            row["error"] = f"{err:.6g}"
            row["error²"] = f"{(err * err):.6g}"
        rows.append(row)
    return rows


def _sigmoid_scalar(z: float) -> float:
    clipped = min(max(float(z), -500.0), 500.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _logistic_ell(y_hat: float, y: float) -> float:
    probability = min(max(float(y_hat), 1e-15), 1.0 - 1e-15)
    return float(-(y * math.log(probability) + (1.0 - y) * math.log(1.0 - probability)))


def logistic_sample_ops_table_rows(
    frame: TrainingMicroFrame,
    *,
    model_x_rows: list[list[float]],
    y_rows: list[float],
    show_x: bool = True,
) -> list[dict[str, str]] | None:
    if not sample_ops_table_visible(frame.micro_step):
        return None
    if len(model_x_rows) != len(y_rows):
        raise ValueError("model_x_rows and y_rows length mismatch")
    n_features = len(frame.feature_names)
    show_ell = frame.micro_step == MICRO_COST
    x_labels = sample_ops_x_labels(frame.feature_names)
    rows: list[dict[str, str]] = []
    for x_row, y in zip(model_x_rows, y_rows, strict=True):
        if len(x_row) != n_features:
            raise ValueError("model_x_rows feature width mismatch")
        z = sum(
            float(w) * float(x) for w, x in zip(frame.weights_before, x_row, strict=True)
        ) + float(frame.intercept_before)
        y_hat = _sigmoid_scalar(z)
        row: dict[str, str] = {}
        if show_x:
            row.update(
                {label: f"{float(x):.6g}" for label, x in zip(x_labels, x_row, strict=True)}
            )
        row["ŷ"] = f"{y_hat:.6g}"
        row["y"] = f"{float(y):.6g}"
        if show_ell:
            row["ℓ"] = f"{_logistic_ell(y_hat, float(y)):.6g}"
        rows.append(row)
    return rows


def regularized_compact_board_lines(
    frame: TrainingMicroFrame,
    *,
    lambda_: float,
) -> list[str]:
    w_sq = sum(float(weight) * float(weight) for weight in frame.weights_before)
    w_sq_after = sum(float(weight) * float(weight) for weight in frame.weights_after)
    lines = [
        f"目前步驟：{MICRO_STEP_LABELS[frame.micro_step]}",
        f"Iteration {frame.iteration:,} / {frame.total_iterations:,}",
        f"α = {frame.learning_rate:g}",
        f"λ = {lambda_:g}",
        f"‖w‖² = {w_sq:.6g}",
        f"b = {frame.intercept_before:.6g}",
    ]
    if frame.micro_step == MICRO_PREDICT:
        lines.append("ŷ = σ(w·φ(x)+b)")
        return lines
    if frame.micro_step in {MICRO_COST, MICRO_GRAD, MICRO_UPDATE}:
        lines.append(f"Cost J = {frame.cost_before:.6g}")
    if frame.micro_step in {MICRO_GRAD, MICRO_UPDATE}:
        lines.append("∂J/∂w = (1/m)Xᵀ(ŷ−y) + (λ/m)w")
        lines.append(f"∂J/∂b = {frame.dj_db:.6g}（不加 λ）")
    if frame.micro_step == MICRO_UPDATE:
        lines.append(f"Δb = −α·∂J/∂b = {frame.delta_b:.6g}")
        lines.append(f"b' = b + Δb = {frame.intercept_after:.6g}")
        lines.append(f"更新後 ‖w‖² = {w_sq_after:.6g}")
        lines.append(f"更新後 Cost J = {frame.cost_after:.6g}")
    return lines


def classification_flow_svg(
    *,
    hot: str | None = None,
    done: Iterable[str] = (),
    live_caption: str = "",
) -> str:
    return regression_flow_svg(
        hot=hot,
        done=done,
        live_caption=live_caption,
        node_labels=CLASSIFICATION_FLOW_NODE_LABELS,
    )


def regression_flow_svg(
    *,
    hot: str | None = None,
    done: Iterable[str] = (),
    live_caption: str = "",
    node_labels: Mapping[str, str] | None = None,
) -> str:
    labels = FLOW_NODE_LABELS if node_labels is None else node_labels
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
        label = escape(labels[name])
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
        f'aria-label="教學流程圖：{"、".join(labels[name] for name in FLOW_NODE_ORDER)}">'
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
.training-micro-wrap {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    margin: 0.2rem 0 0.9rem;
    max-width: 720px;
    margin-left: auto;
    margin-right: auto;
}
.training-micro-wrap .micro-step {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    border: 1px solid rgba(250, 250, 250, 0.28);
    background: rgba(255, 255, 255, 0.04);
    font-size: 0.85rem;
}
.training-micro-wrap .micro-step.hot {
    border-color: rgba(90, 160, 255, 0.95);
    background: rgba(90, 160, 255, 0.16);
    font-weight: 600;
}
.training-micro-wrap .micro-idx {
    opacity: 0.7;
    font-variant-numeric: tabular-nums;
}
.training-micro-wrap .micro-arrow {
    opacity: 0.45;
}
</style>
"""
