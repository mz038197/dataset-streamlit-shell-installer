"""單變量線性回歸：訓練前預測關卡（pure helpers）。"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

SLOPE_POSITIVE = "正的"
SLOPE_NEGATIVE = "負的"
SLOPE_NEAR_ZERO = "接近 0"
SLOPE_OPTIONS = (SLOPE_POSITIVE, SLOPE_NEGATIVE, SLOPE_NEAR_ZERO)

ALPHA_CONVERGE = "大致往下收斂"
ALPHA_EXPLODE = "劇烈震盪或往上爆"
ALPHA_FLAT = "幾乎完全不動"
ALPHA_OPTIONS = (ALPHA_CONVERGE, ALPHA_EXPLODE, ALPHA_FLAT)
ALPHA_CORRECT = ALPHA_EXPLODE

PLEASE_SELECT = "請選擇"
CORR_EPS = 0.05
HINT_COOLDOWN_SEC = 2.5

QID_SLOPE = "slope"
QID_ALPHA = "alpha"

SESSION_SLOPE = "simple_reg_quiz_slope"
SESSION_ALPHA = "simple_reg_quiz_alpha"
SESSION_PAIR = "simple_reg_quiz_pair"
SESSION_FOCUS = "simple_reg_quiz_focus"


def quiz_identity(
    feature: str,
    target: str,
    *,
    source_label: str,
    frame: pd.DataFrame,
) -> tuple[Any, ...]:
    """Identity for quiz reset: column pair + source + data fingerprint."""
    x = frame[feature].astype(float).to_numpy()
    y = frame[target].astype(float).to_numpy()
    digest = hashlib.sha256(x.tobytes() + b"|" + y.tobytes()).hexdigest()[:16]
    return (source_label, feature, target, int(len(frame)), digest)


def needs_quiz_reset(stored_pair: Any, expected: tuple[Any, ...]) -> bool:
    if stored_pair is None:
        return False
    if not isinstance(stored_pair, (tuple, list)):
        return True
    return tuple(stored_pair) != expected


def expected_slope_direction(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    eps: float = CORR_EPS,
) -> str:
    x = frame[feature].astype(float)
    y = frame[target].astype(float)
    if float(x.std(ddof=0)) < 1e-12 or float(y.std(ddof=0)) < 1e-12:
        return SLOPE_NEAR_ZERO
    corr = float(x.corr(y))
    if corr != corr:  # NaN
        return SLOPE_NEAR_ZERO
    if corr > eps:
        return SLOPE_POSITIVE
    if corr < -eps:
        return SLOPE_NEGATIVE
    return SLOPE_NEAR_ZERO


def slope_label_from_weight(
    weight: float,
    *,
    x_std: float,
    y_std: float,
    eps: float = CORR_EPS,
) -> str:
    """Classify trained w with the same scale-independent cutoff as the quiz.

    For simple OLS, ``w * sx / sy`` equals Pearson r, so weak slopes stay
    「接近 0」 instead of flipping on raw weight units.
    """
    sx = float(x_std)
    sy = float(y_std)
    if sx < 1e-12 or sy < 1e-12:
        return SLOPE_NEAR_ZERO
    corr_proxy = float(weight) * sx / sy
    if corr_proxy != corr_proxy:  # NaN
        return SLOPE_NEAR_ZERO
    if corr_proxy > eps:
        return SLOPE_POSITIVE
    if corr_proxy < -eps:
        return SLOPE_NEGATIVE
    return SLOPE_NEAR_ZERO


def build_regression_frame_hint_summary(
    frame: pd.DataFrame,
    feature: str,
    target: str,
) -> str:
    """Non-answer-revealing summary of the displayed regression frame for Agent hints."""
    x = frame[feature].astype(float)
    y = frame[target].astype(float)
    ordered = (
        pd.DataFrame({feature: x, target: y})
        .sort_values(feature, kind="mergesort")
        .reset_index(drop=True)
    )
    n = len(ordered)
    sample_idx = sorted({0, n // 2, n - 1})
    sample_bits = [
        f"({ordered.at[i, feature]:g}, {ordered.at[i, target]:g})" for i in sample_idx
    ]
    third = max(n // 3, 1)
    low_y = float(ordered.iloc[:third][target].mean())
    high_y = float(ordered.iloc[-third:][target].mean())
    lines = [
        "【本頁回歸訓練資料摘要｜供觀察散點，勿直接講正解選項】",
        f"實際用於本頁的資料筆數：{n}（未必等於 working.csv）。",
        (
            f"{feature}：min={float(x.min()):g}，max={float(x.max()):g}，"
            f"mean={float(x.mean()):g}。"
        ),
        (
            f"{target}：min={float(y.min()):g}，max={float(y.max()):g}，"
            f"mean={float(y.mean()):g}。"
        ),
        f"依 {feature} 排序後的樣本點（x, y）：" + "、".join(sample_bits) + "。",
        (
            f"依 {feature} 較低約 1/3 與較高約 1/3 的 {target} 平均："
            f"{low_y:g} vs {high_y:g}。"
        ),
        "請用上述摘要與散點直覺給線索；不要說出應選的選項文字。",
    ]
    return "\n".join(lines)


def is_slope_correct(choice: str, expected: str) -> bool:
    return choice == expected and choice in SLOPE_OPTIONS


def is_alpha_correct(choice: str) -> bool:
    return choice == ALPHA_CORRECT


def both_quiz_correct(
    slope_choice: str,
    alpha_choice: str,
    *,
    expected_slope: str,
) -> bool:
    return is_slope_correct(slope_choice, expected_slope) and is_alpha_correct(alpha_choice)


def quiz_choice_status(choice: str, *, correct: bool) -> str:
    if not choice or choice == PLEASE_SELECT:
        return "未選"
    return "正確" if correct else "錯誤"


def build_quiz_agent_appendix(
    *,
    slope_status: str,
    alpha_status: str,
    focus_qid: str | None,
    feature: str,
    target: str,
    learning_rate: float,
    unlocked: bool,
) -> str:
    focus = focus_qid or "無"
    lines = [
        "【訓練前預測關卡】",
        f"題1斜率方向狀態：{slope_status}；題2 α過大Cost狀態：{alpha_status}。",
        f"目前焦點題：{focus}。feature={feature}，target={target}，α={learning_rate:g}。",
        f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
        "未解鎖前請勿直接告訴學生應選哪一個選項；只給觀察散點或梯度更新的線索。",
        "可提醒學生看散點大致往哪個方向，或想像 α 太大時參數更新步長的影響。",
    ]
    return "\n".join(lines)


def hint_user_text(qid: str, *, feature: str, target: str, learning_rate: float) -> str:
    if qid == QID_SLOPE:
        return (
            f"我在單變量線性回歸的訓練前預測第1題（斜率方向）。"
            f"目前 feature=`{feature}`、target=`{target}`。"
            "請依散點給我判斷斜率正負／接近0的觀察线索，不要直接講正解選項。"
        )
    return (
        f"我在單變量線性回歸的訓練前預測第2題（α過大時Cost）。"
        f"目前設定的學習率 α={learning_rate:g}。"
        "請用梯度更新的直覺說明 α 太大時 Cost 可能怎樣，不要直接講正解選項。"
    )


def hint_display_text(qid: str) -> str:
    if qid == QID_SLOPE:
        return "（Agent 提示）請給我判斷斜率方向的線索，不要直接講正解。"
    return "（Agent 提示）請說明 α 太大時 Cost 可能怎樣，不要直接講正解。"


def can_send_hint(last_ts: float | None, now: float, *, cooldown: float = HINT_COOLDOWN_SEC) -> bool:
    if last_ts is None:
        return True
    return (now - float(last_ts)) >= cooldown


def focus_prompt_lines(focus_qid: str | None, *, unlocked: bool) -> list[str]:
    if not unlocked:
        if focus_qid == QID_ALPHA:
            return [
                "可按題2旁的「Agent 提示」詢問 α 與 Cost 的關係（請 Agent 不要直接講正解）。",
            ]
        return [
            "可按題1旁的「Agent 提示」詢問如何從散點判斷斜率方向（請 Agent 不要直接講正解）。",
        ]
    return [
        "請解釋這條回歸線代表什麼，並用 w 和 b 說明模型公式。",
        "請用 Cost J 說明這個模型目前預測得好不好。",
        "請找出誤差最大的幾筆資料，推測可能原因。",
    ]
