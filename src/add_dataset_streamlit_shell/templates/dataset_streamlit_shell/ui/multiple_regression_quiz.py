"""多變量線性回歸：訓練前預測關卡（目的取向，pure helpers）。"""

from __future__ import annotations

from typing import Any

PLEASE_SELECT = "請選擇"
HINT_COOLDOWN_SEC = 2.5

PURPOSE_MULTI = "用多個因素一起解釋／預測同一個連續 target"
PURPOSE_CLASSIFY = "把分類問題改成回歸"
PURPOSE_ALWAYS_BETTER = "讓模型一定比單變量準"
PURPOSE_OPTIONS = (PURPOSE_MULTI, PURPOSE_CLASSIFY, PURPOSE_ALWAYS_BETTER)
PURPOSE_CORRECT = PURPOSE_MULTI

WEIGHTS_MEANING = "每個（縮放後）feature 對預測的貢獻方向與相對強度，再加上截距"
WEIGHTS_CLASS = "每個樣本該分到哪一類"
WEIGHTS_DROP = "哪個 feature 可以刪掉不管"
WEIGHTS_OPTIONS = (WEIGHTS_MEANING, WEIGHTS_CLASS, WEIGHTS_DROP)
WEIGHTS_CORRECT = WEIGHTS_MEANING

QID_PURPOSE = "purpose"
QID_WEIGHTS = "weights"

SESSION_PURPOSE = "multiple_reg_quiz_purpose"
SESSION_WEIGHTS = "multiple_reg_quiz_weights"
SESSION_PAIR = "multiple_reg_quiz_pair"
SESSION_FOCUS = "multiple_reg_quiz_focus"


def pair_key(features: list[str] | tuple[str, ...], target: str) -> tuple[Any, ...]:
    return (tuple(features), target)


def needs_quiz_reset(stored_pair: Any, features: list[str] | tuple[str, ...], target: str) -> bool:
    expected = pair_key(features, target)
    if stored_pair is None:
        return False
    if not isinstance(stored_pair, (tuple, list)) or len(stored_pair) != 2:
        return True
    stored_features, stored_target = stored_pair[0], stored_pair[1]
    if not isinstance(stored_features, (tuple, list)):
        return True
    return (tuple(stored_features), str(stored_target)) != expected


def is_purpose_correct(choice: str) -> bool:
    return choice == PURPOSE_CORRECT


def is_weights_correct(choice: str) -> bool:
    return choice == WEIGHTS_CORRECT


def both_quiz_correct(purpose_choice: str, weights_choice: str) -> bool:
    return is_purpose_correct(purpose_choice) and is_weights_correct(weights_choice)


def quiz_choice_status(choice: str, *, correct: bool) -> str:
    if not choice or choice == PLEASE_SELECT:
        return "未選"
    return "正確" if correct else "錯誤"


def build_quiz_agent_appendix(
    *,
    purpose_status: str,
    weights_status: str,
    focus_qid: str | None,
    features: list[str],
    target: str,
    learning_rate: float,
    unlocked: bool,
    use_housing_example: bool = False,
) -> str:
    focus = focus_qid or "無"
    feature_txt = "、".join(features)
    if use_housing_example:
        example_line = (
            "可對照本頁房價例子：多個房屋屬性共同預測房價；訓練後每個縮放後 feature 有對應的 w。"
        )
    else:
        example_line = (
            f"請用目前選定的 features（{feature_txt}）與 target（{target}）說明："
            "多個 x 如何共同預測同一個連續 y；訓練後每個縮放後 feature 有對應的 w。"
        )
    return "\n".join(
        [
            "【訓練前預測關卡｜多變量】",
            f"題1多feature目的狀態：{purpose_status}；題2 w／b意義狀態：{weights_status}。",
            f"目前焦點題：{focus}。features={feature_txt}，target={target}，α={learning_rate:g}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "未解鎖前請勿直接告訴學生應選哪一個選項；只給「為什麼要用多個 x」或「w／b 在表示什麼」的線索。",
            example_line,
        ]
    )


def hint_user_text(qid: str, *, features: list[str], target: str) -> str:
    feature_txt = "、".join(f"`{name}`" for name in features)
    if qid == QID_PURPOSE:
        return (
            "我在多變量線性回歸的訓練前預測第1題（為什麼要用多個 feature）。"
            f"目前 features={feature_txt}，target=`{target}`。"
            "請用本頁例子說明為什麼常需要多個 x 一起預測同一個連續 y，不要直接講正解選項。"
        )
    return (
        "我在多變量線性回歸的訓練前預測第2題（w、b 代表什麼）。"
        f"目前 features={feature_txt}，target=`{target}`。"
        "請說明訓練後多個 w 與 b 大致在表達什麼（可提到縮放後 feature），不要直接講正解選項。"
    )


def hint_display_text(qid: str) -> str:
    if qid == QID_PURPOSE:
        return "（Agent 提示）請說明為什麼要用多個 feature，不要直接講正解。"
    return "（Agent 提示）請說明 w、b 代表什麼，不要直接講正解。"


def can_send_hint(last_ts: float | None, now: float, *, cooldown: float = HINT_COOLDOWN_SEC) -> bool:
    if last_ts is None:
        return True
    return (now - float(last_ts)) >= cooldown


def focus_prompt_lines(focus_qid: str | None, *, unlocked: bool) -> list[str]:
    if not unlocked:
        if focus_qid == QID_WEIGHTS:
            return [
                "可按題2旁的「Agent 提示」詢問 w、b 的意義（請 Agent 不要直接講正解）。",
            ]
        return [
            "可按題1旁的「Agent 提示」詢問為什麼要用多個 feature（請 Agent 不要直接講正解）。",
        ]
    return [
        "請解釋每個 w 的正負方向，以及它和 target 的關係。",
        "請說明為什麼多變量線性回歸常需要特徵縮放。",
        "請找出預測誤差最大的資料列，並說明可能原因。",
    ]
