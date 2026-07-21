"""線性 SVM：訓練前預測關卡（pure helpers）。"""

from __future__ import annotations

from typing import Any

PLEASE_SELECT = "請選擇"
HINT_COOLDOWN_SEC = 2.5

# --- 階段 1：硬間隔 ---
OBJ_W_ZERO = "w 會變成 0，形不成有效邊界"
OBJ_ANY_BOUNDARY = "邊界隨便怎麼畫都可以"
OBJ_BEST_SPLIT = "一定會得到分得最好的邊界"
OBJ_NO_MARGIN = "跟 margin 無關，只影響計算速度"
OBJ_OPTIONS = (OBJ_W_ZERO, OBJ_ANY_BOUNDARY, OBJ_BEST_SPLIT, OBJ_NO_MARGIN)
OBJ_CORRECT = OBJ_W_ZERO

NORM_SMALLER = "變小（因為 Margin = 2 / ‖w‖）"
NORM_LARGER = "變大"
NORM_UNCHANGED = "幾乎不變"
NORM_ONLY_B = "只跟 b 有關，跟 ‖w‖ 無關"
NORM_OPTIONS = (NORM_SMALLER, NORM_LARGER, NORM_UNCHANGED, NORM_ONLY_B)
NORM_CORRECT = NORM_SMALLER

QID_OBJ = "objective"
QID_NORM = "norm"

SESSION_OBJ = "svm_hard_quiz_obj"
SESSION_NORM = "svm_hard_quiz_norm"
SESSION_HARD_PAIR = "svm_hard_quiz_pair"
SESSION_HARD_FOCUS = "svm_hard_quiz_focus"

# --- 階段 2：Soft Margin ---
HARD_INFEASIBLE = "找不到同時滿足所有限制的解，需要允許犯錯／soft margin"
HARD_ALWAYS_OK = "硬間隔一定還找得到完美分開的線"
HARD_ONLY_SLOWER = "只會讓訓練變慢，結果不變"
HARD_DROP_FEATURES = "必須刪掉一半的 feature"
HARD_OPTIONS = (HARD_INFEASIBLE, HARD_ALWAYS_OK, HARD_ONLY_SLOWER, HARD_DROP_FEATURES)
HARD_CORRECT = HARD_INFEASIBLE

C_MORE_CORRECT = "更在意把訓練點分對（較不容忍違規），margin 常因此變窄"
C_WIDER_MARGIN = "一定得到更寬的 margin"
C_IRRELEVANT = "跟分類無關，只改數字顯示"
C_ONLY_B = "只改變 b，不影響對錯誤的容忍"
C_OPTIONS = (C_MORE_CORRECT, C_WIDER_MARGIN, C_IRRELEVANT, C_ONLY_B)
C_CORRECT = C_MORE_CORRECT

QID_HARD = "hard_limit"
QID_C = "c_tradeoff"

SESSION_HARD = "svm_soft_quiz_hard"
SESSION_C = "svm_soft_quiz_c"
SESSION_SOFT_PAIR = "svm_soft_quiz_pair"
SESSION_SOFT_FOCUS = "svm_soft_quiz_focus"

SESSION_PAGE_FOCUS = "svm_page_quiz_focus"  # "hard" | "soft"


def pair_key(
    features: list[str] | tuple[str, ...],
    target: str,
    *,
    source_label: str,
    tab: str,
) -> tuple[Any, ...]:
    return (tab, source_label, tuple(features), target)


def needs_quiz_reset(
    stored_pair: Any,
    features: list[str] | tuple[str, ...],
    target: str,
    *,
    source_label: str,
    tab: str,
) -> bool:
    expected = pair_key(features, target, source_label=source_label, tab=tab)
    if stored_pair is None:
        return False
    if not isinstance(stored_pair, (tuple, list)):
        return True
    return tuple(stored_pair) != expected


def is_obj_correct(choice: str) -> bool:
    return choice == OBJ_CORRECT


def is_norm_correct(choice: str) -> bool:
    return choice == NORM_CORRECT


def both_hard_quiz_correct(obj_choice: str, norm_choice: str) -> bool:
    return is_obj_correct(obj_choice) and is_norm_correct(norm_choice)


def is_hard_limit_correct(choice: str) -> bool:
    return choice == HARD_CORRECT


def is_c_correct(choice: str) -> bool:
    return choice == C_CORRECT


def both_soft_quiz_correct(hard_choice: str, c_choice: str) -> bool:
    return is_hard_limit_correct(hard_choice) and is_c_correct(c_choice)


def quiz_choice_status(choice: str, *, correct: bool) -> str:
    if not choice or choice == PLEASE_SELECT:
        return "未選"
    return "正確" if correct else "錯誤"


def can_send_hint(last_ts: float | None, now: float, *, cooldown: float = HINT_COOLDOWN_SEC) -> bool:
    if last_ts is None:
        return True
    return (now - float(last_ts)) >= cooldown


def build_hard_quiz_agent_appendix(
    *,
    obj_status: str,
    norm_status: str,
    focus_qid: str | None,
    features: list[str],
    target: str,
    unlocked: bool,
) -> str:
    focus = focus_qid or "無"
    feature_txt = "、".join(features)
    return "\n".join(
        [
            "【訓練前預測關卡｜線性 SVM 階段 1 硬間隔】",
            f"題1（只有 ½‖w‖² 會怎樣）狀態：{obj_status}；題2（Margin 與 ‖w‖）狀態：{norm_status}。",
            f"目前焦點題：{focus}。features={feature_txt}，target={target}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "本階段只談硬間隔與最大化 margin；未解鎖前請勿直接告訴學生應選哪一個選項。",
            "可引導學生回想限制條件 y(w·x+b)≥1，或建議點開「為什麼目標會變成 ½‖w‖²」推導 expander。",
            "只給線索，不要直接講正解選項文字。",
        ]
    )


def build_soft_quiz_agent_appendix(
    *,
    hard_status: str,
    c_status: str,
    focus_qid: str | None,
    features: list[str],
    target: str,
    C: float,
    unlocked: bool,
) -> str:
    focus = focus_qid or "無"
    feature_txt = "、".join(features)
    return "\n".join(
        [
            "【訓練前預測關卡｜線性 SVM 階段 2 Soft Margin】",
            f"題1（硬間隔遇到不可分）狀態：{hard_status}；題2（C 變大偏向）狀態：{c_status}。",
            f"目前焦點題：{focus}。features={feature_txt}，target={target}，目前 UI 的 C={C:g}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "未解鎖前請勿直接告訴學生應選哪一個選項；只給 soft margin／C 取捨的線索。",
            "可提醒：上一階段假設資料分得開；本階段資料常有重疊。",
        ]
    )


def hard_hint_user_text(qid: str, *, features: list[str], target: str) -> str:
    feature_txt = "、".join(f"`{name}`" for name in features)
    if qid == QID_OBJ:
        return (
            "我在線性 SVM（硬間隔）訓練前預測第1題："
            "若只有最小化 ½‖w‖²、沒有分類限制會怎樣。"
            f"目前 features={feature_txt}，target=`{target}`。"
            "請給線索（可提 w=0 或限制條件的必要性），不要直接講正解選項。"
        )
    return (
        "我在線性 SVM（硬間隔）訓練前預測第2題：Margin 變大時 ‖w‖ 怎麼變。"
        f"目前 features={feature_txt}，target=`{target}`。"
        "請用 Margin=2/‖w‖ 的關係給線索，不要直接講正解選項。"
    )


def soft_hint_user_text(qid: str, *, features: list[str], target: str, C: float) -> str:
    feature_txt = "、".join(f"`{name}`" for name in features)
    if qid == QID_HARD:
        return (
            "我在線性 SVM（Soft Margin）訓練前預測第1題："
            "資料無法用直線完美分開時，硬間隔假設會怎樣。"
            f"目前 features={feature_txt}，target=`{target}`。"
            "請給線索（可行性／需要允許犯錯），不要直接講正解選項。"
        )
    return (
        "我在線性 SVM（Soft Margin）訓練前預測第2題：C 明顯變大時模型比較偏向什麼。"
        f"目前 features={feature_txt}，target=`{target}`，畫面上 C={C:g}。"
        "請給「分對 vs margin」取捨的線索，不要直接講正解選項。"
    )


def hard_hint_display_text(qid: str) -> str:
    if qid == QID_OBJ:
        return "（Agent 提示）請說明為什麼目標函數不能沒有限制，不要直接講正解。"
    return "（Agent 提示）請說明 Margin 與 ‖w‖ 的關係，不要直接講正解。"


def soft_hint_display_text(qid: str) -> str:
    if qid == QID_HARD:
        return "（Agent 提示）請說明硬間隔遇到不可分資料會怎樣，不要直接講正解。"
    return "（Agent 提示）請說明 C 變大時的取捨，不要直接講正解。"


SKLEARN_HARD_EXAMPLE = '''\
import pandas as pd
from sklearn.svm import SVC

# 讀取可線性分開的二分類資料（標籤轉成 -1 / +1）
df = pd.read_csv("svm_blobs_80.csv")
features = ["特徵1", "特徵2"]
X = df[features]
y = df["類別"].replace({0: -1, 1: 1})

# 線性核 SVM：最大化硬間隔
clf = SVC(kernel="linear")
clf.fit(X, y)

print("intercept:", clf.intercept_[0])
print("coef:", clf.coef_[0])
print("n_support:", clf.support_vectors_.shape[0])
print("predict sample:", clf.predict(X.iloc[:3]))
'''

SKLEARN_SOFT_EXAMPLE = '''\
import pandas as pd
from sklearn.svm import SVC

# 讀取類別有重疊、不易完美線性分開的資料
df = pd.read_csv("svm_soft_margin_80.csv")
features = ["特徵1", "特徵2"]
X = df[features]
y = df["類別"].replace({0: -1, 1: 1})

# Soft margin：用 C 權衡「顧 margin」與「顧分對」
# C 愈大 → 愈不容忍訓練錯誤；C 愈小 → 較寬的 margin、較能容忍違規
clf = SVC(kernel="linear", C=1.0)
clf.fit(X, y)

print("C:", clf.C)
print("intercept:", clf.intercept_[0])
print("coef:", clf.coef_[0])
print("n_support:", clf.support_vectors_.shape[0])
print("train accuracy:", clf.score(X, y))
'''
