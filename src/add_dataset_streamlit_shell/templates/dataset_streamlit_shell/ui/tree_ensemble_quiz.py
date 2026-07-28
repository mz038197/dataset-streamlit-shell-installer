"""決策樹與集成：學習階段標籤與訓練前預測關卡（pure helpers）。"""

from __future__ import annotations

PLEASE_SELECT = "請選擇"
HINT_COOLDOWN_SEC = 2.5

STAGE1_LABEL = "單顆決策樹"
STAGE2_LABEL = "隨機森林（Bagging）"
STAGE3_LABEL = "XGBoost（Boosting）"

# --- 階段1：分裂／不純度 ---
ENTROPY_PURE_ZERO = "同一節點內全是同一類時，熵為 0（最純）"
ENTROPY_GAIN = "熵就是資訊增益，數字愈大愈好"
ENTROPY_ALWAYS_ONE = "熵永遠等於 1，跟類別混雜無關"
ENTROPY_ONLY_DEPTH = "熵只跟 max_depth 有關，跟標籤無關"
ENTROPY_OPTIONS = (ENTROPY_PURE_ZERO, ENTROPY_GAIN, ENTROPY_ALWAYS_ONE, ENTROPY_ONLY_DEPTH)
ENTROPY_CORRECT = ENTROPY_PURE_ZERO

IG_CORRECT = "選能讓不純度下降最多的分裂（資訊增益愈大愈好）"
IG_ALWAYS_ZERO = "資訊增益永遠是 0，分裂沒差"
IG_RANDOM = "隨機選一個 feature 分裂就好"
IG_ONLY_GINI = "只有 Gini 能算資訊增益，Entropy 不行"
IG_OPTIONS = (IG_CORRECT, IG_ALWAYS_ZERO, IG_RANDOM, IG_ONLY_GINI)

QID_ENTROPY = "entropy"
QID_IG = "ig"

SESSION_ENTROPY = "tree_ens_quiz_entropy"
SESSION_IG = "tree_ens_quiz_ig"
SESSION_TREE_FOCUS = "tree_ens_quiz_tree_focus"

# --- 階段2：Bagging ---
BAGGING_CORRECT = "多棵樹大致平行訓練，再把結果彙總（例如多數決）"
BAGGING_SEQ = "一棵接一棵序列訓練，後者專門改前一棵的錯"
BAGGING_ONE = "永遠只訓練一棵最深的樹"
BAGGING_NN = "先訓練類神經網路再轉成樹"
BAGGING_OPTIONS = (BAGGING_CORRECT, BAGGING_SEQ, BAGGING_ONE, BAGGING_NN)

VOTE_CORRECT = "多數決：看多數樹投哪一類"
VOTE_AVG = "一定是把每棵樹的深度加總"
VOTE_FIRST = "只採納第一棵樹的答案"
VOTE_LAST = "只採納最後一棵樹的答案"
VOTE_OPTIONS = (VOTE_CORRECT, VOTE_AVG, VOTE_FIRST, VOTE_LAST)

QID_BAGGING = "bagging"
QID_VOTE = "vote"

SESSION_BAGGING = "tree_ens_quiz_bagging"
SESSION_VOTE = "tree_ens_quiz_vote"
SESSION_BAG_FOCUS = "tree_ens_quiz_bag_focus"

# --- 階段3：Boosting 對照 ---
BOOST_CORRECT = "弱學習器序列訓練，後者針對前者錯誤調整"
BOOST_PARALLEL = "跟 Bagging 一樣：多棵樹平行訓練再多數決"
BOOST_ONLY_LR = "只改 learning_rate，訓練方式與單顆樹完全相同"
BOOST_NO_TREES = "XGBoost 不用樹，只用線性回歸"
BOOST_OPTIONS = (BOOST_CORRECT, BOOST_PARALLEL, BOOST_ONLY_LR, BOOST_NO_TREES)

CONTRAST_CORRECT = "Bagging 偏平行多數決；Boosting 偏序列糾錯"
CONTRAST_BAGGING = "兩者都是平行多數決，沒有差別"
CONTRAST_SAME_SEQ = "兩者都是序列糾錯，沒有差別"
CONTRAST_ONLY_NAME = "只是套件名字不同，訓練流程一樣"
CONTRAST_OPTIONS = (CONTRAST_CORRECT, CONTRAST_BAGGING, CONTRAST_SAME_SEQ, CONTRAST_ONLY_NAME)

QID_BOOST = "boost"
QID_CONTRAST = "contrast"

SESSION_BOOST = "tree_ens_quiz_boost"
SESSION_CONTRAST = "tree_ens_quiz_contrast"
SESSION_BOOST_FOCUS = "tree_ens_quiz_boost_focus"

SESSION_PAGE_FOCUS = "tree_ens_page_focus"  # "tree" | "bagging" | "boost"


def learning_stage_labels() -> tuple[str, str, str]:
    return (STAGE1_LABEL, STAGE2_LABEL, STAGE3_LABEL)


def quiz_choice_status(choice: str, *, correct: bool) -> str:
    if not choice or choice == PLEASE_SELECT:
        return "未選"
    return "正確" if correct else "錯誤"


def can_send_hint(last_ts: float | None, now: float, *, cooldown: float = HINT_COOLDOWN_SEC) -> bool:
    if last_ts is None:
        return True
    return (now - float(last_ts)) >= cooldown


def is_entropy_correct(choice: str) -> bool:
    return choice == ENTROPY_CORRECT


def is_ig_correct(choice: str) -> bool:
    return choice == IG_CORRECT


def both_tree_quiz_correct(entropy_choice: str, ig_choice: str) -> bool:
    return is_entropy_correct(entropy_choice) and is_ig_correct(ig_choice)


def is_bagging_correct(choice: str) -> bool:
    return choice == BAGGING_CORRECT


def is_vote_correct(choice: str) -> bool:
    return choice == VOTE_CORRECT


def both_bagging_quiz_correct(bagging_choice: str, vote_choice: str) -> bool:
    return is_bagging_correct(bagging_choice) and is_vote_correct(vote_choice)


def is_boost_correct(choice: str) -> bool:
    return choice == BOOST_CORRECT


def is_contrast_correct(choice: str) -> bool:
    return choice == CONTRAST_CORRECT


def both_boost_quiz_correct(boost_choice: str, contrast_choice: str) -> bool:
    return is_boost_correct(boost_choice) and is_contrast_correct(contrast_choice)


def build_tree_quiz_agent_appendix(
    *,
    entropy_status: str,
    ig_status: str,
    focus_qid: str | None,
    unlocked: bool,
) -> str:
    focus = focus_qid or "無"
    return "\n".join(
        [
            "【訓練前預測關卡｜決策樹與集成 階段1 單顆決策樹】",
            f"題1（熵／純度）狀態：{entropy_status}；題2（資訊增益／分裂）狀態：{ig_status}。",
            f"目前焦點題：{focus}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "未解鎖前請勿直接告訴學生應選哪一個選項；只給熵與資訊增益的線索。",
        ]
    )


def build_bagging_quiz_agent_appendix(
    *,
    bagging_status: str,
    vote_status: str,
    focus_qid: str | None,
    unlocked: bool,
    n_estimators: int,
) -> str:
    focus = focus_qid or "無"
    return "\n".join(
        [
            "【訓練前預測關卡｜決策樹與集成 階段2 隨機森林（Bagging）】",
            f"題1（Bagging 怎麼訓）狀態：{bagging_status}；題2（怎麼彙總）狀態：{vote_status}。",
            f"目前焦點題：{focus}；目前 UI 的 n_estimators={n_estimators}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "未解鎖前請勿直接告訴學生應選哪一個選項；強調平行訓練與多數決。",
        ]
    )


def build_boost_quiz_agent_appendix(
    *,
    boost_status: str,
    contrast_status: str,
    focus_qid: str | None,
    unlocked: bool,
    learning_rate: float,
) -> str:
    focus = focus_qid or "無"
    return "\n".join(
        [
            "【訓練前預測關卡｜決策樹與集成 階段3 XGBoost（Boosting）】",
            f"題1（Boosting 怎麼訓）狀態：{boost_status}；題2（與 Bagging 對照）狀態：{contrast_status}。",
            f"目前焦點題：{focus}；目前 UI 的 learning_rate={learning_rate:g}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "未解鎖前請勿直接告訴學生應選哪一個選項；強調序列糾錯 vs 平行多數決。",
        ]
    )


def tree_hint_user_text(qid: str) -> str:
    if qid == QID_ENTROPY:
        return (
            "我在決策樹與集成（單顆決策樹）訓練前預測第1題：熵與純度的關係。"
            "請給線索（可提「全是同一類」），不要直接講正解選項。"
        )
    return (
        "我在決策樹與集成（單顆決策樹）訓練前預測第2題：訓練時怎麼選分裂。"
        "請給資訊增益／不純度下降的線索，不要直接講正解選項。"
    )


def bagging_hint_user_text(qid: str, *, n_estimators: int) -> str:
    if qid == QID_BAGGING:
        return (
            "我在決策樹與集成（隨機森林／Bagging）訓練前預測第1題："
            f"Bagging 多棵樹怎麼訓練。目前 n_estimators={n_estimators}。"
            "請給「平行 vs 序列」線索，不要直接講正解選項。"
        )
    return (
        "我在決策樹與集成（隨機森林／Bagging）訓練前預測第2題：多棵樹的預測怎麼彙總。"
        f"目前 n_estimators={n_estimators}。"
        "請給多數決線索，不要直接講正解選項。"
    )


def boost_hint_user_text(qid: str, *, learning_rate: float) -> str:
    if qid == QID_BOOST:
        return (
            "我在決策樹與集成（XGBoost／Boosting）訓練前預測第1題："
            f"Boosting 怎麼訓練。目前 learning_rate={learning_rate:g}。"
            "請給序列糾錯線索，不要直接講正解選項。"
        )
    return (
        "我在決策樹與集成（XGBoost／Boosting）訓練前預測第2題："
        "Bagging 與 Boosting 的主要差異。"
        f"目前 learning_rate={learning_rate:g}。"
        "請對照平行多數決 vs 序列糾錯，不要直接講正解選項。"
    )


def bagging_vs_boosting_contrast_markdown() -> str:
    return (
        "### Bagging vs Boosting（對照）\n\n"
        "| | **Bagging（本頁：隨機森林）** | **Boosting（本頁：XGBoost）** |\n"
        "|---|---|---|\n"
        "| 訓練節奏 | 多棵樹大致**平行**訓練 | 弱學習器**序列**訓練 |\n"
        "| 彼此關係 | 各自獨立，再**多數決**彙總 | 後者針對前者**錯誤**調整 |\n"
        "| 本頁旋鈕 | `n_estimators`（幾棵平行樹） | `learning_rate`（每步走多大幅） |\n"
    )
