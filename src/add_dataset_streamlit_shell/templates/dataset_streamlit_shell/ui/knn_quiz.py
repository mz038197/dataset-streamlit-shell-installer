"""K-近鄰分類：訓練前預測關卡（pure helpers）。"""

from __future__ import annotations

from typing import Any

PLEASE_SELECT = "請選擇"
HINT_COOLDOWN_SEC = 2.5

# 公式說明可談鄰居／多數決，但不要寫死具體 k 值
NEIGHBORS_FORMULA_CAPTION = (
    "其中 $y_{(i)}$ 是距離查詢點最近的第 i 個訓練點的類別；距離用歐氏，投票用多數決。"
)


def result_chart_caption() -> str:
    return (
        "按「開始預測演示」鎖定 k 並準備鄰居池；再按「下一步」讓查詢點進來、比距離、取鄰居、多數決。"
        "先走完 3 筆示範後，可在圖上點一下加演。"
    )


def query_prediction_caption(qx: float, qy: float, pred: int, *, k: int) -> str:
    return f"查詢點 ≈ ({qx:.3f}, {qy:.3f}) → 預測類別 **{pred}**（k={k}）"


def vote_progress_caption(tally: dict[int, int], *, finalized_pred: int | None = None) -> str:
    if not tally:
        return "等待鄰居進入…"
    parts = "、".join(f"類別 {lab}×{n}" for lab, n in sorted(tally.items()))
    if finalized_pred is None:
        return f"累計票數：{parts}"
    return f"{parts} → 預測類別 **{finalized_pred}**"


def stage1_ui_leaks_k(*texts: str) -> bool:
    """階段1 表面文案若洩漏具體 k 值或『k 固定』則 True。"""
    banned = ("k 固定", "k=5", "k = 5", "固定為 5", "（k=", "(k=")
    joined = "\n".join(texts)
    return any(token in joined for token in banned)

# --- 訓練前預測：實例型與多數決 ---
INST_NO_W = "沒有；是找出 k 個最近的訓練點再投票決定類別"
INST_HAS_W = "有；會先學一組權重 w 再算 w·x+b"
INST_CLUSTER = "沒有標籤也能分群，跟投票無關"
INST_ONLY_MEAN = "只算全部訓練點的平均當預測"
INST_OPTIONS = (INST_NO_W, INST_HAS_W, INST_CLUSTER, INST_ONLY_MEAN)
INST_CORRECT = INST_NO_W

VOTE_A = "A（多數決：A 出現兩次）"
VOTE_B = "B（因為最後一個鄰居是 B）"
VOTE_TIE = "一定平手，無法預測"
VOTE_RANDOM = "隨機選 A 或 B，跟鄰居無關"
VOTE_OPTIONS = (VOTE_A, VOTE_B, VOTE_TIE, VOTE_RANDOM)
VOTE_CORRECT = VOTE_A

QID_INST = "instance"
QID_VOTE = "vote"

SESSION_INST = "knn_neighbors_quiz_inst"
SESSION_VOTE = "knn_neighbors_quiz_vote"
SESSION_NEIGHBORS_PAIR = "knn_neighbors_quiz_pair"
SESSION_NEIGHBORS_FOCUS = "knn_neighbors_quiz_focus"


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


def is_inst_correct(choice: str) -> bool:
    return choice == INST_CORRECT


def is_vote_correct(choice: str) -> bool:
    return choice == VOTE_CORRECT


def both_neighbors_quiz_correct(inst_choice: str, vote_choice: str) -> bool:
    return is_inst_correct(inst_choice) and is_vote_correct(vote_choice)


def quiz_choice_status(choice: str, *, correct: bool) -> str:
    if not choice or choice == PLEASE_SELECT:
        return "未選"
    return "正確" if correct else "錯誤"


def can_send_hint(last_ts: float | None, now: float, *, cooldown: float = HINT_COOLDOWN_SEC) -> bool:
    if last_ts is None:
        return True
    return (now - float(last_ts)) >= cooldown


def build_neighbors_quiz_agent_appendix(
    *,
    inst_status: str,
    vote_status: str,
    focus_qid: str | None,
    features: list[str],
    target: str,
    unlocked: bool,
) -> str:
    focus = focus_qid or "無"
    feature_txt = "、".join(features)
    return "\n".join(
        [
            "【訓練前預測關卡｜K-近鄰分類】",
            f"題1（實例型／有無 w）狀態：{inst_status}；題2（多數決）狀態：{vote_status}。",
            f"目前焦點題：{focus}。features={feature_txt}，target={target}。",
            f"預測演示是否已解鎖：{'是' if unlocked else '否'}。",
            "本頁只談找鄰居與投票；未解鎖前請勿直接告訴學生應選哪一個選項。",
            "可對照邏輯迴歸／SVM 會學 w，K-近鄰則把訓練點留下來查。",
            "只給線索，不要直接講正解選項文字。",
        ]
    )


def neighbors_hint_user_text(qid: str, *, features: list[str], target: str) -> str:
    feature_txt = "、".join(f"`{name}`" for name in features)
    if qid == QID_INST:
        return (
            "我在 K-近鄰分類訓練前預測第1題："
            "預測時有沒有先學一組 w。"
            f"目前 features={feature_txt}，target=`{target}`。"
            "請給線索（可對照參數模型 vs 查鄰居），不要直接講正解選項。"
        )
    return (
        "我在 K-近鄰分類訓練前預測第2題："
        "三個鄰居標籤為 A、A、B 時多數決預測誰。"
        f"目前 features={feature_txt}，target=`{target}`。"
        "請給多數決線索，不要直接講正解選項。"
    )


def neighbors_hint_display_text(qid: str) -> str:
    if qid == QID_INST:
        return "（Agent 提示）請說明 K-近鄰是不是參數模型，不要直接講正解。"
    return "（Agent 提示）請說明多數決怎麼決定類別，不要直接講正解。"


SKLEARN_NEIGHBORS_EXAMPLE = '''\
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("knn_blobs_80.csv")
features = ["特徵1", "特徵2"]
X = StandardScaler().fit_transform(df[features])
y = df["類別"]

# 可調 k、歐氏距離、多數決（畫面上解鎖後以奇數 slider 設定 k）
clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
clf.fit(X, y)

print("predict sample:", clf.predict(X[:3]))
print("train accuracy:", clf.score(X, y))
'''
