"""K-近鄰分類：訓練前預測關卡（pure helpers）。"""

from __future__ import annotations

from typing import Any

PLEASE_SELECT = "請選擇"
HINT_COOLDOWN_SEC = 2.5

# 階段1 表面文案：可談「鄰居／多數決」，但不可出現具體 k 值或「k 固定為…」
NEIGHBORS_FORMULA_CAPTION = (
    "其中 $y_{(i)}$ 是距離查詢點最近的第 i 個訓練點的類別；距離用歐氏，投票用多數決。"
)


def result_chart_caption(*, expose_k: bool) -> str:
    if expose_k:
        return "在圖上**點一下**（點空白處或資料點附近）設定查詢點，會標出 k 個最近鄰居。"
    return "在圖上**點一下**（點空白處或資料點附近）設定查詢點，會標出最近的鄰居。"


def query_prediction_caption(
    qx: float, qy: float, pred: int, *, k: int, expose_k: bool
) -> str:
    base = f"查詢點 ≈ ({qx:.3f}, {qy:.3f}) → 預測類別 **{pred}**"
    if expose_k:
        return f"{base}（k={k}）"
    return base


def stage1_ui_leaks_k(*texts: str) -> bool:
    """階段1 表面文案若洩漏具體 k 值或『k 固定』則 True。"""
    banned = ("k 固定", "k=5", "k = 5", "固定為 5", "（k=", "(k=")
    joined = "\n".join(texts)
    return any(token in joined for token in banned)

# --- 階段 1：鄰居與投票 ---
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

# --- 階段 2：選擇 k ---
K_SMOOTH = "邊界過度平滑，預測常偏向訓練集裡的多數類"
K_SHARP = "邊界一定變得更尖銳、更貼近每個訓練點"
K_NO_EFFECT = "k 再大也幾乎不影響預測"
K_ONLY_SPEED = "只讓計算變慢，決策邏輯不變"
K_OPTIONS = (K_SMOOTH, K_SHARP, K_NO_EFFECT, K_ONLY_SPEED)
K_CORRECT = K_SMOOTH

SCALE_DOMINATE = "數值範圍較大的那個特徵會主導距離"
SCALE_EQUAL = "兩個特徵對距離的影響力一定一樣"
SCALE_ONLY_LABEL = "只影響類別欄位，不影響距離"
SCALE_DISABLE_K = "會讓 k 自動失效"
SCALE_OPTIONS = (SCALE_DOMINATE, SCALE_EQUAL, SCALE_ONLY_LABEL, SCALE_DISABLE_K)
SCALE_CORRECT = SCALE_DOMINATE

QID_K = "k_large"
QID_SCALE = "scale"

SESSION_K = "knn_k_quiz_k"
SESSION_SCALE = "knn_k_quiz_scale"
SESSION_K_PAIR = "knn_k_quiz_pair"
SESSION_K_FOCUS = "knn_k_quiz_focus"

SESSION_PAGE_FOCUS = "knn_page_quiz_focus"  # "neighbors" | "k"


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


def is_k_correct(choice: str) -> bool:
    return choice == K_CORRECT


def is_scale_correct(choice: str) -> bool:
    return choice == SCALE_CORRECT


def both_k_quiz_correct(k_choice: str, scale_choice: str) -> bool:
    return is_k_correct(k_choice) and is_scale_correct(scale_choice)


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
            "【訓練前預測關卡｜K-近鄰分類 階段1 鄰居與投票】",
            f"題1（實例型／有無 w）狀態：{inst_status}；題2（多數決）狀態：{vote_status}。",
            f"目前焦點題：{focus}。features={feature_txt}，target={target}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "本階段只談找鄰居與投票；未解鎖前請勿直接告訴學生應選哪一個選項。",
            "可對照邏輯迴歸／SVM 會學 w，K-近鄰則把訓練點留下來查。",
            "只給線索，不要直接講正解選項文字。",
        ]
    )


def build_k_quiz_agent_appendix(
    *,
    k_status: str,
    scale_status: str,
    focus_qid: str | None,
    features: list[str],
    target: str,
    k: int,
    standardize: bool,
    unlocked: bool,
) -> str:
    focus = focus_qid or "無"
    feature_txt = "、".join(features)
    return "\n".join(
        [
            "【訓練前預測關卡｜K-近鄰分類 階段2 選擇 k】",
            f"題1（k 很大）狀態：{k_status}；題2（未標準化尺度）狀態：{scale_status}。",
            f"目前焦點題：{focus}。features={feature_txt}，target={target}，UI 的 k={k}，標準化={'開' if standardize else '關'}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "未解鎖前請勿直接告訴學生應選哪一個選項；只給 k／尺度的線索。",
            "可提醒：距離被數值範圍大的軸主導時，另一軸幾乎沒聲音。",
        ]
    )


def neighbors_hint_user_text(qid: str, *, features: list[str], target: str) -> str:
    feature_txt = "、".join(f"`{name}`" for name in features)
    if qid == QID_INST:
        return (
            "我在 K-近鄰分類（鄰居與投票）訓練前預測第1題："
            "預測時有沒有先學一組 w。"
            f"目前 features={feature_txt}，target=`{target}`。"
            "請給線索（可對照參數模型 vs 查鄰居），不要直接講正解選項。"
        )
    return (
        "我在 K-近鄰分類（鄰居與投票）訓練前預測第2題："
        "三個鄰居標籤為 A、A、B 時多數決預測誰。"
        f"目前 features={feature_txt}，target=`{target}`。"
        "請給多數決線索，不要直接講正解選項。"
    )


def k_hint_user_text(
    qid: str, *, features: list[str], target: str, k: int, standardize: bool
) -> str:
    feature_txt = "、".join(f"`{name}`" for name in features)
    if qid == QID_K:
        return (
            "我在 K-近鄰分類（選擇 k）訓練前預測第1題："
            "k 變得很接近訓練筆數時，邊界／預測傾向什麼。"
            f"目前 features={feature_txt}，target=`{target}`，畫面上 k={k}。"
            "請給過度平滑／多數類線索，不要直接講正解選項。"
        )
    return (
        "我在 K-近鄰分類（選擇 k）訓練前預測第2題："
        "某一特徵數值範圍遠大於另一個、又沒標準化時，距離主要被誰主導。"
        f"目前 features={feature_txt}，target=`{target}`，標準化={'開' if standardize else '關'}。"
        "請給尺度與距離的線索，不要直接講正解選項。"
    )


def neighbors_hint_display_text(qid: str) -> str:
    if qid == QID_INST:
        return "（Agent 提示）請說明 K-近鄰是不是參數模型，不要直接講正解。"
    return "（Agent 提示）請說明多數決怎麼決定類別，不要直接講正解。"


def k_hint_display_text(qid: str) -> str:
    if qid == QID_K:
        return "（Agent 提示）請說明 k 很大時邊界會怎樣，不要直接講正解。"
    return "（Agent 提示）請說明未標準化時尺度如何影響距離，不要直接講正解。"


SKLEARN_NEIGHBORS_EXAMPLE = '''\
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("knn_blobs_80.csv")
features = ["特徵1", "特徵2"]
X = StandardScaler().fit_transform(df[features])
y = df["類別"]

# 固定 k、歐氏距離、多數決（不在畫面上調 k）
clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
clf.fit(X, y)

print("predict sample:", clf.predict(X[:3]))
print("train accuracy:", clf.score(X, y))
'''

SKLEARN_K_EXAMPLE = '''\
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("knn_scale_trap_80.csv")
features = ["特徵1", "特徵2"]
y = df["類別"]

# 建議先標準化，再調 k（奇數可減少平票）
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
clf.fit(X, y)

print("k:", clf.n_neighbors)
print("train accuracy:", clf.score(X, y))
'''
