"""邏輯迴歸：學習階段與訓練前預測（pure helpers）。"""

from __future__ import annotations

PLEASE_SELECT = "請選擇"
HINT_COOLDOWN_SEC = 2.5

STAGE_BOUNDARY = "線性邊界"
STAGE_POLY_LAMBDA = "多項式與 λ"
LEARNING_STAGES = (STAGE_BOUNDARY, STAGE_POLY_LAMBDA)

# --- 線性邊界 ---
SIGMOID_PROB = "把線性組合壓到 0～1，表示屬於類別 1 的機率"
SIGMOID_PM_LABEL = "直接輸出類別標籤 ±1"
SIGMOID_DISTANCE = "計算特徵之間的歐氏距離"
SIGMOID_ZERO_COST = "把 Cost 變成永遠為 0"
SIGMOID_OPTIONS = (SIGMOID_PROB, SIGMOID_PM_LABEL, SIGMOID_DISTANCE, SIGMOID_ZERO_COST)
SIGMOID_CORRECT = SIGMOID_PROB

COST_USES_PROBA = "Cost 依預測機率 f 與真實 y 計算，與分類 threshold 無關"
COST_FOLLOWS_THRESHOLD = "Cost 會隨 threshold 同步改變"
COST_THRESHOLD_COEFF = "threshold 決定 Cost 公式裡的係數"
COST_ONLY_HALF = "只有 threshold=0.5 時 Cost 才有意義"
COST_OPTIONS = (COST_USES_PROBA, COST_FOLLOWS_THRESHOLD, COST_THRESHOLD_COEFF, COST_ONLY_HALF)
COST_CORRECT = COST_USES_PROBA

# --- 多項式與 λ ---
MAP_NONLINEAR = "讓決策邊界能彎曲，捕捉非線性可分結構"
MAP_FEWER_FEATURES = "為了減少特徵數量、加快訓練"
MAP_CONTINUOUS_Y = "把標籤從 0／1 改成連續數值"
MAP_REPLACE_SIGMOID = "取代 sigmoid，改用線性輸出"
MAP_OPTIONS = (MAP_NONLINEAR, MAP_FEWER_FEATURES, MAP_CONTINUOUS_Y, MAP_REPLACE_SIGMOID)
MAP_CORRECT = MAP_NONLINEAR

LAMBDA_SMOOTHER = "權重被壓小，邊界通常更平滑，較不易過擬合訓練雜訊"
LAMBDA_FIT_HARDER = "讓模型一定擬合得更彎、更貼訓練點"
LAMBDA_COST_ZERO = "一定讓訓練 Cost 降到 0"
LAMBDA_ONLY_THRESHOLD = "與特徵映射無關，只改 threshold"
LAMBDA_OPTIONS = (LAMBDA_SMOOTHER, LAMBDA_FIT_HARDER, LAMBDA_COST_ZERO, LAMBDA_ONLY_THRESHOLD)
LAMBDA_CORRECT = LAMBDA_SMOOTHER

QID_SIGMOID = "sigmoid"
QID_COST = "cost"
QID_MAP = "map"
QID_LAMBDA = "lambda"

SESSION_SIGMOID = "logistic_quiz_sigmoid"
SESSION_COST = "logistic_quiz_cost"
SESSION_MAP = "logistic_quiz_map"
SESSION_LAMBDA = "logistic_quiz_lambda"
SESSION_FOCUS_BOUNDARY = "logistic_quiz_focus_boundary"
SESSION_FOCUS_POLY = "logistic_quiz_focus_poly"


def is_sigmoid_correct(choice: str) -> bool:
    return choice == SIGMOID_CORRECT


def is_cost_correct(choice: str) -> bool:
    return choice == COST_CORRECT


def both_boundary_quiz_correct(sigmoid_choice: str, cost_choice: str) -> bool:
    return is_sigmoid_correct(sigmoid_choice) and is_cost_correct(cost_choice)


def is_map_correct(choice: str) -> bool:
    return choice == MAP_CORRECT


def is_lambda_correct(choice: str) -> bool:
    return choice == LAMBDA_CORRECT


def both_poly_quiz_correct(map_choice: str, lambda_choice: str) -> bool:
    return is_map_correct(map_choice) and is_lambda_correct(lambda_choice)


def quiz_choice_status(choice: str, *, correct: bool) -> str:
    if not choice or choice == PLEASE_SELECT:
        return "未選"
    return "正確" if correct else "錯誤"


def can_send_hint(last_ts: float | None, now: float, *, cooldown: float = HINT_COOLDOWN_SEC) -> bool:
    if last_ts is None:
        return True
    return (now - float(last_ts)) >= cooldown


def build_boundary_quiz_agent_appendix(
    *,
    sigmoid_status: str,
    cost_status: str,
    focus_qid: str | None,
    unlocked: bool,
) -> str:
    focus = focus_qid or "無"
    return "\n".join(
        [
            "【訓練前預測關卡｜邏輯迴歸 線性邊界】",
            f"題1（sigmoid／機率）狀態：{sigmoid_status}；題2（Cost 與 threshold）狀態：{cost_status}。",
            f"目前焦點題：{focus}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "未解鎖前請勿直接告訴學生應選哪一個選項；只給機率輸出／Cost 定義的線索。",
        ]
    )


def build_poly_quiz_agent_appendix(
    *,
    map_status: str,
    lambda_status: str,
    focus_qid: str | None,
    unlocked: bool,
) -> str:
    focus = focus_qid or "無"
    return "\n".join(
        [
            "【訓練前預測關卡｜邏輯迴歸 多項式與 λ】",
            f"題1（多項式映射）狀態：{map_status}；題2（λ）狀態：{lambda_status}。",
            f"目前焦點題：{focus}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "未解鎖前請勿直接告訴學生應選哪一個選項；只給為何要映射／λ 壓權重的線索。",
        ]
    )


def hint_user_text(qid: str) -> str:
    if qid == QID_SIGMOID:
        return (
            "我在邏輯迴歸（線性邊界）訓練前預測第1題："
            "sigmoid 輸出大致代表什麼。"
            "請給「壓到 0～1 的機率」線索，不要直接講正解選項。"
        )
    if qid == QID_COST:
        return (
            "我在邏輯迴歸（線性邊界）訓練前預測第2題："
            "Cost 與分類 threshold 的關係。"
            "請給「Cost 看機率擬合」的線索，不要直接講正解選項。"
        )
    if qid == QID_MAP:
        return (
            "我在邏輯迴歸（多項式與 λ）訓練前預測第1題："
            "為什麼要多項式特徵映射。"
            "請給「讓邊界能彎曲／非線性」的線索，不要直接講正解選項。"
        )
    return (
        "我在邏輯迴歸（多項式與 λ）訓練前預測第2題："
        "λ 變大時決策邊界／過擬合傾向。"
        "請給「壓小權重、邊界更平滑」的線索，不要直接講正解選項。"
    )


def hint_display_text(qid: str) -> str:
    if qid == QID_SIGMOID:
        return "（Agent 提示）請說明 sigmoid 輸出像不像機率，不要直接講正解。"
    if qid == QID_COST:
        return "（Agent 提示）請說明 Cost 看的是什麼、跟 threshold 差在哪，不要直接講正解。"
    if qid == QID_MAP:
        return "（Agent 提示）請說明為什麼晶片資料常要特徵映射，不要直接講正解。"
    return "（Agent 提示）請說明 λ 變大時權重與邊界通常怎麼變，不要直接講正解。"


def focus_prompt_lines(focus_qid: str | None, *, stage: str, unlocked: bool) -> list[str]:
    if not unlocked:
        if stage == STAGE_BOUNDARY:
            if focus_qid == QID_COST:
                return ["可按題2旁的「Agent 提示」詢問 Cost 與 threshold（請 Agent 不要直接講正解）。"]
            return ["可按題1旁的「Agent 提示」詢問 sigmoid／機率（請 Agent 不要直接講正解）。"]
        if focus_qid == QID_LAMBDA:
            return ["可按題2旁的「Agent 提示」詢問 λ 的作用（請 Agent 不要直接講正解）。"]
        return ["可按題1旁的「Agent 提示」詢問多項式映射（請 Agent 不要直接講正解）。"]
    if stage == STAGE_BOUNDARY:
        return [
            "請解釋這條決策邊界代表什麼，以及錄取機率如何隨考試分數改變。",
            "請用 Cost J 說明模型目前擬合得好不好。",
            "調整 threshold 後，訓練集正確率如何變化？",
        ]
    return [
        "請解釋為什麼晶片資料需要多項式特徵映射與正則化。",
        "λ 變大時，決策邊界與 Cost 可能如何改變？",
        "請找出被判錯的樣本，推測可能原因。",
    ]
