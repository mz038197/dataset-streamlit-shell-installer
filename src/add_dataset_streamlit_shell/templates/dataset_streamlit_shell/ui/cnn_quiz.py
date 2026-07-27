"""卷積神經網路：學習階段／觀念主軸影片／訓練前預測（pure helpers）。"""

from __future__ import annotations

PLEASE_SELECT = "請選擇"
HINT_COOLDOWN_SEC = 2.5

VIDEO_ID = "YGILT182T6w"
VIDEO_TITLE = "DataMListic｜CNNs - Explained"

STAGE_MATRIX = "圖片與矩陣"
STAGE_WHY = "為什麼需要 CNN"
STAGE_CONV = "卷積與特徵圖"
STAGE_RELU_POOL = "ReLU 與 Pooling"
STAGE_FLOW = "CNN 整條流程"
STAGE_HANDS_ON = "動手做 CNN"

LEARNING_STAGES = (
    STAGE_MATRIX,
    STAGE_WHY,
    STAGE_CONV,
    STAGE_RELU_POOL,
    STAGE_FLOW,
    STAGE_HANDS_ON,
)

# 建議起點秒數（對齊字幕關鍵轉折；階段1不嵌片）
STAGE_VIDEO_START_SEC: dict[str, int] = {
    STAGE_WHY: 17,  # 餵影像進一般網路會怎樣
    STAGE_CONV: 87,  # convolution / kernel
    STAGE_RELU_POOL: 365,  # pooling（ReLU 以頁內圖為主）
    STAGE_FLOW: 387,  # 串層＋縮空間長通道
    STAGE_HANDS_ON: 581,  # 收束總結後動手
}

# --- 訓練前預測（階段6）---
KERNEL_TEMPLATE = "用小模板（kernel）在圖上滑動，找出像這個形狀的局部特徵"
KERNEL_SLIDE_ONLY = "只是把整張圖縮小，不管像不像什麼"
KERNEL_GLOBAL = "一次看完整張圖的所有像素再輸出一個答案"
KERNEL_LABEL = "直接讀出圖片檔名裡的類別"
KERNEL_OPTIONS = (KERNEL_TEMPLATE, KERNEL_SLIDE_ONLY, KERNEL_GLOBAL, KERNEL_LABEL)
KERNEL_CORRECT = KERNEL_TEMPLATE

POOL_KEEP_MAX = "縮小特徵圖，並在每個小區塊留下最強的反應（例如取最大）"
POOL_AVERAGE_ONLY = "一定只能做平均，不能取最大"
POOL_ADD_PARAMS = "專門用來增加可訓練參數數量"
POOL_COLOR = "把灰階圖變成彩色"
POOL_OPTIONS = (POOL_KEEP_MAX, POOL_AVERAGE_ONLY, POOL_ADD_PARAMS, POOL_COLOR)
POOL_CORRECT = POOL_KEEP_MAX

QID_KERNEL = "kernel"
QID_POOL = "pool"

SESSION_KERNEL = "cnn_quiz_kernel"
SESSION_POOL = "cnn_quiz_pool"
SESSION_FOCUS = "cnn_quiz_focus"
SESSION_PAGE_FOCUS = "cnn_page_quiz_focus"


def stage_video_start_sec(stage: str) -> int | None:
    return STAGE_VIDEO_START_SEC.get(stage)


def youtube_embed_url(video_id: str, *, start_sec: int = 0) -> str:
    start = max(0, int(start_sec))
    return f"https://www.youtube.com/embed/{video_id}?start={start}"


def youtube_watch_url(video_id: str, *, start_sec: int = 0) -> str:
    start = max(0, int(start_sec))
    return f"https://www.youtube.com/watch?v={video_id}&t={start}s"


def is_kernel_correct(choice: str) -> bool:
    return choice == KERNEL_CORRECT


def is_pool_correct(choice: str) -> bool:
    return choice == POOL_CORRECT


def both_quiz_correct(kernel_choice: str, pool_choice: str) -> bool:
    return is_kernel_correct(kernel_choice) and is_pool_correct(pool_choice)


def quiz_choice_status(choice: str, *, correct: bool) -> str:
    if not choice or choice == PLEASE_SELECT:
        return "未選"
    return "正確" if correct else "錯誤"


def can_send_hint(last_ts: float | None, now: float, *, cooldown: float = HINT_COOLDOWN_SEC) -> bool:
    if last_ts is None:
        return True
    return (now - float(last_ts)) >= cooldown


def build_quiz_agent_appendix(
    *,
    kernel_status: str,
    pool_status: str,
    focus_qid: str | None,
    unlocked: bool,
) -> str:
    focus = focus_qid or "無"
    return "\n".join(
        [
            "【訓練前預測關卡｜卷積神經網路 動手做 CNN】",
            f"題1（kernel／卷積）狀態：{kernel_status}；題2（pooling）狀態：{pool_status}。",
            f"目前焦點題：{focus}。",
            f"訓練是否已解鎖：{'是' if unlocked else '否'}。",
            "未解鎖前請勿直接告訴學生應選哪一個選項；只給局部模板比對／縮小留重點的線索。",
        ]
    )


def hint_user_text(qid: str) -> str:
    if qid == QID_KERNEL:
        return (
            "我在卷積神經網路（動手做 CNN）訓練前預測第1題："
            "kernel／卷積在影像上大致在做什麼。"
            "請給「小模板滑動找局部特徵」的線索，不要直接講正解選項。"
        )
    return (
        "我在卷積神經網路（動手做 CNN）訓練前預測第2題："
        "pooling 大致留下什麼、丟掉什麼。"
        "請給「縮小並保留區塊內最強反應」的線索，不要直接講正解選項。"
    )


def hint_display_text(qid: str) -> str:
    if qid == QID_KERNEL:
        return "（Agent 提示）請說明 kernel 像不像「小模板在圖上找相似形狀」，不要直接講正解。"
    return "（Agent 提示）請說明 pooling 為什麼常取區塊最大值／縮小圖，不要直接講正解。"
