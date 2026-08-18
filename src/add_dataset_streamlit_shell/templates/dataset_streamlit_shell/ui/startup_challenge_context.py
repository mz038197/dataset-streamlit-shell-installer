"""專案展示（Challenge）軌道：路徑、host context、換公司規則（無 Streamlit 依賴）。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

CHALLENGE_COMPANIES: tuple[str, ...] = (
    "edupulse",
    "vitalrisk",
    "airsense",
    "churnlab",
    "flowcast",
)

DATA_VIEW_START = "起點"
DATA_VIEW_WORKING = "工作"
DATA_VIEW_TRAIN = "訓練"
DATA_VIEW_TEST = "測試"
DATA_VIEW_LABELS: tuple[str, ...] = (
    DATA_VIEW_START,
    DATA_VIEW_WORKING,
    DATA_VIEW_TRAIN,
    DATA_VIEW_TEST,
)

CHALLENGE_ARTIFACT_KEY = "challenge_model_artifact"
CHALLENGE_SPLIT_SIGNATURE_KEY = "challenge_split_signature"

COMPANY_SWITCH_DIALOG_TITLE = "確定更換挑戰公司？"
COMPANY_SWITCH_CONFIRM_LABEL = "確認更換"
COMPANY_SWITCH_CANCEL_LABEL = "取消"


def company_switch_dialog_body(new_company: str) -> str:
    return (
        "切換後會清除 Challenge 工作資料與切分，Challenge 模型產物會失效，"
        f"並還原專案展示空殼、重建對話。確定改為 **{new_company}**？"
    )

_COMPANY_FRAGMENTS: dict[str, str] = {
    "edupulse": (
        "【EduPulse 加碼】\n"
        "目標通常是 at_risk。強調不可用 G1/G2/G3 當特徵；不可把預測當懲處或分班依據。\n"
        "協助學生解釋：標籤可能是事後依成績規則產生，真實現場不會先有期末成績。"
    ),
    "vitalrisk": (
        "【VitalRisk 加碼】\n"
        "目標通常是 target。成果必須能談假陽性／假陰性；輸出只能當分流參考，不是診斷。\n"
        "提醒資料未必代表本地診所族群。"
    ),
    "airsense": (
        "【AirSense 加碼】\n"
        "目標通常是 CO_GT（回歸）。若看到 -200，先引導當成缺測／故障碼，不可當「空氣很好」或正常低值。\n"
        "預警文案避免恐嚇或絕對安全保證。"
    ),
    "churnlab": (
        "【ChurnLab 加碼】\n"
        "目標通常是 Churn。先協助類別清理與編碼思考；挽留建議必須包含「不做清單」。\n"
        "預測會流失 ≠ 立刻高強度打擾或差別待遇。"
    ),
    "flowcast": (
        "【FlowCast 加碼】\n"
        "目標通常是 cnt（回歸）。禁止使用會洩漏總量的拆分欄（如 casual/registered）。\n"
        "可引導討論 temp 與 atemp 是否都要留；調度建議需保留人工彈性。"
    ),
}


@dataclass(frozen=True)
class ChallengePaths:
    start_csv: Path
    handbook: Path
    working_csv: Path
    train_csv: Path
    test_csv: Path
    challenge_dir: Path


def challenge_paths(workspace_dir: Path, company: str) -> ChallengePaths:
    challenge_dir = workspace_dir / "challenge"
    return ChallengePaths(
        start_csv=challenge_dir / f"{company}.csv",
        handbook=challenge_dir / f"{company}_資料說明書.md",
        working_csv=challenge_dir / "working.csv",
        train_csv=challenge_dir / "train.csv",
        test_csv=challenge_dir / "test.csv",
        challenge_dir=challenge_dir,
    )


def company_changed_should_reset(
    previous: str | None,
    current: str,
) -> bool:
    if previous is None:
        return False
    return previous != current


def resolve_company_switch(
    committed: str | None,
    selected: str,
    pending: str | None = None,
) -> tuple[str, str | None]:
    """回傳（頁面要顯示的公司, 待確認的新公司）。待確認為 None 表示不跳更換挑戰公司確認。"""
    if committed is None:
        return selected, None
    if selected != committed:
        return committed, selected
    return committed, pending


def split_files_ready(paths: ChallengePaths) -> bool:
    return paths.train_csv.is_file() and paths.test_csv.is_file()


def split_signature(paths: ChallengePaths) -> tuple[float, int, float, int] | None:
    if not split_files_ready(paths):
        return None
    train_stat = paths.train_csv.stat()
    test_stat = paths.test_csv.stat()
    return (
        train_stat.st_mtime,
        train_stat.st_size,
        test_stat.st_mtime,
        test_stat.st_size,
    )


def artifact_matches_current_split(
    paths: ChallengePaths,
    signature: tuple[float, int, float, int] | None,
) -> bool:
    current = split_signature(paths)
    return current is not None and current == signature


def model_zone_unlocked(paths: ChallengePaths) -> bool:
    return split_files_ready(paths)


def result_zone_unlocked(paths: ChallengePaths, *, artifact_present: bool) -> bool:
    return split_files_ready(paths) and artifact_present


def available_data_views(paths: ChallengePaths) -> list[str]:
    mapping = (
        (DATA_VIEW_START, paths.start_csv),
        (DATA_VIEW_WORKING, paths.working_csv),
        (DATA_VIEW_TRAIN, paths.train_csv),
        (DATA_VIEW_TEST, paths.test_csv),
    )
    return [label for label, file_path in mapping if file_path.is_file()]


def default_data_view(paths: ChallengePaths) -> str:
    if paths.working_csv.is_file():
        return DATA_VIEW_WORKING
    return DATA_VIEW_START


def csv_for_view(paths: ChallengePaths, view: str) -> Path | None:
    return {
        DATA_VIEW_START: paths.start_csv,
        DATA_VIEW_WORKING: paths.working_csv,
        DATA_VIEW_TRAIN: paths.train_csv,
        DATA_VIEW_TEST: paths.test_csv,
    }.get(view)


def invalidate_challenge_split(paths: ChallengePaths) -> None:
    for file_path in (paths.train_csv, paths.test_csv):
        if file_path.is_file():
            file_path.unlink()


def clear_challenge_working(working_csv: Path) -> bool:
    """刪除 Challenge 工作資料。回傳是否真的刪除了檔案。"""
    if not working_csv.is_file():
        return False
    working_csv.unlink()
    return True


def clear_challenge_runtime(paths: ChallengePaths) -> None:
    clear_challenge_working(paths.working_csv)
    invalidate_challenge_split(paths)


def sync_split_if_working_stale(paths: ChallengePaths) -> bool:
    """working 比切分檔新時刪除 train／test。回傳是否作廢了切分。"""
    if not paths.working_csv.is_file():
        return False
    split_files = [path for path in (paths.train_csv, paths.test_csv) if path.is_file()]
    if not split_files:
        return False
    working_mtime = paths.working_csv.stat().st_mtime
    if not any(working_mtime > path.stat().st_mtime for path in split_files):
        return False
    invalidate_challenge_split(paths)
    return True


def restore_startup_challenge_ui(empty_shell: Path, live_ui: Path) -> None:
    shutil.copyfile(empty_shell, live_ui)


def challenge_host_context(
    *,
    company: str,
    start_csv: str,
    handbook: str,
    working_csv: str,
    train_csv: str,
    test_csv: str,
    scripts_dir: str,
) -> str:
    """專案展示專用 host_context；不叠加 dataset_base_context。"""
    fragment = _COMPANY_FRAGMENTS.get(
        company,
        f"【目前公司】\n{company}\n請依該公司資料說明書與委託情境協助；不要混用其他公司題目。",
    )
    return f"""【AI Startup Challenge 模式】
你是學生團隊的第五位隊友（資料／專案 Agent），正在協助完成 Startup Challenge 上台展示。
學生才是負責人：最後的清理決策、模型選擇、倫理界線與上台解釋，必須由學生做出並能說清楚。

【本頁目標：資料 → 模型區 → 成果區】
不要引導他們重做整條 ML 教學實驗室頁面。頁上幾乎沒有說明文字；引導發生在對話裡。
1) 先讀說明書、檢查 Challenge 起點資料，再複製成 Challenge 工作資料後清理。起點 CSV 只讀、不可覆寫。
2) 清理後從工作資料切出 Challenge 訓練資料與 Challenge 測試資料（預設 80／20；有類別目標則分層）。頁上無套用按鈕。模型只吃這兩份，不直接吃 working。
3) 沒有訓練／測試檔時，模型區與成果區只顯示空輪廓，不可填入。
4) 與學生討論要呈現的模型與方式後，以 AI coding 寫入 ui/startup_challenge_ui.py。
   - 模型區：選型與訓練（名稱、必要旋鈕、開始訓練），不要放成果圖表。
   - 成果區：訓練後的指標、圖與一次演示；沒有 Challenge 模型產物時維持空輪廓。
   一次 coding 可以寫兩區程式，但成果區在尚未訓練前仍應顯示空輪廓。
   訓練成功後請設定 st.session_state["challenge_model_artifact"]（任何非空值即可），成果區才會渲染。切分檔一變，這個產物會被清掉。
5) 倫理紅線只在對話與口頭 Gate 處理，不要在頁上加第三塊標題。
6) 不要拆掉專案展示空殼「無檔則顯示輪廓」的判斷。

【檔案規則】
- Challenge 起點資料：{start_csv}
  → 只讀起點，不要覆蓋。
- Challenge 資料說明書：{handbook}
  → 解釋欄位與目標前，先用工具讀取說明書。不要把說明書全文貼進頁面。
- Challenge 工作資料：{working_csv}
  （若尚未建立，先從起點檔複製再清理）
- Challenge 訓練資料：{train_csv}
- Challenge 測試資料：{test_csv}
- 改寫 working.csv 時必須刪除 train.csv 與 test.csv（切分作廢）。
- 不要改根目錄的 original.csv／working.csv／ready.csv／train.csv／val.csv／test.csv；那些是雙表整理線，不是本挑戰軌道。
- 不要上網下載替代資料集，不要改用其他公司的 CSV。
- 不要修改其他教學頁（邏輯迴歸、決策樹等）的程式，除非學生明確只要修專案展示頁。
- 若需寫檢查或整理腳本，只放在 {scripts_dir} 下。
- 不要編輯 ui/startup_challenge_empty_shell.py；那是換公司時還原用的專案展示空殼。

【允許改動範圍】
- ui/startup_challenge_ui.py（可填模型區／成果區；不可拆無檔則顯示輪廓）
- workspace/challenge/*（含 working.csv、train.csv、test.csv）
- 必要時 scripts/

【回答與行動規則】
1. 先理解問題：客戶是誰、目標欄是什麼、分類還是回歸。
2. 先讀說明書，再用 read_file／exec 實際看 CSV，不要憑記憶捏造欄位意義。
3. 發現缺失、異常、字串不一致時：先說明現象與選項利弊，再詢問學生要採哪一種；不要默默改完所有資料。
4. 幫寫訓練程式時：讀 challenge/train.csv 訓練、用 challenge/test.csv 評估。只做最小可運行版本。不要做超參大掃描或完整 UI 重構。
5. 用繁體中文、短句協助；技術細節可保留，但最後要能對客戶說人話。
6. 若學生只要分數、不管限制：把對話拉回該公司必講紅線。
7. 若學生要求你「直接全部做完讓我上台」：可以協助，但必須留下他們需要親口解釋的決策點（清理選擇、模型理由、倫理紅線）。
8. 沒有 train.csv 與 test.csv 時，拒絕填模型區／成果區，改引導先切分。

【禁止】
- 把結果說成確定診斷、確定會流失、空氣絕對安全、可完全自動調度。
- 建議歧視性、懲罰性或高強度打擾的自動化行動。
- 把會造成資料洩漏的欄位重新加回（若說明書已排除）。
- 覆寫 Challenge 起點 CSV。
- 剧透「老師故意埋了哪些缺陷」；改為引導學生自己檢查。
- 用 BOARD_* 常數或寫死分數當成果真相。

【目前公司】
{company}
請依該公司資料說明書與委託情境協助；不要混用其他公司題目。

{fragment}
"""


def challenge_page_snapshot(
    *,
    company: str,
    start_exists: bool,
    working_exists: bool,
    train_exists: bool,
    test_exists: bool,
    artifact_present: bool,
) -> str:
    def _state(exists: bool) -> str:
        return "存在" if exists else "不存在"

    return (
        "目前頁面：專案展示\n"
        f"挑戰公司：{company}\n"
        f"Challenge 起點資料：{_state(start_exists)}\n"
        f"Challenge 工作資料：{_state(working_exists)}\n"
        f"Challenge 訓練資料：{_state(train_exists)}\n"
        f"Challenge 測試資料：{_state(test_exists)}\n"
        f"Challenge 模型產物：{'有' if artifact_present else '無'}"
    )
