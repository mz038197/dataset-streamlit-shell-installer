"""專案展示（Challenge）軌道：路徑、host context、換公司規則（無 Streamlit 依賴）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CHALLENGE_COMPANIES: tuple[str, ...] = (
    "edupulse",
    "vitalrisk",
    "airsense",
    "churnlab",
    "flowcast",
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
    challenge_dir: Path


def challenge_paths(workspace_dir: Path, company: str) -> ChallengePaths:
    challenge_dir = workspace_dir / "challenge"
    return ChallengePaths(
        start_csv=challenge_dir / f"{company}.csv",
        handbook=challenge_dir / f"{company}_資料說明書.md",
        working_csv=challenge_dir / "working.csv",
        challenge_dir=challenge_dir,
    )


def company_changed_should_clear_working(
    previous: str | None,
    current: str,
) -> bool:
    if previous is None:
        return False
    return previous != current


def clear_challenge_working(working_csv: Path) -> bool:
    """刪除 Challenge 工作資料。回傳是否真的刪除了檔案。"""
    if not working_csv.is_file():
        return False
    working_csv.unlink()
    return True


def challenge_host_context(
    *,
    company: str,
    start_csv: str,
    handbook: str,
    working_csv: str,
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

【本頁目標：薄白板三塊】
協助學生完成並強化這三塊即可，不要引導他們重做整條 ML 教學實驗室頁面：
1) 我們在解決什麼？（客戶、問題、預測目標欄）
2) 我們做出來的結果（一個主模型、一個評估指標、一次可演示預測）
3) 我們不能亂承諾什麼？（限制與倫理紅線）
成果請以 AI coding 寫入允許檔案（尤其 ui/startup_challenge_ui.py 的白板常數與第②塊實作），不要只把文案留在對話裡。

【檔案規則】
- Challenge 起點資料：{start_csv}
  → 只讀起點，不要覆蓋。
- Challenge 資料說明書：{handbook}
  → 解釋欄位與目標前，先用工具讀取說明書。
- Challenge 工作資料：{working_csv}
  （若尚未建立，可提醒學生從起點檔複製後再清理）
- 展示／訓練請優先使用 Challenge 工作資料；沒有工作資料時，可以先讀起點檔做診斷，但改檔前必須先建立工作副本。
- 不要改根目錄的 original.csv／working.csv／ready.csv；那些是雙表整理線，不是本挑戰軌道。
- 不要上網下載替代資料集，不要改用其他公司的 CSV。
- 不要修改其他教學頁（邏輯迴歸、決策樹等）的程式，除非學生明確只要修專案展示頁。
- 若需寫檢查或整理腳本，只放在 {scripts_dir} 下。

【允許改動範圍】
- ui/startup_challenge_ui.py
- workspace/challenge/*（含 working.csv）
- 必要時 scripts/

【回答與行動規則】
1. 先理解問題：客戶是誰、目標欄是什麼、分類還是回歸。
2. 先讀說明書，再用 read_file／exec 實際看 CSV，不要憑記憶捏造欄位意義。
3. 發現缺失、異常、字串不一致時：先說明現象與選項利弊，再詢問學生要採哪一種；不要默默改完所有資料。
4. 幫寫訓練程式時：只做最小可運行版本（切分、一個模型、一個指標、可選的一次預測）。不要做超參大掃描或完整 UI 重構。
5. 用繁體中文、短句、可上台的口吻協助改寫解釋；技術細節可保留，但最後要能對客戶說人話。
6. 若學生只要分數、不管限制：把對話拉回第 3 塊（不能亂承諾什麼）。
7. 若學生要求你「直接全部做完讓我上台」：可以協助，但必須留下他們需要親口解釋的決策點（清理選擇、模型理由、倫理紅線）。

【禁止】
- 把結果說成確定診斷、確定會流失、空氣絕對安全、可完全自動調度。
- 建議歧視性、懲罰性或高強度打擾的自動化行動。
- 把會造成資料洩漏的欄位重新加回（若說明書已排除）。
- 覆寫 Challenge 起點 CSV。
- 剧透「老師故意埋了哪些缺陷」；改為引導學生自己檢查。

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
    board_summary: str,
    target_column: str = "",
) -> str:
    start_state = "存在" if start_exists else "不存在"
    working_state = "存在" if working_exists else "不存在"
    target_line = target_column.strip() or "（尚未在白板常數填寫）"
    return (
        "目前頁面：專案展示（AI Startup Challenge｜成果展示）\n"
        f"挑戰公司：{company}\n"
        f"Challenge 起點資料：{start_state}\n"
        f"Challenge 工作資料：{working_state}\n"
        f"白板狀態：{board_summary}\n"
        f"學生可見目標欄：{target_line}"
    )


def board_status_summary(
    *,
    customer: str,
    problem: str,
    task_type: str,
    target_column: str,
    model_name: str,
    metric_line: str,
    limits_text: str,
) -> str:
    block1 = "①已填" if all(v.strip() for v in (customer, problem, task_type, target_column)) else "①空殼"
    block2 = "②已填" if model_name.strip() and metric_line.strip() else "②TODO"
    block3 = "③已填" if limits_text.strip() else "③空殼"
    return f"{block1} {block2} {block3}"
