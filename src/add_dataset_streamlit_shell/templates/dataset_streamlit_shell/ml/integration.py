"""鐵達尼雙表整合：載入、對鍵、merge、類別正規化對照與關卡正解。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SHELL_ROOT = Path(__file__).resolve().parent.parent
INTEGRATION_DATA_DIR = SHELL_ROOT / "built-in-data" / "integration"
PASSENGERS_PATH = INTEGRATION_DATA_DIR / "titanic_passengers.csv"
VOYAGE_PATH = INTEGRATION_DATA_DIR / "titanic_voyage.csv"

PASSENGER_KEY = "PassengerId"
VOYAGE_KEY_RAW = "passenger_id"
PASSENGER_TABLE_LABEL = "乘客表"
VOYAGE_TABLE_LABEL = "航程表"

SEX_CANONICAL = {
    "male": "male",
    "Male": "male",
    "M": "male",
    "man": "male",
    "female": "female",
    "Female": "female",
    "F": "female",
    "woman": "female",
}

EMBARKED_CANONICAL = {
    "S": "S",
    "s": "S",
    "Southampton": "S",
    "C": "C",
    "Cherbourg": "C",
    "Q": "Q",
    "Queenstown": "Q",
}

KEY_ALIGN_CORRECT = "須先把航程表的 passenger_id 改成與乘客表相同的鍵名再合併"
KEY_ALIGN_OPTIONS = (
    "請選擇",
    KEY_ALIGN_CORRECT,
    "兩表已經同名，直接合併即可",
    "用 Name 當鍵比較直覺",
)

LEFT_JOIN_CORRECT = "left"
JOIN_HOW_OPTIONS = ("請選擇", "inner", "left")


def load_titanic_integration_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PASSENGERS_PATH.is_file() or not VOYAGE_PATH.is_file():
        raise FileNotFoundError(
            f"缺少整合教材：{PASSENGERS_PATH.name} / {VOYAGE_PATH.name}"
        )
    passengers = pd.read_csv(PASSENGERS_PATH)
    voyage = pd.read_csv(VOYAGE_PATH)
    return passengers, voyage


def align_voyage_key(voyage: pd.DataFrame) -> pd.DataFrame:
    frame = voyage.copy()
    if VOYAGE_KEY_RAW in frame.columns and PASSENGER_KEY not in frame.columns:
        frame = frame.rename(columns={VOYAGE_KEY_RAW: PASSENGER_KEY})
    return frame


def key_overlap_count(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
) -> int:
    if left_on not in left.columns or right_on not in right.columns:
        return 0
    left_keys = set(left[left_on].dropna().tolist())
    right_keys = set(right[right_on].dropna().tolist())
    return len(left_keys & right_keys)


def merge_passenger_voyage(
    passengers: pd.DataFrame,
    voyage: pd.DataFrame,
    *,
    how: str = "left",
    align_key: bool = False,
) -> pd.DataFrame:
    if how not in {"inner", "left"}:
        raise ValueError(f"unsupported join how: {how}")
    right = align_voyage_key(voyage) if align_key else voyage.copy()
    if PASSENGER_KEY not in passengers.columns or PASSENGER_KEY not in right.columns:
        raise ValueError("合併前請先對齊鍵名為 PassengerId")
    return passengers.merge(right, on=PASSENGER_KEY, how=how)


def normalize_sex_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.map(lambda value: SEX_CANONICAL.get(str(value), pd.NA))


def normalize_embarked_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.map(lambda value: EMBARKED_CANONICAL.get(str(value), pd.NA))


def is_key_align_correct(choice: str) -> bool:
    return choice == KEY_ALIGN_CORRECT


def is_join_how_correct(choice: str) -> bool:
    return choice == LEFT_JOIN_CORRECT
