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


def dual_table_copy_paths(workspace: Path) -> tuple[Path, Path]:
    folder = workspace / "integration"
    return folder / "passengers.csv", folder / "voyage.csv"


def write_dual_table_copies(
    workspace: Path,
    passengers: pd.DataFrame,
    voyage: pd.DataFrame,
) -> None:
    passengers_path, voyage_path = dual_table_copy_paths(workspace)
    passengers_path.parent.mkdir(parents=True, exist_ok=True)
    passengers.to_csv(passengers_path, index=False)
    voyage.to_csv(voyage_path, index=False)


def clear_dual_table_copies(workspace: Path) -> None:
    passengers_path, voyage_path = dual_table_copy_paths(workspace)
    for path in (passengers_path, voyage_path):
        if path.is_file():
            path.unlink()
    folder = passengers_path.parent
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()


def load_dual_tables(workspace: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    builtin_p, builtin_v = load_titanic_integration_frames()
    passengers_path, voyage_path = dual_table_copy_paths(workspace)
    passengers = pd.read_csv(passengers_path) if passengers_path.is_file() else builtin_p
    voyage = pd.read_csv(voyage_path) if voyage_path.is_file() else builtin_v
    return passengers, voyage


def voyage_key_is_aligned(voyage: pd.DataFrame) -> bool:
    return PASSENGER_KEY in voyage.columns and VOYAGE_KEY_RAW not in voyage.columns


def merge_from_dual_tables(workspace: Path, *, how: str) -> pd.DataFrame:
    passengers, voyage = load_dual_tables(workspace)
    if not voyage_key_is_aligned(voyage):
        raise ValueError("未對齊鍵名不得合併")
    return merge_passenger_voyage(passengers, voyage, how=how, align_key=False)


def commit_dual_table_merge(
    workspace: Path,
    *,
    how: str,
    original_path: Path,
    working_path: Path,
) -> pd.DataFrame:
    merged = merge_from_dual_tables(workspace, how=how)
    original_path.parent.mkdir(parents=True, exist_ok=True)
    working_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(original_path, index=False)
    merged.to_csv(working_path, index=False)
    clear_dual_table_copies(workspace)
    return merged


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
