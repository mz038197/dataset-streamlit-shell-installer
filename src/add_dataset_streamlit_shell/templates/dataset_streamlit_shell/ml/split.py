"""Ready 資料切分：訓練／驗證／測試。"""

from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.2
DEFAULT_RANDOM_STATE = 42
RATIO_TOLERANCE = 1e-6


def ratios_sum_to_one(train_ratio: float, val_ratio: float, test_ratio: float) -> bool:
    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    return math.isclose(total, 1.0, abs_tol=RATIO_TOLERANCE)


def split_ready_frame(
    ready: pd.DataFrame,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if ready is None or ready.empty:
        raise ValueError("Ready 資料為空，無法切分")
    if not ratios_sum_to_one(train_ratio, val_ratio, test_ratio):
        raise ValueError("訓練／驗證／測試比例加總必須為 100%")
    if any(ratio < 0 for ratio in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("比例不可為負")

    stratify = None
    if stratify_column:
        if stratify_column not in ready.columns:
            raise ValueError(f"找不到分層欄位：{stratify_column}")
        stratify = ready[stratify_column]

    holdout_ratio = val_ratio + test_ratio
    train_df, holdout_df = train_test_split(
        ready,
        test_size=holdout_ratio,
        random_state=random_state,
        stratify=stratify,
    )

    if holdout_ratio <= 0:
        return train_df.reset_index(drop=True), holdout_df.iloc[0:0], holdout_df.iloc[0:0]

    relative_test = test_ratio / holdout_ratio
    holdout_stratify = None
    if stratify_column:
        holdout_stratify = holdout_df[stratify_column]

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test,
        random_state=random_state,
        stratify=holdout_stratify,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
