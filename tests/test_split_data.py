from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.split import (  # noqa: E402
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    ratios_sum_to_one,
    split_ready_frame,
)


def test_default_ratios_are_60_20_20() -> None:
    assert DEFAULT_TRAIN_RATIO == pytest.approx(0.6)
    assert DEFAULT_VAL_RATIO == pytest.approx(0.2)
    assert DEFAULT_TEST_RATIO == pytest.approx(0.2)
    assert ratios_sum_to_one(DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO)


def test_split_row_counts_sum_to_ready() -> None:
    ready = pd.DataFrame(
        {
            "Survived": [0, 1] * 50,
            "x": range(100),
        }
    )
    train, val, test = split_ready_frame(
        ready,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
        stratify_column="Survived",
    )
    assert len(train) + len(val) + len(test) == len(ready)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20


def test_stratified_split_preserves_class_balance_roughly() -> None:
    ready = pd.DataFrame(
        {
            "Survived": [0] * 70 + [1] * 30,
            "x": range(100),
        }
    )
    train, val, test = split_ready_frame(
        ready,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=0,
        stratify_column="Survived",
    )
    for part in (train, val, test):
        rate = float(part["Survived"].mean())
        assert 0.2 <= rate <= 0.4


def test_rejects_bad_ratios() -> None:
    ready = pd.DataFrame({"Survived": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
    with pytest.raises(ValueError):
        split_ready_frame(
            ready,
            train_ratio=0.5,
            val_ratio=0.5,
            test_ratio=0.5,
            random_state=1,
            stratify_column=None,
        )
