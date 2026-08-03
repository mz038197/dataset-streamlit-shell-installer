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

from dataset_streamlit_shell.ml.integration import (  # noqa: E402
    EMBARKED_CANONICAL,
    LEFT_JOIN_CORRECT,
    PASSENGER_KEY,
    SEX_CANONICAL,
    VOYAGE_KEY_RAW,
    align_voyage_key,
    is_join_how_correct,
    is_key_align_correct,
    key_overlap_count,
    load_titanic_integration_frames,
    merge_passenger_voyage,
    normalize_embarked_series,
    normalize_sex_series,
)

INTEGRATION_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "integration"
)


def test_builtin_integration_csvs_exist_with_expected_columns() -> None:
    passengers, voyage = load_titanic_integration_frames()
    assert PASSENGER_KEY in passengers.columns
    assert "Sex" in passengers.columns
    assert VOYAGE_KEY_RAW in voyage.columns
    assert PASSENGER_KEY not in voyage.columns
    assert "Embarked" in voyage.columns
    assert len(passengers) > 100
    assert len(voyage) < len(passengers)


def test_sex_and_embarked_are_intentionally_messy() -> None:
    passengers, voyage = load_titanic_integration_frames()
    sex_values = {str(v) for v in passengers["Sex"].dropna().unique()}
    embarked_values = {str(v) for v in voyage["Embarked"].dropna().unique()}
    assert len(sex_values) > 2
    assert {"man", "woman"} & sex_values or {"Male", "Female"} & sex_values
    assert len(embarked_values) > 3


def test_merge_without_key_align_has_zero_overlap_on_passenger_id() -> None:
    passengers, voyage = load_titanic_integration_frames()
    assert key_overlap_count(passengers, voyage, left_on=PASSENGER_KEY, right_on=PASSENGER_KEY) == 0


def test_merge_left_keeps_all_passengers_and_exceeds_inner() -> None:
    passengers, voyage = load_titanic_integration_frames()
    left = merge_passenger_voyage(passengers, voyage, how="left", align_key=True)
    inner = merge_passenger_voyage(passengers, voyage, how="inner", align_key=True)
    assert len(left) == len(passengers)
    assert len(left) > len(inner)


def test_merge_without_align_flag_raises() -> None:
    passengers, voyage = load_titanic_integration_frames()
    with pytest.raises(ValueError, match="對齊鍵名"):
        merge_passenger_voyage(passengers, voyage, how="left", align_key=False)


def test_normalize_sex_and_embarked_collapse_to_canonical() -> None:
    passengers, voyage = load_titanic_integration_frames()
    sex_norm = normalize_sex_series(passengers["Sex"])
    emb_norm = normalize_embarked_series(voyage["Embarked"])
    assert set(sex_norm.dropna().unique()) <= set(SEX_CANONICAL.values())
    assert set(emb_norm.dropna().unique()) <= set(EMBARKED_CANONICAL.values())


def test_integration_quiz_answers() -> None:
    assert is_key_align_correct("須先把航程表的 passenger_id 改成與乘客表相同的鍵名再合併")
    assert not is_key_align_correct("兩表已經同名，直接合併即可")
    assert is_join_how_correct(LEFT_JOIN_CORRECT)
    assert not is_join_how_correct("inner")


def test_csv_files_on_disk() -> None:
    assert (INTEGRATION_DIR / "titanic_passengers.csv").is_file()
    assert (INTEGRATION_DIR / "titanic_voyage.csv").is_file()
