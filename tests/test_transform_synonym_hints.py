from __future__ import annotations

import sys
from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.integration import (  # noqa: E402
    SYNONYM_HINT_NONE,
    SYNONYM_HINT_YES,
    flagged_synonym_columns,
    hinted_nunique,
    load_titanic_integration_frames,
    synonym_hint_label,
    text_object_columns,
    transform_column_overview,
)


def test_sex_and_embarked_get_synonym_hint_from_builtin_mess() -> None:
    passengers, voyage = load_titanic_integration_frames()
    assert synonym_hint_label(passengers["Sex"], "Sex") == SYNONYM_HINT_YES
    assert hinted_nunique(passengers["Sex"], "Sex") == 2
    assert hinted_nunique(passengers["Sex"], "Sex") < int(
        passengers["Sex"].nunique(dropna=True)
    )

    assert synonym_hint_label(voyage["Embarked"], "Embarked") == SYNONYM_HINT_YES
    assert hinted_nunique(voyage["Embarked"], "Embarked") == 3
    assert hinted_nunique(voyage["Embarked"], "Embarked") < int(
        voyage["Embarked"].nunique(dropna=True)
    )


def test_sex_hint_survives_case_and_whitespace_variants() -> None:
    import pandas as pd

    series = pd.Series([" MAN ", "Woman", "male", "FEMALE"])
    assert hinted_nunique(series, "Sex") == 2
    assert synonym_hint_label(series, "Sex") == SYNONYM_HINT_YES


def test_name_has_no_synonym_hint() -> None:
    passengers, _ = load_titanic_integration_frames()
    assert synonym_hint_label(passengers["Name"], "Name") == SYNONYM_HINT_NONE
    assert hinted_nunique(passengers["Name"], "Name") == int(
        passengers["Name"].nunique(dropna=True)
    )


def test_transform_column_overview_lists_all_text_columns_with_hint_fields() -> None:
    passengers, voyage = load_titanic_integration_frames()
    voyage = voyage.rename(columns={"passenger_id": "PassengerId"})
    merged = passengers.merge(voyage, on="PassengerId", how="left")
    columns = text_object_columns(merged)
    assert "Sex" in columns
    assert "Name" in columns
    assert "PassengerId" not in columns

    overview = transform_column_overview(merged, columns)
    assert list(overview.columns) == [
        "資料型態",
        "空值筆數",
        "不同值數量",
        "提示後不同值數量",
        "同義提示",
        "常見值",
    ]
    assert list(overview.index) == columns
    assert overview.loc["Sex", "同義提示"] == SYNONYM_HINT_YES
    assert overview.loc["Name", "同義提示"] == SYNONYM_HINT_NONE
    assert overview.loc["Sex", "提示後不同值數量"] == 2
    assert overview.loc["Embarked", "同義提示"] == SYNONYM_HINT_YES


def test_flagged_columns_are_those_with_synonym_hint() -> None:
    passengers, voyage = load_titanic_integration_frames()
    voyage = voyage.rename(columns={"passenger_id": "PassengerId"})
    merged = passengers.merge(voyage, on="PassengerId", how="left")
    flagged = flagged_synonym_columns(merged)
    assert "Sex" in flagged
    assert "Embarked" in flagged
    assert "Name" not in flagged
