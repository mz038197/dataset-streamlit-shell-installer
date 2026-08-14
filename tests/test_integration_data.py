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
    clear_dual_table_copies,
    commit_dual_table_merge,
    dual_table_copy_paths,
    ensure_dual_table_copies,
    is_join_how_correct,
    is_key_align_correct,
    key_overlap_count,
    load_dual_tables,
    load_titanic_integration_frames,
    merge_from_dual_tables,
    merge_passenger_voyage,
    normalize_embarked_series,
    normalize_sex_series,
    voyage_key_is_aligned,
    write_dual_table_copies,
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


def test_load_dual_tables_falls_back_to_builtin(tmp_path: Path) -> None:
    passengers, voyage = load_dual_tables(tmp_path)
    builtin_p, builtin_v = load_titanic_integration_frames()
    assert list(passengers.columns) == list(builtin_p.columns)
    assert list(voyage.columns) == list(builtin_v.columns)
    assert VOYAGE_KEY_RAW in voyage.columns
    assert not voyage_key_is_aligned(voyage)


def test_load_dual_tables_prefers_workspace_copies(tmp_path: Path) -> None:
    passengers, voyage = load_titanic_integration_frames()
    aligned = align_voyage_key(voyage)
    write_dual_table_copies(tmp_path, passengers, aligned)
    loaded_p, loaded_v = load_dual_tables(tmp_path)
    assert PASSENGER_KEY in loaded_v.columns
    assert VOYAGE_KEY_RAW not in loaded_v.columns
    assert voyage_key_is_aligned(loaded_v)
    copy_p, copy_v = dual_table_copy_paths(tmp_path)
    assert copy_p.is_file()
    assert copy_v.is_file()


def test_clear_dual_table_copies_restores_builtin_key(tmp_path: Path) -> None:
    passengers, voyage = load_titanic_integration_frames()
    write_dual_table_copies(tmp_path, passengers, align_voyage_key(voyage))
    clear_dual_table_copies(tmp_path)
    _, loaded_v = load_dual_tables(tmp_path)
    assert VOYAGE_KEY_RAW in loaded_v.columns
    assert not dual_table_copy_paths(tmp_path)[1].is_file()


def test_merge_from_dual_tables_rejects_unaligned_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="未對齊鍵名"):
        merge_from_dual_tables(tmp_path, how="left")


def test_merge_from_aligned_copies_supports_left_and_inner(tmp_path: Path) -> None:
    passengers, voyage = load_titanic_integration_frames()
    write_dual_table_copies(tmp_path, passengers, align_voyage_key(voyage))
    left = merge_from_dual_tables(tmp_path, how="left")
    inner = merge_from_dual_tables(tmp_path, how="inner")
    assert len(left) == len(passengers)
    assert len(inner) < len(left)


def test_commit_dual_table_merge_writes_both_files_and_deletes_copies(
    tmp_path: Path,
) -> None:
    passengers, voyage = load_titanic_integration_frames()
    write_dual_table_copies(tmp_path, passengers, align_voyage_key(voyage))
    original = tmp_path / "original.csv"
    working = tmp_path / "working.csv"
    commit_dual_table_merge(
        tmp_path,
        how="left",
        original_path=original,
        working_path=working,
    )
    assert original.is_file()
    assert working.is_file()
    assert len(pd.read_csv(original)) == len(passengers)
    assert not dual_table_copy_paths(tmp_path)[1].is_file()


def test_ensure_dual_table_copies_seeds_both_from_builtin(tmp_path: Path) -> None:
    ensure_dual_table_copies(tmp_path)
    passengers_path, voyage_path = dual_table_copy_paths(tmp_path)
    assert passengers_path.is_file()
    assert voyage_path.is_file()
    _, voyage = load_dual_tables(tmp_path)
    assert VOYAGE_KEY_RAW in voyage.columns
    assert not voyage_key_is_aligned(voyage)


def test_ensure_dual_table_copies_does_not_overwrite_aligned_voyage(
    tmp_path: Path,
) -> None:
    passengers, voyage = load_titanic_integration_frames()
    write_dual_table_copies(tmp_path, passengers, align_voyage_key(voyage))
    ensure_dual_table_copies(tmp_path)
    _, loaded_v = load_dual_tables(tmp_path)
    assert voyage_key_is_aligned(loaded_v)


def test_ensure_dual_table_copies_fills_missing_passengers_only(
    tmp_path: Path,
) -> None:
    passengers, voyage = load_titanic_integration_frames()
    _, voyage_path = dual_table_copy_paths(tmp_path)
    voyage_path.parent.mkdir(parents=True, exist_ok=True)
    align_voyage_key(voyage).to_csv(voyage_path, index=False)
    ensure_dual_table_copies(tmp_path)
    copy_p, copy_v = dual_table_copy_paths(tmp_path)
    assert copy_p.is_file()
    assert PASSENGER_KEY in pd.read_csv(copy_p).columns
    assert voyage_key_is_aligned(pd.read_csv(copy_v))
