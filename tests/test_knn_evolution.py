from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.knn import (  # noqa: E402
    BEAT_APPEAR,
    BEAT_LINES,
    BEAT_RANK,
    BEAT_VOTE,
    DEMO_QUERY_COUNT,
    RANK_TABLE_LIMIT,
    advance_evolution,
    build_knn_artifact,
    can_advance,
    can_click,
    click_query,
    distance_marker_styles,
    fit_knn_classifier,
    neighbor_indices_for_view,
    new_evolution,
    prepare_feature_matrix,
    rank_table_rows,
    shows_boundary,
)


def _artifact(*, k: int = 3):
    df = pd.DataFrame(
        {
            "特徵1": [0.0, 0.0, 10.0, 10.0, 0.2, 9.8],
            "特徵2": [0.0, 1.0, 0.0, 1.0, 0.1, 0.9],
            "類別": [0, 0, 1, 1, 0, 1],
        }
    )
    features = ["特徵1", "特徵2"]
    matrix, scaler = prepare_feature_matrix(df, features, standardize=True)
    clf = fit_knn_classifier(matrix, df["類別"], k=k)
    return build_knn_artifact(
        clf,
        features=features,
        target="類別",
        k=k,
        standardize=True,
        scaler=scaler,
        data_source="tiny",
        feature_frame=matrix,
        target_series=df["類別"],
    )


def _demos() -> tuple[tuple[float, float], ...]:
    return ((0.15, 0.15), (9.7, 0.8), (5.0, 0.5))


def test_start_has_no_query_or_boundary() -> None:
    evo = new_evolution(_demos(), locked_k=3)
    assert evo.active_xy is None
    assert evo.beat is None
    assert evo.labeled == ()
    assert can_advance(evo) is True
    assert can_click(evo) is False
    assert shows_boundary(evo) is False


def test_first_advance_is_appear_beat() -> None:
    evo = advance_evolution(new_evolution(_demos(), locked_k=3), _artifact())
    assert evo.beat == BEAT_APPEAR
    assert evo.active_xy == (0.15, 0.15)
    assert evo.demo_remaining == ((9.7, 0.8), (5.0, 0.5))
    assert evo.distances_by_index is None
    assert neighbor_indices_for_view(evo) is None
    assert shows_boundary(evo) is False
    assert can_click(evo) is False


def test_four_beats_then_label_one_point() -> None:
    artifact = _artifact(k=3)
    evo = new_evolution(_demos(), locked_k=3)
    for _ in range(4):
        evo = advance_evolution(evo, artifact)
    assert evo.beat == BEAT_VOTE
    assert len(evo.labeled) == 1
    assert evo.labeled[0][0] == 0.15
    assert evo.labeled[0][1] == 0.15
    assert evo.labeled[0][2] in (0, 1)
    assert neighbor_indices_for_view(evo) is not None
    assert len(neighbor_indices_for_view(evo)) == 3
    assert shows_boundary(evo) is False


def test_rank_beat_has_table_and_no_lines() -> None:
    artifact = _artifact(k=3)
    evo = new_evolution(_demos(), locked_k=3)
    evo = advance_evolution(evo, artifact)
    evo = advance_evolution(evo, artifact)
    assert evo.beat == BEAT_RANK
    assert evo.distances_by_index is not None
    assert neighbor_indices_for_view(evo) is None
    rows = rank_table_rows(evo, artifact.train_y)
    assert 1 <= len(rows) <= RANK_TABLE_LIMIT
    assert rows[0].rank == 1
    dists = [row.distance for row in rows]
    assert dists == sorted(dists)
    near = np.asarray(evo.order_near_to_far)[:3]
    assert list(near) == [row.index for row in rows[:3]]


def test_lines_beat_exposes_k_neighbors() -> None:
    artifact = _artifact(k=3)
    evo = new_evolution(_demos(), locked_k=3)
    for _ in range(3):
        evo = advance_evolution(evo, artifact)
    assert evo.beat == BEAT_LINES
    idx = neighbor_indices_for_view(evo)
    assert idx == list(evo.order_near_to_far[:3])


def test_three_demo_points_unlock_boundary_and_click() -> None:
    artifact = _artifact(k=3)
    evo = new_evolution(_demos(), locked_k=3)
    for _ in range(DEMO_QUERY_COUNT * 4):
        evo = advance_evolution(evo, artifact)
    assert len(evo.labeled) == 3
    assert evo.demo_remaining == ()
    assert evo.beat == BEAT_VOTE
    assert shows_boundary(evo) is True
    assert can_click(evo) is True
    assert can_advance(evo) is False
    evo2 = advance_evolution(evo, artifact)
    assert evo2.beat == BEAT_VOTE
    assert len(evo2.labeled) == 3


def test_click_ignored_until_three_demos_done() -> None:
    artifact = _artifact(k=3)
    evo = advance_evolution(new_evolution(_demos(), locked_k=3), artifact)
    assert evo.beat == BEAT_APPEAR
    ignored = click_query(evo, (1.0, 1.0))
    assert ignored.active_xy == evo.active_xy
    assert ignored.beat == BEAT_APPEAR


def test_click_after_demos_starts_appear_and_keeps_boundary() -> None:
    artifact = _artifact(k=3)
    evo = new_evolution(_demos(), locked_k=3)
    for _ in range(DEMO_QUERY_COUNT * 4):
        evo = advance_evolution(evo, artifact)
    evo = click_query(evo, (2.0, 2.0))
    assert evo.beat == BEAT_APPEAR
    assert evo.active_xy == (2.0, 2.0)
    assert len(evo.labeled) == 3
    assert shows_boundary(evo) is True
    assert can_click(evo) is False
    evo = advance_evolution(evo, artifact)
    assert evo.beat == BEAT_RANK
    ignored = click_query(evo, (3.0, 3.0))
    assert ignored.active_xy == (2.0, 2.0)
    assert ignored.beat == BEAT_RANK


def test_vote_to_next_demo_clears_neighbor_lines() -> None:
    artifact = _artifact(k=3)
    evo = new_evolution(_demos(), locked_k=3)
    for _ in range(4):
        evo = advance_evolution(evo, artifact)
    assert neighbor_indices_for_view(evo) is not None
    evo = advance_evolution(evo, artifact)
    assert evo.beat == BEAT_APPEAR
    assert neighbor_indices_for_view(evo) is None
    assert len(evo.labeled) == 1


def test_click_add_on_accumulates_fourth_point() -> None:
    artifact = _artifact(k=3)
    evo = new_evolution(_demos(), locked_k=3)
    for _ in range(DEMO_QUERY_COUNT * 4):
        evo = advance_evolution(evo, artifact)
    evo = click_query(evo, (2.0, 2.0))
    for _ in range(4):
        evo = advance_evolution(evo, artifact)
    assert len(evo.labeled) == 4
    assert evo.labeled[-1][0] == 2.0
    assert shows_boundary(evo) is True
    assert can_click(evo) is True


def test_accepted_chart_click_ignores_when_disabled_or_same_point() -> None:
    from dataset_streamlit_shell.ml.knn import accepted_chart_click

    assert accepted_chart_click(click_enabled=False, selected_xy=(1.0, 1.0), active_xy=None) is None
    assert accepted_chart_click(click_enabled=True, selected_xy=None, active_xy=None) is None
    assert accepted_chart_click(
        click_enabled=True, selected_xy=(1.0, 1.0), active_xy=(1.0, 1.0)
    ) is None
    assert accepted_chart_click(
        click_enabled=True, selected_xy=(2.0, 3.0), active_xy=(1.0, 1.0)
    ) == (2.0, 3.0)


def test_status_caption_after_start_prompts_next() -> None:
    from dataset_streamlit_shell.ml.knn import evolution_status_caption

    evo = new_evolution(_demos(), locked_k=3)
    assert "下一步" in evolution_status_caption(evo)


def test_nearer_points_are_larger_and_more_opaque() -> None:
    sizes, opacities = distance_marker_styles(np.array([0.1, 5.0, 10.0]))
    assert sizes[0] > sizes[1] > sizes[2]
    assert opacities[0] > opacities[1] > opacities[2]
