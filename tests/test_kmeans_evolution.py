from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.clustering import (  # noqa: E402
    BEAT_ASSIGN,
    BEAT_INIT,
    BEAT_UPDATE,
    KMEANS_INIT_RANDOM_STATE,
    advance_kmeans_evolution,
    can_advance_kmeans,
    kmeans_evolution_status,
    new_kmeans_evolution,
    sample_initial_centers,
)


def _square() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )


def _evo(centers: np.ndarray, *, max_rounds: int = 15):
    return new_kmeans_evolution(
        locked_k=len(centers),
        initial_centers=centers,
        max_rounds=max_rounds,
    )


def test_start_has_no_centers_or_labels() -> None:
    evo = _evo(np.array([[0.5, 0.5], [10.0, 10.0]]))
    assert evo.beat is None
    assert evo.centers is None
    assert evo.labels is None
    assert evo.sse is None
    assert can_advance_kmeans(evo) is True
    status = kmeans_evolution_status(evo)
    assert "下一步" in status
    assert "平方和" not in status


def test_first_advance_shows_initial_centers_unassigned() -> None:
    centers = np.array([[0.5, 0.5], [10.0, 10.0]])
    evo = advance_kmeans_evolution(_evo(centers), _square())
    assert evo.beat == BEAT_INIT
    assert evo.labels is None
    assert evo.sse is None
    assert evo.previous_centers is None
    np.testing.assert_allclose(np.asarray(evo.centers), centers)
    status = kmeans_evolution_status(evo)
    assert "尚未分群" in status or "初始" in status
    assert "平方和" not in status


def test_assign_then_update_clears_ghost_on_next_assign() -> None:
    x = _square()
    centers = np.array([[0.0, 0.0], [1.0, 1.0]])
    evo = advance_kmeans_evolution(_evo(centers), x)
    evo = advance_kmeans_evolution(evo, x)
    assert evo.beat == BEAT_ASSIGN
    assert evo.labels is not None
    assert len(evo.labels) == 4
    assert evo.sse is not None
    assert evo.previous_centers is None
    assign_sse = evo.sse
    status = kmeans_evolution_status(evo)
    assert "①" in status
    assert "平方和" in status

    evo = advance_kmeans_evolution(evo, x)
    assert evo.beat == BEAT_UPDATE
    assert evo.previous_centers is not None
    assert evo.sse is not None
    assert evo.sse <= assign_sse + 1e-9
    assert "②" in kmeans_evolution_status(evo)

    evo = advance_kmeans_evolution(evo, x)
    assert evo.beat == BEAT_ASSIGN
    assert evo.previous_centers is None


def test_empty_cluster_center_stays_put() -> None:
    x = _square()
    far = np.array([[0.5, 0.5], [100.0, 100.0]])
    evo = advance_kmeans_evolution(_evo(far), x)
    evo = advance_kmeans_evolution(evo, x)
    assert evo.empty_cluster is True
    assert "有一群沒有點" in kmeans_evolution_status(evo)
    evo = advance_kmeans_evolution(evo, x)
    np.testing.assert_allclose(np.asarray(evo.centers)[1], [100.0, 100.0])


def test_stable_assignment_stops_next() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 0.0],
            [10.0, 1.0],
        ]
    )
    centers = np.array([[0.0, 0.5], [10.0, 0.5]])
    evo = _evo(centers)
    for _ in range(8):
        if not can_advance_kmeans(evo):
            break
        evo = advance_kmeans_evolution(evo, x)
    assert evo.converged is True
    assert can_advance_kmeans(evo) is False
    assert "已收斂" in kmeans_evolution_status(evo)
    stuck = advance_kmeans_evolution(evo, x)
    assert stuck.beat == evo.beat
    assert stuck.labels == evo.labels


def test_round_cap_stops_after_max_updates() -> None:
    x = _square()
    centers = np.array([[0.2, 0.1], [0.9, 0.8]])
    evo = _evo(centers, max_rounds=1)
    evo = advance_kmeans_evolution(evo, x)
    evo = advance_kmeans_evolution(evo, x)
    evo = advance_kmeans_evolution(evo, x)
    assert evo.beat == BEAT_UPDATE
    assert evo.hit_round_cap is True
    assert can_advance_kmeans(evo) is False
    assert "已達輪數上限" in kmeans_evolution_status(evo)


def test_sample_initial_centers_is_deterministic() -> None:
    x = _square()
    a = sample_initial_centers(x, n_clusters=2, random_state=KMEANS_INIT_RANDOM_STATE)
    b = sample_initial_centers(x, n_clusters=2, random_state=KMEANS_INIT_RANDOM_STATE)
    np.testing.assert_allclose(a, b)
    assert a.shape == (2, 2)
    for row in a:
        assert np.any(np.all(np.isclose(x, row), axis=1))
