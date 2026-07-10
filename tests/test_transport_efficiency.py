from __future__ import annotations

import numpy as np

from routing_aware_atos.evaluation.transport_efficiency import (
    canonical_correlations,
    compute_transport_efficiency,
    effective_transport_dimensionality,
    transport_r2_ceiling,
)
from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig


def test_canonical_correlations_are_high_for_aligned_arrays():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    Y = X.copy()

    rhos = canonical_correlations(X, Y)

    assert rhos.shape == (4,)
    assert np.all(rhos > 0.99)


def test_transport_ceiling_uses_rank_over_target_dimension():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    Y = X.copy()

    ceiling, rhos = transport_r2_ceiling(X, Y, rank=2)

    assert np.all(rhos > 0.99)
    assert 0.49 < ceiling < 0.51
    assert effective_transport_dimensionality(rhos**2) > 3.9


def test_rank_truncated_operator_has_high_transport_efficiency():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(500, 4)).astype(np.float32)
    Y = X.copy()
    operator = TransportOperator(
        TransportOperatorConfig(ridge_lambda=1e-8, rank=2)
    ).fit(X, Y)

    metrics = compute_transport_efficiency(X, Y, operator.predict(X), rank=2)

    assert metrics.ceiling_r2 > 0.49
    assert metrics.whitened_r2 > 0.45
    assert 0.85 < metrics.efficiency <= 1.0
    assert metrics.raw_efficiency <= 1.05


def test_transport_efficiency_accepts_string_rank_and_eps():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    Y = X.copy()

    metrics = compute_transport_efficiency(X, Y, Y, rank="2", eps="1.0e-8")

    assert metrics.rank == 2
    assert metrics.ceiling_r2 > 0.66


def test_transport_efficiency_rejects_negative_eps():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(20, 2)).astype(np.float32)

    try:
        canonical_correlations(X, X, eps=-1e-8)
    except ValueError as exc:
        assert "eps must be non-negative" in str(exc)
    else:
        raise AssertionError("negative eps should fail")
