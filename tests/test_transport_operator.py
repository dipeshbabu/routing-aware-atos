
import numpy as np

from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig


def test_transport_operator_fits_linear_map_well():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    W = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    b = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    Y = X @ W + b

    op = TransportOperator(TransportOperatorConfig(ridge_lambda=1e-6)).fit(X, Y)
    metrics = op.evaluate(X, Y)
    assert metrics["r2"] > 0.999
    assert metrics["mse"] < 1e-6


def test_transport_operator_rank_truncation_reduces_effective_rank():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    W = rng.normal(size=(5, 5)).astype(np.float32)
    Y = X @ W

    op = TransportOperator(TransportOperatorConfig(ridge_lambda=1e-6, rank=2)).fit(X, Y)
    assert np.linalg.matrix_rank(op.weight) <= 2
