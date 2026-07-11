
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


def test_transport_operator_supports_rectangular_input_output_dims():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 8)).astype(np.float32)
    W = rng.normal(size=(8, 3)).astype(np.float32)
    Y = X @ W

    op = TransportOperator(TransportOperatorConfig(ridge_lambda=1e-6)).fit(X, Y)
    pred = op.predict(X)

    assert pred.shape == Y.shape
    assert op.evaluate(X, Y)["r2"] > 0.999


def test_transport_operator_torch_backend_matches_linear_map_on_cpu():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(96, 6)).astype(np.float32)
    W = rng.normal(size=(6, 4)).astype(np.float32)
    Y = X @ W + 0.2

    op = TransportOperator(
        TransportOperatorConfig(
            ridge_lambda=1e-5,
            compute_backend="torch",
            device="cpu",
        )
    ).fit(X, Y)

    assert op.evaluate(X, Y)["r2"] > 0.999


def test_residual_r2_is_uniform_average_across_output_dimensions():
    operator = TransportOperator()
    operator.weight = np.eye(2, dtype=np.float32)
    operator.bias = np.asarray([0.0, 1.0], dtype=np.float32)
    X = np.asarray([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]], dtype=np.float32)

    metrics = operator.evaluate(X, X)

    expected_second_r2 = 1.0 - 3.0 / 200.0
    assert np.isclose(metrics["r2"], (1.0 + expected_second_r2) / 2.0)
