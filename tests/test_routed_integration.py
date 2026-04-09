import numpy as np

from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig
from routing_aware_atos.routed_dataset import build_same_token_pairs
from routing_aware_atos.routing_policies import SameTokenPolicy, build_routing_policy


def test_build_same_token_pairs_returns_route_metadata():
    samples = make_mock_samples(num_samples=1, seq_len=4, d_model=2)
    pairs = build_same_token_pairs(samples, source_layer=10, target_layer=12)

    assert pairs.X.shape == (4, 2)
    assert pairs.Y.shape == (4, 2)
    assert len(pairs.routes) == 4
    assert pairs.routes[0]["score_type"] == "same_token"
    assert pairs.routes[0]["source_weights"] == [1.0]


def test_routing_policies_module_exposes_builder_and_policy_types():
    policy = build_routing_policy("same_token")
    assert isinstance(policy, SameTokenPolicy)


def test_transport_operator_fit_xy_matches_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 4)).astype(np.float32)
    W = rng.normal(size=(4, 3)).astype(np.float32)
    Y = X @ W

    op = TransportOperator(TransportOperatorConfig(ridge_lambda=1e-6)).fit_xy(X, Y)
    metrics = op.evaluate_xy(X, Y)

    assert metrics["r2"] > 0.999
