import numpy as np

from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.data.routed_dataset import RoutedActivationDataset
from routing_aware_atos.routing.factory import build_routing_policy


def test_routed_dataset_builds_expected_number_of_rows():
    samples = make_mock_samples(num_samples=2, seq_len=5, d_model=3)
    policy = build_routing_policy("attention_topk", top_k=2)
    ds = RoutedActivationDataset(samples=samples, source_layer=10, target_layer=12, routing_policy=policy)
    pairs = ds.build_pairs()
    assert pairs.X.shape == (10, 3)
    assert pairs.Y.shape == (10, 3)
    assert len(pairs.routes) == 10


def test_same_token_routed_dataset_matches_source_position():
    samples = make_mock_samples(num_samples=1, seq_len=4, d_model=2)
    policy = build_routing_policy("same_token")
    ds = RoutedActivationDataset(samples=samples, source_layer=10, target_layer=12, routing_policy=policy)
    pairs = ds.build_pairs()
    src = samples[0].residuals[10]
    assert np.allclose(pairs.X, src)
