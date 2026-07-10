import numpy as np
import pytest

from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.data.routed_dataset import ConcatenatedRoutedActivationDataset, RoutedActivationDataset
from routing_aware_atos.routed_types import RoutedPairs
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


def test_concatenated_routed_dataset_preserves_topk_slots():
    samples = make_mock_samples(num_samples=1, seq_len=4, d_model=3)
    policy = build_routing_policy("attention_topk", top_k=2)
    ds = ConcatenatedRoutedActivationDataset(
        samples=samples,
        source_layer=10,
        target_layer=12,
        routing_policy=policy,
        max_sources=2,
    )

    pairs = ds.build_pairs()

    assert pairs.X.shape == (4, 6)
    assert pairs.Y.shape == (4, 3)
    assert pairs.routes[0]["input_mode"] == "concat"
    assert pairs.routes[0]["max_sources"] == 2
    assert len(pairs.routes[0]["used_source_ids"]) == 2


def test_routed_pairs_rejects_zero_width_arrays():
    with pytest.raises(ValueError, match="non-empty feature dimensions"):
        RoutedPairs(
            X=np.zeros((2, 0), dtype=np.float32),
            Y=np.zeros((2, 3), dtype=np.float32),
            routes=[{}, {}],
        ).validate()
