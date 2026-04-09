import numpy as np

from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.routing.factory import build_routing_policy


def test_same_token_policy_returns_identity_position():
    sample = make_mock_samples(num_samples=1)[0]
    policy = build_routing_policy("same_token")
    route = policy.select_sources(sample=sample, target_pos=2, source_layer=10, target_layer=12)
    assert route.source_ids == [2]
    assert route.source_weights == [1.0]


def test_attention_topk_policy_returns_normalized_weights():
    sample = make_mock_samples(num_samples=1)[0]
    policy = build_routing_policy("attention_topk", top_k=3)
    route = policy.select_sources(sample=sample, target_pos=1, source_layer=10, target_layer=12)
    assert len(route.source_ids) == 3
    assert np.isclose(sum(route.source_weights), 1.0)


def test_attention_topk_policy_exclude_self_drops_target_position_even_when_topk_is_large():
    sample = make_mock_samples(num_samples=1, seq_len=4)[0]
    policy = build_routing_policy("attention_topk", top_k=4, exclude_self=True)
    route = policy.select_sources(sample=sample, target_pos=2, source_layer=10, target_layer=12)
    assert len(route.source_ids) == 3
    assert set(route.source_ids) == {0, 1, 3}
    assert 2 not in route.source_ids
    assert np.isclose(sum(route.source_weights), 1.0)


def test_attention_topk_policy_rejects_non_positive_topk():
    sample = make_mock_samples(num_samples=1)[0]
    policy = build_routing_policy("attention_topk", top_k=0)
    with pytest.raises(ValueError):
        policy.select_sources(sample=sample, target_pos=1, source_layer=10, target_layer=12)


def test_attribution_topk_policy_uses_attribution_scores():
    sample = make_mock_samples(num_samples=1)[0]
    policy = build_routing_policy("attribution_topk", top_k=2)
    route = policy.select_sources(sample=sample, target_pos=0, source_layer=10, target_layer=12)
    assert len(route.source_ids) == 2
    assert route.score_type == "attribution_topk"
