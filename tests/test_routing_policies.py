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


def test_attribution_topk_policy_uses_attribution_scores():
    sample = make_mock_samples(num_samples=1)[0]
    policy = build_routing_policy("attribution_topk", top_k=2)
    route = policy.select_sources(sample=sample, target_pos=0, source_layer=10, target_layer=12)
    assert len(route.source_ids) == 2
    assert route.score_type == "attribution_topk"


def test_previous_and_next_token_controls():
    sample = make_mock_samples(num_samples=1, seq_len=5)[0]
    previous = build_routing_policy("previous_token")
    next_policy = build_routing_policy("next_token")

    assert previous.select_sources(sample, 3, 10, 12).source_ids == [2]
    assert previous.select_sources(sample, 0, 10, 12).source_ids == [0]
    assert next_policy.select_sources(sample, 3, 10, 12).source_ids == [4]
    assert next_policy.select_sources(sample, 4, 10, 12).source_ids == [4]


def test_uniform_topk_control_has_uniform_weights():
    sample = make_mock_samples(num_samples=1, seq_len=5)[0]
    policy = build_routing_policy("uniform_topk", top_k=3, exclude_self=True)
    route = policy.select_sources(sample=sample, target_pos=2, source_layer=10, target_layer=12)

    assert len(route.source_ids) == 3
    assert 2 not in route.source_ids
    assert np.allclose(route.source_weights, np.ones(3) / 3.0)


def test_random_topk_control_is_deterministic():
    sample = make_mock_samples(num_samples=1, seq_len=6)[0]
    policy_a = build_routing_policy("random_topk", top_k=3, random_seed=123)
    policy_b = build_routing_policy("random_topk", top_k=3, random_seed=123)

    route_a = policy_a.select_sources(sample=sample, target_pos=2, source_layer=10, target_layer=12)
    route_b = policy_b.select_sources(sample=sample, target_pos=2, source_layer=10, target_layer=12)

    assert route_a.source_ids == route_b.source_ids
    assert np.allclose(route_a.source_weights, route_b.source_weights)


def test_shuffled_attention_topk_requires_attention_but_breaks_alignment():
    sample = make_mock_samples(num_samples=1, seq_len=6)[0]
    attention = build_routing_policy("attention_topk", top_k=3)
    shuffled = build_routing_policy("shuffled_attention_topk", top_k=3, random_seed=99)

    route_attention = attention.select_sources(sample=sample, target_pos=1, source_layer=10, target_layer=12)
    route_shuffled = shuffled.select_sources(sample=sample, target_pos=1, source_layer=10, target_layer=12)

    assert route_shuffled.score_type == "shuffled_attention_topk"
    assert len(route_shuffled.source_ids) == 3
    assert route_shuffled.source_ids != route_attention.source_ids


def test_causal_only_routes_never_use_future_positions():
    sample = make_mock_samples(num_samples=1, seq_len=8)[0]
    policies = [
        build_routing_policy("attention_topk", top_k=3, causal_only=True),
        build_routing_policy("attribution_topk", top_k=3, causal_only=True),
        build_routing_policy("random_topk", top_k=3, random_seed=5, causal_only=True),
        build_routing_policy(
            "shuffled_attention_topk",
            top_k=3,
            random_seed=5,
            causal_only=True,
        ),
    ]

    for policy in policies:
        route = policy.select_sources(sample, 3, 10, 12)
        assert all(source_position <= 3 for source_position in route.source_ids)


def test_causal_only_next_token_control_falls_back_to_current_position():
    sample = make_mock_samples(num_samples=1, seq_len=5)[0]
    policy = build_routing_policy("next_token", causal_only=True)
    assert policy.select_sources(sample, 2, 10, 12).source_ids == [2]
