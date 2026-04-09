from __future__ import annotations

from routing_aware_atos.data.attribution_cache import AttributionBuildConfig, attach_attribution_scores
from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.evaluation.route_agreement import compare_routing_policies, summarize_route_metadata
from routing_aware_atos.routing.attribution import AttributionTopKPolicy
from routing_aware_atos.routing.attention import AttentionTop1Policy, AttentionTopKPolicy
from routing_aware_atos.routing.base import RoutingPolicyConfig


def test_compare_routing_policies_outputs_summary():
    samples = make_mock_samples(num_samples=2, seq_len=5, d_model=4)
    samples = attach_attribution_scores(
        samples,
        [(10, 12)],
        config=AttributionBuildConfig(methods=["attention_value", "residual_similarity"]),
    )
    policies = [
        ("attention_top1", AttentionTop1Policy(RoutingPolicyConfig(top_k=1))),
        ("attention_topk", AttentionTopKPolicy(RoutingPolicyConfig(top_k=3))),
        ("attribution_topk", AttributionTopKPolicy(RoutingPolicyConfig(top_k=3))),
    ]
    payload = compare_routing_policies(samples, policies, source_layer=10, target_layer=12)
    assert "rows" in payload
    assert "summary" in payload
    assert "attention_top1__vs__attention_topk" in payload["summary"]

    meta = summarize_route_metadata(payload)
    assert meta["num_rows"] > 0
    assert meta["mean_route_length"] >= 1.0
