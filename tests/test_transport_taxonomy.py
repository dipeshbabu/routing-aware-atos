from __future__ import annotations

from routing_aware_atos.evaluation.transport_taxonomy import (
    build_feature_policy_matrix,
    build_transport_taxonomy,
    classify_feature_transport,
)


def test_classify_feature_transport_priority_order():
    label = classify_feature_transport(
        {"same_token": 0.8, "attention_top1": 0.9, "attention_topk": 0.95},
        same_token_policy="same_token",
        cross_token_policies=("attention_top1",),
        multi_source_policies=("attention_topk",),
        threshold=0.7,
    )
    assert label == "same_token_transport"

    label2 = classify_feature_transport(
        {"same_token": 0.2, "attention_top1": 0.75, "attention_topk": 0.9},
        same_token_policy="same_token",
        cross_token_policies=("attention_top1",),
        multi_source_policies=("attention_topk",),
        threshold=0.7,
    )
    assert label2 == "cross_token_transport"

    label3 = classify_feature_transport(
        {"same_token": 0.2, "attention_top1": 0.4, "attention_topk": 0.72},
        same_token_policy="same_token",
        cross_token_policies=("attention_top1",),
        multi_source_policies=("attention_topk",),
        threshold=0.7,
    )
    assert label3 == "multi_source_transport"


def test_build_feature_policy_matrix():
    payloads = [
        {"policy_name": "same_token", "feature_metrics": {"0": {"r2": 0.8}, "1": {"r2": 0.2}}},
        {"policy_name": "attention_top1", "feature_metrics": {"0": {"r2": 0.5}, "1": {"r2": 0.75}}},
    ]
    mat = build_feature_policy_matrix(payloads)
    assert mat[0]["same_token"] == 0.8
    assert mat[1]["attention_top1"] == 0.75


def test_build_transport_taxonomy_counts():
    payloads = [
        {"policy_name": "same_token", "feature_metrics": {"0": {"r2": 0.8}, "1": {"r2": 0.1}, "2": {"r2": 0.1}, "3": {"r2": 0.1}}},
        {"policy_name": "attention_top1", "feature_metrics": {"0": {"r2": 0.6}, "1": {"r2": 0.8}, "2": {"r2": 0.2}, "3": {"r2": 0.2}}},
        {"policy_name": "attention_topk", "feature_metrics": {"0": {"r2": 0.6}, "1": {"r2": 0.6}, "2": {"r2": 0.85}, "3": {"r2": 0.2}}},
    ]
    taxonomy = build_transport_taxonomy(
        payloads,
        same_token_policy="same_token",
        cross_token_policies=("attention_top1",),
        multi_source_policies=("attention_topk",),
        threshold=0.7,
    )
    counts = {row["label"]: row["count"] for row in taxonomy["summary_rows"]}
    assert counts["same_token_transport"] == 1
    assert counts["cross_token_transport"] == 1
    assert counts["multi_source_transport"] == 1
    assert counts["unexplained_or_synthesized"] == 1
