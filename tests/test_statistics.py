from __future__ import annotations

import numpy as np

from routing_aware_atos.evaluation.statistics import bootstrap_mean_ci, paired_metric_deltas


def test_bootstrap_mean_ci_contains_sample_mean():
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    stats = bootstrap_mean_ci(values, n_bootstrap=200, random_seed=3)

    assert np.isclose(stats["mean"], 0.25)
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
    assert stats["n"] == 4.0


def test_paired_metric_deltas_match_rank_when_available():
    rows = [
        {"policy_name": "same_token", "rank": 2, "mean_r2": 0.4},
        {"policy_name": "attention_topk", "rank": 2, "mean_r2": 0.6},
        {"policy_name": "attention_topk", "rank": 4, "mean_r2": 0.7},
    ]

    deltas = paired_metric_deltas(rows, baseline_policy="same_token", metric_keys=["mean_r2"])

    by_policy_rank = {(row["policy_name"], row["rank"]): row for row in deltas}
    assert np.isclose(by_policy_rank[("attention_topk", 2)]["mean_r2_delta"], 0.2)
    assert np.isclose(by_policy_rank[("attention_topk", 4)]["mean_r2_delta"], 0.3)
