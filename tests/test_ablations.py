from __future__ import annotations

from routing_aware_atos.analysis.ablations import compare_against_baseline, summarize_ablation_sweep


def test_summarize_ablation_sweep():
    rows = [
        {"policy_name": "same_token", "mean_r2": 0.5},
        {"policy_name": "same_token", "mean_r2": 0.7},
        {"policy_name": "attention_topk", "mean_r2": 0.8},
    ]
    payload = summarize_ablation_sweep(rows, group_key="policy_name", metric_keys=("mean_r2",))
    assert payload["summary_rows"][0]["count"] >= 1
    names = {row["policy_name"] for row in payload["summary_rows"]}
    assert "same_token" in names


def test_compare_against_baseline():
    rows = [
        {"policy_name": "same_token", "mean_r2": 0.5, "residual_r2": 0.4},
        {"policy_name": "attention_topk", "mean_r2": 0.8, "residual_r2": 0.7},
    ]
    payload = compare_against_baseline(
        rows,
        baseline_policy="same_token",
        metric_keys=("mean_r2", "residual_r2"),
    )
    delta = [row for row in payload["delta_rows"] if row["policy_name"] == "attention_topk"][0]
    assert delta["mean_r2_delta"] > 0
    assert delta["residual_r2_delta"] > 0
