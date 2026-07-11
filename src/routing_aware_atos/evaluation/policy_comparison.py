from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from routing_aware_atos.evaluation.feature_eval import (
    evaluate_operator_from_cached_samples,
    evaluate_operator_in_feature_space,
)
from routing_aware_atos.evaluation.statistics import add_bootstrap_ci, paired_metric_deltas
from routing_aware_atos.utils.io import save_json


def compare_policy_runs(
    runs: Iterable[Dict[str, Any]],
    *,
    feature_ids: Iterable[int] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 0,
    baseline_policy: str | None = None,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for run in runs:
        if run.get("pairs_path"):
            payload = evaluate_operator_in_feature_space(
                operator_path=run['operator_path'],
                pairs_path=run['pairs_path'],
                decoder_path=run['decoder_path'],
                feature_ids=feature_ids,
                split_name=run.get('split_name', 'eval'),
            )
        else:
            routing_cfg = run.get("routing", {})
            payload = evaluate_operator_from_cached_samples(
                operator_path=run["operator_path"],
                decoder_path=run["decoder_path"],
                activation_dir_path=run.get("activation_dir_path"),
                cache_path=run.get("cache_path"),
                source_layer=run["source_layer"],
                target_layer=run["target_layer"],
                routing_policy=routing_cfg.get("policy", "same_token"),
                top_k=routing_cfg.get("top_k", 1),
                normalize_weights=routing_cfg.get("normalize_weights", True),
                exclude_self=routing_cfg.get("exclude_self", False),
                allow_negative_scores=routing_cfg.get("allow_negative_scores", False),
                random_seed=routing_cfg.get("random_seed", 0),
                causal_only=routing_cfg.get("causal_only", False),
                input_mode=routing_cfg.get("input_mode", "weighted_sum"),
                max_sources=routing_cfg.get("max_sources"),
                include_positions=run.get("include_positions"),
                feature_ids=feature_ids,
                split_name=run.get("split_name", "eval"),
                num_samples=run.get("num_samples", 2),
                seq_len=run.get("seq_len", 6),
                d_model=run.get("d_model", 4),
            )
        payload['policy_name'] = run['policy_name']
        payload['rank'] = run.get('rank')
        results.append(payload)

    rows = []
    for item in results:
        row = {
            'policy_name': item['policy_name'],
            'rank': item.get('rank'),
            'mean_r2': item['feature_summary']['mean_r2'],
            'median_r2': item['feature_summary']['median_r2'],
            'mean_corr': item['feature_summary']['mean_corr'],
            'residual_r2': item['residual_metrics']['r2'],
        }
        feature_metrics = item.get("feature_metrics", {})
        if "r2" in feature_metrics:
            add_bootstrap_ci(
                row,
                metric_name="mean_r2",
                values=feature_metrics["r2"],
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                random_seed=random_seed,
            )
        if "corr" in feature_metrics:
            add_bootstrap_ci(
                row,
                metric_name="mean_corr",
                values=feature_metrics["corr"],
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                random_seed=random_seed,
            )
        rows.append(row)

    rows.sort(key=lambda x: (x['policy_name'], -1 if x['rank'] is None else x['rank']))
    payload: Dict[str, Any] = {'runs': results, 'summary_rows': rows}
    if baseline_policy is not None:
        payload["delta_rows"] = paired_metric_deltas(
            rows,
            baseline_policy=baseline_policy,
            metric_keys=("mean_r2", "mean_corr", "residual_r2"),
        )
    return payload


def save_policy_comparison(payload: Dict[str, Any], output_path: str | Path) -> None:
    save_json(output_path, payload)
