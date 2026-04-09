from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from routing_aware_atos.evaluation.feature_eval import (
    evaluate_operator_from_cached_samples,
    evaluate_operator_in_feature_space,
)
from routing_aware_atos.utils.io import save_json


def compare_policy_runs(
    runs: Iterable[Dict[str, Any]],
    *,
    feature_ids: Iterable[int] | None = None,
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
        rows.append({
            'policy_name': item['policy_name'],
            'rank': item.get('rank'),
            'mean_r2': item['feature_summary']['mean_r2'],
            'median_r2': item['feature_summary']['median_r2'],
            'mean_corr': item['feature_summary']['mean_corr'],
            'residual_r2': item['residual_metrics']['r2'],
        })

    rows.sort(key=lambda x: (x['policy_name'], -1 if x['rank'] is None else x['rank']))
    return {'runs': results, 'summary_rows': rows}


def save_policy_comparison(payload: Dict[str, Any], output_path: str | Path) -> None:
    save_json(output_path, payload)
