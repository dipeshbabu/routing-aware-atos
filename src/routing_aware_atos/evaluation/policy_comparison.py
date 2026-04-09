from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from routing_aware_atos.evaluation.feature_eval import evaluate_operator_in_feature_space
from routing_aware_atos.utils.io import save_json


def compare_policy_runs(
    runs: Iterable[Dict[str, Any]],
    *,
    feature_ids: Iterable[int] | None = None,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for run in runs:
        payload = evaluate_operator_in_feature_space(
            operator_path=run['operator_path'],
            pairs_path=run['pairs_path'],
            decoder_path=run['decoder_path'],
            feature_ids=feature_ids,
            split_name=run.get('split_name', 'eval'),
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
