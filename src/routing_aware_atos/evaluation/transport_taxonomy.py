from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from routing_aware_atos.utils.io import load_json, save_json


DEFAULT_ORDER = [
    "same_token_transport",
    "cross_token_transport",
    "multi_source_transport",
    "unexplained_or_synthesized",
]


def classify_feature_transport(
    feature_scores: Mapping[str, float],
    *,
    same_token_policy: str = "same_token",
    cross_token_policies: Sequence[str] = ("attention_top1", "cross_token"),
    multi_source_policies: Sequence[str] = ("attention_topk", "attribution_topk", "multi_source"),
    threshold: float = 0.7,
) -> str:
    same_r2 = float(feature_scores.get(same_token_policy, -1.0))
    cross_r2 = max([float(feature_scores.get(name, -1.0)) for name in cross_token_policies] or [-1.0])
    multi_r2 = max([float(feature_scores.get(name, -1.0)) for name in multi_source_policies] or [-1.0])

    if same_r2 >= threshold:
        return "same_token_transport"
    if cross_r2 >= threshold:
        return "cross_token_transport"
    if multi_r2 >= threshold:
        return "multi_source_transport"
    return "unexplained_or_synthesized"


def build_feature_policy_matrix(run_payloads: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, float]]:
    feature_scores: Dict[int, Dict[str, float]] = {}
    for payload in run_payloads:
        policy_name = str(payload["policy_name"])
        feature_metrics = payload["feature_metrics"]
        if "feature_ids" in feature_metrics and "r2" in feature_metrics:
            feature_ids = feature_metrics["feature_ids"]
            r2_values = feature_metrics["r2"]
            if len(feature_ids) != len(r2_values):
                raise ValueError(
                    f"Mismatched feature metric lengths for policy {policy_name}: "
                    f"{len(feature_ids)} feature_ids vs {len(r2_values)} r2 values"
                )
            for feature_id, r2 in zip(feature_ids, r2_values):
                feature_scores.setdefault(int(feature_id), {})
                feature_scores[int(feature_id)][policy_name] = float(r2)
            continue

        for feature_id_str, metrics in feature_metrics.items():
            try:
                feature_id = int(feature_id_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(metrics, Mapping) or "r2" not in metrics:
                continue
            r2 = float(metrics["r2"])
            feature_scores.setdefault(feature_id, {})
            feature_scores[feature_id][policy_name] = r2
    return feature_scores


def build_transport_taxonomy(
    run_payloads: Sequence[Mapping[str, Any]],
    *,
    same_token_policy: str = "same_token",
    cross_token_policies: Sequence[str] = ("attention_top1",),
    multi_source_policies: Sequence[str] = ("attention_topk", "attribution_topk"),
    threshold: float = 0.7,
) -> Dict[str, Any]:
    feature_policy_matrix = build_feature_policy_matrix(run_payloads)
    feature_rows: List[Dict[str, Any]] = []

    for feature_id, policy_scores in sorted(feature_policy_matrix.items()):
        label = classify_feature_transport(
            policy_scores,
            same_token_policy=same_token_policy,
            cross_token_policies=cross_token_policies,
            multi_source_policies=multi_source_policies,
            threshold=threshold,
        )
        row = {
            "feature_id": int(feature_id),
            "label": label,
            "policy_scores": {k: float(v) for k, v in sorted(policy_scores.items())},
            "best_policy": max(policy_scores, key=policy_scores.get),
            "best_r2": float(max(policy_scores.values())),
        }
        feature_rows.append(row)

    counts = Counter(row["label"] for row in feature_rows)
    total = len(feature_rows)
    summary_rows = []
    for label in DEFAULT_ORDER:
        count = int(counts.get(label, 0))
        summary_rows.append({
            "label": label,
            "count": count,
            "fraction": 0.0 if total == 0 else float(count / total),
        })

    return {
        "threshold": float(threshold),
        "num_features": int(total),
        "feature_rows": feature_rows,
        "summary_rows": summary_rows,
    }


def save_transport_taxonomy(payload: Mapping[str, Any], output_path: str | Path) -> None:
    save_json(output_path, payload)


def load_feature_eval_payload(path: str | Path) -> Dict[str, Any]:
    payload = load_json(path)
    if "feature_metrics" not in payload:
        raise ValueError(f"Expected feature evaluation payload at {path}")
    return payload
