from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from routing_aware_atos.routing.base import RoutingPolicy
from routing_aware_atos.utils.types import CachedSample


def compare_routing_policies(
    samples: Sequence[CachedSample],
    policies: Sequence[Tuple[str, RoutingPolicy]],
    *,
    source_layer: int,
    target_layer: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    pairwise_matches: Dict[str, List[float]] = {}

    for sample_idx, sample in enumerate(samples):
        per_target = []
        for target_pos in range(sample.seq_len):
            route_map: Dict[str, Dict[str, Any]] = {}
            for policy_name, policy in policies:
                route = policy.select_sources(sample, target_pos, source_layer, target_layer)
                top_source = int(route.source_ids[0])
                top_weight = float(route.source_weights[0])
                route_map[policy_name] = {
                    "top_source": top_source,
                    "top_weight": top_weight,
                    "source_ids": [int(x) for x in route.source_ids],
                }
            row = {
                "sample_idx": sample_idx,
                "target_pos": target_pos,
                "routes": route_map,
            }
            rows.append(row)
            per_target.append(route_map)

        names = [name for name, _ in policies]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                key = f"{a}__vs__{b}"
                matches = [
                    1.0 if target_routes[a]["top_source"] == target_routes[b]["top_source"] else 0.0
                    for target_routes in per_target
                ]
                pairwise_matches.setdefault(key, []).extend(matches)

    summary = {
        pair: {
            "top1_source_agreement": float(np.mean(vals)) if vals else 0.0,
            "count": len(vals),
        }
        for pair, vals in pairwise_matches.items()
    }
    return {
        "rows": rows,
        "summary": summary,
    }


def summarize_route_metadata(route_payload: Mapping[str, Any]) -> Dict[str, float]:
    rows = route_payload["rows"]
    route_lengths = []
    top_weights = []
    for row in rows:
        for route in row["routes"].values():
            route_lengths.append(len(route["source_ids"]))
            top_weights.append(route["top_weight"])
    return {
        "num_rows": float(len(rows)),
        "mean_route_length": float(np.mean(route_lengths)) if route_lengths else 0.0,
        "mean_top_weight": float(np.mean(top_weights)) if top_weights else 0.0,
    }
