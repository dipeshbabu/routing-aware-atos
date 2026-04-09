from __future__ import annotations

from routing_aware_atos.routing.base import RoutingPolicy
from routing_aware_atos.utils.types import CachedSample, RouteSelection


class AttributionTopKPolicy(RoutingPolicy):
    name = "attribution_topk"
    requires_attribution = True

    def select_sources(self, sample: CachedSample, target_pos: int, source_layer: int, target_layer: int) -> RouteSelection:
        if sample.attribution_scores is None:
            raise ValueError("AttributionTopKPolicy requires attribution_scores in CachedSample")
        key = (source_layer, target_layer)
        if key not in sample.attribution_scores:
            raise KeyError(f"Missing attribution score matrix for key {key}")

        score_matrix = sample.attribution_scores[key]
        score_vector = score_matrix[target_pos]
        idx, weights = self._take_topk(score_vector=score_vector, target_pos=target_pos)
        return RouteSelection(
            source_ids=idx.tolist(),
            source_weights=weights.tolist(),
            score_type="attribution_topk",
        )
