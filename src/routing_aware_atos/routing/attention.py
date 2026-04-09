from __future__ import annotations

from routing_aware_atos.routing.base import RoutingPolicy
from routing_aware_atos.utils.types import CachedSample, RouteSelection


class AttentionTop1Policy(RoutingPolicy):
    name = "attention_top1"
    requires_attention = True

    def select_sources(self, sample: CachedSample, target_pos: int, source_layer: int, target_layer: int) -> RouteSelection:
        if sample.attention_scores is None:
            raise ValueError("AttentionTop1Policy requires attention_scores in CachedSample")
        key = (source_layer, target_layer)
        if key not in sample.attention_scores:
            raise KeyError(f"Missing attention score matrix for key {key}")

        score_matrix = sample.attention_scores[key]
        score_vector = score_matrix[target_pos]
        idx, _ = self._take_topk(score_vector=score_vector, target_pos=target_pos)
        return RouteSelection(source_ids=[int(idx[0])], source_weights=[1.0], score_type="attention_top1")


class AttentionTopKPolicy(RoutingPolicy):
    name = "attention_topk"
    requires_attention = True

    def select_sources(self, sample: CachedSample, target_pos: int, source_layer: int, target_layer: int) -> RouteSelection:
        if sample.attention_scores is None:
            raise ValueError("AttentionTopKPolicy requires attention_scores in CachedSample")
        key = (source_layer, target_layer)
        if key not in sample.attention_scores:
            raise KeyError(f"Missing attention score matrix for key {key}")

        score_matrix = sample.attention_scores[key]
        score_vector = score_matrix[target_pos]
        idx, weights = self._take_topk(score_vector=score_vector, target_pos=target_pos)
        return RouteSelection(
            source_ids=idx.tolist(),
            source_weights=weights.tolist(),
            score_type="attention_topk",
        )
