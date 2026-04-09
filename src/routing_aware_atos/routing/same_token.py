from __future__ import annotations

from routing_aware_atos.routing.base import RoutingPolicy
from routing_aware_atos.utils.types import CachedSample, RouteSelection


class SameTokenPolicy(RoutingPolicy):
    name = "same_token"

    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        _ = sample, source_layer, target_layer
        return RouteSelection(source_ids=[int(target_pos)], source_weights=[1.0], score_type="same_token")
