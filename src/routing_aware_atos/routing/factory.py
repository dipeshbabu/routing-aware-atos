from __future__ import annotations

from typing import Any, Dict

from routing_aware_atos.routing.attribution import AttributionTopKPolicy
from routing_aware_atos.routing.attention import AttentionTop1Policy, AttentionTopKPolicy
from routing_aware_atos.routing.base import RoutingPolicy, RoutingPolicyConfig
from routing_aware_atos.routing.same_token import SameTokenPolicy


def build_routing_policy(name: str, **kwargs: Any) -> RoutingPolicy:
    config = RoutingPolicyConfig(**kwargs)
    registry = {
        "same_token": SameTokenPolicy,
        "attention_top1": AttentionTop1Policy,
        "attention_topk": AttentionTopKPolicy,
        "attribution_topk": AttributionTopKPolicy,
    }
    if name not in registry:
        raise KeyError(f"Unknown routing policy '{name}'. Available: {sorted(registry)}")
    return registry[name](config=config)
