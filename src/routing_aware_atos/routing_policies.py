from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from routing_aware_atos.routed_types import CachedSample, RouteSelection


@dataclass
class RoutingPolicyConfig:
    top_k: int = 1
    normalize_weights: bool = True
    exclude_self: bool = False
    allow_negative_scores: bool = False


class RoutingPolicy(ABC):
    name: str = "base"
    requires_attention: bool = False
    requires_attribution: bool = False

    def __init__(self, config: Optional[RoutingPolicyConfig] = None):
        self.config = config or RoutingPolicyConfig()

    @abstractmethod
    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        raise NotImplementedError

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores.ndim != 1:
            raise ValueError(f"Expected 1D score vector, got shape {scores.shape}")

        scores = scores.astype(np.float64, copy=False)

        if not self.config.allow_negative_scores:
            scores = np.maximum(scores, 0.0)

        if scores.size == 0:
            raise ValueError("Cannot normalize empty score vector")

        total = float(scores.sum())
        if total <= 0:
            return np.ones_like(scores, dtype=np.float64) / len(scores)

        if self.config.normalize_weights:
            return scores / total
        return scores

    def _take_topk(
        self,
        score_vector: np.ndarray,
        target_pos: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if score_vector.ndim != 1:
            raise ValueError(
                f"Expected score_vector to be 1D, got shape {score_vector.shape}"
            )

        scores = score_vector.astype(np.float64, copy=True)

        if self.config.exclude_self and 0 <= target_pos < len(scores):
            scores[target_pos] = -np.inf

        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            raise ValueError("No finite routing scores available after masking")

        valid_indices = np.where(finite_mask)[0]
        valid_scores = scores[valid_indices]

        k = min(self.config.top_k, len(valid_indices))
        if k <= 0:
            raise ValueError(f"top_k must be positive, got {self.config.top_k}")

        local_idx = np.argpartition(valid_scores, -k)[-k:]
        local_idx = local_idx[np.argsort(valid_scores[local_idx])[::-1]]
        chosen_idx = valid_indices[local_idx]

        raw_scores = score_vector[chosen_idx].astype(np.float64, copy=False)
        weights = self._normalize(raw_scores)

        return chosen_idx.astype(int), weights.astype(float)


class SameTokenPolicy(RoutingPolicy):
    name = "same_token"

    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        sample.validate()

        if target_pos < 0 or target_pos >= sample.seq_len:
            raise IndexError(
                f"target_pos={target_pos} out of bounds for seq_len={sample.seq_len}"
            )

        return RouteSelection(
            source_ids=[int(target_pos)],
            source_weights=[1.0],
            score_type="same_token",
        )


class AttentionTop1Policy(RoutingPolicy):
    name = "attention_top1"
    requires_attention = True

    def __init__(self, config: Optional[RoutingPolicyConfig] = None):
        config = config or RoutingPolicyConfig(top_k=1)
        config.top_k = 1
        super().__init__(config=config)

    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        sample.validate()

        if sample.attention_scores is None:
            raise ValueError("AttentionTop1Policy requires attention_scores in sample")

        key = (source_layer, target_layer)
        if key not in sample.attention_scores:
            raise KeyError(f"Missing attention score matrix for key {key}")

        score_matrix = sample.attention_scores[key]
        score_vector = np.asarray(score_matrix[target_pos], dtype=np.float64)

        idx, weights = self._take_topk(score_vector=score_vector, target_pos=target_pos)
        return RouteSelection(
            source_ids=idx.tolist(),
            source_weights=weights.tolist(),
            score_type="attention_top1",
        )


class AttentionTopKPolicy(RoutingPolicy):
    name = "attention_topk"
    requires_attention = True

    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        sample.validate()

        if sample.attention_scores is None:
            raise ValueError("AttentionTopKPolicy requires attention_scores in sample")

        key = (source_layer, target_layer)
        if key not in sample.attention_scores:
            raise KeyError(f"Missing attention score matrix for key {key}")

        score_matrix = sample.attention_scores[key]
        score_vector = np.asarray(score_matrix[target_pos], dtype=np.float64)

        idx, weights = self._take_topk(score_vector=score_vector, target_pos=target_pos)
        return RouteSelection(
            source_ids=idx.tolist(),
            source_weights=weights.tolist(),
            score_type="attention_topk",
        )


class AttributionTopKPolicy(RoutingPolicy):
    name = "attribution_topk"
    requires_attribution = True

    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        sample.validate()

        if sample.attribution_scores is None:
            raise ValueError(
                "AttributionTopKPolicy requires attribution_scores in sample"
            )

        key = (source_layer, target_layer)
        if key not in sample.attribution_scores:
            raise KeyError(f"Missing attribution score matrix for key {key}")

        score_matrix = sample.attribution_scores[key]
        score_vector = np.asarray(score_matrix[target_pos], dtype=np.float64)

        idx, weights = self._take_topk(score_vector=score_vector, target_pos=target_pos)
        return RouteSelection(
            source_ids=idx.tolist(),
            source_weights=weights.tolist(),
            score_type="attribution_topk",
        )


def build_routing_policy(name: str, **kwargs: object) -> RoutingPolicy:
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

__all__ = [
    "AttentionTop1Policy",
    "AttentionTopKPolicy",
    "AttributionTopKPolicy",
    "RoutingPolicy",
    "RoutingPolicyConfig",
    "SameTokenPolicy",
    "build_routing_policy",
]
