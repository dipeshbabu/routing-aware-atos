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
    random_seed: int = 0


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

    def _valid_source_indices(self, sample: CachedSample, target_pos: int) -> np.ndarray:
        indices = np.arange(sample.seq_len, dtype=int)
        if self.config.exclude_self and 0 <= target_pos < sample.seq_len:
            indices = indices[indices != target_pos]
        if indices.size == 0:
            raise ValueError("No valid source indices available after masking")
        return indices

    def _deterministic_rng(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> np.random.Generator:
        sample_idx = 0
        if sample.metadata and "sample_idx" in sample.metadata:
            sample_idx = int(sample.metadata["sample_idx"])
        seed = (
            int(self.config.random_seed) * 1_000_003
            + sample_idx * 100_003
            + int(target_pos) * 10_007
            + int(source_layer) * 503
            + int(target_layer) * 97
        ) % (2**32)
        return np.random.default_rng(seed)


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


class PreviousTokenPolicy(RoutingPolicy):
    name = "previous_token"

    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        sample.validate()
        if target_pos < 0 or target_pos >= sample.seq_len:
            raise IndexError(f"target_pos={target_pos} out of bounds for seq_len={sample.seq_len}")
        source_pos = max(0, target_pos - 1)
        if self.config.exclude_self and source_pos == target_pos:
            candidates = self._valid_source_indices(sample, target_pos)
            source_pos = int(candidates[0])
        return RouteSelection(
            source_ids=[int(source_pos)],
            source_weights=[1.0],
            score_type="previous_token",
        )


class NextTokenPolicy(RoutingPolicy):
    name = "next_token"

    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        sample.validate()
        if target_pos < 0 or target_pos >= sample.seq_len:
            raise IndexError(f"target_pos={target_pos} out of bounds for seq_len={sample.seq_len}")
        source_pos = min(sample.seq_len - 1, target_pos + 1)
        if self.config.exclude_self and source_pos == target_pos:
            candidates = self._valid_source_indices(sample, target_pos)
            source_pos = int(candidates[-1])
        return RouteSelection(
            source_ids=[int(source_pos)],
            source_weights=[1.0],
            score_type="next_token",
        )


class UniformTopKPolicy(RoutingPolicy):
    name = "uniform_topk"

    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        sample.validate()
        candidates = self._valid_source_indices(sample, target_pos)
        k = min(int(self.config.top_k), len(candidates))
        if k <= 0:
            raise ValueError(f"top_k must be positive, got {self.config.top_k}")
        if k == len(candidates):
            chosen = candidates
        else:
            positions = np.linspace(0, len(candidates) - 1, num=k)
            chosen = candidates[np.rint(positions).astype(int)]
        weights = np.ones(k, dtype=np.float64) / float(k)
        return RouteSelection(
            source_ids=[int(x) for x in chosen],
            source_weights=weights.tolist(),
            score_type="uniform_topk",
        )


class RandomTopKPolicy(RoutingPolicy):
    name = "random_topk"

    def select_sources(
        self,
        sample: CachedSample,
        target_pos: int,
        source_layer: int,
        target_layer: int,
    ) -> RouteSelection:
        sample.validate()
        candidates = self._valid_source_indices(sample, target_pos)
        k = min(int(self.config.top_k), len(candidates))
        if k <= 0:
            raise ValueError(f"top_k must be positive, got {self.config.top_k}")
        rng = self._deterministic_rng(sample, target_pos, source_layer, target_layer)
        chosen = np.sort(rng.choice(candidates, size=k, replace=False))
        weights = np.ones(k, dtype=np.float64) / float(k)
        return RouteSelection(
            source_ids=[int(x) for x in chosen],
            source_weights=weights.tolist(),
            score_type="random_topk",
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


class ShuffledAttentionTopKPolicy(RoutingPolicy):
    name = "shuffled_attention_topk"
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
            raise ValueError("ShuffledAttentionTopKPolicy requires attention_scores in sample")

        key = (source_layer, target_layer)
        if key not in sample.attention_scores:
            raise KeyError(f"Missing attention score matrix for key {key}")

        score_matrix = sample.attention_scores[key]
        score_vector = np.asarray(score_matrix[target_pos], dtype=np.float64)
        rng = self._deterministic_rng(sample, target_pos, source_layer, target_layer)
        shuffled_scores = score_vector[rng.permutation(len(score_vector))]

        idx, weights = self._take_topk(score_vector=shuffled_scores, target_pos=target_pos)
        return RouteSelection(
            source_ids=idx.tolist(),
            source_weights=weights.tolist(),
            score_type="shuffled_attention_topk",
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
        "previous_token": PreviousTokenPolicy,
        "next_token": NextTokenPolicy,
        "uniform_topk": UniformTopKPolicy,
        "random_topk": RandomTopKPolicy,
        "attention_top1": AttentionTop1Policy,
        "attention_topk": AttentionTopKPolicy,
        "shuffled_attention_topk": ShuffledAttentionTopKPolicy,
        "attribution_topk": AttributionTopKPolicy,
    }
    if name not in registry:
        raise KeyError(f"Unknown routing policy '{name}'. Available: {sorted(registry)}")
    return registry[name](config=config)

__all__ = [
    "AttentionTop1Policy",
    "AttentionTopKPolicy",
    "AttributionTopKPolicy",
    "NextTokenPolicy",
    "PreviousTokenPolicy",
    "RandomTopKPolicy",
    "RoutingPolicy",
    "RoutingPolicyConfig",
    "SameTokenPolicy",
    "ShuffledAttentionTopKPolicy",
    "UniformTopKPolicy",
    "build_routing_policy",
]
