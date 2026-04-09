from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from routing_aware_atos.utils.types import CachedSample, RouteSelection


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

        if not self.config.allow_negative_scores:
            scores = np.maximum(scores, 0.0)

        if scores.sum() <= 0:
            return np.ones_like(scores) / max(len(scores), 1)

        if self.config.normalize_weights:
            return scores / scores.sum()
        return scores

    def _take_topk(self, score_vector: np.ndarray, target_pos: int) -> tuple[np.ndarray, np.ndarray]:
        if self.config.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.config.top_k}")

        scores = np.asarray(score_vector, dtype=float)
        candidate_idx = np.arange(len(scores))

        if self.config.exclude_self:
            if not 0 <= target_pos < len(scores):
                raise IndexError(f"target_pos {target_pos} is out of bounds for score vector of length {len(scores)}")
            candidate_idx = candidate_idx[candidate_idx != target_pos]

        if len(candidate_idx) == 0:
            raise ValueError("No candidate source positions available after applying routing constraints")

        k = min(self.config.top_k, len(candidate_idx))
        candidate_scores = scores[candidate_idx]
        topk_local_idx = np.argpartition(candidate_scores, -k)[-k:]
        topk_local_idx = topk_local_idx[np.argsort(candidate_scores[topk_local_idx])[::-1]]
        idx = candidate_idx[topk_local_idx]
        raw = scores[idx]
        weights = self._normalize(raw.astype(float))
        return idx.astype(int), weights.astype(float)
