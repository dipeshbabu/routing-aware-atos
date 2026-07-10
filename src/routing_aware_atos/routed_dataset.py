from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from routing_aware_atos.routed_types import CachedSample, RoutedPairs
from routing_aware_atos.routing_policies import RoutingPolicy


class RoutedActivationDataset:
    """
    Build routed (X, Y) pairs for transport operator training or evaluation.

    For each target token position i:
        - select one or more source positions via routing policy
        - collapse upstream source vectors into one routed vector
        - pair with downstream target vector at position i
    """

    def __init__(
        self,
        samples: Iterable[CachedSample],
        source_layer: int,
        target_layer: int,
        routing_policy: RoutingPolicy,
        include_positions: Optional[list[int]] = None,
    ):
        self.samples = list(samples)
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.routing_policy = routing_policy
        self.include_positions = include_positions

    def build_pairs(self) -> RoutedPairs:
        X_rows: list[np.ndarray] = []
        Y_rows: list[np.ndarray] = []
        routes: list[dict] = []

        for sample_idx, sample in enumerate(self.samples):
            sample.validate()

            if self.source_layer not in sample.residuals:
                raise KeyError(
                    f"Source layer {self.source_layer} missing from sample {sample_idx}"
                )
            if self.target_layer not in sample.residuals:
                raise KeyError(
                    f"Target layer {self.target_layer} missing from sample {sample_idx}"
                )

            H_src = np.asarray(sample.residuals[self.source_layer], dtype=np.float32)
            H_tgt = np.asarray(sample.residuals[self.target_layer], dtype=np.float32)

            if H_src.shape != H_tgt.shape:
                raise ValueError(
                    f"Source and target residual arrays must match shape, got {H_src.shape} vs {H_tgt.shape}"
                )

            positions = (
                self.include_positions
                if self.include_positions is not None
                else list(range(sample.seq_len))
            )

            for target_pos in positions:
                if target_pos < 0 or target_pos >= sample.seq_len:
                    raise IndexError(
                        f"target_pos={target_pos} out of bounds for seq_len={sample.seq_len}"
                    )

                route = self.routing_policy.select_sources(
                    sample=sample,
                    target_pos=target_pos,
                    source_layer=self.source_layer,
                    target_layer=self.target_layer,
                )

                if len(route.source_ids) != len(route.source_weights):
                    raise ValueError(
                        "RouteSelection source_ids and source_weights must have equal length"
                    )
                if len(route.source_ids) == 0:
                    raise ValueError("RouteSelection must contain at least one source")

                x_i = np.zeros(H_src.shape[1], dtype=np.float32)
                for source_pos, weight in zip(route.source_ids, route.source_weights):
                    if source_pos < 0 or source_pos >= sample.seq_len:
                        raise IndexError(
                            f"source_pos={source_pos} out of bounds for seq_len={sample.seq_len}"
                        )
                    x_i += float(weight) * H_src[source_pos]

                y_i = H_tgt[target_pos].astype(np.float32, copy=False)

                X_rows.append(x_i)
                Y_rows.append(y_i)
                routes.append(
                    {
                        "sample_idx": sample_idx,
                        "target_pos": int(target_pos),
                        "token": sample.tokens[target_pos],
                        "source_ids": [int(x) for x in route.source_ids],
                        "source_weights": [float(w) for w in route.source_weights],
                        "score_type": route.score_type,
                        "source_layer": int(self.source_layer),
                        "target_layer": int(self.target_layer),
                    }
                )

        if not X_rows:
            raise ValueError("No routed pairs were constructed")

        payload = RoutedPairs(
            X=np.stack(X_rows).astype(np.float32),
            Y=np.stack(Y_rows).astype(np.float32),
            routes=routes,
        )
        payload.validate()
        return payload


class ConcatenatedRoutedActivationDataset:
    """
    Build routed pairs by concatenating the top-k source vectors instead of
    collapsing them into one weighted average.

    Each input row has shape [max_sources * d_model]. Missing source slots are
    zero-padded. Source weights are applied inside each slot, so the operator can
    learn source-position-specific maps while still respecting route weights.
    """

    def __init__(
        self,
        samples: Iterable[CachedSample],
        source_layer: int,
        target_layer: int,
        routing_policy: RoutingPolicy,
        *,
        max_sources: int,
        include_positions: Optional[list[int]] = None,
    ):
        if max_sources <= 0:
            raise ValueError(f"max_sources must be positive, got {max_sources}")
        self.samples = list(samples)
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.routing_policy = routing_policy
        self.max_sources = int(max_sources)
        self.include_positions = include_positions

    def build_pairs(self) -> RoutedPairs:
        X_rows: list[np.ndarray] = []
        Y_rows: list[np.ndarray] = []
        routes: list[dict] = []

        for sample_idx, sample in enumerate(self.samples):
            sample.validate()

            if self.source_layer not in sample.residuals:
                raise KeyError(f"Source layer {self.source_layer} missing from sample {sample_idx}")
            if self.target_layer not in sample.residuals:
                raise KeyError(f"Target layer {self.target_layer} missing from sample {sample_idx}")

            H_src = np.asarray(sample.residuals[self.source_layer], dtype=np.float32)
            H_tgt = np.asarray(sample.residuals[self.target_layer], dtype=np.float32)

            if H_src.shape != H_tgt.shape:
                raise ValueError(f"Source and target residual arrays must match shape, got {H_src.shape} vs {H_tgt.shape}")

            positions = self.include_positions if self.include_positions is not None else list(range(sample.seq_len))

            for target_pos in positions:
                if target_pos < 0 or target_pos >= sample.seq_len:
                    raise IndexError(f"target_pos={target_pos} out of bounds for seq_len={sample.seq_len}")

                route = self.routing_policy.select_sources(
                    sample=sample,
                    target_pos=target_pos,
                    source_layer=self.source_layer,
                    target_layer=self.target_layer,
                )
                if len(route.source_ids) != len(route.source_weights):
                    raise ValueError("RouteSelection source_ids and source_weights must have equal length")
                if len(route.source_ids) == 0:
                    raise ValueError("RouteSelection must contain at least one source")

                x_i = np.zeros(self.max_sources * H_src.shape[1], dtype=np.float32)
                used_source_ids = route.source_ids[: self.max_sources]
                used_source_weights = route.source_weights[: self.max_sources]
                for slot_idx, (source_pos, weight) in enumerate(zip(used_source_ids, used_source_weights)):
                    if source_pos < 0 or source_pos >= sample.seq_len:
                        raise IndexError(f"source_pos={source_pos} out of bounds for seq_len={sample.seq_len}")
                    start = slot_idx * H_src.shape[1]
                    end = start + H_src.shape[1]
                    x_i[start:end] = float(weight) * H_src[source_pos]

                y_i = H_tgt[target_pos].astype(np.float32, copy=False)

                X_rows.append(x_i)
                Y_rows.append(y_i)
                routes.append(
                    {
                        "sample_idx": sample_idx,
                        "target_pos": int(target_pos),
                        "token": sample.tokens[target_pos],
                        "source_ids": [int(x) for x in route.source_ids],
                        "source_weights": [float(w) for w in route.source_weights],
                        "used_source_ids": [int(x) for x in used_source_ids],
                        "used_source_weights": [float(w) for w in used_source_weights],
                        "score_type": route.score_type,
                        "source_layer": int(self.source_layer),
                        "target_layer": int(self.target_layer),
                        "input_mode": "concat",
                        "max_sources": int(self.max_sources),
                    }
                )

        if not X_rows:
            raise ValueError("No routed pairs were constructed")

        payload = RoutedPairs(
            X=np.stack(X_rows).astype(np.float32),
            Y=np.stack(Y_rows).astype(np.float32),
            routes=routes,
        )
        payload.validate()
        return payload


def build_routed_pairs(
    samples: Iterable[CachedSample],
    source_layer: int,
    target_layer: int,
    routing_policy: RoutingPolicy,
    include_positions: Optional[list[int]] = None,
) -> RoutedPairs:
    dataset = RoutedActivationDataset(
        samples=samples,
        source_layer=source_layer,
        target_layer=target_layer,
        routing_policy=routing_policy,
        include_positions=include_positions,
    )
    return dataset.build_pairs()


def build_concatenated_routed_pairs(
    samples: Iterable[CachedSample],
    source_layer: int,
    target_layer: int,
    routing_policy: RoutingPolicy,
    *,
    max_sources: int,
    include_positions: Optional[list[int]] = None,
) -> RoutedPairs:
    dataset = ConcatenatedRoutedActivationDataset(
        samples=samples,
        source_layer=source_layer,
        target_layer=target_layer,
        routing_policy=routing_policy,
        max_sources=max_sources,
        include_positions=include_positions,
    )
    return dataset.build_pairs()


def build_same_token_pairs(
    samples: Iterable[CachedSample],
    source_layer: int,
    target_layer: int,
    include_positions: Optional[list[int]] = None,
) -> RoutedPairs:
    from routing_aware_atos.routing_policies import SameTokenPolicy

    dataset = RoutedActivationDataset(
        samples=samples,
        source_layer=source_layer,
        target_layer=target_layer,
        routing_policy=SameTokenPolicy(),
        include_positions=include_positions,
    )
    return dataset.build_pairs()


def summarize_routes(routes: list[dict]) -> dict[str, float]:
    if not routes:
        return {"num_routes": 0, "avg_num_sources": 0.0, "avg_route_entropy": 0.0}

    source_counts = [len(route["source_ids"]) for route in routes]
    entropies = []
    for route in routes:
        weights = np.asarray(route["source_weights"], dtype=np.float64)
        weights = weights[weights > 0]
        if len(weights) == 0:
            entropies.append(0.0)
            continue
        entropies.append(float(-(weights * np.log(weights)).sum()))

    return {
        "num_routes": int(len(routes)),
        "avg_num_sources": float(np.mean(source_counts)),
        "avg_route_entropy": float(np.mean(entropies)),
        "max_num_sources": int(np.max(source_counts)),
        "min_num_sources": int(np.min(source_counts)),
    }


__all__ = [
    "CachedSample",
    "ConcatenatedRoutedActivationDataset",
    "RoutedActivationDataset",
    "RoutedPairs",
    "build_concatenated_routed_pairs",
    "RoutingPolicy",
    "build_routed_pairs",
    "build_same_token_pairs",
    "summarize_routes",
]
