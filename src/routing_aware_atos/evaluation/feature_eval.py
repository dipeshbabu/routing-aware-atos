from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.models.transport_operator import TransportOperator
from routing_aware_atos.routed_dataset import build_routed_pairs, summarize_routes
from routing_aware_atos.routing_policies import build_routing_policy
from routing_aware_atos.sae.feature_metrics import evaluate_feature_space, summarize_feature_metrics
from routing_aware_atos.utils.io import load_npz, save_json


def _evaluate_operator_arrays(
    operator: TransportOperator,
    X: np.ndarray,
    Y: np.ndarray,
    decoder: np.ndarray,
    *,
    feature_ids: Iterable[int] | None = None,
    output_path: str | Path | None = None,
    split_name: str = "eval",
    operator_path: str | Path | None = None,
    pairs_path: str | Path | None = None,
    routing_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    Y_pred = operator.predict(X)
    feature_metrics = evaluate_feature_space(
        Y_true=Y,
        Y_pred=Y_pred,
        decoder_matrix=decoder,
        feature_ids=feature_ids,
    )
    summary = summarize_feature_metrics(feature_metrics)
    residual_metrics = operator.evaluate(X, Y)

    payload: Dict[str, Any] = {
        "split": split_name,
        "operator_path": None if operator_path is None else str(operator_path),
        "pairs_path": None if pairs_path is None else str(pairs_path),
        "residual_metrics": residual_metrics,
        "feature_summary": summary,
        "feature_metrics": feature_metrics.to_dict(),
    }
    if routing_info:
        payload.update(routing_info)
    if output_path is not None:
        save_json(output_path, payload)
    return payload


def evaluate_operator_in_feature_space(
    operator_path: str | Path,
    pairs_path: str | Path,
    decoder_path: str | Path,
    *,
    feature_ids: Iterable[int] | None = None,
    output_path: str | Path | None = None,
    split_name: str = "eval",
) -> Dict[str, Any]:
    operator = TransportOperator.load(operator_path)
    pairs = load_npz(pairs_path)
    X = np.asarray(pairs["X"], dtype=np.float32)
    Y = np.asarray(pairs["Y"], dtype=np.float32)
    decoder = np.asarray(load_npz(decoder_path)["decoder"], dtype=np.float32)

    payload = _evaluate_operator_arrays(
        operator,
        X,
        Y,
        decoder,
        feature_ids=feature_ids,
        output_path=output_path,
        split_name=split_name,
        operator_path=operator_path,
        pairs_path=pairs_path,
    )
    payload["decoder_path"] = str(decoder_path)
    if output_path is not None:
        save_json(output_path, payload)
    return payload


def evaluate_operator_from_cached_samples(
    operator_path: str | Path,
    decoder_path: str | Path,
    *,
    activation_dir_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    source_layer: int,
    target_layer: int,
    routing_policy: str = "same_token",
    top_k: int = 1,
    normalize_weights: bool = True,
    exclude_self: bool = False,
    allow_negative_scores: bool = False,
    include_positions: list[int] | None = None,
    feature_ids: Iterable[int] | None = None,
    output_path: str | Path | None = None,
    split_name: str = "eval",
    num_samples: int = 2,
    seq_len: int = 6,
    d_model: int = 4,
) -> Dict[str, Any]:
    operator = TransportOperator.load(operator_path)
    decoder = np.asarray(load_npz(decoder_path)["decoder"], dtype=np.float32)
    policy = build_routing_policy(
        routing_policy,
        top_k=top_k,
        normalize_weights=normalize_weights,
        exclude_self=exclude_self,
        allow_negative_scores=allow_negative_scores,
    )

    if activation_dir_path:
        loader = ActivationLoader(activation_dir_path=activation_dir_path)
        idx_list = list(range(len(loader)))
        samples = list(
            loader.iter_cached_samples(
                idx_list=idx_list,
                layer_indices=[source_layer, target_layer],
                attention_layer_pairs=[(source_layer, target_layer)] if policy.requires_attention else None,
                attribution_layer_pairs=[(source_layer, target_layer)] if policy.requires_attribution else None,
            )
        )
    elif cache_path:
        loader = ActivationLoader(samples_path=cache_path)
        idx_list = list(range(len(loader)))
        samples = list(
            loader.iter_cached_samples(
                idx_list=idx_list,
                layer_indices=[source_layer, target_layer],
                attention_layer_pairs=[(source_layer, target_layer)] if policy.requires_attention else None,
                attribution_layer_pairs=[(source_layer, target_layer)] if policy.requires_attribution else None,
            )
        )
    else:
        samples = make_mock_samples(num_samples=num_samples, seq_len=seq_len, d_model=d_model)

    routed_pairs = build_routed_pairs(
        samples=samples,
        source_layer=source_layer,
        target_layer=target_layer,
        routing_policy=policy,
        include_positions=include_positions,
    )
    routed_pairs.validate()
    route_summary = summarize_routes(routed_pairs.routes)
    routing_info = {
        "decoder_path": str(decoder_path),
        "routing_enabled": True,
        "routing_policy": routing_policy,
        "route_summary": route_summary,
    }
    payload = _evaluate_operator_arrays(
        operator,
        routed_pairs.X.astype(np.float32, copy=False),
        routed_pairs.Y.astype(np.float32, copy=False),
        decoder,
        feature_ids=feature_ids,
        output_path=output_path,
        split_name=split_name,
        operator_path=operator_path,
        pairs_path=None,
        routing_info=routing_info,
    )
    if output_path is not None:
        save_json(output_path, payload)
    return payload
