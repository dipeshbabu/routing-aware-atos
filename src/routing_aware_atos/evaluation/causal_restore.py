from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.models.transport_operator import TransportOperator
from routing_aware_atos.routed_dataset import build_routed_pairs, summarize_routes
from routing_aware_atos.routing_policies import build_routing_policy
from routing_aware_atos.sae.feature_metrics import evaluate_feature_space, summarize_feature_metrics
from routing_aware_atos.utils.io import load_npz, save_json


def restoration_score(restored_error: float, ablated_error: float) -> float:
    if ablated_error <= 1e-12:
        return 0.0
    return float(1.0 - restored_error / ablated_error)


def compute_residual_restoration(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
    Y_true = np.asarray(Y_true, dtype=np.float32)
    Y_pred = np.asarray(Y_pred, dtype=np.float32)
    Y_zero = np.zeros_like(Y_true)

    ablated_mse = float(np.mean((Y_zero - Y_true) ** 2))
    restored_mse = float(np.mean((Y_pred - Y_true) ** 2))
    ablated_mae = float(np.mean(np.abs(Y_zero - Y_true)))
    restored_mae = float(np.mean(np.abs(Y_pred - Y_true)))

    return {
        "ablated_mse": ablated_mse,
        "restored_mse": restored_mse,
        "ablated_mae": ablated_mae,
        "restored_mae": restored_mae,
        "mse_restoration": restoration_score(restored_mse, ablated_mse),
        "mae_restoration": restoration_score(restored_mae, ablated_mae),
    }


def compute_feature_restoration(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    decoder: np.ndarray,
    *,
    feature_ids: Iterable[int] | None = None,
) -> Dict[str, Any]:
    feature_metrics = evaluate_feature_space(
        Y_true=Y_true,
        Y_pred=Y_pred,
        decoder_matrix=decoder,
        feature_ids=feature_ids,
    )
    summary = summarize_feature_metrics(feature_metrics)

    A_true = Y_true @ decoder.T
    A_zero = np.zeros_like(Y_true) @ decoder.T
    A_pred = Y_pred @ decoder.T

    ablated_mse = float(np.mean((A_zero - A_true) ** 2))
    restored_mse = float(np.mean((A_pred - A_true) ** 2))
    ablated_mae = float(np.mean(np.abs(A_zero - A_true)))
    restored_mae = float(np.mean(np.abs(A_pred - A_true)))

    return {
        "feature_summary": summary,
        "ablated_feature_mse": ablated_mse,
        "restored_feature_mse": restored_mse,
        "ablated_feature_mae": ablated_mae,
        "restored_feature_mae": restored_mae,
        "feature_mse_restoration": restoration_score(restored_mse, ablated_mse),
        "feature_mae_restoration": restoration_score(restored_mae, ablated_mae),
        "feature_metrics": feature_metrics.to_dict(),
    }


def compute_logit_restoration(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    readout: np.ndarray,
) -> Dict[str, float]:
    Y_true = np.asarray(Y_true, dtype=np.float32)
    Y_pred = np.asarray(Y_pred, dtype=np.float32)
    readout = np.asarray(readout, dtype=np.float32)

    L_true = Y_true @ readout
    L_zero = np.zeros_like(Y_true) @ readout
    L_pred = Y_pred @ readout

    ablated_mse = float(np.mean((L_zero - L_true) ** 2))
    restored_mse = float(np.mean((L_pred - L_true) ** 2))
    ablated_mae = float(np.mean(np.abs(L_zero - L_true)))
    restored_mae = float(np.mean(np.abs(L_pred - L_true)))

    return {
        "ablated_logit_mse": ablated_mse,
        "restored_logit_mse": restored_mse,
        "ablated_logit_mae": ablated_mae,
        "restored_logit_mae": restored_mae,
        "logit_mse_restoration": restoration_score(restored_mse, ablated_mse),
        "logit_mae_restoration": restoration_score(restored_mae, ablated_mae),
    }


def _evaluate_causal_arrays(
    operator: TransportOperator,
    X: np.ndarray,
    Y: np.ndarray,
    decoder: np.ndarray,
    *,
    readout: np.ndarray | None = None,
    feature_ids: Iterable[int] | None = None,
    output_path: str | Path | None = None,
    split_name: str = "eval",
    operator_path: str | Path | None = None,
    pairs_path: str | Path | None = None,
    decoder_path: str | Path | None = None,
    readout_path: str | Path | None = None,
    routing_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    Y_pred = operator.predict(X)
    residual_metrics = compute_residual_restoration(Y, Y_pred)
    feature_metrics = compute_feature_restoration(Y, Y_pred, decoder, feature_ids=feature_ids)

    payload: Dict[str, Any] = {
        "split": split_name,
        "operator_path": None if operator_path is None else str(operator_path),
        "pairs_path": None if pairs_path is None else str(pairs_path),
        "decoder_path": None if decoder_path is None else str(decoder_path),
        "residual_restoration": residual_metrics,
        "feature_restoration": feature_metrics,
    }

    if readout is not None:
        payload["readout_path"] = None if readout_path is None else str(readout_path)
        payload["logit_restoration"] = compute_logit_restoration(Y, Y_pred, readout)

    if routing_info:
        payload.update(routing_info)

    if output_path is not None:
        save_json(output_path, payload)
    return payload


def evaluate_causal_restoration(
    operator_path: str | Path,
    pairs_path: str | Path,
    decoder_path: str | Path,
    *,
    readout_path: str | Path | None = None,
    feature_ids: Iterable[int] | None = None,
    output_path: str | Path | None = None,
    split_name: str = "eval",
) -> Dict[str, Any]:
    operator = TransportOperator.load(operator_path)
    pairs = load_npz(pairs_path)
    X = np.asarray(pairs["X"], dtype=np.float32)
    Y = np.asarray(pairs["Y"], dtype=np.float32)
    decoder = np.asarray(load_npz(decoder_path)["decoder"], dtype=np.float32)
    readout = None if readout_path is None else np.asarray(load_npz(readout_path)["readout"], dtype=np.float32)

    return _evaluate_causal_arrays(
        operator,
        X,
        Y,
        decoder,
        readout=readout,
        feature_ids=feature_ids,
        output_path=output_path,
        split_name=split_name,
        operator_path=operator_path,
        pairs_path=pairs_path,
        decoder_path=decoder_path,
        readout_path=readout_path,
    )


def evaluate_causal_restoration_from_cached_samples(
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
    readout_path: str | Path | None = None,
    feature_ids: Iterable[int] | None = None,
    output_path: str | Path | None = None,
    split_name: str = "eval",
    num_samples: int = 2,
    seq_len: int = 6,
    d_model: int = 4,
) -> Dict[str, Any]:
    operator = TransportOperator.load(operator_path)
    decoder = np.asarray(load_npz(decoder_path)["decoder"], dtype=np.float32)
    readout = None if readout_path is None else np.asarray(load_npz(readout_path)["readout"], dtype=np.float32)
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
        "routing_enabled": True,
        "routing_policy": routing_policy,
        "route_summary": route_summary,
    }
    return _evaluate_causal_arrays(
        operator,
        routed_pairs.X.astype(np.float32, copy=False),
        routed_pairs.Y.astype(np.float32, copy=False),
        decoder,
        readout=readout,
        feature_ids=feature_ids,
        output_path=output_path,
        split_name=split_name,
        operator_path=operator_path,
        pairs_path=None,
        decoder_path=decoder_path,
        readout_path=readout_path,
        routing_info=routing_info,
    )


def compare_causal_policy_runs(
    runs: Iterable[Mapping[str, Any]],
    *,
    feature_ids: Iterable[int] | None = None,
) -> Dict[str, Any]:
    results = []
    for run in runs:
        if run.get("pairs_path"):
            payload = evaluate_causal_restoration(
                operator_path=run["operator_path"],
                pairs_path=run["pairs_path"],
                decoder_path=run["decoder_path"],
                readout_path=run.get("readout_path"),
                feature_ids=feature_ids,
                split_name=run.get("split_name", "eval"),
            )
        else:
            routing_cfg = run.get("routing", {})
            payload = evaluate_causal_restoration_from_cached_samples(
                operator_path=run["operator_path"],
                decoder_path=run["decoder_path"],
                activation_dir_path=run.get("activation_dir_path"),
                cache_path=run.get("cache_path"),
                source_layer=run["source_layer"],
                target_layer=run["target_layer"],
                routing_policy=routing_cfg.get("policy", "same_token"),
                top_k=routing_cfg.get("top_k", 1),
                normalize_weights=routing_cfg.get("normalize_weights", True),
                exclude_self=routing_cfg.get("exclude_self", False),
                allow_negative_scores=routing_cfg.get("allow_negative_scores", False),
                include_positions=run.get("include_positions"),
                readout_path=run.get("readout_path"),
                feature_ids=feature_ids,
                split_name=run.get("split_name", "eval"),
                num_samples=run.get("num_samples", 2),
                seq_len=run.get("seq_len", 6),
                d_model=run.get("d_model", 4),
            )
        payload["policy_name"] = run["policy_name"]
        payload["rank"] = run.get("rank")
        results.append(payload)

    summary_rows = []
    for item in results:
        row = {
            "policy_name": item["policy_name"],
            "rank": item.get("rank"),
            "residual_mse_restoration": item["residual_restoration"]["mse_restoration"],
            "feature_mse_restoration": item["feature_restoration"]["feature_mse_restoration"],
            "feature_mean_r2": item["feature_restoration"]["feature_summary"]["mean_r2"],
        }
        if "logit_restoration" in item:
            row["logit_mse_restoration"] = item["logit_restoration"]["logit_mse_restoration"]
        if item.get("routing_enabled"):
            row["routing_policy"] = item.get("routing_policy")
        summary_rows.append(row)

    summary_rows.sort(key=lambda x: (x["policy_name"], -1 if x["rank"] is None else x["rank"]))
    return {"runs": results, "summary_rows": summary_rows}
