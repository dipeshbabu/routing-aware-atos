from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.evaluation.causal_restore import compute_residual_restoration
from routing_aware_atos.evaluation.statistics import bootstrap_mean_ci, paired_metric_deltas
from routing_aware_atos.evaluation.transport_efficiency import (
    compute_transport_efficiency_rank_sweep_torch,
)
from routing_aware_atos.models.routed_transport_operator import RoutedTransportOperator
from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig
from routing_aware_atos.provenance import (
    fingerprint_matches,
    predictive_run_provenance,
    sae_artifact_for_layer,
    sha256_file,
    sha256_payload,
)
from routing_aware_atos.routed_dataset import (
    build_concatenated_routed_pairs,
    build_routed_pairs,
    summarize_routes,
)
from routing_aware_atos.routing_policies import build_routing_policy
from routing_aware_atos.sae.feature_metrics import evaluate_feature_space, summarize_feature_metrics
from routing_aware_atos.utils.io import load_json, load_npz, load_yaml, save_json


def _load_split_samples(
    loader: ActivationLoader,
    *,
    split_name: str,
    source_layer: int,
    target_layer: int,
    include_attention: bool,
    include_attribution: bool,
):
    pair = (source_layer, target_layer)
    return list(
        loader.iter_cached_samples(
            layer_indices=[source_layer, target_layer],
            attention_layer_pairs=[pair] if include_attention else None,
            attribution_layer_pairs=[pair] if include_attribution else None,
            split_name=split_name,
            strict=True,
        )
    )


def _build_pairs(samples, *, source_layer: int, target_layer: int, policy, policy_cfg: dict):
    input_mode = str(policy_cfg.get("input_mode", "weighted_sum"))
    if input_mode == "concat":
        return build_concatenated_routed_pairs(
            samples,
            source_layer=source_layer,
            target_layer=target_layer,
            routing_policy=policy,
            max_sources=int(policy_cfg.get("max_sources", policy_cfg.get("top_k", 1))),
        )
    if input_mode == "weighted_sum":
        return build_routed_pairs(
            samples,
            source_layer=source_layer,
            target_layer=target_layer,
            routing_policy=policy,
        )
    raise ValueError(f"Unknown input_mode {input_mode!r}")


def _fit_with_validation(
    train_pairs,
    validation_pairs,
    *,
    policy_name: str,
    route_summary: dict,
    lambdas: list[float],
    rank: int | None,
    compute_backend: str,
    device: str,
    selection_protocol: str,
    cv_folds: int,
) -> tuple[RoutedTransportOperator, list[dict[str, Any]]]:
    if not lambdas:
        raise ValueError("ridge_lambdas cannot be empty")
    candidates: list[dict[str, Any]] = []
    best_operator = None
    best_score = -np.inf

    def fit_operator(X: np.ndarray, Y: np.ndarray, ridge_lambda: float) -> RoutedTransportOperator:
        return RoutedTransportOperator(
            config=TransportOperatorConfig(
                ridge_lambda=float(ridge_lambda),
                rank=rank,
                name=f"real_{policy_name}",
                compute_backend=compute_backend,
                device=device,
            ),
            routing_policy_name=policy_name,
            route_summary=route_summary,
        ).fit(X, Y)

    if selection_protocol == "five_fold_cv":
        if cv_folds < 2 or train_pairs.X.shape[0] < cv_folds:
            raise ValueError(
                f"five_fold_cv requires at least {cv_folds} training rows, got {train_pairs.X.shape[0]}"
            )
        fold_indices = np.array_split(np.arange(train_pairs.X.shape[0]), cv_folds)
        best_lambda = None
        for ridge_lambda in lambdas:
            fold_scores: list[float] = []
            for validation_indices in fold_indices:
                training_mask = np.ones(train_pairs.X.shape[0], dtype=bool)
                training_mask[validation_indices] = False
                operator = fit_operator(
                    train_pairs.X[training_mask],
                    train_pairs.Y[training_mask],
                    ridge_lambda,
                )
                fold_scores.append(
                    float(
                        operator.evaluate(
                            train_pairs.X[validation_indices],
                            train_pairs.Y[validation_indices],
                        )["r2"]
                    )
                )
                del operator, training_mask
            mean_score = float(np.mean(fold_scores))
            candidates.append(
                {
                    "ridge_lambda": float(ridge_lambda),
                    "fold_r2": fold_scores,
                    "mean_cv_r2": mean_score,
                    "std_cv_r2": float(np.std(fold_scores, ddof=1)),
                }
            )
            if mean_score > best_score:
                best_score = mean_score
                best_lambda = float(ridge_lambda)
        assert best_lambda is not None
        best_operator = fit_operator(train_pairs.X, train_pairs.Y, best_lambda)
        heldout_validation = best_operator.evaluate(validation_pairs.X, validation_pairs.Y)
        for candidate in candidates:
            candidate["selected"] = bool(candidate["ridge_lambda"] == best_lambda)
            if candidate["selected"]:
                candidate["heldout_validation_r2"] = float(heldout_validation["r2"])
                candidate["heldout_validation_mse"] = float(heldout_validation["mse"])
        return best_operator, candidates

    if selection_protocol != "heldout_validation":
        raise ValueError(f"Unknown ridge selection protocol {selection_protocol!r}")
    for ridge_lambda in lambdas:
        operator = fit_operator(train_pairs.X, train_pairs.Y, ridge_lambda)
        validation_metrics = operator.evaluate(validation_pairs.X, validation_pairs.Y)
        candidates.append(
            {
                "ridge_lambda": float(ridge_lambda),
                "validation_r2": float(validation_metrics["r2"]),
                "validation_mse": float(validation_metrics["mse"]),
            }
        )
        if validation_metrics["r2"] > best_score:
            best_score = validation_metrics["r2"]
            best_operator = operator
    assert best_operator is not None
    return best_operator, candidates


def _feature_ids_for_layer(cfg: dict, target_layer: int) -> list[int]:
    feature_dir = Path(cfg["feature_selection"].get("output_dir", "artifacts/features"))
    payload = load_json(feature_dir / f"layer_{target_layer}_features.json")
    return [int(feature_id) for feature_id in payload["high_quality_feature_ids"]]


def _sae_artifact_for_layer(cfg: dict, target_layer: int) -> str:
    return str(sae_artifact_for_layer(cfg, target_layer))


def _load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_one(
    *,
    cfg: dict,
    experiment_cfg: dict,
    pair_cfg: dict,
    policy_cfg: dict,
    train_samples,
    validation_samples,
    test_samples,
    force: bool,
) -> dict[str, Any]:
    source_layer = int(pair_cfg["source_layer"])
    target_layer = int(pair_cfg["target_layer"])
    policy_label = str(policy_cfg["name"])
    routing_policy_name = str(policy_cfg.get("routing_policy", policy_label))
    output_dir = (
        Path(experiment_cfg.get("output_dir", "outputs/real"))
        / "runs"
        / f"L{source_layer}_to_L{target_layer}"
        / policy_label
    )
    result_path = output_dir / "result.json"
    sae_path = _sae_artifact_for_layer(cfg, target_layer)
    fingerprint_inputs = predictive_run_provenance(cfg, pair_cfg, policy_cfg)
    run_fingerprint = sha256_payload(fingerprint_inputs)
    operator_path = output_dir / "operator.npz"
    metadata_path = output_dir / "metadata.json"
    bundle_metadata = _load_json_or_empty(metadata_path)
    operator_sha256 = sha256_file(operator_path) if operator_path.exists() else None
    current_operator_bundle = bool(
        operator_sha256
        and fingerprint_matches(
            bundle_metadata,
            expected_provenance=fingerprint_inputs,
        )
        and bundle_metadata.get("operator_sha256") == operator_sha256
    )
    if result_path.exists() and not force:
        existing_result = _load_json_or_empty(result_path)
        if (
            current_operator_bundle
            and fingerprint_matches(
                existing_result,
                expected_provenance=fingerprint_inputs,
            )
            and existing_result.get("operator_sha256") == operator_sha256
        ):
            print(f"Skipping completed run -> {result_path}")
            return existing_result
        print(f"Recomputing stale run -> {result_path}")

    policy = build_routing_policy(
        routing_policy_name,
        top_k=int(policy_cfg.get("top_k", 1)),
        normalize_weights=bool(policy_cfg.get("normalize_weights", True)),
        exclude_self=bool(policy_cfg.get("exclude_self", False)),
        allow_negative_scores=bool(policy_cfg.get("allow_negative_scores", False)),
        random_seed=int(policy_cfg.get("random_seed", 0)),
        causal_only=bool(policy_cfg.get("causal_only", experiment_cfg.get("causal_only", False))),
    )
    train_pairs = _build_pairs(
        train_samples,
        source_layer=source_layer,
        target_layer=target_layer,
        policy=policy,
        policy_cfg=policy_cfg,
    )
    validation_pairs = _build_pairs(
        validation_samples,
        source_layer=source_layer,
        target_layer=target_layer,
        policy=policy,
        policy_cfg=policy_cfg,
    )
    test_pairs = _build_pairs(
        test_samples,
        source_layer=source_layer,
        target_layer=target_layer,
        policy=policy,
        policy_cfg=policy_cfg,
    )
    route_summary = summarize_routes(train_pairs.routes)
    lambdas = [
        float(value)
        for value in policy_cfg.get("ridge_lambdas", experiment_cfg.get("ridge_lambdas", [0.01]))
    ]
    rank = experiment_cfg.get("rank")
    rank = None if rank is None else int(rank)
    reused_operator = bool(
        not force
        and current_operator_bundle
    )
    if reused_operator:
        operator = TransportOperator.load(operator_path)
        tuning = list(bundle_metadata.get("ridge_candidates", []))
        print(f"Reusing fitted operator -> {operator_path}")
    else:
        operator, tuning = _fit_with_validation(
            train_pairs,
            validation_pairs,
            policy_name=policy_label,
            route_summary=route_summary,
            lambdas=lambdas,
            rank=rank,
            compute_backend=str(experiment_cfg.get("compute_backend", "torch")),
            device=str(experiment_cfg.get("device", "cuda")),
            selection_protocol=str(experiment_cfg.get("ridge_selection", "heldout_validation")),
            cv_folds=int(experiment_cfg.get("cv_folds", 5)),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        operator.save_bundle(
            output_dir,
            extra_metadata={
                "source_layer": source_layer,
                "target_layer": target_layer,
                "leap": int(pair_cfg.get("leap", target_layer - source_layer)),
                "policy_label": policy_label,
                "routing_policy": routing_policy_name,
                "input_mode": policy_cfg.get("input_mode", "weighted_sum"),
                "causal_only": bool(policy.config.causal_only),
                "ridge_candidates": tuning,
                "ridge_selection": str(
                    experiment_cfg.get("ridge_selection", "heldout_validation")
                ),
                "cv_folds": int(experiment_cfg.get("cv_folds", 5)),
                "split_protocol": "dataset_manifest",
                "train_rows": int(train_pairs.X.shape[0]),
                "validation_rows": int(validation_pairs.X.shape[0]),
                "test_rows": int(test_pairs.X.shape[0]),
                "run_fingerprint": run_fingerprint,
                "provenance": fingerprint_inputs,
            },
        )
        operator_sha256 = sha256_file(operator_path)

    test_prediction = operator.predict(test_pairs.X)
    residual_metrics = operator.evaluate(test_pairs.X, test_pairs.Y)
    residual_restoration = compute_residual_restoration(test_pairs.Y, test_prediction)
    sae_arrays = load_npz(sae_path)
    feature_ids = _feature_ids_for_layer(cfg, target_layer)
    feature_metrics = evaluate_feature_space(
        test_pairs.Y,
        test_prediction,
        np.asarray(sae_arrays["decoder"], dtype=np.float32),
        feature_ids,
        sae_arrays=sae_arrays,
        activated_only=bool(experiment_cfg.get("activated_only", True)),
        min_activations=int(experiment_cfg.get("min_feature_activations", 10)),
        compute_device=str(experiment_cfg.get("device", "cuda")),
        batch_size=int(experiment_cfg.get("metric_batch_size", 1024)),
        normalize_decoder=bool(experiment_cfg.get("normalize_decoder", True)),
        min_r2=(
            None
            if experiment_cfg.get("min_feature_r2") is None
            else float(experiment_cfg["min_feature_r2"])
        ),
    )
    feature_summary = summarize_feature_metrics(feature_metrics)
    feature_r2_ci = bootstrap_mean_ci(
        feature_metrics.r2,
        n_bootstrap=int(experiment_cfg.get("bootstrap_samples", 1000)),
        random_seed=int(experiment_cfg.get("bootstrap_seed", 42)),
    )
    efficiency_payload = None
    efficiency_cfg = experiment_cfg.get("transport_efficiency", {})
    efficiency_pairs = {
        tuple(int(value) for value in str(item).split(":"))
        for item in efficiency_cfg.get("layer_pairs", [])
    }
    if bool(efficiency_cfg.get("enabled", False)) and policy_label in set(
        efficiency_cfg.get("policies", [])
    ) and (
        not efficiency_pairs or (source_layer, target_layer) in efficiency_pairs
    ):
        max_rank = min(operator.weight.shape)
        rank_start = int(efficiency_cfg.get("rank_start", 1))
        rank_step = int(efficiency_cfg.get("rank_step", 50))
        ranks = list(range(rank_start, max_rank + 1, rank_step))
        if max_rank not in ranks:
            ranks.append(max_rank)
        efficiency_payload = compute_transport_efficiency_rank_sweep_torch(
            operator,
            test_pairs.X,
            test_pairs.Y,
            ranks=ranks,
            device=str(experiment_cfg.get("device", "cuda")),
        )
    result = {
        "source_layer": source_layer,
        "target_layer": target_layer,
        "leap": int(pair_cfg.get("leap", target_layer - source_layer)),
        "policy_name": policy_label,
        "routing_policy": routing_policy_name,
        "input_mode": policy_cfg.get("input_mode", "weighted_sum"),
        "causal_only": bool(policy.config.causal_only),
        "run_fingerprint": run_fingerprint,
        "provenance": fingerprint_inputs,
        "operator_path": str(operator_path),
        "operator_sha256": operator_sha256,
        "reused_operator": reused_operator,
        "sae_artifact": sae_path,
        "selected_ridge_lambda": float(operator.config.ridge_lambda),
        "ridge_candidates": tuning,
        "ridge_selection": str(experiment_cfg.get("ridge_selection", "heldout_validation")),
        "cv_folds": int(experiment_cfg.get("cv_folds", 5)),
        "residual_metrics": residual_metrics,
        "residual_restoration": residual_restoration,
        "feature_summary": feature_summary,
        "feature_mean_r2_ci": feature_r2_ci,
        "feature_metrics": feature_metrics.to_dict(),
        "route_summary": route_summary,
        "train_rows": int(train_pairs.X.shape[0]),
        "validation_rows": int(validation_pairs.X.shape[0]),
        "test_rows": int(test_pairs.X.shape[0]),
    }
    if efficiency_payload is not None:
        result["transport_efficiency"] = efficiency_payload
    save_json(result_path, result)
    print(f"Completed real run L{source_layer}->L{target_layer} policy={policy_label}")
    del train_pairs, validation_pairs, test_pairs, test_prediction, operator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split-safe real routing-aware ATO experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", action="append", help="Run only selected policy names")
    parser.add_argument("--pair", action="append", help="Run only source:target pairs, e.g. 9:10")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    experiment_cfg = cfg["experiments"]
    selected_policies = set(args.policy or [])
    selected_pairs = {
        tuple(int(value) for value in item.split(":"))
        for item in (args.pair or [])
    }
    pair_configs = [
        pair
        for pair in experiment_cfg["layer_pairs"]
        if not selected_pairs
        or (int(pair["source_layer"]), int(pair["target_layer"])) in selected_pairs
    ]
    policy_configs = [
        policy
        for policy in experiment_cfg["policies"]
        if not selected_policies or str(policy["name"]) in selected_policies
    ]
    if not pair_configs or not policy_configs:
        raise ValueError("The requested pair/policy filters selected no experiments")

    include_attention = any(
        build_routing_policy(
            str(policy.get("routing_policy", policy["name"])),
            top_k=int(policy.get("top_k", 1)),
            causal_only=bool(policy.get("causal_only", experiment_cfg.get("causal_only", False))),
        ).requires_attention
        for policy in policy_configs
    )
    include_attribution = any(
        build_routing_policy(
            str(policy.get("routing_policy", policy["name"])),
            top_k=int(policy.get("top_k", 1)),
            causal_only=bool(policy.get("causal_only", experiment_cfg.get("causal_only", False))),
        ).requires_attribution
        for policy in policy_configs
    )
    loader = ActivationLoader(activation_dir_path=experiment_cfg["activation_dir_path"])
    results: list[dict[str, Any]] = []
    try:
        for pair_cfg in pair_configs:
            source_layer = int(pair_cfg["source_layer"])
            target_layer = int(pair_cfg["target_layer"])
            train_samples = _load_split_samples(
                loader,
                split_name="train",
                source_layer=source_layer,
                target_layer=target_layer,
                include_attention=include_attention,
                include_attribution=include_attribution,
            )
            validation_samples = _load_split_samples(
                loader,
                split_name="validation",
                source_layer=source_layer,
                target_layer=target_layer,
                include_attention=include_attention,
                include_attribution=include_attribution,
            )
            test_samples = _load_split_samples(
                loader,
                split_name="test",
                source_layer=source_layer,
                target_layer=target_layer,
                include_attention=include_attention,
                include_attribution=include_attribution,
            )
            for policy_cfg in policy_configs:
                results.append(
                    _run_one(
                        cfg=cfg,
                        experiment_cfg=experiment_cfg,
                        pair_cfg=pair_cfg,
                        policy_cfg=policy_cfg,
                        train_samples=train_samples,
                        validation_samples=validation_samples,
                        test_samples=test_samples,
                        force=args.force,
                    )
                )
            del train_samples, validation_samples, test_samples
            gc.collect()
    finally:
        loader.close()

    summary_rows = [
        {
            "source_layer": row["source_layer"],
            "target_layer": row["target_layer"],
            "leap": row["leap"],
            "policy_name": row["policy_name"],
            "residual_r2": row["residual_metrics"]["r2"],
            "feature_mean_r2": row["feature_summary"]["mean_r2"],
            "feature_median_r2": row["feature_summary"]["median_r2"],
            "residual_mse_restoration": row["residual_restoration"]["mse_restoration"],
        }
        for row in results
    ]
    paired_deltas: list[dict[str, Any]] = []
    for source_layer, target_layer in sorted(
        {(row["source_layer"], row["target_layer"]) for row in summary_rows}
    ):
        pair_rows = [
            row
            for row in summary_rows
            if row["source_layer"] == source_layer and row["target_layer"] == target_layer
        ]
        if any(row["policy_name"] == "same_token" for row in pair_rows):
            deltas = paired_metric_deltas(
                pair_rows,
                baseline_policy="same_token",
                metric_keys=["residual_r2", "feature_mean_r2", "residual_mse_restoration"],
                rank_key="leap",
            )
            for delta in deltas:
                delta["source_layer"] = source_layer
                delta["target_layer"] = target_layer
            paired_deltas.extend(deltas)

    output_dir = Path(experiment_cfg.get("output_dir", "outputs/real"))
    save_json(
        output_dir / "experiment_summary.json",
        {"runs": summary_rows, "paired_deltas": paired_deltas},
    )
    print(f"Saved real experiment summary -> {output_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
