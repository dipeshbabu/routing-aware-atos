from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
from typing import Any

import numpy as np

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.data.baseline_pairs import SameTokenBaselineBuilder
from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.models.routed_transport_operator import RoutedTransportOperator
from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig
from routing_aware_atos.routed_dataset import build_concatenated_routed_pairs, build_routed_pairs, summarize_routes
from routing_aware_atos.routing_policies import build_routing_policy
from routing_aware_atos.utils.io import load_npz, load_yaml, save_json


def _build_routing_policy(cfg: dict[str, Any]):
    routing_cfg = cfg["routing"]
    return build_routing_policy(
        str(routing_cfg["policy"]),
        top_k=int(routing_cfg.get("top_k", 1)),
        normalize_weights=bool(routing_cfg.get("normalize_weights", True)),
        exclude_self=bool(routing_cfg.get("exclude_self", False)),
        allow_negative_scores=bool(routing_cfg.get("allow_negative_scores", False)),
        random_seed=int(routing_cfg.get("random_seed", 0)),
        causal_only=bool(routing_cfg.get("causal_only", False)),
    )


def _build_index_splits(num_samples: int) -> tuple[list[int], list[int], list[int]]:
    if num_samples < 3:
        indices = list(range(num_samples))
        return indices, indices, indices

    train_end = max(1, int(round(num_samples * 0.6)))
    val_end = max(train_end + 1, int(round(num_samples * 0.8)))
    indices = list(range(num_samples))
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end] or train_indices
    test_indices = indices[val_end:] or val_indices
    return train_indices, val_indices, test_indices


def _resolve_index_splits(
    loader: ActivationLoader,
    *,
    use_dataset_splits: bool,
) -> tuple[list[int], list[int], list[int]]:
    if not use_dataset_splits:
        return _build_index_splits(len(loader))
    return (
        loader.indices_for_split("train"),
        loader.indices_for_split("validation"),
        loader.indices_for_split("test"),
    )


def _collect_cached_samples(
    loader: ActivationLoader,
    idx_list: list[int],
    source_layer: int,
    target_layer: int,
    use_attention: bool,
    use_attribution: bool,
):
    return list(
        loader.iter_cached_samples(
            idx_list=idx_list,
            layer_indices=[source_layer, target_layer],
            attention_layer_pairs=[(source_layer, target_layer)] if use_attention else None,
            attribution_layer_pairs=[(source_layer, target_layer)] if use_attribution else None,
            strict=True,
        )
    )


def _evaluate_xy(transport_operator: TransportOperator, X: np.ndarray, Y: np.ndarray) -> dict[str, float]:
    metrics = transport_operator.evaluate(X, Y)
    mse = float(metrics["mse"])
    rmse = float(np.sqrt(mse))
    return {
        "mse": mse,
        "rmse": rmse,
        "r2_score": float(metrics["r2"]),
    }


def _build_pairs_for_mode(
    samples,
    *,
    source_layer: int,
    target_layer: int,
    routing_policy,
    cfg: dict[str, Any],
):
    routing_cfg = cfg.get("routing", {})
    input_mode = routing_cfg.get("input_mode", cfg.get("input_mode", "weighted_sum"))
    if input_mode == "concat":
        return build_concatenated_routed_pairs(
            samples=samples,
            source_layer=source_layer,
            target_layer=target_layer,
            routing_policy=routing_policy,
            max_sources=int(routing_cfg.get("max_sources", routing_cfg.get("top_k", 1))),
            include_positions=cfg.get("include_positions"),
        )
    if input_mode == "weighted_sum":
        return build_routed_pairs(
            samples=samples,
            source_layer=source_layer,
            target_layer=target_layer,
            routing_policy=routing_policy,
            include_positions=cfg.get("include_positions"),
        )
    raise ValueError(f"Unknown input_mode {input_mode!r}")


def _load_loader_or_samples(cfg: dict[str, Any]):
    if cfg.get("activation_dir_path"):
        return ActivationLoader(activation_dir_path=cfg["activation_dir_path"]), None
    if cfg.get("cache_path"):
        return ActivationLoader(samples_path=cfg["cache_path"]), None

    samples = make_mock_samples(
        num_samples=cfg.get("num_samples", 2),
        seq_len=cfg.get("seq_len", 6),
        d_model=cfg.get("d_model", 4),
    )
    return None, samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline or routed transport operators across layer pairs.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    experiment_cfg = cfg.get("experiment", {})
    layer_list = experiment_cfg.get("L", [cfg.get("source_layer", 10)])
    k_list = experiment_cfg.get("k", [cfg.get("k", cfg.get("target_layer", 12) - cfg.get("source_layer", 10))])
    routing_enabled = bool(cfg.get("routing", {}).get("enabled", False))
    routing_policy = _build_routing_policy(cfg) if routing_enabled else None

    loader, fallback_samples = _load_loader_or_samples(cfg)
    results: list[dict[str, Any]] = []

    for source_layer in layer_list:
        for k in k_list:
            target_layer = int(source_layer) + int(k)
            val_pairs = None
            test_pairs = None

            if cfg.get("pairs_path"):
                arrays = load_npz(cfg["pairs_path"])
                X, Y = arrays["X"], arrays["Y"]
                route_summary: dict[str, Any] = {}
            elif not routing_enabled:
                if loader is not None:
                    train_indices, val_indices, test_indices = _resolve_index_splits(
                        loader,
                        use_dataset_splits=bool(cfg.get("use_dataset_splits", False)),
                    )
                    train_samples = _collect_cached_samples(
                        loader=loader,
                        idx_list=train_indices,
                        source_layer=int(source_layer),
                        target_layer=target_layer,
                        use_attention=False,
                        use_attribution=False,
                    )
                    val_samples = _collect_cached_samples(
                        loader=loader,
                        idx_list=val_indices,
                        source_layer=int(source_layer),
                        target_layer=target_layer,
                        use_attention=False,
                        use_attribution=False,
                    )
                    test_samples = _collect_cached_samples(
                        loader=loader,
                        idx_list=test_indices,
                        source_layer=int(source_layer),
                        target_layer=target_layer,
                        use_attention=False,
                        use_attribution=False,
                    )
                else:
                    train_samples = list(fallback_samples or [])
                    val_samples = train_samples
                    test_samples = train_samples

                X, Y, _ = SameTokenBaselineBuilder(
                    samples=train_samples,
                    source_layer=int(source_layer),
                    target_layer=target_layer,
                    include_positions=cfg.get("include_positions"),
                ).build_pairs()
                X_val, Y_val, _ = SameTokenBaselineBuilder(
                    samples=val_samples,
                    source_layer=int(source_layer),
                    target_layer=target_layer,
                    include_positions=cfg.get("include_positions"),
                ).build_pairs()
                X_test, Y_test, _ = SameTokenBaselineBuilder(
                    samples=test_samples,
                    source_layer=int(source_layer),
                    target_layer=target_layer,
                    include_positions=cfg.get("include_positions"),
                ).build_pairs()
                val_pairs = (X_val, Y_val)
                test_pairs = (X_test, Y_test)
                route_summary = {}
            else:
                if loader is not None:
                    train_indices, val_indices, test_indices = _resolve_index_splits(
                        loader,
                        use_dataset_splits=bool(cfg.get("use_dataset_splits", False)),
                    )
                    use_attention = bool(routing_policy.requires_attention)
                    use_attribution = bool(routing_policy.requires_attribution)
                    train_samples = _collect_cached_samples(
                        loader=loader,
                        idx_list=train_indices,
                        source_layer=int(source_layer),
                        target_layer=target_layer,
                        use_attention=use_attention,
                        use_attribution=use_attribution,
                    )
                    val_samples = _collect_cached_samples(
                        loader=loader,
                        idx_list=val_indices,
                        source_layer=int(source_layer),
                        target_layer=target_layer,
                        use_attention=use_attention,
                        use_attribution=use_attribution,
                    )
                    test_samples = _collect_cached_samples(
                        loader=loader,
                        idx_list=test_indices,
                        source_layer=int(source_layer),
                        target_layer=target_layer,
                        use_attention=use_attention,
                        use_attribution=use_attribution,
                    )
                else:
                    all_samples = list(fallback_samples or [])
                    train_samples = all_samples
                    val_samples = all_samples
                    test_samples = all_samples

                train_pairs = _build_pairs_for_mode(
                    samples=train_samples,
                    source_layer=int(source_layer),
                    target_layer=target_layer,
                    routing_policy=routing_policy,
                    cfg=cfg,
                )
                val_pairs = _build_pairs_for_mode(
                    samples=val_samples,
                    source_layer=int(source_layer),
                    target_layer=target_layer,
                    routing_policy=routing_policy,
                    cfg=cfg,
                )
                test_pairs = _build_pairs_for_mode(
                    samples=test_samples,
                    source_layer=int(source_layer),
                    target_layer=target_layer,
                    routing_policy=routing_policy,
                    cfg=cfg,
                )
                X, Y = train_pairs.X, train_pairs.Y
                route_summary = summarize_routes(train_pairs.routes)

            config = TransportOperatorConfig(
                ridge_lambda=cfg.get("ridge_lambda", 1e-2),
                rank=cfg.get("rank"),
                name="routing_aware_operator" if routing_enabled else "same_token_baseline",
                compute_backend=cfg.get("compute_backend", "numpy"),
                device=cfg.get("device", "cpu"),
            )

            if routing_enabled:
                operator = RoutedTransportOperator(
                    config=config,
                    routing_policy_name=routing_policy.name,
                    route_summary=route_summary,
                ).fit_X_y(X, Y)
                metrics_payload = {
                    "train_metrics": operator.train_metrics,
                }
                if val_pairs is not None:
                    metrics_payload["val_metrics"] = _evaluate_xy(operator, val_pairs.X, val_pairs.Y)
                if test_pairs is not None:
                    metrics_payload["test_metrics"] = _evaluate_xy(operator, test_pairs.X, test_pairs.Y)
            else:
                operator = TransportOperator(config=config).fit_X_y(X, Y)
                metrics_payload = {
                    "train_metrics": operator.train_metrics,
                }
                if val_pairs is not None:
                    metrics_payload["val_metrics"] = _evaluate_xy(operator, val_pairs[0], val_pairs[1])
                if test_pairs is not None:
                    metrics_payload["test_metrics"] = _evaluate_xy(operator, test_pairs[0], test_pairs[1])

            out_dir = Path(cfg.get("output_dir", "outputs/train_transport_operators")) / f"L{source_layer}_k{k}"
            out_dir.mkdir(parents=True, exist_ok=True)
            operator.save(out_dir / "operator.npz")
            metadata = {
                **operator.metadata(),
                **metrics_payload,
                "source_layer": int(source_layer),
                "target_layer": int(target_layer),
                "routing_enabled": routing_enabled,
                "routing_policy": routing_policy.name if routing_enabled else "same_token",
                "input_mode": cfg.get("routing", {}).get("input_mode", cfg.get("input_mode", "weighted_sum")),
                "train_shape": [int(X.shape[0]), int(X.shape[1])],
                "route_summary": route_summary,
                "use_dataset_splits": bool(cfg.get("use_dataset_splits", False)),
            }
            save_json(out_dir / "metadata.json", metadata)
            results.append(metadata)
            print(
                f"Saved transport operator for L={source_layer}, k={k} "
                f"policy={metadata['routing_policy']} -> {out_dir}"
            )

    if loader is not None:
        loader.close()

    summary_path = Path(cfg.get("output_dir", "outputs/train_transport_operators")) / "summary.json"
    save_json(summary_path, {"runs": results})
    print(f"Saved training summary -> {summary_path}")


if __name__ == "__main__":
    main()
