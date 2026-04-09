
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.data.routed_dataset import RoutedActivationDataset, summarize_routes
from routing_aware_atos.models.routed_transport_operator import RoutedTransportOperator
from routing_aware_atos.models.transport_operator import TransportOperatorConfig
from routing_aware_atos.routing.factory import build_routing_policy
from routing_aware_atos.utils.io import load_cached_samples, load_npz, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a routing-aware transport operator.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    route_summary = {}
    if cfg.get("pairs_path"):
        arrays = load_npz(cfg["pairs_path"])
        X, Y = arrays["X"], arrays["Y"]
    else:
        policy = build_routing_policy(
            cfg["routing_policy"],
            top_k=cfg.get("top_k", 1),
            normalize_weights=cfg.get("normalize_weights", True),
            exclude_self=cfg.get("exclude_self", False),
            allow_negative_scores=cfg.get("allow_negative_scores", False),
        )
        if cfg.get("activation_dir_path"):
            loader = ActivationLoader(activation_dir_path=cfg["activation_dir_path"])
            samples = list(
                loader.iter_cached_samples(
                    idx_list=cfg.get("idx_list"),
                    layer_indices=[cfg["source_layer"], cfg["target_layer"]],
                    attention_layer_pairs=[(cfg["source_layer"], cfg["target_layer"])] if policy.requires_attention else None,
                    attribution_layer_pairs=[(cfg["source_layer"], cfg["target_layer"])] if policy.requires_attribution else None,
                )
            )
        elif cfg.get("cache_path"):
            samples = load_cached_samples(cfg["cache_path"])
        else:
            samples = make_mock_samples(
                num_samples=cfg.get("num_samples", 2),
                seq_len=cfg.get("seq_len", 6),
                d_model=cfg.get("d_model", 4),
            )

        dataset = RoutedActivationDataset(
            samples=samples,
            source_layer=cfg["source_layer"],
            target_layer=cfg["target_layer"],
            routing_policy=policy,
            include_positions=cfg.get("include_positions"),
        )
        pairs = dataset.build_pairs()
        X, Y = pairs.X, pairs.Y
        route_summary = summarize_routes(pairs.routes)

    operator = RoutedTransportOperator(
        config=TransportOperatorConfig(
            ridge_lambda=cfg.get("ridge_lambda", 1e-2),
            rank=cfg.get("rank"),
            name="routing_aware_operator",
        ),
        routing_policy_name=cfg.get("routing_policy", "from_pairs"),
        route_summary=route_summary,
    ).fit(X, Y)

    out_dir = Path(cfg.get("output_dir", f"outputs/train_{cfg.get('routing_policy', 'routed')}"))
    operator.save_bundle(
        out_dir,
        extra_metadata={
            "train_shape": [int(X.shape[0]), int(X.shape[1])],
            "target_shape": [int(Y.shape[0]), int(Y.shape[1])],
            "source_layer": cfg.get("source_layer"),
            "target_layer": cfg.get("target_layer"),
            "routing_policy": cfg.get("routing_policy"),
            "activation_dir_path": cfg.get("activation_dir_path"),
            "cache_path": cfg.get("cache_path"),
        },
    )
    print(f"Saved routed operator -> {out_dir}")
    print(operator.train_metrics)


if __name__ == "__main__":
    main()
