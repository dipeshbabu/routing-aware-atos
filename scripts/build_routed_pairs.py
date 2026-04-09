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
from routing_aware_atos.data.routed_dataset import RoutedActivationDataset
from routing_aware_atos.routing.factory import build_routing_policy
from routing_aware_atos.utils.io import load_cached_samples, load_yaml, save_json, save_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Build routed training pairs for RA-ATO.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
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

    out_dir = Path(cfg.get("output_dir", f"outputs/{policy.name}"))
    save_npz(out_dir / "pairs.npz", X=pairs.X, Y=pairs.Y)
    save_json(out_dir / "routes.json", pairs.routes)

    print(f"Saved routed pairs with policy={policy.name}: X{pairs.X.shape}, Y{pairs.Y.shape} -> {out_dir}")


if __name__ == "__main__":
    main()
