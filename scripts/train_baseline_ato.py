
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.data.baseline_pairs import SameTokenBaselineBuilder
from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig
from routing_aware_atos.utils.io import load_cached_samples, load_npz, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a same-token baseline transport operator.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if cfg.get("pairs_path"):
        arrays = load_npz(cfg["pairs_path"])
        X, Y = arrays["X"], arrays["Y"]
        pair_metadata = {"pairs_path": cfg["pairs_path"]}
    else:
        if cfg.get("activation_dir_path"):
            loader = ActivationLoader(activation_dir_path=cfg["activation_dir_path"])
            samples = list(
                loader.iter_cached_samples(
                    idx_list=cfg.get("idx_list"),
                    layer_indices=[cfg["source_layer"], cfg["target_layer"]],
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
        builder = SameTokenBaselineBuilder(
            samples=samples,
            source_layer=cfg["source_layer"],
            target_layer=cfg["target_layer"],
            include_positions=cfg.get("include_positions"),
        )
        X, Y, _ = builder.build_pairs()
        pair_metadata = {
            "source_layer": cfg["source_layer"],
            "target_layer": cfg["target_layer"],
            "num_rows": int(X.shape[0]),
        }
        if cfg.get("activation_dir_path"):
            pair_metadata["activation_dir_path"] = cfg["activation_dir_path"]
        if cfg.get("cache_path"):
            pair_metadata["cache_path"] = cfg["cache_path"]

    operator = TransportOperator(
        config=TransportOperatorConfig(
            ridge_lambda=cfg.get("ridge_lambda", 1e-2),
            rank=cfg.get("rank"),
            name="same_token_baseline",
        )
    ).fit(X, Y)

    out_dir = Path(cfg.get("output_dir", "outputs/train_baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)
    operator.save(out_dir / "operator.npz")
    save_json(out_dir / "metadata.json", {**operator.metadata(), **pair_metadata, "train_shape": [int(X.shape[0]), int(X.shape[1])], "target_shape": [int(Y.shape[0]), int(Y.shape[1])]})
    print(f"Saved baseline operator -> {out_dir}")
    print(operator.train_metrics)


if __name__ == "__main__":
    main()
