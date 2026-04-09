from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.evaluation.feature_eval import (
    evaluate_operator_from_cached_samples,
    evaluate_operator_in_feature_space,
)
from routing_aware_atos.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a trained operator in downstream feature space')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_path = cfg.get('output_path')
    if cfg.get("pairs_path"):
        payload = evaluate_operator_in_feature_space(
            operator_path=cfg["operator_path"],
            pairs_path=cfg["pairs_path"],
            decoder_path=cfg["decoder_path"],
            feature_ids=cfg.get("feature_ids"),
            output_path=output_path,
            split_name=cfg.get("split_name", "eval"),
        )
    else:
        routing_cfg = cfg.get("routing", {})
        payload = evaluate_operator_from_cached_samples(
            operator_path=cfg["operator_path"],
            decoder_path=cfg["decoder_path"],
            activation_dir_path=cfg.get("activation_dir_path"),
            cache_path=cfg.get("cache_path"),
            source_layer=cfg["source_layer"],
            target_layer=cfg["target_layer"],
            routing_policy=routing_cfg.get("policy", "same_token"),
            top_k=routing_cfg.get("top_k", 1),
            normalize_weights=routing_cfg.get("normalize_weights", True),
            exclude_self=routing_cfg.get("exclude_self", False),
            allow_negative_scores=routing_cfg.get("allow_negative_scores", False),
            include_positions=cfg.get("include_positions"),
            feature_ids=cfg.get("feature_ids"),
            output_path=output_path,
            split_name=cfg.get("split_name", "eval"),
            num_samples=cfg.get("num_samples", 2),
            seq_len=cfg.get("seq_len", 6),
            d_model=cfg.get("d_model", 4),
        )
    print(payload['feature_summary'])


if __name__ == '__main__':
    main()
