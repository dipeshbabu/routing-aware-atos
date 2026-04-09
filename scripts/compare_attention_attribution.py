from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.evaluation.route_agreement import compare_routing_policies, summarize_route_metadata
from routing_aware_atos.routing.attribution import AttributionTopKPolicy
from routing_aware_atos.routing.attention import AttentionTop1Policy, AttentionTopKPolicy
from routing_aware_atos.routing.base import RoutingPolicyConfig
from routing_aware_atos.utils.io import load_cached_samples, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare attention-based and attribution-based routes")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    samples = load_cached_samples(cfg["samples_path"])

    top_k = int(cfg.get("top_k", 3))
    exclude_self = bool(cfg.get("exclude_self", False))
    policy_cfg = RoutingPolicyConfig(top_k=top_k, exclude_self=exclude_self)

    policies = [
        ("attention_top1", AttentionTop1Policy(RoutingPolicyConfig(top_k=1, exclude_self=exclude_self))),
        ("attention_topk", AttentionTopKPolicy(policy_cfg)),
        ("attribution_topk", AttributionTopKPolicy(policy_cfg)),
    ]
    payload = compare_routing_policies(
        samples,
        policies,
        source_layer=int(cfg["source_layer"]),
        target_layer=int(cfg["target_layer"]),
    )
    payload["route_metadata_summary"] = summarize_route_metadata(payload)
    save_json(cfg["output_path"], payload)
    print(f"Saved route comparison to {cfg['output_path']}")


if __name__ == "__main__":
    main()
