from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.evaluation.transport_taxonomy import (
    build_transport_taxonomy,
    load_feature_eval_payload,
    save_transport_taxonomy,
)
from routing_aware_atos.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature transport taxonomy from saved feature-eval payloads")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_payloads = []
    for run in cfg["runs"]:
        payload = load_feature_eval_payload(run["feature_eval_path"])
        payload["policy_name"] = run["policy_name"]
        run_payloads.append(payload)

    taxonomy = build_transport_taxonomy(
        run_payloads,
        same_token_policy=cfg.get("same_token_policy", "same_token"),
        cross_token_policies=cfg.get("cross_token_policies", ["attention_top1"]),
        multi_source_policies=cfg.get("multi_source_policies", ["attention_topk", "attribution_topk"]),
        threshold=float(cfg.get("threshold", 0.7)),
    )
    save_transport_taxonomy(taxonomy, cfg["output_path"])
    print(f"Saved transport taxonomy to {cfg['output_path']}")
    print(taxonomy["summary_rows"])


if __name__ == "__main__":
    main()
