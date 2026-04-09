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


def _resolve_feature_eval_path(results_dir: Path, policy_name: str) -> Path:
    candidates = [
        results_dir / f"feature_metrics_{policy_name}.json",
        results_dir / f"{policy_name}_feature_eval.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing feature evaluation payload for policy={policy_name}. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def _load_run_payloads(cfg: dict) -> list[dict]:
    if cfg.get("runs"):
        run_payloads = []
        for run in cfg["runs"]:
            payload = load_feature_eval_payload(run["feature_eval_path"])
            payload["policy_name"] = run["policy_name"]
            run_payloads.append(payload)
        return run_payloads

    results_dir = cfg.get("results_dir")
    policies = cfg.get("taxonomy_policies")
    if not results_dir or not policies:
        raise ValueError(
            "Config must define either `runs` or both `results_dir` and "
            "`taxonomy_policies`."
        )

    eval_dir = Path(results_dir)
    run_payloads = []
    for policy_name in policies:
        payload = load_feature_eval_payload(
            _resolve_feature_eval_path(eval_dir, str(policy_name))
        )
        payload["policy_name"] = policy_name
        run_payloads.append(payload)
    return run_payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature transport taxonomy from saved feature-eval payloads")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_payloads = _load_run_payloads(cfg)
    taxonomy_cfg = cfg.get("taxonomy", {})

    taxonomy = build_transport_taxonomy(
        run_payloads,
        same_token_policy=taxonomy_cfg.get(
            "same_token_policy",
            cfg.get("same_token_policy", "same_token"),
        ),
        cross_token_policies=taxonomy_cfg.get(
            "cross_token_policies",
            cfg.get("cross_token_policies", ["attention_top1"]),
        ),
        multi_source_policies=taxonomy_cfg.get(
            "multi_source_policies",
            cfg.get("multi_source_policies", ["attention_topk", "attribution_topk"]),
        ),
        threshold=float(taxonomy_cfg.get("threshold", cfg.get("threshold", 0.7))),
    )
    output_path = cfg.get("output_path")
    if output_path is None:
        results_dir = cfg.get("results_dir")
        if not results_dir:
            raise ValueError("Config must define `output_path` or `results_dir`.")
        output_path = str(Path(results_dir) / "transport_taxonomy.json")

    save_transport_taxonomy(taxonomy, output_path)
    print(f"Saved transport taxonomy to {output_path}")
    print(taxonomy["summary_rows"])


if __name__ == "__main__":
    main()
