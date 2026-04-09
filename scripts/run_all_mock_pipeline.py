from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import json
import subprocess

from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.data.attribution_cache import AttributionBuildConfig, attach_attribution_scores
from routing_aware_atos.data.baseline_pairs import build_same_token_pairs
from routing_aware_atos.data.routed_dataset import build_routed_pairs
from routing_aware_atos.routing.same_token import SameTokenPolicy
from routing_aware_atos.routing.attention import AttentionTop1Policy, AttentionTopKPolicy
from routing_aware_atos.routing.attribution import AttributionTopKPolicy
from routing_aware_atos.routing.base import RoutingPolicyConfig
from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig
from routing_aware_atos.evaluation.feature_eval import evaluate_operator_in_feature_space
from routing_aware_atos.evaluation.causal_restore import compare_causal_policy_runs
from routing_aware_atos.evaluation.transport_taxonomy import build_transport_taxonomy
from routing_aware_atos.utils.io import save_cached_samples, save_json, save_npz
import numpy as np


def main() -> None:
    out = PROJECT_ROOT / "outputs"
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "evals").mkdir(parents=True, exist_ok=True)
    (out / "sae").mkdir(parents=True, exist_ok=True)
    (out / "readout").mkdir(parents=True, exist_ok=True)
    (out / "benchmarks").mkdir(parents=True, exist_ok=True)
    (out / "causal_benchmarks").mkdir(parents=True, exist_ok=True)
    (out / "taxonomy").mkdir(parents=True, exist_ok=True)
    (out / "paper_figures").mkdir(parents=True, exist_ok=True)

    samples = make_mock_samples(num_samples=4, seq_len=6, d_model=6)
    samples = attach_attribution_scores(
        samples, [(10, 12)],
        config=AttributionBuildConfig(methods=["attention_value", "residual_similarity", "attention_similarity_mix"]),
    )
    save_cached_samples(out / "mock_samples_with_attr.json", samples)

    # baseline pairs
    Xb, Yb, _ = build_same_token_pairs(samples, 10, 12)
    save_npz(out / "pairs" / "baseline_pairs.npz", X=Xb, Y=Yb)

    policies = {
        "same_token": SameTokenPolicy(RoutingPolicyConfig(top_k=1)),
        "attention_top1": AttentionTop1Policy(RoutingPolicyConfig(top_k=1)),
        "attention_topk": AttentionTopKPolicy(RoutingPolicyConfig(top_k=3)),
        "attribution_topk": AttributionTopKPolicy(RoutingPolicyConfig(top_k=3)),
    }

    # routed pair files
    for name, policy in policies.items():
        routed = build_routed_pairs(samples, 10, 12, policy)
        save_npz(out / "pairs" / f"{name}_pairs.npz", X=routed.X, Y=routed.Y)

    rng = np.random.default_rng(0)
    decoder = rng.normal(size=(8, 6)).astype(np.float32)
    readout = rng.normal(size=(6, 12)).astype(np.float32)
    save_npz(out / "sae" / "mock_decoder.npz", decoder=decoder)
    save_npz(out / "readout" / "mock_readout.npz", readout=readout)

    # train operators
    eval_payloads = []
    causal_runs = []
    for name in ["same_token", "attention_top1", "attention_topk", "attribution_topk"]:
        pairs_path = out / "pairs" / f"{name}_pairs.npz"
        data = np.load(pairs_path)
        X = data["X"]
        Y = data["Y"]

        op = TransportOperator(TransportOperatorConfig(ridge_lambda=1e-3, rank=None, name=name)).fit(X, Y)
        op_path = out / "models" / f"{name}_operator.npz"
        op.save(op_path)

        eval_payload = evaluate_operator_in_feature_space(
            operator_path=op_path,
            pairs_path=pairs_path,
            decoder_path=out / "sae" / "mock_decoder.npz",
            output_path=out / "evals" / f"{name}_feature_eval.json",
        )
        eval_payload["policy_name"] = name
        eval_payloads.append(eval_payload)

        causal_runs.append({
            "policy_name": name,
            "operator_path": op_path,
            "pairs_path": pairs_path,
            "decoder_path": out / "sae" / "mock_decoder.npz",
            "readout_path": out / "readout" / "mock_readout.npz",
            "rank": None,
        })

    # benchmark payloads
    feature_benchmark = {
        "summary_rows": [
            {
                "policy_name": p["policy_name"],
                "rank": None,
                "mean_r2": p["feature_summary"]["mean_r2"],
                "median_r2": p["feature_summary"]["median_r2"],
                "mean_corr": p["feature_summary"]["mean_corr"],
                "residual_r2": p["residual_metrics"]["r2"],
            }
            for p in eval_payloads
        ]
    }
    save_json(out / "benchmarks" / "comparison.json", feature_benchmark)

    causal_payload = compare_causal_policy_runs(causal_runs)
    save_json(out / "causal_benchmarks" / "causal_comparison.json", causal_payload)

    taxonomy = build_transport_taxonomy(
        eval_payloads,
        same_token_policy="same_token",
        cross_token_policies=("attention_top1",),
        multi_source_policies=("attention_topk", "attribution_topk"),
        threshold=0.5,
    )
    save_json(out / "taxonomy" / "transport_taxonomy.json", taxonomy)

    print("Mock pipeline completed successfully.")


if __name__ == "__main__":
    main()
