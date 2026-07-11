from pathlib import Path

import numpy as np
import pytest

from routing_aware_atos.provenance import (
    live_causal_run_provenance,
    predictive_run_provenance,
    sha256_file,
    sha256_payload,
)
from routing_aware_atos.utils.io import save_json, save_npz
from scripts.make_real_report import build_real_report


def test_real_report_validates_and_writes_tables_and_plots(tmp_path: Path):
    output_dir = tmp_path / "real"
    feature_dir = tmp_path / "features"
    sae_path = tmp_path / "sae.npz"
    save_npz(sae_path, marker=np.asarray([1], dtype=np.int8))
    save_json(feature_dir / "layer_10_features.json", {"high_quality_feature_ids": [1]})
    pair = {"source_layer": 9, "target_layer": 10, "leap": 1}
    policy = {"name": "same_token"}
    run = {"source_layer": 9, "target_layer": 10, "policy": "same_token"}
    cfg = {
        "protocol": "test",
        "sae": {"artifacts": {10: str(sae_path)}},
        "feature_selection": {"output_dir": str(feature_dir)},
        "experiments": {
            "activation_dir_path": str(tmp_path / "cache"),
            "output_dir": str(output_dir),
            "layer_pairs": [pair],
            "policies": [policy],
        },
        "live_causal": {
            "position_counts": [1, 5, "all"],
            "runs": [run],
        },
        "report": {"output_dir": str(output_dir / "report")},
    }
    result_path = output_dir / "runs" / "L9_to_L10" / "same_token" / "result.json"
    operator_path = result_path.with_name("operator.npz")
    save_npz(operator_path, marker=np.asarray([2], dtype=np.int8))
    predictive_provenance = predictive_run_provenance(cfg, pair, policy)
    save_json(
        result_path,
        {
            "source_layer": 9,
            "target_layer": 10,
            "leap": 1,
            "policy_name": "same_token",
            "selected_ridge_lambda": 10.0,
            "residual_metrics": {"r2": 0.8},
            "feature_summary": {"mean_r2": 0.7, "median_r2": 0.75, "num_features": 12},
            "run_fingerprint": sha256_payload(predictive_provenance),
            "provenance": predictive_provenance,
            "operator_sha256": sha256_file(operator_path),
            "transport_efficiency": {
                "ranks": [
                    {"rank": 1, "efficiency": 0.2},
                    {"rank": 2, "efficiency": 0.5},
                ]
            },
        },
    )
    causal_path = output_dir / "live_causal" / "L9_to_L10" / "same_token.json"
    summary = {
        "clean_cross_entropy": 2.0,
        "ablated_cross_entropy": 3.0,
        "restored_cross_entropy": 2.2,
        "null_cross_entropy": 2.9,
        "kl_restoration": 0.7,
        "logit_mse_restoration": 0.8,
    }
    causal_provenance = live_causal_run_provenance(
        cfg,
        operator_path=operator_path,
        source_layer=9,
        target_layer=10,
        policy="same_token",
    )
    save_json(
        causal_path,
        {
            "source_layer": 9,
            "target_layer": 10,
            "routing_policy": "same_token",
            "run_fingerprint": sha256_payload(causal_provenance),
            "provenance": causal_provenance,
            "results": {mode: {"summary": summary} for mode in ("1", "5", "all")},
        },
    )

    report = build_real_report(cfg)

    assert report["complete"] is True
    assert report["predictive_run_count"] == 1
    assert report["causal_run_count"] == 1
    assert (output_dir / "report" / "report.json").exists()
    assert (output_dir / "report" / "predictive_results.csv").exists()
    assert (output_dir / "report" / "feature_r2_by_leap.png").exists()
    assert (output_dir / "report" / "causal_log_perplexity_by_leap.png").exists()

    save_npz(operator_path, marker=np.asarray([99], dtype=np.int8))
    with pytest.raises(ValueError, match="invalid result"):
        build_real_report(cfg)
