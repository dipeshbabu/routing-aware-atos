from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from routing_aware_atos.data.activation_store import SPLIT_IDS, ActivationShardWriter, ActivationStoreSpec
from routing_aware_atos.utils.io import save_json, save_npz


def test_real_experiment_runner_uses_manifest_splits(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    spec = ActivationStoreSpec(
        num_samples=3,
        sequence_length=4,
        d_model=2,
        layer_indices=(1, 2),
        attention_layers=(2,),
        cache_dtype="float32",
    )
    rng = np.random.default_rng(2)
    source = rng.normal(size=(3, 4, 2)).astype(np.float32)
    target = (source @ np.asarray([[1.1, 0.2], [-0.1, 0.9]], dtype=np.float32) + 0.05).astype(np.float32)
    attention = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
    with ActivationShardWriter(cache_dir / "part-00000.zip", spec) as writer:
        writer.write_batch(
            0,
            input_ids=np.arange(12, dtype=np.int32).reshape(3, 4),
            attention_mask=np.ones((3, 4), dtype=np.uint8),
            split_ids=np.asarray([SPLIT_IDS["train"], SPLIT_IDS["validation"], SPLIT_IDS["test"]]),
            sequence_indices=np.arange(3),
            residuals={1: source, 2: target},
            attention_scores={2: attention},
        )

    sae_path = tmp_path / "sae.npz"
    decoder = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    save_npz(sae_path, decoder=decoder, encoder=decoder.T)
    feature_dir = tmp_path / "features"
    save_json(feature_dir / "layer_2_features.json", {"high_quality_feature_ids": [0, 1, 2]})
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "real.yaml"
    config_path.write_text(
        "\n".join(
            [
                "sae:",
                "  artifacts:",
                f"    2: {sae_path.as_posix()}",
                "feature_selection:",
                f"  output_dir: {feature_dir.as_posix()}",
                "experiments:",
                f"  activation_dir_path: {cache_dir.as_posix()}",
                f"  output_dir: {output_dir.as_posix()}",
                "  compute_backend: numpy",
                "  device: cpu",
                "  ridge_selection: five_fold_cv",
                "  cv_folds: 2",
                "  ridge_lambdas: [0.01]",
                "  activated_only: false",
                "  min_feature_activations: 1",
                "  metric_batch_size: 16",
                "  bootstrap_samples: 10",
                "  layer_pairs:",
                "    - {source_layer: 1, target_layer: 2, leap: 1}",
                "  policies:",
                "    - {name: same_token, top_k: 1, input_mode: weighted_sum}",
                "    - {name: attention_top1, top_k: 1, input_mode: weighted_sum}",
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/run_real_experiments.py", "--config", str(config_path)],
        check=True,
    )

    summary = json.loads((output_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    assert len(summary["runs"]) == 2
    assert {row["policy_name"] for row in summary["runs"]} == {"same_token", "attention_top1"}
    result = json.loads(
        (output_dir / "runs" / "L1_to_L2" / "same_token" / "result.json").read_text(encoding="utf-8")
    )
    assert result["train_rows"] == 4
    assert result["validation_rows"] == 4
    assert result["test_rows"] == 4
