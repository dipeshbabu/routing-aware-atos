import json
import subprocess
import sys
from pathlib import Path


def test_train_baseline_script_runs(tmp_path: Path):
    config_path = tmp_path / "train_baseline.yaml"
    output_dir = tmp_path / "baseline_out"
    config_path.write_text(
        "\n".join([
            "source_layer: 10",
            "target_layer: 12",
            f"output_dir: {output_dir.as_posix()}",
            "num_samples: 2",
            "seq_len: 5",
            "d_model: 3",
            "ridge_lambda: 0.01",
        ]),
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "scripts/train_baseline_ato.py", "--config", str(config_path)], check=True)
    assert (output_dir / "operator.npz").exists()
    assert (output_dir / "metadata.json").exists()


def test_train_routed_script_runs(tmp_path: Path):
    config_path = tmp_path / "train_routed.yaml"
    output_dir = tmp_path / "routed_out"
    config_path.write_text(
        "\n".join([
            "source_layer: 10",
            "target_layer: 12",
            "routing_policy: attention_topk",
            "top_k: 2",
            f"output_dir: {output_dir.as_posix()}",
            "num_samples: 2",
            "seq_len: 5",
            "d_model: 3",
            "ridge_lambda: 0.01",
            "rank: 2",
        ]),
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "scripts/train_ra_atos.py", "--config", str(config_path)], check=True)
    assert (output_dir / "operator.npz").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["routing_policy"] == "attention_topk"


def test_train_transport_operators_script_runs(tmp_path: Path):
    config_path = tmp_path / "train_transport_ops.yaml"
    output_dir = tmp_path / "transport_ops_out"
    config_path.write_text(
        "\n".join([
            f"output_dir: {output_dir.as_posix()}",
            "num_samples: 2",
            "seq_len: 5",
            "d_model: 3",
            "ridge_lambda: 0.01",
            "experiment:",
            "  L: [10]",
            "  k: [2]",
            "routing:",
            "  enabled: true",
            "  policy: attention_topk",
            "  top_k: 2",
            "  normalize_weights: true",
            "  exclude_self: false",
            "  allow_negative_scores: false",
        ]),
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "scripts/train_transport_operators.py", "--config", str(config_path)], check=True)
    run_dir = output_dir / "L10_k2"
    assert (run_dir / "operator.npz").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["routing_policy"] == "attention_topk"
