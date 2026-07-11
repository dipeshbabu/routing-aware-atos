from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing_aware_atos.provenance import (
    collection_implementation_sha256,
    feature_selection_implementation_sha256,
    sha256_file,
)


def _load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_selected_feature_ids(path: Path) -> list[int] | None:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if "selected_ids" not in arrays:
                return None
            return arrays["selected_ids"].astype(int).tolist()
    except (OSError, ValueError):
        return None


def _sae_arrays_match_protocol(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            return bool(
                "decoder" in arrays
                and "encoder" in arrays
                and "threshold" in arrays
                and "architecture" in arrays
                and "activation_fn" in arrays
                and arrays["decoder"].shape == (16_384, 2_304)
                and arrays["encoder"].shape == (2_304, 16_384)
                and str(arrays["architecture"].item()) == "jumprelu"
                and str(arrays["activation_fn"].item()) == "relu"
            )
    except (OSError, ValueError):
        return False


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _collection_config_sha256(cfg: dict) -> str:
    payload = {
        key: cfg.get(key)
        for key in (
            "model_name",
            "model_revision",
            "device",
            "dtype",
            "attn_implementation",
            "dataset",
            "collection",
        )
    }
    payload["implementation_sha256"] = collection_implementation_sha256()
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run(arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _preflight(config_path: str, stage: str) -> None:
    _run(["scripts/preflight_real_experiment.py", "--config", config_path, "--stage", stage])


def _collect(config_path: str, cfg: dict, force: bool) -> None:
    manifest = Path(cfg["collection"]["output_dir"]) / "manifest.json"
    if manifest.exists():
        payload = _load_json_dict(manifest)
        if payload.get("config_sha256") == _collection_config_sha256(cfg):
            _preflight(config_path, "cache")
            print(f"SKIP: current activation cache already exists -> {manifest}")
            return
        raise RuntimeError(
            f"Activation cache {manifest} belongs to a different collection configuration; "
            "use a new collection.output_dir"
        )
    _preflight(config_path, "collect")
    _run(["scripts/collect_hf_activations.py", "--config", config_path])


def _export_saes(config_path: str, cfg: dict, force: bool) -> None:
    _preflight(config_path, "sae")
    for target in cfg["sae"]["targets"]:
        output = Path(target["output"])
        if output.exists() and not force:
            metadata_path = output.with_suffix(".json")
            metadata = _load_json_dict(metadata_path)
            if (
                metadata.get("release") == str(cfg["sae"]["release"])
                and metadata.get("sae_id") == str(target["sae_id"])
                and metadata.get("architecture") == "jumprelu"
                and metadata.get("activation_fn") == "relu"
                and metadata.get("artifact_sha256")
                == sha256_file(output)
                and _sae_arrays_match_protocol(output)
            ):
                print(f"SKIP: current SAE artifact already exists -> {output}")
                continue
            print(f"Re-exporting stale SAE artifact -> {output}")
        _run(
            [
                "scripts/export_gemma_scope_decoder.py",
                "--release",
                str(cfg["sae"]["release"]),
                "--sae-id",
                str(target["sae_id"]),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )


def _select_features(config_path: str, cfg: dict, force: bool) -> None:
    _preflight(config_path, "features")
    feature_cfg = cfg["feature_selection"]
    expected_method = {
        "reference_full": "reference_style_with_live_causal_ablation",
        "fast_proxy": "activity_token_entropy_decoder_uniqueness",
    }[str(feature_cfg.get("method", "fast_proxy"))]
    selection_fingerprint_cfg = dict(feature_cfg)
    for key, fallback in (
        ("model_name", cfg.get("model_name")),
        ("model_revision", cfg.get("model_revision")),
        ("model_dtype", cfg.get("dtype", "float32")),
        ("attn_implementation", cfg.get("attn_implementation", "eager")),
    ):
        selection_fingerprint_cfg.setdefault(key, fallback)
    selection_config_sha256 = hashlib.sha256(
        json.dumps(
            selection_fingerprint_cfg,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    output_dir = Path(feature_cfg.get("output_dir", "artifacts/features"))
    outputs = [
        output_dir / f"layer_{int(layer)}_features.json"
        for layer in feature_cfg["target_layers"]
    ]
    stats_outputs = [
        output_dir / f"layer_{int(layer)}_feature_stats.npz"
        for layer in feature_cfg["target_layers"]
    ]
    if (
        not force
        and outputs
        and all(path.exists() for path in outputs)
        and all(path.exists() for path in stats_outputs)
    ):
        payloads = [_load_json_dict(path) for path in outputs]
        manifest_path = Path(cfg["collection"]["output_dir"]) / "manifest.json"
        activation_hash = _load_json_dict(manifest_path).get("config_sha256")
        selected_ids = [_load_selected_feature_ids(path) for path in stats_outputs]
        if all(
            payload.get("method") == expected_method
            and int(payload.get("tokens_processed", -1))
            == int(feature_cfg.get("max_tokens", 120_000))
            and payload.get("model_revision") == cfg.get("model_revision")
            and payload.get("activation_config_sha256") == activation_hash
            and payload.get("selection_config_sha256") == selection_config_sha256
            and payload.get("implementation_sha256")
            == feature_selection_implementation_sha256()
            and payload.get("sae_artifact_sha256")
            == sha256_file(
                Path(
                    cfg["sae"]["artifacts"].get(
                        int(layer), cfg["sae"]["artifacts"].get(str(layer))
                    )
                )
            )
            and stats_ids == payload.get("high_quality_feature_ids")
            for payload, stats_ids, layer in zip(
                payloads,
                selected_ids,
                feature_cfg["target_layers"],
            )
        ):
            print(f"SKIP: current feature artifacts already exist -> {output_dir}")
            return
    command = ["scripts/select_sae_features.py", "--config", config_path]
    if force:
        command.append("--force")
    _run(command)


def _experiments(
    config_path: str,
    *,
    policies: list[str],
    pairs: list[str],
    force: bool,
) -> None:
    _preflight(config_path, "experiments")
    command = ["scripts/run_real_experiments.py", "--config", config_path]
    for policy in policies:
        command.extend(["--policy", policy])
    for pair in pairs:
        command.extend(["--pair", pair])
    if force:
        command.append("--force")
    _run(command)


def _causal(config_path: str, cfg: dict, force: bool) -> None:
    _preflight(config_path, "causal")
    experiment_dir = Path(cfg["experiments"].get("output_dir", "outputs/real"))
    for run in cfg["live_causal"]["runs"]:
        source = int(run["source_layer"])
        target = int(run["target_layer"])
        policy = str(run["policy"])
        operator = experiment_dir / "runs" / f"L{source}_to_L{target}" / policy / "operator.npz"
        output = experiment_dir / "live_causal" / f"L{source}_to_L{target}" / f"{policy}.json"
        command = [
                "scripts/run_live_causal_restore.py",
                "--config",
                config_path,
                "--operator-path",
                str(operator),
                "--source-layer",
                str(source),
                "--target-layer",
                str(target),
                "--policy",
                policy,
                "--output-path",
                str(output),
            ]
        if force:
            command.append("--force")
        _run(command)


def _report(config_path: str) -> None:
    _preflight(config_path, "report")
    _run(["scripts/make_real_report.py", "--config", config_path])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete real Gemma 2 / SlimPajama workflow")
    parser.add_argument("--config", default="configs/real/gemma2_2b_slimpajama.yaml")
    parser.add_argument(
        "--stage",
        choices=["all", "collect", "sae", "features", "experiments", "causal", "report", "preflight"],
        default="all",
    )
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = _load_config(args.config)

    if args.stage == "preflight":
        _preflight(args.config, "all")
        return
    stages = ["collect", "sae", "features", "experiments", "causal", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "collect":
            _collect(args.config, cfg, args.force)
        elif stage == "sae":
            _export_saes(args.config, cfg, args.force)
        elif stage == "features":
            _select_features(args.config, cfg, args.force)
        elif stage == "experiments":
            _experiments(
                args.config,
                policies=args.policy,
                pairs=args.pair,
                force=args.force,
            )
        elif stage == "causal":
            _causal(args.config, cfg, args.force)
        elif stage == "report":
            _report(args.config)


if __name__ == "__main__":
    main()
