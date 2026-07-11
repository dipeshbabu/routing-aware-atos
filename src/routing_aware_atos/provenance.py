from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from routing_aware_atos.utils.io import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _implementation_sha256(relative_paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    tracked_paths = set(relative_paths) | {"pyproject.toml", "uv.lock"}
    for relative_path in sorted(tracked_paths):
        path = PROJECT_ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def collection_implementation_sha256() -> str:
    return _implementation_sha256(
        (
            "scripts/collect_hf_activations.py",
            "src/routing_aware_atos/data/activation_store.py",
            "src/routing_aware_atos/data/slimpajama.py",
        )
    )


def feature_selection_implementation_sha256() -> str:
    return _implementation_sha256(
        (
            "scripts/select_sae_features.py",
            "src/routing_aware_atos/activation_loader.py",
            "src/routing_aware_atos/sae/encoding.py",
            "src/routing_aware_atos/sae/feature_selection.py",
        )
    )


def predictive_implementation_sha256() -> str:
    return _implementation_sha256(
        (
            "scripts/run_real_experiments.py",
            "src/routing_aware_atos/activation_loader.py",
            "src/routing_aware_atos/evaluation/causal_restore.py",
            "src/routing_aware_atos/evaluation/statistics.py",
            "src/routing_aware_atos/evaluation/transport_efficiency.py",
            "src/routing_aware_atos/models/routed_transport_operator.py",
            "src/routing_aware_atos/models/transport_operator.py",
            "src/routing_aware_atos/routed_dataset.py",
            "src/routing_aware_atos/routing_policies.py",
            "src/routing_aware_atos/sae/feature_metrics.py",
        )
    )


def live_causal_implementation_sha256() -> str:
    return _implementation_sha256(
        (
            "scripts/run_live_causal_restore.py",
            "src/routing_aware_atos/activation_loader.py",
            "src/routing_aware_atos/causal_eval/hooks.py",
            "src/routing_aware_atos/causal_eval/live_restore.py",
            "src/routing_aware_atos/models/transport_operator.py",
            "src/routing_aware_atos/routed_dataset.py",
            "src/routing_aware_atos/routing_policies.py",
        )
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_matches(
    payload: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, Any] | None = None,
) -> bool:
    provenance = payload.get("provenance")
    fingerprint = payload.get("run_fingerprint")
    if not isinstance(provenance, Mapping) or not isinstance(fingerprint, str):
        return False
    if expected_provenance is not None and dict(provenance) != dict(expected_provenance):
        return False
    return fingerprint == sha256_payload(provenance)


def _activation_config_sha256(activation_dir: str | Path) -> str | None:
    manifest_path = Path(activation_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    return manifest.get("config_sha256")


def sae_artifact_for_layer(cfg: Mapping[str, Any], target_layer: int) -> Path:
    artifacts = cfg["sae"]["artifacts"]
    path = artifacts.get(target_layer, artifacts.get(str(target_layer)))
    if path is None:
        raise KeyError(f"No SAE artifact configured for target layer {target_layer}")
    return Path(path)


def feature_artifact_for_layer(cfg: Mapping[str, Any], target_layer: int) -> Path:
    feature_dir = Path(
        cfg["feature_selection"].get("output_dir", "artifacts/features")
    )
    return feature_dir / f"layer_{target_layer}_features.json"


def predictive_run_provenance(
    cfg: Mapping[str, Any],
    pair_cfg: Mapping[str, Any],
    policy_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    experiment_cfg = cfg["experiments"]
    target_layer = int(pair_cfg["target_layer"])
    activation_dir = experiment_cfg["activation_dir_path"]
    return {
        "pair": dict(pair_cfg),
        "policy": dict(policy_cfg),
        "experiments": dict(experiment_cfg),
        "model_name": cfg.get("model_name"),
        "model_revision": cfg.get("model_revision"),
        "activation_config_sha256": _activation_config_sha256(activation_dir),
        "sae_artifact_sha256": sha256_file(
            sae_artifact_for_layer(cfg, target_layer)
        ),
        "feature_artifact_sha256": sha256_file(
            feature_artifact_for_layer(cfg, target_layer)
        ),
        "implementation_sha256": predictive_implementation_sha256(),
    }


def live_causal_run_provenance(
    cfg: Mapping[str, Any],
    *,
    operator_path: str | Path,
    source_layer: int,
    target_layer: int,
    policy: str,
) -> dict[str, Any]:
    live_cfg = dict(cfg.get("live_causal", cfg))
    activation_dir = str(
        live_cfg.get("activation_dir_path")
        or cfg.get("collection", {}).get("output_dir")
    )
    model_name = str(live_cfg.get("model_name", cfg.get("model_name")))
    return {
        "model_name": model_name,
        "model_revision": cfg.get("model_revision"),
        "activation_config_sha256": _activation_config_sha256(activation_dir),
        "operator_sha256": sha256_file(operator_path),
        "source_layer": int(source_layer),
        "target_layer": int(target_layer),
        "policy": str(policy),
        "live_causal": {
            key: value for key, value in live_cfg.items() if key != "token"
        },
        "implementation_sha256": live_causal_implementation_sha256(),
    }


__all__ = [
    "feature_artifact_for_layer",
    "collection_implementation_sha256",
    "feature_selection_implementation_sha256",
    "fingerprint_matches",
    "live_causal_run_provenance",
    "live_causal_implementation_sha256",
    "predictive_implementation_sha256",
    "predictive_run_provenance",
    "sae_artifact_for_layer",
    "sha256_file",
    "sha256_payload",
]
