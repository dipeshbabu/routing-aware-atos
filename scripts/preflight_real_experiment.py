from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.data.slimpajama import split_token_budgets
from routing_aware_atos.provenance import (
    collection_implementation_sha256,
    feature_selection_implementation_sha256,
    fingerprint_matches,
    predictive_run_provenance,
    sha256_file,
)
from routing_aware_atos.utils.io import load_json, load_npz, load_yaml


def _require_import(name: str, errors: list[str]) -> None:
    if importlib.util.find_spec(name) is None:
        errors.append(f"Missing Python package: {name}")


def _load_json_artifact(
    path: Path,
    errors: list[str],
    *,
    label: str,
) -> dict[str, Any] | None:
    try:
        payload = load_json(path)
    except Exception as exc:
        errors.append(f"Could not read {label} {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label.capitalize()} {path} must contain a JSON object")
        return None
    return payload


def _load_npz_artifact(
    path: Path,
    errors: list[str],
    *,
    label: str,
) -> dict[str, np.ndarray] | None:
    try:
        return load_npz(path)
    except Exception as exc:
        errors.append(f"Could not read {label} {path}: {exc}")
        return None


def _collection_config_sha256(cfg: dict[str, Any]) -> str:
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


def _check_config(cfg: dict[str, Any], errors: list[str]) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", str(cfg.get("model_revision", ""))):
        errors.append("model_revision must be pinned to an immutable Hugging Face commit")
    dataset_revision = cfg.get("dataset", {}).get("revision")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", str(dataset_revision or "")):
        errors.append("dataset.revision must be pinned to an immutable Hugging Face commit")
    collection = cfg["collection"]
    split_token_budgets(int(collection["operator_tokens"]), collection["split_fractions"])
    layers = {int(layer) for layer in collection["layer_indices"]}
    attention_layers = {int(layer) for layer in collection.get("attention_layers", [])}
    for pair in cfg["experiments"]["layer_pairs"]:
        source = int(pair["source_layer"])
        target = int(pair["target_layer"])
        if int(pair.get("leap", target - source)) != target - source:
            errors.append(f"Layer pair {source}->{target} records an inconsistent leap")
        if source not in layers or target not in layers:
            errors.append(f"Layer pair {source}->{target} is not fully present in collection.layer_indices")
        if target not in attention_layers:
            errors.append(f"Target layer {target} is missing from collection.attention_layers")
    sae_layers = {int(layer) for layer in cfg["sae"]["artifacts"]}
    target_layers = {int(pair["target_layer"]) for pair in cfg["experiments"]["layer_pairs"]}
    missing_sae = sorted(target_layers - sae_layers)
    if missing_sae:
        errors.append(f"Missing SAE artifact configuration for target layers {missing_sae}")
    feature_layers = {int(layer) for layer in cfg.get("feature_selection", {}).get("target_layers", [])}
    if missing_feature_layers := sorted(target_layers - feature_layers):
        errors.append(f"Missing feature-selection configuration for target layers {missing_feature_layers}")
    pair_keys = [
        (int(pair["source_layer"]), int(pair["target_layer"]))
        for pair in cfg["experiments"]["layer_pairs"]
    ]
    if len(pair_keys) != len(set(pair_keys)):
        errors.append("experiments.layer_pairs contains duplicates")
    if any(source >= target for source, target in pair_keys):
        errors.append("Every experiment source layer must precede its target layer")
    policy_names = [str(policy["name"]) for policy in cfg["experiments"]["policies"]]
    if len(policy_names) != len(set(policy_names)):
        errors.append("experiments.policies contains duplicate names")
    live_run_keys = [
        (int(run["source_layer"]), int(run["target_layer"]), str(run["policy"]))
        for run in cfg.get("live_causal", {}).get("runs", [])
    ]
    if len(live_run_keys) != len(set(live_run_keys)):
        errors.append("live_causal.runs contains duplicates")
    for source, target, policy in live_run_keys:
        if (source, target) not in set(pair_keys):
            errors.append(f"Live causal run {source}->{target} is not an experiment layer pair")
        if policy not in set(policy_names):
            errors.append(f"Live causal policy {policy!r} is not an experiment policy")
    feature_selection = cfg.get("feature_selection", {})
    feature_method = str(feature_selection.get("method", "fast_proxy"))
    if feature_method not in {"fast_proxy", "reference_full"}:
        errors.append(f"Unsupported feature_selection.method {feature_method!r}")
    if feature_method == "reference_full" and not feature_selection.get("causal_probe_prompts"):
        errors.append("reference_full feature selection requires causal_probe_prompts")
    if cfg.get("protocol") == "paper_scale_routing_v1":
        if cfg.get("model_name") != "google/gemma-2-2b":
            errors.append("paper_scale_routing_v1 requires google/gemma-2-2b")
        if cfg.get("model_revision") != "0738188b3055bc98daf0fe7211f0091357e5b979":
            errors.append("paper_scale_routing_v1 model revision does not match the registration")
        if cfg.get("dataset", {}).get("name") != "cerebras/SlimPajama-627B":
            errors.append("paper_scale_routing_v1 requires cerebras/SlimPajama-627B")
        dataset_cfg = cfg.get("dataset", {})
        expected_dataset_fields = {
            "config_name": "default",
            "revision": "2d0accd",
            "text_field": "text",
            "operator_source_split": "train",
            "causal_source_split": "test",
            "seed": 42,
            "causal_seed_offset": 1,
            "shuffle_buffer_size": 10_000,
            "causal_shuffle_buffer_size": 1,
            "skip_examples": 0,
            "causal_skip_examples": 10_000,
        }
        if any(dataset_cfg.get(key) != value for key, value in expected_dataset_fields.items()):
            errors.append("paper_scale_routing_v1 dataset stream settings do not match the registration")
        if cfg.get("attn_implementation") != "eager":
            errors.append("paper_scale_routing_v1 requires eager attention collection")
        if cfg.get("dtype") != "float32" or collection.get("cache_dtype") != "float32":
            errors.append("paper_scale_routing_v1 requires float32 model computation and cache storage")
        if int(collection.get("operator_tokens", 0)) != 250_000:
            errors.append("paper_scale_routing_v1 requires exactly 250,000 operator tokens")
        if int(collection.get("sequence_length", 0)) != 256:
            errors.append("paper_scale_routing_v1 requires sequence_length=256")
        if collection.get("packing_mode") != "documents":
            errors.append("paper_scale_routing_v1 requires document-preserving collection")
        if collection.get("split_fractions") != {
            "train": 0.6,
            "validation": 0.2,
            "test": 0.2,
        }:
            errors.append("paper_scale_routing_v1 requires exact 60/20/20 split fractions")
        if int(collection.get("causal_sequences", 0)) != 100:
            errors.append("paper_scale_routing_v1 requires 100 causal sequences")
        if layers != set(range(21)):
            errors.append("paper_scale_routing_v1 requires residual layers 0 through 20")
        if attention_layers != {10, 20}:
            errors.append("paper_scale_routing_v1 requires pooled attention at layers 10 and 20")
        if cfg["experiments"].get("ridge_selection") != "five_fold_cv":
            errors.append("paper_scale_routing_v1 requires five-fold ridge selection")
        if int(cfg["experiments"].get("cv_folds", 0)) != 5:
            errors.append("paper_scale_routing_v1 requires cv_folds=5")
        if [float(value) for value in cfg["experiments"].get("ridge_lambdas", [])] != [
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
            2000.0,
            5000.0,
            10000.0,
        ]:
            errors.append("paper_scale_routing_v1 ridge grid does not match the registered protocol")
        if not bool(cfg["experiments"].get("causal_only", False)):
            errors.append("paper_scale_routing_v1 requires causal-only routing sources")
        expected_pairs = {
            (target - leap, target)
            for target in (10, 20)
            for leap in range(1, 11)
        }
        if set(pair_keys) != expected_pairs:
            errors.append("paper_scale_routing_v1 requires k=1..10 for target layers 10 and 20")
        expected_policies = {
            "same_token",
            "previous_token",
            "attention_top1",
            "attention_topk",
            "random_topk",
            "shuffled_attention_topk",
            "attention_value_proxy_topk",
            "attention_topk_concat",
        }
        if set(policy_names) != expected_policies:
            errors.append("paper_scale_routing_v1 policy set does not match the registered protocol")
        policies_by_name = {
            str(policy["name"]): policy for policy in cfg["experiments"]["policies"]
        }
        expected_policy_shapes = {
            "same_token": ("same_token", 1, "weighted_sum"),
            "previous_token": ("previous_token", 1, "weighted_sum"),
            "attention_top1": ("attention_top1", 1, "weighted_sum"),
            "attention_topk": ("attention_topk", 3, "weighted_sum"),
            "random_topk": ("random_topk", 3, "weighted_sum"),
            "shuffled_attention_topk": (
                "shuffled_attention_topk",
                3,
                "weighted_sum",
            ),
            "attention_value_proxy_topk": ("attribution_topk", 3, "weighted_sum"),
            "attention_topk_concat": ("attention_topk", 3, "concat"),
        }
        for name, (routing_policy, top_k, input_mode) in expected_policy_shapes.items():
            policy = policies_by_name.get(name, {})
            if (
                str(policy.get("routing_policy", name)) != routing_policy
                or int(policy.get("top_k", 0)) != top_k
                or str(policy.get("input_mode")) != input_mode
            ):
                errors.append(f"paper_scale_routing_v1 policy {name!r} has incorrect parameters")
        concat_policy = policies_by_name.get("attention_topk_concat", {})
        if int(concat_policy.get("max_sources", 0)) != 3 or [
            float(value) for value in concat_policy.get("ridge_lambdas", [])
        ] != [100.0, 1000.0, 5000.0]:
            errors.append("paper_scale_routing_v1 concatenated Top-K policy is misconfigured")
        if feature_method != "reference_full":
            errors.append("paper_scale_routing_v1 requires reference_full feature selection")
        if int(feature_selection.get("max_tokens", 0)) != 120_000:
            errors.append("paper_scale_routing_v1 requires 120,000 feature-selection tokens")
        if float(feature_selection.get("top_percent", 0.0)) != 5.0:
            errors.append("paper_scale_routing_v1 requires the 5% SAE feature cutoff")
        expected_feature_fields = {
            "split_name": "train",
            "min_activation_count": 10,
            "max_firing_rate": 0.20,
            "candidate_percent": 15.0,
            "token_hash_bins": 2_048,
            "redundancy_mode": "activation_correlation",
            "redundancy_sample_size": 512,
            "selection_seed": 0,
        }
        if any(
            feature_selection.get(key) != value
            for key, value in expected_feature_fields.items()
        ):
            errors.append("paper_scale_routing_v1 feature-selection settings have drifted")
        experiment_cfg = cfg["experiments"]
        if (
            not bool(experiment_cfg.get("activated_only", False))
            or not bool(experiment_cfg.get("normalize_decoder", False))
            or int(experiment_cfg.get("min_feature_activations", 0)) != 10
            or float(experiment_cfg.get("min_feature_r2", float("nan"))) != -1.0
        ):
            errors.append("paper_scale_routing_v1 feature-evaluation settings have drifted")
        expected_live_runs = {
            (10 - leap, 10, policy)
            for leap in range(1, 11)
            for policy in ("same_token", "attention_topk")
        }
        if set(live_run_keys) != expected_live_runs:
            errors.append("paper_scale_routing_v1 causal runs do not match the registered protocol")
        live_cfg = cfg.get("live_causal", {})
        if (
            live_cfg.get("split_name") != "causal"
            or int(live_cfg.get("num_sequences", 0)) != 100
            or int(live_cfg.get("position_repeats", 0)) != 3
            or [str(value) for value in live_cfg.get("position_counts", [])]
            != ["1", "5", "all"]
            or not bool(live_cfg.get("causal_only", False))
            or live_cfg.get("dtype") != "float32"
            or int(live_cfg.get("position_seed", -1)) != 17
        ):
            errors.append("paper_scale_routing_v1 live-causal settings do not match the protocol")


def _check_cuda(cfg: dict[str, Any], errors: list[str], warnings: list[str], allow_cpu: bool) -> None:
    requires_cuda = any(
        str(section.get("device", "")).startswith("cuda")
        for section in (
            cfg,
            cfg.get("feature_selection", {}),
            cfg.get("experiments", {}),
            cfg.get("live_causal", {}),
        )
    )
    if requires_cuda and not torch.cuda.is_available():
        message = "Configuration requires CUDA, but torch.cuda.is_available() is false"
        (warnings if allow_cpu else errors).append(message)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        if memory_gib < 35:
            warnings.append(
                f"GPU {device_name} has {memory_gib:.1f} GiB; the paper profile is intended for an A100 40/80GB"
            )


def _check_hf_auth(errors: list[str]) -> None:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return
    if not (os.environ.get("HF_TOKEN") or get_token()):
        errors.append(
            "No Hugging Face token found. Gemma 2 is gated; accept its license and run `hf auth login`."
        )


def _check_disk(cfg: dict[str, Any], warnings: list[str]) -> None:
    collection = cfg["collection"]
    output = Path(collection["output_dir"])
    probe = output.parent if output.parent.exists() else PROJECT_ROOT
    free = shutil.disk_usage(probe).free
    total_tokens = int(collection["operator_tokens"]) + int(collection.get("causal_sequences", 0)) * int(
        collection["sequence_length"]
    )
    bytes_per_value = 4 if str(collection.get("cache_dtype", "float32")) == "float32" else 2
    residual_bytes = total_tokens * len(collection["layer_indices"]) * 2304 * bytes_per_value
    sequence_count = int(np.ceil(total_tokens / int(collection["sequence_length"])))
    attention_bytes = (
        sequence_count
        * len(collection.get("attention_layers", []))
        * int(collection["sequence_length"]) ** 2
        * bytes_per_value
    )
    estimate = residual_bytes + attention_bytes
    if free < estimate * 1.5:
        warnings.append(
            f"Only {free / 2**30:.1f} GiB free near {probe}; estimated uncompressed cache is "
            f"{estimate / 2**30:.1f} GiB"
        )


def _check_cache(cfg: dict[str, Any], errors: list[str]) -> None:
    output = Path(cfg["collection"]["output_dir"])
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        errors.append(f"Missing activation manifest: {manifest_path}")
        return
    manifest = _load_json_artifact(manifest_path, errors, label="activation manifest")
    if manifest is None:
        return
    if int(manifest.get("schema_version", -1)) != 3:
        errors.append(f"Unsupported activation schema in {manifest_path}")
    if manifest.get("config_sha256") != _collection_config_sha256(cfg):
        errors.append("Activation cache collection fingerprint does not match the current config")
    if manifest.get("implementation_sha256") != collection_implementation_sha256():
        errors.append("Activation cache was produced by a different collector implementation")
    if manifest.get("layer_semantics") != "post_layer" or manifest.get("hidden_state_offset") != 1:
        errors.append("Activation cache is not explicitly aligned to post-layer Gemma Scope sites")
    if manifest.get("model_name") != cfg.get("model_name"):
        errors.append(
            f"Cache model {manifest.get('model_name')!r} does not match config {cfg.get('model_name')!r}"
        )
    if manifest.get("model_revision") != cfg.get("model_revision"):
        errors.append("Cache model revision does not match the pinned config revision")
    manifest_dataset = manifest.get("dataset", {})
    dataset_defaults = {
        "seed": 42,
        "causal_seed_offset": 1,
        "shuffle_buffer_size": 10_000,
        "causal_shuffle_buffer_size": 1,
        "skip_examples": 0,
        "causal_skip_examples": 10_000,
    }
    for key in (
        "name",
        "revision",
        "operator_source_split",
        "causal_source_split",
        *dataset_defaults,
    ):
        expected = cfg["dataset"].get(key, dataset_defaults.get(key))
        actual = manifest_dataset.get(key)
        if actual != expected:
            errors.append(f"Cache dataset field {key!r} is {actual!r}; expected {expected!r}")
    if set(int(layer) for layer in manifest.get("layer_indices", [])) != set(
        int(layer) for layer in cfg["collection"]["layer_indices"]
    ):
        errors.append("Cache residual layer set does not match collection.layer_indices")
    if set(int(layer) for layer in manifest.get("attention_layers", [])) != set(
        int(layer) for layer in cfg["collection"].get("attention_layers", [])
    ):
        errors.append("Cache attention layer set does not match collection.attention_layers")
    if manifest.get("packing_mode") != cfg["collection"].get("packing_mode", "documents"):
        errors.append("Cache packing mode does not match the collection config")
    if int(manifest.get("sequence_length", -1)) != int(cfg["collection"]["sequence_length"]):
        errors.append("Cache sequence length does not match the collection config")
    if int(manifest.get("d_model", -1)) != 2_304:
        errors.append("Cache d_model is not 2304")
    if manifest.get("cache_dtype") != cfg["collection"].get("cache_dtype"):
        errors.append("Cache dtype does not match the collection config")
    expected_counts = split_token_budgets(
        int(cfg["collection"]["operator_tokens"]),
        cfg["collection"]["split_fractions"],
    )
    actual_counts = manifest.get("split_counts", {})
    for split_name, expected_tokens in expected_counts.items():
        actual_tokens = int(actual_counts.get(split_name, {}).get("tokens", -1))
        if actual_tokens != expected_tokens:
            errors.append(
                f"Cache split {split_name!r} has {actual_tokens} tokens; expected {expected_tokens}"
            )
    expected_causal = int(cfg["collection"].get("causal_sequences", 0)) * int(
        cfg["collection"]["sequence_length"]
    )
    actual_causal = int(actual_counts.get("causal", {}).get("tokens", -1))
    if actual_causal != expected_causal:
        errors.append(f"Causal split has {actual_causal} tokens; expected {expected_causal}")
    shard_records = manifest.get("shards")
    if not isinstance(shard_records, list) or not shard_records:
        errors.append("Activation manifest does not contain per-shard hashes")
        shard_records = []
    manifest_shard_names = {str(record.get("path")) for record in shard_records}
    disk_shard_names = {path.name for path in output.glob("part-*.zip")}
    if manifest_shard_names != disk_shard_names:
        errors.append("Activation manifest shard list does not match files on disk")
    for record in shard_records:
        shard_path = output / str(record.get("path"))
        if shard_path.is_file() and record.get("sha256") != sha256_file(shard_path):
            errors.append(f"Activation shard hash mismatch: {shard_path}")
    loader = None
    try:
        loader = ActivationLoader(activation_dir_path=output)
        expected_num_samples = int(manifest.get("num_samples", -1))
        if len(loader) != expected_num_samples:
            errors.append("Activation shard sample count does not match the manifest")
        shard_size = int(cfg["collection"].get("shard_size", 8))
        expected_shard_names = {
            f"part-{index:05d}.zip"
            for index in range(int(np.ceil(max(expected_num_samples, 0) / shard_size)))
        }
        actual_shard_names = {path.name for path in loader.file_paths}
        if actual_shard_names != expected_shard_names:
            errors.append("Activation shard filenames do not match the expected complete set")
        observed_counts: dict[str, dict[str, int]] = {}
        observed_sequence_indices: list[int] = []
        expected_layers = {int(layer) for layer in cfg["collection"]["layer_indices"]}
        expected_attention_layers = {
            int(layer) for layer in cfg["collection"].get("attention_layers", [])
        }
        sequence_length = int(cfg["collection"]["sequence_length"])
        for part_id, root in loader.root_objects.items():
            shard_path = loader.file_paths[part_id]
            shard_record = next(
                (record for record in shard_records if record.get("path") == shard_path.name),
                None,
            )
            if "complete" not in root or int(np.asarray(root["complete"][0])) != 1:
                errors.append(f"Activation shard {shard_path} is not marked complete")
            if int(root.attrs.get("schema_version", -1)) != 3:
                errors.append(f"Activation shard {shard_path} has the wrong schema")
            if root.attrs.get("collection_sha256") != manifest.get("config_sha256"):
                errors.append(f"Activation shard {shard_path} has the wrong collection fingerprint")
            shard_samples = int(root["input_ids"].shape[0]) if "input_ids" in root else 0
            if shard_record is not None and int(shard_record.get("num_samples", -1)) != shard_samples:
                errors.append(f"Activation manifest has the wrong sample count for {shard_path}")
            expected_shapes = {
                "input_ids": (shard_samples, sequence_length),
                "attention_mask": (shard_samples, sequence_length),
                "split_id": (shard_samples,),
                "sequence_index": (shard_samples,),
            }
            for name, expected_shape in expected_shapes.items():
                if name not in root or tuple(root[name].shape) != expected_shape:
                    errors.append(f"Activation shard {shard_path} has an invalid {name} array")
            if any(name not in root for name in expected_shapes):
                continue
            shard_layers = {int(layer) for layer in root.attrs.get("layer_indices", [])}
            shard_attention_layers = {
                int(layer) for layer in root.attrs.get("attention_layers", [])
            }
            if shard_layers != expected_layers or shard_attention_layers != expected_attention_layers:
                errors.append(f"Activation shard {shard_path} has incorrect layer metadata")
            if "activations" not in root:
                errors.append(f"Activation shard {shard_path} has no activation group")
            else:
                for layer in expected_layers:
                    name = f"layer_{layer}"
                    expected_shape = (shard_samples, sequence_length, 2_304)
                    if name not in root["activations"] or tuple(
                        root["activations"][name].shape
                    ) != expected_shape:
                        errors.append(f"Activation shard {shard_path} has invalid layer {layer}")
            if expected_attention_layers and "attention_scores" not in root:
                errors.append(f"Activation shard {shard_path} has no attention-score group")
            elif "attention_scores" in root:
                for layer in expected_attention_layers:
                    name = f"attention_layer_{layer}"
                    expected_shape = (shard_samples, sequence_length, sequence_length)
                    if name not in root["attention_scores"] or tuple(
                        root["attention_scores"][name].shape
                    ) != expected_shape:
                        errors.append(
                            f"Activation shard {shard_path} has invalid attention layer {layer}"
                        )

            masks = np.asarray(root["attention_mask"], dtype=np.uint8)
            split_ids = np.asarray(root["split_id"], dtype=np.int64)
            if not np.isin(masks, [0, 1]).all() or np.any(
                np.diff(masks.astype(np.int8), axis=1) > 0
            ):
                errors.append(f"Activation shard {shard_path} contains invalid attention masks")
            if not np.isin(split_ids, [0, 1, 2, 3]).all():
                errors.append(f"Activation shard {shard_path} contains invalid split IDs")
            split_names = {int(key): value for key, value in root.attrs.get("split_names", {}).items()}
            for row, split_id in enumerate(split_ids):
                split_name = split_names.get(int(split_id))
                if split_name is None:
                    continue
                counts = observed_counts.setdefault(split_name, {"sequences": 0, "tokens": 0})
                counts["sequences"] += 1
                counts["tokens"] += int(masks[row].sum())
            observed_sequence_indices.extend(
                np.asarray(root["sequence_index"], dtype=np.int64).tolist()
            )
        if observed_counts != actual_counts:
            errors.append("Activation shard split counts do not match the manifest")
        if sorted(observed_sequence_indices) != list(range(max(expected_num_samples, 0))):
            errors.append("Activation sequence indices are missing, duplicated, or out of range")
    except Exception as exc:
        errors.append(f"Could not validate activation shards: {exc}")
    finally:
        if loader is not None:
            loader.close()


def _check_sae(cfg: dict[str, Any], errors: list[str]) -> None:
    targets_by_layer = {
        int(target["layer"]): target
        for target in cfg.get("sae", {}).get("targets", [])
    }
    for layer, raw_path in cfg["sae"]["artifacts"].items():
        layer_index = int(layer)
        path = Path(raw_path)
        if not path.exists():
            errors.append(f"Missing layer-{layer} SAE artifact: {path}")
            continue
        arrays = _load_npz_artifact(path, errors, label="SAE artifact")
        if arrays is None:
            continue
        if "decoder" not in arrays or "encoder" not in arrays:
            errors.append(f"SAE artifact must contain decoder and encoder arrays: {path}")
            continue
        if arrays["decoder"].ndim != 2 or arrays["encoder"].ndim != 2:
            errors.append(f"SAE encoder and decoder must be matrices: {path}")
            continue
        if arrays["decoder"].shape[1] != 2304 or arrays["encoder"].shape[0] != 2304:
            errors.append(f"SAE artifact {path} is not compatible with Gemma 2 2B d_model=2304")
        if arrays["encoder"].shape != (
            arrays["decoder"].shape[1],
            arrays["decoder"].shape[0],
        ):
            errors.append(f"SAE encoder/decoder shapes are incompatible in {path}")
        normalization = (
            str(np.asarray(arrays["normalize_activations"]).item())
            if "normalize_activations" in arrays
            else "none"
        )
        if normalization not in {"none", "None"}:
            errors.append(f"SAE artifact {path} has unsupported activation normalization {normalization!r}")
        metadata_path = path.with_suffix(".json")
        if not metadata_path.exists():
            errors.append(f"Missing SAE metadata sidecar: {metadata_path}")
        else:
            metadata = _load_json_artifact(metadata_path, errors, label="SAE metadata")
            if metadata is None:
                continue
            actual_hash = sha256_file(path)
            if metadata.get("artifact_sha256") != actual_hash:
                errors.append(f"SAE artifact hash does not match its metadata sidecar: {path}")
            target = targets_by_layer.get(layer_index)
            if target is None:
                errors.append(f"Missing SAE target metadata configuration for layer {layer_index}")
            else:
                if Path(target["output"]) != path:
                    errors.append(f"SAE target output and artifact map disagree for layer {layer_index}")
                if metadata.get("release") != cfg["sae"].get("release"):
                    errors.append(f"SAE artifact {path} has the wrong release")
                if metadata.get("sae_id") != target.get("sae_id"):
                    errors.append(f"SAE artifact {path} has the wrong SAE id")
            if cfg.get("protocol") == "paper_scale_routing_v1":
                if arrays["decoder"].shape != (16_384, 2_304):
                    errors.append(f"Paper protocol requires a 16,384-feature SAE at {path}")
                if "threshold" not in arrays:
                    errors.append(f"Paper protocol requires JumpReLU thresholds in {path}")
                architecture = (
                    str(np.asarray(arrays["architecture"]).item())
                    if "architecture" in arrays
                    else None
                )
                activation_fn = (
                    str(np.asarray(arrays["activation_fn"]).item())
                    if "activation_fn" in arrays
                    else None
                )
                if architecture != "jumprelu" or activation_fn != "relu":
                    errors.append(f"Paper protocol requires a ReLU-based JumpReLU SAE at {path}")
                apply_b_dec = (
                    bool(np.asarray(arrays["apply_b_dec_to_input"]).item())
                    if "apply_b_dec_to_input" in arrays
                    else True
                )
                if apply_b_dec:
                    errors.append(f"Gemma Scope canonical SAE should not apply b_dec to inputs: {path}")
                expected_vectors = {
                    "b_dec": (2_304,),
                    "b_enc": (16_384,),
                    "threshold": (16_384,),
                }
                for name, expected_shape in expected_vectors.items():
                    if name not in arrays or arrays[name].shape != expected_shape:
                        errors.append(f"SAE artifact {path} has an invalid {name} shape")
        for name in ("decoder", "encoder", "b_dec", "b_enc", "threshold"):
            if name in arrays and not np.isfinite(arrays[name]).all():
                errors.append(f"SAE artifact {path} contains non-finite {name} values")


def _check_features(cfg: dict[str, Any], errors: list[str]) -> None:
    feature_cfg = cfg["feature_selection"]
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
    expected_method = {
        "reference_full": "reference_style_with_live_causal_ablation",
        "fast_proxy": "activity_token_entropy_decoder_uniqueness",
    }[str(feature_cfg.get("method", "fast_proxy"))]
    expected_features = int(16_384 * float(feature_cfg.get("top_percent", 5.0)) / 100.0)
    activation_manifest_path = Path(cfg["collection"]["output_dir"]) / "manifest.json"
    activation_manifest = (
        _load_json_artifact(
            activation_manifest_path,
            errors,
            label="activation manifest",
        )
        if activation_manifest_path.exists()
        else {}
    )
    activation_manifest = activation_manifest or {}
    for layer in feature_cfg["target_layers"]:
        path = output_dir / f"layer_{int(layer)}_features.json"
        stats_path = output_dir / f"layer_{int(layer)}_feature_stats.npz"
        if not path.exists():
            errors.append(f"Missing selected feature artifact: {path}")
            continue
        payload = _load_json_artifact(path, errors, label="feature artifact")
        if payload is None:
            continue
        feature_ids = payload.get("high_quality_feature_ids", [])
        if not feature_ids:
            errors.append(f"Selected feature list is empty: {path}")
            continue
        if payload.get("method") != expected_method:
            errors.append(
                f"Feature artifact {path} used {payload.get('method')!r}; expected {expected_method!r}"
            )
        if payload.get("model_revision") != cfg.get("model_revision"):
            errors.append(f"Feature artifact {path} has the wrong model revision")
        if payload.get("selection_config_sha256") != selection_config_sha256:
            errors.append(f"Feature artifact {path} used a different selection configuration")
        if payload.get("implementation_sha256") != feature_selection_implementation_sha256():
            errors.append(f"Feature artifact {path} used a different selector implementation")
        if payload.get("activation_config_sha256") != activation_manifest.get("config_sha256"):
            errors.append(f"Feature artifact {path} was built from a different activation cache")
        sae_path = Path(cfg["sae"]["artifacts"].get(int(layer), cfg["sae"]["artifacts"].get(str(layer))))
        if sae_path.exists() and payload.get("sae_artifact_sha256") != sha256_file(sae_path):
            errors.append(f"Feature artifact {path} was built from a different SAE artifact")
        if int(payload.get("tokens_processed", -1)) != int(feature_cfg.get("max_tokens", 120_000)):
            errors.append(f"Feature artifact {path} used the wrong token budget")
        if len(feature_ids) != expected_features or len(set(feature_ids)) != len(feature_ids):
            errors.append(
                f"Feature artifact {path} has {len(feature_ids)} IDs; expected {expected_features} unique IDs"
            )
        if any(int(feature_id) < 0 or int(feature_id) >= 16_384 for feature_id in feature_ids):
            errors.append(f"Feature artifact {path} contains an out-of-range SAE feature ID")
        if int(payload.get("layer", -1)) != int(layer):
            errors.append(f"Feature artifact {path} records the wrong target layer")
        if payload.get("split") != feature_cfg.get("split_name", "train"):
            errors.append(f"Feature artifact {path} records the wrong dataset split")
        if int(payload.get("num_sae_features", -1)) != 16_384:
            errors.append(f"Feature artifact {path} records the wrong SAE width")
        if int(payload.get("selected_count", -1)) != len(feature_ids):
            errors.append(f"Feature artifact {path} has inconsistent selected-feature counts")
        if len(payload.get("features", [])) != len(feature_ids):
            errors.append(f"Feature artifact {path} has incomplete per-feature records")
        record_ids = [
            int(record.get("feature_id", -1))
            for record in payload.get("features", [])
            if isinstance(record, dict)
        ]
        if record_ids != [int(feature_id) for feature_id in feature_ids]:
            errors.append(f"Feature artifact {path} records do not match its selected IDs")
        if not stats_path.exists():
            errors.append(f"Missing selected-feature statistics: {stats_path}")
        else:
            stats = _load_npz_artifact(stats_path, errors, label="feature statistics")
            if stats is None:
                continue
            if "selected_ids" not in stats or stats["selected_ids"].astype(int).tolist() != [
                int(feature_id) for feature_id in feature_ids
            ]:
                errors.append(f"Feature JSON and statistics disagree: {path}")
            if "activation_count" not in stats or stats["activation_count"].shape != (16_384,):
                errors.append(f"Feature statistics {stats_path} has invalid activation counts")
            if "eligible_ids" not in stats or "eligible_scores" not in stats or (
                stats.get("eligible_ids", np.empty(0)).shape
                != stats.get("eligible_scores", np.empty(1)).shape
            ):
                errors.append(f"Feature statistics {stats_path} has inconsistent eligible arrays")


def _check_operators(cfg: dict[str, Any], errors: list[str]) -> None:
    output_dir = Path(cfg["experiments"].get("output_dir", "outputs/real"))
    pair_configs = {
        (int(pair["source_layer"]), int(pair["target_layer"])): pair
        for pair in cfg["experiments"]["layer_pairs"]
    }
    policy_configs = {
        str(policy["name"]): policy for policy in cfg["experiments"]["policies"]
    }
    for run in cfg["live_causal"].get("runs", []):
        source = int(run["source_layer"])
        target = int(run["target_layer"])
        policy = str(run["policy"])
        path = output_dir / "runs" / f"L{source}_to_L{target}" / policy / "operator.npz"
        if not path.exists():
            errors.append(f"Missing operator required for live causal run: {path}")
            continue
        metadata_path = path.with_name("metadata.json")
        if not metadata_path.exists():
            errors.append(f"Missing operator metadata: {metadata_path}")
            continue
        metadata = _load_json_artifact(metadata_path, errors, label="operator metadata")
        arrays = _load_npz_artifact(path, errors, label="transport operator")
        if metadata is None or arrays is None:
            continue
        pair_cfg = pair_configs.get((source, target))
        policy_cfg = policy_configs.get(policy)
        if pair_cfg is None or policy_cfg is None:
            errors.append(f"Operator {path} does not map to a configured experiment")
            continue
        try:
            expected_provenance = predictive_run_provenance(cfg, pair_cfg, policy_cfg)
        except Exception as exc:
            errors.append(f"Could not construct expected provenance for {path}: {exc}")
            continue
        if not fingerprint_matches(metadata, expected_provenance=expected_provenance):
            errors.append(f"Operator metadata fingerprint is stale or invalid: {metadata_path}")
        actual_hash = sha256_file(path)
        if metadata.get("operator_sha256") != actual_hash:
            errors.append(f"Operator hash does not match its metadata: {path}")
        if (
            int(metadata.get("source_layer", -1)) != source
            or int(metadata.get("target_layer", -1)) != target
            or metadata.get("policy_label") != policy
        ):
            errors.append(f"Operator metadata has the wrong run identity: {metadata_path}")

        input_mode = str(policy_cfg.get("input_mode", "weighted_sum"))
        input_width = 2_304
        if input_mode == "concat":
            input_width *= int(policy_cfg.get("max_sources", policy_cfg.get("top_k", 1)))
        expected_shapes = {
            "weight": (input_width, 2_304),
            "bias": (2_304,),
            "x_mean": (input_width,),
            "y_mean": (2_304,),
        }
        for name, expected_shape in expected_shapes.items():
            if name not in arrays or arrays[name].shape != expected_shape:
                errors.append(
                    f"Operator {path} has {name} shape "
                    f"{None if name not in arrays else arrays[name].shape}; expected {expected_shape}"
                )
            elif not np.isfinite(arrays[name]).all():
                errors.append(f"Operator {path} contains non-finite {name} values")


def run_preflight(cfg: dict[str, Any], *, stage: str, allow_cpu: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    _check_config(cfg, errors)
    _require_import("transformers", errors)
    _require_import("datasets", errors)
    _require_import("zarr", errors)
    _check_cuda(cfg, errors, warnings, allow_cpu)
    _check_disk(cfg, warnings)
    if stage in {"collect", "features", "causal", "all"}:
        _check_hf_auth(errors)
    if stage in {"cache", "sae", "features", "experiments", "causal", "report", "all"}:
        _check_cache(cfg, errors)
    if stage in {"sae", "features", "experiments", "causal", "report", "all"}:
        _require_import("sae_lens", errors)
    if stage in {"features", "experiments", "causal", "report", "all"}:
        _check_sae(cfg, errors)
    if stage in {"experiments", "causal", "report", "all"}:
        _check_features(cfg, errors)
    if stage in {"causal", "report", "all"}:
        _check_operators(cfg, errors)
    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate every prerequisite for the real A100 experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["collect", "cache", "sae", "features", "experiments", "causal", "report", "all"],
        default="all",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    errors, warnings = run_preflight(cfg, stage=args.stage, allow_cpu=args.allow_cpu)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"Preflight failed with {len(errors)} error(s)")
    print(f"Preflight passed for stage={args.stage}")


if __name__ == "__main__":
    main()
